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

//! Integration tests for `LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS`.
//!
//! The unit tests in `src/config.rs` pin the parse-time contract (the
//! off/off auth shape is rejected unless the override is set, and the
//! override is inert otherwise). These tests cover the two things the unit
//! tier cannot:
//!
//! 1. **End to end through the real builders.** A `Config` produced by
//!    `Config::from_env` under the override must drive
//!    `build_permissions_client` / `build_userinfo_client` to the no-op
//!    permissions client and the in-process JWT client without either
//!    builder demanding an endpoint. This is the exact runtime shape the
//!    conformance suite relies on, so it is asserted on behaviour: every
//!    permission check is granted with no network call, and an unsigned
//!    JWT-shaped bearer yields a display identity.
//!
//! 2. **The startup log line.** The line is emitted from a separate `Config`
//!    method that `main.rs` calls right after `Config::from_env` returns:
//!    ERROR-level when the override is in effect, WARN when it is set but
//!    inert. Capturing the output here proves the method fires (and at which
//!    level, with which wording) in each of the three shapes: in effect,
//!    set-but-inert, unset. A source scan pins that `main.rs` actually makes
//!    the call, after configuration loads and before the first INFO line;
//!    only WARNs `Config::from_env` itself logs can precede it.
//!
//! Env-var mutation is scoped with `temp_env::with_vars` so parallel sibling
//! tests in this binary do not observe each other's environment.

use std::sync::{Arc, Mutex};

use tracing_subscriber::fmt::MakeWriter;

use live_edit::auth::{
    build_permissions_client, build_userinfo_client, fast_retry, AuthResult, Permission,
};
use live_edit::config::{Config, PodMemorySource};

use crate::common::{make_jwt, test_config};

/// The off/off auth shape plus the override, with both endpoint vars pinned
/// unset so the test also proves neither builder demands one.
const OVERRIDE_ENV: [(&str, Option<&str>); 5] = [
    ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("false")),
    ("LIVE_EDIT_AUTHZ_ENABLED", Some("false")),
    ("LIVE_EDIT_STORAGE_SERVICE_ENDPOINT", None),
    ("LIVE_EDIT_USERINFO_ENDPOINT", None),
    (
        "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
        Some("true"),
    ),
];

/// `MakeWriter` that captures every byte written to it into a shared buffer.
/// Same six-line helper as `service_token_startup_warns.rs`, duplicated per
/// the project convention of defining it per test module.
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

impl std::io::Write for CapturingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Run `f` under a scoped WARN-level subscriber and return what it logged.
fn capture_log_output<F: FnOnce()>(f: F) -> String {
    let (writer, captured) = CapturingWriter::new();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(writer)
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .finish();
    tracing::subscriber::with_default(subscriber, f);
    let guard = captured.lock().unwrap();
    String::from_utf8_lossy(&guard).into_owned()
}

// ── end to end: from_env → builders → behaviour ──────────────────────────────

#[tokio::test]
async fn override_config_builds_no_op_permissions_and_jwt_userinfo_clients() {
    // `from_env` is synchronous and reads the process environment, so the
    // env scope only needs to wrap the parse; the builders read the parsed
    // `Config` and nothing else.
    let config = temp_env::with_vars(OVERRIDE_ENV, || {
        Config::from_env_with(PodMemorySource::Reference)
            .expect("off/off auth + override must parse")
    });
    assert!(
        config.unvalidated_tokens_in_effect(),
        "precondition: the override must be in effect for this shape"
    );

    // Permissions: the no-op client grants both permissions. No endpoint is
    // configured, so a builder that ignored `authz_enabled = false` would
    // panic on the `expect` inside `build_permissions_client` — the
    // successful return is itself part of the assertion.
    let permissions = build_permissions_client(&config, &fast_retry(1))
        .await
        .expect("authz off must yield the no-op permissions client without an endpoint");
    for permission in [Permission::Read, Permission::Write] {
        let verdict = permissions
            .check_permission("any-bearer", "/any/path", permission)
            .await
            .expect("no-op client never errors");
        assert_eq!(
            verdict,
            AuthResult::Allowed,
            "no-op client must grant {permission:?}"
        );
    }

    // Userinfo: the in-process JWT client decodes an unsigned, unverified
    // payload. This is precisely what "no backend validates the token"
    // means: the header is a fixed `{"alg":"none"}` and the signature is
    // empty, yet the caller is accepted with the identity it claims.
    let userinfo = build_userinfo_client(
        &config,
        &fast_retry(1),
        Arc::new(crate::common::metrics_scrape_only("test")),
    )
    .await
    .expect("userinfo off must yield the in-process JWT client without an endpoint");
    let forged = make_jwt(&serde_json::json!({
        "sub": "conformance-user",
        "name": "Conformance Runner",
    }));
    let identity = userinfo
        .get_current_user(&forged)
        .await
        .expect("unsigned JWT-shaped bearer must decode under the override");
    assert_eq!(identity.display_name, "Conformance Runner");
    assert_eq!(identity.id, "conformance-user");
}

// ── startup warning ──────────────────────────────────────────────────────────

#[test]
fn startup_warning_fires_when_override_in_effect() {
    let config = temp_env::with_vars(OVERRIDE_ENV, || {
        Config::from_env_with(PodMemorySource::Reference)
            .expect("off/off auth + override must parse")
    });

    let output = capture_log_output(|| config.log_test_only_auth_override_startup_warning());

    // ERROR, not WARN: this is the only runtime signal that no token is
    // validated, and `LIVE_EDIT_LOG_LEVEL=error` must not be able to hide it.
    assert!(
        output.contains("ERROR"),
        "expected an ERROR-level event, captured: {output:?}"
    );
    assert!(
        !output.contains("WARN"),
        "in-effect line must be ERROR-level, not WARN, captured: {output:?}"
    );
    assert!(
        output.contains("LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS=true"),
        "warning must name the variable and its value, captured: {output:?}"
    );
    assert!(
        output.contains("NO backend validates bearer tokens"),
        "warning must state the consequence plainly, captured: {output:?}"
    );
    assert!(
        !output.contains("has no effect"),
        "in-effect wording must not be confused with the inert wording, captured: {output:?}"
    );
}

#[test]
fn startup_warning_fires_with_inert_wording_when_override_set_but_backend_enabled() {
    // Default auth shape (authz on) with the override present: the flag
    // changes nothing, but a stray copy of a test-only variable in a real
    // deployment's environment must still be visible at startup.
    let mut config = test_config();
    config.authz_enabled = true;
    config.userinfo_service_enabled = false;
    config.test_only_dangerously_allow_unvalidated_tokens = true;
    assert!(
        !config.unvalidated_tokens_in_effect(),
        "precondition: inert"
    );

    let output = capture_log_output(|| config.log_test_only_auth_override_startup_warning());

    assert!(
        output.contains("WARN"),
        "expected a WARN-level event, captured: {output:?}"
    );
    assert!(
        output.contains("has no effect"),
        "inert shape must use the inert wording, captured: {output:?}"
    );
    assert!(
        output.contains("LIVE_EDIT_AUTHZ_ENABLED=true"),
        "inert warning must echo which backend is still validating tokens, captured: {output:?}"
    );
    assert!(
        !output.contains("NO backend validates bearer tokens"),
        "inert shape must not claim token validation is off, captured: {output:?}"
    );
}

#[test]
fn startup_warning_silent_when_override_unset() {
    // The common test config has authz off and userinfo on with the
    // override unset — a supported production shape. Nothing may be logged.
    let config = test_config();
    assert!(!config.test_only_dangerously_allow_unvalidated_tokens);

    // Positive control: a canary emitted inside the same closure proves the
    // capture harness is live, so "nothing captured" cannot be mistaken for
    // "nothing fired" if the subscriber were ever not installed.
    let output = capture_log_output(|| {
        tracing::warn!("capture-canary");
        config.log_test_only_auth_override_startup_warning();
    });

    assert!(
        output.contains("capture-canary"),
        "capture harness is dead, captured: {output:?}"
    );
    assert_eq!(
        output.matches("WARN").count(),
        1,
        "only the canary may be logged when the override is unset, captured: {output:?}"
    );
    assert!(
        !output.contains("ERROR"),
        "no ERROR line may fire when the override is unset, captured: {output:?}"
    );
}

// ── wiring: main.rs actually calls the method, right after config loads ─────

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
#[test]
fn main_calls_startup_warning_after_config_load_and_before_first_info_line() {
    // The three tests above drive the method directly, so deleting the call
    // in `main.rs` would break none of them. Pin the production call site
    // with a source scan (repo idiom: `tests/integration/storage/
    // mapper_routing_test.rs`) and its ordering: it must come after
    // `init_tracing` (otherwise there is no subscriber and the line is
    // dropped), after `Config::from_env` (it reads the loaded `Config`), and
    // before the first ordinary startup line. Only WARNs `Config::from_env`
    // logs while it parses can precede it; `startup_log_binary.rs` proves
    // those reach the log.
    const MAIN: &str = include_str!("../../src/main.rs");

    // Line-wise scan that skips `//` comment lines, so a doc comment naming
    // the call cannot satisfy the probe with the real call deleted. Each
    // probe anchors on the bare call name so a signature change to
    // `init_tracing` does not turn into a misleading "subscriber not
    // installed" failure.
    let first_code_line = |needle: &str| -> Option<usize> {
        MAIN.lines()
            .enumerate()
            .filter(|(_, line)| !line.trim_start().starts_with("//"))
            .find(|(_, line)| line.contains(needle))
            .map(|(idx, _)| idx)
    };

    let init =
        first_code_line("init_tracing(").expect("main.rs must install the tracing subscriber");
    let config = first_code_line("Config::from_env()").expect("main.rs must load the Config");
    let warn = first_code_line("config.log_test_only_auth_override_startup_warning()")
        .expect("main.rs must call Config::log_test_only_auth_override_startup_warning");
    let starting = first_code_line(r#"tracing::info!("live-edit starting")"#)
        .expect("main.rs must log the startup line");

    assert!(
        init < warn,
        "the override line must be emitted after the subscriber is installed"
    );
    assert!(
        config < warn,
        "the override line must be emitted once the Config has loaded"
    );
    assert!(
        warn < starting,
        "the override line must be logged before the first ordinary startup line"
    );
}
