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

//! Integration tests for `live_edit::auth::service_token::log_service_token_startup_warnings`
//! (the SR-6.4 client-credentials startup WARN).
//!
//! These tests exercise the helper at the integration tier, capturing the
//! emitted `tracing::warn!` output via a `CapturingWriter` (mirroring the
//! pattern in `src/grpc/auth_interceptor.rs::tests::CapturingWriter`). They
//! call the helper directly with a `Config` built by the shared
//! `common::test_config()` and mutate its fields, so every adapter and
//! token-file combination is reachable — including `StorageApi` without a
//! token file, which `Config::from_env` refuses and so never reaches the
//! helper in the binary.
//!
//! The helper emits exactly one WARN, and only when the client-credentials
//! flag is set. `tests/misc/startup_log_binary.rs` covers the production call
//! site: the real binary logs this WARN before its first dependency step.
//!
//! Each test scopes env-var mutations via `temp_env::with_vars` to prevent
//! cross-test contamination from the workspace's parallel test execution.
//! The helper itself does not read env vars (it operates on the already-
//! parsed `Config`), so the env-var scoping is belt-and-suspenders against a
//! future change that adds env-var reads to the helper directly.
//!
//! The token-bytes-never-logged test pins SR-6.2 acceptance criterion #9 at
//! the integration tier — a defense-in-depth check that catches a future
//! change which decides to log the configured `LIVE_EDIT_SERVICE_TOKEN_FILE`
//! path or its contents.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tracing_subscriber::fmt::MakeWriter;

use live_edit::auth::service_token::log_service_token_startup_warnings;
use live_edit::checkpoint::CheckpointAdapter;

use crate::common::test_config;

/// `MakeWriter` that captures every byte written to it into a shared buffer.
///
/// Mirrors `src/grpc/auth_interceptor.rs::tests::CapturingWriter`. We
/// duplicate the small helper here rather than pull it into `tests/common`
/// because the existing project pattern (see `auth_interceptor.rs`) defines
/// the writer per-test-module and the duplication is six lines.
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

/// Run `f` under a freshly-installed `tracing` subscriber that captures every
/// event to the returned buffer. The subscriber stays scoped to the closure;
/// once we return, the previous default subscriber (if any) is restored, so
/// concurrent sibling tests in the same binary do not see our captured log
/// lines in their output and vice versa.
///
/// `with_max_level(WARN)` filters at the WARN level so an unrelated `info!` /
/// `debug!` event firing inside the helper body cannot mask the WARN-not-
/// firing assertions. The helper itself only emits WARN, but pinning the
/// filter level here makes the assertions robust against a future helper
/// change that adds `info!` ambient logging.
fn capture_log_output<F: FnOnce()>(f: F) -> String {
    let (writer, captured) = CapturingWriter::new();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(writer)
        .with_max_level(tracing::Level::WARN)
        // ANSI escape codes would otherwise pollute the captured buffer and
        // make substring assertions brittle on environments that auto-enable
        // colorization (e.g. running locally vs. CI).
        .with_ansi(false)
        .finish();
    tracing::subscriber::with_default(subscriber, f);
    let guard = captured.lock().unwrap();
    String::from_utf8_lossy(&guard).into_owned()
}

/// Sanity helper: count `WARN`-level events in the captured output. The
/// tracing-subscriber fmt layer emits one line per event with the level token
/// `WARN` in it, so a simple substring count is reliable enough for the
/// not-fired assertions below.
fn warn_event_count(output: &str) -> usize {
    output.matches("WARN").count()
}

#[test]
fn service_token_client_credentials_enabled_logs_sr_6_4_warn() {
    // Belt-and-suspenders env-var scope (the helper reads only from
    // `Config`, but pinning the env at the test boundary matches the
    // pattern used by sibling tests in this binary and protects against a
    // future helper change that adds env-var reads).
    temp_env::with_vars(
        vec![(
            "LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED",
            Some("true"),
        )],
        || {
            let mut cfg = test_config();
            cfg.service_token_client_credentials_enabled = true;

            let output = capture_log_output(|| {
                log_service_token_startup_warnings(&cfg);
            });

            assert!(
                output.contains("WARN"),
                "expected a WARN-level event for SR-6.4 client-credentials startup; \
                 captured output was: {output:?}"
            );
            // docs/service/docs/troubleshooting.md § Startup and readiness
            // names this variable so operators can match the warning in their
            // logs; a reword must keep the page's description true.
            assert!(
                output.contains(
                    "LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED=true selects the \
                     in-process OAuth client-credentials flow, which is NOT implemented in V1"
                ),
                "WARN must carry the prefix the troubleshooting page matches, naming the \
                 variable and the unsupported mode; captured output: {output:?}"
            );
            // The flag changes nothing else: outbound calls authenticate with
            // the token file whatever it is set to. The WARN must say so, and
            // must not claim the flag makes calls unauthenticated.
            assert!(
                output.contains("no other effect"),
                "WARN must state the setting has no other effect; captured output: {output:?}"
            );
            assert!(
                !output.contains("unauthenticated"),
                "WARN must not claim outbound calls become unauthenticated — the flag \
                 has no such effect; captured output: {output:?}"
            );
        },
    );
}

#[test]
fn service_token_client_credentials_disabled_logs_nothing_for_any_adapter_or_token_file() {
    // The helper's only WARN is the client-credentials one. With the flag
    // off it must stay silent whatever the adapter and token-file state —
    // including `StorageApi` without a token file, which `Config::from_env`
    // refuses at configuration load, so the helper has nothing to add.
    let token_file = PathBuf::from("/var/run/oauth/access_token");
    let shapes = [
        (CheckpointAdapter::Disabled, None),
        (CheckpointAdapter::Disabled, Some(token_file.clone())),
        (CheckpointAdapter::StorageApi, None),
        (CheckpointAdapter::StorageApi, Some(token_file)),
    ];
    temp_env::with_vars(
        vec![(
            "LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED",
            None::<&str>,
        )],
        || {
            for (adapter, service_token_file) in shapes {
                let mut cfg = test_config();
                cfg.service_token_client_credentials_enabled = false;
                cfg.checkpoint_adapter = adapter;
                cfg.service_token_file = service_token_file.clone();

                // Positive control: a canary logged in the same closure
                // proves the capture is live, so "nothing captured" cannot
                // be mistaken for "nothing fired".
                let output = capture_log_output(|| {
                    tracing::warn!("capture-canary");
                    log_service_token_startup_warnings(&cfg);
                });

                assert!(
                    output.contains("capture-canary"),
                    "capture harness is dead; captured output: {output:?}"
                );
                assert_eq!(
                    warn_event_count(&output),
                    1,
                    "only the canary may be logged with the client-credentials flag \
                     unset (adapter {adapter:?}, token file {service_token_file:?}); \
                     captured output: {output:?}"
                );
            }
        },
    );
}

#[test]
fn startup_warns_never_emit_bearer_token_bytes() {
    // SR-6.2 #9 defense-in-depth at the integration tier. We feed the helper
    // a path that contains a distinctive sentinel substring and assert the
    // captured log output never echoes that substring — catching a future
    // change that decides to log the token-file path (or, worse, its
    // contents) from the helper's WARN.
    const SENTINEL: &str = "super-secret-token-bytes-must-not-appear";
    let sentinel_path = format!("/var/run/oauth/{SENTINEL}/access_token");

    temp_env::with_vars(
        vec![
            (
                "LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED",
                Some("true"),
            ),
            ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("storageapi")),
            ("LIVE_EDIT_SERVICE_TOKEN_FILE", Some(sentinel_path.as_str())),
        ],
        || {
            let mut cfg = test_config();
            // The client-credentials flag makes the helper's WARN fire; the
            // token file is set to the sentinel path, so a formatter that
            // ever reached for the path would echo the sentinel.
            cfg.service_token_client_credentials_enabled = true;
            cfg.checkpoint_adapter = CheckpointAdapter::StorageApi;
            cfg.service_token_file = Some(PathBuf::from(&sentinel_path));

            let output = capture_log_output(|| {
                log_service_token_startup_warnings(&cfg);
            });

            // Non-vacuity: the WARN must have fired, or the sentinel check
            // below proves nothing.
            assert!(
                output.contains("WARN")
                    && output.contains("LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED"),
                "the client-credentials WARN must fire — without it the sentinel check \
                 is vacuous; captured output: {output:?}"
            );
            assert!(
                !output.contains(SENTINEL),
                "captured log output must not echo the service-token-file path \
                 sentinel substring (SR-6.2 #9, bearer-bytes-never-logged); \
                 captured output: {output:?}"
            );
        },
    );
}
