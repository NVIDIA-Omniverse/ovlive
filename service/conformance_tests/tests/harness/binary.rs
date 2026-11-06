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

//! The `live-edit-conformance` executable end to end, run with a cleared
//! environment: its command line, what a failed run leaves behind (exit
//! code, report files and `CONFORMANCE_REPORT` lines; no HTTP, transport or
//! credential text, and for a configuration refused on the provisioned path
//! its rule but never the value), and a real SIGTERM or SIGINT, sent with
//! the system's `kill` while the run waits for readiness or on a case.

use std::path::Path;
use std::process::{Output, Stdio};
use std::time::Duration;

use base64::Engine as _;
use live_edit_conformance::deployment::INTERRUPT_DELETE_BOUND;
use live_edit_conformance::runner::{RESULTS_FILE, SUMMARY_FILE};
use tokio::process::{Child, Command};
use tokio::time::Instant;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::fake_live::{Answer, FakeLive};
use crate::support::{junit, summary, SilentServer};

/// How long one run of the executable may take: every run here ends at its
/// first failure or interruption, without waiting out a service.
const RUN_BUDGET: Duration = Duration::from_secs(60);

/// The executable with no environment but `vars`, and `args`, not yet run.
fn command(args: &[&str], vars: &[(&str, &str)]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_live-edit-conformance"));
    command
        .args(args)
        .env_clear()
        .envs(vars.iter().copied())
        .kill_on_drop(true);
    command
}

/// The executable with no environment but `vars`, run with `args`.
async fn run(args: &[&str], vars: &[(&str, &str)]) -> Output {
    tokio::time::timeout(RUN_BUDGET, command(args, vars).output())
        .await
        .expect("the executable exits")
        .expect("the executable starts")
}

/// The executable with no environment but `vars`, started without waiting
/// for it, its output captured.
fn spawn(vars: &[(&str, &str)]) -> Child {
    command(&[], vars)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the executable starts")
}

/// Sends `signal` (`TERM` or `INT`) to `child` with the system's `kill`.
fn send(signal: &str, child: &Child) {
    let pid = child.id().expect("the executable is still running");
    let status = std::process::Command::new("kill")
        .args([format!("-{signal}"), pid.to_string()])
        .status()
        .expect("kill runs");
    assert!(status.success(), "kill -{signal} {pid}: {status}");
}

/// Waits for `child` to exit and returns its output.
async fn finish(child: Child) -> Output {
    tokio::time::timeout(RUN_BUDGET, child.wait_with_output())
        .await
        .expect("the executable exits")
        .expect("the executable is waited for")
}

/// The `CONFORMANCE_REPORT` lines of `stdout`, which may hold progress lines
/// too.
fn report_lines(stdout: &str) -> String {
    stdout
        .lines()
        .filter(|line| line.starts_with("CONFORMANCE_REPORT "))
        .map(|line| format!("{line}\n"))
        .collect()
}

fn text(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).expect("UTF-8 output")
}

/// Checks stdout is exactly one `CONFORMANCE_REPORT <name> <base64>` line
/// per file in `names`, in that order, each decoding to that report file's
/// bytes in `dir`.
fn assert_report_lines(stdout: &str, dir: &Path, names: &[&str]) {
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), names.len(), "{stdout}");
    for (line, name) in lines.iter().zip(names) {
        let fields: Vec<&str> = line.split(' ').collect();
        assert_eq!(fields.len(), 3, "{line}");
        assert_eq!(fields[..2], ["CONFORMANCE_REPORT", *name], "{line}");
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(fields[2])
            .expect("standard base64");
        assert_eq!(decoded, std::fs::read(dir.join(name)).unwrap(), "{name}");
    }
}

#[tokio::test]
async fn help_prints_the_usage_and_exits_0() {
    let output = run(&["--help"], &[]).await;

    assert_eq!(output.status.code(), Some(0));
    assert!(
        text(&output.stdout).starts_with("Usage: live-edit-conformance [--profile <name>]..."),
        "{}",
        text(&output.stdout)
    );
    assert!(output.stderr.is_empty());
}

#[tokio::test]
async fn retired_conformance_subcommand_word_exits_2_without_running_anything() {
    let dir = tempfile::tempdir().unwrap();
    let output = run(
        &["conformance", "--profile", "deployment"],
        &[("CONFORMANCE_REPORT_DIR", dir.path().to_str().unwrap())],
    )
    .await;

    assert_eq!(output.status.code(), Some(2));
    let stderr = text(&output.stderr);
    assert!(
        stderr.starts_with("conformance: unexpected argument \"conformance\""),
        "{stderr}"
    );
    assert!(stderr.contains("Usage: live-edit-conformance"), "{stderr}");
    assert!(output.stdout.is_empty());
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        0,
        "no reports"
    );
}

#[tokio::test]
async fn refused_configuration_is_a_preflight_junit_error_and_exit_2() {
    let dir = tempfile::tempdir().unwrap();
    let output = run(
        &[],
        &[
            ("CONFORMANCE_AUTH_TOKEN", "local-token"),
            ("CONFORMANCE_REPORT_DIR", dir.path().to_str().unwrap()),
            ("CONFORMANCE_REPORT_STDOUT", "true"),
        ],
    )
    .await;

    assert_eq!(output.status.code(), Some(2));
    let xml = junit(dir.path());
    assert!(
        xml.contains(
            "<testsuite name=\"conformance-preflight\" tests=\"1\" failures=\"0\" errors=\"1\""
        ),
        "{xml}"
    );
    assert!(xml.contains("CONFORMANCE_BASE_PATH"), "{xml}");
    // A run that never reached its cases has no summary to print.
    assert_report_lines(text(&output.stdout), dir.path(), &[RESULTS_FILE]);
}

#[tokio::test]
async fn provisioning_failure_exits_2_with_reports_and_never_prints_error_text() {
    const SECRET_TOKEN: &str = "secret-token-must-not-be-logged";
    const SECRET_BODY: &str = "secret-body-must-not-be-logged";
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(500).set_body_string(SECRET_BODY))
        .mount(&server)
        .await;
    Mock::given(method("DELETE"))
        .respond_with(ResponseTemplate::new(204))
        .mount(&server)
        .await;
    let dir = tempfile::tempdir().unwrap();
    let storage = server.uri();

    let output = run(
        &["--profile", "deployment"],
        &[
            ("CONFORMANCE_AUTH_TOKEN", SECRET_TOKEN),
            ("CONFORMANCE_BASE_PATH", "/container/conformance"),
            ("CONFORMANCE_STORAGE_ENDPOINT", &storage),
            ("CONFORMANCE_REPORT_DIR", dir.path().to_str().unwrap()),
            ("CONFORMANCE_REPORT_STDOUT", "true"),
        ],
    )
    .await;

    assert_eq!(output.status.code(), Some(2));
    let (stdout, stderr) = (text(&output.stdout), text(&output.stderr));
    for secret in [SECRET_TOKEN, SECRET_BODY] {
        assert!(
            !stdout.contains(secret) && !stderr.contains(secret),
            "{secret} printed"
        );
    }
    assert_eq!(stderr, "Provisioning or cleanup failed (HttpStatus)\n");
    let xml = junit(dir.path());
    assert!(
        xml.contains(
            "<testsuite name=\"deployment-fixture\" tests=\"1\" failures=\"0\" errors=\"1\""
        ),
        "{xml}"
    );
    assert!(
        xml.contains("<error message=\"Provisioning or cleanup failed (HttpStatus)\">"),
        "{xml}"
    );
    assert!(
        !xml.contains(SECRET_TOKEN) && !xml.contains(SECRET_BODY),
        "{xml}"
    );
    assert_eq!(
        summary(dir.path()),
        serde_json::json!({"exit_code": 2, "failure": "deployment-fixture"})
    );
    assert_report_lines(stdout, dir.path(), &[RESULTS_FILE, SUMMARY_FILE]);
    // The file the failed PUT may have created was still deleted.
    let methods: Vec<String> = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|request| request.method.to_string())
        .collect();
    assert_eq!(methods, ["PUT", "DELETE"]);
}

/// Runs a provisioned run whose configuration (`vars`, on top of a base path
/// and the Storage API) is refused, and checks it reports exactly
/// `Provisioning or cleanup failed (Config): {rule}`, on stderr and in the
/// JUnit, without `secret` or the whole `raw` value anywhere in its output
/// or reports, and without sending the Storage API anything.
async fn refused_provisioned_configuration(
    vars: &[(&str, &str)],
    rule: &str,
    secret: &str,
    raw: &str,
) {
    let storage = MockServer::start().await;
    let dir = tempfile::tempdir().unwrap();
    let storage_endpoint = storage.uri();
    let mut all = vec![
        ("CONFORMANCE_BASE_PATH", "/container/conformance"),
        ("CONFORMANCE_STORAGE_ENDPOINT", storage_endpoint.as_str()),
        ("CONFORMANCE_REPORT_DIR", dir.path().to_str().unwrap()),
        ("CONFORMANCE_REPORT_STDOUT", "true"),
    ];
    all.extend_from_slice(vars);

    let output = run(&[], &all).await;

    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let message = format!("Provisioning or cleanup failed (Config): {rule}");
    let (stdout, stderr) = (text(&output.stdout), text(&output.stderr));
    assert_eq!(stderr, format!("{message}\n"));
    let xml = junit(dir.path());
    assert!(
        xml.contains(&format!("<error message=\"{message}\">{message}</error>")),
        "{xml}"
    );
    let summary_json = std::fs::read_to_string(dir.path().join(SUMMARY_FILE)).unwrap();
    for leaked in [secret, raw] {
        for (place, shown) in [
            ("stdout", stdout),
            ("stderr", stderr),
            ("JUnit", xml.as_str()),
            ("summary.json", summary_json.as_str()),
        ] {
            assert!(!shown.contains(leaked), "{leaked:?} in {place}: {shown}");
        }
    }
    assert_report_lines(stdout, dir.path(), &[RESULTS_FILE, SUMMARY_FILE]);
    assert!(
        storage.received_requests().await.unwrap().is_empty(),
        "the Storage API was called"
    );
}

#[tokio::test]
async fn provisioned_run_refused_endpoint_names_the_variable_never_its_value() {
    const SECRET: &str = "s3cr3t-pass";
    const ENDPOINT: &str =
        "https://user:s3cr3t-pass@service.example.com:50001/path?token=s3cr3t-pass";
    refused_provisioned_configuration(
        &[
            ("CONFORMANCE_AUTH_TOKEN", "local-token"),
            ("CONFORMANCE_GRPC_ENDPOINT", ENDPOINT),
        ],
        "CONFORMANCE_GRPC_ENDPOINT: endpoint must be http[s]://host:port without \
         credentials, path, query or fragment",
        SECRET,
        ENDPOINT,
    )
    .await;
}

#[tokio::test]
async fn provisioned_run_refused_token_never_shows_the_token() {
    // Refused when provisioning reads it, before any request: whitespace is
    // not allowed in a bearer token.
    const SECRET: &str = "s3cr3t";
    const TOKEN: &str = "Bearer s3cr3t tok";
    refused_provisioned_configuration(
        &[("CONFORMANCE_AUTH_TOKEN", TOKEN)],
        "Bearer token must be nonempty printable ASCII without whitespace",
        SECRET,
        TOKEN,
    )
    .await;
}

// --- A real SIGTERM or SIGINT. Without the suite's handlers either signal
// kills the process, which then has no exit code and writes no report.

/// The JUnit error and stderr text of an interruption.
const INTERRUPTED: &str = "interrupted by SIGTERM or SIGINT";

/// Sends `signal` to a run waiting for a service that accepts its
/// connection and never answers, and checks the run reports the preflight
/// interrupted and exits 2.
async fn signal_while_waiting_for_readiness(signal: &str) {
    let service = SilentServer::start().await;
    let dir = tempfile::tempdir().unwrap();
    let child = spawn(&[
        ("CONFORMANCE_AUTH_TOKEN", "local-token"),
        ("CONFORMANCE_BASE_PATH", "/container/conformance"),
        ("CONFORMANCE_GRPC_ENDPOINT", &service.endpoint),
        ("CONFORMANCE_READY_TIMEOUT", "600"),
        ("CONFORMANCE_REPORT_DIR", dir.path().to_str().unwrap()),
        ("CONFORMANCE_REPORT_STDOUT", "true"),
    ]);

    // The first health check is on the wire: the run is waiting for it.
    service.wait_for_a_call(RUN_BUDGET).await;
    send(signal, &child);
    let output = finish(child).await;

    assert_eq!(output.status.code(), Some(2), "{output:?}");
    assert_eq!(
        text(&output.stderr),
        format!("Conformance preflight failed: {INTERRUPTED}\n")
    );
    let xml = junit(dir.path());
    assert!(
        xml.contains(
            "<testsuite name=\"conformance-preflight\" tests=\"1\" failures=\"0\" errors=\"1\""
        ),
        "{xml}"
    );
    assert!(
        xml.contains(&format!("<error message=\"{INTERRUPTED}\">")),
        "{xml}"
    );
    // A run that never reached its cases has no summary to print.
    assert!(!dir.path().join(SUMMARY_FILE).exists());
    assert_report_lines(text(&output.stdout), dir.path(), &[RESULTS_FILE]);
}

#[tokio::test]
async fn sigterm_while_waiting_for_readiness_reports_the_preflight_interrupted_and_exits_2() {
    signal_while_waiting_for_readiness("TERM").await;
}

#[tokio::test]
async fn sigint_while_waiting_for_readiness_reports_the_preflight_interrupted_and_exits_2() {
    signal_while_waiting_for_readiness("INT").await;
}

#[tokio::test]
async fn sigterm_during_a_case_reports_the_run_suite_as_the_failure_and_exits_2() {
    // The service is ready, then answers nothing else: the first case waits
    // on its first call.
    let service = FakeLive::start(|method, _| match method {
        "Check" => Answer::Ok,
        _ => Answer::Hang,
    })
    .await;
    let dir = tempfile::tempdir().unwrap();
    let child = spawn(&[
        ("CONFORMANCE_AUTH_TOKEN", "local-token"),
        ("CONFORMANCE_BASE_PATH", "/container/conformance"),
        ("CONFORMANCE_GRPC_ENDPOINT", &service.endpoint),
        ("CONFORMANCE_TIMEOUT", "600"),
        ("CONFORMANCE_REPORT_DIR", dir.path().to_str().unwrap()),
        ("CONFORMANCE_REPORT_STDOUT", "true"),
    ]);

    let deadline = Instant::now() + RUN_BUDGET;
    while !service.methods().iter().any(|method| method != "Check") {
        assert!(Instant::now() < deadline, "no case reached the service");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    send("TERM", &child);
    let output = finish(child).await;

    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let summary = summary(dir.path());
    assert_eq!(summary["exit_code"], 2);
    assert_eq!(summary["executed"], 1);
    assert_eq!(summary["failure"], "live-edit-conformance");
    let xml = junit(dir.path());
    assert!(
        xml.contains(
            "<testsuite name=\"live-edit-conformance\" tests=\"1\" failures=\"0\" errors=\"1\""
        ),
        "{xml}"
    );
    assert!(
        xml.contains(&format!("<error message=\"{INTERRUPTED}\">")),
        "{xml}"
    );
    let stderr = text(&output.stderr);
    assert!(
        stderr.starts_with(&format!("Conformance run {INTERRUPTED}: 1 of ")),
        "{stderr}"
    );
    assert_report_lines(
        &report_lines(text(&output.stdout)),
        dir.path(),
        &[RESULTS_FILE, SUMMARY_FILE],
    );
}

#[tokio::test]
async fn sigterm_on_a_provisioned_run_bounds_the_unanswered_delete_and_exits_2() {
    // The base layer is created; the run waits for readiness; SIGTERM; the
    // layer's DELETE then never gets an answer. Ten seconds after it starts
    // the DELETE is given up, and the reports are written and printed.
    let storage = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(201))
        .mount(&storage)
        .await;
    Mock::given(method("DELETE"))
        .respond_with(ResponseTemplate::new(204).set_delay(Duration::from_secs(600)))
        .mount(&storage)
        .await;
    let service = SilentServer::start().await;
    let dir = tempfile::tempdir().unwrap();
    let storage_endpoint = storage.uri();
    let child = spawn(&[
        ("CONFORMANCE_AUTH_TOKEN", "local-token"),
        ("CONFORMANCE_BASE_PATH", "/container/conformance"),
        ("CONFORMANCE_GRPC_ENDPOINT", &service.endpoint),
        ("CONFORMANCE_STORAGE_ENDPOINT", &storage_endpoint),
        ("CONFORMANCE_TIMEOUT", "600"),
        ("CONFORMANCE_READY_TIMEOUT", "600"),
        ("CONFORMANCE_REPORT_DIR", dir.path().to_str().unwrap()),
        ("CONFORMANCE_REPORT_STDOUT", "true"),
    ]);

    service.wait_for_a_call(RUN_BUDGET).await;
    let signalled = Instant::now();
    send("TERM", &child);
    let output = finish(child).await;

    assert_eq!(output.status.code(), Some(2), "{output:?}");
    // The DELETE started after the signal and waited out the bound.
    assert!(signalled.elapsed() >= INTERRUPT_DELETE_BOUND);
    assert_eq!(
        text(&output.stderr),
        format!(
            "Conformance preflight failed: {INTERRUPTED}\nProvisioning or cleanup failed (Transport)\n"
        )
    );
    let xml = junit(dir.path());
    let preflight = xml
        .find("<testsuite name=\"conformance-preflight\"")
        .unwrap_or_else(|| panic!("{xml}"));
    let fixture = xml
        .find("<testsuite name=\"deployment-fixture\" tests=\"1\" failures=\"0\" errors=\"1\"")
        .unwrap_or_else(|| panic!("{xml}"));
    assert!(preflight < fixture, "{xml}");
    assert!(
        xml.contains("<error message=\"Provisioning or cleanup failed (Transport)\">"),
        "{xml}"
    );
    assert_eq!(
        summary(dir.path()),
        serde_json::json!({"exit_code": 2, "failure": "deployment-fixture"})
    );
    assert_report_lines(
        text(&output.stdout),
        dir.path(),
        &[RESULTS_FILE, SUMMARY_FILE],
    );
    let methods: Vec<String> = storage
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|request| request.method.to_string())
        .collect();
    assert_eq!(methods, ["PUT", "DELETE"]);
}
