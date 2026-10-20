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

//! Binary-level tests of the service's startup preamble: what the real
//! `live-edit` process logs, and where it stops, before its first dependency
//! step.
//!
//! Each test spawns the built binary (`CARGO_BIN_EXE_live-edit`) with a
//! cleared environment, JSON logs, and `LIVE_EDIT_PORT` set to a port this
//! test holds open with its own `TcpListener`. The child therefore exits 1 at
//! the gRPC bind (`failed to bind gRPC listener`), the first dependency step,
//! right after the startup preamble — no Valkey and no network are needed.
//! `LIVE_EDIT_VALKEY_URL` points at a closed local port with a 1 s connect
//! timeout as a backstop, so a child that ever got past the bind would still
//! fail fast instead of hanging. The service children read their pod memory
//! limit from a 5 GiB cgroup fixture, not from the runner's own cgroup.
//!
//! - **Configuration WARNs reach the log.** `Config::from_env` logs startup
//!   WARNs while it parses; they reach the log only if the subscriber is
//!   installed first. The test is differential: it runs `Config::from_env`
//!   in-process for the same environment under a capturing JSON subscriber,
//!   and asserts every WARN captured there appears in the child's stdout
//!   before the bind ERROR. A configuration WARN added later is covered with
//!   no edit here. An assertion that the in-process set is non-empty keeps
//!   the comparison from passing vacuously, and per-case markers pin which
//!   WARNs the case must raise.
//! - **The pod memory plan is read from the cgroup and logged, or refused,
//!   before the bind.** On the 5 GiB fixture the child logs one INFO line
//!   with the limit, its `cgroup_v2` source and the chart pod's budgets; on
//!   an unlimited (`max`) cgroup, a WARN naming the reference size and the
//!   reason (Linux only, where the WARN is raised); on a 1 GiB cgroup it exits
//!   1 with a configuration error naming the smallest limit it needs.
//! - **StorageApi without a token file is refused at configuration load.**
//!   The child exits 1 with `configuration error:` on stderr naming
//!   `LIVE_EDIT_SERVICE_TOKEN_FILE`, and its stdout shows it never reached
//!   the gRPC bind.
//! - **An OTLP endpoint is `http://` or `https://`, or startup stops.** A
//!   trace or metrics push endpoint that is neither, or an `https://` one
//!   whose trust store holds no certificate (`SSL_CERT_FILE` an empty file),
//!   exits 1 with `configuration error:` on stderr naming the variable, and
//!   never reaches the gRPC bind; `live-edit scaler` refuses a bad trace
//!   endpoint the same way. Any compression setting, `none` included, stops
//!   startup the same way, naming the compression variable the exporter
//!   reads (the signal's own before the generic one) and not the endpoint.
//!   The same empty trust store with `http://` endpoints reaches the bind,
//!   and both export startup lines say `transport: plaintext`: TLS, and its
//!   trust store, are only for `https://`. Neither startup line logs the
//!   endpoint, which may carry credentials. (The `https://` path that works
//!   is `otlp_tls_export_binary`.)
//! - **`live-edit scaler` dispatches before the service configuration
//!   loads.** These children get the `scaler` argument and no service
//!   variable at all, so they never reach the service's startup or its gRPC
//!   bind: a missing scaler variable exits 1 naming it, and a complete scaler
//!   configuration gets as far as the scaler's own first dependency, the
//!   in-cluster Kubernetes environment, which a test process lacks.

use std::collections::HashMap;
use std::io::Read;
use std::net::TcpListener;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use tracing_subscriber::fmt::MakeWriter;

use live_edit::config::{Config, PodMemorySource};

/// The service binary Cargo builds for this package's integration tests.
pub(crate) const LIVE_EDIT_BIN: &str = env!("CARGO_BIN_EXE_live-edit");

/// Deadline for the child to exit. The healthy path exits within a second of
/// spawning (it fails at the gRPC bind, before any dependency); the budget
/// only has to absorb a loaded runner, and stays under nextest's 120 s
/// hang-killer so a stuck child fails with a message instead of a SIGKILL.
const CHILD_EXIT_BUDGET: Duration = Duration::from_secs(60);

/// Interval between `try_wait` probes while waiting for the child to exit.
const CHILD_EXIT_POLL: Duration = Duration::from_millis(20);

/// The ERROR message `main.rs` logs when the gRPC bind fails, the first
/// dependency step of startup.
const BIND_FAILURE_MESSAGE: &str = "failed to bind gRPC listener";

/// Prefix of the INFO message `main.rs` logs once the gRPC bind succeeds.
const BIND_SUCCESS_MESSAGE_PREFIX: &str = "gRPC listener bound";

/// The two endpoints `Config::from_env` requires under the default auth shape
/// (authz on, userinfo off). Never contacted: every child these tests run
/// stops before any auth client is built — here at the gRPC bind, and in
/// `otlp_tls_export_binary` at the Valkey connect, which comes first.
pub(crate) const AUTH_ENV: [(&str, &str); 2] = [
    (
        "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
        "http://auth.example.com",
    ),
    ("LIVE_EDIT_USERINFO_ENDPOINT", "http://userinfo.example.com"),
];

/// One log record, reduced to what the comparison keys on.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LogRecord {
    level: String,
    message: String,
}

/// What the child process left behind.
struct ChildRun {
    status: ExitStatus,
    stdout: String,
    stderr: String,
}

/// A local port held open for the test's lifetime, so the child's gRPC bind
/// on `0.0.0.0:<port>` fails with `EADDRINUSE`. Never accepted from: a
/// client connecting to it completes the TCP handshake in the backlog and
/// then hears nothing.
pub(crate) struct HeldPort {
    _listener: TcpListener,
    pub(crate) port: u16,
}

pub(crate) fn hold_port() -> HeldPort {
    let listener = TcpListener::bind("0.0.0.0:0").expect("bind a port to hold");
    let port = listener.local_addr().expect("held port address").port();
    HeldPort {
        _listener: listener,
        port,
    }
}

/// The chart's default pod memory limit, 5 GiB, in bytes.
const CHART_POD_MEMORY_LIMIT_BYTES: u64 = 5 << 30;

/// A cgroup v2 tree with a 5 GiB `memory.max`, the chart's default limit, for
/// the child to read its pod memory limit from instead of the machine's own
/// (`MEASUREMENT_POD_MEMORY_CGROUP_ROOT`, read by `test-support` builds): a
/// runner's cgroup can be small enough that the service would refuse to start
/// before the startup preamble these tests watch. Removed when dropped.
pub(crate) fn pod_memory_fixture() -> tempfile::TempDir {
    pod_memory_fixture_with(&format!("{CHART_POD_MEMORY_LIMIT_BYTES}\n"))
}

/// [`pod_memory_fixture`] with `memory_max` as the cgroup's `memory.max`.
fn pod_memory_fixture_with(memory_max: &str) -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("create the cgroup fixture");
    let files = [
        ("proc/self/cgroup", "0::/\n"),
        (
            "proc/self/mountinfo",
            "30 23 0:26 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup2 rw\n",
        ),
        ("proc/meminfo", "MemTotal:       67108864 kB\n"),
        ("sys/fs/cgroup/memory.max", memory_max),
    ];
    for (path, text) in files {
        let path = root.path().join(path);
        std::fs::create_dir_all(path.parent().expect("a fixture file has a parent"))
            .expect("create a fixture directory");
        std::fs::write(&path, text).expect("write a fixture file");
    }
    root
}

/// The child's environment: the auth endpoints, the case's variables, JSON
/// logs, the held port, the Valkey backstop, and the pod memory fixture at
/// `cgroup_root` ([`pod_memory_fixture`]).
fn startup_env(
    case: &[(&str, &str)],
    port: u16,
    cgroup_root: &std::path::Path,
) -> Vec<(String, String)> {
    let port = port.to_string();
    let cgroup_root = cgroup_root
        .to_str()
        .expect("the fixture path is UTF-8")
        .to_owned();
    let fixed = [
        ("LIVE_EDIT_LOG_FORMAT", "json"),
        ("LIVE_EDIT_PORT", port.as_str()),
        ("LIVE_EDIT_VALKEY_URL", "redis://127.0.0.1:1"),
        ("LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS", "1"),
        ("MEASUREMENT_POD_MEMORY_CGROUP_ROOT", cgroup_root.as_str()),
    ];
    AUTH_ENV
        .iter()
        .chain(case)
        .chain(fixed.iter())
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// Read a child pipe to its end on a helper thread, so a full pipe can never
/// block the child while the test waits for it to exit.
fn drain_pipe<R: Read + Send + 'static>(mut pipe: R) -> JoinHandle<String> {
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        pipe.read_to_end(&mut bytes).expect("read child pipe");
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

/// Spawn the service binary with exactly `env` (nothing inherited) and wait
/// for it to exit within [`CHILD_EXIT_BUDGET`].
fn run_binary(env: &[(String, String)]) -> ChildRun {
    run_binary_with_args(&[], env)
}

/// [`run_binary`] with command-line arguments.
fn run_binary_with_args(args: &[&str], env: &[(String, String)]) -> ChildRun {
    let mut child = Command::new(LIVE_EDIT_BIN)
        .args(args)
        .env_clear()
        .envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the live-edit binary");
    let stdout = drain_pipe(child.stdout.take().expect("child stdout is piped"));
    let stderr = drain_pipe(child.stderr.take().expect("child stderr is piped"));

    let deadline = Instant::now() + CHILD_EXIT_BUDGET;
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll the child") {
            break status;
        }
        if Instant::now() >= deadline {
            // Best-effort cleanup on the failure path; the panic below is the
            // finding, whether or not the kill lands.
            child.kill().ok();
            child.wait().ok();
            panic!(
                "live-edit {args:?} did not exit within {CHILD_EXIT_BUDGET:?}; every case here \
                 exits at a startup failure (the gRPC bind on the held port, a refused \
                 configuration, or the scaler's missing cluster environment). stdout: {}\n\
                 stderr: {}",
                stdout.join().unwrap_or_default(),
                stderr.join().unwrap_or_default(),
            );
        }
        std::thread::sleep(CHILD_EXIT_POLL);
    };
    ChildRun {
        status,
        stdout: stdout.join().expect("stdout reader thread"),
        stderr: stderr.join().expect("stderr reader thread"),
    }
}

/// Parse JSON log lines (`{"level": .., "fields": {"message": ..}, ..}`).
/// Every non-empty line must be a JSON record: with `LIVE_EDIT_LOG_FORMAT=json`
/// stdout carries the log stream and nothing else.
fn parse_records(output: &str) -> Vec<LogRecord> {
    output
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let value: serde_json::Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("log line is not JSON ({e}): {line}"));
            LogRecord {
                level: value["level"]
                    .as_str()
                    .unwrap_or_else(|| panic!("log record has no level: {line}"))
                    .to_string(),
                message: value["fields"]["message"]
                    .as_str()
                    .unwrap_or_else(|| panic!("log record has no message: {line}"))
                    .to_string(),
            }
        })
        .collect()
}

/// `MakeWriter` capturing every byte into a shared buffer. Same small helper
/// as the sibling modules, duplicated per the project convention of defining
/// it per test module.
#[derive(Clone)]
struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

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

/// Whether an ambient variable could change what `Config::from_env` or the
/// child's subscriber does, and so must be pinned unset in-process to match
/// the child's cleared environment.
fn is_startup_relevant(name: &str) -> bool {
    name.starts_with("LIVE_EDIT_")
        || name.starts_with("OTEL_")
        || name == "RUST_LOG"
        || name == "HOSTNAME"
}

/// The WARN records `Config::from_env` emits in-process for exactly `env`:
/// every startup-relevant ambient variable not in `env` is pinned unset, so
/// the in-process environment matches the child's.
fn config_warns_in_process(env: &[(String, String)]) -> Vec<LogRecord> {
    let mut vars: Vec<(String, Option<String>)> = std::env::vars_os()
        .filter_map(|(name, _)| name.into_string().ok())
        .filter(|name| is_startup_relevant(name) && !env.iter().any(|(k, _)| k == name))
        .map(|name| (name, None))
        .collect();
    vars.extend(env.iter().map(|(k, v)| (k.clone(), Some(v.clone()))));

    let buffer = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .json()
        .with_writer(CapturingWriter(buffer.clone()))
        .with_max_level(tracing::Level::WARN)
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        temp_env::with_vars(vars, || Config::from_env_with(PodMemorySource::Reference))
            .expect("the case's environment must parse in-process")
    });

    let text = String::from_utf8(buffer.lock().unwrap().clone())
        .expect("captured log output must be UTF-8");
    parse_records(&text)
        .into_iter()
        .filter(|record| record.level == "WARN")
        .collect()
}

/// The child's log records before the bind ERROR, asserting the ERROR is
/// there: it is what places every earlier record ahead of the first
/// dependency step.
fn records_before_bind_failure(run: &ChildRun) -> Vec<LogRecord> {
    let records = parse_records(&run.stdout);
    let bind = records
        .iter()
        .position(|r| r.level == "ERROR" && r.message == BIND_FAILURE_MESSAGE)
        .unwrap_or_else(|| {
            panic!(
                "the child must fail at the gRPC bind on the held port; stdout: {}\nstderr: {}",
                run.stdout, run.stderr
            )
        });
    records[..bind].to_vec()
}

/// Assert every in-process WARN (with multiplicity) is among `logged`.
fn assert_every_config_warn_logged(expected: &[LogRecord], logged: &[LogRecord]) {
    let mut want: HashMap<&LogRecord, usize> = HashMap::new();
    for record in expected {
        *want.entry(record).or_default() += 1;
    }
    for (record, count) in want {
        let got = logged.iter().filter(|r| *r == record).count();
        assert!(
            got >= count,
            "Config::from_env emits {record:?} {count} time(s) for this environment, but the \
             binary logged it {got} time(s) before the gRPC bind ERROR — a WARN emitted while \
             no subscriber is installed is dropped. Logged before the bind: {logged:#?}"
        );
    }
}

/// Assert some WARN record in `records` contains `marker`.
fn assert_warn_with(records: &[LogRecord], marker: &str, source: &str) {
    assert!(
        records
            .iter()
            .any(|r| r.level == "WARN" && r.message.contains(marker)),
        "{source} must carry a WARN containing {marker:?}; got: {records:#?}"
    );
}

/// The case's `Config::from_env` WARNs, computed in-process, all appear in the
/// child's stdout before the gRPC bind ERROR. Returns the child's records
/// before the bind so the caller can assert its own markers on them.
fn assert_binary_logs_config_warns_before_bind(
    case: &[(&str, &str)],
    in_process_markers: &[&str],
) -> Vec<LogRecord> {
    let held = hold_port();
    let cgroup = pod_memory_fixture();
    let env = startup_env(case, held.port, cgroup.path());

    let expected = config_warns_in_process(&env);
    // The case's own markers first, so a missing WARN is reported by name
    // rather than by the generic guard after them.
    for marker in in_process_markers {
        assert_warn_with(&expected, marker, "in-process Config::from_env");
    }
    assert!(
        !expected.is_empty(),
        "non-vacuity: Config::from_env must emit at least one WARN for this case, or the \
         differential below proves nothing"
    );

    let run = run_binary(&env);
    assert_eq!(
        run.status.code(),
        Some(1),
        "the child must exit 1 at the gRPC bind; stdout: {}\nstderr: {}",
        run.stdout,
        run.stderr
    );
    let before_bind = records_before_bind_failure(&run);
    assert_every_config_warn_logged(&expected, &before_bind);
    before_bind
}

#[test]
fn live_edit_binary_with_misconfigured_knobs_logs_every_config_warn_before_grpc_bind() {
    let before_bind = assert_binary_logs_config_warns_before_bind(
        &[
            ("LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS", "0"),
            ("LIVE_EDIT_EXT_VALUE_VERIFY", "observe"),
            ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES", "8192"),
            ("LIVE_EDIT_LAYER_STREAM_MAXLEN", "4096"),
            // `0` turns the live stream's byte budget off, which keeps the
            // footprint WARN firing: at the default budget it is silent.
            ("LIVE_EDIT_LAYER_STREAM_MAX_BYTES", "0"),
            ("LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED", "true"),
        ],
        &[
            "LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0",
            "LIVE_EDIT_EXT_VALUE_VERIFY=observe:",
            "exceeds live-stream retention",
            "worst-case per-object live-stream footprint",
        ],
    );

    // Not a `Config::from_env` WARN: `main.rs` logs it from
    // `log_service_token_startup_warnings` right after configuration loads,
    // so it too must precede the first dependency step.
    assert_warn_with(
        &before_bind,
        "LIVE_EDIT_SERVICE_TOKEN_CLIENT_CREDENTIALS_ENABLED=true",
        "the binary's output before the gRPC bind",
    );
}

#[test]
fn live_edit_binary_with_tail_and_stream_byte_caps_logs_every_config_warn_before_grpc_bind() {
    // A 64 MiB per-tail byte cap over a 32 MiB live-stream byte budget raises
    // the tail-bytes WARN. That budget also keeps the footprint WARN silent
    // (32 MiB + 2 × the default max layer size is under its threshold), so
    // the tail-bytes WARN is what makes the in-process set non-empty.
    assert_binary_logs_config_warns_before_bind(
        &[
            ("LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES", "67108864"),
            ("LIVE_EDIT_LAYER_STREAM_MAX_BYTES", "33554432"),
        ],
        &["exceeds the live stream's byte budget"],
    );
}

#[test]
fn live_edit_binary_storage_api_without_token_file_exits_with_config_error_before_grpc_bind() {
    let held = hold_port();
    let cgroup = pod_memory_fixture();
    let env = startup_env(
        &[
            ("LIVE_EDIT_CHECKPOINT_ADAPTER", "storageapi"),
            (
                "LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT",
                "https://storage.example.com",
            ),
            ("LIVE_EDIT_CHECKPOINT_BASE_PATH", "/projects/my-project"),
        ],
        held.port,
        cgroup.path(),
    );

    let run = run_binary(&env);

    assert_eq!(
        run.status.code(),
        Some(1),
        "StorageApi without LIVE_EDIT_SERVICE_TOKEN_FILE must exit 1; stdout: {}\nstderr: {}",
        run.stdout,
        run.stderr
    );
    assert!(
        run.stderr.contains("configuration error:")
            && run.stderr.contains("LIVE_EDIT_SERVICE_TOKEN_FILE"),
        "the refusal must be a configuration error on stderr naming the variable; stderr: {}",
        run.stderr
    );
    // With the port held, a child that got as far as the bind would log one
    // of these two lines; neither may appear, so the refusal preceded every
    // dependency step.
    assert!(
        !run.stdout.contains(BIND_SUCCESS_MESSAGE_PREFIX)
            && !run.stdout.contains(BIND_FAILURE_MESSAGE),
        "the refusal must come before the gRPC bind; stdout: {}",
        run.stdout
    );
}

/// The child's JSON log records before the gRPC bind ERROR, whole (fields
/// included).
fn json_records_before_bind_failure(run: &ChildRun) -> Vec<serde_json::Value> {
    let bind = records_before_bind_failure(run).len();
    run.stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(bind)
        .map(|line| serde_json::from_str(line).expect("a JSON log record"))
        .collect()
}

/// The one record in `records` whose message starts with `prefix`.
fn record_starting<'a>(
    records: &'a [serde_json::Value],
    prefix: &str,
    run: &ChildRun,
) -> &'a serde_json::Value {
    let found: Vec<_> = records
        .iter()
        .filter(|r| {
            r["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with(prefix))
        })
        .collect();
    assert_eq!(
        found.len(),
        1,
        "one {prefix:?} line before the gRPC bind; stdout: {}",
        run.stdout
    );
    found[0]
}

/// The startup line's message with a readable limit.
const POD_MEMORY_PLAN_MESSAGE_PREFIX: &str = "pod memory budgets derived from the pod memory limit";

/// The startup line's message with no readable limit, a WARN on Linux.
const POD_MEMORY_REFERENCE_MESSAGE_PREFIX: &str = "no readable pod memory limit";

/// W12: the binary reads its pod memory limit from its cgroup (a 5 GiB
/// fixture) at startup and logs the plan derived from it, once, at INFO,
/// before the gRPC bind: the limit and its source, and the budgets of the
/// chart's 5 GiB pod, the external-value mirror's at 0 without a backup
/// store.
#[test]
fn live_edit_binary_logs_its_pod_memory_plan_from_its_cgroup_before_grpc_bind() {
    const MIB: u64 = 1024 * 1024;
    let held = hold_port();
    let cgroup = pod_memory_fixture();
    let run = run_binary(&startup_env(&[], held.port, cgroup.path()));
    assert_eq!(
        run.status.code(),
        Some(1),
        "the child exits at the gRPC bind; stderr: {}",
        run.stderr
    );
    let records = json_records_before_bind_failure(&run);
    let line = record_starting(&records, POD_MEMORY_PLAN_MESSAGE_PREFIX, &run);
    assert_eq!(line["level"], "INFO", "{line}");
    let fields = &line["fields"];
    assert_eq!(fields["source"], "cgroup_v2", "{line}");
    assert_eq!(fields["reason"], "none", "{line}");
    assert_eq!(
        fields["pod_memory_limit_bytes"], CHART_POD_MEMORY_LIMIT_BYTES,
        "{line}"
    );
    for (field, value) in [
        ("upload_budget_bytes", 384 * MIB),
        ("structural_cache_budget_bytes", 256 * MIB),
        ("max_subscriptions", 4_096),
        ("retirement_budget_bytes", 10 * MIB),
        // No backup store is configured, so the plan funds no mirror.
        ("ext_mirror_budget_bytes", 0),
    ] {
        assert_eq!(fields[field], value, "{field}: {line}");
    }
    for field in [
        "catchup_budget_bytes",
        "line_total_bytes",
        "line_limit_bytes",
        "line_headroom_bytes",
        "margin_left_bytes",
        "minimum_limit_mib",
    ] {
        assert!(fields[field].is_u64(), "{field}: {line}");
    }
    assert!(
        records.iter().all(|r| !r["fields"]["message"]
            .as_str()
            .is_some_and(|m| m.starts_with(POD_MEMORY_REFERENCE_MESSAGE_PREFIX))),
        "a readable limit raises no reference WARN; stdout: {}",
        run.stdout
    );
}

/// W7 at the binary: a cgroup that sets no memory limit (`max`) sizes the
/// pod as the 5 GiB reference and says so with a WARN on Linux, naming the
/// reason, before the gRPC bind.
#[cfg(target_os = "linux")]
#[test]
fn live_edit_binary_with_an_unlimited_cgroup_warns_it_is_sized_as_the_reference_pod() {
    let held = hold_port();
    let cgroup = pod_memory_fixture_with("max\n");
    let run = run_binary(&startup_env(&[], held.port, cgroup.path()));
    assert_eq!(
        run.status.code(),
        Some(1),
        "the child exits at the gRPC bind; stderr: {}",
        run.stderr
    );
    let records = json_records_before_bind_failure(&run);
    let line = record_starting(&records, POD_MEMORY_REFERENCE_MESSAGE_PREFIX, &run);
    assert_eq!(line["level"], "WARN", "{line}");
    let fields = &line["fields"];
    assert_eq!(fields["source"], "reference", "{line}");
    assert_eq!(fields["reason"], "unlimited", "{line}");
    assert_eq!(
        fields["pod_memory_limit_bytes"], CHART_POD_MEMORY_LIMIT_BYTES,
        "{line}"
    );
}

/// A pod whose memory limit is below what its configuration needs never
/// starts: the child exits 1 with a configuration error naming the smallest
/// limit it needs, before the gRPC bind. 1 GiB is below that for any pool
/// size: the terms no pool setting sizes already pass its 819.2 MiB line
/// (80 % of it) — the 155 MiB floor, the upload budget at its 209 MiB floor,
/// one maximum-size upload (409.1), the checkpoint reference read (172.9) and
/// the fan-out window term (143.0) sum to 1,089.1 MiB, before the cold-start
/// reads, the audit or any catch-up budget.
#[test]
fn live_edit_binary_below_the_minimum_pod_memory_limit_exits_with_a_configuration_error() {
    let held = hold_port();
    let cgroup = pod_memory_fixture_with("1073741824\n");
    let run = run_binary(&startup_env(&[], held.port, cgroup.path()));
    assert_eq!(
        run.status.code(),
        Some(1),
        "stdout: {}\nstderr: {}",
        run.stdout,
        run.stderr
    );
    for part in [
        "configuration error:",
        "this pod's memory limit, 1,024 MiB (cgroup v2), is below the",
        "MiB this configuration needs",
        "raise resources.limits.memory to at least",
    ] {
        assert!(
            run.stderr.contains(part),
            "{part:?} on stderr; stderr: {}",
            run.stderr
        );
    }
    assert!(
        !run.stdout.contains(BIND_SUCCESS_MESSAGE_PREFIX)
            && !run.stdout.contains(BIND_FAILURE_MESSAGE),
        "the refusal must come before the gRPC bind; stdout: {}",
        run.stdout
    );
}

/// The trace export endpoint variable.
const TRACE_ENDPOINT_VAR: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";

/// The metrics push endpoint variable.
const METRICS_ENDPOINT_VAR: &str = "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT";

/// The INFO line `main` logs when trace export is enabled.
const TRACE_EXPORT_ENABLED_MESSAGE: &str = "OpenTelemetry trace export enabled";

/// The INFO line the meter provider logs when the metrics push is attached.
const METRICS_PUSH_ENABLED_MESSAGE: &str = "OTLP metrics push exporter enabled";

/// An empty file to point `SSL_CERT_FILE` at: a trust store with no
/// certificate in it. Removed when dropped.
fn empty_trust_store() -> tempfile::NamedTempFile {
    tempfile::NamedTempFile::new().expect("create the empty trust store")
}

/// Run the child with `case` and assert it refused its configuration: exit 1,
/// `configuration error:` on stderr carrying every one of `parts`, and no
/// gRPC bind line on stdout. Returns the run for further assertions.
fn assert_refused_before_grpc_bind(case: &[(&str, &str)], parts: &[&str]) -> ChildRun {
    let held = hold_port();
    let cgroup = pod_memory_fixture();
    let run = run_binary(&startup_env(case, held.port, cgroup.path()));
    assert_eq!(
        run.status.code(),
        Some(1),
        "{case:?}: stdout: {}\nstderr: {}",
        run.stdout,
        run.stderr
    );
    for part in ["configuration error:"].iter().chain(parts) {
        assert!(
            run.stderr.contains(part),
            "{case:?}: {part:?} on stderr; stderr: {}",
            run.stderr
        );
    }
    assert!(
        !run.stdout.contains(BIND_SUCCESS_MESSAGE_PREFIX)
            && !run.stdout.contains(BIND_FAILURE_MESSAGE),
        "{case:?}: the refusal must come before the gRPC bind; stdout: {}",
        run.stdout
    );
    run
}

/// T6 (a), (b): an endpoint that is not an `http://` or `https://` URL stops
/// startup with a configuration error naming its variable, for traces and
/// for the metrics push: an unsupported scheme, and no scheme at all.
#[test]
fn live_edit_binary_with_an_unsupported_otlp_endpoint_exits_with_a_config_error_naming_it() {
    for var in [TRACE_ENDPOINT_VAR, METRICS_ENDPOINT_VAR] {
        for value in [
            "grpc://collector.example.com:4317",
            "collector.example.com:4317",
        ] {
            assert_refused_before_grpc_bind(&[(var, value)], &[var]);
        }
    }
}

/// T6 (c): an `https://` endpoint whose TLS settings cannot be built — here
/// a trust store with no certificate — stops startup with a configuration
/// error naming the variable and the cause tonic reports, which the
/// exporter's own error would hide behind "transport error".
#[test]
fn live_edit_binary_with_https_otlp_endpoint_and_no_trust_roots_exits_naming_the_cause() {
    let trust_store = empty_trust_store();
    let trust_store_path = trust_store.path().to_str().expect("UTF-8 temp path");
    for var in [TRACE_ENDPOINT_VAR, METRICS_ENDPOINT_VAR] {
        assert_refused_before_grpc_bind(
            &[
                (var, "https://collector.example.com:4317"),
                ("SSL_CERT_FILE", trust_store_path),
            ],
            &[var, "no native certs found"],
        );
    }
}

/// The generic compression variable.
const COMPRESSION_VAR: &str = "OTEL_EXPORTER_OTLP_COMPRESSION";

/// Run the child with `case` (an `http://` endpoint and a compression
/// setting), assert it refused its configuration naming `compression_var`
/// and `value` and saying to unset that variable, and assert the error does
/// not name `endpoint_var`: the compression setting is at fault, not the
/// endpoint.
fn assert_compression_refused(
    case: &[(&str, &str)],
    endpoint_var: &str,
    compression_var: &str,
    value: &str,
) {
    let value_part = format!("{compression_var} is set to `{value}`");
    let unset_part = format!("unset {compression_var}");
    let run = assert_refused_before_grpc_bind(case, &[&value_part, &unset_part]);
    assert!(
        !run.stderr.contains(endpoint_var),
        "{case:?}: the error is about the compression setting, not {endpoint_var}; stderr: {}",
        run.stderr
    );
}

/// A compression setting stops startup for a valid `http://` endpoint, for
/// each signal: here `OTEL_EXPORTER_OTLP_COMPRESSION=gzip`, a compression
/// this build does not include. The error names the compression variable and
/// its value, not the endpoint variable.
#[test]
fn live_edit_binary_with_otlp_compression_set_exits_with_a_config_error_naming_the_compression_variable(
) {
    for var in [TRACE_ENDPOINT_VAR, METRICS_ENDPOINT_VAR] {
        assert_compression_refused(
            &[(var, "http://127.0.0.1:1"), (COMPRESSION_VAR, "gzip")],
            var,
            COMPRESSION_VAR,
            "gzip",
        );
    }
}

/// `none` is refused too — this build accepts no compression setting at all
/// — and a signal's own compression variable is the one named when both it
/// and the generic one are set, as the exporter reads it first.
#[test]
fn live_edit_binary_with_otlp_compression_none_or_per_signal_exits_naming_the_variable_to_unset() {
    assert_compression_refused(
        &[
            (TRACE_ENDPOINT_VAR, "http://127.0.0.1:1"),
            (COMPRESSION_VAR, "none"),
        ],
        TRACE_ENDPOINT_VAR,
        COMPRESSION_VAR,
        "none",
    );
    for (endpoint_var, signal_var) in [
        (TRACE_ENDPOINT_VAR, "OTEL_EXPORTER_OTLP_TRACES_COMPRESSION"),
        (
            METRICS_ENDPOINT_VAR,
            "OTEL_EXPORTER_OTLP_METRICS_COMPRESSION",
        ),
    ] {
        assert_compression_refused(
            &[
                (endpoint_var, "http://127.0.0.1:1"),
                (signal_var, "none"),
                (COMPRESSION_VAR, "gzip"),
            ],
            endpoint_var,
            signal_var,
            "none",
        );
    }
}

/// An endpoint with userinfo and a query, standing in for one carrying
/// credentials.
const ENDPOINT_WITH_CREDENTIALS: &str = "http://otel-user:s3cret@127.0.0.1:1/?token=abc123";

/// Neither export startup line logs the endpoint, which may carry
/// credentials: with both endpoints set to a URL with userinfo and a query,
/// the child reaches the gRPC bind and neither line contains the URL or any
/// part of its credentials.
#[test]
fn live_edit_binary_otlp_export_startup_lines_never_log_the_endpoint() {
    let held = hold_port();
    let cgroup = pod_memory_fixture();
    let run = run_binary(&startup_env(
        &[
            (TRACE_ENDPOINT_VAR, ENDPOINT_WITH_CREDENTIALS),
            (METRICS_ENDPOINT_VAR, ENDPOINT_WITH_CREDENTIALS),
        ],
        held.port,
        cgroup.path(),
    ));
    assert_eq!(
        run.status.code(),
        Some(1),
        "the child exits at the gRPC bind; stdout: {}\nstderr: {}",
        run.stdout,
        run.stderr
    );
    let records = json_records_before_bind_failure(&run);
    for message in [TRACE_EXPORT_ENABLED_MESSAGE, METRICS_PUSH_ENABLED_MESSAGE] {
        let line = record_starting(&records, message, &run).to_string();
        for secret in ["s3cret", "abc123", "otel-user", ENDPOINT_WITH_CREDENTIALS] {
            assert!(
                !line.contains(secret),
                "{message:?} must not log the endpoint ({secret:?} found): {line}"
            );
        }
    }
}

/// T6 (d): `http://` endpoints with the same empty trust store start: the
/// child reaches the gRPC bind, and both export startup lines name the
/// plaintext transport. TLS, and so the trust store, is attached only for
/// `https://`.
#[test]
fn live_edit_binary_with_http_otlp_endpoints_starts_plaintext_without_reading_trust_roots() {
    let trust_store = empty_trust_store();
    let held = hold_port();
    let cgroup = pod_memory_fixture();
    let run = run_binary(&startup_env(
        &[
            (TRACE_ENDPOINT_VAR, "http://127.0.0.1:1"),
            (METRICS_ENDPOINT_VAR, "http://127.0.0.1:1"),
            (
                "SSL_CERT_FILE",
                trust_store.path().to_str().expect("UTF-8 temp path"),
            ),
        ],
        held.port,
        cgroup.path(),
    ));
    assert_eq!(
        run.status.code(),
        Some(1),
        "the child exits at the gRPC bind; stdout: {}\nstderr: {}",
        run.stdout,
        run.stderr
    );
    assert!(
        !run.stderr.contains("configuration error:"),
        "http:// endpoints are accepted whatever the trust store holds; stderr: {}",
        run.stderr
    );
    let records = json_records_before_bind_failure(&run);
    for message in [TRACE_EXPORT_ENABLED_MESSAGE, METRICS_PUSH_ENABLED_MESSAGE] {
        let line = record_starting(&records, message, &run);
        assert_eq!(line["level"], "INFO", "{line}");
        assert_eq!(line["fields"]["transport"], "plaintext", "{line}");
    }
}

/// The ERROR `live-edit scaler` logs when it is not running in a pod with a
/// mounted service-account token.
const SCALER_NOT_IN_CLUSTER_MESSAGE: &str = "live-edit scaler cannot reach the Kubernetes API";

/// Prefix of the INFO line `live-edit scaler` logs once its configuration is
/// accepted.
const SCALER_STARTING_MESSAGE_PREFIX: &str = "live-edit scaler starting";

/// A complete scaler configuration as the chart renders it, with JSON logs.
fn scaler_env() -> Vec<(String, String)> {
    [
        ("LIVE_EDIT_SCALER_NAMESPACE", "editing"),
        ("LIVE_EDIT_SCALER_DEPLOYMENT", "live-edit-service"),
        ("LIVE_EDIT_SCALER_SERVICE", "live-edit-service"),
        ("LIVE_EDIT_SCALER_MIN_REPLICAS", "3"),
        ("LIVE_EDIT_SCALER_MAX_REPLICAS", "10"),
        ("LIVE_EDIT_LOG_FORMAT", "json"),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// `live-edit scaler` with a scaler variable missing exits 1 with a
/// configuration error naming that variable, though no service variable is
/// set (the service would refuse its own configuration first, naming one of
/// its own variables). Each case drops a different one of the required
/// variables from an otherwise complete configuration, so the error tracks
/// the input.
#[test]
fn live_edit_scaler_mode_missing_a_scaler_variable_exits_1_naming_it() {
    let full = scaler_env();
    for missing in [
        "LIVE_EDIT_SCALER_NAMESPACE",
        "LIVE_EDIT_SCALER_DEPLOYMENT",
        "LIVE_EDIT_SCALER_SERVICE",
        "LIVE_EDIT_SCALER_MIN_REPLICAS",
        "LIVE_EDIT_SCALER_MAX_REPLICAS",
    ] {
        let env: Vec<(String, String)> =
            full.iter().filter(|(k, _)| k != missing).cloned().collect();
        let run = run_binary_with_args(&["scaler"], &env);
        assert_eq!(
            run.status.code(),
            Some(1),
            "{missing}: stdout: {}\nstderr: {}",
            run.stdout,
            run.stderr
        );
        assert!(
            run.stderr.contains("configuration error:") && run.stderr.contains(missing),
            "{missing}: stderr must carry a configuration error naming it; got: {}",
            run.stderr
        );
        assert!(
            !run.stdout.contains(BIND_FAILURE_MESSAGE)
                && !run.stdout.contains(BIND_SUCCESS_MESSAGE_PREFIX),
            "{missing}: the scaler must never reach the service's gRPC bind; stdout: {}",
            run.stdout
        );
    }
}

/// `live-edit scaler` refuses a trace endpoint that is not an `http://` or
/// `https://` URL, like the service: the trace pipeline is built before the
/// scaler dispatch, so the scaler exits 1 with a configuration error naming
/// the variable and never logs its startup line.
#[test]
fn live_edit_scaler_mode_with_an_unsupported_trace_endpoint_exits_1_naming_it() {
    let mut env = scaler_env();
    env.push((
        TRACE_ENDPOINT_VAR.to_string(),
        "grpc://collector.example.com:4317".to_string(),
    ));
    let run = run_binary_with_args(&["scaler"], &env);
    assert_eq!(
        run.status.code(),
        Some(1),
        "stdout: {}\nstderr: {}",
        run.stdout,
        run.stderr
    );
    assert!(
        run.stderr.contains("configuration error:") && run.stderr.contains(TRACE_ENDPOINT_VAR),
        "stderr must carry a configuration error naming {TRACE_ENDPOINT_VAR}; got: {}",
        run.stderr
    );
    assert!(
        !run.stdout.contains(SCALER_STARTING_MESSAGE_PREFIX),
        "the refusal must come before the scaler starts; stdout: {}",
        run.stdout
    );
}

/// A complete scaler configuration needs no service variable — not even
/// `LIVE_EDIT_VALKEY_URL` — and gets past configuration to the scaler's own
/// first dependency, the in-cluster Kubernetes environment. A test process
/// has none, so the child logs the startup INFO and then the not-in-cluster
/// ERROR, and exits 1.
#[test]
fn live_edit_scaler_mode_with_a_complete_configuration_needs_no_service_variable() {
    let env = scaler_env();
    assert!(
        env.iter()
            .all(|(k, _)| k.starts_with("LIVE_EDIT_SCALER_") || k == "LIVE_EDIT_LOG_FORMAT"),
        "fixture self-check: no service variable"
    );
    let run = run_binary_with_args(&["scaler"], &env);
    assert_eq!(
        run.status.code(),
        Some(1),
        "stdout: {}\nstderr: {}",
        run.stdout,
        run.stderr
    );
    assert!(
        !run.stderr.contains("configuration error:"),
        "the scaler configuration is complete; stderr: {}",
        run.stderr
    );
    let records = parse_records(&run.stdout);
    let starting = records
        .iter()
        .position(|r| r.level == "INFO" && r.message.starts_with(SCALER_STARTING_MESSAGE_PREFIX));
    let not_in_cluster = records
        .iter()
        .position(|r| r.level == "ERROR" && r.message == SCALER_NOT_IN_CLUSTER_MESSAGE);
    assert!(
        matches!((starting, not_in_cluster), (Some(s), Some(e)) if s < e),
        "the scaler must log its startup line, then fail at the Kubernetes environment; \
         got: {records:#?}"
    );
}
