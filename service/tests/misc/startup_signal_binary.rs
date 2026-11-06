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

//! The real `live-edit` binary stopped by a signal while it is still starting
//! (SDD §5.5): `SIGTERM`, and separately `SIGINT`, sent while the binary is
//! blocked in its first dependency step, the Valkey connect, stop it within
//! seconds with exit code 0, through the startup stop.
//!
//! The binary's Valkey is a local listener that accepts every connection and
//! never answers ([`SilentValkey`]), and the connect timeout is 60 s, so
//! without a working startup stop the child sits in the connect for a minute
//! and then exits 1 with `failed to connect to Valkey`. The test waits until
//! the binary has bound its gRPC socket and opened a connection to that
//! listener — it is then inside the connect — and sends the signal. What it
//! asserts tells the failure modes apart:
//!
//! - the exit within [`STOP_EXIT_BOUND`] and no `failed to connect to Valkey`
//!   line: a handler that is installed but not acted on until the connect
//!   gives up (a watcher spawned after the connect, or none) misses both;
//! - exit code 0: a process with no handler at all dies by the signal, with no
//!   exit code (it is not PID 1 here, so the default action applies);
//! - exactly one startup-stop line naming the signal, with its
//!   `startup_elapsed_ms`: the stop ran once, and for the signal sent.
//!
//! No Valkey is needed. The cgroup fixture gives the child a 5 GiB pod memory
//! limit, as in `startup_log_binary.rs`.
//!
//! # Waits
//!
//! [`STOP_EXIT_BOUND`] (14 s) is the one wall-clock assertion, a liveness
//! deadline: the startup stop is bounded by the 5 s bootstrap handoff budget
//! and the 6 s flush cap, and a healthy one exits well under a second after
//! the signal. Every other wait is a deadline a healthy run never reaches:
//! up to 60 s ([`STARTUP_BUDGET`]) for the bind line and the connection, then
//! the exit bound, then up to 2 × 5 s ([`OUTPUT_DRAIN_BUDGET`]) for the output
//! readers, about 84 s in all, inside nextest's 120 s kill.

use std::io::{BufRead, BufReader, Read};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use live_edit::grpc::transport::POST_BUDGET_FLUSH_BOUND;
use live_edit::observability::startup::{SIGINT, SIGTERM};

use crate::startup_log_binary::{pod_memory_fixture, AUTH_ENV, LIVE_EDIT_BIN};

/// Deadline for the binary to log its gRPC bind and open its first
/// connection to [`SilentValkey`]. A healthy debug build gets there in about
/// a second.
const STARTUP_BUDGET: Duration = Duration::from_secs(60);

/// The bootstrap server's stop budget at a stop during startup
/// (`BOOTSTRAP_HANDOFF_BUDGET` in `main.rs`).
const BOOTSTRAP_STOP_BUDGET: Duration = Duration::from_secs(5);

/// Slack on [`STOP_EXIT_BOUND`] for the process's own exit on a loaded
/// runner.
const EXIT_SLACK: Duration = Duration::from_secs(3);

/// Bound on the exit after the signal: the startup stop's bootstrap server
/// stop and its final flush, both capped, and [`EXIT_SLACK`]. Far below the
/// 60 s connect timeout the child would otherwise sit out.
const STOP_EXIT_BOUND: Duration = Duration::from_secs(
    BOOTSTRAP_STOP_BUDGET.as_secs() + POST_BUDGET_FLUSH_BOUND.as_secs() + EXIT_SLACK.as_secs(),
);

/// The Valkey connect timeout the child is given: far past
/// [`STOP_EXIT_BOUND`], so only the startup stop can end the child in time.
const VALKEY_CONNECT_TIMEOUT_SECS: &str = "60";

/// Deadline for the output readers to reach the end of the exited process's
/// output.
const OUTPUT_DRAIN_BUDGET: Duration = Duration::from_secs(5);

/// Interval between probes while waiting for the child.
const PROBE_INTERVAL: Duration = Duration::from_millis(20);

/// Prefix of the INFO the binary logs once its gRPC socket is bound.
const BIND_LINE_PREFIX: &str = "gRPC listener bound";

/// The ERROR the binary logs when its Valkey connect fails.
const CONNECT_FAILED_LINE: &str = "failed to connect to Valkey";

/// The startup stop's INFO for `signal`.
fn startup_stop_line(signal: &str) -> String {
    format!("received {signal} during startup; stopping before serving")
}

/// A Valkey endpoint that accepts every connection and never answers, so a
/// client connecting to it waits out its connect timeout. Counts the
/// connections it accepted, and holds them open, for the test's lifetime.
struct SilentValkey {
    port: u16,
    accepted: Arc<Mutex<Vec<TcpStream>>>,
}

impl SilentValkey {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind the silent Valkey");
        let port = listener.local_addr().expect("its address").port();
        let accepted = Arc::new(Mutex::new(Vec::new()));
        let held = Arc::clone(&accepted);
        // Lives until the test process exits: nextest runs each test in a
        // process of its own.
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                held.lock().unwrap().push(stream);
            }
        });
        Self { port, accepted }
    }

    fn url(&self) -> String {
        format!("redis://127.0.0.1:{}", self.port)
    }

    fn connections(&self) -> usize {
        self.accepted.lock().unwrap().len()
    }
}

/// The binary, starting against a [`SilentValkey`], its output collected as
/// it runs. Killed if dropped while running.
struct StartingBinary {
    child: Child,
    stdout: Arc<Mutex<Vec<String>>>,
    stderr: Arc<Mutex<String>>,
    readers: Vec<JoinHandle<()>>,
    _cgroup: tempfile::TempDir,
}

impl Drop for StartingBinary {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

impl StartingBinary {
    /// Start the binary with a cleared environment: JSON logs, ephemeral
    /// gRPC and metrics ports, the auth endpoints (never reached), and
    /// `valkey` as its store with a 60 s connect timeout.
    fn spawn(valkey: &SilentValkey) -> Self {
        let cgroup = pod_memory_fixture();
        let valkey_url = valkey.url();
        let fixed = [
            ("LIVE_EDIT_LOG_FORMAT", "json"),
            ("LIVE_EDIT_PORT", "0"),
            ("LIVE_EDIT_METRICS_PORT", "0"),
            ("LIVE_EDIT_VALKEY_URL", valkey_url.as_str()),
            (
                "LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS",
                VALKEY_CONNECT_TIMEOUT_SECS,
            ),
            (
                "MEASUREMENT_POD_MEMORY_CGROUP_ROOT",
                cgroup.path().to_str().expect("a UTF-8 path"),
            ),
        ];
        let mut child = Command::new(LIVE_EDIT_BIN)
            .env_clear()
            .envs(AUTH_ENV.iter().chain(fixed.iter()).copied())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn the live-edit binary");

        let stdout = Arc::new(Mutex::new(Vec::new()));
        let stderr = Arc::new(Mutex::new(String::new()));
        let out_pipe = child.stdout.take().expect("stdout is piped");
        let mut err_pipe = child.stderr.take().expect("stderr is piped");
        let out_lines = Arc::clone(&stdout);
        let err_text = Arc::clone(&stderr);
        let readers = vec![
            std::thread::spawn(move || {
                for line in BufReader::new(out_pipe).lines().map_while(Result::ok) {
                    out_lines.lock().unwrap().push(line);
                }
            }),
            std::thread::spawn(move || {
                let mut text = String::new();
                let _ = err_pipe.read_to_string(&mut text);
                err_text.lock().unwrap().push_str(&text);
            }),
        ];
        Self {
            child,
            stdout,
            stderr,
            readers,
            _cgroup: cgroup,
        }
    }

    /// Everything the binary has written so far, for failure messages.
    fn output(&self) -> String {
        format!(
            "stdout:\n{}\nstderr:\n{}",
            self.stdout.lock().unwrap().join("\n"),
            self.stderr.lock().unwrap()
        )
    }

    /// Every JSON log record the binary has written so far.
    fn records(&self) -> Vec<serde_json::Value> {
        self.stdout
            .lock()
            .unwrap()
            .iter()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .collect()
    }

    /// The records whose message starts with `prefix`.
    fn records_starting(&self, prefix: &str) -> Vec<serde_json::Value> {
        self.records()
            .into_iter()
            .filter(|r| {
                r["fields"]["message"]
                    .as_str()
                    .is_some_and(|m| m.starts_with(prefix))
            })
            .collect()
    }

    /// Wait, under [`STARTUP_BUDGET`], until the binary has bound its gRPC
    /// socket and connected to `valkey`: it is then blocked in the Valkey
    /// connect, which `valkey` never answers.
    fn until_in_the_valkey_connect(&mut self, valkey: &SilentValkey) {
        let deadline = Instant::now() + STARTUP_BUDGET;
        loop {
            if let Some(status) = self.child.try_wait().expect("poll the child") {
                panic!(
                    "the binary exited before reaching the Valkey connect ({status}); {}",
                    self.output()
                );
            }
            if !self.records_starting(BIND_LINE_PREFIX).is_empty() && valkey.connections() > 0 {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "the binary did not bind and reach the Valkey connect within {STARTUP_BUDGET:?}; \
                 {}",
                self.output()
            );
            std::thread::sleep(PROBE_INTERVAL);
        }
    }

    /// Send `signal` to the binary.
    fn signal(&self, signal: libc::c_int) {
        let pid = libc::pid_t::try_from(self.child.id()).expect("the pid fits a pid_t");
        // SAFETY: `kill` takes plain integers and touches no memory of ours.
        let sent = unsafe { libc::kill(pid, signal) };
        assert_eq!(sent, 0, "kill({signal}) failed");
    }

    /// Wait for the binary to exit within `bound` of `since`, then for its
    /// output to be read to the end. Returns its status.
    fn exited_within(&mut self, since: Instant, bound: Duration) -> ExitStatus {
        let status = loop {
            if let Some(status) = self.child.try_wait().expect("poll the child") {
                break status;
            }
            if since.elapsed() > bound {
                let _ = self.child.kill();
                let _ = self.child.wait();
                panic!(
                    "the binary did not exit within {bound:?} of the signal; {}",
                    self.output()
                );
            }
            std::thread::sleep(PROBE_INTERVAL);
        };
        let deadline = Instant::now() + OUTPUT_DRAIN_BUDGET * 2;
        for reader in self.readers.drain(..) {
            while !reader.is_finished() {
                assert!(
                    Instant::now() < deadline,
                    "the output readers reach the end of the output"
                );
                std::thread::sleep(PROBE_INTERVAL);
            }
            reader.join().expect("an output reader");
        }
        status
    }
}

/// Start the binary against a silent Valkey, wait until it is blocked in
/// the connect, send `signal`, and assert the startup stop ended it: exit 0
/// within [`STOP_EXIT_BOUND`], exactly one startup-stop line naming the
/// signal (with the time since the process started), and no connect
/// failure.
fn assert_a_signal_during_the_valkey_connect_stops_startup(signal: libc::c_int, name: &str) {
    let valkey = SilentValkey::start();
    let mut binary = StartingBinary::spawn(&valkey);
    binary.until_in_the_valkey_connect(&valkey);

    let sent = Instant::now();
    binary.signal(signal);
    let status = binary.exited_within(sent, STOP_EXIT_BOUND);

    assert_eq!(
        status.code(),
        Some(0),
        "a {name} during startup ends the process with 0 ({status}); {}",
        binary.output()
    );
    let stops = binary.records_starting(&startup_stop_line(name));
    assert_eq!(
        stops.len(),
        1,
        "exactly one startup-stop line; {}",
        binary.output()
    );
    let stop = &stops[0];
    assert_eq!(stop["level"], "INFO", "{stop}");
    assert_eq!(stop["fields"]["signal"], name, "{stop}");
    assert!(
        stop["fields"]["startup_elapsed_ms"].is_u64(),
        "the line carries the time since the process started: {stop}"
    );
    assert!(
        binary.records_starting(CONNECT_FAILED_LINE).is_empty(),
        "the stop came before the connect gave up; {}",
        binary.output()
    );
}

/// `SIGTERM` while the binary is blocked in the Valkey connect stops it
/// within seconds, with 0, through the startup stop.
#[test]
fn live_edit_binary_sigterm_during_the_valkey_connect_stops_startup_with_zero() {
    assert_a_signal_during_the_valkey_connect_stops_startup(libc::SIGTERM, SIGTERM);
}

/// `SIGINT` while the binary is blocked in the Valkey connect stops it the
/// same way: its handler is registered at once too, not on first poll.
#[test]
fn live_edit_binary_sigint_during_the_valkey_connect_stops_startup_with_zero() {
    assert_a_signal_during_the_valkey_connect_stops_startup(libc::SIGINT, SIGINT);
}
