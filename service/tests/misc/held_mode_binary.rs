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

//! The real `live-edit` binary started while the fleet is held (SDD SR-5):
//! held mode end to end, as a process, against the test store.
//!
//! Each test writes the hold marker under the `TestValkey` fixture's own
//! prefix and starts the binary (`CARGO_BIN_EXE_live-edit`, built with
//! `test-support`) with `LIVE_EDIT_TEST_ONLY_KEY_PREFIX` set to that prefix,
//! so the binary reads that marker and every key it builds is under the
//! prefix; the fixture's drop removes them, and no fleet-wide `le:` key is
//! touched on the shared store. The rest of the environment is cleared and
//! set to what reaches the hold decision with nothing else to talk to: JSON
//! logs, ephemeral ports, auth backends off through the test-only
//! unvalidated-token override, a local-disk backup store in a temporary
//! directory, and the pod memory limit from the 5 GiB cgroup fixture.
//!
//! - **Held mode serves, and SIGTERM ends it with 0**
//!   ([`live_edit_binary_held_at_startup_serves_held_mode_and_exits_zero_on_sigterm`]):
//!   readiness reaches `SERVING` over the wire, a `Ping` is refused
//!   `UNAVAILABLE` `SERVICE_HELD` naming the marker's incident, and SIGTERM
//!   ends the process with 0 within [`SIGTERM_EXIT_BOUND`], with no stop
//!   budget WARN: the signal reached held mode through the process shutdown
//!   request it claimed at its entry (with no handler, the default action
//!   would end the process by the signal, with no exit code) and the server
//!   stopped at once.
//! - **The marker removed, it exits 0**
//!   ([`live_edit_binary_held_at_startup_exits_zero_once_the_marker_is_removed`]):
//!   after one poll interval (set to [`POLL_INTERVAL`]) and the jitter its
//!   un-hold line reports, which is within the 5 s bound.
//!
//! The in-process tests (`tests/integration/dr/held_mode_test.rs`) reach
//! every branch of held mode; these pin what only the process shows: the
//! held-mode decision against a real store, the socket handoff from the
//! bootstrap server, the real process's signal listener reaching held mode,
//! and its exit codes.
//!
//! # Waits
//!
//! [`SIGTERM_EXIT_BOUND`] (8 s) is the one wall-clock assertion, a liveness
//! deadline far above the sub-second healthy exit; every other wait is a
//! deadline a healthy run never reaches. Per test, every wait at its full
//! budget:
//!
//! - startup: up to 60 s of [`STARTUP_BUDGET`] (the bind line and readiness,
//!   under one deadline), plus up to 10 s ([`RPC_BUDGET`]) for the last
//!   readiness probe, since the deadline is checked only after each probe;
//! - the refusal: 10 s ([`RPC_BUDGET`]);
//! - the exit: 20 s ([`UNHOLD_EXIT_DEADLINE`]; the SIGTERM test's is 8 s);
//! - the output drain: up to 2 × 5 s ([`OUTPUT_DRAIN_BUDGET`]), one per
//!   output reader, drained in turn.
//!
//! That is 60 + 10 + 10 + 20 + 10 = about 110 s, plus the 50 ms
//! ([`PROBE_INTERVAL`]) sleeps between probes and the `TestValkey` fixture's
//! setup and drop. It is inside nextest's 120 s kill (40 s × 3; no override
//! in `.config/nextest.toml` matches this binary) with roughly 10 s to spare.
//! A healthy run takes a few seconds, plus up to 5 s of drawn jitter in the
//! un-hold test.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;
use tonic::transport::Channel;
use tonic::Status;
use tonic_health::pb::health_check_response::ServingStatus;

use live_edit::config::key_prefix::TEST_ONLY_KEY_PREFIX_ENV;
use live_edit::dr::hold::UNHOLD_EXIT_JITTER_MAX;
use live_edit::grpc::hold::HELD_MODE_SERVER_STOP_BUDGET;
use live_edit::grpc::proto::live_service_client::LiveServiceClient;
use live_edit::grpc::proto::PingRequest;
use live_edit::storage::test_support::{HoldMarkerSeed, TestValkey};

use crate::common::refusal_detail::assert_service_held;
use crate::startup_log_binary::{pod_memory_fixture, LIVE_EDIT_BIN};

/// Deadline for the binary to log its gRPC bind and then answer readiness
/// `SERVING` in held mode. A healthy debug build gets there in a few seconds
/// (the Valkey connect, the snapshot-read self-test, the hold read).
const STARTUP_BUDGET: Duration = Duration::from_secs(60);

/// Deadline for one RPC, its connect included.
const RPC_BUDGET: Duration = Duration::from_secs(10);

/// The hold poll's interval (`LIVE_EDIT_DR_HOLD_POLL_INTERVAL_MS`): short,
/// so the un-hold exit is read within half a second of the marker's removal.
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Slack on the SIGTERM exit bound for the process's own exit: the trace
/// flush (a no-op: no exporter is configured) and the process teardown. A
/// healthy run exits well under 1 s after the signal.
const EXIT_SLACK: Duration = Duration::from_secs(3);

/// Bound on the exit after SIGTERM: held mode's server stop budget, which no
/// open call can outlast, and [`EXIT_SLACK`].
const SIGTERM_EXIT_BOUND: Duration =
    Duration::from_secs(HELD_MODE_SERVER_STOP_BUDGET.as_secs() + EXIT_SLACK.as_secs());

/// Hang deadline on the exit after the marker is removed: one poll interval
/// ([`POLL_INTERVAL`]) and the 5 s jitter bound, with the same slack for a
/// loaded runner as `EXIT_DEADLINE` in `tests/integration/dr/held_mode_test.rs`.
/// Not the "one poll interval + 5 s" claim itself: that rests on the poll
/// interval this test sets and on the jitter the binary logs drawing, which
/// the test checks against `UNHOLD_EXIT_JITTER_MAX`.
const UNHOLD_EXIT_DEADLINE: Duration = Duration::from_secs(20);

/// Deadline for the log readers to reach the end of the exited process's
/// output.
const OUTPUT_DRAIN_BUDGET: Duration = Duration::from_secs(5);

/// Interval between probes while waiting for the binary.
const PROBE_INTERVAL: Duration = Duration::from_millis(50);

/// The readiness service name the binary registers.
const READINESS: &str = "nvidia.omniverse.live.v1alpha.LiveService";

/// The incident the hold marker names.
const INCIDENT: &str = "20261015t2330-bin1";

/// Prefix of the INFO the binary logs once its gRPC socket is bound, with
/// the bound address as `addr`.
const BIND_LINE_PREFIX: &str = "gRPC listener bound";

/// The INFO held mode logs when a SIGTERM cancels the process shutdown
/// request it claimed at its entry.
const HELD_SIGTERM_LINE: &str = "received SIGTERM in held mode";

/// Prefix of the INFO the held-mode reactor logs before its un-hold exit,
/// with the jitter it drew as `jitter_ms`.
const UNHOLD_LINE_PREFIX: &str = "the disaster-recovery hold cleared";

/// Prefix of the WARN held mode logs when its server outlasts its stop
/// budget (`src/grpc/hold/held_mode.rs`, the stop-budget arm of
/// `run_held_mode`). The SIGTERM test asserts it is absent, so a reworded
/// WARN would make that check pass silently, checking nothing. Its twin,
/// `HELD_STOP_BUDGET_PREFIX` in `tests/integration/dr/held_mode_test.rs`, is
/// asserted present by the open-stream test, which fails when the WARN is
/// reworded: change both copies together.
const STOP_BUDGET_WARN_PREFIX: &str = "held mode: the gRPC server did not stop within its budget";

/// The `live-edit` binary running against `tv`'s store under `tv`'s prefix,
/// its output collected as it runs. Killed if dropped while running.
struct HeldBinary {
    child: Child,
    stdout: Arc<Mutex<Vec<String>>>,
    stderr: Arc<Mutex<String>>,
    readers: Vec<JoinHandle<()>>,
    _cgroup: tempfile::TempDir,
    _backup_store: tempfile::TempDir,
}

impl HeldBinary {
    /// Start the binary with a cleared environment, on `tv`'s store and
    /// prefix.
    fn spawn(tv: &TestValkey) -> Self {
        let cgroup = pod_memory_fixture();
        let backup_store = tempfile::tempdir().expect("create the backup-store directory");
        let valkey_url =
            std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".into());
        let poll_ms = POLL_INTERVAL.as_millis().to_string();
        let env: [(&str, &str); 13] = [
            ("LIVE_EDIT_LOG_FORMAT", "json"),
            ("LIVE_EDIT_PORT", "0"),
            ("LIVE_EDIT_METRICS_PORT", "0"),
            ("LIVE_EDIT_VALKEY_URL", valkey_url.as_str()),
            ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", "false"),
            ("LIVE_EDIT_AUTHZ_ENABLED", "false"),
            (
                "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
                "true",
            ),
            // A backup store, so the durability guard does not probe the
            // test store's AOF setting.
            ("LIVE_EDIT_CHECKPOINT_ENABLED", "true"),
            ("LIVE_EDIT_CHECKPOINT_ADAPTER", "localdisk"),
            (
                "LIVE_EDIT_CHECKPOINT_LOCAL_DISK_PATH",
                backup_store.path().to_str().expect("a UTF-8 path"),
            ),
            (
                "MEASUREMENT_POD_MEMORY_CGROUP_ROOT",
                cgroup.path().to_str().expect("a UTF-8 path"),
            ),
            (TEST_ONLY_KEY_PREFIX_ENV, tv.prefix()),
            ("LIVE_EDIT_DR_HOLD_POLL_INTERVAL_MS", poll_ms.as_str()),
        ];
        let mut child = Command::new(LIVE_EDIT_BIN)
            .env_clear()
            .envs(env)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .expect("spawn the live-edit binary");

        let stdout = Arc::new(Mutex::new(Vec::new()));
        let stderr = Arc::new(Mutex::new(String::new()));
        let out_pipe = child.stdout.take().expect("stdout is piped");
        let err_pipe = child.stderr.take().expect("stderr is piped");
        let out_lines = Arc::clone(&stdout);
        let err_text = Arc::clone(&stderr);
        let readers = vec![
            tokio::spawn(async move {
                let mut lines = BufReader::new(out_pipe).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    out_lines.lock().unwrap().push(line);
                }
            }),
            tokio::spawn(async move {
                let mut text = String::new();
                let _ = BufReader::new(err_pipe).read_to_string(&mut text).await;
                err_text.lock().unwrap().push_str(&text);
            }),
        ];
        Self {
            child,
            stdout,
            stderr,
            readers,
            _cgroup: cgroup,
            _backup_store: backup_store,
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

    /// The JSON log records whose message starts with `prefix`.
    fn records_starting(&self, prefix: &str) -> Vec<serde_json::Value> {
        self.stdout
            .lock()
            .unwrap()
            .iter()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|r| {
                r["fields"]["message"]
                    .as_str()
                    .is_some_and(|m| m.starts_with(prefix))
            })
            .collect()
    }

    /// The address to reach the binary's gRPC socket at, from its bind line.
    fn bound_addr(&self) -> Option<SocketAddr> {
        let line = self.records_starting(BIND_LINE_PREFIX).into_iter().next()?;
        let addr: SocketAddr = line["fields"]["addr"]
            .as_str()
            .unwrap_or_else(|| panic!("the bind line carries its address: {line}"))
            .parse()
            .unwrap_or_else(|e| panic!("the bind line's address parses ({e}): {line}"));
        Some(match addr.ip() {
            IpAddr::V4(ip) if ip.is_unspecified() => {
                SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), addr.port())
            }
            IpAddr::V6(ip) if ip.is_unspecified() => {
                SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), addr.port())
            }
            _ => addr,
        })
    }

    /// Panic with the binary's output if it has already exited.
    fn assert_running(&mut self, what: &str) {
        if let Some(status) = self.child.try_wait().expect("poll the child") {
            panic!(
                "{what}: the binary exited early ({status}); {}",
                self.output()
            );
        }
    }

    /// Wait, under [`STARTUP_BUDGET`], for the bind line and then for
    /// readiness `SERVING` over the wire; returns the address.
    async fn until_ready(&mut self) -> SocketAddr {
        let deadline = Instant::now() + STARTUP_BUDGET;
        let addr = loop {
            self.assert_running("before the gRPC bind");
            if let Some(addr) = self.bound_addr() {
                break addr;
            }
            assert!(
                Instant::now() < deadline,
                "no gRPC bind line within {STARTUP_BUDGET:?}; {}",
                self.output()
            );
            tokio::time::sleep(PROBE_INTERVAL).await;
        };
        loop {
            self.assert_running("before readiness SERVING");
            // A fresh connection each probe: the socket changes hands from the
            // bootstrap server to the held server between probes.
            if readiness(addr).await == Some(ServingStatus::Serving as i32) {
                return addr;
            }
            assert!(
                Instant::now() < deadline,
                "readiness did not reach SERVING within {STARTUP_BUDGET:?}; {}",
                self.output()
            );
            tokio::time::sleep(PROBE_INTERVAL).await;
        }
    }

    /// Send SIGTERM to the binary.
    fn sigterm(&self) {
        let pid = self.child.id().expect("the binary is running");
        let pid = libc::pid_t::try_from(pid).expect("the pid fits a pid_t");
        // SAFETY: `kill` takes plain integers and touches no memory of ours.
        let sent = unsafe { libc::kill(pid, libc::SIGTERM) };
        assert_eq!(sent, 0, "kill(SIGTERM) failed");
    }

    /// Wait for the binary to exit, within `bound` of `since`, and for its
    /// output to be read to the end. Returns its status.
    async fn exited_within(&mut self, since: Instant, bound: Duration, what: &str) -> ExitStatus {
        let left = bound.saturating_sub(since.elapsed());
        let status = match tokio::time::timeout(left, self.child.wait()).await {
            Ok(status) => status.expect("wait for the binary"),
            Err(_) => {
                let _ = self.child.kill().await;
                panic!(
                    "{what}: the binary did not exit within {bound:?}; {}",
                    self.output()
                );
            }
        };
        let elapsed = since.elapsed();
        assert!(
            elapsed <= bound,
            "{what}: the binary exited {elapsed:?} after, past {bound:?}; {}",
            self.output()
        );
        for reader in self.readers.drain(..) {
            tokio::time::timeout(OUTPUT_DRAIN_BUDGET, reader)
                .await
                .expect("the output readers reach the end of the output")
                .expect("an output reader");
        }
        status
    }
}

/// The readiness status the server at `addr` answers, or `None` if it
/// cannot be reached or does not answer within [`RPC_BUDGET`].
async fn readiness(addr: SocketAddr) -> Option<i32> {
    let probe = async {
        let channel = Channel::from_shared(format!("http://{addr}"))
            .expect("a valid endpoint")
            .connect()
            .await
            .ok()?;
        tonic_health::pb::health_client::HealthClient::new(channel)
            .check(tonic_health::pb::HealthCheckRequest {
                service: READINESS.to_owned(),
            })
            .await
            .ok()
            .map(|r| r.into_inner().status)
    };
    tokio::time::timeout(RPC_BUDGET, probe).await.ok().flatten()
}

/// A `Ping` with no credentials to the server at `addr`: its refusal.
async fn ping_refusal(addr: SocketAddr) -> Status {
    let call = async {
        let channel = Channel::from_shared(format!("http://{addr}"))
            .expect("a valid endpoint")
            .connect()
            .await
            .expect("connect to the held binary");
        LiveServiceClient::new(channel)
            .ping(PingRequest {
                client_timestamp_microseconds_since_epoch: 1,
            })
            .await
            .expect_err("a held binary refuses Ping")
    };
    tokio::time::timeout(RPC_BUDGET, call)
        .await
        .expect("the Ping answers within the deadline")
}

/// Write the hold marker naming [`INCIDENT`] under `tv`'s prefix, start the
/// binary, wait for readiness, and check one LiveService RPC is refused
/// `SERVICE_HELD` naming the incident: the binary is in held mode.
async fn start_held(tv: &TestValkey) -> HeldBinary {
    tv.write_hold_marker(&HoldMarkerSeed::default().with_incident(INCIDENT))
        .await;
    let mut binary = HeldBinary::spawn(tv);
    let addr = binary.until_ready().await;
    assert_service_held(&ping_refusal(addr).await, Some(INCIDENT));
    binary
}

/// A binary that starts while the fleet is held runs held mode: readiness
/// `SERVING`, every LiveService RPC (here `Ping`) refused `SERVICE_HELD`
/// naming the incident; and SIGTERM, which the binary as PID 1 would ignore
/// without a handler, ends it with 0 promptly: its server stops without
/// running out its stop budget (no budget WARN), and the process ends within
/// that budget and [`EXIT_SLACK`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_edit_binary_held_at_startup_serves_held_mode_and_exits_zero_on_sigterm() {
    let tv = TestValkey::setup().await;
    let mut binary = start_held(&tv).await;

    let signalled = Instant::now();
    binary.sigterm();
    let status = binary
        .exited_within(signalled, SIGTERM_EXIT_BOUND, "SIGTERM")
        .await;

    assert_eq!(
        status.code(),
        Some(0),
        "a held binary ends with 0 on SIGTERM ({status}); {}",
        binary.output()
    );
    assert_eq!(
        binary.records_starting(HELD_SIGTERM_LINE).len(),
        1,
        "the SIGTERM reached held mode through the shutdown request it claimed; {}",
        binary.output()
    );
    assert!(
        binary.records_starting(STOP_BUDGET_WARN_PREFIX).is_empty(),
        "the held server stopped on the signal, not at the end of its stop budget; {}",
        binary.output()
    );
}

/// A binary that starts while the fleet is held exits 0 once the marker is
/// removed, after one poll interval and the un-hold jitter, so the kubelet
/// restarts it into normal service. The poll interval is the
/// [`POLL_INTERVAL`] this test sets; the jitter is the one the binary logs
/// drawing, which must be within `UNHOLD_EXIT_JITTER_MAX` (5 s). The exit's
/// own deadline, [`UNHOLD_EXIT_DEADLINE`], only bounds a hang.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_edit_binary_held_at_startup_exits_zero_once_the_marker_is_removed() {
    let tv = TestValkey::setup().await;
    let mut binary = start_held(&tv).await;

    assert!(tv.remove_hold_marker().await, "fixture: the marker existed");
    let removed = Instant::now();
    let status = binary
        .exited_within(removed, UNHOLD_EXIT_DEADLINE, "the marker removed")
        .await;

    assert_eq!(
        status.code(),
        Some(0),
        "an un-held binary exits 0 ({status}); {}",
        binary.output()
    );
    let unholds = binary.records_starting(UNHOLD_LINE_PREFIX);
    assert_eq!(
        unholds.len(),
        1,
        "the exit is the held-mode reactor's un-hold; {}",
        binary.output()
    );
    let jitter_ms = unholds[0]["fields"]["jitter_ms"]
        .as_u64()
        .unwrap_or_else(|| panic!("the un-hold line carries its jitter: {}", unholds[0]));
    assert!(
        Duration::from_millis(jitter_ms) <= UNHOLD_EXIT_JITTER_MAX,
        "the un-hold exit waited {jitter_ms} ms of jitter, past the {UNHOLD_EXIT_JITTER_MAX:?} \
         bound; {}",
        binary.output()
    );
}
