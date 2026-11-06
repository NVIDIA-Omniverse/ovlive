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

//! The real `live-edit` binary as a disaster-recovery Job (`live-edit dr …`;
//! SDD SR-8.1, SR-9.1): what only the process shows, the dispatch in
//! `main.rs`, the command-line refusal before anything connects, and the exit
//! code a run outside Kubernetes reaches. The frame's every branch is
//! `tests/integration/dr/job_frame_test.rs`'s, in process.
//!
//! - **A refused command line exits 2 before connecting**
//!   ([`live_edit_dr_refuses_a_bad_command_line_with_exit_2_before_connecting`]):
//!   a `--fleet-*` flag on `restore`, and `accept-loss` without or with the
//!   wrong `--confirm`. The Valkey URL names a port nothing listens on, so a
//!   binary that connected first would end in its connect failure (exit 1)
//!   instead; stderr carries the refusal and the usage text. So is
//!   `accept-loss`'s second confirm alone, and a confirm given twice
//!   ([`live_edit_dr_accept_loss_refuses_the_second_confirm_alone_or_a_repeated_confirm_with_exit_2`]).
//! - **A refused configuration exits 2 in `dr` mode and 1 when serving**
//!   ([`live_edit_dr_exits_2_on_a_refused_configuration_where_the_service_exits_1`];
//!   for the logging settings read before it,
//!   [`live_edit_dr_exits_2_on_a_refused_log_format_where_the_service_exits_1`];
//!   and for the test-only key prefix read after it, in a `test-support`
//!   build,
//!   [`live_edit_dr_exits_2_on_a_refused_key_prefix_where_the_service_exits_1`]):
//!   the same refused setting, so the Job's pod failure policy (the chart's
//!   disaster-recovery Job templates) fails it at once rather than retrying a
//!   refusal no retry can clear.
//! - **A pod memory limit below the configuration's minimum exits 2, naming
//!   the Job's own limit setting**
//!   ([`live_edit_dr_below_the_pod_memory_minimum_exits_2_naming_dr_resources`]):
//!   the Job's refusal tells the operator to raise
//!   `dr.resources.limits.memory`, the chart value its limit comes from, where
//!   the service's names `resources.limits.memory`; at the minimum the Job
//!   gets past the plan.
//! - **An OTLP setting refused exits 2, an exporter not built exits 1**
//!   ([`live_edit_dr_exits_2_on_a_refused_otlp_setting_and_1_on_an_exporter_it_could_not_build`]):
//!   a trace or metrics endpoint refused on its value exits 2; one whose
//!   `https://` exporter cannot be built (no readable trust roots) exits 1,
//!   since a retry may build it.
//! - **A held store past its quiet period reaches the Job's body**
//!   ([`live_edit_dr_restore_on_a_store_held_past_its_quiet_period_reaches_the_body_and_unholds`]):
//!   run with a cleared environment, so outside Kubernetes, the binary takes
//!   its lease, opens its write permit and runs the restore body, which
//!   writes nothing on the empty store, recounts and un-holds, exit 0; the
//!   lease is released and the hold marker is deleted. So does the
//!   accept-loss body, run with both confirms in the reverse order
//!   ([`live_edit_dr_accept_loss_on_an_empty_held_store_reaches_the_body_and_unholds`]),
//!   which discards nothing on the empty store; its summaries carry the
//!   second confirm's clause, and a second run with the first confirm alone
//!   carries none. On linux-gnu its start line says the binary installed
//!   the allocator purge its value removal runs.
//!
//! The held-store tests write the marker under the `TestValkey` fixture's
//! prefix and run the binary (built with `test-support`) with
//! `LIVE_EDIT_TEST_ONLY_KEY_PREFIX` set to it, as
//! `tests/misc/held_mode_binary.rs` does, so no fleet-wide `le:` key is
//! touched on the shared store.
//!
//! # Waits
//!
//! Each run is bounded by [`RUN_BUDGET`] (25 s), a hang bound only. The
//! refusal test runs the binary four times, so at every budget's full value
//! it waits 100 s, inside nextest's 120 s kill; the second-confirm refusal
//! test twice (50 s), and so does the accept-loss held-store test (50 s);
//! the three configuration tests run it three times each (75 s), and the
//! OTLP and memory-limit tests four times each (100 s). A healthy refusal exits
//! within a second; a healthy held run takes a few seconds at most (the
//! connect, two marker reads, the lease, one `TIME`).

use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;

use live_edit::dr::job::USAGE;

use crate::startup_log_binary::{pod_memory_fixture, LIVE_EDIT_BIN};

/// Bound on one run of the binary. Reached only if it hangs.
const RUN_BUDGET: Duration = Duration::from_secs(25);

/// What one run left behind.
struct JobRun {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

impl JobRun {
    /// The JSON log records whose message is exactly `message`.
    fn records(&self, message: &str) -> Vec<serde_json::Value> {
        self.stdout
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|r| r["fields"]["message"].as_str() == Some(message))
            .collect()
    }

    fn output(&self) -> String {
        format!("stdout:\n{}\nstderr:\n{}", self.stdout, self.stderr)
    }
}

/// The environment `Config::from_env` loads with and nothing else talks to:
/// JSON logs, ephemeral ports, auth backends off through the test-only
/// unvalidated-token override, no backup store (the default `disabled`
/// adapter), and the pod memory limit from the cgroup fixture at
/// `cgroup_root`. `valkey_url` and `extra` complete it.
fn job_env(
    valkey_url: &str,
    cgroup_root: &std::path::Path,
    extra: &[(&str, &str)],
) -> Vec<(String, String)> {
    let fixed = [
        ("LIVE_EDIT_LOG_FORMAT", "json"),
        ("LIVE_EDIT_PORT", "0"),
        ("LIVE_EDIT_METRICS_PORT", "0"),
        ("LIVE_EDIT_VALKEY_URL", valkey_url),
        ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", "false"),
        ("LIVE_EDIT_AUTHZ_ENABLED", "false"),
        (
            "LIVE_EDIT_TEST_ONLY_DANGEROUSLY_ALLOW_UNVALIDATED_TOKENS",
            "true",
        ),
        (
            "MEASUREMENT_POD_MEMORY_CGROUP_ROOT",
            cgroup_root.to_str().expect("a UTF-8 path"),
        ),
    ];
    fixed
        .iter()
        .chain(extra)
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

/// Run `live-edit <args>` with exactly `env` (nothing inherited, so no
/// `KUBERNETES_SERVICE_HOST`), to its exit, within [`RUN_BUDGET`].
async fn run_binary(args: &[&str], env: &[(String, String)]) -> JobRun {
    let child = Command::new(LIVE_EDIT_BIN)
        .args(args)
        .env_clear()
        .envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn the live-edit binary");
    let output = tokio::time::timeout(RUN_BUDGET, child.wait_with_output())
        .await
        .unwrap_or_else(|_| panic!("live-edit {args:?} did not exit within {RUN_BUDGET:?}"))
        .expect("wait for the live-edit binary");
    JobRun {
        code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// A command line the Job refuses ends the process with exit 2 before it
/// connects to anything: a `--fleet-*` flag on `restore` (the Job reads no
/// Kubernetes object, so it takes none), and `accept-loss` without the exact
/// `--confirm discard-lost-data`. stderr is `live-edit dr: <refusal>` and the
/// usage text, which names no `--fleet-*` flag. The Valkey URL names a port
/// nothing listens on, so a binary that connected before parsing would end in
/// its connect failure, exit 1, instead.
#[tokio::test]
async fn live_edit_dr_refuses_a_bad_command_line_with_exit_2_before_connecting() {
    let cgroup = pod_memory_fixture();
    let env = job_env(
        "redis://127.0.0.1:1",
        cgroup.path(),
        &[("LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS", "1")],
    );
    for (args, refusal) in [
        (
            &["dr", "restore", "inc-1", "--fleet-namespace", "ns"][..],
            "unexpected argument \"--fleet-namespace\"",
        ),
        (
            &["dr", "restore", "inc-1", "--fleet-deployment=live-edit"][..],
            "unexpected argument \"--fleet-deployment=live-edit\"",
        ),
        (
            &["dr", "accept-loss", "inc-1"][..],
            "dr accept-loss discards data that cannot be restored and runs only with --confirm \
             discard-lost-data",
        ),
        (
            &["dr", "accept-loss", "inc-1", "--confirm", "yes"][..],
            "dr accept-loss runs only with --confirm discard-lost-data, not --confirm \"yes\"",
        ),
    ] {
        let run = run_binary(args, &env).await;
        assert_eq!(
            run.code,
            Some(2),
            "{args:?}: a refused command line exits 2; {}",
            run.output()
        );
        assert_eq!(
            run.stderr,
            format!("live-edit dr: {refusal}\n{USAGE}\n"),
            "{args:?}: stderr names the refusal and prints the usage"
        );
        assert!(
            !USAGE.contains("--fleet"),
            "the usage text names no --fleet-* flag"
        );
        assert!(
            run.records("live-edit dr restore starting").is_empty()
                && run.records("live-edit dr accept-loss starting").is_empty(),
            "{args:?}: the Job never starts; {}",
            run.output()
        );
    }
}

/// Run `restore`, `accept-loss` and the service with `refused` set, a
/// setting `main` refuses before it connects: each DR Job exits 2 and the
/// service 1, each with exactly `refusal` on stderr and nothing started.
/// The Valkey URL names a port nothing listens on, so a binary that got past
/// the configuration would end in its connect failure, with no
/// configuration error, instead. Three runs.
async fn assert_refused_with_2_in_dr_mode_and_1_serving(refused: (&str, &str), refusal: &str) {
    let cgroup = pod_memory_fixture();
    let env = job_env(
        "redis://127.0.0.1:1",
        cgroup.path(),
        &[("LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS", "1"), refused],
    );
    for (args, code) in [
        (&["dr", "restore", "inc-1"][..], 2),
        (
            &[
                "dr",
                "accept-loss",
                "inc-1",
                "--confirm",
                "discard-lost-data",
            ][..],
            2,
        ),
        (&[][..], 1),
    ] {
        let run = run_binary(args, &env).await;
        assert_eq!(
            run.code,
            Some(code),
            "{args:?}: a refused {} exits {code}; {}",
            refused.0,
            run.output()
        );
        assert_eq!(
            run.stderr, refusal,
            "{args:?}: stderr carries the configuration error alone"
        );
        assert!(
            run.records("live-edit dr restore starting").is_empty()
                && run.records("live-edit dr accept-loss starting").is_empty()
                && run.records("live-edit starting").is_empty(),
            "{args:?}: nothing starts; {}",
            run.output()
        );
    }
}

/// A configuration `Config::from_env` refuses ends a DR Job with exit 2, as a
/// refused command line does, and the service with exit 1: the same refused
/// value (`LIVE_EDIT_DR_RESTORE_PARALLELISM=0`) in both modes, with the same
/// `configuration error:` line on stderr.
#[tokio::test]
async fn live_edit_dr_exits_2_on_a_refused_configuration_where_the_service_exits_1() {
    assert_refused_with_2_in_dr_mode_and_1_serving(
        ("LIVE_EDIT_DR_RESTORE_PARALLELISM", "0"),
        "configuration error: invalid value for LIVE_EDIT_DR_RESTORE_PARALLELISM: must be \
         greater than 0 (the restore Job could restore no layer at all)\n",
    )
    .await;
}

/// A logging setting refused before the configuration loads (the tracing
/// settings `main` reads first, so a refusal there is reported before any
/// subscriber exists) ends a DR Job with exit 2 and the service with exit 1,
/// as a refused configuration does: `LIVE_EDIT_LOG_FORMAT=xml`.
#[tokio::test]
async fn live_edit_dr_exits_2_on_a_refused_log_format_where_the_service_exits_1() {
    assert_refused_with_2_in_dr_mode_and_1_serving(
        ("LIVE_EDIT_LOG_FORMAT", "xml"),
        "configuration error: invalid value for LIVE_EDIT_LOG_FORMAT: invalid \
         LIVE_EDIT_LOG_FORMAT value \"xml\": expected one of json, pretty (case-insensitive)\n",
    )
    .await;
}

/// The test-only key prefix, read once the configuration has loaded, refused
/// (`LIVE_EDIT_TEST_ONLY_KEY_PREFIX` empty) ends a DR Job with exit 2 and the
/// service with exit 1, as a refused configuration does. Only a
/// `test-support` build reads the variable, so only such a build runs this.
#[cfg(feature = "test-support")]
#[tokio::test]
async fn live_edit_dr_exits_2_on_a_refused_key_prefix_where_the_service_exits_1() {
    assert_refused_with_2_in_dr_mode_and_1_serving(
        ("LIVE_EDIT_TEST_ONLY_KEY_PREFIX", ""),
        "configuration error: invalid value for LIVE_EDIT_TEST_ONLY_KEY_PREFIX: it is empty; \
         unset it to build keys under the production prefix\n",
    )
    .await;
}

/// An OTLP setting a DR Job refuses on its value alone ends it with exit 2;
/// an OTLP exporter it could not build for a valid endpoint (an `https://`
/// collector with no readable trust roots, which a sidecar may not have
/// written yet) ends it with exit 1, so Kubernetes retries it. Both for the
/// trace endpoint, set up before the configuration loads, and for the
/// metrics endpoint, set up once the Job has parsed its command line; each
/// with the `configuration error:` line naming the variable. The Valkey URL
/// names a port nothing listens on, so a binary that got past the refusal
/// would end in its connect failure instead.
#[tokio::test]
async fn live_edit_dr_exits_2_on_a_refused_otlp_setting_and_1_on_an_exporter_it_could_not_build() {
    const TRACE: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";
    const METRICS: &str = "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT";
    let cgroup = pod_memory_fixture();
    let trust_store = tempfile::NamedTempFile::new().expect("create the empty trust store");
    let trust_store = trust_store.path().to_str().expect("a UTF-8 path");
    for (var, endpoint, code, cause) in [
        (TRACE, "grpc://collector.example.com:4317", 2, "scheme"),
        (METRICS, "grpc://collector.example.com:4317", 2, "scheme"),
        (
            TRACE,
            "https://collector.example.com:4317",
            1,
            "no native certs found",
        ),
        (
            METRICS,
            "https://collector.example.com:4317",
            1,
            "no native certs found",
        ),
    ] {
        let env = job_env(
            "redis://127.0.0.1:1",
            cgroup.path(),
            &[
                ("LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS", "1"),
                (var, endpoint),
                ("SSL_CERT_FILE", trust_store),
            ],
        );
        let run = run_binary(&["dr", "restore", "inc-1"], &env).await;
        assert_eq!(
            run.code,
            Some(code),
            "{var}={endpoint}: exits {code}; {}",
            run.output()
        );
        assert!(
            run.stderr
                .starts_with(&format!("configuration error: {var}"))
                && run.stderr.contains(cause),
            "{var}={endpoint}: stderr names the variable and the cause ({cause:?}); {}",
            run.output()
        );
        assert!(
            !run.stdout.contains("failed to connect to Valkey"),
            "{var}={endpoint}: refused before connecting; {}",
            run.output()
        );
    }
}

/// A DR Job whose pod memory limit is below its configuration's minimum ends
/// with exit 2 before it connects, and its refusal names
/// `dr.resources.limits.memory`, the chart value a Job's limit comes from:
/// raising the service's `resources.limits.memory` would not reach the Job.
/// The service with the same limit exits 1, its refusal the Job's byte for
/// byte but naming `resources.limits.memory`. With a backup store (the
/// `inmemory` adapter, checkpointing and the tail on) the margin sets the
/// minimum, 3,282 MiB whatever the runner's CPU count (the memory plan's
/// `BINARY_DEFAULTS` table), so 3,281 MiB is refused on every runner; at
/// 3,282 MiB the Job gets past the plan and ends in its connect failure,
/// exit 1, instead, which pins the minimum the refusal names. The Valkey URL
/// names a port nothing listens on. Only a `test-support` build reads the
/// cgroup fixture. Four runs.
#[cfg(feature = "test-support")]
#[tokio::test]
async fn live_edit_dr_below_the_pod_memory_minimum_exits_2_naming_dr_resources() {
    use crate::startup_log_binary::pod_memory_fixture_with;

    const MINIMUM_MIB: u64 = 3_282;
    const DR_SETTING: &str = "dr.resources.limits.memory";
    const SERVER_SETTING: &str = "resources.limits.memory";
    let backup_store = [
        ("LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS", "1"),
        ("LIVE_EDIT_CHECKPOINT_ENABLED", "true"),
        ("LIVE_EDIT_CHECKPOINT_ADAPTER", "inmemory"),
        ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", "true"),
    ];
    let below = pod_memory_fixture_with(&format!("{}\n", (MINIMUM_MIB - 1) << 20));
    let env = job_env("redis://127.0.0.1:1", below.path(), &backup_store);
    let accept_loss = [
        "dr",
        "accept-loss",
        "inc-1",
        "--confirm",
        "discard-lost-data",
    ];

    let restore = run_binary(&["dr", "restore", "inc-1"], &env).await;
    assert_eq!(
        restore.code,
        Some(2),
        "a DR Job below the minimum exits 2; {}",
        restore.output()
    );
    for part in [
        "configuration error: this pod's memory limit, 3,281 MiB (cgroup v2), is below the 3,282 \
         MiB this configuration needs, 1 MiB short;",
        "raise dr.resources.limits.memory to at least 3,282 MiB",
    ] {
        assert!(
            restore.stderr.contains(part),
            "{part:?} on stderr; {}",
            restore.output()
        );
    }
    assert!(
        !restore
            .stderr
            .replace(DR_SETTING, "")
            .contains(SERVER_SETTING),
        "the Job's refusal never names the service's setting; {}",
        restore.output()
    );
    let accept = run_binary(&accept_loss, &env).await;
    assert_eq!(accept.code, Some(2), "{}", accept.output());
    assert_eq!(
        accept.stderr, restore.stderr,
        "both actions refuse with the same text"
    );

    let server = run_binary(&[], &env).await;
    assert_eq!(
        server.code,
        Some(1),
        "the service below the minimum exits 1; {}",
        server.output()
    );
    assert_eq!(
        server.stderr,
        restore.stderr.replace(DR_SETTING, SERVER_SETTING),
        "the service's refusal is the Job's, naming the service's setting"
    );

    let at = pod_memory_fixture_with(&format!("{}\n", MINIMUM_MIB << 20));
    let env = job_env("redis://127.0.0.1:1", at.path(), &backup_store);
    let run = run_binary(&["dr", "restore", "inc-1"], &env).await;
    assert_eq!(
        run.code,
        Some(1),
        "at the minimum the Job passes the plan and fails to connect; {}",
        run.output()
    );
    assert!(
        !run.stderr.contains("configuration error"),
        "no configuration refusal at the minimum; {}",
        run.output()
    );
}

/// `accept-loss`'s second `--confirm discard-unconfirmed-data` (option E)
/// is refused with exit 2 before the process connects when it comes alone,
/// and so is either confirm given twice; stderr names the refusal and prints
/// the usage. The Valkey URL names a port nothing listens on, as above. Two
/// runs.
#[tokio::test]
async fn live_edit_dr_accept_loss_refuses_the_second_confirm_alone_or_a_repeated_confirm_with_exit_2(
) {
    let cgroup = pod_memory_fixture();
    let env = job_env(
        "redis://127.0.0.1:1",
        cgroup.path(),
        &[("LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS", "1")],
    );
    for (args, refusal) in [
        (
            &[
                "dr",
                "accept-loss",
                "inc-1",
                "--confirm",
                "discard-unconfirmed-data",
            ][..],
            "--confirm discard-unconfirmed-data is accepted only together with --confirm \
             discard-lost-data",
        ),
        (
            &[
                "dr",
                "accept-loss",
                "inc-1",
                "--confirm",
                "discard-lost-data",
                "--confirm",
                "discard-lost-data",
            ][..],
            "--confirm is given more than once",
        ),
    ] {
        let run = run_binary(args, &env).await;
        assert_eq!(
            run.code,
            Some(2),
            "{args:?}: a refused command line exits 2; {}",
            run.output()
        );
        assert_eq!(
            run.stderr,
            format!("live-edit dr: {refusal}\n{USAGE}\n"),
            "{args:?}: stderr names the refusal and prints the usage"
        );
        assert!(
            run.records("live-edit dr accept-loss starting").is_empty(),
            "{args:?}: the Job never starts; {}",
            run.output()
        );
    }
}

/// An accept-loss Job on an empty store held past its quiet period, run
/// outside Kubernetes with both confirms (the second first: either order is
/// accepted), reaches its body: it takes the lease, opens its write permit,
/// restores and discards nothing (the default `disabled` adapter, no backup
/// store), recounts and un-holds: the hold marker is deleted and the Job
/// exits 0, its lease released. The second confirm reaches the body: AL4
/// and AL11 carry its clause, counting 0 unconfirmed items. On linux-gnu it
/// logs that its value removal purges the allocator, the binary having
/// installed the purge ([`assert_purge_line`]). Run again with
/// the first confirm alone on the same store held again, the Job also exits
/// 0, and neither line carries the clause (fail-on-old: with the body
/// built with the second confirm whatever the command line, the second run's
/// lines carry it). Two runs.
#[cfg(feature = "test-support")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_edit_dr_accept_loss_on_an_empty_held_store_reaches_the_body_and_unholds() {
    use fred::interfaces::KeysInterface;
    use live_edit::config::key_prefix::TEST_ONLY_KEY_PREFIX_ENV;
    use live_edit::storage::dr_job::store_time_ms;
    use live_edit::storage::test_support::{HoldMarkerSeed, TestValkey};

    const INCIDENT: &str = "20261021t0900-job2";
    const SECOND_INCIDENT: &str = "20261021t0905-job3";
    const HOUR_MS: u64 = 3_600_000;

    let tv = TestValkey::setup().await;
    let now = store_time_ms(tv.client(), tv.key_schema())
        .await
        .expect("TIME");
    tv.write_hold_marker(
        &HoldMarkerSeed::default()
            .with_incident(INCIDENT)
            .detected_at(now - HOUR_MS),
    )
    .await;

    let cgroup = pod_memory_fixture();
    let valkey_url =
        std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".into());
    let env = job_env(
        &valkey_url,
        cgroup.path(),
        &[(TEST_ONLY_KEY_PREFIX_ENV, tv.prefix())],
    );
    let run = run_binary(
        &[
            "dr",
            "accept-loss",
            INCIDENT,
            "--confirm",
            "discard-unconfirmed-data",
            "--confirm",
            "discard-lost-data",
        ],
        &env,
    )
    .await;

    assert_eq!(
        run.code,
        Some(0),
        "the body un-holds and exits 0; {}",
        run.output()
    );
    let unheld = run.records(&format!(
        "the accept-loss Job cleared the hold for incident {INCIDENT}: the object-id counter is \
         0 (it was 0), above every id a backup, a surviving layer or a deleted layer used; every \
         replica restarts within seconds"
    ));
    assert_eq!(unheld.len(), 1, "one un-hold INFO; {}", run.output());
    assert_eq!(
        run.records("the quiet period is over: the accept-loss Job's write permit is open")
            .len(),
        1,
        "the permit opened; {}",
        run.output()
    );
    assert_eq!(
        run.records("released the DR Job lease").len(),
        1,
        "the lease is released at the exit; {}",
        run.output()
    );
    assert_eq!(run.records("the accept-loss Job exits 0").len(), 1);
    assert_purge_line(&run);
    let marker: i64 = tv
        .client()
        .pool
        .exists(tv.key_schema().dr_hold_marker())
        .await
        .expect("EXISTS");
    assert_eq!(marker, 0, "the un-hold deleted the hold marker");
    assert_eq!(
        run.records(&format!("{AL4_EMPTY}{UNCONFIRMED_PLAN_CLAUSE}"))
            .len(),
        1,
        "AL4 carries the second confirm's clause; {}",
        run.output()
    );
    assert_eq!(
        run.records(&format!("{AL11_EMPTY}{UNCONFIRMED_SUMMARY_CLAUSE}"))
            .len(),
        1,
        "and so does AL11; {}",
        run.output()
    );

    // The first confirm alone, on the same empty store held again: neither
    // line carries the clause.
    tv.write_hold_marker(
        &HoldMarkerSeed::default()
            .with_incident(SECOND_INCIDENT)
            .detected_at(now - HOUR_MS),
    )
    .await;
    let run = run_binary(
        &[
            "dr",
            "accept-loss",
            SECOND_INCIDENT,
            "--confirm",
            "discard-lost-data",
        ],
        &env,
    )
    .await;

    assert_eq!(
        run.code,
        Some(0),
        "the first confirm alone also un-holds; {}",
        run.output()
    );
    assert_eq!(
        run.records(AL4_EMPTY).len(),
        1,
        "AL4 without the clause; {}",
        run.output()
    );
    assert_eq!(
        run.records(AL11_EMPTY).len(),
        1,
        "AL11 without the clause; {}",
        run.output()
    );
}

/// The accept-loss Job's start line when the binary installed its allocator
/// purge (`dr::job::execute::discard::log_value_removal_purge`): INFO, the
/// value removal admitted at the purged 50 MiB.
const PURGE_INSTALLED: &str = "the accept-loss Job's value removal purges the allocator around \
                               each of its transactions and admits each removal at 50 MiB";
/// The same line when it installed none: WARN, the un-purged 158 MiB.
const NO_PURGE_INSTALLED: &str = "the accept-loss Job runs with no allocator purge installed: its \
                                  value removal admits each removal at 158 MiB, the un-purged \
                                  figure, so the restore memory holds fewer removals at once";

/// W-T7, the binary installs its allocator purge (fail-on-old): on
/// linux-gnu, where it runs on jemalloc, an accept-loss run logs the purge
/// INFO once, before planning, so even on an empty store, and never the
/// no-purge WARN; elsewhere, the WARN. Fails with `run_dr_job` installing
/// none on linux-gnu (in-place perturbation of `main.rs`, `None`): the WARN
/// instead of the INFO.
fn assert_purge_line(run: &JobRun) {
    let (installed, none) = (
        run.records(PURGE_INSTALLED),
        run.records(NO_PURGE_INSTALLED),
    );
    let levels = |records: &[serde_json::Value]| -> Vec<String> {
        records
            .iter()
            .map(|r| r["level"].as_str().unwrap_or_default().to_owned())
            .collect()
    };
    if cfg!(all(target_os = "linux", target_env = "gnu")) {
        assert_eq!(
            (levels(&installed), none.len()),
            (vec!["INFO".to_owned()], 0),
            "the binary installed its purge: one INFO, no WARN; {}",
            run.output()
        );
    } else {
        assert_eq!(
            (installed.len(), levels(&none)),
            (0, vec!["WARN".to_owned()]),
            "no purge off linux-gnu: one WARN; {}",
            run.output()
        );
    }
}

/// AL4 on an empty store (`dr::job::plan::report::render_discard_plan`):
/// nothing to discard.
const AL4_EMPTY: &str = "the accept-loss Job discards 0 layers, 0 sessions and 0 large values \
                         from 0 layers, and rewrites 0 session backups; 0 layers are registered \
                         as deleted";
/// The clause AL4 adds under the second confirm, with nothing unconfirmed.
const UNCONFIRMED_PLAN_CLAUSE: &str = "; 0 items among these could not be confirmed lost and are \
                                       discarded because it runs with --confirm \
                                       discard-unconfirmed-data";
/// AL11 on an empty store (`render_discard_summary`): nothing discarded.
const AL11_EMPTY: &str = "the accept-loss Job's summary: discarded 0 layers and 0 sessions, \
                          removed 0 large values from 0 layers, left 0 mappings as found, \
                          registered 0 layers as deleted, deleted 0 session backups and rewrote \
                          0";
/// The clause AL11 adds under the second confirm, with nothing unconfirmed.
const UNCONFIRMED_SUMMARY_CLAUSE: &str = "; 0 items among these could not be confirmed lost and \
                                          were discarded because it ran with --confirm \
                                          discard-unconfirmed-data";

/// A restore Job on a store held past its quiet period, run outside
/// Kubernetes, reaches its body: it takes the lease, opens its write permit,
/// plans and writes the restore (nothing, on this empty store), recounts the
/// large values' references (none), and un-holds: the hold marker is deleted
/// and the Job exits 0. The lease is released at the exit.
/// Nothing in its output mentions Kubernetes: the Job reads no Kubernetes
/// object.
#[cfg(feature = "test-support")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_edit_dr_restore_on_a_store_held_past_its_quiet_period_reaches_the_body_and_unholds() {
    use fred::interfaces::KeysInterface;
    use live_edit::config::key_prefix::TEST_ONLY_KEY_PREFIX_ENV;
    use live_edit::storage::dr_job::store_time_ms;
    use live_edit::storage::test_support::{HoldMarkerSeed, TestValkey};

    const INCIDENT: &str = "20261016t0900-job1";
    const HOUR_MS: u64 = 3_600_000;

    let tv = TestValkey::setup().await;
    let now = store_time_ms(tv.client(), tv.key_schema())
        .await
        .expect("TIME");
    tv.write_hold_marker(
        &HoldMarkerSeed::default()
            .with_incident(INCIDENT)
            .detected_at(now - HOUR_MS),
    )
    .await;
    let marker_before: std::collections::HashMap<String, String> = {
        use fred::interfaces::HashesInterface;
        tv.client()
            .pool
            .hgetall(tv.key_schema().dr_hold_marker())
            .await
            .expect("HGETALL")
    };

    let cgroup = pod_memory_fixture();
    let valkey_url =
        std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".into());
    let env = job_env(
        &valkey_url,
        cgroup.path(),
        &[(TEST_ONLY_KEY_PREFIX_ENV, tv.prefix())],
    );
    let run = run_binary(&["dr", "restore", INCIDENT], &env).await;

    assert_eq!(
        run.code,
        Some(0),
        "the body un-holds and exits 0; {}",
        run.output()
    );
    let body = run.records(&format!(
        "the restore Job cleared the hold for incident {INCIDENT}: the object-id counter is 0 \
         (it was 0), above every id a backup, a surviving layer or a deleted layer used; every \
         replica restarts within seconds"
    ));
    assert_eq!(body.len(), 1, "one un-hold INFO; {}", run.output());
    assert_eq!(body[0]["level"], "INFO");
    assert_eq!(
        run.records("took the DR Job lease").len(),
        1,
        "the Job took its lease; {}",
        run.output()
    );
    assert_eq!(
        run.records("the quiet period is over: the restore Job's write permit is open")
            .len(),
        1,
        "the permit opened; {}",
        run.output()
    );
    assert_eq!(
        run.records("released the DR Job lease").len(),
        1,
        "the lease is released at the exit; {}",
        run.output()
    );
    assert_eq!(run.records("the restore Job exits 0").len(), 1);
    assert!(
        !run.stdout.to_ascii_lowercase().contains("kubernetes"),
        "the Job reads no Kubernetes object and says nothing about it; {}",
        run.output()
    );
    let lease: i64 = tv
        .client()
        .pool
        .exists(tv.key_schema().dr_job_lease())
        .await
        .expect("EXISTS");
    assert_eq!(lease, 0, "no lease is left behind");
    assert!(
        !marker_before.is_empty(),
        "fixture: the store was held before the run"
    );
    let marker_after: i64 = tv
        .client()
        .pool
        .exists(tv.key_schema().dr_hold_marker())
        .await
        .expect("EXISTS");
    assert_eq!(marker_after, 0, "the un-hold deleted the hold marker");
}
