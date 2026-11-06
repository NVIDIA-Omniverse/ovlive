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

//! The disaster-recovery Jobs' rules that need no store (`live_edit::dr::job`;
//! SDD SR-8.1, SR-9.1): the command line, the exit codes, the quiet period Q
//! read from a configuration loaded the way the binary loads it, and the
//! not-found cache setting the Jobs read.
//!
//! - **The command line.** `dr restore <incident>` alone parses, and any
//!   `--fleet-*` flag is refused as an unexpected argument, for both modes.
//! - **The exit codes.** `JobExit` has exactly the five outcomes, none of
//!   them exit 5.
//! - **Q.** `QuietPeriod::from_config` over the default configuration is
//!   216 s with `StorageApi` and 116 s with every other adapter; 107 s at
//!   D = 10 s, and 686 s at D = 600 s. Each after-entry term is pinned where
//!   it is the largest, so a Q that dropped it reads a different figure: the
//!   hold's write bound at the defaults, the join teardown at D = 10 s, the
//!   backup store's call ceiling C at a raised request timeout, and the
//!   shutdown bound G at a raised shutdown budget. G is pinned equal to the
//!   bound the server logs at startup,
//!   `grpc::transport::minimum_termination_grace`.
//! - **`LIVE_EDIT_CHECKPOINT_STORAGE_API_NOT_FOUND_CACHE_SECS`**: 900 by
//!   default, any whole number of seconds (0 included), and a refusal naming
//!   the variable otherwise.
//! - **`LIVE_EDIT_DR_RESTORE_PARALLELISM`**: 8 by default, any count from 1,
//!   and a refusal naming the variable for 0 or a non-count. No memory
//!   fraction is read.
//! - **The restore memory** (`RestoreMemory::derive`): half a known limit
//!   (fixed, or the reference size named explicitly), and one layer at a
//!   time for every reason no limit was read. Any parallelism starts
//!   wherever the Job's line closes, for either Job and in serial mode
//!   (each accept-loss value removal takes one admission within the budget
//!   and waits for it, so no setting is refused at startup); a limit the
//!   Job's line does not close at is refused for the limit.
//! - **The restore writer's source fence**: its files name no release,
//!   backup mirror or spawned task, and every file of
//!   `src/dr/job/execute/` is among those scanned.
//!
//! The configuration is loaded with `Config::from_env_with` under
//! `temp_env`, so every setting a test does not name is the binary's default,
//! not the test fixture's.

use std::time::Duration;

use live_edit::checkpoint::store::LOCAL_PUBLISH_WORST_CASE;
use live_edit::checkpoint::CheckpointAdapter;
use live_edit::config::pod_memory::ReferenceReason;
use live_edit::config::{Config, PlanInputs, PodMemoryLimit, PodMemoryPlan, PodMemorySource};
use live_edit::dr::hold::STALE_VERDICT_THRESHOLD;
use live_edit::dr::job::execute::memory::value_removal_admission;
use live_edit::dr::job::quiet::{adapter_call_ceiling, shutdown_bound, QuietPeriod};
use live_edit::dr::job::{
    parse_job_args, ArgsError, JobExit, JobMode, RestoreMemory, EXIT_USAGE, USAGE,
};
use live_edit::grpc::graceful_shutdown::StreamPacing;
use live_edit::grpc::transport::minimum_termination_grace;

const S: Duration = Duration::from_secs(1);

/// The settings `Config::from_env` needs to load at all, and the adapter.
/// Everything else is left at the binary's default.
const BASE_ENV: [(&str, &str); 3] = [
    (
        "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
        "http://storage.example.com",
    ),
    ("LIVE_EDIT_USERINFO_ENDPOINT", "http://userinfo.example.com"),
    ("LIVE_EDIT_CHECKPOINT_ADAPTER", "Disabled"),
];

/// Load the configuration as the binary does, from [`BASE_ENV`] overlaid
/// with `overrides`.
fn load_config(overrides: &[(&'static str, &'static str)]) -> Result<Config, String> {
    let mut env: Vec<(&'static str, Option<&'static str>)> = BASE_ENV
        .iter()
        .filter(|(var, _)| !overrides.iter().any(|(o, _)| o == var))
        .map(|(var, value)| (*var, Some(*value)))
        .collect();
    env.extend(overrides.iter().map(|(var, value)| (*var, Some(*value))));
    temp_env::with_vars(env, || Config::from_env_with(PodMemorySource::Reference))
}

/// [`load_config`] that must load, with `adapter` set on it.
fn config_for(adapter: CheckpointAdapter, overrides: &[(&'static str, &'static str)]) -> Config {
    let mut config = load_config(overrides).expect("the configuration loads");
    config.checkpoint_adapter = adapter;
    config
}

const EVERY_ADAPTER: [CheckpointAdapter; 4] = [
    CheckpointAdapter::Disabled,
    CheckpointAdapter::InMemory,
    CheckpointAdapter::LocalDisk,
    CheckpointAdapter::StorageApi,
];

fn parse(args: &[&str]) -> Result<live_edit::dr::job::JobArgs, ArgsError> {
    parse_job_args(args.iter().map(|a| (*a).to_owned()))
}

// ═════════════════════════════════════════════════════════════════════════════
// The command line (T-R1a)
// ═════════════════════════════════════════════════════════════════════════════

/// `dr restore <incident>` is a whole command line: the Job reads no
/// Kubernetes object, so it takes no `--fleet-*` flag. Every one of them, in
/// either spelling and in either mode, is refused as an unexpected argument,
/// naming the whole argument, which the binary turns into exit 2 before it
/// connects to anything. The usage text names no such flag.
#[test]
fn dr_restore_takes_an_incident_alone_and_refuses_every_fleet_flag() {
    let restore = parse(&["restore", "inc-1"]).expect("restore <incident> alone parses");
    assert_eq!(restore.mode, JobMode::Restore);
    assert_eq!(restore.incident, "inc-1");

    let confirm = ["accept-loss", "inc-1", "--confirm", "discard-lost-data"];
    for (flag, value) in [
        ("--fleet-namespace", "ns"),
        ("--fleet-service", "live-edit"),
        ("--fleet-deployment", "live-edit"),
    ] {
        assert_eq!(
            parse(&["restore", "inc-1", flag, value]),
            Err(ArgsError::Unexpected(flag.to_owned())),
            "restore refuses {flag} as unexpected"
        );
        let inline = format!("{flag}={value}");
        assert_eq!(
            parse(&["restore", "inc-1", &inline]),
            Err(ArgsError::Unexpected(inline.clone())),
            "restore refuses {inline}, naming the whole argument"
        );
        let mut accept_loss = confirm.to_vec();
        accept_loss.extend([flag, value]);
        assert_eq!(
            parse(&accept_loss),
            Err(ArgsError::Unexpected(flag.to_owned())),
            "accept-loss refuses {flag} as unexpected"
        );
    }
    assert_eq!(
        ArgsError::Unexpected("--fleet-namespace".to_owned()).to_string(),
        "unexpected argument \"--fleet-namespace\"",
        "the refusal the binary prints names the argument"
    );
    assert_eq!(EXIT_USAGE, 2, "a refused command line exits 2");
    assert_eq!(
        USAGE,
        "usage:\n  live-edit dr restore <incident>\n  live-edit dr accept-loss <incident> \
         --confirm discard-lost-data [--confirm discard-unconfirmed-data]",
        "the usage text names the two subcommands and no --fleet-* flag"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The exit codes (T-R1c)
// ═════════════════════════════════════════════════════════════════════════════

/// A Job ends in one of five ways, with the exit codes 0, 0, 3, 4 and 1; no
/// outcome exits 5. The match is exhaustive, so an outcome added to `JobExit`
/// stops this test compiling until its code is pinned here.
#[test]
fn job_exit_has_five_outcomes_and_none_exits_5() {
    fn contract(exit: JobExit) -> i32 {
        match exit {
            JobExit::Done | JobExit::NothingToDo => 0,
            JobExit::StaysHeld => 3,
            JobExit::DifferentIncident => 4,
            JobExit::Failed => 1,
        }
    }
    let every = [
        JobExit::Done,
        JobExit::NothingToDo,
        JobExit::StaysHeld,
        JobExit::DifferentIncident,
        JobExit::Failed,
    ];
    for exit in every {
        assert_eq!(exit.code(), contract(exit), "{exit:?}");
        assert_ne!(exit.code(), 5, "{exit:?}: no Job outcome exits 5");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Q from the configuration (item 2, T-Q)
// ═════════════════════════════════════════════════════════════════════════════

/// G at the default shutdown settings: the drain (5 s), the 0.25 s GOAWAY
/// delay, the bound after GOAWAY at D = 30 s (max(30 s, 20 s) + 5 s), the
/// overall budget (30 s) and the 6 s flush.
const G_AT_DEFAULTS: Duration = Duration::from_millis(76_250);

/// At the binary's defaults (D = 30 s, T = 5 s, a 30 s request timeout, G =
/// 76.25 s), Q is 216 s with `StorageApi`, whose call ceiling C (180 s) sets
/// the after-entry term, and 116 s with every other adapter, where the hold's
/// write bound (80 s) sets it. Every term is pinned, so a production constant
/// that grows past these figures fails here. The stale threshold is below Q
/// for every adapter.
#[test]
fn quiet_period_at_the_default_settings_is_216_s_with_storage_api_and_116_s_otherwise() {
    for adapter in EVERY_ADAPTER {
        let config = config_for(adapter, &[]);
        assert_eq!(config.rpc_deadline_ms, 30_000, "precondition: default D");
        assert_eq!(
            config.valkey_command_timeout_ms, 5_000,
            "precondition: default T"
        );
        let q = QuietPeriod::from_config(&config);
        assert_eq!(q.stale_threshold, 30 * S, "{adapter:?}");
        assert_eq!(q.poll_interval, 6 * S, "{adapter:?}: the largest poll");
        assert_eq!(q.hold_write_bound, 80 * S, "{adapter:?}: D + T + 45 s");
        assert_eq!(
            q.join_teardown_bound,
            71 * S,
            "{adapter:?}: 5T + 1 s + 45 s"
        );
        assert_eq!(q.shutdown_bound, G_AT_DEFAULTS, "{adapter:?}: G");
        let (c, total) = match adapter {
            CheckpointAdapter::StorageApi => (180 * S, 216 * S),
            CheckpointAdapter::LocalDisk => (5 * S, 116 * S),
            CheckpointAdapter::Disabled | CheckpointAdapter::InMemory => (Duration::ZERO, 116 * S),
        };
        assert_eq!(q.adapter_call_ceiling, c, "{adapter:?}: C");
        assert_eq!(q.total(), total, "{adapter:?}: Q");
        assert!(
            STALE_VERDICT_THRESHOLD < q.total(),
            "{adapter:?}: the stale threshold is below Q"
        );
    }
}

/// A short RPC deadline (D = 10 s, T = 5 s, no backup store): the join
/// teardown, 5T + 46 s = 71 s, is above the hold's write bound (60 s) and G
/// (66.25 s at D = 10 s), and sets Q at 107 s. A Q without the teardown term
/// would read 102.25 s.
#[test]
fn quiet_period_at_a_10_s_deadline_is_set_by_the_join_teardown_at_107_s() {
    let config = config_for(
        CheckpointAdapter::Disabled,
        &[("LIVE_EDIT_RPC_DEADLINE_MS", "10000")],
    );
    let q = QuietPeriod::from_config(&config);
    assert_eq!(q.hold_write_bound, 60 * S);
    assert_eq!(q.join_teardown_bound, 71 * S);
    assert_eq!(q.adapter_call_ceiling, Duration::ZERO);
    assert_eq!(q.shutdown_bound, Duration::from_millis(66_250));
    assert_eq!(q.after_entry(), q.join_teardown_bound);
    assert_eq!(q.total(), 107 * S);
}

/// A large RPC deadline (D = 600 s, `StorageApi` at the default 30 s request
/// timeout): the hold's write bound, D + T + 45 s = 650 s, is above C (180 s)
/// and G (646.25 s), so Q is 686 s. The pre-kl-retire D + max(T, 45 s) would
/// give 681 s.
#[test]
fn quiet_period_at_a_600_s_deadline_is_686_s() {
    let config = config_for(
        CheckpointAdapter::StorageApi,
        &[("LIVE_EDIT_RPC_DEADLINE_MS", "600000")],
    );
    let q = QuietPeriod::from_config(&config);
    assert_eq!(q.hold_write_bound, 650 * S);
    assert_eq!(q.adapter_call_ceiling, 180 * S);
    assert_eq!(q.shutdown_bound, Duration::from_millis(646_250));
    assert_eq!(q.total(), 686 * S);
}

/// C sets Q once the backup store's call ceiling is the largest after-entry
/// term: `StorageApi` with a 200 000 ms request timeout has C = 200 s × 3
/// attempts × 2 auth attempts = 1 200 s, above the hold's write bound at
/// D = 600 s (650 s) and G (646.25 s), so Q is 30 + 6 + 1 200 = 1 236 s. A Q
/// without the C term would read 686 s.
#[test]
fn quiet_period_takes_the_backup_store_call_ceiling_when_it_is_the_largest_term() {
    let config = config_for(
        CheckpointAdapter::StorageApi,
        &[
            ("LIVE_EDIT_RPC_DEADLINE_MS", "600000"),
            ("LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS", "200000"),
        ],
    );
    let q = QuietPeriod::from_config(&config);
    assert_eq!(q.adapter_call_ceiling, 1_200 * S, "C");
    assert!(
        q.adapter_call_ceiling > q.hold_write_bound.max(q.shutdown_bound),
        "precondition: C is the largest after-entry term"
    );
    assert_eq!(q.after_entry(), q.adapter_call_ceiling);
    assert_eq!(q.total(), 1_236 * S);
}

/// C per adapter: `StorageApi` and `LocalDisk` take the configuration's
/// enforced call ceiling (180 s at the default request timeout, and
/// `LOCAL_PUBLISH_WORST_CASE`, 5 s); `Disabled` has no backup store and
/// `InMemory` writes stop with the replica, so both add no term.
#[test]
fn adapter_call_ceiling_is_the_configured_ceiling_for_a_persistent_store_and_zero_otherwise() {
    for adapter in EVERY_ADAPTER {
        let config = config_for(adapter, &[]);
        let want = match adapter {
            CheckpointAdapter::StorageApi => 180 * S,
            CheckpointAdapter::LocalDisk => LOCAL_PUBLISH_WORST_CASE,
            CheckpointAdapter::Disabled | CheckpointAdapter::InMemory => Duration::ZERO,
        };
        assert_eq!(adapter_call_ceiling(&config), want, "{adapter:?}");
        if matches!(
            adapter,
            CheckpointAdapter::StorageApi | CheckpointAdapter::LocalDisk
        ) {
            assert_eq!(
                adapter_call_ceiling(&config),
                config.checkpoint_adapter_call_ceiling(),
                "{adapter:?}: the ceiling the adapter itself enforces"
            );
        }
    }
    assert_eq!(LOCAL_PUBLISH_WORST_CASE, 5 * S);
}

/// G sets Q once the shutdown budgets are raised (PD-A): with
/// `LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS=120` and no `StorageApi`, G is
/// 5 + 0.25 + 35 + 120 + 6 = 166.25 s, above every other after-entry term, so
/// Q is 36 s + G = 202.25 s. A Q without G would read 116 s, and a pod
/// already shutting down when the hold began could write past it.
#[test]
fn quiet_period_takes_the_shutdown_bound_once_the_shutdown_budgets_are_raised() {
    for adapter in [
        CheckpointAdapter::Disabled,
        CheckpointAdapter::InMemory,
        CheckpointAdapter::LocalDisk,
    ] {
        let config = config_for(
            adapter,
            &[("LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS", "120")],
        );
        let q = QuietPeriod::from_config(&config);
        assert_eq!(
            q.shutdown_bound,
            Duration::from_millis(166_250),
            "{adapter:?}: G"
        );
        assert_eq!(q.after_entry(), q.shutdown_bound, "{adapter:?}");
        assert_eq!(
            q.total(),
            Duration::from_millis(202_250),
            "{adapter:?}: Q = 36 s + G"
        );
    }
}

/// G is the server's own bound: `quiet::shutdown_bound` equals
/// `grpc::transport::minimum_termination_grace` of the drain, D, the stream
/// pacing and the overall budget, at the defaults (76.25 s) and at a raised
/// overall budget, drain and deadline. The Job and the server's startup line
/// must never disagree on it.
#[test]
fn shutdown_bound_is_the_servers_minimum_termination_grace() {
    let defaults = config_for(CheckpointAdapter::Disabled, &[]);
    assert_eq!(
        shutdown_bound(&defaults),
        minimum_termination_grace(5 * S, 30 * S, StreamPacing::DEFAULT, 30 * S),
    );
    assert_eq!(shutdown_bound(&defaults), G_AT_DEFAULTS);

    let raised = config_for(
        CheckpointAdapter::Disabled,
        &[
            ("LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS", "12"),
            ("LIVE_EDIT_SHUTDOWN_OVERALL_BUDGET_SECS", "120"),
            ("LIVE_EDIT_RPC_DEADLINE_MS", "45000"),
        ],
    );
    let want = minimum_termination_grace(12 * S, 45 * S, raised.shutdown_stream_pacing(), 120 * S);
    assert_eq!(shutdown_bound(&raised), want);
    assert_eq!(
        want,
        Duration::from_millis(12_000 + 250 + 50_000 + 120_000 + 6_000),
        "drain + GOAWAY delay + max(D, window) + 5 s + overall + flush"
    );
    assert_eq!(QuietPeriod::from_config(&raised).shutdown_bound, want);
}

// ═════════════════════════════════════════════════════════════════════════════
// LIVE_EDIT_CHECKPOINT_STORAGE_API_NOT_FOUND_CACHE_SECS
// ═════════════════════════════════════════════════════════════════════════════

const NOT_FOUND_CACHE: &str = "LIVE_EDIT_CHECKPOINT_STORAGE_API_NOT_FOUND_CACHE_SECS";

/// L, the backup store's not-found cache lifetime: 900 s unless set; any
/// whole number of seconds, 0 included, is taken as it is; a negative,
/// fractional or non-numeric value is refused, naming the variable.
#[test]
fn not_found_cache_setting_defaults_to_900_and_takes_any_whole_number_of_seconds() {
    let default = load_config(&[]).expect("the default loads");
    assert_eq!(default.checkpoint_storage_api_not_found_cache_secs, 900);
    for (value, want) in [("0", 0), ("3", 3), ("86400", 86_400)] {
        let config = load_config(&[(NOT_FOUND_CACHE, value)])
            .unwrap_or_else(|e| panic!("{NOT_FOUND_CACHE}={value} must load: {e}"));
        assert_eq!(
            config.checkpoint_storage_api_not_found_cache_secs, want,
            "{value}"
        );
    }
    for value in ["-1", "1.5", "x"] {
        let err = load_config(&[(NOT_FOUND_CACHE, value)])
            .expect_err("a value that is not a whole number of seconds is refused");
        assert!(
            err.contains(NOT_FOUND_CACHE),
            "{value}: the refusal names the variable: {err}"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// LIVE_EDIT_DR_RESTORE_PARALLELISM and the restore memory (U8, U6)
// ═════════════════════════════════════════════════════════════════════════════

const PARALLELISM: &str = "LIVE_EDIT_DR_RESTORE_PARALLELISM";

/// U8. The most layers the restore Job holds at once: 8 unless set; any
/// count from 1 up is taken as it is; 0 is refused (the Job could restore
/// nothing), and so is a value that is not a count, each naming the
/// variable. It is the one restore setting: there is no memory-fraction
/// setting, so setting one changes nothing.
#[test]
fn restore_parallelism_defaults_to_8_and_refuses_zero_or_a_non_count_naming_the_variable() {
    let default = load_config(&[]).expect("the default loads");
    assert_eq!(default.dr_restore_parallelism.get(), 8);
    for (value, want) in [("1", 1), ("3", 3), ("64", 64)] {
        let config = load_config(&[(PARALLELISM, value)])
            .unwrap_or_else(|e| panic!("{PARALLELISM}={value} must load: {e}"));
        assert_eq!(config.dr_restore_parallelism.get(), want, "{value}");
    }
    let zero = load_config(&[(PARALLELISM, "0")]).expect_err("0 is refused");
    assert_eq!(
        zero,
        "invalid value for LIVE_EDIT_DR_RESTORE_PARALLELISM: must be greater than 0 (the restore \
         Job could restore no layer at all)"
    );
    for value in ["-1", "2.5", "eight"] {
        let err = load_config(&[(PARALLELISM, value)])
            .expect_err("a value that is not a count is refused");
        assert!(
            err.contains(PARALLELISM),
            "{value:?}: the refusal names the variable: {err}"
        );
    }
    let ignored = load_config(&[("LIVE_EDIT_DR_RESTORE_MEMORY_FRACTION", "0.9")])
        .expect("an unknown variable is not read");
    assert_eq!(
        RestoreMemory::derive(&ignored, JobMode::Restore),
        RestoreMemory::derive(&default, JobMode::Restore),
        "no fraction setting: the restore budget is half the limit whatever it says"
    );
}

/// [`load_config`] with the pod memory limit taken from `source`.
fn load_config_with(source: PodMemorySource) -> Config {
    let env: Vec<(&str, Option<&str>)> = BASE_ENV
        .iter()
        .map(|(var, value)| (*var, Some(*value)))
        .collect();
    temp_env::with_vars(env, || Config::from_env_with(source)).expect("the configuration loads")
}

/// U6. The Job's restore memory from a loaded configuration: half the pod
/// memory limit, at the configured parallelism, when a limit is known (a
/// fixed one, or the reference size named explicitly, as a `Config` built
/// field by field has it); one layer at a time, with no byte bound, for
/// every reason the Job could read no limit.
#[test]
fn restore_memory_is_half_a_known_limit_and_serial_for_every_reason_no_limit_was_read() {
    const MIB: u64 = 1024 * 1024;
    for (source, budget_mib) in [
        (PodMemorySource::Fixed(8_192 * MIB), 4_096),
        (PodMemorySource::Reference, 2_560),
    ] {
        let config = load_config_with(source.clone());
        let memory = RestoreMemory::derive(&config, JobMode::Restore).expect("closes");
        assert_eq!(memory.budget_bytes(), Some(budget_mib * MIB), "{source:?}");
        assert_eq!(memory.parallelism(), 8, "{source:?}");
        assert!(matches!(memory, RestoreMemory::Bytes { .. }), "{source:?}");
    }

    for reason in [
        ReferenceReason::Unlimited,
        ReferenceReason::NotFound,
        ReferenceReason::Unreadable("memory.max: permission denied".to_owned()),
    ] {
        let mut config = load_config_with(PodMemorySource::Reference);
        config.pod_memory = PodMemoryPlan::derive(
            PodMemoryLimit::reference(Some(reason.clone())),
            &PlanInputs::from_config(&config),
        )
        .expect("the reference size closes");
        let memory =
            RestoreMemory::derive(&config, JobMode::Restore).expect("serial mode is no refusal");
        assert_eq!(
            memory,
            RestoreMemory::Serial {
                reason: reason.clone(),
                parallelism: config.dr_restore_parallelism,
            },
            "{reason}"
        );
        assert_eq!(memory.parallelism(), 1, "{reason}: one layer at a time");
        assert_eq!(memory.configured_parallelism(), 8, "{reason}");
        assert_eq!(memory.budget_bytes(), None, "{reason}: no byte bound");
    }
}

/// A configuration loaded at a fixed pod memory limit of `limit_mib` with
/// `parallelism` layers at once.
fn config_at(limit_mib: u64, parallelism: usize) -> Config {
    const MIB: u64 = 1024 * 1024;
    let mut config = load_config_with(PodMemorySource::Fixed(limit_mib * MIB));
    config.dr_restore_parallelism = std::num::NonZeroUsize::new(parallelism).expect("non-zero");
    config
}

/// The restore memory envelope's record (plan b.tv3, ruling w6m2), which
/// derives the restore Job's charges and the removals its budget holds.
const RESTORE_ENVELOPE_DERIVATION_JSON: &str =
    "docs-internal/validation/restore-memory-envelope-tv3-20261022/derivation.json";

/// The committed record at `path` (relative to the crate root), parsed;
/// `None` in the published source tree, which has no `docs-internal/`
/// (only there: in this repository a missing record fails).
fn committed_record(path: &str) -> Option<serde_json::Value> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    if !root.join("docs-internal").is_dir() {
        eprintln!("skipped: no docs-internal/ (published source tree), so no {path}");
        return None;
    }
    let raw = std::fs::read_to_string(root.join(path))
        .unwrap_or_else(|e| panic!("the committed record {path}: {e}"));
    Some(serde_json::from_str(&raw).expect("the record is valid JSON"))
}

/// The value at JSON `pointer` in `record`, as an integer.
fn recorded_u64(record: &serde_json::Value, pointer: &str) -> u64 {
    record
        .pointer(pointer)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or_else(|| panic!("the record carries {pointer} as a non-negative integer"))
}

/// The parallelism is not refused on the Job's limit: accept-loss's value
/// removal holds one admission per layer within the restore memory budget,
/// whose first-come-first-served admission bounds the removals in flight
/// (`dr::job::execute::memory::value_removal_admission`), so any setting
/// starts wherever the Job's line closes, for either Job, even one past what
/// the budget holds at one admission each. At 2,687 MiB (the no-backup
/// minimum) and 5 GiB, the removals `R` holds at the purged admission are
/// read from the committed record (`RESTORE_ENVELOPE_DERIVATION_JSON`,
/// `value_removal_admission.removals_in_r`, keys `2687_mib` and `5120_mib`;
/// 26 and 51 when written), and the budget over the admission must equal
/// them, so an edit to the record or to the budget or admission without the
/// other fails here. At each limit, one setting past that and 1,000 start in
/// bytes mode with that parallelism. In the published source tree, which has
/// no `docs-internal/`, the record check is skipped and the count is the
/// code's own budget over its admission, so the startup checks still run.
/// Serial mode, with no limit read, takes 1,000 too and runs one layer at a
/// time.
#[test]
fn restore_memory_takes_any_parallelism_wherever_its_line_closes() {
    let admission = value_removal_admission(true);
    let record = committed_record(RESTORE_ENVELOPE_DERIVATION_JSON);
    for limit_mib in [2_687_u64, 5_120] {
        let removals_in_r = match &record {
            Some(record) => recorded_u64(
                record,
                &format!("/value_removal_admission/removals_in_r/{limit_mib}_mib"),
            ),
            None => {
                RestoreMemory::derive(&config_at(limit_mib, 1), JobMode::Restore)
                    .unwrap_or_else(|refusal| panic!("1 at {limit_mib} MiB: {refusal}"))
                    .budget_bytes()
                    .expect("bytes mode has a budget")
                    / admission
            }
        };
        for p in [removals_in_r as usize + 1, 1_000] {
            for mode in [JobMode::Restore, JobMode::AcceptLoss] {
                let memory = RestoreMemory::derive(&config_at(limit_mib, p), mode)
                    .unwrap_or_else(|refusal| panic!("{p} at {limit_mib} MiB, {mode}: {refusal}"));
                assert!(
                    matches!(memory, RestoreMemory::Bytes { .. }),
                    "{p} at {limit_mib} MiB, {mode}: {memory:?}"
                );
                assert_eq!(memory.parallelism(), p, "{p} at {limit_mib} MiB, {mode}");
                let budget = memory.budget_bytes().expect("bytes mode has a budget");
                assert_eq!(
                    budget / admission,
                    removals_in_r,
                    "{limit_mib} MiB: the removals R holds are the record's \
                     value_removal_admission.removals_in_r.{limit_mib}_mib"
                );
                assert!(
                    p as u64 * admission > budget,
                    "{p} at {limit_mib} MiB: past what R holds at one admission each"
                );
            }
        }
    }

    let mut config = config_at(2_687, 1_000);
    config.pod_memory = PodMemoryPlan::derive(
        PodMemoryLimit::reference(Some(ReferenceReason::Unlimited)),
        &PlanInputs::from_config(&config),
    )
    .expect("the reference size closes");
    let memory = RestoreMemory::derive(&config, JobMode::AcceptLoss)
        .unwrap_or_else(|refusal| panic!("serial mode at 1,000: {refusal}"));
    assert!(
        matches!(memory, RestoreMemory::Serial { parallelism, .. } if parallelism.get() == 1_000),
        "serial mode keeps the setting: {memory:?}"
    );
    assert_eq!(memory.parallelism(), 1, "serial mode: one layer at a time");
}

/// A limit the Job's line does not close at is refused for its limit, at any
/// parallelism: the operator's lever there is the limit. The server's plan,
/// derived at 2,687 MiB for the default pools, closes; the Job's line, at
/// 2,000 clients a pool, does not.
#[test]
fn restore_memory_refuses_a_limit_its_line_does_not_close_at() {
    let mut config = config_at(2_687, 1_000);
    config.valkey_pool_size = 2_000;
    let refusal = RestoreMemory::derive(&config, JobMode::Restore)
        .expect_err("the Job's line does not close at 2,000 clients a pool");
    assert!(
        refusal
            .starts_with("live-edit dr: the restore Job's memory limit of 2687 MiB is below the ")
            && refusal.ends_with("; raise dr.resources"),
        "the limit's refusal: {refusal}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The restore writer's source fence (U10)
// ═════════════════════════════════════════════════════════════════════════════

/// The restore writer's source: its stages, its object and mapping writes,
/// and accept-loss's discards, which run after it.
const WRITER_SOURCES: [(&str, &str); 7] = [
    (
        "src/dr/job/execute/mod.rs",
        include_str!("../../src/dr/job/execute/mod.rs"),
    ),
    (
        "src/dr/job/execute/discard.rs",
        include_str!("../../src/dr/job/execute/discard.rs"),
    ),
    (
        "src/dr/job/execute/object.rs",
        include_str!("../../src/dr/job/execute/object.rs"),
    ),
    (
        "src/dr/job/execute/session.rs",
        include_str!("../../src/dr/job/execute/session.rs"),
    ),
    (
        "src/dr/job/execute/memory.rs",
        include_str!("../../src/dr/job/execute/memory.rs"),
    ),
    (
        "src/storage/dr_write.rs",
        include_str!("../../src/storage/dr_write.rs"),
    ),
    (
        "src/storage/layer/dr_restore.rs",
        include_str!("../../src/storage/layer/dr_restore.rs"),
    ),
];

/// U10. The restore writer emits no external-value release (Task 9's
/// recount owns every count), wires no backup mirror, and spawns no task, so
/// dropping its future (SIGTERM, a lost lease) stops every write: none of
/// its files names `enqueue_ext_releases`, `schedule_old_object_ext_gc`,
/// `with_ext_mirror_sender`, or a task spawn — `tokio::spawn`,
/// `tokio::task::spawn(`, a `JoinSet`, `Handle::spawn` or a method call
/// `.spawn(` (a decode's `spawn_blocking` stays allowed: it writes nothing).
/// The scan reads every `.rs` file in `src/dr/job/execute/` and refuses a
/// subdirectory there, so a writer file added there, at any depth, is
/// either listed in [`WRITER_SOURCES`] or fails the
/// test.
#[test]
fn the_restore_writer_names_no_release_mirror_or_spawned_task() {
    const FORBIDDEN: [&str; 8] = [
        "enqueue_ext_releases",
        "schedule_old_object_ext_gc",
        "with_ext_mirror_sender",
        "tokio::spawn",
        "tokio::task::spawn(",
        "JoinSet",
        "Handle::spawn",
        ".spawn(",
    ];
    let execute_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/dr/job/execute");
    let on_disk: std::collections::BTreeSet<String> = std::fs::read_dir(&execute_dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", execute_dir.display()))
        .map(|entry| entry.expect("a directory entry").path())
        .inspect(|path| {
            assert!(
                !path.is_dir(),
                "{} is a directory: a module there would escape this scan; list its files in \
                 WRITER_SOURCES and scan them",
                path.display()
            );
        })
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .map(|path| {
            format!(
                "src/dr/job/execute/{}",
                path.file_name()
                    .expect("a file name")
                    .to_str()
                    .expect("a UTF-8 file name")
            )
        })
        .collect();
    let listed: std::collections::BTreeSet<String> = WRITER_SOURCES
        .iter()
        .filter(|(path, _)| path.starts_with("src/dr/job/execute/"))
        .map(|(path, _)| (*path).to_owned())
        .collect();
    assert_eq!(
        on_disk, listed,
        "every .rs file under src/dr/job/execute/ is scanned, and only those"
    );
    for (path, source) in WRITER_SOURCES {
        assert!(
            source.contains("fn "),
            "{path}: fixture: the scan reads the file"
        );
        for needle in FORBIDDEN {
            let lines: Vec<(usize, &str)> = source
                .lines()
                .enumerate()
                .filter(|(_, line)| line.contains(needle))
                .map(|(n, line)| (n + 1, line.trim()))
                .collect();
            assert!(
                lines.is_empty(),
                "{path} names `{needle}`, which the restore writer must not: {lines:?}"
            );
        }
    }
}
