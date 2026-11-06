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

//! The public-wire LiveService conformance suite.
//!
//! The suite checks a running service from outside, as a client: it calls
//! only the published `nvidia.omniverse.live.v1alpha.LiveService` contract
//! (generated from `proto/` by this crate's own `build.rs`) and decodes OODF
//! with `oodf-ffi`'s public API. It depends on no service code, so any
//! implementation of the contract can be tested with it.
//!
//! # Flow of one run ([`main_from_env`])
//!
//! 1. **Command line** ([`parse_args`]): the arguments after the program
//!    name, `live-edit-conformance [--profile <name>]...` or `--help`
//!    anywhere. There is no subcommand word: anything else, a leading
//!    `conformance` included, exits 2.
//! 2. **Interruption** (`runner::interrupt_on_signals`): SIGTERM and SIGINT
//!    are caught from here on, so a Job's deadline or a Ctrl-C still cleans
//!    up and reports. Provisioning and the runner each watch for them. An
//!    interruption while provisioning waits for the token file or sends its
//!    `PUT` skips the cases, still attempts the `DELETE` once the `PUT` was
//!    started, and is the `deployment-fixture` failure below. During the
//!    preflight, the runner stops and writes the `conformance-preflight`
//!    error, with no `summary.json`. During the cases, it stops, closes the
//!    running case's clients within 10 s and writes the reports
//!    (`summary.json`'s `failure` naming the `live-edit-conformance` suite).
//!    The `DELETE` follows either way, and the exit code is 2. Once a signal
//!    has arrived, the `DELETE` waits at most 10 s for its answer, counted
//!    from the later of the signal and the `DELETE`'s start, so the reports
//!    are printed within the pod's 30 s termination grace.
//! 3. **Configuration** (`config::Settings`): every `CONFORMANCE_*`
//!    variable, read once; the process environment is never modified.
//! 4. **Provisioning** (`deployment`), only when
//!    `CONFORMANCE_STORAGE_ENDPOINT` is set: a UUID-named base layer is
//!    created through the Storage API for the run and deleted after it,
//!    whatever the outcome. A failure here, an interruption while it waits
//!    for the token file or sends its `PUT`, or a refused configuration on
//!    this path, is the JUnit suite `deployment-fixture` and exit 2.
//! 5. **Preflight** (`runner`): token file, profile prerequisites, the
//!    synthetic fixtures' pinned hashes, endpoint readiness. A failure is the
//!    JUnit suite `conformance-preflight` and exit 2.
//! 6. **Cases** (`cases`, run by `runner`): every selected case in a fixed
//!    order, one at a time, each with its own client; each case's sessions,
//!    streams and pending calls are cleaned up after it, however it ended.
//! 7. **Reports** (`runner::Reports`): `conformance-results.xml` and
//!    `summary.json` in `CONFORMANCE_REPORT_DIR`, and printed as
//!    `CONFORMANCE_REPORT <name> <base64>` lines when
//!    `CONFORMANCE_REPORT_STDOUT=true`.
//!
//! Exit codes: 0 every selected case passed; 1 a case failed, or its setup
//! or cleanup did, or every case passed but the reports could not be
//! written; 2 the command line, configuration, preflight, provisioning or its
//! cleanup failed, or the run was interrupted.

pub mod cases;
pub mod check;
pub mod client;
pub mod config;
pub mod deployment;
pub mod fixtures;
pub mod proto;
pub mod protocol;
pub mod runner;

use crate::deployment::ProvisionError;
use crate::runner::{Interrupt, Profile, Reports, EXIT_PASSED, EXIT_PREFLIGHT};

/// The `--help` text.
pub const USAGE: &str = "\
Usage: live-edit-conformance [--profile <name>]...
       live-edit-conformance --help

Runs the LiveService conformance suite against a running service, over the
public gRPC contract only. The core cases always run; each --profile adds
its cases:
  deployment     LiveService health is SERVING without credentials
  permissions    a denied base path is refused (needs CONFORMANCE_DENIED_PATH)
  multi_replica  writes through two replicas (needs CONFORMANCE_PEER_ENDPOINT)

Environment:
  CONFORMANCE_GRPC_ENDPOINT     http[s]://host[:port] of the service
                                (default http://localhost:50001; the port
                                defaults to 443 for https, 50001 for http)
  CONFORMANCE_BASE_PATH         required: an existing base layer the
                                credential may read and write; with
                                CONFORMANCE_STORAGE_ENDPOINT, a writable
                                directory to create one in
  CONFORMANCE_AUTH_TOKEN        the bearer token; set exactly one of this and
                                CONFORMANCE_AUTH_TOKEN_FILE
  CONFORMANCE_AUTH_TOKEN_FILE   a file holding the token, or JSON with
                                access_token; read again for every call
  CONFORMANCE_CA_FILE           a PEM CA bundle that replaces the default
                                roots of every TLS connection; refused when
                                no connection uses TLS
  CONFORMANCE_TIMEOUT           seconds per call or streamed delivery
                                (default 15, at most 86400)
  CONFORMANCE_READY_TIMEOUT     seconds to wait for the token file and for
                                the service to become ready (default 120, at
                                most 86400)
  CONFORMANCE_PEER_ENDPOINT     a second replica's endpoint (multi_replica)
  CONFORMANCE_DENIED_PATH       a base path the credential may not read
                                (permissions)
  CONFORMANCE_STORAGE_ENDPOINT  a Storage API URL: create a
                                conformance-<uuid>.usda base layer under
                                CONFORMANCE_BASE_PATH for the run, and delete
                                it after
  CONFORMANCE_REPORT_DIR        where conformance-results.xml and
                                summary.json are written (default reports)
  CONFORMANCE_REPORT_STDOUT     true: also print each report as a line
                                CONFORMANCE_REPORT <name> <base64>
An empty or whitespace-only value counts as unset, except that one in
CONFORMANCE_GRPC_ENDPOINT, CONFORMANCE_TIMEOUT or CONFORMANCE_READY_TIMEOUT
is refused.

Exit codes:
  0  every selected case passed
  1  a case failed, or its setup or cleanup did, or every case passed but
     the reports could not be written
  2  bad arguments, or the configuration, preflight, provisioning or its
     cleanup failed, or the run was interrupted
";

/// What the command line asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// `--help`: print [`USAGE`] and exit 0.
    Help,
    /// Run the suite with these profiles selected, in command-line order
    /// (repeats kept).
    Run(Vec<Profile>),
}

/// Parses the command line after the program name (`argv[1..]`):
/// `--profile <name>` with a selectable profile name (repeatable), or
/// `--help` anywhere. Any other argument is refused, a leading `conformance`
/// word included: the suite is its own executable, not a subcommand. The
/// error names the offending argument.
pub fn parse_args(args: &[String]) -> Result<Command, String> {
    if args.iter().any(|arg| arg == "--help") {
        return Ok(Command::Help);
    }
    let mut profiles = Vec::new();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        if arg != "--profile" {
            return Err(format!("unexpected argument {arg:?}"));
        }
        let name = args
            .next()
            .ok_or_else(|| "--profile needs a profile name".to_string())?;
        let profile = Profile::from_name(name).ok_or_else(|| {
            format!("unknown profile {name:?} (deployment, permissions or multi_replica)")
        })?;
        profiles.push(profile);
    }
    Ok(Command::Run(profiles))
}

/// Runs the suite as the process's command line and environment describe,
/// and returns the process exit code (see the module doc for the flow).
pub async fn main_from_env() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let profiles = match parse_args(&args) {
        Ok(Command::Help) => {
            print!("{USAGE}");
            return EXIT_PASSED;
        }
        Ok(Command::Run(profiles)) => profiles,
        Err(message) => {
            eprintln!("conformance: {message}\n\n{USAGE}");
            return EXIT_PREFLIGHT;
        }
    };

    let env = config::process_env;
    let interrupt = runner::interrupt_on_signals();
    let mut reports = Reports::from_lookup(&env);
    let settings = config::Settings::from_lookup(&env);
    let code = if deployment::requested(&env) {
        provisioned_run(settings, &profiles, &mut reports, interrupt).await
    } else {
        runner::run(settings, &profiles, &mut reports, interrupt).await
    };
    reports.emit();
    code
}

/// The run inside a provisioned base layer. A provisioning or cleanup failure
/// is reported by its kind only: the error text of an HTTP or TLS failure can
/// carry credentials. A refused configuration (`Config`) also carries its
/// `ConfigError` text, which states the rule and, where one applies, names
/// the variable: every `ConfigError` that reaches here (from `config::Settings` and
/// `deployment`) is fixed text or a variable name, never a value, a file's
/// content or an I/O error's text.
async fn provisioned_run(
    settings: Result<config::Settings, config::ConfigError>,
    profiles: &[Profile],
    reports: &mut Reports,
    interrupt: Interrupt,
) -> i32 {
    let result = match settings {
        Err(error) => Err(ProvisionError::from(error)),
        Ok(settings) => {
            let run_reports = &mut *reports;
            deployment::provision(&settings, interrupt.clone(), move |provisioned| {
                runner::run(Ok(provisioned), profiles, run_reports, interrupt)
            })
            .await
        }
    };
    match result {
        Ok(code) => code,
        Err(error) => {
            let message = match &error {
                ProvisionError::Config(config) => {
                    format!(
                        "Provisioning or cleanup failed ({}): {config}",
                        error.kind()
                    )
                }
                _ => format!("Provisioning or cleanup failed ({})", error.kind()),
            };
            reports.record_failure(runner::DEPLOYMENT_SUITE, &message);
            eprintln!("{message}");
            EXIT_PREFLIGHT
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;

    use futures::FutureExt as _;

    use super::*;
    use crate::runner::{RESULTS_FILE, SUMMARY_FILE};

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|arg| arg.to_string()).collect()
    }

    #[test]
    fn parse_args_no_arguments_runs_the_core_cases_only() {
        assert_eq!(parse_args(&[]), Ok(Command::Run(Vec::new())));
    }

    #[test]
    fn parse_args_profiles_keep_command_line_order_and_repeats() {
        assert_eq!(
            parse_args(&args(&[
                "--profile",
                "multi_replica",
                "--profile",
                "deployment",
                "--profile",
                "permissions",
                "--profile",
                "deployment",
            ])),
            Ok(Command::Run(vec![
                Profile::MultiReplica,
                Profile::Deployment,
                Profile::Permissions,
                Profile::Deployment,
            ]))
        );
    }

    #[test]
    fn parse_args_help_anywhere_is_help_whatever_else_is_there() {
        for list in [
            &["--help"][..],
            &["--profile", "deployment", "--help"],
            &["conformance", "--help"],
            &["--help", "--profile", "bogus"],
        ] {
            assert_eq!(parse_args(&args(list)), Ok(Command::Help), "{list:?}");
        }
    }

    #[test]
    fn parse_args_leading_conformance_word_is_refused() {
        // The suite is its own executable: the retired subcommand word is an
        // unexpected argument, never skipped.
        for list in [
            &["conformance"][..],
            &["conformance", "--profile", "deployment"],
        ] {
            let error = parse_args(&args(list)).unwrap_err();
            assert!(error.contains("\"conformance\""), "{list:?}: {error}");
        }
    }

    #[test]
    fn parse_args_anything_but_profile_or_help_is_refused_naming_it() {
        for (list, offending) in [
            (&["deployment"][..], "deployment"),
            (&["--profile=deployment"][..], "--profile=deployment"),
            (&["-p", "deployment"][..], "-p"),
            (&["--profiles", "deployment"][..], "--profiles"),
            (&["--profile", "deployment", "extra"][..], "extra"),
            (&["-k", "test_ping"][..], "-k"),
        ] {
            let error = parse_args(&args(list)).unwrap_err();
            assert!(
                error.contains(&format!("{offending:?}")),
                "{list:?}: {error}"
            );
        }
    }

    #[test]
    fn parse_args_profile_without_a_selectable_name_is_refused() {
        // `core` always runs and is not selectable.
        for list in [
            &["--profile"][..],
            &["--profile", "core"],
            &["--profile", "Deployment"],
            &["--profile", "bogus"],
        ] {
            assert!(parse_args(&args(list)).is_err(), "{list:?}");
        }
    }

    /// The configuration of a provisioned run writing its reports to
    /// `report_dir`. The Storage API endpoint is never contacted by these
    /// tests: nothing listens on the discard port.
    fn provisioned_vars(report_dir: &Path) -> HashMap<&'static str, String> {
        HashMap::from([
            ("CONFORMANCE_AUTH_TOKEN", "local-token".to_string()),
            (
                "CONFORMANCE_BASE_PATH",
                "/container/conformance".to_string(),
            ),
            (
                "CONFORMANCE_STORAGE_ENDPOINT",
                "http://127.0.0.1:9".to_string(),
            ),
            (
                "CONFORMANCE_REPORT_DIR",
                report_dir.to_str().unwrap().to_string(),
            ),
        ])
    }

    /// Asserts the reports hold exactly one suite, `deployment-fixture`, with
    /// one errored case carrying `message`, and a summary of exit code 2
    /// naming that suite.
    fn assert_deployment_fixture_failure(report_dir: &Path, message: &str) {
        let xml = std::fs::read_to_string(report_dir.join(RESULTS_FILE)).unwrap();
        assert_eq!(xml.matches("<testsuite ").count(), 1, "{xml}");
        assert!(
            xml.contains(
                "<testsuite name=\"deployment-fixture\" tests=\"1\" failures=\"0\" errors=\"1\""
            ),
            "{xml}"
        );
        assert!(
            xml.contains(&format!("<error message=\"{message}\">{message}</error>")),
            "{xml}"
        );
        let summary: serde_json::Value =
            serde_json::from_slice(&std::fs::read(report_dir.join(SUMMARY_FILE)).unwrap()).unwrap();
        assert_eq!(
            summary,
            serde_json::json!({"exit_code": 2, "failure": "deployment-fixture"})
        );
    }

    #[tokio::test]
    async fn provisioned_run_interrupted_before_the_suite_is_deployment_fixture_exit_2() {
        let dir = tempfile::tempdir().unwrap();
        let vars = provisioned_vars(dir.path());
        let lookup = |name: &str| vars.get(name).cloned();
        let mut reports = Reports::from_lookup(&lookup);
        // Already resolved: SIGTERM arrived before provisioning finished.
        let interrupt = futures::future::ready(()).boxed().shared();

        let code = provisioned_run(
            config::Settings::from_lookup(&lookup),
            &[],
            &mut reports,
            interrupt,
        )
        .await;

        assert_eq!(code, EXIT_PREFLIGHT);
        assert_deployment_fixture_failure(
            dir.path(),
            "Provisioning or cleanup failed (Interrupted)",
        );
    }

    #[tokio::test]
    async fn provisioned_run_refused_configuration_is_deployment_fixture_with_its_message() {
        // A refused configuration adds its `ConfigError` text to the kind;
        // every other kind stays kind-only (the `Interrupted` test above, and
        // `HttpStatus` in the harness's `binary` tests).
        let dir = tempfile::tempdir().unwrap();
        let vars = provisioned_vars(dir.path());
        let lookup = |name: &str| vars.get(name).cloned();
        let mut reports = Reports::from_lookup(&lookup);
        let never = futures::future::pending::<()>().boxed().shared();
        let refused = Err(config::ConfigError::Invalid(
            "CONFORMANCE_TIMEOUT must be finite and positive".to_string(),
        ));

        let code = provisioned_run(refused, &[], &mut reports, never).await;

        assert_eq!(code, EXIT_PREFLIGHT);
        assert_deployment_fixture_failure(
            dir.path(),
            "Provisioning or cleanup failed (Config): CONFORMANCE_TIMEOUT must be finite and positive",
        );
    }
}
