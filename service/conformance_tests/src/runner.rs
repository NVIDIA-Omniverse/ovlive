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

//! The case registry and the run: profile selection, preflight, sequential
//! execution, reports and exit codes.
//!
//! - **Registry.** Every case is a [`Case`] from `crate::cases::all`, in the
//!   order the Python suite's pytest collected them. Its id is
//!   `<file>::<name>` ([`Case::id`]): `file` the case module (`layers` for
//!   `test_layers.py`), `name` the Python test function's name, with
//!   `[<param>]` for a parametrized case. There is no skip: a selected case
//!   runs, and passes or fails.
//! - **Selection** ([`selected`]). [`Profile::Core`] cases always run; a case
//!   of another profile runs only when `--profile` selected it.
//! - **Preflight.** The configuration, the token file (waited for), each
//!   selected profile's prerequisite (`CONFORMANCE_DENIED_PATH` for
//!   `permissions`, a distinct `CONFORMANCE_PEER_ENDPOINT` for
//!   `multi_replica`), the synthetic fixtures (`crate::fixtures::verify`) and
//!   an in-process decode of `initial`, and readiness of each endpoint (the
//!   peer's too whenever one is set): the `grpc.health.v1` check of
//!   LiveService, or an authenticated `Ping` where health is
//!   `UNIMPLEMENTED`, retried on `UNAVAILABLE` and `DEADLINE_EXCEEDED` until
//!   the readiness timeout. A preflight failure is the JUnit suite
//!   `conformance-preflight` and exit [`EXIT_PREFLIGHT`].
//! - **Execution** ([`execute`]). One case at a time, each with a fresh
//!   [`Ctx`]: a new [`Client`], a new session when [`Case::needs_session`],
//!   and a peer client for [`Profile::MultiReplica`] cases. Each case runs
//!   under [`CASE_TIMEOUT`]; a timeout or a panic fails that case only, and
//!   the run continues. The clients are closed after every case, however it
//!   ended: a setup or cleanup failure is the case's JUnit `<error>`, a failed
//!   case body its `<failure>`. One progress line per case goes to stdout.
//! - **Interruption.** SIGTERM or SIGINT ([`interrupt_on_signals`]) cancels
//!   the running case, closes its clients within [`INTERRUPT_CLEANUP_BOUND`]
//!   (under the pod's 30 s termination grace, leaving room for the
//!   provisioned layer's deletion, `crate::deployment::INTERRUPT_DELETE_BOUND`),
//!   writes the reports with `summary.json`'s `failure` naming [`RUN_SUITE`],
//!   and exits [`EXIT_PREFLIGHT`]. The remaining cases do not run.
//! - **Reports** ([`Reports`]): `conformance-results.xml` (JUnit) and
//!   `summary.json` (`profiles`, `executed`, `skipped`,
//!   `unexpected_deselections`, `exit_code`, `build`, and `failure` when an
//!   interruption or a provisioning or cleanup failure set exit 2) in the report
//!   directory, and with `CONFORMANCE_REPORT_STDOUT=true` one
//!   `CONFORMANCE_REPORT <name> <base64>` line per file.

use std::any::Any;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use futures::future::{BoxFuture, Shared};
use futures::FutureExt as _;
use serde::Serialize;
use tokio::time::Instant;
use tonic::Code;
use tonic_health::pb::health_check_response::ServingStatus;

use crate::check::{CaseError, CaseResult};
use crate::client::{CallError, Client, LIVE_SERVICE_NAME};
use crate::config::{self, ConfigError, Lookup, Settings};
use crate::fixtures::{self, Fixture};
use crate::{proto, protocol};

/// Exit code: every selected case passed.
pub const EXIT_PASSED: i32 = 0;
/// Exit code: a case failed or errored (or, impossibly, none ran), or every
/// case passed but the reports could not be written.
pub const EXIT_FAILED: i32 = 1;
/// Exit code: the command line, the preflight, provisioning or its cleanup
/// failed, or the run was interrupted.
pub const EXIT_PREFLIGHT: i32 = 2;

/// The longest one case may run, its setup and cleanup excluded.
pub const CASE_TIMEOUT: Duration = Duration::from_secs(120);

/// How long an interrupted run spends closing the running case's clients
/// (cancelling its calls and deleting its sessions) before it writes the
/// reports. With the provisioned base layer's `DELETE` after it
/// (`crate::deployment::INTERRUPT_DELETE_BOUND`), under the pod's 30 s
/// termination grace.
pub const INTERRUPT_CLEANUP_BOUND: Duration = Duration::from_secs(10);

/// The JUnit suite of a preflight failure.
pub const PREFLIGHT_SUITE: &str = "conformance-preflight";

/// The JUnit suite of a provisioning or cleanup failure (`crate::deployment`).
pub const DEPLOYMENT_SUITE: &str = "deployment-fixture";

/// The JUnit suite holding the executed cases.
pub const RUN_SUITE: &str = "live-edit-conformance";

/// The JUnit report's file name in the report directory.
pub const RESULTS_FILE: &str = "conformance-results.xml";

/// The run summary's file name in the report directory.
pub const SUMMARY_FILE: &str = "summary.json";

/// How often readiness is polled while the service is not yet ready.
const READY_POLL: Duration = Duration::from_millis(250);

/// A case's profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Profile {
    /// Always runs.
    Core,
    /// `--profile deployment`: a deployed service's unauthenticated health.
    Deployment,
    /// `--profile permissions`: needs `CONFORMANCE_DENIED_PATH`.
    Permissions,
    /// `--profile multi_replica`: needs a distinct `CONFORMANCE_PEER_ENDPOINT`.
    MultiReplica,
}

impl Profile {
    /// The profiles `--profile` can select.
    pub const SELECTABLE: [Profile; 3] = [
        Profile::Deployment,
        Profile::Permissions,
        Profile::MultiReplica,
    ];

    /// The profile's name: `core`, `deployment`, `permissions` or
    /// `multi_replica`.
    pub fn name(self) -> &'static str {
        match self {
            Profile::Core => "core",
            Profile::Deployment => "deployment",
            Profile::Permissions => "permissions",
            Profile::MultiReplica => "multi_replica",
        }
    }

    /// The selectable profile named `name`; `None` for `core` or an unknown
    /// name.
    pub fn from_name(name: &str) -> Option<Profile> {
        Self::SELECTABLE
            .into_iter()
            .find(|profile| profile.name() == name)
    }
}

/// What a case's run function returns: the case body, borrowing its [`Ctx`].
pub type CaseFuture<'a> = Pin<Box<dyn Future<Output = CaseResult> + Send + 'a>>;

/// A case's body, called once with its [`Ctx`].
pub type CaseFn = Box<dyn for<'a> Fn(&'a mut Ctx) -> CaseFuture<'a> + Send + Sync>;

/// One registered case. Write the body as
/// `fn test_name(ctx: &mut Ctx) -> CaseFuture<'_> { Box::pin(async move { … }) }`
/// (or with a second, parameter argument) and register it with
/// [`Case::new`] or [`Case::with_param`].
pub struct Case {
    /// The case module: `chained`, `layers`, `profiles`, `sessions` or
    /// `streams`.
    pub file: &'static str,
    /// The Python test function's name, with `[<param>]` for a parametrized
    /// case.
    pub name: String,
    /// The profile that selects the case.
    pub profile: Profile,
    /// Whether the runner creates a session for the case ([`Ctx::session`])
    /// before running it; a failure to create it is the case's `<error>`.
    pub needs_session: bool,
    /// The body.
    pub run: CaseFn,
}

impl Case {
    /// A case named `function`.
    pub fn new(
        file: &'static str,
        function: &'static str,
        profile: Profile,
        needs_session: bool,
        run: for<'a> fn(&'a mut Ctx) -> CaseFuture<'a>,
    ) -> Case {
        Case {
            file,
            name: function.to_string(),
            profile,
            needs_session,
            run: erase(run),
        }
    }

    /// One parametrization of `function`, named `function[param_id]`, whose
    /// body is called with `param`.
    pub fn with_param<P>(
        file: &'static str,
        function: &'static str,
        param_id: &str,
        profile: Profile,
        needs_session: bool,
        param: P,
        run: for<'a> fn(&'a mut Ctx, P) -> CaseFuture<'a>,
    ) -> Case
    where
        P: Clone + Send + Sync + 'static,
    {
        Case {
            file,
            name: format!("{function}[{param_id}]"),
            profile,
            needs_session,
            run: erase(move |ctx| run(ctx, param.clone())),
        }
    }

    /// The case id, `<file>::<name>`: its JUnit test case name.
    pub fn id(&self) -> String {
        format!("{}::{}", self.file, self.name)
    }
}

/// Boxes a run function; the bound gives a closure its higher-ranked
/// signature.
fn erase<F>(run: F) -> CaseFn
where
    F: for<'a> Fn(&'a mut Ctx) -> CaseFuture<'a> + Send + Sync + 'static,
{
    Box::new(run)
}

/// What one case runs with. The runner builds it before the case and closes
/// its clients after it, however the case ended.
pub struct Ctx {
    /// The run's settings.
    pub settings: Arc<Settings>,
    /// The case's client for [`Settings::endpoint`].
    pub client: Client,
    /// The session the runner created for a [`Case::needs_session`] case,
    /// owned by [`Ctx::client`]; empty otherwise.
    pub session: String,
    peer: Option<Client>,
}

impl Ctx {
    /// The case's client for [`Settings::peer_endpoint`]: present for
    /// [`Profile::MultiReplica`] cases, a failure for any other.
    pub fn peer(&self) -> CaseResult<&Client> {
        self.peer
            .as_ref()
            .ok_or_else(|| CaseError::new("no peer client: only multi_replica cases have one"))
    }
}

/// Resolves once SIGTERM or SIGINT has arrived; cloned to every party that
/// must react ([`run`] and `crate::deployment::provision`).
pub type Interrupt = Shared<BoxFuture<'static, ()>>;

/// Installs the SIGTERM and SIGINT handlers (from then on neither signal
/// kills the process) and returns the [`Interrupt`] they resolve. Call it
/// once, inside the runtime, before any network work.
pub fn interrupt_on_signals() -> Interrupt {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        // Both handlers are installed here, not when the future is first
        // polled, so a signal that arrives before then is still caught.
        match (
            signal(SignalKind::terminate()),
            signal(SignalKind::interrupt()),
        ) {
            (Ok(mut terminate), Ok(mut interrupt)) => async move {
                tokio::select! {
                    _ = terminate.recv() => {}
                    _ = interrupt.recv() => {}
                }
            }
            .boxed()
            .shared(),
            _ => {
                eprintln!(
                    "conformance: cannot install the SIGTERM and SIGINT handlers; \
                     an interrupted run will not clean up or write its reports"
                );
                futures::future::pending::<()>().boxed().shared()
            }
        }
    }
    #[cfg(not(unix))]
    {
        async {
            if tokio::signal::ctrl_c().await.is_err() {
                futures::future::pending::<()>().await;
            }
        }
        .boxed()
        .shared()
    }
}

/// The `summary.json` of a run. A key that is `None` is left out: a run that
/// never reached its cases records only `exit_code` and `failure`.
#[derive(Clone, Debug, Default, Serialize)]
struct Summary {
    #[serde(skip_serializing_if = "Option::is_none")]
    profiles: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    executed: Option<usize>,
    /// Always 0: the runner has no skip.
    #[serde(skip_serializing_if = "Option::is_none")]
    skipped: Option<usize>,
    /// Always 0: only the profile selection deselects a case.
    #[serde(skip_serializing_if = "Option::is_none")]
    unexpected_deselections: Option<usize>,
    exit_code: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    build: Option<serde_json::Value>,
    /// The JUnit suite of the failure that set `exit_code` to 2: [`RUN_SUITE`]
    /// when SIGTERM or SIGINT cut the cases short (the interrupted case's
    /// `<error>` is there), or the suite [`Reports::record_failure`] added
    /// after the run, which replaces it.
    #[serde(skip_serializing_if = "Option::is_none")]
    failure: Option<String>,
}

/// One JUnit `<testcase>`.
#[derive(Clone, Debug)]
struct CaseReport {
    name: String,
    classname: Option<String>,
    seconds: f64,
    /// The case body's failure: `<failure>`.
    failure: Option<String>,
    /// Setup, cleanup and interruption failures: one `<error>` each.
    errors: Vec<String>,
}

impl CaseReport {
    fn new(name: String, classname: Option<String>) -> Self {
        Self {
            name,
            classname,
            seconds: 0.0,
            failure: None,
            errors: Vec::new(),
        }
    }

    fn passed(&self) -> bool {
        self.failure.is_none() && self.errors.is_empty()
    }
}

/// One JUnit `<testsuite>`.
#[derive(Clone, Debug)]
struct SuiteReport {
    name: String,
    seconds: f64,
    cases: Vec<CaseReport>,
}

impl SuiteReport {
    /// A suite of one test case named `case` that errored with `message`.
    fn single_error(suite: &str, case: &str, message: &str) -> Self {
        let mut report = CaseReport::new(case.to_string(), None);
        report.errors.push(message.to_string());
        Self {
            name: suite.to_string(),
            seconds: 0.0,
            cases: vec![report],
        }
    }
}

/// `text` escaped for an XML attribute value or text content: `<`, `>`, `&`,
/// `"` and `'` become entity references; tab, newline and carriage return
/// character references (so an attribute keeps them); DEL and the C1
/// controls character references; and every character XML 1.0 cannot carry
/// at all (the other C0 controls, U+FFFE and U+FFFF) the visible text
/// `\uXXXX`.
pub fn escape_xml(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            '\t' | '\n' | '\r' | '\u{7f}'..='\u{9f}' => {
                out.push_str(&format!("&#x{:x};", c as u32));
            }
            '\u{0}'..='\u{1f}' | '\u{fffe}' | '\u{ffff}' => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// The JUnit document of `suites`.
fn junit_xml(suites: &[SuiteReport]) -> String {
    let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<testsuites>\n");
    for suite in suites {
        let failures = suite.cases.iter().filter(|c| c.failure.is_some()).count();
        let errors = suite.cases.iter().filter(|c| !c.errors.is_empty()).count();
        xml.push_str(&format!(
            "  <testsuite name=\"{}\" tests=\"{}\" failures=\"{failures}\" errors=\"{errors}\" skipped=\"0\" time=\"{:.3}\">\n",
            escape_xml(&suite.name),
            suite.cases.len(),
            suite.seconds,
        ));
        for case in &suite.cases {
            let classname = case
                .classname
                .as_deref()
                .map(|classname| format!(" classname=\"{}\"", escape_xml(classname)))
                .unwrap_or_default();
            xml.push_str(&format!(
                "    <testcase{classname} name=\"{}\" time=\"{:.3}\"",
                escape_xml(&case.name),
                case.seconds,
            ));
            if case.passed() {
                xml.push_str("/>\n");
                continue;
            }
            xml.push_str(">\n");
            let outcomes = case
                .failure
                .iter()
                .map(|message| ("failure", message))
                .chain(case.errors.iter().map(|message| ("error", message)));
            for (element, message) in outcomes {
                let message = escape_xml(message);
                xml.push_str(&format!(
                    "      <{element} message=\"{message}\">{message}</{element}>\n"
                ));
            }
            xml.push_str("    </testcase>\n");
        }
        xml.push_str("  </testsuite>\n");
    }
    xml.push_str("</testsuites>\n");
    xml
}

/// The report files of one run.
pub struct Reports {
    dir: PathBuf,
    stdout: bool,
    suites: Vec<SuiteReport>,
    summary: Option<Summary>,
}

impl Reports {
    /// The reports in `CONFORMANCE_REPORT_DIR`, printed to stdout too when
    /// `CONFORMANCE_REPORT_STDOUT` is `true` (`crate::config::report_dir`,
    /// `crate::config::report_stdout`). Read on their own, so a refused
    /// configuration is still reported.
    pub fn from_lookup(lookup: Lookup<'_>) -> Reports {
        Reports {
            dir: config::report_dir(lookup),
            stdout: config::report_stdout(lookup),
            suites: Vec::new(),
            summary: None,
        }
    }

    /// Adds a JUnit suite `suite` holding one errored test case with
    /// `message`, keeping what the file already holds, and sets
    /// `summary.json`'s `exit_code` to 2 and `failure` to `suite`. `message`
    /// must contain no credential or error text that could.
    ///
    /// What the file already holds is what this run wrote to it: the reports
    /// keep their suites in memory and rewrite both files whole.
    pub fn record_failure(&mut self, suite: &str, message: &str) {
        self.suites
            .push(SuiteReport::single_error(suite, suite, message));
        let summary = self.summary.get_or_insert_with(Summary::default);
        summary.exit_code = EXIT_PREFLIGHT;
        summary.failure = Some(suite.to_string());
        self.write();
    }

    /// Prints each report file as a `CONFORMANCE_REPORT <name> <base64>` line
    /// when enabled; nothing otherwise.
    pub fn emit(&self) {
        if !self.stdout {
            return;
        }
        for name in [RESULTS_FILE, SUMMARY_FILE] {
            let path = self.dir.join(name);
            if !path.is_file() {
                continue;
            }
            match std::fs::read(&path) {
                Ok(bytes) => println!(
                    "CONFORMANCE_REPORT {name} {}",
                    base64::engine::general_purpose::STANDARD.encode(bytes)
                ),
                Err(error) => {
                    eprintln!("conformance: cannot read {}: {error}", path.display())
                }
            }
        }
    }

    /// Writes the JUnit report, and the summary once there is one, into the
    /// report directory. A failure is printed to stderr; `false` then.
    fn write(&self) -> bool {
        let written = std::fs::create_dir_all(&self.dir)
            .and_then(|()| std::fs::write(self.dir.join(RESULTS_FILE), junit_xml(&self.suites)))
            .and_then(|()| match &self.summary {
                None => Ok(()),
                Some(summary) => {
                    let mut json =
                        serde_json::to_string_pretty(summary).expect("a summary always serializes");
                    json.push('\n');
                    std::fs::write(self.dir.join(SUMMARY_FILE), json)
                }
            });
        match written {
            Ok(()) => true,
            Err(error) => {
                eprintln!(
                    "conformance: cannot write the reports to {}: {error}",
                    self.dir.display()
                );
                false
            }
        }
    }

    /// Writes the reports for the run's `code`: `code`, or
    /// [`EXIT_FAILED`] when a passing run's reports could not be written.
    fn write_for(&self, code: i32) -> i32 {
        if self.write() {
            code
        } else {
            code.max(EXIT_FAILED)
        }
    }
}

/// The JUnit message of a run or case cut short by SIGTERM or SIGINT.
const INTERRUPTED: &str = "interrupted by SIGTERM or SIGINT";

/// Runs the preflight and then every selected case (see the module doc),
/// writing the reports. `settings` is the configuration as read, a refusal
/// being a preflight failure; `profiles` the selected profiles in command-line
/// order (repeats kept, for `summary.json`). Returns the exit code.
pub async fn run(
    settings: Result<Settings, ConfigError>,
    profiles: &[Profile],
    reports: &mut Reports,
    interrupt: Interrupt,
) -> i32 {
    run_with_cases(settings, crate::cases::all(), profiles, reports, interrupt).await
}

/// [`run`] with `cases` in place of the registry: the preflight, readiness
/// included, then [`execute`] of `cases`. For the crate's integration tests,
/// which drive the preflight against a fake server without the real cases;
/// not part of the suite's interface.
#[doc(hidden)]
pub async fn run_with_cases(
    settings: Result<Settings, ConfigError>,
    cases: Vec<Case>,
    profiles: &[Profile],
    reports: &mut Reports,
    interrupt: Interrupt,
) -> i32 {
    match preflight(settings, profiles, &interrupt).await {
        Ok(settings) => execute(settings, cases, profiles, reports, interrupt).await,
        Err(message) => {
            // The message never carries RPC error text, which can hold a
            // credential: a status is reported by its code only.
            reports.suites.push(SuiteReport::single_error(
                PREFLIGHT_SUITE,
                "preflight",
                &message,
            ));
            eprintln!("Conformance preflight failed: {message}");
            reports.write_for(EXIT_PREFLIGHT)
        }
    }
}

/// The preflight: the settings to run with, or the failure's message.
async fn preflight(
    settings: Result<Settings, ConfigError>,
    profiles: &[Profile],
    interrupt: &Interrupt,
) -> Result<Arc<Settings>, String> {
    let settings = Arc::new(settings.map_err(|error| error.to_string())?);
    let checks = async {
        settings
            .wait_for_token()
            .await
            .map_err(|error| match error {
                ConfigError::TokenFileMissing => {
                    "Token file did not appear within the readiness deadline".to_string()
                }
                other => other.to_string(),
            })?;
        if profiles.contains(&Profile::Permissions)
            && settings.denied_path.as_deref().is_none_or(str::is_empty)
        {
            return Err("permissions profile requires CONFORMANCE_DENIED_PATH".to_string());
        }
        if profiles.contains(&Profile::MultiReplica)
            && settings
                .peer_endpoint
                .as_deref()
                .is_none_or(|peer| peer.is_empty() || peer == settings.endpoint)
        {
            return Err(
                "multi_replica profile requires a distinct CONFORMANCE_PEER_ENDPOINT".to_string(),
            );
        }
        // The fixtures every case sends, and the decoder every catch-up check
        // uses, work before any case relies on them.
        fixtures::verify()?;
        protocol::decode(fixtures::fixture(Fixture::Initial))
            .map_err(|error| error.message().to_string())?;
        for endpoint in endpoints(&settings) {
            wait_ready(&settings, &endpoint).await?;
        }
        Ok(())
    };
    tokio::select! {
        biased;
        _ = interrupt.clone() => Err(INTERRUPTED.to_string()),
        checked = checks => checked.map(|()| settings.clone()),
    }
}

/// The endpoints the preflight waits for: the endpoint, and the peer endpoint
/// when one is set and differs.
fn endpoints(settings: &Settings) -> Vec<String> {
    let mut endpoints = vec![settings.endpoint.clone()];
    if let Some(peer) = settings.peer_endpoint.as_deref() {
        if !peer.is_empty() && peer != settings.endpoint {
            endpoints.push(peer.to_string());
        }
    }
    endpoints
}

/// Waits until LiveService at `endpoint` is ready (see the module doc).
async fn wait_ready(settings: &Arc<Settings>, endpoint: &str) -> Result<(), String> {
    let client =
        Client::new(settings.clone(), Some(endpoint)).map_err(|error| error.to_string())?;
    let ready = poll_ready(&client, settings, endpoint).await;
    // The readiness client owns no session; closing it only ends its calls.
    let closed = client.close().await.map_err(|error| error.to_string());
    ready.and(closed)
}

/// The readiness loop of [`wait_ready`].
async fn poll_ready(client: &Client, settings: &Settings, endpoint: &str) -> Result<(), String> {
    let deadline = Instant::now() + settings.ready_timeout;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(format!("LiveService did not become SERVING at {endpoint}"));
        }
        let per_call = remaining.min(settings.timeout);
        match client.health(LIVE_SERVICE_NAME, per_call).await {
            Ok(response) if response.status == ServingStatus::Serving as i32 => return Ok(()),
            Ok(_) => {}
            // Alternative implementations need not implement gRPC health: an
            // authenticated Ping stands in.
            Err(CallError::Status(status)) if status.code() == Code::Unimplemented => {
                return match client
                    .ping_within(proto::PingRequest::default(), per_call)
                    .await
                {
                    Ok(_) => Ok(()),
                    Err(CallError::Status(status)) => Err(code_name(status.code()).to_string()),
                    Err(CallError::Local(message)) => Err(message),
                };
            }
            Err(CallError::Status(status))
                if matches!(status.code(), Code::Unavailable | Code::DeadlineExceeded) => {}
            Err(CallError::Status(status)) => {
                return Err(format!(
                    "Readiness failed with {}",
                    code_name(status.code())
                ))
            }
            Err(CallError::Local(message)) => return Err(message),
        }
        tokio::time::sleep(READY_POLL.min(deadline.saturating_duration_since(Instant::now())))
            .await;
    }
}

/// A status code's canonical gRPC name, `UNAVAILABLE` style, as the Python
/// suite reported it.
fn code_name(code: Code) -> &'static str {
    match code {
        Code::Ok => "OK",
        Code::Cancelled => "CANCELLED",
        Code::Unknown => "UNKNOWN",
        Code::InvalidArgument => "INVALID_ARGUMENT",
        Code::DeadlineExceeded => "DEADLINE_EXCEEDED",
        Code::NotFound => "NOT_FOUND",
        Code::AlreadyExists => "ALREADY_EXISTS",
        Code::PermissionDenied => "PERMISSION_DENIED",
        Code::ResourceExhausted => "RESOURCE_EXHAUSTED",
        Code::FailedPrecondition => "FAILED_PRECONDITION",
        Code::Aborted => "ABORTED",
        Code::OutOfRange => "OUT_OF_RANGE",
        Code::Unimplemented => "UNIMPLEMENTED",
        Code::Internal => "INTERNAL",
        Code::Unavailable => "UNAVAILABLE",
        Code::DataLoss => "DATA_LOSS",
        Code::Unauthenticated => "UNAUTHENTICATED",
    }
}

/// The cases a run with `profiles` selected executes, in registry order:
/// every [`Profile::Core`] case and each case of a selected profile.
pub fn selected(cases: Vec<Case>, profiles: &[Profile]) -> Vec<Case> {
    cases
        .into_iter()
        .filter(|case| case.profile == Profile::Core || profiles.contains(&case.profile))
        .collect()
}

/// Runs the [`selected`] `cases` one at a time (see the module doc), prints a
/// progress line per case and a closing tally, and writes the reports.
/// Returns the exit code: [`EXIT_PASSED`] when every selected case ran and
/// passed and the reports were written, [`EXIT_PREFLIGHT`] when the run was
/// interrupted, otherwise [`EXIT_FAILED`]. The preflight is [`run`]'s; this
/// is the part after it.
pub async fn execute(
    settings: Arc<Settings>,
    cases: Vec<Case>,
    profiles: &[Profile],
    reports: &mut Reports,
    interrupt: Interrupt,
) -> i32 {
    let cases = selected(cases, profiles);
    let total = cases.len();
    let started = Instant::now();
    let mut profile_names = vec![Profile::Core.name().to_string()];
    profile_names.extend(profiles.iter().map(|profile| profile.name().to_string()));
    println!(
        "conformance: {total} cases selected (profiles: {})",
        profile_names.join(", ")
    );

    let mut results = Vec::with_capacity(total);
    let mut executed = 0;
    let mut interrupted = false;
    for (index, case) in cases.iter().enumerate() {
        let outcome = run_case(&settings, case, &interrupt).await;
        executed += usize::from(outcome.body_ran);
        print_progress(&outcome.report, index + 1, total);
        results.push(outcome.report);
        if outcome.interrupted {
            interrupted = true;
            break;
        }
    }

    let passed = results.iter().filter(|case| case.passed()).count();
    let failed = results.iter().filter(|case| case.failure.is_some()).count();
    let errored = results
        .iter()
        .filter(|case| !case.errors.is_empty())
        .count();
    let seconds = started.elapsed().as_secs_f64();
    for case in results.iter().filter(|case| !case.passed()) {
        let outcome = if case.failure.is_some() {
            "FAILED"
        } else {
            "ERROR"
        };
        let message = case.failure.iter().chain(&case.errors).next();
        let first_line = message.and_then(|m| m.lines().next()).unwrap_or_default();
        println!("{outcome} {} - {first_line}", case.name);
    }
    println!("conformance: {passed} passed, {failed} failed, {errored} errors in {seconds:.2}s");

    let code = if interrupted {
        eprintln!(
            "Conformance run {INTERRUPTED}: {} of {total} cases ran",
            results.len()
        );
        EXIT_PREFLIGHT
    } else if passed == total && executed == total && total > 0 {
        EXIT_PASSED
    } else {
        // A failed or errored case, or (impossible without a skip) a selected
        // case that did not run, or an empty run: never a pass.
        EXIT_FAILED
    };

    reports.suites.push(SuiteReport {
        name: RUN_SUITE.to_string(),
        seconds,
        cases: results,
    });
    reports.summary = Some(Summary {
        profiles: Some(profile_names),
        executed: Some(executed),
        skipped: Some(0),
        unexpected_deselections: Some(0),
        exit_code: code,
        build: Some(fixtures::build_manifest()),
        failure: interrupted.then(|| RUN_SUITE.to_string()),
    });
    reports.write_for(code)
}

/// How one case ended.
struct CaseOutcome {
    report: CaseReport,
    /// Whether the case body was started (its setup passed).
    body_ran: bool,
    /// Whether SIGTERM or SIGINT arrived during the case.
    interrupted: bool,
}

/// Prints the pytest-style progress line of a finished case, and its failure
/// and error messages indented below it.
fn print_progress(report: &CaseReport, done: usize, total: usize) {
    let outcome = if report.failure.is_some() {
        "FAILED"
    } else if !report.errors.is_empty() {
        "ERROR"
    } else {
        "PASSED"
    };
    let percent = done * 100 / total.max(1);
    println!("{} {outcome} [{percent:>3}%]", report.name);
    for message in report.failure.iter().chain(&report.errors) {
        for line in message.lines() {
            println!("    {line}");
        }
    }
}

/// The text of a caught panic.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| message.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic without a message".to_string())
}

/// Sets up, runs and cleans up one case.
async fn run_case(settings: &Arc<Settings>, case: &Case, interrupt: &Interrupt) -> CaseOutcome {
    let started = Instant::now();
    let mut report = CaseReport::new(case.id(), Some(case.file.to_string()));
    let mut body_ran = false;
    let mut interrupted = false;

    let client = match Client::new(settings.clone(), None) {
        Ok(client) => client,
        Err(error) => {
            report
                .errors
                .push(format!("setup: creating the client failed: {error}"));
            report.seconds = started.elapsed().as_secs_f64();
            return CaseOutcome {
                report,
                body_ran,
                interrupted,
            };
        }
    };
    let mut setup_error = None;
    let peer = if case.profile == Profile::MultiReplica {
        match Client::new(settings.clone(), settings.peer_endpoint.as_deref()) {
            Ok(peer) => Some(peer),
            Err(error) => {
                setup_error = Some(format!("setup: creating the peer client failed: {error}"));
                None
            }
        }
    } else {
        None
    };
    let mut ctx = Ctx {
        settings: settings.clone(),
        client,
        session: String::new(),
        peer,
    };

    if setup_error.is_none() && case.needs_session {
        let created = {
            let create = AssertUnwindSafe(ctx.client.session()).catch_unwind();
            tokio::select! {
                biased;
                _ = interrupt.clone() => None,
                created = create => Some(created),
            }
        };
        match created {
            None => {
                interrupted = true;
                setup_error = Some(format!("setup: {INTERRUPTED}"));
            }
            Some(Ok(Ok(session))) => ctx.session = session,
            Some(Ok(Err(error))) => {
                setup_error = Some(format!("setup: creating the session failed: {error}"));
            }
            Some(Err(panic)) => {
                setup_error = Some(format!(
                    "setup: creating the session panicked: {}",
                    panic_message(panic.as_ref())
                ));
            }
        }
    }

    match setup_error {
        Some(error) => report.errors.push(error),
        None => {
            body_ran = true;
            let ended = {
                let body = AssertUnwindSafe((case.run)(&mut ctx)).catch_unwind();
                tokio::select! {
                    biased;
                    _ = interrupt.clone() => None,
                    ended = tokio::time::timeout(CASE_TIMEOUT, body) => Some(ended),
                }
            };
            match ended {
                None => {
                    interrupted = true;
                    report.errors.push(INTERRUPTED.to_string());
                }
                Some(Err(_elapsed)) => {
                    report.failure =
                        Some(format!("case timed out after {} s", CASE_TIMEOUT.as_secs()));
                }
                Some(Ok(Err(panic))) => {
                    report.failure =
                        Some(format!("case panicked: {}", panic_message(panic.as_ref())));
                }
                Some(Ok(Ok(Err(error)))) => report.failure = Some(error.message().to_string()),
                Some(Ok(Ok(Ok(())))) => {}
            }
        }
    }

    interrupted |= cleanup(ctx, interrupt, interrupted, &mut report.errors).await;
    report.seconds = started.elapsed().as_secs_f64();
    CaseOutcome {
        report,
        body_ran,
        interrupted,
    }
}

/// Closes a case's clients (cancelling its calls, deleting its sessions),
/// adding each failure to `errors`. Unbounded unless the run is interrupted,
/// before or during the cleanup: then within [`INTERRUPT_CLEANUP_BOUND`]. An
/// interruption that arrives during the cleanup is one more error,
/// `cleanup: interrupted by SIGTERM or SIGINT`, or, when the cleanup does not
/// finish within the bound, the error saying so. Returns whether the run is
/// interrupted.
async fn cleanup(
    ctx: Ctx,
    interrupt: &Interrupt,
    interrupted: bool,
    errors: &mut Vec<String>,
) -> bool {
    let Ctx { client, peer, .. } = ctx;
    let close = async move {
        let mut failures = Vec::new();
        let clients =
            std::iter::once(("cleanup", client)).chain(peer.map(|peer| ("peer cleanup", peer)));
        for (label, client) in clients {
            match AssertUnwindSafe(client.close()).catch_unwind().await {
                Ok(Ok(())) => {}
                Ok(Err(error)) => failures.push(format!("{label}: {error}")),
                Err(panic) => failures.push(format!(
                    "{label} panicked: {}",
                    panic_message(panic.as_ref())
                )),
            }
        }
        failures
    };
    let mut close = std::pin::pin!(close);

    let interrupted_before = interrupted;
    let mut interrupted = interrupted;
    let finished = if interrupted {
        None
    } else {
        tokio::select! {
            biased;
            failures = &mut close => Some(failures),
            _ = interrupt.clone() => None,
        }
    };
    let failures = match finished {
        Some(failures) => Some(failures),
        None => {
            interrupted = true;
            tokio::time::timeout(INTERRUPT_CLEANUP_BOUND, &mut close)
                .await
                .ok()
        }
    };
    match failures {
        Some(failures) => {
            // An interruption that arrived only now, during a cleanup that
            // then finished, is still recorded on the case, so the JUnit
            // suite `summary.json`'s `failure` names holds it.
            if interrupted && !interrupted_before {
                errors.push(format!("cleanup: {INTERRUPTED}"));
            }
            errors.extend(failures);
        }
        None => errors.push(format!(
            "cleanup: not finished within {} s of the interruption",
            INTERRUPT_CLEANUP_BOUND.as_secs()
        )),
    }
    interrupted
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;

    use super::*;

    /// Reports written to `dir`, not printed.
    fn reports_in(dir: &Path) -> Reports {
        Reports {
            dir: dir.to_path_buf(),
            stdout: false,
            suites: Vec::new(),
            summary: None,
        }
    }

    /// An interrupt that never fires.
    fn never() -> Interrupt {
        futures::future::pending::<()>().boxed().shared()
    }

    /// An interrupt that has already fired.
    fn fired() -> Interrupt {
        futures::future::ready(()).boxed().shared()
    }

    fn passes(_ctx: &mut Ctx) -> CaseFuture<'_> {
        Box::pin(async { Ok(()) })
    }

    fn read(dir: &Path, name: &str) -> String {
        std::fs::read_to_string(dir.join(name)).unwrap()
    }

    /// Settings from `vars` on top of a token and a base path.
    fn settings(vars: &[(&str, &str)]) -> Result<Settings, ConfigError> {
        let mut map: HashMap<&str, &str> = HashMap::from([
            ("CONFORMANCE_AUTH_TOKEN", "local-token"),
            ("CONFORMANCE_BASE_PATH", "/container/base.usda"),
        ]);
        map.extend(vars.iter().copied());
        Settings::from_lookup(&|name| map.get(name).map(|value| value.to_string()))
    }

    #[test]
    fn escape_xml_markup_characters_become_entity_references() {
        assert_eq!(
            escape_xml(r#"<a href="x">&'"#),
            "&lt;a href=&quot;x&quot;&gt;&amp;&apos;"
        );
    }

    #[test]
    fn escape_xml_tab_newline_and_carriage_return_become_character_references() {
        assert_eq!(escape_xml("a\tb\nc\rd"), "a&#x9;b&#xa;c&#xd;d");
    }

    #[test]
    fn escape_xml_del_and_c1_controls_become_character_references() {
        assert_eq!(
            escape_xml("\u{7f}\u{80}\u{85}\u{9f}"),
            "&#x7f;&#x80;&#x85;&#x9f;"
        );
    }

    #[test]
    fn escape_xml_characters_xml_cannot_carry_become_visible_u_escapes() {
        assert_eq!(
            escape_xml("\u{0}\u{1}\u{8}\u{b}\u{c}\u{1f}"),
            "\\u0000\\u0001\\u0008\\u000b\\u000c\\u001f"
        );
        assert_eq!(escape_xml("\u{fffe}\u{ffff}"), "\\ufffe\\uffff");
    }

    #[test]
    fn escape_xml_other_characters_are_unchanged() {
        let text = "plain é \u{a0} ☃ \u{fffd} \u{10000} `x`";
        assert_eq!(escape_xml(text), text);
    }

    #[test]
    fn junit_xml_writes_counts_passes_failures_and_errors() {
        let suites = [SuiteReport {
            name: RUN_SUITE.to_string(),
            seconds: 1.5,
            cases: vec![
                CaseReport {
                    seconds: 0.25,
                    ..CaseReport::new("layers::a".into(), Some("layers".into()))
                },
                CaseReport {
                    failure: Some("check failed: `a & b`".into()),
                    errors: vec!["cleanup: \"x\"".into()],
                    ..CaseReport::new("layers::b[<x>]".into(), Some("layers".into()))
                },
                CaseReport {
                    errors: vec!["setup: one".into(), "cleanup: two".into()],
                    ..CaseReport::new("streams::c".into(), Some("streams".into()))
                },
            ],
        }];
        let expected = "\
<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<testsuites>
  <testsuite name=\"live-edit-conformance\" tests=\"3\" failures=\"1\" errors=\"2\" skipped=\"0\" time=\"1.500\">
    <testcase classname=\"layers\" name=\"layers::a\" time=\"0.250\"/>
    <testcase classname=\"layers\" name=\"layers::b[&lt;x&gt;]\" time=\"0.000\">
      <failure message=\"check failed: `a &amp; b`\">check failed: `a &amp; b`</failure>
      <error message=\"cleanup: &quot;x&quot;\">cleanup: &quot;x&quot;</error>
    </testcase>
    <testcase classname=\"streams\" name=\"streams::c\" time=\"0.000\">
      <error message=\"setup: one\">setup: one</error>
      <error message=\"cleanup: two\">cleanup: two</error>
    </testcase>
  </testsuite>
</testsuites>
";
        assert_eq!(junit_xml(&suites), expected);
    }

    #[test]
    fn junit_xml_single_error_suite_has_no_classname() {
        let suites = [SuiteReport::single_error(
            PREFLIGHT_SUITE,
            "preflight",
            "bad <config>",
        )];
        let xml = junit_xml(&suites);
        assert!(
            xml.contains(
                "  <testsuite name=\"conformance-preflight\" tests=\"1\" failures=\"0\" errors=\"1\" skipped=\"0\" time=\"0.000\">\n    <testcase name=\"preflight\" time=\"0.000\">\n      <error message=\"bad &lt;config&gt;\">bad &lt;config&gt;</error>\n"
            ),
            "{xml}"
        );
    }

    #[test]
    fn summary_of_a_run_has_the_python_suites_keys_in_order() {
        let summary = Summary {
            profiles: Some(vec!["core".into(), "deployment".into()]),
            executed: Some(58),
            skipped: Some(0),
            unexpected_deselections: Some(0),
            exit_code: 0,
            build: Some(serde_json::json!({"contract": "c"})),
            failure: None,
        };
        assert_eq!(
            serde_json::to_string(&summary).unwrap(),
            r#"{"profiles":["core","deployment"],"executed":58,"skipped":0,"unexpected_deselections":0,"exit_code":0,"build":{"contract":"c"}}"#
        );
    }

    #[test]
    fn summary_of_a_run_that_never_reached_its_cases_has_exit_code_and_failure_only() {
        let summary = Summary {
            exit_code: EXIT_PREFLIGHT,
            failure: Some(DEPLOYMENT_SUITE.to_string()),
            ..Summary::default()
        };
        assert_eq!(
            serde_json::to_string(&summary).unwrap(),
            r#"{"exit_code":2,"failure":"deployment-fixture"}"#
        );
    }

    #[test]
    fn record_failure_after_a_run_keeps_its_suites_and_summary_and_sets_exit_code_2() {
        let dir = tempfile::tempdir().unwrap();
        let mut reports = reports_in(dir.path());
        reports.suites.push(SuiteReport {
            name: RUN_SUITE.to_string(),
            seconds: 0.0,
            cases: vec![CaseReport::new(
                "sessions::x".into(),
                Some("sessions".into()),
            )],
        });
        reports.summary = Some(Summary {
            profiles: Some(vec!["core".into()]),
            executed: Some(1),
            skipped: Some(0),
            unexpected_deselections: Some(0),
            exit_code: EXIT_PASSED,
            build: Some(fixtures::build_manifest()),
            failure: None,
        });

        reports.record_failure(
            DEPLOYMENT_SUITE,
            "Provisioning or cleanup failed (HttpStatus)",
        );

        let xml = read(dir.path(), RESULTS_FILE);
        let run_suite = xml
            .find("<testsuite name=\"live-edit-conformance\"")
            .unwrap();
        let fixture_suite = xml
            .find("<testsuite name=\"deployment-fixture\" tests=\"1\" failures=\"0\" errors=\"1\"")
            .unwrap();
        assert!(run_suite < fixture_suite, "{xml}");
        assert!(xml.contains("<testcase name=\"deployment-fixture\" time=\"0.000\">\n      <error message=\"Provisioning or cleanup failed (HttpStatus)\">"), "{xml}");

        let summary: serde_json::Value =
            serde_json::from_str(&read(dir.path(), SUMMARY_FILE)).unwrap();
        assert_eq!(summary["exit_code"], 2);
        assert_eq!(summary["failure"], "deployment-fixture");
        assert_eq!(summary["executed"], 1);
        assert_eq!(summary["profiles"], serde_json::json!(["core"]));
        assert_eq!(summary["build"], fixtures::build_manifest());
    }

    #[test]
    fn write_for_unwritable_reports_raises_a_pass_to_exit_1_only() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a-file");
        std::fs::write(&file, b"").unwrap();
        // A directory cannot be created under a regular file.
        let reports = reports_in(&file.join("reports"));
        assert_eq!(reports.write_for(EXIT_PASSED), EXIT_FAILED);
        assert_eq!(reports.write_for(EXIT_FAILED), EXIT_FAILED);
        assert_eq!(reports.write_for(EXIT_PREFLIGHT), EXIT_PREFLIGHT);

        let writable = reports_in(dir.path());
        assert_eq!(writable.write_for(EXIT_PASSED), EXIT_PASSED);
    }

    #[test]
    fn profile_from_name_selects_only_the_three_named_profiles() {
        for profile in Profile::SELECTABLE {
            assert_eq!(Profile::from_name(profile.name()), Some(profile));
        }
        assert_eq!(Profile::from_name("core"), None);
        assert_eq!(Profile::from_name("multi-replica"), None);
    }

    #[test]
    fn selected_runs_core_always_and_another_profile_only_when_selected() {
        let registry = || {
            vec![
                Case::new("a", "test_core_first", Profile::Core, false, passes),
                Case::new(
                    "a",
                    "test_multi_replica",
                    Profile::MultiReplica,
                    false,
                    passes,
                ),
                Case::new("b", "test_deployment", Profile::Deployment, false, passes),
                Case::new("b", "test_permissions", Profile::Permissions, false, passes),
                Case::new("b", "test_core_last", Profile::Core, false, passes),
            ]
        };
        let ids = |profiles: &[Profile]| -> Vec<String> {
            selected(registry(), profiles)
                .iter()
                .map(Case::id)
                .collect()
        };
        assert_eq!(ids(&[]), ["a::test_core_first", "b::test_core_last"]);
        assert_eq!(
            ids(&[Profile::Deployment]),
            [
                "a::test_core_first",
                "b::test_deployment",
                "b::test_core_last"
            ]
        );
        // Registry order, whatever the command-line order.
        assert_eq!(
            ids(&[
                Profile::Permissions,
                Profile::MultiReplica,
                Profile::Deployment
            ]),
            [
                "a::test_core_first",
                "a::test_multi_replica",
                "b::test_deployment",
                "b::test_permissions",
                "b::test_core_last"
            ]
        );
    }

    #[test]
    fn case_with_param_id_is_file_function_and_bracketed_param() {
        fn run(_ctx: &mut Ctx, _param: u32) -> CaseFuture<'_> {
            Box::pin(async { Ok(()) })
        }
        let case = Case::with_param("layers", "test_x", "100001", Profile::Core, true, 7, run);
        assert_eq!(case.id(), "layers::test_x[100001]");
    }

    /// Runs the preflight on `settings` and returns the exit code and the
    /// one preflight error's message, checking the reports' shape: the
    /// `conformance-preflight` suite and no summary.
    async fn preflight_failure(
        settings: Result<Settings, ConfigError>,
        profiles: &[Profile],
        interrupt: Interrupt,
    ) -> (i32, String) {
        let dir = tempfile::tempdir().unwrap();
        let mut reports = reports_in(dir.path());
        let code = run(settings, profiles, &mut reports, interrupt).await;
        let xml = read(dir.path(), RESULTS_FILE);
        assert!(
            xml.contains(
                "<testsuite name=\"conformance-preflight\" tests=\"1\" failures=\"0\" errors=\"1\""
            ),
            "{xml}"
        );
        assert!(xml.contains("<testcase name=\"preflight\""), "{xml}");
        // As the Python runner: a run that never reached its cases writes no
        // summary.
        assert!(!dir.path().join(SUMMARY_FILE).exists());
        let message = reports.suites[0].cases[0].errors[0].clone();
        (code, message)
    }

    #[tokio::test]
    async fn run_refused_configuration_is_a_conformance_preflight_error_exit_2() {
        let (code, message) =
            preflight_failure(settings(&[("CONFORMANCE_BASE_PATH", " ")]), &[], never()).await;
        assert_eq!(code, EXIT_PREFLIGHT);
        assert!(message.contains("CONFORMANCE_BASE_PATH"), "{message}");
    }

    #[tokio::test]
    async fn run_blank_endpoint_or_timeout_is_a_conformance_preflight_error_exit_2() {
        for name in [
            "CONFORMANCE_GRPC_ENDPOINT",
            "CONFORMANCE_TIMEOUT",
            "CONFORMANCE_READY_TIMEOUT",
        ] {
            // Refused before any wait: a run that took the blank value as
            // unset would wait for the default endpoint instead.
            let (code, message) = tokio::time::timeout(
                Duration::from_secs(10),
                preflight_failure(settings(&[(name, " ")]), &[], never()),
            )
            .await
            .unwrap_or_else(|_| panic!("{name}: the run waited instead of refusing"));
            assert_eq!(code, EXIT_PREFLIGHT, "{name}");
            assert_eq!(
                message,
                format!("{name} is set but empty; unset it to use the default")
            );
        }
    }

    #[tokio::test]
    async fn run_timeout_over_one_day_is_a_conformance_preflight_error_exit_2() {
        for name in ["CONFORMANCE_TIMEOUT", "CONFORMANCE_READY_TIMEOUT"] {
            let (code, message) = tokio::time::timeout(
                Duration::from_secs(10),
                preflight_failure(settings(&[(name, "86401")]), &[], never()),
            )
            .await
            .unwrap_or_else(|_| panic!("{name}: the run waited instead of refusing"));
            assert_eq!(code, EXIT_PREFLIGHT, "{name}");
            assert_eq!(
                message,
                format!("{name} must be at most 86400 seconds (one day)")
            );
        }
    }

    #[tokio::test]
    async fn run_bad_token_file_is_a_conformance_preflight_error_exit_2() {
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, r#"{"token":"no-access-token-key"}"#).unwrap();
        let (code, message) = preflight_failure(
            settings(&[
                ("CONFORMANCE_AUTH_TOKEN", ""),
                ("CONFORMANCE_AUTH_TOKEN_FILE", token.to_str().unwrap()),
            ]),
            &[],
            never(),
        )
        .await;
        assert_eq!(code, EXIT_PREFLIGHT);
        assert!(message.contains("access_token"), "{message}");
    }

    #[tokio::test]
    async fn run_permissions_profile_without_a_denied_path_is_a_preflight_error() {
        let (code, message) =
            preflight_failure(settings(&[]), &[Profile::Permissions], never()).await;
        assert_eq!(code, EXIT_PREFLIGHT);
        assert!(message.contains("CONFORMANCE_DENIED_PATH"), "{message}");
    }

    #[tokio::test]
    async fn run_multi_replica_profile_without_a_distinct_peer_is_a_preflight_error() {
        for peer in [None, Some("http://localhost:50001")] {
            let vars: Vec<(&str, &str)> = peer
                .map(|peer| ("CONFORMANCE_PEER_ENDPOINT", peer))
                .into_iter()
                .collect();
            let (code, message) =
                preflight_failure(settings(&vars), &[Profile::MultiReplica], never()).await;
            assert_eq!(code, EXIT_PREFLIGHT);
            assert!(
                message.contains("CONFORMANCE_PEER_ENDPOINT"),
                "{peer:?}: {message}"
            );
        }
    }

    #[tokio::test]
    async fn run_interrupted_during_the_preflight_is_a_preflight_error_exit_2() {
        let (code, message) = preflight_failure(settings(&[]), &[], fired()).await;
        assert_eq!(code, EXIT_PREFLIGHT);
        assert_eq!(message, INTERRUPTED);
    }
}
