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

//! `runner::execute` with planted cases: how a case's outcome becomes its
//! JUnit entry, the summary and the exit code; and one real case, the gated
//! negative `ErrorInfo` check of `layers`, against a scripted server.
//!
//! A planted case that never calls its client needs no server: the client
//! connects lazily. Planted cases that need a session, or a cleanup to fail
//! or hang, run against [`FakeLive`].
//!
//! The tests of [`INTERRUPT_CLEANUP_BOUND`] wait it out in real time (10 s
//! each): the cleanup they bound is a `DeleteSession` on the wire, which
//! paused time could expire before its request is even sent.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use live_edit_conformance::cases::layers;
use live_edit_conformance::check::{CaseError, CaseResult};
use live_edit_conformance::fixtures::build_manifest;
use live_edit_conformance::runner::{
    execute, Case, CaseFuture, Ctx, Profile, CASE_TIMEOUT, EXIT_FAILED, EXIT_PASSED,
    EXIT_PREFLIGHT, INTERRUPT_CLEANUP_BOUND, RUN_SUITE,
};
use tokio::sync::Notify;
use tokio::time::Instant;
use tonic::codegen::Bytes;
use tonic::{Code, Status};
use tonic_types::{ErrorDetails, StatusExt};

use crate::fake_live::{Answer, FakeLive};
use crate::support::{fired, interrupt, junit, never, summary, testcase, trigger, Vars};

/// The planted cases' module name.
const FILE: &str = "planted";

/// An endpoint nothing listens on: a planted case that never calls its
/// client never connects to it.
const UNUSED_ENDPOINT: &str = "http://127.0.0.1:9";

fn passes(_ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async { Ok(()) })
}

/// Counts its runs.
fn counts(_ctx: &mut Ctx, runs: Arc<AtomicUsize>) -> CaseFuture<'_> {
    Box::pin(async move {
        runs.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })
}

fn fails(_ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async { Err(CaseError::new("planted failure <&>")) })
}

fn hangs(_ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(futures::future::pending::<CaseResult>())
}

fn panics(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        // A case without a session always has an empty one.
        if ctx.session.is_empty() {
            panic!("planted panic");
        }
        Ok(())
    })
}

/// Creates a session through its client (owned, so deleted at cleanup).
fn creates_a_session(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        ctx.client.session().await?;
        Ok(())
    })
}

/// Signals that it started, then never ends.
fn starts_then_hangs(_ctx: &mut Ctx, started: Arc<Notify>) -> CaseFuture<'_> {
    Box::pin(async move {
        started.notify_one();
        futures::future::pending::<CaseResult>().await
    })
}

fn counter() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

fn counted(name: &'static str, profile: Profile, runs: &Arc<AtomicUsize>) -> Case {
    Case::with_param(FILE, name, "x", profile, false, runs.clone(), counts)
}

#[tokio::test]
async fn execute_every_selected_case_passing_is_exit_0_with_its_summary() {
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", UNUSED_ENDPOINT)
        .reports_in(dir.path());
    let (core, deployment, permissions) = (counter(), counter(), counter());
    let cases = vec![
        counted("test_core", Profile::Core, &core),
        counted("test_deployment", Profile::Deployment, &deployment),
        counted("test_permissions", Profile::Permissions, &permissions),
        Case::new(FILE, "test_passes", Profile::Core, false, passes),
    ];
    let mut reports = vars.reports();

    let code = execute(
        vars.arc_settings(),
        cases,
        &[Profile::Deployment],
        &mut reports,
        never(),
    )
    .await;

    assert_eq!(code, EXIT_PASSED);
    // Only the selected profile's case beside the core ones ran.
    let runs: Vec<usize> = [&core, &deployment, &permissions]
        .iter()
        .map(|runs| runs.load(Ordering::SeqCst))
        .collect();
    assert_eq!(runs, [1, 1, 0]);
    assert_eq!(
        summary(dir.path()),
        serde_json::json!({
            "profiles": ["core", "deployment"],
            "executed": 3,
            "skipped": 0,
            "unexpected_deselections": 0,
            "exit_code": 0,
            "build": build_manifest(),
        })
    );
    let xml = junit(dir.path());
    assert!(
        xml.contains("<testsuite name=\"live-edit-conformance\" tests=\"3\" failures=\"0\" errors=\"0\" skipped=\"0\""),
        "{xml}"
    );
    assert!(
        testcase(&xml, "planted::test_passes").ends_with("/>\n"),
        "{xml}"
    );
}

#[tokio::test]
async fn execute_failed_case_is_a_junit_failure_and_exit_1() {
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", UNUSED_ENDPOINT)
        .reports_in(dir.path());
    let cases = vec![Case::new(FILE, "test_fails", Profile::Core, false, fails)];
    let mut reports = vars.reports();

    let code = execute(vars.arc_settings(), cases, &[], &mut reports, never()).await;

    assert_eq!(code, EXIT_FAILED);
    let xml = junit(dir.path());
    let case = testcase(&xml, "planted::test_fails");
    assert!(
        case.contains("<failure message=\"planted failure &lt;&amp;&gt;\">"),
        "{case}"
    );
    assert!(!case.contains("<error"), "{case}");
    assert_eq!(summary(dir.path())["exit_code"], 1);
    assert_eq!(summary(dir.path())["executed"], 1);
}

#[tokio::test(start_paused = true)]
async fn execute_case_timeout_fails_that_case_only_and_the_run_continues() {
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", UNUSED_ENDPOINT)
        .reports_in(dir.path());
    let after = counter();
    let cases = vec![
        Case::new(FILE, "test_hangs", Profile::Core, false, hangs),
        counted("test_after", Profile::Core, &after),
    ];
    let mut reports = vars.reports();
    let started = tokio::time::Instant::now();

    let code = execute(vars.arc_settings(), cases, &[], &mut reports, never()).await;

    assert_eq!(code, EXIT_FAILED);
    assert!(started.elapsed() >= CASE_TIMEOUT);
    assert_eq!(after.load(Ordering::SeqCst), 1, "the next case still ran");
    let xml = junit(dir.path());
    assert!(
        testcase(&xml, "planted::test_hangs")
            .contains("<failure message=\"case timed out after 120 s\">"),
        "{xml}"
    );
    assert!(
        testcase(&xml, "planted::test_after[x]").ends_with("/>\n"),
        "{xml}"
    );
    assert_eq!(summary(dir.path())["executed"], 2);
}

#[tokio::test]
async fn execute_case_panic_fails_that_case_only_and_the_run_continues() {
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", UNUSED_ENDPOINT)
        .reports_in(dir.path());
    let after = counter();
    let cases = vec![
        Case::new(FILE, "test_panics", Profile::Core, false, panics),
        counted("test_after", Profile::Core, &after),
    ];
    let mut reports = vars.reports();

    let code = execute(vars.arc_settings(), cases, &[], &mut reports, never()).await;

    assert_eq!(code, EXIT_FAILED);
    assert_eq!(after.load(Ordering::SeqCst), 1, "the next case still ran");
    let xml = junit(dir.path());
    assert!(
        testcase(&xml, "planted::test_panics")
            .contains("<failure message=\"case panicked: planted panic\">"),
        "{xml}"
    );
    assert_eq!(summary(dir.path())["executed"], 2);
}

#[tokio::test]
async fn execute_no_selected_case_is_never_a_pass() {
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", UNUSED_ENDPOINT)
        .reports_in(dir.path());
    // Its only case belongs to a profile the run did not select.
    let cases = vec![Case::new(
        FILE,
        "test_permissions",
        Profile::Permissions,
        false,
        passes,
    )];
    let mut reports = vars.reports();

    let code = execute(vars.arc_settings(), cases, &[], &mut reports, never()).await;

    assert_eq!(code, EXIT_FAILED);
    assert_eq!(summary(dir.path())["executed"], 0);
}

#[tokio::test]
async fn execute_session_setup_failure_is_an_error_and_the_body_never_runs() {
    let server =
        FakeLive::start(|_, _| Answer::Fail(Status::permission_denied("not this path"))).await;
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", &server.endpoint)
        .reports_in(dir.path());
    let body = counter();
    let cases = vec![Case::with_param(
        FILE,
        "test_needs_a_session",
        "x",
        Profile::Core,
        true,
        body.clone(),
        counts,
    )];
    let mut reports = vars.reports();

    let code = execute(vars.arc_settings(), cases, &[], &mut reports, never()).await;

    // Executed (0) is short of selected (1): never a pass.
    assert_eq!(code, EXIT_FAILED);
    assert_eq!(body.load(Ordering::SeqCst), 0);
    assert_eq!(summary(dir.path())["executed"], 0);
    let xml = junit(dir.path());
    let case = testcase(&xml, "planted::test_needs_a_session[x]");
    assert!(
        case.contains("<error message=\"setup: creating the session failed: call ended with PermissionDenied: not this path\">"),
        "{case}"
    );
    assert!(!case.contains("<failure"), "{case}");
}

#[tokio::test]
async fn execute_cleanup_failure_is_an_error_of_an_otherwise_passing_case() {
    let server = FakeLive::start(|method, _| match method {
        "DeleteSession" => Answer::Fail(Status::internal("store down")),
        _ => Answer::Ok,
    })
    .await;
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", &server.endpoint)
        .reports_in(dir.path());
    let cases = vec![Case::new(
        FILE,
        "test_creates_a_session",
        Profile::Core,
        false,
        creates_a_session,
    )];
    let mut reports = vars.reports();

    let code = execute(vars.arc_settings(), cases, &[], &mut reports, never()).await;

    assert_eq!(code, EXIT_FAILED);
    assert_eq!(server.deleted(), ["session-1"]);
    let xml = junit(dir.path());
    let case = testcase(&xml, "planted::test_creates_a_session");
    assert!(
        case.contains("<error message=\"cleanup: Resource cleanup failed: DeleteSession(session-1): Internal\">"),
        "{case}"
    );
    assert!(!case.contains("<failure"), "{case}");
    assert_eq!(summary(dir.path())["executed"], 1);
}

#[tokio::test]
async fn execute_interrupt_during_a_case_cleans_up_writes_the_reports_and_exits_2() {
    let server = FakeLive::ok().await;
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", &server.endpoint)
        .reports_in(dir.path());
    let started = Arc::new(Notify::new());
    let after = counter();
    let cases = vec![
        Case::with_param(
            FILE,
            "test_interrupted",
            "x",
            Profile::Core,
            true,
            started.clone(),
            starts_then_hangs,
        ),
        counted("test_after", Profile::Core, &after),
    ];
    let mut reports = vars.reports();
    let (fire, interrupt) = interrupt();

    // SIGTERM arrives once the first case's body is running.
    let run = execute(vars.arc_settings(), cases, &[], &mut reports, interrupt);
    let signal = async {
        started.notified().await;
        fire.send(()).unwrap();
    };
    let (code, ()) = tokio::join!(run, signal);

    assert_eq!(code, EXIT_PREFLIGHT);
    assert_eq!(
        after.load(Ordering::SeqCst),
        0,
        "no case runs after the interrupt"
    );
    // The interrupted case's session was still deleted.
    assert_eq!(server.deleted(), ["session-1"]);
    let xml = junit(dir.path());
    assert!(
        testcase(&xml, "planted::test_interrupted[x]")
            .contains("<error message=\"interrupted by SIGTERM or SIGINT\">"),
        "{xml}"
    );
    assert!(!xml.contains("planted::test_after"), "{xml}");
    let summary = summary(dir.path());
    assert_eq!(summary["exit_code"], 2);
    assert_eq!(summary["executed"], 1);
    // The failure that set exit 2 is the run suite's: the interrupted case's
    // <error> is there.
    assert_eq!(summary["failure"], RUN_SUITE);
}

#[tokio::test]
async fn execute_interrupt_before_a_case_ends_the_run_with_exit_2() {
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", UNUSED_ENDPOINT)
        .reports_in(dir.path());
    let after = counter();
    let cases = vec![
        Case::new(FILE, "test_first", Profile::Core, false, hangs),
        counted("test_after", Profile::Core, &after),
    ];
    let mut reports = vars.reports();

    let code = execute(vars.arc_settings(), cases, &[], &mut reports, fired()).await;

    assert_eq!(code, EXIT_PREFLIGHT);
    assert_eq!(after.load(Ordering::SeqCst), 0);
    let summary = summary(dir.path());
    assert_eq!(summary["exit_code"], 2);
    assert_eq!(summary["failure"], RUN_SUITE);
}

#[tokio::test]
async fn execute_uninterrupted_runs_have_no_failure_in_the_summary() {
    // `failure` names the suite of a failure that set exit 2; a run that
    // failed a case (exit 1) has none.
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", UNUSED_ENDPOINT)
        .reports_in(dir.path());
    let cases = vec![Case::new(FILE, "test_fails", Profile::Core, false, fails)];
    let mut reports = vars.reports();

    let code = execute(vars.arc_settings(), cases, &[], &mut reports, never()).await;

    assert_eq!(code, EXIT_FAILED);
    let summary = summary(dir.path());
    assert_eq!(summary["exit_code"], 1);
    assert!(summary.get("failure").is_none(), "{summary}");
}

// --- Cleanup: the peer client, panics, and the bound after an interrupt.

/// What a cleanup bounded by [`INTERRUPT_CLEANUP_BOUND`] may take in all
/// before the test fails rather than waits: an unbounded one would wait for
/// `CONFORMANCE_TIMEOUT` ([`HANG_TIMEOUT`]).
const BOUNDED_BUDGET: Duration = Duration::from_secs(60);

/// The `CONFORMANCE_TIMEOUT` of the tests whose `DeleteSession` hangs:
/// longer than any wait a test measures, so only a bound ends the wait.
const HANG_TIMEOUT: &str = "600";

/// The JUnit error of a cleanup cut off at the bound.
const CLEANUP_CUT_OFF: &str = "cleanup: not finished within 10 s of the interruption";

/// Creates a session on the peer endpoint and one on the endpoint (each
/// owned by its client, so deleted at cleanup).
fn creates_sessions_on_both_replicas(ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        ctx.peer()?.session().await?;
        ctx.client.session().await?;
        Ok(())
    })
}

#[tokio::test]
async fn execute_peer_cleanup_deletes_the_peer_clients_sessions_and_reports_its_failure() {
    let server = FakeLive::ok().await;
    let peer = FakeLive::start(|method, _| match method {
        "DeleteSession" => Answer::Fail(Status::internal("peer store down")),
        _ => Answer::Ok,
    })
    .await;
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", &server.endpoint)
        .set("CONFORMANCE_PEER_ENDPOINT", &peer.endpoint)
        .reports_in(dir.path());
    let cases = vec![Case::new(
        FILE,
        "test_on_two_replicas",
        Profile::MultiReplica,
        false,
        creates_sessions_on_both_replicas,
    )];
    let mut reports = vars.reports();

    let code = execute(
        vars.arc_settings(),
        cases,
        &[Profile::MultiReplica],
        &mut reports,
        never(),
    )
    .await;

    assert_eq!(code, EXIT_FAILED);
    // Each client deleted the session it owns, on its own replica.
    assert_eq!(server.deleted(), ["session-1"]);
    assert_eq!(peer.deleted(), ["session-1"]);
    let xml = junit(dir.path());
    let case = testcase(&xml, "planted::test_on_two_replicas");
    assert!(
        case.contains("<error message=\"peer cleanup: Resource cleanup failed: DeleteSession(session-1): Internal\">"),
        "{case}"
    );
    assert_eq!(case.matches("<error ").count(), 1, "{case}");
    assert!(!case.contains("<failure"), "{case}");
}

/// Moves the paused clock to within two seconds of the last instant
/// `tokio::time::Instant` can hold, so that `Instant::now()` plus two
/// seconds overflows, which panics. [`Client::close`] computes such a
/// deadline (its cancel grace) before anything else, so it panics in the
/// runner's cleanup. (Needs a runtime started paused.)
///
/// [`Client::close`]: live_edit_conformance::client::Client::close
fn moves_the_clock_to_the_end_of_time(_ctx: &mut Ctx) -> CaseFuture<'_> {
    Box::pin(async move {
        let now = Instant::now();
        // The largest whole number of seconds `now` can still be moved by.
        let (mut valid, mut overflows) = (0u64, u64::MAX);
        while overflows - valid > 1 {
            let middle = valid + (overflows - valid) / 2;
            if now.checked_add(Duration::from_secs(middle)).is_some() {
                valid = middle;
            } else {
                overflows = middle;
            }
        }
        tokio::time::advance(Duration::from_secs(valid - 1)).await;
        assert!(Instant::now().checked_add(Duration::from_secs(2)).is_none());
        Ok(())
    })
}

#[tokio::test(start_paused = true)]
async fn execute_cleanup_panic_is_an_error_of_that_case_and_the_peer_is_still_closed() {
    // The case passes; closing its client panics, and so does closing its
    // peer client after it. Neither client ever connects.
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", UNUSED_ENDPOINT)
        .set("CONFORMANCE_PEER_ENDPOINT", "http://127.0.0.1:10")
        .reports_in(dir.path());
    let cases = vec![Case::new(
        FILE,
        "test_ends_time",
        Profile::MultiReplica,
        false,
        moves_the_clock_to_the_end_of_time,
    )];
    let mut reports = vars.reports();

    let code = execute(
        vars.arc_settings(),
        cases,
        &[Profile::MultiReplica],
        &mut reports,
        never(),
    )
    .await;

    assert_eq!(code, EXIT_FAILED);
    let xml = junit(dir.path());
    let case = testcase(&xml, "planted::test_ends_time");
    let main = case
        .find("<error message=\"cleanup panicked: ")
        .unwrap_or_else(|| panic!("{case}"));
    let peer = case
        .find("<error message=\"peer cleanup panicked: ")
        .unwrap_or_else(|| panic!("{case}"));
    assert!(main < peer, "{case}");
    assert!(!case.contains("<failure"), "{case}");
    assert_eq!(summary(dir.path())["executed"], 1);
}

#[tokio::test(start_paused = true)]
async fn execute_session_setup_panic_is_an_error_and_the_body_never_runs() {
    // The first case moves the clock to the end of time (its own cleanup
    // then panics too). Creating the second case's session then panics: its
    // call deadline (`client::with_deadline`, a `tokio::time::timeout`) no
    // longer fits in an `Instant`, and the far-future fallback overflows. No
    // accepted configuration makes setup panic since the timeout cap
    // (`config::MAX_TIMEOUT`), so this is the trigger. No third case: its
    // body's own `CASE_TIMEOUT` would overflow outside any catch.
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", UNUSED_ENDPOINT)
        .reports_in(dir.path());
    let body = counter();
    let cases = vec![
        Case::new(
            FILE,
            "test_ends_time",
            Profile::Core,
            false,
            moves_the_clock_to_the_end_of_time,
        ),
        Case::with_param(
            FILE,
            "test_needs_a_session",
            "x",
            Profile::Core,
            true,
            body.clone(),
            counts,
        ),
    ];
    let mut reports = vars.reports();

    let code = execute(vars.arc_settings(), cases, &[], &mut reports, never()).await;

    // The run got past the panic and wrote its reports.
    assert_eq!(code, EXIT_FAILED);
    assert_eq!(body.load(Ordering::SeqCst), 0, "the body never ran");
    let xml = junit(dir.path());
    let case = testcase(&xml, "planted::test_needs_a_session[x]");
    assert!(
        case.contains("<error message=\"setup: creating the session panicked: "),
        "{case}"
    );
    assert!(!case.contains("<failure"), "{case}");
    assert_eq!(summary(dir.path())["executed"], 1);
}

#[tokio::test]
async fn execute_interrupt_during_a_cleanup_that_then_finishes_is_recorded_on_the_case() {
    // The case passes; SIGTERM arrives while its session's DeleteSession is
    // on the server, which answers it half a second later.
    let (signal, interrupt) = trigger();
    let server = FakeLive::start({
        let signal = signal.clone();
        move |method, _| match method {
            "DeleteSession" => {
                signal.fire();
                Answer::After(Duration::from_millis(500), Box::new(Answer::Ok))
            }
            _ => Answer::Ok,
        }
    })
    .await;
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", &server.endpoint)
        .reports_in(dir.path());
    let after = counter();
    let cases = vec![
        Case::new(
            FILE,
            "test_creates_a_session",
            Profile::Core,
            false,
            creates_a_session,
        ),
        counted("test_after", Profile::Core, &after),
    ];
    let mut reports = vars.reports();

    let code = tokio::time::timeout(
        BOUNDED_BUDGET,
        execute(vars.arc_settings(), cases, &[], &mut reports, interrupt),
    )
    .await
    .expect("the run ends");

    assert_eq!(code, EXIT_PREFLIGHT);
    assert!(signal.fired_at().is_some());
    assert_eq!(server.deleted(), ["session-1"]);
    assert_eq!(after.load(Ordering::SeqCst), 0, "no case runs after it");
    let xml = junit(dir.path());
    let case = testcase(&xml, "planted::test_creates_a_session");
    assert!(
        case.contains("<error message=\"cleanup: interrupted by SIGTERM or SIGINT\">"),
        "{case}"
    );
    assert_eq!(case.matches("<error ").count(), 1, "{case}");
    assert!(!case.contains("<failure"), "{case}");
    let summary = summary(dir.path());
    assert_eq!(summary["exit_code"], 2);
    assert_eq!(summary["failure"], RUN_SUITE);
}

#[tokio::test]
async fn execute_interrupted_cases_cleanup_that_never_finishes_is_cut_off_at_the_bound() {
    // SIGTERM arrives during the case; its session's DeleteSession then
    // never gets an answer.
    let server = FakeLive::start(|method, _| match method {
        "DeleteSession" => Answer::Hang,
        _ => Answer::Ok,
    })
    .await;
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", &server.endpoint)
        .set("CONFORMANCE_TIMEOUT", HANG_TIMEOUT)
        .reports_in(dir.path());
    let started = Arc::new(Notify::new());
    let cases = vec![Case::with_param(
        FILE,
        "test_interrupted",
        "x",
        Profile::Core,
        true,
        started.clone(),
        starts_then_hangs,
    )];
    let mut reports = vars.reports();
    let (fire, interrupt) = interrupt();

    let run = tokio::time::timeout(
        BOUNDED_BUDGET,
        execute(vars.arc_settings(), cases, &[], &mut reports, interrupt),
    );
    let signal = async {
        started.notified().await;
        fire.send(()).unwrap();
        Instant::now()
    };
    let (code, signalled) = tokio::join!(run, signal);
    let code = code.expect("the cleanup is bounded");

    assert_eq!(code, EXIT_PREFLIGHT);
    assert!(signalled.elapsed() >= INTERRUPT_CLEANUP_BOUND);
    assert_eq!(server.deleted(), ["session-1"], "the delete was sent");
    let xml = junit(dir.path());
    let case = testcase(&xml, "planted::test_interrupted[x]");
    let interrupted = case
        .find("<error message=\"interrupted by SIGTERM or SIGINT\">")
        .unwrap_or_else(|| panic!("{case}"));
    let cut_off = case
        .find(&format!("<error message=\"{CLEANUP_CUT_OFF}\">"))
        .unwrap_or_else(|| panic!("{case}"));
    assert!(interrupted < cut_off, "{case}");
    let summary = summary(dir.path());
    assert_eq!(summary["exit_code"], 2);
    assert_eq!(summary["failure"], RUN_SUITE);
}

#[tokio::test]
async fn execute_interrupt_during_a_passing_cases_cleanup_bounds_that_cleanup() {
    // The case passes; SIGTERM arrives while its session's DeleteSession is
    // on the server, which never answers it. The cleanup, unbounded until
    // then, is cut off at the bound from the signal.
    let (signal, interrupt) = trigger();
    let server = FakeLive::start({
        let signal = signal.clone();
        move |method, _| match method {
            "DeleteSession" => {
                signal.fire();
                Answer::Hang
            }
            _ => Answer::Ok,
        }
    })
    .await;
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", &server.endpoint)
        .set("CONFORMANCE_TIMEOUT", HANG_TIMEOUT)
        .reports_in(dir.path());
    let after = counter();
    let cases = vec![
        Case::new(
            FILE,
            "test_creates_a_session",
            Profile::Core,
            false,
            creates_a_session,
        ),
        counted("test_after", Profile::Core, &after),
    ];
    let mut reports = vars.reports();

    let code = tokio::time::timeout(
        BOUNDED_BUDGET,
        execute(vars.arc_settings(), cases, &[], &mut reports, interrupt),
    )
    .await
    .expect("the cleanup is bounded once the signal arrives");

    assert_eq!(code, EXIT_PREFLIGHT);
    let signalled = signal.fired_at().expect("the DeleteSession arrived");
    assert!(signalled.elapsed() >= INTERRUPT_CLEANUP_BOUND);
    assert_eq!(after.load(Ordering::SeqCst), 0, "no case runs after it");
    let xml = junit(dir.path());
    let case = testcase(&xml, "planted::test_creates_a_session");
    assert!(
        case.contains(&format!("<error message=\"{CLEANUP_CUT_OFF}\">")),
        "{case}"
    );
    assert_eq!(case.matches("<error ").count(), 1, "{case}");
    assert!(!case.contains("<failure"), "{case}");
    let summary = summary(dir.path());
    assert_eq!(summary["exit_code"], 2);
    assert_eq!(summary["executed"], 1);
    assert_eq!(summary["failure"], RUN_SUITE);
}

#[tokio::test]
async fn execute_passing_run_with_unwritable_reports_is_exit_1() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("a-file");
    std::fs::write(&file, b"").unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", UNUSED_ENDPOINT)
        .reports_in(&file.join("reports"));
    let cases = vec![Case::new(FILE, "test_passes", Profile::Core, false, passes)];
    let mut reports = vars.reports();

    let code = execute(vars.arc_settings(), cases, &[], &mut reports, never()).await;

    assert_eq!(code, EXIT_FAILED);
}

// --- The gated negative check of
// `layers::test_delta_one_entry_over_the_limit_is_refused_and_one_at_it_never_delta_too_large`:
// an update of exactly the entry limit may be refused, but never
// DELTA_TOO_LARGE, and an ErrorInfo-typed detail that does not decode must
// fail the case rather than read as "no ErrorInfo".

const AT_THE_LIMIT_CASE: &str =
    "test_delta_one_entry_over_the_limit_is_refused_and_one_at_it_never_delta_too_large";

/// The start of the JUnit failure the gated check gives, as `check!` words
/// it (XML-escaped): it names the `ErrorInfo` read, so the case failed at the
/// gate and not at an earlier check.
const GATED_CHECK_FAILED: &str = "<failure message=\"check failed: `error_info(&amp;rejected)";

/// The service's refusal of the 100,000-entry update: DELTA_TOO_LARGE with
/// the counts the case checks.
fn delta_too_large() -> Status {
    Status::with_error_details(
        Code::InvalidArgument,
        "the delta has 100000 entries, over the limit of 99999; split it",
        ErrorDetails::with_error_info(
            "DELTA_TOO_LARGE",
            "nvidia.omniverse.live.v1alpha",
            [
                ("struct_commands".to_string(), "0".to_string()),
                ("set_fields".to_string(), "100000".to_string()),
                ("time_samples".to_string(), "0".to_string()),
                ("max_entries".to_string(), "99999".to_string()),
            ],
        ),
    )
}

fn varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return out;
        }
        out.push(byte | 0x80);
    }
}

fn field(number: u64, payload: &[u8]) -> Vec<u8> {
    [
        varint(number << 3 | 2),
        varint(payload.len() as u64),
        payload.to_vec(),
    ]
    .concat()
}

/// INVALID_ARGUMENT whose details hold an ErrorInfo-typed `Any` whose value
/// does not decode (its first field claims 5 bytes and carries 1).
fn undecodable_error_info() -> Status {
    let any = [
        field(1, b"type.googleapis.com/google.rpc.ErrorInfo"),
        field(2, &[0x0a, 0x05, b'a']),
    ]
    .concat();
    let details = [vec![0x08, 3], field(2, b"refused"), field(3, &any)].concat();
    Status::with_details(Code::InvalidArgument, "refused", Bytes::from(details))
}

fn error_info(reason: &str) -> Status {
    Status::with_error_details(
        Code::InvalidArgument,
        "refused",
        ErrorDetails::with_error_info(
            reason,
            "nvidia.omniverse.live.v1alpha",
            [("detail".to_string(), "1".to_string())],
        ),
    )
}

/// Runs the case against a server that refuses the first update
/// DELTA_TOO_LARGE and answers the second, at the limit, with `at_limit`;
/// returns the exit code and the case's JUnit entry.
async fn run_at_the_limit_case(at_limit: Answer) -> (i32, String) {
    let server = FakeLive::start(move |method, n| match (method, n) {
        ("UpdateLayer", 1) => Answer::Fail(delta_too_large()),
        ("UpdateLayer", 2) => at_limit.clone(),
        _ => Answer::Ok,
    })
    .await;
    let dir = tempfile::tempdir().unwrap();
    let vars = Vars::base()
        .set("CONFORMANCE_GRPC_ENDPOINT", &server.endpoint)
        .reports_in(dir.path());
    let case = layers::cases()
        .into_iter()
        .find(|case| case.name == AT_THE_LIMIT_CASE)
        .expect("the layers case is registered");
    let mut reports = vars.reports();

    let code = execute(vars.arc_settings(), vec![case], &[], &mut reports, never()).await;

    // The case reached its gated check: both updates were sent.
    assert_eq!(
        server
            .methods()
            .iter()
            .filter(|method| *method == "UpdateLayer")
            .count(),
        2
    );
    let xml = junit(dir.path());
    let entry = testcase(&xml, &format!("layers::{AT_THE_LIMIT_CASE}")).to_string();
    (code, entry)
}

#[tokio::test]
async fn layers_at_the_limit_refusal_without_error_info_passes() {
    let (code, entry) =
        run_at_the_limit_case(Answer::Fail(Status::invalid_argument("refused"))).await;
    assert_eq!(code, EXIT_PASSED, "{entry}");
}

#[tokio::test]
async fn layers_at_the_limit_acceptance_passes() {
    let (code, entry) = run_at_the_limit_case(Answer::Ok).await;
    assert_eq!(code, EXIT_PASSED, "{entry}");
}

#[tokio::test]
async fn layers_at_the_limit_refusal_with_another_reason_passes() {
    let (code, entry) = run_at_the_limit_case(Answer::Fail(error_info("INVALID_UPDATE"))).await;
    assert_eq!(code, EXIT_PASSED, "{entry}");
}

#[tokio::test]
async fn layers_at_the_limit_refusal_delta_too_large_fails() {
    let (code, entry) = run_at_the_limit_case(Answer::Fail(error_info("DELTA_TOO_LARGE"))).await;
    assert_eq!(code, EXIT_FAILED);
    assert!(entry.contains(GATED_CHECK_FAILED), "{entry}");
}

#[tokio::test]
async fn layers_at_the_limit_undecodable_error_info_detail_fails() {
    let (code, entry) = run_at_the_limit_case(Answer::Fail(undecodable_error_info())).await;
    assert_eq!(code, EXIT_FAILED);
    assert!(entry.contains(GATED_CHECK_FAILED), "{entry}");
}
