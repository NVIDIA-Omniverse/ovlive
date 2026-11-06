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

//! The preflight's readiness wait, through `runner::run_with_cases` against
//! [`FakeLive`] with planted cases: what it retries, what it falls back to,
//! what fails it and how the failure is reported, and that the peer endpoint
//! is waited for too. A run that got through the preflight runs its planted
//! case; one that did not writes the `conformance-preflight` error and no
//! `summary.json`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use live_edit_conformance::config::Settings;
use live_edit_conformance::runner::{
    run_with_cases, Case, CaseFuture, Ctx, Interrupt, Profile, EXIT_PASSED, EXIT_PREFLIGHT,
    SUMMARY_FILE,
};
use tonic::Status;

use crate::fake_live::{Answer, FakeLive};
use crate::support::{junit, never, trigger, Vars, TOKEN};

/// How long any run here may take before the test fails rather than waits.
const RUN_BUDGET: Duration = Duration::from_secs(60);

/// Status text the server sends with a refusal. It can carry anything a
/// service puts there, a credential included, so no report may repeat it.
const SECRET: &str = "secret-status-text-must-not-be-reported";

fn counts(_ctx: &mut Ctx, runs: Arc<AtomicUsize>) -> CaseFuture<'_> {
    Box::pin(async move {
        runs.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })
}

/// How a run ended.
struct Ran {
    code: i32,
    /// How many times the planted case ran.
    runs: usize,
    /// The JUnit report.
    xml: String,
    /// Whether `summary.json` was written.
    summary: bool,
}

impl Ran {
    /// Asserts the run failed its preflight with exactly `message`, and so
    /// never reached its case.
    fn assert_preflight_failed(&self, message: &str) {
        assert_eq!(self.code, EXIT_PREFLIGHT, "{}", self.xml);
        assert_eq!(self.runs, 0, "the case ran");
        assert!(
            self.xml.contains(
                "<testsuite name=\"conformance-preflight\" tests=\"1\" failures=\"0\" errors=\"1\""
            ),
            "{}",
            self.xml
        );
        assert!(
            self.xml
                .contains(&format!("<error message=\"{message}\">{message}</error>")),
            "{}",
            self.xml
        );
        assert!(
            !self.summary,
            "a run that never reached its cases has no summary"
        );
    }

    /// Asserts the run got through the preflight and ran its case.
    fn assert_passed(&self) {
        assert_eq!(self.code, EXIT_PASSED, "{}", self.xml);
        assert_eq!(self.runs, 1);
        assert!(self.summary);
    }
}

/// Runs the preflight against `server`, with `vars` on top, and then one
/// planted core case.
async fn run(server: &FakeLive, vars: Vars, interrupt: Interrupt) -> Ran {
    let dir = tempfile::tempdir().unwrap();
    let vars = vars
        .set("CONFORMANCE_GRPC_ENDPOINT", &server.endpoint)
        .reports_in(dir.path());
    let runs = Arc::new(AtomicUsize::new(0));
    let cases = vec![Case::with_param(
        "planted",
        "test_after_the_preflight",
        "x",
        Profile::Core,
        false,
        runs.clone(),
        counts,
    )];
    let settings: Settings = vars.settings();
    let mut reports = vars.reports();

    let code = tokio::time::timeout(
        RUN_BUDGET,
        run_with_cases(Ok(settings), cases, &[], &mut reports, interrupt),
    )
    .await
    .expect("the run ends");

    Ran {
        code,
        runs: runs.load(Ordering::SeqCst),
        xml: junit(dir.path()),
        summary: dir.path().join(SUMMARY_FILE).exists(),
    }
}

/// How many `Check`s `server` received.
fn checks(server: &FakeLive) -> usize {
    server
        .methods()
        .iter()
        .filter(|method| *method == "Check")
        .count()
}

#[tokio::test]
async fn readiness_not_serving_is_retried_until_serving() {
    let server = FakeLive::start(|method, n| match (method, n) {
        ("Check", 1 | 2) => Answer::NotServing,
        _ => Answer::Ok,
    })
    .await;

    let ran = run(&server, Vars::base(), never()).await;

    ran.assert_passed();
    assert_eq!(server.methods(), ["Check", "Check", "Check"]);
}

#[tokio::test]
async fn readiness_unanswered_health_checks_are_retried_until_serving() {
    // The first checks get no answer: each ends at its deadline on the client
    // (tonic's own timer usually ends it, reported DEADLINE_EXCEEDED), and is
    // retried. Several, so a client that reported tonic's CANCELLED instead
    // fails all but by chance.
    const UNANSWERED: usize = 4;
    let server = FakeLive::start(|method, n| match method {
        "Check" if n <= UNANSWERED => Answer::Hang,
        _ => Answer::Ok,
    })
    .await;
    let vars = Vars::base()
        .set("CONFORMANCE_TIMEOUT", "0.3")
        .set("CONFORMANCE_READY_TIMEOUT", "30");

    let ran = run(&server, vars, never()).await;

    ran.assert_passed();
    assert_eq!(checks(&server), UNANSWERED + 1);
}

#[tokio::test]
async fn readiness_unimplemented_health_falls_back_to_an_authenticated_ping() {
    let server = FakeLive::start(|method, _| match method {
        "Check" => Answer::Fail(Status::unimplemented(SECRET)),
        _ => Answer::Ok,
    })
    .await;

    let ran = run(&server, Vars::base(), never()).await;

    ran.assert_passed();
    let received = server.received();
    let methods: Vec<&str> = received.iter().map(|r| r.method.as_str()).collect();
    assert_eq!(methods, ["Check", "Ping"]);
    assert_eq!(received[0].authorization, None);
    assert_eq!(
        received[1].authorization.as_deref(),
        Some(format!("Bearer {TOKEN}").as_str())
    );
}

#[tokio::test]
async fn readiness_refused_ping_fallback_fails_the_preflight_naming_only_the_code() {
    let server = FakeLive::start(|method, _| match method {
        "Check" => Answer::Fail(Status::unimplemented("no health service")),
        "Ping" => Answer::Fail(Status::unauthenticated(SECRET)),
        _ => Answer::Ok,
    })
    .await;

    let ran = run(&server, Vars::base(), never()).await;

    ran.assert_preflight_failed("UNAUTHENTICATED");
    assert!(!ran.xml.contains(SECRET), "{}", ran.xml);
    assert_eq!(server.methods(), ["Check", "Ping"]);
}

#[tokio::test]
async fn readiness_unavailable_until_the_ready_timeout_fails_did_not_become_serving() {
    let server = FakeLive::start(|method, _| match method {
        "Check" => Answer::Fail(Status::unavailable(SECRET)),
        _ => Answer::Ok,
    })
    .await;
    let vars = Vars::base().set("CONFORMANCE_READY_TIMEOUT", "1");

    let ran = run(&server, vars, never()).await;

    ran.assert_preflight_failed(&format!(
        "LiveService did not become SERVING at {}",
        server.endpoint
    ));
    assert!(!ran.xml.contains(SECRET), "{}", ran.xml);
    // Retried, every quarter second, until the readiness timeout.
    assert!(checks(&server) >= 2, "{:?}", server.methods());
}

#[tokio::test]
async fn readiness_any_other_code_fails_at_once_naming_only_the_code() {
    // A server's CANCELLED among them: only the client's own deadline is
    // retried.
    for (status, name) in [
        (Status::permission_denied(SECRET), "PERMISSION_DENIED"),
        (Status::unauthenticated(SECRET), "UNAUTHENTICATED"),
        (Status::internal(SECRET), "INTERNAL"),
        (Status::cancelled(SECRET), "CANCELLED"),
    ] {
        let server = FakeLive::start(move |method, _| match method {
            "Check" => Answer::Fail(status.clone()),
            _ => Answer::Ok,
        })
        .await;

        let ran = run(&server, Vars::base(), never()).await;

        ran.assert_preflight_failed(&format!("Readiness failed with {name}"));
        assert!(!ran.xml.contains(SECRET), "{}", ran.xml);
        assert_eq!(server.methods(), ["Check"], "{name}: not retried");
    }
}

#[tokio::test]
async fn readiness_peer_endpoint_is_waited_for_too() {
    let server = FakeLive::ok().await;
    let peer = FakeLive::start(|method, n| match (method, n) {
        ("Check", 1 | 2) => Answer::NotServing,
        _ => Answer::Ok,
    })
    .await;
    let vars = Vars::base().set("CONFORMANCE_PEER_ENDPOINT", &peer.endpoint);

    let ran = run(&server, vars, never()).await;

    ran.assert_passed();
    assert_eq!(server.methods(), ["Check"]);
    assert_eq!(peer.methods(), ["Check", "Check", "Check"]);
}

#[tokio::test]
async fn readiness_peer_endpoint_that_fails_fails_the_preflight() {
    let server = FakeLive::ok().await;
    let peer = FakeLive::start(|method, _| match method {
        "Check" => Answer::Fail(Status::permission_denied(SECRET)),
        _ => Answer::Ok,
    })
    .await;
    let vars = Vars::base().set("CONFORMANCE_PEER_ENDPOINT", &peer.endpoint);

    let ran = run(&server, vars, never()).await;

    ran.assert_preflight_failed("Readiness failed with PERMISSION_DENIED");
    assert_eq!(peer.methods(), ["Check"]);
}

#[tokio::test]
async fn readiness_interrupted_while_waiting_writes_the_preflight_error_and_no_summary() {
    // SIGTERM arrives while the first health check waits for its answer.
    let (signal, interrupt) = trigger();
    let server = FakeLive::start({
        let signal = signal.clone();
        move |method, _| match method {
            "Check" => {
                signal.fire();
                Answer::Hang
            }
            _ => Answer::Ok,
        }
    })
    .await;
    let vars = Vars::base()
        .set("CONFORMANCE_TIMEOUT", "600")
        .set("CONFORMANCE_READY_TIMEOUT", "600");

    let ran = run(&server, vars, interrupt).await;

    assert!(signal.fired_at().is_some());
    ran.assert_preflight_failed("interrupted by SIGTERM or SIGINT");
}
