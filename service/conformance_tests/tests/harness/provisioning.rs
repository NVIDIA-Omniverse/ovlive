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

//! `deployment::provision` against a wiremock Storage API: the requests it
//! sends, what it tolerates, the interrupt windows before the suite runs, and
//! the bound on the `DELETE` once an interrupt has arrived. (The caller's
//! handling of a failure, the `deployment-fixture` report and exit 2, is in
//! `lib.rs`'s tests and in `binary`.)
//!
//! The `DELETE` bound tests wait it out in real time (10 s or more each):
//! wiremock answers over the network, which paused time could overtake.

use std::future::Ready;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use live_edit_conformance::config::Settings;
use live_edit_conformance::deployment::{provision, ProvisionError, INTERRUPT_DELETE_BOUND};
use percent_encoding::percent_decode_str;
use tokio::time::Instant;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use crate::support::{fired, interrupt, never, Vars, TOKEN};

/// The base path provisioning creates its file under; the space shows the
/// path is percent-encoded.
const BASE: &str = "/container/conformance runs";

/// The route prefix of a file by its path.
const BY_ADDRESS: &str = "/v1alpha/fileobject/by-address/";

fn vars(server: &MockServer) -> Vars {
    Vars::base()
        .set("CONFORMANCE_BASE_PATH", BASE)
        .set("CONFORMANCE_STORAGE_ENDPOINT", server.uri())
}

/// Answers `verb` with `status`.
async fn answer(server: &MockServer, verb: &str, status: u16) {
    Mock::given(method(verb))
        .respond_with(ResponseTemplate::new(status))
        .mount(server)
        .await;
}

async fn received(server: &MockServer) -> Vec<Request> {
    server.received_requests().await.expect("recording is on")
}

/// The file path a request names, decoded, after checking the encoded path
/// is the by-address route with the whole file path as one segment.
fn file_path(request: &Request) -> String {
    let encoded = request.url.path();
    let segment = encoded
        .strip_prefix(BY_ADDRESS)
        .unwrap_or_else(|| panic!("not the by-address route: {encoded}"));
    assert!(!segment.contains('/'), "the path is one segment: {segment}");
    percent_decode_str(segment)
        .decode_utf8()
        .unwrap()
        .into_owned()
}

/// A run that records the base path it was given and returns `code`.
fn recording_run(
    code: i32,
) -> (
    Arc<Mutex<Option<String>>>,
    impl FnOnce(Settings) -> Ready<i32>,
) {
    let seen = Arc::new(Mutex::new(None));
    let run = {
        let seen = seen.clone();
        move |settings: Settings| {
            *seen.lock().unwrap() = Some(settings.base_path);
            std::future::ready(code)
        }
    };
    (seen, run)
}

/// A run that must not happen; the flag shows whether it did.
fn forbidden_run() -> (Arc<AtomicBool>, impl FnOnce(Settings) -> Ready<i32>) {
    let ran = Arc::new(AtomicBool::new(false));
    let run = {
        let ran = ran.clone();
        move |_: Settings| {
            ran.store(true, Ordering::SeqCst);
            std::future::ready(0)
        }
    };
    (ran, run)
}

#[tokio::test]
async fn provision_puts_runs_and_deletes_one_uuid_file_under_the_base_path() {
    let server = MockServer::start().await;
    answer(&server, "PUT", 201).await;
    answer(&server, "DELETE", 204).await;
    let (seen, run) = recording_run(1);

    let result = provision(&vars(&server).settings(), never(), run).await;

    // The suite's exit code is passed through.
    assert_eq!(result, Ok(1));
    let requests = received(&server).await;
    let methods: Vec<&str> = requests.iter().map(|r| r.method.as_str()).collect();
    assert_eq!(methods, ["PUT", "DELETE"]);
    let (put, delete) = (&requests[0], &requests[1]);
    let path = file_path(put);
    assert_eq!(file_path(delete), path, "the DELETE names the PUT's file");
    let name = path
        .strip_prefix(&format!("{BASE}/conformance-"))
        .and_then(|rest| rest.strip_suffix(".usda"))
        .unwrap_or_else(|| panic!("not a conformance-<uuid>.usda under the base: {path}"));
    assert!(
        name.len() == 32 && name.chars().all(|c| c.is_ascii_hexdigit()),
        "{name}"
    );
    assert_eq!(seen.lock().unwrap().as_deref(), Some(path.as_str()));
    assert_eq!(put.body, b"#usda 1.0\n");
    assert_eq!(put.url.query(), Some("data_object_size=10"));
    assert_eq!(delete.url.query(), None);
    for request in &requests {
        assert_eq!(
            request.headers.get("authorization").unwrap(),
            &format!("Bearer {TOKEN}")
        );
    }
}

#[tokio::test]
async fn provision_runs_get_a_fresh_file_each_time() {
    let server = MockServer::start().await;
    answer(&server, "PUT", 201).await;
    answer(&server, "DELETE", 204).await;
    let settings = vars(&server).settings();

    provision(&settings, never(), |_| std::future::ready(0))
        .await
        .unwrap();
    provision(&settings, never(), |_| std::future::ready(0))
        .await
        .unwrap();

    let requests = received(&server).await;
    assert_ne!(file_path(&requests[0]), file_path(&requests[2]));
}

#[tokio::test]
async fn provision_failed_put_still_deletes_and_never_runs_the_suite() {
    let server = MockServer::start().await;
    answer(&server, "PUT", 500).await;
    answer(&server, "DELETE", 204).await;
    let (ran, run) = forbidden_run();

    let result = provision(&vars(&server).settings(), never(), run).await;

    assert_eq!(
        result,
        Err(ProvisionError::HttpStatus {
            method: "PUT",
            status: 500
        })
    );
    assert!(!ran.load(Ordering::SeqCst));
    let requests = received(&server).await;
    let methods: Vec<&str> = requests.iter().map(|r| r.method.as_str()).collect();
    assert_eq!(methods, ["PUT", "DELETE"]);
    assert_eq!(file_path(&requests[0]), file_path(&requests[1]));
}

#[tokio::test]
async fn provision_failed_delete_is_the_failure_whatever_the_suite_returned() {
    let server = MockServer::start().await;
    answer(&server, "PUT", 200).await;
    answer(&server, "DELETE", 500).await;
    let (seen, run) = recording_run(0);

    let result = provision(&vars(&server).settings(), never(), run).await;

    assert_eq!(
        result,
        Err(ProvisionError::HttpStatus {
            method: "DELETE",
            status: 500
        })
    );
    assert!(seen.lock().unwrap().is_some(), "the suite ran");
}

#[tokio::test]
async fn provision_delete_answered_404_is_tolerated() {
    let server = MockServer::start().await;
    answer(&server, "PUT", 201).await;
    answer(&server, "DELETE", 404).await;

    let result = provision(&vars(&server).settings(), never(), |_| {
        std::future::ready(0)
    })
    .await;

    assert_eq!(result, Ok(0));
}

#[tokio::test]
async fn provision_redirect_is_never_followed() {
    let elsewhere = MockServer::start().await;
    answer(&elsewhere, "PUT", 201).await;
    let server = MockServer::start().await;
    Mock::given(method("PUT"))
        .respond_with(
            ResponseTemplate::new(302)
                .insert_header("location", format!("{}/elsewhere", elsewhere.uri())),
        )
        .mount(&server)
        .await;
    answer(&server, "DELETE", 204).await;
    let (ran, run) = forbidden_run();

    let result = provision(&vars(&server).settings(), never(), run).await;

    assert_eq!(result, Err(ProvisionError::Redirect));
    assert!(!ran.load(Ordering::SeqCst));
    assert!(
        received(&elsewhere).await.is_empty(),
        "the credential was sent to the redirect target"
    );
}

#[tokio::test]
async fn provision_transport_failure_is_reported_by_kind_only() {
    let (ran, run) = forbidden_run();
    let settings = Vars::base()
        .set("CONFORMANCE_STORAGE_ENDPOINT", "http://127.0.0.1:9")
        .settings();

    let result = provision(&settings, never(), run).await;

    assert_eq!(result, Err(ProvisionError::Transport));
    assert_eq!(ProvisionError::Transport.kind(), "Transport");
    assert!(!ran.load(Ordering::SeqCst));
}

#[tokio::test]
async fn provision_interrupted_while_waiting_for_the_token_file_sends_nothing() {
    let server = MockServer::start().await;
    answer(&server, "PUT", 201).await;
    answer(&server, "DELETE", 204).await;
    let dir = tempfile::tempdir().unwrap();
    // The sidecar has not written the token yet; without the interrupt the
    // wait would last the readiness timeout.
    let settings = vars(&server)
        .unset("CONFORMANCE_AUTH_TOKEN")
        .set(
            "CONFORMANCE_AUTH_TOKEN_FILE",
            dir.path().join("token").to_str().unwrap(),
        )
        .set("CONFORMANCE_READY_TIMEOUT", "30")
        .settings();
    let (ran, run) = forbidden_run();

    let result = provision(&settings, fired(), run).await;

    assert_eq!(result, Err(ProvisionError::Interrupted));
    assert!(!ran.load(Ordering::SeqCst));
    assert!(received(&server).await.is_empty(), "no request is sent");
}

#[tokio::test]
async fn provision_interrupted_during_the_put_still_deletes_and_never_runs_the_suite() {
    let server = MockServer::start().await;
    // The PUT's answer would come long after the interrupt.
    Mock::given(method("PUT"))
        .respond_with(ResponseTemplate::new(201).set_delay(Duration::from_secs(600)))
        .mount(&server)
        .await;
    answer(&server, "DELETE", 204).await;
    let settings = vars(&server).settings();
    let (ran, run) = forbidden_run();
    let (fire, interrupt) = interrupt();

    let provisioning = provision(&settings, interrupt, run);
    // SIGTERM arrives once the server has the PUT.
    let signal = async {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while received(&server).await.is_empty() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the PUT never arrived"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        fire.send(()).unwrap();
    };
    let (result, ()) = tokio::join!(provisioning, signal);

    assert_eq!(result, Err(ProvisionError::Interrupted));
    assert!(!ran.load(Ordering::SeqCst));
    let requests = received(&server).await;
    let methods: Vec<&str> = requests.iter().map(|r| r.method.as_str()).collect();
    assert_eq!(methods, ["PUT", "DELETE"]);
    assert_eq!(file_path(&requests[0]), file_path(&requests[1]));
}

#[tokio::test]
async fn provision_base_path_of_only_slashes_is_refused_before_any_request() {
    let server = MockServer::start().await;
    let settings = vars(&server).set("CONFORMANCE_BASE_PATH", "/").settings();
    let (ran, run) = forbidden_run();

    let result = provision(&settings, never(), run).await;

    assert!(
        matches!(result, Err(ProvisionError::Config(_))),
        "{result:?}"
    );
    assert!(!ran.load(Ordering::SeqCst));
    assert!(received(&server).await.is_empty());
}

// --- The DELETE once SIGTERM or SIGINT has arrived.

/// The answer delay of a `DELETE` that never answers, and the
/// `CONFORMANCE_TIMEOUT` (in seconds) of those tests: longer than any wait
/// measured here, so only the interrupt bound can end the wait.
const NEVER: Duration = Duration::from_secs(600);

/// How long a provisioning whose `DELETE` is bounded may take in all before
/// the test fails rather than waits for [`NEVER`].
const BOUNDED_BUDGET: Duration = Duration::from_secs(60);

/// A Storage API that creates the file and answers its `DELETE` with 204
/// after `delay`.
async fn delete_answered_after(delay: Duration) -> MockServer {
    let server = MockServer::start().await;
    answer(&server, "PUT", 201).await;
    Mock::given(method("DELETE"))
        .respond_with(ResponseTemplate::new(204).set_delay(delay))
        .mount(&server)
        .await;
    server
}

/// Provisioning settings for `server` that leave every wait to the bound.
fn unbounded_vars(server: &MockServer) -> Settings {
    vars(server)
        .set("CONFORMANCE_TIMEOUT", NEVER.as_secs().to_string())
        .settings()
}

/// Waits until `server` has received the `DELETE`.
async fn delete_received(server: &MockServer) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !received(server)
        .await
        .iter()
        .any(|request| request.method.as_str() == "DELETE")
    {
        assert!(Instant::now() < deadline, "the DELETE never arrived");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn provision_unanswered_delete_after_an_interrupt_in_the_suite_ends_at_the_bound() {
    // SIGTERM arrives while the suite runs, and the suite then spends a few
    // seconds on its interrupted case's cleanup (the runner may take up to
    // 10 s) before the DELETE starts: the DELETE still waits the whole bound
    // from its own start, not what is left of it counted from the signal.
    const CLEANUP_AFTER_THE_SIGNAL: Duration = Duration::from_secs(3);
    let server = delete_answered_after(NEVER).await;
    let settings = unbounded_vars(&server);
    let (fire, interrupt) = crate::support::interrupt();
    let suite_ended = Arc::new(Mutex::new(None));
    let run = {
        let suite_ended = suite_ended.clone();
        move |_: Settings| async move {
            fire.send(()).unwrap();
            // Not a synchronization wait: the scenario is a cleanup this long.
            tokio::time::sleep(CLEANUP_AFTER_THE_SIGNAL).await;
            *suite_ended.lock().unwrap() = Some(Instant::now());
            // The runner's exit code for an interrupted run.
            2
        }
    };

    let result = tokio::time::timeout(BOUNDED_BUDGET, provision(&settings, interrupt, run))
        .await
        .expect("the DELETE is bounded after an interrupt");

    // Given up unanswered, as a timed-out request is: the file may be left.
    assert_eq!(result, Err(ProvisionError::Transport));
    let suite_ended = suite_ended.lock().unwrap().expect("the suite ran");
    assert!(suite_ended.elapsed() >= INTERRUPT_DELETE_BOUND);
    let methods: Vec<String> = received(&server)
        .await
        .iter()
        .map(|request| request.method.to_string())
        .collect();
    assert_eq!(methods, ["PUT", "DELETE"]);
}

#[tokio::test]
async fn provision_interrupt_during_the_delete_bounds_it_from_the_signal() {
    // SIGTERM arrives a few seconds into the DELETE's wait: the bound runs
    // from the signal, the later of the two, not from the DELETE's start.
    const INTO_THE_DELETE: Duration = Duration::from_secs(3);
    let server = delete_answered_after(NEVER).await;
    let settings = unbounded_vars(&server);
    let (fire, interrupt) = crate::support::interrupt();

    let provisioning = async {
        let result = tokio::time::timeout(
            BOUNDED_BUDGET,
            provision(&settings, interrupt, |_| std::future::ready(0)),
        )
        .await
        .expect("the DELETE is bounded after an interrupt");
        (result, Instant::now())
    };
    let signal = async {
        delete_received(&server).await;
        let delete_started = Instant::now();
        // Not a synchronization wait: the scenario is a signal that comes
        // this far into the DELETE.
        tokio::time::sleep(INTO_THE_DELETE).await;
        fire.send(()).unwrap();
        (delete_started, Instant::now())
    };
    let ((result, ended), (delete_started, signalled)) = tokio::join!(provisioning, signal);

    assert_eq!(result, Err(ProvisionError::Transport));
    assert!(
        ended - signalled >= INTERRUPT_DELETE_BOUND,
        "ended {:?} after the signal",
        ended - signalled
    );
    assert!(ended - delete_started >= INTERRUPT_DELETE_BOUND + INTO_THE_DELETE);
}

#[tokio::test]
async fn provision_delete_without_an_interrupt_waits_past_the_bound_for_its_answer() {
    // Without SIGTERM or SIGINT the DELETE has only the request timeout:
    // an answer that comes after the interrupt bound is still taken.
    let delay = INTERRUPT_DELETE_BOUND + Duration::from_secs(2);
    let server = delete_answered_after(delay).await;
    let settings = unbounded_vars(&server);
    let started = Instant::now();

    let result = tokio::time::timeout(
        BOUNDED_BUDGET,
        provision(&settings, never(), |_| std::future::ready(0)),
    )
    .await
    .expect("the DELETE is answered");

    assert_eq!(result, Ok(0));
    assert!(started.elapsed() >= delay);
}
