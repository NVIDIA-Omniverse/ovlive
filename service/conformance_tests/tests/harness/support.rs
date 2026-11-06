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

//! Helpers shared by the harness tests: settings from a variable map,
//! interrupts, report files, the `grpc-timeout` header, and peers that never
//! answer.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::FutureExt as _;
use live_edit_conformance::config::Settings;
use live_edit_conformance::runner::{Interrupt, Reports, RESULTS_FILE, SUMMARY_FILE};
use tokio::time::Instant;

/// The bearer token every test configuration uses.
pub const TOKEN: &str = "local-token";

/// A configuration as `CONFORMANCE_*` variables: a token and a base path,
/// plus whatever a test sets.
pub struct Vars(HashMap<String, String>);

impl Vars {
    /// [`TOKEN`] and the base path `/container/conformance`.
    pub fn base() -> Self {
        Self(HashMap::from([
            ("CONFORMANCE_AUTH_TOKEN".to_string(), TOKEN.to_string()),
            (
                "CONFORMANCE_BASE_PATH".to_string(),
                "/container/conformance".to_string(),
            ),
        ]))
    }

    /// Sets (or replaces) `name`.
    pub fn set(mut self, name: &str, value: impl Into<String>) -> Self {
        self.0.insert(name.to_string(), value.into());
        self
    }

    /// Unsets `name`.
    pub fn unset(mut self, name: &str) -> Self {
        self.0.remove(name);
        self
    }

    /// Writes reports to `dir`.
    pub fn reports_in(self, dir: &Path) -> Self {
        self.set("CONFORMANCE_REPORT_DIR", dir.to_str().unwrap())
    }

    fn lookup(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned()
    }

    /// The validated settings; panics on a refused configuration.
    pub fn settings(&self) -> Settings {
        Settings::from_lookup(&|name| self.lookup(name)).expect("a valid test configuration")
    }

    pub fn arc_settings(&self) -> Arc<Settings> {
        Arc::new(self.settings())
    }

    /// The run's reports, as the executable makes them.
    pub fn reports(&self) -> Reports {
        Reports::from_lookup(&|name| self.lookup(name))
    }
}

/// An interrupt that never fires.
pub fn never() -> Interrupt {
    futures::future::pending::<()>().boxed().shared()
}

/// An interrupt that has already fired: SIGTERM arrived before the step
/// under test started waiting.
pub fn fired() -> Interrupt {
    futures::future::ready(()).boxed().shared()
}

/// An interrupt the test fires by sending on the returned sender (dropping
/// the sender never fires it).
pub fn interrupt() -> (tokio::sync::oneshot::Sender<()>, Interrupt) {
    let (fire, fired) = tokio::sync::oneshot::channel::<()>();
    let interrupt = async move {
        // A dropped sender is no signal: never fire then.
        if fired.await.is_err() {
            futures::future::pending::<()>().await;
        }
    }
    .boxed()
    .shared();
    (fire, interrupt)
}

/// The sending side of an interrupt that something other than the test body,
/// such as a [`crate::fake_live::FakeLive`] script, fires when it sees the
/// moment the test is about.
pub struct Trigger {
    fire: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    fired_at: Mutex<Option<Instant>>,
}

impl Trigger {
    /// Fires the interrupt the first time; later calls do nothing.
    pub fn fire(&self) {
        if let Some(fire) = self.fire.lock().unwrap().take() {
            *self.fired_at.lock().unwrap() = Some(Instant::now());
            fire.send(()).unwrap();
        }
    }

    /// When the interrupt was fired, if it was.
    pub fn fired_at(&self) -> Option<Instant> {
        *self.fired_at.lock().unwrap()
    }
}

/// An interrupt fired through the returned [`Trigger`].
pub fn trigger() -> (Arc<Trigger>, Interrupt) {
    let (fire, interrupt) = interrupt();
    let trigger = Trigger {
        fire: Mutex::new(Some(fire)),
        fired_at: Mutex::new(None),
    };
    (Arc::new(trigger), interrupt)
}

/// The JUnit report in `dir`.
pub fn junit(dir: &Path) -> String {
    std::fs::read_to_string(dir.join(RESULTS_FILE)).expect("the JUnit report was written")
}

/// The summary in `dir`.
pub fn summary(dir: &Path) -> serde_json::Value {
    serde_json::from_slice(
        &std::fs::read(dir.join(SUMMARY_FILE)).expect("summary.json was written"),
    )
    .expect("summary.json is JSON")
}

/// The `<testcase>` element of the case named `name` in `xml`, up to and
/// including its end.
pub fn testcase<'a>(xml: &'a str, name: &str) -> &'a str {
    let start = xml
        .find(&format!(" name=\"{name}\""))
        .unwrap_or_else(|| panic!("no testcase {name} in {xml}"));
    let start = xml[..start].rfind("<testcase").unwrap();
    let rest = &xml[start..];
    let end = if let Some(open) = rest.find("/>\n").filter(|at| !rest[..*at].contains('\n')) {
        open + 3
    } else {
        rest.find("</testcase>").expect("a closed testcase") + "</testcase>".len()
    };
    &rest[..end]
}

/// A `grpc-timeout` header value as a duration.
pub fn grpc_timeout(value: &str) -> Duration {
    let (amount, unit) = value.split_at(value.len() - 1);
    let amount: u64 = amount.parse().expect("a grpc-timeout amount");
    match unit {
        "H" => Duration::from_secs(amount * 3600),
        "M" => Duration::from_secs(amount * 60),
        "S" => Duration::from_secs(amount),
        "m" => Duration::from_millis(amount),
        "u" => Duration::from_micros(amount),
        "n" => Duration::from_nanos(amount),
        other => panic!("not a grpc-timeout unit: {other}"),
    }
}

/// A TCP listener that never accepts: a connection to it completes in the
/// kernel's backlog and no byte is ever answered, so only the client can end
/// a call to it. Keep it alive for the test.
pub fn silent_listener() -> (std::net::TcpListener, String) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    (listener, endpoint)
}

/// A peer that accepts every connection and reads everything sent to it but
/// never writes a byte: the client's HTTP/2 connection is established and
/// its requests go out, and no answer, not even the server's HTTP/2
/// settings, ever comes back. Unlike [`silent_listener`], a call to it
/// reaches tonic's channel and so its `grpc-timeout` timer. Stopped when
/// dropped (connections already accepted stay open until the runtime ends).
pub struct SilentServer {
    /// `http://127.0.0.1:<port>`.
    pub endpoint: String,
    bytes: tokio::sync::watch::Receiver<usize>,
    server: tokio::task::JoinHandle<()>,
}

impl SilentServer {
    pub async fn start() -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (count, bytes) = tokio::sync::watch::channel(0usize);
        let count = Arc::new(count);
        let server = tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let count = count.clone();
                tokio::spawn(async move {
                    use tokio::io::AsyncReadExt as _;
                    let mut buffer = [0u8; 4096];
                    while let Ok(read) = socket.read(&mut buffer).await {
                        if read == 0 {
                            break;
                        }
                        count.send_modify(|total| *total += read);
                    }
                });
            }
        });
        SilentServer {
            endpoint,
            bytes,
            server,
        }
    }

    /// Waits up to `budget` until a client has sent something (a
    /// connection's first bytes go out with its first call).
    pub async fn wait_for_a_call(&self, budget: Duration) {
        let mut bytes = self.bytes.clone();
        tokio::time::timeout(budget, bytes.wait_for(|total| *total > 0))
            .await
            .expect("a call reached the silent server")
            .expect("the silent server is running");
    }
}

impl Drop for SilentServer {
    fn drop(&mut self) {
        self.server.abort();
    }
}
