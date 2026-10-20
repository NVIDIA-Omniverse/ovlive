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

//! The `live-edit` binary exports traces and metrics to an `https://` OTLP
//! collector over TLS, end to end.
//!
//! The test runs a TLS collector in-process: `rcgen` generates a throwaway CA
//! and a server certificate for `127.0.0.1` at run time (no key material is
//! committed), a `tokio-rustls` acceptor on the `ring` provider terminates
//! TLS with ALPN `h2`, and an `h2` server records the `:path` of each request
//! and answers it with an empty gRPC OK. The child gets the CA through
//! `SSL_CERT_FILE` (so it trusts exactly that CA, and the certificate check
//! is real) and both endpoint variables pointed at the collector.
//!
//! **Making the child export.** The child runs as in `startup_log_binary`
//! (cleared environment, JSON logs, a pod memory fixture), except that it
//! binds an ephemeral gRPC port and keeps running: its Valkey URL points at
//! a port the test holds and never accepts from, so the startup connect
//! waits there for `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS`. Meanwhile:
//!
//! - **Metrics:** `OTEL_METRIC_EXPORT_INTERVAL` is short, so the push
//!   reader exports on its own timer.
//! - **Traces:** the test sends one `LiveService` `Ping` to the bootstrap
//!   server. The bootstrap server does not serve the `LiveService` and
//!   answers `UNIMPLEMENTED`, but its `MetricsLayer` opens the RPC's INFO
//!   server span for every known `LiveService` method, and the short
//!   `OTEL_BSP_SCHEDULE_DELAY` exports it.
//!
//! The test waits, by channel and with one deadline, until the collector has
//! seen an Export request on both OTLP service paths, then kills the child.
//! No exit or shutdown flush is involved.

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use rustls::pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio::io::{AsyncBufReadExt as _, AsyncReadExt as _, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_rustls::TlsAcceptor;

use crate::common::proto;
use crate::startup_log_binary::{hold_port, pod_memory_fixture, AUTH_ENV, LIVE_EDIT_BIN};

/// The gRPC path of an OTLP trace export (opentelemetry-proto 0.31,
/// `opentelemetry.proto.collector.trace.v1`).
const TRACE_EXPORT_PATH: &str = "/opentelemetry.proto.collector.trace.v1.TraceService/Export";

/// The gRPC path of an OTLP metrics export (opentelemetry-proto 0.31,
/// `opentelemetry.proto.collector.metrics.v1`).
const METRICS_EXPORT_PATH: &str = "/opentelemetry.proto.collector.metrics.v1.MetricsService/Export";

/// One deadline for the whole run, from spawning the child to the collector
/// seeing both export paths. The healthy path takes under a second (startup,
/// then one 100 ms export period on each signal); the budget only has to
/// absorb a loaded runner, and stays under nextest's 120 s hang-killer.
const RUN_BUDGET: Duration = Duration::from_secs(60);

/// How long the child's startup Valkey connect may wait on the held port:
/// longer than [`RUN_BUDGET`], so the child is still running, and exporting,
/// when the test gives up on it.
const CHILD_VALKEY_CONNECT_TIMEOUT_SECS: &str = "120";

/// The child's span-export delay and metrics-export interval, in ms.
const CHILD_EXPORT_PERIOD_MS: &str = "100";

/// Prefix of the INFO line carrying the bootstrap server's bound address.
const BIND_SUCCESS_MESSAGE_PREFIX: &str = "gRPC listener bound";

/// The two export startup lines, each of which carries the transport.
const EXPORT_ENABLED_MESSAGES: [&str; 2] = [
    "OpenTelemetry trace export enabled",
    "OTLP metrics push exporter enabled",
];

/// An empty `Export*ServiceResponse`: one uncompressed gRPC message of length
/// zero.
const EMPTY_GRPC_MESSAGE: [u8; 5] = [0, 0, 0, 0, 0];

/// The in-test collector: the CA file the child trusts, its address, and the
/// request paths it has seen.
struct TlsCollector {
    ca_file: tempfile::NamedTempFile,
    addr: SocketAddr,
    paths: mpsc::UnboundedReceiver<String>,
}

impl TlsCollector {
    async fn start() -> Self {
        let ca_key = rcgen::KeyPair::generate().expect("generate the CA key");
        let mut ca_params =
            rcgen::CertificateParams::new(Vec::<String>::new()).expect("CA parameters");
        ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        ca_params
            .distinguished_name
            .push(rcgen::DnType::CommonName, "live-edit test collector CA");
        ca_params.key_usages = vec![
            rcgen::KeyUsagePurpose::KeyCertSign,
            rcgen::KeyUsagePurpose::DigitalSignature,
        ];
        let ca_cert = ca_params.self_signed(&ca_key).expect("self-sign the CA");
        let issuer = rcgen::Issuer::new(ca_params, ca_key);

        let server_key = rcgen::KeyPair::generate().expect("generate the server key");
        let mut server_params = rcgen::CertificateParams::new(vec!["127.0.0.1".to_string()])
            .expect("server parameters");
        server_params
            .distinguished_name
            .push(rcgen::DnType::CommonName, "127.0.0.1");
        server_params.use_authority_key_identifier_extension = true;
        server_params.extended_key_usages = vec![rcgen::ExtendedKeyUsagePurpose::ServerAuth];
        let server_cert = server_params
            .signed_by(&server_key, &issuer)
            .expect("sign the server certificate");

        let ca_file = tempfile::NamedTempFile::new().expect("create the CA file");
        std::fs::write(ca_file.path(), ca_cert.pem()).expect("write the CA file");

        let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .expect("the ring provider supports the default protocol versions")
        .with_no_client_auth()
        .with_single_cert(
            vec![server_cert.der().clone()],
            PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(server_key.serialize_der())),
        )
        .expect("the server certificate and key match");
        config.alpn_protocols = vec![b"h2".to_vec()];
        let acceptor = TlsAcceptor::from(Arc::new(config));

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the collector");
        let addr = listener.local_addr().expect("collector address");
        let (paths_tx, paths) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                tokio::spawn(serve_connection(stream, acceptor.clone(), paths_tx.clone()));
            }
        });
        Self {
            ca_file,
            addr,
            paths,
        }
    }

    fn https_url(&self) -> String {
        format!("https://127.0.0.1:{}", self.addr.port())
    }
}

/// Terminate TLS and serve HTTP/2 on one connection: record each request's
/// path, then answer it. A connection whose handshake fails is dropped.
async fn serve_connection(
    stream: TcpStream,
    acceptor: TlsAcceptor,
    paths: mpsc::UnboundedSender<String>,
) {
    let Ok(tls) = acceptor.accept(stream).await else {
        return;
    };
    let Ok(mut connection) = h2::server::handshake(tls).await else {
        return;
    };
    while let Some(Ok((request, respond))) = connection.accept().await {
        let _ = paths.send(request.uri().path().to_owned());
        tokio::spawn(answer_export(request, respond));
    }
}

/// Read the request body to its end, then answer with an empty gRPC OK.
async fn answer_export(
    request: http::Request<h2::RecvStream>,
    mut respond: h2::server::SendResponse<Bytes>,
) {
    let mut body = request.into_body();
    while let Some(Ok(chunk)) = body.data().await {
        let _ = body.flow_control().release_capacity(chunk.len());
    }
    let response = http::Response::builder()
        .status(200)
        .header("content-type", "application/grpc")
        .body(())
        .expect("a static response");
    let Ok(mut send) = respond.send_response(response, false) else {
        return;
    };
    if send
        .send_data(Bytes::from_static(&EMPTY_GRPC_MESSAGE), false)
        .is_err()
    {
        return;
    }
    let mut trailers = http::HeaderMap::new();
    trailers.insert("grpc-status", http::HeaderValue::from_static("0"));
    let _ = send.send_trailers(trailers);
}

/// The port in a bound-line `addr` field (`0.0.0.0:<port>`), or an error
/// naming the line when the field is missing or not a socket address.
fn bound_port(line: &serde_json::Value) -> Result<u16, String> {
    line["fields"]["addr"]
        .as_str()
        .and_then(|addr| addr.parse::<SocketAddr>().ok())
        .map(|addr| addr.port())
        .ok_or_else(|| format!("the bound line carries no parseable `addr` field: {line}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_edit_binary_exports_traces_and_metrics_to_an_https_collector_over_tls() {
    let mut collector = TlsCollector::start().await;
    let silent_valkey = hold_port();
    let cgroup = pod_memory_fixture();
    let url = collector.https_url();
    let valkey_url = format!("redis://127.0.0.1:{}", silent_valkey.port);
    let ca_file = collector
        .ca_file
        .path()
        .to_str()
        .expect("UTF-8 temp path")
        .to_owned();
    let cgroup_root = cgroup.path().to_str().expect("UTF-8 temp path").to_owned();
    let case = [
        ("LIVE_EDIT_LOG_FORMAT", "json"),
        ("LIVE_EDIT_PORT", "0"),
        ("LIVE_EDIT_METRICS_PORT", "0"),
        ("LIVE_EDIT_VALKEY_URL", valkey_url.as_str()),
        (
            "LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS",
            CHILD_VALKEY_CONNECT_TIMEOUT_SECS,
        ),
        ("MEASUREMENT_POD_MEMORY_CGROUP_ROOT", cgroup_root.as_str()),
        ("OTEL_EXPORTER_OTLP_ENDPOINT", url.as_str()),
        ("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", url.as_str()),
        ("OTEL_BSP_SCHEDULE_DELAY", CHILD_EXPORT_PERIOD_MS),
        ("OTEL_METRIC_EXPORT_INTERVAL", CHILD_EXPORT_PERIOD_MS),
        ("SSL_CERT_FILE", ca_file.as_str()),
    ];

    let mut child = Command::new(LIVE_EDIT_BIN)
        .env_clear()
        .envs(AUTH_ENV.iter().chain(case.iter()).copied())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn the live-edit binary");

    // Every stdout line is kept for diagnostics and the parsed records are
    // forwarded to the test; stderr is drained to its end.
    let stdout_text = Arc::new(Mutex::new(String::new()));
    let (records_tx, mut records) = mpsc::unbounded_channel::<serde_json::Value>();
    let mut stdout_lines = BufReader::new(child.stdout.take().expect("piped stdout")).lines();
    let stdout_keep = Arc::clone(&stdout_text);
    tokio::spawn(async move {
        while let Ok(Some(line)) = stdout_lines.next_line().await {
            stdout_keep.lock().unwrap().push_str(&line);
            stdout_keep.lock().unwrap().push('\n');
            if let Ok(record) = serde_json::from_str(&line) {
                let _ = records_tx.send(record);
            }
        }
    });
    let mut stderr = child.stderr.take().expect("piped stderr");
    let stderr_task = tokio::spawn(async move {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text).await;
        text
    });

    let mut export_lines = Vec::new();
    let mut seen = BTreeSet::new();
    // Every way the run can end early is an `Err` carried out of the timed
    // block, never a panic inside it, so the child's output is always printed.
    let outcome: Result<Result<(), String>, _> = tokio::time::timeout(RUN_BUDGET, async {
        // The bootstrap server's port, from its bound line.
        let port = loop {
            let Some(record) = records.recv().await else {
                return Err("the child's stdout ended before it bound its gRPC port".to_string());
            };
            let message = record["fields"]["message"].as_str().unwrap_or_default();
            if EXPORT_ENABLED_MESSAGES.contains(&message) {
                export_lines.push(record.clone());
            }
            if message.starts_with(BIND_SUCCESS_MESSAGE_PREFIX) {
                break bound_port(&record)?;
            }
        };

        // One LiveService RPC, for its server span. The bootstrap server
        // answers it UNIMPLEMENTED; only the span matters here.
        let mut client = proto::live_service_client::LiveServiceClient::connect(format!(
            "http://127.0.0.1:{port}"
        ))
        .await
        .map_err(|error| format!("connecting to the child's bootstrap server failed: {error}"))?;
        let _ = client
            .ping(proto::PingRequest {
                client_timestamp_microseconds_since_epoch: 0,
            })
            .await;

        while !(seen.contains(TRACE_EXPORT_PATH) && seen.contains(METRICS_EXPORT_PATH)) {
            let Some(path) = collector.paths.recv().await else {
                return Err("the collector stopped".to_string());
            };
            seen.insert(path);
        }
        Ok(())
    })
    .await;

    child.start_kill().ok();
    let status = child.wait().await.expect("reap the child");
    let stderr_text = stderr_task.await.unwrap_or_default();
    let stdout_text = stdout_text.lock().unwrap().clone();
    let failure = match outcome {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(error),
        Err(_) => Some(format!("{RUN_BUDGET:?} passed")),
    };
    assert!(
        failure.is_none(),
        "within {RUN_BUDGET:?} the https:// collector must receive an Export request on both \
         {TRACE_EXPORT_PATH} and {METRICS_EXPORT_PATH}; {}, and it saw {seen:?}. Child status \
         {status:?}\nstdout: {stdout_text}\nstderr: {stderr_text}",
        failure.unwrap_or_default()
    );

    for message in EXPORT_ENABLED_MESSAGES {
        let line = export_lines
            .iter()
            .find(|line| line["fields"]["message"] == message)
            .unwrap_or_else(|| panic!("the child logs {message:?}; stdout: {stdout_text}"));
        assert_eq!(line["fields"]["transport"], "tls", "{line}");
    }
}
