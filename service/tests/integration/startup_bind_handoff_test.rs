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

//! Permanent regression tests for the b.rgs startup restructure: the gRPC
//! listening socket is bound once at the top of `run()` and handed from the
//! bootstrap server (health + reflection + capabilities) to the full-API
//! server without ever leaving `LISTEN`.
//!
//! The load-bearing claim is the shared file descriptor. `bind_grpc_listener`
//! `try_clone`s the listener, so both `TcpIncoming` streams refer to the same
//! open file description; dropping the bootstrap server's stream at handoff
//! therefore does NOT unbind the port, and a client dialling mid-handoff
//! queues in the shared accept backlog instead of getting `ECONNREFUSED`.
//! That is precisely the property a two-socket implementation (bind → drop →
//! rebind) would silently lose, and the failure it would reintroduce — a
//! kubelet liveness probe seeing `connection refused` and restarting a
//! healthy pod — is invisible to every other test in the suite. Hence the
//! tight dial loop below: it is the only assertion that can observe it.
//!
//! Hermetic by construction: no Valkey, no auth backends, no env vars. Only
//! loopback sockets and a `tonic-health` service.

use std::io::ErrorKind;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use tokio::net::TcpStream;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tonic::transport::{Channel, Server};
use tonic_health::pb::{
    health_check_response::ServingStatus as PbServingStatus, health_client::HealthClient,
    HealthCheckRequest,
};
use tonic_health::ServingStatus;

use live_edit::observability::health::{create_health_service, health_service_from_reporter};
use live_edit::observability::startup::bind_grpc_listener;

/// Readiness service name — the same protobuf FQN `main.rs` registers.
const READINESS_SERVICE_NAME: &str = "nvidia.omniverse.live.v1alpha.LiveService";

/// Liveness service name (the gRPC health protocol's "overall server" key).
const OVERALL_SERVICE: &str = "";

/// `TCP_KEEPALIVE` time requested for accepted connections. Mirrors
/// `main.rs`'s `GRPC_TCP_KEEPALIVE` so the socket-option assertions below
/// pin the production value, not an arbitrary test one.
const KEEPALIVE: Duration = Duration::from_secs(30);

/// Dial cadence for the connection-refused probe. 10 ms is tight enough that
/// even a sub-100 ms unbind/rebind window would be sampled several times.
const DIAL_CADENCE: Duration = Duration::from_millis(10);

/// Outer budget for any single bounded wait in these tests. Generous — every
/// operation here is loopback-local, so hitting it means something wedged.
const WAIT_BUDGET: Duration = Duration::from_secs(10);

fn loopback_port_zero() -> SocketAddr {
    "127.0.0.1:0".parse().expect("literal addr must parse")
}

// ───────────────────────────────────────────────────────────────────────────
// Dial probe — the connection-refused detector
// ───────────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct DialStats {
    attempts: AtomicUsize,
    connected: AtomicUsize,
    refused: AtomicUsize,
    /// Non-refusal failures (timeouts, resets). Recorded rather than asserted
    /// on directly: a full accept backlog manifests as a dropped SYN and a
    /// client-side stall, which is a different (and non-fatal) condition from
    /// the refusal this test exists to catch.
    other: Mutex<Vec<String>>,
}

/// A background task that dials `addr` every [`DIAL_CADENCE`] and classifies
/// each outcome. `ConnectionRefused` is the failure mode under test: it is
/// what a client sees when the listening socket has gone away, and it is
/// impossible while any descriptor for the original open file description
/// remains alive.
struct DialProbe {
    stats: Arc<DialStats>,
    stop: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

impl DialProbe {
    fn start(addr: SocketAddr) -> Self {
        let stats = Arc::new(DialStats::default());
        let stop = Arc::new(AtomicBool::new(false));

        let task_stats = Arc::clone(&stats);
        let task_stop = Arc::clone(&stop);
        let handle = tokio::spawn(async move {
            while !task_stop.load(Ordering::Relaxed) {
                match tokio::time::timeout(Duration::from_secs(2), TcpStream::connect(addr)).await {
                    Ok(Ok(stream)) => {
                        task_stats.connected.fetch_add(1, Ordering::Relaxed);
                        // Drop immediately: the point is that the three-way
                        // handshake completed, not that anything was spoken.
                        drop(stream);
                    }
                    Ok(Err(e)) if e.kind() == ErrorKind::ConnectionRefused => {
                        task_stats.refused.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(Err(e)) => task_stats
                        .other
                        .lock()
                        .expect("dial-probe mutex must not be poisoned")
                        .push(format!("{:?}: {e}", e.kind())),
                    Err(_) => task_stats
                        .other
                        .lock()
                        .expect("dial-probe mutex must not be poisoned")
                        .push("connect timed out after 2s".to_string()),
                }
                task_stats.attempts.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(DIAL_CADENCE).await;
            }
        });

        Self {
            stats,
            stop,
            handle,
        }
    }

    fn attempts(&self) -> usize {
        self.stats.attempts.load(Ordering::Relaxed)
    }

    fn connected(&self) -> usize {
        self.stats.connected.load(Ordering::Relaxed)
    }

    /// Number of non-refusal anomalies recorded so far. Snapshotted around
    /// the handoff gap so the assertion can be scoped to that window.
    fn other_len(&self) -> usize {
        self.stats
            .other
            .lock()
            .expect("dial-probe mutex must not be poisoned")
            .len()
    }

    fn other_snapshot(&self) -> Vec<String> {
        self.stats
            .other
            .lock()
            .expect("dial-probe mutex must not be poisoned")
            .clone()
    }

    /// Block until the probe has made at least `target` attempts. Proves the
    /// loop was actually sampling the window we care about rather than being
    /// starved by the runtime — without this, "zero refusals" could mean
    /// "zero dials".
    async fn wait_for_attempts(&self, target: usize, what: &str) {
        let start = std::time::Instant::now();
        while self.attempts() < target {
            assert!(
                start.elapsed() < WAIT_BUDGET,
                "dial probe made only {} of {target} attempts within {WAIT_BUDGET:?} while {what}",
                self.attempts()
            );
            tokio::time::sleep(DIAL_CADENCE).await;
        }
    }

    /// Stop the probe and join it, propagating a probe-task panic.
    ///
    /// Discarding the `JoinHandle`'s result here would be unsound for the
    /// assertions this file makes: a panicking probe task (a poisoned
    /// `other` mutex is the reachable case) stops incrementing every
    /// counter, and both final assertions accept a frozen counter —
    /// `refused == 0` trivially, and `connected > 0` from whatever the
    /// pre-panic dials had already recorded. The whole test would then
    /// report a green "never refused" verdict having stopped looking.
    async fn stop(self) -> Arc<DialStats> {
        self.stop.store(true, Ordering::Relaxed);
        match tokio::time::timeout(WAIT_BUDGET, self.handle).await {
            Ok(Ok(())) => {}
            Ok(Err(join_err)) => panic!(
                "dial probe task must not panic or be cancelled — its counters are the \
                 evidence every assertion in this test rests on: {join_err}"
            ),
            Err(_) => panic!("dial probe task did not stop within {WAIT_BUDGET:?}"),
        }
        self.stats
    }
}

// ───────────────────────────────────────────────────────────────────────────
// gRPC health-client helpers
// ───────────────────────────────────────────────────────────────────────────

/// Connect a gRPC health client, bounded. The TCP connect itself always
/// succeeds while the socket is in `LISTEN` (that is the whole point of this
/// file), but the HTTP/2 handshake only completes once a server is actually
/// accepting — so this doubles as "wait until a server is serving".
async fn connect_health(addr: SocketAddr) -> HealthClient<Channel> {
    let channel = tokio::time::timeout(
        WAIT_BUDGET,
        Channel::from_shared(format!("http://{addr}"))
            .expect("endpoint uri")
            .connect(),
    )
    .await
    .expect("gRPC channel connect must complete within the wait budget")
    .expect("gRPC channel must connect");
    HealthClient::new(channel)
}

async fn check(client: &mut HealthClient<Channel>, service: &str) -> PbServingStatus {
    let resp = tokio::time::timeout(
        WAIT_BUDGET,
        client.check(HealthCheckRequest {
            service: service.to_string(),
        }),
    )
    .await
    .expect("health Check must complete within the wait budget")
    .expect("health Check must succeed")
    .into_inner();
    PbServingStatus::try_from(resp.status)
        .unwrap_or_else(|_| panic!("unknown ServingStatus code: {}", resp.status))
}

// ───────────────────────────────────────────────────────────────────────────
// The handoff regression
// ───────────────────────────────────────────────────────────────────────────

/// End-to-end pin of the b.rgs handoff contract:
///
/// 1. one `bind_grpc_listener` call yields two incoming streams on one port;
/// 2. a bootstrap health server on the first stream answers `""` = SERVING
///    (liveness: the process is up) and the readiness name = NOT_SERVING
///    (no dependency has been wired yet) — the asymmetry
///    `create_health_service` registers;
/// 3. across the entire bootstrap-shutdown → full-server-start window a
///    client dialling the port NEVER sees `ConnectionRefused`;
/// 4. a second server on the second stream answers on the SAME port;
/// 5. the second server's health service, built via
///    `health_service_from_reporter`, reads the same status map — a
///    readiness flip published through the original reporter is visible
///    through the post-handoff service.
///
/// Point 3 is the regression proper. A rewrite that binds twice would pass
/// every other assertion here and fail only this one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_to_full_server_handoff_never_refuses_a_connection() {
    let (bootstrap_incoming, serving_incoming, bound_addr) =
        bind_grpc_listener(loopback_port_zero(), KEEPALIVE)
            .expect("bind_grpc_listener must succeed");
    assert_ne!(
        bound_addr.port(),
        0,
        "bind_grpc_listener must report the kernel-assigned port, not the requested 0"
    );

    // ── Phase 1: bootstrap server on the first descriptor ────────────────
    let (reporter, bootstrap_health) = create_health_service(READINESS_SERVICE_NAME).await;
    let bootstrap_shutdown = CancellationToken::new();
    let bootstrap_server = {
        let shutdown = bootstrap_shutdown.clone();
        tokio::spawn(
            Server::builder()
                .add_service(bootstrap_health)
                .serve_with_incoming_shutdown(bootstrap_incoming, async move {
                    shutdown.cancelled().await;
                }),
        )
    };

    let mut bootstrap_client = connect_health(bound_addr).await;
    assert_eq!(
        check(&mut bootstrap_client, OVERALL_SERVICE).await,
        PbServingStatus::Serving,
        "the bootstrap server must answer liveness (service=\"\") SERVING before any \
         dependency is wired — that is the whole reason the socket is bound early"
    );
    assert_eq!(
        check(&mut bootstrap_client, READINESS_SERVICE_NAME).await,
        PbServingStatus::NotServing,
        "readiness must be NOT_SERVING while the bootstrap server is up: nothing has \
         probed Valkey / the service token / the counter-seed gate yet, so advertising \
         SERVING would route traffic to a replica with no backing store"
    );

    // ── Phase 2: dial probe live before the handoff starts ───────────────
    let probe = DialProbe::start(bound_addr);
    probe
        .wait_for_attempts(3, "the bootstrap server was still serving")
        .await;

    // ── Phase 3: the handoff ─────────────────────────────────────────────
    // Mirrors `main.rs` Step 8: cancel, then drain under a bounded budget.
    bootstrap_shutdown.cancel();
    let drained = tokio::time::timeout(WAIT_BUDGET, bootstrap_server)
        .await
        .expect("bootstrap server must drain within the wait budget")
        .expect("bootstrap server task must not panic");
    drained.expect("bootstrap server must exit cleanly on shutdown");

    // Baseline for the gap window, sampled AFTER the drain returns. The gap
    // — the interval in which no server is accepting — begins here, not at
    // `cancel()`: the drain itself is unbounded-in-principle and can easily
    // consume more than 5 dial cadences, in which case a pre-cancel baseline
    // would be satisfied by dials that all landed while the bootstrap server
    // was still accepting and the gap would be sampled zero times.
    let attempts_before_gap = probe.attempts();
    let connected_before_gap = probe.connected();
    let other_before_gap = probe.other_len();

    // The bootstrap server (and with it the first `TcpIncoming`, and with it
    // the first descriptor) is now dropped and no server is accepting. Keep
    // dialling through that gap: this is the exact window a bind → drop →
    // rebind implementation would refuse connections in.
    probe
        .wait_for_attempts(
            attempts_before_gap + 5,
            "no server was accepting between the bootstrap drain and the full-API start",
        )
        .await;

    // Gap-window verdict, evaluated before the full-API server can mask it.
    //
    // `connected` advancing here is the affirmative half of the claim: with
    // no server calling `accept()`, the three-way handshake can only complete
    // because the socket is still in `LISTEN` and the kernel is queueing the
    // connections in the shared accept backlog. A bind → drop → rebind
    // implementation refuses instead, and this counter would stand still.
    let connected_in_gap = probe.connected() - connected_before_gap;
    assert!(
        connected_in_gap > 0,
        "the dial probe completed no connection during the handoff gap ({} attempts made \
         in the window) — the zero-refusal verdict below would be vacuous for the one \
         window it exists to cover. Anomalies so far: {:?}",
        probe.attempts() - attempts_before_gap,
        probe.other_snapshot()
    );
    // Non-refusal anomalies inside the gap are a failure too, not just
    // context for one: a dropped SYN (backlog overflow) or a 2 s connect
    // stall in this window means the shared backlog did not in fact absorb
    // the handoff, which is the same operator-visible outcome — a probe that
    // does not get a connection — that the refusal assertion guards against.
    let gap_anomalies: Vec<String> = probe.other_snapshot().split_off(other_before_gap);
    assert!(
        gap_anomalies.is_empty(),
        "dials during the handoff gap must all complete cleanly; recorded non-refusal \
         anomalies: {gap_anomalies:?}"
    );

    // ── Phase 4: full-API server on the second descriptor ────────────────
    let serving_health = health_service_from_reporter(&reporter);
    let serving_shutdown = CancellationToken::new();
    let serving_server = {
        let shutdown = serving_shutdown.clone();
        tokio::spawn(
            Server::builder()
                .add_service(serving_health)
                .serve_with_incoming_shutdown(serving_incoming, async move {
                    shutdown.cancelled().await;
                }),
        )
    };

    let mut serving_client = connect_health(bound_addr).await;
    assert_eq!(
        check(&mut serving_client, OVERALL_SERVICE).await,
        PbServingStatus::Serving,
        "the post-handoff server must answer on the SAME port the bootstrap server used"
    );

    // Shared status map: flip readiness through the ORIGINAL reporter and
    // observe it through the service built by `health_service_from_reporter`.
    reporter
        .set_service_status(READINESS_SERVICE_NAME, ServingStatus::Serving)
        .await;
    assert_eq!(
        check(&mut serving_client, READINESS_SERVICE_NAME).await,
        PbServingStatus::Serving,
        "health_service_from_reporter must share the reporter's status map — otherwise \
         the health supervisor would be writing to a map no served service reads"
    );

    // ── Phase 5: the verdict ─────────────────────────────────────────────
    let stats = probe.stop().await;
    let refused = stats.refused.load(Ordering::Relaxed);
    let attempts = stats.attempts.load(Ordering::Relaxed);
    let connected = stats.connected.load(Ordering::Relaxed);
    let other = stats
        .other
        .lock()
        .expect("dial-probe mutex must not be poisoned")
        .clone();

    assert_eq!(
        refused,
        0,
        "b.rgs: a client dialling :{} across the bootstrap → full-API handoff must NEVER \
         see ConnectionRefused. {refused} of {attempts} dials were refused, which means the \
         listening socket left LISTEN — the two TcpIncoming streams are no longer sharing \
         one open file description. Non-refusal anomalies seen: {other:?}",
        bound_addr.port()
    );
    // Whole-run vacuity floor. The load-bearing version of this check is the
    // gap-scoped `connected_in_gap > 0` in Phase 3 — this one only rules out
    // a probe that dialled but never connected at any point in the test.
    assert!(
        connected > 0,
        "the dial probe never completed a single connection ({attempts} attempts) — the \
         zero-refusal result above would be vacuous. Anomalies: {other:?}"
    );

    serving_shutdown.cancel();
    let _ = tokio::time::timeout(WAIT_BUDGET, serving_server).await;
}

/// Both streams must report the same local address. Cheap structural pin of
/// "one socket, two descriptors": a rewrite that opened a second listener
/// with `TcpListener::bind(addr)` on the *resolved* address would still pass
/// the port-equality check, but one that re-bound port 0 would not.
#[tokio::test]
async fn bind_grpc_listener_returns_two_streams_on_one_bound_address() {
    let (bootstrap_incoming, serving_incoming, bound_addr) =
        bind_grpc_listener(loopback_port_zero(), KEEPALIVE)
            .expect("bind_grpc_listener must succeed");

    assert_eq!(
        bootstrap_incoming
            .local_addr()
            .expect("bootstrap stream local_addr"),
        bound_addr,
        "the bootstrap stream must be bound to the address returned to the caller"
    );
    assert_eq!(
        serving_incoming
            .local_addr()
            .expect("serving stream local_addr"),
        bound_addr,
        "the full-API stream must share the bootstrap stream's bound address — they are \
         two descriptors for one socket, not two sockets"
    );
}

/// `serve_with_incoming*` ignores the `Server` builder's `tcp_nodelay` /
/// `tcp_keepalive`, so `bind_grpc_listener` re-applies both per accepted
/// connection. This asserts the options actually land on accepted sockets —
/// and on BOTH streams, since each carries its own copy of the settings.
///
/// Not merely a smoke test: tonic's `set_accepted_socket_options` swallows
/// `setsockopt` failures with a `warn!` and returns no error, so a wiring
/// regression (settings applied to the builder instead of the stream) is
/// silent at every other layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bind_grpc_listener_applies_nodelay_and_keepalive_to_accepted_connections() {
    let (mut bootstrap_incoming, mut serving_incoming, bound_addr) =
        bind_grpc_listener(loopback_port_zero(), KEEPALIVE)
            .expect("bind_grpc_listener must succeed");

    for (label, incoming) in [
        ("bootstrap", &mut bootstrap_incoming),
        ("full-API", &mut serving_incoming),
    ] {
        let dialer = tokio::spawn(async move { TcpStream::connect(bound_addr).await });

        let accepted = tokio::time::timeout(WAIT_BUDGET, incoming.next())
            .await
            .unwrap_or_else(|_| panic!("{label} stream must accept a connection"))
            .unwrap_or_else(|| panic!("{label} stream must not end"))
            .unwrap_or_else(|e| panic!("{label} stream accept must succeed: {e}"));

        assert!(
            accepted
                .nodelay()
                .unwrap_or_else(|e| panic!("{label}: reading TCP_NODELAY must succeed: {e}")),
            "{label}: TCP_NODELAY must be set on accepted connections — tonic's builder \
             setting is ignored by serve_with_incoming*, so bind_grpc_listener owns it"
        );

        #[cfg(unix)]
        {
            assert_ne!(
                sockopt_int(&accepted, libc::SOL_SOCKET, libc::SO_KEEPALIVE),
                0,
                "{label}: SO_KEEPALIVE must be enabled on accepted connections"
            );
            #[cfg(target_os = "linux")]
            assert_eq!(
                sockopt_int(&accepted, libc::IPPROTO_TCP, libc::TCP_KEEPIDLE),
                KEEPALIVE.as_secs() as libc::c_int,
                "{label}: TCP_KEEPIDLE must carry the keepalive time bind_grpc_listener \
                 was called with"
            );
            // macOS spells the idle time `TCP_KEEPALIVE` (also in seconds).
            // Without this arm a macOS run pins only that keepalive is ON,
            // not the VALUE — so a regression passing, say, the default
            // 2 h idle would be invisible on developer machines and caught
            // only in Linux CI.
            #[cfg(target_os = "macos")]
            assert_eq!(
                sockopt_int(&accepted, libc::IPPROTO_TCP, libc::TCP_KEEPALIVE),
                KEEPALIVE.as_secs() as libc::c_int,
                "{label}: TCP_KEEPALIVE must carry the keepalive time bind_grpc_listener \
                 was called with"
            );
        }

        let client_side = tokio::time::timeout(WAIT_BUDGET, dialer)
            .await
            .expect("dialer must finish")
            .expect("dialer task must not panic");
        client_side.expect("client-side connect must succeed");
    }
}

/// A port already in `LISTEN` must make `bind_grpc_listener` return the
/// underlying `EADDRINUSE` rather than succeeding or panicking.
///
/// This is the failure the b.rgs restructure moved to the top of startup:
/// the whole point of binding first is that a port collision fails fast,
/// before Valkey / checkpoint / auth wiring has run, instead of ~900 lines
/// later. An implementation that swallowed the bind error (or retried with
/// `SO_REUSEPORT`) would turn a hard misconfiguration into a replica that
/// answers health checks while a foreign process owns its traffic.
///
/// Deterministic, not racy: the conflicting socket is a real listener held
/// alive across the call, and `SO_REUSEADDR` — which Rust's `TcpListener`
/// sets — only permits rebinding a `TIME_WAIT` address, never one held by a
/// live `LISTEN` socket.
#[tokio::test]
async fn bind_grpc_listener_reports_address_in_use_for_an_occupied_port() {
    let occupied = std::net::TcpListener::bind(loopback_port_zero())
        .expect("test-owned listener must bind on an ephemeral port");
    let addr = occupied
        .local_addr()
        .expect("test-owned listener local_addr");

    let err = bind_grpc_listener(addr, KEEPALIVE)
        .expect_err("bind_grpc_listener must fail on a port already held in LISTEN");

    assert_eq!(
        err.kind(),
        ErrorKind::AddrInUse,
        "bind_grpc_listener must surface the bind failure verbatim so `run()` can fail \
         fast at the top of startup; got {err:?}"
    );

    // Hold the conflicting listener until after the assertion — dropping it
    // earlier would let the bind succeed and make the test self-defeating.
    drop(occupied);
}

/// Read an integer socket option off an accepted connection.
#[cfg(unix)]
fn sockopt_int(stream: &TcpStream, level: libc::c_int, name: libc::c_int) -> libc::c_int {
    use std::os::fd::AsRawFd;

    let mut value: libc::c_int = 0;
    let mut len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
    // SAFETY: `stream` owns a valid open socket descriptor for the duration
    // of the call, and `value` / `len` are a correctly sized and initialised
    // out-parameter pair for an integer-valued socket option.
    let rc = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            level,
            name,
            std::ptr::addr_of_mut!(value).cast::<libc::c_void>(),
            &mut len,
        )
    };
    assert_eq!(
        rc,
        0,
        "getsockopt(level={level}, name={name}) failed: {}",
        std::io::Error::last_os_error()
    );
    value
}
