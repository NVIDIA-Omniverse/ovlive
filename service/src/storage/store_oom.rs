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

//! The store out-of-memory signal (b.pa1): how a store write refused `-OOM`
//! anywhere in this process reaches the GC's out-of-memory collection.
//!
//! Under the required `noeviction` policy a full shard refuses every write
//! that can grow memory until memory is freed. When what filled it is
//! released external values awaiting garbage collection, the GC can free it
//! itself (`storage::external_values`, "The out-of-memory collection"; SDD
//! §11.3). This module is the trigger:
//!
//! - **Where `-OOM` is noted.** Every site that receives a fred error goes
//!   through a [`FredErrOrigin`](crate::storage::client::FredErrOrigin), whose
//!   `observe_error` notes a store write refused `-OOM` on the
//!   [`StoreOomSignal`] it carries, and the counted upload write sites'
//!   `refuse_store_write` notes it the same way. An origin built from a client
//!   uses the client's signal; one built from a metrics handle names its
//!   signal explicitly, so no receipt site can leave it out.
//! - **What a note does.** At most once per
//!   [`STORE_OOM_REQUEST_MIN_INTERVAL`] — the first note of each interval;
//!   the rest do nothing — it sets the `{gc}` request key
//!   `le:ext:{gc}:oom:request` (`KeySchema::ext_oom_request`) to this
//!   replica's label for 10 s, through `gc_oom_signal.lua`, which writes on a
//!   full `{gc}` shard too, and then wakes this replica's GC loop
//!   ([`StoreOomSignal::woken`]). The wake shares the request's rate limit
//!   because the GC's own refused steps note too: a wake per note would keep
//!   the loop from ever sleeping while the shard stays full. The write is
//!   fire-and-forget: a note never waits and never fails its caller. The GC
//!   leader, on whatever replica, takes the request with a `GETDEL` at the
//!   start of its next pass.
//! - **Behind the hold write fence.** Startup hands the signal the on-access
//!   hold write fence ([`StoreOomSignal::set_hold_fence`]; SDD SR-6.2), which
//!   a disaster-recovery hold's entry and a stale hold verdict's close. Once
//!   it is closed, a note sends no request and only wakes the GC loop, and a
//!   request already in flight checks the fence before a `NOSCRIPT` reload's
//!   `SCRIPT LOAD` and re-send, so it gives up within its command timeout.
//!   The request is not one of the writes the hold aborts by task: it is
//!   spawned by whichever write is refused, at any time, so without the
//!   fence each refused write after entry could start one more
//!   (`crate::dr::hold`'s writer audit). Normal operation is unchanged.
//!
//! A signal belongs to one [`ValkeyClient`](crate::storage::client::ValkeyClient)
//! and every clone of it ([`ValkeyClient::store_oom_signal`](crate::storage::client::ValkeyClient::store_oom_signal));
//! each client `connect` builds its own. It writes no request until startup
//! [`arm`](StoreOomSignal::arm)s it with the key and this replica's label; an
//! unarmed signal only wakes the GC loop, under the same rate limit.

use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Duration;

use fred::clients::Pool;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::storage::client::{evalsha_with_fenced_reload, hold_fence_closed, FencedEvalError};
use crate::storage::keys::KeySchema;

/// The least time between two collection requests one replica writes
/// (`le:ext:{gc}:oom:request`): a store refusing writes for lack of memory
/// refuses many at once, and one request per second is enough for a leader
/// that runs one pass a second. It also limits how often a note wakes this
/// replica's GC loop: notes in between do nothing.
pub const STORE_OOM_REQUEST_MIN_INTERVAL: Duration = Duration::from_secs(1);

/// How long a collection request lives when no leader takes it: the
/// `PX` of `le:ext:{gc}:oom:request`. Long enough for a leader one pass away,
/// short enough that a request from a condition that has cleared starts no
/// collection minutes later.
pub const OOM_COLLECTION_REQUEST_TTL: Duration = Duration::from_secs(10);

/// One process-side end of the store out-of-memory signal; see the module
/// docs. Cheap to clone: clones share one wake-up, one rate limit and one
/// target.
#[derive(Clone)]
pub struct StoreOomSignal {
    /// `None` only for [`StoreOomSignal::unwired`], which notes nothing.
    inner: Option<Arc<Inner>>,
}

struct Inner {
    /// The pool the request is written on (the client's data pool).
    pool: Pool,
    /// Wakes this replica's GC loop.
    wake: Notify,
    /// When this replica last sent a request.
    last_request: Mutex<Option<tokio::time::Instant>>,
    /// Where requests go, once startup armed the signal.
    target: OnceLock<RequestTarget>,
    /// The disaster-recovery hold write fence, once startup handed it over
    /// ([`StoreOomSignal::set_hold_fence`]): closed, no request is sent.
    hold_fence: OnceLock<CancellationToken>,
}

/// The request key and the value a request carries.
struct RequestTarget {
    key: String,
    requester: String,
}

static UNWIRED: StoreOomSignal = StoreOomSignal { inner: None };

impl std::fmt::Debug for StoreOomSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoreOomSignal")
            .field("wired", &self.inner.is_some())
            .field(
                "armed",
                &self
                    .inner
                    .as_ref()
                    .is_some_and(|i| i.target.get().is_some()),
            )
            .finish()
    }
}

impl StoreOomSignal {
    /// A signal writing its requests on `pool` once armed. Built by
    /// `ValkeyClient::connect` from the client's data pool.
    pub(crate) fn new(pool: Pool) -> Self {
        Self {
            inner: Some(Arc::new(Inner {
                pool,
                wake: Notify::new(),
                last_request: Mutex::new(None),
                target: OnceLock::new(),
                hold_fence: OnceLock::new(),
            })),
        }
    }

    /// A signal that notes nothing and never wakes anything. For a receipt
    /// site whose commands the store never refuses `-OOM` — reads and
    /// connection-state commands (`XREAD`, `UNWATCH`), which are not
    /// memory-growing writes — and that holds no client to take a signal
    /// from; and for tests. A site that sends a write takes its client's
    /// signal instead.
    pub fn unwired() -> &'static StoreOomSignal {
        &UNWIRED
    }

    /// Arm the signal: from now on a note also writes a collection request to
    /// `keys`' request key, carrying `requester` (this replica's label).
    /// Called once at startup; a second call changes nothing.
    pub fn arm(&self, keys: &KeySchema, requester: impl Into<String>) {
        let Some(inner) = &self.inner else {
            return;
        };
        let armed = inner
            .target
            .set(RequestTarget {
                key: keys.ext_oom_request(),
                requester: requester.into(),
            })
            .is_ok();
        if !armed {
            tracing::debug!("the store out-of-memory signal was already armed; kept its target");
        }
    }

    /// Put the signal behind the on-access hold write fence (SDD SR-6.2):
    /// once `hold_fence` is cancelled — at a disaster-recovery hold's entry or
    /// a stale hold verdict's — a note sends no collection request, and a
    /// request in flight sends no `NOSCRIPT` reload (module docs). Called
    /// once at startup, with the token the hold's `OnAccessRepairs` entry
    /// cancels; a second call changes nothing.
    pub fn set_hold_fence(&self, hold_fence: CancellationToken) {
        let Some(inner) = &self.inner else {
            return;
        };
        if inner.hold_fence.set(hold_fence).is_err() {
            tracing::debug!(
                "the store out-of-memory signal already had a hold write fence; kept it"
            );
        }
    }

    /// Note that the store refused a write `-OOM`. At most once per
    /// [`STORE_OOM_REQUEST_MIN_INTERVAL`] — the note that takes the turn —
    /// send a collection request (when armed) and wake this replica's GC loop;
    /// every other note does nothing. Rate-limiting the wake with the request
    /// matters because the GC's own refused steps note too: an unlimited wake
    /// would leave a wake-up pending after every pass and run the loop without
    /// sleeping for as long as the shard stays full. Never blocks and never
    /// fails: the request is sent from a spawned task (none without a tokio
    /// runtime), its failure only logged at DEBUG, and the GC loop is woken
    /// after it, so a leader on this replica finds the request it wakes for.
    /// Once the hold write fence is closed ([`Self::set_hold_fence`]), the
    /// note that takes the turn only wakes the GC loop.
    pub fn note(&self) {
        let Some(inner) = &self.inner else {
            return;
        };
        if !inner.take_request_turn() {
            return;
        }
        let armed = inner.target.get().is_some();
        let fenced = hold_fence_closed(inner.hold_fence.get());
        if armed && fenced {
            tracing::debug!(
                "a store write was refused -OOM after this replica's hold write fence closed (a \
                 disaster-recovery hold or a stale hold verdict): no out-of-memory collection \
                 request is sent; the GC loop is woken"
            );
        }
        let send = armed && !fenced;
        let runtime = tokio::runtime::Handle::try_current().ok();
        match (send, runtime) {
            (true, Some(runtime)) => {
                let inner = Arc::clone(inner);
                runtime.spawn(async move {
                    inner.send_request().await;
                    inner.wake.notify_one();
                });
            }
            _ => inner.wake.notify_one(),
        }
    }

    /// Resolve at the next note that takes its turn (at most one per
    /// [`STORE_OOM_REQUEST_MIN_INTERVAL`]), or at once when one came since the
    /// last wake-up; never for [`Self::unwired`]. The GC loop races its sleep
    /// against it.
    pub(crate) async fn woken(&self) {
        match &self.inner {
            Some(inner) => inner.wake.notified().await,
            None => std::future::pending().await,
        }
    }
}

impl Inner {
    /// Whether this note may send a request: none sent in the last
    /// [`STORE_OOM_REQUEST_MIN_INTERVAL`]. Records the send when it may.
    fn take_request_turn(&self) -> bool {
        let now = tokio::time::Instant::now();
        let mut last = self
            .last_request
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        match *last {
            Some(prev) if now.saturating_duration_since(prev) < STORE_OOM_REQUEST_MIN_INTERVAL => {
                false
            }
            _ => {
                *last = Some(now);
                true
            }
        }
    }

    /// Write the request: one `EVALSHA` of `gc_oom_signal.lua` on the request
    /// key, under the hold write fence once startup handed it over
    /// ([`evalsha_with_fenced_reload`]): the fence is checked before the
    /// `EVALSHA` and, on `NOSCRIPT`, before the `SCRIPT LOAD` and the re-send,
    /// so a request in flight when the fence closes gives up within its
    /// command timeout and sends no reload. A failure is logged at DEBUG and
    /// not observed on a `FredErrOrigin`: a refused request must not note
    /// itself again.
    async fn send_request(&self) {
        let Some(target) = self.target.get() else {
            return;
        };
        let sent: Result<i64, FencedEvalError> = evalsha_with_fenced_reload(
            &crate::storage::external_values::GC_OOM_SIGNAL_SCRIPT,
            self.pool.next(),
            vec![target.key.clone()],
            vec![
                target.requester.clone(),
                u64::try_from(OOM_COLLECTION_REQUEST_TTL.as_millis())
                    .unwrap_or(u64::MAX)
                    .to_string(),
            ],
            self.hold_fence.get(),
        )
        .await;
        match sent {
            Ok(_) => {}
            Err(FencedEvalError::Fred(error)) => tracing::debug!(
                error = %error,
                "could not request an out-of-memory GC collection; a later refused write requests \
                 it again"
            ),
            Err(FencedEvalError::Held) => tracing::debug!(
                "out-of-memory GC collection request not sent: this replica's hold write fence \
                 closed (a disaster-recovery hold or a stale hold verdict)"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    //! The signal's turn (b.pa1): a note acts only when it takes the
    //! once-per-[`STORE_OOM_REQUEST_MIN_INTERVAL`] turn, then sends the
    //! request when armed and wakes the GC loop after the send. Behind the
    //! hold write fence (kl-retire F3), a note sends nothing once the fence
    //! is closed, and a request answered `NOSCRIPT` after it closed sends no
    //! `SCRIPT LOAD`.
    //!
    //! The turn tests run an unarmed signal under paused tokio time, so a
    //! note lands at an exact instant of the interval. The armed signal's
    //! pool connects to [`HoldingServer`], a stand-in for the store that
    //! answers fred's connection setup and holds the request's `EVALSHA`
    //! until the test releases it, so the test sees the request's arguments
    //! and the state of the wake while the send is still in flight. The
    //! fence tests have it answer the request `-NOSCRIPT` and count every
    //! `SCRIPT` command: a stand-in is the only way to force `NOSCRIPT`,
    //! since the shared test store's script cache is server-wide and is
    //! never flushed. Where the request lands in a real store and how the GC
    //! leader takes it are covered against Valkey in
    //! `tests/integration/storage/oom_collection_test.rs` and
    //! `store_write_refusal_test.rs`.

    use super::*;

    use fred::interfaces::ClientLike;
    use fred::types::config::ServerConfig;
    use fred::types::Builder;
    use futures::FutureExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Bound on each wait for the stand-in store or the wake-up.
    const WAIT: Duration = Duration::from_secs(10);

    /// A one-connection pool that is never connected: an unarmed signal
    /// sends nothing on it.
    fn offline_pool() -> Pool {
        Builder::default_centralized()
            .build_pool(1)
            .expect("an unconnected pool builds")
    }

    /// Whether the GC loop's wake-up is pending now: one poll of
    /// [`StoreOomSignal::woken`], which consumes the wake when there is one.
    fn woke(signal: &StoreOomSignal) -> bool {
        signal.woken().now_or_never().is_some()
    }

    /// The first note takes the turn and wakes the GC loop, though the signal
    /// is not armed (before startup arms it, a note only wakes); notes later
    /// in the same interval do nothing.
    #[tokio::test(start_paused = true)]
    async fn the_first_out_of_memory_note_wakes_the_gc_loop_even_unarmed() {
        let signal = StoreOomSignal::new(offline_pool());
        assert!(!woke(&signal), "no wake-up before any note");

        signal.note();
        assert!(
            woke(&signal),
            "the first note takes the turn and wakes the loop"
        );
        signal.note();
        assert!(
            !woke(&signal),
            "a second note in the same instant has no turn and wakes nothing"
        );
    }

    /// At most one wake-up per [`STORE_OOM_REQUEST_MIN_INTERVAL`], measured
    /// from the note that took the last turn: a note one millisecond short of
    /// the interval does nothing, one at the interval takes the next turn.
    /// Clones share the turn, so a note on a clone counts against it too.
    #[tokio::test(start_paused = true)]
    async fn notes_wake_the_gc_loop_at_most_once_per_interval() {
        let signal = StoreOomSignal::new(offline_pool());
        let clone = signal.clone();

        signal.note();
        assert!(woke(&signal), "the first note takes the turn");
        tokio::time::advance(STORE_OOM_REQUEST_MIN_INTERVAL - Duration::from_millis(1)).await;
        clone.note();
        signal.note();
        assert!(
            !woke(&signal),
            "notes inside the interval — on the signal or a clone — wake nothing"
        );

        tokio::time::advance(Duration::from_millis(1)).await;
        clone.note();
        assert!(
            woke(&signal),
            "a note at the interval takes the next turn, and a clone's wake is the signal's"
        );
        signal.note();
        assert!(!woke(&signal), "and opens a new interval");
    }

    /// The unwired signal notes nothing and never wakes anything.
    #[tokio::test(start_paused = true)]
    async fn the_unwired_signal_never_wakes() {
        let signal = StoreOomSignal::unwired();
        signal.note();
        assert!(!woke(signal));
    }

    /// An armed note sends one `EVALSHA` of `gc_oom_signal.lua` on the request
    /// key, carrying the label the FIRST `arm` gave and the request's 10 s
    /// lifetime, and wakes the GC loop only after the send has returned: while
    /// the stand-in store holds the reply, no wake-up is pending, so a leader
    /// on this replica cannot wake before its request is in the store.
    ///
    /// A second armed note inside `STORE_OOM_REQUEST_MIN_INTERVAL` — made
    /// back to back with the first, so it is inside the interval whatever the
    /// runner's load — sends no request: the stand-in store counts one
    /// `EVALSHA` in all. The count is read after a `PING` on the pool's one
    /// connection, which the store answers only after every command sent
    /// before it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_armed_note_sends_its_request_and_wakes_the_gc_loop_only_after_the_send() {
        let server = HoldingServer::start().await;
        let pool = server.connected_pool().await;
        let signal = StoreOomSignal::new(pool.clone());
        let keys = KeySchema::default();
        signal.arm(&keys, "replica-first");
        signal.arm(&keys, "replica-second");

        signal.note();
        signal.note();
        let request = tokio::time::timeout(WAIT, server.held_evalsha())
            .await
            .expect("an armed note must send its request");
        assert_eq!(
            request,
            [
                "EVALSHA".to_owned(),
                crate::storage::external_values::GC_OOM_SIGNAL_SCRIPT
                    .sha1()
                    .to_string(),
                "1".to_owned(),
                keys.ext_oom_request(),
                "replica-first".to_owned(),
                OOM_COLLECTION_REQUEST_TTL.as_millis().to_string(),
            ],
            "one `{{gc}}` key, the first armed label, and the request's lifetime in ms"
        );
        assert!(
            !woke(&signal),
            "the GC loop must not wake while the request's send is still in flight"
        );

        server.release();
        tokio::time::timeout(WAIT, signal.woken())
            .await
            .expect("the GC loop must wake once the send returned");
        let _: () = tokio::time::timeout(WAIT, pool.next().ping(None))
            .await
            .expect("the barrier PING must return")
            .expect("the stand-in store answers PING");
        assert_eq!(
            server.evalshas(),
            1,
            "a second note inside the interval sends no request"
        );
    }

    /// What [`HoldingServer::start_answering`] answers a request with
    /// `-NOSCRIPT`: a store whose script cache was lost, as after a restart.
    const NOSCRIPT: &[u8] = b"-NOSCRIPT No matching script. Please use EVAL.\r\n";

    /// A note that takes its turn once the hold write fence is closed sends
    /// no request and still wakes the GC loop (kl-retire F3): after the
    /// wake-up, and a barrier `PING` the stand-in answers only after every
    /// command sent before it, the stand-in has read no `EVALSHA`. The
    /// stand-in answers an `EVALSHA` at once (released before the note), so
    /// a request sent anyway is counted rather than left holding the wake.
    ///
    /// Fail-on-old: with `note()` sending whenever the signal is armed
    /// (`let send = armed;`), the stand-in counts one `EVALSHA`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_note_after_the_hold_write_fence_closed_sends_no_request_and_still_wakes_the_gc_loop()
    {
        let server = HoldingServer::start().await;
        server.release();
        let pool = server.connected_pool().await;
        let signal = StoreOomSignal::new(pool.clone());
        signal.arm(&KeySchema::default(), "replica-held");
        let fence = CancellationToken::new();
        signal.set_hold_fence(fence.clone());
        fence.cancel();

        signal.note();
        tokio::time::timeout(WAIT, signal.woken())
            .await
            .expect("a note behind a closed fence must still wake the GC loop");
        let _: () = tokio::time::timeout(WAIT, pool.next().ping(None))
            .await
            .expect("the barrier PING must return")
            .expect("the stand-in store answers PING");
        assert_eq!(
            server.evalshas(),
            0,
            "no out-of-memory collection request is sent once the hold write fence is closed"
        );
    }

    /// A request in flight when the hold write fence closes, answered
    /// `-NOSCRIPT` after it closed, sends no `SCRIPT LOAD` and no second
    /// `EVALSHA` (kl-retire F3), and the GC loop still wakes. The stand-in
    /// holds the request's reply until the test has closed the fence, so the
    /// close falls between the `EVALSHA` and its `NOSCRIPT`.
    ///
    /// Fail-on-old: with `send_request` passing no fence (`None` in place of
    /// `self.hold_fence.get()`), the `NOSCRIPT` reload sends one `SCRIPT
    /// LOAD`, which the stand-in counts and refuses.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_request_answered_noscript_after_the_fence_closed_sends_no_script_load() {
        let server = HoldingServer::start_answering(NOSCRIPT).await;
        let pool = server.connected_pool().await;
        let signal = StoreOomSignal::new(pool.clone());
        signal.arm(&KeySchema::default(), "replica-in-flight");
        let fence = CancellationToken::new();
        signal.set_hold_fence(fence.clone());

        signal.note();
        tokio::time::timeout(WAIT, server.held_evalsha())
            .await
            .expect("an armed note behind an open fence must send its request");
        fence.cancel();
        server.release();

        tokio::time::timeout(WAIT, signal.woken())
            .await
            .expect("the GC loop must wake once the request gave up");
        let _: () = tokio::time::timeout(WAIT, pool.next().ping(None))
            .await
            .expect("the barrier PING must return")
            .expect("the stand-in store answers PING");
        assert_eq!(
            (server.scripts(), server.evalshas()),
            (0, 1),
            "a NOSCRIPT answered after the fence closed sends no SCRIPT LOAD and no second \
             EVALSHA"
        );
    }

    /// A stand-in for the store on one connection: it answers fred's
    /// connection setup (`PING`, `CLIENT ID`, `INFO`), reports the first
    /// `EVALSHA` it receives and holds its reply until released, then answers
    /// it `:1` (or what [`Self::start_answering`] names), and counts every
    /// `EVALSHA` it reads. It counts every `SCRIPT` command too and refuses
    /// it with an error, so a `NOSCRIPT` reload stops there. Every other
    /// command is answered `+OK`.
    struct HoldingServer {
        port: u16,
        evalshas: Arc<std::sync::atomic::AtomicUsize>,
        scripts: Arc<std::sync::atomic::AtomicUsize>,
        held: tokio::sync::Mutex<tokio::sync::oneshot::Receiver<Vec<String>>>,
        release: std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
        task: tokio::task::JoinHandle<()>,
    }

    impl Drop for HoldingServer {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    impl HoldingServer {
        async fn start() -> Self {
            Self::start_answering(b":1\r\n").await
        }

        /// A stand-in that answers every `EVALSHA` with `evalsha_reply`, the
        /// first one once released.
        async fn start_answering(evalsha_reply: &'static [u8]) -> Self {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("the stand-in store must bind");
            let port = listener.local_addr().expect("a bound address").port();
            let (held_tx, held_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = tokio::sync::oneshot::channel();
            let evalshas = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let counted = Arc::clone(&evalshas);
            let scripts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let scripts_counted = Arc::clone(&scripts);
            let task = tokio::spawn(async move {
                let Ok((mut conn, _)) = listener.accept().await else {
                    return;
                };
                let mut held_tx = Some(held_tx);
                let mut release_rx = Some(release_rx);
                let mut buf = Vec::new();
                loop {
                    while let Some((args, used)) = parse_command(&buf) {
                        buf.drain(..used);
                        let reply: Vec<u8> = match args[0].to_ascii_uppercase().as_str() {
                            "PING" => b"+PONG\r\n".to_vec(),
                            "CLIENT" => b":1\r\n".to_vec(),
                            "INFO" => {
                                let info = "redis_version:7.2.0";
                                format!("${}\r\n{info}\r\n", info.len()).into_bytes()
                            }
                            "EVALSHA" => {
                                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                if let Some(tx) = held_tx.take() {
                                    let _ = tx.send(args);
                                }
                                if let Some(rx) = release_rx.take() {
                                    let _ = rx.await;
                                }
                                evalsha_reply.to_vec()
                            }
                            "SCRIPT" => {
                                scripts_counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                b"-ERR the stand-in store loads no script\r\n".to_vec()
                            }
                            _ => b"+OK\r\n".to_vec(),
                        };
                        if conn.write_all(&reply).await.is_err() {
                            return;
                        }
                    }
                    match conn.read_buf(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(_) => {}
                    }
                }
            });
            Self {
                port,
                evalshas,
                scripts,
                held: tokio::sync::Mutex::new(held_rx),
                release: std::sync::Mutex::new(Some(release_tx)),
                task,
            }
        }

        /// A one-client pool connected to this stand-in, its setup done.
        async fn connected_pool(&self) -> Pool {
            let mut builder = Builder::default_centralized();
            builder.with_config(|config| {
                config.server = ServerConfig::new_centralized("127.0.0.1", self.port);
            });
            let pool = builder.build_pool(1).expect("the test pool must build");
            pool.init()
                .await
                .expect("the pool must connect to the stand-in store");
            pool
        }

        /// The arguments of the `EVALSHA` the stand-in is holding.
        async fn held_evalsha(&self) -> Vec<String> {
            let mut held = self.held.lock().await;
            (&mut *held)
                .await
                .expect("the stand-in store must report the EVALSHA it holds")
        }

        /// Let the held `EVALSHA` be answered.
        fn release(&self) {
            if let Some(tx) = self.release.lock().unwrap().take() {
                let _ = tx.send(());
            }
        }

        /// The `EVALSHA`s the stand-in has read so far.
        fn evalshas(&self) -> usize {
            self.evalshas.load(std::sync::atomic::Ordering::SeqCst)
        }

        /// The `SCRIPT` commands (a `SCRIPT LOAD`) the stand-in has read so
        /// far.
        fn scripts(&self) -> usize {
            self.scripts.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    /// One RESP command — an array of bulk strings — from the front of
    /// `buf`: its arguments and the bytes it spans, or `None` while it is
    /// incomplete.
    fn parse_command(buf: &[u8]) -> Option<(Vec<String>, usize)> {
        fn line(buf: &[u8], at: usize) -> Option<(&[u8], usize)> {
            let end = buf.get(at..)?.windows(2).position(|w| w == b"\r\n")?;
            Some((&buf[at..at + end], at + end + 2))
        }
        let (head, mut at) = line(buf, 0)?;
        let count: usize = std::str::from_utf8(head.strip_prefix(b"*")?)
            .ok()?
            .parse()
            .ok()?;
        let mut args = Vec::with_capacity(count);
        for _ in 0..count {
            let (len, next) = line(buf, at)?;
            let len: usize = std::str::from_utf8(len.strip_prefix(b"$")?)
                .ok()?
                .parse()
                .ok()?;
            let arg = buf.get(next..next + len)?;
            if buf.get(next + len..next + len + 2)? != b"\r\n" {
                return None;
            }
            args.push(String::from_utf8_lossy(arg).into_owned());
            at = next + len + 2;
        }
        Some((args, at))
    }
}
