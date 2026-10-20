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

//! One multiplexed reader: owns a dedicated connection and issues one
//! `XREAD BLOCK` at a time over every stream its subscriptions need.
//!
//! **The connection step.** A reader starts with no connection. Before the
//! first XREAD of a binding — the pass where the idle→active probe is due, or
//! on a cluster where the read binds a different hash slot — it makes sure its
//! connection serves the read: with none it connects (`cause="bind"`); on a
//! cluster, with one to a node other than the slot's current owner in the
//! readers' routing view, it retires that one and connects to the owner
//! (`cause="retarget"`); otherwise it keeps it. The probe then runs on that
//! one connection. A connect that fails transiently charges the transient
//! budget and backs off like a failed probe; an `Auth`, `Tls` or `Config` (any
//! non-transient) failure ends the reader's subscriptions with `Unavailable`
//! like a permanent probe error. Either way one WARN names the reader, the
//! node, the error and what happens to the subscriptions. See the parent
//! module for the retarget triggers (`-MOVED`, a failover found before a forced
//! reconnect) and for the idle reader whose node stopped being a primary.
//!
//! When a subscription the in-flight read does not cover arrives (a new key,
//! or a cursor behind the read id), the reader wakes that read: a spawned task
//! appends to the wake key the read names (`scripts/stream_mux_wake.lua`) and
//! reports on the reader's own control channel whether the append landed.
//! XREAD is level-triggered on the wake cursor, so the wake cannot be lost
//! whichever of the two reaches the server first. From the landing, the read
//! has `wake_ack_timeout` to return before the reader treats its connection as
//! wedged; a wake that never lands draws no wedge verdict.
//!
//! A failure that ends every subscription on the reader — a permanent XREAD
//! error no single key explains, `CROSSSLOT`, or a reply that cannot be
//! converted — also force-reconnects the reader's connection and backs off
//! before the next read, so re-subscribes do not loop onto the same
//! connection.
//!
//! When a reply's first entry for a key is past the entry after the id the key
//! was read from, the reader checks that stream's head (`XRANGE key - + COUNT
//! 1`) on its own connection before fanning the key out — between reads, so no
//! XREAD is in flight to interrupt. The check only classifies: when the head
//! is past the read position, each subscription more than one id below the
//! reply's first entry for the key gets a trim marker carrying that entry's
//! id, ahead of its entries; its consumer decides, against its catch-up floor,
//! whether the subscription ends `stream_gap`.
//!
//! **Credential re-AUTH (b.19i).** Under a credential provider (the
//! `accessTokenFile` and `iam` auth modes) fred schedules no periodic re-AUTH
//! on a reader client — its config carries the provider cadence-wrapped
//! (`storage::client::derive_reader_config`) — so no command of fred's ever
//! reaches a reader's connection during a read. The reader re-AUTHs its own
//! connection instead, between reads:
//! - **Schedule.** While it holds a connection it keeps one re-AUTH tick
//!   scheduled `refresh_interval()` (the provider's, drawn afresh each time)
//!   ahead: set when it connects, and when a restarted reader finds a
//!   connection already held; moved on by each tick; cleared when the
//!   connection is retired. None under `static` auth.
//! - **Tick.** When it falls due — during a read, a backoff or idle, or found
//!   due before the next read is planned — the reader calls the provider's
//!   `fetch(None)`, bounded by `probe_timeout` and raced against shutdown, so
//!   the provider's rotation gate decides as it does for fred's pooled
//!   refresh. No password: nothing is sent. A password is held, never
//!   logged, until its `AUTH` is sent.
//! - **Wake.** With a read in flight and no wake outstanding for it, the
//!   reader sends that read a credential wake, so it returns within one wake
//!   round trip; the wake-ack watchdog applies, so a rotation also checks the
//!   connection's liveness. A wake already outstanding serves it too. If the
//!   wake fails, the `AUTH` waits for the read to return (up to one `BLOCK`).
//! - **Send.** With no read in flight — before the next XREAD, or at once
//!   when idle or backing off — the reader sends `AUTH` on its own connection,
//!   bounded by `probe_timeout` and raced against shutdown, and skips it when
//!   the client is not connected (its reconnect handshake authenticates
//!   afresh). It counts `live_edit_stream_mux_reauths_total{result}`. A
//!   failed `AUTH` logs one WARN (reader, node, error) and forces no
//!   reconnect and charges no budget; a failed fetch counts `failed` and logs
//!   at DEBUG. An answered `AUTH` does not count as the connection answering:
//!   a managed frontend answers `AUTH` even for a dead backend session.
//!
//! The reader also checks its own connection's health, only ever between
//! reads, so no check interrupts an XREAD in flight:
//! - **Idle keepalive.** While it holds a connection and has nothing to read
//!   (no subscription — a reservation alone registers nothing — no XREAD in
//!   flight and no backoff pending) it `PING`s its connection every
//!   `keepalive_interval` ([`super::READER_KEEPALIVE_INTERVAL`]), from a random
//!   offset, so the backing store's idle timeout never closes it. A failed
//!   tick force-reconnects (`phase="keepalive"`) — or, on a cluster whose
//!   routing view no longer lists the connection's node as a primary, retires
//!   the connection instead.
//! - **Idle→active probe.** Its first XREAD after it had nothing to read —
//!   and, on a cluster, its first XREAD bound to a different hash slot than
//!   its previous read — is preceded by the connection step above and then
//!   one `PING` on its connection, bounded by `reader_probe_timeout`
//!   ([`super::READER_PROBE_TIMEOUT`]). A probe that does not answer
//!   force-reconnects (`phase="probe"`) and is retried under the
//!   transient-failure budget; a permanent error ends the reader's
//!   subscriptions and heals like any whole-reader failure.
//! - **Liveness wake.** A joining subscription the in-flight read already
//!   covers still wakes it when the reader's connection has not answered
//!   within `liveness_age` or has failed since, so the join gets a verdict
//!   from the wake-ack watchdog. A join made between reads (a backoff, an
//!   inline step) is decided as the next read is issued. The verdict shares
//!   the reader's timeout streak: it force-reconnects (`phase="wake_ack"`) on
//!   the first and every 8th no-answer verdict of a streak, and is otherwise a
//!   WARN.
//!
//! A failed XREAD takes one of three remedies (`is_retryable_reader_error`
//! decides only whether the reader's subscriptions survive it):
//! - **`-UNBLOCKED`** — a `CLIENT UNBLOCK … ERROR` of a live read — is not a
//!   failure: re-plan at once from the cursors, with no budget charge, no
//!   backoff and a DEBUG line, up to `MAX_CONSECUTIVE_UNBLOCK_RETRIES` in a
//!   row. Since the reader sends its own re-AUTH between reads, no command of
//!   this process's interrupts a live read. The one case left for
//!   `Blocking::Interrupt` — the reader's next command after a client-side
//!   XREAD timeout that did not force a reconnect, while the old read is still
//!   blocked on the server — interrupts that old, abandoned read: fred keeps
//!   the timed-out command first in the connection's buffer (a timeout only
//!   marks it `timed_out`), so the `-UNBLOCKED` reply is handed to its dropped
//!   caller and the reader never sees it. That case costs fred's backchannel
//!   connection and nothing else. A `-UNBLOCKED` the reader does see comes from
//!   outside the service. Three, not one, because fred interrupts a blocking
//!   command on two paths (before the write, and after it when the read buffer
//!   is non-empty), so a re-issued XREAD can be interrupted again by the same
//!   queued command. Past that it is transient.
//! - **`-MOVED`** (`is_moved_redirect`, a cluster only): the reader re-resolves
//!   its slot's owner (syncing the readers' routing view once when that
//!   view still names the reader's node — single-flight across the readers,
//!   so readers redirected together share one sync, and one that waited
//!   while another's sync ended — answered, or cut off by its bound — reads
//!   the view as that sync left it without syncing again; behind a cut sync
//!   that view is usually unchanged, so the reader takes the transient remedy
//!   below) and, when the owner changed, retargets its connection before the
//!   next read. The first such retarget since the reader's last successful
//!   read is free: no budget charge and no backoff. Each later one in the
//!   same streak — the owner moved again before any read succeeded — is
//!   charged and backed off like the transient remedy below, and the reader
//!   retargets after the backoff. The same holds for a `-MOVED` head check or
//!   probe. An unchanged owner falls through to the transient remedy below.
//! - **Transient** (`is_transient_fred_error`, which includes `-TRYAGAIN`):
//!   WARN, a budget charge and full-jitter backoff, up to
//!   `MAX_CONSECUTIVE_TRANSIENT_FAILURES`. No forced reconnect: on the
//!   router-miss shape fred is already reconnecting, and an `-ASK` or
//!   `-TRYAGAIN` means the slot is mid-migration, so forcing would pile onto
//!   a rebuild. The store's
//!   `-LOADING` (a node replaying its data) and `-BUSY` (a long-running
//!   script) are transient too, and come over a healthy connection from a
//!   node that is answering: a reconnect would reach the same node and the
//!   same refusal, so the reader waits them out within the same budget.
//! - **`Timeout`**: transient plus the rate-limited forced reconnect, because
//!   fred cannot see a blocked connection that has wedged.
//!
//! On a cluster, every forced reconnect first checks that the reader's node
//! still serves it, so a reader never keeps reconnecting a node that no longer
//! does: on every phase but the idle keepalive it re-resolves the bound slot's
//! owner and, when that changed (a failover), retires the connection and
//! retargets instead; on the keepalive phase it checks that the node is still
//! a primary and, when it is not, retires the connection without retargeting
//! (the reader connects afresh when it next serves a subscription).
//!
//! The XREAD pins `max_attempts` to `CONSUMER_XREAD_MAX_ATTEMPTS` (1). At fred's
//! default, a blocking XREAD still buffered when its connection drops is re-sent
//! onto the new connection and blocks it behind the reader's own re-planned
//! read. A reader's client is single-node, so it cannot follow a cluster
//! redirect either way: the server's `-MOVED` or `-ASK` reaches the reader as
//! a `Cluster`-kind error, handled as above.
//!
//! `storage::router_supervisor` restarts a reader client's router that ends
//! other than by its `QUIT`, from the client's first connect until the reader
//! retires it.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use bytes::Bytes;
use fred::clients::{Client, Pool};
use fred::interfaces::{AuthInterface, ClientLike, KeysInterface, StreamsInterface};
use fred::types::config::{Options, Server};
use fred::types::scripts::Script;
use fred::types::streams::XReadResponse;
use fred::types::{ClientState, Value};
use futures::FutureExt;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use super::core::{
    Entry, Failure, Head, Heads, OverflowBound, ReadPlan, ReadSizing, ReaderCore, StreamId,
};
use super::{ChannelSink, Ctl, MuxShared, StreamMuxKnobs, DEFAULT_ENTRY_ESTIMATE, WAKE_KEY_TTL_MS};
#[cfg(any(test, feature = "test-fault-injection"))]
use super::{MuxFault, MuxFaultSite};
use crate::observability::metrics::{
    CONSUMER_PHASE_KEEPALIVE, CONSUMER_PHASE_PERMANENT, CONSUMER_PHASE_PROBE,
    CONSUMER_PHASE_WAKE_ACK, CONSUMER_PHASE_XREAD, HEAD_CHECK_RESULT_ERROR, HEAD_CHECK_RESULT_GAP,
    HEAD_CHECK_RESULT_NO_GAP, KEEPALIVE_RESULT_FAILED, KEEPALIVE_RESULT_OK,
    READER_CONNECT_CAUSE_BIND, READER_CONNECT_CAUSE_RETARGET, REAUTH_RESULT_FAILED,
    REAUTH_RESULT_OK, STREAM_EXIT_ERROR, STREAM_EXIT_SHUTDOWN, WAKE_OUTCOME_ERROR, WAKE_OUTCOME_OK,
};
use crate::storage::client::{
    is_client_unblocked_error, is_moved_redirect, is_transient_fred_error, FredErrOrigin,
    PoolUseToken, ReaderConnectError, ReaderConnector, POOL_LABEL_SUBSCRIBER,
};
use crate::storage::keys::KeySchema;
use crate::storage::notifications::{
    LayerNotification, NOTIFICATION_PAYLOAD_FIELD, RELAY_OMPF_FIELD,
};

/// Consecutive XREAD failures — transient ones, plus the whole-reader
/// permanent failures `heal_and_back_off` adds to the same count, plus one per
/// reply in which a stream head check failed, plus one per idle→active
/// connection probe that timed out or failed transiently, plus one per failed
/// connect of the reader's connection, plus one per `-MOVED` retarget after
/// the first since the last successful read — after which a
/// transient failure makes the reader end every subscription it holds with
/// `Unavailable`. Only a successful XREAD whose head checks (if any) all
/// answered, or the budget running out and ending the reader's subscriptions,
/// resets the count; an answered probe and the idle keepalive leave it as it
/// is.
pub const MAX_CONSECUTIVE_TRANSIENT_FAILURES: u32 = 60;
pub const XREAD_TRANSIENT_BACKOFF_INITIAL: Duration = Duration::from_millis(100);
pub const XREAD_TRANSIENT_BACKOFF_CAP: Duration = Duration::from_secs(5);
/// fred must never replay a blocking XREAD on its own after a reconnect: the
/// reader re-plans from its cursors instead.
pub const CONSUMER_XREAD_MAX_ATTEMPTS: u32 = 1;
pub const FORCE_RECONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// A timeout streak forces a reconnect on its first timeout and then every
/// this-many timeouts.
pub const FORCE_RECONNECT_REARM_EVERY: u32 = 8;
/// `-UNBLOCKED` replies re-issued immediately before they count as failures.
pub const MAX_CONSECUTIVE_UNBLOCK_RETRIES: u32 = 3;
const WAKE_ATTEMPTS: u32 = 3;
const WAKE_RETRY_DELAY: Duration = Duration::from_millis(50);
/// One overall bound on a reader's concurrent probes, however many keys or
/// nodes it covers: the `TYPE` probes of a poison-key isolation, the head
/// checks of one reply, and the `PING` of one idle keepalive tick — and on
/// each step of a credential re-AUTH (its `fetch(None)`, and its `AUTH`). The
/// production value of `StreamMuxKnobs::probe_timeout`. (The idle→active
/// connection probe is bounded by `READER_PROBE_TIMEOUT` instead.)
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const SHUTDOWN_WAKE_TIMEOUT: Duration = Duration::from_millis(500);
const PANIC_RESTART_DELAY: Duration = Duration::from_millis(100);
const STANDALONE_WAKE_TAG: &str = "mux";

/// `scripts/stream_mux_wake.lua`: the wake `XADD` and the wake key's TTL in
/// one atomic step, so the key never exists without a TTL. `Script::from_lua`
/// only hashes the source; the first `evalsha_with_reload` sends `SCRIPT LOAD`
/// on `NOSCRIPT` and retries.
static STREAM_MUX_WAKE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("../scripts/stream_mux_wake.lua")));

pub(super) struct ReaderCtx {
    pub idx: usize,
    /// Builds and retires this reader's connections; the connection itself
    /// lives in the reader's slot on `shared`.
    pub connector: Arc<ReaderConnector>,
    pub data: Pool,
    pub cluster: bool,
    pub instance: Arc<str>,
    pub shared: Arc<MuxShared>,
    pub cancel: CancellationToken,
    /// This reader's own control channel, which its wake tasks report on.
    pub ctl: mpsc::UnboundedSender<Ctl>,
}

impl ReaderCtx {
    /// The reader's current client. An `IO`-kind (transient) error when it
    /// holds none — the connection step precedes every read, so only a
    /// connection retired in between leaves a command without one.
    fn client(&self) -> Result<Client, fred::error::Error> {
        self.shared
            .conn(self.idx)
            .map(|(client, _)| client)
            .ok_or_else(|| {
                fred::error::Error::new(
                    fred::error::ErrorKind::IO,
                    "the stream reader holds no connection",
                )
            })
    }

    /// The node the reader's connection targets: `None` without a
    /// connection, or off a cluster.
    fn node(&self) -> Option<Server> {
        self.shared.conn(self.idx).and_then(|(_, node)| node)
    }
}

type Core = ReaderCore<LayerNotification, ChannelSink>;
type XreadFuture = Pin<Box<dyn Future<Output = Result<Value, fred::error::Error>> + Send>>;
type Backoff = Pin<Box<tokio::time::Sleep>>;

pub fn should_force_reconnect(streak: u32) -> bool {
    streak >= 1 && (streak - 1).is_multiple_of(FORCE_RECONNECT_REARM_EVERY)
}

/// Full-jitter exponential backoff: `(window_ms, sleep_ms)`.
pub(crate) fn backoff_window_ms(failures: u32, initial: Duration, cap: Duration) -> (u64, u64) {
    let shift = failures.saturating_sub(1).min(31);
    let window = initial
        .checked_mul(1u32.checked_shl(shift).unwrap_or(u32::MAX))
        .unwrap_or(cap)
        .min(cap);
    let window_ms = window.as_millis() as u64;
    let sleep_ms = {
        use rand::Rng;
        rand::rng().random_range(0..=window_ms)
    };
    (window_ms, sleep_ms)
}

pub(crate) fn is_crossslot_error(e: &fred::error::Error) -> bool {
    matches!(e.kind(), fred::error::ErrorKind::Unknown) && e.details().starts_with("CROSSSLOT")
}

/// Errors a reader retries with backoff while keeping its subscriptions: the
/// shared classifier's transient set (a `-TRYAGAIN` included), plus the
/// reader-only `-UNBLOCKED`.
pub(crate) fn is_retryable_reader_error(e: &fred::error::Error) -> bool {
    is_transient_fred_error(e) || is_client_unblocked_error(e)
}

/// Consecutive-failure state. Only a successful XREAD whose head checks all
/// answered, or the budget running out and ending the reader's subscriptions,
/// resets it: a permanent whole-reader failure, a reply with a failed head
/// check, a failed idle→active probe, a failed connect and every `-MOVED`
/// retarget after the first of a streak add to `failures` too, so repeated
/// failures of any of these kinds back off exponentially up to the cap.
#[derive(Debug, Default)]
pub(crate) struct TransientBudget {
    pub failures: u32,
    pub timeout_streak: u32,
    pub unblock_streak: u32,
    /// `-MOVED` retargets (an XREAD, a head check or a probe redirected while
    /// the bound slot's owner had moved) since the last successful read. The
    /// first is free; each later one is charged to `failures` and backs off
    /// ([`charge_moved_retarget`]), so a routing view that keeps flipping
    /// during a failover cannot loop a reader through connect, probe, XREAD
    /// and `-MOVED` with no backoff.
    pub moved_streak: u32,
}

impl TransientBudget {
    pub fn record_success(&mut self) {
        *self = Self::default();
    }
}

/// Reader-owned gauge state; survives a panic so the gauges return to zero.
///
/// The reader's share of `live_edit_stream_mux_streams` and of
/// `live_edit_valkey_pool_in_use{pool="subscriber_pool"}` is published at the
/// top of every pass of `reader_loop` and again after the idle branch's
/// drain, so the gauges match the reader's core whenever it waits in a
/// `select!`. That drain is the one change to the core between the top-of-loop
/// publish and a wait: the in-flight and backoff waits follow the top-of-loop
/// publish with the core unchanged, and every other change to the core (a
/// `Ctl` taken by a wait, a read's result, a connection or probe outcome) is
/// followed by a return to the top of the loop.
#[derive(Default)]
struct Gauges {
    streams: i64,
    in_use: Option<PoolUseToken>,
}

impl Gauges {
    fn sync(&mut self, ctx: &ReaderCtx, streams: usize) {
        let streams = streams as i64;
        if streams != self.streams {
            if let Some(m) = ctx.shared.metrics() {
                m.add_stream_mux_streams(streams - self.streams);
            }
            self.streams = streams;
        }
        if streams > 0 && self.in_use.is_none() {
            self.in_use = Some(PoolUseToken::new(
                ctx.shared.metrics(),
                POOL_LABEL_SUBSCRIBER,
            ));
        } else if streams == 0 {
            self.in_use = None;
        }
    }
}

/// Runs one reader until shutdown. Returns whether it woke a blocked read on
/// the way out.
pub(super) async fn run(ctx: ReaderCtx, mut rx: mpsc::UnboundedReceiver<Ctl>) -> bool {
    let mut gauges = Gauges::default();
    loop {
        let outcome = AssertUnwindSafe(reader_loop(&ctx, &mut rx, &mut gauges))
            .catch_unwind()
            .await;
        gauges.sync(&ctx, 0);
        if let Ok(woke) = outcome {
            return woke;
        }
        if ctx.cancel.is_cancelled() {
            return false;
        }
        tracing::error!(
            reader = ctx.idx,
            "stream multiplexer reader panicked; its subscriptions end with UNAVAILABLE and the \
             reader restarts"
        );
        tokio::time::sleep(PANIC_RESTART_DELAY).await;
    }
}

/// The wake sent for one read, while its result is outstanding or after it
/// landed. `landed` is set once the wake task reports its append landed; the
/// wake-ack deadline runs from there. A wake that failed is dropped
/// ([`Ctl::WakeFailed`]): the read draws no wedge verdict, and the next
/// subscription that joins it sends a fresh wake.
#[derive(Debug, Clone, Copy)]
struct PendingWake {
    read: u64,
    landed: Option<Instant>,
}

struct LoopState {
    schema: KeySchema,
    slot_tag: Option<(u16, String)>,
    budget: TransientBudget,
    /// Number of the read most recently issued — the reader's running XREAD
    /// count (`ReaderStats::xreads`). While a read is in flight, it is that
    /// read's number.
    read: u64,
    /// The wake sent for the read in flight, if any. Cleared when that read
    /// returns, so a wake result that arrives later names a read that is no
    /// longer `read` and is ignored; cleared by a failure of its own wake.
    wake: Option<PendingWake>,
    /// Whether a subscription was added since the reader last decided whether
    /// to wake the read in flight. Only a join sends a wake, so a failed wake
    /// is retried by the next join, not in a loop. The decision is taken only
    /// with a read in flight ([`Self::take_wake_reason`]): a join made between
    /// reads — during a backoff or an inline step — stays set and is decided
    /// as the next read is issued.
    joined: bool,
    /// When the reader's own connection last answered, or `None` once
    /// anything on it has failed since (and at start). Stamped by every XREAD
    /// reply (empty or converted), a `-UNBLOCKED` reply, a reply's head checks
    /// when all answered, and an answered idle→active probe; cleared by any
    /// other XREAD error, an unconvertible reply, a failed head check, a failed
    /// probe, a wake-ack timeout, a failed connect, a connection retired
    /// because its node no longer serves the reader, and on a cluster a rebind
    /// to another hash slot (the answer came from the old slot's node). A
    /// join that needs no
    /// coverage wake still wakes the read when this is `None` or older than
    /// `liveness_age` (the liveness wake).
    last_answer: Option<Instant>,
    /// Whether the next XREAD must be preceded by the connection step and the
    /// idle→active probe: set whenever the reader finds nothing to read with no
    /// read in flight — idle, or backing off after a failure ended every
    /// subscription — at start, on a cluster when the next read binds to a
    /// different hash slot than the previous one ([`Self::note_slot_rebind`]),
    /// after a failed connect, and when a `-MOVED` or a forced reconnect
    /// found the bound slot's owner moved; cleared when a probe answers.
    probe_due: bool,
    /// Whether the reader's next connect replaces a connection it retired
    /// because its slot's owner moved, so it counts as `cause="retarget"`
    /// even though no connection is held when it runs. Cleared by a
    /// successful connect, and when a failed one ends the subscriptions.
    retarget_due: bool,
    /// When the idle keepalive next ticks.
    keepalive: KeepaliveSchedule,
    /// The held connection's credential re-AUTH schedule and pending
    /// credential (b.19i).
    reauth: ReauthSchedule,
}

/// A held connection's credential re-AUTH state (b.19i; see the module docs).
///
/// Deliberately not `Debug`: `pending` holds a live credential.
#[derive(Default)]
struct ReauthSchedule {
    /// When the next tick is due; `None` with no connection held, or with no
    /// provider that schedules ticks.
    due: Option<Instant>,
    /// A credential a tick fetched, until its `AUTH` is sent (between reads)
    /// or skipped.
    pending: Option<PendingCredential>,
}

/// The credential a re-AUTH tick fetched. Never logged and not `Debug`.
struct PendingCredential {
    username: Option<String>,
    password: String,
}

impl ReauthSchedule {
    /// Schedules the next tick for the reader's held connection —
    /// `refresh_interval()` from now, drawn afresh — or none, without a
    /// provider (`static` auth) or when it schedules none.
    fn schedule(&mut self, ctx: &ReaderCtx) {
        self.due = ctx
            .connector
            .reauth_provider()
            .and_then(|provider| provider.refresh_interval())
            .map(|interval| Instant::now() + interval);
        self.mirror(ctx);
    }

    /// The connection is retired: no tick, and nothing left to send.
    fn clear(&mut self, ctx: &ReaderCtx) {
        self.due = None;
        self.pending = None;
        self.mirror(ctx);
    }

    fn is_due(&self) -> bool {
        self.due.is_some_and(|due| due <= Instant::now())
    }

    fn set_pending(&mut self, ctx: &ReaderCtx, credential: PendingCredential) {
        self.pending = Some(credential);
        self.mirror(ctx);
    }

    fn take_pending(&mut self, ctx: &ReaderCtx) -> Option<PendingCredential> {
        let pending = self.pending.take();
        self.mirror(ctx);
        pending
    }

    /// Mirrors the schedule onto the reader's stats
    /// (`StreamMux::reauth_for_test`).
    fn mirror(&self, ctx: &ReaderCtx) {
        if let Some(stats) = ctx.shared.stats.get(ctx.idx) {
            stats
                .reauth_scheduled
                .store(self.due.is_some(), Ordering::Release);
            stats
                .reauth_pending
                .store(self.pending.is_some(), Ordering::Release);
        }
    }
}

/// The idle keepalive's schedule: ticks on a fixed phase — a random offset
/// within the first interval, then every interval — skipping the ticks that
/// fall while the reader has something to read. A reader that goes idle
/// therefore pings within one interval of its last XREAD returning.
#[derive(Debug, Default)]
struct KeepaliveSchedule {
    /// The interval the schedule was built for, and its next tick.
    tick: Option<(Duration, Instant)>,
}

impl KeepaliveSchedule {
    /// The next tick for `interval` (`None`: the keepalive is off) at `now`,
    /// never at or before `now`. A new or changed interval starts a new phase
    /// at a random offset in `(0, interval]`; otherwise ticks already past
    /// are skipped in whole intervals, keeping the phase.
    fn next(&mut self, interval: Option<Duration>, now: Instant) -> Option<Instant> {
        let Some(interval) = interval else {
            self.tick = None;
            return None;
        };
        let interval = interval.max(Duration::from_millis(1));
        let next = match self.tick {
            Some((scheduled, next)) if scheduled == interval => skip_past(next, interval, now),
            _ => {
                let nanos = u64::try_from(interval.as_nanos()).unwrap_or(u64::MAX);
                let offset = {
                    use rand::Rng;
                    rand::rng().random_range(1..=nanos)
                };
                now + Duration::from_nanos(offset)
            }
        };
        self.tick = Some((interval, next));
        Some(next)
    }
}

/// `next` advanced by the smallest whole number of `interval`s that puts it
/// after `now` (unchanged when it already is).
fn skip_past(next: Instant, interval: Duration, now: Instant) -> Instant {
    if next > now {
        return next;
    }
    let behind = now.duration_since(next).as_nanos();
    let skipped = behind / interval.as_nanos() + 1;
    let advance = u64::try_from(skipped.saturating_mul(interval.as_nanos())).unwrap_or(u64::MAX);
    next.checked_add(Duration::from_nanos(advance))
        .unwrap_or(now + interval)
}

/// `reason` of a wake sent for a joining subscription the in-flight read does
/// not cover.
const WAKE_REASON_COVERAGE: &str = "coverage";
/// `reason` of a wake sent for a joining subscription the in-flight read
/// covers, because the reader's connection has not answered within the
/// liveness age or has failed since: the wake-ack watchdog checks it.
const WAKE_REASON_LIVENESS: &str = "liveness";
/// `reason` of a wake sent so the reader can send a credential re-AUTH that
/// fell due during the in-flight read, between reads (b.19i). The wake-ack
/// watchdog checks it too.
const WAKE_REASON_CREDENTIAL: &str = "credential";

impl LoopState {
    /// The reader's connection just answered.
    fn answered(&mut self) {
        self.last_answer = Some(Instant::now());
    }

    /// Something on the reader's connection just failed: no earlier answer
    /// vouches for it any more.
    fn failed(&mut self) {
        self.last_answer = None;
    }

    /// Whether a join wakes the read in flight, and why.
    ///
    /// With no read in flight, nothing is decided and `joined` is kept, so a
    /// join made between reads is decided as the next read is issued. With a
    /// read in flight, `joined` is taken, and a join wakes the read:
    /// - for coverage ([`WAKE_REASON_COVERAGE`]) when the read does not cover
    ///   it (a key it does not name, or a cursor behind its read id);
    /// - otherwise for liveness ([`WAKE_REASON_LIVENESS`]) when the connection
    ///   has not answered within `liveness_age`, or has failed since.
    ///
    /// No wake while one is outstanding for the read — sent, or landed with its
    /// deadline running: that wake serves this join too.
    fn take_wake_reason(
        &mut self,
        core: &Core,
        read_in_flight: bool,
        liveness_age: Duration,
    ) -> Option<&'static str> {
        if !read_in_flight || !std::mem::take(&mut self.joined) || self.wake.is_some() {
            return None;
        }
        if core.needs_wake() {
            return Some(WAKE_REASON_COVERAGE);
        }
        match self.last_answer {
            Some(at) if at.elapsed() <= liveness_age => None,
            _ => Some(WAKE_REASON_LIVENESS),
        }
    }

    fn apply(&mut self, core: &mut Core, ctl: Ctl) {
        match ctl {
            Ctl::Add {
                sub,
                key,
                cursor,
                sink,
                key_schema,
            } => {
                if key_schema.prefix() != self.schema.prefix() {
                    self.schema = key_schema;
                }
                core.add(sub, key, cursor, sink);
                self.joined = true;
            }
            Ctl::Remove { sub } => {
                core.remove(sub);
            }
            Ctl::WakeLanded { read, at } => {
                if let Some(wake) = self.wake.as_mut().filter(|w| w.read == read) {
                    wake.landed = Some(at);
                }
            }
            Ctl::WakeFailed { read } => {
                if self.wake.is_some_and(|w| w.read == read) {
                    self.wake = None;
                }
            }
        }
    }

    /// Applies every queued control message. `false`: the channel is closed.
    fn drain(&mut self, core: &mut Core, rx: &mut mpsc::UnboundedReceiver<Ctl>) -> bool {
        loop {
            match rx.try_recv() {
                Ok(ctl) => self.apply(core, ctl),
                Err(mpsc::error::TryRecvError::Empty) => return true,
                Err(mpsc::error::TryRecvError::Disconnected) => return false,
            }
        }
    }

    /// On a cluster, records the hash slot the next read binds to (its first
    /// key's, which every key of a read shares) and reports whether it
    /// differs from the slot the reader's previous read was bound to. A
    /// rebind may move the reader to another node — typically after its old
    /// slot's last subscription left while a read was still in flight — and
    /// nothing has vouched for its connection there: the join's liveness
    /// decision went to the old read as its coverage wake, and the last
    /// answer came from the old slot's node. So a rebind clears the last
    /// answer and makes the probe due, and the first XREAD on the new slot is
    /// preceded by the connection step — which retargets the connection when
    /// another node owns the new slot — and a `PING` on the connection.
    /// Standalone (one node) and a reader's first binding (the probe is due
    /// from its start) report `false`.
    fn note_slot_rebind(&mut self, ctx: &ReaderCtx, core: &Core) -> bool {
        if !ctx.cluster {
            return false;
        }
        let first = core.first_key().expect("planned reads name a key");
        let slot = fred::util::redis_keyslot(first.as_bytes());
        let previous = self.slot_tag.as_ref().map(|(s, _)| *s);
        if previous == Some(slot) {
            return false;
        }
        self.slot_tag = Some((slot, super::router::tag_for_slot(slot)));
        if previous.is_none() {
            return false;
        }
        self.failed();
        self.probe_due = true;
        true
    }

    fn wake_key(&mut self, ctx: &ReaderCtx, core: &Core) -> String {
        let tag = if ctx.cluster {
            let first = core.first_key().expect("planned reads name a key");
            let slot = fred::util::redis_keyslot(first.as_bytes());
            match &self.slot_tag {
                Some((s, tag)) if *s == slot => tag.clone(),
                _ => {
                    let tag = super::router::tag_for_slot(slot);
                    self.slot_tag = Some((slot, tag.clone()));
                    tag
                }
            }
        } else {
            STANDALONE_WAKE_TAG.to_owned()
        };
        self.schema.stream_mux_wake(&tag, &ctx.instance, ctx.idx)
    }
}

enum Event {
    Cancelled,
    Read(Result<Value, fred::error::Error>),
    WakeAckTimeout,
    Ctl(Option<Ctl>),
    BackoffDone,
    /// The idle keepalive's tick is due.
    KeepaliveDue,
    /// The connection's credential re-AUTH tick is due.
    ReauthDue,
    /// Test seams only: the knobs changed while the reader waited with
    /// nothing to read; it re-reads them.
    KnobsChanged,
}

async fn sleep_until_opt(deadline: Option<Instant>) {
    match deadline {
        Some(d) => tokio::time::sleep_until(d).await,
        None => std::future::pending().await,
    }
}

async fn reader_loop(
    ctx: &ReaderCtx,
    rx: &mut mpsc::UnboundedReceiver<Ctl>,
    gauges: &mut Gauges,
) -> bool {
    let mut core = Core::new();
    let mut st = LoopState {
        schema: KeySchema::default(),
        slot_tag: None,
        budget: TransientBudget::default(),
        read: 0,
        wake: None,
        joined: false,
        // Nothing on a new (or restarted) reader's connection has answered it
        // yet.
        last_answer: None,
        // A new (or restarted) reader has had nothing to read until now.
        probe_due: true,
        retarget_due: false,
        keepalive: KeepaliveSchedule::default(),
        reauth: ReauthSchedule::default(),
    };
    // A restarted reader may already hold a connection (it outlives a panic):
    // that connection's re-AUTH ticks resume from now.
    if ctx.shared.is_connected(ctx.idx) {
        st.reauth.schedule(ctx);
    } else {
        st.reauth.clear(ctx);
    }
    let mut inflight: Option<XreadFuture> = None;
    let mut backoff: Option<Backoff> = None;

    loop {
        // A cancel observed inside a step (a reconnect, a probe round, a
        // keepalive tick, a re-AUTH) lands here before any new read is issued.
        if ctx.cancel.is_cancelled() {
            return shutdown(ctx, &mut core, &mut st, rx, inflight.is_some()).await;
        }
        // With no read in flight, a re-AUTH tick that fell due (during a
        // backoff, idle, or an inline step) is taken now, and a fetched
        // credential's `AUTH` goes out before anything else is sent — before
        // the next XREAD, the connection step or the probe.
        if inflight.is_none() {
            let knobs = ctx.shared.knobs();
            if st.reauth.is_due() {
                reauth_tick(ctx, &core, &mut st, &knobs, false).await;
                continue;
            }
            if st.reauth.pending.is_some() {
                send_reauth(ctx, &mut st, &knobs).await;
                continue;
            }
        }
        gauges.sync(ctx, core.stream_count());
        if inflight.is_none() && core.is_empty() {
            // Nothing to read and no read in flight — idle, or backing off
            // after a failure ended every subscription: the next XREAD is
            // probed first.
            st.probe_due = true;
        }
        // Read before the knobs, so a change landing between the two wakes
        // the idle wait below instead of being slept through.
        let knobs_seen = ctx.shared.knobs_generation();
        let knobs = ctx.shared.knobs();
        let wake_deadline = st
            .wake
            .and_then(|w| w.landed)
            .map(|at| at + knobs.wake_ack_timeout);
        let reauth_due = st.reauth.due;

        let event = if let Some(fut) = inflight.as_mut() {
            tokio::select! {
                biased;
                () = ctx.cancel.cancelled() => Event::Cancelled,
                res = fut => Event::Read(res),
                () = sleep_until_opt(wake_deadline) => Event::WakeAckTimeout,
                () = sleep_until_opt(reauth_due) => Event::ReauthDue,
                ctl = rx.recv() => Event::Ctl(ctl),
            }
        } else if let Some(sleep) = backoff.as_mut() {
            tokio::select! {
                biased;
                () = ctx.cancel.cancelled() => Event::Cancelled,
                () = sleep => Event::BackoffDone,
                () = sleep_until_opt(reauth_due) => Event::ReauthDue,
                ctl = rx.recv() => Event::Ctl(ctl),
            }
        } else {
            if !st.drain(&mut core, rx) {
                return false;
            }
            // A `Remove` or `Add` queued during an inline step (the
            // connection step, the probe, a keepalive or a re-auth), or
            // since the publish at the top of this pass, is applied by this
            // drain; without this sync the reader would wait below, or run
            // the connection step, with its gauges behind its core.
            gauges.sync(ctx, core.stream_count());
            if core.is_empty() {
                // Nothing to read and no read in flight: the connection, if
                // the reader holds one, is idle. Keep it alive, and probe it
                // before the next read. A reader with no connection has
                // nothing to keep alive.
                st.probe_due = true;
                let tick = if ctx.shared.is_connected(ctx.idx) {
                    st.keepalive.next(knobs.keepalive_interval, Instant::now())
                } else {
                    None
                };
                tokio::select! {
                    biased;
                    () = ctx.cancel.cancelled() => Event::Cancelled,
                    ctl = rx.recv() => Event::Ctl(ctl),
                    () = sleep_until_opt(tick) => Event::KeepaliveDue,
                    () = sleep_until_opt(reauth_due) => Event::ReauthDue,
                    () = ctx.shared.knobs_changed_since(knobs_seen) => Event::KnobsChanged,
                }
            } else if st.note_slot_rebind(ctx, &core)
                || st.probe_due
                || !ctx.shared.is_connected(ctx.idx)
            {
                // The first XREAD after the reader had nothing to read, or,
                // on a cluster, its first XREAD bound to a different hash
                // slot than its previous read (or one whose connection was
                // retired since). `note_slot_rebind` must stay the left
                // operand of `||`: it records the read's slot even when a
                // probe is already due, so an idle reader that comes back on
                // another slot probes once. Swapped, the slot would be
                // recorded only after that probe, and the next pass would
                // probe a second time. Any `Ctl` queued meanwhile is applied
                // at the next step, before the read is planned.
                match ensure_connection(ctx, &mut core, &mut st, &knobs).await {
                    ConnStep::Ready => {}
                    ConnStep::Failed(after) => {
                        backoff = after;
                        continue;
                    }
                    // The loop's top sees the cancel and shuts down.
                    ConnStep::Cancelled => continue,
                }
                let outcome = probe_connection(ctx, &knobs).await;
                backoff = handle_probe(ctx, &mut core, &mut st, &knobs, outcome).await;
                continue;
            } else {
                let wake_key = st.wake_key(ctx, &core);
                core.set_read_sizing(ReadSizing {
                    target_bytes: knobs.reply_target_bytes,
                    max_count: knobs.count,
                    default_entry_estimate: DEFAULT_ENTRY_ESTIMATE,
                });
                let plan = core.plan_read(&wake_key).expect("non-empty core plans");
                let stats = &ctx.shared.stats[ctx.idx];
                stats.keys.store(plan.keys.len() - 1, Ordering::Release);
                stats.last_count.store(plan.count, Ordering::Release);
                st.read = stats.xreads.fetch_add(1, Ordering::AcqRel) + 1;
                st.wake = None;
                if let Some(m) = ctx.shared.metrics() {
                    m.inc_stream_mux_xread();
                }
                inflight = Some(start_xread(ctx, plan, &knobs));
                // The new read covers every subscription added so far, so a
                // join made since the last decision can only need a liveness
                // wake: one made between reads after a failure, with no answer
                // since. A wake that lands before the XREAD is sent is not
                // lost: the read is level-triggered on the wake key's cursor.
                if let Some(reason) = st.take_wake_reason(&core, true, knobs.liveness_age) {
                    send_wake(ctx, &core, &mut st, reason);
                }
                continue;
            }
        };

        match event {
            Event::Cancelled => {
                return shutdown(ctx, &mut core, &mut st, rx, inflight.is_some()).await
            }
            Event::Read(res) => {
                inflight = None;
                st.wake = None;
                backoff = handle_result(ctx, &mut core, &mut st, &knobs, res).await;
            }
            Event::WakeAckTimeout => {
                st.wake = None;
                st.failed();
                st.budget.timeout_streak = st.budget.timeout_streak.saturating_add(1);
                tracing::warn!(
                    reader = ctx.idx,
                    timeout_ms = knobs.wake_ack_timeout.as_millis() as u64,
                    "multiplexed XREAD did not return within the wake-ack timeout of its wake \
                     landing; treating the reader connection as wedged"
                );
                if should_force_reconnect(st.budget.timeout_streak) {
                    force_reconnect(ctx, &mut st, CONSUMER_PHASE_WAKE_ACK).await;
                }
            }
            Event::BackoffDone => backoff = None,
            Event::KeepaliveDue => keepalive(ctx, &mut st, &knobs).await,
            // With a read in flight the tick runs here, and a fetched
            // credential wakes the read; with none (a backoff, idle) the
            // loop's top takes the tick and sends the `AUTH` at once.
            Event::ReauthDue => {
                if inflight.is_some() {
                    reauth_tick(ctx, &core, &mut st, &knobs, true).await;
                }
            }
            Event::KnobsChanged => {}
            Event::Ctl(None) => return false,
            Event::Ctl(Some(ctl)) => {
                st.apply(&mut core, ctl);
                if !st.drain(&mut core, rx) {
                    return false;
                }
                // With a read in flight, a join wakes it when the read does
                // not cover it (a coverage wake), or when the connection has
                // not answered within `liveness_age` or has failed since (a
                // liveness wake: the wake-ack watchdog then gives the join a
                // connection verdict). A wake already outstanding for the read
                // serves this join too. With no read in flight (a backoff),
                // the join is kept and decided as the next read is issued.
                if let Some(reason) =
                    st.take_wake_reason(&core, inflight.is_some(), knobs.liveness_age)
                {
                    send_wake(ctx, &core, &mut st, reason);
                }
            }
        }
    }
}

/// Sends a `reason` wake ([`WAKE_REASON_COVERAGE`], [`WAKE_REASON_LIVENESS`]
/// or [`WAKE_REASON_CREDENTIAL`]) to the read in flight: counts it on the
/// reader's stats, records it as the read's outstanding wake, and spawns the
/// wake task, which reports back [`Ctl::WakeLanded`] or [`Ctl::WakeFailed`].
fn send_wake(ctx: &ReaderCtx, core: &Core, st: &mut LoopState, reason: &'static str) {
    let key = core
        .inflight_wake_key()
        .expect("in-flight read names a wake key")
        .to_owned();
    if reason == WAKE_REASON_CREDENTIAL {
        tracing::debug!(
            reader = ctx.idx,
            read = st.read,
            reason,
            "waking the in-flight multiplexed XREAD so the reader can re-AUTH its connection \
             between reads"
        );
    } else {
        tracing::debug!(
            reader = ctx.idx,
            read = st.read,
            reason,
            "waking the in-flight multiplexed XREAD for a joining subscription"
        );
    }
    ctx.shared.stats[ctx.idx]
        .wakes
        .fetch_add(1, Ordering::AcqRel);
    st.wake = Some(PendingWake {
        read: st.read,
        landed: None,
    });
    tokio::spawn(wake_and_report(
        ctx.data.clone(),
        key,
        ctx.shared.clone(),
        ctx.ctl.clone(),
        st.read,
        reason,
    ));
}

fn start_xread(ctx: &ReaderCtx, plan: ReadPlan, knobs: &StreamMuxKnobs) -> XreadFuture {
    #[cfg(any(test, feature = "test-fault-injection"))]
    match ctx.shared.pop_fault(MuxFaultSite::Xread) {
        None | Some(MuxFault::Pass) => {}
        Some(MuxFault::Err(e)) => return Box::pin(async move { Err(e) }),
        Some(MuxFault::Hang) => {
            // Ends only when the reader force-reconnects, failing as fred fails
            // a replayed single-attempt command (see `MuxFault::Hang`).
            let (shared, idx) = (ctx.shared.clone(), ctx.idx);
            let seen = shared.reconnect_signal(idx).generation();
            return Box::pin(async move {
                shared.reconnect_signal(idx).changed_since(seen).await;
                let (kind, message) = crate::storage::client::BUDGET_EXHAUSTED_SHAPES[0].clone();
                Err(fred::error::Error::new(kind, message))
            });
        }
        Some(MuxFault::Reply(v)) => return Box::pin(async move { Ok(v) }),
        Some(MuxFault::Panic) => panic!("injected stream multiplexer reader panic"),
        Some(MuxFault::Hold(_)) => {
            unreachable!("push_faults_for_test refuses Hold at the Xread site")
        }
    }
    let options = Options {
        timeout: Some(Duration::from_millis(knobs.block_ms + 5_000)),
        max_attempts: Some(CONSUMER_XREAD_MAX_ATTEMPTS),
        ..Default::default()
    };
    let client = match ctx.client() {
        Ok(client) => client.with_options(&options),
        Err(e) => return Box::pin(async move { Err(e) }),
    };
    let block_ms = knobs.block_ms;
    let ReadPlan { keys, ids, count } = plan;
    Box::pin(async move {
        client
            .xread::<Value, _, _>(Some(count), Some(block_ms), keys, ids)
            .await
    })
}

/// Wakes the read watching `key` for `reason` — a joining subscription
/// ([`WAKE_REASON_COVERAGE`] or [`WAKE_REASON_LIVENESS`]) or a due re-AUTH
/// ([`WAKE_REASON_CREDENTIAL`]) — then reports the result for read number
/// `read` to the reader: [`Ctl::WakeLanded`] or [`Ctl::WakeFailed`]. A failed
/// wake's WARN states that kind's cost: a coverage wake's subscription may
/// wait out the in-flight read; a liveness wake's delays nothing, but its join
/// gets no connection check from it; a credential wake's `AUTH` waits until the
/// in-flight read returns.
async fn wake_and_report(
    data: Pool,
    key: String,
    shared: Arc<MuxShared>,
    ctl: mpsc::UnboundedSender<Ctl>,
    read: u64,
    reason: &'static str,
) {
    let result = match wake(&data, &key, &shared).await {
        Ok(at) => Ctl::WakeLanded { read, at },
        Err(e) => {
            let block_ms = shared.knobs().block_ms;
            if reason == WAKE_REASON_LIVENESS {
                tracing::warn!(
                    wake_key = %key,
                    error = %e,
                    block_ms,
                    reason,
                    "stream multiplexer wake failed; the joining subscription's delivery is not \
                     delayed, but it gets no connection check from this wake (the next join \
                     sends another; the XREAD's own timeout still applies)"
                );
            } else if reason == WAKE_REASON_CREDENTIAL {
                tracing::warn!(
                    wake_key = %key,
                    error = %e,
                    block_ms,
                    reason,
                    "stream multiplexer wake failed; the reader's credential re-AUTH waits until \
                     the in-flight XREAD returns (up to XREAD_BLOCK_MS), and is sent then"
                );
            } else {
                tracing::warn!(
                    wake_key = %key,
                    error = %e,
                    block_ms,
                    reason,
                    "stream multiplexer wake failed; the new subscription's first live update may \
                     wait until the in-flight XREAD returns (up to XREAD_BLOCK_MS)"
                );
            }
            Ctl::WakeFailed { read }
        }
    };
    // The reader's receiver outlives every sender it hands out, except at
    // shutdown, when there is no one left to tell.
    let _ = ctl.send(result);
}

/// Appends to wake key `key` (atomically with its TTL) so the XREAD watching
/// it returns, retrying up to [`WAKE_ATTEMPTS`] times. Returns when the append
/// landed, or the last attempt's error. Counts
/// `live_edit_stream_mux_wakes_total{outcome}` once per wake.
async fn wake(data: &Pool, key: &str, shared: &MuxShared) -> Result<Instant, fred::error::Error> {
    let mut last_err = None;
    for attempt in 0..WAKE_ATTEMPTS {
        if attempt > 0 {
            tokio::time::sleep(WAKE_RETRY_DELAY).await;
        }
        match wake_once(data, key, shared).await {
            Ok(()) => {
                let at = Instant::now();
                if let Some(m) = shared.metrics() {
                    m.inc_stream_mux_wake(WAKE_OUTCOME_OK);
                }
                return Ok(at);
            }
            Err(e) => last_err = Some(e),
        }
    }
    if let Some(m) = shared.metrics() {
        m.inc_stream_mux_wake(WAKE_OUTCOME_ERROR);
    }
    Err(last_err.expect("WAKE_ATTEMPTS is at least one"))
}

async fn wake_once(data: &Pool, key: &str, shared: &MuxShared) -> Result<(), fred::error::Error> {
    #[cfg(any(test, feature = "test-fault-injection"))]
    match shared.pop_fault(MuxFaultSite::Wake) {
        None | Some(MuxFault::Pass) => {}
        Some(MuxFault::Err(e)) => return Err(e),
        Some(MuxFault::Hang) => std::future::pending::<()>().await,
        Some(MuxFault::Reply(_) | MuxFault::Panic | MuxFault::Hold(_)) => {
            unreachable!("push_faults_for_test refuses Reply, Panic and Hold at the Wake site")
        }
    }
    #[cfg(not(any(test, feature = "test-fault-injection")))]
    let _ = shared;
    // b.zt3 re-send audit: fred keeps its default write attempts here, so a lost
    // connection can run this twice: an `XADD MAXLEN 1`, so one extra wake.
    STREAM_MUX_WAKE_SCRIPT
        .evalsha_with_reload::<Value, _, _>(
            data.next(),
            vec![key.to_owned()],
            vec![WAKE_KEY_TTL_MS],
        )
        .await
        .map(|_| ())
}

/// Fails the reader's subscriptions and wakes its blocked read, if any, so the
/// reader connection's QUIT is not queued behind a BLOCK. Returns whether a
/// blocked read was woken. The connection itself stays in the reader's slot
/// for [`super::StreamMux::quit_readers`].
async fn shutdown(
    ctx: &ReaderCtx,
    core: &mut Core,
    st: &mut LoopState,
    rx: &mut mpsc::UnboundedReceiver<Ctl>,
    inflight: bool,
) -> bool {
    // Apply what is still queued first: a subscription whose `Add` was queued
    // behind the cancel then ends with the shutdown status below, rather than
    // with the reason-less ending a dropped sink leaves its consumer.
    let _ = st.drain(core, rx);
    // No re-AUTH goes out from here on: the connection is about to be quit.
    st.reauth.clear(ctx);
    let wake_key = core.inflight_wake_key().map(str::to_owned);
    // Same client-visible status as every other shutdown ending
    // (`grpc::transport::server_shutting_down`): the sink's `fail` attaches the
    // shutdown retry hint to a `shutdown`-reason failure.
    core.fail_all(|_key| Failure {
        reason: STREAM_EXIT_SHUTDOWN,
        message: "server shutting down".to_owned(),
    });
    let (true, Some(key)) = (inflight, wake_key) else {
        return false;
    };
    match tokio::time::timeout(SHUTDOWN_WAKE_TIMEOUT, wake(&ctx.data, &key, &ctx.shared)).await {
        Ok(Ok(_)) => true,
        Ok(Err(e)) => {
            tracing::warn!(
                reader = ctx.idx,
                wake_key = %key,
                error = %e,
                "stream multiplexer shutdown wake failed; the reader connection's QUIT may wait \
                 until the in-flight XREAD returns, or is cut off at its timeout"
            );
            false
        }
        Err(_) => false,
    }
}

/// Forces a reconnect of the reader's own connection, counted under
/// `valkey_consumer_forced_reconnects_total{phase}` — the verdict is counted
/// whichever remedy follows.
///
/// On a cluster the reader first re-resolves where its connection should
/// point, so it never keeps reconnecting a node that no longer serves it:
/// - an idle reader (`phase="keepalive"`) whose node the readers' routing
///   view no longer lists as a primary retires its connection, and connects
///   afresh when it next serves a subscription;
/// - on every other phase, when the bound slot's owner is now another node
///   (a failover), the reader retires its connection and retargets at the
///   next connection step.
///
/// Otherwise — and off a cluster, and when the routing view cannot say —
/// fred's `force_reconnection()` rebuilds the connection in place. A reader
/// with no connection has nothing to reconnect.
async fn force_reconnect(ctx: &ReaderCtx, st: &mut LoopState, phase: &'static str) {
    if let Some(m) = ctx.shared.metrics() {
        m.inc_valkey_consumer_forced_reconnect(phase);
    }
    tracing::warn!(
        reader = ctx.idx,
        phase,
        "forcing a reconnect of a stream multiplexer reader connection"
    );
    let Some((client, node)) = ctx.shared.conn(ctx.idx) else {
        #[cfg(any(test, feature = "test-fault-injection"))]
        ctx.shared.reconnect_signal(ctx.idx).bump();
        return;
    };
    if let (true, Some(node)) = (ctx.cluster, node) {
        let moved = if phase == CONSUMER_PHASE_KEEPALIVE {
            ctx.connector.is_primary(&node) == Some(false)
        } else {
            bound_slot(st)
                .and_then(|slot| ctx.connector.slot_owner(slot))
                .is_some_and(|owner| owner != node)
        };
        if moved {
            let retarget = phase != CONSUMER_PHASE_KEEPALIVE;
            tracing::info!(
                reader = ctx.idx,
                phase,
                node = %node,
                retarget,
                "the stream reader's node no longer serves it (no longer a primary, or no longer \
                 the owner of its hash slot); retiring the connection instead of reconnecting it"
            );
            retire_current(ctx, st).await;
            st.failed();
            st.probe_due = true;
            st.retarget_due = retarget;
            return;
        }
    }
    let reconnect = tokio::time::timeout(FORCE_RECONNECT_TIMEOUT, client.force_reconnection());
    let outcome = tokio::select! {
        () = ctx.cancel.cancelled() => return,
        r = reconnect => r,
    };
    #[cfg(any(test, feature = "test-fault-injection"))]
    ctx.shared.reconnect_signal(ctx.idx).bump();
    match outcome {
        Ok(Ok(())) => tracing::info!(reader = ctx.idx, "forced reader reconnect completed"),
        Ok(Err(e)) => {
            tracing::warn!(reader = ctx.idx, error = %e, "forced reader reconnect failed")
        }
        Err(_) => tracing::warn!(
            reader = ctx.idx,
            timeout_ms = FORCE_RECONNECT_TIMEOUT.as_millis() as u64,
            "forced reader reconnect did not complete in time"
        ),
    }
}

/// After a failure that ended every subscription on the reader: charge it to
/// the budget, force-reconnect the connection (`phase="permanent"`), and back
/// off before the next read. Re-subscribes then land on a fresh connection,
/// and repeated failures back off exponentially up to the cap, because only a
/// successful XREAD (with every head check it needed answered), or the budget
/// running out and ending the reader's subscriptions, resets the budget.
async fn heal_and_back_off(
    ctx: &ReaderCtx,
    st: &mut LoopState,
    knobs: &StreamMuxKnobs,
    e: &fred::error::Error,
) -> Option<Backoff> {
    st.budget.failures = st.budget.failures.saturating_add(1);
    force_reconnect(ctx, st, CONSUMER_PHASE_PERMANENT).await;
    let (window_ms, sleep_ms) =
        backoff_window_ms(st.budget.failures, knobs.backoff_initial, knobs.backoff_cap);
    tracing::debug!(
        reader = ctx.idx,
        error = %e,
        consecutive_failures = st.budget.failures,
        backoff_window_ms = window_ms,
        sleep_ms,
        "backing off before the next multiplexed XREAD after a whole-reader failure"
    );
    Some(Box::pin(tokio::time::sleep(Duration::from_millis(
        sleep_ms,
    ))))
}

/// The hash slot the reader's reads are bound to (its latest planned read's),
/// on a cluster; `None` before its first binding and off a cluster.
fn bound_slot(st: &LoopState) -> Option<u16> {
    st.slot_tag.as_ref().map(|(slot, _)| *slot)
}

/// Retires the reader's current connection, if any: takes it out of the
/// reader's slot (clearing its node on the router) and hands it to
/// [`ReaderConnector::retire`], which ends its supervision and hooks and quits
/// it — or, when its client is not connected (a failover's dead primary),
/// aborts its router without a `QUIT`, so the retarget's connect is not held
/// up. A `QUIT` that overran is logged; the retire has aborted its router. The
/// connection's re-AUTH schedule, and any credential still waiting for its
/// `AUTH`, go with it.
async fn retire_current(ctx: &ReaderCtx, st: &mut LoopState) {
    st.reauth.clear(ctx);
    let Some(conn) = ctx.shared.take_conn(ctx.idx) else {
        return;
    };
    let node = conn.node().map(ToString::to_string);
    let outcome = ctx.connector.retire(conn).await;
    #[cfg(any(test, feature = "test-fault-injection"))]
    ctx.shared.reconnect_signal(ctx.idx).bump();
    if outcome.quit_overran {
        tracing::warn!(
            reader = ctx.idx,
            node = node.as_deref().unwrap_or("default"),
            "a retired stream reader connection did not answer its QUIT in time; its router was \
             aborted"
        );
    }
}

/// What the connection step before a binding's first XREAD found.
enum ConnStep {
    /// The reader holds a connection that serves the read; the probe follows.
    Ready,
    /// The connect failed: the backoff before the next step, if any (none
    /// when the failure ended every subscription without one).
    Failed(Option<Backoff>),
    /// Shutdown began mid-connect.
    Cancelled,
}

/// The connection step: before the first XREAD of a binding, make sure the
/// reader's connection serves it.
///
/// - No connection: connect — `cause="retarget"` when the reader retired its
///   last one because its slot's owner moved, `cause="bind"` otherwise.
/// - On a cluster, a connection to a node other than the bound slot's owner
///   in the readers' routing view: retire it, then connect to the owner
///   (`cause="retarget"`). An owner the view does not name keeps the
///   connection; a wrong one fails its reads `-MOVED`, which comes back here.
/// - Otherwise keep it.
///
/// A new connection is published to the reader's slot (its node recorded on
/// the router) and then has its metric hooks armed and its re-AUTH tick
/// scheduled. A failed connect is handled by [`handle_connect_failure`].
///
/// Each connect attempt takes one `MuxFaultSite::Connect` fault when the
/// test seams are compiled in.
async fn ensure_connection(
    ctx: &ReaderCtx,
    core: &mut Core,
    st: &mut LoopState,
    knobs: &StreamMuxKnobs,
) -> ConnStep {
    let slot = if ctx.cluster {
        core.first_key()
            .map(|key| fred::util::redis_keyslot(key.as_bytes()))
    } else {
        None
    };
    let cause = match ctx.shared.conn(ctx.idx) {
        None if st.retarget_due => READER_CONNECT_CAUSE_RETARGET,
        None => READER_CONNECT_CAUSE_BIND,
        Some((_, node)) => {
            let owner = slot.and_then(|slot| ctx.connector.slot_owner(slot));
            match (owner, node) {
                (Some(owner), Some(node)) if owner != node => {
                    tracing::debug!(
                        reader = ctx.idx,
                        from = %node,
                        to = %owner,
                        "the stream reader's new hash slot is served by another node; retargeting \
                         its connection"
                    );
                    retire_current(ctx, st).await;
                    st.retarget_due = true;
                    READER_CONNECT_CAUSE_RETARGET
                }
                _ => return ConnStep::Ready,
            }
        }
    };
    // One `MuxFaultSite::Connect` fault per connect attempt; the connector
    // applies it to the attempt's setup, inside its counted region.
    #[cfg(any(test, feature = "test-fault-injection"))]
    let fault = ctx.shared.pop_fault(MuxFaultSite::Connect);
    #[cfg(not(any(test, feature = "test-fault-injection")))]
    let fault = None;
    match ctx
        .connector
        .connect(ctx.idx, slot, cause, fault, &ctx.cancel)
        .await
    {
        Ok(conn) => {
            st.retarget_due = false;
            match ctx.shared.publish(ctx.idx, conn) {
                None => {
                    // After the publish, so `install_observability_hooks`
                    // running concurrently either finds the connection in its
                    // slot or has installed the metrics this arm reads —
                    // which also counts a connection whose connect came before
                    // the metrics. Under the slot's lock, so a connection
                    // `quit_readers` took meanwhile is not armed.
                    ctx.shared.arm_published(ctx.idx, &ctx.connector);
                    // The new connection authenticated at its handshake; its
                    // first re-AUTH tick is one interval away.
                    st.reauth.schedule(ctx);
                    ConnStep::Ready
                }
                Some(conn) => {
                    // The readers are being quit: this connection came up too
                    // late to be published, so retire it here.
                    let _ = ctx.connector.retire(conn).await;
                    ConnStep::Cancelled
                }
            }
        }
        Err(ReaderConnectError::Cancelled) => ConnStep::Cancelled,
        Err(ReaderConnectError::Failed { node, error }) => ConnStep::Failed(
            handle_connect_failure(ctx, core, st, knobs, cause, node.as_ref(), &error),
        ),
    }
}

/// Acts on a failed connect (`cause`, to `node` — `None`: the URL's endpoint)
/// and logs it in exactly one WARN. Returns the backoff before the next step.
///
/// The connect stays due either way (the probe is left due).
/// - **Transient** (`is_transient_fred_error`): one transient failure on the
///   budget, like a failed probe; the subscriptions are kept and the connect is
///   retried after a full-jitter backoff. Past the budget every subscription
///   on the reader ends `Unavailable` and the budget resets.
/// - **Anything else** (`Auth`, `Tls`, `Config`, …): no retry can succeed by
///   itself, so it is a whole-reader failure, as a permanent probe error is:
///   every subscription on the reader ends `Unavailable` (their clients
///   re-subscribe, which connects again), charged to the budget with a
///   backoff before the next attempt.
fn handle_connect_failure(
    ctx: &ReaderCtx,
    core: &mut Core,
    st: &mut LoopState,
    knobs: &StreamMuxKnobs,
    cause: &'static str,
    node: Option<&Server>,
    e: &fred::error::Error,
) -> Option<Backoff> {
    st.failed();
    st.probe_due = true;
    st.budget.failures = st.budget.failures.saturating_add(1);
    let failures = st.budget.failures;
    let kind = e.kind().to_str();
    let node = node.map_or_else(|| "the configured endpoint".to_owned(), ToString::to_string);
    let transient = is_transient_fred_error(e);
    if transient && failures <= knobs.max_transient_failures {
        let (window_ms, sleep_ms) =
            backoff_window_ms(failures, knobs.backoff_initial, knobs.backoff_cap);
        tracing::warn!(
            reader = ctx.idx,
            node = %node,
            cause,
            kind,
            error = %e,
            consecutive_failures = failures,
            backoff_window_ms = window_ms,
            sleep_ms,
            "stream reader could not connect to the backing store; its subscriptions are kept and \
             the connect is retried after a backoff"
        );
        return Some(Box::pin(tokio::time::sleep(Duration::from_millis(
            sleep_ms,
        ))));
    }
    let n = core.fail_all(|key| Failure {
        reason: STREAM_EXIT_ERROR,
        message: if transient {
            format!(
                "stream reader for {key} gave up after {failures} consecutive transient failures \
                 (last: it could not connect to {node}, kind: {kind})"
            )
        } else {
            format!("stream reader for {key} could not connect to {node} (kind: {kind})")
        },
    });
    st.retarget_due = false;
    if transient {
        tracing::warn!(
            reader = ctx.idx,
            node = %node,
            cause,
            kind,
            error = %e,
            subscriptions = n,
            "stream reader could not connect to the backing store and its transient retry budget \
             is exhausted; its subscriptions end with UNAVAILABLE and their clients re-subscribe, \
             which retries the connect"
        );
        st.budget = TransientBudget::default();
        return None;
    }
    let (window_ms, sleep_ms) =
        backoff_window_ms(failures, knobs.backoff_initial, knobs.backoff_cap);
    tracing::warn!(
        reader = ctx.idx,
        node = %node,
        cause,
        kind,
        error = %e,
        subscriptions = n,
        backoff_window_ms = window_ms,
        sleep_ms,
        "stream reader could not connect to the backing store (an error retrying cannot fix, such \
         as a rejected credential or a TLS failure); its subscriptions end with UNAVAILABLE and \
         their clients re-subscribe, which retries the connect"
    );
    Some(Box::pin(tokio::time::sleep(Duration::from_millis(
        sleep_ms,
    ))))
}

/// After a `-MOVED` on the reader's connection (an XREAD, a head check or a
/// probe): whether its bound slot's owner is now a node other than the one it
/// is connected to. Re-resolves from the readers' routing view, and when
/// that view still names the reader's node (or none), syncs it once
/// (`ReaderConnector::sync_routing`: `CLUSTER SLOTS` on the routing client,
/// single-flight across the readers, bounded by the command timeout — or the
/// connect timeout when the command timeout is disabled) and resolves again.
/// `false` off a cluster, with nothing bound, without a connection, and when
/// shutdown begins mid-sync; the caller then treats the `MOVED` as an
/// ordinary transient failure.
async fn slot_moved(ctx: &ReaderCtx, st: &LoopState) -> bool {
    if !ctx.cluster {
        return false;
    }
    let (Some(slot), Some(node)) = (bound_slot(st), ctx.node()) else {
        return false;
    };
    let elsewhere = |owner: Option<Server>| owner.is_some_and(|owner| owner != node);
    if elsewhere(ctx.connector.slot_owner(slot)) {
        return true;
    }
    let synced = tokio::select! {
        biased;
        () = ctx.cancel.cancelled() => return false,
        synced = ctx.connector.sync_routing() => synced,
    };
    if let Err(e) = synced {
        tracing::debug!(
            reader = ctx.idx,
            error = %e,
            "re-syncing the cluster routing view after a MOVED failed"
        );
    }
    elsewhere(ctx.connector.slot_owner(slot))
}

/// After `what` (an XREAD, a head check or a connection probe) failed
/// `-MOVED` and [`slot_moved`] found the bound slot's owner moved: makes the
/// probe due, so the next connection step retargets the reader's connection
/// to the new owner, and counts the retarget on the streak
/// ([`TransientBudget::moved_streak`]) that only a successful read resets.
/// Returns the backoff before that step, if any.
///
/// - **The first retarget of a streak** is free: no budget charge and no
///   backoff, so a single failover costs a reader one prompt retarget.
/// - **Every later one** — the slot's owner moved again before any read
///   succeeded — takes the transient remedy: one transient failure on the
///   budget, a WARN and a full-jitter backoff, after which the reader
///   retargets. Past the budget every subscription on the reader ends
///   `Unavailable` and the budget (the streak with it) resets.
fn charge_moved_retarget(
    ctx: &ReaderCtx,
    core: &mut Core,
    st: &mut LoopState,
    knobs: &StreamMuxKnobs,
    what: &str,
) -> Option<Backoff> {
    st.probe_due = true;
    st.budget.moved_streak = st.budget.moved_streak.saturating_add(1);
    let streak = st.budget.moved_streak;
    if streak < 2 {
        return None;
    }
    st.budget.failures = st.budget.failures.saturating_add(1);
    let failures = st.budget.failures;
    if failures > knobs.max_transient_failures {
        let n = core.fail_all(|key| Failure {
            reason: STREAM_EXIT_ERROR,
            message: format!(
                "stream reader for {key} gave up after {failures} consecutive transient \
                 failures (last: {what} was redirected (MOVED) to a new hash slot owner again \
                 before any read succeeded)"
            ),
        });
        tracing::warn!(
            reader = ctx.idx,
            consecutive_retargets = streak,
            subscriptions = n,
            "stream reader transient retry budget exhausted by repeated MOVED retargets; ending \
             the reader's subscriptions with Unavailable"
        );
        st.budget = TransientBudget::default();
        return None;
    }
    let (window_ms, sleep_ms) =
        backoff_window_ms(failures, knobs.backoff_initial, knobs.backoff_cap);
    tracing::warn!(
        reader = ctx.idx,
        consecutive_retargets = streak,
        consecutive_failures = failures,
        backoff_window_ms = window_ms,
        sleep_ms,
        "stream reader's hash slot owner moved again (MOVED) before any read succeeded; backing \
         off before retargeting its connection"
    );
    Some(Box::pin(tokio::time::sleep(Duration::from_millis(
        sleep_ms,
    ))))
}

/// What one idle→active connection probe found.
enum ProbeOutcome {
    Answered,
    /// No answer within `reader_probe_timeout`.
    TimedOut,
    Failed(fred::error::Error),
    /// Shutdown began mid-probe; shutdown ends every subscription.
    Cancelled,
}

/// The idle→active probe: one `PING`, before the reader's first XREAD after it
/// had nothing to read (and, on a cluster, before its first XREAD bound to a
/// different hash slot than its previous read), on the reader's one
/// connection — the one that XREAD will use, which the connection step just
/// made sure targets the read's slot owner on a cluster. Bounded by
/// `knobs.reader_probe_timeout` ([`super::READER_PROBE_TIMEOUT`]) and raced
/// against the reader's cancellation. No XREAD is in flight, so it interrupts
/// nothing.
async fn probe_connection(ctx: &ReaderCtx, knobs: &StreamMuxKnobs) -> ProbeOutcome {
    tokio::select! {
        biased;
        () = ctx.cancel.cancelled() => ProbeOutcome::Cancelled,
        answer = tokio::time::timeout(knobs.reader_probe_timeout, probe_ping(ctx)) => {
            match answer {
                Err(_) => ProbeOutcome::TimedOut,
                Ok(Ok(())) => ProbeOutcome::Answered,
                Ok(Err(e)) => ProbeOutcome::Failed(e),
            }
        }
    }
}

async fn probe_ping(ctx: &ReaderCtx) -> Result<(), fred::error::Error> {
    #[cfg(any(test, feature = "test-fault-injection"))]
    match ctx.shared.pop_fault(MuxFaultSite::Probe) {
        None | Some(MuxFault::Pass) => {}
        Some(MuxFault::Err(e)) => return Err(e),
        Some(MuxFault::Hang) => std::future::pending::<()>().await,
        Some(MuxFault::Reply(_) | MuxFault::Panic | MuxFault::Hold(_)) => {
            unreachable!("push_faults_for_test refuses Reply, Panic and Hold at the Probe site")
        }
    }
    ctx.client()?.ping::<()>(None).await
}

/// Acts on an idle→active probe's outcome. Returns the backoff before the next
/// step, if any.
///
/// Every outcome but cancellation updates the connection's last answer: an
/// answered probe stamps it, any failure clears it.
///
/// - **Answered:** the XREAD follows. The budget is left as it is: only a
///   successful XREAD resets it.
/// - **No answer in time** (or a fred `Timeout`): the connection is read as
///   wedged and force-reconnected (`phase="probe"`), and the probe counts one
///   transient failure; the reader backs off and probes again.
/// - **A `-MOVED` whose slot owner has moved** ([`slot_moved`]): the probe
///   stays due and the next connection step retargets — at once, with no
///   charge, when it is the first retarget since a successful read;
///   otherwise after a charged backoff ([`charge_moved_retarget`]). (A
///   `PING` names no key, so only an injected fault produces one.)
/// - **A transient error** (the reader's XREAD classification, a `-MOVED`
///   with an unchanged owner included): one transient failure, back off,
///   probe again — no forced reconnect, as for a transient XREAD error.
/// - Once transient failures exceed the budget, every subscription on the
///   reader ends `Unavailable` and the budget resets, as on the XREAD path.
/// - **A permanent error:** no key can explain a `PING` failing, so it is a
///   whole-reader failure: counted on
///   `valkey_consumer_permanent_exits_total{phase="probe"}`, every
///   subscription on the reader ends `Unavailable`, and the connection heals
///   and backs off ([`heal_and_back_off`], `phase="permanent"`).
async fn handle_probe(
    ctx: &ReaderCtx,
    core: &mut Core,
    st: &mut LoopState,
    knobs: &StreamMuxKnobs,
    outcome: ProbeOutcome,
) -> Option<Backoff> {
    let timeout_ms = knobs.reader_probe_timeout.as_millis() as u64;
    let e = match outcome {
        ProbeOutcome::Answered => {
            st.probe_due = false;
            st.answered();
            return None;
        }
        ProbeOutcome::Cancelled => return None,
        ProbeOutcome::TimedOut => {
            st.failed();
            tracing::warn!(
                reader = ctx.idx,
                timeout_ms,
                "stream reader connection probe did not answer in time; treating the reader \
                 connection as wedged"
            );
            force_reconnect(ctx, st, CONSUMER_PHASE_PROBE).await;
            return charge_failed_probe(ctx, core, st, knobs, "did not answer in time");
        }
        ProbeOutcome::Failed(e) => e,
    };
    st.failed();
    let kind = e.kind().to_str();
    if is_moved_redirect(&e) && slot_moved(ctx, st).await {
        // The probe stays due: the next connection step retargets.
        tracing::info!(
            reader = ctx.idx,
            error = %e,
            "stream reader connection probe was redirected (MOVED) and its hash slot's owner has \
             moved; retargeting the reader's connection"
        );
        return charge_moved_retarget(ctx, core, st, knobs, "a connection probe");
    }
    if matches!(e.kind(), fred::error::ErrorKind::Timeout) {
        tracing::warn!(
            reader = ctx.idx,
            error = %e,
            "stream reader connection probe timed out; treating the reader connection as wedged"
        );
        force_reconnect(ctx, st, CONSUMER_PHASE_PROBE).await;
        return charge_failed_probe(ctx, core, st, knobs, "timed out");
    }
    if is_retryable_reader_error(&e) {
        tracing::warn!(
            reader = ctx.idx,
            error = %e,
            kind,
            "stream reader connection probe failed transiently"
        );
        return charge_failed_probe(
            ctx,
            core,
            st,
            knobs,
            &format!("failed transiently (kind: {kind})"),
        );
    }
    if let Some(m) = ctx.shared.metrics() {
        m.inc_valkey_consumer_permanent_exit(CONSUMER_PHASE_PROBE);
    }
    let n = core.fail_all(|key| Failure {
        reason: STREAM_EXIT_ERROR,
        message: format!(
            "stream reader for {key} failed its connection probe with a permanent error (kind: \
             {kind})"
        ),
    });
    tracing::warn!(
        reader = ctx.idx,
        error = %e,
        kind,
        subscriptions = n,
        "stream reader connection probe failed permanently; ending the reader's subscriptions \
         with Unavailable"
    );
    heal_and_back_off(ctx, st, knobs, &e).await
}

/// Charges one failed idle→active probe (`what` failed it) to the transient
/// budget: backs off before the probe is retried, or — past the budget — ends
/// every subscription on the reader `Unavailable` and resets the budget.
fn charge_failed_probe(
    ctx: &ReaderCtx,
    core: &mut Core,
    st: &mut LoopState,
    knobs: &StreamMuxKnobs,
    what: &str,
) -> Option<Backoff> {
    st.budget.failures = st.budget.failures.saturating_add(1);
    let failures = st.budget.failures;
    if failures > knobs.max_transient_failures {
        let n = core.fail_all(|key| Failure {
            reason: STREAM_EXIT_ERROR,
            message: format!(
                "stream reader for {key} gave up after {failures} consecutive transient \
                 failures (last: its connection probe {what})"
            ),
        });
        tracing::warn!(
            reader = ctx.idx,
            subscriptions = n,
            "stream reader transient retry budget exhausted by failed connection probes; ending \
             the reader's subscriptions with Unavailable"
        );
        st.budget = TransientBudget::default();
        return None;
    }
    let (window_ms, sleep_ms) =
        backoff_window_ms(failures, knobs.backoff_initial, knobs.backoff_cap);
    tracing::debug!(
        reader = ctx.idx,
        consecutive_failures = failures,
        backoff_window_ms = window_ms,
        sleep_ms,
        "backing off before re-probing the stream reader connection"
    );
    Some(Box::pin(tokio::time::sleep(Duration::from_millis(
        sleep_ms,
    ))))
}

/// One idle keepalive tick, on a connected reader with nothing to read (no
/// XREAD in flight, so the `PING` interrupts nothing): one `PING` on the
/// reader's connection under `knobs.probe_timeout`, raced against the reader's
/// cancellation.
///
/// Counts `live_edit_stream_mux_keepalives_total{result}` once per tick: `ok`
/// when the `PING` answered, `failed` otherwise, in which case the reader
/// force-reconnects (`phase="keepalive"`) — or, on a cluster whose routing
/// view no longer lists the connection's node as a primary, retires the
/// connection ([`force_reconnect`]). The transient budget is neither charged
/// nor reset: it tracks reads, and an idle reader serves none.
async fn keepalive(ctx: &ReaderCtx, st: &mut LoopState, knobs: &StreamMuxKnobs) {
    let answer = tokio::select! {
        biased;
        () = ctx.cancel.cancelled() => return,
        answer = tokio::time::timeout(knobs.probe_timeout, keepalive_ping(ctx)) => answer,
    };
    let failure = match answer {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(e.to_string()),
        Err(_) => Some(format!(
            "no answer within {} ms",
            knobs.probe_timeout.as_millis()
        )),
    };
    let metrics = ctx.shared.metrics();
    let Some(failure) = failure else {
        if let Some(m) = &metrics {
            m.inc_stream_mux_keepalive(KEEPALIVE_RESULT_OK);
        }
        tracing::debug!(reader = ctx.idx, "idle stream reader keepalive answered");
        return;
    };
    if let Some(m) = &metrics {
        m.inc_stream_mux_keepalive(KEEPALIVE_RESULT_FAILED);
    }
    let node = ctx.node().map(|node| node.to_string());
    tracing::warn!(
        reader = ctx.idx,
        node = node.as_deref().unwrap_or("default"),
        failure = %failure,
        "idle stream reader keepalive failed; forcing a reconnect"
    );
    force_reconnect(ctx, st, CONSUMER_PHASE_KEEPALIVE).await;
}

/// The keepalive `PING` on the reader's connection.
async fn keepalive_ping(ctx: &ReaderCtx) -> Result<(), fred::error::Error> {
    #[cfg(any(test, feature = "test-fault-injection"))]
    match ctx.shared.pop_fault(MuxFaultSite::Keepalive) {
        None | Some(MuxFault::Pass) => {}
        Some(MuxFault::Err(e)) => return Err(e),
        Some(MuxFault::Hang) => std::future::pending::<()>().await,
        Some(MuxFault::Reply(_) | MuxFault::Panic | MuxFault::Hold(_)) => {
            unreachable!("push_faults_for_test refuses Reply, Panic and Hold at the Keepalive site")
        }
    }
    ctx.client()?.ping::<()>(None).await
}

/// One credential re-AUTH tick on the reader's held connection (b.19i).
///
/// Schedules the next tick, then asks the provider for the periodic
/// credential — `fetch(None)`, bounded by `knobs.probe_timeout` and raced
/// against the reader's cancellation — so the provider's rotation gate decides
/// as it does for fred's pooled refresh:
/// - **No password**: nothing to send.
/// - **A password**: held until its `AUTH` is sent between reads. With a read
///   in flight (`read_in_flight`) and no wake outstanding for it, the read gets
///   a credential wake so it returns now; an outstanding wake serves this too.
///   With no read in flight the loop sends the `AUTH` at once.
/// - **No credential** (the provider's error, or no answer in time): counted
///   `failed` on `live_edit_stream_mux_reauths_total` and logged at DEBUG —
///   the pooled clients' refresh warnings and readiness already report a
///   credential outage — and nothing else.
async fn reauth_tick(
    ctx: &ReaderCtx,
    core: &Core,
    st: &mut LoopState,
    knobs: &StreamMuxKnobs,
    read_in_flight: bool,
) {
    st.reauth.schedule(ctx);
    let Some(provider) = ctx.connector.reauth_provider().cloned() else {
        return;
    };
    if let Some(stats) = ctx.shared.stats.get(ctx.idx) {
        stats.reauth_ticks.fetch_add(1, Ordering::AcqRel);
    }
    let fetched = tokio::select! {
        biased;
        () = ctx.cancel.cancelled() => return,
        fetched = tokio::time::timeout(knobs.probe_timeout, provider.fetch(None)) => fetched,
    };
    let failure = match fetched {
        Ok(Ok((_, None))) => return,
        Ok(Ok((username, Some(password)))) => {
            st.reauth
                .set_pending(ctx, PendingCredential { username, password });
            if read_in_flight && st.wake.is_none() {
                send_wake(ctx, core, st, WAKE_REASON_CREDENTIAL);
            }
            return;
        }
        Ok(Err(e)) => e.to_string(),
        Err(_) => format!("no answer within {} ms", knobs.probe_timeout.as_millis()),
    };
    if let Some(m) = ctx.shared.metrics() {
        m.inc_stream_mux_reauth(REAUTH_RESULT_FAILED);
    }
    tracing::debug!(
        reader = ctx.idx,
        failure = %failure,
        "stream reader could not fetch a credential for its re-AUTH; nothing is sent, and the \
         next tick tries again"
    );
}

/// Sends the credential a re-AUTH tick fetched (b.19i) as `AUTH` on the
/// reader's own connection. Called only with no XREAD in flight, so the `AUTH`
/// never queues behind a `BLOCK`.
///
/// The pending credential is taken either way. Skipped (DEBUG, not counted)
/// when the reader holds no connection or its client is not `Connected`, as
/// fred's own refresh skips: a reconnect authenticates afresh at its
/// handshake. Otherwise bounded by `knobs.probe_timeout` and raced against the
/// reader's cancellation (a cancelled `AUTH` is not counted), and counted on
/// `live_edit_stream_mux_reauths_total{result}`:
/// - **`ok`**: answered. Not the connection's latest answer for the liveness
///   wake: a managed frontend answers `AUTH` even for a dead backend session.
/// - **`failed`**: an error or no answer in time: one WARN naming the reader,
///   the node and the error — never the credential. No forced reconnect and no
///   budget charge: the connection's health stays with the reader's
///   connection checks, and the next rotation or reconnect re-AUTHs it.
async fn send_reauth(ctx: &ReaderCtx, st: &mut LoopState, knobs: &StreamMuxKnobs) {
    let Some(credential) = st.reauth.take_pending(ctx) else {
        return;
    };
    let client = match ctx.client() {
        Ok(client) if client.state() == ClientState::Connected => client,
        _ => {
            tracing::debug!(
                reader = ctx.idx,
                "skipping the stream reader's credential re-AUTH: its connection is not \
                 connected, and a reconnect authenticates afresh"
            );
            return;
        }
    };
    let answer = tokio::select! {
        biased;
        () = ctx.cancel.cancelled() => return,
        answer = tokio::time::timeout(
            knobs.probe_timeout,
            reauth_command(ctx, &client, credential),
        ) => answer,
    };
    let failure = match answer {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(e.to_string()),
        Err(_) => Some(format!(
            "no answer within {} ms",
            knobs.probe_timeout.as_millis()
        )),
    };
    let metrics = ctx.shared.metrics();
    let Some(failure) = failure else {
        if let Some(m) = &metrics {
            m.inc_stream_mux_reauth(REAUTH_RESULT_OK);
        }
        tracing::debug!(
            reader = ctx.idx,
            "stream reader credential re-AUTH answered"
        );
        return;
    };
    if let Some(m) = &metrics {
        m.inc_stream_mux_reauth(REAUTH_RESULT_FAILED);
    }
    let node = ctx.node().map(|node| node.to_string());
    tracing::warn!(
        reader = ctx.idx,
        node = node.as_deref().unwrap_or("default"),
        failure = %failure,
        "stream reader credential re-AUTH failed; the connection is kept and nothing is \
         reconnected (the next rotation or reconnect re-authenticates it)"
    );
}

/// The re-AUTH's `AUTH` on the reader's client. The client is single-node, so
/// fred sends one plain `AUTH` on its one connection.
async fn reauth_command(
    ctx: &ReaderCtx,
    client: &Client,
    credential: PendingCredential,
) -> Result<(), fred::error::Error> {
    #[cfg(any(test, feature = "test-fault-injection"))]
    match ctx.shared.pop_fault(MuxFaultSite::Reauth) {
        None | Some(MuxFault::Pass) => {}
        Some(MuxFault::Err(e)) => return Err(e),
        Some(MuxFault::Hang) => std::future::pending::<()>().await,
        Some(MuxFault::Reply(_) | MuxFault::Panic | MuxFault::Hold(_)) => {
            unreachable!("push_faults_for_test refuses Reply, Panic and Hold at the Reauth site")
        }
    }
    #[cfg(not(any(test, feature = "test-fault-injection")))]
    let _ = ctx;
    let PendingCredential { username, password } = credential;
    client.auth(username, password).await
}

async fn handle_result(
    ctx: &ReaderCtx,
    core: &mut Core,
    st: &mut LoopState,
    knobs: &StreamMuxKnobs,
    res: Result<Value, fred::error::Error>,
) -> Option<Backoff> {
    let metrics = ctx.shared.metrics();
    let e = match res {
        Ok(value) => {
            if value.is_null() {
                st.answered();
                st.budget.record_success();
                core.on_read_empty();
                return None;
            }
            match value.into_xread_response::<String, String, String, Bytes>() {
                Ok(reply) => {
                    st.answered();
                    let wake_key = core.inflight_wake_key().map(str::to_owned);
                    let batch = parse_reply(reply, wake_key.as_deref());
                    if let (Some(m), Some(bytes)) =
                        (&metrics, reply_bytes(&batch, wake_key.as_deref()))
                    {
                        m.observe_stream_mux_xread_reply_bytes(bytes);
                    }
                    let suspects =
                        core.gap_suspects(&batch, |n| matches!(n, LayerNotification::Deleted));
                    let (heads, moved) = if suspects.is_empty() {
                        (Heads::new(), false)
                    } else {
                        let Some(checked) = check_heads(ctx, knobs, &suspects).await else {
                            // Shutdown began mid-check: nothing was consumed,
                            // and shutdown ends every subscription.
                            core.on_read_err();
                            return None;
                        };
                        checked
                    };
                    let failed_checks: Vec<String> = heads
                        .iter()
                        .filter(|(_, head)| **head == Head::Failed)
                        .map(|(key, _)| key.clone())
                        .collect();
                    // A reply with a failed head check is not a success: that
                    // key was not delivered, and it is re-read next. Checks
                    // that all answered are the connection's latest answer; a
                    // failed one voids the reply's.
                    if failed_checks.is_empty() {
                        if !heads.is_empty() {
                            st.answered();
                        }
                        st.budget.record_success();
                    } else {
                        st.failed();
                    }
                    let out = core.on_read_ok(batch, &heads);
                    if out.overflowed > 0 {
                        let on_bytes = out.overflowed_on_bytes;
                        let on_entries = out.overflowed - on_bytes;
                        tracing::warn!(
                            reader = ctx.idx,
                            subscriptions = out.overflowed,
                            on_entries,
                            on_bytes,
                            watchdog_ms = knobs.send_watchdog.as_millis() as u64,
                            capacity = knobs.channel_capacity,
                            hard_cap = knobs.backlog_hard_cap,
                            backlog_bytes = knobs.backlog_bytes,
                            backlog_hard_bytes = knobs.backlog_hard_bytes,
                            "subscribers fell behind beyond their backlog limits; tearing down"
                        );
                        if let Some(m) = &metrics {
                            for (bound, n) in [
                                (OverflowBound::Entries, on_entries),
                                (OverflowBound::Bytes, on_bytes),
                            ] {
                                if n > 0 {
                                    m.add_stream_mux_overflow_kills(bound.label(), n as u64);
                                }
                            }
                        }
                    }
                    if !failed_checks.is_empty() {
                        if moved && slot_moved(ctx, st).await {
                            // The failed keys are re-read on the slot's new
                            // owner once the connection step retargets — at
                            // once, with no budget charge, unless this is not
                            // the first retarget since a successful read.
                            tracing::info!(
                                reader = ctx.idx,
                                keys = ?failed_checks,
                                "stream head check was redirected (MOVED) and its hash slot's \
                                 owner has moved; retargeting the reader's connection"
                            );
                            return charge_moved_retarget(ctx, core, st, knobs, "a head check");
                        }
                        return charge_failed_head_checks(ctx, core, st, knobs, &failed_checks);
                    }
                    return None;
                }
                Err(e) => {
                    // A protocol-level failure: no single key can be blamed,
                    // and the budget is deliberately not reset by the read
                    // having returned — nor is it an answer.
                    st.failed();
                    core.on_read_err();
                    tracing::warn!(
                        reader = ctx.idx,
                        error = %e,
                        "multiplexed XREAD reply could not be converted; ending the reader's \
                         subscriptions with Unavailable"
                    );
                    if let Some(m) = &metrics {
                        m.inc_valkey_consumer_permanent_exit(CONSUMER_PHASE_XREAD);
                    }
                    core.fail_all(|key| Failure {
                        reason: STREAM_EXIT_ERROR,
                        message: format!(
                            "stream reader for {key} could not convert its XREAD response"
                        ),
                    });
                    return heal_and_back_off(ctx, st, knobs, &e).await;
                }
            }
        }
        Err(e) => e,
    };

    core.on_read_err();
    // `XREAD` is a read, which the store never refuses `-OOM`: no store
    // out-of-memory signal to note on (b.pa1).
    FredErrOrigin::metrics(
        metrics.as_ref(),
        POOL_LABEL_SUBSCRIBER,
        crate::storage::store_oom::StoreOomSignal::unwired(),
    )
    .observe_error(&e);
    let kind = e.kind().to_str();
    // `-UNBLOCKED` is the server answering; any other error — including other
    // `Unknown` replies, which cannot be told apart from fred-side failures —
    // voids the connection's last answer.
    if is_client_unblocked_error(&e) {
        st.answered();
    } else {
        st.failed();
    }

    if is_client_unblocked_error(&e) && st.budget.unblock_streak < MAX_CONSECUTIVE_UNBLOCK_RETRIES {
        st.budget.unblock_streak += 1;
        if let Some(m) = &metrics {
            m.inc_valkey_consumer_unblocked_retry();
        }
        tracing::debug!(
            reader = ctx.idx,
            consecutive_unblocks = st.budget.unblock_streak,
            "multiplexed XREAD interrupted by CLIENT UNBLOCK; re-planning"
        );
        return None;
    }

    // A single-node reader cannot follow a redirect: when its slot's owner
    // moved, retarget before the next read rather than retrying this node —
    // at once the first time since a successful read, after a charged backoff
    // every later time. An unchanged owner is an ordinary transient failure
    // (below).
    if is_moved_redirect(&e) && slot_moved(ctx, st).await {
        tracing::info!(
            reader = ctx.idx,
            error = %e,
            "multiplexed XREAD was redirected (MOVED) and its hash slot's owner has moved; \
             retargeting the reader's connection"
        );
        return charge_moved_retarget(ctx, core, st, knobs, "an XREAD");
    }

    if is_retryable_reader_error(&e) {
        st.budget.failures = st.budget.failures.saturating_add(1);
        if st.budget.failures > knobs.max_transient_failures {
            let failures = st.budget.failures;
            let n = core.fail_all(|key| Failure {
                reason: STREAM_EXIT_ERROR,
                message: format!(
                    "stream reader for {key} gave up after {failures} consecutive transient \
                     XREAD failures (last kind: {kind})"
                ),
            });
            tracing::warn!(
                reader = ctx.idx,
                error = %e,
                kind,
                subscriptions = n,
                "multiplexed XREAD transient retry budget exhausted; ending the reader's \
                 subscriptions with Unavailable"
            );
            st.budget = TransientBudget::default();
            return None;
        }
        if matches!(e.kind(), fred::error::ErrorKind::Timeout) {
            st.budget.timeout_streak = st.budget.timeout_streak.saturating_add(1);
            if should_force_reconnect(st.budget.timeout_streak) {
                force_reconnect(ctx, st, CONSUMER_PHASE_XREAD).await;
            }
        }
        let (window_ms, sleep_ms) =
            backoff_window_ms(st.budget.failures, knobs.backoff_initial, knobs.backoff_cap);
        tracing::warn!(
            reader = ctx.idx,
            error = %e,
            kind,
            consecutive_failures = st.budget.failures,
            backoff_window_ms = window_ms,
            sleep_ms,
            "multiplexed XREAD transient failure; backing off and retrying"
        );
        return Some(Box::pin(tokio::time::sleep(Duration::from_millis(
            sleep_ms,
        ))));
    }

    if let Some(m) = &metrics {
        m.inc_valkey_consumer_permanent_exit(CONSUMER_PHASE_XREAD);
    }
    if is_crossslot_error(&e) {
        tracing::error!(
            reader = ctx.idx,
            error = %e,
            "multiplexed XREAD spans hash slots: the Valkey endpoint is a cluster but \
             LIVE_EDIT_VALKEY_URL is not a cluster URL (use redis-cluster:// or \
             rediss-cluster://); ending the reader's subscriptions with Unavailable"
        );
        core.fail_all(|key| Failure {
            reason: STREAM_EXIT_ERROR,
            message: format!("stream reader for {key} hit CROSSSLOT (kind: {kind})"),
        });
        return heal_and_back_off(ctx, st, knobs, &e).await;
    }
    match isolate_permanent_failure(ctx, core, knobs, &e).await {
        Isolation::WholeReader => heal_and_back_off(ctx, st, knobs, &e).await,
        Isolation::KeysOnly | Isolation::Cancelled => None,
    }
}

/// What [`isolate_permanent_failure`] concluded.
enum Isolation {
    /// Some keys fail on their own; only their subscriptions ended.
    KeysOnly,
    /// No key explains the error; every subscription on the reader ended.
    WholeReader,
    /// Shutdown began mid-probe; nothing was ended here (shutdown ends it all).
    Cancelled,
}

/// A permanent error fails the whole multi-key XREAD. End only the keys that
/// fail on their own (for example a key that is not a stream); if none does,
/// the error is not key-specific and every subscription ends.
///
/// The `TYPE` probes run concurrently under one overall probe timeout
/// ([`StreamMuxKnobs::probe_timeout`], [`PROBE_TIMEOUT`] in production) and
/// race the reader's cancellation, so a degraded backing store stalls the
/// reader for one timeout however many keys it holds. A probe that fails or
/// does not answer in time does not mark its key poisoned.
async fn isolate_permanent_failure(
    ctx: &ReaderCtx,
    core: &mut Core,
    knobs: &StreamMuxKnobs,
    e: &fred::error::Error,
) -> Isolation {
    let kind = e.kind().to_str();
    let keys: Vec<String> = core.keys().map(str::to_owned).collect();
    let deadline = Instant::now() + knobs.probe_timeout;
    let probes = futures::future::join_all(keys.iter().map(|key| {
        let client = ctx.data.next();
        async move {
            matches!(
                tokio::time::timeout_at(deadline, client.r#type::<String, _>(key.as_str())).await,
                Ok(Ok(t)) if t != "stream" && t != "none"
            )
        }
    }));
    let verdicts = tokio::select! {
        biased;
        () = ctx.cancel.cancelled() => return Isolation::Cancelled,
        verdicts = probes => verdicts,
    };
    let poisoned: Vec<String> = keys
        .into_iter()
        .zip(verdicts)
        .filter_map(|(key, poisoned)| poisoned.then_some(key))
        .collect();
    if poisoned.is_empty() {
        let n = core.fail_all(|key| Failure {
            reason: STREAM_EXIT_ERROR,
            message: format!("stream reader for {key} hit a permanent XREAD error (kind: {kind})"),
        });
        tracing::warn!(
            reader = ctx.idx,
            error = %e,
            kind,
            subscriptions = n,
            "multiplexed XREAD permanent failure; ending the reader's subscriptions with \
             Unavailable"
        );
        return Isolation::WholeReader;
    }
    if let Some(m) = ctx.shared.metrics() {
        m.add_stream_mux_poison_keys(poisoned.len() as u64);
    }
    let n = core.fail_keys(&poisoned, |key| Failure {
        reason: STREAM_EXIT_ERROR,
        message: format!("stream key {key} is not readable as a stream (kind: {kind})"),
    });
    tracing::warn!(
        reader = ctx.idx,
        error = %e,
        kind,
        keys = ?poisoned,
        subscriptions = n,
        "stream keys failed a multiplexed XREAD on their own; ending only their subscriptions"
    );
    Isolation::KeysOnly
}

/// Checks the head of each stream a reply flagged
/// ([`ReaderCore::gap_suspects`]): `(key, id the key was read from)`.
///
/// Each check is one `XRANGE key - + COUNT 1` on the reader's own connection,
/// which is idle between reads (no XREAD is in flight to interrupt) and, on a
/// cluster, connected to the keys' hash slot's owner. The checks run concurrently under one
/// overall `knobs.probe_timeout` and race the reader's cancellation. Counts
/// `live_edit_stream_mux_head_checks_total{result}` once per key. Returns the
/// heads, and whether any check failed `-MOVED` ([`is_moved_redirect`]); `None`
/// when the reader was cancelled mid-check.
async fn check_heads(
    ctx: &ReaderCtx,
    knobs: &StreamMuxKnobs,
    suspects: &[(String, StreamId)],
) -> Option<(Heads, bool)> {
    let deadline = Instant::now() + knobs.probe_timeout;
    let checks = futures::future::join_all(suspects.iter().map(|(key, read_from)| async move {
        let mut moved = false;
        let head = match tokio::time::timeout_at(deadline, head_check(ctx, key)).await {
            Ok(Ok(first)) => Head::First(first),
            Ok(Err(e)) => {
                moved = is_moved_redirect(&e);
                tracing::warn!(
                    reader = ctx.idx,
                    stream_key = %key,
                    error = %e,
                    "stream head check failed; the stream is re-read by the next XREAD"
                );
                Head::Failed
            }
            Err(_) => {
                tracing::warn!(
                    reader = ctx.idx,
                    stream_key = %key,
                    timeout_ms = knobs.probe_timeout.as_millis() as u64,
                    "stream head check did not answer in time; the stream is re-read by the next \
                     XREAD"
                );
                Head::Failed
            }
        };
        (key.clone(), *read_from, head, moved)
    }));
    let checked = tokio::select! {
        biased;
        () = ctx.cancel.cancelled() => return None,
        checked = checks => checked,
    };
    let metrics = ctx.shared.metrics();
    let mut heads = Heads::with_capacity(checked.len());
    let mut any_moved = false;
    for (key, read_from, head, moved) in checked {
        any_moved |= moved;
        let result = match head {
            Head::Failed => HEAD_CHECK_RESULT_ERROR,
            Head::First(Some(first)) if head.passed(read_from) => {
                tracing::debug!(
                    reader = ctx.idx,
                    stream_key = %key,
                    read_from = %read_from,
                    first_retained = %first,
                    "layer stream trimmed past the reader's read position"
                );
                HEAD_CHECK_RESULT_GAP
            }
            Head::First(_) => HEAD_CHECK_RESULT_NO_GAP,
        };
        if let Some(m) = &metrics {
            m.inc_stream_mux_head_check(result);
        }
        heads.insert(key, head);
    }
    Some((heads, any_moved))
}

/// One head check: the id of `key`'s first retained entry, or `None` when the
/// stream holds no entry.
async fn head_check(ctx: &ReaderCtx, key: &str) -> Result<Option<StreamId>, fred::error::Error> {
    #[cfg(any(test, feature = "test-fault-injection"))]
    match ctx.shared.pop_fault(MuxFaultSite::HeadCheck) {
        None | Some(MuxFault::Pass) => {}
        Some(MuxFault::Err(e)) => return Err(e),
        Some(MuxFault::Hang) => std::future::pending::<()>().await,
        Some(MuxFault::Reply(_) | MuxFault::Panic | MuxFault::Hold(_)) => {
            unreachable!("push_faults_for_test refuses Reply, Panic and Hold at the HeadCheck site")
        }
    }
    let first: Vec<(String, std::collections::HashMap<String, Bytes>)> =
        ctx.client()?.xrange_values(key, "-", "+", Some(1)).await?;
    first_entry_id(first.first().map(|(id, _)| id.as_str()))
}

/// The id of a head check's first entry. An id that does not parse cannot come
/// from a stream, so it is an error rather than a gap claim either way.
fn first_entry_id(id: Option<&str>) -> Result<Option<StreamId>, fred::error::Error> {
    id.map(|id| {
        StreamId::parse(id).ok_or_else(|| {
            fred::error::Error::new(
                fred::error::ErrorKind::Parse,
                format!("head check returned an unparseable stream id {id:?}"),
            )
        })
    })
    .transpose()
}

/// After a reply in which some keys' head checks failed (`failed`): those keys
/// were not delivered, so the reply counts as one transient failure on the
/// reader's budget instead of a success. Like a transient XREAD failure it
/// backs off before the next read, which re-reads those keys from their
/// unchanged cursors; once the budget runs out every subscription on the
/// reader ends `Unavailable`, as for XREAD failures.
fn charge_failed_head_checks(
    ctx: &ReaderCtx,
    core: &mut Core,
    st: &mut LoopState,
    knobs: &StreamMuxKnobs,
    failed: &[String],
) -> Option<Backoff> {
    st.budget.failures = st.budget.failures.saturating_add(1);
    let failures = st.budget.failures;
    if failures > knobs.max_transient_failures {
        let n = core.fail_all(|key| Failure {
            reason: STREAM_EXIT_ERROR,
            message: format!(
                "stream reader for {key} gave up after {failures} consecutive transient \
                 failures (last: a stream head check failed)"
            ),
        });
        tracing::warn!(
            reader = ctx.idx,
            keys = ?failed,
            subscriptions = n,
            "stream reader transient retry budget exhausted by failed head checks; ending the \
             reader's subscriptions with Unavailable"
        );
        st.budget = TransientBudget::default();
        return None;
    }
    let (window_ms, sleep_ms) =
        backoff_window_ms(failures, knobs.backoff_initial, knobs.backoff_cap);
    tracing::debug!(
        reader = ctx.idx,
        keys = ?failed,
        consecutive_failures = failures,
        backoff_window_ms = window_ms,
        sleep_ms,
        "backing off before re-reading streams whose head check failed"
    );
    Some(Box::pin(tokio::time::sleep(Duration::from_millis(
        sleep_ms,
    ))))
}

fn parse_reply(
    reply: XReadResponse<String, String, String, Bytes>,
    wake_key: Option<&str>,
) -> Vec<(String, Vec<Entry<LayerNotification>>)> {
    reply
        .into_iter()
        .map(|(key, entries)| {
            let is_wake = wake_key == Some(key.as_str());
            let parsed = entries
                .into_iter()
                .filter_map(|(id_str, fields)| {
                    let Some(id) = StreamId::parse(&id_str) else {
                        tracing::warn!(stream_key = %key, entry_id = %id_str, "unparseable stream entry id; skipping");
                        return None;
                    };
                    let bytes = entry_bytes(&id_str, &fields);
                    let payload = if is_wake {
                        None
                    } else {
                        parse_payload(&key, &id_str, &fields)
                    };
                    Some(Entry {
                        id,
                        id_str,
                        payload,
                        bytes,
                    })
                })
                .collect();
            (key, parsed)
        })
        .collect()
}

/// One entry's size as the read-size estimate counts it: its id plus the
/// values of its `payload` and `ompf` fields ([`super::stream_entry_bytes`],
/// shared with the paged stream reads outside the multiplexer).
fn entry_bytes(id: &str, fields: &std::collections::HashMap<String, Bytes>) -> u64 {
    super::stream_entry_bytes(id, fields)
}

/// The bytes one XREAD reply carried, as [`entry_bytes`] counts them, over its
/// layer-stream keys (the wake key's entries, and entries whose id does not
/// parse, excluded). Observed on `live_edit_stream_mux_xread_reply_bytes`.
///
/// `None` when the reply carried no layer-stream entry — a reply holding only
/// the reader's wake entry, which every join that needs a wake produces — so
/// such a reply records no sample instead of a 0-byte one.
fn reply_bytes(
    batch: &[(String, Vec<Entry<LayerNotification>>)],
    wake_key: Option<&str>,
) -> Option<u64> {
    let mut layer_entries = batch
        .iter()
        .filter(|(key, _)| wake_key != Some(key.as_str()))
        .flat_map(|(_, entries)| entries.iter().map(|e| e.bytes))
        .peekable();
    layer_entries.peek()?;
    Some(layer_entries.sum())
}

/// Decodes one entry. The relay OMPF is **copied out** of the reply: fred
/// decodes zero-copy, so every field `Bytes` is a slice of the whole
/// multi-key XREAD reply, and keeping that slice would keep every other layer's
/// entries in the reply alive for as long as one subscriber has this entry
/// queued. One copy per entry per reader; every subscriber of the entry shares
/// it.
fn parse_payload(
    key: &str,
    id: &str,
    fields: &std::collections::HashMap<String, Bytes>,
) -> Option<LayerNotification> {
    let Some(raw) = fields.get(NOTIFICATION_PAYLOAD_FIELD) else {
        tracing::warn!(stream_key = %key, entry_id = %id, "stream entry missing `payload` field; skipping");
        return None;
    };
    let Ok(text) = std::str::from_utf8(raw) else {
        tracing::warn!(stream_key = %key, entry_id = %id, "stream entry `payload` is not valid UTF-8; skipping");
        return None;
    };
    match LayerNotification::parse(text) {
        Ok(n) => Some(
            n.with_relay_ompf(
                fields
                    .get(RELAY_OMPF_FIELD)
                    .map(|ompf| Bytes::copy_from_slice(ompf)),
            ),
        ),
        Err(e) => {
            tracing::warn!(stream_key = %key, entry_id = %id, payload = %text, error = %e, "unparseable stream notification; skipping");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fred::error::{Error, ErrorKind};

    #[test]
    fn should_force_reconnect_fires_first_then_every_rearm_period() {
        let fired: Vec<u32> = (0..=20).filter(|&s| should_force_reconnect(s)).collect();
        assert_eq!(fired, [1, 9, 17]);
    }

    #[test]
    fn backoff_window_doubles_until_the_cap() {
        let init = Duration::from_millis(100);
        let cap = Duration::from_secs(5);
        let windows: Vec<u64> = (1..=8).map(|f| backoff_window_ms(f, init, cap).0).collect();
        assert_eq!(windows, [100, 200, 400, 800, 1600, 3200, 5000, 5000]);
        for f in 1..=8 {
            let (w, s) = backoff_window_ms(f, init, cap);
            assert!(s <= w);
        }
        assert_eq!(backoff_window_ms(u32::MAX, init, cap).0, 5000);
    }

    /// A generic server error and `CROSSSLOT` are not retried: each ends the
    /// reader's subscriptions (`CROSSSLOT` without probing, so it must be
    /// recognised as such). The retried kinds and messages are pinned by the
    /// tests below.
    #[test]
    fn reader_retry_classification() {
        let unknown = |d: &str| Error::new(ErrorKind::Unknown, d.to_owned());
        assert!(!is_retryable_reader_error(&unknown("ERR something")));
        assert!(is_crossslot_error(&unknown(
            "CROSSSLOT Keys in request don't hash to the same slot"
        )));
        assert!(!is_retryable_reader_error(&unknown("CROSSSLOT x")));
    }

    /// A reader state with a wake pending, not yet landed, for read 2 — the
    /// read in flight.
    fn state_with_a_pending_wake_for_read_2() -> LoopState {
        LoopState {
            schema: KeySchema::default(),
            slot_tag: None,
            budget: TransientBudget::default(),
            read: 2,
            wake: Some(PendingWake {
                read: 2,
                landed: None,
            }),
            joined: false,
            last_answer: Some(Instant::now()),
            probe_due: false,
            retarget_due: false,
            keepalive: KeepaliveSchedule::default(),
            reauth: ReauthSchedule::default(),
        }
    }

    /// A wake report names the read its wake was sent for. One that arrives
    /// after that read returned — a late `WakeLanded` or `WakeFailed` for read
    /// 1 while read 2 is in flight — must not touch read 2's wake: a landing
    /// would start read 2's wake-ack deadline from someone else's wake, and a
    /// failure would drop read 2's still-outstanding wake so the next join
    /// sends a second one.
    #[test]
    fn a_wake_report_for_a_read_no_longer_in_flight_is_ignored() {
        let mut core = Core::new();
        let at = Instant::now();

        let mut st = state_with_a_pending_wake_for_read_2();
        st.apply(&mut core, Ctl::WakeLanded { read: 1, at });
        let wake = st.wake.expect("a late landing must leave read 2's wake");
        assert_eq!(wake.read, 2);
        assert_eq!(
            wake.landed, None,
            "a late landing must not arm read 2's wake-ack deadline"
        );
        st.apply(&mut core, Ctl::WakeFailed { read: 1 });
        let wake = st.wake.expect("a late failure must leave read 2's wake");
        assert_eq!((wake.read, wake.landed), (2, None));

        // The same reports for the read in flight do apply.
        st.apply(&mut core, Ctl::WakeLanded { read: 2, at });
        assert_eq!(st.wake.map(|w| w.landed), Some(Some(at)));
        let mut st = state_with_a_pending_wake_for_read_2();
        st.apply(&mut core, Ctl::WakeFailed { read: 2 });
        assert!(st.wake.is_none(), "a failure of read 2's own wake drops it");
        assert!(!st.joined, "a wake report is not a join");
    }

    /// Every `ErrorKind` this build can construct. The exhaustive `match` below
    /// fails to compile when fred adds a variant, so the loops over this list
    /// cannot silently miss one.
    fn every_fred_error_kind() -> Vec<ErrorKind> {
        fn listed(kind: &ErrorKind) {
            match kind {
                ErrorKind::Config
                | ErrorKind::Auth
                | ErrorKind::Routing
                | ErrorKind::IO
                | ErrorKind::InvalidCommand
                | ErrorKind::InvalidArgument
                | ErrorKind::Url
                | ErrorKind::Protocol
                | ErrorKind::Canceled
                | ErrorKind::Unknown
                | ErrorKind::Timeout
                | ErrorKind::Cluster
                | ErrorKind::Parse
                | ErrorKind::Sentinel
                | ErrorKind::NotFound
                | ErrorKind::Backpressure => {}
                #[cfg(feature = "valkey-tls")]
                ErrorKind::Tls => {}
            }
        }
        let kinds = vec![
            ErrorKind::Config,
            ErrorKind::Auth,
            ErrorKind::Routing,
            ErrorKind::IO,
            ErrorKind::InvalidCommand,
            ErrorKind::InvalidArgument,
            ErrorKind::Url,
            ErrorKind::Protocol,
            #[cfg(feature = "valkey-tls")]
            ErrorKind::Tls,
            ErrorKind::Canceled,
            ErrorKind::Unknown,
            ErrorKind::Timeout,
            ErrorKind::Cluster,
            ErrorKind::Parse,
            ErrorKind::Sentinel,
            ErrorKind::NotFound,
            ErrorKind::Backpressure,
        ];
        kinds.iter().for_each(listed);
        kinds
    }

    /// The reader's verdict is the shared classifier's, widened by exactly one
    /// reader-only `Unknown` message (`-UNBLOCKED`) and by nothing else: on a
    /// message that carve-out does not name, every kind gets the shared
    /// verdict.
    #[test]
    fn reader_retry_classification_agrees_with_the_shared_classifier_on_every_kind() {
        for kind in every_fred_error_kind() {
            let e = Error::new(kind.clone(), "synthetic reader-classification error");
            assert_eq!(
                is_retryable_reader_error(&e),
                is_transient_fred_error(&e),
                "{kind:?}: the reader must retry exactly what the shared classifier calls \
                 transient, apart from its own message carve-out"
            );
        }
    }

    /// The shared classifier's two router carve-out families reach the reader:
    /// a router miss and each redirect-budget-exhausted shape are retried, not
    /// treated as permanent failures that end every subscription on the reader
    /// (b.c7a, b.cv4). Its third family, the store not serving, is the next
    /// test.
    #[test]
    fn a_router_miss_and_every_budget_exhausted_shape_are_retryable_for_the_reader() {
        for kind in [ErrorKind::Unknown, ErrorKind::Routing] {
            let miss = Error::new(kind.clone(), "Failed to route command.");
            assert!(
                is_retryable_reader_error(&miss),
                "{kind:?} router miss must be retried by the reader"
            );
        }
        for (kind, message) in crate::storage::client::BUDGET_EXHAUSTED_SHAPES {
            let e = Error::new(kind.clone(), message);
            assert!(
                is_retryable_reader_error(&e),
                "{kind:?} {message:?} must be retried by the reader"
            );
        }
    }

    /// The shared classifier's third carve-out family reaches the reader too:
    /// the store's `-LOADING` (a node replaying its data) and `-BUSY` (a long
    /// script) are retried within the transient budget, not treated as
    /// permanent failures that end every subscription on the reader (b.xd1).
    /// Their near misses `-BUSYGROUP` / `-BUSYKEY` are not.
    #[test]
    fn the_stores_not_serving_replies_are_retryable_for_the_reader() {
        for reply in crate::storage::client::STORE_NOT_SERVING_SHAPES {
            let e = Error::new(ErrorKind::Unknown, reply);
            assert!(
                is_retryable_reader_error(&e),
                "{reply:?} must be retried by the reader: the store clears it by itself"
            );
            assert!(
                !is_client_unblocked_error(&e),
                "{reply:?} is retried through the shared classifier, not a reader-only arm"
            );
        }
        for reply in [
            "BUSYGROUP Consumer Group name already exists",
            "BUSYKEY Target key name already exists.",
        ] {
            assert!(
                !is_retryable_reader_error(&Error::new(ErrorKind::Unknown, reply)),
                "{reply:?} is not the store being busy"
            );
        }
    }

    /// A slot migration's `-TRYAGAIN` is retried by the reader through the
    /// shared classifier, where it is transient for every multi-key call site
    /// (b.vn4), not through a reader-only arm. `-UNBLOCKED` is retried by the
    /// reader and stays OUT of the shared taxonomy: no other pool issues a
    /// blocking read, and widening `is_transient_fred_error` would reclassify
    /// a genuine `Unknown` on the write path.
    #[test]
    fn tryagain_is_transient_in_the_shared_classifier_and_unblocked_stays_reader_only() {
        let tryagain = Error::new(ErrorKind::Unknown, crate::storage::client::TRYAGAIN_REPLY);
        assert!(
            is_transient_fred_error(&tryagain),
            "-TRYAGAIN is transient in the shared classifier"
        );
        assert!(
            is_retryable_reader_error(&tryagain),
            "…so the reader retries it"
        );
        assert!(
            !is_client_unblocked_error(&tryagain),
            "…through the shared classifier, not the reader-only arm"
        );

        let unblocked = Error::new(
            ErrorKind::Unknown,
            "UNBLOCKED client unblocked via CLIENT UNBLOCK",
        );
        assert!(
            is_retryable_reader_error(&unblocked),
            "-UNBLOCKED: the reader retries it"
        );
        assert!(
            !is_transient_fred_error(&unblocked),
            "-UNBLOCKED must not be transient for the shared classifier"
        );
    }

    /// The production values the behaviour tests override through
    /// `StreamMuxKnobs` (20 ms backoffs, a budget of 3, shortened watchdogs,
    /// shrunk backlog caps, small reply targets, short probe bounds, a
    /// millisecond keepalive, a short liveness age). Every fault test runs on
    /// the overrides, so without this a change to a production value would
    /// pass every test.
    #[test]
    fn production_reader_constants_are_the_documented_values() {
        use crate::storage::notifications::{
            CONSUMER_BACKLOG_BYTES, CONSUMER_CHANNEL_CAPACITY, CONSUMER_SEND_WATCHDOG,
            OUTER_SEND_DEADLINE, XREAD_BLOCK_MS,
        };
        use crate::storage::stream_mux::{
            BACKLOG_HARD_CAP_FACTOR, DEFAULT_ENTRY_ESTIMATE, READER_KEEPALIVE_INTERVAL,
            READER_PROBE_TIMEOUT, WAKE_ACK_TIMEOUT, XREAD_COUNT, XREAD_REPLY_TARGET_BYTES,
        };

        assert_eq!(
            CONSUMER_XREAD_MAX_ATTEMPTS, 1,
            "fred must never replay a blocking XREAD on its own after a reconnect"
        );
        assert_eq!(MAX_CONSECUTIVE_TRANSIENT_FAILURES, 60);
        assert_eq!(XREAD_TRANSIENT_BACKOFF_INITIAL, Duration::from_millis(100));
        assert_eq!(XREAD_TRANSIENT_BACKOFF_CAP, Duration::from_secs(5));
        assert_eq!(FORCE_RECONNECT_TIMEOUT, Duration::from_secs(10));
        assert_eq!(FORCE_RECONNECT_REARM_EVERY, 8);
        assert_eq!(MAX_CONSECUTIVE_UNBLOCK_RETRIES, 3);
        assert_eq!(
            WAKE_ATTEMPTS, 3,
            "`MuxFaultSite::Wake` documents three faults per failed wake"
        );
        assert_eq!(
            WAKE_ACK_TIMEOUT,
            Duration::from_secs(3),
            "the wake-ack deadline, and the liveness age past which a covered join still wakes \
             the read"
        );
        assert_eq!(
            PROBE_TIMEOUT,
            Duration::from_secs(2),
            "one bound on the poison-key `TYPE` probes, on one reply's head checks and on one \
             idle keepalive tick"
        );
        assert_eq!(
            READER_PROBE_TIMEOUT,
            Duration::from_secs(3),
            "the idle→active probe's bound: the old checkout-probe budget"
        );
        assert_eq!(
            READER_KEEPALIVE_INTERVAL,
            Duration::from_secs(200),
            "a third of Azure Managed Redis's 10-minute idle-connection timeout"
        );
        assert_eq!(XREAD_BLOCK_MS, 30_000);
        assert_eq!(CONSUMER_SEND_WATCHDOG, Duration::from_secs(5));
        assert_eq!(OUTER_SEND_DEADLINE, Duration::from_secs(5));

        // Read sizing: `COUNT = clamp(⌊1 MiB ÷ Σ est⌋, 1, 512)`, an unseen key
        // estimated at 16 KiB.
        assert_eq!(XREAD_REPLY_TARGET_BYTES, 1024 * 1024);
        assert_eq!(DEFAULT_ENTRY_ESTIMATE, 16 * 1024);
        assert_eq!(XREAD_COUNT, 512, "the ceiling of the byte-sized COUNT");

        // Per-subscription backlog: soft 4096 entries or 8 MiB, hard 4× both.
        assert_eq!(CONSUMER_CHANNEL_CAPACITY, 4096);
        assert_eq!(CONSUMER_BACKLOG_BYTES, 8 * 1024 * 1024);
        assert_eq!(BACKLOG_HARD_CAP_FACTOR, 4);
        assert_eq!(
            CONSUMER_BACKLOG_BYTES * BACKLOG_HARD_CAP_FACTOR as u64,
            32 * 1024 * 1024,
            "the hard byte cap is 32 MiB"
        );
        assert_eq!(
            CONSUMER_BACKLOG_BYTES,
            CONSUMER_CHANNEL_CAPACITY as u64 * 2048,
            "the byte thresholds are the entry thresholds × 2 KiB, so a backlog of entries up to \
             2 KiB reaches the entry limits first"
        );
    }

    /// `StreamMuxKnobs::new` — what production runs with — carries those
    /// constants, not stray literals.
    #[test]
    fn production_knobs_carry_the_production_constants() {
        use crate::storage::notifications::{
            CONSUMER_BACKLOG_BYTES, CONSUMER_CHANNEL_CAPACITY, CONSUMER_SEND_WATCHDOG,
            XREAD_BLOCK_MS,
        };
        use crate::storage::stream_mux::{
            BACKLOG_HARD_CAP_FACTOR, READER_KEEPALIVE_INTERVAL, READER_PROBE_TIMEOUT,
            WAKE_ACK_TIMEOUT, XREAD_COUNT, XREAD_REPLY_TARGET_BYTES,
        };

        let k = StreamMuxKnobs::new();
        assert_eq!(k.block_ms, XREAD_BLOCK_MS);
        assert_eq!(k.count, XREAD_COUNT);
        assert_eq!(k.reply_target_bytes, XREAD_REPLY_TARGET_BYTES);
        assert_eq!(k.channel_capacity, CONSUMER_CHANNEL_CAPACITY);
        assert_eq!(
            k.backlog_hard_cap,
            CONSUMER_CHANNEL_CAPACITY * BACKLOG_HARD_CAP_FACTOR
        );
        assert_eq!(k.backlog_hard_cap, 16_384);
        assert_eq!(k.backlog_bytes, CONSUMER_BACKLOG_BYTES);
        assert_eq!(
            k.backlog_hard_bytes,
            CONSUMER_BACKLOG_BYTES * BACKLOG_HARD_CAP_FACTOR as u64
        );
        assert_eq!(k.backlog_hard_bytes, 32 * 1024 * 1024);
        assert_eq!(k.send_watchdog, CONSUMER_SEND_WATCHDOG);
        assert_eq!(k.max_transient_failures, MAX_CONSECUTIVE_TRANSIENT_FAILURES);
        assert_eq!(k.backoff_initial, XREAD_TRANSIENT_BACKOFF_INITIAL);
        assert_eq!(k.backoff_cap, XREAD_TRANSIENT_BACKOFF_CAP);
        assert_eq!(k.wake_ack_timeout, WAKE_ACK_TIMEOUT);
        assert_eq!(k.probe_timeout, PROBE_TIMEOUT);
        assert_eq!(k.reader_probe_timeout, READER_PROBE_TIMEOUT);
        assert_eq!(
            k.keepalive_interval,
            Some(READER_KEEPALIVE_INTERVAL),
            "production readers run the idle keepalive"
        );
        assert_eq!(k.liveness_age, WAKE_ACK_TIMEOUT);
    }

    // ── The liveness wake decision ──────────────────────────────────────────

    /// A reader state with no wake outstanding, `joined` as given, and the
    /// connection's last answer as given.
    fn state_for_a_join(joined: bool, last_answer: Option<Instant>) -> LoopState {
        LoopState {
            joined,
            last_answer,
            wake: None,
            ..state_with_a_pending_wake_for_read_2()
        }
    }

    /// A reader core holding one subscription on key `a` from `0-0`, with a
    /// read over it in flight when `in_flight`.
    fn core_reading_a(in_flight: bool) -> Core {
        let knobs = StreamMuxKnobs::new();
        let mut core = Core::new();
        core.add(
            1,
            "a".to_owned(),
            StreamId { ms: 0, seq: 0 },
            ChannelSink::new(&knobs).0,
        );
        if in_flight {
            core.plan_read("w").expect("a non-empty core plans");
        }
        core
    }

    /// Adds a joining subscription to `core`: on key `a` at the read's own
    /// cursor (covered by the read in flight), or on key `b` (not covered).
    fn join(core: &mut Core, covered: bool) {
        let knobs = StreamMuxKnobs::new();
        let key = if covered { "a" } else { "b" };
        core.add(
            2,
            key.to_owned(),
            StreamId { ms: 0, seq: 0 },
            ChannelSink::new(&knobs).0,
        );
    }

    /// `take_wake_reason`, case by case: a join the read does not cover wakes
    /// it for coverage, whatever the connection's last answer; a covered join
    /// wakes it for liveness when the connection has not answered (`None`:
    /// failed since, or never) or answered longer ago than the liveness age,
    /// and not when it answered within it. There is no wake without a join, or
    /// while a wake is outstanding for the read. With a read in flight the
    /// join is taken — decided once; with none (a backoff, an inline step) it is
    /// kept for the next read's issue, and nothing is decided.
    #[test]
    fn take_wake_reason_decides_coverage_then_liveness_and_carries_a_join_made_between_reads() {
        /// The connection's last answer, relative to the liveness age.
        #[derive(Clone, Copy)]
        enum Answer {
            /// Never answered, or failed since.
            Void,
            /// Within the age.
            Fresh,
            /// Longer ago than the age.
            Stale,
        }
        use Answer::{Fresh, Stale, Void};
        const AGE: Duration = Duration::from_millis(500);
        const COVERAGE: Option<&str> = Some(WAKE_REASON_COVERAGE);
        const LIVENESS: Option<&str> = Some(WAKE_REASON_LIVENESS);
        let last_answer = |answer: Answer| match answer {
            Void => None,
            Fresh => Some(Instant::now()),
            Stale => Some(
                Instant::now()
                    .checked_sub(10 * AGE)
                    .expect("the clock is past five seconds"),
            ),
        };

        #[rustfmt::skip]
        let cases = [
            // case,                                 in flight, covered, joined, answer, wake out, → reason,  joined after
            ("uncovered, fresh answer",                  true,  false,   true,   Fresh,  false,      COVERAGE, false),
            ("coverage beats liveness",                  true,  false,   true,   Void,   false,      COVERAGE, false),
            ("covered, never answered or failed since",  true,  true,    true,   Void,   false,      LIVENESS, false),
            ("covered, answer older than the age",       true,  true,    true,   Stale,  false,      LIVENESS, false),
            ("covered, answer within the age",           true,  true,    true,   Fresh,  false,      None,     false),
            ("no join",                                  true,  false,   false,  Void,   false,      None,     false),
            ("uncovered, a wake outstanding",            true,  false,   true,   Void,   true,       None,     false),
            ("covered, a wake outstanding",              true,  true,    true,   Void,   true,       None,     false),
            ("no read in flight, never answered",        false, true,    true,   Void,   false,      None,     true),
            ("no read in flight, uncovered",             false, false,   true,   Void,   false,      None,     true),
        ];
        for (case, in_flight, covered, joined, answer, outstanding, want, joined_after) in cases {
            let mut core = core_reading_a(in_flight);
            join(&mut core, covered);
            let mut st = state_for_a_join(joined, last_answer(answer));
            if outstanding {
                st.wake = Some(PendingWake {
                    read: st.read,
                    landed: None,
                });
            }
            assert_eq!(
                st.take_wake_reason(&core, in_flight, AGE),
                want,
                "{case}: the wake decision"
            );
            assert_eq!(st.joined, joined_after, "{case}: `joined` afterwards");
        }
    }

    /// A join carried across a backoff is decided when the next read is
    /// issued: kept while no read is in flight, then — the new read covering
    /// it, the connection not having answered since the failure — a liveness
    /// wake, taken once.
    #[test]
    fn a_join_kept_between_reads_is_decided_once_as_the_next_read_is_issued() {
        let age = crate::storage::stream_mux::WAKE_ACK_TIMEOUT;
        let mut core = core_reading_a(false);
        join(&mut core, true);
        let mut st = state_for_a_join(true, None);
        assert_eq!(st.take_wake_reason(&core, false, age), None);
        assert!(st.joined, "no read in flight: the join is kept");

        core.plan_read("w").expect("a non-empty core plans");
        assert!(
            !core.needs_wake(),
            "fixture self-check: the new read covers the join"
        );
        assert_eq!(
            st.take_wake_reason(&core, true, age),
            Some(WAKE_REASON_LIVENESS),
            "the issued read gets the carried join's liveness wake"
        );
        assert!(!st.joined);
        assert_eq!(
            st.take_wake_reason(&core, true, age),
            None,
            "the join was decided once"
        );
    }

    // ── The cluster slot-rebind probe trigger ───────────────────────────────

    /// A reader context for [`LoopState::note_slot_rebind`], which reads only
    /// `cluster` from it. Nothing here connects or issues a command.
    fn rebind_ctx(cluster: bool) -> ReaderCtx {
        let pool = fred::types::Builder::default_centralized()
            .build_pool(1)
            .expect("an unconnected pool builds");
        let (ctl, _) = mpsc::unbounded_channel();
        let routing = if cluster {
            super::super::router::Routing::SlotAffinity
        } else {
            super::super::router::Routing::Hash
        };
        let shared = Arc::new(MuxShared {
            router: std::sync::Mutex::new(super::super::RouterState {
                core: super::super::router::RouterCore::new(routing, 1),
                ctl: Vec::new(),
                closed: false,
            }),
            freed: tokio::sync::Notify::new(),
            next_sub: std::sync::atomic::AtomicU64::new(1),
            knobs: std::sync::Mutex::new(StreamMuxKnobs::new()),
            metrics: Arc::new(std::sync::Mutex::new(None)),
            stats: Vec::new(),
            conns: vec![std::sync::Mutex::new(super::super::ReaderSlot::default())],
            faults: std::sync::Mutex::new(std::collections::HashMap::new()),
            reconnects: Vec::new(),
            knobs_changed: super::super::TestSignal::default(),
        });
        ReaderCtx {
            idx: 0,
            connector: ReaderConnector::unconnected_for_test(pool.clone(), cluster),
            data: pool,
            cluster,
            instance: Arc::from("rebind-test"),
            shared,
            cancel: CancellationToken::new(),
            ctl,
        }
    }

    /// A reader core whose next read is over `key` alone.
    fn core_on(key: &str) -> Core {
        let mut core = Core::new();
        core.add(
            1,
            key.to_owned(),
            StreamId { ms: 0, seq: 0 },
            ChannelSink::new(&StreamMuxKnobs::new()).0,
        );
        core
    }

    fn slot(key: &str) -> u16 {
        fred::util::redis_keyslot(key.as_bytes())
    }

    /// A reader state whose previous read was bound to `slot` (`None`: it has
    /// not read yet), with no probe due and a connection that just answered.
    fn state_bound_to(slot: Option<u16>) -> LoopState {
        LoopState {
            slot_tag: slot.map(|s| (s, super::super::router::tag_for_slot(s))),
            ..state_with_a_pending_wake_for_read_2()
        }
    }

    /// Two keys in different hash slots, and a third in the first one's slot.
    /// The hash tags pin the slots: `{a}` hashes to 15495, `{b}` to 3300.
    const SLOT_A_KEY: &str = "le:{a}:stream";
    const SLOT_B_KEY: &str = "le:{b}:stream";
    const SLOT_A_OTHER_KEY: &str = "le:{a}:other";

    /// On a cluster, a read bound to another hash slot than the reader's
    /// previous read is a rebind: the probe becomes due and the last answer —
    /// the old slot's node's — no longer vouches for the connection. The new
    /// slot is recorded, so the pass after the probe answers plans the read
    /// rather than probing again.
    #[test]
    fn a_cluster_read_bound_to_another_slot_makes_the_probe_due_and_forgets_the_last_answer() {
        assert_ne!(
            slot(SLOT_A_KEY),
            slot(SLOT_B_KEY),
            "fixture self-check: two slots"
        );
        let ctx = rebind_ctx(true);
        let core = core_on(SLOT_B_KEY);
        let mut st = state_bound_to(Some(slot(SLOT_A_KEY)));

        assert!(st.note_slot_rebind(&ctx, &core), "a rebind");
        assert!(st.probe_due, "the first read on the new slot is probed");
        assert!(
            st.last_answer.is_none(),
            "the old node's answer no longer vouches for the connection"
        );
        assert_eq!(
            st.slot_tag.as_ref().map(|(s, _)| *s),
            Some(slot(SLOT_B_KEY))
        );

        st.probe_due = false; // The probe answered.
        assert!(
            !st.note_slot_rebind(&ctx, &core),
            "the next pass on the same slot is no rebind: one probe per rebind"
        );
        assert!(!st.probe_due);
    }

    /// Negative controls: no rebind, so no probe due and the last answer kept,
    /// for a cluster read on the slot the previous read was bound to (another
    /// key in it), for a cluster reader's first binding (its probe is due from
    /// its start already), and on standalone Valkey whatever the keys.
    #[test]
    fn no_rebind_on_the_same_slot_on_a_first_binding_or_on_standalone() {
        assert_eq!(
            slot(SLOT_A_KEY),
            slot(SLOT_A_OTHER_KEY),
            "fixture self-check: one slot"
        );
        #[rustfmt::skip]
        let cases = [
            // case,                       cluster, bound to,               next read on,     → slot recorded
            ("cluster, same slot",         true,    Some(slot(SLOT_A_KEY)), SLOT_A_OTHER_KEY, Some(slot(SLOT_A_KEY))),
            ("cluster, first binding",     true,    None,                   SLOT_B_KEY,       Some(slot(SLOT_B_KEY))),
            ("standalone, another key",    false,   None,                   SLOT_B_KEY,       None),
        ];
        for (case, cluster, bound_to, next, recorded) in cases {
            let ctx = rebind_ctx(cluster);
            let mut st = state_bound_to(bound_to);
            let answered = st.last_answer;
            assert!(
                !st.note_slot_rebind(&ctx, &core_on(next)),
                "{case}: no rebind"
            );
            assert!(!st.probe_due, "{case}: no probe due");
            assert_eq!(st.last_answer, answered, "{case}: the last answer is kept");
            assert_eq!(
                st.slot_tag.as_ref().map(|(s, _)| *s),
                recorded,
                "{case}: the slot recorded"
            );
        }

        // Standalone stays unbound across a change of key too.
        let ctx = rebind_ctx(false);
        let mut st = state_bound_to(None);
        assert!(!st.note_slot_rebind(&ctx, &core_on(SLOT_A_KEY)));
        assert!(!st.note_slot_rebind(&ctx, &core_on(SLOT_B_KEY)));
        assert!(!st.probe_due && st.last_answer.is_some() && st.slot_tag.is_none());
    }

    // ── The idle keepalive's schedule ───────────────────────────────────────

    /// A new schedule ticks at a random offset in `(0, interval]`; `None`
    /// turns it off and forgets the phase; a zero interval is read as 1 ms.
    #[test]
    fn a_new_keepalive_schedule_ticks_within_one_interval_and_none_turns_it_off() {
        let interval = Duration::from_secs(200);
        let now = Instant::now();
        for _ in 0..64 {
            let mut schedule = KeepaliveSchedule::default();
            let tick = schedule
                .next(Some(interval), now)
                .expect("an interval ticks");
            assert!(tick > now && tick <= now + interval, "{:?}", tick - now);
        }

        let mut schedule = KeepaliveSchedule::default();
        schedule.next(Some(interval), now);
        assert_eq!(
            schedule.next(None, now),
            None,
            "`None` turns the keepalive off"
        );
        assert!(schedule.tick.is_none(), "and forgets the phase");

        let tick = KeepaliveSchedule::default()
            .next(Some(Duration::ZERO), now)
            .expect("a zero interval still ticks");
        assert!(
            tick > now && tick <= now + Duration::from_millis(1),
            "a zero interval is read as 1 ms: {:?}",
            tick - now
        );
    }

    /// The schedule keeps its phase: a tick still ahead is unchanged, and
    /// ticks that fell while the reader was active are skipped in whole
    /// intervals, so the next is after `now` and within one interval of it. A
    /// changed interval starts a new phase within the new interval.
    #[test]
    fn a_keepalive_schedule_keeps_its_phase_and_a_changed_interval_starts_a_new_one() {
        let interval = Duration::from_millis(100);
        let start = Instant::now();
        let mut schedule = KeepaliveSchedule::default();
        let first = schedule.next(Some(interval), start).expect("ticks");

        assert_eq!(
            schedule.next(Some(interval), start),
            Some(first),
            "a tick still ahead is kept"
        );
        let later = first + Duration::from_millis(250);
        let next = schedule.next(Some(interval), later).expect("ticks");
        assert_eq!(
            next,
            first + 3 * interval,
            "two and a half intervals past the tick: the next on the same phase"
        );
        assert_eq!(
            schedule.next(Some(interval), first + 3 * interval),
            Some(first + 4 * interval),
            "a tick at exactly `now` is past: the next is never at or before `now`"
        );

        let changed = Duration::from_secs(10);
        let now = first + 5 * interval;
        let tick = schedule.next(Some(changed), now).expect("ticks");
        assert!(tick > now && tick <= now + changed, "{:?}", tick - now);
    }

    /// A head check reads the first entry's id: none on an empty stream, and an
    /// id that does not parse is an error — never a gap claim, never "no gap".
    #[test]
    fn a_head_checks_first_entry_id_parses_or_is_an_error() {
        assert_eq!(first_entry_id(None).unwrap(), None);
        assert_eq!(
            first_entry_id(Some("12-0")).unwrap(),
            Some(StreamId { ms: 12, seq: 0 })
        );
        let e = first_entry_id(Some("not-an-id")).expect_err("an unparseable id");
        assert_eq!(*e.kind(), ErrorKind::Parse);
    }

    #[test]
    fn record_success_clears_every_streak() {
        let mut b = TransientBudget {
            failures: 3,
            timeout_streak: 2,
            unblock_streak: 1,
            moved_streak: 4,
        };
        b.record_success();
        assert_eq!(
            (
                b.failures,
                b.timeout_streak,
                b.unblock_streak,
                b.moved_streak
            ),
            (0, 0, 0, 0),
            "a successful read clears every streak, the -MOVED retargets' included"
        );
    }

    // ── The consecutive-`-MOVED` retarget streak (`charge_moved_retarget`) ──

    /// A one-subscription core on [`SLOT_A_KEY`], and its sink's terminal
    /// slot, which records how the subscription ended.
    fn core_with_terminal() -> (Core, super::super::TerminalSlot) {
        let (sink, _rx, _backlog, terminal) = ChannelSink::new(&StreamMuxKnobs::new());
        let mut core = Core::new();
        core.add(1, SLOT_A_KEY.to_owned(), StreamId { ms: 0, seq: 0 }, sink);
        (core, terminal)
    }

    /// Knobs allowing `max` transient failures.
    fn knobs_allowing(max: u32) -> StreamMuxKnobs {
        StreamMuxKnobs {
            max_transient_failures: max,
            ..StreamMuxKnobs::new()
        }
    }

    /// The first `-MOVED` retarget since a successful read is free: the probe
    /// becomes due (the next connection step retargets), with no budget
    /// charge and no backoff. Every later one before a read succeeds is
    /// charged one transient failure and backed off. A successful read
    /// resets the streak, so the next retarget is free again.
    #[tokio::test]
    async fn the_first_moved_retarget_is_free_and_each_later_one_before_a_read_is_charged() {
        let ctx = rebind_ctx(true);
        let (mut core, _terminal) = core_with_terminal();
        let knobs = knobs_allowing(10);
        let mut st = state_bound_to(Some(slot(SLOT_A_KEY)));

        let backoff = charge_moved_retarget(&ctx, &mut core, &mut st, &knobs, "an XREAD");
        assert!(backoff.is_none(), "the first retarget is not backed off");
        assert!(st.probe_due, "the next connection step retargets");
        assert_eq!((st.budget.moved_streak, st.budget.failures), (1, 0));

        st.probe_due = false;
        for n in 2..=3u32 {
            let backoff = charge_moved_retarget(&ctx, &mut core, &mut st, &knobs, "a head check");
            assert!(
                backoff.is_some(),
                "retarget {n} of a streak is backed off before it retargets"
            );
            assert!(st.probe_due, "retarget {n}: the retarget still follows");
            assert_eq!(
                (st.budget.moved_streak, st.budget.failures),
                (n, n - 1),
                "retarget {n} of a streak is charged one transient failure"
            );
        }
        assert_eq!(
            core.subscription_count(),
            1,
            "inside the budget nothing ends"
        );

        st.budget.record_success();
        let backoff = charge_moved_retarget(&ctx, &mut core, &mut st, &knobs, "a probe");
        assert!(
            backoff.is_none(),
            "after a successful read the next retarget is the first of a new streak: free"
        );
        assert_eq!((st.budget.moved_streak, st.budget.failures), (1, 0));
    }

    /// Past the transient budget, a charged retarget ends every subscription
    /// on the reader `Unavailable`, naming the redirect and the exhausted
    /// budget, and resets the budget — the streak with it — so the next
    /// retarget is free. A retarget the budget still covers ends nothing.
    #[tokio::test]
    async fn a_moved_retarget_streak_past_the_budget_ends_the_subscriptions_and_resets_it() {
        let ctx = rebind_ctx(true);
        let (mut core, terminal) = core_with_terminal();
        let knobs = knobs_allowing(1);
        let mut st = state_bound_to(Some(slot(SLOT_A_KEY)));

        assert!(charge_moved_retarget(&ctx, &mut core, &mut st, &knobs, "a probe").is_none());
        assert!(
            charge_moved_retarget(&ctx, &mut core, &mut st, &knobs, "a probe").is_some(),
            "the second retarget is charged the budget's one allowed failure, and backed off"
        );
        assert_eq!(core.subscription_count(), 1);

        let backoff = charge_moved_retarget(&ctx, &mut core, &mut st, &knobs, "a connection probe");
        assert!(
            backoff.is_none(),
            "the subscriptions ended: nothing to back off for"
        );
        assert_eq!(
            core.subscription_count(),
            0,
            "past the budget every subscription on the reader ends"
        );
        let exit = terminal
            .lock()
            .expect("never poisoned")
            .take()
            .expect("the subscription ended with a terminal exit");
        assert_eq!(exit.reason, STREAM_EXIT_ERROR);
        match exit.error {
            crate::error::ServiceError::Unavailable(message) => assert!(
                message.contains(
                    "gave up after 2 consecutive transient failures (last: a connection probe was \
                     redirected (MOVED) to a new hash slot owner again before any read succeeded)"
                ),
                "the ending names the exhausted budget and the repeated redirect; got {message:?}"
            ),
            other => panic!("a reader-side ending is a retryable Unavailable; got {other:?}"),
        }
        assert_eq!(
            (
                st.budget.failures,
                st.budget.moved_streak,
                st.budget.timeout_streak,
                st.budget.unblock_streak
            ),
            (0, 0, 0, 0),
            "the exhausted budget resets, the streak with it"
        );
        assert!(st.probe_due, "the retarget is still due");
    }

    #[test]
    fn parse_reply_marks_wake_entries_and_bad_payloads_undeliverable() {
        let mut reply: XReadResponse<String, String, String, Bytes> = Default::default();
        let f = |pairs: &[(&str, &[u8])]| {
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), Bytes::copy_from_slice(v)))
                .collect()
        };
        reply.insert(
            "s".into(),
            vec![
                (
                    "1-0".into(),
                    f(&[("payload", b"update|7|1|c|1|2"), ("ompf", b"\x00\x01")]),
                ),
                ("2-0".into(), f(&[("payload", b"garbage")])),
                ("3-0".into(), f(&[("other", b"x")])),
                ("bad".into(), f(&[("payload", b"deleted")])),
            ],
        );
        reply.insert("w".into(), vec![("9-1".into(), f(&[("w", b"1")]))]);
        let mut parsed = parse_reply(reply, Some("w"));
        parsed.sort_by(|a, b| a.0.cmp(&b.0));
        let (_, s) = &parsed[0];
        assert_eq!(s.len(), 3, "an unparseable id is dropped");
        match &s[0].payload {
            Some(LayerNotification::Update {
                object_id: 7,
                relay_ompf: Some(b),
                ..
            }) => assert_eq!(&b[..], b"\x00\x01"),
            other => panic!("{other:?}"),
        }
        assert!(s[1].payload.is_none() && s[2].payload.is_none());
        let (_, w) = &parsed[1];
        assert_eq!(w[0].id, StreamId { ms: 9, seq: 1 });
        assert!(w[0].payload.is_none());
    }

    /// Field map of one stream entry, each value copied into its own buffer.
    fn fields(pairs: &[(&str, &[u8])]) -> std::collections::HashMap<String, Bytes> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), Bytes::copy_from_slice(v)))
            .collect()
    }

    /// An entry's size is its id plus its `payload` and `ompf` values; no other
    /// field counts, and a wake entry or an undeliverable one is still measured.
    #[test]
    fn parse_reply_measures_each_entry_as_id_plus_payload_plus_ompf() {
        let mut reply: XReadResponse<String, String, String, Bytes> = Default::default();
        reply.insert(
            "s".into(),
            vec![
                (
                    "1-0".into(),
                    fields(&[
                        ("payload", b"update|7|1|c|1|2"),
                        ("ompf", &[0u8; 100]),
                        ("unrelated", &[0u8; 1000]),
                    ]),
                ),
                ("12-0".into(), fields(&[("payload", b"garbage")])),
            ],
        );
        reply.insert("w".into(), vec![("9-1".into(), fields(&[("w", b"1")]))]);
        let mut parsed = parse_reply(reply, Some("w"));
        parsed.sort_by(|a, b| a.0.cmp(&b.0));
        let sizes = |key: usize| parsed[key].1.iter().map(|e| e.bytes).collect::<Vec<_>>();
        assert_eq!(
            sizes(0),
            [3 + 16 + 100, 4 + 7],
            "id + payload + ompf; other fields ignored"
        );
        assert_eq!(
            sizes(1),
            [3],
            "a wake entry: its id, no payload or ompf field"
        );
    }

    /// One key's raw entries: `(id, fields)`.
    type RawEntries<'a> = Vec<(&'a str, std::collections::HashMap<String, Bytes>)>;

    /// Parses a reply into the batch the reader fans out.
    fn batch(entries: Vec<(&str, RawEntries<'_>)>) -> Vec<(String, Vec<Entry<LayerNotification>>)> {
        let mut reply: XReadResponse<String, String, String, Bytes> = Default::default();
        for (key, key_entries) in entries {
            reply.insert(
                key.to_owned(),
                key_entries
                    .into_iter()
                    .map(|(id, f)| (id.to_owned(), f))
                    .collect(),
            );
        }
        parse_reply(reply, Some("w"))
    }

    /// A reply that carried no layer-stream entry records no reply-bytes sample
    /// — not a 0-byte one: a reply holding only the reader's wake entry (which
    /// every join that needs a wake produces), an empty batch, a layer key with
    /// no entries, and a layer key whose only entries have an unparseable id.
    #[test]
    fn a_reply_without_a_layer_stream_entry_records_no_reply_bytes_sample() {
        let wake = || vec![("9-1", fields(&[("w", b"1")]))];
        let cases = [
            ("wake-only", batch(vec![("w", wake())])),
            ("empty", Vec::new()),
            ("layer key, no entries", batch(vec![("s", vec![])])),
            (
                "unparseable ids only",
                batch(vec![
                    ("s", vec![("bad", fields(&[("payload", b"deleted")]))]),
                    ("w", wake()),
                ]),
            ),
        ];
        for (what, b) in cases {
            assert_eq!(reply_bytes(&b, Some("w")), None, "{what}");
        }
    }

    /// A reply that carried layer-stream entries records their bytes, the wake
    /// key's excluded, undeliverable entries (a parseable id with a bad
    /// payload) included — every entry of a layer key the reader measured.
    #[test]
    fn a_reply_with_layer_stream_entries_records_their_bytes_without_the_wake_key() {
        let b = batch(vec![
            (
                "s",
                vec![(
                    "1-0",
                    fields(&[("payload", b"update|7|1|c|1|2"), ("ompf", &[0u8; 100])]),
                )],
            ),
            ("t", vec![("2-0", fields(&[("payload", b"garbage")]))]),
            (
                "w",
                vec![(
                    "9-1",
                    fields(&[("w", &[0u8; 5000]), ("payload", &[0u8; 500])]),
                )],
            ),
        ]);
        assert_eq!(
            reply_bytes(&b, Some("w")),
            Some((3 + 16 + 100) + (3 + 7)),
            "the wake key's entry is excluded, the undeliverable entry is not"
        );
        assert_eq!(
            reply_bytes(&b, None),
            Some((3 + 16 + 100) + (3 + 7) + (3 + 500)),
            "with no wake key named, every key counts"
        );
    }

    /// The relay OMPF is copied out of the reply: fred decodes zero-copy, so a
    /// field `Bytes` is a slice of the whole multi-key reply, and keeping the
    /// slice would keep every other layer's entries in that reply alive for as
    /// long as one subscriber has this entry queued.
    #[test]
    fn parse_payload_copies_the_relay_ompf_out_of_the_reply_buffer() {
        let payload: &[u8] = b"update|7|1|c|1|2";
        let ompf: &[u8] = &[0xab; 64];
        let reply_buffer = Bytes::from([payload, ompf, &[0xcd; 4096][..]].concat());
        let buffer_range = reply_buffer.as_ptr_range();
        let mut f = std::collections::HashMap::new();
        f.insert(
            NOTIFICATION_PAYLOAD_FIELD.to_owned(),
            reply_buffer.slice(..payload.len()),
        );
        f.insert(
            RELAY_OMPF_FIELD.to_owned(),
            reply_buffer.slice(payload.len()..payload.len() + ompf.len()),
        );
        assert!(
            buffer_range.contains(&f[RELAY_OMPF_FIELD].as_ptr()),
            "fixture self-check: the input field is a slice of the reply buffer, as fred's is"
        );

        let relay = match parse_payload("s", "1-0", &f) {
            Some(LayerNotification::Update {
                relay_ompf: Some(relay),
                ..
            }) => relay,
            other => panic!("expected an update carrying its relay OMPF, got {other:?}"),
        };
        assert_eq!(&relay[..], ompf, "the relay bytes are unchanged");
        let relay_range = relay.as_ptr_range();
        assert!(
            relay_range.end <= buffer_range.start || relay_range.start >= buffer_range.end,
            "the relay OMPF must own its bytes, not point into the reply buffer: relay \
             {relay_range:?}, reply {buffer_range:?}"
        );
    }
}
