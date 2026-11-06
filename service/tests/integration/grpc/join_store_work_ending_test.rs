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

//! A `JoinSession`'s own store work ends at its stream's ending, and its
//! teardown's unregister gives up within one command timeout (kl-retire F4,
//! F5; SDD §13.1, §13.2, §22.21): the disaster-recovery hold's after-entry
//! bound counts a deadline-exempt stream's work at entry, and its teardown as
//! giving up within `5T + 1 s`, then fred's 45 s write wait
//! (`dr::hold::join_teardown_write_bound`, `5T + 1 s + 45 s`), which holds
//! only if nothing a join awaits runs on past the ending.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`a_join_parked_before_its_register_when_the_hold_ends_it_never_registers`] | T12: a join parked before its register (SR-4 step 4) when the hold ends its stream ends `SERVICE_HELD`, sends no register (no `PeerJoined` is ever published) and is torn down |
//! | [`a_join_holding_its_register_when_the_hold_ends_it_tears_down_without_the_re_add`] | T12: a join whose register is in flight when the hold ends its stream stops waiting for it: its teardown's unregister is sent while the register's reply is still held, the stream ends `SERVICE_HELD`, and no step-5 index re-add follows |
//! | [`a_join_whose_peer_message_publish_is_held_ends_at_the_hold`] | T12: a join whose `PeerMessage` publish is in flight when the hold ends its stream ends `SERVICE_HELD` without waiting for the publish: no command timeout is counted for it |
//! | [`a_join_whose_first_index_add_is_held_ends_at_the_hold`] | T12: step 2's index `SADD` in flight at the ending is given up: the join ends `SERVICE_HELD`, not at the `SADD`'s timeout |
//! | [`a_join_ended_before_its_clock_read_sends_no_time`] | T12: a join ended before step 4's clock read sends no `TIME` |
//! | [`a_join_ended_before_its_snapshot_sends_no_smembers`] | T12: a join ended before step 4's membership snapshot sends no `SMEMBERS` |
//! | [`a_join_whose_index_re_add_is_held_ends_at_the_hold`] | T12: step 5's index re-add in flight at the ending is given up: the join ends `SERVICE_HELD` and tears down |
//! | [`a_teardown_unregister_answered_noscript_is_cut_at_the_command_timeout`] | T11: a teardown whose unregister is answered `NOSCRIPT` and whose reload's `SCRIPT LOAD` never answers gives the unregister up at one command timeout, reload included, and finishes: no command timeout is counted (fred's own timer on the `SCRIPT LOAD` never fired), the member is left for its lease to lapse and the index alone |
//!
//! # Staging
//!
//! The held and refused replies are staged with a [`ScriptReplyProxy`] in
//! front of the test store, which carries every connection of the fixture's
//! client ([`Fx`]): `NOSCRIPT` can be forced only so, since the shared
//! store's script cache is server-wide and never flushed. The proxy's rule
//! ([`Staging`]) is armed once the test knows its keys, and acts on the first
//! frame the step under test sends. A held reply holds every reply on its
//! connection, and a held frame every frame behind it, so the fixture's data
//! pool has [`POOL_SIZE`] connections: the join takes its commands round
//! robin, and its teardown's few commands after the held one land on other
//! connections. Every store read the tests make goes through the maintenance
//! pool (`helpers::store_reader`), which never moves the data pool's cursor.
//! The fast unresponsive verdict is moved out of reach, so no verdict closes
//! a held connection under a test.
//!
//! The first test needs no proxy: it parks the join on
//! `JoinRegistrationSeam::before_register` and reads the presence channel
//! through a [`Tap`]. The clock-read and snapshot tests park the join on a
//! seam just before the await under test, end its stream, and then count
//! the await's frames at the proxy ([`Staged::Count`]) as they release it.
//!
//! The whole file compiles only with `--features test-fault-injection` (the
//! registration seam): without it, every test here is compiled out and a run
//! reports green having covered nothing. CI's test job passes it.
//!
//! # Waits
//!
//! Every wait is a deadline: [`SETTLE`] (10 s) per staged frame reaching the
//! proxy (20 s, `MONITOR_DRAIN_BUDGET`, for a held reply's arrival,
//! `ReplyGate::reply_held`), the helpers' `PRESENCE_WAIT` (10 s) per park,
//! tracker drain or
//! channel drain, and [`STREAM_END`] (30 s) per stream end — long enough that
//! a join which waited out a held command's 5 s command timeout (the old
//! behaviour) still ends inside it, so it fails on its assertion rather than
//! on the wait. The heaviest test, the held register, waits at most a park,
//! a held reply, a staged frame, a stream end and a tracker drain, 80 s at
//! every budget's full value, inside the helpers' 100 s
//! `SCENARIO_BUDGET`
//! each test runs under. A healthy run takes milliseconds, except the
//! unregister test, which waits one command timeout (5 s) by design.

#![cfg(feature = "test-fault-injection")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fred::interfaces::ClientLike;
use tonic::Status;

use live_edit::grpc::graceful_shutdown::HoldCause;
use live_edit::grpc::service::LiveEditService;
use live_edit::presence::PresenceEvent;
use live_edit::storage::client::{
    with_liveness_overrides_for_test, LivenessOverrides, POOL_LABEL_DATA,
};
use live_edit::storage::test_support::{command_timeout_row, command_timeouts_only_on, TestValkey};

use super::helpers::*;
use crate::common::refusal_detail::assert_service_held;
use crate::storage::snapshot_read_support::{
    error_reply_script, test_valkey_addr, ProxyAction, ProxyRule, ReplyGate, ScriptReplyProxy,
};

/// Bound on each wait for a staged frame to reach the proxy.
const SETTLE: Duration = Duration::from_secs(10);

/// Bound on each wait for a join's stream to end; see the module's *Waits*.
const STREAM_END: Duration = Duration::from_secs(30);

/// The fixture's data-pool connections; see the module's *Staging*.
const POOL_SIZE: u32 = 8;

/// The fixture's command timeout T (`TestValkey`'s 5 s).
const COMMAND_TIMEOUT: Duration = Duration::from_secs(5);

/// How long the proxy holds the unregister's `NOSCRIPT`: half of T, so the
/// stalled `SCRIPT LOAD`'s own timer would fire half a T after the cut.
const NOSCRIPT_DELAY: Duration = Duration::from_millis(2_500);

/// What the proxy answers a staged unregister with.
const NOSCRIPT_REPLY: &str = "NOSCRIPT No matching script. Please use EVAL.";

/// A keeper refresh interval under which no keeper step runs inside a test.
const QUIET_REFRESH_SECS: u64 = 3600;

/// The incident every hold ending here names.
const INCIDENT: &str = "20261016t0930-b7e2";

/// Interval between probes of a polled condition.
const POLL: Duration = Duration::from_millis(10);

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// The step a test stages at the proxy.
#[derive(Clone)]
enum Staged {
    /// Hold every reply on the connection of the first presence script
    /// naming `members_key` — the join's register — and count the step-5
    /// re-add (`SADD index_key session_id`).
    HoldRegister {
        members_key: String,
        index_key: String,
        session_id: String,
        gate: ReplyGate,
    },
    /// Hold the first `PUBLISH` on `channel` — the join's `PeerMessage` — and
    /// every frame behind it on its connection.
    HoldPublish { channel: String, gate: ReplyGate },
    /// Answer the first presence script naming `members_key` — the
    /// teardown's unregister — `-NOSCRIPT` after [`NOSCRIPT_DELAY`], and hold
    /// the reload's `SCRIPT LOAD`, and every frame behind it, behind `gate`.
    NoscriptUnregister {
        members_key: String,
        gate: ReplyGate,
    },
    /// Hold every reply on the connection of the first `SADD index_key
    /// session_id` — the join's step-2 index add, or its step-5 re-add when
    /// armed after the register — and count every such `SADD`.
    HoldIndexAdd {
        index_key: String,
        session_id: String,
        gate: ReplyGate,
    },
    /// Count every frame of `command` (upper case) whose arguments include
    /// `naming`, or every frame of it when `naming` is `None`.
    Count {
        command: &'static [u8],
        naming: Option<String>,
    },
}

/// The proxy rule's state: the armed step and what it matched.
#[derive(Default)]
struct Staging {
    step: Mutex<Option<Staged>>,
    /// Presence scripts naming the staged members key, since arming.
    scripts: AtomicUsize,
    /// Step-5 index re-adds, since arming ([`Staged::HoldRegister`]).
    index_adds: AtomicUsize,
    /// `PUBLISH`es on the staged channel, since arming.
    publishes: AtomicUsize,
    /// `SCRIPT` frames after the staged script ([`Staged::NoscriptUnregister`]).
    reloads: AtomicUsize,
    /// Frames [`Staged::Count`] matched.
    counted: AtomicUsize,
}

impl Staging {
    fn arm(&self, step: Staged) {
        *self.step.lock().unwrap() = Some(step);
    }

    fn scripts(&self) -> usize {
        self.scripts.load(Ordering::SeqCst)
    }

    fn index_adds(&self) -> usize {
        self.index_adds.load(Ordering::SeqCst)
    }

    fn publishes(&self) -> usize {
        self.publishes.load(Ordering::SeqCst)
    }

    fn reloads(&self) -> usize {
        self.reloads.load(Ordering::SeqCst)
    }

    fn counted(&self) -> usize {
        self.counted.load(Ordering::SeqCst)
    }

    /// The proxy rule acting on the armed step.
    fn rule(self: &Arc<Self>) -> ProxyRule {
        let staging = Arc::clone(self);
        let noscript = error_reply_script(NOSCRIPT_REPLY);
        Arc::new(move |args: &[Vec<u8>]| {
            let step = staging.step.lock().unwrap().clone()?;
            let name = args.first()?.to_ascii_uppercase();
            let arg_is = |i: usize, want: &str| args.get(i).is_some_and(|a| a == want.as_bytes());
            let names = |want: &str| args.iter().any(|a| a == want.as_bytes());
            match step {
                Staged::HoldRegister {
                    members_key,
                    index_key,
                    session_id,
                    gate,
                } => {
                    if name == b"SADD" && arg_is(1, &index_key) && arg_is(2, &session_id) {
                        staging.index_adds.fetch_add(1, Ordering::SeqCst);
                        return None;
                    }
                    if name == b"EVALSHA" && names(&members_key) {
                        let n = staging.scripts.fetch_add(1, Ordering::SeqCst);
                        return (n == 0).then(|| ProxyAction::HoldReplies(gate));
                    }
                    None
                }
                Staged::HoldPublish { channel, gate } => {
                    if name == b"PUBLISH" && arg_is(1, &channel) {
                        let n = staging.publishes.fetch_add(1, Ordering::SeqCst);
                        return (n == 0)
                            .then(|| ProxyAction::GatedEval(gate, "return 0".to_owned()));
                    }
                    None
                }
                Staged::NoscriptUnregister { members_key, gate } => {
                    if name == b"SCRIPT" && staging.scripts() > 0 {
                        staging.reloads.fetch_add(1, Ordering::SeqCst);
                        return Some(ProxyAction::GatedEval(gate, "return 'held'".to_owned()));
                    }
                    if name == b"EVALSHA" && names(&members_key) {
                        let n = staging.scripts.fetch_add(1, Ordering::SeqCst);
                        return (n == 0)
                            .then(|| ProxyAction::DelayedEval(NOSCRIPT_DELAY, noscript.clone()));
                    }
                    None
                }
                Staged::HoldIndexAdd {
                    index_key,
                    session_id,
                    gate,
                } => {
                    if name == b"SADD" && arg_is(1, &index_key) && arg_is(2, &session_id) {
                        let n = staging.index_adds.fetch_add(1, Ordering::SeqCst);
                        return (n == 0).then(|| ProxyAction::HoldReplies(gate));
                    }
                    None
                }
                Staged::Count { command, naming } => {
                    if name == command && naming.as_deref().is_none_or(names) {
                        staging.counted.fetch_add(1, Ordering::SeqCst);
                    }
                    None
                }
            }
        })
    }
}

/// A `TestValkey` whose every connection runs through a [`ScriptReplyProxy`]
/// acting on [`Staging`]. `tv` is declared first so it drops, and cleans up
/// through the proxy, before the proxy does.
struct Fx {
    tv: TestValkey,
    _proxy: ScriptReplyProxy,
    staging: Arc<Staging>,
}

impl Fx {
    async fn new() -> Self {
        let staging = Arc::new(Staging::default());
        let proxy = ScriptReplyProxy::start(test_valkey_addr(), staging.rule()).await;
        let tv = with_liveness_overrides_for_test(
            LivenessOverrides {
                fast_timeout: Some(Duration::from_secs(60)),
                ..LivenessOverrides::default()
            },
            TestValkey::setup_with_url(proxy.url(), POOL_SIZE),
        )
        .await;
        assert_eq!(
            tv.client()
                .pool
                .next()
                .perf_config()
                .default_command_timeout,
            COMMAND_TIMEOUT,
            "precondition: the fixture's command timeout"
        );
        Self {
            tv,
            _proxy: proxy,
            staging,
        }
    }
}

/// A service over `tv` with no keeper step inside a test, its command
/// timeouts counted on its own metrics.
async fn service(tv: &TestValkey) -> LiveEditService {
    let svc = LiveEditService {
        presence_refresh_interval_secs: QUIET_REFRESH_SECS,
        ..build_service(tv, Arc::new(AllowAllPermissions))
    };
    tv.client()
        .install_observability_hooks(Arc::clone(&svc.metrics))
        .await;
    svc
}

/// The fleet hold every test here ends its join with.
fn hold() -> HoldCause {
    HoldCause::Held {
        incident: Some(INCIDENT.to_owned()),
    }
}

/// Poll `holds` every [`POLL`] until it is true, within [`SETTLE`].
async fn settle(what: &str, holds: impl Fn() -> bool) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while !holds() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} did not happen within {SETTLE:?}"
        );
        tokio::time::sleep(POLL).await;
    }
}

/// Assert `status` is the hold's ending.
#[track_caller]
fn assert_ended_by_the_hold(status: Option<Status>, what: &str) {
    let status = status
        .unwrap_or_else(|| panic!("{what}: the join ended OK; the hold ends it SERVICE_HELD"));
    assert_service_held(&status, Some(INCIDENT));
}

// ═════════════════════════════════════════════════════════════════════════════
// T12 (F5) — SR-4 steps 2, 4 and 5 race the join's token
// ═════════════════════════════════════════════════════════════════════════════

/// T12: a join parked at `before_register` — indexed, subscribed, its clock
/// read and snapshot taken, its register not yet issued — when the hold ends
/// its stream. Released, it sees the ending before its register, which is
/// raced against the join's token: the stream ends `SERVICE_HELD` naming the
/// incident, no register ever reaches the store (no `PeerJoined` is
/// published, no member written), and teardown runs (the session left
/// unindexed, the teardown marker armed).
///
/// Fail-on-old: with the register awaited without the race, it registers and
/// announces the join (a `PeerJoined` reaches the tap) before step 6 sees
/// the ending.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_join_parked_before_its_register_when_the_hold_ends_it_never_registers() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let seam = Arc::new(live_edit::grpc::presence::JoinRegistrationSeam::new());
        let svc = LiveEditService {
            join_registration_seam_for_test: Some(Arc::clone(&seam)),
            ..service(&tv).await
        };
        let (session_id, _token) = create_session(&tv, &svc, "t12-parked", true).await;
        let mut tap = Tap::open(&tv, &session_id).await;

        seam.before_register.arm();
        let (mut stream, _client) = open_join(&svc, &session_id, "t12-parked-app").await;
        wait_parked(&seam.before_register, "the join, before its register").await;
        svc.shutdown.end_streams_for_hold(hold());
        seam.before_register.release();

        let (delivered, status) =
            read_to_end_within(&mut stream, "a hold before the register", STREAM_END).await;
        assert!(
            delivered.is_empty(),
            "the join ends before SessionConnected; got {delivered:?}"
        );
        assert_ended_by_the_hold(status, "a hold before the register");
        drain_tracker(&svc).await;
        let events = tap.drain(&tv).await;
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PresenceEvent::PeerJoined(_))),
            "no register was sent after the ending, so no PeerJoined was published; got \
             {events:?}"
        );
        let keys = presence_keys(&tv, &session_id).await;
        assert!(
            keys.len() == 1 && keys[0].ends_with(":gone"),
            "teardown armed the teardown marker and nothing else is left: no member, info hash \
             or lease; got {keys:?}"
        );
        assert!(
            !is_indexed(&tv, &session_id).await,
            "teardown unindexed the session, which has no member"
        );
        tap.close().await;
    })
    .await;
}

/// T12: a join whose register is in flight when the hold ends its stream —
/// the register's `EVALSHA` reached the store and ran, its reply held by the
/// proxy. The join stops waiting for it at the ending: its teardown's
/// unregister is sent while the register's reply is still held; the stream
/// ends `SERVICE_HELD` naming the incident; and no step-5 index re-add is
/// ever sent. Teardown removes what the register wrote.
///
/// Fail-on-old: with the register awaited without the race, the join waits
/// out the held reply until the register's command timeout and ends
/// `UNAVAILABLE` naming the timed-out register, not by the hold.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_join_holding_its_register_when_the_hold_ends_it_tears_down_without_the_re_add() {
    scenario(async {
        let fx = Fx::new().await;
        let tv = &fx.tv;
        let seam = Arc::new(live_edit::grpc::presence::JoinRegistrationSeam::new());
        let svc = LiveEditService {
            join_registration_seam_for_test: Some(Arc::clone(&seam)),
            ..service(tv).await
        };
        let (session_id, _token) = create_session(tv, &svc, "t12-held-register", true).await;

        seam.before_register.arm();
        let (mut stream, _client) = open_join(&svc, &session_id, "t12-register-app").await;
        wait_parked(&seam.before_register, "the join, before its register").await;
        let gate = ReplyGate::new();
        fx.staging.arm(Staged::HoldRegister {
            members_key: tv.key_schema().presence_members(&session_id),
            index_key: tv.key_schema().presence_active_sessions(),
            session_id: session_id.clone(),
            gate: gate.clone(),
        });
        seam.before_register.release();
        gate.reply_held().await;

        svc.shutdown.end_streams_for_hold(hold());
        let staging = Arc::clone(&fx.staging);
        // The register's reply stays held until the teardown has sent its
        // unregister, so the join cannot have gone on by the register's answer.
        settle("the teardown's unregister to be sent", move || {
            staging.scripts() >= 2
        })
        .await;
        gate.open();

        let (delivered, status) =
            read_to_end_within(&mut stream, "a hold during the register", STREAM_END).await;
        assert!(
            delivered.is_empty(),
            "the join ends before SessionConnected; got {delivered:?}"
        );
        assert_ended_by_the_hold(status, "a hold during the register");
        drain_tracker(&svc).await;
        assert_eq!(
            fx.staging.index_adds(),
            0,
            "no step-5 index re-add follows a register the ending dropped"
        );
        let keys = presence_keys(tv, &session_id).await;
        assert!(
            keys.len() == 1 && keys[0].ends_with(":gone"),
            "teardown removed what the register wrote and armed the teardown marker; got {keys:?}"
        );
        assert!(
            !is_indexed(tv, &session_id).await,
            "teardown unindexed the session, which has no member"
        );
    })
    .await;
}

/// T12: an established join whose `PeerMessage` publish is in flight — its
/// `PUBLISH` held at the proxy — when the hold ends its stream. The publish
/// races the join's token, so the join ends `SERVICE_HELD` naming the
/// incident without waiting for it, and runs its teardown: the publish's
/// caller gave up rather than timing out, so no command timeout is counted.
///
/// Fail-on-old: with the publish awaited without the race, the join ends
/// only once the held publish times out (a `PeerMessage` publish failure is
/// logged and the loop goes on), and one data-pool command timeout is
/// counted.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_join_whose_peer_message_publish_is_held_ends_at_the_hold() {
    scenario(async {
        let fx = Fx::new().await;
        let tv = &fx.tv;
        let svc = service(tv).await;
        let (session_id, _token) = create_session(tv, &svc, "t12-held-publish", true).await;
        let (mut stream, client) = open_join(&svc, &session_id, "t12-publish-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        let gate = ReplyGate::new();
        fx.staging.arm(Staged::HoldPublish {
            channel: tv.key_schema().presence_events_channel(&session_id),
            gate: gate.clone(),
        });

        client
            .send(encode_join_frame(&join_message(b"t12-held".to_vec())))
            .expect("the client channel must accept a message");
        let staging = Arc::clone(&fx.staging);
        settle("the PeerMessage publish to reach the proxy", move || {
            staging.publishes() >= 1
        })
        .await;
        svc.shutdown.end_streams_for_hold(hold());

        let (_, status) =
            read_to_end_within(&mut stream, "a hold during a publish", STREAM_END).await;
        assert_ended_by_the_hold(status, "a hold during a publish");
        drain_tracker(&svc).await;
        assert_eq!(
            command_timeout_row(&svc.metrics),
            command_timeouts_only_on(POOL_LABEL_DATA, 0),
            "the join gave the held publish up at the ending: no command timeout was counted"
        );
        assert!(
            !is_member(tv, &session_id, &cid).await,
            "teardown ran: the member is gone"
        );
        gate.open();
        drop(client);
    })
    .await;
}

// ── The other SR-4 store awaits: steps 2, 4 (clock, snapshot) and 5 ─────────
//
// One test per raced await, so each race has a test that goes red without
// it. The register's own race is the `before_register` and held-register
// tests above.

/// [`service`] on `tv` with the registration and replay seams installed.
fn seamed(
    svc: LiveEditService,
) -> (
    LiveEditService,
    Arc<live_edit::grpc::presence::JoinRegistrationSeam>,
    Arc<live_edit::grpc::presence::PresenceReplaySeam>,
) {
    let registration = Arc::new(live_edit::grpc::presence::JoinRegistrationSeam::new());
    let replay = Arc::new(live_edit::grpc::presence::PresenceReplaySeam::new());
    let svc = LiveEditService {
        join_registration_seam_for_test: Some(Arc::clone(&registration)),
        presence_replay_seam_for_test: Some(Arc::clone(&replay)),
        ..svc
    };
    (svc, registration, replay)
}

/// Read a join that the hold ended before it registered to its end, and
/// check it ended by the hold, before `SessionConnected`, and that its
/// teardown left no member and the session unindexed.
async fn assert_unregistered_join_ended_by_the_hold(
    tv: &TestValkey,
    svc: &LiveEditService,
    stream: &mut JoinResponseStream,
    session_id: &str,
    what: &str,
) {
    let (delivered, status) = read_to_end_within(stream, what, STREAM_END).await;
    assert!(
        delivered.is_empty(),
        "{what}: the join ends before SessionConnected; got {delivered:?}"
    );
    assert_ended_by_the_hold(status, what);
    drain_tracker(svc).await;
    let keys = presence_keys(tv, session_id).await;
    assert!(
        keys.iter().all(|k| k.ends_with(":gone")),
        "{what}: no member, info hash or lease is left; got {keys:?}"
    );
    assert!(
        !is_indexed(tv, session_id).await,
        "{what}: teardown unindexed the session, which has no member"
    );
}

/// Step 2: a join whose first index `SADD` is in flight — it ran on the
/// store, its reply held — when the hold ends its stream stops waiting for
/// it and ends `SERVICE_HELD`, then tears down.
///
/// Fail-on-old: with the index add awaited without the race, the join waits
/// out the held reply until its command timeout and ends `UNAVAILABLE`
/// naming the timed-out `SADD`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_join_whose_first_index_add_is_held_ends_at_the_hold() {
    scenario(async {
        let fx = Fx::new().await;
        let tv = &fx.tv;
        let svc = service(tv).await;
        let (session_id, _token) = create_session(tv, &svc, "f5-index-add", true).await;
        let gate = ReplyGate::new();
        fx.staging.arm(Staged::HoldIndexAdd {
            index_key: tv.key_schema().presence_active_sessions(),
            session_id: session_id.clone(),
            gate: gate.clone(),
        });

        let (mut stream, _client) = open_join(&svc, &session_id, "f5-index-add-app").await;
        gate.reply_held().await;
        svc.shutdown.end_streams_for_hold(hold());

        assert_unregistered_join_ended_by_the_hold(
            tv,
            &svc,
            &mut stream,
            &session_id,
            "a hold during the index add",
        )
        .await;
        gate.open();
    })
    .await;
}

/// Step 4's clock read: a join parked after its index add (before its
/// router subscribe, which is not raced) when the hold ends its stream sends
/// no `TIME` once released, and ends `SERVICE_HELD`.
///
/// Fail-on-old: with the clock read awaited without the race, its `TIME` is
/// sent after the ending (the snapshot's race then ends the join).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_join_ended_before_its_clock_read_sends_no_time() {
    scenario(async {
        let fx = Fx::new().await;
        let tv = &fx.tv;
        let (svc, registration, _replay) = seamed(service(tv).await);
        let (session_id, _token) = create_session(tv, &svc, "f5-clock", true).await;

        registration.after_index_add.arm();
        let (mut stream, _client) = open_join(&svc, &session_id, "f5-clock-app").await;
        wait_parked(
            &registration.after_index_add,
            "the join, after its index add",
        )
        .await;
        fx.staging.arm(Staged::Count {
            command: b"TIME",
            naming: None,
        });
        svc.shutdown.end_streams_for_hold(hold());
        registration.after_index_add.release();

        assert_unregistered_join_ended_by_the_hold(
            tv,
            &svc,
            &mut stream,
            &session_id,
            "a hold before the clock read",
        )
        .await;
        assert_eq!(fx.staging.counted(), 0, "no TIME is sent after the ending");
    })
    .await;
}

/// Step 4's membership snapshot: a join parked just before its `SMEMBERS`
/// (`PresenceReplaySeam::before_snapshot`, its clock already read) when the
/// hold ends its stream sends no `SMEMBERS` once released, and ends
/// `SERVICE_HELD`.
///
/// Fail-on-old: with the snapshot awaited without the race, its `SMEMBERS`
/// is sent after the ending (the register's race then ends the join).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_join_ended_before_its_snapshot_sends_no_smembers() {
    scenario(async {
        let fx = Fx::new().await;
        let tv = &fx.tv;
        let (svc, _registration, replay) = seamed(service(tv).await);
        let (session_id, _token) = create_session(tv, &svc, "f5-snapshot", true).await;

        replay.before_snapshot.arm();
        let (mut stream, _client) = open_join(&svc, &session_id, "f5-snapshot-app").await;
        wait_parked(&replay.before_snapshot, "the join, before its snapshot").await;
        fx.staging.arm(Staged::Count {
            command: b"SMEMBERS",
            naming: Some(tv.key_schema().presence_members(&session_id)),
        });
        svc.shutdown.end_streams_for_hold(hold());
        replay.before_snapshot.release();

        assert_unregistered_join_ended_by_the_hold(
            tv,
            &svc,
            &mut stream,
            &session_id,
            "a hold before the snapshot",
        )
        .await;
        assert_eq!(
            fx.staging.counted(),
            0,
            "no SMEMBERS is sent after the ending"
        );
    })
    .await;
}

/// Step 5: a join whose index re-add after its register is in flight — it
/// ran on the store, its reply held — when the hold ends its stream stops
/// waiting for it, ends `SERVICE_HELD`, and tears down what the register
/// wrote.
///
/// Fail-on-old: with the re-add awaited without the race, the join waits out
/// the held reply until its command timeout and ends `UNAVAILABLE` naming
/// the timed-out `SADD`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_join_whose_index_re_add_is_held_ends_at_the_hold() {
    scenario(async {
        let fx = Fx::new().await;
        let tv = &fx.tv;
        let (svc, registration, _replay) = seamed(service(tv).await);
        let (session_id, _token) = create_session(tv, &svc, "f5-re-add", true).await;

        registration.before_register.arm();
        let (mut stream, _client) = open_join(&svc, &session_id, "f5-re-add-app").await;
        wait_parked(
            &registration.before_register,
            "the join, before its register",
        )
        .await;
        let gate = ReplyGate::new();
        fx.staging.arm(Staged::HoldIndexAdd {
            index_key: tv.key_schema().presence_active_sessions(),
            session_id: session_id.clone(),
            gate: gate.clone(),
        });
        registration.before_register.release();
        gate.reply_held().await;
        svc.shutdown.end_streams_for_hold(hold());

        let what = "a hold during the index re-add";
        let (delivered, status) = read_to_end_within(&mut stream, what, STREAM_END).await;
        assert!(
            delivered.is_empty(),
            "{what}: the join ends before SessionConnected; got {delivered:?}"
        );
        assert_ended_by_the_hold(status, what);
        drain_tracker(&svc).await;
        let keys = presence_keys(tv, &session_id).await;
        assert!(
            keys.len() == 1 && keys[0].ends_with(":gone"),
            "{what}: teardown removed what the register wrote and armed the teardown marker; \
             got {keys:?}"
        );
        gate.open();
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// T11 (F4) — the teardown's unregister is cut at the command timeout
// ═════════════════════════════════════════════════════════════════════════════

/// T11: an established join whose client disconnects, so its teardown runs.
/// The teardown's unregister `EVALSHA` is answered `-NOSCRIPT` after half a
/// command timeout, and the reload's `SCRIPT LOAD` never answers. The cut
/// gives the whole unregister up at one command timeout, reload included,
/// and teardown goes on to its end: the stream ends `OK`; the proxy saw one
/// unregister and one `SCRIPT LOAD`, and no re-sent `EVALSHA`; no command
/// timeout is counted, since fred's own timer on the `SCRIPT LOAD` (half a
/// command timeout after the cut) never fired; and the cut is handled as a
/// failure: the member is left for its lease to lapse and a sweep to remove,
/// and the session index alone.
///
/// Fail-on-old: without the cut, the unregister ends only when fred's own
/// timer on the stalled `SCRIPT LOAD` fires, and that timeout is counted on
/// the data pool.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_teardown_unregister_answered_noscript_is_cut_at_the_command_timeout() {
    scenario(async {
        let fx = Fx::new().await;
        let tv = &fx.tv;
        let svc = service(tv).await;
        let (session_id, _token) = create_session(tv, &svc, "t11-unregister", true).await;
        let (mut stream, client) = open_join(&svc, &session_id, "t11-app").await;
        let cid = expect_session_connected(&mut stream, SETTLE).await;
        let gate = ReplyGate::new();
        fx.staging.arm(Staged::NoscriptUnregister {
            members_key: tv.key_schema().presence_members(&session_id),
            gate: gate.clone(),
        });

        drop(client);
        let (_, status) =
            read_to_end_within(&mut stream, "a teardown with a cut unregister", STREAM_END).await;
        assert!(
            status.is_none(),
            "a client that leaves ends its join OK; got {status:?}"
        );
        drain_tracker(&svc).await;

        assert_eq!(
            (fx.staging.scripts(), fx.staging.reloads()),
            (1, 1),
            "precondition: the unregister was answered NOSCRIPT and its reload's SCRIPT LOAD \
             sent, and no EVALSHA was re-sent"
        );
        assert_eq!(
            command_timeout_row(&svc.metrics),
            command_timeouts_only_on(POOL_LABEL_DATA, 0),
            "the cut gave the unregister up before fred's own timer on the SCRIPT LOAD fired"
        );
        assert!(
            is_member(tv, &session_id, &cid).await && is_indexed(tv, &session_id).await,
            "a cut unregister is handled as a failure: the member stays until its lease lapses \
             and the index is left alone"
        );
        gate.open();
    })
    .await;
}
