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

//! The DR Job lease's refresh (`live_edit::dr::job::lease`; SDD SR-9.1), over
//! a real store reached through a proxy that fails, stalls or watches the
//! lease's own commands:
//!
//! - **One transient refresh error keeps the lease.** A refresh that fails
//!   with an error before the loss deadline is retried every acquire poll,
//!   not at the next interval, so the lease is still this Job's three
//!   intervals past the failed refresh's loss deadline; the WARN says when it
//!   retries and when it would be lost.
//! - **A stalled refresh is lost on time.** With the Valkey command timeout
//!   far above the TTL and a refresh that never answers, the lease is reported
//!   lost by `last confirmation + TTL − refresh interval`, whatever the
//!   command timeout is, naming the unanswered refresh.
//! - **A lease found gone is reported lost at once.** The refresh that finds
//!   it gone reports the loss before the new holder is read for the log,
//!   whose read is stalled here; that read's INFO comes after, as `unknown`.
//!
//! The Job-level exits on a lost lease are `job_frame_test.rs`'s.
//!
//! # Time and capture
//!
//! Real time: the store's replies and the client's timers share the
//! runtime's clock. Each test runs on a current-thread runtime so a
//! [`ThreadCapture`] sees what the spawned refresh task logs, and cleans its
//! own keys with [`purge_keys`] (the fixture's drop cannot on that runtime).
//! The windows are chosen so the behaviour each test pins and the one it
//! rules out are many seconds apart; no assertion bounds a duration from
//! above except as a hang guard well below the alternative. The
//! transient-error test, which asserts the lease stays held, gives each
//! refresh more time before its loss deadline than its client's 4 s command
//! timeout (5 s for a regular refresh, 4.5 s for the retry), so only a refresh
//! the client itself would fail can lose the lease; it runs for 25 s.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::KeysInterface;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use live_edit::config::Config;
use live_edit::dr::job::lease::{JobLease, LeaseTimings};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::external_values::LEADER_REFRESH_IF_HELD_SCRIPT;
use live_edit::storage::leader_lease::LeaderIdentity;
use live_edit::storage::test_support::{connect_test_client, TestValkey};

use super::support::lines_starting;
use crate::grpc::log_capture::{purge_keys, ThreadCapture};
use crate::storage::snapshot_read_support::{
    error_reply_script, test_valkey_addr, ProxyAction, ProxyRule, ScriptReplyProxy,
};

/// The refresh WARN, matched on its fixed start.
const REFRESH_FAILED_PREFIX: &str = "refreshing the DR Job lease failed; retrying in ";

/// The INFO of the detached read of a lost lease's new holder.
const LOST_HOLDER_PREFIX: &str = "the lost DR Job lease is held by ";

/// A client through `proxy` whose Valkey command timeout is
/// `command_timeout_ms`.
async fn client_through(proxy: &ScriptReplyProxy, command_timeout_ms: u64) -> ValkeyClient {
    let mut config: Config = crate::common::test_config();
    config.valkey_url = proxy.url();
    config.valkey_command_timeout_ms = command_timeout_ms;
    connect_test_client(&config)
        .await
        .expect("a client connects through the proxy")
}

/// A rule acting on the refresh script's frames naming `lease` only: the
/// `n`th of them (from 0) gets `act(n)`; `None` forwards it unchanged and
/// uncounted.
fn refresh_rule(
    lease: String,
    act: impl Fn(usize) -> Option<ProxyAction> + Send + Sync + 'static,
) -> ProxyRule {
    let sha = LEADER_REFRESH_IF_HELD_SCRIPT.sha1().to_string();
    let seen = AtomicUsize::new(0);
    Arc::new(move |args: &[Vec<u8>]| {
        let is_refresh = args.first()?.eq_ignore_ascii_case(b"EVALSHA")
            && args.get(1).is_some_and(|a| a.as_slice() == sha.as_bytes())
            && args.iter().any(|a| a.as_slice() == lease.as_bytes());
        if !is_refresh {
            return None;
        }
        act(seen.fetch_add(1, Ordering::SeqCst))
    })
}

/// The fixture's store with the refresh script already loaded, so no
/// refresh is answered `NOSCRIPT` and re-sent: the proxy rules count one
/// frame per refresh.
async fn setup() -> TestValkey {
    let tv = TestValkey::setup().await;
    LEADER_REFRESH_IF_HELD_SCRIPT
        .load(tv.client().pool.next())
        .await
        .expect("SCRIPT LOAD of the refresh script");
    tv
}

fn identity() -> LeaderIdentity {
    LeaderIdentity::new("dr-job-lease-test", uuid::Uuid::new_v4().to_string())
}

async fn holder(tv: &TestValkey) -> Option<String> {
    tv.client()
        .pool
        .get(tv.key_schema().dr_job_lease())
        .await
        .expect("GET")
}

/// The first refresh fails with an error; it is retried after the acquire
/// poll (500 ms), succeeds, and the lease stays this Job's: past where the
/// loss deadline of the failed refresh lay and across three more refresh
/// intervals, `lost()` is not cancelled and the key still holds this Job's
/// value. One WARN carries when it retries (`retry_in_ms`) and how long is
/// left before the lease would be lost (`lost_in_ms`).
///
/// A refresh left to the next interval instead would fall on the deadline
/// itself (two intervals after the last confirmation, with the TTL kept at
/// three intervals), where the loss timer wins, and one transient store
/// error would end the Job.
///
/// The timings are five times the usual short ones (TTL 15 s, refresh 5 s,
/// retry 500 ms), and the client the refreshes run on has a 4 s command
/// timeout. Each regular refresh has a whole 5 s interval before its loss
/// deadline and the retry has 4.5 s, both more than that timeout: a refresh
/// slowed by a loaded runner times out in the client before the loss timer
/// fires, so the lease is lost only where the client itself has failed the
/// refresh. The injected error is a reply, not a delay, so it reaches the
/// client as an error whatever the timeout.
#[tokio::test]
async fn one_transient_refresh_error_keeps_the_lease_and_is_retried_at_the_acquire_poll() {
    let tv = setup().await;
    let capture = ThreadCapture::start();
    let lease_key = tv.key_schema().dr_job_lease();
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        refresh_rule(lease_key.clone(), |n| {
            Some(if n == 0 {
                ProxyAction::Eval(error_reply_script("ERR injected transient refresh failure"))
            } else {
                ProxyAction::Delay(Duration::ZERO)
            })
        }),
    )
    .await;
    // Below both slacks: 5 s for a regular refresh, 4.5 s for the retry.
    let client = client_through(&proxy, 4_000).await;
    let timings = LeaseTimings {
        ttl: Duration::from_secs(15),
        refresh_interval: Duration::from_secs(5),
        acquire_wait: Duration::from_secs(1),
        acquire_poll: Duration::from_millis(500),
    };
    let me = identity();
    let lease = JobLease::acquire(
        &client,
        tv.key_schema(),
        me.clone(),
        timings,
        &CancellationToken::new(),
    )
    .await
    .expect("the lease is free");
    let refresh = lease.spawn_refresh();

    // The failure is at about 5 s; its loss deadline at about 10 s. Watch to
    // 25 s: three more intervals past the deadline.
    let watched = tokio::time::timeout(Duration::from_secs(25), refresh.lost().cancelled()).await;
    assert!(
        watched.is_err(),
        "one transient refresh error lost the lease: {}",
        refresh.reason()
    );
    assert!(
        proxy.hits() >= 4,
        "the failed refresh, its retry and the regular refreshes after it reached the store \
         ({} refreshes)",
        proxy.hits()
    );
    assert_eq!(
        holder(&tv).await,
        Some(me.lock_value()),
        "the lease is still this Job's"
    );
    let captured = capture.snapshot();
    let warns = lines_starting(&captured, REFRESH_FAILED_PREFIX);
    assert_eq!(warns.len(), 1, "one WARN:\n{}", captured.text);
    assert_eq!(warns[0]["level"], "WARN");
    assert_eq!(
        warns[0]["fields"]["retry_in_ms"].as_u64(),
        Some(500),
        "retried after the acquire poll: {}",
        warns[0]
    );
    let lost_in = warns[0]["fields"]["lost_in_ms"]
        .as_u64()
        .expect("the WARN carries the time left");
    assert!(
        lost_in > 500 && lost_in <= 10_000,
        "the time left before the loss is within TTL − refresh interval: {lost_in} ms"
    );

    refresh.stop();
    lease.release().await;
    purge_keys(&tv).await;
}

/// A refresh that never answers, with a Valkey command timeout (60 s) far
/// above the TTL: the loss timer reports the lease lost
/// `TTL − refresh interval` (2 s) after its last confirmation, naming the
/// unanswered refresh, without waiting for the command timeout.
#[tokio::test]
async fn a_refresh_stalled_past_its_deadline_reports_the_lease_lost_before_the_command_timeout() {
    let tv = setup().await;
    let _capture = ThreadCapture::start();
    let lease_key = tv.key_schema().dr_job_lease();
    // The first refresh answers; the second never does.
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        refresh_rule(lease_key.clone(), |n| {
            Some(if n == 0 {
                ProxyAction::Delay(Duration::ZERO)
            } else {
                ProxyAction::Stall
            })
        }),
    )
    .await;
    let client = client_through(&proxy, 60_000).await;
    let timings = LeaseTimings {
        ttl: Duration::from_secs(3),
        refresh_interval: Duration::from_secs(1),
        acquire_wait: Duration::from_secs(1),
        acquire_poll: Duration::from_millis(100),
    };
    let lease = JobLease::acquire(
        &client,
        tv.key_schema(),
        identity(),
        timings,
        &CancellationToken::new(),
    )
    .await
    .expect("the lease is free");
    let refresh = lease.spawn_refresh();

    // The second refresh goes out at about 2 s and stalls; the first, at
    // about 1 s, confirmed the lease, so the loss is due at about 3 s. A
    // loss judged only when the stalled refresh returns would come at its
    // 60 s command timeout.
    tokio::time::timeout(Duration::from_secs(15), refresh.lost().cancelled())
        .await
        .unwrap_or_else(|_| {
            panic!(
                "a refresh stalled past the loss deadline did not lose the lease within 15 s \
                 ({} refreshes sent)",
                proxy.hits()
            )
        });
    assert_eq!(
        proxy.hits(),
        2,
        "one confirmed refresh, then the stalled one"
    );
    let reason = refresh.reason();
    assert!(
        reason.starts_with(&format!(
            "no refresh of the lease {lease_key} succeeded for "
        )) && reason.ends_with(" (the last refresh had not answered)"),
        "the reason names the unanswered refresh: {reason}"
    );

    // The proxied connection is stalled: release through no command on it.
    refresh.stop();
    purge_keys(&tv).await;
}

/// The lease deleted under the Job: the next refresh finds it gone and the
/// loss is reported at once, with the reason `the lease <key> no longer holds
/// this Job's value`, while the read of the new holder for the log is still
/// stalled. That read runs on its own, bounded task, and its INFO comes
/// after, naming the holder `unknown`.
///
/// The TTL here is long (20 s; 19.5 s to the loss deadline), so a loss
/// reported only after the holder read, bounded by that deadline, would come
/// many seconds after the refresh that found the lease gone.
#[tokio::test]
async fn a_lease_found_gone_is_reported_lost_at_once_before_its_holder_is_read() {
    let tv = setup().await;
    let capture = ThreadCapture::start();
    let lease_key = tv.key_schema().dr_job_lease();
    let watched = lease_key.clone();
    let holder_reads = Arc::new(AtomicUsize::new(0));
    let reads = Arc::clone(&holder_reads);
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        Arc::new(move |args: &[Vec<u8>]| {
            let is_holder_read = args.first()?.eq_ignore_ascii_case(b"GET")
                && args
                    .get(1)
                    .is_some_and(|a| a.as_slice() == watched.as_bytes());
            if !is_holder_read {
                return None;
            }
            reads.fetch_add(1, Ordering::SeqCst);
            Some(ProxyAction::Stall)
        }),
    )
    .await;
    let client = client_through(&proxy, 30_000).await;
    let timings = LeaseTimings {
        ttl: Duration::from_secs(20),
        refresh_interval: Duration::from_millis(500),
        acquire_wait: Duration::from_secs(1),
        acquire_poll: Duration::from_millis(100),
    };
    let lease = JobLease::acquire(
        &client,
        tv.key_schema(),
        identity(),
        timings,
        &CancellationToken::new(),
    )
    .await
    .expect("the lease is free");
    let refresh = lease.spawn_refresh();

    let _: i64 = tv.client().pool.del(&lease_key).await.expect("DEL");
    let deleted = Instant::now();
    tokio::time::timeout(Duration::from_secs(5), refresh.lost().cancelled())
        .await
        .unwrap_or_else(|_| {
            panic!(
                "a lease found gone was not reported lost within 5 s of its deletion ({:?} to \
                 the loss deadline)",
                timings.ttl - timings.refresh_interval
            )
        });
    let lost_after = deleted.elapsed();
    assert_eq!(
        refresh.reason(),
        format!("the lease {lease_key} no longer holds this Job's value"),
        "the reason, with no holder in it"
    );
    assert!(
        lines_starting(&capture.snapshot(), LOST_HOLDER_PREFIX).is_empty(),
        "the loss is reported before the holder read answers ({lost_after:?} after the delete)"
    );

    // The holder read was sent, stalls, and gives up after its 1 s budget.
    let deadline = Instant::now() + Duration::from_secs(10);
    let holder_line = loop {
        let captured = capture.snapshot();
        if let Some(line) = lines_starting(&captured, LOST_HOLDER_PREFIX).first() {
            break (*line).clone();
        }
        assert!(
            Instant::now() < deadline,
            "the holder read's INFO never came:\n{}",
            captured.text
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(
        holder_reads.load(Ordering::SeqCst) >= 1,
        "the holder was read"
    );
    assert_eq!(holder_line["level"], "INFO");
    assert_eq!(holder_line["fields"]["holder"], "unknown");

    refresh.stop();
    purge_keys(&tv).await;
}
