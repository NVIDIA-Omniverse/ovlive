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

//! The Job lease (SDD SR-9.1): at most one restore or accept-loss Job writes
//! to a store at a time.
//!
//! The lease is [`KeySchema::dr_job_lease`], in the object-id counter's slot.
//! A Job takes it with the shared acquire-or-refresh script, waiting up to one
//! TTL for a busy lease; a background task refreshes it every TTL/3 for the
//! Job's whole life. It reports the lease lost the moment a refresh finds it
//! gone or held by another Job, or when no refresh has confirmed it for
//! TTL − one refresh interval (20 s at the production timings). A refresh
//! that fails with an error before that deadline is retried every acquire
//! poll (1 s in production) until the deadline, not at the next interval, so
//! one transient store error does not lose the lease. The deadline is a timer
//! raced against each refresh command, not a check made when a refresh
//! returns, so a lease that lapses is noticed one refresh interval before it
//! can have lapsed, whatever the Valkey command timeout is
//! (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` has no upper bound, and a refresh can
//! stall for all of it). The Job releases the lease on every exit after
//! taking it, with a compare-and-delete that never removes another Job's
//! lease.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::storage::client::ValkeyClient;
use crate::storage::external_values::{
    refresh_leader_lock_if_held, release_leader_lock_if_held, try_acquire_leader_lock,
};
use crate::storage::keys::KeySchema;
use crate::storage::leader_lease::LeaderIdentity;

/// The production lease TTL: 30 s.
pub const JOB_LEASE_TTL: Duration = Duration::from_secs(30);

/// How often the lease is refreshed in production: TTL/3, 10 s.
pub const JOB_LEASE_REFRESH_INTERVAL: Duration = Duration::from_secs(10);

/// How often a Job retries a busy lease while it waits for it, and a failed
/// refresh until its loss deadline.
pub const JOB_LEASE_ACQUIRE_POLL: Duration = Duration::from_secs(1);

/// How long the log-only read of a lost lease's new holder may take. It runs
/// after the loss is reported, so it never delays it.
const LOST_HOLDER_READ_BUDGET: Duration = Duration::from_secs(1);

/// The lease's timings. [`LeaseTimings::production`] in the binary; tests
/// shorten them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeaseTimings {
    /// The lease's TTL, set by every acquire and refresh.
    pub ttl: Duration,
    /// How often the background task refreshes it.
    pub refresh_interval: Duration,
    /// How long a Job waits for a busy lease: one TTL in production, so a
    /// lease its dead holder left behind has lapsed by then.
    pub acquire_wait: Duration,
    /// How often a Job retries a busy lease while it waits, and a failed
    /// refresh until its loss deadline.
    pub acquire_poll: Duration,
}

impl LeaseTimings {
    /// The production timings: TTL 30 s, refresh every 10 s, wait up to 30 s
    /// for a busy lease, retrying every second; a failed refresh is retried
    /// every second too.
    pub fn production() -> Self {
        Self {
            ttl: JOB_LEASE_TTL,
            refresh_interval: JOB_LEASE_REFRESH_INTERVAL,
            acquire_wait: JOB_LEASE_TTL,
            acquire_poll: JOB_LEASE_ACQUIRE_POLL,
        }
    }
}

/// A held lease.
#[derive(Clone, Debug)]
pub struct JobLease {
    client: ValkeyClient,
    key: String,
    identity: LeaderIdentity,
    timings: LeaseTimings,
}

/// Why taking the lease did not succeed.
#[derive(Debug, thiserror::Error)]
pub enum LeaseAcquireError {
    /// Another Job held it for the whole wait.
    #[error("another DR Job holds the lease (held by {holder})")]
    Busy {
        /// The holder's lock value, or `unknown` when it could not be read.
        holder: String,
    },
    /// The store failed the last attempt of the wait.
    #[error("the Job lease could not be taken: {0}")]
    Store(#[source] fred::error::Error),
    /// `cancel` fired while waiting.
    #[error("the Job stopped while waiting for its lease")]
    Cancelled,
}

impl JobLease {
    /// The lease key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The value this Job holds the lease with.
    pub fn holder(&self) -> String {
        self.identity.lock_value()
    }

    /// Take the lease for `identity`, waiting up to `timings.acquire_wait`
    /// for a busy one and retrying every `timings.acquire_poll`. Stops early
    /// when `cancel` fires. A store error is retried within the wait, and the
    /// last one is returned when the wait ends on it.
    pub async fn acquire(
        client: &ValkeyClient,
        keys: &KeySchema,
        identity: LeaderIdentity,
        timings: LeaseTimings,
        cancel: &CancellationToken,
    ) -> Result<Self, LeaseAcquireError> {
        let key = keys.dr_job_lease();
        let deadline = Instant::now() + timings.acquire_wait;
        let ttl_ms = super::quiet::duration_ms(timings.ttl);
        let mut logged_busy = false;
        loop {
            let last = match try_acquire_leader_lock(client, &key, &identity.lock_value(), ttl_ms)
                .await
            {
                Ok(true) => {
                    return Ok(Self {
                        client: client.clone(),
                        key,
                        identity,
                        timings,
                    })
                }
                Ok(false) => {
                    if !logged_busy {
                        logged_busy = true;
                        tracing::info!(
                            lease = %key,
                            wait_secs = timings.acquire_wait.as_secs(),
                            "another DR Job holds the lease; waiting for it to be released or to \
                             lapse"
                        );
                    }
                    None
                }
                Err(e) => {
                    tracing::warn!(lease = %key, error = %e, "taking the DR Job lease failed; retrying");
                    Some(e)
                }
            };
            if Instant::now() >= deadline {
                return Err(match last {
                    Some(e) => LeaseAcquireError::Store(e),
                    None => LeaseAcquireError::Busy {
                        holder: read_holder(client, &key).await,
                    },
                });
            }
            tokio::select! {
                biased;
                () = cancel.cancelled() => return Err(LeaseAcquireError::Cancelled),
                () = tokio::time::sleep(
                    timings.acquire_poll.min(deadline.saturating_duration_since(Instant::now()))
                ) => {}
            }
        }
    }

    /// Spawn the refresh task. It refreshes every `refresh_interval` until
    /// [`LeaseRefresh::stop`], and reports the lease lost
    /// ([`LeaseRefresh::lost`], with [`LeaseRefresh::reason`]) when a refresh
    /// answers that the lease is no longer this Job's, or when no refresh has
    /// confirmed it for `ttl` − `refresh_interval` since the last one that
    /// did. A refresh that fails with an error before that deadline is
    /// retried every `acquire_poll` (capped at the time left) until it. The
    /// deadline fires on time even while a refresh command is still waiting
    /// for its answer.
    pub fn spawn_refresh(&self) -> LeaseRefresh {
        let lost = CancellationToken::new();
        let stop = CancellationToken::new();
        let reason: Arc<OnceLock<String>> = Arc::new(OnceLock::new());
        let handle = self.refresh_task(stop.clone(), lost.clone(), Arc::clone(&reason));
        LeaseRefresh {
            handle,
            lost,
            stop,
            reason,
        }
    }

    fn refresh_task(
        &self,
        stop: CancellationToken,
        lost: CancellationToken,
        reason: Arc<OnceLock<String>>,
    ) -> JoinHandle<()> {
        let lease = self.clone();
        tokio::spawn(async move {
            let ttl_ms = super::quiet::duration_ms(lease.timings.ttl);
            let confirm_within = lease
                .timings
                .ttl
                .saturating_sub(lease.timings.refresh_interval);
            // `last_confirmed` is taken before the confirming command is sent,
            // so the TTL that command set runs from a later instant: the lease
            // cannot lapse before `last_confirmed + ttl`, and the loss
            // deadline below sits one refresh interval ahead of that.
            let mut last_confirmed = Instant::now();
            let mut ticker = tokio::time::interval_at(
                Instant::now() + lease.timings.refresh_interval,
                lease.timings.refresh_interval,
            );
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            // Set after a refresh failed with an error before the deadline:
            // the next refresh goes out then, not at the next tick. The
            // deadline is exactly two intervals after a confirmation, so it
            // coincides with the second tick, and a retry left for that tick
            // would get no time at all.
            let mut retry_at: Option<Instant> = None;
            loop {
                // The loss deadline is a timer of its own, raced against the
                // wait for the next refresh and against the refresh command
                // itself, so neither a late tick nor a refresh stalled for its
                // whole command timeout can carry the Job past it, whatever
                // that timeout is.
                let confirmed_at = last_confirmed;
                let deadline = confirmed_at + confirm_within;
                let unconfirmed = |now: Instant| {
                    format!(
                        "no refresh of the lease {} succeeded for {}s",
                        lease.key,
                        now.saturating_duration_since(confirmed_at).as_secs()
                    )
                };
                tokio::select! {
                    biased;
                    () = stop.cancelled() => return,
                    () = tokio::time::sleep_until(deadline) => {
                        let why = unconfirmed(Instant::now());
                        declare_lost(&lease.key, &stop, &reason, &lost, why);
                        return;
                    }
                    () = async {
                        match retry_at {
                            Some(at) => tokio::time::sleep_until(at).await,
                            None => {
                                ticker.tick().await;
                            }
                        }
                    } => {}
                }
                let retrying = retry_at.take().is_some();
                let sent = Instant::now();
                let refreshed = tokio::time::timeout_at(
                    deadline,
                    refresh_leader_lock_if_held(
                        &lease.client,
                        &lease.key,
                        &lease.identity.lock_value(),
                        ttl_ms,
                    ),
                )
                .await;
                let why = match refreshed {
                    Ok(Ok(true)) => {
                        last_confirmed = sent;
                        if retrying {
                            // Back on the regular cadence, counted from this
                            // confirmation, with no missed tick queued.
                            ticker.reset();
                        }
                        continue;
                    }
                    Ok(Ok(false)) => {
                        // The lease is already gone or another Job's: report
                        // it lost now. The new holder's name is for the log
                        // only; it is read afterwards, on its own task, within
                        // a short fixed budget, so it never delays the report.
                        let why =
                            format!("the lease {} no longer holds this Job's value", lease.key);
                        declare_lost(&lease.key, &stop, &reason, &lost, why);
                        log_lost_lease_holder(lease.client.clone(), lease.key.clone());
                        return;
                    }
                    Err(_elapsed) => format!(
                        "{} (the last refresh had not answered)",
                        unconfirmed(Instant::now())
                    ),
                    Ok(Err(e)) if Instant::now() >= deadline => {
                        format!("{} (last error: {e})", unconfirmed(Instant::now()))
                    }
                    Ok(Err(e)) => {
                        let now = Instant::now();
                        let left = deadline.saturating_duration_since(now);
                        let wait = lease.timings.acquire_poll.min(left);
                        retry_at = Some(now + wait);
                        tracing::warn!(
                            lease = %lease.key,
                            error = %e,
                            retry_in_ms = super::quiet::duration_ms(wait),
                            lost_in_ms = super::quiet::duration_ms(left),
                            "refreshing the DR Job lease failed; retrying in {} ms, and the lease \
                             is reported lost unless a refresh succeeds within {} ms",
                            super::quiet::duration_ms(wait),
                            super::quiet::duration_ms(left)
                        );
                        continue;
                    }
                };
                declare_lost(&lease.key, &stop, &reason, &lost, why);
                return;
            }
        })
    }

    /// Release the lease if this Job still holds it (compare-and-delete). An
    /// error is logged; the lease then lapses within its TTL.
    pub async fn release(&self) {
        match release_leader_lock_if_held(&self.client, &self.key, &self.identity).await {
            Ok(true) => tracing::info!(lease = %self.key, "released the DR Job lease"),
            Ok(false) => tracing::info!(
                lease = %self.key,
                "the DR Job lease was no longer this Job's; left it as it is"
            ),
            Err(e) => tracing::warn!(
                lease = %self.key,
                error = %e,
                "could not release the DR Job lease; it lapses within its TTL"
            ),
        }
    }
}

/// A running lease refresh.
#[derive(Debug)]
pub struct LeaseRefresh {
    handle: JoinHandle<()>,
    lost: CancellationToken,
    stop: CancellationToken,
    reason: Arc<OnceLock<String>>,
}

impl LeaseRefresh {
    /// Cancelled when the refresh reports the lease lost.
    pub fn lost(&self) -> &CancellationToken {
        &self.lost
    }

    /// Why the lease was lost, once it was.
    pub fn reason(&self) -> String {
        self.reason
            .get()
            .cloned()
            .unwrap_or_else(|| "the lease refresh reported it lost".to_owned())
    }

    /// Stop refreshing. The lease then lapses within its TTL unless released.
    pub fn stop(&self) {
        self.stop.cancel();
        self.handle.abort();
    }
}

/// Report the lease lost with `why`, unless the refresh was stopped meanwhile.
fn declare_lost(
    key: &str,
    stop: &CancellationToken,
    reason: &OnceLock<String>,
    lost: &CancellationToken,
    why: String,
) {
    if stop.is_cancelled() {
        return;
    }
    tracing::error!(lease = %key, reason = %why, "the DR Job lost its lease");
    // The reason first, so whoever wakes on `lost` reads it.
    let _ = reason.set(why);
    lost.cancel();
}

/// Log who holds the lease now, after it was reported lost. Runs on its own
/// task, so stopping the refresh task does not cut it short, and gives up
/// after [`LOST_HOLDER_READ_BUDGET`] (holder `unknown`).
fn log_lost_lease_holder(client: ValkeyClient, key: String) {
    tokio::spawn(async move {
        let holder = tokio::time::timeout(LOST_HOLDER_READ_BUDGET, read_holder(&client, &key))
            .await
            .unwrap_or_else(|_| "unknown".to_owned());
        tracing::info!(lease = %key, holder = %holder, "the lost DR Job lease is held by {holder}");
    });
}

/// The lease's current value, for a log line: `unknown` when unreadable,
/// `nobody` when absent.
async fn read_holder(client: &ValkeyClient, key: &str) -> String {
    match crate::storage::dr_job::read_lock_holder(client, key).await {
        Ok(Some(holder)) => holder.value,
        Ok(None) => "nobody".to_owned(),
        Err(_) => "unknown".to_owned(),
    }
}
