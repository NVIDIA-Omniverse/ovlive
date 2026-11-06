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

//! The hold poll and the startup hold evaluation: one bounded read of the
//! hold marker, repeated every poll interval, publishing only what answered.

use std::time::Duration;

use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::{Instant, MissedTickBehavior};
use tokio_util::sync::CancellationToken;

use super::verdict::{poll_call_timeout, verdict_from_marker, HoldObservation};
use crate::config::Config;
use crate::storage::client::{is_store_not_serving_reply, is_transient_fred_error, ValkeyClient};
use crate::storage::dr_hold::read_hold_marker;
use crate::storage::keys::KeySchema;

/// The hold poll's timing: how often it reads, and the bound on each read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HoldPollSettings {
    /// Time between the starts of two reads
    /// (`LIVE_EDIT_DR_HOLD_POLL_INTERVAL_MS`). Must be non-zero.
    pub interval: Duration,
    /// The bound on one read: [`poll_call_timeout`] of the Valkey command
    /// timeout and `interval`.
    pub call_timeout: Duration,
}

impl HoldPollSettings {
    /// The settings for `interval` and a Valkey command timeout of
    /// `command_timeout_ms` (0 = none).
    pub fn new(interval: Duration, command_timeout_ms: u64) -> Self {
        Self {
            interval,
            call_timeout: poll_call_timeout(command_timeout_ms, interval),
        }
    }

    /// The settings `config` gives: `dr_hold_poll_interval_ms` and
    /// `valkey_command_timeout_ms`.
    pub fn from_config(config: &Config) -> Self {
        Self::new(
            Duration::from_millis(config.dr_hold_poll_interval_ms),
            config.valkey_command_timeout_ms,
        )
    }
}

/// Why a hold read produced no verdict. Never a verdict itself (SDD SR-2.7):
/// a caller publishes nothing for it.
#[derive(Debug, thiserror::Error)]
pub enum HoldPollError {
    /// The read did not answer within the per-call bound.
    #[error("the hold read got no answer within {0:?}")]
    Timeout(Duration),
    /// The read failed: a connection error, a store reply such as `-LOADING`,
    /// `-BUSY` or `CLUSTERDOWN`, a failed redirect, a script error, or a reply
    /// outside the script's contract.
    #[error("the hold read failed ({class}): {0}", class = classify(.0))]
    Store(#[source] fred::error::Error),
}

impl HoldPollError {
    /// A short class for logs: `call_timeout` (the per-call bound),
    /// `command_timeout` (fred's), `store_not_serving` (`-LOADING` /
    /// `-BUSY`), `transient` (the rest of the shapes the shared classifier
    /// calls transient: connection, cluster and redirect failures), or
    /// `other`.
    pub fn class(&self) -> &'static str {
        match self {
            Self::Timeout(_) => "call_timeout",
            Self::Store(e) => classify(e),
        }
    }
}

fn classify(e: &fred::error::Error) -> &'static str {
    if *e.kind() == fred::error::ErrorKind::Timeout {
        "command_timeout"
    } else if is_store_not_serving_reply(e) {
        "store_not_serving"
    } else if is_transient_fred_error(e) {
        "transient"
    } else {
        "other"
    }
}

/// One hold read, bounded by `call_timeout`: the [`HoldObservation`] when the
/// store answered, stamped with the instant it did, or the reason it did not.
///
/// The bound is a `tokio::time::timeout` around the whole read (a `NOSCRIPT`
/// reload included), so the call returns within `call_timeout` whatever the
/// connection does.
pub async fn poll_hold_once(
    client: &ValkeyClient,
    keys: &KeySchema,
    call_timeout: Duration,
) -> Result<HoldObservation, HoldPollError> {
    match tokio::time::timeout(call_timeout, read_hold_marker(client, keys)).await {
        Err(_) => Err(HoldPollError::Timeout(call_timeout)),
        Ok(Err(e)) => Err(HoldPollError::Store(e)),
        Ok(Ok(marker)) => Ok(HoldObservation {
            verdict: verdict_from_marker(marker.as_ref()),
            at: Instant::now(),
        }),
    }
}

/// The one-shot, fail-closed hold evaluation a process runs at startup, after
/// the Valkey connect and the durability guard, and before it builds any
/// store, dispatcher or store- or backup-writing task (the metrics server and
/// the service-token providers are already running) (SDD SR-5.1a): one
/// bounded read, never retried in process.
///
/// `Ok` carries `Serving` or `Held` and seeds the poll
/// ([`spawn_hold_poll`]'s `initial`). `Err` is no verdict, and the caller
/// must not serve on it: `main.rs` exits non-zero at once, and the kubelet's
/// restart, under its backoff, is the retry. An in-process wait would add a
/// boot step the startup probe's budget does not count.
pub async fn evaluate_hold_at_startup(
    client: &ValkeyClient,
    keys: &KeySchema,
    settings: HoldPollSettings,
) -> Result<HoldObservation, HoldPollError> {
    poll_hold_once(client, keys, settings.call_timeout).await
}

/// Spawn the hold poll (SDD SR-2.2) and return the receiving side of its
/// verdict channel with the task's handle.
///
/// The channel starts at `initial`, the startup evaluation's observation.
/// The first read starts one `settings.interval` later, and every interval
/// after that. Reads run one at a time, each bounded by
/// `settings.call_timeout` ([`HoldPollSettings::new`] keeps it at most the
/// interval, so a stuck read never delays the next one). A read that answers
/// replaces the channel's value with a new observation, even when the verdict
/// is unchanged, so its `at` is always the last successful read. A read that
/// fails publishes nothing (SDD SR-2.7).
///
/// Logs the first failed read after a success at WARN with its class, later
/// ones in the same run at DEBUG, the read that ends a run at INFO, and each
/// change of verdict at INFO. Nothing here reacts to a verdict or to its age.
///
/// Stops when `shutdown` is cancelled, abandoning a read in flight; the
/// receivers then see the channel closed. Panics if `settings.interval` is
/// zero (`Config::from_env` refuses it).
pub fn spawn_hold_poll(
    client: ValkeyClient,
    keys: KeySchema,
    settings: HoldPollSettings,
    initial: HoldObservation,
    shutdown: CancellationToken,
) -> (watch::Receiver<HoldObservation>, JoinHandle<()>) {
    let (tx, rx) = watch::channel(initial);
    let handle = tokio::spawn(run_hold_poll(client, keys, settings, tx, shutdown));
    (rx, handle)
}

async fn run_hold_poll(
    client: ValkeyClient,
    keys: KeySchema,
    settings: HoldPollSettings,
    tx: watch::Sender<HoldObservation>,
    shutdown: CancellationToken,
) {
    let mut ticker =
        tokio::time::interval_at(Instant::now() + settings.interval, settings.interval);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    // Consecutive failed reads since the last success.
    let mut failed_reads: u64 = 0;
    loop {
        tokio::select! {
            biased;
            () = shutdown.cancelled() => return,
            _ = ticker.tick() => {}
        }
        let result = tokio::select! {
            biased;
            () = shutdown.cancelled() => return,
            result = poll_hold_once(&client, &keys, settings.call_timeout) => result,
        };
        match result {
            Ok(observation) => {
                if failed_reads > 0 {
                    tracing::info!(
                        failed_reads,
                        "hold poll answered again after failed reads; publishing its verdict"
                    );
                    failed_reads = 0;
                }
                let previous = tx.send_replace(observation);
                let current = tx.borrow();
                if previous.verdict != current.verdict {
                    tracing::info!(
                        held = current.verdict.is_held(),
                        incident = current.verdict.incident().unwrap_or(""),
                        "hold verdict changed"
                    );
                }
            }
            Err(e) => {
                failed_reads += 1;
                let since_last_success = tx.borrow().at.elapsed();
                if failed_reads == 1 {
                    tracing::warn!(
                        error = %e,
                        class = e.class(),
                        ?since_last_success,
                        "hold poll read failed; no verdict published, so the verdict ages"
                    );
                } else {
                    tracing::debug!(
                        error = %e,
                        class = e.class(),
                        failed_reads,
                        ?since_last_success,
                        "hold poll read failed again; no verdict published"
                    );
                }
            }
        }
    }
}
