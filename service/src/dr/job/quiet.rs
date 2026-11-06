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

//! The quiet period Q and `T_safe`, on the store's clock (SDD SR-9.1): pure
//! arithmetic over the configuration and the hold marker's `detected_at_ms`.
//!
//! # Q's derivation
//!
//! A Job writes nothing until the store's `TIME` reaches `detected_at_ms` +
//! Q. Q must cover the last write any hold-capable replica can still land
//! after the marker was written:
//!
//! 1. **Reaching hold entry.** A replica that cannot see the marker reaches
//!    stale-verdict entry, which aborts the same writers, up to
//!    [`STALE_VERDICT_THRESHOLD`] (30 s) plus one poll interval after
//!    `detected_at`. The poll term is the largest interval the configuration
//!    accepts ([`DR_HOLD_POLL_INTERVAL_MS_MAX`], 6 s), not this process's
//!    setting, so Q holds whatever interval the replicas run with. A replica
//!    that sees the marker enters sooner, within one poll interval.
//! 2. **After entry**, the largest of the bounds below. Each of the two
//!    write bounds counts the latest time a caller gives up and then one 45 s
//!    wait: once a call's
//!    caller gives up — its timer fires, fred answers it with an error, or
//!    its future is dropped — fred writes no frame of it not yet written, and
//!    a frame it had already written reaches the socket within one of fred's
//!    bounded write or flush waits (`UNRESPONSIVE_MAX_TIMEOUT`, 45 s) or is
//!    dropped with its connection (SDD §15.3 *Vendored fred* change 9).
//!    - the hold's write bound, `D + T + 45 s` ([`hold_write_bound`]): a
//!      call already past the gate, an upload's commit included, is dropped
//!      at its RPC deadline D, which is a give-up, so the upload commits'
//!      `txn_timeout` no longer reaches past D; a deadline-exempt stream's
//!      own store work ends at entry, where the hold ending ends the stream;
//!      a clean-up a drop leaves (or an aborted task's) is cut at T after its
//!      spawn, its `NOSCRIPT` reload included; the store out-of-memory
//!      request is behind the hold write fence;
//!    - a `JoinSession` teardown's last write, `5T + 1 s + 45 s`
//!      ([`join_teardown_write_bound`]): the keeper stop (T + 1 s), four
//!      commands each given up within T (the unregister cut at T, its reload
//!      included), then the wait. It is larger than the hold's write bound
//!      once T is above about 7 s at D = 30 s;
//!    - the backup store's call ceiling, C ([`adapter_call_ceiling`]). Entry
//!      aborts every backup writer, which drops its request, but a
//!      `StorageApi` request already sent can still be applied by the
//!      backend: a layer or session checkpoint PUT, a counters-sidecar PUT, a
//!      checkpoint-GC DELETE, a checkpoint clean-up DELETE (of an
//!      external-value copy, of a dead layer's generation or of an orphaned
//!      partial publish), or an external-value blob PUT. For `StorageApi`,
//!      C is `Config::checkpoint_adapter_call_ceiling`, the enforced ceiling
//!      on one adapter call with its retries
//!      (`LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS` × 6, 180 s at the default
//!      30 s). Such a write is harmless once it lands: a late PUT completes a
//!      publish begun before entry, and a late DELETE is a checkpoint-GC or
//!      checkpoint clean-up delete decided before entry (a clean-up delete
//!      licensed by its `{recon}` script). Q outlasts it so that the Job writes
//!      only after it. The restore planner starts during Q but reads nothing
//!      before Q's end, so none of its reads predates such a write
//!      ([`super::JobBody::plan`]). For `LocalDisk`, C is
//!      [`LOCAL_PUBLISH_WORST_CASE`] (5 s), the same per-publish figure the
//!      reconciliation grace uses. The local adapters enforce no ceiling, so C
//!      bounds nothing there and never sets Q: the hold's write bound is at
//!      least 45 s. The honest bound on LocalDisk's late steps is below.
//!      `Disabled` and `InMemory` add no term (C = 0). `Disabled` has no
//!      backup store. An `InMemory` store lives inside each replica's process,
//!      and its writes run on the replica's own task and stop with it, so
//!      nothing in flight at entry lands later. The Job cannot read that store
//!      in any case.
//!    - a graceful shutdown already under way, G ([`shutdown_bound`]). A
//!      replica that received SIGTERM before it could see the marker or go
//!      stale never enters the hold: its hold poll stops at SIGTERM. It goes
//!      on writing through its drain, its paced stream ending and its
//!      post-server sequence until its shutdown ends, within
//!      `Config::minimum_termination_grace` of its SIGTERM: the drain, the
//!      bound after GOAWAY, the post-server budget and the final flush
//!      (76.25 s at the defaults, the same bound the server logs at startup as
//!      the least termination grace period it needs). Its SIGTERM came before
//!      the point at which it would have entered the hold, so G counted from
//!      entry covers it.
//!
//! So `Q = 30 s + 6 s + max(D + T + 45 s, 5T + 46 s, C, G)`, for any
//! configured D and any T > 0: the formula has no floor and no cap, and
//! nothing here refuses a setting. Q covers hold entry, stale entry and a
//! shutdown already under way. It rests on T > 0, which `Config::from_env`
//! enforces at startup for the service and the Jobs alike: at T = 0 fred
//! bounds none of these writes. The Job reads D, T, the request timeout and
//! the shutdown settings from the settings the replicas run with: the chart
//! gives it the same settings ConfigMap. At the defaults (D = 30 s, T = 5 s,
//! a 30 s request timeout, G = 76.25 s), Q is 216 s with `StorageApi` and
//! 116 s with every other adapter; G sets Q only once the shutdown budgets
//! are raised.
//!
//! What Q does not bound, by design:
//! - **A backend that applies an abandoned request late.** C bounds how long
//!   the *client* waits for one call. That the backend applies no request the
//!   client abandoned later than C after it was sent is **assumed**, not
//!   shown: SDD §22.18 rests the counters sidecar on the same assumption
//!   ("its delay is assumed bounded by the backend"), and nothing in this code
//!   or the SDD proves it. A `StorageApi` write the backend applies after
//!   entry + C is outside Q, and is a recorded residual.
//! - **A LocalDisk filesystem step already on the blocking pool at entry.** A
//!   late `put_ext_blob`, rename or remove (at most mirror workers + 3 blob
//!   writes) finishes when that one call does. A filesystem stalled longer
//!   than Q is a recorded residual (`crate::dr::hold`'s writer audit).
//! - **A shutdown that overruns its own budgets.** G is the bound graceful
//!   shutdown keeps under the configured settings. A shutdown that overruns
//!   it, a stalled post-server step for example, is bounded only by the
//!   pod's `terminationGracePeriodSeconds`, after which the kubelet kills it.
//! - **A frame already in the kernel's send queue** (SDD §15.3 *Vendored
//!   fred* change 9). fred's 45 s wait ends when a frame reaches the socket;
//!   from there, when the store runs it is TCP's and the store's, and no
//!   client-side bound covers it. A network or store stall that holds such a
//!   frame past Q is a recorded residual.
//!
//! # `T_safe`
//!
//! `T_safe = detected_at_ms + Q + L`, where L is how long the backup store
//! may answer a by-address read from a cache ([`not_found_cache_lifetime`]):
//! not found for an object written after the cached read, or the earlier
//! content of one rewritten or deleted after it. A listing is fresh. An
//! absence that would change a Job's outcome is acted on only when observed
//! at or after `T_safe` (`super::absence`).
//!
//! Q already outlasts every late backup write within C, so L adds nothing to
//! that cover: it only lets a stale cached by-address answer expire, a cached
//! "not found" included. With L = 0
//! (`checkpoint.notFoundCacheSecs: 0`), `T_safe` is Q's end, which still
//! follows every late write within C. Nor does L cover the residual above: a
//! write the backend applies after entry + C can land after `T_safe` too, and
//! then change what an absence observed at `T_safe` showed.
//!
//! [`LOCAL_PUBLISH_WORST_CASE`]: crate::checkpoint::store::LOCAL_PUBLISH_WORST_CASE

use std::time::Duration;

use crate::checkpoint::CheckpointAdapter;
use crate::config::{Config, DR_HOLD_POLL_INTERVAL_MS_MAX};
use crate::dr::hold::{hold_write_bound, join_teardown_write_bound, STALE_VERDICT_THRESHOLD};

/// Q, term by term (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuietPeriod {
    /// How long a replica that cannot see the marker takes to reach
    /// stale-verdict entry: [`STALE_VERDICT_THRESHOLD`].
    pub stale_threshold: Duration,
    /// One hold-poll interval, at its largest accepted value
    /// ([`DR_HOLD_POLL_INTERVAL_MS_MAX`]).
    pub poll_interval: Duration,
    /// `D + T + 45 s`: [`hold_write_bound`].
    pub hold_write_bound: Duration,
    /// `5T + 1 s + 45 s`: [`join_teardown_write_bound`].
    pub join_teardown_bound: Duration,
    /// C, the backup store's call ceiling: [`adapter_call_ceiling`].
    pub adapter_call_ceiling: Duration,
    /// G, the longest a graceful shutdown already under way takes:
    /// [`shutdown_bound`].
    pub shutdown_bound: Duration,
}

impl QuietPeriod {
    /// Q for an RPC deadline D (`LIVE_EDIT_RPC_DEADLINE_MS`), a Valkey
    /// command timeout T (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`, > 0), a
    /// backup-store call ceiling C ([`adapter_call_ceiling`]) and a graceful
    /// shutdown bound G ([`shutdown_bound`]).
    pub fn derive(
        rpc_deadline: Duration,
        command_timeout: Duration,
        adapter_call_ceiling: Duration,
        shutdown_bound: Duration,
    ) -> Self {
        Self {
            stale_threshold: STALE_VERDICT_THRESHOLD,
            poll_interval: Duration::from_millis(DR_HOLD_POLL_INTERVAL_MS_MAX),
            hold_write_bound: hold_write_bound(rpc_deadline, command_timeout),
            join_teardown_bound: join_teardown_write_bound(command_timeout),
            adapter_call_ceiling,
            shutdown_bound,
        }
    }

    /// Q for `config`'s RPC deadline, command timeout, backup adapter and
    /// shutdown settings.
    pub fn from_config(config: &Config) -> Self {
        Self::derive(
            Duration::from_millis(config.rpc_deadline_ms),
            Duration::from_millis(config.valkey_command_timeout_ms),
            adapter_call_ceiling(config),
            shutdown_bound(config),
        )
    }

    /// The after-entry term: the largest of the two write bounds, the backup
    /// store's call ceiling and the graceful shutdown bound.
    pub fn after_entry(&self) -> Duration {
        self.hold_write_bound
            .max(self.join_teardown_bound)
            .max(self.adapter_call_ceiling)
            .max(self.shutdown_bound)
    }

    /// Q itself: the time to reach entry plus the after-entry term,
    /// saturating.
    pub fn total(&self) -> Duration {
        self.stale_threshold
            .saturating_add(self.poll_interval)
            .saturating_add(self.after_entry())
    }
}

/// C: how long after hold entry a backup-store write a replica sent before
/// entry can still land, within the assumption the module docs name.
/// `Config::checkpoint_adapter_call_ceiling` for the persistent adapters: the
/// enforced per-call ceiling for `StorageApi` (180 s at the default request
/// timeout), and [`LOCAL_PUBLISH_WORST_CASE`] for `LocalDisk`. Zero for
/// `Disabled`, which has no backup store, and for `InMemory`, whose writes
/// stop with the replica task that makes them.
///
/// [`LOCAL_PUBLISH_WORST_CASE`]: crate::checkpoint::store::LOCAL_PUBLISH_WORST_CASE
pub fn adapter_call_ceiling(config: &Config) -> Duration {
    match config.checkpoint_adapter {
        CheckpointAdapter::StorageApi | CheckpointAdapter::LocalDisk => {
            config.checkpoint_adapter_call_ceiling()
        }
        CheckpointAdapter::Disabled | CheckpointAdapter::InMemory => Duration::ZERO,
    }
}

/// G: the longest a graceful shutdown under `config` takes from SIGTERM,
/// `Config::minimum_termination_grace` (76.25 s at the defaults). A replica
/// already shutting down when the hold began writes until its shutdown ends.
pub fn shutdown_bound(config: &Config) -> Duration {
    config.minimum_termination_grace()
}

/// L: how old a by-address answer the backup store gives may be, an absence
/// or an earlier body alike. The `StorageApi` adapter's
/// `LIVE_EDIT_CHECKPOINT_STORAGE_API_NOT_FOUND_CACHE_SECS`; zero for every
/// other adapter, which has no such cache.
pub fn not_found_cache_lifetime(config: &Config) -> Duration {
    match config.checkpoint_adapter {
        CheckpointAdapter::StorageApi => {
            Duration::from_secs(config.checkpoint_storage_api_not_found_cache_secs)
        }
        CheckpointAdapter::Disabled
        | CheckpointAdapter::InMemory
        | CheckpointAdapter::LocalDisk => Duration::ZERO,
    }
}

/// A Job's two deadlines on the store's clock, in milliseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreDeadlines {
    /// The marker's `detected_at_ms`.
    pub detected_at_ms: u64,
    /// `detected_at_ms` + Q: no write before the store's `TIME` reaches it.
    pub quiet_ends_at_ms: u64,
    /// `quiet_ends_at_ms` + L: `T_safe`, from which an absence may change
    /// the outcome.
    pub t_safe_ms: u64,
}

impl StoreDeadlines {
    /// The deadlines for a hold detected at `detected_at_ms`, with quiet
    /// period `quiet` and by-address cache lifetime `not_found_cache` (L),
    /// saturating.
    pub fn new(detected_at_ms: u64, quiet: Duration, not_found_cache: Duration) -> Self {
        let quiet_ends_at_ms = detected_at_ms.saturating_add(duration_ms(quiet));
        Self {
            detected_at_ms,
            quiet_ends_at_ms,
            t_safe_ms: quiet_ends_at_ms.saturating_add(duration_ms(not_found_cache)),
        }
    }
}

/// `d` in whole milliseconds, saturating at `u64::MAX`.
pub(crate) fn duration_ms(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Duration = Duration::from_secs(1);

    /// G at the default shutdown settings (drain 5 s, pacing window 20 s,
    /// overall budget 30 s) with D = 30 s: 5 + 0.25 + 35 + 30 + 6 = 76.25 s.
    const G_AT_DEFAULTS: Duration = Duration::from_millis(76_250);

    /// At the defaults (D = 30 s, T = 5 s, StorageApi's C = 180 s, G =
    /// 76.25 s): 30 + 6 + max(30 + 5 + 45, 5·5 + 46, 180, 76.25) = 216 s.
    #[test]
    fn quiet_period_at_the_storage_api_defaults_is_216_seconds() {
        let q = QuietPeriod::derive(30 * S, 5 * S, 180 * S, G_AT_DEFAULTS);
        assert_eq!(q.stale_threshold, 30 * S);
        assert_eq!(q.poll_interval, 6 * S);
        assert_eq!(q.hold_write_bound, 80 * S);
        assert_eq!(q.join_teardown_bound, 71 * S);
        assert_eq!(q.adapter_call_ceiling, 180 * S);
        assert_eq!(q.shutdown_bound, G_AT_DEFAULTS);
        assert_eq!(q.total(), 216 * S);
    }

    /// With a local adapter's C (5 s) or none, the hold's write bound sets
    /// the after-entry term: 116 s at the defaults.
    #[test]
    fn quiet_period_without_a_remote_store_is_116_seconds() {
        assert_eq!(
            QuietPeriod::derive(30 * S, 5 * S, 5 * S, G_AT_DEFAULTS).total(),
            116 * S
        );
        assert_eq!(
            QuietPeriod::derive(30 * S, 5 * S, Duration::ZERO, G_AT_DEFAULTS).total(),
            116 * S
        );
    }

    /// The join teardown sets the after-entry term once T is large:
    /// 5·20 + 46 = 146 s, against 95 s for the hold's write bound.
    #[test]
    fn quiet_period_takes_the_join_teardown_when_it_is_larger() {
        let q = QuietPeriod::derive(30 * S, 20 * S, 5 * S, G_AT_DEFAULTS);
        assert_eq!(q.after_entry(), 146 * S);
        assert_eq!(q.total(), 182 * S);
    }

    /// A short RPC deadline with the default T: the join teardown (5·5 + 46 =
    /// 71 s) is above the hold's write bound (10 + 5 + 45 = 60 s) and G at
    /// that deadline (5 + 0.25 + 25 + 30 + 6 = 66.25 s), and sets the
    /// after-entry term, so Q is 30 + 6 + 71 = 107 s. Q has no lock-lapse
    /// term (`T + 75 s`, which would make it 116 s here).
    #[test]
    fn quiet_period_with_a_short_deadline_takes_the_join_teardown_with_no_lock_term() {
        let q = QuietPeriod::derive(10 * S, 5 * S, Duration::ZERO, Duration::from_millis(66_250));
        assert_eq!(
            (q.hold_write_bound, q.join_teardown_bound),
            (60 * S, 71 * S)
        );
        assert_eq!(
            q.after_entry(),
            q.join_teardown_bound,
            "the join teardown sets the after-entry term"
        );
        assert_eq!(q.total(), 107 * S);
    }

    /// Unbounded D: Q grows with it, so it always covers stale + poll + D +
    /// T + 45 s.
    #[test]
    fn quiet_period_covers_a_large_deadline() {
        // G at D = 600 s: 5 + 0.25 + 605 + 30 + 6 = 646.25 s, below D + T + 45 s.
        let q = QuietPeriod::derive(600 * S, 5 * S, 180 * S, Duration::from_millis(646_250));
        assert_eq!(q.total(), (30 + 6 + 600 + 5 + 45) * S);
        assert_eq!(q.total(), 686 * S);
    }

    #[test]
    fn deadlines_add_q_then_l() {
        let d = StoreDeadlines::new(1_000, 116 * S, 900 * S);
        assert_eq!(d.quiet_ends_at_ms, 117_000);
        assert_eq!(d.t_safe_ms, 1_017_000);
        let max = StoreDeadlines::new(u64::MAX - 1, 116 * S, Duration::ZERO);
        assert_eq!(max.quiet_ends_at_ms, u64::MAX);
    }
}
