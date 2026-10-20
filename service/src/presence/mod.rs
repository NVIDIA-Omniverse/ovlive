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

mod fanout;
mod relay;
#[cfg(any(test, feature = "test-fault-injection"))]
mod seam_gate;
pub mod session_presence;
pub mod sweeper;

#[cfg(feature = "test-support")]
pub use fanout::RECV_BATCH;
pub use fanout::{PresenceReceiver, PresenceRecvError};
pub use relay::{DrainBatch, DrainEnd, PubsubRelayReceiver};
#[cfg(any(test, feature = "test-fault-injection"))]
pub use seam_gate::PresenceSeamGate;
#[cfg(feature = "test-support")]
pub use session_presence::relay_presence_message_for_bench;
pub use session_presence::{
    publish_presence_event, spawn_presence_subscriber, subscribe_presence_channel,
    unsubscribe_presence_channel, ChannelJoin, ChannelSubscribe, ChannelUnsubscribe, ChannelWait,
    ConnectedUser, LocalPresenceRouter, PresenceEvent, DEFAULT_BROADCAST_CHANNEL_CAPACITY,
};
pub use sweeper::{spawn_presence_sweeper, SweeperBudget};
#[cfg(any(test, feature = "test-fault-injection"))]
pub use sweeper::{spawn_presence_sweeper_with_seam, ObservedSweepPass, PresenceSweeperSeam};

/// Default presence-lease TTL in seconds (b.w29).
///
/// Used as the fallback for `LIVE_EDIT_PRESENCE_LEASE_TTL_SECS` in
/// `Config::from_env`. Raised 30 → 60 in b.w29: at the pre-fix ratio of
/// `TTL=30 / refresh=10 / sweep=15` a 20 s tokio scheduling pause was
/// enough to evict a healthy session. 60 s preserves the same recovery
/// upper bound (`sweep_interval + lease_ttl ≈ 75 s`) while quadrupling
/// the headroom against scheduler hiccups.
///
/// Tied to [`DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS`] via
/// `Config::validate_presence_lease_budgets` (TTL ≥ 2 × refresh).
pub const DEFAULT_PRESENCE_LEASE_TTL_SECS: u64 = 60;

/// Default presence-lease refresh interval in seconds (b.w29).
///
/// Used as the fallback for `LIVE_EDIT_PRESENCE_REFRESH_INTERVAL_SECS`
/// in `Config::from_env`. Raised 10 → 15 in b.w29 so the refresh
/// interval matches the sweeper interval: every sweep tick observes a
/// lease that was refreshed within the past sweep window, so the
/// sweeper's "lease missing ⇒ evict" rule cannot fire on a healthy
/// session unless that session has also missed at least one refresh
/// AND has not yet recovered before the next sweep.
pub const DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS: u64 = 15;

/// Default `JoinSession` client keepalive interval in milliseconds.
///
/// Used as the fallback for `LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS` in
/// `Config::from_env`. A join whose client opted in to keepalives is told
/// this interval in `SessionConnected.keepalive_interval_ms` and is ended
/// after three intervals with no data from the client: 45 s at this default,
/// inside the ~75 s bound a crashed replica's members are removed in (sweep
/// interval + lease TTL), and two lost keepalives are tolerated. Equal to
/// [`DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS`] by choice, not by any rule:
/// the deadline acts on the join's own replica and composes with the lease
/// independently.
pub const DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS: u64 = 15_000;

/// Smallest non-zero `LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS`
/// `Config::from_env` accepts, in milliseconds.
///
/// `1..MIN_PRESENCE_KEEPALIVE_INTERVAL_MS` is refused at startup; `0` (the
/// keepalive off) is not. The interval is the only presence knob in
/// milliseconds, so the values it refuses are the ones a value meant as
/// seconds produces: `15` would end every opted-in join after 45 ms of
/// silence, fleet-wide. 1000 is the smallest interval the conformance suite
/// runs the keepalive at.
pub const MIN_PRESENCE_KEEPALIVE_INTERVAL_MS: u64 = 1_000;
