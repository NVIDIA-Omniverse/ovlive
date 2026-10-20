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

//! The `/scaling/v1` wire type: one server pod's load, as the scaler reads it.
//!
//! **An internal contract between the service and its scaler, not a customer
//! API.** Both ends are built from this one definition — the service
//! serializes it on its metrics port, the scaler parses it — so they cannot
//! drift within one build. Across builds (a rollout runs old and new pods side
//! by side) it stays compatible by two rules:
//!
//! - **Additive only within `v1`.** A new field is added with a default; no
//!   field is renamed, removed or given a new meaning. A change that cannot
//!   follow that rule is a new path (`/scaling/v2`), not a new `version`.
//! - **Tolerant parsing.** Every field defaults when absent (`#[serde(default)]`)
//!   and unknown fields are ignored, so a newer scaler reads an older pod and
//!   an older scaler reads a newer pod.
//!
//! A pod that does not serve the path at all (an older release, mid-rollout)
//! answers 404; one that has not finished starting answers 503. The scaler
//! treats both as not reporting: it never counts them as load.

use serde::{Deserialize, Serialize};

/// The HTTP path the service serves its snapshot on, on its metrics port.
pub const SCALING_SNAPSHOT_PATH: &str = "/scaling/v1";

/// The `version` this build writes.
pub const SNAPSHOT_VERSION: u32 = 1;

/// One server pod's load at the moment it answered.
///
/// Counts are this pod's own; the scaler sums them across the fleet. Every
/// field is read from in-process counters, so serving it costs no metrics
/// collection.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ScalingSnapshot {
    /// [`SNAPSHOT_VERSION`] of the writer. Informational: compatibility comes
    /// from the additive-only rule above, not from comparing versions.
    pub version: u32,
    /// The pod's readiness verdict (the gRPC readiness service is `SERVING`).
    pub ready: bool,
    /// The service's graceful shutdown has begun ending its streams.
    pub shutting_down: bool,
    /// Seconds since the service process started.
    pub uptime_secs: u64,
    /// `SubscribeReadLayer` subscriptions currently admitted.
    pub subscriptions_admitted: u64,
    /// Subscribes queued for admission at the subscription ceiling.
    pub admission_waiters: u64,
    /// The per-replica subscription ceiling, which the service derives from
    /// its pod memory limit. `0` means unlimited: only a replica of an older
    /// version, whose ceiling was a setting, reports it.
    pub max_subscriptions: u64,
    /// Subscriptions whose first catch-up that reads layer state has begun and
    /// has not yet been acknowledged: every catch-up regime except an
    /// up-to-date subscriber's.
    pub catchups_pending: u64,
    /// First catch-ups that read layer state and reached their caught-up
    /// acknowledgement, since the process started. A monotonic counter; the
    /// scaler reads its rate.
    pub catchups_completed_total: u64,
    /// CPU seconds (user plus system) the service process has used since it
    /// started. `None` where the platform does not expose it.
    pub cpu_seconds_total: Option<f64>,
    /// Catch-up atomic snapshot reads allowed at once on this replica
    /// (`LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY`).
    pub catchup_read_concurrency: u32,
    /// Catch-ups allowed to read and build at once on this replica
    /// (`LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS`). `0` means unlimited:
    /// only a replica of an older version, which accepted it, reports it.
    pub max_concurrent_catchups: u32,
}
