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

//! `GET /scaling/v1` on the metrics port: this replica's load, for the
//! autoscaler (`live-edit scaler`).
//!
//! **An internal contract between the service and its scaler, not a customer
//! API.** The body is [`live_edit_scaler::snapshot::ScalingSnapshot`] as JSON,
//! the one type both ends are built from; its module docs carry the
//! compatibility rules (additive only within `v1`, tolerant parsing).
//!
//! The snapshot is read from in-process atomics, so serving it costs no metric
//! collection — unlike `/metrics`, whose every scrape runs a full collect. The
//! counters live on [`Metrics`](crate::observability::metrics::Metrics) as
//! [`ScalingCounters`], moved by the same calls that move the metrics they
//! mirror, so the two cannot disagree.
//!
//! The metrics server starts before the service's dependencies are wired, so
//! the route answers 503 until startup registers a [`ScalingSnapshotSource`]
//! on the [`ScalingEndpoint`] (once the admission gate, the health reporter
//! and the graceful-shutdown handle exist). The scaler treats a 503 like a pod
//! that does not report: never counted as load.
//!
//! Exposure: the metrics port is unauthenticated, so any client that can reach
//! it can read this replica's aggregate counts (subscriptions, pending
//! catch-ups, CPU seconds, its limits). It carries no session, user or layer
//! identifier — the same exposure class as `/metrics`.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use async_trait::async_trait;
use live_edit_scaler::snapshot::{ScalingSnapshot, SNAPSHOT_VERSION};
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::pb::health_server::Health;
use tonic_health::pb::HealthCheckRequest;
use tonic_health::server::HealthService;

use crate::observability::metrics::SharedMetrics;

/// The `/scaling/v1` counters that no metric can be read back from.
///
/// Each is moved only through a [`Metrics`](crate::observability::metrics::Metrics)
/// method that also moves its metric: `add_subscriber_admitted`,
/// `begin_pending_catchup` and `end_pending_catchup`.
#[derive(Debug, Default)]
pub struct ScalingCounters {
    pub(crate) subscriptions_admitted: AtomicI64,
    pub(crate) catchups_pending: AtomicI64,
    pub(crate) catchups_completed: AtomicU64,
}

impl ScalingCounters {
    /// Subscriptions currently admitted.
    pub fn subscriptions_admitted(&self) -> u64 {
        non_negative(self.subscriptions_admitted.load(Ordering::Acquire))
    }

    /// First state-reading catch-ups selected and not yet acknowledged.
    pub fn catchups_pending(&self) -> u64 {
        non_negative(self.catchups_pending.load(Ordering::Acquire))
    }

    /// First state-reading catch-ups acknowledged since the process started.
    pub fn catchups_completed(&self) -> u64 {
        self.catchups_completed.load(Ordering::Acquire)
    }
}

/// Every increment is paired with a decrement on the same value, so a
/// negative reading is a transient of concurrent updates, never a real count.
fn non_negative(v: i64) -> u64 {
    u64::try_from(v).unwrap_or(0)
}

/// Produces this replica's snapshot. The production source is
/// [`ServiceScalingSource`]; a test registers its own.
#[async_trait]
pub trait ScalingSnapshotSource: Send + Sync + 'static {
    async fn snapshot(&self) -> ScalingSnapshot;
}

/// The slot `/scaling/v1` reads its source from: empty (the route answers 503)
/// until startup registers one, then fixed for the process's life.
#[derive(Default)]
pub struct ScalingEndpoint {
    source: OnceLock<Arc<dyn ScalingSnapshotSource>>,
}

impl std::fmt::Debug for ScalingEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalingEndpoint")
            .field("registered", &self.source.get().is_some())
            .finish()
    }
}

impl ScalingEndpoint {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the source. `false`, leaving the first one in place, if one
    /// was already registered.
    pub fn register(&self, source: Arc<dyn ScalingSnapshotSource>) -> bool {
        self.source.set(source).is_ok()
    }

    /// The current snapshot, or `None` before registration.
    pub async fn snapshot(&self) -> Option<ScalingSnapshot> {
        match self.source.get() {
            Some(source) => Some(source.snapshot().await),
            None => None,
        }
    }
}

/// The service's snapshot source, built in `main.rs` once its parts exist.
pub struct ServiceScalingSource {
    /// Holds the [`ScalingCounters`].
    pub metrics: SharedMetrics,
    /// Subscribes queued at the subscription ceiling right now.
    pub admission_waiters: Box<dyn Fn() -> u64 + Send + Sync>,
    /// The subscription ceiling, derived from the pod memory limit
    /// (`config::memory_plan`); never `0`.
    pub max_subscriptions: u64,
    /// Answers for the readiness service, sharing the reporter's status map.
    pub health: HealthService,
    /// The readiness service name the supervisor raises and lowers.
    pub readiness_service_name: String,
    /// Whether graceful shutdown has started ending streams. Production
    /// reads the service's `GracefulShutdown::begun()` in place: it fires
    /// when the paced stream ending begins (drain end + the GOAWAY head
    /// start), the instant the single service-token cancel used to fire.
    pub shutting_down: Box<dyn Fn() -> bool + Send + Sync>,
    /// When the process started, for `uptime_secs`.
    pub process_started: Instant,
    /// `LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY`.
    pub catchup_read_concurrency: u32,
    /// `LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS`; never `0`, which
    /// startup refuses.
    pub max_concurrent_catchups: u32,
    /// Process CPU seconds; production passes
    /// [`crate::observability::process_cpu::cpu_seconds_total`].
    pub cpu_seconds: fn() -> Option<f64>,
}

impl ServiceScalingSource {
    async fn ready(&self) -> bool {
        let request = tonic::Request::new(HealthCheckRequest {
            service: self.readiness_service_name.clone(),
        });
        match self.health.check(request).await {
            Ok(response) => response.into_inner().status == ServingStatus::Serving as i32,
            Err(_) => false,
        }
    }
}

#[async_trait]
impl ScalingSnapshotSource for ServiceScalingSource {
    async fn snapshot(&self) -> ScalingSnapshot {
        let counters = self.metrics.scaling_counters();
        ScalingSnapshot {
            version: SNAPSHOT_VERSION,
            ready: self.ready().await,
            shutting_down: (self.shutting_down)(),
            uptime_secs: self.process_started.elapsed().as_secs(),
            subscriptions_admitted: counters.subscriptions_admitted(),
            admission_waiters: (self.admission_waiters)(),
            max_subscriptions: self.max_subscriptions,
            catchups_pending: counters.catchups_pending(),
            catchups_completed_total: counters.catchups_completed(),
            cpu_seconds_total: (self.cpu_seconds)(),
            catchup_read_concurrency: self.catchup_read_concurrency,
            max_concurrent_catchups: self.max_concurrent_catchups,
        }
    }
}
