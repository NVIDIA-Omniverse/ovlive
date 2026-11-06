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

//! Scaffolding the hold tests share: verdict fixtures, a recording exit seam,
//! readers of the captured log, the hold-entry counter and readiness, and a
//! snapshot of the test's keyspace with the writes it can prove.
//!
//! Where the copies these replace differed, the shared one keeps the stricter
//! behaviour: the keyspace is listed with the cluster-aware `SCAN` of
//! `common::scan_keys`, and [`keyspace_writes`] counts any removed key as a
//! write unless the caller allows a key with a TTL to expire
//! ([`WriteRules::ttl_keys_may_expire`]).

use std::collections::BTreeMap;
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::KeysInterface;
use futures::future::BoxFuture;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tonic::transport::Channel;
use tonic::Request;
use tonic_health::pb::health_check_response::ServingStatus;
use tonic_health::server::HealthReporter;

use live_edit::dr::hold::{HoldObservation, HoldVerdict};
use live_edit::grpc::hold::HoldExit;
use live_edit::observability::metrics::test_util::sample_value;
use live_edit::observability::metrics::{DR_HOLD_REASONS, DR_HOLD_REASON_KEYSPACE_LOST};
use live_edit::storage::test_support::TestValkey;

use crate::grpc::log_capture::Captured;

/// The readiness service name `main.rs` registers.
pub(crate) const READINESS: &str = "nvidia.omniverse.live.v1alpha.LiveService";

// ── Verdicts ─────────────────────────────────────────────────────────────────

/// A `Held` observation naming `incident`, a lost keyspace and its evidence,
/// observed now.
pub(crate) fn held(incident: &str) -> HoldObservation {
    HoldObservation {
        verdict: HoldVerdict::Held {
            incident: Some(incident.to_owned()),
            reason: Some(DR_HOLD_REASON_KEYSPACE_LOST.to_owned()),
            evidence: Some("le:counter:object_id absent on an established store".to_owned()),
            detected_at_ms: Some(1_760_000_000_000),
        },
        at: Instant::now(),
    }
}

/// A `Serving` observation, observed now.
pub(crate) fn serving() -> HoldObservation {
    HoldObservation {
        verdict: HoldVerdict::Serving,
        at: Instant::now(),
    }
}

// ── The exit seam ────────────────────────────────────────────────────────────

/// The exit seam's record: each call's code and the instant of the call (the
/// virtual instant on a paused clock). The call returns at once.
pub(crate) struct RecordingExit {
    pub(crate) calls: mpsc::UnboundedSender<(i32, Instant)>,
}

impl HoldExit for RecordingExit {
    fn exit(&self, code: i32) -> BoxFuture<'static, ()> {
        let _ = self.calls.send((code, Instant::now()));
        Box::pin(async {})
    }
}

// ── The captured log and the metrics ─────────────────────────────────────────

/// The captured lines whose message starts with `prefix`.
pub(crate) fn lines_starting<'c>(
    captured: &'c Captured,
    prefix: &str,
) -> Vec<&'c serde_json::Value> {
    captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with(prefix))
        })
        .collect()
}

/// Every `live_edit_dr_hold_entered_total{reason}` cell of `text`, a rendered
/// registry or a `/metrics` scrape, in `DR_HOLD_REASONS` order. Panics on a
/// cell that is not materialised.
pub(crate) fn hold_entries(text: &str) -> Vec<(&'static str, u64)> {
    DR_HOLD_REASONS
        .iter()
        .map(|reason| {
            let n = sample_value(
                text,
                "live_edit_dr_hold_entered_total",
                &[("reason", reason)],
            )
            .unwrap_or_else(|| panic!("the reason={reason} cell is materialised"));
            (*reason, n as u64)
        })
        .collect()
}

// ── Readiness ────────────────────────────────────────────────────────────────

/// Where a test reads a health status from.
#[derive(Clone, Copy)]
pub(crate) enum HealthSource<'a> {
    /// The reporter, read in process as a health client reads it.
    Reporter(&'a HealthReporter),
    /// A server's health service, over the wire.
    Wire(&'a Channel),
}

/// The status `source` answers a health `Check` of `service` with.
pub(crate) async fn health_status(source: HealthSource<'_>, service: &str) -> i32 {
    let request = tonic_health::pb::HealthCheckRequest {
        service: service.to_owned(),
    };
    match source {
        HealthSource::Reporter(reporter) => {
            use tonic_health::pb::health_server::Health;
            tonic_health::server::HealthService::from_health_reporter(reporter.clone())
                .check(Request::new(request))
                .await
                .expect("the service is registered")
                .into_inner()
                .status
        }
        HealthSource::Wire(channel) => {
            tonic_health::pb::health_client::HealthClient::new(channel.clone())
                .check(request)
                .await
                .expect("the server answers health")
                .into_inner()
                .status
        }
    }
}

/// Wait until `source` answers readiness ([`READINESS`]) `want`, within
/// `within`.
pub(crate) async fn until_readiness(
    source: HealthSource<'_>,
    want: ServingStatus,
    within: Duration,
) {
    let deadline = Instant::now() + within;
    while health_status(source, READINESS).await != want as i32 {
        assert!(
            Instant::now() < deadline,
            "readiness did not reach {want:?} within {within:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

// ── The keyspace ─────────────────────────────────────────────────────────────

/// One key's state: its serialized value (`DUMP`) and its `PTTL` (-1: no
/// TTL).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KeyState {
    pub(crate) dump: Bytes,
    pub(crate) pttl: i64,
}

/// Every key under the fixture's prefix, with its state. A key that expired
/// between the `SCAN` and its reads is left out: it is gone.
pub(crate) async fn keyspace(tv: &TestValkey) -> BTreeMap<String, KeyState> {
    let keys = crate::common::scan_keys(tv.client(), &format!("{}*", tv.prefix())).await;
    let mut out = BTreeMap::new();
    for key in keys {
        let name = key.as_str().expect("keys are UTF-8").to_owned();
        let dump: Option<Bytes> = tv
            .client()
            .pool
            .dump(&name)
            .await
            .expect("DUMP must succeed");
        let pttl: i64 = tv
            .client()
            .pool
            .pttl(&name)
            .await
            .expect("PTTL must succeed");
        if let Some(dump) = dump {
            if pttl != -2 {
                out.insert(name, KeyState { dump, pttl });
            }
        }
    }
    out
}

/// What [`keyspace_writes`] counts as a write, beyond an added key or a
/// changed value, which always are.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WriteRules {
    /// How far a TTL must have fallen between the two snapshots for it not
    /// to count as refreshed. `0`: any TTL that grew was refreshed.
    pub(crate) ttl_must_fall_by_ms: i64,
    /// Whether a key that had a TTL may be missing from the second snapshot
    /// without that counting as a write: it may have expired. A removed key
    /// with no TTL is a write either way.
    pub(crate) ttl_keys_may_expire: bool,
}

impl WriteRules {
    /// Nothing may change: no key added, removed or changed, and no TTL set
    /// or grown.
    pub(crate) const STRICT: Self = Self {
        ttl_must_fall_by_ms: 0,
        ttl_keys_may_expire: false,
    };
}

/// Every way `after` differs from `before` that only a write explains under
/// `rules`, one line each.
pub(crate) fn keyspace_writes(
    before: &BTreeMap<String, KeyState>,
    after: &BTreeMap<String, KeyState>,
    rules: WriteRules,
) -> Vec<String> {
    let mut writes = Vec::new();
    for (key, now) in after {
        match before.get(key) {
            None => writes.push(format!("{key}: appeared")),
            Some(then) if then.dump != now.dump => writes.push(format!("{key}: value changed")),
            Some(then) if then.pttl < 0 && now.pttl >= 0 => {
                writes.push(format!("{key}: a TTL was set ({} ms)", now.pttl))
            }
            Some(then) if then.pttl >= 0 && now.pttl > then.pttl - rules.ttl_must_fall_by_ms => {
                writes.push(format!(
                    "{key}: TTL refreshed ({} ms then, {} ms now; it must fall by at least {} ms)",
                    then.pttl, now.pttl, rules.ttl_must_fall_by_ms
                ))
            }
            Some(_) => {}
        }
    }
    for (key, then) in before {
        if !after.contains_key(key) {
            if then.pttl < 0 {
                writes.push(format!("{key}: deleted, and it had no TTL to expire"));
            } else if !rules.ttl_keys_may_expire {
                writes.push(format!("{key}: removed ({} ms TTL left)", then.pttl));
            }
        }
    }
    writes
}
