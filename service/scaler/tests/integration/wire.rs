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

//! The scaler's contracts with its neighbours: the `/scaling/v1` snapshot's
//! compatibility rules (shared with the service), `LIVE_EDIT_SCALER_*` as
//! the chart renders them, and the label vocabularies of the scaler's own
//! metrics, which dashboards and alerts are written against.

use std::collections::{BTreeSet, HashMap};
use std::time::Duration;

use live_edit_scaler::config::{
    parse_cpu_quantity, CpuTarget, ScalerConfig, ScalerConfigError, ENV_CPU_REQUEST,
    ENV_CPU_TARGET_PERCENT, ENV_DEPLOYMENT, ENV_MAX_REPLICAS, ENV_MIN_REPLICAS, ENV_NAMESPACE,
    ENV_OCCUPANCY_HEADROOM_PERCENT, ENV_POD_DELETION_COST, ENV_POLL_INTERVAL_SECS,
    ENV_SCALE_DOWN_WINDOW_SECS, ENV_SERVICE,
};
use live_edit_scaler::decide::BlockReason;
use live_edit_scaler::obs::{Heartbeat, ScalerMetrics, POLL_ERROR_STAGES};
use live_edit_scaler::snapshot::{ScalingSnapshot, SNAPSHOT_VERSION};

use crate::common::{label_values, render, sample};

// ── /scaling/v1 ────────────────────────────────────────────────────────────

/// Tolerant parsing: a body written by an older build, which lacks fields
/// this build added, parses with each missing field at its default.
#[test]
fn a_snapshot_without_newer_fields_parses_with_defaults() {
    let older: ScalingSnapshot =
        serde_json::from_str(r#"{ "version": 1, "ready": true, "subscriptions_admitted": 2 }"#)
            .expect("an older body must parse");
    assert_eq!(
        older,
        ScalingSnapshot {
            version: 1,
            ready: true,
            subscriptions_admitted: 2,
            ..ScalingSnapshot::default()
        }
    );
    assert_eq!(
        older.cpu_seconds_total, None,
        "no CPU reading, not zero CPU"
    );

    let empty: ScalingSnapshot = serde_json::from_str("{}").expect("even an empty object");
    assert_eq!(empty, ScalingSnapshot::default());
}

/// Tolerant parsing the other way: a body written by a newer build, with
/// fields this build does not know, parses and keeps the known ones.
#[test]
fn a_snapshot_with_unknown_fields_parses_and_keeps_the_known_ones() {
    let newer: ScalingSnapshot = serde_json::from_str(
        r#"{ "version": 1, "catchups_pending": 7, "added_later": [1, 2], "also_later": { "x": true } }"#,
    )
    .expect("a newer body must parse");
    assert_eq!(newer.catchups_pending, 7);
}

/// What one build writes, the same build reads back unchanged, under the
/// field names the wire carries.
#[test]
fn a_snapshot_round_trips_under_its_wire_names() {
    let written = ScalingSnapshot {
        version: SNAPSHOT_VERSION,
        ready: true,
        shutting_down: true,
        uptime_secs: 61,
        subscriptions_admitted: 2,
        admission_waiters: 3,
        max_subscriptions: 4,
        catchups_pending: 5,
        catchups_completed_total: 6,
        cpu_seconds_total: Some(7.5),
        catchup_read_concurrency: 8,
        max_concurrent_catchups: 9,
    };
    let json = serde_json::to_value(&written).expect("serializes");
    let names: BTreeSet<&str> = json
        .as_object()
        .expect("an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        names,
        BTreeSet::from([
            "version",
            "ready",
            "shutting_down",
            "uptime_secs",
            "subscriptions_admitted",
            "admission_waiters",
            "max_subscriptions",
            "catchups_pending",
            "catchups_completed_total",
            "cpu_seconds_total",
            "catchup_read_concurrency",
            "max_concurrent_catchups",
        ]),
        "the v1 field names are a contract between builds: additive only"
    );
    let read: ScalingSnapshot = serde_json::from_value(json).expect("parses back");
    assert_eq!(read, written);
}

// ── LIVE_EDIT_SCALER_* ─────────────────────────────────────────────────────

/// The three names and two bounds the chart always renders.
fn required() -> HashMap<&'static str, String> {
    HashMap::from([
        (ENV_NAMESPACE, "editing".to_owned()),
        (ENV_DEPLOYMENT, "live-edit-service".to_owned()),
        (ENV_SERVICE, "live-edit-service".to_owned()),
        (ENV_MIN_REPLICAS, "3".to_owned()),
        (ENV_MAX_REPLICAS, "10".to_owned()),
    ])
}

fn load(vars: &HashMap<&'static str, String>) -> Result<ScalerConfig, ScalerConfigError> {
    ScalerConfig::from_lookup(|name| vars.get(name).cloned())
}

/// With only the required variables, every other setting takes its
/// documented default, and the CPU term is off.
#[test]
fn scaler_config_with_only_the_required_variables_takes_the_defaults() {
    let cfg = load(&required()).expect("the required set is enough");
    assert_eq!(
        cfg,
        ScalerConfig {
            namespace: "editing".to_owned(),
            deployment: "live-edit-service".to_owned(),
            service: "live-edit-service".to_owned(),
            min_replicas: 3,
            max_replicas: 10,
            cpu: None,
            poll_interval: Duration::from_secs(5),
            catchup_target: Duration::from_secs(10),
            occupancy_headroom_percent: 80,
            scale_down_window: Duration::from_secs(600),
            scale_down_interval: Duration::from_secs(60),
            pod_deletion_cost: false,
            http_port: 9090,
        }
    );
}

/// Each refusal names its variable, and an empty value (which the chart
/// renders for an unset value) counts as unset.
#[test]
fn scaler_config_refusals_name_the_variable() {
    let name_of = |e: ScalerConfigError| match e {
        ScalerConfigError::Missing { name, .. } | ScalerConfigError::Invalid { name, .. } => name,
    };
    for missing in [
        ENV_NAMESPACE,
        ENV_DEPLOYMENT,
        ENV_SERVICE,
        ENV_MIN_REPLICAS,
        ENV_MAX_REPLICAS,
    ] {
        for value in [None, Some("   ")] {
            let mut vars = required();
            match value {
                None => vars.remove(missing),
                Some(v) => vars.insert(missing, v.to_owned()),
            };
            let err = load(&vars).expect_err("a required variable is missing");
            assert!(
                matches!(err, ScalerConfigError::Missing { .. }),
                "{missing}={value:?}: {err}"
            );
            assert_eq!(name_of(err), missing);
        }
    }
    for (var, value) in [
        (ENV_MIN_REPLICAS, "0"),
        (ENV_MAX_REPLICAS, "2"), // below the minimum of 3
        (ENV_MIN_REPLICAS, "three"),
        (ENV_POLL_INTERVAL_SECS, "0"),
        (ENV_OCCUPANCY_HEADROOM_PERCENT, "0"),
        (ENV_OCCUPANCY_HEADROOM_PERCENT, "101"),
        (ENV_POD_DELETION_COST, "yes"),
    ] {
        let mut vars = required();
        vars.insert(var, value.to_owned());
        let err = load(&vars).expect_err("an invalid value is refused");
        assert!(
            matches!(err, ScalerConfigError::Invalid { .. }),
            "{var}={value}: {err}"
        );
        assert_eq!(name_of(err), var, "{var}={value}");
    }

    let mut vars = required();
    vars.insert(ENV_SCALE_DOWN_WINDOW_SECS, "0".to_owned());
    vars.insert(ENV_POD_DELETION_COST, "TRUE".to_owned());
    let cfg = load(&vars).expect("a zero window and an upper-case boolean are accepted");
    assert_eq!(cfg.scale_down_window, Duration::ZERO);
    assert!(cfg.pod_deletion_cost);
}

/// A CPU target needs the server container's CPU request, as the quantity
/// the chart copies from `resources.requests.cpu`.
#[test]
fn scaler_config_cpu_target_needs_the_cpu_request() {
    let mut vars = required();
    vars.insert(ENV_CPU_TARGET_PERCENT, "80".to_owned());
    let err = load(&vars).expect_err("a CPU target without a request");
    assert!(
        matches!(err, ScalerConfigError::Missing { name, .. } if name == ENV_CPU_REQUEST),
        "{err}"
    );

    vars.insert(ENV_CPU_REQUEST, "500m".to_owned());
    assert_eq!(
        load(&vars).expect("a target with its request").cpu,
        Some(CpuTarget {
            request_cores: 0.5,
            target_percent: 80
        })
    );

    vars.insert(ENV_CPU_REQUEST, "1Gi".to_owned());
    assert!(
        matches!(load(&vars), Err(ScalerConfigError::Invalid { name, .. }) if name == ENV_CPU_REQUEST)
    );
}

/// Kubernetes CPU quantities: plain cores or millicores, positive only.
#[test]
fn parse_cpu_quantity_reads_cores_and_millicores_and_refuses_the_rest() {
    for (raw, cores) in [
        ("500m", Some(0.5)),
        ("2", Some(2.0)),
        ("0.25", Some(0.25)),
        (" 750m ", Some(0.75)),
        ("1500m", Some(1.5)),
        ("0", None),
        ("0m", None),
        ("", None),
        ("m", None),
        ("-1", None),
        ("1k", None),
        ("1e3", None),
        ("2 cores", None),
    ] {
        assert_eq!(parse_cpu_quantity(raw), cores, "{raw:?}");
    }
}

// ── The scaler's own metrics ───────────────────────────────────────────────

/// The `reason` and `stage` vocabularies are pre-registered at zero, so an
/// alert on a rate has a series before the first event, and are exactly the
/// documented values. `rollout` is no longer one of them: a scale-down held
/// during a rollout is `unsettled`, and the scaler never reads the
/// Deployment, so it has no `rollout` stage either.
///
/// The vocabularies are restated as literals and compared with the exported
/// constants, so a rename shows up here as an operator-facing change.
#[test]
fn scaler_metric_vocabularies_are_pre_registered_at_zero_and_exactly_the_documented_ones() {
    const REASONS: [&str; 5] = ["startup", "window", "unsettled", "interval", "feasibility"];
    const STAGES: [&str; 5] = [
        "endpoints",
        "scale",
        "snapshot",
        "deletion_cost",
        "patch_scale",
    ];
    let set = |values: &[&str]| {
        values
            .iter()
            .map(|v| v.to_string())
            .collect::<BTreeSet<_>>()
    };

    assert_eq!(
        BlockReason::ALL
            .iter()
            .map(|r| r.label())
            .collect::<Vec<_>>(),
        REASONS
    );
    assert_eq!(POLL_ERROR_STAGES, STAGES);

    let text = render(&ScalerMetrics::new());
    const BLOCKED: &str = "live_edit_scaler_scale_down_blocked_total";
    const ERRORS: &str = "live_edit_scaler_poll_errors_total";
    assert_eq!(label_values(&text, BLOCKED, "reason"), set(&REASONS));
    assert_eq!(label_values(&text, ERRORS, "stage"), set(&STAGES));
    for reason in REASONS {
        assert_eq!(
            sample(&text, BLOCKED, &[("reason", reason)]),
            Some(0.0),
            "{reason}"
        );
    }
    for stage in STAGES {
        assert_eq!(
            sample(&text, ERRORS, &[("stage", stage)]),
            Some(0.0),
            "{stage}"
        );
    }

    const DECISIONS: &str = "live_edit_scaler_decisions_total";
    for (direction, reason) in [
        ("up", "min"),
        ("up", "cpu"),
        ("up", "catchup"),
        ("up", "occupancy"),
        ("down", "below_target"),
        ("none", "at_target"),
        ("none", "scale_down_blocked"),
        ("none", "no_data"),
    ] {
        assert_eq!(
            sample(
                &text,
                DECISIONS,
                &[("direction", direction), ("reason", reason)]
            ),
            Some(0.0),
            "{direction}/{reason}"
        );
    }
}

/// `/healthz` follows the poll loop's heartbeat: fresh until the staleness
/// bound passes without a beat, fresh again after one. The bound is six
/// polls, never under a minute.
#[tokio::test(start_paused = true)]
async fn the_heartbeat_goes_stale_without_a_beat_and_recovers_with_one() {
    assert_eq!(
        Heartbeat::stale_after_for(Duration::from_secs(5)),
        Duration::from_secs(60)
    );
    assert_eq!(
        Heartbeat::stale_after_for(Duration::from_secs(20)),
        Duration::from_secs(120)
    );

    let heartbeat = Heartbeat::new(Duration::from_secs(60));
    assert!(heartbeat.is_fresh(), "fresh at the start");
    tokio::time::advance(Duration::from_secs(60)).await;
    assert!(heartbeat.is_fresh(), "fresh up to the bound");
    tokio::time::advance(Duration::from_secs(1)).await;
    assert!(!heartbeat.is_fresh(), "stale past the bound");
    heartbeat.beat();
    assert!(heartbeat.is_fresh(), "a beat makes it fresh again");
}
