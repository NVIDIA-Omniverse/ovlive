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

//! Helpers shared by the scaler test modules.

use live_edit_scaler::obs::ScalerMetrics;
use live_edit_scaler::snapshot::ScalingSnapshot;

/// A warmed-up, Ready pod's snapshot with no load: an hour of uptime (past
/// the CPU warm-up), one catch-up read at a time, unlimited subscriptions.
pub fn idle_snapshot() -> ScalingSnapshot {
    ScalingSnapshot {
        version: 1,
        ready: true,
        uptime_secs: 3_600,
        catchup_read_concurrency: 1,
        ..ScalingSnapshot::default()
    }
}

/// The Prometheus text `ScalerMetrics` serves on the scaler's `/metrics`.
pub fn render(metrics: &ScalerMetrics) -> String {
    let mut buf = Vec::new();
    metrics
        .exporter()
        .export(&mut buf)
        .expect("the scaler's metrics must encode");
    String::from_utf8(buf).expect("the exposition is UTF-8")
}

/// The value of the sample of `metric` whose labels include every pair in
/// `labels`, or `None` when no such sample is exported.
pub fn sample(text: &str, metric: &str, labels: &[(&str, &str)]) -> Option<f64> {
    text.lines()
        .filter(|line| !line.starts_with('#'))
        .find_map(|line| {
            let (series, value) = line.rsplit_once(' ')?;
            let (name, rest) = match series.split_once('{') {
                Some((name, rest)) => (name, rest.trim_end_matches('}')),
                None => (series, ""),
            };
            if name != metric {
                return None;
            }
            let all = labels
                .iter()
                .all(|(k, v)| rest.split(',').any(|pair| pair == format!("{k}=\"{v}\"")));
            all.then(|| value.parse().expect("a sample value is a number"))
        })
}

/// Every value `label` takes on `metric`'s samples.
pub fn label_values(text: &str, metric: &str, label: &str) -> std::collections::BTreeSet<String> {
    let prefix = format!("{metric}{{");
    let key = format!("{label}=\"");
    text.lines()
        .filter_map(|line| line.strip_prefix(&prefix))
        .filter_map(|rest| {
            let start = rest.find(&key)? + key.len();
            let end = rest[start..].find('"')? + start;
            Some(rest[start..end].to_owned())
        })
        .collect()
}
