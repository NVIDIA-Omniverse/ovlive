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

//! The scaler's configuration: `LIVE_EDIT_SCALER_*`.
//!
//! [`ScalerConfig::from_env`] is the code authority for every variable here.
//! The chart sets them inline on the scaler pod (no ConfigMap), so a values
//! change rolls the scaler. The scaler reads none of the service's own
//! variables: `live-edit scaler` dispatches before the service configuration
//! loads.
//!
//! The Kubernetes API endpoint and credentials are not configuration: they come
//! from the pod's in-cluster environment ([`crate::kube::KubeClient::in_cluster`]).

use std::time::Duration;

/// Namespace of the service Deployment and its EndpointSlices. Required.
pub const ENV_NAMESPACE: &str = "LIVE_EDIT_SCALER_NAMESPACE";
/// Name of the service Deployment whose `/scale` the scaler owns. Required.
pub const ENV_DEPLOYMENT: &str = "LIVE_EDIT_SCALER_DEPLOYMENT";
/// Name of the Service selecting the server pods; its EndpointSlices carry
/// `kubernetes.io/service-name=<this>`. Required.
pub const ENV_SERVICE: &str = "LIVE_EDIT_SCALER_SERVICE";
/// Lower replica bound. Required, at least 1.
pub const ENV_MIN_REPLICAS: &str = "LIVE_EDIT_SCALER_MIN_REPLICAS";
/// Upper replica bound. Required, at least `MIN_REPLICAS`.
pub const ENV_MAX_REPLICAS: &str = "LIVE_EDIT_SCALER_MAX_REPLICAS";
/// Target CPU utilization of the server container's CPU request, in percent.
/// Default 0: the CPU term is off.
pub const ENV_CPU_TARGET_PERCENT: &str = "LIVE_EDIT_SCALER_CPU_TARGET_PERCENT";
/// The server container's CPU request as a Kubernetes quantity (`500m`, `2`).
/// Required when the CPU target is above 0.
pub const ENV_CPU_REQUEST: &str = "LIVE_EDIT_SCALER_CPU_REQUEST";
/// Seconds between polls. Default 5.
pub const ENV_POLL_INTERVAL_SECS: &str = "LIVE_EDIT_SCALER_POLL_INTERVAL_SECS";
/// Seconds the catch-up term sizes the fleet to drain the pending catch-ups
/// in. Default 10.
pub const ENV_CATCHUP_TARGET_SECS: &str = "LIVE_EDIT_SCALER_CATCHUP_TARGET_SECS";
/// Share of each replica's subscription ceiling the occupancy term fills the
/// fleet to, in percent. Default 80.
pub const ENV_OCCUPANCY_HEADROOM_PERCENT: &str = "LIVE_EDIT_SCALER_OCCUPANCY_HEADROOM_PERCENT";
/// Seconds the desired count must stay below the current count before a
/// scale-down. Default 600.
pub const ENV_SCALE_DOWN_WINDOW_SECS: &str = "LIVE_EDIT_SCALER_SCALE_DOWN_WINDOW_SECS";
/// Minimum seconds between two scale-downs, each of one pod. Default 60.
pub const ENV_SCALE_DOWN_INTERVAL_SECS: &str = "LIVE_EDIT_SCALER_SCALE_DOWN_INTERVAL_SECS";
/// `true` to mark the emptiest pod for removal before each scale-down
/// (needs `patch` on pods). Default `false`.
pub const ENV_POD_DELETION_COST: &str = "LIVE_EDIT_SCALER_POD_DELETION_COST";
/// Port of the scaler's own `/metrics` and `/healthz`. Default 9090.
pub const ENV_HTTP_PORT: &str = "LIVE_EDIT_SCALER_HTTP_PORT";

/// The CPU term's target: the HPA's `averageUtilization` against the server
/// container's CPU request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CpuTarget {
    /// The server container's CPU request, in cores.
    pub request_cores: f64,
    /// Target utilization of that request, in percent.
    pub target_percent: u32,
}

impl CpuTarget {
    /// CPU cores per pod at the target: `request × percent / 100`.
    pub fn target_cores(&self) -> f64 {
        self.request_cores * f64::from(self.target_percent) / 100.0
    }
}

/// The scaler's validated configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalerConfig {
    pub namespace: String,
    pub deployment: String,
    pub service: String,
    pub min_replicas: u32,
    pub max_replicas: u32,
    /// `None` when the CPU term is off.
    pub cpu: Option<CpuTarget>,
    pub poll_interval: Duration,
    pub catchup_target: Duration,
    pub occupancy_headroom_percent: u32,
    pub scale_down_window: Duration,
    pub scale_down_interval: Duration,
    pub pod_deletion_cost: bool,
    pub http_port: u16,
}

/// Why the configuration was refused. Each message names the variable.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ScalerConfigError {
    #[error("{name} is required: {what}")]
    Missing {
        name: &'static str,
        what: &'static str,
    },
    #[error("invalid {name} value {value:?}: {reason}")]
    Invalid {
        name: &'static str,
        value: String,
        reason: String,
    },
}

impl ScalerConfig {
    /// Read and validate `LIVE_EDIT_SCALER_*` from the process environment.
    pub fn from_env() -> Result<Self, ScalerConfigError> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    /// [`Self::from_env`] over any variable source, for tests.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, ScalerConfigError> {
        let vars = Vars(&lookup);
        let namespace = vars.required_string(
            ENV_NAMESPACE,
            "the namespace of the service Deployment the scaler resizes",
        )?;
        let deployment = vars.required_string(
            ENV_DEPLOYMENT,
            "the name of the service Deployment the scaler resizes",
        )?;
        let service = vars.required_string(
            ENV_SERVICE,
            "the name of the Service that selects the server pods",
        )?;
        let min_replicas = vars.required_u32(ENV_MIN_REPLICAS, "the lower replica bound")?;
        if min_replicas == 0 {
            return Err(invalid(
                ENV_MIN_REPLICAS,
                "0",
                "must be at least 1: the scaler never scales the service to zero",
            ));
        }
        let max_replicas = vars.required_u32(ENV_MAX_REPLICAS, "the upper replica bound")?;
        if max_replicas < min_replicas {
            return Err(invalid(
                ENV_MAX_REPLICAS,
                &max_replicas.to_string(),
                &format!("must be at least {ENV_MIN_REPLICAS} ({min_replicas})"),
            ));
        }

        let target_percent = vars.u32_or(ENV_CPU_TARGET_PERCENT, 0)?;
        let cpu = if target_percent == 0 {
            None
        } else {
            let raw = vars
                .get(ENV_CPU_REQUEST)
                .ok_or(ScalerConfigError::Missing {
                    name: ENV_CPU_REQUEST,
                    what: "the server container's CPU request (resources.requests.cpu), which the \
                       CPU target is a percentage of; set it, or set \
                       LIVE_EDIT_SCALER_CPU_TARGET_PERCENT to 0 to turn the CPU term off",
                })?;
            let request_cores = parse_cpu_quantity(&raw).ok_or_else(|| {
                invalid(
                    ENV_CPU_REQUEST,
                    &raw,
                    "expected a positive Kubernetes CPU quantity such as 500m or 2",
                )
            })?;
            Some(CpuTarget {
                request_cores,
                target_percent,
            })
        };

        let poll_interval = vars.secs_at_least(ENV_POLL_INTERVAL_SECS, 5, 1)?;
        let catchup_target = vars.secs_at_least(ENV_CATCHUP_TARGET_SECS, 10, 1)?;
        let occupancy_headroom_percent = vars.u32_or(ENV_OCCUPANCY_HEADROOM_PERCENT, 80)?;
        if !(1..=100).contains(&occupancy_headroom_percent) {
            return Err(invalid(
                ENV_OCCUPANCY_HEADROOM_PERCENT,
                &occupancy_headroom_percent.to_string(),
                "must be between 1 and 100",
            ));
        }
        let scale_down_window = vars.secs_at_least(ENV_SCALE_DOWN_WINDOW_SECS, 600, 0)?;
        let scale_down_interval = vars.secs_at_least(ENV_SCALE_DOWN_INTERVAL_SECS, 60, 0)?;
        let pod_deletion_cost = vars.bool_or(ENV_POD_DELETION_COST, false)?;
        let http_port = vars.u32_or(ENV_HTTP_PORT, 9090)?;
        let http_port = u16::try_from(http_port)
            .ok()
            .filter(|p| *p != 0)
            .ok_or_else(|| {
                invalid(
                    ENV_HTTP_PORT,
                    &http_port.to_string(),
                    "must be a TCP port between 1 and 65535",
                )
            })?;

        Ok(Self {
            namespace,
            deployment,
            service,
            min_replicas,
            max_replicas,
            cpu,
            poll_interval,
            catchup_target,
            occupancy_headroom_percent,
            scale_down_window,
            scale_down_interval,
            pod_deletion_cost,
            http_port,
        })
    }

    /// The startup INFO line: every setting the scaler runs with.
    pub fn log_startup(&self) {
        tracing::info!(
            namespace = %self.namespace,
            deployment = %self.deployment,
            service = %self.service,
            min_replicas = self.min_replicas,
            max_replicas = self.max_replicas,
            cpu_request_cores = self.cpu.map(|c| c.request_cores),
            cpu_target_percent = self.cpu.map_or(0, |c| c.target_percent),
            poll_interval_secs = self.poll_interval.as_secs(),
            catchup_target_secs = self.catchup_target.as_secs(),
            occupancy_headroom_percent = self.occupancy_headroom_percent,
            scale_down_window_secs = self.scale_down_window.as_secs(),
            scale_down_interval_secs = self.scale_down_interval.as_secs(),
            pod_deletion_cost = self.pod_deletion_cost,
            http_port = self.http_port,
            "live-edit scaler starting: it owns the service Deployment's replica count"
        );
    }
}

/// A Kubernetes CPU quantity in cores: a plain decimal (`2`, `0.5`) or
/// millicores (`500m`). `None` for anything else, zero or negative included.
/// The other quantity suffixes (`k`, exponents) are not CPU requests anyone
/// writes, and are refused rather than half-supported.
pub fn parse_cpu_quantity(raw: &str) -> Option<f64> {
    let raw = raw.trim();
    let (number, scale) = match raw.strip_suffix('m') {
        Some(milli) => (milli, 1_000.0),
        None => (raw, 1.0),
    };
    if number.is_empty() || !number.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return None;
    }
    let cores = number.parse::<f64>().ok()? / scale;
    (cores.is_finite() && cores > 0.0).then_some(cores)
}

fn invalid(name: &'static str, value: &str, reason: &str) -> ScalerConfigError {
    ScalerConfigError::Invalid {
        name,
        value: value.to_owned(),
        reason: reason.to_owned(),
    }
}

/// Variable reads over a lookup. An empty or all-whitespace value counts as
/// unset, so a chart that renders an empty string gets the default (or the
/// missing-variable error) rather than a parse error.
struct Vars<'a, F: Fn(&str) -> Option<String>>(&'a F);

impl<F: Fn(&str) -> Option<String>> Vars<'_, F> {
    fn get(&self, name: &str) -> Option<String> {
        (self.0)(name)
            .map(|v| v.trim().to_owned())
            .filter(|v| !v.is_empty())
    }

    fn required_string(
        &self,
        name: &'static str,
        what: &'static str,
    ) -> Result<String, ScalerConfigError> {
        self.get(name)
            .ok_or(ScalerConfigError::Missing { name, what })
    }

    fn required_u32(
        &self,
        name: &'static str,
        what: &'static str,
    ) -> Result<u32, ScalerConfigError> {
        let raw = self.required_string(name, what)?;
        parse_u32(name, &raw)
    }

    fn u32_or(&self, name: &'static str, default: u32) -> Result<u32, ScalerConfigError> {
        match self.get(name) {
            Some(raw) => parse_u32(name, &raw),
            None => Ok(default),
        }
    }

    fn secs_at_least(
        &self,
        name: &'static str,
        default: u32,
        floor: u32,
    ) -> Result<Duration, ScalerConfigError> {
        let secs = self.u32_or(name, default)?;
        if secs < floor {
            return Err(invalid(
                name,
                &secs.to_string(),
                &format!("must be at least {floor}"),
            ));
        }
        Ok(Duration::from_secs(u64::from(secs)))
    }

    fn bool_or(&self, name: &'static str, default: bool) -> Result<bool, ScalerConfigError> {
        match self.get(name) {
            None => Ok(default),
            Some(raw) if raw.eq_ignore_ascii_case("true") => Ok(true),
            Some(raw) if raw.eq_ignore_ascii_case("false") => Ok(false),
            Some(raw) => Err(invalid(name, &raw, "expected true or false")),
        }
    }
}

fn parse_u32(name: &'static str, raw: &str) -> Result<u32, ScalerConfigError> {
    raw.parse::<u32>()
        .map_err(|_| invalid(name, raw, "expected a non-negative integer"))
}
