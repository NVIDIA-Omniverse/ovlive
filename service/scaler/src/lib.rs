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

//! The live-edit replica autoscaler, run as `live-edit scaler`.
//!
//! One scaler pod per release owns the service Deployment's `/scale`
//! subresource when the chart's `autoscaling.mode` is `scaler`. Every poll it:
//!
//! 1. lists the service's EndpointSlices ([`kube`]) — the server pods, with
//!    their addresses, readiness and termination state;
//! 2. reads `spec.replicas` and `status.replicas` from the Deployment's
//!    `/scale` subresource, which carries no pod template;
//! 3. reads each non-terminating pod's [`snapshot::ScalingSnapshot`] from
//!    `GET /scaling/v1` on its metrics port;
//! 4. feeds all of it to the pure decision engine ([`decide`]), which combines
//!    a CPU term, a catch-up-queue term and a subscription-occupancy term;
//! 5. applies the decision ([`run`]): a scale-up at once, a scale-down only
//!    after a full window below the current count, only while the fleet is
//!    settled, one pod per interval, and only while no catch-up queue has
//!    built up.
//!
//! The scaler is fail-static: when it is down, or the API server or every pod
//! fails to answer, the replica count stays where it is. It keeps no state
//! outside its process, so a restart forgets its window history and refuses
//! to scale down until it has observed a full window again.
//!
//! [`config`] reads `LIVE_EDIT_SCALER_*`; [`obs`] serves the scaler's own
//! `/metrics` (`live_edit_scaler_*`) and `/healthz`.

pub mod config;
pub mod decide;
pub mod kube;
pub mod obs;
pub mod run;
pub mod snapshot;

pub use run::main_from_env;
