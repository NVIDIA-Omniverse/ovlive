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

// Integration test binary: gRPC server, Ping, health checks, graceful shutdown, and metrics.

#[path = "../common/mod.rs"]
mod common;

mod auth_metrics_endpoint;
mod auth_negative_ttl_flip;
mod checkpoint;
mod grpc;
mod health;
mod health_supervisor_counter_seed_test;
mod health_supervisor_service_token_test;
mod health_supervisor_test;
mod metrics;
mod observability_metrics_test;
mod ping;
mod server_coexist;
mod service_token_rotation;
mod shutdown;
mod startup_bind_handoff_test;
mod storage;
