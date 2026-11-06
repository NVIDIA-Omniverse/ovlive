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

//! The suite's own harness tests: the parts of the runner, the client and
//! provisioning that need a peer on the network or the built executable.
//! Pure logic is tested in each module's `#[cfg(test)]` block instead.
//!
//! - `client_calls`: deadlines on the wire (the `grpc-timeout` header, and
//!   the client ending a call no server answers), and what [`Client`] owns
//!   and cleans up, against [`fake_live::FakeLive`].
//! - `execute`: `runner::execute` with planted cases (their outcomes, their
//!   cleanup, and the cleanup's bound after an interrupt), and one real case
//!   (the gated `ErrorInfo` check in `layers`) against a scripted server.
//! - `readiness`: the preflight's readiness wait, through
//!   `runner::run_with_cases` with planted cases.
//! - `provisioning`: `deployment::provision` against a wiremock Storage API.
//! - `binary`: the `live-edit-conformance` executable end to end: argv, a
//!   provisioning failure's exit code, reports and output, and a real
//!   SIGTERM or SIGINT, sent with the system's `kill`, while the run waits.
//!
//! Every peer is in-process on 127.0.0.1; nothing here reaches a real
//! service. Apart from `binary`'s signal tests, the interrupt paths are
//! driven through the [`Interrupt`] future the runner and provisioning take.
//!
//! [`Client`]: live_edit_conformance::client::Client
//! [`Interrupt`]: live_edit_conformance::runner::Interrupt

mod binary;
mod client_calls;
mod execute;
mod fake_live;
mod provisioning;
mod readiness;
mod support;
