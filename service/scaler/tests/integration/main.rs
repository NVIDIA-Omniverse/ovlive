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

//! The scaler crate's tests, one binary through its public API. None needs
//! Valkey or a cluster: the Kubernetes API server and each server pod's
//! `/scaling/v1` are `wiremock` servers on loopback.
//!
//! - [`engine`] — the pure decision engine, polls driven by injected times.
//! - [`kube_client`] — the REST client's requests and parsing.
//! - [`poll_loop`] — one `Scaler` against a fake API server and fake pods.
//! - [`wire`] — the `/scaling/v1` snapshot's compatibility rules, the
//!   configuration and the scaler's own metric vocabularies.

mod common;
mod engine;
mod kube_client;
mod poll_loop;
mod wire;
