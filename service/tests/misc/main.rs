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

// Consolidated integration-test binary: small top-level tests that don't
// each warrant their own binary. Bundling them amortizes the per-binary
// linker overhead.
//
// Do NOT add `proptest!` modules here — proptest's thread pool and
// `#[tokio::test(flavor = "multi_thread")]` runtimes deadlock at process
// exit when they share a binary (see CLAUDE.md "Proptest Isolation").
// Proptest lives in `tests/proptests.rs` (its own binary) by design.

// Share the integration-suite common helpers (b.ohv item 8 — single
// canonical `Config` constructor in `tests/common/mod.rs`). The misc
// binary intentionally pulls in the same `common::test_config()` so the
// tests below don't need to redefine the Config field-by-field.
#[path = "../common/mod.rs"]
mod common;

mod auth_builder;
mod config_validation;
mod ompf_adversarial_corpus;
mod oodf_bridge_visibility_test;
mod otlp_tls_export_binary;
mod service_token_startup_warns;
mod startup_log_binary;
mod tail_codec_adversarial_corpus;
mod test_only_unvalidated_tokens;
