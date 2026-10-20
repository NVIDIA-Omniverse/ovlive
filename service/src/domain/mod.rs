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

//! Pure business logic.
//!
//! The `domain` module contains zero I/O and zero async code (SDD §3.1 / CLAUDE.md
//! "Module Dependency Rules"). It depends on no other `live_edit` subsystem —
//! `storage`, `grpc`, `auth`, `presence`, and `checkpoint` all sit downstream of
//! it — reaching outward only for `oodf_ffi`'s parsed-delta types and its pure
//! `parse_delta` (used by [`replay`]), the crate-root
//! [`ServiceError`](crate::error::ServiceError), and a handful of pure utility
//! crates (for example `percent-encoding` for the string transform in
//! [`base_path`], plus `ordered_float` and `thiserror`). All storage / network
//! effects live in the `storage` and `grpc` modules; every function here is a
//! pure transform of owned Rust values.

pub mod alloc_model;
pub mod base_path;
pub mod delta;
pub mod delta_write;
pub mod diff;
pub mod limits;
pub mod ompf;
pub mod pass_timing;
pub mod plan_index;
pub mod plan_journal;
pub mod replay;
pub mod time;
pub mod types;
pub mod validation;
