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

//! Clock helpers shared across layers (b.k1r medium consolidation).
//!
//! Every storage and session write path needs microsecond-resolution wall-
//! clock for the `server_ts` / `created_at` / `last_server_ts` fields that
//! feed the stream-entry payload and the object header. Pre-b.k1r three
//! copies of `now_micros()` lived in `storage::session`, `storage::layer`,
//! and (transitively) `grpc::ping`, each one `.expect`ing that the system
//! clock was not before UNIX epoch — a panic that would take the whole
//! process down if any replica's clock was rolled back past 1970 (rare but
//! survivable in other parts of the service).
//!
//! This module is the single source of truth. Saturating arithmetic maps
//! an impossibly-early clock to `0` so a bad clock degrades gracefully
//! instead of panicking. The result is still monotonically non-decreasing
//! for any reasonable clock (SystemTime on every supported platform is
//! ≥ epoch under normal operation); a `0` return is a clear signal in
//! tracing that operators can filter for.
//!
//! Lives in `domain` rather than `storage` because the function is pure
//! (no I/O, no async, no external state). The syscall behind
//! `SystemTime::now` is kernel-local — no network, no file I/O — so it
//! does not violate the `domain` layer's "no I/O" rule (CLAUDE.md Module
//! Dependency Rules). Storage / session / gRPC layers import it instead
//! of duplicating the helper per-module.

use std::time::{SystemTime, UNIX_EPOCH};

/// Current microseconds since UNIX epoch, or `0` if the system clock is
/// before epoch.
///
/// Pre-b.k1r the per-module copies used `.expect("system clock before UNIX epoch")`
/// which panicked on a clock rollback. Saturating the under-epoch case to
/// `0` keeps the commit path running and surfaces the anomaly as a `0`
/// timestamp in the stream payload — operators grep for that on dashboards.
pub fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_micros_is_nonzero_and_recent() {
        // Sanity: the value should be some multiple of a microsecond since
        // 1970. A zero would indicate the saturation branch fired (clock
        // before epoch) — impossible on any reasonable test runner.
        let t = now_micros();
        assert!(t > 0, "now_micros returned 0 — test runner clock anomaly?");
        // Post-2020 UNIX epoch microseconds are > 1.5e15; allow a generous
        // lower bound so the test is not tight.
        assert!(t > 1_600_000_000_000_000, "now_micros below 2020: {t}");
    }

    #[test]
    fn now_micros_is_nondecreasing_across_two_calls() {
        let a = now_micros();
        let b = now_micros();
        assert!(b >= a, "clock went backwards: {a} -> {b}");
    }
}
