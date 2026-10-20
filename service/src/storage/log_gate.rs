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

//! Wall-clock rate limiter for log lines that describe a *condition*, not an
//! event (b.bhb round 4).
//!
//! Two call sites in `storage` had grown byte-identical copies of the same
//! gate — `client`'s steady-state auth hint and `watched_txn`'s two
//! "a defence layer is disarmed on this store" warnings — so the mechanism
//! lives here once and each site supplies its own interval.

use std::sync::Mutex;
use std::time::Duration;

/// Rate-limit gate for a repeated log line.
///
/// A local equivalent of `auth::service_token`'s `ParseFailureGate`
/// (`src/auth/service_token.rs`), deliberately re-implemented rather than
/// shared with it: that struct is module-private to `service_token`, and this
/// module already carries the same local-copy precedent (see
/// `storage::token_file_credentials`). Same two properties, and for the same
/// reasons — the first call always passes so the condition is never fully
/// silent, and the clock is [`tokio::time::Instant`] rather than
/// [`std::time::Instant`] so paused-clock tests advancing virtual time can open
/// the gate.
///
/// A `std::sync::Mutex` rather than a `Cell`: the gate is read from spawned
/// tasks (e.g. `client`'s `spawn_error_metric` hook), so it must be `Sync`. The
/// lock is never held across an `await` — [`Self::allow`] is synchronous and
/// the guard dies inside it.
///
/// [`Self::new`] is `const`, so a gate can be a `static` where the condition
/// being reported is process-wide rather than per-instance.
#[derive(Debug)]
pub(crate) struct LogGate {
    min_interval: Duration,
    last_emitted: Mutex<Option<tokio::time::Instant>>,
}

impl LogGate {
    pub(crate) const fn new(min_interval: Duration) -> Self {
        Self {
            min_interval,
            last_emitted: Mutex::new(None),
        }
    }

    /// Returns `true` at most once per `min_interval`, and always on the first
    /// call so the condition is never fully silent. A poisoned lock reports
    /// `false` (stay quiet) rather than panicking — this is a logging
    /// decoration, and taking the process down over it would be a strictly
    /// worse outcome than losing a line.
    pub(crate) fn allow(&self) -> bool {
        let now = tokio::time::Instant::now();
        let Ok(mut last) = self.last_emitted.lock() else {
            return false;
        };
        match *last {
            Some(prev) if now.saturating_duration_since(prev) < self.min_interval => false,
            _ => {
                *last = Some(now);
                true
            }
        }
    }
}
