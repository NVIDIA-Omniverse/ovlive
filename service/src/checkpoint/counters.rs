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

// Counters-sidecar wire shape — shared between producer and consumer.
//
// The Epic 6 periodic counters-sync tick (SDD SR-5.1) GETs
// `le:counter:object_id`, builds a [`CountersSidecar`], serializes it
// with `serde_json::to_vec`, and writes the bytes through
// [`CheckpointStore::put_counters_sidecar`]. The Epic 6 startup recovery
// hook (SDD SR-5.3, sibling Task 4) reads those bytes back, deserializes
// through the same type, and seeds `le:counter:object_id` past
// `max_object_id_observed` plus the configured safety margin.
//
// Round-tripping both sides through one shared type means a typo or
// field rename surfaces as a Rust compile error rather than as a silent
// recovery miss in production.

use serde::{Deserialize, Serialize};

/// Wire shape of the counters-sidecar JSON blob persisted by the
/// Epic 6 periodic counters-sync tick (SDD SR-5.1) and consumed by the
/// Epic 6 startup counter-seed recovery routine (SDD SR-5.3, Task 4).
///
/// Both ends MUST round-trip through this type so a typo in either site
/// surfaces at compile time rather than as a silent recovery miss.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CountersSidecar {
    /// The `GET le:counter:object_id` reading taken by the counters-sync
    /// tick that wrote this sidecar. A lagging floor, not the exact high
    /// water: no tick writes at or below the stored value, but a write
    /// already in flight can land late and briefly leave an earlier real
    /// reading in place until the next tick. The recovery hook seeds
    /// `le:counter:object_id` past this value (plus the
    /// `LIVE_EDIT_CHECKPOINT_COUNTER_SAFETY_MARGIN`) on startup.
    pub max_object_id_observed: u64,
    /// RFC3339 wall-clock timestamp recorded at producer-side write time.
    /// Operability-only — the recovery hook does NOT use this for
    /// correctness; the safety margin is what defends against clock skew.
    pub updated_at: String,
}
