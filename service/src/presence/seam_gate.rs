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

//! The park point shared by the presence test seams. Compiled only into test
//! builds (`#[cfg(any(test, feature = "test-fault-injection"))]` on the module
//! declaration); absent from the release binary.

/// One park point of the presence test seams: the JoinSession seams in
/// `grpc::presence` (`JoinRegistrationSeam`, `PresenceReplaySeam`,
/// `LeaseKeeperSeam`, `PresenceChannelSeam`) and the presence sweeper's
/// [`crate::presence::sweeper::PresenceSweeperSeam`]. Defined here, below
/// both, because `presence` may not depend on `grpc`; `grpc::presence`
/// re-exports it.
///
/// **One-shot and armed explicitly.** A gate starts disarmed, and a disarmed
/// gate lets every arrival straight through. [`Self::arm`] arms it; the first
/// arrival after that disarms it, raises the `parked` latch and blocks until
/// [`Self::release`]. Later arrivals pass straight through until the gate is
/// armed again. A seam is installed on a whole service or sweeper, so this is
/// what lets a test with several connections, a keeper that ticks every
/// refresh interval, a replay that crosses several page boundaries, or a sweep
/// that runs many chunks, park exactly one arrival: set up what it does not
/// want parked first, arm, then drive the one it does. Re-arm only after the
/// previous release; arming resets the latch and the release.
///
/// **Two phases, both level-triggered.** [`Self::wait_until_parked`] returns
/// once the arrival has parked — at once if it already has — so a test can
/// mutate state knowing the parked task has done everything before the park
/// point and nothing after it. [`Self::release`] lets it go, and may be called
/// before the arrival, which then passes through without blocking. `watch`
/// rather than `Barrier` or `Notify` for the same reason as the catch-up
/// seams' parked latch: a late observer still sees the raised state.
///
/// A parked arrival is not cancellation-aware: it waits for its release
/// whatever happens meanwhile. A test that cancels a parked task fires its
/// trigger, then releases the gate, and the task sees the cancel at its next
/// check.
///
/// **Test-only.** Production code never constructs this — it is gated behind
/// `#[cfg(any(test, feature = "test-fault-injection"))]`, as are the fields
/// and spawn functions that hold the seams.
pub struct PresenceSeamGate {
    armed: std::sync::atomic::AtomicBool,
    parked: tokio::sync::watch::Sender<bool>,
    released: tokio::sync::watch::Sender<bool>,
}

impl PresenceSeamGate {
    /// A disarmed gate.
    pub fn new() -> Self {
        Self {
            armed: std::sync::atomic::AtomicBool::new(false),
            parked: tokio::sync::watch::channel(false).0,
            released: tokio::sync::watch::channel(false).0,
        }
    }

    /// Arm the gate: the next arrival parks.
    pub fn arm(&self) {
        self.parked.send_replace(false);
        self.released.send_replace(false);
        self.armed.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Test side: block until an arrival has parked at this gate.
    pub async fn wait_until_parked(&self) {
        let mut parked = self.parked.subscribe();
        while !*parked.borrow_and_update() {
            if parked.changed().await.is_err() {
                return;
            }
        }
    }

    /// Test side: release the parked arrival (or the next one, if none has
    /// parked yet).
    pub fn release(&self) {
        self.released.send_replace(true);
    }

    /// Production side: park here if the gate is armed.
    pub(crate) async fn arrive(&self) {
        if !self.armed.swap(false, std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        // `send_replace`, not `send`: `send` skips the update when no receiver
        // exists, and none does until the test calls `wait_until_parked`.
        self.parked.send_replace(true);
        let mut released = self.released.subscribe();
        while !*released.borrow_and_update() {
            if released.changed().await.is_err() {
                return;
            }
        }
    }
}

impl Default for PresenceSeamGate {
    fn default() -> Self {
        Self::new()
    }
}
