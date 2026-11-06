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

//! The write permit: the one gate every write of a Job passes (SDD SR-9.1).
//!
//! Three states:
//! - **Paused** — until the quiet period ends. No write unit starts.
//! - **Open** — write units may start.
//! - **Revoked** — the lease was lost or SIGTERM arrived. Final: the frame
//!   aborts the work in flight and exits.
//!
//! Every write unit of Tasks 6–10 calls [`WritePermit::begin_unit`] before
//! it starts: the list-route canary, each restore batch, each backfill PUT,
//! each group of the recount's count sets and each attempt of the un-hold.
//! Only the frame (`super::run`) moves the state; a body only waits on it.
//!
//! The permit also counts the units it let begin
//! ([`WritePermit::units_begun`]): the un-hold refuses when any unit began
//! after the recount (`super::finish`), so a write the recount may not
//! cover never precedes the un-hold.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::watch;

/// Why a permit was revoked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RevokeCause {
    /// The Job's lease refresh found the lease gone, held by another Job, or
    /// unconfirmed for too long.
    LeaseLost(String),
    /// SIGTERM (or SIGINT).
    Shutdown,
}

impl fmt::Display for RevokeCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LeaseLost(why) => write!(f, "the Job's lease was lost: {why}"),
            Self::Shutdown => f.write_str("the Job received SIGTERM"),
        }
    }
}

/// The permit's state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PermitState {
    /// No new write unit may start.
    Paused,
    /// Write units may start.
    Open,
    /// Final: no write unit may start, and the frame aborts the one in
    /// flight.
    Revoked(RevokeCause),
}

/// The write permit. Cheap to clone; every clone is the same permit.
#[derive(Clone, Debug)]
pub struct WritePermit {
    state: Arc<watch::Sender<PermitState>>,
    /// How many units [`Self::begin_unit`] let begin.
    units_begun: Arc<AtomicU64>,
}

impl Default for WritePermit {
    fn default() -> Self {
        Self::new()
    }
}

impl WritePermit {
    /// A paused permit.
    pub fn new() -> Self {
        Self {
            state: Arc::new(watch::channel(PermitState::Paused).0),
            units_begun: Arc::new(AtomicU64::new(0)),
        }
    }

    /// The current state.
    pub fn state(&self) -> PermitState {
        self.state.borrow().clone()
    }

    /// How many write units [`Self::begin_unit`] has let begin, over every
    /// clone of this permit.
    pub fn units_begun(&self) -> u64 {
        self.units_begun.load(Ordering::SeqCst)
    }

    /// Wait until a write unit may start: return `Ok` at once while open,
    /// wait while paused, and return the cause once revoked. Call it
    /// immediately before every write unit. Each `Ok` counts one unit
    /// ([`Self::units_begun`]).
    pub async fn begin_unit(&self) -> Result<(), RevokeCause> {
        let mut rx = self.state.subscribe();
        let outcome = rx
            .wait_for(|state| !matches!(state, PermitState::Paused))
            .await
            .map(|state| state.clone());
        match outcome {
            Ok(PermitState::Open) => {
                self.units_begun.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
            Ok(PermitState::Revoked(cause)) => Err(cause),
            Ok(PermitState::Paused) => unreachable!("wait_for returns a non-paused state"),
            // The sender lives as long as any clone of this permit, which
            // `self` is.
            Err(_) => Err(RevokeCause::Shutdown),
        }
    }

    /// Resolve once the permit is revoked, with the cause.
    pub async fn revoked(&self) -> RevokeCause {
        let mut rx = self.state.subscribe();
        let cause = match rx
            .wait_for(|state| matches!(state, PermitState::Revoked(_)))
            .await
        {
            Ok(state) => match &*state {
                PermitState::Revoked(cause) => cause.clone(),
                _ => unreachable!("wait_for returns a revoked state"),
            },
            Err(_) => RevokeCause::Shutdown,
        };
        cause
    }

    /// Open the permit, unless it is revoked.
    pub(crate) fn open(&self) {
        self.state.send_if_modified(|state| match state {
            PermitState::Paused => {
                *state = PermitState::Open;
                true
            }
            PermitState::Open | PermitState::Revoked(_) => false,
        });
    }

    /// Revoke the permit for `cause`. The first revocation's cause stays.
    pub(crate) fn revoke(&self, cause: RevokeCause) {
        self.state.send_if_modified(|state| match state {
            PermitState::Revoked(_) => false,
            _ => {
                *state = PermitState::Revoked(cause);
                true
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn a_paused_permit_holds_a_unit_until_it_opens() {
        let permit = WritePermit::new();
        let waiting = tokio::spawn({
            let permit = permit.clone();
            async move { permit.begin_unit().await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiting.is_finished(), "a paused permit starts no unit");
        permit.open();
        assert_eq!(waiting.await.unwrap(), Ok(()));
        assert_eq!(permit.begin_unit().await, Ok(()));
    }

    #[tokio::test]
    async fn revocation_is_final_and_keeps_the_first_cause() {
        let permit = WritePermit::new();
        permit.open();
        let lost = RevokeCause::LeaseLost("gone".into());
        permit.revoke(lost.clone());
        permit.revoke(RevokeCause::Shutdown);
        permit.open();
        assert_eq!(permit.state(), PermitState::Revoked(lost.clone()));
        assert_eq!(permit.begin_unit().await, Err(lost.clone()));
        assert_eq!(permit.revoked().await, lost);
    }

    #[tokio::test]
    async fn a_waiting_unit_sees_a_revocation() {
        let permit = WritePermit::new();
        let waiting = tokio::spawn({
            let permit = permit.clone();
            async move { permit.begin_unit().await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        permit.revoke(RevokeCause::LeaseLost("gone".into()));
        assert_eq!(
            waiting.await.unwrap(),
            Err(RevokeCause::LeaseLost("gone".into()))
        );
    }
}
