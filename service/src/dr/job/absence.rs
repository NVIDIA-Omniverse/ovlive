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

//! The absence rule's primitives (SDD SR-9.3; decider ruling A): a Job never
//! acts on absence alone. Tasks 7 and 10 use these for every absence; none
//! re-implements them.
//!
//! An absence is an empty or short listing (an all-empty one on a store whose
//! counters sidecar is present included), a listed artifact or its meta
//! companion reading not-found, `ext_blob_exists` answering `false`, a missing
//! ext-refs sidecar, or an absent counters sidecar. The backup store can keep
//! answering "not found" for up to L after a write ([`super::quiet`]), so:
//!
//! - **`T_safe`** ([`StoreClock`]): an absence that changes the run's outcome
//!   (a step-down to an older generation, a "stays held" exit 3, an
//!   accept-loss discard) is acted on only when observed at or after
//!   `detected_at_ms` + Q + L on the store's clock. One observed earlier
//!   makes the Job wait until `T_safe`, logging the wait, and observe again;
//!   the second observation is authoritative ([`observe_under_absence_rule`]).
//!   A step-down or a discard is never taken on a pre-`T_safe` reading.
//! - **PD2** ([`check_counters_sidecar`]): with a persistent adapter
//!   configured, the run reads the counters sidecar as present before it
//!   trusts any absence; absent at or after `T_safe`, it stays held (exit 3).
//!   A sidecar that reads present but does not parse also stays held (exit
//!   3, PD-7): its value is the counter floor's sidecar term at the un-hold.
//! - **The list-route canary** ([`run_list_route_canary`]): before it trusts
//!   an empty or short listing, the run proves the listing route lists back
//!   an object written now. It is a write, so it runs after the write permit
//!   opens; a failure before `T_safe` is re-run at `T_safe` before the run
//!   stays held.
//! - **The contradiction check** ([`find_contradictions`]): a surviving
//!   object whose header carries a backup stamp, that is not on the dead-object
//!   set, and that has no listed artifact makes the listing untrustworthy; so
//!   does a session whose last landed backup-store operation stored a session
//!   backup that the listing does not show. Re-checked at `T_safe` before the
//!   run stays held.
//! - **The run's own PUTs are evidence** ([`OwnPuts`]): a value the Job
//!   stored in the backup store this run is never probed again.

use std::collections::HashSet;
use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::permit::{RevokeCause, WritePermit};
use super::quiet::StoreDeadlines;
use crate::checkpoint::counters::CountersSidecar;
use crate::checkpoint::{CheckpointError, CheckpointStore};
use crate::storage::client::ValkeyClient;
use crate::storage::dr_job::{store_time_ms, CheckpointStamp};
use crate::storage::keys::KeySchema;

/// The named cause of an absent counters sidecar at or after `T_safe`.
pub const SIDECAR_ABSENT_CAUSE: &str = "the backup store has no counters record: wrong address \
     or credentials, or backups never ran";

/// The named cause of a list-route canary that failed at or after `T_safe`.
pub const LIST_ROUTE_CAUSE: &str =
    "the backup store's listing route is not working: check its address and permissions";

/// Why a Job step could not finish.
#[derive(Debug, thiserror::Error)]
pub enum StepError {
    /// A store command failed.
    #[error("a store command failed: {0}")]
    Store(#[from] fred::error::Error),
    /// A backup-store call failed with a backend fault.
    #[error("a backup-store call failed: {0}")]
    Backup(#[from] CheckpointError),
    /// The write permit was revoked.
    #[error("the write permit was revoked: {0}")]
    Revoked(RevokeCause),
}

/// The store's clock, and the Job's deadlines on it.
#[derive(Clone, Debug)]
pub struct StoreClock {
    client: ValkeyClient,
    keys: KeySchema,
    deadlines: StoreDeadlines,
    log_interval: Duration,
}

impl StoreClock {
    /// A clock reading `TIME` on the hold marker's node through `client`,
    /// with `deadlines`, logging a long wait every `log_interval`.
    pub fn new(
        client: ValkeyClient,
        keys: KeySchema,
        deadlines: StoreDeadlines,
        log_interval: Duration,
    ) -> Self {
        Self {
            client,
            keys,
            deadlines,
            log_interval,
        }
    }

    /// The Job's deadlines.
    pub fn deadlines(&self) -> StoreDeadlines {
        self.deadlines
    }

    /// How often a long wait on this clock logs what is left, and the
    /// cadence a body retries a failed read of it at (the frame's
    /// `JobTimings::wait_log_interval`).
    pub fn log_interval(&self) -> Duration {
        self.log_interval
    }

    /// The store's `TIME`, in milliseconds.
    pub async fn now_ms(&self) -> Result<u64, fred::error::Error> {
        store_time_ms(&self.client, &self.keys).await
    }

    /// Whether the store's clock has reached `T_safe`.
    pub async fn at_or_after_t_safe(&self) -> Result<bool, fred::error::Error> {
        Ok(self.now_ms().await? >= self.deadlines.t_safe_ms)
    }

    /// Wait until the store's clock reaches `target_ms`, re-reading `TIME`
    /// at least every `log_interval` and logging the remaining wait at INFO,
    /// naming `why`. Returns once a read shows the clock there.
    pub async fn wait_until(&self, target_ms: u64, why: &str) -> Result<(), fred::error::Error> {
        loop {
            let now = self.now_ms().await?;
            if now >= target_ms {
                return Ok(());
            }
            let remaining = Duration::from_millis(target_ms - now);
            tracing::info!(
                remaining_secs = remaining.as_secs(),
                "waiting {}s on the store's clock: {why}",
                remaining.as_secs()
            );
            tokio::time::sleep(remaining.min(self.log_interval)).await;
        }
    }

    /// [`Self::wait_until`] `T_safe`.
    pub async fn wait_until_t_safe(&self, why: &str) -> Result<(), fred::error::Error> {
        self.wait_until(self.deadlines.t_safe_ms, why).await
    }
}

/// A value observed under the absence rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observed<T> {
    /// The authoritative observation.
    pub value: T,
    /// Whether it was taken at or after `T_safe`. An absence in `value` may
    /// change the outcome only when this is `true`; it always is when
    /// `value` is an absence.
    pub at_or_after_t_safe: bool,
}

/// Observe something under the absence rule: read the store's clock, run
/// `observe`, and when `is_absence` says the result is an absence and the
/// clock was before `T_safe`, wait until `T_safe` (logging the wait, naming
/// `what`) and observe again. The second observation is authoritative,
/// absent or not.
///
/// An error from `observe` is returned as it is; whether an error counts as
/// an absence is the caller's to decide in `is_absence` over a `Result` it
/// returns as the value.
pub async fn observe_under_absence_rule<T, E, F, Fut>(
    clock: &StoreClock,
    what: &str,
    mut observe: F,
    is_absence: impl Fn(&T) -> bool,
) -> Result<Observed<T>, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: From<fred::error::Error>,
{
    let before_t_safe = !clock.at_or_after_t_safe().await?;
    let value = observe().await?;
    if !is_absence(&value) {
        return Ok(Observed {
            value,
            at_or_after_t_safe: !before_t_safe,
        });
    }
    if !before_t_safe {
        return Ok(Observed {
            value,
            at_or_after_t_safe: true,
        });
    }
    clock
        .wait_until_t_safe(&format!(
            "{what} read as absent before T_safe, when the backup store's not-found answers can \
             still be stale; reading it again then"
        ))
        .await?;
    Ok(Observed {
        value: observe().await?,
        at_or_after_t_safe: true,
    })
}

/// The counters sidecar precondition's verdict.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SidecarCheck {
    /// The sidecar read as present this run, and parsed. Its
    /// `max_object_id_observed` is the counter floor's sidecar term: nothing
    /// writes the sidecar while held (the counters tick does not run), so
    /// this reading stands for the run.
    Present(CountersSidecar),
    /// The sidecar read as present but does not parse (PD-7): the run writes
    /// nothing, stays held and exits 3 rather than drop the counter floor's
    /// sidecar term. Carries the parse error.
    Unreadable(String),
    /// The sidecar read as absent at or after `T_safe`: the run writes
    /// nothing more, stays held and exits 3 with [`SIDECAR_ABSENT_CAUSE`].
    Absent,
}

/// PD2: read the counters sidecar under the absence rule, and parse it.
/// Call it, with a persistent adapter configured, before trusting any
/// absence. A read error is a [`StepError::Backup`].
pub async fn check_counters_sidecar(
    store: &dyn CheckpointStore,
    clock: &StoreClock,
) -> Result<SidecarCheck, StepError> {
    let observed = observe_under_absence_rule(
        clock,
        "the backup store's counters record",
        || async { Ok::<_, StepError>(store.get_counters_sidecar().await?) },
        |sidecar| sidecar.is_none(),
    )
    .await?;
    Ok(match observed.value {
        Some(bytes) => match serde_json::from_slice::<CountersSidecar>(&bytes) {
            Ok(sidecar) => SidecarCheck::Present(sidecar),
            Err(e) => SidecarCheck::Unreadable(e.to_string()),
        },
        None => SidecarCheck::Absent,
    })
}

/// The list-route canary's verdict.
#[derive(Debug)]
pub enum CanaryCheck {
    /// The listing route listed back an object written now.
    Passed,
    /// The canary failed at or after `T_safe`: the run stays held and exits
    /// 3 with [`LIST_ROUTE_CAUSE`]. Carries the adapter's error.
    Failed(CheckpointError),
}

/// Run the list-route canary (`CheckpointStore::verify_list_route`) through
/// the write permit: it is a write. A failure before `T_safe` is re-run at
/// `T_safe` (through the permit again), and that second answer decides. A
/// `Transient` error is a backend fault ([`StepError::Backup`]), not a
/// failed canary.
pub async fn run_list_route_canary(
    store: &dyn CheckpointStore,
    permit: &WritePermit,
    clock: &StoreClock,
) -> Result<CanaryCheck, StepError> {
    let observed = observe_under_absence_rule(
        clock,
        "the backup store's listing route check (the list-route canary)",
        || async {
            permit.begin_unit().await.map_err(StepError::Revoked)?;
            match store.verify_list_route().await {
                Ok(()) => Ok(None),
                Err(e @ CheckpointError::Transient { .. }) => Err(StepError::Backup(e)),
                Err(e) => Ok(Some(e)),
            }
        },
        Option::is_some,
    )
    .await?;
    Ok(match observed.value {
        None => CanaryCheck::Passed,
        Some(e) => CanaryCheck::Failed(e),
    })
}

/// A piece of positive evidence the backup store's listing does not show.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Contradiction {
    /// A surviving object whose header carries a backup stamp, not on the
    /// dead-object set, with no listed artifact.
    Layer(CheckpointStamp),
    /// A session whose last landed backup-store operation stored a session
    /// backup, with no listed session backup.
    Session {
        /// The session.
        session_id: String,
    },
}

impl fmt::Display for Contradiction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Layer(stamp) => write!(
                f,
                "object {} (session {:?}, layer {:?}) survives with a backup stamp \
                 (last_checkpoint_seq {}, last_checkpoint_ms {}), but the backup store lists no \
                 backup of it",
                stamp.object_id,
                stamp.session_id,
                stamp.layer_name,
                stamp
                    .last_checkpoint_seq
                    .map_or_else(|| "absent".to_owned(), |v| v.to_string()),
                stamp
                    .last_checkpoint_ms
                    .map_or_else(|| "absent".to_owned(), |v| v.to_string()),
            ),
            Self::Session { session_id } => write!(
                f,
                "session {session_id:?}'s last backup-store operation stored a session backup, \
                 but the backup store lists none for it"
            ),
        }
    }
}

/// The contradiction check: every piece of positive evidence the listing
/// does not show.
///
/// - `stamps`: the surviving object headers the caller read
///   (`storage::dr_job::read_checkpoint_stamp`); only stamped ones count.
/// - `sessions_backed_up`: sessions whose meta survives and whose publish
///   watermark says the last landed operation was a write
///   (`storage::dr_job::read_session_backup_kind` = `write`).
/// - `dead`: the dead-object set (`storage::dr_job::read_dead_object_ids`).
/// - `listed_objects` / `listed_sessions`: what this run's listing showed.
///
/// A hit observed before `T_safe` is re-checked at `T_safe` (relist, then
/// call this again, through [`observe_under_absence_rule`]) before the run
/// stays held naming it.
pub fn find_contradictions(
    stamps: &[CheckpointStamp],
    sessions_backed_up: &[String],
    dead: &HashSet<u64>,
    listed_objects: &HashSet<u64>,
    listed_sessions: &HashSet<String>,
) -> Vec<Contradiction> {
    let mut found: Vec<Contradiction> = stamps
        .iter()
        .filter(|s| s.is_stamped())
        .filter(|s| !dead.contains(&s.object_id) && !listed_objects.contains(&s.object_id))
        .cloned()
        .map(Contradiction::Layer)
        .collect();
    found.extend(
        sessions_backed_up
            .iter()
            .filter(|sid| !listed_sessions.contains(*sid))
            .map(|sid| Contradiction::Session {
                session_id: sid.clone(),
            }),
    );
    found
}

/// The external values this run stored in the backup store itself. A value
/// here is covered by the run's own successful PUT and is never probed again.
/// Cheap to clone; every clone is the same record.
#[derive(Clone, Debug, Default)]
pub struct OwnPuts(Arc<Mutex<HashSet<String>>>);

impl OwnPuts {
    /// Record that this run's PUT of `hash_hex` succeeded.
    pub fn record(&self, hash_hex: &str) {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(hash_hex.to_owned());
    }

    /// Whether this run stored `hash_hex` itself.
    pub fn contains(&self, hash_hex: &str) -> bool {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(hash_hex)
    }

    /// How many values this run stored.
    pub fn len(&self) -> usize {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// Whether this run stored none.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(object_id: u64, stamped: bool) -> CheckpointStamp {
        CheckpointStamp {
            object_id,
            session_id: "s".into(),
            layer_name: "l".into(),
            last_checkpoint_seq: stamped.then_some(3),
            last_checkpoint_ms: None,
        }
    }

    #[test]
    fn a_stamped_unlisted_live_object_contradicts_the_listing() {
        let dead: HashSet<u64> = [2].into();
        let listed: HashSet<u64> = [3].into();
        let found = find_contradictions(
            &[
                stamp(1, true),
                stamp(2, true),
                stamp(3, true),
                stamp(4, false),
            ],
            &[],
            &dead,
            &listed,
            &HashSet::new(),
        );
        assert_eq!(found, vec![Contradiction::Layer(stamp(1, true))]);
        assert!(found[0].to_string().contains("object 1"));
    }

    #[test]
    fn a_backed_up_session_missing_from_the_listing_contradicts_it() {
        let listed: HashSet<String> = ["a".to_owned()].into();
        let found = find_contradictions(
            &[],
            &["a".to_owned(), "b".to_owned()],
            &HashSet::new(),
            &HashSet::new(),
            &listed,
        );
        assert_eq!(
            found,
            vec![Contradiction::Session {
                session_id: "b".into()
            }]
        );
    }

    #[test]
    fn own_puts_remember_what_this_run_stored() {
        let puts = OwnPuts::default();
        assert!(puts.is_empty());
        puts.clone().record("ab");
        assert!(puts.contains("ab"));
        assert!(!puts.contains("cd"));
        assert_eq!(puts.len(), 1);
    }
}
