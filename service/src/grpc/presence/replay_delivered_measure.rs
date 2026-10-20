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

//! **Test-support only** (compiled only with `--features test-support`, so
//! never into the service binary). The replay-delivered set D
//! ([`ReplayDelivered`]), built from a list of delivered connection ids the
//! way a completed membership replay builds it, so that a local measurement
//! tool (outside the published source) can measure D's heap with allocator
//! counters (SDD SR-7.1). The wrapper exposes building, size and ending only;
//! D's type and its suppression rule stay private to `grpc::presence`.

use super::{DeliveredEnd, ReplayDelivered};

/// D for a replay that delivered a given list of ids. Holds D's tables until
/// dropped or [`Self::end_at_own_event`].
pub struct MeasuredReplayDelivered(ReplayDelivered);

impl MeasuredReplayDelivered {
    /// D as `replay_member_pages` leaves it on completion, built by the
    /// replay's exact calls on the same kind of input: `delivered` stands in
    /// for the snapshot slots the replay delivered, built by the caller before
    /// any measurement.
    ///
    /// The calls are [`ReplayDelivered::with_capacity`] for the
    /// `snapshot_len` entries of the `SMEMBERS` snapshot; for each slot,
    /// `std::mem::take`, [`ReplayDelivered::split_for_delivery`] and
    /// [`ReplayDelivered::record_delivered`]; then
    /// [`ReplayDelivered::shrink_to_fit`]. The half of each split that the
    /// replay sends in the `PeerJoined` is written back into its slot, so it
    /// outlives the call as the message's string outlives the replay's.
    ///
    /// The allocations are therefore the replay's: none for a canonical
    /// UUID, one copy for any other id (the `PeerJoined`'s), plus D's own
    /// tables.
    pub fn build(snapshot_len: usize, delivered: &mut [String]) -> Self {
        let mut d = ReplayDelivered::with_capacity(snapshot_len);
        for slot in delivered.iter_mut() {
            let (peer_joined, key) = ReplayDelivered::split_for_delivery(std::mem::take(slot));
            *slot = peer_joined;
            d.record_delivered(key);
        }
        d.shrink_to_fit();
        Self(d)
    }

    /// The ids D holds: 0 once it has ended.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether D holds no id.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// End D at boundary (a), the join's own first live event, as the bidi
    /// loop does: its tables are freed.
    pub fn end_at_own_event(&mut self) {
        self.0.end(DeliveredEnd::OwnEvent);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// One spelling of a fixed UUID, the form `join_session_inner` mints.
    const UUID_LOWER: &str = "67e55044-10b1-426f-9247-bb680e5fe0c8";

    /// `n` minted connection ids.
    fn minted(n: usize) -> Vec<String> {
        (0..n).map(|_| uuid::Uuid::new_v4().to_string()).collect()
    }

    /// `build` keys each delivered id as the replay does — a canonical UUID
    /// in the 128-bit set, any other spelling by its string — and holds a
    /// repeated id once.
    #[test]
    fn measured_replay_delivered_build_holds_each_delivered_id_once_by_its_key() {
        let upper = UUID_LOWER.to_uppercase();
        let mut delivered: Vec<String> = [UUID_LOWER, upper.as_str(), "seeded-0001", UUID_LOWER]
            .map(str::to_owned)
            .to_vec();

        let d = MeasuredReplayDelivered::build(delivered.len(), &mut delivered);

        assert_eq!(d.len(), 3, "three distinct ids");
        assert!(!d.is_empty());
        assert_eq!(
            d.0.uuids.len(),
            1,
            "only the lowercase form has a 128-bit key"
        );
        assert_eq!(
            d.0.others,
            HashSet::from([upper.clone(), "seeded-0001".to_owned()]),
            "every other spelling is held by its string"
        );
        assert!(d.0.is_live(), "a built D is live until its boundary");
    }

    /// `build` leaves in each slot the string the replay's `PeerJoined` would
    /// carry, allocated as the replay allocates it: a canonical UUID's own
    /// string, uncopied, and for any other id an equal copy, its own string
    /// having become D's key.
    #[test]
    fn measured_replay_delivered_build_writes_each_peer_joined_string_back_into_its_slot() {
        let mut delivered = vec![UUID_LOWER.to_owned(), "seeded-0001".to_owned()];
        let buffers: Vec<*const u8> = delivered.iter().map(|s| s.as_ptr()).collect();

        let d = MeasuredReplayDelivered::build(delivered.len(), &mut delivered);

        assert_eq!(
            delivered,
            [UUID_LOWER, "seeded-0001"],
            "each slot holds its id again"
        );
        assert_eq!(
            delivered[0].as_ptr(),
            buffers[0],
            "a UUID slot holds its own string, not a copy"
        );
        assert_ne!(
            delivered[1].as_ptr(),
            buffers[1],
            "any other id's slot holds a copy"
        );
        let key = d.0.others.get("seeded-0001").expect("the other id is in D");
        assert_eq!(
            key.as_ptr(),
            buffers[1],
            "the other id's own string is D's key"
        );
    }

    /// `build` reserves room for the whole snapshot, then releases what the
    /// skipped entries would have used, as a completed replay does: 50 ids
    /// delivered from a snapshot of 400 leave a table sized for 50.
    #[test]
    fn measured_replay_delivered_build_releases_capacity_reserved_for_skipped_entries() {
        const SNAPSHOT: usize = 400;
        const DELIVERED: usize = 50;
        let mut ids = minted(DELIVERED);
        let reserved = HashSet::<u128>::with_capacity(SNAPSHOT).capacity();
        let mut fitted = HashSet::<u128>::with_capacity(SNAPSHOT);
        fitted.extend((0..DELIVERED).map(|i| i as u128));
        fitted.shrink_to_fit();

        let d = MeasuredReplayDelivered::build(SNAPSHOT, &mut ids);

        assert_eq!(d.len(), DELIVERED);
        assert_eq!(
            d.0.uuids.capacity(),
            fitted.capacity(),
            "D's table is shrunk to its {DELIVERED} ids, not left at the {reserved} reserved \
             for the snapshot"
        );
        assert!(
            fitted.capacity() < reserved,
            "fixture: shrinking frees room"
        );
    }

    /// Ending D at its boundary, the join's own first event, frees both of
    /// its tables: D holds nothing and keeps no capacity afterwards.
    #[test]
    fn measured_replay_delivered_end_at_own_event_frees_both_tables() {
        let mut ids = minted(10);
        ids.push("seeded-0001".to_owned());
        let mut d = MeasuredReplayDelivered::build(ids.len(), &mut ids);
        assert_eq!(d.len(), 11, "fixture: D holds every id before its boundary");

        d.end_at_own_event();

        assert_eq!(d.len(), 0);
        assert!(d.is_empty());
        assert_eq!(d.0.uuids.capacity(), 0, "the 128-bit table is freed");
        assert_eq!(d.0.others.capacity(), 0, "the string table is freed");
        assert_eq!(
            d.0.ended,
            Some(DeliveredEnd::OwnEvent),
            "ended by its own event"
        );
    }
}
