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

//! Pure assignment of stream keys to readers.
//!
//! Standalone: any keys may share one XREAD, so a key's reader is
//! `object_id % readers` and every subscription of one object lands on the
//! same reader. Cluster: one XREAD may only name keys of one hash slot, so a
//! reader is bound to a slot while it has subscriptions and freed at zero.
//!
//! **Node affinity (cluster).** A cluster reader connects only to the primary
//! that owns its bound slot, and has to replace that connection (a retarget)
//! when it next binds a slot another primary owns. The router records, per
//! reader, the node its connection targets ([`RouterCore::set_node`]; the
//! reader reports it on connect, retarget and retire, and the record survives
//! the reader's release). A new binding prefers a free reader already
//! connected to the slot's owner ([`RouterCore::route_near`]) and otherwise
//! takes the lowest-index free reader. The preference only chooses among free
//! readers: it never refuses while one is free, so the number of distinct slots
//! a replica can serve at once is unchanged. Standalone routing ignores it.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Routing {
    Hash,
    SlotAffinity,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReaderBinding {
    pub slot: Option<u16>,
    pub refs: usize,
}

#[derive(Debug)]
pub struct RouterCore {
    routing: Routing,
    readers: Vec<ReaderBinding>,
    by_slot: HashMap<u16, usize>,
    /// Per reader, the node (`host:port`) its connection targets, as the
    /// reader last reported it; `None` while it holds no connection, or holds
    /// one whose node is not named (standalone). Kept across releases.
    nodes: Vec<Option<String>>,
}

impl RouterCore {
    pub fn new(routing: Routing, readers: usize) -> Self {
        assert!(readers > 0, "a stream mux needs at least one reader");
        Self {
            routing,
            readers: vec![ReaderBinding::default(); readers],
            by_slot: HashMap::new(),
            nodes: vec![None; readers],
        }
    }

    pub fn routing(&self) -> Routing {
        self.routing
    }

    pub fn readers(&self) -> &[ReaderBinding] {
        &self.readers
    }

    /// Picks the reader for one new subscription and takes a reference on it,
    /// with no node preference: [`Self::route_near`] with no owner.
    pub fn route(&mut self, object_id: u64, slot: u16) -> Option<usize> {
        self.route_near(object_id, slot, None)
    }

    /// Picks the reader for one new subscription and takes a reference on it.
    /// `None`: every reader is bound to another slot (cluster only).
    ///
    /// On a cluster, a slot already bound keeps its reader. Otherwise, when
    /// `owner` names the node that owns `slot`, a free reader whose connection
    /// targets that node is preferred; failing one, the lowest-index free
    /// reader is bound. Standalone routing ignores `owner`.
    pub fn route_near(&mut self, object_id: u64, slot: u16, owner: Option<&str>) -> Option<usize> {
        let reader = match self.routing {
            Routing::Hash => (object_id % self.readers.len() as u64) as usize,
            Routing::SlotAffinity => match self.by_slot.get(&slot) {
                Some(&r) => r,
                None => {
                    let free = |r: &usize| self.readers[*r].refs == 0;
                    let near = owner.and_then(|owner| {
                        (0..self.readers.len())
                            .filter(free)
                            .find(|&r| self.nodes[r].as_deref() == Some(owner))
                    });
                    let r = match near {
                        Some(r) => r,
                        None => (0..self.readers.len()).find(free)?,
                    };
                    self.readers[r].slot = Some(slot);
                    self.by_slot.insert(slot, r);
                    r
                }
            },
        };
        self.readers[reader].refs += 1;
        Some(reader)
    }

    /// Whether `slot` is bound to a reader now (cluster routing), so a new
    /// subscription to it would join that reader rather than bind one.
    pub fn is_bound(&self, slot: u16) -> bool {
        self.by_slot.contains_key(&slot)
    }

    /// Records the node `reader`'s connection now targets (`None`: it holds
    /// none, or one whose node is not named).
    pub fn set_node(&mut self, reader: usize, node: Option<String>) {
        self.nodes[reader] = node;
    }

    /// The node `reader`'s connection targets, as last recorded.
    pub fn node(&self, reader: usize) -> Option<&str> {
        self.nodes[reader].as_deref()
    }

    /// Drops one reference taken by [`Self::route`]. Returns `true` when a
    /// slot-bound reader became free.
    pub fn release(&mut self, reader: usize) -> bool {
        let b = &mut self.readers[reader];
        debug_assert!(b.refs > 0, "release without a matching route");
        b.refs = b.refs.saturating_sub(1);
        if b.refs == 0 {
            if let Some(slot) = b.slot.take() {
                self.by_slot.remove(&slot);
                return true;
            }
        }
        false
    }

    /// Readers with at least one subscription.
    pub fn active(&self) -> usize {
        self.readers.iter().filter(|b| b.refs > 0).count()
    }

    pub fn subscriptions(&self) -> usize {
        self.readers.iter().map(|b| b.refs).sum()
    }
}

/// A hash tag that maps to `slot`, so a reader's wake key can share the slot
/// of the streams it reads. Deterministic; computed on demand.
pub fn tag_for_slot(slot: u16) -> String {
    (0u64..)
        .map(|n| format!("mx{n}"))
        .find(|t| fred::util::redis_keyslot(t.as_bytes()) == slot)
        .expect("CRC16 covers every slot")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_routing_is_object_id_modulo_readers_and_counts_refs() {
        let mut r = RouterCore::new(Routing::Hash, 4);
        assert_eq!(r.route(10, 0), Some(2));
        assert_eq!(r.route(14, 99), Some(2));
        assert_eq!(r.route(3, 0), Some(3));
        assert_eq!(r.readers()[2].refs, 2);
        assert!(!r.release(2), "hash readers are never bound");
        assert_eq!(r.subscriptions(), 2);
        assert_eq!(r.active(), 2);
    }

    #[test]
    fn slot_affinity_shares_a_slot_binds_free_readers_and_unbinds_at_zero() {
        let mut r = RouterCore::new(Routing::SlotAffinity, 2);
        assert_eq!(r.route(1, 100), Some(0));
        assert_eq!(r.route(2, 100), Some(0), "same slot, same reader");
        assert_eq!(r.route(3, 200), Some(1));
        assert_eq!(r.route(4, 300), None, "both readers bound elsewhere");
        assert!(!r.release(0));
        assert!(r.release(0), "last reference frees the reader");
        assert_eq!(r.readers()[0], ReaderBinding::default());
        assert_eq!(r.route(4, 300), Some(0), "freed reader rebinds");
        assert_eq!(r.readers()[0].slot, Some(300));
    }

    // ── Node affinity (b.azh) ───────────────────────────────────────────────
    //
    // A cluster reader's connection targets one primary, so binding a slot
    // owned elsewhere costs it a retarget (a new connection). The router
    // prefers a free reader already connected to the owner, and only chooses
    // among free readers: it never refuses while one is free.

    const A: &str = "10.0.0.1:6379";
    const B: &str = "10.0.0.2:6379";

    #[test]
    fn route_near_prefers_the_lowest_free_reader_connected_to_the_owner() {
        let mut r = RouterCore::new(Routing::SlotAffinity, 4);
        r.set_node(0, Some(A.to_owned()));
        r.set_node(1, Some(B.to_owned()));
        r.set_node(2, Some(B.to_owned()));

        assert_eq!(
            r.route_near(1, 100, Some(B)),
            Some(1),
            "the lowest-index free reader on the owner, not the lowest-index free reader"
        );
        assert_eq!(
            r.route_near(2, 200, Some(B)),
            Some(2),
            "reader 1 is bound now; the next free reader on the owner"
        );
        assert_eq!(
            r.route_near(3, 300, Some(B)),
            Some(0),
            "no free reader on the owner: the lowest-index free reader, never a refusal"
        );
        assert_eq!(
            r.route_near(4, 400, Some(A)),
            Some(3),
            "the reader on the owner is bound, and an unconnected free reader still serves"
        );
        assert_eq!(
            r.route_near(5, 500, Some(A)),
            None,
            "only with every reader bound elsewhere is a new slot refused"
        );
    }

    #[test]
    fn route_near_without_an_owner_or_a_match_takes_the_lowest_free_reader() {
        let mut r = RouterCore::new(Routing::SlotAffinity, 3);
        r.set_node(1, Some(B.to_owned()));
        assert_eq!(
            r.route_near(1, 100, None),
            Some(0),
            "no owner named: as `route`"
        );
        assert_eq!(
            r.route_near(2, 200, Some("10.0.0.9:6379")),
            Some(1),
            "no free reader on the owner: the lowest free one"
        );
        assert_eq!(
            r.route(3, 300),
            Some(2),
            "`route` is `route_near` with no owner"
        );
    }

    #[test]
    fn a_bound_slot_keeps_its_reader_whatever_owner_is_named() {
        let mut r = RouterCore::new(Routing::SlotAffinity, 2);
        r.set_node(1, Some(B.to_owned()));
        assert_eq!(r.route_near(1, 100, None), Some(0));
        assert!(r.is_bound(100));
        assert!(!r.is_bound(200));
        assert_eq!(
            r.route_near(2, 100, Some(B)),
            Some(0),
            "a slot already bound joins its reader: one slot per reader"
        );
        assert_eq!(r.readers()[0].refs, 2);
        assert_eq!(r.readers()[1].refs, 0, "the reader on the owner stays free");
    }

    #[test]
    fn the_node_record_survives_release_so_a_freed_reader_is_preferred_again() {
        let mut r = RouterCore::new(Routing::SlotAffinity, 2);
        r.set_node(1, Some(B.to_owned()));
        assert_eq!(r.route_near(1, 100, Some(B)), Some(1));
        assert!(r.release(1), "the last reference frees the reader");
        assert_eq!(
            r.node(1),
            Some(B),
            "a release frees the binding, not the connection: the node is kept"
        );
        assert_eq!(
            r.route_near(2, 200, Some(B)),
            Some(1),
            "the freed reader, still connected to the owner, is preferred over reader 0"
        );
        // A retire clears the record (`set_node(_, None)`, which the
        // multiplexer calls when it takes the connection out of the slot):
        // reader 1 is then no longer preferred for the owner it was on.
        r.set_node(1, None);
        assert_eq!(r.node(1), None);
        assert!(r.release(1));
        assert_eq!(
            r.route_near(3, 300, Some(B)),
            Some(0),
            "with its record cleared, the freed reader 1 is no longer preferred for B: the \
             lowest-index free reader takes the binding"
        );
    }

    #[test]
    fn hash_routing_ignores_the_owner() {
        let mut r = RouterCore::new(Routing::Hash, 4);
        r.set_node(0, Some(B.to_owned()));
        assert_eq!(
            r.route_near(10, 0, Some(B)),
            Some(2),
            "standalone routing is object_id % readers whatever the nodes"
        );
        assert!(!r.is_bound(0), "hash routing binds no slot");
    }

    #[test]
    fn tag_for_slot_hashes_to_its_slot_on_a_spread_of_slots() {
        let mut cache: HashMap<u16, String> = HashMap::new();
        // Walk tags once and record the first tag seen for each slot: that
        // walk proves every slot has an `mx{n}` tag. Then check the function
        // returns that first tag on a spread of slots (every 97th, plus both
        // ends). Not every slot: `tag_for_slot` repeats the walk per call, so a
        // full sweep would cost the whole walk 16384 times.
        for n in 0u64..200_000 {
            let t = format!("mx{n}");
            cache
                .entry(fred::util::redis_keyslot(t.as_bytes()))
                .or_insert(t);
            if cache.len() == 16384 {
                break;
            }
        }
        assert_eq!(cache.len(), 16384, "mx{{n}} tags cover every slot");
        for slot in (0u16..16384).step_by(97).chain([0, 16383]) {
            let tag = tag_for_slot(slot);
            assert_eq!(fred::util::redis_keyslot(tag.as_bytes()), slot);
            assert_eq!(tag, cache[&slot]);
        }
    }
}
