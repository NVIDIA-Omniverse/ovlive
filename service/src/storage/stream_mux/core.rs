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

//! Pure, synchronous state of one multiplexed reader: which stream keys it
//! reads, every subscription's cursor, how many entries the next XREAD asks
//! for, which keys of a reply need a head check before they fan out, and how
//! an XREAD reply fans out.
//! No I/O — the async reader in `reader.rs` drives it, and the proptests
//! drive it against an in-memory model.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use crate::observability::metrics::{
    OVERFLOW_BOUND_BYTES, OVERFLOW_BOUND_ENTRIES, STREAM_EXIT_INNER_WATCHDOG,
};

/// A stream entry id (`<ms>-<seq>`), ordered the way Valkey orders them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct StreamId {
    pub ms: u64,
    pub seq: u64,
}

impl StreamId {
    pub const ZERO: StreamId = StreamId { ms: 0, seq: 0 };

    /// Parses `ms-seq` or a bare `ms` (= `ms-0`). Rejects `$`, `+`, `-`, empty.
    pub fn parse(s: &str) -> Option<Self> {
        let (ms, seq) = match s.split_once('-') {
            Some((ms, seq)) => (ms, seq),
            None => (s, "0"),
        };
        let all_digits = |x: &str| !x.is_empty() && x.bytes().all(|b| b.is_ascii_digit());
        if !all_digits(ms) || !all_digits(seq) {
            return None;
        }
        Some(StreamId {
            ms: ms.parse().ok()?,
            seq: seq.parse().ok()?,
        })
    }

    /// `(ms + 1)-0`: the id of the commit that follows `self` on a layer
    /// stream, whose update entries are `{sequence_num}-0`. A stream whose
    /// next entry after `self` is at or below it skipped no commit.
    pub fn successor(self) -> StreamId {
        StreamId {
            ms: self.ms.saturating_add(1),
            seq: 0,
        }
    }
}

impl fmt::Display for StreamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}-{}", self.ms, self.seq)
    }
}

pub type SubId = u64;

/// Why a subscription was ended by the reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub reason: &'static str,
    pub message: String,
}

pub enum Offer {
    Accepted,
    /// The subscriber is gone; drop it silently.
    Closed,
    /// The subscriber fell too far behind; end it with an inner-watchdog
    /// failure. Carries which of its backlog limits ended it.
    Overflow(OverflowBound),
}

/// Which backlog limit ended a subscription that fell behind: the
/// `bound` label on `live_edit_stream_mux_overflow_kills_total`.
///
/// `Bytes` means a byte limit was reached and the entry limit was not; every
/// other overflow, both limits reached included, is `Entries`. So `Bytes`
/// counts exactly the endings an entry-only backlog would not have caused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowBound {
    Entries,
    Bytes,
}

impl OverflowBound {
    /// The `bound` label value.
    pub fn label(self) -> &'static str {
        match self {
            OverflowBound::Entries => OVERFLOW_BOUND_ENTRIES,
            OverflowBound::Bytes => OVERFLOW_BOUND_BYTES,
        }
    }
}

/// Where one subscription's entries go.
pub trait SubSink<P> {
    /// Offers one entry of `bytes` bytes ([`Entry::bytes`]).
    fn offer(&mut self, id: &str, payload: &P, bytes: u64) -> Offer;
    /// Offers a trim marker, queued ahead of the entries the same reply then
    /// offers. `resumes_at` is the id of the key's first entry in that reply,
    /// whether or not it can be delivered (an entry that cannot be converted
    /// is skipped, and the subscription receives the next one); it is past
    /// the entry after this subscription's cursor, and the head check found
    /// the stream's head past the read position, so the ids between the
    /// cursor and `resumes_at` were no longer in the stream when the reader
    /// read it. The consumer decides whether its catch-up already covered
    /// them. A marker is not an entry: it counts against no backlog limit, so
    /// a sink answers `Accepted` or `Closed`.
    fn offer_trimmed(&mut self, resumes_at: StreamId) -> Offer;
    /// Ends the subscription with `failure`. The failure must be observable by
    /// the consumer before it sees end-of-stream.
    fn fail(self, failure: Failure);
}

/// One parsed stream entry. `payload: None` is an entry that cannot be
/// delivered (malformed); it still advances cursors.
#[derive(Debug, Clone)]
pub struct Entry<P> {
    pub id: StreamId,
    pub id_str: String,
    pub payload: Option<P>,
    /// The entry's size as the reader measured it in the reply: its id plus
    /// its `payload` and `ompf` field values, in bytes. It sizes the key's
    /// next read ([`ReaderCore::plan_read`]) and is what each subscription's
    /// byte backlog counts for the entry ([`SubSink::offer`]).
    pub bytes: u64,
}

/// How [`ReaderCore::plan_read`] sizes a read's `COUNT` so that one reply
/// stays near a byte target.
///
/// `COUNT = clamp(⌊target_bytes ÷ Σ_k est_k⌋, 1, max_count)`, summed over the
/// keys the read names (the wake key excluded). `est_k` is the largest entry
/// in `k`'s last non-empty reply; a key with no such reply yet uses
/// `max(largest est_k the reader holds, default_entry_estimate)`. The bound is
/// soft: entries larger than their key's estimate over-commit the reply by at
/// most one reply's worth of the new size, after which the estimate follows
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadSizing {
    /// Bytes one reply aims at ([`super::XREAD_REPLY_TARGET_BYTES`]).
    pub target_bytes: u64,
    /// The most entries requested per stream per read
    /// ([`super::XREAD_COUNT`]).
    pub max_count: u64,
    /// Estimate for a key with no non-empty reply yet
    /// ([`super::DEFAULT_ENTRY_ESTIMATE`]).
    pub default_entry_estimate: u64,
}

impl Default for ReadSizing {
    fn default() -> Self {
        Self {
            target_bytes: super::XREAD_REPLY_TARGET_BYTES,
            max_count: super::XREAD_COUNT,
            default_entry_estimate: super::DEFAULT_ENTRY_ESTIMATE,
        }
    }
}

impl ReadSizing {
    /// `COUNT` for a read whose keys' estimates sum to `estimated_bytes`:
    /// never 0 and never above `max_count` (a `max_count` of 0 is read as 1).
    pub fn count_for(&self, estimated_bytes: u64) -> u64 {
        let per_round = estimated_bytes.max(1);
        (self.target_bytes / per_round).clamp(1, self.max_count.max(1))
    }
}

/// Arguments for one `XREAD COUNT count … STREAMS keys… ids…`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadPlan {
    pub keys: Vec<String>,
    pub ids: Vec<String>,
    /// Entries requested per stream ([`ReadSizing`]).
    pub count: u64,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReadOutcome {
    pub delivered: usize,
    pub closed: usize,
    /// Subscriptions ended because they fell behind, on either bound.
    pub overflowed: usize,
    /// The part of `overflowed` ended on [`OverflowBound::Bytes`].
    pub overflowed_on_bytes: usize,
    /// Trim markers accepted ([`SubSink::offer_trimmed`]).
    pub trimmed: usize,
}

/// What one head check (`XRANGE key - + COUNT 1`, run by the reader on a key
/// [`ReaderCore::gap_suspects`] named) found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Head {
    /// The stream's first retained entry id; `None` when the stream holds no
    /// entry, which makes no gap claim.
    First(Option<StreamId>),
    /// The check failed or did not answer in time. The key is left out of the
    /// reply's fan-out: its cursors stay where they were, so the next read
    /// re-reads it.
    Failed,
}

impl Head {
    /// Whether the check found the stream's head past the entry after
    /// `read_from`, the id the key was read from: the stream no longer held
    /// the ids just above the read position, so the reply's first entry sits
    /// above ids it trimmed. The one "head passed" predicate: the core's
    /// marker rule ([`ReaderCore::on_read_ok`]) and the reader's
    /// `head_checks_total{result="gap"}` classification both use it. An empty
    /// stream, a failed check, or a head at or below `successor(read_from)`
    /// makes no gap claim.
    pub fn passed(&self, read_from: StreamId) -> bool {
        matches!(self, Head::First(Some(head)) if *head > read_from.successor())
    }
}

/// Head-check results by stream key, for [`ReaderCore::on_read_ok`]. A key
/// with no entry was not checked and fans out as usual.
pub type Heads = HashMap<String, Head>;

struct SubState<S> {
    cursor: StreamId,
    sink: S,
}

struct Inflight {
    ids: BTreeMap<String, StreamId>,
    wake_key: String,
}

pub struct ReaderCore<P, S: SubSink<P>> {
    streams: BTreeMap<String, BTreeMap<SubId, SubState<S>>>,
    sub_keys: HashMap<SubId, String>,
    wake_key: Option<String>,
    wake_cursor: StreamId,
    inflight: Option<Inflight>,
    /// `est_k` per key: the largest entry in the key's last non-empty reply.
    /// Only keys the reader still reads have one; it goes with the key.
    estimates: HashMap<String, u64>,
    sizing: ReadSizing,
    _payload: std::marker::PhantomData<fn(&P)>,
}

impl<P, S: SubSink<P>> Default for ReaderCore<P, S> {
    fn default() -> Self {
        Self {
            streams: BTreeMap::new(),
            sub_keys: HashMap::new(),
            wake_key: None,
            wake_cursor: StreamId::ZERO,
            inflight: None,
            estimates: HashMap::new(),
            sizing: ReadSizing::default(),
            _payload: std::marker::PhantomData,
        }
    }
}

impl<P, S: SubSink<P>> ReaderCore<P, S> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces how later reads are sized. Estimates already held are kept.
    pub fn set_read_sizing(&mut self, sizing: ReadSizing) {
        self.sizing = sizing;
    }

    /// The size estimate `key`'s next read uses, or `None` when the reader
    /// holds none for it (no non-empty reply since the key was added).
    pub fn entry_estimate(&self, key: &str) -> Option<u64> {
        self.estimates.get(key).copied()
    }

    /// Drops `key` and everything the reader keeps for it.
    fn remove_key(&mut self, key: &str) -> Option<BTreeMap<SubId, SubState<S>>> {
        self.estimates.remove(key);
        self.streams.remove(key)
    }

    /// Registers `sub` on `key`, delivering entries strictly after `cursor`.
    pub fn add(&mut self, sub: SubId, key: String, cursor: StreamId, sink: S) {
        self.sub_keys.insert(sub, key.clone());
        self.streams
            .entry(key)
            .or_default()
            .insert(sub, SubState { cursor, sink });
    }

    /// Unregisters `sub` without failing it. Returns whether it was present.
    pub fn remove(&mut self, sub: SubId) -> bool {
        let Some(key) = self.sub_keys.remove(&sub) else {
            return false;
        };
        let Some(subs) = self.streams.get_mut(&key) else {
            return false;
        };
        let removed = subs.remove(&sub).is_some();
        if subs.is_empty() {
            self.remove_key(&key);
        }
        removed
    }

    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    pub fn subscription_count(&self) -> usize {
        self.sub_keys.len()
    }

    pub fn first_key(&self) -> Option<&str> {
        self.streams.keys().next().map(String::as_str)
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.streams.keys().map(String::as_str)
    }

    pub fn is_inflight(&self) -> bool {
        self.inflight.is_some()
    }

    /// The wake key named by the in-flight read, if any. A wake must target
    /// this key: it is the one the blocked XREAD is watching.
    pub fn inflight_wake_key(&self) -> Option<&str> {
        self.inflight.as_ref().map(|i| i.wake_key.as_str())
    }

    fn min_cursor(subs: &BTreeMap<SubId, SubState<S>>) -> StreamId {
        subs.values()
            .map(|s| s.cursor)
            .min()
            .unwrap_or(StreamId::ZERO)
    }

    /// Builds the next XREAD: each key is read from the lowest cursor of its
    /// subscriptions, and `wake_key` is appended. `COUNT` is sized from the
    /// keys' entry estimates ([`ReadSizing`]). Records the plan as in-flight.
    /// `None` when there is nothing to read.
    pub fn plan_read(&mut self, wake_key: &str) -> Option<ReadPlan> {
        if self.streams.is_empty() {
            return None;
        }
        if self.wake_key.as_deref() != Some(wake_key) {
            self.wake_key = Some(wake_key.to_owned());
            self.wake_cursor = StreamId::ZERO;
        }
        let ids: BTreeMap<String, StreamId> = self
            .streams
            .iter()
            .map(|(k, subs)| (k.clone(), Self::min_cursor(subs)))
            .collect();
        let unseen = self
            .estimates
            .values()
            .copied()
            .max()
            .unwrap_or(0)
            .max(self.sizing.default_entry_estimate);
        let estimated = ids.keys().fold(0u64, |sum, k| {
            sum.saturating_add(self.estimates.get(k).copied().unwrap_or(unseen))
        });
        let mut plan = ReadPlan {
            keys: Vec::with_capacity(ids.len() + 1),
            ids: Vec::with_capacity(ids.len() + 1),
            count: self.sizing.count_for(estimated),
        };
        for (k, id) in &ids {
            plan.keys.push(k.clone());
            plan.ids.push(id.to_string());
        }
        plan.keys.push(wake_key.to_owned());
        plan.ids.push(self.wake_cursor.to_string());
        self.inflight = Some(Inflight {
            ids,
            wake_key: wake_key.to_owned(),
        });
        Some(plan)
    }

    /// Whether the in-flight read misses something the current state needs:
    /// a key it does not name, or a subscription whose cursor is behind the
    /// id the key is being read from.
    pub fn needs_wake(&self) -> bool {
        let Some(inflight) = &self.inflight else {
            return false;
        };
        self.streams
            .iter()
            .any(|(k, subs)| match inflight.ids.get(k) {
                None => true,
                Some(from) => Self::min_cursor(subs) < *from,
            })
    }

    /// The in-flight read returned nothing (BLOCK elapsed).
    pub fn on_read_empty(&mut self) {
        self.inflight = None;
    }

    /// The in-flight read failed; nothing was consumed.
    pub fn on_read_err(&mut self) {
        self.inflight = None;
    }

    /// The keys of an XREAD reply to the in-flight read that need a head check
    /// before they fan out, each with the id it was read from.
    ///
    /// A key read from `r` qualifies when its first entry in the reply is past
    /// `r`'s successor ([`StreamId::successor`]) and is not a `Deleted`
    /// sentinel (`is_sentinel`; a sentinel's id is auto-generated, so it always
    /// jumps). This only starts a check: an id jump alone never classifies a
    /// gap, because a restore band jumps ids with no entry skipped; the head check
    /// tells the two apart ([`Self::on_read_ok`]). The wake key, and keys the
    /// reader no longer reads, never qualify.
    pub fn gap_suspects(
        &self,
        batch: &[(String, Vec<Entry<P>>)],
        is_sentinel: impl Fn(&P) -> bool,
    ) -> Vec<(String, StreamId)> {
        let Some(inflight) = &self.inflight else {
            return Vec::new();
        };
        batch
            .iter()
            .filter_map(|(key, entries)| {
                if *key == inflight.wake_key || !self.streams.contains_key(key) {
                    return None;
                }
                let read_from = *inflight.ids.get(key)?;
                let first = entries.first()?;
                if first.id <= read_from.successor()
                    || first.payload.as_ref().is_some_and(&is_sentinel)
                {
                    return None;
                }
                Some((key.clone(), read_from))
            })
            .collect()
    }

    /// Fans an XREAD reply out. Entries of key `k` go to a subscription only
    /// when its cursor is at or past the id `k` was read from (otherwise the
    /// reply does not cover its gap; the next plan re-reads from its cursor),
    /// and only entries strictly after its cursor. A non-empty reply for a key
    /// the reader still reads replaces that key's entry estimate with the
    /// largest entry in it.
    ///
    /// `heads` carries the head checks the reader ran for this reply
    /// ([`Self::gap_suspects`]). A check only classifies:
    /// - When the head passed the read position ([`Head::passed`]), the marker
    ///   carries `e1`, the id of the key's first entry in this reply (payload
    ///   or not). Each subscription the reply covers (`read_from ≤ cursor`)
    ///   whose cursor's successor is below `e1` is offered `Trimmed(e1)`
    ///   ([`SubSink::offer_trimmed`]) ahead of the key's entries; no other
    ///   subscription gets one. The reply proves the ids between such a
    ///   cursor and `e1` were not in the stream when it was read; the head's
    ///   value beyond `e1` only describes trims made after the read, which
    ///   cost a covered subscription nothing, so it never sets the marker.
    /// - A head at or below `successor(read_from)`, [`Head::First`]`(None)`
    ///   (an empty stream), or a key with no entries in the reply: no gap
    ///   claim and no marker; the key fans out as usual.
    /// - [`Head::Failed`]: the key is not fanned out and its cursors are
    ///   unchanged, so the next plan re-reads it; its entry estimate is still
    ///   taken from this reply.
    pub fn on_read_ok(
        &mut self,
        batch: Vec<(String, Vec<Entry<P>>)>,
        heads: &Heads,
    ) -> ReadOutcome {
        let mut out = ReadOutcome::default();
        let Some(inflight) = self.inflight.take() else {
            return out;
        };
        for (key, entries) in batch {
            if key == inflight.wake_key {
                if self.wake_key.as_deref() == Some(key.as_str()) {
                    if let Some(last) = entries.iter().map(|e| e.id).max() {
                        self.wake_cursor = self.wake_cursor.max(last);
                    }
                }
                continue;
            }
            let Some(read_from) = inflight.ids.get(&key).copied() else {
                continue;
            };
            let Some(subs) = self.streams.get_mut(&key) else {
                continue;
            };
            // `Some(e1)` when the head check found the head past the read
            // position: the id of the key's first entry in this reply.
            let resumes_at = match heads.get(&key) {
                Some(Head::Failed) => {
                    if let Some(largest) = entries.iter().map(|e| e.bytes).max() {
                        self.estimates.insert(key, largest);
                    }
                    continue;
                }
                Some(head) if head.passed(read_from) => entries.first().map(|e| e.id),
                Some(Head::First(_)) | None => None,
            };
            let mut dead: Vec<(SubId, Option<OverflowBound>)> = Vec::new();
            for (sub_id, st) in subs.iter_mut() {
                if st.cursor < read_from {
                    continue;
                }
                if let Some(resumes_at) = resumes_at.filter(|&e1| st.cursor.successor() < e1) {
                    match st.sink.offer_trimmed(resumes_at) {
                        Offer::Accepted => out.trimmed += 1,
                        Offer::Closed => {
                            dead.push((*sub_id, None));
                            continue;
                        }
                        Offer::Overflow(bound) => {
                            dead.push((*sub_id, Some(bound)));
                            continue;
                        }
                    }
                }
                for e in &entries {
                    if e.id <= st.cursor {
                        continue;
                    }
                    if let Some(p) = &e.payload {
                        match st.sink.offer(&e.id_str, p, e.bytes) {
                            Offer::Accepted => out.delivered += 1,
                            Offer::Closed => {
                                dead.push((*sub_id, None));
                                break;
                            }
                            Offer::Overflow(bound) => {
                                dead.push((*sub_id, Some(bound)));
                                break;
                            }
                        }
                    }
                    st.cursor = e.id;
                }
            }
            for (sub_id, overflow) in dead {
                let st = subs.remove(&sub_id).expect("dead sub present");
                self.sub_keys.remove(&sub_id);
                if let Some(bound) = overflow {
                    out.overflowed += 1;
                    if bound == OverflowBound::Bytes {
                        out.overflowed_on_bytes += 1;
                    }
                    st.sink.fail(Failure {
                        reason: STREAM_EXIT_INNER_WATCHDOG,
                        message: format!(
                            "subscriber for {key} fell behind its delivery watchdog; tearing down"
                        ),
                    });
                } else {
                    out.closed += 1;
                }
            }
            if subs.is_empty() {
                self.remove_key(&key);
            } else if let Some(largest) = entries.iter().map(|e| e.bytes).max() {
                self.estimates.insert(key, largest);
            }
        }
        out
    }

    /// Fails every subscription on the given keys. Returns how many.
    pub fn fail_keys(&mut self, keys: &[String], failure: impl Fn(&str) -> Failure) -> usize {
        let mut n = 0;
        for key in keys {
            let Some(subs) = self.remove_key(key) else {
                continue;
            };
            for (sub_id, st) in subs {
                self.sub_keys.remove(&sub_id);
                st.sink.fail(failure(key));
                n += 1;
            }
        }
        n
    }

    /// Fails every subscription this reader holds. Returns how many.
    pub fn fail_all(&mut self, failure: impl Fn(&str) -> Failure) -> usize {
        let keys: Vec<String> = self.streams.keys().cloned().collect();
        self.fail_keys(&keys, failure)
    }

    /// Current cursor of `sub` (tests and diagnostics).
    pub fn cursor_of(&self, sub: SubId) -> Option<StreamId> {
        let key = self.sub_keys.get(&sub)?;
        self.streams.get(key)?.get(&sub).map(|s| s.cursor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Default)]
    struct Log {
        /// Accepted entries only.
        got: Vec<(SubId, String)>,
        /// Accepted entries and trim markers, in the order offered; a marker
        /// reads `trim@{resumes_at}`.
        seen: Vec<(SubId, String)>,
        /// Accepted trim markers and the `resumes_at` each carried.
        trimmed: Vec<(SubId, StreamId)>,
        /// The `bytes` each accepted offer carried.
        offered_bytes: Vec<(SubId, u64)>,
        failed: Vec<(SubId, Failure)>,
    }

    struct TestSink {
        sub: SubId,
        log: Rc<RefCell<Log>>,
        closed: bool,
        /// Entries accepted before the sink overflows (`None`: unbounded).
        room: Option<usize>,
        /// The bound an overflow reports.
        bound: OverflowBound,
    }

    impl SubSink<u32> for TestSink {
        fn offer(&mut self, id: &str, _p: &u32, bytes: u64) -> Offer {
            if self.closed {
                return Offer::Closed;
            }
            if let Some(room) = self.room.as_mut() {
                if *room == 0 {
                    return Offer::Overflow(self.bound);
                }
                *room -= 1;
            }
            let mut log = self.log.borrow_mut();
            log.got.push((self.sub, id.to_owned()));
            log.seen.push((self.sub, id.to_owned()));
            log.offered_bytes.push((self.sub, bytes));
            Offer::Accepted
        }
        fn offer_trimmed(&mut self, resumes_at: StreamId) -> Offer {
            if self.closed {
                return Offer::Closed;
            }
            let mut log = self.log.borrow_mut();
            log.trimmed.push((self.sub, resumes_at));
            log.seen.push((self.sub, format!("trim@{resumes_at}")));
            Offer::Accepted
        }
        fn fail(self, failure: Failure) {
            self.log.borrow_mut().failed.push((self.sub, failure));
        }
    }

    fn sink(sub: SubId, log: &Rc<RefCell<Log>>) -> TestSink {
        TestSink {
            sub,
            log: log.clone(),
            closed: false,
            room: None,
            bound: OverflowBound::Entries,
        }
    }

    fn id(ms: u64) -> StreamId {
        StreamId { ms, seq: 0 }
    }

    fn entries(ids: &[u64]) -> Vec<Entry<u32>> {
        ids.iter()
            .map(|&ms| Entry {
                id: id(ms),
                id_str: format!("{ms}-0"),
                payload: Some(ms as u32),
                bytes: 0,
            })
            .collect()
    }

    /// Entries `(ms, bytes)`: ids `{ms}-0`, each measured at `bytes`.
    fn sized(entries: &[(u64, u64)]) -> Vec<Entry<u32>> {
        entries
            .iter()
            .map(|&(ms, bytes)| Entry {
                id: id(ms),
                id_str: format!("{ms}-0"),
                payload: Some(ms as u32),
                bytes,
            })
            .collect()
    }

    /// The production sizing: a 1 MiB target, a 512 ceiling, 16 KiB for an
    /// unseen key.
    const SIZING: ReadSizing = ReadSizing {
        target_bytes: 1024 * 1024,
        max_count: 512,
        default_entry_estimate: 16 * 1024,
    };

    fn got(log: &Rc<RefCell<Log>>, sub: SubId) -> Vec<String> {
        log.borrow()
            .got
            .iter()
            .filter(|(s, _)| *s == sub)
            .map(|(_, i)| i.clone())
            .collect()
    }

    #[test]
    fn stream_id_parses_full_and_bare_ids_and_rejects_specials() {
        assert_eq!(StreamId::parse("5-3"), Some(StreamId { ms: 5, seq: 3 }));
        assert_eq!(StreamId::parse("7"), Some(StreamId { ms: 7, seq: 0 }));
        assert_eq!(StreamId::parse("0-0"), Some(StreamId::ZERO));
        for bad in ["$", "+", "-", "", "a-1", "1-", "-1", "1-2-3", " 1"] {
            assert_eq!(StreamId::parse(bad), None, "{bad:?}");
        }
        assert!(StreamId::parse("2-0") > StreamId::parse("1-99"));
        assert_eq!(StreamId::parse("12-4").unwrap().to_string(), "12-4");
    }

    #[test]
    fn plan_reads_each_key_from_its_lowest_cursor_and_appends_the_wake_key() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        assert_eq!(core.plan_read("w"), None);
        core.add(1, "a".into(), id(5), sink(1, &log));
        core.add(2, "a".into(), id(3), sink(2, &log));
        core.add(3, "b".into(), id(9), sink(3, &log));
        let plan = core.plan_read("w").unwrap();
        assert_eq!(plan.keys, ["a", "b", "w"]);
        assert_eq!(plan.ids, ["3-0", "9-0", "0-0"]);
    }

    #[test]
    fn delivery_is_ordered_and_strictly_after_each_cursor() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.add(1, "a".into(), id(3), sink(1, &log));
        core.add(2, "a".into(), id(5), sink(2, &log));
        core.plan_read("w").unwrap();
        let out = core.on_read_ok(vec![("a".into(), entries(&[4, 5, 6]))], &Heads::new());
        assert_eq!(out.delivered, 4);
        assert_eq!(got(&log, 1), ["4-0", "5-0", "6-0"]);
        assert_eq!(got(&log, 2), ["6-0"]);
        assert_eq!(core.cursor_of(1), Some(id(6)));
    }

    #[test]
    fn a_late_joiner_behind_the_inflight_read_is_served_by_the_next_read_not_this_one() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.add(1, "a".into(), id(10), sink(1, &log));
        core.plan_read("w").unwrap();
        core.add(2, "a".into(), id(7), sink(2, &log));
        assert!(core.needs_wake());
        core.on_read_ok(vec![("a".into(), entries(&[11]))], &Heads::new());
        assert!(
            got(&log, 2).is_empty(),
            "(7, 10] was not covered by this reply"
        );
        let plan = core.plan_read("w").unwrap();
        assert_eq!(plan.ids[0], "7-0");
        core.on_read_ok(vec![("a".into(), entries(&[8, 9, 10, 11]))], &Heads::new());
        assert_eq!(got(&log, 2), ["8-0", "9-0", "10-0", "11-0"]);
        assert_eq!(got(&log, 1), ["11-0"], "no duplicates for the covered sub");
    }

    #[test]
    fn needs_wake_truth_table() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.add(1, "a".into(), id(5), sink(1, &log));
        assert!(!core.needs_wake(), "nothing in flight");
        core.plan_read("w").unwrap();
        assert!(!core.needs_wake());
        core.add(2, "a".into(), id(6), sink(2, &log));
        assert!(!core.needs_wake(), "ahead of the read cursor");
        core.add(3, "a".into(), id(5), sink(3, &log));
        assert!(!core.needs_wake(), "equal to the read cursor");
        core.remove(2);
        assert!(!core.needs_wake(), "removal");
        core.add(4, "b".into(), id(0), sink(4, &log));
        assert!(core.needs_wake(), "new key");
        core.remove(4);
        core.add(5, "a".into(), id(4), sink(5, &log));
        assert!(core.needs_wake(), "behind the read cursor");
    }

    #[test]
    fn wake_entries_advance_the_wake_cursor_and_are_never_delivered() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.add(1, "a".into(), id(0), sink(1, &log));
        core.plan_read("w").unwrap();
        core.on_read_ok(vec![("w".into(), entries(&[42]))], &Heads::new());
        assert!(got(&log, 1).is_empty());
        assert_eq!(core.plan_read("w").unwrap().ids[1], "42-0");
        core.on_read_empty();
        assert_eq!(
            core.plan_read("w2").unwrap().ids[1],
            "0-0",
            "a different wake key starts from zero"
        );
    }

    #[test]
    fn undeliverable_entries_advance_the_cursor() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.add(1, "a".into(), id(0), sink(1, &log));
        core.plan_read("w").unwrap();
        let mut e = entries(&[1, 2]);
        e[0].payload = None;
        core.on_read_ok(vec![("a".into(), e)], &Heads::new());
        assert_eq!(got(&log, 1), ["2-0"]);
        assert_eq!(core.cursor_of(1), Some(id(2)));
    }

    #[test]
    fn replies_for_removed_keys_are_ignored() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.add(1, "a".into(), id(0), sink(1, &log));
        core.plan_read("w").unwrap();
        assert!(core.remove(1));
        assert!(!core.remove(1));
        let out = core.on_read_ok(vec![("a".into(), entries(&[1]))], &Heads::new());
        assert_eq!(out, ReadOutcome::default());
        assert!(core.is_empty());
    }

    #[test]
    fn a_closed_sink_is_dropped_and_its_key_removed_when_last() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        let mut s = sink(1, &log);
        s.closed = true;
        core.add(1, "a".into(), id(0), s);
        core.plan_read("w").unwrap();
        let out = core.on_read_ok(vec![("a".into(), entries(&[1]))], &Heads::new());
        assert_eq!(out.closed, 1);
        assert!(core.is_empty());
        assert!(log.borrow().failed.is_empty(), "closed is not a failure");
    }

    #[test]
    fn an_overflowing_sink_is_failed_with_inner_watchdog_and_siblings_keep_receiving() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        let mut slow = sink(1, &log);
        slow.room = Some(1);
        core.add(1, "a".into(), id(0), slow);
        core.add(2, "a".into(), id(0), sink(2, &log));
        core.plan_read("w").unwrap();
        let out = core.on_read_ok(vec![("a".into(), entries(&[1, 2, 3]))], &Heads::new());
        assert_eq!(out.overflowed, 1);
        assert_eq!(got(&log, 2), ["1-0", "2-0", "3-0"]);
        let failed = &log.borrow().failed;
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].1.reason, STREAM_EXIT_INNER_WATCHDOG);
        assert_eq!(core.subscription_count(), 1);
    }

    #[test]
    fn fail_keys_only_ends_subscriptions_on_those_keys() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.add(1, "a".into(), id(0), sink(1, &log));
        core.add(2, "b".into(), id(0), sink(2, &log));
        let f = |k: &str| Failure {
            reason: "error",
            message: k.to_owned(),
        };
        assert_eq!(core.fail_keys(&["a".into()], f), 1);
        assert_eq!(core.keys().collect::<Vec<_>>(), ["b"]);
        assert_eq!(core.fail_all(f), 1);
        assert!(core.is_empty());
        assert_eq!(log.borrow().failed.len(), 2);
    }

    // ── COUNT sizing ────────────────────────────────────────────────────────

    /// `COUNT = clamp(⌊target ÷ estimated⌋, 1, max_count)`: never 0 (a read
    /// always makes progress), never above the ceiling, and a zero estimate or
    /// a zero ceiling cannot break the arithmetic.
    #[test]
    fn count_for_is_the_clamped_quotient_never_zero_never_above_the_ceiling() {
        const MIB: u64 = 1024 * 1024;
        for (estimated, want) in [
            (0, 512),
            (1, 512),
            (2048, 512),
            (2049, 511),
            (16 * 1024, 64),
            (32 * 1024, 32),
            (300_000, 3),
            (MIB, 1),
            (MIB + 1, 1),
            (u64::MAX, 1),
        ] {
            assert_eq!(SIZING.count_for(estimated), want, "estimated {estimated}");
        }
        let no_ceiling = ReadSizing {
            max_count: 0,
            ..SIZING
        };
        assert_eq!(no_ceiling.count_for(1), 1, "a zero ceiling reads as 1");
        let no_target = ReadSizing {
            target_bytes: 0,
            ..SIZING
        };
        assert_eq!(no_target.count_for(1), 1, "a zero target still reads one");
    }

    /// With no reply seen yet, every key counts at the default estimate.
    #[test]
    fn plan_read_estimates_unseen_keys_at_the_default() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.set_read_sizing(SIZING);
        core.add(1, "a".into(), id(0), sink(1, &log));
        assert_eq!(core.plan_read("w").unwrap().count, 64, "1 MiB ÷ 16 KiB");
        core.on_read_empty();
        core.add(2, "b".into(), id(0), sink(2, &log));
        assert_eq!(
            core.plan_read("w").unwrap().count,
            32,
            "1 MiB ÷ (2 × 16 KiB)"
        );
        assert_eq!(core.entry_estimate("a"), None);
    }

    /// A key's estimate is the largest entry in its last non-empty reply, and
    /// follows the entries both up and down; a reply with no entry for the key
    /// keeps it.
    #[test]
    fn a_non_empty_reply_replaces_the_keys_estimate_with_its_largest_entry() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.set_read_sizing(SIZING);
        core.add(1, "a".into(), id(0), sink(1, &log));

        core.plan_read("w").unwrap();
        core.on_read_ok(
            vec![("a".into(), sized(&[(1, 100), (2, 300), (3, 200)]))],
            &Heads::new(),
        );
        assert_eq!(core.entry_estimate("a"), Some(300));
        assert_eq!(
            core.plan_read("w").unwrap().count,
            512,
            "1 MiB ÷ 300 B, clamped"
        );

        core.on_read_ok(vec![("a".into(), sized(&[(4, 400_000)]))], &Heads::new());
        assert_eq!(core.entry_estimate("a"), Some(400_000));
        assert_eq!(core.plan_read("w").unwrap().count, 2, "⌊1 MiB ÷ 400,000 B⌋");

        core.on_read_ok(vec![("a".into(), sized(&[(5, 50_000)]))], &Heads::new());
        assert_eq!(
            core.entry_estimate("a"),
            Some(50_000),
            "the estimate follows down"
        );

        core.plan_read("w").unwrap();
        core.on_read_ok(vec![("a".into(), Vec::new())], &Heads::new());
        assert_eq!(
            core.entry_estimate("a"),
            Some(50_000),
            "a reply with no entry for the key keeps its estimate"
        );
        assert_eq!(core.plan_read("w").unwrap().count, 20, "⌊1 MiB ÷ 50,000 B⌋");
        assert_eq!(
            log.borrow().offered_bytes,
            [(1, 100), (1, 300), (1, 200), (1, 400_000), (1, 50_000)],
            "each entry is offered with its own measured size"
        );
    }

    /// A key with no estimate yet is counted at the largest estimate the
    /// reader holds when that is above the default — so a new key joining a
    /// reader of large entries does not inflate the read — and at the default
    /// otherwise. Estimates are summed over every key the read names.
    #[test]
    fn an_unseen_key_is_estimated_at_the_larger_of_the_readers_largest_and_the_default() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.set_read_sizing(SIZING);
        core.add(1, "big".into(), id(0), sink(1, &log));
        core.plan_read("w").unwrap();
        core.on_read_ok(
            vec![("big".into(), sized(&[(1, 64 * 1024)]))],
            &Heads::new(),
        );

        core.add(2, "new".into(), id(0), sink(2, &log));
        assert_eq!(
            core.plan_read("w").unwrap().count,
            8,
            "1 MiB ÷ (64 KiB + an unseen key at the reader's 64 KiB)"
        );
        core.on_read_ok(vec![("new".into(), sized(&[(1, 1024)]))], &Heads::new());
        assert_eq!(core.entry_estimate("new"), Some(1024));
        assert_eq!(
            core.plan_read("w").unwrap().count,
            15,
            "⌊1 MiB ÷ (64 KiB + 1 KiB)⌋"
        );
        core.on_read_empty();

        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.set_read_sizing(SIZING);
        core.add(1, "small".into(), id(0), sink(1, &log));
        core.plan_read("w").unwrap();
        core.on_read_ok(vec![("small".into(), sized(&[(1, 100)]))], &Heads::new());
        core.add(2, "new".into(), id(0), sink(2, &log));
        assert_eq!(
            core.plan_read("w").unwrap().count,
            63,
            "⌊1 MiB ÷ (100 B + an unseen key at the 16 KiB default)⌋"
        );
    }

    /// The wake key is never sized: its entries set no estimate and leave the
    /// COUNT alone.
    #[test]
    fn wake_entries_never_set_an_estimate() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.set_read_sizing(SIZING);
        core.add(1, "a".into(), id(0), sink(1, &log));
        core.plan_read("w").unwrap();
        core.on_read_ok(
            vec![("w".into(), sized(&[(1, 4 * 1024 * 1024)]))],
            &Heads::new(),
        );
        assert_eq!(core.entry_estimate("w"), None);
        assert_eq!(
            core.plan_read("w").unwrap().count,
            64,
            "a still at the default"
        );
    }

    /// An estimate goes with its key: once the key's last subscription is
    /// removed, or ends in the reply itself, the key is read as unseen again.
    #[test]
    fn a_keys_estimate_goes_with_the_key() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.set_read_sizing(SIZING);
        core.add(1, "a".into(), id(0), sink(1, &log));
        core.plan_read("w").unwrap();
        core.on_read_ok(vec![("a".into(), sized(&[(1, 500_000)]))], &Heads::new());
        assert_eq!(core.entry_estimate("a"), Some(500_000));
        assert!(core.remove(1));
        assert_eq!(
            core.entry_estimate("a"),
            None,
            "removed with the last subscription"
        );
        core.add(2, "a".into(), id(1), sink(2, &log));
        assert_eq!(core.plan_read("w").unwrap().count, 64, "unseen again");

        // A reply that ends the key's only subscription removes the key, and
        // records no estimate for it.
        let mut closed = sink(3, &log);
        closed.closed = true;
        core.on_read_empty();
        core.remove(2);
        core.add(3, "b".into(), id(0), closed);
        core.plan_read("w").unwrap();
        core.on_read_ok(vec![("b".into(), sized(&[(1, 700_000)]))], &Heads::new());
        assert!(core.is_empty());
        assert_eq!(core.entry_estimate("b"), None);
    }

    /// Replacing the sizing keeps the estimates already held.
    #[test]
    fn set_read_sizing_keeps_the_estimates() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.set_read_sizing(SIZING);
        core.add(1, "a".into(), id(0), sink(1, &log));
        core.plan_read("w").unwrap();
        core.on_read_ok(vec![("a".into(), sized(&[(1, 1000)]))], &Heads::new());
        core.set_read_sizing(ReadSizing {
            target_bytes: 10_000,
            ..SIZING
        });
        assert_eq!(core.entry_estimate("a"), Some(1000));
        assert_eq!(core.plan_read("w").unwrap().count, 10);
    }

    // ── Overflow bound ──────────────────────────────────────────────────────

    /// `overflowed` counts every subscription a reply ended for falling
    /// behind; `overflowed_on_bytes` the part whose sink reported
    /// `OverflowBound::Bytes`. Both end `inner_watchdog`.
    #[test]
    fn a_reply_counts_its_overflows_by_bound() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        let mut on_entries = sink(1, &log);
        on_entries.room = Some(0);
        let mut on_bytes = sink(2, &log);
        on_bytes.room = Some(1);
        on_bytes.bound = OverflowBound::Bytes;
        let mut also_on_bytes = sink(3, &log);
        also_on_bytes.room = Some(0);
        also_on_bytes.bound = OverflowBound::Bytes;
        core.add(1, "a".into(), id(0), on_entries);
        core.add(2, "a".into(), id(0), on_bytes);
        core.add(3, "b".into(), id(0), also_on_bytes);
        core.add(4, "b".into(), id(0), sink(4, &log));
        core.plan_read("w").unwrap();
        let out = core.on_read_ok(
            vec![("a".into(), entries(&[1, 2])), ("b".into(), entries(&[1]))],
            &Heads::new(),
        );
        assert_eq!(out.overflowed, 3);
        assert_eq!(out.overflowed_on_bytes, 2);
        assert_eq!(out.delivered, 2, "sub 2's first entry and sub 4's");
        let failed = &log.borrow().failed;
        assert_eq!(failed.len(), 3);
        assert!(failed
            .iter()
            .all(|(_, f)| f.reason == STREAM_EXIT_INNER_WATCHDOG));
        assert_eq!(core.subscription_count(), 1);
    }

    #[test]
    fn overflow_bound_labels_are_the_metric_vocabulary() {
        assert_eq!(OverflowBound::Entries.label(), "entries");
        assert_eq!(OverflowBound::Bytes.label(), "bytes");
        assert_eq!(
            crate::observability::metrics::OVERFLOW_BOUNDS,
            [OverflowBound::Entries.label(), OverflowBound::Bytes.label()]
        );
    }

    // ── Trimmed streams: the gap trigger, the head check and the marker ─────

    /// The payload the tests below treat as a `Deleted` sentinel.
    const SENTINEL: u32 = u32::MAX;

    fn is_sentinel(p: &u32) -> bool {
        *p == SENTINEL
    }

    /// Entries `{ms}-0`, the one at `ms == sentinel_at` a sentinel.
    fn entries_with_sentinel(ids: &[u64], sentinel_at: u64) -> Vec<Entry<u32>> {
        let mut e = entries(ids);
        for entry in &mut e {
            if entry.id == id(sentinel_at) {
                entry.payload = Some(SENTINEL);
            }
        }
        e
    }

    /// Every entry and marker `sub` was offered, in order.
    fn seen(log: &Rc<RefCell<Log>>, sub: SubId) -> Vec<String> {
        log.borrow()
            .seen
            .iter()
            .filter(|(s, _)| *s == sub)
            .map(|(_, i)| i.clone())
            .collect()
    }

    fn heads(key: &str, head: Head) -> Heads {
        Heads::from([(key.to_owned(), head)])
    }

    #[test]
    fn successor_is_the_next_commit_id_and_saturates() {
        assert_eq!(id(5).successor(), id(6));
        assert_eq!(StreamId { ms: 5, seq: 7 }.successor(), id(6));
        assert_eq!(StreamId::ZERO.successor(), id(1));
        assert_eq!(id(u64::MAX).successor(), id(u64::MAX));
    }

    /// The one "head passed" predicate: only a head strictly above the entry
    /// after the read position claims a gap.
    #[test]
    fn head_passed_only_above_the_successor_of_the_read_position() {
        let r = id(5);
        assert!(!Head::First(Some(id(5))).passed(r), "head at the read id");
        assert!(!Head::First(Some(id(6))).passed(r), "head at its successor");
        assert!(Head::First(Some(StreamId { ms: 6, seq: 1 })).passed(r));
        assert!(Head::First(Some(id(7))).passed(r));
        assert!(
            !Head::First(None).passed(r),
            "an empty stream claims nothing"
        );
        assert!(!Head::Failed.passed(r), "a failed check claims nothing");
    }

    /// T4.1: a key qualifies for a head check only when its first reply entry
    /// is past the successor of the id it was read from and is not a sentinel.
    /// The wake key, a key no longer read, an empty entry list and a
    /// contiguous first entry never qualify; a sentinel later in the reply
    /// does not stop an update first entry from qualifying.
    #[test]
    fn gap_suspects_only_a_non_sentinel_first_entry_past_the_successor_of_the_read_id() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        for (sub, key) in [
            (1, "contiguous"),
            (2, "jumped"),
            (3, "sentinel_first"),
            (4, "sentinel_later"),
            (5, "empty"),
            (6, "gone"),
        ] {
            core.add(sub, key.into(), id(5), sink(sub, &log));
        }
        assert!(
            core.gap_suspects(&[("jumped".into(), entries(&[9]))], is_sentinel)
                .is_empty(),
            "nothing is suspect without a read in flight"
        );
        core.plan_read("w").unwrap();
        assert!(core.remove(6));
        let batch = vec![
            ("contiguous".into(), entries(&[6, 7])),
            ("jumped".into(), entries(&[7, 8])),
            ("sentinel_first".into(), entries_with_sentinel(&[40], 40)),
            ("sentinel_later".into(), entries_with_sentinel(&[9, 40], 40)),
            ("empty".into(), Vec::new()),
            ("gone".into(), entries(&[50])),
            ("w".into(), entries(&[99])),
        ];
        let mut suspects = core.gap_suspects(&batch, is_sentinel);
        suspects.sort();
        assert_eq!(
            suspects,
            [
                ("jumped".to_owned(), id(5)),
                ("sentinel_later".to_owned(), id(5)),
            ]
        );
    }

    /// T4.2, the read-to-check race: read from 5, A at 5 and B at 9, reply
    /// `10, 11, 12`, and by the check 10 and 11 are gone too (head 12). A was
    /// denied 6..9 at the read and gets `Trimmed(10)` — the reply's first entry,
    /// not the head — ahead of its entries. B is served contiguously by the
    /// reply and gets no marker. The head-valued rule would give both
    /// `Trimmed(12)`.
    #[test]
    fn a_trim_after_the_read_marks_only_the_subscriptions_the_reply_skipped() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.add(1, "a".into(), id(5), sink(1, &log));
        core.add(2, "a".into(), id(9), sink(2, &log));
        core.plan_read("w").unwrap();
        let out = core.on_read_ok(
            vec![("a".into(), entries(&[10, 11, 12]))],
            &heads("a", Head::First(Some(id(12)))),
        );
        assert_eq!(seen(&log, 1), ["trim@10-0", "10-0", "11-0", "12-0"]);
        assert_eq!(seen(&log, 2), ["10-0", "11-0", "12-0"]);
        assert_eq!(log.borrow().trimmed, [(1, id(10))]);
        assert_eq!(out.trimmed, 1);
        assert_eq!(out.delivered, 6);
    }

    /// T4.3: a subscription at or after the reply's first entry is owed
    /// nothing the reply lacks, whatever its next entry is — here an update
    /// after an id jump, and a `Deleted` sentinel. C at 11 gets no marker and
    /// receives `40`. Marking by the next entry after the cursor would give C
    /// `Trimmed(40)` and end it before a clean re-bootstrap.
    #[test]
    fn a_subscription_at_or_after_the_first_reply_entry_gets_no_marker_before_a_jump() {
        for (case, reply) in [
            ("an update after an id jump", entries(&[10, 11, 40])),
            (
                "a Deleted sentinel",
                entries_with_sentinel(&[10, 11, 40], 40),
            ),
        ] {
            let log = Rc::default();
            let mut core = ReaderCore::<u32, TestSink>::new();
            core.add(1, "a".into(), id(5), sink(1, &log));
            core.add(3, "a".into(), id(11), sink(3, &log));
            core.plan_read("w").unwrap();
            core.on_read_ok(
                vec![("a".into(), reply)],
                &heads("a", Head::First(Some(id(40)))),
            );
            assert_eq!(seen(&log, 3), ["40-0"], "{case}");
            assert_eq!(log.borrow().trimmed, [(1, id(10))], "{case}");
        }
    }

    /// T4.4: a reply with no entry after D's cursor marks D with nothing — the
    /// next read re-checks it — even with the head far past.
    #[test]
    fn a_reply_with_no_entry_after_the_cursor_gives_no_marker() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.add(1, "a".into(), id(5), sink(1, &log));
        core.add(4, "a".into(), id(12), sink(4, &log));
        core.plan_read("w").unwrap();
        core.on_read_ok(
            vec![("a".into(), entries(&[10, 11, 12]))],
            &heads("a", Head::First(Some(id(40)))),
        );
        assert!(seen(&log, 4).is_empty());
        assert_eq!(core.cursor_of(4), Some(id(12)));
        assert_eq!(log.borrow().trimmed, [(1, id(10))]);
    }

    /// T4.5: a subscription whose next commit is the reply's first entry is
    /// contiguous: no marker, however far the head moved after the read.
    #[test]
    fn a_contiguous_subscription_gets_no_marker_whatever_the_head() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.add(5, "a".into(), id(9), sink(5, &log));
        core.plan_read("w").unwrap();
        let out = core.on_read_ok(
            vec![("a".into(), entries(&[10, 11]))],
            &heads("a", Head::First(Some(id(50)))),
        );
        assert_eq!(seen(&log, 5), ["10-0", "11-0"]);
        assert_eq!(out.trimmed, 0);
    }

    /// T4.6: a head at or below the successor of the read position (the ids
    /// were never written: a restore band) and an empty stream claim no gap;
    /// the key fans out as usual.
    #[test]
    fn a_head_at_or_below_the_successor_or_an_empty_stream_gives_no_marker() {
        for (case, head) in [
            (
                "head retained below the read position",
                Head::First(Some(id(1))),
            ),
            (
                "head at the read position's successor",
                Head::First(Some(id(6))),
            ),
            ("empty stream", Head::First(None)),
        ] {
            let log = Rc::default();
            let mut core = ReaderCore::<u32, TestSink>::new();
            core.add(1, "a".into(), id(5), sink(1, &log));
            core.plan_read("w").unwrap();
            let out = core.on_read_ok(
                vec![("a".into(), entries(&[1005, 1006]))],
                &heads("a", head),
            );
            assert_eq!(seen(&log, 1), ["1005-0", "1006-0"], "{case}");
            assert_eq!(out.trimmed, 0, "{case}");
        }
    }

    /// T4.6: a failed head check holds its key out of the reply — nothing
    /// offered, cursors unchanged, so the next plan re-reads it from the same
    /// id — while its estimate is taken from the reply and the reply's other
    /// keys fan out.
    #[test]
    fn a_failed_head_check_holds_its_key_out_and_keeps_its_cursors() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.set_read_sizing(SIZING);
        core.add(1, "a".into(), id(5), sink(1, &log));
        core.add(2, "b".into(), id(0), sink(2, &log));
        core.plan_read("w").unwrap();
        let out = core.on_read_ok(
            vec![
                ("a".into(), sized(&[(10, 700), (11, 900)])),
                ("b".into(), sized(&[(1, 100)])),
            ],
            &heads("a", Head::Failed),
        );
        assert!(seen(&log, 1).is_empty(), "nothing of `a` is offered");
        assert_eq!(core.cursor_of(1), Some(id(5)));
        assert_eq!(core.entry_estimate("a"), Some(900));
        assert_eq!(got(&log, 2), ["1-0"], "`b` fans out");
        assert_eq!(out.delivered, 1);
        assert_eq!(out.trimmed, 0);
        let plan = core.plan_read("w").unwrap();
        assert_eq!(plan.keys[0], "a");
        assert_eq!(plan.ids[0], "5-0", "`a` is re-read from the same id");
    }

    /// T4.7: the marker is offered before the key's entries, and carries the
    /// reply's first entry however far the head moved after the read.
    #[test]
    fn the_marker_precedes_the_entries_and_does_not_depend_on_the_head() {
        for head in [10, 12, 1_000] {
            let log = Rc::default();
            let mut core = ReaderCore::<u32, TestSink>::new();
            core.add(1, "a".into(), id(5), sink(1, &log));
            core.plan_read("w").unwrap();
            core.on_read_ok(
                vec![("a".into(), entries(&[10, 11, 12]))],
                &heads("a", Head::First(Some(id(head)))),
            );
            assert_eq!(
                seen(&log, 1),
                ["trim@10-0", "10-0", "11-0", "12-0"],
                "head {head}"
            );
        }
    }

    /// The marker is `e1` even when `e1` cannot be delivered: an unconvertible
    /// first entry is skipped, and the subscription receives the next one.
    #[test]
    fn the_marker_carries_the_first_reply_entry_even_when_it_is_undeliverable() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.add(1, "a".into(), id(5), sink(1, &log));
        core.plan_read("w").unwrap();
        let mut reply = entries(&[10, 11]);
        reply[0].payload = None;
        core.on_read_ok(
            vec![("a".into(), reply)],
            &heads("a", Head::First(Some(id(10)))),
        );
        assert_eq!(seen(&log, 1), ["trim@10-0", "11-0"]);
        assert_eq!(core.cursor_of(1), Some(id(11)));
    }

    /// A subscription the reply does not cover (its cursor is behind the id
    /// the key was read from) gets neither a marker nor entries: the next plan
    /// re-reads from its cursor. A closed sink refuses the marker and is
    /// dropped as closed, never counted as trimmed.
    #[test]
    fn an_uncovered_subscription_gets_no_marker_and_a_closed_one_is_dropped() {
        let log = Rc::default();
        let mut core = ReaderCore::<u32, TestSink>::new();
        core.add(1, "a".into(), id(5), sink(1, &log));
        let mut closed = sink(3, &log);
        closed.closed = true;
        core.add(3, "a".into(), id(5), closed);
        core.plan_read("w").unwrap();
        core.add(2, "a".into(), id(3), sink(2, &log));
        let out = core.on_read_ok(
            vec![("a".into(), entries(&[10]))],
            &heads("a", Head::First(Some(id(10)))),
        );
        assert!(seen(&log, 2).is_empty(), "behind the read: not covered");
        assert_eq!(core.cursor_of(2), Some(id(3)));
        assert_eq!(out.closed, 1);
        assert_eq!(out.trimmed, 1, "only sub 1's marker was accepted");
        assert_eq!(core.cursor_of(3), None);
        assert_eq!(log.borrow().trimmed, [(1, id(10))]);
    }
}
