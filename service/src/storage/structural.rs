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

//! The structural commit: one fenced script call that lands a structural
//! `UpdateLayer` plan computed on the per-replica cache — contract C4 of the
//! structural-edits plan.
//!
//! Section references: `SR-n.m` is the structural-edits spec SDD (Spec Bee
//! `b.xes`); `§n.m` is `docs-internal/sdd.md`.
//!
//! # Who calls it
//!
//! Under the `lua` backend, `update_layer` commits every structural and reset
//! delta through [`commit_structural`]: `LayerStore::update_layer_once_structural`
//! plans on the checked-out structural-cache entry, commits, and settles the
//! entry by the [`CommitOutcome`]. An object the cache cannot hold, a plan
//! over the size gate's work limit (`storage::structural_size_gate`), and the
//! `lua-watch-structural` and `txn` backends, still take the WATCH path. What
//! this module fixes is the commit's inputs, its outcomes and its counting
//! hook ([`CommitObserver`]), so the commit side and the cache side
//! ([`super::structural_cache`]) are built against one set of signatures. This
//! file has one owner, the commit track (plan `b.dkt`, Epic 5); the items
//! below are the cross-track contract and change only through the plan's
//! decider.
//!
//! # The commit (SR-6)
//!
//! A structural attempt plans in place on its checked-out cache entry
//! (SR-3.5, SR-5.1; `CheckedOut::plan_in_place`), collecting today's
//! `apply_delta` `StorageMutation`s, and
//! hands them to [`commit_structural`] as a [`StructuralCommit`]. The commit
//! is one `EVALSHA` of `update_layer_fast.lua` in its structural mode on the
//! data pool (SR-6.1, SR-6.6): no `txn_pool` slot, no `WATCH` / `MULTI`.
//! The script runs the fast mode's guard skeleton plus two structural-only
//! guards (SR-6.2):
//!
//! - **the fence** — the header's `seq` and `commit_chain` must equal
//!   [`StructuralCommit::expected_seq`] and
//!   [`StructuralCommit::expected_commit_chain`], the cut the plan was made
//!   at (SR-4.5, SR-6.4), else [`CommitOutcome::Stale`];
//! - **the pre-image digests** — the stored section rows and children lists
//!   the plan rewrites, and the field and time-sample rows it deletes, must
//!   match what the cache believed
//!   ([`StructuralCommit::pre_image_digests`], SR-6.3), else
//!   [`CommitOutcome::CacheDiverged`]. Field and time-sample rows the plan
//!   sets are rewritten absolutely and carry no digest.
//!
//! The commit also carries, beside the digests, the cache's pre-plan count of
//! the value rows the plan does **not** touch that hold each external hash
//! the plan's touched rows name. The script's external-value settle uses it
//! only as a floor: a touched hash's `obj:ext` count ends at
//! `max(stored + delta, untouched + ends)`, so the cache's recount can raise
//! an under-counted index but never lower a count or cause a release (see
//! [`PreImageDigests`] and `KeySchema::obj_ext_index`'s `# Drift`).
//!
//! Every refusal writes nothing. The stream entry the commit `XADD`s is
//! byte-identical to the fast path's (`update|…` payload plus the writer's
//! raw OMPF), and the commit folds the header's `commit_chain` over that
//! payload like every other commit (SR-4.5).

use std::collections::{HashMap, HashSet};

use bytes::Bytes;

use super::layer::{
    decode_commit_token_reply, decode_committed_reply, decode_fenced_reply, external_hash_of,
    section_hash_fields, CommitIdentity, CommitTokenReply, LandedRecord, LayerStore, Operation,
    RecordedBy, RecordedCommit,
};
use super::lua::{
    children_list_digest, encode_mutations, evalsha_once_with_reload, script_call_bytes,
    UpdateScriptMode, UpdateScriptPrefix, UPDATE_LAYER_FAST_SCRIPT,
};
use crate::domain::delta::StorageMutation;
use crate::domain::pass_timing::{self, Pass};
use crate::domain::plan_journal::{PreImageView, PrePlanLens, RowImage};
use crate::domain::time::now_micros;
use crate::domain::types::{FieldRows, FieldValue, ObjectStateOf, SampleRows, Section, ValueRows};
use crate::error::ServiceError;

/// The structural-mode refusal tags of `update_layer_fast.lua` (its "Return
/// value"), each an error reply answered as a [`CommitOutcome`] rather than
/// an error. `STALE` carries a decimal after one space; `CACHE_DIVERGED` is
/// followed by `:` and a free-text detail. (`FENCED` is not among them: it is
/// a success reply both modes give, decoded by `layer::decode_fenced_reply`.)
const STALE_TAG: &str = "STALE";
const CACHE_DIVERGED_TAG: &str = "CACHE_DIVERGED";

/// The test-only hook [`commit_structural`] calls at its external-value
/// release enqueue, with the release tokens the script returned. Installed by
/// `LayerStore::with_structural_release_enqueue_hook_for_test`.
#[cfg(any(test, feature = "test-fault-injection"))]
pub type ReleaseEnqueueHook = std::sync::Arc<dyn Fn(&[String]) + Send + Sync>;

/// One structural attempt's commit request (SR-6).
///
/// Built from the checked-out cache entry the attempt planned on and the
/// mutations that plan produced. It carries no pre-plan state: the
/// pre-images the script checks, and the untouched-row counts its
/// external-value settle floors on, travel in [`Self::pre_image_digests`]. The
/// server timestamp and the stream's `MAXLEN` are not inputs —
/// [`commit_structural`] computes the one and reads the other from the store.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralCommit {
    /// The call's session. Carried, with [`Self::layer_name`], because a
    /// `STREAM_ID_COLLISION` reply is disambiguated exactly as the fast path
    /// does it (`LayerStore::disambiguate_stream_collision`), which needs
    /// both (SR-6.5).
    pub session_id: String,
    /// The call's layer name (see [`Self::session_id`]).
    pub layer_name: String,
    /// The object the commit lands on.
    pub object_id: u64,
    /// The cache entry's `seq`, the header `seq` the plan was made at. The
    /// fence requires the header's `seq` to equal it (SR-6.2 guard 6). The
    /// commit lands at `expected_seq + 1`: the stream entry
    /// `{expected_seq+1}-0` (SR-6.2 step 9) and the header record's `seq`
    /// (SR-6.3, `HDR`).
    pub expected_seq: u64,
    /// The cache entry's `commit_chain` at [`Self::expected_seq`] (SR-4.5,
    /// SR-6.4).
    pub expected_commit_chain: String,
    /// The object's `next_set_order` after the plan applied — the plan's
    /// post-apply state's `next_set_order`. Carried separately because
    /// `StorageMutation::SetObjectHeader` has no such field; the script's
    /// header record writes it (SR-6.3, `HDR`).
    pub next_set_order: u64,
    /// The plan: today's complete-mode `apply_delta` mutations, in emission
    /// order, encoded 1:1 into the structural record vocabulary (SR-6.3).
    pub mutations: Vec<StorageMutation>,
    /// The cache's pre-plan beliefs the commit carries — the digests of the
    /// pre-plan rows and lists the plan rewrites, and the untouched-row counts
    /// of the external hashes its value rows name — built by
    /// [`PreImageDigests::from_plan`] from exactly [`Self::mutations`] and
    /// the entry's pre-plan state. The pair is bound: the digests carry a
    /// fingerprint of the plan they were built from, and the record encoder
    /// refuses a plan whose fingerprint differs with `ServiceError::Internal`,
    /// so [`commit_structural`] sends nothing for a mismatched pair (see "Bound
    /// to its plan" on [`PreImageDigests`]).
    pub pre_image_digests: PreImageDigests,
    /// The writer's raw inbound OMPF, `XADD`ed verbatim as the stream entry's
    /// `ompf` field — the same contract as `update_layer`'s `relay_ompf`: a
    /// state-changing commit must carry it, or the stream entry reads as an
    /// empty commit to every consumer.
    pub relay_ompf: Option<Bytes>,
    /// The call's commit token in its recorded form, the same on every
    /// attempt of the call (production: `LayerStore::update_layer`'s minted
    /// token, 32 lowercase hex digits). Without [`Self::operation`] it is the
    /// record's match key: the script's guard 3 answers
    /// [`CommitOutcome::AlreadyLanded`] when the object's `obj:wtok` record
    /// already begins with it, and a commit records it there (SR-6.2,
    /// §7.5.1). With one it rides in the record after the sequence and
    /// timestamp, and tells this call's own commit from an earlier call's.
    pub commit_token: String,
    /// The call's client operation, when it carried an `operation_id`
    /// (`layer::Operation`, with the call's writer guard already set): the
    /// commit is then matched and recorded on `op.<operation_id>`, its record
    /// carries [`Self::commit_token`] and the request digest
    /// (`op.<id>:<seq>:<server_ts>:<token>:<digest>`), and the script checks
    /// its writer guard (§7.5.3): an unchained call is refused `OVERTAKEN`
    /// when the writer's record has moved off its start record, and a chained
    /// call `PREDECESSOR` when the record does not name its predecessor.
    /// `None` keeps the per-call `<token>:<seq>:<server_ts>` record.
    ///
    /// The writer guard is fixed once per call (a start record is read ONCE
    /// and never re-read): every attempt of the call, every `STALE` /
    /// `CACHE_DIVERGED` re-plan included, passes this same operation.
    pub operation: Option<Operation>,
    /// The writing client, recorded in the stream payload and the header's
    /// `last_client_*` columns.
    pub client_id: String,
    /// The client's timestamp, recorded beside [`Self::client_id`].
    pub client_ts: u64,
}

/// The cache's pre-plan beliefs a structural commit carries (SR-6.3): the
/// pre-image digests its pre-pass checks, and the untouched-row counts its
/// external-value settle floors on.
///
/// Opaque, with no `Default`: built only by [`PreImageDigests::from_plan`]
/// from the cache's pre-plan state, and consumed only by the structural
/// record encoder ([`encode_mutations`](super::lua::encode_mutations)),
/// which renders each digest into its record and the counts into the block
/// ahead of the records. It holds one optional digest per mutation of the
/// plan, index-aligned with the plan:
///
/// - `SetSection` / `DeleteSection` — the section row's six stored columns,
///   or "absent" when the pre-plan state has no such section;
/// - `SetChildren` / `DeleteChildrenList` — the children-list digest
///   ([`children_list_digest`](super::lua::children_list_digest)) of the
///   pre-plan list, the absent marker when that list is missing or empty.
///   **Skipped** when the parent is neither the root nor a live section in the
///   pre-plan state (the normalization predicate of SR-3.2: such a stored list
///   is deliberately not held, so there is no belief to check);
/// - `DeleteField` / `DeleteTimeSample` — whether the row exists. This
///   includes the `DeleteTimeSample` a reset or cascade adds for the stored
///   spelling of a section's time 0 the state does not hold (rule E,
///   `domain::delta`'s `ZeroTimeDeletes`), whose digest is therefore "absent":
///   if the store holds that row after all — the cache's spelling of time 0
///   can trail the store's across builds or after the reconciler's
///   signed-zero merge — the commit answers `CACHE_DIVERGED` and the entry is
///   reloaded, so a delete always checks a section's whole set of stored
///   time-0 rows;
/// - every other kind — nothing.
///
/// **Every digest is the key's value in the pre-plan state**, not its value
/// immediately before the mutation: the script's pre-pass reads the store
/// before any write, so it sees the pre-plan value of every key. A key the
/// plan touches more than once is therefore checked once, on the first
/// record of a digest-carrying kind that touches it; the later records for
/// that key carry no digest. "Key" means the stored key: a `SetSection` and a
/// `DeleteSection` of one id share `sec:{sid}`, a `SetChildren` and a
/// `DeleteChildrenList` of one parent share `ch:{p}`, and a `DeleteField` /
/// `DeleteTimeSample` names one row (a time sample by the bits of its time,
/// as its key does). A `SetField` / `SetTimeSample` carries no digest, so it
/// does not use up a later delete's check of the same row.
///
/// This is how SR-6.3 reconciles with this contract: [`StructuralCommit`]
/// carries no pre-plan state, so the structural record encoder
/// (`lua::encode_mutations`) takes `&PreImageDigests` where SR-6.3 names
/// `&pre_state`.
///
/// **Untouched-row counts.** Beside the digests, the value carries, for every
/// external hash `h` in a set `H`, the number of the pre-plan state's value
/// rows that the plan does **not** touch and that hold `h` — zero included —
/// sorted by hash. The touched rows are the rows a `SetField` / `DeleteField`
/// / `SetTimeSample` / `DeleteTimeSample` of the plan names, identified as
/// their stored keys are: a field by `(section_id, key_name)`, a time sample
/// by `(section_id, time bits)`, so a `-0.0` row never excludes an untouched
/// `0.0` row. `H` holds the hash the pre-plan state believes each touched row
/// held (a time sample only when the held sample's time bits match) and each
/// touched row's final external write. A plan whose touched rows name no
/// external hash carries no counts, and the pre-plan state is not walked.
///
/// The script's settle floors each touched hash's `obj:ext` count on these
/// counts plus the touched rows' final writes, which it reads from its own
/// records (`update_layer_fast.lua` step 6b). Only the untouched rows depend
/// on the cache here, and no digest covers them, so the floor may only raise
/// a count: a cache wrong about an untouched row can at worst leak a
/// reference, never release one. Policy and rationale:
/// `KeySchema::obj_ext_index`'s `# Drift`.
///
/// **Bound to its plan.** [`StructuralCommit::mutations`] and
/// [`StructuralCommit::pre_image_digests`] are separate fields, so nothing in
/// the type stops a caller pairing a plan with digests built from another.
/// Such a pair would check the wrong keys' beliefs and let a diverged cache
/// commit. So the value also carries a fingerprint of the plan it was built
/// from, and the encoder refuses a plan whose fingerprint differs. The
/// fingerprint binds the untouched-row counts too: they are derived from the
/// same `(mutations, pre_state)` pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreImageDigests {
    /// One slot per mutation of the plan [`Self::from_plan`] was given, in
    /// plan order; `None` where that record carries no digest.
    per_record: Vec<Option<PreImage>>,
    /// The untouched-row count of each hash in `H` (see "Untouched-row
    /// counts" above), strictly ascending by hash, zeros included.
    untouched_ext_counts: Vec<([u8; 32], u64)>,
    /// [`plan_fingerprint`] of that plan.
    plan_fingerprint: [u8; 20],
}

/// One record's pre-image digest: what the cache believed the record's key
/// held before the plan (see [`PreImageDigests`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PreImage {
    /// A `SetSection` / `DeleteSection`'s row: its six stored columns in
    /// [`section_hash_fields`] order and string form, or `None` when the
    /// pre-plan state had no such section (the row must be absent).
    Section(Option<[String; 6]>),
    /// A `SetChildren` / `DeleteChildrenList`'s list:
    /// [`children_list_digest`] of the pre-plan list.
    ChildrenList(String),
    /// A `DeleteField` / `DeleteTimeSample`'s row: whether it existed.
    RowExists(bool),
}

impl PreImageDigests {
    /// Build the digests and the untouched-row counts for `mutations` from
    /// `pre_state`, the checked-out entry's normalized state before the plan
    /// ran (SR-6.3). See [`PreImageDigests`] for what each kind carries and
    /// how the counts are taken. Either row form gives the same digests and
    /// counts: they read a row's existence and its external hash only, and
    /// a time sample by its time bits.
    pub fn from_plan(mutations: &[StorageMutation], pre_state: &impl PreState) -> Self {
        // The keys a digest has already been recorded for. Only digest-carrying
        // records insert here (see the type's docs on `SetField`).
        let mut sections: HashSet<u64> = HashSet::new();
        let mut lists: HashSet<u64> = HashSet::new();
        let mut fields: HashSet<(u64, &str)> = HashSet::new();
        let mut time_samples: HashSet<(u64, u64)> = HashSet::new();

        let per_record = mutations
            .iter()
            .map(|mutation| match mutation {
                StorageMutation::SetSection { section, .. } => {
                    section_pre_image(&mut sections, section.section_id, pre_state)
                }
                StorageMutation::DeleteSection { section_id, .. } => {
                    section_pre_image(&mut sections, *section_id, pre_state)
                }
                StorageMutation::SetChildren { section_id, .. }
                | StorageMutation::DeleteChildrenList { section_id } => {
                    let parent = *section_id;
                    let held = children_list_digest_held(parent, pre_state);
                    (held && lists.insert(parent)).then(|| {
                        PreImage::ChildrenList(children_list_digest(pre_state.list(parent)))
                    })
                }
                StorageMutation::DeleteField {
                    section_id,
                    key_name,
                } => fields
                    .insert((*section_id, key_name.as_str()))
                    .then(|| PreImage::RowExists(pre_state.field(*section_id, key_name).is_some())),
                StorageMutation::DeleteTimeSample { section_id, time } => {
                    let bits = time.into_inner().to_bits();
                    time_samples.insert((*section_id, bits)).then(|| {
                        // By the row's bits, as the stored key spells it:
                        // `0.0` and `-0.0` are two rows (`SampleRows`).
                        PreImage::RowExists(pre_state.sample(*section_id, bits).is_some())
                    })
                }
                StorageMutation::CreateTombstone { .. }
                | StorageMutation::SetField { .. }
                | StorageMutation::SetTimeSample { .. }
                | StorageMutation::SetObjectHeader { .. } => None,
            })
            .collect();

        Self {
            per_record,
            untouched_ext_counts: pass_timing::time(Pass::UntouchedExtCounts, || {
                untouched_ext_counts(mutations, pre_state)
            }),
            plan_fingerprint: plan_fingerprint(mutations),
        }
    }

    /// Whether these digests were built from `mutations` (see "Bound to its
    /// plan" on [`PreImageDigests`]).
    pub(crate) fn built_from(&self, mutations: &[StorageMutation]) -> bool {
        self.per_record.len() == mutations.len()
            && self.plan_fingerprint == plan_fingerprint(mutations)
    }

    /// The digest the `index`-th mutation of the plan carries, if any.
    pub(crate) fn for_record(&self, index: usize) -> Option<&PreImage> {
        self.per_record.get(index).and_then(Option::as_ref)
    }

    /// The untouched-row counts, strictly ascending by hash (see "Untouched-row
    /// counts" on [`PreImageDigests`]).
    pub(crate) fn untouched_ext_counts(&self) -> &[([u8; 32], u64)] {
        &self.untouched_ext_counts
    }
}

/// [`PreImageDigests`]'s untouched-row counts for `mutations` on `pre_state`
/// (see "Untouched-row counts" there): one entry per hash in `H`, zeros
/// included, strictly ascending by hash. Empty, without reading `pre_state`'s
/// rows, when the touched rows name no external hash. A plain state counts by
/// a walk of its rows; a plan made in place by its persistent counts, in
/// O(touched rows) ([`PreState::count_untouched`]).
fn untouched_ext_counts(
    mutations: &[StorageMutation],
    pre_state: &impl PreState,
) -> Vec<([u8; 32], u64)> {
    // The touched rows, by stored-key identity, each mapped to its final
    // external write (`None` when it ends inline or deleted). The first touch
    // also adds the hash the pre-plan state believes the row held.
    let mut field_final: HashMap<(u64, &str), Option<[u8; 32]>> = HashMap::new();
    let mut sample_final: HashMap<(u64, u64), Option<[u8; 32]>> = HashMap::new();
    let mut hashes: HashSet<[u8; 32]> = HashSet::new();

    for mutation in mutations {
        match mutation {
            StorageMutation::SetField { field } => {
                let row = (field.section_id, field.key_name.as_str());
                if !field_final.contains_key(&row) {
                    hashes.extend(believed_field_hash(pre_state, row));
                }
                field_final.insert(row, external_hash_of(&field.value));
            }
            StorageMutation::DeleteField {
                section_id,
                key_name,
            } => {
                let row = (*section_id, key_name.as_str());
                if !field_final.contains_key(&row) {
                    hashes.extend(believed_field_hash(pre_state, row));
                }
                field_final.insert(row, None);
            }
            StorageMutation::SetTimeSample { sample } => {
                let row = (sample.section_id, sample.time.into_inner().to_bits());
                if !sample_final.contains_key(&row) {
                    hashes.extend(believed_sample_hash(pre_state, row));
                }
                sample_final.insert(row, external_hash_of(&sample.value));
            }
            StorageMutation::DeleteTimeSample { section_id, time } => {
                let row = (*section_id, time.into_inner().to_bits());
                if !sample_final.contains_key(&row) {
                    hashes.extend(believed_sample_hash(pre_state, row));
                }
                sample_final.insert(row, None);
            }
            StorageMutation::SetSection { .. }
            | StorageMutation::DeleteSection { .. }
            | StorageMutation::CreateTombstone { .. }
            | StorageMutation::DeleteChildrenList { .. }
            | StorageMutation::SetChildren { .. }
            | StorageMutation::SetObjectHeader { .. } => {}
        }
    }
    hashes.extend(field_final.values().flatten());
    hashes.extend(sample_final.values().flatten());
    if hashes.is_empty() {
        return Vec::new();
    }

    let mut counts: HashMap<[u8; 32], u64> = hashes.into_iter().map(|h| (h, 0)).collect();
    pre_state.count_untouched(&mut counts, &field_final, &sample_final);

    let mut counts: Vec<([u8; 32], u64)> = counts.into_iter().collect();
    counts.sort_unstable_by_key(|(hash, _)| *hash);
    counts
}

/// The external hash `pre_state` believes field row `(section_id, key_name)`
/// held, if any.
fn believed_field_hash(pre_state: &impl PreState, row: (u64, &str)) -> Option<[u8; 32]> {
    pre_state.field(row.0, row.1).flatten()
}

/// The external hash `pre_state` believes time-sample row `(section_id, time
/// bits)` held, if any. `0.0` and `-0.0` are two different stored keys, so
/// the held sample counts only when its time bits match the row's
/// ([`SampleRows::ext_hash`]).
fn believed_sample_hash(pre_state: &impl PreState, row: (u64, u64)) -> Option<[u8; 32]> {
    pre_state.sample(row.0, row.1).flatten()
}

/// The touched rows' final external writes, by stored-key identity, as
/// [`untouched_ext_counts`] collects them: `None` when the row ends inline
/// or deleted.
pub type FieldFinals<'m> = HashMap<(u64, &'m str), Option<[u8; 32]>>;
/// As [`FieldFinals`], for time-sample rows by `(section_id, time bits)`.
pub type SampleFinals = HashMap<(u64, u64), Option<[u8; 32]>>;

/// What the structural commit reads of the state **before** its plan: the
/// pre-image digests ([`PreImageDigests::from_plan`]), the size gate
/// (`storage::structural_size_gate::StructuralWork::of_plan`) and the
/// untouched-row counts. Implemented by a plain state (`ObjectStateOf`), and
/// by the pre-plan view of a cache entry planned in place
/// (`domain::plan_journal::PreImageView`), which answers each key from the
/// plan's journal when the plan touched it and from the planned state
/// otherwise — so no copy of the pre-plan state is needed.
pub trait PreState {
    /// The section row `id` held.
    fn section(&self, id: u64) -> Option<&Section>;
    /// The children list `parent` held.
    fn list(&self, parent: u64) -> Option<&[u64]>;
    /// Field row `(section_id, name)`: `None` for no row, `Some(None)` for an
    /// inline row, `Some(Some(hash))` for an external one.
    fn field(&self, section_id: u64, name: &str) -> RowImage;
    /// Time-sample row `(section_id, time_bits)`, as [`Self::field`].
    fn sample(&self, section_id: u64, time_bits: u64) -> RowImage;
    /// The table lengths.
    fn lens(&self) -> PrePlanLens;
    /// Raise each `counts` entry by the rows holding its hash that are not
    /// touched: rows outside `field_final` and `sample_final`.
    fn count_untouched(
        &self,
        counts: &mut HashMap<[u8; 32], u64>,
        field_final: &FieldFinals<'_>,
        sample_final: &SampleFinals,
    );
}

impl<R: ValueRows> PreState for ObjectStateOf<R> {
    fn section(&self, id: u64) -> Option<&Section> {
        self.sections.get(&id)
    }

    fn list(&self, parent: u64) -> Option<&[u64]> {
        self.children.get(&parent).map(Vec::as_slice)
    }

    fn field(&self, section_id: u64, name: &str) -> RowImage {
        self.fields
            .contains(section_id, name)
            .then(|| self.fields.ext_hash(section_id, name))
    }

    fn sample(&self, section_id: u64, time_bits: u64) -> RowImage {
        self.time_samples
            .contains(section_id, time_bits)
            .then(|| self.time_samples.ext_hash(section_id, time_bits))
    }

    fn lens(&self) -> PrePlanLens {
        PrePlanLens {
            sections: self.sections.len() as u64,
            children: self.children.len() as u64,
            fields: FieldRows::len(&self.fields) as u64,
            time_samples: SampleRows::len(&self.time_samples) as u64,
        }
    }

    /// One walk over every field and time-sample row.
    fn count_untouched(
        &self,
        counts: &mut HashMap<[u8; 32], u64>,
        field_final: &FieldFinals<'_>,
        sample_final: &SampleFinals,
    ) {
        let mut count = |hash: Option<&[u8; 32]>| {
            if let Some(n) = hash.and_then(|h| counts.get_mut(h)) {
                *n += 1;
            }
        };
        self.fields.for_each(|section_id, key_name, hash| {
            if !field_final.contains_key(&(section_id, key_name)) {
                count(hash);
            }
        });
        // Each row by its time bits, as the stored key spells them (the full
        // form passes the held sample's own bits; its `OrderedFloat` map key
        // conflates `0.0` and `-0.0`).
        self.time_samples.for_each(|section_id, bits, hash| {
            if !sample_final.contains_key(&(section_id, bits)) {
                count(hash);
            }
        });
    }
}

impl PreState for PreImageView<'_> {
    fn section(&self, id: u64) -> Option<&Section> {
        PreImageView::section(self, id)
    }

    fn list(&self, parent: u64) -> Option<&[u64]> {
        PreImageView::list(self, parent)
    }

    fn field(&self, section_id: u64, name: &str) -> RowImage {
        PreImageView::field(self, section_id, name)
    }

    fn sample(&self, section_id: u64, time_bits: u64) -> RowImage {
        PreImageView::sample(self, section_id, time_bits)
    }

    fn lens(&self) -> PrePlanLens {
        PreImageView::lens(self)
    }

    /// In O(touched rows), from the planned entry's persistent counts:
    /// untouched rows are the same before and after the plan, so the rows
    /// untouched holding `h` are the planned state's rows holding `h` less
    /// the touched rows whose final write holds it.
    fn count_untouched(
        &self,
        counts: &mut HashMap<[u8; 32], u64>,
        field_final: &FieldFinals<'_>,
        sample_final: &SampleFinals,
    ) {
        let mut touched: HashMap<&[u8; 32], u64> = HashMap::new();
        for last in field_final.values().chain(sample_final.values()).flatten() {
            *touched.entry(last).or_default() += 1;
        }
        for (hash, count) in counts.iter_mut() {
            let touched = touched.get(hash).copied().unwrap_or(0);
            *count += self.post_ext_count(hash).saturating_sub(touched);
        }
    }
}

/// The section id that names the root as a parent: the root owns a children
/// list but has no section row (`keys::ROOT_PARENT_ID` is its key form).
const ROOT_SECTION_ID: u64 = 0;

/// Whether `parent`'s children list carries a pre-image digest, so the
/// commit script reads and hashes the stored list: the parent is the root or
/// a live section in `pre_state` (see [`PreImageDigests`]). The size gate
/// counts exactly these lists' pre-plan entries
/// (`storage::structural_size_gate::StructuralWork::of_plan`).
pub(crate) fn children_list_digest_held(parent: u64, pre_state: &impl PreState) -> bool {
    parent == ROOT_SECTION_ID || pre_state.section(parent).is_some()
}

/// The SHA-1 of a canonical encoding of `mutations`: the identity
/// [`PreImageDigests`] binds itself to.
///
/// Every field of every mutation is fed, each variant behind its own tag byte,
/// integers as 8 little-endian bytes, strings, byte vectors and lists behind
/// their length, and a time by its bits, so two plans that differ anywhere
/// encode differently. Computed in-process only and never stored, so the
/// encoding carries no compatibility promise. The `match` has no `_` arm: a
/// new variant must say what it feeds.
fn plan_fingerprint(mutations: &[StorageMutation]) -> [u8; 20] {
    use sha1::{Digest, Sha1};

    fn u64_(h: &mut Sha1, v: u64) {
        h.update(v.to_le_bytes());
    }
    fn bytes(h: &mut Sha1, b: &[u8]) {
        u64_(h, b.len() as u64);
        h.update(b);
    }
    fn value(h: &mut Sha1, v: &FieldValue) {
        match v {
            FieldValue::Inline(b) => {
                h.update([0]);
                bytes(h, b);
            }
            FieldValue::External {
                hash,
                size,
                ext_value_index,
            } => {
                h.update([1]);
                h.update(hash);
                u64_(h, *size);
                u64_(h, u64::from(*ext_value_index));
            }
        }
    }

    let mut h = Sha1::new();
    u64_(&mut h, mutations.len() as u64);
    for mutation in mutations {
        match mutation {
            StorageMutation::SetSection { section, previous } => {
                h.update([0]);
                u64_(&mut h, section.section_id);
                u64_(&mut h, section.parent_id);
                bytes(&mut h, section.name.as_bytes());
                u64_(&mut h, u64::from(section.section_type));
                u64_(&mut h, u64::from(section.section_order));
                u64_(&mut h, section.first_version);
                u64_(&mut h, section.last_version);
                match previous {
                    None => h.update([0]),
                    Some(key) => {
                        h.update([1]);
                        u64_(&mut h, key.parent_id);
                        bytes(&mut h, key.name.as_bytes());
                    }
                }
            }
            StorageMutation::DeleteSection {
                section_id,
                parent_id,
                name,
            } => {
                h.update([1]);
                u64_(&mut h, *section_id);
                u64_(&mut h, *parent_id);
                bytes(&mut h, name.as_bytes());
            }
            StorageMutation::CreateTombstone {
                section_id,
                first_version,
                last_version,
            } => {
                h.update([2]);
                u64_(&mut h, *section_id);
                u64_(&mut h, *first_version);
                u64_(&mut h, *last_version);
            }
            StorageMutation::DeleteChildrenList { section_id } => {
                h.update([3]);
                u64_(&mut h, *section_id);
            }
            StorageMutation::SetChildren {
                section_id,
                children,
            } => {
                h.update([4]);
                u64_(&mut h, *section_id);
                u64_(&mut h, children.len() as u64);
                for child in children {
                    u64_(&mut h, *child);
                }
            }
            StorageMutation::SetField { field } => {
                h.update([5]);
                u64_(&mut h, field.section_id);
                bytes(&mut h, field.key_name.as_bytes());
                value(&mut h, &field.value);
                u64_(&mut h, field.set_order);
                u64_(&mut h, field.authoring_set_order);
                u64_(&mut h, field.version);
            }
            StorageMutation::DeleteField {
                section_id,
                key_name,
            } => {
                h.update([6]);
                u64_(&mut h, *section_id);
                bytes(&mut h, key_name.as_bytes());
            }
            StorageMutation::SetTimeSample { sample } => {
                h.update([7]);
                u64_(&mut h, sample.section_id);
                u64_(&mut h, sample.time.into_inner().to_bits());
                value(&mut h, &sample.value);
                u64_(&mut h, sample.version);
            }
            StorageMutation::DeleteTimeSample { section_id, time } => {
                h.update([8]);
                u64_(&mut h, *section_id);
                u64_(&mut h, time.into_inner().to_bits());
            }
            StorageMutation::SetObjectHeader {
                object_id,
                sequence_num,
                first_version,
                source_format,
            } => {
                h.update([9]);
                u64_(&mut h, *object_id);
                u64_(&mut h, *sequence_num);
                u64_(&mut h, *first_version);
                match source_format {
                    None => h.update([0]),
                    Some(src_fmt) => {
                        h.update([1]);
                        bytes(&mut h, src_fmt.as_bytes());
                    }
                }
            }
        }
    }
    h.finalize().into()
}

/// A section row's pre-image digest, or `None` when `section_id`'s row
/// already carries one in this plan.
fn section_pre_image(
    seen: &mut HashSet<u64>,
    section_id: u64,
    pre_state: &impl PreState,
) -> Option<PreImage> {
    seen.insert(section_id).then(|| {
        PreImage::Section(
            pre_state
                .section(section_id)
                .map(|section| section_hash_fields(section).map(|(_, value)| value)),
        )
    })
}

/// What a structural commit's script call came to, when it is not an error
/// (SR-6.2, SR-7.6). On every outcome other than [`Self::Committed`], this
/// call wrote nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitOutcome {
    /// The commit landed at `seq` (= `expected_seq + 1`), stamped `server_ts`.
    /// Its owed external-value releases are already enqueued, so the outcome
    /// carries none.
    ///
    /// The successor cache entry's `commit_chain` — the value the script
    /// wrote to the header — is
    ///
    /// ```text
    /// commit_chain_fold(expected_commit_chain,
    ///                   update_stream_payload(object_id, seq, client_id, client_ts, server_ts))
    /// ```
    ///
    /// over the commit's [`StructuralCommit`] fields and this variant's
    /// `seq` / `server_ts`. The caller derives it from those two Rust
    /// definitions ([`commit_chain_fold`], [`update_stream_payload`]), never
    /// from its own copy of the payload format.
    ///
    /// [`commit_chain_fold`]: super::commit_chain::commit_chain_fold
    /// [`update_stream_payload`]: super::commit_chain::update_stream_payload
    Committed { seq: u64, server_ts: u64 },
    /// This call's own commit was already recorded: an earlier attempt of the
    /// call committed at `seq`, stamped `server_ts`, and its reply was lost.
    /// The call reports that commit (SR-6.2 guard 3, §7.5.1) — the
    /// own-call verdict of the commit-token record parser in `layer.rs`
    /// (`RecordedBy::ThisCall`), the one both commit paths reach.
    ///
    /// `record` is the object's `obj:wtok` record for the call's client,
    /// verbatim as the script read it (`<token>:<seq>:<server_ts>`, or the
    /// operation shape); `seq` and `server_ts` are its parsed fields, the ones
    /// [`CommitObserver::already_landed`] receives.
    AlreadyLanded {
        seq: u64,
        server_ts: u64,
        record: String,
    },
    /// With a client operation only: the record names an EARLIER call that
    /// carried the same `operation_id` (`recorded.by` is
    /// `RecordedBy::EarlierCall` or `RecordedBy::Conflict`, never `ThisCall`).
    /// This call wrote nothing. The caller answers it as both other commit
    /// paths do — `ALREADY_COMMITTED` with the recorded `seq` and `server_ts`
    /// for an earlier call with the same request, `ALREADY_EXISTS` for a
    /// different one — and hands back the call's pre-flight external-value
    /// increments (one
    /// `surplus_acquire` per distinct hash at
    /// `ReleaseEvent::Resend`). That release is the caller's, made once per
    /// call by `LayerStore::update_layer_with_summary`'s shared verdict
    /// handling; this function does not release, so a call answered here is
    /// never released twice.
    Recorded(RecordedCommit),
    /// With an unchained client operation only: the writer's `obj:wtok`
    /// record no longer holds the call's start record (another commit of the
    /// `client_id`, or a checkpoint restore, since the call started). This
    /// call wrote nothing; it is answered
    /// `Unavailable(OPERATION_OVERTAKEN_MESSAGE)` and never retried within
    /// the call. This function releases nothing for it: the caller's shared
    /// verdict handling hands back the call's pre-flight increments on its
    /// first attempt, or looks up
    /// its own landed commit's releases after a counted retry (§7.5.3).
    /// Answered before the superseded guard, `FENCED` and `STALE` (the
    /// canonical step-1b order).
    Overtaken,
    /// With a chained client operation only: the writer's `obj:wtok` record
    /// does not name the call's predecessor (for a chain's first write: a
    /// record exists). `record` is that record as the script read it, `None`
    /// when absent. This call wrote nothing; it is answered
    /// `PREDECESSOR_NOT_COMMITTED` on the call's first attempt (a plain
    /// `UNAVAILABLE`, outcome unknown, after a counted retry) and never
    /// retried within the call. This function releases nothing for it; the
    /// caller's shared verdict handling does, as for [`Self::Overtaken`]
    /// (§7.5.3, §7.5.7). Answered after the superseded guard and before
    /// `FENCED` and `STALE` (the canonical step-1b order).
    PredecessorPending { record: Option<String> },
    /// The fence missed: the header's `seq` or its `commit_chain` no longer
    /// matches the cut the plan was made at; `seq` is the header's current
    /// `seq`. The caller maps it to `Aborted("structural commit fence missed:
    /// seq moved")`, repairs the entry from the stream and re-plans (SR-6.4).
    Stale { seq: u64 },
    /// A pre-image digest mismatched: the cache diverged from the store. The
    /// caller maps it to `Aborted("structural cache diverged from store")`,
    /// discards the entry and reloads (SR-6.4, SR-3.4).
    CacheDiverged,
    /// A live snapshot fence on the object (the script's step 1d, the
    /// `{"FENCED", "<ms>"}` reply both modes give, decoded by
    /// `layer::decode_fenced_reply`): nothing was written, and the caller waits
    /// about `retry_after_ms` without spending a retry attempt (SR-8.1,
    /// b.y3m SR-5.4). The store's test seam
    /// (`LayerStore::with_structural_fenced_replies_for_test`) answers the same
    /// reply in place of the `EVALSHA`.
    Fenced { retry_after_ms: u64 },
}

/// The caller's view of one [`commit_structural`] call's commit size and
/// landing, for exactly-once commit counting (SR-2.6).
///
/// **Why it exists.** SR-2.6 requires every landed commit to be counted
/// exactly once: its size recorded just before it is sent, and its landing
/// counted synchronously at the landing point, before the next `.await`, so
/// that a call whose future is then dropped during post-commit work — the
/// external-value release enqueue, on its deadline or a cancel — has still
/// counted the commit it landed. Only [`commit_structural`] knows the
/// encoded commit's records and bytes, and
/// only it sees the reply decode before that post-commit work, so the caller
/// hands it this observer rather than counting after it returns.
///
/// **Calls.** Every method is synchronous and runs inline in
/// [`commit_structural`], never across an `.await`:
///
/// - [`commit_size`](Self::commit_size) once, just before the `EVALSHA`;
/// - [`evaluated`](Self::evaluated) once, when the `EVALSHA` returns,
///   whatever it returned;
/// - then at most one of [`landed`](Self::landed) (a
///   [`CommitOutcome::Committed`] reply) or
///   [`already_landed`](Self::already_landed) (a
///   [`CommitOutcome::AlreadyLanded`] reply), right after the reply decodes,
///   with no `.await` between the decode and the call;
/// - then [`commit_structural`] itself awaits the store's landed-commit test
///   seam (`LayerStore::after_commit_landed`), and only after that, on a
///   commit, the external-value release enqueue.
///
/// No other outcome and no error calls `landed` or `already_landed`.
///
/// **The `update_layer` adapter's contract.** The implementation an
/// `update_layer` attempt passes is `layer.rs`'s `UpdateLayerCallObserver`,
/// over the call's running record (`UpdateLayerCall`), the store, and the
/// call's session, layer and object ids, which it captures for the commit's
/// log line. It calls the same helpers the fast route calls at its landing
/// point, so a structural commit is counted and logged as a fast one is:
///
/// - **Who sizes:** `commit_size` records the size on the call
///   (`UpdateLayerCall::record_commit_size`): the records excluding `HDR`,
///   and `lua::script_call_bytes` of the `KEYS` + `ARGV` as sent.
/// - **Who times:** `evaluated` observes the `EVALSHA`'s round trip on
///   `live_edit_update_layer_structural_commit_seconds` and logs a WARN, with
///   the plan's work and the layer's size, when it ran past
///   `storage::structural_size_gate::STRUCTURAL_COMMIT_SLOW_WARN`.
/// - **Who counts:** the adapter. `landed` counts the commit through
///   `LayerStore::observe_commit_landed` (one attempts-histogram sample, the
///   attempt's retries on `retries_total{outcome="committed"}`, and one
///   `live_edit_update_layer_relay_bytes` sample);
///   `already_landed` counts it through
///   `LayerStore::report_commit_already_landed` with `backend = "lua"`,
///   which also ticks `commit_already_landed_total{backend="lua"}`.
/// - **Who logs:** the adapter, and only the adapter. `landed` logs the
///   commit's one line through `log_landed_commit!` (the call summary's
///   `attempt`, `path`, five conflict counts, `records` and `bytes`, then the
///   session, layer and object ids and `sequence_num`);
///   `report_commit_already_landed` logs the already-landed route's one line.
///   [`commit_structural`] logs no commit line of its own.
/// - **Who awaits the landed-commit hook:** [`commit_structural`], whatever
///   the observer, right after `landed` / `already_landed` and before the
///   release enqueue — the hook point the fast route awaits it at. The
///   adapter's methods are synchronous and never await it.
///
/// **Shape.** The methods take `&mut self`, so an implementation can record
/// into plain fields without interior mutability. The trait requires
/// [`Send`] because [`commit_structural`] holds its `&mut dyn CommitObserver`
/// across its `.await`s, and the future it returns must stay `Send` to run in
/// a gRPC handler; it does not require `Sync`, since a `&mut` reference is
/// never shared.
pub trait CommitObserver: Send {
    /// The commit about to be sent: `records` structural records, `bytes`
    /// encoded bytes. Called once, just before the `EVALSHA`.
    fn commit_size(&mut self, records: u64, bytes: u64);

    /// The commit's `EVALSHA` returned, a reply or an error, `elapsed` after
    /// it was written: the client-observed round trip of the one script call,
    /// which holds the object's shard for its server-side part. Called once
    /// per `EVALSHA` sent, right after it returns and before its reply is
    /// decoded; not called when nothing was sent (an encode refusal, or a
    /// test seam answering in place of the call). The default does nothing.
    fn evaluated(&mut self, _elapsed: std::time::Duration) {}

    /// The commit landed at `seq`, stamped `server_ts`. Called synchronously
    /// right after the reply decodes, before any further `.await` (the
    /// landed-commit hook, then the release enqueue).
    fn landed(&mut self, seq: u64, server_ts: u64);

    /// This call's commit token was already recorded: an earlier commit of
    /// the call landed at `seq`, stamped `server_ts`. Called synchronously
    /// right after the reply decodes, before any further `.await` (the
    /// landed-commit hook).
    fn already_landed(&mut self, seq: u64, server_ts: u64);
}

/// Run one structural commit against `store` (SR-6), reporting its size, its
/// `EVALSHA`'s round trip and its landing to `observer` (see [`CommitObserver`] for when each method runs).
///
/// The commit is one `EVALSHA` of `update_layer_fast.lua` in structural mode
/// on the data pool (SR-6.1, SR-6.6): no `txn_pool` slot, no `WATCH` /
/// `MULTI`, no witness. Like the fast mode's commit, the `EVALSHA` is written
/// to the wire at most once (`lua::evalsha_once_with_reload`, fred's `custom`
/// command path with `max_attempts` 1): a dropped connection fails the call
/// transiently, and its next attempt meets a commit that did land at the
/// commit-token check rather than a replay. This function:
///
/// - encodes the plan with `lua::encode_mutations`, behind the shared
///   `KEYS` / `ARGV` prefix both script modes take
///   (`lua::UpdateScriptPrefix`), the mode slot, `expected_seq` /
///   `expected_commit_chain`, the untouched-row count block, and the
///   sections, deleted-sections and children-list indexes (the script
///   header's structural-mode layout);
/// - computes the server timestamp itself;
/// - bounds the stream with the store's own `layer_stream_maxlen` (the entry
///   cap) and relay byte budget (`LayerStore::relay_retention`, which the
///   script folds into the header's `relay_gens` and trims the stream head
///   by, exactly as in fast mode);
/// - after a commit, enqueues the owed external-value releases the script
///   returns (as the fast path does, via `enqueue_ext_releases_from_tokens`),
///   so [`CommitOutcome::Committed`] carries none;
/// - answers `STALE`, `CACHE_DIVERGED` and `FENCED` (SR-6.4, SR-8.1) and the
///   step-1b replies — `ALREADY_LANDED` (this call's own commit, or with a
///   client operation an earlier call's: [`CommitOutcome::AlreadyLanded`] /
///   [`CommitOutcome::Recorded`]), `OVERTAKEN` and the writer guard's
///   `PREDECESSOR` ([`CommitOutcome::PredecessorPending`]) — as
///   [`CommitOutcome`]s, decoding the latter with the same function, and so
///   reaching the same verdict, as the fast path; and maps every other script
///   error tag
///   through the one mapping the fast path's `update_layer_once_lua` uses
///   (`LayerStore::map_update_script_error`): the same `ServiceError`
///   variant, prefix and sentinel, the same
///   `keyspace_type_error{backend="lua", site="update"}` cell (under
///   `mode="structural"`, where the fast path's refusals carry
///   `mode="fast"`), and
///   `STREAM_ID_COLLISION` disambiguated against a concurrent retirement
///   with [`StructuralCommit::session_id`] / [`StructuralCommit::layer_name`]
///   (SR-6.5). A plan whose digests were not built from it is refused
///   `Internal` before anything is sent.
///
/// After [`CommitOutcome::Committed`], the successor entry's `commit_chain`
/// is derived as that variant's documentation states.
///
/// It logs no commit line: a landed commit's one line is `observer`'s (see
/// [`CommitObserver`]'s contract).
///
/// # Precondition: external values are already stored
///
/// The script's settle owes back the caller's pre-flight `+1` for a hash the
/// object already held (`surplus_acquire`, as on the fast path), which is
/// sound only if that `+1` was paid: every external hash the plan references
/// must already be stored and counted (`external_values::store_external_values`,
/// which pays the `+1` through `ExternalValueStore::store_values`; the WATCH
/// path's `SurplusPolicy::Release` premise) before this call. `update_layer`'s
/// handler-side pre-store (`grpc::layer`) satisfies it for every route, and a
/// test that commits external values must call it first.
///
/// # Test seams
///
/// Under `test` / `test-fault-injection`, this consumes the store's two Lua
/// fault queues at the same points `update_layer_once_lua` does — the
/// pre-`EVALSHA` queue in place of the call, the post-`EVALSHA` queue in
/// place of its reply (the lost-reply shape) — and, between the two, the
/// store's queue of scripted `{"FENCED", "<retry_after_ms>"}` replies
/// (`LayerStore::with_structural_fenced_replies_for_test`), each answered in
/// place of the `EVALSHA` and decoded as a live fence's refusal. When the
/// `EVALSHA` is sent, it first awaits the store's structural pre-dispatch hook
/// (`LayerStore::with_structural_commit_pre_dispatch_hook_for_test`), if
/// installed, with the client the `EVALSHA` is then written to: after the
/// plan was made and sized, with nothing landed, so a test can change the
/// store between a structural attempt's plan and its commit. It awaits the
/// store's
/// landed-commit hook (`LayerStore::with_commit_landed_hook_for_test`), if
/// installed, right after `landed` or `already_landed`, and then, on a
/// commit, calls the store's release-enqueue hook, if installed, just before
/// the enqueue. The await of the landed-commit hook
/// (`LayerStore::after_commit_landed`) is in every build; without the seam
/// it is a no-op future.
pub async fn commit_structural(
    store: &LayerStore,
    commit: &StructuralCommit,
    observer: &mut dyn CommitObserver,
) -> Result<CommitOutcome, ServiceError> {
    let keys = store.keys();
    let oid_str = commit.object_id.to_string();
    let encoded = encode_mutations(
        &commit.mutations,
        &commit.pre_image_digests,
        commit.next_set_order,
        keys,
        &oid_str,
    )?;
    let server_ts = now_micros();
    // The call's identity on the commit-token record: its token and, when it
    // carries one, its client operation — the match key (ARGV[8]), record
    // suffix (ARGV[9]) and writer guard (ARGV[10]) the fast mode sends too.
    let identity =
        CommitIdentity::from_recorded_token(&commit.commit_token, commit.operation.as_ref());

    // KEYS = [<shared prefix>, secs, dels, chs, <one key per record>]
    // ARGV = [<shared prefix>, "structural", expected_seq,
    //         expected_commit_chain, <untouched-row count block>, <records>]
    // — the structural-mode layout in the script's contract. The block leads
    // the records so the records still run to the end of ARGV.
    let (mut script_keys, mut argv) = UpdateScriptPrefix {
        object_id: commit.object_id,
        server_ts,
        client_id: &commit.client_id,
        client_ts: commit.client_ts,
        stream_maxlen: store.layer_stream_maxlen(),
        relay_retention: store.relay_retention(),
        ring_ttl_ms: store.commit_release_ring_ttl_ms(),
        num_mutations: encoded.num_mutations,
        relay_ompf: commit.relay_ompf.clone(),
        identity: &identity,
    }
    .build(
        keys,
        UpdateScriptMode::Structural,
        3 + encoded.mutation_keys.len(),
        2 + encoded.untouched_ext_block.len() + encoded.mutation_argv.len(),
    );
    script_keys.push(keys.obj_sections_index(&oid_str));
    script_keys.push(keys.obj_deleted_sections_index(&oid_str));
    script_keys.push(keys.obj_children_index(&oid_str));
    script_keys.extend(encoded.mutation_keys);
    argv.push(commit.expected_seq.to_string().into());
    argv.push(commit.expected_commit_chain.clone().into());
    argv.extend(encoded.untouched_ext_block.into_iter().map(Into::into));
    argv.extend(encoded.mutation_argv.into_iter().map(Into::into));

    // The commit's size, just before it is sent: its records without the
    // header bump, and its KEYS + ARGV bytes.
    observer.commit_size(
        encoded.num_size_records,
        script_call_bytes(&script_keys, &argv),
    );

    #[cfg(any(test, feature = "test-fault-injection"))]
    if let Some(injected) = store.take_lua_pre_eval_fault_for_test() {
        return Err(injected);
    }
    // Test-only: a scripted `{"FENCED", "<retry_after_ms>"}` reply, answered
    // in place of the `EVALSHA` below — the script never runs, so nothing is
    // written — and decoded by `layer::decode_fenced_reply` exactly as a live
    // snapshot fence's refusal is. See
    // `LayerStore::with_structural_fenced_replies_for_test`. Always `None` in
    // a build without the seam.
    #[cfg(any(test, feature = "test-fault-injection"))]
    let scripted_fence = store.take_structural_fenced_reply_for_test();
    #[cfg(not(any(test, feature = "test-fault-injection")))]
    let scripted_fence: Option<u64> = None;
    let eval_result = match scripted_fence {
        Some(retry_after_ms) => Ok(vec!["FENCED".to_owned(), retry_after_ms.to_string()]),
        // Written to the wire at most once (`lua::evalsha_once_with_reload`),
        // as the fast mode's commit is: a fred replay after a reconnect could
        // run after this client's next commit. A dropped connection fails the
        // attempt transiently, and the call's next attempt finds a commit that
        // did land through the commit-token check (the in-doubt reconcile),
        // never a replay.
        None => {
            let eval_client = store.client().pool.next();
            // Test-only: the plan is sized and nothing has landed; the
            // `EVALSHA` is about to be written to `eval_client`. See
            // `LayerStore::with_structural_commit_pre_dispatch_hook_for_test`.
            #[cfg(any(test, feature = "test-fault-injection"))]
            if let Some(hook) = store.structural_commit_pre_dispatch_hook_for_test() {
                hook(eval_client).await;
            }
            let sent = std::time::Instant::now();
            let result = evalsha_once_with_reload::<Vec<String>>(
                eval_client,
                &UPDATE_LAYER_FAST_SCRIPT,
                script_keys,
                argv,
            )
            .await;
            observer.evaluated(sent.elapsed());
            result
        }
    };
    // Test-only: the script has run (no scripted fence answered for it), and
    // its reply is replaced by a queued lost-reply error.
    #[cfg(any(test, feature = "test-fault-injection"))]
    let eval_result = match scripted_fence {
        Some(_) => eval_result,
        None => match store.take_lua_post_eval_fault_for_test() {
            Some(lost) => Err(lost),
            None => eval_result,
        },
    };

    match eval_result {
        Ok(reply) => {
            // The script's step 1d: a live snapshot fence refused the commit
            // and nothing was written; the caller waits it out, uncounted.
            if let Some(fenced) = decode_fenced_reply(&reply) {
                return fenced.map(|retry_after_ms| CommitOutcome::Fenced { retry_after_ms });
            }
            // From the decode to the observer's landing call nothing awaits,
            // so the landing is counted and logged at the landing point, and a
            // call dropped at any later `.await` has already counted it
            // (SR-2.6). The landed-commit test seam is the first await after
            // either landing call, ahead of the release enqueue.
            if let Some(answer) = decode_commit_token_reply(&reply, &identity) {
                return match answer? {
                    CommitTokenReply::Landed(LandedRecord { recorded, record })
                        if recorded.by == RecordedBy::ThisCall =>
                    {
                        observer.already_landed(recorded.seq, recorded.server_ts);
                        store.after_commit_landed().await;
                        Ok(CommitOutcome::AlreadyLanded {
                            seq: recorded.seq,
                            server_ts: recorded.server_ts,
                            record,
                        })
                    }
                    // An earlier call carrying the same operation id: the
                    // caller answers it and owes the release (see the variant).
                    CommitTokenReply::Landed(LandedRecord { recorded, .. }) => {
                        Ok(CommitOutcome::Recorded(recorded))
                    }
                    CommitTokenReply::Overtaken => Ok(CommitOutcome::Overtaken),
                    CommitTokenReply::PredecessorPending { record } => {
                        Ok(CommitOutcome::PredecessorPending { record })
                    }
                };
            }
            let (seq, tokens) = decode_committed_reply(&reply)?;
            // The commit has landed: nothing past this point may turn it into
            // an error. A malformed drift element costs only its count.
            let (settle_drift, release_tokens) = decode_settle_drift(tokens);
            // The observer counts and logs the commit; this function logs no
            // commit line of its own, so each commit logs exactly once.
            observer.landed(seq, server_ts);
            match settle_drift {
                Some(drift) => store.report_structural_settle_drift(
                    commit.object_id,
                    seq,
                    drift.under,
                    drift.over,
                ),
                None => tracing::error!(
                    object_id = commit.object_id,
                    sequence_num = seq,
                    reply = ?tokens,
                    "structural commit landed with no well-formed \"drift:<under>:<over>\" reply \
                     element; its settle drift is not counted"
                ),
            }
            store.after_commit_landed().await;
            #[cfg(any(test, feature = "test-fault-injection"))]
            if let Some(hook) = store.structural_release_enqueue_hook_for_test() {
                hook(release_tokens);
            }
            // The owed decrements go to the GC pending set — a separate
            // Cluster slot, so they could not ride the script's atomic unit.
            store
                .enqueue_ext_releases_from_tokens(commit.object_id, seq, release_tokens)
                .await;
            Ok(CommitOutcome::Committed { seq, server_ts })
        }
        Err(e) => {
            if let Some(outcome) = decode_structural_refusal(commit, e.details()) {
                return outcome;
            }
            Err(store
                .map_update_script_error(
                    &commit.session_id,
                    &commit.layer_name,
                    commit.object_id,
                    UpdateScriptMode::Structural,
                    e,
                )
                .await)
        }
    }
}

/// Structural mode's settle drift counts, reply element 2
/// (`"drift:<under>:<over>"`; `update_layer_fast.lua` step 6b / 8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SettleDrift {
    /// Touched hashes whose stored count plus delta fell below the recount.
    under: u64,
    /// Touched hashes whose stored count plus delta stood above it.
    over: u64,
}

/// Split a committed structural reply's tail (everything after the sequence
/// number) into its settle drift element and the release tokens after it.
///
/// The script in this repo always writes the element in structural mode, so
/// a missing or malformed one is a script/caller mismatch. The commit has
/// already landed by then, so this never fails: it answers `None` for the
/// drift, and takes as release tokens everything after a `drift:`-tagged
/// element, or the whole tail when there is none (each token is still
/// checked against the release vocabulary when it is enqueued).
fn decode_settle_drift(tokens: &[String]) -> (Option<SettleDrift>, &[String]) {
    let Some((head, rest)) = tokens.split_first() else {
        return (None, tokens);
    };
    let Some(counts) = head.strip_prefix("drift:") else {
        return (None, tokens);
    };
    let drift = counts.split_once(':').and_then(|(under, over)| {
        Some(SettleDrift {
            under: under.parse().ok()?,
            over: over.parse().ok()?,
        })
    });
    (drift, rest)
}

/// Decode structural mode's own error-reply refusals — `STALE <seq>` and
/// `CACHE_DIVERGED: …` — into their [`CommitOutcome`]s (SR-6.4). `None` for
/// any other error reply, which the caller maps as the fast path does. A
/// `STALE` reply whose number does not parse is `Internal`: the script in this
/// repo always writes a decimal there. (A live snapshot fence is not an error
/// reply; see `layer::decode_fenced_reply`.)
fn decode_structural_refusal(
    commit: &StructuralCommit,
    details: &str,
) -> Option<Result<CommitOutcome, ServiceError>> {
    if let Some(seq) = details
        .strip_prefix(STALE_TAG)
        .and_then(|r| r.strip_prefix(' '))
    {
        return Some(
            seq.parse::<u64>()
                .map(|seq| CommitOutcome::Stale { seq })
                .map_err(|_| unparseable_refusal(details)),
        );
    }
    // The script writes `CACHE_DIVERGED: <record and key>`; the tag counts
    // only with its `:` separator, as `STALE` counts only with its space, so a
    // lookalike tag goes on to the shared mapping.
    if details
        .strip_prefix(CACHE_DIVERGED_TAG)
        .is_some_and(|r| r.starts_with(':'))
    {
        // The outcome carries no detail; the script's message — which record,
        // which key, what differed — is the only place it exists, so log it.
        tracing::warn!(
            session_id = %commit.session_id,
            layer_name = %commit.layer_name,
            object_id = commit.object_id,
            expected_seq = commit.expected_seq,
            detail = %details,
            "structural commit refused: a key the plan writes does not hold what the cache \
             believed; nothing was written"
        );
        return Some(Ok(CommitOutcome::CacheDiverged));
    }
    None
}

/// [`decode_structural_refusal`]'s refusal for a `STALE` reply without its
/// decimal.
fn unparseable_refusal(details: &str) -> ServiceError {
    ServiceError::Internal(format!(
        "update_layer structural commit: unparseable refusal reply {details:?}"
    ))
}

#[cfg(test)]
mod tests {
    //! [`PreImageDigests::from_plan`]'s digest rules (SR-6.3; the type's
    //! docs): which digest each record of a plan carries, taken against the
    //! pre-plan state. How the encoder renders a digest into its record is
    //! pinned beside the encoder, in `lua.rs`'s `encode_mutations_tests`.
    //!
    //! The tests listed below build their plans by hand, because the rule
    //! each pins turns on a mutation sequence the engine does not emit; the
    //! other section-row and children-list tests take their plans from
    //! `apply_delta` on [`seeded_state`]. The hand-built ones:
    //!
    //! - the field and time-sample row tests (every test of a
    //!   `DeleteField` / `DeleteTimeSample` but rule E's): the engine emits
    //!   those deletes only inside a reset or a cascade, for rows the state
    //!   holds plus, for a section whose time 0 it deletes at one spelling,
    //!   the other spelling (rule E, the one row the pre-state lacks that it
    //!   deletes), so it never deletes another row the pre-state lacks, the
    //!   same row twice, or a row a `SetField` / `SetTimeSample` of the same
    //!   plan wrote first. Rule E's tests take their plans from the engine —
    //!   a reset's and a cascade's, on a plain state and planned in place —
    //!   because the record they pin is the engine's own;
    //! - the list-written-twice test: the engine writes each parent's list
    //!   once per delta;
    //! - the dangling-parent test: a normalized entry holds no list under a
    //!   parent with no row;
    //! - the no-digest-kinds test: it wants a plan made only of no-digest
    //!   records, and the engine never emits a tombstone without the
    //!   `DeleteSection` of the same section, whose id's first section record
    //!   always carries a digest.
    //!
    //! Then the untouched-row counts the value carries beside the digests
    //! (hand-built plans too, for the same reasons): which rows count as
    //! touched (by stored-key identity, so `-0.0` and `0.0` are two rows; a
    //! row touched twice once), which hashes get an entry (a touched row's
    //! believed original, its final write, not an intermediate one), zero
    //! entries kept, and no entries for a plan naming no external hash — that
    //! last test is mixed: an engine plan's structural records with hand-built
    //! field records appended. How the encoder renders them is pinned in
    //! `lua.rs`'s `encode_mutations_tests`.
    //!
    //! At the end, [`decode_structural_refusal`]'s reply decoding: each of
    //! its three refusals, including the `FENCED` arm, which no live reply
    //! reaches until the snapshot fence is wired; the `STALE` / `FENCED`
    //! replies without their decimal, which the script in this repo never
    //! writes; and the replies, lookalike prefixes included, it leaves to the
    //! shared error mapping.

    use oodf_ffi::{
        CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection, CommandMoveSection,
        CommandReorderChildren, OwnedDelta, StructCommand,
    };
    use ordered_float::OrderedFloat;

    use super::*;
    use crate::domain::delta::apply_delta;
    use crate::domain::types::{compact, CompactState, Field, ObjectState, Section, TimeSample};
    use crate::storage::keys::KeySchema;
    use crate::storage::lua::DIGEST_ABSENT;

    /// Section `A`: under the root, every stored column distinct so a
    /// pre-image that read the wrong column, or the post-plan row, shows.
    const A: u64 = 1;
    /// Section `B`: under `A`, listed in `A`'s list; no children list.
    const B: u64 = 2;
    /// Section `C`: under `A`, listed in `A`'s list; an empty children list.
    const C: u64 = 3;
    /// The root, as a parent id.
    const ROOT: u64 = 0;
    /// A field `B` holds.
    const KEY: &str = "k";
    /// A time `B` holds a sample at.
    const TIME: f64 = 1.5;

    /// `A`'s six stored columns, in `section_hash_fields` order and string
    /// form: `parent_id, name, type, order, first_ver, last_ver`.
    const A_COLUMNS: [&str; 6] = ["0", "A", "1", "2", "3", "4"];
    const B_COLUMNS: [&str; 6] = ["1", "B", "0", "0", "1", "5"];
    const C_COLUMNS: [&str; 6] = ["1", "C", "0", "1", "2", "6"];

    fn section(
        section_id: u64,
        parent_id: u64,
        name: &str,
        section_type: u8,
        section_order: u32,
        first_version: u64,
        last_version: u64,
    ) -> Section {
        Section {
            section_id,
            parent_id,
            name: name.to_owned(),
            section_type,
            section_order,
            first_version,
            last_version,
        }
    }

    fn field(section_id: u64, key_name: &str) -> Field {
        Field {
            section_id,
            key_name: key_name.to_owned(),
            value: FieldValue::Inline(vec![1]),
            set_order: 1,
            authoring_set_order: 1,
            version: 5,
        }
    }

    fn time_sample(section_id: u64, time: f64) -> TimeSample {
        TimeSample {
            section_id,
            time: OrderedFloat(time),
            value: FieldValue::Inline(vec![2]),
            version: 5,
        }
    }

    /// `A` under the root holding `B` and `C`; `B` holds field [`KEY`] and
    /// samples at `0.0` and [`TIME`]; `C` holds an empty children list (the
    /// shape a normalized entry need not hold, but may).
    fn seeded_state() -> ObjectState {
        let mut state = ObjectState {
            object_id: 42,
            sequence_num: 5,
            first_version: 1,
            next_set_order: 2,
            ..Default::default()
        };
        for s in [
            section(A, ROOT, "A", 1, 2, 3, 4),
            section(B, A, "B", 0, 0, 1, 5),
            section(C, A, "C", 0, 1, 2, 6),
        ] {
            state.sections.insert(s.section_id, s);
        }
        state.children.insert(ROOT, vec![A]);
        state.children.insert(A, vec![B, C]);
        state.children.insert(C, Vec::new());
        state.fields.insert((B, KEY.to_owned()), field(B, KEY));
        for time in [0.0, TIME] {
            state
                .time_samples
                .insert((B, OrderedFloat(time)), time_sample(B, time));
        }
        state
    }

    /// The plan `apply_delta` makes for a non-reset delta of `cmds` on `pre`.
    fn plan(pre: &ObjectState, cmds: Vec<StructCommand>) -> Vec<StorageMutation> {
        let delta = OwnedDelta {
            is_diff: false,
            struct_commands: cmds,
            ..Default::default()
        };
        apply_delta(&mut pre.clone(), Some(delta)).expect("the fixture delta must apply")
    }

    /// Every record's digest, in plan order.
    fn digests(mutations: &[StorageMutation], pre: &ObjectState) -> Vec<Option<PreImage>> {
        let d = PreImageDigests::from_plan(mutations, pre);
        (0..mutations.len())
            .map(|i| d.for_record(i).cloned())
            .collect()
    }

    /// The digests of the records `pick` selects, in plan order; panics when
    /// it selects none, so a plan that lost the shape fails loudly.
    fn digests_where(
        mutations: &[StorageMutation],
        pre: &ObjectState,
        pick: impl Fn(&StorageMutation) -> bool,
    ) -> Vec<Option<PreImage>> {
        let all = digests(mutations, pre);
        let picked: Vec<_> = mutations
            .iter()
            .zip(all)
            .filter(|(m, _)| pick(m))
            .map(|(_, d)| d)
            .collect();
        assert!(
            !picked.is_empty(),
            "fixture precondition: the plan must carry the record under test; got {mutations:?}"
        );
        picked
    }

    fn is_set_section(id: u64) -> impl Fn(&StorageMutation) -> bool {
        move |m| matches!(m, StorageMutation::SetSection { section, .. } if section.section_id == id)
    }

    fn is_delete_section(id: u64) -> impl Fn(&StorageMutation) -> bool {
        move |m| matches!(m, StorageMutation::DeleteSection { section_id, .. } if *section_id == id)
    }

    fn is_set_children(parent: u64) -> impl Fn(&StorageMutation) -> bool {
        move |m| matches!(m, StorageMutation::SetChildren { section_id, .. } if *section_id == parent)
    }

    fn row(columns: [&str; 6]) -> Option<PreImage> {
        Some(PreImage::Section(Some(columns.map(str::to_owned))))
    }

    fn list(children: &[u64]) -> Option<PreImage> {
        Some(PreImage::ChildrenList(children_list_digest(Some(children))))
    }

    fn absent_list() -> Option<PreImage> {
        Some(PreImage::ChildrenList(DIGEST_ABSENT.to_owned()))
    }

    fn create(parent_id: u64, section_id: u64, name: &str) -> StructCommand {
        StructCommand::CreateSection(CommandCreateSection {
            parent_id,
            section_id,
            section_name: name.to_owned(),
            section_type: 1,
        })
    }

    fn delete(section_id: u64) -> StructCommand {
        StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: 0,
            section_id,
        })
    }

    fn rename(section_id: u64, parent_id: u64, new_name: &str) -> StructCommand {
        StructCommand::MoveSection(CommandMoveSection {
            old_parent_id: parent_id,
            new_parent_id: parent_id,
            section_id,
            new_name: new_name.to_owned(),
        })
    }

    fn set_field(section_id: u64, key_name: &str) -> StorageMutation {
        StorageMutation::SetField {
            field: field(section_id, key_name),
        }
    }

    fn delete_field(section_id: u64, key_name: &str) -> StorageMutation {
        StorageMutation::DeleteField {
            section_id,
            key_name: key_name.to_owned(),
        }
    }

    fn delete_time_sample(section_id: u64, time: f64) -> StorageMutation {
        StorageMutation::DeleteTimeSample {
            section_id,
            time: OrderedFloat(time),
        }
    }

    // ── Section rows ─────────────────────────────────────────────────────────

    #[test]
    fn from_plan_set_section_of_a_present_section_carries_its_pre_state_columns() {
        let pre = seeded_state();
        // Rename and reorder `A`: the planned row differs from the stored one
        // in two columns, so a digest of the post-plan row would not match.
        let mutations = plan(
            &pre,
            vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: ROOT,
                section_id: A,
                section_name: "A2".to_owned(),
                section_type: 1,
                section_order: 9,
            })],
        );

        assert_eq!(
            digests_where(&mutations, &pre, is_set_section(A)),
            vec![row(A_COLUMNS)]
        );
    }

    #[test]
    fn from_plan_set_section_of_a_new_section_carries_absent() {
        let pre = seeded_state();
        let mutations = plan(&pre, vec![create(ROOT, 10, "N")]);

        assert_eq!(
            digests_where(&mutations, &pre, is_set_section(10)),
            vec![Some(PreImage::Section(None))]
        );
    }

    #[test]
    fn from_plan_delete_section_cascade_carries_each_rows_pre_state_columns() {
        let pre = seeded_state();
        let mutations = plan(&pre, vec![delete(A)]);

        let got: Vec<_> = [A, B, C]
            .map(|id| digests_where(&mutations, &pre, is_delete_section(id)))
            .into_iter()
            .collect();

        assert_eq!(
            got,
            vec![
                vec![row(A_COLUMNS)],
                vec![row(B_COLUMNS)],
                vec![row(C_COLUMNS)]
            ]
        );
    }

    // ── Children lists ───────────────────────────────────────────────────────

    #[test]
    fn from_plan_children_list_of_a_live_section_carries_the_pre_state_list_digest() {
        let pre = seeded_state();
        let mutations = plan(
            &pre,
            vec![StructCommand::ReorderChildren(CommandReorderChildren {
                section_id: A,
                children_list_id: 0,
                children_list: vec![C, B],
            })],
        );
        assert!(
            mutations.contains(&StorageMutation::SetChildren {
                section_id: A,
                children: vec![C, B],
            }),
            "fixture precondition: the plan must rewrite A's list to a different order; \
             got {mutations:?}"
        );

        assert_eq!(
            digests_where(&mutations, &pre, is_set_children(A)),
            vec![list(&[B, C])]
        );
    }

    #[test]
    fn from_plan_children_list_of_the_root_carries_its_list_digest() {
        let pre = seeded_state();
        let mutations = plan(&pre, vec![create(ROOT, 10, "N")]);

        assert_eq!(
            digests_where(&mutations, &pre, is_set_children(ROOT)),
            vec![list(&[A])]
        );
    }

    #[test]
    fn from_plan_missing_children_list_of_a_live_section_carries_the_absent_marker() {
        let pre = seeded_state();
        assert!(
            !pre.children.contains_key(&B),
            "fixture precondition: B holds no list"
        );
        let mutations = plan(&pre, vec![create(B, 10, "N")]);

        assert_eq!(
            digests_where(&mutations, &pre, is_set_children(B)),
            vec![absent_list()]
        );
    }

    #[test]
    fn from_plan_empty_children_list_of_a_live_section_carries_the_absent_marker() {
        let pre = seeded_state();
        assert_eq!(
            pre.children.get(&C),
            Some(&Vec::new()),
            "fixture precondition: C holds an empty list"
        );
        let mutations = plan(&pre, vec![create(C, 10, "N")]);

        assert_eq!(
            digests_where(&mutations, &pre, is_set_children(C)),
            vec![absent_list()]
        );
    }

    #[test]
    fn from_plan_children_list_of_a_parent_new_in_the_plan_carries_no_digest() {
        let pre = seeded_state();
        // `N` (10) has no row before the plan, so its list is under a parent
        // that is neither the root nor a live section of the pre-state.
        let mutations = plan(&pre, vec![create(ROOT, 10, "N"), create(10, 11, "M")]);

        assert_eq!(
            digests_where(&mutations, &pre, is_set_children(10)),
            vec![None]
        );
    }

    #[test]
    fn from_plan_children_list_of_a_dangling_parent_carries_no_digest() {
        // Hand-built: the pre-state holds a list under 99, which has no row.
        // A normalized entry would not hold it; holding it here shows the skip
        // keys on the parent, not on whether a list is known.
        let mut pre = seeded_state();
        pre.children.insert(99, vec![A]);
        let mutations = vec![
            StorageMutation::DeleteChildrenList { section_id: 99 },
            StorageMutation::SetChildren {
                section_id: 99,
                children: vec![B],
            },
        ];

        assert_eq!(digests(&mutations, &pre), vec![None, None]);
    }

    // ── Field and time-sample rows ───────────────────────────────────────────

    #[test]
    fn from_plan_delete_field_carries_whether_the_pre_state_row_exists() {
        let pre = seeded_state();
        let mutations = vec![delete_field(B, KEY), delete_field(B, "missing")];

        assert_eq!(
            digests(&mutations, &pre),
            vec![
                Some(PreImage::RowExists(true)),
                Some(PreImage::RowExists(false))
            ]
        );
    }

    #[test]
    fn from_plan_delete_time_sample_carries_whether_the_pre_state_row_exists() {
        let pre = seeded_state();
        let mutations = vec![delete_time_sample(B, TIME), delete_time_sample(B, 7.0)];

        assert_eq!(
            digests(&mutations, &pre),
            vec![
                Some(PreImage::RowExists(true)),
                Some(PreImage::RowExists(false))
            ]
        );
    }

    #[test]
    fn from_plan_delete_time_sample_negative_zero_and_zero_are_distinct_rows() {
        // `B` holds a sample at `0.0` only. `-0.0` is a different stored key
        // (its time bits differ) even though `OrderedFloat` equates the two:
        // its row does not exist, and checking it does not use up `0.0`'s
        // check.
        let pre = seeded_state();
        let mutations = vec![delete_time_sample(B, -0.0), delete_time_sample(B, 0.0)];

        assert_eq!(
            digests(&mutations, &pre),
            vec![
                Some(PreImage::RowExists(false)),
                Some(PreImage::RowExists(true))
            ]
        );
    }

    // ── Rule E's added `-0.0` delete (b.8jc) ─────────────────────────────────

    /// [`seeded_state`] with `B`'s `0.0` sample and its field [`KEY`] both
    /// external of `X`, so a plan's untouched-row counts name `X`.
    fn rule_e_pre_state() -> ObjectState {
        with_sample(with_field(seeded_state(), B, KEY, X), B, 0.0, X)
    }

    /// The two engine plans that delete `B`'s time 0 at `+0.0` and so add a
    /// `-0.0` delete (rule E): a reset, and the cascade of deleting `B`.
    fn rule_e_deltas() -> [(&'static str, OwnedDelta); 2] {
        [
            (
                "reset",
                OwnedDelta {
                    is_diff: true,
                    base_version: 0,
                    ..Default::default()
                },
            ),
            (
                "cascade",
                OwnedDelta {
                    struct_commands: vec![delete(B)],
                    ..Default::default()
                },
            ),
        ]
    }

    /// `delta` planned in place on `pre`'s compact projection, journalled and
    /// unbounded, as `CheckedOut::plan_in_place` plans a cache entry: the
    /// planned entry, its indexes and journal (for the pre-image view), and
    /// the plan's mutations.
    fn plan_in_place(
        pre: &ObjectState,
        delta: OwnedDelta,
    ) -> (
        CompactState,
        crate::domain::plan_index::PlanIndexes,
        crate::domain::plan_journal::PlanJournal,
        Vec<StorageMutation>,
    ) {
        use crate::domain::delta::{
            apply_delta_bounded_indexed, ApplyDeltaStats, BoundedPlan, MutationAllowance,
            RewriteCount,
        };
        let mut entry = compact(pre);
        let mut indexes = crate::domain::plan_index::PlanIndexes::build(&entry);
        let mut journal = crate::domain::plan_journal::PlanJournal::new(&entry, &indexes);
        let mut touches = crate::domain::plan_index::PlanTouches::default();
        let plan = apply_delta_bounded_indexed(
            &mut entry,
            Some(delta),
            MutationAllowance::UNBOUNDED,
            &mut |_: RewriteCount| None::<u64>,
            &mut ApplyDeltaStats::default(),
            &mut indexes,
            &mut touches,
            Some(&mut journal),
        )
        .expect("the plan applies in place");
        let mutations = match plan {
            BoundedPlan::Within { mutations, .. } => mutations,
            BoundedPlan::Overflow { needed } => panic!("an unbounded plan overflowed: {needed:?}"),
        };
        (entry, indexes, journal, mutations)
    }

    /// The index of the `DeleteTimeSample` of `(B, time)` by its bits.
    fn sample_delete_at(mutations: &[StorageMutation], time: f64) -> usize {
        mutations
            .iter()
            .position(|m| {
                matches!(m, StorageMutation::DeleteTimeSample { section_id, time: t }
                    if *section_id == B && t.into_inner().to_bits() == time.to_bits())
            })
            .unwrap_or_else(|| panic!("the plan deletes (B, {time:?}): {mutations:#?}"))
    }

    /// Rule E's added record — the `-0.0` delete of a section whose time 0
    /// the pre-plan state holds only at `+0.0` — carries `RowExists(false)`,
    /// from a plain state and from the in-place pre-image view a cache entry
    /// plans through, while the `+0.0` delete carries `RowExists(true)`; the
    /// encoder renders the added record's digest `absent`.
    #[test]
    fn from_plan_rule_e_added_negative_zero_delete_carries_row_absent_on_both_pre_states() {
        let pre = rule_e_pre_state();
        for (what, delta) in rule_e_deltas() {
            // A plain state.
            let mutations =
                apply_delta(&mut pre.clone(), Some(delta.clone())).expect("the plan applies");
            let plain = PreImageDigests::from_plan(&mutations, &pre);
            assert_eq!(
                (
                    plain.for_record(sample_delete_at(&mutations, 0.0)),
                    plain.for_record(sample_delete_at(&mutations, -0.0)),
                ),
                (
                    Some(&PreImage::RowExists(true)),
                    Some(&PreImage::RowExists(false))
                ),
                "{what}, plain state"
            );

            // The in-place view, as `CheckedOut::plan_in_place` plans.
            let (entry, indexes, journal, in_place) = plan_in_place(&pre, delta);
            let view = journal.view(&entry, &indexes);
            let viewed = PreImageDigests::from_plan(&in_place, &view);
            assert_eq!(
                (
                    viewed.for_record(sample_delete_at(&in_place, 0.0)),
                    viewed.for_record(sample_delete_at(&in_place, -0.0)),
                ),
                (
                    Some(&PreImage::RowExists(true)),
                    Some(&PreImage::RowExists(false))
                ),
                "{what}, in-place view"
            );

            // Rendered: the added record's `DT` slot reads `absent`.
            let encoded = crate::storage::lua::encode_mutations(
                &in_place,
                &viewed,
                0,
                &KeySchema::default(),
                "42",
            )
            .expect("the plan encodes");
            let member = KeySchema::time_sample_member(&B.to_string(), -0.0);
            let digest = encoded
                .mutation_argv
                .windows(3)
                .find(|w| w[0] == "DT" && w[1] == member)
                .map(|w| w[2].clone())
                .unwrap_or_else(|| panic!("{what}: a DT record for {member}"));
            assert_eq!(digest, DIGEST_ABSENT, "{what}: rendered");
        }
    }

    /// Rule E's added record touches a row the pre-plan state does not hold,
    /// so it changes no untouched-row count: the counts with it equal the
    /// counts of the same plan without it, from a plain state and from the
    /// in-place view (which counts through the entry's persistent counts).
    #[test]
    fn untouched_ext_counts_are_unchanged_by_rule_e_added_negative_zero_delete() {
        let pre = rule_e_pre_state();
        for (what, delta) in rule_e_deltas() {
            let with_added = apply_delta(&mut pre.clone(), Some(delta)).expect("the plan applies");
            let mut without_added = with_added.clone();
            without_added.remove(sample_delete_at(&with_added, -0.0));

            let plain = counts(&with_added, &pre);
            assert!(
                plain.iter().any(|(hash, _)| *hash == X),
                "{what}: fixture premise: the plan's counts name X: {plain:?}"
            );
            assert_eq!(plain, counts(&without_added, &pre), "{what}, plain state");

            let (entry, indexes, journal, in_place) = plan_in_place(&pre, delta_of(what));
            let view = journal.view(&entry, &indexes);
            assert_eq!(
                PreImageDigests::from_plan(&in_place, &view).untouched_ext_counts(),
                plain.as_slice(),
                "{what}, in-place view"
            );
        }
    }

    /// The [`rule_e_deltas`] delta named `what`.
    fn delta_of(what: &str) -> OwnedDelta {
        rule_e_deltas()
            .into_iter()
            .find(|(name, _)| *name == what)
            .map(|(_, delta)| delta)
            .expect("a rule-E delta of that name")
    }

    // ── A key touched more than once ─────────────────────────────────────────

    #[test]
    fn from_plan_section_deleted_then_recreated_carries_one_pre_state_digest() {
        // `DeleteSection` and `SetSection` of one id share `sec:{sid}`: only
        // the first carries a digest, and it is `B`'s pre-state row.
        let pre = seeded_state();
        let mutations = plan(&pre, vec![delete(B), create(A, B, "B")]);

        let got = digests_where(&mutations, &pre, |m| {
            is_delete_section(B)(m) || is_set_section(B)(m)
        });

        assert_eq!(got, vec![row(B_COLUMNS), None]);
    }

    #[test]
    fn from_plan_section_renamed_twice_carries_its_pre_state_row_once() {
        // The second rename's row is the first's output, not the pre-state;
        // the digest is the pre-state row, on the first record only.
        let pre = seeded_state();
        let mutations = plan(&pre, vec![rename(B, A, "B1"), rename(B, A, "B2")]);

        assert_eq!(
            digests_where(&mutations, &pre, is_set_section(B)),
            vec![row(B_COLUMNS), None]
        );
    }

    #[test]
    fn from_plan_children_list_written_twice_carries_one_pre_state_digest() {
        // Hand-built: `apply_delta` writes each parent's list once per delta,
        // but the rule must not depend on that.
        let pre = seeded_state();
        let mutations = vec![
            StorageMutation::SetChildren {
                section_id: A,
                children: vec![C],
            },
            StorageMutation::DeleteChildrenList { section_id: A },
            StorageMutation::SetChildren {
                section_id: A,
                children: vec![B],
            },
        ];

        assert_eq!(digests(&mutations, &pre), vec![list(&[B, C]), None, None]);
    }

    #[test]
    fn from_plan_field_row_deleted_twice_carries_one_digest() {
        let pre = seeded_state();
        let mutations = vec![delete_field(B, KEY), delete_field(B, KEY)];

        assert_eq!(
            digests(&mutations, &pre),
            vec![Some(PreImage::RowExists(true)), None]
        );
    }

    #[test]
    fn from_plan_time_sample_row_deleted_twice_carries_one_digest() {
        let pre = seeded_state();
        let mutations = vec![delete_time_sample(B, TIME), delete_time_sample(B, TIME)];

        assert_eq!(
            digests(&mutations, &pre),
            vec![Some(PreImage::RowExists(true)), None]
        );
    }

    #[test]
    fn from_plan_set_field_then_delete_field_delete_carries_the_pre_state_absence() {
        // Hand-built. A `SetField` carries no digest, so it does not use up the
        // row's check; and the delete's digest is the pre-state's (absent),
        // not the row the `SetField` wrote just before it.
        let pre = seeded_state();
        let mutations = vec![set_field(B, "new"), delete_field(B, "new")];

        assert_eq!(
            digests(&mutations, &pre),
            vec![None, Some(PreImage::RowExists(false))]
        );
    }

    #[test]
    fn from_plan_set_time_sample_then_delete_delete_carries_the_pre_state_absence() {
        let pre = seeded_state();
        let mutations = vec![
            StorageMutation::SetTimeSample {
                sample: time_sample(B, 7.0),
            },
            delete_time_sample(B, 7.0),
        ];

        assert_eq!(
            digests(&mutations, &pre),
            vec![None, Some(PreImage::RowExists(false))]
        );
    }

    // ── Kinds without a digest ───────────────────────────────────────────────

    #[test]
    fn from_plan_tombstone_set_field_set_time_sample_and_header_carry_no_digest() {
        // The field and sample rows the `SetField` / `SetTimeSample` rewrite
        // exist in the pre-state, so their missing digest is the rule, not an
        // absent row. `ObjectState` holds no tombstones: the `CreateTombstone`
        // names `del:{B}`, which no pre-state can hold; what the pre-state
        // holds is the live section `B`, and the record still carries none.
        // The header carries none whatever the state (the fence covers it).
        let pre = seeded_state();
        let mutations = vec![
            StorageMutation::CreateTombstone {
                section_id: B,
                first_version: 1,
                last_version: 6,
            },
            set_field(B, KEY),
            StorageMutation::SetTimeSample {
                sample: time_sample(B, TIME),
            },
            StorageMutation::SetObjectHeader {
                object_id: 42,
                sequence_num: 6,
                first_version: 1,
                source_format: None,
            },
        ];

        assert_eq!(digests(&mutations, &pre), vec![None, None, None, None]);
    }

    // ── Untouched-row counts ─────────────────────────────────────────────────

    /// External hashes, each the 32-byte repetition of its tag.
    const X: [u8; 32] = [0x58; 32];
    const Y: [u8; 32] = [0x59; 32];
    const Z: [u8; 32] = [0x5A; 32];
    const W: [u8; 32] = [0x57; 32];

    fn external(hash: [u8; 32]) -> FieldValue {
        FieldValue::External {
            hash,
            size: 64,
            ext_value_index: 0,
        }
    }

    /// [`seeded_state`] with field `(section_id, key_name)` holding `hash`.
    fn with_field(
        mut state: ObjectState,
        section_id: u64,
        key_name: &str,
        hash: [u8; 32],
    ) -> ObjectState {
        state.fields.insert(
            (section_id, key_name.to_owned()),
            Field {
                value: external(hash),
                ..field(section_id, key_name)
            },
        );
        state
    }

    /// `state` with the sample at `(section_id, time)` holding `hash`.
    fn with_sample(
        mut state: ObjectState,
        section_id: u64,
        time: f64,
        hash: [u8; 32],
    ) -> ObjectState {
        state.time_samples.insert(
            (section_id, OrderedFloat(time)),
            TimeSample {
                value: external(hash),
                ..time_sample(section_id, time)
            },
        );
        state
    }

    fn set_external_field(section_id: u64, key_name: &str, hash: [u8; 32]) -> StorageMutation {
        StorageMutation::SetField {
            field: Field {
                value: external(hash),
                ..field(section_id, key_name)
            },
        }
    }

    fn counts(mutations: &[StorageMutation], pre: &ObjectState) -> Vec<([u8; 32], u64)> {
        PreImageDigests::from_plan(mutations, pre)
            .untouched_ext_counts()
            .to_vec()
    }

    #[test]
    fn untouched_ext_counts_exclude_a_touched_sample_by_its_stored_key_bits() {
        // `B`'s samples at `0.0` and `TIME` and its field `KEY` all hold `X`.
        // Deleting the `KEY` field puts `X` in the set; deleting a `-0.0`
        // sample touches a row the state does not hold, and must not exclude
        // the `0.0` row the map key's `OrderedFloat` equates it with. So the
        // `0.0` and `TIME` samples are both untouched.
        let pre = with_sample(
            with_sample(with_field(seeded_state(), B, KEY, X), B, 0.0, X),
            B,
            TIME,
            X,
        );
        let mutations = vec![delete_time_sample(B, -0.0), delete_field(B, KEY)];

        assert_eq!(counts(&mutations, &pre), vec![(X, 2)]);
    }

    #[test]
    fn untouched_ext_counts_exclude_the_touched_zero_sample_itself() {
        // The converse: deleting the `0.0` sample does touch it.
        let pre = with_sample(
            with_sample(with_field(seeded_state(), B, KEY, X), B, 0.0, X),
            B,
            TIME,
            X,
        );
        let mutations = vec![delete_time_sample(B, 0.0)];

        assert_eq!(counts(&mutations, &pre), vec![(X, 2)]);
    }

    /// [`seeded_state`]'s compact projection with field `(B, KEY)` and two
    /// time-sample rows of `B`, at `0.0` and at `-0.0`, all holding `X`: two
    /// stored keys the compact form holds apart (b.mr6 PD1), which no full
    /// state can hold at once.
    fn compact_with_signed_zero_rows() -> CompactState {
        let mut pre = compact(&with_field(seeded_state(), B, KEY, X));
        for time in [0.0f64, -0.0] {
            pre.time_samples.insert(B, time.to_bits(), Some(X));
        }
        pre
    }

    /// On a compact pre-image, the untouched rows counted for a hash include
    /// both signed-zero rows: each is its own stored key.
    #[test]
    fn untouched_ext_counts_on_a_compact_state_count_both_signed_zero_rows() {
        let pre = compact_with_signed_zero_rows();

        let untouched = PreImageDigests::from_plan(&[delete_field(B, KEY)], &pre);
        assert_eq!(untouched.untouched_ext_counts(), [(X, 2)]);

        // Touching one of them leaves the other untouched.
        let one_touched = PreImageDigests::from_plan(&[delete_time_sample(B, -0.0)], &pre);
        assert_eq!(one_touched.untouched_ext_counts(), [(X, 2)]);
        assert_eq!(
            one_touched.per_record,
            [Some(PreImage::RowExists(true))],
            "the `-0.0` row is held"
        );
    }

    #[test]
    fn untouched_ext_counts_carry_a_deleted_rows_believed_hash_with_a_zero_count() {
        // `KEY` is the only row holding `X`; `k2` holds `Y`, which the plan
        // does not name. Deleting `KEY` gives `X` an entry of zero untouched
        // rows, and `Y` no entry at all.
        let pre = with_field(with_field(seeded_state(), B, KEY, X), B, "k2", Y);
        let mutations = vec![delete_field(B, KEY)];

        assert_eq!(counts(&mutations, &pre), vec![(X, 0)]);
    }

    #[test]
    fn untouched_ext_counts_carry_a_rewritten_rows_believed_hash() {
        // `KEY` held `X` and is rewritten inline; `k2` still holds `X`.
        let pre = with_field(with_field(seeded_state(), B, KEY, X), B, "k2", X);
        let mutations = vec![set_field(B, KEY)];

        assert_eq!(counts(&mutations, &pre), vec![(X, 1)]);
    }

    #[test]
    fn untouched_ext_counts_carry_final_writes_and_not_intermediate_ones() {
        // `k2` (untouched) holds `Z`. The plan writes `Z` to a new row, and
        // writes `W` to another row it then rewrites inline: `W` is no row's
        // final write and no row's believed original, so it gets no entry.
        let pre = with_field(seeded_state(), B, "k2", Z);
        let mutations = vec![
            set_external_field(B, "new", Z),
            set_external_field(B, "other", W),
            set_field(B, "other"),
        ];

        assert_eq!(counts(&mutations, &pre), vec![(Z, 1)]);
    }

    #[test]
    fn untouched_ext_counts_count_a_row_touched_twice_once() {
        // `KEY` and `k2` hold `X`; the plan touches `KEY` twice. Excluding it
        // twice would leave `k2`'s one row counted as zero.
        let pre = with_field(with_field(seeded_state(), B, KEY, X), B, "k2", X);
        let mutations = vec![
            delete_field(B, KEY),
            set_external_field(B, KEY, X),
            delete_field(B, KEY),
        ];

        assert_eq!(counts(&mutations, &pre), vec![(X, 1)]);
    }

    #[test]
    fn untouched_ext_counts_are_sorted_by_hash_whatever_the_plan_order() {
        // Untouched rows: `k2` holds `Z`, `k3` holds `X`. The plan names
        // `Z`, `Y`, `X` in that order.
        let pre = with_field(with_field(seeded_state(), B, "k2", Z), B, "k3", X);
        let mutations = vec![
            set_external_field(A, "a", Z),
            set_external_field(A, "b", Y),
            set_external_field(A, "c", X),
        ];

        assert_eq!(counts(&mutations, &pre), vec![(X, 1), (Y, 0), (Z, 1)]);
    }

    #[test]
    fn untouched_ext_counts_are_empty_for_a_plan_naming_no_external_hash() {
        // The state holds an external row, but no touched row names a hash:
        // inline rewrites of inline rows, deletes of inline rows, and
        // structural records.
        let pre = with_field(seeded_state(), A, "ext", X);
        let mut mutations = plan(&pre, vec![create(A, 9, "D"), delete(C)]);
        mutations.extend([
            set_field(B, KEY),
            delete_time_sample(B, TIME),
            delete_field(B, "absent"),
        ]);

        assert_eq!(counts(&mutations, &pre), vec![]);
    }

    // ── Refusal decoding ─────────────────────────────────────────────────────

    /// A commit for [`decode_structural_refusal`], which reads only its
    /// identity (for the `CACHE_DIVERGED` log line).
    fn refusal_commit() -> StructuralCommit {
        StructuralCommit {
            session_id: "s".to_owned(),
            layer_name: "l".to_owned(),
            object_id: 42,
            expected_seq: 5,
            expected_commit_chain: String::new(),
            next_set_order: 0,
            mutations: Vec::new(),
            pre_image_digests: PreImageDigests::from_plan(&[], &ObjectState::default()),
            relay_ompf: None,
            commit_token: "t".to_owned(),
            operation: None,
            client_id: "c".to_owned(),
            client_ts: 0,
        }
    }

    fn decoded(details: &str) -> Option<Result<CommitOutcome, ServiceError>> {
        decode_structural_refusal(&refusal_commit(), details)
    }

    /// The drift element is element 2 of a committed structural reply, and
    /// the release tokens follow it untouched.
    #[test]
    fn decode_settle_drift_splits_the_counts_from_the_release_tokens() {
        let tokens: Vec<String> = ["drift:2:1", "rewrite:ab", "surplus_acquire:cd"]
            .map(String::from)
            .to_vec();
        let (drift, releases) = decode_settle_drift(&tokens);
        assert_eq!(drift, Some(SettleDrift { under: 2, over: 1 }));
        assert_eq!(releases, &tokens[1..]);

        let only: Vec<String> = vec!["drift:0:0".to_owned()];
        let (drift, releases) = decode_settle_drift(&only);
        assert_eq!(drift, Some(SettleDrift { under: 0, over: 0 }));
        assert!(releases.is_empty());
    }

    /// A missing or malformed drift element is never read as zero drift,
    /// and never swallows a release token: a malformed `drift:` element is
    /// dropped, and with no `drift:` element the whole tail is releases.
    #[test]
    fn decode_settle_drift_answers_none_for_a_missing_or_malformed_element() {
        let empty: Vec<String> = vec![];
        assert_eq!(decode_settle_drift(&empty), (None, &empty[..]));
        let untagged: Vec<String> = vec!["rewrite:ab".to_owned()];
        assert_eq!(decode_settle_drift(&untagged), (None, &untagged[..]));
        for malformed in ["drift:1", "drift:x:0", "drift:0:-1"] {
            let tokens: Vec<String> = vec![malformed.to_owned(), "rewrite:ab".to_owned()];
            assert_eq!(
                decode_settle_drift(&tokens),
                (None, &tokens[1..]),
                "{malformed:?}: no drift, and the release token after it kept"
            );
        }
    }

    /// Both modes' step-1d reply, `{"FENCED", "<ms>"}`, is a success reply
    /// decoded by the one decoder both modes use; anything but the tag and
    /// exactly one decimal after it is `Internal`, and any other reply is not
    /// a fence.
    #[test]
    fn decode_fenced_reply_takes_the_tag_and_one_decimal_only() {
        let reply = |parts: &[&str]| parts.iter().map(|p| (*p).to_owned()).collect::<Vec<_>>();
        assert!(matches!(
            decode_fenced_reply(&reply(&["FENCED", "250"])),
            Some(Ok(250))
        ));
        for malformed in [
            reply(&["FENCED"]),
            reply(&["FENCED", ""]),
            reply(&["FENCED", "1.5"]),
            reply(&["FENCED", "-1"]),
            reply(&["FENCED", "250", "rewrite:ab"]),
        ] {
            assert!(
                matches!(
                    decode_fenced_reply(&malformed),
                    Some(Err(ServiceError::Internal(_)))
                ),
                "{malformed:?} must be refused Internal"
            );
        }
        for other in [
            reply(&["17"]),
            reply(&["17", "drift:0:0"]),
            reply(&["OVERTAKEN"]),
            reply(&["FENCEDX", "250"]),
            reply(&[]),
        ] {
            assert!(
                decode_fenced_reply(&other).is_none(),
                "{other:?} is not a fence"
            );
        }
    }

    #[test]
    fn decode_structural_refusal_stale_with_a_decimal_is_stale() {
        assert!(matches!(
            decoded("STALE 17"),
            Some(Ok(CommitOutcome::Stale { seq: 17 }))
        ));
    }

    #[test]
    fn decode_structural_refusal_cache_diverged_is_cache_diverged() {
        assert!(matches!(
            decoded("CACHE_DIVERGED: mutation record 1 (SEC) k: differs"),
            Some(Ok(CommitOutcome::CacheDiverged))
        ));
    }

    #[test]
    fn decode_structural_refusal_stale_without_a_decimal_is_internal() {
        let mut failures = Vec::new();
        for details in ["STALE ", "STALE x", "STALE -1"] {
            let expected =
                format!("update_layer structural commit: unparseable refusal reply {details:?}");
            match decoded(details) {
                Some(Err(ServiceError::Internal(msg))) if msg == expected => {}
                other => failures.push(format!("{details:?}: got {other:?}")),
            }
        }
        assert!(
            failures.is_empty(),
            "each must be refused Internal:\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn decode_structural_refusal_other_replies_are_left_to_the_shared_mapping() {
        // `STALE` is a structural refusal only when followed by its space
        // separator, and `CACHE_DIVERGED` only when followed by its `:`, so a
        // lookalike prefix of either goes on to
        // `LayerStore::map_update_script_error`, as does any other tag. The
        // `CACHE_DIVERGED` lookalikes carry a `:` later in the reply, so only
        // a match on the separator right after the tag leaves them alone. A
        // `FENCED` error string is not the script's fence reply (a success
        // reply, `decode_fenced_reply`), so it is left alone too.
        for details in [
            "WRONGTYPE_KEY: k holds a string, expected hash (mutation record 1)",
            "ENCODING: no HDR record",
            "NOT_FOUND: object header missing",
            "STALENESS 5",
            "FENCED 250",
            "FENCEDX 5",
            "CACHE_DIVERGEDX: mutation record 1 (SEC) k: differs",
            "CACHE_DIVERGED mutation record 1 (SEC) k: differs",
        ] {
            assert!(
                decoded(details).is_none(),
                "{details:?} must be left to the shared mapping"
            );
        }
    }
}
