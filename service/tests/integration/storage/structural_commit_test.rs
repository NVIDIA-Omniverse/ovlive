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

//! The structural commit: `storage::structural::commit_structural` over
//! `update_layer_fast.lua`'s structural mode (plan `b.dkt` Epic 5 Task 2;
//! Spec SDD SR-6).
//!
//! Every commit here is planned the way a structural cache entry would plan
//! it, by the test-support planner
//! (`storage::test_support::structural_commit::plan_structural_commit`: a
//! consistent full read at the header `seq`, then complete-mode
//! `apply_delta`), or hand-built on top of such a plan where the rule under
//! test turns on a plan the engine does not emit. Objects are seeded through
//! `create_layer` from `snapshot_read_atomic_test`'s `every_shape_state`.
//!
//! | Area | What it proves |
//! |---|---|
//! | Writes | a commit of each record tag lands that tag's writes — the object read back equals the plan's post-state, and the indexes and tombstones the read does not show hold their values — and answers `Committed` at `expected_seq + 1`, the header `seq` advanced to it |
//! | Header, stream, chain | the `HDR` columns; the stream entry's ID, payload and verbatim `ompf`; the successor `commit_chain` the C4 rustdoc states |
//! | The fence | `STALE` on a `seq` mismatch and on a same-`seq` `commit_chain` change; a header without the column fences on `""`, so a plan made on a chain is `STALE` once the column is gone |
//! | The digests | `CACHE_DIVERGED` per digest kind after an out-of-band edit — section row (on `DSEC`, each of its six columns alone; on the `SEC` of a move and of a rename, the edit undone then landing the same commit), `absent` section (on `SEC` and `DSEC`), children list (on `CHL` and `DCHL`), row existence (on `DF` and `DT`, both directions); a plan touching one section row or one children list twice still commits (digests come from the pre-plan state, never from a mutation's `previous` or the value an earlier mutation wrote); the Rust list digest equals the script's `redis.sha1hex`, and the absent marker is the token the script matches an absent key with |
//! | Refusals | an `HDR` `seq` that is not `expected_seq + 1`, a plan without a header record and one with two are the `ENCODING`-mapped `Internal`; digests built from another plan are refused before anything is sent |
//! | The parse pass | through a raw structural-mode invocation (`StructuralInvocation`, `pub(crate)` for Task 3's hand-crafted records), each malformed input the parse pass refuses — an empty call token, a commit release ring TTL that is empty, `0`, zero-led, not all digits or 13 digits long, mode, `expected_seq`, `expected_commit_chain`, the untouched-row count block (its size, a block past the end of ARGV, a hash's length, case and digits, a count, hashes not strictly ascending), `num_mutations`, a missing record key, an unknown tag, a truncated record, a children count, every digest token, a value-hfield count, `set_order`, `version`, the `HDR` key, `next_set_order`, `has_src_fmt`, a second or missing `HDR`, a trailing slot — answers its `ENCODING` refusal and writes nothing |
//! | Already landed | a commit token recorded by a fast-mode commit, or by an earlier structural commit, answers `AlreadyLanded` with that record's `seq` / `server_ts` |
//! | Client operation ids | a commit carrying an `operation_id` records the `op.<id>:<seq>:<server_ts>:<token>:<digest>` shape; a re-run is answered, ahead of the fence, by whose record it is — this call's own (`AlreadyLanded`), an earlier call's same request (`Recorded(EarlierCall)`) or different request (`Recorded(Conflict)`), or another operation whose start record moved (`Overtaken`), or a chained operation whose predecessor is not the record (`PredecessorPending`) — writing and releasing nothing; an operation without a writer guard is the `ENCODING` refusal |
//! | Once-only send | a killed connection fails the commit transiently with nothing landed, never replayed after the reconnect, and the next attempt commits once (with and without an operation) |
//! | The observer | the calls each outcome makes, and `commit_size`'s records and bytes |
//! | Source shape | both modes call the one already-landed decode and the one error mapping, and size the `KEYS` / `ARGV` they send with the one `script_call_bytes`; no `.await` between the reply decode and `landed` / `already_landed`; no `txn_pool`, `WATCH` or `MULTI` |
//! | Test support | the keyspace dump names every key under the object's tag and normalizes what differs by design, in both commit-token record shapes |
//! | Fault seams (`test-fault-injection`) | the call order size → landed → release enqueue, with the commit visible at `landed`; the pre- and post-`EVALSHA` fault queues; a corrupt token record refused with the fast path's exact `Internal` |
//!
//! Every refusal is checked to write nothing against the object's whole
//! keyspace (`dump_object_keyspace`, plus the header columns that dump leaves
//! out and the external-value pending set).
//!
//! Elsewhere: the guard-parity suite (superseded, stream top, ceilings,
//! `WRONGTYPE_KEY` per key family, `obj:ext` counting, `obj:chs`, replay) is
//! Task 3's, in `structural_guard_parity_test`, which shares this file's
//! fixture (the `pub(crate)` items below); the fold of structural commits
//! interleaved with the other modes is in `commit_chain_tests`; keyspace
//! equality with the WATCH path is Task 4's proptest in `tests/proptests.rs`
//! (`structural_watch_equivalence`), and the named cases of `obj:ext` on a
//! drifted index (where the raise-only floor agrees with the WATCH path and
//! where it differs) are `structural_watch_equivalence_test`; the digest and
//! untouched-row count rules are unit tests in `src/storage/structural.rs`,
//! the record encoder's in `src/storage/lua.rs`. `CommitOutcome::Fenced` on a
//! real lease, and `FENCED` in the step-1b order, are in
//! `structural_guard_parity_test`'s `FENCED` section; the one decoder both
//! modes share (`layer::decode_fenced_reply`) and the `STALE` reply without
//! its decimal are unit tested in `src/storage/structural.rs`.
//!
//! The `fault_seams` module is `#[cfg(feature = "test-fault-injection")]` and
//! is compiled out, not skipped, without that feature.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use bytes::Bytes;
use fred::interfaces::{
    HashesInterface, KeysInterface, ListInterface, SetsInterface, SortedSetsInterface,
    StreamsInterface,
};
use oodf_ffi::{OwnedDelta, SetField, TimeSample as OodfTimeSample};
use ordered_float::OrderedFloat;

use live_edit::config::{
    LAYER_STREAM_MAX_BYTES_DEFAULT, RPC_DEADLINE_MS_DEFAULT, VALKEY_COMMAND_TIMEOUT_MS_DEFAULT,
};
use live_edit::domain::delta::StorageMutation;
use live_edit::domain::time::now_micros;
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section, TimeSample};
use live_edit::error::ServiceError;
use live_edit::storage::commit_chain::{commit_chain_fold, update_stream_payload};
use live_edit::storage::diff_read::{
    read_object_snapshot, snapshot_to_object_state, SnapshotRead, SnapshotReadConsumer,
};
use live_edit::storage::keys::{commit_release_ring_ttl, KeySchema};
use live_edit::storage::layer::{
    operation_request_digest, LayerStore, Operation, StartRecord, WriterGuard,
    HEADER_FIELD_COMMIT_CHAIN,
};
use live_edit::storage::lua::{
    children_list_digest, encode_mutations, update_script_shared_keys, UpdateScriptMode,
    DIGEST_ABSENT, UPDATE_LAYER_FAST_SCRIPT, UPDATE_SCRIPT_ARG_CALL_TOKEN,
    UPDATE_SCRIPT_ARG_CLIENT_ID, UPDATE_SCRIPT_ARG_CLIENT_TS, UPDATE_SCRIPT_ARG_COMMIT_KEY,
    UPDATE_SCRIPT_ARG_NUM_MUTATIONS, UPDATE_SCRIPT_ARG_OBJECT_ID, UPDATE_SCRIPT_ARG_RECORD_SUFFIX,
    UPDATE_SCRIPT_ARG_RELAY_BUDGET_BYTES, UPDATE_SCRIPT_ARG_RELAY_GENERATIONS,
    UPDATE_SCRIPT_ARG_RELAY_GENERATION_BYTES, UPDATE_SCRIPT_ARG_RELAY_MIN_ENTRIES,
    UPDATE_SCRIPT_ARG_RELAY_OMPF, UPDATE_SCRIPT_ARG_RING_TTL_MS, UPDATE_SCRIPT_ARG_SERVER_TS,
    UPDATE_SCRIPT_ARG_STREAM_MAXLEN, UPDATE_SCRIPT_ARG_WRITER_GUARD, UPDATE_SCRIPT_SHARED_ARGV,
    UPDATE_SCRIPT_SHARED_KEYS,
};
use live_edit::storage::relay_retention::RelayRetention;
use live_edit::storage::structural::{
    commit_structural, CommitObserver, CommitOutcome, PreImageDigests, StructuralCommit,
};
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::structural_commit::{
    dump_object_keyspace, plan_structural_commit, CommitTokens, DumpedValue, ObjectKeyspaceDump,
    PlanRequest, PlannedCommit, DUMP_COMMIT_TOKEN, DUMP_DROPPED_HEADER_COLUMNS, DUMP_OBJECT_ID,
    DUMP_SERVER_TS,
};
use live_edit::storage::test_support::{rewrite_out_of_band, OutOfBandChange, TestValkey};

use crate::common::owned_delta::{delta_to_ompf_bytes, delta_with_set_field_inline, reset_delta};
use crate::storage::b_qpf_tests::{ext_store, pre_store};
use crate::storage::snapshot_read_atomic_test::{
    create, create_cmd, create_session, delete_cmd, every_shape_state, layer_store, move_cmd,
    structural, CUBE, ROOT, SIBLING, WORLD,
};

/// The writer every commit names.
pub(crate) const WRITER: &str = "structural-commit-writer";
/// The writer's timestamp. Arbitrary, and distinct from every other number a
/// stream payload carries.
pub(crate) const CLIENT_TS: u64 = 9_000;
/// A section id no fixture section uses.
pub(crate) const NEW_SECTION: u64 = 10;
/// A second unused section id.
pub(crate) const OTHER_NEW_SECTION: u64 = 11;
/// `CUBE`'s external field and time sample in `every_shape_state`.
pub(crate) const CUBE_FIELD: &str = "mesh";
pub(crate) const CUBE_SAMPLE_TIME: f64 = 1.5;
/// `WORLD`'s inline field and time sample in `every_shape_state`.
pub(crate) const WORLD_FIELD: &str = "color";
pub(crate) const WORLD_SAMPLE_TIME: f64 = 0.5;

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// One `every_shape_state` object, created through `create_layer` on the Lua
/// backend.
pub(crate) struct Fixture {
    pub(crate) tv: TestValkey,
    pub(crate) ls: LayerStore,
    pub(crate) session_id: String,
    pub(crate) layer: &'static str,
    pub(crate) oid: u64,
}

pub(crate) async fn fixture(layer: &'static str) -> Fixture {
    fixture_on(TestValkey::setup().await, layer).await
}

/// [`fixture`] on a given test Valkey (a data pool of one connection, for
/// the dropped-connection test).
pub(crate) async fn fixture_on(tv: TestValkey, layer: &'static str) -> Fixture {
    let ls = layer_store(&tv);
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    Fixture {
        tv,
        ls,
        session_id,
        layer,
        oid,
    }
}

/// One `CommitObserver` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ObserverCall {
    CommitSize { records: u64, bytes: u64 },
    Landed { seq: u64, server_ts: u64 },
    AlreadyLanded { seq: u64, server_ts: u64 },
}

/// Records every observer call, in order.
#[derive(Default)]
struct RecordingObserver {
    calls: Vec<ObserverCall>,
}

impl CommitObserver for RecordingObserver {
    fn commit_size(&mut self, records: u64, bytes: u64) {
        self.calls.push(ObserverCall::CommitSize { records, bytes });
    }

    fn landed(&mut self, seq: u64, server_ts: u64) {
        self.calls.push(ObserverCall::Landed { seq, server_ts });
    }

    fn already_landed(&mut self, seq: u64, server_ts: u64) {
        self.calls
            .push(ObserverCall::AlreadyLanded { seq, server_ts });
    }
}

/// Assert `calls` is exactly one `commit_size`: the commit was sent and
/// neither landing method ran.
pub(crate) fn assert_sent_without_landing(calls: &[ObserverCall], context: &str) {
    assert!(
        matches!(calls, [ObserverCall::CommitSize { .. }]),
        "{context}: the observer must record commit_size and no landing call; got {calls:?}"
    );
}

/// The object's whole keyspace, for "nothing was written" checks: the
/// test-support dump with tokens verbatim, the header in full (the dump drops
/// four columns that differ between fixtures, but a refusal must not touch
/// them either), and the external-value pending set a commit enqueues onto.
#[derive(Debug, PartialEq)]
pub(crate) struct Keyspace {
    object: ObjectKeyspaceDump,
    header: BTreeMap<String, String>,
    ext_pending: BTreeSet<String>,
}

impl Fixture {
    pub(crate) fn keys(&self) -> &KeySchema {
        self.tv.key_schema()
    }

    pub(crate) fn oid_str(&self) -> String {
        self.oid.to_string()
    }

    /// `full_key` relative to the object, as `dump_object_keyspace` names it.
    fn relative(&self, full_key: &str) -> String {
        let header_key = self.keys().obj_header(&self.oid_str());
        let stem = header_key
            .strip_suffix("obj:hdr")
            .expect("the header key ends in `obj:hdr`");
        full_key
            .strip_prefix(stem)
            .unwrap_or_else(|| panic!("{full_key} is not one of object {}'s keys", self.oid))
            .to_owned()
    }

    /// Plan `delta` with the test-support planner, as `WRITER` with `token`.
    pub(crate) async fn plan(&self, delta: OwnedDelta, token: &str) -> PlannedCommit {
        let relay_ompf = Bytes::from(delta_to_ompf_bytes(&delta, &[]));
        plan_structural_commit(
            &self.ls,
            PlanRequest {
                session_id: &self.session_id,
                layer_name: self.layer,
                object_id: self.oid,
                delta: Some(delta),
                relay_ompf: Some(relay_ompf),
                commit_token: token,
                client_id: WRITER,
                client_ts: CLIENT_TS,
            },
        )
        .await
        .expect("the planner must plan the fixture delta")
    }

    pub(crate) async fn commit_with(
        &self,
        ls: &LayerStore,
        commit: &StructuralCommit,
    ) -> (Result<CommitOutcome, ServiceError>, Vec<ObserverCall>) {
        let mut observer = RecordingObserver::default();
        let result = commit_structural(ls, commit, &mut observer).await;
        (result, observer.calls)
    }

    pub(crate) async fn commit(
        &self,
        commit: &StructuralCommit,
    ) -> (Result<CommitOutcome, ServiceError>, Vec<ObserverCall>) {
        self.commit_with(&self.ls, commit).await
    }

    /// Commit and require `Committed` at `expected_seq + 1`, with the header
    /// `seq` advanced to it; returns the commit's `server_ts`.
    pub(crate) async fn commit_landed(&self, commit: &StructuralCommit) -> u64 {
        match self.commit(commit).await.0 {
            Ok(CommitOutcome::Committed { seq, server_ts }) => {
                assert_eq!(
                    seq,
                    commit.expected_seq + 1,
                    "a commit lands at expected_seq + 1"
                );
                assert_eq!(
                    self.header_seq().await,
                    seq,
                    "the header seq advances by one"
                );
                server_ts
            }
            other => panic!("the commit must land; got {other:?}"),
        }
    }

    pub(crate) async fn keyspace(&self) -> Keyspace {
        let object = dump_object_keyspace(
            self.tv.client(),
            self.keys(),
            self.oid,
            CommitTokens::Verbatim,
        )
        .await;
        for family in ["obj:hdr", "obj:stream", "obj:secs", "obj:chs"] {
            assert!(
                object.get(family).is_some(),
                "fixture precondition: the dump must hold {family}, or a before / after equality \
                 over it is vacuous; got keys {:?}",
                object.keys.keys().collect::<Vec<_>>()
            );
        }
        let ext_pending: Vec<String> = self
            .tv
            .client()
            .pool
            .smembers(self.keys().ext_pending())
            .await
            .expect("SMEMBERS ext pending must succeed");
        Keyspace {
            object,
            header: self.header().await.into_iter().collect(),
            ext_pending: ext_pending.into_iter().collect(),
        }
    }

    /// Commit, require exactly `expected` with the commit sent and not landed,
    /// and require the keyspace unchanged.
    pub(crate) async fn assert_refused(&self, commit: &StructuralCommit, expected: CommitOutcome) {
        self.assert_refused_in(commit, expected, "").await;
    }

    /// [`Self::assert_refused`], naming `case` in every failure message, for
    /// table-driven tests.
    async fn assert_refused_in(
        &self,
        commit: &StructuralCommit,
        expected: CommitOutcome,
        case: &str,
    ) {
        let before = self.keyspace().await;

        let (result, calls) = self.commit(commit).await;

        match result {
            Ok(outcome) => assert_eq!(outcome, expected, "{case}"),
            Err(e) => panic!("{case}: expected {expected:?}; got the error {e:?}"),
        }
        assert_sent_without_landing(&calls, &format!("{case}: {expected:?}"));
        assert_eq!(
            self.keyspace().await,
            before,
            "{case}: a {expected:?} refusal must write nothing"
        );
    }

    /// Commit, require the script's `ENCODING` refusal naming `fragment`,
    /// mapped to the fast path's encoder-bug `Internal`, with the commit sent,
    /// not landed, and nothing written.
    async fn assert_encoding_refused(&self, commit: &StructuralCommit, fragment: &str) {
        let before = self.keyspace().await;

        let (result, calls) = self.commit(commit).await;

        match result {
            Err(ServiceError::Internal(msg)) => assert!(
                msg.starts_with("update_layer lua encoder bug: ENCODING") && msg.contains(fragment),
                "expected the ENCODING-mapped Internal naming {fragment:?}; got {msg:?}"
            ),
            other => panic!("expected the ENCODING-mapped Internal; got {other:?}"),
        }
        assert_sent_without_landing(&calls, "ENCODING");
        assert_eq!(
            self.keyspace().await,
            before,
            "an ENCODING refusal must write nothing"
        );
    }

    pub(crate) async fn header(&self) -> HashMap<String, String> {
        self.tv
            .client()
            .pool
            .hgetall(self.keys().obj_header(&self.oid_str()))
            .await
            .expect("HGETALL obj:hdr must succeed")
    }

    pub(crate) async fn header_seq(&self) -> u64 {
        self.header()
            .await
            .get("seq")
            .expect("obj:hdr must carry seq")
            .parse()
            .expect("obj:hdr seq must be a u64")
    }

    pub(crate) async fn set_header_column(&self, column: &str, value: &str) {
        let _: () = self
            .tv
            .client()
            .pool
            .hset(self.keys().obj_header(&self.oid_str()), [(column, value)])
            .await
            .expect("HSET obj:hdr must succeed");
    }

    /// Remove the header's `commit_chain` column: the header of an object
    /// materialized before the column existed.
    async fn drop_commit_chain_column(&self) {
        let removed: u64 = self
            .tv
            .client()
            .pool
            .hdel(
                self.keys().obj_header(&self.oid_str()),
                HEADER_FIELD_COMMIT_CHAIN,
            )
            .await
            .expect("HDEL obj:hdr commit_chain must succeed");
        assert_eq!(
            removed, 1,
            "fixture precondition: the header carried the column"
        );
    }

    /// The object as the atomic snapshot read decodes it — the normalized
    /// state a cache entry holds, the form `PlannedCommit::post_state` takes.
    async fn stored_state(&self) -> ObjectState {
        match read_object_snapshot(
            self.tv.client(),
            self.keys(),
            &self.oid_str(),
            SnapshotReadConsumer::Checkpoint,
        )
        .await
        .expect("the snapshot read must succeed")
        {
            SnapshotRead::Snapshot(snapshot) => snapshot_to_object_state(&snapshot.state),
            SnapshotRead::TooLarge(too_large) => {
                panic!("the fixture object is far below the read's limits; got {too_large:?}")
            }
        }
    }

    /// Assert the object read back equals `post_state`, the plan's post-apply
    /// state, normalized as the read normalizes: a list the plan emptied
    /// has no stored form, so it compares as absent.
    pub(crate) async fn assert_store_holds(&self, post_state: &ObjectState) {
        let mut expected = post_state.clone();
        expected.children.retain(|_, list| !list.is_empty());
        assert_eq!(
            self.stored_state().await,
            expected,
            "the object read back must be the plan's post-state"
        );
    }

    pub(crate) async fn members(&self, set_key: String) -> BTreeSet<String> {
        let members: Vec<String> = self
            .tv
            .client()
            .pool
            .smembers(&set_key)
            .await
            .unwrap_or_else(|e| panic!("SMEMBERS {set_key} must succeed: {e}"));
        members.into_iter().collect()
    }

    pub(crate) async fn list(&self, key: String) -> Vec<String> {
        self.tv
            .client()
            .pool
            .lrange(&key, 0, -1)
            .await
            .unwrap_or_else(|e| panic!("LRANGE {key} must succeed: {e}"))
    }

    pub(crate) async fn exists(&self, key: &str) -> bool {
        let n: u64 = self
            .tv
            .client()
            .pool
            .exists(key)
            .await
            .unwrap_or_else(|e| panic!("EXISTS {key} must succeed: {e}"));
        n == 1
    }

    pub(crate) async fn delete_key(&self, key: &str) {
        let removed: u64 = self
            .tv
            .client()
            .pool
            .del(key)
            .await
            .unwrap_or_else(|e| panic!("DEL {key} must succeed: {e}"));
        assert_eq!(
            removed, 1,
            "fixture precondition: {key} must exist to be deleted"
        );
    }

    /// Write a value row at `key` out of band (its columns do not matter, only
    /// that the key exists).
    async fn plant_row(&self, key: &str) {
        let _: () = self
            .tv
            .client()
            .pool
            .hset(key, [("value", "planted")])
            .await
            .unwrap_or_else(|e| panic!("HSET {key} must succeed: {e}"));
    }

    async fn rpush(&self, key: &str, ids: &[u64]) {
        let _: u64 = self
            .tv
            .client()
            .pool
            .rpush(key, ids.iter().map(u64::to_string).collect::<Vec<_>>())
            .await
            .unwrap_or_else(|e| panic!("RPUSH {key} must succeed: {e}"));
    }

    pub(crate) fn section_key(&self, section_id: u64) -> String {
        self.keys()
            .obj_section(&self.oid_str(), &section_id.to_string())
    }

    pub(crate) fn children_key(&self, parent_id: u64) -> String {
        self.keys()
            .obj_children(&self.oid_str(), &parent_id.to_string())
    }

    pub(crate) fn field_key(&self, section_id: u64, key_name: &str) -> String {
        self.keys()
            .obj_field(&self.oid_str(), &section_id.to_string(), key_name)
    }

    pub(crate) fn time_sample_key(&self, section_id: u64, time: f64) -> String {
        self.keys()
            .obj_time_sample(&self.oid_str(), &section_id.to_string(), time)
    }
}

/// A hand-built commit on `base`'s cut: `base`'s identity and fence, the
/// records `body` followed by the header bump to `expected_seq + 1`, and
/// digests built from `pre_state`. For rules that turn on a plan the engine
/// does not emit.
pub(crate) fn hand_built(
    base: &PlannedCommit,
    pre_state: &ObjectState,
    body: Vec<StorageMutation>,
) -> StructuralCommit {
    let mut mutations = body;
    mutations.push(header_bump(base, pre_state));
    hand_built_exact(base, pre_state, mutations)
}

/// The header bump a hand-built commit on `base`'s cut carries.
fn header_bump(base: &PlannedCommit, pre_state: &ObjectState) -> StorageMutation {
    StorageMutation::SetObjectHeader {
        object_id: base.commit.object_id,
        sequence_num: base.commit.expected_seq + 1,
        first_version: pre_state.first_version,
        source_format: None,
    }
}

/// [`hand_built`] with exactly `mutations` — no header bump appended — for
/// the plans the script must refuse for their header records.
pub(crate) fn hand_built_exact(
    base: &PlannedCommit,
    pre_state: &ObjectState,
    mutations: Vec<StorageMutation>,
) -> StructuralCommit {
    StructuralCommit {
        pre_image_digests: PreImageDigests::from_plan(&mutations, pre_state),
        mutations,
        next_set_order: pre_state.next_set_order,
        ..base.commit.clone()
    }
}

fn count(mutations: &[StorageMutation], pick: impl Fn(&StorageMutation) -> bool) -> usize {
    mutations.iter().filter(|m| pick(m)).count()
}

/// Assert the plan carries at least one record `pick` selects, so a fixture
/// that lost the shape under test fails loudly rather than passing vacuously.
fn assert_plan_carries(
    planned: &PlannedCommit,
    kind: &str,
    pick: impl Fn(&StorageMutation) -> bool,
) {
    assert!(
        count(&planned.commit.mutations, pick) > 0,
        "fixture precondition: the plan must carry a {kind}; got {:?}",
        planned.commit.mutations
    );
}

/// `section`'s six stored columns, by name, from a state's row.
fn section_columns(state: &ObjectState, section_id: u64) -> HashMap<String, String> {
    let s = &state.sections[&section_id];
    [
        ("parent_id", s.parent_id.to_string()),
        ("name", s.name.clone()),
        ("type", s.section_type.to_string()),
        ("order", s.section_order.to_string()),
        ("first_ver", s.first_version.to_string()),
        ("last_ver", s.last_version.to_string()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect()
}

fn ids(ids: &[u64]) -> BTreeSet<String> {
    ids.iter().map(u64::to_string).collect()
}

// ═════════════════════════════════════════════════════════════════════════════
// Writes: one commit per record tag
// ═════════════════════════════════════════════════════════════════════════════

/// `SEC` and a non-empty `CHL`: creating a section writes its row, indexes it,
/// rewrites its parent's list and indexes the list.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_create_section_lands_the_sec_row_and_the_parents_list() {
    let fx = fixture("structural-sec").await;
    let planned = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            "tok-sec",
        )
        .await;
    assert_plan_carries(&planned, "SetSection", |m| {
        matches!(m, StorageMutation::SetSection { .. })
    });
    assert_plan_carries(&planned, "SetChildren", |m| {
        matches!(m, StorageMutation::SetChildren { .. })
    });

    fx.commit_landed(&planned.commit).await;

    fx.assert_store_holds(&planned.post_state).await;
    let row: HashMap<String, String> = fx
        .tv
        .client()
        .pool
        .hgetall(fx.section_key(NEW_SECTION))
        .await
        .expect("HGETALL sec must succeed");
    assert_eq!(row, section_columns(&planned.post_state, NEW_SECTION));
    assert!(fx
        .members(fx.keys().obj_sections_index(&fx.oid_str()))
        .await
        .contains(&NEW_SECTION.to_string()));
    assert_eq!(
        fx.list(fx.children_key(WORLD)).await,
        vec![CUBE.to_string(), NEW_SECTION.to_string()]
    );
    assert!(fx
        .members(fx.keys().obj_children_index(&fx.oid_str()))
        .await
        .contains(&WORLD.to_string()));
}

/// `DSEC`, `TOMB`, `DCHL`, `DF`, `DT` and a `CHL` that keeps a list: deleting
/// `WORLD` removes it and `CUBE` (rows, section index members, `WORLD`'s list
/// and its index member, every field and time-sample row with its index
/// member), tombstones both, and rewrites the root's list.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_cascade_delete_lands_dsec_tomb_dchl_df_dt_and_chl_writes() {
    let fx = fixture("structural-cascade").await;
    let planned = fx
        .plan(structural(vec![delete_cmd(ROOT, WORLD)]), "tok-cascade")
        .await;
    for (kind, pick) in [
        (
            "DeleteSection",
            (|m| matches!(m, StorageMutation::DeleteSection { .. }))
                as fn(&StorageMutation) -> bool,
        ),
        ("CreateTombstone", |m| {
            matches!(m, StorageMutation::CreateTombstone { .. })
        }),
        (
            "DeleteChildrenList",
            |m| matches!(m, StorageMutation::DeleteChildrenList { section_id } if *section_id == WORLD),
        ),
        ("DeleteField", |m| {
            matches!(m, StorageMutation::DeleteField { .. })
        }),
        ("DeleteTimeSample", |m| {
            matches!(m, StorageMutation::DeleteTimeSample { .. })
        }),
        (
            "SetChildren",
            |m| matches!(m, StorageMutation::SetChildren { section_id, .. } if *section_id == ROOT),
        ),
    ] {
        assert_plan_carries(&planned, kind, pick);
    }

    fx.commit_landed(&planned.commit).await;

    fx.assert_store_holds(&planned.post_state).await;
    // What the read does not show: index members whose rows are gone, the
    // removed list and its index member, and the tombstones.
    let oid = fx.oid_str();
    let sections = fx.members(fx.keys().obj_sections_index(&oid)).await;
    assert!(
        !sections.contains(&WORLD.to_string()) && !sections.contains(&CUBE.to_string()),
        "obj:secs must no longer name the deleted sections; got {sections:?}"
    );
    assert_eq!(
        fx.members(fx.keys().obj_deleted_sections_index(&oid)).await,
        ids(&[WORLD, CUBE])
    );
    for m in &planned.commit.mutations {
        if let StorageMutation::CreateTombstone {
            section_id,
            first_version,
            last_version,
        } = m
        {
            let tomb: HashMap<String, String> = fx
                .tv
                .client()
                .pool
                .hgetall(fx.keys().obj_deleted_section(&oid, &section_id.to_string()))
                .await
                .expect("HGETALL del must succeed");
            let expected: HashMap<String, String> = [
                ("first_ver".to_owned(), first_version.to_string()),
                ("last_ver".to_owned(), last_version.to_string()),
            ]
            .into();
            assert_eq!(tomb, expected, "tombstone of section {section_id}");
        }
    }
    assert!(
        !fx.exists(&fx.children_key(WORLD)).await,
        "WORLD's list must be gone"
    );
    let indexed_lists = fx.members(fx.keys().obj_children_index(&oid)).await;
    assert!(
        !indexed_lists.contains(&WORLD.to_string()) && indexed_lists.contains(&ROOT.to_string()),
        "obj:chs must drop WORLD and keep the root; got {indexed_lists:?}"
    );
    assert_eq!(
        fx.list(fx.children_key(ROOT)).await,
        vec![SIBLING.to_string()]
    );
    for row in [
        fx.field_key(WORLD, WORLD_FIELD),
        fx.field_key(CUBE, CUBE_FIELD),
        fx.time_sample_key(WORLD, WORLD_SAMPLE_TIME),
        fx.time_sample_key(CUBE, CUBE_SAMPLE_TIME),
    ] {
        assert!(!fx.exists(&row).await, "{row} must be deleted");
    }
    for index in [
        fx.keys().obj_fields_index(&oid),
        fx.keys().obj_time_samples_index(&oid),
    ] {
        let card: u64 = fx
            .tv
            .client()
            .pool
            .zcard(&index)
            .await
            .expect("ZCARD must succeed");
        assert_eq!(card, 0, "{index} must hold no member of a deleted row");
    }
}

/// A `CHL` that empties a list deletes the key and drops its index member, as
/// `execute_mutation`'s `SetChildren` arm does: deleting `CUBE` empties
/// `WORLD`'s list.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_set_children_to_empty_deletes_the_list_and_unindexes_it() {
    let fx = fixture("structural-chl-empty").await;
    let planned = fx
        .plan(structural(vec![delete_cmd(WORLD, CUBE)]), "tok-chl-empty")
        .await;
    assert!(
        planned
            .commit
            .mutations
            .contains(&StorageMutation::SetChildren {
                section_id: WORLD,
                children: Vec::new(),
            }),
        "fixture precondition: the plan must empty WORLD's list; got {:?}",
        planned.commit.mutations
    );

    fx.commit_landed(&planned.commit).await;

    fx.assert_store_holds(&planned.post_state).await;
    assert!(
        !fx.exists(&fx.children_key(WORLD)).await,
        "an emptied list has no stored form"
    );
    assert!(
        !fx.members(fx.keys().obj_children_index(&fx.oid_str()))
            .await
            .contains(&WORLD.to_string()),
        "obj:chs must not name a parent whose list was emptied"
    );
}

/// `SFA` and `STA`: a value delta rewrites one existing field and one existing
/// time sample and adds one of each, with the plan's absolute `set_order`,
/// `authoring_set_order` and `version`, each row scored by its `version`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_value_delta_lands_sfa_and_sta_rows_and_their_scores() {
    let fx = fixture("structural-values").await;
    let value_field = |key: &str, set_order| SetField {
        section_id: WORLD,
        key_name: key.to_owned(),
        value_or_ext_hash: vec![0x5A, set_order as u8],
        ext_value_size: 0,
        ext_value_index: 0,
        set_order,
    };
    let value_sample = |time| OodfTimeSample {
        section_id: WORLD,
        time,
        value_or_ext_hash: vec![0x6B],
        ext_value_size: 0,
        ext_value_index: 0,
    };
    let delta = OwnedDelta {
        set_fields: vec![value_field(WORLD_FIELD, 5), value_field("size", 7)],
        time_samples: vec![value_sample(WORLD_SAMPLE_TIME), value_sample(2.5)],
        ..Default::default()
    };
    let planned = fx.plan(delta, "tok-values").await;
    assert_eq!(
        (
            count(&planned.commit.mutations, |m| matches!(
                m,
                StorageMutation::SetField { .. }
            )),
            count(&planned.commit.mutations, |m| matches!(
                m,
                StorageMutation::SetTimeSample { .. }
            )),
        ),
        (2, 2),
        "fixture precondition: two SetField and two SetTimeSample records"
    );

    fx.commit_landed(&planned.commit).await;

    fx.assert_store_holds(&planned.post_state).await;
    let new_seq = planned.commit.expected_seq + 1;
    for key in [WORLD_FIELD, "size"] {
        let score: Option<f64> = fx
            .tv
            .client()
            .pool
            .zscore(
                fx.keys().obj_fields_index(&fx.oid_str()),
                KeySchema::field_member(&WORLD.to_string(), key),
            )
            .await
            .expect("ZSCORE flds must succeed");
        assert_eq!(
            score,
            Some(new_seq as f64),
            "field {key}'s score is its version"
        );
    }
    for time in [WORLD_SAMPLE_TIME, 2.5] {
        let score: Option<f64> = fx
            .tv
            .client()
            .pool
            .zscore(
                fx.keys().obj_time_samples_index(&fx.oid_str()),
                KeySchema::time_sample_member(&WORLD.to_string(), time),
            )
            .await
            .expect("ZSCORE tss must succeed");
        assert_eq!(
            score,
            Some(new_seq as f64),
            "sample {time}'s score is its version"
        );
    }
}

/// `HDR`: a reset's header record writes `execute_mutation`'s columns — `seq`,
/// the record's `first_ver`, post-apply `next_set_order` and `src_fmt`, the
/// writer, and the commit's server timestamp as `modified_at` and
/// `last_server_ts`. A reset moves `first_ver`, `next_set_order` and
/// `src_fmt`, so each written value differs from the stored one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_reset_lands_the_header_record_columns() {
    let fx = fixture("structural-hdr").await;
    let planned = fx.plan(reset_delta("usdc"), "tok-hdr").await;
    let (pre, post) = (&planned.pre_state, &planned.post_state);
    assert!(
        pre.first_version != post.first_version
            && pre.next_set_order != post.next_set_order
            && pre.source_format != post.source_format,
        "fixture precondition: the reset must move first_ver, next_set_order and src_fmt; \
         pre {:?} post {:?}",
        (pre.first_version, pre.next_set_order, &pre.source_format),
        (post.first_version, post.next_set_order, &post.source_format)
    );

    let server_ts = fx.commit_landed(&planned.commit).await;

    let header = fx.header().await;
    let expected = [
        ("seq", (planned.commit.expected_seq + 1).to_string()),
        ("first_ver", post.first_version.to_string()),
        ("next_set_order", post.next_set_order.to_string()),
        ("src_fmt", post.source_format.clone()),
        ("modified_at", server_ts.to_string()),
        ("last_client_id", WRITER.to_owned()),
        ("last_client_ts", CLIENT_TS.to_string()),
        ("last_server_ts", server_ts.to_string()),
    ];
    for (column, value) in expected {
        assert_eq!(
            header.get(column),
            Some(&value),
            "obj:hdr {column}; header {header:?}"
        );
    }
    fx.assert_store_holds(&planned.post_state).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The stream entry and the commit chain
// ═════════════════════════════════════════════════════════════════════════════

/// The stream entry at `id`: its `payload` and `ompf` fields.
async fn stream_entry(fx: &Fixture, id: &str) -> (Option<String>, Option<Bytes>) {
    let entries: Vec<(String, HashMap<String, Bytes>)> = fx
        .tv
        .client()
        .pool
        .xrange_values(fx.keys().layer_stream(fx.oid), id, id, None)
        .await
        .expect("XRANGE must succeed");
    let [(_, fields)] = entries.as_slice() else {
        panic!("exactly one stream entry must sit at {id}; got {entries:?}");
    };
    (
        fields
            .get("payload")
            .map(|p| String::from_utf8(p.to_vec()).expect("the payload is UTF-8")),
        fields.get("ompf").cloned(),
    )
}

/// The entry sits at `{expected_seq+1}-0`, its payload is
/// `update_stream_payload`'s, and its `ompf` field is the commit's
/// `relay_ompf` byte for byte — absent when the commit carries none.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_stream_entry_carries_the_payload_and_the_relay_ompf_verbatim() {
    for (layer, relay) in [
        ("structural-stream-ompf", true),
        ("structural-stream-no-ompf", false),
    ] {
        let fx = fixture(layer).await;
        let mut commit = fx
            .plan(
                structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
                "tok-stream",
            )
            .await
            .commit;
        if !relay {
            commit.relay_ompf = None;
        }

        let server_ts = fx.commit_landed(&commit).await;

        let seq = commit.expected_seq + 1;
        let (payload, ompf) = stream_entry(&fx, &format!("{seq}-0")).await;
        assert_eq!(
            payload,
            Some(update_stream_payload(
                fx.oid, seq, WRITER, CLIENT_TS, server_ts
            )),
            "{layer}"
        );
        assert_eq!(ompf, commit.relay_ompf, "{layer}");
    }
}

/// The header's `commit_chain` after a commit is the successor the C4 rustdoc
/// states: `commit_chain_fold(expected_commit_chain, update_stream_payload(…))`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_committed_header_commit_chain_is_the_documented_successor() {
    let fx = fixture("structural-chain").await;
    let commit = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            "tok-chain",
        )
        .await
        .commit;
    assert!(
        !commit.expected_commit_chain.is_empty(),
        "fixture precondition: create_layer writes a commit_chain"
    );

    let server_ts = fx.commit_landed(&commit).await;

    let seq = commit.expected_seq + 1;
    assert_eq!(
        fx.header().await.get(HEADER_FIELD_COMMIT_CHAIN),
        Some(&commit_chain_fold(
            &commit.expected_commit_chain,
            &update_stream_payload(fx.oid, seq, WRITER, CLIENT_TS, server_ts),
        ))
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The fence: STALE
// ═════════════════════════════════════════════════════════════════════════════

/// Two plans made at one `seq`: the first commits, the second is `Stale` with
/// the header's new `seq`, and writes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_plan_at_a_superseded_seq_is_stale_with_the_header_seq() {
    let fx = fixture("structural-stale-seq").await;
    let first = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            "tok-first",
        )
        .await
        .commit;
    let second = fx
        .plan(
            structural(vec![create_cmd(WORLD, OTHER_NEW_SECTION, "Desk")]),
            "tok-second",
        )
        .await
        .commit;
    assert_eq!(
        first.expected_seq, second.expected_seq,
        "fixture precondition: one cut"
    );
    fx.commit_landed(&first).await;

    fx.assert_refused(
        &second,
        CommitOutcome::Stale {
            seq: second.expected_seq + 1,
        },
    )
    .await;
}

/// The same `seq` with a different `commit_chain` (a re-materialization, set
/// directly here) is `Stale` with that unchanged `seq`, and writes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_same_seq_with_a_changed_commit_chain_is_stale() {
    let fx = fixture("structural-stale-chain").await;
    let commit = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            "tok-chain-moved",
        )
        .await
        .commit;
    fx.set_header_column(HEADER_FIELD_COMMIT_CHAIN, &"e".repeat(40))
        .await;

    fx.assert_refused(
        &commit,
        CommitOutcome::Stale {
            seq: commit.expected_seq,
        },
    )
    .await;
}

/// A header without the `commit_chain` column fences as `""`: a plan made on
/// it carries `""` and commits, and the fold starts from `""`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_header_without_commit_chain_fences_on_the_empty_string() {
    let fx = fixture("structural-no-chain").await;
    fx.drop_commit_chain_column().await;
    let commit = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            "tok-no-chain",
        )
        .await
        .commit;
    assert_eq!(commit.expected_commit_chain, "", "fixture precondition");

    let server_ts = fx.commit_landed(&commit).await;

    assert_eq!(
        fx.header().await.get(HEADER_FIELD_COMMIT_CHAIN),
        Some(&commit_chain_fold(
            "",
            &update_stream_payload(
                fx.oid,
                commit.expected_seq + 1,
                WRITER,
                CLIENT_TS,
                server_ts
            ),
        ))
    );
}

/// The converse: a plan made while the header carried a `commit_chain`, on a
/// header that has since lost the column, is `Stale` at the unchanged `seq` —
/// the absent column reads as `""`, which is not the plan's chain — and
/// writes nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_header_that_lost_its_commit_chain_is_stale() {
    let fx = fixture("structural-lost-chain").await;
    let commit = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            "tok-lost-chain",
        )
        .await
        .commit;
    assert!(
        !commit.expected_commit_chain.is_empty(),
        "fixture precondition: planned on a header with a commit_chain"
    );
    fx.drop_commit_chain_column().await;

    fx.assert_refused(
        &commit,
        CommitOutcome::Stale {
            seq: commit.expected_seq,
        },
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The digests: CACHE_DIVERGED, and a key touched twice
// ═════════════════════════════════════════════════════════════════════════════

/// A section row's six stored columns, the table the `row` digest test walks.
const SECTION_ROW_COLUMNS: [&str; 6] = [
    "parent_id",
    "name",
    "type",
    "order",
    "first_ver",
    "last_ver",
];

/// Section row, `row` digest (`DSEC`), one case per stored column: each
/// column alone edited out of band is `CACHE_DIVERGED` and writes nothing, so
/// a compare that skips any column fails its case. The column is restored
/// after each case, and the commit then lands: the edit was the only cause.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_section_row_with_any_one_column_edited_out_of_band_is_cache_diverged() {
    let fx = fixture("structural-div-row").await;
    let planned = fx
        .plan(structural(vec![delete_cmd(WORLD, CUBE)]), "tok-div-row")
        .await;
    assert_plan_carries(
        &planned,
        "DeleteSection of CUBE",
        |m| matches!(m, StorageMutation::DeleteSection { section_id, .. } if *section_id == CUBE),
    );
    let row = fx.section_key(CUBE);
    let stored: HashMap<String, String> = fx
        .tv
        .client()
        .pool
        .hgetall(&row)
        .await
        .expect("HGETALL sec must succeed");
    assert_eq!(
        stored,
        section_columns(&planned.pre_state, CUBE),
        "fixture precondition: the stored row is the row the cache holds"
    );
    assert_eq!(
        stored.keys().map(String::as_str).collect::<BTreeSet<_>>(),
        SECTION_ROW_COLUMNS.into_iter().collect(),
        "fixture precondition: the table names every stored column"
    );

    for column in SECTION_ROW_COLUMNS {
        let held = &stored[column];
        let edited = format!("{held}9");
        let _: () = fx
            .tv
            .client()
            .pool
            .hset(&row, [(column, edited.as_str())])
            .await
            .expect("HSET sec must succeed");

        fx.assert_refused_in(
            &planned.commit,
            CommitOutcome::CacheDiverged,
            &format!("column {column} edited {held:?} -> {edited:?}"),
        )
        .await;

        let _: () = fx
            .tv
            .client()
            .pool
            .hset(&row, [(column, held.as_str())])
            .await
            .expect("HSET sec must succeed");
    }

    fx.commit_landed(&planned.commit).await;
}

/// The name [`assert_section_rewrite_with_its_row_edited_out_of_band_is_cache_diverged`]
/// writes over `CUBE`'s stored row out of band: neither the fixture's name nor
/// one either plan writes.
const OUT_OF_BAND_CUBE_NAME: &str = "Cube-out-of-band";

/// Section row, `row` digest on a `SEC`: `delta`, planned by the
/// complete-mode planner on a full read, rewrites the live section `CUBE`,
/// and `rewrite` recognises the new row its `SetSection` writes. With the
/// stored row's `name` edited out of band (`rewrite_out_of_band`), the commit
/// is `CACHE_DIVERGED` and writes nothing; with the edit undone, the same
/// commit lands, so the edit was the only cause. No children list changes
/// with the edit, so of the plan's digests only the `SEC`'s six-column
/// compare can see it. Every other section-row `CACHE_DIVERGED` case drives a
/// `DSEC` or an `absent` `SEC`.
async fn assert_section_rewrite_with_its_row_edited_out_of_band_is_cache_diverged(
    layer: &'static str,
    delta: OwnedDelta,
    rewrite: impl Fn(&Section) -> bool,
) {
    let fx = fixture(layer).await;
    let planned = fx.plan(delta, "tok-div-sec-row").await;
    let cube_records: Vec<&StorageMutation> = planned
        .commit
        .mutations
        .iter()
        .filter(|m| match m {
            StorageMutation::SetSection { section, .. } => section.section_id == CUBE,
            StorageMutation::DeleteSection { section_id, .. } => *section_id == CUBE,
            _ => false,
        })
        .collect();
    assert!(
        matches!(
            cube_records.first(),
            Some(StorageMutation::SetSection { section, previous: Some(_) }) if rewrite(section)
        ),
        "fixture precondition: CUBE's first record is the SetSection under test, rewriting the \
         live row, so it carries CUBE's row digest; got {cube_records:?}"
    );
    let row = fx.section_key(CUBE);
    let stored: HashMap<String, String> = fx
        .tv
        .client()
        .pool
        .hgetall(&row)
        .await
        .expect("HGETALL sec must succeed");
    assert_eq!(
        stored,
        section_columns(&planned.pre_state, CUBE),
        "fixture precondition: the stored row is the six-column row the cache holds, so the \
         SEC's digest is `row`, not `absent`"
    );
    let held_name = stored["name"].clone();

    rewrite_out_of_band(
        &fx.tv,
        fx.oid,
        OutOfBandChange::SectionName {
            section_id: CUBE,
            name: OUT_OF_BAND_CUBE_NAME.to_owned(),
        },
    )
    .await;

    fx.assert_refused(&planned.commit, CommitOutcome::CacheDiverged)
        .await;

    rewrite_out_of_band(
        &fx.tv,
        fx.oid,
        OutOfBandChange::SectionName {
            section_id: CUBE,
            name: held_name,
        },
    )
    .await;
    fx.commit_landed(&planned.commit).await;
    fx.assert_store_holds(&planned.post_state).await;
}

/// A move: `CUBE` from `WORLD` to `SIBLING` under its own name, so its
/// `SetSection` carries a new `parent_id`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_moved_section_with_its_row_edited_out_of_band_is_cache_diverged() {
    assert_section_rewrite_with_its_row_edited_out_of_band_is_cache_diverged(
        "structural-div-sec-move",
        structural(vec![move_cmd(CUBE, WORLD, SIBLING, "Cube")]),
        |section| section.parent_id == SIBLING && section.name == "Cube",
    )
    .await;
}

/// A rename: `CUBE` under `WORLD` still, as `Cube1`, so its `SetSection`
/// carries a new `name`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_renamed_section_with_its_row_edited_out_of_band_is_cache_diverged() {
    assert_section_rewrite_with_its_row_edited_out_of_band_is_cache_diverged(
        "structural-div-sec-rename",
        structural(vec![move_cmd(CUBE, WORLD, WORLD, "Cube1")]),
        |section| section.parent_id == WORLD && section.name == "Cube1",
    )
    .await;
}

/// Section row, `absent` digest: the row the plan creates already exists.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_section_row_the_cache_holds_absent_but_exists_is_cache_diverged() {
    let fx = fixture("structural-div-row-absent").await;
    let commit = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            "tok-div-absent",
        )
        .await
        .commit;
    let _: () = fx
        .tv
        .client()
        .pool
        .hset(fx.section_key(NEW_SECTION), [("parent_id", "1")])
        .await
        .expect("HSET sec must succeed");

    fx.assert_refused(&commit, CommitOutcome::CacheDiverged)
        .await;
}

/// Section row, `absent` digest on a `DSEC`: the cache holds no such section,
/// and the store has its row. Hand-built: the engine deletes only sections its
/// state holds, so only a hand-built plan puts the `absent` digest on a
/// `DeleteSection` rather than a `SetSection`. With the planted row removed,
/// the same commit lands: the row's existence was the only cause.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_deleted_section_the_cache_holds_absent_but_exists_is_cache_diverged() {
    let fx = fixture("structural-div-dsec-absent").await;
    let base = fx.plan(OwnedDelta::default(), "tok-div-dsec-absent").await;
    assert!(
        !base.pre_state.sections.contains_key(&NEW_SECTION),
        "fixture precondition: the cache holds no NEW_SECTION"
    );
    let commit = hand_built(
        &base,
        &base.pre_state,
        vec![StorageMutation::DeleteSection {
            section_id: NEW_SECTION,
            parent_id: WORLD,
            name: "Ghost".to_owned(),
        }],
    );
    let row = fx.section_key(NEW_SECTION);
    fx.plant_row(&row).await;

    fx.assert_refused(&commit, CommitOutcome::CacheDiverged)
        .await;

    fx.delete_key(&row).await;
    fx.commit_landed(&commit).await;
}

/// Children list, list digest: an extra id pushed onto the stored list.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_children_list_edited_out_of_band_is_cache_diverged() {
    let fx = fixture("structural-div-list").await;
    let commit = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            "tok-div-list",
        )
        .await
        .commit;
    fx.rpush(&fx.children_key(WORLD), &[99]).await;

    fx.assert_refused(&commit, CommitOutcome::CacheDiverged)
        .await;
}

/// Children list, list digest: the stored list the cache holds is gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_children_list_the_cache_holds_but_absent_is_cache_diverged() {
    let fx = fixture("structural-div-list-gone").await;
    let commit = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            "tok-div-gone",
        )
        .await
        .commit;
    fx.delete_key(&fx.children_key(WORLD)).await;

    fx.assert_refused(&commit, CommitOutcome::CacheDiverged)
        .await;
}

/// Children list, absent marker: `SIBLING` has no list in the cache, and one
/// exists in the store.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_children_list_the_cache_holds_absent_but_exists_is_cache_diverged() {
    let fx = fixture("structural-div-list-absent").await;
    let commit = fx
        .plan(
            structural(vec![create_cmd(SIBLING, NEW_SECTION, "Lamp")]),
            "tok-div-list-abs",
        )
        .await
        .commit;
    assert!(
        !fx.exists(&fx.children_key(SIBLING)).await,
        "fixture precondition: SIBLING holds no list"
    );
    fx.rpush(&fx.children_key(SIBLING), &[99]).await;

    fx.assert_refused(&commit, CommitOutcome::CacheDiverged)
        .await;
}

/// Children list, list digest on a `DCHL`: the cascade delete of `WORLD`
/// deletes `WORLD`'s list, which carries the list's digest, and an extra id
/// pushed onto the stored list is `CACHE_DIVERGED`. Every case above drives a
/// `CHL`. With the id popped again, the same commit lands: the extra id was
/// the only cause.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_deleted_children_list_edited_out_of_band_is_cache_diverged() {
    let fx = fixture("structural-div-dchl").await;
    let planned = fx
        .plan(structural(vec![delete_cmd(ROOT, WORLD)]), "tok-div-dchl")
        .await;
    assert_plan_carries(
        &planned,
        "DeleteChildrenList of WORLD",
        |m| matches!(m, StorageMutation::DeleteChildrenList { section_id } if *section_id == WORLD),
    );
    assert_eq!(
        count(&planned.commit.mutations, |m| {
            matches!(m, StorageMutation::SetChildren { section_id, .. } if *section_id == WORLD)
        }),
        0,
        "fixture precondition: no CHL of WORLD, so the DCHL is the record carrying \
         WORLD's list digest; got {:?}",
        planned.commit.mutations
    );
    let list = fx.children_key(WORLD);
    fx.rpush(&list, &[99]).await;

    fx.assert_refused(&planned.commit, CommitOutcome::CacheDiverged)
        .await;

    let popped: Option<String> = fx
        .tv
        .client()
        .pool
        .rpop(&list, None)
        .await
        .expect("RPOP ch must succeed");
    assert_eq!(popped.as_deref(), Some("99"));
    fx.commit_landed(&planned.commit).await;
}

/// Row existence, `DF`: the field row the cascade deletes is already gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_deleted_field_row_already_absent_is_cache_diverged() {
    let fx = fixture("structural-div-df").await;
    let commit = fx
        .plan(structural(vec![delete_cmd(WORLD, CUBE)]), "tok-div-df")
        .await
        .commit;
    fx.delete_key(&fx.field_key(CUBE, CUBE_FIELD)).await;

    fx.assert_refused(&commit, CommitOutcome::CacheDiverged)
        .await;
}

/// Row existence, `DT`: the time-sample row the cascade deletes is already
/// gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_deleted_time_sample_row_already_absent_is_cache_diverged() {
    let fx = fixture("structural-div-dt").await;
    let commit = fx
        .plan(structural(vec![delete_cmd(WORLD, CUBE)]), "tok-div-dt")
        .await
        .commit;
    fx.delete_key(&fx.time_sample_key(CUBE, CUBE_SAMPLE_TIME))
        .await;

    fx.assert_refused(&commit, CommitOutcome::CacheDiverged)
        .await;
}

/// The row a hand-built plan sets and then deletes in one commit. Its delete
/// carries the `absent` digest — the pre-plan state has no such row — and the
/// set carries none.
const GHOST_FIELD: &str = "ghost";
const GHOST_SAMPLE_TIME: f64 = 7.0;

/// Row existence, `DF`, the other direction: the cache holds no such field
/// row (`absent`), and the store has one. Hand-built: the engine deletes only
/// rows its state holds, so only a set-then-delete of one new row in a plan
/// carries `absent`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_deleted_field_row_the_cache_holds_absent_but_exists_is_cache_diverged() {
    let fx = fixture("structural-div-df-absent").await;
    let base = fx.plan(OwnedDelta::default(), "tok-div-df-absent").await;
    let commit = hand_built(
        &base,
        &base.pre_state,
        vec![
            StorageMutation::SetField {
                field: Field {
                    section_id: CUBE,
                    key_name: GHOST_FIELD.to_owned(),
                    value: FieldValue::Inline(vec![1]),
                    set_order: 1,
                    authoring_set_order: 1,
                    version: base.commit.expected_seq + 1,
                },
            },
            StorageMutation::DeleteField {
                section_id: CUBE,
                key_name: GHOST_FIELD.to_owned(),
            },
        ],
    );
    fx.plant_row(&fx.field_key(CUBE, GHOST_FIELD)).await;

    fx.assert_refused(&commit, CommitOutcome::CacheDiverged)
        .await;
}

/// The time-sample twin of
/// [`commit_structural_deleted_field_row_the_cache_holds_absent_but_exists_is_cache_diverged`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_deleted_time_sample_row_the_cache_holds_absent_but_exists_is_cache_diverged(
) {
    let fx = fixture("structural-div-dt-absent").await;
    let base = fx.plan(OwnedDelta::default(), "tok-div-dt-absent").await;
    let commit = hand_built(
        &base,
        &base.pre_state,
        vec![
            StorageMutation::SetTimeSample {
                sample: TimeSample {
                    section_id: CUBE,
                    time: OrderedFloat(GHOST_SAMPLE_TIME),
                    value: FieldValue::Inline(vec![2]),
                    version: base.commit.expected_seq + 1,
                },
            },
            StorageMutation::DeleteTimeSample {
                section_id: CUBE,
                time: OrderedFloat(GHOST_SAMPLE_TIME),
            },
        ],
    );
    fx.plant_row(&fx.time_sample_key(CUBE, GHOST_SAMPLE_TIME))
        .await;

    fx.assert_refused(&commit, CommitOutcome::CacheDiverged)
        .await;
}

/// A delta whose two mutations rewrite one section row — `CUBE` renamed twice
/// — commits. The second `SetSection`'s `previous` names the row the first
/// one wrote; a digest taken from it, rather than from the pre-plan state the
/// script's pre-pass reads, would refuse a commit nothing diverged from.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_section_renamed_twice_in_one_delta_commits() {
    let fx = fixture("structural-twice-rename").await;
    let planned = fx
        .plan(
            structural(vec![
                move_cmd(CUBE, WORLD, WORLD, "Cube1"),
                move_cmd(CUBE, WORLD, WORLD, "Cube2"),
            ]),
            "tok-twice-rename",
        )
        .await;
    assert_eq!(
        count(&planned.commit.mutations, |m| {
            matches!(m, StorageMutation::SetSection { section, .. } if section.section_id == CUBE)
        }),
        2,
        "fixture precondition: two SetSection records of CUBE; got {:?}",
        planned.commit.mutations
    );

    fx.commit_landed(&planned.commit).await;

    fx.assert_store_holds(&planned.post_state).await;
}

/// A delta that deletes `CUBE` and creates it again commits: the
/// `DeleteSection` checks the pre-plan row, and the `SetSection` of the same
/// row carries no check of its own. Its `previous` is `None` because an
/// earlier mutation removed the row; a digest taken from it ("absent") would
/// refuse a row the store still holds.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_section_deleted_then_recreated_in_one_delta_commits() {
    let fx = fixture("structural-twice-recreate").await;
    let planned = fx
        .plan(
            structural(vec![
                delete_cmd(WORLD, CUBE),
                create_cmd(WORLD, CUBE, "Cube"),
            ]),
            "tok-twice-recreate",
        )
        .await;
    let touches_cube: Vec<&str> = planned
        .commit
        .mutations
        .iter()
        .filter_map(|m| match m {
            StorageMutation::DeleteSection { section_id, .. } if *section_id == CUBE => {
                Some("DeleteSection")
            }
            StorageMutation::SetSection { section, .. } if section.section_id == CUBE => {
                Some("SetSection")
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        touches_cube,
        ["DeleteSection", "SetSection"],
        "fixture precondition: CUBE's row deleted, then written"
    );

    fx.commit_landed(&planned.commit).await;

    fx.assert_store_holds(&planned.post_state).await;
}

/// A plan that writes one parent's list twice commits. Hand-built: the engine
/// writes each parent's list once per delta, but the digest rule must not
/// depend on that. The second write's would-be belief (`[CUBE, 50]`, what the
/// first wrote) is not what the store holds before the commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_children_list_written_twice_in_one_plan_commits() {
    let fx = fixture("structural-twice-list").await;
    let base = fx.plan(OwnedDelta::default(), "tok-twice-list").await;
    let commit = hand_built(
        &base,
        &base.pre_state,
        vec![
            StorageMutation::SetChildren {
                section_id: WORLD,
                children: vec![CUBE, 50],
            },
            StorageMutation::SetChildren {
                section_id: WORLD,
                children: vec![CUBE],
            },
        ],
    );

    fx.commit_landed(&commit).await;

    assert_eq!(
        fx.list(fx.children_key(WORLD)).await,
        vec![CUBE.to_string()]
    );
}

/// The cross-language list digest: for Task 1's two fixed vectors, Rust's
/// `children_list_digest` is the fixed hex, and the script — which hashes the
/// stored list with `redis.sha1hex` — accepts it as the digest of that list.
/// Hand-built so the stored list is exactly the vector.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn children_list_digest_equals_the_scripts_sha1hex_of_the_stored_list() {
    const VECTORS: [(&str, &[u64], &str); 2] = [
        (
            "structural-digest-a",
            &[5, 12, 1000],
            "ed3e4ac759f1f0b121ae29a57350ea82a6dbb0c3",
        ),
        (
            "structural-digest-b",
            &[u64::MAX],
            "234524f46607504594696f875bd0ca86fe0ee671",
        ),
    ];
    for (layer, list, hex) in VECTORS {
        let fx = fixture(layer).await;
        assert_eq!(
            children_list_digest(Some(list)),
            hex,
            "Task 1's fixed vector"
        );
        let base = fx.plan(OwnedDelta::default(), "tok-digest").await;
        let mut believed = base.pre_state.clone();
        believed.children.insert(WORLD, list.to_vec());
        fx.delete_key(&fx.children_key(WORLD)).await;
        fx.rpush(&fx.children_key(WORLD), list).await;
        let commit = hand_built(
            &base,
            &believed,
            vec![StorageMutation::SetChildren {
                section_id: WORLD,
                children: vec![CUBE],
            }],
        );

        let (result, _) = fx.commit(&commit).await;

        assert!(
            matches!(result, Ok(CommitOutcome::Committed { .. })),
            "{list:?}: the script must accept Rust's digest of the stored list; got {result:?}"
        );
    }
}

/// The absent marker: Rust's digest of a missing list is the token the
/// script matches an absent key with, so a list created under a section that
/// had none commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn children_list_digest_of_an_absent_list_is_the_scripts_absent_token() {
    let fx = fixture("structural-digest-absent").await;
    assert_eq!(children_list_digest(None), DIGEST_ABSENT);
    assert_eq!(children_list_digest(Some(&[])), DIGEST_ABSENT);
    let planned = fx
        .plan(
            structural(vec![create_cmd(SIBLING, NEW_SECTION, "Lamp")]),
            "tok-absent",
        )
        .await;
    assert!(
        !fx.exists(&fx.children_key(SIBLING)).await,
        "fixture precondition: SIBLING holds no list"
    );

    fx.commit_landed(&planned.commit).await;

    assert_eq!(
        fx.list(fx.children_key(SIBLING)).await,
        vec![NEW_SECTION.to_string()]
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Refusals that are errors
// ═════════════════════════════════════════════════════════════════════════════

/// An `HDR` record whose `seq` is not `expected_seq + 1` is the script's
/// `ENCODING`, mapped to the fast path's encoder-bug `Internal`; the commit was
/// sent, did not land, and wrote nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_header_record_seq_not_expected_plus_one_is_encoding_internal() {
    let fx = fixture("structural-hdr-seq").await;
    let planned = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            "tok-hdr-seq",
        )
        .await;
    let mut mutations = planned.commit.mutations.clone();
    let Some(StorageMutation::SetObjectHeader { sequence_num, .. }) = mutations.last_mut() else {
        panic!("fixture precondition: the plan ends with its header bump");
    };
    *sequence_num += 1;
    let commit = StructuralCommit {
        pre_image_digests: PreImageDigests::from_plan(&mutations, &planned.pre_state),
        mutations,
        ..planned.commit.clone()
    };

    fx.assert_encoding_refused(&commit, "HDR seq").await;
}

/// A plan without a header record is refused `ENCODING` before any write.
/// Load-bearing: the header `HSET`, after the `XADD` and the apply pass, reads
/// the `HDR` record's columns, so without this refusal such a plan would fail
/// there with the commit partly applied. Hand-built: `apply_delta` always ends
/// a plan with its header bump.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_plan_without_a_header_record_is_encoding_internal() {
    let fx = fixture("structural-no-hdr").await;
    let base = fx.plan(OwnedDelta::default(), "tok-no-hdr").await;
    let commit = hand_built_exact(
        &base,
        &base.pre_state,
        vec![StorageMutation::SetChildren {
            section_id: WORLD,
            children: vec![CUBE, 50],
        }],
    );

    fx.assert_encoding_refused(&commit, "no HDR record").await;
}

/// A plan with two header records is refused `ENCODING` before any write: the
/// script keeps one `HDR` record to write, and a second would be either
/// silently dropped or silently preferred.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_plan_with_two_header_records_is_encoding_internal() {
    let fx = fixture("structural-two-hdr").await;
    let base = fx.plan(OwnedDelta::default(), "tok-two-hdr").await;
    let commit = hand_built_exact(
        &base,
        &base.pre_state,
        vec![
            header_bump(&base, &base.pre_state),
            header_bump(&base, &base.pre_state),
        ],
    );

    fx.assert_encoding_refused(&commit, "second HDR record")
        .await;
}

/// Digests built from another plan of the same shape are refused `Internal`
/// before anything is sent: no observer call, nothing written. (Sent, they
/// would check the wrong keys' beliefs — here they would even commit.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_digests_from_another_plan_are_refused_before_sending() {
    let fx = fixture("structural-foreign-digests").await;
    let planned = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            "tok-foreign",
        )
        .await;
    let other = fx
        .plan(
            structural(vec![create_cmd(WORLD, OTHER_NEW_SECTION, "Desk")]),
            "tok-other",
        )
        .await;
    let commit = StructuralCommit {
        pre_image_digests: other.commit.pre_image_digests,
        ..planned.commit
    };
    let before = fx.keyspace().await;

    let (result, calls) = fx.commit(&commit).await;

    match result {
        Err(ServiceError::Internal(msg)) => assert!(
            msg.starts_with("structural encode: the pre-image digests were not built from this"),
            "got {msg:?}"
        ),
        other => panic!("a mismatched plan / digests pair must be refused Internal; got {other:?}"),
    }
    assert_eq!(calls, Vec::new(), "nothing is sent, so no observer call");
    assert_eq!(fx.keyspace().await, before);
}

// ═════════════════════════════════════════════════════════════════════════════
// Raw structural-mode invocation: the parse pass
// ═════════════════════════════════════════════════════════════════════════════
//
// The parse pass refuses malformed input `ENCODING` before any write. Most of
// that input cannot be produced through `commit_structural`, whose encoder
// only emits well-formed records, so these cases corrupt one slot of a raw
// invocation instead. `structural_invocation_uncorrupted_commits_its_plan` is
// the companion that keeps them honest: an uncorrupted invocation commits, so
// a mis-built prefix cannot make every corruption case pass for the wrong
// reason.

/// ARGV index (0-based) of `num_mutations` in the shared prefix.
pub(crate) const STRUCTURAL_ARGV_NUM_MUTATIONS: usize = UPDATE_SCRIPT_ARG_NUM_MUTATIONS - 1;
/// ARGV index (0-based) of the mode slot, right after the shared prefix.
pub(crate) const STRUCTURAL_ARGV_MODE: usize = UPDATE_SCRIPT_SHARED_ARGV;
/// ARGV index (0-based) of `expected_seq`.
pub(crate) const STRUCTURAL_ARGV_EXPECTED_SEQ: usize = STRUCTURAL_ARGV_MODE + 1;
/// ARGV index (0-based) of `expected_commit_chain`.
pub(crate) const STRUCTURAL_ARGV_EXPECTED_COMMIT_CHAIN: usize = STRUCTURAL_ARGV_MODE + 2;
/// ARGV index (0-based) of the slot after `expected_commit_chain`: the
/// untouched-row count block's `K`, followed by its `K` `hash, count` pairs
/// and then the structural records
/// ([`StructuralInvocation::records_start`]).
pub(crate) const STRUCTURAL_ARGV_UNTOUCHED_BLOCK: usize = STRUCTURAL_ARGV_MODE + 3;
/// KEYS index (0-based) of the first record's key: past the shared prefix and
/// the sections, deleted-sections and children-list indexes.
pub(crate) const STRUCTURAL_KEYS_RECORDS_START: usize = UPDATE_SCRIPT_SHARED_KEYS + 3;

/// The start-record slot for a call that read `start` at call start, as the
/// script's contract spells it: `"="` for an absent field, `"="` followed by
/// the record otherwise.
pub(crate) fn start_record_arg(start: &StartRecord) -> String {
    match start {
        StartRecord::Absent => "=".to_owned(),
        StartRecord::Record(record) => format!("={record}"),
    }
}

/// The writer-guard slot for a chained call whose predecessor is
/// `previous_operation_id`, as the script's contract spells it: `"^"` for a
/// chain's first write (empty), `"^op."` followed by the predecessor
/// otherwise.
pub(crate) fn predecessor_arg(previous_operation_id: &str) -> String {
    if previous_operation_id.is_empty() {
        "^".to_owned()
    } else {
        format!("^op.{previous_operation_id}")
    }
}

/// The three commit-identity slots (`ARGV[UPDATE_SCRIPT_ARG_COMMIT_KEY]`,
/// `[UPDATE_SCRIPT_ARG_RECORD_SUFFIX]`, `[UPDATE_SCRIPT_ARG_WRITER_GUARD]`)
/// for `commit`, rebuilt
/// from the script's contract rather than through `CommitIdentity`: without
/// an operation, the commit token and two empty slots (the b.75i shape);
/// with one, `op.<operation_id>`, `:<token>:<request_digest>` and the
/// operation's writer guard — an unchained operation's start record
/// ([`start_record_arg`]), a chained one's predecessor ([`predecessor_arg`]),
/// or `""` when it has none, which the script refuses `ENCODING`.
pub(crate) fn commit_identity_argv(commit: &StructuralCommit) -> [String; 3] {
    match &commit.operation {
        None => [commit.commit_token.clone(), String::new(), String::new()],
        Some(op) => [
            op.key().to_owned(),
            format!(":{}:{}", commit.commit_token, op.digest()),
            match op.guard() {
                Some(WriterGuard::Start(start)) => start_record_arg(start),
                Some(WriterGuard::Predecessor(previous)) => predecessor_arg(previous),
                None => String::new(),
            },
        ],
    }
}

/// The commit release ring TTL a store built with [`LayerStore::new`] sends
/// (`ARGV[UPDATE_SCRIPT_ARG_RING_TTL_MS]`): [`commit_release_ring_ttl`] of
/// the default RPC deadline and Valkey command timeout, 95,000 ms.
pub(crate) fn default_ring_ttl_ms() -> u64 {
    commit_release_ring_ttl(RPC_DEADLINE_MS_DEFAULT, VALKEY_COMMAND_TIMEOUT_MS_DEFAULT)
}

/// The shared ARGV prefix of a raw run carrying `commit`'s writer and
/// identity ([`commit_identity_argv`], and its token as the call token), with
/// no relay bytes, a store's default relay byte budget and a store's default
/// commit release ring TTL ([`default_ring_ttl_ms`]): every slot placed
/// through its `UPDATE_SCRIPT_ARG_*` constant, as `UpdateScriptPrefix::build`
/// places it, so a slot that moves on both sides moves here too and a slot
/// added to the prefix without a value here fails.
pub(crate) fn shared_argv(
    commit: &StructuralCommit,
    stream_maxlen: u64,
    num_mutations: u64,
) -> Vec<String> {
    let [commit_key, record_suffix, start_record] = commit_identity_argv(commit);
    // The relay byte budget's B, T, K and MIN: a store's default
    // (`RelayRetention::from_budget` of the default budget).
    let retention = RelayRetention::from_budget(LAYER_STREAM_MAX_BYTES_DEFAULT);
    let mut argv: Vec<Option<String>> = vec![None; UPDATE_SCRIPT_SHARED_ARGV];
    for (slot, value) in [
        (UPDATE_SCRIPT_ARG_SERVER_TS, now_micros().to_string()),
        (UPDATE_SCRIPT_ARG_CLIENT_ID, commit.client_id.clone()),
        (UPDATE_SCRIPT_ARG_CLIENT_TS, commit.client_ts.to_string()),
        (UPDATE_SCRIPT_ARG_STREAM_MAXLEN, stream_maxlen.to_string()),
        (UPDATE_SCRIPT_ARG_OBJECT_ID, commit.object_id.to_string()),
        (UPDATE_SCRIPT_ARG_NUM_MUTATIONS, num_mutations.to_string()),
        (UPDATE_SCRIPT_ARG_RELAY_OMPF, String::new()),
        (UPDATE_SCRIPT_ARG_COMMIT_KEY, commit_key),
        (UPDATE_SCRIPT_ARG_RECORD_SUFFIX, record_suffix),
        (UPDATE_SCRIPT_ARG_WRITER_GUARD, start_record),
        (
            UPDATE_SCRIPT_ARG_RELAY_BUDGET_BYTES,
            retention.budget_bytes.to_string(),
        ),
        (
            UPDATE_SCRIPT_ARG_RELAY_GENERATION_BYTES,
            retention.generation_bytes.to_string(),
        ),
        (
            UPDATE_SCRIPT_ARG_RELAY_GENERATIONS,
            retention.generations.to_string(),
        ),
        (
            UPDATE_SCRIPT_ARG_RELAY_MIN_ENTRIES,
            retention.min_entries.to_string(),
        ),
        // The call token, the key of the commit's release ring entry: the
        // commit's token, as `commit_structural` passes it
        // (`CommitIdentity::from_recorded_token`).
        (UPDATE_SCRIPT_ARG_CALL_TOKEN, commit.commit_token.clone()),
        // The commit release ring's TTL, as a store built with
        // `LayerStore::new` sends it.
        (
            UPDATE_SCRIPT_ARG_RING_TTL_MS,
            default_ring_ttl_ms().to_string(),
        ),
    ] {
        assert!(
            argv[slot - 1].replace(value).is_none(),
            "shared ARGV slot {slot} filled twice"
        );
    }
    argv.into_iter()
        .enumerate()
        .map(|(i, v)| v.unwrap_or_else(|| panic!("shared ARGV slot {} left unset", i + 1)))
        .collect()
}

/// Everything `commit_structural` hands to `UPDATE_LAYER_FAST_SCRIPT` in
/// structural mode, rebuilt from the crate's public surface, for tests that
/// need a record the encoder never emits (a corrupt slot, a value past a
/// ceiling). The composition mirrors `commit_structural`'s and the script
/// header's structural-mode layout; the one deliberate difference, as in
/// `update_layer_lua_test`'s `FastPathInvocation`, is that `relay_ompf` is
/// always the empty string ("no relay bytes"), so every ARGV element is a
/// UTF-8 string and a corruption is a one-slot string swap.
pub(crate) struct StructuralInvocation {
    pub(crate) keys: Vec<String>,
    pub(crate) argv: Vec<String>,
}

impl StructuralInvocation {
    /// The invocation `commit_structural` would send for `commit` (without its
    /// relay OMPF), with a fresh server timestamp and `stream_maxlen`, and the
    /// commit identity [`commit_identity_argv`] spells for its token and
    /// operation.
    pub(crate) fn from_commit(
        tv: &TestValkey,
        commit: &StructuralCommit,
        stream_maxlen: u64,
    ) -> Self {
        let schema = tv.key_schema();
        let oid = commit.object_id;
        let o = oid.to_string();
        let encoded = encode_mutations(
            &commit.mutations,
            &commit.pre_image_digests,
            commit.next_set_order,
            schema,
            &o,
        )
        .expect("the commit's plan encodes");

        // The shared prefix through the one builder both modes use
        // (`lua::update_script_shared_keys`: header … commit-token record,
        // snapshot fence, writer intents, commit release record), then
        // structural mode's three object-level indexes at
        // `KEYS[SHARED_KEYS + 1..=SHARED_KEYS + 3]`.
        let mut keys = update_script_shared_keys(schema, oid, 3 + encoded.mutation_keys.len());
        keys.extend([
            schema.obj_sections_index(&o),
            schema.obj_deleted_sections_index(&o),
            schema.obj_children_index(&o),
        ]);
        assert_eq!(keys.len(), STRUCTURAL_KEYS_RECORDS_START);
        keys.extend(encoded.mutation_keys);

        let mut argv = shared_argv(commit, stream_maxlen, encoded.num_mutations);
        argv.extend([
            UpdateScriptMode::Structural.as_str().to_owned(),
            commit.expected_seq.to_string(),
            commit.expected_commit_chain.clone(),
        ]);
        assert_eq!(argv.len(), STRUCTURAL_ARGV_UNTOUCHED_BLOCK);
        argv.extend(encoded.untouched_ext_block);
        argv.extend(encoded.mutation_argv);

        Self { keys, argv }
    }

    /// ARGV index of the first structural record: past the untouched-row
    /// count block, whose `K` slot must still be the encoder's decimal.
    pub(crate) fn records_start(&self) -> usize {
        let k: usize = self.argv[STRUCTURAL_ARGV_UNTOUCHED_BLOCK]
            .parse()
            .expect("the untouched-row count block's K is a decimal");
        STRUCTURAL_ARGV_UNTOUCHED_BLOCK + 1 + 2 * k
    }

    /// ARGV index of the first record tagged `tag`. Panics when there is none,
    /// so a fixture whose plan lost the record fails here rather than
    /// corrupting a neighbouring slot.
    pub(crate) fn record_slot(&self, tag: &str) -> usize {
        let start = self.records_start();
        self.argv
            .iter()
            .skip(start)
            .position(|slot| slot == tag)
            .map(|i| i + start)
            .unwrap_or_else(|| panic!("no {tag} record in ARGV {:?}", self.argv))
    }

    /// ARGV index of the untouched-row count block's `entry`-th (1-based)
    /// hash; its count is the slot after.
    fn untouched_entry_slot(entry: usize) -> usize {
        STRUCTURAL_ARGV_UNTOUCHED_BLOCK + 2 * entry - 1
    }

    /// Add `delta` to the `num_mutations` slot.
    pub(crate) fn shift_num_mutations(&mut self, delta: i64) {
        let slot = &mut self.argv[STRUCTURAL_ARGV_NUM_MUTATIONS];
        let n: i64 = slot.parse().expect("num_mutations is a decimal");
        *slot = (n + delta).to_string();
    }

    /// Send it and return the script's reply undecoded.
    pub(crate) async fn eval(&self, tv: &TestValkey) -> Result<Vec<String>, fred::error::Error> {
        let argv: Vec<fred::types::Value> = self
            .argv
            .iter()
            .map(|a| fred::types::Value::from(a.clone()))
            .collect();
        UPDATE_LAYER_FAST_SCRIPT
            .evalsha_with_reload::<Vec<String>, _, _>(
                tv.client().pool.next(),
                self.keys.clone(),
                argv,
            )
            .await
    }
}

/// The uncorrupted invocation of a plan commits it: the reply's sequence is
/// `expected_seq + 1` and the object read back is the plan's post-state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_invocation_uncorrupted_commits_its_plan() {
    let fx = fixture("structural-raw-commit").await;
    let planned = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            "tok-raw-commit",
        )
        .await;
    let invocation =
        StructuralInvocation::from_commit(&fx.tv, &planned.commit, fx.ls.layer_stream_maxlen());

    let reply = invocation.eval(&fx.tv).await;

    assert_eq!(
        reply.expect("the uncorrupted invocation must commit"),
        vec![
            (planned.commit.expected_seq + 1).to_string(),
            "drift:0:0".to_owned()
        ]
    );
    fx.assert_store_holds(&planned.post_state).await;
}

/// Which planned commit a parse-pass case corrupts.
#[derive(Debug, Clone, Copy)]
enum ParsePlan {
    /// `SEC`, a `CHL` of two ids under `WORLD`, `HDR`.
    CreateSection,
    /// `DSEC`, `TOMB`, `DCHL`, `DF`, `DT`, an emptying `CHL`, `HDR`; its
    /// untouched-row count block carries two entries, `0xAB…` then `0xCD…`
    /// (the hashes of `CUBE`'s field and sample).
    DeleteCube,
    /// One inline `SFA` record and `HDR`.
    Values,
}

/// One parse-pass refusal: a corruption of a well-formed invocation and the
/// `ENCODING` message fragment it must answer.
struct ParseCase {
    name: &'static str,
    plan: ParsePlan,
    corrupt: fn(&mut StructuralInvocation),
    refusal: &'static str,
}

/// Point a record's slot at `value`: the slot `offset` past `tag`'s record
/// tag.
fn set_record_slot(inv: &mut StructuralInvocation, tag: &str, offset: usize, value: &str) {
    let slot = inv.record_slot(tag) + offset;
    inv.argv[slot] = value.to_owned();
}

/// Every parse-pass refusal of the script's structural mode (the record
/// vocabulary's slot offsets are its header's "Structural-mode records"). The
/// `HDR` seq refusal is pinned through `commit_structural` above.
const PARSE_CASES: &[ParseCase] = &[
    ParseCase {
        // Checked with the shared prefix, before the commit-token record is
        // read: an empty call token would key every token-less call's commit
        // release ring entry alike.
        name: "empty call token",
        plan: ParsePlan::Values,
        corrupt: |inv| inv.argv[UPDATE_SCRIPT_ARG_CALL_TOKEN - 1] = String::new(),
        refusal: "missing call token (ARGV[15])",
    },
    // The commit release ring's TTL, checked with the shared prefix on every
    // run whatever the commit owes: 1 to 12 decimal digits with no leading
    // zero, so step 7a's `PEXPIRE` after the `XADD` can never be refused.
    // `ParsePlan::Values` owes no release, so the refusal cannot come from the
    // ring write.
    ParseCase {
        name: "empty commit release ring TTL",
        plan: ParsePlan::Values,
        corrupt: |inv| inv.argv[UPDATE_SCRIPT_ARG_RING_TTL_MS - 1] = String::new(),
        refusal: "bad commit release ring TTL (ARGV[16])",
    },
    ParseCase {
        name: "zero commit release ring TTL",
        plan: ParsePlan::Values,
        corrupt: |inv| inv.argv[UPDATE_SCRIPT_ARG_RING_TTL_MS - 1] = "0".to_owned(),
        refusal: "bad commit release ring TTL (ARGV[16])",
    },
    ParseCase {
        name: "commit release ring TTL with a leading zero",
        plan: ParsePlan::Values,
        corrupt: |inv| inv.argv[UPDATE_SCRIPT_ARG_RING_TTL_MS - 1] = "095000".to_owned(),
        refusal: "bad commit release ring TTL (ARGV[16])",
    },
    ParseCase {
        name: "commit release ring TTL with a non-digit",
        plan: ParsePlan::Values,
        corrupt: |inv| inv.argv[UPDATE_SCRIPT_ARG_RING_TTL_MS - 1] = "95e3".to_owned(),
        refusal: "bad commit release ring TTL (ARGV[16])",
    },
    ParseCase {
        name: "commit release ring TTL of 13 digits",
        plan: ParsePlan::Values,
        corrupt: |inv| inv.argv[UPDATE_SCRIPT_ARG_RING_TTL_MS - 1] = "1000000000000".to_owned(),
        refusal: "bad commit release ring TTL (ARGV[16])",
    },
    ParseCase {
        name: "unknown mode",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| inv.argv[STRUCTURAL_ARGV_MODE] = "bogus".to_owned(),
        refusal: "unknown mode bogus",
    },
    ParseCase {
        name: "malformed expected_seq",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| inv.argv[STRUCTURAL_ARGV_EXPECTED_SEQ].push('x'),
        refusal: "malformed expected_seq",
    },
    ParseCase {
        name: "missing expected_seq",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| inv.argv.truncate(STRUCTURAL_ARGV_EXPECTED_SEQ),
        refusal: "malformed expected_seq",
    },
    ParseCase {
        name: "missing expected_commit_chain",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| inv.argv.truncate(STRUCTURAL_ARGV_EXPECTED_COMMIT_CHAIN),
        refusal: "missing expected_commit_chain",
    },
    ParseCase {
        name: "missing untouched-row count block",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| inv.argv.truncate(STRUCTURAL_ARGV_UNTOUCHED_BLOCK),
        refusal: "malformed untouched-row count block size",
    },
    ParseCase {
        name: "malformed untouched-row count block size",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| inv.argv[STRUCTURAL_ARGV_UNTOUCHED_BLOCK] = "x".to_owned(),
        refusal: "malformed untouched-row count block size",
    },
    ParseCase {
        name: "untouched-row count block one entry past the end of ARGV",
        plan: ParsePlan::DeleteCube,
        // The slots after `K` hold `remaining / 2` whole entries; one more
        // runs past the end.
        corrupt: |inv| {
            let remaining = inv.argv.len() - STRUCTURAL_ARGV_UNTOUCHED_BLOCK - 1;
            inv.argv[STRUCTURAL_ARGV_UNTOUCHED_BLOCK] = (remaining / 2 + 1).to_string();
        },
        refusal: "entries runs past the end of ARGV",
    },
    ParseCase {
        name: "untouched-row hash one character short",
        plan: ParsePlan::DeleteCube,
        corrupt: |inv| {
            inv.argv[StructuralInvocation::untouched_entry_slot(1)].pop();
        },
        refusal: "malformed hash in untouched-row count entry 1",
    },
    ParseCase {
        name: "uppercase untouched-row hash",
        plan: ParsePlan::DeleteCube,
        corrupt: |inv| {
            let slot = StructuralInvocation::untouched_entry_slot(2);
            inv.argv[slot] = inv.argv[slot].to_uppercase();
        },
        refusal: "malformed hash in untouched-row count entry 2",
    },
    ParseCase {
        name: "non-hex untouched-row hash",
        plan: ParsePlan::DeleteCube,
        corrupt: |inv| inv.argv[StructuralInvocation::untouched_entry_slot(1)] = "g".repeat(64),
        refusal: "malformed hash in untouched-row count entry 1",
    },
    ParseCase {
        name: "malformed untouched-row count",
        plan: ParsePlan::DeleteCube,
        corrupt: |inv| inv.argv[StructuralInvocation::untouched_entry_slot(1) + 1] = "x".to_owned(),
        refusal: "malformed count in untouched-row count entry 1",
    },
    ParseCase {
        name: "untouched-row count past the f64 integer ceiling",
        plan: ParsePlan::DeleteCube,
        // 2^53, one past the ceiling.
        corrupt: |inv| {
            inv.argv[StructuralInvocation::untouched_entry_slot(2) + 1] =
                "9007199254740992".to_owned();
        },
        refusal: "malformed count in untouched-row count entry 2",
    },
    ParseCase {
        name: "untouched-row hashes descending",
        plan: ParsePlan::DeleteCube,
        corrupt: |inv| {
            let first = StructuralInvocation::untouched_entry_slot(1);
            let second = StructuralInvocation::untouched_entry_slot(2);
            inv.argv.swap(first, second);
            inv.argv.swap(first + 1, second + 1);
        },
        refusal: "hash not strictly ascending in untouched-row count entry 2",
    },
    ParseCase {
        name: "duplicate untouched-row hash",
        plan: ParsePlan::DeleteCube,
        corrupt: |inv| {
            let first = StructuralInvocation::untouched_entry_slot(1);
            inv.argv[StructuralInvocation::untouched_entry_slot(2)] = inv.argv[first].clone();
        },
        refusal: "hash not strictly ascending in untouched-row count entry 2",
    },
    ParseCase {
        name: "malformed num_mutations",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| inv.argv[STRUCTURAL_ARGV_NUM_MUTATIONS] = "x".to_owned(),
        refusal: "malformed num_mutations",
    },
    // The KEYS count. CreateSection's plan has three records, so each case
    // below sends one key too few or too many for them.
    ParseCase {
        name: "one KEY short",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| {
            inv.keys.pop();
        },
        refusal: "KEYS for 3 mutation record(s)",
    },
    ParseCase {
        name: "one KEY extra, after the last record's",
        plan: ParsePlan::CreateSection,
        // A key of the object's own hash tag, so a cluster routes it with the
        // rest; every record still reads its own key.
        corrupt: |inv| {
            let extra = inv.keys[STRUCTURAL_KEYS_RECORDS_START].clone();
            inv.keys.push(extra);
        },
        refusal: "KEYS for 3 mutation record(s)",
    },
    ParseCase {
        name: "one KEY extra, before the first record's",
        plan: ParsePlan::CreateSection,
        // A copy of record 1's key ahead of it: records 2 and 3 shift onto
        // their predecessors' keys.
        corrupt: |inv| {
            let extra = inv.keys[STRUCTURAL_KEYS_RECORDS_START].clone();
            inv.keys.insert(STRUCTURAL_KEYS_RECORDS_START, extra);
        },
        refusal: "KEYS for 3 mutation record(s)",
    },
    ParseCase {
        name: "a fast-mode tag",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| set_record_slot(inv, "SEC", 0, "SF"),
        refusal: "unknown mutation tag SF",
    },
    ParseCase {
        name: "truncated record",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| {
            inv.argv.pop();
        },
        refusal: "truncated HDR record",
    },
    ParseCase {
        name: "malformed children count",
        plan: ParsePlan::CreateSection,
        // CHL: tag, parent, n, ids…, digest.
        corrupt: |inv| set_record_slot(inv, "CHL", 2, "x"),
        refusal: "malformed children count",
    },
    ParseCase {
        name: "unknown SEC digest",
        plan: ParsePlan::CreateSection,
        // SEC: tag, sid, six columns, digest.
        corrupt: |inv| set_record_slot(inv, "SEC", 8, "bogus"),
        refusal: "unknown section digest bogus",
    },
    ParseCase {
        name: "unknown DSEC digest",
        plan: ParsePlan::DeleteCube,
        // DSEC: tag, sid, digest.
        corrupt: |inv| set_record_slot(inv, "DSEC", 2, "bogus"),
        refusal: "unknown section digest bogus",
    },
    ParseCase {
        name: "unknown CHL digest",
        plan: ParsePlan::CreateSection,
        // CHL WORLD [CUBE, NEW_SECTION]: the digest is past the two ids.
        corrupt: |inv| set_record_slot(inv, "CHL", 5, "bogus"),
        refusal: "unknown children-list digest bogus",
    },
    ParseCase {
        name: "unknown DCHL digest",
        plan: ParsePlan::DeleteCube,
        // DCHL: tag, parent, digest.
        corrupt: |inv| set_record_slot(inv, "DCHL", 2, "bogus"),
        refusal: "unknown children-list digest bogus",
    },
    ParseCase {
        name: "unknown DF digest",
        plan: ParsePlan::DeleteCube,
        // DF: tag, member, digest.
        corrupt: |inv| set_record_slot(inv, "DF", 2, "bogus"),
        refusal: "unknown row digest bogus",
    },
    ParseCase {
        name: "unknown DT digest",
        plan: ParsePlan::DeleteCube,
        corrupt: |inv| set_record_slot(inv, "DT", 2, "bogus"),
        refusal: "unknown row digest bogus",
    },
    ParseCase {
        name: "malformed value-hfield count",
        plan: ParsePlan::Values,
        // An inline SFA: tag, 1, "value", <value>, set_order,
        // authoring_set_order, version, member.
        corrupt: |inv| set_record_slot(inv, "SFA", 1, "x"),
        refusal: "malformed value-hfield count in mutation record 1",
    },
    ParseCase {
        name: "malformed set_order",
        plan: ParsePlan::Values,
        corrupt: |inv| set_record_slot(inv, "SFA", 4, "x"),
        refusal: "malformed set_order",
    },
    ParseCase {
        name: "malformed version",
        plan: ParsePlan::Values,
        corrupt: |inv| set_record_slot(inv, "SFA", 6, "x"),
        refusal: "malformed version",
    },
    ParseCase {
        name: "HDR key that is not the header key",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| {
            let other = inv.keys[STRUCTURAL_KEYS_RECORDS_START].clone();
            *inv.keys.last_mut().expect("the HDR key") = other;
        },
        refusal: "is not the header key",
    },
    ParseCase {
        name: "malformed HDR next_set_order",
        plan: ParsePlan::CreateSection,
        // HDR: tag, seq, first_ver, next_set_order, has_src_fmt[, src_fmt].
        corrupt: |inv| set_record_slot(inv, "HDR", 3, "x"),
        refusal: "malformed HDR next_set_order",
    },
    ParseCase {
        name: "malformed has_src_fmt",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| set_record_slot(inv, "HDR", 4, "2"),
        refusal: "malformed has_src_fmt 2",
    },
    ParseCase {
        name: "second HDR",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| {
            // The HDR record is the plan's last, so it runs to the end of ARGV.
            let hdr = inv.argv[inv.record_slot("HDR")..].to_vec();
            inv.argv.extend(hdr);
            let header_key = inv.keys.last().expect("the HDR key").clone();
            inv.keys.push(header_key);
            inv.shift_num_mutations(1);
        },
        refusal: "second HDR record",
    },
    ParseCase {
        name: "no HDR",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| {
            inv.argv.truncate(inv.record_slot("HDR"));
            inv.keys.pop();
            inv.shift_num_mutations(-1);
        },
        refusal: "no HDR record",
    },
    ParseCase {
        name: "a slot past the last record",
        plan: ParsePlan::CreateSection,
        corrupt: |inv| inv.argv.push("x".to_owned()),
        refusal: "1 ARGV slot(s) past the last mutation record",
    },
];

/// The planned commit a parse-pass case corrupts.
async fn parse_plan(fx: &Fixture, plan: ParsePlan) -> StructuralCommit {
    let delta = match plan {
        ParsePlan::CreateSection => structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
        ParsePlan::DeleteCube => structural(vec![delete_cmd(WORLD, CUBE)]),
        ParsePlan::Values => delta_with_set_field_inline(WORLD, WORLD_FIELD, vec![0x42], 0),
    };
    fx.plan(delta, "tok-parse").await.commit
}

/// Each parse-pass case answers its `ENCODING` refusal and leaves the
/// keyspace unchanged. Each case gets its own object, so one that wrongly
/// writes cannot move another's cut or baseline; every case runs, and the
/// failures are reported together.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_invocation_malformed_input_is_refused_encoding_before_any_write() {
    let mut failures = Vec::new();
    for case in PARSE_CASES {
        let fx = fixture("structural-raw-parse").await;
        let commit = parse_plan(&fx, case.plan).await;
        let mut invocation =
            StructuralInvocation::from_commit(&fx.tv, &commit, fx.ls.layer_stream_maxlen());
        (case.corrupt)(&mut invocation);
        let before = fx.keyspace().await;

        let reply = invocation.eval(&fx.tv).await;

        match reply {
            Err(e) if e.details().starts_with("ENCODING") && e.details().contains(case.refusal) => {
            }
            Err(e) => failures.push(format!(
                "{}: expected the ENCODING refusal naming {:?}; got {:?}",
                case.name,
                case.refusal,
                e.details()
            )),
            Ok(reply) => failures.push(format!(
                "{}: the script must refuse; got {reply:?}",
                case.name
            )),
        }
        if fx.keyspace().await != before {
            failures.push(format!("{}: the refusal wrote to the keyspace", case.name));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ═════════════════════════════════════════════════════════════════════════════
// Already landed: one token check and one decode for both modes
// ═════════════════════════════════════════════════════════════════════════════

/// The token in `client_id`'s commit-token record and the record itself.
pub(crate) async fn recorded_token(fx: &Fixture, client_id: &str) -> (String, String) {
    let record: Option<String> = fx
        .tv
        .client()
        .pool
        .hget(fx.keys().obj_write_token(&fx.oid_str()), client_id)
        .await
        .expect("HGET obj:wtok must succeed");
    let record = record.expect("a landed commit records its token");
    let token = record
        .split(':')
        .next()
        .expect("a record starts with its token")
        .to_owned();
    (token, record)
}

/// A structural commit whose token a fast-mode commit recorded answers
/// `AlreadyLanded` with that record's `seq` and `server_ts` — and the record
/// verbatim — and writes nothing. The plan is made after the fast commit, at
/// the current `seq`, so without the shared token check it would commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_token_recorded_by_a_fast_mode_commit_is_already_landed() {
    let fx = fixture("structural-landed-fast").await;
    let (fast_seq, fast_ts) = fx
        .ls
        .update_layer_relayed(
            &fx.session_id,
            fx.layer,
            fx.oid,
            Some(delta_with_set_field_inline(
                WORLD,
                WORLD_FIELD,
                vec![0x42],
                0,
            )),
            WRITER,
            CLIENT_TS,
        )
        .await
        .expect("the fast-mode commit must land");
    let (token, record) = recorded_token(&fx, WRITER).await;
    let commit = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            &token,
        )
        .await
        .commit;
    assert_eq!(
        commit.expected_seq, fast_seq,
        "fixture precondition: planned after it"
    );
    let before = fx.keyspace().await;

    let (result, calls) = fx.commit(&commit).await;

    match result {
        Ok(outcome) => assert_eq!(
            outcome,
            CommitOutcome::AlreadyLanded {
                seq: fast_seq,
                server_ts: fast_ts,
                record,
            }
        ),
        Err(e) => panic!("expected AlreadyLanded; got {e:?}"),
    }
    assert!(
        matches!(
            calls.as_slice(),
            [ObserverCall::CommitSize { .. }, ObserverCall::AlreadyLanded { seq, server_ts }]
                if *seq == fast_seq && *server_ts == fast_ts
        ),
        "got {calls:?}"
    );
    assert_eq!(
        fx.keyspace().await,
        before,
        "an already-landed answer writes nothing"
    );
}

/// The same structural commit sent twice: the second answers the first's
/// commit, calling `commit_size` then `already_landed` (and never `landed`).
/// Its fence would be `Stale` — the first commit moved the `seq` — so this
/// also shows the token check runs ahead of the fence.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_resent_commit_calls_commit_size_then_already_landed() {
    let fx = fixture("structural-landed-resend").await;
    let commit = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            "tok-resend",
        )
        .await
        .commit;
    let server_ts = fx.commit_landed(&commit).await;
    let seq = commit.expected_seq + 1;
    let before = fx.keyspace().await;

    let (result, calls) = fx.commit(&commit).await;

    assert!(
        matches!(
            &result,
            Ok(CommitOutcome::AlreadyLanded { seq: s, server_ts: t, .. }) if *s == seq && *t == server_ts
        ),
        "the resend must report the first commit; got {result:?}"
    );
    assert!(
        matches!(
            calls.as_slice(),
            [ObserverCall::CommitSize { .. }, ObserverCall::AlreadyLanded { seq: s, server_ts: t }]
                if *s == seq && *t == server_ts
        ),
        "got {calls:?}"
    );
    assert_eq!(fx.keyspace().await, before);
}

// ═════════════════════════════════════════════════════════════════════════════
// Client operation ids (plan b.v6d revision 3, structural route)
// ═════════════════════════════════════════════════════════════════════════════

/// The client `operation_id` the operation cases carry.
pub(crate) const OP_ID: &str = "structural-op-1";
/// A second, unrelated `operation_id`: the client's next write.
pub(crate) const OTHER_OP_ID: &str = "structural-op-2";
/// A call's commit token in its recorded form (32 lowercase hex digits, as
/// `update_layer` mints them).
pub(crate) const OP_CALL_TOKEN: &str = "0123456789abcdef0123456789abcdef";
/// A different call's commit token.
pub(crate) const OTHER_CALL_TOKEN: &str = "fedcba9876543210fedcba9876543210";
/// The stand-in request bytes an operation's digest is computed over.
pub(crate) const OP_PAYLOAD: &[u8] = b"structural operation payload";
/// Different request bytes: the same id reused for another request.
pub(crate) const OTHER_PAYLOAD: &[u8] = b"structural operation payload, edited";

/// The operation `operation_id` for a request of `payload` to `layer`, with
/// the start record its call read at call start.
pub(crate) fn operation(
    operation_id: &str,
    layer: &str,
    payload: &[u8],
    start: StartRecord,
) -> Operation {
    Operation::new(operation_id, operation_request_digest(layer, payload)).with_start_record(start)
}

/// `commit` re-sent as another call: `token`, and `operation`.
fn as_call(commit: &StructuralCommit, token: &str, operation: Operation) -> StructuralCommit {
    StructuralCommit {
        commit_token: token.to_owned(),
        operation: Some(operation),
        ..commit.clone()
    }
}

/// With a client operation, a structural commit lands the operation record
/// and every later run under that id is answered from it, by whose record it
/// is (the shared parser's verdict), ahead of the fence — each re-send below
/// carries the original plan, whose `expected_seq` the landed commit has
/// since moved, so without the step-1b answer each would be `Stale`:
///
/// - the same call again (its token; its start record, read before its own
///   commit, now stale) → `AlreadyLanded` with the record verbatim, the
///   observer's `already_landed` called;
/// - another call with the same request (a client re-send that read the
///   record at its start) → `Recorded(EarlierCall)` at the landed `seq` and
///   `server_ts`;
/// - another call reusing the id for a different request →
///   `Recorded(Conflict)`;
/// - another operation whose start record no longer holds (the client's
///   earlier write it raced) → `Overtaken`.
///
/// None of the last three is this call's landing, so the observer records
/// `commit_size` alone; no answer writes anything or enqueues a release (the
/// pending set is part of the compared keyspace): `commit_structural` hands
/// `Recorded` and `Overtaken` back to its caller, whose shared verdict
/// handling settles the call's pre-flight increments (`b_pgs_tests`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_with_an_operation_answers_each_rerun_by_whose_record_it_is() {
    use live_edit::storage::layer::{RecordedBy, RecordedCommit};

    let fx = fixture("structural-op-answers").await;
    let original = operation(OP_ID, fx.layer, OP_PAYLOAD, StartRecord::Absent);
    let mut commit = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            OP_CALL_TOKEN,
        )
        .await
        .commit;
    commit.operation = Some(original.clone());
    let server_ts = fx.commit_landed(&commit).await;
    let seq = commit.expected_seq + 1;
    let (_, record) = recorded_token(&fx, WRITER).await;
    assert_eq!(
        record,
        format!(
            "op.{OP_ID}:{seq}:{server_ts}:{OP_CALL_TOKEN}:{}",
            original.digest()
        ),
        "fixture precondition: the commit recorded the operation shape"
    );
    let after_start = StartRecord::Record(record.clone());
    let recorded = |by| RecordedCommit { by, seq, server_ts };

    let cases = [
        (
            "the same call again",
            commit.clone(),
            CommitOutcome::AlreadyLanded {
                seq,
                server_ts,
                record: record.clone(),
            },
        ),
        (
            "a re-send of the same request",
            as_call(
                &commit,
                OTHER_CALL_TOKEN,
                operation(OP_ID, fx.layer, OP_PAYLOAD, after_start.clone()),
            ),
            CommitOutcome::Recorded(recorded(RecordedBy::EarlierCall)),
        ),
        (
            "the id reused for another request",
            as_call(
                &commit,
                OTHER_CALL_TOKEN,
                operation(OP_ID, fx.layer, OTHER_PAYLOAD, after_start),
            ),
            CommitOutcome::Recorded(recorded(RecordedBy::Conflict)),
        ),
        (
            "another operation that started before the commit",
            as_call(
                &commit,
                OTHER_CALL_TOKEN,
                operation(OTHER_OP_ID, fx.layer, OP_PAYLOAD, StartRecord::Absent),
            ),
            CommitOutcome::Overtaken,
        ),
    ];
    for (case, rerun, expected) in cases {
        let before = fx.keyspace().await;

        let (result, calls) = fx.commit(&rerun).await;

        match result {
            Ok(outcome) => assert_eq!(outcome, expected, "{case}"),
            Err(e) => panic!("{case}: expected {expected:?}; got the error {e:?}"),
        }
        if matches!(expected, CommitOutcome::AlreadyLanded { .. }) {
            assert!(
                matches!(
                    calls.as_slice(),
                    [ObserverCall::CommitSize { .. }, ObserverCall::AlreadyLanded { seq: s, server_ts: t }]
                        if *s == seq && *t == server_ts
                ),
                "{case}: this call's own commit is reported through already_landed; got {calls:?}"
            );
        } else {
            assert_sent_without_landing(&calls, case);
        }
        assert_eq!(
            fx.keyspace().await,
            before,
            "{case}: an answered re-run writes nothing and releases nothing"
        );
    }
}

/// b.pav: with a CHAINED client operation (its writer guard its predecessor),
/// a structural commit is answered by the script's PREDECESSOR step when the
/// writer's record does not name that predecessor —
/// `CommitOutcome::PredecessorPending` carrying the record the script read,
/// `None` when absent — writing and releasing nothing itself (the caller's
/// shared verdict handling settles the call's pre-flight increments), and
/// ahead of the fence: every re-run carries the original plan, whose `expected_seq` the
/// landed commit has since moved, so without the answer each would be
/// `Stale`. The cases: a predecessor the record is not; a chain's first write
/// over a record; and, once the record is deleted, a predecessor with no
/// record. The control: the same chained call naming the record's operation
/// as its predecessor goes on to the next guard, `Stale`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_with_a_chained_operation_answers_predecessor_pending_ahead_of_stale() {
    let fx = fixture("structural-op-predecessor").await;
    let mut commit = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            OP_CALL_TOKEN,
        )
        .await
        .commit;
    commit.operation = Some(operation(OP_ID, fx.layer, OP_PAYLOAD, StartRecord::Absent));
    fx.commit_landed(&commit).await;
    let (_, record) = recorded_token(&fx, WRITER).await;
    let chained = |previous: &str| {
        as_call(
            &commit,
            OTHER_CALL_TOKEN,
            Operation::new(OTHER_OP_ID, operation_request_digest(fx.layer, OP_PAYLOAD))
                .with_predecessor(previous),
        )
    };

    let assert_pending = |case: &'static str, rerun: StructuralCommit, read: Option<String>| {
        let fx = &fx;
        async move {
            let before = fx.keyspace().await;

            let (result, calls) = fx.commit(&rerun).await;

            match result {
                Ok(outcome) => assert_eq!(
                    outcome,
                    CommitOutcome::PredecessorPending { record: read },
                    "{case}"
                ),
                Err(e) => panic!("{case}: expected PredecessorPending; got the error {e:?}"),
            }
            assert_sent_without_landing(&calls, case);
            assert_eq!(
                fx.keyspace().await,
                before,
                "{case}: a PREDECESSOR answer writes nothing and releases nothing"
            );
        }
    };
    assert_pending(
        "a predecessor the record is not",
        chained("structural-op-not-the-record"),
        Some(record.clone()),
    )
    .await;
    assert_pending(
        "a chain's first write over a record",
        chained(""),
        Some(record.clone()),
    )
    .await;

    let (control, _) = fx.commit(&chained(OP_ID)).await;
    assert!(
        matches!(control, Ok(CommitOutcome::Stale { .. })),
        "control: with its predecessor the record, the chained re-run goes on to the next \
         guard; got {control:?}"
    );

    let removed: i64 = fx
        .tv
        .client()
        .pool
        .hdel(fx.keys().obj_write_token(&fx.oid_str()), WRITER)
        .await
        .expect("HDEL obj:wtok <writer>");
    assert_eq!(removed, 1, "fixture: the writer's record existed");
    assert_pending("a predecessor with no record", chained(OP_ID), None).await;
}

/// An operation-id call without a writer guard (an unchained call whose start
/// record was never read) sends an empty writer-guard slot with a non-empty
/// record suffix, which the script refuses as an encoder bug (`ENCODING`)
/// rather than run the commit without its writer guard — the mapping the fast
/// mode's caller gets for the same slots — and nothing is written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_with_an_operation_but_no_start_record_is_refused_encoding() {
    let fx = fixture("structural-op-no-start").await;
    let mut commit = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            OP_CALL_TOKEN,
        )
        .await
        .commit;
    commit.operation = Some(Operation::new(
        OP_ID,
        operation_request_digest(fx.layer, OP_PAYLOAD),
    ));

    fx.assert_encoding_refused(
        &commit,
        &format!(
            "operation record suffix (ARGV[{UPDATE_SCRIPT_ARG_RECORD_SUFFIX}]) without a writer \
             guard (ARGV[{UPDATE_SCRIPT_ARG_WRITER_GUARD}])"
        ),
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The observer: commit_size
// ═════════════════════════════════════════════════════════════════════════════

/// `commit_size` reports the plan's records without the header bump, and the
/// bytes of every `KEYS` string and `ARGV` element the call sends — rebuilt
/// here from the script's structural-mode layout and the public encoder,
/// the untouched-row count block included. The plan writes `CUBE`'s external
/// hash to `WORLD`'s field, so the block carries an entry and its bytes
/// weigh in the sum. `landed` follows it with the commit's `seq` and
/// `server_ts`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn commit_structural_commit_size_is_the_non_header_records_and_the_call_bytes() {
    let fx = fixture("structural-size").await;
    let cube_hash = [0xAB_u8; 32];
    let c = fx
        .plan(
            OwnedDelta {
                struct_commands: vec![create_cmd(WORLD, NEW_SECTION, "Lamp")],
                set_fields: vec![SetField {
                    section_id: WORLD,
                    key_name: WORLD_FIELD.to_owned(),
                    value_or_ext_hash: cube_hash.to_vec(),
                    ext_value_size: 1_024,
                    ext_value_index: 1,
                    set_order: 0,
                }],
                ..Default::default()
            },
            "tok-size",
        )
        .await
        .commit;
    pre_store(&ext_store(&fx.tv), &[cube_hash]).await;
    let records = count(&c.mutations, |m| {
        !matches!(m, StorageMutation::SetObjectHeader { .. })
    }) as u64;
    assert_eq!(
        records, 3,
        "fixture precondition: SetSection, SetChildren and SetField"
    );
    let (oid, o, keys) = (fx.oid, fx.oid_str(), fx.keys());
    let encoded = encode_mutations(
        &c.mutations,
        &c.pre_image_digests,
        c.next_set_order,
        keys,
        &o,
    )
    .expect("the plan encodes");
    assert_eq!(
        encoded.untouched_ext_block.len(),
        3,
        "fixture precondition: the block carries one entry (K, hash, count)"
    );

    let (result, calls) = fx.commit(&c).await;

    let Ok(CommitOutcome::Committed { seq, server_ts }) = result else {
        panic!("the commit must land; got {result:?}");
    };
    let key_bytes: usize = update_script_shared_keys(keys, oid, 3)
        .into_iter()
        .chain([
            keys.obj_sections_index(&o),
            keys.obj_deleted_sections_index(&o),
            keys.obj_children_index(&o),
        ])
        .collect::<Vec<_>>()
        .iter()
        .chain(&encoded.mutation_keys)
        .map(String::len)
        .sum();
    // ARGV[11..14], the relay byte budget's B, T, K and MIN: the fixture's
    // store runs `LayerStore::new`'s default budget.
    let retention = RelayRetention::from_budget(LAYER_STREAM_MAX_BYTES_DEFAULT);
    let argv_bytes: usize = [
        server_ts.to_string(),
        WRITER.to_owned(),
        CLIENT_TS.to_string(),
        fx.ls.layer_stream_maxlen().to_string(),
        o.clone(),
        c.mutations.len().to_string(),
        // ARGV[8..10], the call's identity: its commit token as the match key;
        // with no operation the record suffix and start record are empty.
        c.commit_token.clone(),
        retention.budget_bytes.to_string(),
        retention.generation_bytes.to_string(),
        retention.generations.to_string(),
        retention.min_entries.to_string(),
        // ARGV[15], the call token: the commit token again, the key of the
        // call's commit release ring entry.
        c.commit_token.clone(),
        // ARGV[16], the commit release ring's TTL: the fixture's store runs
        // `LayerStore::new`'s default.
        default_ring_ttl_ms().to_string(),
        UpdateScriptMode::Structural.as_str().to_owned(),
        c.expected_seq.to_string(),
        c.expected_commit_chain.clone(),
    ]
    .iter()
    .chain(&encoded.untouched_ext_block)
    .chain(&encoded.mutation_argv)
    .map(String::len)
    .sum::<usize>()
        + c.relay_ompf.as_ref().map_or(0, Bytes::len);
    assert_eq!(
        calls,
        vec![
            ObserverCall::CommitSize {
                records,
                bytes: (key_bytes + argv_bytes) as u64,
            },
            ObserverCall::Landed { seq, server_ts },
        ]
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Source shape
// ═════════════════════════════════════════════════════════════════════════════

const LAYER_RS: &str = include_str!("../../../src/storage/layer.rs");
const STRUCTURAL_RS: &str = include_str!("../../../src/storage/structural.rs");

/// The text of the function whose signature starts with `signature`, through
/// its closing brace at the signature's own indentation (rustfmt's layout).
fn fn_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("no `{signature}` in the source"));
    let line_start = source[..start].rfind('\n').map_or(0, |i| i + 1);
    let indent = &source[line_start..start];
    assert!(
        indent.chars().all(char::is_whitespace),
        "`{signature}` must start its line"
    );
    let close = format!("\n{indent}}}\n");
    let end = source[start..]
        .find(&close)
        .unwrap_or_else(|| panic!("no closing brace for `{signature}`"));
    &source[start..start + end + close.len()]
}

/// `text` without its full-line `//` comments, so prose never satisfies or
/// trips a code check.
fn code_lines(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The code of `body` from the first `from` to the first `to` after it.
fn code_between(body: &str, from: &str, to: &str) -> String {
    let code = code_lines(body);
    let start = code
        .find(from)
        .unwrap_or_else(|| panic!("no `{from}` in the function"));
    let len = code[start..]
        .find(to)
        .unwrap_or_else(|| panic!("no `{to}` after `{from}`"));
    code[start..start + len].to_owned()
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// Both modes' callers decode an already-landed reply — and the step-1d
/// `{"FENCED", ms}` reply — with the one extracted function each, and map
/// every other error tag with the one shared mapping, and the structural
/// module carries no decode of its own — so whatever plan b.v6d adds at that
/// mapping, and the one fence reply form (W1, B3), hold for both modes.
#[test]
fn commit_structural_and_the_fast_path_share_the_already_landed_decode_and_error_mapping() {
    let fast = code_lines(fn_body(LAYER_RS, "async fn update_layer_once_lua("));
    let structural = code_lines(fn_body(STRUCTURAL_RS, "pub async fn commit_structural("));
    for (caller, body) in [
        ("update_layer_once_lua", &fast),
        ("commit_structural", &structural),
    ] {
        for shared in [
            "decode_commit_token_reply(",
            "decode_fenced_reply(",
            "map_update_script_error(",
        ] {
            assert_eq!(
                body.matches(shared).count(),
                1,
                "{caller} must call the shared `{shared}` exactly once"
            );
        }
    }
    let structural_code = code_lines(STRUCTURAL_RS);
    for copy in [
        "parse_commit_token_record",
        "\"ALREADY_LANDED\"",
        "ALREADY_LANDED_REPLY_TAG",
    ] {
        assert!(
            !structural_code.contains(copy),
            "structural.rs must not decode the already-landed reply itself (found `{copy}`)"
        );
    }
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// Both modes size their commit with the one `lua::script_call_bytes`, over
/// the very `KEYS` and `ARGV` vectors their `EVALSHA` then sends, unchanged
/// in between — so a fast and a structural commit's `bytes` compare like for
/// like. The value itself is pinned elsewhere: against the store's `MONITOR`
/// feed for the fast mode
/// (`update_layer_commit_count_test::update_layer_fast_path_commit_line_carries_the_records_and_bytes_its_script_was_sent`),
/// and against the call rebuilt from the script layout for the structural
/// mode (`commit_structural_commit_size_is_the_non_header_records_and_the_call_bytes`).
#[test]
fn commit_structural_and_the_fast_path_size_the_call_they_send_with_script_call_bytes() {
    for (caller, body, keys, argv) in [
        (
            "update_layer_once_lua",
            code_lines(fn_body(LAYER_RS, "async fn update_layer_once_lua(")),
            "keys",
            "argv",
        ),
        (
            "commit_structural",
            code_lines(fn_body(STRUCTURAL_RS, "pub async fn commit_structural(")),
            "script_keys",
            "argv",
        ),
    ] {
        let sizing = format!("script_call_bytes(&{keys}, &{argv})");
        assert_eq!(
            body.matches("script_call_bytes(").count(),
            1,
            "{caller} must size its commit with one `script_call_bytes` call"
        );
        assert!(
            body.contains(&sizing),
            "{caller} must size exactly the vectors it sends: `{sizing}`"
        );
        let sent = code_between(&body, &sizing, "UPDATE_LAYER_FAST_SCRIPT");
        assert!(
            sent.contains("evalsha_once_with_reload::<"),
            "{caller}: the `EVALSHA` follows the sizing"
        );
        let send_args = code_between(&body, "&UPDATE_LAYER_FAST_SCRIPT", ".await");
        let send_args: Vec<&str> = send_args
            .split(',')
            .map(|arg| arg.trim().trim_end_matches(')').trim())
            .filter(|arg| !arg.is_empty())
            .collect();
        assert_eq!(
            send_args,
            ["&UPDATE_LAYER_FAST_SCRIPT", keys, argv],
            "{caller} must send the vectors it sized"
        );
        for mutation in [".push(", ".extend(", ".insert(", ".truncate("] {
            for vector in [keys, argv] {
                assert!(
                    !sent.contains(&format!("{vector}{mutation}")),
                    "{caller} must not change `{vector}` between sizing and sending it \
                     (found `{vector}{mutation}`)"
                );
            }
        }
    }
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// No `.await` sits between the reply decode and `landed` /
/// `already_landed`, so a call dropped at its next await has already counted
/// the commit it landed (SR-2.6).
#[test]
fn commit_structural_has_no_await_between_the_reply_decode_and_the_landing_calls() {
    let body = fn_body(STRUCTURAL_RS, "pub async fn commit_structural(");
    for (decode, landing) in [
        ("decode_commit_token_reply(", "observer.already_landed("),
        ("decode_committed_reply(", "observer.landed("),
    ] {
        let between = code_between(body, decode, landing);
        assert!(
            !between.contains(".await"),
            "an `.await` sits between `{decode}` and `{landing}`:\n{between}"
        );
    }
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// The commit is one `EVALSHA` on the data pool (SR-6.6): no `txn_pool` slot,
/// no `WATCH`, no `MULTI`.
#[test]
fn commit_structural_takes_no_txn_pool_slot_and_sends_no_watch_or_multi() {
    let body = code_lines(fn_body(STRUCTURAL_RS, "pub async fn commit_structural("));
    for forbidden in [
        "txn_pool",
        "checkout",
        "acquire_txn_for",
        "WatchedTxn",
        "watch(",
        "WATCH",
        "multi(",
        "MULTI",
    ] {
        assert!(
            !body.contains(forbidden),
            "commit_structural must not use `{forbidden}`"
        );
    }
    assert!(
        body.contains(".pool.next()"),
        "the EVALSHA goes to the data pool"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Test support: the object-keyspace dump
// ═════════════════════════════════════════════════════════════════════════════

/// The dump names exactly the keys a `SCAN` of the object's hash tag finds,
/// drops the four header columns that differ between fixtures, and normalizes
/// the server timestamp (and, on request, the token) in commit-token records
/// and the object id and server timestamp in stream payloads. Tasks 3 and 4
/// compare whole keyspaces through it, so a normalization that missed would
/// fail their compares and one that over-reached would hide a difference.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dump_object_keyspace_names_every_key_and_normalizes_what_differs_by_design() {
    let fx = fixture("structural-dump").await;
    let (seq, _) = fx
        .ls
        .update_layer_relayed(
            &fx.session_id,
            fx.layer,
            fx.oid,
            Some(delta_with_set_field_inline(
                WORLD,
                WORLD_FIELD,
                vec![0x42],
                0,
            )),
            WRITER,
            CLIENT_TS,
        )
        .await
        .expect("the fast-mode commit must land");
    let (token, _) = recorded_token(&fx, WRITER).await;
    let scanned: BTreeSet<String> =
        crate::common::scan_object_keys(fx.tv.client(), fx.tv.prefix(), fx.oid)
            .await
            .iter()
            .map(|k| fx.relative(&k.as_str_lossy()))
            .collect();

    let verbatim =
        dump_object_keyspace(fx.tv.client(), fx.keys(), fx.oid, CommitTokens::Verbatim).await;
    let normalized =
        dump_object_keyspace(fx.tv.client(), fx.keys(), fx.oid, CommitTokens::Normalized).await;

    assert_eq!(
        verbatim.keys.keys().cloned().collect::<BTreeSet<_>>(),
        scanned
    );
    let Some(DumpedValue::Hash(header)) = verbatim.get("obj:hdr") else {
        panic!("obj:hdr must dump as a hash");
    };
    assert_eq!(header.get("seq"), Some(&seq.to_string()));
    for dropped in DUMP_DROPPED_HEADER_COLUMNS {
        assert!(
            !header.contains_key(dropped),
            "obj:hdr must not carry {dropped}"
        );
    }
    let wtok = |dump: &ObjectKeyspaceDump| match dump.get("obj:wtok") {
        Some(DumpedValue::Hash(records)) => records.get(WRITER).cloned(),
        other => panic!("obj:wtok must dump as a hash; got {other:?}"),
    };
    assert_eq!(
        wtok(&verbatim),
        Some(format!("{token}:{seq}:{DUMP_SERVER_TS}"))
    );
    assert_eq!(
        wtok(&normalized),
        Some(format!("{DUMP_COMMIT_TOKEN}:{seq}:{DUMP_SERVER_TS}"))
    );
    let Some(DumpedValue::Stream(entries)) = verbatim.get("obj:stream") else {
        panic!("obj:stream must dump as a stream");
    };
    let entry = entries
        .iter()
        .find(|e| e.id == format!("{seq}-0"))
        .expect("the fast commit's entry");
    assert_eq!(
        entry.payload.as_deref(),
        Some(
            format!("update|{DUMP_OBJECT_ID}|{seq}|{WRITER}|{CLIENT_TS}|{DUMP_SERVER_TS}").as_str()
        )
    );
}

/// The dump normalizes an operation-id record (plan b.v6d,
/// `op.<operation_id>:<seq>:<server_ts>:<token>:<request_digest>`) as it does
/// the per-call one: the server timestamp always, the call's commit token on
/// request, and nothing else — the match key, `seq` and request digest stay,
/// so two routes that recorded different operations or requests still
/// compare unequal. The raw record carries the commit's real timestamp, so
/// the placeholder is a replacement, not a coincidence.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dump_object_keyspace_normalizes_an_operation_record() {
    let fx = fixture("structural-dump-op").await;
    let mut commit = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            OP_CALL_TOKEN,
        )
        .await
        .commit;
    let op = operation(OP_ID, fx.layer, OP_PAYLOAD, StartRecord::Absent);
    commit.operation = Some(op.clone());
    let server_ts = fx.commit_landed(&commit).await;
    let seq = commit.expected_seq + 1;
    let digest = op.digest();
    let (_, raw) = recorded_token(&fx, WRITER).await;
    assert_eq!(
        raw,
        format!("op.{OP_ID}:{seq}:{server_ts}:{OP_CALL_TOKEN}:{digest}"),
        "fixture precondition: the commit recorded the operation shape"
    );

    let verbatim =
        dump_object_keyspace(fx.tv.client(), fx.keys(), fx.oid, CommitTokens::Verbatim).await;
    let normalized =
        dump_object_keyspace(fx.tv.client(), fx.keys(), fx.oid, CommitTokens::Normalized).await;

    let wtok = |dump: &ObjectKeyspaceDump| match dump.get("obj:wtok") {
        Some(DumpedValue::Hash(records)) => records.get(WRITER).cloned(),
        other => panic!("obj:wtok must dump as a hash; got {other:?}"),
    };
    assert_eq!(
        wtok(&verbatim),
        Some(format!(
            "op.{OP_ID}:{seq}:{DUMP_SERVER_TS}:{OP_CALL_TOKEN}:{digest}"
        ))
    );
    assert_eq!(
        wtok(&normalized),
        Some(format!(
            "op.{OP_ID}:{seq}:{DUMP_SERVER_TS}:{DUMP_COMMIT_TOKEN}:{digest}"
        ))
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The commit EVALSHA is written to the wire at most once
// ═════════════════════════════════════════════════════════════════════════════

/// Bound on each wait in the dropped-connection test: for the commit to be
/// queued behind the stalled connection, and for the pool to serve again
/// after the kill.
const CONN_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

/// Wait until `stall`'s connection has read more than `baseline` bytes: a
/// command was written to it and queues behind the stall.
async fn await_queued_behind(
    stall: &live_edit::storage::test_support::ConnStall,
    baseline: u64,
    what: &str,
) {
    let deadline = tokio::time::Instant::now() + CONN_WAIT;
    loop {
        if stall.net_in().await.unwrap_or(0) > baseline {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} never reached the stalled connection"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
}

/// Wait for the data pool to serve again after its connection was killed. A
/// command fred re-writes after the reconnect (a replay) is written before
/// this `PING`, on the same connection, so once it is answered any replay has
/// executed.
async fn await_reconnected(fx: &Fixture) {
    use fred::interfaces::ClientLike;
    let deadline = tokio::time::Instant::now() + CONN_WAIT;
    loop {
        let ping = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            fx.tv.client().pool.next().ping::<fred::types::Value>(None),
        )
        .await;
        if matches!(ping, Ok(Ok(_))) {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the data pool never reconnected after the kill"
        );
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

/// The structural commit's `EVALSHA` is written to the wire at most once, as
/// the fast mode's is (`lua::evalsha_once_with_reload`): a dropped connection
/// fails the call transiently instead of being replayed by fred, and the
/// call's next attempt reconciles through the commit-token check.
///
/// The store's data pool is ONE connection, parked in a server-side `BLPOP`
/// (`ConnStall`), so the commit's `EVALSHA` is written but queued behind it;
/// then the connection is killed, and everything queued dies with it. Written
/// once, the call answers a transient error with nothing landed — the
/// observer saw `commit_size` only — and the pool serves again without the
/// commit ever running. With fred's default attempt budget the reconnect
/// re-writes the queued `EVALSHA` and the call answers `Committed` — a
/// replay, which after the client's next commit would apply out of order.
/// The call's next attempt (the same commit, the same operation and start
/// record) then meets no landed commit and commits once. (An attempt whose
/// `EVALSHA` DID run before its reply was lost is answered `AlreadyLanded` by
/// that check instead: `fault_seams::post_eval_fault_loses_a_landed_reply_…`.)
///
/// Run with and without a client operation id: the identity changes which
/// record key the reconciling attempt checks, never the once-only send.
async fn killed_connection_fails_the_structural_commit_and_the_retry_commits(
    layer: &'static str,
    with_operation: bool,
) {
    use std::sync::Arc;

    let fx = fixture_on(TestValkey::setup_with_pool_size(1).await, layer).await;
    let mut commit = fx
        .plan(
            structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
            OP_CALL_TOKEN,
        )
        .await
        .commit;
    if with_operation {
        commit.operation = Some(operation(OP_ID, fx.layer, OP_PAYLOAD, StartRecord::Absent));
    }
    let seq_before = fx.header_seq().await;
    // Loaded up front, so the queued command is the commit's own `EVALSHA`
    // and not a `NOSCRIPT` round trip.
    let _: () = UPDATE_LAYER_FAST_SCRIPT
        .load(fx.tv.client().pool.next())
        .await
        .expect("SCRIPT LOAD must succeed");

    let stall = live_edit::storage::test_support::ConnStall::park(
        fx.tv.client().pool.next(),
        fx.tv.client().maintenance_pool.next(),
        format!("{}structural-once-stall", fx.tv.prefix()),
    )
    .await;
    let baseline = stall.net_in().await.unwrap_or(0);
    let ls = Arc::new(layer_store(&fx.tv));
    let sent = Arc::new(commit.clone());
    let call = {
        let (ls, sent) = (Arc::clone(&ls), Arc::clone(&sent));
        tokio::spawn(async move {
            let mut observer = RecordingObserver::default();
            let result = commit_structural(&ls, &sent, &mut observer).await;
            (result, observer.calls)
        })
    };
    await_queued_behind(&stall, baseline, "the structural commit's EVALSHA").await;
    let killed = stall.kill().await;
    assert!(
        killed.is_err(),
        "precondition: the stall was ended by the kill, not released; got {killed:?}"
    );
    let (result, calls) = tokio::time::timeout(CONN_WAIT, call)
        .await
        .expect("the call must answer once its connection is killed")
        .expect("the call task must not panic");
    await_reconnected(&fx).await;

    assert!(
        matches!(
            result,
            Err(ServiceError::Aborted(_) | ServiceError::Unavailable(_))
        ),
        "the killed connection must fail the call transiently. `Committed` here is fred \
         replaying the queued EVALSHA after the reconnect: the commit's one-attempt budget did \
         not reach it; got {result:?}"
    );
    assert_sent_without_landing(&calls, "killed connection");
    assert_eq!(
        fx.header_seq().await,
        seq_before,
        "the commit queued on the killed connection never ran, then or after the reconnect"
    );

    let (retried, retried_calls) = fx.commit(&commit).await;

    assert!(
        matches!(
            retried,
            Ok(CommitOutcome::Committed { seq, .. }) if seq == seq_before + 1
        ),
        "the call's next attempt finds no landed commit and commits once; got {retried:?}"
    );
    assert!(
        matches!(
            retried_calls.as_slice(),
            [ObserverCall::CommitSize { .. }, ObserverCall::Landed { .. }]
        ),
        "got {retried_calls:?}"
    );
    assert_eq!(fx.header_seq().await, seq_before + 1, "exactly one commit");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn killed_connection_fails_the_structural_commit_and_the_retry_commits_without_an_operation()
{
    killed_connection_fails_the_structural_commit_and_the_retry_commits(
        "structural-once-no-op",
        false,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn killed_connection_fails_the_structural_commit_and_the_retry_commits_with_an_operation() {
    killed_connection_fails_the_structural_commit_and_the_retry_commits("structural-once-op", true)
        .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Fault seams
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "test-fault-injection")]
mod fault_seams {
    //! Needs `test-fault-injection`: the release-enqueue hook, the two Lua
    //! fault queues and the fixed commit-token seam are compiled only under
    //! it.

    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use live_edit::storage::client::ValkeyClient;

    use super::*;

    /// One observed step of a commit, from the observer or the release hook.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Step {
        CommitSize,
        /// `landed`, with the header `seq` read from Valkey inside the call.
        Landed {
            seq: u64,
            header_seq_at_call: u64,
        },
        AlreadyLanded,
        ReleaseEnqueue(Vec<String>),
    }

    type Log = Arc<Mutex<Vec<Step>>>;

    /// Logs each observer call into the shared [`Log`]. `landed` reads the
    /// header `seq` synchronously (the method is sync and must stay so), on
    /// a multi-thread runtime through `block_in_place`.
    struct ProbingObserver {
        log: Log,
        client: ValkeyClient,
        header_key: String,
    }

    impl CommitObserver for ProbingObserver {
        fn commit_size(&mut self, _records: u64, _bytes: u64) {
            self.log.lock().unwrap().push(Step::CommitSize);
        }

        fn landed(&mut self, seq: u64, _server_ts: u64) {
            let (pool, key) = (self.client.pool.clone(), self.header_key.clone());
            let header_seq_at_call = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    let seq: String = pool.hget(&key, "seq").await.expect("HGET seq");
                    seq.parse::<u64>().expect("seq is a u64")
                })
            });
            self.log.lock().unwrap().push(Step::Landed {
                seq,
                header_seq_at_call,
            });
        }

        fn already_landed(&mut self, _seq: u64, _server_ts: u64) {
            self.log.lock().unwrap().push(Step::AlreadyLanded);
        }
    }

    fn probing_observer(fx: &Fixture, log: &Log) -> ProbingObserver {
        ProbingObserver {
            log: log.clone(),
            client: fx.tv.client().clone(),
            header_key: fx.keys().obj_header(&fx.oid_str()),
        }
    }

    fn fred_queue(kind: FredErrorKind) -> Arc<Mutex<VecDeque<FredError>>> {
        Arc::new(Mutex::new(VecDeque::from([FredError::new(
            kind,
            "structural-commit-test: injected",
        )])))
    }

    /// A commit that drops two external values (deleting `CUBE` drops its
    /// external field and sample, each the object's only row holding its
    /// hash) runs size → landed → release enqueue, in that order; the commit
    /// is already in Valkey when `landed` runs; the hook sees the two
    /// `rewrite` releases, which are enqueued by the time the call returns.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn committed_commit_calls_size_then_landed_then_release_enqueue() {
        let fx = fixture("structural-order").await;
        let log: Log = Arc::default();
        let hook_log = log.clone();
        let ls = layer_store(&fx.tv).with_structural_release_enqueue_hook_for_test(Arc::new(
            move |tokens: &[String]| {
                let mut tokens = tokens.to_vec();
                tokens.sort();
                hook_log.lock().unwrap().push(Step::ReleaseEnqueue(tokens));
            },
        ));
        let commit = fx
            .plan(structural(vec![delete_cmd(WORLD, CUBE)]), "tok-order")
            .await
            .commit;
        let seq = commit.expected_seq + 1;
        let mut observer = probing_observer(&fx, &log);

        let result = commit_structural(&ls, &commit, &mut observer).await;

        assert!(
            matches!(result, Ok(CommitOutcome::Committed { .. })),
            "got {result:?}"
        );
        // `every_shape_state`'s external hashes are 32 copies of one tag byte.
        let hash_hex = |tag: u8| format!("{tag:02x}").repeat(32);
        let rewrite = |tag: u8| format!("rewrite:{}", hash_hex(tag));
        assert_eq!(
            *log.lock().unwrap(),
            vec![
                Step::CommitSize,
                Step::Landed {
                    seq,
                    header_seq_at_call: seq,
                },
                Step::ReleaseEnqueue(vec![rewrite(0xAB), rewrite(0xCD)]),
            ]
        );
        let pending: BTreeSet<String> = fx
            .members(fx.keys().ext_pending())
            .await
            .into_iter()
            .map(|member| member.split(':').next().unwrap_or_default().to_owned())
            .collect();
        assert_eq!(
            pending,
            [hash_hex(0xAB), hash_hex(0xCD)].into(),
            "both releases must be enqueued before commit_structural returns"
        );
    }

    /// The pre-`EVALSHA` fault queue fails the call in place of the
    /// `EVALSHA`: `commit_size` was reported (it precedes the send), nothing
    /// landed, nothing was written.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pre_eval_fault_reports_size_and_fails_before_the_script_runs() {
        let fx = fixture("structural-pre-eval").await;
        let queue = fred_queue(FredErrorKind::Timeout);
        let ls = layer_store(&fx.tv).with_lua_fred_error_injector_for_test(queue.clone());
        let commit = fx
            .plan(
                structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
                "tok-pre",
            )
            .await
            .commit;
        let before = fx.keyspace().await;

        let (result, calls) = fx.commit_with(&ls, &commit).await;

        assert!(
            matches!(result, Err(ServiceError::Aborted(_))),
            "got {result:?}"
        );
        assert!(
            queue.lock().unwrap().is_empty(),
            "the call must consume the fault"
        );
        assert_sent_without_landing(&calls, "pre-EVALSHA fault");
        assert_eq!(fx.keyspace().await, before, "the script must not have run");
    }

    /// The post-`EVALSHA` fault queue replaces a committed reply — a lost
    /// reply: the call fails and reports no landing, the commit is in the
    /// store, and the call's next attempt (the same commit) answers
    /// `AlreadyLanded` at that commit's `seq`, its observer recording
    /// `commit_size` then one `already_landed` and no `landed`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn post_eval_fault_loses_a_landed_reply_and_the_retry_is_already_landed() {
        let fx = fixture("structural-post-eval").await;
        let queue = fred_queue(FredErrorKind::Timeout);
        let ls = layer_store(&fx.tv).with_lua_post_eval_fred_error_injector_for_test(queue.clone());
        let commit = fx
            .plan(
                structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
                "tok-post",
            )
            .await
            .commit;
        let seq = commit.expected_seq + 1;

        let (lost, lost_calls) = fx.commit_with(&ls, &commit).await;
        let (retried, retried_calls) = fx.commit_with(&ls, &commit).await;

        assert!(
            matches!(lost, Err(ServiceError::Aborted(_))),
            "got {lost:?}"
        );
        assert!(
            queue.lock().unwrap().is_empty(),
            "the call must consume the fault"
        );
        assert_sent_without_landing(&lost_calls, "post-EVALSHA fault");
        let server_ts = match retried {
            Ok(CommitOutcome::AlreadyLanded {
                seq: s, server_ts, ..
            }) if s == seq => server_ts,
            other => panic!("the script ran and committed, so the retry reports it; got {other:?}"),
        };
        assert!(
            matches!(
                retried_calls.as_slice(),
                [ObserverCall::CommitSize { .. }, ObserverCall::AlreadyLanded { seq: s, server_ts: t }]
                    if *s == seq && *t == server_ts
            ),
            "the retry must call commit_size then already_landed with its outcome, and never \
             landed; got {retried_calls:?}"
        );
        assert_eq!(fx.header_seq().await, seq);
    }

    /// A commit-token record carrying the call's token with a tail that does
    /// not parse is refused with the fast path's exact `Internal` — the one
    /// shared decode — and nothing is written.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn corrupt_token_record_is_refused_with_the_fast_paths_exact_internal() {
        const TOKEN: u128 = 0x5c0d_0000_0000_0000_0000_0000_0000_0001;
        let fx = fixture("structural-corrupt-record").await;
        let token = format!("{TOKEN:032x}");
        let _: () = fx
            .tv
            .client()
            .pool
            .hset(
                fx.keys().obj_write_token(&fx.oid_str()),
                [(WRITER, format!("{token}:not-a-seq:x"))],
            )
            .await
            .expect("planting the corrupt record must succeed");
        let fast_ls = layer_store(&fx.tv)
            .with_commit_tokens_for_test(Arc::new(Mutex::new(VecDeque::from([TOKEN]))));
        let fast = fast_ls
            .update_layer_relayed(
                &fx.session_id,
                fx.layer,
                fx.oid,
                Some(delta_with_set_field_inline(
                    WORLD,
                    WORLD_FIELD,
                    vec![0x42],
                    0,
                )),
                WRITER,
                CLIENT_TS,
            )
            .await;
        let Err(ServiceError::Internal(fast_msg)) = fast else {
            panic!("fixture precondition: the fast path refuses the record Internal; got {fast:?}");
        };
        let commit = fx
            .plan(
                structural(vec![create_cmd(WORLD, NEW_SECTION, "Lamp")]),
                &token,
            )
            .await
            .commit;
        let before = fx.keyspace().await;

        let (result, calls) = fx.commit(&commit).await;

        match result {
            Err(ServiceError::Internal(msg)) => assert_eq!(msg, fast_msg),
            other => panic!("expected the fast path's Internal; got {other:?}"),
        }
        assert_sent_without_landing(&calls, "corrupt record");
        assert_eq!(fx.keyspace().await, before);
    }
}
