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

//! Integration tests for the `commit_chain` header column
//! ([`HEADER_FIELD_COMMIT_CHAIN`], Spec SDD SR-4.5): the fence identity every
//! materialization writes, the Rust fold's agreement with Valkey's Lua, and
//! the fold every commit applies on the fast and WATCH paths.
//!
//! # What is pinned here
//!
//! - **Every materialization writes a new value.** Both `queue_object_state`
//!   entry points, each driven through its real caller: `create_layer`
//!   (`write_object`) for a create and an overwrite, and `apply_checkpoint`
//!   (`write_restored_object`) for a DR restore. A create writes a 40-hex
//!   value; an overwrite writes a value different from the overwritten
//!   object's; a restore writes a value different from the pre-loss object's;
//!   and two restores of ONE checkpoint generation (the same OMPF, a wipe in
//!   between) write different values even though they land at the same `seq`
//!   and `created_at` — the case `seq` alone cannot tell apart.
//! - **The column is not structural.** It is absent from
//!   [`HEADER_STRUCTURAL_FIELDS`], and a header without it — an object written
//!   before the column existed, or by an older replica — stays structurally
//!   complete and readable.
//! - **The fold is byte-identical to Lua's.** [`commit_chain_fold`] agrees
//!   with `redis.sha1hex(prev .. "\n" .. payload)`, evaluated by the test
//!   Valkey itself through a read-only `EVAL` (no keys), on an empty `prev`, a
//!   40-hex `prev`, an empty payload, a non-ASCII payload and a payload
//!   carrying NUL and `\n` bytes. The Rust-side known-answer vectors are the
//!   unit tests in `src/storage/commit_chain.rs`.
//! - **Every commit folds the column.** After a history of commits on one
//!   object mixing the Lua fast path and the WATCH path (the txn backend, and a
//!   structural delta on the `lua-watch-structural` backend), an empty commit
//!   and external-value commits included, the header's value equals
//!   [`commit_chain_fold`] folded
//!   over the stream entries' `payload`s (read back with `XRANGE`) in order,
//!   from the materializer's value — checked after every commit, so a failure
//!   names the commit and its path. Every commit's payload equals
//!   [`update_stream_payload`] of its fields, which is the load-bearing check
//!   for the Lua commits: the script builds its payload by concatenation, not
//!   through the Rust builder. [`fold_stream_after`] is the fold over `XRANGE`
//!   payloads every history here is checked against.
//! - **Structural commits fold it too, interleaved with the other modes.** A
//!   history on one object switching mode on every commit — fast, WATCH and
//!   structural (`commit_structural` on a plan from the test-support planner),
//!   every ordered switch between the three among them, with structural empty,
//!   external-value, cascade-delete and reset commits — keeps the header equal
//!   to the fold after each commit, and each step equals Valkey's own
//!   `redis.sha1hex(prev .. "\n" .. payload)`. It is run from the
//!   materializer's value and from `""` (a pre-feature object, whose first
//!   commit is structural). Switching modes between commits is what a flip of
//!   Epic 6's rollback knob does to one object.
//! - **The fold holds under concurrent writers.** Each round races one txn
//!   writer against many Lua fast-path writers on one object; whatever order
//!   they land in, the header equals the fold over the stream. Rounds repeat
//!   until the txn store's retry counter shows the txn writer's `WATCH` was
//!   tripped. With no second txn writer in the round, only a Lua commit landing
//!   between its `WATCH` and `EXEC` trips it (store faults aside). A WATCH-path
//!   `prev` kept stale across that retry breaks the chain; one read outside
//!   the `WATCH` breaks it when a Lua commit lands between that read and the
//!   `WATCH`, an ordering the same racing makes likely but the test does not
//!   gate on.
//! - **A pre-feature object folds from `""`.** An object whose header has no
//!   column gets `commit_chain_fold("", payload)` on its first commit, on each
//!   path — the `lua` backend's structural route (its cold start reads the
//!   absent column as `""`) included.
//! - **A refused commit leaves the column unchanged**, on the fast and WATCH
//!   paths: a `set_order` overflow and a `SUPERSEDED` object, both refused
//!   inside the commit path after its header read. (A G3 wrong-length hash is
//!   refused before either path runs; `g3_external_hash_tests` pins the whole
//!   header byte-identical across it.)

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, LuaInterface, StreamsInterface};
use oodf_ffi::{CommandCreateSection, CommandDeleteSection, OwnedDelta, StructCommand};

use live_edit::checkpoint::apply::apply_checkpoint;
use live_edit::checkpoint::CheckpointMeta;
use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::{ObjectState, Section};
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::commit_chain::{commit_chain_fold, update_stream_payload};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::{
    header_fields_are_structurally_complete, LayerStore, HEADER_FIELD_COMMIT_CHAIN,
    HEADER_STRUCTURAL_FIELDS,
};
use live_edit::storage::lua::is_fast_path;
use live_edit::storage::session::SessionStore;
use live_edit::storage::structural::{commit_structural, CommitObserver, CommitOutcome};
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::structural_commit::{plan_structural_commit, PlanRequest};
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::{
    delta_to_ompf_bytes, delta_with_set_field_external, delta_with_set_field_inline,
    delta_with_time_sample_inline, empty_delta, reset_delta,
};
use crate::storage::checkpoint_apply_test::build_checkpoint_ompf;

const CREATOR: &str = "commit-chain-user";

fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
}

async fn create_test_session(tv: &TestValkey) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("commit-chain-session", "/stage", CREATOR)
        .await
        .expect("session creation must succeed")
}

fn minimal_state() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    }
}

async fn create(ls: &LayerStore, session_id: &str, layer: &str) -> u64 {
    ls.create_layer(
        session_id,
        layer,
        CREATOR,
        &minimal_state(),
        Arc::new(MockExternalValueStore::new()),
    )
    .await
    .expect("create_layer must succeed")
}

/// One `obj:hdr` column, `None` when absent.
async fn header_field(tv: &TestValkey, oid: u64, field: &str) -> Option<String> {
    tv.client()
        .pool
        .hget(tv.key_schema().obj_header(&oid.to_string()), field)
        .await
        .unwrap_or_else(|e| panic!("HGET obj:hdr {field} must succeed: {e}"))
}

/// The object's `commit_chain`, asserted present and 40-lowercase-hex — the
/// one shape the column has, whether a materializer or a fold wrote it.
async fn commit_chain(tv: &TestValkey, oid: u64, after: &str) -> String {
    let value = header_field(tv, oid, HEADER_FIELD_COMMIT_CHAIN)
        .await
        .unwrap_or_else(|| {
            panic!("after {after}, obj:hdr must carry `{HEADER_FIELD_COMMIT_CHAIN}`")
        });
    assert_eq!(
        value.len(),
        40,
        "after {after}, `{HEADER_FIELD_COMMIT_CHAIN}` must be 40 hex digits, got {value:?}"
    );
    assert!(
        value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "after {after}, `{HEADER_FIELD_COMMIT_CHAIN}` must be lowercase hex, got {value:?}"
    );
    value
}

/// Restore `oid` from `checkpoint` over a wiped object keyspace, so the header
/// can come only from the restore.
async fn wipe_and_restore(
    tv: &TestValkey,
    ls: &LayerStore,
    oid: u64,
    checkpoint: &(Bytes, u64, CheckpointMeta),
) {
    let (ompf, snap_seq, meta) = checkpoint;
    crate::common::wipe_object_keys(tv, oid).await;
    apply_checkpoint(
        ls,
        oid,
        ompf.clone(),
        *snap_seq,
        meta,
        None,
        &crate::common::test_config(),
    )
    .await
    .expect("apply_checkpoint must succeed on a wiped obj keyspace");
}

// =============================================================================
// 1. Every materialization writes a new value
// =============================================================================

#[tokio::test]
async fn create_layer_writes_a_40_hex_commit_chain() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let oid = create(&ls, &session_id, "commit-chain-create").await;

    // The shape assertions live in `commit_chain`; reaching the return is the
    // pass.
    commit_chain(&tv, oid, "CreateLayer").await;
}

#[tokio::test]
async fn create_layer_overwrite_writes_a_commit_chain_different_from_the_overwritten_object() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let layer = "commit-chain-overwrite";

    let old_oid = create(&ls, &session_id, layer).await;
    // Read before the overwrite: its cleanup removes the old header.
    let old_chain = commit_chain(&tv, old_oid, "the first CreateLayer").await;

    let new_oid = create(&ls, &session_id, layer).await;
    assert_ne!(
        old_oid, new_oid,
        "fixture precondition: an overwrite allocates a new object"
    );
    let new_chain = commit_chain(&tv, new_oid, "the overwriting CreateLayer").await;

    assert_ne!(
        new_chain, old_chain,
        "an overwrite is a new materialization and must write a new commit_chain"
    );
}

#[tokio::test]
async fn restore_writes_a_commit_chain_different_from_the_pre_loss_object() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let oid = create(&ls, &session_id, "commit-chain-restore").await;
    let pre_loss = commit_chain(&tv, oid, "CreateLayer").await;
    let checkpoint = build_checkpoint_ompf(&tv, oid).await;

    wipe_and_restore(&tv, &ls, oid, &checkpoint).await;
    let restored = commit_chain(&tv, oid, "a DR restore").await;

    // Unlike `created_at` / `last_server_ts`, which a restore reproduces from
    // the checkpoint, the fence identity is never carried over: a holder of the
    // pre-loss value must see the restore as a different object.
    assert_ne!(
        restored, pre_loss,
        "a DR restore is a new materialization and must write a new commit_chain, \
         not reproduce the pre-loss object's"
    );
}

#[tokio::test]
async fn restore_twice_from_one_checkpoint_generation_writes_different_commit_chains() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let oid = create(&ls, &session_id, "commit-chain-double-restore").await;
    // ONE generation, applied twice: the same OMPF bytes, `snap_seq` and meta.
    let checkpoint = build_checkpoint_ompf(&tv, oid).await;

    wipe_and_restore(&tv, &ls, oid, &checkpoint).await;
    let first_seq = header_field(&tv, oid, "seq").await;
    let first_created_at = header_field(&tv, oid, "created_at").await;
    let first = commit_chain(&tv, oid, "the first restore").await;

    wipe_and_restore(&tv, &ls, oid, &checkpoint).await;
    let second_seq = header_field(&tv, oid, "seq").await;
    let second_created_at = header_field(&tv, oid, "created_at").await;
    let second = commit_chain(&tv, oid, "the second restore").await;

    // Fixture preconditions: the two restores are indistinguishable by `seq`
    // and `created_at`, so `commit_chain` is the only header column left to
    // tell them apart — the case the column exists for.
    assert_eq!(
        first_seq, second_seq,
        "fixture precondition: both restores of one generation land at the same seq"
    );
    assert_eq!(
        first_created_at, second_created_at,
        "fixture precondition: both restores copy the checkpoint's created_at"
    );
    assert_ne!(
        first, second,
        "two restores of one checkpoint generation must write different commit_chains; \
         a value derived from the checkpoint would make the second restore \
         indistinguishable from the first to a holder of the first's value"
    );
}

// =============================================================================
// 2. The column is not structural
// =============================================================================

#[tokio::test]
async fn header_without_commit_chain_is_structurally_complete_and_readable() {
    assert!(
        !HEADER_STRUCTURAL_FIELDS.contains(&HEADER_FIELD_COMMIT_CHAIN),
        "`{HEADER_FIELD_COMMIT_CHAIN}` must not be in HEADER_STRUCTURAL_FIELDS, or every \
         object written before the column existed reads as structurally incomplete"
    );

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let oid = create(&ls, &session_id, "commit-chain-pre-feature").await;

    // Strip the column: the header a pre-feature object, or one an older
    // replica materialized, carries.
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let removed: u64 = tv
        .client()
        .pool
        .hdel(&header_key, HEADER_FIELD_COMMIT_CHAIN)
        .await
        .expect("HDEL obj:hdr commit_chain must succeed");
    assert_eq!(
        removed, 1,
        "fixture precondition: create_layer must have written the column"
    );
    let header: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL obj:hdr must succeed");

    assert!(
        header_fields_are_structurally_complete(&header),
        "a header without `{HEADER_FIELD_COMMIT_CHAIN}` must stay structurally complete, \
         or the checkpoint lane would read the object as absent and restore over it"
    );
    ls.read_object_header(oid)
        .await
        .expect("a header without the column must still be served by read_object_header");
}

// =============================================================================
// 3. The Rust fold is byte-identical to Valkey's `redis.sha1hex`
// =============================================================================

/// The fold exactly as a script spells it: `prev .. "\n" .. payload`, both as
/// ARGV byte strings.
const LUA_FOLD: &str = r#"return redis.sha1hex(ARGV[1] .. "\n" .. ARGV[2])"#;

#[tokio::test]
async fn commit_chain_fold_matches_valkey_sha1hex() {
    const HEX_PREV: &str = "0123456789abcdef0123456789abcdef01234567";
    let cases: [(&str, &str, &str); 5] = [
        ("empty prev, empty payload", "", ""),
        (
            "empty prev (an object with no column), update payload",
            "",
            "update|42|7|client-abc|1000000|2000000",
        ),
        ("40-hex prev, empty payload", HEX_PREV, ""),
        (
            "empty prev, non-ASCII payload",
            "",
            "update|42|7|cli\u{e9}nt-\u{6f22}\u{5b57}|1000000|2000000",
        ),
        (
            // U+00FF is `c3 bf` in UTF-8; a `&str` cannot carry a bare 0xff.
            "40-hex prev, payload with U+00FF, NUL and newline bytes",
            HEX_PREV,
            "update|42|7|\u{ff}\u{0}\n|0|0",
        ),
    ];

    let tv = TestValkey::setup().await;
    for (case, prev, payload) in cases {
        let lua: String = tv
            .client()
            .pool
            .eval(
                LUA_FOLD,
                Vec::<String>::new(),
                vec![prev.to_owned(), payload.to_owned()],
            )
            .await
            .unwrap_or_else(|e| panic!("EVAL redis.sha1hex must succeed ({case}): {e}"));
        assert_eq!(
            commit_chain_fold(prev, payload),
            lua,
            "the Rust fold must be byte-identical to the script's for {case}"
        );
    }
}

// =============================================================================
// 4. Every commit folds the column
// =============================================================================

/// The section every commit below writes rows under; the materialized state
/// carries it, so the WATCH path's apply has a live section to write against.
const WORLD: u64 = 1;
/// The section the structural commit creates under [`WORLD`].
const CUBE: u64 = 2;
/// An external value's declared size. Arbitrary but nonzero: zero would make
/// the 32 hash bytes an inline value.
const EXT_SIZE: u64 = 1_024;
/// The `set_order` ceiling (`2^53 - 1`) both commit paths refuse to cross.
const SET_ORDER_CEILING: u64 = (1u64 << 53) - 1;

/// Which commit path an `update_layer` commit takes. [`CommitPath::of`]
/// models the dispatch. In the fold histories
/// ([`history_folds_commit_chain`]) and the pre-feature first-commit tests
/// the path is asserted per commit, not assumed: the model is checked against
/// each history entry's named path as a fixture precondition, and every
/// `update_layer` commit is then read back from
/// `live_edit_update_layer_attempts_histogram`'s `path` cell
/// ([`CommitPath::label`]), so a delta the dispatch routed elsewhere fails
/// rather than pinning the other path twice. The refused-commit helper and
/// the concurrent fold test take the path from the model alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommitPath {
    /// `update_layer_fast.lua`'s fast mode: the `lua` or
    /// `lua-watch-structural` backend with a fast-path-eligible delta.
    Fast,
    /// The structural route: the `lua` backend with a delta the fast path
    /// does not admit, planned on the structural cache and committed by
    /// `update_layer_fast.lua`'s structural mode.
    Structural,
    /// `update_layer_once`'s WATCH/MULTI/EXEC body: the txn backend, or the
    /// `lua-watch-structural` backend with a delta the fast path does not
    /// admit.
    Watch,
}

impl CommitPath {
    fn of(backend: UpdateLayerBackend, delta: &OwnedDelta) -> Self {
        let fast_eligible = is_fast_path(&Some(delta.clone()));
        match backend {
            UpdateLayerBackend::Lua | UpdateLayerBackend::LuaWatchStructural if fast_eligible => {
                CommitPath::Fast
            }
            UpdateLayerBackend::Lua => CommitPath::Structural,
            UpdateLayerBackend::LuaWatchStructural | UpdateLayerBackend::Txn => CommitPath::Watch,
        }
    }

    /// The `path` label a first commit on this path records. A structural
    /// route commit here always plans on a cold start (each object's first
    /// structural commit on a store), so `structural_full`.
    fn label(self) -> &'static str {
        match self {
            CommitPath::Fast => "fast",
            CommitPath::Structural => "structural_full",
            CommitPath::Watch => "watch",
        }
    }
}

/// `live_edit_update_layer_attempts_histogram_count{path}`: the calls that
/// committed on `path`; zero before the first.
fn attempts_on(metrics: &Metrics, path: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_update_layer_attempts_histogram_count",
        &[("path", path)],
    )
    .unwrap_or_default() as u64
}

fn layer_store_with_backend(tv: &TestValkey, backend: UpdateLayerBackend) -> LayerStore {
    layer_store(tv).with_update_layer_backend(backend)
}

/// [`WORLD`] under the synthetic root, so a structural commit can create
/// [`CUBE`] beneath it.
fn state_with_world() -> ObjectState {
    let mut state = minimal_state();
    state.sections.insert(
        WORLD,
        Section {
            section_id: WORLD,
            parent_id: 0,
            name: "World".to_string(),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        },
    );
    state.children.insert(0, vec![WORLD]);
    state
}

async fn create_with_world(ls: &LayerStore, session_id: &str, layer: &str) -> u64 {
    ls.create_layer(
        session_id,
        layer,
        CREATOR,
        &state_with_world(),
        Arc::new(MockExternalValueStore::new()),
    )
    .await
    .expect("create_layer must succeed")
}

/// The object's `seq` column, parsed.
async fn header_seq(tv: &TestValkey, oid: u64) -> u64 {
    header_field(tv, oid, "seq")
        .await
        .expect("obj:hdr must carry `seq`")
        .parse()
        .expect("`seq` must be decimal")
}

/// Every commit entry on the object's stream after `after_seq`, in stream
/// order, as `(seq, payload)`. The entry id is `{seq}-0`; only the `payload`
/// field is read, so an entry carrying a binary `ompf` field reads the same.
async fn stream_commit_payloads(tv: &TestValkey, oid: u64, after_seq: u64) -> Vec<(u64, String)> {
    let entries: Vec<(String, HashMap<String, fred::types::Value>)> = tv
        .client()
        .pool
        .xrange_values(tv.key_schema().layer_stream(oid), "-", "+", None)
        .await
        .expect("XRANGE of the object's stream must succeed");
    entries
        .into_iter()
        .filter_map(|(id, fields)| {
            let seq: u64 = id
                .strip_suffix("-0")
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| panic!("stream entry id {id:?} must be `{{seq}}-0`"));
            (seq > after_seq).then(|| {
                let payload = fields
                    .get("payload")
                    .and_then(|v| v.as_str().map(|s| s.into_owned()))
                    .unwrap_or_else(|| panic!("stream entry {id} must carry a string `payload`"));
                (seq, payload)
            })
        })
        .collect()
}

/// The `commit_chain` a holder of `(after_seq, start)` derives: `start` folded
/// with [`commit_chain_fold`] over every commit payload after `after_seq`, in
/// stream order. The one fold every test here compares the header against,
/// whichever mode wrote the commits.
async fn fold_stream_after(tv: &TestValkey, oid: u64, start: &str, after_seq: u64) -> String {
    stream_commit_payloads(tv, oid, after_seq)
        .await
        .iter()
        .fold(start.to_owned(), |prev, (_, payload)| {
            commit_chain_fold(&prev, payload)
        })
}

/// A section the interleaved history creates under [`CUBE`], so deleting
/// [`CUBE`] cascades to a second section and its rows.
const LEAF: u64 = 3;

/// The mode one commit of a fold-pin history takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Mode {
    /// `update_layer` on the Lua backend with a fast-path-eligible delta.
    Fast,
    /// `update_layer`'s WATCH/MULTI/EXEC body: the txn backend, or a
    /// structural delta on the `lua-watch-structural` backend.
    Watch,
    /// `commit_structural` on a plan from the test-support planner.
    Structural,
}

/// One commit of a fold-pin history. `backend` picks the `update_layer` store
/// for [`Mode::Fast`] and [`Mode::Watch`]; a structural commit runs on the Lua
/// store.
struct HistoryCommit {
    what: &'static str,
    mode: Mode,
    backend: UpdateLayerBackend,
    delta: OwnedDelta,
}

/// `commit_structural` takes an observer; this history counts nothing.
struct NoObserver;

impl CommitObserver for NoObserver {
    fn commit_size(&mut self, _records: u64, _bytes: u64) {}
    fn landed(&mut self, _seq: u64, _server_ts: u64) {}
    fn already_landed(&mut self, _seq: u64, _server_ts: u64) {}
}

fn create_section(parent_id: u64, section_id: u64, name: &str) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id,
            section_id,
            section_name: name.to_string(),
            section_type: 1,
        })],
        ..Default::default()
    }
}

fn history_commit(
    what: &'static str,
    mode: Mode,
    backend: UpdateLayerBackend,
    delta: OwnedDelta,
) -> HistoryCommit {
    HistoryCommit {
        what,
        mode,
        backend,
        delta,
    }
}

/// Eight commits on a [`state_with_world`] object across the fast and WATCH
/// paths — an empty commit and an external-value field on each, a structural
/// delta on the `lua-watch-structural` backend among the WATCH ones.
fn fast_and_watch_history() -> Vec<HistoryCommit> {
    use Mode::{Fast, Watch};
    use UpdateLayerBackend::{Lua, LuaWatchStructural, Txn};
    let commit = history_commit;
    vec![
        commit(
            "inline field",
            Fast,
            Lua,
            delta_with_set_field_inline(WORLD, "color", vec![0x2A], 1),
        ),
        commit("empty commit", Fast, Lua, empty_delta()),
        commit(
            "external field",
            Fast,
            Lua,
            delta_with_set_field_external(WORLD, "tex", vec![0xAB; 32], EXT_SIZE, 1, 0),
        ),
        commit(
            "structural CreateSection (lua-watch-structural)",
            Watch,
            LuaWatchStructural,
            create_section(WORLD, CUBE, "Cube"),
        ),
        commit(
            "inline time sample",
            Watch,
            Txn,
            delta_with_time_sample_inline(CUBE, 1.5, vec![0x07]),
        ),
        commit("empty commit", Watch, Txn, empty_delta()),
        commit(
            "external field",
            Watch,
            Txn,
            delta_with_set_field_external(CUBE, "mesh", vec![0xCD; 32], EXT_SIZE, 1, 0),
        ),
        commit(
            "inline time sample",
            Fast,
            Lua,
            delta_with_time_sample_inline(WORLD, 2.5, vec![0x09]),
        ),
    ]
}

/// Twelve commits on a [`state_with_world`] object, each in a different mode
/// from the one before, every ordered switch between the three modes among
/// them; the structural ones include an empty commit, an external-value field,
/// a cascade delete and a reset.
fn interleaved_history() -> Vec<HistoryCommit> {
    use Mode::{Fast, Structural, Watch};
    use UpdateLayerBackend::{Lua, LuaWatchStructural, Txn};
    let commit = history_commit;
    vec![
        commit(
            "CreateSection CUBE",
            Structural,
            Lua,
            create_section(WORLD, CUBE, "Cube"),
        ),
        commit(
            "inline field",
            Fast,
            Lua,
            delta_with_set_field_inline(WORLD, "color", vec![0x2A], 1),
        ),
        commit(
            "inline time sample (txn)",
            Watch,
            Txn,
            delta_with_time_sample_inline(CUBE, 1.5, vec![0x07]),
        ),
        commit("empty commit", Structural, Lua, empty_delta()),
        commit(
            "CreateSection LEAF (a structural delta on the lua-watch-structural backend)",
            Watch,
            LuaWatchStructural,
            create_section(CUBE, LEAF, "Leaf"),
        ),
        commit(
            "external field",
            Fast,
            Lua,
            delta_with_set_field_external(WORLD, "tex", vec![0xCD; 32], EXT_SIZE, 1, 0),
        ),
        commit(
            "external field",
            Structural,
            Lua,
            delta_with_set_field_external(LEAF, "mesh", vec![0xAB; 32], EXT_SIZE, 1, 0),
        ),
        commit("empty commit (txn)", Watch, Txn, empty_delta()),
        commit(
            "DeleteSection CUBE, cascading to LEAF and both sections' rows",
            Structural,
            Lua,
            OwnedDelta {
                struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
                    parent_id: WORLD,
                    section_id: CUBE,
                })],
                ..Default::default()
            },
        ),
        commit(
            "inline time sample",
            Fast,
            Lua,
            delta_with_time_sample_inline(WORLD, 2.5, vec![0x09]),
        ),
        commit("reset", Structural, Lua, reset_delta("usdc")),
        commit(
            "inline field after the reset (txn)",
            Watch,
            Txn,
            delta_with_set_field_inline(WORLD, "color", vec![0x2B], 2),
        ),
    ]
}

/// Run `history` on one [`state_with_world`] object and, after each commit,
/// require: the delta took the path its mode names; the commit appended exactly
/// one stream entry, whose payload is [`update_stream_payload`] of the commit's
/// fields (the load-bearing check for script commits, which build the payload
/// by concatenation); the header `commit_chain` equals [`fold_stream_after`]
/// from `start` (the materializer's value, or `""` when `from_empty` strips the
/// column first); and that step equals Valkey's own `redis.sha1hex` over the
/// header's previous value and the payload. A structural plan must also fence
/// on that previous value.
async fn history_folds_commit_chain(
    layer: &'static str,
    from_empty: bool,
    history: Vec<HistoryCommit>,
) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(layer));
    let lua = layer_store_with_backend(&tv, UpdateLayerBackend::Lua).with_metrics(metrics.clone());
    let lua_watch_structural =
        layer_store_with_backend(&tv, UpdateLayerBackend::LuaWatchStructural)
            .with_metrics(metrics.clone());
    let txn = layer_store_with_backend(&tv, UpdateLayerBackend::Txn).with_metrics(metrics.clone());
    let session_id = create_test_session(&tv).await;
    let oid = create_with_world(&lua, &session_id, layer).await;
    let start = if from_empty {
        let removed: u64 = tv
            .client()
            .pool
            .hdel(
                tv.key_schema().obj_header(&oid.to_string()),
                HEADER_FIELD_COMMIT_CHAIN,
            )
            .await
            .expect("HDEL obj:hdr commit_chain must succeed");
        assert_eq!(
            removed, 1,
            "fixture precondition: create_layer must have written the column"
        );
        String::new()
    } else {
        commit_chain(&tv, oid, "CreateLayer").await
    };
    // The fold's starting `seq`. CreateLayer's own `1-0` entry is a
    // materialization's, not a commit's, so it is not folded.
    let start_seq = header_seq(&tv, oid).await;

    let mut prev = start.clone();
    for (i, commit) in history.into_iter().enumerate() {
        let what = format!(
            "commit {} ({:?} mode, {:?} backend, {})",
            i + 1,
            commit.mode,
            commit.backend,
            commit.what
        );
        let client_id = format!("fold-writer-{i}");
        let client_ts = 10_000 + i as u64;

        let (seq, server_ts) = match commit.mode {
            Mode::Fast | Mode::Watch => {
                let path = if commit.mode == Mode::Fast {
                    CommitPath::Fast
                } else {
                    CommitPath::Watch
                };
                assert_eq!(
                    CommitPath::of(commit.backend, &commit.delta),
                    path,
                    "{what}: fixture precondition — the delta must take the path named"
                );
                let store = match commit.backend {
                    UpdateLayerBackend::Lua => &lua,
                    UpdateLayerBackend::LuaWatchStructural => &lua_watch_structural,
                    UpdateLayerBackend::Txn => &txn,
                };
                let on_path_before = attempts_on(&metrics, path.label());
                let committed = store
                    .update_layer_relayed(
                        &session_id,
                        layer,
                        oid,
                        Some(commit.delta),
                        &client_id,
                        client_ts,
                    )
                    .await
                    .unwrap_or_else(|e| panic!("{what}: must commit; got {e:?}"));
                assert_eq!(
                    attempts_on(&metrics, path.label()),
                    on_path_before + 1,
                    "{what}: the commit must be recorded on path {:?}",
                    path.label()
                );
                committed
            }
            Mode::Structural => {
                let relay_ompf = Bytes::from(delta_to_ompf_bytes(&commit.delta, &[]));
                let token = format!("fold-token-{i}");
                let planned = plan_structural_commit(
                    &lua,
                    PlanRequest {
                        session_id: &session_id,
                        layer_name: layer,
                        object_id: oid,
                        delta: Some(commit.delta),
                        relay_ompf: Some(relay_ompf),
                        commit_token: &token,
                        client_id: &client_id,
                        client_ts,
                    },
                )
                .await
                .unwrap_or_else(|e| panic!("{what}: the planner must plan; got {e:?}"));
                assert_eq!(
                    planned.commit.expected_commit_chain, prev,
                    "{what}: the plan must fence on the header's current commit_chain"
                );
                match commit_structural(&lua, &planned.commit, &mut NoObserver).await {
                    Ok(CommitOutcome::Committed { seq, server_ts }) => (seq, server_ts),
                    other => panic!("{what}: must commit; got {other:?}"),
                }
            }
        };

        let payloads = stream_commit_payloads(&tv, oid, start_seq).await;
        assert_eq!(
            payloads.len(),
            i + 1,
            "{what}: every commit appends exactly one stream entry"
        );
        let payload = update_stream_payload(oid, seq, &client_id, client_ts, server_ts);
        assert_eq!(
            payloads.last(),
            Some(&(seq, payload.clone())),
            "{what}: the stream payload must equal update_stream_payload of the commit's fields"
        );
        let header = commit_chain(&tv, oid, &what).await;
        assert_eq!(
            header,
            fold_stream_after(&tv, oid, &start, start_seq).await,
            "{what}: the header commit_chain must equal the fold over every stream payload \
             since CreateLayer, from {start:?}"
        );
        let valkey_step: String = tv
            .client()
            .pool
            .eval(LUA_FOLD, Vec::<String>::new(), vec![prev.clone(), payload])
            .await
            .unwrap_or_else(|e| panic!("{what}: EVAL redis.sha1hex must succeed: {e}"));
        assert_eq!(
            header, valkey_step,
            "{what}: the step must equal Valkey's redis.sha1hex(prev .. \"\\n\" .. payload)"
        );
        prev = header;
    }
}

/// After N commits on one object across the fast and WATCH paths — an empty
/// commit and external-value commits among them — the header `commit_chain`
/// equals the fold over the N stream payloads from the materializer's value,
/// and every payload is the shared builder's string for that commit's fields.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_commits_on_fast_and_watch_paths_fold_commit_chain_over_their_stream_payloads()
{
    let history = fast_and_watch_history();
    assert!(
        history.iter().any(|c| c.mode == Mode::Fast)
            && history.iter().any(|c| c.mode == Mode::Watch),
        "fixture precondition: the history must cover both commit paths"
    );
    history_folds_commit_chain("commit-chain-fold", false, history).await;
}

/// [`interleaved_history`] with its preconditions: every commit switches mode,
/// every ordered switch between the three modes occurs, and the first commit is
/// structural (so the from-`""` run's first fold is a structural one).
fn checked_interleaved_history() -> Vec<HistoryCommit> {
    let history = interleaved_history();
    let switches: HashSet<(Mode, Mode)> = history
        .windows(2)
        .map(|pair| (pair[0].mode, pair[1].mode))
        .collect();
    assert!(
        history.windows(2).all(|pair| pair[0].mode != pair[1].mode)
            && switches.len() == 6
            && history[0].mode == Mode::Structural,
        "fixture precondition: every commit switches mode, every ordered switch between the \
         three modes occurs, and the first commit is structural; got switches {switches:?}"
    );
    history
}

/// Fast, structural and WATCH commits interleaved on one object fold the
/// column from the materializer's value without a mismatch at any switch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_and_structural_commits_interleaved_fold_commit_chain_from_the_materializer() {
    history_folds_commit_chain(
        "commit-chain-interleaved",
        false,
        checked_interleaved_history(),
    )
    .await;
}

/// The same history on a pre-feature object folds from `""`, its first
/// (structural) commit fencing on the absent column read as `""`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_and_structural_commits_interleaved_fold_commit_chain_from_empty() {
    history_folds_commit_chain(
        "commit-chain-interleaved-pre-feature",
        true,
        checked_interleaved_history(),
    )
    .await;
}

/// The Lua store's per-object admission limit in the concurrent fold test: how
/// many Lua writers commit at once.
const LUA_ADMISSION: u32 = 8;
/// Lua writers racing the one txn writer in each round. Eight times
/// [`LUA_ADMISSION`], so the Lua commits land in about eight waves: the excess
/// writers queue on the Lua store's admission gate and each is admitted as an
/// earlier Lua commit returns. Lua commits therefore keep landing across the
/// txn writer's whole `WATCH`…`EXEC` window (several round trips), not in one
/// burst that could be over before its `WATCH` is set.
const LUA_RACERS_PER_ROUND: usize = 8 * LUA_ADMISSION as usize;
/// Upper bound on racing rounds. A round ends the loop once the txn store's
/// retry counter has fired; the bound only keeps a scheduler that serializes
/// every round from spinning forever, and then fails the test loudly.
const MAX_RACE_ROUNDS: usize = 16;

/// `live_edit_update_layer_retries_total{outcome="committed"}`: retry
/// attempts taken by calls that went on to commit. On the txn backend a retry
/// is a `WATCH` that a concurrent commit tripped between the `WATCH` and
/// `EXEC`, or a store fault (transient I/O, a replaced connection).
fn committed_retries(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_update_layer_retries_total",
        &[("path", "watch"), ("outcome", "committed")],
    )
    .unwrap_or(0.0) as u64
}

/// One txn writer racing [`LUA_RACERS_PER_ROUND`] Lua fast-path writers on
/// one object, round after round: whatever order they land in, and however
/// often the txn writer's `WATCH` is tripped and it retries, the header equals
/// the fold over the stream. A WATCH-path fold whose `prev` survived a retry
/// stale breaks the chain exactly when a Lua commit trips the txn writer's
/// `WATCH`; one whose `prev` came from a read the `WATCH` does not protect
/// breaks it when a Lua commit lands between that read and the `WATCH`. No
/// sequential test reaches either ordering, and the watch-window hook cannot
/// stage them (it fires before the header read).
///
/// Each round has exactly one txn writer, and the retry counter is wired to
/// the txn store alone, so a counted retry was never a txn writer colliding
/// with another txn writer: none exists. A commit that tripped its `WATCH` was
/// a Lua fast-path commit. Rounds repeat until the counter fires, so the test
/// cannot pass unless the txn writer retried while only Lua writers were
/// committing. Store faults (transient I/O, a replaced connection) also retry,
/// and the counter cannot tell them apart; a healthy test store raises none.
///
/// The fold assertion holds for every interleaving, so it cannot flake; the
/// bound on rounds fails the test loudly if the race never happens.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_concurrent_fast_and_watch_commits_fold_commit_chain_over_their_stream_payloads(
) {
    // One txn writer holds one exclusive connection across WATCH…EXEC; the Lua
    // writers share the multiplexed data pool. The default pool covers both.
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(
        "commit-chain-concurrent-fold",
    ));
    let lua = layer_store_with_backend(&tv, UpdateLayerBackend::Lua)
        .with_update_layer_per_object_concurrency(LUA_ADMISSION);
    // The only store carrying `metrics`, so only its retries are counted.
    let txn = layer_store_with_backend(&tv, UpdateLayerBackend::Txn).with_metrics(metrics.clone());
    let session_id = create_test_session(&tv).await;
    let layer = "commit-chain-concurrent-fold";
    let oid = create_with_world(&lua, &session_id, layer).await;
    let start = commit_chain(&tv, oid, "CreateLayer").await;
    let start_seq = header_seq(&tv, oid).await;

    let delta = || delta_with_set_field_inline(WORLD, "color", vec![0x2A], 1);
    assert_eq!(
        CommitPath::of(UpdateLayerBackend::Lua, &delta()),
        CommitPath::Fast,
        "fixture precondition: the Lua racers' delta must take the fast path"
    );

    let mut rounds = 0;
    while rounds < MAX_RACE_ROUNDS && committed_retries(&metrics) == 0 {
        rounds += 1;
        // The txn writer first, so its WATCH attempt starts as the first Lua
        // wave is dispatched rather than after it.
        let racers = std::iter::once((&txn, format!("txn-racer-{rounds}")))
            .chain((0..LUA_RACERS_PER_ROUND).map(|i| (&lua, format!("lua-racer-{rounds}-{i}"))));
        let results = futures::future::join_all(racers.map(|(store, client_id)| {
            let session_id = &session_id;
            let delta = delta();
            async move {
                let result = store
                    .update_layer_relayed(session_id, layer, oid, Some(delta), &client_id, 7)
                    .await;
                (client_id, result)
            }
        }))
        .await;

        let failed: Vec<_> = results.iter().filter(|(_, r)| r.is_err()).collect();
        assert!(
            failed.is_empty(),
            "round {rounds}: every racing writer must commit; failures: {failed:?}"
        );
        assert_eq!(
            stream_commit_payloads(&tv, oid, start_seq).await.len(),
            rounds * (1 + LUA_RACERS_PER_ROUND),
            "round {rounds}: every commit appends exactly one stream entry"
        );
        assert_eq!(
            commit_chain(&tv, oid, &format!("racing round {rounds}")).await,
            fold_stream_after(&tv, oid, &start, start_seq).await,
            "round {rounds}: after one WATCH-path commit raced {LUA_RACERS_PER_ROUND} \
             fast-path commits, the header commit_chain must equal the fold over every \
             stream payload in stream order"
        );
    }
    assert!(
        committed_retries(&metrics) > 0,
        "fixture precondition: in {rounds} round(s) of one txn writer racing \
         {LUA_RACERS_PER_ROUND} Lua writers, the txn writer never retried, so no Lua \
         commit landed inside its WATCH window — the interleaving this test pins never \
         happened"
    );
}

/// Strip the column from a fresh object's header — the header an object
/// materialized before the column existed carries — then commit once on
/// `backend` and pin the result to `commit_chain_fold("", payload)`.
async fn first_commit_on_object_without_commit_chain_folds_from_empty(
    backend: UpdateLayerBackend,
    delta: OwnedDelta,
    path: CommitPath,
) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(
        "commit-chain-pre-feature",
    ));
    let ls = layer_store_with_backend(&tv, backend).with_metrics(metrics.clone());
    let session_id = create_test_session(&tv).await;
    let layer = "commit-chain-pre-feature-fold";
    let oid = create_with_world(&ls, &session_id, layer).await;
    assert_eq!(
        CommitPath::of(backend, &delta),
        path,
        "fixture precondition: the delta must take the {path:?} path on {backend:?}"
    );
    let removed: u64 = tv
        .client()
        .pool
        .hdel(
            tv.key_schema().obj_header(&oid.to_string()),
            HEADER_FIELD_COMMIT_CHAIN,
        )
        .await
        .expect("HDEL obj:hdr commit_chain must succeed");
    assert_eq!(
        removed, 1,
        "fixture precondition: create_layer must have written the column"
    );
    let before_seq = header_seq(&tv, oid).await;

    let (client_id, client_ts) = ("pre-feature-writer", 4_242);
    let (seq, server_ts) = ls
        .update_layer_relayed(&session_id, layer, oid, Some(delta), client_id, client_ts)
        .await
        .unwrap_or_else(|e| panic!("{path:?}: the first commit must succeed; got {e:?}"));
    assert_eq!(
        attempts_on(&metrics, path.label()),
        1,
        "{path:?}: the commit must be recorded on path {:?}",
        path.label()
    );

    let payload = update_stream_payload(oid, seq, client_id, client_ts, server_ts);
    assert_eq!(
        stream_commit_payloads(&tv, oid, before_seq).await,
        vec![(seq, payload.clone())],
        "{path:?}: the commit's one stream entry must carry the builder's payload"
    );
    assert_eq!(
        commit_chain(&tv, oid, &format!("the first {path:?}-path commit")).await,
        commit_chain_fold("", &payload),
        "{path:?}: an object without the column must fold from \"\" on its first commit"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_first_commit_on_object_without_commit_chain_folds_from_empty() {
    first_commit_on_object_without_commit_chain_folds_from_empty(
        UpdateLayerBackend::Lua,
        delta_with_set_field_inline(WORLD, "color", vec![0x2A], 1),
        CommitPath::Fast,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_structural_first_commit_on_object_without_commit_chain_folds_from_empty()
{
    first_commit_on_object_without_commit_chain_folds_from_empty(
        UpdateLayerBackend::Lua,
        create_section(WORLD, CUBE, "Cube"),
        CommitPath::Structural,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_first_commit_on_object_without_commit_chain_folds_from_empty() {
    first_commit_on_object_without_commit_chain_folds_from_empty(
        UpdateLayerBackend::Txn,
        delta_with_set_field_inline(WORLD, "color", vec![0x2A], 1),
        CommitPath::Watch,
    )
    .await;
}

/// Commit once on `backend` (so the column holds a fold, not the
/// materializer's value), then drive two refusals that each path raises after
/// its header read — a `set_order` overflow and a `SUPERSEDED` object — and
/// assert neither moves `commit_chain` or `seq`.
async fn refused_commits_leave_commit_chain_unchanged(backend: UpdateLayerBackend) {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, backend);
    let session_id = create_test_session(&tv).await;
    let layer = "commit-chain-refused";
    let oid = create_with_world(&ls, &session_id, layer).await;
    let header_key = tv.key_schema().obj_header(&oid.to_string());

    let refused_delta = || delta_with_set_field_inline(WORLD, "color", vec![0x2A], 1);
    let path = CommitPath::of(backend, &refused_delta());
    let materialized = commit_chain(&tv, oid, "CreateLayer").await;
    ls.update_layer_relayed(
        &session_id,
        layer,
        oid,
        Some(refused_delta()),
        "refusal-writer",
        1,
    )
    .await
    .unwrap_or_else(|e| panic!("{path:?}: the control commit must succeed; got {e:?}"));
    let folded = commit_chain(&tv, oid, &format!("the {path:?}-path control commit")).await;
    assert_ne!(
        folded, materialized,
        "{path:?}: fixture precondition — the control commit must fold the column"
    );
    let seq = header_seq(&tv, oid).await;

    // `set_order` overflow: `next_set_order + 1` crosses the ceiling.
    let _: () = tv
        .client()
        .pool
        .hset(
            &header_key,
            [("next_set_order", SET_ORDER_CEILING.to_string())],
        )
        .await
        .expect("HSET next_set_order must succeed");
    match ls
        .update_layer_relayed(
            &session_id,
            layer,
            oid,
            Some(refused_delta()),
            "refusal-writer",
            2,
        )
        .await
    {
        Err(ServiceError::Internal(_)) => {}
        other => panic!("{path:?}: a set_order overflow must be refused Internal; got {other:?}"),
    }
    assert_eq!(
        header_seq(&tv, oid).await,
        seq,
        "{path:?}: the set_order refusal must not advance seq"
    );
    assert_eq!(
        commit_chain(&tv, oid, &format!("a {path:?}-path set_order refusal")).await,
        folded,
        "{path:?}: a set_order refusal must leave commit_chain unchanged"
    );

    // `SUPERSEDED`: the header CreateLayer's overwrite marks before retiring it.
    let _: () = tv
        .client()
        .pool
        .hset(&header_key, [("superseded", "1")])
        .await
        .expect("HSET superseded=1 must succeed");
    match ls
        .update_layer_relayed(
            &session_id,
            layer,
            oid,
            Some(empty_delta()),
            "refusal-writer",
            3,
        )
        .await
    {
        Err(ServiceError::FailedPrecondition(_)) => {}
        other => panic!(
            "{path:?}: a commit to a superseded object must be refused FailedPrecondition; \
             got {other:?}"
        ),
    }
    assert_eq!(
        header_seq(&tv, oid).await,
        seq,
        "{path:?}: the superseded refusal must not advance seq"
    );
    assert_eq!(
        commit_chain(&tv, oid, &format!("a {path:?}-path superseded refusal")).await,
        folded,
        "{path:?}: a superseded refusal must leave commit_chain unchanged"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_refused_commits_leave_commit_chain_unchanged() {
    refused_commits_leave_commit_chain_unchanged(UpdateLayerBackend::Lua).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_refused_commits_leave_commit_chain_unchanged() {
    refused_commits_leave_commit_chain_unchanged(UpdateLayerBackend::Txn).await;
}
