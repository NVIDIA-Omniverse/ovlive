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

//! The layer-root rule on `UpdateLayer` (SDD §9.7): an update never takes a
//! layer from at least one top-level section (a section whose `parent_id` is
//! 0) to none. The rule is checked after the plan on both structural commit
//! routes, so every test here runs on both: the WATCH path (`txn` backend)
//! and the cached structural route (`lua` backend, with a structural cache).
//! Each call goes through `LayerStore::update_layer_with_operation`, the
//! production commit entry point, relaying the delta's OMPF; objects are
//! seeded through `LayerStore::create_layer`, which applies no root rule, so
//! a root-less layer stored before the rule can be staged.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`update_layer_removing_the_last_top_level_section_is_refused_and_writes_nothing`] | five deltas that leave a rooted layer with no top-level section — `DeleteSection` of the root, `MoveSection` of the root to a missing parent and into its own child, a `DiffChangeSection` re-parenting the root, a reset without a root — are refused `INVALID_ARGUMENT`, writing nothing (the object's whole keyspace and header unchanged: sequence, root row, `obj:ch:0`, stream, operation record); on `lua` the cache entry put back equals a fresh read of the stored object, and on `txn` no entry is made |
//! | [`update_layer_keeping_a_top_level_section_is_accepted`] | deleting one of two top-level sections, and a reset recreating a top-level section (the root itself, or another id), commit |
//! | [`update_layer_on_a_rootless_layer_is_never_refused_for_it`] | on a layer stored with no top-level section, updates that leave it with none (an orphan row added, an empty reset) commit: it had none to lose |
//! | `update_layer_retried_after_another_writer_adds_a_root_is_judged_on_the_retrys_state` | the rule judges the state each attempt plans on: an empty reset on a root-less layer passes its first attempt's check, another replica's commit adds the root before that attempt's commit is dispatched, and the retry is refused, leaving the root; needs `test-fault-injection` |
//!
//! Elsewhere: the refusal through the `UpdateLayer` handler on a chained call
//! carrying `INVALID_UPDATE` in `grpc::chained_writes_test`; the rule's
//! predicate and check in `src/domain/layer_root.rs`; the Kit capture
//! replayed on both backends with every delta accepted in
//! `grpc::subscribe_regimes_test`.
//!
//! # Wait budget
//!
//! Every call is under [`CALL_BUDGET`] (10 s). The refusal test makes ten
//! calls, five per backend, one fixture per case: 100 s with every call at its
//! budget, under nextest's 120 s hang-killer. The other tests make three calls
//! per backend (60 s), except the per-attempt test, which makes one call per
//! backend, the other replica's commit running inside that call's budget
//! (20 s). The healthy path takes a few milliseconds per call.

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::HashesInterface;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, SetField, StructCommand};

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::delta::apply_delta;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::storage::layer::{operation_request_digest, LayerStore, Operation, UpdateCommit};
use live_edit::storage::structural_cache::{CacheLookup, StructuralCache};
use live_edit::storage::test_support::structural_commit::{
    dump_object_keyspace, CommitTokens, ObjectKeyspaceDump,
};
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::storage::snapshot_read_atomic_test::{
    create, create_cmd, create_session, delete_cmd, move_cmd, structural,
};
use crate::storage::structural_cache_cold_start_test::{header, normalized_fresh_state};

/// Deadline on one `update_layer_with_operation` call: a few round trips when
/// healthy.
const CALL_BUDGET: Duration = Duration::from_secs(10);

/// A structural cache budget no fixture object comes near.
const BUDGET_BYTES: u64 = 64 * 1024 * 1024;

/// The writer of every call.
const WRITER: &str = "root-rule-writer";
const CLIENT_TS: u64 = 5_000;

/// The layer's root section, as Kit and the service create it.
const ROOT_SECTION: u64 = 1;
/// The root's child.
const CHILD: u64 = 2;
/// A second top-level section.
const OTHER_TOP: u64 = 9;
/// A parent id no section holds.
const MISSING_PARENT: u64 = 99;
/// An orphan row's section id, under [`MISSING_PARENT`].
const ORPHAN: u64 = 5;
/// The section type Kit gives a layer's pseudo-root.
const ROOT_TYPE: u8 = 7;

/// Both structural commit routes: `txn` takes the WATCH path for every call,
/// `lua` the cached structural route for a structural or reset call.
const BACKENDS: [UpdateLayerBackend; 2] = [UpdateLayerBackend::Txn, UpdateLayerBackend::Lua];

/// The refusal's message, as `From<DeltaError>` words it.
const REFUSAL_FRAGMENT: &str = "remove the layer's last top-level section";

/// One object on a store of one backend, with its own structural cache.
struct Fx {
    tv: TestValkey,
    cache: StructuralCache,
    ls: LayerStore,
    backend: UpdateLayerBackend,
    session_id: String,
    layer: String,
    oid: u64,
}

impl Fx {
    /// `state` created through `LayerStore::create_layer` on a store of
    /// `backend` holding its own structural cache.
    async fn new(backend: UpdateLayerBackend, label: &str, state: &ObjectState) -> Self {
        let tv = TestValkey::setup().await;
        let cache = StructuralCache::new(BUDGET_BYTES);
        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(backend)
            .with_structural_cache(cache.clone());
        let session_id = create_session(&tv, label).await;
        let oid = create(&ls, &session_id, label, state).await;
        Self {
            tv,
            cache,
            ls,
            backend,
            session_id,
            layer: label.to_owned(),
            oid,
        }
    }

    /// One `update_layer_with_operation` call by [`WRITER`], carrying
    /// operation `operation_id` and relaying the delta's OMPF, under
    /// [`CALL_BUDGET`].
    async fn update(
        &self,
        delta: OwnedDelta,
        operation_id: &str,
    ) -> Result<UpdateCommit, ServiceError> {
        let ompf = delta_to_ompf_bytes(&delta, &[]);
        let operation = Operation::new(operation_id, operation_request_digest(&self.layer, &ompf));
        tokio::time::timeout(
            CALL_BUDGET,
            self.ls.update_layer_with_operation(
                &self.session_id,
                &self.layer,
                self.oid,
                std::sync::Arc::new(Some(delta)),
                Some(Bytes::from(ompf)),
                WRITER,
                CLIENT_TS,
                Some(operation),
            ),
        )
        .await
        .expect("update_layer_with_operation did not settle within the call budget")
    }

    /// The object's whole keyspace (commit-token records verbatim) and its
    /// header in full.
    async fn keyspace(&self) -> (ObjectKeyspaceDump, HashMap<String, String>) {
        let dump = dump_object_keyspace(
            self.tv.client(),
            self.tv.key_schema(),
            self.oid,
            CommitTokens::Verbatim,
        )
        .await;
        (dump, header(&self.tv, self.oid).await)
    }

    /// The header's `seq`.
    async fn seq(&self) -> u64 {
        let seq: String = self
            .tv
            .client()
            .pool
            .hget(
                self.tv.key_schema().obj_header(&self.oid.to_string()),
                "seq",
            )
            .await
            .expect("HGET obj:hdr seq must succeed");
        seq.parse().expect("the header seq is decimal")
    }

    /// Assert the structural cache after a refused call: on `lua` the entry
    /// put back stands at the header's `seq` and equals a fresh read of the
    /// stored object (the plan was rolled back); on `txn` the call made no
    /// entry.
    async fn assert_cache_after_refusal(&self, case: &str) {
        match (self.backend, self.cache.lookup(self.oid)) {
            (UpdateLayerBackend::Lua, CacheLookup::Hit(guard)) => {
                let entry = guard.entry().clone();
                drop(guard);
                assert_eq!(
                    entry.seq,
                    self.seq().await,
                    "{case}: the entry put back must stand at the header's seq"
                );
                assert_eq!(
                    entry.state,
                    normalized_fresh_state(&self.tv, self.oid).await,
                    "{case}: the entry put back must equal a fresh read of the stored object"
                );
            }
            (UpdateLayerBackend::Txn, CacheLookup::Miss) => {}
            (backend, other) => panic!(
                "{case}: unexpected cache state on {backend:?}: {}",
                match other {
                    CacheLookup::Hit(_) => "a hit".to_owned(),
                    CacheLookup::Miss => "a miss".to_owned(),
                    CacheLookup::Uncacheable { reason } => format!("uncacheable ({reason:?})"),
                }
            ),
        }
    }
}

/// The state `delta` builds on a fresh object, at sequence 1.
fn state_of(delta: OwnedDelta) -> ObjectState {
    let mut state = ObjectState::default();
    apply_delta(&mut state, Some(delta)).expect("the fixture content applies");
    state
}

/// A `CreateSection` of a section of `section_type` under `parent_id`.
fn create_typed(parent_id: u64, section_id: u64, name: &str, section_type: u8) -> StructCommand {
    match create_cmd(parent_id, section_id, name) {
        StructCommand::CreateSection(mut c) => {
            c.section_type = section_type;
            StructCommand::CreateSection(c)
        }
        _ => unreachable!("create_cmd builds a CreateSection"),
    }
}

/// A `DiffChangeSection` of `section_id` under `parent_id`.
fn change_cmd(parent_id: u64, section_id: u64, name: &str, section_type: u8) -> StructCommand {
    StructCommand::DiffChangeSection(CommandDiffChangeSection {
        parent_id,
        section_id,
        section_name: name.to_owned(),
        section_type,
        section_order: 0,
    })
}

/// A whole-layer reset (`is_diff`, `base_version` 0) of `commands`.
fn reset(commands: Vec<StructCommand>) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        struct_commands: commands,
        ..Default::default()
    }
}

/// An inline field `key` on `section_id`.
fn field(section_id: u64, key: &str) -> SetField {
    SetField {
        section_id,
        key_name: key.to_owned(),
        value_or_ext_hash: b"value".to_vec(),
        ..SetField::default()
    }
}

/// A rooted layer, as Kit authors one: the root (section 1, type 7) holding a
/// field, and one child holding a field.
fn rooted_state() -> ObjectState {
    let mut content = structural(vec![
        create_typed(0, ROOT_SECTION, "", ROOT_TYPE),
        create_cmd(ROOT_SECTION, CHILD, "prim"),
    ]);
    content.set_fields = vec![field(ROOT_SECTION, "metadata"), field(CHILD, "attr")];
    state_of(content)
}

/// [`rooted_state`] plus a second top-level section.
fn two_top_level_state() -> ObjectState {
    state_of(structural(vec![
        create_typed(0, ROOT_SECTION, "", ROOT_TYPE),
        create_cmd(ROOT_SECTION, CHILD, "prim"),
        create_cmd(0, OTHER_TOP, "other"),
    ]))
}

/// A layer stored with no top-level section: one orphan row under a missing
/// parent, holding a field.
fn rootless_state() -> ObjectState {
    let mut content = structural(vec![change_cmd(MISSING_PARENT, ORPHAN, "orphan", 0)]);
    content.set_fields = vec![field(ORPHAN, "attr")];
    let state = state_of(content);
    assert!(
        state.sections.contains_key(&ORPHAN) && !state.children.contains_key(&0),
        "fixture: the orphan row is stored and nothing is listed under the root"
    );
    state
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_removing_the_last_top_level_section_is_refused_and_writes_nothing() {
    let cases: [(&str, OwnedDelta); 5] = [
        (
            "DeleteSection of the root",
            structural(vec![delete_cmd(0, ROOT_SECTION)]),
        ),
        (
            "MoveSection of the root to a missing parent",
            structural(vec![move_cmd(ROOT_SECTION, 0, MISSING_PARENT, "")]),
        ),
        (
            "MoveSection of the root into its own child",
            structural(vec![move_cmd(ROOT_SECTION, 0, CHILD, "")]),
        ),
        (
            "DiffChangeSection re-parenting the root",
            structural(vec![change_cmd(
                MISSING_PARENT,
                ROOT_SECTION,
                "",
                ROOT_TYPE,
            )]),
        ),
        ("a reset without a root", reset(Vec::new())),
    ];
    for backend in BACKENDS {
        for (i, (case, delta)) in cases.iter().enumerate() {
            let case = format!("{backend:?}: {case}");
            let fx = Fx::new(backend, &format!("root-refused-{i}"), &rooted_state()).await;
            let before = fx.keyspace().await;
            assert!(
                before.0.get("obj:ch:0").is_some() && before.0.get("obj:sec:1").is_some(),
                "{case}: fixture: the root row and the list under 0 are stored, or the \
                 unchanged-keyspace check below could not see them go"
            );

            let refused = match fx.update(delta.clone(), &format!("root-op-{i}")).await {
                Err(e) => e,
                Ok(commit) => panic!(
                    "{case}: a delta removing the last top-level section must be refused; it \
                     committed {commit:?}"
                ),
            };

            let status: tonic::Status = refused.into();
            assert_eq!(
                status.code(),
                tonic::Code::InvalidArgument,
                "{case}: got {status:?}"
            );
            assert!(
                status.message().contains(REFUSAL_FRAGMENT),
                "{case}: the refusal must name the layer-root rule, got: {}",
                status.message()
            );
            assert_eq!(
                fx.keyspace().await,
                before,
                "{case}: a refused update must write nothing: sequence, rows, obj:ch:0, \
                 stream and operation record all unchanged"
            );
            fx.assert_cache_after_refusal(&case).await;
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_keeping_a_top_level_section_is_accepted() {
    for backend in BACKENDS {
        let fx = Fx::new(backend, "root-kept-two", &two_top_level_state()).await;
        let commit = fx
            .update(structural(vec![delete_cmd(0, ROOT_SECTION)]), "kept-1")
            .await
            .unwrap_or_else(|e| {
                panic!("{backend:?}: deleting one of two top-level sections must commit: {e:?}")
            });
        assert_eq!(commit.seq, 2, "{backend:?}: the delete committed");

        let fx = Fx::new(backend, "root-kept-reset", &rooted_state()).await;
        let resets: [(&str, OwnedDelta); 2] = [
            (
                "a reset recreating the root",
                reset(vec![change_cmd(0, ROOT_SECTION, "", ROOT_TYPE)]),
            ),
            (
                "a reset creating another top-level section",
                reset(vec![change_cmd(0, OTHER_TOP, "other", ROOT_TYPE)]),
            ),
        ];
        for (n, (case, delta)) in resets.into_iter().enumerate() {
            let commit = fx
                .update(delta, &format!("kept-reset-{n}"))
                .await
                .unwrap_or_else(|e| panic!("{backend:?}: {case} must commit: {e:?}"));
            assert_eq!(commit.seq, n as u64 + 2, "{backend:?}: {case} committed");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_on_a_rootless_layer_is_never_refused_for_it() {
    for backend in BACKENDS {
        let fx = Fx::new(backend, "rootless", &rootless_state()).await;
        let updates: [(&str, OwnedDelta); 3] = [
            (
                "another orphan row",
                structural(vec![change_cmd(MISSING_PARENT, ORPHAN + 1, "orphan-2", 0)]),
            ),
            (
                "the orphan's row deleted",
                structural(vec![delete_cmd(MISSING_PARENT, ORPHAN)]),
            ),
            ("an empty reset", reset(Vec::new())),
        ];
        let mut seqs = BTreeMap::new();
        for (n, (case, delta)) in updates.into_iter().enumerate() {
            let commit = fx
                .update(delta, &format!("rootless-{n}"))
                .await
                .unwrap_or_else(|e| {
                    panic!("{backend:?}: {case} on a root-less layer must commit: {e:?}")
                });
            seqs.insert(n, commit.seq);
        }
        assert_eq!(
            seqs.into_values().collect::<Vec<_>>(),
            [2, 3, 4],
            "{backend:?}: every update committed in turn"
        );
    }
}

/// The rule judges the state each attempt plans on, not the state the call
/// first read. An empty reset on a root-less layer plans and passes the check
/// (the layer had no top-level section to lose); before its commit is sent,
/// another replica's commit adds the root, so the first attempt's commit is
/// refused (`STALE` on `lua`, a WATCH-cancelled `EXEC` on `txn`) and the call
/// retries. The retry plans on the state holding the root, so the reset would
/// erase it, and is refused `INVALID_ARGUMENT`. A check that kept the first
/// attempt's answer for the whole call would commit that reset and erase the
/// root the other writer had just added.
///
/// The other replica's commit is landed from the route's own test hook, which
/// runs after the attempt has planned (and checked) and before its commit is
/// dispatched: on `lua` the structural route's pre-dispatch hook (before its
/// commit `EVALSHA`), on `txn` the WATCH path's (before its `EXEC`). The hook
/// fires once per dispatched commit, so its single firing also shows the retry
/// was refused before any commit work.
///
/// The hooks exist only under `test-fault-injection`, so this test is compiled
/// out without it.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_retried_after_another_writer_adds_a_root_is_judged_on_the_retrys_state() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use futures::FutureExt;
    use live_edit::storage::watched_txn::WatchWindowHook;

    /// The writer of the commit another replica lands between the reset's
    /// plan and its commit.
    const OTHER_WRITER: &str = "root-rule-other-writer";

    for backend in BACKENDS {
        let case = format!("{backend:?}");
        let fx = Fx::new(backend, "root-per-attempt", &rootless_state()).await;

        // One shot: the first dispatch lands the other replica's commit of
        // the root, through a store of its own on the same Valkey; a later
        // firing (none is expected) lands nothing.
        let fired = Arc::new(AtomicUsize::new(0));
        let hook: WatchWindowHook = {
            let fired = Arc::clone(&fired);
            let (client, keys) = (fx.tv.client().clone(), fx.tv.key_schema().clone());
            let (session_id, layer, oid) = (fx.session_id.clone(), fx.layer.clone(), fx.oid);
            Arc::new(move |_dispatch_client| {
                let first = fired.fetch_add(1, Ordering::SeqCst) == 0;
                let (client, keys) = (client.clone(), keys.clone());
                let (session_id, layer) = (session_id.clone(), layer.clone());
                async move {
                    if !first {
                        return;
                    }
                    let other = LayerStore::new(client, keys, 4096)
                        .with_update_layer_backend(UpdateLayerBackend::Txn);
                    let root = structural(vec![create_typed(0, ROOT_SECTION, "", ROOT_TYPE)]);
                    let relay = Bytes::from(delta_to_ompf_bytes(&root, &[]));
                    tokio::time::timeout(
                        CALL_BUDGET,
                        other.update_layer(
                            &session_id,
                            &layer,
                            oid,
                            Some(root),
                            Some(relay),
                            OTHER_WRITER,
                            CLIENT_TS,
                        ),
                    )
                    .await
                    .expect("the other replica's commit did not settle within the call budget")
                    .expect("the other replica's commit of the root must land");
                }
                .boxed()
            })
        };
        let fx = Fx {
            ls: match backend {
                UpdateLayerBackend::Lua => fx
                    .ls
                    .with_structural_commit_pre_dispatch_hook_for_test(hook),
                _ => fx.ls.with_pre_dispatch_hook_for_test(hook),
            },
            ..fx
        };

        let refused = match fx.update(reset(Vec::new()), "per-attempt-reset").await {
            Err(e) => e,
            Ok(commit) => panic!(
                "{case}: the retried reset would erase the root another writer added; it \
                 committed {commit:?}"
            ),
        };
        assert_eq!(
            fired.load(Ordering::SeqCst),
            1,
            "{case}: the first attempt reached its commit (the other writer's root landed \
             there), and the retry was refused before dispatching one"
        );
        let status: tonic::Status = refused.into();
        assert_eq!(
            status.code(),
            tonic::Code::InvalidArgument,
            "{case}: got {status:?}"
        );
        assert!(
            status.message().contains(REFUSAL_FRAGMENT),
            "{case}: the refusal must name the layer-root rule, got: {}",
            status.message()
        );

        assert_eq!(
            fx.seq().await,
            2,
            "{case}: only the other writer's commit landed"
        );
        let (keyspace, _) = fx.keyspace().await;
        assert!(
            keyspace.get(&format!("obj:sec:{ROOT_SECTION}")).is_some()
                && keyspace.get(&format!("obj:sec:{ORPHAN}")).is_some(),
            "{case}: the root the other writer added and the orphan row both survive"
        );
        fx.assert_cache_after_refusal(&case).await;
    }
}
