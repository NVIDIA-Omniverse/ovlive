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

//! The unfenced bounded fan-out — `storage::diff_read::read_object_snapshot_slim`
//! for the structural `UpdateLayer` consumer, the fan-out's one reader (b.y3m
//! Epic 2 Task 5; Spec SDD SR-4.3, SR-1.5, SR-11.3). The structural
//! `UpdateLayer` consumer is the WATCH path's: under the default `lua` backend
//! a structural delta plans on the structural cache instead, unless the cache
//! cannot hold the object. (The ext-GC snapshot read, which read this fan-out
//! whole, is gone since b.fsz MR-D4: external-value clean-up walks a retiring
//! object's references instead, `storage::retirement`, pinned in
//! `retirement_walk_test.rs` and `retirement_budget_test.rs`.)
//!
//! The consumer never runs the atomic script or takes a fence: at every size
//! the read is round trips on the snapshot lane (the header and the
//! indices; every hash and the children; a marked object's lists), with
//! the outstanding commands and reply bytes windowed, reading the atomic read's
//! key set under the same marked / unmarked children rule (SR-1.2). Its round
//! trips are plain commands for round 1, a lone children list whose modelled
//! shard time alone exceeds τ, and every window after a script fallback; each
//! other window of its rounds 2 and 3 is one `EVALSHA` of the per-window read
//! script (`fanout_window_read.lua`, b.gvf), which issues the plain window's
//! commands inside one script call. Nothing
//! proves its state is the state at one `seq`; the structural path proves it
//! itself afterwards, and those torn-read tests are inline in
//! `src/storage/layer.rs`, beside the `#[cfg(test)]` seams they need. What is
//! pinned here:
//!
//! | Area | What it proves |
//! |---|---|
//! | Equality with the atomic read | on a quiescent object — created, and after structural and fast-path commits, marked and unmarked; the marked object with a list its index does not name; the unmarked object with a partial index — the slim fan-out returns the compact projection of the atomic read's state, its cut, header timestamps and tallies, on `path="unfenced"` |
//! | Small windows | at a one-command window with a one-byte target, every entry over the target, the read still completes and equals the atomic read |
//! | Never the script | at the default limits and above the atomic ceiling, no `EVALSHA` of `object_snapshot_read.lua` names the object and the answer is never `TOO_LARGE`; nor does a WATCH-path structural commit (`lua-watch-structural`) or a `DeleteLayer` (whose clean-up walks the object's references) through a store above the ceiling run it |
//! | A header without a numeric `seq` (SR-1.5) | `NotFound`, as is an absent header |
//! | Refusals | a wrong-typed key of each of the eleven families the fan-out reads ticks exactly its own `key_family` cell and names its own read; keys it never reads tick nothing |
//! | Metrics | one `live_edit_snapshot_read_total{path="unfenced"}` tick per read under its consumer and outcome, none on `path="atomic"` and none under `ext_gc`; the duration histogram under the path, the size histograms with the read's own tallies |
//! | Round 1 in pages | indices that each take several pages (12,000 sections, 30,000 fields): the read equals the atomic read, every round-1 window within `FANOUT_INDEX_WINDOW_MEMBERS` (the `fan-out window` event); `SSCAN` steps that repeat members (a reply-rewriting proxy) leave each member counted once |
//! | Fault seam (`test-fault-injection`) | an injected transient failure is `Aborted`, a permanent one `Internal`, each classified as a real failure of the command it replaced; on a script window an injected value-row error fails under `HSTRLEN field` with no `EVALSHA` sent, and an injected cluster-state refusal falls the read back to plain commands for its later windows (`live_edit_snapshot_read_window_fallback_total` +1, no row reported as holding both shapes) |
//! | The window script (b.gvf) | on a layer of 1,000 field rows the client issues no value-row or list-length command; each round-2/3 window is one `EVALSHA` of the window script, its one key the header; round 1 is plain. A lone children list over τ is one plain `LRANGE`. An in-script `WRONGTYPE` ticks its own family. A store-side cluster-state refusal tag falls the read back to plain commands for that window and every later one, ticking the fallback counter once; the next read tries the script again. A reply not of the script's shape fails `Internal`, attributed to the header's `EVALSHA`. The store's not-serving `-BUSY` / `-LOADING` on the `EVALSHA` fail the read retryable (`Aborted`, uncounted, no fallback; b.xd1). Every other error reply fails `Internal` without falling back: a refusal tag whose entry is out of range or whose command is foreign to its entry (even with a cluster-state text), and `FANOUT_WINDOW_ARGS` are attributed to the header's `EVALSHA`; a tag on an entry's own command with any other error is that entry's refusal. A `SCRIPT FLUSH` between windows (fault seam) costs the read one `SCRIPT LOAD`, and it completes |
//!
//! The windows' own mechanics — no window issued before the previous window's
//! replies are decoded, the byte target's average and its empty-entry
//! fallback, an over-target entry completing and the windows then shrinking to
//! one command, the bounds taken from the configured limits — are pinned against Valkey inline in
//! `src/storage/diff_read.rs`, where the private window type can be built with
//! chosen bounds.
//!
//! The fault-seam module is `#[cfg(feature = "test-fault-injection")]` and is
//! compiled out — not skipped — without that feature.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface, ListInterface};
use oodf_ffi::OwnedDelta;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::CompactState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::diff_read::{
    read_object_snapshot_slim, AtomicSnapshotOf, SlimSnapshotReadConsumer, SnapshotConsumer,
    SnapshotReadOf, SnapshotReadPath, WindowTransport,
};
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::lua::{FANOUT_WINDOW_READ_SCRIPT, OBJECT_SNAPSHOT_READ_SCRIPT};
use live_edit::storage::test_support::{compact_snapshot, read_object_state_for_test, TestValkey};

use crate::common::owned_delta::{
    delta_with_set_field_external, delta_with_set_field_inline, delta_with_time_sample_inline,
};
use crate::storage::b_atj_tests::{
    assert_keyspace_type_errors_flat, assert_no_family_ticked, assert_only_family_ticked,
    create_section_delta, delete_section_delta, Fixture, DOOMED_SECTION_ID, FIELD_NAME,
    SAMPLE_TIME, SECTION_ID,
};
use crate::storage::b_muu_tests::assert_no_refusal_kind_ticked;
use crate::storage::snapshot_read_atomic_test::{
    commit, create, create_cmd, create_session, delete_cmd, every_shape_item_bound,
    every_shape_state, ext_hash, indexed_parents, layer_store, listed_parents, mock_ext_store,
    move_cmd, structural, unmark, CUBE, DANGLING_PARENT, ROOT, SIBLING, WORLD,
};
use crate::storage::snapshot_read_support::{
    client_with_limits, monitor_fence, read_atomic, test_valkey_addr, MonitorTap, MonitoredCommand,
};

// ═════════════════════════════════════════════════════════════════════════════
// Reads and metric readers
// ═════════════════════════════════════════════════════════════════════════════

/// The read for an unfenced `consumer`, which must answer with the object's
/// state — never `TOO_LARGE`, which the fan-out has no ceiling to answer.
async fn read_unfenced<C: UnfencedConsumer>(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    consumer: C,
) -> AtomicSnapshotOf<C::State> {
    match consumer.read(client, keys, &object_id.to_string()).await {
        Ok(SnapshotReadOf::Snapshot(snapshot)) => *snapshot,
        Ok(SnapshotReadOf::TooLarge(too_large)) => panic!(
            "{consumer:?} reads through the unfenced fan-out, which reads every size: object \
             {object_id} must answer with its state, not TOO_LARGE {too_large:?}"
        ),
        Err(e) => panic!("the {consumer:?} read of object {object_id} must succeed; got {e:?}"),
    }
}

/// The read for an unfenced `consumer`, which must fail.
async fn read_unfenced_err<C: UnfencedConsumer>(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    consumer: C,
) -> ServiceError {
    match consumer.read(client, keys, object_id).await {
        Err(e) => e,
        Ok(SnapshotReadOf::Snapshot(_)) => {
            panic!("the {consumer:?} read of object {object_id} must fail; it answered with state")
        }
        Ok(SnapshotReadOf::TooLarge(t)) => panic!(
            "the {consumer:?} read of object {object_id} must fail; it answered TOO_LARGE {t:?}"
        ),
    }
}

/// A consumer these helpers read for: the structural consumer, through
/// `read_object_snapshot_slim` (the unfenced fan-out's one reader since the
/// ext-GC snapshot read was removed).
trait UnfencedConsumer: SnapshotConsumer {
    async fn read(
        self,
        client: &ValkeyClient,
        keys: &KeySchema,
        object_id: &str,
    ) -> Result<SnapshotReadOf<Self::State>, ServiceError>;
}

impl UnfencedConsumer for SlimSnapshotReadConsumer {
    async fn read(
        self,
        client: &ValkeyClient,
        keys: &KeySchema,
        object_id: &str,
    ) -> Result<SnapshotReadOf<Self::State>, ServiceError> {
        read_object_snapshot_slim(client, keys, object_id, self).await
    }
}

/// `live_edit_snapshot_read_total{consumer, path, outcome}`. Panics on an
/// absent cell: every cell a built path can tick is zero-seeded, so absence
/// means the handle never reached the read.
fn snapshot_reads(metrics: &Metrics, consumer: &str, path: &str, outcome: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_snapshot_read_total",
        &[("consumer", consumer), ("path", path), ("outcome", outcome)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_snapshot_read_total{{consumer=\"{consumer}\",path=\"{path}\",outcome=\
             \"{outcome}\"}} is absent — it is zero-seeded, so this handle never reached the read"
        )
    }) as u64
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Equality with the atomic read on quiescent objects (SR-1.2)
// ═════════════════════════════════════════════════════════════════════════════
//
// The fan-out reads the same keys as the atomic read under the same children
// rule, so on an object nothing is writing the two must agree on everything a
// consumer reads and on the tallies the size metrics and Epic 3's intent sizing
// read (SR-5.4). The atomic read is the oracle here: its own equality with
// the reference reader (`read_object_state_for_test`) is
// `snapshot_read_atomic_test`'s.

/// Assert the structural WATCH consumer's slim fan-out of `oid` through
/// `fanout_client` equals the slim atomic read through `tv`'s client and the
/// compact projection of the full atomic read — on a quiescent object the
/// same keys under the same children rule (SR-1.2) — and return it.
async fn assert_unfenced_equals_atomic(
    fanout_client: &ValkeyClient,
    tv: &TestValkey,
    oid: u64,
    context: &str,
) -> AtomicSnapshotOf<CompactState> {
    let atomic = read_atomic(tv.client(), tv.key_schema(), oid).await;

    // The structural WATCH path's slim fan-out (`StructuralUpdate`, the
    // fan-out decoded into the compact form) against the slim atomic read
    // (the cold start's `StructuralCache`) and the compact projection of the
    // full atomic read: one compact state, one cut, one set of tallies.
    let slim_unfenced = read_unfenced(
        fanout_client,
        tv.key_schema(),
        oid,
        SlimSnapshotReadConsumer::StructuralUpdate,
    )
    .await;
    let slim_atomic = match read_object_snapshot_slim(
        tv.client(),
        tv.key_schema(),
        &oid.to_string(),
        SlimSnapshotReadConsumer::StructuralCache,
    )
    .await
    {
        Ok(SnapshotReadOf::Snapshot(snapshot)) => *snapshot,
        other => panic!(
            "{context}: the slim atomic read must answer with state; got {:?}",
            other.map(|_| ())
        ),
    };
    assert_eq!(
        (slim_unfenced.path, slim_atomic.path),
        (SnapshotReadPath::Unfenced, SnapshotReadPath::Atomic),
        "{context}: fixture: the WATCH consumer's fan-out and the cold start's atomic read"
    );
    assert_eq!(
        slim_atomic.state,
        compact_snapshot(&atomic.state),
        "{context}: the slim atomic read is the compact projection of the full atomic read"
    );
    assert_eq!(
        slim_unfenced.state, slim_atomic.state,
        "{context}: the slim fan-out returns exactly the slim atomic read's compact state"
    );
    assert_eq!(
        (
            slim_unfenced.cut,
            slim_unfenced.last_server_ts,
            slim_unfenced.last_client_ts,
            slim_unfenced.item_count,
            slim_unfenced.reply_bytes,
            slim_unfenced.entry_count,
        ),
        (
            slim_atomic.cut,
            slim_atomic.last_server_ts,
            slim_atomic.last_client_ts,
            atomic.item_count,
            atomic.reply_bytes,
            atomic.entry_count,
        ),
        "{context}: the slim fan-out's cut, timestamps and (items, bytes, entries) are the atomic \
         reads'"
    );
    assert_eq!(
        (
            slim_atomic.item_count,
            slim_atomic.reply_bytes,
            slim_atomic.entry_count
        ),
        (atomic.item_count, atomic.reply_bytes, atomic.entry_count),
        "{context}: the slim atomic read tallies what the full one does"
    );
    slim_unfenced
}

async fn equality_on_a_created_object(marked: bool) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-eq-create";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    if !marked {
        unmark(&tv, oid).await;
    }
    assert!(
        listed_parents(&tv, oid).await.contains(&DANGLING_PARENT),
        "fixture precondition: the object must own a list under a parent that names no section, \
         or the dangling-parent half of the equality is vacuous"
    );

    let unfenced = assert_unfenced_equals_atomic(
        tv.client(),
        &tv,
        oid,
        if marked {
            "created, marked"
        } else {
            "created, unmarked"
        },
    )
    .await;
    assert!(
        !unfenced.state.children.contains_key(&DANGLING_PARENT),
        "a list under a dangling parent stays out of the snapshot — on a marked object that \
         list's parent IS in `obj:chs`, so the fan-out must still filter it against the live \
         sections"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_equals_the_atomic_read_on_a_created_marked_object() {
    equality_on_a_created_object(true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_equals_the_atomic_read_on_a_created_unmarked_object() {
    equality_on_a_created_object(false).await;
}

/// A structural `CreateSection`, a `MoveSection` that empties one list and
/// starts another, a `DeleteSection` that writes a tombstone, then fast-path
/// inline and external rows: every key family the fan-out reads, populated by
/// the object's own writes. Shared with `snapshot_read_fenced_test`, whose
/// fenced pass reads the same keys.
pub(crate) fn commit_steps() -> Vec<(&'static str, OwnedDelta)> {
    const NEW: u64 = 5;
    vec![
        (
            "CreateSection under a leaf",
            structural(vec![create_cmd(CUBE, NEW, "New")]),
        ),
        (
            "MoveSection of WORLD's only child to SIBLING",
            structural(vec![move_cmd(CUBE, WORLD, SIBLING, "Cube")]),
        ),
        (
            "DeleteSection of the now childless WORLD",
            structural(vec![delete_cmd(ROOT, WORLD)]),
        ),
        (
            "fast-path inline field",
            delta_with_set_field_inline(SIBLING, "label", vec![0x2A], 5),
        ),
        (
            "fast-path external field",
            delta_with_set_field_external(CUBE, "mesh", ext_hash(0xEE).to_vec(), 9_000, 1, 6),
        ),
        (
            "fast-path inline time sample",
            delta_with_time_sample_inline(NEW, 3.25, vec![0x07]),
        ),
    ]
}

/// A created object taken through [`commit_steps`], marked or not: the
/// fixture and the object's id.
async fn committed_object(layer: &str, marked: bool) -> (TestValkey, u64) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    if !marked {
        unmark(&tv, oid).await;
    }
    for (_, delta) in commit_steps() {
        commit(&ls, &session_id, layer, oid, delta).await;
    }
    let atomic = read_atomic(tv.client(), tv.key_schema(), oid).await;
    assert!(
        atomic
            .state
            .tombstones
            .iter()
            .any(|(sid, _, _)| *sid == WORLD),
        "fixture precondition: the DeleteSection must have written WORLD's tombstone, or the \
         tombstone family is never compared"
    );
    (tv, oid)
}

async fn equality_after_commits(marked: bool) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-eq-commits";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    if !marked {
        unmark(&tv, oid).await;
    }
    for (what, delta) in commit_steps() {
        commit(&ls, &session_id, layer, oid, delta).await;
        assert_unfenced_equals_atomic(
            tv.client(),
            &tv,
            oid,
            &format!(
                "after {what} ({})",
                if marked { "marked" } else { "unmarked" }
            ),
        )
        .await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_equals_the_atomic_read_after_structural_and_fast_path_commits_marked() {
    equality_after_commits(true).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_equals_the_atomic_read_after_structural_and_fast_path_commits_unmarked() {
    equality_after_commits(false).await;
}

/// The marked children rule's discriminating shape (SR-1.2): a list under a
/// live section that a marked object's `obj:chs` does not name is not part of
/// the object, so the fan-out must not read it — exactly as the atomic read
/// does not, and unlike the reference reader (`read_object_state_for_test`),
/// which probes every live section. A fan-out that ignored the marker would
/// return the stray list.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_a_marked_object_reads_only_the_lists_its_children_index_names() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-marked-unindexed";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    // SIBLING is live and childless; give it an unindexed list.
    let _: i64 = tv
        .client()
        .pool
        .rpush(
            tv.key_schema()
                .obj_children(&oid.to_string(), &SIBLING.to_string()),
            vec![CUBE.to_string()],
        )
        .await
        .expect("RPUSH of the stray list must succeed");
    assert!(
        !indexed_parents(&tv, oid).await.contains(&SIBLING),
        "fixture precondition: the stray list must be unindexed"
    );
    let reference = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .unwrap_or_else(|e| panic!("the reference read must succeed; got {e:?}"));
    assert_eq!(
        reference.children.get(&SIBLING),
        Some(&vec![CUBE]),
        "fixture precondition: the reference read probes every live section, so it returns \
         the stray list"
    );

    let unfenced =
        assert_unfenced_equals_atomic(tv.client(), &tv, oid, "marked, stray unindexed list").await;
    assert!(
        !unfenced.state.children.contains_key(&SIBLING),
        "on a marked object the fan-out takes its lists from `obj:chs`, so a list it does not name \
         must be left out"
    );
}

/// The unmarked-partial-index regression (Spec SDD SR-1.7), on the fan-out: an
/// unmarked object's `obj:chs` may name only some of its lists, so the read
/// must probe the root plus every live section and return every list, not only
/// the indexed ones.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_an_unmarked_object_with_a_partial_children_index_returns_every_list() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-unmarked-partial";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    unmark(&tv, oid).await;
    let _: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().obj_children_index(&oid.to_string()))
        .await
        .expect("DEL obj:chs must succeed");
    const NEW: u64 = 5;
    commit(
        &ls,
        &session_id,
        layer,
        oid,
        structural(vec![create_cmd(CUBE, NEW, "New")]),
    )
    .await;
    assert_eq!(
        indexed_parents(&tv, oid).await,
        BTreeSet::from([CUBE]),
        "fixture precondition: `obj:chs` must name only the list the one `SetChildren` wrote"
    );

    let unfenced =
        assert_unfenced_equals_atomic(tv.client(), &tv, oid, "unmarked, partial obj:chs").await;
    assert_eq!(
        unfenced
            .state
            .children
            .keys()
            .copied()
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([ROOT, WORLD, CUBE]),
        "an unmarked object's `obj:chs` may be partial, so every list under the root and the live \
         sections must be read; trusting the index would drop the root's and WORLD's children \
         from every structural commit's view"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. Small windows (SR-11.3)
// ═════════════════════════════════════════════════════════════════════════════

/// The fan-out at its smallest: a one-item ceiling makes `FANOUT_WINDOW` one
/// command and a one-byte ceiling makes the byte target one byte, so every
/// entry is over the target and every window is one command. The read must
/// still complete — a window never admits fewer than one command — and return
/// exactly what the atomic read does at the default limits, on a marked and an
/// unmarked object carrying every key family.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_at_a_one_command_one_byte_window_completes_and_equals_the_atomic_read() {
    let narrow = client_with_limits(1, 1).await;
    for marked in [true, false] {
        let (tv, oid) = committed_object("sru-narrow", marked).await;
        assert_unfenced_equals_atomic(
            &narrow,
            &tv,
            oid,
            &format!(
                "one-command, one-byte windows ({})",
                if marked { "marked" } else { "unmarked" }
            ),
        )
        .await;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. Never the atomic script (SR-4.3)
// ═════════════════════════════════════════════════════════════════════════════
//
// What the read issued, from the store's own command feed. The structural path
// is the most frequent writer in an agent-built scene, so a script blocking
// the shard for the whole object per attempt would dominate the shard-delay
// allowance (its per-window script blocks the shard one window at a time, for
// at most τ by the recorded model, and is not matched here); and it cannot be
// answered `TOO_LARGE`, which no caller of its handles. The filter keys on the
// script's SHA and on this object's header key,
// so the feed's other traffic — every other test on the shared store included
// — cannot match it.

/// Whether `command` runs `object_snapshot_read.lua` against the object whose
/// header is `header_key`: an `EVALSHA` of its SHA, or an `EVAL` of its body
/// (which carries the reply format tag), naming that header.
fn runs_the_snapshot_script(command: &MonitoredCommand, header_key: &str) -> bool {
    let name = command.name();
    if !matches!(name.as_str(), "EVALSHA" | "EVALSHA_RO" | "EVAL" | "EVAL_RO") {
        return false;
    }
    let script = command.args.get(1).map(String::as_str).unwrap_or_default();
    let is_snapshot_script = script.eq_ignore_ascii_case(OBJECT_SNAPSHOT_READ_SCRIPT.sha1())
        || script.contains("object-snapshot-read-v1");
    is_snapshot_script && command.args.iter().any(|a| a == header_key)
}

/// The unfenced consumer never runs the script, at the default limits or
/// above the atomic ceiling, and is never answered `TOO_LARGE`.
///
/// Vacuity is closed from both sides. Each of the two reads must have read
/// the header itself, as a plain client `HGETALL`, so the reads ran and were
/// recorded; and a catch-up read afterwards, which DOES take the atomic path,
/// must be matched by the filter, so the filter can see a script run when there
/// is one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_reads_never_run_the_snapshot_script_below_or_above_the_atomic_ceiling() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-never-script";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let above_ceiling = client_with_limits(every_shape_item_bound(true) - 1, u64::MAX / 2).await;

    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    for client in [tv.client(), &above_ceiling] {
        let read = read_unfenced(
            client,
            tv.key_schema(),
            oid,
            SlimSnapshotReadConsumer::StructuralUpdate,
        )
        .await;
        assert_eq!(read.path, SnapshotReadPath::Unfenced);
    }
    let fence = monitor_fence(tv.prefix());
    let _: i64 = tv.client().pool.exists(&fence).await.unwrap();
    let seen = tap.drain_until(&fence).await;

    let script_runs: Vec<&MonitoredCommand> = seen
        .iter()
        .filter(|c| runs_the_snapshot_script(c, &header_key))
        .collect();
    assert!(
        script_runs.is_empty(),
        "the structural read must never run the atomic script, at any size (SR-4.3); script \
         runs naming this object: {script_runs:?}"
    );
    let header_reads = seen
        .iter()
        .filter(|c| {
            !c.issued_by_script()
                && c.name() == "HGETALL"
                && c.args.get(1).map(String::as_str) == Some(header_key.as_str())
        })
        .count();
    assert_eq!(
        header_reads, 2,
        "each of the two reads must have read the header as a plain command, or the feed did not \
         record them and the absence above is vacuous"
    );

    // The control: the filter does match a real script run.
    let fence = monitor_fence(tv.prefix());
    read_atomic(tv.client(), tv.key_schema(), oid).await;
    let _: i64 = tv.client().pool.exists(&fence).await.unwrap();
    let control = tap.drain_until(&fence).await;
    assert!(
        control
            .iter()
            .any(|c| runs_the_snapshot_script(c, &header_key)),
        "control: the catch-up read takes the atomic path, so the filter must match its script \
         run — otherwise the absence asserted above could not have failed"
    );
}

/// The same through the consumers themselves: a structural `UpdateLayer` on the
/// WATCH path and a `DeleteLayer` — whose clean-up walks the object's
/// references (`storage::retirement`) — through a store whose client is above
/// the atomic ceiling for this object run no atomic snapshot script on it, and
/// both succeed. The vacuity guard is that both reads ran: the structural
/// commit's read reads the section hash inside its round-2 window script
/// (`fanout_window_read.lua`, the feed's `lua` client), and the reference walk
/// checks the header with a plain `EXISTS`.
///
/// The store runs `lua-watch-structural`, whose structural deltas take the WATCH
/// path and its unfenced read. Under the default `lua` backend a structural
/// delta plans on the structural cache instead, whose cold start reads through
/// the atomic script (its raised-ceiling route above this ceiling).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_commit_and_delete_layer_never_run_the_snapshot_script_above_the_ceiling() {
    let tv = TestValkey::setup().await;
    let layer = "sru-never-script-e2e";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&layer_store(&tv), &session_id, layer, &every_shape_state()).await;
    let oid_str = oid.to_string();
    let header_key = tv.key_schema().obj_header(&oid_str);
    let sibling_key = tv.key_schema().obj_section(&oid_str, &SIBLING.to_string());
    let above_ceiling = client_with_limits(every_shape_item_bound(true) - 1, u64::MAX / 2).await;
    let store = LayerStore::new(above_ceiling, tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::LuaWatchStructural);

    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    commit(
        &store,
        &session_id,
        layer,
        oid,
        structural(vec![create_cmd(SIBLING, 5, "New")]),
    )
    .await;
    store
        .delete_layer(&session_id, layer, mock_ext_store())
        .await
        .expect("DeleteLayer must succeed");
    let fence = monitor_fence(tv.prefix());
    let _: i64 = tv.client().pool.exists(&fence).await.unwrap();
    let seen = tap.drain_until(&fence).await;

    let script_runs: Vec<&MonitoredCommand> = seen
        .iter()
        .filter(|c| runs_the_snapshot_script(c, &header_key))
        .collect();
    assert!(
        script_runs.is_empty(),
        "neither the structural commit nor the retirement's reference walk may run the atomic \
         script; script runs naming this object: {script_runs:?}"
    );
    let in_window_script = seen
        .iter()
        .filter(|c| {
            c.issued_by_script()
                && c.name() == "HGETALL"
                && c.args.get(1).map(String::as_str) == Some(sibling_key.as_str())
        })
        .count();
    let walk_header_checks = seen
        .iter()
        .filter(|c| {
            !c.issued_by_script()
                && c.name() == "EXISTS"
                && c.args.get(1).map(String::as_str) == Some(header_key.as_str())
        })
        .count();
    assert!(
        in_window_script >= 1 && walk_header_checks >= 1,
        "the structural commit's read (the section hash, inside its window script) and the \
         retirement's reference walk (a plain EXISTS of the header) must each have run; \
         {in_window_script} script reads of {sibling_key}, {walk_header_checks} EXISTS of \
         {header_key}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. A header without a numeric `seq` (SR-1.5)
// ═════════════════════════════════════════════════════════════════════════════

/// The two malformed shapes a header can carry `seq` in: absent, and present
/// but not a number.
const MALFORMED_SEQS: [Option<&str>; 2] = [None, Some("not-a-number")];

/// Strip `oid`'s `seq`, then put back `raw` if it is `Some`.
async fn set_malformed_seq(tv: &TestValkey, oid: u64, raw: Option<&str>) {
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let _: i64 = tv.client().pool.hdel(&header_key, "seq").await.unwrap();
    if let Some(raw) = raw {
        let _: () = tv
            .client()
            .pool
            .hset(&header_key, [("seq", raw)])
            .await
            .unwrap();
    }
}

/// For the structural consumer a header without a numeric `seq` is not a live
/// object: `NotFound`, `read_object_header`'s rule. The structural path's
/// WATCHed header read already refuses such a header earlier, so this is the
/// read's own guard rather than a path a commit reaches.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_read_of_a_header_without_a_numeric_seq_is_not_found() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-partial-structural";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;

    for raw in MALFORMED_SEQS {
        set_malformed_seq(&tv, oid, raw).await;
        let err = read_unfenced_err(
            tv.client(),
            tv.key_schema(),
            &oid.to_string(),
            SlimSnapshotReadConsumer::StructuralUpdate,
        )
        .await;
        assert!(
            matches!(err, ServiceError::NotFound(_)),
            "seq {raw:?}: a header without a numeric seq is not a live object for the \
             structural consumer; got {err:?}"
        );
    }
}

/// The retirement fence's ghost: a header holding only `superseded=1` (SDD
/// §7.5.4) is no live object for the structural consumer: `NotFound`. (The
/// ext-GC snapshot read used to read it as the empty state at `seq` 0; the
/// reference walk that replaced it walks such a header on, pinned in
/// `retirement_walk_test.rs`.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_read_of_a_superseded_ghost_header_is_not_found() {
    const GHOST: u64 = 770_041;
    let tv = TestValkey::setup().await;
    let _: () = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_header(&GHOST.to_string()),
            [("superseded", "1")],
        )
        .await
        .expect("HSET of the ghost header must succeed");

    let err = read_unfenced_err(
        tv.client(),
        tv.key_schema(),
        &GHOST.to_string(),
        SlimSnapshotReadConsumer::StructuralUpdate,
    )
    .await;
    assert!(
        matches!(err, ServiceError::NotFound(_)),
        "a ghost header is no live object for the structural consumer; got {err:?}"
    );
}

/// An absent header is `NotFound`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_an_absent_header_is_not_found() {
    let tv = TestValkey::setup().await;
    let err = read_unfenced_err(
        tv.client(),
        tv.key_schema(),
        "987654321",
        SlimSnapshotReadConsumer::StructuralUpdate,
    )
    .await;
    assert!(
        matches!(err, ServiceError::NotFound(_)),
        "structural_update: an absent header is not found; got {err:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. Per-key refusals
// ═════════════════════════════════════════════════════════════════════════════
//
// Each staging plants a string on one key the fan-out reads, on the shared b.atj
// `Fixture` (seeded: the root has one child, so on this marked object `obj:chs`
// names the root's list). The failed command is classified under its own key
// family, so exactly that family's `live_edit_snapshot_read_type_error_total`
// cell must tick, and the refusal must name that command — the `op` label of
// that command shape, the same on both fan-outs. Round 1 reads each of its
// four indices' member count (`SCARD` / `ZCARD`) beside the header before any
// page of members, so a wrong-typed index is refused at its count. The read
// here is `structural_update`'s, which reads by bounded replies (b.26e Stage
// 2): on a marked object it counts `obj:chs` (`SCARD`) before paging it, it
// reads a value row's length and columns (`HSTRLEN` + `HMGET`, the first to
// fail naming the refusal) instead of the whole row, and a children list's
// length (`LLEN`) before its ranged `LRANGE` — so those plants are refused at
// the `SCARD`, the `HSTRLEN` and the `LLEN`. The fenced read keeps its
// whole-key commands (`SMEMBERS`, `HGETALL`, `LRANGE`).
//
// Each plant is read on both of the structural read's transports (b.gvf): the
// production one, where a round-2/3 key is refused inside the window script
// and the script's `FANOUT_WINDOW_REFUSED` tag names the entry and the
// command, and the plain one, where the client's own command is refused. The
// refusal is classified the same way on both, so each read ticks the same
// family once and names the same `op`.

async fn assert_unfenced_wrongtype_ticks_only(fx: &Fixture, key: &str, family: &str, op: &str) {
    fx.plant_string(key).await;

    for (reads, transport) in [(1, WindowTransport::Script), (2, WindowTransport::Plain)] {
        fx.tv.client().set_watch_read_transport_for_test(transport);
        let err = read_unfenced_err(
            fx.tv.client(),
            fx.tv.key_schema(),
            &fx.oid(),
            SlimSnapshotReadConsumer::StructuralUpdate,
        )
        .await;
        fx.tv
            .client()
            .set_watch_read_transport_for_test(WindowTransport::Script);

        let ServiceError::Internal(msg) = &err else {
            panic!(
                "{transport:?}: a wrong-typed key is permanent, so the read must be Internal; got \
                 {err:?}"
            );
        };
        assert!(
            msg.contains(op),
            "{transport:?}: the refusal must be the fan-out's own {op:?} for this plant; got \
             {msg:?}"
        );
        assert_only_family_ticked(&fx.metrics, family, reads);
        assert_no_refusal_kind_ticked(&fx.metrics);
        assert_keyspace_type_errors_flat(&fx.metrics);
        assert_eq!(
            snapshot_reads(
                &fx.metrics,
                "structural_update",
                "unfenced",
                "permanent_error"
            ),
            reads,
            "{transport:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_a_wrongtype_object_header_ticks_header() {
    let fx = Fixture::empty("sru-ref-hdr").await;
    let key = fx.tv.key_schema().obj_header(&fx.oid());
    assert_unfenced_wrongtype_ticks_only(&fx, &key, "header", "HGETALL obj header").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_a_wrongtype_sections_index_ticks_sections_index() {
    let fx = Fixture::empty("sru-ref-secs").await;
    let key = fx.tv.key_schema().obj_sections_index(&fx.oid());
    assert_unfenced_wrongtype_ticks_only(&fx, &key, "sections_index", "SCARD sections index").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_a_wrongtype_deleted_sections_index_ticks_deleted_sections_index() {
    let fx = Fixture::empty("sru-ref-dels").await;
    let key = fx.tv.key_schema().obj_deleted_sections_index(&fx.oid());
    assert_unfenced_wrongtype_ticks_only(
        &fx,
        &key,
        "deleted_sections_index",
        "SCARD deleted sections index",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_a_wrongtype_fields_index_ticks_fields_index() {
    let fx = Fixture::seeded("sru-ref-flds").await;
    let key = fx.tv.key_schema().obj_fields_index(&fx.oid());
    assert_unfenced_wrongtype_ticks_only(&fx, &key, "fields_index", "ZCARD fields index").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_a_wrongtype_time_samples_index_ticks_time_samples_index() {
    let fx = Fixture::seeded("sru-ref-tss").await;
    let key = fx.tv.key_schema().obj_time_samples_index(&fx.oid());
    assert_unfenced_wrongtype_ticks_only(
        &fx,
        &key,
        "time_samples_index",
        "ZCARD time samples index",
    )
    .await;
}

/// `obj:chs` — read only on a marked object, counted (`SCARD`) before it is
/// paged, and labelled `children_index`, never `children`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_a_wrongtype_children_index_ticks_children_index() {
    let fx = Fixture::seeded("sru-ref-chs").await;
    let key = fx.tv.key_schema().obj_children_index(&fx.oid());
    assert_unfenced_wrongtype_ticks_only(&fx, &key, "children_index", "SCARD children index").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_a_wrongtype_section_hash_ticks_section() {
    let fx = Fixture::seeded("sru-ref-sec").await;
    let key = fx
        .tv
        .key_schema()
        .obj_section(&fx.oid(), &SECTION_ID.to_string());
    assert_unfenced_wrongtype_ticks_only(&fx, &key, "section", "HGETALL section").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_a_wrongtype_tombstone_hash_ticks_deleted_section() {
    let fx = Fixture::empty("sru-ref-del").await;
    fx.update(create_section_delta(DOOMED_SECTION_ID, "Doomed"))
        .await
        .expect("the setup CreateSection must land");
    fx.update(delete_section_delta(DOOMED_SECTION_ID))
        .await
        .expect("the setup DeleteSection must land and write the tombstone");
    let key = fx
        .tv
        .key_schema()
        .obj_deleted_section(&fx.oid(), &DOOMED_SECTION_ID.to_string());
    assert_unfenced_wrongtype_ticks_only(&fx, &key, "deleted_section", "HGETALL deleted section")
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_a_wrongtype_field_row_ticks_field() {
    let fx = Fixture::seeded("sru-ref-fld").await;
    let key = fx
        .tv
        .key_schema()
        .obj_field(&fx.oid(), &SECTION_ID.to_string(), FIELD_NAME);
    assert_unfenced_wrongtype_ticks_only(&fx, &key, "field", "HSTRLEN field").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_a_wrongtype_time_sample_row_ticks_time_sample() {
    let fx = Fixture::seeded("sru-ref-ts").await;
    let key = fx
        .tv
        .key_schema()
        .obj_time_sample(&fx.oid(), &SECTION_ID.to_string(), SAMPLE_TIME);
    assert_unfenced_wrongtype_ticks_only(&fx, &key, "time_sample", "HSTRLEN time sample").await;
}

/// On a marked object the root's list is read because `obj:chs` names "0";
/// its length is read first, so the plant is refused at the `LLEN`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_a_wrongtype_children_list_ticks_children() {
    let fx = Fixture::seeded("sru-ref-ch").await;
    let key = fx.tv.key_schema().obj_children(&fx.oid(), "0");
    assert_unfenced_wrongtype_ticks_only(&fx, &key, "children", "LLEN children list (parent 0)")
        .await;
}

/// The keys the fan-out must never touch: the commit-token record (`obj:wtok`),
/// the commit release record (`obj:wrel`), the external-value index, the live
/// stream, the ext-retire carrier and its attempt tokens
/// (`obj:extret:attempts`), and the snapshot fence and writer-intent
/// keys (`obj:snapfence`, `obj:snapintent`; the fence protocol's, not the
/// read's); on a marked object a children list `obj:chs` does not name; and on
/// an unmarked object `obj:chs` itself. A foreign value on any of them leaves
/// the read healthy and every counter flat.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_over_wrongtype_keys_it_never_reads_ticks_nothing() {
    let fx = Fixture::seeded("sru-ref-untouched").await;
    let keys = fx.tv.key_schema();
    let oid = fx.oid();
    for key in [
        keys.obj_write_token(&oid),
        keys.obj_write_releases(&oid),
        keys.obj_ext_index(&oid),
        keys.layer_stream(fx.object_id),
        keys.obj_ext_retire_carrier(&oid),
        keys.obj_ext_retire_attempts(&oid),
        keys.obj_snapshot_fence(&oid),
        keys.obj_snapshot_intent(&oid),
        // SECTION_ID is a leaf: no list, so `obj:chs` does not name one.
        keys.obj_children(&oid, &SECTION_ID.to_string()),
    ] {
        fx.plant_string(&key).await;
    }
    let marked = read_unfenced(
        fx.tv.client(),
        keys,
        fx.object_id,
        SlimSnapshotReadConsumer::StructuralUpdate,
    )
    .await;
    assert_eq!(
        marked.state.sections.len(),
        1,
        "structural_update: the read must still carry the fixture's section"
    );

    unmark(&fx.tv, fx.object_id).await;
    let _: i64 = fx
        .tv
        .client()
        .pool
        .del(keys.obj_children(&oid, &SECTION_ID.to_string()))
        .await
        .unwrap();
    fx.plant_string(&keys.obj_children_index(&oid)).await;
    read_unfenced(
        fx.tv.client(),
        keys,
        fx.object_id,
        SlimSnapshotReadConsumer::StructuralUpdate,
    )
    .await;

    assert_no_family_ticked(&fx.metrics);
    assert_no_refusal_kind_ticked(&fx.metrics);
    assert_eq!(
        snapshot_reads(&fx.metrics, "structural_update", "unfenced", "ok"),
        2
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. Metrics (SR-8.1, SR-8.2)
// ═════════════════════════════════════════════════════════════════════════════

/// One tick per read on `path="unfenced"`, under the consumer that read and
/// the outcome it had, and none on `path="atomic"`; the duration histogram
/// records every read under the path, the size histograms only a read that
/// returned state, with that read's own tallies.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_reads_record_one_outcome_per_read_under_their_consumer() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-metrics";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("sru-metrics"));
    tv.client()
        .install_observability_hooks(Arc::clone(&metrics))
        .await;

    let ok = read_unfenced(
        tv.client(),
        tv.key_schema(),
        oid,
        SlimSnapshotReadConsumer::StructuralUpdate,
    )
    .await;
    let missing = read_unfenced_err(
        tv.client(),
        tv.key_schema(),
        "987654321",
        SlimSnapshotReadConsumer::StructuralUpdate,
    )
    .await;
    assert!(matches!(missing, ServiceError::NotFound(_)));

    // `ext_gc` is the other `path="unfenced"` consumer, the clean-up's
    // reference walk (`storage::retirement`): no snapshot read here may tick
    // it.
    let expected: BTreeMap<(&str, &str), u64> = BTreeMap::from([
        (("structural_update", "ok"), 1),
        (("structural_update", "not_found"), 1),
    ]);
    for consumer in ["structural_update", "ext_gc"] {
        for outcome in ["ok", "not_found", "transient_error", "permanent_error"] {
            assert_eq!(
                snapshot_reads(&metrics, consumer, "unfenced", outcome),
                expected.get(&(consumer, outcome)).copied().unwrap_or(0),
                "consumer={consumer}, outcome={outcome}: exactly the two reads above may tick, \
                 each once, on path=unfenced under its own consumer and outcome"
            );
        }
    }
    for consumer in ["catchup", "catchup_legacy_per_commit", "checkpoint"] {
        for outcome in [
            "ok",
            "not_found",
            "too_large",
            "transient_error",
            "permanent_error",
        ] {
            assert_eq!(
                snapshot_reads(&metrics, consumer, "atomic", outcome),
                0,
                "no read here took the atomic path"
            );
        }
    }

    let text = render(&metrics);
    let count = |name: &str, labels: &[(&str, &str)]| sample_value(&text, name, labels);
    assert_eq!(
        count(
            "live_edit_snapshot_read_duration_seconds_count",
            &[("consumer", "structural_update"), ("path", "unfenced")]
        ),
        Some(2.0),
        "every read records its wall time under path=unfenced, a NOT_FOUND too"
    );
    assert_eq!(
        (
            count(
                "live_edit_snapshot_read_reply_bytes_sum",
                &[("consumer", "structural_update")]
            ),
            count(
                "live_edit_snapshot_read_reply_bytes_count",
                &[("consumer", "structural_update")]
            ),
            count(
                "live_edit_snapshot_read_entries_sum",
                &[("consumer", "structural_update")]
            ),
        ),
        (
            Some(ok.reply_bytes as f64),
            Some(1.0),
            Some(ok.entry_count as f64)
        ),
        "the size histograms record the successful read's own tallies, and nothing for the read \
         that returned no state"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. Round 1 in pages (Spec SDD SR-11.3, SR-11.6 (i); design revision 5 § C)
// ═════════════════════════════════════════════════════════════════════════════
//
// Round 1 reads the header with the four indices' member counts, then the
// members in windows of at most `FANOUT_INDEX_WINDOW_MEMBERS` (26,855), all
// indices together: `ZRANGE` pages by rank for the sorted indices, `SSCAN`
// steps for the sets, a set's repeats dropped. A layer whose sorted and set
// indices each take more than one page must still read exactly as the atomic
// read does, which reads each index whole. Shared with
// `snapshot_read_fenced_test`, whose fenced pass runs the same round 1.

/// Sections [`add_paged_bulk`] adds: past the store's compact set encodings
/// (`set-max-intset-entries`, 512), so `SSCAN` pages the sections index, and
/// past the sections' share of one 26,855-member window.
pub(crate) const PAGED_SECTIONS: u64 = 12_000;

/// Fields [`add_paged_bulk`] adds: past the fields' share of one window.
pub(crate) const PAGED_FIELDS: u64 = 30_000;

/// The first section id [`add_paged_bulk`] uses: clear of every fixture id.
pub(crate) const PAGED_SECTION_BASE: u64 = 1_000_000;

/// Add `sections` sections (ids from [`PAGED_SECTION_BASE`] + 1, each with
/// its hash) and `fields` inline fields over them (each with its hash, ranked
/// in the order written) to object `oid`, writing the keys as the commit
/// paths lay them out: enough index members that round 1 needs several
/// pages of each.
pub(crate) async fn add_paged_bulk(tv: &TestValkey, oid: u64, sections: u64, fields: u64) {
    use fred::interfaces::{SetsInterface, SortedSetsInterface};

    let keys = tv.key_schema();
    let pool = &tv.client().pool;
    let oid_str = oid.to_string();
    let section_ids: Vec<String> = (1..=sections)
        .map(|i| (PAGED_SECTION_BASE + i).to_string())
        .collect();
    for chunk in section_ids.chunks(1_000) {
        futures::future::try_join_all(chunk.iter().map(|sid| {
            pool.hset::<(), _, _>(
                keys.obj_section(&oid_str, sid),
                [
                    ("parent_id", ROOT.to_string()),
                    ("name", format!("paged-{sid}")),
                    ("type", "1".to_owned()),
                    ("order", "0".to_owned()),
                    ("first_ver", "1".to_owned()),
                    ("last_ver", "0".to_owned()),
                ],
            )
        }))
        .await
        .expect("HSET of the paged sections must succeed");
    }
    let _: i64 = pool
        .sadd(keys.obj_sections_index(&oid_str), section_ids.clone())
        .await
        .expect("SADD of the paged sections must succeed");
    if fields == 0 {
        return;
    }
    let rows: Vec<(String, String)> = (0..fields)
        .map(|i| {
            (
                section_ids[(i % sections) as usize].clone(),
                format!("paged_{i}"),
            )
        })
        .collect();
    for (n, chunk) in rows.chunks(1_000).enumerate() {
        futures::future::try_join_all(chunk.iter().enumerate().map(|(j, (sid, name))| {
            pool.hset::<(), _, _>(
                keys.obj_field(&oid_str, sid, name),
                [
                    ("value", "Kg==".to_owned()),
                    ("set_order", (n * 1_000 + j + 1).to_string()),
                    ("version", "1".to_owned()),
                ],
            )
        }))
        .await
        .expect("HSET of the paged fields must succeed");
    }
    let members: Vec<(f64, String)> = rows
        .iter()
        .enumerate()
        .map(|(i, (sid, name))| (1_000.0 + i as f64, KeySchema::field_member(sid, name)))
        .collect();
    let _: i64 = pool
        .zadd(
            keys.obj_fields_index(&oid_str),
            None,
            None,
            false,
            false,
            members,
        )
        .await
        .expect("ZADD of the paged fields must succeed");
}

/// The `fan-out window` events of a captured read: every window's round-1
/// index members are within the shipped bound it reports, and the windows
/// that carried members — the pages — are returned.
pub(crate) fn assert_round_one_windows_within_the_bound(
    captured: &crate::grpc::log_capture::Captured,
    what: &str,
) -> Vec<serde_json::Value> {
    let windows = captured.with_message("fan-out window");
    assert!(!windows.is_empty(), "{what}: the read logged its windows");
    let mut pages = Vec::new();
    for w in windows {
        let fields = &w["fields"];
        assert_eq!(
            fields["member_target"],
            live_edit::config::FANOUT_INDEX_WINDOW_MEMBERS,
            "{what}: every window reports the shipped round-1 bound; window: {w}"
        );
        let members = fields["index_members"]
            .as_u64()
            .unwrap_or_else(|| panic!("{what}: index_members is an integer; window: {w}"));
        assert!(
            members <= live_edit::config::FANOUT_INDEX_WINDOW_MEMBERS,
            "{what}: no round-1 window may read more than {} index members; window: {w}",
            live_edit::config::FANOUT_INDEX_WINDOW_MEMBERS
        );
        if members > 0 {
            pages.push(fields.clone());
        }
    }
    pages
}

/// `SSCAN`s of the sections index and `ZRANGE`s of the fields index in a
/// `MONITOR` record: each index's pages.
pub(crate) fn index_pages(
    seen: &[MonitoredCommand],
    sections_key: &str,
    fields_key: &str,
) -> (usize, usize) {
    let count = |name: &str, key: &str| {
        seen.iter()
            .filter(|c| !c.issued_by_script() && c.name() == name)
            .filter(|c| c.args.get(1).map(String::as_str) == Some(key))
            .count()
    };
    (count("SSCAN", sections_key), count("ZRANGE", fields_key))
}

/// A layer whose sections set (12,000 members) and fields sorted set (30,000)
/// each take more than one page: the structural consumer's fan-out reads
/// exactly the compact projection of the atomic read's state and its tallies
/// (the atomic read run on a client whose ceilings it is under), every
/// round-1 window reads at most `FANOUT_INDEX_WINDOW_MEMBERS` index members,
/// and the store saw several `SSCAN`s of the sections index and several
/// `ZRANGE`s of the fields index.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_of_indices_that_take_several_pages_equals_the_atomic_read() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-paged";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    add_paged_bulk(&tv, oid, PAGED_SECTIONS, PAGED_FIELDS).await;
    let oid_str = oid.to_string();
    let (sections_key, fields_key) = (
        tv.key_schema().obj_sections_index(&oid_str),
        tv.key_schema().obj_fields_index(&oid_str),
    );
    let unbounded = client_with_limits(u64::MAX / 4, u64::MAX / 4).await;
    let atomic = read_atomic(&unbounded, tv.key_schema(), oid).await;
    assert!(
        atomic.state.sections.len() as u64 > PAGED_SECTIONS
            && atomic.state.fields.len() as u64 > PAGED_FIELDS,
        "fixture precondition: the atomic read carries the bulk"
    );

    let what = "structural_update";
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let (unfenced, captured) = crate::grpc::log_capture::capture_debug(read_unfenced(
        tv.client(),
        tv.key_schema(),
        oid,
        SlimSnapshotReadConsumer::StructuralUpdate,
    ))
    .await;
    let fence = monitor_fence(tv.prefix());
    let _: i64 = tv.client().pool.exists(&fence).await.unwrap();
    let seen = tap.drain_until(&fence).await;

    let pages = assert_round_one_windows_within_the_bound(&captured, what);
    assert!(
        pages.len() >= 2,
        "{what}: 42,000 index members take more than one window: {pages:?}"
    );
    let (sscans, zranges) = index_pages(&seen, &sections_key, &fields_key);
    assert!(
        sscans >= 2 && zranges >= 2,
        "{what}: each index is past its share of one window, so each took more than one page \
         ({sscans} SSCANs of the sections index, {zranges} ZRANGEs of the fields index)"
    );
    assert_eq!(
        unfenced.state,
        compact_snapshot(&atomic.state),
        "{what}: the paged round 1 must collect exactly the whole indices"
    );
    assert_eq!(
        (
            unfenced.item_count,
            unfenced.reply_bytes,
            unfenced.entry_count
        ),
        (atomic.item_count, atomic.reply_bytes, atomic.entry_count),
        "{what}: the tallies count every index member once, as the atomic read does"
    );
}

/// `SSCAN` may return a member more than once (a set rehashed between
/// steps); round 1 keeps each once. A proxy answers every `SSCAN` of the
/// sections index with a real step (`COUNT` 50, so 600 sections take many
/// steps) whose members each come twice, plus one member in every step: the
/// read still equals the atomic read, its tallies counting each member once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unfenced_read_drops_the_members_an_sscan_step_repeats() {
    use crate::storage::snapshot_read_support::{client_through, ProxyAction, ScriptReplyProxy};

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-sscan-repeats";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    add_paged_bulk(&tv, oid, 600, 0).await;
    let sections_key = tv.key_schema().obj_sections_index(&oid.to_string());
    let repeated = (PAGED_SECTION_BASE + 1).to_string();
    let rule = {
        let sections_key = sections_key.clone();
        let repeated = repeated.clone();
        Arc::new(move |args: &[Vec<u8>]| {
            let arg = |i: usize| args.get(i).map(|a| String::from_utf8_lossy(a).into_owned());
            if !arg(0)?.eq_ignore_ascii_case("SSCAN") || arg(1)? != sections_key {
                return None;
            }
            let cursor = arg(2)?;
            Some(ProxyAction::Eval(format!(
                "local r = redis.call('SSCAN', '{sections_key}', '{cursor}', 'COUNT', '50') \
                 local out = {{}} \
                 for _, m in ipairs(r[2]) do out[#out + 1] = m; out[#out + 1] = m end \
                 out[#out + 1] = '{repeated}' \
                 return {{r[1], out}}"
            )))
        }) as crate::storage::snapshot_read_support::ProxyRule
    };
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let proxied = client_through(&proxy).await;
    let atomic = read_atomic(tv.client(), tv.key_schema(), oid).await;

    // The structural WATCH consumer's slim fan-out over the rewritten steps:
    // the compact projection of the atomic read, the same tallies.
    let slim = read_unfenced(
        &proxied,
        tv.key_schema(),
        oid,
        SlimSnapshotReadConsumer::StructuralUpdate,
    )
    .await;
    assert!(
        proxy.hits() >= 3,
        "precondition: the sections index was scanned in several rewritten steps; {} hits",
        proxy.hits()
    );
    assert_eq!(slim.state, compact_snapshot(&atomic.state));
    assert_eq!(
        (slim.item_count, slim.reply_bytes, slim.entry_count),
        (atomic.item_count, atomic.reply_bytes, atomic.entry_count),
        "each section index member counted once — and its hash read once — however often its \
         SSCAN steps returned it"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 8. The window script (b.gvf; SDD §10.5)
// ═════════════════════════════════════════════════════════════════════════════
//
// The structural read issues plain commands for round 1, for a lone children
// list whose modelled shard time alone exceeds τ, and for every window after
// a script fallback; every other window of its rounds 2 and 3 is one `EVALSHA`
// of `fanout_window_read.lua`, which runs the plain window's commands inside
// the script. These tests read what the client sent and what the script ran
// from the store's own command feed (`MONITOR`, whose `lua` client is a
// script's), and the windows from the read's `fan-out window` debug events
// (`transport`: `script`, `plain` or `plain_fallback`). Each filters the feed
// by its own object's keys, so the shared store's other traffic cannot match.

/// Inline field rows [`many_field_rows_state`] adds.
const MANY_FIELD_ROWS: u64 = 1_000;

/// `every_shape_state` with [`MANY_FIELD_ROWS`] inline field rows added on
/// `WORLD`.
fn many_field_rows_state() -> live_edit::domain::types::ObjectState {
    use live_edit::domain::types::{Field, FieldValue};
    let mut state = every_shape_state();
    for i in 0..MANY_FIELD_ROWS {
        let key = format!("attr{i:04}");
        state.fields.insert(
            (WORLD, key.clone()),
            Field {
                section_id: WORLD,
                key_name: key,
                value: FieldValue::Inline(vec![(i % 251) as u8; 8]),
                set_order: 10 + i,
                authoring_set_order: 10 + i,
                version: 1,
            },
        );
    }
    state
}

/// The key prefix every key of object `oid` carries (`{prefix}{oid}:obj:`),
/// the window script's `base`.
fn object_key_base(tv: &TestValkey, oid: u64) -> String {
    format!("{}{{{oid}}}:obj:", tv.prefix())
}

/// Whether `command` is an `EVALSHA` of the window script whose one declared
/// key is `header_key`: one window of the structural read of that object.
fn is_window_script_run(command: &MonitoredCommand, header_key: &str) -> bool {
    !command.issued_by_script()
        && command.name() == "EVALSHA"
        && command
            .args
            .get(1)
            .is_some_and(|sha| sha.eq_ignore_ascii_case(FANOUT_WINDOW_READ_SCRIPT.sha1()))
        && command.args.get(2).map(String::as_str) == Some("1")
        && command.args.get(3).map(String::as_str) == Some(header_key)
}

/// The window scripts of the read of the object whose header is
/// `header_key` in a `MONITOR` record, one per window: distinct `EVALSHA`s
/// ([`is_window_script_run`]) by their arguments. A window re-sent after a
/// `NOSCRIPT` (a concurrent test's `SCRIPT FLUSH` on the shared store) sends
/// the same arguments again and is counted once.
fn window_scripts_sent(seen: &[MonitoredCommand], header_key: &str) -> usize {
    seen.iter()
        .filter(|c| is_window_script_run(c, header_key))
        .map(|c| c.args.as_slice())
        .collect::<BTreeSet<_>>()
        .len()
}

/// The `fan-out window` events of object `oid`'s reads in `captured`.
fn window_events(
    captured: &crate::grpc::log_capture::Captured,
    oid: u64,
) -> Vec<serde_json::Value> {
    let oid = oid.to_string();
    captured
        .with_message("fan-out window")
        .into_iter()
        .filter(|e| e["fields"]["object_id"].as_str() == Some(oid.as_str()))
        .cloned()
        .collect()
}

/// How many of `events` were issued as `transport`.
fn windows_issued_as(events: &[serde_json::Value], transport: &str) -> usize {
    events
        .iter()
        .filter(|e| e["fields"]["transport"].as_str() == Some(transport))
        .count()
}

/// The key families of round 1: the header and the index parts.
const ROUND_ONE_FAMILIES: [&str; 6] = [
    "header",
    "sections_index",
    "deleted_sections_index",
    "fields_index",
    "time_samples_index",
    "children_index",
];

/// Run `read` with a `MONITOR` tap open and its debug events captured: its
/// answer, the store's command feed up to a fence issued after it, and the
/// capture.
async fn monitored<F, T>(
    tv: &TestValkey,
    read: F,
) -> (T, Vec<MonitoredCommand>, crate::grpc::log_capture::Captured)
where
    F: std::future::Future<Output = T>,
{
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let (answer, captured) = crate::grpc::log_capture::capture_debug(read).await;
    let fence = monitor_fence(tv.prefix());
    let _: i64 = tv
        .client()
        .pool
        .exists(&fence)
        .await
        .expect("EXISTS the fence must succeed");
    (answer, tap.drain_until(&fence).await, captured)
}

/// The structural read of a layer of 1,000 field rows, on the production
/// transport (b.gvf, fail-on-old): the client issues no value-row or
/// list-length command (`HSTRLEN`, `HMGET`, `HEXISTS`, `EXISTS`, `LLEN`) on
/// the object's keys — the window scripts issue them, one `HSTRLEN` per value
/// row; each round-2/3 window is one `EVALSHA` of the window script, its one
/// declared key (`KEYS[1]`) the object's header, one per `script` window
/// event; and round 1 is plain: every plain window carries only round 1's
/// families, the header read once by the client. On this layer no window is
/// a lone over-τ list and nothing falls back, so no other window is plain.
/// The read is the compact projection of the atomic read.
///
/// Before the window script every round-2/3 window was plain, and the read
/// issued two client commands per value row: the first assertion fails.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_read_issues_each_round_two_and_three_window_as_one_window_script_b_gvf() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-window-script";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &many_field_rows_state()).await;
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let base = object_key_base(&tv, oid);
    let atomic = read_atomic(tv.client(), tv.key_schema(), oid).await;
    let value_rows = atomic.state.fields.len() + atomic.state.time_samples.len();
    assert!(
        atomic.state.fields.len() as u64 > MANY_FIELD_ROWS,
        "fixture precondition: the layer holds the added field rows"
    );

    let (read, seen, captured) = monitored(
        &tv,
        read_unfenced(
            tv.client(),
            tv.key_schema(),
            oid,
            SlimSnapshotReadConsumer::StructuralUpdate,
        ),
    )
    .await;
    assert_eq!(
        read.state,
        compact_snapshot(&atomic.state),
        "the read is the atomic read's compact projection"
    );

    let on_object = |c: &&MonitoredCommand| c.args.get(1).is_some_and(|k| k.starts_with(&base));
    let client_reads: Vec<&MonitoredCommand> = seen
        .iter()
        .filter(on_object)
        .filter(|c| {
            !c.issued_by_script()
                && matches!(
                    c.name().as_str(),
                    "HSTRLEN" | "HMGET" | "HEXISTS" | "EXISTS" | "LLEN"
                )
        })
        .collect();
    assert!(
        client_reads.is_empty(),
        "no value-row or list-length command may come from the client: rounds 2 and 3 run \
         inside window scripts; client commands: {client_reads:?}"
    );
    let script_value_rows = seen
        .iter()
        .filter(on_object)
        .filter(|c| c.issued_by_script() && c.name() == "HSTRLEN")
        .count();
    assert_eq!(
        script_value_rows, value_rows,
        "the window scripts read every value row once (one HSTRLEN each)"
    );

    let events = window_events(&captured, oid);
    let script_runs = window_scripts_sent(&seen, &header_key);
    assert!(
        windows_issued_as(&events, "script") >= 1,
        "the read issued script windows; events: {events:?}"
    );
    assert_eq!(
        script_runs,
        windows_issued_as(&events, "script"),
        "one EVALSHA of the window script, KEYS[1] the header, per script window"
    );
    assert_eq!(
        windows_issued_as(&events, "plain_fallback"),
        0,
        "nothing fell back"
    );
    for event in events
        .iter()
        .filter(|e| e["fields"]["transport"].as_str() == Some("plain"))
    {
        for end in ["first_family", "last_family"] {
            let family = event["fields"][end].as_str().unwrap_or_default();
            assert!(
                ROUND_ONE_FAMILIES.contains(&family),
                "a plain window here is round 1's; its {end} is {family:?}: {event}"
            );
        }
    }
    let header_reads = seen
        .iter()
        .filter(|c| {
            !c.issued_by_script()
                && c.name() == "HGETALL"
                && c.args.get(1).map(String::as_str) == Some(header_key.as_str())
        })
        .count();
    assert_eq!(
        header_reads, 1,
        "round 1 reads the header as a plain command"
    );
}

/// The entries of one children list a lone script window may read within τ
/// by the window model at the wired constants: the largest `n` with
/// `fanout_script_window_modelled_shard_ns(1, n, n × 20 B)` within τ (one
/// wire command, the list's `LRANGE`; 20 B of reply bound per entry).
/// Computed from the constants, so it follows a re-fit of the window model.
pub(crate) fn one_list_tau_capacity() -> u64 {
    use live_edit::config::{
        fanout_script_window_modelled_shard_ns, FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS,
    };
    let fits = |n: u64| {
        fanout_script_window_modelled_shard_ns(1, n, n * 20) <= FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS
    };
    assert!(fits(1), "the window model admits a one-entry list");
    let (mut lo, mut hi) = (1u64, 1u64 << 40);
    assert!(
        !fits(hi),
        "the window model's per-entry terms are zero: no list is over τ"
    );
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// A lone children list whose modelled shard time alone exceeds τ is read
/// by one plain `LRANGE` from the client (b.gvf refinement 3, the
/// children-list case): on a layer with one section whose list holds one
/// entry more than a script window may read within τ
/// ([`one_list_tau_capacity`], from the wired window model), that list's
/// `LRANGE` comes from the client while its `LLEN` ran inside a window
/// script, one window is `plain` beyond round 1's, and the read equals the
/// atomic read's compact projection.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_read_of_a_lone_children_list_over_tau_ranges_it_by_one_plain_lrange_b_gvf() {
    use live_edit::domain::types::{ObjectState, Section};
    const PARENT: u64 = 1;
    let children = one_list_tau_capacity() + 1;
    assert!(
        children <= 100_000,
        "fixture: a lone over-τ list needs {children} children at the wired window model; \
         resize this fixture"
    );
    let section = |id: u64, parent: u64, order: u32| Section {
        section_id: id,
        parent_id: parent,
        name: format!("s{id}"),
        section_type: 1,
        section_order: order,
        first_version: 1,
        last_version: 1,
    };
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        ..Default::default()
    };
    state.sections.insert(PARENT, section(PARENT, ROOT, 0));
    state.children.insert(ROOT, vec![PARENT]);
    let kids: Vec<u64> = (100..100 + children).collect();
    for (order, id) in kids.iter().enumerate() {
        state
            .sections
            .insert(*id, section(*id, PARENT, order as u32));
    }
    state.children.insert(PARENT, kids);

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-lone-long-list";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &state).await;
    let list_key = tv
        .key_schema()
        .obj_children(&oid.to_string(), &PARENT.to_string());
    let atomic = read_atomic(
        &client_with_limits(u64::MAX / 4, u64::MAX / 4).await,
        tv.key_schema(),
        oid,
    )
    .await;

    let (read, seen, captured) = monitored(
        &tv,
        read_unfenced(
            tv.client(),
            tv.key_schema(),
            oid,
            SlimSnapshotReadConsumer::StructuralUpdate,
        ),
    )
    .await;
    assert_eq!(read.state, compact_snapshot(&atomic.state));
    let on_list = |name: &str, by_script: bool| {
        seen.iter()
            .filter(|c| {
                c.issued_by_script() == by_script
                    && c.name() == name
                    && c.args.get(1).map(String::as_str) == Some(list_key.as_str())
            })
            .count()
    };
    assert_eq!(
        (on_list("LRANGE", false), on_list("LRANGE", true)),
        (1, 0),
        "the over-τ list is ranged by one plain LRANGE, never inside a window script"
    );
    assert_eq!(
        (on_list("LLEN", true), on_list("LLEN", false)),
        (1, 0),
        "its length is read inside a round-2 window script"
    );
    let events = window_events(&captured, oid);
    let lone_list_windows: Vec<&serde_json::Value> = events
        .iter()
        .filter(|e| {
            e["fields"]["transport"].as_str() == Some("plain")
                && e["fields"]["first_family"].as_str() == Some("children")
        })
        .collect();
    assert_eq!(
        lone_list_windows.len(),
        1,
        "one plain window beyond round 1's: the lone list's; events: {events:?}"
    );
    assert_eq!(lone_list_windows[0]["fields"]["commands"], 1);
}

/// An in-script `WRONGTYPE` (b.gvf): a string planted on the root's children
/// list, read by the structural read's round-2 window script, is refused
/// inside the script — its `LLEN` on that key ran as the script's and never
/// as the client's — and the script's `FANOUT_WINDOW_REFUSED` tag is
/// classified as the list's own refusal: `Internal`, naming `LLEN children
/// list (parent 0)`, ticking `live_edit_snapshot_read_type_error_total
/// {key_family="children"}` once and no refusal kind.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_read_refused_wrongtype_inside_the_window_script_ticks_children_b_gvf() {
    let fx = Fixture::seeded("sru-script-wrongtype-ch").await;
    let key = fx.tv.key_schema().obj_children(&fx.oid(), "0");
    fx.plant_string(&key).await;

    let (err, seen, _) = monitored(
        &fx.tv,
        read_unfenced_err(
            fx.tv.client(),
            fx.tv.key_schema(),
            &fx.oid(),
            SlimSnapshotReadConsumer::StructuralUpdate,
        ),
    )
    .await;

    let ServiceError::Internal(msg) = &err else {
        panic!("a wrong-typed key is permanent, so the read must be Internal; got {err:?}");
    };
    assert!(
        msg.contains("LLEN children list (parent 0)"),
        "the refusal names the list's own LLEN; got {msg:?}"
    );
    let llen = |by_script: bool| {
        seen.iter()
            .filter(|c| {
                c.issued_by_script() == by_script
                    && c.name() == "LLEN"
                    && c.args.get(1).map(String::as_str) == Some(key.as_str())
            })
            .count()
    };
    assert_eq!(
        (llen(true), llen(false)),
        (1, 0),
        "the LLEN that met the WRONGTYPE ran inside the window script"
    );
    assert_only_family_ticked(&fx.metrics, "children", 1);
    assert_no_refusal_kind_ticked(&fx.metrics);
}

/// A window script's reply that is not this binary's shape (b.gvf): a proxy
/// answers the window script's `EVALSHA` with an older format tag and no
/// entries. The read fails `Internal`, attributed to the object's header and
/// the window's `EVALSHA` (`EVALSHA fanout_window_read`), and names the
/// malformed reply; it is never decoded into a state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_read_of_a_window_script_reply_not_of_its_shape_fails_internal_at_the_header_b_gvf(
) {
    use crate::storage::snapshot_read_support::{
        client_through, script_frames_naming, ProxyAction, ScriptReplyProxy,
    };

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-window-script-malformed";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        script_frames_naming(
            FANOUT_WINDOW_READ_SCRIPT.sha1().to_string(),
            ProxyAction::Eval("return {'fanout-window-read-v0', 'OK'}".to_owned()),
        ),
    )
    .await;
    let proxied = client_through(&proxy).await;

    let err = read_unfenced_err(
        &proxied,
        tv.key_schema(),
        &oid.to_string(),
        SlimSnapshotReadConsumer::StructuralUpdate,
    )
    .await;

    assert_eq!(
        proxy.hits(),
        1,
        "the read's first window script was answered"
    );
    let ServiceError::Internal(msg) = &err else {
        panic!("a reply not of the script's shape is Internal; got {err:?}");
    };
    for part in [
        "EVALSHA fanout_window_read",
        header_key.as_str(),
        "is malformed",
    ] {
        assert!(
            msg.contains(part),
            "the error is attributed to the header's window EVALSHA and names the malformed \
             reply ({part:?}); got {msg:?}"
        );
    }
}

/// A window script refused on its slot's state at the store (b.gvf): a proxy
/// answers the window script's first `EVALSHA` with the script's own
/// per-entry refusal tag, `FANOUT_WINDOW_REFUSED: 1 HGETALL` and a
/// cluster-state text (a slot not served). The read falls back: it re-issues
/// that window and every later one as plain commands (no further `EVALSHA`
/// reaches the proxy; the windows are `plain_fallback`), completes with the
/// atomic read's compact projection, and ticks
/// `live_edit_snapshot_read_window_fallback_total` once. The next read tries
/// the script again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_read_falls_back_to_plain_commands_on_a_cluster_state_refusal_tag_b_gvf() {
    use crate::storage::snapshot_read_support::{
        client_through, error_reply_script, script_frames_naming, ProxyAction, ScriptReplyProxy,
    };

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-window-script-tag-fallback";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let atomic = read_atomic(tv.client(), tv.key_schema(), oid).await;
    let proxy = ScriptReplyProxy::start(
        test_valkey_addr(),
        script_frames_naming(
            FANOUT_WINDOW_READ_SCRIPT.sha1().to_string(),
            ProxyAction::Eval(error_reply_script(
                "FANOUT_WINDOW_REFUSED: 1 HGETALL ERR Script attempted to access a slot not served",
            )),
        ),
    )
    .await;
    let proxied = client_through(&proxy).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("sru-window-fallback"));
    proxied
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    let fallbacks = || {
        sample_value(
            &render(&metrics),
            "live_edit_snapshot_read_window_fallback_total",
            &[],
        )
        .expect("the fallback counter is seeded at zero")
    };
    assert_eq!(fallbacks(), 0.0);

    let (read, captured) = crate::grpc::log_capture::capture_debug(read_unfenced(
        &proxied,
        tv.key_schema(),
        oid,
        SlimSnapshotReadConsumer::StructuralUpdate,
    ))
    .await;

    assert_eq!(read.state, compact_snapshot(&atomic.state));
    assert_eq!(
        proxy.hits(),
        1,
        "only the first window was sent as a script; every later window is plain"
    );
    let events = window_events(&captured, oid);
    assert_eq!(
        windows_issued_as(&events, "script"),
        0,
        "no window completed as a script"
    );
    assert!(
        windows_issued_as(&events, "plain_fallback") >= 2,
        "the refused window and the later ones (round 3's lists) are plain after the \
         fallback; events: {events:?}"
    );
    assert_eq!(fallbacks(), 1.0, "one tick for the read that fell back");

    read_unfenced(
        &proxied,
        tv.key_schema(),
        oid,
        SlimSnapshotReadConsumer::StructuralUpdate,
    )
    .await;
    assert_eq!(
        proxy.hits(),
        2,
        "the next read starts on the script again (and falls back again)"
    );
    assert_eq!(fallbacks(), 2.0);
}

/// A cluster-state refusal text, as a window script's `FANOUT_WINDOW_REFUSED`
/// tag carries it when an entry's command met a slot not served.
const SLOT_NOT_SERVED: &str = "ERR Script attempted to access a slot not served";

/// Where a failed window script's error is attributed: the object's header
/// under the window's `EVALSHA`, or the first entry's section hash under its
/// own `op`.
#[derive(Debug, Clone, Copy)]
enum Attributed {
    Header,
    FirstSection,
}

/// Every exit of the script window's failure classifier (b.gvf) but the two
/// pinned above (a cluster-state tag on an entry's own command falls back; an
/// in-script `WRONGTYPE` ticks its family) and the store's not-serving replies
/// (retryable, pinned by the test after this one): a proxy answers the read's
/// first window script `EVALSHA` with each error reply in turn. Each fails the
/// read `Internal` — none is retryable, none falls back
/// (`live_edit_snapshot_read_window_fallback_total` stays 0), none ticks
/// `live_edit_snapshot_read_type_error_total` — under the `op` and key its
/// attribution names, and ticks one `live_edit_snapshot_read_refused_total`
/// cell:
///
/// - a tag whose entry is out of range (0, or past the window), even with a
///   cluster-state raw error: the header, kind `unknown` — it says nothing
///   about any one entry, so it fails closed rather than falling back;
/// - a tag naming a command foreign to its entry (`LLEN` on the first entry,
///   a section hash, whose one command is `HGETALL`), also with a
///   cluster-state raw error: the header, kind `unknown`;
/// - a tag on the entry's own command whose raw error is neither `WRONGTYPE`
///   nor cluster-state: that entry's own refusal, `HGETALL section` on a
///   section key, kind `unknown`;
/// - `FANOUT_WINDOW_ARGS` (a caller bug the script detected): the header,
///   kind `unknown`.
///
/// The first window script's first entry is a section hash: round 2 lists
/// every section hash first, and `every_shape_state` has sections. The
/// attributed key is read from the refusal's log event, which names it; the
/// error's message names the `op`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_read_classifies_each_other_window_script_failure_as_internal_b_gvf() {
    use crate::storage::b_muu_tests::assert_only_refusal_kind_ticked;
    use crate::storage::snapshot_read_support::{
        client_through, error_reply_script, script_frames_naming, ProxyAction, ScriptReplyProxy,
    };

    let cases: [(&str, String, Attributed, &str); 5] = [
        (
            "entry-zero",
            format!("FANOUT_WINDOW_REFUSED: 0 HGETALL {SLOT_NOT_SERVED}"),
            Attributed::Header,
            "unknown",
        ),
        (
            "entry-past-the-window",
            format!("FANOUT_WINDOW_REFUSED: 1000000 HGETALL {SLOT_NOT_SERVED}"),
            Attributed::Header,
            "unknown",
        ),
        (
            "foreign-command",
            format!("FANOUT_WINDOW_REFUSED: 1 LLEN {SLOT_NOT_SERVED}"),
            Attributed::Header,
            "unknown",
        ),
        (
            "entry-own-permanent",
            "FANOUT_WINDOW_REFUSED: 1 HGETALL ERR a permanent in-script failure".to_owned(),
            Attributed::FirstSection,
            "unknown",
        ),
        (
            "args",
            "FANOUT_WINDOW_ARGS: KEYS[1] is not the object's header".to_owned(),
            Attributed::Header,
            "unknown",
        ),
    ];

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-window-script-classified";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let oid_str = oid.to_string();
    let header_key = tv.key_schema().obj_header(&oid_str);
    let section_key_prefix = tv.key_schema().obj_section(&oid_str, "");

    for (name, reply, attributed, kind) in cases {
        let proxy = ScriptReplyProxy::start(
            test_valkey_addr(),
            script_frames_naming(
                FANOUT_WINDOW_READ_SCRIPT.sha1().to_string(),
                ProxyAction::Eval(error_reply_script(&reply)),
            ),
        )
        .await;
        let proxied = client_through(&proxy).await;
        let metrics = Arc::new(crate::common::metrics_scrape_only(&format!(
            "sru-window-classified-{name}"
        )));
        proxied
            .install_observability_hooks(Arc::clone(&metrics))
            .await;

        let (err, captured) = crate::grpc::log_capture::capture_debug(read_unfenced_err(
            &proxied,
            tv.key_schema(),
            &oid_str,
            SlimSnapshotReadConsumer::StructuralUpdate,
        ))
        .await;

        assert_eq!(
            proxy.hits(),
            1,
            "{name}: the read's first window script was answered, and nothing was re-sent"
        );
        let (op, family) = match attributed {
            Attributed::Header => ("EVALSHA fanout_window_read", "header"),
            Attributed::FirstSection => ("HGETALL section", "section"),
        };
        let ServiceError::Internal(msg) = &err else {
            panic!("{name}: the failure is permanent, so the read is Internal; got {err:?}");
        };
        assert!(
            msg.starts_with(&format!("{op} failed:")),
            "{name}: the error names {op:?}; got {msg:?}"
        );
        let refusals: Vec<&serde_json::Value> = captured
            .events
            .iter()
            .filter(|e| {
                e["fields"]["message"]
                    .as_str()
                    .is_some_and(|m| m.starts_with("object-state touch refused"))
            })
            .collect();
        assert_eq!(
            refusals.len(),
            1,
            "{name}: one refusal is logged; events: {:?}",
            captured.events
        );
        let fields = &refusals[0]["fields"];
        assert_eq!(fields["op"].as_str(), Some(op), "{name}: {fields:?}");
        assert_eq!(
            fields["key_family"].as_str(),
            Some(family),
            "{name}: {fields:?}"
        );
        assert_eq!(fields["kind"].as_str(), Some(kind), "{name}: {fields:?}");
        let key = fields["key"].as_str().expect("the refusal names its key");
        match attributed {
            Attributed::Header => assert_eq!(key, header_key, "{name}"),
            Attributed::FirstSection => assert!(
                key.starts_with(&section_key_prefix),
                "{name}: the refusal names a section hash of the object; got {key:?}"
            ),
        }
        assert_only_refusal_kind_ticked(&metrics, kind, 1);
        assert_no_family_ticked(&metrics);
        assert_eq!(
            sample_value(
                &render(&metrics),
                "live_edit_snapshot_read_window_fallback_total",
                &[],
            ),
            Some(0.0),
            "{name}: the read did not fall back"
        );
    }
}

/// The store not serving — every `STORE_NOT_SERVING_SHAPES` row: `-LOADING`
/// while a node replays its data, `-BUSY` behind a long script or module
/// command — answering the read's first window script `EVALSHA` (b.xd1). fred
/// hands each over as `ErrorKind::Unknown`; the classifier's last arm
/// attributes it to the header under `EVALSHA fanout_window_read`, and the
/// shared snapshot-read classifier calls it transient
/// (`client::is_store_not_serving_reply`). So the read fails retryable —
/// `Aborted` with the `transient I/O:` prefix, which the structural path's
/// retry loop re-issues within its attempt budget (an exhausted call reaches
/// the client as retryable `UNAVAILABLE`) — and it is uncounted: no
/// `live_edit_snapshot_read_refused_total` cell ticks, no refusal event is
/// logged, the read does not fall back to plain commands, and nothing is
/// re-sent by the read itself. Pre-b.xd1 each was a permanent `Internal`
/// ticking `backend_unavailable`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_read_answers_a_not_serving_window_script_failure_as_retryable_b_xd1() {
    use live_edit::storage::client::STORE_NOT_SERVING_SHAPES;

    use crate::storage::snapshot_read_support::{
        client_through, error_reply_script, script_frames_naming, ProxyAction, ScriptReplyProxy,
    };

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "sru-window-script-not-serving";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let oid_str = oid.to_string();

    for reply in STORE_NOT_SERVING_SHAPES {
        let proxy = ScriptReplyProxy::start(
            test_valkey_addr(),
            script_frames_naming(
                FANOUT_WINDOW_READ_SCRIPT.sha1().to_string(),
                ProxyAction::Eval(error_reply_script(reply)),
            ),
        )
        .await;
        let proxied = client_through(&proxy).await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("sru-window-not-serving"));
        proxied
            .install_observability_hooks(Arc::clone(&metrics))
            .await;

        let (err, captured) = crate::grpc::log_capture::capture_debug(read_unfenced_err(
            &proxied,
            tv.key_schema(),
            &oid_str,
            SlimSnapshotReadConsumer::StructuralUpdate,
        ))
        .await;

        assert_eq!(
            proxy.hits(),
            1,
            "{reply:?}: the read's first window script was answered, and nothing was re-sent"
        );
        let ServiceError::Aborted(msg) = &err else {
            panic!(
                "{reply:?}: the store not serving is transient, so the read is the retryable \
                 Aborted the structural path's retry loop re-issues; pre-b.xd1 it was a permanent \
                 Internal; got {err:?}"
            );
        };
        assert!(
            msg.starts_with("transient I/O: EVALSHA fanout_window_read failed"),
            "{reply:?}: the retry prefix and the window script's op; got {msg:?}"
        );
        let refusals = captured
            .events
            .iter()
            .filter(|e| {
                e["fields"]["message"]
                    .as_str()
                    .is_some_and(|m| m.starts_with("object-state touch refused"))
            })
            .count();
        assert_eq!(
            refusals, 0,
            "{reply:?}: a transient failure logs no refusal; events: {:?}",
            captured.events
        );
        assert_no_refusal_kind_ticked(&metrics);
        assert_no_family_ticked(&metrics);
        assert_eq!(
            sample_value(
                &render(&metrics),
                "live_edit_snapshot_read_window_fallback_total",
                &[],
            ),
            Some(0.0),
            "{reply:?}: the read did not fall back"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 9. The fault seam
// ═════════════════════════════════════════════════════════════════════════════

/// On the fan-out the seam answers in place of the first command of the window
/// that consults it — on a fresh read, round 1's header `HGETALL` — and the
/// error is classified as a real failure of that command: a transient kind is
/// `Aborted` (the structural path's retry loop absorbs it) and ticks no refusal
/// counter, a permanent one is `Internal` and ticks its refusal kind. One
/// queued error fails one read: the next read is the real one.
#[cfg(feature = "test-fault-injection")]
mod fault_seam {
    use super::*;

    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use live_edit::storage::diff_read::{
        ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget, ObjectSnapshotReader,
    };

    use crate::storage::b_muu_tests::assert_only_refusal_kind_ticked;

    async fn read_with_injected(fx: &Fixture, injected: FredError) -> ServiceError {
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        injector.push(injected);
        fx.tv
            .client()
            .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
        let err = read_unfenced_err(
            fx.tv.client(),
            fx.tv.key_schema(),
            &fx.oid(),
            SlimSnapshotReadConsumer::StructuralUpdate,
        )
        .await;
        assert_eq!(
            injector.remaining(),
            0,
            "the read must have consumed the queued error, or the verdict is some other failure's"
        );
        read_unfenced(
            fx.tv.client(),
            fx.tv.key_schema(),
            fx.object_id,
            SlimSnapshotReadConsumer::StructuralUpdate,
        )
        .await;
        err
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unfenced_read_with_an_injected_transient_failure_is_aborted_and_counts_no_refusal() {
        let fx = Fixture::seeded("sru-seam-transient").await;

        let err = read_with_injected(
            &fx,
            FredError::new(FredErrorKind::Timeout, "injected fan-out timeout"),
        )
        .await;

        assert!(
            matches!(&err, ServiceError::Aborted(m) if m.contains("HGETALL obj header")),
            "a transient failure is retriable, and on a fresh read it replaces the header \
             HGETALL; got {err:?}"
        );
        assert_no_family_ticked(&fx.metrics);
        assert_no_refusal_kind_ticked(&fx.metrics);
        assert_eq!(
            (
                snapshot_reads(
                    &fx.metrics,
                    "structural_update",
                    "unfenced",
                    "transient_error"
                ),
                snapshot_reads(&fx.metrics, "structural_update", "unfenced", "ok"),
            ),
            (1, 1),
            "the failed read is a transient outcome and the read after it the real one"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unfenced_read_with_an_injected_permanent_failure_is_internal_and_counts_its_kind() {
        let fx = Fixture::seeded("sru-seam-permanent").await;

        let err = read_with_injected(
            &fx,
            FredError::new(FredErrorKind::Auth, "NOAUTH Authentication required."),
        )
        .await;

        assert!(
            matches!(&err, ServiceError::Internal(m) if m.contains("HGETALL obj header")),
            "a permanent failure is not retriable, and on a fresh read it replaces the header \
             HGETALL; got {err:?}"
        );
        assert_only_refusal_kind_ticked(&fx.metrics, "auth", 1);
        assert_no_family_ticked(&fx.metrics);
        assert_eq!(
            snapshot_reads(
                &fx.metrics,
                "structural_update",
                "unfenced",
                "permanent_error"
            ),
            1
        );
    }

    /// The bound on waiting for a read to park on a queued hold: unreachable
    /// on a healthy store, well inside nextest's 120 s hang-killer.
    const HOLD_BUDGET: std::time::Duration = std::time::Duration::from_secs(20);

    /// The seam on a script window (b.gvf): it is consulted once per command
    /// before anything is sent, so an error queued for the seeded object's
    /// one field row answers in place of that row's command and fails the
    /// read with no `EVALSHA` of the window script sent, classified as the
    /// plain read's failure of the row would be — under `HSTRLEN field`, the
    /// value-row read's first command — and ticking its refusal kind.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn structural_read_with_an_injected_value_row_error_fails_under_hstrlen_field_and_sends_no_window_script_b_gvf(
    ) {
        let fx = Fixture::seeded("sru-seam-value-row").await;
        let header_key = fx.tv.key_schema().obj_header(&fx.oid());
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        injector.push_at(
            ObjectSnapshotReadFaultTarget::nth_of_family(
                ObjectSnapshotReader::UnfencedFanout,
                "field",
                0,
            ),
            FredError::new(FredErrorKind::Auth, "NOAUTH Authentication required."),
        );
        fx.tv
            .client()
            .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));

        let (err, seen, _) = monitored(
            &fx.tv,
            read_unfenced_err(
                fx.tv.client(),
                fx.tv.key_schema(),
                &fx.oid(),
                SlimSnapshotReadConsumer::StructuralUpdate,
            ),
        )
        .await;

        assert_eq!(
            injector.remaining(),
            0,
            "the field row's command took the error"
        );
        assert!(
            matches!(&err, ServiceError::Internal(m) if m.contains("HSTRLEN field")),
            "a permanent failure of the value row is Internal under its first command; got \
             {err:?}"
        );
        let runs = seen
            .iter()
            .filter(|c| is_window_script_run(c, &header_key))
            .count();
        assert_eq!(
            runs, 0,
            "the seam answered the window before anything was sent: no window script ran"
        );
        assert_only_refusal_kind_ticked(&fx.metrics, "auth", 1);
        assert_no_family_ticked(&fx.metrics);
    }

    /// The fallback (b.gvf), driven by the seam: a cluster-state refusal
    /// queued for the seeded object's field row answers in place of its
    /// script window, before anything is sent. The read falls back for the
    /// rest of the read: that window and every later one are plain
    /// (`plain_fallback`; no `EVALSHA` of the window script is sent), the
    /// field row is read by the client's own `HSTRLEN`, and
    /// `live_edit_snapshot_read_window_fallback_total` ticks once. A row
    /// planted with both value shapes is not reported, as no plain window
    /// reports one. The next read, with nothing queued, runs its windows as
    /// scripts again, reports the row, and ticks nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn structural_read_with_an_injected_cluster_state_refusal_falls_back_to_plain_windows_b_gvf(
    ) {
        use fred::interfaces::HashesInterface;

        let fx = Fixture::seeded("sru-seam-fallback").await;
        let header_key = fx.tv.key_schema().obj_header(&fx.oid());
        let field_key =
            fx.tv
                .key_schema()
                .obj_field(&fx.oid(), &SECTION_ID.to_string(), FIELD_NAME);
        let _: () = fx
            .tv
            .client()
            .pool
            .hset(&field_key, [("hash", "ab".repeat(32))])
            .await
            .expect("fixture: plant a hash beside the row's inline value");
        let atomic = read_atomic(fx.tv.client(), fx.tv.key_schema(), fx.object_id).await;
        let fallbacks = || {
            sample_value(
                &render(&fx.metrics),
                "live_edit_snapshot_read_window_fallback_total",
                &[],
            )
            .expect("the fallback counter is seeded at zero")
        };
        let both_shapes = |captured: &crate::grpc::log_capture::Captured| {
            captured
                .events
                .iter()
                .filter(|e| {
                    e["fields"]["message"]
                        .as_str()
                        .is_some_and(|m| m.starts_with("value row carries both"))
                })
                .count()
        };
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        injector.push_at(
            ObjectSnapshotReadFaultTarget::nth_of_family(
                ObjectSnapshotReader::UnfencedFanout,
                "field",
                0,
            ),
            FredError::new(
                FredErrorKind::Unknown,
                "ERR Script attempted to access a slot not served",
            ),
        );
        fx.tv
            .client()
            .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
        let read = || {
            read_unfenced(
                fx.tv.client(),
                fx.tv.key_schema(),
                fx.object_id,
                SlimSnapshotReadConsumer::StructuralUpdate,
            )
        };

        let (fell_back, seen, captured) = monitored(&fx.tv, read()).await;

        assert_eq!(
            injector.remaining(),
            0,
            "the field row's window took the refusal"
        );
        assert_eq!(fell_back.state, compact_snapshot(&atomic.state));
        let events = window_events(&captured, fx.object_id);
        assert_eq!(
            windows_issued_as(&events, "script"),
            0,
            "no window completed as a script after the fallback; events: {events:?}"
        );
        assert!(
            windows_issued_as(&events, "plain_fallback") >= 2,
            "the refused window and round 3's are plain after the fallback; events: {events:?}"
        );
        assert_eq!(
            seen.iter()
                .filter(|c| is_window_script_run(c, &header_key))
                .count(),
            0,
            "no window script is sent: the seam answered the first, and the read issued every \
             later window plain"
        );
        let client_row_reads = seen
            .iter()
            .filter(|c| {
                !c.issued_by_script()
                    && c.name() == "HSTRLEN"
                    && c.args.get(1).map(String::as_str) == Some(field_key.as_str())
            })
            .count();
        assert_eq!(
            client_row_reads, 1,
            "the row was read by the plain re-issue"
        );
        assert_eq!(
            both_shapes(&captured),
            0,
            "a plain window reports no row holding both shapes"
        );
        assert_eq!(fallbacks(), 1.0, "one tick for the read that fell back");

        let (again, seen, captured) = monitored(&fx.tv, read()).await;
        assert_eq!(again.state, compact_snapshot(&atomic.state));
        assert!(
            seen.iter().any(|c| is_window_script_run(c, &header_key)),
            "the next read tries the window script again"
        );
        assert_eq!(
            both_shapes(&captured),
            1,
            "its script window reports the row holding both shapes"
        );
        assert_eq!(
            fallbacks(),
            1.0,
            "a read that does not fall back ticks nothing"
        );
    }

    /// A `SCRIPT FLUSH` between two windows of one read (b.gvf): a hold parks
    /// the read on its round-3 window (the root list's `LRANGE`, the second
    /// command of the `children` family on this marked object, after round
    /// 2's `LLEN`), the test flushes the store's scripts, and releases it.
    /// The window's `EVALSHA` meets `NOSCRIPT`, the script is loaded once
    /// from the read's connection, and the window is sent again: the read
    /// completes with the atomic read's compact projection.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn structural_read_after_a_script_flush_between_windows_reloads_the_window_script_once_b_gvf(
    ) {
        use fred::interfaces::LuaInterface;

        let fx = Fixture::seeded("sru-seam-script-flush").await;
        let header_key = fx.tv.key_schema().obj_header(&fx.oid());
        let atomic = read_atomic(fx.tv.client(), fx.tv.key_schema(), fx.object_id).await;
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        let hold = injector.push_hold_at(ObjectSnapshotReadFaultTarget::nth_of_family(
            ObjectSnapshotReader::UnfencedFanout,
            "children",
            1,
        ));
        fx.tv
            .client()
            .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));

        let mut tap = MonitorTap::start(&test_valkey_addr()).await;
        let flush_between = async {
            let parked = tokio::time::timeout(HOLD_BUDGET, hold.parked()).await;
            if parked.is_ok() {
                let _: () = fx
                    .tv
                    .client()
                    .pool
                    .next()
                    .script_flush(false)
                    .await
                    .expect("SCRIPT FLUSH must succeed");
            }
            hold.release();
            parked.is_ok()
        };
        let (read, parked) = tokio::join!(
            read_unfenced(
                fx.tv.client(),
                fx.tv.key_schema(),
                fx.object_id,
                SlimSnapshotReadConsumer::StructuralUpdate,
            ),
            flush_between
        );
        let fence = monitor_fence(fx.tv.prefix());
        let _: i64 = fx.tv.client().pool.exists(&fence).await.unwrap();
        let seen = tap.drain_until(&fence).await;

        assert!(parked, "fixture: the read parked on its round-3 window");
        assert_eq!(read.state, compact_snapshot(&atomic.state));
        let runs: Vec<usize> = seen
            .iter()
            .enumerate()
            .filter(|(_, c)| is_window_script_run(c, &header_key))
            .map(|(i, _)| i)
            .collect();
        let (Some(&first), Some(&last)) = (runs.first(), runs.last()) else {
            panic!("the read ran its window scripts; feed: {seen:?}");
        };
        let is_script = |c: &MonitoredCommand, sub: &str| {
            c.name() == "SCRIPT"
                && c.args
                    .get(1)
                    .is_some_and(|arg| arg.eq_ignore_ascii_case(sub))
        };
        let flush_at = seen[..last]
            .iter()
            .rposition(|c| is_script(c, "FLUSH"))
            .unwrap_or_else(|| panic!("fixture: the flush landed before the read's last window"));
        assert!(
            flush_at > first,
            "fixture: the flush landed after the read's first window script"
        );
        let reader = &seen[last].client;
        let loads = seen[flush_at..last]
            .iter()
            .filter(|c| {
                &c.client == reader
                    && is_script(c, "LOAD")
                    && c.args
                        .get(2)
                        .is_some_and(|body| body.contains("fanout-window-read-v1"))
            })
            .count();
        assert_eq!(
            loads, 1,
            "the window that met NOSCRIPT loaded the script once, from its own connection, then \
             was sent again"
        );
    }
}
