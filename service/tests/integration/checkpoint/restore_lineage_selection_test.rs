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

//! Restore-lineage **selection** against a real on-disk store — Task
//! `t2.dfh.yj.wg` Subtask `t3.dfh.yj.wg.e2`.
//!
//! The contract restated here is Epic 4's gather entry point
//! (`live_edit::checkpoint::gather_restore_lineage`), stated over artifacts
//! that a filesystem actually holds:
//!
//!   "The newest verifying lineage is the head tail plus the contiguous run of
//!    tails beneath it down to their base; a tail whose link resolves to
//!    nothing is a step-down verdict, not an error; and a store with no base is
//!    observably indistinguishable from a store with nothing in it."
//!
//! ## What this proves that the in-module unit tests structurally cannot
//!
//! The gather's own `#[cfg(test)]` tests and the Task's `InMemory` suites run
//! against stores whose listing is minted directly from the [`CheckpointKind`]
//! a put was handed, and whose meta never round-trips through JSON. Such a
//! store cannot mis-classify a generation, because it never parses a filename;
//! and it cannot produce a torn meta residual, because it has no files. Three
//! things the unit tests each see only half of meet the gather here, and only
//! on this adapter:
//!
//! 1. **Stem parsing** — `<oid>.<seq>.ompf` vs `<oid>.<to_seq>.tail`. A listing
//!    that parsed a `.tail` stem into a **base**-kind handle would move the
//!    candidate head and silently change the gather's answer, handing a `LETL`
//!    envelope to the reset-only decoder — the stranded-recovery shape FR-6 /
//!    SR-2.5 forbid. Test 1 asserts the parsed listing shape as a precondition
//!    so that failure is legible where it happens.
//! 2. **Mixed-kind listing order** — `adapters::generation_sort_key`: newest
//!    sequence first, `Base` before `Tail` at an equal sequence. The gather
//!    consumes that order verbatim and never re-sorts it.
//! 3. **Meta JSON round-trip**, including its *absence*. Test 2 removes the
//!    `<oid>.<to_seq>.tail.meta.json` companion from the test's own tempdir, so
//!    `read_generation_at` takes its artifact-present / meta-absent arm and —
//!    since b.hka (d) — REFUSES the generation as a typed `Permanent` rather
//!    than fabricating an identity-empty placeholder meta. Here that refusal is
//!    the one the adapter *really* emits rather than a wrapper's simulation of
//!    it, and the claim is that the walk recognises exactly that one
//!    `Permanent`, abandons the candidate, and serves the newest generation
//!    that CAN be restored. Its sibling (`gather_propagates_a_permanent_that_
//!    is_not_a_missing_meta_companion`) is the fence keeping that arm narrow.
//!    Test 5 is the same residual one member DOWN the chain, where the failure
//!    it guards is a different one — a silently TRUNCATED chain rather than a
//!    mis-chosen candidate; see that test's own doc for why the head position
//!    cannot cover it. Test 6 is the same residual on the other STEM PAIR — a
//!    base's `<oid>.<seq>.meta.json` rather than a tail's — which is the arm
//!    that used to serve the identity-empty placeholder, and which no
//!    tail-position case can hold red on its own.
//!
//! This is the same argument `recovery_mixed_kind_selection_test` makes for
//! Epic 2's SR-2.5 selection, sharpened by (3).
//!
//! ## Scope fences — both load-bearing
//!
//! - **Selection only, not restore.** Composition is this Epic's Task 1 and the
//!   restore wiring is Task 3. This file drives the gather entry point
//!   **directly** against the store: no `handle_stat_layer`, no
//!   `try_restore_layer`, no [`CheckpointHarness`](super::helpers::CheckpointHarness),
//!   and nothing asserted about restored `obj:hdr` state. The end-to-end
//!   restore-to-exact-state proof is Task 6's.
//! - **No Valkey.** A [`LocalDiskCheckpointStore`] over a `tempfile::tempdir()`
//!   is the whole fixture. `TestValkey::setup` is deliberately never called —
//!   the file lives in the integration binary for module-boundary and
//!   filesystem reasons, not because it needs a backing store, and staying
//!   Valkey-free also keeps it off the shared instance's `noeviction` keyspace.
//!   If a future change made a store-only gather need Valkey, that would mean
//!   the entry point had picked up restore-hook coupling the Task forbids
//!   (Epic 5's re-base calls the same gather from the checkpoint tick) and it
//!   is an escalation, not something to absorb by adding a harness.
//!
//! Every test here is filesystem-only: a handful of small writes and reads
//! under a per-test tempdir, running in milliseconds, orders of magnitude under
//! nextest's 40 s slow mark.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::store::{META_SUFFIX, OMPF_SUFFIX, TAIL_META_SUFFIX};
use live_edit::checkpoint::{
    gather_restore_lineage, no_deadline, CheckpointHandle, CheckpointKind, CheckpointStore,
    Exhaustion, LineageBreak, TAIL_ENVELOPE_MAGIC, TAIL_FORMAT_VERSION,
};
use live_edit::domain::ompf::OmpfContainer;
use tempfile::TempDir;

use super::helpers::{put_base_generation, put_tail_generation};

// ---------------------------------------------------------------------------
// Fixture constants
// ---------------------------------------------------------------------------

/// The object every test seeds under. Each test owns its own tempdir, so one
/// value is enough — except in test 4, which needs a second, deliberately
/// unseeded id.
const OBJECT_ID: u64 = 77;

/// The id used for the "nothing on disk" arm of test 4. Never seeded.
const UNSEEDED_OBJECT_ID: u64 = 78;

/// Identity stamped on every hand-put meta.
///
/// Any stable non-empty pair works here: nothing in this file reaches
/// `apply_checkpoint`'s SR-2.4 identity guard (see the scope fences above), so
/// these need not correspond to a live `le:{sid}:layer:{hex(name)}` mapping.
/// They stay non-empty anyway so that the base's meta round-trip through JSON
/// is observable — test 1 asserts it, and identity on a composed lineage is
/// base-sourced.
const SESSION_ID: &str = "restore-lineage-selection-session";
const LAYER_NAME: &str = "restore-lineage-selection-layer";

/// The superseded base. Present in every mixed-kind fixture and a member of no
/// lineage the gather returns — its whole job is to be the generation a
/// mis-parsed listing or a mis-walked link would wrongly select.
///
/// Test 6 is the deliberate exception: there it is the generation the selection
/// is *supposed* to land on, because the newest base above it is the unusable
/// one.
const OLD_BASE_SEQ: u64 = 5;

/// The lineage root every seeded tail claims.
const HEAD_BASE_SEQ: u64 = 9;

/// `to_seq` of the middle tail — `(9, 12]`.
const MID_TAIL_TO_SEQ: u64 = 12;

/// `to_seq` of the head tail — `(12, 15]` in tests 1/2, `(11, 15]` in test 3.
const HEAD_TAIL_TO_SEQ: u64 = 15;

/// The head tail's `from_seq` in test 3: a link into a generation that was
/// never written. Chosen strictly between [`HEAD_BASE_SEQ`] and
/// [`HEAD_TAIL_TO_SEQ`] so the walk resolves it as a **tail** predecessor
/// (`from_seq != base_snap_seq`) rather than as the lineage root.
const BROKEN_FROM_SEQ: u64 = 11;

/// Per-member fetch reserve handed to the gather.
///
/// Generous by design: these fetches are local file reads, and the reserve's
/// two jobs (reserve-then-abandon, and the per-fetch `timeout` cap) are unit
/// tested against a fake store. Sizing it large keeps a slow CI filesystem from
/// turning any test here into a [`LineageBreak::Budget`] step-down, which would
/// be a false red on assertions about lineage shape.
const FETCH_RESERVE: Duration = Duration::from_secs(60);

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

/// A LocalDisk store over a fresh tempdir, plus the `TempDir` guard the caller
/// must hold to end-of-test (its `Drop` recursively removes the root).
fn local_disk_store() -> (TempDir, Arc<dyn CheckpointStore>) {
    let dir = tempfile::tempdir().expect("tempdir must be creatable");
    let store = LocalDiskCheckpointStore::new(dir.path().to_path_buf())
        .expect("LocalDiskCheckpointStore::new must succeed on a fresh tempdir");
    (dir, Arc::new(store))
}

/// `<root>/.live-edit/layers/<oid>.<to_seq>.tail.meta.json` — the TAIL stem's
/// meta companion, built in `LocalDiskCheckpointStore::tail_meta_path` (`:285`).
///
/// The suffix is the shared constant rather than an inline literal, per the
/// adapters-module rule.
fn tail_meta_path(root: &TempDir, object_id: u64, to_seq: u64) -> PathBuf {
    layers_dir(root).join(format!("{object_id}.{to_seq}{TAIL_META_SUFFIX}"))
}

/// `<root>/.live-edit/layers/<oid>.<snap_seq>.meta.json` — the BASE stem's meta
/// companion, the exact analogue of [`tail_meta_path`].
///
/// Same literal-segment / shared-suffix-constant reasoning as that function.
/// The suffix must come from [`META_SUFFIX`] rather than being spelled inline
/// because `.meta.json` is a proper substring of `.tail.meta.json`, and a
/// hand-written literal is exactly how a base helper silently starts addressing
/// a tail's companion.
fn base_meta_path(root: &TempDir, object_id: u64, snap_seq: u64) -> PathBuf {
    layers_dir(root).join(format!("{object_id}.{snap_seq}{META_SUFFIX}"))
}

/// `<root>/.live-edit/layers/<oid>.<snap_seq>.ompf` — the BASE artifact itself.
fn base_artifact_path(root: &TempDir, object_id: u64, snap_seq: u64) -> PathBuf {
    layers_dir(root).join(format!("{object_id}.{snap_seq}{OMPF_SUFFIX}"))
}

/// `<root>/.live-edit/layers` — the adapter's generation directory.
///
/// The two path segments are literals because `NAMESPACE_DIR` / `LAYERS_SUBDIR`
/// are private to the adapter; the layout they spell is the one documented at
/// `src/checkpoint/adapters/local_disk.rs:13`.
fn layers_dir(root: &TempDir) -> PathBuf {
    root.path().join(".live-edit").join("layers")
}

/// Seed the mixed-kind fixture tests 1 and 2 share: `base@5`, `base@9`,
/// `tail(9, 12]`, `tail(12, 15]`.
///
/// Composed from the shared `helpers.rs` builders rather than re-implementing
/// them — they already write real `encode_tail_envelope` bytes and
/// identity-complete metas, and a second seeder would be free to drift.
async fn seed_head_lineage(store: &Arc<dyn CheckpointStore>) {
    put_base_generation(store, OBJECT_ID, SESSION_ID, LAYER_NAME, OLD_BASE_SEQ).await;
    put_base_generation(store, OBJECT_ID, SESSION_ID, LAYER_NAME, HEAD_BASE_SEQ).await;
    put_tail_generation(
        store,
        OBJECT_ID,
        SESSION_ID,
        LAYER_NAME,
        HEAD_BASE_SEQ,
        HEAD_BASE_SEQ,
        MID_TAIL_TO_SEQ,
    )
    .await;
    put_tail_generation(
        store,
        OBJECT_ID,
        SESSION_ID,
        LAYER_NAME,
        HEAD_BASE_SEQ,
        MID_TAIL_TO_SEQ,
        HEAD_TAIL_TO_SEQ,
    )
    .await;
}

/// `(seq, kind)` for a listing, in listing order — the shape assertions read.
fn listing_shape(listing: &[CheckpointHandle]) -> Vec<(u64, CheckpointKind)> {
    listing.iter().map(|h| (h.seq(), h.kind)).collect()
}

/// The listed handle at `(seq, kind)`, or a panic naming what was listed.
fn handle_at(listing: &[CheckpointHandle], seq: u64, kind: CheckpointKind) -> CheckpointHandle {
    listing
        .iter()
        .find(|h| h.seq() == seq && h.kind == kind)
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "listing must hold {} at sequence {seq}; listed: {:?}",
                kind.as_str(),
                listing_shape(listing)
            )
        })
}

// ---------------------------------------------------------------------------
// 1. Mixed-kind LocalDisk listing → the head lineage, in replay order
// ---------------------------------------------------------------------------

#[tokio::test]
async fn gather_over_a_localdisk_mixed_kind_store_returns_the_head_lineage_in_order() {
    let (dir, store) = local_disk_store();
    seed_head_lineage(&store).await;

    // PRECONDITION — the adapter's stem parsing and sort order, asserted where
    // a regression in either is legible. A `.tail` stem parsed into a Base
    // handle, or a sort that put `base@9` above `tail@15`, would move the
    // candidate head and change every assertion below for a reason that has
    // nothing to do with the gather.
    let listing = store
        .list_layer_checkpoints(OBJECT_ID)
        .await
        .expect("listing a seeded LocalDisk object must succeed");
    assert_eq!(
        listing_shape(&listing),
        vec![
            (HEAD_TAIL_TO_SEQ, CheckpointKind::Tail),
            (MID_TAIL_TO_SEQ, CheckpointKind::Tail),
            (HEAD_BASE_SEQ, CheckpointKind::Base),
            (OLD_BASE_SEQ, CheckpointKind::Base),
        ],
        "LocalDisk must list both stems newest-first with Base before Tail at an equal sequence"
    );

    let mut selection =
        gather_restore_lineage(store.as_ref(), OBJECT_ID, no_deadline(), FETCH_RESERVE)
            .await
            .expect("the single listing must succeed");
    let lineage = selection
        .next_restorable()
        .await
        .expect("a healthy on-disk lineage is never an error")
        .expect("a base plus two contiguous tails must be restorable");

    // The lineage root, and the sequence the lineage restores TO.
    assert_eq!(
        lineage.base.handle.seq(),
        HEAD_BASE_SEQ,
        "the newest base roots the lineage"
    );
    assert_eq!(lineage.base.handle.kind, CheckpointKind::Base);
    assert_eq!(
        lineage.head_seq, HEAD_TAIL_TO_SEQ,
        "S is the head tail's to_seq"
    );
    assert_eq!(lineage.chain_len(), 2);
    assert!(!lineage.is_base_only());
    assert_eq!(lineage.outcome_label(), "gathered_composed");

    // Replay order is pinned oldest-first.
    let spans: Vec<(u64, u64)> = lineage
        .tails
        .iter()
        .map(|t| (t.header.from_seq, t.header.to_seq))
        .collect();
    assert_eq!(
        spans,
        vec![
            (HEAD_BASE_SEQ, MID_TAIL_TO_SEQ),
            (MID_TAIL_TO_SEQ, HEAD_TAIL_TO_SEQ)
        ],
        "tails must come back oldest-first with resolved (from, to] links"
    );

    for tail in &lineage.tails {
        assert_eq!(
            tail.header.base_snap_seq, HEAD_BASE_SEQ,
            "every member roots at the lineage's base"
        );
        assert_eq!(tail.handle.kind, CheckpointKind::Tail);
        assert_eq!(
            tail.handle.seq(),
            tail.header.to_seq,
            "the handle addresses the tail by its to_seq"
        );
        assert_eq!(tail.format_version, TAIL_FORMAT_VERSION);
        // Bytes come back UNDECODED — composition (Task 1) decodes, the gather
        // never does. `LETL` at offset 0 is the observable form of that.
        assert_eq!(
            tail.bytes.get(..4),
            Some(&TAIL_ENVELOPE_MAGIC[..]),
            "the gather must hand back the raw envelope"
        );
    }

    // `base@5` is a member of nothing.
    assert_ne!(lineage.base.handle.seq(), OLD_BASE_SEQ);
    assert!(
        lineage.tails.iter().all(|t| t.handle.seq() != OLD_BASE_SEQ),
        "the superseded base must not appear in the returned lineage"
    );

    // Meta round-tripped through JSON — identity on a composed lineage is
    // base-sourced, so the base's meta is the one that has to survive.
    assert_eq!(lineage.base.meta.session_id, SESSION_ID);
    assert_eq!(lineage.base.meta.layer_name, LAYER_NAME);
    assert_eq!(lineage.base.meta.kind, CheckpointKind::Base);

    assert!(
        selection.step_downs().is_empty(),
        "a healthy lineage abandons no candidate: {:?}",
        selection.step_downs()
    );
    assert!(selection.exhaustion().is_none());

    drop(dir);
}

// ---------------------------------------------------------------------------
// 2. Torn tail meta on the HEAD — the candidate is unusable and the selection
//    steps down (b.hka (d))
// ---------------------------------------------------------------------------

/// A durable envelope whose `.tail.meta.json` companion is gone is REFUSED by
/// the adapter, the head candidate is abandoned, and the selection serves an
/// older restorable generation.
///
/// # What this case used to assert, and why it changed
///
/// It previously asserted that the adapter served a lineage-less **placeholder**
/// meta and the gather resolved the full run off the envelope headers anyway.
/// b.hka (d) removed that placeholder: an artifact whose meta companion never
/// landed carries no `session_id` / `layer_name`, and a generation whose
/// identity the restore path cannot verify is unrestorable — so the adapter now
/// refuses it as a typed `Permanent`
/// (`CheckpointError::is_missing_meta_companion`) instead of fabricating one.
///
/// The header-is-authoritative rule that the old assertion rested on is
/// **unchanged and still load-bearing** — it is what makes a meta that
/// *disagrees* with its envelope harmless, and test 1 and the
/// `a_meta_that_disagrees_with_its_envelope_loses_to_the_header` unit case both
/// still pin it. What changed is only that the meta-ABSENT case no longer rides
/// it.
///
/// # Why stepping down is the better outcome, not merely a stricter one
///
/// The placeholder carried an unrestorable member forward to fail three layers
/// up, as a generic `NotFound` naming neither the artifact nor the missing file.
/// The refusal is caught by exactly one arm of the walk and converted into a
/// step-down, so DR serves the newest generation that CAN be restored — here the
/// lineage headed at the middle tail — instead of failing outright.
#[tokio::test]
async fn gather_steps_down_when_the_head_tails_meta_companion_is_missing() {
    let (dir, store) = local_disk_store();
    seed_head_lineage(&store).await;

    // Produce the residual: a durable envelope whose `.tail.meta.json`
    // companion is gone. Publish order (meta before envelope) and delete order
    // (envelope before meta) make this unreachable in normal operation, so it
    // can only be reproduced by removing the file — inside this test's OWN
    // tempdir, which is fixture behaviour, not a repo-tree deletion.
    let meta_path = tail_meta_path(&dir, OBJECT_ID, HEAD_TAIL_TO_SEQ);
    assert!(
        meta_path.is_file(),
        "the seeded tail must have published its meta companion at {}",
        meta_path.display()
    );
    tokio::fs::remove_file(&meta_path)
        .await
        .expect("removing the tail meta companion must succeed");

    let listing = store
        .list_layer_checkpoints(OBJECT_ID)
        .await
        .expect("listing must succeed with a torn meta");
    assert_eq!(
        listing_shape(&listing),
        vec![
            (HEAD_TAIL_TO_SEQ, CheckpointKind::Tail),
            (MID_TAIL_TO_SEQ, CheckpointKind::Tail),
            (HEAD_BASE_SEQ, CheckpointKind::Base),
            (OLD_BASE_SEQ, CheckpointKind::Base),
        ],
        "the listing is stem-driven, so a missing meta must not hide the tail"
    );

    // PRECONDITION — the adapter really refuses, with the TYPED marker.
    // Asserted before the gather outcome so an adapter change that stopped
    // classifying fails HERE rather than turning the step-down assertions below
    // into a vacuous pass over a healthy store.
    let head_handle = handle_at(&listing, HEAD_TAIL_TO_SEQ, CheckpointKind::Tail);
    let refusal = store
        .get_layer_checkpoint(&head_handle)
        .await
        .expect_err("a durable envelope with no meta companion must be refused");
    assert!(
        refusal.is_missing_meta_companion(),
        "the refusal must carry the typed artifact-present / meta-absent marker — the walk steps \
         down on that predicate and on nothing else, so an untyped Permanent here would hard-fail \
         the restore instead. Got {refusal:?}"
    );

    let mut selection =
        gather_restore_lineage(store.as_ref(), OBJECT_ID, no_deadline(), FETCH_RESERVE)
            .await
            .expect("the single listing must succeed");
    let lineage = selection
        .next_restorable()
        .await
        .expect("an unrestorable member is a step-down verdict, not an error")
        .expect("an older restorable generation must still be served");

    // THE CLAIM: DR still restores, from the next-older generation.
    assert_eq!(lineage.base.handle.seq(), HEAD_BASE_SEQ);
    assert_eq!(
        lineage.head_seq, MID_TAIL_TO_SEQ,
        "the head candidate was abandoned, so the answer is the lineage headed at the middle \
         tail — the newest one that can actually be restored"
    );
    assert_eq!(lineage.chain_len(), 1);
    let spans: Vec<(u64, u64)> = lineage
        .tails
        .iter()
        .map(|t| (t.header.from_seq, t.header.to_seq))
        .collect();
    assert_eq!(spans, vec![(HEAD_BASE_SEQ, MID_TAIL_TO_SEQ)]);

    // And the step-down is OBSERVABLE, under its own outcome label.
    let step_downs = selection.step_downs();
    assert_eq!(
        step_downs.len(),
        1,
        "exactly one candidate was abandoned: {step_downs:?}"
    );
    assert_eq!(step_downs[0].head_seq, HEAD_TAIL_TO_SEQ);
    assert_eq!(step_downs[0].head_kind, CheckpointKind::Tail);
    match &step_downs[0].reason {
        LineageBreak::MissingMetaCompanion { seq, message } => {
            assert_eq!(*seq, HEAD_TAIL_TO_SEQ);
            assert!(
                message.contains(TAIL_META_SUFFIX),
                "the record must carry the adapter's own message naming the absent companion, \
                 which is the one thing an operator needs to go look at; got {message}"
            );
        }
        other => panic!(
            "a torn put must be its OWN break, not folded into `Decode` — the two carry \
             different operator remediations (a write-path interruption vs. bit-rot or version \
             skew). Got {other:?}"
        ),
    }
    assert_eq!(
        step_downs[0].reason.reason_label(),
        "stepped_down_missing_meta",
        "and the outcome label must be the dedicated one, so a dashboard does not report a \
         routine torn put as storage corruption"
    );
    assert!(selection.exhaustion().is_none());

    drop(dir);
}

/// The FENCE on the arm above: a `Permanent` that is NOT the missing-meta
/// marker still propagates as an error and is never converted into a step-down.
///
/// This is the assertion that keeps the step-down arm narrow. Widening it would
/// report a credential outage, a malformed response, or a store outage as "this
/// layer's checkpoint is unusable" and quietly restore older state — which is
/// the wrong verdict in the one situation where an operator most needs the right
/// one, and it is silent by construction because a stepped-down restore still
/// ends its attempt on `result="restored"`.
///
/// A corrupt `.tail.meta.json` is the cheapest non-missing-meta `Permanent` this
/// adapter produces: the bytes are durable and were last written by us, so a
/// retry cannot fix them.
#[tokio::test]
async fn gather_propagates_a_permanent_that_is_not_a_missing_meta_companion() {
    let (dir, store) = local_disk_store();
    seed_head_lineage(&store).await;

    let meta_path = tail_meta_path(&dir, OBJECT_ID, HEAD_TAIL_TO_SEQ);
    tokio::fs::write(&meta_path, b"not-json-at-all{{{")
        .await
        .expect("overwriting the meta companion with garbage must succeed");

    let listing = store
        .list_layer_checkpoints(OBJECT_ID)
        .await
        .expect("listing must succeed with a corrupt meta");
    let head_handle = handle_at(&listing, HEAD_TAIL_TO_SEQ, CheckpointKind::Tail);

    // PRECONDITION — this really is a Permanent, and really is NOT the
    // missing-meta one. Without both halves the case could pass because the
    // adapter happened to return some other class entirely.
    let direct = store
        .get_layer_checkpoint(&head_handle)
        .await
        .expect_err("a corrupt meta companion must be refused");
    assert!(
        direct.is_permanent(),
        "corrupt durable bytes are Permanent — a retry re-reads the same bytes. Got {direct:?}"
    );
    assert!(
        !direct.is_missing_meta_companion(),
        "and it must NOT carry the missing-meta marker: the companion is PRESENT, it is the \
         bytes that are bad. Got {direct:?}"
    );

    let mut selection =
        gather_restore_lineage(store.as_ref(), OBJECT_ID, no_deadline(), FETCH_RESERVE)
            .await
            .expect("the single listing must succeed");
    let err = selection
        .next_restorable()
        .await
        .expect_err("every Permanent OTHER than the missing-meta marker must propagate");
    assert!(
        err.is_permanent() && !err.is_missing_meta_companion(),
        "the propagated error must be the adapter's own, unconverted. Got {err:?}"
    );
    assert!(
        selection.step_downs().is_empty(),
        "and nothing may be recorded as a step-down: a propagating error is not an abandoned \
         candidate. Got {:?}",
        selection.step_downs()
    );

    drop(dir);
}

// ---------------------------------------------------------------------------
// 3. Broken on-disk head lineage → one observable step-down
// ---------------------------------------------------------------------------

#[tokio::test]
async fn gather_steps_down_observably_when_the_on_disk_head_lineage_is_broken() {
    let (dir, store) = local_disk_store();
    put_base_generation(&store, OBJECT_ID, SESSION_ID, LAYER_NAME, OLD_BASE_SEQ).await;
    put_base_generation(&store, OBJECT_ID, SESSION_ID, LAYER_NAME, HEAD_BASE_SEQ).await;
    // `(11, 15]` rooted at base@9: sequence 11 holds no generation, so the head
    // tail's predecessor link resolves to nothing.
    put_tail_generation(
        &store,
        OBJECT_ID,
        SESSION_ID,
        LAYER_NAME,
        HEAD_BASE_SEQ,
        BROKEN_FROM_SEQ,
        HEAD_TAIL_TO_SEQ,
    )
    .await;

    let listing = store
        .list_layer_checkpoints(OBJECT_ID)
        .await
        .expect("listing must succeed");
    assert_eq!(
        listing_shape(&listing),
        vec![
            (HEAD_TAIL_TO_SEQ, CheckpointKind::Tail),
            (HEAD_BASE_SEQ, CheckpointKind::Base),
            (OLD_BASE_SEQ, CheckpointKind::Base),
        ],
        "the broken tail heads the listing — it is the first candidate"
    );

    let mut selection =
        gather_restore_lineage(store.as_ref(), OBJECT_ID, no_deadline(), FETCH_RESERVE)
            .await
            .expect("the single listing must succeed");
    let lineage = selection
        .next_restorable()
        .await
        .expect("a broken link is a verdict, never an error")
        .expect("the newest base is still restorable beneath the broken tail");

    // The floor answer: base@9 alone, which is byte-identical to the
    // pre-Epic-4 selection result.
    assert!(lineage.is_base_only());
    assert_eq!(lineage.base.handle.seq(), HEAD_BASE_SEQ);
    assert_eq!(lineage.head_seq, HEAD_BASE_SEQ);
    assert_eq!(lineage.chain_len(), 0);
    assert_eq!(lineage.outcome_label(), "gathered_base");

    // Observability, as the returned reason RECORD. The gather takes no metrics
    // handle by design (the increment is the caller's), so the record is the
    // observable — and it is strictly richer than a counter cell: it names the
    // abandoned head AND the link that broke.
    let records = selection.step_downs();
    assert_eq!(
        records.len(),
        1,
        "exactly one candidate is abandoned: {records:?}"
    );
    let record = &records[0];
    assert_eq!(record.head_seq, HEAD_TAIL_TO_SEQ);
    assert_eq!(record.head_kind, CheckpointKind::Tail);
    assert_eq!(
        record.reason,
        LineageBreak::LinkAbsent {
            seq: BROKEN_FROM_SEQ,
            kind: CheckpointKind::Tail,
            // The listing never held sequence 11 — the absence is known before
            // any fetch is issued, so no `get_layer_checkpoint` is spent on it.
            at_fetch: false,
        },
        "the break must name the missing predecessor exactly"
    );
    assert_eq!(record.reason.reason_label(), "stepped_down_link_broken");
    assert_eq!(record.reason.link_seq(), BROKEN_FROM_SEQ);

    assert!(
        selection.exhaustion().is_none(),
        "the selection resolved a lineage, so it is not exhausted"
    );

    drop(dir);
}

// ---------------------------------------------------------------------------
// 4. Tails-only and empty are the same observable outcome — and neither errs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn gather_over_a_tails_only_localdisk_store_is_indistinguishable_from_an_empty_store() {
    let (dir, store) = local_disk_store();

    // Arm (a) — tails on disk, no base anywhere. The tails claim a root at
    // sequence 9 that was never written.
    put_tail_generation(
        &store,
        OBJECT_ID,
        SESSION_ID,
        LAYER_NAME,
        HEAD_BASE_SEQ,
        HEAD_BASE_SEQ,
        MID_TAIL_TO_SEQ,
    )
    .await;
    put_tail_generation(
        &store,
        OBJECT_ID,
        SESSION_ID,
        LAYER_NAME,
        HEAD_BASE_SEQ,
        MID_TAIL_TO_SEQ,
        HEAD_TAIL_TO_SEQ,
    )
    .await;

    let listing = store
        .list_layer_checkpoints(OBJECT_ID)
        .await
        .expect("listing must succeed");
    assert_eq!(
        listing_shape(&listing),
        vec![
            (HEAD_TAIL_TO_SEQ, CheckpointKind::Tail),
            (MID_TAIL_TO_SEQ, CheckpointKind::Tail),
        ],
        "the fixture really is tails-only — a base here would void the test"
    );

    let mut tails_only =
        gather_restore_lineage(store.as_ref(), OBJECT_ID, no_deadline(), FETCH_RESERVE)
            .await
            .expect("listing a tails-only object must succeed");
    // The load-bearing assertion: a VERDICT, never an `Err`. Letting a tail
    // envelope's `LETL` magic reach the reset decoder — and reporting the
    // resulting `Permanent` as the outcome — is the stranded-recovery class
    // FR-6 / SR-2.5 forbid, and is what operators would page on.
    let tails_only_outcome = tails_only
        .next_restorable()
        .await
        .unwrap_or_else(|e| panic!("a tails-only store must be a verdict, never an error: {e}"));
    assert!(
        tails_only_outcome.is_none(),
        "no lineage has a reset-decodable root without a base"
    );
    assert_eq!(
        tails_only.exhaustion(),
        Some(&Exhaustion::NoBase {
            generations: 2,
            newest_sequence: HEAD_TAIL_TO_SEQ,
        }),
        "the diagnostic distinction survives even though the outcome label does not"
    );
    assert!(
        tails_only.step_downs().is_empty(),
        "no candidate is walked at all: {:?}",
        tails_only.step_downs()
    );

    // Arm (b) — an object with nothing on disk. Same store, so the two arms
    // differ ONLY in what was seeded; that also pins the gather's per-object
    // scoping (the tails above are invisible here).
    let mut empty = gather_restore_lineage(
        store.as_ref(),
        UNSEEDED_OBJECT_ID,
        no_deadline(),
        FETCH_RESERVE,
    )
    .await
    .expect("listing an unseeded object must succeed, not error");
    let empty_outcome = empty
        .next_restorable()
        .await
        .unwrap_or_else(|e| panic!("an empty store must be a verdict, never an error: {e}"));
    assert!(empty_outcome.is_none());
    assert_eq!(empty.exhaustion(), Some(&Exhaustion::EmptyListing));
    assert!(empty.step_downs().is_empty());

    // Indistinguishable at the reported outcome — the whole point of the pair.
    //
    // Both arms are CONCLUSIVE exhaustions: each adjudicated every listed
    // `Base` (vacuously — there are none), so each has positive evidence that
    // the layer is unrestorable and each earns the terminal `NOT_FOUND` that
    // `exhausted` accompanies. That is why the fold survives b.hxk, which split
    // `Exhaustion::BudgetInconclusive` onto its own `exhausted_budget` cell: the
    // split is between conclusive and inconclusive, not between the diagnostic
    // variants, and neither arm here is inconclusive.
    assert_eq!(
        tails_only.exhaustion().map(Exhaustion::outcome_label),
        empty.exhaustion().map(Exhaustion::outcome_label),
        "both arms must report the same `exhausted` outcome"
    );
    assert_eq!(
        tails_only
            .exhaustion()
            .expect("arm (a) recorded an exhaustion")
            .outcome_label(),
        "exhausted"
    );

    drop(dir);
}

// ---------------------------------------------------------------------------
// 5. Torn tail meta on a NON-head member — the chain is not truncated
// ---------------------------------------------------------------------------

/// Test 2's residual, moved one member down the chain: the selection steps
/// PAST both candidates that need the meta-less member and lands on the base.
///
/// # Why the mid-chain position is a separate case
///
/// Test 2 removes the HEAD tail's companion, so exactly one candidate is
/// abandoned and the next one down is intact. Here the unusable member sits
/// INSIDE the chain, which means it takes out two candidates rather than one:
/// the head lineage `{base@9, (9,12], (12,15]}` needs it to walk down, and the
/// candidate headed at the mid tail IS it. Both must be abandoned, and the
/// answer must be the base alone.
///
/// The failure this guards is a silent SHORTENING rather than a wrong choice.
/// An implementation that treated an unusable middle member as a chain boundary
/// — resolving the walk as complete at `tail(9,12]` and returning a one-tail
/// lineage headed at `MID_TAIL_TO_SEQ` — would claim a restore covering commits
/// it cannot actually replay, and every assertion in test 2 stays green under
/// it, because test 2's unusable member sits at the head where there is no
/// successor to strand.
///
/// # What this case used to assert
///
/// Before b.hka (d) the adapter served a lineage-less **placeholder** meta here
/// and the gather walked THROUGH the member off its envelope header, returning
/// the whole two-tail run. That placeholder is gone: an artifact whose meta
/// companion never landed carries no identity and is unrestorable, so the
/// adapter refuses it and the member is genuinely unusable rather than merely
/// under-described. The header-is-authoritative rule the old assertion rested on
/// is unchanged — test 1 and the disagreeing-meta unit case still pin it — it
/// simply no longer covers the meta-ABSENT case.
///
/// # Provenance
///
/// This case belongs to `t3.dfh.be.mk.xu` (Epic 5's meta-not-keyed retention
/// pin) and LANDED HERE rather than there, deliberately. At retention the
/// mid-chain question is moot — Epic 5's `plan_retention` groups by fetch-free
/// adjacency over `(kind, sequence)` and never walks a link, so there is no
/// chain for an unusable member to truncate. At the gather there is, and this
/// file is the lane that owns gather-side selection against a real on-disk
/// store. The audit that routed it here is recorded in `checkpointer.rs`'s
/// meta-perturbation group banner, which cites this test by name as one of the
/// three discharging the 2026-08-28 tail-meta deferral.
#[tokio::test]
async fn gather_steps_down_past_every_candidate_needing_a_meta_less_mid_chain_tail() {
    let (dir, store) = local_disk_store();
    seed_head_lineage(&store).await;

    // The residual, on the MIDDLE tail this time. Same adapter path as test 2:
    // remove the companion the adapter itself published, inside this test's own
    // tempdir.
    let mid_meta_path = tail_meta_path(&dir, OBJECT_ID, MID_TAIL_TO_SEQ);
    assert!(
        mid_meta_path.is_file(),
        "the seeded middle tail must have published its meta companion at {}",
        mid_meta_path.display()
    );
    tokio::fs::remove_file(&mid_meta_path)
        .await
        .expect("removing the middle tail's meta companion must succeed");

    let listing = store
        .list_layer_checkpoints(OBJECT_ID)
        .await
        .expect("listing must succeed with a torn mid-chain meta");
    assert_eq!(
        listing_shape(&listing),
        vec![
            (HEAD_TAIL_TO_SEQ, CheckpointKind::Tail),
            (MID_TAIL_TO_SEQ, CheckpointKind::Tail),
            (HEAD_BASE_SEQ, CheckpointKind::Base),
            (OLD_BASE_SEQ, CheckpointKind::Base),
        ],
        "the listing is stem-driven, so a missing mid-chain meta must not hide the tail"
    );

    // PRECONDITIONS — the refusal is on the MIDDLE member and the head's meta
    // is intact. Both halves matter: a removal that missed would leave a healthy
    // store, and a removal that took out the head's companion too would silently
    // re-run test 2.
    let mid_handle = handle_at(&listing, MID_TAIL_TO_SEQ, CheckpointKind::Tail);
    let refusal = store
        .get_layer_checkpoint(&mid_handle)
        .await
        .expect_err("a durable envelope with no meta companion must be refused");
    assert!(
        refusal.is_missing_meta_companion(),
        "the middle member must be refused with the TYPED marker — the one Permanent the walk \
         is allowed to convert into a step-down. Got {refusal:?}"
    );
    let head_handle = handle_at(&listing, HEAD_TAIL_TO_SEQ, CheckpointKind::Tail);
    let intact = store
        .get_layer_checkpoint(&head_handle)
        .await
        .expect("fetching the head envelope must not error")
        .expect("the head envelope is durable");
    assert_eq!(
        (intact.meta.base_snap_seq, intact.meta.from_seq),
        (Some(HEAD_BASE_SEQ), Some(MID_TAIL_TO_SEQ)),
        "the HEAD's meta must be untouched — this case is about an unusable member strictly \
         inside the chain, and a head that had also lost its companion would re-run test 2"
    );

    let mut selection =
        gather_restore_lineage(store.as_ref(), OBJECT_ID, no_deadline(), FETCH_RESERVE)
            .await
            .expect("the single listing must succeed");
    let lineage = selection
        .next_restorable()
        .await
        .expect("an unrestorable member is a step-down verdict, not an error")
        .expect("the base beneath the broken chain must still be served");

    // THE CLAIM: the answer is the BASE ALONE. Not `{base@9, tail(9,12]}` — the
    // shape an implementation that treated the unusable member as a chain
    // boundary would return, claiming a restore over commits (9, 12] it cannot
    // replay because those are exactly the commits the refused envelope carries.
    assert_eq!(lineage.base.handle.seq(), HEAD_BASE_SEQ);
    assert!(
        lineage.is_base_only(),
        "the mid tail is the member that is unusable, so no chain can include it and none of \
         the candidates above it can verify. Got a lineage of {} tail(s)",
        lineage.chain_len()
    );
    assert_eq!(lineage.head_seq, HEAD_BASE_SEQ);

    // Both candidates that needed the unusable member are recorded, and both
    // under the dedicated outcome.
    let step_downs = selection.step_downs();
    assert_eq!(
        step_downs.iter().map(|r| r.head_seq).collect::<Vec<_>>(),
        vec![HEAD_TAIL_TO_SEQ, MID_TAIL_TO_SEQ],
        "newest-first: the head lineage is abandoned when its walk reaches the unusable member, \
         then the candidate headed AT that member is abandoned on the cached fact. Got \
         {step_downs:?}"
    );
    for record in step_downs {
        assert_eq!(
            record.reason.reason_label(),
            "stepped_down_missing_meta",
            "every abandonment here has one cause — the torn put — and must say so rather than \
             reporting as a decode failure. Got {record:?}"
        );
        assert!(
            matches!(record.reason, LineageBreak::MissingMetaCompanion { seq, .. } if seq == MID_TAIL_TO_SEQ),
            "and each must name the OFFENDING member's sequence ({MID_TAIL_TO_SEQ}), not the \
             abandoned candidate's head — that is the artifact an operator goes and looks at. \
             Got {record:?}"
        );
    }
    assert!(selection.exhaustion().is_none());

    drop(dir);
}

// ---------------------------------------------------------------------------
// 6. Torn BASE meta on the newest base — the selection steps down to the older
//    base (b.hka (d), base arm)
// ---------------------------------------------------------------------------

/// The **base** arm of b.hka (d): a durable `<oid>.<seq>.ompf` whose
/// `<oid>.<seq>.meta.json` companion is gone is refused by the adapter, the
/// newest base is abandoned, and the selection serves the base beneath it.
///
/// # Why this needs its own case
///
/// Tests 2 and 5 both perturb a TAIL's companion, so between them they cover
/// only one of the two stem pairs `read_generation_at` addresses
/// (`src/checkpoint/adapters/local_disk.rs:597-606`). The kinds take
/// *different* branches on the way in — different artifact suffix, different
/// meta suffix — and the base branch is the one that carried the older
/// placeholder: before b.hka (d) it synthesized `CheckpointMeta::default()` and
/// returned `Ok(Some(..))`, so the newest base was SELECTED, carried an empty
/// `session_id` / `layer_name` three layers up, and was refused at apply by
/// `apply::live_mapping_resolves_blob_identity` as a generic `NotFound` naming
/// neither the artifact nor the missing file. DR reported the layer
/// unrecoverable while a perfectly restorable older base sat right beneath it.
/// That is the regression this pins, and no tail-position case can: with the
/// placeholder restored, tests 2 and 5 both go red on their own (their member is
/// a tail), but nothing here would notice the base arm regressing alone.
///
/// It is also the only position where the step-down record's `head_kind` is
/// `Base`, and the only one where the walk short-circuits at
/// `verify_links`' base-headed arm (`lineage.rs:1040`) rather than after a link
/// hop — so the `LinkFact::Unusable` → `LineageBreak::MissingMetaCompanion`
/// conversion is exercised on the head fact itself, with zero links walked.
///
/// # Fixture: two bases, no tails
///
/// Deliberately NOT [`seed_head_lineage`]. Tails above the abandoned base would
/// add candidates whose own walks reach the unusable member, turning the
/// "exactly one step-down record" claim into an arithmetic argument about the
/// chain instead of the flat statement it should be. Two bases is the smallest
/// fixture in which "stepped DOWN" means anything at all.
///
/// # Scope
///
/// "The restore succeeds from the older base" is asserted at this file's
/// boundary (see the module's scope fences): the served lineage is base-rooted
/// at [`OLD_BASE_SEQ`], carries the identity meta the SR-2.4 guard needs, and
/// its bytes really are a reset-decodable OMPF container rather than a `LETL`
/// envelope. Driving `try_restore_layer` end-to-end over a stepped-down
/// selection belongs to the restore-wiring lane, not here.
#[tokio::test]
async fn gather_steps_down_to_the_older_base_when_the_newest_bases_meta_companion_is_missing() {
    let (dir, store) = local_disk_store();
    put_base_generation(&store, OBJECT_ID, SESSION_ID, LAYER_NAME, OLD_BASE_SEQ).await;
    put_base_generation(&store, OBJECT_ID, SESSION_ID, LAYER_NAME, HEAD_BASE_SEQ).await;

    // Produce the residual on the NEWEST base: remove the meta companion the
    // adapter itself published, leaving its `.ompf` artifact durable. Inside
    // this test's own tempdir — fixture behaviour, not a repo-tree deletion.
    let meta_path = base_meta_path(&dir, OBJECT_ID, HEAD_BASE_SEQ);
    assert!(
        meta_path.is_file(),
        "the seeded base must have published its meta companion at {}",
        meta_path.display()
    );
    tokio::fs::remove_file(&meta_path)
        .await
        .expect("removing the base meta companion must succeed");

    // The artifact SURVIVES — that is the whole condition. A removal that took
    // the `.ompf` with it would drop the generation out of the listing and test
    // an entirely different (and uninteresting) path.
    let artifact_path = base_artifact_path(&dir, OBJECT_ID, HEAD_BASE_SEQ);
    assert!(
        artifact_path.is_file(),
        "the newest base's artifact must remain durable at {}",
        artifact_path.display()
    );

    let listing = store
        .list_layer_checkpoints(OBJECT_ID)
        .await
        .expect("listing must succeed with a torn base meta");
    assert_eq!(
        listing_shape(&listing),
        vec![
            (HEAD_BASE_SEQ, CheckpointKind::Base),
            (OLD_BASE_SEQ, CheckpointKind::Base),
        ],
        "the listing is stem-driven, so a missing meta must not hide the newest base — it is \
         still the first candidate, which is what makes the step-down observable"
    );

    // PRECONDITION — the adapter really refuses the BASE stem, with the TYPED
    // marker. Asserted before the gather outcome so a base-arm regression fails
    // HERE rather than turning the assertions below into a vacuous pass.
    let head_handle = handle_at(&listing, HEAD_BASE_SEQ, CheckpointKind::Base);
    let refusal = store
        .get_layer_checkpoint(&head_handle)
        .await
        .expect_err("a durable base artifact with no meta companion must be refused");
    assert!(
        refusal.is_missing_meta_companion(),
        "the base arm must carry the same typed artifact-present / meta-absent marker as the \
         tail arm — the walk steps down on that predicate and on nothing else, so an untyped \
         Permanent here would hard-fail the restore instead. Got {refusal:?}"
    );

    let mut selection =
        gather_restore_lineage(store.as_ref(), OBJECT_ID, no_deadline(), FETCH_RESERVE)
            .await
            .expect("the single listing must succeed");
    let lineage = selection
        .next_restorable()
        .await
        .expect("an unrestorable base is a step-down verdict, not an error")
        .expect("the older base must still be served");

    // THE CLAIM: DR restores from the OLDER base. Pre-b.hka (d) this was
    // `HEAD_BASE_SEQ` with an identity-empty meta, refused at apply as NOT_FOUND.
    assert_eq!(
        lineage.base.handle.seq(),
        OLD_BASE_SEQ,
        "the newest base was abandoned, so the answer is the next base down — the newest \
         generation that can actually be restored"
    );
    assert_eq!(lineage.base.handle.kind, CheckpointKind::Base);
    assert_eq!(lineage.head_seq, OLD_BASE_SEQ);
    assert_eq!(lineage.chain_len(), 0);
    assert!(lineage.is_base_only());
    assert_eq!(lineage.outcome_label(), "gathered_base");

    // And the served lineage is genuinely restorable, not merely selected:
    // identity for the SR-2.4 guard, and a reset-decodable root.
    assert_eq!(lineage.base.meta.session_id, SESSION_ID);
    assert_eq!(lineage.base.meta.layer_name, LAYER_NAME);
    assert_eq!(lineage.base.meta.kind, CheckpointKind::Base);
    assert_ne!(
        lineage.base.bytes.get(..4),
        Some(&TAIL_ENVELOPE_MAGIC[..]),
        "a lineage root is a full-state OMPF snapshot, never a tail envelope"
    );
    OmpfContainer::parse(&lineage.base.bytes).expect(
        "the served base must parse as an OMPF container — that is what 'restorable' means here",
    );

    // And the step-down is OBSERVABLE, under the dedicated outcome and with
    // `head_kind = Base` — the field no other case in this file exercises.
    let step_downs = selection.step_downs();
    assert_eq!(
        step_downs.len(),
        1,
        "exactly one candidate was abandoned: {step_downs:?}"
    );
    assert_eq!(step_downs[0].head_seq, HEAD_BASE_SEQ);
    assert_eq!(
        step_downs[0].head_kind,
        CheckpointKind::Base,
        "the abandoned candidate is base-headed; a `Tail` here would mean the listing mis-parsed \
         the `.ompf` stem"
    );
    match &step_downs[0].reason {
        LineageBreak::MissingMetaCompanion { seq, message } => {
            assert_eq!(*seq, HEAD_BASE_SEQ);
            assert!(
                message.contains(&format!("{OBJECT_ID}.{HEAD_BASE_SEQ}{META_SUFFIX}")),
                "the record must carry the adapter's own message naming the absent BASE \
                 companion — the one thing an operator needs to go look at, and the half that \
                 would be wrong if the base arm reported a tail's companion path; got {message}"
            );
        }
        other => panic!(
            "a torn base put must be its OWN break, not folded into `Decode` — the two carry \
             different operator remediations (a write-path interruption vs. bit-rot or version \
             skew). Got {other:?}"
        ),
    }
    assert_eq!(
        step_downs[0].reason.reason_label(),
        "stepped_down_missing_meta",
        "and the outcome label must be the dedicated one, so a dashboard does not report a \
         routine torn put as storage corruption"
    );
    assert!(selection.exhaustion().is_none());

    drop(dir);
}
