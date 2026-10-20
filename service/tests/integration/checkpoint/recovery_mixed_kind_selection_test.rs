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

//! End-to-end restore-source selection against a **mixed-kind LocalDisk
//! store** — Task `t2.dfh.gy.4o` Subtask `t3.dfh.gy.4o.h6`.
//!
//! The contract this file pins is SDD SR-3.1 / FR-6, stated end-to-end
//! rather than at the hook boundary:
//!
//!   "With one or more tail generations newer than the newest base,
//!    on-access recovery restores the newest *restorable lineage* — the
//!    base plus the contiguous run of tails that link onto it, replayed
//!    to the head sequence — rather than failing on a tail; and a store
//!    holding *only* tails is observably indistinguishable from a store
//!    holding nothing at all."
//!
//! ## The selection rule this file used to pin, and why it changed
//!
//! Test 1 originally asserted SR-2.5's **newest-base-wins** rule: with tails
//! newer than the newest base, selection *skipped* them and restored the base
//! alone, landing `obj:hdr.seq` at `base.snap_seq + margin`. That rule was a
//! consequence of a restore path that could not compose — `decode_checkpoint_
//! ompf` hard-rejects any delta that is not a reset, so a tail's `LETL` bytes
//! had nowhere to go but an error.
//!
//! Epic 4 (`t1.dfh.yj`) swapped that single selection point for
//! **newest-restorable-lineage-wins** (SR-3.1). The identical fixture — a base
//! with two contiguous tails linked onto it — is now the composed-restore
//! **happy path** rather than a skip: the lineage verifies, the tails replay
//! onto the base, and the restore lands at the head tail's `to_seq`. So test 1
//! now asserts the composed outcome. Everything about it that was not a
//! statement of the superseded rule survives unchanged, because those halves
//! are exactly the ones the two rules agree on: the RPC returns `Ok`, identity
//! comes from the base's meta, `error` and `not_recoverable` stay flat, and the
//! tails are read past rather than consumed or deleted.
//!
//! SR-2.5's floor is *not* gone and is not untested — a lineage whose newest
//! verified member is a base alone still restores byte-for-byte as it did
//! before. That floor is asserted in `e2e_composed_restore_test.rs`'s base-only
//! regression fence, which is the file that owns the composed-restore contract
//! in full.
//!
//! ## Why this file exists alongside the hook-level unit tests
//!
//! The hook-level tests for `RecoveryHook::select_restore_source` run
//! against `InMemoryCheckpointStore`, whose listing is minted directly from
//! the `CheckpointKind` a put was handed — it cannot mis-classify a
//! generation because it never parses a filename. `LocalDisk` does: its
//! listing is stem-driven (`<oid>.<seq>.ompf` vs `<oid>.<to_seq>.tail`), its
//! meta round-trips through JSON, and its `get_layer_checkpoint` addresses
//! off `(object_id, snap_seq, kind)`. An InMemory-only proof would therefore
//! not catch a LocalDisk listing that parses `{oid}.{to_seq}.tail` into a
//! **base-kind** handle — which is exactly the stranded-recovery shape FR-6
//! and SR-2.5 forbid: the reset-only decoder is handed a `LETL` tail
//! envelope, `decode_checkpoint_ompf` rejects it `Permanent`, and a layer
//! with a perfectly good base one generation behind it fails the RPC
//! instead of restoring.
//!
//! So the surface under test here is the join of three things the unit
//! tests each see only half of: LocalDisk stem parsing, mixed-kind listing
//! order (`generation_sort_key`: newest sequence first, `Base` before `Tail`
//! at an equal sequence), and meta round-trip — driven through a real gRPC
//! handler (`handle_stat_layer`) so the handler's recovery branch, the
//! admission latch, and the recovery-attempt metric are all in the path.
//!
//! ## Pre-fix behaviour — both tests FAIL against the kind-blind hook
//!
//! Before restore-source selection became kind-aware, the hook selected via
//! `get_latest_layer_checkpoint`, i.e. `list().first()`, which on both fixtures
//! below is a **tail**. Its bytes (`LETL` magic, not `OMPF`) reach
//! `decode_checkpoint_ompf`, fail `OmpfContainer::parse`, and surface as
//! `CheckpointError::Permanent` → `RecoveryOutcome::Error(Permanent)`. That
//! produces, in both tests, an error gRPC status — `INTERNAL` today, since
//! `handle_stat_layer` classifies `RecoveryOutcome::Error` by kind and
//! `Permanent` is non-retriable (b.33e); it was `UNAVAILABLE` when this
//! contrast was first recorded, under the variant-blind mapping that sent every
//! kind to `ServiceError::Unavailable` — and an `error` recovery-attempt tick.
//! No ABSOLUTE status code is pinned anywhere in this file: test 1 asserts only
//! that the RPC returns `Ok`, and test 2 cross-checks its code against the
//! control layer's never-checkpointed status rather than hardcoding one, which
//! is why a remapping like b.33e moves the prose here but not an assertion. So
//! test 1 fails on its `Ok` expectation and its `restored` delta, and test 2
//! fails on both its status cross-check and its `error` delta of 0. Verified at
//! the time by temporarily reverting `select_restore_source` to
//! `get_latest_layer_checkpoint`: both tests went red, test 1 with
//! `OMPF parse failed during checkpoint apply: bad magic: expected
//! 0x46504d4f, got 0x4c54454c` — `LETL`, the tail magic, reaching the reset
//! decoder.
//!
//! That contrast still holds under SR-3.1, by a different route: test 1 now
//! *replays* the tails rather than skipping them, so a kind-blind pick is still
//! the one thing that cannot happen. Test 2 still reports `not_recoverable`
//! with no decode ever attempted — its tails name a base that does not exist,
//! so no lineage roots and there is nothing to compose.
//!
//! ## Real tail bytes are load-bearing — more so now, not less
//!
//! The tails are built with the real tail-envelope codec
//! (`encode_tail_envelope`, `src/checkpoint/tail_codec.rs`), not arbitrary
//! filler. Under the newest-base rule that mattered because the failure had to
//! be the genuine rejection a genuine tail produces. Under SR-3.1 it is
//! load-bearing twice over: test 1's tails are decoded, link-verified and
//! **replayed** by the composition engine, which no filler could survive.
//!
//! What those tails carry is still SYNTHETIC, and test 1 is scoped
//! accordingly: [`put_tail_generation`]'s record payloads are reset OMPFs, so
//! replaying them WIPES state on every record. This file therefore asserts
//! *which lineage* was selected and *where the sequence landed* — never the
//! restored content. Content equivalence against the pre-wipe layer needs a
//! real-commit lineage and lives in `e2e_composed_restore_test.rs`.
//!
//! ## Retention-prune interaction — read this before debugging a failure
//!
//! The harness runs the loop's retention prune at `retain_generations = 2`.
//! The prune counts **lineage units** (`prune_old_generations` via
//! `plan_retention`: each Base closes the tails above it into one unit), and
//! this file's fixture is exactly two units — the lone `base@loop_seq`, and
//! `base@loop_seq + BASE_GAP` with both hand-put tails grouped above it. Two
//! units at `retain = 2` prunes nothing, so a tick landing after the hand-puts
//! deletes neither base and neither tail. That is the whole reason retention is
//! kind-aware at all: the position-based `split_off(retain)` it replaced would
//! have kept the two newest generations — here, the two **tails** — and deleted
//! BOTH bases, collapsing test 1's store into test 2's fixture. That failure
//! presents as "restored=0, not_recoverable=1" and looks exactly like a
//! selection bug.
//!
//! The fixture's tails are genuinely contiguous onto the newer base
//! (`(newer_base, newer_base + 1]` and `(newer_base + 1, newer_base + 2]`,
//! both with `base_snap_seq == newer_base`) — re-verified when lineage-aware
//! retention landed. That matters beyond restorability: under adjacency
//! grouping the tails group with the base directly BELOW them in the listing
//! regardless of their links, so a fixture whose tails sat below the older base
//! would form a THIRD unit and the fence below would acquire a new way to fire
//! at `retain = 2`.
//!
//! The fences below therefore guard a regression rather than a live hazard,
//! and they stay: two make a post-hand-put tick unreachable in the first
//! place, and a third makes a lost base legible if one ever happens anyway.
//!
//! 1. The loop is **cancelled and drained** (bounded `await` on its
//!    `JoinHandle`) before any hand-put runs, so no tick can be scheduled
//!    into the window at all.
//! 2. [`TICK_SECS`] is sized so that even if the drain timed out, the next
//!    tick is seconds away while the hand-put window is milliseconds — see
//!    that constant's doc for the sizing.
//! 3. A guard assertion immediately before the wipe re-lists the store and
//!    fails with a prune-specific diagnostic if the hand-put base is gone.
//!
//! Tail flag: both tests build their harness through
//! [`setup_checkpoint_harness_tails_off`] — `LIVE_EDIT_CHECKPOINT_TAIL_ENABLED`
//! pinned explicitly OFF, not inherited from the binary default (`true` since
//! b.3n1). The subject is **lineage SELECTION** — which generations the walk
//! picks and which survive it — asserted against an exact, hand-planted
//! listing. With the flag on, `CreateLayer` plants an SR-7.1 bootstrap base and
//! a successful restore spawns an SR-3.6 re-base, both of which add generations
//! to that listing and move the `seeded_seq` the head-tail assertion is derived
//! from. Test 2's whole claim — a store holding *only* tails — is also not
//! constructible while a bootstrap keeps planting bases underneath it.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; see
//! CLAUDE.md.

use std::path::Path;
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface};
use tonic::Request;

use live_edit::checkpoint::{CheckpointHandle, CheckpointKind};
use live_edit::grpc::layer::{handle_stat_layer, handle_update_layer};
use live_edit::grpc::proto::StatLayerRequest;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::{SharedMetrics, CHECKPOINT_RESTORE_LINEAGE_OUTCOMES};
use oodf_ffi::{OwnedDelta, SetField};

use super::helpers::*;
use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::grpc::helpers::create_session_and_layer;

/// Checkpoint-loop tick cadence for both tests.
///
/// Sized against two opposing constraints, per the Task's prune fence:
///
/// - **Large enough** that a SECOND tick cannot land inside the hand-put /
///   pre-wipe window. That window is a handful of local filesystem writes —
///   single-digit milliseconds — so any multi-second tick clears it by three
///   orders of magnitude. This is a backstop only: the primary fence is the
///   explicit cancel-and-drain of the loop before the first hand-put (see
///   [`drive_checkpointed_layer`]), after which no tick exists to land.
/// - **Small enough** that the FIRST tick still fires promptly. The loop's
///   `tokio::time::interval` fires immediately at spawn, so the wait for the
///   first generation is bimodal: if that immediate pass enumerates after
///   the CreateLayer + UpdateLayer have landed it checkpoints straight away
///   (observed ~0.13 s end-to-end), and otherwise the wait runs to the
///   second tick at `TICK_SECS` (observed ~5.15 s). This value is therefore
///   very nearly the whole worst-case latency of the test; an over-large one
///   pushes it toward nextest's 40 s slow mark and 120 s hard-kill for no
///   benefit.
///
/// 5 s sits comfortably between: a ~5.2 s worst case is an eighth of the
/// slow mark, while the backstop margin over the millisecond-scale hand-put
/// window stays three orders of magnitude wide.
const TICK_SECS: u64 = 5;

/// `client_id` stamped on the UpdateLayer header. Only needs to be stable
/// and non-empty — self-echo suppression is not under test here.
const CLIENT_ID: &str = "mixed-kind-selection-client";

/// How far above the loop's own generation the hand-put base is placed.
///
/// Any positive gap works — the point is only that the hand-put base is
/// strictly newer than the loop's, so "the newest base" and "the loop's
/// base" are distinguishable in the restored `seq`. A gap of 10 keeps the
/// two an obvious distance apart in a failure message.
const BASE_GAP: u64 = 10;

// ─────────────────────────────────────────────────────────────────────────────
// Metric / request shape helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Probe one label cell of
/// `live_edit_checkpoint_recovery_attempts_total{result, scope, service}`.
/// Same shape `e2e_stat_layer_recovery_test.rs` uses, so a reader comparing
/// the two files is comparing like with like.
fn recovery_metric(metrics: &SharedMetrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_recovery_attempts_total",
        &[
            ("result", result),
            ("scope", "layer"),
            ("service", SERVICE_LABEL),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// A `StatLayerRequest` carrying the stub bearer token the harness's
/// `AllowAllPermissions` / `StubUserinfo` wiring expects.
fn stat_request(session_id: &str, layer_name: &str) -> Request<StatLayerRequest> {
    let mut request = Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ─────────────────────────────────────────────────────────────────────────────
// Payload builders
// ─────────────────────────────────────────────────────────────────────────────

/// A one-`SetField` diff on the layer root, used to drive `obj:hdr.seq`
/// forward through a real `UpdateLayer` before the loop checkpoints.
///
/// `section_id = 0` addresses the root scope that exists by construction
/// after `CreateLayer`, so no parent section has to be materialized first.
fn field_update_delta(set_order: u64, marker: u8) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        set_fields: vec![SetField {
            section_id: 0,
            key_name: format!("mixed-kind-field-{set_order}"),
            value_or_ext_hash: vec![marker, marker.wrapping_add(1), marker.wrapping_add(2)],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        }],
        ..Default::default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Valkey read helpers
// ─────────────────────────────────────────────────────────────────────────────

/// `HGETALL le:{object_id}:obj:hdr`.
async fn read_obj_header(
    harness: &CheckpointHarness,
    object_id: u64,
) -> std::collections::HashMap<String, String> {
    let header_key = harness.tv.key_schema().obj_header(&object_id.to_string());
    harness
        .tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL obj:hdr must succeed")
}

/// `EXISTS le:{object_id}:obj:hdr`.
async fn obj_header_exists(harness: &CheckpointHarness, object_id: u64) -> bool {
    let header_key = harness.tv.key_schema().obj_header(&object_id.to_string());
    harness
        .tv
        .client()
        .pool
        .exists(&header_key)
        .await
        .expect("EXISTS obj:hdr must succeed")
}

/// Parse a `u64` field off an `obj:hdr` map with a diagnostic on absence.
fn parse_header_u64(header: &std::collections::HashMap<String, String>, field: &str) -> u64 {
    let raw = header.get(field).unwrap_or_else(|| {
        panic!(
            "obj:hdr post-recovery MUST carry `{field}`; got fields: {:?}",
            header.keys().collect::<Vec<_>>()
        )
    });
    raw.parse::<u64>()
        .unwrap_or_else(|e| panic!("obj:hdr `{field}` MUST parse as u64; got {raw:?}: {e}"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared preamble
// ─────────────────────────────────────────────────────────────────────────────

/// CreateSession → CreateLayer → one UpdateLayer → wait for the loop's base
/// → **cancel and drain the loop**.
///
/// Returns `(session_id, object_id, loop_snap_seq)` where `loop_snap_seq` is
/// the `snap_seq` of the generation the periodic loop landed.
///
/// The cancel-and-drain is the primary retention-prune fence described in
/// this module's header: once the loop's task has exited, no further tick
/// can fire, so the position-based `split_off(retain)` prune cannot run
/// between the hand-puts and the wipe. It is bounded by a 5 s timeout so a
/// wedged loop fails the test promptly rather than hanging the binary; the
/// [`TICK_SECS`] backstop and the callers' pre-wipe guard assertion cover
/// the case where that bound is hit.
///
/// Takes `&mut CheckpointHarness` so the `JoinHandle` can be `mem::take`n
/// and awaited while `tv` / `svc` / `store` stay usable for the rest of the
/// test; a no-op handle is swapped in so the harness's drop stays safe.
async fn drive_checkpointed_layer(
    harness: &mut CheckpointHarness,
    session_name: &str,
    layer_name: &str,
) -> (String, u64, u64) {
    let (session_id, object_id) =
        create_session_and_layer(&harness.tv, &harness.svc, session_name, layer_name).await;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );

    // One real commit through the gRPC handler so the loop's generation has
    // a `snap_seq` above the CreateLayer's own, and so the fixture exercises
    // the ordinary write path rather than a bare create.
    let ompf = delta_to_ompf_bytes(&field_update_delta(1, 0x11), &[]);
    let stream = streaming_update(vec![
        make_update_header(&session_id, layer_name, object_id, 1_001, CLIENT_ID),
        make_update_data(ompf),
    ]);
    let committed_seq = handle_update_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("UpdateLayer must succeed during the pre-wipe seeding phase")
        .into_inner()
        .sequence_num;

    // Gate on the committed sequence, not merely "any generation": with
    // `interval=0`/`quiesce=0` the loop can fire between CreateLayer and the
    // update and land an earlier generation, which would leave `loop_snap_seq`
    // non-deterministic and the `BASE_GAP` arithmetic below meaningless.
    let loop_handle_gen =
        wait_for_checkpoint_at_least(&harness.store, object_id, committed_seq, 10).await;

    // Primary prune fence — see this function's doc and the module header.
    harness.cancel.cancel();
    let loop_task = std::mem::replace(&mut harness.loop_handle, tokio::spawn(async {}));
    let _ = tokio::time::timeout(Duration::from_secs(5), loop_task).await;

    (session_id, object_id, loop_handle_gen.snap_seq)
}

/// Re-list the store and assert the hand-put base at `expected_base_seq` is
/// still present — the pre-wipe guard the Task requires.
///
/// Named separately from the assertions it protects so a prune-induced
/// failure reports as a prune, not as a selection bug. The distinction
/// matters because the two present identically downstream: a pruned base
/// leaves a tails-only store, and a tails-only store correctly reports
/// `not_recoverable` — which is precisely what a broken selection would
/// also produce.
async fn assert_base_survived_prune(
    harness: &CheckpointHarness,
    object_id: u64,
    expected_base_seq: u64,
) -> Vec<CheckpointHandle> {
    let generations = harness
        .store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        generations
            .iter()
            .any(|h| h.kind == CheckpointKind::Base && h.snap_seq == expected_base_seq),
        "PRUNE FENCE (not a selection failure): the hand-put base@{expected_base_seq} is gone \
         from the store before the wipe. The loop's retention prune counts LINEAGE UNITS at \
         `retain_generations = 2`, and this fixture is exactly two units (the lone loop base, \
         and this base with both tails grouped above it), so nothing should be pruned by any \
         tick — and the loop is cancelled and drained before the hand-puts so no tick can even \
         be scheduled. Reaching this assertion means either the drain did not take AND a third \
         unit appeared, or the prune stopped grouping by kind and reverted to the \
         position-based `split_off(retain)` that deletes every base under two newer tails (the \
         FR-6 stranded-generation shape). Live listing: {generations:?}"
    );
    generations
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 1 — the composed lineage is restored, and the tails survive it.
// ═════════════════════════════════════════════════════════════════════════════

/// SR-3.1 end-to-end: with two tails contiguous onto the newest base,
/// on-access recovery through `handle_stat_layer` restores **the composed
/// lineage** — base + both tails, replayed to the head tail's `to_seq`.
///
/// Fixture (newest-first, exactly as `list_layer_checkpoints` returns it):
///
/// ```text
///   tail @ loop_seq + BASE_GAP + 2     ← the lineage head; the restore lands here
///   tail @ loop_seq + BASE_GAP + 1
///   base @ loop_seq + BASE_GAP         ← the lineage root
///   base @ loop_seq                    ← the periodic loop's own generation
/// ```
///
/// This fixture was authored for the superseded SR-2.5 newest-base rule, under
/// which the correct answer was "restore base@`+BASE_GAP`, skip both tails".
/// The tails link contiguously onto that base, which is exactly what SR-3.1
/// calls a restorable lineage, so the same bytes now describe the composed
/// happy path. See the module header for the full before/after.
///
/// Four independent observations discriminate the correct outcome:
///
/// - The RPC returns `Ok` at all. A kind-blind pick hands `LETL` bytes to the
///   reset-only decoder → `Permanent` → `INTERNAL` (b.33e; `UNAVAILABLE` under
///   the superseded variant-blind mapping).
/// - The restored `obj:hdr.seq` equals `(loop_seq + BASE_GAP + 2) +
///   checkpoint_seq_safety_margin` — the **head tail's** sequence, not the
///   base's. `wipe_layer_state` takes `obj:stream` with the rest of `obj:*`, so
///   the high-water probe folds to `None` and the restore floor is the lineage
///   head alone; the guard band is `max(margin, head span)` and the head tail
///   spans one commit, so the margin dominates. Every wrong answer produces a
///   different number: the base-only answer is `+BASE_GAP + margin`, the loop's
///   older base is `loop_seq + margin`.
/// - `last_checkpoint_seq` — the audit-trail column — equals the composed head
///   too, which is the sharpest single probe that the composed path stamped
///   `GatheredLineage::head_seq` rather than the base's `meta.snap_seq`.
/// - `restored` ticks exactly once with `error` and `not_recoverable` flat, and
///   on the lineage family `gathered_composed` ticks once with every
///   `stepped_down_*` cell flat — a step-down here would mean an intact lineage
///   was judged broken.
///
/// The last assertion is about what a restore must NOT do: both tail artifacts
/// are still on disk, still listed, and still byte-identical afterwards.
/// Restore is a pure read — composition consumes the artifact *bytes* in
/// memory, never the artifacts.
///
/// The restored *content* is deliberately not asserted: these tails carry
/// synthetic reset records (see the module header), so their replay wipes
/// state. Content equivalence lives in `e2e_composed_restore_test.rs`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_recovers_the_composed_lineage_when_newer_tails_are_contiguous() {
    let mut harness = setup_checkpoint_harness_tails_off(TICK_SECS).await;
    let layer_name = "mixed-kind-newest-base-layer";

    let (session_id, object_id, loop_snap_seq) =
        drive_checkpointed_layer(&mut harness, "mixed-kind-newest-base-session", layer_name).await;

    // ── Hand-put the mixed-kind fixture ──────────────────────────────────
    let newer_base_seq = loop_snap_seq + BASE_GAP;
    let base_handle = put_base_generation(
        &harness.store,
        object_id,
        &session_id,
        layer_name,
        newer_base_seq,
    )
    .await;
    assert_eq!(
        base_handle.kind,
        CheckpointKind::Base,
        "put_layer_checkpoint must mint a Base-kind handle for a Base-kind meta"
    );

    let tail_one = put_tail_generation(
        &harness.store,
        object_id,
        &session_id,
        layer_name,
        newer_base_seq,
        newer_base_seq,
        newer_base_seq + 1,
    )
    .await;
    let tail_two = put_tail_generation(
        &harness.store,
        object_id,
        &session_id,
        layer_name,
        newer_base_seq,
        newer_base_seq + 1,
        newer_base_seq + 2,
    )
    .await;
    for tail in [&tail_one, &tail_two] {
        assert_eq!(
            tail.kind,
            CheckpointKind::Tail,
            "put_layer_checkpoint must mint a Tail-kind handle for a Tail-kind meta; \
             a Base here means the adapter lost the kind on the way to the handle"
        );
    }

    // Capture the tail bytes as the store reports them, so the
    // "not consumed" assertion at the end compares like with like.
    let tail_bytes_before: Vec<Bytes> = {
        let mut out = Vec::new();
        for tail in [&tail_one, &tail_two] {
            let blob = harness
                .store
                .get_layer_checkpoint(tail)
                .await
                .expect("get_layer_checkpoint on a freshly-put tail must succeed")
                .expect("a freshly-put tail must be fetchable");
            out.push(blob.bytes);
        }
        out
    };

    // ── Pre-wipe guards ──────────────────────────────────────────────────
    let generations = assert_base_survived_prune(&harness, object_id, newer_base_seq).await;

    // The fixture is only discriminating if the NEWEST generation is a tail:
    // that is what makes "the lineage head" and "the lineage root" different
    // sequences, and it is the exact input a kind-blind `list().first()` pick
    // would take.
    assert_eq!(
        generations.first().map(|h| (h.kind, h.snap_seq)),
        Some((CheckpointKind::Tail, newer_base_seq + 2)),
        "fixture precondition: the newest generation MUST be the newer of the two tails, \
         otherwise the composed head and the lineage root coincide and the test cannot \
         distinguish a composed restore from a base-only one. Live listing: {generations:?}"
    );
    assert_eq!(
        generations
            .iter()
            .find(|h| h.kind == CheckpointKind::Base)
            .map(|h| h.snap_seq),
        Some(newer_base_seq),
        "fixture precondition: the first Base in newest-first listing order MUST be the \
         hand-put base@{newer_base_seq}, not the loop's base@{loop_snap_seq}. \
         Live listing: {generations:?}"
    );

    // ── Provoke on-access recovery ───────────────────────────────────────
    wipe_layer_state(&harness.tv, object_id).await;

    let restored_pre = recovery_metric(&harness.metrics, "restored");
    let not_recoverable_pre = recovery_metric(&harness.metrics, "not_recoverable");
    let error_pre = recovery_metric(&harness.metrics, "error");
    let lineage_pre: Vec<(&str, u64)> = CHECKPOINT_RESTORE_LINEAGE_OUTCOMES
        .iter()
        .map(|outcome| {
            (
                *outcome,
                read_restore_lineage_counter(&harness.metrics, outcome),
            )
        })
        .collect();

    let response = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .unwrap_or_else(|status| {
            panic!(
                "StatLayer after the wipe MUST recover the composed lineage and return Ok, \
                 got {:?}: {}. An INTERNAL here is the kind-blind shape verbatim: the newest \
                 generation is a tail, its `LETL` bytes reached `decode_checkpoint_ompf`, and \
                 `OmpfContainer::parse` rejected them `Permanent`. A NOT_FOUND instead means \
                 no lineage verified at all in a store that demonstrably holds two restorable \
                 roots",
                status.code(),
                status.message()
            )
        });

    let stat = response.into_inner();
    assert_eq!(
        stat.created_by_user_name, HAND_PUT_CREATED_BY,
        "the restored header's created_by MUST come from the HAND-PUT base's meta — identity \
         columns are BASE-sourced even on a composed restore. The rule is structural (a tail's \
         identity is base-sourced by construction) and does NOT rest on the adapters' \
         identity-empty meta-absent placeholder, which b.hka (d) replaced with a refusal. A \
         different value means a different lineage was restored"
    );
    assert!(
        stat.created_at > 0,
        "StatLayer after recovery MUST surface a non-zero created_at, restored from the \
         selected generation's meta `server_ts`"
    );

    // ── The load-bearing assertion: WHICH lineage was restored ───────────
    let header = read_obj_header(&harness, object_id).await;
    let recovered_seq = parse_header_u64(&header, "seq");
    let margin = crate::common::test_config().checkpoint_seq_safety_margin;
    let head_seq = newer_base_seq + 2;
    let expected_seq = head_seq + margin;
    assert_eq!(
        recovered_seq,
        expected_seq,
        "post-recovery obj:hdr.seq ({recovered_seq}) MUST equal the composed lineage HEAD \
         ({head_seq}, the newer tail's to_seq) + safety_margin ({margin}) = {expected_seq}. \
         {} would mean the lineage root base@{newer_base_seq} was restored alone — the \
         tails skipped rather than replayed; {} would mean the loop's OLDER base@\
         {loop_snap_seq} was selected instead.",
        newer_base_seq + margin,
        loop_snap_seq + margin
    );
    assert_eq!(
        parse_header_u64(&header, "last_checkpoint_seq"),
        head_seq,
        "the audit-trail column MUST record the composed head ({head_seq}), not the lineage \
         root's snap_seq ({newer_base_seq}). This is the one column that distinguishes a \
         restore that passed `GatheredLineage::head_seq` to apply from one that passed the \
         base blob's own `meta.snap_seq` — the latter also under-seeds the writer cursor by \
         the whole replayed span"
    );

    assert_eq!(
        recovery_metric(&harness.metrics, "restored") - restored_pre,
        1,
        "exactly ONE `restored` tick: not zero (recovery never fired / never found a base), \
         not two (the handler retried past the admission latch)"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "error") - error_pre,
        0,
        "`error` MUST stay flat. A tick here is the pre-fix signature: the tail's bytes \
         reached the reset decoder and came back `Permanent`"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "not_recoverable") - not_recoverable_pre,
        0,
        "`not_recoverable` MUST stay flat — a restorable lineage is on disk"
    );

    // The lineage family is orthogonal to `recovery_attempts_total` and is NOT
    // one-tick-per-attempt: `gathered_composed` records the verdict, and any
    // `stepped_down_*` tick would mean the walk judged this intact lineage
    // broken and served older state.
    for (outcome, before) in lineage_pre {
        let delta = read_restore_lineage_counter(&harness.metrics, outcome) - before;
        let expected = u64::from(outcome == "gathered_composed");
        assert_eq!(
            delta, expected,
            "lineage outcome {outcome:?} MUST tick {expected} time(s) for one composed \
             restore of an intact base + two contiguous tails. A `gathered_base` tick means \
             the tails were skipped; any `stepped_down_*` tick names the link the walk \
             believed was broken; `exhausted` means no lineage verified at all"
        );
    }

    // ── A restore READS artifacts; it must not consume them ──────────────
    for (tail, bytes_before) in [&tail_one, &tail_two].into_iter().zip(tail_bytes_before) {
        assert!(
            Path::new(&tail.adapter_key).is_file(),
            "tail@{} MUST still exist on disk at {} after being replayed — composition \
             consumes the artifact BYTES in memory, never the artifact, and a later restore \
             (or a peer replica's) needs exactly these bytes again",
            tail.snap_seq,
            tail.adapter_key
        );
        let blob = harness
            .store
            .get_layer_checkpoint(tail)
            .await
            .expect("get_layer_checkpoint on a replayed tail must succeed")
            .unwrap_or_else(|| {
                panic!(
                    "tail@{} MUST still be fetchable after the restore that replayed it",
                    tail.snap_seq
                )
            });
        assert_eq!(
            blob.bytes, bytes_before,
            "tail@{}'s envelope bytes MUST be byte-identical after the restore — a \
             difference means the restore path rewrote or truncated an artifact it was only \
             supposed to read",
            tail.snap_seq
        );
    }

    let post_generations = harness
        .store
        .list_layer_checkpoints(object_id)
        .await
        .expect("post-recovery list_layer_checkpoints must succeed");
    assert_eq!(
        post_generations, generations,
        "the store's listing MUST be unchanged by a restore: recovery is a pure read, so \
         no generation of either kind may be added, removed, or re-kinded"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 2 — a tails-only store behaves exactly like an empty one.
// ═════════════════════════════════════════════════════════════════════════════

/// The inverse of test 1, and unchanged by SR-3.1: when every durable
/// generation is a tail, recovery MUST report the layer unrecoverable
/// **exactly as for a never-checkpointed layer** — same gRPC status,
/// `not_recoverable` cell, and critically an `error` delta of **zero**.
///
/// Composed restore does not weaken this. A lineage is rooted in a base by
/// definition, and these tails name a base that has been deleted, so the walk
/// finds no root, no candidate verifies, and selection exhausts. Nothing is
/// decoded and nothing is replayed.
///
/// The zero-`error` half is the load-bearing one. Under a kind-blind hook the
/// newest generation — a tail — was fetched and its `LETL` bytes handed to
/// `decode_checkpoint_ompf`, which rejects them `Permanent`; the verdict became
/// `RecoveryOutcome::Error(Permanent)`, surfacing an error status (`INTERNAL`
/// under the kind-classifying mapping of b.33e; `UNAVAILABLE` when this was
/// written) and an `error` tick. That is "Permanent-decode noise as the
/// verdict", the shape this Task forbids: a store that simply cannot supply a
/// restorable generation is not a store that malfunctioned. Post-fix no decode
/// is attempted at all, because no base is ever selected.
///
/// The expected status is **cross-checked against this harness's own
/// no-checkpoint path** rather than hardcoded: a second layer is created
/// after the loop is cancelled (so it is never checkpointed at all), wiped,
/// and statted, and its status is the reference the tails-only layer must
/// match. Hardcoding `NOT_FOUND` from memory would let a future change to
/// the empty-store status silently split the two paths apart — which is the
/// one property this test exists to deny.
///
/// **Fails against a kind-blind hook** (an error status — `INTERNAL` under
/// today's mapping — instead of the no-checkpoint status, and `error` = 1
/// instead of 0), passes after. See the module header for the retention-prune
/// interaction; here the bases are deleted deliberately, so the guard runs on
/// the tails instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_tails_only_store_behaves_like_empty_store() {
    let mut harness = setup_checkpoint_harness_tails_off(TICK_SECS).await;
    let layer_name = "mixed-kind-tails-only-layer";

    let (session_id, object_id, loop_snap_seq) =
        drive_checkpointed_layer(&mut harness, "mixed-kind-tails-only-session", layer_name).await;

    // ── Reference layer: never checkpointed ──────────────────────────────
    // Created AFTER the loop was cancelled and drained, so no generation of
    // any kind can exist for it. This is the harness's own "empty store"
    // path, and its status is what the tails-only path must match.
    let control_layer_name = "mixed-kind-no-checkpoint-layer";
    let (control_session_id, control_object_id) = create_session_and_layer(
        &harness.tv,
        &harness.svc,
        "mixed-kind-no-checkpoint-session",
        control_layer_name,
    )
    .await;
    assert!(
        harness
            .store
            .list_layer_checkpoints(control_object_id)
            .await
            .expect("list_layer_checkpoints must succeed for the control layer")
            .is_empty(),
        "control precondition: the reference layer MUST have no generations at all — it was \
         created after the loop was drained. A generation here means the drain did not take, \
         and the 'empty store' reference is not empty"
    );

    // ── Collapse the target's store to tails only ────────────────────────
    let pre_delete = harness
        .store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    let bases: Vec<CheckpointHandle> = pre_delete
        .iter()
        .filter(|h| h.kind == CheckpointKind::Base)
        .cloned()
        .collect();
    assert!(
        !bases.is_empty(),
        "fixture precondition: the loop must have landed at least one base before this test \
         deletes them; got {pre_delete:?}"
    );
    for base in bases {
        harness
            .store
            .delete_layer_checkpoint(base.clone())
            .await
            .unwrap_or_else(|e| panic!("delete of base@{} must succeed: {e}", base.snap_seq));
    }

    // The tails name a base that no longer exists — the "manual pruning of
    // the base / partial checkpoint-store loss" shape SR-2.5 calls out. The
    // shipped write paths cannot produce it, which is exactly why it needs a
    // test: it is the state in which a kind-blind selection is guaranteed to
    // reach for a tail.
    let tail_one = put_tail_generation(
        &harness.store,
        object_id,
        &session_id,
        layer_name,
        loop_snap_seq,
        loop_snap_seq,
        loop_snap_seq + 1,
    )
    .await;
    let tail_two = put_tail_generation(
        &harness.store,
        object_id,
        &session_id,
        layer_name,
        loop_snap_seq,
        loop_snap_seq + 1,
        loop_snap_seq + 2,
    )
    .await;

    // ── Pre-wipe guard ───────────────────────────────────────────────────
    // The prune fence still applies, inverted: here the store MUST hold both
    // tails and NO base. A stray tick would have re-checkpointed the layer
    // and put a base back, which would make the whole fixture vacuous.
    let generations = harness
        .store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert_eq!(
        generations
            .iter()
            .map(|h| (h.kind, h.snap_seq))
            .collect::<Vec<_>>(),
        vec![
            (CheckpointKind::Tail, tail_two.snap_seq),
            (CheckpointKind::Tail, tail_one.snap_seq),
        ],
        "PRUNE / TICK FENCE (not a selection failure): the store must hold exactly the two \
         hand-put tails and no base. A base here means a checkpoint-loop tick landed after \
         the drain and re-checkpointed the layer, which would make this fixture vacuous; a \
         missing tail means the position-based retention prune ran. Live listing: \
         {generations:?}"
    );

    wipe_layer_state(&harness.tv, control_object_id).await;
    wipe_layer_state(&harness.tv, object_id).await;

    // ── Reference probe: what does a never-checkpointed layer report? ────
    let control_status = handle_stat_layer(
        &harness.svc,
        stat_request(&control_session_id, control_layer_name),
    )
    .await
    .err()
    .unwrap_or_else(|| {
        panic!(
            "control precondition: StatLayer on a wiped, never-checkpointed layer MUST fail — \
             an Ok here means the wipe did not take or recovery restored from somewhere"
        )
    });

    // ── The tails-only probe ─────────────────────────────────────────────
    let restored_pre = recovery_metric(&harness.metrics, "restored");
    let not_recoverable_pre = recovery_metric(&harness.metrics, "not_recoverable");
    let error_pre = recovery_metric(&harness.metrics, "error");

    let status = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .err()
        .unwrap_or_else(|| {
            panic!(
                "StatLayer against a tails-only store MUST fail — no restorable LINEAGE \
                 exists. Every lineage is rooted in a base, and these tails name one that was \
                 deleted, so the walk finds no root. An Ok here means a restore was composed \
                 out of tail envelopes alone, with no full-state generation underneath them"
            )
        });

    assert_eq!(
        status.code(),
        control_status.code(),
        "a tails-only store MUST be observably indistinguishable from a never-checkpointed \
         layer: got {:?} for tails-only vs {:?} for the no-checkpoint reference. A divergence \
         on the tails-only side — an INTERNAL under today's kind-classifying mapping (b.33e) — \
         is the pre-fix shape verbatim: the tail's bytes reached `decode_checkpoint_ompf`, were \
         rejected `Permanent`, and that Permanent became the verdict instead of a plain \
         'nothing restorable here'",
        status.code(),
        control_status.code()
    );

    assert_eq!(
        recovery_metric(&harness.metrics, "not_recoverable") - not_recoverable_pre,
        1,
        "exactly ONE `not_recoverable` tick — the attempt must still be accounted for \
         (`sum by (result)` equals the attempt count), just with the correct verdict"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "error") - error_pre,
        0,
        "LOAD-BEARING: `error` MUST stay flat. A tick here means the tail's `LETL` bytes were \
         fetched and handed to `decode_checkpoint_ompf`, rejected `Permanent`, and that \
         Permanent-decode noise became the recovery verdict — the exact failure mode this \
         selection exists to make impossible. A store that cannot supply a restorable \
         generation has not malfunctioned"
    );
    assert_eq!(
        recovery_metric(&harness.metrics, "restored") - restored_pre,
        0,
        "`restored` MUST stay flat — nothing restorable is on disk"
    );

    assert!(
        !obj_header_exists(&harness, object_id).await,
        "a refused restore MUST leave NO obj:hdr behind: selection returns before any decode, \
         so nothing can be half-applied. A header here means partial materialization from a \
         source that was never restorable"
    );

    // The tails themselves are untouched by the refusal, same as in test 1.
    for tail in [&tail_one, &tail_two] {
        assert!(
            Path::new(&tail.adapter_key).is_file(),
            "tail@{} MUST still exist at {} after a refused restore — declining to restore \
             is not a licence to delete the artifacts that were declined",
            tail.snap_seq,
            tail.adapter_key
        );
    }
}
