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

//! Composed restore through the production recovery hook — Task
//! `t2.dfh.yj.b7`, Subtask `t3.dfh.yj.b7.x7`.
//!
//! ## What this file proves that its neighbours structurally cannot
//!
//! Task 2's selection suites drive `gather_restore_lineage` directly: they end
//! at a `GatheredLineage`, with no restore, no Valkey, and — because the gather
//! module holds no metrics handle at all — no increment site for the lineage
//! metric family. Task 6's suites layer proptests, goldens and a fault matrix
//! on top of whatever wiring exists. Between them sits the claim nobody else
//! makes: that the **wiring** exists and fires in the right order —
//! gather → compose → external-value pre-store → ONE `write_object` — and that
//! the four cursor columns it stamps carry the composed head's numbers rather
//! than the base's.
//!
//! Everything here is observed through a production handler
//! (`handle_stat_layer`), so the hook, the per-`object_id` admission latch and
//! the recovery-attempt metric are all in the path, exactly as they are for a
//! real client. One test steps outside that to call
//! `RecoveryHook::try_restore_layer` directly — it is the one whose subject IS
//! the hook's return type.
//!
//! ## The fence: no existing recovery file may be edited
//!
//! `src/checkpoint/recovery.rs`'s `mod tests` and every `e2e_*_recovery_test.rs`
//! in this module are the base-only regression fence for Epic `t1.dfh.yj`: a
//! composed-restore change that needed one of them edited would be a change to
//! base-only behaviour, which is an escalation signal rather than a cleanup. So
//! the composed coverage lands here, new, and the base-only fence is restated
//! from this side by Test 2 rather than by touching those files.
//!
//! (`recovery_mixed_kind_selection_test.rs` is the one deliberate exception, and
//! it is not an exception to *this* rule: its fixture — a base with two
//! contiguous tails — was the newest-base rule's discriminator and is the
//! composed happy path under SR-3.1, so the Epic revised it on purpose. Its
//! module header records the swap.)
//!
//! ## The fixture problem, and the real-commit lineage that resolves it
//!
//! A round-trip assertion needs a lineage whose composition equals the live
//! layer's state. Neither existing tail builder can produce one:
//!
//! - `put_tail_generation`'s record payloads are reset OMPFs (`is_diff = true,
//!   base_version = 0`). Replaying one takes `apply_delta`'s reset branch and
//!   **WIPES** state on every record, so base + such a tail composes to an
//!   empty layer — a fixture that goes green while proving nothing.
//! - `drive_captured_tail` captures a genuine tail but plants its lineage root
//!   with an **empty** base at `CAPTURED_TAIL_SEED_SEQ`, while its live layer
//!   carries 120 sections. Its composition cannot equal the live state either.
//! - `build_reset_ompf_from_state` is `pub(crate)`, so the base cannot be
//!   hand-built carrying state from an integration binary.
//!
//! The shape that works is a **real-commit lineage**: let the harness's own
//! periodic loop write the base (a real assembler-produced snapshot carrying
//! real state at a real `snap_seq`), fence the loop, drive further
//! `UpdateLayer`s, and hand-encode a tail from the exact OMPF bytes those
//! commits sent, at their returned `sequence_num`s. That is byte-for-byte what
//! the capture path persists (`le:{oid}:obj:stream`'s `ompf` field is the
//! writer's raw inbound OMPF), so the artifact is real — and unlike
//! `drive_captured_tail`'s hot-writer drive it is fully deterministic.
//! [`put_tail_generation_from_records`] is the shared builder for it.
//!
//! ## Synchronization budget
//!
//! No `tokio::time::sleep` is used for coordination anywhere in this file. The
//! two waits are:
//!
//! | Wait | Bound | Why that number |
//! |---|---|---|
//! | first base lands | [`LOOP_BASE_DEADLINE_SECS`] (bounded poll on the store listing) | the loop ticks at [`TICK_SECS`]; the helper additionally clamps to its own load-tolerant floor |
//! | loop task drains after `cancel()` | [`LOOP_DRAIN_BUDGET`] | a cancelled loop returns at its next `select!`; a budget only stops a wedged one from hanging the binary |
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; see CLAUDE.md.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::HashesInterface;
use tonic::Request;

use live_edit::checkpoint::{CheckpointHandle, CheckpointMeta, RecoveryOutcome};
use live_edit::grpc::layer::{handle_create_layer, handle_stat_layer, handle_update_layer};
use live_edit::grpc::proto::StatLayerRequest;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::CHECKPOINT_RESTORE_LINEAGE_OUTCOMES;
use live_edit::storage::diff_read::ObjectSnapshot;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use oodf_ffi::{
    CommandDiffChangeSection, OwnedDelta, SetField, StructCommand, TimeSample as OodfTimeSample,
};

use super::helpers::*;
use crate::common::owned_delta::delta_to_ompf_bytes;

// ─────────────────────────────────────────────────────────────────────────────
// Named waits and fixture constants — see the module header's budget table
// ─────────────────────────────────────────────────────────────────────────────

/// Checkpoint-loop tick cadence. `1` is the value every e2e recovery file in
/// this module uses: the harness sets `interval = quiesce = 0`, so a
/// just-written layer is due on the very next tick and the first base lands in
/// well under a second.
const TICK_SECS: u64 = 1;

/// Deadline handed to `wait_for_checkpoint_at_least`. The helper clamps it up
/// to its own load-tolerant floor, so this is the *lower* bound on how long a
/// contended Valkey may take before the wait is called a failure.
const LOOP_BASE_DEADLINE_SECS: u64 = 10;

/// Bound on the post-`cancel()` drain of the checkpoint loop's `JoinHandle`.
///
/// The loop returns at its next `select!` on the cancellation token, so this is
/// never the wait actually observed — it exists so a wedged loop fails the test
/// promptly instead of hanging the binary. Same shape and same number as
/// `recovery_mixed_kind_selection_test`'s prune fence.
const LOOP_DRAIN_BUDGET: Duration = Duration::from_secs(5);

/// `client_id` on every UpdateLayer header. Only needs to be stable and
/// non-empty; self-echo suppression is not under test.
const CLIENT_ID: &str = "composed-restore-client";

const SESSION_BASE_PATH: &str = "/stage";
const CREATED_BY: &str = "composed-restore-user";

/// The section the CreateLayer delta and the pre-base commits write into.
const ROOT_SECTION_ID: u64 = 42;

/// First section id minted by a tail-range commit. Each such commit creates its
/// own section, so the composed replay has structural work to do and not only
/// field writes.
const TAIL_SECTION_BASE_ID: u64 = 1_000;

/// Authoring rank stamped on every `SetField` in this file.
///
/// Load-bearing for Test 2 and nothing else. A base-only restore is a reset
/// decode, and `build_reset_ompf_from_state` can carry only ONE of the two
/// persisted ranks through OODF's single `set_order` slot — it carries the
/// authoring rank, and the reset replay re-accumulates the cumulative cursor
/// from scratch, in the base blob's own field order. With every field written
/// exactly once at rank `0`, the re-accumulated cursor lands on the same
/// `next_set_order` the live path reached whatever that order happens to be —
/// so the rank buys this file INDEPENDENCE from the emission order, not a
/// dependency on it. (Since b.gcb / b.vz7 that order is itself total over
/// `ObjectState`'s content: `DiffEngine::plan_full_content` sorts. This
/// constant predates the sort and nothing here leans on it either way.) See
/// `assert_cumulative_set_order_columns` for the half that still cannot be
/// compared key-by-key against the LIVE oracle, and why.
const AUTHORING_RANK: u64 = 0;

// ─────────────────────────────────────────────────────────────────────────────
// Payload builders
// ─────────────────────────────────────────────────────────────────────────────

/// CreateLayer payload: one section plus one inline field, so a restored
/// snapshot has structure to compare rather than being vacuously equal.
fn populated_initial_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_string(),
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: ROOT_SECTION_ID,
            section_name: "composed-restore-root".to_string(),
            section_type: 1,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: "initial-field".to_string(),
            value_or_ext_hash: vec![0xAA, 0xBB, 0xCC],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// A pre-base commit: one new inline field on the root section.
///
/// `base_version: 1` (any non-zero) marks this an INCREMENTAL diff. With
/// `base_version == 0` and `is_diff == true` the domain engine takes the
/// reset branch and wipes prior state, which would make every comparison in
/// this file vacuous.
fn pre_base_commit_delta(index: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: format!("pre-base-field-{index}"),
            value_or_ext_hash: vec![0x10 + index as u8, 0x11, 0x12],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// A tail-range commit: a NEW section under the root, one inline field inside
/// it, and one time sample.
///
/// All three record kinds in one delta on purpose. A composed restore that
/// replayed field writes but dropped structural commands (or vice versa) would
/// still satisfy a fields-only oracle; it cannot satisfy this one.
fn tail_commit_delta(index: u64) -> OwnedDelta {
    let section_id = TAIL_SECTION_BASE_ID + index;
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: ROOT_SECTION_ID,
            section_id,
            section_name: format!("tail-section-{index}"),
            section_type: 2,
            section_order: index as u32,
        })],
        set_fields: vec![SetField {
            section_id,
            key_name: format!("tail-field-{index}"),
            value_or_ext_hash: vec![0xC0 + index as u8, 0xDE, index as u8],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        time_samples: vec![OodfTimeSample {
            section_id,
            time: index as f64,
            value_or_ext_hash: vec![0xF0 + index as u8],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Drive + capture helpers
// ─────────────────────────────────────────────────────────────────────────────

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

/// Issue one `UpdateLayer` and return `(sequence_num, the OMPF bytes it sent)`.
///
/// Returning the sent bytes is the whole point: they are what
/// [`put_tail_generation_from_records`] encodes into the tail envelope, so the
/// artifact carries exactly what the live commit carried rather than a
/// re-synthesised approximation of it.
async fn drive_commit(
    harness: &CheckpointHarness,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    delta: &OwnedDelta,
    client_timestamp: u64,
) -> (u64, Bytes) {
    let ompf = delta_to_ompf_bytes(delta, &[]);
    let stream = streaming_update(vec![
        make_update_header(
            session_id,
            layer_name,
            object_id,
            client_timestamp,
            CLIENT_ID,
        ),
        make_update_data(ompf.clone()),
    ]);
    let sequence_num = handle_update_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("UpdateLayer must succeed while building the lineage fixture")
        .into_inner()
        .sequence_num;
    (sequence_num, Bytes::from(ompf))
}

/// The live state at one instant, as both a decoded snapshot and the raw
/// `obj:hdr` map — the two views the restored state is compared against.
struct Oracle {
    snapshot: ObjectSnapshot,
    header: HashMap<String, String>,
}

async fn capture_oracle(harness: &CheckpointHarness, object_id: u64) -> Oracle {
    let snapshot = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &object_id.to_string(),
    )
    .await
    .expect("read_object_state_for_test must succeed while the live layer is intact");
    let header = read_obj_header(harness, object_id).await;
    Oracle { snapshot, header }
}

/// `HGETALL le:{object_id}:obj:hdr`.
async fn read_obj_header(harness: &CheckpointHarness, object_id: u64) -> HashMap<String, String> {
    let header_key = harness.tv.key_schema().obj_header(&object_id.to_string());
    harness
        .tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL obj:hdr must succeed")
}

/// Parse a `u64` column off an `obj:hdr` map with a diagnostic on absence.
fn header_u64(header: &HashMap<String, String>, field: &str) -> u64 {
    let raw = header.get(field).unwrap_or_else(|| {
        panic!(
            "obj:hdr MUST carry `{field}`; got fields: {:?}",
            header.keys().collect::<Vec<_>>()
        )
    });
    raw.parse::<u64>()
        .unwrap_or_else(|e| panic!("obj:hdr `{field}` MUST parse as u64; got {raw:?}: {e}"))
}

/// Like [`header_u64`] but folds an absent column to `0`. Only for the three
/// restore-cursor columns, which a never-restored layer legitimately lacks.
fn header_u64_or_zero(header: &HashMap<String, String>, field: &str) -> u64 {
    header
        .get(field)
        .map(|raw| {
            raw.parse::<u64>()
                .unwrap_or_else(|e| panic!("obj:hdr `{field}` MUST parse as u64; got {raw:?}: {e}"))
        })
        .unwrap_or(0)
}

/// The oracle's own self-check, run at capture time: the decoded snapshot and
/// the raw header agree on the sequence, and the layer has never been restored.
///
/// The second half is what gives every post-restore `restore_seq_bump ==
/// <band>` assertion in this file its meaning: the column is proved absent (or
/// cleared) BEFORE the restore, so its post-restore value is a change this
/// restore made rather than a residue that was already there.
fn assert_live_oracle_is_unrestored(oracle: &Oracle, context: &str) {
    assert_eq!(
        header_u64(&oracle.header, "seq"),
        oracle.snapshot.sequence_num,
        "{context}: the live obj:hdr and the decoded snapshot must agree on `seq`"
    );
    assert_eq!(
        header_u64_or_zero(&oracle.header, "restore_seq_bump"),
        0,
        "{context}: a layer that has only ever been written live MUST carry no guard band — \
         the periodic loop clears the column on its next successful write. A non-zero value \
         here means a restore already ran, and every post-restore bump assertion in this file \
         would be reading a residue"
    );
}

/// Everything the tests share: a layer whose lineage root is a REAL
/// loop-written base, with the checkpoint loop fenced off afterwards so nothing
/// but the test can write a generation.
struct LineageFixture {
    session_id: String,
    layer_name: String,
    object_id: u64,
    /// `S0` — the loop-written base's `snap_seq`, and the layer's live sequence
    /// at the instant the base was assembled.
    s0: u64,
    base_handle: CheckpointHandle,
    base_meta: CheckpointMeta,
    /// The live state at `S0`: what a base-only restore must reproduce.
    s0_oracle: Oracle,
    /// `created_by_user_name` as StatLayer reported it while the layer was
    /// live. The restored value is compared against this rather than against a
    /// hardcoded string, so the assertion tracks whatever the create path
    /// actually stamped.
    live_created_by: String,
}

/// CreateSession → CreateLayer (populated) → two commits → wait for the loop's
/// base → **fence the loop** → capture the `S0` oracle.
///
/// The fence is `cancel()` plus a bounded join on the loop's `JoinHandle`, the
/// shape `recovery_mixed_kind_selection_test` uses. After it, no tick exists to
/// land a generation, so nothing can move the fixture underneath a test: not
/// the retention prune at `retain_generations = 2`, and not a fresh base at a
/// sequence the hand-put tail does not link onto.
async fn setup_lineage_fixture(harness: &mut CheckpointHarness, name: &str) -> LineageFixture {
    let session_store =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
    let session_id = session_store
        .create(&format!("{name}-session"), SESSION_BASE_PATH, CREATED_BY)
        .await
        .expect("session creation must succeed");
    let layer_name = format!("{name}-layer");

    let ompf = delta_to_ompf_bytes(&populated_initial_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, &layer_name),
        make_create_data(ompf),
    ]);
    let object_id = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed while building the lineage fixture")
        .into_inner()
        .object_id;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );

    // Two ordinary commits before the base, so the base carries state that no
    // single delta produced and the restored snapshot is a genuine round trip
    // of accumulated live state.
    let mut committed_seq = 0;
    for index in 0..2 {
        let (seq, _) = drive_commit(
            harness,
            &session_id,
            &layer_name,
            object_id,
            &pre_base_commit_delta(index),
            1_000 + index,
        )
        .await;
        committed_seq = seq;
    }

    // Gate on the committed sequence, not merely "any base": with
    // `interval = quiesce = 0` the loop can fire between CreateLayer and the
    // commits and land a generation that predates them, which would leave `S0`
    // non-deterministic and the tail's link arithmetic meaningless.
    let base_handle = wait_for_checkpoint_at_least(
        &harness.store,
        object_id,
        committed_seq,
        LOOP_BASE_DEADLINE_SECS,
    )
    .await;

    // ── The fence ────────────────────────────────────────────────────────
    harness.cancel.cancel();
    let loop_task = std::mem::replace(&mut harness.loop_handle, tokio::spawn(async {}));
    let _ = tokio::time::timeout(LOOP_DRAIN_BUDGET, loop_task).await;

    let s0 = base_handle.snap_seq;
    assert_eq!(
        s0, committed_seq,
        "fixture precondition: the loop's base MUST have been assembled at the layer's live \
         sequence ({committed_seq}). A higher value means a commit landed that this fixture \
         does not know about, and the hand-put tail's `from_seq = S0` link would name a \
         predecessor covering commits nothing replays"
    );

    let base_meta = harness
        .store
        .get_layer_checkpoint(&base_handle)
        .await
        .expect("get_layer_checkpoint on the loop's base must succeed")
        .expect("the loop's base must be fetchable")
        .meta;

    let live_created_by = handle_stat_layer(&harness.svc, stat_request(&session_id, &layer_name))
        .await
        .expect("StatLayer on the live layer must succeed")
        .into_inner()
        .created_by_user_name;

    let s0_oracle = capture_oracle(harness, object_id).await;
    assert_eq!(
        s0_oracle.snapshot.sequence_num, s0,
        "fixture precondition: the S0 oracle must be captured at the base's own sequence"
    );
    assert_live_oracle_is_unrestored(&s0_oracle, "S0 oracle");

    LineageFixture {
        session_id,
        layer_name,
        object_id,
        s0,
        base_handle,
        base_meta,
        s0_oracle,
        live_created_by,
    }
}

/// Drive `count` tail-range commits, returning `(sequence_num, sent OMPF)` per
/// commit and asserting the sequence run is contiguous and starts exactly one
/// above `after_seq`.
///
/// Contiguity is a **fixture precondition**, not an assumption: a gap would be
/// rejected downstream by the composition engine as a `RecordSeqGap`
/// step-down, and the test would then be asserting the step-down path while
/// claiming to assert the happy one.
async fn drive_tail_commits(
    harness: &CheckpointHarness,
    fixture: &LineageFixture,
    after_seq: u64,
    count: u64,
    first_index: u64,
) -> Vec<(u64, Bytes)> {
    let mut records = Vec::with_capacity(count as usize);
    let mut expected = after_seq + 1;
    for step in 0..count {
        let index = first_index + step;
        let record = drive_commit(
            harness,
            &fixture.session_id,
            &fixture.layer_name,
            fixture.object_id,
            &tail_commit_delta(index),
            2_000 + index,
        )
        .await;
        assert_eq!(
            record.0, expected,
            "fixture precondition: commit sequences must be contiguous. Commit {step} \
             returned {}, expected {expected}. A gap here means another writer shares this \
             layer, and the hand-put tail would carry a span its records cannot cover",
            record.0
        );
        expected += 1;
        records.push(record);
    }
    records
}

// ─────────────────────────────────────────────────────────────────────────────
// Oracle comparison
//
// The faithful-column pair this file originated —
// [`assert_faithful_columns`] and its version companion (now
// [`assert_restored_entry_versions_flattened`], b.2ti) — lives in `helpers.rs`,
// on the SUPERSET basis (it additionally covers the section version columns,
// which the three integration copies omitted). The authoritative
// excluded-column list — what a checkpoint round
// trip re-derives rather than carries, and why — is the doc block on
// `helpers::assert_faithful_columns`; do not restate it here, and do not add a
// file-local weaker copy of either assertion.
//
// The cumulative-cursor assertions below stay local: they are base-only claims
// about THIS file's fixture, not part of the shared basis, and the shared
// excluded-column block points back at them.
// ─────────────────────────────────────────────────────────────────────────────

/// The cumulative-cursor half of a **base-only** comparison, split out because
/// exactly one of its two claims survives contact with a LIVE reference.
///
/// `build_reset_ompf_from_state` enumerates fields via
/// `DiffEngine::plan_full_content`, which since b.gcb / b.vz7 emits them on a
/// total order over the CONTENT of `ObjectState` — so the reset delta's field
/// order IS deterministic, and so is the per-key assignment the re-accumulating
/// `apply_set_field` derives from it. Determinism is not agreement, though, and
/// the conclusion below is unchanged by the sort: the live cursor accumulated
/// across many commits in **authoring** interleaving, including over fields
/// later overwritten or deleted, while the reset re-accumulates from `0` in the
/// base's **emission** order over the surviving fields only. Two different walks
/// over the same key set need not hand the same number to the same key, so
/// per-key equality against the live oracle is not a property this path has —
/// sort or no sort. Asserting it would be wrong, not merely fragile.
///
/// What DOES hold against live, with every field written exactly once at
/// authoring rank `0` (see [`AUTHORING_RANK`]), is:
///
/// - the **multiset** of cumulative `set_order` values (`0..N`, whichever key
///   each one lands on), and
/// - `next_set_order` itself, which lands on `N` either way — and is the column
///   the next live write actually reads.
///
/// Asserting nothing would let the cursor silently reset to zero on a restore,
/// which is the b.qw6-adjacent hazard this pins.
///
/// The per-key assignment is not left unfenced: it is pinned
/// restore-against-restore instead of restore-against-live, by
/// [`assert_repeated_restore_agrees_per_key`].
fn assert_cumulative_set_order_columns(restored: &ObjectSnapshot, oracle: &ObjectSnapshot) {
    assert_eq!(
        restored.next_set_order, oracle.next_set_order,
        "a base-only restore MUST reproduce `next_set_order`: it is where the next live write \
         starts accumulating, and a restore that reset it to 0 would re-issue cumulative ranks \
         the surviving fields already hold"
    );
    let orders = |snap: &ObjectSnapshot| {
        let mut out: Vec<u64> = snap.fields.iter().map(|f| f.set_order).collect();
        out.sort_unstable();
        out
    };
    assert_eq!(
        orders(restored),
        orders(oracle),
        "a base-only restore MUST reproduce the cumulative set_order VALUES; only their per-key \
         assignment may differ, because the reset re-accumulates from 0 in the base's emission \
         order rather than in live's authoring interleaving"
    );
}

/// Per-key view of the cumulative cursor: `(section_id, key_name)` →
/// `Field::set_order`.
fn per_key_cumulative_orders(snap: &ObjectSnapshot) -> BTreeMap<(u64, String), u64> {
    snap.fields
        .iter()
        .map(|f| ((f.section_id, f.key_name.clone()), f.set_order))
        .collect()
}

/// The per-key claim [`assert_cumulative_set_order_columns`] deliberately does
/// NOT make against live, made where it IS sound: **two restores of the same
/// base agree key-for-key.**
///
/// The ground is that a committed base is a fixed sequence of bytes. Whatever
/// order `build_reset_ompf_from_state` chose when it built the blob is frozen in
/// the blob, so `apply_reset` / `apply_set_field` walk the same fields in the
/// same order on every restore of it and hand the same key the same cumulative
/// number every time. That is true today and was true before b.gcb / b.vz7 —
/// the pre-sort non-determinism was in BUILDING a blob, never in restoring one —
/// and nothing else in the suite currently gates it. The claim one level up (two
/// independent ASSEMBLIES of the same state produce the same blob) is the sort's
/// contribution and is asserted by the assembler lane, not here.
///
/// So a failure here is not an ordering regression: it is the restore path
/// having acquired a dependency on something outside the blob — the live state
/// it is overwriting, a clock, a map built at apply time.
///
/// The caller drives the second restore itself (wipe + `StatLayer`) rather than
/// this helper doing it, so the metric assertions that bracket the FIRST restore
/// stay one-restore assertions.
fn assert_repeated_restore_agrees_per_key(first: &ObjectSnapshot, second: &ObjectSnapshot) {
    let first_orders = per_key_cumulative_orders(first);
    let second_orders = per_key_cumulative_orders(second);

    // Non-vacuity: the assertion is only a real fence if the cursor actually
    // distributes DIFFERENT numbers over the keys. With every field written once
    // at rank 0 the assignment is a bijection onto `0..N`, so `N >= 2` is
    // exactly the condition — and it is checked, not assumed, because a fixture
    // change that collapsed the base to one field would otherwise leave this
    // green and meaningless.
    let distinct: BTreeSet<u64> = first_orders.values().copied().collect();
    assert!(
        distinct.len() >= 2,
        "fixture precondition: the base must carry at least two fields at DISTINCT cumulative \
         set_order values, or agreement between two restores is satisfied by construction. Got \
         {first_orders:?}"
    );

    assert_eq!(
        first_orders, second_orders,
        "two restores of the SAME base MUST agree on the per-key cumulative set_order, key for \
         key: the blob's field order is fixed in its bytes, so the re-accumulation that produced \
         the first assignment must produce the second. A divergence means the restore path reads \
         something outside the blob"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Metric helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot every cell of `live_edit_checkpoint_restore_lineage_total` for a
/// before/after delta. The whole pinned vocabulary, never one cell: a
/// step-down that ticks the WRONG reason is a different defect from one that
/// does not tick at all, and only a full sweep tells them apart.
fn lineage_cells(harness: &CheckpointHarness) -> Vec<(&'static str, u64)> {
    CHECKPOINT_RESTORE_LINEAGE_OUTCOMES
        .iter()
        .map(|outcome| {
            (
                *outcome,
                read_restore_lineage_counter(&harness.metrics, outcome),
            )
        })
        .collect()
}

/// Assert the per-outcome deltas since `before`. Outcomes absent from
/// `expected` must be flat.
fn assert_lineage_deltas(
    harness: &CheckpointHarness,
    before: &[(&'static str, u64)],
    expected: &[(&str, u64)],
    context: &str,
) {
    for (outcome, pre) in before {
        let want = expected
            .iter()
            .find(|(name, _)| name == outcome)
            .map(|(_, value)| *value)
            .unwrap_or(0);
        let got = read_restore_lineage_counter(&harness.metrics, outcome) - pre;
        assert_eq!(
            got, want,
            "{context}: lineage outcome {outcome:?} MUST tick {want} time(s), got {got}. This \
             family is NOT one-tick-per-attempt — one attempt that steps down twice before \
             resolving ticks three cells"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 1 — the composed round trip through the production handler
// ═════════════════════════════════════════════════════════════════════════════

/// SR-3.1 / SR-3.3 end-to-end: a base plus a real-commit tail restores the
/// **composed head state**, stamped with the composed head's numbers.
///
/// The four cursor columns `apply_restore` folds into `write_object`'s single
/// MULTI/EXEC are independently assertable, and each excludes a different
/// regression:
///
/// - `last_checkpoint_seq == S` — the sharpest probe in this file. The composed
///   path must stamp `GatheredLineage::head_seq`, not the base blob's
///   `meta.snap_seq`. A regression that passes the base's value writes `S0`
///   here AND seeds `seq` at `S0 + band`, i.e. BELOW the layer's real pre-wipe
///   sequence — the sequence-reuse staleness the guard band exists to prevent.
/// - `seq == S + margin` and `seq > pre-wipe max`.
/// - `restore_seq_bump == margin` — the band the next assembly subtracts.
/// - `last_checkpoint_ms == HAND_PUT_WRITTEN_AT_UNIX_MS` — the **newest**
///   generation's own write time. The base is loop-written, so its
///   `written_at_unix_ms` is a real `now`; a value near now here means the
///   base's meta was stamped instead of the head tail's.
///
/// Plus the faithful-column oracle (see `assert_faithful_columns` for the basis
/// and the excluded columns), the identity column (base-sourced), and the two
/// metric families.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn composed_restore_round_trips_a_tail_lineage_layer_through_the_recovery_hook() {
    let mut harness = setup_checkpoint_harness(TICK_SECS).await;
    let fixture = setup_lineage_fixture(&mut harness, "composed-round-trip").await;

    // Three real commits above the base; the tail is encoded from exactly the
    // bytes they sent.
    let records = drive_tail_commits(&harness, &fixture, fixture.s0, 3, 0).await;
    let head_seq = records.last().expect("three commits were driven").0;
    let head_oracle = capture_oracle(&harness, fixture.object_id).await;
    assert_live_oracle_is_unrestored(&head_oracle, "head oracle");
    let pre_wipe_max_seq = head_oracle.snapshot.sequence_num;
    assert_eq!(
        pre_wipe_max_seq, head_seq,
        "fixture precondition: the live sequence must equal the last commit's"
    );

    let tail_handle = put_tail_generation_from_records(
        &harness.store,
        fixture.object_id,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.s0,
        fixture.s0,
        &records,
        HAND_PUT_WRITTEN_AT_UNIX_MS,
        0,
    )
    .await;

    // `wipe_layer_state` UNLINKs `le:{oid}:obj:*` INCLUDING the stream, so the
    // high-water probe folds to `None` and the restore floor is `S` exactly. It
    // leaves the layer mapping intact, so apply's SR-2.4 identity guard passes.
    wipe_layer_state(&harness.tv, fixture.object_id).await;

    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let not_recoverable_pre = layer_recovery_metric(&harness, "not_recoverable");
    let lineage_pre = lineage_cells(&harness);

    let stat = handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "StatLayer after the wipe MUST restore the composed lineage and return Ok, got \
             {:?}: {}. An UNAVAILABLE means the restore errored rather than composing; a \
             NOT_FOUND means no lineage verified over a store holding a base and a tail that \
             links onto it",
            status.code(),
            status.message()
        )
    })
    .into_inner();

    assert_eq!(
        stat.created_by_user_name, fixture.live_created_by,
        "identity columns are BASE-sourced even on a composed restore. The rule is structural — \
         a tail's identity is base-sourced by construction — and does NOT rest on the adapters' \
         identity-empty meta-absent placeholder, which b.hka (d) replaced with a refusal. A \
         mismatch here means identity was taken from the head generation"
    );

    let header = read_obj_header(&harness, fixture.object_id).await;
    let margin = harness.config.checkpoint_seq_safety_margin;

    assert_eq!(
        header_u64(&header, "last_checkpoint_seq"),
        head_seq,
        "last_checkpoint_seq MUST be the COMPOSED HEAD ({head_seq}), not the base's snap_seq \
         ({}). The base's value is the regression this line exists to exclude: it under-states \
         what the restore materialized and seeds the writer cursor {} commits too low, which \
         re-issues sequence numbers the replayed tail already used",
        fixture.s0,
        head_seq - fixture.s0
    );
    assert_eq!(
        header_u64(&header, "seq"),
        head_seq + margin,
        "seq MUST be the composed head ({head_seq}) + the safety margin ({margin}). The stream \
         was wiped with the rest of `obj:*`, so the high-water probe folds to None and the \
         restore floor is the head alone"
    );
    assert!(
        header_u64(&header, "seq") > pre_wipe_max_seq,
        "the seeded writer cursor MUST sit strictly above the pre-wipe maximum \
         ({pre_wipe_max_seq}); anything else re-issues live sequence numbers"
    );
    assert_eq!(
        header_u64(&header, "restore_seq_bump"),
        margin,
        "restore_seq_bump MUST record the band actually applied — the next assembly subtracts \
         it so the artificial restore jump is not measured as commit traffic"
    );
    assert_eq!(
        header_u64(&header, "last_checkpoint_ms"),
        HAND_PUT_WRITTEN_AT_UNIX_MS,
        "last_checkpoint_ms MUST be the NEWEST generation's own write time (the head tail's). \
         The base is loop-written, so a value near `now` here means the base's meta was \
         stamped instead — the cadence signal the periodic loop's interval arm reads would \
         then describe the wrong generation"
    );

    // Both spans are strictly below the margin, so the band is the margin
    // whichever meta sources `observed_commit_span`. This test therefore does
    // NOT discriminate the sourcing decision — that is the engineer's
    // adjudication, pinned by
    // `the_guard_band_takes_its_span_from_the_head_tail_header` below and
    // reported to Task 7's SDD pass. The assertions above stay exact without
    // being brittle on it.
    let tail_meta = harness
        .store
        .get_layer_checkpoint(&tail_handle)
        .await
        .expect("get_layer_checkpoint on the hand-put tail must succeed")
        .expect("the hand-put tail must be fetchable")
        .meta;
    for (label, span) in [
        ("base", fixture.base_meta.observed_commit_span),
        ("head tail", tail_meta.observed_commit_span),
    ] {
        assert!(
            span < margin,
            "fixture precondition: the {label} meta's observed_commit_span ({span}) must stay \
             below the margin ({margin}), so `band = max(margin, span)` is the margin whichever \
             meta is consulted"
        );
    }

    let restored = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &fixture.object_id.to_string(),
    )
    .await
    .expect("post-recovery read_object_state_for_test must succeed");
    assert_faithful_columns(
        &restored,
        &head_oracle.snapshot,
        FaithfulColumnsOptions::default(),
        "composed restore",
    );
    // The version columns are a FLATTENING claim on the persisted side (b.2ti):
    // `apply_restore` re-stamps the three a diff builder filters on
    // (`Section::last_version`, `Field::version`, `TimeSample::version`) onto
    // `seeded_seq` before `write_object`, so those carry no per-commit spread at
    // all. `Section::first_version` is exempt and comes back PRESERVED, which is
    // what `RestoredFirstVersions::ComposedLineage` names: this restore resolved
    // `gathered_composed` (asserted above), so base-era sections carry the
    // decode's stamp and any section the replayed range created carries the
    // sequence composition replayed. `seeded_seq` here is `header seq` as
    // asserted above — the restore floor folds to the composed head because the
    // stream was wiped with the rest of `obj:*`. The presence half of the old
    // assertion is retained: every entry the replayed range wrote must still
    // exist after the restore.
    // `fixture.s0` twice: it is BOTH the floor of the range whose survival the
    // presence half checks AND the base-era/tail-era split point the exempt
    // column is predicted from. They coincide here because the fixture has one
    // lineage generation; the helper keeps them separate because they answer
    // different questions (see its rustdoc).
    assert_restored_entry_versions_flattened(
        &restored,
        &head_oracle.snapshot,
        RestoredVersionRanges {
            replayed_range_floor: fixture.s0,
            lineage_root: fixture.s0,
        },
        head_seq + margin,
        RestoredFirstVersions::ComposedLineage,
        FaithfulColumnsOptions::default(),
        "composed restore",
    );
    // Fixture-level non-vacuity for the section arm of the assertion above.
    // A section's `last_version` moves only on a structural command
    // (`src/domain/delta.rs`), so a tail range of pure field / time-sample
    // commits would leave `sections_above` empty on the ORACLE and the fixture
    // would stop covering structural replay entirely. Deliberately here and not
    // inside the shared helper: a tail range that touches no section
    // legitimately contributes nothing, so a guard there would go red on
    // correct callers.
    assert!(
        !sections_above(&head_oracle.snapshot, fixture.s0).is_empty(),
        "fixture precondition: the replayed range must stamp at least one section above the base \
         ({}), otherwise this fixture's replayed commits are pure field / time-sample traffic and \
         the restore's structural half silently stops being covered here",
        fixture.s0
    );
    assert_eq!(
        restored.object_id, fixture.object_id,
        "the restore must materialize under the ORIGINAL object_id, never a re-minted one"
    );

    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        1,
        "exactly ONE `restored` tick: not zero (recovery never fired), not two (the handler \
         retried past the admission latch)"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        0,
        "`error` MUST stay flat — a tick here means the composed path failed and the verdict \
         was an internal error rather than a restore"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "not_recoverable") - not_recoverable_pre,
        0,
        "`not_recoverable` MUST stay flat — a restorable lineage is on disk"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("gathered_composed", 1)],
        "an intact base + tail lineage",
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 2 — the base-only regression fence
// ═════════════════════════════════════════════════════════════════════════════

/// The floor: a store holding only a base restores **exactly as it did before
/// composed restore existed**.
///
/// Same fixture, no tail hand-put. Every expected value is pinned as a literal
/// derived from the documented formula (`restore_floor + band`, `band =
/// max(margin, observed_commit_span)`), never by diffing against a HEAD run —
/// a fence that recorded whatever HEAD happened to produce would ratify a
/// regression instead of catching one.
///
/// The state comparison here is the strictest in the file: a base-only restore
/// is a single reset decode with **no replay**, so the cumulative `set_order`
/// cursor is compared too — against live on the two columns that admit it (see
/// `assert_cumulative_set_order_columns` for why per-key equality against live
/// is not one of them), and then key-for-key against a SECOND restore of the
/// same base at the end of the test (`assert_repeated_restore_agrees_per_key`).
/// The second restore runs after every metric assertion so those stay
/// one-restore assertions.
///
/// Lineage-wise this must resolve `gathered_base` with every `stepped_down_*`
/// cell flat: base-only is deliberately NOT routed through the composition
/// engine, so a corrupt base still surfaces as a decode failure inside apply
/// rather than as a step-down to an older generation.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_base_only_store_restores_unchanged_through_the_composed_wiring() {
    let mut harness = setup_checkpoint_harness(TICK_SECS).await;
    let fixture = setup_lineage_fixture(&mut harness, "base-only-fence").await;

    wipe_layer_state(&harness.tv, fixture.object_id).await;

    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let not_recoverable_pre = layer_recovery_metric(&harness, "not_recoverable");
    let lineage_pre = lineage_cells(&harness);

    let stat = handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "StatLayer after the wipe MUST restore the base-only store and return Ok, got \
             {:?}: {}",
            status.code(),
            status.message()
        )
    })
    .into_inner();
    assert_eq!(
        stat.created_by_user_name, fixture.live_created_by,
        "the base's identity columns restore verbatim"
    );

    let header = read_obj_header(&harness, fixture.object_id).await;
    let margin = harness.config.checkpoint_seq_safety_margin;
    assert_eq!(
        header_u64(&header, "last_checkpoint_seq"),
        fixture.s0,
        "on a base-only lineage the head IS the base, so last_checkpoint_seq is the base's own \
         snap_seq ({}). A different value means the base-only path started consulting a \
         lineage head it does not have",
        fixture.s0
    );
    assert_eq!(
        header_u64(&header, "seq"),
        fixture.s0 + margin,
        "seq = restore_floor + band = base snap_seq ({}) + margin ({margin}), the pre-Epic-4 \
         formula unchanged",
        fixture.s0
    );
    assert_eq!(
        header_u64(&header, "restore_seq_bump"),
        margin,
        "the base's observed_commit_span is below the margin, so the band IS the margin"
    );
    assert_eq!(
        header_u64(&header, "last_checkpoint_ms"),
        fixture.base_meta.written_at_unix_ms,
        "last_checkpoint_ms MUST be the restored base's own written_at_unix_ms, read back off \
         its meta rather than hardcoded — the loop stamps a real `now` there"
    );
    assert!(
        fixture.base_meta.written_at_unix_ms > 0,
        "fixture precondition: the loop-written base must carry a real write time, or the \
         assertion above would be satisfied by apply's `0 -> now_unix_ms()` fallback"
    );

    let restored = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &fixture.object_id.to_string(),
    )
    .await
    .expect("post-recovery read_object_state_for_test must succeed");
    // Content only, deliberately: a base-only restore has NO replayed range, so
    // `assert_restored_entry_versions_flattened` is omitted here — its
    // non-vacuity guard (the oracle must hold a field above the base) would fire
    // on a fixture that is correct by construction. The b.2ti flattening claim
    // it also carries is covered for the base-only shape by
    // `src/checkpoint/apply.rs`'s `Ompf` arm unit test.
    assert_faithful_columns(
        &restored,
        &fixture.s0_oracle.snapshot,
        FaithfulColumnsOptions::default(),
        "base-only restore",
    );
    assert_cumulative_set_order_columns(&restored, &fixture.s0_oracle.snapshot);

    // A restore is a pure read: the generation it restored FROM must still be
    // listed and still be fetchable afterwards.
    let post_listing = harness
        .store
        .list_layer_checkpoints(fixture.object_id)
        .await
        .expect("post-recovery list_layer_checkpoints must succeed");
    assert!(
        post_listing.contains(&fixture.base_handle),
        "the restored base@{} MUST still be listed after the restore — recovery reads a \
         generation, it never consumes one. Live listing: {post_listing:?}",
        fixture.s0
    );

    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        1,
        "exactly ONE `restored` tick"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        0,
        "`error` MUST stay flat on the base-only path"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "not_recoverable") - not_recoverable_pre,
        0,
        "`not_recoverable` MUST stay flat — a base is on disk"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("gathered_base", 1)],
        "a base-only store",
    );

    // ── The per-key cursor fence: restore the SAME base a second time ────
    //
    // Last, so every assertion above still describes exactly one restore. The
    // wipe is the same full `le:{oid}:obj:*` UNLINK, and the store is untouched
    // (asserted above: the generation is still listed), so the second StatLayer
    // fetches the identical blob and the comparison is restore-vs-restore with
    // the blob held constant. See `assert_repeated_restore_agrees_per_key` for
    // what a failure would mean.
    let restored_before_second = layer_recovery_metric(&harness, "restored");
    wipe_layer_state(&harness.tv, fixture.object_id).await;
    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "the SECOND StatLayer must restore the same base and return Ok, got {:?}: {}. \
             Recovery reads a generation rather than consuming one, so a failure here is about \
             the restore path, not about the store",
            status.code(),
            status.message()
        )
    });
    let restored_again = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &fixture.object_id.to_string(),
    )
    .await
    .expect("read_object_state_for_test after the second restore must succeed");
    // A SECOND restore genuinely ran. Without this the per-key comparison below
    // would go green on a StatLayer that found live state and restored nothing —
    // it would be comparing the first restore against itself.
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_before_second,
        1,
        "the second StatLayer MUST have performed a second restore, not served surviving state"
    );
    // Base-only again, so the same omission as the first restore above:
    // content only, no `assert_restored_entry_versions_flattened` (see its doc
    // in `helpers.rs`).
    assert_faithful_columns(
        &restored_again,
        &fixture.s0_oracle.snapshot,
        FaithfulColumnsOptions::default(),
        "second base-only restore",
    );
    assert_repeated_restore_agrees_per_key(&restored, &restored_again);
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 3 — a broken newest lineage steps down, to both destinations
// ═════════════════════════════════════════════════════════════════════════════

/// Step-down arm (a): the newest generation's link points at nothing, and the
/// walk falls back to **the base**.
///
/// The tail claims `from_seq = S0 + 1` — a predecessor generation that does not
/// exist — so link verification reports `LinkAbsent` and the selection steps
/// down to the base at `S0`. The restore is still `Restored`, not an error:
/// stepping down is a verdict, and stranding a perfectly restorable base one
/// generation behind is exactly what FR-6 forbids.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_broken_newest_lineage_steps_down_to_the_base() {
    let mut harness = setup_checkpoint_harness(TICK_SECS).await;
    let fixture = setup_lineage_fixture(&mut harness, "step-down-to-base").await;

    let records = drive_tail_commits(&harness, &fixture, fixture.s0, 3, 0).await;
    // Drop the first commit from the tail and claim a predecessor at `S0 + 1`:
    // a link into nothing. The remaining records stay contiguous, so the
    // artifact itself is well-formed — only its LINK is broken, which is the
    // partial-store-loss shape rather than a corrupt file.
    let broken_records = &records[1..];
    let broken_tail = put_tail_generation_from_records(
        &harness.store,
        fixture.object_id,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.s0,
        fixture.s0 + 1,
        broken_records,
        HAND_PUT_WRITTEN_AT_UNIX_MS,
        0,
    )
    .await;
    assert!(
        broken_tail.snap_seq > fixture.s0,
        "fixture precondition: the broken tail must be the LISTING HEAD, or the walk would \
         never consider it and no step-down would be exercised"
    );

    wipe_layer_state(&harness.tv, fixture.object_id).await;

    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let lineage_pre = lineage_cells(&harness);

    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "a broken newest lineage MUST step down to the base and still restore, got {:?}: \
             {}. An error verdict here is the stranded-generation failure FR-6 forbids",
            status.code(),
            status.message()
        )
    });

    let header = read_obj_header(&harness, fixture.object_id).await;
    assert_eq!(
        header_u64(&header, "last_checkpoint_seq"),
        fixture.s0,
        "the step-down destination is the base, so the restore is stamped at the base's own \
         snap_seq"
    );
    assert_eq!(
        header_u64(&header, "seq"),
        fixture.s0 + harness.config.checkpoint_seq_safety_margin,
        "the seeded cursor follows the base-only formula — nothing from the abandoned tail may \
         leak into it"
    );

    let restored = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &fixture.object_id.to_string(),
    )
    .await
    .expect("post-recovery read_object_state_for_test must succeed");
    // The destination is the base, so this is a base-only restore's state: no
    // replayed range, hence no `assert_restored_entry_versions_flattened` (see
    // its doc in `helpers.rs`).
    assert_faithful_columns(
        &restored,
        &fixture.s0_oracle.snapshot,
        FaithfulColumnsOptions::default(),
        "step-down to the base",
    );
    assert_cumulative_set_order_columns(&restored, &fixture.s0_oracle.snapshot);

    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        1,
        "a step-down that still restores reports `restored`, once"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        0,
        "`error` MUST stay flat: a broken lineage is a verdict, never an error"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("stepped_down_link_broken", 1), ("gathered_base", 1)],
        "a link into nothing above a restorable base",
    );
}

/// Step-down arm (b): the newest generation is broken and the walk falls back
/// to an **older COMPOSED lineage** — a different claim from arm (a), because
/// the destination is itself a composition rather than the floor.
///
/// Fixture: base@S0, tail_A over `(S0, S1]` (intact, real commits), tail_B over
/// `(S1 + 1, S2]` whose link names a generation that does not exist. `S2 > S1`,
/// so tail_B is the listing head and is reached first.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_broken_newest_lineage_steps_down_to_an_older_composed_lineage() {
    let mut harness = setup_checkpoint_harness(TICK_SECS).await;
    let fixture = setup_lineage_fixture(&mut harness, "step-down-to-composed").await;

    // tail_A: the intact lineage the walk must land on.
    let records_a = drive_tail_commits(&harness, &fixture, fixture.s0, 2, 0).await;
    let s1 = records_a.last().expect("two commits were driven").0;
    let s1_oracle = capture_oracle(&harness, fixture.object_id).await;
    let tail_a = put_tail_generation_from_records(
        &harness.store,
        fixture.object_id,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.s0,
        fixture.s0,
        &records_a,
        HAND_PUT_WRITTEN_AT_UNIX_MS,
        0,
    )
    .await;

    // tail_B: three more real commits, of which the first is dropped so the
    // envelope can claim a predecessor at `S1 + 1` that no generation occupies.
    let records_b = drive_tail_commits(&harness, &fixture, s1, 3, 10).await;
    let tail_b = put_tail_generation_from_records(
        &harness.store,
        fixture.object_id,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.s0,
        s1 + 1,
        &records_b[1..],
        HAND_PUT_WRITTEN_AT_UNIX_MS,
        0,
    )
    .await;
    assert!(
        tail_b.snap_seq > tail_a.snap_seq,
        "fixture precondition: tail_B ({}) must be the listing head over tail_A ({})",
        tail_b.snap_seq,
        tail_a.snap_seq
    );

    wipe_layer_state(&harness.tv, fixture.object_id).await;

    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let lineage_pre = lineage_cells(&harness);

    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "the walk MUST step down from the broken head onto the older COMPOSED lineage and \
             restore, got {:?}: {}",
            status.code(),
            status.message()
        )
    });

    let header = read_obj_header(&harness, fixture.object_id).await;
    assert_eq!(
        header_u64(&header, "last_checkpoint_seq"),
        s1,
        "the step-down destination is base + tail_A, so the stamp is tail_A's head ({s1}) — \
         not the base's ({}), and not the abandoned tail_B's",
        fixture.s0
    );
    assert_eq!(
        header_u64(&header, "seq"),
        s1 + harness.config.checkpoint_seq_safety_margin,
        "the seeded cursor follows the composed formula against tail_A's head"
    );

    let restored = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &fixture.object_id.to_string(),
    )
    .await
    .expect("post-recovery read_object_state_for_test must succeed");
    assert_faithful_columns(
        &restored,
        &s1_oracle.snapshot,
        FaithfulColumnsOptions::default(),
        "step-down to an older composed lineage",
    );
    // `seeded_seq` is tail_A's head + the margin — the value asserted on the
    // header just above (the wipe took the stream, so the restore floor is the
    // step-down destination's head alone). The step-down still resolves a
    // COMPOSED lineage — `base@s0` plus tail_A — so `first_version` splits at
    // the same lineage root: base-era sections at the decode's stamp, anything
    // tail_A created at the sequence `s1_oracle` holds for it. `fixture.s0` is
    // passed for both the presence-range floor and the lineage root — the same
    // value, two claims (see the helper's rustdoc).
    assert_restored_entry_versions_flattened(
        &restored,
        &s1_oracle.snapshot,
        RestoredVersionRanges {
            replayed_range_floor: fixture.s0,
            lineage_root: fixture.s0,
        },
        s1 + harness.config.checkpoint_seq_safety_margin,
        RestoredFirstVersions::ComposedLineage,
        FaithfulColumnsOptions::default(),
        "step-down to an older composed lineage",
    );

    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        1,
        "a step-down that still restores reports `restored`, once"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        0,
        "`error` MUST stay flat"
    );
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("stepped_down_link_broken", 1), ("gathered_composed", 1)],
        "a broken head above an intact composed lineage",
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 4 — the guard band's span input is HEADER-sourced
// ═════════════════════════════════════════════════════════════════════════════

/// The band is `max(configured margin, observed_commit_span)`, and on a
/// composed restore the span comes from the head tail's **envelope header**
/// (`to_seq - from_seq`), not from its meta.
///
/// The two sourcings are indistinguishable in every other test in this file
/// because the margin dominates there. This one inverts that deliberately:
/// `margin = 5`, a tail spanning `10` commits, and a meta whose
/// `observed_commit_span` is `0` — the placeholder shape both adapters' tail
/// meta-absent fallbacks produce.
///
/// - **Header-sourced** (the landed decision): `band = max(5, 10) = 10`, so
///   `restore_seq_bump == 10` and `seq == S + 10`.
/// - **Meta-sourced**: the placeholder zero folds the band back to `max(5, 0) =
///   5`, so `seq == S + 5` — a band SMALLER than the layer's own measured
///   commit rate, on exactly the hot layer that needs it most. That is the
///   b.8y5 smaller-band hazard direction, and it is why the header (which is
///   the artifact's own framing and cannot be zeroed by a placeholder) is the
///   right input.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_guard_band_takes_its_span_from_the_head_tail_header() {
    const MARGIN: u64 = 5;
    const TAIL_COMMITS: u64 = 10;

    let mut harness = setup_checkpoint_harness_with_seq_margin(TICK_SECS, MARGIN).await;
    let fixture = setup_lineage_fixture(&mut harness, "span-source").await;

    let records = drive_tail_commits(&harness, &fixture, fixture.s0, TAIL_COMMITS, 0).await;
    let head_seq = records.last().expect("ten commits were driven").0;
    assert_eq!(
        head_seq - fixture.s0,
        TAIL_COMMITS,
        "fixture precondition: the tail must span exactly {TAIL_COMMITS} commits, which is what \
         makes the header span ({TAIL_COMMITS}) and the placeholder meta span (0) straddle the \
         margin ({MARGIN})"
    );

    // `observed_commit_span: 0` — the placeholder shape. A restore that read
    // the span off the meta would see no measurement at all.
    put_tail_generation_from_records(
        &harness.store,
        fixture.object_id,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.s0,
        fixture.s0,
        &records,
        HAND_PUT_WRITTEN_AT_UNIX_MS,
        0,
    )
    .await;

    wipe_layer_state(&harness.tv, fixture.object_id).await;

    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "StatLayer must restore the composed lineage, got {:?}: {}",
            status.code(),
            status.message()
        )
    });

    let header = read_obj_header(&harness, fixture.object_id).await;
    assert_eq!(
        header_u64(&header, "restore_seq_bump"),
        TAIL_COMMITS,
        "the band MUST be the head tail's HEADER span ({TAIL_COMMITS}), which dominates the \
         configured margin ({MARGIN}). A bump of {MARGIN} means the span was read off the tail \
         META, whose placeholder zero degrades a hot layer's band back to the margin — the \
         smaller-band hazard the header sourcing exists to avoid"
    );
    assert_eq!(
        header_u64(&header, "seq"),
        head_seq + TAIL_COMMITS,
        "seq = restore_floor + band = composed head ({head_seq}) + {TAIL_COMMITS}"
    );
    assert_eq!(
        header_u64(&header, "last_checkpoint_seq"),
        head_seq,
        "the composed head stamp is unaffected by which meta sourced the span"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 5 — the RecoveryOutcome vocabulary is unchanged
// ═════════════════════════════════════════════════════════════════════════════

/// Composed restore adds NO new recovery verdict and no fifth `result` label.
///
/// Two halves:
///
/// 1. `try_restore_layer`'s return is matched **exhaustively, with no wildcard
///    arm**, so a fourth `RecoveryOutcome` variant becomes a compile error in
///    this file rather than a silently-unasserted path. The three gRPC handlers
///    branch on exactly these three variants; a fourth would have to be handled
///    at five call sites whose correct behaviour is identical to one that
///    already exists.
/// 2. The set of `result` label values present on
///    `checkpoint_recovery_attempts_total{scope="layer"}` is **exactly** the
///    four-value grid. That grid's `sum by (result) == attempts` identity is
///    what makes the counter readable at all, and it must survive the lineage
///    family being added alongside it — the lineage outcomes are a separate
///    family precisely so they cannot land here as a fifth `result`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn recovery_outcome_vocabulary_is_unchanged_by_composed_restore() {
    let mut harness = setup_checkpoint_harness(TICK_SECS).await;
    let fixture = setup_lineage_fixture(&mut harness, "outcome-vocabulary").await;

    let records = drive_tail_commits(&harness, &fixture, fixture.s0, 2, 0).await;
    put_tail_generation_from_records(
        &harness.store,
        fixture.object_id,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.s0,
        fixture.s0,
        &records,
        HAND_PUT_WRITTEN_AT_UNIX_MS,
        0,
    )
    .await;

    wipe_layer_state(&harness.tv, fixture.object_id).await;

    // The hook is built directly (not reached through a handler) because this
    // test's subject IS its return type. `harness.config` is the very config
    // the harness's own hook runs under, so this one cannot drift into a
    // disabled short-circuit.
    let hook = build_recovery_hook_for_e2e(
        Arc::clone(&harness.svc.layer_store),
        Arc::clone(&harness.store),
        harness.config.clone(),
        Arc::clone(&harness.metrics),
        // The harness's OWN collaborators, not seconds: the same ext store the
        // service writes through, and the same `BaseBootstrap` unit (whose
        // in-flight bound is a field on the unit). This hook is a second handle
        // on the harness's wiring, not a second wiring.
        E2eRecoveryHookWiring {
            ext_value_store: Arc::clone(&harness.svc.ext_value_store),
            base_bootstrap: harness
                .wire_restore_rebase
                .then(|| Arc::clone(&harness.base_bootstrap)),
        },
    );

    match hook.try_restore_layer(fixture.object_id).await {
        RecoveryOutcome::Restored => {}
        RecoveryOutcome::NotRecoverable => panic!(
            "an intact base + tail lineage MUST restore; NotRecoverable means selection found \
             nothing restorable over a store that demonstrably holds a linked pair"
        ),
        RecoveryOutcome::Error(err) => panic!(
            "an intact base + tail lineage MUST restore; got Error({err:?}). A Permanent here \
             is a composition or decode rejection; a Transient is a store or write failure"
        ),
    }

    let scrape = live_edit::observability::metrics::test_util::render(&harness.metrics);
    let mut results: BTreeSet<String> = BTreeSet::new();
    for line in scrape.lines() {
        let line = line.trim();
        if !line.starts_with("live_edit_checkpoint_recovery_attempts_total{") {
            continue;
        }
        if !line.contains("scope=\"layer\"") {
            continue;
        }
        let needle = "result=\"";
        let Some(index) = line.find(needle) else {
            continue;
        };
        let after = &line[index + needle.len()..];
        let Some(end) = after.find('"') else {
            continue;
        };
        results.insert(after[..end].to_owned());
    }

    let expected: BTreeSet<String> = ["already_present", "error", "not_recoverable", "restored"]
        .into_iter()
        .map(str::to_owned)
        .collect();
    assert_eq!(
        results, expected,
        "`live_edit_checkpoint_recovery_attempts_total{{scope=\"layer\"}}` must expose EXACTLY \
         the four-value result grid. An extra value means a lineage outcome leaked onto this \
         counter as a fifth `result`, which breaks `sum by (result) == attempts`; a missing one \
         means the pre-init seeding loop stopped covering the grid"
    );
}
