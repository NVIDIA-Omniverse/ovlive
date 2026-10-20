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

//! **The fence on the shared E2E harness's recovery-hook wiring.** Plan
//! `b.dfh`, Epic 5 (`t1.dfh.be`), Task 2 (`t2.dfh.be.5s`), Subtask
//! `t3.dfh.be.5s.gk`. Closes the regression class Issue **b.kfg** records.
//!
//! ## Why this file exists
//!
//! `src/main.rs:731-738` builds the production `RecoveryHook` as
//! `RecoveryHook::new(..).with_ext_value_store(..).with_base_bootstrap(..)` —
//! four constructor arguments and **two** builder calls. The shared harness
//! assembles the same object by hand
//! (`helpers::recovery_hook_for_e2e` / `build_recovery_hook_for_e2e`), so a
//! dropped builder call is invisible: no compile error, and no existing test
//! observes the collaborator it lost. That is exactly how b.kfg happened —
//! **both** builders were missing from the shared helper for the whole of
//! Epic 4, six consumer files grew a field-swap workaround apiece, and the
//! stock harness quietly exercised a hook production does not ship.
//!
//! It is also the second instance of the class. The first was the b.b2j
//! object-id counter-repair hook, which is why
//! `helpers::assert_harness_carries_object_id_counter_repair` had to be
//! written. A fold with no fence regresses the same way, so the fold
//! (Subtask `t3.dfh.be.5s.qd`) lands with this module beside it.
//!
//! **These tests mirror `src/main.rs`'s wiring, not any one consumer's
//! scenario.** Every one of them builds a STOCK harness — no field swap, no
//! locally-rebuilt hook — because the claim under test is precisely "what the
//! harness hands you by default is what the binary ships".
//!
//! ## Structural where possible, behavioural where necessary
//!
//! `RecoveryHook` exposes no accessor for its optional collaborators
//! (`ext_value_store` / `base_bootstrap` are private fields with no getter), and
//! this Task forbids production-code changes. So five of the six tests below
//! are **behavioural**: they drive a real restore through the stock hook and
//! assert an outcome only a correctly-wired collaborator can produce.
//!
//! | Test | Builder fenced | How the claim is expressed |
//! |---|---|---|
//! | 1 | (neither) | structural — one `BaseBootstrap` `Arc`, service side |
//! | 2 | `.with_ext_value_store(..)` | a composed restore's tail-carried blob materialises in Valkey |
//! | 3 | `.with_base_bootstrap(..)` | at the DEFAULT tail flag a restore re-bases: `Base` at `seeded_seq`, both cursor columns restamped |
//! | 4 | the documented opt-out | `wire_restore_rebase = false` ⇒ the RESTORE's own stamp survives |
//! | 5 | the three-way gate | tail flag PINNED off ⇒ no re-base, gate DERIVED from `harness.config` |
//! | 6 | both, through `install_recovery_hook` | a REBUILT hook still re-bases — Test 3's claim re-run through the second entry point |
//!
//! Tests 4 and 5 fence the two ways "no re-base happened" can be *correct*, so
//! that neither can be confused with the wiring defect Test 3 fences.
//!
//! ## Synchronization
//!
//! Per `docs-internal/test-writing-guide.md` § Synchronization Patterns: no
//! `tokio::time::sleep` is used as coordination anywhere in this file, and no
//! assertion is made about elapsed time. Every wait is a bounded poll on
//! observable state under a named constant:
//!
//! | Wait | Bound | What a timeout means |
//! |---|---|---|
//! | the loop's base lands | [`LOOP_BASE_DEADLINE_SECS`] (clamped up by the helper's own load-tolerant floor) | the periodic loop never assembled a base at or above the fixture's committed sequence |
//! | the fenced loop task drains | [`LOOP_DRAIN_BUDGET`] | a cancelled loop did not return at its next `select!` |
//! | the re-base task completes | [`TRACKER_IDLE_DEADLINE_SECS`] via `helpers::wait_for_writer_idle` | a task spawned on the shared `TaskTracker` never finished |
//!
//! The re-base drain is the tracker, never a timer — the SR-3.6 re-base is a
//! fire-and-forget task the hook does not await, so tracker quiescence is what
//! makes it deterministically observable. It is equally what makes Tests 4 and
//! 5's NEGATIVE claims sound: after the tracker is empty, a re-base that *had*
//! been spawned has already completed and would be visible.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; see CLAUDE.md.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface};

use live_edit::checkpoint::{CheckpointAdapter, CheckpointHandle, CheckpointKind, RecoveryOutcome};
use live_edit::config::Config;
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::storage::external_values::{
    ext_content_hash, ExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::session::SessionStore;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, SetField, StructCommand};

use super::helpers::*;
// Reused rather than re-implemented: the surgical ext-namespace wipe is a
// correctness contract (explicit hashes only — a wildcard would also catch the
// `le:ext:{gc}:*` bookkeeping sets and change what Test 2 means). It is
// `pub(super)` precisely so siblings can consume it; this is a read-only
// dependency and does not modify that file.
use super::e2e_ext_value_recovery_test::wipe_ext_namespace_for_hashes;
use crate::common::owned_delta::delta_to_ompf_bytes;

// ─────────────────────────────────────────────────────────────────────────────
// Named waits and fixture constants — see the module header's budget table
// ─────────────────────────────────────────────────────────────────────────────

/// Checkpoint-loop tick cadence. `1` is the value every e2e recovery file in
/// this module uses: the harness sets `interval = quiesce = 0`, so a
/// just-written layer is due on the very next tick.
const TICK_SECS: u64 = 1;

/// Deadline handed to `wait_for_checkpoint_at_least`. The helper clamps it up to
/// its own load-tolerant floor, so this is the *lower* bound on how long a
/// contended Valkey may take before the wait is called a failure.
const LOOP_BASE_DEADLINE_SECS: u64 = 10;

/// Bound on the post-`cancel()` drain of the checkpoint loop's `JoinHandle`. The
/// loop returns at its next `select!` on the cancellation token, so this is
/// never the wait actually observed — it exists so a wedged loop fails the test
/// promptly instead of hanging the binary.
const LOOP_DRAIN_BUDGET: Duration = Duration::from_secs(5);

/// Deadline handed to `helpers::wait_for_writer_idle` when draining the shared
/// `TaskTracker` after a restore. A timeout here is a `BaseBootstrap` bug (a
/// spawned task that never completes), never load — the helper clamps this up to
/// its own load-tolerant floor.
const TRACKER_IDLE_DEADLINE_SECS: u64 = 10;

/// `client_id` on every UpdateLayer header. Only needs to be stable and
/// non-empty; self-echo suppression is not under test.
const CLIENT_ID: &str = "harness-parity-client";

const SESSION_BASE_PATH: &str = "/stage/harness-parity";
const CREATED_BY: &str = "harness-parity-user";

/// The section every commit in this file writes into.
const ROOT_SECTION_ID: u64 = 7;

/// Section minted by the tail-range commit, so the replayed range carries
/// structural work as well as the external-value field.
const TAIL_SECTION_ID: u64 = 700;

/// Authoring rank stamped on every `SetField`. Every field here is written
/// exactly once, so the value only has to be stable.
const AUTHORING_RANK: u64 = 0;

/// Byte length of Test 2's external value. Large enough to be unambiguously an
/// out-of-band OMPF part rather than something that could be mistaken for an
/// inline value, small enough to stay free.
const EXT_BLOB_LEN: usize = 4 * 1024;

/// The one-based OMPF part index the ext-carrying `SetField` points at. Part 0
/// is the OODF delta; `delta_to_ompf_bytes(delta, &[blob])` puts `blob` at
/// Part 1.
const EXT_VALUE_INDEX: u32 = 1;

/// Field name carrying Test 2's external value.
const EXT_FIELD: &str = "harness-parity-ext-field";

/// `restore_seq_bump` after a SUCCESSFUL re-base: the column is RETIRED, not
/// merely lowered. Spelled once here and read by Test 3 (see
/// `restore_rebase_test.rs`, which owns the full landing contract).
const RETIRED_BUMP: u64 = 0;

// ─────────────────────────────────────────────────────────────────────────────
// Payload builders
// ─────────────────────────────────────────────────────────────────────────────

/// CreateLayer payload: one section plus one inline field, so the loop's base
/// carries real state.
fn initial_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_string(),
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: ROOT_SECTION_ID,
            section_name: "harness-parity-root".to_string(),
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

/// One inline-field UpdateLayer payload.
///
/// `base_version: 1` (any non-zero) marks this an INCREMENTAL diff. With
/// `base_version == 0` and `is_diff == true` the domain engine takes the reset
/// branch and wipes prior state, which would empty the fixture.
fn inline_field_delta(name: &str) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: name.to_string(),
            value_or_ext_hash: vec![0x01, 0x02, 0x03],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// The tail-range commit for Test 2: a new section plus one field whose value is
/// an EXTERNAL reference, so the replayed range carries a blob inline in its
/// OMPF parts and the restore has something to pre-store (SR-3.4).
fn tail_ext_delta(hash: &[u8; 32], blob_len: usize) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: ROOT_SECTION_ID,
            section_id: TAIL_SECTION_ID,
            section_name: "harness-parity-tail-section".to_string(),
            section_type: 2,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: TAIL_SECTION_ID,
            key_name: EXT_FIELD.to_string(),
            value_or_ext_hash: hash.to_vec(),
            ext_value_size: blob_len as u64,
            ext_value_index: EXT_VALUE_INDEX,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// Deterministic external-value payload. An RNG would make a failing run
/// unreproducible for no benefit — the bytes only have to be non-trivial and
/// stable, so that `ext_content_hash` of them is the fixture's own contract.
fn ext_blob() -> Vec<u8> {
    (0..EXT_BLOB_LEN)
        .map(|i| ((i % 251) as u8) ^ 0x5A)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture + read helpers
//
// All file-local. `helpers.rs` is owned by sibling Subtasks in this Task and is
// deliberately not edited from here; nothing below is general enough to belong
// there anyway (each item is either a payload shape or a two-line Valkey read).
// ─────────────────────────────────────────────────────────────────────────────

/// The three-way SR-7.1 / SR-3.6 gate, DERIVED from a `Config` — the same
/// conjunction `src/main.rs` spells at its `BaseBootstrap::new` call site and
/// `helpers::base_bootstrap_for_harness` mirrors.
///
/// Never hardcoded at a call site. A literal `false` in Test 5 would keep
/// passing if production stopped gating on the tail flag, which is the exact
/// silent-green this file exists to prevent.
fn derive_bootstrap_gate(cfg: &Config) -> bool {
    cfg.checkpoint_enabled
        && !matches!(cfg.checkpoint_adapter, CheckpointAdapter::Disabled)
        && cfg.checkpoint_tail_enabled
}

/// Cancel the harness's periodic loop and join it under one bounded budget.
///
/// After the fence no tick exists to land a generation, so retention cannot
/// prune the fixture and no fresh base can appear at a sequence the assertions
/// do not know about. Every test in this file fences before it wipes.
async fn fence_loop(harness: &mut CheckpointHarness) {
    harness.cancel.cancel();
    let loop_task = std::mem::replace(&mut harness.loop_handle, tokio::spawn(async {}));
    let _ = tokio::time::timeout(LOOP_DRAIN_BUDGET, loop_task).await;
}

/// Identity of the layer under test.
struct LayerFixture {
    session_id: String,
    layer_name: String,
    object_id: u64,
    /// `snap_seq` of the newest generation in the store after the fence — the
    /// generation the restore will be seeded from, and therefore what the
    /// restore's own `last_checkpoint_seq` stamp must equal.
    head_snap_seq: u64,
}

/// CreateSession → CreateLayer → one inline commit → wait for the loop's base at
/// or above that commit → **fence the loop**.
///
/// Gating on the committed sequence rather than "any base" is load-bearing: with
/// `interval = quiesce = 0` the loop can fire between CreateLayer and the commit
/// and land a generation that predates it.
async fn plant_base_lineage(harness: &mut CheckpointHarness, name: &str) -> LayerFixture {
    let session_store =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
    let session_id = session_store
        .create(&format!("{name}-session"), SESSION_BASE_PATH, CREATED_BY)
        .await
        .expect("session creation must succeed");
    let layer_name = format!("{name}-layer");

    let ompf = delta_to_ompf_bytes(&initial_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, &layer_name),
        make_create_data(ompf),
    ]);
    let object_id = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed while building the fixture")
        .into_inner()
        .object_id;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );

    let (committed_seq, _) = drive_commit(
        harness,
        &session_id,
        &layer_name,
        object_id,
        &inline_field_delta("pre-base-field"),
        &[],
        1_000,
    )
    .await;

    let base = wait_for_checkpoint_at_least(
        &harness.store,
        object_id,
        committed_seq,
        LOOP_BASE_DEADLINE_SECS,
    )
    .await;

    fence_loop(harness).await;

    // After the fence the listing is frozen. Pinning the head here — rather than
    // assuming the base just waited for is still the newest — is what lets every
    // downstream cursor-column claim name one concrete generation.
    let handles = listing(harness, object_id).await;
    let head = handles
        .first()
        .expect("the fixture must leave at least one generation in the store")
        .clone();
    assert_eq!(
        head.kind,
        CheckpointKind::Base,
        "fixture precondition: with no concurrent writer the assembler wins every stable cut, so \
         the newest generation must be a Base. A Tail here means the fixture raced something it \
         does not model; got {handles:?}"
    );
    assert!(
        head.snap_seq >= committed_seq,
        "fixture precondition: the head generation ({}) must cover the fixture's committed \
         sequence ({committed_seq}), otherwise the restore replays state the test never wrote",
        head.snap_seq
    );
    assert_eq!(
        base.snap_seq, head.snap_seq,
        "fixture precondition: the base waited for must still be the head after the fence"
    );

    LayerFixture {
        session_id,
        layer_name,
        object_id,
        head_snap_seq: head.snap_seq,
    }
}

/// Issue one `UpdateLayer` through the real handler and return
/// `(sequence_num, the OMPF bytes it sent)`.
///
/// Returning the sent bytes is the point for Test 2: they are what
/// `put_tail_generation_from_records` encodes into the tail envelope, so the
/// artifact carries exactly what the live commit carried — including the blob as
/// OMPF Part 1, which is byte-for-byte what the capture path persists.
async fn drive_commit(
    harness: &CheckpointHarness,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    delta: &OwnedDelta,
    external_values: &[&[u8]],
    client_timestamp: u64,
) -> (u64, Bytes) {
    let ompf = delta_to_ompf_bytes(delta, external_values);
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
        .expect("UpdateLayer must succeed while building the fixture")
        .into_inner()
        .sequence_num;
    (sequence_num, Bytes::from(ompf))
}

/// The newest-first generation listing for `object_id`.
async fn listing(harness: &CheckpointHarness, object_id: u64) -> Vec<CheckpointHandle> {
    harness
        .store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed")
}

/// The three columns every re-base claim in this file reads, in ONE `HMGET` so
/// they cannot disagree about which write they describe.
struct CursorColumns {
    /// The live writer cursor — i.e. the restore's `seeded_seq`. READ, never
    /// recomputed: the guard band is production's to choose.
    seq: u64,
    last_checkpoint_seq: Option<u64>,
    restore_seq_bump: Option<u64>,
}

async fn read_cursor_columns(harness: &CheckpointHarness, object_id: u64) -> CursorColumns {
    let key = harness.tv.key_schema().obj_header(&object_id.to_string());
    let raw: Vec<Option<String>> = harness
        .tv
        .client()
        .pool
        .hmget(&key, vec!["seq", "last_checkpoint_seq", "restore_seq_bump"])
        .await
        .expect("HMGET obj:hdr cursor columns must succeed");
    let parse = |slot: usize, field: &str| -> Option<u64> {
        raw.get(slot).cloned().flatten().map(|v| {
            v.parse::<u64>()
                .unwrap_or_else(|e| panic!("obj:hdr `{field}` MUST parse as u64; got {v:?}: {e}"))
        })
    };
    CursorColumns {
        seq: parse(0, "seq").expect("obj:hdr `seq` must be present after a restore"),
        last_checkpoint_seq: parse(1, "last_checkpoint_seq"),
        restore_seq_bump: parse(2, "restore_seq_bump"),
    }
}

/// `EXISTS` on a fully-qualified key.
async fn key_exists(harness: &CheckpointHarness, key: &str) -> bool {
    harness
        .tv
        .client()
        .pool
        .exists(key)
        .await
        .expect("EXISTS must succeed against the live test Valkey")
}

/// Wipe the object's materialized state and assert the wipe took.
///
/// The assertion is not ceremony. `try_restore_layer` short-circuits to
/// `Restored` **without restoring anything** when the layer is already present
/// (`recovery.rs`, the post-admission re-check), so a wipe that silently did
/// nothing would make every `RecoveryOutcome::Restored` assertion below vacuous.
async fn wipe_and_assert_absent(harness: &CheckpointHarness, object_id: u64) {
    wipe_layer_state(&harness.tv, object_id).await;
    let header_key = harness.tv.key_schema().obj_header(&object_id.to_string());
    assert!(
        !key_exists(harness, &header_key).await,
        "wipe precondition: `{header_key}` MUST be absent after wipe_layer_state — \
         try_restore_layer returns Restored without restoring when the layer is still present, so \
         every outcome assertion downstream would be vacuous"
    );
}

/// Drive the restore through the harness's OWN hook and assert it restored.
///
/// `harness.svc.recovery` — no field swap, no locally-rebuilt hook. That is the
/// subject: this file asserts on what the stock harness hands you.
/// `try_restore_layer` is called directly rather than through a handler because
/// it returns the `RecoveryOutcome` itself, which is a stronger and more direct
/// claim than a metric delta derived from it.
async fn restore_through_stock_hook(harness: &CheckpointHarness, object_id: u64) {
    let outcome = harness.svc.recovery.try_restore_layer(object_id).await;
    assert!(
        matches!(outcome, RecoveryOutcome::Restored),
        "the STOCK harness's own recovery hook must restore this lineage; got {outcome:?}. An \
         `Error(..)` carrying `no ExternalValueStore was wired` means \
         `helpers::recovery_hook_for_e2e` lost its `.with_ext_value_store(..)` call — the b.kfg \
         defect, which this module exists to fence"
    );
}

// =============================================================================
// Test 1 — one BaseBootstrap unit, shared
// =============================================================================

/// The structural half of the single-unit claim: the harness constructs exactly
/// ONE [`BaseBootstrap`] and the service holds that very `Arc`.
///
/// **Why identity and not merely "both are wired".** The unit's in-flight
/// admission bound `BOOTSTRAP_MAX_INFLIGHT` is a field ON the unit
/// (`RecoveryHook::with_base_bootstrap`'s rustdoc, `src/checkpoint/recovery.rs`),
/// so a second instance built for the second call site would be a second,
/// independent bound — the fleet-wide "at most N concurrent bootstrap writes"
/// property would be silently untrue while every functional test stayed green.
/// `bootstrap_concurrency_bound_test` pins the bound itself; this pins that
/// there is one unit for it to be a bound ON.
///
/// **This is the structural HALF.** The hook stores its copy in a private field
/// with no accessor and this Task forbids production-code changes, so the hook's
/// side of the identity is unreachable from here. It is carried behaviourally by
/// Tests 3 and 5: a re-base that lands (3) proves the hook holds a bootstrap at
/// all, and a re-base that correctly does NOT land under the harness's own
/// derived gate (5) proves the one it holds reads the harness's `Config` rather
/// than a re-derived one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stock_harness_shares_one_base_bootstrap_between_service_and_hook() {
    let harness = setup_checkpoint_harness(TICK_SECS).await;

    assert!(
        Arc::ptr_eq(&harness.svc.base_bootstrap, &harness.base_bootstrap),
        "the harness must construct exactly ONE BaseBootstrap and clone that same `Arc` into the \
         service. Two instances would be two independent BOOTSTRAP_MAX_INFLIGHT bounds — a wiring \
         defect no functional assertion in this module could see"
    );

    // The default posture, pinned: production calls `.with_base_bootstrap(..)`
    // UNCONDITIONALLY and lets the unit's own three-way gate decide whether the
    // spawn does anything. A harness defaulting to the opt-out would ship a hook
    // production does not — the b.kfg shape, re-introduced through the fix.
    assert!(
        harness.wire_restore_rebase,
        "the stock harness must default to WIRING the re-base (the production posture); the \
         opt-out is opt-IN"
    );
    assert!(
        CheckpointHarnessOptions::new(TICK_SECS).wire_restore_rebase,
        "`CheckpointHarnessOptions::new` must carry the same default — a harness constructed \
         through the options struct is the same harness"
    );
}

// =============================================================================
// Test 2 — the hook carries the external-value store
// =============================================================================

/// A composed restore through the STOCK hook materialises the lineage's
/// tail-carried external value in Valkey.
///
/// This is the behavioural fence on `.with_ext_value_store(..)`. Under the
/// pre-fold wiring the hook carried `ext_value_store: None` and this restore
/// **fails closed** at the apply core's `Permanent` arm (`src/checkpoint/apply.rs`,
/// step 4b: "no ExternalValueStore was wired") — loud, but only for a lineage
/// that actually carries values, which is why no pre-existing stock-harness test
/// saw it.
///
/// **The load-bearing assertion is `le:ext:data:{hash}`**, not the outcome:
/// `RecoveryOutcome::Restored` alone would also be produced by a base-only
/// lineage carrying no values at all. Two guards keep it honest:
///
/// - the ext store is built WITHOUT `.with_checkpoint_store(..)`, so the SR-3.2
///   lazy fetch-on-miss route is unarmed and the SR-3.4 eager pre-store is the
///   ONLY way those bytes can reach Valkey;
/// - the fixture's own live-commit residue is wiped by explicit hash before the
///   restore, so the presence assertion cannot pass on it.
///
/// The store is installed through `CheckpointHarnessOptions::ext_value_store_factory`
/// — the seam that resolves the ordering constraint (a `ValkeyExternalValueStore`
/// needs the harness's own `TestValkey`, which does not exist until the harness
/// does) that previously forced four files to field-swap the store in afterwards
/// and leave the hook pointing at a different one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stock_harness_hook_carries_the_ext_value_store() {
    let mut opts = CheckpointHarnessOptions::new(TICK_SECS);
    opts.ext_value_store_factory = Some(Box::new(|ctx: ExtStoreCtx<'_>| {
        // `.with_metrics(..)` because production does; deliberately NOT
        // `.with_checkpoint_store(..)` — see the doc-comment above.
        Arc::new(
            ValkeyExternalValueStore::new(ctx.tv.client().clone(), ctx.tv.key_schema().clone())
                .with_metrics(Arc::clone(ctx.metrics)),
        ) as Arc<dyn ExternalValueStore>
    }));
    let mut harness = setup_checkpoint_harness_with_options(opts).await;

    let fixture = plant_base_lineage(&mut harness, "ext-parity").await;

    // ── The replayed range: one commit carrying an external value ────────
    let blob = ext_blob();
    let hash = ext_content_hash(&blob);
    let hash_hex = sha256_to_hex(&hash);
    let record = drive_commit(
        &harness,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.object_id,
        &tail_ext_delta(&hash, blob.len()),
        &[&blob],
        2_000,
    )
    .await;
    assert_eq!(
        record.0,
        fixture.head_snap_seq + 1,
        "fixture precondition: the tail-range commit must land at head + 1; a gap would be \
         rejected as a RecordSeqGap and the restore would step down to the base, where no \
         external value travels at all"
    );

    put_tail_generation_from_records(
        &harness.store,
        fixture.object_id,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.head_snap_seq,
        fixture.head_snap_seq,
        std::slice::from_ref(&record),
        HAND_PUT_WRITTEN_AT_UNIX_MS,
        0,
    )
    .await;

    // The live UpdateLayer pre-flighted the value through `store_external_values`,
    // so both content-addressed keys exist right now. Wipe them by explicit hash:
    // otherwise the post-restore presence assertion passes on this residue and
    // proves nothing.
    let data_key = harness.tv.key_schema().ext_data(&hash_hex);
    let meta_key = harness.tv.key_schema().ext_meta(&hash_hex);
    assert!(
        key_exists(&harness, &data_key).await,
        "fixture precondition: the live commit must have stored `{data_key}`"
    );
    wipe_and_assert_absent(&harness, fixture.object_id).await;
    wipe_ext_namespace_for_hashes(&harness.tv, std::slice::from_ref(&hash_hex)).await;
    assert!(
        !key_exists(&harness, &data_key).await,
        "wipe precondition: `{data_key}` MUST be absent before the restore"
    );
    assert!(
        !key_exists(&harness, &meta_key).await,
        "wipe precondition: `{meta_key}` MUST be absent before the restore"
    );

    // ── The restore, through the harness's own hook ──────────────────────
    let composed_pre = read_restore_lineage_counter(&harness.metrics, "gathered_composed");
    restore_through_stock_hook(&harness, fixture.object_id).await;
    assert_eq!(
        read_restore_lineage_counter(&harness.metrics, "gathered_composed") - composed_pre,
        1,
        "the lineage MUST resolve COMPOSED. A base-only restore carries no external values at all \
         (the assembler emits zero ext parts), so the presence claim below would be testing \
         nothing on one"
    );

    // ── The claim ────────────────────────────────────────────────────────
    assert!(
        key_exists(&harness, &data_key).await,
        "the composed restore MUST have pre-stored the tail-carried external value: `{data_key}` \
         is absent. With no `.with_checkpoint_store(..)` on this store there is no lazy \
         fetch-on-miss route, so the SR-3.4 pre-store through the HOOK's ext store is the only \
         thing that can create this key"
    );
    assert!(
        key_exists(&harness, &meta_key).await,
        "`{meta_key}` must be written alongside the data key — `store_values` writes the pair"
    );
}

// =============================================================================
// Test 3 — the hook carries the base bootstrap
// =============================================================================

/// A restore through the STOCK harness — at the binary tail default, which is
/// `true` since b.3n1 — spawns the SR-3.6 re-base.
///
/// The flag is INHERITED here, not set: this arm's subject is what a stock
/// deployment does, so pinning it `true` would weaken the claim to "a harness
/// someone configured on". Test 5 is the pinned-off twin.
///
/// This is the behavioural fence on `.with_base_bootstrap(..)`, and the one that
/// matters most: a missing `with_ext_value_store` at least fails loudly on an
/// ext-carrying lineage, whereas a missing `with_base_bootstrap` is **silent** —
/// no re-base is spawned, and every test asserting anything other than the
/// re-base outcome stays green while exercising non-production wiring.
///
/// The claim is the landing shape `restore_rebase_test` owns in full, reduced to
/// what a wiring fence needs: a `Base` at the restore's own seeded cursor, and
/// the two cursor columns that only the re-base moves. `seeded_seq` is READ off
/// `obj:hdr`, never recomputed — the guard band is production's to choose, and
/// re-deriving it here would make this test a restatement of the arithmetic
/// rather than an observation of the write.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stock_harness_at_the_default_tail_flag_spawns_the_restore_rebase() {
    let mut harness = setup_checkpoint_harness(TICK_SECS).await;

    assert!(
        derive_bootstrap_gate(&harness.config),
        "at the BINARY tail default the three-way gate must resolve TRUE, so a stock restore \
         re-bases. Derived from `harness.config`, never spelled as a literal. A FALSE here means \
         either the default flipped back or `CheckpointHarnessOptions::tail` stopped tracking it — \
         and the assertions below would then be asserting against a no-op"
    );
    assert!(
        harness.wire_restore_rebase,
        "fixture precondition: this arm runs the DEFAULT wiring posture, not the opt-out"
    );

    let fixture = plant_base_lineage(&mut harness, "rebase-parity").await;
    wipe_and_assert_absent(&harness, fixture.object_id).await;

    restore_through_stock_hook(&harness, fixture.object_id).await;

    // The re-base is a fire-and-forget task on the shared TaskTracker that the
    // hook never awaits. Tracker quiescence — not a timer — is what makes it
    // deterministically observable.
    wait_for_writer_idle(
        &harness.session_checkpoint_tracker,
        TRACKER_IDLE_DEADLINE_SECS,
    )
    .await;

    let columns = read_cursor_columns(&harness, fixture.object_id).await;
    let seeded_seq = columns.seq;
    assert!(
        seeded_seq > fixture.head_snap_seq,
        "every restore seeds `seq` strictly above the generation it restored (band >= 1 \
         unconditionally, post-b.8y5). seeded_seq = {seeded_seq}, restored generation = {}. An \
         equality here would make every claim below vacuous",
        fixture.head_snap_seq
    );

    let handles = listing(&harness, fixture.object_id).await;
    assert!(
        handles
            .iter()
            .any(|h| h.snap_seq == seeded_seq && h.kind == CheckpointKind::Base),
        "a Base generation MUST exist at seeded_seq = {seeded_seq} — that is the SR-3.6 re-base \
         closing the gap the restore opened. Its absence is the b.kfg defect: the harness's hook \
         was built without `.with_base_bootstrap(..)`, so nothing was ever spawned. Got {handles:?}"
    );
    assert_eq!(
        columns.last_checkpoint_seq,
        Some(seeded_seq),
        "the re-base must advance last_checkpoint_seq to the seeded cursor (the restore itself \
         stamped {}, the generation it read)",
        fixture.head_snap_seq
    );
    assert_eq!(
        columns.restore_seq_bump,
        Some(RETIRED_BUMP),
        "the re-base must RETIRE restore_seq_bump to {RETIRED_BUMP} — that is what unblocks tail \
         capture, which refuses `PostRestoreUnrebased` while the column is non-zero"
    );
}

// =============================================================================
// Test 4 — the documented opt-out stays honest
// =============================================================================

/// `wire_restore_rebase = false` spawns nothing, and the RESTORE's own cursor
/// stamp survives into the assertion phase.
///
/// The opt-out exists for exactly one landed consumer shape —
/// `e2e_full_wipe_composed_restore_test` Arm 1, whose claim is
/// `last_checkpoint_seq == S`, what the RESTORE stamped, which a successful
/// re-base overwrites with `seeded_seq`. Expressing the fence HERE rather than
/// there is deliberate: that file could change its arms, or drop the opt-out for
/// an unrelated reason, and the posture would be left untested. This test is
/// what keeps the opt-out meaningful independently of any consumer.
///
/// Note the harness is otherwise IDENTICAL to Test 3's — tail flag on, gate
/// TRUE. That is what makes the pair a controlled experiment: the only variable
/// between "re-base lands" and "re-base does not" is the one builder call.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn wire_restore_rebase_false_spawns_nothing() {
    let mut opts = CheckpointHarnessOptions::new(TICK_SECS);
    opts.tail.enabled = true;
    opts.wire_restore_rebase = false;
    let mut harness = setup_checkpoint_harness_with_options(opts).await;

    assert!(
        derive_bootstrap_gate(&harness.config),
        "fixture precondition: the three-way gate must resolve TRUE, so that the ONLY reason no \
         re-base lands is the opt-out. With the gate false this test would pass for the wrong \
         reason — that is Test 5's claim, not this one"
    );
    assert!(
        !harness.wire_restore_rebase,
        "fixture precondition: the harness must record the opt-out it was built with"
    );

    let fixture = plant_base_lineage(&mut harness, "optout-parity").await;
    wipe_and_assert_absent(&harness, fixture.object_id).await;

    restore_through_stock_hook(&harness, fixture.object_id).await;

    // Same drain as Test 3: after the tracker is empty, a re-base that HAD been
    // spawned would already have completed and be visible below. This is what
    // makes the negative claim sound rather than merely early.
    wait_for_writer_idle(
        &harness.session_checkpoint_tracker,
        TRACKER_IDLE_DEADLINE_SECS,
    )
    .await;

    let columns = read_cursor_columns(&harness, fixture.object_id).await;
    let seeded_seq = columns.seq;
    assert!(
        seeded_seq > fixture.head_snap_seq,
        "non-vacuity: the restore must have opened a gap ({seeded_seq} > {}), so that a re-base \
         had something to close and its absence is a real observation",
        fixture.head_snap_seq
    );

    assert_eq!(
        columns.last_checkpoint_seq,
        Some(fixture.head_snap_seq),
        "the RESTORE's own stamp must survive: `last_checkpoint_seq` is the generation the restore \
         read ({}), NOT the seeded cursor ({seeded_seq}). Seeing the latter means a re-base ran \
         despite the opt-out",
        fixture.head_snap_seq
    );
    let bump = columns
        .restore_seq_bump
        .expect("a restore always stamps restore_seq_bump; its absence means nothing restored");
    assert_ne!(
        bump, RETIRED_BUMP,
        "the restore's own guard band must still be recorded — a retired bump is the re-base's \
         signature and must not appear under the opt-out"
    );

    let handles = listing(&harness, fixture.object_id).await;
    assert!(
        !handles
            .iter()
            .any(|h| h.snap_seq == seeded_seq && h.kind == CheckpointKind::Base),
        "no generation may exist at seeded_seq = {seeded_seq} under the opt-out; got {handles:?}"
    );
}

// =============================================================================
// Test 5 — the three-way gate, derived
// =============================================================================

/// With the tail flag PINNED OFF — the operator opt-out — the three-way gate is
/// `false`, so a restore through the fully-wired hook leaves `restore_seq_bump`
/// set and writes no base.
///
/// This is the structural claim every tails-off suite in this module rests on.
/// The fold wires `.with_base_bootstrap(..)` into every harness, which is only
/// safe because the unit's own gate — not the wiring — decides whether the spawn
/// does anything. If that stopped being true, every suite that pins the flag off
/// to keep its pre-tail numbers would start observing a re-base's restamp.
///
/// The flag is pinned rather than inherited: since b.3n1 the binary default is
/// `true` (Test 3 is what pins that), so an inherited value would make this arm
/// assert the opposite of its own name.
///
/// **The gate is DERIVED from `harness.config`, never hardcoded.** A literal
/// `false` would keep passing if production stopped gating on the tail flag,
/// which is precisely the regression this assertion exists to catch. Same
/// pattern as `e2e_full_wipe_composed_restore_test`'s own fixture precondition.
///
/// Note the contrast with Test 4: there the bootstrap is absent from the hook,
/// here it is PRESENT and declines. Both produce "no re-base"; only one of them
/// is how a tail-flag-off deployment is expressed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tail_pinned_off_harness_restore_spawns_no_rebase() {
    let mut harness = setup_checkpoint_harness_tails_off(TICK_SECS).await;

    assert!(
        !derive_bootstrap_gate(&harness.config),
        "fixture precondition: with the tail flag pinned off the three-way gate must resolve \
         FALSE. Derived from `harness.config`, never spelled as a literal — that is the whole \
         point of this test"
    );
    assert!(
        harness.wire_restore_rebase,
        "fixture precondition: the bootstrap IS wired into the hook here. The claim is that the \
         GATE declines, not that the collaborator is missing — that distinction is what separates \
         this test from Test 4"
    );

    let fixture = plant_base_lineage(&mut harness, "gate-parity").await;
    wipe_and_assert_absent(&harness, fixture.object_id).await;

    restore_through_stock_hook(&harness, fixture.object_id).await;
    wait_for_writer_idle(
        &harness.session_checkpoint_tracker,
        TRACKER_IDLE_DEADLINE_SECS,
    )
    .await;

    let columns = read_cursor_columns(&harness, fixture.object_id).await;
    let seeded_seq = columns.seq;
    assert!(
        seeded_seq > fixture.head_snap_seq,
        "non-vacuity: the restore must have opened a gap ({seeded_seq} > {})",
        fixture.head_snap_seq
    );
    let bump = columns
        .restore_seq_bump
        .expect("a restore always stamps restore_seq_bump; its absence means nothing restored");
    assert_ne!(
        bump, RETIRED_BUMP,
        "with the gate FALSE the restore's own guard band must survive: a retired bump would mean \
         a re-base ran with the tail feature pinned off"
    );
    assert_eq!(
        columns.last_checkpoint_seq,
        Some(fixture.head_snap_seq),
        "with the gate FALSE `last_checkpoint_seq` stays at the generation the restore read"
    );

    let handles = listing(&harness, fixture.object_id).await;
    assert!(
        !handles
            .iter()
            .any(|h| h.snap_seq == seeded_seq && h.kind == CheckpointKind::Base),
        "no generation may exist at seeded_seq = {seeded_seq} with the gate false; got {handles:?}"
    );
}

// =============================================================================
// Test 6 — the SECOND entry point into the same wiring
// =============================================================================

/// A hook rebuilt through `CheckpointHarness::install_recovery_hook` still
/// carries the collaborators, so the re-base still lands.
///
/// **Beyond this Subtask's five enumerated tests, and here on purpose.**
/// `install_recovery_hook` is the second way a consumer obtains a harness hook —
/// the seam that exists so a test needing `with_restore_fault_injector_for_test`
/// chains it instead of forking a `RecoveryHook::new(..)`, which is how b.kfg
/// reached six copies. It re-derives the wiring from the harness's own pieces,
/// which means it can re-derive it WRONG, and in exactly the silent direction:
/// a rebuilt hook that quietly loses the bootstrap spawns no re-base, and the
/// consumer chaining a fault injector is asserting about something else
/// entirely. Tests 1-5 all observe the hook the harness *constructed*; none of
/// them would notice.
///
/// The closure is the identity function deliberately: the subject is what
/// `install_recovery_hook` threads, not what a caller chains onto it. A
/// rebuilt-and-unmodified hook must be indistinguishable from the original, so
/// this is Test 3's assertion re-run through the second entry point.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn install_recovery_hook_rebuild_preserves_the_wiring_and_the_posture() {
    let mut opts = CheckpointHarnessOptions::new(TICK_SECS);
    opts.tail.enabled = true;
    let mut harness = setup_checkpoint_harness_with_options(opts).await;

    assert!(
        derive_bootstrap_gate(&harness.config),
        "fixture precondition: the three-way gate must resolve TRUE, otherwise a rebuilt hook that \
         dropped the bootstrap would be indistinguishable from a correct one"
    );

    let fixture = plant_base_lineage(&mut harness, "rebuild-parity").await;
    wipe_and_assert_absent(&harness, fixture.object_id).await;

    // The rebuild under test. `install_recovery_hook` re-derives store,
    // layer_store, config, metrics, the ONE ext store and the ONE bootstrap
    // under the harness's own `wire_restore_rebase` posture.
    harness.install_recovery_hook(|hook| hook);

    restore_through_stock_hook(&harness, fixture.object_id).await;
    wait_for_writer_idle(
        &harness.session_checkpoint_tracker,
        TRACKER_IDLE_DEADLINE_SECS,
    )
    .await;

    let columns = read_cursor_columns(&harness, fixture.object_id).await;
    let seeded_seq = columns.seq;
    assert!(
        seeded_seq > fixture.head_snap_seq,
        "non-vacuity: the restore must have opened a gap ({seeded_seq} > {})",
        fixture.head_snap_seq
    );

    let handles = listing(&harness, fixture.object_id).await;
    assert!(
        handles
            .iter()
            .any(|h| h.snap_seq == seeded_seq && h.kind == CheckpointKind::Base),
        "the REBUILT hook must still re-base: a Base generation MUST exist at seeded_seq = \
         {seeded_seq}. Its absence means `install_recovery_hook` re-derived the wiring without the \
         bootstrap — the b.kfg defect reintroduced through the seam that exists to prevent it. Got \
         {handles:?}"
    );
    assert_eq!(
        columns.restore_seq_bump,
        Some(RETIRED_BUMP),
        "the rebuilt hook's re-base must retire restore_seq_bump exactly as the original's does"
    );
    assert_eq!(
        columns.last_checkpoint_seq,
        Some(seeded_seq),
        "the rebuilt hook's re-base must advance last_checkpoint_seq to the seeded cursor"
    );
}
