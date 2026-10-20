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

//! End-to-end recovery tests pinning Epic AC #9 — `apply_checkpoint`
//! preserves `object_id` across recovery and stamps the SR-2.4 / SR-2.5
//! header invariants (`last_checkpoint_seq = snap_seq`, sequence-number
//! advance to `max(snap_seq, max_seq_observed.unwrap_or(0)) +
//! checkpoint_seq_safety_margin`).
//!
//! Subtask `t3.fgj.ya.wg.4q` under Task `t2.fgj.ya.wg`, Epic `t1.fgj.ya`,
//! Bee `b.fgj`.
//!
//! The three invariants this file pins (in order of test discovery):
//!
//! 1. **object_id preservation** — a CreateLayer that mints `object_id_A`
//!    followed by a write → checkpoint → wipe → recovery cycle MUST
//!    re-materialize the layer state under the SAME `object_id_A`. A
//!    regression that re-minted a fresh `object_id` would surface here as
//!    a mismatched `obj:hdr.object_id` field or a mapping that no longer
//!    resolves to the captured pre-wipe `object_id`. SDD §SR-2.4.
//!
//! 2. **last_checkpoint_seq == snap_seq (NOT seeded_seq)** — apply step 6
//!    HSETs `last_checkpoint_seq` to the snapshot's authoritative
//!    `snap_seq`, NOT to the safety-margin-bumped `seeded_seq` that the
//!    writer cursor was advanced to. Operators reading
//!    `HGET le:{oid}:obj:hdr last_checkpoint_seq` need to see which
//!    checkpoint GENERATION produced the live state, so the field is the
//!    audit-trail pointer back to the checkpoint that seeded recovery —
//!    not a derived writer-cursor value. SDD §SR-2.5 step 6 and the
//!    apply.rs step-6 comment block.
//!
//! 3. **seq advances to restore_floor + safety_margin** — `seeded_seq =
//!    max(snap_seq, max_seq_observed.unwrap_or(0)) + safety_margin`. The
//!    margin is an ADDITIVE guard band above the highest sequence number
//!    the restore can prove was issued, at every input — including
//!    `snap_seq > margin` (b.8y5). When the per-object stream did not
//!    survive the wipe, `max_seq_observed` is `None` and the floor is
//!    `snap_seq` alone.
//!
//! 4. **the guard band is real for a long-lived layer** (b.8y5) — test 4
//!    drives a restore with `snap_seq > margin`, the steady state for any
//!    layer past its first `margin` commits, and asserts the post-recovery
//!    `seq` still strictly exceeds the pre-wipe max. The pre-fix
//!    `max(snap_seq, observed + margin)` form resumed at exactly `snap_seq`
//!    there.
//!
//! 5. **a surviving per-object stream raises the floor** (b.8y5) — tests 5
//!    and 6 wipe `obj:*` but SPARE `le:{oid}:obj:stream`, so the recovery
//!    hook's high-water probe returns `Some` and the seeded seq clears the
//!    stream's `last-generated-id`. Test 5 then drives a real
//!    `UpdateLayer` on top: it is the only post-recovery commit in this
//!    suite that runs against a stream which survived, and therefore the
//!    only one that can catch the stale-stream-top wedge the probe exists
//!    to prevent. Since b.ph9 that wedge surfaces as a refused commit
//!    (`ServiceError::Internal` prefixed `update_layer stream-id
//!    collision:`, from Valkey's own `ERR The ID specified in XADD is
//!    equal or smaller than the target stream top item`) rather than as a
//!    half-commit, so a mis-computed floor fails this test loudly instead
//!    of silently advancing the header with no stream entry to match.
//!    Test 6 additionally pins that the probe reads
//!    `last-generated-id` and not the `XREVRANGE` tail — those diverge
//!    after an `XDEL` of the top entry, and only the former bounds what
//!    the next `XADD` will accept. SDD §SR-2.5 step 4.
//!
//!    These tests are deliberately blind to HOW the probe reads those two
//!    scalars. It is a server-side Lua projection today (the client-side
//!    `XINFO STREAM` form would have pulled up to two full stream entries
//!    — bounded only by `LIVE_EDIT_MAX_LAYER_SIZE` — across the wire to
//!    read a ~15-byte ID); the assertions are on the restored header, so
//!    they hold across that implementation change.
//!
//! 6. **the probe refuses a foreign high-water, and removes the stream
//!    that refusal would otherwise corrupt** (b.8y5) — the per-object
//!    stream is not written exclusively by the two commit paths, so not
//!    every `last-generated-id` on it is a sequence number. Tests 7–8
//!    cover the single, cause-agnostic plausibility gate that rejects the
//!    ones that are not: the `payload=deleted` tombstone the delete /
//!    overwrite paths append with an auto-ID, whose wall-clock ID lands
//!    three orders of magnitude above the bound (test 7, production's own
//!    shape), and a timestamp-shaped ID whose tombstone has since been
//!    trimmed away, leaving no sentinel to recognize it by (test 8). Both
//!    fall back to `snap_seq` alone. Without the gate a tombstoned stream
//!    would seed the restored layer's `sequence_num` from a wall clock —
//!    ~1.79e12, silently under the `2^53 − 1` ceiling that would otherwise
//!    have caught it.
//!
//!    Falling back to `snap_seq` seeds BELOW the surviving stream top,
//!    which would **half-commit** every subsequent `UpdateLayer`: the
//!    fast-path script writes fields/sections and `HSET`s the header
//!    before it `XADD`s at `{new_seq}-0`, and a mid-script failure rolls
//!    back nothing — so the state mutation and header advance land, the
//!    client is told the write failed, and live subscribers (fed by the
//!    stream) never see the commit at all. The gate therefore `UNLINK`s
//!    the stale stream BEFORE the restore materializes, and each of tests
//!    7–8 asserts the key is gone outright (`PTTL = -2`) and then drives a
//!    real `UpdateLayer` that must SUCCEED, recreating the stream from the
//!    seeded sequence. Test 5 carries the negative control: a probe that
//!    SUCCEEDED must leave the live stream untouched (`PTTL = -1`).
//!
//! 7. **the probe survives a flushed script cache** (b.8y5) — test 9
//!    re-runs test 5's success arm after a `SCRIPT FLUSH`, standing in for
//!    the Valkey restart that typically caused the state loss recovery is
//!    responding to. Needed as its own test because a NOSCRIPT regression
//!    used to be **silent**: pre-b.4et the probe folded every error to
//!    `None`, which is the same `snap_seq` fallback most of this file
//!    already expects. Post-b.4et a NOSCRIPT on a surviving stream refuses
//!    the restore instead, so the regression is loud too — but this test
//!    still pins the value the reload actually produces, which no other
//!    test does.
//!
//! 8. **the probe is FAIL-CLOSED, not best-effort** (b.4et) — tests 10–19.
//!    Pre-b.4et every unreadable outcome (`Err` from the `EVALSHA`, and the
//!    `Ok(None)` the script also answers for a wrong-type key) folded to
//!    `None` and the restore proceeded, seeding `obj:hdr` at
//!    `snap_seq + band` — potentially far BELOW a stream top nobody ever
//!    read. Post-b.ph9 that is not corruption but a full write outage on
//!    the restored layer, discovered only by the first client to try a
//!    commit. b.4et un-conflates the outcomes with one `TYPE` round-trip
//!    and refuses the restore only on the arm that has positive evidence of
//!    a surviving, unreadable stream:
//!
//!    - probe unreadable + `TYPE` says `stream` → **REFUSE** (test 10),
//!      after a bounded retry that a merely transient blip survives
//!      (test 11);
//!    - probe unreadable + `TYPE` says `none` → proceed at `snap_seq`,
//!      byte-identical to pre-b.4et. This is the shape a full keyspace
//!      loss produces, i.e. the one DR actually runs on, and test 12 is the
//!      fence that keeps the fix from taking DR offline on it;
//!    - probe unreadable + `TYPE` says anything else → proceed, and report
//!      the foreign key rather than deleting it (test 13);
//!    - the plausibility gate's own `UNLINK` is now VERIFIED, and refuses
//!      the restore when it cannot be (test 14) — and does not run at all
//!      when the SR-2.4 identity guard is about to refuse the restore
//!      anyway (test 15);
//!    - every refusal first re-checks `obj:hdr`, so a peer replica that
//!      landed the restore while we were retrying turns the refusal back
//!      into `Restored` (test 16);
//!    - the `TYPE` round-trip that un-conflates all of the above can itself
//!      fail, and does not get to fold back to "no stream" one step later.
//!      It is reached from two places and reports a DIFFERENT reason on
//!      each: `probe_unreadable` from the classification (test 17),
//!      `unlink_unverified` from the post-`UNLINK` confirmation (test 18);
//!    - an `UNLINK` that reports deleting nothing is nonetheless a VERIFIED
//!      removal when the confirming `TYPE` reads `none` — a peer replica
//!      legitimately cleared the key first — so the restore PROCEEDS
//!      (test 19). Test 18 is the same staged race with the confirmation
//!      taken away; the pair is what separates "the key is provably gone"
//!      from "we could not look".
//!
//! 9. **the adjudication has a wall-clock ceiling, not just an attempt count**
//!    (b.4et) — tests 20–21. The retry budget bounds *backoff sleep*; a store
//!    that accepts a command and never answers costs a full command timeout per
//!    attempt, so the whole sequence additionally runs under one
//!    `RECOVERY_STREAM_ADJUDICATION_BUDGET` (10 s) deadline, with a
//!    `RECOVERY_STREAM_ADJUDICATION_PEER_GRACE` (1 s) for the peer-landed
//!    re-check on the expiry arm. Test 20 drives the expiry itself and pins
//!    that the refusal ticks `probe_unreadable` EXACTLY once — the expiry arm
//!    owns the tick, and a double-tick (its own plus `refuse_restore`'s) is the
//!    hazard the cancel-safety argument in `probe_stream_high_water` turns on.
//!    Test 21 is the grace path: a peer lands the restore during the stall, the
//!    over-budget re-check sees it, and the outcome is `Restored` with nothing
//!    ticked at all.
//!
//!    **Both cost ≈11 s of wall clock each** (one deliberate stall past the
//!    budget, driven through the probe seam). That is well inside nextest's 40 s
//!    slow mark, so no `.config/nextest.toml` override is needed — but it is why
//!    the pair is two tests and not six: the arms that do NOT need the timeout
//!    to fire are covered above at millisecond cost.
//!
//! 10. **the identity gate's PERMANENT exit, and the pre-`UNLINK`
//!     peer-presence re-read** (b.h4m) — tests 22–25.
//!
//!     Test 22 is the identity gate's third exit, which nothing above covers:
//!     a WRONG-TYPED layer mapping. It rides the retry driver's `Ok` channel
//!     (`Ok(Err(permanent))`) so it reaches the caller intact and propagates to
//!     `INTERNAL`, ticking NOTHING on the unresolved family. Pre-fix it folded
//!     into the gate's `Err` arm and was refused as a retriable
//!     `unlink_unverified`, so the handlers' retry-once dispatch re-drove a
//!     restore that could never succeed. `restore_permanent_refusal_test.rs`'s
//!     `wrong_typed_layer_mapping_refuses_permanently` covers the SAME key at
//!     `apply_restore`'s step-2 guard, a different and later gate; neither test
//!     can see the other's arm.
//!
//!     Tests 23–25 cover the re-read itself — the structural `HGETALL obj:hdr`
//!     issued immediately before the removal, closing the window between the
//!     evidence (gathered pre-blob-fetch) and the destructive act. One test per
//!     exit: a peer's complete header SKIPS the removal and the apply reports
//!     the layer restored (23); a transient failure is absorbed by the same
//!     bounded retry as its neighbours and the removal proceeds (24); an
//!     exhausted budget refuses as `unlink_unverified` with the stream
//!     untouched (25).
//!
//! Recovery trigger: every test drives recovery through `handle_stat_layer` —
//! the cheapest handler-level recovery trigger in the wired set. It does not
//! require encoding an UpdateLayer body, it does not allocate a streaming
//! channel, and it does not depend on the engineer's concurrent
//! UpdateLayer recovery-wiring patch (sibling subtask). The recovery
//! contract under test is "apply_checkpoint preserves invariants" —
//! which handler tripped the recovery branch is incidental.
//!
//! The two exceptions are tests 16 and 21, the peer-landed arms, which call
//! `RecoveryHook::try_restore_layer` directly: their claim is about the
//! `RecoveryOutcome` variant, and the handler folds `Restored` and a plain
//! header hit into one indistinguishable `Ok`.
//!
//! Tail flag: every harness in this file is built through
//! [`setup_checkpoint_harness_tails_off`] /
//! [`setup_checkpoint_harness_tails_off_with_seq_margin`], i.e. with
//! `LIVE_EDIT_CHECKPOINT_TAIL_ENABLED` pinned explicitly OFF rather than
//! inherited from the binary default (which is `true` since b.3n1). The subject
//! here is the SR-2.5 step-4 restore FLOOR — `seeded_seq` and
//! `last_checkpoint_seq` asserted to the unit against a hand-derived
//! `floor + band` — and a tails-on restore additionally spawns the SR-3.6
//! post-restore re-base, which commits once more and restamps both columns.
//! Absorbing that into re-derived constants would quietly change every
//! assertion below from "the floor is computed correctly" to "the floor plus
//! whatever the re-base did". The re-base's own behaviour is
//! `harness_parity_test` / `restore_rebase_test` / `e2e_rebase_lifecycle_test`.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`. Under
//! the project sandbox the URL must include the password and
//! `host.docker.internal` host; see CLAUDE.md.

use std::time::Duration;

use fred::interfaces::{HashesInterface, KeysInterface, LuaInterface, StreamsInterface};
// `ClientLike::is_clustered` is consumed only by `assert_restore_wrote_nothing`,
// which is itself gated on the fault-injection feature — an unconditional
// import would be dead on the release-shape clippy pass.
#[cfg(feature = "test-checkpoint-fault-injection")]
use fred::interfaces::ClientLike;
use fred::types::Value;
use tonic::Request;

use live_edit::checkpoint::RecoveryOutcome;
use live_edit::grpc::layer::{handle_create_layer, handle_stat_layer, handle_update_layer};
use live_edit::grpc::proto::StatLayerRequest;
use live_edit::grpc::BearerToken;
use live_edit::storage::session::SessionStore;
use oodf_ffi::{OwnedDelta, SetField};

use super::helpers::*;

use crate::common::owned_delta::{delta_to_ompf_bytes, empty_delta};

const SESSION_BASE_PATH: &str = "/stage";
const CREATED_BY: &str = "e2e-object-id-preservation-user";
const CLIENT_ID: &str = "e2e-object-id-preservation-client";

/// Safety margin for the b.8y5 tests (4, 5, 6), deliberately BELOW the
/// pre-wipe `seq` these preambles reach (4 for [`drive_preamble`], 7 for
/// [`drive_split_preamble`]).
///
/// `snap_seq > margin` is the production steady state — every layer past
/// its first `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN` commits sits there —
/// and it is exactly the region where the pre-fix `max(snap_seq, observed +
/// margin)` formula dropped the guard band. Reaching it by dialling the
/// margin down costs one config field; reaching it at the default margin of
/// 1000 would cost a thousand UpdateLayer round-trips per test.
const SMALL_SEQ_MARGIN: u64 = 2;

/// Replica id the preambles park the checkpoint leader lock under while they
/// seed phase 1. Anything other than the harness loop's own lock value (its
/// [`SERVICE_LABEL`] leader identity's `lock_value()`) works — the loop's
/// `try_acquire_leader_lock` CAS leaves a lock it does not own untouched and
/// skips the tick.
const FOREIGN_LEADER_ID: &str = "e2e-object-id-preservation-foreign-leader";

/// TTL on the parked leader lock, in seconds.
///
/// A backstop only — both preambles release the lock explicitly. Long enough
/// that a slow seeding phase cannot lapse it, short enough that a preamble
/// panicking mid-seed cannot wedge a subsequent test against the same prefixed
/// key (`TestValkey` prefixes are per-test, so this is belt-and-braces).
const FOREIGN_LEADER_TTL_SECS: i64 = 60;

/// Park the harness's checkpoint loop by claiming its leader lock under
/// [`FOREIGN_LEADER_ID`], so no tick can land a generation until
/// [`release_checkpoint_leader_lock`] runs.
///
/// **Why the preambles need this, and what breaks without it.** Both preambles
/// below gate on `wait_for_checkpoint_at_least(.., pre_wipe_max_seq, ..)`, which
/// pins the generation's `snap_seq` deterministically. It does NOT pin the
/// generation's `observed_commit_span`, and since b.vnw that quantity is half of
/// the restore guard band: `apply_checkpoint` seeds
/// `snap_seq + max(config.checkpoint_seq_safety_margin, observed_commit_span)`.
///
/// `measure_observed_commit_span` (`src/checkpoint/assembler.rs`) returns the
/// no-measurement sentinel when `last_checkpoint_seq == 0`, i.e. for an object's
/// FIRST generation, and otherwise `snap_seq − last_checkpoint_seq`. So whether
/// the loop's immediate first tick happened to fire between the `CreateLayer`
/// and the phase-1 updates decides the band:
///
/// - no early generation → span = sentinel → band = [`SMALL_SEQ_MARGIN`] (2)
/// - early generation at `snap_seq = 1` → span = 3 → band = 3
///
/// Both are correct production behaviour; only one matches the exact
/// `snap_seq + SMALL_SEQ_MARGIN` these tests assert, and which one occurs is a
/// scheduler race. Parking the lock removes the race at its source rather than
/// weakening the assertion: with no tick possible during seeding, the generation
/// the preamble gates on is always the object's first, and the band is always
/// the configured margin.
async fn hold_checkpoint_leader_lock(harness: &CheckpointHarness) {
    let key = harness.tv.key_schema().checkpoint_leader_lock();
    let _: () = harness
        .tv
        .client()
        .pool
        .set(
            &key,
            FOREIGN_LEADER_ID,
            Some(fred::types::Expiration::EX(FOREIGN_LEADER_TTL_SECS)),
            None,
            false,
        )
        .await
        .unwrap_or_else(|e| panic!("parking the checkpoint leader lock must succeed: {e}"));
}

/// Release the lock parked by [`hold_checkpoint_leader_lock`], letting the
/// harness's loop win the next tick's election.
async fn release_checkpoint_leader_lock(harness: &CheckpointHarness) {
    let key = harness.tv.key_schema().checkpoint_leader_lock();
    let _: i64 = harness
        .tv
        .client()
        .pool
        .del(&key)
        .await
        .unwrap_or_else(|e| panic!("releasing the checkpoint leader lock must succeed: {e}"));
}

// ── Streaming framing helpers ────────────────────────────────────────────────
//
// Imported from `tests/integration/checkpoint/helpers.rs` via
// `use super::helpers::*;` (b.dzd Phase 1 deduplication).

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

// ── Delta builders ───────────────────────────────────────────────────────────
//
// `field_update_delta` emits one SetField on a deterministic key so each
// UpdateLayer drives `seq` forward by one without colliding on the same
// key. The fixture purposely does NOT carry a section (and uses
// `section_id = 0`) so we don't need to materialize a parent section
// first — the diff engine on `section_id=0` writes to the layer's "root"
// scope which exists by construction after CreateLayer.

fn field_update_delta(set_order: u64, marker: u8) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        set_fields: vec![SetField {
            section_id: 0,
            key_name: format!("preserve-field-{set_order}"),
            value_or_ext_hash: vec![marker, marker.wrapping_add(1), marker.wrapping_add(2)],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        }],
        ..Default::default()
    }
}

// ── Driver helpers ───────────────────────────────────────────────────────────

/// CreateSession via the SessionStore (no gRPC layer — the session
/// creation path is not under test) then CreateLayer via
/// `handle_create_layer`. Returns `(session_id, object_id)`.
async fn create_session_and_layer(
    harness: &CheckpointHarness,
    session_name: &str,
    layer_name: &str,
) -> (String, u64) {
    let session_store =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
    let session_id = session_store
        .create(session_name, SESSION_BASE_PATH, CREATED_BY)
        .await
        .expect("session creation must succeed");

    // CreateLayer with an empty initial delta — Epic AC #9 only requires
    // that the recovery cycle preserve `object_id`; populating sections
    // here would not strengthen the assertion. The post-recovery
    // assertions read `obj:hdr` directly (HGET), so an empty layer is
    // sufficient — `obj:hdr` is still written by CreateLayer regardless
    // of payload content.
    let ompf = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, layer_name),
        make_create_data(ompf),
    ]);
    let request = make_streaming_token_request(stream);
    let response = handle_create_layer(&harness.svc, request)
        .await
        .expect("CreateLayer must succeed for E2E object-id-preservation harness setup");
    let object_id = response.into_inner().object_id;
    (session_id, object_id)
}

/// Issue one `UpdateLayer` carrying a tiny field-update delta. Returns
/// the produced `sequence_num` so callers can assert the seq ratchet
/// behaviour before / after recovery.
async fn issue_update(
    harness: &CheckpointHarness,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    delta: OwnedDelta,
    client_timestamp: u64,
) -> u64 {
    try_issue_update(
        harness,
        session_id,
        layer_name,
        object_id,
        delta,
        client_timestamp,
    )
    .await
    .expect("UpdateLayer must succeed in the pre-wipe seeding phase")
}

/// [`issue_update`] without the `expect`. Returns the handler's `Result`
/// verbatim so post-recovery callers can name the specific failure in their
/// assertion message — the one this file cares about is what a restore
/// seeded at or below a surviving stream's high-water produces on the very
/// next commit: since b.ph9 an `INTERNAL` whose message carries
/// `update_layer stream-id collision:` and, inside it, Valkey's own
/// `ERR The ID specified in XADD is equal or smaller than the target stream
/// top item`. Before b.ph9 the same condition half-committed — the state
/// mutation and header advance landed and only then did the error surface —
/// so a failure here used to understate the damage.
async fn try_issue_update(
    harness: &CheckpointHarness,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    delta: OwnedDelta,
    client_timestamp: u64,
) -> Result<u64, tonic::Status> {
    let ompf = delta_to_ompf_bytes(&delta, &[]);
    let stream = streaming_update(vec![
        make_update_header(
            session_id,
            layer_name,
            object_id,
            client_timestamp,
            CLIENT_ID,
        ),
        make_update_data(ompf),
    ]);
    let request = make_streaming_token_request(stream);
    handle_update_layer(&harness.svc, request)
        .await
        .map(|response| response.into_inner().sequence_num)
}

/// Read `obj:hdr` as a `HashMap<String, String>` post-recovery. Helper
/// pinned here because all three tests below read the same shape.
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

/// Parse an `obj:hdr` `u64` field, panicking with a diagnostic on
/// missing or unparsable values. Used by every test below for `seq` /
/// `last_checkpoint_seq` reads.
fn parse_header_u64(header: &std::collections::HashMap<String, String>, field: &str) -> u64 {
    let raw = header.get(field).unwrap_or_else(|| {
        panic!(
            "obj:hdr post-recovery MUST carry `{field}` field; got header fields: {:?}",
            header.keys().collect::<Vec<_>>()
        )
    });
    raw.parse::<u64>().unwrap_or_else(|e| {
        panic!("obj:hdr `{field}` MUST parse as u64; got {raw:?}, parse error: {e}")
    })
}

/// Drive the harness through the canonical "create session, create layer,
/// run a few updates, wait for checkpoint, cancel the loop, wipe layer state"
/// preamble that all three tests share. Cancelling the loop before the wipe
/// pins the snap_seq the recovery hook will fetch — without it, a subsequent
/// loop tick could (in theory) land a fresher generation between
/// `wait_for_checkpoint` and the `get_latest_layer_checkpoint` call that
/// drives the test's expectation.
///
/// Takes `&mut CheckpointHarness` so the loop handle can be `mem::take`d
/// and awaited without consuming the rest of the harness (svc / tv /
/// store all stay accessible for the post-recovery assertions).
///
/// Returns `(session_id, object_id, snap_seq_used_by_recovery,
/// pre_wipe_max_seq)`. The `snap_seq` returned here is the snap_seq the
/// recovery hook will observe — read via `get_latest_layer_checkpoint`
/// AFTER the loop has fully drained, so no further generations can land.
async fn drive_preamble(
    harness: &mut CheckpointHarness,
    session_name: &str,
    layer_name: &str,
) -> (String, u64, u64, u64) {
    // No tick may land while phase 1 is being seeded — see
    // [`hold_checkpoint_leader_lock`] for the guard band this protects.
    hold_checkpoint_leader_lock(harness).await;

    let (session_id, object_id) = create_session_and_layer(harness, session_name, layer_name).await;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );

    // A handful of UpdateLayers advance `seq` to a non-trivial value.
    // Three updates plus the CreateLayer's own seq increment leaves
    // pre-wipe `seq` at 4. Two consumers depend on that exact figure:
    // test 3 runs at the default margin (1000), where the guard band
    // dwarfs `snap_seq`; test 4 runs at `SMALL_SEQ_MARGIN` (2), where
    // `snap_seq` dwarfs the guard band — the b.8y5 shape.
    let mut pre_wipe_max_seq = 0_u64;
    for (set_order, marker, ts) in [(1u64, 0x11u8, 1_001u64), (2, 0x22, 1_002), (3, 0x33, 1_003)] {
        let seq = issue_update(
            harness,
            &session_id,
            layer_name,
            object_id,
            field_update_delta(set_order, marker),
            ts,
        )
        .await;
        assert!(
            seq > pre_wipe_max_seq,
            "UpdateLayer sequence_num must advance monotonically; \
             previous max={pre_wipe_max_seq}, got {seq}"
        );
        pre_wipe_max_seq = seq;
    }

    release_checkpoint_leader_lock(harness).await;

    // Block until the leader-elected checkpoint loop lands a generation
    // that captured EVERY update above. Gating on `pre_wipe_max_seq` pins
    // `snap_seq == pre_wipe_max_seq` deterministically (nothing writes after
    // this point, so no later generation can move it), and the parked lock
    // above pins the OTHER half a caller's exact expected `seeded_seq`
    // depends on: with no earlier generation, this one is the object's first,
    // so `observed_commit_span` is the no-measurement sentinel and the restore
    // guard band is exactly the configured safety margin. Under the additive
    // b.8y5 formula both halves are load-bearing, where the old
    // `max(snap_seq, 1000)` form masked any drift below 1000.
    // Returned handle is ignored — we re-read after cancellation to get
    // the FINAL snap_seq the recovery hook sees.
    let _initial_handle =
        wait_for_checkpoint_at_least(&harness.store, object_id, pre_wipe_max_seq, 10).await;

    // Cancel the loop so no further generations can land between here
    // and the recovery trigger. Without this, a race where an
    // additional tick fires post-`wait_for_checkpoint` would shift the
    // snap_seq the recovery hook observes out from under the test's
    // expectation. `mem::replace` swaps in a no-op JoinHandle so the
    // harness's Drop is safe; we await the real handle here under a
    // bounded timeout (so a stuck loop fails fast rather than hanging
    // the suite).
    harness.cancel.cancel();
    let loop_handle = std::mem::replace(&mut harness.loop_handle, tokio::spawn(async {}));
    let _ = tokio::time::timeout(Duration::from_secs(5), loop_handle).await;

    // Capture the snap_seq the recovery hook will fetch. We use
    // `get_latest_layer_checkpoint` (not `list_layer_checkpoints`)
    // because that is the exact entry point `try_restore_layer` calls;
    // any divergence between the two would surface as a snap_seq
    // mismatch in the assertion. After the loop has been cancelled,
    // calling get_latest is deterministic.
    let latest = harness
        .store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest_layer_checkpoint must succeed after wait_for_checkpoint observed one")
        .expect("at least one checkpoint blob must exist after wait_for_checkpoint");
    let snap_seq_for_recovery = latest.handle.snap_seq;
    assert!(
        snap_seq_for_recovery > 0,
        "post-checkpoint snap_seq MUST be > 0; got {snap_seq_for_recovery}"
    );

    // Wipe `le:{object_id}:obj:*` only — the layer mapping survives so
    // `handle_stat_layer`'s mapping-resolve succeeds and the recovery
    // branch fires from the empty header surface.
    wipe_layer_state(&harness.tv, object_id).await;

    (
        session_id,
        object_id,
        snap_seq_for_recovery,
        pre_wipe_max_seq,
    )
}

/// [`drive_preamble`]'s sibling for the high-water-probe tests: land the
/// checkpoint FIRST, then drive further commits with the loop already
/// cancelled, so `snap_seq` and the per-object stream's high-water ID end
/// up at DISTINCT values.
///
/// That separation is what gives the probe tests their signal. If the two
/// coincided, a probe that returned `None` and a probe that returned
/// `Some(high_water)` would seed the same sequence number and the
/// assertions could not tell them apart.
///
/// Deliberately does NOT wipe — the caller chooses the wipe shape
/// ([`wipe_layer_state_preserving_stream`] for the probe tests).
///
/// Returns `(session_id, object_id, snap_seq, stream_high_water)`.
async fn drive_split_preamble(
    harness: &mut CheckpointHarness,
    session_name: &str,
    layer_name: &str,
) -> (String, u64, u64, u64) {
    // No tick may land while phase 1 is being seeded — see
    // [`hold_checkpoint_leader_lock`] for the guard band this protects.
    hold_checkpoint_leader_lock(harness).await;

    let (session_id, object_id) = create_session_and_layer(harness, session_name, layer_name).await;

    // Phase 1 — three commits that the checkpoint WILL capture. CreateLayer
    // itself lands seq=1, so this leaves the layer at seq=4.
    let mut max_seq = 0_u64;
    for (set_order, marker, ts) in [(1u64, 0x11u8, 2_001u64), (2, 0x22, 2_002), (3, 0x33, 2_003)] {
        max_seq = issue_update(
            harness,
            &session_id,
            layer_name,
            object_id,
            field_update_delta(set_order, marker),
            ts,
        )
        .await;
    }

    release_checkpoint_leader_lock(harness).await;

    // Land a generation that captured all of phase 1 — the object's FIRST,
    // guaranteed by the parked lock above, so its `observed_commit_span` is
    // the no-measurement sentinel and the restore band is exactly the
    // configured safety margin. Then stop the loop so phase 2's commits are
    // provably OUTSIDE the checkpoint — the RPO window a real crash would
    // lose.
    let _handle = wait_for_checkpoint_at_least(&harness.store, object_id, max_seq, 10).await;
    harness.cancel.cancel();
    let loop_handle = std::mem::replace(&mut harness.loop_handle, tokio::spawn(async {}));
    let _ = tokio::time::timeout(Duration::from_secs(5), loop_handle).await;

    let latest = harness
        .store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest_layer_checkpoint must succeed after the loop drained")
        .expect("at least one checkpoint blob must exist");
    let snap_seq = latest.handle.snap_seq;

    // Phase 2 — three more commits. These advance both `obj:hdr.seq` and
    // the per-object stream's `last-generated-id` past `snap_seq`.
    let mut stream_high_water = max_seq;
    for (set_order, marker, ts) in [(4u64, 0x44u8, 2_004u64), (5, 0x55, 2_005), (6, 0x66, 2_006)] {
        stream_high_water = issue_update(
            harness,
            &session_id,
            layer_name,
            object_id,
            field_update_delta(set_order, marker),
            ts,
        )
        .await;
    }

    assert!(
        stream_high_water > snap_seq,
        "split preamble must leave the stream high-water ({stream_high_water}) strictly \
         above snap_seq ({snap_seq}); otherwise the probe tests cannot distinguish a \
         working high-water probe from one that returned None"
    );
    assert!(
        snap_seq > SMALL_SEQ_MARGIN,
        "split preamble must also land in the b.8y5 `snap_seq > margin` region; \
         snap_seq={snap_seq}, margin={SMALL_SEQ_MARGIN}"
    );

    (session_id, object_id, snap_seq, stream_high_water)
}

/// Read `last-generated-id` off a live stream and return its `ms`
/// component — which is the layer's sequence number, since both commit
/// paths `XADD` at `{sequence_num}-0`.
///
/// Decodes the reply independently of `RecoveryHook`'s private probe (the
/// probe is what's under test, so reusing its parser would make these
/// assertions circular). Handles both wire shapes: RESP2 delivers a flat
/// `[name, value, …]` array, RESP3 a map.
async fn read_stream_last_generated_id(harness: &CheckpointHarness, stream_key: &str) -> u64 {
    let info: Value = harness
        .tv
        .client()
        .pool
        .xinfo_stream(stream_key, false, None)
        .await
        .expect("XINFO STREAM must succeed on a surviving stream");
    let raw = match &info {
        Value::Map(map) => map
            .iter()
            .find(|(name, _)| name.as_str() == Some("last-generated-id"))
            .and_then(|(_, value)| value.as_str().map(|id| id.into_owned())),
        Value::Array(items) => items
            .chunks_exact(2)
            .find(|pair| pair[0].as_str().as_deref() == Some("last-generated-id"))
            .and_then(|pair| pair[1].as_str().map(|id| id.into_owned())),
        other => panic!("unexpected XINFO STREAM reply shape: {other:?}"),
    }
    .expect("XINFO STREAM reply must carry last-generated-id");
    raw.split_once('-')
        .unwrap_or_else(|| panic!("stream entry id must be `<ms>-<seq>`; got {raw:?}"))
        .0
        .parse::<u64>()
        .unwrap_or_else(|e| panic!("last-generated-id ms component must parse as u64: {e}"))
}

/// Remaining TTL on a key in milliseconds, verbatim from `PTTL`.
///
/// Valkey's sentinel values are passed through rather than normalized, and
/// both matter to the callers below: `-1` = the key exists with no TTL,
/// `-2` = the key does not exist. Collapsing either into an `Option` would
/// let "the stream was deleted outright" masquerade as "no expiry was
/// scheduled".
async fn read_key_ttl_ms(harness: &CheckpointHarness, key: &str) -> i64 {
    harness
        .tv
        .client()
        .pool
        .pttl(key)
        .await
        .unwrap_or_else(|e| panic!("PTTL {key} must succeed: {e}"))
}

/// Assert that a gate-fired recovery removed the stale per-object stream
/// outright, before the restore materialized over it (b.8y5 R4).
///
/// When the probe's plausibility gate rejects a stream top, `apply_checkpoint`
/// seeds `snap_seq + margin` — which sits BELOW that surviving top — and both
/// commit paths `XADD` at `{new_seq}-0`. Leaving the stream in place therefore
/// makes the restored layer permanently unwritable: since b.ph9 every
/// subsequent `UpdateLayer` is refused whole (`INTERNAL`, `update_layer
/// stream-id collision:`) with no state mutation and no header advance, so the
/// writer cursor can never climb past the stale top on its own — only an
/// operator clearing the stream ends it.
///
/// Before b.ph9 the same condition was worse than a write outage: the fast
/// path wrote fields/sections, `HSET` the header, and only THEN `XADD`ed, and
/// a mid-script `redis.call` failure rolled back none of it — so every
/// subsequent `UpdateLayer` **half-committed** (state mutation and header
/// advance landed, the `XADD` failed, the client was told the write failed,
/// and live subscribers never saw the commit and diverged permanently). That
/// is the failure mode the R4 `UNLINK` was designed against, and it is why an
/// earlier revision's `PEXPIRE` with `STREAM_DELETE_GRACE_MS` — reasoning by
/// analogy with the tombstone-writing delete paths — was wrong: the grace
/// window WAS the half-commit window. R4 replaced it with an immediate
/// `UNLINK` issued BEFORE the apply, so the restored layer's first write
/// recreates the stream cleanly from the seeded sequence and there is no
/// window at all. Nothing reads the stale stream to its benefit — a parked
/// subscriber is healed by its own reconnect path — and everything that writes
/// it is refused.
///
/// `PTTL` sentinels are checked rather than `EXISTS` alone so the two
/// regressions are distinguishable in the failure message: `-2` (pass) is
/// "key gone", `-1` is "still there, no expiry" (the unbounded-wedge
/// regression), and any positive value is "back on a grace lease" (the R3
/// bounded-wedge regression). Since b.4et the `UNLINK` is VERIFIED in
/// production rather than best-effort — the reply count is bound as `u64`, a
/// zero is confirmed with a `TYPE` re-probe, and an unverifiable removal
/// refuses the restore outright instead of seeding a header underneath it
/// (test 14) — so on a healthy test Valkey reaching this assertion at all
/// already implies the removal landed. It stays unconditional because it pins
/// the durable fact, not the code path that produced it.
async fn assert_stale_stream_unlinked(harness: &CheckpointHarness, stream_key: &str) {
    let exists: bool = harness
        .tv
        .client()
        .pool
        .exists(stream_key)
        .await
        .unwrap_or_else(|e| panic!("EXISTS {stream_key} must succeed: {e}"));
    let ttl_ms = read_key_ttl_ms(harness, stream_key).await;
    assert!(
        !exists && ttl_ms == -2,
        "b.8y5 R4: after a gate-fired recovery the stale per-object stream MUST be gone \
         outright (EXISTS=0, PTTL=-2); got EXISTS={}, PTTL={ttl_ms}. PTTL=-1 means the key \
         survives with no expiry, so the seeded sequence sits below a stream top that will \
         never lapse and every subsequent UpdateLayer is refused forever (b.ph9 stream-id \
         collision) — pre-b.ph9 it half-committed instead; a positive PTTL means the stream \
         is back on a grace lease, which is the same wedge merely time-boxed",
        u8::from(exists)
    );
}

/// Live value of the `live_edit_checkpoint_recovery_stream_refused_total`
/// counter cell for this harness's service label (b.8y5 R4).
///
/// Each harness owns its own metrics registry, so the reading is scoped to the
/// calling test and an absolute assertion is sound. Mirrors the
/// `recovery_metric` helper in `e2e_live_mapping_guard_test.rs`.
fn stream_refused_metric(harness: &CheckpointHarness) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&harness.metrics),
        "live_edit_checkpoint_recovery_stream_refused_total",
        &[("service", SERVICE_LABEL)],
    )
    .unwrap_or(0.0) as u64
}

/// Drive one real `UpdateLayer` on a just-restored layer whose stale stream the
/// gate removed, and assert the commit lands (b.8y5 R4).
///
/// This is the assertion that actually proves the wedge is gone, and it was
/// impossible to write under the R3 grace design: while the stale stream still
/// existed, this very write was the corrupting half-commit. Now the stream was
/// `UNLINK`ed before the apply, so the commit's `{seq}-0` `XADD` has an empty
/// key to create and must succeed outright.
///
/// Asserts three things, because "returned Ok" alone is weaker than it looks:
/// the commit's `sequence_num` clears the seeded floor, it resumes from the
/// seeded header value by exactly one, and the stream's `last-generated-id`
/// now equals it — i.e. the stream really was recreated by this write from the
/// restored sequence, rather than the handler having returned Ok from a path
/// that skipped the `XADD`.
///
/// `commit_tag` is a per-caller discriminator that seeds the delta's
/// `set_order` (and through it the field key), its byte marker, and the client
/// timestamp — the delta's contents are incidental to what is being asserted,
/// they just have to be distinct per test.
async fn assert_restored_layer_accepts_a_commit(
    harness: &CheckpointHarness,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    stream_key: &str,
    recovered_seq: u64,
    commit_tag: u64,
) {
    let marker = (commit_tag * 0x11) as u8;
    let post_recovery_seq = try_issue_update(
        harness,
        session_id,
        layer_name,
        object_id,
        field_update_delta(commit_tag, marker),
        2_000 + commit_tag,
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "b.8y5 R4: the first UpdateLayer after a gate-fired recovery MUST succeed — the \
             gate UNLINKed the stale stream before the apply, so this commit's \
             `{{seq}}-0` XADD creates the stream fresh. Seeded seq was {recovered_seq}. A \
             failure here — especially an INTERNAL carrying `update_layer stream-id \
             collision:` (b.ph9's refusal, raised from Valkey's `ERR The ID specified in \
             XADD is equal or smaller than the target stream top item`) — means the stale \
             stream is still present. Since b.ph9 that write is refused whole rather than \
             half-committing, so the layer is unwritable rather than silently diverged; \
             either way the gate did not do its job. Status: {status:?}"
        )
    });

    assert_eq!(
        post_recovery_seq,
        recovered_seq + 1,
        "the post-recovery commit must resume from the seeded header value \
         ({recovered_seq}), advancing it by exactly one"
    );
    assert!(
        post_recovery_seq > recovered_seq,
        "the post-recovery commit's sequence_num ({post_recovery_seq}) must clear the \
         seeded floor ({recovered_seq})"
    );
    assert_eq!(
        read_stream_last_generated_id(harness, stream_key).await,
        post_recovery_seq,
        "b.8y5 R4: the post-recovery commit MUST have recreated the per-object stream at \
         its own sequence number. A mismatch means the XADD did not land — the write \
         half-committed and every live subscriber missed this commit"
    );
}

// =============================================================================
// Test 1 — object_id preservation.
// =============================================================================

/// Pins Epic AC #9: a checkpoint with `object_id = A` recovers into
/// Valkey with the same `object_id = A`, not a freshly-allocated one.
///
/// The `object_id` is encoded in the `{object_id}` hash tag of the
/// `le:{A}:obj:hdr` key, not stored as a hash field — preservation
/// therefore means "apply wrote the restored header under the same key
/// the lookup keyed on", which we verify by independently observing:
///
/// 1. The session→layer mapping `GET le:{session_id}:layer:{hex(name)}`
///    still resolves to `object_id_A` post-recovery. This is the lookup
///    surface every gRPC handler keys off — if `object_id` were
///    re-minted, the recovery pipeline would have to also rewrite the
///    mapping pointer to be observationally consistent, and the
///    `RecoveryHook` deliberately does NOT touch the layer mapping
///    (the mapping is what RESOLVED to `object_id_A` in the first place;
///    rewriting it would orphan the captured `object_id`).
/// 2. The `le:{object_id_A}:obj:hdr` key is non-empty and carries the
///    fields the apply pipeline (`LayerStore::write_object` step 5)
///    populates: `session_id`, `layer_name`, `created_by`, `seq`,
///    `last_checkpoint_seq`, etc. A regression that re-minted a fresh
///    `object_id` would surface here as an EMPTY header under
///    `object_id_A` (because apply wrote to the new id's key instead).
/// 3. The `session_id` and `layer_name` fields on the restored
///    `obj:hdr` echo the original CreateLayer values — these are
///    sourced from `CheckpointMeta` (which was assembled BEFORE the
///    wipe), so a regression that re-minted `object_id` AND somehow
///    populated `le:{A}:obj:hdr` (e.g., by writing a placeholder) would
///    surface as mismatched session_id / layer_name vs. the captured
///    pre-wipe values.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn apply_checkpoint_preserves_object_id_across_recovery() {
    let mut harness = setup_checkpoint_harness_tails_off(1).await;
    let layer_name = "e2e-object-id-preservation-layer-1";

    let (session_id, object_id_a, _snap_seq, _pre_wipe_max_seq) = drive_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-1",
        layer_name,
    )
    .await;

    // Trigger recovery via the cheapest wired handler.
    let response = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect(
            "post-wipe StatLayer MUST recover from checkpoint and return Ok; \
             a NotFound here means the recovery branch never fired",
        );
    drop(response);

    // Recovery-metric sanity — exactly one `restored` increment.
    // Mirrored from the sibling `e2e_stat_layer_recovery_test.rs` so a
    // regression that bypassed recovery would fail loudly here before
    // the object_id assertion below.
    assert_eq!(
        live_edit::observability::metrics::test_util::sample_value(
            &live_edit::observability::metrics::test_util::render(&harness.metrics),
            "live_edit_checkpoint_recovery_attempts_total",
            &[
                ("result", "restored"),
                ("scope", "layer"),
                ("service", SERVICE_LABEL)
            ],
        )
        .unwrap_or(0.0) as u64,
        1,
        "exactly ONE recovery attempt must have reported `restored` for AC #9"
    );

    // Load-bearing assertion 1 (AC #9): the session→layer mapping
    // still resolves to `object_id_a`. The recovery pipeline does NOT
    // touch the layer mapping — if a regression re-minted a fresh
    // object_id for the restored state, the mapping pointer would be
    // out of sync with the restored obj:hdr key, AND the post-recovery
    // gRPC call (StatLayer) would have to have observed that fresh id
    // before this point. The mapping read here closes the loop: the
    // gRPC handler resolved the same `object_id_a` we captured at
    // CreateLayer time, end-to-end across the recovery boundary.
    let layer_store = harness.svc.layer_store.clone();
    let mapped_oid = layer_store
        .get_layer_mapping(&session_id, layer_name)
        .await
        .expect("get_layer_mapping must succeed post-recovery")
        .expect("layer mapping MUST survive the obj:* wipe (we only UNLINKed obj:* keys)");
    assert_eq!(
        mapped_oid, object_id_a,
        "Epic AC #9: post-recovery layer mapping ({mapped_oid}) MUST resolve to the \
         pre-wipe captured object_id ({object_id_a}). A mismatch means the recovery \
         pipeline (or a sibling write) re-minted the object_id."
    );

    // Load-bearing assertion 2 (AC #9): the obj:hdr key at the SAME
    // `object_id_a` is non-empty. The apply pipeline writes to
    // `le:{object_id_a}:obj:hdr` if and only if it consumed the
    // recovery hook's `object_id` argument verbatim — re-minting would
    // either leave this key empty (apply wrote to a fresh-id key) or
    // produce a hybrid state that breaks the gRPC handler chain.
    let header = read_obj_header(&harness, object_id_a).await;
    assert!(
        !header.is_empty(),
        "post-recovery obj:hdr at key le:{{{object_id_a}}}:obj:hdr MUST be non-empty; \
         empty means apply_checkpoint wrote to a different object_id's key (re-mint regression)"
    );

    // Load-bearing assertion 3 (AC #9): the restored header's
    // `session_id` and `layer_name` fields match the pre-wipe captured
    // values. These were assembled into the `CheckpointMeta` BEFORE
    // the wipe and re-stamped by `LayerStore::write_object` step 5
    // during apply — a re-mint regression that also somehow seeded
    // `le:{A}:obj:hdr` (e.g., by re-running CreateLayer on the side)
    // would surface here as mismatched identity fields.
    let restored_session_id = header.get("session_id").unwrap_or_else(|| {
        panic!(
            "post-recovery obj:hdr MUST carry `session_id`; got fields: {:?}",
            header.keys().collect::<Vec<_>>()
        )
    });
    let restored_layer_name = header.get("layer_name").unwrap_or_else(|| {
        panic!(
            "post-recovery obj:hdr MUST carry `layer_name`; got fields: {:?}",
            header.keys().collect::<Vec<_>>()
        )
    });
    assert_eq!(
        restored_session_id, &session_id,
        "Epic AC #9: post-recovery obj:hdr.session_id MUST equal pre-wipe captured \
         session_id; got {restored_session_id:?}, expected {session_id:?}"
    );
    assert_eq!(
        restored_layer_name, layer_name,
        "Epic AC #9: post-recovery obj:hdr.layer_name MUST equal pre-wipe captured \
         layer_name; got {restored_layer_name:?}, expected {layer_name:?}"
    );
}

// =============================================================================
// Test 2 — last_checkpoint_seq == snap_seq (NOT seeded_seq).
// =============================================================================

/// Pins apply step 6 (SR-2.5): `HSET obj:hdr last_checkpoint_seq=snap_seq`.
/// The value is `snap_seq` — the snapshot's authoritative version — NOT
/// the safety-margin-bumped `seeded_seq` that the writer cursor was
/// advanced to.
///
/// Also pins the second cursor column b.yut added alongside it,
/// `last_checkpoint_ms` — the restored blob's own recorded write time. The
/// two travel together in one `CheckpointCursorStamp` for exactly this
/// reason: a header must never disagree with itself about whether, and when,
/// a generation landed.
///
/// With `safety_margin=1000` (default) and a pre-wipe `seq` of 4
/// (CreateLayer + 3 UpdateLayers), `snap_seq=4` and `seeded_seq=1004` —
/// they differ by the full margin, giving the assertion unambiguous
/// signal. A regression that wrote `seeded_seq` into the
/// `last_checkpoint_seq` field would surface here as a value far above the
/// snap_seq the test captured.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn last_checkpoint_seq_advances_on_apply_to_snap_seq() {
    let mut harness = setup_checkpoint_harness_tails_off(1).await;
    let layer_name = "e2e-object-id-preservation-layer-2";

    let (session_id, object_id, snap_seq_for_recovery, _pre_wipe_max_seq) = drive_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-2",
        layer_name,
    )
    .await;

    // Trigger recovery.
    let _ = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect("post-wipe StatLayer MUST recover from checkpoint and return Ok");

    let header = read_obj_header(&harness, object_id).await;
    let last_checkpoint_seq = parse_header_u64(&header, "last_checkpoint_seq");

    // Load-bearing assertion: `last_checkpoint_seq` MUST equal the
    // `snap_seq` of the handle the recovery hook fetched — NOT
    // `seeded_seq`. The apply pipeline computes `seeded_seq =
    // max(snap_seq, max_seq_observed.unwrap_or(0)) + safety_margin` and
    // stamps `last_checkpoint_seq=snap_seq` independently. A regression
    // that conflated the two would surface here as
    // `last_checkpoint_seq=seeded_seq` (strictly above `snap_seq`, since
    // b.8y5 makes the margin additive at every input), failing the
    // strict equality below.
    assert_eq!(
        last_checkpoint_seq, snap_seq_for_recovery,
        "SR-2.5 step 6: post-recovery obj:hdr.last_checkpoint_seq ({last_checkpoint_seq}) MUST \
         equal the recovery snap_seq ({snap_seq_for_recovery}). A mismatch — especially \
         `last_checkpoint_seq` matching `seeded_seq` (>= safety_margin) — indicates apply \
         stamped the writer cursor into the audit-trail field instead of the snapshot \
         generation."
    );

    // b.yut — the OTHER half of the checkpoint cursor. `apply_checkpoint`
    // stamps `last_checkpoint_ms` with the RESTORED BLOB'S OWN recorded write
    // time (`CheckpointMeta::written_at_unix_ms`), in the same atomic HSET as
    // the seq half.
    //
    // Why the blob's time and not `now`, and not absent:
    //
    //   - It is the truthful value. That column answers "when did the newest
    //     durable copy of this layer land", and post-restore the newest
    //     durable copy IS this blob, written when it says it was. A recovered
    //     layer therefore re-enters the cadence at the true age of its newest
    //     durable copy.
    //   - Leaving it absent reads as 0 = "never checkpointed", which is not
    //     merely conservative: it is wrong in the direction that costs a
    //     redundant full re-snapshot of bytes the fleet just read back.
    //   - Stamping `now` would be wrong in the other direction — it would
    //     silently restart the cadence clock on every recovery.
    //
    // Read the blob back out of the store rather than threading the meta
    // through `drive_preamble`: the loop was cancelled before the wipe, so
    // `get_latest_layer_checkpoint` is deterministic here and returns the same
    // blob the recovery hook consumed.
    let restored_blob = harness
        .store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("get_latest_layer_checkpoint must succeed post-recovery")
        .expect("the recovery blob must still be on the adapter");
    let blob_written_at_ms = restored_blob.meta.written_at_unix_ms;
    assert_ne!(
        blob_written_at_ms, 0,
        "test precondition: the assembler must have stamped a real \
         written_at_unix_ms on the checkpoint blob, otherwise this case is \
         silently exercising apply's now_unix_ms fallback branch instead of the \
         pass-through one"
    );
    let restored_checkpoint_ms = parse_header_u64(&header, "last_checkpoint_ms");
    assert_eq!(
        restored_checkpoint_ms, blob_written_at_ms,
        "b.yut: post-recovery obj:hdr.last_checkpoint_ms ({restored_checkpoint_ms}) \
         MUST equal the restored blob's CheckpointMeta::written_at_unix_ms \
         ({blob_written_at_ms}). Absent (or 0) reads as 'never checkpointed' and \
         costs a redundant re-snapshot of bytes we just read back; `now` would \
         silently restart the cadence clock on every recovery"
    );

    // Sanity guard: confirm `snap_seq` and `seeded_seq` actually diverge
    // for this test, so the strict equality above is discriminating
    // rather than accidentally true. Post-b.8y5 the margin is additive at
    // EVERY input, so `seeded_seq = snap_seq + margin > snap_seq` holds
    // unconditionally for any accepted (non-zero) margin — we assert the
    // divergence directly off the header rather than via the old
    // `snap_seq < safety_margin` proxy, which was only ever a sufficient
    // condition for the `max()` form and made this check vacuous the
    // moment a layer outlived its first `margin` commits.
    let recovered_seq = parse_header_u64(&header, "seq");
    assert!(
        recovered_seq > last_checkpoint_seq,
        "test pre-condition: seeded seq ({recovered_seq}) MUST strictly exceed \
         last_checkpoint_seq ({last_checkpoint_seq}) so the equality assertion above \
         can distinguish `last_checkpoint_seq=snap_seq` from `last_checkpoint_seq=seeded_seq`"
    );
}

// =============================================================================
// Test 3 — seq advances to restore_floor + safety_margin (margin >> snap_seq).
// =============================================================================

/// Pins apply step 4 (SR-2.5): `seeded_seq = max(snap_seq,
/// max_seq_observed.unwrap_or(0)) + checkpoint_seq_safety_margin` is
/// stamped on the restored `obj:hdr.seq`. `wipe_layer_state` takes
/// `le:{oid}:obj:stream` along with the rest of `obj:*`, so the recovery
/// hook's high-water probe finds no stream and `max_seq_observed` is
/// `None` — the restore floor is `snap_seq` alone.
///
/// This is the "short-lived layer" half of the b.8y5 pair: the default
/// margin (1000) towers over the pre-wipe `seq` of 4, so the post-recovery
/// `seq` lands at 1004. Test 4 covers the complementary — and far more
/// common in production — `snap_seq > margin` shape, which is where the
/// pre-fix `max()` form silently dropped the guard band.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_number_advances_to_snap_seq_plus_safety_margin() {
    let mut harness = setup_checkpoint_harness_tails_off(1).await;
    let layer_name = "e2e-object-id-preservation-layer-3";

    let (session_id, object_id, snap_seq_for_recovery, pre_wipe_max_seq) = drive_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-3",
        layer_name,
    )
    .await;

    // Trigger recovery.
    let _ = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect("post-wipe StatLayer MUST recover from checkpoint and return Ok");

    let header = read_obj_header(&harness, object_id).await;
    let recovered_seq = parse_header_u64(&header, "seq");

    // Compute the expectation from the live config so a future change to
    // the default surfaces as a clean assertion failure rather than a
    // hand-tuned 1004. The stream went with the wipe, so the probe yields
    // `None` and the restore floor is `snap_seq` alone.
    let cfg = crate::common::test_config();
    let expected_seq = snap_seq_for_recovery + cfg.checkpoint_seq_safety_margin;

    // Load-bearing assertion: `obj:hdr.seq` MUST equal the SR-2.5
    // seeded value. The apply pipeline writes
    // `state.sequence_num = seeded_seq` via
    // `LayerStore::write_object`, which serializes it into the
    // `obj:hdr.seq` field verbatim.
    assert_eq!(
        recovered_seq, expected_seq,
        "SR-2.5 step 4: post-recovery obj:hdr.seq ({recovered_seq}) MUST equal \
         snap_seq={snap_seq_for_recovery} + safety_margin={} = {expected_seq}. \
         A mismatch indicates apply lost the safety-margin advance (regression in step 4).",
        cfg.checkpoint_seq_safety_margin
    );

    // Cross-check: post-recovery `seq` strictly exceeds the pre-wipe
    // max. A regression that wrote `pre_wipe_max_seq` (or any value
    // below the floor) would surface here even before the strict
    // equality check above — this gives a clearer diagnostic on a
    // common regression shape (apply restored seq from snapshot
    // without the guard band).
    assert!(
        recovered_seq > pre_wipe_max_seq,
        "SR-2.5 step 4: post-recovery seq ({recovered_seq}) MUST be > pre-wipe max \
         ({pre_wipe_max_seq}); equal-or-below indicates the safety_margin advance was lost"
    );
}

// =============================================================================
// Test 4 — b.8y5 regression: the guard band survives `snap_seq > margin`.
// =============================================================================

/// The b.8y5 regression guard, end-to-end.
///
/// The pre-fix formula was `seeded_seq = max(snap_seq,
/// max_seq_observed.unwrap_or(0) + safety_margin)`. It anchored the
/// additive term at zero, so the margin only ever mattered while
/// `snap_seq < margin` — i.e. for a layer's first
/// `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN` commits. Past that, every
/// restore resumed at *exactly* `snap_seq`: the very next `UpdateLayer`
/// re-issued a sequence number that pre-restore subscribers already hold
/// attached to different, now-lost content (the b.dy5-class staleness
/// hazard the knob is advertised to prevent). Test 3 above could not
/// catch it — with a default margin of 1000 against a pre-wipe seq of 4,
/// both formulas agree.
///
/// This test inverts that relationship the cheap way: dial the margin down
/// to `SMALL_SEQ_MARGIN` rather than issuing a thousand UpdateLayers to
/// push `snap_seq` past 1000. The formulas then disagree —
/// `max(4, 0 + 2) = 4` vs `max(4, 0) + 2 = 6` — and the strict-inequality
/// assertion (`recovered_seq > pre_wipe_max_seq`) fails under the old one
/// and passes under the new.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_guard_band_applies_when_snap_seq_exceeds_safety_margin() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-4";

    let (session_id, object_id, snap_seq_for_recovery, pre_wipe_max_seq) = drive_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-4",
        layer_name,
    )
    .await;

    // The whole point of this case. If a future change to `drive_preamble`
    // pushed the commit count above the margin's reach, the assertions
    // below would quietly degenerate into a copy of test 3.
    assert!(
        snap_seq_for_recovery > SMALL_SEQ_MARGIN,
        "test pre-condition: snap_seq ({snap_seq_for_recovery}) MUST exceed the safety margin \
         ({SMALL_SEQ_MARGIN}) — that is the b.8y5 shape this test exists to cover"
    );

    // Trigger recovery.
    let _ = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect("post-wipe StatLayer MUST recover from checkpoint and return Ok");

    let header = read_obj_header(&harness, object_id).await;
    let recovered_seq = parse_header_u64(&header, "seq");

    assert!(
        recovered_seq > pre_wipe_max_seq,
        "b.8y5: post-recovery seq ({recovered_seq}) MUST strictly exceed the pre-wipe max \
         ({pre_wipe_max_seq}) even though snap_seq ({snap_seq_for_recovery}) is larger than \
         the safety margin ({SMALL_SEQ_MARGIN}). Equality means the margin went inert and \
         the restored layer resumed at exactly the last provable sequence number, with no \
         guard band over the commits lost inside the RPO window"
    );

    assert_eq!(
        recovered_seq,
        snap_seq_for_recovery + SMALL_SEQ_MARGIN,
        "b.8y5: post-recovery seq ({recovered_seq}) MUST equal snap_seq \
         ({snap_seq_for_recovery}) + safety_margin ({SMALL_SEQ_MARGIN}) — the margin is \
         additive above the restore floor, not an alternative floor"
    );

    // The audit-trail column is unaffected by the guard band: it still
    // points at the generation that seeded the restore.
    assert_eq!(
        parse_header_u64(&header, "last_checkpoint_seq"),
        snap_seq_for_recovery,
        "last_checkpoint_seq must still stamp snap_seq verbatim, not the guard-banded seed"
    );
}

// =============================================================================
// Test 5 — b.8y5 probe: a surviving per-object stream raises the floor.
// =============================================================================

/// Pins the success arm of the recovery hook's `XINFO STREAM` high-water
/// probe (b.8y5).
///
/// `apply_checkpoint` runs whenever `obj:hdr` is absent, but header absence
/// does NOT imply the per-object stream is gone — `le:{oid}:obj:stream` is
/// an independent key that merely shares the `{object_id}` hash tag. When
/// it survived, its `last-generated-id` is the EXACT pre-crash sequence
/// number, strictly better than the checkpoint's `snap_seq` (which lags by
/// up to one RPO window).
///
/// It is also a hard correctness requirement, not just a tightening: both
/// commit paths `XADD` at `{new_seq}-0`, so seeding at or below a surviving
/// stream's high-water wedges every subsequent `UpdateLayer` on `ERR The ID
/// specified in XADD is equal or smaller than the target stream top item`.
/// Since b.ph9 those writes are refused whole — an `INTERNAL` prefixed
/// `update_layer stream-id collision:`, with no state mutation and no header
/// advance — so the cursor never climbs past the stream top on its own and
/// the layer stays unwritable until an operator clears the stream. The
/// correct floor is what keeps the layer writable at all.
///
/// Shape: land a checkpoint, THEN drive further commits with the checkpoint
/// loop cancelled, so `snap_seq` (4) and the stream high-water (7) are
/// distinct. A probe that returned `None` would seed `4 + margin`; the
/// probe working seeds `7 + margin`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_folds_in_surviving_stream_high_water() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-5";

    let (session_id, object_id, snap_seq_for_recovery, stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-5",
        layer_name,
    )
    .await;

    // Partial loss: `obj:*` gone, stream intact.
    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    assert_eq!(
        read_stream_last_generated_id(&harness, &stream_key).await,
        stream_high_water,
        "test pre-condition: the surviving stream's last-generated-id must carry the \
         post-checkpoint commits, otherwise the probe has nothing to contribute"
    );

    // Trigger recovery.
    let _ = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect("post-wipe StatLayer MUST recover from checkpoint and return Ok");

    let header = read_obj_header(&harness, object_id).await;
    let recovered_seq = parse_header_u64(&header, "seq");

    assert_eq!(
        recovered_seq,
        stream_high_water + SMALL_SEQ_MARGIN,
        "b.8y5: with the per-object stream surviving the wipe, the restore floor must be \
         max(snap_seq={snap_seq_for_recovery}, stream_high_water={stream_high_water}) and \
         the seeded seq must clear it by the safety margin ({SMALL_SEQ_MARGIN}). A value of \
         {} means the high-water probe returned None and the floor fell back to snap_seq — \
         leaving the seeded seq at or below the surviving stream top, which wedges every \
         subsequent UpdateLayer's XADD",
        snap_seq_for_recovery + SMALL_SEQ_MARGIN
    );

    // The property the probe exists to guarantee, asserted directly rather
    // than inferred from the arithmetic above.
    assert!(
        recovered_seq > stream_high_water,
        "b.8y5: seeded seq ({recovered_seq}) MUST strictly exceed the surviving stream's \
         high-water ({stream_high_water}); both commit paths XADD at `{{seq}}-0`, so a \
         seed at or below it makes the next UpdateLayer fail on a stale stream top"
    );

    // ...and the consequence of that property, driven for real rather than
    // reasoned about. Every other post-recovery write in this suite runs
    // against a stream that went WITH the wipe, so the seeded sequence has
    // an empty stream to XADD into and the wedge cannot surface. This is the
    // one case where the stream survived, which is precisely the shape that
    // trips the b.ph9 stream-ID collision guard (`update_layer stream-id
    // collision:`, raised from Valkey's `ERR The ID specified in XADD is
    // equal or smaller than the target stream top item`) when the floor is
    // computed wrong. A header-only assertion would pass against a broken
    // probe whose seeded value merely *looks* right; only an actual commit
    // proves the writer is not wedged.
    let post_recovery_seq = try_issue_update(
        &harness,
        &session_id,
        layer_name,
        object_id,
        field_update_delta(7, 0x77),
        2_007,
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "b.8y5: the first UpdateLayer after recovering onto a SURVIVING per-object \
             stream must succeed. Seeded seq was {recovered_seq} and the surviving stream's \
             last-generated-id was {stream_high_water}; a failure here — especially an \
             INTERNAL carrying `update_layer stream-id collision:` (b.ph9's refusal, raised \
             from Valkey's `ERR The ID specified in XADD is equal or smaller than the \
             target stream top item`) — means the restore floor did not clear the stream \
             top and every \
             subsequent commit on this layer is wedged. Status: {status:?}"
        )
    });

    assert!(
        post_recovery_seq > stream_high_water,
        "b.8y5: the post-recovery commit's sequence_num ({post_recovery_seq}) MUST clear the \
         surviving stream's pre-recovery high-water ({stream_high_water}) — that is the ID \
         the server compares the commit's own `{{seq}}-0` XADD against"
    );
    assert_eq!(
        post_recovery_seq,
        recovered_seq + 1,
        "the post-recovery commit must resume from the seeded header value \
         ({recovered_seq}), advancing it by exactly one"
    );

    // Read the stream back independently: the commit's XADD must have landed
    // as the stream's new top, not merely returned Ok from a code path that
    // skipped the XADD.
    assert_eq!(
        read_stream_last_generated_id(&harness, &stream_key).await,
        post_recovery_seq,
        "the post-recovery commit MUST have advanced the surviving stream's \
         last-generated-id to its own sequence number"
    );

    // Negative control for `unlink_stale_stream` (b.8y5 R4), and the reason
    // the gate tests' key-gone assertions are not vacuous: removal is done by
    // a REFUSING gate, never by a successful probe. Here the probe folded the
    // high-water in, the layer is healthy, and this is the live stream the
    // layer keeps writing to — removing it would strand every in-window
    // subscriber on it and throw away their catch-up for nothing. Note this
    // runs AFTER the post-recovery commit above, so it also pins that the
    // commit path itself does not put the stream on a fuse.
    assert_eq!(
        read_key_ttl_ms(&harness, &stream_key).await,
        -1,
        "b.8y5 R4: a SUCCESSFUL high-water probe must leave the surviving stream in place \
         and untouched (PTTL=-1, key present with no expiry). -2 means the stream was \
         unlinked on the success path, which would strand in-window subscribers; a positive \
         value means it was put on an expiry fuse, which would delete a live, \
         actively-written stream out from under an active layer"
    );
}

// =============================================================================
// Test 6 — b.8y5 probe: last-generated-id, not the XREVRANGE tail.
// =============================================================================

/// Same partial-loss shape as test 5, but the top stream entry is `XDEL`ed
/// before recovery runs.
///
/// This discriminates the two candidate probe implementations. `XDEL`
/// lowers what `XREVRANGE + - COUNT 1` reports (it drops to the previous
/// surviving entry) while `XINFO STREAM`'s `last-generated-id` holds — and
/// `last-generated-id` is precisely the value the server compares an
/// incoming `XADD` ID against, so a deleted top entry still rejects a
/// re-used ID. A probe built on the `XREVRANGE` tail would under-report the
/// floor by exactly the deleted suffix, in exactly the case that produces
/// the wedged-writer failure.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_uses_last_generated_id_not_tail_after_xdel() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-6";

    let (session_id, object_id, _snap_seq, stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-6",
        layer_name,
    )
    .await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;

    // Delete the top entry. `last-generated-id` must NOT move.
    let deleted: u64 = harness
        .tv
        .client()
        .pool
        .xdel(&stream_key, vec![format!("{stream_high_water}-0")])
        .await
        .expect("XDEL of the stream top entry must succeed");
    assert_eq!(
        deleted, 1,
        "test pre-condition: XDEL must have removed exactly the top entry \
         ({stream_high_water}-0)"
    );
    assert_eq!(
        read_stream_last_generated_id(&harness, &stream_key).await,
        stream_high_water,
        "Valkey contract this probe rests on: XDEL of the top entry does NOT lower \
         `last-generated-id` (only the XREVRANGE tail moves)"
    );

    // Trigger recovery.
    let _ = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect("post-wipe StatLayer MUST recover from checkpoint and return Ok");

    let header = read_obj_header(&harness, object_id).await;
    let recovered_seq = parse_header_u64(&header, "seq");

    assert_eq!(
        recovered_seq,
        stream_high_water + SMALL_SEQ_MARGIN,
        "b.8y5: the high-water probe must read `last-generated-id` ({stream_high_water}), \
         which survives the XDEL, not the XREVRANGE tail, which does not. A lower value \
         means the probe is reading the tail and would under-report the floor exactly \
         when a trimmed stream is about to reject the next XADD"
    );
}

// =============================================================================
// Tests 7–8 — b.8y5 probe: the gate that refuses a foreign high-water, and
// the stale-stream removal that refusal requires.
// =============================================================================

/// Stream-entry field name carrying the notification payload. Mirrors
/// `NOTIFICATION_PAYLOAD_FIELD` in `src/storage/notifications.rs`.
const PAYLOAD_FIELD: &str = "payload";

/// Payload value the two tombstone `XADD` sites in `src/storage/layer.rs`
/// (`delete_layer` and the CreateLayer-overwrite cascade) append.
const DELETED_SENTINEL: &str = "deleted";

/// Gap above `snap_seq` past which the probe refuses to fold a high-water
/// (`MAX_PLAUSIBLE_HIGH_WATER_GAP` in `src/checkpoint/recovery.rs`). Named
/// here rather than imported — the constant is private to the production
/// module, and a test that re-derived it from the module under test could
/// not detect a change to it.
const MAX_PLAUSIBLE_HIGH_WATER_GAP: u64 = 1_000_000_000;

/// Append a stream entry with an explicit ID, returning the ID the server
/// assigned. Panics on failure — a rejected `XADD` here means the test's own
/// setup is wrong (usually a non-monotonic ID), not a production defect.
async fn xadd_entry(
    harness: &CheckpointHarness,
    stream_key: &str,
    id: &str,
    payload: &str,
) -> String {
    harness
        .tv
        .client()
        .pool
        .xadd(
            stream_key,
            false,
            None,
            id,
            vec![(PAYLOAD_FIELD, Value::from(payload))],
        )
        .await
        .unwrap_or_else(|e| panic!("XADD {id} on {stream_key} must succeed in test setup: {e}"))
}

/// The plausibility gate on production's own foreign-high-water shape: an
/// auto-ID tombstone.
///
/// `delete_layer` and the CreateLayer-overwrite cascade both append their
/// `payload=deleted` sentinel with the **auto-ID `*`**, so Valkey mints the
/// entry ID from the wall clock and the stream's `last-generated-id`
/// permanently becomes a unix-millisecond value (~1.79e12 and rising). That
/// is *below* `apply_checkpoint`'s `2^53 − 1` ceiling, so nothing downstream
/// would have caught it: the layer would come back materialized with a
/// timestamp-shaped `sequence_num`, three orders of magnitude above anything
/// the service issued, permanently.
///
/// A tombstoned object is normally unreachable from recovery — apply's
/// SR-2.4 live-mapping guard rejects first — but session-checkpoint restore
/// re-creates `layer_name → object_id` mappings verbatim, so a pre-delete
/// session checkpoint can re-point a live mapping at a tombstoned object and
/// defeat that guard. Hence "handled", not "unreachable".
///
/// One cause-agnostic gate covers this and everything shaped like it. An
/// earlier revision paired it with a dedicated tombstone check that asked
/// whether the top entry carried the `deleted` sentinel; that check was
/// provably subsumed — an auto-ID is ~1.787e12 while the bound rejects
/// anything more than 1e9 above `snap_seq`, three orders of magnitude of
/// separation — and both arms took identical action, so it was dropped. Test 8
/// covers the same gate on the input the dropped check could NOT have caught.
///
/// This test asserts all three halves of the gate's contract: the floor falls
/// back to `snap_seq`, the stale stream is removed outright before the apply,
/// and — the part that was impossible to assert while the stream lingered on a
/// grace lease — a real `UpdateLayer` on the restored layer then SUCCEEDS,
/// recreating the stream from the seeded sequence.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_ignores_tombstoned_stream_with_auto_id() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-7";

    let (session_id, object_id, snap_seq, stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-7",
        layer_name,
    )
    .await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    let refused_before = stream_refused_metric(&harness);

    // The tombstone, in exactly the shape `LayerStore::delete_layer` writes
    // it: auto-ID, one `payload=deleted` field.
    let tombstone_id: String = harness
        .tv
        .client()
        .pool
        .xadd(
            &stream_key,
            false,
            None,
            "*",
            vec![(PAYLOAD_FIELD, Value::from(DELETED_SENTINEL))],
        )
        .await
        .expect("XADD of the `deleted` tombstone must succeed");

    // Pre-condition: the auto-ID really is timestamp-shaped, i.e. far enough
    // above `snap_seq` that folding it would be catastrophic rather than
    // merely wrong. Without this the test could pass vacuously on a Valkey
    // whose clock made the auto-ID indistinguishable from a sequence number.
    let tombstone_seq = read_stream_last_generated_id(&harness, &stream_key).await;
    assert!(
        tombstone_seq > snap_seq.saturating_add(MAX_PLAUSIBLE_HIGH_WATER_GAP),
        "test pre-condition: the auto-ID tombstone ({tombstone_id}) must be a wall-clock \
         timestamp far above snap_seq ({snap_seq}); got ms component {tombstone_seq}"
    );

    // Trigger recovery.
    let _ = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect("post-wipe StatLayer MUST recover from checkpoint and return Ok");

    let header = read_obj_header(&harness, object_id).await;
    let recovered_seq = parse_header_u64(&header, "seq");

    assert_eq!(
        recovered_seq,
        snap_seq + SMALL_SEQ_MARGIN,
        "b.8y5 R2: with a `deleted` tombstone on top of the surviving stream, the probe \
         MUST return None and the restore floor MUST fall back to snap_seq ({snap_seq}). \
         A value near {tombstone_seq} means the wall-clock auto-ID was folded into the \
         floor and the layer came back with a timestamp for a sequence number; a value of \
         {} means the tombstone was ignored and the pre-tombstone high-water was folded \
         instead",
        stream_high_water + SMALL_SEQ_MARGIN
    );

    // The gate does not merely decline — it removes the stale stream its
    // declining would otherwise corrupt against. Without this every
    // subsequent UpdateLayer on the restored layer half-commits forever.
    assert_stale_stream_unlinked(&harness, &stream_key).await;

    // The consequence of that removal, driven for real rather than reasoned
    // about. This is the assertion the R3 grace-window design could not
    // support: back then this very write was the half-commit.
    assert_restored_layer_accepts_a_commit(
        &harness,
        &session_id,
        layer_name,
        object_id,
        &stream_key,
        recovered_seq,
        7,
    )
    .await;

    // Operator-visible signal for a refusal, which is otherwise only a log
    // line. Additive to the recovery-attempt counter — the restore itself
    // still reports success — so it is the only metric that distinguishes
    // "restored cleanly" from "restored after throwing away a stream".
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before + 1,
        "b.8y5 R4: a gate-fired refusal MUST increment \
         live_edit_checkpoint_recovery_stream_refused_total{{service=\"{SERVICE_LABEL}\"}}. \
         Without it a fleet silently discarding surviving streams is indistinguishable from \
         one recovering cleanly, because the recovery-attempt counter reports success either \
         way"
    );
}

/// The same plausibility gate, on the input a dedicated tombstone check could
/// never have caught.
///
/// A tombstone check would have to read the stream's `last-entry` — and
/// `MAXLEN ~` trimming evicts entries while `last-generated-id` does NOT move
/// back. So a stream whose tombstone has been trimmed away still reports a
/// wall-clock high-water with no `deleted` sentinel left to recognize it by.
/// Being cause-agnostic is exactly what makes one gate sufficient: it equally
/// catches a trimmed-away tombstone, and any future writer that puts a
/// non-sequence-shaped ID on this stream, rather than silently seeding from
/// it.
///
/// The top entry here therefore carries an ordinary `update|...` payload, not
/// the `deleted` sentinel — there is nothing about it to recognize except the
/// magnitude of its ID. That ID is `snap_seq + 2e9`, double the bound, and far
/// below the `2^53 − 1` ceiling that would have caught it downstream, which is
/// exactly why the gate has to exist.
///
/// Asserts the same three-part contract as test 7 — floor falls back, stale
/// stream removed outright, restored layer then accepts a real commit — on the
/// input that isolates the gate from the tombstone shape.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_ignores_implausible_stream_high_water() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-8";

    let (session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-8",
        layer_name,
    )
    .await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    let refused_before = stream_refused_metric(&harness);

    // Twice the plausibility gap above snap_seq, carrying a payload in the
    // shape a real commit writes (`update|oid|seq|client|ts|ts`, per
    // `LayerNotification::parse`) rather than the `deleted` sentinel — the
    // magnitude of the ID is the only thing the gate can go on.
    let implausible_seq = snap_seq + 2 * MAX_PLAUSIBLE_HIGH_WATER_GAP;
    xadd_entry(
        &harness,
        &stream_key,
        &format!("{implausible_seq}-0"),
        &format!("update|{object_id}|{implausible_seq}|{CLIENT_ID}|2100|2100"),
    )
    .await;
    assert_eq!(
        read_stream_last_generated_id(&harness, &stream_key).await,
        implausible_seq,
        "test pre-condition: the synthetic entry must be the stream's new high-water"
    );

    // Trigger recovery.
    let _ = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect("post-wipe StatLayer MUST recover from checkpoint and return Ok");

    let header = read_obj_header(&harness, object_id).await;
    let recovered_seq = parse_header_u64(&header, "seq");

    assert_eq!(
        recovered_seq,
        snap_seq + SMALL_SEQ_MARGIN,
        "b.8y5 R2: a high-water more than {MAX_PLAUSIBLE_HIGH_WATER_GAP} above snap_seq \
         ({snap_seq}) MUST NOT be folded into the restore floor — it cannot be a genuine \
         commit count and is far below the 2^53-1 ceiling that would otherwise catch it. \
         A value of {} means the bound is absent and the foreign ID was seeded from",
        implausible_seq + SMALL_SEQ_MARGIN
    );

    // Refusing without removing the stream is the half-commit window; the
    // stale stream must be gone before the apply materialized over it.
    assert_stale_stream_unlinked(&harness, &stream_key).await;

    // Proof the write path is genuinely healthy afterwards, not merely
    // arithmetically plausible: the commit's XADD recreates the stream.
    assert_restored_layer_accepts_a_commit(
        &harness,
        &session_id,
        layer_name,
        object_id,
        &stream_key,
        recovered_seq,
        9,
    )
    .await;

    assert_eq!(
        stream_refused_metric(&harness),
        refused_before + 1,
        "b.8y5 R4: the plausibility gate MUST increment \
         live_edit_checkpoint_recovery_stream_refused_total{{service=\"{SERVICE_LABEL}\"}} on \
         this arm too — the restore reports success either way, so this counter is the only \
         signal that a surviving stream was discarded"
    );
}

// =============================================================================
// Test 9 — b.8y5: the probe survives a flushed server-side script cache.
// =============================================================================

/// Test 5's success arm, re-run with the server's script cache flushed
/// immediately before recovery.
///
/// This is not a hypothetical. The probe's whole reason to exist is state
/// loss, and the single most common cause of that state loss — a Valkey
/// restart — also empties the `EVALSHA` cache. So "first probe after the
/// event this feature exists for" is very often the NOSCRIPT case, and it
/// is the case a plain `evalsha` would fail.
///
/// It has to be its own test because of what the failure looked like when
/// this test was written, and because of what it looks like now — neither
/// of which any other test in the file pins.
///
/// Pre-b.4et the probe was best-effort: every error path, NOSCRIPT
/// included, folded to `None` and the restore quietly fell back to
/// `snap_seq` — the exact value tests 1–4 and 7–8 all expect. A regression
/// from `evalsha_with_reload` to plain `evalsha` would therefore have left
/// the entire rest of the suite green while wedge-prevention was off in
/// production. Post-b.4et that same regression is loud on THIS shape (a
/// surviving stream turns the unreadable probe into a refused restore), but
/// silent on the shape that dominates production — a full keyspace loss,
/// where `TYPE` reads `none` and the restore proceeds at `snap_seq`
/// regardless. So the reload still needs its own witness, and this is it:
/// the assertion is on the value the reload PRODUCES (`stream_high_water +
/// margin`), which only a working reload can reach.
///
/// Only test 5 could otherwise catch it, and only by luck: whether the
/// script is still cached from an earlier test in the same shared Valkey is
/// not something test 5 controls.
///
/// `SCRIPT FLUSH` is server-global, so it also empties the cache for any
/// test running concurrently against the shared Valkey. That is safe here
/// because every script call site in service code reloads the script on
/// `NOSCRIPT` and retries (`evalsha_with_reload`, or
/// `storage::lua::evalsha_once_with_reload` on the `update_layer` fast path)
/// — a concurrent caller reloads and proceeds. The
/// same pattern and reasoning back
/// `update_layer_lua_recovers_from_script_flush_via_evalsha_reload` in
/// `tests/integration/storage/update_layer_lua_test.rs`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_probe_recovers_from_script_flush_via_evalsha_reload() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-9";

    let (session_id, object_id, snap_seq, stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-9",
        layer_name,
    )
    .await;

    // Partial loss: `obj:*` gone, stream intact — so a working probe has a
    // high-water to contribute and the two candidate floors are distinct.
    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;

    // Empty the script cache, standing in for the Valkey restart that
    // produced the state loss above. `script_flush(false)` is synchronous
    // on the server, so the next EVALSHA is guaranteed to see NOSCRIPT.
    let _: () = harness
        .tv
        .client()
        .pool
        .next()
        .script_flush(false)
        .await
        .expect("SCRIPT FLUSH must succeed");

    // Trigger recovery. The probe's EVALSHA now gets NOSCRIPT and must
    // SCRIPT LOAD + retry rather than folding to None.
    let _ = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect("post-wipe StatLayer MUST recover from checkpoint and return Ok");

    let header = read_obj_header(&harness, object_id).await;
    let recovered_seq = parse_header_u64(&header, "seq");

    assert_eq!(
        recovered_seq,
        stream_high_water + SMALL_SEQ_MARGIN,
        "b.8y5 R3: the high-water probe MUST recover from a flushed script cache via \
         evalsha_with_reload. A value of {} means the EVALSHA returned NOSCRIPT and the \
         probe folded to None — silently seeding below the surviving stream top \
         ({stream_high_water}) and wedging every subsequent UpdateLayer, in exactly the \
         post-restart scenario the probe exists for",
        snap_seq + SMALL_SEQ_MARGIN
    );

    // The probe succeeded, so no gate fired and the live stream must not
    // have been put on an expiry fuse.
    assert_eq!(
        read_key_ttl_ms(&harness, &stream_key).await,
        -1,
        "a successful probe must leave the surviving stream with no TTL"
    );
}

// =============================================================================
// Tests 10–16 — b.4et: the probe is fail-closed, and the stale-stream removal
// is verified.
// =============================================================================
//
// What b.4et changed, and therefore what every test below would do PRE-fix.
//
// `probe_stream_high_water` used to return `Option<u64>` and never an error.
// Four distinguishable situations all folded to `None`:
//
//   1. no stream key at all             (script answers nil)
//   2. a wrong-typed stream key         (script answers nil — the SAME reply)
//   3. an unparseable last-generated-id (a string we cannot read)
//   4. the EVALSHA itself failed        (a transient Valkey blip mid-restore)
//
// `None` seeds `obj:hdr.seq = snap_seq + band`. On situation 1 that is right.
// On 2–4 it is a header materialized UNDER a stream key whose top nobody ever
// read, and post-b.ph9 every subsequent `UpdateLayer` is then refused whole
// (`update_layer stream-id collision:`) until an operator clears the key — a
// write outage on a layer DR has just brought back.
//
// So, uniformly: pre-fix tests 10 and 14 would each have returned `Ok` from
// StatLayer with a header seeded at `snap_seq + margin`; tests 11 / 12 / 13
// would have taken the same `None` fold with no retry, no `TYPE` probe and no
// counter; test 15 would have UNLINKed the stream of a layer the SR-2.4 guard
// was about to refuse; and test 16's peer re-check did not exist, because
// there was no refusal to yield from. Each test names its own pre-fix shape
// again at the assertion that discriminates it.

// `RECOVERY_STREAM_PROBE_ATTEMPTS` — the `max_attempts` of the production
// `RECOVERY_STREAM_PROBE_RETRY` — lives in `super::helpers` (glob-imported
// above), alongside `RECOVERY_STREAM_ADJUDICATION_BUDGET`. It was mirrored
// here AND under a second name in `restore_fault_injection_matrix_test.rs`
// (`B4ET_PROBE_ATTEMPTS`); one mirror per production constant is the rule, and
// the helpers copy carries the mirror-not-import reasoning and names the unit
// test that pins the relationship.

// `stream_unresolved_metric` / `stream_unresolved_cells` /
// `assert_stream_unresolved_deltas` live in `super::helpers` (glob-imported
// above). They were byte-identical here and in
// `restore_fault_injection_matrix_test.rs`; the shared copy is the one both
// files now read, so a change to the family's vocabulary or to the
// flat-unless-expected contract lands in one place.

/// Assert the restore wrote NOTHING: `obj:hdr` absent, and no `obj:*` key
/// beyond the ones the test itself planted.
///
/// `obj:hdr` alone would be too weak. Apply's EXISTS short-circuit keys on the
/// header, so state written without it is both unreachable and unrepairable —
/// and the whole point of a refusal is that the apply never ran, which means
/// there must be no trace of it at all.
///
/// Gated with its callers: only a seam injection can produce a refusal, so
/// under the default feature set this is dead code.
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn assert_restore_wrote_nothing(
    harness: &CheckpointHarness,
    object_id: u64,
    planted: &[String],
    context: &str,
) {
    let header = read_obj_header(harness, object_id).await;
    assert!(
        header.is_empty(),
        "{context}: `obj:hdr` MUST be absent — a REFUSED restore never calls apply_restore, so \
         no header is ever seeded. Pre-b.4et this arm folded to `None` and seeded \
         `snap_seq + band` here, which is precisely the write outage b.4et closes. Got: {:?}",
        header.keys().collect::<Vec<_>>()
    );

    let pattern = format!("{}{{{object_id}}}:obj:*", harness.tv.key_schema().prefix());
    let client = harness.tv.client().pool.next();
    let mut stream: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<fred::types::Key, fred::error::Error>> + Send>,
    > = if client.is_clustered() {
        Box::pin(client.scan_cluster_buffered(
            pattern,
            Some(100),
            None::<fred::types::scan::ScanType>,
        ))
    } else {
        Box::pin(client.scan_buffered(pattern, Some(100), None::<fred::types::scan::ScanType>))
    };
    let mut unexpected = Vec::new();
    while let Some(item) = futures::StreamExt::next(&mut stream).await {
        // `expect`, not `if let Ok(..)`: a SCAN that errors mid-walk would
        // otherwise let this assertion pass having enumerated nothing, which is
        // exactly the green-but-vacuous shape a "no unplanted key exists" claim
        // cannot afford.
        let key = item.expect("SCAN of obj:* must succeed");
        let rendered: Option<&str> = key.as_str();
        let rendered = rendered.unwrap_or("<non-utf8 key>").to_string();
        if !planted.contains(&rendered) {
            unexpected.push(rendered);
        }
    }
    unexpected.sort();
    assert!(
        unexpected.is_empty(),
        "{context}: a refused restore MUST leave the object's keyspace exactly as it found it, \
         but {} unplanted `obj:*` key(s) appeared: {unexpected:?}",
        unexpected.len()
    );
}

/// Plant an entry whose ID sits far above anything the plausibility gate will
/// accept, and return that ID's sequence component.
///
/// `snap_seq + 2 x MAX_PLAUSIBLE_HIGH_WATER_GAP` — double the bound, and the
/// same shape [`seq_floor_ignores_implausible_stream_high_water`] uses, so the
/// gate's *refusal* is a given for every test below and what is under test is
/// what happens around it.
async fn plant_implausible_stream_top(
    harness: &CheckpointHarness,
    stream_key: &str,
    object_id: u64,
    snap_seq: u64,
) -> u64 {
    let implausible_seq = snap_seq + 2 * MAX_PLAUSIBLE_HIGH_WATER_GAP;
    xadd_entry(
        harness,
        stream_key,
        &format!("{implausible_seq}-0"),
        &format!("update|{object_id}|{implausible_seq}|{CLIENT_ID}|2100|2100"),
    )
    .await;
    assert_eq!(
        read_stream_last_generated_id(harness, stream_key).await,
        implausible_seq,
        "test pre-condition: the synthetic entry must be the stream's new high-water"
    );
    implausible_seq
}

/// Drive the production recovery entry point and require an `UNAVAILABLE`
/// naming `reason`.
///
/// The refusal's boxed source carries `recovery refused for object {oid}
/// ({reason}, N attempts): …`, which `ServiceError::from_recovery_error`
/// (b.33e) surfaces inside the status message. Matching on it keeps this
/// assertion pinned to the b.4et refusal rather than to any other
/// `UNAVAILABLE` the restore path can produce (an adapter blip, a wedged
/// fetch, the SR-3.4 seam).
///
/// Gated with its callers, for the same reason as
/// [`assert_restore_wrote_nothing`].
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn stat_expecting_refusal(
    harness: &CheckpointHarness,
    session_id: &str,
    layer_name: &str,
    reason: &str,
    context: &str,
) {
    let status = match handle_stat_layer(&harness.svc, stat_request(session_id, layer_name)).await {
        Err(status) => status,
        Ok(response) => panic!(
            "{context}: StatLayer MUST refuse the restore, never return Ok. Pre-b.4et it \
             returned Ok here with `obj:hdr` seeded below a stream top nobody read. Got: {:?}",
            response.into_inner()
        ),
    };
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "{context}: a b.4et refusal is `CheckpointError::Transient`, which \
         `ServiceError::from_recovery_error` maps to UNAVAILABLE — retriable, because the \
         evidence may be readable on the next attempt. A NOT_FOUND would mean the restore was \
         declared impossible; an INTERNAL would mean it was declared permanent. Got {:?}: {}",
        status.code(),
        status.message()
    );
    assert!(
        status.message().contains(reason),
        "{context}: the surfaced status MUST name the refusal reason ({reason}) so an operator \
         reading the client's error and the counter see the same word; got: {}",
        status.message()
    );
}

/// A fault injector that fails the first `fail_first` calls at ONE named op
/// with a `Transient`, then passes.
///
/// `Transient` is the retriable in-band signal: the hook maps it onto the
/// `ServiceError::Aborted` its retry driver converges on, so `fail_first = 2`
/// proves the budget is spent as documented and `fail_first = usize::MAX`
/// proves exhaustion refuses.
///
/// Calls at any OTHER op pass through untouched and are NOT counted — both
/// b.4et seams can fire on one restore (the probe's, then the `UNLINK`'s), so
/// an injector that counted both could not say which one it had been consulted
/// for.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct SeamInjector {
    target_op: &'static str,
    fail_first: usize,
    calls: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl SeamInjector {
    /// Fail every attempt at `target_op`.
    fn always(target_op: &'static str) -> Self {
        Self {
            target_op,
            fail_first: usize::MAX,
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Fail the first `n` attempts at `target_op`, then pass.
    fn failing_first(target_op: &'static str, n: usize) -> Self {
        Self {
            target_op,
            fail_first: n,
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// How many times the injector was consulted at `target_op`.
    fn calls(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::Acquire)
    }
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl live_edit::checkpoint::adapters::CheckpointFaultInjector for SeamInjector {
    fn maybe_inject(&self, op: &str) -> Result<(), live_edit::checkpoint::CheckpointError> {
        if op != self.target_op {
            return Ok(());
        }
        let seen = self.calls.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        if seen < self.fail_first {
            return Err(live_edit::checkpoint::CheckpointError::Transient {
                source: SEAM_FAULT_MARKER.into(),
            });
        }
        Ok(())
    }
}

/// Marker on every injected fault's boxed source below.
#[cfg(feature = "test-checkpoint-fault-injection")]
const SEAM_FAULT_MARKER: &str = "b4et-seam-fault";

/// Install `injector` on the harness's recovery hook, rebuilt from the
/// harness's own collaborators.
///
/// Always AFTER the preamble: the preamble drives real `CreateLayer` /
/// `UpdateLayer` traffic and a live checkpoint loop, none of which may see an
/// armed seam.
#[cfg(feature = "test-checkpoint-fault-injection")]
fn arm_seam(harness: &mut CheckpointHarness, injector: &std::sync::Arc<SeamInjector>) {
    let injector = std::sync::Arc::clone(injector);
    harness.install_recovery_hook(move |hook| {
        hook.with_restore_fault_injector_for_test(
            injector
                as std::sync::Arc<dyn live_edit::checkpoint::adapters::CheckpointFaultInjector>,
        )
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 10 — an unreadable probe over a SURVIVING stream refuses the restore.
// ─────────────────────────────────────────────────────────────────────────────

/// **The b.4et defect, end to end.** A transient Valkey failure during the
/// high-water probe, on a layer whose per-object stream survived the state
/// loss, must NOT be spent as "there is no stream".
///
/// Shape: split preamble (so `snap_seq` and the stream top are distinct) →
/// partial loss sparing the stream → plant a top the plausibility gate would
/// refuse if it could read it → fail EVERY probe attempt.
///
/// The planted top is what makes the pre-fix damage concrete rather than
/// theoretical. Pre-b.4et the injected `Err` folded to `None`, StatLayer
/// returned `Ok`, `obj:hdr.seq` was seeded at `snap_seq + margin` — far below
/// the surviving top — the stream was left in place (the `UNLINK` lived on the
/// plausibility branch, which an unreadable probe never reaches), and the very
/// next `UpdateLayer` was refused forever with `update_layer stream-id
/// collision:`. Every assertion below negates one step of that sequence: no
/// `Ok`, no header, no touched stream.
///
/// `..._refused_total` must stay FLAT: nothing was refused on plausibility and
/// no stream was thrown away. Reporting this on that counter would tell an
/// operator a stream had been cleared while one is in fact still sitting
/// there, unread.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_refuses_restore_when_probe_evidence_is_unobtainable() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-10";

    let (session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-10",
        layer_name,
    )
    .await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    let implausible_seq =
        plant_implausible_stream_top(&harness, &stream_key, object_id, snap_seq).await;

    let injector = std::sync::Arc::new(SeamInjector::always(
        live_edit::checkpoint::RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE,
    ));
    arm_seam(&mut harness, &injector);

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);

    stat_expecting_refusal(
        &harness,
        &session_id,
        layer_name,
        "probe_unreadable",
        "b.4et: an unreadable probe over a surviving stream",
    )
    .await;

    assert_eq!(
        injector.calls(),
        RECOVERY_STREAM_PROBE_ATTEMPTS,
        "b.4et: the probe MUST spend its whole bounded retry budget \
         ({RECOVERY_STREAM_PROBE_ATTEMPTS} attempts) before refusing. One call means the retry \
         is absent and a single reconnect-shaped blip now costs a restore; more than \
         {RECOVERY_STREAM_PROBE_ATTEMPTS} means the budget is unbounded and the refusal is paid \
         for under the per-object admission mutex"
    );

    assert_restore_wrote_nothing(
        &harness,
        object_id,
        std::slice::from_ref(&stream_key),
        "b.4et: after a probe_unreadable refusal",
    )
    .await;

    assert_eq!(
        read_stream_last_generated_id(&harness, &stream_key).await,
        implausible_seq,
        "b.4et: a refusal MUST leave the surviving stream exactly as it found it. We could not \
         read its top, so we have no basis for calling it stale — and unlinking a HEALTHY stream \
         on a transient blip throws away every in-window subscriber's catch-up"
    );
    assert_eq!(
        read_key_ttl_ms(&harness, &stream_key).await,
        -1,
        "b.4et: the stream must be present with no expiry (PTTL=-1). -2 means the refusal path \
         deleted it, which is the opposite of no-evidence-no-action"
    );

    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[("probe_unreadable", 1)],
        "b.4et: an unreadable probe over a live stream key",
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before,
        "b.4et: `..._refused_total` MUST stay flat. It counts a restore that LANDED after \
         verifiably discarding a stale stream; here nothing was discarded and nothing landed"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 11 — the probe retries a transient failure before giving up.
// ─────────────────────────────────────────────────────────────────────────────

/// The other half of test 10's budget claim: a blip that does NOT outlast the
/// budget must cost nothing at all.
///
/// Without this, "fail closed" would be indistinguishable from "fail on the
/// first hiccup", and the fix would trade a rare write outage for a common
/// restore outage — the fred reconnect a managed-proxy invalidation produces is
/// exactly a one-or-two-attempt failure.
///
/// Attempts 1 and 2 are injected `Transient`; attempt 3 runs for real, reads
/// the planted implausible top, the plausibility gate refuses it, the SR-2.4
/// identity guard authorizes the cleanup, and the verified `UNLINK` lands. So
/// the restore reaches the ordinary b.8y5 refusal outcome — floor back to
/// `snap_seq`, stale stream gone, layer writable — with no trace of the blip on
/// the unresolved counter.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_probe_retries_transient_failure_before_giving_up() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-11";

    let (session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-11",
        layer_name,
    )
    .await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    plant_implausible_stream_top(&harness, &stream_key, object_id, snap_seq).await;

    let injector = std::sync::Arc::new(SeamInjector::failing_first(
        live_edit::checkpoint::RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE,
        RECOVERY_STREAM_PROBE_ATTEMPTS - 1,
    ));
    arm_seam(&mut harness, &injector);

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);

    let _ = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect(
            "b.4et: a probe blip that clears inside the retry budget MUST NOT cost a restore. An \
             UNAVAILABLE here means the budget is being spent as if it were one attempt",
        );

    assert_eq!(
        injector.calls(),
        RECOVERY_STREAM_PROBE_ATTEMPTS,
        "the injector must have been consulted once per attempt: two failures and the successful \
         third. A count of {} would mean the retry stopped early and the success arm was never \
         reached",
        RECOVERY_STREAM_PROBE_ATTEMPTS - 1
    );

    let header = read_obj_header(&harness, object_id).await;
    let recovered_seq = parse_header_u64(&header, "seq");
    assert_eq!(
        recovered_seq,
        snap_seq + SMALL_SEQ_MARGIN,
        "the surviving third attempt read the planted implausible top, so the b.8y5 plausibility \
         gate refuses it and the floor falls back to snap_seq ({snap_seq})"
    );

    assert_stale_stream_unlinked(&harness, &stream_key).await;
    assert_restored_layer_accepts_a_commit(
        &harness,
        &session_id,
        layer_name,
        object_id,
        &stream_key,
        recovered_seq,
        11,
    )
    .await;

    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[],
        "b.4et: a transient blip absorbed by the retry budget",
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before + 1,
        "the eventual outcome is an ordinary b.8y5 plausibility refusal, so `..._refused_total` \
         ticks exactly as it does when the probe never blipped at all"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 12 — the fence: an ABSENT stream is safe even when the probe fails.
// ─────────────────────────────────────────────────────────────────────────────

/// **The fence that keeps b.4et from taking DR offline.**
///
/// The shape DR actually runs on is a full keyspace loss: the stream is gone
/// with everything else, and the checkpoint's `snap_seq + band` is the correct
/// and only available floor. A fix that refused whenever the probe failed would
/// convert every restore-during-a-Valkey-blip into an outage — on the majority
/// shape, not the rare one.
///
/// So the refusal is conditioned on POSITIVE evidence that the stream is there:
/// the `TYPE` round-trip. Here `TYPE` reads `none`, and the restore must
/// proceed byte-identically to pre-b.4et — including when the `EVALSHA` itself
/// failed, which is what the always-failing injector reproduces (a broken
/// script cache must not cost a restore we can prove is safe).
///
/// Nothing may tick: this is not a degraded restore, it is the normal one.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_probe_treats_absent_stream_as_safe_even_when_evalsha_fails() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-12";

    // `drive_preamble` ends in a FULL `obj:*` wipe — the stream goes with it,
    // which is the whole point of this arm.
    let (session_id, object_id, snap_seq, pre_wipe_max_seq) = drive_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-12",
        layer_name,
    )
    .await;
    let stream_key = harness.tv.key_schema().layer_stream(object_id);
    let stream_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&stream_key)
        .await
        .expect("EXISTS obj:stream must succeed");
    assert!(
        !stream_exists,
        "test pre-condition: `drive_preamble` must have taken the stream with the rest of \
         `obj:*`, or this arm is exercising the surviving-stream case instead"
    );

    let injector = std::sync::Arc::new(SeamInjector::always(
        live_edit::checkpoint::RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE,
    ));
    arm_seam(&mut harness, &injector);

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);

    let _ = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect(
            "b.4et: with the stream provably ABSENT, an unreadable probe MUST NOT block the \
             restore. An UNAVAILABLE here means the fix refuses on the shape a full keyspace \
             loss produces — i.e. it has taken DR offline on the majority case to close a \
             minority one",
        );

    assert_eq!(
        injector.calls(),
        RECOVERY_STREAM_PROBE_ATTEMPTS,
        "the probe still spends its budget before consulting TYPE — the retry is not skipped \
         just because the eventual verdict turns out benign"
    );

    let header = read_obj_header(&harness, object_id).await;
    let recovered_seq = parse_header_u64(&header, "seq");
    assert_eq!(
        recovered_seq,
        snap_seq + SMALL_SEQ_MARGIN,
        "the floor is snap_seq alone, exactly as in test 4 — an unreadable probe over an absent \
         key contributes nothing and costs nothing"
    );
    assert!(
        recovered_seq > pre_wipe_max_seq,
        "the guard band still clears the pre-wipe max ({pre_wipe_max_seq}); got {recovered_seq}"
    );

    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[],
        "b.4et: an unreadable probe over an ABSENT stream key",
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before,
        "nothing was refused and nothing was discarded on this arm"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 13 — a foreign value on the stream key is left alone and reported.
// ─────────────────────────────────────────────────────────────────────────────

/// A non-stream value on `le:{oid}:obj:stream` is NOT ours to delete, and NOT a
/// reason to refuse.
///
/// The probe script cannot tell this apart from an absent key — it answers
/// `nil` for both, and that conflation is half of b.4et. `TYPE` can, and the
/// two arms diverge: absent proceeds silently, foreign proceeds LOUDLY.
///
/// Proceeding is right for the restore itself: a non-stream key carries no
/// surviving high-water, so the guard band covers the RPO window exactly as it
/// does on a full wipe. Deleting the key is NOT: nothing in this service
/// retypes a key it owns, so the value belongs to an out-of-band writer, and
/// removing it would destroy evidence — and possibly someone else's data —
/// during a DR restore.
///
/// The consequence an operator must be told about is that the layer comes back
/// READABLE but not WRITABLE, so this test drives a real `UpdateLayer`
/// afterwards and pins the wedge as a KNOWN, REPORTED state rather than a
/// surprise: the commit is refused whole and non-retriably, and the counter
/// already carries `wrongtype_key`. The refusal's exact message is the storage
/// lane's ("Both also fail closed on a wrong-typed key they are about to
/// write") and is deliberately not asserted here — see the comment at that
/// assertion.
///
/// Pre-b.4et this arm was completely silent: `nil` → `None` → restore at
/// `snap_seq + margin`, no counter, no WARN, and the first client to write
/// discovered it.
///
/// No injector — the production `nil` is reached honestly, by handing the
/// script a key it cannot read a stream out of.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_leaves_a_wrongtype_stream_key_in_place_and_reports_it() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-13";

    let (session_id, object_id, snap_seq, _pre_wipe_max_seq) = drive_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-13",
        layer_name,
    )
    .await;

    // The wipe took the stream; put a foreign STRING on its key, the way an
    // out-of-band writer on the `le:` keyspace would.
    let stream_key = harness.tv.key_schema().layer_stream(object_id);
    let _: () = harness
        .tv
        .client()
        .pool
        .set(&stream_key, "foreign", None, None, false)
        .await
        .expect("SET of the foreign value on the stream key must succeed in test setup");

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);

    let _ = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect(
            "b.4et: a foreign value on the stream key MUST NOT refuse the restore — it carries \
             no surviving high-water, so the guard band covers the RPO window exactly as on a \
             full wipe, and the layer is still worth bringing back READABLE",
        );

    let header = read_obj_header(&harness, object_id).await;
    assert_eq!(
        parse_header_u64(&header, "seq"),
        snap_seq + SMALL_SEQ_MARGIN,
        "the floor is snap_seq alone — there is no high-water on a string"
    );

    // The key is left exactly as found. This is the assertion that separates
    // `wrongtype_key` from the plausibility gate's arm, which DOES delete.
    let key_type: String = harness
        .tv
        .client()
        .pool
        .r#type(&stream_key)
        .await
        .expect("TYPE obj:stream must succeed");
    assert_eq!(
        key_type, "string",
        "b.4et: the foreign value MUST still be there, untouched and still a string. Nothing in \
         this service retypes a key it owns, so this value is an out-of-band writer's — deleting \
         it during a DR restore would destroy the evidence an operator needs, and possibly \
         someone else's data"
    );

    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[("wrongtype_key", 1)],
        "b.4et: a foreign value on the per-object stream key",
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before,
        "`..._refused_total` is the plausibility gate's counter and no plausibility verdict was \
         reached here — there was no ID to judge"
    );

    // The wedge, pinned as a KNOWN state rather than a surprise. This is why
    // the counter has to exist at all: the restore reports success, and only
    // this counter (plus its WARN) distinguishes "came back healthy" from
    // "came back unwritable".
    let status = try_issue_update(
        &harness,
        &session_id,
        layer_name,
        object_id,
        field_update_delta(13, 0xD1),
        2_013,
    )
    .await
    .expect_err(
        "a commit against a layer whose stream key holds a foreign value MUST fail closed. \
         Success would mean the write path silently retyped or clobbered a key it does not own",
    );
    // The claim is FAIL-CLOSED, and nothing narrower. Which of the commit
    // path's refusal messages a wrong-typed stream key produces is the storage
    // lane's behaviour ("Both also fail closed on a wrong-typed key they are
    // about to write" — CLAUDE.md, SDD §7.5.4/§7.5.5) and is pinned by that
    // lane's own tests; asserting the message shape from here would couple a
    // recovery-lane test to a commit-path decision it does not own, and would
    // go red on a message the storage lane is free to change.
    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "the commit MUST be refused whole and non-retriably (INTERNAL). An OK would mean the \
         write path silently retyped or clobbered a key it does not own; an UNAVAILABLE would \
         advertise a retry that can never succeed. Got {:?}: {}",
        status.code(),
        status.message()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 14 — an UNVERIFIABLE stale-stream UNLINK refuses the restore.
// ─────────────────────────────────────────────────────────────────────────────

/// The plausibility gate's `UNLINK` is the other half of b.4et: pre-fix it was
/// best-effort — the reply was bound as `()`, so the call could not tell
/// "deleted it" from "there was nothing there", and any error was logged and
/// swallowed while the restore ran on regardless.
///
/// That is the same defect as the unreadable probe, one step later: a header
/// seeded at `snap_seq + margin` underneath a stale top that is still sitting
/// there. Here the probe SUCCEEDS and reads the planted implausible top, the
/// gate refuses it, the identity guard authorizes the cleanup — and every
/// `UNLINK` attempt fails.
///
/// Pre-fix: `Ok` from StatLayer, header at `snap_seq + margin`, stale stream
/// still present, first commit wedged. Post-fix all three of those flip, and
/// the counter says which arm refused — `unlink_unverified`, NOT
/// `probe_unreadable`: the probe worked fine, it is the cleanup that did not,
/// and the two send an operator at different remediations.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_refuses_restore_when_the_stale_stream_unlink_cannot_be_verified() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-14";

    let (session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-14",
        layer_name,
    )
    .await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    let implausible_seq =
        plant_implausible_stream_top(&harness, &stream_key, object_id, snap_seq).await;

    let injector = std::sync::Arc::new(SeamInjector::always(
        live_edit::checkpoint::RECOVERY_SEAM_STALE_STREAM_UNLINK,
    ));
    arm_seam(&mut harness, &injector);

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);

    stat_expecting_refusal(
        &harness,
        &session_id,
        layer_name,
        "unlink_unverified",
        "b.4et: a stale-stream UNLINK that could not be verified",
    )
    .await;

    assert_eq!(
        injector.calls(),
        RECOVERY_STREAM_PROBE_ATTEMPTS,
        "the UNLINK gets the same bounded retry budget as the probe — a removal that fails once \
         must not cost a restore"
    );

    assert_restore_wrote_nothing(
        &harness,
        object_id,
        std::slice::from_ref(&stream_key),
        "b.4et: after an unlink_unverified refusal",
    )
    .await;

    assert_eq!(
        read_stream_last_generated_id(&harness, &stream_key).await,
        implausible_seq,
        "the stale stream MUST still be there — that is precisely why the restore was refused. \
         If it were gone the refusal would be a bug rather than a guard"
    );

    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[("unlink_unverified", 1)],
        "b.4et: an unverifiable stale-stream removal",
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before,
        "`..._refused_total` counts a stream that was verifiably DISCARDED; this one survived, \
         so ticking it would over-report cleanups the fleet never performed"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 15 — no identity, no delete.
// ─────────────────────────────────────────────────────────────────────────────

/// The stale-stream removal is authorized by the SR-2.4 identity guard BEFORE
/// it runs, not merely followed by it.
///
/// `apply_restore` runs that guard too — but AFTER this point, so pre-b.4et the
/// `UNLINK` fired first and destroyed the stream of a layer the guard was about
/// to refuse. Not hypothetical: "header absent, stream present" is the NORMAL
/// state for up to `STREAM_DELETE_GRACE_MS` (30 s) on every `DeleteLayer`,
/// whose retirement order is DEL mapping → XADD `deleted` → UNLINK header →
/// PEXPIRE stream. A recovery racing a delete would blow away the graced stream
/// — the very thing that lets a parked subscriber see the `deleted` sentinel
/// and exit cleanly — for a restore that then returns `NotFound` anyway.
///
/// Modelled here by deleting the layer mapping, which is exactly what the guard
/// consults. The hook is driven directly: with no mapping there is nothing for
/// `handle_stat_layer` to resolve, so the handler would answer NOT_FOUND before
/// recovery ever fired and the arm would be vacuous.
///
/// This is a *proceed* arm, not a refusal: nothing about the stream was left
/// unresolved, we simply had no business touching it. So no counter moves, and
/// `apply_restore` reaches its existing `NotFound` verdict on its own.
///
/// No injector — the guard is driven by real keyspace state.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_does_not_unlink_a_stream_whose_layer_identity_no_longer_resolves() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-15";

    let (session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-15",
        layer_name,
    )
    .await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    let implausible_seq =
        plant_implausible_stream_top(&harness, &stream_key, object_id, snap_seq).await;

    // Retire the identity the guard consults, the way DeleteLayer's first step
    // does.
    let mapping_key = harness
        .tv
        .key_schema()
        .layer_mapping(&session_id, layer_name);
    let removed: i64 = harness
        .tv
        .client()
        .pool
        .del(&mapping_key)
        .await
        .expect("DEL of the layer mapping must succeed in test setup");
    assert_eq!(
        removed, 1,
        "test pre-condition: the layer mapping must have existed and been removed"
    );

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);

    let outcome = harness.svc.recovery.try_restore_layer(object_id).await;
    assert!(
        matches!(outcome, RecoveryOutcome::NotRecoverable),
        "with no live mapping resolving to this object_id, apply's SR-2.4 guard refuses and the \
         hook reports NotRecoverable. Got {outcome:?}"
    );

    assert_eq!(
        read_stream_last_generated_id(&harness, &stream_key).await,
        implausible_seq,
        "b.4et item 6: the stream MUST survive. Pre-fix the UNLINK ran before the identity guard \
         and destroyed it — during a DeleteLayer's 30s grace window that stream is what lets a \
         parked subscriber see the `deleted` sentinel and exit cleanly, and the restore that \
         justified destroying it never happened"
    );
    assert_eq!(
        read_key_ttl_ms(&harness, &stream_key).await,
        -1,
        "the stream must be untouched, not put on a fuse"
    );

    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[],
        "b.4et: a stale top on a layer whose identity no longer resolves",
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before,
        "no stream was discarded, so the refusal counter must not move either"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 16 — a refusal yields to a peer that already restored.
// ─────────────────────────────────────────────────────────────────────────────

/// A fault injector that, on its first consultation at the probe seam, writes
/// `obj:hdr` back into Valkey — standing in for a peer replica that landed this
/// same recovery while we were spending retries on a stream we could not read —
/// and then fails every attempt.
///
/// # Why it writes from a sync callback
///
/// The seam has to fire BETWEEN the hook's step-4 `EXISTS obj:hdr` re-check and
/// the probe's refusal, because that is the only window in which the re-check
/// inside `refuse_restore` can observe something step 4 did not. Seeding the
/// header before the call instead would short-circuit at step 4 and never reach
/// the probe, making the arm vacuous.
///
/// `maybe_inject` is synchronous, so the write goes through `block_in_place` +
/// `Handle::block_on` — the sanctioned tokio pattern for blocking work inside a
/// `flavor = "multi_thread"` runtime. That makes the ordering DETERMINISTIC
/// (the header is durable before the injected error is returned) rather than a
/// sleep-and-hope race, which is what a `seq_floor_*` test under cluster-CI
/// load cannot afford.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct PeerRestoresDuringProbeInjector {
    client: live_edit::storage::client::ValkeyClient,
    header_key: String,
    header_fields: Vec<(String, String)>,
    seeded: std::sync::atomic::AtomicBool,
    calls: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl live_edit::checkpoint::adapters::CheckpointFaultInjector for PeerRestoresDuringProbeInjector {
    fn maybe_inject(&self, op: &str) -> Result<(), live_edit::checkpoint::CheckpointError> {
        if op != live_edit::checkpoint::RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE {
            return Ok(());
        }
        self.calls.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        if !self.seeded.swap(true, std::sync::atomic::Ordering::AcqRel) {
            let pool = self.client.pool.clone();
            let key = self.header_key.clone();
            let fields = self.header_fields.clone();
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    let _: () = fred::interfaces::HashesInterface::hset(&pool, &key, fields)
                        .await
                        .expect("the simulated peer's obj:hdr write must succeed");
                })
            });
        }
        Err(live_edit::checkpoint::CheckpointError::Transient {
            source: SEAM_FAULT_MARKER.into(),
        })
    }
}

/// A refusal is not allowed to turn a COMPLETED recovery into an `UNAVAILABLE`.
///
/// The probe runs before `apply_restore`'s own `EXISTS obj:hdr` short-circuit,
/// so on the refusal path the hook has not yet consulted the one condition that
/// makes the whole restore moot. A peer replica racing the same recovery may
/// have landed it while we burned three retries on an unreadable stream — and
/// its restore did whatever needed doing to that stream. Refusing then would
/// report a live, restored layer as unavailable.
///
/// So every refusal re-checks the header first and, on a hit, returns the
/// no-high-water answer: the caller proceeds to `apply_restore`, whose EXISTS
/// short-circuit answers `AlreadyPresent`, which the hook already maps to
/// `Restored`. Nothing is written on that path, so the `None` floor is never
/// spent — which is what lets the guard reuse the existing return shape instead
/// of adding a fourth `RecoveryOutcome`.
///
/// Pre-b.4et there was no refusal for this to yield from, so the arm has no
/// pre-fix analogue at all: it exists to bound the cost of the new one.
///
/// Driven at the hook rather than through `handle_stat_layer` so the assertion
/// is on `RecoveryOutcome` itself — the handler folds `Restored` and a plain
/// header hit into the same `Ok` and could not tell them apart.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seq_floor_refusal_yields_to_a_peer_that_already_restored() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-16";

    let (_session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-16",
        layer_name,
    )
    .await;

    // Capture the live header BEFORE the wipe: that is what the "peer" puts
    // back, so this test converges on a real restored header rather than a
    // synthetic key that merely satisfies EXISTS.
    let peer_header: Vec<(String, String)> = read_obj_header(&harness, object_id)
        .await
        .into_iter()
        .collect();
    assert!(
        !peer_header.is_empty(),
        "test pre-condition: the live header must be non-empty before the wipe"
    );

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    let implausible_seq =
        plant_implausible_stream_top(&harness, &stream_key, object_id, snap_seq).await;

    let injector = std::sync::Arc::new(PeerRestoresDuringProbeInjector {
        client: harness.tv.client().clone(),
        header_key: harness.tv.key_schema().obj_header(&object_id.to_string()),
        header_fields: peer_header,
        seeded: std::sync::atomic::AtomicBool::new(false),
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    harness.install_recovery_hook({
        let injector = std::sync::Arc::clone(&injector);
        move |hook| {
            hook.with_restore_fault_injector_for_test(
                injector
                    as std::sync::Arc<dyn live_edit::checkpoint::adapters::CheckpointFaultInjector>,
            )
        }
    });

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);

    let outcome = harness.svc.recovery.try_restore_layer(object_id).await;
    assert!(
        matches!(outcome, RecoveryOutcome::Restored),
        "b.4et: the refusal MUST yield to the peer that landed the restore while we were \
         probing. An Error here means a completed recovery was reported UNAVAILABLE to a client \
         whose layer is in fact live. Got {outcome:?}"
    );

    assert_eq!(
        injector.calls.load(std::sync::atomic::Ordering::Acquire),
        RECOVERY_STREAM_PROBE_ATTEMPTS,
        "test pre-condition: the probe must actually have been driven to exhaustion, i.e. this \
         arm reached `refuse_restore` and was turned around by ITS header re-check — not \
         short-circuited at the hook's step-4 re-check, which would make the arm vacuous"
    );

    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[],
        "b.4et: a refusal that yielded to a peer's completed restore",
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before,
        "nothing was discarded on this arm either"
    );

    assert_eq!(
        read_stream_last_generated_id(&harness, &stream_key).await,
        implausible_seq,
        "the yielding path writes nothing and deletes nothing — the peer's restore owns whatever \
         needed doing to this stream"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests 17–18 — the `TYPE` round-trip is itself evidence, and can itself fail.
// ─────────────────────────────────────────────────────────────────────────────
//
// `TYPE` is what un-conflates the probe script's single `nil`, so it is the
// load-bearing round-trip of the whole b.4et adjudication — and it is reached
// from TWO places that must not be folded together:
//
//   * `classify_unreadable_stream`, deciding what an unreadable probe means
//     → a failure there is `probe_unreadable`;
//   * `unlink_stale_stream_once`'s post-`UNLINK` confirmation, reached only
//     when the `UNLINK` reported deleting nothing
//     → a failure there is `unlink_unverified`.
//
// Both go through the same one-attempt helper and therefore the same seam, so
// the pair below is what pins that ONE injected fault produces the RIGHT reason
// on each path. Getting that wrong is not cosmetic: the two remediations are
// "check Valkey health" and "find whatever is re-creating this stream key out
// of band".

/// Probe seam → retriable `Transient` forever; `TYPE` seam → TERMINAL.
///
/// The pairing is what makes the classification `TYPE` reachable at all: the
/// classification only runs once the probe has failed to produce a plausible
/// sequence, so an arm that wants to fail the `TYPE` must first fail the probe.
///
/// The `TYPE` injection is `Permanent`, not `Transient`, precisely so the
/// attempt count is discriminating: a terminal kind must stop the retry dead,
/// exactly as a non-transient fred error does.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct UnreadableProbeAndTypeInjector {
    probe_calls: std::sync::atomic::AtomicUsize,
    type_calls: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl live_edit::checkpoint::adapters::CheckpointFaultInjector for UnreadableProbeAndTypeInjector {
    fn maybe_inject(&self, op: &str) -> Result<(), live_edit::checkpoint::CheckpointError> {
        if op == live_edit::checkpoint::RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE {
            self.probe_calls
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            return Err(live_edit::checkpoint::CheckpointError::Transient {
                source: SEAM_FAULT_MARKER.into(),
            });
        }
        if op == live_edit::checkpoint::RECOVERY_SEAM_STREAM_KEY_TYPE {
            self.type_calls
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            return Err(live_edit::checkpoint::CheckpointError::Permanent {
                source: SEAM_FAULT_MARKER.into(),
            });
        }
        Ok(())
    }
}

/// The `TYPE` round-trip that decides what an unreadable probe MEANS can itself
/// be unreadable — and that is a refusal, not a fold to `None`.
///
/// This is the arm that closes b.4et's own tail: the fix replaces "the probe
/// failed, assume no stream" with "ask the key", and if the asking also fails
/// we are back to knowing nothing. Folding THAT to `Ok(None)` would reintroduce
/// the exact defect one round-trip later — a header seeded under a stream key
/// whose very existence is unestablished.
///
/// The reason is `probe_unreadable`, shared with the plain unreadable-probe arm
/// (test 10) on purpose: the operator situation is identical ("Valkey did not
/// answer"), and the remediation is identical. The detail string is where the
/// two are separated, and it names both failures.
///
/// Pre-b.4et there was no `TYPE` round-trip at all, so this arm has no pre-fix
/// analogue: it exists to keep the fix from having a soft edge.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_refuses_when_the_key_type_classification_cannot_be_read() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-17";

    let (session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-17",
        layer_name,
    )
    .await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    let implausible_seq =
        plant_implausible_stream_top(&harness, &stream_key, object_id, snap_seq).await;

    let injector = std::sync::Arc::new(UnreadableProbeAndTypeInjector {
        probe_calls: std::sync::atomic::AtomicUsize::new(0),
        type_calls: std::sync::atomic::AtomicUsize::new(0),
    });
    harness.install_recovery_hook({
        let injector = std::sync::Arc::clone(&injector);
        move |hook| {
            hook.with_restore_fault_injector_for_test(
                injector
                    as std::sync::Arc<dyn live_edit::checkpoint::adapters::CheckpointFaultInjector>,
            )
        }
    });

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);

    stat_expecting_refusal(
        &harness,
        &session_id,
        layer_name,
        "probe_unreadable",
        "b.4et: the TYPE classification could not be read either",
    )
    .await;

    assert_eq!(
        injector
            .probe_calls
            .load(std::sync::atomic::Ordering::Acquire),
        RECOVERY_STREAM_PROBE_ATTEMPTS,
        "test pre-condition: the probe must have been driven to exhaustion, which is what makes \
         the classification TYPE reachable at all"
    );
    assert_eq!(
        injector
            .type_calls
            .load(std::sync::atomic::Ordering::Acquire),
        1,
        "a TERMINAL injected fault MUST stop the TYPE retry dead — the budget is for transient \
         failures, and re-running a round-trip that failed permanently only lengthens the \
         per-object admission-mutex hold for a verdict that cannot change"
    );

    assert_restore_wrote_nothing(
        &harness,
        object_id,
        std::slice::from_ref(&stream_key),
        "b.4et: after a classification-TYPE refusal",
    )
    .await;
    assert_eq!(
        read_stream_last_generated_id(&harness, &stream_key).await,
        implausible_seq,
        "we could not even learn what the key is, so we have no basis for touching it"
    );

    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[("probe_unreadable", 1)],
        "b.4et: a TYPE classification that could not be read",
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before,
        "nothing was discarded on this arm"
    );
}

/// `UNLINK` seam → delete the key out from under the command, then pass;
/// `TYPE` seam → TERMINAL.
///
/// This is the only way to reach the post-`UNLINK` confirmation deterministically:
/// that `TYPE` runs if and only if the `UNLINK` reported deleting NOTHING, which
/// in production means a peer replica cleared the key in the window between our
/// probe and our removal. Deleting it from inside the seam — which fires
/// immediately before the `UNLINK` goes on the wire — reproduces that window
/// exactly, with no sleep and no race.
///
/// The write goes through `block_in_place` + `Handle::block_on` for the same
/// reason [`PeerRestoresDuringProbeInjector`] does: `maybe_inject` is sync, and
/// the ordering has to be a fact rather than a hope.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct VanishingStreamInjector {
    client: live_edit::storage::client::ValkeyClient,
    stream_key: String,
    unlink_calls: std::sync::atomic::AtomicUsize,
    type_calls: std::sync::atomic::AtomicUsize,
    vanished: std::sync::atomic::AtomicBool,
    /// `true` — the confirming `TYPE` fails terminally (test 18: the removal
    /// cannot be observed, so it is not verified). `false` — the `TYPE` is
    /// merely COUNTED and runs for real against a key the injector has already
    /// removed, so it reads `none` (test 19: the removal is verified and the
    /// restore proceeds).
    ///
    /// One injector for both halves on purpose: the two arms must differ in
    /// exactly one thing — whether the confirmation could be read — and a
    /// second injector type would let the rest of the staging drift apart.
    fail_type_confirmation: bool,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl live_edit::checkpoint::adapters::CheckpointFaultInjector for VanishingStreamInjector {
    fn maybe_inject(&self, op: &str) -> Result<(), live_edit::checkpoint::CheckpointError> {
        if op == live_edit::checkpoint::RECOVERY_SEAM_STALE_STREAM_UNLINK {
            self.unlink_calls
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            if !self
                .vanished
                .swap(true, std::sync::atomic::Ordering::AcqRel)
            {
                let pool = self.client.pool.clone();
                let key = self.stream_key.clone();
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        let _: i64 = fred::interfaces::KeysInterface::del(&pool, &key)
                            .await
                            .expect("the simulated peer's stream removal must succeed");
                    })
                });
            }
            return Ok(());
        }
        if op == live_edit::checkpoint::RECOVERY_SEAM_STREAM_KEY_TYPE {
            self.type_calls
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            if self.fail_type_confirmation {
                return Err(live_edit::checkpoint::CheckpointError::Permanent {
                    source: SEAM_FAULT_MARKER.into(),
                });
            }
        }
        Ok(())
    }
}

/// An `UNLINK` that reports deleting nothing is NOT self-evidently a success,
/// and when the confirming `TYPE` cannot be read the removal is unverified —
/// which refuses, and refuses as `unlink_unverified`.
///
/// The zero-count case is real: a peer replica racing the same recovery may
/// legitimately have cleared the key first, and that IS a verified removal.
/// What distinguishes it from the failure is the confirming `TYPE` reading
/// `none`. Take that reading away and the two are indistinguishable — which is
/// precisely the pre-b.4et posture, where the reply was bound as `()` and
/// "deleted it" and "there was nothing there" were the same observation.
///
/// The discriminating assertion is the REASON. The failing round-trip here is a
/// `TYPE`, exactly as in test 17, but it is reached from the cleanup path — so
/// reporting it as `probe_unreadable` would send an operator at Valkey health
/// when the actionable finding is a stream key being re-created out of band.
/// One seam, two paths, two reasons.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seq_floor_refuses_when_the_post_unlink_type_confirmation_cannot_be_read() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-18";

    let (session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-18",
        layer_name,
    )
    .await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    plant_implausible_stream_top(&harness, &stream_key, object_id, snap_seq).await;

    let injector = std::sync::Arc::new(VanishingStreamInjector {
        client: harness.tv.client().clone(),
        stream_key: stream_key.clone(),
        unlink_calls: std::sync::atomic::AtomicUsize::new(0),
        type_calls: std::sync::atomic::AtomicUsize::new(0),
        vanished: std::sync::atomic::AtomicBool::new(false),
        fail_type_confirmation: true,
    });
    harness.install_recovery_hook({
        let injector = std::sync::Arc::clone(&injector);
        move |hook| {
            hook.with_restore_fault_injector_for_test(
                injector
                    as std::sync::Arc<dyn live_edit::checkpoint::adapters::CheckpointFaultInjector>,
            )
        }
    });

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);

    stat_expecting_refusal(
        &harness,
        &session_id,
        layer_name,
        "unlink_unverified",
        "b.4et: an UNLINK that deleted nothing, with an unreadable confirmation",
    )
    .await;

    assert_eq!(
        injector
            .unlink_calls
            .load(std::sync::atomic::Ordering::Acquire),
        1,
        "test pre-condition: the plausibility gate must have reached the UNLINK exactly once — a \
         terminal failure inside the attempt stops the retry dead"
    );
    assert_eq!(
        injector
            .type_calls
            .load(std::sync::atomic::Ordering::Acquire),
        1,
        "the confirming TYPE MUST have run: it is reached if and only if the UNLINK reported \
         deleting nothing, which is the whole condition this arm stages. A zero here means the \
         zero-count reply was taken as success without confirmation — the pre-b.4et posture"
    );

    assert_restore_wrote_nothing(
        &harness,
        object_id,
        std::slice::from_ref(&stream_key),
        "b.4et: after an unverifiable post-UNLINK confirmation",
    )
    .await;

    // The key really is gone here — the injector removed it, standing in for the
    // peer replica that would have. That is NOT the assertion: the point is that
    // we could not OBSERVE that it was gone, and an unobserved removal is not a
    // verified one, so the restore is refused regardless of the durable truth.
    let survives: bool = harness
        .tv
        .client()
        .pool
        .exists(&stream_key)
        .await
        .expect("EXISTS obj:stream must succeed");
    assert!(
        !survives,
        "test pre-condition: the injector must have removed the key out from under the UNLINK, \
         or the reply would have been 1 and the confirming TYPE unreachable"
    );

    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[("unlink_unverified", 1)],
        "b.4et: a TYPE failure on the CLEANUP path, which must not be reported as \
         probe_unreadable — the probe worked, and the two reasons carry different remediations",
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before,
        "`..._refused_total` marks a VERIFIED discard; this removal is exactly what could not be \
         verified"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 19 — the success half: a peer-cleared stream IS a verified removal.
// ─────────────────────────────────────────────────────────────────────────────

/// The other half of test 18's staged race: an `UNLINK` that reported deleting
/// nothing, with the confirming `TYPE` readable, IS a verified removal — so the
/// restore PROCEEDS.
///
/// Same injector, same window (the key is deleted out from under the `UNLINK` by
/// a stand-in peer replica, immediately before the command goes on the wire),
/// one difference: nothing fails the confirmation, so the `TYPE` runs for real
/// and reads `none`. Everything downstream follows from that single reading —
/// the removal counts as verified, `..._refused_total` ticks the ordinary
/// plausibility refusal, the apply materializes the layer at
/// `snap_seq + margin`, and the first commit on top succeeds.
///
/// # What this guards that test 18 cannot
///
/// `unlink_stale_stream_once` ends in `Ok(matches!(type_reading, Absent))`. Test
/// 18 pins the `false` side of that expression (an unreadable `TYPE` is not a
/// verification); only this arm pins the `true` side. A regression to a bare
/// `Ok(false)` — "a zero count is never a verification" — leaves test 18 green
/// and turns the ordinary peer-cleared-it-first race into a permanent
/// `unlink_unverified` refusal: a restore declined because a peer did the very
/// thing we were about to do. That is a DR-availability defect whose only
/// symptom is an `UNAVAILABLE`, and this is the one test that sees it.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seq_floor_treats_a_peer_cleared_stream_as_a_verified_removal_and_restores() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-19";

    let (session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-19",
        layer_name,
    )
    .await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    plant_implausible_stream_top(&harness, &stream_key, object_id, snap_seq).await;

    let injector = std::sync::Arc::new(VanishingStreamInjector {
        client: harness.tv.client().clone(),
        stream_key: stream_key.clone(),
        unlink_calls: std::sync::atomic::AtomicUsize::new(0),
        type_calls: std::sync::atomic::AtomicUsize::new(0),
        vanished: std::sync::atomic::AtomicBool::new(false),
        fail_type_confirmation: false,
    });
    harness.install_recovery_hook({
        let injector = std::sync::Arc::clone(&injector);
        move |hook| {
            hook.with_restore_fault_injector_for_test(
                injector
                    as std::sync::Arc<dyn live_edit::checkpoint::adapters::CheckpointFaultInjector>,
            )
        }
    });

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);

    let _ = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect(
            "b.4et: an UNLINK that deleted nothing, CONFIRMED absent by the TYPE re-probe, is a \
             verified removal and the restore MUST proceed. An UNAVAILABLE here means the \
             zero-count reply is being read as an unverifiable removal — which declines a restore \
             because a peer replica cleared the stale stream first, exactly the race the \
             confirmation exists to accept",
        );

    assert_eq!(
        injector
            .unlink_calls
            .load(std::sync::atomic::Ordering::Acquire),
        1,
        "test pre-condition: the plausibility gate must have reached the UNLINK exactly once — a \
         verified removal is not retried"
    );
    assert_eq!(
        injector
            .type_calls
            .load(std::sync::atomic::Ordering::Acquire),
        1,
        "the confirming TYPE MUST have run exactly once: it is reached if and only if the UNLINK \
         reported deleting nothing, which is the condition this arm stages. A zero here would \
         mean the zero-count reply was taken as success WITHOUT confirmation — the pre-b.4et \
         posture, in which `deleted it` and `there was nothing there` are the same observation"
    );

    let header = read_obj_header(&harness, object_id).await;
    let recovered_seq = parse_header_u64(&header, "seq");
    assert_eq!(
        recovered_seq,
        snap_seq + SMALL_SEQ_MARGIN,
        "the probe read the planted implausible top, so the plausibility gate refuses it and the \
         floor falls back to snap_seq ({snap_seq}) — identical to the arm where our own UNLINK did \
         the removing"
    );

    assert_stale_stream_unlinked(&harness, &stream_key).await;
    assert_restored_layer_accepts_a_commit(
        &harness,
        &session_id,
        layer_name,
        object_id,
        &stream_key,
        recovered_seq,
        19,
    )
    .await;

    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[],
        "b.4et: a peer-cleared stale stream, confirmed absent",
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before + 1,
        "`..._refused_total` MUST tick: this is the ordinary plausibility refusal — evidence read, \
         top judged stale, stream verifiably gone, restore landed. WHO removed the key does not \
         change what was accomplished"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests 20–21 — the adjudication's overall wall-clock budget, and its grace.
// ─────────────────────────────────────────────────────────────────────────────
//
// Every arm above fails a round-trip and lets the bounded retry decide. That
// bounds BACKOFF SLEEP — tens of milliseconds — and nothing else: a store that
// accepts a command and never answers costs a full
// `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` per attempt, so on attempt counts alone
// the adjudication admits a ~35 s (refuse path) to ~65 s (UNLINK path) hold of
// the K=1 per-`object_id` admission mutex — inline on an RPC exempt from the
// tower deadline, with every other reader of that layer queued behind it. The
// single `RECOVERY_STREAM_ADJUDICATION_BUDGET` deadline over the whole sequence
// is what makes the latency claim true, and these two tests are what make that
// deadline more than a line of rustdoc.
//
// Cost: ≈11 s of wall clock each, spent in ONE deliberate stall. That is the
// honest price of the arm — a deadline cannot be witnessed without letting it
// expire — and it stays well inside nextest's 40 s slow mark, so no
// `.config/nextest.toml` override is needed. Runner load pushes TOWARD the
// asserted outcome (a loaded box makes the deadline MORE likely to expire, not
// less), so this pair does not carry the usual timing-assertion flakiness:
// nothing here asserts that anything was fast.

// The overall deadline this pair races — `RECOVERY_STREAM_ADJUDICATION_BUDGET`
// — is the shared mirror in `super::helpers` (glob-imported above), beside
// `RECOVERY_STREAM_PROBE_ATTEMPTS`. Its relationship to the default command
// timeout, and the `budget + grace` ceiling SDD §22.5 step 6 records, are
// pinned by `recovery.rs`'s own
// `the_stream_adjudication_budgets_stay_within_their_documented_bounds`.

/// How long [`StallingProbeInjector`] parks the probe for.
///
/// The whole budget plus a margin, so the deadline is provably in the past by
/// the time the adjudication next yields — which it does at the real `EVALSHA`
/// the injector waves through after the stall. Sizing it at exactly the budget
/// would make the arm a coin flip on scheduler timing.
#[cfg(feature = "test-checkpoint-fault-injection")]
const ADJUDICATION_STALL: Duration =
    RECOVERY_STREAM_ADJUDICATION_BUDGET.saturating_add(Duration::from_millis(500));

/// A fault injector that parks the first high-water probe attempt past the
/// adjudication's overall deadline, and optionally seeds `obj:hdr` — a peer
/// replica landing this same recovery — before it does.
///
/// # Why a stall and not an error
///
/// Every other injector in this file fails a round-trip, which the bounded retry
/// then absorbs or spends. The budget-expiry arm is the one verdict a retry
/// count cannot produce: it is reached only when wall clock runs out, which
/// needs a round-trip that neither succeeds nor fails for longer than the
/// deadline. The stall reproduces exactly the production condition the deadline
/// exists for — a store that took the command and went quiet.
///
/// # Why the stall is DETERMINISTIC rather than a race
///
/// `maybe_inject` is synchronous, so the sleep goes through `block_in_place` +
/// `Handle::block_on` — the same sanctioned pattern
/// [`PeerRestoresDuringProbeInjector`] and [`VanishingStreamInjector`] use for
/// their writes. The injected call therefore does not return until the budget is
/// provably spent, and the very next `.await` inside the adjudication (the real
/// `EVALSHA` this injector waves through) hands control to the
/// `tokio::time::timeout`, which finds its deadline already elapsed. No
/// sleep-and-hope, and no dependence on how long a Valkey round-trip takes.
///
/// The seeded header, when present, is written BEFORE the stall for the same
/// reason: it must be durable before the deadline expires, so the over-budget
/// re-check's verdict is a fact rather than a race.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct StallingProbeInjector {
    /// Set when the arm wants the over-budget re-check to find a peer's
    /// completed restore; `None` when it must find nothing.
    peer_header: Option<PeerHeaderSeed>,
    calls: std::sync::atomic::AtomicUsize,
    stalled: std::sync::atomic::AtomicBool,
}

/// The `obj:hdr` a stand-in peer replica writes back mid-stall.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct PeerHeaderSeed {
    client: live_edit::storage::client::ValkeyClient,
    header_key: String,
    header_fields: Vec<(String, String)>,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl live_edit::checkpoint::adapters::CheckpointFaultInjector for StallingProbeInjector {
    fn maybe_inject(&self, op: &str) -> Result<(), live_edit::checkpoint::CheckpointError> {
        if op != live_edit::checkpoint::RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE {
            return Ok(());
        }
        self.calls.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        if !self.stalled.swap(true, std::sync::atomic::Ordering::AcqRel) {
            let peer = self.peer_header.as_ref().map(|seed| {
                (
                    seed.client.pool.clone(),
                    seed.header_key.clone(),
                    seed.header_fields.clone(),
                )
            });
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    if let Some((pool, key, fields)) = peer {
                        let _: () = fred::interfaces::HashesInterface::hset(&pool, &key, fields)
                            .await
                            .expect("the simulated peer's obj:hdr write must succeed");
                    }
                    tokio::time::sleep(ADJUDICATION_STALL).await;
                })
            });
        }
        // Deliberately NOT an error: the real EVALSHA that follows is the
        // `.await` that hands control back to the deadline, and letting it run
        // is what makes the refusal attributable to the timeout rather than to
        // an injected failure the retry would have classified on its own.
        Ok(())
    }
}

/// **The budget expiry itself.** A store that accepts the probe and never
/// answers must not hold the layer's admission mutex for the retry counts' worst
/// case — the adjudication is cut off at
/// [`RECOVERY_STREAM_ADJUDICATION_BUDGET`] and refuses on the same fail-closed
/// terms as every other unresolved-stream verdict.
///
/// Three properties, none of which any other test in this file can see:
///
/// 1. **The deadline fires at all.** The probe is consulted exactly ONCE. No
///    attempt failed, so the retry budget is untouched and wall clock is the
///    only thing that could have ended the adjudication — which is the claim.
/// 2. **`probe_unreadable` ticks EXACTLY once.** The expiry arm has its own
///    counter tick, for the case where the peer-landed re-check also blows its
///    grace, and `refuse_restore` — which the arm calls first, under that
///    grace — has one too. A double tick is the live hazard, and the argument
///    that it cannot happen (`refuse_restore`'s single `.await` precedes both
///    its tick and its WARN, so a cancelled call provably ticked nothing) is a
///    cancel-safety claim about tokio: exactly the kind of reasoning that
///    deserves a test rather than a comment. A count of 2 here means one refusal
///    was reported to an operator as two.
/// 3. **Nothing was written and nothing was touched.** Same fail-closed posture
///    as tests 10 and 17: no header, no `obj:*` key, and the surviving stream
///    left exactly as found — we never read its top, so we have no basis for
///    calling it stale.
///
/// Wall clock: ≈11 s, one stall. See the block comment above the pair.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seq_floor_refuses_restore_when_the_adjudication_exceeds_its_overall_budget() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-20";

    let (session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-20",
        layer_name,
    )
    .await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    let implausible_seq =
        plant_implausible_stream_top(&harness, &stream_key, object_id, snap_seq).await;

    let injector = std::sync::Arc::new(StallingProbeInjector {
        peer_header: None,
        calls: std::sync::atomic::AtomicUsize::new(0),
        stalled: std::sync::atomic::AtomicBool::new(false),
    });
    harness.install_recovery_hook({
        let injector = std::sync::Arc::clone(&injector);
        move |hook| {
            hook.with_restore_fault_injector_for_test(
                injector
                    as std::sync::Arc<dyn live_edit::checkpoint::adapters::CheckpointFaultInjector>,
            )
        }
    });

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);

    stat_expecting_refusal(
        &harness,
        &session_id,
        layer_name,
        "probe_unreadable",
        "b.4et: an adjudication that blew its overall budget",
    )
    .await;

    assert_eq!(
        injector.calls.load(std::sync::atomic::Ordering::Acquire),
        1,
        "the probe must have been consulted EXACTLY once: no attempt failed, so the retry budget \
         is untouched and the only thing that could have ended the adjudication is the overall \
         deadline. A count of {RECOVERY_STREAM_PROBE_ATTEMPTS} would mean the stall was absorbed \
         and the refusal came from the ordinary retry-exhaustion path instead"
    );

    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[("probe_unreadable", 1)],
        "b.4et: an adjudication cut off at its overall budget reports ONE probe_unreadable — the \
         expiry arm and the refusal it delegates to must not both tick",
    );

    assert_restore_wrote_nothing(
        &harness,
        object_id,
        std::slice::from_ref(&stream_key),
        "b.4et: after an over-budget refusal",
    )
    .await;

    assert_eq!(
        read_stream_last_generated_id(&harness, &stream_key).await,
        implausible_seq,
        "an over-budget refusal MUST leave the surviving stream exactly as it found it: the probe \
         never produced a reading, so there is no evidence the top is stale and no basis for \
         removing it"
    );
    assert_eq!(
        read_key_ttl_ms(&harness, &stream_key).await,
        -1,
        "the stream must be present with no expiry (PTTL=-1); -2 would mean the expiry path \
         deleted a stream it never read"
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before,
        "`..._refused_total` MUST stay flat — it counts a restore that LANDED after verifiably \
         discarding a stale stream, and here nothing was discarded and nothing landed"
    );
}

/// The over-budget arm's **grace path**: even out of budget, a peer replica that
/// landed this recovery meanwhile must turn the refusal back into `Restored`.
///
/// The expiry arm does not simply refuse. It re-checks `obj:hdr` under
/// `RECOVERY_STREAM_ADJUDICATION_PEER_GRACE` first, because refusing a recovery
/// a peer already completed reports a live layer to its client as `UNAVAILABLE`.
/// This is the arm that exercises that second, nested deadline — the ONLY caller
/// of `refuse_restore` that runs with its budget already spent — and the only
/// one where an implementation that skipped the re-check on expiry (a tempting
/// simplification: "we are already over budget, do not wait again") would be
/// caught.
///
/// Staged exactly as test 16, one layer deeper: the peer's header is written
/// from inside the seam, so it lands strictly between the hook's step-4 `EXISTS`
/// re-check and the refusal — the only window in which the re-check can observe
/// something step 4 did not. Seeding it beforehand would short-circuit at step 4
/// and never reach the probe at all.
///
/// Driven at the hook rather than through `handle_stat_layer` for test 16's
/// reason: the handler folds `Restored` and a plain header hit into the same
/// `Ok` and could not tell them apart.
///
/// Wall clock: ≈11 s, one stall. See the block comment above the pair.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seq_floor_over_budget_refusal_yields_to_a_peer_that_already_restored() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-21";

    let (_session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-21",
        layer_name,
    )
    .await;

    // Captured BEFORE the wipe, so the "peer" restores a real header rather than
    // a synthetic key that merely satisfies EXISTS.
    let peer_header: Vec<(String, String)> = read_obj_header(&harness, object_id)
        .await
        .into_iter()
        .collect();
    assert!(
        !peer_header.is_empty(),
        "test pre-condition: the live header must be non-empty before the wipe"
    );

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    let implausible_seq =
        plant_implausible_stream_top(&harness, &stream_key, object_id, snap_seq).await;

    let injector = std::sync::Arc::new(StallingProbeInjector {
        peer_header: Some(PeerHeaderSeed {
            client: harness.tv.client().clone(),
            header_key: harness.tv.key_schema().obj_header(&object_id.to_string()),
            header_fields: peer_header,
        }),
        calls: std::sync::atomic::AtomicUsize::new(0),
        stalled: std::sync::atomic::AtomicBool::new(false),
    });
    harness.install_recovery_hook({
        let injector = std::sync::Arc::clone(&injector);
        move |hook| {
            hook.with_restore_fault_injector_for_test(
                injector
                    as std::sync::Arc<dyn live_edit::checkpoint::adapters::CheckpointFaultInjector>,
            )
        }
    });

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);

    let outcome = harness.svc.recovery.try_restore_layer(object_id).await;
    assert!(
        matches!(outcome, RecoveryOutcome::Restored),
        "b.4et: even out of budget, the refusal MUST yield to the peer that landed the restore \
         while we were stalled. An Error here means a completed recovery was reported UNAVAILABLE \
         to a client whose layer is in fact live — and, specifically, that the expiry arm skipped \
         the graced re-check that exists to prevent exactly that. Got {outcome:?}"
    );

    assert_eq!(
        injector.calls.load(std::sync::atomic::Ordering::Acquire),
        1,
        "test pre-condition: the probe must have been consulted once and stalled — this arm is \
         only meaningful if the verdict was reached through the budget-expiry path"
    );

    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[],
        "b.4et: an over-budget adjudication that found a peer's completed restore MUST tick \
         NOTHING — the restore succeeded, and a counter that says otherwise sends an operator \
         after an outage that did not happen",
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before,
        "nothing was discarded on this arm either"
    );

    assert_eq!(
        read_stream_last_generated_id(&harness, &stream_key).await,
        implausible_seq,
        "the yielding path writes nothing and deletes nothing — the peer's restore owns whatever \
         needed doing to this stream"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 22 — the identity gate's PERMANENT exit propagates, and refuses nothing.
// ─────────────────────────────────────────────────────────────────────────────

/// A wrong-typed `le:{sid}:layer:{hex(name)}` under the SR-2.4 identity gate is
/// **terminal**, not a retriable `unlink_unverified`.
///
/// # The regression this fences (b.h4m, b.f4t sub-task 2)
///
/// `live_mapping_resolves_identity_once` returns the wrong-typed-mapping verdict
/// on the retry driver's `Ok` channel — `Ok(Err(permanent))` — precisely so it
/// reaches `remove_stale_stream`'s caller **unchanged**, and that caller
/// propagates it verbatim. Route it any other way and it is stripped twice
/// (folded to `ServiceError::Internal` by the driver, then folded back to
/// `Transient` by `service_err_to_checkpoint_err`, which preserves nothing but
/// `NotFound`), lands on the gate's `Err` arm, and is reported as an
/// `unlink_unverified` refusal → `Transient` → `UNAVAILABLE`. The handlers'
/// retry-once recovery dispatch then re-drives a restore that can never succeed:
/// no retry re-types a key, so the identical `WRONGTYPE` re-emits its ERROR line,
/// `live_edit_keyspace_type_error_total` re-ticks, and the client is told to come
/// back and do it all again. The honest status is a non-retriable `INTERNAL`
/// naming the key an operator has to remove.
///
/// # What this covers that `restore_permanent_refusal_test` does not
///
/// `wrong_typed_layer_mapping_refuses_permanently` (gate 1 in that file) drives
/// `apply_checkpoint` DIRECTLY, so it exercises `apply_restore`'s step-2 identity
/// guard and nothing else. The gate under test here is a **different, earlier**
/// one — `remove_stale_stream`'s step-6 pre-`UNLINK` authorization — reached only
/// when the plausibility gate has already refused a surviving stream top, which
/// `apply_checkpoint` never runs. Neither test can see the other's arm.
///
/// # Why the hook, not `handle_stat_layer`
///
/// Forced rather than stylistic, for the same reason
/// `restore_permanent_refusal_test.rs`'s wrong-typed-header pair splits in two:
/// `handle_stat_layer` resolves the layer mapping BEFORE it can answer
/// `NotFound`, so the same `WRONGTYPE` refuses the RPC at the read path and the
/// recovery branch never fires. Driving the hook is the only way to reach the
/// gate at all. The wire-level verdict is asserted through
/// `ServiceError::from_recovery_error` — the ONE function that decides
/// retriable-vs-terminal for every recovery caller — rather than assumed.
///
/// No injector: the gate is driven by real keyspace state, like test 15.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_propagates_a_wrongtyped_layer_mapping_from_the_identity_gate_as_permanent() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-22";

    let (session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-22",
        layer_name,
    )
    .await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    let implausible_seq =
        plant_implausible_stream_top(&harness, &stream_key, object_id, snap_seq).await;

    // Retype the key the gate consults: a HASH where a string is expected, so
    // its `GET` is a `WRONGTYPE`. `DEL` first because `CreateLayer` already put a
    // string there — an `HSET` onto a live string would itself be the WRONGTYPE
    // and the fixture would never be built. The service retypes no key it owns,
    // so this stands in for whatever foreign writer an operator has to hunt down.
    let mapping_key = harness
        .tv
        .key_schema()
        .layer_mapping(&session_id, layer_name);
    let removed: i64 = harness
        .tv
        .client()
        .pool
        .del(&mapping_key)
        .await
        .expect("DEL of the layer mapping must succeed in test setup");
    assert_eq!(
        removed, 1,
        "test pre-condition: the layer mapping must have existed and been removed before it is \
         re-planted at the wrong type"
    );
    let _: () = harness
        .tv
        .client()
        .pool
        .hset(&mapping_key, [("not", "a-string")])
        .await
        .expect("HSET to mistype the layer mapping must succeed in test setup");

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);
    let error_before = layer_recovery_metric(&harness, "error");
    let restored_before = layer_recovery_metric(&harness, "restored");
    let not_recoverable_before = layer_recovery_metric(&harness, "not_recoverable");

    let outcome = harness.svc.recovery.try_restore_layer(object_id).await;
    let RecoveryOutcome::Error(err) = outcome else {
        panic!(
            "a wrong-typed layer mapping under the identity gate must end the attempt as Error; \
             got {outcome:?}. `Restored` would claim a repair that never happened; \
             `NotRecoverable` would report a permanently broken keyspace as a layer that simply no \
             longer exists, and no operator would go looking for the key"
        );
    };

    assert!(
        matches!(
            err,
            live_edit::checkpoint::CheckpointError::Permanent { .. }
        ),
        "the gate's wrong-typed-mapping verdict MUST reach the caller as `Permanent`. A \
         `Transient` here is the pre-fix fold — it becomes an `unlink_unverified` refusal and the \
         retry-once dispatch re-drives an unwinnable restore; got {err:?}"
    );
    assert!(
        !live_edit::error::ServiceError::recovery_error_is_retriable(&err),
        "`recovery_error_is_retriable` must answer false — that predicate is what picks \
         UNAVAILABLE vs INTERNAL for every recovery caller, so a Permanent it calls retriable \
         reproduces the whole defect one layer down"
    );
    let status = tonic::Status::from(live_edit::error::ServiceError::from_recovery_error(
        &err,
        format!("checkpoint recovery failed for object {object_id}"),
    ));
    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "at the wire the verdict MUST be non-retriable INTERNAL. UNAVAILABLE means `retry me`, and \
         no retry re-types a key, so a conforming client would hammer a dead layer forever. Got \
         {:?}: {}",
        status.code(),
        status.message()
    );
    assert!(
        err.to_string().contains(&mapping_key),
        "the refusal MUST name the offending key — Valkey's WRONGTYPE reply names none, and this \
         key is not even under the object's hash tag, so an operator has nothing else to go on. \
         Got: {err}"
    );

    // ── Nothing was refused, and nothing was touched ─────────────────────
    //
    // The unresolved family is the sharp half of this test. `unlink_unverified`
    // is EXACTLY the cell the pre-fix fold ticked, so a tick here would mean the
    // permanent verdict was re-reported as a retriable stream verdict even if the
    // returned error somehow stayed `Permanent`.
    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[],
        "b.h4m: a wrong-typed layer mapping is a keyspace fault, not an unresolved stream — the \
         stream's state was never in question, and reporting it on `unlink_unverified` sends an \
         operator hunting a stream key when the actionable finding is the mapping",
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before,
        "`..._refused_total` counts a stream verifiably DISCARDED ahead of a landed restore; \
         nothing was discarded and nothing landed"
    );

    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_before,
        1,
        "the attempt must be recorded as `error`, exactly once — step 6's `Err` arm owns that \
         tick, and it is the cell an operator alerts on"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_before,
        0,
        "`restored` MUST stay flat: nothing was materialized"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "not_recoverable") - not_recoverable_before,
        0,
        "`not_recoverable` MUST stay flat: this layer is recoverable the moment the foreign key is \
         removed, and reporting it as unrecoverable would tell an operator to stop looking"
    );

    // ── The UNLINK never ran ─────────────────────────────────────────────
    assert_eq!(
        read_stream_last_generated_id(&harness, &stream_key).await,
        implausible_seq,
        "the stale stream MUST survive: the gate refused BEFORE authorizing its removal, and \
         destroying a stream on an unauthorized path is the b.4et item-6 defect — during a \
         DeleteLayer's 30s grace window that stream is what lets a parked subscriber see the \
         `deleted` sentinel and exit cleanly"
    );
    assert_eq!(
        read_key_ttl_ms(&harness, &stream_key).await,
        -1,
        "the stream must be untouched, not put on a fuse"
    );
    let header = read_obj_header(&harness, object_id).await;
    assert!(
        header.is_empty(),
        "nothing may be materialized on this arm — the restore was refused before the apply ran. \
         Got {header:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests 23–25 — the b.h4m pre-`UNLINK` peer-presence re-read.
// ─────────────────────────────────────────────────────────────────────────────
//
// Every decision above the `UNLINK` is made against evidence gathered BEFORE
// step 5's checkpoint-blob fetch — a multi-second I/O during which a peer replica
// can land the whole rebuild and re-admit writers to the very stream we are about
// to destroy. `remove_stale_stream` therefore re-asks the structural presence
// question immediately before the removal, on the same bounded retry budget as
// its neighbours, through `RESTORE_SEAM_BEFORE_STALE_STREAM_UNLINK`.
//
// Three arms, one per exit of that re-read, because they are three different
// operator situations:
//
//   * the peer landed a complete header → SKIP the removal, and the restore is
//     already done (test 23);
//   * the re-read blipped and recovered inside the budget → the removal proceeds
//     exactly as if it had never blipped (test 24);
//   * the re-read failed for the whole budget → refuse, because removing a live
//     stream on ignorance is the failure this read exists to prevent (test 25).
//
// Test 24 is what keeps 25 honest: without a fail-N/pass-N+1 pair, "fail closed"
// is indistinguishable from "fail on the first hiccup", and the retry this read
// was given would be untested — a single fred reconnect (what a managed-proxy
// invalidation produces) refusing a whole restore is a regression this stretch of
// the adjudication did NOT have before b.h4m.

/// `created_by` on the header the stand-in peer replica lands inside the seam.
///
/// Deliberately different from [`CREATED_BY`], which is what this fixture's own
/// restore would stamp. `StatLayer` echoes `created_by` straight out of the live
/// header, so a response carrying [`CREATED_BY`] would mean this replica's
/// `write_object` overwrote the peer's restore — the exact thing the skip exists
/// to prevent — while one carrying this value means the peer's header served.
#[cfg(feature = "test-checkpoint-fault-injection")]
const UNLINK_SEAM_PEER_CREATED_BY: &str = "e2e-object-id-preservation-peer-replica";

/// Lands a structurally-complete `obj:hdr` on its first consultation at
/// `RESTORE_SEAM_BEFORE_STALE_STREAM_UNLINK`, then passes — a peer replica
/// completing this same recovery in the window the re-read exists to close.
///
/// # Why the write goes through the seam rather than being staged up front
///
/// Seeding the header before the call would short-circuit `try_restore_layer`'s
/// own step-4 presence re-check and never reach step 6 at all, making the arm
/// vacuous. The seam fires INSIDE `peer_presence_reread_once`, after the identity
/// gate and immediately before the structural `HGETALL obj:hdr` — which is
/// exactly the window a peer's restore lands in, and the only place a test can
/// put it deterministically.
///
/// `maybe_inject` is synchronous, so the write goes through `block_in_place` +
/// `Handle::block_on`, the sanctioned pattern for driving one await from a sync
/// callback on a `flavor = "multi_thread"` runtime (and the same one
/// [`PeerRestoresDuringProbeInjector`] and [`VanishingStreamInjector`] use). That
/// makes the ordering a FACT — the header is durable before the seam returns —
/// rather than a sleep-and-hope race.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct PeerLandsHeaderInUnlinkSeamInjector {
    client: live_edit::storage::client::ValkeyClient,
    header_key: String,
    header_fields: Vec<(String, String)>,
    seeded: std::sync::atomic::AtomicBool,
    calls: std::sync::atomic::AtomicUsize,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl live_edit::checkpoint::adapters::CheckpointFaultInjector
    for PeerLandsHeaderInUnlinkSeamInjector
{
    fn maybe_inject(&self, op: &str) -> Result<(), live_edit::checkpoint::CheckpointError> {
        if op != live_edit::checkpoint::RESTORE_SEAM_BEFORE_STALE_STREAM_UNLINK {
            return Ok(());
        }
        self.calls.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        if !self.seeded.swap(true, std::sync::atomic::Ordering::AcqRel) {
            let pool = self.client.pool.clone();
            let key = self.header_key.clone();
            let fields = self.header_fields.clone();
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    let _: () = fred::interfaces::HashesInterface::hset(&pool, &key, fields)
                        .await
                        .expect("the simulated peer's obj:hdr write must succeed");
                })
            });
        }
        Ok(())
    }
}

// Test 23's restore-fenced readers — `wire_client_metrics`,
// `restore_fenced_cells` and `assert_restore_fenced_deltas` — live in
// `helpers.rs`, shared with `restore_write_fence_test.rs` and
// `restore_fault_injection_matrix_test.rs`. `wire_client_metrics` is what keeps
// test 23's `presence_probe` assertion from reading a permanently-zero cell:
// `live_edit_checkpoint_restore_fenced_total` is ticked from `apply_restore`
// through `client.metrics_snapshot()`, not through the hook's `SharedMetrics`.

/// **Test 23 — the skip.** A peer replica landing a structurally-complete
/// `obj:hdr` in the pre-`UNLINK` window means the stream under this `object_id`
/// may be that peer's LIVE one, so the removal must not happen at all.
///
/// # What is at stake if it does
///
/// The stale top the plausibility gate refused was read before the blob fetch. By
/// the time the `UNLINK` goes on the wire, a peer's completed restore may have
/// re-admitted writers to that key — and destroying it then is not a no-op: every
/// live subscriber loses the commits sitting between its cursor and the removed
/// top, permanently, on a layer that is healthy. The re-read narrows that
/// exposure from the whole multi-second fetch to one round trip.
///
/// # What each assertion adds
///
/// - **The stream survives, with no fuse.** The load-bearing durable fact:
///   `PTTL = -1`, top unchanged. `-2` is the defect.
/// - **`..._refused_total` FLAT.** That counter means "a stream was verifiably
///   cleared ahead of a restore", which is precisely what did not happen. Ticking
///   it would over-report cleanups the fleet never performed.
/// - **`fenced{presence_probe}` +1, the other two stages flat.** The skip returns
///   `Ok(None)`, so the apply runs and its step-1 structural check is what
///   short-circuits — `residue_wipe` or `object_write` moving instead would mean
///   the attempt got PAST step 1 and the peer was noticed later than claimed.
/// - **`StatLayer` Ok carrying the PEER's `created_by`.** The peer's header is
///   what serves, unclobbered; [`CREATED_BY`] here would mean this replica's
///   materializer ran anyway.
/// - **`restored` +1, `error` flat.** `AlreadyPresent` counts as `restored` (the
///   pre-existing layer-scope vocabulary), and a fence is not a failure.
/// - **Nothing on the unresolved family.** The stream's state was never in doubt;
///   we simply had no business touching it.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seq_floor_skips_the_stale_stream_unlink_when_a_peer_lands_a_header_in_the_window() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-23";

    let (session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-23",
        layer_name,
    )
    .await;

    // Captured BEFORE the wipe so the "peer" restores a real materialized header
    // rather than a synthetic key that merely satisfies the predicate, with
    // `created_by` swapped to a marker only a peer could have written.
    let mut peer_header: Vec<(String, String)> = read_obj_header(&harness, object_id)
        .await
        .into_iter()
        .collect();
    for entry in peer_header.iter_mut() {
        if entry.0 == "created_by" {
            entry.1 = UNLINK_SEAM_PEER_CREATED_BY.to_owned();
        }
    }
    assert!(
        peer_header
            .iter()
            .any(|(name, value)| name == "created_by" && value == UNLINK_SEAM_PEER_CREATED_BY),
        "test pre-condition: the captured header must carry `created_by`, or the peer-vs-us \
         assertion below is vacuous; got {peer_header:?}"
    );

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    let implausible_seq =
        plant_implausible_stream_top(&harness, &stream_key, object_id, snap_seq).await;

    wire_client_metrics(&harness).await;

    let injector = std::sync::Arc::new(PeerLandsHeaderInUnlinkSeamInjector {
        client: harness.tv.client().clone(),
        header_key: harness.tv.key_schema().obj_header(&object_id.to_string()),
        header_fields: peer_header,
        seeded: std::sync::atomic::AtomicBool::new(false),
        calls: std::sync::atomic::AtomicUsize::new(0),
    });
    harness.install_recovery_hook({
        let injector = std::sync::Arc::clone(&injector);
        move |hook| {
            hook.with_restore_fault_injector_for_test(
                injector
                    as std::sync::Arc<dyn live_edit::checkpoint::adapters::CheckpointFaultInjector>,
            )
        }
    });

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);
    let fenced_before = restore_fenced_cells(&harness);
    let restored_before = layer_recovery_metric(&harness, "restored");
    let error_before = layer_recovery_metric(&harness, "error");

    let response = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .unwrap_or_else(|status| {
            panic!(
                "b.h4m: a peer's completed restore is a SKIP, not an error — the removal is \
                 declined, the apply short-circuits on the peer's header, and StatLayer serves it. \
                 Got {:?}: {}",
                status.code(),
                status.message()
            )
        })
        .into_inner();

    assert_eq!(
        injector.calls.load(std::sync::atomic::Ordering::Acquire),
        1,
        "test pre-condition: the pre-UNLINK seam must have been consulted EXACTLY once. Zero means \
         the re-read is not on this path at all and every assertion below is vacuous; more than \
         one would mean the seam's `Ok` return was somehow retried"
    );

    // ── The stream was NOT removed ───────────────────────────────────────
    //
    // `PTTL` first, deliberately: it is the assertion that distinguishes the
    // regression this arm names (`-2`, the removal ran anyway) from a healthy
    // key, and it reads a sentinel rather than erroring. Asking for the stream's
    // top first would fail inside `read_stream_last_generated_id`'s `XINFO`
    // (`ERR no such key`) and report the defect as a broken fixture.
    assert_eq!(
        read_key_ttl_ms(&harness, &stream_key).await,
        -1,
        "b.h4m: the stale-stream UNLINK MUST be skipped, so the key must be present with no expiry \
         (PTTL=-1). A -2 means the removal ran anyway: a peer landed the rebuild while we were \
         fetching the blob, so the stream under this object_id may be its LIVE one — and removing \
         it costs every in-window subscriber the commits between its cursor and the top, on a \
         layer that is healthy"
    );
    assert_eq!(
        read_stream_last_generated_id(&harness, &stream_key).await,
        implausible_seq,
        "the skipped stream must be untouched down to its top — a changed high-water would mean \
         something on this path wrote to a stream it had just declined to adjudicate"
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before,
        "`..._refused_total` MUST stay flat — it counts a stream verifiably DISCARDED ahead of a \
         restore, and this arm's whole claim is that nothing was discarded"
    );

    // ── The apply short-circuited at step 1, and served the peer ─────────
    assert_restore_fenced_deltas(
        &harness,
        &fenced_before,
        &[("presence_probe", 1)],
        "b.h4m: the skip returns Ok(None), so apply_restore runs and its step-1 structural probe \
         is what short-circuits. A tick on `residue_wipe` or `object_write` would mean the attempt \
         got past step 1 — i.e. the peer was noticed later than this arm claims",
    );
    assert_eq!(
        response.created_by_user_name, UNLINK_SEAM_PEER_CREATED_BY,
        "StatLayer MUST serve the PEER's header. `{CREATED_BY}` here would mean this replica's \
         write_object overwrote a restore that had already landed"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_before,
        1,
        "AlreadyPresent counts as `restored` — the layer scope's result vocabulary is \
         `restored | not_recoverable | error`, and the presence short-circuit has always used the \
         first"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_before,
        0,
        "`error` MUST stay flat: a peer landing the restore is the re-read working, not a failure"
    );
    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[],
        "b.h4m: a skip is not an unresolved stream — the re-read answered, and what it answered \
         was `do not touch this`",
    );
}

/// **Test 24 — the retry.** A blip on the pre-`UNLINK` re-read that clears inside
/// the budget must cost nothing: the removal proceeds and the restore lands
/// exactly as if the re-read had never failed.
///
/// # Why this arm exists at all
///
/// The re-read is a NEW round-trip on a stretch of the adjudication that had none
/// before b.h4m, and a new round-trip is a new way to fail. Un-retried, a single
/// fred reconnect — what a managed-proxy invalidation produces, i.e. the common
/// case, not the rare one — would refuse a whole restore as `unlink_unverified`.
/// The fail-N/pass-N+1 shape is what proves the budget is really spent: the
/// injector fails attempt 1 and passes attempt 2, so a count of 1 would mean the
/// retry is absent and a count above 2 would mean the success arm was never
/// reached.
///
/// Downstream of the recovered re-read everything is the ORDINARY b.8y5 refusal:
/// the floor falls back to `snap_seq`, the stale stream is verifiably `UNLINK`ed,
/// `..._refused_total` ticks once, and the first commit on the restored layer
/// succeeds — with no trace of the blip on the unresolved family.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_retries_a_transient_peer_presence_reread_before_unlinking() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-24";

    let (session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-24",
        layer_name,
    )
    .await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    plant_implausible_stream_top(&harness, &stream_key, object_id, snap_seq).await;

    let injector = std::sync::Arc::new(SeamInjector::failing_first(
        live_edit::checkpoint::RESTORE_SEAM_BEFORE_STALE_STREAM_UNLINK,
        1,
    ));
    arm_seam(&mut harness, &injector);

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);

    let _ = handle_stat_layer(&harness.svc, stat_request(&session_id, layer_name))
        .await
        .expect(
            "b.h4m: a peer-presence re-read blip that clears inside the retry budget MUST NOT cost \
             a restore. An UNAVAILABLE here means the budget is being spent as if it were one \
             attempt — and a single fred reconnect then refuses a restore this stretch of the \
             adjudication used to complete",
        );

    assert_eq!(
        injector.calls(),
        2,
        "the injector must have been consulted once per attempt: one injected failure and the \
         successful second. A count of 1 means the retry is absent; a count of \
         {RECOVERY_STREAM_PROBE_ATTEMPTS} would mean the pass arm was never taken and the restore \
         reached its outcome some other way"
    );

    let header = read_obj_header(&harness, object_id).await;
    let recovered_seq = parse_header_u64(&header, "seq");
    assert_eq!(
        recovered_seq,
        snap_seq + SMALL_SEQ_MARGIN,
        "the recovered re-read found no peer header, so the ordinary b.8y5 plausibility refusal \
         runs to completion and the floor falls back to snap_seq ({snap_seq})"
    );

    assert_stale_stream_unlinked(&harness, &stream_key).await;
    assert_restored_layer_accepts_a_commit(
        &harness,
        &session_id,
        layer_name,
        object_id,
        &stream_key,
        recovered_seq,
        24,
    )
    .await;

    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[],
        "b.h4m: a transient peer-presence re-read absorbed by the retry budget",
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before + 1,
        "the eventual outcome is an ordinary b.8y5 plausibility refusal, so `..._refused_total` \
         ticks exactly as it does when the re-read never blipped at all"
    );
}

/// **Test 25 — exhaustion.** A pre-`UNLINK` re-read that fails for its whole
/// budget refuses the restore as `unlink_unverified`, and leaves the stream
/// alone.
///
/// # Why refusing is the only safe answer
///
/// We hold positive evidence of a stale top and cannot establish whether the
/// stream under it is a peer's live one. Removing it on that ignorance is exactly
/// the failure the re-read exists to prevent; proceeding without removing it
/// seeds `obj:hdr` beneath a top that will never lapse, which since b.ph9 refuses
/// every subsequent `UpdateLayer` forever. So the restore is refused whole — the
/// caller gets a retriable `UNAVAILABLE` and nothing is written.
///
/// The reason is `unlink_unverified`, shared with the `UNLINK`'s own failure arms
/// on purpose: the operator situation is the same one ("the cleanup could not be
/// carried out"), and it is emphatically NOT `probe_unreadable` — the probe
/// worked, read the top, and judged it stale.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seq_floor_refuses_when_the_peer_presence_reread_cannot_be_completed() {
    let mut harness = setup_checkpoint_harness_tails_off_with_seq_margin(1, SMALL_SEQ_MARGIN).await;
    let layer_name = "e2e-object-id-preservation-layer-25";

    let (session_id, object_id, snap_seq, _stream_high_water) = drive_split_preamble(
        &mut harness,
        "e2e-object-id-preservation-session-25",
        layer_name,
    )
    .await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
    let implausible_seq =
        plant_implausible_stream_top(&harness, &stream_key, object_id, snap_seq).await;

    let injector = std::sync::Arc::new(SeamInjector::always(
        live_edit::checkpoint::RESTORE_SEAM_BEFORE_STALE_STREAM_UNLINK,
    ));
    arm_seam(&mut harness, &injector);

    let refused_before = stream_refused_metric(&harness);
    let unresolved_before = stream_unresolved_cells(&harness);

    stat_expecting_refusal(
        &harness,
        &session_id,
        layer_name,
        "unlink_unverified",
        "b.h4m: a peer-presence re-read that never completed",
    )
    .await;

    assert_eq!(
        injector.calls(),
        RECOVERY_STREAM_PROBE_ATTEMPTS,
        "the re-read gets the same bounded retry budget as its neighbours \
         ({RECOVERY_STREAM_PROBE_ATTEMPTS} attempts) — a read that fails once must not cost a \
         restore, and a budget that never ends is paid for under the per-object admission mutex"
    );

    assert_restore_wrote_nothing(
        &harness,
        object_id,
        std::slice::from_ref(&stream_key),
        "b.h4m: after a peer-presence-re-read refusal",
    )
    .await;

    assert_eq!(
        read_stream_last_generated_id(&harness, &stream_key).await,
        implausible_seq,
        "the stale stream MUST still be there — we could not establish whether it is a peer's live \
         one, and that ignorance is precisely why the restore was refused rather than the stream \
         removed"
    );
    assert_eq!(
        read_key_ttl_ms(&harness, &stream_key).await,
        -1,
        "the stream must be present with no expiry (PTTL=-1); -2 would mean the refusal path \
         removed a stream it could not adjudicate"
    );

    assert_stream_unresolved_deltas(
        &harness,
        &unresolved_before,
        &[("unlink_unverified", 1)],
        "b.h4m: the cleanup could not be carried out, which is `unlink_unverified` — NOT \
         `probe_unreadable`, because the probe worked and read the top; the two send an operator \
         at different remediations",
    );
    assert_eq!(
        stream_refused_metric(&harness),
        refused_before,
        "`..._refused_total` marks a VERIFIED discard; this removal was never even attempted"
    );
}
