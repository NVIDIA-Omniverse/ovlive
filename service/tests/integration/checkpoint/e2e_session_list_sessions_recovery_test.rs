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

//! End-to-end session-recovery integration tests for `ListSessions`
//! (Subtask `t3.fgj.7v.o7.wz` under Task `t2.fgj.7v.o7`, Epic `t1.fgj.7v`,
//! Bee `b.fgj`).
//!
//! Epic AC #7 (SR-2.7) pins three contracts on the `ListSessions` RPC
//! after Task 6 wired session-recovery into the
//! [`reconcile_ghost_members`] helper:
//!
//! 1. **Path-index walk surfaces ghosts; ghosts with a session-checkpoint
//!    blob are recovered, restored, and the composite member is
//!    preserved.** The pre-recovery code SREM'd every empty-HGETALL
//!    member unconditionally; SR-2.7 changes that to "recover or SREM"
//!    based on the adapter's `get_session_checkpoint` outcome.
//!
//! 2. **Ghosts with NO backing checkpoint stay SREM'd** — the legacy
//!    opportunistic-cleanup behaviour customers depending on
//!    `checkpoint_enabled=false` still see, AND the cleanup path for a
//!    genuinely-orphaned path-index member.
//!
//! 3. **Composite-name index membership preserved exactly** after a
//!    recovery pass. SR-2.7 explicitly calls out `name\0session_id` as
//!    the load-bearing identity — a regression that SREM'd a recovered
//!    member's composite (or replaced it with a derived-by-name string)
//!    would break the duplicate-name uniqueness invariant
//!    `SessionStore::create` relies on via the `path_names_index` set.
//!
//! ## Wipe strategy
//!
//! The Subtask body invokes [`full_valkey_wipe`] as the wipe primitive.
//! `full_valkey_wipe` removes EVERY key under the test prefix — including
//! the `path_names_index` set whose composite members are the entrypoint
//! the path-index walk inside [`SessionStore::list`] reads. To exercise
//! the actual recovery path the parent Task body pins, the path-index
//! must contain the original composite members at the time of the
//! `handle_list_sessions` call.
//!
//! The two-step shape "[`full_valkey_wipe`] + manual re-`SADD` of the
//! composite members" simulates the realistic disaster scenario: the
//! checkpoint adapter (LocalDisk in this harness, StorageApi or a peer
//! storage tier in production) survived the wipe and carries the per-
//! session blobs, AND the `path_names_index` was re-materialised from
//! some external source — for instance, an operator running a recovery
//! script that primes the path-index from a backup. The recovery hook's
//! `get_session_checkpoint` round-trip is what reads the surviving
//! adapter state. This is exactly SR-2.7's asymmetric self-healing
//! contract: path-index is the "outer" recovery entrypoint, and the
//! checkpoint blob is the "inner" payload that re-materialises
//! `le:{sid}:sess`.
//!
//! Using `wipe_session_state` per-session would not exercise the
//! contract: that helper preserves the path-index intact already, so it
//! is the wrong tool for the "everything except path-index was lost"
//! disaster. The hybrid pattern below is the faithful model.
//!
//! ## Test scope
//!
//! Three scenarios mapped 1:1 to the three SR-2.7 contracts above. The
//! `ListSessions` RPC is the only handler exercised; sibling test files
//! cover the seven other session-resolving handlers via the
//! `SessionBasePathCache::get_base_path` choke point. `ListSessions`
//! bypasses the cache entirely (see `handle_list_sessions`'s docstring),
//! so the wiring contract under test here is `reconcile_ghost_members`
//! — a different choke point that funnels recovery through the same
//! `SessionRecoveryHook`.
//!
//! Sandbox env note: this test uses `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use fred::interfaces::{HashesInterface, SetsInterface};
use tonic::Request;

use live_edit::grpc::proto::{CreateSessionRequest, ListSessionsRequest};
use live_edit::grpc::session::{handle_create_session, handle_list_sessions};
use live_edit::storage::keys::KeySchema;

use super::helpers::*;

const SESSION_BASE_PATH: &str = "/stage/list-sessions-recovery";
const SESSION_NAME_A: &str = "list-sessions-recovery-alpha";
const SESSION_NAME_B: &str = "list-sessions-recovery-beta";
const SESSION_NAME_C: &str = "list-sessions-recovery-gamma";

const GHOST_BASE_PATH: &str = "/stage/list-sessions-ghost";
const GHOST_SESSION_ID: &str = "00000000-0000-4000-8000-000000000fff";
const GHOST_SESSION_NAME: &str = "list-sessions-ghost-orphan";

// `StubUserinfo::display_name` (see `tests/integration/grpc/helpers.rs`)
// is what `handle_create_session` plumbs into `created_by` via
// `user_info.display_name`. The post-recovery `ListSessions` response
// surfaces this verbatim as `Session.created_by_user_name`.
const STUB_USER_DISPLAY_NAME: &str = "Test User";

// ── Request builders ─────────────────────────────────────────────────────────
//
// `make_unary_token_request` is imported from
// `tests/integration/checkpoint/helpers.rs` via `use super::helpers::*;`
// (b.dzd Phase 1 deduplication).

fn create_session_request(path: &str, name: &str) -> Request<CreateSessionRequest> {
    make_unary_token_request(CreateSessionRequest {
        path: path.to_owned(),
        name: name.to_owned(),
    })
}

fn list_sessions_request(path: &str) -> Request<ListSessionsRequest> {
    make_unary_token_request(ListSessionsRequest {
        path: path.to_owned(),
    })
}

// ── Composite-member SADD helper ─────────────────────────────────────────────

/// `SADD` a composite `name\0session_id` member into
/// `le:path:{hex(base_path)}:names` directly. Used by Scenarios 1 and 3
/// to restore the path-index after `full_valkey_wipe`, simulating the
/// disaster scenario where only the path-index was re-materialised from
/// an external source.
///
/// Also `SADD`s the same `session_id` into the legacy `path_index` set
/// to match the production `SessionStore::create` shape (both sets are
/// populated symmetrically under MULTI/EXEC; tests that only populate
/// one of the two would observe asymmetric post-state and mask any
/// regression that depends on the dual-population invariant).
async fn sadd_composite_member(
    harness: &CheckpointHarness,
    base_path: &str,
    name: &str,
    session_id: &str,
) {
    let path_names_key = harness.tv.key_schema().path_names_index(base_path);
    let path_index_key = harness.tv.key_schema().path_index(base_path);
    let composite = KeySchema::path_name_member(name, session_id);

    let _: () = harness
        .tv
        .client()
        .pool
        .sadd(&path_names_key, composite.clone())
        .await
        .expect("SADD composite member into path_names_index must succeed");
    let _: () = harness
        .tv
        .client()
        .pool
        .sadd(&path_index_key, session_id)
        .await
        .expect("SADD session_id into legacy path_index must succeed");
}

// ── CreateSession + wait_for_session_checkpoint helper ───────────────────────

/// Captured baseline shape: the `session_id` returned by
/// `handle_create_session` and the four pre-call session-scope
/// recovery-metric cells. The metric cells are captured AFTER the wipe
/// so the assertion phase compares against the recovery-attempt counts
/// that the test under-test actually drove.
struct CreatedSession {
    session_id: String,
    name: String,
}

/// CreateSession through the real `handle_create_session` RPC entry
/// point, then wait for the fire-and-forget session-checkpoint
/// dispatcher to land the blob on disk.
///
/// CreateSession's success arm calls `SessionStore::create`, which
/// enqueues a session-checkpoint write through the dispatcher. The blob
/// landing on disk is what `try_recover_session` reads back during
/// recovery — without it, every recovery attempt would surface
/// `NoCheckpoint` instead of `Recovered`, and Scenario 1's recovery
/// assertion would fail with `restored == 0` / `not_recoverable == 3`.
async fn create_session_and_wait_for_checkpoint(
    harness: &CheckpointHarness,
    base_path: &str,
    name: &str,
) -> CreatedSession {
    let response = handle_create_session(&harness.svc, create_session_request(base_path, name))
        .await
        .expect("CreateSession must succeed against AllowAllPermissions harness");
    let session_id = response.into_inner().session_id;
    assert!(
        !session_id.is_empty(),
        "CreateSession must mint a non-empty session_id"
    );

    wait_for_session_checkpoint(&harness.store, &session_id, 10).await;

    CreatedSession {
        session_id,
        name: name.to_owned(),
    }
}

// ── Post-wipe path-index restoration helper ──────────────────────────────────

/// Re-`SADD` the composite members for every session in `sessions` into
/// the path-index `:names` set under `base_path`. The legacy `path_index`
/// set is also restored. Documented at length on
/// [`sadd_composite_member`]; this helper is the multi-session
/// convenience wrapper consumed by Scenario 1 / 3.
async fn restore_path_index_after_wipe(
    harness: &CheckpointHarness,
    base_path: &str,
    sessions: &[CreatedSession],
) {
    for session in sessions {
        sadd_composite_member(harness, base_path, &session.name, &session.session_id).await;
    }
}

// =============================================================================
// Scenario 1 — list_sessions recovers session via path-index walk.
// =============================================================================

/// Pin SR-2.7 contract #1: the path-index walk inside
/// `reconcile_ghost_members` invokes `try_recover_session` once per
/// composite member, every recovered session is preserved in the
/// response, and the `restored` metric increments by EXACTLY the number
/// of ghosts (3 here — one per session).
///
/// Why each step is load-bearing:
///
/// - Three sessions on the same `base_path` exercises the per-ghost
///   loop in `reconcile_ghost_members` (not just a single ghost). A
///   regression that only recovered the FIRST ghost would show
///   `restored == 1` and `sessions.len() == 1`, masking the loop bug.
/// - The post-wipe path-index restoration is what triggers the path-
///   index walk to surface 3 ghosts. Without it, `SessionStore::list`
///   would return empty `present` + empty `ghosts` and recovery would
///   never fire — Scenario 1 would silently pass with `restored == 0`
///   and `sessions.is_empty()`.
/// - The response-shape assertions pin every field the parent Task body
///   names explicitly: `name`, `base_path`, `created_by`, `session_id`
///   must match the originals. A regression that recovered the session
///   but materialised the wrong `created_by` (e.g., propagated a stale
///   identity from the checkpoint blob) would surface here.
/// - The metric assertion uses [`session_recovery_metric`] with
///   `scope="session"` (NOT `"layer"`) — layer-recovery and session-
///   recovery tick disjoint cells, so the assertion would silently pass
///   under any layer-scope regression. The `restored == 3` exact-match
///   pins the per-member at-most-once contract.
/// - `not_recoverable` / `error` / `already_present` stay at their pre-
///   call zero values — three checkpoint blobs were present so every
///   ghost recovered, not a single one fell through to the
///   `NoCheckpoint` arm.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn list_sessions_recovers_session_via_path_index_walk() {
    let harness = setup_checkpoint_harness(1).await;

    // CreateSession for three sessions on the same base_path, waiting
    // for each session-checkpoint blob to land before the next call so
    // every blob is durable on disk before the wipe step.
    let session_a =
        create_session_and_wait_for_checkpoint(&harness, SESSION_BASE_PATH, SESSION_NAME_A).await;
    let session_b =
        create_session_and_wait_for_checkpoint(&harness, SESSION_BASE_PATH, SESSION_NAME_B).await;
    let session_c =
        create_session_and_wait_for_checkpoint(&harness, SESSION_BASE_PATH, SESSION_NAME_C).await;
    let sessions = vec![session_a, session_b, session_c];

    // full_valkey_wipe removes EVERY key under the test prefix — every
    // session_meta hash, every layer-mapping, every path-index entry.
    // The checkpoint adapter (LocalDisk tempdir on this harness) is
    // OUTSIDE the test prefix so the three session-checkpoint blobs
    // survive the wipe — exactly the disaster scenario SR-2.7 is
    // designed to recover from.
    full_valkey_wipe(&harness.tv).await;

    // Restore the path-index `:names` set with the original composite
    // members. Mirrors a recovery script priming the path-index from a
    // backup; without this, the path-index walk would surface zero
    // ghosts and recovery would never fire (the test would pass for
    // the wrong reason — empty list with no recovery activity).
    restore_path_index_after_wipe(&harness, SESSION_BASE_PATH, &sessions).await;

    // Capture metric baselines AFTER setup so the assertions track only
    // the recovery activity driven by the ListSessions call below.
    let restored_pre = session_recovery_metric(&harness, "restored");
    let not_recoverable_pre = session_recovery_metric(&harness, "not_recoverable");
    let error_pre = session_recovery_metric(&harness, "error");
    let already_present_pre = session_recovery_metric(&harness, "already_present");

    // Dispatch ListSessions through the real handler. ListSessions
    // bypasses the SessionBasePathCache entirely; the path-index walk
    // inside SessionStore::list returns three ghosts (HGETALL returned
    // empty for each post-wipe session_id), and reconcile_ghost_members
    // invokes try_recover_session once per ghost.
    let response = handle_list_sessions(&harness.svc, list_sessions_request(SESSION_BASE_PATH))
        .await
        .expect(
            "ListSessions MUST succeed via the path-index walk + per-ghost recovery; \
             a failure here means reconcile_ghost_members did not invoke try_recover_session, \
             or the session_recovery hook on LiveEditService is not the enabled variant",
        );
    let proto_sessions = response.into_inner().sessions;

    // The response carries all three recovered sessions.
    assert_eq!(
        proto_sessions.len(),
        3,
        "ListSessions MUST return all three recovered sessions; got {} sessions. \
         A partial count indicates the per-ghost loop in reconcile_ghost_members \
         bailed early, or try_recover_session reported NoCheckpoint for some \
         (sessions with surviving blobs should all recover).",
        proto_sessions.len()
    );

    // Response shape: every field the parent Task body names must match
    // the original CreateSession inputs. Build a name-keyed map so the
    // assertions are order-independent (ListSessions returns sessions in
    // an unspecified order — the underlying `SMEMBERS` is unordered).
    let by_name: std::collections::HashMap<String, &live_edit::grpc::proto::Session> =
        proto_sessions.iter().map(|s| (s.name.clone(), s)).collect();

    for original in &sessions {
        let recovered = by_name.get(&original.name).unwrap_or_else(|| {
            panic!(
                "ListSessions response MUST contain session name '{}'; got {:?}",
                original.name,
                by_name.keys().collect::<Vec<_>>()
            )
        });
        assert_eq!(
            recovered.session_id, original.session_id,
            "post-recovery session_id MUST match the original CreateSession \
             return value for name '{}'; got {}, expected {}",
            original.name, recovered.session_id, original.session_id
        );
        assert_eq!(
            recovered.created_by_user_name, STUB_USER_DISPLAY_NAME,
            "post-recovery created_by_user_name MUST equal the StubUserinfo \
             display_name '{}' that handle_create_session plumbed in; got {:?} \
             for name '{}'. A regression in apply_session_checkpoint would \
             surface here by restoring a stale or empty created_by field.",
            STUB_USER_DISPLAY_NAME, recovered.created_by_user_name, original.name
        );
        assert!(
            recovered.created_at > 0,
            "post-recovery created_at MUST be a non-zero microseconds-since-epoch \
             value (the original CreateSession call wrote `now_micros()`); got 0 \
             for name '{}'. A zero value here indicates apply_session_checkpoint \
             failed to restore the created_at field.",
            original.name
        );
    }

    // Cross-check the per-session base_path via the post-recovery
    // session_meta HGETALL. The proto `Session` message does NOT
    // include `base_path` (it is implicit in the request), so the
    // parent Task body's "expected shape (name, base_path, ...)" check
    // for base_path lands on the storage layer rather than the proto
    // surface.
    for original in &sessions {
        let session_meta_key = harness.tv.key_schema().session_meta(&original.session_id);
        let recovered_base_path: Option<String> = harness
            .tv
            .client()
            .pool
            .hget(&session_meta_key, "base_path")
            .await
            .expect("HGET session_meta.base_path must succeed against healthy Valkey");
        assert_eq!(
            recovered_base_path.as_deref(),
            Some(SESSION_BASE_PATH),
            "post-recovery session_meta.base_path MUST equal the original \
             CreateSession base_path for session_id={}; got {:?}",
            original.session_id,
            recovered_base_path
        );
    }

    // Metric assertions: exactly THREE recovery attempts in `restored`,
    // every other session-scope outcome cell unchanged.
    assert_eq!(
        session_recovery_metric(&harness, "restored") - restored_pre,
        3,
        "exactly THREE session-recovery attempts MUST have reported `restored` \
         — one per ghost surfaced by the path-index walk. Pre={restored_pre}, \
         observed={}. A count of 1 indicates the loop bailed early; a count \
         of 0 indicates recovery never fired (wiring regression).",
        session_recovery_metric(&harness, "restored")
    );
    assert_eq!(
        session_recovery_metric(&harness, "not_recoverable"),
        not_recoverable_pre,
        "post-recovery `not_recoverable` MUST equal pre-call snapshot — every \
         session_id had a surviving checkpoint blob so no ghost fell through \
         to the NoCheckpoint arm"
    );
    assert_eq!(
        session_recovery_metric(&harness, "error"),
        error_pre,
        "post-recovery `error` MUST equal pre-call snapshot — no transient \
         backend failure was injected"
    );
    assert_eq!(
        session_recovery_metric(&harness, "already_present"),
        already_present_pre,
        "post-recovery `already_present` MUST equal pre-call snapshot — the \
         wipe drove the pre-acquire EXISTS check to false for every session, \
         so the short-circuit cell stays untouched"
    );

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Scenario 2 — ghost member with no checkpoint is swept.
// =============================================================================

/// Pin SR-2.7 contract #2: a path-index member whose `session_id` has
/// neither a surviving `le:{sid}:sess` hash NOR a session-checkpoint
/// blob must be SREM'd from `path_names_index` by
/// `reconcile_ghost_members`, AND the corresponding session MUST be
/// absent from the `ListSessions` response. The `not_recoverable`
/// metric ticks by exactly 1.
///
/// Why each step is load-bearing:
///
/// - The synthetic ghost is `SADD`'d directly into `path_names_index`
///   WITHOUT going through CreateSession — there is no session_meta
///   hash, no checkpoint blob, no legitimate identity. This is the
///   "truly orphaned path-index member" surface that opportunistic-
///   cleanup is designed to retire. A test that used a wiped real
///   session would mix the NoCheckpoint outcome with whatever residual
///   adapter blob the harness might still hold (the LocalDisk path
///   never went through the dispatcher because CreateSession never
///   ran).
/// - The `SISMEMBER ... == 0` post-call assertion is the load-bearing
///   SREM proof. A regression that SKIPPED the SREM (e.g., a refactor
///   that collapsed the `NoCheckpoint` arm with the `Error` arm) would
///   leave the ghost in the names set; the next ListSessions walk
///   would hit it again and tick another `not_recoverable` — an
///   observable but easy-to-miss leak.
/// - The metric assertion uses `wait_for_recovery_metric` rather than
///   a direct `session_recovery_metric` read because the
///   `try_recover_session` admission cleanup runs on Drop and (per
///   `docs-internal/test-writing-guide.md`'s synchronization-patterns guidance)
///   may race the metric observation on a heavily-loaded runner. The
///   bounded poll converges in milliseconds under normal conditions
///   and pins the exact same invariant; the helper exists specifically
///   for this ghost-sweep surface (per its docstring).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn list_sessions_ghost_member_with_no_checkpoint_is_swept() {
    let harness = setup_checkpoint_harness(1).await;

    // Capture metric baselines BEFORE injecting the ghost so the
    // assertion phase tracks only the recovery activity the
    // ListSessions call drove.
    let restored_pre = session_recovery_metric(&harness, "restored");
    let not_recoverable_pre = session_recovery_metric(&harness, "not_recoverable");
    let error_pre = session_recovery_metric(&harness, "error");
    let already_present_pre = session_recovery_metric(&harness, "already_present");

    // Inject a synthetic ghost: SADD `name\0session_id` into
    // path_names_index with no session_meta hash and no checkpoint
    // blob. This is the "truly orphaned" surface.
    sadd_composite_member(
        &harness,
        GHOST_BASE_PATH,
        GHOST_SESSION_NAME,
        GHOST_SESSION_ID,
    )
    .await;

    // Sanity-check: the ghost member is in the names set pre-call. A
    // failed SADD here (somehow) would make the post-call SISMEMBER==0
    // assertion meaningless — it would pass even if the SREM never
    // fired.
    let path_names_key = harness.tv.key_schema().path_names_index(GHOST_BASE_PATH);
    let ghost_composite = KeySchema::path_name_member(GHOST_SESSION_NAME, GHOST_SESSION_ID);
    let pre_call_member: bool = harness
        .tv
        .client()
        .pool
        .sismember(&path_names_key, ghost_composite.clone())
        .await
        .expect("SISMEMBER pre-call probe must succeed");
    assert!(
        pre_call_member,
        "pre-call sanity: composite ghost member MUST be present in \
         path_names_index before the ListSessions walk; a failed SADD would \
         make the post-call SREM assertion meaningless"
    );

    // Dispatch ListSessions. reconcile_ghost_members invokes
    // try_recover_session for the ghost; the LocalDisk adapter returns
    // Ok(None) (no blob for the synthetic session_id), so the hook
    // reports NoCheckpoint; the ghost is SREM'd and excluded from the
    // response.
    let response = handle_list_sessions(&harness.svc, list_sessions_request(GHOST_BASE_PATH))
        .await
        .expect("ListSessions MUST succeed; the ghost-sweep is silent");
    let proto_sessions = response.into_inner().sessions;

    // The ghost is absent from the response — reconcile_ghost_members
    // returns ONLY the `Recovered`/`AlreadyPresent` arms in its
    // result vec; the NoCheckpoint arm SREMs and excludes.
    assert!(
        proto_sessions.is_empty(),
        "ListSessions response MUST be empty when the only path-index member \
         is a NoCheckpoint ghost; got {} session(s). A non-empty response \
         indicates the NoCheckpoint arm somehow appended to the response, \
         or reconcile_ghost_members fell through to the Recovered arm \
         (the synthetic session_id has no blob — Recovered is impossible)",
        proto_sessions.len()
    );

    // SISMEMBER post-call: the ghost has been SREM'd from
    // path_names_index. This is the load-bearing opportunistic-cleanup
    // assertion. The Subtask body pins `SISMEMBER ... == 0` explicitly.
    let post_call_member: bool = harness
        .tv
        .client()
        .pool
        .sismember(&path_names_key, ghost_composite.clone())
        .await
        .expect("SISMEMBER post-call probe must succeed");
    assert!(
        !post_call_member,
        "post-call SISMEMBER MUST be 0 for the ghost composite member — \
         reconcile_ghost_members's NoCheckpoint arm calls srem_ghost_member \
         which SREMs both path_names_index and the legacy path_index. A \
         surviving member means the SREM never fired (regression in the \
         NoCheckpoint arm, or the Error arm silently caught the NoCheckpoint \
         outcome and conservatively skipped SREM)."
    );

    // The legacy path_index entry is also SREM'd (defense-in-depth —
    // srem_ghost_member targets both indices). A surviving entry here
    // would surface in the next walk's `present` arm if the composite
    // member somehow re-appeared.
    let path_index_key = harness.tv.key_schema().path_index(GHOST_BASE_PATH);
    let legacy_member: bool = harness
        .tv
        .client()
        .pool
        .sismember(&path_index_key, GHOST_SESSION_ID)
        .await
        .expect("SISMEMBER legacy path_index probe must succeed");
    assert!(
        !legacy_member,
        "post-call SISMEMBER MUST be 0 for the ghost session_id in the legacy \
         path_index — srem_ghost_member SREMs both indices symmetrically"
    );

    // Metric assertion: `not_recoverable` ticked by exactly 1 (the
    // single ghost). Bounded-poll via wait_for_recovery_metric per the
    // helper's docstring guidance for the ListSessions ghost-sweep
    // surface (recovery decision may land between RPC return and
    // metric observation).
    wait_for_recovery_metric(
        &harness,
        "session",
        "not_recoverable",
        not_recoverable_pre + 1,
        10,
    )
    .await;
    assert_eq!(
        session_recovery_metric(&harness, "restored"),
        restored_pre,
        "post-call `restored` MUST equal pre-call snapshot — the adapter \
         returned NoCheckpoint so the apply pipeline never ran"
    );
    assert_eq!(
        session_recovery_metric(&harness, "error"),
        error_pre,
        "post-call `error` MUST equal pre-call snapshot — no transient \
         backend failure was injected"
    );
    assert_eq!(
        session_recovery_metric(&harness, "already_present"),
        already_present_pre,
        "post-call `already_present` MUST equal pre-call snapshot — the \
         synthetic session_id never had a session_meta hash, so the \
         pre-acquire EXISTS check returned false"
    );

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// =============================================================================
// Scenario 3 — post-recovery composite-name membership preserved.
// =============================================================================

/// Pin SR-2.7 contract #3: after a recovery pass that restores N
/// sessions, EVERY original composite `name\0session_id` member from
/// `path_names_index` is still present in the set — built via
/// `KeySchema::path_name_member`, no inline string concat.
///
/// This is the dual of Scenario 1's metric assertion. Scenario 1 pins
/// the RPC-visible behaviour ("response carries the recovered
/// sessions"); Scenario 3 pins the storage-level invariant
/// (`path_names_index` membership is preserved exactly, with the
/// original composite encoding). A regression that recovered the
/// session but accidentally re-encoded the composite (e.g., a refactor
/// that switched separator from `\0` to `:` and re-`SADD`'d under the
/// new encoding) would break the duplicate-name uniqueness invariant
/// `SessionStore::create` relies on — and Scenario 1 alone would not
/// catch that regression, because the response shape and the metric
/// count would both still be correct.
///
/// Why each step is load-bearing:
///
/// - The setup mirrors Scenario 1 exactly (3 sessions on the same
///   base_path → full_valkey_wipe → restore path-index → ListSessions
///   triggers per-ghost recovery). A regression specifically in the
///   `Recovered` arm of `reconcile_ghost_members` (which the parent
///   Task body describes as "member is kept" — i.e., the recovered
///   ghost's composite is NOT SREM'd) would surface here as a missing
///   composite member.
/// - The composite is constructed via
///   [`KeySchema::path_name_member`]
///   (rather than inline `format!("{name}\0{sid}")`) so a downstream
///   rename or encoding change in the helper surfaces here as a
///   compile error AND a behavior change in production simultaneously
///   — the test pins the source-of-truth helper, not a duplicate
///   format string. The Subtask body explicitly calls this out: "built
///   via `KeySchema::path_name_member`, no inline concat".
/// - `SISMEMBER` for each composite individually pins per-member
///   presence — a `SMEMBERS` + set-membership probe would also work
///   but would hide WHICH composite went missing on regression.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn list_sessions_post_recovery_composite_name_membership_preserved() {
    let harness = setup_checkpoint_harness(1).await;

    // Setup mirrors Scenario 1 exactly — three sessions, wipe, restore
    // path-index, dispatch ListSessions.
    let session_a =
        create_session_and_wait_for_checkpoint(&harness, SESSION_BASE_PATH, SESSION_NAME_A).await;
    let session_b =
        create_session_and_wait_for_checkpoint(&harness, SESSION_BASE_PATH, SESSION_NAME_B).await;
    let session_c =
        create_session_and_wait_for_checkpoint(&harness, SESSION_BASE_PATH, SESSION_NAME_C).await;
    let sessions = vec![session_a, session_b, session_c];

    full_valkey_wipe(&harness.tv).await;
    restore_path_index_after_wipe(&harness, SESSION_BASE_PATH, &sessions).await;

    // Dispatch ListSessions through the real handler. We do not need
    // to inspect the response — Scenario 1 already pins that surface.
    // Scenario 3 inspects post-call Valkey state instead.
    let _response = handle_list_sessions(&harness.svc, list_sessions_request(SESSION_BASE_PATH))
        .await
        .expect("ListSessions MUST succeed via path-index walk recovery");

    // Every original composite member is still in path_names_index.
    // `KeySchema::path_name_member` is the source-of-truth constructor
    // — a regression that changed the encoding would surface here as
    // each `SISMEMBER` returning false (helper updated → test members
    // updated, but the production-restored entry uses the OLD
    // encoding, or vice versa). Pins SR-2.7's
    // "composite-name index membership preserved" requirement.
    let path_names_key = harness.tv.key_schema().path_names_index(SESSION_BASE_PATH);
    for session in &sessions {
        let composite_member = KeySchema::path_name_member(&session.name, &session.session_id);
        let is_member: bool = harness
            .tv
            .client()
            .pool
            .sismember(&path_names_key, composite_member.clone())
            .await
            .expect("SISMEMBER composite probe must succeed");
        assert!(
            is_member,
            "post-recovery, composite member '{composite_member}' (name={}, \
             session_id={}) MUST be present in path_names_index. A missing \
             composite indicates reconcile_ghost_members's `Recovered` arm \
             accidentally SREM'd a recovered member (regression: collapsed \
             with the NoCheckpoint arm), or apply_session_checkpoint failed \
             to restore the path_names_index entry via a different code path.",
            session.name, session.session_id
        );
    }

    // Cross-check: the names set carries EXACTLY the three original
    // members — no phantom composites, no leftover ghosts from a
    // partial wipe. SMEMBERS is unordered, so we compare as sets.
    let post_members: Vec<String> = harness
        .tv
        .client()
        .pool
        .smembers::<Vec<String>, _>(&path_names_key)
        .await
        .expect("SMEMBERS path_names_index post-call must succeed");
    let expected_members: std::collections::HashSet<String> = sessions
        .iter()
        .map(|s| KeySchema::path_name_member(&s.name, &s.session_id))
        .collect();
    let actual_members: std::collections::HashSet<String> = post_members.into_iter().collect();
    assert_eq!(
        actual_members, expected_members,
        "post-recovery path_names_index MUST carry EXACTLY the three original \
         composites (no extras, no missing). Mismatch indicates either a \
         leaked ghost from setup or a regression in reconcile_ghost_members's \
         Recovered-arm handling."
    );

    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}
