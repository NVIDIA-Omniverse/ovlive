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

// Guard tests: scripts/guard-tests.sh runs this whole module by name; rename it there too.

// Workspace-wide site-level routing guard for fred-error classification
// (b.u96 item 5 — generalization of the original b.fd9 follow-up scope).
//
// The b.fd9 sweep wired every transient-vs-permanent Valkey error site
// through one of two surface mappers:
//   * `map_transient_fred_err` — sites inside `retry_aborted_with_jitter`
//     loops (transient → `ServiceError::Aborted`).
//   * `map_fred_err_for_grpc` — single-shot sites surfacing directly to a
//     gRPC handler (transient → `ServiceError::Unavailable`).
//
// Per-site fault-injection tests in `transient_classification_test.rs`,
// `stream_mux_fault_test.rs`, `b_j9j_tests.rs` and friends cover the
// *behaviour* of each named site. This file covers the *shape* of the
// code — a grep-based source walk that fails if any error path inside
// `src/**/*.rs` lands on a direct `format!` wrap of a fred error into an
// INTERNAL-mapped `ServiceError` that should instead route through one
// of the two surface mappers. "INTERNAL-mapped" is three source
// spellings since b.6de sub-task 2, not one literal; they are named and
// justified at `WRAP_TOKEN_INTERNAL_FORMAT` below.
//
// b.u96 item 5: the original test scanned only the b.fd9-named functions
// (`list_layer_names`, `get_layer_mapping`, `read_object_header`,
// `process_pending`, `presence.rs`). The narrower scope was a clean
// b.fd9 acceptance criterion but couldn't catch new violations in
// unrelated subsystems (e.g. a new GC helper, a new presence call, a
// new storage helper added between b.fd9 and today). This generalized
// scan covers ALL `src/**/*.rs` files with an explicit allowlist of
// every pre-existing site that has been audited and confirmed
// legitimate. Any *new* `ServiceError::Internal(format!` wrap of a fred
// error must either:
//   (a) replace it with one of the two surface mappers (correct ~99% of
//       the time for fred-error sites), or
//   (b) add the call site to the allowlist with a one-line rationale
//       comment justifying the exception.
//
// The allowlist is intentionally *per-site*, not per-function or
// per-file: each entry is a (file path, distinctive substring) pair
// precise enough to allow the legitimate site without silently waving
// through *new* violations on the same line, function, or file. A
// future commit that adds a new direct-Internal wrap in `session.rs`
// must extend the allowlist explicitly, with its own rationale row.
//
// The test is gated on the production-default build (`cfg(test)` only)
// so it runs as part of the regular `cargo test` suite — no
// `test-fault-injection` feature required.

use std::path::{Path, PathBuf};

/// One source file the scan walks. Each `Entry::source` is loaded via
/// `include_str!` at compile time so the scan has no I/O cost at
/// runtime — the bytes are baked into the integration-test binary.
struct Entry {
    /// Relative file path (under the repo root) used for error messages
    /// and matched against the allowlist `(file, needle)` entries.
    file: &'static str,
    /// Source text. Loaded via `include_str!` keyed by `file`.
    source: &'static str,
}

// The expanded workspace-wide scan. The Engineer's b.u96 diff did not
// add any new files; the list below is the union of (a) the original
// b.fd9 scope and (b) every other `src/**/*.rs` file that contains at
// least one occurrence of `ServiceError::Internal(format!` per a
// repo-wide grep. Files with no occurrences are omitted to keep the
// compile-time `include_str!` footprint small; a regression that adds
// a new file with a direct-Internal wrap would surface in the next
// audit (and the maintenance burden is identical either way — we
// would add the new file path here).
//
// **Deliberate exception — `src/grpc/presence.rs`:** kept in SOURCES
// even though it currently has zero `ServiceError::Internal(format!`
// matches, because the dedicated `presence_rs_routes_fred_errors_through_mappers`
// test below pins the b.fd9 invariant ("presence.rs has been audited
// clean, the next regression adds one wrap") at a tighter granularity
// than the workspace-wide scan. A regression that adds a new wrap to
// presence.rs would fail BOTH the workspace-wide test and the
// presence-specific test; keeping presence.rs in SOURCES is what
// makes the latter possible (the test reads the file's source via
// `SOURCES.iter().find(...)`). Removing presence.rs from SOURCES
// would silently delete the b.fd9-scope subsystem pin.
//
// To regenerate this list, take the union of the three scanned spellings
// (see `is_internal_mapped_wrap_anchor`) — the first literal alone under-covers
// the file set, which is what b.6de's review caught:
//
//   rg -l 'ServiceError::Internal\(format!' src/
//   rg -l -U 'ServiceError::PermanentKeyspaceRefusal \{[\s\S]{0,200}?detail: format!\(' src/
//   rg -l 'let (mut )?\w+ =\s*$|let (mut )?\w+ = format!\(' src/    # then keep
//       # only files where that binding is handed to one of the two above
//
// …and then let `sources_covers_every_file_holding_a_scanned_anchor` below be
// the authority: it walks `src/**/*.rs` at runtime with the real predicate and
// fails naming any file this list is missing. The list is a compile-time
// `include_str!` convenience; the walk is the completeness proof.
const SOURCES: &[Entry] = &[
    Entry {
        file: "src/checkpoint/apply.rs",
        source: include_str!("../../../src/checkpoint/apply.rs"),
    },
    Entry {
        file: "src/checkpoint/recovery.rs",
        source: include_str!("../../../src/checkpoint/recovery.rs"),
    },
    Entry {
        file: "src/checkpoint/session.rs",
        source: include_str!("../../../src/checkpoint/session.rs"),
    },
    Entry {
        file: "src/error.rs",
        source: include_str!("../../../src/error.rs"),
    },
    Entry {
        file: "src/grpc/auth_helpers.rs",
        source: include_str!("../../../src/grpc/auth_helpers.rs"),
    },
    Entry {
        file: "src/grpc/presence.rs",
        source: include_str!("../../../src/grpc/presence.rs"),
    },
    Entry {
        file: "src/grpc/session_cache.rs",
        source: include_str!("../../../src/grpc/session_cache.rs"),
    },
    Entry {
        file: "src/grpc/subscribe.rs",
        source: include_str!("../../../src/grpc/subscribe.rs"),
    },
    Entry {
        file: "src/storage/client.rs",
        source: include_str!("../../../src/storage/client.rs"),
    },
    Entry {
        file: "src/storage/counters.rs",
        source: include_str!("../../../src/storage/counters.rs"),
    },
    Entry {
        file: "src/storage/diff_read.rs",
        source: include_str!("../../../src/storage/diff_read.rs"),
    },
    Entry {
        file: "src/storage/external_values.rs",
        source: include_str!("../../../src/storage/external_values.rs"),
    },
    Entry {
        file: "src/storage/layer.rs",
        source: include_str!("../../../src/storage/layer.rs"),
    },
    Entry {
        file: "src/storage/lua.rs",
        source: include_str!("../../../src/storage/lua.rs"),
    },
    Entry {
        file: "src/storage/retry.rs",
        source: include_str!("../../../src/storage/retry.rs"),
    },
    Entry {
        file: "src/storage/session.rs",
        source: include_str!("../../../src/storage/session.rs"),
    },
    Entry {
        file: "src/storage/structural.rs",
        source: include_str!("../../../src/storage/structural.rs"),
    },
    Entry {
        file: "src/storage/structural_cache_cold_start.rs",
        source: include_str!("../../../src/storage/structural_cache_cold_start.rs"),
    },
    Entry {
        file: "src/storage/structural_cache_read.rs",
        source: include_str!("../../../src/storage/structural_cache_read.rs"),
    },
    Entry {
        file: "src/storage/structural_cache_repair.rs",
        source: include_str!("../../../src/storage/structural_cache_repair.rs"),
    },
    Entry {
        file: "src/storage/structural_cache_sync.rs",
        source: include_str!("../../../src/storage/structural_cache_sync.rs"),
    },
    Entry {
        file: "src/storage/tail_slice.rs",
        source: include_str!("../../../src/storage/tail_slice.rs"),
    },
    Entry {
        file: "src/storage/test_support/structural_commit.rs",
        source: include_str!("../../../src/storage/test_support/structural_commit.rs"),
    },
];

/// Allowlist of legitimately-permanent INTERNAL-mapped `format!` wraps
/// (`ServiceError::Internal`, and since b.6de sub-task 2 also
/// `ServiceError::PermanentKeyspaceRefusal` — see
/// [`WRAP_TOKEN_INTERNAL_FORMAT`] for the full signature set).
/// Each row pairs (file, distinctive substring) — both must
/// match for the entry to be excused. The substring is matched
/// verbatim against the wrap's own text ([`site_text`]: the anchor line
/// and its lookahead, up to the next wrap); pick something distinctive
/// enough that a *different* violation in the same file doesn't
/// silently fall under the same row. Each row must match exactly one
/// site unless [`ALLOWLIST_MULTI_SITE_ROWS`] declares otherwise.
///
/// Categories of legitimate exceptions:
///
///   * **Structured non-fred error conversions** — `From<DeltaError>`,
///     `From<DiffError>`, `From<AuthError::InvalidResponse>`,
///     `From<OodfError>` in `error.rs`. These wrap deterministic /
///     structural error types, not fred I/O — `Internal` is the
///     correct gRPC surface (a malformed delta isn't retriable; a
///     protocol-level auth-backend contract violation isn't retriable
///     either). Documented per-variant in `error.rs`.
///
///   * **Mapper-internal permanent branches** — `client.rs`'s
///     `map_snapshot_load_err` and `map_fred_err_for_grpc` each have a
///     permanent-classification branch that wraps the error as
///     `Internal`. Those branches ARE the canonical routing target —
///     they're how the rest of the codebase produces `Internal` for
///     non-transient fred kinds. Allowlisting them inside themselves
///     is not circular; it's how the helpers are spelled.
///
///   * **MULTI / pipeline queue-time sites** — the fred return from
///     *queueing* a command into a `Pipeline` (incl. the `WatchedTxn`
///     pipeline) or a fred `Transaction` (`client.multi()`), before
///     anything reaches the wire. fred's source is why these stay
///     `Internal`: `Pipeline::send_command` and `WithOptions::send_command`
///     only push onto a local buffer (`fred-10.1.0/src/clients/pipeline.rs`,
///     `options.rs`); `Transaction::send_command` adds two client-side slot
///     checks, `disallow_all_cluster_commands` and `update_hash_slot`
///     (`transaction.rs`), both raising `ErrorKind::Cluster`; and the only
///     other queue-time failure is argument conversion. None of that is a
///     transport blip — every transaction here uses one hash tag, so the
///     same-node check cannot fire, and the "missing cluster routing state"
///     check can fire only before the client's first cluster sync. Routing
///     such a site through a mapper would be actively wrong:
///     `is_transient_fred_error` treats `ErrorKind::Cluster` as transient,
///     so a cross-slot programming bug would come back as a retriable
///     `UNAVAILABLE`. SDD §15.3 states the same rule. What this category
///     does NOT cover is a plain round trip — a data-pool `GET` / `SET` /
///     `SREM` / `HGETALL` / `SMEMBERS` / `XADD`, or an `EXEC` / `WATCH`
///     dispatch — whatever retry layer does or does not surround it: those
///     reach the server, can time out or lose their connection, and route
///     through a mapper (b.62m moved the last of them off this list).
///
///   * **Corrupt-data / parse failures on Valkey responses** — the
///     fred call succeeded, but the response payload failed
///     validation (e.g. `s.parse::<u64>()` on a layer-mapping value
///     that should be numeric). A real producer bug, not a network
///     blip — `Internal` is correct.
///
///   * **External-value integrity violations** — `diff_read.rs`'s
///     "external value missing: {hash}" path indicates a reference-
///     count / GC contract violation, not a transient blip.
///
///   * **`read_object_snapshot` parse pre-condition** — the caller
///     supplied a non-numeric `object_id` string; programmer error.
///
///   * **`update_layer`'s relay precondition** — a caller passed a delta
///     without the non-empty relay OMPF every delta-carrying commit must
///     carry (SR-1.2); refused before any store I/O, so no fred error is
///     involved. Programmer error.
///
///   * **Snapshot-read answer and cut invariants** — a
///     `read_object_snapshot` answer its consumer can never receive
///     (`TOO_LARGE` from the unfenced fan-out, a fan-out reply of the wrong
///     shape, an `object_snapshot_read.lua` reply that is not this binary's
///     script's shape), or a build handed a cut it cannot serve. The fred
///     call, where there was one, already succeeded, and its errors route
///     through `classify_object_snapshot_read_err` /
///     `classify_snapshot_read_err`; a retry would see the same answer.
///
///   * **`subscribe.rs::out_of_window XRANGE`** — the out-of-window
///     fallback path is intentionally non-retriable (a client whose
///     cursor fell out of the trimmed window must restart from
///     scratch, not retry with the same cursor). A transient blip
///     here doesn't change that — the client still needs the
///     bootstrap-and-restart contract.
///
///   * **`retry.rs`'s defensive unreachable** — the retry helper's
///     "loop exited without result" arm is unreachable in practice;
///     surfacing `Internal` is a defensive safety net so the caller
///     doesn't silently hang. Pre-b.k1r this path didn't exist; the
///     loop simply panicked.
const ALLOWLIST: &[(&str, &str)] = &[
    // ── error.rs: structured non-fred From-impls ────────────────────────
    //
    // DeltaError -> Internal: catastrophic delta-state violations
    // (InconsistentState, SequenceOverflow, SetOrderOverflow). Pre-fix
    // these mapped to InvalidArgument with a u64::MAX-based message
    // that was never reachable at runtime; b.31g moved them to
    // Internal because the trigger is server-side state accumulation,
    // not client input. Non-fred error, no transient classification.
    ("src/error.rs", "delta error: {e}"),
    // DiffError::SectionTreeTooDeep -> Internal: stored state has
    // accumulated past the per-subscriber depth ceiling; producer bug,
    // not a network blip.
    ("src/error.rs", "diff error: {e}"),
    // AuthError::InvalidResponse -> Internal: protocol-level contract
    // violation from the external auth backend. Pre-b.ohw this lived
    // alongside Timeout/Unreachable; b.ohw split the transient kinds
    // out to Unavailable. The InvalidResponse arm stays Internal
    // because a malformed payload from the backend is non-retriable.
    // AuthError::UserNotFound (reached only with authorization off)
    // shares the arm, so this one site covers both.
    ("src/error.rs", "auth backend: {e}"),
    // OodfError -> Internal: AllocationFailed / InternalError(_) /
    // Ffi(_) / BuildUnknownCommandTag(_). The branch comment in
    // error.rs explains each variant's reachability.
    ("src/error.rs", "OODF error: {e}"),
    // ── auth_helpers.rs: stored base path refused by the auth backend ───
    //
    // `invalid_request_error`'s `BasePathSource::Session` arm: the storage
    // service answered the permission probe for a session's stored base_path
    // with a malformed-request verdict (`AuthResult::InvalidRequest`). Not a
    // fred error — no Valkey call is wrapped; it is a deterministic backend
    // refusal of a path live-edit accepted earlier, so a retry sees the same
    // answer. `INTERNAL` is the SDD §14.1 decision: the client never sent
    // this path, and an operator has to look.
    (
        "src/grpc/auth_helpers.rs",
        "refused the session's stored base_path",
    ),
    // ── client.rs: canonical mapper-internal permanent branches ─────────
    //
    // map_snapshot_load_err and map_fred_err_for_grpc each have a
    // permanent-classification branch that wraps as Internal. These
    // are how the rest of the codebase produces Internal for
    // non-transient fred kinds; allowlisting them inside the helpers
    // themselves is the contract this whole test exists to enforce
    // elsewhere. One row, two sites: see `ALLOWLIST_MULTI_SITE_ROWS`.
    ("src/storage/client.rs", "{op} failed: {e}"),
    // ── retry.rs: defensive unreachable ─────────────────────────────────
    //
    // The retry helper's defensive Internal at loop exit; this arm is
    // unreachable in practice (the loop returns on success,
    // exhaustion, or a non-retriable error) but exists so a future
    // change to control-flow doesn't silently hang the caller.
    ("src/storage/retry.rs", "retry loop exited without result"),
    // ── layer.rs: corrupt-data parse failures on Valkey responses ───────
    //
    // get_layer_mapping: s.parse::<u64>() on the layer mapping value.
    // The fred GET returned Ok(Some(s)); the failure mode is a malformed
    // mapping value (someone wrote a non-numeric string), not a network
    // blip. Pre-existing exception from the original b.fd9 scope.
    ("src/storage/layer.rs", "corrupt layer mapping value"),
    // b.ua9 REMOVED THE `corrupt old object_id` ROW — do not restore it. The
    // create_layer overwrite no longer refuses a non-numeric old mapping value
    // after its commit point: it overwrites it with no fence, wake-up or
    // cleanup and logs ERROR, so no `Internal` wrap remains there.
    // write_object and write_restored_object: each opens by parsing the
    // caller's oid_str into u64. Caller-supplied stringification produced a
    // non-numeric value; programmer error, not blip. One row, two sites: see
    // `ALLOWLIST_MULTI_SITE_ROWS`.
    ("src/storage/layer.rs", "corrupt oid_str"),
    // ── layer.rs / external_values.rs / session.rs: MULTI / pipeline
    //    queue-time sites ─────────────────────────────────────────────────
    //
    // The rows in this category are spread through the list below —
    // external_values.rs's pipeline pushes, layer.rs's `write_object` /
    // `queue_object_state` pushes, session.rs's `create_once`
    // / `delete` pushes — and each wraps the fred return from QUEUEING a
    // command into a `Pipeline` / `WatchedTxn` pipeline or a fred
    // `Transaction`: a local buffer push, never a round trip. Why that stays
    // `Internal` is the category note on `ALLOWLIST` above: the only ways it
    // can fail are argument conversion and fred's two client-side
    // `ErrorKind::Cluster` slot checks, and a mapper would classify the latter
    // as a transient `UNAVAILABLE` for what is really a cross-slot programming
    // bug (SDD §15.3). Whether a retry loop surrounds the site plays no part
    // in the decision.
    //
    // The substring on each row must match exactly the site it excuses (the
    // wrap-message verb + operand). A different violation on the same line —
    // e.g. a typo "HSET object header" — would not match these needles and
    // would fail the test. A needle that is a substring of ANOTHER row's
    // literal, or of several sites' literals, silently excuses every one of
    // them, and a fence that only asks "does the row still match something?"
    // cannot see that, because the broad row does. So the fence below counts:
    // each row must match exactly one site (or the count declared for it in
    // `ALLOWLIST_MULTI_SITE_ROWS`), no site may be matched by two rows, and no
    // needle may be a substring of another row's needle in the same file.
    //
    // b.62m REMOVED NINE ROWS (TWELVE SITES) FROM THIS CLUSTER — do not
    // restore them. The earlier version of this note claimed every row here
    // was a queue-time wrap whose surrounding retry layer decided retry vs.
    // exhaustion. Neither half held for these nine: each wrapped a plain
    // data-pool round trip (`GET` / `SET` / `SREM` / `HGETALL` / `SMEMBERS` /
    // `XADD`), and `create_layer`, `SessionStore::delete`, `list` and
    // `get_session_meta` run no retry loop at all. Several needles were also
    // too broad — `"HGETALL failed"` excused three sites, `"SMEMBERS names
    // failed"` two, and `"GET ext_data failed"` is a substring of the separate
    // `"queue GET ext_data failed"` row, so the at-least-one staleness fence
    // of the time could never have flagged it. Each site now routes through a
    // mapper:
    //
    //   session.rs         "HGETALL failed" (delete)          -> map_fred_err_for_grpc
    //   session.rs         "SREM path_index failed"           -> map_fred_err_for_grpc
    //   session.rs         "SREM path_names failed"           -> map_fred_err_for_grpc
    //   session.rs         "SMEMBERS names failed" (list)     -> map_fred_err_for_grpc
    //   session.rs         "HGETALL failed" (list)            -> map_fred_err_for_grpc
    //   session.rs         "HGETALL failed" (get_session_meta) -> map_fred_err_for_grpc
    //   session.rs         "HSET failed" (finish_create)      -> undo_create_attempt
    //   session.rs         "SMEMBERS names failed" (re-scan)  -> undo_create_attempt
    //   layer.rs           "GET layer mapping failed"         -> map_fred_err_for_grpc
    //   layer.rs           "SET layer mapping failed"         -> map_fred_err_for_grpc
    //   layer.rs           "XADD deleted on old stream failed" -> map_fred_err_for_grpc
    //                      (since b.ua9 a WARN in the overwrite tail, answered OK)
    //   external_values.rs "GET ext_data failed"              -> map_fred_err_for_grpc
    //
    // The two `undo_create_attempt` sites (and the in-doubt `SISMEMBER`, which
    // used to return a bare `Aborted`) run AFTER `create_once`'s commit was
    // sent, so they fail through the undo rather than straight through a
    // mapper; the undo picks the mapper by whether it released the name (see
    // `undo_create_attempt` / SDD §7.10). Their shape is pinned by
    // `create_session_post_commit_failures_route_through_undo_create_attempt`.
    //
    // b.bhb REMOVED SIX ROWS FROM THIS CLUSTER — do not restore them.
    // The txn pool now writes each command exactly once
    // (`max_command_attempts = 1`, `ValkeyClient::connect`), so a routine
    // reconnect that fred used to absorb by re-issuing the command now
    // surfaces at these EXEC / WATCH sites as an error. A direct `Internal`
    // there would turn a transient blip into a terminal failure, so each of
    // these six moved onto a canonical mapper and no longer wraps directly:
    //
    //   session.rs        "WATCH failed"                       -> map_transient_fred_err
    //   session.rs        "EXEC session DEL meta+layers failed" -> map_transient_fred_err
    //   session.rs        "EXEC failed"                        -> map_transient_fred_err
    //   layer.rs          "EXEC session mapping failed"        -> map_fred_err_for_grpc
    //   layer.rs          "EXEC object write failed"           -> map_fred_err_for_grpc
    //   external_values.rs "EXEC ext store_values failed"      -> map_fred_err_for_grpc
    //
    // That last site has since moved again: b.rej replaced the per-hash
    // `MULTI/EXEC` with one `EVALSHA` of `scripts/ext_store_value.lua` on the
    // data pool, so the op string the mapper is called with today is
    // `"EVALSHA ext store_values"`. The mapper choice is unchanged and so is the
    // reasoning below. The layer.rs `EXEC session mapping` site moved the same
    // way under b.v6d: `create_layer`'s mapping `MULTI` (and the overwrite's
    // bare mapping `SET`, b.62m's `SET layer mapping failed` row above) are now
    // one `EVALSHA` of `scripts/create_layer_commit.lua`, mapped as
    // `"EVALSHA create_layer_commit"` by the same `map_fred_err_for_grpc`.
    //
    // The bare `session.rs "EXEC failed"` row is the one with two steps behind
    // it. `create_once`'s EXEC no longer wraps a raw fred error at all: the
    // `WatchedTxn::exec` -> `WatchedExec` decoding re-shaped the site first
    // (the reply is now decoded into a committed / void-watch / in-doubt
    // verdict rather than an opaque `Result`), and the residual error arm was
    // then routed onto the canonical transient mapper —
    // `map_transient_fred_err("EXEC create_session", e)` (`src/storage/session.rs`).
    // Both the old needle and the old direct `Internal(format!(...))` wrap are
    // gone, so the row matched nothing and failed the allowlist-liveness fence.
    //
    // Which mapper each site takes is decided by whether it sits inside a
    // retry loop (`Aborted`, picked up by `retry_aborted_with_jitter`) or
    // reaches a gRPC handler directly (`Unavailable`, retried by the client)
    // — `store_values` runs no retry loop of its own, which is why it takes
    // the gRPC mapper.
    // ── counters.rs: corrupt-data parse failures on Valkey responses ────
    //
    // current_object_id and seed_object_id_if_below each parse a String
    // reply (GET counter value / Lua-returned post-script value) into u64.
    // The fred call succeeded; the parse failure means the counter key
    // holds a non-numeric value — a real producer / Valkey-corruption
    // bug, not a transient blip. Internal is correct (same shape as
    // layer.rs's "corrupt layer mapping value" entry below).
    (
        "src/storage/counters.rs",
        "GET object_id counter: non-numeric reply",
    ),
    (
        "src/storage/counters.rs",
        "seed_object_id_if_below: non-numeric reply",
    ),
    // next_object_id: the alloc_object_id.lua EVAL succeeded (fred call
    // Ok) but the INCR result is non-positive, which is only possible if
    // the counter key holds a corrupt (negative / overflow) value —
    // integrity violation, not a transient blip. The absent-key case is
    // the nil reply mapped to Ok(None) above this arm, so it never
    // reaches this wrap. The needle quotes this site's own wording: the
    // bare `"corrupt counter"` it replaced also matched the two
    // non-numeric-reply sites above, each already excused by its own row.
    (
        "src/storage/counters.rs",
        "INCR returned a non-positive id (corrupt counter)",
    ),
    // b.rej REMOVED TWO MORE ROWS FROM THIS CLUSTER — `queue SET ext_data
    // failed` and `queue HINCRBY ext_meta ref_count failed`. Do not restore
    // them. `store_values`'s per-hash `MULTI { SET; HINCRBY } EXEC` is now a
    // single `EVALSHA` of `scripts/ext_store_value.lua` on the data pool, and a
    // script has no local queueing step whose argument / slot check could fail
    // separately from the call — so both wraps are gone from the source and
    // both rows would have failed the allowlist-liveness fence below.
    //
    // external_values.rs `read_batch_with_recovery_counter`'s two pipeline
    // queue pushes (`GET ext_data`, `HEXISTS ext_meta ref_count`) — local
    // buffer pushes into a fred `Pipeline`. The `pipeline.all()` round trip
    // beside them is NOT allowlisted: it routes through
    // `map_fred_err_for_grpc` (no retry loop wraps it, so a transient blip
    // answers retriable `UNAVAILABLE`) and is therefore not scanned at all.
    // Neither is `load_values`'s plain `GET ext_data` round trip, which uses
    // the same mapper.
    (
        "src/storage/external_values.rs",
        "queue GET ext_data failed",
    ),
    (
        "src/storage/external_values.rs",
        "queue HEXISTS ext_meta ref_count failed",
    ),
    // b.v6d REMOVED TWO ROWS FROM THIS CLUSTER — `SET layer mapping (txn)
    // failed` and `SADD layers index (txn) failed`. Do not restore them.
    // `create_layer`'s new-layer `MULTI { SET mapping; SADD layers index }` and
    // its overwrite branch's bare mapping `SET` are now one `EVALSHA` of
    // `scripts/create_layer_commit.lua` on a `txn_pool` connection, whose
    // transport failure routes through `map_fred_err_for_grpc("EVALSHA
    // create_layer_commit", …)`. A script has no local queueing step, so both
    // queue-time wraps are gone from the source.
    //
    // ── layer.rs: the `CreateLayer` operation record (b.v6d) ────────────────
    //
    // Six structured refusals on the create-with-operation-id path. None wraps
    // a fred error: each is reached only after the fred call that fed it
    // returned `Ok` (or, for the two keyspace refusals, holding the one
    // tagged `WRONGTYPE` verdict), and no retry can change the bytes it read.
    //
    // `CreateOperationRecord::parse`: a `le:{sid}:createop:{operation_id}`
    // value that is not `<object_id>:<previous_object_id>:<digest>`. Neither
    // writer produces one, so it was edited out of band; guessing would either
    // re-create a committed layer or answer the wrong object.
    (
        "src/storage/layer.rs",
        "create_layer operation record {key} is unparseable",
    ),
    // `create_layer_with_operation`'s `MGET` of the mapping and the record
    // returned a value count other than two — a reply-shape fault, not a blip.
    (
        "src/storage/layer.rs",
        "create_layer MGET of two keys returned",
    ),
    // The commit script answered `ALREADY_COMMITTED` without the record it
    // promises in element 1 — unreachable against the script in this repo.
    (
        "src/storage/layer.rs",
        "create_layer commit script answered {CREATE_COMMIT_REPLY_ALREADY_COMMITTED}",
    ),
    // The commit script's reply carried neither tag — a stale or foreign
    // script body behind the SHA, which no retry repairs.
    (
        "src/storage/layer.rs",
        "create_layer commit script returned an unexpected reply",
    ),
    // The commit script answered `MAPPING_CHANGED` naming the very value it
    // compares against — unreachable against the script in this repo.
    (
        "src/storage/layer.rs",
        "create_layer commit script answered MAPPING_CHANGED",
    ),
    // The `MGET` of the mapping, the commit claim and (with an operation id)
    // the operation record in `create_layer_with_operation`'s check after a
    // failed commit run returned a value count other than the key count — a
    // reply-shape fault, not a blip.
    (
        "src/storage/layer.rs",
        "create_layer commit check MGET returned",
    ),
    // The commit script's `WRONGTYPE_KEY` refusal (the layers index or the
    // operation record holds a foreign type), re-emitted as a
    // `PermanentKeyspaceRefusal`. Same category and reasoning as the
    // `"{} keyspace type error: {}"` probe row below: the arm holds a REAL
    // fred error, but only the one whose `details()` carries the script's own
    // tag. The needle stops at `{}` so it cannot match the record-read
    // refusal's `{KEYSPACE_WRONGTYPE_TAG}` spelling below it.
    (
        "src/storage/layer.rs",
        "create_layer keyspace type error: {}",
    ),
    // `classify_create_operation_record_read_error`: the handler's `GET` of
    // the operation record answered `WRONGTYPE` — the predicate
    // `classify_restore_header_read_error` uses (kind `InvalidArgument`,
    // `details` prefix-anchored on `WRONGTYPE`). Permanent by construction: no
    // retry can `GET` a non-string. Every other error from that `GET` routes
    // through `map_fred_err_for_grpc("GET create operation record", …)`.
    (
        "src/storage/layer.rs",
        "non-string value, expected string (create-layer operation record)",
    ),
    // layer.rs Lua-script-returned structured-error wraps. These wrap
    // a `details` string emitted by the Lua fast path (NOT a fred
    // error); the script returns `ENCODING:...` / `SEQ_OVERFLOW:...` /
    // `SET_ORDER_OVERFLOW:...` to signal non-retriable conditions
    // (Lua encoder bug, sequence ceiling reached, set_order ceiling
    // reached). The fred call already succeeded; these arms wrap the
    // script's structured error payload, not a transient blip.
    ("src/storage/layer.rs", "update_layer lua encoder bug"),
    (
        "src/storage/layer.rs",
        "update_layer sequence ceiling reached",
    ),
    (
        "src/storage/layer.rs",
        "update_layer set_order ceiling reached",
    ),
    // `update_layer_with_summary`'s SR-1.2 relay precondition: a delta
    // without non-empty relay OMPF is refused before admission and before
    // any store I/O. A caller-contract failure, no fred call involved.
    (
        "src/storage/layer.rs",
        "update_layer called with a delta and no non-empty relay OMPF",
    ),
    // b.qpf: two malformed-reply / corrupt-data integrity refusals on the
    // external-value reference-bookkeeping path. Neither wraps a fred
    // error — every fred round trip feeding them already routes through
    // `map_snapshot_load_err`, and both of these arms are reached only
    // AFTER the fred call returned `Ok`.
    //
    // `update_layer_once_lua`'s reply decode: the EVALSHA succeeded and
    // returned `Vec<String>`, but element 1 (the new `seq`) did not parse
    // as `u64`. The script in this repo always emits
    // `string.format("%d", new_seq)` first, so this is reachable only
    // against a stale cached script body on the server — a malformed
    // reply, not a network blip. Refusing is the point: the alternative is
    // guessing a sequence number the caller would report to a client.
    (
        "src/storage/layer.rs",
        "update_layer lua returned an unparseable reply head",
    ),
    // The `ALREADY_LANDED` arm of the same decode: the script answered the
    // tag without a record carrying this call's commit token. Unreachable
    // against the script in this repo (it answers the tag only after matching
    // the token); a protocol fault, not a fred error, and no retry can change
    // the reply shape.
    (
        "src/storage/layer.rs",
        "update_layer lua answered {ALREADY_LANDED_REPLY_TAG} without a record",
    ),
    // `decode_fenced_reply`, the one decode of the script's step-1d reply in
    // both modes: the script answered the `FENCED` tag without exactly one
    // decimal after it. Unreachable against the script in this repo (it always
    // pairs the tag with `string.format("%d", ms)`); a protocol fault, not a
    // fred error, and no retry can change the reply shape.
    (
        "src/storage/layer.rs",
        "update_layer lua answered {FENCED_REPLY_TAG} without exactly one decimal retry",
    ),
    // `read_release_ring_window`, the WATCH path's slot-window read of the
    // commit release ring: the `HMGET` returned `Ok` but answered a different
    // number of values than fields requested. A reply-shape integrity
    // violation, not a fred error — the `HMGET` and each value's conversion
    // already route through `map_snapshot_load_err`. The attempt ends
    // `Internal`, which by design never runs the lost-reply lookup.
    ("src/storage/layer.rs", "HMGET commit release ring answered"),
    // `parse_commit_token_record`: a commit-token record that carries this
    // call's token but whose `<seq>:<server_ts>` tail does not parse. Corrupt
    // data written out of band — neither backend writes that shape — and a
    // retry would read the same bytes, while re-applying would risk the very
    // double commit the record exists to stop. Not a fred error.
    (
        "src/storage/layer.rs",
        "update_layer commit-token record carries this call's token",
    ),
    // `read_ext_pre_image`'s per-chunk frame-count check
    // (`decode_ext_pre_image_chunk`, b.nxz) before it pairs a chunk's frames
    // with its rows: the chunk's pipeline call returned `Ok`, but its row frame
    // count is not two per queued row (one `HEXISTS <row> value` plus one
    // `HGET <row> hash`). Frames are paired with rows POSITIONALLY, so a misaligned reply
    // would map a row key to another row's `hash` (or `value` presence) and
    // enqueue a `ref_count` decrement against the WRONG
    // blob (over-release → GC deletes `ext:meta` + `ext:data` under a live
    // reference). A structural fault (client-library change, a pipeline
    // that dropped a command) that no retry can repair, so `Internal`, not
    // `Aborted` — the same fail-closed posture `hash_from_hex` and
    // `enqueue_ext_releases_from_tokens` take on malformed input.
    (
        "src/storage/layer.rs",
        "external-value pre-image pipeline returned {} row frame(s)",
    ),
    // The same read's phase 2 (b.26e Stage 2, `decode_ext_count_chunk`): one
    // `HGET obj:ext <hash>` per touched hash, paired with its hash by
    // position. A chunk whose pipeline returned `Ok` with a frame count other
    // than one per queued count read would pair a count with the wrong hash
    // and plan the reference bookkeeping against it. The same structural
    // fault, fail-closed for the same reason; not a fred error.
    (
        "src/storage/layer.rs",
        "external-value pre-image count pipeline returned {} frame(s)",
    ),
    // layer.rs write_object's MULTI queue-time site cluster — every
    // HSET/SADD/ZADD/XADD queue site between the MULTI and the EXEC.
    ("src/storage/layer.rs", "HSET obj header failed"),
    ("src/storage/layer.rs", "HSET section failed"),
    ("src/storage/layer.rs", "SADD sections index failed"),
    ("src/storage/layer.rs", "RPUSH children failed"),
    // b.zvd: the children list is DEL'd immediately before its RPUSHes, so a
    // restore that materializes over a surviving keyspace replaces the list
    // rather than appending behind it. Same MULTI, same queue-time category.
    ("src/storage/layer.rs", "DEL children list failed"),
    // b.jdr: the children-index SADD queued in the same MULTI as its RPUSHes.
    ("src/storage/layer.rs", "SADD children index failed"),
    // b.zvd: both value shapes are HDEL'd immediately before each value HSET so
    // a row can never carry `value` and `hash`+`size` at once. Queued in the
    // same MULTI as the HSET beside it.
    ("src/storage/layer.rs", "HDEL field value shape failed"),
    ("src/storage/layer.rs", "HSET field failed"),
    ("src/storage/layer.rs", "ZADD fields index failed"),
    (
        "src/storage/layer.rs",
        "HDEL time sample value shape failed",
    ),
    ("src/storage/layer.rs", "HSET time sample failed"),
    ("src/storage/layer.rs", "ZADD time samples index failed"),
    ("src/storage/layer.rs", "XADD 1-0 queue failed"),
    // b.ua9 A11: the create operation's id and digest, queued on the new
    // header in the same materialization MULTI as the XADD above.
    ("src/storage/layer.rs", "HSET create operation queue failed"),
    // b.ph9: fail-closed stream-ID collision refusals and the WRONGTYPE_KEY
    // translation — structured integrity-violation errors built from a Lua
    // script reply or a probe verdict; no fred error is being wrapped (the
    // fred round-trips feeding them route through the canonical mappers).
    //
    // b.459 nuance on the last row, which now has two arms. The txn arm
    // (`probe_keyspace_types`) reaches its `format!` holding a REAL
    // `fred::error::Error` — but only the one whose `details()` carries the
    // script's own `WRONGTYPE_KEY:` tag, i.e. a verdict the script
    // deliberately encoded as an error reply because Lua has no other way to
    // return one. It is an integrity violation the same way the Lua arm's is,
    // it is permanent by construction (no retry can retype the key), and the
    // wrap re-emits only `details()` — never the fred kind. Every OTHER error
    // from that same round trip falls to the arm below it and routes through
    // `map_snapshot_load_err`, so a Valkey blip on the probe is still a
    // retriable `Aborted` and this exemption covers exactly the tagged verdict.
    (
        "src/storage/layer.rs",
        "already exists for freshly-minted object",
    ),
    ("src/storage/layer.rs", "at or above the attempted"),
    // b.zvd: the txn probe's refusal prefix is now composed at runtime from
    // `ProbeSite::message_prefix()` (`update_layer` | `create_layer`), so the
    // source literal changed shape — `probe_keyspace_types`' one `format!` is
    // shared by `update_layer_once_watch`'s pre-MULTI probe and `write_object`'s.
    // Spelled as the verbatim source text rather than as a shortened needle
    // that would match both it and the Lua arm below: this file's convention is
    // an entry per call site quoting the literal exactly (cf. the
    // `STREAM_ID_COLLISION_UPDATE_PREFIX` row), so a widened substring would
    // silently re-cover a third site someone added later.
    //
    // b.6de sub-task 2 HOISTED this `format!` out of the construction: it is now
    // `let detail = format!(…)` consumed by `ServiceError::Internal(detail)` on
    // the `ProbeSite::Update` arm and by
    // `ServiceError::PermanentKeyspaceRefusal { refusal, detail }` on the
    // `ProbeSite::Create` arm, twenty-odd lines below (the split's rationale sits
    // between them). The needle is unchanged — the literal is — and the scan now
    // anchors on the `let … = format!(` line, so the window still opens on it.
    // ONE row covers both arms because there is one `format!`; which variant
    // carries it is a runtime `match` on `site`, invisible to any needle.
    //
    // The same row also covers `commit_token_record_wrong_type_refusal`, which
    // spells this literal on purpose: the structural cache's token check
    // returns `update_layer`'s wrong-typed commit-token refusal byte for byte,
    // built from a `TYPE` answer the repair read carried as data — no fred
    // error is wrapped. Counted in `ALLOWLIST_MULTI_SITE_ROWS`.
    ("src/storage/layer.rs", "{} keyspace type error: {}"),
    // The Lua fast path's own arm, which spells the prefix inline because it
    // has only the one site.
    (
        "src/storage/layer.rs",
        "update_layer keyspace type error: {details}",
    ),
    (
        "src/storage/layer.rs",
        "STREAM_ID_COLLISION_UPDATE_PREFIX}: {details}",
    ),
    // b.h4m: `classify_restore_header_read_error`, the restore fence's
    // `HGETALL obj:hdr`. Same category as the `WRONGTYPE_KEY` arm two rows
    // above, and exempt for the same three reasons: it reaches its `format!`
    // holding a REAL `fred::error::Error`, but only the one whose kind is
    // `InvalidArgument` AND whose `details()` is prefix-anchored on `WRONGTYPE`
    // — a Valkey keyspace-integrity verdict, not a transport fault; the wrap
    // re-emits only `e.details()` and never the fred kind; and it is permanent
    // by construction, because no retry can `HGETALL` a key that is not a hash,
    // so a retriable verdict would re-drive a restore that cannot succeed
    // forever. Every OTHER error from those same two round trips falls through
    // this classifier to `map_fred_err_for_grpc` (the `wrongtype` test's else
    // arm), so a Valkey blip on the fence is still a retriable `Unavailable`
    // and this exemption covers exactly the tagged verdict.
    //
    // Spelled as the verbatim source literal per this file's per-call-site
    // convention. ONE row covers both fence reads — the pre-`MULTI` one and the
    // post-conflict re-read — because both go through this single classifier,
    // which is itself the b.h4m fix for the two having disagreed; the `{label}`
    // interpolation is what says which read produced it, and it is a runtime
    // argument rather than source text, so no needle could tell them apart.
    //
    // b.6de sub-task 2 moved this site from `ServiceError::Internal(format!(…))`
    // to `ServiceError::PermanentKeyspaceRefusal { refusal:
    // KeyspaceRefusal::WrongType, detail: format!(…) }` — same gRPC code, same
    // message, permanence now carried by the type instead of by the
    // `WRONGTYPE_KEY` tag the checkpoint lane used to grep for. The scan's second
    // anchor (variant construction + an inline `detail: format!(`) is what keeps
    // the site in view. The needle lost its trailing `object` because the
    // conversion re-wrapped the literal: the source now breaks the line after
    // `for the restore of \`, so `object {object_id}` sits on the next one and no
    // contiguous needle can span the two.
    (
        "src/storage/layer.rs",
        "{KEYSPACE_WRONGTYPE_TAG}: {label} on key '{header_key}' for the restore of",
    ),
    // (No layer.rs snapshot-read answer invariant is left: the ext-GC release
    // read's `TOO_LARGE` refusal went with that read, b.fsz MR-D4. Its
    // replacement, the reference walk in `storage::retirement`, wraps no fred
    // error in an `Internal` of its own; its failures route through
    // `classify_snapshot_read_err`. The structural update's twin is
    // `read_object_snapshot_slim_charged`'s, in the diff_read.rs rows below.)
    //
    // session.rs queue pushes, allowlisted per call site: `create_once`'s two
    // `SADD`s into the `WatchedTxn` pipeline (one `{hex(base_path)}` tag), and
    // `delete` step 3's two `DEL`s and deletion-marker `SET` into a fred
    // `Transaction` (one `{session_id}` tag). Every session.rs round trip — the `WATCH`, the
    // duplicate scan, the `EXEC`s, and the post-commit and `delete` / `list`
    // data-pool commands — routes through a mapper or `undo_create_attempt`
    // and has no row here.
    ("src/storage/session.rs", "SADD path_index queue failed"),
    ("src/storage/session.rs", "SADD path_names queue failed"),
    ("src/storage/session.rs", "DEL session meta queue failed"),
    ("src/storage/session.rs", "DEL layers index queue failed"),
    // `delete` step 3's third push, into the same `Transaction` on the same
    // `{session_id}` tag: the deletion marker's `SET … EX`, queued beside the
    // two `DEL`s so the marker exists from the instant the metadata is gone.
    (
        "src/storage/session.rs",
        "SET session deleted marker queue failed",
    ),
    // ── diff_read.rs: parse pre-condition + integrity violation ──────────
    //
    // External value missing during diff build: ref-count / GC
    // contract violation. The fred-side resolution succeeded; the
    // post-resolution lookup found the meta but not the bytes.
    ("src/storage/diff_read.rs", "external value missing"),
    // `read_object_snapshot_atomic` and `fanout_snapshot_read` (the fan-out
    // behind both the unfenced read and the fenced read's pass) each open by
    // parsing the caller's `object_id` into u64, character for
    // character the same wrap: a caller bug, reached before any Valkey call.
    // The string came from the layer mapping (already corrupt-checked by
    // `get_layer_mapping`) or from internal call sites that should never
    // produce a non-numeric value. One row, two sites: see
    // `ALLOWLIST_MULTI_SITE_ROWS`.
    (
        "src/storage/diff_read.rs",
        "read_object_snapshot called with non-numeric object_id",
    ),
    // `decode_object_snapshot_reply`'s format-tag check and `reply_shape_err`
    // (every shape refusal of the same decode): the `EVALSHA` of
    // `object_snapshot_read.lua` returned `Ok`, but the reply is not the
    // shape this binary's script emits — a stale or foreign script body on
    // the server, the same malformed-reply category as `update_layer`'s
    // unparseable reply head. A retry would read the same script's reply.
    // The `EVALSHA`'s own fred errors route through
    // `classify_object_snapshot_read_err` and are not scanned.
    (
        "src/storage/diff_read.rs",
        "object_snapshot_read.lua answered for object {object_id} in format {format:?}",
    ),
    (
        "src/storage/diff_read.rs",
        "object_snapshot_read.lua reply for object {object_id} is malformed",
    ),
    // `fanout_shape_err`: a fan-out reply (unfenced, or the fenced read's
    // pass) whose shape does not match its command. Unreachable while
    // `fanout_op` picks both the command and the reply variant; the fred round
    // trip succeeded, and its own errors route through
    // `classify_snapshot_read_err`. A logic invariant.
    (
        "src/storage/diff_read.rs",
        "fan-out snapshot read of object {object_id}: the reply to {} ({}) has the wrong shape",
    ),
    // `issue_script_window`: a command no script window carries handed to
    // the structural WATCH read's window script, refused before anything is
    // sent. Unreachable while `FanoutWindows::bounded_window` builds the
    // script windows from round 2's and round 3's commands only; no fred
    // call is wrapped. A logic invariant.
    ("src/storage/diff_read.rs", "({}) cannot be read by a"),
    // `script_window_shape_err` (every shape refusal of
    // `decode_fanout_window_reply`): the `EVALSHA` of
    // `fanout_window_read.lua` returned `Ok`, but the reply is not the shape
    // this binary's script emits, or its follow-up answers contradict the
    // follow-up rule — a stale or foreign script body on the server, the
    // malformed-reply category of the atomic read's rows above. A retry would
    // read the same script's reply. The `EVALSHA`'s own fred errors route
    // through `classify_script_window_err` and are not scanned.
    (
        "src/storage/diff_read.rs",
        "the {FANOUT_WINDOW_READ_OP} reply",
    ),
    // b.26e Stage 2, three answers of the snapshot read its consumer can never
    // receive (the "snapshot-read answer and cut invariants" category). The
    // fred round trips, where there were any, already succeeded, and their
    // errors route through `classify_snapshot_read_err`; these wrap none, and
    // a retry would see the same answer.
    //
    // `read_object_snapshot_slim_charged`: the structural update's charged
    // read is on the unfenced fan-out, which has no ceiling and so never
    // answers `TOO_LARGE`; the arm refuses rather than plan from a guess.
    (
        "src/storage/diff_read.rs",
        "structural snapshot read of object {object_id} answered TOO_LARGE",
    ),
    // `uncharged`: a read that carried no `FanoutAllowance` answered a stop
    // for its upload charge, which only the charged read can answer.
    (
        "src/storage/diff_read.rs",
        "stopped for its upload charge, which only the",
    ),
    // `shape_read_into_full_state`: a value row read by its shape (the
    // bounded-reply fan-out) decoded into the full state. Unreachable while
    // only the structural update's read, whose state is compact, reads rows
    // that way (`ConsumerRules::fanout_replies`).
    (
        "src/storage/diff_read.rs",
        "a value row read by its shape cannot be decoded",
    ),
    // `proven_cut`: a catch-up or per-commit build handed a snapshot read on
    // a path whose cut is not proven (Spec SDD SR-3.1: no snapshot is served
    // unverified). A caller-side invariant over a value already read; no
    // Valkey call is wrapped, and a retry would hand over the same path.
    (
        "src/storage/diff_read.rs",
        "catch-up build for object {object_id} was handed a snapshot read on the {} path",
    ),
    // `decode_field_value`: a stored field or time-sample row whose external
    // `hash` column is not exactly 64 ASCII hex digits (`hash_from_hex`
    // returned `None`). The corrupt-data category: the read that fetched the
    // row — the atomic `EVALSHA`, or a fenced or unfenced fan-out `HGETALL` —
    // already returned `Ok`, and its fred errors
    // route through `classify_snapshot_read_err` /
    // `classify_object_snapshot_read_err`. Every writer stores a 32-byte
    // hash's hex, so only out-of-band corruption reaches this; a retry would
    // read the same column, and decoding it leniently would name the wrong blob.
    (
        "src/storage/diff_read.rs",
        "has a malformed external-value `hash` column",
    ),
    // ── structural_cache_sync.rs: repair-read reply shape ─────────────────
    //
    // `decode_structural_cache_sync_reply`'s format-tag check and
    // `Decoder::malformed` (every shape refusal of the same decode): the
    // `EVALSHA` of `structural_cache_sync.lua` returned `Ok`, but the reply is
    // not the shape this binary's script emits — a stale or foreign script body
    // on the server, the same category as `object_snapshot_read.lua`'s two rows
    // above. A retry would read the same script's reply. The `EVALSHA`'s own
    // fred errors route through `map_transient_fred_err` and are not scanned.
    (
        "src/storage/structural_cache_sync.rs",
        "structural_cache_sync.lua answered for object {object_id} in format {format:?}",
    ),
    (
        "src/storage/structural_cache_sync.rs",
        "structural_cache_sync.lua reply for object {} is malformed",
    ),
    // ── The structural cache's cold start and repair: invariants in hand ───
    //
    // structural_cache_cold_start.rs, `load`: the `structural_cache`
    // consumer's snapshot came back without its structural facts, which
    // `read_object_snapshot` always returns for that consumer — a programmer
    // error over a value already decoded, not a fred error (the read's own
    // errors route through `classify_object_snapshot_read_err`). A retry would
    // run the same code.
    (
        "src/storage/structural_cache_cold_start.rs",
        "the structural_cache snapshot read of object {object_id} carried no structural",
    ),
    // structural_cache_cold_start.rs, `cold_start_for`: a fenced read that
    // verified while its hook holds no cache read permits — the hook takes
    // the pair before every acquire and keeps it until `take_hold`, so this
    // is a programmer error over state in hand, not a fred error (the read's
    // own errors are classified by the fenced read). A retry would run the
    // same code.
    (
        "src/storage/structural_cache_cold_start.rs",
        "the fenced cold-start read of object {object_id} verified without holding",
    ),
    // structural_cache_repair.rs, `contradiction`: a decoded repair-read reply
    // that contradicts itself or the request (no range under a decimal header
    // `seq`, a stop at the bounds before the first entry, a range ending off
    // its goal). The same stale-or-foreign-script category as
    // the `structural_cache_sync.rs` rows above, judged one layer up over values
    // already decoded; the read's fred errors route through
    // `map_transient_fred_err` in `structural_cache_sync.rs`.
    (
        "src/storage/structural_cache_repair.rs",
        "structural cache repair of object {object_id}: {what}",
    ),
    // structural_cache_read.rs, `read_cold`: the probe's decoded reply carries
    // a chunk other than `PROBE`, which the script returns for every probe
    // request — a reply that contradicts its request, the same
    // stale-or-foreign-script category as the `structural_cache_repair.rs` row
    // above, judged over a value already decoded. The probe's fred errors route
    // through `map_transient_fred_err` in `structural_cache_sync.rs`; a retry
    // would read the same script's reply.
    (
        "src/storage/structural_cache_read.rs",
        "structural_cache_sync.lua answered a probe of object {object_id} with a {kind}",
    ),
    // ── subscribe.rs: establishment invariant ─────────────────────────────
    //
    // The catch-up `XRANGE in-window slice` read is NOT allowlisted: it is a
    // single-attempt read on the SubscribeReadLayer path and routes through
    // `map_fred_err_for_grpc`, so a transient blip answers retriable
    // `UNAVAILABLE` and only a permanent one `INTERNAL`. Not being an
    // INTERNAL-mapped wrap, it is not scanned and needs no row.
    //
    // `open_consumer`'s stream registration needs no row. Its old
    // `consume_layer_stream failed` wrap is gone: it maps each `ConsumeError`
    // variant to its own ending instead — `ShuttingDown` to the shared
    // `server_shutting_down()` status, `Saturated` (no free reader in cluster
    // mode) to a retryable `UNAVAILABLE` under `reader_capacity`, both built
    // without a `ServiceError`, and the unreachable `InvalidCursor` to
    // `ServiceError::Internal(err.to_string())`. That last one carries no
    // `format!` literal and wraps no fred error, so it is not an
    // INTERNAL-mapped wrap in this scan's sense — see
    // `is_internal_mapped_wrap_anchor`, which anchors on
    // `ServiceError::Internal(format!` and on the `PermanentKeyspaceRefusal`
    // ctor, and on nothing else — and the site is not scanned. A row keyed on
    // the old wrap's message would be worse than useless: it could never
    // match, so the staleness guard below would fail on it immediately.
    //
    // Brand-new / empty-session establishment invariant (SR-2.4): the
    // from-empty full-state builder (`compute_faithful_full_content_at_cut` /
    // `plan_full_content`) always serializes a valid non-empty create,
    // even for an empty net-live scene. An empty OMPF payload on this
    // path is therefore a structured logic-invariant violation, not a
    // fred error — there is no Valkey/fred failure to map. Surfacing as
    // Internal prevents silently dropping establishment message #1 and
    // collapsing the stream to the fatal terminator-only sequence.
    (
        "src/grpc/subscribe.rs",
        "brand-new full-state establishment produced an empty OMPF payload",
    ),
    // `deliver_catch_up_content`: a catch-up payload was built at a cut that
    // is not proven (anything but the `DiffCut::Atomic` every catch-up build
    // from an atomic or a shared fenced snapshot carries), so the payload
    // cannot say which version it serves. A logic invariant on a value the
    // catch-up already holds, not a fred error; refusing keeps an unprovable
    // version out of the content header and the caught-up ack.
    (
        "src/grpc/subscribe.rs",
        "catch-up payload for object_id={object_id} built at {cut:?} was delivered",
    ),
    // ── The five files the runtime walk added to SOURCES (b.6de review item
    //    4) ─────────────────────────────────────────────────────────────────
    //
    // Each of these held a scanned wrap while sitting outside the
    // hand-maintained SOURCES list, i.e. nothing was scanning them at all. They
    // are audited below one site at a time, on the same per-call-site terms as
    // every row above.
    //
    // tail_slice.rs: a pre-read arithmetic invariant, not a fred error — the
    // base snapshot's sequence leaves no window below the ceiling both commit
    // paths enforce. No Valkey call has been made at this point, so there is
    // nothing to classify transient.
    (
        "src/storage/tail_slice.rs",
        "tail slice read for object {object_id}: base_snap_seq",
    ),
    // session_cache.rs: the non-retriable half of the b.33e session-recovery
    // classification. The retriability DECISION is made one line above by
    // `ServiceError::recovery_error_is_retriable`, whose transient arm returns
    // `Unavailable` — so this `Internal` is the *already-classified* permanent
    // verdict, not an unclassified wrap.
    ("src/grpc/session_cache.rs", "session-recovery failed:"),
    // checkpoint/session.rs: two genuine fred-error wraps — but both are
    // immediately handed to `service_err_to_checkpoint_err`, which maps
    // `ServiceError::Internal` to `CheckpointError::Transient` (and
    // `debug_assert!`s that `Internal` is the only variant it is ever given).
    // The `Internal` is a private currency inside that module, never reaching a
    // gRPC surface, and the retriable verdict the mappers exist to produce is
    // what the caller gets. A conversion to `map_transient_fred_err` here would
    // trip that `debug_assert!`.
    (
        "src/checkpoint/session.rs",
        "SISMEMBER path_index failed for",
    ),
    (
        "src/checkpoint/session.rs",
        "EVALSHA session_checkpoint_read failed for",
    ),
    // checkpoint/recovery.rs: both sites are the terminal `Err(other)` arm of a
    // match that has already peeled `CheckpointError::Transient` off into
    // `ServiceError::Aborted` (the in-band retriable signal
    // `retry_aborted_with_jitter` converges on). What reaches these lines is a
    // `CheckpointError` kind that is terminal by construction, not a fred
    // error.
    (
        "src/checkpoint/recovery.rs",
        "restore identity gate for object",
    ),
    ("src/checkpoint/recovery.rs", "injected fault at seam"),
    // checkpoint/apply.rs: inside the crate's own `#[cfg(test)]` module — the
    // wrap IS the fixture, constructing the input
    // `write_object_err_to_checkpoint_err` is asserted on. `include_str!` bakes
    // the whole file, test module included, so the scan sees it.
    (
        "src/checkpoint/apply.rs",
        "HSET section failed while queueing (unrelated to",
    ),
    // ── The structural commit (plan b.dkt, Epic 5) ──────────────────────────
    //
    // lua.rs, `encode_mutations` and its digest helpers: a plan paired with
    // pre-image digests that were not built from it — a fingerprint mismatch,
    // or (defensively) a digest of the wrong kind for its record. A caller-side
    // invariant over values already in hand: no Valkey call has been made, and
    // a retry would hand over the same pair. Refusing is the point — sending
    // the pair would check the wrong keys' beliefs. One row per site.
    (
        "src/storage/lua.rs",
        "structural encode: the pre-image digests were not built from this",
    ),
    ("src/storage/lua.rs", "carries no digest, got {other:?}"),
    (
        "src/storage/lua.rs",
        "structural encode: record {index} expects a {expected} digest",
    ),
    // structural.rs, `commit_structural`'s refusal decode: the `EVALSHA`
    // returned the script's `STALE` reply, but without the decimal
    // the script always writes after it — a stale or foreign script body on
    // the server, the same malformed-reply category as `update_layer`'s
    // unparseable reply head. The script's own fred errors route through
    // `LayerStore::map_update_script_error`; this wraps none.
    (
        "src/storage/structural.rs",
        "update_layer structural commit: unparseable refusal reply",
    ),
    // ── external_values.rs: the retirement's attempt-mode enqueue verdict ──
    //
    // `enqueue_ext_release_attempt`'s verdict decode (b.yuo Part R): the attempt
    // mode of `ext_release_enqueue.lua` returned `Ok`, but its verdict is none
    // of the three the script in this repo answers (`enqueued`, `already`,
    // `void`) — a stale or foreign script body behind the SHA, the same
    // malformed-reply category as `update_layer`'s unparseable reply head. A
    // protocol / integrity violation, not a fred error: the `EVALSHA`'s own
    // fred errors (and the `NOSCRIPT` reload's) route through
    // `map_transient_fred_err` and are not scanned, and a retry would read the
    // same script's reply.
    (
        "src/storage/external_values.rs",
        "ext_release_enqueue answered an unknown attempt verdict",
    ),
    // test_support/structural_commit.rs, the test-support planner: the atomic
    // snapshot read answered `TOO_LARGE`, and the planner has no other read.
    // An answer, not a fred error — the read's own errors route through
    // `classify_object_snapshot_read_err`, and the planner's header reads
    // through `map_snapshot_load_err`.
    (
        "src/storage/test_support/structural_commit.rs",
        "structural planner: object {oid_str} is above the atomic read's limits",
    ),
];

/// ALLOWLIST rows that genuinely cover more than one wrap site, with the exact
/// number each covers. Every row not listed here must match exactly one site
/// (`workspace_wide_allowlist_rows_each_excuse_exactly_their_declared_sites`).
/// A row belongs here only when one needle cannot tell its sites apart AND
/// all of them are exempt for the same reason; each entry states that reason.
const ALLOWLIST_MULTI_SITE_ROWS: &[(&str, &str, usize)] = &[
    // `map_snapshot_load_err` and `map_fred_err_for_grpc` spell their permanent
    // branch identically, and both are the canonical mapper exemption.
    ("src/storage/client.rs", "{op} failed: {e}", 2),
    // `write_object` and `write_restored_object` open with the same `oid_str`
    // parse, character for character, for the same programmer-error reason.
    ("src/storage/layer.rs", "corrupt oid_str", 2),
    // The txn probe's keyspace refusal and `commit_token_record_wrong_type_refusal`
    // spell `update_layer`'s wrong-type refusal identically by design: the
    // structural cache's token check must return it byte for byte.
    ("src/storage/layer.rs", "{} keyspace type error: {}", 2),
    // The atomic snapshot read and the fan-out open with the same `object_id`
    // parse, character for character, for the same programmer-error reason.
    (
        "src/storage/diff_read.rs",
        "read_object_snapshot called with non-numeric object_id",
        2,
    ),
];

/// How many lines after the `ServiceError::Internal(format!` token to
/// include when checking the allowlist. Many wraps in this codebase
/// span multiple lines (e.g.
/// ```ignore
///     ServiceError::Internal(format!(
///         "EXEC session DEL meta+layers failed: {e}"
///     ))
/// ```
/// ), so the `format!(` token line carries no message content. The
/// allowlist matches against the verbatim text of the format string,
/// which lives on a following line; this window includes them.
///
/// Picked empirically: every multi-line wrap in src/ today fits inside
/// 4 trailing lines. Bump if a wider wrap appears.
const ALLOWLIST_LOOKAHEAD_LINES: usize = 4;

// ── The scanned signature set (b.6de sub-task 2 widened it) ──────────────────
//
// This guard hunts a `format!` whose product becomes a gRPC-`INTERNAL` -mapped
// `ServiceError` — that is the shape a fred error hides inside when it should
// have routed through `map_transient_fred_err` / `map_fred_err_for_grpc`. Until
// b.6de there was exactly one spelling of that, `ServiceError::Internal(format!`,
// and the scan hunted the literal token.
//
// b.6de added a second INTERNAL-mapped variant,
// `ServiceError::PermanentKeyspaceRefusal { refusal, detail }` (same gRPC code,
// same redacted wire message, same single `error!` at the `Status` conversion —
// see its rustdoc in `src/error.rs`), and in converting two sites to it moved
// them off the one literal in two different ways:
//
//   * `classify_restore_header_read_error` builds the message inline, as the
//     variant's `detail: format!(…)` field;
//   * `classify_keyspace_probe_error` HOISTS one `format!` into `let detail =
//     …` and hands it to `ServiceError::Internal(detail)` on the update arm and
//     to the variant on the create arm.
//
// Both remained on the ALLOWLIST and both stopped being scanned — so a live
// `fred::error::Error` wrapped into an INTERNAL-mapped error silently left the
// guard's sight, and the two now-unreachable rows would have failed
// `workspace_wide_allowlist_rows_each_excuse_exactly_their_declared_sites`
// (whose at-least-one predecessor is what surfaced this). Hence three
// recognised anchors, below.

/// Anchor 1 — the original: the `format!` written inline into
/// [`ServiceError::Internal`].
const WRAP_TOKEN_INTERNAL_FORMAT: &str = "ServiceError::Internal(format!";

/// Anchor 2, part one — a construction of the typed refusal variant. Matched
/// only in combination with [`WRAP_TOKEN_REFUSAL_INLINE_DETAIL`] inside the
/// lookahead window: the token alone also matches the variant's *patterns*
/// (`clone_service_error`'s match arm) and its test constructions from string
/// literals, none of which wrap a `format!` at all.
const WRAP_TOKEN_REFUSAL_CTOR: &str = "ServiceError::PermanentKeyspaceRefusal {";

/// Anchor 2, part two — the variant's one `String` field, built inline. This is
/// the field carrying the operator-facing message, i.e. the exact text an
/// ALLOWLIST row quotes.
const WRAP_TOKEN_REFUSAL_INLINE_DETAIL: &str = "detail: format!(";

/// `ServiceError::Internal(` — the prefix [`internal_wrap_consumed_ident`]
/// splits on. Deliberately NOT the `(format!` token: the hoisted shape is
/// precisely the one where `format!` is absent at the construction.
const INTERNAL_CTOR_PREFIX: &str = "ServiceError::Internal(";

/// The ways a hoisted `String` binding reaches the construction that wraps it,
/// beyond the bare move.
///
/// All four are the same hand-off with a copy in the middle, and they are what
/// a refactor reaches for the moment the binding is used twice — which is
/// exactly the edit that would otherwise slip a wrap out of the scan: rewriting
/// the two `classify_keyspace_probe_error` consumption sites as
/// `ServiceError::Internal(detail.clone())` / `detail: detail.clone()` takes the
/// `let detail = format!(` above them out of anchor position and removes the
/// site from the guard's sight entirely.
const CONSUMING_SUFFIXES: &[&str] = &[".clone()", ".to_owned()", ".into()", ".to_string()"];

/// The raw lookahead window: the anchor line plus up to
/// [`ALLOWLIST_LOOKAHEAD_LINES`] following ones. The anchor predicate reads it
/// directly; ALLOWLIST rows are matched against [`site_text`], this window cut
/// at the next anchor, which the scan and the allowlist fence share so the two
/// can never disagree about what a row is matched against.
fn wrap_window<'a, 'b>(lines: &'b [&'a str], idx: usize) -> &'b [&'a str] {
    let end = (idx + 1 + ALLOWLIST_LOOKAHEAD_LINES).min(lines.len());
    &lines[idx..end]
}

/// The leading identifier of `s`, if `s` starts with one.
fn leading_ident(s: &str) -> Option<(&str, &str)> {
    let len = s
        .bytes()
        .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
        .count();
    if len == 0 {
        return None;
    }
    Some(s.split_at(len))
}

/// The identifier a *consuming* expression hands on, paired with whatever
/// follows the expression.
///
/// Recognises the bare move `<ident>` and the four [`CONSUMING_SUFFIXES`]
/// spellings `<ident>.clone()` / `.to_owned()` / `.into()` / `.to_string()`, and
/// nothing else.
/// Two exclusions fall out by construction rather than from a denylist: a
/// literal (`"lit".into()`) has no leading identifier at all, and an arbitrary
/// method call (`e.details()`) keeps its `.details()` in the returned tail, so
/// the caller's terminator check rejects it.
fn consuming_expr_ident(expr: &str) -> Option<(&str, &str)> {
    let (ident, tail) = leading_ident(expr)?;
    if ident == "_" || !ident.starts_with(|c: char| c.is_ascii_lowercase() || c == '_') {
        return None;
    }
    for suffix in CONSUMING_SUFFIXES {
        if let Some(rest) = tail.strip_prefix(suffix) {
            return Some((ident, rest));
        }
    }
    Some((ident, tail))
}

/// The identifier in `ServiceError::Internal(<ident>)` when that spelling is a
/// *construction* rather than a pattern.
///
/// Pattern spellings — `ServiceError::Internal(msg) => …` and
/// `let ServiceError::Internal(msg) = &err else` — are excluded by looking at
/// what follows the closing paren (`=>` / `=`); `Internal(_)` and string
/// literals are excluded by [`consuming_expr_ident`] requiring a leading
/// identifier. Only the first occurrence on a line is considered, which is
/// exactly right for `Internal(msg) => Internal(msg.clone())`: the leading
/// pattern disqualifies the line.
fn internal_wrap_consumed_ident(line: &str) -> Option<&str> {
    let rest = line.split(INTERNAL_CTOR_PREFIX).nth(1)?;
    let (ident, tail) = consuming_expr_ident(rest)?;
    let tail = tail.strip_prefix(')')?.trim_start();
    if tail.starts_with("=>") || tail.starts_with('=') {
        return None;
    }
    Some(ident)
}

/// Does the `ServiceError::PermanentKeyspaceRefusal { … }` construction opening
/// at `lines[idx]` hand `ident` to its `detail` field?
///
/// Both spellings count: the long form `detail: <ident>,` (with or without one
/// of the [`CONSUMING_SUFFIXES`]) and the field shorthand `detail,`, which can
/// only ever consume a binding literally named `detail`.
fn refusal_ctor_consumes(lines: &[&str], idx: usize, ident: &str) -> bool {
    wrap_window(lines, idx).iter().any(|field_line| {
        let t = field_line.trim();
        if t == "detail," || t == "detail" {
            return ident == "detail";
        }
        let Some(rest) = t.strip_prefix("detail: ") else {
            return false;
        };
        let Some((found, tail)) = consuming_expr_ident(rest) else {
            return false;
        };
        found == ident && matches!(tail.trim_end(), "" | ",")
    })
}

/// `let [mut] <ident> = format!(` — the hoisted-wrap anchor, as of `lines[idx]`.
///
/// Takes the whole slice rather than one line because rustfmt breaks a long
/// binding after the `=`, leaving `format!(` alone on the next line; a
/// single-line check reads that shape as "not a `format!` binding" and drops the
/// wrap out of the scan.
fn hoisted_format_binding<'a>(lines: &[&'a str], idx: usize) -> Option<&'a str> {
    let trimmed = lines[idx].trim();
    let rest = trimmed.strip_prefix("let ")?;
    let rest = rest.strip_prefix("mut ").unwrap_or(rest);
    let (ident, tail) = leading_ident(rest)?;
    let tail = tail.trim_start().strip_prefix('=')?.trim_start();
    if tail.starts_with("format!(") {
        return Some(ident);
    }
    if tail.is_empty()
        && lines
            .get(idx + 1)
            .is_some_and(|next| next.trim_start().starts_with("format!("))
    {
        return Some(ident);
    }
    None
}

/// Does `line` open a `fn` item — at any visibility, and through any
/// combination of the qualifier keywords that may precede `fn`?
///
/// Used to bound the hoisted-binding search to one function body. Comment lines
/// (`// pub fn foo`) and doc lines fail the final `starts_with("fn ")` because
/// their leading `//` is not a qualifier, so they are excluded without a
/// special case.
fn is_fn_signature_line(line: &str) -> bool {
    const QUALIFIERS: &[&str] = &[
        "pub(crate) ",
        "pub(super) ",
        "pub(in crate) ",
        "pub ",
        "async ",
        "const ",
        "unsafe ",
        "extern \"C\" ",
        "extern ",
        "default ",
    ];
    let mut rest = line.trim_start();
    while let Some(stripped) = QUALIFIERS.iter().find_map(|q| rest.strip_prefix(q)) {
        rest = stripped.trim_start();
    }
    rest.starts_with("fn ")
}

/// The index of the line closing the body opened by the `fn` signature at
/// `start_idx` — the same brace walk [`extract_function_body`] does, by line
/// index rather than byte offset, and with `//` comments stripped for the same
/// reason.
fn fn_body_end_line(lines: &[&str], start_idx: usize) -> Option<usize> {
    let mut depth: i32 = 0;
    let mut seen_open = false;
    for (offset, line) in lines.iter().enumerate().skip(start_idx) {
        for ch in strip_line_comment(line).chars() {
            match ch {
                '{' => {
                    depth += 1;
                    seen_open = true;
                }
                '}' => depth -= 1,
                _ => {}
            }
        }
        if seen_open && depth == 0 {
            return Some(offset);
        }
    }
    None
}

/// The last line of the `fn` body containing `idx`, if `idx` is inside one.
fn enclosing_fn_end_line(lines: &[&str], idx: usize) -> Option<usize> {
    let start = (0..=idx).rev().find(|&i| is_fn_signature_line(lines[i]))?;
    let end = fn_body_end_line(lines, start)?;
    (end >= idx).then_some(end)
}

/// Is the `let <ident> = format!(…)` at `let_idx` consumed by an
/// INTERNAL-mapped construction *below it, inside the same function*?
///
/// Function-scoped rather than file-scoped deliberately. `detail` and `msg` are
/// generic binding names: matched across a 10k-line file, an unrelated
/// `let detail = format!("unrelated diagnostic …")` anywhere in `layer.rs` is
/// implicated by the `ServiceError::Internal(detail)` in
/// `classify_keyspace_probe_error` and the guard fails on a diff that did
/// nothing wrong — verified, not hypothetical. Below-the-`let` is the other half
/// of the same narrowing: a binding cannot be consumed by a construction that
/// precedes it.
///
/// Falls back to the rest of the slice when no enclosing `fn` can be identified.
/// That is the permissive-for-the-*file*, strict-for-the-*guard* direction: a
/// wider search can only make the anchor fire, and an anchor that fires is a
/// site the ALLOWLIST must account for.
fn hoisted_binding_is_consumed_below(lines: &[&str], let_idx: usize, ident: &str) -> bool {
    let end = enclosing_fn_end_line(lines, let_idx).unwrap_or(lines.len().saturating_sub(1));
    lines
        .iter()
        .enumerate()
        .take(end + 1)
        .skip(let_idx + 1)
        .any(|(idx, line)| {
            if line.trim_start().starts_with("//") {
                return false;
            }
            internal_wrap_consumed_ident(line) == Some(ident)
                || (line.contains(WRAP_TOKEN_REFUSAL_CTOR)
                    && refusal_ctor_consumes(lines, idx, ident))
        })
}

/// Is `lines[idx]` the anchor line of a `format!` that becomes an
/// INTERNAL-mapped `ServiceError`?
///
/// The anchor is chosen so the ALLOWLIST keeps its per-call-site convention:
/// the format string's verbatim text always falls inside
/// `[idx, idx + ALLOWLIST_LOOKAHEAD_LINES]`, whichever of the three spellings
/// matched. For the hoisted shape that means anchoring on the `let … = format!(`
/// line rather than on the construction that consumes it — the construction can
/// sit an arbitrary distance below (20 lines of rationale comment, at the
/// keyspace-probe site), and a lookbehind wide enough to reach back over it
/// would start excusing unrelated wraps.
///
/// Specified in both directions by
/// [`internal_mapped_wrap_anchor_scan_flags_every_wrap_spelling_and_no_ordinary_line`].
fn is_internal_mapped_wrap_anchor(lines: &[&str], idx: usize) -> bool {
    let trimmed = lines[idx].trim();
    // Skip comments / doc strings.
    if trimmed.starts_with("//") {
        return false;
    }
    if trimmed.contains(WRAP_TOKEN_INTERNAL_FORMAT) {
        return true;
    }
    if trimmed.contains(WRAP_TOKEN_REFUSAL_CTOR)
        && wrap_window(lines, idx)
            .iter()
            .any(|l| l.contains(WRAP_TOKEN_REFUSAL_INLINE_DETAIL))
    {
        return true;
    }
    match hoisted_format_binding(lines, idx) {
        Some(ident) => hoisted_binding_is_consumed_below(lines, idx, ident),
        None => false,
    }
}

/// The text an ALLOWLIST needle is matched against for the wrap anchored at
/// `lines[idx]`: its [`wrap_window`], cut short before the next anchor line.
///
/// Without the cut, two wraps written within [`ALLOWLIST_LOOKAHEAD_LINES`] of
/// each other share text — `create_once`'s two queue-time `SADD` wraps sit four
/// lines apart — so the first wrap's window also holds the second wrap's
/// message, and the second wrap's row excuses the first as well. Deleting the
/// first wrap's own row would then change nothing, and the row-count fence in
/// [`workspace_wide_allowlist_rows_each_excuse_exactly_their_declared_sites`]
/// would read that one row as covering two sites.
fn site_text(lines: &[&str], idx: usize) -> String {
    let window = wrap_window(lines, idx);
    let len = (1..window.len())
        .find(|offset| is_internal_mapped_wrap_anchor(lines, idx + offset))
        .unwrap_or(window.len());
    window[..len].join("\n")
}

/// Scan one entry's source for the regression pattern, applying the
/// allowlist. Returns offending `(absolute_lineno, line_text)` pairs.
///
/// The scanner is multi-line aware (b.u96 item 5): the offending
/// `ServiceError::Internal(format!` token frequently lives on a line
/// that carries no message text, with the format-string literal on the
/// next 1–3 lines. The allowlist match therefore checks both the
/// offending line itself AND the next [`ALLOWLIST_LOOKAHEAD_LINES`]
/// lines for the needle — up to, not including, the next wrap's anchor
/// ([`site_text`]).
fn find_unmapped_internal_wraps(entry: &Entry) -> Vec<(usize, String)> {
    let mut hits = Vec::new();
    let lines: Vec<&str> = entry.source.lines().collect();
    for (idx, line) in lines.iter().enumerate() {
        // Hunt the regression signature — all three INTERNAL-mapped wrap
        // spellings (comment lines are skipped inside the predicate).
        if !is_internal_mapped_wrap_anchor(&lines, idx) {
            continue;
        }
        // Build a multi-line window: this line plus up to N following, cut
        // short at the next wrap. The allowlist needle may appear on the line
        // that opens the `format!(` token or on one of the following lines
        // that carries the format-string literal.
        let window = site_text(&lines, idx);
        let excused = ALLOWLIST
            .iter()
            .any(|(file, needle)| *file == entry.file && window.contains(needle));
        if excused {
            continue;
        }
        hits.push((idx + 1, line.to_string()));
    }
    hits
}

/// Verify every source file the test plans to scan actually exists at
/// the expected path. A future refactor that moves or renames a file
/// would change the `include_str!` target and would compile-fail at
/// that line, but the on-disk check here catches a different category
/// of mistake: a typo in `SOURCES` that *coincidentally* still resolves
/// (e.g. the bytes-on-disk version is stale because the build cached a
/// previous file). Belt-and-suspenders.
fn assert_sources_exist_on_disk() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for entry in SOURCES {
        let abs = manifest_dir.join(entry.file);
        assert!(
            abs.exists(),
            "scan source `{}` (resolved to `{}`) is missing from disk — \
             a SOURCES entry was added without the corresponding file, or the file \
             was renamed/deleted without updating SOURCES",
            entry.file,
            abs.display()
        );
    }
}

/// Every `.rs` file under `src/`, depth-first, sorted for a deterministic
/// failure message. `std::fs` only — the walk is a dozen lines and a new
/// dev-dependency for it would be a worse trade.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let read = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read directory `{}`: {e}", dir.display()));
    let mut entries: Vec<PathBuf> = read
        .map(|e| {
            e.unwrap_or_else(|err| panic!("cannot read an entry of `{}`: {err}", dir.display()))
                .path()
        })
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// [`SOURCES`] must name every `src/**/*.rs` file that holds a scanned wrap
/// anchor.
///
/// # The gap this closes
///
/// `SOURCES` is hand-maintained, and everything else in this file scans only
/// what it names — so a file absent from the list is not "partially covered",
/// it is **invisible**, and its absence looks identical to a clean file. The
/// regeneration recipe was a single `rg` for `ServiceError::Internal(format!`,
/// which is one of three spellings since b.6de and was never re-run against the
/// other two; the list had in fact drifted five files behind that single
/// literal alone.
///
/// This walk is the same predicate the scan uses, applied to the real
/// directory tree at runtime, so the list cannot silently under-cover again:
/// a new file with a wrap fails HERE (naming itself) rather than slipping
/// through unscanned.
///
/// # Runtime walk, not `include_str!`
///
/// `include_str!` needs a literal path, so a compile-time version of this check
/// is impossible by construction — the set of files is exactly what is unknown.
/// Reading them at test time costs a few ms and is the only way to ask the
/// question.
///
/// The fix for a failure is to ADD the named file to `SOURCES` (and then audit
/// whatever the scan finds in it), never to narrow this walk.
#[test]
fn sources_covers_every_file_holding_a_scanned_anchor() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_root = manifest_dir.join("src");
    let mut files = Vec::new();
    collect_rs_files(&src_root, &mut files);
    assert!(
        files.len() > 50,
        "the walk found only {} `.rs` files under `{}` — the recursion or the root is wrong, and \
         a walk that sees nothing cannot report a missing file",
        files.len(),
        src_root.display()
    );

    let mut anchored: Vec<String> = Vec::new();
    for path in &files {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read `{}`: {e}", path.display()));
        let lines: Vec<&str> = text.lines().collect();
        if !(0..lines.len()).any(|idx| is_internal_mapped_wrap_anchor(&lines, idx)) {
            continue;
        }
        let rel = path
            .strip_prefix(&manifest_dir)
            .unwrap_or_else(|e| panic!("`{}` is not under the manifest dir: {e}", path.display()))
            .to_string_lossy()
            .replace('\\', "/");
        anchored.push(rel);
    }

    assert!(
        !anchored.is_empty(),
        "the walk found no wrap anchor anywhere in `{}`. The tree has many — so the predicate \
         matched nothing, and every scan in this file is passing vacuously",
        src_root.display()
    );

    let missing: Vec<&String> = anchored
        .iter()
        .filter(|rel| !SOURCES.iter().any(|entry| entry.file == rel.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "these `src/**/*.rs` files hold a scanned INTERNAL-mapped `format!` wrap but are missing \
         from SOURCES, so NOTHING scans them — their wraps are invisible to \
         `workspace_wide_fred_error_wraps_route_through_surface_mappers`, allowlist rows and all: \
         {missing:?}\n\nAdd an `Entry` for each (with the matching `include_str!`), then audit \
         whatever the scan reports in it — one allowlist row per site with a rationale, or a \
         conversion to `map_transient_fred_err` / `map_fred_err_for_grpc`. {} of {} walked files \
         hold an anchor.",
        anchored.len(),
        files.len()
    );
}

#[test]
fn workspace_wide_fred_error_wraps_route_through_surface_mappers() {
    assert_sources_exist_on_disk();

    let mut violations: Vec<(String, usize, String)> = Vec::new();
    for entry in SOURCES {
        for (lineno, text) in find_unmapped_internal_wraps(entry) {
            violations.push((entry.file.to_string(), lineno, text));
        }
    }

    if !violations.is_empty() {
        let mut report = String::new();
        report.push_str(
            "Found `format!` wraps into an INTERNAL-mapped `ServiceError` in `src/**/*.rs` \
             that are not on the b.u96 allowlist. Three spellings are scanned: \
             `ServiceError::Internal(format!`, a `ServiceError::PermanentKeyspaceRefusal { .. }` \
             built with an inline `detail: format!(`, and a `let <ident> = format!(` hoisted \
             into either of those:\n\n",
        );
        report
            .push_str("Every direct-Internal wrap of a fred error should route through one of:\n");
        report
            .push_str("  * `map_transient_fred_err`  — inside `retry_aborted_with_jitter` loops\n");
        report.push_str("    (transient → Aborted; helper picks it up and retries)\n");
        report.push_str(
            "  * `map_fred_err_for_grpc`   — for single-shot calls surfacing to a gRPC handler\n",
        );
        report.push_str("    (transient → Unavailable; retriable per the gRPC retry contract)\n\n");
        report.push_str(
            "If the wrap is legitimate (structured non-fred error, MULTI/EXEC queue-time, \
             corrupt-data parse, integrity violation, out-of-window fallback, etc.), add a \
             (file, distinctive-substring) row to the `ALLOWLIST` in this test with a \
             rationale comment explaining the category. Otherwise replace the wrap with the \
             appropriate mapper.\n\nViolations:\n",
        );
        for (file, lineno, text) in &violations {
            report.push_str(&format!("  {file}:{lineno}\n    {text}\n"));
        }
        panic!("{report}");
    }
}

/// Every scanned wrap site in `lines`: its 0-based anchor line index and the
/// text ALLOWLIST needles are matched against ([`site_text`]).
fn wrap_sites(lines: &[&str]) -> Vec<(usize, String)> {
    (0..lines.len())
        .filter(|&idx| is_internal_mapped_wrap_anchor(lines, idx))
        .map(|idx| (idx, site_text(lines, idx)))
        .collect()
}

/// The ALLOWLIST as exact claims: each row excuses exactly the sites it names.
///
/// Four checks, all sharing the anchor predicate and [`site_text`] with
/// [`find_unmapped_internal_wraps`], so a row can never be counted differently
/// here from how it excuses there:
///
/// 1. **Each row matches exactly one site**, or exactly the count declared for
///    it in [`ALLOWLIST_MULTI_SITE_ROWS`]. Zero is the old staleness check — a
///    row surviving a refactor that removed its site would silently excuse a
///    future violation that happens to reuse the old wording. More than
///    declared is the hazard b.62m found: `"HGETALL failed"` excused three
///    round trips and `"SMEMBERS names failed"` two, and a fence asking only
///    "does the row match something?" passes a row that broad.
/// 2. **No site is matched by two rows.** Otherwise either row can be deleted
///    without the scan noticing, and the one that stays is broader than the
///    site it was written for.
/// 3. **No needle is a substring of another row's needle in the same file.**
///    `"GET ext_data failed"` was a substring of `"queue GET ext_data
///    failed"`: the short row matched the queue site, so check 1 read it as
///    live even after its own round trip had been converted. Scoped per file
///    because matching is per file — a needle can only ever excuse a site in
///    its own file, so a cross-file substring (`layer.rs`'s `"HSET section
///    failed"` inside `checkpoint/apply.rs`'s fixture message) excuses
///    nothing extra.
/// 4. **Every declared multi-site count names a real row, and is at least
///    two** — a declaration that outlives its row, or declares one, is noise
///    that hides the next real one.
#[test]
fn workspace_wide_allowlist_rows_each_excuse_exactly_their_declared_sites() {
    let mut problems: Vec<String> = Vec::new();

    for (file, needle, count) in ALLOWLIST_MULTI_SITE_ROWS {
        if !ALLOWLIST.contains(&(*file, *needle)) {
            problems.push(format!(
                "ALLOWLIST_MULTI_SITE_ROWS declares {count} site(s) for {file}: {needle:?}, \
                 which is not an ALLOWLIST row — remove the declaration"
            ));
        }
        if *count < 2 {
            problems.push(format!(
                "ALLOWLIST_MULTI_SITE_ROWS declares {count} site(s) for {file}: {needle:?}; \
                 one site is the default, so declare only rows that cover two or more"
            ));
        }
    }

    // (file, 1-based line number, site text) for every scanned wrap site.
    let mut sites: Vec<(&str, usize, String)> = Vec::new();
    for (file, _) in ALLOWLIST {
        if !SOURCES.iter().any(|e| e.file == *file) {
            panic!(
                "ALLOWLIST row references unknown file `{file}` — every allowlist row \
                 must point at a file that appears in SOURCES (so the test actually \
                 scans the file the row is excusing). Add the file to SOURCES or fix \
                 the typo in the ALLOWLIST row."
            );
        }
    }
    for entry in SOURCES {
        let lines: Vec<&str> = entry.source.lines().collect();
        for (idx, text) in wrap_sites(&lines) {
            sites.push((entry.file, idx + 1, text));
        }
    }

    // Check 1.
    for (file, needle) in ALLOWLIST {
        let matched: Vec<usize> = sites
            .iter()
            .filter(|(f, _, text)| f == file && text.contains(needle))
            .map(|(_, lineno, _)| *lineno)
            .collect();
        let declared = ALLOWLIST_MULTI_SITE_ROWS
            .iter()
            .find(|(f, n, _)| f == file && n == needle)
            .map_or(1, |(_, _, count)| *count);
        if matched.is_empty() {
            problems.push(format!(
                "{file}: needle = {needle:?} matches no INTERNAL-mapped `format!` wrap site. \
                 The underlying site was probably refactored (renamed, deleted, or its message \
                 changed) without updating the allowlist; remove the stale row or update its \
                 needle. A stale row silently excuses a future violation that happens to reuse \
                 the old wording."
            ));
        } else if matched.len() != declared {
            problems.push(format!(
                "{file}: needle = {needle:?} matches {} wrap sites (lines {matched:?}) but is \
                 declared to cover {declared}. A row must name exactly the sites it excuses: \
                 narrow the needle to one site's own wording and give each other site its own \
                 row with its own rationale — or, if one row genuinely covers several sites for \
                 one reason, declare the count in ALLOWLIST_MULTI_SITE_ROWS with that reason.",
                matched.len()
            ));
        }
    }

    // Check 2.
    for (file, lineno, text) in &sites {
        let rows: Vec<&str> = ALLOWLIST
            .iter()
            .filter(|(f, needle)| f == file && text.contains(needle))
            .map(|(_, needle)| *needle)
            .collect();
        if rows.len() > 1 {
            problems.push(format!(
                "{file}:{lineno} is excused by {} rows ({rows:?}); each site must be excused by \
                 exactly one, or deleting either row goes unnoticed",
                rows.len()
            ));
        }
    }

    // Check 3.
    for (i, (file_a, needle_a)) in ALLOWLIST.iter().enumerate() {
        for (j, (file_b, needle_b)) in ALLOWLIST.iter().enumerate() {
            if i != j && file_a == file_b && needle_b.contains(needle_a) {
                problems.push(format!(
                    "{file_a}: needle {needle_a:?} is a substring of the row {needle_b:?} (or a \
                     duplicate of it), so it matches every site the longer row does — lengthen \
                     the short needle until it quotes its own site's wording and no other row's"
                ));
            }
        }
    }

    assert!(
        problems.is_empty(),
        "ALLOWLIST rows must each excuse exactly the sites they name (see \
         `is_internal_mapped_wrap_anchor` for the three scanned spellings):\n\n  {}",
        problems.join("\n  ")
    );
}

/// b.u96 item 5 — pin the historical scope. The original b.fd9-named
/// functions (`list_layer_names`, `get_layer_mapping`,
/// `read_object_header`, `process_pending`) must NOT contain any
/// `ServiceError::Internal(format!` wrap of a fred error — they were
/// the canonical b.fd9 acceptance scope and the workspace-wide scan
/// above generalises the same contract. b.62m added the `SessionStore`
/// functions whose round trips it moved off the allowlist (`list`,
/// `get_session_meta`, `finish_create`, `reconcile_in_doubt_create`) and
/// the `undo_create_attempt` helper they now fail through. This narrower per-function
/// scan catches a regression that re-introduces a wrap inside one of
/// the originally-scoped functions but happens to type a message
/// string that escapes the workspace-wide allowlist via a coincidental
/// substring match.
///
/// A failure here PLUS a passing workspace-wide scan would mean the
/// allowlist has a needle that's too broad (matching both the
/// legitimate and the new site). A failure here ALONE means a
/// b.fd9-scope regression slipped past both pins.
#[test]
fn original_b_fd9_functions_remain_free_of_direct_internal_fred_wraps() {
    // Reuse SOURCES to load layer.rs and external_values.rs.
    let layer_src = SOURCES
        .iter()
        .find(|e| e.file == "src/storage/layer.rs")
        .expect("layer.rs must be in SOURCES")
        .source;
    let ext_src = SOURCES
        .iter()
        .find(|e| e.file == "src/storage/external_values.rs")
        .expect("external_values.rs must be in SOURCES")
        .source;
    let session_src = SOURCES
        .iter()
        .find(|e| e.file == "src/storage/session.rs")
        .expect("session.rs must be in SOURCES")
        .source;

    // The session.rs functions were added by b.62m, whose round trips the
    // workspace-wide allowlist used to excuse through needles broad enough to
    // cover several sites at once. Each of these functions now holds no
    // INTERNAL-mapped wrap at all, so pinning the whole body is exact. The
    // no-row `SessionStore::delete` is deliberately absent: its step-3
    // `Transaction` queue pushes are legitimate `Internal` wraps, and the
    // workspace-wide scan already covers its three converted round trips.
    // `load_values` is absent for a mechanical reason: its first prefix match
    // is the body-less trait declaration, which `extract_function_body`
    // cannot slice.
    let scopes: &[(&str, &str, &str)] = &[
        ("src/storage/session.rs", "pub async fn list", session_src),
        (
            "src/storage/session.rs",
            "pub async fn get_session_meta",
            session_src,
        ),
        (
            "src/storage/session.rs",
            "async fn finish_create",
            session_src,
        ),
        (
            "src/storage/session.rs",
            "async fn reconcile_in_doubt_create",
            session_src,
        ),
        (
            "src/storage/session.rs",
            "async fn undo_create_attempt",
            session_src,
        ),
        (
            "src/storage/layer.rs",
            "pub async fn list_layer_names",
            layer_src,
        ),
        (
            "src/storage/layer.rs",
            "pub async fn get_layer_mapping",
            layer_src,
        ),
        (
            "src/storage/layer.rs",
            "pub async fn read_object_header",
            layer_src,
        ),
        (
            "src/storage/external_values.rs",
            "async fn process_pending",
            ext_src,
        ),
    ];

    let b_fd9_only_allowlist: &[(&str, &str)] = &[
        // `get_layer_mapping` parses a successful GET response. Same
        // corrupt-data justification as the workspace-wide allowlist.
        ("src/storage/layer.rs", "corrupt layer mapping value"),
    ];

    let mut violations: Vec<(String, String, usize, String)> = Vec::new();
    for (file, signature_prefix, source) in scopes {
        let body = extract_function_body(source, signature_prefix).unwrap_or_else(|| {
            panic!(
                "function `{signature_prefix}` not found in `{file}` — signature changed? \
                 Update the scope list in this test."
            )
        });
        // The extracted body is itself a whole `fn`, which is exactly the scope
        // the hoisted-binding check wants — the `let … = format!(…)` and the
        // construction consuming it are both inside it, and the body's first
        // line is the signature the check's enclosing-`fn` scan finds.
        let body_lines: Vec<&str> = body.lines().collect();
        for idx in 0..body_lines.len() {
            if !is_internal_mapped_wrap_anchor(&body_lines, idx) {
                continue;
            }
            let line = body_lines[idx];
            let trimmed = line.trim();
            let excused = b_fd9_only_allowlist.iter().any(|(f, needle)| {
                f == file && (trimmed.contains(needle) || line.contains(needle))
            });
            if excused {
                continue;
            }
            violations.push((
                file.to_string(),
                signature_prefix.to_string(),
                idx + 1,
                line.to_string(),
            ));
        }
    }

    if !violations.is_empty() {
        let mut report = String::new();
        report.push_str(
            "Found `ServiceError::Internal(format!` wraps inside the b.fd9 / b.62m-scoped \
             functions — these must route through `map_fred_err_for_grpc` (single-shot read) \
             or `map_transient_fred_err` (retry-loop-wrapped), or, after `create_once`'s \
             commit was sent, through `undo_create_attempt`:\n\n",
        );
        for (file, func, lineno, text) in &violations {
            report.push_str(&format!(
                "  {file}  [in `{func}`, body line {lineno}]\n    {text}\n"
            ));
        }
        panic!("{report}");
    }
}

/// b.62m — `create_once`'s post-commit round trips must fail through
/// `undo_create_attempt`, never straight through a mapper.
///
/// Once the commit has been sent, the attempt's own names member may already be
/// in the index, so a failure must remove the attempt's entries before an
/// error is chosen, and a bare `map_transient_fred_err` or
/// `map_fred_err_for_grpc` removes nothing (why each is wrong here: see
/// `undo_create_attempt` / SDD §7.10). The workspace-wide scan cannot see that
/// regression: both are the canonical mappers it exists to steer sites onto.
/// So this pins the shape directly:
///
/// * neither `finish_create` nor `reconcile_in_doubt_create` calls a mapper in
///   its own body (exact today — all their classification happens inside
///   `undo_create_attempt`);
/// * each `undo_create_attempt` call carries the op of the round trip it
///   answers for, in source order — one per post-commit round trip. A dropped
///   call or a swapped label fails here, and the label is what reaches the
///   client's `UNAVAILABLE` message and the undo's WARN line;
/// * both duplicate-name scans — `create_once`'s under the `WATCH` and the
///   in-doubt re-scan in `reconcile_in_doubt_create` — decide a competing
///   holder through exactly one `names_a_competing_holder` call each, so the
///   two cannot drift. No behavioural test drives the re-scan past an earlier
///   attempt's member, so an inline `sid != session_id` there would pass every
///   one of them; this count is its only guard.
///
/// The behaviour behind each call is driven for real in `b_62m_tests.rs`; this
/// is the per-site guard that survives a refactor no stall happens to hit.
#[test]
fn create_session_post_commit_failures_route_through_undo_create_attempt() {
    let session_src = SOURCES
        .iter()
        .find(|e| e.file == "src/storage/session.rs")
        .expect("session.rs must be in SOURCES")
        .source;

    let cases: &[(&str, &[&str])] = &[
        ("async fn finish_create", &["HSET session meta (create)"]),
        (
            "async fn reconcile_in_doubt_create",
            &[
                "SISMEMBER path names index (in-doubt reconcile)",
                "SMEMBERS path names index (in-doubt re-scan)",
            ],
        ),
    ];

    for (signature_prefix, want_ops) in cases {
        let body = extract_function_body(session_src, signature_prefix).unwrap_or_else(|| {
            panic!(
                "function `{signature_prefix}` not found in `src/storage/session.rs` — signature \
                 changed? Update this test."
            )
        });
        // Code only: a comment explaining why a mapper is NOT called here must
        // not read as a call.
        let code: Vec<&str> = body
            .lines()
            .map(|line| match line.find("//") {
                Some(pos) => &line[..pos],
                None => line,
            })
            .collect();

        for mapper in ["map_transient_fred_err(", "map_fred_err_for_grpc("] {
            let calls: Vec<&str> = code
                .iter()
                .copied()
                .filter(|line| line.contains(mapper))
                .collect();
            assert!(
                calls.is_empty(),
                "`{signature_prefix}` calls `{mapper}` directly. A round trip after \
                 `create_once`'s commit was sent must fail through `undo_create_attempt`, \
                 which removes the attempt's entries before choosing the error (see its \
                 rustdoc / SDD §7.10). Offending lines: {calls:?}"
            );
        }

        assert_eq!(
            undo_call_ops(&code),
            want_ops.to_vec(),
            "`{signature_prefix}` must fail each post-commit round trip through its own \
             `undo_create_attempt` call, labelled with that round trip's op"
        );
    }

    for signature_prefix in [
        "async fn create_once(",
        "async fn reconcile_in_doubt_create(",
    ] {
        let body = extract_function_body(session_src, signature_prefix).unwrap_or_else(|| {
            panic!(
                "function `{signature_prefix}` not found in `src/storage/session.rs` — signature \
                 changed? Update this test."
            )
        });
        let calls = body
            .lines()
            .map(strip_line_comment)
            .map(|code| code.matches("names_a_competing_holder(").count())
            .sum::<usize>();
        assert_eq!(
            calls, 1,
            "`{signature_prefix}`'s duplicate-name scan must decide a competing holder through \
             exactly one `names_a_competing_holder(` call, so both scans skip every id the call \
             created and cannot drift apart (see `names_a_competing_holder` / SDD §7.10)"
        );
    }
}

/// The string literals passed to each `undo_create_attempt(` call in `code`, in
/// source order. A call's argument list is taken to run from the line naming
/// the method to the line carrying its `.await`, which is how every call site
/// in `session.rs` is formatted; a literal-free call contributes nothing, so a
/// call that lost its label is caught by the count.
fn undo_call_ops(code: &[&str]) -> Vec<String> {
    let mut ops = Vec::new();
    let mut idx = 0;
    while idx < code.len() {
        if code[idx].contains("undo_create_attempt(") {
            loop {
                ops.extend(code[idx].split('"').skip(1).step_by(2).map(str::to_owned));
                if code[idx].contains(".await") || idx + 1 == code.len() {
                    break;
                }
                idx += 1;
            }
        }
        idx += 1;
    }
    ops
}

/// The no-retry-loop round trips b.62m converted to `map_fred_err_for_grpc`, as
/// `(file, impl anchor, function signature prefix, op label)`.
///
/// The impl anchor narrows the function lookup to the text after it, for a
/// function whose first match by signature is a body-less trait declaration
/// (`load_values`). Every signature prefix ends in `(` so it cannot also match
/// a longer function name.
const B_62M_GRPC_MAPPED_SITES: &[(&str, Option<&str>, &str, &str)] = &[
    (
        "src/storage/session.rs",
        None,
        "pub async fn delete(",
        "HGETALL session meta (delete)",
    ),
    (
        "src/storage/session.rs",
        None,
        "pub async fn delete(",
        "SREM path index (delete)",
    ),
    (
        "src/storage/session.rs",
        None,
        "pub async fn delete(",
        "SREM path names index (delete)",
    ),
    (
        "src/storage/session.rs",
        None,
        "pub async fn list(",
        "SMEMBERS path names index (list)",
    ),
    (
        "src/storage/session.rs",
        None,
        "pub async fn list(",
        "HGETALL session meta (list)",
    ),
    (
        "src/storage/session.rs",
        None,
        "pub async fn get_session_meta(",
        "HGETALL session meta",
    ),
    // `create_layer` is now a thin wrapper over `create_layer_with_operation`
    // (b.v6d), which holds every round trip below. b.62m's `SET layer mapping
    // (overwrite)` is gone: the mapping flip is the commit script's, mapped as
    // `EVALSHA create_layer_commit`.
    (
        "src/storage/layer.rs",
        None,
        "pub async fn create_layer_with_operation(",
        "GET layer mapping (create)",
    ),
    (
        "src/storage/layer.rs",
        None,
        "pub async fn create_layer_with_operation(",
        "MGET layer mapping + create operation record",
    ),
    (
        "src/storage/layer.rs",
        None,
        "pub async fn create_layer_with_operation(",
        "HSET obj:hdr superseded (overwrite)",
    ),
    (
        "src/storage/layer.rs",
        None,
        "pub async fn create_layer_with_operation(",
        "EVALSHA create_layer_commit",
    ),
    // b.ua9 removed the `XADD deleted on old stream (overwrite)` site: the
    // old-stream wake-up now runs after the commit point in the overwrite tail
    // (`LayerStore::run_overwrite_tail`), where a failure is logged at WARN and
    // retried, never answered, so it maps to no `ServiceError`.
    //
    // The `CreateLayer` handler's pre-allocation read of the operation record
    // (b.v6d). Its error goes through
    // `classify_create_operation_record_read_error`, which peels off only the
    // `WRONGTYPE` verdict and maps every other failure here.
    (
        "src/storage/layer.rs",
        None,
        "fn classify_create_operation_record_read_error(",
        "GET create operation record",
    ),
    (
        "src/storage/external_values.rs",
        Some("impl ExternalValueStore for ValkeyExternalValueStore"),
        "async fn load_values(",
        "GET ext_data",
    ),
];

/// The body of the function `signature_prefix` in `file`'s SOURCES text,
/// searched from `impl_anchor` when one is given.
fn scoped_function_body(
    file: &str,
    impl_anchor: Option<&str>,
    signature_prefix: &str,
) -> &'static str {
    let source = SOURCES
        .iter()
        .find(|e| e.file == file)
        .unwrap_or_else(|| panic!("`{file}` must be in SOURCES"))
        .source;
    let scoped = match impl_anchor {
        Some(anchor) => {
            let at = source.find(anchor).unwrap_or_else(|| {
                panic!("`{anchor}` not found in `{file}` — impl renamed? Update this test.")
            });
            &source[at..]
        }
        None => source,
    };
    extract_function_body(scoped, signature_prefix).unwrap_or_else(|| {
        panic!(
            "function `{signature_prefix}` not found in `{file}` — signature changed? Update \
             this test."
        )
    })
}

/// The name of the innermost call whose argument list holds byte `pos` of
/// `code`: walk back from `pos` to the first unmatched `(` and read the
/// identifier directly before it (path segments excluded). `None` when `pos`
/// is not inside any call.
///
/// Parentheses inside string literals are not special-cased: every text this
/// runs on is comment-stripped call arguments with balanced literals.
fn enclosing_call_name(code: &str, pos: usize) -> Option<&str> {
    let bytes = code.as_bytes();
    let mut depth = 0usize;
    let mut i = pos;
    while i > 0 {
        i -= 1;
        match bytes[i] {
            b')' => depth += 1,
            b'(' if depth == 0 => {
                // A macro call (`format!(`) is named without its `!`.
                let head = code[..i].strip_suffix('!').unwrap_or(&code[..i]);
                let start = head
                    .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .map_or(0, |p| p + 1);
                let name = &head[start..];
                return (!name.is_empty()).then_some(name);
            }
            b'(' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// b.62m — each converted no-loop round trip carries its op label inside a
/// `map_fred_err_for_grpc(` call, per site.
///
/// The workspace-wide scan only sees a site that goes back to a direct
/// `Internal(format!` wrap; it treats both canonical mappers as correct. None
/// of the functions here runs a retry loop, so `map_transient_fred_err` is the
/// wrong one: its `Aborted` is the in-loop retry signal, and nothing here
/// retries. `HSET obj:hdr superseded (overwrite)` was exactly that before
/// b.62m, and reverting it to the in-loop mapper passed every other test.
/// Several of these sites sit behind an earlier data-pool command, so no
/// whole-pool stall can reach them alone and no behavioural test drives them;
/// this table is their only per-site guard. The sites a stall does reach
/// first (`delete`'s `HGETALL`, `list`'s `SMEMBERS`, `get_session_meta`,
/// `load_values`) and the old-stream `XADD` are driven in `b_62m_tests.rs` and
/// `update_layer_admission_test.rs` as well, and are listed here so the table
/// is the whole inventory.
///
/// Per site: the quoted op label appears exactly once in the function's code
/// (comments stripped), and the innermost call around it is
/// `map_fred_err_for_grpc`. A return to a direct `Internal` wrap drops the
/// label and fails the count; a swap to `map_transient_fred_err` fails the
/// callee check.
#[test]
fn b_62m_no_loop_round_trips_each_route_through_the_grpc_mapper() {
    let mut problems: Vec<String> = Vec::new();
    for (file, impl_anchor, signature_prefix, op) in B_62M_GRPC_MAPPED_SITES {
        let body = scoped_function_body(file, *impl_anchor, signature_prefix);
        let code = body
            .lines()
            .map(strip_line_comment)
            .collect::<Vec<_>>()
            .join("\n");
        let label = format!("\"{op}\"");
        let positions: Vec<usize> = code.match_indices(&label).map(|(pos, _)| pos).collect();
        let [pos] = positions.as_slice() else {
            problems.push(format!(
                "{file} `{signature_prefix}`: the op label {label} appears {} times in the \
                 function's code, expected exactly once — the round trip went back to a direct \
                 wrap, lost its label, or gained a second site under the same label",
                positions.len()
            ));
            continue;
        };
        let callee = enclosing_call_name(&code, *pos);
        if callee != Some("map_fred_err_for_grpc") {
            problems.push(format!(
                "{file} `{signature_prefix}`: {label} is an argument of {callee:?}, expected \
                 `map_fred_err_for_grpc` — no retry loop encloses this function, so a transient \
                 failure must answer `Unavailable`, not the in-loop `Aborted` retry signal"
            ));
        }
    }
    assert!(
        problems.is_empty(),
        "b.62m's converted no-loop round trips must each route through \
         `map_fred_err_for_grpc`:\n\n  {}",
        problems.join("\n  ")
    );
}

/// b.62m — the functions whose round trips route through a mapper leave the
/// timeout counting to it.
///
/// Both mappers count a timed-out fred error on the origin they are handed.
/// Each b.62m site used to call `observe_error` itself before its direct
/// `Internal` wrap; a site that kept that call beside a mapper would count
/// every timeout twice. The behavioural tests pin the count at sites a stall
/// reaches first; the ones behind an earlier data-pool command (`delete`'s two
/// `SREM`s, `list`'s per-session `HGETALL`, `create_layer_with_operation`'s
/// mapping `GET` / `MGET`, and the handler's operation-record `GET`) are
/// pinned only here. `finish_create` and `reconcile_in_doubt_create` hand
/// their failure to `undo_create_attempt`, which counts it. None of these
/// functions has any other reason to count, so the whole body is checked.
///
/// `create_layer_with_operation`'s commit `EVALSHA` is counted elsewhere: by
/// `dispatch_create_layer_commit`, which receives every run's error, while
/// the function's mapping of that error takes an origin without metrics.
/// [`create_layer_commit_runs_are_counted_by_their_dispatch_and_not_again_by_the_mapping`]
/// pins both halves, since this scan sees neither.
#[test]
fn b_62m_mapped_functions_leave_timeout_counting_to_the_mapper() {
    let scopes: &[(&str, Option<&str>, &str)] = &[
        ("src/storage/session.rs", None, "pub async fn delete("),
        ("src/storage/session.rs", None, "pub async fn list("),
        (
            "src/storage/session.rs",
            None,
            "pub async fn get_session_meta(",
        ),
        ("src/storage/session.rs", None, "async fn finish_create("),
        (
            "src/storage/session.rs",
            None,
            "async fn reconcile_in_doubt_create(",
        ),
        (
            "src/storage/layer.rs",
            None,
            "pub async fn create_layer_with_operation(",
        ),
        (
            "src/storage/layer.rs",
            None,
            "pub async fn resolve_create_layer_operation(",
        ),
        (
            "src/storage/layer.rs",
            None,
            "fn classify_create_operation_record_read_error(",
        ),
        (
            "src/storage/external_values.rs",
            Some("impl ExternalValueStore for ValkeyExternalValueStore"),
            "async fn load_values(",
        ),
    ];
    let mut problems: Vec<String> = Vec::new();
    for (file, impl_anchor, signature_prefix) in scopes {
        let body = scoped_function_body(file, *impl_anchor, signature_prefix);
        let calls: Vec<&str> = body
            .lines()
            .map(strip_line_comment)
            .filter(|code| code.contains("observe_error("))
            .map(str::trim)
            .collect();
        if !calls.is_empty() {
            problems.push(format!("{file} `{signature_prefix}`: {calls:?}"));
        }
    }
    assert!(
        problems.is_empty(),
        "these functions route every fred error through a mapper (or `undo_create_attempt`), \
         which counts the timeout itself; an `observe_error(` beside it counts every timeout \
         twice:\n\n  {}",
        problems.join("\n  ")
    );
}

/// b.ua9 × b.pa1 — a `CreateLayer` commit run's error is observed once, by
/// the dispatch that receives it, and counted no second time by the mapping
/// `create_layer_with_operation` gives up with.
///
/// Every run of `create_layer_commit.lua` (the call's, its finisher's, and
/// the orphan sweep's completion of a stranded overwrite) goes through
/// `LayerStore::dispatch_create_layer_commit`, which observes a failed run's
/// error on the txn pool where it receives it: a command timeout is counted,
/// and a store write refused `-OOM` is noted on the store out-of-memory
/// signal. That has to happen there, because a caller may run the commit
/// again, re-plan or give up without ever answering the error. The call's
/// mapping of the error it gives up with (`"EVALSHA create_layer_commit"`,
/// through `map_fred_err_for_grpc`, which counts a timeout on the origin it
/// is handed) must therefore take an origin without metrics,
/// `FredErrOrigin::metrics(None, ..)`. With a counting
/// `FredErrOrigin::client(..)` there, every give-up's last timeout is counted
/// twice, and the observe-scan above still passes, since it looks only for
/// `observe_error(`.
///
/// Comments stripped, two checks: (1) after the dispatch's last
/// `evalsha_with_reload` (the real run, after the test-only fault branch),
/// the run's error is observed through `FredErrOrigin::client` on the txn
/// pool, the origin that counts; (2) the op label appears exactly once in
/// `create_layer_with_operation`, as an argument of `map_fred_err_for_grpc`
/// whose origin is `FredErrOrigin::metrics(None, ..)`. The count itself is
/// pinned behaviourally through the injected-fault path in `b_ua9_tests.rs`
/// (`create_layer_overwrite_whose_two_commit_runs_time_out_counts_each_timeout_once`),
/// which observes a fault on its own line; (1) is what pins the real run's.
#[test]
fn create_layer_commit_runs_are_counted_by_their_dispatch_and_not_again_by_the_mapping() {
    let strip = |body: &str| {
        body.lines()
            .map(strip_line_comment)
            .collect::<Vec<_>>()
            .join("\n")
    };
    let squeeze = |text: &str| text.split_whitespace().collect::<String>();

    let dispatch = strip(scoped_function_body(
        "src/storage/layer.rs",
        None,
        "async fn dispatch_create_layer_commit(",
    ));
    let real_run = dispatch
        .rfind("evalsha_with_reload")
        .expect("the dispatch must send the commit script — renamed? Update this test.");
    let observation = "FredErrOrigin::client(&self.client,POOL_LABEL_TXN).observe_error(";
    assert!(
        squeeze(&dispatch[real_run..]).contains(observation),
        "`dispatch_create_layer_commit` must observe a failed commit run's error where it \
         receives it, through the counting txn-pool origin (`{observation}`): its callers \
         re-run, re-plan or give up without answering it, so nothing else counts the timeout \
         or notes the -OOM. Code after the real run:\n{}",
        &dispatch[real_run..]
    );

    let create = strip(scoped_function_body(
        "src/storage/layer.rs",
        None,
        "pub async fn create_layer_with_operation(",
    ));
    let label = "\"EVALSHA create_layer_commit\"";
    let positions: Vec<usize> = create.match_indices(label).map(|(pos, _)| pos).collect();
    let [pos] = positions.as_slice() else {
        panic!(
            "the op label {label} appears {} times in `create_layer_with_operation`'s code, \
             expected exactly once (the mapping of the error the call gives up with)",
            positions.len()
        );
    };
    assert_eq!(
        enclosing_call_name(&create, *pos),
        Some("map_fred_err_for_grpc"),
        "the commit's give-up error is mapped by `map_fred_err_for_grpc`"
    );
    let call = "map_fred_err_for_grpc(";
    let args_start = create[..*pos]
        .rfind(call)
        .expect("the enclosing call's name was just read")
        + call.len();
    let origin = squeeze(&create[args_start..*pos]);
    assert!(
        origin.starts_with("FredErrOrigin::metrics(None,"),
        "the commit's give-up mapping must take an origin without metrics \
         (`FredErrOrigin::metrics(None, ..)`): the dispatch already counted the run's timeout, \
         and a counting origin counts it twice; got `{origin}`"
    );
}

#[test]
fn enclosing_call_name_reads_the_innermost_call() {
    // Sanity check on the helper the per-site mapper table rests on, in both
    // directions: a helper that always answered `map_fred_err_for_grpc`, or
    // read an outer call, would pass every site vacuously.
    let cases: &[(&str, &str, Option<&str>)] = &[
        (
            "map_err(|e| map_fred_err_for_grpc(origin, \"OP\", e))",
            "\"OP\"",
            Some("map_fred_err_for_grpc"),
        ),
        (
            "map_fred_err_for_grpc(\n    FredErrOrigin::client(&self.client, POOL),\n    \"OP\",\n    e,\n)",
            "\"OP\"",
            Some("map_fred_err_for_grpc"),
        ),
        (
            "map_transient_fred_err(\n    FredErrOrigin::client(&c, P),\n    \"OP\",\n    e,\n)",
            "\"OP\"",
            Some("map_transient_fred_err"),
        ),
        (
            "map_fred_err_for_grpc(origin, x, e); ServiceError::Internal(format!(\"OP {e}\"))",
            "\"OP",
            Some("format"),
        ),
        (
            "map_fred_err_for_grpc(origin, x, e); let op = \"OP\";",
            "\"OP\"",
            None,
        ),
    ];
    for (code, needle, expected) in cases {
        let pos = code.find(needle).expect("needle must be in the snippet");
        assert_eq!(
            enclosing_call_name(code, pos),
            *expected,
            "misread the call around {needle} in {code:?}"
        );
    }
}

/// Extract a single function body from a Rust source file. Slices from
/// the line whose trimmed start matches `signature_prefix` through the
/// matching closing brace. Implementation copied verbatim from the
/// pre-b.u96 mapper_routing_test (used by the b.fd9-scope test above).
fn extract_function_body<'a>(source: &'a str, signature_prefix: &str) -> Option<&'a str> {
    let lines: Vec<&str> = source.lines().collect();
    let start_idx = lines
        .iter()
        .position(|line| line.trim_start().starts_with(signature_prefix))?;

    let mut depth: i32 = 0;
    let mut seen_open = false;
    let mut end_idx = start_idx;

    for (offset, line) in lines.iter().enumerate().skip(start_idx) {
        let code = match line.find("//") {
            Some(pos) => &line[..pos],
            None => *line,
        };
        for ch in code.chars() {
            match ch {
                '{' => {
                    depth += 1;
                    seen_open = true;
                }
                '}' => {
                    depth -= 1;
                }
                _ => {}
            }
        }
        if seen_open && depth == 0 {
            end_idx = offset;
            break;
        }
    }

    if !seen_open || depth != 0 {
        return None;
    }

    let body_start: usize = lines.iter().take(start_idx).map(|l| l.len() + 1).sum();
    let body_end: usize = lines
        .iter()
        .take(end_idx + 1)
        .map(|l| l.len() + 1)
        .sum::<usize>()
        .min(source.len());
    Some(&source[body_start..body_end])
}

/// `src/grpc/presence.rs` — pinned narrow-scope test, kept from the
/// original b.fd9 follow-up. presence.rs currently has zero
/// `ServiceError::Internal(format!` wraps (every site was migrated to
/// `map_fred_err_for_grpc` during b.fd9), so this test should always
/// pass with an empty violations list. A regression that adds one new
/// wrap in presence.rs would fail this AND the workspace-wide scan;
/// the narrower test gives a more focused failure message.
///
/// The "deliberate exception" block at the top of SOURCES documents
/// why presence.rs stays in SOURCES even with zero current matches:
/// removing it would silently delete this subsystem-level pin.
#[test]
fn presence_rs_routes_fred_errors_through_mappers() {
    let presence_src = SOURCES
        .iter()
        .find(|e| e.file == "src/grpc/presence.rs")
        .expect("presence.rs must be in SOURCES")
        .source;

    let mut violations: Vec<(usize, String)> = Vec::new();
    let presence_lines: Vec<&str> = presence_src.lines().collect();
    for idx in 0..presence_lines.len() {
        if !is_internal_mapped_wrap_anchor(&presence_lines, idx) {
            continue;
        }
        violations.push((idx + 1, presence_lines[idx].to_string()));
    }

    if !violations.is_empty() {
        let mut report = String::new();
        report.push_str(
            "Found `ServiceError::Internal(format!` wraps in `src/grpc/presence.rs` — every \
             call site should route transient errors through `map_fred_err_for_grpc` so \
             clients get `Code::Unavailable` (retriable) rather than `Code::Internal` \
             (non-retriable):\n\n",
        );
        for (lineno, text) in &violations {
            report.push_str(&format!("  src/grpc/presence.rs:{lineno}\n    {text}\n"));
        }
        panic!("{report}");
    }
}

#[test]
fn extract_function_body_finds_a_named_function() {
    // Sanity check on the extractor used by the b.fd9-scope test.
    let layer_src = SOURCES
        .iter()
        .find(|e| e.file == "src/storage/layer.rs")
        .expect("layer.rs must be in SOURCES")
        .source;
    let body = extract_function_body(layer_src, "pub async fn list_layer_names")
        .expect("list_layer_names must be present");
    assert!(
        body.contains("SMEMBERS layers index"),
        "extracted body should contain the SMEMBERS op label — got {} bytes",
        body.len()
    );
    assert!(
        body.contains("map_fred_err_for_grpc"),
        "extracted body should call the mapper today — got {} bytes",
        body.len()
    );
}

// =============================================================================
// b.459 — `mutation_probe_keys` must track `execute_mutation` variant for
// variant.
// =============================================================================

/// b.459: the pre-`MULTI` keyspace type probe is only as good as its key list.
/// A `StorageMutation` variant that `execute_mutation` writes but
/// `mutation_probe_keys` does not name is a key that can still raise a
/// `WRONGTYPE` *inside* the `EXEC` — where Valkey neither aborts the
/// transaction nor rolls back its siblings, so the commit half-lands. That is
/// precisely the defect b.459 closed, re-opened one variant at a time.
///
/// # What the compiler already guarantees, and what it cannot
///
/// Both functions `match` on `&StorageMutation` with **no `_` arm**, so adding
/// a variant to the enum fails to compile until both are updated. That is the
/// forcing function, and it is stronger than any test.
///
/// What it cannot see is a catch-all arm being *added* — which would silently
/// restore the drift it prevents. So this fence asserts exactly the two things
/// the type system does not: the same set of `StorageMutation::` variant names
/// appears in each body, and neither body contains a catch-all arm in ANY of its
/// spellings (`_ =>`, and the bare bindings `other =>` / `m =>` that mean the
/// same thing — see [`is_catch_all_arm`]).
///
/// # Why a source-shape test rather than a behavioural one
///
/// The behavioural coverage (`b_459_tests.rs`) drives one commit per key
/// family, so it can only fail for a family a test was written for. A new
/// variant is exactly the case nobody writes a test for — and it is the case
/// the half-commit shape is reached through. This test costs one string scan
/// and fails on the diff that introduces the gap rather than on the incident
/// that exploits it.
#[test]
fn mutation_probe_keys_covers_the_same_variants_execute_mutation_writes() {
    let layer_src = SOURCES
        .iter()
        .find(|e| e.file == "src/storage/layer.rs")
        .expect("layer.rs must be in SOURCES")
        .source;

    let execute_body = extract_function_body(layer_src, "async fn execute_mutation")
        .expect("`execute_mutation` must be present in src/storage/layer.rs — signature changed?");
    let probe_body = extract_function_body(layer_src, "fn mutation_probe_keys").expect(
        "`mutation_probe_keys` must be present in src/storage/layer.rs — the b.459 pre-`MULTI` \
         type probe has no key list without it, and every mutated key would be back to raising \
         its WRONGTYPE inside the EXEC",
    );

    let execute_variants = storage_mutation_variants(execute_body);
    let probe_variants = storage_mutation_variants(probe_body);

    assert!(
        !execute_variants.is_empty(),
        "the variant scan found nothing in `execute_mutation` — the extractor or the \
         `StorageMutation::` spelling changed, and this fence is passing vacuously"
    );

    let missing_from_probe: Vec<&String> = execute_variants.difference(&probe_variants).collect();
    assert!(
        missing_from_probe.is_empty(),
        "`execute_mutation` writes {missing_from_probe:?} but `mutation_probe_keys` does not \
         name them. Every key a mutation writes must be probed before `MULTI` — or excluded \
         with a stated reason in `mutation_probe_keys`'s doc-comment, which is what the \
         `DEL`-first families are. An unprobed key raises its `WRONGTYPE` inside the `EXEC`, \
         where the siblings still commit and the caller learns only the first error frame"
    );

    let missing_from_execute: Vec<&String> = probe_variants.difference(&execute_variants).collect();
    assert!(
        missing_from_execute.is_empty(),
        "`mutation_probe_keys` names {missing_from_execute:?} but `execute_mutation` does not \
         handle them. The probe would refuse commits over a key nothing writes — a write \
         outage with no corresponding corruption"
    );

    for (name, body) in [
        ("execute_mutation", execute_body),
        ("mutation_probe_keys", probe_body),
    ] {
        let wildcard = body
            .lines()
            .map(strip_line_comment)
            .find(|code| is_catch_all_arm(code));
        assert!(
            wildcard.is_none(),
            "`{name}` contains a catch-all match arm ({:?}). The exhaustive `match` over \
             `StorageMutation` is the compile-time forcing function tying these two functions \
             together: with a `_ =>` arm — or any bare binding such as `other =>`, which is the \
             same thing with a name — a new variant compiles cleanly on both sides and its new \
             key is silently unprobed",
            wildcard.unwrap_or("").trim()
        );
    }
}

/// Does `code` open a match arm whose pattern matches EVERYTHING?
///
/// Two spellings, one meaning: `_ =>` and a bare binding like `other =>` /
/// `m =>` both swallow every remaining variant, and only the first is obvious on
/// sight. The original check was `trim_start().starts_with("_ =>")`, which read
/// the one spelling literally — it missed `other =>`, `m =>`, and even `_  =>`
/// with two spaces, each of which reopens the exact drift this fence exists to
/// prevent.
///
/// The rule: take the leading identifier-ish token, skip any whitespace, and
/// require `=>` next. The token is a catch-all iff it is `_` or a bare
/// lowercase-initial identifier — which is precisely Rust's binding-pattern
/// spelling. Everything else that can start an arm is therefore excluded by
/// construction and not by a hand-kept denylist:
///
/// * `StorageMutation::SetField { .. } =>` — the token stops at `:`, so what
///   follows the token is not `=>`;
/// * `Some(x) =>`, `Ok(v) =>` — same, the token is followed by `(`;
/// * `None =>`, `Self::Foo =>` — uppercase initial, i.e. a path/unit variant,
///   not a binding;
/// * `1 =>`, `"s" =>`, `b'x' =>` — literals do not start with a lowercase
///   letter or `_`;
/// * `} => {` (a multi-line pattern's closing brace) — no leading token at all.
///
/// `true` / `false` are the one lowercase-initial *literal* pattern in the
/// language, so they are excluded explicitly; they match one value, not
/// everything.
///
/// A guarded catch-all (`_ if cond =>`) is deliberately NOT flagged: a guard
/// does not count toward exhaustiveness, so the compiler still forces every
/// variant to be handled and the forcing function survives.
fn is_catch_all_arm(code: &str) -> bool {
    let trimmed = code.trim_start();
    let token_len = trimmed
        .find(|c: char| !c.is_alphanumeric() && c != '_')
        .unwrap_or(trimmed.len());
    let token = &trimmed[..token_len];
    if token.is_empty() || token == "true" || token == "false" {
        return false;
    }
    let binding = token == "_"
        || token
            .starts_with(|c: char| c.is_ascii_lowercase() || c == '_')
            // A binding is an identifier: no digits-only, no `1abc`. The
            // leading-char check above already rules out a leading digit.
            && token.chars().all(|c| c.is_alphanumeric() || c == '_');
    binding && trimmed[token_len..].trim_start().starts_with("=>")
}

/// Every `StorageMutation::<Variant>` name mentioned in `body`, comments
/// stripped.
///
/// Deliberately a plain scan rather than a parse: what is under test is that
/// the two match statements mention the same names, and the scan is the whole
/// claim.
fn storage_mutation_variants(body: &str) -> std::collections::HashSet<String> {
    const MARKER: &str = "StorageMutation::";
    let mut found = std::collections::HashSet::new();
    for line in body.lines() {
        let mut rest = strip_line_comment(line);
        while let Some(pos) = rest.find(MARKER) {
            let tail = &rest[pos + MARKER.len()..];
            let end = tail
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(tail.len());
            found.insert(tail[..end].to_owned());
            rest = &tail[end..];
        }
    }
    found.remove("");
    found
}

/// `line` with any `//`-introduced trailing comment removed.
fn strip_line_comment(line: &str) -> &str {
    match line.find("//") {
        Some(pos) => &line[..pos],
        None => line,
    }
}

#[test]
fn storage_mutation_variant_scan_reads_names_and_ignores_comments() {
    // Sanity check on the scanner the b.459 fence rests on. Without it, a
    // scanner that silently found nothing would make that fence vacuous in
    // exactly the direction that matters: no variants found on either side
    // means no difference between them.
    let sample = "match m {\n\
        // StorageMutation::CommentedOut => {}\n\
        StorageMutation::SetField { field } => f(field), // StorageMutation::AlsoComment\n\
        StorageMutation::DeleteField { .. } => {}\n\
        }";
    let found = storage_mutation_variants(sample);
    let expected: std::collections::HashSet<String> = ["SetField", "DeleteField"]
        .into_iter()
        .map(String::from)
        .collect();
    assert_eq!(
        found, expected,
        "the scan must read variant names from code and ignore both comment forms"
    );
}

/// Sanity check on [`is_catch_all_arm`], the other half of the b.459 fence.
///
/// Both directions are load-bearing and both have a real failure mode. A
/// false NEGATIVE (a catch-all spelling the scanner does not recognise) makes
/// the fence silently permissive — which is what the previous literal
/// `starts_with("_ =>")` check was for `other =>`, `m =>` and `_  =>`. A false
/// POSITIVE fails the build on a perfectly ordinary arm, and since this test is
/// the only place the predicate is specified, it is where that gets caught.
#[test]
fn catch_all_arm_scan_flags_every_binding_spelling_and_no_ordinary_arm() {
    for positive in [
        "_ => {}",
        "            _ => {}",
        "_  => {}", // two spaces — the literal check missed this
        "_=> {}",   // no space at all
        "other => {}",
        "m => {}",
        "mutation => unreachable!(),",
        "_ => panic!(\"x\"),",
    ] {
        assert!(
            is_catch_all_arm(positive),
            "{positive:?} matches every remaining variant, so it defeats the exhaustive `match` \
             that ties `execute_mutation` and `mutation_probe_keys` together — the scanner must \
             flag it"
        );
    }

    for negative in [
        "StorageMutation::SetField { field } => {}",
        "StorageMutation::DeleteField { .. } => {}",
        "StorageMutation::Foo { .. } => {}",
        "Some(x) => x,",
        "Ok(v) => v,",
        "None => return,",
        "Self::Hash => \"hash\",",
        "1 => {}",
        "\"s\" => {}",
        "true => {}",
        "} => {",          // a multi-line pattern's closing brace
        "_ if cond => {}", // a guard does not count toward exhaustiveness
        "",
        "    let x = 1;",
        ".map(|code| code.trim())",
    ] {
        assert!(
            !is_catch_all_arm(negative),
            "{negative:?} is not a catch-all arm — flagging it would fail the build on ordinary \
             code and the fence would be deleted rather than fixed"
        );
    }
}

/// Every 0-based line index of `sample` that [`is_internal_mapped_wrap_anchor`]
/// fires on.
fn anchor_line_indices(sample: &[&str]) -> Vec<usize> {
    (0..sample.len())
        .filter(|idx| is_internal_mapped_wrap_anchor(sample, *idx))
        .collect()
}

/// Sanity check on [`is_internal_mapped_wrap_anchor`] — the predicate every
/// scan in this file routes through — in both directions.
///
/// Each scanner predicate here is specified by a self-test (cf.
/// [`catch_all_arm_scan_flags_every_binding_spelling_and_no_ordinary_arm`]),
/// and this one is specified on *snippets* rather than on production source so
/// the spellings it must cover can be written down even when no site uses them
/// today. The spellings a refactor reaches for next are exactly the ones the
/// guard has never seen.
///
/// Both directions have a live failure mode, and they are not symmetric:
///
/// * A false NEGATIVE is **silent**. The site simply leaves the scan, its
///   ALLOWLIST row goes stale, and a new non-allowlisted wrap in that spelling
///   is never reported. Two of these were real: rewriting the keyspace-probe
///   consumption as `ServiceError::Internal(detail.clone())` took the whole
///   site out of the guard, and a rustfmt line break after `let detail =` did
///   the same.
/// * A false POSITIVE is loud but corrosive — it fails the build on ordinary
///   code (an unrelated `let detail = format!("…")` elsewhere in a 10k-line
///   file used to do exactly that), and a guard that cries wolf gets deleted
///   rather than fixed.
#[test]
fn internal_mapped_wrap_anchor_scan_flags_every_wrap_spelling_and_no_ordinary_line() {
    // (what it is, the snippet, the line indices that must anchor)
    let positives: &[(&str, &[&str], &[usize])] = &[
        (
            "anchor 1 — the `format!` written inline into `Internal`",
            &[
                "fn f(e: Error) -> ServiceError {",
                "    ServiceError::Internal(format!(\"boom: {e}\"))",
                "}",
            ],
            &[1],
        ),
        (
            "anchor 2 — refusal-variant construction with an inline `detail: format!(`",
            &[
                "fn f(e: Error) -> ServiceError {",
                "    ServiceError::PermanentKeyspaceRefusal {",
                "        refusal: KeyspaceRefusal::WrongType,",
                "        detail: format!(\"boom: {e}\"),",
                "    }",
                "}",
            ],
            &[1],
        ),
        (
            "anchor 3 — `let detail = format!(` moved into `Internal(detail)`",
            &[
                "fn f(e: Error) -> ServiceError {",
                "    let detail = format!(\"boom: {e}\");",
                "    ServiceError::Internal(detail)",
                "}",
            ],
            &[1],
        ),
        (
            "anchor 3 — hoisted binding handed on by `.clone()` (the verified false negative)",
            &[
                "fn f(e: Error) -> (ServiceError, String) {",
                "    let detail = format!(\"boom: {e}\");",
                "    (ServiceError::Internal(detail.clone()), detail)",
                "}",
            ],
            &[1],
        ),
        (
            "anchor 3 — `.to_owned()` / `.into()` are the same hand-off",
            &[
                "fn f(e: Error) -> Vec<ServiceError> {",
                "    let detail = format!(\"boom: {e}\");",
                "    let second = format!(\"boom again: {e}\");",
                "    vec![",
                "        ServiceError::Internal(detail.to_owned()),",
                "        ServiceError::Internal(second.into()),",
                "    ]",
                "}",
            ],
            &[1, 2],
        ),
        (
            "anchor 3 — `let mut`, mutated before the wrap",
            &[
                "fn f(e: Error) -> ServiceError {",
                "    let mut detail = format!(\"boom: {e}\");",
                "    detail.push_str(\" (retries exhausted)\");",
                "    ServiceError::Internal(detail)",
                "}",
            ],
            &[1],
        ),
        (
            "anchor 3 — rustfmt broke the line after `=`, leaving `format!(` below it",
            &[
                "fn f(e: Error) -> ServiceError {",
                "    let detail =",
                "        format!(\"a message long enough that rustfmt moved it down: {e}\");",
                "    ServiceError::Internal(detail)",
                "}",
            ],
            &[1],
        ),
        (
            "anchor 3 — consumed by the refusal variant's `detail` field shorthand \
             (`classify_keyspace_probe_error`'s real shape: one `format!`, two arms)",
            &[
                "fn f(e: Error, site: ProbeSite) -> ServiceError {",
                "    let detail = format!(\"boom: {e}\");",
                "    match site {",
                "        ProbeSite::Update => ServiceError::Internal(detail),",
                "        ProbeSite::Create { .. } => ServiceError::PermanentKeyspaceRefusal {",
                "            refusal: KeyspaceRefusal::WrongType,",
                "            detail,",
                "        },",
                "    }",
                "}",
            ],
            &[1],
        ),
        (
            "anchor 3 — consumed by the field's long form with a `.clone()`; the construction \
             line itself is NOT an anchor (no inline `detail: format!(`)",
            &[
                "fn f(e: Error) -> ServiceError {",
                "    let detail = format!(\"boom: {e}\");",
                "    ServiceError::PermanentKeyspaceRefusal {",
                "        refusal: KeyspaceRefusal::WrongType,",
                "        detail: detail.clone(),",
                "    }",
                "}",
            ],
            &[1],
        ),
    ];

    for (what, sample, expected) in positives {
        let found = anchor_line_indices(sample);
        assert_eq!(
            found,
            expected.to_vec(),
            "{what}: the scan must anchor on line(s) {expected:?} and nowhere else, or this wrap \
             spelling is invisible to every scan in this file — no violation reported, and the \
             ALLOWLIST row covering it silently goes stale. Snippet:\n{}",
            sample.join("\n")
        );
    }

    let negatives: &[(&str, &[&str])] = &[
        (
            "a match arm PATTERN, not a construction",
            &[
                "fn f(err: ServiceError) -> usize {",
                "    match err {",
                "        ServiceError::Internal(msg) => msg.len(),",
                "        _ => 0,",
                "    }",
                "}",
            ],
        ),
        (
            "a let-else PATTERN — and a `format!` binding it does NOT consume",
            &[
                "fn f(err: &ServiceError) -> String {",
                "    let detail = format!(\"unrelated diagnostic {err}\");",
                "    let ServiceError::Internal(detail) = err else {",
                "        return detail;",
                "    };",
                "    detail.clone()",
                "}",
            ],
        ),
        (
            "`Internal(_)`, a string literal, and a `.clone()` of something never bound to a \
             `format!`",
            &[
                "fn f(err: &ServiceError, msg: &String) -> ServiceError {",
                "    if matches!(err, ServiceError::Internal(_)) {",
                "        return ServiceError::Internal(\"static text\".into());",
                "    }",
                "    ServiceError::Internal(msg.clone())",
                "}",
            ],
        ),
        (
            "the item-3 false positive: an unrelated `let detail = format!` in a DIFFERENT \
             function from the `Internal(detail)` — `detail` and `msg` are generic binding \
             names, and file-level matching implicated every one of them",
            &[
                "fn unrelated(x: u8) {",
                "    let detail = format!(\"unrelated diagnostic {x}\");",
                "    tracing::debug!(\"{detail}\");",
                "}",
                "",
                "fn wrapper(detail: String) -> ServiceError {",
                "    ServiceError::Internal(detail)",
                "}",
            ],
        ),
        (
            "the consuming construction sits ABOVE the binding — it cannot be consuming it",
            &[
                "fn f(detail: String) -> ServiceError {",
                "    let first = ServiceError::Internal(detail);",
                "    let detail = format!(\"second: {first:?}\");",
                "    drop(detail);",
                "    first",
                "}",
            ],
        ),
        (
            "a refusal-variant construction with no inline `detail: format!(` and no hoisted \
             binding — `clone_service_error`'s arm and the test constructions from literals",
            &[
                "fn f(err: &ServiceError) -> ServiceError {",
                "    match err {",
                "        ServiceError::PermanentKeyspaceRefusal { refusal, detail } => {",
                "            ServiceError::PermanentKeyspaceRefusal {",
                "                refusal: *refusal,",
                "                detail: detail.clone(),",
                "            }",
                "        }",
                "        other => other.clone(),",
                "    }",
                "}",
            ],
        ),
        (
            "prose quoting the wrap in a comment",
            &[
                "fn f() {",
                "    // ServiceError::Internal(format!(\"this is documentation, not code\"))",
                "    // let detail = format!(\"nor is this\");",
                "}",
            ],
        ),
    ];

    for (what, sample) in negatives {
        let found = anchor_line_indices(sample);
        assert!(
            found.is_empty(),
            "{what}: the scan anchored on line(s) {found:?}, which is a false positive — it fails \
             the build on ordinary code, and a guard that cries wolf gets deleted rather than \
             fixed. Snippet:\n{}",
            sample.join("\n")
        );
    }
}

/// Sanity check on [`site_text`], the text every ALLOWLIST needle is matched
/// against, in both directions: a wrap's own message on a following line stays
/// in its text, and a neighbouring wrap's message inside the lookahead does
/// not — otherwise the neighbour's row would excuse both wraps.
#[test]
fn site_text_keeps_its_own_message_and_stops_at_the_next_wrap() {
    let sample: &[&str] = &[
        "fn f() -> Result<(), ServiceError> {",
        "    a.map_err(|e| ServiceError::Internal(format!(\"FIRST failed: {e}\")))?;",
        "    let _: () = trx",
        "        .sadd(k, v)",
        "        .map_err(|e| ServiceError::Internal(format!(\"SECOND failed: {e}\")))?;",
        "    Err(ServiceError::Internal(format!(",
        "        \"THIRD failed\"",
        "    )))",
        "}",
    ];
    assert_eq!(anchor_line_indices(sample), vec![1, 4, 5]);
    let first = site_text(sample, 1);
    assert!(first.contains("FIRST failed") && !first.contains("SECOND failed"));
    let second = site_text(sample, 4);
    assert!(second.contains("SECOND failed") && !second.contains("THIRD failed"));
    assert!(site_text(sample, 5).contains("THIRD failed"));
}

/// Sanity check on [`internal_wrap_consumed_ident`], the half of the
/// hoisted-binding check that decides whether `ServiceError::Internal(<x>)` is
/// a construction consuming `<x>` or a pattern binding it.
///
/// Specified separately from the composed predicate because its two directions
/// have different consequences: missing a construction drops a wrap out of the
/// scan (silent), while reading a pattern as a construction puts a binding name
/// into circulation that was never a wrap at all.
#[test]
fn internal_wrap_consumed_ident_reads_constructions_and_not_patterns() {
    for (line, expected) in [
        ("        ServiceError::Internal(detail)", Some("detail")),
        ("        ServiceError::Internal(detail),", Some("detail")),
        (
            "        ServiceError::Internal(detail.clone()),",
            Some("detail"),
        ),
        (
            "        ServiceError::Internal(detail.to_owned())",
            Some("detail"),
        ),
        ("        ServiceError::Internal(msg.into())", Some("msg")),
        (
            "        ServiceError::Internal(detail.to_string()),",
            Some("detail"),
        ),
        ("        ServiceError::Internal(msg) => msg.len(),", None),
        (
            "        let ServiceError::Internal(msg) = &err else {",
            None,
        ),
        ("        ServiceError::Internal(_) => {}", None),
        ("        ServiceError::Internal(\"literal\".into())", None),
        ("        ServiceError::Internal(e.details())", None),
        ("        ServiceError::Internal(format!(\"x: {e}\"))", None),
        ("        ServiceError::Unavailable(detail)", None),
    ] {
        assert_eq!(
            internal_wrap_consumed_ident(line),
            expected,
            "`internal_wrap_consumed_ident` misread {line:?}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// No RPC emits gRPC ABORTED but a chained UpdateLayer's predecessor refusal
// (b.27b, b.pav)
// ═══════════════════════════════════════════════════════════════════════════════
//
// The two surface mappers above differ only in-process: `map_transient_fred_err`
// returns `ServiceError::Aborted` (the retry loops' in-band signal) and
// `map_fred_err_for_grpc` returns `ServiceError::Unavailable`. Both reach the
// wire as `UNAVAILABLE`, because `From<ServiceError> for tonic::Status`
// (`src/error.rs`) maps an escaped `Aborted` there — the Kit client treats
// `ABORTED` as terminal. That single mapping is pinned by
// `error::tests::aborted_maps_to_unavailable_with_message_verbatim`; what it
// cannot see is a handler building a gRPC `ABORTED` status directly, which
// would bypass it. This walk closes that gap for every `src/**/*.rs` file.
//
// A `Code::Aborted` that is *matched* rather than *constructed* — an auth
// client classifying an upstream service's reply, a table enumerating every
// code — is not a producer and is not flagged.
//
// One producer is allowed (b.pav, SDD §18.1): a chained `UpdateLayer`'s
// predecessor refusal, `ServiceError::PredecessorNotCommitted`, which only a
// client that sent `chained_header` can receive. Its arm in `src/error.rs` is
// the single entry of [`ALLOWED_ABORTED_PRODUCERS`], and the walk fails if
// that entry matches anything but exactly one site.

/// `tonic::Status` constructors that take a `Code` as their first argument,
/// and the refusal-detail builders of `src/error.rs` that do
/// (`RefusalDetail::status`, `refusal_status`).
const STATUS_CODE_CTORS: &[&str] = &[
    "Status::new(",
    "Status::with_details(",
    "Status::with_metadata(",
    "Status::with_details_and_metadata(",
    ".status(",
    "refusal_status(",
];

/// The gRPC `ABORTED` producers the walk allows: `(file, arm)` — a flagged
/// line in `file` whose own line or nearest preceding non-blank line names
/// `arm`. Each entry must match exactly one site.
const ALLOWED_ABORTED_PRODUCERS: &[(&str, &str)] =
    &[("src/error.rs", "ServiceError::PredecessorNotCommitted")];

/// The [`ALLOWED_ABORTED_PRODUCERS`] entry the flagged line `idx` of `rel`
/// (comment-stripped `lines`) falls under, if any.
fn allowed_aborted_producer(rel: &str, lines: &[&str], idx: usize) -> Option<usize> {
    let arm_line = std::iter::once(lines[idx]).chain(
        lines[..idx]
            .iter()
            .rev()
            .copied()
            .find(|prev| !prev.trim().is_empty()),
    );
    let arm_lines: Vec<&str> = arm_line.collect();
    ALLOWED_ABORTED_PRODUCERS
        .iter()
        .position(|(file, arm)| *file == rel && arm_lines.iter().any(|line| line.contains(arm)))
}

/// Whether line `idx` (comment-stripped `lines`) builds a gRPC `ABORTED`
/// status: `Status::aborted(…)`, or `Code::Aborted` as the code argument of a
/// `Status` constructor — on the constructor's own line, or as the first
/// argument after a constructor that ends the previous non-blank line (the
/// rustfmt-wrapped shape).
fn constructs_aborted_status(lines: &[&str], idx: usize) -> bool {
    let line = lines[idx];
    if line.contains("Status::aborted(") {
        return true;
    }
    if !line.contains("Code::Aborted") {
        return false;
    }
    if STATUS_CODE_CTORS.iter().any(|ctor| line.contains(ctor)) {
        return true;
    }
    lines[..idx]
        .iter()
        .rev()
        .find(|prev| !prev.trim().is_empty())
        .is_some_and(|prev| {
            let prev = prev.trim_end();
            STATUS_CODE_CTORS.iter().any(|ctor| prev.ends_with(ctor))
        })
}

#[test]
fn no_source_file_constructs_a_grpc_aborted_status() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_root = manifest_dir.join("src");
    let mut files = Vec::new();
    collect_rs_files(&src_root, &mut files);
    assert!(
        files.len() > 50,
        "the walk found only {} `.rs` files under `{}` — a walk that sees nothing cannot \
         report a producer",
        files.len(),
        src_root.display()
    );

    let mut producers: Vec<String> = Vec::new();
    let mut allowed_hits = vec![0usize; ALLOWED_ABORTED_PRODUCERS.len()];
    for path in &files {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read `{}`: {e}", path.display()));
        let code: Vec<&str> = text.lines().map(strip_line_comment).collect();
        for (idx, raw_line) in text.lines().enumerate() {
            if constructs_aborted_status(&code, idx) {
                let rel = path.strip_prefix(&manifest_dir).unwrap_or(path);
                let rel_text = rel.display().to_string().replace('\\', "/");
                if let Some(entry) = allowed_aborted_producer(&rel_text, &code, idx) {
                    allowed_hits[entry] += 1;
                    continue;
                }
                producers.push(format!(
                    "  {}:{}\n    {}",
                    rel.display(),
                    idx + 1,
                    raw_line.trim()
                ));
            }
        }
    }

    for ((file, arm), hits) in ALLOWED_ABORTED_PRODUCERS.iter().zip(&allowed_hits) {
        assert_eq!(
            *hits, 1,
            "the allowed ABORTED producer `{arm}` in `{file}` must match exactly one site, \
             found {hits}: a stale or widened allowlist entry"
        );
    }

    assert!(
        producers.is_empty(),
        "these sites build a gRPC ABORTED status, which the Kit client does not retry. No RPC \
         may emit ABORTED except a chained UpdateLayer's predecessor refusal \
         (`ServiceError::PredecessorNotCommitted`): return `ServiceError::Unavailable` for a \
         transient condition (or let an in-loop `ServiceError::Aborted` escape — \
         `src/error.rs` maps it to UNAVAILABLE), or the code that genuinely fits a permanent \
         one:\n{}",
        producers.join("\n")
    );
}

#[test]
fn aborted_status_scan_flags_every_constructor_spelling_and_no_consumer() {
    // Sanity check on the predicate the walk above rests on: a predicate that
    // matched nothing would make it pass vacuously.
    let flagged: &[&[&str]] = &[
        &["        return Err(Status::aborted(msg));"],
        &["        tonic::Status::aborted(\"conflict\")"],
        &["        Status::new(Code::Aborted, msg)"],
        &["        tonic::Status::new(tonic::Code::Aborted, msg)"],
        &["        Status::with_details(Code::Aborted, msg, details)"],
        &["        Status::new(", "            tonic::Code::Aborted,"],
        &["        Status::new(", "", "            Code::Aborted,"],
        &["                detail.status(tonic::Code::Aborted, message)"],
        &["        refusal_status(Code::Aborted, msg, reason, [], None)"],
        &[
            "        refusal_status(",
            "            tonic::Code::Aborted,",
        ],
    ];
    for sample in flagged {
        let last = sample.len() - 1;
        assert!(
            constructs_aborted_status(sample, last),
            "the scan must flag the ABORTED construction in {sample:?}"
        );
    }

    let not_flagged: &[&[&str]] = &[
        // Matching an upstream reply's code is classification, not production.
        &["            tonic::Code::Aborted | tonic::Code::Internal => Err(AuthError::Unreachable),"],
        &["        Code::Aborted | Code::Unavailable => Some(Level::WARN),"],
        // An enumeration of every code.
        &["        Code::FailedPrecondition,", "        Code::Aborted,"],
        &["            ServiceError::Aborted(msg) => Status::unavailable(msg),"],
        &["        Status::new(Code::Unavailable, msg)"],
        &["            .status(code, \"too large\")"],
    ];
    for sample in not_flagged {
        let last = sample.len() - 1;
        assert!(
            !constructs_aborted_status(sample, last),
            "the scan must not flag the non-producer {sample:?}"
        );
    }
}
