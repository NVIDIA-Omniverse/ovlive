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

//! Session-checkpoint assembler — bytes-producer for the Epic 2 session /
//! layer-mapping DR path (SR-2.7).
//!
//! [`build_session_checkpoint`] snapshots the session-scoped Valkey state for a
//! single `session_id` and returns a [`SessionCheckpoint`] payload struct. It
//! is the counterpart to the recovery applier (sibling Task
//! `t2.fgj.7v.32`); together they form the durable round-trip for the four
//! session-scoped key families: `le:{sid}:sess`, `le:{sid}:layers`,
//! `le:{sid}:layer:{hex(name)}`, and the legacy + composite path-index
//! membership.
//!
//! # Three load-bearing properties
//!
//! - **Cross-slot non-atomicity.** In Cluster mode the path-index keys
//!   (`le:path:{hex(base_path)}*`) live in a different hash slot than the
//!   session-scoped keys (`le:{sid}:*`). This assembler does NOT see a single
//!   point-in-time snapshot across both families. The downstream applier
//!   tolerates this because (a) the recovery applier writes path-index first /
//!   session-scoped second with each step independently atomic (SR-2.7), (b)
//!   the asymmetric self-healing rationale in SR-2.7 holds at restore time
//!   even if the assembler captures a mid-update state, and (c) the
//!   orphan-sweep / `ListSessions` opportunistic-cleanup path tolerates
//!   ghost-vs-session inconsistency.
//!
//! - **Lifecycle-event triggered, not periodic.** Unlike the Epic-1
//!   layer-checkpoint assembler at [`crate::checkpoint::assembler`], this
//!   function has no leader-elected loop. Sibling Task `t2.fgj.7v.p9` calls
//!   it from lifecycle-event hooks (CreateSession / CreateLayer / DeleteLayer
//!   / DeleteSession).
//!
//! - **Atomic within the session slot, and it mints a build stamp.** The
//!   session-scoped read runs as ONE Lua script
//!   (`src/storage/scripts/session_checkpoint_read.lua`), so the layers-index
//!   membership and the per-layer mappings are observed at a single instant —
//!   a lifecycle event landing mid-read can no longer tear the payload. The
//!   same script `HINCRBY`s a monotone build stamp on
//!   `le:{sid}:ckpt`, atomically with the state it stamps, and `HSETNX`es an
//!   incarnation token on the same key that scopes the stamp to one key
//!   LIFETIME; the pair is what [`crate::checkpoint::session_writer`]
//!   publishes after its adapter call so a stale build can never durably
//!   clobber a fresher one — including a build minted before the key was lost
//!   and the counter restarted. The assembler is
//!   therefore NOT a pure read — it writes at most two hash fields plus a TTL
//!   refresh, on a key that belongs to no other subsystem. It still does not
//!   call the adapter and does not touch metrics; the writer chains
//!   serialization, the adapter put, and the publish around this call.
//!
use std::collections::HashMap;
use std::sync::LazyLock;

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, TransactionInterface};
use fred::types::scripts::Script;
use fred::types::Value;

use crate::checkpoint::session_payload::SessionCheckpoint;
use crate::checkpoint::store::CheckpointError;
use crate::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_DATA, POOL_LABEL_TXN};
use crate::storage::keys::{hex_decode, KeySchema};

/// Compiled Lua script backing [`build_session_checkpoint`]'s atomic
/// session-state read (b.443).
///
/// Collapses the old `HGETALL` → `SMEMBERS` → per-layer `GET` fan-out into
/// one round-trip that cannot be torn by a concurrent lifecycle event, and
/// mints the monotone build stamp in the same atomic block. See the
/// script's own header for the undeclared-keys justification and the
/// soundness argument the stamp carries.
///
/// The script lives in `src/storage/scripts/` (repo convention: that
/// directory is the home for every Lua script regardless of which module
/// owns the call site) while the `Script` handle is declared next to its
/// only caller — mirroring `STREAM_HIGH_WATER_PROBE_SCRIPT` in
/// `src/checkpoint/recovery.rs`.
///
/// `Script::from_lua` is infallible — it just hashes the source. The first
/// `evalsha_with_reload` call sends `SCRIPT LOAD` on `NOSCRIPT` before
/// retrying.
static SESSION_CHECKPOINT_READ_SCRIPT: LazyLock<Script> = LazyLock::new(|| {
    Script::from_lua(include_str!(
        "../storage/scripts/session_checkpoint_read.lua"
    ))
});

/// TTL applied to `le:{sid}:ckpt`
/// ([`KeySchema::session_checkpoint_state`]) on every touch by either
/// b.443 script — 24 hours.
///
/// The key is deliberately absent from `SessionStore::delete`'s teardown
/// transaction (it must outlive the session it names, so a repair can read
/// the watermark after a DELETE has landed), so a TTL is the only thing
/// that reaps it. Two bounds fix the value:
///
/// - **Lower bound — much longer than any in-flight dispatch.** The stamp
///   ordering is only meaningful while `seq` survives. If the key expired
///   between one build's mint and its adapter put, a concurrently-minted
///   stamp would restart from 1 and the stale build's higher stamp would
///   win. A dispatch operation is bounded by
///   `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` plus the adapter's request
///   timeout plus this module's retry ladder — seconds, not hours — and
///   the whole tracker is drained at shutdown. A day is ~4 orders of
///   magnitude of headroom.
/// - **Upper bound — bounded footprint.** One small hash per session that
///   has seen a lifecycle event, including sessions already deleted (the
///   DELETE dispatch mints a stamp too). A day caps the resident set at
///   roughly one day of session churn; for a session still in use the key
///   is refreshed on every event, so an active session never loses its
///   ordering anchor no matter how long it lives.
///
/// Expiry — and key loss generally — is safe because of the INCARNATION
/// token, not because `seq` and `pub` share a key.
///
/// The "they always reset together" reading was the original argument and it
/// is not sufficient: a build minted BEFORE the key was lost is still holding
/// a stamp drawn from the old counter, so when it finally PUTs and publishes,
/// its stamp (say 60) outranks the fresh watermark (1). The stale bytes become
/// durable and `pub = 60` then silently swallows the next 59 legitimate builds.
/// Resetting together only helps if nothing survives the reset; an in-flight
/// build does.
///
/// What actually makes it safe is that every mint reads back the key's
/// `inc` field — a random token claimed with `HSETNX` by whichever mint brought
/// the key into existence — and every publish carries that token. A key that
/// was lost and re-created has a different token (or none), so
/// `session_checkpoint_publish.lua` refuses the pre-loss build outright,
/// writing nothing, and the writer repairs by rebuilding from live state. See
/// both scripts' headers for the full argument.
pub(crate) const SESSION_CHECKPOINT_STATE_TTL_MS: u64 = 24 * 60 * 60 * 1000;

/// Map a `ServiceError` from the storage layer into a `CheckpointError`.
///
/// This helper is called only with `ServiceError::Internal(...)` constructed
/// locally at the two call sites in this module (the read path here uses a Lua
/// script and a raw fred `SISMEMBER` rather than the storage layer's
/// session-store API, and classifies a missing session via the script's
/// `present` flag, so no `NotFound` is ever synthesised on this path). The
/// mapping is therefore unconditional:
/// every input is treated as `Transient` — backend blips (timeouts, IO,
/// cluster-redirect storms) heal on retry.
///
/// The sibling helper in `src/checkpoint/assembler.rs` intentionally retains a
/// `NotFound` arm: its caller is the storage-layer session-store API, which
/// DOES surface `ServiceError::NotFound`. That arm is reachable there and is
/// out of scope for this helper.
///
/// Kept local to this module rather than lifted into a shared
/// `src/checkpoint/mod.rs` location to keep the change scope of this Subtask
/// narrow (`t3.fgj.7v.du.zc`'s "Do NOT edit assembler.rs" guard). A future
/// refactor that lifts both copies into a shared helper (with the assembler's
/// richer two-arm shape) is welcome but out-of-scope here.
fn service_err_to_checkpoint_err(err: crate::error::ServiceError) -> CheckpointError {
    debug_assert!(
        matches!(err, crate::error::ServiceError::Internal(_)),
        "session-checkpoint helper only receives ServiceError::Internal in this module; \
         got {err:?}"
    );
    CheckpointError::Transient {
        source: format!("{err}").into(),
    }
}

/// A completed session-checkpoint build: the payload plus the monotone
/// stamp that was minted atomically with the state read it reflects.
///
/// The two travel together because they are only meaningful together — the
/// stamp is the proof of freshness for exactly these bytes. The writer
/// serializes `checkpoint`, PUTs it, and then publishes `stamp` via
/// `session_checkpoint_publish.lua`; if the publish is refused, the bytes it
/// just made durable are known-stale and get repaired.
#[derive(Debug)]
pub(crate) struct SessionBuild {
    /// The assembled payload, ready for `to_bytes()`.
    pub(crate) checkpoint: SessionCheckpoint,
    /// Monotone build stamp from `HINCRBY le:{sid}:ckpt seq 1`, minted
    /// inside the same atomic script as the read.
    pub(crate) stamp: u64,
    /// The incarnation token that was live on `le:{sid}:ckpt` at mint time.
    ///
    /// Scopes [`Self::stamp`] to one key LIFETIME. `stamp` is only monotone
    /// while the key survives; if it is lost (FLUSH, failover to a lagging
    /// replica, out-of-band DEL, TTL expiry on an idle session) the counter
    /// restarts and an in-flight build's higher stamp would beat the fresh
    /// one. The publish script refuses any operation whose incarnation is
    /// not the token currently on the key, so the pair — not the stamp —
    /// is the unit of comparability.
    pub(crate) incarnation: String,
}

/// Raw reply of `session_checkpoint_read.lua`, decoded but not yet
/// validated. See the script header for the wire contract.
struct RawSessionState {
    stamp: u64,
    /// `false` when `le:{sid}:sess` was empty — the session is gone.
    /// Every other field is empty in that case.
    present: bool,
    meta: HashMap<String, String>,
    /// Hex-encoded layers-index members that had a live mapping key.
    hex_names: Vec<String>,
    /// Mapping values, positionally aligned with `hex_names`.
    mapping_values: Vec<String>,
    /// Count of layers-index members whose mapping key was absent.
    dropped: u64,
    /// Effective `inc` field on `le:{sid}:ckpt` — this mint's own candidate
    /// only when it was the mint that created the key.
    incarnation: String,
}

/// Mint a fresh candidate incarnation token for `le:{sid}:ckpt`.
///
/// 128 bits of OS randomness, rendered as 32 lowercase hex characters. It is a
/// CANDIDATE: `session_checkpoint_read.lua` installs it with `HSETNX`, so it
/// only becomes the key's incarnation when this mint is the one that brought
/// the key into existence. Every other mint gets the incumbent back.
///
/// Randomness (rather than, say, a replica id plus a clock) is what makes the
/// token safe under the failure it exists for: after a keyspace wipe there is
/// no surviving state to derive a *distinct* value from, and two replicas
/// re-creating the key must not be able to agree by accident. 128 bits from the
/// OS CSPRNG makes a collision — which would silently re-admit exactly the
/// stale publish the token exists to refuse — not worth reasoning about.
/// Follows `auth::cache::generate_hmac_seed`'s panic-on-CSPRNG-failure
/// posture: a CSPRNG that cannot produce 16 bytes is not a condition to paper
/// over with a weaker token.
fn mint_incarnation_token() -> String {
    use rand::TryRngCore;
    let mut bytes = [0u8; 16];
    rand::rngs::OsRng
        .try_fill_bytes(&mut bytes)
        .expect("failed to read 16 bytes from OS CSPRNG for a session-checkpoint incarnation");
    let mut out = String::with_capacity(32);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Build a [`SessionCheckpoint`] for `session_id` from one atomic read of
/// the session-scoped key families, plus the live path-index sanity-check.
///
/// # Return shape
///
/// - `Ok(None)` — session does not exist (the script saw an empty
///   `le:{sid}:sess`). The caller MUST treat this as "nothing to do" and
///   exit silently. It is deliberately NOT converted into a checkpoint
///   delete: an absent session-meta hash is also what selective key loss
///   looks like, and tombstoning on it would destroy the very artifact
///   recovery needs.
/// - `Ok(Some(build))` — assembled successfully. The
///   [`SessionCheckpoint::layers_index`] set equals
///   [`SessionCheckpoint::layer_mappings`]`.keys()`.
/// - `Err(CheckpointError::Transient)` — backend (Valkey) failure on the
///   script or the path-index check.
/// - `Err(CheckpointError::Permanent)` — wire-shape violation: corrupt
///   non-numeric `created_at` (when non-empty), corrupt non-numeric
///   `object_id` in any layer mapping, missing `base_path` on a non-empty
///   session-meta hash, an embedded NUL in a decoded layer name, or a
///   script reply that does not match the documented shape.
///
/// # Reads issued
///
/// 1. `EVALSHA session_checkpoint_read.lua` — one round-trip that claims the
///    incarnation (`HSETNX le:{sid}:ckpt inc <token>` + read-back) and mints
///    the build stamp (`HINCRBY le:{sid}:ckpt seq 1` + `PEXPIRE`), `HGETALL`s
///    the session meta, `SMEMBERS`es the layers index, and `GET`s every
///    member's mapping key. All of it observes ONE instant, so the
///    layers-index membership and the mappings cannot disagree because of a
///    concurrent CreateLayer / DeleteLayer. This replaces a 2 + N
///    round-trip sequence with 1 and removes the torn-payload window that
///    made a checkpoint silently unrestorable (b.443).
/// 2. `SISMEMBER le:path:{hex(base_path)} {session_id}` — sanity-check the
///    legacy session-id-only path-index entry; absence emits a warn and
///    falls through (the applier's restore is authoritative; orphan-sweep
///    self-heals). Deliberately NOT folded into the script: the path-index
///    keys hash under `{hex(base_path)}`, a different Cluster slot, so no
///    single script can touch both families.
///
/// Both run against `client.pool` — the multiplexing data pool, matching
/// Epic-1's [`crate::checkpoint::assembler`] pattern (no `acquire_txn`, no
/// `maintenance_pool`).
///
/// # Layers-index members without a mapping
///
/// The script filters them out, so `layers_index` and `layer_mappings` stay
/// consistent with each other. This is no longer a read race (atomicity
/// rules that out) but a real keyspace state: `LayerStore::delete_layer`
/// DELs the mapping several steps before it SREMs the index member, so a
/// build that lands in that window legitimately sees an index member with
/// no mapping. Dropping it is correct — a checkpoint naming a layer with no
/// `object_id` would restore an unreachable layer — and the count is logged
/// at debug for diagnosis.
pub(crate) async fn build_session_checkpoint(
    client: &ValkeyClient,
    keys: &KeySchema,
    session_id: &str,
) -> Result<Option<SessionBuild>, CheckpointError> {
    // ── Step 1: atomic session-scoped read + stamp mint ─────────────────────
    let raw = read_session_state(client, keys, session_id).await?;
    if !raw.present {
        // Absent session — the script short-circuited after the mint and
        // issued no further reads.
        return Ok(None);
    }
    if raw.dropped > 0 {
        tracing::debug!(
            session_id = %session_id,
            dropped = raw.dropped,
            "session-checkpoint assembler: layers-index members had no mapping key and were \
             excluded from the payload (expected mid-`delete_layer`; persistent counts here \
             indicate orphaned index members)"
        );
    }
    let stamp = raw.stamp;
    let incarnation = raw.incarnation;
    let meta = raw.meta;

    // ── Step 2: extract session-meta fields ─────────────────────────────────
    // Missing string fields fall to empty (mirrors SessionStore::list's
    // tolerance for in-flight writes between EXEC and HSET, b.91j C-7).
    let name = meta.get("name").cloned().unwrap_or_default();
    let created_by = meta.get("created_by").cloned().unwrap_or_default();
    // `base_path` missing on a non-empty session-meta hash is a permanent
    // invariant violation — the EXEC + HSET sequence in `SessionStore::create`
    // writes `base_path` synchronously with `name` / `created_by` /
    // `created_at`, so a non-empty meta without `base_path` indicates
    // corruption or a foreign writer.
    let base_path = meta
        .get("base_path")
        .cloned()
        .ok_or_else(|| CheckpointError::Permanent {
            source: format!("session_meta for {session_id} is non-empty but missing base_path")
                .into(),
        })?;
    // `created_at` extraction is intentionally STRICTER than
    // `SessionStore::list` (which uses `.parse::<u64>().ok().unwrap_or(0)`)
    // because durable checkpoint bytes must not silently encode corrupt
    // values — a runtime `ListSessions` can survive a momentary `0`, but a
    // checkpoint round-trip that emits `0` for what was a real microsecond
    // timestamp is silent data corruption. A MISSING `created_at` still
    // folds to `0` (matches the EXEC-before-HSET tolerance window).
    let created_at: u64 = match meta.get("created_at") {
        None => 0,
        Some(raw) => raw.parse::<u64>().map_err(|e| CheckpointError::Permanent {
            source: format!(
                "session_meta for {session_id} has non-numeric created_at '{raw}': {e}"
            )
            .into(),
        })?,
    };

    // ── Step 3: decode the layers-index members ─────────────────────────────
    // `raw.hex_names` and `raw.mapping_values` arrive positionally aligned
    // from the script — it emits a pair only for members whose mapping key
    // was live, so there is no `Option` to unwrap and no per-member GET to
    // issue here.
    let hex_names = raw.hex_names;
    let mapping_values = raw.mapping_values;

    // Decode hex layer names; tolerate empty-decode-from-non-empty (mirrors
    // `LayerStore::list_layer_names` skip-and-warn semantics on the same
    // codec). Reject NUL-bearing decoded names as Permanent corruption —
    // `validate_session_name` rejects NUL upstream at the CreateSession /
    // CreateLayer chokepoints, so a NUL-bearing layer name on disk only
    // arises from corruption or an adversarial write. We surface the
    // corruption to the operator rather than silently encoding a torn
    // layer name into `layer_mappings` / `layers_index` (the recovered
    // hex-encoded `layers_index` member would round-trip the corrupt
    // bytes verbatim).
    let mut decoded_pairs: Vec<(String, String)> = Vec::with_capacity(hex_names.len());
    for (hex_name, mapping_value) in hex_names.iter().zip(mapping_values.iter()) {
        let decoded = hex_decode(hex_name);
        if hex_name.is_empty() {
            // Skip-silently — the layers-index member was itself empty.
            // Matches `LayerStore::list_layer_names`'s implicit tolerance
            // (it maps empty -> empty via `hex_decode`).
            continue;
        }
        if decoded.is_empty() {
            // Non-empty hex that failed to decode to a non-empty string.
            // `hex_decode` already emitted a tracing::warn for the
            // underlying failure (invalid hex / non-UTF-8); we skip the
            // member here rather than propagating an empty layer name into
            // the payload, matching the tolerance pattern used by
            // `LayerStore::list_layer_names`.
            tracing::warn!(
                session_id = %session_id,
                hex_layer_name = %hex_name,
                "session-checkpoint assembler: layers_index member hex_decode failed; skipping"
            );
            continue;
        }
        if decoded.contains('\0') {
            // Render the NUL visibly so an operator can see WHERE the
            // corruption sits in the decoded byte string. `\0` is the
            // canonical 2-character rendering used elsewhere in the
            // codebase's error strings.
            let display = decoded.replace('\0', "\\0");
            return Err(CheckpointError::Permanent {
                source: format!(
                    "session-checkpoint assembler: decoded layer name in le:{{{session_id}}}:layers \
                     contains embedded NUL: '{display}'"
                )
                .into(),
            });
        }
        decoded_pairs.push((decoded, mapping_value.clone()));
    }

    // ── Step 4: populate the SessionCheckpoint ──────────────────────────────
    let mut checkpoint = SessionCheckpoint {
        schema_version: crate::checkpoint::session_payload::SESSION_CHECKPOINT_SCHEMA_VERSION,
        session_id: session_id.to_string(),
        name: name.clone(),
        base_path: base_path.clone(),
        created_by,
        created_at,
        layer_mappings: std::collections::BTreeMap::new(),
        layers_index: std::collections::BTreeSet::new(),
        path_name_members: std::collections::BTreeSet::new(),
        path_index_session_id_member: session_id.to_string(),
    };

    for (name, raw_value) in &decoded_pairs {
        let oid = raw_value
            .parse::<u64>()
            .map_err(|e| CheckpointError::Permanent {
                source: format!(
                    "session-checkpoint assembler: layer_mapping for ({session_id}, '{name}') \
                     is not a valid u64 ('{raw_value}'): {e}"
                )
                .into(),
            })?;
        checkpoint.layer_mappings.insert(name.clone(), oid);
        checkpoint.layers_index.insert(name.clone());
    }

    // Emit exactly ONE `path_names_index` composite per session — the
    // `session_name\0session_id` composite that mirrors what
    // `SessionStore::create_once` (`src/storage/session.rs:308`) writes on
    // the live path. The composite is emitted unconditionally (irrespective
    // of layer count) so the recovered keyspace is byte-equivalent to a
    // freshly-created session: `SessionStore::list` walks this set to
    // enumerate sessions, `create_once`'s duplicate-name check iterates it
    // to enforce per-base_path uniqueness, and `DeleteSession` SREMs the
    // session-name composite by `KeySchema::path_name_member(name,
    // session_id)`. Emitting a per-layer composite (the pre-fix shape) broke
    // all three invariants. Use the sanctioned wrapper — never a direct
    // `KeySchema::path_name_member` call from this Subtask's source.
    // NUL-discipline is enforced upstream by `validate_session_name` at
    // CreateSession time, so a `name` extracted from `session_meta` cannot
    // carry an embedded NUL.
    checkpoint.add_path_name_member(&name, session_id);

    // ── Step 5: sanity-check the legacy session-id path-index membership ────
    // Stays OUTSIDE the atomic script: the path-index keys hash under
    // `{hex(base_path)}`, a different Cluster slot from the `{session_id}`
    // family the script owns, so no single script can read both. The
    // resulting cross-slot non-atomicity is the long-standing, tolerated
    // property documented at the top of this module — the applier's restore
    // is authoritative for `le:path:{hex(base_path)}`, so a `false` here
    // doesn't block the checkpoint. We warn and still populate the field;
    // the orphan-sweep self-heals any divergence.
    let path_index_key = keys.path_index(&base_path);
    let in_path_index: bool = client
        .pool
        .sismember(&path_index_key, session_id)
        .await
        .map_err(|e| {
            FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(&e);
            service_err_to_checkpoint_err(crate::error::ServiceError::Internal(format!(
                "SISMEMBER path_index failed for {session_id}: {e}"
            )))
        })?;
    if !in_path_index {
        tracing::warn!(
            session_id = %session_id,
            base_path = %base_path,
            "session-checkpoint assembler: session_id missing from live path_index; \
             populating payload anyway (applier restore is authoritative; orphan-sweep self-heals)"
        );
    }

    Ok(Some(SessionBuild {
        checkpoint,
        stamp,
        incarnation,
    }))
}

/// Run `session_checkpoint_read.lua` and decode its reply.
///
/// Returns the raw, unvalidated state — field extraction, hex decoding and
/// `u64` parsing all happen in [`build_session_checkpoint`], which owns the
/// error classification for each.
///
/// # Reply-shape errors are `Permanent`
///
/// A reply that does not match the script's documented shape means the
/// script and this decoder have diverged — a code bug, not a backend blip,
/// and retrying it forever would just spin. Transport / Lua-runtime errors
/// from fred stay `Transient` (the established classification for every
/// fred error on this path); a genuine script bug surfacing as `Transient`
/// is the acceptable cost of not being able to tell the two apart from the
/// error alone.
async fn read_session_state(
    client: &ValkeyClient,
    keys: &KeySchema,
    session_id: &str,
) -> Result<RawSessionState, CheckpointError> {
    // The layers-index members ARE the hex-encoded layer names, so the
    // script builds each mapping key as `prefix .. member`. Derive the
    // prefix from `KeySchema` itself — `layer_mapping(sid, "")` appends
    // `hex_encode("") == ""` — rather than re-spelling the literal here, so
    // the script keeps tracking the key schema if it ever changes.
    let mapping_prefix = keys.layer_mapping(session_id, "");
    debug_assert!(
        mapping_prefix.ends_with(":layer:"),
        "layer_mapping prefix derivation broke: got '{mapping_prefix}'"
    );

    let script_keys = vec![
        keys.session_checkpoint_state(session_id),
        keys.session_meta(session_id),
        keys.layers_index(session_id),
    ];
    // ARGV[3] is a CANDIDATE incarnation token: the script `HSETNX`es it and
    // returns whatever is actually on the key, so a fresh token is minted on
    // every call but only adopted by the call that created the key.
    let script_argv = vec![
        SESSION_CHECKPOINT_STATE_TTL_MS.to_string(),
        mapping_prefix,
        mint_incarnation_token(),
    ];

    // b.zt3 re-send audit: fred keeps its default write attempts here, so a lost
    // connection can run this twice: the second run mints a higher stamp over a
    // later read, and its reply is the one received, so a stamp is skipped and
    // the publish watermark's argument still holds (script header).
    let reply: Value = SESSION_CHECKPOINT_READ_SCRIPT
        .evalsha_with_reload(client.pool.next(), script_keys, script_argv)
        .await
        .map_err(|e| {
            FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(&e);
            service_err_to_checkpoint_err(crate::error::ServiceError::Internal(format!(
                "EVALSHA session_checkpoint_read failed for {session_id}: {e}"
            )))
        })?;

    let parts = reply.into_array();
    if parts.len() != 7 {
        return Err(CheckpointError::Permanent {
            source: format!(
                "session-checkpoint assembler: session_checkpoint_read.lua returned {} elements \
                 for {session_id}, expected 7",
                parts.len()
            )
            .into(),
        });
    }
    let mut parts = parts.into_iter();
    let mut take = |field: &str| -> Result<Value, CheckpointError> {
        parts.next().ok_or_else(|| CheckpointError::Permanent {
            source: format!(
                "session-checkpoint assembler: session_checkpoint_read.lua reply missing \
                 '{field}' for {session_id}"
            )
            .into(),
        })
    };

    let stamp = scalar_u64(take("stamp")?, "stamp", session_id)?;
    let present = scalar_u64(take("present")?, "present", session_id)? == 1;
    let meta_flat = string_vec(take("sess_fields")?, "sess_fields", session_id)?;
    let hex_names = string_vec(take("hex_names")?, "hex_names", session_id)?;
    let mapping_values = string_vec(take("mapping_values")?, "mapping_values", session_id)?;
    let dropped = scalar_u64(take("dropped")?, "dropped", session_id)?;
    // Non-string / absent here is a wire-shape violation, not a keyspace
    // state: the script `HSETNX`es `inc` before it mints, so an atomic reply
    // always carries a token.
    let incarnation =
        take("incarnation")?
            .into_string()
            .ok_or_else(|| CheckpointError::Permanent {
                source: format!(
                    "session-checkpoint assembler: session_checkpoint_read.lua returned a \
                     non-UTF-8 or absent 'incarnation' for {session_id}"
                )
                .into(),
            })?;

    if hex_names.len() != mapping_values.len() {
        return Err(CheckpointError::Permanent {
            source: format!(
                "session-checkpoint assembler: session_checkpoint_read.lua returned {} layer \
                 names but {} mapping values for {session_id}; the script emits them pairwise",
                hex_names.len(),
                mapping_values.len()
            )
            .into(),
        });
    }
    if meta_flat.len() % 2 != 0 {
        return Err(CheckpointError::Permanent {
            source: format!(
                "session-checkpoint assembler: session_checkpoint_read.lua returned an odd-length \
                 ({}) session-meta field array for {session_id}",
                meta_flat.len()
            )
            .into(),
        });
    }
    let mut meta: HashMap<String, String> = HashMap::with_capacity(meta_flat.len() / 2);
    let mut flat = meta_flat.into_iter();
    while let (Some(field), Some(value)) = (flat.next(), flat.next()) {
        meta.insert(field, value);
    }

    Ok(RawSessionState {
        stamp,
        present,
        meta,
        hex_names,
        mapping_values,
        dropped,
        incarnation,
    })
}

/// Decode one scalar `u64` element of the script reply.
fn scalar_u64(value: Value, field: &str, session_id: &str) -> Result<u64, CheckpointError> {
    value.as_u64().ok_or_else(|| CheckpointError::Permanent {
        source: format!(
            "session-checkpoint assembler: session_checkpoint_read.lua returned a non-integer \
             '{field}' for {session_id}"
        )
        .into(),
    })
}

/// Decode one array-of-strings element of the script reply.
///
/// A non-UTF-8 member is `Permanent`, not `Transient`: these are hex layer
/// names, decimal object ids, and session-meta values written by validated
/// code paths, so raw bytes that are not UTF-8 mean corruption or a foreign
/// writer — retrying cannot fix it, and encoding it into a durable
/// checkpoint would launder the corruption.
fn string_vec(value: Value, field: &str, session_id: &str) -> Result<Vec<String>, CheckpointError> {
    value
        .into_array()
        .into_iter()
        .map(|v| {
            v.into_string().ok_or_else(|| CheckpointError::Permanent {
                source: format!(
                    "session-checkpoint assembler: session_checkpoint_read.lua returned a \
                     non-UTF-8 or non-scalar member in '{field}' for {session_id}"
                )
                .into(),
            })
        })
        .collect()
}

/// Mint a build stamp — and read back the incarnation it is scoped to — for
/// `session_id`, without assembling a payload.
///
/// The pair is returned together because the stamp is meaningless alone: it
/// orders operations only within one lifetime of `le:{sid}:ckpt`, and the
/// incarnation names that lifetime. See [`SessionBuild::incarnation`].
///
/// The DELETE dispatch path needs an ordered stamp — its adapter call has
/// to be comparable against concurrent writes on the publish watermark —
/// but has no payload to build. It calls the same read script, which mints
/// the stamp first and then short-circuits on the (already deleted) session
/// meta, so the read half costs nothing. Reusing the script rather than
/// issuing a bare `HINCRBY` + `PEXPIRE` keeps the mint and the TTL refresh
/// atomic: a crash between two separate commands would leave `le:{sid}:ckpt`
/// with no TTL, and nothing else would ever reap it.
pub(crate) async fn mint_session_checkpoint_stamp(
    client: &ValkeyClient,
    keys: &KeySchema,
    session_id: &str,
) -> Result<(u64, String), CheckpointError> {
    let raw = read_session_state(client, keys, session_id).await?;
    Ok((raw.stamp, raw.incarnation))
}

// ───────────────────────────────────────────────────────────────────────────
// Session-recovery applier (SR-2.7)
// ───────────────────────────────────────────────────────────────────────────

/// Identifies which half of [`apply_session_checkpoint`]'s two-step protocol
/// a fault-injection entry targets. Per-step keying is the contract that
/// Task 7's integration tests (`t3.fgj.7v.o7.6f` + `t3.fgj.7v.o7.xt`) rely
/// on: AC #3 exercises Step B injection, AC #4 exercises Step A injection
/// and asserts cross-step independence.
///
/// **Test-only.** Production code never names this enum — release builds
/// compile without the `test-checkpoint-fault-injection` feature, leaving
/// the entire fault-injection plumbing absent from the binary.
#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApplyStep {
    /// Path-index MULTI/EXEC on the `{hex(base_path)}` hash slot.
    StepA,
    /// Session-scoped MULTI/EXEC on the `{session_id}` hash slot.
    StepB,
}

/// Process-global per-step fault-injection queues for
/// [`apply_session_checkpoint`].
///
/// Consulted before executing each step; if the matching queue has an entry,
/// the applier pops it and short-circuits the step with the popped error.
/// An empty queue lets the step run normally. Tests must explicitly install
/// queues via [`set_apply_step_fault_injector_for_test`] — initial state is
/// empty.
///
/// The applier is a free function (not a method on a struct), so the seam
/// must be process-global rather than instance-bound. Tests use unique
/// `session_id`s via `TestValkey::setup()` for keyspace isolation; the
/// fault-injection queues are reset by callers (or by the
/// [`clear_apply_step_fault_injectors_for_test`] helper) at the top of each
/// test that installs them.
///
/// **Test-only.** Gated behind `#[cfg(any(test, feature =
/// "test-checkpoint-fault-injection"))]`.
#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
static APPLY_STEP_FAULT_INJECTORS: std::sync::OnceLock<
    std::sync::Mutex<
        std::collections::HashMap<ApplyStep, std::collections::VecDeque<CheckpointError>>,
    >,
> = std::sync::OnceLock::new();

/// Install a queue of synthetic [`CheckpointError`]s for the chosen
/// [`ApplyStep`], consulted before [`apply_session_checkpoint`] executes
/// that step. Each call to `apply_session_checkpoint` pops one entry from
/// the matching queue (if any); a popped entry short-circuits the step.
///
/// Idempotent — calling twice for the same step REPLACES the previous
/// queue. Use [`clear_apply_step_fault_injectors_for_test`] to drain both
/// step queues at the top of a test (recommended to avoid cross-test
/// bleed when tests run in the same process).
///
/// **Test-only.** Production code never invokes this — it is gated behind
/// `#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]` so
/// release builds never see the symbol.
#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
pub fn set_apply_step_fault_injector_for_test(
    step: ApplyStep,
    queue: std::collections::VecDeque<CheckpointError>,
) {
    let map = APPLY_STEP_FAULT_INJECTORS
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut guard = map
        .lock()
        .expect("APPLY_STEP_FAULT_INJECTORS lock poisoned");
    guard.insert(step, queue);
}

/// Drain BOTH [`ApplyStep::StepA`] and [`ApplyStep::StepB`] fault-injection
/// queues. Recommended at the top of any test that installs an injector,
/// so a previous test's leftover queue does not interfere.
///
/// **Test-only.** Same feature-gate as
/// [`set_apply_step_fault_injector_for_test`].
#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
pub fn clear_apply_step_fault_injectors_for_test() {
    if let Some(map) = APPLY_STEP_FAULT_INJECTORS.get() {
        let mut guard = map
            .lock()
            .expect("APPLY_STEP_FAULT_INJECTORS lock poisoned");
        guard.clear();
    }
}

/// Consult the matching per-step queue and pop one entry if present.
/// Returns `Some(err)` if the step should short-circuit; `None` if the
/// step should proceed normally.
#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
fn pop_apply_step_fault(step: ApplyStep) -> Option<CheckpointError> {
    let map = APPLY_STEP_FAULT_INJECTORS.get()?;
    let mut guard = map
        .lock()
        .expect("APPLY_STEP_FAULT_INJECTORS lock poisoned");
    let queue = guard.get_mut(&step)?;
    queue.pop_front()
}

/// Apply a [`SessionCheckpoint`] to live Valkey, materializing the session
/// so the next `le:{sid}:sess` access finds it present (SR-2.7).
///
/// # Two-step restore protocol — path-index FIRST, session-scoped SECOND
///
/// In Cluster mode the path-index keys (`le:path:{hex(base_path)}` and
/// `le:path:{hex(base_path)}:names`) hash under `{hex(base_path)}` while
/// the session-scoped keys (`le:{sid}:sess`, `le:{sid}:layers`,
/// `le:{sid}:layer:{hex(name)}`) hash under `{session_id}`. They CANNOT
/// live in one MULTI/EXEC, so the applier splits into two independently-
/// atomic transactions:
///
/// 1. **Step A** — SADD the path-index set + the composite path-names set
///    on the `{hex(base_path)}` slot. Atomic across both sets.
/// 2. **Step B** — HSET session-meta, SET each layer-mapping, SADD the
///    layers-index on the `{session_id}` slot. Atomic across all three
///    key families.
///
/// **Order is load-bearing per SR-2.7. Do not reorder without a spec change.**
///
/// The ordering exploits an asymmetric self-healing property: if Step A
/// commits and Step B fails (or the process crashes between), the next
/// `ListSessions` opportunistic-cleanup walk (SDD §7.13) sees a path-index
/// member pointing at a non-existent session and removes the ghost member.
/// The reverse — session-scoped keys present without a path-index entry —
/// would be a "ghost-but-functional" session with NO self-healing path:
/// the session exists for direct lookups but is invisible to
/// `ListSessions`. Reordering would re-open this unrecoverable partial-
/// failure mode.
///
/// # Idempotency
///
/// Re-applying the same checkpoint after a Step-B-only failure converges
/// to the same final state. SADD / HSET / SET are all idempotent at the
/// per-key level; no DEL / SREM / HDEL is issued anywhere in this
/// function (and adding one would break the idempotency contract — a
/// destructive pre-clean would erase the Step A path-index half if a
/// caller retried after Step A succeeded but Step B failed). The applier
/// is purely additive.
///
/// # Cross-replica race
///
/// The applier acquires NO admission lock — two replicas concurrently
/// restoring the same session race benignly because every write is
/// idempotent. The Task-6 consumer (`t2.fgj.7v.fp`) is the right surface
/// for any per-`session_id` admission-control gate.
///
/// # Session-meta field set
///
/// The four HSET fields written to `le:{sid}:sess` are:
/// `name`, `base_path`, `created_by`, `created_at`. **This set is a
/// load-bearing co-location with `SessionStore::create_once` at
/// `src/storage/session.rs:386-394`.** A recovered session is
/// byte-identical to one created by `CreateSession`. Adding a fifth
/// field to either side without updating the other is a contract
/// violation — the recovered session would diverge from the
/// freshly-created shape on that field.
///
/// # Connection pool
///
/// Each step acquires an exclusive `txn_pool` connection via
/// [`ValkeyClient::acquire_txn_for`], pinned to the slot that step's
/// transaction actually runs on: Step A pins on
/// `keys.path_index(&checkpoint.base_path)` (`{hex(base_path)}`), Step B on
/// `keys.session_meta(session_id)` (`{session_id}`). Each pin key is hoisted
/// above its checkout and reused as that step's first queued key, so the
/// checkout hygiene probe and the slot fred derives for `EXEC` are the same
/// string by construction — without the pin, `DISCARD` / `UNWATCH` carry no key
/// and a cluster router sends them to an arbitrary primary, so the verdict
/// would describe a connection the transaction never touches (b.4t3, the
/// call-site half of b.7td). The token is held for the duration of the step and
/// dropped before the next step's acquisition (independent transactions, no
/// shared state). This mirrors `SessionStore::create_once`
/// (`src/storage/session.rs:309`) and `create_layer`'s exclusive-connection
/// requirement for MULTI/EXEC — the multiplexed `pool.next()` would
/// interleave commands from other callers between MULTI and EXEC.
///
/// # Error classification
///
/// Step A and Step B errors both map to [`CheckpointError::Transient`]
/// with a message that names the step (`"Step A path-index MULTI/EXEC
/// failed: ..."` / `"Step B session-scoped MULTI/EXEC failed: ..."`) so
/// an operator log can locate the partial-failure boundary.
///
/// **The blanket `Transient` wrap is deliberate — do not route these arms
/// through the storage lane's `is_transient_fred_error` /
/// `map_fred_err_for_grpc` classifier, and do not add a local retry loop**
/// (b.4t3). Four reasons, in order of weight:
///
/// 1. **Re-drive is always safe.** Neither step holds a `WATCH` and every write
///    is additive and idempotent (see *Idempotency* above), so the single
///    `MULTI … EXEC` replay fred's transaction router performs under the
///    `txn_pool`'s `max_command_attempts = 1` cannot corrupt anything, and a
///    half-done apply (A committed, B failed) is self-describing and converges
///    on the next attempt — pinned by
///    `recovery_retry_after_step_b_failure_converges`.
/// 2. **A caller re-drives on `Transient`.** The re-drive is request-path, not
///    a maintenance tick: `SessionBasePathCache::get_base_path` maps
///    `Transient` → `ServiceError::Unavailable` (the client retries), and
///    `grpc::session::reconcile_ghost_members` deliberately does not `SREM` on
///    a recovery error, so the member survives and the next `ListSessions`
///    walk retries.
/// 3. **This site's retriability is a property of its own writes, not of the
///    shared classifier.** The shape that matters most here is the routine
///    cluster reshard: with `max_redirections = 0` pinned deliberately on the
///    `txn_pool` (a `MOVED` on one member of a pipelined `MULTI` must not be
///    re-sent alone as an unqueued write — the pool's comment in
///    `ValkeyClient::connect`), fred's transaction router fails a redirected
///    transaction with its own `Unknown` + `"Max redirections exceeded"`
///    (`fred-10.1.0/src/router/transactions.rs:65-67`, sent at `:216` and
///    `:242`). Its other manufactured shape is `Unknown` + `"Max attempts
///    exceeded"` from the `retry!(None)` arms (a failed `router.drain_all`,
///    `:142-145`; a route miss, `:147-156`). Every other failure forwards its
///    real `IO` / `Timeout`.
///
///    The storage classifier grades all of those transient
///    (`storage::client::is_transaction_router_exhausted_error`, b.cyu), so
///    today the wrap and the classifier agree on every shape this applier is
///    expected to meet. The wrap stays blanket anyway: a graded wrap would
///    make a joiner's retry depend on the shared classifier's next edit,
///    when what makes a re-drive safe here is reason 1, which the classifier
///    cannot see. The cost of the blanket grade is the deterministic class
///    below. Each step's `EXEC` goes through `ValkeyClient::exec_txn`, as
///    every fred `Transaction` caller's does, so on a `"Max redirections
///    exceeded"` the checkout's routing is re-synced before the error reaches
///    the arms, because fred does not re-sync it.
/// 4. **A local retry loop would serialise joiners.** The applier runs under
///    `SessionRecoveryHook::try_recover_session`'s K=1 per-`session_id`
///    admission mutex on a deadline-exempt RPC; spinning there holds that
///    mutex and stalls every other joiner for the same session.
///
/// The blanket grade does sweep in a **deterministic-rejection** class that a
/// re-drive can never clear: a `WRONGTYPE` from a corrupted `le:path:{hex(bp)}`
/// or `le:{sid}:sess` lands in the catch-all `Err(e)` arms and is graded
/// `Transient`, so the client re-drives it as `UNAVAILABLE` instead of getting a
/// non-retriable `INTERNAL`. That is accepted deliberately, not an oversight in
/// those arms — a corrupted restore target is an operator-visible keyspace fault
/// under either status, and the blanket grade is what keeps this site's
/// retriability independent of the shared classifier (reason 3 above).
/// Splitting the deterministic shapes out into their own non-retriable arm is a
/// possible future refinement.
///
/// # Production caller
///
/// [`SessionRecoveryHook::try_recover_session`](crate::checkpoint::session_recovery::SessionRecoveryHook::try_recover_session)
/// step 7 (`src/checkpoint/session_recovery.rs`), reached from the
/// request-path recovery arms described under *Error classification* above.
/// (The earlier `#[allow(dead_code)]` placeholder for the pre-consumer window
/// is gone — this is a live path.)
pub(crate) async fn apply_session_checkpoint(
    client: &ValkeyClient,
    keys: &KeySchema,
    session_id: &str,
    checkpoint: &SessionCheckpoint,
) -> Result<(), CheckpointError> {
    // Defensive pre-check: the function operates on `session_id` for key
    // construction; the field on the payload is informational. They MUST
    // match for the applied state to be coherent; a debug-build mismatch
    // catches a bug at the call site without slowing release builds.
    debug_assert_eq!(
        checkpoint.session_id, session_id,
        "apply_session_checkpoint: checkpoint.session_id must match session_id argument"
    );

    // ── Step A — Path-index MULTI/EXEC ({hex(base_path)} slot) ──────────────
    // ORDER IS LOAD-BEARING per SR-2.7. Do not move Step B above Step A
    // without a spec change — see the module-level doc comment for the
    // self-healing rationale.
    #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
    if let Some(err) = pop_apply_step_fault(ApplyStep::StepA) {
        return Err(err);
    }

    // Empty-set degenerate case: a real assembler always emits a non-empty
    // `path_index_session_id_member` (the assembler sets it to the literal
    // session_id at the top of `build_session_checkpoint`), so reaching
    // this skip branch on real input is itself a bug signal. We warn and
    // proceed — applying a malformed checkpoint with no path-index
    // contribution would emit a no-op MULTI/EXEC (Valkey returns an empty
    // reply array, not an error), which is silent. Surface the anomaly.
    let path_index_empty_contribution = checkpoint.path_index_session_id_member.is_empty()
        && checkpoint.path_name_members.is_empty();
    if path_index_empty_contribution {
        tracing::warn!(
            session_id = %session_id,
            "session checkpoint has empty path-index contribution; skipping Step A \
             (malformed checkpoint — real assembler always emits a non-empty path_index_session_id_member)"
        );
        debug_assert!(
            false,
            "apply_session_checkpoint: empty path-index contribution indicates a malformed checkpoint"
        );
    } else {
        // b.4t3: every key this transaction touches — `le:path:{hex(bp)}` and
        // `le:path:{hex(bp)}:names` — carries the same `{hex(base_path)}` hash
        // tag (`keys.rs::path_index_and_path_names_index_share_hash_tag`), so
        // the path-index key names the one slot this MULTI/EXEC runs on and the
        // checkout hygiene probe is pinned to it. Hoisted ABOVE the checkout
        // and reused as the first queued key so the probe's pin and the slot
        // fred derives for EXEC are the same string by construction rather than
        // by assertion (idiom: `SessionStore::delete`, src/storage/session.rs).
        // On the degenerate branch where `path_index_session_id_member` is
        // empty but `path_name_members` is not, the pin key is a *sibling* of
        // the key actually touched — still the same slot, by that tag-sharing
        // property.
        let path_index_key = keys.path_index(&checkpoint.base_path);
        let exclusive_client = client.acquire_txn_for(&path_index_key).await;
        let trx = exclusive_client.multi();

        // Legacy session-id-only set: SADD the bare session_id member.
        // Skipped only on the all-empty degenerate branch above.
        if !checkpoint.path_index_session_id_member.is_empty() {
            let _: () = trx
                .sadd(
                    &path_index_key,
                    checkpoint.path_index_session_id_member.as_str(),
                )
                .await
                .map_err(|e| CheckpointError::Transient {
                    source: format!(
                        "Step A path-index MULTI/EXEC failed: SADD path_index queue: {e}"
                    )
                    .into(),
                })?;
        }

        // Composite path-names set: one SADD per composite member. The
        // assembler emits these via `add_path_name_member`; the applier
        // MUST NOT re-derive them — the pre-composed strings are
        // authoritative.
        if !checkpoint.path_name_members.is_empty() {
            let path_names_key = keys.path_names_index(&checkpoint.base_path);
            for member in &checkpoint.path_name_members {
                let _: () = trx
                    .sadd(&path_names_key, member.as_str())
                    .await
                    .map_err(|e| CheckpointError::Transient {
                        source: format!(
                            "Step A path-index MULTI/EXEC failed: SADD path_names queue: {e}"
                        )
                        .into(),
                    })?;
            }
        }

        // Decode EXEC as `fred::types::Value` — same contract as
        // `create_layer` / `SessionStore::create_once`. A null reply
        // surfaces as `Value::Null`; `is_canceled()` matches the
        // connection-drop pattern fred reports for canceled transactions.
        //
        // `Ok(Value::Null)` has TWO sources, only one of which reaches the wire:
        //
        //  1. Valkey aborted the transaction because a WATCHed key changed.
        //     This applier issues NO `WATCH` of its own, so the only reachable
        //     cause is residue: a `WATCH` leaked onto this slot by a previous
        //     borrower and inherited at checkout. That is exactly what the
        //     now-slot-pinned checkout probe (`acquire_txn_for` above) clears.
        //  2. fred short-circuits an EMPTY command queue — `exec()` returns
        //     `Ok(Value::Null)` before sending anything
        //     (`fred-10.1.0/src/clients/transaction.rs:285-287`). Here that is
        //     excluded only by the surrounding guards: this branch is the `else`
        //     of `path_index_empty_contribution`, so at least one of the two
        //     SADD groups queued a command. An edit that makes ALL of Step A's
        //     writes conditional re-opens this path, and the null would then be
        //     reported as WATCH residue that never happened.
        //
        // Either way a null stays classified `Transient` — the re-drive is
        // idempotent, and neither cause is a data-corrupting outcome.
        //
        // Every error arm below wraps as `Transient` deliberately; see the
        // `# Error classification` section of this function's doc comment.
        // `exec_txn` re-syncs this checkout's routing when the transaction was
        // redirected (fred re-syncs nothing on it), before any arm runs.
        match client
            .exec_txn::<fred::types::Value>(
                &exclusive_client,
                trx,
                &path_index_key,
                "session checkpoint apply Step A",
                true,
            )
            .await
        {
            Ok(v) if v.is_null() => {
                return Err(CheckpointError::Transient {
                    source: "Step A path-index MULTI/EXEC failed: EXEC aborted (inherited WATCH \
                             residue, or an empty command queue)"
                        .into(),
                });
            }
            Ok(_) => {}
            Err(e) if e.is_canceled() => {
                return Err(CheckpointError::Transient {
                    source: format!(
                        "Step A path-index MULTI/EXEC failed: transaction canceled (connection drop): {e}"
                    )
                    .into(),
                });
            }
            Err(e) => {
                FredErrOrigin::client(client, POOL_LABEL_TXN).observe_error(&e);
                return Err(CheckpointError::Transient {
                    source: format!("Step A path-index MULTI/EXEC failed: EXEC failed: {e}").into(),
                });
            }
        }
        // `exclusive_client` drops here, releasing the txn slot — and with it
        // the gauge and the checkout permit, in that order — before Step B.
    }

    // ── Step B — Session-scoped MULTI/EXEC ({session_id} slot) ──────────────
    #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
    if let Some(err) = pop_apply_step_fault(ApplyStep::StepB) {
        return Err(err);
    }

    // b.4t3: every key this transaction touches — `le:{sid}:sess`,
    // `le:{sid}:layer:{hex(name)}`, `le:{sid}:layers` — carries the
    // `{session_id}` hash tag (Valkey's tag is first `{` … first `}`, so the
    // trailing `{hex(name)}` on the mapping key is inert), so the session-meta
    // key names the one slot this MULTI/EXEC runs on and the checkout hygiene
    // probe is pinned to it. Hoisted ABOVE the checkout and reused as the first
    // queued key so the probe's pin and the slot fred derives for EXEC are the
    // same string by construction (idiom: `SessionStore::delete`,
    // src/storage/session.rs).
    let session_meta_key = keys.session_meta(session_id);
    let exclusive_client = client.acquire_txn_for(&session_meta_key).await;
    let trx = exclusive_client.multi();

    // (B.1) session-meta HSET — the four-field set MUST match
    // `SessionStore::create_once` at `src/storage/session.rs:386-394`.
    // See the module-level doc comment for the contract.
    let _: () = trx
        .hset(
            &session_meta_key,
            [
                ("name", checkpoint.name.clone()),
                ("base_path", checkpoint.base_path.clone()),
                ("created_by", checkpoint.created_by.clone()),
                ("created_at", checkpoint.created_at.to_string()),
            ],
        )
        .await
        .map_err(|e| CheckpointError::Transient {
            source: format!(
                "Step B session-scoped MULTI/EXEC failed: HSET session_meta queue: {e}"
            )
            .into(),
        })?;

    // (B.2) layer-mapping SETs — one per `(layer_name, object_id)`.
    // No TTL, no NX/XX — overwrite is the idempotency contract.
    for (layer_name, object_id) in &checkpoint.layer_mappings {
        let mapping_key = keys.layer_mapping(session_id, layer_name);
        let _: () = trx
            .set(&mapping_key, object_id.to_string(), None, None, false)
            .await
            .map_err(|e| CheckpointError::Transient {
                source: format!(
                    "Step B session-scoped MULTI/EXEC failed: SET layer_mapping queue ({layer_name}): {e}"
                )
                .into(),
            })?;
    }

    // (B.3) layers-index SADDs — one per layer name. In a well-formed
    // checkpoint this equals `layer_mappings.keys()`, but we drive it from
    // the explicit `layers_index` field per the payload contract (the
    // assembler may emit a divergent index in a torn-snapshot case).
    if !checkpoint.layers_index.is_empty() {
        let layers_index_key = keys.layers_index(session_id);
        for name in &checkpoint.layers_index {
            // The layers-index stores hex-encoded names (matches
            // `LayerStore::create_layer`'s SADD); apply the same encoding
            // here so the recovered set is byte-identical to one written
            // by the live write path.
            let hex_name = crate::storage::keys::hex_encode(name);
            let _: () = trx.sadd(&layers_index_key, hex_name).await.map_err(|e| {
                CheckpointError::Transient {
                    source: format!(
                        "Step B session-scoped MULTI/EXEC failed: SADD layers_index queue ({name}): {e}"
                    )
                    .into(),
                }
            })?;
        }
    }

    // Decode EXEC as `fred::types::Value` — same contract as Step A, including
    // the two-source null reply: no `WATCH` is issued here, so an abort can only
    // come from residue inherited at checkout (which the pinned probe clears),
    // and fred's empty-command-queue short-circuit
    // (`fred-10.1.0/src/clients/transaction.rs:285-287`) is excluded only
    // because (B.1)'s session-meta HSET is UNCONDITIONAL — making it conditional
    // re-opens that path. Every arm wraps as `Transient` deliberately — see the
    // `# Error classification` section of this function's doc comment. As in
    // Step A, `exec_txn` re-syncs a redirected checkout's routing first.
    match client
        .exec_txn::<fred::types::Value>(
            &exclusive_client,
            trx,
            &session_meta_key,
            "session checkpoint apply Step B",
            true,
        )
        .await
    {
        Ok(v) if v.is_null() => Err(CheckpointError::Transient {
            source: "Step B session-scoped MULTI/EXEC failed: EXEC aborted (inherited WATCH \
                     residue, or an empty command queue)"
                .into(),
        }),
        Ok(_) => Ok(()),
        Err(e) if e.is_canceled() => Err(CheckpointError::Transient {
            source: format!(
                "Step B session-scoped MULTI/EXEC failed: transaction canceled (connection drop): {e}"
            )
            .into(),
        }),
        Err(e) => {
            FredErrOrigin::client(client, POOL_LABEL_TXN).observe_error(&e);
            Err(CheckpointError::Transient {
                source: format!("Step B session-scoped MULTI/EXEC failed: EXEC failed: {e}")
                    .into(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    //! Integration-flavored unit tests for [`build_session_checkpoint`].
    //!
    //! These tests use a live `TestValkey` (single-node) — Cluster-mode
    //! coverage lives in sibling Task `t2.fgj.7v.o7`. Per the project's
    //! sandbox note in `CLAUDE.md`, inside the sandbox the harness exports
    //! `TEST_VALKEY_URL` to reach `host.docker.internal:6379`.
    use super::*;
    use crate::checkpoint::session_payload::SESSION_CHECKPOINT_SCHEMA_VERSION;
    use crate::domain::types::ObjectState;
    use crate::storage::external_values::MockExternalValueStore;
    use crate::storage::keys::{hex_encode, KeySchema};
    use crate::storage::layer::LayerStore;
    use crate::storage::session::SessionStore;
    use crate::storage::test_support::TestValkey;
    use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};
    use std::sync::Arc;

    /// Compose a fresh empty `ObjectState` for `create_layer`.
    fn fresh_state() -> ObjectState {
        ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usd".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn build_session_checkpoint_round_trip_via_test_valkey_returns_expected_struct() {
        let tv = TestValkey::setup().await;
        let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let ext_store: Arc<MockExternalValueStore> = Arc::new(MockExternalValueStore::new());

        let session_id = ss
            .create("round-trip-stage", "/stage/round-trip", "round-trip-user")
            .await
            .expect("session creation must succeed");

        let oid_root = ls
            .create_layer(
                &session_id,
                "root.usda",
                "round-trip-user",
                &fresh_state(),
                ext_store.clone(),
            )
            .await
            .expect("create_layer root.usda must succeed");
        let oid_sub = ls
            .create_layer(
                &session_id,
                "sublayer.usda",
                "round-trip-user",
                &fresh_state(),
                ext_store.clone(),
            )
            .await
            .expect("create_layer sublayer.usda must succeed");

        let checkpoint = build_session_checkpoint(tv.client(), tv.key_schema(), &session_id)
            .await
            .expect("assembler must succeed")
            .expect("session must exist")
            .checkpoint;

        assert_eq!(checkpoint.schema_version, SESSION_CHECKPOINT_SCHEMA_VERSION);
        assert_eq!(checkpoint.session_id, session_id);
        assert_eq!(checkpoint.name, "round-trip-stage");
        assert_eq!(checkpoint.base_path, "/stage/round-trip");
        assert_eq!(checkpoint.created_by, "round-trip-user");
        assert!(
            checkpoint.created_at > 0,
            "created_at must be the non-zero now_micros() captured by SessionStore::create"
        );
        assert_eq!(checkpoint.layer_mappings.len(), 2);
        assert_eq!(
            checkpoint.layer_mappings.get("root.usda"),
            Some(&oid_root),
            "layer_mapping for root.usda must round-trip"
        );
        assert_eq!(
            checkpoint.layer_mappings.get("sublayer.usda"),
            Some(&oid_sub),
            "layer_mapping for sublayer.usda must round-trip"
        );
        assert_eq!(checkpoint.layers_index.len(), 2);
        assert!(checkpoint.layers_index.contains("root.usda"));
        assert!(checkpoint.layers_index.contains("sublayer.usda"));
        // The assembler emits EXACTLY ONE `path_names_index` composite per
        // session — the `session_name\0session_id` composite that mirrors
        // what `SessionStore::create_once` writes on the live path
        // (`src/storage/session.rs:308`). Emitting a per-LAYER composite
        // (the pre-fix shape) broke `ListSessions` (returned N duplicate
        // entries), session-name uniqueness inside `create_once` (any
        // composite carrying the new name would falsely match), and
        // `DeleteSession`'s SREM (only the `session_name\0session_id`
        // composite is targeted). The check goes through
        // `KeySchema::split_path_name_member` — the sanctioned INVERSE
        // helper — to avoid duplicating the composite-construction call
        // site outside `SessionCheckpoint::add_path_name_member`.
        assert_eq!(checkpoint.path_name_members.len(), 1);
        let only_member = checkpoint
            .path_name_members
            .iter()
            .next()
            .expect("exactly one composite expected");
        let (recovered_name, recovered_sid) =
            KeySchema::split_path_name_member(only_member).expect("composite must split on NUL");
        assert_eq!(
            recovered_name, "round-trip-stage",
            "composite must carry the SESSION name, not a layer name"
        );
        assert_eq!(recovered_sid, session_id, "session_id half must round-trip");
        assert_eq!(checkpoint.path_index_session_id_member, session_id);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn build_session_checkpoint_absent_session_returns_ok_none() {
        let tv = TestValkey::setup().await;

        // Never-created session id — no key with the `le:{ghost-sid}:sess`
        // pattern exists.
        let result = build_session_checkpoint(tv.client(), tv.key_schema(), "ghost-sid")
            .await
            .expect("assembler call must not fail on absent session");
        assert!(
            result.is_none(),
            "absent session must yield Ok(None); got {result:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn build_session_checkpoint_multi_layer_session_returns_all_mappings() {
        let tv = TestValkey::setup().await;
        let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let ext_store: Arc<MockExternalValueStore> = Arc::new(MockExternalValueStore::new());

        let session_id = ss
            .create("multi-stage", "/stage/multi", "multi-user")
            .await
            .expect("session creation must succeed");

        for layer_name in ["alpha.usda", "beta.usda", "gamma.usda"] {
            ls.create_layer(
                &session_id,
                layer_name,
                "multi-user",
                &fresh_state(),
                ext_store.clone(),
            )
            .await
            .expect("create_layer must succeed");
        }

        let checkpoint = build_session_checkpoint(tv.client(), tv.key_schema(), &session_id)
            .await
            .expect("assembler must succeed")
            .expect("session must exist")
            .checkpoint;

        assert_eq!(checkpoint.layer_mappings.len(), 3);
        assert_eq!(checkpoint.layers_index.len(), 3);
        // Exactly one composite per session — `session_name\0session_id`
        // — irrespective of layer count. Mirrors the live-write contract
        // in `SessionStore::create_once`.
        assert_eq!(checkpoint.path_name_members.len(), 1);
        let only_member = checkpoint
            .path_name_members
            .iter()
            .next()
            .expect("exactly one composite expected");
        let (recovered_name, recovered_sid) =
            KeySchema::split_path_name_member(only_member).expect("composite must split on NUL");
        assert_eq!(
            recovered_name, "multi-stage",
            "composite must carry the SESSION name, not a layer name"
        );
        assert_eq!(recovered_sid, &session_id);
    }

    /// A layers-index member whose mapping key is absent is filtered out of
    /// the payload rather than propagated as a layer with no `object_id`.
    ///
    /// **This is no longer a read race** (b.443 made the read atomic, so the
    /// index membership and the mappings are observed at one instant). It is
    /// a genuine keyspace state and still reachable: `delete_layer` DELs the
    /// mapping several steps before it SREMs the index member, so a build
    /// landing in that window legitimately sees an index member with no
    /// mapping. Dropping it is correct — a checkpoint naming a layer with no
    /// `object_id` would restore an unreachable layer, which is exactly what
    /// `apply_checkpoint`'s SR-2.4 live-mapping guard then refuses.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn build_session_checkpoint_layers_index_member_without_mapping_is_filtered() {
        let tv = TestValkey::setup().await;
        let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let ext_store: Arc<MockExternalValueStore> = Arc::new(MockExternalValueStore::new());

        let session_id = ss
            .create("race-stage", "/stage/race", "race-user")
            .await
            .expect("session creation must succeed");
        ls.create_layer(
            &session_id,
            "raced.usda",
            "race-user",
            &fresh_state(),
            ext_store.clone(),
        )
        .await
        .expect("create_layer must succeed");

        // Reproduce the mid-`delete_layer` state: DEL the mapping key
        // directly (bypassing `delete_layer`'s cascade, which would also
        // SREM the layers-index member). The layers-index keeps the orphan
        // member.
        let mapping_key = tv.key_schema().layer_mapping(&session_id, "raced.usda");
        let _: () = tv
            .client()
            .pool
            .del(&mapping_key)
            .await
            .expect("DEL mapping key must succeed");

        let checkpoint = build_session_checkpoint(tv.client(), tv.key_schema(), &session_id)
            .await
            .expect("assembler must succeed")
            .expect("session must exist")
            .checkpoint;

        assert_eq!(
            checkpoint.layer_mappings.len(),
            0,
            "orphan layers-index member must not propagate into layer_mappings"
        );
        assert_eq!(
            checkpoint.layers_index.len(),
            0,
            "orphan layers-index member must not propagate into layers_index"
        );
        // The session itself still exists with a populated `session_meta`,
        // so the assembler emits the one session-name composite
        // (`session_name\0session_id`) irrespective of the layer-mapping
        // race outcome.
        assert_eq!(
            checkpoint.path_name_members.len(),
            1,
            "session-name composite must be present even with zero surviving layers"
        );
        let only = checkpoint
            .path_name_members
            .iter()
            .next()
            .expect("composite expected");
        let (n, s) = KeySchema::split_path_name_member(only).expect("composite must split on NUL");
        assert_eq!(n, "race-stage");
        assert_eq!(s, session_id);
    }

    /// Filtering an orphan index member must not shift the pairing of the
    /// SURVIVING members with their mapping values.
    ///
    /// b.443's read script returns layer names and mapping values as two
    /// positionally-aligned arrays, and builds that alignment by appending
    /// to both only when a mapping is live. That is a new failure mode with
    /// no analogue in the pre-b.443 per-name `GET` fan-out: an off-by-one in
    /// the filter would hand every layer after the hole its NEIGHBOUR's
    /// `object_id`. Nothing downstream can catch that — the ids are all
    /// plausible `u64`s, so recovery would silently restore layers pointing
    /// at each other's objects.
    ///
    /// Three layers with the middle one's mapping removed puts a hole in the
    /// middle of the array, which is where a shift shows up.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn build_session_checkpoint_orphan_member_does_not_shift_surviving_mappings() {
        let tv = TestValkey::setup().await;
        let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let ext_store: Arc<MockExternalValueStore> = Arc::new(MockExternalValueStore::new());

        let session_id = ss
            .create("align-stage", "/stage/align", "align-user")
            .await
            .expect("session creation must succeed");

        let mut oids = Vec::new();
        for name in ["a.usda", "b.usda", "c.usda"] {
            oids.push(
                ls.create_layer(
                    &session_id,
                    name,
                    "align-user",
                    &fresh_state(),
                    ext_store.clone(),
                )
                .await
                .expect("create_layer must succeed"),
            );
        }
        // Distinct object_ids are what make a shift detectable at all.
        assert_eq!(
            oids.iter().collect::<std::collections::HashSet<_>>().len(),
            3,
            "precondition: the three layers must have distinct object_ids"
        );

        let orphan_key = tv.key_schema().layer_mapping(&session_id, "b.usda");
        let _: () = tv
            .client()
            .pool
            .del(&orphan_key)
            .await
            .expect("DEL mapping key must succeed");

        let checkpoint = build_session_checkpoint(tv.client(), tv.key_schema(), &session_id)
            .await
            .expect("assembler must succeed")
            .expect("session must exist")
            .checkpoint;

        assert_eq!(
            checkpoint.layer_mappings.get("a.usda"),
            Some(&oids[0]),
            "a.usda must keep its OWN object_id after the hole was filtered; getting b/c's id \
             here means the name and value arrays came back misaligned"
        );
        assert_eq!(
            checkpoint.layer_mappings.get("c.usda"),
            Some(&oids[2]),
            "c.usda sits AFTER the filtered hole — this is the position a filter off-by-one \
             corrupts, and the corrupted value is a perfectly plausible object_id that nothing \
             downstream would reject"
        );
        assert!(
            !checkpoint.layer_mappings.contains_key("b.usda"),
            "the orphan member itself must be absent from the payload"
        );
    }

    /// The build stamp is minted atomically with the read, is monotone
    /// across calls, and carries a TTL.
    ///
    /// All three properties are load-bearing for the publish watermark:
    ///
    /// - **Monotone** is the freshness order itself. Two builds that
    ///   overlap must be comparable, and the later-minted one must be the
    ///   one whose payload reflects at least as much committed state.
    /// - **Minted in Valkey, not in-process**, because there is no session
    ///   affinity — the racing builds are typically on different replicas,
    ///   and an in-process counter could not order them.
    /// - **TTL'd**, because `le:{sid}:ckpt` is deliberately absent from
    ///   `SessionStore::delete`'s teardown (a repair reads the watermark
    ///   after the DELETE lands), so nothing else ever reaps it. A missing
    ///   TTL is an unbounded leak of one hash per session ever touched.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn build_session_checkpoint_mints_monotone_stamp_with_ttl() {
        let tv = TestValkey::setup().await;
        let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

        let session_id = ss
            .create("stamp-stage", "/stage/stamp", "stamp-user")
            .await
            .expect("session creation must succeed");

        let ckpt_key = tv.key_schema().session_checkpoint_state(&session_id);
        let existed_before: bool = tv
            .client()
            .pool
            .exists(&ckpt_key)
            .await
            .expect("EXISTS must succeed");
        assert!(
            !existed_before,
            "precondition: nothing may have touched {ckpt_key} before the first build"
        );

        let mut stamps = Vec::new();
        for _ in 0..3 {
            stamps.push(
                build_session_checkpoint(tv.client(), tv.key_schema(), &session_id)
                    .await
                    .expect("assembler must succeed")
                    .expect("session must exist")
                    .stamp,
            );
        }

        assert!(
            stamps.windows(2).all(|w| w[1] > w[0]),
            "build stamps MUST be strictly increasing — the publish watermark treats a higher \
             stamp as 'strictly fresher payload', so a repeated or decreasing stamp would let a \
             stale build's PUT be accepted as the newest; got {stamps:?}"
        );

        let ttl_ms: i64 = tv
            .client()
            .pool
            .pttl(&ckpt_key)
            .await
            .expect("PTTL must succeed");
        assert!(
            ttl_ms > 0,
            "{ckpt_key} MUST carry a TTL — it is deliberately not deleted by \
             SessionStore::delete, so a TTL is the only thing that reaps it; PTTL returned \
             {ttl_ms} (-1 = no expiry, -2 = key missing)"
        );
        assert!(
            ttl_ms as u64 <= SESSION_CHECKPOINT_STATE_TTL_MS,
            "PTTL {ttl_ms} exceeds the configured {SESSION_CHECKPOINT_STATE_TTL_MS} ms — the \
             script applied some other TTL than SESSION_CHECKPOINT_STATE_TTL_MS"
        );
    }

    /// **A minted incarnation token is 32 lowercase hex characters, and no two
    /// mints collide.**
    ///
    /// This pins the property [`mint_incarnation_token`]'s rustdoc actually
    /// promises — 128 bits of OS randomness, so that "two replicas re-creating
    /// the key must not be able to agree by accident" — which nothing else
    /// covers. Its sibling
    /// `the_incarnation_is_stable_across_builds_and_changes_when_the_key_is_lost`
    /// asserts only that the token is stable while the key lives and different
    /// after the key is lost, and a per-replica counter (`replica-7:0001`,
    /// `replica-7:0002`, …) satisfies both halves of that while colliding
    /// head-on with a peer's — precisely the accidental agreement the CSPRNG
    /// exists to rule out. So the randomness needs its own test.
    ///
    /// Two halves, both cheap and neither flaky:
    ///
    /// - **Shape.** Exactly 32 characters, every one of them `0-9a-f`. The
    ///   width is the bit count made observable — 31 characters would mean 4
    ///   bits went missing (e.g. a `{:x}` that dropped a leading zero instead
    ///   of the `{:02x}` the function uses), and a case flip would break the
    ///   byte-for-byte equality the publish gate performs against the token
    ///   stored in Valkey.
    /// - **Distinctness.** 1000 consecutive mints are all different. This
    ///   cannot be a flake: over a 2^128 space the birthday probability of a
    ///   collision among 1000 draws is ~10^-33. It fails loudly and
    ///   deterministically for the failure that matters — a token seeded once
    ///   per process, derived from a coarse clock, or otherwise not re-drawn
    ///   per call — which is exactly the shape a hand-rolled replacement for
    ///   `OsRng` would take.
    #[test]
    fn mint_incarnation_token_is_32_lowercase_hex_chars_and_never_repeats() {
        use std::collections::HashSet;

        const MINTS: usize = 1000;

        let mut seen: HashSet<String> = HashSet::with_capacity(MINTS);
        for _ in 0..MINTS {
            let token = mint_incarnation_token();

            assert_eq!(
                token.len(),
                32,
                "an incarnation token renders 16 CSPRNG bytes as `{{:02x}}` each, so it is always \
                 32 characters wide; got {} for {token:?}. A short token means a byte lost its \
                 leading zero and the value carries fewer than the 128 bits the collision \
                 argument rests on",
                token.len()
            );
            assert!(
                token
                    .chars()
                    .all(|c| c.is_ascii_digit() || matches!(c, 'a'..='f')),
                "the token must be lowercase hex only — the publish gate compares it byte-for-\
                 byte against what is stored on `le:{{sid}}:ckpt`, so any uppercase digit or \
                 non-hex character would make an incarnation fail to match itself; got {token:?}"
            );

            assert!(
                seen.insert(token.clone()),
                "mint #{} repeated the already-minted token {token:?}. Over 128 bits a genuine \
                 collision in {MINTS} draws is ~10^-33, so this is not chance: the token is not \
                 being re-drawn from the CSPRNG per call (seeded once per process, or derived \
                 from a clock/counter). Two replicas re-creating `le:{{sid}}:ckpt` could then \
                 agree by accident, and the publish gate would admit exactly the stale operation \
                 the incarnation exists to refuse",
                seen.len() + 1
            );
        }
    }

    /// **The incarnation names the KEY's lifetime, not the build (b.hka (a)).**
    ///
    /// One test, one property, expressed as the two halves that make the
    /// property useful — and they are only meaningful together, because
    /// either alone is trivially satisfiable (a constant satisfies "stable";
    /// a fresh random per call satisfies "changes"):
    ///
    /// - **Stable while the key survives.** Every build against a live
    ///   `le:{sid}:ckpt` must read back the SAME token, or the publish gate
    ///   would refuse a perfectly ordered operation and every session
    ///   checkpoint would enter a permanent rebuild loop. `HSETNX` is what
    ///   buys this: only the mint that brought the key into existence
    ///   installs its candidate.
    /// - **Different after the key is lost.** That is the entire point: a
    ///   build minted before the loss holds a stamp drawn from a counter that
    ///   no longer exists, and the changed token is the only evidence
    ///   available at publish time — `seq` has restarted from 1 and looks
    ///   perfectly healthy.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_incarnation_is_stable_across_builds_and_changes_when_the_key_is_lost() {
        let tv = TestValkey::setup().await;
        let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

        let session_id = ss
            .create("inc-stage", "/stage/inc", "inc-user")
            .await
            .expect("session creation must succeed");

        let first = read_session_state(tv.client(), tv.key_schema(), &session_id)
            .await
            .expect("the first read must succeed");
        let second = read_session_state(tv.client(), tv.key_schema(), &session_id)
            .await
            .expect("the second read must succeed");

        assert!(
            !first.incarnation.is_empty(),
            "an empty incarnation would make the publish gate vacuous — every operation would \
             compare equal to every other"
        );
        assert_eq!(
            first.incarnation, second.incarnation,
            "two builds against a SURVIVING key MUST observe the same incarnation. A token \
             re-minted per build would refuse every publish as stale and put every session \
             checkpoint into a permanent rebuild loop"
        );
        assert!(
            second.stamp > first.stamp,
            "precondition: the two reads really were distinct mints ({} then {})",
            first.stamp,
            second.stamp
        );

        // Keyspace loss, scoped to this one key. NEVER a FLUSH — the backing
        // instance is shared (see CLAUDE.md).
        let ckpt_key = tv.key_schema().session_checkpoint_state(&session_id);
        let _: () = tv
            .client()
            .pool
            .del(&ckpt_key)
            .await
            .expect("DEL must succeed");

        let after_loss = read_session_state(tv.client(), tv.key_schema(), &session_id)
            .await
            .expect("the post-loss read must succeed");

        assert_ne!(
            after_loss.incarnation, first.incarnation,
            "a key that was lost and re-created MUST carry a NEW incarnation — it is the only \
             evidence that distinguishes the re-created key from the original, and without it a \
             build minted before the loss publishes a stamp that outranks the fresh watermark \
             and durably reinstates pre-loss bytes"
        );
        assert_eq!(
            after_loss.stamp, 1,
            "and the counter really did restart, which is exactly why the stamp alone cannot be \
             trusted: stamp 1 here looks perfectly healthy next to the pre-loss stamp of {}",
            second.stamp
        );
    }

    /// **`session_checkpoint_read.lua` returns SEVEN elements.**
    ///
    /// The decoder rejects any other arity outright, so a script that grew or
    /// lost an element would take every session checkpoint down with a
    /// `Permanent` error rather than degrading. Asserted against the raw
    /// `EVALSHA` reply because that is the only place the shape is
    /// observable — `read_session_state` has already collapsed it into a
    /// struct by the time a caller sees it, and the struct would still
    /// typecheck against a six-element script that happened to shift a field.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_read_script_reply_carries_seven_elements() {
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        let session_id = "arity-probe";

        let reply: Value = SESSION_CHECKPOINT_READ_SCRIPT
            .evalsha_with_reload(
                tv.client().pool.next(),
                vec![
                    keys.session_checkpoint_state(session_id),
                    keys.session_meta(session_id),
                    keys.layers_index(session_id),
                ],
                vec![
                    SESSION_CHECKPOINT_STATE_TTL_MS.to_string(),
                    keys.layer_mapping(session_id, ""),
                    "arity-probe-incarnation".to_string(),
                ],
            )
            .await
            .expect("EVALSHA session_checkpoint_read.lua must succeed");

        let parts = reply.into_array();
        assert_eq!(
            parts.len(),
            7,
            "the reply MUST carry exactly 7 elements — [7] is the effective incarnation added by \
             b.hka (a), and it is returned on the `present == 0` arm too because the DELETE \
             dispatch path mints against an already-deleted session and still has to publish \
             under the right incarnation"
        );
        assert_eq!(
            parts[1].as_u64(),
            Some(0),
            "precondition: this probe names a session that does not exist, so element [2] \
             (`present`) must be 0 — the arm on which the incarnation is easiest to forget"
        );
        assert_eq!(
            parts[6].as_string().as_deref(),
            Some("arity-probe-incarnation"),
            "and on that arm element [7] MUST be the token this mint installed, since the mint \
             created the key"
        );
    }

    /// An absent session mints a stamp and returns `Ok(None)` — and the mint
    /// stays ordered against the still-live sessions' stamps.
    ///
    /// The unconditional mint is what lets the DELETE dispatch path reuse
    /// this script purely to obtain an ordered stamp: by construction its
    /// session is already gone, so the read half short-circuits and costs
    /// nothing. If the mint were gated on the session existing, a DELETE
    /// could not be ordered against writes already in flight on another
    /// replica, and the watermark would have nothing to compare.
    ///
    /// The `Ok(None)` half is separately load-bearing: it must NOT be
    /// escalated into a checkpoint delete. An empty `le:{sid}:sess` is also
    /// what selective key loss looks like, and tombstoning on it would
    /// destroy the recovery artifact that exists to repair exactly that.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn absent_session_still_mints_an_ordered_stamp() {
        let tv = TestValkey::setup().await;
        let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

        let session_id = ss
            .create("delete-stamp-stage", "/stage/delete-stamp", "stamp-user")
            .await
            .expect("session creation must succeed");

        let live_stamp = build_session_checkpoint(tv.client(), tv.key_schema(), &session_id)
            .await
            .expect("assembler must succeed")
            .expect("session must exist")
            .stamp;

        ss.delete(&session_id)
            .await
            .expect("SessionStore::delete must succeed");

        let build_after_delete =
            build_session_checkpoint(tv.client(), tv.key_schema(), &session_id)
                .await
                .expect("assembler must succeed on an absent session, not error");
        assert!(
            build_after_delete.is_none(),
            "an absent session MUST yield Ok(None) so the writer exits silently — converting it \
             into a checkpoint delete would destroy the artifact that repairs selective key loss"
        );

        let (delete_stamp, _incarnation) =
            mint_session_checkpoint_stamp(tv.client(), tv.key_schema(), &session_id)
                .await
                .expect("stamp mint must succeed for a deleted session");
        assert!(
            delete_stamp > live_stamp,
            "the DELETE path's stamp ({delete_stamp}) MUST outrank the earlier live build's \
             ({live_stamp}) — otherwise a delete cannot be ordered against a write whose build \
             is already in flight on another replica, and the watermark has nothing to compare"
        );
    }

    /// **Atomic-cut smoke test (b.443).** Under a concurrent lifecycle
    /// driver, every payload the assembler emits should reflect ONE instant
    /// of the keyspace: never empty, and never naming a layer the driver did
    /// not create.
    ///
    /// The driver keeps the session non-empty at every instant — it creates
    /// the next layer BEFORE deleting the previous one — so an atomic read
    /// can never produce an empty layer set, whereas the pre-b.443
    /// `SMEMBERS` → per-member `GET` sequence could: read the index while it
    /// held only `{A}`, have the driver create `B` and delete `A` (DELing
    /// A's mapping) before the `GET`s run, and the payload comes back EMPTY
    /// — a layer set that existed at no instant at all. Restoring that
    /// payload re-creates a session with no layer mappings, and
    /// `apply_checkpoint`'s SR-2.4 guard then refuses every one of that
    /// session's layer checkpoints.
    ///
    /// **Calibration — read this before trusting it as a regression gate.**
    /// It is a *consistency smoke test*, not a reliable detector of the torn
    /// read. Falsifying it (reverting the assembler to the non-atomic
    /// two-phase read) did not fail it as written: the window between the
    /// index read and the mapping reads had to be widened by ~2 ms before
    /// the empty payload appeared. Un-widened, the driver rarely lands its
    /// create+delete inside a window that narrow. So a PASS here is weak
    /// evidence; the real gate on atomicity is that the read is ONE
    /// `EVALSHA` — see `read_session_state`, whose script has no interleave
    /// point to race against. What this test does buy cheaply is the
    /// never-empty / only-known-names invariant holding across a few hundred
    /// concurrent reads.
    ///
    /// Bounded iterations, no sleeps — the driver and the reader race
    /// naturally on a multi-thread runtime, and the reader re-reads until
    /// the driver signals it is done.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_lifecycle_never_yields_a_torn_payload() {
        /// Layer-name generations the driver churns through. Enough to
        /// interleave many times against the reader; small enough that the
        /// test stays sub-second.
        const GENERATIONS: usize = 40;

        let tv = TestValkey::setup().await;
        let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let ext_store: Arc<MockExternalValueStore> = Arc::new(MockExternalValueStore::new());

        let session_id = ss
            .create("cut-stage", "/stage/cut", "cut-user")
            .await
            .expect("session creation must succeed");

        // Seed generation 0 so the session is non-empty from the start.
        let layer_name = |gen: usize| format!("gen-{gen}.usda");
        ls.create_layer(
            &session_id,
            &layer_name(0),
            "cut-user",
            &fresh_state(),
            ext_store.clone(),
        )
        .await
        .expect("seed create_layer must succeed");

        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let driver = {
            let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
            let session_id = session_id.clone();
            let ext_store = ext_store.clone();
            let done = Arc::clone(&done);
            tokio::spawn(async move {
                for gen in 1..GENERATIONS {
                    // CREATE the new generation before DELETING the old one:
                    // the session is never empty, which is what makes the
                    // "payload is never empty" assertion sound.
                    ls.create_layer(
                        &session_id,
                        &layer_name(gen),
                        "cut-user",
                        &fresh_state(),
                        ext_store.clone(),
                    )
                    .await
                    .expect("driver create_layer must succeed");
                    ls.delete_layer(&session_id, &layer_name(gen - 1), ext_store.clone())
                        .await
                        .expect("driver delete_layer must succeed");
                }
                done.store(true, std::sync::atomic::Ordering::Release);
            })
        };

        let valid_names: std::collections::HashSet<String> =
            (0..GENERATIONS).map(layer_name).collect();
        let mut reads = 0usize;
        while !done.load(std::sync::atomic::Ordering::Acquire) {
            let checkpoint = build_session_checkpoint(tv.client(), tv.key_schema(), &session_id)
                .await
                .expect("assembler must succeed")
                .expect("session must exist for the whole run")
                .checkpoint;
            reads += 1;

            assert!(
                !checkpoint.layer_mappings.is_empty(),
                "torn payload: the driver keeps at least one layer live at every instant, so an \
                 ATOMIC read can never observe an empty layer set. An empty payload here means \
                 the read spanned two instants — index read before a delete, mappings read \
                 after — and restoring it would produce a session whose every layer checkpoint \
                 SR-2.4 then refuses. (read #{reads})"
            );
            for name in checkpoint.layer_mappings.keys() {
                assert!(
                    valid_names.contains(name),
                    "payload names '{name}', which the driver never created (read #{reads})"
                );
            }
        }

        driver.await.expect("driver task must not panic");
        assert!(
            reads > 0,
            "the reader must have observed at least one payload while the driver ran — zero \
             reads would mean this test asserted nothing"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn build_session_checkpoint_corrupt_layer_mapping_returns_permanent() {
        let tv = TestValkey::setup().await;
        let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let ext_store: Arc<MockExternalValueStore> = Arc::new(MockExternalValueStore::new());

        let session_id = ss
            .create("corrupt-stage", "/stage/corrupt", "corrupt-user")
            .await
            .expect("session creation must succeed");
        ls.create_layer(
            &session_id,
            "corrupt.usda",
            "corrupt-user",
            &fresh_state(),
            ext_store.clone(),
        )
        .await
        .expect("create_layer must succeed");

        // Overwrite the mapping value with a non-numeric string.
        let mapping_key = tv.key_schema().layer_mapping(&session_id, "corrupt.usda");
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key, "not-a-u64", None, None, false)
            .await
            .expect("SET corrupt mapping must succeed");

        let err = build_session_checkpoint(tv.client(), tv.key_schema(), &session_id)
            .await
            .expect_err("corrupt mapping must surface a Permanent error");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "expected Permanent for non-numeric layer_mapping, got {err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn build_session_checkpoint_nul_in_decoded_layer_name_returns_permanent() {
        let tv = TestValkey::setup().await;
        let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

        let session_id = ss
            .create("nul-stage", "/stage/nul", "nul-user")
            .await
            .expect("session creation must succeed");

        // Bypass `validate_session_name`'s upstream NUL rejection by writing
        // the hex-encoded NUL-bearing name directly into the layers index
        // and seeding a valid u64 at the matching layer-mapping key so the
        // GET succeeds (forcing the failure to come from the assembler-side
        // NUL check rather than a parse error).
        let nul_name = "bad\u{0000}name";
        let hex_nul_name = hex_encode(nul_name);
        let layers_index_key = tv.key_schema().layers_index(&session_id);
        let _: () = tv
            .client()
            .pool
            .sadd(&layers_index_key, &hex_nul_name)
            .await
            .expect("SADD hex-encoded NUL-bearing layer name must succeed");
        let mapping_key = tv.key_schema().layer_mapping(&session_id, nul_name);
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key, "12345", None, None, false)
            .await
            .expect("SET layer mapping must succeed");

        let err = build_session_checkpoint(tv.client(), tv.key_schema(), &session_id)
            .await
            .expect_err("NUL in decoded layer name must surface a Permanent error");
        match &err {
            CheckpointError::Permanent { source } => {
                let msg = source.to_string();
                assert!(
                    msg.contains(&session_id),
                    "error must name the offending session_id; got: {msg}"
                );
                assert!(
                    msg.contains("\\0"),
                    "error must render the embedded NUL as '\\0'; got: {msg}"
                );
                assert!(
                    msg.contains("bad") && msg.contains("name"),
                    "error must include the offending layer name; got: {msg}"
                );
            }
            other => panic!("expected Permanent for NUL-bearing layer name, got {other:?}"),
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Session-recovery applier (`apply_session_checkpoint`) — SR-2.7 inline
    // tests for Subtask `t3.fgj.7v.32.n4`.
    //
    // These tests construct a `SessionCheckpoint` directly and apply it
    // against a fresh `TestValkey::setup()`. They deliberately do NOT call
    // `build_session_checkpoint` (the Task 3 assembler) — coupling this
    // Subtask's tests to a sibling Subtask's wire-up would create a
    // landing-order dependency that the Subtask body explicitly forbids.
    //
    // Re-read uses raw `HGETALL` / `SMEMBERS` / `GET` calls against
    // `TestValkey::client().pool` to assert byte-equality between input and
    // post-apply state.
    // ───────────────────────────────────────────────────────────────────────

    use std::collections::{BTreeMap, BTreeSet, VecDeque};

    use crate::checkpoint::session::{
        apply_session_checkpoint, clear_apply_step_fault_injectors_for_test,
        set_apply_step_fault_injector_for_test, ApplyStep,
    };
    use crate::storage::keys::hex_encode as ks_hex_encode;

    /// Serializes the tests below that drive `apply_session_checkpoint` through
    /// the process-global `APPLY_STEP_FAULT_INJECTORS` seam.
    ///
    /// The injector is process-wide (the applier is a free function, not a
    /// struct that could own per-instance state), so under `cargo test`'s
    /// in-process parallelism one test's injected Step B fault leaks into a
    /// concurrently-running sibling — producing the contradictory failures
    /// observed in CI: `apply_session_checkpoint_step_b_injected_failure...`
    /// panicking "Step B fault must surface as Err" (its queued fault was
    /// consumed by a peer's apply first), and a peer panicking "first apply must
    /// succeed: Transient { ... injected ... }" (it consumed a fault it never
    /// requested). Clearing the queue at the top of each test is insufficient
    /// under parallelism — the window between clear and the awaited apply still
    /// races peers.
    ///
    /// Holding this guard for each test's full duration restores serial
    /// fault-injection semantics. `cargo nextest`'s process-per-test isolation
    /// makes this moot (each test gets a fresh static), but it keeps plain
    /// `cargo test` — the orchestrator's Task/Issue-boundary run — correct.
    /// A tokio `Mutex` (not `std`) so the guard can be held across `.await`
    /// without the `clippy::await_holding_lock` hazard, and so a panicking test
    /// does not poison the lock for the rest of the binary.
    static APPLY_FAULT_SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// Build a fully-populated `SessionCheckpoint` for `session_id` with two
    /// layer mappings. Mirrors the shape an assembler would emit but is
    /// constructed by hand to keep this Subtask's tests independent of
    /// `build_session_checkpoint`'s landing order.
    ///
    /// The single `path_name_members` entry is the `session_name\0session_id`
    /// composite — matches what `SessionStore::create_once` writes on the
    /// live path (`src/storage/session.rs:308`) and what
    /// `build_session_checkpoint` now emits (exactly one composite per
    /// session, irrespective of layer count).
    fn sample_apply_checkpoint(session_id: &str) -> SessionCheckpoint {
        let base_path = "/stage/apply".to_string();
        let session_name = "apply-stage";
        let mut sc = SessionCheckpoint {
            schema_version: SESSION_CHECKPOINT_SCHEMA_VERSION,
            session_id: session_id.to_string(),
            name: session_name.to_string(),
            base_path: base_path.clone(),
            created_by: "apply-user".to_string(),
            created_at: 1_700_000_000_000_000,
            layer_mappings: BTreeMap::from([
                ("root.usda".to_string(), 1001_u64),
                ("sublayer.usda".to_string(), 1002_u64),
            ]),
            layers_index: BTreeSet::from(["root.usda".to_string(), "sublayer.usda".to_string()]),
            path_name_members: BTreeSet::new(),
            path_index_session_id_member: session_id.to_string(),
        };
        // Compose the `session_name\0session_id` composite via the
        // sanctioned wrapper. NEVER inline-`format!` the NUL separator.
        sc.add_path_name_member(session_name, session_id);
        sc
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_session_checkpoint_round_trip_materializes_all_keys() {
        let _serial = APPLY_FAULT_SERIAL.lock().await;
        clear_apply_step_fault_injectors_for_test();
        let tv = TestValkey::setup().await;
        let session_id = format!("apply-rt-{}", uuid::Uuid::new_v4());
        let sc = sample_apply_checkpoint(&session_id);

        apply_session_checkpoint(tv.client(), tv.key_schema(), &session_id, &sc)
            .await
            .expect("apply must succeed against fresh keyspace");

        // (1) Path-index legacy set: session_id is a member.
        let path_index_key = tv.key_schema().path_index(&sc.base_path);
        let path_index_members: Vec<String> = tv
            .client()
            .pool
            .smembers(&path_index_key)
            .await
            .expect("SMEMBERS path_index");
        assert!(
            path_index_members.contains(&session_id),
            "path_index must carry session_id; got {path_index_members:?}"
        );

        // (2) Path-names composite set: both composites are members.
        let path_names_key = tv.key_schema().path_names_index(&sc.base_path);
        let path_names_members: Vec<String> = tv
            .client()
            .pool
            .smembers(&path_names_key)
            .await
            .expect("SMEMBERS path_names");
        for expected in &sc.path_name_members {
            assert!(
                path_names_members.contains(expected),
                "path_names must carry composite {expected:?}; got {path_names_members:?}"
            );
        }

        // (3) session-meta four-field HSET: name, base_path, created_by,
        //     created_at. Field-for-field byte-equality with the input.
        let session_meta_key = tv.key_schema().session_meta(&session_id);
        let meta: HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(&session_meta_key)
            .await
            .expect("HGETALL session_meta");
        assert_eq!(meta.get("name"), Some(&sc.name));
        assert_eq!(meta.get("base_path"), Some(&sc.base_path));
        assert_eq!(meta.get("created_by"), Some(&sc.created_by));
        assert_eq!(meta.get("created_at"), Some(&sc.created_at.to_string()));
        assert_eq!(
            meta.len(),
            4,
            "session_meta must carry exactly the four sanctioned fields, got {meta:?}"
        );

        // (4) Layer-mapping SETs: per-layer GET returns the expected u64
        //     rendered as a decimal string.
        for (name, expected_oid) in &sc.layer_mappings {
            let mapping_key = tv.key_schema().layer_mapping(&session_id, name);
            let raw: Option<String> = tv
                .client()
                .pool
                .get(&mapping_key)
                .await
                .expect("GET layer_mapping");
            assert_eq!(
                raw.as_deref(),
                Some(expected_oid.to_string().as_str()),
                "layer_mapping for {name} must equal {expected_oid}"
            );
        }

        // (5) Layers-index SADDs: members are HEX-ENCODED layer names
        //     (matches LayerStore::create_layer's encoding). Direct
        //     comparison against `hex_encode(name)` pins this contract.
        let layers_index_key = tv.key_schema().layers_index(&session_id);
        let layers_members: Vec<String> = tv
            .client()
            .pool
            .smembers(&layers_index_key)
            .await
            .expect("SMEMBERS layers_index");
        for name in &sc.layers_index {
            let hex = ks_hex_encode(name);
            assert!(
                layers_members.contains(&hex),
                "layers_index must carry hex-encoded {name:?} (= {hex:?}); got {layers_members:?}"
            );
        }
        assert_eq!(
            layers_members.len(),
            sc.layers_index.len(),
            "layers_index member count must equal input layers_index size"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_session_checkpoint_idempotent_re_apply_converges_to_same_state() {
        let _serial = APPLY_FAULT_SERIAL.lock().await;
        clear_apply_step_fault_injectors_for_test();
        let tv = TestValkey::setup().await;
        let session_id = format!("apply-idem-{}", uuid::Uuid::new_v4());
        let sc = sample_apply_checkpoint(&session_id);

        // First apply.
        apply_session_checkpoint(tv.client(), tv.key_schema(), &session_id, &sc)
            .await
            .expect("first apply must succeed");

        // Snapshot the post-apply state.
        let path_index_key = tv.key_schema().path_index(&sc.base_path);
        let path_names_key = tv.key_schema().path_names_index(&sc.base_path);
        let session_meta_key = tv.key_schema().session_meta(&session_id);
        let layers_index_key = tv.key_schema().layers_index(&session_id);

        let path_index_after_1: Vec<String> = tv
            .client()
            .pool
            .smembers(&path_index_key)
            .await
            .expect("SMEMBERS path_index #1");
        let path_names_after_1: Vec<String> = tv
            .client()
            .pool
            .smembers(&path_names_key)
            .await
            .expect("SMEMBERS path_names #1");
        let meta_after_1: HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(&session_meta_key)
            .await
            .expect("HGETALL session_meta #1");
        let layers_after_1: Vec<String> = tv
            .client()
            .pool
            .smembers(&layers_index_key)
            .await
            .expect("SMEMBERS layers_index #1");

        // Second apply — must succeed, must not error.
        apply_session_checkpoint(tv.client(), tv.key_schema(), &session_id, &sc)
            .await
            .expect("second apply must succeed (idempotency)");

        // Compare post-apply #2 state byte-for-byte against #1. Sets are
        // sorted to defuse non-deterministic SMEMBERS ordering across
        // Valkey versions.
        let mut path_index_after_2: Vec<String> = tv
            .client()
            .pool
            .smembers(&path_index_key)
            .await
            .expect("SMEMBERS path_index #2");
        let mut path_names_after_2: Vec<String> = tv
            .client()
            .pool
            .smembers(&path_names_key)
            .await
            .expect("SMEMBERS path_names #2");
        let meta_after_2: HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(&session_meta_key)
            .await
            .expect("HGETALL session_meta #2");
        let mut layers_after_2: Vec<String> = tv
            .client()
            .pool
            .smembers(&layers_index_key)
            .await
            .expect("SMEMBERS layers_index #2");

        let mut path_index_after_1_sorted = path_index_after_1;
        let mut path_names_after_1_sorted = path_names_after_1;
        let mut layers_after_1_sorted = layers_after_1;
        path_index_after_1_sorted.sort();
        path_names_after_1_sorted.sort();
        layers_after_1_sorted.sort();
        path_index_after_2.sort();
        path_names_after_2.sort();
        layers_after_2.sort();

        assert_eq!(
            path_index_after_2, path_index_after_1_sorted,
            "path_index members must be byte-identical across re-apply"
        );
        assert_eq!(
            path_names_after_2, path_names_after_1_sorted,
            "path_names members must be byte-identical across re-apply"
        );
        assert_eq!(
            meta_after_2, meta_after_1,
            "session_meta must be byte-identical across re-apply"
        );
        assert_eq!(
            layers_after_2, layers_after_1_sorted,
            "layers_index members must be byte-identical across re-apply"
        );

        // And the layer-mapping GETs return the same values.
        for (name, expected_oid) in &sc.layer_mappings {
            let mapping_key = tv.key_schema().layer_mapping(&session_id, name);
            let raw: Option<String> = tv
                .client()
                .pool
                .get(&mapping_key)
                .await
                .expect("GET layer_mapping #2");
            assert_eq!(
                raw.as_deref(),
                Some(expected_oid.to_string().as_str()),
                "layer_mapping {name} must persist across re-apply"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_session_checkpoint_step_b_injected_failure_leaves_step_a_committed() {
        // Partial-failure scenario from the Subtask body: Step A succeeds,
        // Step B fails via fault-injection. Post-condition is the
        // "ghost-but-functional" path-index half is present and the
        // session-scoped half is absent — exactly the recoverable
        // ghost-member state SR-2.7 self-heals via ListSessions' opportunistic
        // cleanup.
        let _serial = APPLY_FAULT_SERIAL.lock().await;
        clear_apply_step_fault_injectors_for_test();
        let tv = TestValkey::setup().await;
        let session_id = format!("apply-stepb-fail-{}", uuid::Uuid::new_v4());
        let sc = sample_apply_checkpoint(&session_id);

        // Install a single Step B failure.
        let mut q: VecDeque<CheckpointError> = VecDeque::new();
        q.push_back(CheckpointError::Transient {
            source: "injected: simulated Step B EXEC failure".into(),
        });
        set_apply_step_fault_injector_for_test(ApplyStep::StepB, q);

        // Apply — must return Transient.
        let err = apply_session_checkpoint(tv.client(), tv.key_schema(), &session_id, &sc)
            .await
            .expect_err("Step B fault must surface as Err");
        assert!(
            matches!(err, CheckpointError::Transient { .. }),
            "Step B injection must propagate as Transient; got {err:?}"
        );

        // Step A's path-index writes MUST have committed.
        let path_index_key = tv.key_schema().path_index(&sc.base_path);
        let path_index_members: Vec<String> = tv
            .client()
            .pool
            .smembers(&path_index_key)
            .await
            .expect("SMEMBERS path_index post-fail");
        assert!(
            path_index_members.contains(&session_id),
            "Step A must commit before Step B injection fires; path_index missing session_id"
        );

        let path_names_key = tv.key_schema().path_names_index(&sc.base_path);
        let path_names_members: Vec<String> = tv
            .client()
            .pool
            .smembers(&path_names_key)
            .await
            .expect("SMEMBERS path_names post-fail");
        for expected in &sc.path_name_members {
            assert!(
                path_names_members.contains(expected),
                "Step A path_names must commit; missing {expected:?}"
            );
        }

        // Step B's session-scoped writes MUST be absent.
        let session_meta_key = tv.key_schema().session_meta(&session_id);
        let meta: HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(&session_meta_key)
            .await
            .expect("HGETALL session_meta post-fail");
        assert!(
            meta.is_empty(),
            "Step B must not commit when injected to fail; got meta {meta:?}"
        );

        let layers_index_key = tv.key_schema().layers_index(&session_id);
        let layers_members: Vec<String> = tv
            .client()
            .pool
            .smembers(&layers_index_key)
            .await
            .expect("SMEMBERS layers_index post-fail");
        assert!(
            layers_members.is_empty(),
            "Step B must not commit when injected to fail; got layers_index {layers_members:?}"
        );

        for name in sc.layer_mappings.keys() {
            let mapping_key = tv.key_schema().layer_mapping(&session_id, name);
            let raw: Option<String> = tv
                .client()
                .pool
                .get(&mapping_key)
                .await
                .expect("GET layer_mapping post-fail");
            assert!(
                raw.is_none(),
                "Step B must not commit; layer_mapping {name} was written: {raw:?}"
            );
        }

        // Drain to avoid cross-test bleed.
        clear_apply_step_fault_injectors_for_test();
    }

    // ───────────────────────────────────────────────────────────────────────
    // b.4t3 — slot-pinned `txn_pool` checkouts for the two applier steps.
    //
    // Five tests, each pinning a different part of the claim the two
    // `acquire_txn_for` call sites rest on:
    //
    //  1. the pins are ARITHMETICALLY sufficient — every key each step
    //     touches collapses to the pin key's slot (pure, no Valkey);
    //  2. the brace-hostile fixture that arithmetic is computed over is a
    //     checkpoint the applier can actually COMMIT — same fixture, driven
    //     end to end against a live Valkey (without this, #1 could be proving
    //     a property of an unrestorable checkpoint);
    //  3. the one Step A branch where the pin key is NOT among the keys the
    //     transaction queues is reachable and still commits;
    //  4. the probe those pins steer actually RUNS on the applier's own
    //     checkouts, each checkout is slot-PINNED (`pinned=true`), and a dirty
    //     slot is repaired rather than inherited (behavioural, live Valkey +
    //     the probe fault seam);
    //  5. the blanket `Transient` wrap on the EXEC arms must NOT be routed
    //     through the storage lane's fred classifier.
    //
    // The source-shape half — "no unpinned checkout survives in this lane,
    // and both pins are still present" — is
    // `tests/integration/checkpoint/acquire_txn_pinning_fence_test.rs`.
    // ───────────────────────────────────────────────────────────────────────

    /// A checkpoint whose Step A and Step B key sets are both genuinely
    /// multi-key, so the slot-collapse assertion below has something to prove.
    ///
    /// Deliberately hostile in three ways [`sample_apply_checkpoint`] is not:
    ///
    /// - **three** path-name composites, not one, so `path_names_index` is
    ///   exercised alongside `path_index` under one pin;
    /// - **three** layer mappings plus a three-member `layers_index`, so Step B
    ///   queues five distinct keys behind its one pin;
    /// - a layer name and a base path each carrying literal `{` / `}`, which is
    ///   the shape that would break the pin if Valkey's hash tag were anything
    ///   other than *first `{` … first `}` after it*.
    fn brace_hostile_apply_checkpoint(session_id: &str) -> SessionCheckpoint {
        let session_name = "apply-{brace}-stage";
        let mut sc = SessionCheckpoint {
            schema_version: SESSION_CHECKPOINT_SCHEMA_VERSION,
            session_id: session_id.to_string(),
            name: session_name.to_string(),
            // Braces inside the base path: `path_index` hex-encodes it before
            // wrapping it in the tag, so these can never reach the key — but a
            // future encoding change that let them through would move the slot
            // out from under the pin, and this test would catch it.
            base_path: "/stage/{4t3}/apply".to_string(),
            created_by: "apply-user".to_string(),
            created_at: 1_700_000_000_000_000,
            layer_mappings: BTreeMap::from([
                ("root.usda".to_string(), 2001_u64),
                // A layer name whose braces would form a *second* tag if the
                // key were not hex-encoded and if Valkey scanned for anything
                // but the first pair.
                ("weird{name}.usda".to_string(), 2002_u64),
                ("deep/nested/sub.usda".to_string(), 2003_u64),
            ]),
            layers_index: BTreeSet::from([
                "root.usda".to_string(),
                "weird{name}.usda".to_string(),
                "deep/nested/sub.usda".to_string(),
            ]),
            path_name_members: BTreeSet::new(),
            path_index_session_id_member: session_id.to_string(),
        };
        // Three composites on the same base path — one for this session, two
        // standing in for siblings rooted at the same path. `add_path_name_member`
        // is the sanctioned constructor; never inline-`format!` the NUL.
        sc.add_path_name_member(session_name, session_id);
        sc.add_path_name_member("sibling-a", "sid-sibling-a");
        sc.add_path_name_member("sibling-b", "sid-sibling-b");
        sc
    }

    /// Each step's pin key names the slot EVERY key that step queues lands on.
    ///
    /// This is the arithmetic the two `acquire_txn_for` call sites rest on, and
    /// it is the half a behavioural test on a single-node fixture can never
    /// see: with one primary every routing decision is correct by accident, so
    /// a pin naming the wrong slot passes every other test in this file.
    ///
    /// `ClusterHash::from(&str)` is exactly the conversion `probe_command`
    /// applies to the `hash_key` argument (`storage::txn_checkout`), so
    /// comparing two `ClusterHash` values here compares the two slots fred's
    /// router would pick — the same idiom
    /// `txn_checkout::tests::probe_commands_pin_the_callers_slot` uses.
    ///
    /// Asserting *equality within a step* is the contract; the extra assertion
    /// that the two steps land on DIFFERENT slots is why there are two pins
    /// rather than one — a single checkout spanning both would be a CROSSSLOT
    /// transaction, which is precisely what SR-2.7's two-step split avoids.
    #[test]
    fn apply_session_checkpoint_step_pins_collapse_every_touched_key_to_one_slot() {
        use fred::types::ClusterHash;

        let keys = KeySchema::default();
        let session_id = "sid-4t3-{collapse}";
        let sc = brace_hostile_apply_checkpoint(session_id);

        // ── Step A: the pin, and every key the Step A MULTI/EXEC queues ──
        let step_a_pin = keys.path_index(&sc.base_path);
        let step_a_slot = ClusterHash::from(step_a_pin.as_str());
        assert!(
            matches!(step_a_slot, ClusterHash::Custom(_)),
            "a pinned checkout must resolve to a concrete slot, not a routing policy — \
             `ClusterHash::FirstKey` on the keyless DISCARD/UNWATCH probe is the whole defect \
             b.4t3 fixes. Got {step_a_slot:?}"
        );
        for touched in [
            keys.path_index(&sc.base_path),
            keys.path_names_index(&sc.base_path),
        ] {
            assert_eq!(
                ClusterHash::from(touched.as_str()),
                step_a_slot,
                "Step A queues `{touched}`, which must share the pin key `{step_a_pin}`'s slot. \
                 Both carry the `{{hex(base_path)}}` tag; if one ever stops, the checkout probe \
                 lands on a primary the transaction does not use and the pin is decorative. \
                 `path_names_index` matters twice over: on the degenerate branch where \
                 `path_index_session_id_member` is empty but `path_name_members` is not, it is \
                 the ONLY key Step A queues, and the pin key is then a sibling that the \
                 transaction never names — correct only because of this equality."
            );
        }

        // ── Step B: the pin, and every key the Step B MULTI/EXEC queues ──
        let step_b_pin = keys.session_meta(session_id);
        let step_b_slot = ClusterHash::from(step_b_pin.as_str());
        assert!(
            matches!(step_b_slot, ClusterHash::Custom(_)),
            "see the Step A rationale; got {step_b_slot:?}"
        );
        let mut step_b_touched = vec![keys.session_meta(session_id), keys.layers_index(session_id)];
        step_b_touched.extend(
            sc.layer_mappings
                .keys()
                .map(|name| keys.layer_mapping(session_id, name)),
        );
        for touched in step_b_touched {
            assert_eq!(
                ClusterHash::from(touched.as_str()),
                step_b_slot,
                "Step B queues `{touched}`, which must share the pin key `{step_b_pin}`'s slot. \
                 The layer-mapping key ends in a second `{{...}}`-looking group \
                 (`:layer:{{hex(name)}}` is hex, so it has no braces at all — and even a literal \
                 one would be inert, since Valkey's hash tag is the first `{{` up to the first \
                 `}}` after it). A key that fell outside this slot would make the Step B \
                 transaction CROSSSLOT, not merely mis-probed."
            );
        }

        // Two pins, not one: the steps are on different slots by construction.
        assert_ne!(
            step_a_slot, step_b_slot,
            "the path-index half (`{{hex(base_path)}}`) and the session-scoped half \
             (`{{session_id}}`) are different slots, which is exactly why SR-2.7 splits the \
             applier into two independently-atomic transactions with two separate pinned \
             checkouts. If these ever collapsed to one slot the split would still be correct, \
             but this fixture would have stopped testing anything — pick different fixture \
             values rather than deleting the assertion."
        );
    }

    /// The brace-hostile fixture is a checkpoint the applier can actually
    /// commit — round-tripped end to end, not just hashed.
    ///
    /// [`apply_session_checkpoint_step_pins_collapse_every_touched_key_to_one_slot`]
    /// computes `ClusterHash` values over [`brace_hostile_apply_checkpoint`]
    /// and never writes a key, which leaves that fixture unfalsifiable in one
    /// direction: if a brace-bearing layer name, a brace-bearing base path or
    /// the three-composite path-names set were rejected, mis-encoded or
    /// silently dropped by the applier, the arithmetic would still balance and
    /// the assertion would still pass — having proven a slot property of a
    /// checkpoint that cannot be restored at all.
    ///
    /// [`apply_session_checkpoint_round_trip_materializes_all_keys`] owns the
    /// round trip, but only for the tame two-layer
    /// [`sample_apply_checkpoint`]: one path-name composite, no braces
    /// anywhere, no nested layer name. This test closes the gap by driving the
    /// hostile fixture through `apply_session_checkpoint` and asserting the
    /// committed state key by key.
    ///
    /// The `session_id` carries braces of its own, so `{session_id}` is a tag
    /// with a nested brace pair — the shape that reaches `session_meta`,
    /// `layers_index` and every `layer_mapping` key.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_session_checkpoint_brace_hostile_checkpoint_round_trips() {
        let _serial = APPLY_FAULT_SERIAL.lock().await;
        clear_apply_step_fault_injectors_for_test();
        let tv = TestValkey::setup().await;
        let session_id = format!("apply-4t3-brace-{{{}}}", uuid::Uuid::new_v4());
        let sc = brace_hostile_apply_checkpoint(&session_id);

        apply_session_checkpoint(tv.client(), tv.key_schema(), &session_id, &sc)
            .await
            .expect("the brace-hostile checkpoint must apply against a fresh keyspace");

        // (1) Step A, path-names half: all THREE composites land. Two of them
        //     belong to sibling sessions rooted at the same base path, so this
        //     also pins that the applier writes the set it was handed rather
        //     than re-deriving one member from `session_id`.
        let path_names: Vec<String> = tv
            .client()
            .pool
            .smembers(&tv.key_schema().path_names_index(&sc.base_path))
            .await
            .expect("SMEMBERS path_names");
        assert_eq!(
            path_names.len(),
            3,
            "all three `name\\0session_id` composites must land, and nothing else; got \
             {path_names:?}"
        );
        for expected in &sc.path_name_members {
            assert!(
                path_names.contains(expected),
                "path_names must carry composite {expected:?}; got {path_names:?}"
            );
        }

        // (2) Step A, legacy half: the session-id member, on a base path whose
        //     literal braces are hex-encoded out of the key by `path_index`.
        let path_index: Vec<String> = tv
            .client()
            .pool
            .smembers(&tv.key_schema().path_index(&sc.base_path))
            .await
            .expect("SMEMBERS path_index");
        assert_eq!(
            path_index,
            vec![session_id.clone()],
            "path_index must carry exactly the checkpoint's session-id member"
        );

        // (3) Step B, session meta: the four sanctioned fields, byte-equal,
        //     including a session NAME that itself contains braces.
        let meta: HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(&tv.key_schema().session_meta(&session_id))
            .await
            .expect("HGETALL session_meta");
        assert_eq!(meta.get("name"), Some(&sc.name));
        assert_eq!(meta.get("base_path"), Some(&sc.base_path));
        assert_eq!(meta.get("created_by"), Some(&sc.created_by));
        assert_eq!(meta.get("created_at"), Some(&sc.created_at.to_string()));
        assert_eq!(
            meta.len(),
            4,
            "session_meta must carry exactly the four sanctioned fields, got {meta:?}"
        );

        // (4) Step B, layer mappings: every name is readable at the key
        //     `layer_mapping` derives for it — the brace-bearing and the
        //     slash-bearing one included. Asserting the key SUFFIX as well
        //     pins that the layer name reaches the key only hex-encoded: a
        //     raw `weird{name}.usda` in the key text would introduce a second
        //     brace pair, and while Valkey's tag is the first `{`…`}` (so the
        //     slot would survive), the key would no longer be the one every
        //     other reader of this mapping computes.
        for (name, expected_oid) in &sc.layer_mappings {
            let mapping_key = tv.key_schema().layer_mapping(&session_id, name);
            assert!(
                mapping_key.ends_with(&ks_hex_encode(name)),
                "layer_mapping key for {name:?} must end in the HEX of the name; got \
                 {mapping_key:?}"
            );
            let raw: Option<String> = tv
                .client()
                .pool
                .get(&mapping_key)
                .await
                .expect("GET layer_mapping");
            assert_eq!(
                raw.as_deref(),
                Some(expected_oid.to_string().as_str()),
                "layer_mapping for {name:?} must equal {expected_oid}"
            );
        }

        // (5) Step B, layers index: hex-encoded members only, one per layer.
        let layers: Vec<String> = tv
            .client()
            .pool
            .smembers(&tv.key_schema().layers_index(&session_id))
            .await
            .expect("SMEMBERS layers_index");
        assert_eq!(
            layers.len(),
            sc.layers_index.len(),
            "layers_index member count must equal the checkpoint's; got {layers:?}"
        );
        for name in &sc.layers_index {
            let hex = ks_hex_encode(name);
            assert!(
                layers.contains(&hex),
                "layers_index must carry hex-encoded {name:?} (= {hex:?}); got {layers:?}"
            );
            assert!(
                !layers.contains(name),
                "layers_index must NOT carry the raw layer name {name:?} — the encoding is what \
                 keeps a name containing `{{`, `}}` or `/` from changing the key it indexes; got \
                 {layers:?}"
            );
        }
    }

    /// The hygiene probe runs on the applier's OWN checkouts: a slot carrying
    /// leaked transaction state is detected, counted and repaired on each of the
    /// two steps, and the restore still lands.
    ///
    /// The checkpoint-lane analogue of
    /// `tests/integration/storage/txn_checkout_test.rs::pinned_txn_checkout_detects_a_leaked_multi_recycles_and_lets_the_borrower_commit`,
    /// and it exists because that test proves the mechanism on a checkout the
    /// *test* performs. Nothing there says the DR-restore applier's two
    /// checkouts go through the same path — before b.4t3 they went through the
    /// unpinned shim, and a future edit that reverted them, or bypassed the
    /// pool entirely, would leave every assertion over there green.
    ///
    /// Driven through the `ProbeFaultInjector` seam rather than by leaving a
    /// real `MULTI` open, for one reason: fred ships `MULTI`…`EXEC` as a single
    /// router command, so its transaction API cannot leave residue, and the raw
    /// `custom_raw` injection the storage-lane test uses needs a guard on the
    /// specific pooled connection — which this test, by construction, never
    /// holds (the applier owns both checkouts end to end). `ProbeFault::ok()` on
    /// `DISCARD` is the same verdict a real leaked `MULTI` produces: an
    /// acknowledged `DISCARD` is only reachable when a transaction was already
    /// open.
    ///
    /// **Two faults, one per step.** `remaining() == 0` afterwards is what makes
    /// the count attributable: an applier that took only one checkout, or none,
    /// would leave a fault queued.
    ///
    /// **`pinned=true` on both events is the b.4t3 regression assertion.** The
    /// counter and the queue drain identically over the unpinned shim — it
    /// probes too — so without this field a revert of the two call sites to
    /// `acquire_txn()` would leave every other assertion here green. That field
    /// is the ONLY observable separating the two entry points on a single-node
    /// fixture, which is why it is asserted despite the test-writing guide's
    /// "don't assert exact log messages" default; the storage-lane test asserts
    /// it for the same reason.
    ///
    /// `pool_size = 1`: one txn slot, so both steps borrow the same connection
    /// and the second checkout provably runs on the slot the first one repaired.
    ///
    /// Default (current-thread) runtime, deliberately: the capturing subscriber
    /// is installed with `tracing::subscriber::set_default`, which is
    /// thread-local, and the applier's WARNs are emitted inline by
    /// `acquire_txn_inner`. On a multi-thread runtime an `.await` could resume
    /// the applier on a worker with no subscriber and the capture would come
    /// back empty.
    #[tokio::test]
    async fn apply_session_checkpoint_dirty_slot_on_each_step_is_counted_and_repaired() {
        use crate::observability::metrics::test_util::{render, sample_value};
        use crate::storage::txn_checkout::{ProbeFault, ProbeFaultInjector};
        use std::sync::Mutex;
        use tracing_subscriber::fmt::MakeWriter;

        /// The dirty-checkout event, verbatim from
        /// `ValkeyClient::acquire_txn_inner`.
        const DIRTY_CHECKOUT_EVENT: &str =
            "txn pool checkout found the slot unfit for a transaction";

        /// `MakeWriter` capturing every byte into a shared buffer. Duplicated
        /// from `tests/integration/storage/txn_checkout_test.rs` (itself a copy
        /// of `src/grpc/auth_interceptor.rs::tests::CapturingWriter`) — the
        /// project pattern is to repeat these few lines rather than hoist them.
        #[derive(Clone)]
        struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

        impl<'a> MakeWriter<'a> for CapturingWriter {
            type Writer = CapturingWriter;
            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        impl std::io::Write for CapturingWriter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let _serial = APPLY_FAULT_SERIAL.lock().await;
        clear_apply_step_fault_injectors_for_test();

        let tv = TestValkey::setup_with_pool_size(1).await;
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only("b4t3"));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;

        let read_leaked_multi = || {
            sample_value(
                &render(&metrics),
                "live_edit_valkey_txn_dirty_checkouts_total",
                &[("reason", "leaked_multi")],
            )
            .unwrap_or(0.0) as u64
        };
        assert_eq!(
            read_leaked_multi(),
            0,
            "precondition: a freshly built registry pre-registers this cell at zero, so the \
             delta below is attributable to the two injected verdicts"
        );

        let injector = Arc::new(ProbeFaultInjector::new());
        tv.client()
            .install_probe_fault_injector_for_test(Arc::clone(&injector));
        // One per step. Queued on DISCARD: `+OK` from DISCARD IS the
        // leaked-MULTI verdict.
        injector.push_discard(ProbeFault::ok());
        injector.push_discard(ProbeFault::ok());

        let session_id = format!("apply-4t3-probe-{}", uuid::Uuid::new_v4());
        let sc = sample_apply_checkpoint(&session_id);

        let captured = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(CapturingWriter(Arc::clone(&captured)))
            .with_max_level(tracing::Level::WARN)
            // ANSI escapes would otherwise make the field match depend on
            // whether colour was auto-enabled.
            .with_ansi(false)
            .finish();
        let capture = tracing::subscriber::set_default(subscriber);

        // Act — the applier's own two checkouts.
        let applied =
            apply_session_checkpoint(tv.client(), tv.key_schema(), &session_id, &sc).await;

        // Stop capturing before the assertions so a panic's own output is not
        // swallowed by the buffer.
        let logged = {
            let buffer = captured.lock().expect("capture mutex is never poisoned");
            String::from_utf8_lossy(&buffer).into_owned()
        };
        drop(capture);

        applied.expect(
            "the restore must still land: a dirty slot is REPAIRED at checkout, not surfaced as \
             an apply failure",
        );

        // The b.4t3 assertion: one dirty-checkout event per step, each naming
        // the SLOT-PINNED probe form.
        let dirty_events: Vec<&str> = logged
            .lines()
            .filter(|line| line.contains(DIRTY_CHECKOUT_EVENT))
            .collect();
        assert_eq!(
            dirty_events.len(),
            EXPECTED_APPLIER_CHECKOUTS,
            "one dirty-checkout event per applier checkout. Captured output: {logged:?}"
        );
        for event in &dirty_events {
            assert!(
                event.contains("pinned=true"),
                "every `txn_pool` checkout the DR-restore applier takes must be SLOT-PINNED \
                 (`acquire_txn_for`). `pinned=false` means the checkout went through the \
                 `acquire_txn()` shim: the probe still runs, so the counter and the injected \
                 queue below drain identically and nothing else in this test can see the \
                 difference — but `DISCARD` / `UNWATCH` then carry no key, fred's \
                 `ClusterHash::FirstKey` finds nothing to hash, and on a cluster the router \
                 sends them to an arbitrary primary. The verdict would describe a connection \
                 this destructive restore transaction never runs on (b.4t3). Event: {event:?}"
            );
        }

        assert_eq!(
            injector.remaining(),
            0,
            "both queued verdicts must have been consumed — one per applier step. A leftover \
             fault means `apply_session_checkpoint` took fewer `txn_pool` checkouts than SR-2.7's \
             two-step split requires, so the probe is not covering every transaction it opens"
        );
        assert_eq!(
            read_leaked_multi(),
            EXPECTED_APPLIER_CHECKOUTS as u64,
            "each applier checkout that found the slot unfit must tick \
             live_edit_valkey_txn_dirty_checkouts_total{{reason=\"leaked_multi\"}} — this counter \
             is the only first-order evidence an operator gets that a DR restore ran on a \
             connection carrying another borrower's residue"
        );

        // And the restore actually landed on both halves — a repair that
        // wedged the connection would fail here rather than above.
        let path_index_members: Vec<String> = tv
            .client()
            .pool
            .smembers(&tv.key_schema().path_index(&sc.base_path))
            .await
            .expect("SMEMBERS path_index");
        assert!(
            path_index_members.contains(&session_id),
            "Step A must have committed on the repaired slot; got {path_index_members:?}"
        );
        let meta: HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(&tv.key_schema().session_meta(&session_id))
            .await
            .expect("HGETALL session_meta");
        assert_eq!(
            meta.get("base_path"),
            Some(&sc.base_path),
            "Step B must have committed on the repaired slot; got {meta:?}"
        );
    }

    /// The Step A branch where the pin key is a key the transaction never
    /// names: `path_name_members` present, `path_index_session_id_member`
    /// empty.
    ///
    /// This branch is reachable — the all-empty guard above it requires BOTH
    /// collections to be empty — and it is the one place the b.4t3 pin argument
    /// leans on sibling-key tag sharing rather than on the pin key being queued.
    /// `apply_session_checkpoint_step_pins_collapse_every_touched_key_to_one_slot`
    /// owns the slot arithmetic; this test owns the reachability and the write
    /// effect, so a future edit that made the branch queue nothing (or hit the
    /// malformed-checkpoint `debug_assert`) fails here rather than leaving the
    /// arithmetic assertion describing dead code.
    #[tokio::test]
    async fn apply_session_checkpoint_path_names_only_contribution_still_commits_step_a() {
        let _serial = APPLY_FAULT_SERIAL.lock().await;
        clear_apply_step_fault_injectors_for_test();
        let tv = TestValkey::setup().await;
        let session_id = format!("apply-4t3-namesonly-{}", uuid::Uuid::new_v4());

        let mut sc = sample_apply_checkpoint(&session_id);
        sc.path_index_session_id_member = String::new();
        assert!(
            !sc.path_name_members.is_empty(),
            "fixture precondition: only the legacy session-id member is cleared, so Step A still \
             has work and the all-empty malformed-checkpoint guard is NOT the branch under test"
        );

        apply_session_checkpoint(tv.client(), tv.key_schema(), &session_id, &sc)
            .await
            .expect("a path-names-only contribution is well-formed and must apply");

        let path_names: Vec<String> = tv
            .client()
            .pool
            .smembers(&tv.key_schema().path_names_index(&sc.base_path))
            .await
            .expect("SMEMBERS path_names");
        for expected in &sc.path_name_members {
            assert!(
                path_names.contains(expected),
                "the composite `{expected:?}` must land even though the pin key \
                 (`path_index`) is not among the keys this transaction queues; got {path_names:?}"
            );
        }

        let path_index: Vec<String> = tv
            .client()
            .pool
            .smembers(&tv.key_schema().path_index(&sc.base_path))
            .await
            .expect("SMEMBERS path_index");
        assert!(
            path_index.is_empty(),
            "the legacy session-id set must stay untouched when the checkpoint carries no member \
             for it — the applier must not re-derive one; got {path_index:?}"
        );
    }

    /// How many `txn_pool` checkouts one `apply_session_checkpoint` performs —
    /// one per SR-2.7 step. Restated from
    /// `tests/integration/checkpoint/acquire_txn_pinning_fence_test.rs`, which
    /// asserts the same number against the source text.
    const EXPECTED_APPLIER_CHECKOUTS: usize = 2;

    /// One verdict per applier checkout must stay inside `acquire_txn_inner`'s
    /// repair-ineffective breaker budget, or the last checkout is handed over
    /// UNREPAIRED and
    /// [`apply_session_checkpoint_dirty_slot_on_each_step_is_counted_and_repaired`]
    /// silently stops testing the repair it is named for.
    ///
    /// Read from production's own `pub(crate)` constant rather than restated —
    /// this module is in-crate, so there is no separate-crate excuse for a copy,
    /// and a compile-time assertion is the cheapest possible way to make a
    /// future tightening of that budget fail loudly at the fixture rather than
    /// quietly in the assertion.
    const _: () = assert!(
        EXPECTED_APPLIER_CHECKOUTS
            <= crate::storage::txn_checkout::TXN_STATE_PROOF_REPAIR_ATTEMPTS as usize
    );

    /// Both sides of reason 3 of `apply_session_checkpoint`'s `# Error
    /// classification` agree on fred's two transaction-router shapes: the
    /// shared storage classifier grades each transient (b.cyu, which folded
    /// in b.bm5 (a)), and the applier grades every `EXEC` error `Transient`
    /// on its own, by its blanket wrap. This test pins the classifier side,
    /// over the table the classifier itself keys on
    /// ([`TRANSACTION_ROUTER_EXHAUSTED_SHAPES`]); the applier side, a real
    /// fred `Transaction` redirected at its `EXEC` and graded `Transient` by
    /// the applier, is pinned through session recovery in
    /// `tests/integration/storage/b_cyu_txn_redirect_test.rs`
    /// (`the_session_checkpoint_applier_grades_a_redirected_exec_transient_as_the_classifier_does`).
    ///
    /// The two sides are kept apart on purpose: the wrap stays blanket so a
    /// joiner's retry does not depend on the shared classifier's next edit.
    /// If this test fails, the classifier dropped a shape the applier still
    /// retries — the other `txn_pool` callers (`DeleteSession`, the
    /// `CreateLayer` materialization) would answer a slot migration
    /// `INTERNAL` again.
    ///
    /// [`TRANSACTION_ROUTER_EXHAUSTED_SHAPES`]: crate::storage::client::TRANSACTION_ROUTER_EXHAUSTED_SHAPES
    #[test]
    fn the_storage_classifier_agrees_with_the_applier_that_the_fred_transaction_router_shapes_are_transient(
    ) {
        use crate::storage::client::{
            is_transient_fred_error, TRANSACTION_ROUTER_EXHAUSTED_SHAPES,
        };
        use fred::error::Error as FredError;

        let texts: Vec<&str> = TRANSACTION_ROUTER_EXHAUSTED_SHAPES
            .iter()
            .map(|(_, detail)| *detail)
            .collect();
        assert_eq!(
            texts,
            ["Max redirections exceeded", "Max attempts exceeded"],
            "the shapes this agreement covers are fred's two transaction-router texts"
        );
        for (kind, detail) in &TRANSACTION_ROUTER_EXHAUSTED_SHAPES {
            assert!(
                is_transient_fred_error(&FredError::new(kind.clone(), *detail)),
                "the storage classifier must grade fred's transaction-router shape ({kind:?} / \
                 {detail:?}) transient, as `apply_session_checkpoint`'s blanket wrap does: that \
                 shape is what a cluster reshard gives every txn-pool transaction"
            );
        }
    }

    /// A session-state read whose `EVALSHA` gets no reply within the command
    /// timeout counts exactly one
    /// `valkey_errors_total{pool="data", kind="command_timeout"}` and reports
    /// `Transient`. The read is held behind a server-side `BLPOP` on every
    /// data-pool connection, so the `Timeout` is fred's real one.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_session_state_timed_out_evalsha_counts_one_data_command_timeout() {
        use crate::storage::test_support::{
            command_timeout_row, command_timeouts_only_on, release_all,
        };
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only(
            "session-checkpoint",
        ));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;

        let stalls = tv.stall_data_pool().await;
        let read = read_session_state(tv.client(), tv.key_schema(), "stalled-session").await;
        release_all(stalls).await;

        assert!(
            matches!(read, Err(CheckpointError::Transient { .. })),
            "an EVALSHA that timed out is transient; got {:?}",
            read.err()
        );
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 1),
            "the timed-out EVALSHA must count once, on the data pool"
        );
    }
}
