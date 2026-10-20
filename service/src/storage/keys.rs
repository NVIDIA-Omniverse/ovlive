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

//! Key schema builder for all Valkey keys used by the Live Edit service.
//!
//! Every key starts with a configurable prefix (default `"le:"`). Key patterns
//! follow SDD §7.1 exactly, including Valkey Cluster hash tags `{id}` for
//! co-locating related keys on the same hash slot.
//!
//! Hex encoding (§7.3): layer names, field key names, base paths, and f64 time
//! values are hex-encoded in keys to avoid delimiter conflicts with `:`.

/// Parent id reserved for the synthetic root in the children-ordering lists
/// (`le:{oid}:obj:ch:{parent_id}`).
///
/// Section ids are non-zero, so `0` is unambiguous as the "top-level sections"
/// parent. It is deliberately NOT a member of `obj:secs` or `obj:dels` — it is
/// not a section — so index walks over those two sets never yield it.
///
/// Since b.jdr the root *is* an ordinary member of the parent-id index
/// [`KeySchema::obj_children_index`] alongside every other parent that owns a
/// children list, so the retirement inventory normally learns about it from
/// there. [`KeySchema::object_cleanup_keys`] nevertheless keeps emitting the
/// root key unconditionally (b.9fn), as rolling-deploy cover for objects
/// materialized by a pre-b.jdr replica — which wrote no index — and retired by
/// a post-b.jdr one.
pub(crate) const ROOT_PARENT_ID: &str = "0";

/// Hex-encode the raw UTF-8 bytes of a string.
///
/// Used for layer names, field key names, and base paths that may contain `:`
/// or other characters that would conflict with the key delimiter.
pub fn hex_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for byte in s.as_bytes() {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Hex-encode an f64 as its IEEE 754 bit pattern (16 hex digits).
///
/// Uses `f64::to_bits()` to get the u64 representation, then formats as
/// zero-padded lowercase hex. This produces deterministic, sortable-in-key
/// strings for time values.
///
/// Bit-exact on purpose: `-0.0` encodes as `8000000000000000` and `0.0` as
/// `0000000000000000`. This build never writes a value row at `-0.0` (every
/// time-sample time is canonicalized at `oodf_ffi::parse_delta`); a `-0.0`
/// key is named only to read, delete or merge a legacy row an older build
/// wrote — by the readers, by a delete of a stored row (including the reset's
/// and cascade's delete of both spellings of a section's time 0, rule E in
/// `domain::delta`), and by the reconciler's signed-zero merge
/// (`storage::time_sample_merge`) that folds it into `0.0`.
pub fn hex_encode_f64(v: f64) -> String {
    format!("{:016x}", v.to_bits())
}

/// Decode a hex-encoded UTF-8 string back to the original string.
///
/// Inverse of `hex_encode`. Returns an empty string if the hex is invalid.
///
/// b.k1r medium: emits a `warn` log when the input contains non-hex
/// characters or fails UTF-8 validation so on-wire corruption of a
/// layer-name / field-key / base-path is not silently hidden behind an
/// empty string. The fallback-to-empty is preserved to keep the callers
/// infallible — they feed the result into tracing output and key-member
/// parsing, both of which already tolerate empty strings.
pub fn hex_decode(hex: &str) -> String {
    // Odd-length strings or non-hex digits are signs of corruption: real
    // outputs of `hex_encode` are always even-length ASCII hex.
    let odd_length = !hex.len().is_multiple_of(2);
    let mut had_bad_pair = false;
    let bytes: Vec<u8> = (0..hex.len())
        .step_by(2)
        .filter_map(|i| {
            let Some(h) = hex.get(i..i + 2) else {
                had_bad_pair = true;
                return None;
            };
            match u8::from_str_radix(h, 16) {
                Ok(b) => Some(b),
                Err(_) => {
                    had_bad_pair = true;
                    None
                }
            }
        })
        .collect();
    if odd_length || had_bad_pair {
        tracing::warn!(
            input = hex,
            len = hex.len(),
            "hex_decode: input is not well-formed ASCII hex; returning best-effort prefix"
        );
    }
    match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(
                input = hex,
                error = %e,
                "hex_decode: decoded bytes are not valid UTF-8; returning empty string"
            );
            String::new()
        }
    }
}

/// Decode a hex-encoded f64 (16 hex digits = IEEE 754 bit pattern).
///
/// Inverse of `hex_encode_f64`.
///
/// b.k1r medium: logs a `warn` when the input is not parseable as 16 hex
/// digits (the shape [`hex_encode_f64`] always produces) so a corrupted
/// sorted-set member is visible in operator tracing instead of silently
/// mapping to `0.0` (the same f64 that `0.0` encodes to). The fallback
/// is preserved to keep callers infallible.
pub fn hex_decode_f64(hex: &str) -> f64 {
    match u64::from_str_radix(hex, 16) {
        Ok(bits) => f64::from_bits(bits),
        Err(e) => {
            tracing::warn!(
                input = hex,
                len = hex.len(),
                error = %e,
                "hex_decode_f64: input is not a 16-hex-digit IEEE-754 bit pattern; \
                 falling back to 0.0"
            );
            0.0
        }
    }
}

/// Convert a SHA-256 hash (32 bytes) to a lowercase hex string (64 chars).
///
/// Used for external value deduplication keys (`le:ext:meta:{hash_hex}`).
pub fn sha256_to_hex(hash: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in hash {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// TTL of the session deletion marker ([`KeySchema::session_deleted_marker`]):
/// 7 days.
///
/// The marker has to outlive the window in which a deleted session's
/// checkpoint can still be fetched, which closes when the DELETE that
/// `SessionStore::delete` enqueues on the session-checkpoint writer lands at
/// the adapter. That writer (`src/checkpoint/session_writer.rs`) bounds the
/// delay:
///
/// - **One operation.** `MAX_ADAPTER_ATTEMPTS` (3) adapter-phase attempts, each
///   one Valkey script that mints the stamp (and, for a write, reads the
///   session state; at most `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`, 5 s by
///   default) plus one adapter call (at most
///   `Config::checkpoint_adapter_call_ceiling`: on the StorageApi adapter 6 ×
///   `LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS`, so 180 s at the stock 30 s; on
///   the local adapters a fixed 5 s), then `MAX_PUBLISH_ATTEMPTS`
///   (3) publishes of one Valkey command each, with ~250 ms of backoff per
///   phase (`RETRY_INITIAL_DELAY`). Roughly 9.5 minutes at stock settings.
/// - **Queueing.** Dispatch is serialized per session, so the DELETE can wait
///   behind one in-flight operation and its repair chain, at most
///   `1 + MAX_PUBLISH_REPAIRS` (4) operations, and can then run a chain of its
///   own of the same length. A pending DELETE supersedes a pending write, so
///   nothing else queues ahead of it.
///
/// That totals at most 8 operations, about 76 minutes at stock settings
/// on StorageApi (`8 × (3 × (5 s + 6T) + 3 × 5 s + 0.5 s)`, T the request
/// timeout). Seven days is ~130× that stock bound, and stays above it for any
/// T up to ~70 minutes.
///
/// Why the TTL can be this generous: session ids are never reused (every
/// `CreateSession` mints a fresh UUID), so a marker can only ever block the
/// recovery of an id that is gone for good. The cost is one small string key
/// per deleted session for 7 days.
///
/// A DELETE the writer abandons (a permanent adapter error, three transient
/// ones, or a process killed before it dispatched) leaves the checkpoint in
/// place with nothing left to retry it. The marker blocks recovery of that
/// session until it expires or is lost with the keyspace (the loss session
/// recovery mostly runs after), and no longer; no TTL closes that case.
pub const SESSION_DELETED_MARKER_TTL_SECS: i64 = 7 * 24 * 60 * 60;

/// Lifetime, in milliseconds, of a [`KeySchema::create_layer_operation`]
/// record: 24 hours, fixed and deliberately not configurable.
///
/// The record exists so a client re-send of a `CreateLayer` whose reply was
/// lost is answered with the original `object_id` instead of creating the
/// layer again, so it only has to outlive the client's retry window. Kit's
/// client library spends 7.55–15.1 s of backoff across its 11 attempts, and
/// the C++ client library re-sends within its own endpoint lifetime; a day is several
/// orders of magnitude past either. The cost is one short string key per
/// id-carrying create for a day.
///
/// TTL is the record's ONLY removal path (see the key builder for why), so
/// this bound is also the bound on how long a deleted session's records
/// linger.
pub const CREATE_LAYER_OPERATION_TTL_MS: u64 = 86_400_000;

/// Lifetime, in milliseconds, of a [`KeySchema::create_layer_claim`] that
/// reads [`CREATE_LAYER_CLAIM_VETOED`]: 24 hours, fixed and deliberately not
/// configurable.
///
/// The veto only has to outlive every commit of its object that could still
/// run: a `CreateLayer` that materialized the object and is still before its
/// commit point, its in-process finisher, or a commit script already sent and
/// still queued on the store. Each of those ends within the sweep grace (one
/// or two RPC deadlines, `orphan_sweep::orphan_sweep_grace`) of the object's
/// creation, and the sweep vetoes only objects older than that grace; a day is
/// orders of magnitude past it. The cost is one short string key per orphan
/// the sweep retires, for a day.
pub const CREATE_LAYER_CLAIM_VETOED_TTL_MS: u64 = 86_400_000;

/// The floor of a `committed` claim's lifetime
/// ([`create_layer_claim_committed_ttl_ms`]): 10 minutes.
const CREATE_LAYER_CLAIM_COMMITTED_TTL_FLOOR_MS: u64 = 600_000;

/// Lifetime, in milliseconds, of a [`KeySchema::create_layer_claim`] that
/// reads [`CREATE_LAYER_CLAIM_COMMITTED`]: `max(10 min, 2 × grace)`, where
/// `grace` is the orphan sweep grace (`orphan_sweep::orphan_sweep_grace`).
/// Derived, never configured.
///
/// The claim only has to outlive the commit runs that could still try to map
/// its object again: the object's own `CreateLayer` call and its in-process
/// finisher. Both stop by the object's `created_at` plus the grace, on that
/// replica's own clock, and the claim is written after `created_at`, so no
/// clock skew between replicas enters the bound. The orphan sweep's completion
/// of a stranded overwrite runs only after that grace and writes the claim
/// itself when it lands. The cost is one short string key per committed
/// create, for at least ten minutes.
pub fn create_layer_claim_committed_ttl_ms(grace: std::time::Duration) -> u64 {
    let twice_grace = u64::try_from(grace.as_millis())
        .unwrap_or(u64::MAX)
        .saturating_mul(2);
    CREATE_LAYER_CLAIM_COMMITTED_TTL_FLOOR_MS.max(twice_grace)
}

/// The value of a [`KeySchema::create_layer_claim`] a commit-script run wrote
/// when it mapped the object.
pub const CREATE_LAYER_CLAIM_COMMITTED: &str = "committed";

/// The value of a [`KeySchema::create_layer_claim`] the orphan sweep's veto
/// wrote before it retired the object.
pub const CREATE_LAYER_CLAIM_VETOED: &str = "vetoed";

/// The largest commit release ring TTL a store passes to the update script:
/// twelve decimal digits, the most `update_layer_fast.lua`'s `ENCODING` check
/// on `ARGV[16]` accepts (about 31.7 years). Far below `i64::MAX`, so
/// `PEXPIRE` can never overflow on it.
pub(crate) const COMMIT_RELEASE_RING_TTL_MAX_MS: u64 = 999_999_999_999;

/// The part of the commit release ring TTL beyond the RPC deadline and the
/// Valkey command timeout ([`commit_release_ring_ttl`]).
const COMMIT_RELEASE_RING_TTL_MARGIN_MS: u64 = 60_000;

/// The TTL, in ms, of the per-object commit release ring
/// ([`KeySchema::obj_write_releases`]): `rpc_deadline_ms`
/// (`LIVE_EDIT_RPC_DEADLINE_MS`) + `valkey_command_timeout_ms`
/// (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`) + 60 s, saturating, and clamped to
/// `[1, 999_999_999_999]` so it is never `0` and always fits the script's
/// check. 95,000 ms at the defaults (30 s and 5 s). Every commit that writes
/// the ring sets it, so the key expires that long after the object's last
/// releasing commit. Pure.
///
/// **Derivation.** An entry is read only by the call that wrote it, within
/// that call ([`KeySchema::obj_write_releases`]), and a call cannot outlive
/// its RPC's deadline: the server drops the handler at
/// `LIVE_EDIT_RPC_DEADLINE_MS`, or at a shorter client `grpc-timeout`
/// (`grpc::deadline`). The commit that writes the entry runs within that
/// deadline too, so the deadline alone bounds the gap from a write to its
/// read on one clock. The command timeout covers the reader's own `HGET`,
/// issued before the deadline and answered after it. The 60 s covers the
/// rest: the store's wall clock against the replica's monotonic deadline, and
/// a failover onto a primary whose clock differs (a key's TTL is an absolute
/// expiry on the primary's clock, which a promoted replica keeps). 60 s is an
/// internal choice, not a measured bound. A ring that expires before its
/// reader only leaks: the lookup reads it as absent and enqueues nothing,
/// and the reference-count reconciler repairs what was owed.
///
/// Derived from configuration rather than fixed in the script, so a
/// configured deadline longer than the default lengthens it with it.
pub fn commit_release_ring_ttl(rpc_deadline_ms: u64, valkey_command_timeout_ms: u64) -> u64 {
    rpc_deadline_ms
        .saturating_add(valkey_command_timeout_ms)
        .saturating_add(COMMIT_RELEASE_RING_TTL_MARGIN_MS)
        .clamp(1, COMMIT_RELEASE_RING_TTL_MAX_MS)
}

/// Builder for Valkey keys following the schema in SDD §7.1.
///
/// All methods return owned `String` keys. The struct is cheap to clone and
/// share across async tasks.
#[derive(Clone, Debug)]
pub struct KeySchema {
    prefix: String,
}

impl Default for KeySchema {
    fn default() -> Self {
        Self {
            prefix: "le:".to_owned(),
        }
    }
}

impl KeySchema {
    /// Create a `KeySchema` with a custom prefix.
    ///
    /// Tests use a random prefix for isolation against a shared Valkey instance.
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_owned(),
        }
    }

    /// Return the raw prefix string (e.g. `"le:"`).
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    // ── Session-scoped keys (hash tag: {session_id}) ────────────────────

    /// Session metadata hash.
    /// Pattern: `le:{sid}:sess`
    pub fn session_meta(&self, session_id: &str) -> String {
        format!("{}{{{session_id}}}:sess", self.prefix)
    }

    /// Session deletion marker: a plain string key written by
    /// `SessionStore::delete` in the same MULTI/EXEC that deletes
    /// [`Self::session_meta`] and [`Self::layers_index`], with a TTL of
    /// [`SESSION_DELETED_MARKER_TTL_SECS`].
    ///
    /// Pattern: `le:{sid}:sess:deleted`
    ///
    /// `SessionRecoveryHook::try_recover_session` reads it before it fetches a
    /// session checkpoint and treats a present marker as "no checkpoint". It
    /// closes the window between a `DeleteSession` removing the session from
    /// the live store and that session's checkpoint tombstone landing at the
    /// adapter (`SessionCheckpointWriter::enqueue_session_checkpoint_delete`
    /// runs off the hot path). Without the marker a lookup in that window
    /// finds the session absent, fetches the not-yet-deleted checkpoint and
    /// restores the session that was just deleted.
    ///
    /// The two cancellers that find a session missing from the store
    /// (`acquire_session_cancel` and the `session_tokens` sweeper) read it too,
    /// through `SessionStore::deleted_marker_exists`, to end a deleted
    /// session's streams with the deletion endings rather than the retryable
    /// ones meant for a keyspace loss.
    ///
    /// `scripts/create_layer_commit.lua` checks it at every `CreateLayer`'s
    /// commit point and, when it exists, writes nothing, so a create
    /// authorized before a `DeleteSession` cannot map a layer into the deleted
    /// session. The orphan sweep's `scripts/create_layer_veto.lua` checks it
    /// too, and writes no `vetoed` claim for a deleted session's object (no
    /// commit can map it anyway).
    ///
    /// Carries the `{session_id}` hash tag, so it shares the delete
    /// transaction's Cluster slot — and the commit script's. Written in that
    /// transaction, the marker is present from the instant the session's
    /// metadata is gone.
    pub fn session_deleted_marker(&self, session_id: &str) -> String {
        format!("{}{{{session_id}}}:sess:deleted", self.prefix)
    }

    /// Sessions-by-path index set.
    /// Pattern: `le:path:{hex(base_path)}`
    ///
    /// The literal `{...}` around the hex-encoded base path is the Valkey
    /// Cluster hash tag — slot routing keys off the bytes inside the braces,
    /// not the full key. This pins the set onto the same slot as
    /// [`Self::path_names_index`], which is the precondition for the
    /// WATCH/SMEMBERS/MULTI/EXEC sequence in `SessionStore::create` to be
    /// slot-safe on Cluster.
    pub fn path_index(&self, base_path: &str) -> String {
        format!("{}path:{{{}}}", self.prefix, hex_encode(base_path))
    }

    /// Sessions-by-path names index set — holds composite members of the
    /// form `name\0session_id` for every session rooted at `base_path`
    /// (b.91j C-7). Shares the `{hex(base_path)}` hash tag with
    /// [`Self::path_index`] so both sets live on the same Valkey Cluster
    /// slot, allowing WATCH + SMEMBERS uniqueness check + SADD to run
    /// atomically in a single MULTI/EXEC.
    ///
    /// This set is the authoritative surface for session-name uniqueness
    /// and for `ListSessions` enumeration. NUL (`\0`) is the separator —
    /// `validate_session_name` rejects NUL, so name parsing is unambiguous.
    ///
    /// Pattern: `le:path:{hex(base_path)}:names`
    pub fn path_names_index(&self, base_path: &str) -> String {
        format!("{}path:{{{}}}:names", self.prefix, hex_encode(base_path))
    }

    /// Layer-to-object mapping string.
    /// Pattern: `le:{sid}:layer:{hex(layer_name)}`
    pub fn layer_mapping(&self, session_id: &str, layer_name: &str) -> String {
        format!(
            "{}{{{session_id}}}:layer:{}",
            self.prefix,
            hex_encode(layer_name)
        )
    }

    /// Layers-in-session index set.
    /// Pattern: `le:{sid}:layers`
    pub fn layers_index(&self, session_id: &str) -> String {
        format!("{}{{{session_id}}}:layers", self.prefix)
    }

    /// Session-checkpoint coordination hash — the Valkey-side ordering
    /// anchor for session-checkpoint builds and adapter publishes (b.443).
    ///
    /// Pattern: `le:{sid}:ckpt`
    ///
    /// Fields:
    ///
    /// - `seq` — monotone build stamp. `HINCRBY`'d once per session-checkpoint
    ///   build inside `session_checkpoint_read.lua`, atomically with the state
    ///   read it stamps. Higher stamp ⇒ strictly fresher payload.
    /// - `pub` — the highest stamp whose adapter operation (PUT or DELETE) has
    ///   actually landed. Advanced by `session_checkpoint_publish.lua` after
    ///   every successful adapter call.
    /// - `pub_kind` — `"write"` or `"delete"`, the kind of the operation that
    ///   set the current `pub`. A writer whose publish is refused reads this
    ///   to decide how to repair (rebuild vs re-delete).
    /// - `inc` — a random 128-bit incarnation token identifying this key's
    ///   LIFETIME (not any one build). `HSETNX`-claimed by
    ///   `session_checkpoint_read.lua` on the first touch after the key comes
    ///   into existence and read back by every later mint, so a key that was
    ///   lost (FLUSHALL, failover onto a lagging replica, out-of-band `DEL`,
    ///   TTL expiry) and then re-created carries a fresh token.
    ///   `session_checkpoint_publish.lua` refuses outright — writing nothing —
    ///   when the token an operation's build read is absent from the key or
    ///   differs from the live one, so a build minted against a dead
    ///   incarnation can never advance the live incarnation's watermark. `seq`
    ///   alone cannot do this: it restarts at 1 after key loss, so a pre-loss
    ///   build's stamp can outrank the fresh watermark, and an equal stamp
    ///   would be mistaken for the publisher's own retry.
    ///
    /// Carries the `{session_id}` hash tag so it shares a Cluster slot with
    /// [`Self::session_meta`], [`Self::layers_index`] and every
    /// [`Self::layer_mapping`] — the precondition for
    /// `session_checkpoint_read.lua` to read all of them inside one atomic
    /// script.
    ///
    /// **Deliberately NOT deleted by `SessionStore::delete`.** The key must
    /// outlive the session it names: a stale in-flight build whose PUT lands
    /// after the session-checkpoint DELETE is exactly the race the watermark
    /// exists to repair, and that repair reads `pub` / `pub_kind` after the
    /// session is already gone. A TTL — refreshed on every touch by both
    /// scripts, with one deliberate exception: the publish script's
    /// `stale_incarnation` refusal arm writes NOTHING, not even the
    /// `PEXPIRE`, because expiring a key this operation no longer owns would
    /// resurrect a `seq`-less, `inc`-less husk that the next mint would then
    /// adopt as its own incarnation; see
    /// `crate::checkpoint::session::SESSION_CHECKPOINT_STATE_TTL_MS` for the
    /// value and the bounds that fix it — self-reaps the key once no writer
    /// has touched it for a day.
    pub fn session_checkpoint_state(&self, session_id: &str) -> String {
        format!("{}{{{session_id}}}:ckpt", self.prefix)
    }

    /// `CreateLayer` operation record: a STRING holding
    /// `<object_id>:<previous_object_id>:<request_digest>` for one committed
    /// create that carried a client `operation_id`.
    ///
    /// Pattern: `le:{sid}:createop:{operation_id}`
    ///
    /// - `object_id` — the object the create committed (the answer a re-send
    ///   gets).
    /// - `previous_object_id` — the object the create replaced, or `0` for a
    ///   new layer. A re-send that is answered from the record re-drives the
    ///   overwrite's subscriber wake-up on it.
    /// - `request_digest` — lowercase-hex SHA-256 over `layer_name`, one
    ///   `0x00`, then the OMPF bytes. A re-send whose digest differs is refused
    ///   `ALREADY_EXISTS`.
    ///
    /// The `operation_id` sits in the key verbatim; `validate_operation_id`
    /// admits only `A–Z a–z 0–9 - . _ ~`, so it can neither add a `{…}` hash
    /// tag nor a `:` that would read as another key segment.
    ///
    /// # Lifecycle
    ///
    /// - **Written** by `scripts/create_layer_commit.lua`, in the same atomic
    ///   step that flips the layer mapping, with `PX`
    ///   [`CREATE_LAYER_OPERATION_TTL_MS`] counted from that landing — by the
    ///   `CreateLayer` call's own runs, its in-process finisher's, and the
    ///   orphan sweep's completion of a stranded overwrite. The completion
    ///   builds the byte-identical record from the operation id and request
    ///   digest the call stored on its new object's header at materialization
    ///   (`layer::HEADER_FIELD_CREATE_OPERATION_ID`,
    ///   `layer::HEADER_FIELD_CREATE_OPERATION_DIGEST`); an object an older
    ///   build materialized carries neither, and its completion writes no
    ///   record. Carries the `{session_id}` hash tag,
    ///   so it shares a Cluster slot with [`Self::layer_mapping`] and
    ///   [`Self::layers_index`], which is what lets that one script check and
    ///   write all three.
    /// - **Read** by the `CreateLayer` handler before it parses anything, by
    ///   `LayerStore::create_layer`'s `MGET` (with the mapping, before it
    ///   allocates), by its check after a failed commit run (one `MGET` with
    ///   the mapping and the commit claim), by its in-process finisher and the
    ///   orphan sweep's completion when a run answers `ALREADY_COMMITTED`, and
    ///   inside the commit script.
    /// - **Removed by TTL only** — never by `DeleteLayer` or `DeleteSession`,
    ///   the [`Self::session_checkpoint_state`] precedent. A re-send after the
    ///   layer was deleted is answered from the record rather than re-creating
    ///   the layer. It matches no `SCAN` pattern (`le:{*}:layer:*`,
    ///   `le:{*}:obj:hdr`, `le:{*}:obj:extret`, [`Self::ext_meta_scan_pattern`]).
    /// - **Never checkpointed.** A re-send after a DR restore finds no record
    ///   and creates the layer again.
    pub fn create_layer_operation(&self, session_id: &str, operation_id: &str) -> String {
        format!("{}{{{session_id}}}:createop:{operation_id}", self.prefix)
    }

    /// `CreateLayer` commit claim: a STRING that decides, once, the fate of
    /// one object a `CreateLayer` materialized — [`CREATE_LAYER_CLAIM_COMMITTED`]
    /// (a commit-script run mapped the object) or [`CREATE_LAYER_CLAIM_VETOED`]
    /// (the orphan sweep vetoed it). With either value no commit-script run
    /// ever maps a layer of the session to `object_id` again.
    ///
    /// Pattern: `le:{sid}:createclaim:{object_id}`
    ///
    /// - **`committed`** makes every commit idempotent per object, not per
    ///   mapping value. A `CreateLayer` whose run landed with its reply lost
    ///   learns it from the claim even after another create replaced the
    ///   object or a `DeleteLayer` removed the layer, so it never re-plans and
    ///   maps the object a second time — which would map the layer back to an
    ///   object that is being deleted, or bring a deleted layer back.
    /// - **`vetoed`**: the orphan sweep retires an object that no layer mapping
    ///   names and no overwrite marked superseded — the object of a
    ///   `CreateLayer` that materialized it and never committed. That create
    ///   may still be about to commit (a slow call, or a lost reply whose script
    ///   is still queued), so before retiring the object the sweep claims it
    ///   `vetoed`, and the create's commit refuses from then on. Both scripts
    ///   run atomically on one slot, so exactly one side wins without any
    ///   clock: the sweep never deletes an object whose commit could still
    ///   land, and a commit never maps a layer to an object the sweep is
    ///   deleting.
    ///
    /// # Lifecycle
    ///
    /// - **Written** by `scripts/create_layer_commit.lua` as `committed`,
    ///   atomically with every flip it makes (the `CreateLayer` call's runs,
    ///   its in-process finisher's, the orphan sweep's completion of a stranded
    ///   overwrite), with `PX` [`create_layer_claim_committed_ttl_ms`]; and by
    ///   `scripts/create_layer_veto.lua` (the orphan sweep) as `vetoed`, with
    ///   `PX` [`CREATE_LAYER_CLAIM_VETOED_TTL_MS`], only when no claim exists,
    ///   the layer mapping does not name `object_id` and the session's deletion
    ///   marker is absent.
    /// - **Never overwritten**, cleared or deleted by any code path: the first
    ///   writer wins. The veto keeps an existing claim, whichever value it
    ///   holds.
    /// - **Read** only by those two scripts — the commit script before its
    ///   first write, answering `COMMITTED_EARLIER` or `VETOED` and writing
    ///   nothing while it exists — and by `CreateLayer`'s check after a commit
    ///   run failed (one `MGET` with the layer mapping). Never by `UpdateLayer`,
    ///   subscribe, checkpoint or restore.
    /// - **Removed by TTL only** — never by `DeleteLayer` or `DeleteSession`,
    ///   the [`Self::create_layer_operation`] precedent. A sweep cancelled
    ///   between the veto and the deletion leaves the key, and the next pass
    ///   finishes the retirement. It matches no `SCAN` pattern
    ///   (`le:{*}:layer:*`, `le:{*}:obj:hdr`, `le:{*}:obj:extret`,
    ///   [`Self::ext_meta_scan_pattern`]).
    /// - **Never checkpointed.**
    ///
    /// Carries the `{session_id}` hash tag, so it shares a Cluster slot with
    /// [`Self::layer_mapping`] and [`Self::session_deleted_marker`], which both
    /// scripts read beside it. `object_id` is the decimal object id.
    pub fn create_layer_claim(&self, session_id: &str, object_id: u64) -> String {
        format!("{}{{{session_id}}}:createclaim:{object_id}", self.prefix)
    }

    // ── Object-scoped keys (hash tag: {object_id}) ─────────────────────

    /// Object header hash.
    /// Pattern: `le:{oid}:obj:hdr`
    pub fn obj_header(&self, object_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:hdr", self.prefix)
    }

    /// Section hash.
    /// Pattern: `le:{oid}:obj:sec:{section_id}`
    pub fn obj_section(&self, object_id: &str, section_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:sec:{section_id}", self.prefix)
    }

    /// Sections index set.
    /// Pattern: `le:{oid}:obj:secs`
    pub fn obj_sections_index(&self, object_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:secs", self.prefix)
    }

    /// Children ordering list.
    /// Pattern: `le:{oid}:obj:ch:{parent_id}`
    pub fn obj_children(&self, object_id: &str, parent_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:ch:{parent_id}", self.prefix)
    }

    /// Children-list parent-id index set (b.jdr).
    /// Pattern: `le:{oid}:obj:chs`
    ///
    /// Members are the decimal parent ids — *as strings*, and including the
    /// synthetic root `"0"` — that currently own a `le:{oid}:obj:ch:{parent_id}`
    /// list. Sibling of `obj:secs` / `obj:flds` / `obj:tss` / `obj:dels` in both
    /// naming and purpose: it is the index that makes the `ch:*` family
    /// enumerable at retirement.
    ///
    /// It exists because a parent id is **not** derivable from the other
    /// indices. `CommandDiffChangeSection` does not validate parent existence
    /// (SDD §10.4; PRD §2.3.4 — the server itself emits such deltas in catch-up
    /// diffs), and earlier builds stored a children list under such a
    /// *dangling* parent, which appears in neither `obj:secs` nor `obj:dels`:
    /// chiefly `apply_diff_change_section` under a missing parent, and also a
    /// create or move whose name-conflict cascade deleted the parent it
    /// named. The current engine keeps no list under a
    /// parent with no row, but those stored lists remain until a commit
    /// creates their parent (which deletes the list) or the object retires.
    /// Before this index, [`Self::object_cleanup_keys`] could only
    /// reconstruct *sections ∪ tombstones ∪ root*, and every such dangling
    /// parent's list outlived the object that owned it.
    ///
    /// Maintained in the same atomic unit as the children lists themselves, by
    /// every writer of those lists: inside the same `MULTI/EXEC` on the WATCH
    /// path (both materialization entry points — `storage::layer::write_object`
    /// for `CreateLayer` and `storage::layer::write_restored_object` for the DR
    /// restore, which share the one `queue_object_state` body — plus
    /// `execute_mutation`'s `SetChildren` / `DeleteChildrenList` arms), and
    /// inside the same script by `update_layer_fast.lua`'s structural mode,
    /// whose `CHL` / `DCHL` records issue the same `SADD` / `SREM` beside the
    /// list's `DEL` / `RPUSH`. So index and lists cannot diverge
    /// non-atomically. The `{object_id}` hash tag keeps it on the retiring
    /// object's Cluster slot, so it rides the existing chunked `UNLINK`.
    ///
    /// Both drift directions are *individually* benign: one stale extra member
    /// costs one `UNLINK` of a key that does not exist (a no-op), and one
    /// missing member degrades to exactly the pre-b.jdr behaviour. That is a
    /// statement about a single member, not a licence to skip the maintenance:
    /// stale members accrete without bound on a `noeviction` store, so the
    /// `SREM` forks in the `SetChildren` / `DeleteChildrenList` mutation arms
    /// (`storage::layer`, and the script's `CHL` / `DCHL` records) are what
    /// keep this Set proportional to the object's live children lists rather
    /// than to its lifetime write count. They are not optional.
    ///
    /// **Read-side load-bearing on marked objects.** Every object materialized
    /// through `queue_object_state` carries the header column
    /// `chs_complete = "1"` (`storage::layer::HEADER_FIELD_CHS_COMPLETE`),
    /// written in the same transaction as its lists and their members here. The
    /// marker certifies this Set as complete, so a reader that trusts it may
    /// read only the lists named here instead of probing the root plus every
    /// section. On such an object the "missing member" direction above is no
    /// longer benign: it hides a live list from that reader. An object without
    /// the marker (materialized before it existed) may carry a partial Set and
    /// must still be read the full per-section way.
    pub fn obj_children_index(&self, object_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:chs", self.prefix)
    }

    /// Per-object external-value reference index (b.qpf).
    /// Pattern: `le:{oid}:obj:ext`
    ///
    /// A Valkey HASH whose field is a lowercase `hash_hex` and whose value is
    /// the **number of value rows of this object** (fields + time samples)
    /// currently holding that hash.
    ///
    /// A row holds hash h iff it has **no `value` column** and its `hash`
    /// column is h — the decoder's rule (`diff_read::decode_field_value` reads
    /// `value` first). Every per-object reader and writer of this index counts
    /// by it: both materializers, both commit backends (the Lua script's
    /// pre-image read and `LayerStore::read_ext_pre_image`), retirement, and
    /// the per-object recount (`storage::ext_index_audit`). So a pre-b.zvd
    /// dual-shape row (`value` plus a stale `hash`) holds no hash. The
    /// reference-count reconciler's global walk alone reads the raw `hash`
    /// column, deliberately (a raw read can only over-count a membership,
    /// which leaks).
    ///
    /// # Invariant
    ///
    /// *Field `hash_hex` present with value `n > 0` ⟺ exactly `n` value rows of
    /// this object hold that hash.* A field that would fall to `0` is `HDEL`ed
    /// rather than stored, so presence alone answers "does this object still
    /// reference that blob?".
    ///
    /// # Why it exists
    ///
    /// `le:ext:meta:{hash_hex}`'s `ref_count` counts **live objects referencing
    /// the hash**, one per object per distinct hash. Neither commit backend can
    /// decide locally whether a rewrite actually dropped the object's last
    /// reference — two rows of the same object may share a hash, and
    /// `store_values` increments once per *distinct* hash per write, not once
    /// per row. This index is the per-object bookkeeping that makes the
    /// question answerable inside the object's own `{object_id}` slot, which is
    /// the only slot a commit is atomic on (`ext_meta` lives on `{hash_hex}`,
    /// the GC sets on `{gc}` — see [`Self::ext_pending`]).
    ///
    /// # Maintenance
    ///
    /// Written inside the same atomic unit as the value rows themselves — the
    /// slow path's `MULTI/EXEC`, the fast path's `update_layer_fast.lua` body,
    /// and the two materialization entry points' transactions (`write_object`'s
    /// `MULTI/EXEC` and `write_restored_object`'s `WATCH`ed one) — so index and
    /// rows cannot diverge non-atomically, save through a torn `EXEC` (see
    /// "Drift" below). Both backends write **absolute** `HSET` / `HDEL` values
    /// rather than `HINCRBY`: on the Lua path that keeps the b.ph9 "nothing
    /// after the `XADD` can reject" proof intact (an `HINCRBY` against a
    /// non-integer field value would abort the script mid-apply), and on the
    /// txn path it keeps the queued commands independent of what a replay would
    /// have already applied.
    ///
    /// The `{object_id}` hash tag keeps it on the retiring object's Cluster
    /// slot, so it rides the existing chunked `UNLINK` via
    /// [`Self::object_cleanup_keys`]. b.jdr's [`Self::obj_children_index`] is
    /// the exact precedent: same-atomic-unit index, same tag, rebuilt from
    /// `ObjectState` on DR restore so the checkpoint blob format needs no
    /// change.
    ///
    /// # Drift
    ///
    /// Not durable checkpoint state, and **not migrated**. An object
    /// materialized by a pre-b.qpf replica carries no `obj:ext`, and the two
    /// commit paths that trust this index — `update_layer_fast.lua` step 6b and
    /// the txn backend's `!needs_snapshot` branch, which reads it through
    /// `LayerStore::read_ext_pre_image` — read `0` for a hash such an object
    /// still references.
    ///
    /// That is not only a leak. A same-hash rewrite folds to `delta == 0` and
    /// is handled on row-local evidence (b.yda: when a row's original and final
    /// hash agree, the object provably already held the hash, so the commit's
    /// `+1` is `surplus_acquire` regardless of `before`, which cannot
    /// over-release). But a commit that CHANGES a hash's row count **seeds**
    /// the field with the absolute count `0 + delta`, computed from the rows
    /// this commit touched while other, unindexed rows still hold the hash —
    /// an **under-count**. Its pre-flight `+1` reads as surplus to the
    /// reference-count reconciler, and if that surplus were drained, the next
    /// rewrite that takes the short count to 0 would release a reference a
    /// live row still holds: the GC deletes a blob in use.
    ///
    /// What closes that seeding path is the reconciler's **per-object audit**
    /// (`storage::ext_reconcile`, `storage::ext_index_audit`) with its drain
    /// gate. Every pass compares each live object's index against its rows,
    /// at no extra read, and on a disagreement recounts the object atomically
    /// (`scripts/ext_index_recount.lua`) by the rule above, rewriting the
    /// differing fields and releasing nothing. A hash whose object it cannot
    /// repair (above the recount's row ceiling, or not live) is gated: no
    /// surplus is drained for it that pass. The audit runs before the pass's
    /// drain phase, and the pre-flight stamp keeps the seed's surplus
    /// undrainable for at least a grace, so an audit on this build reaches the
    /// object before a drain on this build can. What it guarantees, **from the
    /// first audit pass on this build**, is **no loss through a drain** — and
    /// only for drains this build performs. A seed surplus a pre-fix reconciler
    /// (which has no audit or gate) already drained, or one an old-build
    /// replica holding `le:gc:leader` drains during a rolling upgrade, leaves
    /// an under-count with no surplus behind it, exposed to the commit paths
    /// until that first audit pass repairs it. And on the txn backend, a
    /// commit whose `WATCH` was void can overwrite a same-pass repair before
    /// phase 5 drains (see the header-write paragraph below), so that
    /// combination can still lose through a drain. Beyond those, the
    /// guarantee is not a repair: a gated hash leaks until its object is
    /// repaired, shrinks or retires; a wrong-typed key the audit reads forfeits
    /// every pass until it is cleared, so nothing is repaired or drained and an
    /// under-count already standing stays exposed to the commit paths; and
    /// with the reconciler disabled there is neither audit nor drain, so only a
    /// seeding under-count whose surplus is still standing is safe and drift
    /// from any other producer is never repaired by a pass.
    ///
    /// The other sources of drift have no surplus behind them, so they can
    /// over-release on a commit path before any pass runs; the audit only
    /// shortens that exposure. One is the signed-zero time-sample aliasing of
    /// an older build, whose WATCH path's snapshot branch planned on a state
    /// that collapsed a `-0.0` / `+0.0` row pair into one row and wrote the
    /// pair's hash one row short; this build plans on the compact form, which
    /// keys a row by its stored time bits, and writes no `-0.0` row. The other
    /// is a torn txn
    /// commit (`WatchedExec::ExecError` in `LayerStore::update_layer_once`):
    /// absolute counts planned before `MULTI` can land while other frames of
    /// the commit do not. The commit path recounts the object immediately
    /// after a tear, but a commit that lands between the tear and the recount
    /// plans from the stale counts and can over-release; nothing covers that
    /// window. What the recount misses (an object above the ceiling, a recount
    /// that fails, an RPC cancelled before it is sent) stays exposed to the
    /// commit paths until the next audit pass, with the same limits as above.
    /// A layer with a malformed — uppercase or mixed-case hex — `hash` row
    /// outlasts that: the recount and every audit are raises-only there and
    /// do not count the row, so neither can raise to the decoder's count. An
    /// under-count from such a row stays exposed to the commit paths until
    /// the malformed rows are corrected; the audit only holds the layer's
    /// mismatched hashes from drain meanwhile.
    ///
    /// Besides the audit, four write paths repair drift on their own:
    ///
    /// * `update_layer_fast.lua`'s **structural mode** (the structural commit,
    ///   `storage::structural`) reads `before` from this key in-script, and
    ///   writes the **raise-only floor** `max(before + delta, untouched + ends)`
    ///   whenever the stored field is not exactly that value (`HDEL` at 0):
    ///   `untouched` is the cache's pre-plan count of the rows the commit does
    ///   not touch (`PreImageDigests`), `ends` the touched rows whose final
    ///   write holds the hash. So it repairs an under-count on every hash it
    ///   touches, including one whose row count the commit leaves unchanged.
    ///   It never lowers a count or causes a release on the cache's word:
    ///   untouched rows carry no pre-image digest, and the cache's decoded
    ///   values are not the store's raw `hash` columns, so a cache wrong about
    ///   one untouched row could otherwise write a count below the rows and
    ///   release a live blob. Releases stay on store-side evidence — `before`,
    ///   `held`, and an `after` of 0, which also needs `before + delta <= 0`.
    ///   An over-count is kept at `before + delta` for the audit to lower;
    ///   lowering it to the cache's recount was rejected for the loss path it
    ///   opens.
    /// * The WATCH path's **snapshot** branch — taken by every structural
    ///   commit that does not go through structural mode (the `txn` and
    ///   `lua-watch-structural` backends, and uncacheable objects on `lua`) —
    ///   derives `before` from the loaded compact state's row counts rather
    ///   than the index, so a hash it touches is rewritten with the count its
    ///   loaded state holds when the commit changes that hash's row count.
    ///   That is the true count: the compact form keys a time-sample row by
    ///   its stored time bits, so a legacy `-0.0` / `+0.0` pair counts as the
    ///   two rows it is.
    /// * The two materialization entry points — `LayerStore::write_object` (the
    ///   `CreateLayer` materializer) and `LayerStore::write_restored_object`
    ///   (the DR-restore one) — are an absolute rebuild: the delta spans every
    ///   hash of `state` as well as every field already present in the index,
    ///   so each field is rewritten with its true count (or deleted).
    ///
    /// A repair writes absolute counts without advancing `seq`, and stamps the
    /// header column `ext_index_repaired_ms` so in-flight **watched** txn
    /// commits on the object are cancelled through their header `WATCH`. A
    /// commit whose `WATCH` was void commits unconditionally and can overwrite
    /// a concurrent repair; its `EXEC` returns all-OK, so no torn-commit
    /// recount follows and the drift waits for the next audit. When the
    /// overwritten repair was the reconciler's own, made earlier in the same
    /// pass, that pass's phase 5 can still drain the surplus the restored
    /// under-count relied on (txn backend only). No cache keyed
    /// on `seq` may hold this index's counts.
    ///
    /// What still has no safe Lua-side repair is the **count clamp**: clamping
    /// `after` to `1` when the field is absent would over-release whenever two
    /// rows of the object share the hash — an unmatched decrement, i.e. data
    /// loss, the failure direction this whole index exists to prevent. That
    /// narrowing is about the *count*, and it is untouched by the `held` rule
    /// above, which asserts nothing about how many rows hold the hash. The
    /// structural floor is not that clamp: it counts every row of the object,
    /// the untouched ones from the plan's cache and the touched ones from the
    /// commit's own records, and only raises.
    ///
    /// Retirement still reclaims the object's *surviving* references:
    /// `schedule_old_object_ext_gc` releases the **union** of `obj:ext`'s
    /// fields and the hashes the object's value rows name (its reference walk,
    /// `storage::retirement`), exactly the rolling-deploy cover
    /// [`Self::object_cleanup_keys`] keeps for `obj:ch:0` — one release per
    /// distinct hash, so any surplus increments left above it are not reclaimed
    /// by it and are the reconciler's to drain (SDD §11.3).
    pub fn obj_ext_index(&self, object_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:ext", self.prefix)
    }

    /// Per-object `UpdateLayer` commit-token record. HASH whose field is a
    /// writer's `client_id` (the empty string included) and whose value
    /// describes that writer's most recent landed commit, in one of two
    /// shapes:
    ///
    /// - `<token>:<seq>:<server_ts>` — the call carried no `operation_id`:
    ///   its per-call commit token, the sequence it landed at, and the server
    ///   timestamp it was stamped with.
    /// - `op.<operation_id>:<seq>:<server_ts>:<token>:<request_digest>` — the
    ///   call carried a client `operation_id`. The match key is `op.` plus the
    ///   id; the per-call token and the lowercase-hex SHA-256 request digest
    ///   (`layer_name`, one `0x00`, the OMPF bytes) ride after the two fields
    ///   the no-id shape carries.
    ///
    /// Pattern: `le:{oid}:obj:wtok`
    ///
    /// The leading `<key>:` is what an attempt matches. Neither key can contain
    /// `:` (a token is 32 hex digits; `validate_operation_id` admits only
    /// `A–Z a–z 0–9 - . _ ~`), and an `op.` key can never equal a hex token, so
    /// a record can only ever match the shape it was written in.
    ///
    /// # What it is for
    ///
    /// `LayerStore::update_layer_with_operation` mints one random 128-bit token
    /// per call and hands it to every in-process retry attempt. Each attempt
    /// reads this record before it applies anything:
    ///
    /// - A record carrying this call's own token means an earlier attempt of
    ///   the SAME call already committed (its reply was lost), so the attempt
    ///   reports that commit — the recorded
    ///   sequence and server timestamp — instead of applying the delta again.
    ///   That is what makes one call commit at most once on both backends.
    /// - With an `operation_id`, a record whose `op.` key matches but whose
    ///   token is another call's means a client re-send of an operation an
    ///   earlier call already committed: it is answered as that commit
    ///   (`ALREADY_COMMITTED`) when the digests agree, and refused
    ///   `ALREADY_EXISTS` when they differ. That is what makes a client re-send
    ///   exactly-once across calls.
    ///
    /// # Lifecycle
    ///
    /// - **Written** with every commit, inside the commit's own unit: one
    ///   `HSET` in `update_layer_fast.lua`'s apply pass (after the `XADD`,
    ///   where nothing can reject, so on the fast path the record exists
    ///   exactly when its commit does), or one `HSET` queued in the slow
    ///   path's `MULTI`. The slow path has one residual: `MULTI`/`EXEC` does
    ///   not roll back on an execution-time error (SDD §7.5.2), so a TORN
    ///   commit runs every queued frame except the failing one. The failing
    ///   frame is the `XADD` (the commit never reached the stream), or a
    ///   field, section or index key retyped between the pre-`MULTI` type
    ///   probe and the `EXEC` — only the header, stream and snapshot fence are watched, the
    ///   SDD §7.5.4 residual — in which case the `XADD` lands too and the
    ///   stored state is half-applied. Either way, unless the record's own
    ///   `HSET` is the failing frame, the record lands for a commit that did
    ///   not fully land. Within the call that is harmless only because
    ///   `update_layer_once` reports a tear as a non-retriable `Internal`, so
    ///   no later attempt of that call ever reads the record (see the
    ///   `WatchedExec::ExecError` arm). A record carrying an `operation_id`
    ///   would still be read by the client's re-send, which would then answer
    ///   the torn commit `ALREADY_COMMITTED`; so that arm also **retracts** it —
    ///   a compare-and-delete of this `client_id`'s field, only while it still
    ///   holds this call's exact record — in a task on the store's cleanup
    ///   tracker whose handle the arm awaits, so it normally completes before
    ///   the tear is answered and a deadline drop of the call cannot cancel
    ///   it. The residual is a process crash between the torn `EXEC` and the
    ///   retract; a retract that fails (logged at ERROR); and a drop of the
    ///   call while it awaits the retract (server deadline, client cancel or
    ///   reset), after which the client's prompt re-send can read the record
    ///   before the detached retract runs.
    ///   Never written by `write_object` or `write_restored_object`.
    /// - **Read** at the head of every `update_layer` attempt, after the
    ///   not-found and partial-header guards and before everything else.
    /// - **Removed** by [`Self::object_cleanup_keys`], which covers
    ///   `DeleteLayer`, the `CreateLayer` overwrite cleanup, the orphan sweep
    ///   and the DR-restore residue wipe. The `{object_id}` hash tag keeps it
    ///   on the object's slot for that chunked `UNLINK`.
    /// - **Never checkpointed.** It describes in-flight calls, not layer
    ///   state, and losing it fails open: a call whose record is gone simply
    ///   cannot recognise its own landed commit, or its re-sent operation, and
    ///   re-applies, which is the behaviour without the record. That includes a
    ///   re-send after a DR restore, which wipes the record with the rest of
    ///   the object's residue.
    ///
    /// # Scope of the guarantee
    ///
    /// One field per `client_id`, so the guarantee holds while a client keeps
    /// at most one write in flight per object and `client_id`, and sends a
    /// re-send before that `client_id`'s next write to the object. Two
    /// concurrent calls sharing a `client_id` overwrite each other's record,
    /// and that fails OPEN — each can then miss its own landed commit and
    /// re-apply — never to a false "already landed": a random 128-bit token
    /// cannot match a different call's commit, and an `op.` key matches only a
    /// commit made under the same `operation_id`. Without an `operation_id` the
    /// guarantee does not span calls: a client that retries after losing a
    /// reply starts a new call with a new token. An `operation_id` call must
    /// carry a non-empty `client_id` (the gRPC boundary refuses one without).
    ///
    /// Because the field holds only the LATEST commit, an unchained
    /// operation-id call also reads it ONCE at call start — its start record —
    /// and every attempt refuses itself as overtaken when the field has moved
    /// to another commit by then. That is what keeps an attempt that executes
    /// late (a retry, a delayed `EVALSHA`) from applying after the same
    /// `client_id`'s next commit; see `LayerStore::update_layer_with_operation`.
    ///
    /// A chained `UpdateLayer` (`UpdateLayerRequest.chained_header`) lifts the
    /// one-write-in-flight limit for its writer: the field's latest operation
    /// is a position in the writer's chain, and a chained call commits only
    /// while that position is its predecessor's. The chain gate waits for it
    /// (`storage::chain_gate`), and every commit attempt checks it — its
    /// writer guard is its predecessor, not a start record, so it is never
    /// overtaken and reads no start record; a late attempt whose predecessor
    /// is no longer the field is refused `PREDECESSOR_NOT_COMMITTED`, writing
    /// nothing. Its writer may keep several chained writes in flight; they
    /// commit in chain order, each at most once, and a re-send of one is
    /// answered from the field only while that write is still the latest, as
    /// for any call. The one-field-per-`client_id` rule still binds: two
    /// writers sharing a `client_id` share one chain.
    ///
    /// Growth is one field per distinct writer `client_id` (each at most 256
    /// bytes, `validate_client_id`) for the object's lifetime.
    pub fn obj_write_token(&self, object_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:wtok", self.prefix)
    }

    /// Per-object `UpdateLayer` commit release ring. A hash of three field
    /// shapes:
    ///
    /// - `|t|<token>` → `"<seq> <reason>:<hash_hex> …"`: the ENTRY of the
    ///   commit made by the call whose commit token, in its recorded form
    ///   (`CommitIdentity::token`), is `<token>` — its sequence, then every
    ///   external-value release it owed, space-separated, in the
    ///   `update_layer_fast.lua` release-token grammar (`reason` is `rewrite`
    ///   or `surplus_acquire`);
    /// - `|s|<k>`, k = 0..RING−1 in plain decimal → the token whose entry's
    ///   run of slots starts at slot k, or `""` for a slot a wider entry
    ///   covers;
    /// - `|n|` → the next slot to write, in plain decimal, 0..RING−1.
    ///
    /// Pattern: `le:{oid}:obj:wrel`
    ///
    /// Written in the commit's own atomic unit, beside the commit-token record
    /// ([`Self::obj_write_token`]): step 7a of `update_layer_fast.lua` in both
    /// modes, and the WATCH path's `MULTI`, by one rule
    /// (`storage::layer::plan_release_ring_write`). A commit that owes no
    /// releases does not touch the key. A commit owing R releases takes
    /// min(RING, ⌈R / M⌉) slots from `|n|`, deletes the entry of every other
    /// token whose start slot it takes, writes its entry and slots and
    /// advances `|n|` (RING = `layer::COMMIT_RELEASE_RING_SLOTS`, M =
    /// `layer::COMMIT_RELEASE_RING_SLOT_RELEASES`). So the key holds at most
    /// 2·RING + 1 fields and at most RING × M retained release tokens plus one
    /// oversized entry, and an entry survives until later releasing commits on
    /// the object wrap the ring onto its start slot, or the key expires
    /// (below); an evicted entry's
    /// releases leak to the reference-count reconciler, never release twice.
    ///
    /// Read by exactly one call per entry: the call whose token keys it,
    /// within that call (one `HGET`). An attempt that finds its OWN call's
    /// commit already landed (an earlier attempt's reply was lost) enqueues
    /// the entry's releases at `Commit(seq)` when its sequence equals the
    /// landed commit's; a call that ends without proof of its outcome — retry
    /// exhaustion, or a writer-guard refusal or refusal-detail answer after a
    /// counted retry — enqueues whatever entry its token holds at the entry's
    /// sequence (`LayerStore`'s `enqueue_own_commit_releases`). The token is
    /// random per call, so a found entry is that call's own lost reply, and
    /// no successor's write replaces it: a chained call whose successor
    /// committed before its next attempt recovers its releases too. A
    /// received reply ends the call, so the releases are enqueued once per
    /// commit.
    ///
    /// **TTL.** Every commit that writes the ring sets the key's TTL to
    /// [`commit_release_ring_ttl`] (`LIVE_EDIT_RPC_DEADLINE_MS` +
    /// `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` + 60 s, 95 s at the defaults),
    /// in the same atomic unit as the write: a `PEXPIRE` after the ring
    /// `HSET`, in step 7a and in the WATCH path's `MULTI`. So the key lives
    /// while the layer is being edited and expires that long after its last
    /// releasing commit; a commit that owes no releases neither writes nor
    /// refreshes it. The TTL outlives every reader: an entry is read only by
    /// its own call, before that call's deadline. Expiry only ever leaks: an
    /// expired ring reads as absent everywhere — the lost-reply lookup counts
    /// `absent` and enqueues nothing, a missing `|n|` reads as slot 0 and a
    /// missing slot evicts nothing — and the reference-count reconciler, which
    /// never reads the ring, repairs what was owed. A WATCH attempt whose
    /// window read the key just before it expired writes a one-entry ring at
    /// its planned slot (the key is not watched). A torn `EXEC` can leave the
    /// `HSET` without the `PEXPIRE`: the key then persists until the next
    /// releasing commit, a memory-only leak bounded by the ring. Expiry can
    /// precede an in-flight lookup only across builds or configurations whose
    /// TTLs differ, or across a step in the store's clock — a leak either way.
    ///
    /// `|` cannot occur in a `client_id` (`validate_client_id`), so these
    /// fields never collide with the per-`client_id` field
    /// (`"<seq> <tokens>"`) an older build wrote here. This build neither
    /// reads nor writes such a field, and an older build neither reads nor
    /// writes the ring's: across a rolling upgrade or a rollback each build
    /// sees only its own shape, and the other shape's fields stop growing
    /// and are removed with the key — by its expiry too: an older build's
    /// field written into a key this build set a TTL on keeps that TTL and
    /// expires with it (a leak only, for that build's lookups), and an older
    /// build's write into an absent key creates it without one, persistent
    /// until this build's next releasing commit on the object sets it.
    ///
    /// Type-checked before the commit writes — after the own-key match, the
    /// superseded guard, a chained call's predecessor check and the snapshot
    /// fence wait, and ahead of the `seq` /
    /// `set_order` ceilings and the stream-id collision (the update script's
    /// step 1e; the WATCH path's step 2c, and again its keyspace probe; the
    /// cached structural route sends a plan that meets a ceiling to the WATCH
    /// path): a wrong-typed key refuses the commit `WRONGTYPE_KEY`, the same
    /// reply on every route, with nothing written. (A delta the structural
    /// route's plan cannot apply to the cached state at all is refused by
    /// that plan before this check and the snapshot fence check; that
    /// route's read has already answered the own-key match, the superseded
    /// guard and the predecessor check.) Removed by its TTL (above), and with
    /// the object ([`Self::object_cleanup_keys`]),
    /// including by the restore residue wipe — the ring's fields and any
    /// older build's per-`client_id` fields alike. This build's writes are
    /// bounded by the ring (above), not by the number of writers.
    pub fn obj_write_releases(&self, object_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:wrel", self.prefix)
    }

    /// Per-object snapshot fence (Spec SDD SR-5.1). HASH written only by
    /// fenced snapshot readers, through the `snapshot_fence_*.lua` scripts that
    /// `storage::snapshot_fence` wraps; nothing else writes it.
    /// Pattern: `le:{oid}:obj:snapfence`
    ///
    /// # Fields
    ///
    /// - `r:<token>` → lease deadline in ms on the store's `TIME`, one per
    ///   live fenced reader (`<token>` is a random 128-bit hex string per read);
    /// - `epoch_start` → the ms the current fence epoch began;
    /// - `gap_until` → the earliest ms a new epoch may begin;
    /// - `size_e`, `size_b`, `bpi` → the layer's size the scripts compute every
    ///   per-layer limit from (items, reply bytes, bytes per item).
    ///
    /// # Lifecycle
    ///
    /// - **Created** by a granted acquire (`snapshot_fence_acquire.lua`). A
    ///   refused or waiting acquire writes nothing, so it never trips the `WATCH`
    ///   a WATCH-path writer holds on this key.
    /// - **Read** by other readers (join the epoch or `RETRY_AFTER`), by intent
    ///   registration (refused while a reader is live), and, under the writer
    ///   protocol of Spec SDD SR-5.4, by every commit: the update script
    ///   reads it in both modes at its step 1d, as
    ///   `KEYS[lua::UPDATE_SCRIPT_KEY_FENCE]` (any live `r:*` refuses the
    ///   commit `{"FENCED", ms}`, writing nothing), and the WATCH path
    ///   (`LayerStore::update_layer_once_watch_attempt`) `WATCH`es it beside
    ///   the header and stream and runs `snapshot_fence_probe.lua` on the
    ///   `WATCH`ed connection beside its header read (a live lease ends the
    ///   attempt as a fence wait; a grant after the `WATCH` fails its `EXEC`).
    ///   Every path refuses a wrong-typed key with the same `WRONGTYPE_KEY`
    ///   reply.
    /// - **Renewed** by the holder's renewal timer (`snapshot_fence_renew.lua`).
    /// - **Released** by `snapshot_fence_release.lua`, by the lease guard's
    ///   best-effort release on drop, by lease expiry (judged against `TIME`),
    ///   by the key's own `PEXPIRE` — every writing script sets it to the later
    ///   of the latest live deadline and `gap_until`, plus `key_linger(L)` — and
    ///   by `UNLINK` with the object: it is in [`Self::object_cleanup_keys`] and
    ///   the restore residue probe. Release never recreates the key on a wiped
    ///   object: it writes nothing when its reader's field or the header is gone.
    /// - **Never checkpointed.** It describes reads in flight, not layer state.
    ///
    /// The `{object_id}` hash tag puts it on the object's slot, so the scripts
    /// can read it with the header and the intent key, and the update script
    /// can declare it beside the commit's other keys.
    pub fn obj_snapshot_fence(&self, object_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:snapfence", self.prefix)
    }

    /// Per-object writer intents (Spec SDD SR-5.1, SR-5.4). HASH written only
    /// by WATCH-path attempts that read a snapshot, on layers above the
    /// atomic-read ceiling (the cached structural route registers none,
    /// structural-edits SR-8.2), through `snapshot_intent_register.lua` /
    /// `snapshot_intent_renew.lua` and the intent guard's `HDEL`.
    /// Pattern: `le:{oid}:obj:snapintent`
    ///
    /// # Fields
    ///
    /// `w:<token>` → intent deadline in ms on the store's `TIME`, one per
    /// in-flight WATCH-path attempt.
    ///
    /// # Why a key of its own
    ///
    /// Writers registering intents must never modify the fence key
    /// ([`Self::obj_snapshot_fence`]): other WATCH-path writers `WATCH` it,
    /// and a write there would abort their `EXEC`.
    ///
    /// # Lifecycle
    ///
    /// - **Created** by intent registration, refused (writing nothing) while a
    ///   fenced reader is live, and never on an object whose header is gone.
    /// - **Read** by fence acquire, which delays a new epoch while an intent is
    ///   live, but only until `gap_until + intent_ttl(L)` so writers cannot
    ///   starve readers. Its type alone is also read by every commit — the
    ///   update script declares it in both modes as
    ///   `KEYS[lua::UPDATE_SCRIPT_KEY_INTENT]` and `TYPE`-probes it at step
    ///   1d, and the WATCH path's fence probe names it — so every path refuses
    ///   a wrong-typed key the same way.
    /// - **Renewed** while the attempt runs; **cleared** by `HDEL` on every exit
    ///   of the attempt (and, for a dropped attempt, by the renewal task or the
    ///   intent guard's spawned `HDEL`).
    /// - **Backstops:** an intent lapses on its own deadline, and every writing
    ///   script sets the key's `PEXPIRE` to its latest deadline. It is
    ///   `UNLINK`ed with the object ([`Self::object_cleanup_keys`], the restore
    ///   residue probe).
    /// - **Never checkpointed.**
    pub fn obj_snapshot_intent(&self, object_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:snapintent", self.prefix)
    }

    /// The migration sentinel of hash tag `tag` (b.cyu): a key that is
    /// **never written**, declared by every writing script that could
    /// otherwise half-apply during a legacy cluster slot migration, so that
    /// the store refuses such a script before it starts while its slot is
    /// migrating or importing.
    /// Pattern: `le:{<tag>}:migration-sentinel`
    ///
    /// # Why declaring an absent key does that
    ///
    /// During a legacy move (`CLUSTER SETSLOT … MIGRATING / IMPORTING` plus
    /// key-by-key `MIGRATE`), the store's pre-execution cluster check refuses
    /// a multi-key command that names a key absent from the node it reached:
    /// `-TRYAGAIN` on the migrating source while some of its keys are still
    /// there, `-ASK` once none are, and `-TRYAGAIN` on the importing target
    /// (Valkey 9.0.3 `cluster.c:1264-1284`). This key is always absent, so a
    /// script that declares it is refused for the slot's whole move, before
    /// any of its commands runs. Once a script runs, each of its in-script
    /// commands is checked again, and one naming a key the run itself emptied
    /// is refused mid-script with the earlier writes kept; the sentinel keeps
    /// such a script from starting. The client already treats both replies as
    /// transient (`-TRYAGAIN` via `client::is_tryagain_reply`; the `-ASK`,
    /// which the vendored fred never follows for a script, as `"Max attempts
    /// reached."`).
    ///
    /// On a stable slot the key is never looked up: the cluster check reads
    /// the keyspace only for a migrating or importing slot (Valkey 9.0.3
    /// `cluster.c:1101-1127`; ≤ 8.x computes each key's slot but skips the
    /// lookup). On a standalone store there is no cluster check at all.
    ///
    /// # Lifecycle
    ///
    /// - **Never written, read or deleted** by any command: a script that
    ///   declares it only checks, before anything else, that its `KEYS` slot
    ///   holds this name for the script's own hash tag, and refuses
    ///   `ENCODING` otherwise. So it holds no memory and is in no cleanup or
    ///   restore inventory.
    /// - **Built only here**, at the call sites that assemble those scripts'
    ///   `KEYS`: the commit's shared prefix (`lua::update_script_shared_keys`),
    ///   the two GC ledger scripts (`tag` `gc`), the external-value index
    ///   recount and the time-sample signed-zero merge
    ///   (`time_sample_merge::apply_signed_zero_merge`; `tag` the object id).
    ///
    /// The hash tag puts it on the script's own slot, so it never makes a
    /// script cross slots.
    pub fn migration_sentinel(&self, tag: &str) -> String {
        format!("{}{{{tag}}}:migration-sentinel", self.prefix)
    }

    /// Retirement intent carrier for one object (b.yda; write-ahead since b.yuo
    /// Part R). HASH (since b.1jt) whose fields are the `le:ext:{gc}:pending`
    /// member strings (`external_values::ExtRelease::member`) a retirement
    /// owes, each with the value `1`.
    /// Pattern: `le:{oid}:obj:extret`
    ///
    /// # The type tells this build's carrier from an older build's
    ///
    /// Every build before the GC protocol marker (0.1.0 through 0.2.0-rc.48)
    /// writes a SET at this key, and its carrier claim and reconciler read it
    /// with `SCARD` / `SMEMBERS`. On this build's HASH those fail `WRONGTYPE`
    /// before writing anything, so no older build can pay or delete an
    /// obligation this build recorded. Conversely a SET here is an older
    /// build's carrier, which may record none of its in-doubt attempts, and
    /// this build never pays one: the orphan sweep's drain clears it if a
    /// recorded attempt landed and otherwise drops it (a leak the reconciler
    /// repairs, unless it is turned off with
    /// `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0`, when the leak is
    /// permanent). Readers that need the members dispatch on `TYPE` (the
    /// re-token, the reconciler's carrier walk); the others (`EXISTS`, `DEL`,
    /// the backstop `SCAN`) are type-agnostic.
    ///
    /// # Written BEFORE the enqueue, deleted only once an attempt is decided
    ///
    /// Every retirement that owes releases writes its members here first —
    /// `scripts/ext_retire_begin.lua`, in one step with `ext_gc_scheduled=1` on
    /// the header and the retirement's attempt token on
    /// [`Self::obj_ext_retire_attempts`] — and only then sends its attempt-mode
    /// enqueue (`scripts/ext_release_enqueue.lua`) carrying that token. The
    /// carrier is deleted only by `scripts/ext_retire_clear.lua`, once one of
    /// its attempts is known to have landed (or, for an older build's SET
    /// carrier with none landed, to drop it), or by
    /// `scripts/ext_retire_settle.lua` when every recorded attempt is decided
    /// void and the object id is live again (the cancel). Nothing else removes
    /// it: the re-token claim
    /// (`scripts/ext_retire_retry_claim.lua`) is non-destructive, and there is
    /// no put-back.
    ///
    /// That ordering is what makes a re-presented member safe without any
    /// timing assumption in what the service decides: a member here is enqueued
    /// again only after the `{gc}` slot has decided, once and for good, that
    /// every earlier attempt — this build records each before sending it —
    /// never ran
    /// (`scripts/ext_enqueue_attempt_resolve.lua`'s `HSETNX` against the
    /// enqueue's own check-then-set, on one slot), and the
    /// `{gc}` record of each attempt outlives every send under its token (see
    /// [`Self::ext_enqueue_attempts`]).
    ///
    /// # Why it lives on the OBJECT's slot
    ///
    /// The enqueue it guards is a write on the `{gc}` slot, and the natural
    /// Cluster shape of that failing is "the object-slot reads succeeded, the
    /// `{gc}`-slot write did not". The carrier therefore lands on `{object_id}`:
    /// the slot whose reads just succeeded, and the slot whose `UNLINK` teardown
    /// follows.
    ///
    /// Members are the fully-formatted pending members rather than bare hashes
    /// so the drain re-enqueues byte-identical text, and the scripts that write
    /// and return them never interpret them.
    ///
    /// # Deliberately absent from [`Self::object_cleanup_keys`], and with no TTL
    ///
    /// Every other `obj:*` key is UNLINKed by the teardown that follows the
    /// retirement. This one must NOT be: when the enqueue's outcome is in doubt
    /// the teardown runs anyway, and the carrier is the only record of the
    /// obligation and of the attempts that may have paid it. It outlives its
    /// object by design. No TTL either, for the same reason the three
    /// `le:ckpt:{recon}:*` carriers have none: an expiry would silently drop the
    /// only record of an owed reclamation.
    ///
    /// The exclusion is argued in full at [`Self::object_cleanup_keys`].
    ///
    /// # Paid while the id is dead; CANCELLED once it is live again
    ///
    /// A drain that finds every recorded attempt void settles the carrier in
    /// one step with a liveness gate on `obj:hdr` and `obj:ext`. An id that is
    /// live again (a checkpoint restore at the same id, or a teardown that
    /// never completed) has inherited the standing, unreleased `+1` per hash
    /// these members represent, so the carrier and its attempt tokens are deleted and
    /// `ext_gc_scheduled` is cleared off any surviving header — never paid,
    /// because paying would be an unmatched decrement, `ext_meta` and
    /// `ext_data` deleted under a live reference. Cancelling leaks instead, and
    /// the leak is the reference-count reconciler's to drain.
    ///
    /// A carrier one of whose attempts LANDED is not gated at all: the
    /// obligation was already paid, so the carrier is cleared and
    /// `ext_gc_scheduled` is kept even when the id is live again (policy 12).
    /// Clearing the flag there would re-arm a re-derivation that pays the same
    /// members a second time.
    ///
    /// # `ext_gc_scheduled` is written with it
    ///
    /// The begin script writes the members here and marks `ext_gc_scheduled=1`
    /// on the object's header in one step, before the enqueue. The flag is what
    /// stops a `delete_layer` cascade re-run, the orphan sweep's revisit of a
    /// surviving header, or a concurrent second retirement from re-deriving the
    /// same members while this carrier still records them. It is cleared only by
    /// a settle that cancels; its contract is "GC has been considered *for this
    /// incarnation*".
    pub fn obj_ext_retire_carrier(&self, object_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:extret", self.prefix)
    }

    /// The attempt tokens of one object's retirement intent (b.yuo Part R).
    /// HASH: attempt token (32 lowercase hex) → the `{object_id}` primary's
    /// `TIME` in ms when the token was added.
    /// Pattern: `le:{oid}:obj:extret:attempts`
    ///
    /// Every attempt-mode enqueue of the carrier's members
    /// ([`Self::obj_ext_retire_carrier`]) carries one token, and the token is
    /// added here BEFORE that enqueue is sent: by `scripts/ext_retire_begin.lua`
    /// for the retirement's own attempts, and by the drain's re-token
    /// (`scripts/ext_retire_retry_claim.lua`) for each re-drive. So whatever
    /// attempt may have run, its token is here, and the drain asks the `{gc}`
    /// slot about every one of them ([`Self::ext_enqueue_attempts`]) before it
    /// sends another.
    ///
    /// - **Read** by the re-token (the newest token's age is the "a retirement
    ///   is still in progress" check), by `scripts/ext_retire_settle.lua` (the
    ///   token set must be unchanged since the re-token), by
    ///   `scripts/ext_retire_clear.lua` (a newer token means another drain owns
    ///   the carrier), and by the GC loop's attempt audit (`HEXISTS`, to tell a
    ///   landed record nothing references any more).
    /// - **Removed** only together with a carrier that exists — by the clear's
    ///   `cleared` (which also drops an older build's SET carrier), by a
    ///   settle that cancels. A HASH with no carrier beside it
    ///   (only an older build's destructive claim leaves one) may be the only
    ///   evidence that this incarnation's obligation was paid, so nothing inside
    ///   an incarnation removes it: the re-token's `empty` arm, the clear and
    ///   the settle all leave it, and the begin answers `busy` over it. Only the
    ///   incarnation boundary may remove it, and nothing in this build does:
    ///   b.9ud sub-task 3 plans the only safe remover (the restore's residue
    ///   wipe, extended to void the tokens and delete the HASH — not built).
    ///
    /// `{object_id}`-tagged, so it shares a slot with the carrier and the
    /// header and every script above is single-slot. Like the carrier it is
    /// NOT in [`Self::object_cleanup_keys`] and has no TTL: it must survive the
    /// teardown the carrier survives. It matches no `SCAN` pattern this service
    /// issues — in particular not `le:{*}:obj:extret`, which ends at `extret`.
    pub fn obj_ext_retire_attempts(&self, object_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:extret:attempts", self.prefix)
    }

    /// Field hash.
    /// Pattern: `le:{oid}:obj:fld:{section_id}:{hex(key_name)}`
    pub fn obj_field(&self, object_id: &str, section_id: &str, key_name: &str) -> String {
        format!(
            "{}{{{object_id}}}:obj:fld:{section_id}:{}",
            self.prefix,
            hex_encode(key_name)
        )
    }

    /// Fields index sorted set (score = version).
    /// Pattern: `le:{oid}:obj:flds`
    pub fn obj_fields_index(&self, object_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:flds", self.prefix)
    }

    /// Time sample hash.
    /// Pattern: `le:{oid}:obj:ts:{section_id}:{time_hex}`
    ///
    /// `time_hex` is [`hex_encode_f64`], bit-exact, so `-0.0` names a
    /// different key from `0.0`: a value write only ever names a canonical
    /// time, and a `-0.0` key is named only to read, delete or merge a legacy
    /// row (see [`hex_encode_f64`]).
    pub fn obj_time_sample(&self, object_id: &str, section_id: &str, time: f64) -> String {
        format!(
            "{}{{{object_id}}}:obj:ts:{section_id}:{}",
            self.prefix,
            hex_encode_f64(time)
        )
    }

    /// Time samples index sorted set (score = version).
    /// Pattern: `le:{oid}:obj:tss`
    pub fn obj_time_samples_index(&self, object_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:tss", self.prefix)
    }

    /// Deleted section tombstone hash.
    /// Pattern: `le:{oid}:obj:del:{section_id}`
    pub fn obj_deleted_section(&self, object_id: &str, section_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:del:{section_id}", self.prefix)
    }

    /// Deleted sections index set.
    /// Pattern: `le:{oid}:obj:dels`
    pub fn obj_deleted_sections_index(&self, object_id: &str) -> String {
        format!("{}{{{object_id}}}:obj:dels", self.prefix)
    }

    /// Per-layer Streams key carrying live-update entries for the object.
    /// Pattern: `le:{oid}:obj:stream`
    ///
    /// Hash-tagged on `object_id` so the stream co-locates on the same hash
    /// slot as the object's other keys, allowing `XADD` inside the same
    /// `MULTI/EXEC` as the state mutation.
    pub fn layer_stream(&self, object_id: u64) -> String {
        format!("{}{{{object_id}}}:obj:stream", self.prefix)
    }

    /// Wake stream of one stream-multiplexer reader.
    /// Pattern: `le:{<tag>}:mux:wake:<instance>:<reader>`
    ///
    /// `tag` must hash to the slot of the layer streams the reader's XREAD
    /// names, since the wake key rides in the same XREAD.
    pub fn stream_mux_wake(&self, tag: &str, instance: &str, reader: usize) -> String {
        format!("{}{{{tag}}}:mux:wake:{instance}:{reader}", self.prefix)
    }

    // ── External value keys (hash tag: {hash_hex} for meta+data co-location) ──

    /// External value metadata hash.
    /// Pattern: `le:ext:meta:{hash_hex}`
    ///
    /// Three fields:
    /// - `ref_count`: live objects referencing the hash;
    /// - `last_store_ms`: the Valkey-side millisecond timestamp of the most
    ///   recent pre-flight store, written by `scripts/ext_store_value.lua` and
    ///   read by `storage::ext_reconcile` and the backup repair;
    /// - `backed_up_in`: the backup confirmation, `<store id>:<ms>`, present
    ///   only once a backup store answered that it holds the bytes; written by
    ///   `scripts/ext_backup_confirm.lua` and cleared by the checkpoint collect
    ///   (`storage::ext_backup`). Its absence means the copy is owed.
    ///
    /// The GC deletes the whole hash, every field with it, when it collects
    /// the value.
    ///
    /// The hash-hex string is wrapped in `{…}` so that `ext_meta` and
    /// `ext_data` for the same content hash share a Valkey Cluster slot.
    /// This lets the writer atomically create both in one single-slot Lua
    /// script (SDD §11.1) and lets GC atomically delete both.
    pub fn ext_meta(&self, hash_hex: &str) -> String {
        format!("{}ext:meta:{{{hash_hex}}}", self.prefix)
    }

    /// External value data blob.
    /// Pattern: `le:ext:data:{hash_hex}`
    ///
    /// Content-addressed: the key is derived directly from the value's content
    /// hash (its `valueOrExtHash`; see `external_values::ext_content_hash`),
    /// eliminating the pointer-indirection through `value_id` used in earlier
    /// revisions of the schema. Shares a Valkey Cluster slot with
    /// [`Self::ext_meta`] for the same hash via the `{hash_hex}` hash tag.
    /// Written set-if-absent by `scripts/ext_store_value.lua`: a later store
    /// does not overwrite the copy stored under a hash, except that a verified
    /// store (the live upload path under `LIVE_EDIT_EXT_VALUE_VERIFY=enforce`)
    /// replaces a key that is not a string or whose length differs from the
    /// verified bytes.
    ///
    /// **A new reader of the blob's bytes must follow the GC miss protocol**
    /// (probe; on a miss take a `{gc}` read hold, re-read the layer and
    /// rebuild; SDD §11.2, §11.3): the GC may collect a released blob 5 s
    /// after its release, so a blob a reader's snapshot references can be
    /// gone. Every production call site of this builder is allowlisted, with
    /// the reason it is safe, by the guard test
    /// `storage::ext_blob_reader_guard_test::every_external_value_blob_read_site_under_src_is_allowlisted`
    /// (`tests/integration/storage/ext_blob_reader_guard_test.rs`).
    pub fn ext_data(&self, hash_hex: &str) -> String {
        format!("{}ext:data:{{{hash_hex}}}", self.prefix)
    }

    /// External value pending-release set.
    /// Pattern: `le:ext:{gc}:pending`
    ///
    /// The `{gc}` hash tag co-locates this set with [`Self::ext_inflight`] on
    /// the same Valkey Cluster slot so that the claim step in `process_pending`
    /// — the atomic `SREM pending` + `ZADD inflight` pair issued via
    /// `EVALSHA gc_claim_pending.lua` (`src/storage/scripts/gc_claim_pending.lua`,
    /// b.j9j item 2) — is slot-safe (would otherwise fail with CROSSSLOT).
    /// The tag is distinct from the per-hash `{hash_hex}` tag used for
    /// `ext_meta`/`ext_data`, since the GC sets are global across all hashes.
    ///
    /// # Member format (b.qpf; extended by b.yda)
    ///
    /// A member is `{hash_hex}:{object_id}:{event}` — see
    /// `storage::external_values::ExtRelease` — or a legacy bare `{hash_hex}`
    /// written by a pre-b.qpf replica. A `hash_hex` cannot contain `:`, so the
    /// prefix up to the first `:` is the hash under both spellings
    /// (`external_values::pending_member_hash_hex`).
    ///
    /// b.yda adds a third `event` spelling, `rec{n}`, written by the
    /// reference-count reconciler with `object_id = 0` — the documented
    /// no-object sentinel, safe because `object_id`s are minted by `INCR` from
    /// 1 and can never be 0. `n` is the surplus amount the repair was claimed
    /// against, which is what keeps two repairs of the same hash a pass apart
    /// from collapsing when they carry different amounts. Every GC script is
    /// member-opaque, so the new spelling needed no script change.
    ///
    /// The event is part of the member because a `ref_count` decrement is
    /// *counted*, not idempotent: with the bare-`hash_hex` member, two
    /// independently-owed decrements for one hash (two objects retiring before
    /// a GC tick; a rewrite and a retirement) collapsed into one Set member and
    /// the second reference leaked permanently. Naming the event keeps SADD
    /// idempotent *within* an event — which the `ext_gc_scheduled` guard and
    /// `delete_layer`'s retry rely on — while making distinct events distinct.
    ///
    /// # The release ledger beside it (b.yuo)
    ///
    /// The Set stays the membership and idempotence authority, unchanged.
    /// Three companion keys account it — [`Self::ext_pending_age`],
    /// [`Self::ext_pending_bytes`] and [`Self::ext_pending_total`] — and every
    /// production enqueue writes the member and its ledger entries in one
    /// script (`scripts/ext_release_enqueue.lua`); no production path `SADD`s
    /// here directly. A member with no age entry (an older build's bare
    /// `SADD`) is stamped with one by the GC loop before it can be collected
    /// (b.1jt).
    ///
    /// # Only obligations no decrement was attempted for (b.1jt)
    ///
    /// This build never moves a member back here once it is claimed — a stale
    /// claim is re-claimed in place in [`Self::ext_inflight`] — and the enqueue
    /// refuses a member still claimed, settled within the replay window, or
    /// accounted. So every member here is one no decrement of this build was
    /// attempted for, and an older build's GC that inherits this Set (a
    /// rollback; record-blind in 0.1.0 through 0.2.0-rc.47) decrements each
    /// correctly, once.
    pub fn ext_pending(&self) -> String {
        format!("{}ext:{{gc}}:pending", self.prefix)
    }

    /// External value in-flight GC set. Sorted Set scored by claim time —
    /// each member is a [`Self::ext_pending`] member (b.qpf:
    /// `{hash_hex}:{object_id}:{event}`, including b.yda's `rec{n}` event, or a
    /// legacy bare `{hash_hex}`), moved
    /// across verbatim by the claim script (`scripts/gc_claim_pending.lua`,
    /// b.j9j item 2), which atomically `SREM`s `pending` and, if the SREM
    /// removed it, `ZADD`s it here.
    /// Pattern: `le:ext:{gc}:inflight`
    ///
    /// # The score is encoded (b.1jt)
    ///
    /// This build scores a claim with the `{gc}` primary's claim time in ms
    /// PLUS `external_values::GC_CLAIM_SCORE_OFFSET_MS` (10^13). Every older
    /// build scores with the bare claim time, so a score below the offset is an
    /// older build's claim, and every reader decodes by that rule:
    ///
    /// - **Written** by `scripts/gc_claim_pending.lua` (a fresh claim) and
    ///   `scripts/gc_reclaim_stale.lua` (an in-place re-claim of an own claim
    ///   older than the re-claim age D, 30 s in production, only while its
    ///   score is unchanged) — the only scripts that write a score here.
    /// - **Read** by `scripts/gc_select.lua`: own claims older than D (to
    ///   re-claim) and an older build's claims older than the decrement fence F
    ///   (to drop).
    /// - **Removed** by `scripts/gc_complete.lua` once a decrement was decided,
    ///   and by `scripts/gc_drop_foreign_claims.lua` for an older build's claim,
    ///   which this build never decrements.
    /// - Membership only (no score) is read by the enqueue's replay refusal,
    ///   the selection's settled check, the ledger audit and the
    ///   reference-count reconciler.
    ///
    /// No claim ever goes back to [`Self::ext_pending`]: an older build's drain
    /// moves claims scored in `[0, now − 30 s]` back there for its GC
    /// (record-blind in 0.1.0 through 0.2.0-rc.47), and an encoded score is
    /// never in that range, so a
    /// member this build may have decremented never reaches that GC again.
    /// The claim time the decrement's claim-age fence checks is the claim
    /// script's (unencoded) reply, not the score.
    ///
    /// Shares the `{gc}` hash tag with [`Self::ext_pending`] so the two sets
    /// live on the same Valkey Cluster slot (see [`Self::ext_pending`]).
    pub fn ext_inflight(&self) -> String {
        format!("{}ext:{{gc}}:inflight", self.prefix)
    }

    /// Release-ledger age index (b.yuo). Sorted Set whose members are
    /// [`Self::ext_pending`] members and whose score is the **Valkey-side**
    /// millisecond time the member was FIRST enqueued.
    /// Pattern: `le:ext:{gc}:pending:age`
    ///
    /// One of three companion keys beside the unchanged pending Set (with
    /// [`Self::ext_pending_bytes`] and [`Self::ext_pending_total`]). The Set
    /// stays the membership and idempotence authority; this index only orders
    /// it, so the GC collects oldest first and never a member younger than its
    /// age floor.
    ///
    /// # Lifecycle
    ///
    /// - **Written** by `scripts/ext_release_enqueue.lua` (`ZADD NX`, only when
    ///   its `SADD` added the member) and by `scripts/gc_stamp_unaged.lua`
    ///   (`ZADD NX` with the current time, for an un-aged pending member;
    ///   b.1jt).
    /// - **Read** by `scripts/gc_select.lua` (candidate selection, oldest age,
    ///   and the un-aged sample).
    /// - **Removed** by `scripts/gc_claim_pending.lua` when it claims the member
    ///   (or self-heals a leftover one), by `scripts/gc_drop_foreign_claims.lua`
    ///   when it drops an older build's claim of a member not back in
    ///   `pending`, and by `scripts/gc_ledger_audit.lua`.
    ///
    /// A pending member with no entry here is "un-aged" (an older build's bare
    /// `SADD`, or a test fixture). It is never a candidate: `gc_select.lua`
    /// samples such members on every pass and the pass stamps each with the
    /// current time, from which its age floor and the read-hold clip then run.
    /// A stamp is never earlier than the true enqueue, so both stay sound.
    ///
    /// Shares the `{gc}` tag with [`Self::ext_pending`], so every script above
    /// is single-slot. Matches no `SCAN` pattern this service issues.
    pub fn ext_pending_age(&self) -> String {
        format!("{}ext:{{gc}}:pending:age", self.prefix)
    }

    /// Release-ledger byte record (b.yuo). Hash: [`Self::ext_pending`] member
    /// → the bytes that member accounts for (the blob's `STRLEN` when it was
    /// enqueued; `0` for a commit's `surplus_acquire` release, whose decrement
    /// cannot free the blob, and for a member whose `STRLEN` failed).
    /// Pattern: `le:ext:{gc}:pending:bytes`
    ///
    /// # Lifecycle
    ///
    /// - **Written** by `scripts/ext_release_enqueue.lua` (only when the member
    ///   has no field yet, so a re-enqueue never re-counts it), and by
    ///   `scripts/gc_claim_pending.lua`'s `HSETNX <member> 0` on every claim.
    ///   That second write is the ledger trail: every member this build may
    ///   have decremented has a field here until it settles, which is what lets
    ///   `scripts/gc_ledger_audit.lua` find a member an older build completed
    ///   with a bare `ZREM` and settle it.
    /// - **Read** by `scripts/ext_release_enqueue.lua` (a member out of
    ///   `pending` that still holds a field is refused as a replay, b.1jt).
    /// - **Read** by `scripts/gc_complete.lua`, the claim's self-heal, the
    ///   drop of an older build's claim (`scripts/gc_drop_foreign_claims.lua`,
    ///   b.1jt) and the audit, each of which removes the field and subtracts it
    ///   from
    ///   [`Self::ext_pending_total`] at the member's FIRST settle (gated on
    ///   `HDEL` returning 1, so bytes leave the total exactly once); and,
    ///   read-only, by `scripts/gc_select.lua` under pressure, which adds each
    ///   member younger than the age floor only while the bytes of every
    ///   member selected so far, those past the age floor included, do not
    ///   exceed the total's excess over the pressure threshold.
    ///
    /// Same `{gc}` tag as [`Self::ext_pending`]. Matches no `SCAN` pattern.
    pub fn ext_pending_bytes(&self) -> String {
        format!("{}ext:{{gc}}:pending:bytes", self.prefix)
    }

    /// Release-ledger byte total (b.yuo). String holding the sum of
    /// [`Self::ext_pending_bytes`] over every member in pending ∪ inflight.
    /// Pattern: `le:ext:{gc}:pending:total`
    ///
    /// The quantity the release budget `H`
    /// (`LIVE_EDIT_EXT_RELEASE_MAX_BYTES`) bounds: uploads carrying external
    /// values wait, and are finally refused `UNAVAILABLE`, while it is at or
    /// above `H` (`storage::ext_release_gate`), and the GC loop collects under
    /// pressure above `H / 2`. `INCRBY` by `scripts/ext_release_enqueue.lua`,
    /// `DECRBY` by `scripts/gc_complete.lua`, the claim's self-heal, the drop
    /// of an older build's claim (`scripts/gc_drop_foreign_claims.lua`) and
    /// `scripts/gc_ledger_audit.lua`, each gated so a member's bytes are
    /// added once and removed once. Read with a plain `GET` by the upload gate
    /// and by `scripts/gc_select.lua`. Absent reads as 0.
    ///
    /// Same `{gc}` tag as [`Self::ext_pending`]. Matches no `SCAN` pattern.
    pub fn ext_pending_total(&self) -> String {
        format!("{}ext:{{gc}}:pending:total", self.prefix)
    }

    /// Settle carrier of the GC decrement replay records (b.yuo). Sorted Set:
    /// [`Self::ext_pending`] member → the **Valkey-side** millisecond time the
    /// member SETTLED, i.e. left both pending and inflight.
    /// Pattern: `le:ext:{gc}:settled`
    ///
    /// The `{gc}`-side channel that tells the GC loop when a member's replay
    /// record ([`Self::ext_applied`]) can be torn down. Only the `{gc}` slot
    /// knows when a member can no longer be presented for decrement again; the
    /// record lives on the hash's own slot. A member settles once it has left
    /// both sets, by `scripts/gc_complete.lua`, by the claim's self-heal, by the
    /// drop of an older build's claim (`scripts/gc_drop_foreign_claims.lua`),
    /// or by `scripts/gc_ledger_audit.lua`; a later settle of the same member
    /// overwrites the score. While its entry stands, the enqueue refuses the
    /// member as a replay (`scripts/ext_release_enqueue.lua`, b.1jt).
    /// `scripts/gc_select.lua` returns the entries whose
    /// score is at least the replay-record retention `U` old and whose member
    /// is in neither set; the loop then runs `scripts/gc_unmark.lua` on each
    /// member's hash slot and `scripts/gc_settled_retire.lua` here, which
    /// removes an entry only while its score still equals the one read.
    ///
    /// Bounded by the release rate × (U + one pass). Observable as
    /// `live_edit_ext_release_settled_count`. Same `{gc}` tag as
    /// [`Self::ext_pending`]. Matches no `SCAN` pattern.
    pub fn ext_settled(&self) -> String {
        format!("{}ext:{{gc}}:settled", self.prefix)
    }

    /// The GC read hold (b.yuo). HASH with one field per holder:
    /// `<object_id>:<token>` (the token random, one per catch-up miss) →
    /// `<start_ms>:<expires_ms>`, both on the `{gc}` primary's **Valkey-side**
    /// clock. Pattern: `le:ext:{gc}:read_hold`
    ///
    /// While any field is live (`expires_ms` above now),
    /// `scripts/gc_select.lua` selects no member enqueued within one second of
    /// the EARLIEST live field's start, or later — an un-aged member counts as
    /// enqueued when the GC loop stamps it, and is never selected before — so
    /// a reader that missed a blob can re-read under its hold and know that
    /// any blob its fresh read references is still stored.
    ///
    /// # Lifecycle
    ///
    /// - **Written** only by `scripts/ext_read_hold.lua`
    ///   (`external_values::ReadHold`): `take` — run by a catch-up whose build
    ///   found an external value missing, before it reads the layer again —
    ///   writes the holder's field, then prunes the other expired fields; `renew`, every
    ///   few seconds while the re-read and rebuild run, extends a live field
    ///   and never revives a lapsed one. Each write sets the key's expiry to
    ///   `EXT_READ_HOLD_TTL` (15 s) from now. A catch-up that finds every
    ///   blob writes nothing.
    /// - **Read** by `scripts/gc_select.lua` on every GC pass (the earliest
    ///   live start, and the live-field count the leader publishes as
    ///   `live_edit_ext_read_holds`), and by the holder's own `renew`.
    /// - **Removed:** a field by its holder's `HDEL` when the rebuild ends
    ///   (or by a `HDEL` spawned when an unreleased holder is dropped), by
    ///   lapsing one TTL after its last write if neither runs, and by the next
    ///   `take`'s pruning; the key by its expiry after the last write.
    ///
    /// A key of another type fails every GC pass's `gc_select.lua` and every
    /// `take`, until it is removed. Observable as
    /// `live_edit_catchup_ext_value_miss_total`, which counts the misses that
    /// take a hold, and `live_edit_ext_read_holds`. Same `{gc}` tag as
    /// [`Self::ext_pending`]. Matches no `SCAN` pattern.
    pub fn ext_read_hold(&self) -> String {
        format!("{}ext:{{gc}}:read_hold", self.prefix)
    }

    /// The GC's out-of-memory collection request (b.pa1). String: the label of
    /// the replica that asked. Pattern: `le:ext:{gc}:oom:request`
    ///
    /// # Lifecycle
    ///
    /// - **Written** only by `scripts/gc_oom_signal.lua`, with `PX` 10 s
    ///   (`store_oom::OOM_COLLECTION_REQUEST_TTL`), by any replica that
    ///   saw a store write refused `-OOM`
    ///   (`storage::store_oom::StoreOomSignal`), at most once a second per
    ///   replica. A later request overwrites an earlier one.
    /// - **Read and removed** by the GC leader's `GETDEL` at the start of
    ///   every pass, which starts an out-of-memory collection when the
    ///   collection's rate limit allows.
    /// - **Expires** after 10 s when no leader takes it.
    ///
    /// Written past `maxmemory` on purpose (the script's opener), so a
    /// request reaches the leader when the `{gc}` shard is the full one; one
    /// small key. Same `{gc}` tag as [`Self::ext_pending`]. Matches no `SCAN`
    /// pattern this service issues.
    pub fn ext_oom_request(&self) -> String {
        format!("{}ext:{{gc}}:oom:request", self.prefix)
    }

    /// The upload hold of a running out-of-memory collection (b.pa1). String:
    /// a random token, one per collection. Pattern:
    /// `le:ext:{gc}:oom:upload_hold`
    ///
    /// While it exists, every replica's release-budget gate
    /// (`storage::ext_release_gate`) holds uploads carrying external values,
    /// within the same deadline-bounded wait the budget uses; the hold never
    /// refuses an upload on its own.
    ///
    /// # Lifecycle
    ///
    /// - **Written** only by `scripts/gc_oom_signal.lua`, with `PX` 15 s
    ///   (`external_values::OOM_UPLOAD_HOLD_TTL`), by the GC leader as an
    ///   out-of-memory collection starts, and only when the release budget is
    ///   enabled.
    /// - **Read** by the upload gate's `MGET` beside
    ///   [`Self::ext_pending_total`], on every gated upload and every poll.
    /// - **Removed** by the leader's `DEL` when the collection ends, on every
    ///   exit (a drop guard spawns it when the collection is cancelled), and
    ///   by its 15 s expiry if the leader dies.
    ///
    /// Written past `maxmemory` on purpose, like [`Self::ext_oom_request`].
    /// Same `{gc}` tag as [`Self::ext_pending`], so the gate's `MGET` is
    /// single-slot. Matches no `SCAN` pattern this service issues.
    pub fn ext_oom_upload_hold(&self) -> String {
        format!("{}ext:{{gc}}:oom:upload_hold", self.prefix)
    }

    /// GC decrement replay record for one content hash (b.yuo). Sorted Set:
    /// [`Self::ext_pending`] member → the claim time (ms, the `{gc}` primary's
    /// `TIME`) of the latest claim whose decrement reached this hash.
    /// Pattern: `le:ext:applied:{hash_hex}`
    ///
    /// What makes a release member's decrement apply at most once. Written by
    /// `scripts/gc_decrement_and_collect.lua` in the same atomic step as the
    /// `HINCRBY -1` it records; a later run of the same member (fred's
    /// transparent re-send of the `EVALSHA`, or the in-place re-claim of a
    /// stale claim)
    /// finds the record and answers `replayed` without decrementing.
    ///
    /// Carries the `{hash_hex}` tag so it shares a Cluster slot with
    /// [`Self::ext_meta`] and [`Self::ext_data`]: only that slot can check and
    /// decrement atomically. A separate key, never a field of the meta, on
    /// purpose: the meta is deleted at zero, so a record kept inside it would
    /// vanish with it and let a replay decrement a blob re-stored in between.
    ///
    /// # Lifecycle
    ///
    /// - **Created / raised** by the decrement script (a replay raises the
    ///   member's score to the larger claim time and writes nothing else).
    /// - **Torn down** by `scripts/gc_unmark.lua`, which the GC loop runs once
    ///   the member has been settled ([`Self::ext_settled`]) for the replay
    ///   record retention `U`; it removes the member only while its score is
    ///   at or below the settle time, so a newer lifecycle's record survives.
    ///   The key vanishes when its last member goes.
    /// - The decrement's claim-age fence is what makes the teardown safe: a
    ///   copy of a claim can act only until claim time + fence, which is
    ///   before settle + `U`.
    ///
    /// Never deleted by the decrement's `DEL meta data`. Its keys match no
    /// `SCAN` pattern this service issues (in particular not
    /// [`Self::ext_meta_scan_pattern`]).
    pub fn ext_applied(&self, hash_hex: &str) -> String {
        format!("{}ext:applied:{{{hash_hex}}}", self.prefix)
    }

    /// Reference-count reconciler candidate ZSET (b.yda). Sorted Set whose
    /// members are `{hash_hex}:{surplus}` and whose score is the **Valkey-side**
    /// millisecond timestamp at which the reconciler FIRST observed that exact
    /// surplus amount for that hash.
    /// Pattern: `le:ext:{gc}:recon`
    ///
    /// # The amount is part of the member, and that is the safety argument
    ///
    /// The pass compares each `ext:meta`'s `ref_count` against the number of
    /// live objects that reference the hash, and repairs the difference. It
    /// enumerates with `SCAN`, whose guarantee is weak: an object created or
    /// missed during a walk changes the count. Encoding the surplus AMOUNT in
    /// the member is what makes that acceptable — only a surplus that is the
    /// *same value* a full grace later is claimable, and the repair is exactly
    /// that value. Any legitimately-in-flight window (a pre-store awaiting its
    /// commit, a commit awaiting its enqueue, an enqueue awaiting the GC)
    /// resolves in seconds, changes the amount, and produces a different member
    /// that starts its own grace from scratch.
    ///
    /// Driven by the same `scripts/recon_claim_expired.lua` the checkpoint-store
    /// reconciliation carriers use, unmodified — the `ZADD NX`
    /// first-observation pinning is the property both protocols need, and this
    /// one needs it for the same reason: an `XX` or plain `ZADD` would refresh
    /// the score every pass and nothing would ever expire.
    ///
    /// Hash-tagged `{gc}` rather than `{recon}` deliberately: this ZSET is read
    /// in the same pass that reads [`Self::ext_pending`] and
    /// [`Self::ext_inflight`] (the `owed` subtraction), and it feeds repairs
    /// straight back onto `ext_pending`. It belongs to the GC protocol's slot,
    /// not the checkpoint reconciliation's.
    pub fn ext_recon_candidates(&self) -> String {
        format!("{}ext:{{gc}}:recon", self.prefix)
    }

    /// Retirement release-retry work list (b.yda). Sorted Set whose members are
    /// `object_id` strings and whose score is the **Valkey-side** millisecond
    /// timestamp at which the object was first registered.
    /// Pattern: `le:ext:{gc}:retire-retry`
    ///
    /// The index into the per-object carriers at
    /// [`Self::obj_ext_retire_carrier`]. A retirement whose attempt-mode enqueue
    /// failed twice — its outcome in doubt, the members still on the carrier
    /// its begin wrote — registers the `object_id` here, so the orphan sweep's
    /// retire-retry phase can find the work without scanning the keyspace.
    /// Registered through `external_values::recon_register` (`ZADD NX`), claimed
    /// with a zero grace — there is nothing to age out, the obligation is owed
    /// the moment it is registered — and retired with `recon_retire` once the
    /// carrier is settled (an attempt found landed, a re-drive landed, a
    /// cancel) or found empty.
    ///
    /// **Not the authority.** This is a fast path, not the record: the carrier
    /// is. A lost work-list entry costs latency only, because the sweep's phase
    /// B backstop SCANs `le:{*}:obj:extret` directly and finds any carrier this
    /// list forgot. That asymmetry is why the registration comes after the
    /// carrier is written and never before.
    ///
    /// No TTL, for the same reason as the carrier itself.
    pub fn ext_retire_retry(&self) -> String {
        format!("{}ext:{{gc}}:retire-retry", self.prefix)
    }

    /// The `{gc}`-slot record of every retirement enqueue attempt (b.yuo Part
    /// R). HASH: attempt token (32 lowercase hex) → `landed:<ms>:<object_id>` or
    /// `void:<ms>:<object_id>`, where `<ms>` is the `{gc}` primary's `TIME`.
    /// Pattern: `le:ext:{gc}:enqueue_attempt`
    ///
    /// Each attempt is decided here exactly once, on the slot the enqueue
    /// writes. The attempt-mode enqueue (`scripts/ext_release_enqueue.lua`)
    /// checks the token and, only when it is absent, enqueues and records it
    /// `landed`; the drain's resolve (`scripts/ext_enqueue_attempt_resolve.lua`)
    /// records an absent token `void` with `HSETNX`. Both run on this one slot,
    /// so an attempt either landed before the resolve or never lands: a copy of
    /// its enqueue that fred writes later finds `void` and writes nothing, and a
    /// second send under a landed token answers `already`. That is what lets the
    /// drain re-enqueue a carrier's members after an attempt whose reply was
    /// lost, after any delay the service itself decides, without a second
    /// decrement.
    ///
    /// A value is never changed once written, and a record is the only guard
    /// against a SEND under its token, so it is removed only once no send under
    /// the token can still arrive:
    /// - **F1** — by the token's own sender (the retirement, or the drain's
    ///   pay), at once, only when its single send under the token replied
    ///   cleanly and its carrier was then cleared (a best-effort `HDEL`);
    /// - **F2** — otherwise only by the GC loop's attempt audit
    ///   (`scripts/gc_enqueue_attempt_audit.lua`), once older than V
    ///   (`external_values::ENQUEUE_ATTEMPT_VOID_RETENTION`, 24 h): a `void`
    ///   entry always, and a `landed` entry only while its object's
    ///   [`Self::obj_ext_retire_attempts`] no longer holds its token.
    ///
    /// **F3** bounds the senders: no service send is decided more than Y
    /// (`external_values::RETIRE_INTENT_SEND_WINDOW`, 30 s) after its token was
    /// recorded. What is left is transit only — a command whose bytes reach the
    /// store more than V − Y after it was decided, or an earlier copy of a
    /// cleanly-replied send more than settle + U later.
    ///
    /// `{gc}`-tagged, beside [`Self::ext_pending`], which is what lets the
    /// enqueue check and write it in the same script. No TTL: a TTL would
    /// expire the `landed` and `void` records the drain still depends on
    /// together; the audit retires each by its own rule.
    pub fn ext_enqueue_attempts(&self) -> String {
        format!("{}ext:{{gc}}:enqueue_attempt", self.prefix)
    }

    /// `SCAN MATCH` pattern covering every [`Self::ext_meta`] key (b.yda).
    ///
    /// The braces around the hash-hex are literal in the key (they are the
    /// Cluster hash tag), so the glob goes INSIDE them — `le:ext:meta:{*}`, not
    /// `le:ext:meta:*`. Kept here beside the key builder it must track, because
    /// the pattern and [`Self::ext_meta`] are one shape and a change to either
    /// without the other produces a scan that silently matches nothing.
    ///
    /// Keyless, so its caller must issue it through the `is_clustered` fan-out
    /// dispatch (`orphan_sweep::scan_keyspace`); a bare `SCAN` walks one
    /// primary's slot range and would report a fraction of the blob keyspace as
    /// the whole of it.
    pub fn ext_meta_scan_pattern(&self) -> String {
        format!("{}ext:meta:{{*}}", self.prefix)
    }

    /// The `hash_hex` carried by an [`Self::ext_meta`] key, or `None` when the
    /// key does not have that shape (b.yda).
    ///
    /// The strict inverse of [`Self::ext_meta`], used by the reference-count
    /// reconciler to turn a `SCAN` hit back into the hash it is about. Strict on
    /// purpose — a lenient parse would let an unrelated key contribute a
    /// mis-derived hash to the surplus computation, and a surplus is what drives
    /// a decrement. The 64-hex-character check is left to
    /// `external_values::hash_from_hex`, which the caller runs on the result;
    /// this only strips the prefix and the literal hash-tag braces.
    pub fn ext_meta_hash_hex<'k>(&self, key: &'k str) -> Option<&'k str> {
        key.strip_prefix(&format!("{}ext:meta:{{", self.prefix))?
            .strip_suffix('}')
    }

    /// Maintenance leader-election lock (b.kf1 H-stor-4).
    /// Pattern: `le:gc:leader`
    ///
    /// A short-TTL (30 s) lock that gates the external-value GC loop, the
    /// orphan sweep, the reference-count reconciler
    /// (`storage::ext_reconcile`) and the external-value backup repair
    /// (`checkpoint::ext_backup_repair`), so that normally one replica runs them at a
    /// time (two can briefly overlap after a lock takeover; see below). The
    /// previous behavior (every replica sweeps every tick) caused N× load on
    /// Valkey from the SCAN+HGETALL passes that the orphan sweeper performs;
    /// the lock normally confines that work to a single replica.
    ///
    /// # The value is protocol-marked (b.1jt)
    ///
    /// Taken only through `external_values::try_acquire_gc_leader`
    /// (`scripts/gc_leader_acquire.lua`); its value is the holder's
    /// `LeaderIdentity::gc_lock_value()` —
    /// `le-gc/<v>|<service_label>:<instance>`, unique per process, with the
    /// GC protocol version `v` (`leader_lease::GC_LEADER_PROTOCOL_VERSION`) —
    /// and the four consumers in one process share that identity so they
    /// ride one leadership. Every build before the marker (0.1.0 through
    /// 0.2.0-rc.48) writes a bare value (`service_label:instance`, or the bare
    /// label in 0.1.0) through the plain CAS, which claims only an absent key
    /// and refreshes only its own exact value.
    ///
    /// - A value with no marker, or a lower version, is an older build's:
    ///   taken over at once (WARN, `live_edit_gc_older_build_total{event=
    ///   "lock_taken_over"}`), after which the older build starts no new pass
    ///   while the key stays marked.
    /// - A same-version peer is taken over once its TTL is at or below
    ///   `external_values::GC_LEADER_TAKEOVER_BELOW_MS` (10 s), and a marked
    ///   peer of any version when it has no TTL (INFO): failover takes about
    ///   20 s after the leader's last refresh.
    /// - A higher-version holder with a TTL is left alone at any TTL, so this
    ///   build does not lead across a newer GC protocol while the newer holder
    ///   keeps refreshing; if it stops refreshing it lapses within the TTL. A
    ///   same-version holder above the threshold is left alone too.
    ///
    /// Every marked process calls the script once a second from a refresh
    /// ticker beside its GC loop, independent of any pass, renewing only while
    /// that loop is running and making progress (stopped with the loop, and
    /// paused once it has made no progress for
    /// `external_values::GC_PROGRESS_STALL_AFTER`, 20 s;
    /// `external_values::spawn_gc_task`), and again before each GC,
    /// orphan-sweep, reconciler and backup-repair pass. So the leader refreshes the key
    /// mid-pass however long a progressing pass runs, a peer takes it over
    /// when the leader's refreshes stop (its loop died or stalled included),
    /// and the key never lapses while any process of this build with a
    /// running, progressing GC loop can reach the store. The progress gate is
    /// in-process and leaves this key's value unchanged. The only lapse is
    /// every such process losing the store for the TTL (or every one's loop
    /// dying or stalling): an older build — unmarked, or a lower
    /// marked version, which answers `held` to a higher version and so never
    /// keeps the key alive for it — can then lead a pass until a process of
    /// this build reaches the store and takes it back. Safety there, and for an
    /// older pass already running when the key was taken over, rests on the
    /// store state, not the lock (`storage::external_values`, "Older builds
    /// sharing the store"): the exposure is an older pass ignoring read holds
    /// (a catch-up can fail `INTERNAL`), never a double decrement; a future
    /// protocol bump needs the same store-state analysis. After the last
    /// marked process exits the key lapses within the TTL, and an older build
    /// can lead again (a rollback).
    ///
    /// `le:checkpoint:leader` and `le:presence:sweeper:leader` keep the plain
    /// CAS and unmarked value: the checkpoint lock is a correctness lock a
    /// takeover would break, and the presence sweeper renews mid-pass through
    /// `LeaderLease`.
    ///
    /// Four consumers, four independent intervals, one lease — the same shape
    /// SDD §22.7.2 argues for. The 30 s TTL is shorter than the orphan sweep's,
    /// the reconciler's and the backup repair's cadences, and although the refresh ticker keeps a
    /// running pass's key alive while the holder's GC loop runs and makes
    /// progress, no lease can exclude a pass whose holder's refreshes stopped
    /// — they no longer reach the store, or the holder's GC loop died or
    /// stalled and its ticker stopped renewing:
    /// a peer takes the key over while that pass may still run, so duplicate
    /// leadership across a failover is reachable and every consumer must be
    /// safe under it rather than relying on the lock for correctness. Expiring
    /// takeovers make a brief overlap reachable between two marked leaders
    /// too. The GC loop is safe under it (every script it
    /// runs is safe under two sweepers; see `storage::external_values`). The
    /// reconciler is, by construction: its candidate members
    /// are `ZADD NX` (so a second leader cannot re-pin a score) and its repair
    /// members are deterministic strings (so a second leader's `SADD` collapses
    /// into the first's). The backup repair is too: it never deletes, and a
    /// second leader's existence checks, uploads of content-addressed bytes and
    /// stamp-guarded confirmations only repeat the first's.
    ///
    /// Deliberately NOT hash-tagged with `{gc}`. The lock is read/written
    /// by exactly one command at a time and shares no transaction with
    /// `ext:pending` / `ext:inflight`, so co-locating it on the GC slot
    /// would just concentrate operator-visible key activity without an
    /// atomicity benefit.
    pub fn gc_leader_lock(&self) -> String {
        format!("{}gc:leader", self.prefix)
    }

    // ── Global counters ─────────────────────────────────────────────────

    /// Object ID counter (atomic via INCR).
    /// Pattern: `le:counter:object_id`
    pub fn counter_object_id(&self) -> String {
        format!("{}counter:object_id", self.prefix)
    }

    // ── Presence keys (hash tag: {session_id}) ──────────────────────────

    /// Presence members set.
    ///
    /// The orphan sweep's index backstop finds these with
    /// `SCAN MATCH {prefix}{*}:presence`, which relies on this key ending at
    /// `:presence` while every other `{sid}` presence key continues past it;
    /// a change to this shape must be mirrored in that pattern.
    ///
    /// Pattern: `le:{sid}:presence`
    pub fn presence_members(&self, session_id: &str) -> String {
        format!("{}{{{session_id}}}:presence", self.prefix)
    }

    /// Presence user info hash.
    /// Pattern: `le:{sid}:presence:{connection_id}`
    pub fn presence_user(&self, session_id: &str, connection_id: &str) -> String {
        format!("{}{{{session_id}}}:presence:{connection_id}", self.prefix)
    }

    /// Presence lease key (string with TTL).
    /// Pattern: `le:{sid}:presence:{connection_id}:lease`
    pub fn presence_lease(&self, session_id: &str, connection_id: &str) -> String {
        format!(
            "{}{{{session_id}}}:presence:{connection_id}:lease",
            self.prefix
        )
    }

    /// Presence teardown marker (string with TTL).
    ///
    /// Armed by `presence_unregister.lua` with `EX` equal to the configured
    /// lease TTL (`LIVE_EDIT_PRESENCE_LEASE_TTL_SECS`) and never deleted
    /// explicitly. While it exists, `presence_register.lua` and
    /// `presence_refresh.lua` refuse the connection without writing or
    /// publishing anything, which fences a register or refresh that reaches the
    /// store after teardown (a client-side timeout that still ran, or a fred
    /// re-send after a reconnect). It is the first part of a two-part fence:
    /// after it expires, a late register is refused by its own store-clock
    /// deadline (`not_after_ms` in `presence_register.lua`). Provided the
    /// register's caller read its store time while the connection's teardown
    /// had not yet started (the obligation stated on
    /// `storage::presence::register_connection`), that deadline falls before
    /// the marker's expiry, so no register or refresh for a torn-down
    /// connection writes anything however late it arrives.
    ///
    /// Carries the `{session_id}` tag so the scripts can declare it in KEYS
    /// beside the member set, info hash and lease.
    ///
    /// Pattern: `le:{sid}:presence:{connection_id}:gone`
    pub fn presence_gone(&self, session_id: &str, connection_id: &str) -> String {
        format!(
            "{}{{{session_id}}}:presence:{connection_id}:gone",
            self.prefix
        )
    }

    /// Global set of session IDs with at least one live JoinSession stream
    /// somewhere in the fleet (b.w29): the session index.
    ///
    /// A joining connection adds its session twice, by plain `SADD`: before
    /// it registers, and again after its register completes. Lease keepers
    /// re-add it on renewal at most once per session per replica per refresh
    /// interval, and after every rejoin. The orphan sweep's index backstop
    /// re-adds, every orphan-sweep interval, any session whose member set it
    /// finds by keyspace `SCAN` and that is missing from this set — the member
    /// sets no live keeper will re-add. A connection's teardown removes the
    /// entry when it left the session's member set empty, through
    /// `storage::presence::remove_session_from_index` — remove-then-recheck
    /// (`SREM`, then `SCARD` of the member set, then `SADD` back if it is
    /// non-empty). The index is in a different slot from the session's keys,
    /// so no removal can be atomic with a concurrent join; together with the
    /// join's register-then-re-add order, the recheck keeps a registered member
    /// indexed, provided the joining replica survives from its register to its
    /// re-add; a member left un-indexed when it does not is what the backstop
    /// re-adds. The presence sweeper removes the entry, through the same
    /// remove-then-recheck helper, of a session whose sweep left its member
    /// set empty or found it empty, which also clears an entry left by a join
    /// that failed or crashed between its first add and its register. Nothing
    /// else removes an entry.
    ///
    /// The presence sweeper enumerates this set, one `SSCAN` page at a time,
    /// to find the sessions whose member sets may hold stale connections,
    /// including sessions whose only members were on a replica that crashed —
    /// which an enumeration of a replica's own local sessions would never
    /// reach.
    ///
    /// Not hash-tagged — the set is global across all sessions and shares no
    /// transaction with any session-scoped key, so co-locating it onto a
    /// single slot would just concentrate fleet-wide JoinSession traffic on
    /// one Valkey Cluster shard without an atomicity benefit.
    ///
    /// Pattern: `le:presence:active_sessions`
    pub fn presence_active_sessions(&self) -> String {
        format!("{}presence:active_sessions", self.prefix)
    }

    /// Presence-sweeper leader-election lock (b.w29). A short-TTL (30 s)
    /// lock that gates the periodic presence sweep so only one replica runs
    /// it per tick. Taken once per tick through
    /// `external_values::try_acquire_leader_lock` (the shared
    /// acquire-or-refresh CAS) and renewed mid-pass through
    /// `storage::leader_lease::LeaderLease`, which a pass consults before
    /// every sweep script call; its value is the holder's
    /// `LeaderIdentity::lock_value()` (`service_label:instance`).
    ///
    /// Deliberately distinct from `gc_leader_lock()`: the GC + orphan sweeps
    /// share a single leader so their SCAN passes normally serialize on one
    /// replica (two can briefly overlap after a lock takeover), while the presence sweep's workload (`SSCAN` pages of the session
    /// index and of each member set, and one batched check-and-remove script
    /// per chunk of members, on the data pool) is separate from theirs, and
    /// co-locating it onto the same leader would just stretch each tick's
    /// wall-clock budget. Using a dedicated lock lets the two maintenance
    /// loops run on different replicas in steady state while each still
    /// bounds its own load to one replica per tick.
    ///
    /// The 30 s TTL bounds failover lag (`gc_leader_lock`, whose peers take an
    /// expiring lock over, fails over sooner): a crashed leader's lock expires
    /// within one TTL, the next replica's
    /// next attempt acquires it, and the sweep resumes from the start of the
    /// index.
    ///
    /// Pattern: `le:presence:sweeper:leader`
    pub fn presence_sweeper_leader_lock(&self) -> String {
        format!("{}presence:sweeper:leader", self.prefix)
    }

    /// Checkpointer leader-election lock (SDD SR-2.1). A short-TTL (30 s)
    /// lock that gates the periodic checkpoint loop so only one replica writes
    /// a checkpoint per tick. Acquired via `try_acquire_leader_lock` once per
    /// tick and renewed mid-tick through `storage::leader_lease::LeaderLease`;
    /// its value is the holder's `LeaderIdentity::lock_value()`
    /// (`service_label:instance`).
    ///
    /// Deliberately distinct from `gc_leader_lock()` and
    /// `presence_sweeper_leader_lock()`: each of the three maintenance loops
    /// has a different cadence and workload profile (GC + orphan sweep do
    /// SCAN+HGETALL passes; presence sweeping does `SSCAN` pages and one
    /// batched check-and-remove script per chunk of members; the checkpointer
    /// streams a snapshot of layer
    /// state to durable storage). Using three independent locks lets the
    /// loops run on different replicas in steady state while each still
    /// bounds its own load to one replica per tick.
    ///
    /// The 30 s TTL bounds failover lag the same way the sibling leader
    /// locks do: a crashed leader's lock expires within one tick, the next
    /// replica's next attempt acquires it, and the checkpoint loop resumes.
    ///
    /// Pattern: `le:checkpoint:leader`
    pub fn checkpoint_leader_lock(&self) -> String {
        format!("{}checkpoint:leader", self.prefix)
    }

    /// Checkpoint-store orphan-candidate ZSET (b.9qd). Sorted Set whose members
    /// are candidate identifiers — `<object_id>.<snap_seq>` for an orphaned
    /// generation stem, `<object_id>.<snap_seq>.tmp` for a leaked write-temp —
    /// and whose score is the **Valkey-side** millisecond timestamp at which the
    /// reconciliation pass FIRST observed that candidate.
    ///
    /// The score is the entire safety argument. The pass cannot ask the
    /// checkpoint store how old an artifact is: neither adapter exposes a
    /// trustworthy creation time (a blob store's `last_modified` is not the
    /// publish start, and a filesystem `mtime` survives neither a restore nor a
    /// copy). So instead of dating the *artifact*, the protocol dates its own
    /// *observations*: a candidate seen twice, at least one grace window apart,
    /// with no OMPF appearing in between, cannot be a publish still in flight —
    /// the adapter's own `publish_worst_case` bounds how long that could take,
    /// and the grace is a multiple of it. First sighting only registers; the
    /// second acts.
    ///
    /// Hash-tagged `{recon}` so it shares a Cluster slot with
    /// [`Self::checkpoint_reconcile_tombstones`] — the pass touches both within
    /// one logical operation, and a future script that spans them stays
    /// slot-safe. Same reasoning as the `{gc}` tag on
    /// [`Self::ext_pending`] / [`Self::ext_inflight`], whose claim-then-collect
    /// protocol this one mirrors (SDD §11.3).
    ///
    /// Pattern: `le:ckpt:{recon}:orphans`
    pub fn checkpoint_reconcile_orphans(&self) -> String {
        format!("{}ckpt:{{recon}}:orphans", self.prefix)
    }

    /// Checkpoint-store ext-blob tombstone ZSET (b.9qd). Sorted Set whose
    /// members are `hash_hex` strings and whose score is the **Valkey-side**
    /// millisecond timestamp at which a hash was tombstoned — i.e. the moment
    /// the last generation known to reference it was pruned or reaped.
    ///
    /// This is what makes ext-blob reclamation a **claim-then-collect**
    /// protocol instead of an inline cascade. Pre-b.9qd the retention prune
    /// computed `pruned − live` and deleted immediately, which had two defects:
    /// the classification could race a concurrent publish (SDD §22.17.3's
    /// accepted residual), and every failure arm leaked permanently, because the
    /// sidecar naming the hashes was deleted in the same function — "the next
    /// tick re-prunes and catches up" was false, the next tick had nothing left
    /// to read. Recording the hashes here BEFORE the generations are deleted
    /// makes the retry claim true, and re-checking the live set a grace window
    /// later collapses the race window to "still absent after the adapter's own
    /// worst-case publish, times three".
    ///
    /// Tombstoning a hash that is still live is harmless — the collect phase
    /// finds it in the live set and spares it. Deleting a generation without
    /// first recording its hashes is the unrecoverable direction. That
    /// asymmetry is why the ZADD-before-delete ordering is load-bearing.
    ///
    /// Shares the `{recon}` hash tag with
    /// [`Self::checkpoint_reconcile_orphans`] (see there).
    ///
    /// Pattern: `le:ckpt:{recon}:tomb`
    pub fn checkpoint_reconcile_tombstones(&self) -> String {
        format!("{}ckpt:{{recon}}:tomb", self.prefix)
    }

    /// Checkpoint-store dead-object ZSET (b.vn6). Sorted Set whose members are
    /// decimal `object_id` strings and whose score is the **Valkey-side**
    /// millisecond timestamp at which that object was first observed to have
    /// DIED — i.e. the moment a delete path finished retiring it.
    ///
    /// # Why it exists
    ///
    /// Nothing in `delete_layer_inner` or the CreateLayer-overwrite cleanup
    /// touches the checkpoint store, so **every** deleted layer leaks whatever
    /// generations it had accumulated: the retention prune only ever runs for a
    /// layer that is still being checkpointed, and the orphan sweep only reaps
    /// *incomplete* stems, so a dead object's complete generations are never
    /// revisited by anything. This ZSET is the durable work list that closes
    /// that leak — the reconciliation pass drains it, reaping every generation
    /// of every object on it.
    ///
    /// # The DR-safety predicate — read this before adding a producer
    ///
    /// **A member exists ONLY because a code path OBSERVED that this layer is
    /// gone and put it there. Nothing infers membership, and nothing may.**
    ///
    /// The producers, exhaustively — all of them route through
    /// `recon_register`, and all of them hold positive evidence of death:
    ///
    /// - **The two delete paths.** `LayerStore::delete_layer_inner` (the
    ///   DeleteLayer RPC) and the CreateLayer-overwrite cleanup, each at its
    ///   success-only exit: the layer was deleted, by this call, just now.
    /// - **The four refused-stamp arms** (b.vn6). The checkpoint loop's periodic
    ///   snapshot, tail capture and chain-collapse re-base, plus the
    ///   creation-time base bootstrap. Each has just PUT a durable generation
    ///   and then had its `obj:hdr` cursor stamp refused by
    ///   `stamp_cursor_if_live.lua` — a server-side conditional whose `false` is
    ///   the atomic observation "this header no longer exists". That is evidence
    ///   of the same quality as the delete paths', from the other side of the
    ///   race, and it is what re-arms a member the pass may already have retired
    ///   under a generation that landed after the sweep looked.
    ///
    /// Note the shape a new producer must have: *an act or an atomic read that
    /// established the layer is gone*. Absence observed at a distance is not
    /// that. The tempting alternative — sweep the store for object ids with no
    /// `obj:hdr` and no session→layer mapping — is a data-loss bug rather than
    /// an optimisation, because a layer's restorability depends on a live mapping
    /// that session-checkpoint apply re-creates only LAZILY, when a client next
    /// touches the session. After a keyspace wipe every surviving base
    /// therefore satisfies "no header and no mapping" until someone touches it,
    /// and a reaper keyed on that predicate would delete the entire DR corpus
    /// precisely when it is the only copy. Positive evidence of death is the
    /// only admissible input here.
    ///
    /// The failure direction that leaves is benign and deliberate: if the ZSET
    /// is lost in a wipe, the pending reaps are simply forgotten. That is a
    /// storage leak, never a loss.
    ///
    /// Same `ZADD NX` first-observation scoring as the two sibling `{recon}`
    /// ZSETs — the score is pinned by the first writer and never refreshed —
    /// and, like them, **no TTL**: an expiry would silently drop reclamation
    /// work whose only record this is.
    ///
    /// Shares the `{recon}` hash tag with
    /// [`Self::checkpoint_reconcile_orphans`] /
    /// [`Self::checkpoint_reconcile_tombstones`] (see the first for why).
    ///
    /// Pattern: `le:ckpt:{recon}:dead`
    pub fn checkpoint_reconcile_dead(&self) -> String {
        format!("{}ckpt:{{recon}}:dead", self.prefix)
    }

    // ── Sorted set member formatters ──────────────────────────────────

    /// Format a fields sorted set member.
    /// Pattern: `{section_id}:{hex(key_name)}`
    ///
    /// Used as the member string in `obj_fields_index` sorted sets (§7.3).
    pub fn field_member(section_id: &str, key_name: &str) -> String {
        format!("{section_id}:{}", hex_encode(key_name))
    }

    /// Build a `path_names_index` set member from a session `(name, session_id)`
    /// pair (b.91j C-7). Composition: `name\0session_id`. NUL is forbidden in
    /// session names by `validate_session_name`, so the separator cannot
    /// collide with name content.
    pub fn path_name_member(name: &str, session_id: &str) -> String {
        format!("{name}\0{session_id}")
    }

    /// Split a `path_names_index` set member back into `(name, session_id)`.
    /// Returns `None` if the encoded form is malformed (no separator). Used by
    /// [`crate::storage::session::SessionStore::list`].
    pub fn split_path_name_member(member: &str) -> Option<(&str, &str)> {
        member.split_once('\0')
    }

    /// Format a time samples sorted set member.
    /// Pattern: `{section_id}:{hex_f64(time)}`
    ///
    /// Used as the member string in `obj_time_samples_index` sorted sets.
    /// Bit-exact like [`Self::obj_time_sample`]: a `-0.0` member
    /// (`{section_id}:8000000000000000`) is a legacy row's, named only to
    /// read, delete or merge it; the reconciler's signed-zero merge looks
    /// for it.
    pub fn time_sample_member(section_id: &str, time: f64) -> String {
        format!("{section_id}:{}", hex_encode_f64(time))
    }

    /// Reconstruct the full `obj_field` key for an object from a
    /// `{section_id}:{hex(key_name)}` sorted-set member read out of
    /// `obj_fields_index` (b.k1r medium).
    ///
    /// Pre-b.k1r callers (`cleanup_old_object` in `layer.rs` and the
    /// orphan sweep in `orphan_sweep.rs`) inline-concatenated this key
    /// shape by hand, duplicating both the `"{prefix}{{oid}}:obj:fld:"`
    /// template AND the knowledge that the member string already carries
    /// `section_id:hex(key_name)` in the right order. A future rename of
    /// either the prefix pattern or the member format would silently
    /// desynchronize the key builder from those call sites and leak
    /// orphan keys. This helper keeps all that knowledge on `KeySchema`.
    ///
    /// Note we do NOT re-hex-encode the key name: `member` already carries
    /// the hex form read back from the sorted set. The inverse pairing
    /// with [`Self::field_member`] is an identity for `(section_id,
    /// hex(key_name))` round-trips.
    pub fn obj_field_from_member(&self, object_id: &str, member: &str) -> String {
        format!("{}{{{object_id}}}:obj:fld:{member}", self.prefix)
    }

    /// Reconstruct the full `obj_time_sample` key for an object from a
    /// `{section_id}:{hex_f64(time)}` sorted-set member read out of
    /// `obj_time_samples_index` (b.k1r medium). Same rationale as
    /// [`Self::obj_field_from_member`] — pre-b.k1r the orphan sweep and
    /// the overwrite cleanup duplicated this concat.
    pub fn obj_ts_from_member(&self, object_id: &str, member: &str) -> String {
        format!("{}{{{object_id}}}:obj:ts:{member}", self.prefix)
    }

    /// Build the complete UNLINK key inventory for a retiring object, given
    /// the five index enumerations its caller has already read out of Valkey
    /// (b.9fn; fifth index added by b.jdr).
    ///
    /// The inventory authority for every consumer that removes an object's
    /// keys (b.6de): all three object-retirement routes —
    /// `layer.rs::cleanup_old_object` (DeleteLayer, session-deletion cascade)
    /// and `layer.rs::cleanup_old_object_retrying` (CreateLayer overwrite),
    /// and `orphan_sweep.rs::cleanup_orphaned_object` (sweep reclamation) —
    /// plus the DR restore's step-3b residue wipe,
    /// `checkpoint::residue::wipe_restore_residue`, which is not a retirement
    /// at all: it clears surviving `obj:*` residue so the restore is an
    /// authoritative REPLACE rather than a union. The residue wipe feeds a
    /// **filtered copy** of this Vec to its atomic wipe — `obj:ext` is removed
    /// because it is the materialization's external-value pre-image and must
    /// outlive the wipe (b.qpf × b.f4t; see that function). The three
    /// retirement routes never build the whole Vec: they run the paged teardown
    /// (`storage::retirement::Teardown`), which unlinks the same
    /// keys page by page, built from the same pieces this list is built from
    /// ([`Self::object_teardown_first_keys`] — the teardown's first UNLINK —
    /// the per-member keys of the index pages, and
    /// [`Self::object_teardown_index_keys`] last), so the inventory cannot
    /// drift between them. Each consumer keeps its own enumerate commands, pool
    /// selection, error mapping, retry posture, and fault-injection seams.
    ///
    /// # Ordering invariant — header first, and it is load-bearing
    ///
    /// **Do not reorder (b.vd8, b.pw6).** The header is element 0 here, and
    /// every consumer removes it first: the residue wipe feeds the
    /// `obj:ext`-less copy of this Vec to an UNLINK that preserves order, and
    /// the paged teardown's first UNLINK starts with it, so the header is
    /// removed before any section / field / time-sample key. Cleanup is the
    /// one mutator of these keys that does NOT bump the
    /// header sequence and does NOT wait for a snapshot fence, so every
    /// snapshot read that proves its state by the header depends on this
    /// order (SDD §7.9 step 7b, §22.4):
    ///
    /// - **The fenced read** — catch-up's shared fenced-read flights and the
    ///   checkpoint assembler's fenced read, above the atomic-read limits.
    ///   Any cleanup far enough along to have deleted a key a fenced pass
    ///   read has necessarily already deleted the header, so the fence's
    ///   release (`snapshot_fence_release.lua`) answers `NOT_FOUND` rather
    ///   than `VERIFIED`, and a torn snapshot is never proven. The fenced
    ///   read has no second check behind the release.
    /// - **The atomic read** — it cannot interleave with one cleanup command
    ///   but can run between two, and only the header going first makes every
    ///   such read answer `NotFound` instead of returning a half-removed
    ///   object as the state at its cut.
    /// - **The structural `UpdateLayer` path's proof** — its one
    ///   `HGET obj:hdr seq` after its unfenced fan-out (Spec SDD SR-4.3) finds
    ///   no `seq` and the attempt is `Aborted`, rather than confirming a torn
    ///   snapshot. This path also has the header `WATCH` at `EXEC` behind it.
    ///
    /// Deleting the header later would open a window where a read sees
    /// partially-cleaned state while the header still answers with an
    /// unchanged sequence.
    ///
    /// # The per-object stream is never in the inventory
    ///
    /// `le:{oid}:obj:stream` is not emitted. Both retirement consumers first
    /// append a `deleted` entry to it, so a subscriber still tailing the object
    /// wakes and re-reads the layer mapping, and then give it an expiry
    /// (`PEXPIRE` with a grace window) BEFORE they unlink any of this
    /// inventory, so a
    /// subscriber lagging between `XREAD` round trips can still read that entry
    /// (SDD §7.9 step 7, §12.4). Unlinking the stream with the object would cut
    /// that grace short. The residue wipe leaves it for its own reason: the
    /// restore adjudicates that stream separately in `checkpoint::recovery`,
    /// and a wipe must not pre-empt it.
    ///
    /// # Coverage
    ///
    /// Every `obj:*` family is represented — `hdr`, `secs`, `sec:*`, `flds`,
    /// `fld:*`, `tss`, `ts:*`, `dels`, `del:*`, `chs`, `ch:*`, `ext`, `wtok`,
    /// `wrel`, `snapfence` and `snapintent` — except the per-object `stream`,
    /// which expires on its own (above), and **one deliberate exclusion**: the
    /// retirement intent, a carrier and its attempt tokens.
    ///
    /// [`Self::obj_ext_retire_carrier`] (`obj:extret`, b.yda) and
    /// [`Self::obj_ext_retire_attempts`] (`obj:extret:attempts`, b.yuo) are NOT
    /// in the inventory and must not be added. The retirement writes them before
    /// its release enqueue, and when that enqueue's outcome is in doubt the
    /// teardown runs anyway: the two keys are then the only record of the
    /// obligation and of the attempts that may already have paid it. Including
    /// them here would delete that record microseconds later — turning a
    /// recoverable in-doubt release back into the standing leak the carrier
    /// exists to prevent, and, for the tokens, discarding what stops a re-drive
    /// from paying a landed attempt a second time. They are the `obj:*` keys
    /// that deliberately outlive their object, and they are removed only by the
    /// retirement intent's own scripts, both keys in one step and only while
    /// the carrier exists: the clear, once an attempt is known landed; the
    /// settle, when it cancels. A token HASH with no carrier (only an older
    /// build leaves one) is removed by nothing in this build; b.9ud sub-task 3
    /// plans the only safe remover (the restore's residue wipe, extended to
    /// void the tokens and delete the HASH — not built).
    ///
    /// **The exclusion is about the two RETIREMENT consumers; the residue
    /// wipe is made safe elsewhere.** `layer::cleanup_old_object`
    /// (DeleteLayer), `layer::cleanup_old_object_retrying` (CreateLayer
    /// overwrite) and
    /// `orphan_sweep::cleanup_orphaned_object` are terminal: the `object_id`
    /// dies with them and is never re-issued (ids come from
    /// `INCR le:counter:object_id`), so a carrier they leave behind can only
    /// ever be drained against the retirement that wrote it, and the exclusion
    /// above is the whole argument for those two.
    ///
    /// The residue wipe, `checkpoint::residue.rs`'s, is
    /// NOT a retirement: it re-materialises the **same** `object_id`, so a stale
    /// carrier and a freshly-restored object CAN share one id. What makes that
    /// safe is not this inventory but the drain's settle
    /// (`scripts/ext_retire_settle.lua`), which reads `obj:hdr` and `obj:ext` in
    /// the same atomic step as its pay-or-cancel decision and **cancels** —
    /// removes the carrier and its tokens, clears `ext_gc_scheduled` — whenever
    /// either exists. A live incarnation has inherited the standing, unreleased
    /// `+1` per hash those members stand for (a base-only restore pre-stores
    /// nothing yet rebuilds the external rows, which is why the DR-restore
    /// materialiser — the `SurplusPolicy::Suppress` entry point in
    /// `storage::layer` — suppresses surplus acquires), so the obligation is
    /// genuinely no longer owed, and cancelling is correct rather than merely
    /// cautious. `obj:ext` is in the gate because the wipe deliberately
    /// WITHHOLDS it while unlinking `obj:hdr`, leaving a window in which the
    /// header is absent and the index stands; on a genuinely dead object the two
    /// go in one `UNLINK`, the teardown's first UNLINK
    /// ([`Self::object_teardown_first_keys`]), so `obj:ext` without a header
    /// never means "dead". A carrier one of
    /// whose attempts already landed is not gated: it was paid, and the drain
    /// only clears it (policy 12). An older build's SET carrier is never paid
    /// at all — cleared or dropped (b.1jt) — so the gate concerns only this
    /// build's HASH carriers.
    ///
    /// Adding either key here would still re-open the permanent leak the
    /// exclusion exists to prevent.
    ///
    /// One residual is acknowledged rather than closed: a settle taken while both
    /// keys are absent, whose re-drive onto `le:ext:{gc}:pending` is still in
    /// flight when a restore lands at that id, is drained by the GC against the
    /// new incarnation. It is not closable inside the `{object_id}`-slot
    /// discipline (the enqueue is on the `{gc}` slot), and it is a facet of the
    /// pre-existing "a checkpoint restore materialises external references
    /// without acquiring them" defect, filed as its own checkpoint-lane Issue.
    ///
    /// The `ch:*` family is the only one whose keys are not named by a section /
    /// field / time-sample id, so it gets its parent ids from two sources whose
    /// **union** is emitted (deduped, first occurrence wins):
    ///
    /// 1. **The recorded index** `obj:chs` ([`Self::obj_children_index`],
    ///    b.jdr) — every parent id that actually owns a children list, written
    ///    in the same atomic unit as the list itself (the same `MULTI/EXEC` on
    ///    the WATCH path and the materializers, the same script for
    ///    `update_layer_fast.lua`'s structural-mode `CHL` / `DCHL` records).
    ///    This is what makes
    ///    the family enumerable at all: a parent id need not be a section.
    ///    `CommandDiffChangeSection` does not validate parent existence (SDD
    ///    §10.4; the server itself emits such deltas in catch-up diffs, and
    ///    validation rejects only self-parenting), and earlier builds stored a
    ///    list under such a *dangling* parent, which is in neither `obj:secs`
    ///    nor `obj:dels` — chiefly from `apply_diff_change_section`, and also
    ///    from a create or move whose name-conflict cascade deleted the parent
    ///    it named. The current
    ///    engine stores none, but a list an earlier build stored can still be
    ///    here at retirement.
    /// 2. **The pre-b.jdr reconstruction** — *sections ∪ tombstones ∪ root `0`*.
    ///    Deliberately KEPT rather than replaced by (1):
    ///    - **Rolling-deploy cover.** An object materialized by a pre-b.jdr
    ///      replica carries no `obj:chs`; retired by a post-b.jdr replica, the
    ///      index enumerates empty and the reconstruction is the whole
    ///      inventory. Dropping it would regress b.9fn's root-`0` fix for the
    ///      lifetime of every object created before the upgrade.
    ///      The reverse direction is *not* covered and cannot be: a pre-b.jdr
    ///      replica retiring a post-b.jdr object does not know `obj:chs`
    ///      exists, so it strands that one Set key — and, having unlinked the
    ///      header, leaves nothing for the orphan sweep to rediscover it from.
    ///      Bounded to the rolling window, one small key per object affected,
    ///      and unavoidable for any newly-introduced index key.
    ///    - **Defence in depth**, for free: an index that lost a member (see
    ///      below) still reaps every well-formed parent.
    ///
    ///   Both drift directions are *individually* benign. One stale extra
    ///   member costs one `UNLINK` of a key that does not exist — a no-op. One
    ///   missing member degrades to exactly the pre-b.jdr behaviour, i.e. the
    ///   reconstruction alone. In aggregate the stale direction is not free:
    ///   stale members accrete without bound on a `noeviction` store, so the
    ///   `SREM` forks in the `SetChildren` / `DeleteChildrenList` mutation arms
    ///   (`storage::layer`) are load-bearing for the Set's size, not an
    ///   optimisation — see [`Self::obj_children_index`]. What is *not* benign
    ///   even once is a consumer substituting an empty enumeration for a failed
    ///   read of `obj:chs` (the residue wipe's `SMEMBERS`, the paged teardown's
    ///   `SSCAN`), which silently re-opens the leak; both therefore propagate
    ///   that error rather than `unwrap_or_default()`ing it (b.u96 item 1 /
    ///   b.j9j item 3).
    ///
    /// The per-tombstone `ch:*` from (2) stays belt-and-suspenders:
    /// `cascade_delete_section` normally already emitted an explicit
    /// `DeleteChildrenList`, which also drops the id from `obj:chs`.
    pub(crate) fn object_cleanup_keys(
        &self,
        object_id: &str,
        section_ids: &[String],
        field_members: &[String],
        ts_members: &[String],
        tombstone_ids: &[String],
        children_parent_ids: &[String],
    ) -> Vec<String> {
        // Header FIRST — see the ordering invariant in the doc-comment. The
        // dedupe pass below preserves first-occurrence order, so element 0
        // stays the header no matter what the enumerations contain. Built from
        // the paged teardown's own pieces
        // ([`Self::object_teardown_first_keys`],
        // [`Self::object_teardown_index_keys`] and the per-member helpers), so
        // the two cannot drift on which keys an object owns: the header, then
        // the five indices, then the rest of the first UNLINK's fixed keys.
        let mut first = self.object_teardown_first_keys(object_id).into_iter();
        let mut keys: Vec<String> = first.next().into_iter().collect();
        keys.extend(self.object_teardown_index_keys(object_id));
        keys.extend(first);

        // Recorded parent ids (b.jdr). Emitted before the reconstruction so a
        // parent that is both indexed and a live section is deduped down to
        // the single key it names either way.
        for parent_id in children_parent_ids {
            keys.push(self.obj_children(object_id, parent_id));
        }

        for sid in section_ids {
            keys.extend(self.section_teardown_keys(object_id, sid));
        }

        for member in field_members {
            keys.push(self.obj_field_from_member(object_id, member));
        }

        for member in ts_members {
            keys.push(self.obj_ts_from_member(object_id, member));
        }

        for sid in tombstone_ids {
            keys.extend(self.tombstone_teardown_keys(object_id, sid));
        }

        // The index and the reconstruction overlap by construction (every live
        // section that owns children is in both), so dedupe. `retain` keeps the
        // FIRST occurrence of each key, which is what preserves the header-first
        // ordering invariant above — do not swap this for a sort-based dedupe.
        let mut seen: std::collections::HashSet<String> =
            std::collections::HashSet::with_capacity(keys.len());
        keys.retain(|k| seen.insert(k.clone()));

        keys
    }

    /// The paged teardown's **first `UNLINK`** for a retiring object
    /// (`storage::retirement::Teardown`): every fixed
    /// object-level key but the five indices, header first — `obj:hdr`,
    /// `obj:ext`, `obj:wtok`, `obj:wrel`, `obj:snapfence`, `obj:snapintent` and
    /// the root children list `obj:ch:0`. The row pages follow; the five
    /// indices ([`Self::object_teardown_index_keys`]) go last, so a teardown
    /// stopped partway still has the indices that name its remaining rows.
    ///
    /// **`obj:hdr` and `obj:ext` always go in one command.**
    /// `scripts/ext_retire_settle.lua` treats an object as live while either
    /// exists (`EXISTS obj:hdr OR EXISTS obj:ext`) and then cancels the
    /// carrier, so an `obj:ext` that outlived its header — for the length of
    /// the row pages, or for good when a later page failed — would cancel a
    /// dead object's in-doubt carrier and leak every release it recorded.
    /// `obj:ext`'s contents are consumed before the teardown runs (the
    /// retirement's reference walk pages it), so unlinking it first costs
    /// nothing.
    ///
    /// Each key's reason to be here, and its rolling-deploy cover:
    ///
    /// - `obj:ext` (b.qpf) is unconditional for the same rolling-deploy reason
    ///   as `obj:ch:0`: a pre-b.qpf object has no such key, and UNLINKing one
    ///   that does not exist is a no-op. The residue wipe
    ///   (`checkpoint::residue::wipe_restore_residue`) `retain`s it OUT of its
    ///   copy of [`Self::object_cleanup_keys`] on purpose: it is the ext
    ///   pre-image `write_restored_object` reads, and its survival is also the
    ///   `obj:ext` liveness leg of the settle that covers the wipe→materialize
    ///   window (b.yda, b.yuo; SDD §11.3.1, §22.6 step 3b).
    /// - `obj:wtok`, the `UpdateLayer` commit-token record, and `obj:wrel`, its
    ///   commit release ring: unconditional for the same rolling-deploy reason
    ///   (an object that never took an `UpdateLayer` has neither), and removed
    ///   by every consumer, the restore residue wipe included, so a restored
    ///   object starts with no record or ring a re-issued call could match or
    ///   enqueue from.
    /// - `obj:snapfence` and `obj:snapintent` (Spec SDD SR-5.1): unconditional
    ///   too, and removed by every consumer: a fence or intent left behind on
    ///   a restored object would hold the restored layer's writers or readers
    ///   to a pre-restore epoch until its PEXPIRE. Their position after the
    ///   header is not load-bearing: once the header is gone, acquire answers
    ///   `NOT_FOUND` and release writes nothing, so neither can recreate the
    ///   key ahead of a later `UNLINK`.
    /// - `obj:ch:0`, the root children-ordering list: object-level, not
    ///   per-section — neither the sections nor the tombstones index contains
    ///   the synthetic root (b.9fn) — and still unconditional though `obj:chs`
    ///   now records the root too: the rolling-deploy cover for pre-b.jdr
    ///   objects (see [`Self::object_cleanup_keys`]).
    ///
    /// No stream key, ever: the per-object stream expires on its own
    /// ([`Self::object_cleanup_keys`], *The per-object stream is never in the
    /// inventory*).
    pub(crate) fn object_teardown_first_keys(&self, object_id: &str) -> Vec<String> {
        vec![
            self.obj_header(object_id),
            self.obj_ext_index(object_id),
            self.obj_write_token(object_id),
            self.obj_write_releases(object_id),
            self.obj_snapshot_fence(object_id),
            self.obj_snapshot_intent(object_id),
            self.obj_children(object_id, ROOT_PARENT_ID),
        ]
    }

    /// The paged teardown's **last `UNLINK`**: the five structural indices —
    /// `obj:secs`, `obj:flds`, `obj:tss`, `obj:dels` and `obj:chs` (b.jdr) —
    /// whose members name the row keys the pages before it remove. Last, so a
    /// teardown that stops partway leaves the indices behind with the rows they
    /// name (a retry pages them again); nothing finds an object by an index key
    /// (discovery reads `obj:hdr` and `obj:extret` only), so an index left
    /// without its header strands keys but never misleads.
    pub(crate) fn object_teardown_index_keys(&self, object_id: &str) -> Vec<String> {
        vec![
            self.obj_sections_index(object_id),
            self.obj_fields_index(object_id),
            self.obj_time_samples_index(object_id),
            self.obj_deleted_sections_index(object_id),
            self.obj_children_index(object_id),
        ]
    }

    /// The keys one `obj:secs` member names in a teardown: the live section's
    /// hash and the children list under it.
    pub(crate) fn section_teardown_keys(&self, object_id: &str, section_id: &str) -> [String; 2] {
        [
            self.obj_section(object_id, section_id),
            self.obj_children(object_id, section_id),
        ]
    }

    /// The keys one `obj:dels` member names in a teardown: the tombstone's
    /// hash and the children list under the deleted section (it may have had
    /// children before deletion; `cascade_delete_section` normally emits an
    /// explicit `DeleteChildrenList`, so this is belt-and-suspenders).
    pub(crate) fn tombstone_teardown_keys(&self, object_id: &str, section_id: &str) -> [String; 2] {
        [
            self.obj_deleted_section(object_id, section_id),
            self.obj_children(object_id, section_id),
        ]
    }

    // ── Pub/Sub channels (no hash tag relevance) ────────────────────────

    /// Presence events pub/sub channel.
    /// Pattern: `le:presence:events:{session_id}`
    pub fn presence_events_channel(&self, session_id: &str) -> String {
        format!("{}presence:events:{session_id}", self.prefix)
    }

    /// Session-deleted pub/sub channel (cross-replica fanout).
    /// Pattern: `le:session:deleted`
    ///
    /// Publishes a session_id payload whenever `DeleteSession` finishes its
    /// Valkey cascade so every replica can drop + cancel its local
    /// `session_tokens` entry (SDD §7.8 cross-replica cancellation).
    pub fn session_deleted_channel(&self) -> String {
        format!("{}session:deleted", self.prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── A. Key builder exact output tests ───────────────────────────────

    #[test]
    fn session_meta_default_prefix_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.session_meta("s1"), "le:{s1}:sess");
    }

    #[test]
    fn session_meta_custom_prefix_returns_expected_key() {
        let ks = KeySchema::new("test_abc:");
        assert_eq!(ks.session_meta("s1"), "test_abc:{s1}:sess");
    }

    #[test]
    fn path_index_hex_encodes_base_path() {
        let ks = KeySchema::default();
        // "/stage" → hex "2f7374616765"; the literal `{…}` wraps the hex so
        // Valkey Cluster slot-routes on the base-path bytes alone.
        assert_eq!(ks.path_index("/stage"), "le:path:{2f7374616765}");
    }

    #[test]
    fn path_names_index_hex_encodes_base_path() {
        let ks = KeySchema::default();
        assert_eq!(
            ks.path_names_index("/stage"),
            "le:path:{2f7374616765}:names"
        );
    }

    /// Regression guard for the b.* cluster-CROSSSLOT bug: `path_index` and
    /// `path_names_index` MUST emit literal `{hex(base_path)}` so both sets
    /// land on the same Cluster slot. Without the braces, the slot is hashed
    /// over the full key, and `SessionStore::create`'s WATCH/SMEMBERS/MULTI
    /// trips `CROSSSLOT Keys in request don't hash to the same slot`.
    #[test]
    fn path_index_and_path_names_index_share_hash_tag() {
        let ks = KeySchema::default();
        let idx = ks.path_index("/stage");
        let names = ks.path_names_index("/stage");

        let extract_tag = |k: &str| -> String {
            let open = k.find('{').expect("hash tag opening brace missing");
            let close = k[open..].find('}').expect("hash tag closing brace missing");
            k[open + 1..open + close].to_owned()
        };

        let idx_tag = extract_tag(&idx);
        let names_tag = extract_tag(&names);
        assert_eq!(
            idx_tag, names_tag,
            "path_index and path_names_index must share a hash tag so MULTI/EXEC \
             across both sets is slot-safe on Valkey Cluster (idx={idx}, names={names})"
        );
        assert!(
            !idx_tag.is_empty(),
            "hash tag must be non-empty (Valkey treats empty `{{}}` as no tag)"
        );
    }

    #[test]
    fn layer_mapping_hex_encodes_layer_name() {
        let ks = KeySchema::default();
        // "root" → hex "726f6f74"
        assert_eq!(ks.layer_mapping("s1", "root"), "le:{s1}:layer:726f6f74");
    }

    #[test]
    fn layers_index_contains_session_hash_tag() {
        let ks = KeySchema::default();
        assert_eq!(ks.layers_index("s1"), "le:{s1}:layers");
    }

    #[test]
    fn obj_header_contains_object_hash_tag() {
        let ks = KeySchema::default();
        assert_eq!(ks.obj_header("42"), "le:{42}:obj:hdr");
    }

    #[test]
    fn obj_section_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.obj_section("42", "sec7"), "le:{42}:obj:sec:sec7");
    }

    #[test]
    fn obj_sections_index_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.obj_sections_index("42"), "le:{42}:obj:secs");
    }

    #[test]
    fn obj_children_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.obj_children("42", "p3"), "le:{42}:obj:ch:p3");
    }

    #[test]
    fn obj_field_hex_encodes_key_name() {
        let ks = KeySchema::default();
        // "color" → hex "636f6c6f72"
        assert_eq!(
            ks.obj_field("42", "sec7", "color"),
            "le:{42}:obj:fld:sec7:636f6c6f72"
        );
    }

    #[test]
    fn obj_fields_index_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.obj_fields_index("42"), "le:{42}:obj:flds");
    }

    #[test]
    fn obj_time_sample_hex_encodes_time() {
        let ks = KeySchema::default();
        // 1.0_f64.to_bits() = 0x3FF0000000000000
        assert_eq!(
            ks.obj_time_sample("42", "sec7", 1.0),
            "le:{42}:obj:ts:sec7:3ff0000000000000"
        );
    }

    #[test]
    fn obj_time_samples_index_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.obj_time_samples_index("42"), "le:{42}:obj:tss");
    }

    #[test]
    fn obj_deleted_section_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.obj_deleted_section("42", "sec7"), "le:{42}:obj:del:sec7");
    }

    #[test]
    fn obj_deleted_sections_index_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.obj_deleted_sections_index("42"), "le:{42}:obj:dels");
    }

    #[test]
    fn layer_stream_contains_object_hash_tag() {
        let ks = KeySchema::default();
        assert_eq!(ks.layer_stream(42), "le:{42}:obj:stream");
    }

    #[test]
    fn layer_stream_custom_prefix_propagates() {
        let ks = KeySchema::new("test_abc:");
        assert_eq!(ks.layer_stream(7), "test_abc:{7}:obj:stream");
    }

    #[test]
    fn layer_stream_co_locates_with_obj_header_on_same_hash_slot() {
        // The stream and the object's header must share the `{object_id}` hash
        // tag so that XADD can run inside the same MULTI/EXEC as the header
        // mutation — same hash slot in Valkey Cluster. A drift here would
        // silently regress the atomicity that Epic 2 relies on.
        let ks = KeySchema::default();
        let stream = ks.layer_stream(42);
        let header = ks.obj_header("42");
        let tag = "{42}";
        assert!(
            stream.contains(tag),
            "layer_stream missing hash tag: {stream}"
        );
        assert!(
            header.contains(tag),
            "obj_header missing hash tag: {header}"
        );
    }

    #[test]
    fn layer_stream_large_object_id_formats_correctly() {
        // u64::MAX must format as decimal digits, not accidentally truncated or
        // hex-encoded — downstream XADD callers pass decimal object IDs.
        let ks = KeySchema::default();
        let key = ks.layer_stream(u64::MAX);
        assert_eq!(key, format!("le:{{{}}}:obj:stream", u64::MAX));
    }

    #[test]
    fn stream_mux_wake_uses_prefix_and_tag_and_matches_no_sweeper_pattern() {
        let ks = KeySchema::new("test_x:");
        let key = ks.stream_mux_wake("mx7", "inst", 3);
        assert_eq!(key, "test_x:{mx7}:mux:wake:inst:3");
        assert_eq!(
            fred::util::redis_keyslot(key.as_bytes()),
            fred::util::redis_keyslot(b"mx7")
        );
        for suffix in [":obj:hdr", ":obj:extret", ":obj:stream", ":sess", ":layers"] {
            assert!(!key.ends_with(suffix), "{key} must not look like {suffix}");
        }
        assert!(!key.contains(":layer:"));
    }

    /// A Valkey `SCAN MATCH` glob restricted to what this crate's scanners use:
    /// `*` matches any run of bytes, every other byte is literal (`{` and `}`
    /// are not glob metacharacters).
    fn scan_glob_matches(pattern: &str, key: &str) -> bool {
        match pattern.split_once('*') {
            None => pattern == key,
            Some((head, tail)) => {
                key.starts_with(head)
                    && (head.len()..=key.len())
                        .any(|at| key.is_char_boundary(at) && scan_glob_matches(tail, &key[at..]))
            }
        }
    }

    /// b.v6d T17: the `CreateLayer` operation record is
    /// `<prefix>{sid}:createop:<operation_id>`, carries the session's hash tag
    /// — so it shares a Cluster slot with the layer mapping and the layers
    /// index, the keys `create_layer_commit.lua` checks and writes in one
    /// script — and matches none of the `SCAN` patterns the sweepers and the
    /// checkpoint loop walk (the record is removed by TTL only; a sweeper that
    /// matched it would misread it as a layer mapping or an object).
    ///
    /// Ids are chosen to look like the patterns' own segments, which is the
    /// adversarial case for a glob.
    #[test]
    fn create_layer_operation_carries_the_session_tag_and_matches_no_scan_pattern() {
        let ks = KeySchema::new("test_x:");
        assert_eq!(
            ks.create_layer_operation("s1", "op-1"),
            "test_x:{s1}:createop:op-1"
        );

        // The patterns the scanners build: orphan sweep (`{*}:layer:*`,
        // `{*}:obj:hdr`, `{*}:obj:extret`), the checkpoint loop and the ext
        // reconciler (`{*}:obj:hdr`, `{*}:obj:extret`), and the ext-meta walk.
        let prefix = ks.prefix();
        let patterns = [
            format!("{prefix}{{*}}:layer:*"),
            format!("{prefix}{{*}}:obj:hdr"),
            format!("{prefix}{{*}}:obj:extret"),
            ks.ext_meta_scan_pattern(),
        ];
        // Control: each pattern matches the key family it exists for, so a
        // broken matcher cannot pass the negative assertions vacuously.
        for (pattern, key) in patterns.iter().zip([
            ks.layer_mapping("s1", "root"),
            ks.obj_header("42"),
            ks.obj_ext_retire_carrier("42"),
            ks.ext_meta("abcdef01"),
        ]) {
            assert!(
                scan_glob_matches(pattern, &key),
                "control: {pattern} must match {key}"
            );
        }

        for operation_id in [
            "op-1",
            "layer",
            "obj.hdr",
            "obj.extret",
            "x~layer_",
            "ext.meta",
        ] {
            let key = ks.create_layer_operation("s1", operation_id);
            assert_eq!(
                fred::util::redis_keyslot(key.as_bytes()),
                fred::util::redis_keyslot(b"s1"),
                "{key} must hash on the session tag"
            );
            for colocated in [ks.layer_mapping("s1", "root"), ks.layers_index("s1")] {
                assert_eq!(
                    fred::util::redis_keyslot(key.as_bytes()),
                    fred::util::redis_keyslot(colocated.as_bytes()),
                    "{key} must share a slot with {colocated}: one script writes both"
                );
            }
            for pattern in &patterns {
                assert!(
                    !scan_glob_matches(pattern, &key),
                    "{key} must match no scanner pattern, but {pattern} matches it"
                );
            }
        }
    }

    /// b.ua9: the `CreateLayer` commit claim is
    /// `<prefix>{sid}:createclaim:<object_id>`, carries the session's hash tag
    /// — so it shares a Cluster slot with the layer mapping and the deletion
    /// marker, the keys the commit script and the veto script read beside it
    /// in one script — and matches none of the `SCAN` patterns the sweepers
    /// and the checkpoint loop walk (it is removed by TTL only; the orphan
    /// sweep would misread a matched claim as a layer mapping or an object).
    #[test]
    fn create_layer_claim_carries_the_session_tag_and_matches_no_scan_pattern() {
        let ks = KeySchema::new("test_x:");
        assert_eq!(
            ks.create_layer_claim("s1", 42),
            "test_x:{s1}:createclaim:42"
        );

        let prefix = ks.prefix();
        let patterns = [
            format!("{prefix}{{*}}:layer:*"),
            format!("{prefix}{{*}}:obj:hdr"),
            format!("{prefix}{{*}}:obj:extret"),
            ks.ext_meta_scan_pattern(),
        ];
        // Control: each pattern matches the key family it exists for, so the
        // negative assertions below cannot pass on a broken matcher.
        for (pattern, key) in patterns.iter().zip([
            ks.layer_mapping("s1", "root"),
            ks.obj_header("42"),
            ks.obj_ext_retire_carrier("42"),
            ks.ext_meta("abcdef01"),
        ]) {
            assert!(
                scan_glob_matches(pattern, &key),
                "control: {pattern} must match {key}"
            );
        }

        // Session ids shaped like the patterns' own segments: the adversarial
        // case for a glob.
        for session_id in ["s1", "layer", "obj", "x:layer:y", "ext"] {
            let key = ks.create_layer_claim(session_id, 42);
            for colocated in [
                ks.layer_mapping(session_id, "root"),
                ks.session_deleted_marker(session_id),
                ks.layers_index(session_id),
            ] {
                assert_eq!(
                    fred::util::redis_keyslot(key.as_bytes()),
                    fred::util::redis_keyslot(colocated.as_bytes()),
                    "{key} must share a slot with {colocated}: one script reads both"
                );
            }
            for pattern in &patterns {
                assert!(
                    !scan_glob_matches(pattern, &key),
                    "{key} must match no scanner pattern, but {pattern} matches it"
                );
            }
        }
    }

    /// b.ua9: a `committed` claim lives at least ten minutes, and at least
    /// twice the sweep grace when the grace is longer, so it outlives every
    /// commit run of its object that could still try to map it.
    #[test]
    fn create_layer_claim_committed_ttl_is_the_larger_of_ten_minutes_and_twice_the_grace() {
        use std::time::Duration;
        assert_eq!(
            create_layer_claim_committed_ttl_ms(Duration::ZERO),
            600_000,
            "the floor"
        );
        assert_eq!(
            create_layer_claim_committed_ttl_ms(Duration::from_secs(60)),
            600_000,
            "the default grace's twice is under the floor"
        );
        assert_eq!(
            create_layer_claim_committed_ttl_ms(Duration::from_secs(400)),
            800_000,
            "twice a grace above five minutes"
        );
        assert_eq!(
            create_layer_claim_committed_ttl_ms(Duration::MAX),
            u64::MAX,
            "a grace past u64 milliseconds saturates rather than wrapping"
        );
    }

    /// b.yuo: the release-ledger keys, the settle carrier and the read hold
    /// carry the `{gc}` tag, so every GC script that touches them is
    /// single-slot with `pending` and `inflight`; the decrement replay record
    /// carries its hash's `{hash_hex}` tag, so it shares the slot with that
    /// hash's meta and data and the decrement can check and write it
    /// atomically. None of them matches a `SCAN` pattern a scanner walks:
    /// the replay record in particular must not look like an ext meta to the
    /// reference-count reconciler's walk.
    #[test]
    fn the_gc_ledger_keys_carry_their_slot_tags_and_match_no_scan_pattern() {
        let ks = KeySchema::new("test_x:");
        let hex = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        assert_eq!(ks.ext_pending_age(), "test_x:ext:{gc}:pending:age");
        assert_eq!(ks.ext_pending_bytes(), "test_x:ext:{gc}:pending:bytes");
        assert_eq!(ks.ext_pending_total(), "test_x:ext:{gc}:pending:total");
        assert_eq!(ks.ext_settled(), "test_x:ext:{gc}:settled");
        assert_eq!(ks.ext_read_hold(), "test_x:ext:{gc}:read_hold");
        assert_eq!(ks.ext_applied(hex), format!("test_x:ext:applied:{{{hex}}}"));

        let gc_slot = fred::util::redis_keyslot(ks.ext_pending().as_bytes());
        for key in [
            ks.ext_pending_age(),
            ks.ext_pending_bytes(),
            ks.ext_pending_total(),
            ks.ext_settled(),
            ks.ext_read_hold(),
        ] {
            assert_eq!(
                fred::util::redis_keyslot(key.as_bytes()),
                gc_slot,
                "{key} must share the {{gc}} slot with pending and inflight"
            );
        }
        let applied = ks.ext_applied(hex);
        for colocated in [ks.ext_meta(hex), ks.ext_data(hex)] {
            assert_eq!(
                fred::util::redis_keyslot(applied.as_bytes()),
                fred::util::redis_keyslot(colocated.as_bytes()),
                "{applied} must share a slot with {colocated}: one script reads and writes both"
            );
        }

        let prefix = ks.prefix();
        let patterns = [
            format!("{prefix}{{*}}:layer:*"),
            format!("{prefix}{{*}}:obj:hdr"),
            format!("{prefix}{{*}}:obj:extret"),
            ks.ext_meta_scan_pattern(),
        ];
        assert!(
            scan_glob_matches(&ks.ext_meta_scan_pattern(), &ks.ext_meta(hex)),
            "control: the ext-meta walk matches an ext meta"
        );
        for key in [
            ks.ext_pending_age(),
            ks.ext_pending_bytes(),
            ks.ext_pending_total(),
            ks.ext_settled(),
            ks.ext_read_hold(),
            applied,
        ] {
            for pattern in &patterns {
                assert!(
                    !scan_glob_matches(pattern, &key),
                    "{key} must match no scanner pattern, but {pattern} matches it"
                );
            }
        }
    }

    /// b.cyu: the migration sentinel is `<prefix>{<tag>}:migration-sentinel`,
    /// and in each of the four key sets that declare one — the commit's shared
    /// prefix, the reference-index recount's four keys, and the two GC ledger
    /// scripts' six `{gc}` keys — it hashes to the slot of every other key,
    /// `KEYS[1]` included: a sentinel on another slot would make its script
    /// `CROSSSLOT` on a cluster instead of refusing it only during a move. It
    /// is never written, so it is in no cleanup inventory, and it matches no
    /// `SCAN` pattern a scanner walks.
    #[test]
    fn the_migration_sentinel_carries_its_scripts_slot_tag_and_matches_no_scan_pattern() {
        let ks = KeySchema::new("test_x:");
        assert_eq!(
            ks.migration_sentinel("42"),
            "test_x:{42}:migration-sentinel"
        );
        assert_eq!(
            ks.migration_sentinel("gc"),
            "test_x:{gc}:migration-sentinel"
        );

        let oid = "42";
        let commit_prefix = crate::storage::lua::update_script_shared_keys(&ks, 42, 0);
        let key_sets: [(&str, Vec<String>, String); 4] = [
            (
                "update_layer_fast",
                commit_prefix,
                ks.migration_sentinel(oid),
            ),
            (
                "ext_index_recount",
                vec![
                    ks.obj_header(oid),
                    ks.obj_ext_index(oid),
                    ks.obj_fields_index(oid),
                    ks.obj_time_samples_index(oid),
                ],
                ks.migration_sentinel(oid),
            ),
            (
                "gc_drop_foreign_claims",
                vec![
                    ks.ext_inflight(),
                    ks.ext_pending(),
                    ks.ext_pending_age(),
                    ks.ext_pending_bytes(),
                    ks.ext_pending_total(),
                    ks.ext_settled(),
                ],
                ks.migration_sentinel("gc"),
            ),
            (
                "gc_ledger_audit",
                vec![
                    ks.ext_pending(),
                    ks.ext_inflight(),
                    ks.ext_pending_age(),
                    ks.ext_pending_bytes(),
                    ks.ext_pending_total(),
                    ks.ext_settled(),
                ],
                ks.migration_sentinel("gc"),
            ),
        ];
        for (script, keys, sentinel) in &key_sets {
            assert!(!keys.is_empty(), "{script}: the key set is not empty");
            let sentinel_slot = fred::util::redis_keyslot(sentinel.as_bytes());
            assert_eq!(
                sentinel_slot,
                fred::util::redis_keyslot(keys[0].as_bytes()),
                "{script}: {sentinel} must share KEYS[1]'s slot ({})",
                keys[0]
            );
            for key in keys {
                assert_eq!(
                    fred::util::redis_keyslot(key.as_bytes()),
                    sentinel_slot,
                    "{script}: {key} must share the sentinel's slot: one script declares both"
                );
            }
        }
        // The commit's prefix carries the sentinel itself (its last slot).
        assert_eq!(
            key_sets[0].1.last(),
            Some(&ks.migration_sentinel(oid)),
            "the commit's shared prefix ends with the object's migration sentinel"
        );

        let (secs, flds, tss, dels, chs) = cleanup_fixture();
        let inventory = ks.object_cleanup_keys(oid, &secs, &flds, &tss, &dels, &chs);
        assert!(
            !inventory.contains(&ks.migration_sentinel(oid)),
            "the sentinel is never written, so no cleanup removes it: {inventory:?}"
        );

        let prefix = ks.prefix();
        let patterns = [
            format!("{prefix}{{*}}:layer:*"),
            format!("{prefix}{{*}}:obj:hdr"),
            format!("{prefix}{{*}}:obj:extret"),
            ks.ext_meta_scan_pattern(),
        ];
        assert!(
            scan_glob_matches(&patterns[1], &ks.obj_header(oid)),
            "control: the header walk matches a header"
        );
        // Tags shaped like the patterns' own segments are the adversarial case.
        for tag in [oid, "gc", "layer", "obj", "x:obj:hdr"] {
            let sentinel = ks.migration_sentinel(tag);
            for pattern in &patterns {
                assert!(
                    !scan_glob_matches(pattern, &sentinel),
                    "{sentinel} must match no scanner pattern, but {pattern} matches it"
                );
            }
        }
    }

    #[test]
    fn ext_meta_returns_expected_key_with_hash_tag() {
        let ks = KeySchema::default();
        assert_eq!(ks.ext_meta("abcdef01"), "le:ext:meta:{abcdef01}");
    }

    #[test]
    fn ext_data_returns_expected_key_with_hash_tag() {
        let ks = KeySchema::default();
        assert_eq!(ks.ext_data("abcdef01"), "le:ext:data:{abcdef01}");
    }

    #[test]
    fn ext_meta_and_ext_data_share_hash_tag_for_same_hash() {
        // Co-location requirement (SDD §11.1 / §11.3): meta and data for the
        // same content hash must live on the same Valkey Cluster slot so a
        // single-slot Lua script can atomically create or delete them as a
        // pair (`ext_store_value.lua` / `gc_decrement_and_collect.lua`).
        let ks = KeySchema::default();
        let hash_hex = "abcdef0123456789";
        let tag = format!("{{{hash_hex}}}");
        assert!(ks.ext_meta(hash_hex).contains(&tag));
        assert!(ks.ext_data(hash_hex).contains(&tag));
    }

    /// The reconciler's meta SCAN pattern puts its glob INSIDE the hash-tag
    /// braces, and therefore matches what [`KeySchema::ext_meta`] builds
    /// (b.yda).
    ///
    /// The braces are literal characters in the key — they are the Cluster hash
    /// tag — so the intuitive `le:ext:meta:*` would match every meta key by
    /// accident of the trailing `*`, while `le:ext:meta:{*}` is the shape that
    /// stays correct if the builder ever gains a suffix. Getting this wrong
    /// produces a SCAN that silently matches NOTHING: the reconciler enumerates
    /// zero metas, computes zero surplus, publishes a confident zero on every
    /// gauge, and reports a healthy store forever.
    #[test]
    fn the_ext_meta_scan_pattern_matches_the_keys_ext_meta_builds() {
        let ks = KeySchema::default();
        let pattern = ks.ext_meta_scan_pattern();
        assert_eq!(pattern, "le:ext:meta:{*}");

        // Hand-checked rather than glob-matched: the assertion is that the
        // literal text either side of the glob is what the builder emits.
        let key = ks.ext_meta("abcdef01");
        let (head, tail) = pattern
            .split_once('*')
            .expect("the pattern must carry exactly one glob");
        assert!(
            key.starts_with(head) && key.ends_with(tail),
            "a real key ({key}) must sit inside the pattern's literal head ({head:?}) and tail \
             ({tail:?}), or the reconciler's meta walk enumerates nothing and reports a healthy \
             store forever"
        );
    }

    /// `ext_meta_hash_hex` is the strict inverse of `ext_meta`, and rejects
    /// anything else (b.yda).
    ///
    /// It is how the reconciler turns a `SCAN` hit back into the hash it is
    /// about, and a lenient parse would let an unrelated key contribute a
    /// mis-derived hash to the surplus computation — which is what drives a
    /// decrement. So the negative cases are the point: a key from another
    /// family, a truncated one, and one carrying a different prefix must all
    /// answer `None` rather than a plausible-looking substring.
    #[test]
    fn ext_meta_hash_hex_inverts_ext_meta_and_rejects_everything_else() {
        let ks = KeySchema::default();
        let hash_hex = "abcdef0123456789";

        assert_eq!(
            ks.ext_meta_hash_hex(&ks.ext_meta(hash_hex)),
            Some(hash_hex),
            "the round trip is the contract: whatever `ext_meta` builds, this must take apart"
        );

        for (key, why) in [
            (
                ks.ext_data(hash_hex),
                "a DATA key shares the tag and the prefix and differs only in one segment",
            ),
            (
                format!("le:ext:meta:{{{hash_hex}}}:extra"),
                "a key with a suffix past the closing brace is not an ext:meta key",
            ),
            (
                format!("le:ext:meta:{hash_hex}"),
                "and neither is one missing the literal hash-tag braces",
            ),
            (
                format!("other:ext:meta:{{{hash_hex}}}"),
                "a foreign prefix must not be stripped — two deployments can share a Valkey",
            ),
        ] {
            assert_eq!(
                ks.ext_meta_hash_hex(&key),
                None,
                "{why}, so `{key}` must answer None. A lenient parse feeds a mis-derived hash \
                 into the surplus computation, and a surplus is what drives a decrement"
            );
        }
    }

    /// The two b.yda `{gc}`-slot ZSETs sit on the GC protocol's slot, beside
    /// pending and inflight.
    ///
    /// `ext_recon_candidates` is read in the same pass that reads pending and
    /// inflight (the `owed` subtraction) and feeds repairs straight back onto
    /// pending, so it belongs to the GC protocol's slot rather than the
    /// checkpoint reconciliation's `{recon}` one. `ext_retire_retry` is the
    /// index into the per-object carriers and is written by the retirement path,
    /// whose `{gc}`-slot `SADD` is exactly the write that just failed — which is
    /// why the carrier itself deliberately lives elsewhere; see
    /// [`the_retire_carrier_is_not_in_the_object_cleanup_inventory`].
    #[test]
    fn the_reconciler_and_retire_retry_zsets_carry_the_gc_hash_tag() {
        let ks = KeySchema::default();
        assert_eq!(ks.ext_recon_candidates(), "le:ext:{gc}:recon");
        assert_eq!(ks.ext_retire_retry(), "le:ext:{gc}:retire-retry");
        for key in [ks.ext_recon_candidates(), ks.ext_retire_retry()] {
            assert!(
                key.contains("{gc}"),
                "{key} must share the `{{gc}}` slot with pending and inflight — the reconciler \
                 reads all three in one pass and writes its repairs onto pending"
            );
        }
    }

    /// The retirement retry carrier is `{object_id}`-tagged and is deliberately
    /// NOT in the object's cleanup inventory (b.yda).
    ///
    /// Both halves are the design rather than details of it.
    ///
    /// The SLOT: what failed when a deferral happens is a `{gc}`-slot `SADD`,
    /// and the natural Cluster shape of that is "this object's slot answered
    /// every read, the `{gc}` slot did not" — so a second `{gc}`-slot write
    /// would recover nothing. The carrier lands on the slot whose reads just
    /// succeeded.
    ///
    /// The EXCLUSION: the carrier's whole purpose is to outlive the teardown
    /// that follows the deferral. Listed here it would be UNLINKed in the same
    /// chunk as `obj:hdr`, which is the pre-b.yda permanent leak wearing a new
    /// key name — and it carries no TTL for the same reason. The sibling
    /// assertion in `object_cleanup_keys_emits_every_obj_family_including_root_children`
    /// pins the inventory as an exact set and so would also catch an addition;
    /// this one names the key, so a failure says WHICH key and why.
    #[test]
    fn the_retire_carrier_is_not_in_the_object_cleanup_inventory() {
        let ks = KeySchema::default();
        let carrier = ks.obj_ext_retire_carrier("42");
        assert_eq!(carrier, "le:{42}:obj:extret");

        let (secs, flds, tss, dels, chs) = cleanup_fixture();
        let inventory = ks.object_cleanup_keys("42", &secs, &flds, &tss, &dels, &chs);
        assert!(
            !inventory.contains(&carrier),
            "the carrier must NOT be reaped with the object. It holds the release members a \
             failed retirement deferred, and the orphan sweep's retire-retry phase is what \
             settles them — UNLINKing it here discards the obligation and restores the exact \
             permanent leak b.yda exists to close. Inventory: {inventory:?}"
        );

        assert!(
            ks.obj_header("42").contains("{42}")
                && ks.obj_ext_index("42").contains("{42}")
                && carrier.contains("{42}"),
            "and it must share the object's slot with the two keys the settle script gates on: \
             the settle reads carrier + tokens + header + index in ONE atomic step, which is only \
             available because all of them carry the same hash tag"
        );
    }

    /// The retirement intent's token HASH (b.yuo Part R) sits on the object's
    /// slot beside the carrier, is NOT in the cleanup inventory, and matches no
    /// `SCAN` pattern a scanner walks — in particular not the orphan sweep's
    /// carrier backstop `{*}:obj:extret`, whose glob ends at `extret`.
    ///
    /// The slot is what lets `ext_retire_begin.lua`, the re-token, the settle
    /// and the clear read and write the carrier, the tokens and the header in
    /// one atomic step. The exclusion is the carrier's: the tokens must outlive
    /// the teardown so the drain can ask `{gc}` whether each attempt ran before
    /// it sends anything again. And a backstop that matched the HASH would
    /// parse `extret:attempts` into an object id and drain a carrier that is not
    /// there.
    #[test]
    fn the_retire_attempt_tokens_share_the_carriers_slot_and_match_no_scan_pattern() {
        let ks = KeySchema::new("test_x:");
        let tokens = ks.obj_ext_retire_attempts("42");
        assert_eq!(tokens, "test_x:{42}:obj:extret:attempts");
        for colocated in [
            ks.obj_ext_retire_carrier("42"),
            ks.obj_header("42"),
            ks.obj_ext_index("42"),
        ] {
            assert_eq!(
                fred::util::redis_keyslot(tokens.as_bytes()),
                fred::util::redis_keyslot(colocated.as_bytes()),
                "{tokens} must share a slot with {colocated}: one script reads and writes both"
            );
        }

        let (secs, flds, tss, dels, chs) = cleanup_fixture();
        let inventory = ks.object_cleanup_keys("42", &secs, &flds, &tss, &dels, &chs);
        assert!(
            !inventory.contains(&tokens),
            "the token HASH must NOT be reaped with the object: a deferred carrier's tokens are \
             what the drain resolves before it re-drives anything. Inventory: {inventory:?}"
        );

        let prefix = ks.prefix();
        let patterns = [
            format!("{prefix}{{*}}:layer:*"),
            format!("{prefix}{{*}}:obj:hdr"),
            format!("{prefix}{{*}}:obj:extret"),
            ks.ext_meta_scan_pattern(),
        ];
        assert!(
            scan_glob_matches(&patterns[2], &ks.obj_ext_retire_carrier("42")),
            "control: the backstop pattern matches the carrier itself"
        );
        for pattern in &patterns {
            assert!(
                !scan_glob_matches(pattern, &tokens),
                "{tokens} must match no scanner pattern, but {pattern} matches it"
            );
        }
    }

    /// The `{gc}` enqueue-attempt record (b.yuo Part R) is
    /// `<prefix>ext:{gc}:enqueue_attempt`, on the `{gc}` slot with `pending`, so
    /// the attempt-mode enqueue can check and record a token in the same
    /// script that adds its members, and the resolve's `HSETNX` lands on the
    /// same slot — which is what decides each attempt once. It matches no
    /// `SCAN` pattern a scanner walks.
    #[test]
    fn the_enqueue_attempt_record_carries_the_gc_tag_and_matches_no_scan_pattern() {
        let ks = KeySchema::new("test_x:");
        let record = ks.ext_enqueue_attempts();
        assert_eq!(record, "test_x:ext:{gc}:enqueue_attempt");
        assert_eq!(
            fred::util::redis_keyslot(record.as_bytes()),
            fred::util::redis_keyslot(ks.ext_pending().as_bytes()),
            "{record} must share the {{gc}} slot with pending: one script writes both"
        );
        let prefix = ks.prefix();
        for pattern in [
            format!("{prefix}{{*}}:layer:*"),
            format!("{prefix}{{*}}:obj:hdr"),
            format!("{prefix}{{*}}:obj:extret"),
            ks.ext_meta_scan_pattern(),
        ] {
            assert!(
                !scan_glob_matches(&pattern, &record),
                "{record} must match no scanner pattern, but {pattern} matches it"
            );
        }
    }

    #[test]
    fn ext_pending_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.ext_pending(), "le:ext:{gc}:pending");
    }

    #[test]
    fn ext_inflight_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.ext_inflight(), "le:ext:{gc}:inflight");
    }

    #[test]
    fn ext_pending_and_ext_inflight_share_hash_tag_for_cluster_colocation() {
        // Co-location requirement (b.i8x follow-up): pending and inflight must
        // share a Valkey Cluster slot so `SMOVE pending → inflight` in
        // `process_pending` does not fail with CROSSSLOT in cluster mode. The
        // `{gc}` tag is global across all hashes and deliberately distinct
        // from the per-hash `{hash_hex}` tag used for ext_meta/ext_data.
        let ks = KeySchema::default();
        let tag = "{gc}";
        assert!(
            ks.ext_pending().contains(tag),
            "ext_pending missing {{gc}} hash tag: {}",
            ks.ext_pending()
        );
        assert!(
            ks.ext_inflight().contains(tag),
            "ext_inflight missing {{gc}} hash tag: {}",
            ks.ext_inflight()
        );
    }

    #[test]
    fn counter_object_id_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.counter_object_id(), "le:counter:object_id");
    }

    #[test]
    fn presence_members_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.presence_members("s1"), "le:{s1}:presence");
    }

    #[test]
    fn presence_user_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.presence_user("s1", "conn5"), "le:{s1}:presence:conn5");
    }

    #[test]
    fn presence_lease_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(
            ks.presence_lease("s1", "conn5"),
            "le:{s1}:presence:conn5:lease"
        );
    }

    #[test]
    fn presence_gone_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(
            ks.presence_gone("s1", "conn5"),
            "le:{s1}:presence:conn5:gone"
        );
    }

    // ── Checkpoint reconciliation ZSET tests (b.9qd) ───────────────────
    //
    // Two keys, one shared hash tag. The exact strings are a cross-replica
    // durable contract: the ZSETs outlive every process that touches them, so
    // a prefix drift would not fail anything — it would silently strand the
    // old key's members (orphan candidates never acted on, tombstoned hashes
    // never collected) while the new key started from empty.

    #[test]
    fn checkpoint_reconcile_keys_default_prefix_return_expected_keys() {
        let ks = KeySchema::default();
        assert_eq!(ks.checkpoint_reconcile_orphans(), "le:ckpt:{recon}:orphans");
        assert_eq!(ks.checkpoint_reconcile_tombstones(), "le:ckpt:{recon}:tomb");
    }

    #[test]
    fn checkpoint_reconcile_keys_custom_prefix_propagates() {
        let ks = KeySchema::new("test_abc:");
        assert_eq!(
            ks.checkpoint_reconcile_orphans(),
            "test_abc:ckpt:{recon}:orphans"
        );
        assert_eq!(
            ks.checkpoint_reconcile_tombstones(),
            "test_abc:ckpt:{recon}:tomb"
        );
    }

    /// Both keys carry the SAME `{recon}` hash tag, so they land on one
    /// Cluster slot.
    ///
    /// The reconciliation pass touches both inside one logical operation
    /// (reap an orphan → tombstone its hashes), and a future script that spans
    /// them must stay slot-safe. Same reasoning as the `{gc}` tag on
    /// `ext_pending` / `ext_inflight`, whose claim-then-collect protocol this
    /// one mirrors.
    #[test]
    fn checkpoint_reconcile_keys_share_one_hash_tag_and_slot() {
        let ks = KeySchema::default();
        let orphans = ks.checkpoint_reconcile_orphans();
        let tomb = ks.checkpoint_reconcile_tombstones();
        assert_ne!(
            orphans, tomb,
            "the two ZSETs must be distinct keys: orphan candidates and ext-blob tombstones \
             are different member vocabularies with different remedies"
        );
        for key in [&orphans, &tomb] {
            assert!(
                key.contains("{recon}"),
                "`{key}` must carry the shared {{recon}} hash tag so the pair is slot-safe \
                 under Valkey Cluster"
            );
        }
        assert_eq!(
            hash_tag_of(&orphans),
            hash_tag_of(&tomb),
            "the two keys must resolve to the SAME hash tag, not merely both have one"
        );
    }

    /// The reconciliation ZSETs must not collide with the GC protocol's own
    /// claim-then-collect keys. Both are ext-blob reclamation, both use a
    /// pending/inflight-shaped ZSET, and a collision would let one sweep
    /// consume the other's members — each of which names a delete.
    #[test]
    fn checkpoint_reconcile_keys_are_distinct_from_the_gc_protocol_keys() {
        let ks = KeySchema::default();
        for recon in [
            ks.checkpoint_reconcile_orphans(),
            ks.checkpoint_reconcile_tombstones(),
        ] {
            for gc in [ks.ext_pending(), ks.ext_inflight(), ks.gc_leader_lock()] {
                assert_ne!(
                    recon, gc,
                    "reconciliation key `{recon}` must not collide with GC key `{gc}`"
                );
            }
        }
    }

    /// Extract the `{...}` hash tag from a key, or the whole key when it has
    /// none. Mirrors the Cluster slot rule closely enough for the assertion
    /// above (first `{`, first `}` after it, non-empty).
    fn hash_tag_of(key: &str) -> &str {
        let Some(open) = key.find('{') else {
            return key;
        };
        let rest = &key[open + 1..];
        let Some(close) = rest.find('}') else {
            return key;
        };
        if close == 0 {
            return key;
        }
        &rest[..close]
    }
    // ── Checkpoint leader-lock tests (SDD SR-2.1) ──────────────────────
    //
    // The leader-lock key string is the Cross-Task contract that the
    // periodic checkpoint loop's `try_acquire_leader_lock` consumes. A
    // typo, a missing prefix propagation, or an inadvertent hash-tag would
    // silently land leader election on the wrong key. These tests pin the
    // exact-string output, custom-prefix propagation, lack-of-hash-tag, and
    // pairwise distinctness from the sibling `gc_leader_lock()` and
    // `presence_sweeper_leader_lock()` accessors so a refactor can't
    // silently break the contract.

    #[test]
    fn checkpoint_leader_lock_default_prefix_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.checkpoint_leader_lock(), "le:checkpoint:leader");
    }

    #[test]
    fn checkpoint_leader_lock_custom_prefix_returns_expected_key() {
        let ks = KeySchema::new("test_abc:");
        assert_eq!(ks.checkpoint_leader_lock(), "test_abc:checkpoint:leader");
    }

    #[test]
    fn checkpoint_leader_lock_has_no_hash_tag() {
        // Deliberately NOT hash-tagged (same rationale as `gc_leader_lock`):
        // the lock is read/written by exactly one command at a time and
        // shares no transaction with other keys, so a hash-tag would just
        // concentrate operator-visible activity onto one slot without an
        // atomicity benefit.
        let ks = KeySchema::default();
        let key = ks.checkpoint_leader_lock();
        assert!(
            !key.contains('{'),
            "checkpoint_leader_lock must not contain a hash tag: {key}"
        );
        assert!(
            !key.contains('}'),
            "checkpoint_leader_lock must not contain a hash tag: {key}"
        );
    }

    #[test]
    fn checkpoint_leader_lock_distinct_from_gc_and_presence_sweeper_locks() {
        // The three maintenance loops (GC + orphan sweep, presence sweep,
        // checkpoint) are deliberately gated by three independent leader
        // locks so they can run on different replicas in steady state. A
        // typo that collapses any two of these onto the same key would
        // defeat that property and silently serialize unrelated loops onto
        // one replica.
        let ks = KeySchema::default();
        let checkpoint = ks.checkpoint_leader_lock();
        let gc = ks.gc_leader_lock();
        let sweeper = ks.presence_sweeper_leader_lock();
        assert_ne!(
            checkpoint, gc,
            "checkpoint_leader_lock must differ from gc_leader_lock (checkpoint={checkpoint}, gc={gc})"
        );
        assert_ne!(
            checkpoint, sweeper,
            "checkpoint_leader_lock must differ from presence_sweeper_leader_lock \
             (checkpoint={checkpoint}, sweeper={sweeper})"
        );
        assert_ne!(
            gc, sweeper,
            "gc_leader_lock must differ from presence_sweeper_leader_lock \
             (gc={gc}, sweeper={sweeper})"
        );
    }

    #[test]
    fn checkpoint_leader_lock_is_stable_across_calls() {
        // Two successive calls on the same `KeySchema` must return the
        // exact same string — guards against an accidental future change
        // that mixes wall-clock time, randomness, or replica identity into
        // the key (the replica identity is the SET *value*, not the key).
        let ks = KeySchema::default();
        let first = ks.checkpoint_leader_lock();
        let second = ks.checkpoint_leader_lock();
        assert_eq!(first, second);
    }

    #[test]
    fn field_member_hex_encodes_key_name() {
        // "color" → hex "636f6c6f72"
        assert_eq!(KeySchema::field_member("sec7", "color"), "sec7:636f6c6f72");
    }

    #[test]
    fn time_sample_member_hex_encodes_time() {
        // 1.0_f64.to_bits() = 0x3FF0000000000000
        assert_eq!(
            KeySchema::time_sample_member("sec7", 1.0),
            "sec7:3ff0000000000000"
        );
    }

    #[test]
    fn presence_events_channel_returns_expected_key() {
        let ks = KeySchema::default();
        assert_eq!(ks.presence_events_channel("s1"), "le:presence:events:s1");
    }

    // ── B. Hex encoding tests ──────────────────────────────────────────

    #[test]
    fn hex_encode_ascii_string_returns_correct_hex() {
        assert_eq!(hex_encode("hello"), "68656c6c6f");
    }

    #[test]
    fn hex_encode_non_ascii_utf8_returns_correct_hex() {
        // "日本" in UTF-8 is [0xe6, 0x97, 0xa5, 0xe6, 0x9c, 0xac]
        assert_eq!(hex_encode("日本"), "e697a5e69cac");
    }

    #[test]
    fn hex_encode_empty_string_returns_empty() {
        assert_eq!(hex_encode(""), "");
    }

    #[test]
    fn hex_encode_special_characters_colon_slash_space() {
        // ":/a b" → [0x3a, 0x2f, 0x61, 0x20, 0x62]
        assert_eq!(hex_encode(":/a b"), "3a2f612062");
    }

    #[test]
    fn hex_encode_colon_prevents_delimiter_conflict() {
        // A layer name containing ":" must not produce a literal ":" in the key
        let encoded = hex_encode("layer:with:colons");
        assert!(
            !encoded.contains(':'),
            "hex-encoded string must not contain literal colons: {encoded}"
        );
    }

    // ── C. f64 hex encoding edge cases ─────────────────────────────────

    #[test]
    fn hex_encode_f64_zero_returns_16_zeros() {
        assert_eq!(hex_encode_f64(0.0), "0000000000000000");
    }

    #[test]
    fn hex_encode_f64_negative_zero_differs_from_positive_zero() {
        let pos = hex_encode_f64(0.0);
        let neg = hex_encode_f64(-0.0);
        assert_ne!(
            pos, neg,
            "0.0 and -0.0 have different IEEE 754 bit patterns"
        );
        // -0.0 bit pattern: sign bit set, rest zero → 0x8000000000000000
        assert_eq!(neg, "8000000000000000");
    }

    #[test]
    fn hex_encode_f64_one_returns_expected_bits() {
        // 1.0_f64.to_bits() == 0x3FF0000000000000
        assert_eq!(hex_encode_f64(1.0), "3ff0000000000000");
    }

    #[test]
    fn hex_encode_f64_nan_returns_expected_bits() {
        let result = hex_encode_f64(f64::NAN);
        // Standard NaN is 0x7FF8000000000000
        assert_eq!(result, "7ff8000000000000");
    }

    #[test]
    fn hex_encode_f64_positive_infinity_returns_expected_bits() {
        // +inf → 0x7FF0000000000000
        assert_eq!(hex_encode_f64(f64::INFINITY), "7ff0000000000000");
    }

    #[test]
    fn hex_encode_f64_negative_infinity_returns_expected_bits() {
        // -inf → 0xFFF0000000000000
        assert_eq!(hex_encode_f64(f64::NEG_INFINITY), "fff0000000000000");
    }

    #[test]
    // 3.14 is an intentional literal testing a specific bit pattern, not an approximation of PI.
    #[allow(clippy::approx_constant)]
    fn hex_encode_f64_normal_value_3_14() {
        // 3.14_f64.to_bits() verified via Rust
        let expected = format!("{:016x}", 3.14_f64.to_bits());
        assert_eq!(hex_encode_f64(3.14), expected);
    }

    #[test]
    // 3.14 is an intentional literal testing a specific bit pattern, not an approximation of PI.
    #[allow(clippy::approx_constant)]
    fn hex_encode_f64_always_produces_16_chars() {
        for v in [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            3.14,
        ] {
            let result = hex_encode_f64(v);
            assert_eq!(
                result.len(),
                16,
                "hex_encode_f64({v}) produced {}-char string: {result}",
                result.len()
            );
        }
    }

    // ── D. Configurable prefix propagation ─────────────────────────────

    #[test]
    fn custom_prefix_propagates_to_all_key_types() {
        let ks = KeySchema::new("test_abc:");

        // Session-scoped
        assert!(
            ks.session_meta("s1").starts_with("test_abc:"),
            "session_meta"
        );
        assert!(ks.path_index("/p").starts_with("test_abc:"), "path_index");
        assert!(
            ks.layer_mapping("s1", "l").starts_with("test_abc:"),
            "layer_mapping"
        );
        assert!(
            ks.layers_index("s1").starts_with("test_abc:"),
            "layers_index"
        );

        // Object-scoped
        assert!(ks.obj_header("1").starts_with("test_abc:"), "obj_header");
        assert!(
            ks.obj_section("1", "s").starts_with("test_abc:"),
            "obj_section"
        );
        assert!(
            ks.obj_sections_index("1").starts_with("test_abc:"),
            "obj_sections_index"
        );
        assert!(
            ks.obj_children("1", "p").starts_with("test_abc:"),
            "obj_children"
        );
        assert!(
            ks.obj_field("1", "s", "k").starts_with("test_abc:"),
            "obj_field"
        );
        assert!(
            ks.obj_fields_index("1").starts_with("test_abc:"),
            "obj_fields_index"
        );
        assert!(
            ks.obj_time_sample("1", "s", 0.0).starts_with("test_abc:"),
            "obj_time_sample"
        );
        assert!(
            ks.obj_time_samples_index("1").starts_with("test_abc:"),
            "obj_time_samples_index"
        );
        assert!(
            ks.obj_deleted_section("1", "s").starts_with("test_abc:"),
            "obj_deleted_section"
        );
        assert!(
            ks.obj_deleted_sections_index("1").starts_with("test_abc:"),
            "obj_deleted_sections_index"
        );

        // External value keys
        assert!(ks.ext_meta("h").starts_with("test_abc:"), "ext_meta");
        assert!(ks.ext_data("h").starts_with("test_abc:"), "ext_data");
        assert!(ks.ext_pending().starts_with("test_abc:"), "ext_pending");
        assert!(ks.ext_inflight().starts_with("test_abc:"), "ext_inflight");

        // Counters
        assert!(
            ks.counter_object_id().starts_with("test_abc:"),
            "counter_object_id"
        );

        // Presence
        assert!(
            ks.presence_members("s1").starts_with("test_abc:"),
            "presence_members"
        );
        assert!(
            ks.presence_user("s1", "c").starts_with("test_abc:"),
            "presence_user"
        );
        assert!(
            ks.presence_lease("s1", "c").starts_with("test_abc:"),
            "presence_lease"
        );
        assert!(
            ks.presence_gone("s1", "c").starts_with("test_abc:"),
            "presence_gone"
        );

        // Pub/Sub
        assert!(
            ks.presence_events_channel("s1").starts_with("test_abc:"),
            "presence_events_channel"
        );

        // Maintenance leader locks
        assert!(
            ks.checkpoint_leader_lock().starts_with("test_abc:"),
            "checkpoint_leader_lock"
        );
    }

    #[test]
    fn session_scoped_keys_contain_session_hash_tag() {
        let ks = KeySchema::default();
        let sid = "my-session";
        let tag = format!("{{{sid}}}");

        assert!(ks.session_meta(sid).contains(&tag), "session_meta");
        assert!(ks.layer_mapping(sid, "l").contains(&tag), "layer_mapping");
        assert!(ks.layers_index(sid).contains(&tag), "layers_index");
        assert!(ks.presence_members(sid).contains(&tag), "presence_members");
        assert!(ks.presence_user(sid, "c").contains(&tag), "presence_user");
        assert!(ks.presence_lease(sid, "c").contains(&tag), "presence_lease");
        assert!(ks.presence_gone(sid, "c").contains(&tag), "presence_gone");
        assert!(
            ks.session_deleted_marker(sid).contains(&tag),
            "session_deleted_marker"
        );
    }

    /// The deletion marker is written in `SessionStore::delete`'s MULTI/EXEC
    /// with the session-meta DEL, so it must name the very same hash tag — a
    /// different tag would put it on another Cluster slot and the transaction
    /// would fail `CROSSSLOT`.
    #[test]
    fn session_deleted_marker_returns_expected_key_on_the_session_meta_slot() {
        let ks = KeySchema::default();
        assert_eq!(ks.session_deleted_marker("s1"), "le:{s1}:sess:deleted");
        assert_eq!(
            KeySchema::new("test_abc:").session_deleted_marker("s1"),
            "test_abc:{s1}:sess:deleted"
        );
        assert_eq!(
            hash_tag_of(&ks.session_deleted_marker("s1")),
            hash_tag_of(&ks.session_meta("s1")),
            "the marker and the metadata it outlives must share one hash tag"
        );
        assert_ne!(
            ks.session_deleted_marker("s1"),
            ks.session_meta("s1"),
            "the marker is its own key, never the metadata hash"
        );
    }

    /// The marker's TTL: seven days, and in any case at least the 24 h floor
    /// the design set, far above the checkpoint writer's worst-case delete
    /// latency (about 76 minutes at stock settings; see the constant's docs).
    /// Session ids are never reused, so a generous TTL only costs a few bytes
    /// per deleted session.
    #[test]
    fn session_deleted_marker_ttl_is_seven_days() {
        assert_eq!(SESSION_DELETED_MARKER_TTL_SECS, 7 * 24 * 60 * 60);
        const _: () = assert!(SESSION_DELETED_MARKER_TTL_SECS >= 24 * 60 * 60);
    }

    #[test]
    fn object_scoped_keys_contain_object_hash_tag() {
        let ks = KeySchema::default();
        let oid = "42";
        let tag = format!("{{{oid}}}");

        assert!(ks.obj_header(oid).contains(&tag), "obj_header");
        assert!(ks.obj_section(oid, "s").contains(&tag), "obj_section");
        assert!(
            ks.obj_sections_index(oid).contains(&tag),
            "obj_sections_index"
        );
        assert!(ks.obj_children(oid, "p").contains(&tag), "obj_children");
        assert!(ks.obj_field(oid, "s", "k").contains(&tag), "obj_field");
        assert!(ks.obj_fields_index(oid).contains(&tag), "obj_fields_index");
        assert!(
            ks.obj_time_sample(oid, "s", 0.0).contains(&tag),
            "obj_time_sample"
        );
        assert!(
            ks.obj_time_samples_index(oid).contains(&tag),
            "obj_time_samples_index"
        );
        assert!(
            ks.obj_deleted_section(oid, "s").contains(&tag),
            "obj_deleted_section"
        );
        assert!(
            ks.obj_deleted_sections_index(oid).contains(&tag),
            "obj_deleted_sections_index"
        );
    }

    // ── E. sha256_to_hex tests ─────────────────────────────────────────

    #[test]
    fn sha256_to_hex_all_zeros_returns_64_zero_chars() {
        let hash = [0u8; 32];
        let hex = sha256_to_hex(&hash);
        assert_eq!(hex.len(), 64);
        assert_eq!(hex, "0".repeat(64));
    }

    #[test]
    fn sha256_to_hex_all_ff_returns_64_f_chars() {
        let hash = [0xffu8; 32];
        let hex = sha256_to_hex(&hash);
        assert_eq!(hex.len(), 64);
        assert_eq!(hex, "f".repeat(64));
    }

    #[test]
    fn sha256_to_hex_known_value() {
        // SHA-256 of empty string = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash: [u8; 32] = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(
            sha256_to_hex(&hash),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_to_hex_is_lowercase() {
        let hash = [0xAB; 32];
        let hex = sha256_to_hex(&hash);
        assert_eq!(hex, hex.to_lowercase(), "hex output must be lowercase");
    }

    #[test]
    fn sha256_to_hex_always_64_chars() {
        // Leading zero byte must not be dropped
        let mut hash = [0u8; 32];
        hash[0] = 0x01;
        let hex = sha256_to_hex(&hash);
        assert_eq!(hex.len(), 64);
        assert!(hex.starts_with("01"));
    }

    // ── F. object_cleanup_keys — retirement inventory (b.9fn) ──────────
    //
    // These pin the *complete* `obj:*` family a retired object owns.
    // `object_cleanup_keys` is the inventory authority: the restore residue
    // wipe UNLINKs (a filtered copy of) it, and the retirement routes' paged
    // teardown (`storage::retirement::Teardown`) UNLINKs the same keys page by
    // page, built from the same pieces (`object_teardown_first_keys`, the
    // per-member keys, `object_teardown_index_keys`; section G pins that the
    // pieces add up to this inventory). So a family that exists on the write
    // surface but is missing from the expected set below leaks one key per
    // retired object id, forever — which is exactly what b.9fn was for
    // `obj:ch:0`.
    //
    // The set-equality assertion is deliberately exact rather than a
    // "contains" check: adding a key family to the writer without adding it
    // here cannot pass silently, and removing one from the helper fails at
    // unit scope instead of at the next storage audit.

    /// The five index enumerations `object_cleanup_keys` takes, in argument
    /// order: section ids, field members, time-sample members, tombstone ids,
    /// children-list parent ids.
    type CleanupFixture = (
        Vec<String>,
        Vec<String>,
        Vec<String>,
        Vec<String>,
        Vec<String>,
    );

    /// Enumerations shaped like what the two callers read out of Valkey for a
    /// populated object: one live section, one field, one time sample, one
    /// tombstoned section, and a children-list parent-id index (b.jdr) holding
    /// the root, the live section, and a *dangling* parent — the id that is in
    /// neither `obj:secs` nor `obj:dels` and that only the index can name.
    fn cleanup_fixture() -> CleanupFixture {
        let section_ids = vec!["10".to_string()];
        let field_members = vec![KeySchema::field_member("10", "color")];
        let ts_members = vec![KeySchema::time_sample_member("10", 1.0)];
        let tombstone_ids = vec!["20".to_string()];
        let children_parent_ids = vec!["0".to_string(), "10".to_string(), "99".to_string()];
        (
            section_ids,
            field_members,
            ts_members,
            tombstone_ids,
            children_parent_ids,
        )
    }

    #[test]
    fn object_cleanup_keys_emits_every_obj_family_including_root_children() {
        let ks = KeySchema::default();
        let (secs, flds, tss, dels, chs) = cleanup_fixture();

        let got = ks.object_cleanup_keys("42", &secs, &flds, &tss, &dels, &chs);

        // Hand-written literals, not re-derived from the builders: a rename of
        // a key pattern must show up here as a failing test, not be masked by
        // both sides calling the same (renamed) function.
        let expected = vec![
            "le:{42}:obj:hdr",
            "le:{42}:obj:secs",
            "le:{42}:obj:flds",
            "le:{42}:obj:tss",
            "le:{42}:obj:dels",
            // The b.jdr index itself.
            "le:{42}:obj:chs",
            // The b.qpf per-object external-value reference index.
            "le:{42}:obj:ext",
            // The per-object UpdateLayer commit-token record.
            "le:{42}:obj:wtok",
            // The per-object UpdateLayer commit release ring.
            "le:{42}:obj:wrel",
            // The snapshot fence and the writer-intent key (Spec SDD SR-5.1).
            "le:{42}:obj:snapfence",
            "le:{42}:obj:snapintent",
            // The b.9fn key: root parent id 0 is in neither obj:secs nor
            // obj:dels, so the reconstruction must name it explicitly (it is
            // also an `obj:chs` member since b.jdr; the union dedupes).
            "le:{42}:obj:ch:0",
            // The b.jdr key: a dangling parent, in neither obj:secs nor
            // obj:dels, reachable only through the recorded index.
            "le:{42}:obj:ch:99",
            "le:{42}:obj:sec:10",
            "le:{42}:obj:ch:10",
            // "color" → hex 636f6c6f72
            "le:{42}:obj:fld:10:636f6c6f72",
            // 1.0f64 → big-endian bits hex 3ff0000000000000
            "le:{42}:obj:ts:10:3ff0000000000000",
            "le:{42}:obj:del:20",
            "le:{42}:obj:ch:20",
        ];

        let mut got_sorted = got.clone();
        got_sorted.sort();
        let mut expected_sorted: Vec<String> = expected.iter().map(|s| (*s).to_string()).collect();
        expected_sorted.sort();
        assert_eq!(
            got_sorted, expected_sorted,
            "object_cleanup_keys must emit exactly the obj:* family a retired \
             object owns; a missing entry leaks one key per retired object id \
             forever (b.9fn)"
        );
    }

    #[test]
    fn object_cleanup_keys_emits_header_first() {
        // Load-bearing ordering (b.vd8, b.pw6): the restore residue wipe's
        // UNLINK preserves this order, so the header must be reaped first —
        // that is what keeps every snapshot read that proves its state by the
        // header (the fenced read's release, the atomic read, the structural
        // path's proof) sound against a concurrent removal (SDD §7.9, §22.4).
        // The paged teardown's first UNLINK leads with the header too
        // (section G).
        let ks = KeySchema::default();
        let (secs, flds, tss, dels, chs) = cleanup_fixture();

        let got = ks.object_cleanup_keys("42", &secs, &flds, &tss, &dels, &chs);
        assert_eq!(got[0], ks.obj_header("42"), "obj:hdr must be element 0");
    }

    #[test]
    fn object_cleanup_keys_empty_enumerations_still_emit_root_children() {
        // The b.9fn tripwire in its purest form: an object whose sections and
        // tombstones have already been reaped (or that only ever held
        // top-level state) yields NO per-section children keys, yet
        // `obj:ch:0` must still be on the list.
        let ks = KeySchema::default();

        let got = ks.object_cleanup_keys("42", &[], &[], &[], &[], &[]);

        assert_eq!(
            got,
            vec![
                "le:{42}:obj:hdr",
                "le:{42}:obj:secs",
                "le:{42}:obj:flds",
                "le:{42}:obj:tss",
                "le:{42}:obj:dels",
                "le:{42}:obj:chs",
                "le:{42}:obj:ext",
                "le:{42}:obj:wtok",
                "le:{42}:obj:wrel",
                "le:{42}:obj:snapfence",
                "le:{42}:obj:snapintent",
                "le:{42}:obj:ch:0",
            ],
            "with all five enumerations empty the inventory is the eleven \
             object-level keys plus the root children list"
        );
    }

    #[test]
    fn object_cleanup_keys_never_emits_the_stream_key() {
        // Every retirement route appends `deleted` to the stream and gives it
        // an expiry before it unlinks the inventory, so lagging subscribers
        // still observe the `deleted` entry within the grace window. Checked
        // for a numeric id, and for a non-numeric one: the inventory is built
        // from the raw id string, so it still names (and leads with) that id's
        // header, and it carries no stream for either.
        let ks = KeySchema::default();
        let (secs, flds, tss, dels, chs) = cleanup_fixture();

        for object_id in ["42", "not-a-u64"] {
            let inventory = ks.object_cleanup_keys(object_id, &secs, &flds, &tss, &dels, &chs);

            assert!(
                !inventory.iter().any(|k| k.ends_with(":obj:stream")),
                "the inventory of {object_id} must NOT emit a stream key — every retirement \
                 route PEXPIREs it with grace instead so lagging subscribers still observe the \
                 `deleted` entry; got {inventory:?}"
            );
            assert_eq!(
                inventory[0],
                format!("le:{{{object_id}}}:obj:hdr"),
                "the header still leads"
            );
        }
    }

    #[test]
    fn object_cleanup_keys_every_key_carries_the_prefix_and_object_hash_tag() {
        // Two per-key structural invariants, asserted over one inventory:
        //
        // - Prefix: integration tests run under a per-test prefix; a key
        //   family that hardcoded "le:" would escape both the cleanup list
        //   and the `TestValkey` Drop sweep.
        // - Hash tag: every key in a single UNLINK batch must land on one
        //   Cluster slot, or the UNLINK (the paged teardown's, the residue
        //   wipe's) raises CROSSSLOT.
        //
        // The literal-set tests above pin *which* keys are emitted; these two
        // pin the shape every emitted key must have regardless of the set, so
        // a newly-added family that forgot either one fails here.
        let ks = KeySchema::new("test_abc:");
        let (secs, flds, tss, dels, chs) = cleanup_fixture();

        let got = ks.object_cleanup_keys("42", &secs, &flds, &tss, &dels, &chs);

        assert!(!got.is_empty(), "fixture must produce keys");
        for key in &got {
            assert!(
                key.starts_with("test_abc:"),
                "every cleanup key must carry the configured prefix: {key}"
            );
            assert!(
                key.contains("{42}"),
                "cleanup key must carry the {{object_id}} hash tag: {key}"
            );
        }
    }

    // ── G. The paged teardown's pieces (b.fsz MR-D4, D4-T7) ────────────
    //
    // `storage::retirement::Teardown` UNLINKs a retiring object in pages:
    // `object_teardown_first_keys` first, then the keys each index page's
    // members name (`section_teardown_keys`, `tombstone_teardown_keys`, the
    // children list per `obj:chs` member, the field and time-sample rows),
    // `object_teardown_index_keys` last. These pin the first UNLINK's
    // contents and order, the last's, that no piece names a stream key, and
    // that the pieces add up to exactly `object_cleanup_keys`, so the
    // teardown and the inventory authority cannot drift.

    #[test]
    fn object_teardown_first_keys_hold_the_header_and_obj_ext_together_with_every_fixed_key() {
        let ks = KeySchema::default();
        // Hand-written literals: header first, `obj:ext` in the same command
        // (the settle's `EXISTS obj:hdr OR EXISTS obj:ext` liveness gate must
        // never see an `obj:ext` without its header on a dead object), then
        // every other fixed object-level key — no index, which goes last.
        assert_eq!(
            ks.object_teardown_first_keys("42"),
            vec![
                "le:{42}:obj:hdr",
                "le:{42}:obj:ext",
                "le:{42}:obj:wtok",
                "le:{42}:obj:wrel",
                "le:{42}:obj:snapfence",
                "le:{42}:obj:snapintent",
                "le:{42}:obj:ch:0",
            ],
            "the teardown's first UNLINK holds obj:hdr (first) and obj:ext together, with every \
             fixed key but the five indices"
        );
    }

    #[test]
    fn object_teardown_index_keys_are_the_five_indices() {
        let ks = KeySchema::default();
        assert_eq!(
            ks.object_teardown_index_keys("42"),
            vec![
                "le:{42}:obj:secs",
                "le:{42}:obj:flds",
                "le:{42}:obj:tss",
                "le:{42}:obj:dels",
                "le:{42}:obj:chs",
            ],
            "the teardown's last UNLINK is the five structural indices, whose members name the \
             rows the pages before it remove"
        );
    }

    #[test]
    fn object_teardown_pieces_name_no_stream_key() {
        let ks = KeySchema::default();
        let mut pieces = ks.object_teardown_first_keys("42");
        pieces.extend(ks.object_teardown_index_keys("42"));
        pieces.extend(ks.section_teardown_keys("42", "10"));
        pieces.extend(ks.tombstone_teardown_keys("42", "20"));
        assert!(
            !pieces.iter().any(|k| k.ends_with(":obj:stream")),
            "no teardown page may UNLINK the per-object stream: every route PEXPIREs it with \
             grace before the teardown; got {pieces:?}"
        );
    }

    #[test]
    fn object_teardown_pieces_add_up_to_exactly_the_cleanup_inventory() {
        let ks = KeySchema::default();
        let (secs, flds, tss, dels, chs) = cleanup_fixture();

        let mut paged: Vec<String> = ks.object_teardown_first_keys("42");
        for sid in &secs {
            paged.extend(ks.section_teardown_keys("42", sid));
        }
        for sid in &dels {
            paged.extend(ks.tombstone_teardown_keys("42", sid));
        }
        for parent in &chs {
            paged.push(ks.obj_children("42", parent));
        }
        for member in &flds {
            paged.push(ks.obj_field_from_member("42", member));
        }
        for member in &tss {
            paged.push(ks.obj_ts_from_member("42", member));
        }
        paged.extend(ks.object_teardown_index_keys("42"));

        let paged: std::collections::BTreeSet<String> = paged.into_iter().collect();
        let inventory: std::collections::BTreeSet<String> = ks
            .object_cleanup_keys("42", &secs, &flds, &tss, &dels, &chs)
            .into_iter()
            .collect();
        assert_eq!(
            paged, inventory,
            "the paged teardown must remove exactly the keys object_cleanup_keys names — a key \
             only the inventory names leaks per retired object; one only the teardown names is a \
             key the inventory authority does not know"
        );
    }
}
