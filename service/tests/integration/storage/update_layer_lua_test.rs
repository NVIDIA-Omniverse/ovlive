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

// Integration tests for the b.67u Lua EVAL fast path on `update_layer`.
//
// The engineer's fix replaces the WATCH/MULTI/EXEC sequence on field /
// time-sample-only deltas with a single-RTT server-side-atomic EVALSHA
// (Option A — structural commands kept the WATCH/MULTI/EXEC slow path; under
// the default `lua` backend they now commit through the same script's
// structural mode, planned on the structural cache, and keep the slow path
// under `lua-watch-structural` and `txn`).
// The Lua script atomically verifies the header exists, writes the mutation
// set, HSETs the new header, and XADDs to the live stream. Because Lua
// execution on Valkey is atomic-by-construction, no WATCH is needed and no
// retry-on-conflict can fire: `ABORTED` should drop to zero under contention
// and commit-attempts should collapse to 1 per call.
//
// Invariants locked in (each test block is labelled by the suite from the
// task brief):
//
//   1. **Parity** — for every fast-path-eligible delta shape, the resulting
//      Valkey state (header HGETALL, field / time-sample HGETALL, fields /
//      samples sorted set contents, stream XRANGE entries, and the b.75i
//      commit-token record `obj:wtok`) is byte-identical
//      between the legacy WATCH/MULTI/EXEC backend (`UpdateLayerBackend::Txn`)
//      and the new Lua backend (`UpdateLayerBackend::Lua`). This is the tight
//      regression net — a drift anywhere in the Lua ARGV encoder shows up as
//      an HSET/ZADD arg mismatch. The one header column that cannot be
//      compared by equality (`last_server_ts`, a wall clock, and the two
//      backends run on independent Valkey instances) is compared as a bounded
//      skew instead — see `assert_commit_clock_parity`.
//
//   2. **Single-RTT** — at N=64 concurrent writers on one object, ≥99 % of
//      Lua-backend commits record `attempts == 1` in the attempts histogram
//      and 0 writers surface ABORTED.
//
//   3. **Header-anomaly mapping** — Lua EVAL against an object whose
//      `obj:hdr` is gone surfaces `NotFound` (not `Aborted`, and — since
//      b.f1v — not `FailedPrecondition`). Its counterpart claim, that a
//      header which is PRESENT but carries no usable `seq` stays
//      `FailedPrecondition`, is asserted in `b_k1r_tests` (Lua) and
//      `b_qw6_tests` (txn); the absent-header txn half lives in
//      `update_layer_transaction_test`. Those four tests together pin
//      "absent and partial are distinct surfaces, identically on both
//      backends".
//
//   4. **NOSCRIPT recovery** — after a `SCRIPT FLUSH` on the server, the
//      next Lua-backend commit auto-reloads the script via
//      `storage::lua::evalsha_once_with_reload` (reload on `NOSCRIPT`, then
//      one retry) and succeeds.
//
//   5. **Backend toggle** — `with_update_layer_backend(Txn)` routes through
//      the legacy WATCH/MULTI/EXEC path, and commits behave like the
//      pre-b.67u service.

use std::collections::HashMap;
use std::sync::Arc;

use fred::interfaces::{
    ClientLike, HashesInterface, KeysInterface, LuaInterface, SortedSetsInterface, StreamsInterface,
};

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;

use live_edit::config::{UpdateLayerBackend, LAYER_STREAM_MAX_BYTES_DEFAULT};
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::{operation_request_digest, LayerStore, Operation};
use live_edit::storage::lua::{
    encode_delta, is_fast_path, update_script_shared_keys, EncodedMutations, UpdateScriptMode,
    STREAM_COLLISION_PROBE_SCRIPT, UPDATE_LAYER_FAST_SCRIPT, UPDATE_SCRIPT_ARG_CALL_TOKEN,
    UPDATE_SCRIPT_ARG_CLIENT_ID, UPDATE_SCRIPT_ARG_CLIENT_TS, UPDATE_SCRIPT_ARG_COMMIT_KEY,
    UPDATE_SCRIPT_ARG_NUM_MUTATIONS, UPDATE_SCRIPT_ARG_OBJECT_ID, UPDATE_SCRIPT_ARG_RECORD_SUFFIX,
    UPDATE_SCRIPT_ARG_RELAY_BUDGET_BYTES, UPDATE_SCRIPT_ARG_RELAY_GENERATIONS,
    UPDATE_SCRIPT_ARG_RELAY_GENERATION_BYTES, UPDATE_SCRIPT_ARG_RELAY_MIN_ENTRIES,
    UPDATE_SCRIPT_ARG_RELAY_OMPF, UPDATE_SCRIPT_ARG_RING_TTL_MS, UPDATE_SCRIPT_ARG_SERVER_TS,
    UPDATE_SCRIPT_ARG_STREAM_MAXLEN, UPDATE_SCRIPT_ARG_WRITER_GUARD,
    UPDATE_SCRIPT_KEY_COMMIT_RELEASE, UPDATE_SCRIPT_KEY_COMMIT_TOKEN, UPDATE_SCRIPT_KEY_EXT_INDEX,
    UPDATE_SCRIPT_KEY_FENCE, UPDATE_SCRIPT_KEY_FIELDS_INDEX, UPDATE_SCRIPT_KEY_HEADER,
    UPDATE_SCRIPT_KEY_INTENT, UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL, UPDATE_SCRIPT_KEY_STREAM,
    UPDATE_SCRIPT_KEY_TS_INDEX, UPDATE_SCRIPT_SHARED_ARGV, UPDATE_SCRIPT_SHARED_KEYS,
};
use live_edit::storage::notifications::{DELETED_STREAM_PAYLOAD, NOTIFICATION_PAYLOAD_FIELD};
use live_edit::storage::relay_retention::RelayRetention;
use live_edit::storage::session::SessionStore;
use live_edit::storage::snapshot_fence::{
    self as fence, AcquireVerdict, LayerSize, ReleaseVerdict,
};
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{OwnedDelta, SetField, TimeSample as OodfTimeSample};

use crate::common::owned_delta::{delta_to_ompf_bytes, delta_with_set_field_inline, empty_delta};
/// Mirror of the module-private `storage::layer::EMPTY_HEADER_SENTINEL`. Shared
/// via `tests/common` so every module that asserts it holds one literal.
use crate::common::EMPTY_HEADER_SENTINEL_MSG;

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers (mirrored on the patterns in `update_layer_transaction_test.rs`
// and `update_layer_admission_test.rs` so future diffs show up as assertion
// changes rather than scaffolding churn).
// ─────────────────────────────────────────────────────────────────────────────

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

async fn create_test_session(tv: &TestValkey) -> String {
    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    ss.create("test-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed")
}

async fn create_test_layer(ls: &LayerStore, session_id: &str, layer_name: &str) -> u64 {
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    ls.create_layer(
        session_id,
        layer_name,
        "test-user",
        &state,
        mock_ext_store(),
    )
    .await
    .expect("create_layer must succeed")
}

/// Construct a LayerStore pinned to a given backend. Default
/// `layer_stream_maxlen` matches the production helm chart (4096) — keeping
/// it the same on both backends is part of the parity contract.
fn layer_store_with_backend(tv: &TestValkey, backend: UpdateLayerBackend) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(backend)
}

// ─────────────────────────────────────────────────────────────────────────────
// State-dumping utilities. The parity tests compare full per-object keyspaces
// between the two backends. A single normalised struct captures everything a
// fast-path delta could touch.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
struct HashSnapshot(Vec<(String, String)>);

impl HashSnapshot {
    fn from_map(m: HashMap<String, String>) -> Self {
        let mut v: Vec<(String, String)> = m.into_iter().collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        HashSnapshot(v)
    }
}

/// Normalised view of everything a fast-path update can touch. Fields that
/// are wall-clock-dependent (`created_at`, `modified_at`, `last_server_ts`)
/// are dropped so two backends running on independent TestValkey instances
/// still compare equal, and so is the header's `commit_chain`, which each
/// materialization seeds with a fresh random value. The `session_id` and
/// `object_id` are rewritten to sentinel strings by `normalise_oid` after the
/// dump.
///
/// `last_server_ts` being un-comparable by EQUALITY does not make it
/// un-comparable: it is returned alongside the keyspace as a [`CommitClock`]
/// and checked by [`assert_commit_clock_parity`]. Dropping it outright left
/// the checkpoint gate's only commit-clock input outside the parity net —
/// a unit or omission drift between the two writers (b.yut was exactly a unit
/// bug on this column) would not have surfaced anywhere in this file.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ObjectKeyspace {
    header: HashSnapshot,
    fields: Vec<(String, HashSnapshot)>, // sorted by field_member
    fields_index: Vec<(String, i64)>,    // (member, score) sorted by member
    time_samples: Vec<(String, HashSnapshot)>, // sorted by ts_member
    time_samples_index: Vec<(String, i64)>,
    stream_payloads: Vec<(String, String)>, // (entry_id, normalised payload) in order
    /// `le:{oid}:obj:wtok` (b.75i): `(client_id, record)` sorted by
    /// `client_id`, with the wall-clock `<server_ts>` replaced by a sentinel the
    /// same way the stream payloads' is. The record is
    /// `<token>:<seq>:<server_ts>` for a call without a client operation id, and
    /// `op.<operation_id>:<seq>:<server_ts>:<token>:<request_digest>` for one
    /// with (b.v6d). Both backends write the record inside the commit's own
    /// atomic unit, so it is part of the parity surface; and a refused commit
    /// must leave it untouched, which every before/after comparison in this
    /// file now checks too.
    commit_token_records: Vec<(String, String)>,
}

/// Whether `s` is `len` lowercase hex digits.
fn is_lower_hex(s: &str, len: usize) -> bool {
    s.len() == len
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A commit-token record, split into its parts.
#[derive(Debug)]
struct CommitRecord<'a> {
    /// The match key: the 32-hex per-call token, or `op.<operation_id>`.
    key: &'a str,
    seq: u64,
    /// `(token, request_digest)` — present exactly when `key` is an `op.` key.
    operation_tail: Option<(&'a str, &'a str)>,
}

impl CommitRecord<'_> {
    /// The record with its wall-clock `server_ts` replaced by a sentinel.
    fn with_server_ts_normalised(&self) -> String {
        match self.operation_tail {
            None => format!("{}:{}:<server_ts>", self.key, self.seq),
            Some((token, digest)) => {
                format!("{}:{}:<server_ts>:{token}:{digest}", self.key, self.seq)
            }
        }
    }
}

/// Split a commit-token record, asserting one of the two shapes both backends
/// write: `<token>:<seq>:<server_ts>` (32 lowercase hex digits, then two
/// decimals), or, for a call carrying a client operation id,
/// `op.<operation_id>:<seq>:<server_ts>:<token>:<request_digest>` (the token
/// again 32 hex digits, the digest 64).
fn parse_commit_token_record(record: &str) -> CommitRecord<'_> {
    let parts: Vec<&str> = record.split(':').collect();
    let decimal = |slot: &str, what: &str| -> u64 {
        slot.parse()
            .unwrap_or_else(|e| panic!("record {what} must be a u64 in {record:?}: {e}"))
    };
    match parts.as_slice() {
        [token, seq, ts] => {
            assert!(
                is_lower_hex(token, 32),
                "a record without an operation id is keyed by its 32-lowercase-hex token; got \
                 {record:?}"
            );
            // Parsed for its shape only: the dump replaces it with a sentinel.
            decimal(ts, "server_ts");
            CommitRecord {
                key: token,
                seq: decimal(seq, "seq"),
                operation_tail: None,
            }
        }
        [key, seq, ts, token, digest] => {
            assert!(
                key.starts_with("op.") && key.len() > "op.".len(),
                "a five-part record must be keyed `op.<operation_id>`; got {record:?}"
            );
            assert!(
                is_lower_hex(token, 32) && is_lower_hex(digest, 64),
                "an operation record carries the 32-hex call token and the 64-hex request \
                 digest after its timestamp; got {record:?}"
            );
            decimal(ts, "server_ts");
            CommitRecord {
                key,
                seq: decimal(seq, "seq"),
                operation_tail: Some((token, digest)),
            }
        }
        _ => panic!(
            "a commit-token record is `<token>:<seq>:<server_ts>` or \
             `op.<id>:<seq>:<server_ts>:<token>:<digest>`; got {record:?}"
        ),
    }
}

/// Hex-decode a `field_member` key component (`hex(key_name)`) back to the raw
/// UTF-8 key name. Mirrors `hex_encode` in `src/storage/keys.rs`.
fn decode_hex_key(hex: &str) -> String {
    let bytes: Vec<u8> = (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16).expect("field member key must be valid hex")
        })
        .collect();
    String::from_utf8(bytes).expect("field key_name must be valid UTF-8")
}

/// The `obj:hdr last_server_ts` a commit stamped, plus the wall clock read at
/// dump time. Carried out of the dump separately because the value is
/// wall-clock-dependent — two backends on two independent TestValkey
/// instances can never produce the same one — but is still a parity surface:
/// it is the ONLY commit-clock input the periodic checkpoint gate has, and
/// b.yut was a units bug on precisely this column.
#[derive(Debug, Clone, Copy)]
struct CommitClock {
    /// Raw `last_server_ts` as stored, `None` when the column is absent.
    /// Production writes MICROSECONDS (`domain::time::now_micros`).
    last_server_ts: Option<u64>,
    /// `now_micros()` sampled at dump time, so each backend's stamp can be
    /// bounded against its OWN observation window rather than against the
    /// other backend's (the two dumps are separated by a whole TestValkey
    /// setup, which is not a bounded quantity on a loaded runner).
    observed_at_micros: u64,
}

/// Assert both backends stamped `obj:hdr last_server_ts` with the same clock,
/// in the same unit, at commit time.
///
/// Exact equality is unavailable — the two commits happen on two independent
/// TestValkey instances, seconds apart — so the contract is checked as a
/// bounded skew, per backend, against a clock read at that backend's own dump:
///
/// - **Present.** A backend that stops stamping the column strands the
///   checkpoint gate with no commit clock at all.
/// - **Not ahead of the observation.** A commit cannot be stamped in the
///   future.
/// - **Within `MAX_COMMIT_AGE_MICROS` of it.** This is the units assertion.
///   The window is deliberately loose in absolute terms (10 minutes) and still
///   catches either failure mode by many orders of magnitude: milliseconds
///   written where microseconds are expected lands the stamp ~56 years in the
///   PAST, and microseconds read as milliseconds ~56,000 years in the FUTURE.
fn assert_commit_clock_parity(txn: CommitClock, lua: CommitClock) {
    /// 10 minutes. Generous enough that a stalled runner between the commit
    /// and the dump cannot flake it; ~6 orders of magnitude tighter than any
    /// 1000× unit error.
    const MAX_COMMIT_AGE_MICROS: u64 = 600_000_000;

    for (backend, clock) in [("txn", txn), ("lua", lua)] {
        let stamped = clock.last_server_ts.unwrap_or_else(|| {
            panic!(
                "the {backend} backend must stamp obj:hdr last_server_ts on \
                 every commit — it is the checkpoint gate's only commit clock, \
                 and a backend that omits it leaves its layers with no quiesce \
                 arm and no lag observation at all (b.yut)"
            )
        });
        assert!(
            stamped <= clock.observed_at_micros,
            "the {backend} backend stamped last_server_ts={stamped} µs, which is \
             AHEAD of the µs clock read after the commit \
             ({}) — a commit cannot happen in the future. Microseconds read as \
             milliseconds land ~56,000 years out and look exactly like this",
            clock.observed_at_micros
        );
        let age = clock.observed_at_micros - stamped;
        assert!(
            age < MAX_COMMIT_AGE_MICROS,
            "the {backend} backend stamped last_server_ts={stamped}, which is \
             {age} µs before the µs clock read after the commit \
             ({}) — that commit supposedly happened over 10 minutes ago. \
             Milliseconds written where MICROSECONDS are expected land ~56 years \
             in the past and look exactly like this (b.yut was this bug on the \
             consuming side)",
            clock.observed_at_micros
        );
    }

    // Both stamps are now known to be recent µs wall clocks. Their difference
    // is therefore bounded by the inter-dump gap, which pins the two backends
    // to the same clock and the same unit relative to each other — the claim
    // the removed equality comparison used to make.
    let skew = txn
        .last_server_ts
        .unwrap()
        .abs_diff(lua.last_server_ts.unwrap());
    let observation_gap = txn.observed_at_micros.abs_diff(lua.observed_at_micros);
    assert!(
        skew <= observation_gap + MAX_COMMIT_AGE_MICROS,
        "the two backends' last_server_ts stamps differ by {skew} µs across an \
         observation gap of only {observation_gap} µs — they are not reading the \
         same clock in the same unit"
    );
}

async fn dump_object_keyspace(tv: &TestValkey, oid: u64) -> (ObjectKeyspace, CommitClock) {
    let keys = tv.key_schema();
    let oid_str = oid.to_string();
    let pool = &tv.client().pool;

    // Header HGETALL — drop wall-clock-dependent fields, and the
    // per-materialization random `commit_chain`, so two independent
    // `create_layer` calls compare equal. `last_server_ts` is returned
    // separately (see `CommitClock`) rather than simply discarded.
    let header_key = keys.obj_header(&oid_str);
    let mut header_map: HashMap<String, String> = pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL header must succeed");
    header_map.remove("created_at");
    header_map.remove("modified_at");
    header_map.remove(live_edit::storage::layer::HEADER_FIELD_COMMIT_CHAIN);
    let commit_clock = CommitClock {
        last_server_ts: header_map
            .remove("last_server_ts")
            .map(|raw| raw.parse().expect("last_server_ts must be numeric")),
        observed_at_micros: live_edit::domain::time::now_micros(),
    };

    // Fields index ZRANGE WITHSCORES.
    let fields_index_key = keys.obj_fields_index(&oid_str);
    let fields_idx: Vec<(String, f64)> = pool
        .zrange(&fields_index_key, 0, -1, None, false, None, true)
        .await
        .expect("ZRANGE fields index must succeed");
    let mut fields_index: Vec<(String, i64)> =
        fields_idx.into_iter().map(|(m, s)| (m, s as i64)).collect();
    fields_index.sort_by(|a, b| a.0.cmp(&b.0));

    // Per-field HGETALL — uses the indexed member to reconstruct the field
    // key. The member is `KeySchema::field_member("{sid}", "{key_name}")`
    // which is `"{sid}:{hex(key_name)}"` — see src/storage/keys.rs. The
    // second component is ALREADY hex-encoded, so it must be hex-decoded back
    // to the raw key_name before handing to `obj_field` (which hex-encodes
    // internally); passing the hex string straight through double-encodes and
    // reads an empty non-existent hash. (The time-samples loop below already
    // decodes its hex component the same way.)
    let mut fields: Vec<(String, HashSnapshot)> = Vec::new();
    for (member, _) in &fields_index {
        let (sid, key_hex) = member.split_once(':').expect("field_member always has ':'");
        let key_name = decode_hex_key(key_hex);
        let field_key = keys.obj_field(&oid_str, sid, &key_name);
        let fmap: HashMap<String, String> = pool
            .hgetall(&field_key)
            .await
            .expect("HGETALL field must succeed");
        fields.push((member.clone(), HashSnapshot::from_map(fmap)));
    }

    // Time samples index + per-sample HGETALL.
    let ts_index_key = keys.obj_time_samples_index(&oid_str);
    let ts_idx: Vec<(String, f64)> = pool
        .zrange(&ts_index_key, 0, -1, None, false, None, true)
        .await
        .expect("ZRANGE time samples index must succeed");
    let mut time_samples_index: Vec<(String, i64)> =
        ts_idx.into_iter().map(|(m, s)| (m, s as i64)).collect();
    time_samples_index.sort_by(|a, b| a.0.cmp(&b.0));

    let mut time_samples: Vec<(String, HashSnapshot)> = Vec::new();
    for (member, _) in &time_samples_index {
        // `KeySchema::time_sample_member` returns "{sid}:{f64_bits_hex}".
        let (sid, time_enc) = member.split_once(':').expect("ts_member always has ':'");
        let bits = u64::from_str_radix(time_enc, 16).expect("ts_member bit-encoding must parse");
        let time_f64 = f64::from_bits(bits);
        let ts_key = keys.obj_time_sample(&oid_str, sid, time_f64);
        let tmap: HashMap<String, String> = pool
            .hgetall(&ts_key)
            .await
            .expect("HGETALL time sample must succeed");
        time_samples.push((member.clone(), HashSnapshot::from_map(tmap)));
    }

    // Stream XRANGE — normalise payloads by replacing the server_ts slot
    // (sixth pipe-separated field) with a sentinel so wall-clock differences
    // don't break parity. Every other slot (tag, oid, seq, client_id,
    // client_ts) is deterministic under the test's controlled inputs.
    let stream_key = keys.layer_stream(oid);
    // Values read as bytes: a commit's relay `ompf` field is binary. Only the
    // payload enters the snapshot; both backends relay the same bytes.
    let entries: Vec<(String, HashMap<String, bytes::Bytes>)> = pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE must succeed");
    let stream_payloads: Vec<(String, String)> = entries
        .into_iter()
        .map(|(id, fmap)| {
            let raw = fmap
                .get("payload")
                .map(|p| String::from_utf8_lossy(p).into_owned())
                .unwrap_or_default();
            let parts: Vec<&str> = raw.split('|').collect();
            let normalised = if parts.len() == 6 {
                format!(
                    "{}|{}|{}|{}|{}|<server_ts>",
                    parts[0], parts[1], parts[2], parts[3], parts[4]
                )
            } else {
                raw
            };
            (id, normalised)
        })
        .collect();

    // Commit-token record (b.75i) — normalise the server_ts slot exactly as
    // the stream payloads' is above. The token and the sequence stay: the
    // sequence is deterministic, a before/after comparison reads one object's
    // record twice, and the parity harnesses either fix the token (see
    // `parity_store`) or, without the seam, normalise it in `normalise_oid`.
    let records: HashMap<String, String> = pool
        .hgetall(keys.obj_write_token(&oid_str))
        .await
        .expect("HGETALL commit-token record must succeed");
    let mut commit_token_records: Vec<(String, String)> = records
        .into_iter()
        .map(|(client_id, record)| {
            let normalised = parse_commit_token_record(&record).with_server_ts_normalised();
            (client_id, normalised)
        })
        .collect();
    commit_token_records.sort();

    (
        ObjectKeyspace {
            header: HashSnapshot::from_map(header_map),
            fields,
            fields_index,
            time_samples,
            time_samples_index,
            stream_payloads,
            commit_token_records,
        },
        commit_clock,
    )
}

/// Rewrite the object_id / session_id occurrences to constant sentinels so
/// two backends (which each own their own ID counters) compare equal.
fn normalise_oid(mut ks: ObjectKeyspace, oid: u64, session_id: &str) -> ObjectKeyspace {
    let oid_s = oid.to_string();
    let replace_in = |v: String| {
        if v == oid_s {
            "<object_id>".to_owned()
        } else if v == session_id {
            "<session_id>".to_owned()
        } else {
            v
        }
    };
    ks.header = HashSnapshot(
        ks.header
            .0
            .into_iter()
            .map(|(k, v)| (k, replace_in(v)))
            .collect(),
    );
    ks.stream_payloads = ks
        .stream_payloads
        .into_iter()
        .map(|(id, payload)| {
            let parts: Vec<String> = payload
                .split('|')
                .map(|p| {
                    if p == oid_s {
                        "<object_id>".to_owned()
                    } else {
                        p.to_owned()
                    }
                })
                .collect();
            (id, parts.join("|"))
        })
        .collect();
    // Without `test-fault-injection` there is no fixed-token seam, so each
    // backend's calls mint random tokens and only the record's shape and
    // sequence are comparable. With it, `parity_store` hands both backends the
    // same token sequence and the token is compared verbatim.
    #[cfg(not(feature = "test-fault-injection"))]
    {
        ks.commit_token_records = ks
            .commit_token_records
            .into_iter()
            .map(|(client_id, record)| {
                // The dump already checked the shape; blank the random token,
                // which is the match key without an operation id and the
                // fourth part with one. The digest is deterministic and stays.
                let mut parts: Vec<&str> = record.split(':').collect();
                if parts.len() == 5 {
                    parts[3] = "<token>";
                } else {
                    parts[0] = "<token>";
                }
                (client_id, parts.join(":"))
            })
            .collect();
    }
    ks
}

/// The tokens [`parity_store`] hands out, one per `update_layer` call, in
/// order. Both backends consume the same sequence, so the N-th commit on each
/// records the same token. 64 is well above the longest commit sequence any
/// parity harness here drives; a harness that outgrew it would fall back to
/// random tokens and fail the parity comparison loudly rather than pass.
#[cfg(feature = "test-fault-injection")]
fn parity_commit_tokens() -> Arc<std::sync::Mutex<std::collections::VecDeque<u128>>> {
    const PARITY_TOKEN_BASE: u128 = 0x7a11_7e57_0000_0000_0000_0000_0000_0000;
    Arc::new(std::sync::Mutex::new(
        (1..=64).map(|i| PARITY_TOKEN_BASE | i).collect(),
    ))
}

/// A parity-harness store: [`layer_store_with_backend`], plus — under
/// `test-fault-injection` — a fixed commit-token sequence, so the two
/// backends' `obj:wtok` records can be compared byte for byte (b.75i).
fn parity_store(tv: &TestValkey, backend: UpdateLayerBackend) -> LayerStore {
    let store = layer_store_with_backend(tv, backend);
    #[cfg(feature = "test-fault-injection")]
    let store = store.with_commit_tokens_for_test(parity_commit_tokens());
    store
}

// =============================================================================
// 1. PARITY SUITE — commit same delta via Lua vs Txn, assert identical state.
//
// Each test spins up two independent `TestValkey` fixtures (one per backend),
// creates a layer on each, commits the same delta through each backend, and
// asserts the dumped per-object keyspace is byte-identical modulo the known-
// volatile timestamp / id fields already normalised in `dump_object_keyspace`.
// =============================================================================

/// Shared parity harness. Run the builder delta against a fresh layer on each
/// backend and compare. The delta closure is called twice (once per backend)
/// so it should be deterministic.
async fn run_parity<F>(layer_name: &str, build_delta: F)
where
    F: Fn() -> OwnedDelta,
{
    run_parity_with(layer_name, None, build_delta).await;
}

/// [`run_parity`] for a call that carries a client `operation_id` (b.v6d): both
/// backends commit through `update_layer_with_operation` with `operation`, so
/// the dumped `obj:wtok` record is the `op.<id>:…:<token>:<digest>` shape and
/// is compared byte for byte along with the rest of the keyspace.
async fn run_parity_with<F>(layer_name: &str, operation: Option<Operation>, build_delta: F)
where
    F: Fn() -> OwnedDelta,
{
    let tv_txn = TestValkey::setup().await;
    let ls_txn = parity_store(&tv_txn, UpdateLayerBackend::Txn);
    let sid_txn = create_test_session(&tv_txn).await;
    let oid_txn = create_test_layer(&ls_txn, &sid_txn, layer_name).await;
    ls_txn
        .update_layer_with_operation_relayed(
            &sid_txn,
            layer_name,
            oid_txn,
            Some(build_delta()),
            "parity-client",
            42,
            operation.clone(),
        )
        .await
        .expect("txn-backend update must succeed");
    let (ks_txn, clock_txn) = dump_object_keyspace(&tv_txn, oid_txn).await;
    let ks_txn = normalise_oid(ks_txn, oid_txn, &sid_txn);

    let tv_lua = TestValkey::setup().await;
    let ls_lua = parity_store(&tv_lua, UpdateLayerBackend::Lua);
    let sid_lua = create_test_session(&tv_lua).await;
    let oid_lua = create_test_layer(&ls_lua, &sid_lua, layer_name).await;
    ls_lua
        .update_layer_with_operation_relayed(
            &sid_lua,
            layer_name,
            oid_lua,
            Some(build_delta()),
            "parity-client",
            42,
            operation,
        )
        .await
        .expect("lua-backend update must succeed");
    let (ks_lua, clock_lua) = dump_object_keyspace(&tv_lua, oid_lua).await;
    let ks_lua = normalise_oid(ks_lua, oid_lua, &sid_lua);

    assert_eq!(
        ks_lua, ks_txn,
        "Lua-vs-Txn byte-parity violated.\n\nLUA: {ks_lua:#?}\n\nTXN: {ks_txn:#?}"
    );
    assert_commit_clock_parity(clock_txn, clock_lua);
}

/// b.75i: both backends record the commit token as the identical string
/// `<token>:<seq>:<server_ts>`, and the timestamp in it is the commit's own.
///
/// The parity harnesses compare the record with its `server_ts` normalised,
/// because two backends on two fixtures cannot share a wall clock. This test
/// closes that gap per backend: the recorded `server_ts` must equal the
/// header's `last_server_ts` AND the `server_ts` in the stream entry at the
/// recorded sequence. A retry reports the record's `(seq, server_ts)` as the
/// commit's, so a record whose timestamp drifted from the commit's would hand
/// the client a timestamp no subscriber ever saw.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_commit_token_record_is_identical_on_both_backends() {
    const TOKEN: u128 = 0x0bad_cafe_0000_0000_0000_0000_0000_0075;
    const CLIENT: &str = "parity-client";

    let mut records = Vec::new();
    for backend in [UpdateLayerBackend::Txn, UpdateLayerBackend::Lua] {
        let tv = TestValkey::setup().await;
        let ls = layer_store_with_backend(&tv, backend).with_commit_tokens_for_test(Arc::new(
            std::sync::Mutex::new([TOKEN].into_iter().collect()),
        ));
        let sid = create_test_session(&tv).await;
        let oid = create_test_layer(&ls, &sid, "b75i-record-parity").await;
        let (seq, server_ts) = ls
            .update_layer_relayed(
                &sid,
                "b75i-record-parity",
                oid,
                Some(delta_with_set_field_inline(7, "color", vec![0x75], 0)),
                CLIENT,
                42,
            )
            .await
            .unwrap_or_else(|e| panic!("{backend:?}: the update must commit: {e:?}"));

        let pool = &tv.client().pool;
        let oid_str = oid.to_string();
        let record: HashMap<String, String> = pool
            .hgetall(tv.key_schema().obj_write_token(&oid_str))
            .await
            .expect("HGETALL commit-token record must succeed");
        let expected = format!("{TOKEN:032x}:{seq}:{server_ts}");
        assert_eq!(
            record,
            HashMap::from([(CLIENT.to_owned(), expected.clone())]),
            "{backend:?}: exactly one record, under the writer's client_id, carrying the token \
             verbatim and the (seq, server_ts) the call returned"
        );

        let last_server_ts: Option<String> = pool
            .hget(tv.key_schema().obj_header(&oid_str), "last_server_ts")
            .await
            .expect("HGET last_server_ts must succeed");
        assert_eq!(
            last_server_ts,
            Some(server_ts.to_string()),
            "{backend:?}: the recorded server_ts must be the one the commit stamped on the header"
        );
        let stream_id = format!("{seq}-0");
        // Fields read as bytes: an entry's relay `ompf` field is binary.
        let entry: Vec<(String, HashMap<String, bytes::Bytes>)> = pool
            .xrange_values(
                tv.key_schema().layer_stream(oid),
                &stream_id,
                &stream_id,
                None,
            )
            .await
            .expect("XRANGE must succeed");
        let payload = entry
            .first()
            .and_then(|(_, fields)| fields.get("payload"))
            .map(|p| std::str::from_utf8(p).expect("a stream payload is UTF-8"))
            .unwrap_or_else(|| panic!("{backend:?}: no stream entry at {stream_id}"));
        assert_eq!(
            payload.split('|').nth(5),
            Some(server_ts.to_string().as_str()),
            "{backend:?}: the recorded server_ts must be the one the commit published"
        );

        records.push(format!("{TOKEN:032x}:{seq}:<server_ts>"));
    }
    assert_eq!(
        records[0], records[1],
        "the txn and Lua records must be byte-identical apart from the wall clock"
    );
}

/// The client `operation_id` every operation-record parity case here carries.
const PARITY_OPERATION_ID: &str = "2f1c6b0e-9d1a-4c55-8a37-6e2b4f0d9c11";

/// The operation a parity case sends: [`PARITY_OPERATION_ID`] with the digest
/// the gRPC handler would compute for `layer_name` and a stand-in payload.
/// What the digest is computed over does not matter at this layer; that both
/// backends record the one they were handed does.
fn parity_operation(layer_name: &str) -> Operation {
    Operation::new(
        PARITY_OPERATION_ID,
        operation_request_digest(layer_name, b"b.v6d parity payload"),
    )
}

/// b.v6d T9: with a client `operation_id`, both backends record
/// `op.<operation_id>:<seq>:<server_ts>:<token>:<request_digest>` — byte for
/// byte the same string — and its timestamp is the commit's own.
///
/// The twin of [`update_layer_commit_token_record_is_identical_on_both_backends`]
/// for the operation shape. The match key and the suffix reach the two
/// backends by different routes (ARGV[8] / ARGV[9] into the script, the
/// `CommitIdentity::record` string into the `MULTI`), so a drift in either
/// spelling shows up here as a record the other backend's re-send would not
/// recognise.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_operation_record_is_identical_on_both_backends() {
    use live_edit::storage::layer::OperationOutcome;

    const TOKEN: u128 = 0x0bad_cafe_0000_0000_0000_0000_0000_06d6;
    const CLIENT: &str = "parity-client";
    const LAYER: &str = "bv6d-record-parity";

    let operation = parity_operation(LAYER);
    let mut records = Vec::new();
    for backend in [UpdateLayerBackend::Txn, UpdateLayerBackend::Lua] {
        let tv = TestValkey::setup().await;
        let ls = layer_store_with_backend(&tv, backend).with_commit_tokens_for_test(Arc::new(
            std::sync::Mutex::new([TOKEN].into_iter().collect()),
        ));
        let sid = create_test_session(&tv).await;
        let oid = create_test_layer(&ls, &sid, LAYER).await;
        let commit = ls
            .update_layer_with_operation_relayed(
                &sid,
                LAYER,
                oid,
                Some(delta_with_set_field_inline(7, "color", vec![0x6d], 0)),
                CLIENT,
                42,
                Some(operation.clone()),
            )
            .await
            .unwrap_or_else(|e| panic!("{backend:?}: the update must commit: {e:?}"));
        assert_eq!(
            commit.outcome,
            OperationOutcome::Applied,
            "{backend:?}: a first commit carrying an operation id is APPLIED"
        );

        let pool = &tv.client().pool;
        let oid_str = oid.to_string();
        let record: HashMap<String, String> = pool
            .hgetall(tv.key_schema().obj_write_token(&oid_str))
            .await
            .expect("HGETALL commit-token record must succeed");
        let expected = format!(
            "op.{PARITY_OPERATION_ID}:{}:{}:{TOKEN:032x}:{}",
            commit.seq,
            commit.server_ts,
            operation.digest()
        );
        assert_eq!(
            record,
            HashMap::from([(CLIENT.to_owned(), expected)]),
            "{backend:?}: exactly one record, under the writer's client_id, keyed `op.<id>` and \
             carrying the (seq, server_ts) the call returned, then the call token and the digest"
        );
        let last_server_ts: Option<String> = pool
            .hget(tv.key_schema().obj_header(&oid_str), "last_server_ts")
            .await
            .expect("HGET last_server_ts must succeed");
        assert_eq!(
            last_server_ts,
            Some(commit.server_ts.to_string()),
            "{backend:?}: the recorded server_ts must be the one the commit stamped on the header"
        );

        records.push(format!(
            "op.{PARITY_OPERATION_ID}:{}:<server_ts>:{TOKEN:032x}:{}",
            commit.seq,
            operation.digest()
        ));
    }
    assert_eq!(
        records[0], records[1],
        "the txn and Lua operation records must be byte-identical apart from the wall clock"
    );
}

/// b.v6d T9: the full-keyspace parity dump for a commit carrying a client
/// `operation_id`. Everything [`run_parity`] compares, with the record in its
/// operation shape — so the operation path changes nothing else a commit
/// writes on either backend.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_parity_set_field_inline_with_operation_id() {
    const LAYER: &str = "parity-layer-set-field-op";
    run_parity_with(LAYER, Some(parity_operation(LAYER)), || {
        delta_with_set_field_inline(7, "color", vec![0xDE, 0xAD, 0xBE, 0xEF], 0)
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_parity_set_field_inline() {
    run_parity("parity-layer-set-field", || {
        delta_with_set_field_inline(7, "color", vec![0xDE, 0xAD, 0xBE, 0xEF], 0)
    })
    .await;
}

/// b.yvh: a run of commits whose relay bytes pass a small byte budget leaves a
/// byte-identical keyspace on both backends — the header's `relay_gens` (the
/// dump keeps it) and the stream the byte trims left behind. Every relay is
/// 20 KiB or more, so each entry is a stream node of its own and `XTRIM MINID
/// ~` trims exactly the same entries on both (see `relay_retention_test.rs`).
/// The budget is 64 KiB (generation size 8 KiB, so each commit closes its own
/// generation): the third commit, at seq 4, brings the kept bytes to 20480 +
/// 30720 + 20480 = 71680 > 64 KiB and already drops generation `2:`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_parity_relay_byte_budget_folds_and_trims_identically() {
    const LAYER: &str = "parity-relay-budget";
    const BUDGET: u64 = 64 * 1024;
    const RELAY_LENGTHS: [usize; 8] = [
        20_480, 30_720, 20_480, 40_960, 20_480, 81_920, 20_480, 24_576,
    ];

    let mut dumps = Vec::new();
    for backend in [UpdateLayerBackend::Txn, UpdateLayerBackend::Lua] {
        let tv = TestValkey::setup().await;
        let ls = parity_store(&tv, backend).with_layer_stream_max_bytes(BUDGET);
        let sid = create_test_session(&tv).await;
        let oid = create_test_layer(&ls, &sid, LAYER).await;
        for (i, len) in RELAY_LENGTHS.into_iter().enumerate() {
            let delta = delta_with_set_field_inline(7, &format!("budget-{i}"), vec![i as u8], 0);
            let overhead = delta_to_ompf_bytes(&delta, &[&[]]).len();
            let pad = vec![0x5A_u8; len - overhead];
            let relay = bytes::Bytes::from(delta_to_ompf_bytes(&delta, &[&pad]));
            ls.update_layer_with_operation(
                &sid,
                LAYER,
                oid,
                Arc::new(Some(delta)),
                Some(relay),
                "parity-client",
                42 + i as u64,
                None,
            )
            .await
            .unwrap_or_else(|e| panic!("{backend:?}: commit {i} must succeed: {e:?}"));
        }
        let (ks, _) = dump_object_keyspace(&tv, oid).await;
        let ks = normalise_oid(ks, oid, &sid);
        assert!(
            ks.header
                .0
                .iter()
                .any(|(k, _)| k == live_edit::storage::relay_retention::HEADER_FIELD_RELAY_GENS),
            "{backend:?}: every commit with the budget on writes relay_gens"
        );
        assert!(
            ks.stream_payloads.len() < RELAY_LENGTHS.len() + 1,
            "{backend:?}: fixture: the byte budget must have trimmed the stream; got {:?}",
            ks.stream_payloads
        );
        dumps.push(ks);
    }
    assert_eq!(
        dumps[1], dumps[0],
        "Lua-vs-Txn byte-parity violated under the relay byte budget.\n\nLUA: {:#?}\n\nTXN: \
         {:#?}",
        dumps[1], dumps[0]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_parity_set_field_external() {
    // External SetField — value_or_ext_hash is a 32-byte hash,
    // ext_value_size is nonzero. The value-encoding branch writes "hash" /
    // "size" hash fields (not "value"), which exercises a different slot
    // in `encode_value_fields`.
    run_parity("parity-layer-set-field-ext", || OwnedDelta {
        set_fields: vec![SetField {
            section_id: 3,
            key_name: "mesh".to_owned(),
            value_or_ext_hash: vec![0x01; 32],
            ext_value_size: 1024,
            ext_value_index: 0,
            set_order: 2,
        }],
        ..Default::default()
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_parity_set_time_sample_inline() {
    run_parity("parity-layer-ts", || OwnedDelta {
        time_samples: vec![OodfTimeSample {
            section_id: 2,
            time: 12.5,
            value_or_ext_hash: vec![0xAA, 0xBB, 0xCC],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_parity_set_time_sample_external() {
    run_parity("parity-layer-ts-ext", || OwnedDelta {
        time_samples: vec![OodfTimeSample {
            section_id: 2,
            time: 99.25,
            value_or_ext_hash: vec![0x02; 32],
            ext_value_size: 4096,
            ext_value_index: 1,
        }],
        ..Default::default()
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_parity_empty_delta() {
    run_parity("parity-layer-empty", empty_delta).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_parity_multiple_fields_one_delta() {
    // Mixed: three SetFields in one delta. Every ARGV record needs to
    // byte-match the legacy execute_mutation output, and the ZADD ordering
    // under identical scores must not differ between backends.
    run_parity("parity-layer-multi-fields", || OwnedDelta {
        set_fields: vec![
            SetField {
                section_id: 1,
                key_name: "a".to_owned(),
                value_or_ext_hash: vec![0x01],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            },
            SetField {
                section_id: 1,
                key_name: "b".to_owned(),
                value_or_ext_hash: vec![0x02],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            },
            SetField {
                section_id: 2,
                key_name: "c".to_owned(),
                value_or_ext_hash: vec![0x03, 0x04],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            },
        ],
        ..Default::default()
    })
    .await;
}

// =============================================================================
// 1b. PERSISTED AUTHORING set_order (b.msb) — parity PLUS positive correctness.
//
// Task 1 persists a NEW per-field hash field, `authoring_set_order`, carrying
// the writer's TRUE per-delta authoring rank (`SetField.set_order`) verbatim,
// distinct from the cumulative `set_order` cursor (`next_set_order + delta
// set_order`). `dump_object_keyspace` already HGETALLs the whole per-field
// hash, so the field is compared automatically by the parity assertion. But
// parity alone is insufficient — two identically-wrong dumps would pass — so
// these cases ALSO assert the concrete persisted authoring value against the
// value the writer passed in the delta.
// =============================================================================

/// Same contract as `run_parity` (commit ONE delta through each backend and
/// assert byte-identical keyspaces), but returns the parity-verified,
/// normalised keyspace so a caller can layer positive correctness assertions
/// on top. Returns the Lua-backend dump (== the Txn dump by the assertion).
async fn run_parity_checked<F>(layer_name: &str, build_delta: F) -> ObjectKeyspace
where
    F: Fn() -> OwnedDelta,
{
    let tv_txn = TestValkey::setup().await;
    let ls_txn = parity_store(&tv_txn, UpdateLayerBackend::Txn);
    let sid_txn = create_test_session(&tv_txn).await;
    let oid_txn = create_test_layer(&ls_txn, &sid_txn, layer_name).await;
    ls_txn
        .update_layer_relayed(
            &sid_txn,
            layer_name,
            oid_txn,
            Some(build_delta()),
            "parity-client",
            42,
        )
        .await
        .expect("txn-backend update must succeed");
    let (ks_txn, clock_txn) = dump_object_keyspace(&tv_txn, oid_txn).await;
    let ks_txn = normalise_oid(ks_txn, oid_txn, &sid_txn);

    let tv_lua = TestValkey::setup().await;
    let ls_lua = parity_store(&tv_lua, UpdateLayerBackend::Lua);
    let sid_lua = create_test_session(&tv_lua).await;
    let oid_lua = create_test_layer(&ls_lua, &sid_lua, layer_name).await;
    ls_lua
        .update_layer_relayed(
            &sid_lua,
            layer_name,
            oid_lua,
            Some(build_delta()),
            "parity-client",
            42,
        )
        .await
        .expect("lua-backend update must succeed");
    let (ks_lua, clock_lua) = dump_object_keyspace(&tv_lua, oid_lua).await;
    let ks_lua = normalise_oid(ks_lua, oid_lua, &sid_lua);

    assert_eq!(
        ks_lua, ks_txn,
        "Lua-vs-Txn byte-parity violated.\n\nLUA: {ks_lua:#?}\n\nTXN: {ks_txn:#?}"
    );
    assert_commit_clock_parity(clock_txn, clock_lua);
    ks_lua
}

/// Multi-commit sibling of `run_parity_checked`: commit a SEQUENCE of deltas
/// (in order) through each backend before dumping. This lets `next_set_order`
/// advance well past any single delta's authoring rank, so the persisted
/// `authoring_set_order` and the cumulative `set_order` genuinely diverge.
/// Returns the parity-verified, normalised Lua-backend keyspace.
async fn run_parity_seq<F>(layer_name: &str, build_deltas: F) -> ObjectKeyspace
where
    F: Fn() -> Vec<OwnedDelta>,
{
    let tv_txn = TestValkey::setup().await;
    let ls_txn = parity_store(&tv_txn, UpdateLayerBackend::Txn);
    let sid_txn = create_test_session(&tv_txn).await;
    let oid_txn = create_test_layer(&ls_txn, &sid_txn, layer_name).await;
    for delta in build_deltas() {
        ls_txn
            .update_layer_relayed(
                &sid_txn,
                layer_name,
                oid_txn,
                Some(delta),
                "parity-client",
                42,
            )
            .await
            .expect("txn-backend update must succeed");
    }
    let (ks_txn, clock_txn) = dump_object_keyspace(&tv_txn, oid_txn).await;
    let ks_txn = normalise_oid(ks_txn, oid_txn, &sid_txn);

    let tv_lua = TestValkey::setup().await;
    let ls_lua = parity_store(&tv_lua, UpdateLayerBackend::Lua);
    let sid_lua = create_test_session(&tv_lua).await;
    let oid_lua = create_test_layer(&ls_lua, &sid_lua, layer_name).await;
    for delta in build_deltas() {
        ls_lua
            .update_layer_relayed(
                &sid_lua,
                layer_name,
                oid_lua,
                Some(delta),
                "parity-client",
                42,
            )
            .await
            .expect("lua-backend update must succeed");
    }
    let (ks_lua, clock_lua) = dump_object_keyspace(&tv_lua, oid_lua).await;
    let ks_lua = normalise_oid(ks_lua, oid_lua, &sid_lua);

    assert_eq!(
        ks_lua, ks_txn,
        "Lua-vs-Txn byte-parity violated across the commit sequence.\n\nLUA: {ks_lua:#?}\n\nTXN: {ks_txn:#?}"
    );
    assert_commit_clock_parity(clock_txn, clock_lua);
    ks_lua
}

/// Read the persisted `(authoring_set_order, set_order)` pair for the field
/// whose `key_name` matches, out of a dumped keyspace. The field member is
/// `"{sid}:{hex(key_name)}"` (see `dump_object_keyspace`), so we hex-decode the
/// portion after the first ':' before comparing. Panics with a descriptive
/// message if the field or either hash slot is absent — a missing
/// `authoring_set_order` means the engineer's write path regressed, and we
/// want that to fail loudly.
fn field_set_orders(ks: &ObjectKeyspace, key_name: &str) -> (u64, u64) {
    let (_member, snap) = ks
        .fields
        .iter()
        .find(|(member, _)| {
            member
                .split_once(':')
                .map(|(_, k)| decode_hex_key(k) == key_name)
                .unwrap_or(false)
        })
        .unwrap_or_else(|| {
            panic!(
                "field {key_name:?} not present in dumped keyspace: {:#?}",
                ks.fields
            )
        });
    let get = |slot: &str| -> u64 {
        snap.0
            .iter()
            .find(|(k, _)| k == slot)
            .unwrap_or_else(|| {
                panic!(
                    "field {key_name:?} missing hash slot {slot:?}: {:#?}",
                    snap.0
                )
            })
            .1
            .parse()
            .unwrap_or_else(|e| panic!("hash slot {slot:?} for {key_name:?} not a u64: {e}"))
    };
    (get("authoring_set_order"), get("set_order"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_parity_authoring_set_order_single() {
    // Single inline SetField with a non-zero authoring rank. On a fresh layer
    // (next_set_order == 0) the cumulative stored set_order equals the
    // authoring value for this FIRST commit, so this case pins the plumbing:
    // both backends persist `authoring_set_order` byte-identically AND the
    // stored value equals the writer's `set_order` (7).
    let ks = run_parity_checked("parity-layer-authoring-single", || {
        delta_with_set_field_inline(4, "opacity", vec![0x11, 0x22], 7)
    })
    .await;
    let (authoring, cumulative) = field_set_orders(&ks, "opacity");
    assert_eq!(
        authoring, 7,
        "persisted authoring_set_order must equal the writer's per-delta set_order"
    );
    // First commit on a fresh layer: cumulative == authoring (documents the
    // baseline the multi-commit case then forces apart).
    assert_eq!(
        cumulative, 7,
        "first-commit stored set_order == next_set_order(0) + 7"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_parity_authoring_set_order_burst_distinct() {
    // Three SetFields in ONE delta with distinct authoring ranks 0/5/2. Within
    // a single delta the running cursor advances per applied field, so the
    // cumulative stored set_orders end up 0/6/9 — but the persisted
    // authoring_set_order MUST stay the writer's verbatim 0/5/2, proving the
    // field is the raw per-delta rank and not the running-cursor value.
    let ks = run_parity_checked("parity-layer-authoring-burst", || OwnedDelta {
        set_fields: vec![
            SetField {
                section_id: 1,
                key_name: "a".to_owned(),
                value_or_ext_hash: vec![0x01],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            },
            SetField {
                section_id: 1,
                key_name: "b".to_owned(),
                value_or_ext_hash: vec![0x02],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 5,
            },
            SetField {
                section_id: 2,
                key_name: "c".to_owned(),
                value_or_ext_hash: vec![0x03, 0x04],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 2,
            },
        ],
        ..Default::default()
    })
    .await;

    let (auth_a, cum_a) = field_set_orders(&ks, "a");
    let (auth_b, cum_b) = field_set_orders(&ks, "b");
    let (auth_c, cum_c) = field_set_orders(&ks, "c");

    // Authoring rank persisted verbatim, unaffected by the running cursor.
    assert_eq!(
        (auth_a, auth_b, auth_c),
        (0, 5, 2),
        "authoring ranks must be stored verbatim"
    );
    // Cumulative cursor: a=0, b=0+1+5=6, c=6+1+2=9. Distinct from authoring for
    // b and c — so a both-paths-wrong regression that stored the cursor into
    // authoring_set_order would fail here rather than passing on parity alone.
    assert_eq!(
        (cum_a, cum_b, cum_c),
        (0, 6, 9),
        "cumulative set_order folds in the running cursor"
    );
    assert_ne!(
        auth_b, cum_b,
        "authoring rank must diverge from cumulative cursor for field b"
    );
    assert_ne!(
        auth_c, cum_c,
        "authoring rank must diverge from cumulative cursor for field c"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_parity_authoring_set_order_external() {
    // External SetField (hash + size branch of the value encoder) with a
    // non-zero authoring rank. Exercises the value-hash-fields external branch
    // AND the new authoring_set_order slot together.
    let ks = run_parity_checked("parity-layer-authoring-ext", || OwnedDelta {
        set_fields: vec![SetField {
            section_id: 3,
            key_name: "mesh".to_owned(),
            value_or_ext_hash: vec![0x07; 32],
            ext_value_size: 2048,
            ext_value_index: 0,
            set_order: 9,
        }],
        ..Default::default()
    })
    .await;
    let (authoring, cumulative) = field_set_orders(&ks, "mesh");
    assert_eq!(
        authoring, 9,
        "external SetField authoring_set_order must equal the writer's set_order"
    );
    assert_eq!(
        cumulative, 9,
        "first-commit external field: cumulative == authoring"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_parity_authoring_set_order_multi_commit() {
    // Multi-commit: repeatedly overwrite the SAME field so next_set_order
    // advances well past any single delta's authoring rank. After the last
    // commit the persisted authoring_set_order must equal the LAST writing
    // delta's set_order (1), while the cumulative stored set_order has diverged
    // to a much larger running-cursor value — and both backends still agree.
    //
    //   next_set_order starts 0.
    //   d1 set_order=3 -> stored=0+3=3,  next=4
    //   d2 set_order=4 -> stored=4+4=8,  next=9
    //   d3 set_order=1 -> stored=9+1=10, next=11
    let ks = run_parity_seq("parity-layer-authoring-seq", || {
        vec![
            delta_with_set_field_inline(5, "x", vec![0xA0], 3),
            delta_with_set_field_inline(5, "x", vec![0xA1], 4),
            delta_with_set_field_inline(5, "x", vec![0xA2], 1),
        ]
    })
    .await;
    let (authoring, cumulative) = field_set_orders(&ks, "x");
    assert_eq!(
        authoring, 1,
        "authoring_set_order must equal the LAST writing delta's per-delta set_order"
    );
    assert_eq!(
        cumulative, 10,
        "cumulative set_order advanced with the running cursor"
    );
    assert_ne!(
        authoring, cumulative,
        "after a multi-commit sequence the authoring rank must diverge from the cumulative cursor"
    );
}

// =============================================================================
// 1c. MIXED-RECORD PARITY — one delta carrying BOTH SetFields and time samples.
//
// Every parity case above is single-tag: all-SetField or all-TimeSample. b.ph9
// split the script into two independent ARGV walks (a validation pre-pass that
// issues zero `redis.call`s, and an apply pass that consumes the pre-pass's
// stashed `stored_set_order` values), and BOTH advance their cursor with
// per-tag arithmetic: an `SF` record spans `1 + 2n + 2` slots (tag, hfield
// count, n name/value pairs, delta_set_order, zadd_member) while an `ST`
// record spans `1 + 2n + 1` (no delta_set_order).
//
// A single-tag delta cannot detect a one-slot error in either stride. With one
// record there is nothing after it to mis-read; with N records of the SAME tag
// a wrong stride is uniformly wrong and the existing cases already cover it.
// The shape that does detect it is a record stream that CROSSES the tag
// boundary and still has a record after the crossing. `encode_delta` emits
// every SetField before every time sample, so the crossing reachable from a
// single `OwnedDelta` is `SF … SF ST … ST` — two of each therefore drives the
// SF→SF, SF→ST and ST→ST transitions in one commit. (ST→SF is not reachable
// through `OwnedDelta`; it is driven at the ARGV level in section 13.)
//
// A drift here is silently corrupting rather than loudly failing: the apply
// pass would HSET a field hash using the neighbouring record's zadd_member as
// a hash-field NAME, or read a `stored_set_order` stashed against the wrong
// record ordinal. The txn backend shares no code with the script, so parity
// against it is the oracle; the positive assertions below then rule out the
// both-paths-wrong case parity alone would pass.
// =============================================================================

/// Section every mixed-record fixture writes into. One section keeps the dump
/// small — the record TAGS, not the sections, are what is under test.
const MIXED_SECTION_ID: u64 = 6;
/// The two sample times in the mixed delta. Distinct so they land on distinct
/// time-samples-index members.
const MIXED_TS_INLINE_TIME: f64 = 1.5;
const MIXED_TS_EXTERNAL_TIME: f64 = 2.5;
/// `create_test_layer` seeds `seq = 1`, so the single commit under test is
/// sequence 2 — the value the script stamps into every touched record's
/// `version` and into both index scores.
const MIXED_COMMIT_SEQ: &str = "2";

/// One delta whose encoded record stream is `SF SF ST ST`, with one external
/// value on each side of the tag boundary (`blob` and the sample at
/// [`MIXED_TS_EXTERNAL_TIME`]) so the two-hash-field `hash`/`size` variant of
/// the value-encoding branch is walked as well as the one-hash-field `value`
/// variant. The authoring ranks (3 and 2) are distinct and non-zero so the
/// per-record `set_order` fold is observable rather than degenerate.
fn mixed_fields_and_time_samples_delta() -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![
            SetField {
                section_id: MIXED_SECTION_ID,
                key_name: "alpha".to_owned(),
                value_or_ext_hash: vec![0x0A, 0x0B],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 3,
            },
            SetField {
                section_id: MIXED_SECTION_ID,
                key_name: "blob".to_owned(),
                value_or_ext_hash: vec![0x5A; 32],
                ext_value_size: 8192,
                ext_value_index: 0,
                set_order: 2,
            },
        ],
        time_samples: vec![
            OodfTimeSample {
                section_id: MIXED_SECTION_ID,
                time: MIXED_TS_INLINE_TIME,
                value_or_ext_hash: vec![0xC0, 0xFF, 0xEE],
                ext_value_size: 0,
                ext_value_index: 0,
            },
            OodfTimeSample {
                section_id: MIXED_SECTION_ID,
                time: MIXED_TS_EXTERNAL_TIME,
                value_or_ext_hash: vec![0x7B; 32],
                ext_value_size: 65536,
                ext_value_index: 1,
            },
        ],
        ..Default::default()
    }
}

/// Read the persisted hash slots for the time sample at `(section_id, time)`
/// out of a dumped keyspace.
///
/// The dump indexes time samples by `KeySchema::time_sample_member` — exactly
/// the string the encoder emitted as that record's ZADD member — so a lookup
/// that misses is itself the symptom of a cursor drift: the apply pass ZADDed
/// some other ARGV slot's contents as the member.
fn time_sample_slots(ks: &ObjectKeyspace, section_id: u64, time: f64) -> &HashSnapshot {
    let member = KeySchema::time_sample_member(&section_id.to_string(), time);
    &ks.time_samples
        .iter()
        .find(|(m, _)| *m == member)
        .unwrap_or_else(|| {
            panic!(
                "time sample {member:?} not present in dumped keyspace: {:#?}",
                ks.time_samples
            )
        })
        .1
}

fn slot(name: &str, value: impl Into<String>) -> (String, String) {
    (name.to_owned(), value.into())
}

/// Lowercase hex of `len` repetitions of `byte` — the shape
/// `lua::value_hash_fields` produces for an external value's 32-byte hash.
fn hex_repeat(byte: u8, len: usize) -> String {
    std::iter::repeat_n(format!("{byte:02x}"), len).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_parity_mixed_fields_and_time_samples_one_delta() {
    let ks = run_parity_checked(
        "parity-layer-mixed-records",
        mixed_fields_and_time_samples_delta,
    )
    .await;

    // ── SetField side of the boundary. next_set_order starts 0:
    //    alpha  set_order=3 -> stored = 0 + 3 = 3, cursor -> 4
    //    blob   set_order=2 -> stored = 4 + 2 = 6, cursor -> 7
    // The cumulative values diverge from the authoring ranks, so a regression
    // that stashed the cursor into `authoring_set_order` (or read the wrong
    // record's stash) cannot pass by coincidence.
    assert_eq!(
        field_set_orders(&ks, "alpha"),
        (3, 3),
        "first SetField: authoring rank verbatim, cumulative folded from cursor 0"
    );
    assert_eq!(
        field_set_orders(&ks, "blob"),
        (2, 6),
        "second SetField: authoring rank 2 verbatim, cumulative 4 + 2 = 6"
    );

    // ── Time-sample side. These are the records the SF→ST crossing lands on:
    // a one-slot drift in the SF stride would have the ST record read `alpha`'s
    // zadd_member as its hash-field count, and a drift in the ST stride would
    // corrupt the SECOND sample (which is exactly why there are two).
    assert_eq!(
        time_sample_slots(&ks, MIXED_SECTION_ID, MIXED_TS_INLINE_TIME),
        &HashSnapshot(vec![
            slot("value", BASE64.encode([0xC0, 0xFF, 0xEE])),
            slot("version", MIXED_COMMIT_SEQ),
        ]),
        "inline time sample must carry exactly its base64 value plus the commit version — \
         no `set_order` slot (time samples have no authoring rank) and no stray slot picked \
         up from a neighbouring record"
    );
    assert_eq!(
        time_sample_slots(&ks, MIXED_SECTION_ID, MIXED_TS_EXTERNAL_TIME),
        &HashSnapshot(vec![
            slot("hash", hex_repeat(0x7B, 32)),
            slot("size", "65536"),
            slot("version", MIXED_COMMIT_SEQ),
        ]),
        "external time sample must carry the hex hash + size pair plus the commit version"
    );

    // ── Both indexes scored at the commit sequence, and each carries exactly
    // the members its own tag contributed — the cross-index confusion a drifted
    // cursor would produce (a time sample ZADDed into the fields index) shows
    // up here.
    let expect_seq: i64 = MIXED_COMMIT_SEQ.parse().unwrap();
    assert_eq!(
        ks.fields_index,
        vec![
            (
                KeySchema::field_member(&MIXED_SECTION_ID.to_string(), "alpha"),
                expect_seq
            ),
            (
                KeySchema::field_member(&MIXED_SECTION_ID.to_string(), "blob"),
                expect_seq
            ),
        ],
        "the fields index must hold exactly the two SetField members, scored at the commit seq"
    );
    assert_eq!(
        ks.time_samples_index,
        vec![
            (
                KeySchema::time_sample_member(&MIXED_SECTION_ID.to_string(), MIXED_TS_INLINE_TIME),
                expect_seq
            ),
            (
                KeySchema::time_sample_member(
                    &MIXED_SECTION_ID.to_string(),
                    MIXED_TS_EXTERNAL_TIME
                ),
                expect_seq
            ),
        ],
        "the time-samples index must hold exactly the two ST members, scored at the commit seq"
    );
}

// =============================================================================
// 2. SINGLE-RTT PROOF
//
// N=64 concurrent writers × 30 SetField updates each = 1920 commits. All must
// land first-try under the Lua backend — the attempts histogram sample_sum
// equals sample_count and 0 writers surface ABORTED.
// =============================================================================

/// `path` label of a commit on the Lua field fast path.
const FAST_PATH: &str = "fast";
/// `path` label of a commit on the WATCH slow path — every delta under the
/// `txn` backend.
const WATCH_PATH: &str = "watch";

/// The `path` a field-only delta commits on under `backend`: the Lua fast path
/// under `lua` and `lua-watch-structural`, the WATCH slow path under `txn`.
fn field_delta_path(backend: UpdateLayerBackend) -> &'static str {
    match backend {
        UpdateLayerBackend::Lua | UpdateLayerBackend::LuaWatchStructural => FAST_PATH,
        UpdateLayerBackend::Txn => WATCH_PATH,
    }
}

/// `live_edit_update_layer_attempts_histogram{path}`'s (sample_count,
/// sample_sum).
fn attempts_histogram(metrics: &Metrics, path: &str) -> (u64, f64) {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    let text = render(metrics);
    let count = sample_value(
        &text,
        "live_edit_update_layer_attempts_histogram_count",
        &[("path", path)],
    )
    .unwrap_or(0.0) as u64;
    let sum = sample_value(
        &text,
        "live_edit_update_layer_attempts_histogram_sum",
        &[("path", path)],
    )
    .unwrap_or(0.0);
    (count, sum)
}

/// `live_edit_update_layer_retries_total{path,outcome}`.
fn retry_counter(metrics: &Metrics, path: &str, outcome: &str) -> u64 {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    sample_value(
        &render(metrics),
        "live_edit_update_layer_retries_total",
        &[("path", path), ("outcome", outcome)],
    )
    .unwrap_or(0.0) as u64
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_lua_backend_n64_all_first_try_no_aborts() {
    const WRITERS: usize = 64;
    const UPDATES_PER_WRITER: usize = 30;
    const TOTAL: u64 = (WRITERS * UPDATES_PER_WRITER) as u64;

    let tv = TestValkey::setup_with_pool_size(WRITERS as u32).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("b67u-lua-n64"));
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua).with_metrics(metrics.clone());
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "b67u-lua-n64-layer").await;

    let barrier = Arc::new(tokio::sync::Barrier::new(WRITERS));

    let mut handles = Vec::with_capacity(WRITERS);
    for w in 0..WRITERS {
        let ls_i = ls.clone();
        let sid_i = sid.clone();
        let barrier_i = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier_i.wait().await;
            let mut oks = 0u32;
            let mut aborts = 0u32;
            let mut others: Vec<String> = Vec::new();
            for u in 0..UPDATES_PER_WRITER {
                let delta = delta_with_set_field_inline(
                    1,
                    &format!("w{w}-u{u}"),
                    vec![w as u8, u as u8],
                    0,
                );
                let client_id = format!("w{w}");
                match ls_i
                    .update_layer_relayed(
                        &sid_i,
                        "b67u-lua-n64-layer",
                        oid,
                        Some(delta),
                        &client_id,
                        0,
                    )
                    .await
                {
                    Ok(_) => oks += 1,
                    Err(ServiceError::Aborted(msg)) => {
                        aborts += 1;
                        others.push(format!("ABORTED: {msg}"));
                    }
                    Err(e) => others.push(format!("{e:?}")),
                }
            }
            (oks, aborts, others)
        }));
    }

    let mut total_ok = 0u32;
    let mut total_abort = 0u32;
    let mut errs: Vec<String> = Vec::new();
    for h in handles {
        let (o, a, e) = h.await.expect("writer task must not panic");
        total_ok += o;
        total_abort += a;
        errs.extend(e);
    }

    assert_eq!(
        total_abort, 0,
        "Lua backend must never ABORT — atomic by construction. Got {total_abort} aborts.\n\
         First few errors: {errs:#?}"
    );
    assert_eq!(
        total_ok as u64, TOTAL,
        "every writer must commit (total={TOTAL}); got {total_ok} ok + {total_abort} abort. \
         Errors: {errs:#?}"
    );

    // Attempts histogram: one sample per commit, value == 1.0 for every one.
    let (sample_count, sample_sum) = attempts_histogram(&metrics, FAST_PATH);
    assert_eq!(
        sample_count, TOTAL,
        "histogram must record one sample per commit; got {sample_count}, expected {TOTAL}"
    );
    // Lua is atomic-by-construction — WATCH contention cannot happen, so
    // `sample_sum - sample_count` counts only injected I/O-driven retries.
    // No fault is injected in this test, so the observed mean MUST be ~1.0
    // exactly. Tightened from `< 1.01` to `< 1.001` (b.ohv item 9) so a
    // regression that re-introduced a Lua-script `redis.error_reply` on
    // benign paths (mis-classified as retriable by the caller) shows up
    // as `mean ≈ 2.0` and fails the assertion instead of slipping under
    // the old 1% slop budget. The companion test
    // `update_layer_lua_backend_one_injected_retry_recovers_on_attempt_two`
    // exercises the retry path explicitly via fault injection.
    let mean = sample_sum / sample_count as f64;
    assert!(
        mean < 1.001,
        "Lua backend must commit on attempt 1 (no WATCH conflict possible); \
         observed mean attempts = {mean:.6} over {sample_count} commits (sum = {sample_sum}). \
         CLAUDE.md spec: Lua aborts should drop to zero — a mean > 1.001 indicates a \
         retry path is firing on a benign code path."
    );

    assert_eq!(
        retry_counter(&metrics, FAST_PATH, "exhausted"),
        0,
        "Lua backend must never exhaust retry budget — no WATCH conflict can fire"
    );
    let committed = retry_counter(&metrics, FAST_PATH, "committed");
    // Tightened from `< 1.0%` to `== 0` (b.ohv item 9). With no fault
    // injection the expected value is exactly 0; any positive count
    // indicates a real regression in the Lua path.
    assert_eq!(
        committed, 0,
        "Lua backend must produce zero `committed` retries under non-faulty conditions; \
         got {committed} retries over {TOTAL} commits. The 1% slop the test used to \
         tolerate predated the b.ohv tightening — Lua aborts should drop to zero."
    );
}

// =============================================================================
// 2b. b.ohv item 9 — fault-injection sibling for the n64 retry test above.
//
// Pre-b.ohv the n64 test allowed up to 1% of commits to record `attempts > 1`
// as a slop budget for "CI pathological timing". That budget was vacuous in
// practice — Lua is atomic on the server, so retries fire only on injected
// transient errors. Without an explicit fault-injection test we had no
// coverage of the retry path on the Lua backend at all.
//
// This sibling injects one transient `Aborted("transient I/O: ...")` on
// attempt 1; attempt 2 runs the real Lua script and commits. Asserts:
//   * `update_layer` returns Ok (recovered).
//   * The attempts-histogram observation for this commit is 2.0.
//   * `retries_total{committed} == 1`.
//   * The injected queue is fully drained.
// =============================================================================

#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_backend_one_injected_retry_recovers_on_attempt_two() {
    use std::collections::VecDeque;
    use std::sync::Mutex as StdMutex;

    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("bohv-lua-retry"));

    // Cooked Aborted in the exact shape `map_transient_fred_err` produces
    // for a transient fred kind — same shape the b.k1r exec-arm test uses,
    // so the Lua retry loop matches and retries identically.
    let timeout_label = fred::error::ErrorKind::Timeout.to_str();
    let cooked = ServiceError::Aborted(format!(
        "transient I/O: EVALSHA failed ({timeout_label}): synthetic Lua-path transient drop"
    ));

    let queue: Arc<StdMutex<VecDeque<ServiceError>>> =
        Arc::new(StdMutex::new(std::iter::once(cooked).collect()));

    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua)
        .with_metrics(metrics.clone())
        .with_fault_injector_for_test(queue.clone());

    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bohv-lua-retry-layer").await;

    // One field write — fast-path eligible, so the Lua dispatch fires.
    let delta = delta_with_set_field_inline(1, "k", vec![0x01], 0);
    let (seq, _ts) = ls
        .update_layer_relayed(&sid, "bohv-lua-retry-layer", oid, Some(delta), "c", 0)
        .await
        .expect(
            "Lua-backend update_layer must recover from a single transient Aborted and \
             commit on the second attempt. A panic here means the retry loop never \
             matched the injected Aborted (or attempt 2 ran into a real error).",
        );
    assert!(
        seq > 1,
        "seq must advance past CreateLayer's initial 1; got {seq}"
    );

    assert_eq!(
        queue.lock().unwrap().len(),
        0,
        "fault injector queue must be drained by exactly one attempt"
    );

    // committed retry counter: commit on attempt 2 → 1 retry.
    let committed = retry_counter(&metrics, FAST_PATH, "committed");
    assert_eq!(
        committed, 1,
        "one transient-Aborted retry must count as exactly one `committed` retry; got {committed}"
    );
    assert_eq!(
        retry_counter(&metrics, FAST_PATH, "exhausted"),
        0,
        "single retry must not exhaust the budget"
    );

    // Attempts histogram: exactly one sample with value 2.0 (commit on attempt 2).
    let (sample_count, sample_sum) = attempts_histogram(&metrics, FAST_PATH);
    assert_eq!(
        sample_count, 1,
        "exactly one commit ran in this test; got {sample_count} histogram samples"
    );
    assert!(
        (sample_sum - 2.0).abs() < f64::EPSILON,
        "the lone sample must record attempts=2 (one retry, one commit); sum={sample_sum}"
    );
}

// =============================================================================
// 3. HEADER-ANOMALY MAPPING
// =============================================================================

/// A `DEL`ed `obj:hdr` — the object is *absent* — must surface as
/// `NotFound(EMPTY_HEADER_SENTINEL_MSG)`.
///
/// b.f1v flipped this from `FailedPrecondition`. The variant is the whole
/// point: `grpc::layer::handle_update_layer` keys its single checkpoint-recovery
/// arm on `ServiceError::NotFound`, so a `FailedPrecondition` here means a
/// post-wipe UpdateLayer never attempts recovery and the client is told
/// "precondition failed" for a layer the service could have restored. The
/// message is asserted by equality (not `contains`) so the txn path, the Lua
/// path and `disambiguate_stream_collision` cannot drift apart.
///
/// This test is one corner of a four-test claim — **an ABSENT header and a
/// PARTIAL one are distinct surfaces, identically on both backends**. The other
/// three: `update_layer_transaction_test::update_layer_header_deleted_returns_not_found`
/// (absent / txn), `b_k1r_tests::update_layer_lua_partial_header_returns_failed_precondition`
/// (partial / Lua) and `b_qw6_tests::update_layer_txn_partial_header_returns_failed_precondition`
/// (partial / txn). Each carries the opposite-variant `panic!` arm, so the split
/// fails loudly from whichever corner regresses.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_missing_header_returns_not_found() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "lua-missing-header").await;

    // Fast-path eligibility is a PRECONDITION of this test, not an accident of
    // the fixture. `update_layer_once`'s dispatch is
    // `backend == Lua && is_fast_path(&delta)`, so an ineligible delta would
    // quietly run this down the WATCH/MULTI/EXEC path — and since b.f1v made
    // both backends emit the identical message, it would still pass, leaving
    // the Lua script's `NOT_FOUND:` reply covered by nothing.
    let delta = delta_with_set_field_inline(1, "k", vec![0x01], 0);
    assert!(
        is_fast_path(&Some(delta.clone())),
        "this delta must be fast-path eligible, or this test silently re-tests the txn path's \
         guard under a `_lua` name"
    );

    // DEL the header so the Lua script's own `EXISTS`/HGETALL check fires.
    // There is no Rust-side header pre-read on the fast path — the script is
    // the sole guard, and its `NOT_FOUND:` reply is what the dispatch maps.
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let _: () = tv
        .client()
        .pool
        .del(&header_key)
        .await
        .expect("DEL header must succeed");

    let result = ls
        .update_layer_relayed(
            &sid,
            "lua-missing-header",
            oid,
            Some(delta),
            "client-x",
            123,
        )
        .await;

    let err = result.expect_err("update against deleted header must fail");
    match err {
        ServiceError::NotFound(msg) => {
            assert_eq!(
                msg, EMPTY_HEADER_SENTINEL_MSG,
                "the Lua path's absent-header surface must carry the empty-header sentinel \
                 byte-for-byte — the txn path's empty-HGETALL guard and \
                 `disambiguate_stream_collision` raise the same string, and the SDD §7.5 \
                 parity contract is asserted on the message text; got {msg:?}"
            );
        }
        ServiceError::Aborted(msg) => panic!(
            "Lua NOT_FOUND must map to a non-retriable status — not Aborted (clients rely on \
             this distinction to surface 'object gone' vs 'retry conflict'). Got Aborted({msg})"
        ),
        ServiceError::FailedPrecondition(msg) => panic!(
            "b.f1v: an ABSENT header must surface as NotFound, not FailedPrecondition — the \
             gRPC handler's single checkpoint-recovery arm matches on the NotFound VARIANT, so \
             this shape silently disarms post-wipe recovery. Got FailedPrecondition({msg:?})"
        ),
        other => panic!("unexpected error from missing-header Lua update: {other:?}"),
    }
}

// =============================================================================
// 4. NOSCRIPT RECOVERY — SCRIPT FLUSH between two Lua-backend updates.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_recovers_from_script_flush_via_evalsha_reload() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "lua-noscript-recovery").await;

    // First update loads the script via the first evalsha_once_with_reload
    // call and commits normally.
    ls.update_layer_relayed(
        &sid,
        "lua-noscript-recovery",
        oid,
        Some(delta_with_set_field_inline(1, "before", vec![0xAA], 0)),
        "client-a",
        1000,
    )
    .await
    .expect("first Lua update must succeed");

    // Flush the script cache. `script_flush(false)` is synchronous on the
    // server so the effect is observable immediately for the next EVALSHA.
    let _: () = tv
        .client()
        .pool
        .next()
        .script_flush(false)
        .await
        .expect("SCRIPT FLUSH must succeed");

    // Second update — the cached SHA on the server is gone. The fast path's
    // `storage::lua::evalsha_once_with_reload` must catch NOSCRIPT, re-send
    // SCRIPT LOAD, retry EVALSHA once, and commit. A regression (e.g. plain `evalsha` without the
    // reload fallback) would surface as a fred error classified by
    // map_snapshot_load_err as Internal and the `.expect(...)` below would
    // panic.
    let (seq, _ts) = ls
        .update_layer_relayed(
            &sid,
            "lua-noscript-recovery",
            oid,
            Some(delta_with_set_field_inline(1, "after", vec![0xBB], 0)),
            "client-b",
            2000,
        )
        .await
        .expect(
            "second Lua update must recover from SCRIPT FLUSH via \
             storage::lua::evalsha_once_with_reload. If this panics with a fred error, the \
             Rust caller used plain EVALSHA instead of evalsha_once_with_reload and is not \
             handling NOSCRIPT correctly.",
        );
    assert!(seq > 0, "seq must advance on recovered update");
}

// =============================================================================
// 5. BACKEND TOGGLE — pin `UpdateLayerBackend::Txn` and verify legacy path.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_backend_explicitly_selected_still_commits() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("b67u-txn-toggle"));
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Txn).with_metrics(metrics.clone());
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "txn-toggle-layer").await;

    for i in 0..2u64 {
        let (seq, _ts) = ls
            .update_layer_relayed(
                &sid,
                "txn-toggle-layer",
                oid,
                Some(delta_with_set_field_inline(
                    1,
                    &format!("k{i}"),
                    vec![i as u8],
                    0,
                )),
                "client-txn",
                1000 + i,
            )
            .await
            .expect("txn-backend update must succeed");
        assert_eq!(seq, 2 + i, "txn path must produce contiguous seqs");
    }

    // Histogram: two first-try commits => sample_count == 2, sample_sum == 2.
    let (count, sum) = attempts_histogram(&metrics, WATCH_PATH);
    assert_eq!(count, 2, "two commits must record two histogram samples");
    assert!(
        (sum - 2.0).abs() < f64::EPSILON,
        "two first-try commits under Txn must sum to 2.0; got {sum}"
    );
    assert_eq!(retry_counter(&metrics, WATCH_PATH, "committed"), 0);
    assert_eq!(retry_counter(&metrics, WATCH_PATH, "exhausted"), 0);

    assert_eq!(
        ls.semaphore_count(),
        1,
        "b.3gx admission-gate entry must exist after txn-backend commits"
    );
}

// Stronger toggle discriminator, available only on the fault-injection build:
// the `with_fred_error_injector_for_test` seam lives inside the Txn branch, so
// if an injected fred Timeout causes a retry-then-commit under a `Txn`-pinned
// store, we know the Txn branch actually executed (the Lua branch would leave
// the injected error in the queue and the test's drain assertion would fail).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_backend_txn_executes_watch_branch_confirmed_via_fred_injector() {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(
        "b67u-txn-branch-discriminator",
    ));

    let queue: Arc<Mutex<VecDeque<fred::error::Error>>> = Arc::new(Mutex::new(
        vec![fred::error::Error::new(
            fred::error::ErrorKind::Timeout,
            "synthetic discriminator",
        )]
        .into_iter()
        .collect(),
    ));

    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Txn)
        .with_metrics(metrics.clone())
        .with_fred_error_injector_for_test(queue.clone());

    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "txn-discriminator").await;

    ls.update_layer_relayed(
        &sid,
        "txn-discriminator",
        oid,
        Some(delta_with_set_field_inline(1, "k", vec![0x55], 0)),
        "c",
        1,
    )
    .await
    .expect("txn-backend update must recover from injected fred Timeout");

    assert_eq!(
        queue.lock().unwrap().len(),
        0,
        "fred-error injector must have fired on the Txn branch (Lua wouldn't consume it)"
    );
    assert_eq!(retry_counter(&metrics, WATCH_PATH, "committed"), 1);
}

// =============================================================================
// 6. b.3gx admission gate under the Lua backend.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn layer_store_semaphore_map_populated_and_drained_under_lua_backend() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "b3gx-under-lua").await;

    assert_eq!(ls.semaphore_count(), 0, "baseline");

    ls.update_layer_relayed(
        &sid,
        "b3gx-under-lua",
        oid,
        Some(delta_with_set_field_inline(1, "k", vec![0x01], 0)),
        "c",
        1,
    )
    .await
    .expect("lua-backend update must commit");

    assert_eq!(
        ls.semaphore_count(),
        1,
        "b.3gx admission-gate must populate the map on Lua-backend commits too"
    );

    ls.delete_layer(&sid, "b3gx-under-lua", mock_ext_store())
        .await
        .expect("delete_layer must succeed");

    assert_eq!(
        ls.semaphore_count(),
        0,
        "b.3gx delete_layer-cleanup must drain the map regardless of backend"
    );
}

// =============================================================================
// 7. Sequence-num contiguity under Lua N-concurrent writers.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_lua_concurrent_writers_produce_contiguous_seqs_and_no_conflicts() {
    const WRITERS: usize = 20;

    let tv = TestValkey::setup_with_pool_size(WRITERS as u32 + 4).await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "lua-contiguous").await;
    let barrier = Arc::new(tokio::sync::Barrier::new(WRITERS));

    let mut handles = Vec::with_capacity(WRITERS);
    for i in 0..WRITERS {
        let ls_i = ls.clone();
        let sid_i = sid.clone();
        let barrier_i = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier_i.wait().await;
            ls_i.update_layer_relayed(
                &sid_i,
                "lua-contiguous",
                oid,
                Some(delta_with_set_field_inline(
                    1,
                    &format!("k{i}"),
                    vec![i as u8],
                    0,
                )),
                &format!("w{i}"),
                1000 + i as u64,
            )
            .await
        }));
    }

    let mut successes: Vec<u64> = Vec::new();
    for h in handles {
        match h.await.expect("task must not panic") {
            Ok((seq, _)) => successes.push(seq),
            Err(ServiceError::Aborted(msg)) => {
                panic!("Lua backend must never ABORT under contention; got: {msg}")
            }
            Err(e) => panic!("unexpected error under Lua contention: {e:?}"),
        }
    }

    let mut unique: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for seq in &successes {
        assert!(
            unique.insert(*seq),
            "duplicate seq={seq} across Lua writers: {successes:?}"
        );
    }
    let mut sorted = successes.clone();
    sorted.sort();
    for (i, &seq) in sorted.iter().enumerate() {
        assert_eq!(
            seq,
            2 + i as u64,
            "Lua-backend commits must produce contiguous seqs from 2; \
             sorted={sorted:?}, position {i} got {seq}"
        );
    }

    let stream_key = tv.key_schema().layer_stream(oid);
    // Fields read as bytes: an entry's relay `ompf` field is binary.
    let entries: Vec<(String, HashMap<String, bytes::Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE must succeed");
    assert_eq!(
        entries.len(),
        1 + successes.len(),
        "Lua stream XLEN must equal 1 + {}; entries={entries:?}",
        successes.len()
    );
}

// =============================================================================
// 8. src_fmt preservation (engineer's self-noted edge case).
//
// The Lua HSET writes src_fmt from the planner's state. apply_delta on a
// fast-path delta does NOT modify state.source_format. A regression that
// initialised the planner state with an empty src_fmt would wipe the header
// field on every fast-path commit.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_preserves_src_fmt_across_fast_path_commits() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "lua-srcfmt").await;

    // Baseline: create_layer seeded src_fmt=usd.
    let header_before: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().obj_header(&oid.to_string()))
        .await
        .expect("HGETALL must succeed");
    assert_eq!(
        header_before.get("src_fmt").map(String::as_str),
        Some("usd"),
        "setup failed: create_layer must seed src_fmt=usd"
    );

    // Run three Lua-backend updates.
    for i in 0..3u64 {
        ls.update_layer_relayed(
            &sid,
            "lua-srcfmt",
            oid,
            Some(delta_with_set_field_inline(
                1,
                &format!("k{i}"),
                vec![i as u8],
                0,
            )),
            "c",
            i,
        )
        .await
        .expect("lua update must commit");
    }

    let header_after: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().obj_header(&oid.to_string()))
        .await
        .expect("HGETALL must succeed");
    assert_eq!(
        header_after.get("src_fmt").map(String::as_str),
        Some("usd"),
        "Lua backend must preserve src_fmt across fast-path commits. \
         Got header={header_after:?}"
    );
}

// =============================================================================
// 9. b.1dz fault-injection analog for the Lua path.
//
// The pre-b.67u b.1dz regression tests in `update_layer_retry_test.rs` inject
// a raw `fred::error::Error` into the WATCH call site via
// `with_fred_error_injector_for_test` to prove the transient-vs-terminal
// classifier (`map_snapshot_load_err`) routes transient errors into the
// retry loop and non-transient errors into Internal. The Lua path has its
// own call site (`EVALSHA update_layer_fast`); engineer-67u added
// `with_lua_fred_error_injector_for_test` to inject a raw `fred::error::Error`
// there and pipe it through the same `map_snapshot_load_err("EVALSHA
// update_layer_fast", ...)` call that a real EVALSHA failure would go
// through. These two tests lock in that the Lua path's classifier wiring
// matches the Txn path's — a regression (e.g. swapping `Internal` for
// `Aborted` on a non-transient error, or vice versa) shows up here.
//
// Gated behind `#[cfg(feature = "test-fault-injection")]` just like the
// matching seam on `LayerStore`.
// =============================================================================

#[cfg(feature = "test-fault-injection")]
mod lua_fault_injection {
    use super::*;
    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    fn synthetic_fred_err(kind: FredErrorKind) -> FredError {
        FredError::new(kind, "synthetic test error")
    }

    fn fred_injector(errs: Vec<FredError>) -> Arc<Mutex<VecDeque<FredError>>> {
        Arc::new(Mutex::new(errs.into_iter().collect()))
    }

    /// A fred Timeout on the first EVALSHA must classify as transient, retry,
    /// and commit on attempt 2. PRE-FIX reasoning: if `map_snapshot_load_err`
    /// mis-classified the EVALSHA transient kind as `Internal`, the retry
    /// loop would not match and the `.expect(...)` would panic.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_lua_fred_timeout_on_first_eval_flows_through_retry_loop() {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("b67u-lua-timeout-chain"));

        let queue = fred_injector(vec![synthetic_fred_err(FredErrorKind::Timeout)]);

        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_retry_max_attempts(32)
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .with_metrics(metrics.clone())
            .with_lua_fred_error_injector_for_test(queue.clone());

        let sid = create_test_session(&tv).await;
        let oid = create_test_layer(&ls, &sid, "lua-fred-timeout").await;

        let (seq, _ts) = ls
            .update_layer_relayed(
                &sid,
                "lua-fred-timeout",
                oid,
                Some(delta_with_set_field_inline(1, "k", vec![0x11], 0)),
                "solo-client",
                5000,
            )
            .await
            .expect(
                "Lua-backend update must commit on attempt 2 after attempt 1's raw fred \
                 Timeout is classified by map_snapshot_load_err as Aborted(\"transient I/O: \
                 ...\") and picked up by the retry loop. A regression that classifies the \
                 EVALSHA-call-site fred error as Internal instead of Aborted would make the \
                 retry loop bypass it and this .expect would panic.",
            );
        assert!(seq > 0);

        assert_eq!(
            queue.lock().unwrap().len(),
            0,
            "Lua fred injector queue must be fully drained by exactly one attempt"
        );

        // committed counter: commit on attempt 2 ⇒ 1 retry.
        let committed = live_edit::observability::metrics::test_util::sample_value(
            &live_edit::observability::metrics::test_util::render(&metrics),
            "live_edit_update_layer_retries_total",
            &[("path", FAST_PATH), ("outcome", "committed")],
        )
        .unwrap_or(0.0) as u64;
        assert_eq!(
            committed, 1,
            "commit on attempt 2 must record exactly one committed retry; got {committed}"
        );
        assert_eq!(
            live_edit::observability::metrics::test_util::sample_value(
                &live_edit::observability::metrics::test_util::render(&metrics),
                "live_edit_update_layer_retries_total",
                &[("path", FAST_PATH), ("outcome", "exhausted")],
            )
            .unwrap_or(0.0) as u64,
            0
        );
    }

    /// b.21r, mapper funnel: a fred `Timeout` received through
    /// `map_snapshot_load_err` counts exactly one
    /// `valkey_errors_total{pool="data", kind="command_timeout"}` — the pool the
    /// EVALSHA was issued on — and changes nothing about the commit, which still
    /// lands on the retry. fred reports a timed-out command only to its caller,
    /// so without this count the timeout would be invisible on the metric.
    ///
    /// The metrics handle is installed on the CLIENT, not the store: a
    /// `FredErrOrigin::client` origin resolves it through the client's metrics
    /// slot on the error path, as production does.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_lua_fred_timeout_counts_one_data_pool_command_timeout() {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("b21r-lua-timeout-count"));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;

        let queue = fred_injector(vec![synthetic_fred_err(FredErrorKind::Timeout)]);
        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_retry_max_attempts(32)
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .with_lua_fred_error_injector_for_test(queue.clone());

        let sid = create_test_session(&tv).await;
        let oid = create_test_layer(&ls, &sid, "lua-fred-timeout-count").await;
        assert_eq!(
            live_edit::storage::test_support::command_timeout_row(&metrics),
            live_edit::storage::test_support::command_timeouts_only_on("data", 0),
            "baseline: no command has timed out yet"
        );

        ls.update_layer_relayed(
            &sid,
            "lua-fred-timeout-count",
            oid,
            Some(delta_with_set_field_inline(1, "k", vec![0x33], 0)),
            "solo-client",
            7000,
        )
        .await
        .expect("the commit must still land on attempt 2: counting steers nothing");
        assert_eq!(
            queue.lock().unwrap().len(),
            0,
            "precondition: the injected Timeout was consumed by the EVALSHA call site"
        );

        assert_eq!(
            live_edit::storage::test_support::command_timeout_row(&metrics),
            live_edit::storage::test_support::command_timeouts_only_on("data", 1),
            "one timed-out EVALSHA on the data pool must count exactly once, on that pool"
        );
    }

    /// A non-transient fred Protocol error on the first EVALSHA must surface
    /// as Internal without retrying — mirroring the Txn-path contract in
    /// `update_layer_fred_protocol_error_surfaces_internal_without_retry`.
    /// Guards against a regression that widens the Lua retry classifier to
    /// over-match and silently retry protocol/parse/auth bugs.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_lua_fred_protocol_error_surfaces_internal_without_retry() {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only(
            "b67u-lua-protocol-no-retry",
        ));

        let queue = fred_injector(vec![synthetic_fred_err(FredErrorKind::Protocol)]);

        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_retry_max_attempts(32)
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .with_metrics(metrics.clone())
            .with_lua_fred_error_injector_for_test(queue.clone());

        let sid = create_test_session(&tv).await;
        let oid = create_test_layer(&ls, &sid, "lua-fred-protocol").await;

        let result = ls
            .update_layer_relayed(
                &sid,
                "lua-fred-protocol",
                oid,
                Some(delta_with_set_field_inline(1, "k", vec![0x22], 0)),
                "solo-client",
                6000,
            )
            .await;

        match result {
            Err(ServiceError::Internal(msg)) => {
                let protocol_label = FredErrorKind::Protocol.to_str();
                assert!(
                    msg.contains(protocol_label),
                    "non-transient Lua-path fred Protocol error must surface with fred's \
                     own ErrorKind label (`{protocol_label}`); got: {msg:?}"
                );
                assert!(
                    !msg.starts_with("transient I/O:"),
                    "non-transient error must not be misclassified as transient; got: {msg:?}"
                );
            }
            Ok(_) => panic!(
                "a non-transient fred Protocol error on the Lua EVALSHA call site MUST \
                 surface as Internal, not be silently retried. Observed Ok — the Lua \
                 retry classifier over-matched (b.67u scope regression)."
            ),
            Err(other) => panic!("expected Internal, got {other:?}"),
        }

        assert_eq!(
            queue.lock().unwrap().len(),
            0,
            "Lua fred injector queue must be drained by exactly one attempt; a remaining \
             entry would mean the retry loop swallowed the Internal error and tried again"
        );

        // No commit → no histogram sample, no retry counters.
        assert_eq!(
            live_edit::observability::metrics::test_util::sample_value(
                &live_edit::observability::metrics::test_util::render(&metrics),
                "live_edit_update_layer_retries_total",
                &[("path", FAST_PATH), ("outcome", "committed")],
            )
            .unwrap_or(0.0) as u64,
            0
        );
        assert_eq!(
            live_edit::observability::metrics::test_util::sample_value(
                &live_edit::observability::metrics::test_util::render(&metrics),
                "live_edit_update_layer_retries_total",
                &[("path", FAST_PATH), ("outcome", "exhausted")],
            )
            .unwrap_or(0.0) as u64,
            0
        );
    }

    /// Cluster-kind fred error — also classified transient (same set as
    /// Timeout in `is_transient_fred_error`). Pins the transient-kind table
    /// end-to-end on the Lua path so recovery covers every kind the Txn-path
    /// test covers.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_lua_fred_cluster_error_on_first_eval_flows_through_retry_loop() {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("b67u-lua-cluster-chain"));

        let queue = fred_injector(vec![synthetic_fred_err(FredErrorKind::Cluster)]);

        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_retry_max_attempts(32)
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .with_metrics(metrics.clone())
            .with_lua_fred_error_injector_for_test(queue.clone());

        let sid = create_test_session(&tv).await;
        let oid = create_test_layer(&ls, &sid, "lua-fred-cluster").await;

        ls.update_layer_relayed(
            &sid,
            "lua-fred-cluster",
            oid,
            Some(delta_with_set_field_inline(1, "k", vec![0x33], 0)),
            "solo-client",
            7000,
        )
        .await
        .expect(
            "a raw fred Cluster error on Lua EVALSHA must be classified as transient and \
             retried. A regression that treats Cluster as non-transient would surface as \
             Internal here and this .expect would panic.",
        );

        assert_eq!(queue.lock().unwrap().len(), 0);
        assert_eq!(
            live_edit::observability::metrics::test_util::sample_value(
                &live_edit::observability::metrics::test_util::render(&metrics),
                "live_edit_update_layer_retries_total",
                &[("path", FAST_PATH), ("outcome", "committed")],
            )
            .unwrap_or(0.0) as u64,
            1,
            "commit on attempt 2 must record exactly one committed retry"
        );
        assert_eq!(
            live_edit::observability::metrics::test_util::sample_value(
                &live_edit::observability::metrics::test_util::render(&metrics),
                "live_edit_update_layer_retries_total",
                &[("path", FAST_PATH), ("outcome", "exhausted")],
            )
            .unwrap_or(0.0) as u64,
            0
        );
    }
}

// =============================================================================
// 10. b.ph9 — a rejected stream write must fail the commit WHOLE.
//
// The hazard: `XADD {new_seq}-0` is the one call on a healthy keyspace that
// can be refused (stream-ID monotonicity — the stream's top entry already
// sits at or above the ID the commit intends to write). Pre-fix, both commit
// paths had already written the materialized state and advanced the header
// by the time that rejection surfaced, so the caller got an error while the
// object's `seq` had moved and NO stream entry existed for the commit: live
// subscribers silently missed it and diverged permanently. Acked-to-nobody
// state.
//
// The property these tests pin is the one the fix establishes: on refusal the
// per-object keyspace is bit-for-bit what it was before the call. That is
// what makes them genuine regression tests — against the pre-fix script the
// headline pair fails on the keyspace comparison (header `seq` advanced, the
// field was written), not merely on the error string.
//
// Both backends are covered because the SDD §7.5 parity contract makes this a
// behavioural contract, not a fast-path detail: `LIVE_EDIT_UPDATE_LAYER_BACKEND`
// is an operator rollback knob, so a guard present on only one backend is a
// guard one env-var away from being absent.
// =============================================================================

/// Default distance above the object's current `seq` at which the out-of-band
/// entry is planted.
///
/// EVERY offset `>= 1` is a genuine collision here, including 1, because these
/// fixtures leave the header UN-ADVANCED: with `seq = N` and an entry at
/// `{N + 1}-0`, the stream top has run ahead of the header, which is precisely
/// the anomaly both guards exist to catch. The benign shape the header-anchored
/// probe must report clean is NOT "an entry sits at `{seq + 1}-0`" — it is "an
/// entry sits at `{seq + 1}-0` **and the header has MOVED to `N + 1`**", i.e.
/// a rival writer committed the whole thing and this writer merely lost the
/// optimistic-concurrency race. That shape cannot be produced by planting an
/// entry at all (planting moves the stream without moving the header); it is
/// produced ~30 times per run by the 50-writer contention test in
/// `update_layer_retry_test.rs`, which is where it is covered. See the
/// "Why the header is re-read here, atomically" block in
/// `src/storage/scripts/stream_collision_probe.lua`.
///
/// 5 is the default because it is unambiguous at a glance. Offset 1 — the
/// exact-equality boundary, where the slow path's inclusive `XRANGE <id> +`
/// start and the fast path's "equal or smaller" `XADD` rejection are the
/// deciding predicate — is driven separately by the `_entry_at_exact_next_seq_`
/// pair below.
const STALE_STREAM_SEQ_OFFSET: u64 = 5;

/// The `ServiceError::Internal` message prefix both backends produce for a
/// stream-ID collision on the update path (`STREAM_ID_COLLISION_UPDATE_PREFIX`
/// in `src/storage/layer.rs`, which is `pub(crate)` and therefore not nameable
/// from an integration-test crate).
const COLLISION_UPDATE_PREFIX: &str = "update_layer stream-id collision";

/// Ditto for `write_object`'s CreateLayer guard.
const COLLISION_CREATE_PREFIX: &str = "create_layer stream-id collision";

/// Everything a b.ph9 collision test needs: a fixture, a backend-pinned store
/// with its own metrics registry, and the session / layer / object the writes
/// target.
struct CollisionHarness {
    tv: TestValkey,
    ls: LayerStore,
    metrics: Arc<Metrics>,
    sid: String,
    layer_name: String,
    oid: u64,
}

impl CollisionHarness {
    /// One field-update commit. `key` discriminates the field so repeated
    /// calls in one test are distinguishable in the keyspace dump.
    async fn update(&self, key: &str, client_ts: u64) -> Result<(u64, u64), ServiceError> {
        self.ls
            .update_layer_relayed(
                &self.sid,
                &self.layer_name,
                self.oid,
                Some(delta_with_set_field_inline(1, key, vec![0xB9], 0)),
                "ph9-client",
                client_ts,
            )
            .await
    }

    async fn header(&self) -> HashMap<String, String> {
        self.tv
            .client()
            .pool
            .hgetall(self.tv.key_schema().obj_header(&self.oid.to_string()))
            .await
            .expect("HGETALL obj:hdr must succeed")
    }

    async fn header_u64(&self, field: &str) -> u64 {
        let header = self.header().await;
        header
            .get(field)
            .unwrap_or_else(|| panic!("obj:hdr must carry `{field}`; got {header:?}"))
            .parse()
            .expect("header field must be a decimal u64")
    }

    fn stream_key(&self) -> String {
        self.tv.key_schema().layer_stream(self.oid)
    }

    /// Append an entry at `entry_id` out of band — the observable shape of a
    /// DR restore that seeded the header below a surviving stream's top, or of
    /// an out-of-band writer on the object's stream.
    async fn plant_stream_entry(&self, entry_id: &str) {
        let _: String = self
            .tv
            .client()
            .pool
            .xadd(
                &self.stream_key(),
                false,
                None,
                entry_id,
                vec![("payload", "out-of-band|planted-by-test")],
            )
            .await
            .expect("planting the out-of-band stream entry must succeed");
    }

    async fn stream_entry_ids(&self) -> Vec<String> {
        let entries: Vec<(String, HashMap<String, bytes::Bytes>)> = self
            .tv
            .client()
            .pool
            .xrange_values(&self.stream_key(), "-", "+", None)
            .await
            .expect("XRANGE must succeed");
        entries.into_iter().map(|(id, _)| id).collect()
    }
}

async fn setup_collision_harness(
    backend: UpdateLayerBackend,
    metrics_label: &str,
    layer_name: &str,
) -> CollisionHarness {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(metrics_label));
    let ls = layer_store_with_backend(&tv, backend).with_metrics(metrics.clone());
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, layer_name).await;
    CollisionHarness {
        tv,
        ls,
        metrics,
        sid,
        layer_name: layer_name.to_owned(),
        oid,
    }
}

/// Live value of one `live_edit_stream_id_collision_total` cell. Each harness
/// owns its own registry, so absolute assertions are sound.
///
/// The counter is deliberately NOT named `update_layer_*`: the create site
/// (`write_object`'s pre-`MULTI` `EXISTS`) reports into the same metric under
/// `site=create`, so the name spans both commit entry points.
///
/// The `{backend="lua", site="update"}` cell is split by the update script's
/// `mode`; it is read through [`lua_update_fast_mode_count`], so a collision
/// counted under `mode="structural"` fails the read.
fn collision_counter(metrics: &Metrics, backend: &str, site: &str) -> u64 {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    let text = render(metrics);
    if (backend, site) == ("lua", "update") {
        return lua_update_fast_mode_count(&text, "live_edit_stream_id_collision_total");
    }
    sample_value(
        &text,
        "live_edit_stream_id_collision_total",
        &[("backend", backend), ("site", site)],
    )
    .unwrap_or(0.0) as u64
}

/// The `mode="fast"` series of `metric`'s `{backend="lua", site="update"}`
/// cell, in rendered exposition `text` — the series every field-delta
/// Lua-backend refusal in this file ticks, since `update_layer` sends a
/// field delta through the update script's fast mode — after asserting the
/// `mode="structural"` series still reads 0. A refusal counted under the
/// structural mode fails here instead of adding up to the right total.
/// Panics when either series is absent: both are zero-seeded in
/// `Metrics::new`. The structural route's own refusal is read by
/// [`lua_update_structural_mode_count`].
fn lua_update_fast_mode_count(text: &str, metric: &str) -> u64 {
    use live_edit::observability::metrics::test_util::sample_value;
    let series = |mode: &str| {
        sample_value(
            text,
            metric,
            &[("backend", "lua"), ("site", "update"), ("mode", mode)],
        )
        .unwrap_or_else(|| {
            panic!("{metric}{{backend=\"lua\",site=\"update\",mode=\"{mode}\"}} must be present")
        }) as u64
    };
    assert_eq!(
        series("structural"),
        0,
        "a fast-mode refusal must not tick {metric}'s mode=\"structural\" series"
    );
    series("fast")
}

/// The `mode="structural"` series of `metric`'s `{backend="lua",
/// site="update"}` cell, after asserting the `mode="fast"` series still reads
/// 0: the twin of [`lua_update_fast_mode_count`] for a refusal of the `lua`
/// backend's structural route.
fn lua_update_structural_mode_count(text: &str, metric: &str) -> u64 {
    use live_edit::observability::metrics::test_util::sample_value;
    let series = |mode: &str| {
        sample_value(
            text,
            metric,
            &[("backend", "lua"), ("site", "update"), ("mode", mode)],
        )
        .unwrap_or_else(|| {
            panic!("{metric}{{backend=\"lua\",site=\"update\",mode=\"{mode}\"}} must be present")
        }) as u64
    };
    assert_eq!(
        series("fast"),
        0,
        "a structural-mode refusal must not tick {metric}'s mode=\"fast\" series"
    );
    series("structural")
}

/// Assert the error is the distinct, non-retriable collision surface — the
/// same `ServiceError` variant and prefix on both backends (SDD §7.5).
fn assert_collision_error(err: ServiceError, prefix: &str) -> String {
    match err {
        ServiceError::Internal(msg) => {
            assert!(
                msg.starts_with(prefix),
                "a stream-ID collision must surface with the `{prefix}:` prefix so operators \
                 can tell it apart from every other Internal; got: {msg:?}"
            );
            assert!(
                msg.contains("STREAM_ID_COLLISION"),
                "the tagged reply must survive into the operator-visible message; got: {msg:?}"
            );
            msg
        }
        // Aborted would be the damaging mis-classification: the retry loop
        // would burn all 128 attempts against a condition no retry can clear.
        // FailedPrecondition would send the client down the re-resolve-the-
        // mapping recovery path, which cannot help either.
        other => panic!(
            "a stream-ID collision must be a non-retriable `Internal` — `Aborted` would burn \
             the whole retry budget and `FailedPrecondition` would advertise a recovery path \
             that cannot clear a stale stream top; got {other:?}"
        ),
    }
}

/// Shared body for the headline regression on both backends.
///
/// Plant an entry `seq_offset` above the header, snapshot the ENTIRE
/// per-object keyspace, issue a commit, and assert the refusal left that
/// keyspace untouched.
///
/// `seq_offset` is a parameter rather than a constant because offset 1 is a
/// genuinely different test from offset 5: it is the exact-equality boundary
/// of the "at or above" predicate both guards implement (the slow path's
/// `XRANGE <new_seq>-0 +` start is INCLUSIVE; the fast path's `XADD` rejects
/// an ID equal to or smaller than the stream top). An off-by-one that
/// weakened either into a strict "above" would keep passing at offset 5 and
/// let the offset-1 collision commit — the pre-b.ph9 half-commit, reachable
/// again through the narrowest possible window.
async fn assert_stale_stream_top_refuses_commit_whole(
    backend: UpdateLayerBackend,
    backend_label: &str,
    metrics_label: &str,
    layer_name: &str,
    seq_offset: u64,
) {
    assert!(
        seq_offset >= 1,
        "offset 0 plants at the header's OWN sequence, which is already occupied by the \
         previous commit's entry — not the condition under test"
    );
    let h = setup_collision_harness(backend, metrics_label, layer_name).await;

    let seq_before = h.header_u64("seq").await;
    let next_set_order_before = h.header_u64("next_set_order").await;
    h.plant_stream_entry(&format!("{}-0", seq_before + seq_offset))
        .await;

    let (ks_before, clock_before) = dump_object_keyspace(&h.tv, h.oid).await;
    let stream_ids_before = h.stream_entry_ids().await;

    let err = h
        .update("wedged", 4242)
        .await
        .expect_err("a commit whose {new_seq}-0 XADD is already occupied must NOT report success");
    assert_collision_error(err, COLLISION_UPDATE_PREFIX);

    // (a) The header did not move. Pre-fix this is where the Lua path failed:
    // `seq` and `next_set_order` had already been HSET when the XADD was
    // rejected, so the object's version cursor advanced for a commit no
    // subscriber ever saw.
    assert_eq!(
        h.header_u64("seq").await,
        seq_before,
        "a refused commit must not advance obj:hdr seq — an advanced seq with no matching \
         stream entry is exactly the silent divergence b.ph9 closes"
    );
    assert_eq!(
        h.header_u64("next_set_order").await,
        next_set_order_before,
        "a refused commit must not advance obj:hdr next_set_order"
    );

    // (b) Nothing else moved either. The full-keyspace comparison is the
    // load-bearing assertion: it also catches the field HSET / index ZADD
    // that the pre-fix script had already performed.
    let (ks_after, clock_after) = dump_object_keyspace(&h.tv, h.oid).await;
    assert_eq!(
        ks_after, ks_before,
        "the {backend_label} backend must leave the per-object keyspace bit-for-bit unchanged \
         when the stream write is refused.\n\nAFTER: {ks_after:#?}\n\nBEFORE: {ks_before:#?}"
    );

    // (b2) ...including the header's commit clock. `dump_object_keyspace`
    // strips the timestamp columns before the byte comparison above (they are
    // wall clocks, un-comparable across the two independent fixtures the
    // parity harness uses), so `last_server_ts` would slip through (b) as a
    // blind spot. It is not a cosmetic one: it is the checkpoint gate's only
    // commit clock, so a future reordering that hoisted just the header write
    // ahead of the stream write would advertise a commit that never happened
    // to the quiesce arm — with (b) still green.
    assert!(
        clock_before.last_server_ts.is_some(),
        "pre-condition: create_layer must have stamped last_server_ts, or the comparison \
         below is vacuous"
    );
    assert_eq!(
        clock_after.last_server_ts, clock_before.last_server_ts,
        "a refused commit must not restamp obj:hdr last_server_ts"
    );

    // (c) ...including the stream itself: no entry appended, none trimmed.
    assert_eq!(
        h.stream_entry_ids().await,
        stream_ids_before,
        "a refused commit must not append to (or trim) the per-object stream"
    );

    // (d) Exactly one attempt reached the guard. The counter is the honest
    // attempt probe here: the retry loop increments it once per attempt, so a
    // mis-classification as `Aborted` would show up as 128 (the default
    // LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS), not 1.
    assert_eq!(
        collision_counter(&h.metrics, backend_label, "update"),
        1,
        "the collision must be counted exactly once — a value > 1 means the error was \
         classified retriable and the retry budget is being burned against a condition no \
         retry can clear"
    );
    assert_eq!(
        retry_counter(&h.metrics, field_delta_path(backend), "committed"),
        0,
        "a refused commit must not record a committed retry"
    );
    assert_eq!(
        retry_counter(&h.metrics, field_delta_path(backend), "exhausted"),
        0,
        "a refused commit must not consume the retry budget"
    );
    assert_eq!(
        collision_counter(&h.metrics, backend_label, "create"),
        0,
        "an update-path collision must not be attributed to the create site"
    );
}

/// Headline b.ph9 regression, Lua fast path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_stale_stream_top_refuses_commit_without_mutating_keyspace() {
    assert_stale_stream_top_refuses_commit_whole(
        UpdateLayerBackend::Lua,
        "lua",
        "bph9-lua-stale-stream",
        "bph9-lua-stale-stream-layer",
        STALE_STREAM_SEQ_OFFSET,
    )
    .await;
}

/// Headline b.ph9 regression, WATCH/MULTI/EXEC slow path. Same input, same
/// observable outcome — the backend toggle must not change the verdict.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_stale_stream_top_refuses_commit_without_mutating_keyspace() {
    assert_stale_stream_top_refuses_commit_whole(
        UpdateLayerBackend::Txn,
        "txn",
        "bph9-txn-stale-stream",
        "bph9-txn-stale-stream-layer",
        STALE_STREAM_SEQ_OFFSET,
    )
    .await;
}

/// Exact-equality boundary, Lua fast path: the planted entry sits at EXACTLY
/// the ID the commit intends to write (`{seq + 1}-0`), with the header left at
/// `seq`.
///
/// This is the tightest window in which the hazard exists, and the one a
/// weakened predicate reopens first. The fast path decides it inside
/// `redis.pcall("XADD", ...)`, whose rejection rule is "equal OR smaller than
/// the target stream top" — drop the equality half and this commit lands its
/// state with no stream entry, exactly the pre-b.ph9 divergence.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_stream_entry_at_exact_next_seq_refuses_commit_without_mutating_keyspace()
{
    assert_stale_stream_top_refuses_commit_whole(
        UpdateLayerBackend::Lua,
        "lua",
        "bph9-lua-exact-next-seq",
        "bph9-lua-exact-next-seq-layer",
        1,
    )
    .await;
}

/// Exact-equality boundary, WATCH/MULTI/EXEC slow path.
///
/// Distinct from its Lua twin because the deciding code is different: the slow
/// path never reaches an `XADD` it could be refused by (a command that fails
/// inside `MULTI` does not roll its siblings back), so the verdict comes from
/// `stream_collision_probe.lua`'s `XRANGE <new_seq>-0 + COUNT 1`, whose START
/// IS INCLUSIVE. Narrowing that to an exclusive start — or to `XRANGE
/// (<new_seq>-0` — would report clean here and half-commit.
///
/// The probe reports clean only when the HEADER has moved, which it has not
/// here: this fixture plants the entry without advancing `seq`, so the probe's
/// atomic header re-read still observes the sequence the caller WATCHed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_stream_entry_at_exact_next_seq_refuses_commit_without_mutating_keyspace()
{
    assert_stale_stream_top_refuses_commit_whole(
        UpdateLayerBackend::Txn,
        "txn",
        "bph9-txn-exact-next-seq",
        "bph9-txn-exact-next-seq-layer",
        1,
    )
    .await;
}

/// Positive control for the pair above: the guard refuses the stale-stream
/// condition and NOTHING else.
///
/// Without this, a guard that refused every commit outright would pass both
/// headline tests. Clearing the planted stream (the documented operator
/// remediation) must restore writability, and the very next commit must land
/// its own stream entry — proving the refusal path left no residue behind and
/// that the layer resumes from the un-advanced header.
async fn assert_commit_lands_once_stale_stream_cleared(
    backend: UpdateLayerBackend,
    backend_label: &str,
    metrics_label: &str,
    layer_name: &str,
) {
    let h = setup_collision_harness(backend, metrics_label, layer_name).await;
    let seq_before = h.header_u64("seq").await;
    h.plant_stream_entry(&format!("{}-0", seq_before + STALE_STREAM_SEQ_OFFSET))
        .await;

    let err = h
        .update("wedged", 1)
        .await
        .expect_err("pre-condition: the planted entry must wedge the first commit");
    assert_collision_error(err, COLLISION_UPDATE_PREFIX);

    // The operator remediation named in the WARN: delete the per-object
    // stream (subscribers re-bootstrap).
    let _: () =
        h.tv.client()
            .pool
            .del(h.stream_key())
            .await
            .expect("DEL of the wedged stream must succeed");

    let (seq, _ts) = h.update("recovered", 2).await.expect(
        "with the stale stream cleared the commit must land — the guard exists to refuse a \
         stream top that ran ahead of the header, not to refuse writes",
    );

    assert_eq!(
        seq,
        seq_before + 1,
        "the recovered commit must resume from the header the refused commit left \
         un-advanced ({seq_before}), advancing it by exactly one"
    );
    assert_eq!(
        h.header_u64("seq").await,
        seq,
        "obj:hdr seq must match the committed sequence number"
    );
    assert_eq!(
        h.stream_entry_ids().await,
        vec![format!("{seq}-0")],
        "the recovered commit must add exactly one entry, at its own sequence — the \
         {backend_label} backend returning Ok without an XADD would be the same silent \
         divergence in the other direction"
    );
    assert_eq!(
        collision_counter(&h.metrics, backend_label, "update"),
        1,
        "only the first (wedged) commit may be counted as a collision"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_commit_lands_after_stale_stream_cleared() {
    assert_commit_lands_once_stale_stream_cleared(
        UpdateLayerBackend::Lua,
        "lua",
        "bph9-lua-cleared",
        "bph9-lua-cleared-layer",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_commit_lands_after_stale_stream_cleared() {
    assert_commit_lands_once_stale_stream_cleared(
        UpdateLayerBackend::Txn,
        "txn",
        "bph9-txn-cleared",
        "bph9-txn-cleared-layer",
    )
    .await;
}

/// Under `lua-watch-structural` structural deltas take the WATCH/MULTI/EXEC
/// path, and there the probe rides the structural snapshot read
/// (`tokio::try_join!`) against a *predicted* sequence rather than
/// running serially against the applied one. That is a second, distinct call
/// site — the headline txn test above only exercises the serial one, because
/// a field-only delta forced down the slow path takes the `!needs_snapshot`
/// branch. (The default `lua` backend commits a structural delta through the
/// structural route instead: the next test.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_structural_delta_stale_stream_top_refuses_via_fanout_probe() {
    use crate::common::owned_delta::delta_with_diff_change_section;

    let h = setup_collision_harness(
        UpdateLayerBackend::LuaWatchStructural,
        "bph9-structural",
        "bph9-structural-layer",
    )
    .await;

    let seq_before = h.header_u64("seq").await;
    h.plant_stream_entry(&format!("{}-0", seq_before + STALE_STREAM_SEQ_OFFSET))
        .await;
    let (ks_before, clock_before) = dump_object_keyspace(&h.tv, h.oid).await;

    // `DiffChangeSection` is a structural command, so under
    // `lua-watch-structural` the dispatch routes to the slow path with
    // `needs_snapshot = true`.
    let err =
        h.ls.update_layer_relayed(
            &h.sid,
            &h.layer_name,
            h.oid,
            Some(delta_with_diff_change_section(0, 9, "sec", 1, 0)),
            "ph9-client",
            7,
        )
        .await
        .expect_err("a structural commit must be refused by the fan-out probe too");
    assert_collision_error(err, COLLISION_UPDATE_PREFIX);

    assert_eq!(
        h.header_u64("seq").await,
        seq_before,
        "the fan-out probe must fail the commit before MULTI opens — a moved seq means the \
         transaction ran and half-committed (MULTI does NOT roll back a command that fails \
         at execution time)"
    );
    let (ks_after, clock_after) = dump_object_keyspace(&h.tv, h.oid).await;
    assert_eq!(
        ks_after, ks_before,
        "a refused structural commit must leave the per-object keyspace untouched.\n\n\
         AFTER: {ks_after:#?}\n\nBEFORE: {ks_before:#?}"
    );
    // The keyspace comparison strips the timestamp columns, so assert the
    // commit clock separately — see the same assertion in
    // `assert_stale_stream_top_refuses_commit_whole` for why it is not
    // cosmetic.
    assert_eq!(
        clock_after.last_server_ts, clock_before.last_server_ts,
        "a refused structural commit must not restamp obj:hdr last_server_ts"
    );
    assert_eq!(
        collision_counter(&h.metrics, "txn", "update"),
        1,
        "the WATCH path is served by the txn backend whatever the configured update-layer \
         backend, so the collision must be attributed to `txn`"
    );
}

/// The default `lua` backend's structural route meets the same occupied
/// `{new_seq}-0` and fails closed the same way: the client-visible
/// collision refusal, the object's keyspace and commit clock untouched, and
/// the collision counted once on the update script's structural-mode series
/// (`{backend="lua", site="update", mode="structural"}`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_structural_delta_stale_stream_top_on_lua_backend_is_refused_by_the_structural_route(
) {
    use crate::common::owned_delta::delta_with_diff_change_section;
    use live_edit::observability::metrics::test_util::render;

    let h = setup_collision_harness(
        UpdateLayerBackend::Lua,
        "bph9-structural-route",
        "bph9-structural-route-layer",
    )
    .await;

    let seq_before = h.header_u64("seq").await;
    h.plant_stream_entry(&format!("{}-0", seq_before + STALE_STREAM_SEQ_OFFSET))
        .await;
    let (ks_before, clock_before) = dump_object_keyspace(&h.tv, h.oid).await;

    let err =
        h.ls.update_layer_relayed(
            &h.sid,
            &h.layer_name,
            h.oid,
            Some(delta_with_diff_change_section(0, 9, "sec", 1, 0)),
            "ph9-client",
            7,
        )
        .await
        .expect_err("a structural commit onto an occupied stream top must be refused");
    assert_collision_error(err, COLLISION_UPDATE_PREFIX);

    assert_eq!(
        h.header_u64("seq").await,
        seq_before,
        "the refused structural commit must not advance obj:hdr seq"
    );
    let (ks_after, clock_after) = dump_object_keyspace(&h.tv, h.oid).await;
    assert_eq!(
        ks_after, ks_before,
        "a refused structural commit must leave the per-object keyspace untouched.\n\n\
         AFTER: {ks_after:#?}\n\nBEFORE: {ks_before:#?}"
    );
    assert_eq!(
        clock_after.last_server_ts, clock_before.last_server_ts,
        "a refused structural commit must not restamp obj:hdr last_server_ts"
    );
    let text = render(&h.metrics);
    assert_eq!(
        lua_update_structural_mode_count(&text, "live_edit_stream_id_collision_total"),
        1,
        "the structural route's collision is the update script's, in its structural mode"
    );
    assert_eq!(
        collision_counter(&h.metrics, "txn", "update"),
        0,
        "the structural route takes no WATCH path, so nothing is attributed to `txn`"
    );
}

/// A stream key holding a non-stream value is the other way the commit's
/// `XADD` gets refused, and it must fail closed identically: the `XADD` inside
/// the MULTI would hit the same `WRONGTYPE` at execution time and half-commit.
///
/// The two backends reach the verdict through different code — the fast path
/// via its `redis.pcall`ed `XADD`, the slow path via the probe's tagged
/// `WRONGTYPE` reply arm — so both are driven.
async fn assert_wrongtype_stream_key_refuses_commit(
    backend: UpdateLayerBackend,
    backend_label: &str,
    metrics_label: &str,
    layer_name: &str,
) {
    let h = setup_collision_harness(backend, metrics_label, layer_name).await;
    let seq_before = h.header_u64("seq").await;

    // Replace the stream with a plain string.
    let _: () =
        h.tv.client()
            .pool
            .del(h.stream_key())
            .await
            .expect("DEL must succeed");
    let _: () =
        h.tv.client()
            .pool
            .set(h.stream_key(), "not-a-stream", None, None, false)
            .await
            .expect("SET must succeed");

    let err = h
        .update("wrongtype", 11)
        .await
        .expect_err("a commit whose stream key is not a stream must be refused");
    assert_collision_error(err, COLLISION_UPDATE_PREFIX);

    assert_eq!(
        h.header_u64("seq").await,
        seq_before,
        "a WRONGTYPE stream key must fail the commit closed, before any state write — \
         treating it as 'clean' would half-commit exactly like a stale stream top"
    );
    let fields_index_card: u64 =
        h.tv.client()
            .pool
            .zcard(h.tv.key_schema().obj_fields_index(&h.oid.to_string()))
            .await
            .expect("ZCARD fields index must succeed");
    assert_eq!(
        fields_index_card, 0,
        "no field may be written when the commit is refused; the layer was created empty"
    );
    assert_eq!(
        collision_counter(&h.metrics, backend_label, "update"),
        1,
        "a WRONGTYPE refusal is a collision-guard refusal and must be counted as one"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_wrongtype_stream_key_refuses_commit() {
    assert_wrongtype_stream_key_refuses_commit(
        UpdateLayerBackend::Lua,
        "lua",
        "bph9-lua-wrongtype",
        "bph9-lua-wrongtype-layer",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_wrongtype_stream_key_refuses_commit() {
    assert_wrongtype_stream_key_refuses_commit(
        UpdateLayerBackend::Txn,
        "txn",
        "bph9-txn-wrongtype",
        "bph9-txn-wrongtype-layer",
    )
    .await;
}

// =============================================================================
// 10b. b.ph9 review — a commit that raced a LAYER RETIREMENT is not a keyspace
// anomaly, and must not be reported as one.
//
// Both retirement paths append a tombstone with an AUTO id: `delete_layer`
// does `DEL mapping` → `XADD * payload deleted` → ext-GC → `cleanup_old_object`
// (which UNLINKs the header), and the CreateLayer-overwrite path does
// `HSET superseded=1` → flip mapping → `XADD * payload deleted` on the old
// stream. An auto id is a wall-clock millisecond, so it sits astronomically
// above any `{new_seq}-0` and refuses the in-flight writer's stream write on
// BOTH backends — while the header still reads `new_seq - 1`, because neither
// the tombstone nor `superseded=1` moves `seq`. Reported raw that is wrong
// twice over: a non-retriable INTERNAL "your stream is corrupt" for ordinary
// concurrency, and a tick on a counter whose entire value is that its baseline
// is zero.
//
// `LayerStore::disambiguate_stream_collision` re-reads the evidence before
// counting or shouting. The fixtures below plant BOTH halves of the race — the
// tombstone that produces the collision verdict AND the retirement evidence —
// and assert the ordinary retirement shape comes back with the collision
// counter untouched.
//
// Reachability note, so the coverage is not read as stronger than it is: the
// `mapping_deleted` / `mapping_moved` verdicts are reachable ONLY through
// `disambiguate_stream_collision` (nothing else on the commit path re-reads
// the layer mapping — `update_layer` takes `object_id` as an argument), so
// those two cases drive the new code directly. The `header_deleted` and
// `superseded` verdicts exist for the narrower window in which the header is
// retired between the commit's own header read and the disambiguation's, which
// no static fixture can pin open; planted statically they are decided by each
// backend's pre-existing header guard instead. They are still worth pinning,
// because what the caller must observe — the exact sentinel and status variant
// (`NotFound(EMPTY_HEADER_SENTINEL_MSG)` for a deleted header,
// `FailedPrecondition(SUPERSEDED_SENTINEL_MSG)` for a superseded one), and NO
// collision tick — is a single behavioural contract that the disambiguation and
// those guards are required to agree on. If the two ever disagree, the client's
// recovery depends on which side of a race it landed.
//
// Not covered here: the server-state pass-throughs by their real names —
// `-OOM` (`scriptVerifyOOM`), `-READONLY` / `-MISCONF` / `-NOREPLICAS`
// (`scriptVerifyWriteCommandAllow`), all of which land on the script's FIRST
// write, i.e. this XADD. Each needs `CONFIG SET` on the backing instance or a
// replica topology, and the shared sandbox / CI Valkey is neither
// reconfigurable nor replicated; `lua_fred_error_injector_for_test` cannot
// stand in for them either, because it short-circuits BEFORE the EVALSHA and so
// never reaches the tag-matching arms. The *classification rule* those shapes
// depend on — "untagged unless the reply is one of the two allowlisted
// shapes" — is covered in section 15 with a real, server-generated
// non-allowlisted refusal (`XADD 0-0`).
// =============================================================================

// `EMPTY_HEADER_SENTINEL_MSG` (imported at the top of this file from
// `tests/common`) mirrors the module-private
// `storage::layer::EMPTY_HEADER_SENTINEL`; it is shared rather than redeclared
// here because four test modules assert the same literal.

/// Mirror of `storage::layer::SUPERSEDED_SENTINEL`. Kept local (unlike the two
/// header-anomaly sentinels above) because this file is its only assertion site.
const SUPERSEDED_SENTINEL_MSG: &str = "object superseded by a concurrent CreateLayer overwrite";

/// Which half of a layer retirement the fixture leaves behind, alongside the
/// tombstone, for the disambiguation to find.
#[derive(Clone, Copy, Debug)]
enum RetirementEvidence {
    /// The `cleanup_old_object` cascade UNLINKed the object header.
    HeaderDeleted,
    /// A CreateLayer overwrite HSET `superseded=1` on the OLD header.
    Superseded,
    /// `delete_layer` DELeted the session → object mapping.
    MappingDeleted,
    /// A CreateLayer overwrite flipped the mapping to the NEW object.
    MappingMoved,
}

impl CollisionHarness {
    fn header_key(&self) -> String {
        self.tv.key_schema().obj_header(&self.oid.to_string())
    }

    fn mapping_key(&self) -> String {
        self.tv
            .key_schema()
            .layer_mapping(&self.sid, &self.layer_name)
    }

    /// Append the retirement tombstone exactly as `delete_layer` and the
    /// CreateLayer-overwrite path do: `XADD * payload deleted`. The auto id is
    /// a wall-clock millisecond, which is what puts it astronomically above the
    /// `{new_seq}-0` the next commit intends to write. Returns that id.
    async fn plant_deleted_tombstone(&self) -> String {
        self.tv
            .client()
            .pool
            .xadd(
                &self.stream_key(),
                false,
                None,
                "*",
                vec![(NOTIFICATION_PAYLOAD_FIELD, DELETED_STREAM_PAYLOAD)],
            )
            .await
            .expect("planting the retirement tombstone must succeed")
    }

    async fn plant_retirement_evidence(&self, evidence: RetirementEvidence) {
        let pool = &self.tv.client().pool;
        match evidence {
            RetirementEvidence::HeaderDeleted => {
                let _: () = pool
                    .del(self.header_key())
                    .await
                    .expect("DEL obj:hdr must succeed");
            }
            RetirementEvidence::Superseded => {
                let _: () = pool
                    .hset(self.header_key(), [("superseded", "1")])
                    .await
                    .expect("HSET superseded must succeed");
            }
            RetirementEvidence::MappingDeleted => {
                let _: () = pool
                    .del(self.mapping_key())
                    .await
                    .expect("DEL layer mapping must succeed");
            }
            RetirementEvidence::MappingMoved => {
                let _: () = pool
                    .set(
                        self.mapping_key(),
                        (self.oid + 1000).to_string(),
                        None,
                        None,
                        false,
                    )
                    .await
                    .expect("SET layer mapping must succeed");
            }
        }
    }
}

/// Assert the error is the ordinary retirement shape `evidence` calls for —
/// never the `Internal` collision surface.
fn assert_retirement_error(err: ServiceError, evidence: RetirementEvidence, h: &CollisionHarness) {
    match (evidence, err) {
        (RetirementEvidence::HeaderDeleted, ServiceError::NotFound(msg)) => {
            // Asserted by equality, and on a DIFFERENT message from the
            // `MappingDeleted` arm below, because `NotFound` now carries both
            // origins: "the mapping vanished" (that arm) and "the mapping
            // resolves but the header is gone" (this one). The variant alone no
            // longer tells them apart, so the message is what keeps the two
            // fixtures from silently testing the same path.
            assert_eq!(
                msg, EMPTY_HEADER_SENTINEL_MSG,
                "a commit whose object header was deleted underneath it must surface the \
                 exact empty-header sentinel — the disambiguation's `header_deleted` verdict \
                 and both backends' own header guards are required to agree on it"
            );
        }
        (RetirementEvidence::Superseded, ServiceError::FailedPrecondition(msg)) => {
            assert_eq!(
                msg, SUPERSEDED_SENTINEL_MSG,
                "a commit against an object a CreateLayer overwrite superseded must surface \
                 the exact superseded sentinel"
            );
        }
        (RetirementEvidence::MappingDeleted, ServiceError::NotFound(msg)) => {
            assert!(
                msg.contains(&h.layer_name) && msg.contains("not found in session"),
                "a commit whose layer mapping was deleted underneath it must surface the \
                 ordinary layer-not-found message naming the layer; got {msg:?}"
            );
        }
        (RetirementEvidence::MappingMoved, ServiceError::FailedPrecondition(msg)) => {
            assert_eq!(
                msg,
                format!(
                    "object_id mismatch: expected {}, got {}",
                    h.oid + 1000,
                    h.oid
                ),
                "a commit whose layer mapping was re-pointed underneath it must be told which \
                 object_id to re-resolve to — that message IS the client's recovery instruction"
            );
        }
        // The regression this whole block exists for. Pre-fix, every case
        // above surfaced here instead.
        (evidence, ServiceError::Internal(msg)) if msg.starts_with(COLLISION_UPDATE_PREFIX) => {
            panic!(
                "a commit that raced a layer retirement ({evidence:?}) must NOT be reported as \
                 a stream-id collision: `Internal` is non-retriable and tells the client its \
                 stream is corrupt, when the correct instruction is to re-resolve the layer \
                 mapping. Got: {msg:?}"
            )
        }
        (evidence, other) => {
            panic!("unexpected error shape for the {evidence:?} retirement race: {other:?}")
        }
    }
}

/// Shared body: plant the tombstone AND the retirement evidence, issue a
/// commit, and assert it comes back as the ordinary retirement status with the
/// keyspace untouched and the collision counter never ticked.
async fn assert_retirement_race_is_not_a_collision(
    backend: UpdateLayerBackend,
    metrics_label: &str,
    layer_name: &str,
    evidence: RetirementEvidence,
) {
    let h = setup_collision_harness(backend, metrics_label, layer_name).await;

    let tombstone_id = h.plant_deleted_tombstone().await;
    let tombstone_ms: u64 = tombstone_id
        .split_once('-')
        .expect("an auto stream id is always `<ms>-<seq>`")
        .0
        .parse()
        .expect("the millisecond component must be numeric");
    // Pre-condition for the whole race: the tombstone's auto id has to sit
    // above the `{seq + 1}-0` the commit below will attempt, or nothing
    // collides and the fixture is vacuous.
    assert!(
        tombstone_ms > h.header_u64("seq").await + 1,
        "the tombstone's wall-clock id ({tombstone_id}) must sit above the sequence the \
         commit intends to write, or this fixture does not reproduce the race"
    );

    h.plant_retirement_evidence(evidence).await;

    let header_before = h.header().await;
    let stream_ids_before = h.stream_entry_ids().await;

    let err = h
        .update("raced-a-retirement", 4343)
        .await
        .expect_err("a commit against a retired layer must not report success");
    assert_retirement_error(err, evidence, &h);

    // The commit failed whole — same requirement as a genuine collision.
    assert_eq!(
        h.header().await,
        header_before,
        "a commit refused by the retirement race must leave the object header untouched"
    );
    assert_eq!(
        h.stream_entry_ids().await,
        stream_ids_before,
        "a commit refused by the retirement race must not append to the per-object stream"
    );

    // The point of the fix. `live_edit_stream_id_collision_total` is an
    // operator alarm whose entire value is that its baseline is zero; a routine
    // DeleteLayer or CreateLayer overwrite racing a writer must not move it.
    // Both backend labels are asserted, so a tick attributed to the wrong
    // backend cannot slip past either.
    for backend_label in ["lua", "txn"] {
        for site in ["update", "create"] {
            assert_eq!(
                collision_counter(&h.metrics, backend_label, site),
                0,
                "a layer retirement racing an in-flight commit is ordinary concurrency and \
                 must not tick live_edit_stream_id_collision_total{{backend={backend_label},\
                 site={site}}} — that counter is an operator alarm with a zero baseline"
            );
        }
    }
    assert_eq!(
        retry_counter(&h.metrics, field_delta_path(backend), "exhausted"),
        0,
        "the retirement shapes are all non-retriable; none may burn the retry budget"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_header_deleted_under_tombstone_reports_empty_header_not_collision() {
    assert_retirement_race_is_not_a_collision(
        UpdateLayerBackend::Lua,
        "bph9-lua-retire-hdr",
        "bph9-lua-retire-hdr-layer",
        RetirementEvidence::HeaderDeleted,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_header_deleted_under_tombstone_reports_empty_header_not_collision() {
    assert_retirement_race_is_not_a_collision(
        UpdateLayerBackend::Txn,
        "bph9-txn-retire-hdr",
        "bph9-txn-retire-hdr-layer",
        RetirementEvidence::HeaderDeleted,
    )
    .await;
}

/// The CreateLayer-overwrite hole the PM flagged on the slow path: the txn
/// backend's own `superseded` guard runs against the header it read under
/// WATCH, so a flag set after that read is only caught by the disambiguation —
/// and until b.ph9's review round the probe reported it as a collision.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_superseded_under_tombstone_reports_superseded_not_collision() {
    assert_retirement_race_is_not_a_collision(
        UpdateLayerBackend::Txn,
        "bph9-txn-retire-superseded",
        "bph9-txn-retire-superseded-layer",
        RetirementEvidence::Superseded,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_superseded_under_tombstone_reports_superseded_not_collision() {
    assert_retirement_race_is_not_a_collision(
        UpdateLayerBackend::Lua,
        "bph9-lua-retire-superseded",
        "bph9-lua-retire-superseded-layer",
        RetirementEvidence::Superseded,
    )
    .await;
}

/// `delete_layer` DELs the mapping BEFORE it appends the tombstone and long
/// before `cleanup_old_object` UNLINKs the header, so "mapping gone, header
/// still live, tombstone on the stream" is the widest — and the most likely to
/// be hit — window of the whole race. Nothing else on the commit path re-reads
/// the mapping, so this verdict comes from the disambiguation or from nowhere.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_mapping_deleted_under_tombstone_reports_not_found_not_collision() {
    assert_retirement_race_is_not_a_collision(
        UpdateLayerBackend::Lua,
        "bph9-lua-retire-mapping",
        "bph9-lua-retire-mapping-layer",
        RetirementEvidence::MappingDeleted,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_mapping_deleted_under_tombstone_reports_not_found_not_collision() {
    assert_retirement_race_is_not_a_collision(
        UpdateLayerBackend::Txn,
        "bph9-txn-retire-mapping",
        "bph9-txn-retire-mapping-layer",
        RetirementEvidence::MappingDeleted,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_mapping_moved_under_tombstone_reports_oid_mismatch_not_collision() {
    assert_retirement_race_is_not_a_collision(
        UpdateLayerBackend::Lua,
        "bph9-lua-retire-moved",
        "bph9-lua-retire-moved-layer",
        RetirementEvidence::MappingMoved,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_mapping_moved_under_tombstone_reports_oid_mismatch_not_collision() {
    assert_retirement_race_is_not_a_collision(
        UpdateLayerBackend::Txn,
        "bph9-txn-retire-moved",
        "bph9-txn-retire-moved-layer",
        RetirementEvidence::MappingMoved,
    )
    .await;
}

/// Negative control for the whole block: the disambiguation is EVIDENCE-based,
/// not "a tombstone means retirement".
///
/// A tombstone sitting on the stream of a layer that is still live — header
/// present, not superseded, mapping still pointing here — is a real keyspace
/// anomaly (a `deleted` sentinel was appended by something that then failed to
/// retire the layer), and must keep the fail-closed `Internal` + counter tick.
/// Without this, a disambiguation that short-circuited on `kind == "deleted"`
/// would pass every test above while silently disarming the guard for any
/// stream carrying a stray sentinel.
async fn assert_tombstone_on_a_live_layer_is_still_a_collision(
    backend: UpdateLayerBackend,
    backend_label: &str,
    metrics_label: &str,
    layer_name: &str,
) {
    let h = setup_collision_harness(backend, metrics_label, layer_name).await;
    let seq_before = h.header_u64("seq").await;
    h.plant_deleted_tombstone().await;

    let err = h
        .update("tombstone-but-live", 55)
        .await
        .expect_err("a stream top above the header must still refuse the commit");
    assert_collision_error(err, COLLISION_UPDATE_PREFIX);

    assert_eq!(
        h.header_u64("seq").await,
        seq_before,
        "the refused commit must not advance obj:hdr seq"
    );
    assert_eq!(
        collision_counter(&h.metrics, backend_label, "update"),
        1,
        "the layer is still live — header present, not superseded, mapping still resolving to \
         this object — so the tombstone is an anomaly, not a retirement, and must be counted"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_tombstone_on_live_layer_is_still_counted_as_a_collision() {
    assert_tombstone_on_a_live_layer_is_still_a_collision(
        UpdateLayerBackend::Lua,
        "lua",
        "bph9-lua-live-tombstone",
        "bph9-lua-live-tombstone-layer",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_tombstone_on_live_layer_is_still_counted_as_a_collision() {
    assert_tombstone_on_a_live_layer_is_still_a_collision(
        UpdateLayerBackend::Txn,
        "txn",
        "bph9-txn-live-tombstone",
        "bph9-txn-live-tombstone-layer",
    )
    .await;
}

// =============================================================================
// 10c. b.ph9 review — `stream_collision_probe.lua`'s reply shape, driven
// directly.
//
// The script's contract widened from "an id, or nil" to "`{id, kind}`, or
// nil". `probe_stream_id_collision` decodes it positionally
// (`it.next()` / `it.next().unwrap_or_default()`), so a script that returned
// the bare id again — or put `kind` first — would silently classify every
// collision as non-tombstone and quietly re-arm the false alarm the review
// round closed. Driving the script directly pins the wire shape that the
// through-the-API tests above can only observe indirectly.
// =============================================================================

/// EVAL the probe exactly as `probe_stream_id_collision` does.
async fn eval_stream_collision_probe(
    h: &CollisionHarness,
    attempted_id: &str,
    expected_header_seq: u64,
) -> Option<Vec<String>> {
    STREAM_COLLISION_PROBE_SCRIPT
        .evalsha_with_reload::<Option<Vec<String>>, _, _>(
            h.tv.client().pool.next(),
            vec![h.stream_key(), h.header_key()],
            vec![attempted_id.to_owned(), expected_header_seq.to_string()],
        )
        .await
        .expect("the probe must not error on a well-typed stream key")
}

async fn probe_harness(layer_name: &str, metrics_label: &str) -> CollisionHarness {
    // The probe is only ever issued by the WATCH/MULTI/EXEC path, so the
    // fixture pins that backend even though the script is EVALed directly.
    setup_collision_harness(UpdateLayerBackend::Txn, metrics_label, layer_name).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_collision_probe_returns_nil_when_nothing_sits_at_or_above_the_attempted_id() {
    let h = probe_harness("bph9-probe-clear-layer", "bph9-probe-clear").await;
    let seq = h.header_u64("seq").await;

    assert_eq!(
        eval_stream_collision_probe(&h, &format!("{}-0", seq + 1), seq).await,
        None,
        "with only CreateLayer's own `1-0` entry present, nothing sits at or above `{{seq+1}}-0` \
         and the probe must report clean"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_collision_probe_reports_a_plain_colliding_entry_with_an_empty_kind() {
    let h = probe_harness("bph9-probe-hit-layer", "bph9-probe-hit").await;
    let seq = h.header_u64("seq").await;
    let planted = format!("{}-0", seq + STALE_STREAM_SEQ_OFFSET);
    h.plant_stream_entry(&planted).await;

    assert_eq!(
        eval_stream_collision_probe(&h, &format!("{}-0", seq + 1), seq).await,
        Some(vec![planted.clone(), String::new()]),
        "a colliding entry that is NOT a retirement tombstone must come back as \
         `[<id>, \"\"]` — the empty second element is what tells the caller to keep the \
         fail-closed collision verdict when its own disambiguation reads fail"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_collision_probe_returns_nil_when_the_header_has_moved() {
    let h = probe_harness("bph9-probe-moved-layer", "bph9-probe-moved").await;
    let seq = h.header_u64("seq").await;
    h.plant_stream_entry(&format!("{}-0", seq + STALE_STREAM_SEQ_OFFSET))
        .await;

    // Same colliding entry as the test above; only the header sequence the
    // caller claims to have WATCHed differs. This is the ordinary
    // optimistic-concurrency loss — a rival writer committed and advanced the
    // header — and reporting it as a collision turns the most routine event on
    // this path into a hard error (~30 of 50 writes in the contention test).
    assert_eq!(
        eval_stream_collision_probe(&h, &format!("{}-0", seq + 1), seq + 99).await,
        None,
        "when the header no longer holds the sequence the caller WATCHed, the probe must \
         report clean and let the EXEC cancel on the invalidated WATCH"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_collision_probe_tags_a_retirement_tombstone_as_deleted() {
    let h = probe_harness("bph9-probe-tombstone-layer", "bph9-probe-tombstone").await;
    let seq = h.header_u64("seq").await;
    let tombstone_id = h.plant_deleted_tombstone().await;

    assert_eq!(
        eval_stream_collision_probe(&h, &format!("{}-0", seq + 1), seq).await,
        Some(vec![tombstone_id, DELETED_STREAM_PAYLOAD.to_owned()]),
        "the retirement tombstone must be tagged `deleted` — that tag is the only evidence \
         the caller has when its own header / mapping re-reads fail, and without it a routine \
         DeleteLayer racing a writer is reported as a corrupt stream"
    );
}

/// EVAL the probe and return the error details, for the two cases where the
/// probe is expected to refuse rather than answer.
async fn eval_stream_collision_probe_err(
    h: &CollisionHarness,
    attempted_id: &str,
    expected_header_seq: u64,
) -> String {
    STREAM_COLLISION_PROBE_SCRIPT
        .evalsha_with_reload::<Option<Vec<String>>, _, _>(
            h.tv.client().pool.next(),
            vec![h.stream_key(), h.header_key()],
            vec![attempted_id.to_owned(), expected_header_seq.to_string()],
        )
        .await
        .expect_err("the probe must refuse this input")
        .details()
        .to_string()
}

/// The probe's error arm is an allowlist too, and `WRONGTYPE` is its single
/// member — the one shape that IS a verdict about this stream key, and the one
/// the fast path folds into the same tag for backend parity.
///
/// The end-to-end consequence is pinned by
/// `update_layer_txn_wrongtype_stream_key_refuses_commit`; this pins the reply
/// the caller's `starts_with(STREAM_ID_COLLISION_TAG)` arm actually matches on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_collision_probe_tags_a_wrongtype_stream_key() {
    let h = probe_harness("bph9-probe-wrongtype-layer", "bph9-probe-wrongtype").await;
    let seq = h.header_u64("seq").await;
    wrong_type_key(&h.tv, &h.stream_key()).await;

    let details = eval_stream_collision_probe_err(&h, &format!("{}-0", seq + 1), seq).await;
    assert!(
        details.starts_with("STREAM_ID_COLLISION"),
        "a WRONGTYPE stream key is the probe's one allowlisted refusal and must carry the tag \
         the caller matches by prefix — untagged it would route through `map_snapshot_load_err` \
         and the operator would never be told the stream key is unusable; got: {details:?}"
    );
    assert!(
        details.contains("WRONGTYPE"),
        "the tagged reply must quote the server's own `WRONGTYPE`, which the probe matches as a \
         9-character prefix; got: {details:?}"
    );
}

/// The probe's half of the allowlist-inversion regression guard (section 15
/// covers the fast path's).
///
/// Pre-review the probe tagged its `pcall` error arm by exclusion, so ANY
/// refusal of the `XRANGE` came back as `STREAM_ID_COLLISION` — a non-retriable
/// "your stream is corrupt" plus a counter tick for something that is not a
/// verdict about the stream at all. It is now `WRONGTYPE`-only.
///
/// The class-B refusals that motivate the fix (`-NOPERM`, `-MOVED`) need an ACL
/// or a cluster and cannot be provoked on the shared sandbox instance; what IS
/// reachable is a malformed `XRANGE` start ID, which the server refuses with a
/// plain `ERR`. That is the same branch and the same string test, and it fails
/// on the pre-review script.
///
/// Untagged still fails CLOSED: the caller's fallthrough `Err` arm routes it
/// through `map_snapshot_load_err`, so the commit never opens its `MULTI` — it
/// is retried or reported as itself, rather than counted as a collision.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stream_collision_probe_returns_a_non_wrongtype_refusal_verbatim_and_untagged() {
    let h = probe_harness("bph9-probe-passthrough-layer", "bph9-probe-passthrough").await;
    let seq = h.header_u64("seq").await;

    // Reaches the `XRANGE` (the header re-read matches), which then refuses the
    // malformed start ID.
    let details = eval_stream_collision_probe_err(&h, "not-a-stream-id", seq).await;
    assert!(
        !details.contains("STREAM_ID_COLLISION"),
        "a probe refusal that is not a `WRONGTYPE` is not a verdict about this stream key and \
         must come back untagged, to be classified as itself. This is the assertion the \
         pre-review tag-by-exclusion arm fails; got: {details:?}"
    );
    assert!(
        details.contains("Invalid stream ID"),
        "the server's own reply must survive verbatim; got: {details:?}"
    );
}

// =============================================================================
// 11. b.ph9 — the other two mid-script exits the validation pre-pass closed.
//
// `SET_ORDER_OVERFLOW` and `ENCODING` used to be raised from inside the apply
// loop, so a delta whose Nth record tripped one had already committed records
// 1..N-1 — with `version = new_seq` sitting above a header whose `seq` was
// never advanced, which breaks the version-filtered-diff invariant catch-up
// reconstruction depends on. The fix hoists both decisions into a pre-pass
// that issues zero `redis.call`s.
//
// The boundary itself (2^53 - 1, both backends) is already pinned by
// `b_qw6_tests.rs`; what is new here is *where in the delta* it trips and what
// that leaves behind.
// =============================================================================

const SET_ORDER_CEILING: u64 = (1u64 << 53) - 1;

/// The Lua script's SET_ORDER_OVERFLOW exit must leave NO record written, even
/// when the record that trips it is not the first one.
///
/// Shape: `next_set_order` seeded 10 below the ceiling, then a single delta of
/// three SetFields whose SECOND record's `delta_set_order` folds the cursor
/// past it. Pre-fix, record 1 was HSET and its index entry ZADDed before the
/// error fired; post-fix the pre-pass decides it with the keyspace untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_set_order_overflow_on_second_record_writes_nothing() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bph9-set-order-midway").await;

    let keys = tv.key_schema();
    let oid_str = oid.to_string();
    let header_key = keys.obj_header(&oid_str);

    // Park the cursor just under the ceiling. Reaching it organically is not
    // possible in a test (it is ~9e15 commits away), and the header column is
    // the only input the fold reads.
    let seeded_next_set_order = SET_ORDER_CEILING - 10;
    let _: () = tv
        .client()
        .pool
        .hset(
            &header_key,
            [("next_set_order", seeded_next_set_order.to_string())],
        )
        .await
        .expect("seeding next_set_order must succeed");
    let seq_before: u64 = tv
        .client()
        .pool
        .hget::<String, _, _>(&header_key, "seq")
        .await
        .expect("HGET seq must succeed")
        .parse()
        .expect("seq must be numeric");
    let stream_len_before: u64 = tv
        .client()
        .pool
        .xlen(keys.layer_stream(oid))
        .await
        .expect("XLEN must succeed");

    // Record 1 folds cleanly (cursor → ceiling - 9); record 2 asks for +20,
    // which crosses; record 3 would also cross but is never reached.
    let delta = OwnedDelta {
        set_fields: vec![
            SetField {
                section_id: 1,
                key_name: "first".to_owned(),
                value_or_ext_hash: vec![0x01],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            },
            SetField {
                section_id: 1,
                key_name: "second".to_owned(),
                value_or_ext_hash: vec![0x02],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 20,
            },
            SetField {
                section_id: 1,
                key_name: "third".to_owned(),
                value_or_ext_hash: vec![0x03],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            },
        ],
        ..Default::default()
    };

    let err = ls
        .update_layer_relayed(
            &sid,
            "bph9-set-order-midway",
            oid,
            Some(delta),
            "ph9-client",
            5,
        )
        .await
        .expect_err("a delta that folds next_set_order past 2^53 - 1 must be refused");
    match err {
        ServiceError::Internal(msg) => assert!(
            msg.contains("SET_ORDER_OVERFLOW"),
            "the set_order ceiling must stay distinguishable from every other Internal; \
             got: {msg:?}"
        ),
        other => panic!("the set_order ceiling must be a non-retriable Internal; got {other:?}"),
    }

    // The regression: record 1 must NOT have been written.
    let first_field_key = keys.obj_field(&oid_str, "1", "first");
    let first_exists: u64 = tv
        .client()
        .pool
        .exists(&first_field_key)
        .await
        .expect("EXISTS must succeed");
    assert_eq!(
        first_exists, 0,
        "the record BEFORE the one that tripped the ceiling must not have been written. \
         Pre-fix it was HSET with `version = new_seq`, a version above a header seq that \
         never advanced — which is precisely the state version-filtered diff \
         reconstruction cannot represent"
    );
    let fields_index_card: u64 = tv
        .client()
        .pool
        .zcard(keys.obj_fields_index(&oid_str))
        .await
        .expect("ZCARD must succeed");
    assert_eq!(
        fields_index_card, 0,
        "no fields-index entry may survive a refused commit"
    );

    let header_after: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL must succeed");
    assert_eq!(
        header_after.get("seq").map(String::as_str),
        Some(seq_before.to_string().as_str()),
        "a refused commit must not advance seq"
    );
    assert_eq!(
        header_after.get("next_set_order").map(String::as_str),
        Some(seeded_next_set_order.to_string().as_str()),
        "a refused commit must not advance next_set_order"
    );
    let stream_len_after: u64 = tv
        .client()
        .pool
        .xlen(keys.layer_stream(oid))
        .await
        .expect("XLEN must succeed");
    assert_eq!(
        stream_len_after, stream_len_before,
        "the pre-pass rejects before the XADD, so no stream entry may be appended either"
    );
}

// =============================================================================
// 12. b.ph9 — CreateLayer's own stream-ID guard (`write_object`).
//
// `CreateLayer` XADDs `1-0` inside the same non-rolling-back MULTI as every
// state write, so a pre-existing stream on the freshly-minted `object_id`
// would materialize the object while erroring to the caller — and no
// subscriber would ever see the layer. `object_id` comes straight off an
// `INCR`, so any pre-existing stream on it is by definition an anomaly and a
// plain `EXISTS` is the whole test.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_pre_existing_stream_on_minted_object_id_refuses_and_writes_no_state() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("bph9-create-guard"));
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua).with_metrics(metrics.clone());
    let sid = create_test_session(&tv).await;

    // The object-id counter is a plain INCR, so the next CreateLayer takes
    // `current + 1`. Reading it is the only way to plant a stream under an id
    // that has not been minted yet — which is the condition the guard exists
    // for (a leaked stream from a rolled-back counter, or an out-of-band
    // writer).
    let counter_key = tv.key_schema().counter_object_id();
    let current: u64 = tv
        .client()
        .pool
        .get::<Option<String>, _>(&counter_key)
        .await
        .expect("GET object-id counter must succeed")
        .map(|v| v.parse().expect("counter must be numeric"))
        .unwrap_or(0);
    let doomed_oid = current + 1;

    let stream_key = tv.key_schema().layer_stream(doomed_oid);
    let _: String = tv
        .client()
        .pool
        .xadd(
            &stream_key,
            false,
            None,
            "1-0",
            vec![("payload", "leaked|from-a-previous-life")],
        )
        .await
        .expect("planting the leaked stream must succeed");

    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let err = ls
        .create_layer(
            &sid,
            "bph9-create-guard-layer",
            "test-user",
            &state,
            mock_ext_store(),
        )
        .await
        .expect_err(
            "CreateLayer must refuse to materialize a layer over a stream that already \
             exists on its freshly-minted object_id",
        );
    let msg = assert_collision_error(err, COLLISION_CREATE_PREFIX);
    assert!(
        msg.contains(&doomed_oid.to_string()),
        "the refusal must name the object_id an operator has to investigate; got: {msg:?}"
    );

    // No partial object: the guard runs before MULTI opens, so neither the
    // header nor the session→layer mapping may exist.
    let header_exists: u64 = tv
        .client()
        .pool
        .exists(tv.key_schema().obj_header(&doomed_oid.to_string()))
        .await
        .expect("EXISTS obj:hdr must succeed");
    assert_eq!(
        header_exists, 0,
        "a refused CreateLayer must not materialize the object header — the pre-fix shape \
         wrote the whole object and then returned an error, leaving a layer no subscriber \
         could ever be told about"
    );
    let mapping: Option<String> = tv
        .client()
        .pool
        .get(
            tv.key_schema()
                .layer_mapping(&sid, "bph9-create-guard-layer"),
        )
        .await
        .expect("GET layer mapping must succeed");
    assert_eq!(
        mapping, None,
        "a refused CreateLayer must not publish a session→layer mapping"
    );
    assert_eq!(
        collision_counter(&metrics, "txn", "create"),
        1,
        "the create-site refusal must be counted under site=create (the create path is \
         served exclusively by write_object's MULTI/EXEC, hence backend=txn)"
    );
    assert_eq!(
        collision_counter(&metrics, "txn", "update"),
        0,
        "a create-site refusal must not be attributed to the update site"
    );
}

// =============================================================================
// 13. b.ph9 — the validation pre-pass, driven directly at the EVALSHA boundary.
//
// Two of the pre-pass's three rejections (`ENCODING: unknown mutation tag`,
// `ENCODING: malformed value-hfield count`, `ENCODING: malformed
// delta_set_order`) describe an ARGV record the Rust encoder cannot produce.
// That is the point of them — they are the guard against a FUTURE encoder bug
// — and it is also why no `update_layer` call can reach them: every path into
// the script runs `lua::encode_delta`, which emits well-formed records by
// construction. Section 11 above covers the third (`SET_ORDER_OVERFLOW`)
// because that one IS reachable from a delta.
//
// So the honest way to test these is to speak the script's own contract:
// rebuild the invocation `update_layer_once_lua` makes from the same public
// pieces it uses (`lua::encode_delta` → `EncodedMutations`, plus the fixed
// KEYS / ARGV prefix), corrupt exactly one slot, and dispatch
// `UPDATE_LAYER_FAST_SCRIPT` directly. No production seam is added for a
// condition production cannot create.
//
// What these pin, and why the b.ph9 restructuring made them worth writing:
// pre-fix both errors were raised from inside the APPLY loop, so a malformed
// record N committed records 1..N-1 — field hashes carrying `version =
// new_seq` above a header whose `seq` never advanced, which the
// version-filtered diff reconstruction cannot represent. Every case below
// therefore corrupts the SECOND record and asserts the FIRST one was not
// written, on top of a full before/after keyspace comparison.
//
// `update_layer_fast_script_direct_invocation_commits_like_the_call_site` is
// the load-bearing companion: it commits an UNCORRUPTED hand-built invocation
// and checks the result, so a mis-built prefix cannot make every corruption
// case pass for the wrong reason.
// =============================================================================

/// Everything `update_layer_once_lua` hands to `UPDATE_LAYER_FAST_SCRIPT`,
/// rebuilt from the crate's public surface. The KEYS / ARGV composition below
/// mirrors `src/storage/layer.rs`'s `update_layer_once_lua` line for line; the
/// only deliberate difference is that `relay_ompf` is always the empty string
/// (the script's documented "no relay bytes" shape), which keeps every ARGV
/// element a UTF-8 string so a corruption is a one-slot string swap.
struct FastPathInvocation {
    keys: Vec<String>,
    argv: Vec<String>,
}

/// Number of shared-prefix ARGV elements both script modes take (server_ts,
/// client_id, client_ts, stream_maxlen, object_id, num_mutations, relay_ompf,
/// commit match key, commit record suffix, writer guard, the relay byte
/// budget's B, T, K and MIN, the call token, and the commit release ring's
/// TTL), before the mode slot.
///
/// Restated as a literal rather than read from
/// `storage::lua::UPDATE_SCRIPT_SHARED_ARGV`: this harness rebuilds the
/// invocation from the contract, and must fail if the production constant and
/// the script move together to a layout the harness no longer describes. The
/// assertion in `update_layer_fast_script_direct_invocation_commits_like_the_call_site`
/// pins that the two agree today.
const FAST_PATH_ARGV_PREFIX_LEN: usize = 16;

/// The commit release ring TTL a store built with `LayerStore::new` sends
/// (`ARGV[16]`): `commit_release_ring_ttl` of the default RPC deadline and
/// Valkey command timeout. The hand-built invocations send it so they stay
/// byte-identical to the call site's.
fn default_ring_ttl_ms() -> String {
    live_edit::storage::keys::commit_release_ring_ttl(
        live_edit::config::RPC_DEADLINE_MS_DEFAULT,
        live_edit::config::VALKEY_COMMAND_TIMEOUT_MS_DEFAULT,
    )
    .to_string()
}

/// ARGV index (0-based) of the first fast-mode mutation record: past the
/// shared prefix and the mode slot. The script's own cursor starts at the Lua
/// index one past this.
const FAST_MODE_RECORDS_START: usize = FAST_PATH_ARGV_PREFIX_LEN + 1;

/// The shared ARGV prefix from `(one-based UPDATE_SCRIPT_ARG_* slot, value)`
/// pairs, which must fill every one of the [`FAST_PATH_ARGV_PREFIX_LEN`]
/// slots exactly once: each slot is placed through its constant, as the
/// production builder (`UpdateScriptPrefix::build`) places it, so a slot that
/// moves on both sides moves here too, and a slot added to the prefix without
/// a value here fails.
fn fixed_argv(slots: [(usize, String); FAST_PATH_ARGV_PREFIX_LEN]) -> Vec<String> {
    let mut argv: Vec<Option<String>> = vec![None; FAST_PATH_ARGV_PREFIX_LEN];
    for (slot, value) in slots {
        assert!(
            argv[slot - 1].replace(value).is_none(),
            "shared ARGV slot {slot} filled twice"
        );
    }
    argv.into_iter()
        .enumerate()
        .map(|(i, v)| v.unwrap_or_else(|| panic!("shared ARGV slot {} left unset", i + 1)))
        .collect()
}

/// A fixed commit token for hand-built invocations: 32 lowercase hex digits,
/// the shape `LayerStore::update_layer_with_operation` mints. Each fixture runs
/// once against an object with no commit-token record for its `client_id`, so a
/// fixed value cannot make the script's step-1b check match a commit this test
/// did not make. Sent as the match key with an empty record suffix: the shape
/// of a call that carries no client `operation_id`.
const DIRECT_INVOCATION_COMMIT_TOKEN: &str = "0123456789abcdef0123456789abcdef";

/// Slot span of one INLINE SetField record: tag, hfield count (`1`), the
/// `value` name, the base64 value, delta_set_order, zadd_member.
const INLINE_SF_RECORD_LEN: usize = 6;

/// Relative slot offsets within an inline SetField record, used to aim the
/// corruptions below.
const SF_SLOT_TAG: usize = 0;
const SF_SLOT_HFIELD_COUNT: usize = 1;
const SF_SLOT_DELTA_SET_ORDER: usize = 4;

impl FastPathInvocation {
    fn build(
        tv: &TestValkey,
        oid: u64,
        delta: OwnedDelta,
        client_id: &str,
        client_ts: u64,
    ) -> Self {
        let schema = tv.key_schema();
        let oid_str = oid.to_string();
        let encoded: EncodedMutations = encode_delta(&Some(delta), schema, &oid_str);

        // The shared KEYS prefix through the one builder the production call
        // site uses (`lua::update_script_shared_keys`: header … commit-token
        // record, snapshot fence, writer intents, commit release record): the
        // script reads record `i`
        // at `KEYS[SHARED_KEYS + i]`, so a hand-counted prefix would aim every
        // mutation key at the wrong slot the next time a shared key is added,
        // and the validation pre-pass would refuse the commit on a key it was
        // never handed.
        let mut keys = update_script_shared_keys(schema, oid, encoded.mutation_keys.len());
        keys.extend(encoded.mutation_keys);

        // The shared ARGV prefix, each slot placed through its
        // `UPDATE_SCRIPT_ARG_*` constant as the production call site places it.
        let retention = RelayRetention::from_budget(LAYER_STREAM_MAX_BYTES_DEFAULT);
        let mut argv = fixed_argv([
            (
                UPDATE_SCRIPT_ARG_SERVER_TS,
                live_edit::domain::time::now_micros().to_string(),
            ),
            (UPDATE_SCRIPT_ARG_CLIENT_ID, client_id.to_owned()),
            (UPDATE_SCRIPT_ARG_CLIENT_TS, client_ts.to_string()),
            // Matches `layer_store_with_backend`'s maxlen, so the XADD this
            // invocation would issue is byte-identical to the production one.
            (UPDATE_SCRIPT_ARG_STREAM_MAXLEN, "4096".to_owned()),
            (UPDATE_SCRIPT_ARG_OBJECT_ID, oid_str),
            (
                UPDATE_SCRIPT_ARG_NUM_MUTATIONS,
                encoded.num_mutations.to_string(),
            ),
            (UPDATE_SCRIPT_ARG_RELAY_OMPF, String::new()),
            (
                UPDATE_SCRIPT_ARG_COMMIT_KEY,
                DIRECT_INVOCATION_COMMIT_TOKEN.to_owned(),
            ),
            // The record suffix: empty, as for a call with no operation id, so
            // the record keeps the `<token>:<seq>:<ts>` shape.
            (UPDATE_SCRIPT_ARG_RECORD_SUFFIX, String::new()),
            // The writer guard: empty, as for a call with no operation id, so
            // no writer-guard check runs.
            (UPDATE_SCRIPT_ARG_WRITER_GUARD, String::new()),
            // The relay byte budget's B, T, K and MIN: the production store's
            // (`RelayRetention::from_budget` of the default budget), so the
            // commit folds `relay_gens` as the call site's does.
            (
                UPDATE_SCRIPT_ARG_RELAY_BUDGET_BYTES,
                retention.budget_bytes.to_string(),
            ),
            (
                UPDATE_SCRIPT_ARG_RELAY_GENERATION_BYTES,
                retention.generation_bytes.to_string(),
            ),
            (
                UPDATE_SCRIPT_ARG_RELAY_GENERATIONS,
                retention.generations.to_string(),
            ),
            (
                UPDATE_SCRIPT_ARG_RELAY_MIN_ENTRIES,
                retention.min_entries.to_string(),
            ),
            // The call token, the key of the call's commit release ring
            // entry: the call's token, as the match key is without an
            // operation id.
            (
                UPDATE_SCRIPT_ARG_CALL_TOKEN,
                DIRECT_INVOCATION_COMMIT_TOKEN.to_owned(),
            ),
            // The commit release ring's TTL: the production store's default.
            (UPDATE_SCRIPT_ARG_RING_TTL_MS, default_ring_ttl_ms()),
        ]);
        // The mode slot, right after the shared prefix.
        argv.push(UpdateScriptMode::Fast.as_str().to_owned());
        argv.extend(encoded.mutation_argv);

        Self { keys, argv }
    }

    /// Absolute ARGV index of slot `slot_offset` inside inline SetField record
    /// `n` (1-based).
    ///
    /// Asserts the record actually opens with an `SF` tag, so a future change
    /// to the encoder's record layout fails HERE — loudly, naming the layout —
    /// rather than silently aiming the corruption at a neighbouring slot and
    /// leaving the test green for the wrong reason.
    fn inline_set_field_slot(&self, n: usize, slot_offset: usize) -> usize {
        let start = FAST_MODE_RECORDS_START + (n - 1) * INLINE_SF_RECORD_LEN;
        assert_eq!(
            self.argv[start], "SF",
            "expected ARGV[{start}] to open inline SetField record {n}; `lua::encode_delta`'s \
             record layout changed and this test would now corrupt the wrong slot. ARGV: {:?}",
            self.argv
        );
        start + slot_offset
    }

    /// Dispatch and decode the script's reply down to the committed sequence.
    ///
    /// b.qpf: the reply is `{new_seq, "<reason>:<hash_hex>"…}` — a flat array of
    /// bulk strings whose FIRST element is the sequence and whose tail is the
    /// owed external-value releases the Rust caller SADDs onto
    /// `le:ext:{gc}:pending`. Decoding it as a bare `u64` fails at RUNTIME
    /// (fred cannot coerce an array to an integer), so the decode has to match
    /// `update_layer_once_lua`'s. This harness's cases carry no external
    /// values, so the tail is always empty — asserted rather than ignored, so a
    /// future fixture that grows one does not silently drop the releases it
    /// owes and pass anyway.
    async fn dispatch(&self, tv: &TestValkey) -> Result<u64, fred::error::Error> {
        let reply = self.dispatch_raw(tv).await?;
        Ok(decode_fast_path_reply(&reply))
    }

    /// Dispatch and return the script's reply undecoded — for the b.75i
    /// replay case, whose `{ALREADY_LANDED, <record>}` reply is not a
    /// sequence.
    async fn dispatch_raw(&self, tv: &TestValkey) -> Result<Vec<String>, fred::error::Error> {
        let argv: Vec<fred::types::Value> = self
            .argv
            .iter()
            .map(|a| fred::types::Value::from(a.clone()))
            .collect();
        UPDATE_LAYER_FAST_SCRIPT
            .evalsha_with_reload::<Vec<String>, _, _>(
                tv.client().pool.next(),
                self.keys.clone(),
                argv,
            )
            .await
    }
}

/// ARGV index (0-based) of the commit match key (Lua `ARGV[8]`): the per-call
/// token, or `op.<operation_id>`.
const ARGV_COMMIT_TOKEN: usize = UPDATE_SCRIPT_ARG_COMMIT_KEY - 1;

/// ARGV index (0-based) of the commit record suffix (Lua `ARGV[9]`).
const ARGV_RECORD_SUFFIX: usize = UPDATE_SCRIPT_ARG_RECORD_SUFFIX - 1;

/// ARGV index (0-based) of the call token (Lua `ARGV[15]`): the key of the
/// call's commit release ring entry.
const ARGV_CALL_TOKEN: usize = UPDATE_SCRIPT_ARG_CALL_TOKEN - 1;

/// ARGV index (0-based) of the writer guard (Lua `ARGV[10]`): `""` without an
/// operation id; an unchained call's start record — `"="` when the writer's
/// record field was absent at call start, `"=" .. record` otherwise; or a
/// chained call's predecessor — `"^"` for a chain's first write,
/// `"^op." .. previous` otherwise.
const ARGV_WRITER_GUARD: usize = UPDATE_SCRIPT_ARG_WRITER_GUARD - 1;

/// The writer-guard slot of an unchained call whose writer's record field
/// held `record` when it started (`None`: absent).
fn start_record_arg(record: Option<&str>) -> String {
    format!("={}", record.unwrap_or(""))
}

/// The writer-guard slot of a chained call whose predecessor is
/// `previous_operation_id` (`None`: the chain's first write).
fn predecessor_arg(previous_operation_id: Option<&str>) -> String {
    previous_operation_id.map_or_else(|| "^".to_owned(), |previous| format!("^op.{previous}"))
}

/// ARGV index (0-based) of the server timestamp, the first fixed-prefix slot.
const ARGV_SERVER_TS: usize = 0;

/// Element 0 of the fast path's reply is the committed sequence; the rest are
/// `"<reason>:<hash_hex>"` release records (b.qpf).
///
/// Panics on a release record because every direct-invocation fixture in this
/// section writes inline values only. A record appearing here would mean the
/// fixture drifted into referencing an external value, and the assertions built
/// on the returned sequence would be reasoning about a commit that also owes
/// `ref_count` decrements nothing in this harness enqueues.
fn decode_fast_path_reply(reply: &[String]) -> u64 {
    let (seq, releases) = reply
        .split_first()
        .expect("the fast path always replies with at least the committed sequence");
    assert!(
        releases.is_empty(),
        "this fixture references no external values, so the script must owe no releases; got \
         {releases:?}"
    );
    seq.parse().unwrap_or_else(|e| {
        panic!("reply element 0 must be the decimal sequence, got {seq:?}: {e}")
    })
}

/// Two inline SetFields with distinct authoring ranks. Record 2 is the one the
/// corruption cases target; record 1 exists so "records before the bad one are
/// not committed" is an assertable claim.
fn two_inline_set_fields_delta() -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![
            SetField {
                section_id: 1,
                key_name: "before".to_owned(),
                value_or_ext_hash: vec![0xB1],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 4,
            },
            SetField {
                section_id: 1,
                key_name: "after".to_owned(),
                value_or_ext_hash: vec![0xB2],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            },
        ],
        ..Default::default()
    }
}

/// Harness-faithfulness control for section 13.
///
/// Dispatches the hand-built invocation UNCORRUPTED and asserts it commits the
/// same thing `update_layer` would: sequence 2, both fields written with the
/// folded set_orders, both index members scored at the commit sequence, one
/// stream entry at `2-0`, and its migration sentinel left absent (b.qtk).
/// Without it, a prefix built one slot short would make
/// every corruption case below pass on a `NOT_FOUND` or a coincidental
/// `ENCODING` and prove nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_direct_invocation_commits_like_the_call_site() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bph9-direct-eval-baseline").await;
    assert_eq!(
        FAST_PATH_ARGV_PREFIX_LEN, UPDATE_SCRIPT_SHARED_ARGV,
        "this harness's fixed ARGV prefix must be the production one"
    );

    let inv = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 77);
    let new_seq = inv.dispatch(&tv).await.expect(
        "the hand-built invocation must be accepted by the script exactly as the \
                 production call site's is — a failure here means this section's KEYS/ARGV \
                 contract has drifted from `update_layer_once_lua`, not that the script broke",
    );
    assert_eq!(new_seq, 2, "create_layer seeded seq=1, so the commit is 2");

    let (ks, _clock) = dump_object_keyspace(&tv, oid).await;
    // next_set_order starts 0: `before` folds to 0 + 4 = 4 (cursor -> 5),
    // `after` to 5 + 0 = 5.
    assert_eq!(field_set_orders(&ks, "before"), (4, 4));
    assert_eq!(field_set_orders(&ks, "after"), (0, 5));
    assert_eq!(
        ks.stream_payloads.len(),
        2,
        "CreateLayer's 1-0 entry plus this commit's 2-0; got {:#?}",
        ks.stream_payloads
    );
    assert_eq!(
        ks.stream_payloads[1].0, "2-0",
        "the commit's stream entry must sit at its own sequence"
    );
    // b.qtk: the landed commit left its migration sentinel absent. The
    // sentinel refuses the commit during a move only while it is absent from
    // every node, so a script that wrote it, however it spelled the key,
    // would switch the refusal off for this object's slot.
    let sentinel = tv.key_schema().migration_sentinel(&oid.to_string());
    let sentinel_exists: i64 = tv
        .client()
        .pool
        .exists(&sentinel)
        .await
        .expect("EXISTS sentinel must succeed");
    assert_eq!(
        sentinel_exists, 0,
        "the landed commit must leave its migration sentinel {sentinel} absent"
    );
}

/// b.75i: a commit records `<token>:<new_seq>:<server_ts>` under its
/// `client_id` in `obj:wtok`, and a byte-identical REPLAY of the same
/// `EVALSHA` — what fred sends when it re-issues a command that was still in
/// flight when its connection dropped — answers that commit and writes
/// nothing.
///
/// The replay is the one shape the fixed-token seam in `b_75i_tests` cannot
/// reach from outside the script: the second run carries not just the same
/// token but the same server timestamp, client timestamp and records. Pre-b.75i
/// the script had no record to consult and simply committed the delta a second
/// time at `seq + 1`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_replayed_invocation_answers_the_landed_commit_without_writing() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "b75i-direct-replay").await;

    let inv = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 77);
    let new_seq = inv.dispatch(&tv).await.expect("the first run must commit");
    assert_eq!(new_seq, 2);

    let expected_record = format!(
        "{DIRECT_INVOCATION_COMMIT_TOKEN}:2:{}",
        inv.argv[ARGV_SERVER_TS]
    );
    let recorded: Option<String> = tv
        .client()
        .pool
        .hget(tv.key_schema().obj_write_token(&oid.to_string()), "direct")
        .await
        .expect("HGET commit-token record must succeed");
    assert_eq!(
        recorded.as_deref(),
        Some(expected_record.as_str()),
        "the commit must record `<token>:<new_seq>:<server_ts>` under its client_id, built \
         from the match key in ARGV[8], the server timestamp in ARGV[1] and the empty suffix \
         in ARGV[9] (the empty writer guard in ARGV[10] adds nothing to it)"
    );

    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;
    let reply = inv
        .dispatch_raw(&tv)
        .await
        .expect("a replayed run must answer, not error");
    assert_eq!(
        reply,
        vec!["ALREADY_LANDED".to_owned(), expected_record],
        "the replay must answer the landed commit's record verbatim. A decimal here is the \
         delta committed a second time"
    );
    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(
        ks_after, ks_before,
        "the replay must write nothing — no header bump, no field rewrite, no stream entry"
    );
    assert_eq!(clock_after.last_server_ts, clock_before.last_server_ts);
}

/// b.75i / b.v6d: an empty commit match key (ARGV[8]) is refused `ENCODING`
/// before the record is read and before any write.
///
/// An empty key would prefix-match every record written with an empty key,
/// turning the at-most-once check into "anything recorded counts as mine". The
/// Rust caller always sends 32 hex digits or `op.<operation_id>`, so an empty
/// one is an encoder bug and gets the encoder-bug tag.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_empty_commit_token_is_refused_before_any_write() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "b75i-direct-empty-token").await;

    let mut inv = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 66);
    assert_eq!(
        inv.argv[ARGV_COMMIT_TOKEN], DIRECT_INVOCATION_COMMIT_TOKEN,
        "precondition: this slot must be the commit match key, or the test corrupts the wrong one"
    );
    inv.argv[ARGV_COMMIT_TOKEN] = String::new();

    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;
    let err = inv
        .dispatch(&tv)
        .await
        .expect_err("an empty commit match key must be refused");
    let details = err.details().to_string();
    assert_eq!(
        details,
        format!("ENCODING: missing commit match key (ARGV[{UPDATE_SCRIPT_ARG_COMMIT_KEY}])"),
        "the refusal must carry the `ENCODING` tag the Rust caller classifies non-retriable, and \
         name the match key slot; got {details:?}"
    );
    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(
        ks_after, ks_before,
        "an ENCODING refusal must write nothing"
    );
    assert_eq!(clock_after.last_server_ts, clock_before.last_server_ts);
}

/// An empty call token (ARGV[15]) is refused `ENCODING` before the record is
/// read and before any write.
///
/// The call token keys the call's entry in the commit release ring
/// (`obj:wrel`'s `|t|<token>`). An empty one would key every token-less
/// call's entry alike, so a call ending without proof of its outcome could
/// enqueue another call's releases. The Rust caller always sends the call's
/// token, so an empty one is an encoder bug and gets the encoder-bug tag.
/// The structural mode's twin is the `empty call token` case of
/// `structural_commit_test`'s parse-pass table.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_empty_call_token_is_refused_before_any_write() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "pgs-direct-empty-call-token").await;

    let mut inv = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 68);
    assert_eq!(
        inv.argv[ARGV_CALL_TOKEN], DIRECT_INVOCATION_COMMIT_TOKEN,
        "precondition: this slot must be the call token, or the test corrupts the wrong one"
    );
    inv.argv[ARGV_CALL_TOKEN] = String::new();

    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;
    let err = inv
        .dispatch(&tv)
        .await
        .expect_err("an empty call token must be refused");
    assert_eq!(
        err.details(),
        format!("ENCODING: missing call token (ARGV[{UPDATE_SCRIPT_ARG_CALL_TOKEN}])"),
        "the refusal must carry the `ENCODING` tag and name the call token slot"
    );
    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(
        ks_after, ks_before,
        "an ENCODING refusal must write nothing"
    );
    assert_eq!(clock_after.last_server_ts, clock_before.last_server_ts);
}

/// ARGV index (0-based) of the commit release ring's TTL (Lua `ARGV[16]`).
const ARGV_RING_TTL: usize = UPDATE_SCRIPT_ARG_RING_TTL_MS - 1;

/// The commit release ring's TTL (ARGV[16]) must be 1 to 12 decimal digits
/// with no leading zero, or the run is refused `ENCODING` before any write —
/// on an existing object, so the check is reached (an absent object answers
/// `NOT_FOUND` first), and whether or not the commit owes releases (this one
/// owes none). Each malformed value — empty, `0`, a leading zero, a
/// non-digit, 13 digits — is refused naming the slot, with the object
/// unchanged. The control: the bounds, `1` and twelve nines, commit, so the
/// check refuses no valid TTL. The structural mode's twin is the ring-TTL
/// rows of `structural_commit_test`'s parse-pass table.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_malformed_ring_ttl_is_refused_before_any_write() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let refusal =
        format!("ENCODING: bad commit release ring TTL (ARGV[{UPDATE_SCRIPT_ARG_RING_TTL_MS}])");

    for (i, bad) in ["", "0", "095000", "95e3", "1000000000000"]
        .into_iter()
        .enumerate()
    {
        let oid = create_test_layer(&ls, &sid, &format!("ring-ttl-bad-{i}")).await;
        let mut inv =
            FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 69);
        assert_eq!(
            inv.argv[ARGV_RING_TTL],
            default_ring_ttl_ms(),
            "precondition: this slot must be the ring TTL, or the test corrupts the wrong one"
        );
        inv.argv[ARGV_RING_TTL] = bad.to_owned();

        let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;
        let err = inv
            .dispatch(&tv)
            .await
            .expect_err("a malformed ring TTL must be refused");
        assert_eq!(err.details(), refusal, "TTL {bad:?}");
        let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
        assert_eq!(
            ks_after, ks_before,
            "TTL {bad:?}: an ENCODING refusal must write nothing"
        );
        assert_eq!(clock_after.last_server_ts, clock_before.last_server_ts);
    }

    for (i, good) in ["1", "999999999999"].into_iter().enumerate() {
        let oid = create_test_layer(&ls, &sid, &format!("ring-ttl-good-{i}")).await;
        let mut inv =
            FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 70);
        inv.argv[ARGV_RING_TTL] = good.to_owned();
        let new_seq = inv
            .dispatch(&tv)
            .await
            .unwrap_or_else(|e| panic!("TTL {good:?} is valid and must commit; got {e:?}"));
        assert_eq!(new_seq, 2, "TTL {good:?}");
    }
}

/// b.v6d: a missing commit record suffix (ARGV[9]) is refused `ENCODING`
/// before the record is read and before any write.
///
/// The suffix may legitimately be the empty string (a call without an
/// operation id), so only its ABSENCE is an encoder bug: an invocation built
/// with the pre-b.v6d eight-slot prefix. Reached here by truncating the ARGV
/// to that prefix — with the slot gone, `ARGV[9]` is `nil` — rather than by
/// shifting the mutation records down a slot, which would make record 1's tag
/// the suffix and fail later, on the tag walk, for a different reason.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_missing_record_suffix_is_refused_before_any_write() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bv6d-direct-missing-suffix").await;

    let mut inv = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 67);
    assert_eq!(
        inv.argv[ARGV_RECORD_SUFFIX], "",
        "precondition: this slot must be the (empty) record suffix"
    );
    inv.argv.truncate(ARGV_RECORD_SUFFIX);

    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;
    let err = inv
        .dispatch(&tv)
        .await
        .expect_err("an absent record suffix must be refused");
    assert_eq!(
        err.details(),
        format!("ENCODING: missing commit record suffix (ARGV[{UPDATE_SCRIPT_ARG_RECORD_SUFFIX}])"),
        "the refusal must carry the `ENCODING` tag and name the suffix slot"
    );
    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(
        ks_after, ks_before,
        "an ENCODING refusal must write nothing"
    );
    assert_eq!(clock_after.last_server_ts, clock_before.last_server_ts);
}

/// b.v6d: with an `op.<operation_id>` match key and a `:<token>:<digest>`
/// suffix, the script records `op.<id>:<seq>:<server_ts>:<token>:<digest>`,
/// and a later run carrying the SAME key — a different call token, server
/// timestamp and client timestamp, i.e. a client re-send — is answered from
/// that record and writes nothing. A run under a DIFFERENT operation id
/// matches nothing and commits.
///
/// The script itself never compares tokens or digests (the Rust caller turns
/// the matched record into an own-call / earlier-call / conflict verdict), so
/// what is pinned here is the script's half: the record is built by appending
/// ARGV[9] verbatim, and the step-1b match keys on ARGV[8] alone.
///
/// Every operation-key run carries a start record (ARGV[10]), as the Rust
/// caller always sends one with an operation id: the first run's field was
/// absent (`"="`), and the later runs read the first run's record. A twin of
/// the different-id run that still carries the first run's ABSENT start
/// record is overtaken: the field moved after it started.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_operation_key_records_its_suffix_and_matches_a_resend() {
    const OPERATION_KEY: &str = "op.bv6d-direct-1";
    const FIRST_TOKEN: &str = "11111111111111111111111111111111";
    const RESEND_TOKEN: &str = "22222222222222222222222222222222";
    let digest = "d".repeat(64);

    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bv6d-direct-operation").await;

    let mut first =
        FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 70);
    first.argv[ARGV_COMMIT_TOKEN] = OPERATION_KEY.to_owned();
    first.argv[ARGV_RECORD_SUFFIX] = format!(":{FIRST_TOKEN}:{digest}");
    first.argv[ARGV_WRITER_GUARD] = start_record_arg(None);
    assert_eq!(
        first
            .dispatch(&tv)
            .await
            .expect("the first run must commit"),
        2
    );

    let expected_record = format!(
        "{OPERATION_KEY}:2:{}:{FIRST_TOKEN}:{digest}",
        first.argv[ARGV_SERVER_TS]
    );
    let recorded: Option<String> = tv
        .client()
        .pool
        .hget(tv.key_schema().obj_write_token(&oid.to_string()), "direct")
        .await
        .expect("HGET commit-token record must succeed");
    assert_eq!(
        recorded.as_deref(),
        Some(expected_record.as_str()),
        "the record is `<match_key>:<seq>:<server_ts>` followed by ARGV[9] verbatim"
    );

    // A re-send: same operation key, its own call token, a later server
    // timestamp and a different client timestamp.
    let mut resend =
        FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 71);
    resend.argv[ARGV_COMMIT_TOKEN] = OPERATION_KEY.to_owned();
    resend.argv[ARGV_RECORD_SUFFIX] = format!(":{RESEND_TOKEN}:{digest}");
    resend.argv[ARGV_WRITER_GUARD] = start_record_arg(Some(&expected_record));
    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;
    let reply = resend
        .dispatch_raw(&tv)
        .await
        .expect("a re-send must be answered, not refused");
    assert_eq!(
        reply,
        vec!["ALREADY_LANDED".to_owned(), expected_record.clone()],
        "a run carrying the recorded operation key must be answered with the ORIGINAL record \
         verbatim. A decimal here is the re-send committed a second time"
    );
    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(
        ks_after, ks_before,
        "the answered re-send must write nothing"
    );
    assert_eq!(clock_after.last_server_ts, clock_before.last_server_ts);

    // A different operation id whose call started BEFORE the first run
    // committed (its start record says the field was absent) is overtaken:
    // it would apply after that later commit.
    let mut stale =
        FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 72);
    stale.argv[ARGV_COMMIT_TOKEN] = "op.bv6d-direct-2".to_owned();
    stale.argv[ARGV_RECORD_SUFFIX] = format!(":{RESEND_TOKEN}:{digest}");
    stale.argv[ARGV_WRITER_GUARD] = start_record_arg(None);
    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;
    let reply = stale
        .dispatch_raw(&tv)
        .await
        .expect("a stale start record must be answered, not refused");
    assert_eq!(
        reply,
        vec!["OVERTAKEN".to_owned()],
        "a run whose start record no longer matches the field must be refused OVERTAKEN. A \
         decimal here is a late attempt applied after its client's next commit"
    );
    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(ks_after, ks_before, "the overtaken run must write nothing");
    assert_eq!(clock_after.last_server_ts, clock_before.last_server_ts);

    // The same different operation id, started after the first run committed
    // (its start record is that run's record), is a different write intent:
    // it commits.
    let mut other =
        FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 72);
    other.argv[ARGV_COMMIT_TOKEN] = "op.bv6d-direct-2".to_owned();
    other.argv[ARGV_RECORD_SUFFIX] = format!(":{RESEND_TOKEN}:{digest}");
    other.argv[ARGV_WRITER_GUARD] = start_record_arg(Some(&expected_record));
    assert_eq!(
        other
            .dispatch(&tv)
            .await
            .expect("a different operation id must commit"),
        3,
        "an operation key matches only its own id: a different one is a fresh commit"
    );
}

// -----------------------------------------------------------------------------
// b.v6d T26 — the start record (ARGV[10]) on the script itself
// -----------------------------------------------------------------------------

/// `HGET le:{oid}:obj:wtok <client>`.
async fn wtok_field(tv: &TestValkey, oid: u64, client: &str) -> Option<String> {
    tv.client()
        .pool
        .hget(tv.key_schema().obj_write_token(&oid.to_string()), client)
        .await
        .expect("HGET commit-token record must succeed")
}

/// A hand-built operation-id run: match key `op.<operation_id>`, suffix
/// `:<token>:<digest>`, and start record `start` (`None`: the field was absent
/// when the call started).
fn operation_run(
    tv: &TestValkey,
    oid: u64,
    operation_id: &str,
    token: &str,
    start: Option<&str>,
) -> FastPathInvocation {
    let mut run = FastPathInvocation::build(tv, oid, two_inline_set_fields_delta(), "direct", 90);
    run.argv[ARGV_COMMIT_TOKEN] = format!("op.{operation_id}");
    run.argv[ARGV_RECORD_SUFFIX] = format!(":{token}:{}", "e".repeat(64));
    run.argv[ARGV_WRITER_GUARD] = start_record_arg(start);
    run
}

/// Dispatch `run`, assert it answers exactly `{"OVERTAKEN"}`, and that the
/// per-object keyspace (stream and commit clock included) is byte-identical.
async fn assert_overtaken_without_writing(tv: &TestValkey, oid: u64, run: &FastPathInvocation) {
    let (ks_before, clock_before) = dump_object_keyspace(tv, oid).await;
    let reply = run
        .dispatch_raw(tv)
        .await
        .expect("a stale start record must be answered, not refused");
    assert_eq!(
        reply,
        vec!["OVERTAKEN".to_owned()],
        "a run whose start record no longer matches its writer's record field must answer \
         OVERTAKEN. A decimal here is an attempt applied after its client's next commit"
    );
    let (ks_after, clock_after) = dump_object_keyspace(tv, oid).await;
    assert_eq!(
        ks_after, ks_before,
        "an OVERTAKEN run must write nothing — no header bump, no field, no stream entry"
    );
    assert_eq!(clock_after.last_server_ts, clock_before.last_server_ts);
}

/// T26: a run whose start record no longer matches its writer's field is
/// answered `{"OVERTAKEN"}` and writes nothing — both when the field moved to
/// another commit of the writer, and when it holds nothing although the call
/// started with a record (a checkpoint restore wiped it). A run whose start
/// record matches commits (the control: the check refuses only a moved field).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_stale_start_record_is_overtaken_without_writing() {
    const TOKEN_X: &str = "33333333333333333333333333333333";
    const TOKEN_Y: &str = "44444444444444444444444444444444";
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bv6d-t26-overtaken").await;

    let x = operation_run(&tv, oid, "bv6d-t26-x", TOKEN_X, None);
    assert_eq!(x.dispatch(&tv).await.expect("X must commit"), 2);
    let record_x = wtok_field(&tv, oid, "direct")
        .await
        .expect("precondition: X recorded its commit");

    // Y started before X committed: its start record is "absent".
    assert_overtaken_without_writing(
        &tv,
        oid,
        &operation_run(&tv, oid, "bv6d-t26-y", TOKEN_Y, None),
    )
    .await;

    // Y started after X committed: its start record matches, so it commits.
    let y = operation_run(&tv, oid, "bv6d-t26-y", TOKEN_Y, Some(&record_x));
    assert_eq!(
        y.dispatch(&tv)
            .await
            .expect("a run whose start record matches must commit"),
        3
    );
    let record_y = wtok_field(&tv, oid, "direct")
        .await
        .expect("precondition: Y recorded its commit");

    // The field wiped (as a restore that rebuilt the object without it would):
    // a run that started with Y's record is overtaken too.
    let _: i64 = tv
        .client()
        .pool
        .hdel(tv.key_schema().obj_write_token(&oid.to_string()), "direct")
        .await
        .expect("HDEL of the record field must succeed");
    assert_overtaken_without_writing(
        &tv,
        oid,
        &operation_run(&tv, oid, "bv6d-t26-z", TOKEN_X, Some(&record_y)),
    )
    .await;
}

/// T26: step 1b answers an own-key match BEFORE it checks the start record.
/// An in-call retry whose earlier attempt landed (and lost its reply) carries
/// the start record the call read BEFORE that commit, which no longer matches
/// the field — yet it must be answered `ALREADY_LANDED` with its own record,
/// not `OVERTAKEN`: otherwise every lost reply of an operation-id call would be
/// reported as "applied nothing" for an edit that landed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_own_key_match_precedes_the_start_record_check() {
    const TOKEN: &str = "55555555555555555555555555555555";
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bv6d-t26-own-key-first").await;

    let attempt = operation_run(&tv, oid, "bv6d-t26-own", TOKEN, None);
    assert_eq!(
        attempt.dispatch(&tv).await.expect("attempt 1 must commit"),
        2
    );
    let landed = wtok_field(&tv, oid, "direct")
        .await
        .expect("precondition: attempt 1 recorded its commit");

    // The retry: the same match key and the same — now stale — start record.
    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;
    let reply = attempt
        .dispatch_raw(&tv)
        .await
        .expect("the retry must be answered, not refused");
    assert_eq!(
        reply,
        vec!["ALREADY_LANDED".to_owned(), landed],
        "an own-key match must answer ALREADY_LANDED even though the start record is stale. \
         OVERTAKEN here means the start-record check runs before the own-key match"
    );
    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(ks_after, ks_before, "the answered retry must write nothing");
    assert_eq!(clock_after.last_server_ts, clock_before.last_server_ts);
}

/// T26 and b.pav group 15: the inconsistent writer-guard slot shapes are
/// encoder bugs, refused `ENCODING` before any write: ARGV[10] absent; an
/// operation suffix (ARGV[9]) with no guard (`""`); a guard — a start record
/// `"="` or a chain's first write `"^"` — with no operation suffix; and,
/// with the suffix, a guard whose tag is neither `=` nor `^`. Guessing which
/// half is right, or which guard an unknown tag meant, would either skip the
/// writer guard for an operation-id call, apply it to a call it cannot cover,
/// or run the attempt under a guard it did not ask for.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_inconsistent_writer_guard_slots_are_refused_before_any_write() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bv6d-t26-encoding").await;
    const TOKEN: &str = "66666666666666666666666666666666";

    let mut missing =
        FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 91);
    missing.argv.truncate(ARGV_WRITER_GUARD);
    let mut suffix_without_guard = operation_run(&tv, oid, "bv6d-t26-enc", TOKEN, None);
    suffix_without_guard.argv[ARGV_WRITER_GUARD] = String::new();
    let mut start_without_suffix =
        FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 92);
    start_without_suffix.argv[ARGV_WRITER_GUARD] = start_record_arg(None);
    let mut predecessor_without_suffix =
        FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 93);
    predecessor_without_suffix.argv[ARGV_WRITER_GUARD] = predecessor_arg(None);
    let mut unknown_tag = operation_run(&tv, oid, "bv6d-t26-enc", TOKEN, None);
    unknown_tag.argv[ARGV_WRITER_GUARD] = "~op.bv6d-t26-previous".to_owned();

    // The script names each slot by its step-0 constant, so the texts are
    // spelled from the Rust twins.
    let (suffix, guard) = (
        UPDATE_SCRIPT_ARG_RECORD_SUFFIX,
        UPDATE_SCRIPT_ARG_WRITER_GUARD,
    );
    let guard_without_suffix = format!(
        "ENCODING: writer guard (ARGV[{guard}]) without an operation record suffix \
         (ARGV[{suffix}])"
    );
    for (run, expected) in [
        (
            &missing,
            format!("ENCODING: missing writer guard (ARGV[{guard}])"),
        ),
        (
            &suffix_without_guard,
            format!(
                "ENCODING: operation record suffix (ARGV[{suffix}]) without a writer guard \
                 (ARGV[{guard}])"
            ),
        ),
        (&start_without_suffix, guard_without_suffix.clone()),
        (&predecessor_without_suffix, guard_without_suffix),
        (
            &unknown_tag,
            format!("ENCODING: unknown writer guard tag (ARGV[{guard}])"),
        ),
    ] {
        let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;
        let err = run
            .dispatch_raw(&tv)
            .await
            .expect_err("an inconsistent writer-guard slot must be refused");
        assert_eq!(err.details(), expected, "the refusal must name the slot");
        let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
        assert_eq!(
            ks_after, ks_before,
            "{expected}: an ENCODING refusal must write nothing"
        );
        assert_eq!(clock_after.last_server_ts, clock_before.last_server_ts);
    }
}

// -----------------------------------------------------------------------------
// b.pav — the writer guard's `^` tag: a chained call's predecessor
// -----------------------------------------------------------------------------

/// A hand-built chained run: match key `op.<operation_id>`, suffix
/// `:<token>:<digest>`, and the predecessor guard for `previous` (`None`: the
/// chain's first write).
fn chained_run(
    tv: &TestValkey,
    oid: u64,
    operation_id: &str,
    token: &str,
    previous: Option<&str>,
) -> FastPathInvocation {
    let mut run = operation_run(tv, oid, operation_id, token, None);
    run.argv[ARGV_WRITER_GUARD] = predecessor_arg(previous);
    run
}

/// Dispatch `run`, assert it answers exactly `{"PREDECESSOR", <record>}` —
/// `record` verbatim, `""` when absent — and that the per-object keyspace
/// (stream and commit clock included) is byte-identical.
async fn assert_predecessor_pending_without_writing(
    tv: &TestValkey,
    oid: u64,
    run: &FastPathInvocation,
    record: Option<&str>,
    what: &str,
) {
    let (ks_before, clock_before) = dump_object_keyspace(tv, oid).await;
    let reply = run
        .dispatch_raw(tv)
        .await
        .unwrap_or_else(|e| panic!("{what}: a pending predecessor must be answered; got {e:?}"));
    assert_eq!(
        reply,
        vec!["PREDECESSOR".to_owned(), record.unwrap_or("").to_owned()],
        "{what}: a chained run whose predecessor is not the record must answer PREDECESSOR with \
         the record it read. A decimal here is a chained commit applied out of chain order"
    );
    let (ks_after, clock_after) = dump_object_keyspace(tv, oid).await;
    assert_eq!(
        ks_after, ks_before,
        "{what}: a PREDECESSOR run must write nothing — no header bump, no field, no stream entry"
    );
    assert_eq!(clock_after.last_server_ts, clock_before.last_server_ts);
}

/// b.pav group 15, the script's PREDECESSOR step (the `^` tag): a chained run
/// commits only when its writer's record names its predecessor (`op.<previous>:…`)
/// or, for a chain's first write, is absent; otherwise it answers
/// `{"PREDECESSOR", <record or "">}` and writes nothing — for a first write
/// over a record, for a run whose predecessor has no record yet, and for one
/// whose record moved past its predecessor. A predecessor that is only a
/// prefix of the recorded operation id does not match. A chained run is never
/// answered `OVERTAKEN`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_predecessor_guard_commits_only_directly_after_the_predecessor() {
    const TOKEN_P: &str = "77777777777777777777777777777777";
    const TOKEN_S: &str = "88888888888888888888888888888888";
    const TOKEN_T: &str = "99999999999999999999999999999999";
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bpav-predecessor-guard").await;

    // S, after P, before P committed: no record yet.
    assert_predecessor_pending_without_writing(
        &tv,
        oid,
        &chained_run(&tv, oid, "bpav-s", TOKEN_S, Some("bpav-p")),
        None,
        "a predecessor with no record",
    )
    .await;
    // P, a chain's first write, over no record: commits.
    assert_eq!(
        chained_run(&tv, oid, "bpav-p", TOKEN_P, None)
            .dispatch(&tv)
            .await
            .expect("a chain's first write over no record must commit"),
        2
    );
    let record_p = wtok_field(&tv, oid, "direct")
        .await
        .expect("precondition: P recorded its commit");
    // Another first write, over P's record.
    assert_predecessor_pending_without_writing(
        &tv,
        oid,
        &chained_run(&tv, oid, "bpav-other-first", TOKEN_T, None),
        Some(&record_p),
        "a chain's first write over a record",
    )
    .await;
    // A predecessor that is a prefix of P's id is not P.
    assert_predecessor_pending_without_writing(
        &tv,
        oid,
        &chained_run(&tv, oid, "bpav-s", TOKEN_S, Some("bpav")),
        Some(&record_p),
        "a predecessor naming a prefix of the recorded operation",
    )
    .await;
    // S, after P: commits.
    assert_eq!(
        chained_run(&tv, oid, "bpav-s", TOKEN_S, Some("bpav-p"))
            .dispatch(&tv)
            .await
            .expect("a chained run whose predecessor is the record must commit"),
        3
    );
    let record_s = wtok_field(&tv, oid, "direct")
        .await
        .expect("precondition: S recorded its commit");
    // T, also after P, once the record moved past P to S.
    assert_predecessor_pending_without_writing(
        &tv,
        oid,
        &chained_run(&tv, oid, "bpav-t", TOKEN_T, Some("bpav-p")),
        Some(&record_s),
        "a predecessor the record moved past",
    )
    .await;
}

/// b.pav group 15: the own-key match precedes the PREDECESSOR step. A chained
/// run that already landed (an in-call retry, or a re-send) finds its OWN
/// record, which does not name its predecessor — yet it must be answered
/// `ALREADY_LANDED` with that record, not `PREDECESSOR`: otherwise every lost
/// reply of a chained call would be reported refused for an edit that landed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_own_key_match_precedes_the_predecessor_check() {
    const TOKEN: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab";
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bpav-own-key-first").await;

    let first = chained_run(&tv, oid, "bpav-own-p", TOKEN, None);
    assert_eq!(first.dispatch(&tv).await.expect("P must commit"), 2);
    let attempt = chained_run(&tv, oid, "bpav-own-s", TOKEN, Some("bpav-own-p"));
    assert_eq!(attempt.dispatch(&tv).await.expect("S must commit"), 3);
    let landed = wtok_field(&tv, oid, "direct")
        .await
        .expect("precondition: S recorded its commit");

    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;
    let reply = attempt
        .dispatch_raw(&tv)
        .await
        .expect("the retry must be answered, not refused");
    assert_eq!(
        reply,
        vec!["ALREADY_LANDED".to_owned(), landed],
        "an own-key match must answer ALREADY_LANDED although the record no longer names the \
         predecessor. PREDECESSOR here means the predecessor check runs before the own-key match"
    );
    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(ks_after, ks_before, "the answered retry must write nothing");
    assert_eq!(clock_after.last_server_ts, clock_before.last_server_ts);
}

/// Shared body for the three pre-pass ENCODING rejections.
///
/// Corrupts one slot of the SECOND mutation record and asserts (a) the script
/// refuses with an `ENCODING`-tagged reply naming the offending record, and
/// (b) the per-object keyspace — including the commit clock and the stream —
/// is bit-for-bit what it was before the call.
///
/// The `ENCODING` tag itself is a contract, not an incidental string: the Rust
/// caller keys on `details.starts_with("ENCODING")` to surface a non-retriable
/// `Internal` (`src/storage/layer.rs`). Retrying a malformed record would burn
/// the whole 128-attempt budget against an input no retry can change.
async fn assert_corrupt_second_record_rejected_before_any_write(
    layer_name: &str,
    slot_offset: usize,
    corrupt_value: &str,
    expected_detail: &str,
) {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, layer_name).await;

    let mut inv = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 88);
    let target = inv.inline_set_field_slot(2, slot_offset);
    inv.argv[target] = corrupt_value.to_owned();

    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;

    let err = inv
        .dispatch(&tv)
        .await
        .expect_err("a malformed mutation record must be refused by the validation pre-pass");
    let details = err.details().to_string();
    assert!(
        details.starts_with("ENCODING"),
        "an encoder-shaped fault must carry the `ENCODING` tag the Rust caller matches on to \
         classify it non-retriable; got: {details:?}"
    );
    assert!(
        details.contains(expected_detail),
        "the reply must name what was wrong and which record it was on, so a real encoder bug \
         is diagnosable from the log line alone; expected it to contain {expected_detail:?}, \
         got: {details:?}"
    );

    // The b.ph9 property: record 1 was well-formed and precedes the bad one,
    // and must still not have been written. Pre-fix it was HSET with
    // `version = new_seq` above a header whose `seq` never advanced.
    let first_field_exists: u64 = tv
        .client()
        .pool
        .exists(tv.key_schema().obj_field(&oid.to_string(), "1", "before"))
        .await
        .expect("EXISTS must succeed");
    assert_eq!(
        first_field_exists, 0,
        "the record BEFORE the malformed one must not have been written — the validation \
         pre-pass issues zero redis.calls, so it decides this with the keyspace untouched"
    );

    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(
        ks_after, ks_before,
        "a pre-pass rejection must leave the per-object keyspace bit-for-bit unchanged, \
         stream included.\n\nAFTER: {ks_after:#?}\n\nBEFORE: {ks_before:#?}"
    );
    assert_eq!(
        clock_after.last_server_ts, clock_before.last_server_ts,
        "a pre-pass rejection must not restamp obj:hdr last_server_ts (the keyspace \
         comparison above strips the timestamp columns)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_unknown_mutation_tag_rejected_before_any_write() {
    assert_corrupt_second_record_rejected_before_any_write(
        "bph9-direct-eval-bad-tag",
        SF_SLOT_TAG,
        "QQ",
        "unknown mutation tag QQ",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_non_numeric_hfield_count_rejected_before_any_write() {
    assert_corrupt_second_record_rejected_before_any_write(
        "bph9-direct-eval-bad-count",
        SF_SLOT_HFIELD_COUNT,
        "not-a-number",
        "malformed value-hfield count in mutation record 2",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_non_numeric_delta_set_order_rejected_before_any_write() {
    assert_corrupt_second_record_rejected_before_any_write(
        "bph9-direct-eval-bad-set-order",
        SF_SLOT_DELTA_SET_ORDER,
        "not-a-number",
        "malformed delta_set_order in mutation record 2",
    )
    .await;
}

/// Shared body for the fast-mode KEYS-count refusals: one key too few or too
/// many for the invocation's two records.
///
/// `corrupt` edits the hand-built KEYS; the script must answer `ENCODING`
/// naming the count it got and the count two records need, and leave the
/// per-object keyspace bit-for-bit unchanged. Without the count check a
/// short list reaches a `TYPE` on a nil key (an untagged script error), and an
/// extra trailing key is simply ignored: the commit lands.
async fn assert_fast_mode_keys_count_mismatch_rejected_before_any_write(
    layer_name: &str,
    corrupt: fn(&mut Vec<String>),
) {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, layer_name).await;

    let mut inv = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 89);
    let expected_len = inv.keys.len();
    corrupt(&mut inv.keys);
    assert_ne!(
        inv.keys.len(),
        expected_len,
        "precondition: the corruption must change the KEYS count"
    );
    let expected_detail = format!(
        "ENCODING: {} KEYS for 2 mutation record(s); expected {expected_len}",
        inv.keys.len()
    );

    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;

    let err = inv
        .dispatch(&tv)
        .await
        .expect_err("a KEYS count that does not match the records must be refused");
    assert_eq!(
        err.details(),
        expected_detail,
        "the refusal must carry the `ENCODING` tag the Rust caller classifies non-retriable \
         and name both counts"
    );

    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(
        ks_after, ks_before,
        "a KEYS-count refusal must leave the per-object keyspace bit-for-bit unchanged, \
         stream included.\n\nAFTER: {ks_after:#?}\n\nBEFORE: {ks_before:#?}"
    );
    assert_eq!(
        clock_after.last_server_ts, clock_before.last_server_ts,
        "a KEYS-count refusal must not restamp obj:hdr last_server_ts"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_one_key_short_rejected_before_any_write() {
    assert_fast_mode_keys_count_mismatch_rejected_before_any_write(
        "fast-keys-count-short",
        |keys| {
            keys.pop();
        },
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_one_key_extra_rejected_before_any_write() {
    // A copy of the last record's key: the object's own hash tag, so a
    // cluster routes it with the rest, and every record still reads its own
    // key.
    assert_fast_mode_keys_count_mismatch_rejected_before_any_write(
        "fast-keys-count-extra",
        |keys| {
            let extra = keys.last().expect("a record key").clone();
            keys.push(extra);
        },
    )
    .await;
}

/// Shared body for the migration-sentinel refusals (b.qtk): `KEYS[10]` is
/// replaced by `replace`'s key. The script's step 0 checks the slot holds the
/// migration sentinel of `KEYS[1]`'s hash tag before any read or write — on a
/// cluster that undeclared sentinel is what keeps the commit from starting
/// while its slot migrates, so a caller that drops it must fail on every
/// store, this standalone one included. Answers `ENCODING` and leaves the
/// per-object keyspace bit-for-bit unchanged. The uncorrupted invocation's
/// commit is pinned by
/// `update_layer_fast_script_direct_invocation_commits_like_the_call_site`.
async fn assert_migration_sentinel_refused_before_any_write(
    layer_name: &str,
    replace: fn(&KeySchema, u64) -> String,
) {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, layer_name).await;

    let mut inv = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 91);
    assert_eq!(
        inv.keys[UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL - 1],
        tv.key_schema().migration_sentinel(&oid.to_string()),
        "precondition: the builder put the sentinel in its slot"
    );
    inv.keys[UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL - 1] = replace(tv.key_schema(), oid);

    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;

    let err = inv
        .dispatch(&tv)
        .await
        .expect_err("a commit without its migration sentinel must be refused");
    assert_eq!(
        err.details(),
        "ENCODING: KEYS[10] is not the migration sentinel of KEYS[1]'s hash tag",
        "the refusal must carry the `ENCODING` tag and name the slot"
    );

    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(
        ks_after, ks_before,
        "a sentinel refusal must leave the per-object keyspace bit-for-bit unchanged, stream \
         included.\n\nAFTER: {ks_after:#?}\n\nBEFORE: {ks_before:#?}"
    );
    assert_eq!(
        clock_after.last_server_ts, clock_before.last_server_ts,
        "a sentinel refusal must not restamp obj:hdr last_server_ts"
    );
}

/// A sentinel of another hash tag (on a cluster, another slot) in `KEYS[10]`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_without_its_migration_sentinel_is_refused_before_any_write() {
    assert_migration_sentinel_refused_before_any_write("fast-sentinel-other-tag", |keys, oid| {
        keys.migration_sentinel(&format!("{oid}0"))
    })
    .await;
}

/// A key of the object's own tag that is not its sentinel in `KEYS[10]`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_with_another_key_in_the_sentinel_slot_is_refused_before_any_write(
) {
    assert_migration_sentinel_refused_before_any_write(
        "fast-sentinel-not-a-sentinel",
        |keys, oid| keys.obj_header(&oid.to_string()),
    )
    .await;
}

/// A non-numeric `num_mutations` (ARGV[6]) is refused `ENCODING` before any
/// write: the KEYS count check needs the number, and the record walk is
/// bounded by it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_malformed_num_mutations_rejected_before_any_write() {
    const ARGV_NUM_MUTATIONS: usize = 5;
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "fast-keys-count-bad-num").await;

    let mut inv = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 90);
    assert_eq!(
        inv.argv[ARGV_NUM_MUTATIONS], "2",
        "precondition: this slot must be num_mutations, or the test corrupts the wrong one"
    );
    inv.argv[ARGV_NUM_MUTATIONS] = "x".to_owned();
    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;

    let err = inv
        .dispatch(&tv)
        .await
        .expect_err("a malformed num_mutations must be refused");
    assert_eq!(
        err.details(),
        format!("ENCODING: malformed num_mutations (ARGV[{UPDATE_SCRIPT_ARG_NUM_MUTATIONS}])")
    );

    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(
        ks_after, ks_before,
        "an ENCODING refusal must write nothing"
    );
    assert_eq!(clock_after.last_server_ts, clock_before.last_server_ts);
}

/// The remaining tag transition: `ST → SF`.
///
/// Section 1c drives SF→SF, SF→ST and ST→ST, which is everything an
/// `OwnedDelta` can express — `encode_delta` emits every SetField before every
/// time sample, so no delta produces a SetField record AFTER a time-sample
/// record. The script's contract is nonetheless tag-order-independent (both
/// walks dispatch per record on the tag they read), and ST→SF is the ONLY
/// transition in which a wrong `ST` stride is detectable: get it wrong when
/// the ST records are last and the mis-parked cursor is simply never read
/// again.
///
/// So this case hand-assembles `SF ST SF` at the ARGV level. A one-slot ST
/// drift makes the pre-pass read record 3's tag from the wrong slot and the
/// dispatch fails with `ENCODING: unknown mutation tag`; a mis-indexed
/// `stored_set_orders` stash gives record 3 record 1's fold. Both are pinned
/// by the concrete set_order assertions below.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_interleaved_sf_st_sf_records_apply_in_record_order() {
    const SECTION_ID: u64 = 9;
    const SAMPLE_TIME: f64 = 3.5;

    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bph9-interleaved-records").await;

    let schema = tv.key_schema();
    let oid_str = oid.to_string();
    let sid_str = SECTION_ID.to_string();

    // The shared prefix through `update_script_shared_keys`, the builder the
    // production call site uses: record `i` is read at `KEYS[SHARED_KEYS + i]`,
    // so the per-record keys below must follow exactly that prefix or the
    // record walk aims short and the pre-pass refuses the commit.
    let mut keys = update_script_shared_keys(schema, oid, 3);
    keys.extend([
        // Per-mutation keys, in record order.
        schema.obj_field(&oid_str, &sid_str, "one"),
        schema.obj_time_sample(&oid_str, &sid_str, SAMPLE_TIME),
        schema.obj_field(&oid_str, &sid_str, "two"),
    ]);

    // The shared prefix, each slot through its `UPDATE_SCRIPT_ARG_*` constant.
    let retention = RelayRetention::from_budget(LAYER_STREAM_MAX_BYTES_DEFAULT);
    let prefix = fixed_argv([
        (
            UPDATE_SCRIPT_ARG_SERVER_TS,
            live_edit::domain::time::now_micros().to_string(),
        ),
        (UPDATE_SCRIPT_ARG_CLIENT_ID, "interleaved".to_owned()),
        (UPDATE_SCRIPT_ARG_CLIENT_TS, "99".to_owned()),
        (UPDATE_SCRIPT_ARG_STREAM_MAXLEN, "4096".to_owned()),
        (UPDATE_SCRIPT_ARG_OBJECT_ID, oid_str.clone()),
        (UPDATE_SCRIPT_ARG_NUM_MUTATIONS, "3".to_owned()),
        (UPDATE_SCRIPT_ARG_RELAY_OMPF, String::new()),
        (
            UPDATE_SCRIPT_ARG_COMMIT_KEY,
            DIRECT_INVOCATION_COMMIT_TOKEN.to_owned(),
        ),
        // The commit record suffix: empty, no operation id.
        (UPDATE_SCRIPT_ARG_RECORD_SUFFIX, String::new()),
        // The writer guard: empty, no operation id.
        (UPDATE_SCRIPT_ARG_WRITER_GUARD, String::new()),
        // The relay byte budget's B, T, K and MIN: a store's default.
        (
            UPDATE_SCRIPT_ARG_RELAY_BUDGET_BYTES,
            retention.budget_bytes.to_string(),
        ),
        (
            UPDATE_SCRIPT_ARG_RELAY_GENERATION_BYTES,
            retention.generation_bytes.to_string(),
        ),
        (
            UPDATE_SCRIPT_ARG_RELAY_GENERATIONS,
            retention.generations.to_string(),
        ),
        (
            UPDATE_SCRIPT_ARG_RELAY_MIN_ENTRIES,
            retention.min_entries.to_string(),
        ),
        // The call token: the call's token, as the match key is.
        (
            UPDATE_SCRIPT_ARG_CALL_TOKEN,
            DIRECT_INVOCATION_COMMIT_TOKEN.to_owned(),
        ),
        // The commit release ring's TTL: a store's default.
        (UPDATE_SCRIPT_ARG_RING_TTL_MS, default_ring_ttl_ms()),
    ]);
    let argv: Vec<fred::types::Value> = prefix
        .into_iter()
        .chain([
            // The mode slot.
            UpdateScriptMode::Fast.as_str().to_owned(),
            // Record 1 — SF, authoring rank 4.
            "SF".to_owned(),
            "1".to_owned(),
            "value".to_owned(),
            BASE64.encode([0x11]),
            "4".to_owned(),
            KeySchema::field_member(&sid_str, "one"),
            // Record 2 — ST. One slot SHORTER than an SF record: no
            // delta_set_order.
            "ST".to_owned(),
            "1".to_owned(),
            "value".to_owned(),
            BASE64.encode([0x22]),
            KeySchema::time_sample_member(&sid_str, SAMPLE_TIME),
            // Record 3 — SF again, authoring rank 0. Reachable only if the ST
            // stride above was walked correctly.
            "SF".to_owned(),
            "1".to_owned(),
            "value".to_owned(),
            BASE64.encode([0x33]),
            "0".to_owned(),
            KeySchema::field_member(&sid_str, "two"),
        ])
        .map(fred::types::Value::from)
        .collect();

    let reply: Vec<String> = UPDATE_LAYER_FAST_SCRIPT
        .evalsha_with_reload::<Vec<String>, _, _>(tv.client().pool.next(), keys, argv)
        .await
        .expect(
            "an SF ST SF record stream must commit: both of the script's walks dispatch per \
             record on the tag they read, so record order is not part of the contract. An \
             `ENCODING: unknown mutation tag` here means the ST stride is wrong and the walk \
             lost sync before record 3",
        );
    let new_seq = decode_fast_path_reply(&reply);
    assert_eq!(new_seq, 2);

    let (ks, _clock) = dump_object_keyspace(&tv, oid).await;
    // next_set_order starts 0. Record 1 folds to 0 + 4 = 4 (cursor -> 5); the
    // ST record does not touch the cursor; record 3 folds to 5 + 0 = 5.
    assert_eq!(
        field_set_orders(&ks, "one"),
        (4, 4),
        "record 1: authoring rank verbatim, cumulative folded from cursor 0"
    );
    assert_eq!(
        field_set_orders(&ks, "two"),
        (0, 5),
        "record 3 must consume ITS OWN stashed fold (cursor 5 + rank 0), not record 1's — a \
         stash indexed by anything other than the record ordinal lands here"
    );
    assert_eq!(
        time_sample_slots(&ks, SECTION_ID, SAMPLE_TIME),
        &HashSnapshot(vec![
            slot("value", BASE64.encode([0x22])),
            slot("version", "2"),
        ]),
        "the interleaved time sample must carry its own value and the commit version"
    );
    let header_next_set_order: String = tv
        .client()
        .pool
        .hget(schema.obj_header(&oid_str), "next_set_order")
        .await
        .expect("HGET next_set_order must succeed");
    assert_eq!(
        header_next_set_order, "6",
        "the header must persist the cursor the pre-pass folded across all three records \
         (5 + 1), not a value recomputed by the apply pass"
    );
}

// =============================================================================
// 13b. b.y3m Epic 3 Task 3 / plan b.dkt Task 11 (W1) — the script's
// snapshot-fence check (step 1d, Spec SDD SR-5.4) and its slot layout.
//
// Driven through the same hand-built invocation as section 13, so the reply is
// the script's own: `{FENCED, <retry_after_ms>}` is decoded by no Rust code
// here. A live reader is either a real lease granted by
// `snapshot_fence::acquire_fence`, or a staged `r:<token>` field whose
// deadline is on the store's own clock (read by the same `TIME` expression
// the script uses), so "live" and "expired" are set, not waited for. The
// structural mode's twin of each is in `structural_commit_test.rs` and
// `structural_guard_parity_test.rs`; the writer-side behaviour built on the
// reply — the uncounted wait, the cap — is in `snapshot_fence_writer_test.rs`.
// =============================================================================

/// The store clock in whole ms, read as the fence scripts read it.
async fn store_now_ms(tv: &TestValkey, oid: u64) -> i64 {
    fred::types::scripts::Script::from_lua(
        "local t = redis.call('TIME') \
         return tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)",
    )
    .evalsha_with_reload::<i64, _, _>(
        tv.client().pool.next(),
        vec![tv.key_schema().obj_snapshot_fence(&oid.to_string())],
        Vec::<String>::new(),
    )
    .await
    .expect("TIME must answer")
}

/// Stage reader fields on `oid`'s fence key: `(field, deadline − now ms)`.
async fn stage_readers(tv: &TestValkey, oid: u64, readers: &[(&str, i64)]) -> i64 {
    let now = store_now_ms(tv, oid).await;
    let pairs: Vec<(String, String)> = readers
        .iter()
        .map(|(f, offset)| ((*f).to_owned(), (now + offset).to_string()))
        .collect();
    let _: () = tv
        .client()
        .pool
        .hset(tv.key_schema().obj_snapshot_fence(&oid.to_string()), pairs)
        .await
        .expect("staging the fence must succeed");
    now
}

async fn fence_image(tv: &TestValkey, oid: u64) -> HashMap<String, String> {
    tv.client()
        .pool
        .hgetall(tv.key_schema().obj_snapshot_fence(&oid.to_string()))
        .await
        .expect("HGETALL fence must succeed")
}

/// Assert `reply` is `{FENCED, <ms>}` with `ms` in `range`; return `ms`.
fn assert_fenced_reply(reply: &[String], range: std::ops::RangeInclusive<i64>) -> i64 {
    assert_eq!(reply.len(), 2, "{{FENCED, <ms>}}; got {reply:?}");
    assert_eq!(reply[0], "FENCED", "got {reply:?}");
    let hint: i64 = reply[1].parse().expect("the hint is a decimal");
    assert!(
        range.contains(&hint),
        "the hint is the latest live deadline − now, at least 1: {hint} ∉ {range:?}"
    );
    hint
}

/// A live reader lease refuses the commit `{FENCED, <ms>}` — the LATEST live
/// deadline minus the script's `now`, whole ms — and writes nothing: the
/// object's keyspace, its commit clock, the commit-token record and the fence
/// key are all as they were. An expired reader field beside the live ones
/// does not shorten the hint.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_under_a_live_fence_answers_fenced_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "y3m-direct-fenced").await;
    let now = stage_readers(
        &tv,
        oid,
        &[
            ("r:early", 20_000),
            ("r:late", 60_000),
            ("r:dead", -5),
            ("epoch_start", -1_000),
        ],
    )
    .await;
    let latest = now + 60_000;
    let fence_before = fence_image(&tv, oid).await;
    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;
    let wtok_key = tv.key_schema().obj_write_token(&oid.to_string());

    let t0 = store_now_ms(&tv, oid).await;
    let reply = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 5)
        .dispatch_raw(&tv)
        .await
        .expect("a fenced run answers, it does not error");
    let t1 = store_now_ms(&tv, oid).await;

    assert_fenced_reply(&reply, latest - t1..=latest - t0);
    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(ks_after, ks_before, "a FENCED run writes nothing");
    assert_eq!(clock_after.last_server_ts, clock_before.last_server_ts);
    let wtok_exists: u64 = tv.client().pool.exists(&wtok_key).await.unwrap();
    assert_eq!(wtok_exists, 0, "no commit-token record is written");
    assert_eq!(
        fence_image(&tv, oid).await,
        fence_before,
        "the commit never writes the fence"
    );
}

/// T1, fast mode, on a real lease (`snapshot_fence::acquire_fence`): the run
/// answers `{FENCED, <ms>}` with `1 ≤ ms ≤` what is left of the lease and
/// writes nothing; once the reader releases — `VERIFIED`, itself proof that
/// nothing landed under it — the same run commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_under_a_real_lease_answers_fenced_then_commits_after_release() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "y3m-direct-real-lease").await;
    let size = LayerSize {
        items: 150_216,
        bytes: 1_799_752,
    };
    let lease = match fence::acquire_fence(tv.client(), tv.key_schema(), oid, size)
        .await
        .expect("acquire_fence must answer")
    {
        AcquireVerdict::Granted(lease) => lease,
        other => panic!("the fence must be granted; got {other:?}"),
    };
    let deadline: i64 = fence_image(&tv, oid)
        .await
        .into_iter()
        .find(|(field, _)| field.starts_with("r:"))
        .expect("the lease's reader field")
        .1
        .parse()
        .expect("a deadline");
    let (ks_before, _) = dump_object_keyspace(&tv, oid).await;
    let inv = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 5);

    let t0 = store_now_ms(&tv, oid).await;
    let reply = inv
        .dispatch_raw(&tv)
        .await
        .expect("a fenced run answers, it does not error");

    assert_fenced_reply(&reply, 1..=deadline - t0);
    assert_eq!(
        dump_object_keyspace(&tv, oid).await.0,
        ks_before,
        "a FENCED run writes nothing"
    );
    assert_eq!(
        lease
            .release(Some(size))
            .await
            .expect("release must answer"),
        ReleaseVerdict::Verified,
        "nothing landed under the lease"
    );
    assert_eq!(
        inv.dispatch(&tv)
            .await
            .expect("with the lease released the run commits"),
        2
    );
}

/// The fence check sits after the commit-token guard (step 1b): a run whose
/// commit already landed is reported `ALREADY_LANDED` under a live fence, not
/// told to wait for a reader.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_reports_an_already_landed_commit_under_a_live_fence() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "y3m-direct-landed-fenced").await;
    let inv = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 5);
    assert_eq!(inv.dispatch(&tv).await.expect("the first run commits"), 2);
    stage_readers(&tv, oid, &[("r:reader", 60_000)]).await;
    let (ks_before, _) = dump_object_keyspace(&tv, oid).await;

    let reply = inv.dispatch_raw(&tv).await.expect("the replay answers");

    assert_eq!(
        reply.first().map(String::as_str),
        Some("ALREADY_LANDED"),
        "an already-landed commit is reported under a fence (step 1b before 1d); got {reply:?}"
    );
    assert_eq!(dump_object_keyspace(&tv, oid).await.0, ks_before);
}

/// The fence check sits after the superseded guard (step 1c): a superseded
/// object is refused `SUPERSEDED` at once, never made to wait for a reader
/// first (the b.v6d order: own-key match → SUPERSEDED → FENCED).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_refuses_superseded_before_fenced() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "y3m-direct-superseded-fenced").await;
    stage_readers(&tv, oid, &[("r:reader", 60_000)]).await;
    let _: () = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_header(&oid.to_string()),
            ("superseded", "1"),
        )
        .await
        .unwrap();

    let err = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 5)
        .dispatch_raw(&tv)
        .await
        .expect_err("a superseded object is refused");

    assert!(
        err.details().starts_with("SUPERSEDED"),
        "SUPERSEDED, not FENCED; got {err:?}"
    );
}

/// A fence key with no live reader — every lease expired, only the epoch and
/// gap bookkeeping left (a layer fenced within its `key_linger(L)`) — refuses
/// nothing: the run commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_commits_past_a_fence_key_with_no_live_reader() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "y3m-direct-lingering").await;
    stage_readers(
        &tv,
        oid,
        &[
            ("r:expired", -1),
            ("epoch_start", -5_000),
            ("gap_until", 30_000),
        ],
    )
    .await;

    let new_seq = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "direct", 5)
        .dispatch(&tv)
        .await
        .expect("no live reader: the run commits");

    assert_eq!(new_seq, 2);
}

/// T4, runtime half (the build-time half is `storage::lua`'s `const` check,
/// which refuses to compile on the same drift): the script reads every shared
/// `KEYS` / `ARGV` slot through a named step-0 local, and the Rust builder
/// places each through its `storage::lua::UPDATE_SCRIPT_*` twin — the two sets
/// must carry the same numbers, name for name, and the same two lengths, or
/// every producer aims at the wrong slot. Parses the script's step-0 block.
#[test]
fn update_layer_fast_script_slot_locals_match_the_rust_constants() {
    const SCRIPT: &str = include_str!("../../../src/storage/scripts/update_layer_fast.lua");
    let lua: HashMap<String, usize> = SCRIPT
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("local ")?;
            let (name, value) = rest.split_once('=')?;
            let name = name.trim();
            let is_slot = name.starts_with("KEY_")
                || name.starts_with("ARG_")
                || name == "SHARED_KEYS"
                || name == "SHARED_ARGV";
            if !is_slot {
                return None;
            }
            let value = value.trim().parse().ok()?;
            Some((name.to_owned(), value))
        })
        .collect();
    let rust: HashMap<String, usize> = [
        ("KEY_HEADER", UPDATE_SCRIPT_KEY_HEADER),
        ("KEY_FIELDS_INDEX", UPDATE_SCRIPT_KEY_FIELDS_INDEX),
        ("KEY_TS_INDEX", UPDATE_SCRIPT_KEY_TS_INDEX),
        ("KEY_STREAM", UPDATE_SCRIPT_KEY_STREAM),
        ("KEY_EXT_INDEX", UPDATE_SCRIPT_KEY_EXT_INDEX),
        ("KEY_COMMIT_TOKEN", UPDATE_SCRIPT_KEY_COMMIT_TOKEN),
        ("KEY_FENCE", UPDATE_SCRIPT_KEY_FENCE),
        ("KEY_INTENT", UPDATE_SCRIPT_KEY_INTENT),
        ("KEY_COMMIT_RELEASE", UPDATE_SCRIPT_KEY_COMMIT_RELEASE),
        (
            "KEY_MIGRATION_SENTINEL",
            UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL,
        ),
        ("SHARED_KEYS", UPDATE_SCRIPT_SHARED_KEYS),
        ("ARG_SERVER_TS", UPDATE_SCRIPT_ARG_SERVER_TS),
        ("ARG_CLIENT_ID", UPDATE_SCRIPT_ARG_CLIENT_ID),
        ("ARG_CLIENT_TS", UPDATE_SCRIPT_ARG_CLIENT_TS),
        ("ARG_STREAM_MAXLEN", UPDATE_SCRIPT_ARG_STREAM_MAXLEN),
        ("ARG_OBJECT_ID", UPDATE_SCRIPT_ARG_OBJECT_ID),
        ("ARG_NUM_MUTATIONS", UPDATE_SCRIPT_ARG_NUM_MUTATIONS),
        ("ARG_RELAY_OMPF", UPDATE_SCRIPT_ARG_RELAY_OMPF),
        ("ARG_COMMIT_KEY", UPDATE_SCRIPT_ARG_COMMIT_KEY),
        ("ARG_RECORD_SUFFIX", UPDATE_SCRIPT_ARG_RECORD_SUFFIX),
        ("ARG_WRITER_GUARD", UPDATE_SCRIPT_ARG_WRITER_GUARD),
        (
            "ARG_RELAY_BUDGET_BYTES",
            UPDATE_SCRIPT_ARG_RELAY_BUDGET_BYTES,
        ),
        (
            "ARG_RELAY_GENERATION_BYTES",
            UPDATE_SCRIPT_ARG_RELAY_GENERATION_BYTES,
        ),
        ("ARG_RELAY_GENERATIONS", UPDATE_SCRIPT_ARG_RELAY_GENERATIONS),
        ("ARG_RELAY_MIN_ENTRIES", UPDATE_SCRIPT_ARG_RELAY_MIN_ENTRIES),
        ("ARG_CALL_TOKEN", UPDATE_SCRIPT_ARG_CALL_TOKEN),
        ("ARG_RING_TTL_MS", UPDATE_SCRIPT_ARG_RING_TTL_MS),
        ("SHARED_ARGV", UPDATE_SCRIPT_SHARED_ARGV),
    ]
    .into_iter()
    .map(|(n, v)| (n.to_owned(), v))
    .collect();

    assert_eq!(
        lua, rust,
        "update_layer_fast.lua's step-0 slot locals and storage::lua's UPDATE_SCRIPT_* \
         constants must be the same set with the same numbers"
    );
}

/// T4: `update_script_shared_keys` puts each shared key in its
/// `UPDATE_SCRIPT_KEY_*` slot and nothing else: the fence is `KEYS[7]`, the
/// intent key `KEYS[8]`, the commit release ring `KEYS[9]`, the object's
/// migration sentinel `KEYS[10]` (b.qtk), so the first key past the prefix
/// is `KEYS[UPDATE_SCRIPT_SHARED_KEYS + 1]` = `KEYS[11]`.
#[test]
fn update_script_shared_keys_places_each_key_in_its_slot() {
    let schema = KeySchema::new("t:");
    let oid = 42_u64;
    let o = oid.to_string();
    let keys = update_script_shared_keys(&schema, oid, 3);

    assert_eq!(keys.len(), UPDATE_SCRIPT_SHARED_KEYS);
    for (slot, expected) in [
        (UPDATE_SCRIPT_KEY_HEADER, schema.obj_header(&o)),
        (UPDATE_SCRIPT_KEY_FIELDS_INDEX, schema.obj_fields_index(&o)),
        (
            UPDATE_SCRIPT_KEY_TS_INDEX,
            schema.obj_time_samples_index(&o),
        ),
        (UPDATE_SCRIPT_KEY_STREAM, schema.layer_stream(oid)),
        (UPDATE_SCRIPT_KEY_EXT_INDEX, schema.obj_ext_index(&o)),
        (UPDATE_SCRIPT_KEY_COMMIT_TOKEN, schema.obj_write_token(&o)),
        (UPDATE_SCRIPT_KEY_FENCE, schema.obj_snapshot_fence(&o)),
        (UPDATE_SCRIPT_KEY_INTENT, schema.obj_snapshot_intent(&o)),
        (
            UPDATE_SCRIPT_KEY_COMMIT_RELEASE,
            schema.obj_write_releases(&o),
        ),
        (
            UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL,
            schema.migration_sentinel(&o),
        ),
    ] {
        assert_eq!(keys[slot - 1], expected, "KEYS[{slot}]");
    }
    assert_eq!(
        (
            UPDATE_SCRIPT_KEY_FENCE,
            UPDATE_SCRIPT_KEY_INTENT,
            UPDATE_SCRIPT_KEY_COMMIT_RELEASE,
            UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL,
        ),
        (7, 8, 9, 10)
    );
    assert_eq!(UPDATE_SCRIPT_SHARED_KEYS, 10);
}

/// T4, through the production builders: the `EVALSHA` each script mode
/// actually sends — fast mode from `update_layer_once_lua`, structural mode
/// from `commit_structural` on the cached structural route — carries the
/// fence and intent keys at `KEYS[7]` / `KEYS[8]`, the commit release
/// ring at `KEYS[9]` and the object's migration sentinel at `KEYS[10]`
/// (after the header at `KEYS[1]`), and structural mode's sections index at
/// `KEYS[UPDATE_SCRIPT_SHARED_KEYS + 1]`; the call token at
/// `ARGV[UPDATE_SCRIPT_ARG_CALL_TOKEN]`, the call's 32-hex commit token —
/// without an operation id the same token as the commit match key at
/// `ARGV[8]`; and the store's commit release ring TTL at `ARGV[16]`, the last
/// shared slot. Read off the store's own `MONITOR`
/// feed (`args[0]` is the command, `[1]` the SHA, `[2]` the key count).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_script_modes_send_the_shared_prefix_with_the_fence_at_7_the_intent_at_8_and_secs_after_it(
) {
    use crate::storage::snapshot_read_atomic_test::{create_cmd, structural, ROOT};
    use crate::storage::snapshot_read_support::{monitor_fence, test_valkey_addr, MonitorTap};

    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "w1-prefix-on-the-wire").await;
    let o = oid.to_string();
    let schema = tv.key_schema();
    // Warm both script modes (one script) and the structural cache's
    // cold-start scripts, so the tap sees no `NOSCRIPT` retry.
    ls.update_layer_relayed(
        &sid,
        "w1-prefix-on-the-wire",
        oid,
        Some(structural(vec![create_cmd(ROOT, 700, "warm")])),
        "w1",
        1,
    )
    .await
    .expect("the warm-up structural commit must land");
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    ls.update_layer_relayed(
        &sid,
        "w1-prefix-on-the-wire",
        oid,
        Some(delta_with_set_field_inline(1, "color", vec![1], 0)),
        "w1",
        2,
    )
    .await
    .expect("the fast-mode commit must land");
    ls.update_layer_relayed(
        &sid,
        "w1-prefix-on-the-wire",
        oid,
        Some(structural(vec![create_cmd(ROOT, 701, "child")])),
        "w1",
        3,
    )
    .await
    .expect("the structural-mode commit must land");
    let marker = monitor_fence(tv.prefix());
    let _: i64 = tv.client().pool.exists(&marker).await.unwrap();
    let seen = tap.drain_until(&marker).await;

    let header = schema.obj_header(&o);
    // `args` index of the last shared key, `KEYS[UPDATE_SCRIPT_SHARED_KEYS]`.
    let last_shared_key = 2 + UPDATE_SCRIPT_SHARED_KEYS;
    let evals: Vec<&Vec<String>> = seen
        .iter()
        .filter(|c| {
            c.name() == "EVALSHA"
                && c.args.get(1).map(String::as_str) == Some(UPDATE_LAYER_FAST_SCRIPT.sha1())
                && c.args.get(2 + UPDATE_SCRIPT_KEY_HEADER) == Some(&header)
        })
        .map(|c| &c.args)
        .collect();
    assert_eq!(
        evals.len(),
        2,
        "one update-script EVALSHA per commit, fast then structural: {evals:#?}"
    );
    for (mode, args) in ["fast", "structural"].into_iter().zip(&evals) {
        assert_eq!(
            args.get(2 + UPDATE_SCRIPT_KEY_FENCE),
            Some(&schema.obj_snapshot_fence(&o)),
            "{mode} mode: KEYS[{UPDATE_SCRIPT_KEY_FENCE}] is the snapshot fence"
        );
        assert_eq!(
            args.get(2 + UPDATE_SCRIPT_KEY_INTENT),
            Some(&schema.obj_snapshot_intent(&o)),
            "{mode} mode: KEYS[{UPDATE_SCRIPT_KEY_INTENT}] is the writer-intent key"
        );
        assert_eq!(
            args.get(2 + UPDATE_SCRIPT_KEY_COMMIT_RELEASE),
            Some(&schema.obj_write_releases(&o)),
            "{mode} mode: KEYS[{UPDATE_SCRIPT_KEY_COMMIT_RELEASE}] is the commit release ring"
        );
        assert_eq!(
            args.get(2 + UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL),
            Some(&schema.migration_sentinel(&o)),
            "{mode} mode: KEYS[{UPDATE_SCRIPT_KEY_MIGRATION_SENTINEL}] is the object's \
             migration sentinel (b.qtk)"
        );
        let numkeys: usize = args[2].parse().expect("numkeys");
        // The mode slot is ARGV[SHARED_ARGV + 1]: past the command, the SHA,
        // numkeys and the keys.
        assert_eq!(
            args.get(3 + numkeys + UPDATE_SCRIPT_SHARED_ARGV)
                .map(String::as_str),
            Some(mode),
            "{mode} mode: the mode slot follows the shared ARGV prefix"
        );
        // `ARGV[i]` is `args[3 + numkeys + i - 1]`.
        let argv = |slot: usize| args.get(3 + numkeys + slot - 1).map(String::as_str);
        let call_token = argv(UPDATE_SCRIPT_ARG_CALL_TOKEN);
        assert!(
            call_token.is_some_and(|t| t.len() == 32
                && t.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))),
            "{mode} mode: ARGV[{UPDATE_SCRIPT_ARG_CALL_TOKEN}] is the call's 32-hex commit \
             token; got {call_token:?}"
        );
        assert_eq!(
            call_token,
            argv(UPDATE_SCRIPT_ARG_COMMIT_KEY),
            "{mode} mode: without an operation id the call token is the commit match key"
        );
        assert_eq!(
            argv(UPDATE_SCRIPT_ARG_RING_TTL_MS),
            Some(default_ring_ttl_ms().as_str()),
            "{mode} mode: ARGV[{UPDATE_SCRIPT_ARG_RING_TTL_MS}] is the store's commit release \
             ring TTL, the default deadline + command timeout + 60 s for a store built with \
             `LayerStore::new`"
        );
    }
    assert_eq!(
        (UPDATE_SCRIPT_ARG_RING_TTL_MS, UPDATE_SCRIPT_SHARED_ARGV),
        (16, 16),
        "the ring TTL is the last shared slot, so the mode slot is ARGV[17]"
    );
    assert_eq!(
        evals[1].get(last_shared_key + 1),
        Some(&schema.obj_sections_index(&o)),
        "structural mode: KEYS[UPDATE_SCRIPT_SHARED_KEYS + 1] is the sections index"
    );
}

// =============================================================================
// 14. b.ph9 — the pre-pass's wrong-typed-key exit (`WRONGTYPE_KEY`).
//
// The validation pre-pass probes every key it is about to write — each
// per-record field / time-sample hash, and each index ZSet this commit will
// ZADD — before the XADD. Without that probe a wrong-typed key aborts the
// APPLY pass, and by then the stream entry is already published: subscribers
// are told about a commit whose materialized state was never written, which is
// the b.ph9 divergence with the arrow reversed.
//
// The tag is deliberately NOT in the `STREAM_ID_COLLISION` family, and that
// separation is itself the contract under test. Nothing is wrong with the
// stream here, so (a) the collision counter must not tick — its whole value
// rests on a zero baseline — and (b) the collision remediation ("clear the
// per-object stream") would send an operator at the wrong key entirely.
// =============================================================================

/// The `ServiceError::Internal` message prefix the Lua path produces for a
/// wrong-typed per-record or index key, and the reply tag inside it. Both live
/// in `src/storage/layer.rs` as `pub(crate)` items, so they are re-stated here
/// rather than imported.
const WRONGTYPE_KEY_PREFIX: &str = "update_layer keyspace type error";
const WRONGTYPE_KEY_TAG: &str = "WRONGTYPE_KEY";

/// Assert the error is the wrong-typed-key surface: a non-retriable `Internal`
/// carrying both the operator-facing prefix and the tag, and NOT the
/// collision family (which would carry a remediation aimed at the stream).
fn assert_wrongtype_key_error(err: ServiceError, expected_key: &str, expected_detail: &str) {
    match err {
        ServiceError::Internal(msg) => {
            assert!(
                msg.starts_with(WRONGTYPE_KEY_PREFIX),
                "a wrong-typed keyspace entry must surface with the `{WRONGTYPE_KEY_PREFIX}:` \
                 prefix; got: {msg:?}"
            );
            assert!(
                msg.contains(WRONGTYPE_KEY_TAG),
                "the pre-pass's tagged reply must survive into the operator-visible message; \
                 got: {msg:?}"
            );
            assert!(
                !msg.contains("STREAM_ID_COLLISION"),
                "a wrong-typed field / index key must NOT be reported as a stream-ID collision \
                 — the stream is fine, and the collision remediation points at the wrong key; \
                 got: {msg:?}"
            );
            assert!(
                msg.contains(expected_key),
                "the refusal must name the broken key so an operator can go straight to it; \
                 expected {expected_key:?} in: {msg:?}"
            );
            assert!(
                msg.contains(expected_detail),
                "the refusal must say what type was expected; expected {expected_detail:?} in: \
                 {msg:?}"
            );
        }
        // Aborted would spin the retry loop against a condition no retry can
        // clear; FailedPrecondition would advertise a client-side recovery
        // (re-resolve the mapping) that cannot help.
        other => {
            panic!("a wrong-typed keyspace entry must be a non-retriable `Internal`; got {other:?}")
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_wrongtype_field_key_refuses_commit_before_the_stream_write() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("bph9-wrongtype-field"));
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua).with_metrics(metrics.clone());
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bph9-wrongtype-field-layer").await;

    // A plain string where the per-field HASH belongs. The apply pass's HSET
    // would hit WRONGTYPE and abort the script — after the XADD.
    let field_key = tv.key_schema().obj_field(&oid.to_string(), "1", "k");
    let _: () = tv
        .client()
        .pool
        .set(&field_key, "not-a-hash", None, None, false)
        .await
        .expect("SET must succeed");

    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;

    let err = ls
        .update_layer_relayed(
            &sid,
            "bph9-wrongtype-field-layer",
            oid,
            Some(delta_with_set_field_inline(1, "k", vec![0x01], 0)),
            "ph9-client",
            5,
        )
        .await
        .expect_err("a commit whose per-record key is wrong-typed must be refused");
    assert_wrongtype_key_error(err, &field_key, "expected hash");

    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(
        ks_after, ks_before,
        "the pre-pass probes the record keys BEFORE the XADD, so a refusal must leave the \
         per-object keyspace — the stream above all — bit-for-bit unchanged.\n\n\
         AFTER: {ks_after:#?}\n\nBEFORE: {ks_before:#?}"
    );
    assert_eq!(
        clock_after.last_server_ts, clock_before.last_server_ts,
        "a refused commit must not restamp obj:hdr last_server_ts"
    );
    assert_eq!(
        collision_counter(&metrics, "lua", "update"),
        0,
        "a wrong-typed FIELD key is not a stream-ID collision; ticking the collision counter \
         here would destroy its zero baseline and page an operator toward the stream key"
    );
    assert_eq!(
        lua_update_fast_mode_count(
            &live_edit::observability::metrics::test_util::render(&metrics),
            "live_edit_keyspace_type_error_total",
        ),
        1,
        "the refusal is counted once on the type-error cell, under the script mode that \
         raised it"
    );
    assert_eq!(
        retry_counter(&metrics, FAST_PATH, "exhausted"),
        0,
        "a wrong-typed key is non-retriable and must not consume the retry budget"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_wrongtype_fields_index_refuses_commit_before_the_stream_write() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("bph9-wrongtype-index"));
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua).with_metrics(metrics.clone());
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bph9-wrongtype-index-layer").await;

    // The index ZSet is probed once per commit rather than per record, and
    // only when the delta actually contains a record of that kind — a
    // separate branch from the per-record probe above.
    let index_key = tv.key_schema().obj_fields_index(&oid.to_string());
    let _: () = tv
        .client()
        .pool
        .del(&index_key)
        .await
        .expect("DEL must succeed");
    let _: () = tv
        .client()
        .pool
        .set(&index_key, "not-a-zset", None, None, false)
        .await
        .expect("SET must succeed");

    let stream_key = tv.key_schema().layer_stream(oid);
    let stream_len_before: u64 = tv
        .client()
        .pool
        .xlen(&stream_key)
        .await
        .expect("XLEN must succeed");
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let header_before: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL must succeed");

    let err = ls
        .update_layer_relayed(
            &sid,
            "bph9-wrongtype-index-layer",
            oid,
            Some(delta_with_set_field_inline(1, "k", vec![0x01], 0)),
            "ph9-client",
            6,
        )
        .await
        .expect_err("a commit whose fields index is wrong-typed must be refused");
    assert_wrongtype_key_error(err, &index_key, "expected zset");

    // The whole header, not just `seq` — a refusal must not restamp the
    // commit clock or the last-client columns either. (The keyspace dump is
    // unusable here: it ZRANGEs the very key this test wrong-typed.)
    let header_after: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL must succeed");
    assert_eq!(
        header_after, header_before,
        "a refused commit must leave obj:hdr bit-for-bit unchanged — seq, next_set_order and \
         last_server_ts alike"
    );
    let stream_len_after: u64 = tv
        .client()
        .pool
        .xlen(&stream_key)
        .await
        .expect("XLEN must succeed");
    assert_eq!(
        stream_len_after, stream_len_before,
        "the index probe runs before the XADD, so no stream entry may be appended"
    );
    assert_eq!(
        collision_counter(&metrics, "lua", "update"),
        0,
        "a wrong-typed INDEX key is not a stream-ID collision"
    );
}

// ── 14b. The ST half of the pre-pass, and the `saw_field` / `saw_ts` gating ──
//
// The two cases above drive the SF half only: an `SF` record's per-record hash
// and the FIELDS index. The pre-pass has a second, structurally identical arm
// for `ST` records (`obj:ts:*` per-record hash) and a second index probe
// (`obj:tss`), and a mutation of either that only *looked* correct — probing
// the fields index for a time-sample record, say — would be invisible to the
// SF-only coverage.
//
// The gating is a contract in its own right, stated in the script: "an index is
// probed only when a record of its class exists: an all-time-sample delta must
// not be refused because the *fields* index is corrupt, and an empty delta
// touches neither." Without a negative control, hoisting either probe out of
// its `if saw_* then` guard would pass every test above while turning one
// corrupt index into a total write outage for the object — including for
// deltas that never touch it.

/// Section / sample time the time-sample fixtures below write into. The values
/// are arbitrary; what matters is that both the per-record key and the ZADD
/// member are derived from them, so a test can name the exact key it corrupted.
const TS_ONLY_SECTION_ID: u64 = 4;
const TS_ONLY_SAMPLE_TIME: f64 = 7.25;

/// A delta carrying exactly one inline time sample and no SetField, so the
/// encoded record stream is a single `ST` — `saw_ts` true, `saw_field` false.
fn time_sample_only_delta(value: Vec<u8>) -> OwnedDelta {
    OwnedDelta {
        time_samples: vec![OodfTimeSample {
            section_id: TS_ONLY_SECTION_ID,
            time: TS_ONLY_SAMPLE_TIME,
            value_or_ext_hash: value,
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    }
}

/// The per-record key an [`time_sample_only_delta`] commit would HSET.
fn ts_only_record_key(tv: &TestValkey, oid: u64) -> String {
    tv.key_schema().obj_time_sample(
        &oid.to_string(),
        &TS_ONLY_SECTION_ID.to_string(),
        TS_ONLY_SAMPLE_TIME,
    )
}

/// Replace `key` with a plain string, whatever it held before.
async fn wrong_type_key(tv: &TestValkey, key: &str) {
    let _: () = tv
        .client()
        .pool
        .del(key)
        .await
        .expect("DEL before re-typing must succeed");
    let _: () = tv
        .client()
        .pool
        .set(key, "not-the-right-type", None, None, false)
        .await
        .expect("SET must succeed");
}

/// ST sibling of `update_layer_lua_wrongtype_field_key_refuses_commit_…`.
///
/// The per-record probe is driven by the record's KEYS slot, not by its tag, so
/// this is the same branch — but only this case proves the `ST` arm actually
/// reaches it. A pre-pass that probed `if tag == "SF"` would leave every
/// time-sample commit aborting mid-apply with its stream entry already
/// published, which is the b.ph9 divergence.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_wrongtype_time_sample_key_refuses_commit_before_the_stream_write() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("bph9-wrongtype-ts"));
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua).with_metrics(metrics.clone());
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bph9-wrongtype-ts-layer").await;

    let ts_key = ts_only_record_key(&tv, oid);
    wrong_type_key(&tv, &ts_key).await;

    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;

    let err = ls
        .update_layer_relayed(
            &sid,
            "bph9-wrongtype-ts-layer",
            oid,
            Some(time_sample_only_delta(vec![0x11])),
            "ph9-client",
            21,
        )
        .await
        .expect_err("a commit whose per-record TIME-SAMPLE key is wrong-typed must be refused");
    // Record 1 — the delta carries exactly one mutation.
    assert_wrongtype_key_error(err, &ts_key, "expected hash (mutation record 1)");

    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(
        ks_after, ks_before,
        "the pre-pass probes the record keys BEFORE the XADD, so a refusal must leave the \
         per-object keyspace — the stream above all — bit-for-bit unchanged.\n\n\
         AFTER: {ks_after:#?}\n\nBEFORE: {ks_before:#?}"
    );
    assert_eq!(
        clock_after.last_server_ts, clock_before.last_server_ts,
        "a refused commit must not restamp obj:hdr last_server_ts"
    );
    assert_eq!(
        collision_counter(&metrics, "lua", "update"),
        0,
        "a wrong-typed TIME-SAMPLE key is not a stream-ID collision; ticking the collision \
         counter here would destroy its zero baseline and page an operator toward the stream key"
    );
    assert_eq!(
        retry_counter(&metrics, FAST_PATH, "exhausted"),
        0,
        "a wrong-typed key is non-retriable and must not consume the retry budget"
    );
}

/// ST sibling of `update_layer_lua_wrongtype_fields_index_refuses_commit_…`.
///
/// `obj:tss` is a distinct probe behind a distinct `if saw_ts then` guard, with
/// its own message. Only a time-sample-carrying delta reaches it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_wrongtype_ts_index_refuses_commit_before_the_stream_write() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("bph9-wrongtype-tsidx"));
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua).with_metrics(metrics.clone());
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bph9-wrongtype-tsidx-layer").await;

    let index_key = tv.key_schema().obj_time_samples_index(&oid.to_string());
    wrong_type_key(&tv, &index_key).await;

    let stream_key = tv.key_schema().layer_stream(oid);
    let stream_len_before: u64 = tv
        .client()
        .pool
        .xlen(&stream_key)
        .await
        .expect("XLEN must succeed");
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let header_before: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL must succeed");

    let err = ls
        .update_layer_relayed(
            &sid,
            "bph9-wrongtype-tsidx-layer",
            oid,
            Some(time_sample_only_delta(vec![0x22])),
            "ph9-client",
            22,
        )
        .await
        .expect_err("a commit whose time-samples index is wrong-typed must be refused");
    assert_wrongtype_key_error(err, &index_key, "expected zset (time-samples index)");

    // The keyspace dump is unusable here — it ZRANGEs the very key this test
    // wrong-typed — so the header and the stream are checked directly.
    let header_after: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL must succeed");
    assert_eq!(
        header_after, header_before,
        "a refused commit must leave obj:hdr bit-for-bit unchanged — seq, next_set_order and \
         last_server_ts alike"
    );
    let stream_len_after: u64 = tv
        .client()
        .pool
        .xlen(&stream_key)
        .await
        .expect("XLEN must succeed");
    assert_eq!(
        stream_len_after, stream_len_before,
        "the index probe runs before the XADD, so no stream entry may be appended"
    );
    assert_eq!(
        collision_counter(&metrics, "lua", "update"),
        0,
        "a wrong-typed TIME-SAMPLES INDEX key is not a stream-ID collision"
    );
    assert_eq!(
        retry_counter(&metrics, FAST_PATH, "exhausted"),
        0,
        "a wrong-typed key is non-retriable and must not consume the retry budget"
    );
}

/// Read the raw string value planted by [`wrong_type_key`], asserting the key
/// was neither re-typed nor written through.
async fn assert_still_wrong_typed(tv: &TestValkey, key: &str) {
    let raw: Option<String> = tv
        .client()
        .pool
        .get(key)
        .await
        .expect("GET on the planted string must succeed");
    assert_eq!(
        raw.as_deref(),
        Some("not-the-right-type"),
        "{key} must be untouched by a commit that does not index into it — a probe (or a ZADD) \
         that ignored its `saw_*` guard would have refused the commit or re-typed the key"
    );
}

/// Negative control for `saw_field`, and the case the script's comment states
/// outright: "an all-time-sample delta must not be refused because the *fields*
/// index is corrupt."
///
/// This is the one that fails if either index probe is hoisted out of its
/// guard — and the failure mode it prevents is severe, because a single corrupt
/// `obj:fld` ZSet would otherwise wedge *every* write to the object, including
/// the time-sample traffic that never touches it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_time_sample_only_delta_commits_despite_a_corrupt_fields_index() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("bph9-saw-field-gate"));
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua).with_metrics(metrics.clone());
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bph9-saw-field-gate-layer").await;

    let fields_index_key = tv.key_schema().obj_fields_index(&oid.to_string());
    wrong_type_key(&tv, &fields_index_key).await;

    let (seq, _ts) = ls
        .update_layer_relayed(
            &sid,
            "bph9-saw-field-gate-layer",
            oid,
            Some(time_sample_only_delta(vec![0x33])),
            "ph9-client",
            23,
        )
        .await
        .expect(
            "an ST-only delta ZADDs the time-samples index and never names the fields index, so \
             a corrupt fields index must not refuse it — `saw_field` is false for this delta",
        );
    assert_eq!(seq, 2, "create_test_layer seeded seq=1");

    // The commit really landed, not just "did not error": the sample hash, its
    // index member, and the stream entry.
    let ts_slots: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(ts_only_record_key(&tv, oid))
        .await
        .expect("HGETALL time sample must succeed");
    assert_eq!(
        ts_slots.get("version").map(String::as_str),
        Some("2"),
        "the time sample must carry the commit version; got {ts_slots:?}"
    );
    let ts_member =
        KeySchema::time_sample_member(&TS_ONLY_SECTION_ID.to_string(), TS_ONLY_SAMPLE_TIME);
    let ts_score: Option<f64> = tv
        .client()
        .pool
        .zscore(
            tv.key_schema().obj_time_samples_index(&oid.to_string()),
            &ts_member,
        )
        .await
        .expect("ZSCORE time-samples index must succeed");
    assert_eq!(
        ts_score,
        Some(2.0),
        "the time-samples index must carry the sample at the commit sequence"
    );
    // Fields read as bytes: an entry's relay `ompf` field is binary.
    let stream_ids: Vec<(String, HashMap<String, bytes::Bytes>)> = tv
        .client()
        .pool
        .xrange_values(tv.key_schema().layer_stream(oid), "-", "+", None)
        .await
        .expect("XRANGE must succeed");
    assert_eq!(
        stream_ids
            .iter()
            .map(|(id, _)| id.as_str())
            .collect::<Vec<_>>(),
        vec!["1-0", "2-0"],
        "CreateLayer's entry plus this commit's own"
    );

    assert_still_wrong_typed(&tv, &fields_index_key).await;
    assert_eq!(
        collision_counter(&metrics, "lua", "update"),
        0,
        "a successful commit must not tick the collision counter"
    );
}

/// Mirror control for `saw_ts`: a field-only delta must commit with a corrupt
/// `obj:tss`. Same argument with the classes swapped — and the mirror is worth
/// having because the two probes are separate `if` blocks, so a guard can be
/// dropped from one without touching the other.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_field_only_delta_commits_despite_a_corrupt_ts_index() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("bph9-saw-ts-gate"));
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua).with_metrics(metrics.clone());
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bph9-saw-ts-gate-layer").await;

    let ts_index_key = tv.key_schema().obj_time_samples_index(&oid.to_string());
    wrong_type_key(&tv, &ts_index_key).await;

    let (seq, _ts) = ls
        .update_layer_relayed(
            &sid,
            "bph9-saw-ts-gate-layer",
            oid,
            Some(delta_with_set_field_inline(1, "k", vec![0x44], 0)),
            "ph9-client",
            24,
        )
        .await
        .expect(
            "an SF-only delta never names the time-samples index, so a corrupt one must not \
             refuse it — `saw_ts` is false for this delta",
        );
    assert_eq!(seq, 2);

    let field_slots: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().obj_field(&oid.to_string(), "1", "k"))
        .await
        .expect("HGETALL field must succeed");
    assert_eq!(
        field_slots.get("version").map(String::as_str),
        Some("2"),
        "the field must carry the commit version; got {field_slots:?}"
    );
    assert_still_wrong_typed(&tv, &ts_index_key).await;
    assert_eq!(collision_counter(&metrics, "lua", "update"), 0);
}

/// The third clause of the same sentence: "an empty delta touches neither."
///
/// With BOTH indexes corrupt, an empty delta must still commit — it advances
/// the header and publishes a stream entry (the heartbeat/`Ping`-shaped commit)
/// without ZADDing anything.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_empty_delta_commits_despite_both_indexes_corrupt() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("bph9-empty-gate"));
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua).with_metrics(metrics.clone());
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bph9-empty-gate-layer").await;

    let fields_index_key = tv.key_schema().obj_fields_index(&oid.to_string());
    let ts_index_key = tv.key_schema().obj_time_samples_index(&oid.to_string());
    wrong_type_key(&tv, &fields_index_key).await;
    wrong_type_key(&tv, &ts_index_key).await;

    let (seq, _ts) = ls
        .update_layer_relayed(
            &sid,
            "bph9-empty-gate-layer",
            oid,
            Some(empty_delta()),
            "ph9-client",
            25,
        )
        .await
        .expect("an empty delta probes neither index and must commit");
    assert_eq!(seq, 2);

    let header_seq: String = tv
        .client()
        .pool
        .hget(tv.key_schema().obj_header(&oid.to_string()), "seq")
        .await
        .expect("HGET seq must succeed");
    assert_eq!(header_seq, "2", "the empty commit must advance the header");
    assert_still_wrong_typed(&tv, &fields_index_key).await;
    assert_still_wrong_typed(&tv, &ts_index_key).await;
    assert_eq!(collision_counter(&metrics, "lua", "update"), 0);
}

// =============================================================================
// 15. b.ph9 post-completion review — the XADD-refusal ALLOWLIST.
//
// The fast path's `redis.pcall`ed `XADD` used to be classified by EXCLUSION:
// "tag it STREAM_ID_COLLISION unless it starts with `OOM`". That is a denylist,
// and it is unmaintainable by construction, because the set of things a server
// can say to a script's FIRST write belongs to the server, not to us — `-OOM`,
// `-READONLY`, `-MISCONF`, `-NOREPLICAS`, stale-replica, `-NOPERM`, `-MOVED` /
// `-CROSSSLOT`, and whatever gate a future version adds. Each of those, tagged,
// becomes: a non-retriable `INTERNAL` handed to a client for a *transient*
// server-wide condition, a tick on a pageable zero-baseline counter, and a WARN
// telling an operator to delete a perfectly healthy stream.
//
// It is now an allowlist of exactly two shapes — the ID-monotonicity refusal,
// matched on the plain substring `equal or smaller`, and `WRONGTYPE` as a
// 9-character prefix — with everything else returned verbatim and untagged into
// the shared fred classifier. This block is the regression guard for that
// direction, in three parts:
//
//   (a) the two allowlist members still get the tag, asserted at the SCRIPT
//       boundary, where the underlying server text is still visible. Those two
//       tests are what pin the *substrings* the allowlist matches on: if a
//       future Valkey reworded either reply, the fast path would silently stop
//       tagging a real collision, and the through-the-API tests in section 10
//       would report only "wrong error variant" without saying why;
//   (b) a NON-allowlisted refusal comes back verbatim and untagged, again at
//       the script boundary;
//   (c) the same non-allowlisted refusal driven through `LayerStore::update_
//       layer`, pinning the Rust classification arm: no collision prefix, no
//       counter tick, no retry-budget burn, and the keyspace untouched.
//
// ── Reachability, stated plainly ──────────────────────────────────────────
//
// The class-B refusals the fix is actually *about* (`-OOM`, `-READONLY`,
// `-MISCONF`, `-NOREPLICAS`) cannot be provoked here: each needs `CONFIG SET`
// on the backing instance or a replica topology, and the shared sandbox / CI
// Valkey is neither reconfigurable nor replicated. Nor can they be faked
// through a Rust seam — `lua_fred_error_injector_for_test` short-circuits
// BEFORE the EVALSHA and returns `map_snapshot_load_err(...)` directly, so it
// never reaches the tag-matching arms this section exists to test, and adding a
// seam that did would be a source change (out of scope for a test round).
//
// What IS reachable is a real, server-generated XADD refusal that is not on the
// allowlist: `XADD <key> 0-0`. `streamGenericAddCommand` (valkey
// `src/t_stream.c`) rejects a fully-specified `0-0` unconditionally, with
// [`XADD_ZERO_ID_REFUSAL`], before the monotonicity comparison runs — so it is
// the *other* XADD ID error, and the script's own comment names it as the case
// that must fall through. Production cannot produce it (`new_seq` is
// `current_seq + 1 >= 1`), so the fixtures below drive `new_seq` to 0 by
// planting `seq = -1` on the header, exactly as the section-10 fixtures plant a
// stream entry the service would never write.
//
// That is a weaker stimulus than a `-READONLY` but the SAME classification
// path: the branch under test reads only the reply string, and this string is
// neither allowlist member. Under the old denylist every assertion in (b) and
// (c) fails — which is the property a regression guard needs.
// =============================================================================

/// Valkey's refusal of a fully-specified `0-0` stream ID. Neither allowlist
/// member: it does not contain `equal or smaller`, and it does not begin
/// `WRONGTYPE`.
const XADD_ZERO_ID_REFUSAL: &str = "The ID specified in XADD must be greater than 0-0";

/// The fragment the fast path's allowlist matches to recognise a genuine
/// stream-ID collision (`string.find(err, ..., 1, true)` — plain, not a
/// pattern). Restated here so a Valkey rewording fails as "the fragment the
/// script keys on is gone", not as a mystery.
const XADD_MONOTONICITY_FRAGMENT: &str = "equal or smaller";

/// Overwrite `obj:hdr seq`, which is what the script derives `new_seq =
/// seq + 1` from. Signed, because the fixtures below plant `-1` to force the
/// script's XADD onto the unconditionally-refused `0-0`.
async fn plant_header_seq(tv: &TestValkey, oid: u64, seq: i64) {
    let _: () = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_header(&oid.to_string()),
            [("seq", seq.to_string())],
        )
        .await
        .expect("HSET obj:hdr seq must succeed");
}

/// (a) Allowlist member 1 — the ID-monotonicity refusal — still gets the tag,
/// and the reply still carries the fragment the match is keyed on.
///
/// Section 10 asserts the end-to-end consequence (a non-retriable `Internal`
/// with the collision prefix). This asserts the *mechanism*: the script saw a
/// reply containing [`XADD_MONOTONICITY_FRAGMENT`] and chose to tag it. If
/// Valkey ever rewords that sentence, this fails first and names the cause.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_monotonicity_refusal_is_tagged_and_quotes_the_matched_fragment() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bph9-allowlist-monotonic").await;

    // Occupy the exact ID the next commit will attempt (`{seq + 1}-0`).
    let header_seq: u64 = tv
        .client()
        .pool
        .hget(tv.key_schema().obj_header(&oid.to_string()), "seq")
        .await
        .map(|s: String| s.parse().expect("seq must be numeric"))
        .expect("HGET seq must succeed");
    let _: String = tv
        .client()
        .pool
        .xadd(
            tv.key_schema().layer_stream(oid),
            false,
            None,
            format!("{}-0", header_seq + 1),
            vec![("payload", "out-of-band|planted-by-test")],
        )
        .await
        .expect("planting the colliding entry must succeed");

    let inv = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "allowlist", 1);
    let details = inv
        .dispatch(&tv)
        .await
        .expect_err("an occupied {new_seq}-0 must refuse the commit")
        .details()
        .to_string();

    assert!(
        details.contains("STREAM_ID_COLLISION"),
        "the monotonicity refusal is allowlist member 1 and MUST carry the tag — untagged it \
         would fall through to the shared classifier and the collision counter would never \
         tick on a genuinely wedged stream; got: {details:?}"
    );
    assert!(
        details.contains(XADD_MONOTONICITY_FRAGMENT),
        "the tagged reply must still quote the server's own words, and they must still contain \
         {XADD_MONOTONICITY_FRAGMENT:?} — that fragment is the entire matching rule, so if \
         Valkey reworded the reply the tag above is being applied for some other reason and \
         the real collision would go untagged; got: {details:?}"
    );
}

/// (a) Allowlist member 2 — `WRONGTYPE` — still gets the tag, with the
/// 9-character prefix intact in the quoted reply.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_wrongtype_stream_refusal_is_tagged_and_quotes_the_prefix() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bph9-allowlist-wrongtype").await;

    let stream_key = tv.key_schema().layer_stream(oid);
    wrong_type_key(&tv, &stream_key).await;

    let inv = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "allowlist", 2);
    let details = inv
        .dispatch(&tv)
        .await
        .expect_err("a non-stream at the stream key must refuse the commit")
        .details()
        .to_string();

    assert!(
        details.contains("STREAM_ID_COLLISION"),
        "a WRONGTYPE stream key is allowlist member 2 and MUST carry the tag — same fail-closed \
         operator condition, same remediation, and the slow path's probe agrees; got: {details:?}"
    );
    assert!(
        details.contains("WRONGTYPE"),
        "the tagged reply must quote the server's `WRONGTYPE` reply, which the script matches \
         as a 9-character prefix; got: {details:?}"
    );
}

/// (b) The regression guard, at the script boundary: a refusal that is on
/// NEITHER allowlist is returned verbatim and untagged, with nothing written.
///
/// Under the pre-review denylist this reply was tagged `STREAM_ID_COLLISION`
/// (it is not an `OOM`), so the first assertion below fails outright on the old
/// script.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_fast_script_non_allowlisted_xadd_refusal_returns_verbatim_and_untagged() {
    let tv = TestValkey::setup().await;
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "bph9-allowlist-passthrough").await;

    // `new_seq = seq + 1 = 0`, so the script's XADD is `0-0`.
    plant_header_seq(&tv, oid, -1).await;

    let inv = FastPathInvocation::build(&tv, oid, two_inline_set_fields_delta(), "allowlist", 3);
    let (ks_before, clock_before) = dump_object_keyspace(&tv, oid).await;

    let details = inv
        .dispatch(&tv)
        .await
        .expect_err("XADD 0-0 is refused unconditionally, so the commit must fail")
        .details()
        .to_string();

    assert!(
        !details.contains("STREAM_ID_COLLISION"),
        "a refusal that is neither the monotonicity fragment nor a WRONGTYPE prefix must NOT be \
         tagged — it is a verdict about the server (or, here, about the requested ID), not \
         about this stream key, and tagging it hands the client a non-retriable INTERNAL, ticks \
         a pageable zero-baseline counter, and points an operator at a healthy stream. This is \
         the assertion the pre-review denylist (`tag everything that is not OOM`) fails; \
         got: {details:?}"
    );
    assert!(
        details.contains(XADD_ZERO_ID_REFUSAL),
        "the server's own reply must survive VERBATIM so the shared fred classifier can map it \
         as itself; got: {details:?}"
    );
    assert!(
        !details.contains(XADD_MONOTONICITY_FRAGMENT),
        "fixture sanity: this reply must not contain the allowlisted fragment, or the test is \
         driving allowlist member 1 by accident; got: {details:?}"
    );

    // Same all-or-nothing guarantee as a tagged refusal: the XADD is the
    // script's first write, so an untagged pass-through must leave the
    // keyspace exactly as it found it.
    let (ks_after, clock_after) = dump_object_keyspace(&tv, oid).await;
    assert_eq!(
        ks_after, ks_before,
        "an untagged pass-through must still fail the commit WHOLE — the classification changed, \
         the atomicity did not.\n\nAFTER: {ks_after:#?}\n\nBEFORE: {ks_before:#?}"
    );
    assert_eq!(
        clock_after.last_server_ts, clock_before.last_server_ts,
        "a refused commit must not restamp obj:hdr last_server_ts"
    );
}

/// (c) The same non-allowlisted refusal through `LayerStore::update_layer` —
/// the Rust classification arm.
///
/// This is the half that pins the operator-visible consequences: which
/// `ServiceError` the client gets, whether `live_edit_stream_id_collision_total`
/// moves, and whether the retry budget is burned. All three were wrong under
/// the denylist, and none of them is observable from the script-level test
/// above.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_non_allowlisted_xadd_refusal_is_not_reported_as_a_collision() {
    let h = setup_collision_harness(
        UpdateLayerBackend::Lua,
        "bph9-allowlist-rust",
        "bph9-allowlist-rust-layer",
    )
    .await;

    plant_header_seq(&h.tv, h.oid, -1).await;

    let (ks_before, clock_before) = dump_object_keyspace(&h.tv, h.oid).await;

    let err = h
        .update("passthrough", 26)
        .await
        .expect_err("XADD 0-0 is refused unconditionally, so the commit must fail");

    match err {
        ServiceError::Internal(msg) => {
            assert!(
                !msg.starts_with(COLLISION_UPDATE_PREFIX),
                "a non-allowlisted refusal must not reach the collision arm — that arm's message \
                 tells an operator the stream is corrupt and to delete it; got: {msg:?}"
            );
            assert!(
                !msg.contains("STREAM_ID_COLLISION"),
                "the tag must not appear anywhere in the operator-visible message; got: {msg:?}"
            );
            // The shared classifier's own wrapper — i.e. the error travelled
            // through `map_snapshot_load_err`, which is where every class-B
            // server refusal is supposed to land.
            assert!(
                msg.contains("EVALSHA update_layer_fast"),
                "the refusal must be classified by the shared fred classifier, whose message \
                 names the operation; got: {msg:?}"
            );
            assert!(
                msg.contains(XADD_ZERO_ID_REFUSAL),
                "the server's own words must reach the operator undisturbed; got: {msg:?}"
            );
        }
        // `Aborted` would be correct for a *transient* class-B member
        // (`-READONLY` during a failover, say) and is what the shared
        // classifier produces for those; this particular stimulus is a
        // permanent `ErrorKind::Unknown`, so `Internal` is the honest answer.
        // What must never appear is the collision surface asserted against
        // above.
        other => panic!(
            "a non-allowlisted XADD refusal must surface through the shared classifier as \
             `Internal`; got {other:?}"
        ),
    }

    // The counter is the whole point of the fix. Both backend labels and both
    // sites, because the assertion is "this condition is not a collision
    // anywhere", not "not a collision under one label".
    for backend_label in ["lua", "txn"] {
        for site in ["update", "create"] {
            assert_eq!(
                collision_counter(&h.metrics, backend_label, site),
                0,
                "a refusal that is not a verdict about this stream key must not tick \
                 live_edit_stream_id_collision_total{{backend={backend_label},site={site}}} — \
                 that counter is an operator alarm whose entire value is its zero baseline"
            );
        }
    }
    assert_eq!(
        retry_counter(&h.metrics, FAST_PATH, "exhausted"),
        0,
        "this stimulus classifies permanent, so the 128-attempt budget must not be burned"
    );
    assert_eq!(
        retry_counter(&h.metrics, FAST_PATH, "committed"),
        0,
        "a refused commit must not record a committed retry"
    );

    let (ks_after, clock_after) = dump_object_keyspace(&h.tv, h.oid).await;
    assert_eq!(
        ks_after, ks_before,
        "the commit must still fail whole.\n\nAFTER: {ks_after:#?}\n\nBEFORE: {ks_before:#?}"
    );
    assert_eq!(
        clock_after.last_server_ts, clock_before.last_server_ts,
        "a refused commit must not restamp obj:hdr last_server_ts"
    );
}

// =============================================================================
// Cluster-mode parity (b.ohv item 1).
//
// Gated on `LIVE_EDIT_TEST_CLUSTER_URL`. On cluster, every per-object key
// shares the `{object_id}` hash tag, so the EVALSHA inside `update_layer_fast.lua`
// stays inside one slot. A regression that dropped the hash tag from any of
// the object's key builders would surface here as a CROSSSLOT error during
// the script's HSET / ZADD / XADD steps.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_update_layer_lua_fast_path_field_only_commits() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };

    let tv = TestValkey::setup_with_url(cluster_url, 4).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must be a redis-cluster:// URL — the cluster \
         slot-co-location contract is only exercised when `is_clustered()` is true"
    );

    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "cluster-lua-layer").await;

    // Field-only delta — exactly the shape the fast path was built for.
    let delta = delta_with_set_field_inline(1, "k", vec![0xAB, 0xCD], 0);
    let (seq, _ts) = ls
        .update_layer_relayed(&sid, "cluster-lua-layer", oid, Some(delta), "cluster", 0)
        .await
        .expect(
            "Lua fast path on cluster must commit cleanly — CROSSSLOT here means \
             the {object_id} hash tag was dropped from one of the per-object key \
             builders touched by the EVALSHA",
        );
    assert!(
        seq > 1,
        "seq must advance past CreateLayer(seq=1); got {seq}"
    );

    // Verify the field was actually written and is readable via cluster.
    let keys = tv.key_schema();
    let oid_str = oid.to_string();
    let header_key = keys.obj_header(&oid_str);
    let header_seq: String = tv
        .client()
        .pool
        .hget(&header_key, "seq")
        .await
        .expect("HGET header.seq on cluster must succeed");
    assert_eq!(header_seq, seq.to_string(), "header seq must match commit");
}

/// The `lua` backend's structural route on a cluster: three structural calls
/// on one object — a cold start (`structural_full`), a cache hit
/// (`structural`), and one whose commit lands with its reply lost (the
/// post-`EVALSHA` fault queue), answered by the retry's read-side token check
/// as already landed. Every read (the probe, the cold start's atomic read, the
/// repair read) and the structural commit name only `{object_id}`-tagged keys,
/// so a CROSSSLOT anywhere on the route fails the call.
///
/// **Two activation conditions.** `LIVE_EDIT_TEST_CLUSTER_URL` (a `SKIP:` line
/// without it) and the `test-fault-injection` feature, which the lost-reply
/// queue needs: without the feature this test is compiled out silently.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_update_layer_lua_structural_route_cold_hit_and_already_landed_commit() {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    use oodf_ffi::{CommandCreateSection, StructCommand};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    const LAYER: &str = "cluster-lua-structural-layer";
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };

    let tv = TestValkey::setup_with_url(cluster_url, 4).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must be a redis-cluster:// URL — the cluster \
         slot-co-location contract is only exercised when `is_clustered()` is true"
    );
    let metrics = Arc::new(crate::common::metrics_scrape_only(LAYER));
    tv.client()
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    let lost: Arc<Mutex<VecDeque<fred::error::Error>>> = Arc::new(Mutex::new(VecDeque::new()));
    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua)
        .with_metrics(Arc::clone(&metrics))
        .with_lua_post_eval_fred_error_injector_for_test(Arc::clone(&lost));
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, LAYER).await;

    let create_section = |section_id: u64| OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id,
            section_name: format!("ClusterSection{section_id}"),
            section_type: 1,
        })],
        ..Default::default()
    };
    let attempts = |path: &str| {
        let text = render(&metrics);
        let read = |suffix: &str| {
            sample_value(
                &text,
                &format!("live_edit_update_layer_attempts_histogram_{suffix}"),
                &[("path", path)],
            )
            .unwrap_or_default()
        };
        (read("count"), read("sum"))
    };
    let commit = |section_id: u64, client_id: &'static str| {
        let ls = &ls;
        let sid = &sid;
        async move {
            ls.update_layer_relayed(
                sid,
                LAYER,
                oid,
                Some(create_section(section_id)),
                client_id,
                section_id,
            )
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "the structural route on a cluster must commit section {section_id}; got \
                     {e:?}. A CROSSSLOT here means a key the route reads or writes lost the \
                     {{object_id}} hash tag"
                )
            })
        }
    };

    let (cold_seq, _) = commit(10, "cluster-structural-cold").await;
    assert_eq!(
        attempts("structural_full"),
        (1.0, 1.0),
        "the first call cold-starts the object's cache entry"
    );
    let (hit_seq, _) = commit(11, "cluster-structural-hit").await;
    assert_eq!(hit_seq, cold_seq + 1);
    assert_eq!(
        attempts("structural"),
        (1.0, 1.0),
        "the second call plans on the cached entry"
    );

    lost.lock().unwrap().push_back(fred::error::Error::new(
        fred::error::ErrorKind::Timeout,
        "cluster: the structural commit's reply is lost",
    ));
    let (landed_seq, _) = commit(12, "cluster-structural-lost").await;
    assert!(
        lost.lock().unwrap().is_empty(),
        "fixture: the lost reply was consumed, so the commit's reply was lost"
    );
    assert_eq!(
        landed_seq,
        hit_seq + 1,
        "the retry answers the landed commit's seq: one commit, not two"
    );
    assert_eq!(
        sample_value(
            &render(&metrics),
            "live_edit_update_layer_commit_already_landed_total",
            &[("backend", "lua")],
        ),
        Some(1.0),
        "the retry's read side answered the landed commit"
    );

    let keys = tv.key_schema();
    let oid_str = oid.to_string();
    let header_seq: String = tv
        .client()
        .pool
        .hget(keys.obj_header(&oid_str), "seq")
        .await
        .expect("HGET header.seq on cluster must succeed");
    assert_eq!(
        header_seq,
        landed_seq.to_string(),
        "nothing was applied twice: the header stands at the landed commit"
    );
    for section_id in [10u64, 11, 12] {
        let name: Option<String> = tv
            .client()
            .pool
            .hget(keys.obj_section(&oid_str, &section_id.to_string()), "name")
            .await
            .expect("HGET obj:sec name on cluster must succeed");
        assert_eq!(
            name.as_deref(),
            Some(format!("ClusterSection{section_id}").as_str()),
            "section {section_id} must be stored"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_update_layer_lua_concurrent_writers_no_crossslot_no_aborts() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set");
        return;
    };

    const WRITERS: usize = 16;
    const UPDATES_PER_WRITER: usize = 5;

    let tv = TestValkey::setup_with_url(cluster_url, WRITERS as u32 * 2).await;
    assert!(tv.client().pool.next().is_clustered());

    let ls = layer_store_with_backend(&tv, UpdateLayerBackend::Lua);
    let sid = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &sid, "cluster-lua-race-layer").await;

    // Barrier so all writers EVALSHA into the same {object_id} slot
    // simultaneously. Cluster's per-slot single-thread serialization
    // turns this into a synchronous queue inside Valkey — no aborts.
    let barrier = Arc::new(tokio::sync::Barrier::new(WRITERS));

    let mut handles = Vec::with_capacity(WRITERS);
    for w in 0..WRITERS {
        let ls_i = ls.clone();
        let sid_i = sid.clone();
        let barrier_i = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier_i.wait().await;
            let mut oks = 0u32;
            let mut errs: Vec<String> = Vec::new();
            for u in 0..UPDATES_PER_WRITER {
                let delta = delta_with_set_field_inline(
                    1,
                    &format!("w{w}-u{u}"),
                    vec![w as u8, u as u8],
                    0,
                );
                match ls_i
                    .update_layer_relayed(
                        &sid_i,
                        "cluster-lua-race-layer",
                        oid,
                        Some(delta),
                        &format!("w{w}"),
                        0,
                    )
                    .await
                {
                    Ok(_) => oks += 1,
                    Err(e) => errs.push(format!("{e:?}")),
                }
            }
            (oks, errs)
        }));
    }

    let mut total_ok = 0u32;
    let mut all_errs: Vec<String> = Vec::new();
    for h in handles {
        let (o, e) = h.await.expect("writer must not panic");
        total_ok += o;
        all_errs.extend(e);
    }

    assert!(
        all_errs.is_empty(),
        "cluster + Lua: no errors expected (Lua is atomic, slot is single-threaded). \
         Got {} errors. First few: {:?}",
        all_errs.len(),
        all_errs.iter().take(5).collect::<Vec<_>>()
    );
    assert_eq!(
        total_ok as usize,
        WRITERS * UPDATES_PER_WRITER,
        "every writer must commit on cluster + Lua; got {total_ok}"
    );
}
