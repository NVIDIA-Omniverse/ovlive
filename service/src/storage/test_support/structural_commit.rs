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

//! Test support for the structural commit (plan `b.dkt`, Epic 5; Spec SDD
//! SR-6): a planner that stands in for the structural cache, and a dump of
//! one object's whole keyspace for before / after and cross-route compares.
//!
//! Gated by the `test-support` feature (with the rest of
//! [`crate::storage::test_support`]), so both `tests/integration` and
//! `tests/proptests.rs` can call it.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use bytes::Bytes;
use fred::interfaces::{
    HashesInterface, KeysInterface, ListInterface, SetsInterface, SortedSetsInterface,
    StreamsInterface,
};
use futures::StreamExt;

use crate::domain::delta::apply_delta;
use crate::domain::types::ObjectState;
use crate::error::ServiceError;
use crate::storage::client::{map_snapshot_load_err, FredErrOrigin, POOL_LABEL_DATA};
use crate::storage::diff_read::{
    read_object_snapshot, snapshot_to_object_state, SnapshotRead, SnapshotReadConsumer,
};
use crate::storage::keys::KeySchema;
use crate::storage::layer::{LayerStore, HEADER_FIELD_COMMIT_CHAIN};
use crate::storage::notifications::{NOTIFICATION_PAYLOAD_FIELD, RELAY_OMPF_FIELD};
use crate::storage::structural::{PreImageDigests, StructuralCommit};
use crate::storage::test_support::scan_matching;

// ── The planner ─────────────────────────────────────────────────────────────

/// What [`plan_structural_commit`] plans: one `UpdateLayer` call's delta and
/// the call's identity, everything a [`StructuralCommit`] carries beyond what
/// the planner reads from the store.
pub struct PlanRequest<'a> {
    /// The call's session (carried for the collision disambiguation).
    pub session_id: &'a str,
    /// The call's layer name.
    pub layer_name: &'a str,
    /// The object to plan against.
    pub object_id: u64,
    /// The delta, applied in complete mode to the object's stored state.
    pub delta: Option<oodf_ffi::OwnedDelta>,
    /// The writer's raw OMPF, relayed verbatim as the stream entry's `ompf`.
    pub relay_ompf: Option<Bytes>,
    /// The call's commit token.
    pub commit_token: &'a str,
    /// The writing client.
    pub client_id: &'a str,
    /// The client's timestamp.
    pub client_ts: u64,
}

/// A structural commit planned by [`plan_structural_commit`], with the states
/// it was planned between.
pub struct PlannedCommit {
    /// The commit, ready for `structural::commit_structural`.
    pub commit: StructuralCommit,
    /// The object's state as the planner read it, at
    /// [`StructuralCommit::expected_seq`] — the state the digests describe.
    pub pre_state: ObjectState,
    /// The state `apply_delta` left after the plan: what the store holds once
    /// the commit lands.
    pub post_state: ObjectState,
}

/// Plan a structural commit the way a structural cache entry would (SR-5.1),
/// standing in for that cache (plan `b.dkt` Epic 4).
///
/// Reads the object fully and consistently at its current `seq`: the header's
/// `seq` and `commit_chain`, then the atomic snapshot read
/// (`object_snapshot_read.lua`, one `EVALSHA`), then the header again. The
/// read is used only if the snapshot's cut and both header reads agree;
/// otherwise a commit landed during it and the planner answers `Aborted`
/// rather than plan on a mixed state. The decoded state
/// (`snapshot_to_object_state`) is the normalized state a cache entry holds
/// (SR-3.2): the snapshot read leaves out lists under dangling parents, and
/// an empty list has no stored form. One exception: a legacy `-0.0`
/// time-sample row, which the full form holds at `+0.0` (resolving a stored
/// `-0.0` / `+0.0` pair into one sample) where a cache entry keys each stored
/// row by its own time bits. A plan here therefore believes no `-0.0` row:
/// a reset or cascade over a section holding a legacy one names the `-0.0`
/// key with an "absent" digest (rule E), so the commit is refused
/// `CACHE_DIVERGED`, and over a lone legacy row so is the `+0.0` key's
/// "exists" digest. Exercise legacy `-0.0` rows through the real cached
/// route.
///
/// Then runs complete-mode `apply_delta` on a clone of that state and builds
/// the [`StructuralCommit`]: `expected_seq` / `expected_commit_chain` from the
/// read's header (`""` for a header without the column), the plan's
/// mutations, `next_set_order` from the post-apply state, and the digests from
/// [`PreImageDigests::from_plan`] over the read state.
///
/// # Errors
///
/// - `Aborted` when a commit landed during the read;
/// - `Internal` when the object is above the atomic read's limits (the
///   planner has no other read);
/// - the read's own errors (`NotFound` for an absent header) and the delta's
///   (`ServiceError::from(DeltaError)`), unchanged.
pub async fn plan_structural_commit(
    store: &LayerStore,
    request: PlanRequest<'_>,
) -> Result<PlannedCommit, ServiceError> {
    let keys = store.keys();
    let oid_str = request.object_id.to_string();
    let header_key = keys.obj_header(&oid_str);

    let before = read_fence_columns(store, &header_key).await?;
    let snapshot = match read_object_snapshot(
        store.client(),
        keys,
        &oid_str,
        SnapshotReadConsumer::Checkpoint,
    )
    .await?
    {
        SnapshotRead::Snapshot(snapshot) => snapshot,
        SnapshotRead::TooLarge(too_large) => {
            return Err(ServiceError::Internal(format!(
                "structural planner: object {oid_str} is above the atomic read's limits \
                 ({too_large:?})"
            )))
        }
    };
    let after = read_fence_columns(store, &header_key).await?;
    if before != after || before.0.as_deref() != Some(snapshot.cut.to_string().as_str()) {
        return Err(ServiceError::Aborted(format!(
            "structural planner: object {oid_str} changed during its read (header before \
             {before:?}, snapshot cut {}, header after {after:?})",
            snapshot.cut
        )));
    }

    let pre_state = snapshot_to_object_state(&snapshot.state);
    let mut post_state = pre_state.clone();
    let mutations = apply_delta(&mut post_state, request.delta).map_err(ServiceError::from)?;
    let pre_image_digests = PreImageDigests::from_plan(&mutations, &pre_state);

    Ok(PlannedCommit {
        commit: StructuralCommit {
            session_id: request.session_id.to_owned(),
            layer_name: request.layer_name.to_owned(),
            object_id: request.object_id,
            expected_seq: snapshot.cut,
            expected_commit_chain: before.1.unwrap_or_default(),
            next_set_order: post_state.next_set_order,
            mutations,
            pre_image_digests,
            relay_ompf: request.relay_ompf,
            commit_token: request.commit_token.to_owned(),
            operation: None,
            client_id: request.client_id.to_owned(),
            client_ts: request.client_ts,
        },
        pre_state,
        post_state,
    })
}

/// The header's `seq` and `commit_chain`, as stored.
async fn read_fence_columns(
    store: &LayerStore,
    header_key: &str,
) -> Result<(Option<String>, Option<String>), ServiceError> {
    let columns: Vec<Option<String>> = store
        .client()
        .pool
        .hmget(header_key, vec!["seq", HEADER_FIELD_COMMIT_CHAIN])
        .await
        .map_err(|e| {
            map_snapshot_load_err(
                FredErrOrigin::client(store.client(), POOL_LABEL_DATA),
                "HMGET obj header seq commit_chain (structural planner)",
                e,
            )
        })?;
    let mut columns = columns.into_iter();
    Ok((columns.next().flatten(), columns.next().flatten()))
}

// ── The extended object-keyspace dump ───────────────────────────────────────

/// Header columns [`dump_object_keyspace`] leaves out: they differ between
/// two fixtures by design (wall clocks, and the per-materialization random
/// `commit_chain`), so a cross-fixture compare could never match on them. A
/// test that needs `commit_chain` reads it itself.
pub const DUMP_DROPPED_HEADER_COLUMNS: [&str; 4] = [
    "created_at",
    "modified_at",
    "last_server_ts",
    HEADER_FIELD_COMMIT_CHAIN,
];

/// The placeholder [`dump_object_keyspace`] writes for a wall-clock server
/// timestamp (in stream payloads and commit-token records).
pub const DUMP_SERVER_TS: &str = "<server_ts>";

/// The placeholder [`dump_object_keyspace`] writes for the object id in
/// stream payloads.
pub const DUMP_OBJECT_ID: &str = "<object_id>";

/// The placeholder [`dump_object_keyspace`] writes for a commit token under
/// [`CommitTokens::Normalized`].
pub const DUMP_COMMIT_TOKEN: &str = "<token>";

/// How [`dump_object_keyspace`] renders the commit token in each `obj:wtok`
/// record (`<token>:<seq>:<server_ts>`, or
/// `op.<operation_id>:<seq>:<server_ts>:<token>:<request_digest>`), and the
/// call tokens in the `obj:wrel` commit release ring (its `|t|<token>` field
/// names and `|s|<k>` values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitTokens {
    /// Keep the token: for compares where both sides used fixed tokens, and
    /// for before / after compares of one object.
    Verbatim,
    /// Replace it with [`DUMP_COMMIT_TOKEN`] in `obj:wtok`, and with
    /// `<token:k>` in `obj:wrel` ([`dump_release_ring_token`]): for compares
    /// of fixtures that minted their own random tokens.
    Normalized,
}

/// The placeholder [`dump_object_keyspace`] writes under
/// [`CommitTokens::Normalized`] for the `k`-th distinct call token of an
/// `obj:wrel` commit release ring: `<token:k>`. Tokens are numbered from 0 in
/// the order of their start slot (`|s|<slot>`), lowest slot first; a token
/// named by an entry field but held by no slot (which the ring's write rule
/// never leaves) is numbered after them, in token order. Distinct
/// placeholders, unlike `obj:wtok`'s one [`DUMP_COMMIT_TOKEN`], because the
/// ring's tokens are field names: one placeholder would merge its entries.
pub fn dump_release_ring_token(k: usize) -> String {
    format!("<token:{k}>")
}

/// One object's whole keyspace: every key under the object's `{object_id}`
/// hash tag, by its name relative to the object (`obj:hdr`, `obj:sec:5`,
/// `obj:ch:0`, `obj:stream`, …), with its value.
///
/// Every key the object holds is here — header, `obj:flds` / `obj:tss` with
/// their scores, field and time-sample rows, `obj:sec:*`, `obj:secs`,
/// `obj:ch:*` (root included, in list order), `obj:chs`, `obj:del:*`,
/// `obj:dels`, `obj:ext`, `obj:wtok`, the stream, and anything else — found by
/// scanning the tag rather than walking the indexes, so a key no index names
/// still shows. Values that differ between fixtures by design are normalized
/// (see [`dump_object_keyspace`]), so two objects built the same way by two
/// routes compare equal, and a refused commit's before and after dumps are
/// equal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectKeyspaceDump {
    /// Relative key name → value.
    pub keys: BTreeMap<String, DumpedValue>,
}

/// One key's value in an [`ObjectKeyspaceDump`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DumpedValue {
    /// A hash, by column.
    Hash(BTreeMap<String, String>),
    /// A set.
    Set(BTreeSet<String>),
    /// A sorted set: `(member, score)` in member order, the score in its
    /// shortest decimal form (`6`, not `6.0`).
    SortedSet(Vec<(String, String)>),
    /// A list, in list order.
    List(Vec<String>),
    /// A stream, in entry order.
    Stream(Vec<DumpedStreamEntry>),
    /// A string's bytes.
    String(Vec<u8>),
    /// A type the dump does not render, by its `TYPE` name.
    Other(String),
}

/// One stream entry in an [`ObjectKeyspaceDump`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DumpedStreamEntry {
    /// The entry ID (`<seq>-0` for a commit).
    pub id: String,
    /// The `payload` field, normalized: in an `update|…` payload the object id
    /// becomes [`DUMP_OBJECT_ID`] and the server timestamp [`DUMP_SERVER_TS`];
    /// any other payload (a retirement's `deleted`) is verbatim. `None` when
    /// the entry has no `payload` field.
    pub payload: Option<String>,
    /// The `ompf` field's bytes, verbatim; `None` when absent (an empty
    /// commit).
    pub ompf: Option<Vec<u8>>,
    /// Every other field, verbatim.
    pub other_fields: BTreeMap<String, Vec<u8>>,
}

impl ObjectKeyspaceDump {
    /// The value at `relative_key` (`obj:hdr`, `obj:sec:5`, …), if the key
    /// exists.
    pub fn get(&self, relative_key: &str) -> Option<&DumpedValue> {
        self.keys.get(relative_key)
    }
}

/// Dump object `object_id`'s whole keyspace (see [`ObjectKeyspaceDump`]).
///
/// Normalizations, each because the value differs between fixtures by design:
///
/// - `obj:hdr` drops [`DUMP_DROPPED_HEADER_COLUMNS`];
/// - each `obj:wtok` record — `<token>:<seq>:<server_ts>`, or
///   `op.<operation_id>:<seq>:<server_ts>:<token>:<request_digest>` for a
///   call with a client operation id — keeps its `seq` (and the operation's
///   match key and request digest), has its server timestamp replaced by
///   [`DUMP_SERVER_TS`], and its commit token kept or replaced per `tokens`;
///   a record of neither shape is kept verbatim;
/// - under [`CommitTokens::Normalized`], each call token in the `obj:wrel`
///   commit release ring — in its `|t|<token>` entry field names and its
///   `|s|<k>` slot values — is replaced by [`dump_release_ring_token`]'s
///   `<token:k>`, numbered by start slot, so two fixtures that wrote the ring
///   in the same order compare equal;
/// - each stream entry's `update|…` payload has its object id and server
///   timestamp replaced (see [`DumpedStreamEntry::payload`]).
///
/// Panics on a failed read: test support only.
pub async fn dump_object_keyspace(
    client: &crate::storage::client::ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    tokens: CommitTokens,
) -> ObjectKeyspaceDump {
    let oid_str = object_id.to_string();
    // Every object key is `<prefix>{<oid>}:obj:…`; the stem is everything up
    // to and including the colon after the hash tag.
    let header_key = keys.obj_header(&oid_str);
    let stem = header_key
        .strip_suffix("obj:hdr")
        .expect("the header key ends in `obj:hdr`")
        .to_owned();

    let pool = &client.pool;
    let mut dump = BTreeMap::new();
    for key in scan_keys(pool, &format!("{}*", escape_glob(&stem))).await {
        let relative = key
            .strip_prefix(&stem)
            .expect("a scanned key starts with the pattern's stem")
            .to_owned();
        let key_type: String = pool
            .r#type(&key)
            .await
            .unwrap_or_else(|e| panic!("TYPE {key} failed: {e}"));
        let value = match key_type.as_str() {
            "hash" => {
                let mut map: BTreeMap<String, String> = pool
                    .hgetall::<HashMap<String, String>, _>(&key)
                    .await
                    .unwrap_or_else(|e| panic!("HGETALL {key} failed: {e}"))
                    .into_iter()
                    .collect();
                if relative == "obj:hdr" {
                    for column in DUMP_DROPPED_HEADER_COLUMNS {
                        map.remove(column);
                    }
                }
                if relative == "obj:wtok" {
                    for record in map.values_mut() {
                        *record = normalize_commit_token_record(record, tokens);
                    }
                }
                if relative == "obj:wrel" && tokens == CommitTokens::Normalized {
                    map = normalize_release_ring(map);
                }
                DumpedValue::Hash(map)
            }
            "set" => DumpedValue::Set(
                pool.smembers::<Vec<String>, _>(&key)
                    .await
                    .unwrap_or_else(|e| panic!("SMEMBERS {key} failed: {e}"))
                    .into_iter()
                    .collect(),
            ),
            "zset" => {
                let mut members: Vec<(String, f64)> = pool
                    .zrange(&key, 0, -1, None, false, None, true)
                    .await
                    .unwrap_or_else(|e| panic!("ZRANGE {key} failed: {e}"));
                members.sort_by(|a, b| a.0.cmp(&b.0));
                DumpedValue::SortedSet(
                    members
                        .into_iter()
                        .map(|(member, score)| (member, score.to_string()))
                        .collect(),
                )
            }
            "list" => DumpedValue::List(
                pool.lrange::<Vec<String>, _>(&key, 0, -1)
                    .await
                    .unwrap_or_else(|e| panic!("LRANGE {key} failed: {e}")),
            ),
            "stream" => {
                let entries: Vec<(String, HashMap<String, Bytes>)> = pool
                    .xrange_values(&key, "-", "+", None)
                    .await
                    .unwrap_or_else(|e| panic!("XRANGE {key} failed: {e}"));
                DumpedValue::Stream(
                    entries
                        .into_iter()
                        .map(|(id, fields)| dump_stream_entry(id, fields, &oid_str))
                        .collect(),
                )
            }
            "string" => DumpedValue::String(
                pool.get::<Option<Bytes>, _>(&key)
                    .await
                    .unwrap_or_else(|e| panic!("GET {key} failed: {e}"))
                    .map(|bytes| bytes.to_vec())
                    .unwrap_or_default(),
            ),
            other => DumpedValue::Other(other.to_owned()),
        };
        dump.insert(relative, value);
    }
    ObjectKeyspaceDump { keys: dump }
}

/// The `SCAN` page hint [`scan_keys`] walks the database with.
///
/// The dump has to walk the whole database: a standalone server keeps no
/// index of the keys under one hash tag, and the dump exists to catch keys no
/// object index names. So its cost is one round trip per `COUNT` keys in the
/// database — the shared test Valkey's, not the object's. Measured against a
/// ~43k-key database: at 100 a dump took ~80 ms, most of a property case's
/// cost, and the 256-case keyspace-equivalence property (`tests/proptests.rs`)
/// ran ~65 s; at 1 000 it ran ~18 s, and 10 000 was no faster, so 1 000 keeps
/// each call's hold on the shared server short.
const DUMP_SCAN_COUNT: u32 = 1_000;

/// Every key matching `pattern`, sorted. Walks every shard on a cluster.
async fn scan_keys(pool: &fred::clients::Pool, pattern: &str) -> Vec<String> {
    let mut stream = scan_matching(pool.next(), pattern, DUMP_SCAN_COUNT);
    let mut found = Vec::new();
    while let Some(key) = stream.next().await {
        let key = key.unwrap_or_else(|e| panic!("SCAN {pattern} failed: {e}"));
        found.push(
            key.into_string()
                .unwrap_or_else(|| panic!("a key under {pattern} is not UTF-8")),
        );
    }
    found.sort();
    found.dedup();
    found
}

/// `s` with every glob metacharacter `SCAN MATCH` honours escaped.
fn escape_glob(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(c, '*' | '?' | '[' | ']' | '\\') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// A commit-token record with its server timestamp, and per `tokens` its
/// commit token, replaced, in either record shape:
///
/// - `<token>:<seq>:<server_ts>`, a call without a client operation id;
/// - `op.<operation_id>:<seq>:<server_ts>:<token>:<request_digest>`, a call
///   with one (plan b.v6d): the match key, `seq` and request digest are kept,
///   the call's commit token is the one replaced.
///
/// Verbatim when it is neither.
fn normalize_commit_token_record(record: &str, tokens: CommitTokens) -> String {
    let token = |token: &str| match tokens {
        CommitTokens::Verbatim => token.to_owned(),
        CommitTokens::Normalized => DUMP_COMMIT_TOKEN.to_owned(),
    };
    let parts: Vec<&str> = record.split(':').collect();
    match parts.as_slice() {
        [call_token, seq, _server_ts] => {
            format!("{}:{seq}:{DUMP_SERVER_TS}", token(call_token))
        }
        [key, seq, _server_ts, call_token, digest]
            if key.starts_with(crate::storage::layer::OPERATION_KEY_PREFIX) =>
        {
            format!(
                "{key}:{seq}:{DUMP_SERVER_TS}:{}:{digest}",
                token(call_token)
            )
        }
        _ => record.to_owned(),
    }
}

/// An `obj:wrel` commit release ring with every call token replaced by its
/// [`dump_release_ring_token`] placeholder, in both the `|t|<token>` entry
/// field names and the `|s|<k>` slot values. Every other field — `|n|`, an
/// empty slot, a field of another shape — is kept verbatim.
fn normalize_release_ring(ring: BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut starts: Vec<(usize, &str)> = ring
        .iter()
        .filter_map(|(field, value)| {
            let slot = field.strip_prefix("|s|")?.parse::<usize>().ok()?;
            (!value.is_empty()).then_some((slot, value.as_str()))
        })
        .collect();
    starts.sort_unstable();
    let mut numbered: HashMap<String, usize> = HashMap::new();
    for (_, token) in starts {
        let next = numbered.len();
        numbered.entry(token.to_owned()).or_insert(next);
    }
    // `BTreeMap` iterates in field order, so entry-only tokens are numbered
    // in token order.
    for field in ring.keys() {
        if let Some(token) = field.strip_prefix("|t|") {
            let next = numbered.len();
            numbered.entry(token.to_owned()).or_insert(next);
        }
    }
    ring.into_iter()
        .map(|(field, value)| {
            if let Some(token) = field.strip_prefix("|t|") {
                (
                    format!("|t|{}", dump_release_ring_token(numbered[token])),
                    value,
                )
            } else if field.starts_with("|s|") && !value.is_empty() {
                let placeholder = numbered
                    .get(&value)
                    .map_or(value.clone(), |k| dump_release_ring_token(*k));
                (field, placeholder)
            } else {
                (field, value)
            }
        })
        .collect()
}

/// One stream entry, its payload normalized (see [`DumpedStreamEntry`]).
fn dump_stream_entry(
    id: String,
    mut fields: HashMap<String, Bytes>,
    oid_str: &str,
) -> DumpedStreamEntry {
    let payload = fields.remove(NOTIFICATION_PAYLOAD_FIELD).map(|raw| {
        let raw = String::from_utf8_lossy(&raw).into_owned();
        let mut parts: Vec<&str> = raw.split('|').collect();
        if parts.len() == 6 && parts[0] == "update" {
            if parts[1] == oid_str {
                parts[1] = DUMP_OBJECT_ID;
            }
            parts[5] = DUMP_SERVER_TS;
            parts.join("|")
        } else {
            raw
        }
    });
    let ompf = fields.remove(RELAY_OMPF_FIELD).map(|bytes| bytes.to_vec());
    let other_fields = fields
        .into_iter()
        .map(|(name, value)| (name, value.to_vec()))
        .collect();
    DumpedStreamEntry {
        id,
        payload,
        ompf,
        other_fields,
    }
}
