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

//! The store reads of the disaster recovery Jobs (`crate::dr::job`): the
//! store's clock, the holder of the Job lease and the evidence the absence
//! rule's contradiction check reads. Every function here reads;
//! the Job's one pre-quiet-period write, its lease, goes through
//! `storage::external_values`' lease scripts. The restore writer's store
//! writes live elsewhere in `storage`: the mapping compare-and-sets and
//! path-index members in `storage::dr_write`, and the object writes in
//! `storage::layer::dr_restore`; its stream high-water probe is
//! `storage::stream_high_water`'s.
//!
//! - [`store_time_ms`] — `TIME` on the node that holds the hold marker, the
//!   clock the marker's `detected_at_ms` was written in, so the quiet period
//!   and `T_safe` are measured on one clock.
//! - [`read_lock_holder`] — `GET` and `PTTL` of one lock key, for the Job
//!   lease's lost-holder log line (`crate::dr::job::lease`).
//! - [`read_checkpoint_stamp`], [`read_session_backup_kind`] and
//!   [`read_dead_object_ids`] — the contradiction check's evidence.
//! - The restore planner's reads (`crate::dr::job::plan`), all taken after the
//!   quiet period ends: [`read_layer_mappings`] and [`read_object_headers`] —
//!   one agreeing keyspace walk each
//!   (`storage::scan_guard::collect_keyspace_agreed`: two walks, the second
//!   after the first, each sending each primary's commands to one named node
//!   and failing when the slot map changed or a node pin missed its
//!   primary's slots, or when its `CLUSTER NODES` gives no usable myself
//!   line, and the keys used only when the two walks returned the same set),
//!   then one read per key; [`read_session_facts`] — one pipeline per session;
//!   [`read_object_ext_hashes`] — a surviving object's `obj:ext` fields; and
//!   [`ext_value_stored`] — `EXISTS` on one external value's blob key, which
//!   never reads the blob's bytes.
//! - Accept-loss's read before its discards: [`read_object_header`] — one
//!   object's header.

use std::collections::{HashMap, HashSet};

use fred::interfaces::{
    ClientLike, HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface,
};
use fred::types::config::Options;
use fred::types::{ClusterHash, CustomCommand, Value};
use futures::{StreamExt, TryStreamExt};

use crate::storage::client::{
    FredErrOrigin, ValkeyClient, POOL_LABEL_DATA, POOL_LABEL_MAINTENANCE,
};
use crate::storage::keys::{hex_decode, hex_encode, KeySchema};
use crate::storage::layer::{
    header_fields_are_structurally_complete, HEADER_FIELD_CREATED_BY, HEADER_FIELD_SEQ,
    HEADER_FIELD_SUPERSEDED, HEADER_FIELD_SUPERSEDED_BY,
};
use crate::storage::orphan_sweep::extract_hash_tag_id;
use crate::storage::scan_guard::{collect_keyspace_agreed, WalkError};

/// The store's clock, in milliseconds: `TIME`, pinned to the hold marker's
/// slot ([`KeySchema::dr_hold_marker`]), so it runs on the node whose clock
/// the marker's `detected_at_ms` was written in (`TIME` is keyless; fred
/// would otherwise send it to an arbitrary primary).
///
/// One command on the data pool under its command timeout; a timeout is
/// counted once on the data pool. A reply that is not two integers is a
/// `Protocol` error.
pub async fn store_time_ms(
    client: &ValkeyClient,
    keys: &KeySchema,
) -> Result<u64, fred::error::Error> {
    let pin = Options {
        cluster_hash: Some(ClusterHash::from(keys.dr_hold_marker().as_str())),
        ..Default::default()
    };
    let reply: Vec<String> = client
        .pool
        .next()
        .with_options(&pin)
        .custom(
            CustomCommand::new_static("TIME", ClusterHash::FirstKey, false),
            Vec::<Value>::new(),
        )
        .await
        .inspect_err(|e| FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(e))?;
    decode_time_reply(&reply)
}

/// `TIME`'s two decimal integers, seconds and microseconds, as milliseconds.
fn decode_time_reply(reply: &[String]) -> Result<u64, fred::error::Error> {
    match reply {
        [secs, usecs] => match (secs.parse::<u64>(), usecs.parse::<u64>()) {
            (Ok(s), Ok(us)) => Ok(s.saturating_mul(1000).saturating_add(us / 1000)),
            _ => Err(protocol_error(format!("TIME replied {reply:?}"))),
        },
        _ => Err(protocol_error(format!("TIME replied {reply:?}"))),
    }
}

/// One lock key as [`read_lock_holder`] reads it: its value and its
/// remaining time to live.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LockHolder {
    /// The lock's value: its holder's lock value.
    pub value: String,
    /// `PTTL` after the `GET`, in milliseconds: `None` when the key carries
    /// no expiry or vanished between the two commands.
    pub pttl_ms: Option<u64>,
}

/// Read one lock key: `Some` with its value and remaining TTL when the key
/// exists, `None` when it does not.
///
/// Two single-key commands, `GET` then `PTTL`, on the data pool. The `PTTL`
/// is for the log only: a key that expires between the two commands is still
/// reported as held, since the `GET` saw it.
pub async fn read_lock_holder(
    client: &ValkeyClient,
    lock_key: &str,
) -> Result<Option<LockHolder>, fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_DATA);
    let value: Option<String> = client
        .pool
        .next()
        .get(lock_key)
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    let Some(value) = value else {
        return Ok(None);
    };
    let pttl: i64 = client
        .pool
        .next()
        .pttl(lock_key)
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    Ok(Some(LockHolder {
        value,
        pttl_ms: u64::try_from(pttl).ok(),
    }))
}

/// A surviving object header's checkpoint stamp: what `stamp_cursor_if_live.lua`
/// wrote after a backup generation of the object landed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointStamp {
    /// The object.
    pub object_id: u64,
    /// The header's `session_id`, empty when absent.
    pub session_id: String,
    /// The header's `layer_name`, empty when absent.
    pub layer_name: String,
    /// `last_checkpoint_seq`, when present and a decimal integer.
    pub last_checkpoint_seq: Option<u64>,
    /// `last_checkpoint_ms`, when present and a decimal integer.
    pub last_checkpoint_ms: Option<u64>,
}

impl CheckpointStamp {
    /// Whether the header carries a checkpoint stamp at all: a backup
    /// generation of this object was written and stamped.
    pub fn is_stamped(&self) -> bool {
        self.last_checkpoint_seq.is_some() || self.last_checkpoint_ms.is_some()
    }
}

/// Read `object_id`'s header (`le:{oid}:obj:hdr`) for the contradiction
/// check: `None` when the header is absent or carries no `seq` (deleted, or a
/// cursor-only phantom), otherwise its identity and checkpoint stamp.
///
/// One `HMGET` on the data pool.
pub async fn read_checkpoint_stamp(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
) -> Result<Option<CheckpointStamp>, fred::error::Error> {
    let key = keys.obj_header(&object_id.to_string());
    let fields: Vec<Option<String>> = client
        .pool
        .next()
        .hmget(
            &key,
            vec![
                "seq",
                "session_id",
                "layer_name",
                "last_checkpoint_seq",
                "last_checkpoint_ms",
            ],
        )
        .await
        .inspect_err(|e| FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(e))?;
    Ok(stamp_from_fields(object_id, &fields))
}

fn stamp_from_fields(object_id: u64, fields: &[Option<String>]) -> Option<CheckpointStamp> {
    let field = |i: usize| fields.get(i).cloned().flatten();
    field(0)?;
    let number = |i: usize| field(i).and_then(|s| s.trim().parse::<u64>().ok());
    Some(CheckpointStamp {
        object_id,
        session_id: field(1).unwrap_or_default(),
        layer_name: field(2).unwrap_or_default(),
        last_checkpoint_seq: number(3),
        last_checkpoint_ms: number(4),
    })
}

/// The kind of the last session-checkpoint operation to land for
/// `session_id`, from its publish watermark (`le:{sid}:ckpt`'s `pub_kind`,
/// written by `session_checkpoint_publish.lua` only after the backup store
/// accepted the operation): `Some("write")` when the last landed operation
/// stored a session backup, `Some("delete")` when it removed one, `None`
/// when the key or field is absent.
///
/// Absence is no evidence: the key carries a TTL (a day after its last
/// touch) and is lost with the keyspace. Only `Some("write")` is evidence,
/// and it is positive: a session backup was stored in the backup store.
///
/// One `HGET` on the data pool.
pub async fn read_session_backup_kind(
    client: &ValkeyClient,
    keys: &KeySchema,
    session_id: &str,
) -> Result<Option<String>, fred::error::Error> {
    client
        .pool
        .next()
        .hget(keys.session_checkpoint_state(session_id), "pub_kind")
        .await
        .inspect_err(|e| FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(e))
}

/// Every object id on the checkpoint dead-object set
/// (`le:ckpt:{recon}:dead`, [`KeySchema::checkpoint_reconcile_dead`]): the
/// objects a code path observed deleted. A member that is not a decimal
/// object id is skipped.
///
/// One `ZRANGE 0 -1` on the data pool.
pub async fn read_dead_object_ids(
    client: &ValkeyClient,
    keys: &KeySchema,
) -> Result<HashSet<u64>, fred::error::Error> {
    let members: Vec<String> = client
        .pool
        .next()
        .zrange(
            keys.checkpoint_reconcile_dead(),
            0,
            -1,
            None,
            false,
            None,
            false,
        )
        .await
        .inspect_err(|e| FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(e))?;
    Ok(members
        .iter()
        .filter_map(|m| m.trim().parse::<u64>().ok())
        .collect())
}

/// The `obj:hdr` fields the restore planner reads per surviving object
/// ([`read_object_headers`]), in this order. It holds every member of
/// [`crate::storage::layer::HEADER_STRUCTURAL_FIELDS`], so
/// [`HeaderFacts::is_structurally_complete`] sees each one (a unit test
/// below pins it).
pub const HEADER_FACT_FIELDS: [&str; 9] = [
    HEADER_FIELD_SEQ,
    HEADER_FIELD_CREATED_BY,
    "session_id",
    "layer_name",
    HEADER_FIELD_SUPERSEDED,
    HEADER_FIELD_SUPERSEDED_BY,
    "ext_gc_scheduled",
    "last_checkpoint_seq",
    "last_checkpoint_ms",
];

/// One surviving object header, as the restore planner reads it: the present
/// fields among [`HEADER_FACT_FIELDS`]. Pure accessors only.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeaderFacts {
    /// The object.
    pub object_id: u64,
    /// The present fields among [`HEADER_FACT_FIELDS`], by name.
    pub fields: HashMap<String, String>,
}

impl HeaderFacts {
    /// Whether the header is structurally complete
    /// (`storage::layer::header_fields_are_structurally_complete` over the
    /// present fields, which include every structural one).
    pub fn is_structurally_complete(&self) -> bool {
        header_fields_are_structurally_complete(&self.fields)
    }

    /// `seq`, when present and a decimal integer.
    pub fn seq(&self) -> Option<u64> {
        self.number(HEADER_FIELD_SEQ)
    }

    /// `session_id`, empty when absent.
    pub fn session_id(&self) -> &str {
        self.fields.get("session_id").map_or("", String::as_str)
    }

    /// `layer_name`, empty when absent.
    pub fn layer_name(&self) -> &str {
        self.fields.get("layer_name").map_or("", String::as_str)
    }

    /// Whether the object is being retired: `superseded` or
    /// `ext_gc_scheduled` is present, whatever its value.
    pub fn is_retiring(&self) -> bool {
        self.fields.contains_key(HEADER_FIELD_SUPERSEDED)
            || self.fields.contains_key("ext_gc_scheduled")
    }

    /// `superseded_by`, the successor a `CreateLayer` overwrite's fence
    /// named, when present and a decimal object id.
    pub fn superseded_by(&self) -> Option<u64> {
        self.number(HEADER_FIELD_SUPERSEDED_BY)
    }

    /// The checkpoint cursor columns, `(last_checkpoint_seq,
    /// last_checkpoint_ms)`, each when present and a decimal integer.
    pub fn cursor(&self) -> (Option<u64>, Option<u64>) {
        (
            self.number("last_checkpoint_seq"),
            self.number("last_checkpoint_ms"),
        )
    }

    /// The header as the contradiction check reads it
    /// ([`read_checkpoint_stamp`]'s shape): `None` without `seq`.
    pub fn stamp(&self) -> Option<CheckpointStamp> {
        self.seq()?;
        let (last_checkpoint_seq, last_checkpoint_ms) = self.cursor();
        Some(CheckpointStamp {
            object_id: self.object_id,
            session_id: self.session_id().to_owned(),
            layer_name: self.layer_name().to_owned(),
            last_checkpoint_seq,
            last_checkpoint_ms,
        })
    }

    fn number(&self, field: &str) -> Option<u64> {
        self.fields
            .get(field)
            .and_then(|v| v.trim().parse::<u64>().ok())
    }
}

/// One surviving layer mapping (`le:{sid}:layer:{hex(name)}`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerMappingFact {
    /// The session the key's hash tag names.
    pub session_id: String,
    /// The layer name, hex-decoded from the key.
    pub layer_name: String,
    /// The object the mapping names.
    pub object_id: u64,
}

/// The planner's agreeing keyspace walk of `pattern` on one
/// maintenance-pool client (`storage::scan_guard::collect_keyspace_agreed`:
/// two walks, compared before any per-key read), each key once, a store
/// fault's fred error counted on the maintenance pool. A change of slot map,
/// a node pin that missed its primary's slots, a missing myself line or two
/// walks that disagree is not a fred error and is not counted.
async fn planner_keyspace_walk(
    client: &ValkeyClient,
    pool: &fred::clients::Pool,
    pattern: &str,
) -> Result<Vec<fred::types::Key>, WalkError> {
    collect_keyspace_agreed(pool.next(), pattern)
        .await
        .inspect_err(|e| {
            if let Some(source) = e.fred_error() {
                FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE).observe_error(source);
            }
        })
}

/// A per-key read of the planner's, failed: a store fault named by its
/// command.
fn planner_read_failed(op: &'static str) -> impl Fn(fred::error::Error) -> WalkError {
    move |source| WalkError::Read { op, source }
}

/// Every surviving object header: one agreeing keyspace walk of
/// [`KeySchema::obj_header_scan_pattern`]
/// (`storage::scan_guard::collect_keyspace_agreed`: two walks, each complete
/// on every primary, each primary's commands sent to one named node whose
/// own myself line lists exactly that primary's slots at the start and the
/// end, its slot map unchanged from start to end, and the two walks'
/// key sets equal, or an error), then one `HMGET` of [`HEADER_FACT_FIELDS`]
/// per header, at most `concurrency` in flight, all on the maintenance pool.
/// A key whose hash tag is not a decimal object id, and a header gone by its
/// `HMGET`, are left out.
///
/// An error is a store fault ([`WalkError::Step`] or [`WalkError::Read`]),
/// a walk whose slot map changed or whose node pin missed its primary's
/// slots ([`WalkError::Changed`]), two walks that disagree
/// ([`WalkError::Incomplete`]), or a missing myself line
/// ([`WalkError::WitnessMissing`]); the planner exits 1 on the first three
/// and 3 on the last.
pub async fn read_object_headers(
    client: &ValkeyClient,
    keys: &KeySchema,
    concurrency: usize,
) -> Result<Vec<HeaderFacts>, WalkError> {
    let (pool, _token) = client.acquire_maintenance();
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let found = planner_keyspace_walk(client, pool, &keys.obj_header_scan_pattern()).await?;
    let prefix = keys.prefix();
    let object_ids: Vec<u64> = found
        .iter()
        .filter_map(|key| key.as_str())
        .filter_map(|key| extract_hash_tag_id(key, prefix, ":obj:hdr"))
        .filter_map(|id| id.parse::<u64>().ok())
        .collect::<HashSet<u64>>()
        .into_iter()
        .collect();
    let headers: Vec<Option<HeaderFacts>> = futures::stream::iter(object_ids)
        .map(|object_id| async move {
            let values: Vec<Option<String>> = pool
                .next()
                .hmget(
                    keys.obj_header(&object_id.to_string()),
                    HEADER_FACT_FIELDS.to_vec(),
                )
                .await
                .inspect_err(|e| origin.observe_error(e))
                .map_err(planner_read_failed("HMGET obj:hdr"))?;
            let fields: HashMap<String, String> = HEADER_FACT_FIELDS
                .iter()
                .zip(values)
                .filter_map(|(name, value)| value.map(|v| ((*name).to_owned(), v)))
                .collect();
            Ok::<_, WalkError>((!fields.is_empty()).then_some(HeaderFacts { object_id, fields }))
        })
        .buffer_unordered(concurrency.max(1))
        .try_collect()
        .await?;
    let mut headers: Vec<HeaderFacts> = headers.into_iter().flatten().collect();
    headers.sort_by_key(|h| h.object_id);
    Ok(headers)
}

/// Every surviving layer mapping: one agreeing keyspace walk of
/// [`KeySchema::layer_mapping_scan_pattern`], as [`read_object_headers`]
/// walks (and fails), then one `GET` per mapping, at most `concurrency` in
/// flight, all on the maintenance pool. The session id is the key's hash tag
/// and the layer name its hex-decoded suffix. A mapping gone by its `GET` is
/// left out; one whose value is not a decimal object id is left out with a
/// WARN (it names no object a restore could plan).
pub async fn read_layer_mappings(
    client: &ValkeyClient,
    keys: &KeySchema,
    concurrency: usize,
) -> Result<Vec<LayerMappingFact>, WalkError> {
    let (pool, _token) = client.acquire_maintenance();
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    let found = planner_keyspace_walk(client, pool, &keys.layer_mapping_scan_pattern()).await?;
    let prefix = keys.prefix();
    let mut parsed: Vec<(String, String, String)> = Vec::with_capacity(found.len());
    for key in &found {
        let Some(key) = key.as_str() else { continue };
        let Some(session_id) = extract_hash_tag_id(key, prefix, ":layer:") else {
            continue;
        };
        // The empty layer name hex-encodes to nothing, so this is exactly the
        // key's `{prefix}{session_id}:layer:` head, hash-tag braces included.
        let head = keys.layer_mapping(&session_id, "");
        let Some(hex_name) = key.strip_prefix(&head) else {
            continue;
        };
        parsed.push((key.to_owned(), session_id, hex_decode(hex_name)));
    }
    let mappings: Vec<Option<LayerMappingFact>> = futures::stream::iter(parsed)
        .map(|(key, session_id, layer_name)| async move {
            let value: Option<String> = pool
                .next()
                .get(&key)
                .await
                .inspect_err(|e| origin.observe_error(e))
                .map_err(planner_read_failed("GET layer mapping"))?;
            let Some(value) = value else {
                return Ok(None);
            };
            match value.trim().parse::<u64>() {
                Ok(object_id) => Ok::<_, WalkError>(Some(LayerMappingFact {
                    session_id,
                    layer_name,
                    object_id,
                })),
                Err(_) => {
                    tracing::warn!(
                        key = %key,
                        value = %value,
                        "a layer mapping names no decimal object id; the restore plan leaves it \
                         out"
                    );
                    Ok(None)
                }
            }
        })
        .buffer_unordered(concurrency.max(1))
        .try_collect()
        .await?;
    let mut mappings: Vec<LayerMappingFact> = mappings.into_iter().flatten().collect();
    mappings.sort_by(|a, b| {
        (a.session_id.as_str(), a.layer_name.as_str())
            .cmp(&(b.session_id.as_str(), b.layer_name.as_str()))
    });
    Ok(mappings)
}

/// What the store holds for one session, as the restore planner reads it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionFacts {
    /// `le:{sid}:sess` exists.
    pub meta_exists: bool,
    /// `le:{sid}:sess:deleted` exists: a `DeleteSession` removed the session.
    pub deleted_marker_exists: bool,
    /// `le:{sid}:ckpt`'s `pub_kind` ([`read_session_backup_kind`]'s answer).
    pub backup_kind: Option<String>,
    /// The layer names `le:{sid}:layers`, the session's layers index, holds
    /// (its hex-encoded members, decoded), sorted: the names an abandoned
    /// delete removes even when no mapping or header names them any more. A
    /// member that is not UTF-8, or that does not decode to a name encoding
    /// back to it, is left out, since no name removes it.
    pub layers_index: Vec<String>,
}

/// One session's [`SessionFacts`]: `EXISTS` of its meta and of its deletion
/// marker, `HGET` of its checkpoint state's `pub_kind` and `SMEMBERS` of its
/// layers index, in one pipeline on the data pool (the four keys share the
/// `{session_id}` slot).
pub async fn read_session_facts(
    client: &ValkeyClient,
    keys: &KeySchema,
    session_id: &str,
) -> Result<SessionFacts, fred::error::Error> {
    let origin = FredErrOrigin::client(client, POOL_LABEL_DATA);
    let pipeline = client.pool.next().pipeline();
    let _: () = pipeline.exists(keys.session_meta(session_id)).await?;
    let _: () = pipeline
        .exists(keys.session_deleted_marker(session_id))
        .await?;
    let _: () = pipeline
        .hget(keys.session_checkpoint_state(session_id), "pub_kind")
        .await?;
    let _: () = pipeline.smembers(keys.layers_index(session_id)).await?;
    let replies: Vec<Value> = pipeline
        .all()
        .await
        .inspect_err(|e| origin.observe_error(e))?;
    session_facts_from_replies(replies)
}

fn session_facts_from_replies(replies: Vec<Value>) -> Result<SessionFacts, fred::error::Error> {
    let [meta, marker, kind, layers]: [Value; 4] = replies
        .try_into()
        .map_err(|r: Vec<Value>| protocol_error(format!("session facts replied {r:?}")))?;
    let count =
        |v: Value| -> Result<bool, fred::error::Error> { v.convert::<i64>().map(|n| n > 0) };
    // Each member as a `Value`, so one that is not UTF-8 is skipped rather
    // than failing the whole read (it covers live sessions too).
    let members: Vec<Value> = layers.convert()?;
    let mut layers_index: Vec<String> = members
        .into_iter()
        .filter_map(Value::into_string)
        .filter_map(|member| {
            let name = hex_decode(&member);
            (!name.is_empty() && hex_encode(&name) == member).then_some(name)
        })
        .collect();
    layers_index.sort();
    Ok(SessionFacts {
        meta_exists: count(meta)?,
        deleted_marker_exists: count(marker)?,
        backup_kind: kind.convert::<Option<String>>()?,
        layers_index,
    })
}

/// The external values a surviving object references: the fields of its
/// `obj:ext` index (`HKEYS`, lowercase hash hex). One command on the data
/// pool; an object without the index answers empty.
pub async fn read_object_ext_hashes(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
) -> Result<Vec<String>, fred::error::Error> {
    client
        .pool
        .next()
        .hkeys(keys.obj_ext_index(&object_id.to_string()))
        .await
        .inspect_err(|e| FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(e))
}

/// One object's header, as [`read_object_headers`] reads each: one `HMGET`
/// of [`HEADER_FACT_FIELDS`] on the data pool; `None` when no field is
/// present. Accept-loss's re-read of a lost layer's header before its wipe,
/// and of a value-removal object's `seq` and flags before its removal.
pub async fn read_object_header(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
) -> Result<Option<HeaderFacts>, fred::error::Error> {
    let values: Vec<Option<String>> = client
        .pool
        .next()
        .hmget(
            keys.obj_header(&object_id.to_string()),
            HEADER_FACT_FIELDS.to_vec(),
        )
        .await
        .inspect_err(|e| FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(e))?;
    let fields: HashMap<String, String> = HEADER_FACT_FIELDS
        .iter()
        .zip(values)
        .filter_map(|(name, value)| value.map(|v| ((*name).to_owned(), v)))
        .collect();
    Ok((!fields.is_empty()).then_some(HeaderFacts { object_id, fields }))
}

/// Whether Valkey holds the external value `hash_hex`'s blob: `EXISTS` on
/// its blob key, which never reads the blob's bytes (the restore planner's
/// coverage check, SR-9.3). One command on the data pool.
pub async fn ext_value_stored(
    client: &ValkeyClient,
    keys: &KeySchema,
    hash_hex: &str,
) -> Result<bool, fred::error::Error> {
    let n: i64 = client
        .pool
        .next()
        .exists(keys.ext_data(hash_hex))
        .await
        .inspect_err(|e| FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(e))?;
    Ok(n > 0)
}

/// The store answered, but not in the shape asked for.
fn protocol_error(details: String) -> fred::error::Error {
    fred::error::Error::new(fred::error::ErrorKind::Protocol, details)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The planner reads every structural header field: a field added to
    /// `HEADER_STRUCTURAL_FIELDS` but not read here would make every
    /// surviving header read as partial, and the planner restore over live
    /// layers.
    #[test]
    fn header_fact_fields_include_every_structural_field() {
        for field in crate::storage::layer::HEADER_STRUCTURAL_FIELDS {
            assert!(
                HEADER_FACT_FIELDS.contains(&field),
                "HEADER_FACT_FIELDS lacks the structural header field {field:?}"
            );
        }
    }

    #[test]
    fn decode_time_reply_is_milliseconds() {
        let reply = ["1760000000".to_owned(), "123456".to_owned()];
        assert_eq!(decode_time_reply(&reply).unwrap(), 1_760_000_000_123);
    }

    #[test]
    fn decode_time_reply_refuses_another_shape() {
        for reply in [
            vec![],
            vec!["1".to_owned()],
            vec!["x".to_owned(), "1".to_owned()],
            vec!["1".to_owned(), "2".to_owned(), "3".to_owned()],
        ] {
            let err = decode_time_reply(&reply).expect_err("not a TIME reply");
            assert_eq!(*err.kind(), fred::error::ErrorKind::Protocol, "{reply:?}");
        }
    }

    /// A header without `seq` is no surviving object; one with it reports
    /// its identity and whichever stamp columns parse.
    #[test]
    fn stamp_from_fields_needs_seq_and_parses_the_stamp() {
        assert_eq!(stamp_from_fields(7, &[None, None, None, None, None]), None);
        assert_eq!(
            stamp_from_fields(
                7,
                &[
                    None,
                    Some("s".into()),
                    Some("l".into()),
                    Some("4".into()),
                    None
                ]
            ),
            None,
            "a cursor-only phantom is no surviving object"
        );
        let stamp = stamp_from_fields(
            7,
            &[
                Some("9".into()),
                Some("s".into()),
                Some("l".into()),
                Some("4".into()),
                Some("x".into()),
            ],
        )
        .unwrap();
        assert_eq!(stamp.object_id, 7);
        assert_eq!(stamp.session_id, "s");
        assert_eq!(stamp.layer_name, "l");
        assert_eq!(stamp.last_checkpoint_seq, Some(4));
        assert_eq!(stamp.last_checkpoint_ms, None);
        assert!(stamp.is_stamped());
        let unstamped = stamp_from_fields(7, &[Some("9".into()), None, None, None, None]).unwrap();
        assert!(!unstamped.is_stamped());
    }

    /// The four pipeline replies of [`read_session_facts`], the layers index's
    /// `SMEMBERS` answering `members`.
    fn facts_with_members(members: Vec<Value>) -> SessionFacts {
        session_facts_from_replies(vec![
            Value::Integer(1),
            Value::Integer(0),
            Value::Null,
            Value::Array(members),
        ])
        .expect("the replies have the shape asked for")
    }

    /// A session's layers index decodes to the names its members encode,
    /// sorted. A member no name encodes to — upper-case hex, an odd length,
    /// bytes that are not UTF-8, hex of bytes that are not UTF-8, or the empty
    /// member — is left out, since no unmapping could remove it by a name.
    /// The other facts read as their replies say.
    #[test]
    fn session_facts_decode_the_layers_index_and_leave_out_members_no_name_encodes_to() {
        let facts = facts_with_members(vec![
            Value::from(hex_encode("b.usda")),
            Value::from(hex_encode("a.usda")),
            Value::from(hex_encode("a.usda").to_ascii_uppercase()),
            Value::from("6433a"),
            Value::Bytes(bytes::Bytes::from_static(&[0xFF, 0xFE])),
            Value::from("fffe"),
            Value::from(""),
        ]);
        assert_eq!(
            facts,
            SessionFacts {
                meta_exists: true,
                deleted_marker_exists: false,
                backup_kind: None,
                layers_index: vec!["a.usda".to_owned(), "b.usda".to_owned()],
            }
        );

        assert_eq!(
            facts_with_members(Vec::new()).layers_index,
            Vec::<String>::new(),
            "no layers index reads as no member"
        );
    }

    /// Replies of another shape than the four the pipeline sends are a
    /// protocol error, not facts.
    #[test]
    fn session_facts_refuse_replies_of_another_shape() {
        let err = session_facts_from_replies(vec![Value::Integer(1)])
            .expect_err("one reply is not the four asked for");
        assert_eq!(*err.kind(), fred::error::ErrorKind::Protocol);
    }
}
