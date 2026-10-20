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

//! The structural cache's repair read — `storage::structural_cache_sync` over
//! `scripts/structural_cache_sync.lua` (b.dkt Epic 4 Task 4; Spec SDD SR-4.1).
//!
//! Every object here is seeded through the real writers: `create_layer`, then
//! `update_layer` commits — field-only deltas on the Lua fast path and, in
//! the writes-nothing history and every `mixed` history of ten commits or
//! more, structural ones on the `lua` backend's structural route — each
//! carrying its delta's OMPF as relay bytes the way a production commit
//! does. So the
//! stream entries the script reads are the ones production writes. Expected
//! entries and byte sizes come from an independent `XRANGE` and from the relay
//! bytes each commit sent, never from the script.
//!
//! | Area | What it proves |
//! |---|---|
//! | Writes nothing | the keyspace (every key's `DUMP` and `PTTL`) is identical before and after a run of probe, chunk, byte-stopped, `TOO_LARGE`, target-capped, empty-range and absent-header calls |
//! | Probe mode | the header facts, this call's commit-token record and the five index cardinalities (each equal to a direct `SCARD` / `ZCARD`, in `KEYS[4..8]` order) come back; the `MONITOR` record is `HGETALL`, `TYPE`, `HGET`, a `TYPE` then a count for each index, and no stream read |
//! | Chunks | 300 pending entries come back as 256 then 44, contiguous, byte-identical to the stream, with `reached_end` only on the last |
//! | Byte bound | an exact fit is taken; one byte short stops before the crossing entry; a first pending entry alone over the bound answers `TOO_LARGE` with its id and size |
//! | Incremental reads | the `MONITOR` record shows one-entry `XRANGE` pages from an exclusive cursor: a byte-stopped chunk reads only its entries plus the crossing one, an entry-bound chunk reads no entry past the bound |
//! | Wider pages | at a page size of three: an entry bound of five returns five (`COUNT` 3 then 2), a byte stop inside a page returns the entries before the crossing one, and a range ending inside a page or on its last entry is reached |
//! | Target `seq` | caps the range; a target above the header is the header; an empty range is reached |
//! | Header facts | absent; `seq` absent, non-decimal, negative or above 2^53 − 1 → no `seq` and `NO_RANGE`; at 2^53 − 1 → reported; `superseded`; `chs_complete` unset; `commit_chain` absent → `""`, present verbatim |
//! | Commit-token record | this writer's record; another client's is absent; a wrong-typed key is reported as data, not a script error, in probe and chunk modes |
//! | Refusals | a wrong-typed header fails both modes `Internal` with `STRUCTURAL_SYNC_KEY_REFUSED: header` (not an absent header); a wrong-typed stream fails repair mode with `STRUCTURAL_SYNC_KEY_REFUSED: stream` (not an empty chunk) and leaves probe mode answering; a wrong-typed index — each of the five — is reported as data: probe mode answers no cardinalities (not a zero one) with its header facts and token record unchanged, and repair mode still answers; an entry or target `seq` above 2^53 − 1 fails `Internal` with `STRUCTURAL_SYNC_ARGS` |
//!
//! `reached_end` is asserted as the script reports it: the chunk covers the
//! range through its end, which is also true when the range simply ran out of
//! entries (the empty-range and 2^53 − 1 cases).
//!
//! The source scans (declared keys only, read-only, every read through
//! `pcall`) and the reply decoder's shape tests are inline in
//! `src/storage/structural_cache_sync.rs`. The `MONITOR` instrument is in
//! `snapshot_read_support.rs`; the object fixture builders are shared with
//! `snapshot_read_atomic_test.rs`.

use std::collections::HashMap;

use bytes::Bytes;
use fred::interfaces::{
    HashesInterface, KeysInterface, ListInterface, SetsInterface, SortedSetsInterface,
    StreamsInterface,
};

use live_edit::error::ServiceError;
use live_edit::storage::layer::{
    CHS_COMPLETE_MARKED, HEADER_FIELD_CHS_COMPLETE, HEADER_FIELD_COMMIT_CHAIN,
};
use live_edit::storage::structural_cache_sync::{
    read_structural_cache_sync, StructuralSyncReply, StructuralSyncRequest, SyncChunk,
    SyncCommitToken, SyncHeader, SyncHeaderFacts, SyncIndexCardinalities, SyncStreamEntry,
    STRUCTURAL_REPAIR_MAX_ENTRIES,
};
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::{delta_to_ompf_bytes, delta_with_set_field_inline};
use crate::common::scan_keys;
use crate::storage::snapshot_read_atomic_test::{
    create, create_cmd, create_session, every_shape_state, layer_store, structural, unmark, ROOT,
    WORLD,
};
use crate::storage::snapshot_read_support::{
    monitor_fence, test_valkey_addr, MonitorTap, MonitoredCommand,
};

/// The `client_id` every seeded commit is made under, so its commit-token
/// record is the one a probe under this id finds.
const WRITER: &str = "structural-sync-writer";
/// A `client_id` that never committed to the object.
const OTHER_CLIENT: &str = "structural-sync-other";
const CLIENT_TS: u64 = 7_000;

/// The inline value of a [`Step::FastLarge`] commit: its stream entry is
/// larger than [`SMALL_BYTE_BOUND`], and every other commit's is far smaller.
const LARGE_VALUE_BYTES: usize = 16 * 1024;
/// A byte bound between the ordinary entries and the large one.
const SMALL_BYTE_BOUND: u64 = 8 * 1024;

/// The highest `seq` either commit path writes (2^53 − 1), which the script
/// reports as a `seq`; one above it is reported as no `seq`.
const MAX_SEQ: u64 = (1 << 53) - 1;

/// Header columns the script reads by literal name, for which no public Rust
/// constant exists (`seq` is `pub(crate)`; `superseded` has none). The
/// `commit_chain` column is `HEADER_FIELD_COMMIT_CHAIN`.
const SEQ_COLUMN: &str = "seq";
const SUPERSEDED_COLUMN: &str = "superseded";

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// One seeded commit.
#[derive(Debug, Clone, Copy)]
enum Step {
    /// A small inline field write: the Lua fast path.
    Fast,
    /// A `CreateSection` under the root: the `lua` backend's structural
    /// route (the structural cache's plan, the update script's structural
    /// mode).
    Structural,
    /// A [`LARGE_VALUE_BYTES`] inline field write: the fast path, one large
    /// stream entry.
    FastLarge,
}

/// `n` commits, every tenth structural and the rest on the fast path.
fn mixed(n: usize) -> Vec<Step> {
    (0..n)
        .map(|i| {
            if i % 10 == 9 {
                Step::Structural
            } else {
                Step::Fast
            }
        })
        .collect()
}

/// An object created and then committed to, step by step.
struct Seeded {
    tv: TestValkey,
    oid: u64,
    /// The header `seq` `create_layer` left: the pending entries start after it.
    base_seq: u64,
    /// The relay OMPF each commit sent, in commit order; commit `i` landed at
    /// `base_seq + 1 + i`.
    relays: Vec<Bytes>,
}

impl Seeded {
    /// The header `seq` after every seeded commit.
    fn header_seq(&self) -> u64 {
        self.base_seq + self.relays.len() as u64
    }

    fn header_key(&self) -> String {
        self.tv.key_schema().obj_header(&self.oid.to_string())
    }

    fn wtok_key(&self) -> String {
        self.tv.key_schema().obj_write_token(&self.oid.to_string())
    }

    fn stream_key(&self) -> String {
        self.tv.key_schema().layer_stream(self.oid)
    }
}

async fn seed(layer: &str, steps: &[Step]) -> Seeded {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let base_seq = header_column(&tv, oid, SEQ_COLUMN)
        .await
        .and_then(|s| s.parse::<u64>().ok())
        .expect("fixture precondition: create_layer writes a decimal header seq");
    let mut relays = Vec::with_capacity(steps.len());
    for (i, step) in steps.iter().enumerate() {
        let n = i as u64;
        let delta = match step {
            Step::Fast => delta_with_set_field_inline(
                WORLD,
                &format!("k{}", i % 8),
                n.to_le_bytes().to_vec(),
                n + 1,
            ),
            Step::Structural => structural(vec![create_cmd(ROOT, 1_000 + n, &format!("s{i}"))]),
            Step::FastLarge => {
                delta_with_set_field_inline(WORLD, "large", vec![0xA5; LARGE_VALUE_BYTES], n + 1)
            }
        };
        let relay = Bytes::from(delta_to_ompf_bytes(&delta, &[]));
        let (seq, _) = ls
            .update_layer(
                &session_id,
                layer,
                oid,
                Some(delta),
                Some(relay.clone()),
                WRITER,
                CLIENT_TS,
            )
            .await
            .unwrap_or_else(|e| panic!("fixture commit {i} ({step:?}) must land: {e:?}"));
        assert_eq!(
            seq,
            base_seq + n + 1,
            "fixture precondition: commit {i} ({step:?}) lands at the next seq"
        );
        relays.push(relay);
    }
    Seeded {
        tv,
        oid,
        base_seq,
        relays,
    }
}

async fn header_column(tv: &TestValkey, oid: u64, column: &str) -> Option<String> {
    tv.client()
        .pool
        .hget(tv.key_schema().obj_header(&oid.to_string()), column)
        .await
        .expect("HGET obj:hdr must succeed")
}

async fn set_header_column(s: &Seeded, column: &str, value: &str) {
    let _: () =
        s.tv.client()
            .pool
            .hset(s.header_key(), [(column, value)])
            .await
            .expect("HSET obj:hdr must succeed");
}

async fn del_header_column(s: &Seeded, column: &str) {
    let _: u64 =
        s.tv.client()
            .pool
            .hdel(s.header_key(), column)
            .await
            .expect("HDEL obj:hdr must succeed");
}

fn entry_id(seq: u64) -> String {
    format!("{seq}-0")
}

// ═════════════════════════════════════════════════════════════════════════════
// The stream oracle
// ═════════════════════════════════════════════════════════════════════════════

/// One stream entry as a plain `XRANGE` returns it, and its size as the byte
/// bound counts it: the id plus every field name and value.
struct Stored {
    entry: SyncStreamEntry,
    size: u64,
}

/// The entries `{after+1}-0 ..= {through}-0`, read by a client `XRANGE`.
async fn stored_entries(s: &Seeded, after: u64, through: u64) -> Vec<Stored> {
    let raw: Vec<(String, HashMap<String, Bytes>)> =
        s.tv.client()
            .pool
            .xrange_values(s.stream_key(), entry_id(after + 1), entry_id(through), None)
            .await
            .expect("XRANGE must succeed");
    raw.into_iter()
        .map(|(id, fields)| {
            let size = id.len() as u64
                + fields
                    .iter()
                    .map(|(name, value)| (name.len() + value.len()) as u64)
                    .sum::<u64>();
            let payload = fields
                .get("payload")
                .and_then(|b| String::from_utf8(b.to_vec()).ok());
            let ompf = fields.get("ompf").cloned();
            Stored {
                entry: SyncStreamEntry { id, payload, ompf },
                size,
            }
        })
        .collect()
}

/// Every pending entry of `s`, checked against what the commits wrote: one
/// entry per commit at its `seq`, an `update` payload naming that `seq` and
/// [`WRITER`], and the commit's relay OMPF byte for byte.
async fn pending_entries(s: &Seeded) -> Vec<Stored> {
    let stored = stored_entries(s, s.base_seq, s.header_seq()).await;
    assert_eq!(
        stored.len(),
        s.relays.len(),
        "fixture precondition: one stream entry per seeded commit"
    );
    for (i, (st, relay)) in stored.iter().zip(&s.relays).enumerate() {
        let seq = s.base_seq + 1 + i as u64;
        assert_eq!(st.entry.id, entry_id(seq), "fixture precondition");
        let payload = st.entry.payload.as_deref().unwrap_or_default();
        assert!(
            payload.starts_with(&format!("update|{}|{seq}|{WRITER}|{CLIENT_TS}|", s.oid)),
            "fixture precondition: entry {seq} carries its commit's payload; got {payload:?}"
        );
        assert_eq!(
            st.entry.ompf.as_ref(),
            Some(relay),
            "fixture precondition: entry {seq} carries its commit's relay OMPF"
        );
    }
    stored
}

fn entries_of(stored: &[Stored]) -> Vec<SyncStreamEntry> {
    stored.iter().map(|st| st.entry.clone()).collect()
}

fn size_of(stored: &[Stored]) -> u64 {
    stored.iter().map(|st| st.size).sum()
}

// ═════════════════════════════════════════════════════════════════════════════
// Calls
// ═════════════════════════════════════════════════════════════════════════════

async fn sync(tv: &TestValkey, request: &StructuralSyncRequest<'_>) -> StructuralSyncReply {
    read_structural_cache_sync(tv.client(), tv.key_schema(), request)
        .await
        .unwrap_or_else(|e| panic!("the repair read {request:?} must succeed; got {e:?}"))
}

/// A repair call from `from` under the given entry and byte bounds (the
/// production page size).
fn bounded(
    oid: u64,
    from: u64,
    max_entries: u64,
    max_bytes: u64,
) -> StructuralSyncRequest<'static> {
    StructuralSyncRequest {
        max_entries,
        max_bytes,
        ..StructuralSyncRequest::repair(oid, WRITER, from)
    }
}

/// The fields of an `OK` chunk.
#[derive(Debug)]
struct Chunk {
    end_seq: u64,
    entries: Vec<SyncStreamEntry>,
    reached_end: bool,
    bytes: u64,
}

#[track_caller]
fn ok_chunk(reply: StructuralSyncReply) -> Chunk {
    match reply.chunk {
        SyncChunk::Entries {
            end_seq,
            entries,
            reached_end,
            bytes,
        } => Chunk {
            end_seq,
            entries,
            reached_end,
            bytes,
        },
        other => panic!("the call must answer an OK chunk; got {other:?}"),
    }
}

#[track_caller]
fn present(reply: &StructuralSyncReply) -> &SyncHeaderFacts {
    match &reply.header {
        SyncHeader::Present(facts) => facts,
        SyncHeader::Absent => panic!("the header must be reported present"),
    }
}

/// The `seq` a stream id names.
fn seq_of(entry: &SyncStreamEntry) -> u64 {
    entry
        .id
        .strip_suffix("-0")
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("stream id {} is not `{{seq}}-0`", entry.id))
}

// ═════════════════════════════════════════════════════════════════════════════
// MONITOR
// ═════════════════════════════════════════════════════════════════════════════

/// The commands `structural_cache_sync.lua` issued for the call whose
/// `EVALSHA` named `header_key` as `KEYS[1]`: the `lua` lines straight after
/// that `EVALSHA` line. A script runs atomically, so nothing interleaves.
fn script_commands(seen: Vec<MonitoredCommand>, header_key: &str) -> Vec<MonitoredCommand> {
    let start = seen
        .iter()
        .rposition(|c| {
            c.name() == "EVALSHA" && c.args.get(3).map(String::as_str) == Some(header_key)
        })
        .unwrap_or_else(|| panic!("the MONITOR record holds no EVALSHA naming {header_key}"));
    seen.into_iter()
        .skip(start + 1)
        .take_while(MonitoredCommand::issued_by_script)
        .collect()
}

/// Run `request` under a `MONITOR` tap and return the reply and the commands
/// the script issued.
async fn sync_monitored(
    s: &Seeded,
    request: &StructuralSyncRequest<'_>,
) -> (StructuralSyncReply, Vec<MonitoredCommand>) {
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let reply = sync(&s.tv, request).await;
    let fence = monitor_fence(s.tv.prefix());
    let _: i64 = s.tv.client().pool.exists(&fence).await.unwrap();
    let script = script_commands(tap.drain_until(&fence).await, &s.header_key());
    (reply, script)
}

/// The arguments of every `XRANGE` the script issued.
fn xranges(script: &[MonitoredCommand]) -> Vec<Vec<String>> {
    script
        .iter()
        .filter(|c| c.name() == "XRANGE")
        .map(|c| c.args.clone())
        .collect()
}

/// The `XRANGE` page the script issues to read up to `count` entries from
/// entry `seq` of a range ending at `end`: the first page starts inclusively
/// at `{seq}-0`, every later one exclusively after the previous entry,
/// `({seq-1}-0`.
fn page_of(s: &Seeded, seq: u64, first: bool, end: u64, count: u64) -> Vec<String> {
    let start = if first {
        entry_id(seq)
    } else {
        format!("({}", entry_id(seq - 1))
    };
    vec![
        "XRANGE".to_owned(),
        s.stream_key(),
        start,
        entry_id(end),
        "COUNT".to_owned(),
        count.to_string(),
    ]
}

/// The one-entry page (the production page size) that reads entry `seq`.
fn page(s: &Seeded, seq: u64, first: bool, end: u64) -> Vec<String> {
    page_of(s, seq, first, end, 1)
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Writes nothing
// ═════════════════════════════════════════════════════════════════════════════

/// Every key under the test's prefix with its `DUMP` (the serialized value, so
/// any change to a hash column, a stream entry or a stream's metadata shows)
/// and its absolute expiry (`PEXPIRETIME`: -1 for none), sorted by key. The
/// absolute expiry, not the `PTTL`: a key that carries a TTL from the fixture
/// (a `CreateLayer` commit claim) counts down between two reads, while any
/// write of an expiry moves the absolute time.
async fn keyspace(tv: &TestValkey) -> Vec<(String, Option<Vec<u8>>, i64)> {
    let mut keys: Vec<String> = scan_keys(tv.client(), &format!("{}*", tv.prefix()))
        .await
        .into_iter()
        .map(|k| k.as_str_lossy().into_owned())
        .collect();
    keys.sort();
    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let dumped: Option<Vec<u8>> = tv.client().pool.dump(&key).await.expect("DUMP");
        let expires_at: i64 = tv
            .client()
            .pool
            .pexpire_time(&key)
            .await
            .expect("PEXPIRETIME");
        out.push((key, dumped, expires_at));
    }
    out
}

/// The script writes nothing: across every reply shape it can give, in probe
/// and chunk modes, the keyspace — keys, values, TTLs — is identical before and
/// after, and no key appears (an absent header's keys included).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_leaves_the_keyspace_identical_in_probe_and_chunk_modes() {
    let s = seed(
        "scs-writes-nothing",
        &[
            Step::Fast,
            Step::Structural,
            Step::FastLarge,
            Step::Fast,
            Step::Structural,
        ],
    )
    .await;
    let (oid, base, header) = (s.oid, s.base_seq, s.header_seq());
    const ABSENT_OID: u64 = 987_654_321;
    let before = keyspace(&s.tv).await;
    assert!(
        before.iter().any(|(k, _, _)| *k == s.wtok_key())
            && before.iter().any(|(k, _, _)| *k == s.stream_key()),
        "fixture precondition: the commit-token record and the stream exist, or 'reads them \
         without writing' is vacuous"
    );

    let requests = [
        StructuralSyncRequest::probe(oid, WRITER),
        StructuralSyncRequest::probe(oid, OTHER_CLIENT),
        StructuralSyncRequest::repair(oid, WRITER, base),
        bounded(oid, base, 2, u64::MAX / 2),
        bounded(oid, base, 256, SMALL_BYTE_BOUND),
        bounded(oid, base + 2, 256, SMALL_BYTE_BOUND),
        StructuralSyncRequest::repair_to(oid, WRITER, base, base + 3),
        StructuralSyncRequest::repair(oid, WRITER, header),
        StructuralSyncRequest::probe(ABSENT_OID, WRITER),
        StructuralSyncRequest::repair(ABSENT_OID, WRITER, 0),
    ];
    let mut shapes = std::collections::BTreeSet::new();
    for request in &requests {
        let reply = sync(&s.tv, request).await;
        shapes.insert(match reply.chunk {
            SyncChunk::Probe(_) => "probe",
            SyncChunk::NoRange => "no_range",
            SyncChunk::Entries {
                reached_end: true, ..
            } => "ok_reached",
            SyncChunk::Entries {
                reached_end: false, ..
            } => "ok_stopped",
            SyncChunk::TooLarge { .. } => "too_large",
        });
    }
    assert_eq!(
        shapes.into_iter().collect::<Vec<_>>(),
        ["no_range", "ok_reached", "ok_stopped", "probe", "too_large"],
        "the calls must have exercised every reply shape, or 'writes nothing' covers less than it \
         claims"
    );

    let after = keyspace(&s.tv).await;
    assert_eq!(
        after.iter().map(|(k, _, _)| k.as_str()).collect::<Vec<_>>(),
        before
            .iter()
            .map(|(k, _, _)| k.as_str())
            .collect::<Vec<_>>(),
        "the repair read must create and delete no key"
    );
    let changed: Vec<&str> = before
        .iter()
        .zip(&after)
        .filter(|(b, a)| b != a)
        .map(|((key, _, _), _)| key.as_str())
        .collect();
    assert!(
        changed.is_empty(),
        "the repair read must leave every key's value and TTL byte-identical (SR-4.1: it reads, \
         it never writes); changed: {changed:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. Probe mode
// ═════════════════════════════════════════════════════════════════════════════

/// Probe mode reports the header facts, the call's commit-token record and the
/// object's five index cardinalities, and reads nothing else: the `MONITOR`
/// record is exactly the header read, the two token reads and, per index, its
/// `TYPE` then its count, and no `XRANGE`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_probe_reports_header_and_token_without_reading_the_stream() {
    let s = seed("scs-probe", &mixed(3)).await;
    let header = s.header_seq();
    let record: Option<String> =
        s.tv.client()
            .pool
            .hget(s.wtok_key(), WRITER)
            .await
            .expect("HGET obj:wtok must succeed");
    let record = record.expect("fixture precondition: the last commit wrote WRITER's record");
    assert!(
        record.contains(&format!(":{header}:")),
        "fixture precondition: WRITER's record names the last commit's seq; got {record:?}"
    );

    let keys = s.tv.key_schema();
    let oid = s.oid.to_string();
    let index_keys = [
        keys.obj_sections_index(&oid),
        keys.obj_deleted_sections_index(&oid),
        keys.obj_fields_index(&oid),
        keys.obj_time_samples_index(&oid),
        keys.obj_children_index(&oid),
    ];
    let pool = &s.tv.client().pool;
    // The fixture deletes no section; one tombstone-index member, written
    // out of band, gives that index a count to report.
    let _: i64 = pool
        .sadd(&index_keys[1], "424242")
        .await
        .expect("SADD obj:dels must succeed");
    let scard = |key: &String| {
        let key = key.clone();
        async move { pool.scard::<u64, _>(key).await.expect("SCARD must succeed") }
    };
    let zcard = |key: &String| {
        let key = key.clone();
        async move { pool.zcard::<u64, _>(key).await.expect("ZCARD must succeed") }
    };
    let stored = SyncIndexCardinalities {
        sections: scard(&index_keys[0]).await,
        tombstones: scard(&index_keys[1]).await,
        fields: zcard(&index_keys[2]).await,
        time_samples: zcard(&index_keys[3]).await,
        children_index: scard(&index_keys[4]).await,
    };
    assert!(
        stored.sections > 0
            && stored.tombstones > 0
            && stored.fields > 0
            && stored.time_samples > 0
            && stored.children_index > 0,
        "fixture precondition: every index is non-empty, or 'reports them as stored' does not \
         tell a read from a zero; got {stored:?}"
    );

    let (reply, script) = sync_monitored(&s, &StructuralSyncRequest::probe(s.oid, WRITER)).await;

    assert_eq!(
        reply.chunk,
        SyncChunk::Probe(Some(stored)),
        "probe mode reads no chunk, and reports the five index cardinalities as stored"
    );
    assert_eq!(
        present(&reply),
        &SyncHeaderFacts {
            seq: Some(header),
            commit_chain: header_column(&s.tv, s.oid, HEADER_FIELD_COMMIT_CHAIN)
                .await
                .unwrap_or_default(),
            superseded: None,
            chs_complete: Some(CHS_COMPLETE_MARKED.to_owned()),
        },
        "probe mode reports the header facts as stored"
    );
    assert_eq!(
        reply.commit_token,
        SyncCommitToken {
            key_type: "hash".to_owned(),
            record: Some(record),
        },
        "probe mode reports this call's commit-token record verbatim"
    );
    let calls: Vec<Vec<String>> = script.iter().map(|c| c.args.clone()).collect();
    assert_eq!(
        calls,
        [
            vec!["HGETALL".to_owned(), s.header_key()],
            vec!["TYPE".to_owned(), s.wtok_key()],
            vec!["HGET".to_owned(), s.wtok_key(), WRITER.to_owned()],
            vec!["TYPE".to_owned(), index_keys[0].clone()],
            vec!["SCARD".to_owned(), index_keys[0].clone()],
            vec!["TYPE".to_owned(), index_keys[1].clone()],
            vec!["SCARD".to_owned(), index_keys[1].clone()],
            vec!["TYPE".to_owned(), index_keys[2].clone()],
            vec!["ZCARD".to_owned(), index_keys[2].clone()],
            vec!["TYPE".to_owned(), index_keys[3].clone()],
            vec!["ZCARD".to_owned(), index_keys[3].clone()],
            vec!["TYPE".to_owned(), index_keys[4].clone()],
            vec!["SCARD".to_owned(), index_keys[4].clone()],
        ],
        "probe mode reads the header, the token record and, type first, the five index \
         cardinalities, and never the stream"
    );

    let other = sync(&s.tv, &StructuralSyncRequest::probe(s.oid, OTHER_CLIENT)).await;
    assert_eq!(
        other.commit_token,
        SyncCommitToken {
            key_type: "hash".to_owned(),
            record: None,
        },
        "a client that never committed has no record in the hash"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. Chunks
// ═════════════════════════════════════════════════════════════════════════════

/// 300 pending entries come back as 256 (the entry bound) then 44: contiguous,
/// byte-identical to the stream, `reached_end` only on the last chunk, each
/// chunk's byte tally the sum of its entries.
///
/// Seeds 300 commits one at a time, 30 of them structural: the slowest test
/// here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_chunks_300_pending_entries_as_256_then_44() {
    const PENDING: usize = 300;
    let s = seed("scs-300", &mixed(PENDING)).await;
    let header = s.header_seq();
    let stored = pending_entries(&s).await;
    let bound = STRUCTURAL_REPAIR_MAX_ENTRIES as usize;
    assert_eq!(
        bound, 256,
        "the acceptance numbers assume the 256-entry bound"
    );

    let first = ok_chunk(
        sync(
            &s.tv,
            &StructuralSyncRequest::repair(s.oid, WRITER, s.base_seq),
        )
        .await,
    );
    assert_eq!(first.end_seq, header, "the range ends at the header seq");
    assert_eq!(
        first.entries,
        entries_of(&stored[..bound]),
        "the first chunk is the first 256 pending entries, verbatim"
    );
    assert!(
        !first.reached_end,
        "the entry bound stopped the first chunk short of the header seq"
    );
    assert_eq!(first.bytes, size_of(&stored[..bound]));

    let resume = seq_of(first.entries.last().expect("the first chunk is not empty"));
    let second = ok_chunk(sync(&s.tv, &StructuralSyncRequest::repair(s.oid, WRITER, resume)).await);
    assert_eq!(second.end_seq, header);
    assert_eq!(
        second.entries,
        entries_of(&stored[bound..]),
        "continuing from the first chunk's last entry returns the remaining 44, with no gap and \
         no repeat"
    );
    assert!(second.reached_end, "the last chunk reaches the header seq");
    assert_eq!(second.bytes, size_of(&stored[bound..]));
}

/// The byte bound ends a chunk before the entry that would take the tally past
/// it: an exact fit is taken, one byte less stops a whole entry earlier.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_byte_bound_stops_before_the_entry_that_would_cross_it() {
    let s = seed("scs-byte-bound", &mixed(12)).await;
    let stored = pending_entries(&s).await;
    let three = size_of(&stored[..3]);

    let exact = ok_chunk(sync(&s.tv, &bounded(s.oid, s.base_seq, 256, three)).await);
    assert_eq!(
        (exact.entries, exact.reached_end, exact.bytes),
        (entries_of(&stored[..3]), false, three),
        "a bound the first three entries fill exactly takes all three and stops before the fourth"
    );

    let short = ok_chunk(sync(&s.tv, &bounded(s.oid, s.base_seq, 256, three - 1)).await);
    assert_eq!(
        (short.entries, short.reached_end, short.bytes),
        (entries_of(&stored[..2]), false, size_of(&stored[..2])),
        "one byte less and the third entry would cross the bound, so the chunk stops before it"
    );

    let all = ok_chunk(sync(&s.tv, &bounded(s.oid, s.base_seq, 256, size_of(&stored))).await);
    assert_eq!(
        (all.entries, all.reached_end),
        (entries_of(&stored), true),
        "a bound the whole range fits returns it all and reaches the header seq"
    );
}

/// A first pending entry alone larger than the byte bound answers `TOO_LARGE`
/// with that entry's id and size and no entry; the chunk before it stops in
/// front of it; a bound it fits exactly returns it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_first_pending_entry_over_the_byte_bound_answers_too_large() {
    let s = seed(
        "scs-too-large",
        &[Step::Fast, Step::Fast, Step::FastLarge, Step::Fast],
    )
    .await;
    let (base, header) = (s.base_seq, s.header_seq());
    let stored = pending_entries(&s).await;
    assert!(
        size_of(&stored[..2]) <= SMALL_BYTE_BOUND && stored[2].size > SMALL_BYTE_BOUND,
        "fixture precondition: the two small entries fit the bound and the large one alone does not"
    );

    let before = ok_chunk(sync(&s.tv, &bounded(s.oid, base, 256, SMALL_BYTE_BOUND)).await);
    assert_eq!(
        (before.entries, before.reached_end),
        (entries_of(&stored[..2]), false),
        "the chunk stops in front of the large entry"
    );

    let too_large = sync(&s.tv, &bounded(s.oid, base + 2, 256, SMALL_BYTE_BOUND)).await;
    assert_eq!(
        too_large.chunk,
        SyncChunk::TooLarge {
            end_seq: header,
            entry_id: entry_id(base + 3),
            entry_bytes: stored[2].size,
        },
        "a first pending entry larger than the byte bound answers TOO_LARGE with its id and size"
    );

    let first_small = sync(&s.tv, &bounded(s.oid, base, 256, stored[0].size - 1)).await;
    assert_eq!(
        first_small.chunk,
        SyncChunk::TooLarge {
            end_seq: header,
            entry_id: entry_id(base + 1),
            entry_bytes: stored[0].size,
        },
        "the rule is relative to the bound: an ordinary entry one byte over it is TOO_LARGE too"
    );

    let fits = ok_chunk(sync(&s.tv, &bounded(s.oid, base + 2, 256, stored[2].size)).await);
    assert_eq!(
        (fits.entries, fits.reached_end, fits.bytes),
        (entries_of(&stored[2..3]), false, stored[2].size),
        "a bound the large entry fits exactly returns it alone"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. Incremental reads
// ═════════════════════════════════════════════════════════════════════════════

/// The script reads a chunk one entry per `XRANGE`, each page after the first
/// starting exclusively after the last entry read, and stops reading where the
/// chunk stops: a byte-stopped chunk reads its entries plus the one crossing
/// entry (so at most the bound plus one entry is copied into Lua), and an
/// entry-bound chunk reads no entry past the bound. Never one wide `XRANGE`
/// truncated afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_reads_the_stream_one_entry_per_xrange_and_stops_with_the_chunk() {
    let s = seed("scs-incremental", &mixed(12)).await;
    let (base, header) = (s.base_seq, s.header_seq());
    let stored = pending_entries(&s).await;
    let three = size_of(&stored[..3]);

    let (reply, script) = sync_monitored(&s, &bounded(s.oid, base, 256, three)).await;
    assert_eq!(ok_chunk(reply).entries, entries_of(&stored[..3]));
    let expected: Vec<Vec<String>> = (1..=4)
        .map(|k| page(&s, base + k, k == 1, header))
        .collect();
    assert_eq!(
        xranges(&script),
        expected,
        "a chunk the byte bound stops after three entries reads entries 1..=4 one page each — the \
         fourth, the crossing entry, is the only one read beyond the chunk, so the script copied \
         the bound plus one entry; script commands: {script:?}"
    );

    let (reply, script) = sync_monitored(&s, &bounded(s.oid, base, 5, u64::MAX / 2)).await;
    assert_eq!(ok_chunk(reply).entries, entries_of(&stored[..5]));
    let expected: Vec<Vec<String>> = (1..=5)
        .map(|k| page(&s, base + k, k == 1, header))
        .collect();
    assert_eq!(
        xranges(&script),
        expected,
        "a chunk the entry bound stops at five reads exactly five one-entry pages and never the \
         sixth entry; script commands: {script:?}"
    );

    for c in &script {
        assert!(
            ["HGETALL", "TYPE", "HGET", "XRANGE"].contains(&c.name().as_str()),
            "the script issues only reads; it issued {c:?}"
        );
    }
}

/// The page size the bounds are applied across. Three does not divide the
/// entry bound of five, so the bound falls part-way through the second page.
const WIDE_PAGE: u64 = 3;

/// With a page size above one the bounds and the range end still hold exactly,
/// wherever they fall inside a page: the last page asks for only the entries
/// the entry bound has left (`COUNT` 3 then 2, five entries for a bound of
/// five); a byte stop part-way through a page returns the entries before the
/// crossing one; and a range ending part-way through a page, or on a page's
/// last entry, is reached.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_with_a_wider_page_keeps_the_bounds_and_the_range_end_exact() {
    let s = seed("scs-wide-page", &mixed(12)).await;
    let (base, header) = (s.base_seq, s.header_seq());
    let stored = pending_entries(&s).await;
    let wide = |request: StructuralSyncRequest<'static>| StructuralSyncRequest {
        page_entries: WIDE_PAGE,
        ..request
    };

    let (reply, script) = sync_monitored(&s, &wide(bounded(s.oid, base, 5, u64::MAX / 2))).await;
    let chunk = ok_chunk(reply);
    assert_eq!(
        (chunk.entries, chunk.reached_end),
        (entries_of(&stored[..5]), false),
        "an entry bound of five over pages of three returns exactly five entries"
    );
    assert_eq!(
        xranges(&script),
        [
            page_of(&s, base + 1, true, header, WIDE_PAGE),
            page_of(&s, base + 4, false, header, 2),
        ],
        "the second page asks only for the two entries the bound has left; script commands: \
         {script:?}"
    );

    let four = size_of(&stored[..4]);
    let (reply, script) = sync_monitored(&s, &wide(bounded(s.oid, base, 256, four))).await;
    let chunk = ok_chunk(reply);
    assert_eq!(
        (chunk.entries, chunk.reached_end, chunk.bytes),
        (entries_of(&stored[..4]), false, four),
        "a byte bound four entries fill stops after the first entry of the second page"
    );
    assert_eq!(
        xranges(&script),
        [
            page_of(&s, base + 1, true, header, WIDE_PAGE),
            page_of(&s, base + 4, false, header, WIDE_PAGE),
        ],
        "the byte stop falls inside the second page, which is the last one read; script \
         commands: {script:?}"
    );

    for (target, what) in [
        (base + 5, "part-way through the second page"),
        (base + 6, "on the second page's last entry"),
    ] {
        let n = (target - base) as usize;
        let chunk = ok_chunk(
            sync(
                &s.tv,
                &wide(StructuralSyncRequest::repair_to(
                    s.oid, WRITER, base, target,
                )),
            )
            .await,
        );
        assert_eq!(
            (chunk.end_seq, chunk.entries, chunk.reached_end),
            (target, entries_of(&stored[..n]), true),
            "a range ending {what} returns it whole and is reached"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. Target seq
// ═════════════════════════════════════════════════════════════════════════════

/// A target `seq` caps the range below the header `seq`; a target above the
/// header is the header; a range with nothing after the entry's `seq` is empty
/// and reached.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_target_seq_caps_the_range() {
    let s = seed("scs-target", &mixed(20)).await;
    let (base, header) = (s.base_seq, s.header_seq());
    let stored = pending_entries(&s).await;

    let capped = ok_chunk(
        sync(
            &s.tv,
            &StructuralSyncRequest::repair_to(s.oid, WRITER, base, base + 5),
        )
        .await,
    );
    assert_eq!(
        (capped.end_seq, capped.entries, capped.reached_end),
        (base + 5, entries_of(&stored[..5]), true),
        "a target below the header ends the range at the target"
    );

    let above = ok_chunk(
        sync(
            &s.tv,
            &StructuralSyncRequest::repair_to(s.oid, WRITER, base, header + 30),
        )
        .await,
    );
    assert_eq!(
        (above.end_seq, above.entries, above.reached_end),
        (header, entries_of(&stored), true),
        "a target above the header is capped at the header seq"
    );

    for (from, target, what) in [
        (base + 5, Some(base + 5), "an entry already at the target"),
        (header, None, "an entry already at the header seq"),
        (header + 3, None, "an entry ahead of the header seq"),
    ] {
        let request = match target {
            Some(t) => StructuralSyncRequest::repair_to(s.oid, WRITER, from, t),
            None => StructuralSyncRequest::repair(s.oid, WRITER, from),
        };
        let empty = ok_chunk(sync(&s.tv, &request).await);
        assert_eq!(
            (empty.entries, empty.reached_end, empty.bytes),
            (Vec::new(), true, 0),
            "{what}: the range is empty and reached"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. Header facts
// ═════════════════════════════════════════════════════════════════════════════

/// An object with no header is reported absent, with no token record, and a
/// repair call has no range to read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_of_an_absent_header_reports_absent_and_no_range() {
    let tv = TestValkey::setup().await;
    const NEVER_CREATED: u64 = 424_242;

    let probe = sync(&tv, &StructuralSyncRequest::probe(NEVER_CREATED, WRITER)).await;
    assert_eq!(
        (probe.header, probe.commit_token, probe.chunk),
        (
            SyncHeader::Absent,
            SyncCommitToken {
                key_type: "none".to_owned(),
                record: None,
            },
            SyncChunk::Probe(Some(SyncIndexCardinalities {
                sections: 0,
                tombstones: 0,
                fields: 0,
                time_samples: 0,
                children_index: 0,
            })),
        ),
        "probe of an absent object: absent indexes count 0"
    );

    let repair = sync(
        &tv,
        &StructuralSyncRequest::repair(NEVER_CREATED, WRITER, 0),
    )
    .await;
    assert_eq!(
        (repair.header, repair.chunk),
        (SyncHeader::Absent, SyncChunk::NoRange),
        "a repair of an absent object has no range to read"
    );
}

/// The header's `seq` is reported only when it is a decimal of at most
/// 2^53 − 1. Absent, non-decimal, negative and over-ceiling values are
/// reported as a present header without a `seq`, and a repair call then has no
/// range; exactly 2^53 − 1 is reported and has a (here empty) range.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_reports_seq_only_when_it_is_a_decimal_within_the_ceiling() {
    let s = seed("scs-seq", &mixed(2)).await;
    let base = s.base_seq;

    let over_ceiling = (MAX_SEQ + 1).to_string();
    let not_a_seq: [Option<&str>; 6] = [
        None,
        Some("12x"),
        Some("-1"),
        Some(""),
        Some("1.5"),
        Some(over_ceiling.as_str()),
    ];
    for raw in not_a_seq {
        match raw {
            Some(v) => set_header_column(&s, SEQ_COLUMN, v).await,
            None => del_header_column(&s, SEQ_COLUMN).await,
        }
        let probe = sync(&s.tv, &StructuralSyncRequest::probe(s.oid, WRITER)).await;
        assert_eq!(
            (
                present(&probe).seq,
                matches!(probe.chunk, SyncChunk::Probe(_))
            ),
            (None, true),
            "header seq {raw:?}: the header is present without a seq"
        );
        let repair = sync(&s.tv, &StructuralSyncRequest::repair(s.oid, WRITER, base)).await;
        assert_eq!(
            (present(&repair).seq, repair.chunk),
            (None, SyncChunk::NoRange),
            "header seq {raw:?}: a repair has no range to read"
        );
    }

    set_header_column(&s, SEQ_COLUMN, &MAX_SEQ.to_string()).await;
    let at_ceiling = sync(
        &s.tv,
        &StructuralSyncRequest::repair(s.oid, WRITER, MAX_SEQ - 1),
    )
    .await;
    assert_eq!(present(&at_ceiling).seq, Some(MAX_SEQ), "2^53 - 1 is a seq");
    let chunk = ok_chunk(at_ceiling);
    assert_eq!(
        (chunk.end_seq, chunk.entries, chunk.reached_end),
        (MAX_SEQ, Vec::new(), true),
        "the range ends at 2^53 - 1 and holds no entry"
    );
}

/// `superseded`, `chs_complete` and `commit_chain` are reported as stored:
/// absent `superseded` / `chs_complete` as `None`, an absent `commit_chain` as
/// `""`, and each present value verbatim.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_reports_superseded_chs_complete_and_commit_chain_as_stored() {
    let s = seed("scs-columns", &mixed(1)).await;
    let header = s.header_seq();
    // Every commit writes a folded `commit_chain`; start from the column's
    // absence.
    del_header_column(&s, HEADER_FIELD_COMMIT_CHAIN).await;

    let initial = sync(&s.tv, &StructuralSyncRequest::probe(s.oid, WRITER)).await;
    let facts = present(&initial);
    assert_eq!(
        facts,
        &SyncHeaderFacts {
            seq: Some(header),
            commit_chain: String::new(),
            superseded: None,
            chs_complete: Some(CHS_COMPLETE_MARKED.to_owned()),
        },
        "a created object: marked, not superseded, and an absent commit_chain reads as \"\""
    );
    assert!(facts.chs_complete_marked());

    const CHAIN: &str = "0123456789abcdef0123456789abcdef01234567";
    set_header_column(&s, HEADER_FIELD_COMMIT_CHAIN, CHAIN).await;
    set_header_column(&s, SUPERSEDED_COLUMN, "1").await;
    unmark(&s.tv, s.oid).await;
    assert_eq!(
        header_column(&s.tv, s.oid, HEADER_FIELD_CHS_COMPLETE).await,
        None,
        "fixture precondition: the marker is gone"
    );

    let changed = sync(&s.tv, &StructuralSyncRequest::probe(s.oid, WRITER)).await;
    let facts = present(&changed);
    assert_eq!(
        facts,
        &SyncHeaderFacts {
            seq: Some(header),
            commit_chain: CHAIN.to_owned(),
            superseded: Some("1".to_owned()),
            chs_complete: None,
        },
        "a present commit_chain and superseded are reported verbatim; an unset chs_complete as None"
    );
    assert!(!facts.chs_complete_marked());
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. Commit-token record
// ═════════════════════════════════════════════════════════════════════════════

/// A commit-token key of a foreign type is reported as data — its type, no
/// record — and the call still succeeds, in probe and chunk modes, with the
/// stream chunk unaffected.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_reports_a_wrong_typed_commit_token_record_as_data() {
    let s = seed("scs-wrong-type", &mixed(3)).await;
    let stored = pending_entries(&s).await;
    let wtok = s.wtok_key();

    for foreign in ["string", "list"] {
        let _: i64 = s.tv.client().pool.del(&wtok).await.expect("DEL obj:wtok");
        if foreign == "string" {
            let _: () =
                s.tv.client()
                    .pool
                    .set(&wtok, "planted-by-an-out-of-band-writer", None, None, false)
                    .await
                    .expect("SET obj:wtok");
        } else {
            let _: i64 =
                s.tv.client()
                    .pool
                    .rpush(&wtok, "planted-by-an-out-of-band-writer")
                    .await
                    .expect("RPUSH obj:wtok");
        }
        let token = SyncCommitToken {
            key_type: foreign.to_owned(),
            record: None,
        };

        let probe = sync(&s.tv, &StructuralSyncRequest::probe(s.oid, WRITER)).await;
        assert_eq!(
            (
                &probe.commit_token,
                matches!(probe.chunk, SyncChunk::Probe(_))
            ),
            (&token, true),
            "probe mode over a {foreign} commit-token key"
        );
        assert!(probe.commit_token.is_wrong_type());

        let repair = sync(
            &s.tv,
            &StructuralSyncRequest::repair(s.oid, WRITER, s.base_seq),
        )
        .await;
        assert_eq!(
            &repair.commit_token, &token,
            "chunk mode over a {foreign} commit-token key"
        );
        let chunk = ok_chunk(repair);
        assert_eq!(
            (chunk.entries, chunk.reached_end),
            (entries_of(&stored), true),
            "a {foreign} commit-token key does not affect the chunk"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 8. Refusals
// ═════════════════════════════════════════════════════════════════════════════

/// The call fails `Internal`, and the message names the refusal.
async fn sync_internal(tv: &TestValkey, request: &StructuralSyncRequest<'_>) -> String {
    match read_structural_cache_sync(tv.client(), tv.key_schema(), request).await {
        Err(ServiceError::Internal(msg)) => msg,
        other => panic!("the repair read {request:?} must fail Internal; got {other:?}"),
    }
}

/// Replace `key` with a string, as an out-of-band writer would.
async fn plant_string(tv: &TestValkey, key: &str) {
    let _: () = tv
        .client()
        .pool
        .set(key, "planted-by-an-out-of-band-writer", None, None, false)
        .await
        .unwrap_or_else(|e| panic!("planting a string at {key} must succeed: {e}"));
}

/// A wrong-typed header fails the call in both modes with the script's tagged
/// refusal naming the header, rather than reading as an absent header (which
/// the caller would map to `NotFound`).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_of_a_wrong_typed_header_is_internal_in_probe_and_repair_modes() {
    let s = seed("scs-wrongtype-header", &mixed(2)).await;
    plant_string(&s.tv, &s.header_key()).await;

    for request in [
        StructuralSyncRequest::probe(s.oid, WRITER),
        StructuralSyncRequest::repair(s.oid, WRITER, s.base_seq),
    ] {
        let msg = sync_internal(&s.tv, &request).await;
        assert!(
            msg.contains("STRUCTURAL_SYNC_KEY_REFUSED: header") && msg.contains("WRONGTYPE"),
            "{:?}: a wrong-typed header is the script's header refusal, carrying the WRONGTYPE; \
             got {msg:?}",
            request.mode
        );
    }
}

/// A wrong-typed stream fails a repair call with the script's tagged refusal
/// naming the stream, rather than reading as an empty, reached chunk; a probe,
/// which never reads the stream, still answers.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_of_a_wrong_typed_stream_is_internal_in_repair_mode_only() {
    let s = seed("scs-wrongtype-stream", &mixed(2)).await;
    let header = s.header_seq();
    plant_string(&s.tv, &s.stream_key()).await;

    let msg = sync_internal(
        &s.tv,
        &StructuralSyncRequest::repair(s.oid, WRITER, s.base_seq),
    )
    .await;
    assert!(
        msg.contains("STRUCTURAL_SYNC_KEY_REFUSED: stream") && msg.contains("WRONGTYPE"),
        "a wrong-typed stream is the script's stream refusal, carrying the WRONGTYPE; got {msg:?}"
    );

    let probe = sync(&s.tv, &StructuralSyncRequest::probe(s.oid, WRITER)).await;
    assert_eq!(
        (
            present(&probe).seq,
            matches!(probe.chunk, SyncChunk::Probe(_))
        ),
        (Some(header), true),
        "probe mode does not read the stream, so a wrong-typed one does not fail it"
    );
}

/// A wrong-typed index key is reported as data, like a wrong-typed
/// commit-token record: the probe answers, with no cardinalities
/// (`Probe(None)`, not a zero one) and its header facts and token record
/// unchanged — for each of the five indexes. A repair call, which neither
/// declares nor reads the indexes, still answers its chunk.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_of_a_wrong_typed_index_reports_no_cardinalities_in_probe_mode() {
    for index in 0..5 {
        let s = seed("scs-wrongtype-index", &mixed(2)).await;
        let keys = s.tv.key_schema();
        let oid = s.oid.to_string();
        let key = [
            keys.obj_sections_index(&oid),
            keys.obj_deleted_sections_index(&oid),
            keys.obj_fields_index(&oid),
            keys.obj_time_samples_index(&oid),
            keys.obj_children_index(&oid),
        ][index]
            .clone();
        let before = sync(&s.tv, &StructuralSyncRequest::probe(s.oid, WRITER)).await;
        assert!(
            matches!(before.chunk, SyncChunk::Probe(Some(_))),
            "{key}: fixture precondition — the untouched object reports cardinalities"
        );
        plant_string(&s.tv, &key).await;

        let after = sync(&s.tv, &StructuralSyncRequest::probe(s.oid, WRITER)).await;

        assert_eq!(
            (&after.header, &after.commit_token, &after.chunk),
            (
                &before.header,
                &before.commit_token,
                &SyncChunk::Probe(None)
            ),
            "{key}: no cardinalities, and the header facts and token record as before"
        );
        let repair = sync(
            &s.tv,
            &StructuralSyncRequest::repair(s.oid, WRITER, s.base_seq),
        )
        .await;
        assert!(
            matches!(repair.chunk, SyncChunk::Entries { .. }),
            "{key}: a repair call does not read the indexes, so a wrong-typed one does not \
             fail it; got {:?}",
            repair.chunk
        );
    }
}

/// An entry `seq` or a target `seq` above 2^53 − 1 is a caller bug the script
/// refuses with `STRUCTURAL_SYNC_ARGS` before reading anything, so the call
/// fails `Internal` and never answers with a reply built on a rounded `seq`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_cache_sync_with_a_seq_argument_above_the_ceiling_is_internal() {
    let s = seed("scs-args", &mixed(2)).await;
    let base = s.base_seq;

    for (request, what) in [
        (
            StructuralSyncRequest::repair(s.oid, WRITER, MAX_SEQ + 1),
            "an entry seq of 2^53",
        ),
        (
            StructuralSyncRequest::repair(s.oid, WRITER, u64::MAX),
            "an entry seq of u64::MAX",
        ),
        (
            StructuralSyncRequest::repair_to(s.oid, WRITER, base, MAX_SEQ + 1),
            "a target seq of 2^53",
        ),
        (
            StructuralSyncRequest::repair_to(s.oid, WRITER, base, u64::MAX),
            "a target seq of u64::MAX",
        ),
    ] {
        let msg = sync_internal(&s.tv, &request).await;
        assert!(
            msg.contains("STRUCTURAL_SYNC_ARGS"),
            "{what}: the script refuses the argument; got {msg:?}"
        );
    }
}
