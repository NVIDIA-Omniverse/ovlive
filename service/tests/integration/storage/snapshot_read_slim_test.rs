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

//! The slim snapshot reads of the structural write-planning consumers
//! (b.mr6 T4): `storage::diff_read::read_object_snapshot_slim` on its atomic
//! path (the cold start's and the audit's `EVALSHA` with the script's `slim`
//! flag, `ARGV[13]`), on the WATCH path's unfenced fan-out, and the fenced
//! fan-out (`read_object_snapshot_fenced` for a slim consumer).
//!
//! Each slim read must return the compact projection
//! (`test_support::compact_snapshot`) of the full read of the same path, or
//! fail exactly as that full read fails. The generated histories are
//! `tests/proptests.rs`'s `proptest_every_slim_read_is_the_projected_full_read`;
//! here are the stored row shapes no commit writes, each planted on
//! `snapshot_read_atomic_test`'s `every_shape_state` object:
//!
//! | Row | Every slim path |
//! |---|---|
//! | `value` and a valid `hash` (a pre-b.zvd row) | inline, no hash; the both-shapes conflict reported once per read (on the WATCH path's read, only by its window scripts, whose commands per row run atomically; none on its plain transport, whose two commands per row can tear) |
//! | a malformed `hash`, no `value` | fails `Internal` with the full read's message |
//! | a malformed `hash` beside a `value` | inline: `value` is checked first |
//! | an empty `value`, alone or beside a writer's other columns | inline |
//! | an external row | its hash |
//! | an index member with no row (absent) | no row |
//! | foreign columns only | inline (empty; the WATCH path's read does not tally the foreign column) |
//! | a string where the row's hash should be (wrong-typed) | the full read's refusal (on the WATCH path naming `HSTRLEN field`) |
//! | a `value` that is not base64 | inline, no failure |
//!
//! The WATCH path's read is `structural_update`'s bounded-reply read (b.26e
//! Stage 2): it reads a value row's `value` length and its other columns
//! (`HSTRLEN` + `HMGET`, with `HEXISTS` / `EXISTS` follow-ups) rather than the
//! whole row, so the parenthesised differences above are its own; everywhere
//! else it equals the full fan-out's projection. It is held to that on both
//! of its transports, as two slim paths: [`SlimPath::Unfenced`], the
//! production rule, under which each window of its rounds 2 and 3 is one
//! window script (`fanout_window_read.lua`) issuing those commands inside the
//! script — but for round 1, a lone children list whose modelled shard time
//! alone exceeds τ, and every window after a script fallback, which are plain
//! commands — and [`SlimPath::UnfencedPlain`], every window plain commands
//! (forced through the test-support switch
//! `ValkeyClient::set_watch_read_transport_for_test`; the transport a read
//! falls back to on a cluster-state refusal).
//! `slim_unfenced_read_asks_each_follow_up_only_where_the_row_needs_it_b26e`
//! pins which rows take which follow-up on each transport, from a `MONITOR`
//! record.
//!
//! And neither the slim atomic reply nor the WATCH path's read carries an
//! inline value: on a layer shaped like a recorded rig (time samples of large
//! inline values) the bytes the slim atomic read received for its value rows
//! fall short of the full read's by at least the inline values' bytes, and the
//! WATCH path's read receives none, while each tallies the same `reply_bytes`
//! as its full counterpart.

use fred::interfaces::{HashesInterface, KeysInterface};

use live_edit::domain::types::{
    CompactState, Field, FieldRows, FieldValue, ObjectState, SampleRows, Section, TimeSample,
};
use live_edit::error::ServiceError;
use live_edit::storage::diff_read::{
    read_object_snapshot, read_object_snapshot_slim, AtomicSnapshotOf, ObjectSnapshot,
    SlimSnapshotReadConsumer, SnapshotConsumer, SnapshotReadConsumer, SnapshotReadOf,
    WindowTransport,
};
use live_edit::storage::snapshot_fence::read::{read_object_snapshot_fenced, CappedFenceWait};
use live_edit::storage::snapshot_fence::LayerSize;
use live_edit::storage::test_support::{compact_snapshot, TestValkey};
use ordered_float::OrderedFloat;

use crate::grpc::log_capture::capture;
use crate::storage::snapshot_read_atomic_test::{
    create, create_session, every_shape_state, ext_hash, layer_store, CUBE, WORLD,
};
use crate::storage::snapshot_read_support::{monitor_fence, test_valkey_addr, MonitorTap};

/// The size the fenced reads are given as their estimate: it only picks the
/// fence's timing coefficients, and every fixture here is far below it.
const FENCED_ESTIMATE: LayerSize = LayerSize {
    items: 1_000,
    bytes: 100_000,
};

/// The message prefix of the both-shapes (b.zvd) WARN.
const BOTH_SHAPES_MSG_PREFIX: &str = "value row carries both an inline `value` and an external";

/// The read paths a slim read takes, each with its full-read counterpart.
#[derive(Debug, Clone, Copy)]
enum SlimPath {
    /// The cold start's atomic read.
    AtomicColdStart,
    /// The audit's atomic read.
    AtomicAudit,
    /// The WATCH path's unfenced fan-out, on the production transport: each
    /// window of its rounds 2 and 3 one window script (the windows that stay
    /// plain: round 1, a lone over-τ children list, every window after a
    /// script fallback).
    Unfenced,
    /// The WATCH path's unfenced fan-out with every window issued as plain
    /// commands (`WindowTransport::Plain`): the transport it falls back to.
    UnfencedPlain,
    /// The cold start's fenced fan-out.
    Fenced,
}

impl SlimPath {
    /// Whether this path is the WATCH path's bounded-reply read, on either
    /// transport.
    fn is_watch_read(self) -> bool {
        matches!(self, SlimPath::Unfenced | SlimPath::UnfencedPlain)
    }
}

const SLIM_PATHS: [SlimPath; 5] = [
    SlimPath::AtomicColdStart,
    SlimPath::AtomicAudit,
    SlimPath::Unfenced,
    SlimPath::UnfencedPlain,
    SlimPath::Fenced,
];

/// A snapshot read's answer as a `Result` of its snapshot: `TOO_LARGE` is a
/// fixture failure here, every object being within every ceiling.
fn snapshot_of<S>(
    path: SlimPath,
    read: Result<SnapshotReadOf<S>, ServiceError>,
) -> Result<AtomicSnapshotOf<S>, ServiceError> {
    match read? {
        SnapshotReadOf::Snapshot(snapshot) => Ok(*snapshot),
        SnapshotReadOf::TooLarge(t) => panic!("{path:?}: fixture: within every ceiling; got {t:?}"),
    }
}

/// The fenced read of `oid` for `consumer`.
async fn fenced<C: SnapshotConsumer>(
    tv: &TestValkey,
    oid: u64,
    consumer: C,
) -> Result<AtomicSnapshotOf<C::State>, ServiceError> {
    read_object_snapshot_fenced(
        tv.client(),
        tv.key_schema(),
        oid,
        consumer,
        FENCED_ESTIMATE,
        &mut CappedFenceWait::new(tv.client(), oid, consumer),
    )
    .await
}

/// The slim read of `oid` on `path`.
async fn slim_read(
    tv: &TestValkey,
    oid: u64,
    path: SlimPath,
) -> Result<AtomicSnapshotOf<CompactState>, ServiceError> {
    let (client, keys, oid_str) = (tv.client(), tv.key_schema(), oid.to_string());
    let slim = |consumer| read_object_snapshot_slim(client, keys, &oid_str, consumer);
    match path {
        SlimPath::AtomicColdStart => {
            snapshot_of(path, slim(SlimSnapshotReadConsumer::StructuralCache).await)
        }
        SlimPath::AtomicAudit => snapshot_of(
            path,
            slim(SlimSnapshotReadConsumer::StructuralCacheAudit).await,
        ),
        SlimPath::Unfenced => {
            snapshot_of(path, slim(SlimSnapshotReadConsumer::StructuralUpdate).await)
        }
        SlimPath::UnfencedPlain => {
            // The switch reaches every clone of the client; put the
            // production rule back before anything else reads through it.
            client.set_watch_read_transport_for_test(WindowTransport::Plain);
            let read = slim(SlimSnapshotReadConsumer::StructuralUpdate).await;
            client.set_watch_read_transport_for_test(WindowTransport::Script);
            snapshot_of(path, read)
        }
        SlimPath::Fenced => fenced(tv, oid, SlimSnapshotReadConsumer::StructuralCache).await,
    }
}

/// The full read of `oid` on `path`: a reader's read of the same keys by the
/// same route — the catch-up's atomic read, and for the fan-outs the
/// catch-up's fenced read. The WATCH path's unfenced fan-out has no full
/// reader of its own any more (b.fsz MR-D4 removed the ext-GC snapshot read,
/// which read it whole), so its counterpart is the fenced pass: the same
/// fan-out of plain commands over the same keys, reading every row whole
/// under its lease, which on an object nothing writes reads what the
/// unfenced fan-out reads.
async fn full_read(
    tv: &TestValkey,
    oid: u64,
    path: SlimPath,
) -> Result<AtomicSnapshotOf<ObjectSnapshot>, ServiceError> {
    let (client, keys, oid_str) = (tv.client(), tv.key_schema(), oid.to_string());
    match path {
        SlimPath::AtomicColdStart | SlimPath::AtomicAudit => snapshot_of(
            path,
            read_object_snapshot(client, keys, &oid_str, SnapshotReadConsumer::Catchup).await,
        ),
        SlimPath::Unfenced | SlimPath::UnfencedPlain | SlimPath::Fenced => {
            fenced(tv, oid, SnapshotReadConsumer::Catchup).await
        }
    }
}

/// How the WATCH path's slim read (`structural_update`'s unfenced fan-out,
/// on either transport) may differ from its full counterpart, the fenced
/// fan-out's whole-row read, on a planted row. That read reads a value row by bounded replies
/// — `HSTRLEN value` plus `HMGET` of the columns writers write, then `HEXISTS
/// value` for a row whose `value` length is 0 and that has a `hash` or none
/// of those columns, and `EXISTS` for a row with none of them and no `value`
/// (b.26e Stage 2), inside a window script or as plain commands — instead of
/// the whole row, so it rebuilds the row's tally from what it read: a column
/// it does not read is not tallied (an empty `value` is, `HEXISTS` or the
/// writers' rule finding it); and its failure names the command that failed,
/// not `HGETALL`. Every other slim path reads what its full counterpart
/// reads.
#[derive(Clone, Copy, Default)]
struct UnfencedDifference {
    /// The (items, bytes) of the planted row's columns the bounded-reply
    /// read does not tally: one item per column, its name's and its value's
    /// bytes, as the full read tallies them.
    untallied: (u64, u64),
    /// The full read's failing `op` and the one the bounded-reply read names
    /// in its place.
    relabel: Option<(&'static str, &'static str)>,
}

/// On every slim path, the slim read is the compact projection of the full
/// read of that path, or fails as it fails (the same error, message
/// included). Returns the slim states.
async fn assert_every_slim_read_is_the_projection(
    tv: &TestValkey,
    oid: u64,
    context: &str,
) -> Vec<Result<CompactState, ServiceError>> {
    assert_every_slim_read_is_the_projection_but(tv, oid, context, UnfencedDifference::default())
        .await
}

/// [`assert_every_slim_read_is_the_projection`], with the WATCH path's slim
/// read, on each transport, allowed exactly `unfenced`'s difference from its
/// full counterpart.
async fn assert_every_slim_read_is_the_projection_but(
    tv: &TestValkey,
    oid: u64,
    context: &str,
    unfenced: UnfencedDifference,
) -> Vec<Result<CompactState, ServiceError>> {
    let mut states = Vec::new();
    for path in SLIM_PATHS {
        let full = full_read(tv, oid, path).await;
        let slim = slim_read(tv, oid, path).await;
        let difference = if path.is_watch_read() {
            unfenced
        } else {
            UnfencedDifference::default()
        };
        match (&full, &slim) {
            (Ok(full), Ok(slim)) => {
                assert_eq!(
                    slim.state,
                    compact_snapshot(&full.state),
                    "{context}, {path:?}: the slim read is the full read's compact projection"
                );
                let (untallied_items, untallied_bytes) = difference.untallied;
                assert_eq!(
                    (
                        slim.cut,
                        slim.item_count,
                        slim.reply_bytes,
                        slim.entry_count
                    ),
                    (
                        full.cut,
                        full.item_count - untallied_items,
                        full.reply_bytes - untallied_bytes,
                        full.entry_count
                    ),
                    "{context}, {path:?}: the cut and the (items, bytes, entries) tallies, less \
                     the columns this path does not tally ({untallied_items} items, \
                     {untallied_bytes} bytes)"
                );
            }
            (Err(full), Err(slim)) => {
                let full = format!("{full:?}");
                let full = match difference.relabel {
                    Some((full_op, slim_op)) => {
                        assert!(
                            full.contains(full_op),
                            "{context}, {path:?}: fixture: the full read names {full_op:?}; got \
                             {full}"
                        );
                        full.replace(full_op, slim_op)
                    }
                    None => full,
                };
                assert_eq!(
                    format!("{slim:?}"),
                    full,
                    "{context}, {path:?}: the slim read fails as the full read fails"
                );
            }
            _ => panic!(
                "{context}, {path:?}: one read failed and the other did not: full {:?}, slim {:?}",
                full.as_ref().map(|_| ()),
                slim.as_ref().map(|_| ())
            ),
        }
        states.push(slim.map(|read| read.state));
    }
    states
}

/// A fresh `every_shape_state` object.
async fn every_shape_object(layer: &str) -> (TestValkey, u64) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    (tv, oid)
}

/// The key of `WORLD`'s inline field `color`.
fn color_row(tv: &TestValkey, oid: u64) -> String {
    tv.key_schema()
        .obj_field(&oid.to_string(), &WORLD.to_string(), "color")
}

/// The key of `CUBE`'s external field `mesh`.
fn mesh_row(tv: &TestValkey, oid: u64) -> String {
    tv.key_schema()
        .obj_field(&oid.to_string(), &CUBE.to_string(), "mesh")
}

async fn hset(tv: &TestValkey, key: &str, columns: &[(&str, &str)]) {
    let _: () = tv
        .client()
        .pool
        .hset(key, columns.to_vec())
        .await
        .expect("HSET of the planted row must succeed");
}

async fn del(tv: &TestValkey, key: &str) {
    let _: i64 = tv
        .client()
        .pool
        .del(key)
        .await
        .expect("DEL of the row must succeed");
}

/// Every slim read's state, each required to have succeeded.
fn succeeded(states: Vec<Result<CompactState, ServiceError>>, context: &str) -> Vec<CompactState> {
    states
        .into_iter()
        .map(|state| state.unwrap_or_else(|e| panic!("{context}: the reads succeed; got {e:?}")))
        .collect()
}

/// Assert `color` is held inline (no hash) on every slim read.
fn assert_color_inline(states: &[CompactState], context: &str) {
    for state in states {
        assert!(
            state.fields.contains(WORLD, "color"),
            "{context}: the row is held"
        );
        assert_eq!(
            state.fields.ext_hash(WORLD, "color"),
            None,
            "{context}: the row is inline"
        );
    }
}

/// The both-shapes (b.zvd) conflict lines one slim read of `oid` on `path`
/// emits, which must succeed, with the capture's text for a failure message.
async fn both_shapes_lines(
    tv: &TestValkey,
    oid: u64,
    path: SlimPath,
) -> (Vec<serde_json::Value>, String) {
    let (read, captured) = capture(slim_read(tv, oid, path)).await;
    read.unwrap_or_else(|e| panic!("{path:?}: the read succeeds; got {e:?}"));
    let lines = captured
        .events
        .iter()
        .filter(|e| {
            e["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with(BOTH_SHAPES_MSG_PREFIX))
        })
        .cloned()
        .collect();
    (lines, captured.text)
}

/// A row holding `value` and a valid `hash` (a pre-b.zvd row): every slim
/// path reads it inline with no hash, as the full read decodes the `value`.
/// Every path whose read of the row is atomic reports the both-shapes
/// conflict once per read — the whole-row readers, and the WATCH path's read
/// on its window scripts (b.gvf), which run a row's `HSTRLEN` and `HMGET` in
/// one script, so a value length beside a `hash` was on the row at once
/// (`ValueRowReply::shape`). The WATCH path's plain transport reports none:
/// its two commands per row can run on different connections with other
/// clients' writes between them, so a value beside a `hash` there may be a
/// torn read of a rewrite (b.26e Stage 2).
///
/// Before the window script the WATCH path's read issued every window plain
/// and reported no conflict, which fails the [`SlimPath::Unfenced`] arm.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slim_reads_of_a_legacy_row_with_value_and_hash_are_inline_and_report_the_conflict_where_the_row_is_read_atomically(
) {
    let (tv, oid) = every_shape_object("slim-legacy").await;
    hset(
        &tv,
        &color_row(&tv, oid),
        &[("hash", "ab".repeat(32).as_str())],
    )
    .await;

    let states = succeeded(
        assert_every_slim_read_is_the_projection(&tv, oid, "legacy row").await,
        "legacy row",
    );
    assert_color_inline(&states, "legacy row");

    for path in SLIM_PATHS {
        let (conflicts, text) = both_shapes_lines(&tv, oid, path).await;
        if let SlimPath::UnfencedPlain = path {
            assert!(
                conflicts.is_empty(),
                "{path:?}: the plain transport's two commands per row can tear, so it reports \
                 no conflict; capture:\n{text}"
            );
            continue;
        }
        assert_eq!(
            conflicts.len(),
            1,
            "{path:?}: one conflict line per read; capture:\n{text}"
        );
        assert_eq!(conflicts[0]["fields"]["conflicts"], 1, "{path:?}");
        assert_eq!(conflicts[0]["fields"]["exemplar_name"], "color", "{path:?}");
    }
}

/// Two field rows holding both shapes — the legacy `color` (its `value` and
/// a planted `hash`) and the external `mesh` (its `hash` and `size` and a
/// planted `value`): every path that reports the conflict reports both rows
/// in its one aggregated line (`conflicts` 2, the exemplar one of them), the
/// WATCH path's read on its window scripts included; its plain transport
/// reports none. Both rows read inline, as the full read decodes `value`
/// first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slim_reads_of_two_legacy_rows_report_both_in_one_conflict_line_where_the_rows_are_read_atomically(
) {
    let (tv, oid) = every_shape_object("slim-legacy-two").await;
    hset(
        &tv,
        &color_row(&tv, oid),
        &[("hash", "ab".repeat(32).as_str())],
    )
    .await;
    // `AQI=` is two bytes, base64.
    hset(&tv, &mesh_row(&tv, oid), &[("value", "AQI=")]).await;

    let states = succeeded(
        assert_every_slim_read_is_the_projection(&tv, oid, "two legacy rows").await,
        "two legacy rows",
    );
    for state in &states {
        for (section, name) in [(WORLD, "color"), (CUBE, "mesh")] {
            assert_eq!(
                state.fields.ext_hash(section, name),
                None,
                "two legacy rows: {name} reads inline"
            );
        }
    }

    for path in SLIM_PATHS {
        let (conflicts, text) = both_shapes_lines(&tv, oid, path).await;
        if let SlimPath::UnfencedPlain = path {
            assert!(
                conflicts.is_empty(),
                "{path:?}: the plain transport reports no conflict; capture:\n{text}"
            );
            continue;
        }
        assert_eq!(
            conflicts.len(),
            1,
            "{path:?}: the two rows' conflicts are one aggregated line; capture:\n{text}"
        );
        assert_eq!(
            conflicts[0]["fields"]["conflicts"], 2,
            "{path:?}: both rows are counted; capture:\n{text}"
        );
        let exemplar = conflicts[0]["fields"]["exemplar_name"].as_str();
        assert!(
            matches!(exemplar, Some("color") | Some("mesh")),
            "{path:?}: the exemplar is one of the two rows; got {exemplar:?}"
        );
    }
}

/// A malformed `hash` with no `value`: every slim path fails `Internal`,
/// with the full read's message.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slim_reads_of_a_malformed_hash_fail_internal_as_the_full_read_does() {
    let (tv, oid) = every_shape_object("slim-malformed").await;
    hset(&tv, &mesh_row(&tv, oid), &[("hash", "zz")]).await;

    for (path, state) in SLIM_PATHS
        .into_iter()
        .zip(assert_every_slim_read_is_the_projection(&tv, oid, "malformed hash").await)
    {
        match state {
            Err(ServiceError::Internal(msg)) => assert!(
                msg.contains("malformed external-value `hash` column"),
                "{path:?}: names the malformed column; got {msg:?}"
            ),
            other => panic!("{path:?}: must fail Internal; got {:?}", other.map(|_| ())),
        }
    }
}

/// A malformed `hash` beside a `value`: `value` is checked first, so every
/// slim path reads the row inline and never validates the `hash`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slim_reads_of_a_malformed_hash_beside_a_value_are_inline() {
    let (tv, oid) = every_shape_object("slim-shadowed").await;
    hset(&tv, &color_row(&tv, oid), &[("hash", "zz")]).await;

    let states = succeeded(
        assert_every_slim_read_is_the_projection(&tv, oid, "shadowed hash").await,
        "shadowed hash",
    );
    assert_color_inline(&states, "shadowed hash");
}

/// An empty inline `value`, a `value` that is not base64, and a row holding
/// only columns no reader knows: each is an inline row on every slim path
/// (the full read decodes the last two to an empty inline value, PD2), and
/// none fails the read.
///
/// The WATCH path's bounded-reply read tallies what it reads: the `value`
/// of the bad-base64 row by its `HSTRLEN`, and
/// the empty `value` — a row holding nothing else — by its `HEXISTS`
/// follow-up, each as the full read does; but not the foreign column, which
/// no writer writes and it never reads: that row is one item and its name's
/// and value's bytes short of the full read's tally.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slim_reads_of_empty_bad_base64_and_foreign_only_rows_are_inline() {
    let (tv, oid) = every_shape_object("slim-inline-shapes").await;
    let row = color_row(&tv, oid);
    let untallied = |column: &str, value: &str| (1, (column.len() + value.len()) as u64);
    for (shape, columns, untallied) in [
        ("empty value", vec![("value", "")], (0, 0)),
        ("bad base64", vec![("value", "!!not base64!!")], (0, 0)),
        (
            "foreign columns only",
            vec![("foreign", "x")],
            untallied("foreign", "x"),
        ),
    ] {
        del(&tv, &row).await;
        hset(&tv, &row, &columns).await;

        let states = succeeded(
            assert_every_slim_read_is_the_projection_but(
                &tv,
                oid,
                shape,
                UnfencedDifference {
                    untallied,
                    relabel: None,
                },
            )
            .await,
            shape,
        );
        assert_color_inline(&states, shape);
    }
}

/// The follow-up commands the WATCH path's read on `path` (one of its two
/// transports) issued on `row`, in the order the store ran them, from a
/// `MONITOR` record of one such read of `oid`: every `HEXISTS` and `EXISTS`
/// naming the row. On [`SlimPath::Unfenced`] every command on the row is the
/// window script's (the feed's `lua` client) and none the client's; on
/// [`SlimPath::UnfencedPlain`] the reverse. Asserts the read issued the row's
/// `HSTRLEN` and `HMGET` exactly once, so an empty answer means no
/// follow-up, not a row the read never reached.
async fn unfenced_follow_ups(
    tv: &TestValkey,
    oid: u64,
    row: &str,
    context: &str,
    path: SlimPath,
) -> Vec<String> {
    let by_script = match path {
        SlimPath::Unfenced => true,
        SlimPath::UnfencedPlain => false,
        other => panic!("fixture: {other:?} is not the WATCH path's read"),
    };
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let read = slim_read(tv, oid, path).await;
    let fence = monitor_fence(tv.prefix());
    let _: i64 = tv
        .client()
        .pool
        .exists(&fence)
        .await
        .expect("EXISTS the fence must succeed");
    let seen = tap.drain_until(&fence).await;
    read.unwrap_or_else(|e| {
        panic!("{context}, {path:?}: the WATCH path's read succeeds; got {e:?}")
    });
    let (on_row, by_the_other): (Vec<_>, Vec<_>) = seen
        .iter()
        .filter(|c| c.args.get(1).map(String::as_str) == Some(row))
        .partition(|c| c.issued_by_script() == by_script);
    assert!(
        by_the_other.is_empty(),
        "{context}, {path:?}: every command on the row is issued by {}; issued otherwise: \
         {by_the_other:?}",
        if by_script {
            "the window script"
        } else {
            "the client"
        }
    );
    let on_row: Vec<String> = on_row.iter().map(|c| c.name()).collect();
    for first in ["HSTRLEN", "HMGET"] {
        assert_eq!(
            on_row.iter().filter(|name| *name == first).count(),
            1,
            "{context}, {path:?}: the read of the row issues one {first}; the row's commands: \
             {on_row:?}"
        );
    }
    on_row
        .into_iter()
        .filter(|name| name == "HEXISTS" || name == "EXISTS")
        .collect()
}

/// b.26e Stage 2: the WATCH path's bounded-reply read
/// asks a value row's follow-ups only where its first two replies leave the
/// row's shape or tally open, and each shape's tally then equals the whole
/// row's read's (the full fan-out's), but for the foreign column it never
/// reads. The same table holds on both transports (b.gvf): the window script
/// issues each row's commands by the plain read's rule
/// (`ValueRowReply::asks_value_exists` / `asks_row_exists`), and the Rust
/// decode rejects a reply whose follow-ups differ from it. A `MONITOR` record
/// of each transport's read shows, per planted `color` row, the commands the
/// window script issued (the production transport) or the client issued (the
/// plain transport):
///
/// | Row | Follow-ups | Why |
/// |---|---|---|
/// | the writer's inline row | none | a value length |
/// | a writer-shaped row with an empty `value` (its `version` and orders kept) | none | no `hash` but a writer's column: inline by the writers' rule, its empty `value` tallied |
/// | an empty `value` alone | `HEXISTS` | no known column: `HEXISTS` finds the `value`, so no `EXISTS` |
/// | a `hash` beside an empty `value` | `HEXISTS` | a `hash` and a length of 0: `HEXISTS` decides inline |
/// | the writer's external row (`mesh`) | `HEXISTS` | a `hash` alone: `HEXISTS` answers no `value`; a known column, so no `EXISTS` |
/// | foreign columns only | `HEXISTS`, then `EXISTS` | no known column and no `value`: `EXISTS` finds the row |
/// | absent (an index member whose row is gone) | `HEXISTS`, then `EXISTS` | as above; `EXISTS` finds nothing |
///
/// The empty-`value`-alone, foreign and absent rows fail on the read before
/// the empty-inline change (`EXISTS` alone, and the lone empty `value`
/// untallied); the rest pin the rule.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slim_unfenced_read_asks_each_follow_up_only_where_the_row_needs_it_b26e() {
    /// Which planted row a case reads.
    #[derive(Clone, Copy)]
    enum Row {
        /// `WORLD`'s inline `color`.
        Color,
        /// `CUBE`'s external `mesh`.
        Mesh,
    }
    /// How a case plants its row on a fresh `every_shape_state` object.
    enum Planting {
        /// The row as its writer wrote it.
        AsWritten,
        /// These columns written over the writer's row, the rest kept.
        Over(Vec<(&'static str, String)>),
        /// The row deleted, then these columns written (none: absent).
        Replaced(Option<Vec<(&'static str, String)>>),
    }
    /// (shape, the row it reads, its planting, the follow-ups the read
    /// asks, in order, and the (items, bytes) it does not tally).
    type Case = (
        &'static str,
        Row,
        Planting,
        &'static [&'static str],
        (u64, u64),
    );
    let cases: Vec<Case> = vec![
        (
            "writer's inline row",
            Row::Color,
            Planting::AsWritten,
            &[],
            (0, 0),
        ),
        (
            "writer-shaped empty value",
            Row::Color,
            Planting::Over(vec![("value", String::new())]),
            &[],
            (0, 0),
        ),
        (
            "empty value alone",
            Row::Color,
            Planting::Replaced(Some(vec![("value", String::new())])),
            &["HEXISTS"],
            (0, 0),
        ),
        (
            "hash beside an empty value",
            Row::Color,
            Planting::Replaced(Some(vec![
                ("hash", "ab".repeat(32)),
                ("value", String::new()),
            ])),
            &["HEXISTS"],
            (0, 0),
        ),
        (
            "writer's external row",
            Row::Mesh,
            Planting::AsWritten,
            &["HEXISTS"],
            (0, 0),
        ),
        (
            "foreign columns only",
            Row::Color,
            Planting::Replaced(Some(vec![("foreign", "x".to_owned())])),
            &["HEXISTS", "EXISTS"],
            (1, ("foreign".len() + "x".len()) as u64),
        ),
        (
            "absent",
            Row::Color,
            Planting::Replaced(None),
            &["HEXISTS", "EXISTS"],
            (0, 0),
        ),
    ];
    for (i, (shape, row, planting, want, untallied)) in cases.into_iter().enumerate() {
        let (tv, oid) = every_shape_object(&format!("slim-follow-ups-{i}")).await;
        let row = match row {
            Row::Color => color_row(&tv, oid),
            Row::Mesh => mesh_row(&tv, oid),
        };
        fn as_pairs<'a>(columns: &'a [(&'static str, String)]) -> Vec<(&'a str, &'a str)> {
            columns.iter().map(|(k, v)| (*k, v.as_str())).collect()
        }
        match &planting {
            Planting::AsWritten => {}
            Planting::Over(columns) => hset(&tv, &row, &as_pairs(columns)).await,
            Planting::Replaced(columns) => {
                del(&tv, &row).await;
                if let Some(columns) = columns {
                    hset(&tv, &row, &as_pairs(columns)).await;
                }
            }
        }

        assert_every_slim_read_is_the_projection_but(
            &tv,
            oid,
            shape,
            UnfencedDifference {
                untallied,
                relabel: None,
            },
        )
        .await;
        for path in [SlimPath::Unfenced, SlimPath::UnfencedPlain] {
            let follow_ups = unfenced_follow_ups(&tv, oid, &row, shape, path).await;
            assert_eq!(
                follow_ups, want,
                "{shape}, {path:?}: the WATCH path's read asks exactly these follow-ups of the \
                 row, in order"
            );
        }
    }
}

/// An external row: every slim path keeps its hash; the full read's value
/// row bookkeeping (size, index) is not held.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slim_reads_of_an_external_row_keep_its_hash() {
    let (tv, oid) = every_shape_object("slim-external").await;

    let states = succeeded(
        assert_every_slim_read_is_the_projection(&tv, oid, "external row").await,
        "external row",
    );
    for state in states {
        assert_eq!(state.fields.ext_hash(CUBE, "mesh"), Some(ext_hash(0xAB)));
        assert_eq!(
            state.time_samples.ext_hash(CUBE, 1.5f64.to_bits()),
            Some(ext_hash(0xCD))
        );
    }
}

/// An index member whose row is gone: no row on any slim path, as on the
/// full read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slim_reads_of_an_index_member_without_a_row_hold_no_row() {
    let (tv, oid) = every_shape_object("slim-absent").await;
    del(&tv, &color_row(&tv, oid)).await;

    let states = succeeded(
        assert_every_slim_read_is_the_projection(&tv, oid, "absent row").await,
        "absent row",
    );
    for state in states {
        assert!(!state.fields.contains(WORLD, "color"));
        assert_eq!(FieldRows::len(&state.fields), 1, "only `mesh` is held");
    }
}

/// A row key holding a string: every slim path refuses it as the full read
/// of its path does — on the WATCH path naming the bounded-reply read's
/// `HSTRLEN field` where the full read names its `HGETALL field`, the
/// message otherwise the same.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slim_reads_of_a_wrong_typed_row_fail_as_the_full_read_does() {
    let (tv, oid) = every_shape_object("slim-wrongtype").await;
    let row = color_row(&tv, oid);
    del(&tv, &row).await;
    let _: () = tv
        .client()
        .pool
        .set(&row, "not-a-hash", None, None, false)
        .await
        .expect("SET of the wrong-typed row must succeed");

    for (path, state) in SLIM_PATHS.into_iter().zip(
        assert_every_slim_read_is_the_projection_but(
            &tv,
            oid,
            "wrong-typed row",
            UnfencedDifference {
                untallied: (0, 0),
                relabel: Some(("HGETALL field", "HSTRLEN field")),
            },
        )
        .await,
    ) {
        assert!(state.is_err(), "{path:?}: a wrong-typed row fails the read");
    }
}

/// Time samples of [`RIG_SAMPLE_BYTES`] inline bytes in [`rig_shaped_state`].
const RIG_FRAMES: u32 = 1_000;
/// Each sample's inline value: large enough that the values dominate the
/// rows, small enough that the layer stays within the default atomic byte
/// ceiling (`SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT`).
const RIG_SAMPLE_BYTES: usize = 256;

/// A recorded-rig-shaped layer: a handful of sections and fields, and
/// [`RIG_FRAMES`] time samples of [`RIG_SAMPLE_BYTES`] inline bytes on one
/// section — the shape whose rows are almost all inline value.
fn rig_shaped_state() -> ObjectState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        next_set_order: 10,
        ..Default::default()
    };
    for (id, parent) in [(1u64, 0u64), (2, 1), (3, 2), (4, 2)] {
        state.sections.insert(
            id,
            Section {
                section_id: id,
                parent_id: parent,
                name: format!("rig{id}"),
                section_type: 1,
                section_order: 0,
                first_version: 1,
                last_version: 1,
            },
        );
    }
    state.children.insert(0, vec![1]);
    state.children.insert(1, vec![2]);
    state.children.insert(2, vec![3, 4]);
    for (id, key) in [(3u64, "typeName"), (3, "default"), (4, "typeName")] {
        state.fields.insert(
            (id, key.to_owned()),
            Field {
                section_id: id,
                key_name: key.to_owned(),
                value: FieldValue::Inline(vec![1; 8]),
                set_order: 1,
                authoring_set_order: 1,
                version: 1,
            },
        );
    }
    for frame in 0..RIG_FRAMES {
        let time = OrderedFloat(f64::from(frame));
        state.time_samples.insert(
            (4, time),
            TimeSample {
                section_id: 4,
                time,
                value: FieldValue::Inline(vec![(frame % 251) as u8; RIG_SAMPLE_BYTES]),
                version: 1,
            },
        );
    }
    state
}

/// The slim atomic reply carries no inline value: on a rig-shaped layer the
/// value-row bytes the slim atomic read received fall short of the full
/// atomic read's by at least the layer's inline value bytes, while both
/// tally the same store-side `reply_bytes`. (The fenced fan-out reads every
/// row whole for every consumer; the WATCH path's fan-out has its own test
/// below.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slim_atomic_read_of_a_rig_shaped_layer_receives_no_inline_values() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "slim-rig";
    let session_id = create_session(&tv, layer).await;
    let state = rig_shaped_state();
    let oid = create(&ls, &session_id, layer, &state).await;
    let inline_bytes = (RIG_FRAMES as u64) * (RIG_SAMPLE_BYTES as u64) + 3 * 8;

    let full = full_read(&tv, oid, SlimPath::AtomicColdStart)
        .await
        .expect("the full atomic read succeeds");
    let slim = slim_read(&tv, oid, SlimPath::AtomicColdStart)
        .await
        .expect("the slim atomic read succeeds");

    assert_eq!(
        SampleRows::len(&slim.state.time_samples),
        RIG_FRAMES as usize,
        "fixture: every frame is read"
    );
    assert_eq!(
        slim.reply_bytes, full.reply_bytes,
        "the same store-side tally"
    );
    assert!(
        full.value_row_bytes_received >= slim.value_row_bytes_received + inline_bytes,
        "the slim reply must carry no inline value: full received {} value-row bytes, slim {}, \
         inline values {inline_bytes}",
        full.value_row_bytes_received,
        slim.value_row_bytes_received
    );
}

/// b.26e Stage 2 (test 16): the WATCH path's read (`structural_update`'s
/// unfenced fan-out) transfers no inline value, on either transport (its
/// window scripts return a value row's length, never its value). On the
/// rig-shaped layer,
/// whose value rows are all inline, it receives no value-row bytes at all
/// (`value_row_bytes_received` counts a bounded row's `hash` and `size`
/// columns, which an inline row lacks), where the full fan-out over the same
/// keys (the fenced read's pass, reading every row whole) receives at least
/// every inline value's bytes. The
/// two tally the same store-side items, bytes and entries: every row here is
/// writer-shaped, and the bounded read rebuilds each row's tally as the
/// whole row's read would.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slim_unfenced_read_of_a_rig_shaped_layer_receives_no_inline_values_b26e() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "slim-rig-unfenced";
    let session_id = create_session(&tv, layer).await;
    let state = rig_shaped_state();
    let oid = create(&ls, &session_id, layer, &state).await;
    let inline_bytes = (RIG_FRAMES as u64) * (RIG_SAMPLE_BYTES as u64) + 3 * 8;

    let full = full_read(&tv, oid, SlimPath::Unfenced)
        .await
        .expect("the full fan-out read succeeds");
    assert!(
        full.value_row_bytes_received >= inline_bytes,
        "control: the full fan-out does transfer every inline value ({} value-row bytes \
         received, {inline_bytes} inline)",
        full.value_row_bytes_received
    );
    for path in [SlimPath::Unfenced, SlimPath::UnfencedPlain] {
        let slim = slim_read(&tv, oid, path)
            .await
            .unwrap_or_else(|e| panic!("{path:?}: the bounded-reply read succeeds; got {e:?}"));

        assert_eq!(
            SampleRows::len(&slim.state.time_samples),
            RIG_FRAMES as usize,
            "{path:?}: fixture: every frame is read"
        );
        assert_eq!(slim.state, compact_snapshot(&full.state), "{path:?}");
        assert_eq!(
            slim.value_row_bytes_received, 0,
            "{path:?}: the bounded-reply read must transfer no inline value"
        );
        assert_eq!(
            (slim.item_count, slim.reply_bytes, slim.entry_count),
            (full.item_count, full.reply_bytes, full.entry_count),
            "{path:?}: the same store-side tallies"
        );
    }
}
