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

//! b.zvd — three silent-corruption defects on the write and read paths, all
//! reachable through ordinary client traffic and none of them producing an
//! error anyone could see.
//!
//! # The three defects
//!
//! 1. **The children-list read swallowed every failure.** The snapshot
//!    loader of the time (`load_object_snapshot`, since replaced by the
//!    atomic read and the fan-outs) ran a per-parent `LRANGE` on
//!    `le:{oid}:obj:ch:{pid}` as `pool.lrange(...).await.unwrap_or_default()`,
//!    so a `Timeout` / `IO` / `Canceled` / `WRONGTYPE` all read as "this
//!    parent has no children". The *transient* half is the severe one: on the
//!    update slow path `apply_delta` then computes against that wrong state and
//!    a structural delta emits
//!    `SetChildren` / `DeleteChildrenList`, which `DEL`s a REAL children list
//!    and `SREM`s `obj:chs` — destroying structure under a fully valid `WATCH`.
//!    Every snapshot consumer (catch-up, subscribe reconstruction, checkpoint)
//!    inherited the same silent truncation.
//!
//! 2. **A value row could carry both value shapes at once.** `HSET` merges, so
//!    a field first written inline (`value`) and later rewritten as an external
//!    value (`hash` + `size`) kept the stale `value` column — and
//!    `decode_field_value` checks `value` first, so the row decoded as the OLD
//!    inline bytes forever. Reachable through the ordinary client path
//!    (`domain::delta::field_value_from_oodf` picks External purely on
//!    `ext_value_size > 0`, a per-write client decision), symmetric in both
//!    rewrite directions, and it also stopped `schedule_old_object_ext_gc` from
//!    ever seeing the shadowed `hash` — a permanent blob leak. Fixed at the
//!    writer: all six value-write sites now `HDEL value hash size` immediately
//!    before their `HSET` — the four Rust ones (`execute_mutation`'s `SetField`
//!    and `SetTimeSample` arms, `write_object`'s fields and time-samples loops)
//!    plus the Lua fast path's step-6 `SF` and `ST` arms — so **a value row
//!    holds exactly one shape, always.** (SDD §7.5.4's count; the two Lua arms
//!    are separate `redis.call("HDEL", …)` lines, not one apply pass.)
//!
//! 3. **`write_object` had no pre-`MULTI` type probe.** b.459 gave
//!    `update_layer`'s slow path one; the create/restore path did not have it,
//!    so a key already holding a foreign Valkey type half-committed the
//!    materialization inside `EXEC` exactly as the update path used to.
//!    `write_object`'s children loop also `RPUSH`ed without a preceding `DEL`,
//!    so a DR restore over a surviving keyspace appended behind the survivors
//!    instead of replacing them.
//!
//! # What lives here
//!
//! | Test | Sub-task | What it pins |
//! |---|---|---|
//! | [`snapshot_read_of_a_wrongtype_children_list_is_a_loud_internal_naming_the_parent`] | 1 | the read side of the classification: a wrong-typed list is a loud `Internal` naming the parent, not an empty children map |
//! | [`update_layer_lua_field_value_shape_rewrites_leave_exactly_one_shape`] / [`update_layer_txn_field_value_shape_rewrites_leave_exactly_one_shape`] | 2 | inline→external→inline on a FIELD, on both commit backends: the decoded value is the new one and the raw row carries `value` XOR `hash`+`size` |
//! | [`update_layer_lua_time_sample_value_shape_rewrites_leave_exactly_one_shape`] / [`update_layer_txn_time_sample_value_shape_rewrites_leave_exactly_one_shape`] | 2 | the same for a TIME SAMPLE, which shares the value encoder/decoder and inherited the same hazard |
//! | [`create_layer_over_a_stale_inline_field_writes_only_the_external_shape`] / [`create_layer_over_a_stale_inline_time_sample_writes_only_the_external_shape`] | 2 | `write_object`'s own two `HDEL`s — one per loop — staged on a pre-planted row at the id the next `create_layer` mints |
//! | [`snapshot_read_of_a_dual_shape_field_decodes_value_first_and_warns`] / [`snapshot_read_of_a_dual_shape_time_sample_decodes_value_first_and_warns`] | 2 | the reader reports rather than repairs: a pre-b.zvd row still decodes (value-first) and raises ONE aggregated `warn` carrying the row kind, the count and an exemplar — one tally per family, so one test per family |
//! | [`create_layer_refuses_whole_when_a_key_it_would_write_holds_the_wrong_type`] | 3 | the create entry point's refusal: `create_layer keyspace type error: ` prefix, `site="create"` ticked once, nothing written |
//! | [`create_layer_with_more_keys_than_one_probe_chunk_succeeds`] | 3 | the chunked probe's positive control: a key set spanning five `EVALSHA`s across TWO waves still commits whole, boundary rows and all, and never ticks the type-error counter |
//! | [`create_layer_wrongtype_in_the_last_probe_chunk_is_still_refused_whole`] | 3 | the fail-fast wave loop: a wrong-typed key in the SECOND wave refuses identically, ticks the counter exactly once (not per chunk, not per wave), and writes nothing |
//! | [`checkpoint_restore_refuses_whole_when_a_key_it_would_write_holds_the_wrong_type`] | 3 | the same guard on the DR restore entry point, which is the one that materializes over a keyspace that may have survived |
//! | [`checkpoint_restore_replaces_a_surviving_children_list_rather_than_appending`] | 3 | the `DEL`-before-`RPUSH` in `write_object`'s children loop |
//!
//! Three feature-gated families carry the arms a live server cannot stage:
//! [`snapshot_read_fault_injection`] (sub-task 1's transient arm: a failed
//! children `LRANGE` on the structural commit path's unfenced read and on the
//! fenced read, plus a failure of the commit path's read as a whole; the
//! atomic read's transient failure is tested by
//! `snapshot_read_atomic_test::fault_seam::atomic_read_script_level_failures_take_the_unchanged_classifier`
//! and
//! `b_atj_tests::non_wrongtype_snapshot_read_errors::snapshot_read_refused_by_a_transient_timeout_ticks_neither_counter`),
//! [`create_probe_fault_injection`] (a transient failure of
//! the create-site type probe), and
//! [`ext_index_read_fault_injection`] (b.j2x: the `HGETALL obj:ext` beside the
//! create-site probe failing first, which must not displace the probe's
//! verdict on either entry point, and failing over a clean keyspace, where its
//! own error must still refuse). Each has its own header table.
//!
//! # What is deliberately NOT re-tested here
//!
//! The **commit-path** refusal over a planted `WRONGTYPE` children list. That
//! is the same scenario `b_459_tests::txn_wrongtype_children_list_key_is_caught_by_the_snapshot_read_not_the_probe`
//! and its `DEL`-only sibling now assert — b.zvd is exactly what moved those two
//! from asserting a heal to asserting the snapshot read's refusal, and
//! duplicating them here would give the contract two homes that could drift.
//! (Since the structural path moved to `read_object_snapshot`'s unfenced
//! fan-out, that refusal holds for a list key the read covers; on an object
//! carrying the `chs_complete` marker a list `obj:chs` does not name is not
//! read, and `b_459_tests` pins the commit's heal of it too.) What this file
//! adds on that axis is the READ side (the first row above), which no b.459
//! test covers because b.459 is about commits.
//!
//! The pure inventory/classification properties — `create_probe_keys`' key set
//! and ordering, its two exclusions, and `ProbeSite`'s metric cell / message
//! prefix / transient-classification table — are unit tests beside the
//! functions in `src/storage/layer.rs`, where they are compile-cheap and need
//! no Valkey. This file is the behavioural half.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use fred::interfaces::{HashesInterface, KeysInterface, ListInterface, SetsInterface};
use ordered_float::OrderedFloat;
use tracing_subscriber::fmt::MakeWriter;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::apply::apply_checkpoint;
use live_edit::checkpoint::assembler::{assemble_layer_checkpoint, AssemblyDecision};
use live_edit::checkpoint::{CheckpointError, CheckpointMeta, CheckpointStore};
use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section, TimeSample};
use live_edit::error::{KeyspaceRefusal, ServiceError};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::diff_read::ObjectSnapshot;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::{
    LayerStore, KEYSPACE_PROBE_CHUNK_SIZE, KEYSPACE_PROBE_CONCURRENCY,
};
use live_edit::storage::lua::is_fast_path;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::{read_object_state_for_test, TestValkey};

use super::b_atj_tests::fenced_read;
use super::snapshot_read_support::read_atomic;
use crate::common::owned_delta::{
    delta_with_set_field_external, delta_with_set_field_inline, delta_with_time_sample_external,
    delta_with_time_sample_inline,
};

// ═════════════════════════════════════════════════════════════════════════════
// Contract constants — restated, not imported
// ═════════════════════════════════════════════════════════════════════════════

/// The operator-visible prefix `write_object`'s pre-`MULTI` probe produces.
///
/// Restated rather than imported for the same reason `b_459_tests` restates its
/// update-path twin: the items that build it (`ProbeSite::message_prefix`,
/// `KEYSPACE_WRONGTYPE_TAG`) are private to `src/storage/layer.rs`, and the
/// string IS the contract — a test that imported it could not fail when it
/// changed.
const CREATE_KEYSPACE_TYPE_ERROR_PREFIX: &str = "create_layer keyspace type error: WRONGTYPE_KEY:";

/// The `entry_point` log-field value the DR restore path carries. Distinguishes
/// the two callers that share `write_object`'s one metric cell.
const ENTRY_POINT_CHECKPOINT_RESTORE: &str = "checkpoint_restore";

/// The section every value row and planted key in this file hangs off. The
/// seam module's structural tests mint their own sections (`CHILD_A` /
/// `CHILD_B`) because they are about the children list rather than about a row.
const SECTION_ID: u64 = 1;

/// The time carried by every time-sample row here. Its decimal spelling is
/// also the `exemplar_name` the reader's aggregated warn reports for a
/// time-sample row (`t=2.5`), so it must stay exactly representable in `f64`.
const SAMPLE_TIME: f64 = 2.5;

/// The key name carried by every field row here, and the `exemplar_name` the
/// reader's aggregated warn reports for one.
const FIELD_NAME: &str = "color";

/// The second field / time sample [`RowKind::add_second_row`] adds, used only
/// by the dual-shape test's aggregation arm.
const SECOND_FIELD_NAME: &str = "roughness";
/// The time-sample twin of [`SECOND_FIELD_NAME`].
const SECOND_SAMPLE_TIME: f64 = 3.5;

/// The `version` both second rows carry, deliberately above the seeded rows' 1.
///
/// The snapshot reads walk each value family in `ZRANGE <index> 0 -1` rank
/// order and the index score IS the version, so a higher version puts the
/// second row strictly after the first — and `ValueShapeConflictTally::observe`
/// runs in that same walk order. That is what keeps the *first* offender, and
/// therefore the reported exemplar, the row [`RowKind::exemplar_name`] names.
/// Leaving both at version 1 would hand the exemplar to Valkey's
/// member-lexicographic tie-breaking instead.
const SECOND_ROW_VERSION: u64 = 2;

/// How many fields [`wide_state`] carries — **derived from the production chunk
/// size AND wave width, not restated beside them.**
///
/// Sized so the probe key set spans more than one WAVE, which is the unit
/// `probe_keyspace_types` actually loops over: it walks
/// `chunks.chunks_mut(KEYSPACE_PROBE_CONCURRENCY)`, `join_all`s each wave, then
/// scans that wave's results before issuing the next. At
/// `4 × 500 + 100 = 2 100` fields the key set is 2 105 keys → 5 chunks → two
/// waves (chunks 0-3, then chunk 4). See [`wide_state_probe_key_count`] for the
/// key arithmetic.
///
/// **One wave is not enough.** A fixture that fits in a single wave leaves the
/// outer loop's second iteration exercised by nothing, so a regression that
/// processed only the first wave — leaving every probe key past
/// `CONCURRENCY × CHUNK` unguarded — would pass the whole suite. That is the
/// same class of hole the pre-chunking fixture had, one level up.
///
/// The import is the point. Both constants are **tunables**, not contracts —
/// their doc-comments invite an operator or a future benchmark to retune them —
/// and what these tests assert is behaviour AT the wave boundary wherever that
/// boundary sits, never that it sits at 2 000 keys. A restated literal gets
/// that exactly backwards: retune either constant upward and a hard-coded
/// field count silently collapses to a single wave, so both tests keep passing
/// while covering the case they exist to replace — green, and testing nothing.
/// Deriving from the real values means the fixture grows with them and the
/// coverage survives the retune.
///
/// That is the opposite call from [`CREATE_KEYSPACE_TYPE_ERROR_PREFIX`], and
/// deliberately so: there the string IS the contract, so a test that imported
/// it could not fail when it changed. Here the numbers are implementation
/// details the tests must track rather than pin.
const CHUNK_SPAN_FIELD_COUNT: usize = KEYSPACE_PROBE_CONCURRENCY * KEYSPACE_PROBE_CHUNK_SIZE + 100;

/// A 32-byte external-value hash. `byte` makes two calls distinguishable, which
/// is what lets an external→external rewrite be told from a no-op.
fn ext_hash(byte: u8) -> Vec<u8> {
    vec![byte; 32]
}

/// The hex spelling `encode_value_fields` writes for [`ext_hash`].
fn ext_hash_hex(byte: u8) -> String {
    (0..32).map(|_| format!("{byte:02x}")).collect()
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixture helpers
// ═════════════════════════════════════════════════════════════════════════════

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
}

async fn create_session(tv: &TestValkey, name: &str) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, &format!("/{name}"), "b-zvd-user")
        .await
        .expect("session creation must succeed")
}

/// An `ObjectState` with no sections, fields or time samples — the shape
/// `create_layer` materializes for a fresh layer whose content arrives by
/// `update_layer`.
fn empty_state() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        next_set_order: 0,
        ..Default::default()
    }
}

/// A minimal populated state: one root-child section, its children entry, one
/// inline field and one inline time sample.
///
/// Used by the two `write_object` families — it names one key of every family
/// `create_probe_keys` walks, so a probe that stopped covering a family shows up
/// as a missing refusal rather than as a passing test over a thinner state.
fn seeded_state() -> ObjectState {
    let mut state = empty_state();
    state.sections.insert(
        SECTION_ID,
        Section {
            section_id: SECTION_ID,
            parent_id: 0,
            name: "World".to_owned(),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        },
    );
    state.children.insert(0, vec![SECTION_ID]);
    state.fields.insert(
        (SECTION_ID, FIELD_NAME.to_owned()),
        Field {
            section_id: SECTION_ID,
            key_name: FIELD_NAME.to_owned(),
            value: FieldValue::Inline(RowKind::Field.seeded_inline_value()),
            set_order: 1,
            authoring_set_order: 1,
            version: 1,
        },
    );
    state.time_samples.insert(
        (SECTION_ID, OrderedFloat(SAMPLE_TIME)),
        TimeSample {
            section_id: SECTION_ID,
            time: OrderedFloat(SAMPLE_TIME),
            value: FieldValue::Inline(RowKind::TimeSample.seeded_inline_value()),
            version: 1,
        },
    );
    state
}

/// Decimal digit count of `n`, so [`WIDE_FIELD_NAME_WIDTH`] can track a
/// [`CHUNK_SPAN_FIELD_COUNT`] that is now derived rather than literal.
const fn decimal_width(mut n: usize) -> usize {
    let mut width = 1;
    while n >= 10 {
        n /= 10;
        width += 1;
    }
    width
}

/// The zero-padding width [`wide_field_name`] uses: exactly wide enough for the
/// greatest index in the fixture.
///
/// **Derived, not literal, and that is load-bearing.** A hard-coded width of 4
/// is correct only while `CHUNK_SPAN_FIELD_COUNT` stays under 10 000. Since
/// that count now follows `KEYSPACE_PROBE_CHUNK_SIZE`, a retune of the chunk
/// size to 5 000 would produce names of two different lengths (`f9999` and
/// `f10099`), and `"f10099" < "f9999"` in string order — so the highest-indexed
/// field would no longer be the last-probed one and the last-chunk test's
/// staging would silently move into an earlier chunk.
const WIDE_FIELD_NAME_WIDTH: usize = decimal_width(CHUNK_SPAN_FIELD_COUNT - 1);

/// The name of the `i`th field in [`wide_state`].
///
/// **Zero-padded to a fixed width on purpose.** `create_probe_keys` sorts
/// fields by `(section_id, key_name.as_str())` — a STRING comparison — so an
/// unpadded `f9` would sort after `f1099` and "the highest-numbered field" and
/// "the last-probed field" would be different keys. With every name the same
/// length, lexicographic order is numeric order and `wide_field_name(N - 1)` is
/// provably the last field key the probe walks. The last-chunk test below rests
/// entirely on that.
fn wide_field_name(i: usize) -> String {
    format!("f{i:0width$}", width = WIDE_FIELD_NAME_WIDTH)
}

/// An `ObjectState` whose probe key set spans several
/// [`KEYSPACE_PROBE_CHUNK_SIZE`] chunks: one section under the root and
/// [`CHUNK_SPAN_FIELD_COUNT`] one-byte inline fields under that section.
///
/// Values are one byte each so the fixture costs a `MULTI` of small `HSET`s
/// rather than a megabyte of payload — what is under test is the *arity* of the
/// pre-`MULTI` probe, not the size of anything.
fn wide_state() -> ObjectState {
    let mut state = empty_state();
    state.sections.insert(
        SECTION_ID,
        Section {
            section_id: SECTION_ID,
            parent_id: 0,
            name: "World".to_owned(),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        },
    );
    state.children.insert(0, vec![SECTION_ID]);
    for i in 0..CHUNK_SPAN_FIELD_COUNT {
        let key_name = wide_field_name(i);
        state.fields.insert(
            (SECTION_ID, key_name.clone()),
            Field {
                section_id: SECTION_ID,
                key_name,
                value: FieldValue::Inline(vec![0x01]),
                set_order: 1,
                authoring_set_order: 1,
                version: 1,
            },
        );
    }
    state
}

/// How many entries `create_probe_keys` yields for [`wide_state`], derived the
/// way that function builds them (`src/storage/layer.rs`), in its order:
///
/// | # | Key | Why |
/// |---|---|---|
/// | 1 | `obj:hdr` | pushed unconditionally, first |
/// | 2 | `obj:sec:1` | the one section |
/// | 3 | `obj:secs` | the shared sections index, deduped to its first offer |
/// | 4 | `obj:chs` | the children parent-id index — present because the root has a child |
/// | 5 | `obj:fld:1:f0000` | the first field in `(section_id, key_name)` order |
/// | 6 | `obj:flds` | the shared fields index, deduped to its first offer |
/// | 7.. | `obj:fld:1:f0001` … | the remaining fields, ascending |
///
/// No time samples, so nothing follows the fields. That is what puts the
/// highest-named field key LAST in the whole list, and therefore in the last
/// chunk.
fn wide_state_probe_key_count() -> usize {
    // header + section hash + sections index + children index + fields index,
    // plus one hash per field.
    5 + CHUNK_SPAN_FIELD_COUNT
}

/// Guard the arithmetic above against a fixture that stopped spanning waves.
///
/// The unit that matters is the WAVE, not the chunk: `probe_keyspace_types`
/// loops over `chunks.chunks_mut(KEYSPACE_PROBE_CONCURRENCY)`, so a key set of
/// four chunks or fewer runs the outer loop exactly once and its second
/// iteration is covered by nothing. Asserting only "more than one chunk" would
/// have let that through — the fixture would still be multi-chunk, still be
/// green, and still never issue a second wave.
///
/// With [`CHUNK_SPAN_FIELD_COUNT`] derived from both real constants this holds
/// by construction at any tuning, which is exactly why the derivation is worth
/// keeping — but the assertion stays, cheaply, because the two things it still
/// catches are the two ways that construction gets broken: someone replacing
/// the derived expression with a literal (the failure mode the doc-comment
/// above argues against), and [`wide_state_probe_key_count`] drifting out of
/// step with what `create_probe_keys` actually emits.
///
/// It also carries the last-chunk test's staging claim as arithmetic rather
/// than as prose: `> CONCURRENCY × CHUNK` keys means the LAST key sits at an
/// index `>= CONCURRENCY × CHUNK`, i.e. in the second wave or beyond. Combined
/// with the ordering derivation on that test — the highest-named field is the
/// last entry `create_probe_keys` emits — that is what makes the plant provably
/// out of reach of the first wave, never mind the first `EVALSHA`.
fn assert_fixture_spans_multiple_waves() {
    let keys = wide_state_probe_key_count();
    let chunks = keys.div_ceil(KEYSPACE_PROBE_CHUNK_SIZE);
    assert!(
        keys > KEYSPACE_PROBE_CONCURRENCY * KEYSPACE_PROBE_CHUNK_SIZE,
        "the fixture must span more than one probe WAVE — {keys} keys at a chunk size of \
         {KEYSPACE_PROBE_CHUNK_SIZE} is {chunks} chunk(s), and at a concurrency of \
         {KEYSPACE_PROBE_CONCURRENCY} that is {} wave(s). At one wave the outer \
         `chunks.chunks_mut(..)` loop runs a single iteration, so a regression that handled only \
         the first wave — leaving every key past {} unguarded — would pass both tests below",
        chunks.div_ceil(KEYSPACE_PROBE_CONCURRENCY),
        KEYSPACE_PROBE_CONCURRENCY * KEYSPACE_PROBE_CHUNK_SIZE
    );
}

/// `HGETALL` on a value row.
async fn raw_row(tv: &TestValkey, key: &str) -> HashMap<String, String> {
    tv.client()
        .pool
        .hgetall(key)
        .await
        .expect("HGETALL on a value row must succeed")
}

/// `live_edit_keyspace_type_error_total{backend, site}`.
///
/// Panics on an absent series rather than defaulting to zero: every seeded cell
/// exists on a fresh registry, so absence means the store never received this
/// metrics handle and every count read from it would be vacuous.
fn keyspace_type_errors(metrics: &Metrics, backend: &str, site: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_keyspace_type_error_total",
        &[("backend", backend), ("site", site)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_keyspace_type_error_total{{backend=\"{backend}\", site=\"{site}\"}} is \
             absent from the exposition. Every reachable cell is zero-seeded in `Metrics::new`, \
             so absence means this store holds no metrics handle — the counter assertions below \
             could not fail for any reason"
        )
    }) as u64
}

/// `MakeWriter` capturing every byte into a shared buffer.
///
/// Duplicated from the per-module helpers in `b_ji8_tests.rs` /
/// `txn_checkout_test.rs` (themselves mirroring
/// `src/grpc/auth_interceptor.rs::tests::CapturingWriter`) — the established
/// project pattern is to repeat these few lines rather than hoist them into
/// `tests/common`.
#[derive(Clone)]
struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

impl CapturingWriter {
    fn new() -> (Self, Arc<Mutex<Vec<u8>>>) {
        let buf = Arc::new(Mutex::new(Vec::new()));
        (Self(buf.clone()), buf)
    }
}

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

// ═════════════════════════════════════════════════════════════════════════════
// Sub-task 1 — children-list read classification (read side)
// ═════════════════════════════════════════════════════════════════════════════

/// b.zvd: a wrong-typed children list makes the snapshot read fail closed and
/// name the offending parent.
///
/// The commit-path form of this scenario belongs to
/// `b_459_tests::txn_wrongtype_children_list_key_is_caught_by_the_snapshot_read_not_the_probe`;
/// what is asserted here is the read consumers' half, which no b.459 test
/// reaches. Above the atomic-read limits, catch-up and the checkpoint
/// assembler read a layer through the fenced fan-out
/// (`read_object_snapshot_fenced`, here through `b_atj_tests::fenced_read`),
/// which issues the per-parent `LRANGE` this defect lived in; pre-b.zvd the
/// loader those consumers then used received a snapshot whose `children` map
/// silently omitted the parent. (At or below the limits the read is one
/// `EVALSHA` and has no per-list read to fail on its own.)
///
/// Staged on the seeded layer — one leaf section under the root — so the
/// fan-out reads exactly one list, the root's: `create_layer` marks the object,
/// a marked object's read takes its lists only from `obj:chs`, and `obj:chs`
/// names the root alone (the leaf's list is empty). With a second list read,
/// `try_join_all` polls the fan-out in construction order and the seam-free
/// `LRANGE`s race, so the refusal could name either parent and the diagnostic
/// could not be pinned; with none (a layer with no sections leaves `obj:chs`
/// empty), the plant is never read and the read succeeds.
///
/// Three separable claims:
///
/// 1. **It is an error at all** — not an `Ok` carrying an empty children map.
/// 2. **It is `Internal`, not `Aborted`** — a `WRONGTYPE` is permanent, and
///    advertising a retry would spin a caller against a condition only an
///    operator can clear.
/// 3. **It names the parent** — the one diagnostic the static sibling labels
///    cannot carry, and what sends an operator to the offending key. The
///    `WRONGTYPE_KEY:` tag must NOT appear: that tag is minted only by
///    `scripts/keyspace_type_probe.lua`, and seeing it here would mean the
///    commit path's probe answered rather than this read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_read_of_a_wrongtype_children_list_is_a_loud_internal_naming_the_parent() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv, "zvd-ch-wrongtype").await;
    let oid = ls
        .create_layer(
            &session_id,
            "zvd-ch-wrongtype",
            "b-zvd-user",
            &seeded_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let indexed_parents: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().obj_children_index(&oid.to_string()))
        .await
        .expect("SMEMBERS obj:chs must succeed");
    assert_eq!(
        indexed_parents,
        vec!["0".to_owned()],
        "fixture precondition: `obj:chs` must name the root alone, so the root's list is the one \
         list the read issues"
    );

    let list_key = tv.key_schema().obj_children(&oid.to_string(), "0");
    let _: () = tv
        .client()
        .pool
        .set(
            &list_key,
            "planted-by-an-out-of-band-writer",
            None,
            None,
            false,
        )
        .await
        .expect("planting a string on the children list must succeed");

    // `AtomicSnapshot` is not `Debug`, so the `Ok` arm is spelled out rather
    // than reached through `expect_err`.
    let err = match fenced_read(tv.client(), tv.key_schema(), oid).await {
        Err(e) => e,
        Ok(snapshot) => panic!(
            "a wrong-typed children list must fail the snapshot read closed. This `Ok` is the \
             b.zvd defect: every consumer of this read — catch-up and the checkpoint assembler, \
             above the atomic-read limits — receives a snapshot silently missing this parent's \
             children (got children = {:?})",
            snapshot.state.children
        ),
    };

    let ServiceError::Internal(msg) = &err else {
        panic!(
            "a `WRONGTYPE` is permanent — no retry can retype the key — so the refusal must be a \
             non-retriable `Internal`; got {err:?}"
        );
    };
    assert!(
        msg.contains("WRONGTYPE"),
        "the refusal must carry Valkey's own type error so an operator can tell a wrong-typed \
         key from a transport blip; got {msg:?}"
    );
    assert!(
        msg.contains("LRANGE children list (parent 0)"),
        "the refusal must name the offending parent — the sibling snapshot reads carry static \
         labels, so the parent id is the only thing that routes an operator to the key; got \
         {msg:?}"
    );
    assert!(
        !msg.contains("WRONGTYPE_KEY:"),
        "the `WRONGTYPE_KEY:` tag is minted only by the keyspace type probe. Seeing it on a bare \
         snapshot read would mean the message came from somewhere else entirely; got {msg:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Sub-task 2 — one value shape per row
// ═════════════════════════════════════════════════════════════════════════════

/// Which of the two value-row families a parameterized test is staged on.
///
/// Fields and time samples share `encode_value_fields` / `decode_field_value`
/// and therefore share the b.zvd hazard — but they share no *call site*. Every
/// place b.zvd touched has one arm per family and each arm can be deleted
/// without the other noticing: two `execute_mutation` arms, two
/// `write_object` loops (`src/storage/layer.rs` — the fields loop's `HDEL` and
/// the time-samples loop's), two `update_layer_fast.lua` apply branches, and
/// two `ValueShapeConflictTally` instances with two `emit` calls
/// (`src/storage/diff_read.rs`). Driving both families off one enum is what
/// stops the pair drifting: a new claim can only be added to both at once.
#[derive(Clone, Copy, Debug)]
enum RowKind {
    Field,
    TimeSample,
}

impl RowKind {
    /// The other family — used to assert an aggregated warn line names exactly
    /// one of them.
    fn other(self) -> Self {
        match self {
            RowKind::Field => RowKind::TimeSample,
            RowKind::TimeSample => RowKind::Field,
        }
    }

    /// The `row_kind` value `ValueShapeConflictTally::emit` labels its warn
    /// with. Doubles as a test-name / session-name discriminator.
    fn log_label(self) -> &'static str {
        match self {
            RowKind::Field => "field",
            RowKind::TimeSample => "time_sample",
        }
    }

    /// This family's value-row key under `object_id`.
    fn row_key(self, tv: &TestValkey, oid_str: &str) -> String {
        let sid = SECTION_ID.to_string();
        match self {
            RowKind::Field => tv.key_schema().obj_field(oid_str, &sid, FIELD_NAME),
            RowKind::TimeSample => tv.key_schema().obj_time_sample(oid_str, &sid, SAMPLE_TIME),
        }
    }

    /// The non-value columns this family's writer owns, which the value-shape
    /// clear must leave standing. Deliberately different per family: a field
    /// row carries the ordering columns, a time-sample row carries only
    /// `version` (`time_sample_hash_fields`).
    fn non_value_columns(self) -> &'static [&'static str] {
        match self {
            RowKind::Field => &["set_order", "authoring_set_order", "version"],
            RowKind::TimeSample => &["version"],
        }
    }

    /// A whole stale-inline row for this family: the `value` column plus every
    /// non-value column, so what survives the materialization distinguishes
    /// "the stale half was cleared" from "the row was flattened".
    fn stale_inline_row(self) -> Vec<(&'static str, &'static str)> {
        let mut columns = vec![("value", "c3RhbGUtaW5saW5l")];
        columns.extend(self.non_value_columns().iter().map(|c| (*c, "9")));
        columns
    }

    /// The inline bytes [`seeded_state`] gives this family's row.
    fn seeded_inline_value(self) -> Vec<u8> {
        match self {
            RowKind::Field => vec![0xFF, 0x00],
            RowKind::TimeSample => vec![0xAA],
        }
    }

    /// The `exemplar_name` the tally reports for this family's row —
    /// `key_name` for a field, `t={time}` for a time sample.
    fn exemplar_name(self) -> String {
        match self {
            RowKind::Field => FIELD_NAME.to_owned(),
            RowKind::TimeSample => format!("t={SAMPLE_TIME}"),
        }
    }

    /// Add a SECOND row of this family to `state`, so the dual-shape test can
    /// forge two offenders and pin the aggregation.
    ///
    /// See [`SECOND_ROW_VERSION`] for why it is versioned above the seeded row
    /// rather than beside it.
    fn add_second_row(self, state: &mut ObjectState) {
        let value = FieldValue::Inline(vec![0x5A]);
        match self {
            RowKind::Field => {
                state.fields.insert(
                    (SECTION_ID, SECOND_FIELD_NAME.to_owned()),
                    Field {
                        section_id: SECTION_ID,
                        key_name: SECOND_FIELD_NAME.to_owned(),
                        value,
                        set_order: 2,
                        authoring_set_order: 2,
                        version: SECOND_ROW_VERSION,
                    },
                );
            }
            RowKind::TimeSample => {
                state.time_samples.insert(
                    (SECTION_ID, OrderedFloat(SECOND_SAMPLE_TIME)),
                    TimeSample {
                        section_id: SECTION_ID,
                        time: OrderedFloat(SECOND_SAMPLE_TIME),
                        value,
                        version: SECOND_ROW_VERSION,
                    },
                );
            }
        }
    }

    /// The key of the row [`Self::add_second_row`] adds.
    fn second_row_key(self, tv: &TestValkey, oid_str: &str) -> String {
        let sid = SECTION_ID.to_string();
        match self {
            RowKind::Field => tv.key_schema().obj_field(oid_str, &sid, SECOND_FIELD_NAME),
            RowKind::TimeSample => {
                tv.key_schema()
                    .obj_time_sample(oid_str, &sid, SECOND_SAMPLE_TIME)
            }
        }
    }

    /// Replace this family's row in `state` with an External value.
    fn set_external(self, state: &mut ObjectState, hash: [u8; 32], size: u64) {
        let value = FieldValue::External {
            hash,
            size,
            ext_value_index: 1,
        };
        match self {
            RowKind::Field => {
                state.fields.insert(
                    (SECTION_ID, FIELD_NAME.to_owned()),
                    Field {
                        section_id: SECTION_ID,
                        key_name: FIELD_NAME.to_owned(),
                        value,
                        set_order: 1,
                        authoring_set_order: 1,
                        version: 1,
                    },
                );
            }
            RowKind::TimeSample => {
                state.time_samples.insert(
                    (SECTION_ID, OrderedFloat(SAMPLE_TIME)),
                    TimeSample {
                        section_id: SECTION_ID,
                        time: OrderedFloat(SAMPLE_TIME),
                        value,
                        version: 1,
                    },
                );
            }
        }
    }

    /// This family's decoded value, read back through the same reader the
    /// subscribe path uses.
    fn snapshot_value(self, snapshot: &ObjectSnapshot) -> FieldValue {
        match self {
            RowKind::Field => snapshot
                .fields
                .iter()
                .find(|f| f.key_name == FIELD_NAME)
                .unwrap_or_else(|| panic!("the `{FIELD_NAME}` field must be in the snapshot"))
                .value
                .clone(),
            RowKind::TimeSample => snapshot
                .time_samples
                .iter()
                .find(|s| s.time.into_inner() == SAMPLE_TIME)
                .unwrap_or_else(|| {
                    panic!("the t={SAMPLE_TIME} time sample must be in the snapshot")
                })
                .value
                .clone(),
        }
    }
}

/// A value row must carry `value` XOR `hash` + `size`, plus every non-value
/// column it owns.
///
/// The XOR is the whole invariant: a row carrying both decodes as the inline
/// bytes forever (`decode_field_value` checks `value` first) whichever column is
/// actually current, and the shadowed `hash` is invisible to
/// `schedule_old_object_ext_gc`, so the blob's refcount is never released.
///
/// `required_columns` is asserted alongside because the delete set
/// (`VALUE_HFIELDS`) is a strict SUPERSET of what the following `HSET` writes:
/// clearing one column too many, or queueing the `HDEL` after the `HSET`, would
/// erase part of the row rather than just its stale half.
pub(crate) fn assert_exactly_one_value_shape(
    row: &HashMap<String, String>,
    expect_external: bool,
    required_columns: &[&str],
    ctx: &str,
) {
    let has_inline = row.contains_key("value");
    let has_external = row.contains_key("hash") || row.contains_key("size");
    assert!(
        has_inline != has_external,
        "{ctx}: a value row must hold exactly ONE shape — `value` for an inline value, \
         `hash` + `size` for an external one. Both present means the writer merged into a stale \
         column (the row will decode as the inline bytes forever and the external blob's \
         refcount can never be released); neither present means the write erased the value \
         itself. Got {row:?}"
    );
    assert_eq!(
        has_external, expect_external,
        "{ctx}: the surviving shape must be the one just written. The wrong shape here means the \
         `HDEL` and the `HSET` are queued in the wrong order — the delete set is a strict \
         superset of what the write puts back. Got {row:?}"
    );
    if expect_external {
        assert!(
            row.contains_key("hash") && row.contains_key("size"),
            "{ctx}: the external shape is `hash` AND `size`; half of it is not a decodable row. \
             Got {row:?}"
        );
    }
    for column in required_columns {
        assert!(
            row.contains_key(*column),
            "{ctx}: the non-value column `{column}` must survive the value-shape clear. \
             `VALUE_HFIELDS` deliberately excludes it — clearing `version` in particular would \
             open a window where a concurrent reader sees a row with no version. Got {row:?}"
        );
    }
}

/// Prove which commit backend the delta about to be sent will actually run on,
/// rather than assuming it.
///
/// `update_layer_once`'s dispatch is "the backend is `Lua` or
/// `LuaWatchStructural`, and `is_fast_path(&delta)`" (`update_layer_route`,
/// `src/storage/layer.rs`), and the delta that reaches it is the one handed to
/// `update_layer` — the retry loop clones the value, nothing rewrites it — so
/// evaluating `is_fast_path` here evaluates the dispatch's own predicate on
/// the dispatch's own input. `with_update_layer_backend` pins the first
/// conjunct; this pins the second, which is the one the `_lua` tests silently
/// depend on: if a plain field / time-sample delta ever became
/// fast-path-ineligible they would quietly re-run the TXN path under a `_lua`
/// name, and `scripts/update_layer_fast.lua`'s `HDEL` would lose every test it
/// has while the suite stayed green.
///
/// Asserted on BOTH backends because it carries a claim for each: under `Lua`
/// that the fast path really ran, and under `Txn` that the knob is doing work —
/// forcing a delta that WOULD have gone fast down the slow path is the only
/// reason the `_txn` variant is not a duplicate of its sibling.
///
/// There is no metric to read this off instead. Both `backend`-labelled series
/// in `src/observability/metrics.rs` — `keyspace_type_error_total` and
/// `stream_id_collision_total` — count a *refusal*, so both are zero on a
/// healthy commit and neither can say which path produced one.
fn assert_fast_path_eligible(delta: &oodf_ffi::OwnedDelta, ctx: &str) {
    assert!(
        is_fast_path(&Some(delta.clone())),
        "{ctx}: this delta must be fast-path ELIGIBLE, on both backends. Under `Lua` an \
         ineligible delta silently falls through to the WATCH/MULTI/EXEC path, so the `_lua` \
         test would be re-testing its `_txn` sibling and the Lua script's value-shape `HDEL` \
         would be covered by nothing; under `Txn` an ineligible delta would have taken the slow \
         path anyway, so the backend knob — and with it the whole point of the two-variant \
         split — would be a no-op"
    );
}

/// Both value shapes, both rewrite directions, on a FIELD — asserted through
/// the same reader the subscribe path uses AND on the raw row.
///
/// Shared body for the two backends. `LIVE_EDIT_UPDATE_LAYER_BACKEND` selects
/// between two independent implementations of the same write —
/// `scripts/update_layer_fast.lua`'s step-6 apply pass and `execute_mutation`'s
/// `SetField` arm — and the pre-b.zvd defect was present in both, so a fix in
/// one is not a fix. Kept as two `#[tokio::test]` wrappers rather than a loop so
/// nextest isolates them and a failure names the backend.
///
/// Three writes, because the staleness is symmetric: inline→external strands
/// `value`, external→inline strands `hash` + `size`, and no reader-side
/// preference is correct in both directions.
async fn field_value_shape_rewrites_leave_exactly_one_shape(backend: UpdateLayerBackend) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv).with_update_layer_backend(backend);
    let session_id = create_session(&tv, "zvd-field-shape").await;
    let oid = ls
        .create_layer(
            &session_id,
            "zvd-field-shape",
            "b-zvd-user",
            &empty_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let oid_str = oid.to_string();
    let field_key = RowKind::Field.row_key(&tv, &oid_str);
    let non_value_columns = RowKind::Field.non_value_columns();

    let update = |delta: oodf_ffi::OwnedDelta| {
        assert_fast_path_eligible(&delta, "the field value-shape rewrite");
        ls.update_layer_relayed(
            &session_id,
            "zvd-field-shape",
            oid,
            Some(delta),
            "zvd-writer",
            1_000,
        )
    };

    // ── Write 1: inline ─────────────────────────────────────────────────────
    update(delta_with_set_field_inline(
        SECTION_ID,
        FIELD_NAME,
        vec![0x11],
        0,
    ))
    .await
    .expect("the inline field commit must land");
    assert_exactly_one_value_shape(
        &raw_row(&tv, &field_key).await,
        false,
        non_value_columns,
        "after the initial inline write",
    );

    // ── Write 2: inline → external ──────────────────────────────────────────
    update(delta_with_set_field_external(
        SECTION_ID,
        FIELD_NAME,
        ext_hash(0xAB),
        1_024,
        1,
        1,
    ))
    .await
    .expect("the external field commit must land");

    let snapshot = read_atomic(tv.client(), tv.key_schema(), oid).await.state;
    let field = snapshot
        .fields
        .iter()
        .find(|f| f.key_name == FIELD_NAME)
        .expect("the field must be in the snapshot");
    match &field.value {
        FieldValue::External { hash, size, .. } => {
            assert_eq!(
                hash.as_slice(),
                ext_hash(0xAB).as_slice(),
                "the reader must decode the NEW external hash. Decoding the old inline bytes is \
                 the b.zvd defect: `HSET` merged, the stale `value` column survived, and \
                 `decode_field_value` checks `value` first"
            );
            assert_eq!(*size, 1_024, "the external size must round-trip");
        }
        other => panic!(
            "an inline→external rewrite must read back as External. `Inline` here means the row \
             still carries the stale `value` column and every subsequent read of this field — \
             catch-up, subscribe, checkpoint — serves the OLD payload forever; got {other:?}"
        ),
    }
    let row = raw_row(&tv, &field_key).await;
    assert_exactly_one_value_shape(&row, true, non_value_columns, "after inline→external");
    assert_eq!(
        row.get("hash").map(String::as_str),
        Some(ext_hash_hex(0xAB).as_str()),
        "the surviving `hash` column must be the newly written one"
    );

    // ── Write 3: external → inline ──────────────────────────────────────────
    update(delta_with_set_field_inline(
        SECTION_ID,
        FIELD_NAME,
        vec![0x22, 0x33],
        2,
    ))
    .await
    .expect("the second inline field commit must land");

    let snapshot = read_atomic(tv.client(), tv.key_schema(), oid).await.state;
    let field = snapshot
        .fields
        .iter()
        .find(|f| f.key_name == FIELD_NAME)
        .expect("the field must be in the snapshot");
    assert_eq!(
        field.value,
        FieldValue::Inline(vec![0x22, 0x33]),
        "the reverse rewrite must strand nothing either — a surviving `hash` + `size` pair is \
         invisible to this reader but keeps an external blob's refcount pinned forever"
    );
    assert_exactly_one_value_shape(
        &raw_row(&tv, &field_key).await,
        false,
        non_value_columns,
        "after external→inline",
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_field_value_shape_rewrites_leave_exactly_one_shape() {
    field_value_shape_rewrites_leave_exactly_one_shape(UpdateLayerBackend::Lua).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_field_value_shape_rewrites_leave_exactly_one_shape() {
    field_value_shape_rewrites_leave_exactly_one_shape(UpdateLayerBackend::Txn).await;
}

/// The same claim on a TIME SAMPLE, on both backends.
///
/// Worth its own family rather than a parameter of the field one: time samples
/// reach storage through a different `StorageMutation` variant, a different Lua
/// `ST` arm and a different `write_object` loop, and they carry a different
/// non-value column set (`version` only — no `set_order`). One arm can be fixed
/// without the other.
async fn time_sample_value_shape_rewrites_leave_exactly_one_shape(backend: UpdateLayerBackend) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv).with_update_layer_backend(backend);
    let session_id = create_session(&tv, "zvd-ts-shape").await;
    let oid = ls
        .create_layer(
            &session_id,
            "zvd-ts-shape",
            "b-zvd-user",
            &empty_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let oid_str = oid.to_string();
    let ts_key = RowKind::TimeSample.row_key(&tv, &oid_str);
    let non_value_columns = RowKind::TimeSample.non_value_columns();

    let update = |delta: oodf_ffi::OwnedDelta| {
        assert_fast_path_eligible(&delta, "the time-sample value-shape rewrite");
        ls.update_layer_relayed(
            &session_id,
            "zvd-ts-shape",
            oid,
            Some(delta),
            "zvd-writer",
            1_000,
        )
    };

    update(delta_with_time_sample_inline(
        SECTION_ID,
        SAMPLE_TIME,
        vec![0x44],
    ))
    .await
    .expect("the inline time-sample commit must land");
    assert_exactly_one_value_shape(
        &raw_row(&tv, &ts_key).await,
        false,
        non_value_columns,
        "after the initial inline write",
    );

    update(delta_with_time_sample_external(
        SECTION_ID,
        SAMPLE_TIME,
        ext_hash(0xCD),
        2_048,
        1,
    ))
    .await
    .expect("the external time-sample commit must land");

    let snapshot = read_atomic(tv.client(), tv.key_schema(), oid).await.state;
    let sample = snapshot
        .time_samples
        .iter()
        .find(|s| s.time.into_inner() == SAMPLE_TIME)
        .expect("the time sample must be in the snapshot");
    match &sample.value {
        FieldValue::External { hash, size, .. } => {
            assert_eq!(
                hash.as_slice(),
                ext_hash(0xCD).as_slice(),
                "the reader must decode the NEW external hash"
            );
            assert_eq!(*size, 2_048, "the external size must round-trip");
        }
        other => panic!(
            "an inline→external rewrite of a time sample must read back as External — time \
             samples share `encode_value_fields` / `decode_field_value` with fields and inherit \
             the identical stale-column hazard; got {other:?}"
        ),
    }
    assert_exactly_one_value_shape(
        &raw_row(&tv, &ts_key).await,
        true,
        non_value_columns,
        "after inline→external",
    );

    update(delta_with_time_sample_inline(
        SECTION_ID,
        SAMPLE_TIME,
        vec![0x55, 0x66],
    ))
    .await
    .expect("the second inline time-sample commit must land");

    let snapshot = read_atomic(tv.client(), tv.key_schema(), oid).await.state;
    let sample = snapshot
        .time_samples
        .iter()
        .find(|s| s.time.into_inner() == SAMPLE_TIME)
        .expect("the time sample must be in the snapshot");
    assert_eq!(
        sample.value,
        FieldValue::Inline(vec![0x55, 0x66]),
        "the reverse rewrite must strand no `hash` + `size` pair"
    );
    assert_exactly_one_value_shape(
        &raw_row(&tv, &ts_key).await,
        false,
        non_value_columns,
        "after external→inline",
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_time_sample_value_shape_rewrites_leave_exactly_one_shape() {
    time_sample_value_shape_rewrites_leave_exactly_one_shape(UpdateLayerBackend::Lua).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_time_sample_value_shape_rewrites_leave_exactly_one_shape() {
    time_sample_value_shape_rewrites_leave_exactly_one_shape(UpdateLayerBackend::Txn).await;
}

/// `write_object`'s own value-shape clear, staged at the `object_id` the next
/// `create_layer` will mint.
///
/// The two `update_layer` families above cover the commit paths; this covers the
/// third and fourth write sites, which no `update_layer` test can reach. The
/// staging is deterministic because `TestValkey` gives every test its own key
/// prefix, so `le:counter:object_id` is this test's alone and the next
/// allocation is exactly `INCR`'s successor.
///
/// The planted row is a HASH holding a stale `value` — deliberately *not* a
/// wrong-typed key. The pre-`MULTI` probe passes it (a hash is what the commit
/// wants), so what is under test is the `HDEL` inside the `MULTI` and nothing
/// else. Without it the restore path's `HSET` merges into the survivor and the
/// restored external field reads back as the pre-loss inline bytes.
/// Parameterized over [`RowKind`] because `write_object` clears the two
/// families in two separate loops, each with its own `HDEL` line
/// (`src/storage/layer.rs` — "HDEL field value shape failed" and "HDEL time
/// sample value shape failed"). Either line can be deleted with the other left
/// standing, so a single-family test would let half the sub-task regress green.
async fn create_layer_over_a_stale_inline_value_column_writes_only_the_external_shape(
    kind: RowKind,
) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let label = kind.log_label();
    let layer_name = format!("zvd-wo-shape-{label}");
    let session_id = create_session(&tv, &layer_name).await;

    // Establish the counter, then read it: the next `create_layer` mints
    // exactly `counter + 1`.
    ls.create_layer(
        &session_id,
        &format!("{layer_name}-seed"),
        "b-zvd-user",
        &empty_state(),
        mock_ext_store(),
    )
    .await
    .expect("the seed create_layer must succeed");
    let counter: u64 = tv
        .client()
        .pool
        .get::<String, _>(tv.key_schema().counter_object_id())
        .await
        .expect("the object-id counter must exist after one create_layer")
        .parse()
        .expect("the object-id counter must be numeric");
    let next_oid = (counter + 1).to_string();

    // Plant a survivor row carrying the OLD inline shape at the value key the
    // materialization is about to write.
    let row_key = kind.row_key(&tv, &next_oid);
    let _: () = tv
        .client()
        .pool
        .hset(&row_key, kind.stale_inline_row())
        .await
        .unwrap_or_else(|e| panic!("planting the survivor {label} row must succeed: {e}"));

    let mut state = seeded_state();
    kind.set_external(&mut state, [0xEF; 32], 4_096);

    let oid = ls
        .create_layer(
            &session_id,
            &layer_name,
            "b-zvd-user",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed over a hash-typed survivor row");
    assert_eq!(
        oid.to_string(),
        next_oid,
        "the staging depends on the minted id being the counter's successor — a different id \
         means the planted row was never on the materialization's path and every assertion \
         below is vacuous"
    );

    let snapshot = read_atomic(tv.client(), tv.key_schema(), oid).await.state;
    let value = kind.snapshot_value(&snapshot);
    assert!(
        matches!(value, FieldValue::External { size: 4_096, .. }),
        "the materialized {label} row must read back as the External value the state named. \
         `Inline` here means `write_object`'s `HSET` merged into the survivor's stale `value` \
         column and the layer serves pre-loss content; got {value:?}"
    );
    let row = raw_row(&tv, &row_key).await;
    // Anti-vacuity: the planted row carries `version = 9` and the materialized
    // one carries `1`. Without this, a `row_key` that drifted away from the one
    // `write_object` builds would leave a CLEAN external row at the real key and
    // every assertion here would pass over a plant nothing ever touched.
    assert_eq!(
        row.get("version").map(String::as_str),
        Some("1"),
        "the materialization must have landed on the PLANTED {label} row — the plant carries \
         `version=9` and the state carries `1`. A `9` here (or an absent column) means this test \
         is reading a key `write_object` never wrote, so the value-shape assertion below would \
         be judging an untouched plant. Got {row:?}"
    );
    assert_exactly_one_value_shape(
        &row,
        true,
        kind.non_value_columns(),
        &format!("after write_object materialized a {label} over a stale inline row"),
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_over_a_stale_inline_field_writes_only_the_external_shape() {
    create_layer_over_a_stale_inline_value_column_writes_only_the_external_shape(RowKind::Field)
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_over_a_stale_inline_time_sample_writes_only_the_external_shape() {
    create_layer_over_a_stale_inline_value_column_writes_only_the_external_shape(
        RowKind::TimeSample,
    )
    .await;
}

/// The reader reports a dual-shape row rather than repairing it — and still
/// decodes.
///
/// Rows written by this service now hold exactly one shape, so the only
/// producers left are rows that predate b.zvd or were written out of band. The
/// reader deliberately keeps preferring `value`: staleness is symmetric and the
/// row carries no ordering information saying which column is newer, so flipping
/// the preference would merely move the corruption to the other rewrite
/// direction. What the reader CAN do is make the coexistence visible, and that
/// visibility is the contract this test pins.
///
/// The `warn` is **aggregated** — one line per snapshot-read pass, not one per
/// row. Staged on the atomic read, the read catch-up and the checkpoint
/// assembler take at these sizes; the fan-outs decode through the same tallies. That is load-bearing rather than cosmetic: the trigger is
/// object-wide and a pass reads the whole object, whose rows no per-delta
/// limit bounds (a layer holds as many rows as its writes made, far more than
/// one delta's `MAX_DELTA_ENTRIES`), so a per-row line would bury the incident
/// it is reporting. `conflicts` plus an
/// exemplar is what keeps one line diagnostic.
///
/// Runs on the default current-thread runtime so the scoped
/// `tracing::subscriber::set_default` — which is thread-local — covers the whole
/// body; a multi-thread runtime could poll the read on another worker and
/// capture nothing.
///
/// Parameterized over [`RowKind`] because the reader tallies the two families
/// independently: two `ValueShapeConflictTally` values and two `emit` calls
/// (`src/storage/diff_read.rs` — `field_shape_conflicts.emit("field", …)` and
/// `ts_shape_conflicts.emit("time_sample", …)`). The time-sample call would
/// otherwise survive deletion with the whole suite green, and no test in
/// `tests/` would assert `row_kind="time_sample"` at all.
async fn snapshot_read_of_a_dual_shape_row_decodes_value_first_and_warns(kind: RowKind) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let label = kind.log_label();
    let layer_name = format!("zvd-dual-shape-{label}");
    let session_id = create_session(&tv, &layer_name).await;
    let mut state = seeded_state();
    kind.add_second_row(&mut state);
    let oid = ls
        .create_layer(
            &session_id,
            &layer_name,
            "b-zvd-user",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let oid_str = oid.to_string();
    let row_key = kind.row_key(&tv, &oid_str);
    let second_row_key = kind.second_row_key(&tv, &oid_str);

    // Forge the pre-b.zvd shape: an external half merged onto a row that still
    // carries its inline `value`. This is exactly what an inline→external
    // rewrite produced before the writer started clearing both shapes.
    //
    // TWO rows of the one family, deliberately. With a single offender an
    // aggregated `emit` and a hypothetical per-row `warn!` inside `observe` are
    // observationally identical — both produce exactly one line carrying
    // `conflicts=1` — so the aggregation the doc-comment above calls
    // load-bearing would not actually be pinned. Two offenders separate them:
    // aggregation is ONE line saying `conflicts=2`, per-row logging is two
    // lines saying `conflicts=1`.
    //
    // Both are the SAME family, and the sibling family is left clean, so its
    // tally stays empty and its `emit` returns without a line. That is what
    // keeps the negative `row_kind` assertion below unambiguous — with both
    // families forged, one line's fields could be read off the other's.
    for (which, key) in [("first", &row_key), ("second", &second_row_key)] {
        let _: () = tv
            .client()
            .pool
            .hset(
                key,
                [("hash", ext_hash_hex(0x77).as_str()), ("size", "512")],
            )
            .await
            .unwrap_or_else(|e| panic!("forging the {which} dual-shape row must succeed: {e}"));
    }

    let (writer, captured) = CapturingWriter::new();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(writer)
        .with_max_level(tracing::Level::WARN)
        // ANSI escapes would otherwise land in the buffer and make the
        // substring matches depend on whether colour was auto-enabled.
        .with_ansi(false)
        .finish();
    let capture = tracing::subscriber::set_default(subscriber);

    // `read_atomic` panics on a failed read; a failure here means the reader
    // refused a dual-shape row it must still DECODE — failing the whole
    // snapshot closed would take a layer offline over a row the service can
    // still serve, and the reader's job is to report, not to refuse.
    let snapshot = read_atomic(tv.client(), tv.key_schema(), oid).await.state;

    // Stop capturing before asserting, so a panic's own output is not swallowed
    // by this subscriber.
    let logged = {
        let buffer = captured
            .lock()
            .expect("the log-capture mutex is never poisoned");
        String::from_utf8_lossy(&buffer).into_owned()
    };
    drop(capture);

    let value = kind.snapshot_value(&snapshot);
    assert_eq!(
        value,
        FieldValue::Inline(kind.seeded_inline_value()),
        "`decode_field_value` checks `value` first and stays that way — no reader-side \
         preference is correct in both rewrite directions, so the reader must not start \
         guessing; got {value:?}"
    );

    let lines = logged.matches("carries both an inline").count();
    assert_eq!(
        lines, 1,
        "the coexistence must raise EXACTLY ONE warn for the whole pass, over TWO offending \
         rows. Zero means the only signal an operator gets that a row is serving possibly-stale \
         content never fires. Two means the reader logs per row rather than per pass — the \
         trigger is object-wide and a pass reads every row of the object, which no per-delta \
         limit bounds, so per-row logging buries the incident it is reporting under its own \
         output. Captured: {logged}"
    );
    assert!(
        logged.contains(&format!("row_kind=\"{label}\"")),
        "the warn must say which row family it found, so an operator knows whether to look at \
         `obj:fld:*` or `obj:ts:*`. Captured: {logged}"
    );
    assert!(
        !logged.contains(&format!("row_kind=\"{}\"", kind.other().log_label())),
        "only the forged family may be reported. A line for the other one means the two tallies \
         are not independent — either they share an instance, or one family's `emit` is being \
         passed the other's label, and an operator would be sent to the wrong key family. \
         Captured: {logged}"
    );
    assert!(
        logged.contains("conflicts=2"),
        "the one aggregated line must carry the COUNT of everything it stands for — both forged \
         rows, not just the one that supplied the exemplar. A `conflicts=1` here means the tally \
         reports the exemplar rather than the total, and one line saying `1` and one saying \
         `100000` call for very different responses. Captured: {logged}"
    );
    assert!(
        logged.contains(&format!("exemplar_section_id={SECTION_ID}")),
        "an aggregated line with no exemplar names no key at all, which leaves the operator \
         with nothing to act on. Captured: {logged}"
    );
    assert!(
        logged.contains(&kind.exemplar_name()),
        "the exemplar must be the FIRST offender the pass walked ({}) and must name the row \
         within its section; without it the section id alone can cover thousands of rows. The \
         second forged row is versioned above this one precisely so the walk order — and so this \
         assertion — is deterministic. Captured: {logged}",
        kind.exemplar_name()
    );
}

#[tokio::test]
async fn snapshot_read_of_a_dual_shape_field_decodes_value_first_and_warns() {
    snapshot_read_of_a_dual_shape_row_decodes_value_first_and_warns(RowKind::Field).await;
}

#[tokio::test]
async fn snapshot_read_of_a_dual_shape_time_sample_decodes_value_first_and_warns() {
    snapshot_read_of_a_dual_shape_row_decodes_value_first_and_warns(RowKind::TimeSample).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Sub-task 3 — write_object's pre-MULTI probe and children replace
// ═════════════════════════════════════════════════════════════════════════════

/// b.zvd: a key `write_object` would write holding the wrong Valkey type refuses
/// the whole `create_layer`, before `MULTI` opens.
///
/// The b.459 argument transposed to the create path: `EXEC` neither aborts on a
/// runtime error nor rolls back its siblings, and fred surfaces only the first
/// error frame, so a wrong-typed key half-commits the materialization — the
/// header, the indices and the stream's `1-0` entry land while the refused
/// `HSET` does not, and the caller gets one `Internal` naming a single key.
///
/// Staged on a section hash rather than the header so the assertion covers the
/// enumerated record keys, not just `create_probe_keys`' first unconditional
/// entry. On this entry point the `object_id` was minted by `INCR` microseconds
/// earlier and should own nothing, which is why the accompanying remediation
/// text points at a rewound counter rather than at an out-of-band writer.
///
/// The counter assertion at the end is not incidental: `allocate_object_id`
/// `INCR`s before any guard runs, so a refused materialization BURNS an object
/// id. That is the current behaviour and it is harmless (ids are cheap and
/// monotone) — pinned here so a future change to the allocation order is a
/// visible decision rather than a silent one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_refuses_whole_when_a_key_it_would_write_holds_the_wrong_type() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let ls = layer_store(&tv).with_metrics(metrics.clone());
    let session_id = create_session(&tv, "zvd-create-probe").await;

    ls.create_layer(
        &session_id,
        "zvd-create-probe-seed",
        "b-zvd-user",
        &empty_state(),
        mock_ext_store(),
    )
    .await
    .expect("the seed create_layer must succeed");
    let counter: u64 = tv
        .client()
        .pool
        .get::<String, _>(tv.key_schema().counter_object_id())
        .await
        .expect("the object-id counter must exist after one create_layer")
        .parse()
        .expect("the object-id counter must be numeric");
    let next_oid = (counter + 1).to_string();

    let section_key = tv
        .key_schema()
        .obj_section(&next_oid, &SECTION_ID.to_string());
    let _: () = tv
        .client()
        .pool
        .set(
            &section_key,
            "planted-by-an-out-of-band-writer",
            None,
            None,
            false,
        )
        .await
        .expect("planting a string on the section hash must succeed");

    let err = ls
        .create_layer(
            &session_id,
            "zvd-create-probe",
            "b-zvd-user",
            &seeded_state(),
            mock_ext_store(),
        )
        .await
        .expect_err(
            "a wrong-typed key the materialization would write must refuse the whole \
             `create_layer`. An `Ok` here means the `WRONGTYPE` was raised inside `EXEC`, where \
             Valkey runs every sibling anyway — the layer materializes minus one section, the \
             `1-0` stream entry publishes it, and nobody learns the object is incomplete",
        );

    let ServiceError::PermanentKeyspaceRefusal { refusal, detail } = &err else {
        panic!(
            "a wrong-typed key is a permanent condition — no retry can retype it — so the refusal \
             must arrive as `PermanentKeyspaceRefusal`, the VARIANT that says so. That is the \
             contract, not the message text: this create-site refusal is the one the checkpoint \
             lane's `write_object_err_to_checkpoint_err` classifies `Permanent`, and it does so \
             by type, so a plain `Internal` here — however it is worded — would fold to \
             `Transient` and re-drive a restore that cannot succeed; got {err:?}"
        );
    };
    assert_eq!(
        *refusal,
        KeyspaceRefusal::WrongType,
        "the kind must name the condition: `WrongType` sends an operator at the key to remove, \
         `StreamIdCollision` would send them at the per-object stream; got {refusal:?}"
    );
    assert!(
        detail.starts_with(CREATE_KEYSPACE_TYPE_ERROR_PREFIX),
        "the refusal must open with {CREATE_KEYSPACE_TYPE_ERROR_PREFIX:?}. The `create_layer` \
         half of that prefix is what tells an operator which of the two probe sites refused, \
         and therefore which remediation applies — the update-path text talks about an \
         out-of-band writer, the create-path text about a rewound object-id counter. `detail` \
         carries the emitter's string verbatim, so this is byte-identical to the pre-variant \
         `Internal`; got {detail:?}"
    );
    assert!(
        detail.contains(&section_key)
            && detail.contains("holds a string")
            && detail.contains("expected hash"),
        "the refusal must name the offending key, what it actually holds and what the commit \
         needed — that triple IS the remediation; got {detail:?}"
    );

    assert_eq!(
        keyspace_type_errors(&metrics, "txn", "create"),
        1,
        "exactly one refusal must tick \
         live_edit_keyspace_type_error_total{{backend=\"txn\", site=\"create\"}} — once per \
         refused materialization, not once per probed key, on the cell whose alert value rests \
         on a zero baseline"
    );
    assert_eq!(
        keyspace_type_errors(&metrics, "txn", "update"),
        0,
        "the update cell must stay at its zero baseline: a create-path refusal filed under \
         `site=\"update\"` would send an operator hunting an `UpdateLayer` that never happened"
    );

    for (label, key) in [
        ("the object header", tv.key_schema().obj_header(&next_oid)),
        (
            "the sections index",
            tv.key_schema().obj_sections_index(&next_oid),
        ),
        ("the live stream", tv.key_schema().layer_stream(counter + 1)),
        (
            "the layer mapping",
            tv.key_schema()
                .layer_mapping(&session_id, "zvd-create-probe"),
        ),
    ] {
        let exists: u64 = tv
            .client()
            .pool
            .exists(&key)
            .await
            .expect("EXISTS must succeed");
        assert_eq!(
            exists, 0,
            "{label} ({key}) must not exist: the probe runs BEFORE `MULTI`, so a refused \
             materialization writes nothing at all and publishes nothing to subscribers"
        );
    }

    let counter_after: u64 = tv
        .client()
        .pool
        .get::<String, _>(tv.key_schema().counter_object_id())
        .await
        .expect("the object-id counter must still exist")
        .parse()
        .expect("the object-id counter must be numeric");
    assert_eq!(
        counter_after,
        counter + 1,
        "the refused materialization still consumed its minted id — `allocate_object_id` `INCR`s \
         before any guard runs. Harmless (ids are cheap and monotone) but pinned, so moving the \
         allocation behind the probe is a visible decision rather than a silent change"
    );
}

/// b.zvd: a create whose probe key set does not fit in one `EVALSHA` still
/// commits, and commits whole.
///
/// `probe_keyspace_types` issues one `EVALSHA` per `KEYSPACE_PROBE_CHUNK_SIZE`
/// keys, in waves of `KEYSPACE_PROBE_CONCURRENCY`: each wave is awaited whole
/// and its results scanned in INDEX order (`join_all` over
/// `chunks.chunks_mut(..)`, deliberately not a completion-ordered
/// `buffer_unordered` / `JoinSet` — `src/storage/layer.rs`). The update path
/// never needed chunking at all — its key set is bounded by one delta — but
/// `create_probe_keys` walks the whole `ObjectState`, so a `CreateLayer` of
/// about 100,000 fields (`MAX_DELTA_ENTRIES`) would put ~200 000 `KEYS` +
/// `ARGV` arguments on a single command, and the DR-restore entry point has no
/// per-call cap at all.
///
/// [`assert_fixture_spans_multiple_waves`] is what makes this a test of the
/// wave LOOP rather than of one wave: the fixture is sized to five chunks, so
/// the outer loop runs twice and its second iteration is reached.
///
/// Nothing here depends on the fan-out's width or on chunks completing in issue
/// order, only on the two properties the split must preserve: every chunk
/// finishes before `MULTI` opens, and the verdict is the same one an un-chunked
/// probe would have reached.
///
/// This is the positive control for that loop, and every claim it makes is one
/// a chunking bug breaks in a different way:
///
/// 1. **`create_layer` succeeds.** A loop that never terminates, or that probes
///    a chunk twice and trips over its own answer, fails here first.
/// 2. **A first-chunk AND a last-chunk field row exist and decode.** The
///    interesting failure is not the probe refusing — it is the probe silently
///    narrowing to its first chunk (or first wave) while `write_object` writes
///    everything, or the reverse. Both boundary rows plus the exact field COUNT
///    catch a chunk that went missing from either end.
/// 3. **The type-error counter never ticks.** N chunks over a clean keyspace is
///    N clean answers; a per-chunk tick would put this counter's zero baseline —
///    the thing its alert rests on — at the mercy of object size.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_with_more_keys_than_one_probe_chunk_succeeds() {
    assert_fixture_spans_multiple_waves();

    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let ls = layer_store(&tv).with_metrics(metrics.clone());
    let session_id = create_session(&tv, "zvd-probe-chunks").await;

    let oid = ls
        .create_layer(
            &session_id,
            "zvd-probe-chunks",
            "b-zvd-user",
            &wide_state(),
            mock_ext_store(),
        )
        .await
        .unwrap_or_else(|e| {
            panic!(
                "a create whose probe key set spans {} chunks must still commit. The probe is \
                 read-only and every chunk completes before `MULTI` opens, so splitting it \
                 changes nothing a caller can observe — an error here means the chunking loop \
                 itself refused; got {e:?}",
                wide_state_probe_key_count().div_ceil(KEYSPACE_PROBE_CHUNK_SIZE)
            )
        });
    let oid_str = oid.to_string();

    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid_str)
        .await
        .expect("reading the created layer's state must succeed");
    assert_eq!(
        snapshot.fields.len(),
        CHUNK_SPAN_FIELD_COUNT,
        "every uploaded field must be materialized. A count short by roughly a chunk is what a \
         probe/write loop that agreed on the first {KEYSPACE_PROBE_CHUNK_SIZE} keys and diverged \
         after them looks like"
    );

    // The two boundary rows: the first field the probe walks (chunk 1) and the
    // last (the final chunk). Read through the raw row rather than the snapshot
    // so a decode is proven against stored bytes, not against a `Vec` index.
    for (label, index) in [
        ("the first-chunk", 0),
        ("the last-chunk", CHUNK_SPAN_FIELD_COUNT - 1),
    ] {
        let name = wide_field_name(index);
        let key = tv
            .key_schema()
            .obj_field(&oid_str, &SECTION_ID.to_string(), &name);
        let row = raw_row(&tv, &key).await;
        assert!(
            !row.is_empty(),
            "{label} field row ({name}) must exist at {key} — an empty row means the \
             materialization stopped short of it"
        );
        assert_exactly_one_value_shape(
            &row,
            false,
            RowKind::Field.non_value_columns(),
            &format!("{label} field row ({name})"),
        );
        let value = snapshot
            .fields
            .iter()
            .find(|f| f.key_name == name)
            .unwrap_or_else(|| panic!("{label} field ({name}) must be in the snapshot"))
            .value
            .clone();
        assert_eq!(
            value,
            FieldValue::Inline(vec![0x01]),
            "{label} field ({name}) must decode to the byte it was uploaded with"
        );
    }

    assert_eq!(
        keyspace_type_errors(&metrics, "txn", "create"),
        0,
        "a clean keyspace is a clean answer however many chunks it took to ask. A tick here \
         would mean the counter reads object SIZE rather than keyspace corruption, and its \
         alert — which pages on any non-zero rate — would fire on ordinary large uploads"
    );
}

/// b.zvd: a wrong-typed key in the LAST probe chunk refuses the whole create,
/// exactly as one in the first would.
///
/// This is the fail-fast wave loop's real regression: `probe_keyspace_types`
/// awaits a wave at a time and scans each wave's results in index order,
/// returning on the first that refuses. A bug that stopped after chunk 1 — or
/// after WAVE 1, or that reported only the last chunk's verdict — would leave
/// the tail of a large object unguarded and half-commit it inside `EXEC`, which
/// is the entire condition b.459 and b.zvd exist to prevent. The plant lands in
/// chunk 4, i.e. the **second wave**, so this is the only test in the file that
/// proves a refusal is still surfaced from past the first
/// `chunks.chunks_mut(..)` iteration.
///
/// One offending key, deliberately, so the assertions do not depend on which
/// chunk within a wave resolves first: with a single refusal there is exactly
/// one verdict to report whatever the scheduling.
///
/// # What this test deliberately does NOT claim
///
/// That a wave-1 refusal prevents wave 2 from being issued. Fail-fast is real —
/// the loop returns before `join_all`ing the next wave — but it is not
/// observable from out here: the probe is read-only, so a wave that ran anyway
/// would leave no trace in the keyspace, in the metric, or in the error. Pinning
/// it would take counting `EVALSHA`s, which needs a seam that does not exist.
/// The claims below are the ones the outside can actually see.
///
/// # Why `wide_field_name(CHUNK_SPAN_FIELD_COUNT - 1)` is in the last chunk
///
/// `create_probe_keys` (`src/storage/layer.rs`) pushes in a fixed order —
/// header, then sections + the sections index, then the children index, then
/// **fields**, then time samples — and `probe_keyspace_types` chunks that list
/// in order and waves those chunks in order, so "last in the list" is "in the
/// last chunk of the last wave". Three facts put this key at the end of the
/// list:
///
/// 1. The fields loop iterates `state.fields` sorted by
///    `(section_id, key_name.as_str())`. All [`wide_state`] fields share one
///    section, so the order is the `key_name` string order.
/// 2. [`wide_field_name`] zero-pads every name to the same width
///    ([`WIDE_FIELD_NAME_WIDTH`], itself derived from the count), so that
///    string order IS numeric order and the highest-indexed name is the
///    greatest.
/// 3. [`wide_state`] carries no time samples, so nothing is pushed after the
///    fields.
///
/// The `assert_fixture_spans_multiple_waves` call is what keeps that derivation
/// load-bearing: without it a shrunk fixture would put this key in wave 1 — or
/// in chunk 1 — and the test would quietly become its own sibling.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_wrongtype_in_the_last_probe_chunk_is_still_refused_whole() {
    assert_fixture_spans_multiple_waves();

    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let ls = layer_store(&tv).with_metrics(metrics.clone());
    let session_id = create_session(&tv, "zvd-probe-chunk-tail").await;

    // Establish the counter, then read it: the next `create_layer` mints
    // exactly `counter + 1`, which is where the plant has to land.
    ls.create_layer(
        &session_id,
        "zvd-probe-chunk-tail-seed",
        "b-zvd-user",
        &empty_state(),
        mock_ext_store(),
    )
    .await
    .expect("the seed create_layer must succeed");
    let counter: u64 = tv
        .client()
        .pool
        .get::<String, _>(tv.key_schema().counter_object_id())
        .await
        .expect("the object-id counter must exist after one create_layer")
        .parse()
        .expect("the object-id counter must be numeric");
    let next_oid = (counter + 1).to_string();

    let last_name = wide_field_name(CHUNK_SPAN_FIELD_COUNT - 1);
    let doomed_key = tv
        .key_schema()
        .obj_field(&next_oid, &SECTION_ID.to_string(), &last_name);
    let planted = "planted-by-an-out-of-band-writer";
    let _: () = tv
        .client()
        .pool
        .set(&doomed_key, planted, None, None, false)
        .await
        .expect("planting a string on the last-chunk field key must succeed");

    let err = ls
        .create_layer(
            &session_id,
            "zvd-probe-chunk-tail",
            "b-zvd-user",
            &wide_state(),
            mock_ext_store(),
        )
        .await
        .expect_err(
            "a wrong-typed key BEYOND the first probe chunk must refuse the whole create. An \
             `Ok` here means the probe stopped at its first chunk: everything past that boundary \
             is unguarded, the `HSET` raises its `WRONGTYPE` inside `EXEC` where Valkey commits \
             every sibling anyway, and the layer publishes its `1-0` entry missing one field",
        );

    let ServiceError::PermanentKeyspaceRefusal { refusal, detail } = &err else {
        panic!(
            "a wrong-typed key is permanent wherever in the key set it sits — chunk position \
             cannot make it retriable, and permanence is carried by the VARIANT. A refusal that \
             arrived as a plain `Internal` because the offending key happened to sit in the last \
             chunk would be classified `Transient` downstream on a property of the batching; got \
             {err:?}"
        );
    };
    assert_eq!(
        *refusal,
        KeyspaceRefusal::WrongType,
        "the kind is a property of the condition, not of where in the key set it was found; got \
         {refusal:?}"
    );
    assert!(
        detail.starts_with(CREATE_KEYSPACE_TYPE_ERROR_PREFIX),
        "a last-chunk refusal must be spelled EXACTLY like a first-chunk one — same \
         {CREATE_KEYSPACE_TYPE_ERROR_PREFIX:?} opening, including the `WRONGTYPE_KEY:` tag the \
         probe script mints. The chunk that answered is deliberately invisible to the caller; \
         got {detail:?}"
    );
    assert!(
        detail.contains(&doomed_key)
            && detail.contains("holds a string")
            && detail.contains("expected hash"),
        "the refusal must name the offending key even when it sat in the last chunk — the \
         script reports the key, never an ordinal, so an operator's remediation does not depend \
         on how the probe was batched; got {detail:?}"
    );

    assert_eq!(
        keyspace_type_errors(&metrics, "txn", "create"),
        1,
        "EXACTLY one tick. Once per refused materialization — not once per wave, not once per \
         chunk probed, not once per probed key. Each wave is awaited whole and scanned in index \
         order, and the scan RETURNS on the first refusal, so the classifier runs once however \
         many chunks were in flight beside the offending one. A {} (chunks) or {} (waves) here \
         would be the batching leaking into a counter whose alert pages on any non-zero rate",
        wide_state_probe_key_count().div_ceil(KEYSPACE_PROBE_CHUNK_SIZE),
        wide_state_probe_key_count()
            .div_ceil(KEYSPACE_PROBE_CHUNK_SIZE)
            .div_ceil(KEYSPACE_PROBE_CONCURRENCY)
    );

    // Nothing written — including the first chunk's own field row, which is the
    // one a probe that refused *after* opening `MULTI` would have left behind.
    let first_chunk_field_key =
        tv.key_schema()
            .obj_field(&next_oid, &SECTION_ID.to_string(), &wide_field_name(0));
    for (label, key) in [
        ("the object header", tv.key_schema().obj_header(&next_oid)),
        (
            "the sections index",
            tv.key_schema().obj_sections_index(&next_oid),
        ),
        (
            "the fields index",
            tv.key_schema().obj_fields_index(&next_oid),
        ),
        ("the first-chunk field row", first_chunk_field_key),
        ("the live stream", tv.key_schema().layer_stream(counter + 1)),
        (
            "the layer mapping",
            tv.key_schema()
                .layer_mapping(&session_id, "zvd-probe-chunk-tail"),
        ),
    ] {
        let exists: u64 = tv
            .client()
            .pool
            .exists(&key)
            .await
            .expect("EXISTS must succeed");
        assert_eq!(
            exists, 0,
            "{label} ({key}) must not exist. Every chunk completes before `MULTI` opens, so a \
             refusal in the LAST chunk still writes nothing at all — a key here means the \
             chunking loop let the commit start before it had finished asking"
        );
    }

    // The plant itself is untouched: the refusal is a read-only verdict, and an
    // operator has to be able to find the value they were told about.
    let survivor: Option<String> = tv
        .client()
        .pool
        .get(&doomed_key)
        .await
        .expect("GET on the planted key must succeed");
    assert_eq!(
        survivor.as_deref(),
        Some(planted),
        "the probe must not have modified the offending key — it is a read-only guard, and the \
         remediation the message asks for is the operator inspecting exactly this value"
    );
}

/// The same guard on the DR restore entry point — the one that matters most.
///
/// `apply_checkpoint` materializes at a KNOWN `object_id`, deliberately over a
/// keyspace that may have survived, and it queues no `XADD` at all, so the b.ph9
/// `EXISTS` stream guard does not run: the type probe is the ONLY pre-`MULTI`
/// keyspace guard there is. (Since b.h4m that materialization is
/// `LayerStore::write_restored_object`, a function of its own; before it, the
/// same property was spelled as `initial_stream_payload = None` on the shared
/// `write_object` both entry points went through.)
///
/// The refusal reaches the caller as `CheckpointError::Permanent`, and that IS
/// pinned. b.f4t made it so (a wrong-typed key is permanent — no retry can
/// retype it), and b.6de sub-task 2 made the dispatch structural: the probe's
/// create arm answers `ServiceError::PermanentKeyspaceRefusal` and
/// `write_object_err_to_checkpoint_err` classifies on that VARIANT rather than
/// on the message text it used to grep for a tag. The message the wrap carries
/// is asserted too, because it is what an operator reads — but the two are now
/// independent claims: a reworded message can no longer silently demote the
/// verdict to `Transient`, and this test would catch it if it did.
///
/// The `entry_point` field is asserted off the log line because it is the only
/// thing that says which of the two callers refused — the metric's `site` label
/// deliberately does not split.
#[tokio::test]
async fn checkpoint_restore_refuses_whole_when_a_key_it_would_write_holds_the_wrong_type() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv, "zvd-restore-probe").await;
    let oid = ls
        .create_layer(
            &session_id,
            "zvd-restore-probe",
            "b-zvd-user",
            &seeded_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;
    crate::common::wipe_object_keys(&tv, oid).await;

    let oid_str = oid.to_string();
    let section_key = tv
        .key_schema()
        .obj_section(&oid_str, &SECTION_ID.to_string());
    let _: () = tv
        .client()
        .pool
        .set(
            &section_key,
            "survived-the-loss-with-the-wrong-type",
            None,
            None,
            false,
        )
        .await
        .expect("planting a string on the surviving section hash must succeed");

    let (writer, captured) = CapturingWriter::new();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(writer)
        .with_max_level(tracing::Level::ERROR)
        .with_ansi(false)
        .finish();
    let capture = tracing::subscriber::set_default(subscriber);

    let err = apply_checkpoint(
        &ls,
        oid,
        ompf,
        snap_seq,
        &meta,
        None,
        &crate::common::test_config(),
    )
    .await
    .expect_err(
        "the restore must refuse whole over a wrong-typed surviving key. An `Ok` here means the \
         `WRONGTYPE` was raised inside `EXEC`: the restore reports success while the object is \
         missing whatever the refused command carried, and the DR path's one job is to not do \
         that",
    );

    let logged = {
        let buffer = captured
            .lock()
            .expect("the log-capture mutex is never poisoned");
        String::from_utf8_lossy(&buffer).into_owned()
    };
    drop(capture);

    // The `CheckpointError` VARIANT is pinned, and so is the message.
    //
    // They are two independent claims since b.6de sub-task 2. The probe's
    // create arm answers `ServiceError::PermanentKeyspaceRefusal`, and
    // `write_object_err_to_checkpoint_err` matches that VARIANT to reach
    // `Permanent` — where it used to grep the message for the script's
    // `WRONGTYPE_KEY:` tag. So the message assertion below no longer doubles as
    // the classification assertion: it proves the operator's locator survived
    // the wrap, and the `Permanent` check proves the recovery hook will stop
    // re-driving a restore no retry can clear (b.f4t sub-task 2).
    //
    // Pinning `Permanent` here rather than leaving the variant open is the
    // point: `Transient` would make the recovery loop retry against a fresh
    // adapter read forever, re-ticking `keyspace_type_error_total` on every
    // pass, for a keyspace an operator has to repair by hand.
    let rendered = err.to_string();
    assert!(
        matches!(err, CheckpointError::Permanent { .. }),
        "a wrong-typed surviving key must reach the recovery hook as \
         `CheckpointError::Permanent`. `Transient` re-drives a restore that cannot succeed on \
         every attempt; got {err:?}"
    );
    assert!(
        rendered.contains(CREATE_KEYSPACE_TYPE_ERROR_PREFIX),
        "the underlying refusal must survive the checkpoint-error wrap intact — it is the only \
         thing naming the key an operator has to remove. The prefix carries the script's \
         `WRONGTYPE_KEY:` tag inside it, so this one check also proves the verdict came from the \
         keyspace type probe rather than from some other restore failure; got {rendered:?}"
    );
    assert!(
        logged.contains(&format!("entry_point=\"{ENTRY_POINT_CHECKPOINT_RESTORE}\"")),
        "the refusal log line must say WHICH create-path caller refused. Both share one metric \
         cell by design, so `entry_point` is the only thing that tells an operator whether to \
         look at a just-minted id (create_layer) or at a surviving keyspace (this path). \
         Captured: {logged}"
    );

    let header_exists: u64 = tv
        .client()
        .pool
        .exists(tv.key_schema().obj_header(&oid_str))
        .await
        .expect("EXISTS must succeed");
    assert_eq!(
        header_exists, 0,
        "the object header must not exist: the probe runs before `MULTI`, so a refused restore \
         leaves the post-loss keyspace exactly as it found it. A header here would additionally \
         make the next replica's `EXISTS obj:hdr` short-circuit treat the recovery as done"
    );
}

/// b.zvd: `write_object` replaces a surviving children list rather than
/// appending behind it.
///
/// `execute_mutation`'s `SetChildren` arm has always `DEL`'d before its
/// `RPUSH`es; `write_object`'s children loop did not. On a freshly minted
/// `object_id` that is a no-op, which is why it went unnoticed — but the restore
/// path materializes at a KNOWN id over a keyspace that may have survived, and
/// there every restored child id landed a SECOND time behind the survivors. The
/// result is a children list that is both wrong and self-inconsistent with the
/// sections index, silently.
///
/// The same `DEL` is what lets `create_probe_keys` leave the list keys out of
/// the type probe (the push lands on a key that no longer exists whatever type
/// it held), so this test and that exclusion stand or fall together.
#[tokio::test]
async fn checkpoint_restore_replaces_a_surviving_children_list_rather_than_appending() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv, "zvd-restore-children").await;
    let oid = ls
        .create_layer(
            &session_id,
            "zvd-restore-children",
            "b-zvd-user",
            &seeded_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;
    crate::common::wipe_object_keys(&tv, oid).await;

    // Survivors: ids that are NOT in the checkpoint, so an append leaves them
    // visible and a replace does not.
    let oid_str = oid.to_string();
    let list_key = tv.key_schema().obj_children(&oid_str, "0");
    let _: () = tv
        .client()
        .pool
        .rpush(&list_key, vec!["7", "8"])
        .await
        .expect("planting the surviving children list must succeed");

    apply_checkpoint(
        &ls,
        oid,
        ompf,
        snap_seq,
        &meta,
        None,
        &crate::common::test_config(),
    )
    .await
    .expect("the restore must succeed over a surviving children list");

    let children: Vec<String> = tv
        .client()
        .pool
        .lrange(&list_key, 0, -1)
        .await
        .expect("LRANGE must succeed");
    assert_eq!(
        children,
        vec![SECTION_ID.to_string()],
        "the restored children list must be exactly what the checkpoint carried. Survivors still \
         present means the loop appended behind them; the checkpoint's own ids appearing twice \
         means it appended behind a partial re-run. Either way the layer's structure disagrees \
         with its sections index and nothing reported it"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Checkpoint staging helpers
// ═════════════════════════════════════════════════════════════════════════════

/// Assemble a checkpoint blob from the live layer via the production assembler.
///
/// Mirrors `checkpoint_apply_test.rs::build_checkpoint_ompf` — using the real
/// assembler keeps the restore authentic (the apply path consumes the exact byte
/// shape the assembler writes) without hand-constructing an `OwnedDelta`.
async fn build_checkpoint_ompf(
    tv: &TestValkey,
    object_id: u64,
) -> (bytes::Bytes, u64, CheckpointMeta) {
    let store = Arc::new(InMemoryCheckpointStore::new());
    let outcome = match assemble_layer_checkpoint(
        tv.client(),
        tv.key_schema(),
        object_id,
        store.as_ref(),
        "b-zvd-replica",
        &mut live_edit::checkpoint::assembler::NoLeaseToKeep,
    )
    .await
    .expect("the assembler must succeed against a populated live layer")
    {
        AssemblyDecision::Written(outcome) => outcome,
        AssemblyDecision::SkippedNotQuiesced => panic!(
            "the fixture layer has no concurrent writer and no regression, so the assembler's \
             proven read must verify; got SkippedNotQuiesced"
        ),
        AssemblyDecision::LeaseLost => {
            panic!("the assembler was handed a keeper that never loses its lease; got LeaseLost")
        }
        AssemblyDecision::ShuttingDown => {
            panic!("the assembler was handed a keeper that never shuts down; got ShuttingDown")
        }
    };

    let blob = store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("checkpoint get must succeed")
        .expect("a checkpoint must exist for the assembled object");

    (blob.bytes.clone(), outcome.snap_seq, blob.meta.clone())
}

// The local `wipe_object_keys` that used to sit here is gone (b.6de sub-task 4).
// It had converged, line for line, on `crate::common::wipe_object_keys` — same
// cluster-aware `scan_object_keys` enumeration, same UNLINK, same
// survivors-empty postcondition — and a helper duplicated per test file is how
// one copy quietly stops matching the others. Both call sites above now name
// the shared helper, whose doc-comment carries the `obj:stream` rationale and
// the argument for the postcondition.

// ═════════════════════════════════════════════════════════════════════════════
// The arm a live server cannot stage: a transient snapshot-read failure
// ═════════════════════════════════════════════════════════════════════════════

/// b.zvd's severe arm, driven through the snapshot-read fault seams.
///
/// # Why a seam is needed at all
///
/// The tests above this module stage the classification's `WRONGTYPE` half: a
/// real string on a real key, a real `LRANGE`, a real error reply. That is the
/// arm a keyspace can hold still for — and it is the *benign* half, because a
/// string sitting on a children-list key held no real children, so the old
/// "read it as empty" answer happened to be almost right.
///
/// The severe half is transient. A timed-out / reconnecting `LRANGE` on a
/// **populated** parent made that parent look childless, `apply_delta` computed
/// against the wrong state, and the structural commit then `DEL`'d a real
/// children list and `SREM`'d `obj:chs` under a fully valid `WATCH`. There is no
/// way to make a healthy Valkey time out one `LRANGE` in a fan-out on demand, so
/// that arm cannot be staged against a live server at all.
///
/// # What the seam is, and what it is not
///
/// One seam, `ValkeyClient::install_object_snapshot_read_fault_injector_for_test`,
/// covers the three object-snapshot reads below. Each queued canned
/// `fred::error::Error` answers in place of the call its
/// `ObjectSnapshotReadFaultTarget` names and is fed into the **same**
/// classification a real failure of that call takes, so an injected failure
/// and a real one provably share one classification path:
///
/// - **`read_object_snapshot`'s unfenced fan-out** (Spec SDD SR-4.3), the read
///   the structural commit path takes at every size. The default target
///   (`push`) answers in place of the first command of a fan-out window — on a
///   fresh read, the header `HGETALL`. `nth_of_family(UnfencedFanout,
///   "children", n)` singles out the `n`th children `LRANGE` instead: in round
///   2 on an object without the `chs_complete` marker (one list per live
///   section, the root's last), in round 3 after `SMEMBERS obj:chs` on a
///   marked one.
/// - **The fenced read's fan-out** (`read_object_snapshot_fenced`, Spec SDD
///   SR-5.3), which catch-up and the checkpoint assembler take above the
///   atomic-read limits: the same commands in the same order as the unfenced
///   fan-out, issued under the layer's fence. `nth_of_family(FencedFanout,
///   "children", n)` fails its `n`th children `LRANGE`, counted the same way —
///   the call the loader these consumers used before ran as
///   `.unwrap_or_default()` before b.zvd.
/// - **The atomic read** — catch-up, subscribe's per-commit reconstruction and
///   the checkpoint assembler at or below the limits. It is one `EVALSHA` that
///   reads every key in one call, so there is no per-list round trip left to
///   fail on its own: a failed read returns no state at all. Its transient
///   classification is pinned elsewhere (below).
///
/// So b.zvd's severe arm is staged on both reads that issue the list read:
/// the root's populated list fails transiently, and the read must fail —
/// `Aborted`, naming the parent — rather than answer with the root childless,
/// and on the commit path the retry must then commit against the real list.
/// The wrong-typed half stays staged against a live server, by
/// [`snapshot_read_of_a_wrongtype_children_list_is_a_loud_internal_naming_the_parent`]
/// and `b_atj_tests::fenced_read_of_a_wrongtype_children_list_ticks_children`.
///
/// **A positional target needs a drained-queue check.** The seam consults
/// only the front entry and pops it only when the command it names comes up,
/// so a target the read never reaches — an index past the read's last list, a
/// target of the other reader — stays queued without a sound. Every test
/// here that arms one asserts `remaining() == 0`, and
/// [`fenced_read_leaves_a_target_it_never_reaches_queued_and_reads_whole`]
/// proves that check can fail.
///
/// The seam is not a simulation of the retry loop: the loop, its backoff, its
/// metrics and the second attempt's real reads all run for real against real
/// Valkey.
///
/// | Test | What it pins |
/// |---|---|
/// | [`transient_snapshot_read_failure_on_the_commit_path_is_retried_and_the_commit_lands`] | the commit path: a transient failure of its snapshot read's header `HGETALL` → `Aborted` → the retry loop absorbs it → the commit lands once, with the children list intact |
/// | [`an_unarmed_seam_commits_without_a_retry`] | the negative control: the identical commit with no injection takes no retry, so the tests that count one are not describing an ordinary commit |
/// | [`update_layer_transient_children_lrange_failure_on_a_marked_object_is_retried_and_the_list_survives`] / [`update_layer_transient_children_lrange_failure_on_an_unmarked_object_is_retried_and_the_list_survives`] | the defect on the commit path: the root's list read fails → `Aborted` → retry → the commit lands with the seeded child kept, never a commit computed against a childless root that drops it — once per round the list read sits in |
/// | [`unfenced_read_transient_children_lrange_failure_on_a_marked_object_is_aborted_and_names_the_parent`] / [`unfenced_read_transient_children_lrange_failure_on_an_unmarked_object_is_aborted_and_names_the_parent`] | the fan-out's own verdict, read off the error: `Aborted` naming the root's list, never a snapshot missing the root's children |
/// | [`unfenced_read_transient_children_llen_failure_on_a_marked_object_is_aborted_and_names_the_parent`] / [`unfenced_read_transient_children_llen_failure_on_an_unmarked_object_is_aborted_and_names_the_parent`] | the same verdict for the structural read's other list command, the `LLEN` it reads before its ranged `LRANGE`: a swallowed failure there reads as an empty list too |
/// | [`fenced_read_transient_children_lrange_failure_on_a_marked_object_is_aborted_and_names_the_parent`] / [`fenced_read_transient_children_lrange_failure_on_an_unmarked_object_is_aborted_and_names_the_parent`] | the same verdict on the fenced read's fan-out, which ends the read with that command's error rather than retrying it as a broken attempt |
/// | [`fenced_read_leaves_a_target_it_never_reaches_queued_and_reads_whole`] | the control behind every drained-queue check: a target the fenced read never reaches stays queued and changes nothing |
/// | [`nth_of_family_with_an_unknown_family_label_panics`] | a misspelt family is refused at arming rather than never matching |
///
/// The atomic read's own classification of a transient failure — `Aborted`,
/// naming `EVALSHA object_snapshot_read`, no state returned — is pinned
/// elsewhere, not here: by
/// `snapshot_read_atomic_test::fault_seam::atomic_read_script_level_failures_take_the_unchanged_classifier`
/// and by
/// `b_atj_tests::non_wrongtype_snapshot_read_errors::snapshot_read_refused_by_a_transient_timeout_ticks_neither_counter`.
///
/// # Two activation conditions, one of them silent
///
/// The seam is `#[cfg(any(test, feature = "test-fault-injection"))]`, so this
/// module is feature-gated to match. Without `--features test-fault-injection`
/// these tests are compiled out entirely and the run reports green having
/// covered none of the above — pass the feature explicitly in any hand-rolled
/// invocation.
#[cfg(feature = "test-fault-injection")]
mod snapshot_read_fault_injection {
    use super::*;

    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use live_edit::storage::diff_read::{
        read_object_snapshot_slim, ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget,
        ObjectSnapshotReader, SlimSnapshotReadConsumer, SnapshotReadOf,
    };
    use live_edit::storage::layer::HEADER_FIELD_CHS_COMPLETE;
    use oodf_ffi::{CommandCreateSection, OwnedDelta, StructCommand};

    /// `live_edit_update_layer_retries_total{path="watch",outcome}` — the
    /// path every structural delta below commits on: [`seeded_layer`]'s store
    /// runs the `txn` backend, whose WATCH path is the one taking the snapshot
    /// read these seams disturb.
    ///
    /// Lives in this module rather than beside its `keyspace_type_error_total`
    /// sibling because only the seam-driven tests read it — the unfeatured
    /// families all assert on state or on an error, never on a retry count.
    fn update_layer_retries(metrics: &Metrics, outcome: &str) -> u64 {
        sample_value(
            &render(metrics),
            "live_edit_update_layer_retries_total",
            &[("path", "watch"), ("outcome", outcome)],
        )
        .unwrap_or_else(|| {
            panic!(
                "live_edit_update_layer_retries_total{{path=\"watch\",outcome=\"{outcome}\"}} is \
                 absent — every cell is zero-seeded, so absence means this store holds no \
                 metrics handle"
            )
        }) as u64
    }

    /// The section the structural deltas below create under the root.
    const CHILD_A: u64 = 7;
    /// A second root child, so the commit under test both reads and rewrites a
    /// non-empty children list.
    const CHILD_B: u64 = 8;

    fn create_section_delta(section_id: u64, name: &str) -> OwnedDelta {
        OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id,
                section_name: name.to_owned(),
                section_type: 1,
            })],
            ..Default::default()
        }
    }

    /// A layer whose root already has one child, plus the store and metrics.
    async fn seeded_layer(name: &str) -> (TestValkey, LayerStore, Arc<Metrics>, String, u64) {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
        let ls = layer_store(&tv)
            .with_update_layer_backend(UpdateLayerBackend::Txn)
            .with_metrics(metrics.clone());
        let session_id = create_session(&tv, name).await;
        let oid = ls
            .create_layer(
                &session_id,
                name,
                "b-zvd-user",
                &empty_state(),
                mock_ext_store(),
            )
            .await
            .expect("create_layer must succeed");
        ls.update_layer_relayed(
            &session_id,
            name,
            oid,
            Some(create_section_delta(CHILD_A, "First")),
            "zvd-writer",
            1_000,
        )
        .await
        .expect("the seeding structural commit must land");
        (tv, ls, metrics, session_id, oid)
    }

    /// A transient failure of the structural commit path's snapshot read is
    /// retried, and the commit lands with the children list intact.
    ///
    /// The claims, and why each is separately load-bearing:
    ///
    /// 1. **`Ok`** — the injected error became an `Aborted`, which is the retry
    ///    loop's in-band signal, so attempt 2 ran a real read and committed. An
    ///    `Internal` here would turn a Valkey blip on one read of a fan-out into
    ///    a client-visible write failure.
    /// 2. **The children list is correct** — an end-state check: the commit
    ///    that landed after the retry wrote both root children. The injected
    ///    failure replaces the read's header `HGETALL`, not a children
    ///    `LRANGE`, so this does not discriminate b.zvd's truncated-list
    ///    defect; the `children` tests below arm the root's list read itself.
    /// 3. **`obj:chs` still names the root** — likewise an end-state check:
    ///    the index and the list agree after the retried commit.
    /// 4. **Exactly one retry** — the operator-visible evidence that the blip
    ///    was absorbed rather than surfaced, and proof the commit was
    ///    re-attempted rather than repeated.
    /// 5. **The queue is drained** — the vacuity guard. Without it every claim
    ///    above is a statement about an ordinary healthy commit.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transient_snapshot_read_failure_on_the_commit_path_is_retried_and_the_commit_lands() {
        let (tv, ls, metrics, session_id, oid) = seeded_layer("zvd-seam-commit").await;

        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        injector.push(FredError::new(
            FredErrorKind::Timeout,
            "injected structural snapshot-read timeout",
        ));
        tv.client()
            .install_object_snapshot_read_fault_injector_for_test(injector.clone());

        ls.update_layer_relayed(
            &session_id,
            "zvd-seam-commit",
            oid,
            Some(create_section_delta(CHILD_B, "Second")),
            "zvd-writer",
            2_000,
        )
        .await
        .unwrap_or_else(|e| {
            panic!(
                "a fred Timeout on one command of the structural path's snapshot fan-out must be \
                 classified transient and retried. An error here means a Valkey blip on that \
                 read becomes a client-visible write failure; got {e:?}"
            )
        });

        let oid_str = oid.to_string();
        let children: Vec<String> = tv
            .client()
            .pool
            .lrange(tv.key_schema().obj_children(&oid_str, "0"), 0, -1)
            .await
            .expect("LRANGE must succeed");
        assert_eq!(
            children,
            vec![CHILD_A.to_string(), CHILD_B.to_string()],
            "after the retried commit the root's children must be both sections — the seeded \
             {CHILD_A} and the committed {CHILD_B}; anything else means the retry committed \
             against a state other than the object's"
        );

        let parents: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().obj_children_index(&oid_str))
            .await
            .expect("SMEMBERS must succeed");
        assert!(
            parents.contains(&"0".to_owned()),
            "`obj:chs` must still name the root after the retried commit: it is the index that \
             makes the `obj:ch:*` family enumerable at retirement, and it must agree with the \
             list. Got {parents:?}"
        );

        assert_eq!(
            update_layer_retries(&metrics, "committed"),
            1,
            "committing on attempt 2 is exactly one retry — and it is what proves the injected \
             error reached the retry loop as `Aborted` rather than being swallowed somewhere \
             upstream"
        );
        assert_eq!(
            injector.remaining(),
            0,
            "the injector queue must be drained by exactly one attempt. A leftover entry means \
             the seam never fired, which would make every assertion above a statement about an \
             ordinary healthy commit"
        );
    }

    /// The negative control: with no injection, the identical commit lands
    /// without a retry.
    ///
    /// Without this the commit-path tests in this module are unfalsifiable in
    /// one direction — a commit that always retried once would satisfy them —
    /// and their "exactly one retry" claim would be describing the fixture
    /// rather than the seam.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unarmed_seam_commits_without_a_retry() {
        let (tv, ls, metrics, session_id, oid) = seeded_layer("zvd-seam-control").await;

        ls.update_layer_relayed(
            &session_id,
            "zvd-seam-control",
            oid,
            Some(create_section_delta(CHILD_B, "Second")),
            "zvd-writer",
            2_000,
        )
        .await
        .expect("the uninjected structural commit must land");

        let children: Vec<String> = tv
            .client()
            .pool
            .lrange(tv.key_schema().obj_children(&oid.to_string(), "0"), 0, -1)
            .await
            .expect("LRANGE must succeed");
        assert_eq!(
            children,
            vec![CHILD_A.to_string(), CHILD_B.to_string()],
            "the uninjected commit must produce the same children list the injected one does"
        );
        assert_eq!(
            update_layer_retries(&metrics, "committed"),
            0,
            "an uninjected commit must take NO retry. A nonzero reading here means the fixture \
             itself retries and the sibling tests' `== 1` says nothing about the seam"
        );
    }

    // ── b.zvd's severe arm: a transient failure of the root's children LRANGE ──

    /// The parent id the root's children list is keyed and named by.
    const ROOT: &str = "0";

    /// The `op` both fan-outs put in a failed root-list `LRANGE`'s error.
    /// Restated rather than imported: the label is built inside
    /// `diff_read.rs`, and naming the parent is the contract the operator
    /// relies on.
    const ROOT_LIST_OP: &str = "LRANGE children list (parent 0)";

    /// The `op` the structural read puts in a failed root-list length read's
    /// error: that read takes a list's length (`LLEN`) before its ranged
    /// `LRANGE` (b.26e Stage 2), and labels it with the parent the same way.
    const ROOT_LIST_LENGTH_OP: &str = "LLEN children list (parent 0)";

    /// Which of the structural read's two commands for a children list a
    /// test fails: its length (`LLEN`, round 2) or its ranged ids (`LRANGE
    /// 0 n−1`, round 3, issued only for a non-empty list). A swallowed failure
    /// of either reads as a childless parent — a length of 0 issues no
    /// `LRANGE` at all — which is the b.zvd defect.
    #[derive(Clone, Copy, Debug)]
    enum ListRead {
        Length,
        Range,
    }

    impl ListRead {
        /// The `op` a failure of this command names for the root's list.
        fn op(self) -> &'static str {
            match self {
                ListRead::Length => ROOT_LIST_LENGTH_OP,
                ListRead::Range => ROOT_LIST_OP,
            }
        }

        /// This command's index among the structural read's `children`
        /// commands on the [`seeded_layer`].
        fn index_in_structural_read(self, marking: Marking) -> usize {
            match self {
                ListRead::Length => marking.root_list_in_fanout(),
                ListRead::Range => marking.root_list_range_in_structural_read(),
            }
        }
    }

    fn children_lrange_timeout() -> FredError {
        FredError::new(
            FredErrorKind::Timeout,
            "injected children-list LRANGE timeout",
        )
    }

    /// The seam with one error queued for `target`, installed on `tv`'s
    /// client (and every clone of it, so on the store's reads too).
    fn arm(
        tv: &TestValkey,
        target: ObjectSnapshotReadFaultTarget,
    ) -> Arc<ObjectSnapshotReadFaultInjector> {
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        injector.push_at(target, children_lrange_timeout());
        tv.client()
            .install_object_snapshot_read_fault_injector_for_test(injector.clone());
        injector
    }

    /// The fixture shape every positional index here is counted against: one
    /// live section, [`CHILD_A`], whose own children list is empty, and
    /// `obj:chs` naming the root alone. A layer that grew a second section or
    /// indexed list would move the root's list to another index, and the
    /// armed target would land on some other command.
    async fn assert_seeded_shape(tv: &TestValkey, oid: u64) {
        let oid_str = oid.to_string();
        let keys = tv.key_schema();
        let sections: Vec<String> = tv
            .client()
            .pool
            .smembers(keys.obj_sections_index(&oid_str))
            .await
            .expect("SMEMBERS obj:secs must succeed");
        let indexed: Vec<String> = tv
            .client()
            .pool
            .smembers(keys.obj_children_index(&oid_str))
            .await
            .expect("SMEMBERS obj:chs must succeed");
        let child_a_list: Vec<String> = tv
            .client()
            .pool
            .lrange(keys.obj_children(&oid_str, &CHILD_A.to_string()), 0, -1)
            .await
            .expect("LRANGE must succeed");
        let root_list: Vec<String> = tv
            .client()
            .pool
            .lrange(keys.obj_children(&oid_str, ROOT), 0, -1)
            .await
            .expect("LRANGE must succeed");
        assert_eq!(
            (sections, indexed, child_a_list, root_list),
            (
                vec![CHILD_A.to_string()],
                vec![ROOT.to_owned()],
                Vec::<String>::new(),
                vec![CHILD_A.to_string()],
            ),
            "fixture precondition: one live section {CHILD_A} with an empty list, `obj:chs` \
             naming only the root, and the root's list populated — the positional targets below \
             are counted against exactly this shape"
        );
    }

    /// Whether the object reads its children lists through `obj:chs` (marked,
    /// as `create_layer` leaves it) or the pre-index way (unmarked). Both
    /// fan-outs read the lists in a different round for each, so the
    /// commit-path and fan-out tests run once per marking.
    #[derive(Clone, Copy, Debug)]
    enum Marking {
        Marked,
        Unmarked,
    }

    impl Marking {
        /// The root list's first command among a fan-out's `children`
        /// commands on the [`seeded_layer`]: the fenced fan-out's whole-list
        /// `LRANGE`, and the structural read's `LLEN`, which it issues where
        /// the fenced fan-out issues that `LRANGE`. Marked: the lists
        /// `obj:chs` names, which is the root's alone. Unmarked: [`CHILD_A`]'s
        /// list first and the root's last.
        fn root_list_in_fanout(self) -> usize {
            match self {
                Marking::Marked => 0,
                Marking::Unmarked => 1,
            }
        }

        /// The root list's ranged `LRANGE` among the structural read's
        /// `children` commands: after every list's `LLEN`, and issued only
        /// for a non-empty list — so [`CHILD_A`]'s empty list issues none.
        /// Marked: the root's `LLEN`, then its `LRANGE`. Unmarked: both
        /// `LLEN`s, then the root's `LRANGE`.
        fn root_list_range_in_structural_read(self) -> usize {
            match self {
                Marking::Marked => 1,
                Marking::Unmarked => 2,
            }
        }

        /// Check the seeded shape, then bring the object to this marking.
        async fn stage(self, tv: &TestValkey, oid: u64) {
            assert_seeded_shape(tv, oid).await;
            if let Marking::Unmarked = self {
                let removed: u64 = tv
                    .client()
                    .pool
                    .hdel(
                        tv.key_schema().obj_header(&oid.to_string()),
                        HEADER_FIELD_CHS_COMPLETE,
                    )
                    .await
                    .expect("HDEL obj:hdr chs_complete must succeed");
                assert_eq!(
                    removed, 1,
                    "fixture precondition: `create_layer` marks the object, so there must have \
                     been a marker to remove"
                );
            }
        }

        fn layer_name(self, test: &str) -> String {
            match self {
                Marking::Marked => format!("{test}-marked"),
                Marking::Unmarked => format!("{test}-unmarked"),
            }
        }
    }

    /// The commit path through a transient failure of the root's children
    /// `LRANGE`: the structural `update_layer`'s unfenced read fails, the
    /// retry loop absorbs it, and the commit lands against the real list.
    ///
    /// The claims, and why each is separately load-bearing:
    ///
    /// 1. **`Ok`** — the injected error became an `Aborted`, the retry loop's
    ///    in-band signal. An `Internal` would turn a Valkey blip on one list
    ///    read into a client-visible write failure.
    /// 2. **The root still lists [`CHILD_A`] before [`CHILD_B`]** — the b.zvd
    ///    defect itself. A read that swallowed the failure would hand
    ///    `apply_delta` a childless root, and the structural commit would then
    ///    rewrite the root's list as `[CHILD_B]` alone, dropping the seeded
    ///    child under a fully valid `WATCH`. In a debug build such a
    ///    regression trips the domain's cross-map drift `debug_assert!` inside
    ///    `apply_delta` first (a live section missing from its parent's list),
    ///    so the test fails there; this assertion is what fails it in a build
    ///    without debug assertions.
    /// 3. **`obj:chs` still names the root** — the index agrees with the list.
    /// 4. **Exactly one retry** — the failure reached the loop as `Aborted`
    ///    and the commit was re-attempted, not repeated.
    /// 5. **The queue is drained** — the vacuity guard: a positional target the
    ///    read never reached stays queued, and every claim above would then be
    ///    about an ordinary healthy commit.
    async fn commit_through_a_failed_root_list_read(marking: Marking) {
        let layer = marking.layer_name("zvd-seam-ch-commit");
        let (tv, ls, metrics, session_id, oid) = seeded_layer(&layer).await;
        marking.stage(&tv, oid).await;
        let injector = arm(
            &tv,
            ObjectSnapshotReadFaultTarget::nth_of_family(
                ObjectSnapshotReader::UnfencedFanout,
                "children",
                ListRead::Range.index_in_structural_read(marking),
            ),
        );

        ls.update_layer_relayed(
            &session_id,
            &layer,
            oid,
            Some(create_section_delta(CHILD_B, "Second")),
            "zvd-writer",
            2_000,
        )
        .await
        .unwrap_or_else(|e| {
            panic!(
                "a fred Timeout on the structural read's children LRANGE ({marking:?}) must be \
                 classified transient and retried; an error here turns a Valkey blip on one list \
                 read into a client-visible write failure; got {e:?}"
            )
        });

        let oid_str = oid.to_string();
        let children: Vec<String> = tv
            .client()
            .pool
            .lrange(tv.key_schema().obj_children(&oid_str, ROOT), 0, -1)
            .await
            .expect("LRANGE must succeed");
        assert_eq!(
            children,
            vec![CHILD_A.to_string(), CHILD_B.to_string()],
            "the root must list the seeded {CHILD_A} and the committed {CHILD_B} ({marking:?}). \
             `[{CHILD_B}]` alone is the b.zvd defect: the failed LRANGE read as a childless \
             root and the commit rewrote the real list from that"
        );
        let parents: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().obj_children_index(&oid_str))
            .await
            .expect("SMEMBERS must succeed");
        assert!(
            parents.contains(&ROOT.to_owned()),
            "`obj:chs` must still name the root after the retried commit ({marking:?}); got \
             {parents:?}"
        );
        assert_eq!(
            update_layer_retries(&metrics, "committed"),
            1,
            "committing on attempt 2 is exactly one retry ({marking:?}), and it is what proves \
             the failed list read reached the retry loop as `Aborted` rather than being swallowed"
        );
        assert_eq!(
            injector.remaining(),
            0,
            "the root-list target must have fired ({marking:?}). A positional target the read \
             never reaches stays queued, and every assertion above would then describe an \
             ordinary healthy commit"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_transient_children_lrange_failure_on_a_marked_object_is_retried_and_the_list_survives(
    ) {
        commit_through_a_failed_root_list_read(Marking::Marked).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn update_layer_transient_children_lrange_failure_on_an_unmarked_object_is_retried_and_the_list_survives(
    ) {
        commit_through_a_failed_root_list_read(Marking::Unmarked).await;
    }

    /// The unfenced fan-out's own verdict on a transient failure of one of
    /// the structural read's two commands for the root's children list (its
    /// `LLEN` or its ranged `LRANGE`), read off the error rather than
    /// inferred from a retry.
    ///
    /// The clean read first is the control: the root really does list
    /// [`CHILD_A`], so a snapshot without it cannot be mistaken for the truth.
    /// Then the armed read must be `Aborted` (retriable — a transport blip is
    /// not permanent) and must name the root's list under the failed
    /// command's own `op`, which also proves the target counted its way to
    /// the root's list and to that command, not to [`CHILD_A`]'s list or the
    /// other command.
    async fn unfenced_read_through_a_failed_root_list_read(marking: Marking, read: ListRead) {
        let layer = marking.layer_name(match read {
            ListRead::Length => "zvd-seam-ch-fanout-len",
            ListRead::Range => "zvd-seam-ch-fanout",
        });
        let (tv, _ls, _metrics, _session_id, oid) = seeded_layer(&layer).await;
        marking.stage(&tv, oid).await;
        let oid_str = oid.to_string();

        match read_object_snapshot_slim(
            tv.client(),
            tv.key_schema(),
            &oid_str,
            SlimSnapshotReadConsumer::StructuralUpdate,
        )
        .await
        {
            Ok(SnapshotReadOf::Snapshot(clean)) => assert_eq!(
                clean.state.children.get(&0).map(Vec::as_slice),
                Some([CHILD_A].as_slice()),
                "control ({marking:?}): the root really has a child"
            ),
            Ok(SnapshotReadOf::TooLarge(t)) => {
                panic!("the unfenced fan-out reads every size; got TOO_LARGE {t:?}")
            }
            Err(e) => panic!("the uninjected unfenced read must succeed ({marking:?}); got {e:?}"),
        }

        let injector = arm(
            &tv,
            ObjectSnapshotReadFaultTarget::nth_of_family(
                ObjectSnapshotReader::UnfencedFanout,
                "children",
                read.index_in_structural_read(marking),
            ),
        );
        let err = match read_object_snapshot_slim(
            tv.client(),
            tv.key_schema(),
            &oid_str,
            SlimSnapshotReadConsumer::StructuralUpdate,
        )
        .await
        {
            Err(e) => e,
            Ok(SnapshotReadOf::Snapshot(snapshot)) => panic!(
                "a failed children-list {read:?} read must NOT return a snapshot ({marking:?}). \
                 This is the b.zvd defect: the read answered children = {:?} for a root that \
                 lists {CHILD_A}, and the structural commit would compute against it",
                snapshot.state.children
            ),
            Ok(SnapshotReadOf::TooLarge(t)) => {
                panic!("the unfenced fan-out reads every size; got TOO_LARGE {t:?}")
            }
        };

        let ServiceError::Aborted(msg) = &err else {
            panic!(
                "a transient failure of one list read must be RETRIABLE ({marking:?}): `Aborted` \
                 is what sends the structural path round its retry loop; got {err:?}"
            );
        };
        assert!(
            msg.contains(read.op()),
            "the error must name the root's list under the failed command's op ({marking:?}, \
             {read:?}) — the parent id is the one diagnostic the read carries for a list, and \
             naming another parent or command would mean the target landed on a different \
             command; got {msg:?}"
        );
        assert_eq!(
            injector.remaining(),
            0,
            "the root-list target must have fired ({marking:?}), or the error above is some other \
             failure's"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unfenced_read_transient_children_lrange_failure_on_a_marked_object_is_aborted_and_names_the_parent(
    ) {
        unfenced_read_through_a_failed_root_list_read(Marking::Marked, ListRead::Range).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unfenced_read_transient_children_lrange_failure_on_an_unmarked_object_is_aborted_and_names_the_parent(
    ) {
        unfenced_read_through_a_failed_root_list_read(Marking::Unmarked, ListRead::Range).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unfenced_read_transient_children_llen_failure_on_a_marked_object_is_aborted_and_names_the_parent(
    ) {
        unfenced_read_through_a_failed_root_list_read(Marking::Marked, ListRead::Length).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unfenced_read_transient_children_llen_failure_on_an_unmarked_object_is_aborted_and_names_the_parent(
    ) {
        unfenced_read_through_a_failed_root_list_read(Marking::Unmarked, ListRead::Length).await;
    }

    /// The fenced read's verdict on a transient failure of the root's
    /// children `LRANGE` — the read catch-up and the checkpoint assembler take
    /// above the atomic-read limits, issuing the per-parent list read b.zvd
    /// fixed.
    ///
    /// The same claims as the unfenced fan-out's: the clean read is the
    /// control, the armed read is `Aborted` naming the root's list (the
    /// classification `classify_snapshot_read_err` gives a real `Timeout` of
    /// that call), and the queue is drained. A failed fan-out command ends the
    /// fenced read with its error rather than counting as a broken attempt, so
    /// the read's later attempts never get to read the list for real and hide
    /// the failure.
    async fn fenced_read_through_a_failed_root_list_read(marking: Marking) {
        let layer = marking.layer_name("zvd-seam-ch-fenced");
        let (tv, _ls, _metrics, _session_id, oid) = seeded_layer(&layer).await;
        marking.stage(&tv, oid).await;

        let clean = match fenced_read(tv.client(), tv.key_schema(), oid).await {
            Ok(clean) => clean,
            Err(e) => panic!("the uninjected fenced read must succeed ({marking:?}); got {e:?}"),
        };
        assert_eq!(
            clean.state.children.get(&0).map(Vec::as_slice),
            Some([CHILD_A].as_slice()),
            "control ({marking:?}): the root really has a child"
        );

        let injector = arm(
            &tv,
            ObjectSnapshotReadFaultTarget::nth_of_family(
                ObjectSnapshotReader::FencedFanout,
                "children",
                marking.root_list_in_fanout(),
            ),
        );
        // `AtomicSnapshot` is not `Debug`, so the `Ok` arm is spelled out
        // rather than reached through `expect_err`.
        let err = match fenced_read(tv.client(), tv.key_schema(), oid).await {
            Err(e) => e,
            Ok(snapshot) => panic!(
                "a failed children-list LRANGE must NOT return a snapshot ({marking:?}). This is \
                 the b.zvd defect: the read answered children = {:?} for a root that lists \
                 {CHILD_A}, and every consumer of this read above the atomic-read limits would \
                 believe it",
                snapshot.state.children
            ),
        };

        let ServiceError::Aborted(msg) = &err else {
            panic!(
                "a transient failure of one list read must be RETRIABLE ({marking:?}): \
                 `Internal` here turns a Valkey blip into a permanent failure for every consumer \
                 of this read; got {err:?}"
            );
        };
        assert!(
            msg.contains(ROOT_LIST_OP),
            "the error must name the root's list ({marking:?}) — the parent id is the one \
             diagnostic the read carries for a list, and naming another parent would mean the \
             target landed on a different command; got {msg:?}"
        );
        assert_eq!(
            injector.remaining(),
            0,
            "the root-list target must have fired ({marking:?}), or the error above is some other \
             failure's"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fenced_read_transient_children_lrange_failure_on_a_marked_object_is_aborted_and_names_the_parent(
    ) {
        fenced_read_through_a_failed_root_list_read(Marking::Marked).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fenced_read_transient_children_lrange_failure_on_an_unmarked_object_is_aborted_and_names_the_parent(
    ) {
        fenced_read_through_a_failed_root_list_read(Marking::Unmarked).await;
    }

    /// The control behind every `remaining() == 0` above: a target the
    /// fenced read never reaches stays queued, and the read is the real one.
    ///
    /// Two targets it must pass by: one past its last children list, and one
    /// for the unfenced fan-out's lists. (`next_read` is not one: the fenced
    /// fan-out answers it at its first window, as the unfenced one does.) If
    /// an unmatched entry were dropped rather than kept, a drained queue would
    /// prove nothing.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fenced_read_leaves_a_target_it_never_reaches_queued_and_reads_whole() {
        let (tv, _ls, _metrics, _session_id, oid) = seeded_layer("zvd-seam-ch-unreached").await;
        Marking::Marked.stage(&tv, oid).await;

        for target in [
            ObjectSnapshotReadFaultTarget::nth_of_family(
                ObjectSnapshotReader::FencedFanout,
                "children",
                Marking::Marked.root_list_in_fanout() + 1,
            ),
            ObjectSnapshotReadFaultTarget::nth_of_family(
                ObjectSnapshotReader::UnfencedFanout,
                "children",
                0,
            ),
        ] {
            let injector = arm(&tv, target);
            let snapshot = match fenced_read(tv.client(), tv.key_schema(), oid).await {
                Ok(snapshot) => snapshot,
                Err(e) => panic!("{target:?} is not a command this read issues; got {e:?}"),
            };
            assert_eq!(
                snapshot.state.children.get(&0).map(Vec::as_slice),
                Some([CHILD_A].as_slice()),
                "{target:?} must leave the read whole"
            );
            assert_eq!(
                injector.remaining(),
                1,
                "{target:?} was never reached, so it must still be queued — a seam that dropped \
                 it would make every drained-queue check in this module vacuous"
            );
        }
    }

    /// A family label that matches no key family is refused when the target
    /// is built: it would otherwise never match, and a test arming it would
    /// pass without injecting anything.
    #[test]
    #[should_panic(expected = "is not a snapshot-read key family")]
    fn nth_of_family_with_an_unknown_family_label_panics() {
        let _ = ObjectSnapshotReadFaultTarget::nth_of_family(
            ObjectSnapshotReader::FencedFanout,
            "childern",
            0,
        );
    }

    /// A retirement size-read target is refused by the delay queue: no
    /// fan-out command matches it, so the delay would never be taken and
    /// every delay queued behind it would wait forever.
    #[test]
    #[should_panic(expected = "push_delay_at: a retirement_size_read(\"sections_index\") target")]
    fn push_delay_at_with_a_retirement_size_read_target_panics() {
        ObjectSnapshotReadFaultInjector::new().push_delay_at(
            ObjectSnapshotReadFaultTarget::retirement_size_read("sections_index"),
            std::time::Duration::from_millis(1),
        );
    }

    /// The hold queue refuses it too, for the same reason.
    #[test]
    #[should_panic(expected = "push_hold_at: a retirement_size_read(\"header\") target")]
    fn push_hold_at_with_a_retirement_size_read_target_panics() {
        let _ = ObjectSnapshotReadFaultInjector::new().push_hold_at(
            ObjectSnapshotReadFaultTarget::retirement_size_read("header"),
        );
    }

    /// A family label naming no command of the size read's pipeline is
    /// refused when the target is built: `"children"` is a key family, but
    /// the pipeline counts `obj:chs` (`"children_index"`), not the lists.
    #[test]
    #[should_panic(expected = "names no command of the retirement size read")]
    fn retirement_size_read_with_a_family_outside_its_pipeline_panics() {
        let _ = ObjectSnapshotReadFaultTarget::retirement_size_read("children");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The other arm a live server cannot stage: a failed create-site type probe
// ═════════════════════════════════════════════════════════════════════════════

/// The one property that makes reusing b.459's probe at the create site
/// *correct* rather than merely convenient.
///
/// `update_layer_once` sits inside the retry loop, which is what makes `Aborted`
/// honest there: it advertises a retry budget that genuinely exists and the loop
/// consumes it before the client ever sees it. `LayerStore::write_object` runs
/// no retry loop at all — `CreateLayer` and the DR restore both call it once — so
/// the same failure must come back as `Unavailable`. Getting this backwards
/// hands an in-process caller the `Aborted` retry signal for a retry nobody
/// performs (on the wire the two are indistinguishable: an escaped `Aborted`
/// reaches the client as `UNAVAILABLE`; the service's one `ABORTED` is a
/// chained `UpdateLayer`'s predecessor refusal, which neither path produces).
///
/// Reproducing it against a live server means killing Valkey inside the window
/// between the `INCR` and the `MULTI`, so it goes through
/// `LayerStore::with_keyspace_probe_fred_error_injector_for_test` — the same
/// seam b.459's transient tests use, injecting *upstream* of
/// `classify_keyspace_probe_error`, which is the production classifier both
/// sites share. The classification table itself is pinned as a pure unit test
/// (`src/storage/layer.rs::tests::a_transient_probe_failure_is_aborted_on_update_and_unavailable_on_create`);
/// what this adds is that `create_layer` really does route through it, against
/// real Valkey, and writes nothing when it does.
#[cfg(feature = "test-fault-injection")]
mod create_probe_fault_injection {
    use super::*;

    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use std::collections::VecDeque;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_transient_probe_failure_at_the_create_site_is_unavailable_not_aborted() {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
        let queue: Arc<Mutex<VecDeque<FredError>>> = Arc::new(Mutex::new(VecDeque::new()));
        let ls = layer_store(&tv)
            .with_metrics(metrics.clone())
            .with_keyspace_probe_fred_error_injector_for_test(queue.clone());
        let session_id = create_session(&tv, "zvd-create-transient").await;

        // Armed AFTER construction so nothing but the call under test can burn
        // an entry, which is what makes the drained-queue assertion below
        // attributable.
        queue.lock().unwrap().push_back(FredError::new(
            FredErrorKind::Timeout,
            "injected create-site probe timeout",
        ));

        let err = ls
            .create_layer(
                &session_id,
                "zvd-create-transient",
                "b-zvd-user",
                &seeded_state(),
                mock_ext_store(),
            )
            .await
            .expect_err("the injected probe failure must surface");

        assert!(
            matches!(err, ServiceError::Unavailable(_)),
            "`write_object` runs no retry loop, so an inconclusive probe must reach the client as \
             UNAVAILABLE — a status standard client retry policies act on. `Aborted` would \
             advertise a server-side retry budget that does not exist here (it is the update \
             path's, whose loop consumes it before the client sees anything), and `Internal` \
             would turn a Valkey blip into a permanent CreateLayer failure; got {err:?}"
        );
        assert_eq!(
            keyspace_type_errors(&metrics, "txn", "create"),
            0,
            "a transport failure on the probe is not a keyspace anomaly. Ticking here would put \
             this counter on the same footing as ordinary Valkey noise and cost it the zero \
             baseline its alert depends on"
        );
        assert_eq!(
            queue.lock().unwrap().len(),
            0,
            "the injector queue must be drained by this one call — a leftover entry means \
             `create_layer` never issued the probe and the assertions above describe some other \
             error"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The race a live server can only sample: the `obj:ext` read failing first
// ═════════════════════════════════════════════════════════════════════════════

/// b.j2x on the two materialization entry points: a wrong-typed `obj:ext`
/// refuses with the keyspace probe's tagged `PermanentKeyspaceRefusal`, even
/// when the `HGETALL obj:ext` pre-image read beside the probe fails first.
///
/// # The defect
///
/// `create_probe_keys` names `obj:ext` unconditionally, and both
/// `write_object` (CreateLayer) and `write_restored_object` (DR restore)
/// `HGETALL` it at the same time as the probe. Before b.j2x the two sat under
/// one fail-fast `try_join!`, so a wrong-typed `obj:ext` was refused by
/// whichever failed first. When the read won, the refusal was a plain
/// `Internal` from `map_fred_err_for_grpc`, with no `site="create"` tick and no
/// ERROR line naming the key. On the restore path
/// `write_object_err_to_checkpoint_err` then folded it to
/// `CheckpointError::Transient`, so the recovery hook would re-drive a restore
/// no retry can clear.
///
/// # The seam
///
/// `LayerStore::with_ext_pre_image_read_fred_error_injector_for_test` fails
/// the read before its first await, sending nothing, through
/// `map_fred_err_for_grpc` and the read's own op label, as a real failure
/// would. Under a fail-fast join that error always arrives first, so the old
/// code fails the two wrong-type rows below on every run. Each test asserts
/// the queue drained: a leftover entry means the read never ran, and then
/// nothing here says anything about the ordering.
///
/// # The other half: the read's own failure still decides
///
/// Consulting the checks first must not mean ignoring the read. The last four
/// rows inject the read's failure over a CLEAN keyspace, so the guards pass
/// and the read's error is the only one. They pass with or without the
/// ordering fix; what they catch is a site that drops the read's error once
/// the guards pass (`guards.map(|_| ext.unwrap_or_default())` in place of
/// `guards.and(ext)`), which would return `Ok` having planned against an index
/// it never read. The update site's equivalent is
/// `b_459_tests::ext_pre_image_read_fault_injection`.
///
/// | Test | What it pins |
/// |---|---|
/// | [`create_layer_wrongtype_ext_index_refuses_tagged_even_when_the_ext_read_fails_first`] | CreateLayer: `PermanentKeyspaceRefusal { WrongType }` naming `obj:ext`, `{txn,create}` +1, nothing materialized |
/// | [`checkpoint_restore_wrongtype_ext_index_is_permanent_even_when_the_ext_read_fails_first`] | DR restore: `CheckpointError::Permanent` naming `obj:ext`, `{txn,create}` +1, the ERROR line carrying `entry_point="checkpoint_restore"` |
/// | [`create_layer_transient_ext_index_read_failure_is_unavailable_naming_the_read`] | CreateLayer, clean keyspace, `Timeout`: `Unavailable` naming the read, no tick, nothing materialized |
/// | [`create_layer_non_transient_ext_index_read_failure_is_internal_naming_the_read`] | CreateLayer, clean keyspace, `Protocol`: `Internal` naming the read, no tick, nothing materialized |
/// | [`checkpoint_restore_transient_ext_index_read_failure_is_transient`] | DR restore, clean keyspace, `Timeout`: `CheckpointError::Transient` carrying the read's `Unavailable`, no tick, no header |
/// | [`checkpoint_restore_non_transient_ext_index_read_failure_is_transient_not_permanent`] | DR restore, clean keyspace, `Protocol`: `CheckpointError::Transient`, not `Permanent`, carrying the read's `Internal`, no tick, no header |
///
/// # Two activation conditions, one of them silent
///
/// Feature-gated on `test-fault-injection` to match the seam. Without the
/// feature these tests are compiled out and the run reports green having
/// covered neither.
#[cfg(feature = "test-fault-injection")]
mod ext_index_read_fault_injection {
    use super::*;

    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use std::collections::VecDeque;

    type FredQueue = Arc<Mutex<VecDeque<FredError>>>;

    /// The error a real `HGETALL` against a non-hash key produces: Valkey's
    /// `-WRONGTYPE Operation against a key holding the wrong kind of value`,
    /// which fred maps to `InvalidArgument`. It carries no `WRONGTYPE_KEY:` tag,
    /// so a refusal built from it can only be the untagged one. Same shape as
    /// `b_459_tests::ext_pre_image_read_fault_injection::server_wrongtype_reply`
    /// on the update path.
    fn server_wrongtype_reply() -> FredError {
        FredError::new(
            FredErrorKind::InvalidArgument,
            "WRONGTYPE Operation against a key holding the wrong kind of value",
        )
    }

    /// A layer store reporting into `metrics`, with the `obj:ext` read wired to
    /// an initially EMPTY queue.
    ///
    /// Empty because any setup `create_layer` a test runs pops from the same
    /// queue through its own `HGETALL obj:ext`. Every test arms just before the
    /// call under test, which keeps the drained-queue assertion about that call
    /// alone.
    fn store_with_ext_read_injector(
        tv: &TestValkey,
        metrics: &Arc<Metrics>,
    ) -> (LayerStore, FredQueue) {
        let queue: FredQueue = Arc::new(Mutex::new(VecDeque::new()));
        let ls = layer_store(tv)
            .with_metrics(metrics.clone())
            .with_ext_pre_image_read_fred_error_injector_for_test(queue.clone());
        (ls, queue)
    }

    async fn plant_string(tv: &TestValkey, key: &str, value: &str) {
        let _: () = tv
            .client()
            .pool
            .set(key, value, None, None, false)
            .await
            .unwrap_or_else(|e| panic!("planting a string at {key} must succeed: {e}"));
    }

    /// b.j2x: CreateLayer over a wrong-typed `obj:ext` refuses with the probe's
    /// tagged `PermanentKeyspaceRefusal` even when its `HGETALL obj:ext` fails
    /// first.
    ///
    /// Staged at the id the next `create_layer` mints, as in
    /// [`create_layer_refuses_whole_when_a_key_it_would_write_holds_the_wrong_type`];
    /// only the planted key differs. `seeded_state` writes no external value,
    /// so `obj:ext` is on the probe list only because `create_probe_keys` names
    /// it unconditionally, and every other key that list names is absent.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_wrongtype_ext_index_refuses_tagged_even_when_the_ext_read_fails_first() {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
        let (ls, queue) = store_with_ext_read_injector(&tv, &metrics);
        let session_id = create_session(&tv, "zvd-j2x-create").await;

        ls.create_layer(
            &session_id,
            "zvd-j2x-create-seed",
            "b-zvd-user",
            &empty_state(),
            mock_ext_store(),
        )
        .await
        .expect("the seed create_layer must succeed");
        let counter: u64 = tv
            .client()
            .pool
            .get::<String, _>(tv.key_schema().counter_object_id())
            .await
            .expect("the object-id counter must exist after one create_layer")
            .parse()
            .expect("the object-id counter must be numeric");
        let next_oid = (counter + 1).to_string();

        let ext_key = tv.key_schema().obj_ext_index(&next_oid);
        plant_string(&tv, &ext_key, "planted-by-an-out-of-band-writer").await;
        queue.lock().unwrap().push_back(server_wrongtype_reply());

        let err = ls
            .create_layer(
                &session_id,
                "zvd-j2x-create",
                "b-zvd-user",
                &seeded_state(),
                mock_ext_store(),
            )
            .await
            .expect_err("a wrong-typed `obj:ext` must refuse the whole `create_layer`");

        let ServiceError::PermanentKeyspaceRefusal { refusal, detail } = &err else {
            panic!(
                "a wrong-typed `obj:ext` must refuse with the keyspace probe's \
                 `PermanentKeyspaceRefusal`, however the `HGETALL obj:ext` beside it fared. A \
                 plain `Internal` reading `HGETALL ext index failed: … WRONGTYPE …` is the read's \
                 own failure reported in the probe's place: no `site=\"create\"` tick, no ERROR \
                 line naming the key; got {err:?}"
            );
        };
        assert_eq!(
            *refusal,
            KeyspaceRefusal::WrongType,
            "the kind must name the condition; got {refusal:?}"
        );
        assert!(
            detail.starts_with(CREATE_KEYSPACE_TYPE_ERROR_PREFIX),
            "the refusal must open with {CREATE_KEYSPACE_TYPE_ERROR_PREFIX:?}; got {detail:?}"
        );
        assert!(
            detail.contains(&ext_key)
                && detail.contains("holds a string")
                && detail.contains("expected hash"),
            "the refusal must name `obj:ext`, what it holds and what the materialization needs \
             — that triple is the remediation; got {detail:?}"
        );

        assert_eq!(
            queue.lock().unwrap().len(),
            0,
            "the `HGETALL obj:ext` must have consumed its injected `WRONGTYPE`. A leftover entry \
             means the read never ran, so it was never in the race this test settles"
        );
        assert_eq!(
            keyspace_type_errors(&metrics, "txn", "create"),
            1,
            "exactly one refusal must tick \
             live_edit_keyspace_type_error_total{{backend=\"txn\", site=\"create\"}}"
        );
        assert_eq!(
            keyspace_type_errors(&metrics, "txn", "update"),
            0,
            "a create-path refusal must not be filed under `site=\"update\"`"
        );

        for (label, key) in [
            ("the object header", tv.key_schema().obj_header(&next_oid)),
            (
                "the sections index",
                tv.key_schema().obj_sections_index(&next_oid),
            ),
            ("the live stream", tv.key_schema().layer_stream(counter + 1)),
            (
                "the layer mapping",
                tv.key_schema().layer_mapping(&session_id, "zvd-j2x-create"),
            ),
        ] {
            let exists: u64 = tv
                .client()
                .pool
                .exists(&key)
                .await
                .expect("EXISTS must succeed");
            assert_eq!(
                exists, 0,
                "{label} ({key}) must not exist: both guards and the read run before `MULTI`, so \
                 a refused materialization writes nothing"
            );
        }
    }

    /// b.j2x: a DR restore over a surviving wrong-typed `obj:ext` reaches the
    /// recovery hook as `CheckpointError::Permanent`, even when its
    /// `HGETALL obj:ext` fails first.
    ///
    /// Staged as in
    /// [`checkpoint_restore_refuses_whole_when_a_key_it_would_write_holds_the_wrong_type`]
    /// (create, assemble, wipe, plant), with the string at `obj:ext`. That key
    /// is what the restore residue wipe deliberately leaves standing, so it is
    /// the one a surviving keyspace can hand this path. Runs on a current-thread
    /// runtime so the thread-local log capture sees the probe's ERROR line.
    #[tokio::test]
    async fn checkpoint_restore_wrongtype_ext_index_is_permanent_even_when_the_ext_read_fails_first(
    ) {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
        let (ls, queue) = store_with_ext_read_injector(&tv, &metrics);
        let session_id = create_session(&tv, "zvd-j2x-restore").await;
        let oid = ls
            .create_layer(
                &session_id,
                "zvd-j2x-restore",
                "b-zvd-user",
                &seeded_state(),
                mock_ext_store(),
            )
            .await
            .expect("create_layer must succeed");

        let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;
        crate::common::wipe_object_keys(&tv, oid).await;

        let oid_str = oid.to_string();
        let ext_key = tv.key_schema().obj_ext_index(&oid_str);
        plant_string(&tv, &ext_key, "survived-the-loss-with-the-wrong-type").await;
        queue.lock().unwrap().push_back(server_wrongtype_reply());

        let (writer, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(writer)
            .with_max_level(tracing::Level::ERROR)
            .with_ansi(false)
            .finish();
        let capture = tracing::subscriber::set_default(subscriber);

        let err = apply_checkpoint(
            &ls,
            oid,
            ompf,
            snap_seq,
            &meta,
            None,
            &crate::common::test_config(),
        )
        .await
        .expect_err("the restore must refuse over a wrong-typed surviving `obj:ext`");

        let logged = {
            let buffer = captured
                .lock()
                .expect("the log-capture mutex is never poisoned");
            String::from_utf8_lossy(&buffer).into_owned()
        };
        drop(capture);

        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "a wrong-typed `obj:ext` must reach the recovery hook as `CheckpointError::Permanent`. \
             `Transient` is what the `HGETALL obj:ext`'s own untagged failure folds to, and it \
             re-drives a restore no retry can clear; got {err:?}"
        );
        let rendered = err.to_string();
        assert!(
            rendered.contains(CREATE_KEYSPACE_TYPE_ERROR_PREFIX)
                && rendered.contains(&ext_key)
                && rendered.contains("expected hash"),
            "the probe's refusal, naming `obj:ext`, must survive the checkpoint-error wrap; got \
             {rendered:?}"
        );
        assert!(
            logged.contains(&format!("entry_point=\"{ENTRY_POINT_CHECKPOINT_RESTORE}\""))
                && logged.contains(&ext_key),
            "the probe's ERROR line must be emitted, name the key and say the restore entry \
             point refused. Captured: {logged}"
        );

        assert_eq!(
            queue.lock().unwrap().len(),
            0,
            "the `HGETALL obj:ext` must have consumed its injected `WRONGTYPE`. A leftover entry \
             means the read never ran, so it was never in the race this test settles"
        );
        assert_eq!(
            keyspace_type_errors(&metrics, "txn", "create"),
            1,
            "the restore's refusal ticks the create cell exactly once; `entry_point` in the log, \
             not the metric, tells the two callers apart"
        );

        let header_exists: u64 = tv
            .client()
            .pool
            .exists(tv.key_schema().obj_header(&oid_str))
            .await
            .expect("EXISTS must succeed");
        assert_eq!(
            header_exists, 0,
            "the object header must not exist: a refused restore writes nothing"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // The other half: once the probe passes, the read's own failure decides
    // ─────────────────────────────────────────────────────────────────────────

    /// The op label a failure of the `HGETALL obj:ext` read carries on both
    /// entry points (`LayerStore::EXT_INDEX_HGETALL_OP`). Restated rather than
    /// imported: the constant is private to `src/storage/layer.rs`, and the
    /// string is what an operator reads.
    const EXT_INDEX_HGETALL_OP: &str = "HGETALL ext index";

    /// The two classes of `HGETALL obj:ext` failure the rows below inject over
    /// a clean keyspace. `map_fred_err_for_grpc` splits them: a transient kind
    /// becomes `Unavailable`, any other kind `Internal`.
    #[derive(Clone, Copy, Debug)]
    enum ReadFault {
        /// `Timeout`, a transient kind.
        Timeout,
        /// `Protocol`, a non-transient kind.
        Protocol,
    }

    impl ReadFault {
        fn kind(self) -> FredErrorKind {
            match self {
                Self::Timeout => FredErrorKind::Timeout,
                Self::Protocol => FredErrorKind::Protocol,
            }
        }

        fn injected_message(self) -> &'static str {
            match self {
                Self::Timeout => "injected ext index read timeout",
                Self::Protocol => "injected ext index read protocol fault",
            }
        }

        fn error(self) -> FredError {
            FredError::new(self.kind(), self.injected_message())
        }
    }

    /// Assert the queue drained and that neither site of
    /// `live_edit_keyspace_type_error_total{backend="txn"}` ticked.
    ///
    /// The drained queue is what makes the failure attributable to the read: a
    /// leftover entry means the read never ran, and the error under test came
    /// from somewhere else. The zero ticks say the probe passed over the clean
    /// keyspace, so the read's own failure is what was reported.
    fn assert_read_failure_alone_was_reported(queue: &FredQueue, metrics: &Metrics, ctx: &str) {
        assert_eq!(
            queue.lock().unwrap().len(),
            0,
            "{ctx}: the `HGETALL obj:ext` must have consumed its injected failure. A leftover \
             entry means the read never ran, so the error asserted on did not come from it"
        );
        for site in ["create", "update"] {
            assert_eq!(
                keyspace_type_errors(metrics, "txn", site),
                0,
                "{ctx}: a failed read over a clean keyspace is not a keyspace anomaly; \
                 live_edit_keyspace_type_error_total{{backend=\"txn\", site=\"{site}\"}} must not \
                 tick"
            );
        }
    }

    /// Drive `create_layer` over a clean keyspace with its `HGETALL obj:ext`
    /// armed to fail with `fault`, and return the refusal.
    ///
    /// Asserts what holds for every fault: the call is refused (a fix that
    /// dropped the read's error once the guards passed would return `Ok` here,
    /// having planned against an index it never read), the failure came from
    /// the read and from nothing else, and nothing was materialized under the
    /// minted id. The caller asserts the variant and message.
    async fn create_layer_refused_by_a_failed_ext_read(
        name: &str,
        fault: ReadFault,
    ) -> ServiceError {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
        let (ls, queue) = store_with_ext_read_injector(&tv, &metrics);
        let session_id = create_session(&tv, name).await;
        queue.lock().unwrap().push_back(fault.error());

        let err = ls
            .create_layer(
                &session_id,
                name,
                "b-zvd-user",
                &seeded_state(),
                mock_ext_store(),
            )
            .await
            .expect_err(
                "a failed `HGETALL obj:ext` must refuse the `create_layer` once the guards have \
                 passed. An `Ok` means the read's error was dropped and the commit was planned \
                 against an index that was never read",
            );

        let ctx = format!("create_layer, {fault:?}");
        assert_read_failure_alone_was_reported(&queue, &metrics, &ctx);

        // `create_layer` minted its id by `INCR` before `write_object` ran, so
        // the counter names the id the refused materialization would have used.
        let oid: u64 = tv
            .client()
            .pool
            .get::<String, _>(tv.key_schema().counter_object_id())
            .await
            .expect("the object-id counter must exist after `create_layer` allocated an id")
            .parse()
            .expect("the object-id counter must be numeric");
        let oid_str = oid.to_string();
        for (label, key) in [
            ("the object header", tv.key_schema().obj_header(&oid_str)),
            (
                "the sections index",
                tv.key_schema().obj_sections_index(&oid_str),
            ),
            ("the live stream", tv.key_schema().layer_stream(oid)),
            (
                "the layer mapping",
                tv.key_schema().layer_mapping(&session_id, name),
            ),
        ] {
            let exists: u64 = tv
                .client()
                .pool
                .exists(&key)
                .await
                .expect("EXISTS must succeed");
            assert_eq!(
                exists, 0,
                "{ctx}: {label} ({key}) must not exist: the read fails before `MULTI`, so a \
                 refused materialization writes nothing"
            );
        }

        err
    }

    /// Drive a DR restore over a clean keyspace with its `HGETALL obj:ext`
    /// armed to fail with `fault`, and return the checkpoint error.
    ///
    /// Staged as [`checkpoint_restore_wrongtype_ext_index_is_permanent_even_when_the_ext_read_fails_first`]
    /// is (create, assemble, wipe) but with nothing planted, so the probe
    /// passes and the read's failure is the only error. Asserts what holds for
    /// every fault: the restore is refused, the failure came from the read
    /// alone, and no header was written. The caller asserts the classification.
    async fn restore_refused_by_a_failed_ext_read(name: &str, fault: ReadFault) -> CheckpointError {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
        let (ls, queue) = store_with_ext_read_injector(&tv, &metrics);
        let session_id = create_session(&tv, name).await;
        let oid = ls
            .create_layer(
                &session_id,
                name,
                "b-zvd-user",
                &seeded_state(),
                mock_ext_store(),
            )
            .await
            .expect("create_layer must succeed");

        let (ompf, snap_seq, meta) = build_checkpoint_ompf(&tv, oid).await;
        crate::common::wipe_object_keys(&tv, oid).await;
        queue.lock().unwrap().push_back(fault.error());

        let err = apply_checkpoint(
            &ls,
            oid,
            ompf,
            snap_seq,
            &meta,
            None,
            &crate::common::test_config(),
        )
        .await
        .expect_err(
            "a failed `HGETALL obj:ext` must refuse the restore once the probe has passed. An \
             `Ok` means the read's error was dropped and the rebuild was planned against an \
             index that was never read",
        );

        let ctx = format!("checkpoint restore, {fault:?}");
        assert_read_failure_alone_was_reported(&queue, &metrics, &ctx);

        let header_exists: u64 = tv
            .client()
            .pool
            .exists(tv.key_schema().obj_header(&oid.to_string()))
            .await
            .expect("EXISTS must succeed");
        assert_eq!(
            header_exists, 0,
            "{ctx}: the object header must not exist: a refused restore writes nothing"
        );

        err
    }

    /// b.j2x: over a clean keyspace, a transient failure of CreateLayer's
    /// `HGETALL obj:ext` is `Unavailable` naming the read.
    ///
    /// `write_object` runs no retry loop, so the transient kind must reach the
    /// client as the retriable UNAVAILABLE, as the guards' own transient
    /// failures do.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_transient_ext_index_read_failure_is_unavailable_naming_the_read() {
        let fault = ReadFault::Timeout;
        let err =
            create_layer_refused_by_a_failed_ext_read("zvd-j2x-create-read-timeout", fault).await;

        let ServiceError::Unavailable(msg) = &err else {
            panic!(
                "a transient failure of the `HGETALL obj:ext` must be the retriable \
                 `Unavailable`: `write_object` runs no retry loop, so `Aborted` would promise a \
                 retry nobody performs and `Internal` would make a Valkey blip permanent; got \
                 {err:?}"
            );
        };
        let prefix = format!("{EXT_INDEX_HGETALL_OP} failed ({})", fault.kind().to_str());
        assert!(
            msg.starts_with(&prefix) && msg.contains(fault.injected_message()),
            "the refusal must name the read and carry fred's kind label and message, so an \
             operator can tell it from the guards beside it; expected a {prefix:?} prefix, got: \
             {msg:?}"
        );
    }

    /// b.j2x: over a clean keyspace, a non-transient failure of CreateLayer's
    /// `HGETALL obj:ext` is `Internal` naming the read, never the probe's
    /// `PermanentKeyspaceRefusal`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn create_layer_non_transient_ext_index_read_failure_is_internal_naming_the_read() {
        let fault = ReadFault::Protocol;
        let err =
            create_layer_refused_by_a_failed_ext_read("zvd-j2x-create-read-protocol", fault).await;

        let ServiceError::Internal(msg) = &err else {
            panic!(
                "a non-transient failure of the `HGETALL obj:ext` over a clean keyspace must be a \
                 plain `Internal`; a `PermanentKeyspaceRefusal` would blame a key that is not \
                 wrong-typed; got {err:?}"
            );
        };
        let prefix = format!("{EXT_INDEX_HGETALL_OP} failed: ");
        assert!(
            msg.starts_with(&prefix) && msg.contains(fault.injected_message()),
            "the refusal must name the read as the failing operation and carry fred's message; \
             expected a {prefix:?} prefix, got: {msg:?}"
        );
        assert!(
            !msg.contains(CREATE_KEYSPACE_TYPE_ERROR_PREFIX),
            "a failed read over a clean keyspace is not a wrong-typed key; got: {msg:?}"
        );
    }

    /// b.j2x: over a clean keyspace, a transient failure of the restore's
    /// `HGETALL obj:ext` reaches the recovery hook as
    /// `CheckpointError::Transient`, carrying the read's `Unavailable`.
    ///
    /// `write_restored_object` maps it to `Unavailable` (its doc table's
    /// transient row), which `write_object_err_to_checkpoint_err` folds to
    /// `Transient`, so the hook's next attempt re-runs the restore.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn checkpoint_restore_transient_ext_index_read_failure_is_transient() {
        let fault = ReadFault::Timeout;
        let err = restore_refused_by_a_failed_ext_read("zvd-j2x-restore-read-timeout", fault).await;

        assert!(
            matches!(err, CheckpointError::Transient { .. }),
            "a Valkey blip on the restore's `HGETALL obj:ext` must stay retriable; got {err:?}"
        );
        let rendered = err.to_string();
        let expected = format!(
            "unavailable: {EXT_INDEX_HGETALL_OP} failed ({})",
            fault.kind().to_str()
        );
        assert!(
            rendered.contains(&expected) && rendered.contains(fault.injected_message()),
            "the checkpoint error must carry the read's `Unavailable`, naming the read; expected \
             {expected:?} in {rendered:?}"
        );
    }

    /// b.j2x: over a clean keyspace, a non-transient failure of the restore's
    /// `HGETALL obj:ext` is `CheckpointError::Transient`, NOT `Permanent`.
    ///
    /// The read's `Internal` carries no structural verdict, and permanence
    /// travels only as the `PermanentKeyspaceRefusal` variant, so
    /// `write_object_err_to_checkpoint_err` folds it to `Transient`. Promoting
    /// it would abandon a restorable layer on a read fault the probe beside it
    /// found no keyspace cause for.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn checkpoint_restore_non_transient_ext_index_read_failure_is_transient_not_permanent() {
        let fault = ReadFault::Protocol;
        let err =
            restore_refused_by_a_failed_ext_read("zvd-j2x-restore-read-protocol", fault).await;

        assert!(
            matches!(err, CheckpointError::Transient { .. }),
            "a non-transient read fault over a clean keyspace must not be classified \
             `Permanent`: only the probe's structural refusal is; got {err:?}"
        );
        let rendered = err.to_string();
        let expected = format!("internal error: {EXT_INDEX_HGETALL_OP} failed: ");
        assert!(
            rendered.contains(&expected) && rendered.contains(fault.injected_message()),
            "the checkpoint error must carry the read's `Internal`, naming the read; expected \
             {expected:?} in {rendered:?}"
        );
    }
}
