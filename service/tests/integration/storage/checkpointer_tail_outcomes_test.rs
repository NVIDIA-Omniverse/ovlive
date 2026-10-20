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

// Plan b.dfh, Epic 3 (t1.dfh.57), Task 7, Subtask t3.dfh.57.1u.ne —
// PRD AC-2: the per-outcome matrix. One dedicated test per SR-8.2 capture
// outcome, over REAL Valkey streams and a REAL checkpoint store, so every arm
// of the outcome vocabulary is reached by a fixture that could actually occur
// in production rather than by a hand-built classifier input.
//
// ── How every arm is driven, and why NOT through the periodic loop ──────────
//
// This module calls Task 3's capture unit DIRECTLY —
// `live_edit::checkpoint::tail_capture::capture_layer_tail(client, keys,
// object_id, store, replica_id, cfg)` — against a live `TestValkey` stream and
// an `InMemoryCheckpointStore`. Same route `tests/integration/checkpoint/
// assembler_test.rs` takes to `assemble_layer_checkpoint`, and it is a
// requirement, not a convenience:
//
//   - The loop's capture fallback is reachable only after the assembler LOSES
//     its stable-cut proof (`assembler.rs`), and `run_one_tick` is private, so
//     a spawned loop is the only wiring path — inherently wall-clock-bound and
//     racy.
//   - Worse, the terminal planted blockers below (the `*`-ID `deleted`
//     sentinel, the ascending jump) make every SUBSEQUENT commit fail: the
//     writer dies, the layer quiesces, the assembler then PROVES its cut, and
//     the fallback is never reached at all. Those arms are structurally
//     unreachable through the loop as fixtured.
//
// Driving the unit directly is what makes these fixtures hand-planted,
// deterministic and wall-clock-free (no `sleep`, no polling, no deadline).
//
// ── What this module deliberately does NOT assert ───────────────────────────
//
// The capture unit is metrics-free and cursor-free BY DESIGN (its module
// header): it decides and persists, and the tick wiring maps the decision onto
// `live_edit_checkpoint_tail_total` and stamps the `obj:hdr` cursor columns.
// So there is nothing here to read a metric cell or a cursor column off of, and
// asserting either would only assert the fixture. The assertion surface is:
//
//   1. `TailCaptureDecision::outcome_label()` — equal to exactly this arm's
//      string and to no other member of `CHECKPOINT_TAIL_OUTCOMES`;
//   2. the precise `TailCaptureRefusal` variant (two arms of THIS UNIT share
//      the `refused_seq_gap` label and have different operator remediations;
//      a third producer of that label lives loop-side and is out of reach
//      here — see `outstanding_restore_seq_bump_refuses_seq_gap_before_the_stream_read`);
//   3. adapter state — artifact present/absent, and for a capture the decoded
//      envelope's span and record kinds.
//
// The outcome→metric-cell and outcome→cursor-stamp mappings are pinned ONCE,
// loop-driven, in the sibling cadence Subtask (`t3.dfh.57.1u.dg`,
// `checkpointer_tail_cadence_test.rs`).
//
// ── THE OMPF TRAP (b.pw6) — read before adding a commit here ────────────────
//
// `LayerStore::update_layer` takes `relay_ompf: Option<Bytes>`, and the stream
// entry carries an `ompf` field IFF it is non-empty `Some`
// (`src/storage/notifications.rs`, `RELAY_OMPF_FIELD`). The tail classifier
// reads an `update`-typed `{seq}-0` entry with NO `ompf` field as an EMPTY
// COMMIT BY INVARIANT. `update_layer` refuses a delta without non-empty relay
// bytes before any store I/O, so a `Some(delta), None` pair can no longer
// record real content as a replayable no-op.
//
// In THIS module: every state-changing commit passes
// `Some(Bytes::from(delta_to_ompf_bytes(&delta, &[])))`, the writer's bytes the
// capture assertions compare against. `None` is RESERVED for the empty commit,
// which passes no delta either: the truthful §8.4 shape.
//
// ── THE FIXTURE PRINCIPLE that makes a refusal reachable ────────────────────
//
// SR-1.3: a refusal fires ONLY on an EMPTY sound prefix. The capture's read
// window opens at `{S0+1}-0` where S0 is the newest generation in the STORE
// LISTING. So a refusal fixture must place its blocking condition FIRST in that
// window — which means hand-putting the lineage-head base at
// `snap_seq == the newest committed sequence`, THEN planting the blocker above
// it. Get it backwards and the capture succeeds on the sound prefix below the
// blocker, the refusal never fires, and the test is green while proving
// nothing. That is the most likely silent failure in this module, so each
// refusal test names its floor explicitly.
//
// The `refused_trimmed` and ascending-jump arms are the two deliberate
// exceptions: a trim REMOVES the window floor, and the jump arm exists
// precisely to prove that a non-empty accumulated prefix is discarded too.
//
// ── `refused_missing_bytes`: SCOPED no-producer banner ──────────────────────
//
// `refused_missing_bytes` folds two distinct conditions, and their integration
// reachability differs — the claim is scoped accordingly:
//
//   - The OVERSIZE-FIRST-ENTRY fold IS reachable from an integration fixture:
//     one commit whose verbatim OMPF alone exceeds the per-tail byte budget is
//     excluded whole, the slice comes back empty with `cap_hit == Some(Bytes)`,
//     and the classifier must say `missing_bytes` (NOT `trimmed` — the
//     remediation is `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES`, not the stream
//     retention knob). Covered here by
//     `oversize_first_commit_refuses_missing_bytes_not_trimmed`.
//   - The NO-REPLAYABLE-FORM arm — an undecodable entry, a zero-length `ompf`
//     field, an `update` at a non-`-0` auto ID, a stream-ID/payload sequence
//     disagreement — has NO deployed producer: every shipped writer emits a
//     well-formed entry, so reaching it from an integration fixture would mean
//     hand-planting a stream shape no writer can write. It is a
//     defense-in-depth arm, and its arm-level coverage home is Task 2's unit
//     suite (`t3.dfh.57.oz.5k`, `src/checkpoint/tail_classify.rs`'s in-file
//     lane), which drives the classifier over exactly those shapes.
//
// ── Why the `deleted` sentinel is PLANTED, not produced ─────────────────────
//
// Both production sentinel writers — `delete_layer` (`src/storage/layer.rs`)
// and the `CreateLayer`-overwrite path — retire the object's `obj:*` keys in
// the SAME cascade that appends the sentinel. A capture over a sentinel-topped
// stream would therefore find no live `obj:hdr` and return `NotFound` long
// before the classifier ran, so neither writer can produce this arm. The
// planted entry is byte-identical to what both emit (`XADD <stream> * payload
// deleted`, `DELETED_STREAM_PAYLOAD`). Its auto `*` ID is astronomically above
// any `{seq}-0`, which is SR-1.2's "terminal" property: it must be planted LAST,
// because afterwards Valkey rejects every further explicit `{seq}-0` XADD.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, StreamsInterface};
use fred::types::streams::XCap;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::tail_capture::{
    capture_layer_tail, TailCaptureDecision, TailCaptureRefusal,
};
use live_edit::checkpoint::tail_classify::TailRefusal;
use live_edit::checkpoint::{
    decode_tail_envelope, CheckpointHandle, CheckpointKind, CheckpointStore, TailCaptureConfig,
    TailEnvelope, TailRecordKind,
};
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section};
use live_edit::observability::metrics::CHECKPOINT_TAIL_OUTCOMES;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::notifications::{
    DELETED_STREAM_PAYLOAD, NOTIFICATION_PAYLOAD_FIELD, RELAY_OMPF_FIELD,
};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{OwnedDelta, SetField};

use crate::checkpoint::helpers::{put_base_generation, put_tail_generation};
use crate::common::owned_delta::delta_to_ompf_bytes;

// ── Constants ───────────────────────────────────────────────────────────────

const REPLICA: &str = "tail-outcomes-replica";
const CREATED_BY: &str = "tail-outcomes-user";

/// The sequence every fixture layer is CREATED at, and therefore the sequence
/// its first real commit lands at + 1.
///
/// Deliberately far above 1 so that `CreateLayer`'s ompf-less `1-0` seed entry
/// sits well BELOW every read window this module opens. That entry accompanies
/// the full initial layer content while carrying no `ompf`, so a window that
/// reached it would record the whole layer as an empty commit — the exact
/// corruption the capture unit's `S0 >= 1` guard exists to prevent.
const SEED_SEQ: u64 = 50;

/// Section id every commit's `SetField` targets. Created by [`seed_state`].
const SECTION_ID: u64 = 1;

/// `server_ts` (microseconds) stamped into hand-planted stream entries.
/// A constant, not a clock read: this module is wall-clock-free.
const PLANTED_SERVER_TS: i64 = 1_700_000_000_000_000;

/// Payload for the oversize commit. 4 KiB of field value puts the encoded OMPF
/// far above the 135-byte payload budget the tiny-cap config below derives,
/// and far below `TestValkey`'s 1 MiB `max_layer_size`.
const OVERSIZE_VALUE_BYTES: usize = 4096;

// ── Fixture ─────────────────────────────────────────────────────────────────

/// One layer on a live Valkey plus the checkpoint store its lineage lives in.
///
/// Every test builds its own — `TestValkey` namespaces its whole keyspace under
/// a per-instance `test_{uuid}:` prefix, and the `InMemoryCheckpointStore` is
/// per-fixture, so the tests are mutually isolated and safe to run in parallel.
struct TailFixture {
    tv: TestValkey,
    store: Arc<dyn CheckpointStore>,
    layer_store: LayerStore,
    session_id: String,
    layer_name: String,
    object_id: u64,
}

impl TailFixture {
    /// Create a session and a layer at [`SEED_SEQ`], ready for commits, on a
    /// store with the production relay byte budget.
    async fn new(layer_name: &str) -> Self {
        Self::new_with_budget(
            layer_name,
            live_edit::config::LAYER_STREAM_MAX_BYTES_DEFAULT,
        )
        .await
    }

    /// [`Self::new`] on a store whose relay byte budget
    /// (`LIVE_EDIT_LAYER_STREAM_MAX_BYTES`) is `budget` bytes.
    async fn new_with_budget(layer_name: &str, budget: u64) -> Self {
        let tv = TestValkey::setup().await;
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create("tail-outcomes-session", "/stage", CREATED_BY)
            .await
            .expect("session creation must succeed");

        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_layer_stream_max_bytes(budget);
        let object_id = layer_store
            .create_layer(
                &session_id,
                layer_name,
                CREATED_BY,
                &seed_state(),
                Arc::new(MockExternalValueStore::new()),
            )
            .await
            .expect("create_layer must succeed");

        Self {
            tv,
            store,
            layer_store,
            session_id,
            layer_name: layer_name.to_string(),
            object_id,
        }
    }

    /// A state-changing commit carrying REAL relay OMPF — the only shape a
    /// non-empty commit may take in this module (see the OMPF-trap banner).
    ///
    /// Returns `(new_sequence, the exact ompf bytes written)` so a caller can
    /// assert byte identity against the decoded record.
    async fn commit(&self, nonce: u64) -> (u64, Bytes) {
        self.commit_sized(nonce, 8).await
    }

    /// [`Self::commit`] with a caller-chosen field-value width, for the byte-cap
    /// fixture.
    async fn commit_sized(&self, nonce: u64, value_bytes: usize) -> (u64, Bytes) {
        let mut value = nonce.to_le_bytes().to_vec();
        value.resize(value_bytes.max(8), 0xAB);
        let delta = OwnedDelta {
            set_fields: vec![SetField {
                section_id: SECTION_ID,
                key_name: "tail-outcome-field".to_string(),
                value_or_ext_hash: value,
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: nonce,
            }],
            ..Default::default()
        };
        let ompf = Bytes::from(delta_to_ompf_bytes(&delta, &[]));
        let (seq, _server_ts) = self
            .layer_store
            .update_layer(
                &self.session_id,
                &self.layer_name,
                self.object_id,
                Some(delta),
                // REAL relay bytes — never `None` for a state-changing commit.
                Some(ompf.clone()),
                "tail-outcomes-writer",
                0,
            )
            .await
            .unwrap_or_else(|e| panic!("commit {nonce} must succeed: {e}"));
        (seq, ompf)
    }

    /// The §8.4 empty commit: no delta and no relay bytes — what the
    /// `UpdateLayer` handler passes for an empty request body, and the ONE
    /// shape for which a `None` relay is truthful (a delta without relay
    /// bytes is refused before any store I/O). It rides the Lua fast path and
    /// writes an entry with no `ompf` field.
    async fn commit_empty(&self) -> u64 {
        let (seq, _server_ts) = self
            .layer_store
            .update_layer(
                &self.session_id,
                &self.layer_name,
                self.object_id,
                None,
                None,
                "tail-outcomes-writer",
                0,
            )
            .await
            .expect("empty-delta commit must succeed");
        seq
    }

    async fn put_base(&self, snap_seq: u64) -> CheckpointHandle {
        put_base_generation(
            &self.store,
            self.object_id,
            &self.session_id,
            &self.layer_name,
            snap_seq,
        )
        .await
    }

    async fn put_tail(&self, base_snap_seq: u64, from_seq: u64, to_seq: u64) -> CheckpointHandle {
        put_tail_generation(
            &self.store,
            self.object_id,
            &self.session_id,
            &self.layer_name,
            base_snap_seq,
            from_seq,
            to_seq,
        )
        .await
    }

    fn stream_key(&self) -> String {
        self.tv.key_schema().layer_stream(self.object_id)
    }

    /// Invoke the capture unit under test.
    async fn capture(&self, cfg: &TailCaptureConfig) -> TailCaptureDecision {
        capture_layer_tail(
            self.tv.client(),
            self.tv.key_schema(),
            self.object_id,
            self.store.as_ref(),
            REPLICA,
            cfg,
        )
        .await
        .expect("capture must not error; every arm under test is a decision, not a failure")
    }

    async fn listing(&self) -> Vec<CheckpointHandle> {
        self.store
            .list_layer_checkpoints(self.object_id)
            .await
            .expect("listing the lineage must succeed")
    }

    /// Fetch and decode the tail envelope a capture persisted.
    async fn decode(&self, handle: &CheckpointHandle) -> TailEnvelope {
        let blob = self
            .store
            .get_layer_checkpoint(handle)
            .await
            .expect("fetching the persisted artifact must succeed")
            .expect("the capture's handle must resolve to a stored artifact");
        assert_eq!(
            blob.meta.kind,
            CheckpointKind::Tail,
            "a capture must persist a Tail generation"
        );
        decode_tail_envelope(&blob.bytes).expect("the persisted envelope must decode")
    }

    /// Every stream entry at or above `{from_seq}-0`, in stream order.
    ///
    /// Field VALUES decode as `Bytes`, never `String`: the per-object stream
    /// carries a BINARY `ompf` field beside the UTF-8 `payload` one, and a
    /// `String` decode would fail the whole read on that field's invalid UTF-8.
    async fn stream_entries_from(&self, from_seq: u64) -> Vec<(String, HashMap<String, Bytes>)> {
        self.tv
            .client()
            .pool
            .xrange_values(
                &self.stream_key(),
                format!("{from_seq}-0"),
                "+",
                Option::<u64>::None,
            )
            .await
            .expect("XRANGE over the layer stream must succeed")
    }

    /// Append an `update`-typed entry at an explicit `{seq}-0` id, out of band.
    ///
    /// The payload is byte-identical in shape to what both commit paths write
    /// (`update|oid|seq|cid|cts|sts`, `src/storage/notifications.rs`), and it
    /// carries real `ompf` bytes so the entry is a well-formed COMMIT — the
    /// refusal under test must come from the ID progression, not from a
    /// malformed entry falling into `missing_bytes`.
    async fn plant_update_entry(&self, seq: u64) {
        let oid = self.object_id;
        let delta = OwnedDelta {
            set_fields: vec![SetField {
                section_id: SECTION_ID,
                key_name: "planted-field".to_string(),
                value_or_ext_hash: seq.to_le_bytes().to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: seq,
            }],
            ..Default::default()
        };
        let ompf = Bytes::from(delta_to_ompf_bytes(&delta, &[]));
        let fields: Vec<(&str, fred::types::Value)> = vec![
            (
                NOTIFICATION_PAYLOAD_FIELD,
                format!("update|{oid}|{seq}||0|{PLANTED_SERVER_TS}").into(),
            ),
            (RELAY_OMPF_FIELD, fred::types::Value::Bytes(ompf)),
        ];
        let entry_id = format!("{seq}-0");
        let _: String = self
            .tv
            .client()
            .pool
            .xadd(&self.stream_key(), false, None, entry_id.as_str(), fields)
            .await
            .unwrap_or_else(|e| panic!("planting the entry at {seq}-0 must succeed: {e}"));
    }

    /// Append the layer-retirement sentinel exactly as both production writers
    /// do: auto id, one `payload = deleted` field, nothing else.
    async fn plant_deleted_sentinel(&self) {
        let fields: Vec<(&str, fred::types::Value)> = vec![(
            NOTIFICATION_PAYLOAD_FIELD,
            DELETED_STREAM_PAYLOAD.to_string().into(),
        )];
        let _: String = self
            .tv
            .client()
            .pool
            .xadd(&self.stream_key(), false, None, "*", fields)
            .await
            .expect("planting the deleted sentinel must succeed");
    }

    /// Deterministic head trim (`XTRIM MINID =`), not the approximate `MAXLEN ~`
    /// every XADD applies. Precedent:
    /// `tests/integration/grpc/subscribe_regimes_test.rs`.
    async fn trim_minid(&self, min_seq: u64) {
        let floor = format!("{min_seq}-0");
        let cap: XCap = XCap::try_from(("MINID", "=", floor.as_str()))
            .expect("building the MINID xcap must succeed");
        let _: () = self
            .tv
            .client()
            .pool
            .xtrim(&self.stream_key(), cap)
            .await
            .expect("XTRIM MINID must succeed");
    }

    /// Stamp `obj:hdr restore_seq_bump` — the column a DR restore writes
    /// (`src/checkpoint/apply.rs`) and a post-restore generation clears
    /// (`src/checkpoint/checkpointer.rs`).
    async fn set_restore_seq_bump(&self, value: u64) {
        let key = self.tv.key_schema().obj_header(&self.object_id.to_string());
        let _: () = self
            .tv
            .client()
            .pool
            .hset(&key, [("restore_seq_bump", value.to_string())])
            .await
            .expect("HSET obj:hdr restore_seq_bump must succeed");
    }
}

/// Two sections under root, one seed field each. Enough that `SECTION_ID`
/// exists for every commit's `SetField` to land on.
fn seed_state() -> ObjectState {
    let mut state = ObjectState {
        sequence_num: SEED_SEQ,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };
    let ids = vec![SECTION_ID, SECTION_ID + 1];
    for (order, sid) in ids.iter().enumerate() {
        state.sections.insert(
            *sid,
            Section {
                section_id: *sid,
                parent_id: 0,
                name: format!("tail-outcomes-{sid}"),
                section_type: 1,
                section_order: order as u32,
                first_version: 1,
                last_version: 1,
            },
        );
        state.fields.insert(
            (*sid, "seed-field".to_string()),
            Field {
                section_id: *sid,
                key_name: "seed-field".to_string(),
                value: FieldValue::Inline(vec![0x5E, 0xED]),
                set_order: 0,
                authoring_set_order: 0,
                version: 1,
            },
        );
    }
    state.children.insert(0, ids);
    state
}

// ── Config fixtures ─────────────────────────────────────────────────────────

/// Stock knobs with `enabled` PINNED `false` everywhere in this module on
/// purpose: the capture unit is flag-agnostic (the gate lives in the tick
/// wiring), and every capture below succeeding under `enabled: false` is the
/// proof.
///
/// Pinned rather than inherited since b.3n1 flipped
/// `TailCaptureConfig::default()`'s `enabled` to `true`. Inheriting it would
/// destroy the proof silently: a capture succeeding under the flag it would
/// normally be gated by demonstrates nothing about flag-agnosticism, and the
/// module's whole claim would have evaporated with every test still green.
fn cfg() -> TailCaptureConfig {
    TailCaptureConfig {
        enabled: false,
        ..TailCaptureConfig::default()
    }
}

fn cfg_with_chain(max_chain: usize) -> TailCaptureConfig {
    TailCaptureConfig { max_chain, ..cfg() }
}

fn cfg_with_entries(max_entries: u64) -> TailCaptureConfig {
    TailCaptureConfig {
        max_entries,
        ..cfg()
    }
}

/// A byte cap small enough that a 4 KiB commit cannot fit, yet large enough to
/// clear the unit's own pathological-config refusal.
///
/// The unit reserves envelope framing before handing the slice primitive a
/// payload budget: `max_entries = min(1, (200 - 44) / 21) = 1`, `reserve = 44 +
/// 21 = 65`, `max_payload_bytes = 200 - 65 = 135`. 135 > 0, so the config is
/// admissible (it does NOT take the zero-budget arm) and the refusal under test
/// genuinely comes from the FIRST ENTRY being oversize.
fn cfg_tiny_bytes() -> TailCaptureConfig {
    TailCaptureConfig {
        max_entries: 1,
        max_bytes: 200,
        ..cfg()
    }
}

// ── Assertion helpers ───────────────────────────────────────────────────────

/// Assert the decision's `outcome` label is exactly `expected` — and that it is
/// a member of the pinned vocabulary, matching that one member and no other.
///
/// The vocabulary filter is what makes this more than an equality check: a
/// label that drifted out of `CHECKPOINT_TAIL_OUTCOMES` entirely would
/// materialise its metric series only on first occurrence, which for a refusal
/// outcome is indistinguishable from a feature that was never wired.
#[track_caller]
fn assert_outcome(decision: &TailCaptureDecision, expected: &str) {
    let label = decision.outcome_label();
    let matched: Vec<&str> = CHECKPOINT_TAIL_OUTCOMES
        .iter()
        .copied()
        .filter(|member| *member == label)
        .collect();
    assert_eq!(
        matched,
        vec![expected],
        "outcome_label() = {label:?} must match exactly one member of CHECKPOINT_TAIL_OUTCOMES \
         and that member must be {expected:?}; decision was {decision:?}"
    );
}

/// Assert the decision is exactly this refusal variant.
///
/// The variant, not merely "not a capture", and not merely the label: two arms
/// (`Classified(SeqGap)` and `PostRestoreUnrebased`) deliberately share the
/// `refused_seq_gap` label while having different operator remediations, and
/// the enum variant is where that distinction is carried.
#[track_caller]
fn assert_refused(decision: &TailCaptureDecision, expected: TailCaptureRefusal) {
    match decision {
        TailCaptureDecision::Refused(refusal) => assert_eq!(
            *refusal, expected,
            "wrong refusal arm — each arm has a different operator remediation"
        ),
        TailCaptureDecision::Captured {
            from_seq, to_seq, ..
        } => panic!(
            "expected Refused({expected:?}), got a capture over ({from_seq}, {to_seq}] — the \
             fixture's blocking condition is NOT the first entry in the read window, so the \
             capture succeeded on a sound prefix below it (see the fixture-principle banner)"
        ),
    }
}

/// Destructure a capture, or fail naming the refusal.
#[track_caller]
fn captured_parts(decision: &TailCaptureDecision) -> (&CheckpointHandle, u64, u64, usize, bool) {
    match decision {
        TailCaptureDecision::Captured {
            handle,
            from_seq,
            to_seq,
            record_count,
            capped,
        } => (handle, *from_seq, *to_seq, *record_count, *capped),
        TailCaptureDecision::Refused(refusal) => {
            panic!("expected a capture, got Refused({refusal:?})")
        }
    }
}

fn addresses(handles: &[CheckpointHandle]) -> Vec<(u64, u64, CheckpointKind)> {
    handles
        .iter()
        .map(|h| (h.object_id, h.snap_seq, h.kind))
        .collect()
}

/// SR-1.3's zero-record-never-PUT half, as observable from outside the unit:
/// the lineage is byte-for-byte the one the fixture seeded, and it holds no
/// tail.
///
/// (The cursor half of the original criterion — "neither cursor column
/// stamped" — is the tick wiring's, not this unit's: the capture unit never
/// writes `obj:hdr`. It is pinned loop-driven in `t3.dfh.57.1u.dg`.)
#[track_caller]
fn assert_nothing_persisted(before: &[CheckpointHandle], after: &[CheckpointHandle]) {
    let before_addresses = addresses(before);
    let after_addresses = addresses(after);
    // Stated as "no tail the fixture did not already seed" rather than "no tail
    // at all": the chain-bound fixture seeds tails on purpose, and a
    // no-tail-at-all assertion would fail there for the wrong reason while
    // saying nothing about whether the CAPTURE wrote one.
    let new_tails: Vec<&(u64, u64, CheckpointKind)> = after_addresses
        .iter()
        .filter(|address| address.2 == CheckpointKind::Tail && !before_addresses.contains(address))
        .collect();
    assert!(
        new_tails.is_empty(),
        "a refusal must persist NO new tail generation; the capture added {new_tails:?} \
         (listing before: {before_addresses:?}, after: {after_addresses:?})"
    );
    assert_eq!(
        after_addresses, before_addresses,
        "a refusal must leave the lineage exactly as it found it"
    );
}

/// The b.pw6 integration-level guard: every `update`-typed `{seq}-0` entry the
/// read window will see carries the `ompf` field.
///
/// Asserted BEFORE the decoded record kinds in the `captured` test, because an
/// ompf-less non-empty commit decodes as an `EmptyDelta` no-op over real
/// content — a green record-count assertion sitting on top of silent
/// corruption. This is the invariant at the STREAM, one layer below the
/// codec-level coverage in Task 2's suite.
#[track_caller]
fn assert_every_commit_carries_ompf(entries: &[(String, HashMap<String, Bytes>)]) {
    let mut checked = 0usize;
    for (id, fields) in entries {
        let payload = fields
            .get(NOTIFICATION_PAYLOAD_FIELD)
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_default();
        if !payload.starts_with("update|") || !id.ends_with("-0") {
            continue;
        }
        let ompf = fields.get(RELAY_OMPF_FIELD);
        assert!(
            ompf.is_some_and(|b| !b.is_empty()),
            "stream entry {id} is an update-typed commit with no (or empty) `{RELAY_OMPF_FIELD}` \
             field. The classifier reads that as an EMPTY COMMIT by invariant, so this commit \
             would be captured as a replayable no-op over real content (b.pw6). Every \
             state-changing commit in this module must pass Some(relay_ompf)"
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "no update-typed {{seq}}-0 entries in the read window — the fixture wrote nothing the \
         capture could see, so this guard would have passed vacuously"
    );
}

// ── captured ────────────────────────────────────────────────────────────────

/// A sound prefix over real ompf-bearing commits captures, and the persisted
/// envelope's CLAIMED range equals its CONTENT — the b.pw6 guard, stated as an
/// equality between three independently-derived numbers (the decision's span,
/// the header's span, the record count).
#[tokio::test]
async fn sound_prefix_captures_a_tail_whose_claimed_range_equals_its_content() {
    let fx = TailFixture::new("captured.live").await;

    // Four real commits at SEED+1 .. SEED+4.
    let mut written: Vec<(u64, Bytes)> = Vec::new();
    for nonce in 0..4u64 {
        written.push(fx.commit(nonce).await);
    }
    assert_eq!(
        written.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
        (SEED_SEQ + 1..=SEED_SEQ + 4).collect::<Vec<_>>(),
        "commits must land at contiguous {{SEED+n}} sequences"
    );

    // The lineage head at the layer's creation sequence, so the window opens on
    // exactly the first real commit.
    fx.put_base(SEED_SEQ).await;

    // PM condition 4 — the ompf invariant, asserted at the stream, before any
    // record-kind claim is made.
    assert_every_commit_carries_ompf(&fx.stream_entries_from(SEED_SEQ + 1).await);

    let decision = fx.capture(&cfg()).await;
    assert_outcome(&decision, "captured");
    let (handle, from_seq, to_seq, record_count, capped) = captured_parts(&decision);
    assert_eq!(from_seq, SEED_SEQ, "S0 is the lineage head");
    assert_eq!(to_seq, SEED_SEQ + 4, "to_seq is the last covered commit");
    assert_eq!(record_count, 4);
    assert!(!capped, "no cap was configured low enough to truncate");
    assert_eq!(handle.kind, CheckpointKind::Tail);
    assert_eq!(
        handle.snap_seq, to_seq,
        "a tail's ordering sequence IS its to_seq"
    );

    let envelope = fx.decode(handle).await;
    assert_eq!(envelope.header().object_id, fx.object_id);
    assert_eq!(envelope.header().base_snap_seq, SEED_SEQ);
    assert_eq!(
        envelope.header().from_seq,
        SEED_SEQ,
        "the envelope header's from_seq is the EXCLUSIVE lower bound (S0 itself)"
    );
    assert_eq!(envelope.header().to_seq, SEED_SEQ + 4);
    assert_eq!(
        envelope.len(),
        (envelope.header().to_seq - envelope.header().from_seq) as usize,
        "claimed range MUST equal content: the record count is the span"
    );

    for (index, record) in envelope.iter().enumerate() {
        let (seq, ompf) = &written[index];
        assert_eq!(record.kind(), TailRecordKind::Verbatim);
        assert_eq!(record.sequence_num(), *seq);
        assert_eq!(
            record.ompf(),
            Some(ompf),
            "SR-2.2 stores the writer's raw inbound container with no re-encode at the capture \
             boundary — record {index} must be byte-identical to what the writer supplied"
        );
    }
}

// ── empty-delta commits as replayable records ───────────────────────────────

/// §8.4 empty-delta commits interleaved with ompf-bearing ones are captured as
/// replayable records, with the record KINDS alternating in commit order.
///
/// An empty commit is not a hole: it advances the sequence, so dropping it (or
/// mis-recording it as `Verbatim` with no payload) would break SR-2.1's
/// contiguity at replay.
#[tokio::test]
async fn empty_delta_commits_are_captured_as_records_with_alternating_kinds() {
    let fx = TailFixture::new("empty-delta.live").await;

    // Verbatim, empty, verbatim, empty, verbatim — SEED+1 .. SEED+5.
    let (s1, ompf1) = fx.commit(1).await;
    let s2 = fx.commit_empty().await;
    let (s3, ompf3) = fx.commit(3).await;
    let s4 = fx.commit_empty().await;
    let (s5, ompf5) = fx.commit(5).await;
    assert_eq!(
        vec![s1, s2, s3, s4, s5],
        (SEED_SEQ + 1..=SEED_SEQ + 5).collect::<Vec<_>>(),
        "an empty-delta commit still advances the sequence"
    );

    fx.put_base(SEED_SEQ).await;

    let decision = fx.capture(&cfg()).await;
    assert_outcome(&decision, "captured");
    let (handle, _from, to_seq, record_count, _capped) = captured_parts(&decision);
    assert_eq!(to_seq, SEED_SEQ + 5);
    assert_eq!(record_count, 5);

    let envelope = fx.decode(handle).await;
    let kinds: Vec<TailRecordKind> = envelope.iter().map(|r| r.kind()).collect();
    assert_eq!(
        kinds,
        vec![
            TailRecordKind::Verbatim,
            TailRecordKind::EmptyDelta,
            TailRecordKind::Verbatim,
            TailRecordKind::EmptyDelta,
            TailRecordKind::Verbatim,
        ],
        "record kinds must alternate in commit order — an ompf-less update entry IS the empty \
         shape, and an ompf-bearing one is never that"
    );
    let seqs: Vec<u64> = envelope.iter().map(|r| r.sequence_num()).collect();
    assert_eq!(seqs, vec![s1, s2, s3, s4, s5]);

    for (index, expected) in [(0usize, &ompf1), (2, &ompf3), (4, &ompf5)] {
        assert_eq!(
            envelope.records()[index].ompf(),
            Some(expected),
            "verbatim record {index} must carry the writer's exact bytes"
        );
    }
    for index in [1usize, 3] {
        assert!(
            envelope.records()[index].ompf().is_none(),
            "an empty-delta record carries no payload"
        );
    }
}

// ── captured_capped ─────────────────────────────────────────────────────────

/// The entry cap truncates the prefix, and the artifact carries the CAPPED
/// prefix with `to_seq` matching its content exactly — a capped tail is still
/// a tail whose claimed range equals what it holds.
#[tokio::test]
async fn entry_cap_yields_captured_capped_over_the_truncated_prefix() {
    let fx = TailFixture::new("capped.live").await;

    // Six commits, cap at three: the cap must fire with entries still pending,
    // which is what makes `cap_hit` a genuine truncation rather than a stop
    // that happened to land on the end of the stream.
    for nonce in 0..6u64 {
        fx.commit(nonce).await;
    }
    fx.put_base(SEED_SEQ).await;

    let decision = fx.capture(&cfg_with_entries(3)).await;
    assert_outcome(&decision, "captured_capped");
    let (handle, from_seq, to_seq, record_count, capped) = captured_parts(&decision);
    assert!(capped, "the entry cap truncated the read");
    assert_eq!(from_seq, SEED_SEQ);
    assert_eq!(
        to_seq,
        SEED_SEQ + 3,
        "to_seq must name the last CAPTURED commit, not the newest committed one"
    );
    assert_eq!(record_count, 3);

    let envelope = fx.decode(handle).await;
    assert_eq!(envelope.header().to_seq, SEED_SEQ + 3);
    assert_eq!(
        envelope.len(),
        3,
        "the capped artifact's content must equal its claimed range"
    );
    assert_eq!(
        envelope
            .iter()
            .map(|r| r.sequence_num())
            .collect::<Vec<_>>(),
        vec![SEED_SEQ + 1, SEED_SEQ + 2, SEED_SEQ + 3],
        "the captured prefix is the OLDEST three commits, not the newest three"
    );
}

// ── refused_trimmed ─────────────────────────────────────────────────────────

/// Retention trimmed the stream past the window floor, so the first entry the
/// window sees is not `{S0+1}-0` — an empty sound prefix, `refused_trimmed`,
/// nothing persisted.
///
/// The trim is `XTRIM MINID =`, not the approximate `MAXLEN ~` every XADD
/// applies: `~` is explicitly allowed to leave entries above the bound, so a
/// MAXLEN-driven fixture would be nondeterministic about which entry is first.
#[tokio::test]
async fn head_trim_past_the_window_floor_refuses_trimmed_and_persists_nothing() {
    let fx = TailFixture::new("trimmed.live").await;

    for nonce in 0..5u64 {
        fx.commit(nonce).await;
    }
    fx.put_base(SEED_SEQ).await;
    let before = fx.listing().await;

    // Window floor is {SEED+1}-0; trim it (and SEED+2) away.
    fx.trim_minid(SEED_SEQ + 3).await;
    let survivors = fx.stream_entries_from(SEED_SEQ + 1).await;
    let expected_first = format!("{}-0", SEED_SEQ + 3);
    assert_eq!(
        survivors.first().map(|(id, _)| id.as_str()),
        Some(expected_first.as_str()),
        "the trim must leave a first entry ABOVE the window floor; without that the capture \
         succeeds and this test proves nothing"
    );

    let decision = fx.capture(&cfg()).await;
    assert_outcome(&decision, "refused_trimmed");
    assert_refused(
        &decision,
        TailCaptureRefusal::Classified(TailRefusal::Trimmed),
    );
    assert_nothing_persisted(&before, &fx.listing().await);
}

// ── refused_sentinel ────────────────────────────────────────────────────────

/// A layer-retirement sentinel is the first entry in the window, so there is no
/// sound prefix before it: `refused_sentinel`, nothing persisted.
///
/// Fixture principle in force: the base sits at the NEWEST committed sequence,
/// so the window opens ABOVE every real commit and the sentinel — whose auto id
/// is far above any `{seq}-0` — is the window's first entry. With the base
/// lower, the capture would succeed on the commits below the sentinel and this
/// arm would never fire.
///
/// Planted, not produced — see the module banner for why neither production
/// sentinel writer can reach this state, and why the plant must be LAST.
#[tokio::test]
async fn deleted_sentinel_at_the_window_floor_refuses_sentinel_and_persists_nothing() {
    let fx = TailFixture::new("sentinel.live").await;

    let mut newest = SEED_SEQ;
    for nonce in 0..3u64 {
        newest = fx.commit(nonce).await.0;
    }
    // The lineage head AT the newest committed sequence — the fixture principle.
    fx.put_base(newest).await;
    let before = fx.listing().await;

    // LAST: after this, Valkey rejects every further explicit {seq}-0 XADD.
    fx.plant_deleted_sentinel().await;

    let decision = fx.capture(&cfg()).await;
    assert_outcome(&decision, "refused_sentinel");
    assert_refused(
        &decision,
        TailCaptureRefusal::Classified(TailRefusal::Sentinel),
    );
    assert_nothing_persisted(&before, &fx.listing().await);
}

// ── refused_seq_gap, producer (i): the ascending jump ───────────────────────

/// An ascending jump in the `+1` progression refuses EVERYTHING — including the
/// sound, non-empty prefix accumulated below it.
///
/// That asymmetry is SR-1.2's, and it is the whole point of this test: entries
/// below an ascending jump are fossils of an abandoned pre-restore timeline, so
/// composing the prefix would mint a generation whose restore resurrects state
/// the live timeline has already superseded. A `Captured` over
/// `(SEED, SEED+3]` here would be the regression.
#[tokio::test]
async fn ascending_jump_refuses_seq_gap_discarding_the_accumulated_prefix() {
    let fx = TailFixture::new("seq-gap-jump.live").await;

    for nonce in 0..3u64 {
        fx.commit(nonce).await;
    }
    // Base LOW on purpose (contrast with every other refusal fixture): the
    // window must contain the three real commits so the discarded prefix is
    // genuinely non-empty.
    fx.put_base(SEED_SEQ).await;
    let before = fx.listing().await;

    // Terminal: after an explicit id this far ahead, no further {seq}-0 XADD in
    // the SEED+4..SEED+8 range is accepted.
    fx.plant_update_entry(SEED_SEQ + 9).await;

    let window = fx.stream_entries_from(SEED_SEQ + 1).await;
    assert_eq!(
        window.len(),
        4,
        "the window must hold three sound commits and the jumped entry"
    );

    let decision = fx.capture(&cfg()).await;
    assert_outcome(&decision, "refused_seq_gap");
    assert_refused(
        &decision,
        TailCaptureRefusal::Classified(TailRefusal::SeqGap),
    );
    assert_nothing_persisted(&before, &fx.listing().await);
}

// ── refused_seq_gap, producer (ii): the restore_seq_bump call-site guard ────

/// An outstanding DR-restore stamp refuses BEFORE the stream is read, and folds
/// onto the same `refused_seq_gap` label.
///
/// The fold is adjudicated (`t2.dfh.57.6e`): the guard fires on exactly the
/// state SR-1.2 / SR-3.6 name as the post-restore fossil boundary — the same
/// operator-facing condition, caught one read earlier because a jump beyond the
/// slice cap is invisible to the slice by construction. The DISTINCT enum
/// variant is where the diagnostic difference lives, which is why this test
/// asserts `PostRestoreUnrebased` and not merely the label — a regression that
/// collapsed the two variants would still pass a label-only assertion.
///
/// The real producer is a full DR restore re-seeding the object (heavier than
/// either fixture here); the column is the observable it leaves behind, and the
/// guard reads nothing else.
///
/// # There is now a THIRD producer of this label, and it is not reachable here
///
/// The checkpoint loop's chain-at-K admission gate
/// (`checkpointer::rebase_on_chain_full`) declines to start an SR-4.1 re-base on
/// the same header condition and ticks the same cell, because the capture unit
/// produces `ChainFull` *before* its own fossil guard and so hands the loop a
/// refusal that carries no fossil check. That producer is loop-side: it mints no
/// `TailCaptureRefusal` variant and cannot appear through this unit's API, which
/// is why the variant assertion below still enumerates exactly two. What binds
/// all three is the convention this test protects — one label, three distinct
/// tracing events — so a new producer that reused an existing event would be the
/// regression, not a new variant.
#[tokio::test]
async fn outstanding_restore_seq_bump_refuses_seq_gap_before_the_stream_read() {
    let fx = TailFixture::new("seq-gap-bump.live").await;

    for nonce in 0..3u64 {
        fx.commit(nonce).await;
    }
    fx.put_base(SEED_SEQ).await;
    let before = fx.listing().await;

    fx.set_restore_seq_bump(7).await;

    let decision = fx.capture(&cfg()).await;
    assert_outcome(&decision, "refused_seq_gap");
    assert_refused(
        &decision,
        TailCaptureRefusal::PostRestoreUnrebased {
            restore_seq_bump: 7,
        },
    );
    assert!(
        !matches!(
            &decision,
            TailCaptureDecision::Refused(TailCaptureRefusal::Classified(TailRefusal::SeqGap))
        ),
        "the two refused_seq_gap producers must stay DISTINCT enum variants; collapsing them \
         loses the only signal that tells a DR-restore fossil window apart from a live \
         ascending jump"
    );
    assert_nothing_persisted(&before, &fx.listing().await);

    // Control: clearing the stamp — exactly what a post-restore generation
    // landing does — and nothing else, and the same fixture captures. This is
    // what makes the refusal attributable to the column rather than to the
    // lineage or the stream shape.
    fx.set_restore_seq_bump(0).await;
    let control = fx.capture(&cfg()).await;
    assert_outcome(&control, "captured");
    let (_handle, from_seq, to_seq, record_count, _capped) = captured_parts(&control);
    assert_eq!(
        (from_seq, to_seq, record_count),
        (SEED_SEQ, SEED_SEQ + 3, 3)
    );
}

// ── refused_no_base ─────────────────────────────────────────────────────────

/// A layer whose checkpoint store holds no generation at all refuses at the
/// unit's FIRST check — one listing, no header read, no stream read.
///
/// No wide-hot-layer apparatus: reaching this arm needs an empty listing and
/// nothing else, and the unit refuses on it before it has looked at anything a
/// layer's shape could influence.
#[tokio::test]
async fn empty_store_listing_refuses_no_base_and_persists_nothing() {
    let fx = TailFixture::new("no-base.live").await;

    for nonce in 0..3u64 {
        fx.commit(nonce).await;
    }
    // Deliberately NO put_base: the lineage has no root.
    let before = fx.listing().await;
    assert!(before.is_empty(), "the fixture must seed no generation");

    let decision = fx.capture(&cfg()).await;
    assert_outcome(&decision, "refused_no_base");
    assert_refused(&decision, TailCaptureRefusal::NoBase);
    assert_nothing_persisted(&before, &fx.listing().await);
}

// ── refused_chain_full ──────────────────────────────────────────────────────

/// The chain bound refuses at K — and it counts CONSERVATIVELY.
///
/// # This drives the UNIT, which still refuses; the LOOP no longer does
///
/// `TailCaptureRefusal::ChainFull` is unchanged and is exactly what
/// `capture_layer_tail_with*` returns at the bound — that is what these two
/// phases pin, and they stay valid verbatim. What changed above this unit is who
/// gets the last word: the checkpoint loop intercepts this return and attempts
/// the SR-4.1 chain-collapse re-base, so on a healthy adapter a chain-at-K tick
/// surfaces `rebased`, not `refused_chain_full`. Do not read either phase below
/// as a claim about the tick.
///
/// Phase 2 in particular is the input to the loop's `UnderBound` fork rather
/// than a dead end: the loop gathers, learns the true link-walk length is 1, and
/// re-runs THIS unit under `ChainBound::Verified(1)`, which admits. The
/// conservative direction the phase pins is what makes that safe — the unit may
/// refuse early, and a caller that has paid for the walk may overrule it.
///
/// Two lineages, because the second pins a documented over-approximation the
/// first cannot show:
///
/// Phase 1 — a genuine chain at K. Base + one tail with `K = 1`: the raw count
/// of generations above the newest base is 1, the bound is reached, capture
/// refuses.
///
/// Phase 2 — the LANDED conservative count. `chain_len` is the POSITION of the
/// newest base in the listing, not the `from`/`to` link-walk SR-9.2 describes,
/// because no meta-only read exists on `CheckpointStore` and a true walk would
/// fetch up to K multi-MB blobs per eligible tick — outside SR-4.2's cost
/// budget. So an OVERLAPPING NON-MEMBER tail (same `from_seq`, different
/// `to_seq` — the residue of SR-9.2's divergent-S duplicate-leader fork) counts
/// toward the bound even though the true link-walk from the head would find a
/// chain of length 1 and admit. The approximation errs SAFE: it can refuse
/// earlier than K, never admit an over-long chain. This phase pins that
/// direction, so a future "fix" that quietly switched to a per-tick link-walk
/// (and its blob-fetch cost) inside this unit fails here — the link-walk's home
/// is the re-base, downstream of a gather this bound is what authorises.
#[tokio::test]
async fn chain_bound_refuses_chain_full_and_counts_overlapping_non_members() {
    // ── Phase 1: a genuine chain at K = 1 ───────────────────────────────
    let genuine = TailFixture::new("chain-full-genuine.live").await;
    for nonce in 0..3u64 {
        genuine.commit(nonce).await;
    }
    genuine.put_base(SEED_SEQ).await;
    genuine.put_tail(SEED_SEQ, SEED_SEQ, SEED_SEQ + 2).await;
    let before = genuine.listing().await;
    assert_eq!(
        before.len(),
        2,
        "phase 1 seeds exactly one base and one tail"
    );

    let decision = genuine.capture(&cfg_with_chain(1)).await;
    assert_outcome(&decision, "refused_chain_full");
    assert_refused(
        &decision,
        TailCaptureRefusal::ChainFull {
            chain_len: 1,
            max_chain: 1,
        },
    );
    assert_nothing_persisted(&before, &genuine.listing().await);

    // ── Phase 2: the conservative over-approximation at K = 2 ───────────
    let overlapping = TailFixture::new("chain-full-overlapping.live").await;
    for nonce in 0..5u64 {
        overlapping.commit(nonce).await;
    }
    overlapping.put_base(SEED_SEQ).await;
    // Both tails claim `from_seq == SEED_SEQ`, so at most ONE of them is a
    // member of the head's true chain; the other is a duplicate-leader fork
    // residue. A link-walk from the head (tail@SEED+3, whose from_seq is the
    // base's snap_seq) would report a chain of length 1 and ADMIT at K = 2.
    overlapping.put_tail(SEED_SEQ, SEED_SEQ, SEED_SEQ + 2).await;
    overlapping.put_tail(SEED_SEQ, SEED_SEQ, SEED_SEQ + 3).await;
    let before = overlapping.listing().await;
    assert_eq!(
        addresses(&before),
        vec![
            (overlapping.object_id, SEED_SEQ + 3, CheckpointKind::Tail),
            (overlapping.object_id, SEED_SEQ + 2, CheckpointKind::Tail),
            (overlapping.object_id, SEED_SEQ, CheckpointKind::Base),
        ],
        "the listing must arrive newest-first with the base last; the chain count is the base's \
         POSITION in exactly this order"
    );

    let decision = overlapping.capture(&cfg_with_chain(2)).await;
    assert_outcome(&decision, "refused_chain_full");
    assert_refused(
        &decision,
        TailCaptureRefusal::ChainFull {
            chain_len: 2,
            max_chain: 2,
        },
    );
    assert_nothing_persisted(&before, &overlapping.listing().await);
}

// ── oversize first entry → refused_missing_bytes ────────────────────────────

/// A first commit whose verbatim OMPF alone exceeds the per-tail byte budget is
/// excluded WHOLE, and the refusal is `refused_missing_bytes` — NOT
/// `refused_trimmed`.
///
/// The distinction is the operator's remediation, and it is the reason this
/// fold may not collapse into `trimmed`: `trimmed`'s documented lever is
/// `LIVE_EDIT_LAYER_STREAM_MAXLEN`, which is the wrong knob entirely for a
/// commit that is simply larger than `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES`.
/// The slice primitive WARNs naming the byte knob at this exact site; this test
/// deliberately asserts the OUTCOME and the STATE rather than log text.
///
/// The control at the end is what makes the refusal attributable: the same
/// stream, the same lineage, a stock byte cap — and it captures.
#[tokio::test]
async fn oversize_first_commit_refuses_missing_bytes_not_trimmed() {
    let fx = TailFixture::new("oversize.live").await;

    let (seq, ompf) = fx.commit_sized(1, OVERSIZE_VALUE_BYTES).await;
    assert_eq!(seq, SEED_SEQ + 1);
    assert!(
        ompf.len() > 135,
        "the fixture commit ({} bytes of OMPF) must exceed the 135-byte payload budget \
         cfg_tiny_bytes() derives, or the refusal under test cannot fire",
        ompf.len()
    );
    fx.put_base(SEED_SEQ).await;
    let before = fx.listing().await;

    let decision = fx.capture(&cfg_tiny_bytes()).await;
    assert_outcome(&decision, "refused_missing_bytes");
    assert_refused(
        &decision,
        TailCaptureRefusal::Classified(TailRefusal::MissingBytes),
    );
    assert_nothing_persisted(&before, &fx.listing().await);

    // Control: only the byte cap was in the way.
    let control = fx.capture(&cfg()).await;
    assert_outcome(&control, "captured");
    let (handle, from_seq, to_seq, record_count, _capped) = captured_parts(&control);
    assert_eq!(
        (from_seq, to_seq, record_count),
        (SEED_SEQ, SEED_SEQ + 1, 1)
    );
    let envelope = fx.decode(handle).await;
    assert_eq!(envelope.records()[0].ompf(), Some(&ompf));
}

// ── b.yvh: tail coverage under the relay byte budget ────────────────────────
//
// The relay byte budget (`LIVE_EDIT_LAYER_STREAM_MAX_BYTES`) trims the stream
// head by relay bytes. A capture whose window the budget left whole captures
// it in full, and one whose window floor it trimmed refuses `refused_trimmed`
// and persists nothing — never a tail claiming coverage the stream no longer
// holds. A checkpointer slower than the traffic is the second: the bytes
// committed since its last generation outran the budget.

/// The test-scaled byte budget: 64 KiB.
const TAIL_BUDGET: u64 = 64 * 1024;
/// A small commit's field value, far below the budget's 8 KiB generations.
const SMALL_VALUE_BYTES: usize = 1000;
/// A large commit's field value: a third of the budget, and a stream node of
/// its own (above the default `stream-node-max-bytes`), so the budget's
/// `XTRIM MINID ~` trims these entries exactly.
const LARGE_VALUE_BYTES: usize = 20 * 1024;

/// Ten small commits and one 16 KiB commit fit the budget: the capture
/// covers every one of them, byte for byte. The per-entry formula
/// `clamp(⌊B/s⌋, MIN, MAXLEN)` would have cut the window to ⌊64 KiB / the
/// large commit's relay⌋ = 3 entries at the large commit and refused this
/// capture `refused_trimmed`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn byte_budget_small_history_and_one_large_commit_is_captured_in_full() {
    let fx = TailFixture::new_with_budget("budget-captured.live", TAIL_BUDGET).await;
    let mut written: Vec<(u64, Bytes)> = Vec::new();
    for nonce in 0..10u64 {
        written.push(fx.commit_sized(nonce, SMALL_VALUE_BYTES).await);
    }
    written.push(fx.commit_sized(10, 16 * 1024).await);
    fx.put_base(SEED_SEQ).await;
    assert_eq!(
        fx.stream_entries_from(SEED_SEQ + 1).await.len(),
        written.len(),
        "fixture: the budget trimmed nothing of the window"
    );

    let decision = fx.capture(&cfg()).await;
    assert_outcome(&decision, "captured");
    let (handle, from_seq, to_seq, record_count, capped) = captured_parts(&decision);
    assert_eq!(
        (from_seq, to_seq, record_count, capped),
        (
            SEED_SEQ,
            SEED_SEQ + written.len() as u64,
            written.len(),
            false
        )
    );
    let envelope = fx.decode(handle).await;
    for (record, (seq, ompf)) in envelope.iter().zip(&written) {
        assert_eq!(record.sequence_num(), *seq);
        assert_eq!(
            record.ompf(),
            Some(ompf),
            "record {seq} must carry the writer's exact bytes"
        );
    }
}

/// A checkpointer slower than the traffic: one capture covers the first ten
/// small commits; then eight 20 KiB commits (160 KiB, past the 64 KiB budget)
/// land before the next capture. The budget trimmed the next window's floor,
/// so that capture refuses `refused_trimmed` and persists nothing — the first
/// tail stands, and nothing claims the trimmed commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn byte_budget_outrun_by_traffic_since_the_last_generation_refuses_trimmed() {
    let fx = TailFixture::new_with_budget("budget-outrun.live", TAIL_BUDGET).await;
    outrun_the_budget_to_refused_trimmed(&fx).await;
}

/// A layer refused `refused_trimmed` under the byte budget is not stuck: its
/// next full backup (a base generation at the layer's current sequence, as
/// the checkpointer's re-base writes) opens a new window above the trimmed
/// commits, and the next capture — two more large commits later, the budget
/// still trimming — covers them, byte for byte.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn byte_budget_refused_trimmed_layer_captures_again_after_its_next_full_backup() {
    let fx = TailFixture::new_with_budget("budget-rebase.live", TAIL_BUDGET).await;
    let rebase_seq = outrun_the_budget_to_refused_trimmed(&fx).await;

    fx.put_base(rebase_seq).await;
    let written = vec![
        fx.commit_sized(18, LARGE_VALUE_BYTES).await,
        fx.commit_sized(19, LARGE_VALUE_BYTES).await,
    ];
    assert_eq!(
        written.iter().map(|(seq, _)| *seq).collect::<Vec<_>>(),
        vec![rebase_seq + 1, rebase_seq + 2],
        "fixture: the two commits land right above the new base"
    );
    let decision = fx.capture(&cfg()).await;
    assert_outcome(&decision, "captured");
    let (handle, from_seq, to_seq, record_count, capped) = captured_parts(&decision);
    assert_eq!(
        (from_seq, to_seq, record_count, capped),
        (rebase_seq, rebase_seq + 2, 2, false),
        "the capture after the re-base covers exactly the commits above it"
    );
    let envelope = fx.decode(handle).await;
    for (record, (seq, ompf)) in envelope.iter().zip(&written) {
        assert_eq!(record.sequence_num(), *seq);
        assert_eq!(
            record.ompf(),
            Some(ompf),
            "record {seq} must carry the writer's exact bytes"
        );
    }
}

/// The outrun fixture shared by the two tests above: ten small commits and a
/// capture of them; eight 20 KiB commits (160 KiB, past the 64 KiB budget);
/// then a capture that refuses `refused_trimmed` and persists nothing.
/// Returns the layer's sequence after the last commit.
async fn outrun_the_budget_to_refused_trimmed(fx: &TailFixture) -> u64 {
    for nonce in 0..10u64 {
        fx.commit_sized(nonce, SMALL_VALUE_BYTES).await;
    }
    fx.put_base(SEED_SEQ).await;
    let first = fx.capture(&cfg()).await;
    assert_outcome(&first, "captured");
    let (_, _, first_to, _, _) = captured_parts(&first);
    assert_eq!(
        first_to,
        SEED_SEQ + 10,
        "the first tail covers the ten commits"
    );

    let mut last_seq = first_to;
    for nonce in 10..18u64 {
        last_seq = fx.commit_sized(nonce, LARGE_VALUE_BYTES).await.0;
    }
    let floor = first_to + 1;
    let head = fx.stream_entries_from(0).await;
    let head_seq: u64 = head
        .first()
        .and_then(|(id, _)| id.strip_suffix("-0")?.parse().ok())
        .expect("the stream holds commits");
    assert!(
        head_seq > floor,
        "fixture: the bytes since the first tail must have trimmed its window floor {floor}; \
         the stream starts at {head_seq}"
    );
    let before = fx.listing().await;

    let decision = fx.capture(&cfg()).await;
    assert_outcome(&decision, "refused_trimmed");
    assert_refused(
        &decision,
        TailCaptureRefusal::Classified(TailRefusal::Trimmed),
    );
    assert_nothing_persisted(&before, &fx.listing().await);
    last_seq
}
