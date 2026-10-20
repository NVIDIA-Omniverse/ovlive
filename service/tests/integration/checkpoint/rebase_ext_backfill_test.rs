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

//! SR-4.3 — the chain-collapse re-base's **external-value backfill** as a HARD
//! precondition for the base PUT. Task `t2.dfh.be.84`, Subtask
//! `t3.dfh.be.84.p4`.
//!
//! ## The claim under test
//!
//! > Every inline external-value part the replayed lineage carries is durably
//! > acknowledged by the checkpoint store **before** the collapsed base is PUT.
//! > Not one acknowledgement short, not "PUT first and backfill best-effort",
//! > and not "await the first ack and fire the rest".
//!
//! The hazard is specific and silent.
//! `assembler::build_reset_ompf_from_state` emits a base whose fields keep their
//! hash and `ext_value_index` while the BYTES do not travel (stated at
//! `composed_restore_ext_matrix_test.rs`'s header, and visible in that function
//! as the literal `OmpfContainer::serialize(&delta_bytes, &[])`). So a
//! partially-backfilled re-base lands a base whose by-hash refs are backed only
//! by live Valkey — and Task 4's lineage-aware retention then prunes the old
//! tails, which were the only durable copies of those bytes. That is precisely
//! the stranded-generation class FR-6 exists to eliminate, and **every**
//! assertion that merely checks "a base landed" goes green on it.
//!
//! ## The observation: ONE interleaved op log, not two counters
//!
//! The recording newtype below appends `put_ext_blob` and `put_layer_checkpoint`
//! into a **single** `Vec<StoreOp>`. Two independent counters can only say "4
//! ext puts and 1 base put happened", which is satisfied by every wrong
//! ordering; a single sequence carries the relative order, and the ordering
//! claim is then a recorded-index comparison
//! (`max(PutExt index) < index of the base PutLayer`). This is the
//! `TestStore::op_log` design note in `checkpointer.rs` applied to this lane.
//!
//! **When each op is recorded is load-bearing, and the two sides are recorded at
//! deliberately different moments:**
//!
//! | Op | Recorded | Why that moment |
//! |---|---|---|
//! | `PutExt` | after the inner call RETURNS (`Ok` or `Err`) | the claim is *durable acknowledgement*, which is a completion, not an issue. Recording at issue would let an implementation that fires every put and awaits none pass the ordering row |
//! | `PutLayer` | before delegating, i.e. at ISSUE | the claim is that the base PUT was never *issued* until every ack was in hand — strictly stronger than "was issued and failed", and it is also what makes the refusal rows' "no base PUT appears anywhere" real |
//!
//! Read together, `max(PutExt index) < PutLayer index` says: every inline part
//! had been acknowledged before the base PUT was even issued.
//!
//! ## Fixture — three hash classes in one lineage
//!
//! `base@100` + `tail(100,102]` + `tail(102,104]` + `tail(104,106]`, K = 3.
//!
//! | Hash | Referenced by | Class | What it pins |
//! |---|---|---|---|
//! | [`BLOB_TAG_BASE_ONLY`] | the BASE's `SetField` only, with **zero** OMPF parts | base-era | the D4 residual: it MUST NOT be backfilled (it was never inline), yet it MUST still appear in the new base's ext-refs sidecar |
//! | [`BLOB_TAG_DUP`] | records 101 **and** 103 — two different tail commits | cross-commit duplicate | lineage-wide dedup: exactly ONE `put_ext_blob` for it, not two |
//! | [`BLOB_TAG_TAIL_A`] / `_B` / `_C` | one tail commit each | tail-inline | the ordinary all-of precondition |
//!
//! Record 106 carries an **inline** field and no ext part at all, so the
//! backfill set is provably sourced from the ext-carrying records rather than
//! from "however many records the chain happened to hold".
//!
//! The base is built the way the assembler builds one — a reset delta whose
//! `SetField` carries a hash, a non-zero `ext_value_size` and a non-zero
//! `ext_value_index`, wrapped in an OMPF with **no** external parts. That is not
//! a synthetic shape: it is exactly what `build_reset_ompf_from_state` emits,
//! and `decode_checkpoint_ompf` never resolves an index against the parts (only
//! `decode_tail_record_ompf` does), so it decodes as a live `FieldValue::External`
//! reference with no bytes behind it.
//!
//! ## SR-4.3's scope, and the residual this file pins
//!
//! The hard precondition covers **tail-inline** parts — `ComposedLineage::ext_values`,
//! SR-4.3's literal wording. Base-era hashes keep the pre-existing best-effort
//! live-mirror posture (engineer decision D4, recorded in `rebase.rs`'s "SR-4.3
//! scope, and its recorded residual" section). [`rebase_backfills_exactly_the_composed_ext_set_and_never_the_base_only_hash`]
//! asserts that residual in **both** directions — not backfilled, still
//! sidecar-listed — so the scope decision is pinned by a test rather than only
//! by a doc comment.
//!
//! ## Valkey-free, deliberately
//!
//! `rebase_layer_lineage` takes `&dyn CheckpointStore` and nothing else, and
//! nothing in this file restores a checkpoint, so no `obj:hdr` and no live layer
//! mapping is reachable from any path under test. The sibling refusal lane
//! (`rebase_refusal_test.rs`) already owns the cursor-immobility fence with
//! sentinel columns, and that fence is a property of the **unit's signature**,
//! not of one refusal arm — re-planting it here would buy a duplicate of an
//! arm-independent claim at the cost of a session + layer per test. The
//! intactness claim this file makes is therefore the store-side half: listing,
//! artifact enumeration, and an op log with no mutating call in it.
//!
//! ## Metrics: no `rebased`-did-not-tick assertion, and why
//!
//! Per the orchestrator's reconciliation, `rebase_layer_lineage` holds no
//! metrics handle by design (its signature mirrors `gather_restore_lineage`'s
//! no-`Config`/no-metrics cross-task contract), so a unit-driven lane has no
//! cell to read. On every refusal row here the discriminator is the pair
//! *(`RebaseRefusal::ExtBackfill`, no `PutLayer` anywhere in the op log)*. The
//! metric claim rides the loop-level lane (`t3.dfh.be.84.1k`).
//!
//! ## Bounded gates — no sleeps, no wall-clock assertions
//!
//! | Gate | Ceiling | What reaching it means |
//! |------|---------|------------------------|
//! | every `rebase_layer_lineage` call | [`REBASE_CALL_CEILING`] (30 s) | the unit wedged: it neither refused nor landed inside its own budgets |
//! | the unit's internal gather budget | `(max_chain + 2) x` [`REBASE_FETCH_RESERVE`] = 10 s | a member fetch against a tempdir holding sub-100-KiB artifacts overran 2 s, i.e. the filesystem, not the code |
//! | the unit's per-blob backfill budget | [`REBASE_FETCH_RESERVE`] (2 s) | the adapter did not answer one `put_ext_blob` in 2 s. [`rebase_refuses_ext_backfill_when_a_part_exceeds_its_per_blob_budget`] drives this deliberately and is the ONLY test that pays it |
//!
//! [`REBASE_FETCH_RESERVE`] is 2 s rather than the shipped 30 s precisely so the
//! elapsed-timeout row costs two seconds instead of thirty. The unit reads the
//! value from [`TailCaptureConfig::rebase_fetch_reserve`], so nothing here
//! depends on the shipped default.
//!
//! The stall is injected as `std::future::pending()` INSIDE the store wrapper —
//! a `put_ext_blob` that never answers. That is fault injection (a wedged
//! adapter), not `sleep`-as-synchronization: no test waits on a clock for a
//! result, the production `timeout(reserve, ..)` is what ends the call, and its
//! elapsing is the thing under test.
//!
//! ## Mutation evidence
//!
//! Four mutations were applied to `src/checkpoint/rebase.rs` in place and each
//! reverted byte-identically (`sha256` verified against the pre-mutation
//! digest). Each row names the assertion class it targets and the exact red
//! count out of this module's **7** tests; reproduce by re-applying the mutation
//! and running this file's narrow filter
//! (`cargo nextest run -p live-edit --test integration --features test-support
//! -E 'test(checkpoint::rebase_ext_backfill_test::)'`).
//!
//! | # | Mutation | Red | What it proves is asserted |
//! |---|---|---|---|
//! | M1 | the base PUT is hoisted ABOVE the backfill loop (§4 moved after §6) — the "PUT first, backfill best-effort" implementation | 4 of 7 — the ordering row plus all three refusal rows | the ordering claim, and that the three refusal rows are genuinely discriminating: each failed naming the recorded `PutLayer { snap_seq: 106, kind: Base }` that must not exist. The completeness, idempotency and readback rows stayed GREEN — correctly: they assert WHAT was backfilled, never WHEN, so only the ordering row carries the interleaving |
//! | M2 | the backfill is made fire-and-forget (`tokio::spawn` per blob, no await, loop never refuses) | 6 of 7 — everything but the readback row | that the claim rests on *acknowledgement*, not on the call merely happening. The single survivor is the measurement: on a fast LocalDisk adapter the spawned puts did land, so the readback row alone cannot tell a fire-and-forget backfill from an awaited one — which is exactly why the ordering and refusal rows exist |
//! | M3 | only the FIRST blob is awaited; blobs `1..` are put with their result discarded | 2 of 7 — the last-part row and the timeout row | that row 4 is NOT redundant with row 3. Row 3 (first part fails) stayed GREEN under this mutation — its one awaited put is the injected one — which is precisely the implementation the Subtask says row 3 alone cannot catch |
//! | M4 | the per-blob `tokio::time::timeout` is dropped (`store.put_ext_blob(..).await` bare) | 1 of 7 — the timeout row | the PM-correction-6 claim, and nothing else. The row ran to [`REBASE_CALL_CEILING`] and failed on its "a call went UNBOUNDED" expect, i.e. the slow-adapter arm is the only thing holding that bound |
//!
//! M1's RED message from the ordering row, recorded verbatim because the
//! Subtask asks for it — note that it prints the whole interleaved log, which is
//! what makes a failure here diagnosable without a re-run:
//!
//! ```text
//! SR-4.3: EVERY inline ext part must have been durably acknowledged before the base PUT was
//! issued. The last `put_ext_blob` completed at op index 4 and the base PUT was issued at index 0.
//! A base that precedes any ack is one whose by-hash refs are backed only by live Valkey while
//! retention prunes the tails that carried the bytes. Op log: [PutLayer { object_id: 4242,
//! snap_seq: 106, kind: Base }, PutExt { hash_hex: "17ecac66…", ok: true }, PutExt { hash_hex:
//! "d56bcb91…", ok: true }, PutExt { hash_hex: "ab4b3581…", ok: true }, PutExt { hash_hex:
//! "56320ed3…", ok: true }]
//! ```

use std::collections::{BTreeSet, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use tempfile::TempDir;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::{
    compose_from_artifacts, rebase_layer_lineage, CheckpointBlob, CheckpointError,
    CheckpointHandle, CheckpointKind, CheckpointMeta, CheckpointStore, LayerArtifact,
    LayerArtifactKind, RebaseDecision, RebaseRefusal, TailCaptureConfig,
};
use live_edit::storage::external_values::ext_content_hash;
use live_edit::storage::keys::sha256_to_hex;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, SetField, StructCommand};

use super::helpers::{
    put_tail_generation_from_records, HAND_PUT_CREATED_BY, HAND_PUT_SERVER_TS,
    HAND_PUT_WRITTEN_AT_UNIX_MS, SERVICE_LABEL,
};
use crate::common::owned_delta::delta_to_ompf_bytes;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// `replica_id` handed to the unit. Reaches only `CheckpointMeta`.
const REPLICA_ID: &str = "rebase-ext-backfill-test-replica";

/// Identity written onto every hand-put meta. Never resolved against a live
/// layer mapping — nothing here applies a checkpoint — but non-empty is
/// required by `CheckpointMeta`'s identity lockstep and by `validate_tail_put`.
const SESSION_ID: &str = "rebase-ext-backfill-session";
const LAYER_NAME: &str = "rebase-ext-backfill-layer";

/// K. Three rather than the shipped eight, for [`rebase_refusal_test`]'s
/// reason: the chain length is a bound this file must merely reach, and three
/// tails keep the planted lineage and the derived gather budget small.
///
/// [`rebase_refusal_test`]: super::rebase_refusal_test
const MAX_CHAIN: usize = 3;

/// The lineage root's `snap_seq`.
const BASE_SEQ: u64 = 100;

/// `to_seq` of the tail rooted directly at the base — `(100, 102]`.
const TAIL_1_TO_SEQ: u64 = 102;
/// `to_seq` of the middle tail — `(102, 104]`.
const TAIL_2_TO_SEQ: u64 = 104;
/// `to_seq` of the head tail — `(104, 106]`. Also **S**, the sequence a
/// successful collapse PUTs its new base at.
const HEAD_TAIL_TO_SEQ: u64 = 106;

/// The two objects this file plants lineages under. Two are needed by
/// [`rebase_backfill_is_idempotent_by_hash_across_a_second_rebase`] alone; every
/// other test uses [`OBJECT_ID_A`].
const OBJECT_ID_A: u64 = 4_242;
/// The second lineage's object id — see [`OBJECT_ID_A`].
const OBJECT_ID_B: u64 = 4_343;

/// Per-member fetch reserve AND per-blob backfill ceiling — the unit derives
/// both from this one knob. See the module header's budget table for why 2 s.
const REBASE_FETCH_RESERVE: Duration = Duration::from_secs(2);

/// Ceiling on one `rebase_layer_lineage` call. Strictly above the unit's own
/// derived gather budget (10 s at the knobs above), so a real budget step-down
/// still returns a decision; reaching THIS ceiling means the unit wedged.
const REBASE_CALL_CEILING: Duration = Duration::from_secs(30);

/// The section every field in the fixture is written into. Created by the
/// base's reset delta, so the tail records' `SetField`s land in an existing
/// section exactly as a live incremental commit's would.
const ROOT_SECTION_ID: u64 = 42;

/// Authoring rank stamped on every `SetField`. Every field is written exactly
/// once, so the value only has to be stable.
const AUTHORING_RANK: u64 = 0;

/// The one-based OMPF part index every ext-carrying `SetField` points at. Part 0
/// is the OODF delta, so `delta_to_ompf_bytes(delta, &[blob])` puts `blob` at
/// Part 1.
const EXT_VALUE_INDEX: u32 = 1;

/// Byte length of every external value. Large enough to be unambiguously an
/// out-of-band OMPF part, small enough that four of them keep the tail
/// artifacts trivially inside the reserve.
const EXT_BLOB_LEN: usize = 4 * 1024;

/// Per-blob discriminator, so the five fixture blobs hash differently and a
/// byte-identity assertion cannot pass by comparing the wrong pair.
const BLOB_TAG_BASE_ONLY: u8 = 0x11;
/// See [`BLOB_TAG_BASE_ONLY`]. Referenced by TWO tail commits.
const BLOB_TAG_DUP: u8 = 0x22;
/// See [`BLOB_TAG_BASE_ONLY`].
const BLOB_TAG_TAIL_A: u8 = 0x33;
/// See [`BLOB_TAG_BASE_ONLY`].
const BLOB_TAG_TAIL_B: u8 = 0x44;
/// See [`BLOB_TAG_BASE_ONLY`].
const BLOB_TAG_TAIL_C: u8 = 0x55;

// Field names. Constants rather than literals because the base-only field is
// looked up by name when the sidecar claim is checked.
const BASE_ONLY_FIELD: &str = "base-only-field";
const DUP_FIELD_FIRST: &str = "dup-field-first";
const DUP_FIELD_SECOND: &str = "dup-field-second";
const TAIL_A_FIELD: &str = "tail-a-field";
const TAIL_B_FIELD: &str = "tail-b-field";
const TAIL_C_FIELD: &str = "tail-c-field";
const INLINE_FIELD: &str = "inline-field";

/// How many distinct inline ext parts the fixture lineage carries: `dup`, `a`,
/// `b`, `c`. Deduplicated — `dup` is referenced twice and counts once.
const EXPECTED_INLINE_EXT_COUNT: usize = 4;

// ─────────────────────────────────────────────────────────────────────────────
// The op log
// ─────────────────────────────────────────────────────────────────────────────

/// One mutating call the wrapper saw. Both halves of the ordering claim live in
/// ONE sequence — see the module header for why an interleaved log rather than
/// two counters, and for when each variant is appended.
///
/// Every mutating `CheckpointStore` method has a variant, not just the two under
/// test: a refusal row's claim is that the re-base wrote **nothing**, and a log
/// that recorded only puts would go green for a unit that had started deleting
/// superseded generations (which this unit must never do — retention owns every
/// deletion).
#[derive(Debug, Clone, PartialEq, Eq)]
enum StoreOp {
    /// Appended AFTER the inner call returns. `ok` distinguishes a durable
    /// acknowledgement from a call that was made and failed.
    PutExt {
        hash_hex: String,
        ok: bool,
    },
    /// Appended BEFORE delegating — the claim is about issue, not completion.
    PutLayer {
        object_id: u64,
        snap_seq: u64,
        kind: CheckpointKind,
    },
    DeleteLayer {
        object_id: u64,
        snap_seq: u64,
        kind: CheckpointKind,
    },
    DeleteExtBlob {
        hash_hex: String,
    },
    DeleteLayerTemp {
        object_id: u64,
        snap_seq: u64,
    },
    PutSession {
        session_id: String,
    },
    DeleteSession {
        session_id: String,
    },
    PutCountersSidecar,
}

impl StoreOp {
    /// The hash a successful ext backfill acknowledged, or `None` for every
    /// other op (including a FAILED ext put — an unacknowledged write is not a
    /// backfill).
    fn acked_ext_hash(&self) -> Option<&str> {
        match self {
            StoreOp::PutExt { hash_hex, ok: true } => Some(hash_hex.as_str()),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The recording / injecting store newtype
// ─────────────────────────────────────────────────────────────────────────────

/// `CheckpointStore` newtype over a real [`LocalDiskCheckpointStore`] that
/// records every mutating call into one interleaved op log and can fail or stall
/// **one targeted `put_ext_blob`**, keyed by `hash_hex`.
///
/// # Why a newtype rather than `CheckpointFaultInjector`
///
/// Same reason the sibling refusal lane records: the injector seam keys on a
/// method NAME, so a `put_ext_blob` injection would fail the first, the middle
/// and the last blob indistinguishably — and rows 3 and 4 exist precisely to
/// tell those apart. Keying on `hash_hex` is what makes "the LAST part failed"
/// expressible at all.
///
/// # Delegation
///
/// Composed atop [`DelegatingCheckpointStore`](super::helpers::DelegatingCheckpointStore)
/// and forwards **every** method, including the four with non-trivial trait
/// defaults (`read_ext_refs_sidecar`, `list_layer_artifacts`,
/// `delete_layer_temp_artifacts`, `publish_worst_case`). Omitting those would not
/// leave a call un-instrumented, it would SUBSTITUTE behaviour: `list_layer_artifacts`'
/// default answers `Ok(vec![])`, which is what the no-partial-artifact claim
/// expects to see unchanged, and `read_ext_refs_sidecar`'s answers `Ok(empty)`,
/// which would make the base-only sidecar claim pass vacuously.
struct RecordingExtBackfillStore {
    delegate: super::helpers::DelegatingCheckpointStore<LocalDiskCheckpointStore>,
    ops: Mutex<Vec<StoreOp>>,
    /// `Some(hash_hex)` while armed: that blob's `put_ext_blob` returns
    /// `Transient` instead of delegating.
    fail_put_ext_at: Mutex<Option<String>>,
    /// `Some(hash_hex)` while armed: that blob's `put_ext_blob` never answers,
    /// so the unit's per-blob `timeout(reserve, ..)` is what ends it.
    stall_put_ext_at: Mutex<Option<String>>,
    /// How many `put_ext_blob` calls were ISSUED (as opposed to acknowledged).
    /// The stall row's only observable, since a stalled call completes never and
    /// therefore records no op.
    ext_puts_issued: AtomicUsize,
    /// How many times the armed failure fired. Read by every row that arms one,
    /// so a mis-targeted arm surfaces as a failed assertion rather than as a
    /// refusal that happened for some other reason.
    injected_failures: AtomicUsize,
    /// How many times the armed stall fired — same non-vacuity role.
    injected_stalls: AtomicUsize,
}

impl RecordingExtBackfillStore {
    fn new(inner: Arc<LocalDiskCheckpointStore>) -> Self {
        Self {
            delegate: super::helpers::DelegatingCheckpointStore::new(inner),
            ops: Mutex::new(Vec::new()),
            fail_put_ext_at: Mutex::new(None),
            stall_put_ext_at: Mutex::new(None),
            ext_puts_issued: AtomicUsize::new(0),
            injected_failures: AtomicUsize::new(0),
            injected_stalls: AtomicUsize::new(0),
        }
    }

    /// Fail `put_ext_blob` for exactly `hash_hex` with a `Transient` error.
    ///
    /// `Transient` deliberately: on a read path that classification means "retry
    /// me", and SR-4.5's claim is that here it becomes a *decision* the next tick
    /// retries rather than an `Err` the caller must classify.
    fn fail_put_ext(&self, hash_hex: &str) {
        *self
            .fail_put_ext_at
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(hash_hex.to_string());
    }

    /// Wedge `put_ext_blob` for exactly `hash_hex` — it never answers.
    fn stall_put_ext(&self, hash_hex: &str) {
        *self
            .stall_put_ext_at
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(hash_hex.to_string());
    }

    /// Lift both injections — the "next tick" of rows 3, 4 and 7.
    fn heal(&self) {
        *self
            .fail_put_ext_at
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = None;
        *self
            .stall_put_ext_at
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = None;
    }

    fn armed_failure(&self) -> Option<String> {
        self.fail_put_ext_at
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    fn armed_stall(&self) -> Option<String> {
        self.stall_put_ext_at
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    fn record(&self, op: StoreOp) {
        self.ops.lock().unwrap_or_else(|p| p.into_inner()).push(op);
    }

    fn ops(&self) -> Vec<StoreOp> {
        self.ops.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// A snapshot of the op log's length, taken immediately before an act step.
    fn op_mark(&self) -> usize {
        self.ops.lock().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// Ops recorded after `mark`, in order. Relative index within this slice is
    /// what every ordering assertion compares.
    fn ops_since(&self, mark: usize) -> Vec<StoreOp> {
        self.ops().into_iter().skip(mark).collect()
    }

    fn ext_puts_issued(&self) -> usize {
        self.ext_puts_issued.load(Ordering::Acquire)
    }

    fn injected_failures(&self) -> usize {
        self.injected_failures.load(Ordering::Acquire)
    }

    fn injected_stalls(&self) -> usize {
        self.injected_stalls.load(Ordering::Acquire)
    }
}

#[async_trait]
impl CheckpointStore for RecordingExtBackfillStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        // Recorded BEFORE delegation — see the module header's table: the claim
        // is that the base PUT was never ISSUED until every ack was in hand.
        self.record(StoreOp::PutLayer {
            object_id,
            snap_seq,
            kind: meta.kind,
        });
        self.delegate
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.ext_puts_issued.fetch_add(1, Ordering::AcqRel);

        if self.armed_stall().as_deref() == Some(hash_hex) {
            self.injected_stalls.fetch_add(1, Ordering::AcqRel);
            // A wedged adapter, injected as a future that never resolves. The
            // unit's own `timeout(cfg.rebase_fetch_reserve, ..)` is what ends
            // this call, and its elapsing is the thing under test — so nothing
            // is recorded here, because nothing was ever acknowledged.
            std::future::pending::<()>().await;
            unreachable!("std::future::pending never resolves");
        }

        if self.armed_failure().as_deref() == Some(hash_hex) {
            self.injected_failures.fetch_add(1, Ordering::AcqRel);
            self.record(StoreOp::PutExt {
                hash_hex: hash_hex.to_string(),
                ok: false,
            });
            return Err(CheckpointError::Transient {
                source: format!(
                    "synthetic transient ext-blob backfill failure at {hash_hex} (SR-4.3 test)"
                )
                .into(),
            });
        }

        // Recorded AFTER the inner call returns — the op log's `PutExt` entries
        // are ACKNOWLEDGEMENTS, which is what SR-4.3's precondition is about.
        let result = self.delegate.put_ext_blob(hash_hex, bytes).await;
        self.record(StoreOp::PutExt {
            hash_hex: hash_hex.to_string(),
            ok: result.is_ok(),
        });
        result
    }

    // ── Forwarded reads ─────────────────────────────────────────────────────

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_latest_layer_checkpoint(object_id).await
    }

    // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST — its trait
    // default returns `NotImplemented`, which would report "no such generation"
    // for a generation the delegate holds, to the very walk this file drives.
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.delegate.list_layer_checkpoints(object_id).await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.delegate.list_session_checkpoint_ids().await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.delegate.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.delegate.ext_blob_exists(hash_hex).await
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.delegate.get_counters_sidecar().await
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.delegate.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.delegate.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.delegate.list_live_ext_set().await
    }

    // ── Remaining mutating methods, all recorded ────────────────────────────

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.record(StoreOp::DeleteLayer {
            object_id: handle.object_id,
            snap_seq: handle.seq(),
            kind: handle.kind,
        });
        self.delegate.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.record(StoreOp::PutSession {
            session_id: session_id.to_string(),
        });
        self.delegate
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.record(StoreOp::DeleteSession {
            session_id: session_id.to_string(),
        });
        self.delegate.delete_session_checkpoint(session_id).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.record(StoreOp::DeleteExtBlob {
            hash_hex: hash_hex.to_string(),
        });
        self.delegate.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.record(StoreOp::PutCountersSidecar);
        self.delegate.put_counters_sidecar(body).await
    }

    // ── The four non-trivial trait defaults ─────────────────────────────────
    //
    // Forwarded because omitting them SUBSTITUTES behaviour rather than merely
    // skipping instrumentation — see the type's doc-comment.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        self.delegate.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.delegate.list_layer_artifacts().await
    }

    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.record(StoreOp::DeleteLayerTemp {
            object_id,
            snap_seq,
        });
        self.delegate
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> Duration {
        self.delegate.publish_worst_case()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture blobs and payload builders
// ─────────────────────────────────────────────────────────────────────────────

/// One external value in the three spellings the assertions need.
struct ExtBlob {
    hash: [u8; 32],
    hash_hex: String,
    bytes: Vec<u8>,
}

impl ExtBlob {
    fn new(tag: u8) -> Self {
        // Deterministic rather than random: an RNG would make a failing run
        // unreproducible for no benefit. The bytes only have to be non-trivial,
        // distinct per class, and stable, so `ext_content_hash` of them is the
        // fixture's own contract.
        let bytes: Vec<u8> = (0..EXT_BLOB_LEN).map(|i| ((i % 251) as u8) ^ tag).collect();
        let hash = ext_content_hash(&bytes);
        Self {
            hash_hex: sha256_to_hex(&hash),
            hash,
            bytes,
        }
    }
}

/// The five external values the fixture lineage references.
struct FixtureBlobs {
    /// Referenced by the BASE's `SetField` only, with zero OMPF parts — the D4
    /// residual class.
    base_only: ExtBlob,
    /// Referenced inline by records 101 AND 103 — the cross-commit duplicate.
    dup: ExtBlob,
    tail_a: ExtBlob,
    tail_b: ExtBlob,
    tail_c: ExtBlob,
}

impl FixtureBlobs {
    fn new() -> Self {
        Self {
            base_only: ExtBlob::new(BLOB_TAG_BASE_ONLY),
            dup: ExtBlob::new(BLOB_TAG_DUP),
            tail_a: ExtBlob::new(BLOB_TAG_TAIL_A),
            tail_b: ExtBlob::new(BLOB_TAG_TAIL_B),
            tail_c: ExtBlob::new(BLOB_TAG_TAIL_C),
        }
    }

    /// The four TAIL-INLINE blobs, in the lineage's first-appearance order —
    /// the order `ComposedLineage::ext_values` must produce, and therefore the
    /// order the sequential backfill must put them in.
    fn inline_in_first_appearance_order(&self) -> [&ExtBlob; EXPECTED_INLINE_EXT_COUNT] {
        [&self.dup, &self.tail_a, &self.tail_b, &self.tail_c]
    }

    fn inline_hash_hexes(&self) -> Vec<String> {
        self.inline_in_first_appearance_order()
            .iter()
            .map(|blob| blob.hash_hex.clone())
            .collect()
    }
}

/// The lineage ROOT's reset OMPF: one section, one **external** field whose
/// bytes do not travel, and zero OMPF external parts.
///
/// That is not a synthetic shape — it is byte-for-byte the shape
/// `assembler::build_reset_ompf_from_state` emits for a state carrying an
/// external field (`OmpfContainer::serialize(&delta_bytes, &[])`, with the
/// per-field `ext_value_index` still stamped so an apply-side reassembly can
/// pair it with the sidecar). It decodes through `decode_checkpoint_ompf`, which
/// never resolves an index against the parts, so the base-only hash lands as a
/// live `FieldValue::External` reference with no durable bytes behind it —
/// exactly the state the D4 residual describes.
fn base_reset_ompf(base_only: &ExtBlob) -> Bytes {
    let delta = OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_string(),
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: ROOT_SECTION_ID,
            section_name: "rebase-ext-backfill-root".to_string(),
            section_type: 1,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: BASE_ONLY_FIELD.to_string(),
            value_or_ext_hash: base_only.hash.to_vec(),
            ext_value_size: base_only.bytes.len() as u64,
            ext_value_index: EXT_VALUE_INDEX,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    };
    Bytes::from(delta_to_ompf_bytes(&delta, &[]))
}

/// One tail record: an INCREMENTAL commit writing `key_name` as an external
/// reference, with the blob carried inline as OMPF Part 1.
///
/// `base_version: 1` (any non-zero) is what marks this incremental. With
/// `base_version == 0` and `is_diff == true` the domain engine takes the reset
/// branch and wipes prior state, which would empty the fixture on every record
/// and leave the composed lineage carrying nothing.
fn ext_record_ompf(key_name: &str, blob: &ExtBlob) -> Bytes {
    let delta = OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: key_name.to_string(),
            value_or_ext_hash: blob.hash.to_vec(),
            ext_value_size: blob.bytes.len() as u64,
            ext_value_index: EXT_VALUE_INDEX,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    };
    Bytes::from(delta_to_ompf_bytes(&delta, &[blob.bytes.as_slice()]))
}

/// A tail record carrying NO external value at all — an ordinary inline field
/// write.
///
/// Load-bearing for the completeness claim: with it in the chain, the backfilled
/// set is provably sourced from the ext-carrying records rather than from
/// "however many records the chain happened to hold".
fn inline_record_ompf() -> Bytes {
    let delta = OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: INLINE_FIELD.to_string(),
            value_or_ext_hash: vec![0xC0, 0xDE, 0x01],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    };
    Bytes::from(delta_to_ompf_bytes(&delta, &[]))
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture
// ─────────────────────────────────────────────────────────────────────────────

/// A `LocalDiskCheckpointStore` over a fresh tempdir behind the recording
/// newtype, plus the five fixture blobs.
///
/// LocalDisk rather than InMemory on purpose, for the sibling lane's reason:
/// [`LayerArtifactKind::Temp`] is a **filesystem** residual (`<final-name>.<uuid>.tmp`
/// from the adapter's write-temp-then-rename), and the InMemory adapter can
/// never produce one — the no-partial-artifact half of the refusal claim is only
/// falsifiable on an adapter that has temps.
///
/// No `TestValkey`: see the module header.
struct BackfillFixture {
    /// Held so its `Drop` removes the per-test checkpoint root. Never read.
    _dir: TempDir,
    store: Arc<RecordingExtBackfillStore>,
    store_dyn: Arc<dyn CheckpointStore>,
    blobs: FixtureBlobs,
}

/// The pre-attempt reference a refusal is compared against.
#[derive(Debug)]
struct StoreSnapshot {
    listing: Vec<CheckpointHandle>,
    artifacts: Vec<LayerArtifact>,
}

fn backfill_fixture() -> BackfillFixture {
    let dir = tempfile::tempdir().expect("tempfile::tempdir must succeed");
    let inner = Arc::new(
        LocalDiskCheckpointStore::new(dir.path().to_path_buf())
            .expect("LocalDiskCheckpointStore::new must succeed on a fresh tempdir"),
    );
    let store = Arc::new(RecordingExtBackfillStore::new(inner));
    let store_dyn: Arc<dyn CheckpointStore> = Arc::clone(&store) as Arc<dyn CheckpointStore>;
    BackfillFixture {
        _dir: dir,
        store,
        store_dyn,
        blobs: FixtureBlobs::new(),
    }
}

impl BackfillFixture {
    /// The knobs the unit reads. Built from [`TailCaptureConfig::default`] so a
    /// new field surfaces here as the shipped default rather than as a
    /// hand-invented one, with only the two this file has an opinion about
    /// overridden.
    fn tail_config(&self) -> TailCaptureConfig {
        TailCaptureConfig {
            max_chain: MAX_CHAIN,
            rebase_fetch_reserve: REBASE_FETCH_RESERVE,
            ..TailCaptureConfig::default()
        }
    }

    /// Drive the unit under [`REBASE_CALL_CEILING`].
    async fn rebase(&self, object_id: u64) -> RebaseDecision {
        tokio::time::timeout(
            REBASE_CALL_CEILING,
            rebase_layer_lineage(
                self.store_dyn.as_ref(),
                object_id,
                REPLICA_ID,
                &self.tail_config(),
            ),
        )
        .await
        .expect(
            "rebase_layer_lineage must produce a decision inside REBASE_CALL_CEILING — every \
             budget the unit derives (the gather deadline and the per-blob backfill timeout) is \
             strictly smaller, so reaching this ceiling means a call went UNBOUNDED rather than \
             that the tempdir was slow",
        )
        .expect(
            "rebase_layer_lineage must not Err: NotFound is its ONLY error arm and this fixture's \
             generations never vanish; every SR-4.5 cause is a RebaseDecision",
        )
    }

    /// Hand-put the lineage root at [`BASE_SEQ`].
    ///
    /// A file-local builder rather than Task 2's `put_base_generation` because
    /// that helper hard-codes `common::empty_reset_ompf()` and `sidecar_bytes:
    /// None` — an EMPTY base, which carries no external reference and therefore
    /// cannot express this file's base-only hash class. Everything else about
    /// the meta is [`put_base_generation`](super::helpers::put_base_generation)'s
    /// shape verbatim, field for field, so the two fixtures cannot drift on the
    /// identity / lineage / `observed_commit_span` triple.
    async fn put_fixture_base(&self, object_id: u64) -> CheckpointHandle {
        let meta = CheckpointMeta {
            creator_replica_id: SERVICE_LABEL.to_string(),
            written_at_unix_ms: HAND_PUT_WRITTEN_AT_UNIX_MS,
            snap_seq: Some(BASE_SEQ),
            session_id: SESSION_ID.to_string(),
            layer_name: LAYER_NAME.to_string(),
            created_by: HAND_PUT_CREATED_BY.to_string(),
            server_ts: HAND_PUT_SERVER_TS,
            observed_commit_span: 0,
            kind: CheckpointKind::Base,
            // SR-2.3: a base carries no lineage.
            base_snap_seq: None,
            from_seq: None,
            // The OLD base's own sidecar is not this file's subject — nothing
            // reads it, and the re-base sources its sidecar from the COMPOSED
            // state, not from the root's. `None` keeps the fixture honest about
            // what it is asserting.
            sidecar_bytes: None,
        };
        self.store_dyn
            .put_layer_checkpoint(
                object_id,
                BASE_SEQ,
                base_reset_ompf(&self.blobs.base_only),
                meta,
            )
            .await
            .unwrap_or_else(|e| panic!("hand-put of the fixture base@{BASE_SEQ} must succeed: {e}"))
    }

    /// `base@100` + three linked tails carrying six records, four of which
    /// reference a distinct inline external value and one of which references an
    /// already-seen one.
    ///
    /// Record layout — the shape every claim in this file reads:
    ///
    /// | Seq | Tail | Carries |
    /// |---|---|---|
    /// | 101 | `(100,102]` | `dup` (first appearance) |
    /// | 102 | `(100,102]` | `tail_a` |
    /// | 103 | `(102,104]` | `dup` AGAIN — the cross-commit duplicate |
    /// | 104 | `(102,104]` | `tail_b` |
    /// | 105 | `(104,106]` | `tail_c` |
    /// | 106 | `(104,106]` | an INLINE field, no ext part |
    ///
    /// Two records per tail, not one: a single-record envelope makes an
    /// off-by-one between a generation's claimed span and its carried records
    /// unobservable (the b.pw6 lesson, applied to a fixture).
    async fn plant_lineage(&self, object_id: u64) {
        self.put_fixture_base(object_id).await;

        let tails: [(u64, Vec<(u64, Bytes)>); MAX_CHAIN] = [
            (
                BASE_SEQ,
                vec![
                    (
                        BASE_SEQ + 1,
                        ext_record_ompf(DUP_FIELD_FIRST, &self.blobs.dup),
                    ),
                    (
                        TAIL_1_TO_SEQ,
                        ext_record_ompf(TAIL_A_FIELD, &self.blobs.tail_a),
                    ),
                ],
            ),
            (
                TAIL_1_TO_SEQ,
                vec![
                    (
                        TAIL_1_TO_SEQ + 1,
                        ext_record_ompf(DUP_FIELD_SECOND, &self.blobs.dup),
                    ),
                    (
                        TAIL_2_TO_SEQ,
                        ext_record_ompf(TAIL_B_FIELD, &self.blobs.tail_b),
                    ),
                ],
            ),
            (
                TAIL_2_TO_SEQ,
                vec![
                    (
                        TAIL_2_TO_SEQ + 1,
                        ext_record_ompf(TAIL_C_FIELD, &self.blobs.tail_c),
                    ),
                    (HEAD_TAIL_TO_SEQ, inline_record_ompf()),
                ],
            ),
        ];

        for (from_seq, records) in tails {
            put_tail_generation_from_records(
                &self.store_dyn,
                object_id,
                SESSION_ID,
                LAYER_NAME,
                BASE_SEQ,
                from_seq,
                &records,
                HAND_PUT_WRITTEN_AT_UNIX_MS,
                0,
            )
            .await;
        }
    }

    async fn listing(&self, object_id: u64) -> Vec<CheckpointHandle> {
        self.store_dyn
            .list_layer_checkpoints(object_id)
            .await
            .expect("list_layer_checkpoints must succeed against a healthy tempdir")
    }

    /// Every artifact the checkpoint root holds, sorted so the comparison is
    /// about content rather than about `read_dir` order.
    async fn artifacts(&self) -> Vec<LayerArtifact> {
        let mut artifacts = self
            .store_dyn
            .list_layer_artifacts()
            .await
            .expect("list_layer_artifacts must succeed against a healthy tempdir");
        artifacts.sort_by_key(|a| (a.object_id, a.snap_seq, format!("{:?}", a.kind)));
        artifacts
    }

    /// The op log after `mark` — a thin alias so every assertion reads
    /// `fx.ops_after(mark)` rather than reaching through to the store.
    fn ops_after(&self, mark: usize) -> Vec<StoreOp> {
        self.store.ops_since(mark)
    }

    async fn snapshot(&self, object_id: u64) -> StoreSnapshot {
        StoreSnapshot {
            listing: self.listing(object_id).await,
            artifacts: self.artifacts().await,
        }
    }

    /// `ComposedLineage::ext_values`' hash list for the planted lineage, in
    /// order, read through the SAME composition engine the unit uses.
    ///
    /// This is what the completeness claim is asserted against — the requirement
    /// names `ComposedLineage::ext_values`, not the fixture's own idea of what it
    /// planted, and the two are cross-checked once in
    /// [`rebase_backfills_exactly_the_composed_ext_set_and_never_the_base_only_hash`].
    /// Deriving it here rather than hard-coding also means the rows that target
    /// the FIRST and LAST part address whatever the engine actually orders first
    /// and last.
    async fn composed_ext_hash_hexes(&self, object_id: u64) -> Vec<String> {
        let (base_bytes, tail_bytes) = self.lineage_artifact_bytes(object_id).await;
        let composed = compose_from_artifacts(object_id, &base_bytes, BASE_SEQ, &tail_bytes)
            .expect("the planted lineage must compose — every record is a real OMPF");
        composed
            .ext_values
            .iter()
            .map(|(hash, _)| sha256_to_hex(hash))
            .collect()
    }

    /// The base OMPF bytes and the tail artifact bytes of the planted lineage,
    /// oldest tail first — the exact inputs the unit hands
    /// `compose_from_artifacts`.
    async fn lineage_artifact_bytes(&self, object_id: u64) -> (Bytes, Vec<Bytes>) {
        let mut listing = self.listing(object_id).await;
        listing.sort_by_key(|handle| handle.seq());
        let mut base_bytes = None;
        let mut tails = Vec::new();
        for handle in listing {
            let blob = self
                .store_dyn
                .get_layer_checkpoint(&handle)
                .await
                .expect("get_layer_checkpoint must succeed against a healthy tempdir")
                .expect("a listed generation must be fetchable");
            match handle.kind {
                CheckpointKind::Base => {
                    assert!(
                        base_bytes.is_none(),
                        "lineage_artifact_bytes is only valid BEFORE a successful re-base: a \
                         second base means the collapsed head already landed, and composing \
                         across two roots is meaningless"
                    );
                    base_bytes = Some(blob.bytes);
                }
                CheckpointKind::Tail => tails.push(blob.bytes),
            }
        }
        (
            base_bytes.expect("the planted lineage must carry a base"),
            tails,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared assertions
// ─────────────────────────────────────────────────────────────────────────────

/// Destructure a `Rebased` decision or fail naming what came back instead.
#[track_caller]
fn expect_rebased(decision: RebaseDecision, context: &str) -> (CheckpointHandle, usize) {
    match decision {
        RebaseDecision::Rebased {
            handle,
            base_snap_seq,
            from_chain_len,
            to_seq,
            ext_backfilled,
        } => {
            assert_eq!(
                to_seq, HEAD_TAIL_TO_SEQ,
                "{context}: the collapsed head must be S — the head tail's `to_seq`"
            );
            assert_eq!(
                handle.seq(),
                to_seq,
                "{context}: the new base is addressed at S"
            );
            assert_eq!(
                handle.kind,
                CheckpointKind::Base,
                "{context}: a chain collapse mints a BASE"
            );
            assert_eq!(
                base_snap_seq, BASE_SEQ,
                "{context}: the reported lineage root is the OLD base's sequence"
            );
            assert_eq!(
                from_chain_len, MAX_CHAIN,
                "{context}: the true link-walk chain length must be the K this fixture planted"
            );
            (handle, ext_backfilled)
        }
        other => panic!(
            "{context}: the lineage is intact, at K, and every backfill is healthy, so the unit \
             MUST collapse it; got {other:?}"
        ),
    }
}

/// SR-4.3's "no base PUT unless every inline part was acknowledged", plus
/// SR-4.5's "the lineage is untouched", in one place.
///
/// Extracted rather than copy-pasted per row (the test-review guide's
/// copy-paste-with-one-line-differences anti-pattern): the claims are identical
/// across the three refusal rows *by requirement*, and a per-row copy would let
/// one arm's claim silently weaken.
///
/// Note what is NOT asserted here: that the op log is EMPTY. Rows 4 and 7 refuse
/// *after* several parts were legitimately acknowledged, and those acks are
/// correct — the whole point of an all-of precondition is that partial progress
/// is allowed as long as the base never lands. The empty-log form is the sibling
/// refusal lane's, whose arms all refuse before the backfill loop runs.
#[track_caller]
fn assert_no_base_landed_and_lineage_intact(
    before: &StoreSnapshot,
    after: &StoreSnapshot,
    ops_since_mark: &[StoreOp],
    context: &str,
) {
    let puts: Vec<&StoreOp> = ops_since_mark
        .iter()
        .filter(|op| matches!(op, StoreOp::PutLayer { .. }))
        .collect();
    assert!(
        puts.is_empty(),
        "{context}: SR-4.3 makes durable acknowledgement of EVERY inline ext part a HARD \
         precondition, so an unmet backfill must issue no `put_layer_checkpoint` at all. A base \
         here would be one whose by-hash refs are backed only by live Valkey, and Task 4's \
         retention would then prune the tails that carried the bytes. Recorded: {puts:?}"
    );
    let destructive: Vec<&StoreOp> = ops_since_mark
        .iter()
        .filter(|op| {
            matches!(
                op,
                StoreOp::DeleteLayer { .. }
                    | StoreOp::DeleteExtBlob { .. }
                    | StoreOp::DeleteLayerTemp { .. }
                    | StoreOp::DeleteSession { .. }
            )
        })
        .collect();
    assert!(
        destructive.is_empty(),
        "{context}: this unit deletes NOTHING in any arm — lineage-aware retention owns every \
         deletion. Recorded: {destructive:?}"
    );
    assert_eq!(
        after.listing, before.listing,
        "{context}: the generation listing must be identical to the pre-attempt listing, element \
         for element and in order — that identity IS `the lineage is untouched`, and it is what \
         makes the next tick's re-attempt start from the same place"
    );
    assert_eq!(
        after.artifacts, before.artifacts,
        "{context}: the raw artifact enumeration must be unchanged too. The listing alone would \
         not catch a stray sidecar or a half-written stem, because neither becomes a listed \
         generation"
    );
    let temps: Vec<&LayerArtifact> = after
        .artifacts
        .iter()
        .filter(|a| a.kind == LayerArtifactKind::Temp)
        .collect();
    assert!(
        temps.is_empty(),
        "{context}: a refusal must leave no write-temp-then-rename leftover. A `.tmp` here means \
         the base PUT was started and abandoned, which is the partial artifact SR-4.5 forbids; \
         found {temps:?}"
    );
}

/// Every acknowledged backfill hash in `ops`, in acknowledgement order.
fn acked_hashes(ops: &[StoreOp]) -> Vec<String> {
    ops.iter()
        .filter_map(|op| op.acked_ext_hash().map(str::to_string))
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Ordering — every inline part ACKED before the base PUT is ISSUED
// ═════════════════════════════════════════════════════════════════════════════

/// **Row 1 (ordering proof).** Over a lineage carrying four distinct inline ext
/// parts, every `put_ext_blob` had RETURNED before the single base
/// `put_layer_checkpoint` was ISSUED.
///
/// The assertion is a recorded-index comparison over ONE interleaved op log —
/// `max(PutExt index) < PutLayer index` — never a bare count. A count of "4 ext
/// puts and 1 base put" is satisfied by every wrong ordering, including the
/// "PUT first, backfill best-effort" implementation this row exists to exclude;
/// only the relative index carries the claim. See the module header for why the
/// two sides are recorded at deliberately different moments (ext on completion,
/// base on issue), which is what makes this "acked before issued" rather than
/// the weaker "issued before issued".
///
/// The base PUT is also asserted to be the LAST op: with the max-index
/// comparison above, that closes the sequence — nothing was backfilled
/// afterwards either, which a "backfill the stragglers after the PUT"
/// implementation would show.
#[tokio::test]
async fn rebase_acks_every_inline_ext_part_before_it_issues_the_base_put() {
    // Arrange.
    let fx = backfill_fixture();
    fx.plant_lineage(OBJECT_ID_A).await;
    let mark = fx.store.op_mark();

    // Act.
    let decision = fx.rebase(OBJECT_ID_A).await;

    // Assert — the collapse landed.
    let (_handle, ext_backfilled) = expect_rebased(decision, "the ordering row");
    assert_eq!(
        ext_backfilled, EXPECTED_INLINE_EXT_COUNT,
        "the decision must report one acknowledgement per DISTINCT inline part"
    );

    // Assert — the interleaving.
    let ops = fx.ops_after(mark);
    let base_index = ops
        .iter()
        .position(|op| {
            matches!(
                op,
                StoreOp::PutLayer {
                    snap_seq,
                    kind: CheckpointKind::Base,
                    ..
                } if *snap_seq == HEAD_TAIL_TO_SEQ
            )
        })
        .unwrap_or_else(|| {
            panic!("the collapsed base must appear in the op log; recorded: {ops:?}")
        });
    let ext_indices: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, op)| matches!(op, StoreOp::PutExt { .. }))
        .map(|(index, _)| index)
        .collect();
    assert_eq!(
        ext_indices.len(),
        EXPECTED_INLINE_EXT_COUNT,
        "non-vacuity: the ordering claim is empty unless the lineage really carried inline parts. \
         Recorded: {ops:?}"
    );
    for op in ops.iter() {
        if let StoreOp::PutExt { hash_hex, ok } = op {
            assert!(
                *ok,
                "every backfill on the success path must be a durable ACK; {hash_hex} was recorded \
                 as failed"
            );
        }
    }
    let max_ext_index = *ext_indices
        .iter()
        .max()
        .expect("checked non-empty immediately above");
    assert!(
        max_ext_index < base_index,
        "SR-4.3: EVERY inline ext part must have been durably acknowledged before the base PUT was \
         issued. The last `put_ext_blob` completed at op index {max_ext_index} and the base PUT was \
         issued at index {base_index}. A base that precedes any ack is one whose by-hash refs are \
         backed only by live Valkey while retention prunes the tails that carried the bytes. Op \
         log: {ops:?}"
    );
    assert_eq!(
        base_index,
        ops.len() - 1,
        "the base PUT must be the LAST mutating call of the collapse. An op after it means \
         something was backfilled (or deleted) once the base was already durable, which is the \
         same stranded-generation hazard read from the other end. Op log: {ops:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. Completeness, dedup, and the D4 base-only residual
// ═════════════════════════════════════════════════════════════════════════════

/// **Row 2 (completeness).** The backfilled hashes are exactly
/// `ComposedLineage::ext_values`' hashes — every inline part, deduplicated, none
/// missed — and the base-only hash is neither backfilled nor lost.
///
/// Three claims, each excluding a different regression:
///
/// - **Completeness against the composed set.** Asserted as an ordered `Vec`
///   equality rather than a set equality: the backfill is sequential over
///   `ext_values`, so order is observable and asserting it additionally pins the
///   dedup's first-appearance ordering that `compose.rs` documents as part of its
///   contract. The fixture's own expected order is cross-checked against the
///   engine's in the same test, so a fixture that drifted would fail here rather
///   than silently re-defining the claim.
/// - **Dedup.** The cross-commit duplicate is backfilled exactly ONCE, not once
///   per referencing commit.
/// - **The D4 residual, in BOTH directions.** The base-only hash is never
///   `put_ext_blob`-ed (it was never inline, so there are no bytes to travel) and
///   the adapter still does not hold it afterwards — yet it IS listed in the new
///   base's ext-refs sidecar, because the composed state still references it.
///   Asserting only the negative would also pass for a re-base that had silently
///   dropped the reference; asserting only the positive would not pin the scope
///   decision at all.
#[tokio::test]
async fn rebase_backfills_exactly_the_composed_ext_set_and_never_the_base_only_hash() {
    // Arrange.
    let fx = backfill_fixture();
    fx.plant_lineage(OBJECT_ID_A).await;
    let composed_hashes = fx.composed_ext_hash_hexes(OBJECT_ID_A).await;
    assert_eq!(
        composed_hashes,
        fx.blobs.inline_hash_hexes(),
        "fixture integrity: the engine's `ComposedLineage::ext_values` order must be the \
         first-appearance order this fixture believes it planted (dup, a, b, c). A divergence here \
         means the fixture and the engine disagree about what the lineage carries, which would \
         make every claim below a statement about the wrong set"
    );
    let mark = fx.store.op_mark();

    // Act.
    let decision = fx.rebase(OBJECT_ID_A).await;

    // Assert — completeness.
    let (handle, ext_backfilled) = expect_rebased(decision, "the completeness row");
    let ops = fx.ops_after(mark);
    assert_eq!(
        acked_hashes(&ops),
        composed_hashes,
        "SR-4.3's precondition is over `ComposedLineage::ext_values` — every inline part, \
         deduplicated, in first-appearance order. The backfilled sequence must equal it exactly: \
         one missing hash is a base whose ref is backed only by live Valkey. Op log: {ops:?}"
    );
    assert_eq!(
        ext_backfilled,
        composed_hashes.len(),
        "the decision's reported count must be the composed set's size, not the number of \
         referencing commits"
    );

    // Assert — dedup.
    let dup_acks = acked_hashes(&ops)
        .iter()
        .filter(|hash| *hash == &fx.blobs.dup.hash_hex)
        .count();
    assert_eq!(
        dup_acks, 1,
        "the hash referenced by records 101 AND 103 must be backfilled exactly ONCE — the dedup is \
         lineage-wide (`compose.rs`'s `seen_hashes`), and a second put would mean the backfill is \
         iterating referencing commits rather than the deduplicated part list"
    );

    // Assert — the D4 residual, negative half.
    let base_only_acks = acked_hashes(&ops)
        .iter()
        .filter(|hash| *hash == &fx.blobs.base_only.hash_hex)
        .count();
    assert_eq!(
        base_only_acks, 0,
        "engineer decision D4: SR-4.3's hard precondition covers TAIL-INLINE parts only. The \
         base-only hash was never carried inline by any record, so there are no bytes for the \
         re-base to backfill and it must not invent any"
    );
    assert!(
        fx.store_dyn
            .get_ext_blob(&fx.blobs.base_only.hash_hex)
            .await
            .expect("probing the adapter must succeed")
            .is_none(),
        "the adapter must still not hold the base-only blob after the collapse — it keeps the \
         pre-existing best-effort live-mirror posture, which is the recorded D4 residual"
    );

    // Assert — the D4 residual, positive half.
    let sidecar = fx
        .store_dyn
        .read_ext_refs_sidecar(&handle)
        .await
        .expect("reading the new base's ext-refs sidecar must succeed");
    assert!(
        sidecar.contains(&fx.blobs.base_only.hash_hex),
        "the base-only hash must still be LISTED in the collapsed base's ext-refs sidecar: the \
         composed state still carries that external field, so dropping it from the sidecar would \
         un-pin a live reference and hand it to the orphan sweep. Sidecar: {sidecar:?}"
    );
    let inline_set: BTreeSet<String> = composed_hashes.iter().cloned().collect();
    let sidecar_set: BTreeSet<String> = sidecar.iter().cloned().collect();
    assert!(
        inline_set.is_subset(&sidecar_set),
        "every backfilled hash must also be listed in the new base's sidecar — a durable blob no \
         sidecar references is exactly what the reconciliation sweep collects. Backfilled: \
         {inline_set:?}; sidecar: {sidecar_set:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. Hard precondition — the FIRST part fails
// ═════════════════════════════════════════════════════════════════════════════

/// **Row 3 (first part fails).** A `Transient` failure on the FIRST
/// `put_ext_blob` refuses `ExtBackfill`, issues NO base PUT anywhere, leaves the
/// lineage intact — and the next attempt with the failure lifted lands the base.
///
/// This is the discriminator against a "PUT first, backfill best-effort"
/// implementation: such an implementation lands a base here and the op log shows
/// it.
///
/// Also asserted: the loop stopped at the failure. Exactly one `PutExt` op is
/// recorded, which is what "the FIRST `Err` refuses the WHOLE re-base" means at
/// the observable level — an implementation that logged the error and carried on
/// would record four.
///
/// Per the orchestrator's reconciliation, no `rebased`-did-not-tick assertion:
/// the unit holds no metrics handle, so on a unit-driven row the discriminator is
/// the `RebaseRefusal::ExtBackfill` variant plus the absent base PUT.
#[tokio::test]
async fn rebase_refuses_ext_backfill_when_the_first_part_fails_and_lands_on_the_retry() {
    // Arrange.
    let fx = backfill_fixture();
    fx.plant_lineage(OBJECT_ID_A).await;
    let composed_hashes = fx.composed_ext_hash_hexes(OBJECT_ID_A).await;
    let first = composed_hashes
        .first()
        .expect("the fixture lineage carries inline parts")
        .clone();
    let before = fx.snapshot(OBJECT_ID_A).await;
    let mark = fx.store.op_mark();
    fx.store.fail_put_ext(&first);

    // Act 1 — the failing tick.
    let decision = fx.rebase(OBJECT_ID_A).await;

    // Assert — the arm.
    assert!(
        matches!(
            decision,
            RebaseDecision::Refused(RebaseRefusal::ExtBackfill)
        ),
        "an ext-blob backfill that cannot be durably acknowledged must take the `ExtBackfill` arm \
         specifically — folding it into `BasePut` would say the base failed to persist when the \
         base was never attempted, and the two have different operator remediations; got \
         {decision:?}"
    );
    assert_eq!(
        fx.store.injected_failures(),
        1,
        "the injection must have fired exactly once, or this row's refusal came from somewhere \
         other than the failure it meant to inject"
    );

    // Assert — nothing landed, and the loop stopped at the failure.
    let ops = fx.ops_after(mark);
    assert_eq!(
        ops,
        vec![StoreOp::PutExt {
            hash_hex: first.clone(),
            ok: false,
        }],
        "the FIRST `Err` must refuse the WHOLE re-base: exactly one `put_ext_blob` attempt, no \
         further backfill, and no base PUT. More `PutExt` entries would mean the loop logged the \
         error and carried on, which is the weakened precondition SR-4.3 forbids"
    );
    let after = fx.snapshot(OBJECT_ID_A).await;
    assert_no_base_landed_and_lineage_intact(&before, &after, &ops, "first ext part failing");

    // Act 2 — the next tick, against a healed store.
    fx.store.heal();
    let retry_mark = fx.store.op_mark();
    let retry = fx.rebase(OBJECT_ID_A).await;

    // Assert — the retry converged. This is what separates "refused" from
    // "wedged", and it is also the positive control proving the refusal above
    // came from the injected failure rather than from a lineage built wrong.
    let (_handle, ext_backfilled) = expect_rebased(retry, "the retry after the first-part failure");
    assert_eq!(
        ext_backfilled,
        composed_hashes.len(),
        "the retry must backfill the WHOLE set, not merely the parts the failed attempt had not \
         reached — the precondition is over the composed list, and the unit keeps no cross-tick \
         progress state"
    );
    assert_eq!(
        acked_hashes(&fx.ops_after(retry_mark)),
        composed_hashes,
        "the retry's acknowledgement sequence must be the complete composed list"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. Hard precondition — the LAST part fails
// ═════════════════════════════════════════════════════════════════════════════

/// **Row 4 (last part fails).** The same claim with the failure moved to the
/// FINAL part — the row that proves the backfill is an all-of precondition
/// rather than a first-one check.
///
/// Without it, an implementation that awaits only the first acknowledgement and
/// fires the rest passes row 3 completely: its first put is awaited, so an
/// injected failure there still refuses. Only a failure on a LATER part can tell
/// the two apart, and this row additionally asserts that the three EARLIER parts
/// really were acknowledged — so it is falsifiable in the other direction too
/// (an implementation that refused before reaching the last part would fail it).
#[tokio::test]
async fn rebase_refuses_ext_backfill_when_the_last_part_fails_and_lands_on_the_retry() {
    // Arrange.
    let fx = backfill_fixture();
    fx.plant_lineage(OBJECT_ID_A).await;
    let composed_hashes = fx.composed_ext_hash_hexes(OBJECT_ID_A).await;
    let last = composed_hashes
        .last()
        .expect("the fixture lineage carries inline parts")
        .clone();
    let before = fx.snapshot(OBJECT_ID_A).await;
    let mark = fx.store.op_mark();
    fx.store.fail_put_ext(&last);

    // Act 1 — the failing tick.
    let decision = fx.rebase(OBJECT_ID_A).await;

    // Assert — the arm.
    assert!(
        matches!(
            decision,
            RebaseDecision::Refused(RebaseRefusal::ExtBackfill)
        ),
        "a failure on the LAST part must take the SAME `ExtBackfill` arm the first part's takes — \
         the precondition is all-of, and an implementation that only awaits the first \
         acknowledgement would land a base here; got {decision:?}"
    );
    assert_eq!(
        fx.store.injected_failures(),
        1,
        "the injection must have fired exactly once, against the last part"
    );

    // Assert — the earlier parts really were acked (so the row is falsifiable in
    // both directions), and the failure still refused the whole re-base.
    let ops = fx.ops_after(mark);
    let expected_prefix: Vec<String> = composed_hashes[..composed_hashes.len() - 1].to_vec();
    assert_eq!(
        acked_hashes(&ops),
        expected_prefix,
        "every part BEFORE the failing one must have been durably acknowledged — otherwise this \
         row would be indistinguishable from row 3 and would not prove the precondition is \
         all-of. Op log: {ops:?}"
    );
    assert_eq!(
        ops.last(),
        Some(&StoreOp::PutExt {
            hash_hex: last.clone(),
            ok: false,
        }),
        "the failing attempt must be the final recorded op: the refusal is immediate, so nothing \
         follows it. Op log: {ops:?}"
    );
    let after = fx.snapshot(OBJECT_ID_A).await;
    assert_no_base_landed_and_lineage_intact(&before, &after, &ops, "last ext part failing");

    // Act 2 — the next tick, against a healed store.
    fx.store.heal();
    let retry_mark = fx.store.op_mark();
    let retry = fx.rebase(OBJECT_ID_A).await;

    // Assert — the retry converged, re-putting the parts the failed attempt had
    // already acknowledged. That re-put is the idempotency SR-4.3 relies on, met
    // here on the retry path rather than in the abstract.
    let (_handle, ext_backfilled) = expect_rebased(retry, "the retry after the last-part failure");
    assert_eq!(ext_backfilled, composed_hashes.len());
    assert_eq!(
        acked_hashes(&fx.ops_after(retry_mark)),
        composed_hashes,
        "the retry must re-put the ALREADY-PRESENT parts as well as the failed one — `put_ext_blob` \
         is idempotent by hash, and skipping them would require the unit to keep progress state it \
         deliberately does not have"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. Idempotency by hash across a second re-base
// ═════════════════════════════════════════════════════════════════════════════

/// **Row 5 (idempotency by hash).** Re-driving a re-base against an adapter that
/// already holds every one of these hashes re-puts them all without error, and
/// the stored bytes are unchanged — the property the SR-9.2 duplicate-leader lane
/// (`t3.dfh.be.84.pa`) then relies on.
///
/// **Why the second drive is a second LINEAGE and not a second call on the
/// first.** After a successful collapse the store holds `base@S` *and* `tail@S`,
/// and `adapters::generation_sort_key` ranks `Base` ahead of `Tail` at equal
/// sequence — so the walk resolves the new base as the listing head, finds a
/// zero-tail lineage, and answers `UnderBound { true_chain_len: 0 }` without
/// reaching the backfill at all. That is correct behaviour (there is nothing left
/// to collapse), but it makes a literal repeat call useless as an idempotency
/// probe. The middle act step below ASSERTS that fact rather than asserting it in
/// prose, so a future change to the sort key surfaces here.
///
/// The second lineage is planted under a different `object_id` on the SAME store
/// root, carrying the SAME blobs. That is precisely the duplicate-leader shape at
/// the level SR-4.3 operates: `put_ext_blob` is content-addressed and
/// object-agnostic, so "a hash another leader already wrote" and "a hash another
/// object already wrote" are the same call.
#[tokio::test]
async fn rebase_backfill_is_idempotent_by_hash_across_a_second_rebase() {
    // Arrange + Act 1 — the first collapse.
    let fx = backfill_fixture();
    fx.plant_lineage(OBJECT_ID_A).await;
    let composed_hashes = fx.composed_ext_hash_hexes(OBJECT_ID_A).await;
    let first_mark = fx.store.op_mark();
    let first = fx.rebase(OBJECT_ID_A).await;
    expect_rebased(first, "the first collapse");
    assert_eq!(
        acked_hashes(&fx.ops_after(first_mark)),
        composed_hashes,
        "the first collapse must have backfilled the whole set, or the second drive is not \
         re-putting anything"
    );

    // Act 2 — the literal repeat, which is deliberately NOT the probe. See the
    // doc-comment: the landed base outranks the tail at S, so the walk sees a
    // zero-tail lineage.
    let repeat_mark = fx.store.op_mark();
    let repeat = fx.rebase(OBJECT_ID_A).await;
    assert!(
        matches!(repeat, RebaseDecision::UnderBound { true_chain_len: 0 }),
        "once the collapse has landed there is no chain left: `generation_sort_key` ranks the new \
         `base@S` ahead of `tail@S`, so the walk terminates immediately on the base. Anything else \
         means the unit is re-collapsing an already-collapsed lineage every tick; got {repeat:?}"
    );
    assert!(
        fx.ops_after(repeat_mark).is_empty(),
        "the `UnderBound` arm persists nothing at all — not a base, and not a backfill"
    );

    // Act 3 — the probe: an independent lineage over the SAME content-addressed
    // blobs, all of which the adapter already holds.
    fx.plant_lineage(OBJECT_ID_B).await;
    let second_mark = fx.store.op_mark();
    let second = fx.rebase(OBJECT_ID_B).await;

    // Assert.
    let (_handle, ext_backfilled) = expect_rebased(second, "the second lineage's collapse");
    assert_eq!(ext_backfilled, composed_hashes.len());
    assert_eq!(
        acked_hashes(&fx.ops_after(second_mark)),
        composed_hashes,
        "a re-put of bytes already present must SUCCEED and must not be skipped: SR-4.3 makes \
         `put_ext_blob` atomic and idempotent by hash precisely so two leaders racing on the same \
         hash cannot conflict, and the unit therefore never checks for presence first"
    );

    // Assert — the bytes did not change under the re-put.
    for blob in fx.blobs.inline_in_first_appearance_order() {
        let stored = fx
            .store_dyn
            .get_ext_blob(&blob.hash_hex)
            .await
            .expect("reading a backfilled blob must succeed")
            .unwrap_or_else(|| panic!("{} must still be present", blob.hash_hex));
        assert_eq!(
            stored.as_ref(),
            blob.bytes.as_slice(),
            "an idempotent re-put must leave the content-addressed bytes byte-identical for {}",
            blob.hash_hex
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. Post-condition readback
// ═════════════════════════════════════════════════════════════════════════════

/// **Row 6 (readback).** After a successful collapse, every hash in the new
/// base's ext-refs sidecar that came from an inline part is fetchable via
/// `get_ext_blob` and byte-identical to the bytes the tail record carried.
///
/// Rows 1 and 2 assert what the unit *called*; this row asserts what the store
/// *holds*. The distinction is not pedantic: a backfill that put the wrong bytes
/// under the right hash, or that put them under a stem the adapter's own reader
/// cannot address, satisfies every call-level claim and still loses the data —
/// and content-addressed storage makes the corrupted case indistinguishable
/// downstream from real content.
#[tokio::test]
async fn every_inline_hash_in_the_new_bases_sidecar_reads_back_byte_identically() {
    // Arrange.
    let fx = backfill_fixture();
    fx.plant_lineage(OBJECT_ID_A).await;
    let composed_hashes: BTreeSet<String> = fx
        .composed_ext_hash_hexes(OBJECT_ID_A)
        .await
        .into_iter()
        .collect();

    // Act.
    let decision = fx.rebase(OBJECT_ID_A).await;
    let (handle, _) = expect_rebased(decision, "the readback row");

    // Assert — the sidecar's inline half is exactly the composed set.
    let sidecar = fx
        .store_dyn
        .read_ext_refs_sidecar(&handle)
        .await
        .expect("reading the new base's ext-refs sidecar must succeed");
    let sidecar_inline: BTreeSet<String> = sidecar
        .iter()
        .filter(|hash| composed_hashes.contains(*hash))
        .cloned()
        .collect();
    assert_eq!(
        sidecar_inline, composed_hashes,
        "every inline part must still be referenced by the collapsed base's sidecar; a backfilled \
         blob the sidecar does not pin is what the reconciliation sweep collects. Sidecar: \
         {sidecar:?}"
    );

    // Assert — every one of them reads back byte-identically.
    for blob in fx.blobs.inline_in_first_appearance_order() {
        assert!(
            sidecar_inline.contains(&blob.hash_hex),
            "fixture integrity: {} must be one of the sidecar's inline hashes",
            blob.hash_hex
        );
        let stored = fx
            .store_dyn
            .get_ext_blob(&blob.hash_hex)
            .await
            .expect("get_ext_blob must succeed against a healthy tempdir")
            .unwrap_or_else(|| {
                panic!(
                    "{} was acknowledged by the backfill, so the adapter MUST hold it — an \
                     acknowledged write that cannot be read back is the stranded-generation \
                     hazard SR-4.3 exists to prevent, just one layer deeper",
                    blob.hash_hex
                )
            });
        assert_eq!(
            stored.as_ref(),
            blob.bytes.as_slice(),
            "the durable blob for {} must be byte-identical to what the tail record carried \
             inline — storing the wrong bytes under the right hash loses the data exactly as \
             thoroughly as not storing them, and is undetectable downstream",
            blob.hash_hex
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. Hard precondition — a part exceeds its per-blob budget (PM correction 6)
// ═════════════════════════════════════════════════════════════════════════════

/// **Row 7 (elapsed timeout).** A `put_ext_blob` that never answers takes the
/// SAME `RebaseRefusal::ExtBackfill` arm an injected `Transient` takes — the hard
/// precondition holds on a SLOW adapter, not only on a failing one.
///
/// The per-blob `tokio::time::timeout(cfg.rebase_fetch_reserve, ..)` is what
/// makes that true and is not optional: `ext_values` is bounded only by
/// `TAIL_MAX_BYTES x MAX_CHAIN`, so N can reach the hundreds, and an unbounded
/// `put_ext_blob` is capped only by the StorageApi adapter's ENFORCED per-call
/// ceiling of `request_timeout x 3 retries x 2 auth attempts` = 180 s. Without
/// the cap the loop's worst case is `N x 180 s`; with it, `N x reserve`.
///
/// The stall is targeted at the LAST part deliberately, so the row also shows
/// that earlier acknowledgements do not rescue a later overrun — the timeout is
/// part of the same all-of precondition row 4 pins for outright failures.
///
/// This row is the only one that pays [`REBASE_FETCH_RESERVE`] in wall time
/// (~2 s); every other test in the file returns in milliseconds. The stall is
/// `std::future::pending()` inside the store wrapper — fault injection, not
/// `sleep`-as-synchronization: nothing here waits on a clock for a result, and
/// the production timeout elapsing IS the thing under test.
#[tokio::test]
async fn rebase_refuses_ext_backfill_when_a_part_exceeds_its_per_blob_budget() {
    // Arrange.
    let fx = backfill_fixture();
    fx.plant_lineage(OBJECT_ID_A).await;
    let composed_hashes = fx.composed_ext_hash_hexes(OBJECT_ID_A).await;
    let last = composed_hashes
        .last()
        .expect("the fixture lineage carries inline parts")
        .clone();
    let before = fx.snapshot(OBJECT_ID_A).await;
    let mark = fx.store.op_mark();
    fx.store.stall_put_ext(&last);

    // Act 1 — the wedged tick.
    let decision = fx.rebase(OBJECT_ID_A).await;

    // Assert — the same arm a hard failure takes.
    assert!(
        matches!(
            decision,
            RebaseDecision::Refused(RebaseRefusal::ExtBackfill)
        ),
        "an ELAPSED per-blob budget must be the same refusal an injected `Transient` is: SR-4.3's \
         precondition is durable ACKNOWLEDGEMENT, and an adapter that never answers has \
         acknowledged nothing. Anything else — a distinct arm, or worse a base PUT — would make \
         the guarantee hold only against adapters that fail fast; got {decision:?}"
    );
    assert_eq!(
        fx.store.injected_stalls(),
        1,
        "the stall must have fired exactly once, or the refusal came from something other than the \
         elapsed budget"
    );
    assert_eq!(
        fx.store.ext_puts_issued(),
        composed_hashes.len(),
        "the backfill must have ISSUED every part up to and including the stalled one — the \
         stalled call records no op (it never completed), so the issue counter is what proves the \
         loop reached it rather than refusing earlier"
    );

    // Assert — nothing landed; the earlier acks stand but do not rescue it.
    let ops = fx.ops_after(mark);
    assert_eq!(
        acked_hashes(&ops),
        composed_hashes[..composed_hashes.len() - 1].to_vec(),
        "every part before the stalled one was legitimately acknowledged; the stalled one recorded \
         nothing at all. Op log: {ops:?}"
    );
    let after = fx.snapshot(OBJECT_ID_A).await;
    assert_no_base_landed_and_lineage_intact(
        &before,
        &after,
        &ops,
        "ext part exceeding its budget",
    );

    // Act 2 — the next tick, against a healed store.
    fx.store.heal();
    let retry_mark = fx.store.op_mark();
    let retry = fx.rebase(OBJECT_ID_A).await;

    // Assert — a slow adapter is a refusal the next tick retries, not a wedge.
    let (_handle, ext_backfilled) = expect_rebased(retry, "the retry after the elapsed budget");
    assert_eq!(ext_backfilled, composed_hashes.len());
    assert_eq!(
        acked_hashes(&fx.ops_after(retry_mark)),
        composed_hashes,
        "the retry must acknowledge the complete composed list"
    );
}
