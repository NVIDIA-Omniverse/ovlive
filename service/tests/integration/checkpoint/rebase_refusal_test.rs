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

//! SR-4.5 — the chain-collapse re-base's **refusal** arms: what a failed
//! re-base must leave behind, and what it must say. Task `t2.dfh.be.84`,
//! Subtask `t3.dfh.be.84.j7`.
//!
//! ## The claim under test
//!
//! > A re-base failure refuses **observably** and leaves the existing lineage
//! > intact; the layer re-attempts next tick. A *compose* failure additionally
//! > flags the lineage as suspect in the log — the early tripwire for a
//! > condition that would otherwise first surface at DR-restore time as an
//! > SR-3.5 step-down, i.e. during the incident, on state nobody can still act
//! > on.
//!
//! "Observably" decomposes into five separately-assertable facts, and every
//! refusal case here asserts all five through
//! [`assert_refusal_left_everything_intact`]:
//!
//! 1. the specific [`RebaseRefusal`] variant (a `matches!` on the variant, never
//!    `is_err()` — `docs-internal/test-writing-guide.md` § Assertions);
//! 2. the store saw **no mutating call at all** — no base PUT at `S`, no ext
//!    backfill, no delete, nothing (the op log below records every mutating
//!    trait method, not just `put_layer_checkpoint`);
//! 3. `list_layer_checkpoints` is identical to the pre-attempt listing,
//!    element for element and in order;
//! 4. `list_layer_artifacts` is identical too, and holds no
//!    [`LayerArtifactKind::Temp`] member — the "no partial artifact in any
//!    refusal arm" half;
//! 5. no cursor column moved on `le:{oid}:obj:hdr`.
//!
//! ## Why the cursor columns are read at all when the unit holds no Valkey
//!
//! [`rebase_layer_lineage`] takes `&dyn CheckpointStore` and nothing else, so it
//! is *structurally* unable to stamp a cursor — stamping is the loop's (design
//! decision D8). That makes the assertion a fence rather than a probe: it is the
//! test that fails if a future edit gives the unit a `LayerStore` and starts
//! stamping from inside it, which would double-stamp against the loop and, on a
//! refusal, advance a cursor past state no base covers.
//!
//! The fixture stamps **sentinel** values into the three columns before the act
//! step ([`SENTINEL_LAST_CHECKPOINT_SEQ`] and friends) rather than reading
//! whatever a fresh layer happens to carry. A freshly-created layer has those
//! columns absent, and "absent before, absent after" is a comparison that also
//! passes for a unit that cleared them. Non-`None` sentinels make both
//! directions — a write and a clear — visible.
//!
//! ## Instrumentation: a `DelegatingCheckpointStore` newtype, not the injector
//!
//! Same reasoning `restore_rebase_test.rs`'s header records, and the reason it
//! applies harder here: the fixtures below fail **one specific member fetch**,
//! keyed on `(snap_seq, kind)`. `CheckpointFaultInjector` keys on a method name,
//! so a `get_layer_checkpoint` injection would fail the head tail, the middle
//! tail and the base indistinguishably — and cases 1 and 2 exist precisely to
//! show the refusal is not tail-specific.
//!
//! ## Why this module is UNGATED
//!
//! No `#![cfg(feature = "…")]`, deliberately, for the reason
//! `restore_rebase_test.rs` states: a gated module compiles to nothing and
//! reports green (`docs-internal/test-writing-guide.md`), and this file is the
//! only evidence for SR-4.5's refusal contract at the unit's own boundary. It
//! would silently cover nothing under a feature-free narrow run.
//!
//! ## Scope fence — what deliberately is NOT here
//!
//! - **The ext-backfill refusal arm** ([`RebaseRefusal::ExtBackfill`]) is
//!   Subtask `t3.dfh.be.84.p4`'s, not an oversight. It needs that Subtask's
//!   ext-bearing lineage fixture, and duplicating that fixture here is exactly
//!   the fork Task 2's consolidated helper surface exists to prevent.
//! - **The `BasePut` refusal arm** rides the loop-level lane
//!   (`t3.dfh.be.84.1k`), which is also where every *metric* claim about a
//!   refusal lives: `rebase_layer_lineage` takes no metrics handle by design
//!   (its signature mirrors `gather_restore_lineage`'s no-`Config`/no-metrics
//!   cross-task contract), so a unit-driven lane has no cell to read. The
//!   observables here are the variant, the op log, the listing and the cursor.
//! - **`UnderBound`** (the SR-9.2 divergent-`S` fork) is not a refusal and is
//!   not this file's.
//!
//! ## Asserting on a log line, and why it is legitimate here
//!
//! `docs-internal/test-writing-guide.md` § What NOT to Test says exact log
//! messages are observability, not contracts. The suspect-lineage WARN is the
//! documented exception: SR-4.5 makes *raising it* the requirement — it is the
//! entire difference between "refuse and retry next tick" being safe and being
//! silent — and the refusal enum is field-free by design, so no structured
//! return value carries it. The repo precedent for asserting a log claim
//! without pinning its prose is
//! `lifecycle_session_publish_watermark_test.rs::publish_exhaustion_warns_that_the_blob_is_durable_but_unordered`;
//! this file follows it, matching a **distinctive substring**
//! ([`SUSPECT_LINEAGE_MARKER`]) plus the offending generation's structured
//! fields, never the whole message.
//!
//! `src/checkpoint/lineage.rs`'s own `capture_logs` (the other precedent the
//! Subtask names) is **not reachable from here**: it lives inside that module's
//! `#[cfg(test)] mod tests`, which `cfg(test)` makes false for an external
//! integration crate. The [`CapturingWriter`] below is therefore a file-local
//! copy of the `lifecycle_session_publish_watermark_test.rs` shape, kept
//! file-local because `helpers.rs` is Task 2's consolidated surface and this
//! lane consumes it rather than extending it.
//!
//! **`flavor = "current_thread"` is load-bearing** on the two log-capturing
//! cases — see [`capture_logs`].
//!
//! ## Bounded gates — no sleeps, no wall-clock assertions
//!
//! | Gate | Ceiling | What a timeout means |
//! |------|---------|----------------------|
//! | every `rebase_layer_lineage` call | [`REBASE_CALL_CEILING`] (30 s) | the unit wedged: it neither refused nor landed inside its own gather budget, which is a bug in the unit or in the adapter under it — never expected filesystem latency |
//! | the unit's internal gather budget | `(max_chain + 2) x` [`REBASE_FETCH_RESERVE`] = 25 s | a member fetch against a tempdir holding sub-kilobyte artifacts overran 5 s, i.e. the filesystem, not the code. It would surface as a `LineageBreak::Budget` step-down and therefore as a `LineageBroken` refusal, which no case here expects except case 5 |
//!
//! Worst case per test is case 3 at two ceilings (60 s), comfortably inside
//! `.config/nextest.toml`'s 120 s hard kill — a SIGKILLed test prints none of
//! its diagnostics. The healthy path returns in milliseconds and pays none of
//! it.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; under the
//! project sandbox that variable is preset (CLAUDE.md).
//!
//! ## Mutation evidence
//!
//! Four mutations were applied to `src/checkpoint/rebase.rs` in place and each
//! reverted byte-identically (`sha256` verified against the pre-mutation copy).
//! A lane that cannot name what it would catch is a lane nobody can audit later,
//! so each row names the assertion class it targets and the exact red count out
//! of this module's **6** tests. Reproduce by re-applying the mutation and
//! running the narrow filter in this file's Subtask body.
//!
//! | # | Mutation | Red | What it proves is asserted |
//! |---|---|---|---|
//! | M1 | the compose arm's suspect-lineage `warn!` is deleted outright (not demoted — the capture runs at `TRACE`, so a demotion is still caught) | 1 of 6 — case 4 | the tripwire itself. Every other claim in case 4 stayed green, so the WARN is the ONLY thing that assertion rests on |
//! | M2 | the compose arm calls `selection.record_compose_failure(head_seq, Tail)` before returning — design decision D6's forbidden move | 1 of 6 — case 4's `stepped_down_compose` negative | that the negative is real. Case 5's identical negative stayed green, which is correct: the mutation is on the compose arm |
//! | M3 | the member-fetch `Gather` arm issues a base PUT before returning its refusal | 3 of 6 — cases 1, 2 and 3 | the op-log claim. Every failing case named the recorded `PutLayer` in its message, and case 3 failed on its *pre-retry* refusal rather than on the retry — i.e. the retry case carries the refusal claim too |
//! | M4 | the SR-4.5 no-step-down fence is disabled (`if false && ..`) | 1 of 6 — case 5 | the fence. The decision degraded to `UnderBound { true_chain_len: 2 }` — the stepped-down lineage's shorter chain — which is exactly the silent wrong-generation collapse D6 forbids |
//!
//! The rows that turned exactly one test red are the measurement that says the
//! arm was missing rather than redundant.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::HashesInterface;
use tempfile::TempDir;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::{
    compose_from_artifacts, rebase_layer_lineage, CheckpointBlob, CheckpointError,
    CheckpointHandle, CheckpointKind, CheckpointMeta, CheckpointStore, LayerArtifact,
    LayerArtifactKind, RebaseDecision, RebaseRefusal, TailCaptureConfig,
};
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section};
use live_edit::storage::external_values::{ExternalValueStore, ValkeyExternalValueStore};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::{
    put_base_generation, put_tail_generation, put_tail_generation_from_records,
    HAND_PUT_WRITTEN_AT_UNIX_MS,
};
use crate::common::empty_reset_ompf;

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// `replica_id` handed to the unit. It reaches only `CheckpointMeta`, which no
/// refusal ever writes; carried so the successful re-base in case 3 stamps a
/// recognisable creator.
const REPLICA_ID: &str = "rebase-refusal-test-replica";

const CREATED_BY: &str = "rebase-refusal-test-user";
const SESSION_BASE_PATH: &str = "/rebase-refusal";
const LAYER_NAME: &str = "rebase-refusal-layer";

/// Stream retention for the fixture's `LayerStore` — the production default.
/// Nothing here reads the stream; the layer exists so `obj:hdr` does.
const STREAM_MAXLEN: u64 = 4096;

/// K. Deliberately 3 rather than the shipped 8: the chain length is a *bound*
/// this file must merely reach, and three tails keep the planted lineage (and
/// therefore the gather budget derived from it) small. The unit reads K from
/// [`TailCaptureConfig::max_chain`], so nothing here depends on the shipped
/// default.
const MAX_CHAIN: usize = 3;

/// The lineage root's `snap_seq`.
const BASE_SEQ: u64 = 100;

/// `to_seq` of the tail rooted directly at the base — `(100, 102]`.
const TAIL_1_TO_SEQ: u64 = 102;

/// `to_seq` of the middle tail — `(102, 104]`. This is the member cases 1 and 3
/// fail the GET for: a *middle* member, so the failure is neither the candidate
/// head the walk starts at nor the base it terminates on.
const TAIL_2_TO_SEQ: u64 = 104;

/// `to_seq` of the head tail — `(104, 106]`. Also **S**, the sequence a
/// successful collapse PUTs its new base at.
const HEAD_TAIL_TO_SEQ: u64 = 106;

/// Two commits per tail, not one: a single-record envelope makes an off-by-one
/// between a generation's claimed span and its carried records unobservable
/// (the b.pw6 lesson, applied to a fixture instead of a capture).
const TAIL_SPAN: u64 = 2;

/// The head tail's `from_seq` in case 5: a link into a generation that was never
/// written. Strictly between [`BASE_SEQ`] and [`HEAD_TAIL_TO_SEQ`] so the walk
/// resolves it as a **tail** predecessor (`from_seq != base_snap_seq`) rather
/// than as the lineage root — the same discipline
/// `restore_lineage_selection_test.rs::BROKEN_FROM_SEQ` records.
const BROKEN_FROM_SEQ: u64 = 105;

/// `to_seq` of case 5's dangling head tail — above [`HEAD_TAIL_TO_SEQ`] so it
/// heads the listing and is the first candidate the walk tries.
const BROKEN_HEAD_TO_SEQ: u64 = 107;

/// Per-member fetch reserve handed to the unit, which derives its whole-gather
/// deadline as `(max_chain + 2) x` this.
///
/// 5 s against a `tempfile::tempdir()` holding sub-kilobyte artifacts is roughly
/// four orders of magnitude of headroom. Sized down from the shipped 30 s so the
/// derived budget (25 s) stays well inside [`REBASE_CALL_CEILING`] and the
/// module header's worst-case arithmetic closes against nextest's 120 s kill.
const REBASE_FETCH_RESERVE: Duration = Duration::from_secs(5);

/// Ceiling on one `rebase_layer_lineage` call. Strictly above the unit's own
/// derived gather budget (25 s at the knobs above) so a real budget step-down
/// still returns a decision this file can assert on; reaching THIS ceiling means
/// the unit wedged without producing either.
const REBASE_CALL_CEILING: Duration = Duration::from_secs(30);

/// Sentinel `obj:hdr last_checkpoint_seq` planted before every act step. See the
/// module header for why non-`None` sentinels rather than a fresh layer's absent
/// columns.
const SENTINEL_LAST_CHECKPOINT_SEQ: u64 = 7;
/// Sentinel `obj:hdr last_checkpoint_ms`.
const SENTINEL_LAST_CHECKPOINT_MS: u64 = 1_700_000_000_123;
/// Sentinel `obj:hdr restore_seq_bump`. Non-zero so a unit that "retired" it
/// (wrote `0`) is caught as a change rather than mistaken for the value a
/// successful SR-3.6 re-base leaves.
const SENTINEL_RESTORE_SEQ_BUMP: u64 = 5;

/// The distinctive slice of the SR-4.5 suspect-lineage WARN.
///
/// Deliberately not the whole message and deliberately not a prefix any other
/// event in the composed-restore family shares: `warn_step_down`'s message
/// ("restore selection abandoned a candidate lineage and stepped down") and the
/// gather-failure WARNs carry none of these words.
const SUSPECT_LINEAGE_MARKER: &str = "SUSPECT LINEAGE";

/// `LineageBreak::Compose::reason_label()` — the `outcome` field
/// `LineageSelection::record_compose_failure` renders, and the ONLY producer of
/// it.
///
/// Its **absence** is how case 5 asserts design decision D6's "a broken or short
/// walk never calls `record_compose_failure`": the call is internal to the unit
/// and has no return value, so the WARN it unconditionally emits is the
/// observable.
const COMPOSE_STEP_DOWN_LABEL: &str = "stepped_down_compose";

/// `LineageBreak::LinkAbsent::reason_label()`. Case 5 asserts its **presence**,
/// which is what keeps that case's two negative claims from passing vacuously on
/// a fixture that never stepped down at all.
const LINK_BROKEN_STEP_DOWN_LABEL: &str = "stepped_down_link_broken";

/// Bytes planted as one tail record's payload in case 4: structurally a legal
/// record (the codec never parses a record payload, so this encodes and peeks
/// and decodes exactly like any other), semantically unreplayable — it is not an
/// OMPF container, so composition rejects it at `ComposeError::OmpfParse`.
const UNREPLAYABLE_RECORD_OMPF: &[u8] = b"this is not an OMPF container";

// ─────────────────────────────────────────────────────────────────────────────
// The recording / injecting store newtype
// ─────────────────────────────────────────────────────────────────────────────

/// One mutating call the wrapper saw, recorded whether or not it succeeded.
///
/// Every mutating `CheckpointStore` method has a variant — not just
/// `put_layer_checkpoint`. SR-4.5's claim is that a refusal writes **nothing**,
/// and a log that recorded only the base PUT would go green for a unit that had
/// started deleting superseded generations (which this unit must never do:
/// retention owns all deletion) or half-backfilling ext blobs.
#[derive(Debug, Clone, PartialEq, Eq)]
enum StoreWrite {
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
    PutExtBlob {
        hash_hex: String,
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

/// `CheckpointStore` newtype over a real [`LocalDiskCheckpointStore`] that
/// records every mutating call and can fail **one targeted member fetch**.
///
/// # Targeting
///
/// [`Self::fail_get_at`] keys on `(snap_seq, kind)`, which is what the
/// `CheckpointFaultInjector` seam cannot express — see the module header. The
/// injected error is [`CheckpointError::Transient`]: `gather_restore_lineage`
/// propagates store errors **verbatim** (a broken lineage is a verdict, a failed
/// store is an error), so the classification chosen here is what the unit's
/// gather arm sees, and `Transient` is the honest shape of "the adapter was
/// briefly unavailable" that SR-4.5's retry-next-tick claim is about.
///
/// # Delegation
///
/// Composed atop [`DelegatingCheckpointStore`](super::helpers::DelegatingCheckpointStore)
/// and forwards **every** method, including the four with non-trivial trait
/// defaults (`read_ext_refs_sidecar`, `list_layer_artifacts`,
/// `delete_layer_temp_artifacts`, `publish_worst_case`). Omitting those would
/// not leave a call un-instrumented, it would SUBSTITUTE behaviour: the artifact
/// listing would answer `Ok(vec![])` and this file's whole "no partial artifact"
/// claim would pass vacuously.
struct RecordingRebaseRefusalStore {
    delegate: super::helpers::DelegatingCheckpointStore<LocalDiskCheckpointStore>,
    writes: Mutex<Vec<StoreWrite>>,
    /// `Some((seq, kind))` while armed: `get_layer_checkpoint` for exactly that
    /// member returns `Transient` instead of delegating.
    fail_get_at: Mutex<Option<(u64, CheckpointKind)>>,
    /// How many times the armed injection actually fired. Read by every case
    /// that arms one, so a mis-targeted arm surfaces as a failed assertion
    /// rather than as a refusal that happened for some other reason.
    injected_get_failures: AtomicUsize,
}

impl RecordingRebaseRefusalStore {
    fn new(inner: Arc<LocalDiskCheckpointStore>) -> Self {
        Self {
            delegate: super::helpers::DelegatingCheckpointStore::new(inner),
            writes: Mutex::new(Vec::new()),
            fail_get_at: Mutex::new(None),
            injected_get_failures: AtomicUsize::new(0),
        }
    }

    /// Fail every `get_layer_checkpoint` addressed at `(snap_seq, kind)`.
    fn fail_get(&self, snap_seq: u64, kind: CheckpointKind) {
        *self.fail_get_at.lock().unwrap_or_else(|p| p.into_inner()) = Some((snap_seq, kind));
    }

    /// Lift the injection — the "next tick" of case 3.
    fn heal(&self) {
        *self.fail_get_at.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }

    fn injected_get_failures(&self) -> usize {
        self.injected_get_failures.load(Ordering::Acquire)
    }

    fn record(&self, write: StoreWrite) {
        self.writes
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(write);
    }

    /// Every mutating call recorded so far.
    fn writes(&self) -> Vec<StoreWrite> {
        self.writes
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Mutating calls recorded after `mark` (a snapshot of `writes().len()`
    /// taken immediately before the act step).
    fn writes_since(&self, mark: usize) -> Vec<StoreWrite> {
        self.writes().into_iter().skip(mark).collect()
    }

    fn write_mark(&self) -> usize {
        self.writes.lock().unwrap_or_else(|p| p.into_inner()).len()
    }
}

#[async_trait]
impl CheckpointStore for RecordingRebaseRefusalStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        // Recorded BEFORE delegation: the claim is that the call was never
        // ISSUED, which is stronger than "it was issued and failed".
        self.record(StoreWrite::PutLayer {
            object_id,
            snap_seq,
            kind: meta.kind,
        });
        self.delegate
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

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
        let armed = *self.fail_get_at.lock().unwrap_or_else(|p| p.into_inner());
        if armed == Some((handle.seq(), handle.kind)) {
            self.injected_get_failures.fetch_add(1, Ordering::AcqRel);
            return Err(CheckpointError::Transient {
                source: format!(
                    "synthetic transient member-fetch failure at ({}, {:?}) (SR-4.5 test)",
                    handle.seq(),
                    handle.kind
                )
                .into(),
            });
        }
        self.delegate.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.delegate.list_layer_checkpoints(object_id).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.record(StoreWrite::DeleteLayer {
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
        self.record(StoreWrite::PutSession {
            session_id: session_id.to_string(),
        });
        self.delegate
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.record(StoreWrite::DeleteSession {
            session_id: session_id.to_string(),
        });
        self.delegate.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.delegate.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.record(StoreWrite::PutExtBlob {
            hash_hex: hash_hex.to_string(),
        });
        self.delegate.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.delegate.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.delegate.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.record(StoreWrite::DeleteExtBlob {
            hash_hex: hash_hex.to_string(),
        });
        self.delegate.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.record(StoreWrite::PutCountersSidecar);
        self.delegate.put_counters_sidecar(body).await
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

    // ── The four non-trivial trait defaults ─────────────────────────────────
    //
    // Forwarded because omitting them SUBSTITUTES behaviour rather than merely
    // skipping instrumentation. `list_layer_artifacts`'s default returns
    // `Ok(vec![])`, which is exactly the value this file's no-partial-artifact
    // assertion expects to see UNCHANGED — so an omission here would make that
    // whole claim vacuously green.
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
        self.record(StoreWrite::DeleteLayerTemp {
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
// Fixture
// ─────────────────────────────────────────────────────────────────────────────

/// Everything one refusal case needs, owned together so `TestValkey`'s
/// prefix-scoped Drop cleanup and the `TempDir`'s recursive delete both outlive
/// the assertions.
struct RefusalFixture {
    tv: TestValkey,
    session_id: String,
    /// Held so its `Drop` removes the per-test checkpoint root. Never read.
    _dir: TempDir,
    store: Arc<RecordingRebaseRefusalStore>,
    store_dyn: Arc<dyn CheckpointStore>,
    object_id: u64,
}

/// A real session + a real layer in Valkey, a real `LocalDiskCheckpointStore`
/// over a fresh tempdir behind the recording newtype, and the three cursor
/// columns stamped with sentinels.
///
/// LocalDisk rather than InMemory on purpose: [`LayerArtifactKind::Temp`] is a
/// **filesystem** residual (`<final-name>.<uuid>.tmp` from the adapter's
/// write-temp-then-rename), and the InMemory adapter can never produce one. The
/// no-partial-artifact claim is only falsifiable on an adapter that has temps.
async fn refusal_fixture() -> RefusalFixture {
    let tv = TestValkey::setup().await;

    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("rebase-refusal-session", SESSION_BASE_PATH, CREATED_BY)
        .await
        .expect("session creation must succeed against a healthy Valkey");

    let layers = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), STREAM_MAXLEN);
    let ext_values: Arc<dyn ExternalValueStore> = Arc::new(ValkeyExternalValueStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));

    let object_id = layers
        .create_layer(
            &session_id,
            LAYER_NAME,
            CREATED_BY,
            &seed_state(),
            ext_values,
        )
        .await
        .expect("create_layer must succeed against a healthy Valkey");

    let dir = tempfile::tempdir().expect("tempfile::tempdir must succeed");
    let inner = Arc::new(
        LocalDiskCheckpointStore::new(dir.path().to_path_buf())
            .expect("LocalDiskCheckpointStore::new must succeed on a fresh tempdir"),
    );
    let store = Arc::new(RecordingRebaseRefusalStore::new(inner));
    let store_dyn: Arc<dyn CheckpointStore> = Arc::clone(&store) as Arc<dyn CheckpointStore>;

    let fx = RefusalFixture {
        tv,
        session_id,
        _dir: dir,
        store,
        store_dyn,
        object_id,
    };
    fx.stamp_sentinel_cursor_columns().await;
    fx
}

/// A NON-TRIVIAL creation state. Deliberately not `ObjectState::default()`: the
/// layer exists so `le:{oid}:obj:hdr` exists, and a layer created from nothing
/// is a shape no production path produces.
fn seed_state() -> ObjectState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 0,
        ..Default::default()
    };
    let section_id = 1u64;
    state.sections.insert(
        section_id,
        Section {
            section_id,
            parent_id: 0,
            name: "rebase-refusal-sec".to_string(),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        },
    );
    state.fields.insert(
        (section_id, "k1".to_string()),
        Field {
            section_id,
            key_name: "k1".to_string(),
            value: FieldValue::Inline(vec![0xAB, 0xCD]),
            set_order: 1,
            authoring_set_order: 1,
            version: 1,
        },
    );
    state.children.insert(0, vec![section_id]);
    state
}

/// The four `obj:hdr` columns every immobility assertion reads, in ONE HMGET so
/// they cannot disagree about which write they describe.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CursorColumns {
    /// The live writer cursor. Not a checkpoint column, but read alongside them:
    /// a unit that touched the header at all would most likely move this too.
    seq: Option<String>,
    last_checkpoint_seq: Option<String>,
    last_checkpoint_ms: Option<String>,
    restore_seq_bump: Option<String>,
}

impl RefusalFixture {
    fn header_key(&self) -> String {
        self.tv.key_schema().obj_header(&self.object_id.to_string())
    }

    /// Plant the sentinels. See the module header for why non-`None` values.
    async fn stamp_sentinel_cursor_columns(&self) {
        let _: u64 = self
            .tv
            .client()
            .pool
            .hset(
                &self.header_key(),
                vec![
                    (
                        "last_checkpoint_seq",
                        SENTINEL_LAST_CHECKPOINT_SEQ.to_string(),
                    ),
                    (
                        "last_checkpoint_ms",
                        SENTINEL_LAST_CHECKPOINT_MS.to_string(),
                    ),
                    ("restore_seq_bump", SENTINEL_RESTORE_SEQ_BUMP.to_string()),
                ],
            )
            .await
            .expect("HSET of the sentinel cursor columns must succeed");
    }

    async fn cursor_columns(&self) -> CursorColumns {
        let raw: Vec<Option<String>> = self
            .tv
            .client()
            .pool
            .hmget(
                &self.header_key(),
                vec![
                    "seq",
                    "last_checkpoint_seq",
                    "last_checkpoint_ms",
                    "restore_seq_bump",
                ],
            )
            .await
            .expect("HMGET obj:hdr cursor columns must succeed");
        CursorColumns {
            seq: raw.first().cloned().flatten(),
            last_checkpoint_seq: raw.get(1).cloned().flatten(),
            last_checkpoint_ms: raw.get(2).cloned().flatten(),
            restore_seq_bump: raw.get(3).cloned().flatten(),
        }
    }

    async fn listing(&self) -> Vec<CheckpointHandle> {
        self.store_dyn
            .list_layer_checkpoints(self.object_id)
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

    /// Listing + artifacts + cursor columns, read together as the pre-attempt
    /// reference.
    async fn snapshot(&self) -> StoreSnapshot {
        StoreSnapshot {
            listing: self.listing().await,
            artifacts: self.artifacts().await,
            cursor: self.cursor_columns().await,
        }
    }

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
    async fn rebase(&self) -> RebaseDecision {
        tokio::time::timeout(
            REBASE_CALL_CEILING,
            rebase_layer_lineage(
                self.store_dyn.as_ref(),
                self.object_id,
                REPLICA_ID,
                &self.tail_config(),
            ),
        )
        .await
        .expect(
            "rebase_layer_lineage must produce a decision inside REBASE_CALL_CEILING — the unit's \
             own gather budget is strictly smaller, so reaching this ceiling means the unit \
             wedged rather than that the tempdir was slow",
        )
        .expect(
            "rebase_layer_lineage must not Err: NotFound is its ONLY error arm and this fixture's \
             layer never vanishes; every SR-4.5 cause is a RebaseDecision",
        )
    }

    // ── Lineage planting ────────────────────────────────────────────────────

    /// `base@100`, `tail(100,102]`, `tail(102,104]`, `tail(104,106]` — a
    /// well-formed chain of exactly K, every record a real reset OMPF that
    /// composes.
    ///
    /// Built from the Task 2 helpers rather than a local encoder: they already
    /// write real `encode_tail_envelope` bytes and identity-complete metas, and
    /// a second seeder would be free to drift from them.
    async fn plant_intact_chain(&self) {
        put_base_generation(
            &self.store_dyn,
            self.object_id,
            &self.session_id,
            LAYER_NAME,
            BASE_SEQ,
        )
        .await;
        for (from_seq, to_seq) in [
            (BASE_SEQ, TAIL_1_TO_SEQ),
            (TAIL_1_TO_SEQ, TAIL_2_TO_SEQ),
            (TAIL_2_TO_SEQ, HEAD_TAIL_TO_SEQ),
        ] {
            put_tail_generation(
                &self.store_dyn,
                self.object_id,
                &self.session_id,
                LAYER_NAME,
                BASE_SEQ,
                from_seq,
                to_seq,
            )
            .await;
        }
    }

    /// The intact chain's first two tails, plus a head tail whose SECOND record
    /// carries [`UNREPLAYABLE_RECORD_OMPF`].
    ///
    /// Structurally impeccable and semantically fatal, which is the shape this
    /// case needs: the tail codec never parses a record payload, so the envelope
    /// encodes, its header peeks, the link walk verifies it and the gather
    /// returns it with NO step-down — and composition then rejects it at
    /// `ComposeError::OmpfParse`. A malformed *envelope* would have been caught
    /// one layer earlier as a `LineageBreak::Decode` step-down and produced
    /// `LineageBroken`, which is case 5's arm, not this one's.
    ///
    /// The first record is a real reset OMPF so replay genuinely reaches the
    /// second: a head tail that failed on record 0 would not distinguish "the
    /// engine replayed and hit a bad record" from "the engine refused the
    /// generation outright".
    async fn plant_unreplayable_head_tail(&self) {
        put_base_generation(
            &self.store_dyn,
            self.object_id,
            &self.session_id,
            LAYER_NAME,
            BASE_SEQ,
        )
        .await;
        for (from_seq, to_seq) in [(BASE_SEQ, TAIL_1_TO_SEQ), (TAIL_1_TO_SEQ, TAIL_2_TO_SEQ)] {
            put_tail_generation(
                &self.store_dyn,
                self.object_id,
                &self.session_id,
                LAYER_NAME,
                BASE_SEQ,
                from_seq,
                to_seq,
            )
            .await;
        }
        let records = vec![
            (TAIL_2_TO_SEQ + 1, empty_reset_ompf()),
            (
                HEAD_TAIL_TO_SEQ,
                Bytes::from_static(UNREPLAYABLE_RECORD_OMPF),
            ),
        ];
        assert_eq!(
            records.len() as u64,
            TAIL_SPAN,
            "the hand-built head tail must carry the same span as its siblings, or the chain \
             shape this case shares with the others silently diverges"
        );
        put_tail_generation_from_records(
            &self.store_dyn,
            self.object_id,
            &self.session_id,
            LAYER_NAME,
            BASE_SEQ,
            TAIL_2_TO_SEQ,
            &records,
            HAND_PUT_WRITTEN_AT_UNIX_MS,
            0,
        )
        .await;
    }

    /// The intact chain's first two tails, plus a head tail at
    /// [`BROKEN_HEAD_TO_SEQ`] whose `from_seq` names a generation that was never
    /// written.
    ///
    /// The walk therefore abandons the listing head on `LineageBreak::LinkAbsent`
    /// and steps down to `tail(102,104]`, which verifies. The gather answers
    /// `Ok(Some(..))` with a NON-EMPTY step-down list, which is the exact input
    /// SR-4.5's no-step-down fence exists for: the returned lineage is not the
    /// listing head's, and minting a base from it would collapse the wrong
    /// generations.
    async fn plant_broken_head_link(&self) {
        put_base_generation(
            &self.store_dyn,
            self.object_id,
            &self.session_id,
            LAYER_NAME,
            BASE_SEQ,
        )
        .await;
        for (from_seq, to_seq) in [(BASE_SEQ, TAIL_1_TO_SEQ), (TAIL_1_TO_SEQ, TAIL_2_TO_SEQ)] {
            put_tail_generation(
                &self.store_dyn,
                self.object_id,
                &self.session_id,
                LAYER_NAME,
                BASE_SEQ,
                from_seq,
                to_seq,
            )
            .await;
        }
        put_tail_generation(
            &self.store_dyn,
            self.object_id,
            &self.session_id,
            LAYER_NAME,
            BASE_SEQ,
            BROKEN_FROM_SEQ,
            BROKEN_HEAD_TO_SEQ,
        )
        .await;
    }

    /// The base OMPF bytes and the tail artifact bytes of the CURRENT lineage,
    /// oldest tail first — the exact inputs the unit hands
    /// `compose_from_artifacts`.
    ///
    /// Read straight off the store rather than re-derived, so case 6 composes
    /// the same bytes case 4 refused.
    async fn lineage_artifact_bytes(&self) -> (Bytes, Vec<Bytes>) {
        let mut listing = self.listing().await;
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
                CheckpointKind::Base => base_bytes = Some(blob.bytes),
                CheckpointKind::Tail => tails.push(blob.bytes),
            }
        }
        (
            base_bytes.expect("the planted lineage must carry a base"),
            tails,
        )
    }
}

/// The pre-attempt reference every refusal is compared against.
struct StoreSnapshot {
    listing: Vec<CheckpointHandle>,
    artifacts: Vec<LayerArtifact>,
    cursor: CursorColumns,
}

// ─────────────────────────────────────────────────────────────────────────────
// The shared post-refusal assertion
// ─────────────────────────────────────────────────────────────────────────────

/// SR-4.5's "leaves the existing lineage intact", in one place.
///
/// Every refusal case calls this with its own pre-attempt [`StoreSnapshot`] and
/// write mark. Extracted rather than copy-pasted per case (the test-review
/// guide's copy-paste-with-one-line-differences anti-pattern): the four claims
/// are identical across arms *by requirement*, and a per-case copy would let one
/// arm's claim silently weaken.
#[track_caller]
fn assert_refusal_left_everything_intact(
    before: &StoreSnapshot,
    after: &StoreSnapshot,
    writes_after_mark: &[StoreWrite],
    context: &str,
) {
    assert!(
        writes_after_mark.is_empty(),
        "{context}: a refusal must issue NO mutating store call — no base PUT at the collapsed \
         head, no ext backfill, and no delete (retention owns every deletion, not this unit). \
         Recorded: {writes_after_mark:?}"
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
    assert_eq!(
        after.cursor, before.cursor,
        "{context}: no `obj:hdr` cursor column may move. The unit holds no Valkey client at all \
         (design decision D8 puts the stamp on the LOOP), so a difference here means the unit \
         grew a header write — which on a refusal would advance a cursor past state no base \
         covers"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Log capture
// ─────────────────────────────────────────────────────────────────────────────

/// Drive `f` under a thread-local `tracing` subscriber and return its value
/// alongside everything it emitted.
///
/// **`flavor = "current_thread"` is load-bearing on every caller.**
/// `tracing::subscriber::set_default` installs the dispatcher in a
/// **thread-local**, and a `#[tokio::test]` future is polled on the calling
/// thread only under the current-thread runtime. On a multi-thread runtime the
/// future can resume on a worker thread after any `.await` — and
/// `rebase_layer_lineage` awaits a `spawn_blocking` join in the middle of the
/// very path whose WARN is under test — so the event would be emitted with no
/// dispatcher installed and the capture buffer would come back empty. The test
/// would then report "the tripwire was not raised" for a unit that raised it,
/// which is the worst failure mode available. Same pattern and same rationale as
/// `lifecycle_session_publish_watermark_test.rs::publish_exhaustion_warns_that_the_blob_is_durable_but_unordered`.
///
/// `with_max_level(TRACE)` rather than `WARN`: the negative assertions (case 5's
/// "no compose tripwire") must be able to see an event that was emitted at a
/// lower level, or a mutation that merely demoted the WARN to `debug!` would
/// pass.
async fn capture_logs<T>(f: impl std::future::Future<Output = T>) -> (T, String) {
    let buf = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(CapturingWriter(Arc::clone(&buf)))
        .with_max_level(tracing::Level::TRACE)
        // ANSI off: the assertions match plain substrings, and colour escapes
        // land inside them.
        .with_ansi(false)
        .finish();
    let out = {
        let _guard = tracing::subscriber::set_default(subscriber);
        f.await
    };
    let captured = buf.lock().unwrap_or_else(|p| p.into_inner()).clone();
    (
        out,
        String::from_utf8(captured).expect("captured tracing output must be UTF-8"),
    )
}

/// A `MakeWriter` that captures everything written to it.
#[derive(Clone)]
struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturingWriter {
    type Writer = CapturingWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

impl std::io::Write for CapturingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Gather — a middle TAIL's fetch fails
// ─────────────────────────────────────────────────────────────────────────────

/// **Case 1.** A store failure on one middle tail's `get_layer_checkpoint`
/// refuses `Gather` and writes nothing.
///
/// The member is chosen to be neither the candidate head the walk starts at nor
/// the base it terminates on, so the refusal cannot be an artefact of failing
/// the first or last fetch. `Transient` is deliberate: on a read path that
/// classification means "retry me", and SR-4.5's whole claim is that here it
/// becomes a *decision* the next tick retries rather than an `Err` the caller
/// must classify.
#[tokio::test]
async fn rebase_refuses_gather_when_a_middle_tail_fetch_fails_and_leaves_the_lineage_intact() {
    // Arrange.
    let fx = refusal_fixture().await;
    fx.plant_intact_chain().await;
    let before = fx.snapshot().await;
    let mark = fx.store.write_mark();
    fx.store.fail_get(TAIL_2_TO_SEQ, CheckpointKind::Tail);

    // Act.
    let decision = fx.rebase().await;

    // Assert.
    assert!(
        matches!(decision, RebaseDecision::Refused(RebaseRefusal::Gather)),
        "a failed member fetch is a STORE failure, not a lineage verdict, so it must take the \
         `Gather` arm — `LineageBroken` would claim the persisted lineage is defective when \
         nothing about it is known; got {decision:?}"
    );
    assert_eq!(
        fx.store.injected_get_failures(),
        1,
        "the injection must have fired exactly once, or this case's refusal came from somewhere \
         other than the failure it meant to inject"
    );
    let after = fx.snapshot().await;
    assert_refusal_left_everything_intact(
        &before,
        &after,
        &fx.store.writes_since(mark),
        "middle-tail fetch failure",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Gather — the BASE's fetch fails
// ─────────────────────────────────────────────────────────────────────────────

/// **Case 2.** The same arm, reached through the lineage ROOT instead of a tail —
/// so the refusal is not tail-specific.
///
/// Worth its own case rather than a parameter: the base is the one member the
/// walk reaches last and the only one whose meta is identity-bearing, so a
/// hypothetical implementation that special-cased it (retried it, or fell back
/// to `get_latest_layer_checkpoint`) would pass case 1 and fail here.
#[tokio::test]
async fn rebase_refuses_gather_when_the_base_fetch_fails_and_leaves_the_lineage_intact() {
    // Arrange.
    let fx = refusal_fixture().await;
    fx.plant_intact_chain().await;
    let before = fx.snapshot().await;
    let mark = fx.store.write_mark();
    fx.store.fail_get(BASE_SEQ, CheckpointKind::Base);

    // Act.
    let decision = fx.rebase().await;

    // Assert.
    assert!(
        matches!(decision, RebaseDecision::Refused(RebaseRefusal::Gather)),
        "a failed BASE fetch must take the same `Gather` arm a failed tail fetch takes — the \
         refusal is about the store, not about which member the store failed; got {decision:?}"
    );
    assert_eq!(
        fx.store.injected_get_failures(),
        1,
        "the injection must have fired exactly once against the base"
    );
    let after = fx.snapshot().await;
    assert_refusal_left_everything_intact(
        &before,
        &after,
        &fx.store.writes_since(mark),
        "base fetch failure",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Retry next tick
// ─────────────────────────────────────────────────────────────────────────────

/// **Case 3.** The half that separates "refused" from "wedged": lift the
/// injection and re-drive, and the SAME layer collapses.
///
/// This is what SR-4.5's "the next tick re-attempts" actually claims, and it is
/// also cases 1 and 2's positive control — it proves their refusals came from
/// the injected failure and not from a lineage the fixture built wrong, because
/// the second drive runs against byte-identical planted artifacts.
///
/// The success assertions are deliberately narrow: this file owns the refusal
/// arms, and the landed-base proof matrix (byte determinism, SR-9.2 duplicate
/// leaders, the metric cells) belongs to the sibling lanes. What is asserted
/// here is the shape that makes "the retry converged" meaningful — the base
/// landed at `S`, it is the ONLY thing written, the old lineage survived it, and
/// the cursor columns STILL did not move (the loop stamps, not the unit).
#[tokio::test]
async fn rebase_succeeds_on_the_next_attempt_once_the_injected_fetch_failure_is_lifted() {
    // Arrange — the exact fixture case 1 refuses on.
    let fx = refusal_fixture().await;
    fx.plant_intact_chain().await;
    let before = fx.snapshot().await;
    let refusal_mark = fx.store.write_mark();
    fx.store.fail_get(TAIL_2_TO_SEQ, CheckpointKind::Tail);

    // Act 1 — the failing tick.
    let refused = fx.rebase().await;
    assert!(
        matches!(refused, RebaseDecision::Refused(RebaseRefusal::Gather)),
        "the first drive must refuse, or the retry it is supposed to demonstrate is vacuous; got \
         {refused:?}"
    );
    let after_refusal = fx.snapshot().await;
    assert_refusal_left_everything_intact(
        &before,
        &after_refusal,
        &fx.store.writes_since(refusal_mark),
        "the refused tick preceding the retry",
    );

    // Act 2 — the next tick, against a healed store.
    fx.store.heal();
    let retry_mark = fx.store.write_mark();
    let decision = fx.rebase().await;

    // Assert.
    let RebaseDecision::Rebased {
        handle,
        base_snap_seq,
        from_chain_len,
        to_seq,
        ext_backfilled,
    } = decision
    else {
        panic!(
            "once the injected failure is lifted the SAME layer must collapse — a refusal that \
             survives the fault clearing is a wedge, which is precisely what SR-4.5's \
             retry-next-tick claim rules out; got {decision:?}"
        );
    };
    assert_eq!(
        to_seq, HEAD_TAIL_TO_SEQ,
        "the collapsed head must be S — the head tail's `to_seq`"
    );
    assert_eq!(handle.seq(), to_seq, "the new base is addressed at S");
    assert_eq!(
        handle.kind,
        CheckpointKind::Base,
        "a chain collapse mints a BASE; a tail here would leave the chain uncollapsed"
    );
    assert_eq!(
        base_snap_seq, BASE_SEQ,
        "the reported lineage root is the OLD base's sequence — the generation retention will \
         later prune the superseded lineage from"
    );
    assert_eq!(
        from_chain_len, MAX_CHAIN,
        "the true link-walk chain length must be the K this fixture planted, read from the walk \
         rather than from the listing's over-approximation (SR-9.2)"
    );
    assert_eq!(
        ext_backfilled, 0,
        "the planted records reference no inline external values, so SR-4.3's backfill loop must \
         have had nothing to acknowledge"
    );

    assert_eq!(
        fx.store.writes_since(retry_mark),
        vec![StoreWrite::PutLayer {
            object_id: fx.object_id,
            snap_seq: HEAD_TAIL_TO_SEQ,
            kind: CheckpointKind::Base,
        }],
        "the successful collapse must be ONE base PUT and nothing else: no delete (retention owns \
         every deletion, and the superseded lineage must survive this unit) and no ext backfill \
         on a lineage that carries none"
    );

    let after = fx.snapshot().await;
    for handle in &before.listing {
        assert!(
            after.listing.contains(handle),
            "every pre-collapse generation must survive the re-base — the old lineage stays \
             complete and restorable until retention prunes it as a unit. Missing: {handle:?}"
        );
    }
    assert_eq!(
        after.cursor, before.cursor,
        "even on the SUCCESS arm the unit must not stamp a cursor column: it holds no Valkey \
         client, and design decision D8 puts the stamp on the loop"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Compose failure + the suspect-lineage tripwire
// ─────────────────────────────────────────────────────────────────────────────

/// **Case 4.** A lineage that gathers cleanly but will not compose refuses
/// `Compose`, writes nothing, and raises the SR-4.5 suspect-lineage tripwire.
///
/// The tripwire is the point of the case. Without it, an unrestorable persisted
/// lineage stays invisible until a DR restore selects it and steps down (SR-3.5)
/// — during the incident, on state nobody can still act on. The assertion
/// matches [`SUSPECT_LINEAGE_MARKER`] plus the offending generation's structured
/// fields, never the whole message; see the module header for why asserting a
/// log line is legitimate here at all.
///
/// `flavor = "current_thread"` is load-bearing — see [`capture_logs`].
#[tokio::test(flavor = "current_thread")]
async fn rebase_refuses_compose_and_raises_the_suspect_lineage_tripwire() {
    // Arrange.
    let fx = refusal_fixture().await;
    fx.plant_unreplayable_head_tail().await;
    let before = fx.snapshot().await;
    let mark = fx.store.write_mark();

    // Act.
    let (decision, logs) = capture_logs(fx.rebase()).await;

    // Assert — the decision.
    assert!(
        matches!(decision, RebaseDecision::Refused(RebaseRefusal::Compose)),
        "a lineage whose records will not replay must take the `Compose` arm specifically: it is \
         the only arm that raises the tripwire, so folding it into `Gather` or `LineageBroken` \
         would silently retire the SR-4.5 early warning; got {decision:?}"
    );

    // Assert — the tripwire.
    assert!(
        logs.contains(SUSPECT_LINEAGE_MARKER),
        "the compose arm MUST raise the suspect-lineage tripwire — it is the only signal that an \
         unrestorable lineage exists BEFORE a DR restore trips over it (SR-3.5). Captured: {logs}"
    );
    for field in [
        format!("object_id={}", fx.object_id),
        format!("base_snap_seq={BASE_SEQ}"),
        format!("chain_len={MAX_CHAIN}"),
        format!("to_seq={HEAD_TAIL_TO_SEQ}"),
        // The offending generation, which is the whole point of the structured
        // fields: `Display` on the error carries it too, but an operator must
        // be able to query it without parsing a message.
        format!("offending_to_seq=Some({HEAD_TAIL_TO_SEQ})"),
    ] {
        assert!(
            logs.contains(&field),
            "the suspect-lineage WARN MUST carry the structured field {field:?} — it names WHICH \
             lineage and WHICH generation is suspect, and the field-free `RebaseRefusal` enum \
             carries neither. Captured: {logs}"
        );
    }
    assert!(
        !logs.contains(COMPOSE_STEP_DOWN_LABEL),
        "design decision D6: the re-base never calls `record_compose_failure`. Its \
         `{COMPOSE_STEP_DOWN_LABEL}` step-down WARN appearing here would mean the unit walked to \
         an older lineage — which is the restore path's move, and here it would collapse the \
         wrong generations. Captured: {logs}"
    );

    // Assert — nothing moved.
    let after = fx.snapshot().await;
    assert_refusal_left_everything_intact(
        &before,
        &after,
        &fx.store.writes_since(mark),
        "compose failure",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. A broken walk — LineageBroken, and NO tripwire
// ─────────────────────────────────────────────────────────────────────────────

/// **Case 5.** A lineage whose listing head links into a generation that was
/// never written refuses `LineageBroken` — and does **not** raise the compose
/// tripwire.
///
/// That negative is design decision D6's whole point: the tripwire must stay
/// attributable to a genuine composition failure. A `LineageBroken` refusal says
/// the *walk* could not verify the head, which is a different operator condition
/// with a different remediation, and firing "SUSPECT LINEAGE" on it would train
/// the reader to ignore it.
///
/// The positive half — the `stepped_down_link_broken` WARN — is what keeps the
/// two negatives from passing vacuously: it proves the walk really did abandon
/// the listing head and hand the unit a stepped-down lineage, which is the input
/// SR-4.5's no-step-down fence exists to refuse.
///
/// `flavor = "current_thread"` is load-bearing — see [`capture_logs`].
#[tokio::test(flavor = "current_thread")]
async fn rebase_refuses_lineage_broken_on_a_stepped_down_walk_without_raising_the_tripwire() {
    // Arrange.
    let fx = refusal_fixture().await;
    fx.plant_broken_head_link().await;
    let before = fx.snapshot().await;
    let mark = fx.store.write_mark();

    // Act.
    let (decision, logs) = capture_logs(fx.rebase()).await;

    // Assert — the decision.
    assert!(
        matches!(
            decision,
            RebaseDecision::Refused(RebaseRefusal::LineageBroken)
        ),
        "a walk that had to step down must refuse `LineageBroken`: the verified lineage is not \
         the listing head's, and minting a base from an older one would publish a head that \
         supersedes a lineage nobody adjudicated as unrestorable; got {decision:?}"
    );

    // Assert — the walk really stepped down (non-vacuity for the two negatives
    // below).
    assert!(
        logs.contains(LINK_BROKEN_STEP_DOWN_LABEL),
        "the walk must have abandoned the listing head on a broken link, or this case is not \
         exercising the no-step-down fence at all. Captured: {logs}"
    );

    // Assert — the tripwire stayed silent.
    assert!(
        !logs.contains(SUSPECT_LINEAGE_MARKER),
        "the suspect-lineage tripwire belongs to the `Compose` arm ALONE (design decision D6). \
         Raising it for a broken walk makes it unattributable, and an operator who learns it \
         fires on link breakage will not act on it when a lineage really is unrestorable. \
         Captured: {logs}"
    );
    assert!(
        !logs.contains(COMPOSE_STEP_DOWN_LABEL),
        "`record_compose_failure` must NOT be called on this arm — nothing was composed. Its \
         `{COMPOSE_STEP_DOWN_LABEL}` WARN here would mean the unit recorded a compose verdict it \
         never reached. Captured: {logs}"
    );

    // Assert — nothing moved.
    let after = fx.snapshot().await;
    assert_refusal_left_everything_intact(
        &before,
        &after,
        &fx.store.writes_since(mark),
        "stepped-down walk",
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. The compose arm is never retried against the same bytes
// ─────────────────────────────────────────────────────────────────────────────

/// **Case 6.** Composing case 4's exact artifact bytes classifies `Permanent`.
///
/// SR-4.5's refusal is "retry next tick", and for the store-failure arms that is
/// right. For the compose arm it must never mean "retry me against the same
/// bytes": composition performs no I/O, so a retry re-reads the identical
/// in-memory buffers and reaches the identical verdict. The classification
/// contract (`From<ComposeError> for CheckpointError` ⇒ `Permanent`) is what
/// encodes that, and this case asserts it on the bytes case 4 actually refused
/// — read back off the store rather than re-derived, so the two cases cannot
/// drift onto different fixtures.
///
/// Pure and Valkey-free apart from the shared fixture's layer: `compose_from_artifacts`
/// is synchronous and does no I/O.
#[tokio::test]
async fn compose_failure_on_the_refused_bytes_classifies_permanent_and_names_the_generation() {
    // Arrange — case 4's lineage, byte for byte.
    let fx = refusal_fixture().await;
    fx.plant_unreplayable_head_tail().await;
    let (base_bytes, tail_bytes) = fx.lineage_artifact_bytes().await;
    assert_eq!(
        tail_bytes.len(),
        MAX_CHAIN,
        "the fixture must hand composition the whole K-length chain, oldest first"
    );

    // Act.
    let err = compose_from_artifacts(fx.object_id, &base_bytes, BASE_SEQ, &tail_bytes)
        .expect_err("this lineage's head tail carries an unreplayable record; it must NOT compose");

    // Assert.
    let classified = CheckpointError::from(err);
    assert!(
        classified.is_permanent(),
        "every compose failure classifies `Permanent`: the engine does no I/O, so retrying the \
         same bytes reaches the same verdict. A `Transient` here would have the loop re-burn a \
         full lineage GET plus a compose on every tick forever; got {classified}"
    );
    assert!(
        !classified.is_transient(),
        "the two classifications are exclusive; a compose failure that also read as transient \
         would make the retry posture ambiguous at the call site"
    );
    assert!(
        classified
            .to_string()
            .contains(&HEAD_TAIL_TO_SEQ.to_string()),
        "the classified error must still name the offending generation ({HEAD_TAIL_TO_SEQ}) — \
         that attribution is what the tripwire's structured fields are derived from; got \
         {classified}"
    );
}
