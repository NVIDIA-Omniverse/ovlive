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

// Subtask `t3.dfh.be.84.pa` — SDD SR-9.2 for the **chain-collapse re-base**:
// concurrent re-bases are idempotent at `(object_id, S)`.
//
// ── The twin, and what is reused by reference ───────────────────────────────
//
// This file is the re-base twin of
// `checkpointer_tail_duplicate_leader_test.rs`. Its three structural arguments
// are REUSED, not re-derived — read them there, they are stated once:
//
//   1. **Two loops cannot express the window.** `spawn_checkpoint_loop`s
//      serialize on `le:checkpoint:leader` by design, and
//      `checkpointer_leader_race_test.rs` proves the OPPOSITE contract (a
//      SINGLETON recorded `creator_replica_id` set). So the unit is invoked
//      directly, twice, concurrently, with DISTINCT `replica_id`s — which is
//      also why `rebase_layer_lineage` is `pub` (its module doc names this
//      test).
//   2. **Observation protocol: RECORD the puts, never read the survivor.**
//      Both puts target one key, so any post-hoc read sees exactly one blob no
//      matter what the two leaders wrote, and `RebaseDecision` carries a handle
//      rather than bytes. The byte-equality claim is therefore made between the
//      two RECORDED payloads; the survivor is then checked to equal them.
//   3. **The window is made deterministic at the listing seam**, bounded and
//      never slept: a `Barrier(2)` rendezvous for identical-S, a one-shot HOLD
//      for divergent-S.
//
// ── Why duplicate leadership is the EXPECTED shape here, not a rare race ────
//
// The tail twin's window is a genuine race. This lane's is steady state. Under
// decision D5 the re-base's gather deadline is `(max_chain + 2) x
// rebase_fetch_reserve` — 270 s or more at stock knobs (K=8, 30 s reserve) —
// against a `LEADER_LOCK_TTL_MS` of 30 s, and `CheckpointLease::ensure_held`
// performs no mid-unit renewal: the lease is checked only BETWEEN pass-2 units
// (`src/checkpoint/checkpointer.rs`, `CheckpointLease`). So on a large layer the
// lock genuinely lapses mid-re-base and a peer acquires leadership while the
// first re-base is still running. SR-9.2's duplicate-leader re-base is
// consequently the expected steady-state shape for the target layer class, and
// the byte-idempotency this file pins is load-bearing in production rather than
// only under injected concurrency. That is the strongest available argument for
// this lane's existence.
//
// ── Three deliberate divergences from the twin ─────────────────────────────
//
// A. **One recording wrapper PER LEADER** over one shared inner adapter, rather
//    than one shared wrapper. Acceptance criterion 4 requires each gather's
//    `get_layer_checkpoint` set to be attributable to ITS OWN leader ("an
//    overlapping tail outside the walked chain was never consulted"), which a
//    single shared recording store cannot express. The gates live in a shared
//    [`RaceGates`] so "the first listing across both leaders" stays well defined.
//
// B. **The divergent-S hold gates BEFORE the inner listing read**, where the
//    twin gates after it. The twin's held leader diverges through a LATER
//    Valkey stream read, so holding its already-taken listing is exactly the
//    duplicate-leadership window. A re-base has no second read: the listing IS
//    its entire view of the lineage. A read-then-gate hold would therefore hand
//    both leaders the SAME listing, both would re-base at the same S, and the
//    fork fixture would prove nothing. Gating before the read is the same window
//    expressed against this unit's actual input.
//
// C. **`InMemoryCheckpointStore` as the inner adapter**, where the twin uses
//    LocalDisk. The re-base's base PUT always carries `meta.sidecar_bytes`
//    (`Some`, even when empty), and SR-9.2's byte claim covers the SIDECAR bytes
//    as well as the OMPF. InMemory round-trips `meta` verbatim including
//    `sidecar_bytes` (its `get_layer_checkpoint` doc says so); LocalDisk returns
//    `sidecar_bytes: None` on read-back because the sidecar is a separate file,
//    so the "surviving artifact equals both recordings" assertion could not be
//    made on it at byte granularity.
//
// ── No Valkey, deliberately ────────────────────────────────────────────────
//
// `rebase_layer_lineage(store, object_id, replica_id, cfg)` takes a
// `&dyn CheckpointStore` and nothing else — no `LayerStore`, no client, no
// `Config` (its module doc pins that signature as a cross-task contract). The
// whole fixture is therefore hand-planted generations plus pure composition, and
// no `TestValkey` appears in this file. Nothing is lost: a live layer would only
// add a keyspace neither the unit nor these assertions read.
//
// ── SR-9.2 byte scope (Task-1 PM review, 2026-09-02) — binding ─────────────
//
// Byte equality is asserted on the base OMPF blob and the ext-refs sidecar bytes
// ONLY. `CheckpointMeta::creator_replica_id` and `written_at_unix_ms` differ
// between two leaders BY DESIGN (per-replica identity, wall clock) and live
// deliberately outside the bytes — `rebase.rs`'s "Determinism boundary" header
// states this. Comparing whole metas would be wrong, not flaky.
//
// ── Assertion order: byte equality GATES, semantics only CLASSIFY ──────────
//
// Task 1's landed shape (`assembler.rs`, `first_base_bootstrap_test.rs` case 2)
// runs isomorphism first so a content divergence names itself. This lane
// inverts that on purpose, per its own acceptance criterion: the gating
// assertion is byte equality, and the semantic comparison exists ONLY inside the
// failure path, as [`classify_byte_divergence`], whose job is to say which of
// two very different defects fired:
//
//   - semantically EQUAL, bytes differ  ⇒ Task 1's emission determinism
//     regressed (a sort in `plan_full_content` / `sections_in_dfs_order`);
//   - semantically DIFFERENT             ⇒ a real re-base defect (the two
//     leaders composed different lineages).
//
// The classifier is built on Task 2's consolidated SUPERSET faithful-column
// basis (`helpers::assert_faithful_columns`) — CONSUMED, never forked.
//
// ── Bounded gates, named ceilings (acceptance criterion 5) ─────────────────
//
// | ceiling            | value | what a timeout there MEANS                   |
// |--------------------|-------|----------------------------------------------|
// | `GATE_WAIT`        | 10 s  | a listing gate was never released — the other |
// |                    |       | leader never reached its rendezvous, or the   |
// |                    |       | driver never fired the release                |
// | `RACE_DEADLINE`    | 30 s  | the whole two-leader join (plus the driver)   |
// |                    |       | did not finish; the outer bound on every test |
// | gather budget      | 25 s  | `(MAX_CHAIN + 2) x REBASE_FETCH_RESERVE`,      |
// |                    |       | derived by the unit itself from the shrunk    |
// |                    |       | fixture knobs below                           |
// | per-ext-blob PUT   |  5 s  | `REBASE_FETCH_RESERVE`, applied by the unit    |
//
// Every ceiling above is strictly INSIDE `RACE_DEADLINE`, so the documented
// per-test worst case is **30 s** — under nextest's 120 s hard kill and under
// its 40 s slow mark. In practice each test runs in milliseconds: every artifact
// is in-memory and no test sleeps or asserts on elapsed time.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use tokio::sync::{oneshot, Barrier, Notify};

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    rebase_layer_lineage, CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointKind,
    CheckpointMeta, CheckpointStore, LayerArtifact, RebaseDecision, TailCaptureConfig,
};
use live_edit::domain::delta::apply_delta;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::ObjectState;
use live_edit::storage::diff_read::ObjectSnapshot;
use live_edit::storage::keys::sha256_to_hex;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand, TimeSample};

use crate::checkpoint::helpers::{
    assert_faithful_columns, put_base_generation, put_tail_generation_from_records,
    FaithfulColumnsOptions, HAND_PUT_CREATED_BY, HAND_PUT_SERVER_TS, HAND_PUT_WRITTEN_AT_UNIX_MS,
};
use crate::common::owned_delta::delta_to_ompf_bytes;

// ── Fixture constants ───────────────────────────────────────────────────────

/// The object every fixture plants under. Arbitrary: no live layer exists (see
/// the header's "No Valkey" note), so nothing resolves this id.
const OBJECT_ID: u64 = 4242;

/// The lineage root — a base at this sequence, exactly the shape SR-7.1's
/// creation bootstrap produces.
const LINEAGE_ROOT_SEQ: u64 = 1;

/// **K**, shrunk from the stock 8. The chain planted below is 4 tails long, so
/// `true_chain_len >= K` holds and the unit takes its re-base arm rather than
/// [`RebaseDecision::UnderBound`].
const MAX_CHAIN: usize = 3;

/// The per-artifact reserve, shrunk from the stock 30 s. It is the unit's
/// per-fetch cap AND its per-ext-blob PUT cap, and the whole-gather deadline is
/// `(MAX_CHAIN + 2) x` this — 25 s, comfortably inside [`RACE_DEADLINE`].
const REBASE_FETCH_RESERVE: Duration = Duration::from_secs(5);

/// **S** for the un-forked lineage: the head tail's `to_seq`, and therefore the
/// address of the base a re-base writes.
const HEAD_SEQ: u64 = 9;

/// **S'** for the divergent-S fork: the head of the extended lineage the driver
/// lands while the second leader is held.
const FORK_HEAD_SEQ: u64 = 13;

/// Tails of the walked chain, as `(from_seq, to_seq]`. Deliberately NOT dense in
/// sequence — the gaps are what leave room for the out-of-chain siblings below.
const CHAIN_SPANS: [(u64, u64); 4] = [(1, 3), (3, 5), (5, 7), (7, 9)];

/// Tails that sit in the listing but are NOT members of any chain walked back
/// from the head: partial windows a rival leader captured. Their `to_seq`s
/// (4, 6) are never named by a `from_seq` link, so the walk addresses straight
/// past them. Acceptance criterion 4's subject.
const OUT_OF_CHAIN_SPANS: [(u64, u64); 2] = [(3, 4), (5, 6)];

/// The tails the divergent-S driver lands while the held leader is parked.
const FORK_SPANS: [(u64, u64); 2] = [(9, 11), (11, 13)];

/// Ceilings. See the header's table for what a timeout at each one means.
const GATE_WAIT: Duration = Duration::from_secs(10);
const RACE_DEADLINE: Duration = Duration::from_secs(30);

const SESSION_ID: &str = "rebase-duplicate-leader-session";
const LAYER_NAME: &str = "rebase-duplicate-leader-layer";
const LEADER_A: &str = "rebase-duplicate-leader-a";
const LEADER_B: &str = "rebase-duplicate-leader-b";

/// Ext-blob payloads. Distinct lengths so a part mix-up cannot survive a
/// length-blind comparison.
fn blob_a() -> Vec<u8> {
    b"rebase-ext-blob-A".to_vec()
}
fn blob_b() -> Vec<u8> {
    b"rebase-ext-blob-BB".to_vec()
}
fn blob_c() -> Vec<u8> {
    b"rebase-ext-blob-CCC".to_vec()
}
fn blob_d() -> Vec<u8> {
    b"rebase-ext-blob-DDDD".to_vec()
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// `now` in unix milliseconds — the window `written_at_unix_ms` is checked
/// against.
fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock must be after the unix epoch")
        .as_millis() as u64
}

/// The knob set every leader runs on. Shrunk from `Default` at exactly two
/// fields — see [`MAX_CHAIN`] and [`REBASE_FETCH_RESERVE`].
fn rebase_config() -> TailCaptureConfig {
    TailCaptureConfig {
        max_chain: MAX_CHAIN,
        rebase_fetch_reserve: REBASE_FETCH_RESERVE,
        ..TailCaptureConfig::default()
    }
}

// ── Recorded observations ───────────────────────────────────────────────────

/// One `put_layer_checkpoint` as one leader's wrapper saw it, payload included.
///
/// Recording these — rather than reading the survivor back — is what makes the
/// idempotency claim observable at all (header, argument 2).
#[derive(Debug, Clone)]
struct PutRecord {
    object_id: u64,
    snap_seq: u64,
    kind: CheckpointKind,
    /// The exact OMPF bytes handed to the adapter.
    ompf: Bytes,
    /// The whole meta. Only the lineage-carrying fields plus the sidecar bytes
    /// are compared BETWEEN leaders; `creator_replica_id` and
    /// `written_at_unix_ms` are expected to differ and are the point.
    meta: CheckpointMeta,
}

impl PutRecord {
    /// The SR-4.3 sidecar bytes, which the re-base always attaches (`Some`, even
    /// for an empty hash list — the b.s5y publish-order obligation).
    fn sidecar(&self) -> &Bytes {
        self.meta.sidecar_bytes.as_ref().unwrap_or_else(|| {
            panic!(
                "every re-base base PUT must carry `sidecar_bytes: Some(..)` — `rebase.rs` spells \
                 it ALWAYS Some, even when the hash list is empty. Record: {self:?}"
            )
        })
    }
}

/// The gates the two leaders share. Held in an `Arc` by both per-leader
/// wrappers so "the FIRST listing across both leaders" is well defined even
/// though the recordings are per-leader.
struct RaceGates {
    /// Identical-S: both listing calls park here before returning.
    rendezvous: Mutex<Option<Arc<Barrier>>>,
    /// Divergent-S: the FIRST listing call takes this and parks on it BEFORE
    /// reading the inner listing (header, divergence B).
    hold: Mutex<Option<oneshot::Receiver<()>>>,
    /// Fired after every successful base PUT. The divergent-S driver waits on
    /// this instead of polling, which is what keeps this file sleep-free.
    /// `Notify::notify_one` stores a permit when no waiter is parked, so the
    /// driver cannot miss a put that landed before it awaited.
    base_put_landed: Notify,
}

impl RaceGates {
    fn new() -> Self {
        Self {
            rendezvous: Mutex::new(None),
            hold: Mutex::new(None),
            base_put_landed: Notify::new(),
        }
    }

    /// Hold every listing return until `parties` of them have arrived.
    fn arm_rendezvous(&self, parties: usize) {
        *self.rendezvous.lock().unwrap_or_else(|p| p.into_inner()) =
            Some(Arc::new(Barrier::new(parties)));
    }

    /// Park the first listing call — before it reads — until the returned
    /// sender fires.
    fn arm_hold(&self) -> oneshot::Sender<()> {
        let (tx, rx) = oneshot::channel();
        *self.hold.lock().unwrap_or_else(|p| p.into_inner()) = Some(rx);
        tx
    }

    /// Clear both gates once a race has finished, so the test's own post-hoc
    /// listings do not park on a rendezvous that will never see a second party
    /// (which would cost a full [`GATE_WAIT`] per read).
    fn disarm(&self) {
        *self.rendezvous.lock().unwrap_or_else(|p| p.into_inner()) = None;
        *self.hold.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }
}

/// One leader's view of the store: every call forwards to the shared inner
/// adapter, and the three calls this lane reasons about are recorded.
///
/// Every trait method is spelled out — including the four with non-trivial
/// defaults (`read_ext_refs_sidecar`, `list_layer_artifacts`,
/// `delete_layer_temp_artifacts`, `publish_worst_case`). Omitting one of those
/// does not leave a call un-instrumented; it SUBSTITUTES behaviour behind the
/// wrapper and makes every assertion about it vacuously green, which is the trap
/// `helpers::DelegatingCheckpointStore`'s doc records. This type does not hold a
/// `DelegatingCheckpointStore` because it overrides three of the methods that
/// skeleton would forward anyway, and a two-layer forward for the rest buys
/// nothing here.
struct LeaderRecordingStore {
    inner: Arc<InMemoryCheckpointStore>,
    gates: Arc<RaceGates>,
    label: &'static str,
    puts: Mutex<Vec<PutRecord>>,
    /// `(seq, kind)` per `get_layer_checkpoint`, in call order. Acceptance
    /// criterion 4 reads this: the gather must consult ONLY its own walked
    /// chain's members.
    fetches: Mutex<Vec<(u64, CheckpointKind)>>,
    /// `(hash_hex, bytes)` per `put_ext_blob`, in call order.
    ext_puts: Mutex<Vec<(String, Bytes)>>,
}

impl LeaderRecordingStore {
    fn new(
        inner: Arc<InMemoryCheckpointStore>,
        gates: Arc<RaceGates>,
        label: &'static str,
    ) -> Self {
        Self {
            inner,
            gates,
            label,
            puts: Mutex::new(Vec::new()),
            fetches: Mutex::new(Vec::new()),
            ext_puts: Mutex::new(Vec::new()),
        }
    }

    fn puts(&self) -> Vec<PutRecord> {
        self.puts.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// The single base PUT this leader issued, with a message naming the count
    /// otherwise — zero means the leader refused, more than one means the unit
    /// grew a retry loop it must not have (`rebase.rs`: "There is no retry loop;
    /// the tick is the retry").
    fn sole_put(&self) -> PutRecord {
        let puts = self.puts();
        assert_eq!(
            puts.len(),
            1,
            "leader {} must have issued EXACTLY one `put_layer_checkpoint`. Zero means it refused \
             (the decision assertion above would already have named the arm); more than one means \
             the unit acquired a retry loop, which `rebase.rs` forbids. Puts: {puts:?}",
            self.label
        );
        puts.into_iter().next().expect("length checked above")
    }

    fn fetches(&self) -> Vec<(u64, CheckpointKind)> {
        self.fetches
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    fn ext_puts(&self) -> Vec<(String, Bytes)> {
        self.ext_puts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }
}

#[async_trait]
impl CheckpointStore for LeaderRecordingStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        let kind = meta.kind;
        self.puts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(PutRecord {
                object_id,
                snap_seq,
                kind,
                ompf: ompf.clone(),
                meta: meta.clone(),
            });
        let result = self
            .inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await;
        // Signal AFTER the adapter acknowledged: the divergent-S driver's whole
        // precondition is that the first leader's base is DURABLE before it
        // extends the lineage, and a pre-await signal would only prove the call
        // was issued.
        if result.is_ok() && kind == CheckpointKind::Base {
            self.gates.base_put_landed.notify_one();
        }
        result
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        // GATE FIRST, read second — the divergent-S shape. See the header,
        // divergence B: a re-base's listing IS its whole view of the lineage,
        // so a held leader must take its listing AFTER the fork, not before.
        let hold = self
            .gates
            .hold
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        if let Some(rx) = hold {
            let _ = tokio::time::timeout(GATE_WAIT, rx).await;
        }

        let handles = self.inner.list_layer_checkpoints(object_id).await?;

        // Read-then-gate for the rendezvous: both leaders must have taken a
        // listing before either walks, and neither may see the other's base.
        let rendezvous = self
            .gates
            .rendezvous
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        if let Some(barrier) = rendezvous {
            let _ = tokio::time::timeout(GATE_WAIT, barrier.wait()).await;
        }

        Ok(handles)
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_latest_layer_checkpoint(object_id).await
    }

    // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST — its trait
    // default returns `NotImplemented`, which would report "no such generation"
    // for a generation the delegate holds, to the reader whose whole job is
    // deciding what exists.
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.fetches
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((handle.seq(), handle.kind));
        self.inner.get_layer_checkpoint(handle).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.inner.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.inner.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.inner.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.ext_puts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push((hash_hex.to_string(), bytes.clone()));
        self.inner.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.inner.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.inner.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_counters_sidecar(body).await
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_counters_sidecar().await
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.inner.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(
        &self,
    ) -> Result<std::collections::HashSet<String>, CheckpointError> {
        self.inner.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.inner.list_live_ext_set().await
    }

    // The four below carry non-trivial trait defaults — see the type's doc.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<std::collections::HashSet<String>, CheckpointError> {
        self.inner.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.inner.list_layer_artifacts().await
    }

    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.inner
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> Duration {
        self.inner.publish_worst_case()
    }
}

// ── The commit timeline ─────────────────────────────────────────────────────

/// One planted commit: the OMPF bytes the writer sent at `sequence_num`.
///
/// These are the bytes the capture path persists verbatim
/// (`le:{oid}:obj:stream`'s `ompf` field is the writer's raw inbound OMPF), so a
/// tail built from them is byte-for-byte what production would hold.
type Commit = (u64, Bytes);

fn inline_field(section_id: u64, key: &str, seq: u64, set_order: u64) -> SetField {
    SetField {
        section_id,
        key_name: key.to_string(),
        value_or_ext_hash: format!("payload-seq-{seq}-{key}").into_bytes(),
        ext_value_size: 0,
        ext_value_index: 0,
        set_order,
    }
}

fn ext_field(
    section_id: u64,
    key: &str,
    blob: &[u8],
    ext_value_index: u32,
    set_order: u64,
) -> SetField {
    SetField {
        section_id,
        key_name: key.to_string(),
        value_or_ext_hash: sha256(blob).to_vec(),
        ext_value_size: blob.len() as u64,
        ext_value_index,
        set_order,
    }
}

fn inline_sample(section_id: u64, time: f64, seq: u64) -> TimeSample {
    TimeSample {
        section_id,
        time,
        value_or_ext_hash: format!("ts-seq-{seq}").into_bytes(),
        ext_value_size: 0,
        ext_value_index: 0,
    }
}

fn ext_sample(section_id: u64, time: f64, blob: &[u8], ext_value_index: u32) -> TimeSample {
    TimeSample {
        section_id,
        time,
        value_or_ext_hash: sha256(blob).to_vec(),
        ext_value_size: blob.len() as u64,
        ext_value_index,
    }
}

fn create_section(parent_id: u64, section_id: u64, name: &str, section_type: u8) -> StructCommand {
    StructCommand::CreateSection(CommandCreateSection {
        parent_id,
        section_id,
        section_name: name.to_string(),
        section_type,
    })
}

/// The commits the walked chain replays, `commits[k]` landing at
/// `LINEAGE_ROOT_SEQ + 1 + k` (sequences 2..=[`HEAD_SEQ`]).
///
/// **Sizing is deliberate, not incidental.** Task 1's mutation lesson: minimums
/// of "two" make mutation detection probabilistic. The composed state below
/// carries **nine fields** (four of them external), **four time samples** (one
/// external) and four sections, so neutralizing `plan_full_content`'s field sort
/// reorders two independently-built `HashMap`s with probability `1 - 1/9!`
/// rather than a coin flip.
///
/// Four distinct ext blobs appear across three records — `blob_b` twice, in two
/// different records, which is the lineage-wide dedup the SR-4.3 backfill count
/// must reflect (four blobs, not five).
fn chain_commits() -> Vec<Commit> {
    let mut out: Vec<Commit> = Vec::new();
    let mut push = |seq: u64, delta: OwnedDelta, ext: &[&[u8]]| {
        out.push((seq, Bytes::from(delta_to_ompf_bytes(&delta, ext))));
    };

    // seq 2 — the structural root plus the first inline field.
    push(
        2,
        OwnedDelta {
            struct_commands: vec![
                create_section(0, 10, "root_10", 1),
                create_section(10, 11, "child_11", 2),
            ],
            set_fields: vec![inline_field(10, "k_alpha", 2, 0)],
            ..OwnedDelta::default()
        },
        &[],
    );
    // seq 3 — a second child, an inline field and a time sample.
    push(
        3,
        OwnedDelta {
            struct_commands: vec![create_section(10, 12, "child_12", 2)],
            set_fields: vec![inline_field(11, "k_bravo", 3, 1)],
            time_samples: vec![inline_sample(11, 0.5, 3)],
            ..OwnedDelta::default()
        },
        &[],
    );
    // seq 4 — the first external reference (blob A at part 1).
    push(
        4,
        OwnedDelta {
            set_fields: vec![
                ext_field(12, "ext_charlie", &blob_a(), 1, 2),
                inline_field(10, "k_delta", 4, 3),
            ],
            ..OwnedDelta::default()
        },
        &[&blob_a()],
    );
    // seq 5 — a second root section, an inline field and a time sample.
    push(
        5,
        OwnedDelta {
            struct_commands: vec![create_section(0, 13, "root_13", 1)],
            set_fields: vec![inline_field(13, "k_echo", 5, 4)],
            time_samples: vec![inline_sample(13, 1.5, 5)],
            ..OwnedDelta::default()
        },
        &[],
    );
    // seq 6 — an external field AND an external time sample (blobs B, C).
    push(
        6,
        OwnedDelta {
            set_fields: vec![ext_field(13, "ext_foxtrot", &blob_b(), 1, 5)],
            time_samples: vec![ext_sample(11, 2.5, &blob_c(), 2)],
            ..OwnedDelta::default()
        },
        &[&blob_b(), &blob_c()],
    );
    // seq 7 — a new field plus an overwrite of a seq-2 field, so the surviving
    // entry's version is tail-era rather than root-era.
    push(
        7,
        OwnedDelta {
            set_fields: vec![
                inline_field(11, "k_golf", 7, 6),
                inline_field(10, "k_alpha", 7, 7),
            ],
            ..OwnedDelta::default()
        },
        &[],
    );
    // seq 8 — blob B AGAIN (lineage-wide dedup) alongside a new blob D.
    push(
        8,
        OwnedDelta {
            set_fields: vec![
                ext_field(12, "ext_hotel", &blob_b(), 1, 8),
                ext_field(13, "ext_india", &blob_d(), 2, 9),
            ],
            ..OwnedDelta::default()
        },
        &[&blob_b(), &blob_d()],
    );
    // seq 9 — S. A trailing inline field and time sample.
    push(
        9,
        OwnedDelta {
            set_fields: vec![inline_field(13, "k_juliett", 9, 10)],
            time_samples: vec![inline_sample(10, 3.5, 9)],
            ..OwnedDelta::default()
        },
        &[],
    );

    out
}

/// The commits the divergent-S driver lands while the held leader is parked,
/// landing at sequences 10..=[`FORK_HEAD_SEQ`].
fn fork_commits() -> Vec<Commit> {
    let mut out: Vec<Commit> = Vec::new();
    let mut push = |seq: u64, delta: OwnedDelta| {
        out.push((seq, Bytes::from(delta_to_ompf_bytes(&delta, &[]))));
    };
    push(
        10,
        OwnedDelta {
            set_fields: vec![inline_field(10, "k_kilo", 10, 11)],
            ..OwnedDelta::default()
        },
    );
    push(
        11,
        OwnedDelta {
            set_fields: vec![inline_field(11, "k_lima", 11, 12)],
            ..OwnedDelta::default()
        },
    );
    push(
        12,
        OwnedDelta {
            set_fields: vec![inline_field(12, "k_mike", 12, 13)],
            ..OwnedDelta::default()
        },
    );
    push(
        13,
        OwnedDelta {
            set_fields: vec![inline_field(13, "k_november", 13, 14)],
            time_samples: vec![inline_sample(12, 4.5, 13)],
            ..OwnedDelta::default()
        },
    );
    out
}

/// Every distinct ext blob the walked chain's records carry, in
/// first-appearance order — which is exactly the order
/// `ComposedLineage::ext_values` presents them in, and therefore the order the
/// SR-4.3 backfill PUTs them in.
fn expected_ext_backfill() -> Vec<(String, Bytes)> {
    [blob_a(), blob_b(), blob_c(), blob_d()]
        .into_iter()
        .map(|blob| (sha256_to_hex(&sha256(&blob)), Bytes::from(blob)))
        .collect()
}

/// The commits inside `(from_seq, to_seq]`, taken from `commits`.
fn slice_of(commits: &[Commit], from_seq: u64, to_seq: u64) -> Vec<Commit> {
    commits
        .iter()
        .filter(|(seq, _)| *seq > from_seq && *seq <= to_seq)
        .cloned()
        .collect()
}

// ── Fixture ─────────────────────────────────────────────────────────────────

/// A planted lineage plus the two per-leader recording wrappers over it.
struct RebaseFixture {
    inner: Arc<InMemoryCheckpointStore>,
    gates: Arc<RaceGates>,
    a: Arc<LeaderRecordingStore>,
    b: Arc<LeaderRecordingStore>,
    /// The envelope bytes of every out-of-chain tail at plant time, keyed by
    /// `to_seq`. Acceptance criterion 4 compares against these to prove the
    /// siblings were left INTACT, not merely still listed.
    out_of_chain_bytes: BTreeMap<u64, Bytes>,
}

impl RebaseFixture {
    /// Plant `base@1 -> 4 chain tails -> S = 9`, plus the two out-of-chain
    /// sibling tails.
    ///
    /// Everything is planted through the INNER store: a hand-put fixture
    /// generation is not one of the leaders' puts and must not appear in either
    /// recording.
    async fn setup() -> Self {
        let inner = Arc::new(InMemoryCheckpointStore::new());
        let inner_dyn: Arc<dyn CheckpointStore> = Arc::clone(&inner) as Arc<dyn CheckpointStore>;

        put_base_generation(
            &inner_dyn,
            OBJECT_ID,
            SESSION_ID,
            LAYER_NAME,
            LINEAGE_ROOT_SEQ,
        )
        .await;

        let commits = chain_commits();
        for (from_seq, to_seq) in CHAIN_SPANS {
            put_tail_generation_from_records(
                &inner_dyn,
                OBJECT_ID,
                SESSION_ID,
                LAYER_NAME,
                LINEAGE_ROOT_SEQ,
                from_seq,
                &slice_of(&commits, from_seq, to_seq),
                HAND_PUT_WRITTEN_AT_UNIX_MS,
                0,
            )
            .await;
        }

        // The siblings. They carry the SAME real commit bytes a rival leader
        // capturing that partial window would have captured — faithful, and it
        // makes "never consulted" a claim about the WALK rather than about
        // content the walk could have rejected.
        let mut out_of_chain_bytes = BTreeMap::new();
        for (from_seq, to_seq) in OUT_OF_CHAIN_SPANS {
            let handle = put_tail_generation_from_records(
                &inner_dyn,
                OBJECT_ID,
                SESSION_ID,
                LAYER_NAME,
                LINEAGE_ROOT_SEQ,
                from_seq,
                &slice_of(&commits, from_seq, to_seq),
                HAND_PUT_WRITTEN_AT_UNIX_MS,
                0,
            )
            .await;
            let blob = inner
                .get_layer_checkpoint(&handle)
                .await
                .expect("get_layer_checkpoint must succeed")
                .expect("the sibling tail must be live immediately after planting");
            out_of_chain_bytes.insert(to_seq, blob.bytes);
        }

        let gates = Arc::new(RaceGates::new());
        let a = Arc::new(LeaderRecordingStore::new(
            Arc::clone(&inner),
            Arc::clone(&gates),
            LEADER_A,
        ));
        let b = Arc::new(LeaderRecordingStore::new(
            Arc::clone(&inner),
            Arc::clone(&gates),
            LEADER_B,
        ));

        Self {
            inner,
            gates,
            a,
            b,
            out_of_chain_bytes,
        }
    }

    /// Land the fork tails through the INNER store, extending the lineage from
    /// [`HEAD_SEQ`] to [`FORK_HEAD_SEQ`].
    async fn land_fork_tails(&self) {
        let inner_dyn: Arc<dyn CheckpointStore> =
            Arc::clone(&self.inner) as Arc<dyn CheckpointStore>;
        let commits = fork_commits();
        for (from_seq, to_seq) in FORK_SPANS {
            put_tail_generation_from_records(
                &inner_dyn,
                OBJECT_ID,
                SESSION_ID,
                LAYER_NAME,
                LINEAGE_ROOT_SEQ,
                from_seq,
                &slice_of(&commits, from_seq, to_seq),
                HAND_PUT_WRITTEN_AT_UNIX_MS,
                0,
            )
            .await;
        }
    }

    /// Invoke the re-base unit as `replica_id`, through that leader's wrapper.
    async fn rebase(&self, store: &Arc<LeaderRecordingStore>, replica_id: &str) -> RebaseDecision {
        rebase_layer_lineage(store.as_ref(), OBJECT_ID, replica_id, &rebase_config())
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "rebase_layer_lineage({replica_id}) must not error — the unit's only `Err` is \
                     `NotFound`, and nothing in this fixture removes the layer: {e}"
                )
            })
    }

    /// The listing as the fixture (not a leader) sees it. Reads through the
    /// inner store so no gate applies and nothing is recorded.
    async fn listing(&self) -> Vec<CheckpointHandle> {
        self.inner
            .list_layer_checkpoints(OBJECT_ID)
            .await
            .expect("list_layer_checkpoints must succeed")
    }
}

/// Unwrap a decision to its `Rebased` facts, with a message naming the other arm
/// otherwise — neither `UnderBound` nor `Refused` is reachable from a healthy
/// fixture here, so both are fixture defects rather than the property under test.
fn expect_rebased(who: &str, decision: &RebaseDecision) -> RebasedFacts {
    match decision {
        RebaseDecision::Rebased {
            handle,
            base_snap_seq,
            from_chain_len,
            to_seq,
            ext_backfilled,
        } => RebasedFacts {
            handle: handle.clone(),
            base_snap_seq: *base_snap_seq,
            from_chain_len: *from_chain_len,
            to_seq: *to_seq,
            ext_backfilled: *ext_backfilled,
        },
        RebaseDecision::UnderBound { true_chain_len } => panic!(
            "leader {who} answered UnderBound(true_chain_len = {true_chain_len}) instead of \
             re-basing. The planted chain is {} tails against MAX_CHAIN = {MAX_CHAIN}, so this \
             means a link in CHAIN_SPANS no longer resolves and the walk stopped short",
            CHAIN_SPANS.len()
        ),
        RebaseDecision::Refused(refusal) => panic!(
            "leader {who} refused ({refusal:?}, outcome {:?}) instead of re-basing. Every refusal \
             reachable from this fixture is a fixture defect: a broken link, a non-contiguous \
             record run, or an ext part whose index does not resolve",
            decision.outcome_label()
        ),
    }
}

/// The `Rebased` payload, flattened so both tests read it the same way.
#[derive(Debug, Clone)]
struct RebasedFacts {
    handle: CheckpointHandle,
    base_snap_seq: u64,
    from_chain_len: usize,
    to_seq: u64,
    ext_backfilled: usize,
}

// ── The diagnostic classifier (acceptance criterion 2) ──────────────────────

/// Decode a base OMPF into the `ObjectState` its reset delta reconstructs.
///
/// The reset is applied to a fresh state, which is what a DR restore does — so
/// this is the semantic content of the generation, not an incidental encoding of
/// it. Same shape as `first_base_bootstrap_test.rs`'s file-local
/// `decode_reset_ompf`; re-homed here rather than shared because that one is
/// file-local by design and this lane needs an `ObjectSnapshot`, not an
/// `ObjectState`, to reach Task 2's consolidated comparator.
fn decode_reset_ompf(ompf_bytes: &Bytes) -> ObjectState {
    let parsed = OmpfContainer::parse(ompf_bytes).expect("OMPF parse must succeed");
    let delta = oodf_ffi::parse_delta(parsed.delta_bytes)
        .expect("OODF delta parse must succeed — a re-base must never emit malformed bytes");
    let mut state = ObjectState::default();
    apply_delta(&mut state, Some(delta)).expect("apply_delta on the reset must succeed");
    state
}

/// Re-shape an [`ObjectState`] as the [`ObjectSnapshot`] Task 2's comparator
/// takes. Pure projection — `tombstones` is empty because a reset delta carries
/// no tombstone list by construction (`build_reset_ompf_from_state`), and the
/// comparator does not read it.
fn snapshot_of(state: &ObjectState) -> ObjectSnapshot {
    ObjectSnapshot {
        object_id: state.object_id,
        sequence_num: state.sequence_num,
        first_version: state.first_version,
        source_format: state.source_format.clone(),
        next_set_order: state.next_set_order,
        sections: state.sections.clone(),
        children: state.children.clone(),
        fields: state.fields.values().cloned().collect(),
        time_samples: state.time_samples.values().cloned().collect(),
        tombstones: Vec::new(),
    }
}

/// Called ONLY when the two recorded base payloads differ byte-for-byte.
/// Classifies the failure and panics either way; it never returns.
///
/// This is the whole reason the semantic comparison exists in this file. The two
/// diagnoses need completely different responses and a bare `assert_eq!` on
/// bytes cannot tell them apart:
///
/// - **semantically EQUAL, bytes differ** — the two leaders composed the same
///   state and emitted it differently, i.e. Task 1's byte-determinism
///   (`plan_full_content`'s field / time-sample sorts, `sections_in_dfs_order`)
///   regressed. SR-9.2's blind last-writer-wins PUT is then unsound and the fix
///   is in the emission order, not here.
/// - **semantically DIFFERENT** — the two leaders composed different lineages,
///   which is a real re-base defect (a gather that saw different members, a
///   replay that applied records out of order).
///
/// The comparison basis is Task 2's consolidated SUPERSET faithful-column
/// helper, consumed as-is. `catch_unwind` is what turns its assertion into a
/// verdict; the inner panic's own message rides the classification so the
/// diverging column is named.
fn classify_byte_divergence(a: &PutRecord, b: &PutRecord, context: &str) -> ! {
    let a_state = decode_reset_ompf(&a.ompf);
    let b_state = decode_reset_ompf(&b.ompf);
    let a_snap = snapshot_of(&a_state);
    let b_snap = snapshot_of(&b_state);

    let semantic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_faithful_columns(
            &a_snap,
            &b_snap,
            FaithfulColumnsOptions::default(),
            "the two duplicate-leader re-base payloads",
        );
    }));

    match semantic {
        Ok(()) => panic!(
            "{context}: SEMANTICALLY EQUAL, BYTES DIFFER ({} vs {} bytes) => TASK 1's EMISSION \
             DETERMINISM REGRESSED. The two leaders composed the same state and serialized it \
             differently, so `DiffEngine::plan_full_content`'s field / time-sample sort or \
             `sections_in_dfs_order` no longer imposes a total order over the state's CONTENT. \
             SR-9.2's blind last-writer-wins base PUT is UNSOUND until that is restored; the fix \
             is in the emission order, not in the re-base. (FlatBuffers shares vtables and widens \
             offsets, so a reorder usually shows up as a LENGTH change first.)",
            a.ompf.len(),
            b.ompf.len()
        ),
        Err(payload) => {
            let detail = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            panic!(
                "{context}: SEMANTICALLY DIFFERENT => A REAL RE-BASE DEFECT. The two leaders did \
                 not compose the same state, so this is NOT an emission-order regression: look at \
                 the gather (did the two walks resolve different members?) and at the replay \
                 order, not at `plan_full_content`. Faithful-column divergence: {detail}"
            )
        }
    }
}

/// The gating byte assertions for one duplicate-leader pair: OMPF first, then
/// the ext-refs sidecar. Neither is ever weakened to the semantic comparison —
/// that one only runs inside [`classify_byte_divergence`].
fn assert_payloads_byte_identical(a: &PutRecord, b: &PutRecord) {
    if a.ompf != b.ompf {
        classify_byte_divergence(a, b, "SR-9.2 base OMPF byte equality");
    }
    let (a_side, b_side) = (a.sidecar(), b.sidecar());
    assert_eq!(
        a_side,
        b_side,
        "SR-9.2 ext-refs SIDECAR byte equality: the two leaders' base OMPFs are byte-identical but \
         their sidecars are not, which can only mean `ext_refs_sidecar_json`'s input order moved \
         independently of the delta's. The sidecar is derived from the SAME sorted field / \
         time-sample vectors (`collect_external_value_hashes` runs over \
         `plan_full_content`'s output), so this is an ordering defect in that hand-off. \
         A: {:?}\nB: {:?}",
        String::from_utf8_lossy(a_side),
        String::from_utf8_lossy(b_side)
    );
}

/// A sortable, readable spelling of a generation kind. `CheckpointKind` is not
/// `Ord`, and an addressing assertion wants a stable order plus a message that
/// names the kind rather than debug-printing it.
fn kind_label(kind: CheckpointKind) -> &'static str {
    match kind {
        CheckpointKind::Base => "base",
        CheckpointKind::Tail => "tail",
    }
}

/// Assert one leader's recorded fetch set is EXACTLY the members of the chain it
/// walked — the acceptance criterion 4 claim that most needs evidence.
///
/// The expected set is spelled as `(base@root) + one tail per link`, and the
/// comparison is over a SET rather than a sequence because the walk's direction
/// (head-down) is `lineage.rs`'s business, not this lane's.
fn assert_fetched_exactly_the_chain(
    label: &str,
    fetched: &[(u64, CheckpointKind)],
    chain_to_seqs: &[u64],
) {
    // Ordered by `(seq, kind_label)` rather than by `CheckpointKind`, which is
    // deliberately not `Ord` — the label also makes the failure message read as
    // the addresses it is about.
    let observed: BTreeSet<(u64, &'static str)> = fetched
        .iter()
        .map(|(seq, kind)| (*seq, kind_label(*kind)))
        .collect();
    let mut expected: BTreeSet<(u64, &'static str)> = chain_to_seqs
        .iter()
        .map(|seq| (*seq, kind_label(CheckpointKind::Tail)))
        .collect();
    expected.insert((LINEAGE_ROOT_SEQ, kind_label(CheckpointKind::Base)));

    assert_eq!(
        observed,
        expected,
        "leader {label} must have fetched EXACTLY its own walked chain: the lineage root plus one \
         tail per link. An extra entry means the gather consulted a generation no `from_seq` link \
         named — an overlapping tail outside the walk (to_seqs {:?}), or a base another leader \
         published — which is the SR-9.2 claim this assertion exists for. A missing entry means \
         the walk stopped short. Fetch order was: {fetched:?}",
        OUT_OF_CHAIN_SPANS.map(|(_, to)| to)
    );
    assert_eq!(
        fetched.len(),
        expected.len(),
        "leader {label} must fetch each member exactly ONCE — a repeat means the walk re-fetched a \
         generation it already held. Fetch order: {fetched:?}"
    );
}

/// Assert every out-of-chain sibling is still listed AND still carries the bytes
/// it was planted with.
///
/// "Intact", not merely "present": this Task writes nothing that assumes their
/// absence, and Task 4's retention is what eventually prunes them as orphans.
async fn assert_out_of_chain_tails_intact(fixture: &RebaseFixture, listing: &[CheckpointHandle]) {
    for (to_seq, planted) in &fixture.out_of_chain_bytes {
        let handle = listing
            .iter()
            .find(|h| h.kind == CheckpointKind::Tail && h.seq() == *to_seq)
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "the out-of-chain tail at to_seq {to_seq} must still be listed — a re-base \
                     deletes NOTHING (lineage-aware retention owns every deletion). \
                     Listing: {listing:?}"
                )
            });
        let blob = fixture
            .inner
            .get_layer_checkpoint(&handle)
            .await
            .expect("get_layer_checkpoint must succeed")
            .expect("the out-of-chain handle must address a live artifact");
        assert_eq!(
            blob.bytes, *planted,
            "the out-of-chain tail at to_seq {to_seq} must be byte-unchanged: neither leader may \
             read, rewrite or truncate a generation outside its own walked chain"
        );
    }
}

/// The lineage-carrying meta columns, i.e. every column EXCEPT the two the
/// determinism boundary deliberately keeps outside the bytes
/// (`creator_replica_id`, `written_at_unix_ms`) and the sidecar, which is
/// compared as bytes by [`assert_payloads_byte_identical`].
///
/// A named struct rather than a tuple so a new `CheckpointMeta` column surfaces
/// here as a decision — "is this lineage-carrying?" — rather than silently
/// widening a positional comparison.
#[derive(Debug, PartialEq, Eq)]
struct LineageColumns {
    kind: CheckpointKind,
    snap_seq: Option<u64>,
    base_snap_seq: Option<u64>,
    from_seq: Option<u64>,
    session_id: String,
    layer_name: String,
    created_by: String,
    server_ts: u64,
    observed_commit_span: u64,
}

fn lineage_columns(meta: &CheckpointMeta) -> LineageColumns {
    LineageColumns {
        kind: meta.kind,
        snap_seq: meta.snap_seq,
        base_snap_seq: meta.base_snap_seq,
        from_seq: meta.from_seq,
        session_id: meta.session_id.clone(),
        layer_name: meta.layer_name.clone(),
        created_by: meta.created_by.clone(),
        server_ts: meta.server_ts,
        observed_commit_span: meta.observed_commit_span,
    }
}

/// Assert the shared shape of one leader's base PUT: the address, the kind, the
/// base-sourced identity, and the re-derived span.
fn assert_base_put_shape(record: &PutRecord, expected_to_seq: u64, expected_span: u64) {
    assert_eq!(record.object_id, OBJECT_ID);
    assert_eq!(
        record.kind,
        CheckpointKind::Base,
        "a re-base publishes a BASE at the collapsed head, never a tail"
    );
    assert_eq!(
        record.snap_seq, expected_to_seq,
        "the base's address IS S — the head of the lineage that was collapsed"
    );
    assert_eq!(record.meta.snap_seq, Some(expected_to_seq));
    assert_eq!(
        (record.meta.base_snap_seq, record.meta.from_seq),
        (None, None),
        "a Base carries neither lineage field — `CheckpointMeta`'s three-field invariant"
    );
    assert_eq!(
        (
            record.meta.session_id.as_str(),
            record.meta.layer_name.as_str(),
            record.meta.created_by.as_str(),
            record.meta.server_ts
        ),
        (
            SESSION_ID,
            LAYER_NAME,
            HAND_PUT_CREATED_BY,
            HAND_PUT_SERVER_TS
        ),
        "identity is BASE-sourced by the gather's MUST: a tail meta must never reach \
         `apply::live_mapping_resolves_blob_identity`, and both adapters' meta-absent tail \
         placeholders are identity-EMPTY"
    );
    assert_eq!(
        record.meta.observed_commit_span, expected_span,
        "the collapsed lineage's TOTAL span (S - base_snap_seq) — this base genuinely covers every \
         commit from the old root to S. The planted base carries 0, so a 0 here would mean the \
         span rode through from `adjudicate_composed_meta`'s struct update instead of being \
         re-derived"
    );
}

// ── Shape 1: identical S ────────────────────────────────────────────────────

/// SR-9.2, identical-S: two leaders that gather the same lineage behind a
/// rendezvous compose the same state and write BYTE-IDENTICAL base OMPF and
/// BYTE-IDENTICAL ext-refs sidecar to one address, so the blind
/// last-writer-wins PUT is safe.
///
/// The assertion is made on the two RECORDED payloads, not on the survivor (file
/// header, argument 2). The two leaders carry distinct `replica_id`s, which is
/// what makes the byte equality mean something: the replica label and the wall
/// clock reach `CheckpointMeta` and must NOT reach the bytes.
///
/// Items 1, 2 and 3 of the Subtask all land here: the byte pair, the diagnostic
/// classifier behind it, and the concurrent SR-4.3 ext backfill.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn duplicate_leaders_rebasing_one_lineage_write_byte_identical_bases() {
    let fixture = RebaseFixture::setup().await;

    // Neither leader leaves its listing until both have taken one, so neither
    // can observe the other's base and both root at the same lineage head.
    //
    // Load-bearing for determinism, not for today's schedule — the same status
    // the tail twin records for its own rendezvous, and verified the same way:
    // with the barrier removed this test still passes, because
    // `compose_from_artifacts` runs on `spawn_blocking` and the first leader
    // therefore YIELDS mid-re-base, letting the second take its listing before
    // any base is published. That is a scheduling coincidence of one `.await`
    // point. The barrier is what makes "both listed before either put" a FACT
    // under parallel-suite load; without it a second leader that listed after
    // the first published base@S would see a Base at the listing head, gather a
    // base-only lineage, answer `UnderBound`, and the whole comparison below
    // would never run.
    fixture.gates.arm_rendezvous(2);

    let before_ms = now_unix_ms();
    let (a, b) = tokio::time::timeout(
        RACE_DEADLINE,
        futures::future::join(
            fixture.rebase(&fixture.a, LEADER_A),
            fixture.rebase(&fixture.b, LEADER_B),
        ),
    )
    .await
    .expect(
        "the two concurrent re-bases must finish within RACE_DEADLINE. A timeout here means a \
         listing gate was never released (each is bounded by GATE_WAIT, so this should surface as \
         an assertion instead) or the gather budget was exhausted",
    );
    fixture.gates.disarm();
    let after_ms = now_unix_ms();

    // ── The decisions ───────────────────────────────────────────────────────
    let a_facts = expect_rebased(LEADER_A, &a);
    let b_facts = expect_rebased(LEADER_B, &b);
    for (who, facts) in [(LEADER_A, &a_facts), (LEADER_B, &b_facts)] {
        assert_eq!(
            (
                facts.base_snap_seq,
                facts.from_chain_len,
                facts.to_seq,
                facts.ext_backfilled
            ),
            (LINEAGE_ROOT_SEQ, CHAIN_SPANS.len(), HEAD_SEQ, 4),
            "leader {who} must collapse the whole planted chain: rooted at the planted base, \
             {} tails walked by the TRUE link walk (never the listing-position count, which the \
             two sibling tails inflate to {}), landing at S = {HEAD_SEQ}, with the four distinct \
             ext blobs backfilled (blob B appears in two records and must dedup to one)",
            CHAIN_SPANS.len(),
            CHAIN_SPANS.len() + OUT_OF_CHAIN_SPANS.len()
        );
        assert_eq!(
            (facts.handle.seq(), facts.handle.kind),
            (HEAD_SEQ, CheckpointKind::Base),
            "leader {who}'s returned handle must address the base it just published"
        );
    }

    // ── Both leaders really raced ───────────────────────────────────────────
    let a_put = fixture.a.sole_put();
    let b_put = fixture.b.sole_put();
    assert_base_put_shape(&a_put, HEAD_SEQ, HEAD_SEQ - LINEAGE_ROOT_SEQ);
    assert_base_put_shape(&b_put, HEAD_SEQ, HEAD_SEQ - LINEAGE_ROOT_SEQ);

    // ── THE IDEMPOTENCY EVIDENCE (items 1 + 2) ──────────────────────────────
    assert_payloads_byte_identical(&a_put, &b_put);

    // ── The lineage-carrying meta agrees; identity/clock deliberately do not ─
    assert_eq!(
        lineage_columns(&a_put.meta),
        lineage_columns(&b_put.meta),
        "both puts must claim the same lineage and the same base-sourced identity"
    );
    assert_ne!(
        a_put.meta.creator_replica_id, b_put.meta.creator_replica_id,
        "the two puts must be attributable to the two DISTINCT leaders — otherwise the byte \
         equality above would be trivially true"
    );
    assert_eq!(
        [
            a_put.meta.creator_replica_id.as_str(),
            b_put.meta.creator_replica_id.as_str()
        ],
        [LEADER_A, LEADER_B],
        "and each must carry its own label, not the base's"
    );
    // `written_at_unix_ms` is the OTHER value the determinism boundary keeps out
    // of the bytes. It is asserted as INDEPENDENTLY CLOCK-SAMPLED — not as
    // "differs between the two leaders": two leaders can legitimately land in the
    // same millisecond, so an inequality there would be a coin flip, not
    // evidence. The load-bearing claim is that the stamp comes from
    // `now_unix_ms()` at each leader rather than riding through from the
    // lineage, and that is exactly what these two bounds pin.
    for (who, meta) in [(LEADER_A, &a_put.meta), (LEADER_B, &b_put.meta)] {
        assert_ne!(
            meta.written_at_unix_ms, HAND_PUT_WRITTEN_AT_UNIX_MS,
            "leader {who} must stamp its OWN write time, not carry the planted base's through \
             `adjudicate_composed_meta`'s struct update"
        );
        assert!(
            meta.written_at_unix_ms >= before_ms && meta.written_at_unix_ms <= after_ms,
            "leader {who}'s written_at_unix_ms ({}) must fall inside the race window \
             [{before_ms}, {after_ms}]",
            meta.written_at_unix_ms
        );
    }

    // ── Item 3: concurrent ext backfill of the same hashes ──────────────────
    let expected_ext = expected_ext_backfill();
    for (who, store) in [(LEADER_A, &fixture.a), (LEADER_B, &fixture.b)] {
        let recorded = store.ext_puts();
        assert_eq!(
            recorded, expected_ext,
            "leader {who} must have PUT every distinct inline ext blob its replayed commits \
             referenced, once each, in first-appearance order — SR-4.3's hard precondition. \
             Neither leader errored (both returned Rebased), so this is the concurrent instance of \
             the idempotency claim ST-3 owns for a single leader"
        );
    }
    let a_ext: HashMap<String, Bytes> = fixture.a.ext_puts().into_iter().collect();
    let b_ext: HashMap<String, Bytes> = fixture.b.ext_puts().into_iter().collect();
    assert_eq!(
        a_ext, b_ext,
        "both leaders must have PUT byte-identical bytes under every hash — a content-addressed \
         re-put of the same hash with different bytes would make the store's last-writer-wins \
         behaviour a corruption rather than a no-op"
    );
    for (hash_hex, bytes) in &expected_ext {
        let stored = fixture
            .inner
            .get_ext_blob(hash_hex)
            .await
            .expect("get_ext_blob must succeed")
            .unwrap_or_else(|| panic!("ext blob {hash_hex} must be durable after the backfill"));
        assert_eq!(
            &stored, bytes,
            "the surviving bytes under {hash_hex} must equal what both leaders wrote"
        );
    }

    // ── Acceptance criterion 4, identical-S half ────────────────────────────
    let chain_to_seqs: Vec<u64> = CHAIN_SPANS.iter().map(|(_, to)| *to).collect();
    assert_fetched_exactly_the_chain(LEADER_A, &fixture.a.fetches(), &chain_to_seqs);
    assert_fetched_exactly_the_chain(LEADER_B, &fixture.b.fetches(), &chain_to_seqs);

    // ── The surviving artifact ──────────────────────────────────────────────
    let listing = fixture.listing().await;
    let bases: Vec<&CheckpointHandle> = listing
        .iter()
        .filter(|h| h.kind == CheckpointKind::Base)
        .collect();
    assert_eq!(
        bases.len(),
        2,
        "the planted lineage root plus ONE collapsed base: two puts at the same \
         `(object_id, S, Base)` must collapse to a single artifact. Listing: {listing:?}"
    );
    assert_eq!(
        listing.len(),
        1 + CHAIN_SPANS.len() + OUT_OF_CHAIN_SPANS.len() + 1,
        "a re-base deletes NOTHING: the planted base, every chain tail, both out-of-chain \
         siblings, and the new base must all be listed. Listing: {listing:?}"
    );

    let survivor = fixture
        .inner
        .get_layer_checkpoint(&a_facts.handle)
        .await
        .expect("get_layer_checkpoint must succeed")
        .expect("the returned handle must address a live artifact");
    assert_eq!(
        survivor.bytes, a_put.ompf,
        "whichever leader landed second, the surviving OMPF must equal what BOTH recorded — that \
         is what makes the overwrite a no-op rather than a coin flip"
    );
    assert_eq!(
        survivor.meta.sidecar_bytes.as_ref(),
        Some(a_put.sidecar()),
        "and so must the surviving ext-refs sidecar: the b.s5y publish-order obligation makes it \
         part of the published generation, so it is inside the SR-9.2 byte claim"
    );

    assert_out_of_chain_tails_intact(&fixture, &listing).await;
}

// ── Shape 2: divergent S ────────────────────────────────────────────────────

/// SR-9.2, divergent-S fork: one leader's listing is held while the driver
/// extends the lineage and the other leader re-bases at S. The held leader then
/// re-bases at S' > S. Two bases at DISTINCT keys, both durable, nothing
/// deleted — and neither gather ever consulted a generation outside its own
/// walked chain.
///
/// # Why the hold gates BEFORE the listing read
///
/// See the file header, divergence B. A re-base's listing is its ENTIRE view of
/// the lineage; holding an already-taken listing would hand both leaders the
/// same view and produce identical S, which is the other test.
///
/// # Why the driver waits on a signal rather than polling
///
/// The recording wrapper fires `base_put_landed` AFTER the adapter acknowledges
/// a base PUT, so "the un-held leader's base is durable" is an event, not a
/// condition to sample. That keeps this file free of sleeps entirely, and the
/// wait is bounded by the enclosing `RACE_DEADLINE`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn divergent_s_rebases_fork_into_two_bases_without_crossing_chains() {
    let fixture = RebaseFixture::setup().await;

    // The first leader to reach the listing is parked there — BEFORE it reads —
    // so its view of the lineage is the one taken after its rival published.
    let release = fixture.gates.arm_hold();

    let driver = async {
        // The un-held leader's DURABLE base is the signal that the window is
        // open. `notify_one` stores a permit, so a put that landed before this
        // await is not missed.
        tokio::time::timeout(RACE_DEADLINE, fixture.gates.base_put_landed.notified())
            .await
            .expect(
                "the un-held leader never published a durable base, so the fork window never \
                 opened. Both leaders cannot be held: the hold is one-shot and only the FIRST \
                 listing takes it",
            );
        fixture.land_fork_tails().await;
        let _ = release.send(());
    };

    let ((a, b), ()) = tokio::time::timeout(
        RACE_DEADLINE,
        futures::future::join(
            futures::future::join(
                fixture.rebase(&fixture.a, LEADER_A),
                fixture.rebase(&fixture.b, LEADER_B),
            ),
            driver,
        ),
    )
    .await
    .expect("the forked re-bases must finish within RACE_DEADLINE");
    fixture.gates.disarm();

    let a_facts = expect_rebased(LEADER_A, &a);
    let b_facts = expect_rebased(LEADER_B, &b);

    // Which leader was held is whichever reached the listing first, so the
    // assertions are on the sorted pair rather than on a named leader.
    let mut ends = [a_facts.to_seq, b_facts.to_seq];
    ends.sort_unstable();
    assert_eq!(
        ends,
        [HEAD_SEQ, FORK_HEAD_SEQ],
        "the held leader listed AFTER the driver extended the lineage, so the two re-bases must \
         land at S and S' respectively. Two equal values here mean the hold never applied and the \
         fixture proved nothing"
    );

    // Identify the two leaders by their outcome, not by their label.
    let (early, late) = if a_facts.to_seq < b_facts.to_seq {
        (
            (LEADER_A, &fixture.a, &a_facts),
            (LEADER_B, &fixture.b, &b_facts),
        )
    } else {
        (
            (LEADER_B, &fixture.b, &b_facts),
            (LEADER_A, &fixture.a, &a_facts),
        )
    };

    assert_eq!(
        (early.2.base_snap_seq, early.2.from_chain_len),
        (LINEAGE_ROOT_SEQ, CHAIN_SPANS.len()),
        "the un-held leader ({}) walked only the pre-fork chain",
        early.0
    );
    assert_eq!(
        (late.2.base_snap_seq, late.2.from_chain_len),
        (LINEAGE_ROOT_SEQ, CHAIN_SPANS.len() + FORK_SPANS.len()),
        "the held leader ({}) walked the EXTENDED chain — same lineage root, more links",
        late.0
    );

    // ── Two bases at DISTINCT keys, both durable ────────────────────────────
    let early_put = early.1.sole_put();
    let late_put = late.1.sole_put();
    assert_base_put_shape(&early_put, HEAD_SEQ, HEAD_SEQ - LINEAGE_ROOT_SEQ);
    assert_base_put_shape(&late_put, FORK_HEAD_SEQ, FORK_HEAD_SEQ - LINEAGE_ROOT_SEQ);
    assert_ne!(
        early_put.ompf, late_put.ompf,
        "different collapsed heads must carry different content; identical bytes would mean the \
         held leader re-composed the same lineage and this fixture proved nothing"
    );

    let listing = fixture.listing().await;
    assert_eq!(
        listing.len(),
        1 + CHAIN_SPANS.len() + OUT_OF_CHAIN_SPANS.len() + FORK_SPANS.len() + 2,
        "nothing may be deleted: the planted base, every chain tail, both siblings, both fork \
         tails, and BOTH new bases must be listed. Listing: {listing:?}"
    );
    for (seq, who) in [(HEAD_SEQ, early.0), (FORK_HEAD_SEQ, late.0)] {
        let handle = listing
            .iter()
            .find(|h| h.kind == CheckpointKind::Base && h.seq() == seq)
            .cloned()
            .unwrap_or_else(|| {
                panic!("{who}'s base at seq {seq} must be durable. Listing: {listing:?}")
            });
        let blob = fixture
            .inner
            .get_layer_checkpoint(&handle)
            .await
            .expect("get_layer_checkpoint must succeed")
            .expect("a listed base must address a live artifact");
        let expected = if seq == HEAD_SEQ {
            &early_put
        } else {
            &late_put
        };
        assert_eq!(
            blob.bytes, expected.ompf,
            "the base at seq {seq} must hold exactly what {who} recorded — the two re-bases \
             address DIFFERENT keys, so neither overwrites the other"
        );
        assert_eq!(
            blob.meta.sidecar_bytes.as_ref(),
            Some(expected.sidecar()),
            "and its ext-refs sidecar likewise"
        );
    }
    assert!(
        listing
            .iter()
            .any(|h| h.kind == CheckpointKind::Base && h.seq() == LINEAGE_ROOT_SEQ),
        "the ORIGINAL lineage root survives too: a re-base mints a new head and leaves the \
         superseded lineage for retention to prune as a unit. Listing: {listing:?}"
    );

    // ── The SR-9.2 claim that most needs evidence (item 4) ──────────────────
    //
    // The held leader's listing contained the OTHER leader's base at seq
    // HEAD_SEQ as well as both out-of-chain siblings, and its walk touched
    // neither: every link below the head names a Tail (the walk resolves `Base`
    // only at `base_snap_seq`), and the siblings' to_seqs are named by no link
    // at all.
    let early_chain: Vec<u64> = CHAIN_SPANS.iter().map(|(_, to)| *to).collect();
    let late_chain: Vec<u64> = CHAIN_SPANS
        .iter()
        .chain(FORK_SPANS.iter())
        .map(|(_, to)| *to)
        .collect();
    assert_fetched_exactly_the_chain(early.0, &early.1.fetches(), &early_chain);
    assert_fetched_exactly_the_chain(late.0, &late.1.fetches(), &late_chain);
    assert!(
        !late
            .1
            .fetches()
            .iter()
            .any(|(seq, kind)| *seq == HEAD_SEQ && *kind == CheckpointKind::Base),
        "the held leader ({}) must never have fetched the base its rival published at seq \
         {HEAD_SEQ}, even though its listing carried it: the link at that sequence names a TAIL, \
         because the walk resolves `Base` only at `base_snap_seq`. Fetches: {:?}",
        late.0,
        late.1.fetches()
    );

    assert_out_of_chain_tails_intact(&fixture, &listing).await;
}
