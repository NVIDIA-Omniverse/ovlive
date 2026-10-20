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

// Subtask `t3.dfh.57.1u.e6`, Part 2 — SDD SR-9.2: the duplicate-leader race on
// tail capture.
//
// ── Why this file drives the capture UNIT, not two loops ────────────────────
//
// SR-9.2's claim is that a *duplicate leadership* window is safe: two replicas
// that both believe they hold `le:checkpoint:leader` may both capture a tail for
// the same object, and neither outcome corrupts the lineage. There are two
// shapes, and the difference is whether the stream moved between their reads:
//
//   - identical S (nothing committed in between) — both write the SAME
//     `(object_id, to_seq, tail)` address with BYTE-IDENTICAL content, so the
//     blind last-writer-wins PUT is a no-op whichever lands second;
//   - divergent S (commits landed in between) — they write DIFFERENT addresses,
//     leaving an overlapped tail that is simply not a member of the lineage
//     walked back from the head.
//
// Two `spawn_checkpoint_loop`s cannot express either: they serialize on the
// leader lock by design. b.s5y (commit 975a7a6) inverted the design goal to SAFE
// UNDER duplicate leadership but landed no two-concurrent-leaders driver, and
// `checkpointer_leader_race_test.rs:257-264` proves the OPPOSITE contract (the
// recorded `creator_replica_id` set is a SINGLETON). Defeating the lock to
// manufacture the window would be neither deterministic nor faithful.
//
// So this file invokes `live_edit::checkpoint::tail_capture::capture_layer_tail`
// directly, twice, concurrently — the same way `assembler_test.rs:34-44` drives
// `assemble_layer_checkpoint` from the integration crate, and the reason that
// entry point is `pub` (its doc names this test). The two invocations carry
// DISTINCT `replica_id`s, which is what makes the byte-equality assertion mean
// something: the replica label reaches `CheckpointMeta`, deliberately outside
// the envelope bytes the determinism argument depends on (SDD §22.19 Invariant
// 2, restated in `tail_capture.rs`'s "Determinism boundary" header).
//
// ── Observation protocol: RECORD the puts, do not read the survivor ─────────
//
// Reading the artifact back after two captures cannot witness idempotency: both
// puts target one key, so the second OVERWRITES the first and any post-hoc read
// sees exactly one blob no matter what the two leaders wrote. The decision type
// carries a handle, not bytes, so it cannot close the gap either. `RecordingTailPutStore`
// therefore captures `(snap_seq, kind, ompf, meta)` per `put_layer_checkpoint`;
// the byte-equality assertion is made between the two RECORDED payloads, and the
// surviving artifact is then checked to equal them.
//
// ── Making the race deterministic ───────────────────────────────────────────
//
// Every capture reads its lineage head (`list_layer_checkpoints`) before it
// writes, so "two leaders in the same window" is exactly "both listed before
// either put". The recording store gates that seam, which is the only place a
// test can express the window without sleeping:
//
//   - identical-S: a rendezvous `Barrier(2)` on the listing return — neither
//     capture leaves its listing until both have taken one, so neither can see
//     the other's artifact and both root at the same S0.
//   - divergent-S: a one-shot HOLD on the FIRST listing return — the inner
//     listing is read first and handed back only after release, which models a
//     leader whose view of the lineage was taken before its rival's put landed.
//     The driver lands more commits in that window, so the held leader's slice
//     read (which happens after release) ends higher.
//
// Both gates are bounded by `GATE_WAIT`; a test that never releases fails on an
// assertion rather than hanging until nextest's 120 s hard kill.
//
// Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; under the
// project sandbox that variable is preset (CLAUDE.md).

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use tokio::sync::{oneshot, Barrier};
use tokio::time::Instant;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::tail_capture::{capture_layer_tail, TailCaptureDecision};
use live_edit::checkpoint::{
    decode_tail_envelope, CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointKind,
    CheckpointMeta, CheckpointStore, LayerArtifact, TailCaptureConfig,
};
use live_edit::domain::types::ObjectState;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{OwnedDelta, SetField};

use crate::checkpoint::helpers::{put_base_generation, DelegatingCheckpointStore};

/// The lineage root every fixture here plants, and the `sequence_num` its layer
/// is created at.
///
/// The coupling is load-bearing, not cosmetic (the lesson `helpers::CAPTURED_TAIL_SEED_SEQ`
/// records): the capture opens its read window at `{S0 + 1}-0`, so a base whose
/// `snap_seq` differs from the layer's creation sequence leaves a hole below the
/// window floor and every capture refuses `Trimmed` instead. `1` also mirrors
/// SR-7.1's own bootstrap base, so these fixtures sit on the lineage shape
/// production actually produces.
const LINEAGE_ROOT_SEQ: u64 = 1;

/// Commits landed before the two leaders race.
const PRE_RACE_COMMITS: u64 = 3;

/// Commits the divergent-S driver lands between the two leaders' slice reads.
const FORK_COMMITS: u64 = 2;

/// Ceiling on either listing gate, and on the whole two-leader join.
const GATE_WAIT: Duration = Duration::from_secs(10);
const RACE_DEADLINE: Duration = Duration::from_secs(30);

/// The one section every fixture layer carries, so the writer's field-set
/// deltas land somewhere real.
const RACE_SECTION_ID: u64 = 1;

const CREATED_BY: &str = "tail-duplicate-leader-user";
const LEADER_A: &str = "duplicate-leader-a";
const LEADER_B: &str = "duplicate-leader-b";

// ── Recording store ─────────────────────────────────────────────────────────

/// One `put_layer_checkpoint` as the wrapper saw it, payload included.
#[derive(Debug, Clone)]
struct PutRecord {
    object_id: u64,
    snap_seq: u64,
    kind: CheckpointKind,
    /// The exact bytes handed to the adapter. Recording these — rather than
    /// reading the survivor back — is what makes the idempotency claim
    /// observable at all.
    ompf: Bytes,
    /// Kept so lineage fields can be compared BETWEEN the two puts. Only the
    /// lineage-carrying fields are asserted; `creator_replica_id` and
    /// `written_at_unix_ms` are expected to differ and are the point.
    meta: CheckpointMeta,
}

/// `CheckpointStore` newtype recording every layer PUT and gating the listing
/// seam, composed atop [`DelegatingCheckpointStore`] so every other method
/// forwards verbatim — including the four with non-trivial trait defaults
/// (`read_ext_refs_sidecar`, `list_layer_artifacts`, `delete_layer_temp_artifacts`,
/// `publish_worst_case`). Omitting one of those does not leave a call
/// un-instrumented; it SUBSTITUTES behaviour behind the wrapper and makes every
/// assertion about it vacuously green, which is the trap that helper's doc
/// records.
struct RecordingTailPutStore {
    delegate: DelegatingCheckpointStore<LocalDiskCheckpointStore>,
    records: Mutex<Vec<PutRecord>>,
    /// Both listing calls wait here before returning (identical-S shape).
    list_rendezvous: Mutex<Option<Arc<Barrier>>>,
    /// The FIRST listing call takes this and parks on it, having already read
    /// the inner listing (divergent-S shape).
    list_hold: Mutex<Option<oneshot::Receiver<()>>>,
}

impl RecordingTailPutStore {
    fn new(inner: Arc<LocalDiskCheckpointStore>) -> Self {
        Self {
            delegate: DelegatingCheckpointStore::new(inner),
            records: Mutex::new(Vec::new()),
            list_rendezvous: Mutex::new(None),
            list_hold: Mutex::new(None),
        }
    }

    /// Hold every listing return until `parties` of them have arrived.
    fn arm_list_rendezvous(&self, parties: usize) {
        *self
            .list_rendezvous
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(Arc::new(Barrier::new(parties)));
    }

    /// Park the first listing return until the returned sender fires.
    fn arm_list_hold(&self) -> oneshot::Sender<()> {
        let (tx, rx) = oneshot::channel();
        *self.list_hold.lock().unwrap_or_else(|p| p.into_inner()) = Some(rx);
        tx
    }

    /// Clear both gates. Called once a race has finished so the test's own
    /// post-hoc listings do not park on a rendezvous that will never see a
    /// second party (which would cost a full [`GATE_WAIT`] per read).
    fn disarm_gates(&self) {
        *self
            .list_rendezvous
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = None;
        *self.list_hold.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }

    fn records(&self) -> Vec<PutRecord> {
        self.records
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    fn record_count(&self) -> usize {
        self.records.lock().unwrap_or_else(|p| p.into_inner()).len()
    }
}

#[async_trait]
impl CheckpointStore for RecordingTailPutStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.records
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(PutRecord {
                object_id,
                snap_seq,
                kind: meta.kind,
                ompf: ompf.clone(),
                meta: meta.clone(),
            });
        self.delegate
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        // Read FIRST, gate second. That ordering is the divergent-S shape: the
        // held leader's view of the lineage is the one it took before its rival
        // wrote, which is precisely what a duplicate-leadership window is.
        let handles = self.delegate.list_layer_checkpoints(object_id).await?;

        let rendezvous = self
            .list_rendezvous
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        if let Some(barrier) = rendezvous {
            let _ = tokio::time::timeout(GATE_WAIT, barrier.wait()).await;
        }

        let hold = self
            .list_hold
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        if let Some(rx) = hold {
            let _ = tokio::time::timeout(GATE_WAIT, rx).await;
        }

        Ok(handles)
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_latest_layer_checkpoint(object_id).await
    }

    // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST — its trait
    // default returns `NotImplemented`, which would report "no such generation"
    // for a generation the delegate holds, to the reader whose whole job is
    // deciding what exists.
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_layer_checkpoint(handle).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.delegate.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
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
        self.delegate.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.delegate.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.delegate.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.delegate.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.delegate.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.delegate.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
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

    // The four below carry non-trivial trait defaults — see the type's doc.
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
        self.delegate
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> Duration {
        self.delegate.publish_worst_case()
    }
}

// ── Fixture ─────────────────────────────────────────────────────────────────

/// Everything a race needs: a live layer at [`LINEAGE_ROOT_SEQ`] with a base
/// planted at the same sequence, plus the recording store the leaders write
/// through.
struct RaceFixture {
    tv: TestValkey,
    layer_store: Arc<LayerStore>,
    store: Arc<RecordingTailPutStore>,
    session_id: String,
    layer_name: String,
    object_id: u64,
    /// LocalDisk root; held so the adapter's files outlive the fixture.
    _tempdir: tempfile::TempDir,
}

impl RaceFixture {
    async fn setup(layer_name: &str) -> Self {
        let tv = TestValkey::setup_with_pool_size(8).await;
        let tempdir = tempfile::tempdir().expect("tempdir creation must succeed");
        let inner = Arc::new(
            LocalDiskCheckpointStore::new(tempdir.path().to_path_buf())
                .expect("LocalDiskCheckpointStore::new must succeed on a fresh root"),
        );
        let store = Arc::new(RecordingTailPutStore::new(Arc::clone(&inner)));

        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create("tail-duplicate-leader-session", "/stage", CREATED_BY)
            .await
            .expect("session creation must succeed");

        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        // One real section, so the writer's field-set deltas below land on a
        // section that exists rather than being dropped by apply.
        let mut state = ObjectState {
            sequence_num: LINEAGE_ROOT_SEQ,
            first_version: 1,
            source_format: "usda".to_string(),
            ..Default::default()
        };
        state.sections.insert(
            RACE_SECTION_ID,
            live_edit::domain::types::Section {
                section_id: RACE_SECTION_ID,
                parent_id: 0,
                name: "race-section".to_string(),
                section_type: 1,
                section_order: 0,
                first_version: 1,
                last_version: 1,
            },
        );
        state.children.insert(0, vec![RACE_SECTION_ID]);
        let object_id = layer_store
            .create_layer(
                &session_id,
                layer_name,
                CREATED_BY,
                &state,
                Arc::new(MockExternalValueStore::new()),
            )
            .await
            .expect("create_layer must succeed");

        // Planted through the INNER store: a hand-put fixture generation is not
        // one of the leaders' puts and must not appear in the recording.
        let inner_dyn: Arc<dyn CheckpointStore> = inner;
        put_base_generation(
            &inner_dyn,
            object_id,
            &session_id,
            layer_name,
            LINEAGE_ROOT_SEQ,
        )
        .await;

        Self {
            tv,
            layer_store,
            store,
            session_id,
            layer_name: layer_name.to_string(),
            object_id,
            _tempdir: tempdir,
        }
    }

    /// Land `n` ompf-bearing field-set commits, returning the new header
    /// sequence.
    ///
    /// The `relay_ompf` is mandatory, not decoration: an `ompf`-less update
    /// entry is read by the tail classifier as an EMPTY commit by invariant, so
    /// a fixture built without it would record real commits as no-ops and every
    /// byte-comparison below would be comparing two envelopes full of nothing.
    async fn commit(&self, n: u64) -> u64 {
        let mut seq = 0;
        for i in 0..n {
            let nonce = 10_000 + i;
            let delta = OwnedDelta {
                set_fields: vec![SetField {
                    section_id: RACE_SECTION_ID,
                    key_name: "race-field".to_string(),
                    value_or_ext_hash: nonce.to_le_bytes().to_vec(),
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: nonce,
                }],
                ..Default::default()
            };
            let ompf = Bytes::from(crate::common::owned_delta::delta_to_ompf_bytes(&delta, &[]));
            let (new_seq, _) = self
                .layer_store
                .update_layer(
                    &self.session_id,
                    &self.layer_name,
                    self.object_id,
                    Some(delta),
                    Some(ompf),
                    "race-writer",
                    0,
                )
                .await
                .expect("update_layer must succeed for the race fixture");
            seq = new_seq;
        }
        seq
    }

    /// Invoke the capture unit as `replica_id` would.
    async fn capture(&self, replica_id: &str) -> TailCaptureDecision {
        capture_layer_tail(
            self.tv.client(),
            self.tv.key_schema(),
            self.object_id,
            self.store.as_ref(),
            replica_id,
            &TailCaptureConfig::default(),
        )
        .await
        .unwrap_or_else(|e| panic!("capture_layer_tail({replica_id}) must not error: {e}"))
    }

    async fn listing(&self) -> Vec<CheckpointHandle> {
        self.store
            .list_layer_checkpoints(self.object_id)
            .await
            .expect("list_layer_checkpoints must succeed")
    }
}

/// Unwrap a decision to its `Captured` facts, with a message naming the refusal
/// otherwise — a refusal here is always a fixture defect, never the property
/// under test.
fn expect_captured(who: &str, decision: &TailCaptureDecision) -> (u64, u64, usize) {
    match decision {
        TailCaptureDecision::Captured {
            from_seq,
            to_seq,
            record_count,
            ..
        } => (*from_seq, *to_seq, *record_count),
        TailCaptureDecision::Refused(refusal) => panic!(
            "leader {who} refused ({refusal:?}, outcome {:?}) instead of capturing. Every refusal \
             reachable from this fixture is a fixture defect: no base planted, the S0 / creation \
             sequence coupling broken, or an ompf-less commit in the window",
            decision.outcome_label()
        ),
    }
}

async fn poll_until<F, Fut>(deadline: Duration, mut pred: F) -> bool
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let start = Instant::now();
    while start.elapsed() < deadline {
        if pred().await {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    pred().await
}

// ── Shape 1: identical S ────────────────────────────────────────────────────

/// SR-9.2, identical-S: two leaders that read the same lineage head over an
/// unchanged stream write byte-identical envelopes to one address, so the blind
/// last-writer-wins PUT is safe.
///
/// The assertion is made on the two RECORDED payloads, not on the survivor —
/// see the file header's observation-protocol note. The two leaders carry
/// distinct `replica_id`s, so the byte equality is evidence for the determinism
/// boundary itself: the replica label and the wall clock reach `CheckpointMeta`
/// and must NOT reach the envelope.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn duplicate_leaders_over_an_unchanged_stream_write_byte_identical_envelopes() {
    let fixture = RaceFixture::setup("identical-s-layer").await;
    let head_seq = fixture.commit(PRE_RACE_COMMITS).await;
    assert_eq!(
        head_seq,
        LINEAGE_ROOT_SEQ + PRE_RACE_COMMITS,
        "fixture precondition: the header sequence must advance one per commit"
    );

    // Neither leader leaves its listing until both have taken one, so neither
    // can observe the other's artifact and both root at S0 = LINEAGE_ROOT_SEQ.
    //
    // Load-bearing for determinism, not for today's schedule: with the barrier
    // removed the test still passes, because `join` polls A into its listing
    // (a real I/O await) and B lists before A returns. That is a scheduling
    // coincidence under an unloaded box — the barrier is what makes "both
    // listed before either put" a fact under parallel-suite load.
    fixture.store.arm_list_rendezvous(2);

    let (a, b) = tokio::time::timeout(
        RACE_DEADLINE,
        futures::future::join(fixture.capture(LEADER_A), fixture.capture(LEADER_B)),
    )
    .await
    .expect("the two concurrent captures must finish within the race deadline");
    fixture.store.disarm_gates();

    let (a_from, a_to, a_records) = expect_captured(LEADER_A, &a);
    let (b_from, b_to, b_records) = expect_captured(LEADER_B, &b);
    assert_eq!(
        (a_from, a_to, a_records),
        (b_from, b_to, b_records),
        "over an unchanged stream both leaders must classify the same prefix"
    );
    assert_eq!(a_from, LINEAGE_ROOT_SEQ, "S0 is the planted lineage root");
    assert_eq!(a_to, head_seq, "the window must reach the header sequence");

    // ── The idempotency evidence ────────────────────────────────────────────
    let records = fixture.store.records();
    assert_eq!(
        records.len(),
        2,
        "exactly two puts must have been ISSUED — one per leader. Fewer means the rendezvous \
         failed and the second leader saw the first's artifact (its capture would then refuse); \
         more means something else wrote. Records: {records:?}"
    );
    for record in &records {
        assert_eq!(record.kind, CheckpointKind::Tail);
        assert_eq!(record.object_id, fixture.object_id);
        assert_eq!(
            record.snap_seq, head_seq,
            "a tail's ordering sequence IS its to_seq, so both puts address the same key"
        );
    }
    assert_eq!(
        records[0].ompf,
        records[1].ompf,
        "SDD §22.19 Invariant 2: two leaders over the same immutable slice and lineage head must \
         emit BYTE-IDENTICAL envelopes, which is the whole basis for the blind last-writer-wins \
         PUT. Lengths: {} vs {}",
        records[0].ompf.len(),
        records[1].ompf.len()
    );

    // Lineage fields — NOT whole-meta: `creator_replica_id` and
    // `written_at_unix_ms` are expected to differ, and are exactly the values
    // the determinism boundary keeps OUT of the bytes.
    let (m0, m1) = (&records[0].meta, &records[1].meta);
    assert_eq!(
        (m0.kind, m0.base_snap_seq, m0.from_seq, m0.snap_seq),
        (m1.kind, m1.base_snap_seq, m1.from_seq, m1.snap_seq),
        "both puts must claim the same lineage: (kind, base_snap_seq, from_seq, snap_seq)"
    );
    assert_eq!(
        (
            m0.session_id.as_str(),
            m0.layer_name.as_str(),
            m0.observed_commit_span
        ),
        (
            m1.session_id.as_str(),
            m1.layer_name.as_str(),
            m1.observed_commit_span
        ),
        "identity and span are header-derived, so they must agree across leaders too"
    );
    let replicas: HashSet<&str> = records
        .iter()
        .map(|r| r.meta.creator_replica_id.as_str())
        .collect();
    assert_eq!(
        replicas,
        HashSet::from([LEADER_A, LEADER_B]),
        "the two puts must be attributable to the two distinct leaders — otherwise the byte \
         equality above would be trivially true"
    );

    // ── The surviving artifact ──────────────────────────────────────────────
    let listing = fixture.listing().await;
    let tails: Vec<&CheckpointHandle> = listing
        .iter()
        .filter(|h| h.kind == CheckpointKind::Tail)
        .collect();
    assert_eq!(
        tails.len(),
        1,
        "duplicate `(object_id, to_seq, tail)` puts must collapse to exactly ONE artifact. \
         Listing: {listing:?}"
    );
    assert_eq!(tails[0].snap_seq, head_seq);
    let blob = fixture
        .store
        .get_layer_checkpoint(tails[0])
        .await
        .expect("get_layer_checkpoint must succeed")
        .expect("the surviving handle must address a live artifact");
    assert_eq!(
        blob.bytes, records[0].ompf,
        "whichever leader landed second, the surviving bytes must equal what BOTH recorded — that \
         is what makes the overwrite a no-op rather than a coin flip"
    );

    let envelope = decode_tail_envelope(&blob.bytes).expect("the artifact must be a tail envelope");
    assert_eq!(envelope.header().from_seq, LINEAGE_ROOT_SEQ);
    assert_eq!(envelope.header().to_seq, head_seq);
    assert_eq!(
        envelope.records().len() as u64,
        head_seq - LINEAGE_ROOT_SEQ,
        "claimed range must EQUAL carried content (b.pw6)"
    );
}

// ── Shape 2: divergent S ────────────────────────────────────────────────────

/// SR-9.2, divergent-S fork: two leaders that read the same lineage head but
/// different stream extents write DIFFERENT addresses. The newer artifact
/// becomes the lineage head; the overlapped one is left behind as a non-member.
///
/// # Non-membership is asserted via listing + lineage metas, and why
///
/// The unit exposes no chain-walk API to assert against, and the `chain_len` it
/// reports is deliberately the CONSERVATIVE count — the position of the newest
/// base in the listing, which counts the overlapped tail even though a true
/// `from`/`to` link-walk would not traverse it (`tail_capture.rs`, "The chain
/// count is a documented conservative over-approximation"). Asserting
/// non-membership through that count would therefore assert the opposite of the
/// truth.
///
/// A true link-walk length DOES exist in the system now —
/// `GatheredLineage::chain_len`, which the SR-4.1 chain-collapse re-base
/// consumes — but it is reached only downstream of a full lineage gather, which
/// is a cost class this unit-level fixture has no business paying to make one
/// membership assertion. The over-count's consequence is likewise no longer a
/// lost tick: the loop's re-base answers it with `RebaseDecision::UnderBound` and
/// re-runs the capture under `ChainBound::Verified`, so the layer captures. That
/// resolution is pinned in the re-base lane; the raw refusal it starts from is
/// pinned by the chain-full outcome test. Neither is this test's subject.
///
/// So the walk is performed on the retrieved metas, which is where the link data
/// actually lives: from the newest tail, `from_seq` names the generation this
/// tail layers onto, and `base_snap_seq` names the lineage root. The assertion
/// form is: `head.from_seq == head.base_snap_seq == base.snap_seq`, i.e. the
/// walk from the head reaches the base in ONE hop, and the overlapped tail's own
/// `to_seq` is not that hop — so no chain rooted at the head visits it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn divergent_s_fork_leaves_two_keys_with_the_newest_as_the_lineage_head() {
    let fixture = RaceFixture::setup("divergent-s-layer").await;
    let early_seq = fixture.commit(PRE_RACE_COMMITS).await;

    // The first leader to reach the listing is parked there, holding the view it
    // took BEFORE its rival's put — the duplicate-leadership window itself.
    // Removing this gate makes both leaders read the SAME window (both end at
    // `early_seq`) and the fork assertion below fails loudly — verified, so the
    // gate is what creates the divergence rather than decorating it.
    let release = fixture.store.arm_list_hold();

    let driver = async {
        // The un-held leader's put is the signal that the window is open.
        let landed = poll_until(RACE_DEADLINE, || async {
            fixture.store.record_count() >= 1
        })
        .await;
        assert!(
            landed,
            "the un-held leader never issued a put, so the fork window never opened"
        );
        let late_seq = fixture.commit(FORK_COMMITS).await;
        let _ = release.send(());
        late_seq
    };

    let ((a, b), late_seq) = tokio::time::timeout(
        RACE_DEADLINE,
        futures::future::join(
            futures::future::join(fixture.capture(LEADER_A), fixture.capture(LEADER_B)),
            driver,
        ),
    )
    .await
    .expect("the forked captures must finish within the race deadline");
    fixture.store.disarm_gates();

    let (a_from, a_to, _) = expect_captured(LEADER_A, &a);
    let (b_from, b_to, _) = expect_captured(LEADER_B, &b);
    assert_eq!(
        (a_from, b_from),
        (LINEAGE_ROOT_SEQ, LINEAGE_ROOT_SEQ),
        "both leaders listed before either wrote, so BOTH root at the same S0 — that is what \
         makes this a fork rather than a chain"
    );
    let mut ends = [a_to, b_to];
    ends.sort_unstable();
    assert_eq!(
        ends,
        [early_seq, late_seq],
        "the held leader's slice read happened after the driver's commits, so the two windows \
         must end at the pre-fork and post-fork header sequences respectively"
    );

    // Two puts, two DISTINCT addresses.
    let records = fixture.store.records();
    assert_eq!(records.len(), 2, "one put per leader. Records: {records:?}");
    assert!(
        records.iter().all(|r| r.kind == CheckpointKind::Tail),
        "both artifacts are tails: {records:?}"
    );
    let mut recorded_seqs: Vec<u64> = records.iter().map(|r| r.snap_seq).collect();
    recorded_seqs.sort_unstable();
    assert_eq!(
        recorded_seqs,
        vec![early_seq, late_seq],
        "divergent S means divergent `(object_id, to_seq, tail)` keys — nothing is overwritten \
         here, which is why the overlap survives"
    );
    assert_ne!(
        records[0].ompf, records[1].ompf,
        "different windows must carry different content; identical bytes would mean the held \
         leader re-read the same slice and this fixture proved nothing"
    );

    // The listing: newest-first, so the head is the newer tail.
    let listing = fixture.listing().await;
    assert_eq!(
        listing.len(),
        3,
        "base + two forked tails must all be durable. Listing: {listing:?}"
    );
    assert_eq!(
        (listing[0].kind, listing[0].snap_seq),
        (CheckpointKind::Tail, late_seq),
        "`list_layer_checkpoints` is newest-first, so the lineage head is the LATER tail. \
         Listing: {listing:?}"
    );

    // The lineage-meta walk (see this test's doc for the form and why).
    let head = listing[0].clone();
    let overlapped = listing
        .iter()
        .find(|h| h.kind == CheckpointKind::Tail && h.snap_seq == early_seq)
        .cloned()
        .unwrap_or_else(|| panic!("the overlapped tail must still exist. Listing: {listing:?}"));
    let base = listing
        .iter()
        .find(|h| h.kind == CheckpointKind::Base)
        .cloned()
        .unwrap_or_else(|| panic!("the lineage root must survive. Listing: {listing:?}"));

    let head_meta = fixture
        .store
        .get_layer_checkpoint(&head)
        .await
        .expect("get_layer_checkpoint must succeed")
        .expect("the head handle must address a live artifact")
        .meta;
    let overlapped_meta = fixture
        .store
        .get_layer_checkpoint(&overlapped)
        .await
        .expect("get_layer_checkpoint must succeed")
        .expect("the overlapped handle must address a live artifact")
        .meta;

    assert_eq!(
        head_meta.from_seq,
        Some(base.snap_seq),
        "one hop: the head tail layers directly onto the BASE, so the walk terminates without \
         visiting anything else"
    );
    assert_eq!(
        head_meta.base_snap_seq,
        Some(base.snap_seq),
        "and it names the same generation as its lineage root"
    );
    assert_ne!(
        head_meta.from_seq,
        Some(overlapped.snap_seq),
        "NON-MEMBERSHIP: the head does not link to the overlapped tail's to_seq ({}), so no chain \
         walked back from the head traverses it",
        overlapped.snap_seq
    );
    assert_eq!(
        overlapped_meta.from_seq,
        Some(base.snap_seq),
        "the overlapped tail is a sibling of the head, not its predecessor: it roots at the same \
         base, which is exactly what makes the two a FORK"
    );

    // Both envelopes are individually well-formed — the fork leaves no torn
    // artifact behind, it leaves two honest ones.
    for (handle, expected_to) in [(&head, late_seq), (&overlapped, early_seq)] {
        let blob = fixture
            .store
            .get_layer_checkpoint(handle)
            .await
            .expect("get_layer_checkpoint must succeed")
            .expect("handle must address a live artifact");
        let envelope = decode_tail_envelope(&blob.bytes).expect("each forked artifact must decode");
        assert_eq!(envelope.header().from_seq, LINEAGE_ROOT_SEQ);
        assert_eq!(envelope.header().to_seq, expected_to);
        assert_eq!(
            envelope.records().len() as u64,
            expected_to - LINEAGE_ROOT_SEQ,
            "claimed range must EQUAL carried content (b.pw6) for {handle:?}"
        );
    }
}
