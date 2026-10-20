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

//! Cross-cutting recovery × snapshot-catch-up integration test (Subtask
//! `t3.msb.js.6j.1a`, Plan `b.msb`).
//!
//! ## What this test proves
//!
//! The faithful late-join catch-up snapshot
//! (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=true`, `compute_faithful_full_content_at_cut`)
//! emits each field with a DENSE-UNIQUE `set_order` (`0..N-1`, b.bdx), assigned
//! by sorting the net-live fields on the composite key `(version,
//! authoring_set_order)` — NOT the raw authoring rank and NOT the collapsed
//! cumulative cursor. `authoring_set_order` is materialized server-side (distinct
//! from the cumulative `set_order`) and is the WITHIN-VERSION SORT KEY that fixes
//! the emitted order.
//!
//! The DR checkpoint round-trip must preserve `authoring_set_order`: the
//! Engineer's `build_reset_ompf` fix threads each field's raw authoring
//! rank through the checkpoint serialize → restore cycle. If it did not, the
//! dense-ordering sort key would be lost and a snapshot served AFTER a
//! checkpoint restore would reconstruct the WRONG field order even though the
//! pre-restore snapshot for the same scene was correct.
//!
//! This is the equivalence proof (Task acceptance item (b)): a from-empty
//! snapshot catch-up served BEFORE a checkpoint capture/restore and one
//! served AFTER it, for the SAME scene, emit identical per-field
//! `SetField.set_order` (the dense-unique index) and identical relative
//! ordering — and the net-live pruning of a created-then-abandoned
//! scratch prim survives the round-trip too.
//!
//! ## Why the divergence is observable
//!
//! The scene mirrors the live-path establishment fixture in
//! `tests/integration/grpc/subscribe_regimes_test.rs::snapshot_late_join_catch_up_establishes_correct_scene`:
//! `childA.posX` is written in commit 1 (authoring rank 5) then BURST-rewritten
//! in commit 2 (authoring rank 2, last-writer-wins). Because commit 1 already
//! advanced the object's `next_set_order` cursor, the burst field's stored
//! CUMULATIVE `set_order` (10) diverges from both its authoring rank (2) and its
//! emitted dense index. If the checkpoint dropped `authoring_set_order`, the
//! within-version sort key would collapse and the post-restore snapshot would
//! emit fields in a different relative order, failing the equivalence assertion
//! below.
//!
//! ## Recovery path
//!
//! We use the SAME from-empty (brand-new: `object_id=0, seq=0`) subscribe
//! shape for BOTH the pre- and post-restore snapshots so the two bodies are
//! produced by the IDENTICAL builder (`catch_up_brand_new` →
//! `send_full_content_snapshot` → `compute_faithful_full_content_at_cut`) — the
//! equivalence comparison is then apples-to-apples. Recovery is driven
//! explicitly via `RecoveryHook::try_restore_layer` (the same call the lazy
//! subscribe-path recovery makes) after wiping `le:{oid}:obj:*`, which is
//! the deterministic form the Subtask body lists as acceptable.

use std::collections::{BTreeMap, HashSet};
use std::time::Duration;

use futures::{Stream, StreamExt};
use tonic::{Request, Status};

use live_edit::checkpoint::RecoveryOutcome;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, SubscribeReadLayerRequest, SubscribeReadLayerResponse,
};
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::diff_read::snapshot_to_object_state;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, SetField, StructCommand};

use super::helpers::*;
use crate::common::owned_delta::{delta_to_ompf_bytes, empty_delta};
use crate::grpc::helpers::{make_create_data, make_create_header, make_request, streaming_create};

const LAYER_NAME: &str = "authoring-rank-recovery-layer";
const SESSION_NAME: &str = "authoring-rank-recovery-session";
const WRITER_CLIENT_ID: &str = "authoring-rank-recovery-writer";
const SUBSCRIBER_CLIENT_ID: &str = "authoring-rank-recovery-subscriber";

/// Non-existent parent the scratch prim S(99) is reparented under to make it
/// net-live-unreachable (abandoned) WITHOUT deleting it — so it stays
/// materialized in Valkey (an orphan in `read_object_state_for_test`) and is
/// therefore captured by the checkpoint AND restored, yet must be DROPPED by
/// the faithful snapshot builder on both sides of the restore. Mirrors
/// `ABANDON_ORPHAN_PARENT` in the live-path establishment fixture.
const ABANDON_ORPHAN_PARENT: u64 = 888_888;

/// CreateLayer(seq=1) + 4 commits → established scene at seq=5.
const ESTABLISHED_SEQ: u64 = 5;

// ── Subscribe-stream helpers (file-local; the ones in the live-path fixture
//    and the sibling e2e recovery test are module-private to their files) ────

fn subscribe_request(
    session_id: &str,
    object_id: u64,
    sequence_num: u64,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: LAYER_NAME.to_owned(),
        object_id,
        sequence_num,
        client_id: SUBSCRIBER_CLIENT_ID.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// Collect a `Header` plus any `Chunk`s until `total_size` bytes are gathered.
/// Mirrors `recv_message` in the sibling checkpoint / grpc subscribe tests.
async fn recv_message<S>(
    stream: &mut S,
) -> (live_edit::grpc::proto::SubscribeReadLayerHeader, Vec<u8>)
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    let first = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("stream timed out waiting for next header")
        .expect("stream closed unexpectedly")
        .expect("stream yielded an error");

    let header = match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
        other => panic!("expected Header, got: {other:?}"),
    };

    let mut buf = Vec::new();
    while (buf.len() as u64) < header.total_size {
        let next = tokio::time::timeout(Duration::from_secs(10), stream.next())
            .await
            .expect("stream timed out waiting for chunk")
            .expect("stream closed before total_size reached")
            .expect("stream yielded an error");
        match next.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                buf.extend_from_slice(&c.data);
            }
            other => panic!("expected Chunk while filling body, got: {other:?}"),
        }
    }

    (header, buf)
}

/// Drain the `total_size == 0` caught-up terminator `run_catch_up` emits once
/// per catch-up phase (SDD §12.3). Keeps the stream cleanly terminated.
async fn recv_caught_up_ack<S>(stream: &mut S, object_id: u64)
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    let (header, body) = recv_message(stream).await;
    assert_eq!(
        header.total_size, 0,
        "caught-up terminator must be a zero-body ack (total_size==0)"
    );
    assert!(body.is_empty(), "caught-up ack must carry no body bytes");
    assert_eq!(
        header.object_id, object_id,
        "caught-up ack object_id must match the subscribed object"
    );
}

// ── Scene-build helper ──────────────────────────────────────────────────────

/// Issue one `UpdateLayer` carrying `delta` on the harness service. The
/// per-delta authoring ranks ride the `SetField.set_order` values inside
/// `delta`; the server persists each field's raw rank as `authoring_set_order`
/// while threading the divergent cumulative `set_order` cursor.
async fn apply_update(
    harness: &CheckpointHarness,
    session_id: &str,
    object_id: u64,
    timestamp: u64,
    delta: &OwnedDelta,
) {
    let ompf = delta_to_ompf_bytes(delta, &[]);
    let stream = streaming_update(vec![
        make_update_header(
            session_id,
            LAYER_NAME,
            object_id,
            timestamp,
            WRITER_CLIENT_ID,
        ),
        make_update_data(ompf),
    ]);
    handle_update_layer(&harness.svc, make_request(stream))
        .await
        .unwrap_or_else(|e| panic!("UpdateLayer (ts={timestamp}) failed: {e}"));
}

// ── Decode helpers ──────────────────────────────────────────────────────────

/// Decode a catch-up OMPF body and collect the emitted `SetField.set_order`
/// per `(section_id, key_name)` — for the faithful snapshot this is each
/// field's DENSE-UNIQUE index (b.bdx), ordered by `(version,
/// authoring_set_order)`. Simplest sufficient path for the ordering-equivalence
/// assertion: no full state reconstruction needed.
fn emitted_set_orders(body: &[u8]) -> BTreeMap<(u64, String), u64> {
    let container = OmpfContainer::parse(body).expect("parse catch-up OMPF");
    assert!(
        !container.delta_bytes.is_empty(),
        "catch-up body carried an EMPTY OODF delta (burst-coalesce-to-empty regression)"
    );
    let decoded = oodf_ffi::parse_delta(container.delta_bytes).expect("parse OODF delta");
    decoded
        .set_fields
        .iter()
        .map(|sf| ((sf.section_id, sf.key_name.clone()), sf.set_order))
        .collect()
}

/// Decode a catch-up OMPF body and collect the emitted `DiffChangeSection`
/// section ids (the net-live section set).
fn emitted_section_ids(body: &[u8]) -> HashSet<u64> {
    let container = OmpfContainer::parse(body).expect("parse catch-up OMPF");
    let decoded = oodf_ffi::parse_delta(container.delta_bytes).expect("parse OODF delta");
    decoded
        .struct_commands
        .iter()
        .filter_map(|c| match c {
            StructCommand::DiffChangeSection(s) => Some(s.section_id),
            _ => None,
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Task acceptance item (b): snapshot catch-up survives the DR round-trip.
// ═══════════════════════════════════════════════════════════════════════════

/// Establish a scene whose per-field authoring ranks diverge from the stored
/// cumulative cursor (plus a created-then-abandoned scratch prim), capture the
/// pre-restore from-empty snapshot as the baseline, then checkpoint → wipe →
/// restore and re-capture the from-empty snapshot. The two snapshots MUST emit
/// identical per-field `SetField.set_order` (the dense-unique index, whose
/// within-version sort key `authoring_set_order` must survive the round-trip)
/// and identical relative order, and both MUST drop the abandoned prim.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn snapshot_catch_up_authoring_rank_survives_checkpoint_restore() {
    let mut harness = setup_checkpoint_harness(1).await;
    // Faithful late-join snapshot mode — the path whose emitted set_order is
    // the true authoring rank rather than the cumulative cursor.
    harness.svc.catchup_snapshot_enabled = true;

    // Create the session + layer. The harness exposes an Arc<SessionStore>, but
    // a fresh store against the same Valkey is sufficient for the write path
    // (mirrors the sibling e2e recovery test).
    let session_store =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
    let session_id = session_store
        .create(SESSION_NAME, "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let create_ompf = delta_to_ompf_bytes(&empty_delta(), &[]);
    let create_stream = streaming_create(vec![
        make_create_header(&session_id, LAYER_NAME),
        make_create_data(create_ompf),
    ]);
    let object_id = handle_create_layer(&harness.svc, make_request(create_stream))
        .await
        .expect("CreateLayer must succeed")
        .into_inner()
        .object_id;

    // ── Commit 1 (seq=2): root R(1)+name(rank 1); child A(2)+posX="first"(rank 5). ──
    let commit1 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 0,
                section_id: 1,
                section_name: "root".to_owned(),
                section_type: 0,
                section_order: 0,
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 1,
                section_id: 2,
                section_name: "childA".to_owned(),
                section_type: 0,
                section_order: 10,
            }),
        ],
        set_fields: vec![
            SetField {
                section_id: 1,
                key_name: "name".to_owned(),
                value_or_ext_hash: b"scene".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            },
            SetField {
                section_id: 2,
                key_name: "posX".to_owned(),
                value_or_ext_hash: b"first".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 5,
            },
        ],
        ..Default::default()
    };
    apply_update(&harness, &session_id, object_id, 100, &commit1).await;

    // ── Commit 2 (seq=3): child B(3)+color(rank 7); BURST A.posX="final"(rank 2). ──
    // Commit 1 advanced next_set_order, so A.posX's stored CUMULATIVE set_order
    // (10) now diverges from its authoring rank (2) — the observable the
    // faithful builder must emit as the rank, not the cursor.
    let commit2 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 3,
            section_name: "childB".to_owned(),
            section_type: 0,
            section_order: 20,
        })],
        set_fields: vec![
            SetField {
                section_id: 2,
                key_name: "posX".to_owned(),
                value_or_ext_hash: b"final".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 2,
            },
            SetField {
                section_id: 3,
                key_name: "color".to_owned(),
                value_or_ext_hash: b"red".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 7,
            },
        ],
        ..Default::default()
    };
    apply_update(&harness, &session_id, object_id, 200, &commit2).await;

    // ── Commit 3 (seq=4): create scratch prim S(99) under child A + a field. ──
    let commit3 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 99,
            section_name: "scratch".to_owned(),
            section_type: 0,
            section_order: 30,
        })],
        set_fields: vec![SetField {
            section_id: 99,
            key_name: "temp".to_owned(),
            value_or_ext_hash: b"junk".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 3,
        }],
        ..Default::default()
    };
    apply_update(&harness, &session_id, object_id, 300, &commit3).await;

    // ── Commit 4 (seq=5): ABANDON scratch S(99) — reparent under a non-existent
    //    parent so it is unreachable from root (net-live pruning) WITHOUT
    //    deleting it (stays materialized as an orphan, so the checkpoint
    //    captures + restores it). ──
    let commit4 = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: ABANDON_ORPHAN_PARENT,
            section_id: 99,
            section_name: "scratch".to_owned(),
            section_type: 0,
            section_order: 30,
        })],
        ..Default::default()
    };
    apply_update(&harness, &session_id, object_id, 400, &commit4).await;

    // ── Pre-restore server sanity: A.posX's cumulative cursor (10) diverges
    //    from its authoring rank (2); the abandoned prim is still materialized. ──
    let pre_snapshot = read_object_state_for_test(
        &harness.svc.valkey,
        &harness.svc.key_schema,
        &object_id.to_string(),
    )
    .await
    .expect("pre-restore read_object_state_for_test must succeed");
    let pre_state = snapshot_to_object_state(&pre_snapshot);
    let pre_posx = pre_state
        .fields
        .get(&(2, "posX".to_owned()))
        .expect("server must carry A.posX");
    assert_eq!(
        pre_posx.authoring_set_order, 2,
        "pre-restore server A.posX authoring rank must be 2 (the true last-writer rank)"
    );
    assert_eq!(
        pre_posx.set_order, 10,
        "pre-restore server A.posX cumulative cursor must be 10, distinct from the rank"
    );
    assert!(
        pre_state.sections.contains_key(&99),
        "abandoned prim 99 must still be materialized server-side pre-restore (an orphan)"
    );
    assert_eq!(
        pre_state.sequence_num, ESTABLISHED_SEQ,
        "established scene must be at seq={ESTABLISHED_SEQ}"
    );

    // ── Pre-restore BASELINE: from-empty (brand-new) snapshot catch-up. ──
    let baseline_ranks = {
        let sub_req = subscribe_request(&session_id, 0, 0);
        let response = handle_subscribe_read_layer(&harness.svc, sub_req)
            .await
            .expect("pre-restore subscribe must succeed");
        let mut stream = response.into_inner();
        let (header, body) = recv_message(&mut stream).await;
        assert_eq!(header.object_id, object_id, "baseline header object_id");
        assert_eq!(
            header.sequence_num, ESTABLISHED_SEQ,
            "baseline snapshot delivers the current scene at seq={ESTABLISHED_SEQ}"
        );
        assert!(
            header.total_size > 0,
            "baseline snapshot body must be non-empty"
        );

        let section_ids = emitted_section_ids(&body);
        assert_eq!(
            section_ids,
            [1u64, 2, 3].into_iter().collect(),
            "baseline snapshot must emit exactly the live sections {{1,2,3}} — abandoned prim 99 dropped"
        );
        let ranks = emitted_set_orders(&body);
        assert!(
            !ranks.keys().any(|(sid, _)| *sid == 99),
            "baseline snapshot must not emit the abandoned prim's field"
        );
        // Per-field DENSE-UNIQUE set_order (b.bdx), ordered by (version,
        // authoring_set_order): name(v2,auth1)→0, posX(v3,auth2)→1,
        // color(v3,auth7)→2. NOT the raw authoring ranks (1/2/7) and NOT the
        // cumulative cursor.
        assert_eq!(
            ranks.get(&(1, "name".to_owned())),
            Some(&0),
            "R.name dense set_order"
        );
        assert_eq!(
            ranks.get(&(2, "posX".to_owned())),
            Some(&1),
            "A.posX dense set_order (last-writer burst survives; ordered after name)"
        );
        assert_eq!(
            ranks.get(&(3, "color".to_owned())),
            Some(&2),
            "B.color dense set_order"
        );
        // Global uniqueness across the catch-up (client drops duplicates).
        let mut orders: Vec<u64> = ranks.values().copied().collect();
        let n = orders.len();
        orders.sort_unstable();
        orders.dedup();
        assert_eq!(
            orders.len(),
            n,
            "baseline set_order values must be globally unique"
        );
        let name = ranks[&(1, "name".to_owned())];
        let posx = ranks[&(2, "posX".to_owned())];
        let color = ranks[&(3, "color".to_owned())];
        assert!(
            name < posx && posx < color,
            "baseline relative order must be name < posX < color"
        );

        recv_caught_up_ack(&mut stream, object_id).await;
        ranks
    };

    // ── Wait for a checkpoint generation capturing the full seq=5 scene. ──
    let _gen =
        wait_for_checkpoint_at_least(&harness.store, object_id, pre_state.sequence_num, 10).await;

    // ── Wipe every le:{oid}:obj:* key — the production data-loss surface. ──
    wipe_layer_state(&harness.tv, object_id).await;
    let post_wipe_probe = read_object_state_for_test(
        &harness.svc.valkey,
        &harness.svc.key_schema,
        &object_id.to_string(),
    )
    .await;
    match &post_wipe_probe {
        Err(live_edit::error::ServiceError::NotFound(_)) => {}
        Ok(_) => panic!(
            "wipe_layer_state must leave the object header missing — read_object_state_for_test \
             returned Ok unexpectedly"
        ),
        Err(other) => panic!(
            "wipe_layer_state must surface NotFound from read_object_state_for_test, got error: {other}"
        ),
    }

    // ── Restore from the checkpoint (the same call the lazy subscribe-path
    //    recovery makes). ──
    let outcome = harness.svc.recovery.try_restore_layer(object_id).await;
    assert!(
        matches!(outcome, RecoveryOutcome::Restored),
        "checkpoint restore must succeed (Restored), got {outcome:?}"
    );

    // ── Direct DR-fix cross-check: the restored object's A.posX must carry the
    //    TRUE authoring rank (2), NOT the cumulative cursor. Pre-fix, the
    //    checkpoint round-trip collapses authoring_set_order onto the cursor. ──
    let restored_snapshot = read_object_state_for_test(
        &harness.svc.valkey,
        &harness.svc.key_schema,
        &object_id.to_string(),
    )
    .await
    .expect("post-restore read_object_state_for_test must succeed");
    let restored_state = snapshot_to_object_state(&restored_snapshot);
    let restored_posx = restored_state
        .fields
        .get(&(2, "posX".to_owned()))
        .expect("restored state must carry A.posX");
    assert_eq!(
        restored_posx.authoring_set_order, 2,
        "restored A.posX authoring_set_order must equal the original rank (2), not the \
         cumulative cursor — this is the DR round-trip fidelity the build_reset_ompf fix restores"
    );
    assert!(
        restored_state.sections.contains_key(&99),
        "restore must re-materialize the abandoned orphan 99 (checkpoint captured it)"
    );

    // ── Post-restore snapshot: same from-empty (brand-new) shape → identical
    //    builder → the equivalence proof. ──
    let post_ranks = {
        let sub_req = subscribe_request(&session_id, 0, 0);
        let response = handle_subscribe_read_layer(&harness.svc, sub_req)
            .await
            .expect("post-restore subscribe must succeed");
        let mut stream = response.into_inner();
        let (header, body) = recv_message(&mut stream).await;
        assert_eq!(
            header.object_id, object_id,
            "post-restore header object_id must match"
        );
        assert!(
            header.total_size > 0,
            "post-restore snapshot body must be non-empty"
        );

        let section_ids = emitted_section_ids(&body);
        assert_eq!(
            section_ids,
            [1u64, 2, 3].into_iter().collect(),
            "post-restore snapshot must STILL drop the abandoned prim 99 (net-live pruning \
             survives the round-trip) — emit exactly {{1,2,3}}"
        );
        let ranks = emitted_set_orders(&body);
        assert!(
            !ranks.keys().any(|(sid, _)| *sid == 99),
            "post-restore snapshot must not emit the abandoned prim's field"
        );

        recv_caught_up_ack(&mut stream, object_id).await;
        ranks
    };

    // ── EQUIVALENCE (item (b)): pre- and post-restore per-field dense set_order
    //    values are identical. Fails pre-fix (post-restore emits the collapsed
    //    cumulative order once the within-version sort key is lost); passes after
    //    the build_reset_ompf fix threads authoring_set_order through DR. ──
    assert_eq!(
        post_ranks, baseline_ranks,
        "post-restore snapshot must emit per-field SetField.set_order identical to the \
         pre-restore baseline (dense-unique order preserved because authoring_set_order — the \
         within-version sort key — survives the DR round-trip)"
    );
    // Relative authoring order preserved post-restore too.
    let name = post_ranks[&(1, "name".to_owned())];
    let posx = post_ranks[&(2, "posX".to_owned())];
    let color = post_ranks[&(3, "color".to_owned())];
    assert!(
        name < posx && posx < color,
        "post-restore relative authoring order must be name < posX < color"
    );

    // Tear the periodic loop down cleanly.
    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}
