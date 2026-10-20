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

// b.nvf — per-object external-value reference counts (`le:{oid}:obj:ext`)
// drifting from the rows, at the two commit sites this crate's external tests
// can reach.
//
// A count that drifts LOW lets a later commit release a reference a live row
// still holds, and the GC then deletes a blob in use. Two producers are pinned
// here; the reconciler's audit and drain gate, and the recount script itself,
// are pinned in-crate (`src/storage/ext_reconcile.rs`,
// `src/storage/ext_index_audit.rs`) because their entry points are private.
//
// ── 1. The counting rule at the commit pre-image read ────────────────────────
//
// A value row holds external hash h iff it has NO `value` column and its
// `hash` column is h — the decoder's rule. A pre-b.zvd dual-shape row (an
// inline `value` beside a stale `hash`) therefore holds no hash. Both commit
// paths read a rewritten row's OLD hash before clearing it: the Lua script's
// `SF` / `ST` arms and the txn backend's `read_ext_pre_image`. Read raw, the
// stale column is "released" as the row's old hash, decrementing a count the
// row never contributed to — and when another row holds that hash, the count
// reaches 0 and the release deletes the blob under it.
//
// Run over both backends and both row kinds, because the rule is transcribed
// once per row kind in the script (`SF`, `ST`) and once, for both kinds, in the
// txn pre-image read: a field-only arm would leave the `ST` transcription
// deletable with the suite green.
//
// ── 2. A torn txn commit ─────────────────────────────────────────────────────
//
// A txn commit queues absolute `obj:ext` counts planned before `MULTI`. When
// its `EXEC` tears — the frames that ran landed, the failing one did not — those
// counts can land against rows another commit has since changed. The torn-EXEC
// arm recounts the object. Staged with the b.bhb mechanism (a competing commit
// takes the sequence, then the `WATCH`'s connection is killed so the queued
// commit runs unwatched and its `XADD` collides), which needs
// `test-fault-injection` for the pre-dispatch hook.

use std::sync::Arc;

use fred::interfaces::HashesInterface;

use live_edit::config::UpdateLayerBackend;
use live_edit::storage::external_values::process_pending_gc;
use live_edit::storage::lua::is_fast_path;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{OwnedDelta, SetField, TimeSample};

use crate::storage::b_qpf_tests::{
    assert_blob_alive, create_session, empty_state, ext_hash, ext_hash_hex, ext_hash_vec,
    ext_ref_released, ext_store, layer_store, obj_ext_sorted, pending_members, pre_store,
    ref_count, ALL_RELEASE_REASONS, EXT_SIZE, FIELD_A, SECTION_ID,
};

/// R2 — the field row that holds the hash throughout, so a release of it is
/// a release under a live reference.
const FIELD_B: &str = "roughness";
/// The time-sample row's time, when R1 is a time sample.
const SAMPLE_TIME: f64 = 2.5;

/// Which value-row kind R1, the dual-shape row, is.
#[derive(Clone, Copy, Debug)]
enum RowKind {
    Field,
    TimeSample,
}

impl RowKind {
    fn label(self) -> &'static str {
        match self {
            RowKind::Field => "field",
            RowKind::TimeSample => "ts",
        }
    }

    /// One commit: R1 inline, and R2 := `hash`. R2's field is always a field
    /// row, so for the time-sample kind the delta mixes both families, which
    /// the fast path accepts.
    fn first_commit(self, hash: Vec<u8>) -> OwnedDelta {
        let r2 = SetField {
            section_id: SECTION_ID,
            key_name: FIELD_B.to_owned(),
            value_or_ext_hash: hash,
            ext_value_size: EXT_SIZE,
            ext_value_index: 1,
            set_order: 1,
        };
        match self {
            RowKind::Field => OwnedDelta {
                set_fields: vec![
                    SetField {
                        section_id: SECTION_ID,
                        key_name: FIELD_A.to_owned(),
                        value_or_ext_hash: vec![0x0A],
                        ext_value_size: 0,
                        ext_value_index: 0,
                        set_order: 0,
                    },
                    r2,
                ],
                ..Default::default()
            },
            RowKind::TimeSample => OwnedDelta {
                set_fields: vec![r2],
                time_samples: vec![TimeSample {
                    section_id: SECTION_ID,
                    time: SAMPLE_TIME,
                    value_or_ext_hash: vec![0x0A],
                    ext_value_size: 0,
                    ext_value_index: 0,
                }],
                ..Default::default()
            },
        }
    }

    /// R1 rewritten to a new inline value.
    fn inline_rewrite(self) -> OwnedDelta {
        match self {
            RowKind::Field => OwnedDelta {
                set_fields: vec![SetField {
                    section_id: SECTION_ID,
                    key_name: FIELD_A.to_owned(),
                    value_or_ext_hash: vec![0x0B],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 2,
                }],
                ..Default::default()
            },
            RowKind::TimeSample => OwnedDelta {
                time_samples: vec![TimeSample {
                    section_id: SECTION_ID,
                    time: SAMPLE_TIME,
                    value_or_ext_hash: vec![0x0B],
                    ext_value_size: 0,
                    ext_value_index: 0,
                }],
                ..Default::default()
            },
        }
    }

    /// R1's row key.
    fn row_key(self, tv: &TestValkey, oid: u64) -> String {
        let oid = oid.to_string();
        match self {
            RowKind::Field => tv.key_schema().obj_field(&oid, "1", FIELD_A),
            RowKind::TimeSample => tv.key_schema().obj_time_sample(&oid, "1", SAMPLE_TIME),
        }
    }
}

fn assert_fast_path_eligible(delta: &OwnedDelta, ctx: &str) {
    assert!(
        is_fast_path(&Some(delta.clone())),
        "{ctx}: this delta must be fast-path ELIGIBLE, or the `Lua` variant silently runs the \
         txn path and the script's transcription of the rule is untested"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Rewriting a dual-shape row releases nothing
// ═════════════════════════════════════════════════════════════════════════════

/// Rewriting a dual-shape row — an inline `value` beside a stale `hash` naming
/// H — releases nothing, so H's blob survives while another row holds it.
///
/// R1 is committed inline and R2 := H in one commit, so `obj:ext` reads H = 1
/// and `ref_count(H)` = 1, both exact. R1 is then forged into the pre-b.zvd
/// shape by merging a `hash` column naming H onto it, as an inline→external
/// rewrite did before the writer cleared both shapes. R1 still decodes as its
/// inline bytes and holds no reference, so the index is still exact.
///
/// The rewrite of R1 must therefore leave H alone. Read raw, R1's old hash is
/// H, the commit plans H 1 → 0 and releases it: the GC tick then deletes H's
/// blob under R2.
async fn rewriting_a_dual_shape_row_releases_nothing(backend: UpdateLayerBackend, kind: RowKind) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-nvf"));
    let ls = layer_store(&tv, backend, &metrics);
    let ext = ext_store(&tv);
    let layer = format!("nvf-dual-shape-{backend:?}-{}", kind.label()).to_lowercase();
    let session_id = create_session(&tv, &layer).await;
    let oid = ls
        .create_layer(
            &session_id,
            &layer,
            "b-nvf-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");
    let h = ext_hash(0xC1);

    // ── Commit 1: R1 inline, R2 := H ────────────────────────────────────────
    let delta = kind.first_commit(ext_hash_vec(0xC1));
    assert_fast_path_eligible(&delta, "the first commit");
    pre_store(&ext, &[h]).await;
    ls.update_layer_relayed(&session_id, &layer, oid, Some(delta), "nvf-writer", 1)
        .await
        .expect("the first commit must land");

    // ── Forge R1 into the dual shape ────────────────────────────────────────
    let r1 = kind.row_key(&tv, oid);
    let size = EXT_SIZE.to_string();
    let _: () = tv
        .client()
        .pool
        .hset(
            &r1,
            [
                ("hash", ext_hash_hex(0xC1).as_str()),
                ("size", size.as_str()),
            ],
        )
        .await
        .expect("forging the dual-shape row must succeed");
    let has_value: bool = tv
        .client()
        .pool
        .hexists(&r1, "value")
        .await
        .expect("HEXISTS must succeed");
    assert!(has_value, "fixture precondition: R1 keeps its inline value");
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0xC1), 1)],
        "fixture precondition: one row — R2 — holds H"
    );
    assert_eq!(ref_count(&tv, h).await, Some(1));

    // ── Commit 2: R1 rewritten inline ───────────────────────────────────────
    let released_before: Vec<u64> = ALL_RELEASE_REASONS
        .iter()
        .map(|r| ext_ref_released(&metrics, r))
        .collect();
    let delta = kind.inline_rewrite();
    assert_fast_path_eligible(&delta, "the inline rewrite of the dual-shape row");
    ls.update_layer_relayed(&session_id, &layer, oid, Some(delta), "nvf-writer", 2)
        .await
        .expect("the rewrite must land");

    assert!(
        pending_members(&tv).await.is_empty(),
        "nothing is owed: R1 held no reference. A `rewrite` member naming H here is R1's \
         stale hash column released as if it were a reference — the over-release"
    );
    let released_after: Vec<u64> = ALL_RELEASE_REASONS
        .iter()
        .map(|r| ext_ref_released(&metrics, r))
        .collect();
    assert_eq!(
        released_after, released_before,
        "no release in any reason ({ALL_RELEASE_REASONS:?})"
    );
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0xC1), 1)],
        "H's count is untouched: R2 still holds it, and R1 never counted toward it"
    );

    process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_blob_alive(&tv, h, "H, still held by R2").await;
    assert_eq!(ref_count(&tv, h).await, Some(1));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_rewriting_a_dual_shape_field_releases_nothing() {
    rewriting_a_dual_shape_row_releases_nothing(UpdateLayerBackend::Lua, RowKind::Field).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_rewriting_a_dual_shape_field_releases_nothing() {
    rewriting_a_dual_shape_row_releases_nothing(UpdateLayerBackend::Txn, RowKind::Field).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_rewriting_a_dual_shape_ts_releases_nothing() {
    rewriting_a_dual_shape_row_releases_nothing(UpdateLayerBackend::Lua, RowKind::TimeSample).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_rewriting_a_dual_shape_ts_releases_nothing() {
    rewriting_a_dual_shape_row_releases_nothing(UpdateLayerBackend::Txn, RowKind::TimeSample).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. A torn txn commit is followed by a recount of its object
// ═════════════════════════════════════════════════════════════════════════════

/// A txn commit whose `EXEC` tears leaves `obj:ext` equal to the rows, because
/// the torn-EXEC arm recounts the object.
///
/// The layer starts with R1 and R2 both holding H (H = 2). The disturbed
/// commit rewrites R1 := K on the txn backend; before `MULTI` it reads H = 2
/// and queues the absolute counts H = 1, K = 1. Inside the pre-dispatch hook a
/// competing commit adds R3 := H at the sequence the disturbed commit computed
/// (H = 3), and the `WATCH`'s connection is killed and replaced. The disturbed
/// pipeline then runs unwatched: its row write and index writes land, and its
/// `XADD` collides with the competitor's entry. The rows now hold H twice (R2,
/// R3) and K once (R1), while the index the tear wrote says H = 1 — an
/// under-count, the dangerous direction.
///
/// `committed_unwatched` ticking once is the evidence the commit took the
/// `ExecError` arm through the stream-id collision, rather than cancelling or
/// committing; without it the index assertion could pass for a scenario that
/// never tore.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_torn_txn_commit_is_followed_by_a_recount_of_its_object() {
    use fred::clients::Client;
    use futures::FutureExt;
    use live_edit::error::ServiceError;
    use live_edit::observability::metrics::test_util::{render, sample_value};
    use live_edit::storage::watched_txn::WatchWindowHook;

    use crate::storage::b_bhb_tests::{
        await_reconnect, guarded_connection, kill_connection, once_only, POOL_SIZE, SCENARIO_BUDGET,
    };
    use crate::storage::b_qpf_tests::watch_voided;

    const LAYER: &str = "nvf-torn";
    const FIELD_C: &str = "metallic";
    const H: u8 = 0xD1;
    const K: u8 = 0xD2;

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-nvf"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, LAYER).await;
    let plain = layer_store(&tv, UpdateLayerBackend::Txn, &metrics);
    let oid = plain
        .create_layer(
            &session_id,
            LAYER,
            "b-nvf-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let field = |key: &str, hash_byte: u8, set_order: u64| OwnedDelta {
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: key.to_owned(),
            value_or_ext_hash: ext_hash_vec(hash_byte),
            ext_value_size: EXT_SIZE,
            ext_value_index: 1,
            set_order,
        }],
        ..Default::default()
    };

    // ── R1 := H and R2 := H, in one commit (seq 2) ──────────────────────────
    let mut both = field(FIELD_A, H, 0);
    both.set_fields.extend(field(FIELD_B, H, 1).set_fields);
    pre_store(&ext, &[ext_hash(H)]).await;
    plain
        .update_layer_relayed(&session_id, LAYER, oid, Some(both), "nvf-writer", 1)
        .await
        .expect("the two-row commit must land");
    assert_eq!(obj_ext_sorted(&tv, oid).await, vec![(ext_hash_hex(H), 2)]);

    // ── The hook: a competitor takes seq 3 adding R3 := H, then the WATCH's
    //    connection is replaced ──────────────────────────────────────────────
    let competitor = Arc::new(layer_store(&tv, UpdateLayerBackend::Lua, &metrics));
    let competitor_ext = ext.clone();
    let competitor_session = session_id.clone();
    let competitor_delta = field(FIELD_C, H, 2);
    let killer = tv.client().pool.next().clone();
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let hook: WatchWindowHook = once_only(Arc::new(move |client: &Client| {
        let competitor = competitor.clone();
        let competitor_ext = competitor_ext.clone();
        let competitor_session = competitor_session.clone();
        let competitor_delta = competitor_delta.clone();
        let killer = killer.clone();
        let header_key = header_key.clone();
        async move {
            pre_store(&competitor_ext, &[ext_hash(H)]).await;
            let (seq, _) = competitor
                .update_layer_relayed(
                    &competitor_session,
                    LAYER,
                    oid,
                    Some(competitor_delta),
                    "nvf-competitor",
                    1_000,
                )
                .await
                .expect("the competing commit is uncontended and must land");
            assert_eq!(
                seq, 3,
                "the competitor must take the sequence the disturbed commit computed, or its \
                 XADD does not collide and nothing tears"
            );
            let (node, conn_id) = guarded_connection(client, &header_key);
            kill_connection(&killer, &node, conn_id).await;
            await_reconnect(client, &node, conn_id).await;
        }
        .boxed()
    }));
    let disturbed =
        layer_store(&tv, UpdateLayerBackend::Txn, &metrics).with_pre_dispatch_hook_for_test(hook);

    // ── The disturbed commit: R1 := K ───────────────────────────────────────
    pre_store(&ext, &[ext_hash(K)]).await;
    let result = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.update_layer_relayed(
            &session_id,
            LAYER,
            oid,
            Some(field(FIELD_A, K, 3)),
            "nvf-disturbed",
            2_000,
        ),
    )
    .await
    .expect("the disturbed commit did not settle within the scenario budget");
    match result {
        Err(ServiceError::Internal(msg)) => assert!(
            msg.contains("EXEC"),
            "a tear is reported as an EXEC failure; got {msg:?}"
        ),
        other => panic!("the disturbed commit must tear, non-retriably; got {other:?}"),
    }
    assert_eq!(
        watch_voided(&metrics, "update_layer", "committed_unwatched"),
        1,
        "the EXEC must have run unwatched and torn on the stream-id collision — the shape \
         this test exists for"
    );

    // ── The rows, read independently of the index ───────────────────────────
    let oid_str = oid.to_string();
    let mut row_hashes = Vec::new();
    for key in [FIELD_A, FIELD_B, FIELD_C] {
        let hash: Option<String> = tv
            .client()
            .pool
            .hget(tv.key_schema().obj_field(&oid_str, "1", key), "hash")
            .await
            .expect("HGET row hash must succeed");
        row_hashes.push(hash);
    }
    assert_eq!(
        row_hashes,
        vec![
            Some(ext_hash_hex(K)),
            Some(ext_hash_hex(H)),
            Some(ext_hash_hex(H))
        ],
        "the tear landed R1 := K; R2 and the competitor's R3 hold H"
    );

    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(H), 2), (ext_hash_hex(K), 1)],
        "the index must equal the rows. H = 1 is the count the torn commit planned before the \
         competitor added R3, left in place: the next rewrite of either H row would release H \
         under the other"
    );
    let cell = |name: &str, label: (&str, &str)| {
        sample_value(&render(&metrics), name, &[("site", "torn_commit"), label])
            .unwrap_or_else(|| panic!("{name} {label:?} is zero-seeded")) as u64
    };
    assert_eq!(
        cell("live_edit_ext_index_recount_total", ("outcome", "repaired")),
        1,
        "the torn-EXEC arm recounted the object once and repaired it"
    );
    assert_eq!(
        cell("live_edit_ext_index_drift_total", ("direction", "under")),
        1,
        "one field corrected upward"
    );
}
