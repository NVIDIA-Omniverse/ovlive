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

//! End-to-end conformance tests for the layer-checkpoint assembler
//! (Task `t2.fgj.ya.78` Subtask `t3.fgj.ya.78.kn`).
//!
//! These tests exercise the full SR-2.3 pipeline against a live Valkey:
//!
//! 1. The state read against real Valkey data: `read_object_snapshot`'s atomic
//!    read for a layer at or below the atomic-read limits, and
//!    `read_object_snapshot_fenced` (the fenced read, with its capped fence
//!    wait and its `FENCE_READ_ATTEMPTS`) for one above them — reached here
//!    through `fenced_path_client`'s forced-low limits.
//! 2. `load_consistent_snapshot`'s pin, its proven cut (the atomic read's own,
//!    or the fenced read's), the leader-lease keeper raced beside a fenced
//!    read (its two stops: a lost lease and a shutdown), and the subordinate
//!    version filter.
//! 3. `build_reset_ompf`'s OODF reset-delta + OMPF wrap, routed through the
//!    real `oodf-ffi` C++ FFI bridge.
//! 4. `OmpfContainer::serialize` wrap.
//! 5. The ext-refs sidecar emission threaded through `CheckpointMeta::sidecar_bytes`.
//!
//! Each test populates a layer via `LayerStore::create_layer` (and where
//! required, `LayerStore::update_layer` for tombstone setup), runs the
//! assembler with an [`InMemoryCheckpointStore`] sink, then reads the OMPF
//! back out of the store, decodes through `OmpfContainer::parse` →
//! `oodf_ffi::parse_delta`, applies it to a fresh `ObjectState` via
//! `apply_delta`, and compares the resulting state against an
//! `ObjectSnapshot` loaded fresh from the same Valkey object.
//!
//! ## Wait budgets vs. nextest's hang-killer
//!
//! `.config/nextest.toml` marks a test slow at 40 s and **hard-terminates it at
//! 120 s** (`terminate-after = 3`), and a SIGKILLed test prints nothing — so
//! each test's worst case must finish inside that window, with room for its
//! failure to be reported. The budgets are the module constants named below,
//! so the arithmetic can be re-derived. Worst case = every wait consuming its
//! whole budget:
//!
//! | Test | Waits | Worst case |
//! |---|---|---|
//! | 5 `assembler_fenced_read_of_a_hot_layer_claims_exactly_the_commits_at_its_cut` | ONE [`HOT_WRITER_TEST_BUDGET`] deadline over the warm-up, all 3 fenced assemblies with their ledger and writer-advance waits, the writer's drain and the quiet assembly | **90 s** |
//! | 5b `assembler_writes_a_hot_layer_at_its_exact_cut_every_time_on_both_proven_reads` | ONE [`HOT_WRITER_TEST_BUDGET`] deadline over the warm-up, all 12 assemblies (6 atomic, 6 fenced) with their ledger and writer-advance waits, and the writer's drain | **90 s** |
//! | 5d `assembler_fenced_read_broken_on_every_attempt_is_transient_and_persists_nothing` | the fenced read's own capped wait¹ + [`FENCE_RELEASE_BUDGET`] 10 s | **≈ 40 s** |
//! | 5d `assembler_fenced_read_refuses_a_fence_wait_past_reader_wait_max_as_transient` | none: the cap refuses the first `RETRY_AFTER` at once | **one assembly** |
//! | 5d `assembler_lease_lost_mid_fenced_read_persists_nothing_and_releases_the_fence`, `assembler_shutdown_mid_fenced_read_persists_nothing_and_releases_the_fence` | `HELD_BACK / 2` 30 s + [`FENCE_RELEASE_BUDGET`] 10 s | **40 s** |
//! | 5e `assembler_fenced_read_takes_no_catch_up_slot_permit_or_memory_budget` | [`UNGATED_ASSEMBLY_BUDGET`] 20 s | **20 s** |
//!
//! Every other test runs single assemblies of a quiet layer and no wait.
//!
//! ¹ `CappedFenceWait`'s clock spans every attempt and its `RETRY_AFTER`
//!   sleeps, capped at `reader_wait_max(L)` — about 29 s at these layers'
//!   sizes, where `intent_ttl` sits at its floor. The seq bumper never makes
//!   the read wait, so a healthy run spends milliseconds of it.
//!
//! The hot-writer tests are the ones a per-wait budget would break: a fenced
//! round costs about 1.1 s healthy (the writer the fence refused sleeps about
//! one poll before it commits again), and per-round budgets summed to minutes.
//! Their one deadline also watches the writer task, so a writer that panics
//! fails the test at once with its own message rather than as a timeout.
//!
//! Budgets are deadlines, not sleeps: the healthy path pays none of them.
//!
//! Sandbox env note: these tests use `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};
use ordered_float::OrderedFloat;
use serde::Deserialize;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::assembler::{
    assemble_layer_checkpoint, AssemblerOutcome, AssemblyDecision, NoLeaseToKeep,
};
use live_edit::checkpoint::CheckpointStore;
use live_edit::domain::delta::apply_delta;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section, TimeSample};
use live_edit::storage::external_values::{ExternalValueStore, ValkeyExternalValueStore};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{OwnedDelta, StructCommand};

use super::helpers::fenced_path_client;

const SERVICE_LABEL: &str = "assembler-test-replica";

/// Sidecar shape the adapter writes — matches `assemble_layer_checkpoint`'s
/// internal struct. Declared locally so the test asserts the exact
/// observable on-the-wire JSON shape rather than re-exporting the
/// `assembler`-private struct.
#[derive(Debug, Deserialize, PartialEq, Eq)]
struct SidecarShape {
    ext_hashes: Vec<String>,
}

/// Unwrap the `Written` arm of an [`AssemblyDecision`].
///
/// Every caller expects a proven cut: the layer's one proven read — atomic at
/// or below the atomic-read limits, fenced above them — proves a cut however
/// hot the layer is, and no header-sequence regression is staged, so the
/// assembler always persists. A `SkippedNotQuiesced` here would itself be the
/// regression, and every caller passes a keeper that never loses its lease.
fn expect_written(decision: AssemblyDecision) -> AssemblerOutcome {
    match decision {
        AssemblyDecision::Written(outcome) => outcome,
        AssemblyDecision::SkippedNotQuiesced => panic!(
            "assembler must prove a cut here (one proven read, no regression staged); \
             got SkippedNotQuiesced"
        ),
        AssemblyDecision::LeaseLost => {
            panic!("assembler was handed a keeper that never loses its lease; got LeaseLost")
        }
        AssemblyDecision::ShuttingDown => {
            panic!("assembler was handed a keeper that never shuts down; got ShuttingDown")
        }
    }
}

/// Helper: build a `LayerStore` with the default 4096 stream maxlen.
fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
}

/// Helper: build a `ValkeyExternalValueStore` over the same fixture.
fn ext_store(tv: &TestValkey) -> Arc<ValkeyExternalValueStore> {
    Arc::new(ValkeyExternalValueStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ))
}

/// Helper: create a fresh session for the layer to live in.
async fn create_session(tv: &TestValkey) -> String {
    create_session_named(tv, "assembler-test-session").await
}

/// Helper: create a fresh session named `name` — for a test that creates more
/// than one, since a session name is unique.
async fn create_session_named(tv: &TestValkey, name: &str) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, "/stage", "assembler-test-user")
        .await
        .expect("session creation must succeed")
}

/// Build a populated `ObjectState` with 3 sections, 2 fields per section, and
/// 4 time samples spread across the sections. Section ids 10, 20, 30 are all
/// children of root (parent 0). Section ordering is canonical (insert order
/// of children matters for `apply_reset`'s reconstruction).
fn populated_state() -> ObjectState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        next_set_order: 0,
        ..Default::default()
    };

    let section_ids = [10_u64, 20, 30];
    for (order, sid) in section_ids.iter().enumerate() {
        state.sections.insert(
            *sid,
            Section {
                section_id: *sid,
                parent_id: 0,
                name: format!("sec-{sid}"),
                section_type: 1,
                section_order: order as u32,
                first_version: 1,
                last_version: 1,
            },
        );
    }
    // Root children list — required so `DiffEngine::sections_in_dfs_order`
    // can walk from root and emit sections in the same order both at
    // assembler time and at re-apply time.
    state.children.insert(0, section_ids.to_vec());

    // 2 inline fields per section.
    for sid in section_ids {
        for fi in 0..2u8 {
            let key = format!("field-{fi}");
            state.fields.insert(
                (sid, key.clone()),
                Field {
                    section_id: sid,
                    key_name: key,
                    value: FieldValue::Inline(vec![sid as u8, fi, 0xAA]),
                    set_order: u64::from(fi),
                    authoring_set_order: u64::from(fi),
                    version: 1,
                },
            );
        }
    }

    // 4 time samples — two on sec 10, one each on 20 and 30.
    let samples: [(u64, f64, u8); 4] = [
        (10, 0.0, 0xC0),
        (10, 1.5, 0xC1),
        (20, 2.25, 0xC2),
        (30, 3.75, 0xC3),
    ];
    for (sid, t, marker) in samples {
        state.time_samples.insert(
            (sid, OrderedFloat(t)),
            TimeSample {
                section_id: sid,
                time: OrderedFloat(t),
                value: FieldValue::Inline(vec![sid as u8, marker]),
                version: 1,
            },
        );
    }

    state
}

/// Decode the OMPF bytes from a successful assembler put: parse the OMPF
/// container, then the inner OODF delta, returning the decoded `OwnedDelta`
/// plus the full sidecar JSON bytes.
async fn fetch_blob(store: &InMemoryCheckpointStore, object_id: u64) -> (OwnedDelta, Bytes, Bytes) {
    let blob = store
        .get_latest_layer_checkpoint(object_id)
        .await
        .expect("checkpoint get must succeed")
        .expect("checkpoint must exist for the assembled object");
    let ompf_bytes = blob.bytes.clone();
    let sidecar_bytes = blob
        .meta
        .sidecar_bytes
        .clone()
        .expect("assembler must thread sidecar bytes through CheckpointMeta");

    let parsed = OmpfContainer::parse(&ompf_bytes).expect("OMPF parse must succeed");
    let delta = oodf_ffi::parse_delta(parsed.delta_bytes)
        .expect("OODF delta parse must succeed (no malformed bytes from assembler)");

    (delta, ompf_bytes, sidecar_bytes)
}

// =============================================================================
// Test 1 — Round-trip sections, fields, time samples, and a tombstone.
// =============================================================================

/// The headline acceptance test: a layer with sections, fields, time samples,
/// and a tombstone is round-trippable through the assembler → OMPF →
/// `oodf-ffi` parse → `apply_delta` pipeline, producing a state isomorphic to
/// the live Valkey state at the time of assembly.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_round_trips_sections_fields_time_samples_and_tombstones() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;

    // Step 1: create_layer with a populated initial state.
    let initial = populated_state();
    let oid = ls
        .create_layer(
            &session_id,
            "round-trip-layer",
            "assembler-test-user",
            &initial,
            ext_store(&tv),
        )
        .await
        .expect("create_layer must succeed");

    // Step 2: issue an update_layer with a DeleteSection so a tombstone
    // lands in `obj:dels` for the assembler to fold into the reset-delta.
    // DeleteSection on the ingest path emits no struct-command (SDD §9.1.4f
    // marks DeleteSection as a structural command — see apply_delete_section)
    // but writes a tombstone hash; on the apply side the reset-delta omits
    // the deleted section entirely (it's no longer in `state.sections`).
    let delete_delta = OwnedDelta {
        struct_commands: vec![StructCommand::DeleteSection(
            oodf_ffi::CommandDeleteSection {
                parent_id: 0,
                section_id: 30,
            },
        )],
        ..Default::default()
    };
    ls.update_layer_relayed(
        &session_id,
        "round-trip-layer",
        oid,
        Some(delete_delta),
        "assembler-test-client",
        100,
    )
    .await
    .expect("DeleteSection update must succeed");

    // Step 3: invoke the assembler with an InMemoryCheckpointStore sink.
    let store = Arc::new(InMemoryCheckpointStore::new());
    let outcome = expect_written(
        assemble_layer_checkpoint(
            tv.client(),
            tv.key_schema(),
            oid,
            store.as_ref(),
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        )
        .await
        .expect("assembler must succeed against a populated live layer"),
    );
    assert_eq!(outcome.handle.object_id, oid);
    // No external values were referenced.
    assert_eq!(outcome.ext_hash_count, 0);

    // Step 4: decode the OMPF + delta. Apply to a fresh ObjectState.
    let (delta, _ompf_bytes, _sidecar_bytes) = fetch_blob(&store, oid).await;
    let mut reconstructed = ObjectState::default();
    apply_delta(&mut reconstructed, Some(delta)).expect("apply_delta on the reset must succeed");

    // Step 5: load a fresh snapshot of the live Valkey state and compare
    // isomorphically. We compare by structural identity rather than full
    // equality because per-entry versions in `reconstructed` are normalized
    // to `new_version = 1` by `apply_delta`'s sequence-bump, while the
    // live snapshot carries the actual `last_version` / `version` values.
    let live = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test must succeed");

    // Section set isomorphism: same set of section_ids, same (parent_id,
    // name, type) per id. After the DeleteSection on 30 the live snapshot's
    // sections HashMap must NOT contain 30, and the reconstructed state
    // must not either.
    let live_sec_ids: HashSet<u64> = live.sections.keys().copied().collect();
    let recon_sec_ids: HashSet<u64> = reconstructed.sections.keys().copied().collect();
    assert_eq!(
        recon_sec_ids, live_sec_ids,
        "reconstructed sections must match live; live={live_sec_ids:?} recon={recon_sec_ids:?}"
    );
    assert!(
        !recon_sec_ids.contains(&30),
        "section 30 was DeleteSection'd; must NOT appear in reconstructed state"
    );
    for sid in &live_sec_ids {
        let l = &live.sections[sid];
        let r = &reconstructed.sections[sid];
        assert_eq!(r.section_id, l.section_id);
        assert_eq!(r.parent_id, l.parent_id, "parent_id mismatch on sec {sid}");
        assert_eq!(r.name, l.name, "name mismatch on sec {sid}");
        assert_eq!(
            r.section_type, l.section_type,
            "section_type mismatch on sec {sid}"
        );
    }

    // Field set isomorphism: same (section_id, key_name) pairs and identical
    // FieldValue payloads.
    let live_field_map: HashMap<(u64, String), FieldValue> = live
        .fields
        .iter()
        .map(|f| ((f.section_id, f.key_name.clone()), f.value.clone()))
        .collect();
    let recon_field_map: HashMap<(u64, String), FieldValue> = reconstructed
        .fields
        .iter()
        .map(|((sid, k), f)| ((*sid, k.clone()), f.value.clone()))
        .collect();
    assert_eq!(
        recon_field_map, live_field_map,
        "reconstructed fields must match live (set + payloads)"
    );

    // Time sample isomorphism: same (section_id, time) keys and identical
    // FieldValue payloads.
    let live_ts_map: HashMap<(u64, OrderedFloat<f64>), FieldValue> = live
        .time_samples
        .iter()
        .map(|ts| ((ts.section_id, ts.time), ts.value.clone()))
        .collect();
    let recon_ts_map: HashMap<(u64, OrderedFloat<f64>), FieldValue> = reconstructed
        .time_samples
        .iter()
        .map(|((sid, t), ts)| ((*sid, *t), ts.value.clone()))
        .collect();
    assert_eq!(
        recon_ts_map, live_ts_map,
        "reconstructed time samples must match live (keys + payloads)"
    );

    // source_format must round-trip via the reset path (SDD §9.1.3c).
    assert_eq!(reconstructed.source_format, live.source_format);
}

// =============================================================================
// Test 2 — Ext-refs sidecar carries exactly the referenced hashes.
// =============================================================================

/// A layer carrying two ext-value-referenced fields must produce a sidecar
/// JSON whose `ext_hashes` array is exactly the set of hashes the OMPF
/// references — no extras, no omissions. Hash-case drift, schema drift, or
/// missing/extra entries all fail the test.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_emits_ext_refs_sidecar_with_referenced_hashes() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;
    let evs = ext_store(&tv);

    // Step 1: pre-store two distinct ext-value blobs in Valkey. These will
    // be referenced by `FieldValue::External` entries on the layer.
    let payload_a = b"alpha-payload";
    let payload_b = b"beta-payload-with-distinct-bytes";
    let hash_a = sha256(payload_a);
    let hash_b = sha256(payload_b);
    let hash_a_hex = sha256_to_hex(&hash_a);
    let hash_b_hex = sha256_to_hex(&hash_b);
    evs.store_values(&[
        (&hash_a, payload_a.as_slice()),
        (&hash_b, payload_b.as_slice()),
    ])
    .await
    .expect("ext store_values must succeed");

    // Step 2: build a state where fields reference those hashes by content
    // address. Layout: one section (sid=10) carrying two External fields.
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        next_set_order: 0,
        ..Default::default()
    };
    state.sections.insert(
        10,
        Section {
            section_id: 10,
            parent_id: 0,
            name: "with-ext".to_string(),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        },
    );
    state.children.insert(0, vec![10]);
    state.fields.insert(
        (10, "ext-a".to_string()),
        Field {
            section_id: 10,
            key_name: "ext-a".to_string(),
            value: FieldValue::External {
                hash: hash_a,
                size: payload_a.len() as u64,
                ext_value_index: 0,
            },
            set_order: 0,
            authoring_set_order: 0,
            version: 1,
        },
    );
    state.fields.insert(
        (10, "ext-b".to_string()),
        Field {
            section_id: 10,
            key_name: "ext-b".to_string(),
            value: FieldValue::External {
                hash: hash_b,
                size: payload_b.len() as u64,
                ext_value_index: 0,
            },
            set_order: 1,
            authoring_set_order: 1,
            version: 1,
        },
    );

    let oid = ls
        .create_layer(
            &session_id,
            "ext-refs-layer",
            "assembler-test-user",
            &state,
            evs.clone() as Arc<dyn ExternalValueStore>,
        )
        .await
        .expect("create_layer with ext-referenced fields must succeed");

    // Step 3: run the assembler.
    let store = Arc::new(InMemoryCheckpointStore::new());
    let outcome = expect_written(
        assemble_layer_checkpoint(
            tv.client(),
            tv.key_schema(),
            oid,
            store.as_ref(),
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        )
        .await
        .expect("assembler must succeed with ext-referenced fields"),
    );
    assert_eq!(outcome.ext_hash_count, 2);

    // Step 4: decode the OMPF. Collect every ext_value reference in the
    // decoded delta — the sidecar must enumerate exactly this set.
    let (delta, _ompf, sidecar_bytes) = fetch_blob(&store, oid).await;
    let referenced_in_ompf: HashSet<String> = delta
        .set_fields
        .iter()
        .filter(|sf| sf.ext_value_size > 0)
        .map(|sf| {
            // Hash bytes appear as `value_or_ext_hash` for the External
            // discriminant (see `encode_field_value` in storage::diff_read).
            assert_eq!(
                sf.value_or_ext_hash.len(),
                32,
                "External field must carry a 32-byte hash in value_or_ext_hash"
            );
            let mut buf = [0u8; 32];
            buf.copy_from_slice(&sf.value_or_ext_hash);
            sha256_to_hex(&buf)
        })
        .collect();
    let expected_in_ompf: HashSet<String> = [hash_a_hex.clone(), hash_b_hex.clone()]
        .into_iter()
        .collect();
    assert_eq!(
        referenced_in_ompf, expected_in_ompf,
        "OMPF-encoded fields must reference exactly the two stored hashes"
    );

    // Step 5: parse the sidecar JSON. Must be exactly the `{ "ext_hashes":
    // [...] }` shape with the same set of hashes (lowercase hex, no extras,
    // no omissions). Hash-case drift, extra fields, or omissions all surface
    // as a parse / set-equality failure here.
    let sidecar: SidecarShape = serde_json::from_slice(&sidecar_bytes)
        .expect("sidecar must parse as the documented JSON shape");
    let sidecar_set: HashSet<String> = sidecar.ext_hashes.into_iter().collect();
    assert_eq!(
        sidecar_set, expected_in_ompf,
        "sidecar ext_hashes must equal the set of hashes referenced by the OMPF"
    );

    // And as a stricter shape check, parsing as a raw `serde_json::Value`
    // must reveal exactly one top-level field named `ext_hashes`. An
    // accidental extra field (e.g. a schema version) would fail this check
    // and force a deliberate spec update before landing.
    let raw: serde_json::Value =
        serde_json::from_slice(&sidecar_bytes).expect("sidecar must be valid JSON");
    let obj = raw
        .as_object()
        .expect("sidecar JSON must be a top-level object");
    assert_eq!(
        obj.len(),
        1,
        "sidecar JSON must carry exactly one top-level field; got {:?}",
        obj.keys().collect::<Vec<_>>()
    );
    assert!(
        obj.contains_key("ext_hashes"),
        "sidecar JSON must have an `ext_hashes` field; got {:?}",
        obj.keys().collect::<Vec<_>>()
    );

    // Every hash string must be lowercase 64-char hex — the contract pinned
    // by SDD §11 and `sha256_to_hex`. A future drift to uppercase or a
    // different encoding would break the orphan sweep's reverse-index lookup.
    for h in obj["ext_hashes"]
        .as_array()
        .expect("ext_hashes must be a JSON array")
    {
        let s = h.as_str().expect("ext_hashes entries must be strings");
        assert_eq!(s.len(), 64, "ext_hash {s:?} must be 64 hex characters");
        assert!(
            s.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')),
            "ext_hash {s:?} must be lowercase hex only"
        );
    }
}

// =============================================================================
// Test 3 — Version filter drops entries with version > snap_seq.
// =============================================================================

/// The assembler's SR-2.3 step 4 filter must drop any section whose
/// `last_ver` is strictly greater than the post-fan-out `snap_seq`. We
/// manually inject a section with `last_ver = 15` via raw HSET into a layer
/// whose header `seq = 10`; the resulting OMPF MUST NOT carry that section.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_skips_entries_with_version_above_snap_seq() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;

    // Step 1: create the layer with a known seq=10 by including one section
    // in the initial state and forcing seq via the ObjectState.
    let mut state = ObjectState {
        sequence_num: 10,
        first_version: 1,
        source_format: "usd".to_string(),
        next_set_order: 0,
        ..Default::default()
    };
    state.sections.insert(
        1,
        Section {
            section_id: 1,
            parent_id: 0,
            name: "kept-sec".to_string(),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 5,
        },
    );
    state.children.insert(0, vec![1]);

    let oid = ls
        .create_layer(
            &session_id,
            "version-filter-layer",
            "assembler-test-user",
            &state,
            ext_store(&tv),
        )
        .await
        .expect("create_layer must succeed");

    // Sanity: the live header carries seq=10 (create_layer's write_object
    // persisted ObjectState::sequence_num verbatim per layer.rs).
    let header_seq: String = tv
        .client()
        .pool
        .hget(tv.key_schema().obj_header(&oid.to_string()), "seq")
        .await
        .expect("HGET obj:hdr seq must succeed");
    assert_eq!(header_seq, "10", "test precondition: snap_seq must be 10");

    // Step 2: manually inject a section with last_ver=15. The injected
    // section's section hash is added via HSET (mirroring the writer's
    // schema) and its id is SADD'd into obj_sections_index so
    // read_object_state_for_test picks it up.
    let oid_str = oid.to_string();
    let future_section_id = "99";
    let future_section_key = tv.key_schema().obj_section(&oid_str, future_section_id);
    let _: () = tv
        .client()
        .pool
        .hset(
            &future_section_key,
            [
                ("parent_id", "0"),
                ("name", "future-sec"),
                ("type", "1"),
                ("order", "0"),
                ("first_ver", "12"),
                ("last_ver", "15"),
            ],
        )
        .await
        .expect("HSET future-section hash must succeed");
    let _: () = tv
        .client()
        .pool
        .sadd(
            tv.key_schema().obj_sections_index(&oid_str),
            future_section_id,
        )
        .await
        .expect("SADD future-section id must succeed");

    // Sanity-check the injection: read_object_state_for_test must observe BOTH
    // sections before the filter runs.
    let pre_filter = read_object_state_for_test(tv.client(), tv.key_schema(), &oid_str)
        .await
        .expect("read_object_state_for_test must succeed");
    assert!(
        pre_filter.sections.contains_key(&99),
        "raw snapshot must contain the injected future section; got: {:?}",
        pre_filter.sections.keys().collect::<Vec<_>>()
    );

    // Step 3: run the assembler. snap_seq read post-fan-out will be 10
    // (header is untouched between fan-out and re-read), so the section
    // with last_ver=15 must be filtered out.
    let store = Arc::new(InMemoryCheckpointStore::new());
    let outcome = expect_written(
        assemble_layer_checkpoint(
            tv.client(),
            tv.key_schema(),
            oid,
            store.as_ref(),
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        )
        .await
        .expect("assembler must succeed"),
    );
    assert_eq!(outcome.snap_seq, 10);

    // Step 4: decode the OMPF. The section with last_ver=15 must NOT
    // appear in the reset-delta's struct commands; section 1 (last_ver=5)
    // MUST appear.
    let (delta, _ompf, _sidecar) = fetch_blob(&store, oid).await;
    let encoded_section_ids: HashSet<u64> = delta
        .struct_commands
        .iter()
        .filter_map(|cmd| match cmd {
            StructCommand::DiffChangeSection(c) => Some(c.section_id),
            _ => None,
        })
        .collect();
    assert!(
        encoded_section_ids.contains(&1),
        "section 1 (last_ver=5 ≤ snap_seq=10) must appear in the reset-delta; \
         encoded_section_ids={encoded_section_ids:?}"
    );
    assert!(
        !encoded_section_ids.contains(&99),
        "section 99 (last_ver=15 > snap_seq=10) MUST be filtered; \
         encoded_section_ids={encoded_section_ids:?}"
    );
}

// =============================================================================
// Test 4 — Reset-delta shape is the locked cross-Task contract.
// =============================================================================

/// The cross-Task contract with the apply Task (`t2.fgj.ya.n9`) requires
/// the assembler to emit a reset-delta: `is_diff == true && base_version == 0`.
/// This shape pins the case-1 (FullContent) branch in `DiffEngine::plan`
/// and the reset path in `apply_delta` (SDD §9.1.3).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_emits_reset_delta_shape_is_diff_true_base_version_zero() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;

    let state = populated_state();
    let oid = ls
        .create_layer(
            &session_id,
            "reset-shape-layer",
            "assembler-test-user",
            &state,
            ext_store(&tv),
        )
        .await
        .expect("create_layer must succeed");

    let store = Arc::new(InMemoryCheckpointStore::new());
    expect_written(
        assemble_layer_checkpoint(
            tv.client(),
            tv.key_schema(),
            oid,
            store.as_ref(),
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        )
        .await
        .expect("assembler must succeed"),
    );

    let (delta, _ompf, _sidecar) = fetch_blob(&store, oid).await;
    assert!(
        delta.is_diff,
        "checkpoint reset-delta MUST set is_diff=true (locked cross-Task contract)"
    );
    assert_eq!(
        delta.base_version, 0,
        "checkpoint reset-delta MUST set base_version=0 (locked cross-Task contract)"
    );
}

// =============================================================================
// Test 5 — b.pw6: a commit landing mid-read is never silently omitted from a
//          blob whose `snap_seq` claims to cover it — above the atomic-read
//          limits, through the fenced read.
// =============================================================================

/// Sections seeded before the race starts. The number is the point: the fenced
/// read's fan-out is a multi-round-trip pipeline (indices, section hashes,
/// children and field/time-sample indices, entity hashes), and a wide object
/// stretches it wide enough that a writer committing in a tight loop reliably
/// tries to land *inside* it — which is exactly the window the b.pw6 hole lived
/// in, and the window the snapshot fence now holds writers out of.
const RACE_SEED_SECTIONS: u64 = 120;

/// First section id the hot writer allocates. Kept clear of the seeded range so
/// "did this section come from a racing commit?" is answerable by id alone.
const RACE_HOT_SECTION_BASE: u64 = 10_000;

/// Assemblies the fenced race runs while the writer hammers. Each one is a
/// fenced read the writer's commits queue behind, so a handful is enough for
/// several of them to straddle a commit attempt.
const RACE_ASSEMBLIES: usize = 3;

/// The ONE deadline a hot-writer test (5, 5b) holds from the moment its
/// writer starts to the moment it is drained: every warm-up, assembly, ledger
/// catch-up and writer-advance wait in the test runs under what is left of it
/// (module header, "Wait budgets"). The healthy path costs about 1.1 s per
/// fenced round — a writer the fence refused sleeps about one poll before it
/// commits again — so each test spends under 10 s on a healthy runner; 90 s
/// leaves a loaded one several times that and still reports inside nextest's
/// 120 s kill.
const HOT_WRITER_TEST_BUDGET: std::time::Duration = std::time::Duration::from_secs(90);

/// Run one step of a hot-writer test under the test's one `deadline`
/// ([`HOT_WRITER_TEST_BUDGET`]), failing with `what` when the budget runs out.
async fn within_hot_writer_budget<F: std::future::Future>(
    deadline: tokio::time::Instant,
    what: &str,
    step: F,
) -> F::Output {
    tokio::time::timeout_at(deadline, step)
        .await
        .unwrap_or_else(|_| {
            panic!(
                "FIXTURE TIMING, not an assembler verdict: the test's one \
                 {HOT_WRITER_TEST_BUDGET:?} budget ran out during {what}"
            )
        })
}

/// What ended a hot writer's task early: its panic message, when it panicked.
fn hot_writer_ending(ended: Result<(), tokio::task::JoinError>) -> String {
    match ended {
        Ok(()) => "it returned".to_string(),
        Err(e) if e.is_panic() => {
            let payload = e.into_panic();
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("<non-string panic payload>");
            format!("it panicked: {message}")
        }
        Err(e) => format!("it was cancelled: {e}"),
    }
}

/// Poll until `ready()` holds, under the test's one `deadline`
/// ([`HOT_WRITER_TEST_BUDGET`]). Every wait on the hot writer goes through
/// here, so a writer task that ended early — its `update_layer` panicked, say —
/// fails the test at once with the writer's own message, rather than as a
/// timeout that never reaches the `writer.await` which would show it.
async fn await_hot_writer(
    deadline: tokio::time::Instant,
    writer: &mut tokio::task::JoinHandle<()>,
    what: &str,
    mut ready: impl FnMut() -> bool,
) {
    loop {
        if ready() {
            return;
        }
        if writer.is_finished() {
            let ended = hot_writer_ending(writer.await);
            panic!("the hot writer task ended before {what}: {ended}");
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "FIXTURE TIMING, not an assembler verdict: the test's one \
             {HOT_WRITER_TEST_BUDGET:?} budget ran out waiting for {what}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
}

/// Stop-and-join for a hot writer whose `stop` flag is already set, under the
/// test's one `deadline`.
async fn drain_hot_writer(deadline: tokio::time::Instant, writer: tokio::task::JoinHandle<()>) {
    let ended = within_hot_writer_budget(deadline, "the hot writer's drain", writer).await;
    if ended.is_err() {
        panic!(
            "the hot writer task must end cleanly once stopped, but {}",
            hot_writer_ending(ended)
        );
    }
}

/// Section ids carried by the checkpoint blob, read back out of the persisted
/// reset-delta. The assembler emits live sections as `DiffChangeSection`
/// commands (see Test 3).
async fn blob_section_ids(store: &InMemoryCheckpointStore, object_id: u64) -> HashSet<u64> {
    let (delta, _ompf, _sidecar) = fetch_blob(store, object_id).await;
    delta
        .struct_commands
        .iter()
        .filter_map(|cmd| match cmd {
            StructCommand::DiffChangeSection(c) => Some(c.section_id),
            _ => None,
        })
        .collect()
}

/// The coverage oracle. A checkpoint stamped `snap_seq` asserts to the recovery
/// path that it carries **every** commit at or below `snap_seq` and none above
/// it. `commits` is the writer's own ledger of `(sequence_num, section_id)`
/// pairs, so this checks both halves of that claim directly against ground
/// truth: the racing sections the blob carries are exactly the ledger's at or
/// below the cut.
fn assert_blob_is_exactly_the_commits_at_or_below(
    blob_ids: &HashSet<u64>,
    snap_seq: u64,
    commits: &[(u64, u64)],
) {
    let expected: HashSet<u64> = commits
        .iter()
        .filter(|(seq, _)| *seq <= snap_seq)
        .map(|(_, section_id)| *section_id)
        .collect();
    let carried: HashSet<u64> = blob_ids
        .iter()
        .copied()
        .filter(|id| *id >= RACE_HOT_SECTION_BASE)
        .collect();
    let omitted: Vec<&u64> = expected.difference(&carried).collect();
    let leaked: Vec<&u64> = carried.difference(&expected).collect();
    assert!(
        omitted.is_empty() && leaked.is_empty(),
        "checkpoint stamped snap_seq={snap_seq} must carry exactly the racing sections created \
         at or below it: it omits {omitted:?} (claimed coverage of commits it does not carry — \
         b.pw6) and carries {leaked:?} from above its cut. Ledger has {} commits; blob carries \
         {} sections.",
        commits.len(),
        blob_ids.len(),
    );
}

/// **The b.pw6 end-to-end regression, driven by REAL concurrent commits, on
/// the fenced read.**
///
/// A writer commits `CreateSection` deltas against a layer above the
/// atomic-read limits ([`fenced_path_client`]) in a tight loop while the
/// assembler reads it, recording every `(sequence_num, section_id)` it lands.
/// Every assembly must be WRITTEN — the fenced read proves its cut however hot
/// the layer is, so a skip here is the regression — and its blob must carry
/// exactly the commits at or below its `snap_seq`: none omitted (the b.pw6
/// hole, where a commit landing mid-fan-out passed the `version <= snap_seq`
/// filter while its section was never read) and none from above the cut.
///
/// Then the writer stops and one more assembly must cover the entire ledger.
///
/// The pre-fix shape cannot be *run* here (it read `snap_seq` after an
/// unfenced fan-out and had no skip arm to report), so the discrimination this
/// test carries is against a read that does not prove its cut: an assembly
/// whose state and `snap_seq` came from different moments fails the
/// exact-coverage check.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn assembler_fenced_read_of_a_hot_layer_claims_exactly_the_commits_at_its_cut() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;
    let layer_name = "fenced-race-layer";

    // Step 1: seed a wide layer so one fan-out spans a meaningful window.
    let (state, seed_ids) = wide_race_state();

    let oid = ls
        .create_layer(
            &session_id,
            layer_name,
            "assembler-test-user",
            &state,
            ext_store(&tv),
        )
        .await
        .expect("create_layer must succeed");

    // Step 2: start the hot writer. Every commit creates exactly one new
    // section and records the sequence number the commit landed at, so the
    // ledger is ground truth for "what a cut at seq N must contain".
    let stop = Arc::new(AtomicBool::new(false));
    let commits: Arc<Mutex<Vec<(u64, u64)>>> = Arc::new(Mutex::new(Vec::new()));
    let mut writer = {
        let writer_store = layer_store(&tv);
        let stop = stop.clone();
        let commits = commits.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move {
            let mut next_section_id = RACE_HOT_SECTION_BASE;
            while !stop.load(Ordering::Relaxed) {
                let delta = OwnedDelta {
                    struct_commands: vec![StructCommand::CreateSection(
                        oodf_ffi::CommandCreateSection {
                            parent_id: 0,
                            section_id: next_section_id,
                            section_name: format!("hot-{next_section_id}"),
                            section_type: 1,
                        },
                    )],
                    ..Default::default()
                };
                let (seq, _ts) = writer_store
                    .update_layer_relayed(
                        &session_id,
                        layer_name,
                        oid,
                        Some(delta),
                        "hot-writer",
                        0,
                    )
                    .await
                    .expect("hot-writer update_layer must succeed");
                commits.lock().unwrap().push((seq, next_section_id));
                next_section_id += 1;
            }
        })
    };

    // One deadline for everything from here to the writer's drain (module
    // header, "Wait budgets").
    let deadline = tokio::time::Instant::now() + HOT_WRITER_TEST_BUDGET;

    // Let the writer get going so the first read meets a layer under pressure.
    // Bounded poll rather than a fixed sleep.
    await_hot_writer(
        deadline,
        &mut writer,
        "the hot writer's first 5 commits, without which this test cannot exercise a \
         mid-read race",
        || commits.lock().unwrap().len() >= 5,
    )
    .await;

    // Step 3: assemble WHILE the writer hammers — above the atomic-read limits,
    // so the assembler reads through its fenced read.
    let fenced = fenced_path_client().await;
    let mut cuts: Vec<u64> = Vec::with_capacity(RACE_ASSEMBLIES);
    for round in 0..RACE_ASSEMBLIES {
        let store = Arc::new(InMemoryCheckpointStore::new());
        let outcome = match within_hot_writer_budget(
            deadline,
            &format!("fenced assembly {round}"),
            assemble_layer_checkpoint(
                &fenced,
                tv.key_schema(),
                oid,
                store.as_ref(),
                SERVICE_LABEL,
                &mut NoLeaseToKeep,
            ),
        )
        .await
        .expect("assembler must not error against a healthy backend")
        {
            AssemblyDecision::Written(outcome) => outcome,
            other => panic!(
                "assembly {round} of a hot layer above the atomic-read limits must be written: \
                 the fenced read proves its cut however hot the layer is; got {other:?}"
            ),
        };
        // The writer records a commit only after `update_layer` returns, so the
        // ledger can trail the store by one commit; wait for it, under the
        // test's one deadline.
        await_hot_writer(
            deadline,
            &mut writer,
            &format!(
                "the writer's ledger to reach assembly {round}'s snap_seq {}",
                outcome.snap_seq
            ),
            || {
                commits
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|(seq, _)| *seq >= outcome.snap_seq)
            },
        )
        .await;
        let blob_ids = blob_section_ids(&store, oid).await;
        let ledger = commits.lock().unwrap().clone();
        assert_blob_is_exactly_the_commits_at_or_below(&blob_ids, outcome.snap_seq, &ledger);
        cuts.push(outcome.snap_seq);

        // Let the writer land a commit above this cut before the next read, so
        // successive reads meet a layer that moved between them. A writer the
        // fence refused sleeps about one poll before it retries, so back-to-back
        // fenced reads would otherwise all meet the same cut.
        await_hot_writer(
            deadline,
            &mut writer,
            &format!("a commit above cut {}", outcome.snap_seq),
            || {
                commits
                    .lock()
                    .unwrap()
                    .iter()
                    .any(|(seq, _)| *seq > outcome.snap_seq)
            },
        )
        .await;
    }

    // Step 4: stop the writer and drain it, then assemble against the quiet
    // layer: the blob must carry the whole ledger.
    stop.store(true, Ordering::Relaxed);
    drain_hot_writer(deadline, writer).await;

    let ledger = commits.lock().unwrap().clone();
    assert!(
        cuts.last() > cuts.first(),
        "the writer must have committed between the assemblies for this test to mean \
         anything — the reads met a layer moving under them; cuts {cuts:?}, {} commits",
        ledger.len()
    );

    let quiet_store = Arc::new(InMemoryCheckpointStore::new());
    let quiet = expect_written(
        within_hot_writer_budget(
            deadline,
            "the quiet assembly",
            assemble_layer_checkpoint(
                &fenced,
                tv.key_schema(),
                oid,
                quiet_store.as_ref(),
                SERVICE_LABEL,
                &mut NoLeaseToKeep,
            ),
        )
        .await
        .expect("assembler must succeed against the now-quiet layer"),
    );
    let last_commit_seq = ledger.last().expect("ledger is non-empty").0;
    assert!(
        quiet.snap_seq >= last_commit_seq,
        "a cut proven after the writer stopped must sit at or past the final \
         commit; snap_seq={} last_commit_seq={last_commit_seq}",
        quiet.snap_seq
    );
    let quiet_ids = blob_section_ids(&quiet_store, oid).await;
    assert_blob_is_exactly_the_commits_at_or_below(&quiet_ids, quiet.snap_seq, &ledger);
    for sid in &seed_ids {
        assert!(
            quiet_ids.contains(sid),
            "the quiescent checkpoint must also carry every seeded section; \
             missing {sid}"
        );
    }
}

// =============================================================================
// Test 5b — Spec SDD SR-4.1, SR-11.5: a hot layer is written on every
//           assembly, in one proven read, at its exact cut — at the shipped
//           limits (the atomic read) and above them (the fenced read).
// =============================================================================

/// Assemblies the hot-layer test runs back to back on each path. Six is enough
/// that a read which could not prove its cut against the writer would have no
/// realistic chance of writing all of them, and few enough that the healthy run
/// stays well inside the 40 s slow mark (module header, "Wait budgets").
const HOT_LAYER_ASSEMBLIES: usize = 6;

/// Commits the hot writer must land before the first assembly, so the first
/// read meets a layer already under pressure.
const HOT_WARM_UP_COMMITS: usize = 20;

/// The field every hot-writer commit sets on seeded section 1.
const HOT_FIELD: &str = "hot-field";

/// The section the hot writer writes to: the first of the seeded ones.
const HOT_SECTION: u64 = 1;

/// The 120-section layer [`RACE_SEED_SECTIONS`] describes, at sequence 1.
fn wide_race_state() -> (ObjectState, Vec<u64>) {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        next_set_order: 0,
        ..Default::default()
    };
    let seed_ids: Vec<u64> = (1..=RACE_SEED_SECTIONS).collect();
    for (order, sid) in seed_ids.iter().enumerate() {
        state.sections.insert(
            *sid,
            Section {
                section_id: *sid,
                parent_id: 0,
                name: format!("seed-{sid}"),
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
    state.children.insert(0, seed_ids.clone());
    (state, seed_ids)
}

/// **The single-read contract at the assembler (Spec SDD SR-4.1, SR-11.5).** A
/// layer under sustained commit pressure is WRITTEN on every assembly — never
/// `SkippedNotQuiesced` — in one proven read, at the shipped limits through
/// the atomic read and above them through the fenced read, and each blob is
/// exactly the state at its `snap_seq`.
///
/// The writer sets one inline field to a fresh nonce per commit and records
/// `seq → nonce`, so the ledger says what the field held at every sequence.
/// The blob's value of that field must equal the ledger's at `snap_seq`: a
/// commit above the cut that leaked in, or one at or below it that was missed,
/// would both show up as the wrong nonce. That is each proven read's whole
/// claim, end to end through the OMPF the adapter persisted.
///
/// The writer rides `update_layer`'s Lua fast path (a field-only delta), which
/// commits roughly an order of magnitude faster than a structural one — the
/// pressure under which the retired quiescence loop lost its proof on
/// essentially every attempt.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn assembler_writes_a_hot_layer_at_its_exact_cut_every_time_on_both_proven_reads() {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;
    let layer_name = "hot-on-both-paths-layer";
    let (state, _seed_ids) = wide_race_state();
    let oid = ls
        .create_layer(
            &session_id,
            layer_name,
            "assembler-test-user",
            &state,
            ext_store(&tv),
        )
        .await
        .expect("create_layer must succeed");

    let stop = Arc::new(AtomicBool::new(false));
    let ledger: Arc<Mutex<BTreeMap<u64, u64>>> = Arc::new(Mutex::new(BTreeMap::new()));
    let mut writer = {
        let writer_store = layer_store(&tv);
        let stop = stop.clone();
        let ledger = ledger.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move {
            let mut nonce = 10_000u64;
            while !stop.load(Ordering::Relaxed) {
                let delta = OwnedDelta {
                    set_fields: vec![oodf_ffi::SetField {
                        section_id: HOT_SECTION,
                        key_name: HOT_FIELD.to_string(),
                        value_or_ext_hash: nonce.to_le_bytes().to_vec(),
                        ext_value_size: 0,
                        ext_value_index: 0,
                        set_order: nonce,
                    }],
                    ..Default::default()
                };
                let (seq, _ts) = writer_store
                    .update_layer_relayed(
                        &session_id,
                        layer_name,
                        oid,
                        Some(delta),
                        "hot-writer",
                        0,
                    )
                    .await
                    .expect("hot-writer update_layer must succeed");
                ledger.lock().unwrap().insert(seq, nonce);
                nonce += 1;
            }
        })
    };

    // One deadline for everything from here to the writer's drain (module
    // header, "Wait budgets").
    let deadline = tokio::time::Instant::now() + HOT_WRITER_TEST_BUDGET;
    await_hot_writer(
        deadline,
        &mut writer,
        &format!(
            "the hot writer's first {HOT_WARM_UP_COMMITS} commits, without which the layer is \
             not under pressure"
        ),
        || ledger.lock().unwrap().len() >= HOT_WARM_UP_COMMITS,
    )
    .await;

    let fenced = fenced_path_client().await;
    for (path, client) in [("atomic", tv.client()), ("fenced", &fenced)] {
        let commits_before = ledger.lock().unwrap().len();
        let mut cuts: Vec<u64> = Vec::with_capacity(HOT_LAYER_ASSEMBLIES);
        for round in 0..HOT_LAYER_ASSEMBLIES {
            let store = InMemoryCheckpointStore::new();
            let decision = within_hot_writer_budget(
                deadline,
                &format!("{path} assembly {round}"),
                assemble_layer_checkpoint(
                    client,
                    tv.key_schema(),
                    oid,
                    &store,
                    SERVICE_LABEL,
                    &mut NoLeaseToKeep,
                ),
            )
            .await
            .expect("assembler must not error against a healthy backend");
            let outcome = match decision {
                AssemblyDecision::Written(outcome) => outcome,
                other => panic!(
                    "{path} assembly {round} of a hot layer was not written: its one proven read \
                     proves a cut however hot the layer is, so only a header-sequence \
                     regression may skip it (Spec SDD SR-4.1, SR-11.5); got {other:?}"
                ),
            };

            let (delta, _ompf, _sidecar) = fetch_blob(&store, oid).await;
            let persisted: Vec<&Vec<u8>> = delta
                .set_fields
                .iter()
                .filter(|f| f.section_id == HOT_SECTION && f.key_name == HOT_FIELD)
                .map(|f| &f.value_or_ext_hash)
                .collect();
            // The writer records `seq → nonce` only after `update_layer`
            // returns, so a proven read can see the commit at `snap_seq` before
            // the ledger does. The ledger is at most that one commit behind (the
            // writer starts its next commit only after recording the last), so
            // wait for it, under the test's one deadline, rather than read it
            // once and blame the assembler for the gap. A snap_seq the ledger
            // never reaches names a commit this writer did not make (it is the
            // layer's only committer after the warm-up).
            await_hot_writer(
                deadline,
                &mut writer,
                &format!(
                    "{path} assembly {round}'s snap_seq {} to appear in the writer's ledger",
                    outcome.snap_seq
                ),
                || ledger.lock().unwrap().contains_key(&outcome.snap_seq),
            )
            .await;
            let ledger_now = ledger.lock().unwrap().clone();
            let expected_nonce = ledger_now[&outcome.snap_seq];
            assert_eq!(
                persisted,
                vec![&expected_nonce.to_le_bytes().to_vec()],
                "{path} assembly {round}: the blob stamped snap_seq={} must carry exactly the \
                 {HOT_FIELD} value the commit AT that sequence wrote (nonce {expected_nonce}) — \
                 any other value means the persisted state is not the state at its claimed cut",
                outcome.snap_seq
            );
            cuts.push(outcome.snap_seq);

            // Let the writer land a commit above this cut before the next
            // read: a writer the fence refused sleeps about one poll before it
            // retries, so back-to-back fenced reads would otherwise all meet the
            // same cut. Instant on the atomic path.
            await_hot_writer(
                deadline,
                &mut writer,
                &format!("{path}: a commit above cut {}", outcome.snap_seq),
                || {
                    ledger
                        .lock()
                        .unwrap()
                        .keys()
                        .next_back()
                        .is_some_and(|top| *top > outcome.snap_seq)
                },
            )
            .await;
        }
        let commits_during = ledger.lock().unwrap().len() - commits_before;
        assert!(
            commits_during >= HOT_LAYER_ASSEMBLIES,
            "{path}: the writer must have kept committing through the assemblies for the layer to \
             count as hot; it landed {commits_during} commit(s) across {HOT_LAYER_ASSEMBLIES} \
             assemblies"
        );
        assert!(
            cuts.windows(2).all(|pair| pair[0] <= pair[1]),
            "{path}: successive proven cuts of one layer never move backwards: {cuts:?}"
        );
        assert!(
            cuts.last() > cuts.first(),
            "{path}: the cuts must have advanced across the run — the assemblies read a layer \
             that was moving under them: {cuts:?}"
        );
    }

    stop.store(true, Ordering::Relaxed);
    drain_hot_writer(deadline, writer).await;
}

// =============================================================================
// Test 5c — a corrupt section index is refused on BOTH proven reads
// =============================================================================
//
// An `obj:secs` member whose section hash is gone. Under a proven cut — the
// atomic read at or below the atomic-read limits, the fenced read above them —
// the read saw the index and every hash at one cut, so the member is a corrupt
// index, never a race, and the assembler refuses the layer as Permanent:
// persisting it would drop the section without a trace. A matching tombstone
// beside it changes nothing — no tombstone re-read is issued on either path
// (the retired quiescence read's SR-2.3 reconciliation is gone).

/// The `obj:secs` member planted without a section hash.
const GHOST_SECTION: u64 = 99;

/// Create a [`populated_state`] layer and plant [`GHOST_SECTION`] in its
/// section index with no section hash behind it. With `with_tombstone`, also
/// plant a tombstone for it (on `obj:dels` and its `obj:del` hash) at a version
/// at or below the layer's sequence — what a reconciliation would have folded
/// in, had one run.
async fn layer_with_a_ghost_section(tv: &TestValkey, label: &str, with_tombstone: bool) -> u64 {
    let ls = layer_store(tv);
    let session_id = create_session_named(tv, label).await;
    let oid = ls
        .create_layer(
            &session_id,
            label,
            "assembler-test-user",
            &populated_state(),
            ext_store(tv),
        )
        .await
        .expect("create_layer must succeed");
    let oid_str = oid.to_string();
    let keys = tv.key_schema();
    let _: i64 = tv
        .client()
        .pool
        .sadd(keys.obj_sections_index(&oid_str), GHOST_SECTION.to_string())
        .await
        .expect("SADD of the ghost member must succeed");
    if with_tombstone {
        let _: i64 = tv
            .client()
            .pool
            .sadd(
                keys.obj_deleted_sections_index(&oid_str),
                GHOST_SECTION.to_string(),
            )
            .await
            .expect("SADD of the ghost tombstone member must succeed");
        let _: i64 = tv
            .client()
            .pool
            .hset(
                keys.obj_deleted_section(&oid_str, &GHOST_SECTION.to_string()),
                vec![("first_ver", "1"), ("last_ver", "1")],
            )
            .await
            .expect("HSET of the ghost tombstone must succeed");
    }
    oid
}

/// The ghost member is a corrupt index on both proven reads, tombstone or
/// not: refused as `Permanent`, naming the member and the read that proved
/// the cut, and nothing is persisted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_refuses_a_corrupt_section_index_on_the_atomic_and_the_fenced_read() {
    let tv = TestValkey::setup().await;
    let fenced = fenced_path_client().await;

    for (path, client) in [("atomic", tv.client()), ("fenced", &fenced)] {
        for with_tombstone in [true, false] {
            let label = format!("corrupt-index-{path}-{with_tombstone}");
            let oid = layer_with_a_ghost_section(&tv, &label, with_tombstone).await;

            let store = InMemoryCheckpointStore::new();
            let err = assemble_layer_checkpoint(
                client,
                tv.key_schema(),
                oid,
                &store,
                SERVICE_LABEL,
                &mut NoLeaseToKeep,
            )
            .await
            .err()
            .unwrap_or_else(|| {
                panic!(
                    "{label}: an indexed section with no hash under a proven cut is a corrupt \
                     index; the assembler must refuse it rather than persist a blob that \
                     silently drops the section"
                )
            });
            let live_edit::checkpoint::CheckpointError::Permanent { source } = &err else {
                panic!(
                    "{label}: a corrupt index is Permanent — retrying reads the same corruption, \
                     and no reconciliation (which would fold a planted tombstone in) runs under a \
                     proven cut; got {err:?}"
                );
            };
            let source = source.to_string();
            assert!(
                source.contains(&GHOST_SECTION.to_string()) && source.contains(path),
                "{label}: the refusal must name the member an operator has to repair and the \
                 read that proved the cut; got {source}"
            );
            assert_eq!(
                store
                    .list_layer_checkpoints(oid)
                    .await
                    .expect("listing an in-memory store cannot fail")
                    .len(),
                0,
                "{label}: nothing may be persisted for a refused layer"
            );
        }
    }
}

/// The assembler's state read is visible to an operator as the `checkpoint`
/// consumer of the snapshot read: an atomic read that returned state ticks
/// `live_edit_snapshot_read_total{consumer="checkpoint", path="atomic",
/// outcome="ok"}`, and a layer above the atomic-read limits ticks
/// `path="atomic", outcome="too_large"` and then, for its fenced read,
/// `path="fenced", outcome="ok"` — a cell zero-seeded before any fenced
/// checkpoint read runs.
///
/// Asserted as metric values because the cells ARE the contract here (the
/// test-writing guide's carve-out): `docs/service/docs/metrics-reference.md`
/// lists these cells under `consumer="checkpoint"`, where an operator reads
/// which layers take the fenced read and how it went, so a production loader
/// that named another consumer, a fenced route that skipped the atomic
/// attempt, or a fenced cell that only materialised on first use would leave
/// that reading wrong.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_reads_as_the_checkpoint_consumer_atomically_or_fenced_by_layer_size() {
    use live_edit::observability::metrics::test_util::{render, sample_value};

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;
    let oid = ls
        .create_layer(
            &session_id,
            "checkpoint-consumer-cells",
            "assembler-test-user",
            &populated_state(),
            ext_store(&tv),
        )
        .await
        .expect("create_layer must succeed");
    let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    tv.client()
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    let fenced = fenced_path_client().await;
    fenced
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    let checkpoint_reads = |path: &str, outcome: &str| {
        sample_value(
            &render(&metrics),
            "live_edit_snapshot_read_total",
            &[
                ("consumer", "checkpoint"),
                ("path", path),
                ("outcome", outcome),
            ],
        )
        .unwrap_or_else(|| {
            panic!(
                "live_edit_snapshot_read_total{{consumer=\"checkpoint\",path=\"{path}\",\
                 outcome=\"{outcome}\"}} is zero-seeded, so it must be present"
            )
        }) as u64
    };
    assert_eq!(
        checkpoint_reads("fenced", "ok"),
        0,
        "the checkpoint consumer's fenced cells are seeded at zero before any fenced read"
    );

    expect_written(
        assemble_layer_checkpoint(
            tv.client(),
            tv.key_schema(),
            oid,
            &InMemoryCheckpointStore::new(),
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        )
        .await
        .expect("the at-the-limits assembly must succeed"),
    );
    assert_eq!(
        (
            checkpoint_reads("atomic", "ok"),
            checkpoint_reads("atomic", "too_large"),
            checkpoint_reads("fenced", "ok"),
        ),
        (1, 0, 0),
        "at or below the limits the assembly is one atomic read under consumer=\"checkpoint\""
    );

    expect_written(
        assemble_layer_checkpoint(
            &fenced,
            tv.key_schema(),
            oid,
            &InMemoryCheckpointStore::new(),
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        )
        .await
        .expect("the above-the-limits assembly must succeed"),
    );
    assert_eq!(
        (
            checkpoint_reads("atomic", "ok"),
            checkpoint_reads("atomic", "too_large"),
            checkpoint_reads("fenced", "ok"),
        ),
        (1, 1, 1),
        "above the limits the atomic attempt answers TOO_LARGE under consumer=\"checkpoint\" — \
         the cell that tells an operator which layers take the fenced read — and the fenced read \
         then ticks its own ok cell under the same consumer"
    );
}

/// A failed atomic read fails the assembly as classified — `Transient` for a
/// transport error — and never falls back to the fenced read, which would
/// otherwise read the layer under its fence and write it. Staged through
/// `read_object_snapshot`'s own fault seam (Spec SDD SR-1.5), which answers in
/// place of the script's `EVALSHA` and is classified like a real failure.
///
/// Needs `--features test-fault-injection`; without it the seam does not exist
/// and this test is compiled out.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_fails_on_a_failed_atomic_read_without_falling_back_to_the_fenced_read() {
    use live_edit::storage::diff_read::ObjectSnapshotReadFaultInjector;

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;
    let oid = ls
        .create_layer(
            &session_id,
            "failed-atomic-read",
            "assembler-test-user",
            &populated_state(),
            ext_store(&tv),
        )
        .await
        .expect("create_layer must succeed");

    let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
    injector.push(fred::error::Error::new(
        fred::error::ErrorKind::Timeout,
        "injected atomic-read timeout",
    ));
    tv.client()
        .install_object_snapshot_read_fault_injector_for_test(injector.clone());

    let store = InMemoryCheckpointStore::new();
    let err = assemble_layer_checkpoint(
        tv.client(),
        tv.key_schema(),
        oid,
        &store,
        SERVICE_LABEL,
        &mut NoLeaseToKeep,
    )
    .await
    .err()
    .unwrap_or_else(|| {
        panic!(
            "a failed atomic read must fail the assembly; an Ok means it fell back to another \
             read of the layer"
        )
    });
    assert_eq!(
        injector.remaining(),
        0,
        "the canned failure must have answered the atomic read, or this test staged nothing"
    );
    assert!(
        matches!(
            err,
            live_edit::checkpoint::CheckpointError::Transient { .. }
        ),
        "a transport failure of the atomic read is Transient (the next tick retries); got {err:?}"
    );
    assert_eq!(
        store
            .list_layer_checkpoints(oid)
            .await
            .expect("listing an in-memory store cannot fail")
            .len(),
        0,
        "nothing may be persisted when the read failed"
    );
    assert!(
        fence_readers(&tv, oid).await.is_empty() && !fence_key_exists(&tv, oid).await,
        "and no fenced read was attempted: the fence key was never created"
    );

    // The seam is spent: the next assembly reads for real and writes.
    expect_written(
        assemble_layer_checkpoint(
            tv.client(),
            tv.key_schema(),
            oid,
            &store,
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        )
        .await
        .expect("with the seam spent the assembly must succeed"),
    );
}

// =============================================================================
// Test 5d — the fenced read's own endings (Spec SDD SR-5.3, SR-11.5): it never
//           persists an unproven state, and a lost lease releases its fence.
// =============================================================================

/// The live reader fields (`r:<token>`) on `object_id`'s snapshot fence.
async fn fence_readers(tv: &TestValkey, object_id: u64) -> Vec<String> {
    let fields: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().obj_snapshot_fence(&object_id.to_string()))
        .await
        .expect("HGETALL the snapshot fence must succeed");
    fields.into_keys().filter(|k| k.starts_with("r:")).collect()
}

/// Whether `object_id`'s snapshot fence key exists at all.
#[cfg(feature = "test-fault-injection")]
async fn fence_key_exists(tv: &TestValkey, object_id: u64) -> bool {
    let n: i64 = tv
        .client()
        .pool
        .exists(tv.key_schema().obj_snapshot_fence(&object_id.to_string()))
        .await
        .expect("EXISTS the snapshot fence must succeed");
    n > 0
}

/// Bound on a dropped fenced read's background fence release.
const FENCE_RELEASE_BUDGET: std::time::Duration = std::time::Duration::from_secs(10);

/// Poll until `object_id`'s fence has no live reader field, within
/// [`FENCE_RELEASE_BUDGET`]; `true` if it got there.
async fn fence_released_within_budget(tv: &TestValkey, object_id: u64) -> bool {
    tokio::time::timeout(FENCE_RELEASE_BUDGET, async {
        while !fence_readers(tv, object_id).await.is_empty() {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .is_ok()
}

/// Create a [`populated_state`] layer — above [`fenced_path_client`]'s limits —
/// and return its id.
async fn fenced_layer(tv: &TestValkey, label: &str) -> u64 {
    let session_id = create_session_named(tv, label).await;
    layer_store(tv)
        .create_layer(
            &session_id,
            label,
            "assembler-test-user",
            &populated_state(),
            ext_store(tv),
        )
        .await
        .expect("create_layer must succeed")
}

/// **A fenced read broken `FENCE_READ_ATTEMPTS` times is `Transient`, and
/// nothing is persisted.** A task bumps the layer's `obj:hdr seq` directly in a
/// tight loop — no commit, no fence check — so every attempt's release finds
/// the header moved off its grant and answers `BROKEN`. After the last attempt
/// the storage read ends `Unavailable`, which the loader classifies
/// `Transient`: the tick skips the layer and the next one reads it afresh.
/// Never an unproven state.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn assembler_fenced_read_broken_on_every_attempt_is_transient_and_persists_nothing() {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    use live_edit::storage::snapshot_fence::FENCE_READ_ATTEMPTS;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    let tv = TestValkey::setup_with_pool_size(8).await;
    let oid = fenced_layer(&tv, "fenced-broken").await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    let fenced = fenced_path_client().await;
    fenced
        .install_observability_hooks(Arc::clone(&metrics))
        .await;

    let stop = Arc::new(AtomicBool::new(false));
    let bumps = Arc::new(AtomicU64::new(0));
    let bumper = {
        let client = tv.client().clone();
        let header = tv.key_schema().obj_header(&oid.to_string());
        let (stop, bumps) = (Arc::clone(&stop), Arc::clone(&bumps));
        tokio::spawn(async move {
            while !stop.load(Ordering::Relaxed) {
                let _: i64 = client
                    .pool
                    .hincrby(&header, "seq", 1)
                    .await
                    .expect("HINCRBY obj:hdr seq must succeed");
                bumps.fetch_add(1, Ordering::Relaxed);
            }
        })
    };

    let store = InMemoryCheckpointStore::new();
    let result = assemble_layer_checkpoint(
        &fenced,
        tv.key_schema(),
        oid,
        &store,
        SERVICE_LABEL,
        &mut NoLeaseToKeep,
    )
    .await;
    stop.store(true, Ordering::Relaxed);
    bumper.await.expect("the seq bumper must not panic");

    let broken = sample_value(
        &render(&metrics),
        "live_edit_snapshot_read_total",
        &[
            ("consumer", "checkpoint"),
            ("path", "fenced"),
            ("outcome", "fence_broken"),
        ],
    )
    .unwrap_or(0.0) as u32;
    assert_eq!(
        broken,
        FENCE_READ_ATTEMPTS,
        "every attempt's release found the header moved, so the read was broken exactly \
         FENCE_READ_ATTEMPTS times ({} bumps landed)",
        bumps.load(Ordering::Relaxed)
    );
    match result {
        Err(live_edit::checkpoint::CheckpointError::Transient { .. }) => {}
        Err(other) => panic!("an exhausted fenced read is Transient; got {other:?}"),
        Ok(decision) => panic!(
            "a fenced read broken on every attempt must never yield a decision — no unproven \
             state may be persisted or judged; got {decision:?}"
        ),
    }
    assert!(
        store
            .list_layer_checkpoints(oid)
            .await
            .expect("listing an in-memory store cannot fail")
            .is_empty(),
        "nothing may be persisted off a read that never proved its cut"
    );
    assert!(
        fence_released_within_budget(&tv, oid).await,
        "every broken attempt's lease was released; still live: {:?}",
        fence_readers(&tv, oid).await
    );
}

/// **A fence wait that would pass `reader_wait_max(L)` is `Transient`, and
/// nothing is persisted.** A writers' gap is staged on the layer's fence ten
/// minutes past the store's clock — far beyond the cap for this layer's size —
/// so the checkpoint's `CappedFenceWait` refuses the first `RETRY_AFTER` at
/// once, with no sleep, and the storage read ends `Unavailable`. The
/// checkpoint takes no capacity, so nothing else bounds the wait.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_fenced_read_refuses_a_fence_wait_past_reader_wait_max_as_transient() {
    use fred::interfaces::LuaInterface;

    /// How far past the store's clock the staged gap ends.
    const FAR_MS: i64 = 600_000;

    let tv = TestValkey::setup().await;
    let oid = fenced_layer(&tv, "fenced-wait-cap").await;
    let fenced = fenced_path_client().await;

    let now_ms: i64 = tv
        .client()
        .pool
        .eval(
            "local t = redis.call('TIME') \
             return tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)",
            Vec::<String>::new(),
            Vec::<String>::new(),
        )
        .await
        .expect("the store clock must be readable");
    let fence_key = tv.key_schema().obj_snapshot_fence(&oid.to_string());
    let _: () = tv
        .client()
        .pool
        .hset(
            &fence_key,
            [
                ("gap_until", (now_ms + FAR_MS).to_string()),
                ("size_e", "1000".to_string()),
                ("size_b", "12000".to_string()),
            ],
        )
        .await
        .expect("staging the gap must succeed");
    let _: i64 = tv
        .client()
        .pool
        .pexpire(&fence_key, FAR_MS * 2, None)
        .await
        .expect("staging the key expiry must succeed");

    let store = InMemoryCheckpointStore::new();
    let result = assemble_layer_checkpoint(
        &fenced,
        tv.key_schema(),
        oid,
        &store,
        SERVICE_LABEL,
        &mut NoLeaseToKeep,
    )
    .await;

    match result {
        Err(live_edit::checkpoint::CheckpointError::Transient { source }) => assert!(
            source.to_string().contains("reader_wait_max"),
            "the refusal names the cap it would have passed; got {source}"
        ),
        Err(other) => panic!("a fence wait past reader_wait_max(L) is Transient; got {other:?}"),
        Ok(decision) => panic!(
            "a fenced read that never acquired its fence must not yield a decision; got \
             {decision:?}"
        ),
    }
    assert!(
        store
            .list_layer_checkpoints(oid)
            .await
            .expect("listing an in-memory store cannot fail")
            .is_empty(),
        "nothing may be persisted when the read never ran"
    );
    assert!(
        fence_readers(&tv, oid).await.is_empty(),
        "no lease was ever granted, so no reader field may exist"
    );
}

/// The message of the assembler's per-attempt timing event, whose `outcome`
/// field the validation tooling reads.
#[cfg(feature = "test-fault-injection")]
const TIMING_EVENT_MSG: &str = "layer checkpoint timing";

/// Run one assembly of a fenced layer whose keeper stops it with `stop` once
/// the read is inside its held-back first fan-out window, under its fence:
/// the fence lease granted (a reader field on the fence key) and the
/// held-back delay taken by the fan-out (the `test-fault-injection` delay
/// seam, held far longer than this test runs). Waiting on the reader field
/// alone would race the fan-out's start, and a keeper that fired before it
/// would end the read before the fan-out began.
///
/// Asserts what every stop shares: the assembly decides (no error) within
/// half the held-back window, the window was reached, nothing is persisted,
/// the dropped read's fence lease is released (its reader field gone — only
/// the drop can have released it in time), and the assembly's one
/// `layer checkpoint timing` event carries `outcome`. Returns the decision.
#[cfg(feature = "test-fault-injection")]
async fn stopped_mid_fenced_read(
    label: &str,
    stop: live_edit::checkpoint::assembler::KeeperStop,
    outcome: &str,
) -> AssemblyDecision {
    use live_edit::checkpoint::assembler::{KeeperStop, LeaderLeaseKeeper};
    use live_edit::storage::diff_read::{
        ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget,
    };

    /// How long the read's first window is held back: far past this test's
    /// whole budget, so only a dropped read can release the fence in time.
    const HELD_BACK: std::time::Duration = std::time::Duration::from_secs(60);

    /// Stops the read with `stop` once it is inside its held-back fan-out
    /// window: the injector's delay has been taken, and `object_id`'s fence
    /// shows a live reader.
    struct StopInsideTheHeldBackWindow {
        tv_client: live_edit::storage::client::ValkeyClient,
        fence_key: String,
        injector: Arc<ObjectSnapshotReadFaultInjector>,
        stop: KeeperStop,
    }

    #[async_trait::async_trait]
    impl LeaderLeaseKeeper for StopInsideTheHeldBackWindow {
        async fn hold_until_lost(&mut self) -> KeeperStop {
            loop {
                let fields: HashMap<String, String> = self
                    .tv_client
                    .pool
                    .hgetall(&self.fence_key)
                    .await
                    .unwrap_or_default();
                if self.injector.delays_remaining() == 0
                    && fields.keys().any(|k| k.starts_with("r:"))
                {
                    return self.stop;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        }
    }

    let tv = TestValkey::setup().await;
    let oid = fenced_layer(&tv, label).await;
    let fenced = fenced_path_client().await;
    let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
    injector.push_delay_at(ObjectSnapshotReadFaultTarget::next_read(), HELD_BACK);
    fenced.install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
    let mut keeper = StopInsideTheHeldBackWindow {
        tv_client: tv.client().clone(),
        fence_key: tv.key_schema().obj_snapshot_fence(&oid.to_string()),
        injector: Arc::clone(&injector),
        stop,
    };

    let store = InMemoryCheckpointStore::new();
    let (decision, captured) = crate::grpc::log_capture::capture_debug(tokio::time::timeout(
        HELD_BACK / 2,
        assemble_layer_checkpoint(
            &fenced,
            tv.key_schema(),
            oid,
            &store,
            SERVICE_LABEL,
            &mut keeper,
        ),
    ))
    .await;
    let decision = decision
        .expect("a stopped read must end the assembly long before the held-back window would")
        .expect("a stopped read is a decision, not an error");

    assert_eq!(
        injector.delays_remaining(),
        0,
        "fixture: the held-back window was reached (the keeper fires only once it is), so the \
         read really was under its fence"
    );
    assert!(
        store
            .list_layer_checkpoints(oid)
            .await
            .expect("listing an in-memory store cannot fail")
            .is_empty(),
        "nothing may be persisted off a stopped read"
    );
    assert!(
        fence_released_within_budget(&tv, oid).await,
        "the dropped read's fence lease must be released (its reader field gone) — the held-back \
         window had {HELD_BACK:?} to run, so only the drop can have released it; still live: {:?}",
        fence_readers(&tv, oid).await
    );
    let timing = captured.only(TIMING_EVENT_MSG);
    assert_eq!(
        timing["fields"]["outcome"], outcome,
        "the timing event names why the attempt ended: {timing}"
    );
    decision
}

/// **A lease lost mid fenced read ends the assembly `LeaseLost`, persists
/// nothing, and releases the fence.** The keeper reports its lease lost once
/// the read is under its fence ([`stopped_mid_fenced_read`]); the assembler
/// drops the read — whose fence lease guard releases the lease in the
/// background — and decides `LeaseLost`, its timing event `lease_lost`.
///
/// Needs `--features test-fault-injection`; without it the delay seam does not
/// exist and this test is compiled out.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn assembler_lease_lost_mid_fenced_read_persists_nothing_and_releases_the_fence() {
    let decision = stopped_mid_fenced_read(
        "fenced-lease-lost",
        live_edit::checkpoint::assembler::KeeperStop::LeaseLost,
        "lease_lost",
    )
    .await;
    assert!(
        matches!(decision, AssemblyDecision::LeaseLost),
        "the keeper lost the lease while the read was under its fence; got {decision:?}"
    );
}

/// **A shutdown mid fenced read ends the assembly `ShuttingDown`, persists
/// nothing, and releases the fence** (post-completion review F12). The keeper
/// reports its owner shutting down once the read is under its fence, as the
/// periodic loop's keeper does when the loop is cancelled; the assembler
/// drops the read and decides `ShuttingDown` — never `LeaseLost`, which the
/// loop would demote and count — and its timing event says `shutdown`.
///
/// Needs `--features test-fault-injection`, as above.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn assembler_shutdown_mid_fenced_read_persists_nothing_and_releases_the_fence() {
    let decision = stopped_mid_fenced_read(
        "fenced-shutdown",
        live_edit::checkpoint::assembler::KeeperStop::ShuttingDown,
        "shutdown",
    )
    .await;
    assert!(
        matches!(decision, AssemblyDecision::ShuttingDown),
        "the keeper's owner shut down while the read was under its fence; got {decision:?}"
    );
}

// =============================================================================
// Test 5e — Spec SDD SR-4.5: the checkpoint's fenced read takes no catch-up
//           capacity.
// =============================================================================

/// Bound on the fenced assembly of a quiet layer while every catch-up capacity
/// bound on its client is held. Healthy: one fenced read, milliseconds. A read
/// that waited on any of those bounds would wait forever — nothing frees them
/// until the test ends — so this deadline is what turns that hang into a
/// failure that prints.
const UNGATED_ASSEMBLY_BUDGET: std::time::Duration = std::time::Duration::from_secs(20);

/// **The checkpoint's fenced read takes no catch-up slot, read permit or
/// memory budget (Spec SDD SR-4.5).** The test holds every catch-up capacity
/// bound on the reading client — every `catchup_gate` slot, every permit of
/// both read-permit classes (`catchup_fenced_read_gate` and the atomic
/// `catchup_read_gate`), and the whole `catchup_build_budget` — and a fenced
/// assembly of a layer above the atomic-read limits must still be written
/// within [`UNGATED_ASSEMBLY_BUDGET`]. The checkpoint is the leader's
/// durability work: queued behind subscriber catch-ups, a burst of reconnects
/// would stall it for as long as the burst lasts.
///
/// It holds today because the loader hands the fenced read a
/// `CappedFenceWait`, a hook that takes no capacity; a hook that took any of
/// these bounds — a flight's, say — would hang here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_fenced_read_takes_no_catch_up_slot_permit_or_memory_budget() {
    use live_edit::observability::metrics::test_util::{render, sample_value};
    use live_edit::storage::diff_read::SnapshotReadConsumer;

    let tv = TestValkey::setup().await;
    let oid = fenced_layer(&tv, "fenced-ungated").await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    let fenced = fenced_path_client().await;
    fenced
        .install_observability_hooks(Arc::clone(&metrics))
        .await;

    // Hold every catch-up capacity bound on the reading client. A free slot
    // or permit is granted at once, so the short deadline only bounds a
    // fixture that went wrong.
    let take_by = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    let slot_limit = fenced.catchup_gate.limit().expect(
        "fixture: the test client caps catch-up slots; an uncapped gate never waits, so there \
         would be no slot to hold",
    );
    let slots: Vec<_> = (0..slot_limit)
        .map(|_| {
            fenced
                .catchup_gate
                .try_acquire()
                .expect("fixture: an idle gate grants every slot at once")
        })
        .collect();
    assert_eq!(
        fenced.catchup_gate.available(),
        0,
        "fixture: every slot is held"
    );
    let mut read_permits = Vec::new();
    for gate in [&fenced.catchup_fenced_read_gate, &fenced.catchup_read_gate] {
        for _ in 0..gate.limit() {
            read_permits.push(
                gate.acquire(take_by, SnapshotReadConsumer::Catchup)
                    .await
                    .unwrap_or_else(|e| panic!("fixture: a free permit of {gate:?}: {e:?}")),
            );
        }
        assert_eq!(
            gate.available(),
            0,
            "fixture: every permit of {gate:?} is held"
        );
    }
    let budget = &fenced.catchup_build_budget;
    let memory = budget
        .try_grant(budget.total_bytes())
        .expect("fixture: an idle budget grants its whole total at once");
    assert_eq!(
        budget.reserved_bytes(),
        budget.total_bytes(),
        "fixture: the whole memory budget is held"
    );

    let store = InMemoryCheckpointStore::new();
    let decision = tokio::time::timeout(
        UNGATED_ASSEMBLY_BUDGET,
        assemble_layer_checkpoint(
            &fenced,
            tv.key_schema(),
            oid,
            &store,
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        ),
    )
    .await
    .unwrap_or_else(|_| {
        panic!(
            "the fenced assembly did not finish within {UNGATED_ASSEMBLY_BUDGET:?} while every \
             catch-up slot, read permit and the whole catch-up memory budget were held: the \
             checkpoint's read is waiting on catch-up capacity, which it must never take (Spec \
             SDD SR-4.5)"
        )
    })
    .expect("the assembly must succeed against a healthy backend");
    expect_written(decision);

    let fenced_ok = sample_value(
        &render(&metrics),
        "live_edit_snapshot_read_total",
        &[
            ("consumer", "checkpoint"),
            ("path", "fenced"),
            ("outcome", "ok"),
        ],
    )
    .unwrap_or(0.0) as u64;
    assert_eq!(
        fenced_ok, 1,
        "fixture: the assembly read the layer through the fenced read, or this test pinned \
         nothing about it"
    );
    assert_eq!(
        store
            .list_layer_checkpoints(oid)
            .await
            .expect("listing an in-memory store cannot fail")
            .len(),
        1,
        "the assembly persisted its base"
    );
    assert_eq!(
        (
            fenced.catchup_gate.available(),
            fenced.catchup_fenced_read_gate.available(),
            fenced.catchup_read_gate.available(),
            budget.reserved_bytes(),
        ),
        (0, 0, 0, budget.total_bytes()),
        "every bound is still exactly as the test holds it"
    );
    drop((slots, read_permits, memory));
}

/// **The test-only unprovable-cut seam reaches the regression arm through
/// the public client API, on both proven reads.** With the seam armed for a
/// layer, its pin reads above every cut, so the real read that follows —
/// atomic, or fenced above the limits — is judged regressed and the assembly
/// ends `SkippedNotQuiesced` with nothing persisted; a fenced read's lease is
/// released as on any completed read. Disarmed, the next assembly writes.
///
/// Needs `--features test-checkpoint-fault-injection`; without it the seam
/// does not exist and this test is compiled out.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_with_the_unprovable_cut_seam_armed_skips_on_both_proven_reads() {
    let tv = TestValkey::setup().await;
    let fenced = fenced_path_client().await;

    for (path, client) in [("atomic", tv.client()), ("fenced", &fenced)] {
        let oid = fenced_layer(&tv, &format!("seam-{path}")).await;
        client.arm_unprovable_checkpoint_cut_for_test(oid);

        let store = InMemoryCheckpointStore::new();
        match assemble_layer_checkpoint(
            client,
            tv.key_schema(),
            oid,
            &store,
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        )
        .await
        .expect("a regression is a decision, not an error")
        {
            AssemblyDecision::SkippedNotQuiesced => {}
            other => panic!("{path}: the armed layer must be skipped; got {other:?}"),
        }
        assert!(
            store
                .list_layer_checkpoints(oid)
                .await
                .expect("listing an in-memory store cannot fail")
                .is_empty(),
            "{path}: a skipped assembly persists nothing"
        );
        assert!(
            fence_released_within_budget(&tv, oid).await,
            "{path}: no fence lease may be left behind"
        );

        client.disarm_unprovable_checkpoint_cut_for_test(oid);
        expect_written(
            assemble_layer_checkpoint(
                client,
                tv.key_schema(),
                oid,
                &store,
                SERVICE_LABEL,
                &mut NoLeaseToKeep,
            )
            .await
            .expect("a disarmed layer must assemble"),
        );
    }
}

// =============================================================================
// b.vnw — `observed_commit_span` is measured off the two cursor columns of
//         the SAME header read that produces the identity fields
// =============================================================================
//
// The assembler stamps `span = snap_seq − last_checkpoint_seq −
// restore_seq_bump`. Both subtrahends come from `read_header_identity`'s
// single `HGETALL`, and both are read with a "missing or unparseable folds to
// 0" rule.
//
// That rule is a deliberate refusal to fail: a header column that a human
// repaired badly, or that a future schema change stops writing, must cost the
// blob its span measurement and nothing more. Failing the assembly instead
// would drop the layer out of DR entirely — trading a degraded guard band for
// no durable copy at all, which inverts the severity.
//
// The three cases below pin: the incremental measurement across generations
// (the number the field is FOR); the fold-to-0 tolerance (the behaviour that
// keeps a bad column from cascading); and the trust bit that distinguishes a
// READ 0 from an INVENTED one, which only becomes observable when exactly one
// of the two columns is unparseable.
//
// The restore-bump subtraction is exercised end-to-end in
// `storage::checkpoint_apply_test::post_restore_recheckpoint_measures_commits_not_the_restore_band`,
// which drives it through a real `apply_checkpoint` rather than a planted
// column.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_measures_the_span_between_consecutive_generations() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;
    let layer_name = "span-between-generations";

    let oid = ls
        .create_layer(
            &session_id,
            layer_name,
            "assembler-test-user",
            &populated_state(),
            ext_store(&tv),
        )
        .await
        .expect("create_layer must succeed");

    let store = InMemoryCheckpointStore::new();
    let first = expect_written(
        assemble_layer_checkpoint(
            tv.client(),
            tv.key_schema(),
            oid,
            &store,
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        )
        .await
        .expect("first assembly must succeed"),
    );

    // Generation 1 has no predecessor (`last_checkpoint_seq` absent → 0), so
    // there is nothing to measure a per-interval span against: the assembler
    // stamps the no-measurement sentinel 0 (b.vnw round-2). Measuring here
    // would report commits-since-CREATION, a different quantity in the same
    // units as the one the guard band consumes; the sentinel folds the band
    // back to the configured margin instead.
    let first_meta = store
        .get_latest_layer_checkpoint(oid)
        .await
        .expect("get must succeed")
        .expect("generation 1 must exist")
        .meta
        .clone();
    assert_eq!(
        first_meta.observed_commit_span, 0,
        "a first generation (last_checkpoint_seq == 0) has no predecessor to \
         measure against, so it must carry the no-measurement sentinel rather \
         than a commits-since-creation count"
    );

    // Stamp the cursor the way the periodic loop's success arm does, then
    // commit a known number of updates. From here the measurement must be
    // INCREMENTAL — the commits since generation 1, not since creation.
    const COMMITS: u64 = 4;
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let _: () = tv
        .client()
        .pool
        .hset(
            &header_key,
            vec![
                ("last_checkpoint_seq", first.snap_seq.to_string()),
                ("restore_seq_bump", "0".to_string()),
            ],
        )
        .await
        .expect("HSET cursor columns must succeed");

    for i in 0..COMMITS {
        ls.update_layer_relayed(
            &session_id,
            layer_name,
            oid,
            Some(OwnedDelta {
                struct_commands: vec![StructCommand::CreateSection(
                    oodf_ffi::CommandCreateSection {
                        parent_id: 0,
                        section_id: 100 + i,
                        section_name: format!("span-{i}"),
                        section_type: 1,
                    },
                )],
                ..Default::default()
            }),
            "span-writer",
            0,
        )
        .await
        .unwrap_or_else(|e| panic!("commit {i} must succeed: {e:?}"));
    }

    let second = expect_written(
        assemble_layer_checkpoint(
            tv.client(),
            tv.key_schema(),
            oid,
            &store,
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        )
        .await
        .expect("second assembly must succeed"),
    );
    assert_eq!(
        second.snap_seq,
        first.snap_seq + COMMITS,
        "fixture invariant: {COMMITS} commits must advance the cut by {COMMITS}"
    );

    let second_meta = store
        .get_latest_layer_checkpoint(oid)
        .await
        .expect("get must succeed")
        .expect("generation 2 must exist")
        .meta
        .clone();
    assert_eq!(
        second_meta.observed_commit_span, COMMITS,
        "b.vnw: generation 2's span must be the {COMMITS} commits it covers \
         that generation 1 did not — snap_seq({}) − last_checkpoint_seq({}). \
         A value of {} means the measurement is cumulative-since-creation \
         rather than per-interval, which over-sizes the band without bound as \
         a long-lived layer accumulates commits",
        second.snap_seq, first.snap_seq, second.snap_seq
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_folds_an_unparseable_cursor_column_to_zero_rather_than_failing() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;

    let oid = ls
        .create_layer(
            &session_id,
            "garbage-cursor-layer",
            "assembler-test-user",
            &populated_state(),
            ext_store(&tv),
        )
        .await
        .expect("create_layer must succeed");

    // Both cursor columns carry non-numeric garbage — the shape a bad manual
    // repair leaves behind.
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let _: () = tv
        .client()
        .pool
        .hset(
            &header_key,
            vec![
                ("last_checkpoint_seq", "not-a-number"),
                ("restore_seq_bump", ""),
            ],
        )
        .await
        .expect("HSET garbage cursor columns must succeed");

    let store = InMemoryCheckpointStore::new();
    let outcome = expect_written(
        assemble_layer_checkpoint(
            tv.client(),
            tv.key_schema(),
            oid,
            &store,
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        )
        .await
        .expect(
            "an unparseable cursor column must cost the blob its span \
                 measurement, NOT the whole assembly — failing here would drop \
                 the layer out of DR over a bookkeeping column",
        ),
    );

    let meta = store
        .get_latest_layer_checkpoint(oid)
        .await
        .expect("get must succeed")
        .expect("a generation must exist")
        .meta
        .clone();
    assert_eq!(
        meta.observed_commit_span, 0,
        "b.vnw round-2: both garbage columns fold to 0, but that 0 is a \
         FABRICATED baseline, so the assembler must decline to measure and \
         stamp the no-measurement sentinel. A value of {} here would mean it \
         measured against the invented zeros and reported \
         commits-since-creation as if it were this layer's per-interval rate",
        outcome.snap_seq
    );
}

/// The case above cannot, on its own, prove that `cursor_columns_trusted`
/// exists: with BOTH columns garbage, `last_checkpoint_seq` folds to 0 and the
/// separate `last_checkpoint_seq == 0` first-generation arm returns the same
/// sentinel. The two arms are indistinguishable there.
///
/// This case separates them. `last_checkpoint_seq` parses cleanly to a real
/// non-zero generation, so the first-generation arm does NOT fire; only
/// `restore_seq_bump` is unparseable. Delete the trust bit and the assembler
/// measures `snap_seq − PLANTED_CURSOR − 0` — a confident, wrong, non-zero
/// per-interval reading built on a subtrahend nobody read. That is the failure
/// mode the flag was added for, and this is the only test that sees it.
///
/// A fabricated `restore_seq_bump` is the more dangerous of the two columns to
/// invent, which is why it is the one held garbage here: it is *subtracted*,
/// so inventing 0 for it can only make the span come out LARGER than the truth
/// — and an over-sized span propagates straight into the next restore's guard
/// band.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_declines_to_measure_when_only_the_restore_bump_column_is_unparseable() {
    // A real, cleanly-parsing predecessor generation. Deliberately low so the
    // fixture's `snap_seq` is comfortably above it and the span the assembler
    // would report without the trust bit is unmistakably non-zero.
    const PLANTED_CURSOR: u64 = 1;

    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;
    let layer_name = "half-garbage-cursor-layer";

    let oid = ls
        .create_layer(
            &session_id,
            layer_name,
            "assembler-test-user",
            &populated_state(),
            ext_store(&tv),
        )
        .await
        .expect("create_layer must succeed");

    // Commit so `snap_seq` runs well clear of the planted cursor.
    for i in 0..3u64 {
        ls.update_layer_relayed(
            &session_id,
            layer_name,
            oid,
            Some(OwnedDelta {
                struct_commands: vec![StructCommand::CreateSection(
                    oodf_ffi::CommandCreateSection {
                        parent_id: 0,
                        section_id: 500 + i,
                        section_name: format!("half-garbage-{i}"),
                        section_type: 1,
                    },
                )],
                ..Default::default()
            }),
            "half-garbage-writer",
            0,
        )
        .await
        .unwrap_or_else(|e| panic!("commit {i} must succeed: {e:?}"));
    }

    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let _: () = tv
        .client()
        .pool
        .hset(
            &header_key,
            vec![
                ("last_checkpoint_seq", PLANTED_CURSOR.to_string()),
                ("restore_seq_bump", "0x10".to_string()),
            ],
        )
        .await
        .expect("HSET cursor columns must succeed");

    let store = InMemoryCheckpointStore::new();
    let outcome = expect_written(
        assemble_layer_checkpoint(
            tv.client(),
            tv.key_schema(),
            oid,
            &store,
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        )
        .await
        .expect("an unparseable restore_seq_bump must not fail the assembly"),
    );
    assert!(
        outcome.snap_seq > PLANTED_CURSOR,
        "fixture invariant: snap_seq ({}) must exceed the planted cursor \
         ({PLANTED_CURSOR}), otherwise the un-trusted measurement this case \
         rules out would have been 0 anyway and the assertion below is vacuous",
        outcome.snap_seq
    );

    let meta = store
        .get_latest_layer_checkpoint(oid)
        .await
        .expect("get must succeed")
        .expect("a generation must exist")
        .meta
        .clone();
    assert_eq!(
        meta.observed_commit_span,
        0,
        "b.vnw round-2: one unparseable cursor column poisons the WHOLE \
         measurement, not just its own term. last_checkpoint_seq={PLANTED_CURSOR} \
         parsed fine, so the first-generation arm did not fire — a non-zero \
         value here (specifically {}) means the assembler subtracted a \
         restore_seq_bump of 0 that it never actually read, and stamped the \
         resulting over-estimate as this layer's per-interval commit rate",
        outcome.snap_seq - PLANTED_CURSOR
    );
}

// =============================================================================
// Test 6 — b.gcb / b.vz7: two INDEPENDENT assemblies of the SAME idle layer
//          produce byte-identical base OMPF and a byte-identical ext-refs
//          sidecar.
// =============================================================================
//
// The in-crate sibling
// (`checkpoint::assembler::tests::build_reset_ompf_from_state_is_byte_identical_for_equal_state`)
// pins the emitter given two hand-built `ObjectState`s. This closes the gap to
// the REAL path: each `assemble_layer_checkpoint` call runs its own proven
// read (`read_object_snapshot`) → `snapshot_to_object_state`, so each run builds its
// own freshly-seeded `HashMap`s off a fresh Valkey read. `RandomState::new()`
// draws a new key per map INSTANCE, so the two runs iterate their sections /
// fields / time-sample maps differently in exactly the way two replicas' maps
// do — which is what the Task's "two assemblies of the same state in different
// processes" success criterion is really asking for evidence of. (The genuinely
// cross-PROCESS half, for `base.ompf` only, is Subtask T6's two generator
// invocations; nothing here claims to be that.)

/// Dangling parent id for the fixture's orphan sections. `DET_ORPHAN_PARENT` is
/// never itself a section and is never registered in `state.children`, so the
/// reachable-from-root DFS cannot reach its children and they land in
/// `sections_in_dfs_order`'s orphan tail (seam E-1(ii)).
const DET_ORPHAN_PARENT: u64 = 900;

/// Ext-value payload for `tag`. Distinct per tag, so each tag has a distinct
/// SHA-256: the fixture's hashes are real content addresses with a real blob
/// behind them in the ext-value store, not planted constants.
fn det_ext_payload(tag: &str) -> Vec<u8> {
    format!("det-ext-payload-{tag}").into_bytes()
}

fn det_section(id: u64, parent: u64) -> Section {
    Section {
        section_id: id,
        parent_id: parent,
        name: format!("det-sec-{id}"),
        section_type: 1,
        section_order: id as u32,
        first_version: 1,
        last_version: 9,
    }
}

/// `cumulative_set_order` and `authoring_set_order` are separate arguments on
/// purpose — see NON-VACUITY (d) in [`determinism_fixture`].
fn det_field(
    section_id: u64,
    key: &str,
    version: u64,
    cumulative_set_order: u64,
    authoring_set_order: u64,
) -> Field {
    Field {
        section_id,
        key_name: key.to_string(),
        value: FieldValue::Inline(vec![version as u8, section_id as u8]),
        set_order: cumulative_set_order,
        authoring_set_order,
        version,
    }
}

fn det_ext_field(
    section_id: u64,
    key: &str,
    version: u64,
    cumulative_set_order: u64,
    authoring_set_order: u64,
    hash: [u8; 32],
    size: u64,
) -> Field {
    Field {
        section_id,
        key_name: key.to_string(),
        value: FieldValue::External {
            hash,
            size,
            ext_value_index: 0,
        },
        set_order: cumulative_set_order,
        authoring_set_order,
        version,
    }
}

fn det_time_sample(section_id: u64, time: f64, version: u64) -> TimeSample {
    TimeSample {
        section_id,
        time: OrderedFloat(time),
        value: FieldValue::Inline(vec![version as u8, section_id as u8]),
        version,
    }
}

fn det_ext_time_sample(
    section_id: u64,
    time: f64,
    version: u64,
    hash: [u8; 32],
    size: u64,
) -> TimeSample {
    TimeSample {
        section_id,
        time: OrderedFloat(time),
        value: FieldValue::External {
            hash,
            size,
            ext_value_index: 0,
        },
        version,
    }
}

/// Everything the determinism test needs: the layer state to create, the
/// ext-value blobs that must exist behind its content addresses, and the
/// ext-hash order the landed total order forces.
struct DeterminismFixture {
    state: ObjectState,
    /// `(hash, payload)` pairs, pre-stored via `ValkeyExternalValueStore`.
    ext_blobs: Vec<([u8; 32], Vec<u8>)>,
    /// Lowercase-hex ext hashes in the order `collect_external_value_hashes`
    /// must produce: emitted-field order first, then emitted-time-sample order.
    expected_hash_order: Vec<String>,
}

/// The non-vacuous fixture for the two-assembly determinism test.
///
/// Deliberately a SIBLING of `populated_state()` rather than a mutation of it:
/// four tests above depend on that helper's exact shape, and this one needs
/// three properties it does not have (orphans, external values, and ties on the
/// leading field sort key).
///
/// Every NON-VACUITY condition is annotated with what its removal would let
/// through. Each guards a distinct one of the three landed sorts; drop any and
/// at least one sort stops being tested here.
///
/// Note what this fixture does NOT need to do, and why: it is built ONCE and
/// pushed through `create_layer`. The two assemblies each re-read it out of
/// Valkey and rebuild their own maps, so the map-layout divergence this test
/// rides on comes from the two independent loads — not from any permuted
/// insertion order in the builder (which is the in-crate sibling's technique,
/// where there is no store to round-trip through).
fn determinism_fixture() -> DeterminismFixture {
    // Ext blobs. Five are referenced FIRST from fields, four FIRST from time
    // samples; `a2` is additionally referenced from a later time sample so the
    // first-appearance dedup path is covered (it must NOT produce a second
    // sidecar entry).
    let tags = ["a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4"];
    let blobs: Vec<([u8; 32], Vec<u8>)> = tags
        .iter()
        .map(|t| {
            let payload = det_ext_payload(t);
            (sha256(&payload), payload)
        })
        .collect();
    let h = |tag: &str| -> ([u8; 32], u64) {
        let idx = tags.iter().position(|t| *t == tag).expect("known tag");
        (blobs[idx].0, blobs[idx].1.len() as u64)
    };

    let mut state = ObjectState {
        sequence_num: 9,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 908,
        ..Default::default()
    };

    // Reachable tree: root(0) └─1 ├─2 └─3 ; 2 └─4. `children` is real
    // `RPUSH`/`LRANGE`-ordered persisted data, so the DFS prefix was always
    // deterministic and is NOT what this test is about (seam E-5).
    for s in [
        det_section(1, 0),
        det_section(2, 1),
        det_section(3, 1),
        det_section(4, 2),
    ] {
        state
            .children
            .entry(s.parent_id)
            .or_default()
            .push(s.section_id);
        state.sections.insert(s.section_id, s);
    }

    // NON-VACUITY (a) — FIVE ORPHAN sections under the dangling
    // `DET_ORPHAN_PARENT`, unreachable from root, so they exercise
    // `sections_in_dfs_order`'s orphan append (seam E-1(ii)).
    //
    // Removing them would let a FIELDS-ONLY sort pass this test: with a purely
    // reachable tree the emitted `struct_commands` order is already fixed by
    // `children` alone, and the OMPF byte comparison would no longer cover the
    // section sequence at all.
    //
    // FIVE rather than the "at least two" minimum so a red is not a coin flip:
    // an unordered tail has to draw the same one of 120 permutations in BOTH
    // assemblies to slip past, instead of one of two.
    for sid in [80_u64, 81, 82, 83, 84] {
        state
            .sections
            .insert(sid, det_section(sid, DET_ORPHAN_PARENT));
    }

    // NON-VACUITY (b) — fields TIE on the leading `(version,
    // authoring_set_order)` pair `(6, 1)` ACROSS four sections AND WITHIN one
    // section (`e_c` / `e_d`, both section 1). Without the cross-section tie the
    // `section_id` component of the sort key is never exercised; without the
    // within-section tie the `key_name` tiebreak is never exercised, and a
    // 3-tuple sort — or a *stable* sort riding on `HashMap` order for the ties —
    // would pass (seam E-2).
    //
    // FIVE of the seven carry distinct ext hashes, for the same non-flakiness
    // reason as the orphan tail: the sidecar's field-derived hash order then has
    // 120 ways to be wrong per assembly rather than two. `early` and `later`
    // stay INLINE at the two ends of the sorted sequence, so the hash scan's
    // "skip non-external" branch is exercised inside the sequence, not around it.
    //
    // NON-VACUITY (d) — every field's cumulative `set_order` (the 900s) is
    // deliberately out of line with its `authoring_set_order` (0..2), and the
    // cumulative cursors are not even ascending in emission order. Removing that
    // divergence would let a broken `set_order = authoring_set_order` remap
    // through: the wire slot would carry the same number either way and the
    // decode assertion below would be vacuous.
    let (a1, a1_len) = h("a1");
    let (a2, a2_len) = h("a2");
    let (a3, a3_len) = h("a3");
    let (a4, a4_len) = h("a4");
    let (a5, a5_len) = h("a5");
    for f in [
        det_ext_field(2, "e_a", 6, 900, 1, a3, a3_len),
        det_ext_field(3, "e_b", 6, 901, 1, a4, a4_len),
        det_ext_field(1, "e_c", 6, 902, 1, a1, a1_len),
        det_ext_field(1, "e_d", 6, 903, 1, a2, a2_len),
        det_ext_field(4, "s4_k", 6, 905, 1, a5, a5_len),
        det_field(1, "early", 5, 907, 2),
        det_field(3, "later", 7, 904, 0),
    ] {
        state.fields.insert((f.section_id, f.key_name.clone()), f);
    }

    // NON-VACUITY (c) — `b1`..`b4` appear FIRST on a TIME SAMPLE and on NO
    // field, and the four samples carrying them share `(version = 7,
    // section_id = 2)`, differing only in `time`.
    //
    // `collect_external_value_hashes` scans fields THEN time samples into ONE
    // shared first-appearance pool (seam E-1(i)), so removing them would leave
    // the sidecar assertion a fields-only claim and the time-sample sort
    // untested here — a state whose ext refs sit on time samples would still
    // produce a non-deterministic sidecar. Sharing `(version, section_id)` is
    // what exercises the `time` component; without it a `(version,
    // section_id)`-only sort would pass. FOUR rather than two, again so an
    // unsorted red is not a coin flip.
    let (b1, b1_len) = h("b1");
    let (b2, b2_len) = h("b2");
    let (b3, b3_len) = h("b3");
    let (b4, b4_len) = h("b4");
    for ts in [
        det_ext_time_sample(2, 3.0, 7, b3, b3_len),
        det_ext_time_sample(2, 1.0, 7, b1, b1_len),
        det_ext_time_sample(2, 4.0, 7, b4, b4_len),
        det_ext_time_sample(2, 2.0, 7, b2, b2_len),
        det_time_sample(3, 2.0, 7),
        // Already seen on field `e_d` — first-appearance dedup, no new entry.
        det_ext_time_sample(3, 2.5, 8, a2, a2_len),
        det_time_sample(4, 0.5, 9),
    ] {
        state.time_samples.insert((ts.section_id, ts.time), ts);
    }

    // Fields sorted `(version, authoring_set_order, section_id, key_name)`:
    //   (5,2,1,early) inline, (6,1,1,e_c)→a1, (6,1,1,e_d)→a2, (6,1,2,e_a)→a3,
    //   (6,1,3,e_b)→a4, (6,1,4,s4_k)→a5, (7,0,3,later) inline
    // then time samples sorted `(version, section_id, time)`:
    //   (7,2,1.0)→b1, (7,2,2.0)→b2, (7,2,3.0)→b3, (7,2,4.0)→b4, (7,3,2.0)
    //   inline, (8,3,2.5)→a2 (dup, no new entry), (9,4,0.5) inline
    let expected_hash_order = ["a1", "a2", "a3", "a4", "a5", "b1", "b2", "b3", "b4"]
        .iter()
        .map(|t| sha256_to_hex(&h(t).0))
        .collect();

    DeterminismFixture {
        state,
        ext_blobs: blobs,
        expected_hash_order,
    }
}

/// b.gcb / b.vz7 — two `assemble_layer_checkpoint` runs against ONE idle Valkey
/// layer, into two SEPARATE `InMemoryCheckpointStore`s, agree byte-for-byte on
/// both the base OMPF and the ext-refs sidecar.
///
/// This is the structural ground SR-9.2's duplicate-leader re-base idempotency
/// argument stands on: identical state ⇒ identical bytes ⇒ a re-base PUT at the
/// same key is byte-idempotent, rather than "the two blobs mean the same thing",
/// which every future reader would have to re-derive.
///
/// **Idleness is structural, not timed.** There is no concurrent writer and no
/// unprovable-cut seam armed, so both runs read the same state and prove its
/// cut with the atomic read. There are no sleeps, no wall-clock waits and no
/// `Instant` arithmetic anywhere here; a `SkippedNotQuiesced` would be the
/// regression, and `expect_written` says so.
///
/// **What is deliberately NOT compared.** `CheckpointMeta::written_at_unix_ms`
/// is `SystemTime::now()` at put time and is expected to differ between the two
/// runs, so the two metas are never compared as wholes — only `sidecar_bytes`
/// is, and only as raw bytes. The sidecar comparison is likewise never made
/// against `read_ext_refs_sidecar`'s `HashSet<String>`: a set cannot witness
/// order, which is the b.vz7 defect itself.
///
/// **Mutation evidence.** Each of the three landed sorts in `src/domain/diff.rs`
/// was neutralised in turn (scratch edit, restored byte-identically; never
/// committed), five `--retries 0` repeats each. This test observed red 5/5 on
/// all three:
///
/// | neutralised | first assertion red | content-only or length-too |
/// |---|---|---|
/// | `Self::sort_fields_by_authoring(&mut fields)` in `plan_full_content` | MUTATION-EVIDENCE-LENGTH (5/5) | **length-too** every run (1744 vs 1752 / 1752 vs 1736 / 1752 vs 1744 bytes) |
/// | `time_samples.sort_by_key(…)` in `plan_full_content` | MUTATION-EVIDENCE-LENGTH (3/5) or MUTATION-EVIDENCE-CONTENT (2/5) | **length-too** (1752 vs 1736 bytes); content-only at equal length (first differing byte 16 / 142) on the rest |
/// | `orphans.sort_unstable_by_key(…)` in `sections_in_dfs_order` | MUTATION-EVIDENCE-CONTENT (5/5) | **content-only**, never length — the five orphans differ only in `section_id` and their names (`det-sec-80`..`det-sec-84`) are equal-length, so a permuted tail changes no vtable count and no offset width |
///
/// Which of LENGTH / CONTENT fires for the first two is a property of the
/// particular pair of `HashMap` seeds that run drew, not of the defect — both
/// are the same ordering regression. That is exactly why the length assertion is
/// separate and first: a run that diverges at equal length must not read as "a
/// value differs".
///
/// Each neutralisation was then re-run with the byte assertions temporarily
/// inert, to confirm the decoded spot checks NAME the regression rather than
/// leaving it an opaque byte inequality (3/3 each):
///
/// | neutralised | spot check red |
/// |---|---|
/// | field sort | MUTATION-EVIDENCE-FIELDS — `the wire field order must follow (version, authoring_set_order, section_id, key_name)` |
/// | time-sample sort | MUTATION-EVIDENCE-SAMPLES — `the wire time-sample order must follow (version, section_id, time)`; with that one ALSO inert, MUTATION-EVIDENCE-SIDECAR-ORDER (3/3), which is the end-to-end witness for seam E-1(i) |
/// | orphan sort | MUTATION-EVIDENCE-ORPHANS — `the emitted orphan tail must be section_id-ascending, got [1, 2, 4, 3, 80, 84, 83, 82, 81]` |
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_produces_byte_identical_blobs_across_independent_assemblies() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;
    let evs = ext_store(&tv);

    let fixture = determinism_fixture();

    // Real blobs behind the fixture's content addresses.
    let blob_refs: Vec<(&[u8; 32], &[u8])> = fixture
        .ext_blobs
        .iter()
        .map(|(hash, payload)| (hash, payload.as_slice()))
        .collect();
    evs.store_values(&blob_refs)
        .await
        .expect("ext store_values must succeed");

    let oid = ls
        .create_layer(
            &session_id,
            "determinism-layer",
            "assembler-test-user",
            &fixture.state,
            evs.clone() as Arc<dyn ExternalValueStore>,
        )
        .await
        .expect("create_layer must succeed");

    // Two full assemblies of the same idle layer, each into its OWN sink. Each
    // one independently runs `read_object_snapshot` → `snapshot_to_object_state`
    // and therefore builds its own freshly-seeded maps.
    let store_a = InMemoryCheckpointStore::new();
    let outcome_a = expect_written(
        assemble_layer_checkpoint(
            tv.client(),
            tv.key_schema(),
            oid,
            &store_a,
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        )
        .await
        .expect("the first assembly must succeed against a populated idle layer"),
    );
    let store_b = InMemoryCheckpointStore::new();
    let outcome_b = expect_written(
        assemble_layer_checkpoint(
            tv.client(),
            tv.key_schema(),
            oid,
            &store_b,
            SERVICE_LABEL,
            &mut NoLeaseToKeep,
        )
        .await
        .expect("the second assembly must succeed against a populated idle layer"),
    );

    // The CUT first. If the two assemblies pinned different cuts the blobs
    // describe different states and are not comparable at all — that must be
    // diagnosed as a cut drift here rather than surfacing as a byte inequality
    // below. Nothing writes to the layer between the two calls, so this is
    // deterministic, not a timing assumption.
    assert_eq!(
        outcome_a.snap_seq, outcome_b.snap_seq,
        "the two assemblies must pin the SAME stable cut against an idle layer \
         ({} vs {}); with different cuts the blobs describe different states and \
         the byte comparison below would be meaningless",
        outcome_a.snap_seq, outcome_b.snap_seq
    );

    // Non-vacuity on the ext half before any byte comparison: an empty hash pool
    // would make the sidecar equality trivially true, and four of the nine
    // hashes must arrive via the TIME-SAMPLE half of the scan (seam E-1(i)).
    assert_eq!(
        outcome_a.ext_hash_count, 9,
        "the fixture must carry nine distinct ext hashes (five field-first, four \
         time-sample-first) through the real load path, or the sidecar \
         comparison is vacuous"
    );
    assert_eq!(
        outcome_b.ext_hash_count, outcome_a.ext_hash_count,
        "both assemblies must see the same ext-hash count"
    );

    let (delta_a, ompf_a, sidecar_a) = fetch_blob(&store_a, oid).await;
    let (_delta_b, ompf_b, sidecar_b) = fetch_blob(&store_b, oid).await;

    // LENGTH FIRST, and as its own assertion. A length divergence is a different
    // diagnosis from a content divergence: the FlatBuffers builder shares one
    // vtable between tables emitted with identical field-presence layouts and
    // widens offsets as the buffer grows, so a reordered field / time-sample /
    // struct-command sequence changes how many vtables are shared and how wide
    // the offsets are. `tests/fixtures/tail_lineage/README.md` records exactly
    // that for the pre-sort `base.ompf`. Read a red here as "emission order
    // moved", never as "a value differs".
    assert_eq!(
        ompf_a.len(),
        ompf_b.len(),
        "MUTATION-EVIDENCE-LENGTH: two independent assemblies of the same idle \
         layer produced base OMPF of DIFFERENT LENGTH ({} vs {} bytes). Length \
         moves with emission order via FlatBuffers vtable sharing and offset \
         widths — this is an ordering defect in `plan_full_content` / \
         `sections_in_dfs_order`, not a differing value",
        ompf_a.len(),
        ompf_b.len()
    );

    // The headline. Deliberately NOT weakened to semantic isomorphism: Task 3's
    // SR-9.2 duplicate-leader tests and Task 7's cross-replica prose are designed
    // on this byte basis, and downgrading here would make both unsound.
    let first_divergence = ompf_a
        .iter()
        .zip(ompf_b.iter())
        .position(|(a, b)| a != b)
        .map_or("none".to_string(), |i| i.to_string());
    assert_eq!(
        ompf_a, ompf_b,
        "MUTATION-EVIDENCE-CONTENT: two independent assemblies of the same idle \
         layer produced base OMPF that diverged at EQUAL length (first differing \
         byte index: {first_divergence}) — one of `plan_full_content`'s three \
         emitted vectors is still HashMap-derived, or a fourth un-ordered vector \
         reached `build_delta`. Decode both with `OmpfContainer::parse` + \
         `oodf_ffi::parse_delta` and report the exact divergence; do NOT downgrade \
         this to an isomorphism assertion"
    );

    // The sidecar, on RAW BYTES off `CheckpointMeta::sidecar_bytes` — never on
    // the set-semantic `read_ext_refs_sidecar` observable, which by construction
    // cannot witness the order b.vz7 is about.
    assert_eq!(
        sidecar_a, sidecar_b,
        "MUTATION-EVIDENCE-SIDECAR: the ext-refs sidecar JSON must be \
         byte-identical across two independent assemblies of the same idle layer. \
         The sidecar's first-appearance hash order walks the emitted field order \
         then the emitted time-sample order, so a red here is an emission-order \
         defect in `plan_full_content`"
    );

    // `written_at_unix_ms` is `SystemTime::now()` at put time and is NOT
    // byte-deterministic. It is excluded from every comparison above — the two
    // `CheckpointMeta`s are never compared as wholes, only `sidecar_bytes` is.
    // Asserting only that both carry a real stamp documents WHY they are
    // excluded, without asserting anything about the clock (they may legitimately
    // land in the same millisecond).
    let meta_a = store_a
        .get_latest_layer_checkpoint(oid)
        .await
        .expect("get must succeed")
        .expect("assembly A must have persisted a generation")
        .meta
        .clone();
    let meta_b = store_b
        .get_latest_layer_checkpoint(oid)
        .await
        .expect("get must succeed")
        .expect("assembly B must have persisted a generation")
        .meta
        .clone();
    assert!(
        meta_a.written_at_unix_ms > 0 && meta_b.written_at_unix_ms > 0,
        "both generations must carry a real wall-clock stamp — the field that is \
         deliberately kept OUT of the byte comparison above"
    );

    // ── Non-vacuity: the bytes really carry the fixture ──────────────────────
    //
    // Without this block a future change that emptied the fixture, or a version
    // filter that dropped everything, would leave both equalities trivially true.
    // It also witnesses the claim the Task actually makes — that the PLAN order
    // propagates through the real store round-trip to the BYTES.
    let emitted_sections: Vec<u64> = delta_a
        .struct_commands
        .iter()
        .map(|c| match c {
            StructCommand::DiffChangeSection(s) => s.section_id,
            other => panic!("expected DiffChangeSection, got {other:?}"),
        })
        .collect();
    assert_eq!(
        emitted_sections.len(),
        9,
        "4 reachable + 5 orphan sections must survive the load and reach the \
         wire; got {emitted_sections:?}"
    );
    // E-1(ii): the ORPHAN TAIL is the half that gained an order. E-5: the
    // reachable DFS prefix MUST NOT move, so nothing new is asserted about it
    // beyond the parent-before-child shape existing coverage already carries.
    let (prefix, tail) = emitted_sections.split_at(4);
    assert_eq!(
        tail,
        [80, 81, 82, 83, 84],
        "MUTATION-EVIDENCE-ORPHANS: the emitted orphan tail must be \
         section_id-ascending, got {emitted_sections:?}"
    );
    let mut prefix_sorted = prefix.to_vec();
    prefix_sorted.sort_unstable();
    assert_eq!(
        prefix_sorted,
        vec![1, 2, 3, 4],
        "every reachable section must precede every orphan, got {emitted_sections:?}"
    );
    let pos = |id: u64| prefix.iter().position(|&x| x == id).unwrap();
    assert!(
        pos(1) < pos(2) && pos(1) < pos(3) && pos(2) < pos(4),
        "the reachable prefix must stay parent-before-child, got {prefix:?}"
    );

    let emitted_fields: Vec<(u64, &str)> = delta_a
        .set_fields
        .iter()
        .map(|f| (f.section_id, f.key_name.as_str()))
        .collect();
    assert_eq!(
        emitted_fields,
        vec![
            (1, "early"),
            (1, "e_c"),
            (1, "e_d"),
            (2, "e_a"),
            (3, "e_b"),
            (4, "s4_k"),
            (3, "later"),
        ],
        "MUTATION-EVIDENCE-FIELDS: the wire field order must follow (version, \
         authoring_set_order, section_id, key_name)"
    );
    // NON-VACUITY (d) — the wire `set_order` slot carries the writer's AUTHORING
    // rank, not the cumulative cursor (the 900s), and that remap must survive
    // both the sort and the Valkey round-trip. The fixture's two ranks differ on
    // every field, so this cannot pass by coincidence.
    assert_eq!(
        delta_a
            .set_fields
            .iter()
            .map(|f| f.set_order)
            .collect::<Vec<u64>>(),
        vec![2, 1, 1, 1, 1, 1, 0],
        "the assembler's set_order = authoring_set_order remap must survive the \
         sort and the store round-trip — the cumulative cursors (900..907) must \
         not reach the wire"
    );

    let emitted_samples: Vec<(u64, f64)> = delta_a
        .time_samples
        .iter()
        .map(|ts| (ts.section_id, ts.time))
        .collect();
    assert_eq!(
        emitted_samples,
        vec![
            (2, 1.0),
            (2, 2.0),
            (2, 3.0),
            (2, 4.0),
            (3, 2.0),
            (3, 2.5),
            (4, 0.5),
        ],
        "MUTATION-EVIDENCE-SAMPLES: the wire time-sample order must follow \
         (version, section_id, time)"
    );

    // The sidecar's concrete hash ORDER, as a VECTOR — so a red names the order
    // rather than leaving it an opaque inequality between two blobs. `a2` must
    // appear exactly once despite being referenced from both a field and a later
    // time sample (first-appearance dedup).
    let sidecar: SidecarShape = serde_json::from_slice(&sidecar_a)
        .expect("sidecar must parse as the documented JSON shape");
    assert_eq!(
        sidecar.ext_hashes, fixture.expected_hash_order,
        "MUTATION-EVIDENCE-SIDECAR-ORDER: the sidecar's hash order must follow \
         the emitted field order, then the emitted time-sample order"
    );
}

// =============================================================================
// b.hka (c) — the assembler declines rather than writing an identity-empty
// generation
// =============================================================================
//
// Before b.hka the identity read at step 6.5 folded through
// `identity_read.unwrap_or_default()`: an `obj:hdr` that had vanished, or one
// whose `session_id` / `layer_name` columns were absent, produced a
// `CheckpointMeta` with EMPTY identity and the blob was written anyway.
//
// That blob is permanently unrestorable and silently so. The apply path's
// identity guard (`checkpoint::apply::live_mapping_resolves_blob_identity`)
// refuses a generation whose identity it cannot verify against the live
// keyspace — it cannot rebuild `obj:hdr` without inventing session metadata,
// and inventing it is how a restore lands a layer under the wrong session. So
// the write cost a PUT, occupied a retention slot a restorable generation could
// have held, and produced an artifact whose only possible future was to fail a
// restore.
//
// `NotFound` is the right refusal shape rather than a new outcome: it is the arm
// the checkpoint loop already treats as benign and terminal (silent `debug!`, no
// cursor stamp, no metric tick), which is exactly the truth here — the object is
// gone, or its header is unusable, and it will not be enumerated again.
//
// ## Which arm each case reaches, and why the split
//
// Since b.hka review round 2 the assembler refuses on ONE condition at ONE site:
// `read_header_identity` returning `None`. That single answer now covers both
// header states — the key is gone, and the key is present with an empty
// `session_id` / `layer_name` — because the loader takes the decision, and the
// assembler's own second empty-column `if` was deleted as a duplicate. Moving it
// there is what makes the rule structural: the tail-capture and bootstrap
// producers share that read and previously had no such check at all, so a
// degenerate header reached `validate_generation_identity` and ticked
// adapter-health every tick. (The loader-level translation is pinned in
// `src/checkpoint/assembler.rs`'s own `mod tests`; the sibling producers' arms in
// `reconcile_dead_object_test.rs` and `first_base_bootstrap_test.rs`.)
//
// Of the two header states only the empty column is deterministically reachable
// from an integration test — the vanished header is a genuine race whose window
// is the microseconds between the proven state read and the identity read, and
// forcing it there would need a seam the assembler does not expose
// (`assemble_inner` builds its own `ValkeySnapshotLoader`; the `FakeLoader` in
// the module's own `mod tests` reaches only `load_consistent_snapshot`). So the
// empty-column cases are the regression tests, and the vanished-header case below
// asserts the same observable — `NotFound`, zero puts — reaching it through the
// PIN read instead. Both halves of the claim that matters are covered: no
// assembly ever writes an identity-empty generation, and a dead object costs the
// store nothing.

/// The count of generations an `InMemoryCheckpointStore` holds for `oid`.
///
/// "Zero puts" is read off the adapter's own listing rather than a call counter:
/// a generation that was put is listed and one that was not is not, and a
/// counting decorator would need all eighteen trait methods to observe strictly
/// less.
async fn generation_count(store: &InMemoryCheckpointStore, oid: u64) -> usize {
    store
        .list_layer_checkpoints(oid)
        .await
        .expect("listing an in-memory store cannot fail")
        .len()
}

/// An `obj:hdr` that exists and carries `seq` — so the pin read and the
/// proven state read both succeed — but whose `session_id` column is gone is
/// REFUSED, and nothing is written.
///
/// This is the b.hka (c) regression test. Pre-fix, `read_header_identity`
/// returned `Some(HeaderIdentity { session_id: "", .. })` (the loader folded an
/// absent field to `String::new()`), `unwrap_or_default()` was a no-op on it,
/// and the assembler PUT a generation carrying empty identity — `Ok(Written)`
/// with one generation in the store. Post-fix it is `Err(NotFound)` with none.
///
/// The refusal MOVED in review round 2 and the observable did not: the loader
/// now answers `Ok(None)` for a degenerate header, so this lands on the same
/// `NotFound` early return the vanished-header case does, rather than on a
/// second empty-column check local to this producer. That relocation is the
/// whole point — the check being local is what left the other three producers
/// without one — and this case is the fence that it did not change what an
/// identity-stripped layer costs the store.
///
/// The `HDEL` is the honest fixture for a partially-written or hand-repaired
/// header, which is the only way an `obj:hdr` that still carries `seq` can lack
/// its identity columns.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_declines_a_header_whose_session_id_column_is_absent() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;

    let oid = ls
        .create_layer(
            &session_id,
            "identity-stripped-layer",
            "assembler-test-user",
            &populated_state(),
            ext_store(&tv),
        )
        .await
        .expect("create_layer must succeed");

    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let removed: u64 = tv
        .client()
        .pool
        .hdel(&header_key, "session_id")
        .await
        .expect("HDEL session_id must succeed");
    assert_eq!(
        removed, 1,
        "fixture precondition: the column must have been there to remove — if `obj:hdr` stopped \
         carrying `session_id` this case would pass vacuously against a header that never had it"
    );
    let still_live: bool = tv
        .client()
        .pool
        .hexists(&header_key, "seq")
        .await
        .expect("HEXISTS seq must succeed");
    assert!(
        still_live,
        "fixture precondition: `seq` must survive, or the assembler refuses at the PIN read \
         instead and this case would be testing the vanished-header arm by accident"
    );

    let store = InMemoryCheckpointStore::new();
    let err = assemble_layer_checkpoint(
        tv.client(),
        tv.key_schema(),
        oid,
        &store,
        SERVICE_LABEL,
        &mut NoLeaseToKeep,
    )
    .await
    .expect_err(
        "an identity-empty generation is permanently unrestorable — the apply path's identity \
         guard refuses it — so the assembler must decline rather than spend a PUT and a \
         retention slot on it",
    );

    assert!(
        matches!(err, live_edit::checkpoint::CheckpointError::NotFound),
        "`NotFound` specifically: it is the arm the checkpoint loop already treats as benign and \
         terminal (silent `debug!`, no cursor stamp, no metric tick), so declining here mints no \
         new outcome and never feeds the adapter-health alert from a healthy adapter. Got {err:?}"
    );
    assert_eq!(
        generation_count(&store, oid).await,
        0,
        "and the store MUST NOT have been called: the whole point is that no unrestorable \
         artifact is written. Pre-b.hka this listed ONE generation, carrying empty identity"
    );
}

/// The same refusal when `layer_name` is the absent column.
///
/// Asserted separately rather than folded in: the production check is a
/// two-clause `||`, and a fix that read only `session_id` would pass the case
/// above and write an unrestorable blob for every layer whose name column was
/// lost. The apply path requires BOTH — it cannot rebuild `obj:hdr` knowing the
/// session but not which layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_declines_a_header_whose_layer_name_column_is_absent() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;

    let oid = ls
        .create_layer(
            &session_id,
            "layer-name-stripped-layer",
            "assembler-test-user",
            &populated_state(),
            ext_store(&tv),
        )
        .await
        .expect("create_layer must succeed");

    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let removed: u64 = tv
        .client()
        .pool
        .hdel(&header_key, "layer_name")
        .await
        .expect("HDEL layer_name must succeed");
    assert_eq!(removed, 1, "fixture precondition: the column was there");

    let store = InMemoryCheckpointStore::new();
    let err = assemble_layer_checkpoint(
        tv.client(),
        tv.key_schema(),
        oid,
        &store,
        SERVICE_LABEL,
        &mut NoLeaseToKeep,
    )
    .await
    .expect_err("an absent `layer_name` is as unrestorable as an absent `session_id`");

    assert!(
        matches!(err, live_edit::checkpoint::CheckpointError::NotFound),
        "got {err:?}"
    );
    assert_eq!(generation_count(&store, oid).await, 0);
}

/// A vanished `obj:hdr` costs the store nothing.
///
/// The DeleteLayer / `cleanup_old_object` / orphan-sweep race, in its
/// deterministic form: the header is gone before the call rather than during it,
/// so the refusal lands at the PIN read rather than at the identity read. That
/// arm predates b.hka, so this is a FENCE and not a regression test — but it is
/// the fence for the claim the (c) fix is really making, which is about the
/// store and not about which line refused: *no* arm of the assembler writes a
/// generation for an object whose header is not there.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn assembler_writes_nothing_when_the_object_header_is_gone() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_session(&tv).await;

    let oid = ls
        .create_layer(
            &session_id,
            "vanished-header-layer",
            "assembler-test-user",
            &populated_state(),
            ext_store(&tv),
        )
        .await
        .expect("create_layer must succeed");

    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let deleted: u64 = tv
        .client()
        .pool
        .del(&header_key)
        .await
        .expect("DEL obj:hdr must succeed");
    assert_eq!(deleted, 1, "fixture precondition: the header was there");

    let store = InMemoryCheckpointStore::new();
    let err = assemble_layer_checkpoint(
        tv.client(),
        tv.key_schema(),
        oid,
        &store,
        SERVICE_LABEL,
        &mut NoLeaseToKeep,
    )
    .await
    .expect_err("a vanished header is a refusal, not an assembly");

    assert!(
        matches!(err, live_edit::checkpoint::CheckpointError::NotFound),
        "got {err:?}"
    );
    assert_eq!(
        generation_count(&store, oid).await,
        0,
        "an object that will never be enumerated again must cost the checkpoint store nothing"
    );
}

// =============================================================================
// Internal helpers
// =============================================================================

/// SHA-256 of a single byte slice — used to compute hashes that match the
/// `ValkeyExternalValueStore` content-addressed key shape.
fn sha256(bytes: &[u8]) -> [u8; 32] {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}
