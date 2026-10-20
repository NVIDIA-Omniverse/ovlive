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

// Task t2.fgj.ya.9j Subtask .s2 — end-to-end loop against LocalDisk.
//
// Walking-skeleton coverage of the full SR-2.1 / SR-2.2 phase chain against
// the production `LocalDiskCheckpointStore`:
//
//   1. Spawn `spawn_checkpoint_loop` (the leader-elected periodic task).
//   2. The loop's per-tick body acquires `le:checkpoint:leader` (SR-2.1).
//   3. It SCANs every `le:{*}:obj:hdr` and enumerates per-layer state.
//   4. It runs the assembler (SR-2.3) for layers past the skip-if-no-commits
//      and quiesce/interval gates.
//   5. The assembler routes through `put_layer_checkpoint`, which on
//      `LocalDiskCheckpointStore` lands a `.ompf`, `.ext-refs.json`, and
//      `.meta.json` triple under `<root>/.live-edit/layers/`.
//   6. Retention prune keeps only the `retain_generations` newest per layer.
//
// Each sibling per-phase test substitutes a stub or mock for the surrounding
// phases (e.g. `checkpointer_leader_race_test` uses a counting in-memory
// store; `checkpointer_retention_test` runs only one layer); this test
// exercises every phase together against the real disk-backed adapter, so a
// phase-composition regression that no per-phase test catches surfaces here.
//
// Coordination notes (see subtask body):
//   - LocalDisk now persists `.meta.json` alongside the OMPF + ext-refs
//     sidecar (Task n9 follow-up). At `retain_generations=2` the steady
//     state is 2 of each per layer = 6 files per layer = 18 across 3 layers.
//     The pre-prune transient may briefly exceed 2 per layer, so we cancel
//     the loop before asserting file counts to read a quiescent state.
//   - `writes_total{result=*, kind="layer"}` is read off a `SharedMetrics`
//     constructed locally and threaded into `spawn_checkpoint_loop`; the
//     sibling `.by` (metrics test) uses the same `Metrics::new(...)`
//     + `.with_label_values(...).get()` pattern.
//   - All polling is bounded with `tokio::time::timeout` against an explicit
//     deadline per CLAUDE.md; no naked `sleep` loops.
//
// Sandbox env note: this test uses `TestValkey::setup`, which reads
// `TEST_VALKEY_URL`. Under the project sandbox set the URL to
// `redis://:${VALKEY_PASSWORD}@host.docker.internal:6379` so the integration
// test can reach the dockerised Valkey.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{ClientLike, HashesInterface};
use tempfile::TempDir;
use tokio::time::{sleep, timeout};
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::{
    spawn_checkpoint_loop, CheckpointLoopConfig, CheckpointStore, TailCaptureConfig,
};
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section};
use live_edit::observability::metrics::{Metrics, SharedMetrics};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::OwnedDelta;

const RETAIN_GENERATIONS: usize = 2;
const SEEDED_LAYERS: usize = 3;
const UPDATES_PER_LAYER: usize = 2;
const REPLICA_ID: &str = "e2e-test-replica";

/// One distinct seed configuration per layer so the assembler walks
/// non-trivial state. Section + field shapes vary across layers so the
/// per-layer OMPF decode confirms each layer gets its own bytes rather
/// than a single cached blob.
fn seed_state(seed: u8) -> ObjectState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 0,
        ..Default::default()
    };
    // Two sections per layer at the root: sids `(seed*10+1, seed*10+2)`.
    let sec_a = u64::from(seed) * 10 + 1;
    let sec_b = u64::from(seed) * 10 + 2;
    state.sections.insert(
        sec_a,
        Section {
            section_id: sec_a,
            parent_id: 0,
            name: format!("layer-{seed}-sec-a"),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        },
    );
    state.sections.insert(
        sec_b,
        Section {
            section_id: sec_b,
            parent_id: 0,
            name: format!("layer-{seed}-sec-b"),
            section_type: 1,
            section_order: 1,
            first_version: 1,
            last_version: 1,
        },
    );
    state.children.insert(0, vec![sec_a, sec_b]);

    // One inline field on each section so the OMPF decode has SetField rows
    // to count and the ext-refs sidecar stays empty (`Inline` values never
    // promote to the ext store).
    for sid in [sec_a, sec_b] {
        let key = format!("k{sid}");
        state.fields.insert(
            (sid, key.clone()),
            Field {
                section_id: sid,
                key_name: key,
                value: FieldValue::Inline(vec![seed, sid as u8, 0xAA]),
                set_order: u64::from(seed),
                authoring_set_order: u64::from(seed),
                version: 1,
            },
        );
    }

    state
}

/// Construct a `SharedMetrics` over a fresh `Registry` so the test owns the
/// counters end-to-end. The sibling `.by` (metrics test) uses the same shape.
fn make_metrics() -> SharedMetrics {
    Arc::new(crate::common::metrics_scrape_only(REPLICA_ID))
}

/// Read `live_edit_checkpoint_writes_total{result=<result>, kind="layer"}`
/// off the supplied metrics handle. Returns a `u64` (the underlying counter
/// is `f64` but it's only ever bumped by 1.0 per call).
fn read_writes(metrics: &Metrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_writes_total",
        &[("result", result), ("kind", "layer")],
    )
    .unwrap_or(0.0) as u64
}

/// Read `last_checkpoint_seq` off `le:{oid}:obj:hdr`. Returns `None` when
/// the field is absent (e.g. the loop has not yet written for this oid).
async fn read_last_checkpoint_seq(tv: &TestValkey, object_id: u64) -> Option<u64> {
    let key = tv.key_schema().obj_header(&object_id.to_string());
    let raw: Option<String> = tv
        .client()
        .pool
        .hget(&key, "last_checkpoint_seq")
        .await
        .expect("HGET last_checkpoint_seq must succeed");
    raw.and_then(|s| s.parse::<u64>().ok())
}

/// Bounded-poll until `last_checkpoint_seq` for every supplied object_id is
/// `Some(v)` with `v >= min`. Returns the per-object observed cursors.
/// Bounded to 10 s to keep the runtime budget for the test under the
/// per-AC 10 s ceiling.
async fn wait_for_cursors_at_least(
    tv: &TestValkey,
    object_ids: &[u64],
    min: u64,
) -> HashMap<u64, u64> {
    let deadline = Duration::from_secs(10);
    let result = timeout(deadline, async {
        loop {
            let mut observed: HashMap<u64, u64> = HashMap::new();
            for &oid in object_ids {
                if let Some(cursor) = read_last_checkpoint_seq(tv, oid).await {
                    if cursor >= min {
                        observed.insert(oid, cursor);
                    }
                }
            }
            if observed.len() == object_ids.len() {
                return observed;
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await;
    result.unwrap_or_else(|_| {
        panic!(
            "timed out waiting for last_checkpoint_seq>={min} on every object_id \
             in {object_ids:?} within {deadline:?}"
        )
    })
}

/// Inventory of files under `<root>/.live-edit/layers/`, bucketed by the
/// `<oid>.<snap_seq>.<suffix>` filename triple. The map is keyed by
/// object_id; each value is the list of suffixes observed for that oid,
/// in arbitrary order. We use this to assert exactly N OMPF + N sidecar
/// + N meta files per layer remain after retention prune.
fn inventory_layers_dir(layers_dir: &std::path::Path) -> HashMap<u64, Vec<String>> {
    let mut by_oid: HashMap<u64, Vec<String>> = HashMap::new();
    let read = std::fs::read_dir(layers_dir).expect("read_dir layers must succeed");
    for entry in read {
        let entry = entry.expect("dir entry must succeed");
        let name = entry.file_name().to_string_lossy().into_owned();
        // Filenames are `<oid>.<snap_seq>.<suffix>` where `<suffix>` is one
        // of `ompf`, `ext-refs.json`, `meta.json`. Split on the leading
        // dot pair; the suffix may itself contain a dot
        // (e.g. `ext-refs.json`).
        let mut parts = name.splitn(3, '.');
        let oid_str = parts.next().unwrap_or("");
        let _snap_str = parts.next().unwrap_or("");
        let suffix = parts.next().unwrap_or("");
        if let Ok(oid) = oid_str.parse::<u64>() {
            by_oid.entry(oid).or_default().push(suffix.to_string());
        }
    }
    by_oid
}

/// Decode an `<oid>.<snap_seq>.ompf` file: OMPF parse → OODF delta parse.
/// Returns `(struct_command_count, set_field_count, time_sample_count)` so
/// the test can pin the round-trip count without recomputing the seed
/// shape inline.
fn decode_ompf_counts(path: &std::path::Path) -> (usize, usize, usize) {
    let bytes = std::fs::read(path).expect("OMPF file read must succeed");
    let container = OmpfContainer::parse(&bytes).expect("OMPF parse must succeed");
    let delta = oodf_ffi::parse_delta(container.delta_bytes)
        .expect("OODF FFI delta parse must succeed on assembler-emitted bytes");
    (
        delta.struct_commands.len(),
        delta.set_fields.len(),
        delta.time_samples.len(),
    )
}

/// Parsed shape of the `<oid>.<snap_seq>.ext-refs.json` sidecar. The
/// assembler always emits `{"ext_hashes": [...]}` even when the list is
/// empty (no Inline-only fields reference an ext value).
#[derive(serde::Deserialize, Debug)]
struct ExtRefsSidecar {
    ext_hashes: Vec<String>,
}

/// Drive a single end-to-end run against the supplied `TestValkey` fixture.
/// Extracted so the single-node and cluster entrypoints below can share
/// the body without duplicating the seed / spawn / assert sequence.
async fn run_end_to_end(tv: TestValkey) {
    // --- Seed: 3 sessions × 1 layer each = 3 distinct object_ids. ---------
    // Putting each layer in its own session keeps the layer-name namespace
    // independent so the loop's SCAN sees three unambiguous `obj:hdr` keys.
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let mock_ext = Arc::new(MockExternalValueStore::new());

    let mut object_ids: Vec<u64> = Vec::with_capacity(SEEDED_LAYERS);
    let mut session_ids: Vec<String> = Vec::with_capacity(SEEDED_LAYERS);
    let mut layer_names: Vec<String> = Vec::with_capacity(SEEDED_LAYERS);
    for i in 0..SEEDED_LAYERS {
        let session_id = session_store
            .create(
                &format!("e2e-session-{i}"),
                &format!("/stage/e2e-{i}"),
                "e2e-test-user",
            )
            .await
            .expect("session create must succeed");
        let layer_name = format!("e2e-layer-{i}");
        let state = seed_state(i as u8);
        let oid = layer_store
            .create_layer(
                &session_id,
                &layer_name,
                "e2e-test-user",
                &state,
                mock_ext.clone(),
            )
            .await
            .expect("create_layer must succeed");

        // Two follow-on UpdateLayers per layer ratchet `obj:hdr seq`
        // (3, 5 — initial create stamps seq=1 then each empty-delta update
        // advances by 2). Empty `OwnedDelta::default()` is a no-op on
        // state but still bumps seq and XADDs to the per-object stream.
        for u in 0..UPDATES_PER_LAYER {
            layer_store
                .update_layer_relayed(
                    &session_id,
                    &layer_name,
                    oid,
                    Some(OwnedDelta::default()),
                    "e2e-test-client",
                    1_000 + u as u64,
                )
                .await
                .expect("update_layer must succeed");
        }

        object_ids.push(oid);
        session_ids.push(session_id);
        layer_names.push(layer_name);
    }

    // --- Adapter: LocalDisk under a tempdir scoped to this test run. ------
    // `TempDir` is RAII-cleaned on drop at the end of the test, so a panic
    // mid-test still scrubs the on-disk garbage.
    let root: TempDir = tempfile::tempdir().expect("tempdir creation must succeed");
    let store: Arc<dyn CheckpointStore> = Arc::new(
        LocalDiskCheckpointStore::new(root.path().to_path_buf())
            .expect("LocalDiskCheckpointStore::new must succeed on a fresh root"),
    );

    // --- Spawn the loop. tick=200ms; interval=0 + quiesce=0 disables the --
    // skip-if-idle gates so every commit is immediately due. retain=2 caps
    // the steady-state at exactly two generations per layer.
    let metrics = make_metrics();
    let loop_config = CheckpointLoopConfig {
        tick: Duration::from_millis(200),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: RETAIN_GENERATIONS,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(REPLICA_ID),
        // Tails pinned explicitly OFF. b.3n1 promoted the binary default
        // to `true`; this suite predates the tail feature and was written
        // against an inert tail path, so the flag is part of the fixture
        // rather than something to inherit.
        tail: TailCaptureConfig {
            enabled: false,
            ..TailCaptureConfig::default()
        },
    };
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        tv.client().clone(),
        tv.key_schema().clone(),
        Arc::clone(&store),
        loop_config,
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    // --- First wave: wait until every layer's last_checkpoint_seq is set. -
    // After 2 UpdateLayers on top of CreateLayer the live seq is at least 3
    // (CreateLayer + 2 empty deltas); the loop's post-checkpoint HSET pins
    // `last_checkpoint_seq` to the snap_seq at assembly time. The exact
    // value depends on race-with-the-loop timing, so we only assert "set
    // and >= 1" rather than equality to any specific seq.
    let first_cursors = wait_for_cursors_at_least(&tv, &object_ids, 1).await;
    for &oid in &object_ids {
        assert!(
            first_cursors.contains_key(&oid),
            "first wave must observe last_checkpoint_seq for object_id={oid}; \
             cursors={first_cursors:?}"
        );
    }

    // --- Force a second-wave checkpoint via one more UpdateLayer per oid. -
    // After the first wave the skip-if-no-commits gate would block further
    // checkpoints until the live seq advances. One more empty delta bumps
    // seq past last_checkpoint_seq → next tick re-checkpoints → ≥2
    // generations per layer once the loop tickles them.
    for (i, &oid) in object_ids.iter().enumerate() {
        layer_store
            .update_layer_relayed(
                &session_ids[i],
                &layer_names[i],
                oid,
                Some(OwnedDelta::default()),
                "e2e-test-client",
                2_000_u64,
            )
            .await
            .expect("second-wave update_layer must succeed");
    }

    // Wait for the cursor to advance past the first-wave value on every
    // layer. The cursor strictly increases per checkpoint, so polling on
    // `cursor > first_wave_cursor` is the right "second wave landed"
    // signal.
    let second_deadline = Duration::from_secs(10);
    let second_cursors = timeout(second_deadline, async {
        loop {
            let mut all_advanced = true;
            let mut observed: HashMap<u64, u64> = HashMap::new();
            for &oid in &object_ids {
                let prev = first_cursors[&oid];
                match read_last_checkpoint_seq(&tv, oid).await {
                    Some(cur) if cur > prev => {
                        observed.insert(oid, cur);
                    }
                    _ => {
                        all_advanced = false;
                        break;
                    }
                }
            }
            if all_advanced {
                return observed;
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "second wave did not advance last_checkpoint_seq past the first-wave \
             values within {second_deadline:?}; first={first_cursors:?}"
        )
    });
    for &oid in &object_ids {
        assert!(
            second_cursors[&oid] > first_cursors[&oid],
            "second-wave cursor must strictly exceed first-wave cursor for oid={oid}; \
             first={}, second={}",
            first_cursors[&oid],
            second_cursors[&oid]
        );
    }

    // --- Cancel + bounded-join. We assert on the QUIESCENT post-cancel ----
    // state so a late-firing tick can't perturb the count assertions
    // below — matches the discipline in `checkpointer_retention_test`.
    cancel.cancel();
    timeout(Duration::from_secs(5), handle)
        .await
        .expect("checkpoint loop must exit within 5s of cancellation")
        .expect("checkpoint loop must not panic on shutdown");

    // --- Leader-lock contract: SR-2.1 requires the loop to have acquired --
    // `le:checkpoint:leader` before any per-layer write. The single-replica
    // case must produce a `last_checkpoint_seq` HSET per layer, which by
    // the loop's logic only fires inside the `Ok(true)` arm of
    // `try_acquire_leader_lock`. The per-layer cursor presence asserted
    // above is the observable proof that the leader-lock branch ran.
    for &oid in &object_ids {
        let final_cursor = read_last_checkpoint_seq(&tv, oid).await.unwrap_or_else(|| {
            panic!(
                "post-cancel: last_checkpoint_seq must be present for oid={oid} \
                         (its presence proves the leader-lock branch ran at least once)"
            )
        });
        assert!(
            final_cursor >= second_cursors[&oid],
            "post-cancel cursor must be >= the observed second-wave cursor; \
             oid={oid}, observed={}, post_cancel={final_cursor}",
            second_cursors[&oid]
        );
    }

    // --- Retention: with retain_generations=2 every layer has exactly 2 ---
    // OMPF + 2 ext-refs.json + 2 meta.json files on disk. The pre-prune
    // transient is gone because we cancelled the loop before reading the
    // file system.
    let layers_dir = root.path().join(".live-edit").join("layers");
    let by_oid = inventory_layers_dir(&layers_dir);
    assert_eq!(
        by_oid.len(),
        SEEDED_LAYERS,
        "exactly {SEEDED_LAYERS} object_ids must have files on disk, found {}: {by_oid:?}",
        by_oid.len()
    );
    for &oid in &object_ids {
        let suffixes = by_oid
            .get(&oid)
            .unwrap_or_else(|| panic!("oid={oid} must have files on disk; got {by_oid:?}"));
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for s in suffixes {
            *counts.entry(s.as_str()).or_insert(0) += 1;
        }
        assert_eq!(
            counts.get("ompf").copied().unwrap_or(0),
            RETAIN_GENERATIONS,
            "oid={oid}: retain_generations={RETAIN_GENERATIONS} must cap .ompf files; counts={counts:?}"
        );
        assert_eq!(
            counts.get("ext-refs.json").copied().unwrap_or(0),
            RETAIN_GENERATIONS,
            "oid={oid}: retain_generations={RETAIN_GENERATIONS} must cap .ext-refs.json sidecars; \
             counts={counts:?}"
        );
        assert_eq!(
            counts.get("meta.json").copied().unwrap_or(0),
            RETAIN_GENERATIONS,
            "oid={oid}: retain_generations={RETAIN_GENERATIONS} must cap .meta.json sidecars \
             (Task n9 added meta-sidecar parity to the prune path); counts={counts:?}"
        );
    }
    // Total file count across all 3 layers: 3 layers × 2 gens × 3 file types = 18.
    let total_files: usize = by_oid.values().map(|v| v.len()).sum();
    assert_eq!(
        total_files,
        SEEDED_LAYERS * RETAIN_GENERATIONS * 3,
        "steady-state file count must be SEEDED_LAYERS({SEEDED_LAYERS}) × \
         RETAIN_GENERATIONS({RETAIN_GENERATIONS}) × 3-file-types = {} files; got {total_files}",
        SEEDED_LAYERS * RETAIN_GENERATIONS * 3
    );

    // --- Every OMPF round-trips through OODF-FFI decode. -------------------
    // Each retained OMPF must contain at least the 2 sections + 2 fields
    // from the seed state (the inserted second-wave deltas are empty, so
    // the snapshot count never decreases). We pin the lower bound to catch
    // a regression that writes truncated / corrupt bytes for some layers.
    let mut decoded_oids: HashSet<u64> = HashSet::new();
    for entry in std::fs::read_dir(&layers_dir).expect("read_dir layers must succeed") {
        let entry = entry.expect("entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".ompf") {
            continue;
        }
        let path = entry.path();
        let (sc, fields, ts) = decode_ompf_counts(&path);
        let oid: u64 = name
            .split('.')
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("could not parse oid from filename {name}"));
        // The reset-delta emitted by the assembler routes through
        // `DiffEngine::plan` → FullContent, which produces a CreateSection
        // struct command per section and a SetField per field. Time
        // samples count 0 because the seed state inserts none.
        assert!(
            sc >= 2,
            "decoded delta from {name} must carry >=2 struct_commands (one per seeded \
             section); got struct_commands={sc}"
        );
        assert!(
            fields >= 2,
            "decoded delta from {name} must carry >=2 set_fields (one per seeded field); \
             got set_fields={fields}"
        );
        assert_eq!(
            ts, 0,
            "decoded delta from {name} must carry 0 time_samples (seed state inserted none); \
             got time_samples={ts}"
        );
        decoded_oids.insert(oid);
    }
    assert_eq!(
        decoded_oids.len(),
        SEEDED_LAYERS,
        "every seeded oid must contribute at least one decoded OMPF; got {decoded_oids:?}"
    );

    // --- Every ext-refs.json parses as `{"ext_hashes": [...]}`. -----------
    // The Inline-only seed state produces an empty `ext_hashes` array on
    // every layer; the assertion is on shape (parses as the expected
    // struct) rather than content (which is always `[]` here).
    for entry in std::fs::read_dir(&layers_dir).expect("read_dir layers") {
        let entry = entry.expect("entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".ext-refs.json") {
            continue;
        }
        let bytes = std::fs::read(entry.path()).expect("sidecar file read must succeed");
        let parsed: ExtRefsSidecar = serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            panic!("ext-refs sidecar {name} must parse as {{ext_hashes: [...]}}; got error {e}")
        });
        // Inline-only seed never references an ext value.
        assert!(
            parsed.ext_hashes.is_empty(),
            "sidecar {name} must have empty ext_hashes for Inline-only seed state; \
             got ext_hashes={:?}",
            parsed.ext_hashes
        );
    }

    // --- Metrics: success counter advanced; error counter still zero. -----
    // Lower-bound on `success`: every retained generation came from a
    // `result="success"` increment in the loop. Across SEEDED_LAYERS × 2
    // observed waves we expect at least SEEDED_LAYERS successes (the
    // pruned generations counted too — we don't model the exact number).
    let success_count = read_writes(&metrics, "success");
    let error_count = read_writes(&metrics, "error");
    assert!(
        success_count >= SEEDED_LAYERS as u64,
        "checkpoint_writes_total{{result=\"success\", kind=\"layer\"}} must be \
         >= {SEEDED_LAYERS} (at least one success per seeded layer); got {success_count}"
    );
    assert_eq!(
        error_count, 0,
        "checkpoint_writes_total{{result=\"error\", kind=\"layer\"}} must stay zero on the \
         happy path; got {error_count} — investigate any logged warn lines from the loop"
    );
}

// =============================================================================
// Single-node entrypoint — always runs.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn end_to_end_localdisk_three_layers_round_trip() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    run_end_to_end(tv).await;
}

// =============================================================================
// Cluster entrypoint — gated on LIVE_EDIT_TEST_CLUSTER_URL.
// =============================================================================
//
// The same body re-runs against a clustered Valkey. The loop's SCAN must
// route through `scan_cluster_buffered` (see
// `checkpointer::run_one_tick`) so all three shards' `obj:hdr` keys are
// visited; a regression that dropped the cluster branch would silently
// miss layers on sibling shards. We re-use the same assertion set — the
// post-prune file inventory must still equal SEEDED_LAYERS × 2 × 3.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn end_to_end_localdisk_three_layers_round_trip_cluster() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };
    let tv = TestValkey::setup_with_url(cluster_url, 8).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment for this test \
         to exercise the cluster SCAN branch"
    );
    run_end_to_end(tv).await;
}
