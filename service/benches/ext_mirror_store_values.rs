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

//! Criterion benchmark for the **composed** SR-3.1 call site:
//! `ValkeyExternalValueStore::store_values`, measured with the ext-mirror
//! wired and unwired.
//!
//! ## Report-only, by design
//!
//! This bench gates **nothing**. `tools/check_bench_thresholds.py` applies
//! its ceiling to `ext_mirror_enqueue/mirror_drained/4096` only; every id
//! produced here is reported informationally and never fails a pipeline.
//!
//! The reason is the same one that retired the in-tree timing assertion
//! this bench descends from: `store_values` is dominated by the data-pool
//! `EVALSHA` round trip, so the guarded ext-mirror effect (~1 µs of
//! byte charge + `try_send`) is a fraction of a percent of the signal. Any
//! threshold placed here would be a threshold on runner-to-Valkey network
//! latency and CI scheduling, not on this repository's code. The
//! precision instrument for the mirror itself is
//! `benches/ext_mirror_enqueue.rs`; the authoritative composed ±2 %
//! measurement is an internal release-build Python workload simulator.
//!
//! What this bench *is* good for: local triage. When the enqueue bench
//! shows a regression, running this one on the same machine answers "does
//! it survive composition with the real call site, or is it lost in the
//! RTT?" — and it keeps the real call site compiling and exercised so the
//! two arms cannot silently diverge from production wiring.
//!
//! One thing it no longer covers: post-b.rej `store_values` runs on the
//! shared data pool via `EVALSHA` and takes no `txn_pool` checkout, so
//! nothing measured here exercises the per-checkout `DISCARD`/`UNWATCH`
//! hygiene probe. See `src/storage/txn_checkout.rs` (the caveat under the
//! probe's cost model) — that probe's cost is arithmetic over a measured
//! per-command round trip, not something this bench can express as a delta.
//!
//! ## Fixtures
//!
//! One fixed set of [`BATCH_HASHES`] × [`PAYLOAD_BYTES`] content-addressed
//! blobs is generated once and re-stored on every iteration. That is a
//! deliberate memory bound, not a shortcut: the backing Valkey runs
//! `maxmemory-policy noeviction`, so a bench that minted fresh hashes per
//! iteration would grow the keyspace without limit for the whole
//! measurement window and could push a shared instance into OOM. Re-storing
//! the same hashes issues the same client-side command profile — the same
//! single `EVALSHA` per blob, the same payload bytes on the wire, the same
//! mirror enqueue — with a constant-size keyspace.
//!
//! It does NOT exercise the same server-side work as a first store. The
//! script's data write is `SET … NX`, so once the warm-up has stored each
//! hash every measured call takes the skip path: Valkey receives the payload
//! but writes no blob, and only the `HINCRBY` (so `ref_count` climbs, which
//! costs nothing) and the `last_store_ms` stamp run. The numbers are
//! therefore the re-upload-of-a-known-hash cost; a first store additionally
//! pays Valkey's allocation and copy of the blob.
//!
//! ## Logging
//!
//! Like the sibling bench, this one installs a global
//! `tracing_subscriber::fmt()` writing to `std::io::sink()`. Production runs
//! with a subscriber installed, so the `tracing` statements on the measured
//! path have a real formatting cost; with none installed they collapse to a
//! disabled-callsite check and the measurement quietly omits them. Writing
//! to a sink keeps stdio contention (and interleaving with criterion's own
//! output) out of the numbers while preserving that cost.
//!
//! ## Skipping
//!
//! Requires a reachable Valkey. When `TEST_VALKEY_URL` is unset the bench
//! prints a skip notice and returns without measuring, so `cargo bench` and
//! `cargo test --benches` stay green on a machine with no Valkey. CI's
//! `perf:ext-mirror-bench` job inherits `TEST_VALKEY_URL` from
//! `.rust-integration-base`, so the bench runs there.
//!
//! ## Running it
//!
//! ```text
//! cargo bench -p live-edit --bench ext_mirror_store_values --features test-support
//! ```

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{ext_mirror_channel, spawn_ext_mirror_workers, CheckpointStore};
use live_edit::observability::metrics::{Metrics, SharedMetrics};
use live_edit::storage::external_values::{
    ext_content_hash, ExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::test_support::TestValkey;

/// Blobs per `store_values` call. Matches the shape a real `CreateLayer`
/// produces: the OMPF's external-values list, a handful per call rather
/// than one giant batch. Also stays under the `STORE_VALUES_MAX_INFLIGHT`
/// (8) buffered-stream cap by only a factor of two, so the call exercises
/// the cap's queueing behaviour without spending the whole measurement
/// there.
const BATCH_HASHES: usize = 16;

/// Per-blob payload size. Held equal to the CI-gated size in
/// `benches/ext_mirror_enqueue.rs` so the two benches' numbers are
/// directly comparable when triaging.
const PAYLOAD_BYTES: usize = 4 * 1024;

/// Production-default mirror channel capacity
/// (`LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY`).
const DEFAULT_MIRROR_CAPACITY: usize = 1000;

/// Production-default mirror worker count
/// (`LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS`).
const DEFAULT_MIRROR_WORKERS: usize = 2;

/// The mirror's byte budget: `M_ext` at the chart's 5 GiB pod, 128 MiB,
/// thousands of times what one call's 16 × 4 KiB values are charged, so the
/// budget never sheds and the measured path is the production steady state.
const MIRROR_BUDGET_BYTES: u64 = 128 * 1024 * 1024;

/// Worker threads for the bench runtime. `store_values` fans out over a
/// `buffered(8)` stream of Valkey round trips and the mirror workers drain
/// on the same runtime, so four threads keep both sides off each other's
/// critical path without over-subscribing a shared CI runner.
const RUNTIME_WORKER_THREADS: usize = 4;

/// Service label stamped on this bench's metrics registry.
const SERVICE_LABEL: &str = "ext-mirror-store-values-bench";

/// Env var naming the Valkey to bench against. Absent → skip.
const TEST_VALKEY_URL_VAR: &str = "TEST_VALKEY_URL";

/// Env vars that make [`Metrics::new`] attach a live OTLP push exporter.
/// See the identical note in `benches/ext_mirror_enqueue.rs`.
const OTLP_ENDPOINT_VARS: [&str; 2] = [
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
];

/// Install a process-global `tracing` subscriber that formats every enabled
/// event and writes it to `std::io::sink()`, so log statements on the
/// measured path carry their production-shaped formatting cost instead of
/// collapsing to a disabled-callsite check. Same rationale, and the same
/// `try_init`-and-discard idempotence, as
/// `benches/ext_mirror_enqueue.rs::install_sink_log_subscriber`.
fn install_sink_log_subscriber() {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::sink)
        .try_init();
}

/// Build the metrics registry with both OTLP endpoint vars pinned unset.
fn scrape_only_metrics() -> SharedMetrics {
    Arc::new(temp_env::with_vars(
        OTLP_ENDPOINT_VARS.map(|k| (k, None::<&str>)),
        || Metrics::new(SERVICE_LABEL),
    ))
}

/// The fixed `(content hash, payload)` fixture set, generated once: each
/// payload keyed by `ext_content_hash`, the content address a real upload
/// carries. Index-seeded payloads guarantee distinct content addresses, so
/// each blob is its own key rather than collapsing onto one refcounted entry.
fn fixture_blobs() -> Vec<([u8; 32], Vec<u8>)> {
    (0..BATCH_HASHES)
        .map(|i| {
            let mut payload = vec![0u8; PAYLOAD_BYTES];
            payload[..8].copy_from_slice(&(i as u64).to_le_bytes());
            let hash = ext_content_hash(&payload);
            (hash, payload)
        })
        .collect()
}

/// A `TestValkey`-backed store plus, when the mirror is wired, the drain
/// pipeline it feeds.
struct Arm {
    store: ValkeyExternalValueStore,
    /// Held so the prefix-scoped SCAN + UNLINK teardown runs at the end of
    /// the arm rather than before the measurement. Never read.
    test_valkey: Option<TestValkey>,
    cancel: CancellationToken,
    handles: Vec<tokio::task::JoinHandle<()>>,
}

impl Arm {
    /// Build one arm. `with_mirror = false` reproduces the wiring
    /// `main.rs` installs with checkpointing disabled (no sender, so the
    /// hot path's `if let` guard is a total no-op); `true` reproduces the
    /// wiring for any enabled adapter, with production-default capacity
    /// and worker count draining into an `InMemoryCheckpointStore` (chosen
    /// so no adapter-side I/O variance lands in the measurement).
    ///
    /// Must be called from inside the runtime — `TestValkey::setup`
    /// connects and `spawn_ext_mirror_workers` calls `tokio::spawn`.
    async fn build(with_mirror: bool, metrics: SharedMetrics) -> Self {
        let test_valkey = TestValkey::setup().await;
        let mut builder = ValkeyExternalValueStore::new(
            test_valkey.client().clone(),
            test_valkey.key_schema().clone(),
        )
        .with_metrics(Arc::clone(&metrics));

        let cancel = CancellationToken::new();
        let mut handles = Vec::new();
        if with_mirror {
            let (sender, rx) = ext_mirror_channel(
                DEFAULT_MIRROR_CAPACITY,
                std::num::NonZeroU64::new(MIRROR_BUDGET_BYTES).expect("non-zero"),
                Arc::clone(&metrics),
            );
            let checkpoint_store: Arc<dyn CheckpointStore> =
                Arc::new(InMemoryCheckpointStore::new());
            handles = spawn_ext_mirror_workers(
                checkpoint_store,
                rx,
                DEFAULT_MIRROR_WORKERS,
                Arc::clone(&metrics),
                None,
                cancel.clone(),
            );
            builder = builder.with_ext_mirror_sender(sender);
        }

        Self {
            store: builder,
            test_valkey: Some(test_valkey),
            cancel,
            handles,
        }
    }

    /// Cancel + join the mirror workers, then drop the `TestValkey` from a
    /// spawned task.
    ///
    /// The spawn is load-bearing: `TestValkey::drop` runs its prefix
    /// cleanup under `tokio::task::block_in_place`, which is only legal on
    /// a multi-thread runtime **worker** thread. `Runtime::block_on` does
    /// not make its caller a worker, so dropping the fixture directly
    /// inside `block_on` would panic; dropping it inside a spawned task
    /// puts it on a real worker thread where `block_in_place` is valid.
    async fn shutdown(mut self) {
        self.cancel.cancel();
        let join_all = async {
            for handle in self.handles.drain(..) {
                let _ = handle.await;
            }
        };
        let _ = tokio::time::timeout(Duration::from_secs(5), join_all).await;

        if let Some(test_valkey) = self.test_valkey.take() {
            let _ = tokio::spawn(async move { drop(test_valkey) }).await;
        }
    }
}

fn bench_store_values(c: &mut Criterion) {
    install_sink_log_subscriber();

    if std::env::var(TEST_VALKEY_URL_VAR).is_err() {
        println!(
            "[ext_mirror_store_values] SKIPPED: {TEST_VALKEY_URL_VAR} is unset, so there is no \
             Valkey to bench against. Set it (e.g. `redis://127.0.0.1:6379`, or \
             `redis://valkey:6379` inside the dev container) and re-run. The Valkey-free \
             precision bench `ext_mirror_enqueue` is unaffected and still ran."
        );
        return;
    }

    let runtime = build_runtime();
    let metrics = scrape_only_metrics();
    let blobs = fixture_blobs();
    // Borrowed view rebuilt once, not per iteration: `store_values` takes
    // `&[(&[u8; 32], &[u8])]`, and building that vector inside the timed
    // closure would add 16 pointer-pair pushes to a measurement whose unit
    // is a network round trip.
    let batch: Vec<(&[u8; 32], &[u8])> = blobs
        .iter()
        .map(|(hash, data)| (hash, data.as_slice()))
        .collect();

    let mut group = c.benchmark_group("ext_mirror_store_values");
    // A single iteration is `BATCH_HASHES` Valkey round trips (tens of ms
    // under CI load), so criterion's default 100-sample plan would take
    // minutes. Ten samples over a bounded window is enough for the
    // report-only signal this bench provides.
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Bytes((BATCH_HASHES * PAYLOAD_BYTES) as u64));

    for (id, with_mirror) in [("no_mirror", false), ("mirror_drained", true)] {
        let arm = runtime.block_on(Arm::build(with_mirror, Arc::clone(&metrics)));
        group.bench_with_input(BenchmarkId::new(id, BATCH_HASHES), &BATCH_HASHES, |b, _| {
            b.to_async(&runtime).iter(|| async {
                arm.store
                    .store_values(black_box(batch.as_slice()))
                    .await
                    .expect("store_values must succeed against the bench Valkey");
            });
        });
        runtime.block_on(arm.shutdown());
    }

    group.finish();
}

/// The multi-thread runtime both arms run on. Multi-thread (not
/// current-thread) is required, not stylistic: `TestValkey::drop` uses
/// `block_in_place`, which panics on a current-thread runtime, and the
/// mirror workers need to make progress concurrently with the producer.
fn build_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(RUNTIME_WORKER_THREADS)
        .enable_all()
        .build()
        .expect("bench runtime must build")
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_store_values
}
criterion_main!(benches);
