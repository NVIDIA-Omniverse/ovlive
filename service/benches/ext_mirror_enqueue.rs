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

//! Criterion microbench for the SR-3.1 ext-mirror **enqueue** surface: the
//! post-commit [`ExtMirrorSender::send`] that
//! `ValkeyExternalValueStore::store_values` performs after the ext-value
//! `EVALSHA`.
//!
//! ## It measures the production helper
//!
//! [`enqueue_once`] below calls the same [`ExtMirrorSender`] helpers the
//! production site does (`src/storage/external_values.rs`, inside
//! `ValkeyExternalValueStore::store_values_with`): the value's copy is made
//! by [`ExtMirrorSender::copy_value`] before the `EVALSHA`, and the
//! post-`EVALSHA` `if let (Some(sender), Some(copy)) = …` hands it to
//! [`ExtMirrorSender::send`], with the store stamp the `EVALSHA` returned.
//! There is no hand-maintained copy of the production statements to drift.
//! The copy itself is made whether or not a sender is wired (a plain copy
//! when none is), and the stamp is decoded from the reply either way, so both
//! are built outside every timed region here and are not part of the enqueue
//! cost.
//!
//! ## What this bench guards
//!
//! Exactly one thing (the `ext_mirror_copy` group below only reports):
//! **the per-send cost trend** — the byte charge (the
//! oversized check, a non-waiting acquire on the budget semaphore, the
//! charge slot's fill and a gauge add), a `String` clone of the hex hash,
//! and a non-blocking `try_send` onto the bounded mpsc channel. That effect
//! is on the order of a microsecond. It is the only quantity in this
//! subsystem that criterion can resolve, and it is the quantity that
//! regresses when someone adds a hot-path log line, a copy of the blob, a
//! serialization step, or swaps `try_send` for `send().await`.
//!
//! This bench replaces the retired in-tree timing assertion
//! `ext_mirror_hot_path_within_two_percent_of_baseline`, which asserted a
//! ratio on a *composed* `store_values` call. That measurement was
//! defective by construction: `store_values` is dominated by the Valkey
//! `EVALSHA` round trip (tens of milliseconds under CI load), so the
//! ~1 µs guarded effect was roughly 0.05 % of the signal and the ratio it
//! asserted was measuring runner scheduling noise, not the mirror.
//!
//! ## Where the old file's other failure modes are covered now
//!
//! The retired test tried to cover four concerns at once. Three of them
//! do not belong in a benchmark at all and have deterministic homes:
//!
//! - **"The send must never block / must drop on a full channel."**
//!   Deterministic unit + integration gating tests, not timing:
//!   `src/checkpoint/ext_mirror.rs::try_send_saturated_channel_drops_and_ticks_dropped_counter`
//!   and
//!   `tests/integration/checkpoint/ext_mirror_e2e_test.rs::create_layer_with_ext_value_drop_on_full_increments_dropped_metric`.
//!   Those assert the *behaviour* (item dropped, counter ticked, producer
//!   returned) with no wall-clock in the assertion, so they run in the
//!   release-gating suite and cannot flake on runner load.
//! - **"The mirror must not back-pressure production under saturation."**
//!   The operational signal is the
//!   `live_edit_checkpoint_ext_mirror_dropped_total` counter and the
//!   alerts built on it — a production observation, not a CI measurement.
//! - **"The composed `store_values` hot path stays within ±2 % of pre-Epic
//!   latency" (Epic AC #5).** That is a release-build, production-class-
//!   hardware, steady-state claim. Its authoritative home is an internal
//!   Python workload simulator. The sibling bench
//!   `benches/ext_mirror_store_values.rs` reports the composed call-site
//!   numbers for local triage but deliberately gates nothing.
//!
//! ## The three arms
//!
//! Each payload size is measured three ways so a regression report can be
//! read without re-deriving what changed:
//!
//! - `no_mirror/<size>` — sender is `None`, and so is the copy. The
//!   `if let (Some(sender), Some(copy))` guard short-circuits to a complete
//!   no-op: no charge, no enqueue, no metric. This is the pre-Epic hot path
//!   verbatim, i.e. the wiring `main.rs` installs when
//!   `LIVE_EDIT_CHECKPOINT_ENABLED=false`. It is the floor the other two
//!   arms are read against.
//! - `mirror_drained/<size>` — a real [`ExtMirrorSender`] over a bounded
//!   channel with [`spawn_ext_mirror_workers`] draining into an
//!   [`InMemoryCheckpointStore`]. This is the production steady state, and
//!   the CI-gated arm. `InMemoryCheckpointStore` is deliberate: `LocalDisk`
//!   would add fsync variance and `StorageApi` would add HTTP variance to a
//!   consumer that this bench does not measure and does not want perturbing
//!   the producer. Held genuinely non-full, in items and in bytes, by a
//!   drain barrier and asserted to have dropped nothing — see "Keeping the
//!   drained arm actually drained" below.
//! - `mirror_saturated/<size>` — capacity-1 channel, no drainers, so
//!   every send takes the charge and then the `TrySendError::Full`
//!   drop-on-full branch (counter tick + WARN). This pins the cost of the
//!   *degraded* path, which is the one an operator hits during a mirror
//!   stall and the one most likely to be accidentally made expensive. The
//!   WARN is only a real cost because a subscriber is installed — see
//!   "Logging" below.
//!
//! The tokio runtime exists only to host the drain workers in the
//! `mirror_drained` arm; the measured closure is **synchronous** in all
//! three arms, exactly like the producer-side effect it models. Nothing
//! measured here touches Valkey.
//!
//! ## The value's copy, report-only
//!
//! The copy the three arms build outside their timed regions is still a
//! per-value cost on the store path, and it differs with the mirror wired.
//! A second group, `ext_mirror_copy`, measures it at the same payload sizes,
//! two ways:
//!
//! - `copy_value/<size>` — [`ExtMirrorSender::copy_value`] on a wired
//!   sender, what the store path builds when the mirror is on: the
//!   value-sized allocation plus two small ones, the copy's charge slot (an
//!   `Arc<OnceLock<…>>`) and the shared owner `Bytes::from_owner` allocates.
//! - `copy_from_slice/<size>` — `Bytes::copy_from_slice`, what it builds
//!   with no mirror wired: the value-sized allocation alone.
//!
//! Each iteration builds one copy and drops it, so both allocations and
//! frees are in the number, as both are paid per stored value. The
//! difference between the two ids is the mirror's per-value copy overhead;
//! a `copy_value` that grows apart from `copy_from_slice` in proportion to
//! the payload is a second copy of the value. No copy is sent, so no charge
//! is taken. Both ids are **report-only**: neither is in
//! `tools/check_bench_thresholds.py`'s `THRESHOLDS_NS`, whose gate matches
//! ids exactly, so CI prints them and can never fail on them.
//!
//! ## Keeping the drained arm actually drained
//!
//! A criterion loop is not a production producer. Production pays an
//! ext-value `EVALSHA` round trip before every enqueue, so the mirror
//! receives one item per tens of microseconds and two workers hold the
//! queue near-empty indefinitely. The measured closure has no round trip:
//! it enqueues in well under a microsecond, faster than any consumer doing
//! real work can retire items. A plain `Bencher::iter` therefore drives the
//! bounded channel to Full and turns this arm into a second
//! `mirror_saturated` — before the barrier below existed, ~22 % of the
//! 4 KiB arm's iterations were taking the drop branch, and the id CI gates
//! was a blend of two different code paths.
//!
//! Two mechanisms fix that, and neither is optional:
//!
//! 1. **An untimed drain barrier.** The arm runs under `iter_custom` and
//!    splits criterion's iteration count into chunks of half the channel
//!    capacity, waiting (outside the timed region) for every charge to be
//!    returned to the byte budget before each chunk — the queue empty and
//!    every copy the workers took dropped. Every measured send then lands
//!    on a channel that is at most half full and a budget sized to hold the
//!    whole channel — the production invariant — by construction rather
//!    than by hope. See [`DrainHarness::wait_until_drained`].
//! 2. **A zero-drop assertion.** [`bench_ext_mirror_enqueue`] reads
//!    `live_edit_checkpoint_ext_mirror_dropped_total`, summed over every
//!    `reason`, off this bench's own [`Metrics`] registry immediately
//!    before and after each drained-arm measurement and **panics** on any
//!    increase. A panicking bench fails the job, which is the right
//!    outcome: the alternative is a green gate over a number that silently
//!    measures the wrong path. [`assert_drop_counter_is_readable`] proves
//!    at startup that the reader really does observe a drop, so the
//!    zero-delta assertion cannot pass vacuously.
//!
//! Note that the CI ceiling alone would *not* have caught this. The two
//! paths cost the same order of magnitude — measured when each enqueue still
//! copied its blob, the drop branch ran ~2.4x dearer at 1 KiB and ~1.5x
//! dearer at 4 KiB (its structured WARN dominates). Against a ceiling with
//! ~40x headroom, no amount of stalling produces a breach. A stalled drain
//! is a correctness problem for the measurement, not a threshold breach, so
//! it has to be caught here.
//!
//! ## Logging
//!
//! Both benches install a global `tracing_subscriber::fmt()` writing to
//! `std::io::sink()`. Without a subscriber every `tracing::warn!` in the
//! measured code is a disabled-callsite branch costing ~nothing, and the
//! `mirror_saturated` arm would not in fact be pinning the cost of the
//! degraded path — the drop branch's structured WARN would vanish from
//! the measurement. With the subscriber the event is filtered, its fields
//! formatted, and the line written; only the final `write` goes to a sink
//! instead of stdout, which keeps stdio contention (and criterion's own
//! output) out of the numbers while leaving the formatting cost — the part
//! that regresses when someone adds a hot-path log line — fully present.
//!
//! ## Why an absolute ceiling, not `criterion --baseline`
//!
//! Criterion's own `--save-baseline` / `--baseline` comparison was
//! considered for the CI gate and deliberately **not** wired. Two reasons,
//! both structural to this repo's CI:
//!
//! 1. `.gitlab-ci.yml` declares no `cache:` block anywhere, so a
//!    `target/criterion/` baseline written by one pipeline does not exist
//!    in the next. Wiring the comparison would mean introducing a CI cache
//!    solely to carry benchmark state between unrelated pipelines.
//! 2. The CI runner pools are heterogeneous (different pod sizes on
//!    different clusters). A baseline
//!    captured on one tier compared against a run on another produces
//!    hardware deltas, not code deltas — precisely the false-signal class
//!    that got the old assertion de-gated in the first place.
//!
//! CI therefore checks a single **absolute ceiling** on
//! `ext_mirror_enqueue/mirror_drained/4096` via
//! `tools/check_bench_thresholds.py`, set with wide headroom so it fires
//! only on an order-of-magnitude regression (a `send().await`, a
//! serialization step, a synchronous adapter call) and never on runner
//! noise. Everything else is reported informationally.
//!
//! The precise instrument is **local**: `cargo bench -p live-edit --bench
//! ext_mirror_enqueue` compares against the previous run stored in
//! `target/criterion/` on the same machine, which is the comparison that
//! is actually apples-to-apples.
//!
//! ## Running it
//!
//! ```text
//! cargo bench -p live-edit --bench ext_mirror_enqueue
//! cargo bench -p live-edit --bench ext_mirror_enqueue -- --output-format bencher
//! ```
//!
//! This target declares no `required-features` and runs bare, exactly as
//! above — that is the local default and it stays supported. CI
//! (`perf:ext-mirror-bench`) nonetheless passes `--features test-support`
//! here, purely so this invocation and the sibling
//! `ext_mirror_store_values` one — which *requires* that feature — resolve
//! to a single feature set and therefore a single release build of
//! `live-edit` instead of two. The feature gates only
//! `storage::test_support` and `dep:libc`; nothing on the measured path
//! changes, and this file compiles and behaves identically either way.

use std::hint::black_box;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::ext_mirror::item_charge_kib;
use live_edit::checkpoint::{
    ext_mirror_channel, spawn_ext_mirror_workers, CheckpointStore, ExtMirrorCopy, ExtMirrorSender,
};
use live_edit::observability::metrics::{Metrics, SharedMetrics};
use live_edit::storage::external_values::ext_content_hash;

/// Payload sizes swept by every arm, in bytes. 4 KiB is the CI-gated size
/// (it sits at the small end of realistic ext-values, where a newly-added
/// constant-cost step is most visible); 1 KiB brackets it from below, and
/// 64 KiB / 1 MiB establish that the cost stays flat in payload size: the
/// send only charges and moves the copy it is handed, so a cost that grows
/// with the payload is a copy of the blob reintroduced.
const PAYLOAD_SIZES: [usize; 4] = [1024, 4 * 1024, 64 * 1024, 1024 * 1024];

/// The gated size — kept as a named constant so the id string in
/// `tools/check_bench_thresholds.py` (`ext_mirror_enqueue/mirror_drained/4096`)
/// has a visible anchor on this side of the boundary.
const GATED_PAYLOAD_SIZE: usize = 4 * 1024;

/// Compile-time guard that the gated size is actually swept. Dropping 4 KiB
/// from [`PAYLOAD_SIZES`] would otherwise stop emitting the gated id, and
/// the checker's missing-id failure would surface only in CI.
const _: () = assert!(PAYLOAD_SIZES[1] == GATED_PAYLOAD_SIZE);

/// Production-default mirror channel capacity
/// (`LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY`). Used verbatim by the
/// `mirror_drained` arm at small payload sizes; see [`drained_capacity`]
/// for the large-payload clamp.
const DEFAULT_MIRROR_CAPACITY: usize = 1000;

/// Production-default mirror worker count
/// (`LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS`).
const DEFAULT_MIRROR_WORKERS: usize = 2;

/// Memory ceiling the copies one chunk of sends carries are sized against.
/// Each send carries its own copy of the payload (production makes one per
/// stored value), built untimed before the chunk, so a chunk of `n` sends at
/// 1 MiB holds `n` MiB. [`drained_capacity`] and [`saturated_chunk`] clamp
/// the large sizes to this bound; the small ones (the CI-gated 4 KiB
/// included) keep the production capacity.
const CHUNK_MEMORY_CEILING: usize = 32 * 1024 * 1024;

/// Worker threads for the runtime hosting the drain workers. Two workers
/// plus scheduler overhead; the measured closure runs on the criterion
/// thread, not on this runtime, so a larger pool would only add
/// cross-thread wakeup jitter to the consumer side.
const DRAIN_RUNTIME_THREADS: usize = 2;

/// Poll interval for [`DrainHarness::wait_until_drained`]. Long enough not
/// to compete with the drain workers for a core, short enough that the
/// untimed barrier stays a small fraction of each chunk's wall time.
const DRAIN_BARRIER_POLL: Duration = Duration::from_micros(50);

/// Ceiling on one [`DrainHarness::wait_until_drained`] wait. Draining a
/// full queue takes low milliseconds even on a loaded runner, so a wait
/// this long means the workers are wedged, not slow.
const DRAIN_BARRIER_TIMEOUT: Duration = Duration::from_secs(10);

/// Service label stamped on the benchmark's metrics registry. Distinct
/// from the `"test"` label the integration suite uses so a stray scrape
/// during triage is unambiguous about its origin.
const SERVICE_LABEL: &str = "ext-mirror-enqueue-bench";

/// Env vars that make [`Metrics::new`] attach a live OTLP push exporter.
/// Mirrors `tests/common::metrics_scrape_only`: a leaked endpoint (CI, a
/// sibling shell) would otherwise have the bench push metrics to a real
/// collector and add a background exporter task to every measurement.
/// Copied rather than imported because `tests/common` is a test-binary
/// module, not part of the library's public API.
const OTLP_ENDPOINT_VARS: [&str; 2] = [
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
];

/// The counter the `mirror_drained` self-check reads, summed over its
/// `reason` cells. Ticked by [`ExtMirrorSender::send`] on every drop
/// (`queue_full`, `budget_full`, `oversized`), so a non-zero delta across a
/// drained-arm measurement means the drain workers were not keeping up and
/// the arm silently measured a drop path.
const DROPPED_METRIC: &str = "live_edit_checkpoint_ext_mirror_dropped_total";

/// Channel capacity for the `mirror_drained` arm at a given payload size:
/// the production default, clamped to `capacity × payload ≤`
/// [`CHUNK_MEMORY_CEILING`] (see there). Small payloads (including the
/// CI-gated 4 KiB) get the production capacity unchanged; only the 1 MiB
/// size is clamped, and the clamp does not change what is measured — the
/// producer-side send cost is independent of capacity as long as the
/// queue is not full, which the drain workers ensure.
fn drained_capacity(payload_bytes: usize) -> usize {
    (CHUNK_MEMORY_CEILING / payload_bytes.max(1)).clamp(1, DEFAULT_MIRROR_CAPACITY)
}

/// Sends per timed chunk in the `mirror_saturated` arm: as many as fit
/// [`CHUNK_MEMORY_CEILING`], at most the production capacity. Their copies'
/// handles are held across the chunk (see the arm), so the budget must hold
/// them all.
fn saturated_chunk(payload_bytes: usize) -> u64 {
    (CHUNK_MEMORY_CEILING / payload_bytes.max(1)).clamp(1, DEFAULT_MIRROR_CAPACITY) as u64
}

/// A byte budget that holds `items` charged values of `payload_bytes` with
/// room to spare, so it never sheds where the arm does not mean it to.
fn budget_for(items: u64, payload_bytes: usize) -> NonZeroU64 {
    let charge = item_charge_kib(payload_bytes as u64) * 1024;
    NonZeroU64::new((items + 2) * charge).expect("a charge is never zero")
}

/// Build the shared metrics registry with both OTLP endpoint vars pinned
/// to unset for the duration of construction.
fn scrape_only_metrics() -> SharedMetrics {
    Arc::new(temp_env::with_vars(
        OTLP_ENDPOINT_VARS.map(|k| (k, None::<&str>)),
        || Metrics::new(SERVICE_LABEL),
    ))
}

/// A deterministic payload of `len` bytes. The repeating non-zero pattern
/// keeps the allocation from being a page of zeros that an allocator or
/// `memcpy` could special-case.
fn payload_of(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

/// The lowercase 64-char hex of `payload`'s content hash (`ext_content_hash`,
/// the upload's verified claim), matching the `sha256_to_hex` output the real
/// hot path passes to `send`.
fn hash_hex_of(payload: &[u8]) -> String {
    ext_content_hash(payload)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Install a process-global `tracing` subscriber that formats every enabled
/// event and writes it to `std::io::sink()`.
///
/// Load-bearing, not cosmetic. With no subscriber installed, `tracing`
/// short-circuits every callsite to a disabled-level check, so the WARN in
/// [`ExtMirrorSender::send`]'s drop branch costs a predictable
/// ~nothing — and the `mirror_saturated` arm's claim to pin the cost of the
/// degraded path, plus the checker's "a log statement promoted to an
/// always-on level" regression cause, would both be unmeasurable here. The
/// default `fmt()` max level is INFO, so WARN is enabled: the event is
/// filtered, its fields are formatted, and the line is written. Only the
/// destination is a sink, which keeps stdio contention and interleaving
/// with criterion's own output out of the measurement while leaving the
/// formatting cost — the part a new hot-path log line actually adds — in
/// it.
///
/// `try_init` (not `init`) and a discarded result: benches run in one
/// process, and a second call, or a subscriber some other harness already
/// installed, must not abort the run.
fn install_sink_log_subscriber() {
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::sink)
        .try_init();
}

/// Current value of [`DROPPED_METRIC`] on this bench's own registry, summed
/// over every `reason` cell.
///
/// Reads it the same way `/metrics` does — `PrometheusExporter::export`,
/// which drives a fresh SDK collect — and parses the samples out of the
/// rendered exposition. Deliberately not `observability::metrics::
/// test_util::sample_sum`: that module is gated behind `cfg(test)` /
/// `feature = "test-support"` and this bench must work under a bare
/// `cargo bench` too. `Metrics::prometheus_exporter` is unconditionally
/// public, so this path has no feature dependency.
///
/// Every `reason` cell exists at 0 from startup; a family with no series
/// would read 0, the "nothing has been dropped yet" state.
fn dropped_total(metrics: &Metrics) -> u64 {
    let mut buf = Vec::new();
    metrics
        .prometheus_exporter()
        .export(&mut buf)
        .expect("rendering the bench's own metrics exposition must succeed");
    let text = String::from_utf8(buf).expect("Prometheus exposition must be valid UTF-8");

    let mut total = 0.0;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // `name{labels} value` or `name value`; split the value off the end
        // first because the label body may itself contain spaces.
        let Some((series, value)) = line.rsplit_once(char::is_whitespace) else {
            continue;
        };
        let name = series.split_once('{').map_or(series, |(name, _)| name);
        if name == DROPPED_METRIC {
            total += value.trim().parse::<f64>().unwrap_or(0.0);
        }
    }
    total as u64
}

/// Prove, before any measurement, that [`dropped_total`] actually observes
/// a drop.
///
/// Without this the drained arm's zero-delta assertion would pass
/// vacuously if the metric were ever renamed, re-labelled, or rendered in a
/// shape the parser above does not match — turning the self-check into
/// decoration. Forces exactly one drop on a throwaway registry (capacity-1
/// channel, filled, then a second send) and asserts the reader sees the
/// tick.
fn assert_drop_counter_is_readable() {
    let metrics = scrape_only_metrics();
    let (sender, _rx) = ext_mirror_channel(1, budget_for(2, 64), Arc::clone(&metrics));

    // First send fills the single slot and returns `Ok` — no tick.
    enqueue_once(
        Some(&sender),
        &"00".repeat(32),
        Some((sender.copy_value(b"fill"), stamp())),
    );
    let before = dropped_total(&metrics);
    // Second send finds the channel Full and takes the drop branch.
    enqueue_once(
        Some(&sender),
        &"11".repeat(32),
        Some((sender.copy_value(b"dropped"), stamp())),
    );
    let after = dropped_total(&metrics);

    assert_eq!(
        after,
        before + 1,
        "the bench's own {DROPPED_METRIC} reader is broken: a deliberately \
         forced drop moved the counter from {before} to {after}, expected \
         {}. The mirror_drained arm's drop self-check reads the counter the \
         same way, so it would silently pass on every run. Check whether the \
         metric was renamed or its rendered series shape changed.",
        before + 1
    );
}

/// The production post-commit send, in the production call shape: with a
/// sender wired, hand the value's copy (made before the `EVALSHA` by
/// [`ExtMirrorSender::copy_value`]) to [`ExtMirrorSender::send`] along with a
/// clone of the hex hash. With `sender == None` there is no copy either, both
/// halves of the `if let` tuple are `None`, and the whole thing compiles to
/// the pre-Epic no-op.
///
/// Kept as a standalone `fn` rather than inlined into each closure so all
/// three arms provably measure the same code shape.
#[inline]
fn enqueue_once(
    sender: Option<&ExtMirrorSender>,
    hash_hex: &str,
    copy: Option<(ExtMirrorCopy, String)>,
) {
    if let (Some(sender), Some((copy, last_store_ms))) = (sender, copy) {
        sender.send(hash_hex.to_owned(), copy, last_store_ms);
    }
}

/// A store stamp of the production shape (a decimal millisecond `String`),
/// as the store script's reply decodes to.
fn stamp() -> String {
    "1760000000000".to_owned()
}

/// `n` copies of `payload` made the way the store path makes them, each with
/// its store stamp, built outside every timed region: production makes the
/// copy and decodes the stamp whether or not the mirror is wired, so neither
/// is part of the enqueue cost.
fn copies_of(sender: &ExtMirrorSender, payload: &[u8], n: u64) -> Vec<(ExtMirrorCopy, String)> {
    (0..n)
        .map(|_| (sender.copy_value(payload), stamp()))
        .collect()
}

/// Live drain pipeline for the `mirror_drained` arm: a bounded channel and
/// byte budget whose receiver is owned by [`spawn_ext_mirror_workers`]
/// tasks running on `runtime`, writing into an [`InMemoryCheckpointStore`].
///
/// Every enqueued item reuses the same hash, so the adapter's map holds
/// exactly one entry regardless of iteration count — the store is a real
/// consumer, not a memory sink, and it keeps no item's bytes, so each
/// charge returns to the budget once a worker is done with its item.
struct DrainHarness {
    sender: ExtMirrorSender,
    capacity: usize,
    runtime: tokio::runtime::Runtime,
    cancel: CancellationToken,
    handles: Vec<tokio::task::JoinHandle<()>>,
}

impl DrainHarness {
    fn new(capacity: usize, payload_bytes: usize, metrics: SharedMetrics) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(DRAIN_RUNTIME_THREADS)
            .enable_all()
            .build()
            .expect("bench drain runtime must build");

        // A budget that holds the whole channel and the items in the
        // workers' hands, so only a stalled drain could make it shed.
        let (sender, rx) = ext_mirror_channel(
            capacity,
            budget_for(
                capacity as u64 + DEFAULT_MIRROR_WORKERS as u64,
                payload_bytes,
            ),
            Arc::clone(&metrics),
        );
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let cancel = CancellationToken::new();

        // `spawn_ext_mirror_workers` calls `tokio::spawn`, which needs an
        // ambient runtime context; the enter guard supplies it without
        // making the (synchronous) caller async.
        let handles = {
            let _guard = runtime.enter();
            spawn_ext_mirror_workers(
                store,
                rx,
                DEFAULT_MIRROR_WORKERS,
                metrics,
                None,
                cancel.clone(),
            )
        };

        Self {
            sender,
            capacity,
            runtime,
            cancel,
            handles,
        }
    }

    /// How many sends may be timed back-to-back without any chance of
    /// filling the channel: half the capacity, so a chunk started against a
    /// fully-drained queue leaves at least half the slots free even if the
    /// workers make no progress at all during it. Never zero.
    fn enqueue_chunk(&self) -> u64 {
        (self.capacity as u64 / 2).max(1)
    }

    /// Block (untimed) until every charge is back in the byte budget: the
    /// queue is empty and the workers have dropped every copy they took.
    ///
    /// Necessary because a criterion loop is not a production producer. In
    /// production each enqueue is preceded by an ext-value `EVALSHA`
    /// round trip, so the mirror sees one item per tens of microseconds and
    /// two workers keep the queue near-empty indefinitely. The measured
    /// closure here has no round trip: it enqueues in well under a
    /// microsecond, faster than any consumer that does real work can
    /// retire items, so an unbounded `iter` loop drives *any* bounded
    /// channel to Full and silently converts the arm into a second
    /// `mirror_saturated`. (It did exactly that before this barrier
    /// existed: ~22 % of the 4 KiB arm's iterations were taking the
    /// drop branch.) Running the timed region in chunks separated by this
    /// barrier restores the production invariant — every measured send
    /// lands on a non-full channel with room in the budget — without
    /// putting the wait itself in the measurement.
    ///
    /// Spins with a short sleep rather than `yield_now`: the drain workers
    /// need the cores, and burning one thread on a spin loop is what makes
    /// them slow in the first place on a shared runner.
    fn wait_until_drained(&self) {
        let deadline = std::time::Instant::now() + DRAIN_BARRIER_TIMEOUT;
        while self.sender.available_budget_bytes() < self.sender.budget_bytes() {
            assert!(
                std::time::Instant::now() < deadline,
                "the ext-mirror drain workers did not return every charge to a {} B budget \
                 (a {}-slot channel) within {DRAIN_BARRIER_TIMEOUT:?}; {} B still charged. The \
                 mirror_drained arm cannot measure the non-full send path while the queue \
                 stays backed up, so this is a hard failure rather than a silently degraded \
                 measurement.",
                self.sender.budget_bytes(),
                self.capacity,
                self.sender.budget_bytes() - self.sender.available_budget_bytes(),
            );
            std::thread::sleep(DRAIN_BARRIER_POLL);
        }
    }

    /// Cancel the workers and join them before the runtime is dropped.
    /// Dropping a `Runtime` from inside async context panics and dropping
    /// it with live tasks blocks on their shutdown, so the join is
    /// explicit and bounded.
    fn shutdown(self) {
        let Self {
            sender,
            capacity: _,
            runtime,
            cancel,
            handles,
        } = self;
        // The producer handle must go before the workers are asked to stop;
        // a live sender keeps the channel open.
        drop(sender);
        cancel.cancel();
        runtime.block_on(async {
            let join_all = async {
                for handle in handles {
                    let _ = handle.await;
                }
            };
            let _ = tokio::time::timeout(Duration::from_secs(5), join_all).await;
        });
    }
}

fn bench_ext_mirror_enqueue(c: &mut Criterion) {
    install_sink_log_subscriber();
    assert_drop_counter_is_readable();

    let metrics = scrape_only_metrics();
    let mut group = c.benchmark_group("ext_mirror_enqueue");

    for size in PAYLOAD_SIZES {
        let payload = payload_of(size);
        let hash_hex = hash_hex_of(&payload);
        group.throughput(Throughput::Bytes(size as u64));

        // ── Arm 1: no mirror wired (pre-Epic no-op guard). ──────────────
        group.bench_with_input(BenchmarkId::new("no_mirror", size), &size, |b, _| {
            let sender: Option<ExtMirrorSender> = None;
            b.iter(|| {
                enqueue_once(
                    black_box(sender.as_ref()),
                    black_box(hash_hex.as_str()),
                    black_box(None),
                );
            });
        });

        // ── Arm 2: real sender, workers draining (production steady state). ─
        {
            let harness = DrainHarness::new(drained_capacity(size), size, Arc::clone(&metrics));
            // Bracket the measurement with the drop counter: this arm is only
            // meaningful while the workers keep the channel and the budget
            // non-full. See the module doc — a stall mixes the drop branch
            // into this number. That branch costs the same order of magnitude
            // and is dearer (it does everything a successful send does, then
            // a counter add, a formatted WARN and a charge release), so the
            // number then describes the wrong code path while staying far
            // below the CI ceiling, which cannot catch it. The check has to
            // be here.
            let dropped_before = dropped_total(&metrics);
            group.bench_with_input(BenchmarkId::new("mirror_drained", size), &size, |b, _| {
                // `iter_custom`, not `iter`, so the criterion-chosen
                // iteration count can be split into chunks that provably fit
                // in the channel and the budget, with an untimed drain
                // barrier and the chunk's copies built between them (see
                // `DrainHarness::wait_until_drained`). The timed region is
                // still nothing but back-to-back `enqueue_once` calls — one
                // `Instant` pair per few hundred iterations is far below the
                // per-iteration cost being reported.
                b.iter_custom(|iters| {
                    let chunk = harness.enqueue_chunk();
                    let mut elapsed = Duration::ZERO;
                    let mut remaining = iters;
                    while remaining > 0 {
                        let this_chunk = remaining.min(chunk);
                        harness.wait_until_drained();
                        let mut copies = copies_of(&harness.sender, &payload, this_chunk);
                        let started = std::time::Instant::now();
                        for copy in copies.drain(..) {
                            enqueue_once(
                                black_box(Some(&harness.sender)),
                                black_box(hash_hex.as_str()),
                                black_box(Some(copy)),
                            );
                        }
                        elapsed += started.elapsed();
                        remaining -= this_chunk;
                    }
                    elapsed
                });
            });
            let dropped_after = dropped_total(&metrics);
            assert!(
                dropped_after == dropped_before,
                "mirror_drained/{size} did NOT stay drained: {} send(s) took a drop \
                 branch during the measurement ({DROPPED_METRIC} went \
                 {dropped_before} -> {dropped_after}, channel capacity {}, {} drain \
                 worker(s)).\n\
                 The reported number for this id is therefore invalid: it mixes the \
                 drop branch (the same order of magnitude as a successful enqueue, and \
                 dearer, as it adds a counter add, a formatted WARN and a charge \
                 release) into what should be production steady state, so it describes \
                 the wrong code path, and the CI ceiling in tools/check_bench_thresholds.py \
                 (~40x above the expected cost) would have passed it. Failing loudly \
                 instead.\n\
                 Likely causes: the drain workers could not keep up on this runner, the \
                 worker pool failed to spawn, or InMemoryCheckpointStore::put_ext_blob \
                 became slow enough to back the channel up.",
                dropped_after - dropped_before,
                drained_capacity(size),
                DEFAULT_MIRROR_WORKERS,
            );
            harness.shutdown();
        }

        // ── Arm 3: saturated channel, no drainers (drop-on-full branch). ─
        {
            // Capacity 1, pre-filled, receiver parked and never polled:
            // the channel is Full — not Closed — for every measured
            // iteration, so each send takes its charge and then the
            // `queue_full` drop branch (counter tick + WARN) rather than the
            // silent shutdown branch. The budget holds a whole chunk's
            // charges: each copy's charge lasts as long as its memory, and
            // the chunk holds a handle on every copy (below).
            let chunk = saturated_chunk(size);
            let (saturated, _rx) =
                ext_mirror_channel(1, budget_for(chunk + 1, size), Arc::clone(&metrics));
            enqueue_once(
                Some(&saturated),
                &hash_hex,
                Some((saturated.copy_value(&payload), stamp())),
            );
            group.bench_with_input(BenchmarkId::new("mirror_saturated", size), &size, |b, _| {
                b.iter_custom(|iters| {
                    let mut elapsed = Duration::ZERO;
                    let mut remaining = iters;
                    while remaining > 0 {
                        let this_chunk = remaining.min(chunk);
                        let mut copies = copies_of(&saturated, &payload, this_chunk);
                        // A handle on each copy, held past the timed region:
                        // in production the dropped copy's memory goes with
                        // it, but freeing a payload-sized block is the
                        // allocator's cost, not the send's, and it would make
                        // the drop branch's cost grow with the payload.
                        let held: Vec<Bytes> =
                            copies.iter().map(|(c, _)| c.bytes().clone()).collect();
                        let started = std::time::Instant::now();
                        for copy in copies.drain(..) {
                            enqueue_once(
                                black_box(Some(&saturated)),
                                black_box(hash_hex.as_str()),
                                black_box(Some(copy)),
                            );
                        }
                        elapsed += started.elapsed();
                        drop(held);
                        remaining -= this_chunk;
                    }
                    elapsed
                });
            });
        }
    }

    group.finish();
}

/// The report-only `ext_mirror_copy` group (module docs, *The value's copy,
/// report-only*): the store path's copy of one value with a sender wired
/// ([`ExtMirrorSender::copy_value`]) against the plain copy it makes with
/// none (`Bytes::copy_from_slice`), at each of [`PAYLOAD_SIZES`]. Each
/// iteration's copy is dropped inside the timed loop, so the number is one
/// copy's allocations and frees.
fn bench_ext_mirror_copy(c: &mut Criterion) {
    let metrics = scrape_only_metrics();
    let mut group = c.benchmark_group("ext_mirror_copy");

    for size in PAYLOAD_SIZES {
        let payload = payload_of(size);
        group.throughput(Throughput::Bytes(size as u64));

        // A wired sender, as the store path holds one with the mirror on.
        // Nothing is sent on it, so its channel and budget stay untouched.
        let (sender, _rx) = ext_mirror_channel(1, budget_for(1, size), Arc::clone(&metrics));
        group.bench_with_input(BenchmarkId::new("copy_value", size), &size, |b, _| {
            b.iter(|| sender.copy_value(black_box(payload.as_slice())));
        });

        group.bench_with_input(BenchmarkId::new("copy_from_slice", size), &size, |b, _| {
            b.iter(|| Bytes::copy_from_slice(black_box(payload.as_slice())));
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    // Short windows on purpose: the measured effect is sub-microsecond to
    // low-microsecond, so a 2 s window still collects tens of thousands of
    // samples per id while keeping the whole 20-id sweep (the enqueue
    // group's 12, the copy group's 8) to under a minute on a shared,
    // CPU-constrained runner.
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_ext_mirror_enqueue, bench_ext_mirror_copy
}
criterion_main!(benches);
