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

// Service metrics, instrumented on the OpenTelemetry metrics API (OMPE-98174).
//
// ── Export model: one MeterProvider, two readers ────────────────────────────
//
// Every instrument below is created from a single `Meter` backed by one
// `SdkMeterProvider` that fans each recording out to TWO readers:
//
//   1. An `opentelemetry-prometheus-text-exporter` `PrometheusExporter`. This
//      is itself the OTel `MetricReader` (attached via `.with_reader`) AND the
//      `/metrics` server: `metrics_server.rs` renders text exposition directly
//      through `PrometheusExporter::export(&mut writer)`, which drives a fresh
//      SDK collect (firing observable-gauge callbacks) on every scrape. No
//      `prometheus::Registry` round-trip. Always attached. Replaces the
//      upstream-deprecated `opentelemetry-prometheus`, which pulled `protobuf`
//      (a RustSec exposure this repo is clearing); the text exporter has no
//      `protobuf` dependency.
//   2. An OTLP/gRPC `PeriodicReader` that pushes to the collector named by
//      `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`. Attached ONLY when that endpoint
//      is set. There is deliberately NO fallback to the generic
//      `OTEL_EXPORTER_OTLP_ENDPOINT`: the standardized Helm schema has
//      independent `observability.tracing` and `observability.metrics` blocks,
//      and the metrics block is the sole control for metrics push — enabling
//      tracing alone (which sets the generic endpoint) must NOT silently start
//      pushing metrics. Endpoint absent → no OTLP reader, scrape still works.
//      The endpoint is `http://` (plaintext) or `https://` (TLS against the
//      native trust roots); any other value, or a push exporter that cannot
//      be built, stops startup (`observability::otlp`). A failed push is
//      logged as a rate-limited WARN (`observability::metrics_export`).
//
// The Prometheus exporter is configured `without_counter_suffixes`,
// `without_units`, `without_scope_info`, and `without_target_info` so the wire
// names it emits are byte-identical to the pre-migration `prometheus`-crate
// output: our counter names already carry `_total`, our histogram names already
// carry their unit, and we do not want the `otel_scope_*` / `target_info`
// synthetic series. Per-histogram explicit bucket boundaries are pinned via
// `with_view` so `le=` buckets match the pre-migration output exactly.
//
// ── Metric-name prefix convention (b.3z7 medium) ────────────────────────────
//
// Two prefixes coexist by subsystem:
//
// - `live_*` — the core gRPC / domain metrics specified in SDD §17.1
//   (endpoint, session, diff, presence-event counters/histograms). These
//   names are load-bearing for operator dashboards and alerting rules and
//   must not change without a dashboard migration.
// - `live_edit_*` — everything added after the SDD was frozen (auth cache,
//   update_layer retries, presence broadcast drops, Valkey connection
//   health, background-task panics). Use this prefix for any new metric.
//
// Adding a new metric under `live_*` requires SDD §17.1 to be updated in
// the same change.
//
// One documented exception carries neither prefix:
// `process_resident_memory_bytes` (b.u5k) uses the name, meaning and unit of
// the Prometheus client libraries' standard process metric, so the dashboards
// and alert rules written against that standard name work unchanged. It has no
// labels, like the standard metric. Do not add further unprefixed names;
// allocator-specific memory gauges use `live_edit_*`
// (`live_edit_jemalloc_allocated_bytes`, `live_edit_jemalloc_resident_bytes`,
// `live_edit_jemalloc_dirty_bytes`).

use std::sync::Arc;
use std::time::Duration;

use opentelemetry::metrics::{Counter, Gauge, Histogram, Meter, MeterProvider, UpDownCounter};
use opentelemetry::KeyValue;
use opentelemetry_prometheus_text_exporter::PrometheusExporter;
use opentelemetry_sdk::metrics::{
    Aggregation, Instrument, PeriodicReader, SdkMeterProvider, Stream,
};

use crate::observability::metrics_export::{MetricExportReporter, ReportingMetricExporter};
use crate::observability::otlp;
use crate::observability::scaling::ScalingCounters;
use crate::pacing::stuck_threshold_secs;

/// Histogram bucket boundaries for endpoint duration (microseconds).
/// Wire contract (SDD §17.1): boundaries may be added, never removed.
const DURATION_BUCKETS: &[f64] = &[
    100.0,
    500.0,
    1_000.0,
    5_000.0,
    10_000.0,
    50_000.0,
    100_000.0,
    500_000.0,
    1_000_000.0,
    5_000_000.0,
    10_000_000.0,
    30_000_000.0,
];

/// Bucket boundaries shared by the diff-build and subscribe-catch-up duration
/// histograms (seconds). Finite buckets reach 60 s so a multi-second catch-up
/// lands in a real bucket rather than `+Inf`.
const DIFF_DURATION_BUCKETS: &[f64] = &[
    0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
];

/// Bucket boundaries for `live_edit_join_latency_seconds` (seconds): handler
/// entry to the first delivered state, which spans auth, admission and the
/// catch-up itself.
const JOIN_LATENCY_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0,
];

/// Bucket boundaries for `live_edit_presence_sweep_duration_seconds`
/// (seconds): the wall time of one presence-sweeper leader pass. A pass over a
/// quiet fleet is a few round-trips (tens of milliseconds); `le=15` brackets
/// the default 15 s sweep interval, so a pass that outran its own cadence lands
/// above it; the tail reaches 300 s, ten times the 30 s leader-lock TTL a pass
/// renews as it goes.
const PRESENCE_SWEEP_DURATION_BUCKETS: &[f64] = &[
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0,
];

/// Bucket boundaries for `live_edit_presence_stale_removal_delay_seconds`
/// (seconds): how long after a connection's lease expired the sweeper removed
/// it. A healthy fleet removes within one sweep interval (15 s by default) plus
/// a pass, so the low buckets resolve that band; `le=60` is one default lease
/// TTL, and the tail reaches 600 s for a sweeper that stopped running.
const PRESENCE_STALE_REMOVAL_DELAY_BUCKETS: &[f64] = &[
    1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 120.0, 300.0, 600.0,
];

/// Bucket boundaries for the subscribe-catch-up delivered-bytes histogram
/// (bytes). Byte-scale, spanning ~1 KB up through and beyond
/// `LIVE_EDIT_MAX_LAYER_SIZE` (100 MB default) so a long / trimmed-session
/// snapshot lands in a real bucket rather than saturating `+Inf`.
const SUBSCRIBE_CATCH_UP_DELIVERED_BYTES_BUCKETS: &[f64] = &[
    1_024.0,
    4_096.0,
    16_384.0,
    65_536.0,
    262_144.0,
    1_048_576.0,
    4_194_304.0,
    16_777_216.0,
    67_108_864.0,
    134_217_728.0,
];

/// Bucket boundaries for `live_edit_stream_mux_xread_reply_bytes` (bytes).
/// Each read's `COUNT` is sized so that one reply aims at
/// `storage::stream_mux::XREAD_REPLY_TARGET_BYTES` (1 MiB): `le=1048576` holds
/// the replies that met it, and the buckets above resolve the soft bound's
/// over-commit (an entry larger than its key's estimate) up to replies of
/// tens of MiB.
const STREAM_MUX_XREAD_REPLY_BYTES_BUCKETS: &[f64] = &[
    1_024.0,
    4_096.0,
    16_384.0,
    65_536.0,
    262_144.0,
    524_288.0,
    1_048_576.0,
    2_097_152.0,
    4_194_304.0,
    16_777_216.0,
    67_108_864.0,
];

/// Bucket boundaries for external auth call duration (seconds).
const AUTH_CALL_DURATION_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// Bucket boundaries for the `update_layer` attempts histogram (b.avd item 6 —
/// stretched to 64/128 to match the 128-attempt default retry cap from b.3gx).
const UPDATE_LAYER_ATTEMPTS_BUCKETS: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];

/// Bucket boundaries for `live_edit_update_layer_commit_duration_seconds`
/// (seconds), 1 ms to 32 s (Spec SDD SR-2.2). The top bound brackets the 30 s
/// default `LIVE_EDIT_RPC_DEADLINE_MS`, so a call that returns just inside
/// that deadline lands in the (16, 32] band rather than in `+Inf`.
const UPDATE_LAYER_COMMIT_DURATION_BUCKETS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0,
];

/// Bucket boundaries for `live_edit_update_layer_structural_commit_seconds`
/// (seconds): the commit-duration buckets' low end, then half-second steps
/// from 1 s to 3 s and 4, 5 and 8 s above, so the structural size gate's
/// decision lines (`storage::structural_size_gate`) are bucket boundaries:
/// the 2 s slow-commit WARN, the 2.5 s hold that is the trigger to lower the
/// work limit, and the 5 s command timeout and `lua-time-limit` past which
/// other clients of the shard are answered `BUSY`.
const UPDATE_LAYER_STRUCTURAL_COMMIT_BUCKETS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0,
];

/// Bucket boundaries for the per-layer checkpoint pending-age histogram
/// (seconds, SDD SR-2.2). Boundaries are unchanged from the pre-b.jq3
/// `live_edit_checkpoint_lag_seconds` histogram this replaced: under the
/// pending-scoped definition a healthy fleet at the default
/// `interval=60s` / `tick=10s` lands its p99 around 70 s, so the
/// documented `> 300s` alert threshold sits ~4-5× above the bound the
/// operator knob promises, and the long tail (30 m / 1 h / 2 h) still has
/// resolution for a genuinely stuck layer.
const CHECKPOINT_PENDING_AGE_BUCKETS: &[f64] = &[
    5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0,
];

/// Bucket boundaries for the startup counter-seed recovery duration histogram
/// (seconds, SDD SR-5.3).
const COUNTER_SEED_BUCKETS: &[f64] = &[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0];

/// Bucket boundaries for the exclusive-pool checkout latency histogram
/// (seconds, b.f4f).
///
/// Weighted at the fast end because an uncontended checkout is a permit
/// `try_acquire` plus a non-blocking scan over at most `pool_size` mutexes —
/// microseconds. Everything above ~1 ms is queueing, which is the signal.
const POOL_CHECKOUT_BUCKETS: &[f64] = &[
    0.000_05, 0.000_1, 0.000_5, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0,
];

/// Bucket boundaries for the capacity-wait histograms (seconds, b.f4f): the
/// `SubscribeReadLayer` admission, catch-up slot, snapshot read-permit and
/// catch-up memory waits, and the upload admission and processing waits. The
/// subscriber waits are bounded by their capacity budget,
/// `LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS` (default 20 s); the upload
/// admission wait by `LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS` (default 2 s) and the
/// upload processing wait by the RPC deadline (default 30 s). `le=20` and
/// `le=30` bracket those defaults, so a wait that ran out its whole bound lands
/// in a finite band rather than in `+Inf`; a bound configured above 30 s puts
/// its timeouts in `+Inf`. A wait made with no deadline — a shared fenced-read
/// flight's slot, fenced read-permit and memory waits (Spec SDD SR-7.2,
/// SR-7.3), whose attached catch-ups are each charged their share instead —
/// can run past the budget: one under 30 s lands in the (20, 30] band, and
/// only a longer one in `+Inf`. The boundaries below 20 are unchanged from the
/// 2 s default this list first bracketed, so existing dashboards keep their
/// series.
const ADMISSION_WAIT_BUCKETS: &[f64] = &[
    0.000_1, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0,
];

/// Bucket boundaries for `live_edit_snapshot_read_duration_seconds` (seconds,
/// Spec SDD SR-8.2). `le=0.05` is the product owner's 50 ms per-read bound
/// (SR-2.3). `le=0.075`, `le=0.15` and `le=0.2` resolve an atomic catch-up
/// read under a join burst, which runs at around 100–200 ms end to end and
/// would otherwise be interpolated across the whole 100–250 ms band; the
/// boundaries above resolve the fan-out reads of layers above the atomic
/// ceiling, which take seconds.
///
/// Also `live_edit_update_layer_fence_wait_seconds` (SR-8.3): a writer's fence
/// wait is one fenced read's epoch plus a polling step (plus, for a structural
/// attempt an epoch interrupted, the attempt it lost), the same scale as the
/// reads that cause it, and SR-9.1 reports the two side by side. `le=30`
/// brackets the default `LIVE_EDIT_RPC_DEADLINE_MS`, which caps the wait. And
/// `live_edit_snapshot_fence_hold_seconds` (SR-8.3), the reader side of that
/// pause: one fenced read's hold of its lease.
const SNAPSHOT_READ_DURATION_BUCKETS: &[f64] = &[
    0.000_5, 0.001, 0.002_5, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.5,
    5.0, 10.0, 30.0,
];

/// Bucket boundaries for `live_edit_snapshot_read_reply_bytes` (bytes, Spec SDD
/// SR-8.2). `le=1048576` brackets the default byte ceiling (604,619 B), so an
/// atomic read near it lands in the (524288, 1048576] band; the boundaries
/// above resolve the fan-out reads of larger layers.
const SNAPSHOT_READ_REPLY_BYTES_BUCKETS: &[f64] = &[
    1_024.0,
    4_096.0,
    16_384.0,
    65_536.0,
    131_072.0,
    262_144.0,
    524_288.0,
    1_048_576.0,
    4_194_304.0,
    16_777_216.0,
    67_108_864.0,
];

/// Bucket boundaries for `live_edit_snapshot_read_entries` (entries read, Spec
/// SDD SR-8.2). `le=10000` brackets the entry count of a layer at the default
/// item ceiling (about 7,131 entries at Kitchen_set's 5.266 items per entry).
const SNAPSHOT_READ_ENTRIES_BUCKETS: &[f64] = &[
    10.0, 100.0, 500.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0, 25_000.0, 50_000.0, 100_000.0,
    250_000.0,
];

/// Bucket boundaries for `live_edit_structural_cache_replayed_entries`
/// (stream entries one repair replayed, structural-edits SDD SR-2.4). `le=0`
/// holds the repairs that found the entry already current; the powers of two
/// end at 1,024, `STRUCTURAL_REPAIR_MAX_TOTAL`, the most one repair replays
/// before it gives up with `too_far`, and `le=256` is one full repair read.
const STRUCTURAL_CACHE_REPLAYED_ENTRIES_BUCKETS: &[f64] = &[
    0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1_024.0,
];

/// Bucket boundaries for `live_edit_structural_cache_repair_bytes` (stream
/// bytes one repair read, structural-edits SDD SR-2.4). `le=0` holds the
/// repairs that read no entry; `le=262144` is one repair read at its first
/// byte bound (`STRUCTURAL_REPAIR_INITIAL_BYTES`, 256 KiB); `le=4194304` is
/// one repair read at its hard cap (`STRUCTURAL_REPAIR_MAX_BYTES`, 4 MiB),
/// and the boundaries above resolve repairs that took several such reads.
const STRUCTURAL_CACHE_REPAIR_BYTES_BUCKETS: &[f64] = &[
    0.0,
    1_024.0,
    4_096.0,
    16_384.0,
    65_536.0,
    262_144.0,
    1_048_576.0,
    4_194_304.0,
    16_777_216.0,
    67_108_864.0,
];

/// Bucket boundaries for `live_edit_update_layer_relay_bytes` (bytes): ×4 steps
/// from 1 KiB to 64 MiB, then 128 MiB, the default relay byte budget
/// (`LIVE_EDIT_LAYER_STREAM_MAX_BYTES`), so a commit above it lands past the
/// last finite boundary. Sizes the byte budget and the backing store from the
/// relay sizes a deployment actually sees.
const UPDATE_LAYER_RELAY_BYTES_BUCKETS: &[f64] = &[
    1_024.0,
    4_096.0,
    16_384.0,
    65_536.0,
    262_144.0,
    1_048_576.0,
    4_194_304.0,
    16_777_216.0,
    67_108_864.0,
    134_217_728.0,
];

/// Bucket boundaries for `live_edit_upload_object_state_bytes` (bytes): ×4
/// steps from 16 KiB to 1 GiB. A structural edit's object-state term runs
/// from tens of KB (the WATCH attempt's fixed cost, a small cached entry) to
/// hundreds of MB (a long rewrite on a large layer).
const UPLOAD_OBJECT_STATE_BYTES_BUCKETS: &[f64] = &[
    16_384.0,
    65_536.0,
    262_144.0,
    1_048_576.0,
    4_194_304.0,
    16_777_216.0,
    67_108_864.0,
    268_435_456.0,
    1_073_741_824.0,
];

/// Bucket boundaries for `live_edit_subscriber_delivery_lag_seconds` (seconds):
/// 1 ms to 2 min, so a delivery tens of seconds behind its commit still lands
/// in a finite bucket.
const SUBSCRIBER_DELIVERY_LAG_BUCKETS: &[f64] = &[
    0.001, 0.002_5, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0,
    60.0, 120.0,
];

/// Bucket boundaries for `live_edit_subscriber_send_wait_seconds` (seconds).
/// The top finite boundary is `OUTER_SEND_DEADLINE` (5 s), so a send that
/// waited out the whole deadline — a `slow_reader_deadline` tear-down — lands
/// above `le=5`.
const SUBSCRIBER_SEND_WAIT_BUCKETS: &[f64] = &[
    0.000_1, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
];

/// Bucket boundaries for `live_edit_subscriber_backlog_entries` (entries).
/// `le=0` holds a subscription whose queue was empty at every sample in the
/// window; `le=4096` is the soft entry threshold
/// (`storage::notifications::CONSUMER_CHANNEL_CAPACITY`) and `le=16384` the hard
/// cap (4× that).
const SUBSCRIBER_BACKLOG_ENTRIES_BUCKETS: &[f64] =
    &[0.0, 1.0, 4.0, 16.0, 64.0, 256.0, 1_024.0, 4_096.0, 16_384.0];

/// Bucket boundaries for `live_edit_subscriber_backlog_bytes` (bytes). `le=8388608`
/// is the soft byte threshold (`storage::notifications::CONSUMER_BACKLOG_BYTES`,
/// 8 MiB) and `le=33554432` the hard cap (32 MiB).
const SUBSCRIBER_BACKLOG_BYTES_BUCKETS: &[f64] = &[
    0.0,
    1_024.0,
    16_384.0,
    131_072.0,
    1_048_576.0,
    4_194_304.0,
    8_388_608.0,
    16_777_216.0,
    33_554_432.0,
];

/// Bucket boundaries for `live_edit_snapshot_fence_reader_wait_seconds`
/// (seconds, Spec SDD SR-8.3): a fenced reader's wait for its lease. Up to
/// `le=10` the boundaries are the snapshot-read ones without `0.0005`,
/// `0.0025`, `0.075`, `0.15` and `0.2`; they resolve the waits SR-5.5 bounds (one epoch, a gap, the
/// intents and a polling step: a few seconds). `le=30`, `le=60` and `le=120`
/// bracket `reader_wait_max(L)`, the cap a wait ends at (Spec SDD SR-11.4:
/// about 29 s at the shipped ceiling's floor, about 56 s for a 200k-entry
/// layer), so a wait that reached its cap lands in a band of its own rather
/// than in `+Inf`.
const SNAPSHOT_FENCE_READER_WAIT_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
];

/// The pinned `outcome` vocabulary of `live_edit_checkpoint_reconcile_total`
/// (b.9qd). Every cell is zero-seeded at `Metrics::new` from this list, and
/// `Metrics::add_checkpoint_reconcile` callers must use one of these values —
/// a label outside the list would materialise a series only on first
/// occurrence, which for a once-an-hour pass is indistinguishable from a pass
/// that never ran.
///
/// Per-outcome semantics live on the
/// [`Metrics::checkpoint_reconcile_total`] field doc.
pub const CHECKPOINT_RECONCILE_OUTCOMES: &[&str] = &[
    "orphan_deferred",
    "orphan_deleted",
    "temp_deleted",
    "dead_object_reaped",
    "dead_deferred",
    "dead_spared",
    "ext_tombstoned",
    "ext_collected",
    "ext_spared",
    "live_set_failed",
];

/// The pinned `phase` vocabulary of
/// `live_edit_checkpoint_adapter_ceiling_elapsed_total` (b.gg7).
///
/// Zero-seeded at `Metrics::new` from this list for the same reason the
/// reconciliation outcomes are: this counter is expected to read zero forever
/// on a healthy deployment, so an absent series and a never-elapsed ceiling
/// would be indistinguishable — and "the ceiling never fires" is precisely the
/// thing an operator needs to be able to confirm.
///
/// - `call` — one `send_classified` request (retries, backoff, and the 401
///   refresh included) hit `storage_api_publish_worst_case`.
/// - `enumeration` — a paginated LIST walk or its ext-refs sidecar GET fan-out
///   hit `storage_api_enumeration_ceiling`.
pub const CHECKPOINT_ADAPTER_CEILING_PHASES: &[&str] = &["call", "enumeration"];

/// The pinned `(verb, outcome)` vocabulary of
/// `live_edit_checkpoint_adapter_redirect_total` (b.z22).
///
/// Zero-seeded at `Metrics::new` from this list for the same reason the ceiling
/// phases are, with one extra twist: the *expected* reading of this counter is
/// not "zero forever" but "zero on the read arm, non-zero on the write arm for
/// large artifacts". `get`-side cells that never leave zero are the healthy
/// state (reads declare `download_preference=body`, so storage-api serves them
/// inline and the fallback never engages); a `get` cell that starts climbing
/// means the deployment's storage backend cannot honour the declared preference
/// and every large read is now paying an extra hop through blob storage. That
/// distinction is unreadable from an absent series.
///
/// The vocabulary is a pinned list of **pairs**, not a cross product: several
/// outcomes are structurally reachable on only one verb (there is no completion
/// call on a read, and no multipart alternative on a write's read path), and
/// seeding the cross product would publish cells that can never fire.
///
/// - `("get", "followed")` — a by-address GET answered 300 *despite* the
///   declared `download_preference=body`, and the adapter fetched the artifact
///   from the presigned target in one extra hop. Contract drift or a backend
///   that cannot serve inline; the read succeeded.
/// - `("get", "target_failed")` — the presigned target itself failed
///   (connect/DNS/timeout/5xx/403/404/409/412). Classified `Transient`, so the
///   retry — which runs inside the same ladder attempt's budget — re-runs the
///   by-address GET and obtains a *fresh* redirect rather than re-attempting an
///   expired presigned URL.
/// - `("get", "malformed")` — the 300 body did not carry a usable
///   `redirect_target_url`. `Permanent`: retrying cannot repair a contract
///   violation.
/// - `("put", "followed")` — a by-address PUT answered 300 with a write
///   redirect and the payload was delivered to the presigned target.
/// - `("put", "completed")` — the follow-up
///   `POST .../redirect/complete` handshake succeeded; the write is finished.
/// - `("put", "target_failed")` — the presigned target rejected or dropped the
///   payload.
/// - `("put", "complete_failed")` — the payload landed at the target but the
///   completion handshake failed. See the adapter's `run_write_redirect` for
///   why this is surfaced as an error even though the bytes are durable.
/// - `("put", "multipart_reissued")` — the server proposed a multipart upload
///   (payload at or above its `write_redirect_max_size`); the adapter re-issued
///   the PUT once with an explicit `upload_preference=redirect`, which
///   storage-api honours unconditionally, rather than implementing multipart.
/// - `("put", "malformed")` — the write-redirect protocol was violated in a way
///   that leaves the artifact unpublishable. Seven sites tick it, not only the
///   body parse: the 300 body did not parse; it carried neither a `redirect`
///   nor a `multipart` block; it named an upload method other than `put`/`post`;
///   its `redirect_target_url` was not an absolute `http(s)` URL; one of its
///   `additional_headers` was not a legal HTTP header; the server proposed
///   multipart *again* after an explicit `upload_preference=redirect`; or the
///   server answered a payload-free `upload_preference=redirect` request with a
///   non-300, which cannot be reported as a successful write because the
///   payload was never sent.
/// - `("put", "preflight_redirect")` — the upload-options pre-flight said this
///   payload's size does not travel inline, so the by-address PUT was issued
///   payload-free with an explicit `upload_preference=redirect`. The expected
///   steady state for large artifacts, and it should track `followed` /
///   `completed`.
/// - `("put", "preflight_unavailable")` — the pre-flight route could not be
///   consulted (missing on an older storage-api, no `FetchWriteTypeInfo`
///   permission, malformed answer, or a transient failure / 401 on the probe's
///   single attempt — the probe is deliberately un-retried, since every failure
///   lands on the same fallback). Writes fall back to sending the payload and
///   handling a 300 reactively, which still publishes; at most one tick per
///   re-probe window, so a steady trickle means the route is permanently
///   unavailable rather than that every write is probing.
/// - `("put", "preflight_stale")` — a write whose inline mode came from a
///   *cached pre-flight interval* got a 300 anyway, so the deployment's write
///   sizing changed under the cached answer. The redirect is handled reactively
///   and the cache is dropped. Strictly a contradiction counter: an inline write
///   the pre-flight never classified — one below the payload floor the probe is
///   worth paying for, one inside a re-probe backoff window, or one whose size
///   no published interval covers — takes the reactive path silently and never
///   ticks here, however many 300s it meets. The first such tick buys one
///   re-probe; a *second* tick inside the same re-probe window means the
///   deployment is contradicting its own published intervals, and the adapter
///   then suppresses the pre-flight for that window exactly as
///   `preflight_unavailable` describes — without ticking that cell, which stays
///   reserved for a route that did not answer. So a repeating pair-per-window
///   here is the signal that the two disagree persistently.
pub const CHECKPOINT_ADAPTER_REDIRECT_CELLS: &[(&str, &str)] = &[
    ("get", "followed"),
    ("get", "target_failed"),
    ("get", "malformed"),
    ("put", "followed"),
    ("put", "completed"),
    ("put", "target_failed"),
    ("put", "complete_failed"),
    ("put", "multipart_reissued"),
    ("put", "malformed"),
    ("put", "preflight_redirect"),
    ("put", "preflight_unavailable"),
    ("put", "preflight_stale"),
];

/// The pinned `outcome` vocabulary of `live_edit_object_id_counter_repair_total`
/// (b.b2j).
///
/// Zero-seeded at `Metrics::new` from this list for the same reason as the two
/// vocabularies above: this counter is expected to read zero forever on a
/// healthy deployment, so an absent series and a keyspace that has never been
/// lost would otherwise be indistinguishable — and "no repair has ever fired"
/// is exactly the assertion an operator alerting on the loss-evidence and
/// failure cells needs to be able to make from a first scrape.
///
/// Per-outcome semantics live on the
/// [`Metrics::object_id_counter_repair_total`] field doc.
pub const OBJECT_ID_COUNTER_REPAIR_OUTCOMES: &[&str] = &[
    "repaired_seeded",
    "repaired_noop",
    "repaired_initialized",
    "already_repaired",
    "failed",
    "failed_after_repair",
    "unhooked_initialized",
    "wait_timeout",
];

/// The pinned `outcome` vocabulary of `live_edit_checkpoint_tail_total`
/// (SDD SR-8.2).
///
/// Zero-seeded at `Metrics::new` from this list, and
/// [`Metrics::inc_checkpoint_tail`] callers must use a member: an unlisted
/// value materialises its series only on first occurrence, which for a refusal
/// outcome is indistinguishable from a feature that was never wired at all.
///
/// The sharper form of that argument for this family: on a healthy enabled
/// deployment every `refused_*` cell is expected to read ZERO forever, and
/// "tails have never been refused" is exactly the assertion an operator needs
/// to be able to make from a *first* scrape — before any tick has run. A
/// missing series cannot support that assertion; a zero one can.
///
/// `rebased` and `refused_chain_full` both have LIVE producers — a tick that
/// finds the chain at K attempts an SR-4.1 chain-collapse re-base, which ticks
/// the first when it lands and the second when it does not. Neither is reserved
/// and neither is retired.
///
/// `refused_lineage_unrootable` is the TENTH member, added by Epic 5's SR-7
/// adjudication to split the checkpoint-store-damage refusals out of
/// `refused_no_base` (SDD SR-7). The addition is **purely additive and the new
/// cell is zero-seeded from the first scrape**, so no dashboard, alert or recipe
/// keyed on the nine older cells changes meaning or needs migrating: the
/// rollout-tracker query on `refused_no_base` keeps counting exactly what it
/// counted, only more precisely, because the two arms that were never part of
/// the rollout population now report themselves.
///
/// Per-outcome semantics live on the [`Metrics::checkpoint_tail_total`] field
/// doc, the same way [`CHECKPOINT_RECONCILE_OUTCOMES`] delegates to its
/// counter's field doc.
pub const CHECKPOINT_TAIL_OUTCOMES: &[&str] = &[
    "captured",
    "captured_capped",
    "rebased",
    "refused_trimmed",
    "refused_missing_bytes",
    "refused_sentinel",
    "refused_seq_gap",
    "refused_no_base",
    "refused_chain_full",
    "refused_lineage_unrootable",
];

/// The pinned `outcome` vocabulary of
/// `live_edit_checkpoint_restore_lineage_total` (SDD SR-3.1).
///
/// Zero-seeded at `Metrics::new` from this list, and
/// [`Metrics::inc_checkpoint_restore_lineage`] callers must use a member, for
/// the same reason as the vocabularies above: an unlisted value materialises
/// its series only on first occurrence, and for a step-down outcome that is
/// indistinguishable from a feature that was never wired.
///
/// The sharper form for this family: on a healthy deployment every
/// `stepped_down_*` cell is expected to read ZERO forever, and "DR has never
/// had to serve older state than the store nominally holds" is precisely the
/// assertion an operator needs to make from a *first* scrape, before any
/// restore has run.
///
/// **No two members are merged.** Each names a distinct operator remediation —
/// a missing artifact (retention / partial store loss), a header that
/// contradicts its link (a torn or hand-edited artifact, or a non-conforming
/// adapter), an unreadable envelope (a version skew or bit-rot), an artifact
/// whose `.meta.json` companion never landed (a torn put on the write path), a
/// budget that ran out (adapter latency), and a composition the engine rejected
/// — so folding any pair would cost a diagnosis and buy nothing. The two
/// `exhausted*` members are the sharpest case: they differ in the answer the
/// client got, terminal `NOT_FOUND` versus retriable `UNAVAILABLE`, so folding
/// them would hide the difference between a layer with no usable backup and a
/// store that was too slow to say.
///
/// Per-outcome semantics live on the
/// [`Metrics::checkpoint_restore_lineage_total`] field doc, the same way
/// [`CHECKPOINT_TAIL_OUTCOMES`] delegates to its counter's field doc. The label
/// spellings are minted by `checkpoint::lineage`'s decision types
/// (`GatheredLineage::outcome_label`, `LineageBreak::reason_label`,
/// `Exhaustion::outcome_label`), which delegate here and must not re-spell this
/// list.
pub const CHECKPOINT_RESTORE_LINEAGE_OUTCOMES: &[&str] = &[
    "gathered_base",
    "gathered_composed",
    "stepped_down_link_broken",
    "stepped_down_link_mismatch",
    "stepped_down_decode",
    "stepped_down_missing_meta",
    "stepped_down_budget",
    "stepped_down_compose",
    "exhausted",
    "exhausted_budget",
];

/// The pinned `stage` vocabulary of
/// `live_edit_checkpoint_restore_fenced_total` (b.h4m).
///
/// Zero-seeded at `Metrics::new` from this list, and
/// [`Metrics::inc_checkpoint_restore_fenced`] callers must use a member — the
/// same argument as the vocabularies above, and with the sharper form: on a
/// single-replica or serialized deployment every cell is expected to read ZERO
/// forever, and "no two replicas have ever raced to restore the same layer" is
/// exactly the assertion an operator wants to make from a *first* scrape.
///
/// The three members are the three points **inside `apply_restore`** at which
/// one restore attempt can discover that a peer replica already landed this
/// object's rebuild; `residue_wipe` and `object_write` are therefore complete
/// counts, but `presence_probe` is a *partial* count of pre-work discoveries.
/// `RecoveryHook` has ONE short-circuit of its own — the post-admission
/// `is_layer_present` (§22.5 step 4) — that returns `Restored` without ever
/// reaching `apply_restore` and so ticks no cell here, and it catches most
/// same-replica and many cross-replica cases earlier. Its
/// post-stream-adjudication `is_layer_present` re-check is NOT a second one:
/// that arm returns `Ok(None)` and PROCEEDS into `apply_restore`, whose step 1
/// makes the same structural discovery and ticks `presence_probe` like any
/// other pre-work one. The three are ordered by how late the discovery is, and no two are
/// merged, because the distance between them is the operator's whole read on
/// how tight the race is:
///
/// * `presence_probe` — `apply_restore`'s step-1 structural header read saw a
///   complete header before this attempt did any work. The cheapest and by far
///   the most common outcome; a lazily-recovering fleet reaches it whenever a
///   sibling replica finished first.
/// * `residue_wipe` — the peer's header landed under the residue wipe's
///   `WATCH`, i.e. between step 1 and the wipe's `EXEC`. Only reachable on a
///   restore that found surviving residue.
/// * `object_write` — the peer's header landed under the materialization's own
///   `WATCH`, i.e. inside the last few round trips before this replica would
///   have written the object. This is the innermost fence, the one that closes
///   the mixture-of-two-backups hazard, and the only one of the two
///   `WATCH`-held fences reachable on a total-loss restore (where there is no
///   residue to wipe). `presence_probe` is not `WATCH`-held and every restore
///   reaches it, total loss included.
///
/// Per-stage semantics live on the
/// [`Metrics::checkpoint_restore_fenced_total`] field doc.
pub const CHECKPOINT_RESTORE_FENCED_STAGES: &[&str] = &[
    CHECKPOINT_RESTORE_FENCED_STAGE_PRESENCE_PROBE,
    CHECKPOINT_RESTORE_FENCED_STAGE_RESIDUE_WIPE,
    CHECKPOINT_RESTORE_FENCED_STAGE_OBJECT_WRITE,
];
/// `stage` label value for the fence at `apply_restore`'s step-1 structural
/// header read — the peer's rebuild was already complete before this attempt
/// did any work (b.h4m).
pub(crate) const CHECKPOINT_RESTORE_FENCED_STAGE_PRESENCE_PROBE: &str = "presence_probe";
/// `stage` label value for the fence at the step-3b residue wipe — the peer's
/// header landed under that wipe's `WATCH` (b.h4m). Only reachable on a restore
/// that found surviving residue.
pub(crate) const CHECKPOINT_RESTORE_FENCED_STAGE_RESIDUE_WIPE: &str = "residue_wipe";
/// `stage` label value for the innermost fence — the peer's header landed under
/// the materialization's own `WATCH` in `LayerStore::write_restored_object`
/// (b.h4m). The only one of the two `WATCH`-held fences a total-loss restore
/// can reach — `presence_probe` (step 1) is a non-fenced point-in-time check
/// every restore runs, total loss included — and the one that closes the
/// mixture-of-two-backups hazard.
pub(crate) const CHECKPOINT_RESTORE_FENCED_STAGE_OBJECT_WRITE: &str = "object_write";

/// The pinned `reason` vocabulary of
/// `live_edit_checkpoint_recovery_stream_unresolved_total` (b.4et).
///
/// Zero-seeded at `Metrics::new` from this list, and
/// [`Metrics::inc_checkpoint_recovery_stream_unresolved`] callers must use a
/// member — same argument as the vocabularies above, and sharper here for the
/// same reason as the lineage outcomes: every cell is expected to read ZERO
/// forever on a healthy deployment, and "DR has never had to refuse a restore
/// for want of stream evidence" is an assertion an operator needs to be able to
/// make from a *first* scrape.
///
/// **No two members are merged.** Each names a different operator situation and
/// a different remediation, and two of the three end the restore while the
/// third does not. Per-value semantics live on the
/// [`Metrics::checkpoint_recovery_stream_unresolved_total`] field doc.
pub const CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS: &[&str] =
    &["probe_unreadable", "unlink_unverified", "wrongtype_key"];

/// The pinned `result` vocabulary of
/// `live_edit_checkpoint_session_publish_superseded_total`.
///
/// Zero-seeded at `Metrics::new` from this list so every cell exists before
/// the first supersession, and
/// [`Metrics::inc_checkpoint_session_publish_superseded`] callers must use a
/// member. Per-value semantics live on the
/// [`Metrics::checkpoint_session_publish_superseded_total`] field doc.
///
/// `stale_incarnation` is deliberately NOT folded into `write`, even though
/// both repair by rebuilding: `write` means the ordering protocol worked and
/// caught a genuine race between two live builds, while `stale_incarnation`
/// means the coordination key itself was lost — a keyspace-integrity event
/// (FLUSH, failover onto a lagging replica, out-of-band DEL, TTL expiry) that
/// an operator wants to see on its own. It shares the wire spelling with
/// `session_checkpoint_publish.lua`'s refusal sentinel — see
/// [`CHECKPOINT_SESSION_PUBLISH_STALE_INCARNATION`].
pub const CHECKPOINT_SESSION_PUBLISH_SUPERSEDED_RESULTS: &[&str] = &[
    "write",
    "delete",
    CHECKPOINT_SESSION_PUBLISH_STALE_INCARNATION,
    "exhausted",
];

/// The one spelling of the stale-incarnation sentinel, shared by three places
/// that must agree byte-for-byte or the signal silently disappears.
///
/// `session_checkpoint_publish.lua` **returns** this string when the operation's
/// incarnation is not the one currently on `le:{sid}:ckpt`;
/// `checkpoint::session_writer` **decodes** it (as
/// `session_writer::STALE_INCARNATION_RESULT`, an alias for this const) to
/// classify the refusal; and it is a member of
/// [`CHECKPOINT_SESSION_PUBLISH_SUPERSEDED_RESULTS`], so it is also the
/// **metric label** the decode then reports under. A drift between the decoder
/// and the label would mint an unseeded cell no dashboard queries; a drift
/// between the decoder and Lua would send the refusal down the unrecognised-kind
/// arm instead. The Rust half of that agreement is now structural — the Lua
/// literal is the only remaining copy, and the Lua↔decoder edge is test-covered.
pub const CHECKPOINT_SESSION_PUBLISH_STALE_INCARNATION: &str = "stale_incarnation";

/// The pinned `outcome` vocabulary of
/// `live_edit_checkpoint_base_bootstrap_total` (SDD SR-7, the Epic-5
/// observability adjudication).
///
/// Zero-seeded at `Metrics::new` from this list, and
/// [`Metrics::inc_checkpoint_base_bootstrap`] callers must use a member, for the
/// same reason as the vocabularies above: an unlisted value materialises its
/// series only on first occurrence, and for an arm that is *supposed* to be
/// silent that is indistinguishable from a producer nobody wired.
///
/// The sharper form for this family: on a healthy deployment every `shed`,
/// every `unstamped_*` and `failed` cell reads ZERO forever, and **"no bootstrap
/// has ever been shed"** is exactly the assertion an operator needs to make from
/// a *first* scrape — before any layer has been created and before any restore
/// has run. A missing series cannot support that assertion; a zero one can. It
/// matters more here than for the sibling families because the shed's
/// concentration case is a mass restore against a degraded adapter, i.e. the
/// moment an operator is reading a dashboard for the first time under pressure.
///
/// The benign members (`persisted`, `skipped_*`) are not decoration: this family
/// is one-increment-per-invocation, so they are the DENOMINATOR that turns
/// `shed` into an admission-saturation ratio rather than a bare count.
///
/// Per-outcome semantics live on the
/// [`Metrics::checkpoint_base_bootstrap_total`] field doc, the same way
/// [`CHECKPOINT_TAIL_OUTCOMES`] delegates to its counter's field doc.
pub const CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES: &[&str] = &[
    "persisted",
    "shed",
    "skipped_header_absent",
    "skipped_generation_present",
    "skipped_untrusted_cursor",
    "unstamped_header_absent",
    "unstamped_generation_present",
    "unstamped_foreign_restore",
    "unstamped_write_error",
    "failed",
];

/// The pinned `site` vocabulary of `live_edit_checkpoint_base_bootstrap_total`
/// — which call site of the bootstrap unit an invocation was serving.
///
/// - `creation_base` — SR-7.1's creation-time first base, spawned by
///   `CreateLayer` post-commit.
/// - `restore_rebase` — SR-3.6's post-restore gap re-base, spawned by the
///   recovery hook's applied arm.
///
/// These spellings are minted by `checkpoint::base_bootstrap`'s
/// `BootstrapTarget::label()`, which delegates here and **must not re-spell
/// them** — the same discipline `GatheredLineage::outcome_label` follows against
/// [`CHECKPOINT_RESTORE_LINEAGE_OUTCOMES`]. The label doubles as the structured
/// log discriminator at that unit, so a metric cell and a log line join on one
/// string; re-spelling either side silently breaks that join.
pub const CHECKPOINT_BASE_BOOTSTRAP_SITES: &[&str] = &["creation_base", "restore_rebase"];

/// The pinned `kind` vocabulary of `live_edit_checkpoint_writes_total`.
///
/// Pinned here rather than spelled inline at the seeding loop for the same
/// reason as the vocabularies above: the grid is `result × kind`, and a `kind`
/// added at an increment site but not at the seeding loop renders as a series
/// that appears only once that write path first fires.
///
/// - `layer` — a layer **base** artifact: a full-state snapshot, or the fresh
///   base a re-base composes from an existing lineage. A re-base deliberately
///   has no separate kind, because what it writes *is* a base.
/// - `layer_tail` — a layer **tail** artifact (SDD SR-8.2): the stream-replay
///   generation that extends a lineage without rebuilding full state.
/// - `session` — a session-checkpoint blob.
/// - `counters_sidecar` — the object-id counters sidecar.
pub const CHECKPOINT_WRITE_KINDS: &[&str] = &["layer", "layer_tail", "session", "counters_sidecar"];

/// `phase` label value: a stream-multiplexer reader's `XREAD BLOCK`. On
/// `valkey_consumer_forced_reconnects_total` it means a client-side XREAD
/// timeout was read as a wedge; on `valkey_consumer_permanent_exits_total` it
/// means a reader hit a permanent XREAD error or an unconvertible XREAD
/// response.
pub const CONSUMER_PHASE_XREAD: &str = "xread";
/// `phase` label value on `valkey_consumer_forced_reconnects_total` only: a
/// multiplexed XREAD did not return within `stream_mux::WAKE_ACK_TIMEOUT` of
/// its wake landing, so the reader read its connection as wedged. The wake is
/// one a joining subscription's coverage needed, the liveness wake a join
/// sends when the reader's connection has not answered within the liveness age
/// or has failed since, or the credential wake a reader sends when a re-AUTH
/// falls due during a read.
pub const CONSUMER_PHASE_WAKE_ACK: &str = "wake_ack";
/// `phase` label value: the `PING` a reader sends before its first XREAD after
/// it had nothing to read, and on a cluster before its first XREAD bound to a
/// different hash slot (`stream_mux::READER_PROBE_TIMEOUT`). On
/// `valkey_consumer_forced_reconnects_total` it means the probe did not answer
/// in time, so the reader read its connection as wedged and re-probes under
/// its transient-failure budget; on `valkey_consumer_permanent_exits_total` it
/// means the probe failed with a permanent error, which ended the reader's
/// subscriptions (its heal is counted under `permanent`).
pub const CONSUMER_PHASE_PROBE: &str = "probe";
/// `phase` label value on `valkey_consumer_forced_reconnects_total` only: a
/// tick of the idle keepalive (`stream_mux::READER_KEEPALIVE_INTERVAL`) found a
/// `PING` that failed or did not answer in time on a reader with nothing to
/// read.
pub const CONSUMER_PHASE_KEEPALIVE: &str = "keepalive";
/// `phase` label value on `valkey_consumer_forced_reconnects_total` only: a
/// reader healed its connection after a failure that ended every subscription
/// on it — a permanent XREAD or probe error no single key explains,
/// `CROSSSLOT`, or an unconvertible XREAD response. Not a wedge verdict; the
/// failure itself is also counted on
/// `valkey_consumer_permanent_exits_total{phase="xread"|"probe"}`.
pub const CONSUMER_PHASE_PERMANENT: &str = "permanent";

/// `phase` label vocabulary for `valkey_consumer_forced_reconnects_total`.
/// Every value is pre-registered by [`Metrics::new`].
pub const CONSUMER_FORCED_RECONNECT_PHASES: &[&str] = &[
    CONSUMER_PHASE_XREAD,
    CONSUMER_PHASE_WAKE_ACK,
    CONSUMER_PHASE_PROBE,
    CONSUMER_PHASE_KEEPALIVE,
    CONSUMER_PHASE_PERMANENT,
];

/// `phase` label vocabulary for `valkey_consumer_permanent_exits_total`.
/// Every value is pre-registered by [`Metrics::new`].
pub const CONSUMER_PERMANENT_EXIT_PHASES: &[&str] = &[CONSUMER_PHASE_XREAD, CONSUMER_PHASE_PROBE];

/// `result` label on `live_edit_stream_mux_keepalives_total`: the tick's one
/// `PING`, on the reader's one connection, answered in time.
pub const KEEPALIVE_RESULT_OK: &str = "ok";
/// `result` label on `live_edit_stream_mux_keepalives_total`: the tick's one
/// `PING`, on the reader's one connection, failed or did not answer in time;
/// the reader force-reconnects (`phase="keepalive"`).
pub const KEEPALIVE_RESULT_FAILED: &str = "failed";
/// `result` label vocabulary for `live_edit_stream_mux_keepalives_total`.
/// Every value is pre-registered by [`Metrics::new`].
pub const KEEPALIVE_RESULTS: &[&str] = &[KEEPALIVE_RESULT_OK, KEEPALIVE_RESULT_FAILED];

/// `result` label on `live_edit_stream_mux_reauths_total`: the `AUTH` a stream
/// reader sent on its own connection between reads was answered OK.
pub const REAUTH_RESULT_OK: &str = "ok";
/// `result` label on `live_edit_stream_mux_reauths_total`: a stream reader's
/// re-AUTH tick could not fetch a credential, or the `AUTH` it sent failed or
/// did not answer in time. The connection is kept; nothing is reconnected.
pub const REAUTH_RESULT_FAILED: &str = "failed";
/// `result` label vocabulary for `live_edit_stream_mux_reauths_total`.
/// Every value is pre-registered by [`Metrics::new`].
pub const REAUTH_RESULTS: &[&str] = &[REAUTH_RESULT_OK, REAUTH_RESULT_FAILED];

/// `cause` label on `live_edit_stream_mux_reader_connects_total`: a reader with
/// no connection connected for the subscriptions it was bound to.
pub const READER_CONNECT_CAUSE_BIND: &str = "bind";
/// `cause` label on `live_edit_stream_mux_reader_connects_total`: a cluster
/// reader replaced its connection with one to a different primary: it bound a
/// hash slot another primary owns, or its slot's owner moved (a `MOVED`, or a
/// failover found before a forced reconnect).
pub const READER_CONNECT_CAUSE_RETARGET: &str = "retarget";
/// `cause` label vocabulary for `live_edit_stream_mux_reader_connects_total`.
/// Every value is pre-registered by [`Metrics::new`], crossed with
/// [`READER_CONNECT_RESULTS`].
pub const READER_CONNECT_CAUSES: &[&str] =
    &[READER_CONNECT_CAUSE_BIND, READER_CONNECT_CAUSE_RETARGET];
/// `result` label on `live_edit_stream_mux_reader_connects_total`: the
/// connection is up and the reader uses it.
pub const READER_CONNECT_RESULT_OK: &str = "ok";
/// `result` label on `live_edit_stream_mux_reader_connects_total`: the connect
/// failed or did not finish within `LIVE_EDIT_VALKEY_CONNECT_TIMEOUT_SECS`.
pub const READER_CONNECT_RESULT_FAILED: &str = "failed";
/// `result` label vocabulary for `live_edit_stream_mux_reader_connects_total`.
pub const READER_CONNECT_RESULTS: &[&str] =
    &[READER_CONNECT_RESULT_OK, READER_CONNECT_RESULT_FAILED];

/// `outcome` label on `live_edit_stream_mux_wakes_total`: the wake XADD landed.
pub const WAKE_OUTCOME_OK: &str = "ok";
/// `outcome` label on `live_edit_stream_mux_wakes_total`: every wake attempt failed.
pub const WAKE_OUTCOME_ERROR: &str = "error";
pub const WAKE_OUTCOMES: &[&str] = &[WAKE_OUTCOME_OK, WAKE_OUTCOME_ERROR];

/// `bound` label on `live_edit_stream_mux_overflow_kills_total`: the
/// subscription's undelivered backlog reached its entry limit (with or without
/// its byte limit too).
pub const OVERFLOW_BOUND_ENTRIES: &str = "entries";
/// `bound` label on `live_edit_stream_mux_overflow_kills_total`: the
/// subscription's undelivered backlog reached its byte limit while under its
/// entry limit — an ending an entry-only backlog would not have caused.
pub const OVERFLOW_BOUND_BYTES: &str = "bytes";
/// `bound` label vocabulary for `live_edit_stream_mux_overflow_kills_total`.
/// Every value is pre-registered by [`Metrics::new`].
pub const OVERFLOW_BOUNDS: &[&str] = &[OVERFLOW_BOUND_ENTRIES, OVERFLOW_BOUND_BYTES];

/// `result` label on `live_edit_stream_mux_head_checks_total`: at the check
/// the stream's head was past the read position, so the reply's first entry
/// sits above ids the stream no longer holds. Each subscription more than one
/// id below that entry is queued a trim marker and ends stream_gap unless its
/// catch-up covered those ids.
pub const HEAD_CHECK_RESULT_GAP: &str = "gap";
/// `result` label on `live_edit_stream_mux_head_checks_total`: the entries
/// before the reply's first one are still retained (an id jump, as a restore
/// band makes), or the stream holds no entry. No gap is claimed.
pub const HEAD_CHECK_RESULT_NO_GAP: &str = "no_gap";
/// `result` label on `live_edit_stream_mux_head_checks_total`: the check
/// failed or did not answer within the probe timeout. The key is not fanned
/// out from that reply and is re-read by the next XREAD; the failure is
/// charged to the reader's transient-failure budget.
pub const HEAD_CHECK_RESULT_ERROR: &str = "error";
/// `result` label vocabulary for `live_edit_stream_mux_head_checks_total`.
/// Every value is pre-registered by [`Metrics::new`].
pub const HEAD_CHECK_RESULTS: &[&str] = &[
    HEAD_CHECK_RESULT_GAP,
    HEAD_CHECK_RESULT_NO_GAP,
    HEAD_CHECK_RESULT_ERROR,
];

/// `result` label on `live_edit_ext_value_verify_total`: the external-value
/// reference's part matched both its `extValueSize` and its content hash.
pub const EXT_VALUE_VERIFY_OK: &str = "ok";
/// `result` label on `live_edit_ext_value_verify_total`: the part's 1 MiB
/// block content hash differs from the reference's `valueOrExtHash`.
pub const EXT_VALUE_VERIFY_HASH_MISMATCH: &str = "hash_mismatch";
/// `result` label on `live_edit_ext_value_verify_total`: the part's length
/// differs from the reference's `extValueSize`. Checked before the hash, so a
/// reference counted here was not hashed.
pub const EXT_VALUE_VERIFY_SIZE_MISMATCH: &str = "size_mismatch";
/// `result` label vocabulary for `live_edit_ext_value_verify_total`. Every
/// value is pre-registered by [`Metrics::new`].
pub const EXT_VALUE_VERIFY_RESULTS: &[&str] = &[
    EXT_VALUE_VERIFY_OK,
    EXT_VALUE_VERIFY_HASH_MISMATCH,
    EXT_VALUE_VERIFY_SIZE_MISMATCH,
];

/// `reason` label on `live_edit_checkpoint_ext_mirror_dropped_total`: the
/// mirror's channel held as many items as its capacity
/// (`LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY`).
pub const EXT_MIRROR_DROP_QUEUE_FULL: &str = "queue_full";
/// `reason` label on `live_edit_checkpoint_ext_mirror_dropped_total`: the
/// value's charge did not fit what is left of the mirror's byte budget.
pub const EXT_MIRROR_DROP_BUDGET_FULL: &str = "budget_full";
/// `reason` label on `live_edit_checkpoint_ext_mirror_dropped_total`: the
/// value's charge is larger than the whole byte budget, so it could never
/// fit; reachable only for a value larger than the current upload size cap.
pub const EXT_MIRROR_DROP_OVERSIZED: &str = "oversized";
/// `reason` label on `live_edit_checkpoint_ext_mirror_dropped_total`: the
/// value was still queued when the mirror workers stopped at shutdown.
pub const EXT_MIRROR_DROP_SHUTDOWN: &str = "shutdown";
/// `reason` label vocabulary for `live_edit_checkpoint_ext_mirror_dropped_total`.
/// Every value is pre-registered by [`Metrics::new`].
pub const EXT_MIRROR_DROP_REASONS: &[&str] = &[
    EXT_MIRROR_DROP_QUEUE_FULL,
    EXT_MIRROR_DROP_BUDGET_FULL,
    EXT_MIRROR_DROP_OVERSIZED,
    EXT_MIRROR_DROP_SHUTDOWN,
];

/// `state` label on `live_edit_checkpoint_ext_mirror_bytes`: charged to the
/// byte budget and waiting in the mirror's channel.
pub const EXT_MIRROR_STATE_QUEUED: &str = "queued";
/// `state` label on `live_edit_checkpoint_ext_mirror_bytes`: taken off the
/// channel by a mirror worker, its charge held until the last handle on the
/// value's memory is dropped (the adapter's request included).
pub const EXT_MIRROR_STATE_IN_FLIGHT: &str = "in_flight";
/// `state` label vocabulary for `live_edit_checkpoint_ext_mirror_bytes`.
/// Every value is pre-registered by [`Metrics::new`].
pub const EXT_MIRROR_STATES: &[&str] = &[EXT_MIRROR_STATE_QUEUED, EXT_MIRROR_STATE_IN_FLIGHT];

/// `outcome` label on `live_edit_checkpoint_ext_backup_repair_total`: the
/// backup store already held the bytes of a value the repair checked, and the
/// repair confirmed them with no upload. Counts both owed values found present
/// and the daily audit's re-checks of confirmed ones (about one per value per
/// day).
pub const EXT_BACKUP_REPAIR_ALREADY_STORED: &str = "already_stored";
/// `outcome` label on `live_edit_checkpoint_ext_backup_repair_total`: the
/// repair uploaded an owed value's bytes from Valkey.
pub const EXT_BACKUP_REPAIR_RESENT: &str = "resent";
/// `outcome` label on `live_edit_checkpoint_ext_backup_repair_total`: a
/// re-send failed and the value stays owed. A failed or timed-out upload also
/// stops the pass re-sending; a failed Valkey read, or a value whose length
/// changed between its size and its read, fails that value alone.
pub const EXT_BACKUP_REPAIR_RESEND_FAILED: &str = "resend_failed";
/// `outcome` label on `live_edit_checkpoint_ext_backup_repair_total`: an
/// existence check on the backup store failed; the pass stops checking and
/// re-sending, and finishes its scan for the owed gauge.
pub const EXT_BACKUP_REPAIR_PROBE_FAILED: &str = "probe_failed";
/// `outcome` label on `live_edit_checkpoint_ext_backup_repair_total`: Valkey
/// no longer held an owed value when the repair came to read it (it was
/// collected), so there is nothing to back up.
pub const EXT_BACKUP_REPAIR_VALUE_GONE: &str = "value_gone";
/// `outcome` label on `live_edit_checkpoint_ext_backup_repair_total`: the
/// mirror's byte budget had no room for a re-send for the whole wait; the
/// pass stops re-sending.
pub const EXT_BACKUP_REPAIR_BUDGET_BUSY: &str = "budget_busy";
/// `outcome` label on `live_edit_checkpoint_ext_backup_repair_total`: an
/// owed value is larger than the mirror's whole byte budget, so it cannot be
/// re-sent and stays owed.
pub const EXT_BACKUP_REPAIR_OVERSIZED: &str = "oversized";
/// `outcome` label on `live_edit_checkpoint_ext_backup_repair_total`: the
/// daily audit found a value confirmed in the backup store missing from it;
/// the value is re-sent like any owed one, unless re-sending has stopped for
/// the pass.
pub const EXT_BACKUP_REPAIR_CONFIRMED_COPY_MISSING: &str = "confirmed_copy_missing";
/// `outcome` label vocabulary for
/// `live_edit_checkpoint_ext_backup_repair_total`. Every value is
/// pre-registered by [`Metrics::new`].
pub const EXT_BACKUP_REPAIR_OUTCOMES: &[&str] = &[
    EXT_BACKUP_REPAIR_ALREADY_STORED,
    EXT_BACKUP_REPAIR_RESENT,
    EXT_BACKUP_REPAIR_RESEND_FAILED,
    EXT_BACKUP_REPAIR_PROBE_FAILED,
    EXT_BACKUP_REPAIR_VALUE_GONE,
    EXT_BACKUP_REPAIR_BUDGET_BUSY,
    EXT_BACKUP_REPAIR_OVERSIZED,
    EXT_BACKUP_REPAIR_CONFIRMED_COPY_MISSING,
];

/// `op` label on `live_edit_checkpoint_ext_backup_mark_failed_total`: writing
/// a value's backup confirmation (`backed_up_in`) failed, after a successful
/// upload (the mirror or the repair) or an existence check that found the
/// bytes (the repair). The value keeps whatever confirmation it had (none, for
/// a new copy) and the repair checks it again.
pub const EXT_BACKUP_MARK_OP_CONFIRM: &str = "confirm";
/// `op` label on `live_edit_checkpoint_ext_backup_mark_failed_total`: clearing
/// a value's backup confirmation after the checkpoint collect deleted its
/// backup copy failed.
pub const EXT_BACKUP_MARK_OP_CLEAR: &str = "clear";
/// `op` label vocabulary for
/// `live_edit_checkpoint_ext_backup_mark_failed_total`. Every value is
/// pre-registered by [`Metrics::new`].
pub const EXT_BACKUP_MARK_OPS: &[&str] = &[EXT_BACKUP_MARK_OP_CONFIRM, EXT_BACKUP_MARK_OP_CLEAR];

/// `pool` label vocabulary for `valkey_pool_waiters`: the `txn` permit queue
/// and the `subscriber_pool` label's `SubscribeReadLayer` admission queue.
/// `valkey_pool_checkout_seconds` is recorded for `txn` only.
pub const CHECKOUT_POOLS: &[&str] = &[VALKEY_POOL_TXN, VALKEY_POOL_SUBSCRIBER_POOL];

/// `pool` label value: the dedicated pub/sub connection (presence and
/// session-deleted channels).
pub const VALKEY_POOL_SUBSCRIBER: &str = "subscriber";
/// `pool` label value: the multiplexed data pool (`ValkeyClient::pool`).
pub const VALKEY_POOL_DATA: &str = "data";
/// `pool` label value: the WATCH/MULTI/EXEC `ExclusivePool`.
pub const VALKEY_POOL_TXN: &str = "txn";
/// `pool` label value: the stream multiplexer's reader connections, each
/// running one multi-key `XREAD BLOCK` (`storage::stream_mux`). On
/// `valkey_pool_waiters` the same label names the `SubscribeReadLayer`
/// admission queue.
pub const VALKEY_POOL_SUBSCRIBER_POOL: &str = "subscriber_pool";
/// `pool` label value: the GC / orphan-sweep maintenance pool.
pub const VALKEY_POOL_MAINTENANCE: &str = "maintenance";
/// `pool` label value: the multiplexed snapshot lane
/// (`ValkeyClient::snapshot_pool`), the dedicated connections for
/// object-snapshot reads.
pub const VALKEY_POOL_SNAPSHOT: &str = "snapshot";

/// `pool` label vocabulary shared by `valkey_reconnects_total`,
/// `valkey_errors_total`, `valkey_router_restarts_total` and
/// `valkey_unresponsive_total`: every Valkey connection pool the service
/// opens. [`Metrics::new`] seeds all four counters per pool from this one
/// list (`valkey_unresponsive_total` once per verdict as well).
/// `storage::client`'s `POOL_LABEL_*` constants alias these values.
pub const VALKEY_POOLS: &[&str] = &[
    VALKEY_POOL_SUBSCRIBER,
    VALKEY_POOL_DATA,
    VALKEY_POOL_TXN,
    VALKEY_POOL_SUBSCRIBER_POOL,
    VALKEY_POOL_MAINTENANCE,
    VALKEY_POOL_SNAPSHOT,
];

/// `verdict` label value on `valkey_unresponsive_total`: the fast verdict —
/// one connection silent for the fast bound while its shard answered other
/// connections (fred's error `"Unresponsive connection (fast verdict)."`).
pub const VALKEY_UNRESPONSIVE_VERDICT_FAST: &str = "fast";
/// `verdict` label value on `valkey_unresponsive_total`: the slow verdict —
/// no reply progress for the slow bound whatever other connections saw (a
/// shard silent for everyone) — or a bounded router wait that expired; fred
/// gives both stock fred's `"Unresponsive connection."`.
pub const VALKEY_UNRESPONSIVE_VERDICT_SLOW: &str = "slow";
/// `verdict` label vocabulary on `valkey_unresponsive_total`, seeded per
/// [`VALKEY_POOLS`] entry by [`Metrics::new`]. Each value is
/// `fred::live_edit::liveness::Verdict::as_str`.
pub const VALKEY_UNRESPONSIVE_VERDICTS: &[&str] = &[
    VALKEY_UNRESPONSIVE_VERDICT_FAST,
    VALKEY_UNRESPONSIVE_VERDICT_SLOW,
];

/// `pool` label vocabulary on `valkey_member_replacements_total`: the pools
/// whose members can be replaced by a fresh client (`storage::member_replacement`).
pub const VALKEY_REPLACEABLE_POOLS: &[&str] = &[
    VALKEY_POOL_DATA,
    VALKEY_POOL_TXN,
    VALKEY_POOL_MAINTENANCE,
    VALKEY_POOL_SNAPSHOT,
];

/// `outcome` label value on `valkey_member_replacements_total`: the member now
/// runs on a fresh, connected client, and the old one is being retired.
pub const VALKEY_MEMBER_REPLACEMENT_REPLACED: &str = "replaced";
/// `outcome` label value on `valkey_member_replacements_total`: the fresh
/// client did not connect within the connect timeout; the member keeps its
/// client.
pub const VALKEY_MEMBER_REPLACEMENT_CONNECT_FAILED: &str = "connect_failed";
/// `outcome` label value on `valkey_member_replacements_total`: a replacement
/// was due but the pool's replacement budget is used up — a multiplexed pool's
/// spare cells, or the `txn` pool's count of `max(size, 4)` replacements; the
/// member keeps its client until the process restarts.
pub const VALKEY_MEMBER_REPLACEMENT_BUDGET_EXHAUSTED: &str = "budget_exhausted";
/// `outcome` label vocabulary on `valkey_member_replacements_total`, seeded per
/// [`VALKEY_REPLACEABLE_POOLS`] entry by [`Metrics::new`].
pub const VALKEY_MEMBER_REPLACEMENT_OUTCOMES: &[&str] = &[
    VALKEY_MEMBER_REPLACEMENT_REPLACED,
    VALKEY_MEMBER_REPLACEMENT_CONNECT_FAILED,
    VALKEY_MEMBER_REPLACEMENT_BUDGET_EXHAUSTED,
];

/// `kind` value on `valkey_errors_total` for a Valkey command the service
/// issued whose client-side timeout expired, as observed by service code.
///
/// The one `kind` value this service owns; every other value on that counter is
/// fred's `ErrorKind::to_str()` from the connection-event hook. A distinct
/// snake_case value, not fred's `"Timeout Error"`, because the connection hook
/// already reports connect timeouts under that string: a dashboard must be able
/// to tell "cannot connect" apart from "the server stopped answering commands".
/// fred reports a timed-out command only to its caller, never on the event hook,
/// so this cell is ticked by `storage::client::FredErrOrigin::observe_error`
/// where service code receives the error.
pub const VALKEY_ERROR_KIND_COMMAND_TIMEOUT: &str = "command_timeout";

/// `endpoint` label value for `SubscribeReadLayer` on
/// `live_subscriber_updates_total` and `live_edit_join_latency_seconds`.
pub const ENDPOINT_SUBSCRIBE_READ_LAYER: &str = "SubscribeReadLayer";
/// `endpoint` label value for `JoinSession` on `live_edit_join_latency_seconds`.
pub const ENDPOINT_JOIN_SESSION: &str = "JoinSession";

/// `event_type` label value: a `PeerJoined` presence event.
pub const PRESENCE_EVENT_JOINED: &str = "joined";
/// `event_type` label value: a `PeerLeft` presence event.
pub const PRESENCE_EVENT_LEFT: &str = "left";
/// `event_type` label value: a `PeerMessage` presence event.
pub const PRESENCE_EVENT_MESSAGE: &str = "message";

/// `relay` label value on `live_edit_pubsub_relay_lagged_total`: the
/// replica-wide presence relay (`presence::spawn_presence_subscriber`), which
/// forwards `le:presence:events:{sid}` messages to the local presence router.
pub const PUBSUB_RELAY_PRESENCE: &str = "presence";
/// `relay` label value on `live_edit_pubsub_relay_lagged_total`: the
/// replica-wide session-deleted relay
/// (`grpc::session::spawn_session_delete_subscriber`), which applies
/// `le:session:deleted` notices to this replica's session tokens.
pub const PUBSUB_RELAY_SESSION_DELETED: &str = "session_deleted";

/// `relay` label vocabulary for `live_edit_pubsub_relay_lagged_total`: one
/// value per replica-wide pub/sub relay. [`Metrics::new`] seeds every value per
/// `service`.
pub const PUBSUB_RELAYS: &[&str] = &[PUBSUB_RELAY_PRESENCE, PUBSUB_RELAY_SESSION_DELETED];

/// `event_type` label vocabulary for `live_presence_events_total`: one value per
/// `PresenceEvent` variant. [`Metrics::new`] seeds every value per `service`.
pub const PRESENCE_EVENT_TYPES: &[&str] = &[
    PRESENCE_EVENT_JOINED,
    PRESENCE_EVENT_LEFT,
    PRESENCE_EVENT_MESSAGE,
];

/// `outcome` label value on `live_edit_presence_sweep_duration_seconds`: the
/// leader pass walked the whole session index. A store error on one session's
/// member page or chunk ends only that session, so such a pass still completes.
pub const PRESENCE_SWEEP_OUTCOME_COMPLETED: &str = "completed";
/// `outcome` label value on `live_edit_presence_sweep_duration_seconds`: the
/// pass stopped because its leader lease could not be proven held before a
/// chunk (see `live_edit_presence_sweep_lease_total`).
pub const PRESENCE_SWEEP_OUTCOME_LEASE_LOST: &str = "lease_lost";
/// `outcome` label value on `live_edit_presence_sweep_duration_seconds`: the
/// pass stopped at a page or chunk boundary because the replica is shutting
/// down.
pub const PRESENCE_SWEEP_OUTCOME_CANCELLED: &str = "cancelled";
/// `outcome` label value on `live_edit_presence_sweep_duration_seconds`: a
/// store error on the session index's `SSCAN` ended the pass.
pub const PRESENCE_SWEEP_OUTCOME_ERROR: &str = "error";

/// `outcome` label vocabulary for `live_edit_presence_sweep_duration_seconds`:
/// how one presence-sweeper leader pass ended. A histogram, so its cells cannot
/// be zero-seeded; each appears on its first observation.
pub const PRESENCE_SWEEP_OUTCOMES: &[&str] = &[
    PRESENCE_SWEEP_OUTCOME_COMPLETED,
    PRESENCE_SWEEP_OUTCOME_LEASE_LOST,
    PRESENCE_SWEEP_OUTCOME_CANCELLED,
    PRESENCE_SWEEP_OUTCOME_ERROR,
];

/// `outcome` label value on `live_edit_presence_sweep_lease_total`: a due
/// refresh found the presence-sweeper lock still this process's and extended
/// it.
pub const PRESENCE_SWEEP_LEASE_RENEWED: &str = "renewed";
/// `outcome` label value on `live_edit_presence_sweep_lease_total`: the lease
/// could not be proven held, and the pass ended before its next chunk.
pub const PRESENCE_SWEEP_LEASE_LOST: &str = "lost";
/// `outcome` label value on `live_edit_presence_sweep_lease_total`: the
/// refresh round-trip failed. Counted in addition to
/// [`PRESENCE_SWEEP_LEASE_LOST`], never instead of it.
pub const PRESENCE_SWEEP_LEASE_REFRESH_ERROR: &str = "refresh_error";

/// `outcome` label vocabulary for `live_edit_presence_sweep_lease_total`, the
/// same vocabulary as `live_edit_checkpoint_lease_total`. [`Metrics::new`]
/// seeds every value per `service`.
pub const PRESENCE_SWEEP_LEASE_OUTCOMES: &[&str] = &[
    PRESENCE_SWEEP_LEASE_RENEWED,
    PRESENCE_SWEEP_LEASE_LOST,
    PRESENCE_SWEEP_LEASE_REFRESH_ERROR,
];

/// `result` label value: the subscription was admitted and holds its admission
/// permit for its RPC lifetime (b.f4f). On `live_edit_upload_admission_total`:
/// the upload was admitted and holds its reservation for its RPC lifetime.
pub const ADMISSION_RESULT_ADMITTED: &str = "admitted";
/// `result` label value: every slot was taken and the bounded admission wait
/// (`LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS`) elapsed — or the wait is
/// configured to zero, the fail-fast policy (b.f4f). On
/// `live_edit_upload_admission_total`: upload capacity stayed full for the
/// whole `LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS` (or that wait is zero) — either
/// the replica's memory budget (derived from the pod memory limit) or the
/// session's limit on uploads still sending content.
pub const ADMISSION_RESULT_REFUSED_SATURATED: &str = "refused_saturated";
/// `result` label value: the waiter queue was already at
/// `LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS`, so the request was refused
/// without joining it (b.f4f). On `live_edit_upload_admission_total` the
/// queue's bound is `LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS`.
pub const ADMISSION_RESULT_REFUSED_WAITER_CAP: &str = "refused_waiter_cap";
/// `result` label value: admission observed a cancellation, for one of three
/// causes:
///
/// - the session was deleted: `NOT_FOUND "session was deleted"`;
/// - the service is shutting down (the subscription's enrolled shutdown token
///   fired — born cancelled for a subscription arriving after graceful
///   shutdown's paced stream ending began, or cancelled in its batch of that
///   ending while queued — or the pool's permit semaphore was closed by
///   `quit`): `UNAVAILABLE "server shutting down"`;
/// - the session went missing from the live store with no deletion marker (or
///   a marker read that failed): `UNAVAILABLE` with `session_absent_status()`'s
///   message, so the retry can recover it.
///
/// The causes are told apart on the wire (`refusal_to_status` in
/// `grpc::subscribe`) and in the handler's refusal line, but share this one
/// label: each means "the request went away, not the capacity", which is the
/// distinction this counter exists to draw. Separate `shutdown` or
/// `session_absent` values are deliberately not added.
///
/// On `live_edit_upload_admission_total` it means the same thing: an upload
/// refused admission, or refused while its processing step waited for room or
/// the overflow slot, because the service is shutting down (`UNAVAILABLE
/// "server shutting down"`), or one whose admission or processing wait was
/// dropped (a client reset, or the RPC deadline).
pub const ADMISSION_RESULT_CANCELLED: &str = "cancelled";

/// `result` label vocabulary for `subscriber_admission_total`. Closed and
/// complete over `ExclusiveCheckout::admit`'s return arms.
///
/// `refused_budget` is deliberately **absent**: the per-subscriber byte budget
/// lives in b.ntm, and a pre-registered value the shipped binary cannot emit is
/// a permanently-zero series that dashboards bind to as real. Adding a label
/// value later is the additive, safe direction.
pub const ADMISSION_RESULTS: &[&str] = &[
    ADMISSION_RESULT_ADMITTED,
    ADMISSION_RESULT_REFUSED_SATURATED,
    ADMISSION_RESULT_REFUSED_WAITER_CAP,
    ADMISSION_RESULT_CANCELLED,
];

/// `result` label value on `live_edit_upload_admission_total` only: an admitted
/// upload, still receiving its content, asked to grow its reservation and the
/// upload budget could not hold the growth at that moment. Receive growth
/// never waits, so the upload is refused `UNAVAILABLE` at once (stage `body`).
/// The processing steps are never refused for capacity: they wait instead.
pub const ADMISSION_RESULT_REFUSED_GROWTH: &str = "refused_growth";

/// `result` label vocabulary for `live_edit_upload_admission_total`: every
/// arm of `grpc::upload_admission`'s admission and receive-growth decisions,
/// plus `cancelled` for a processing wait. `cancelled` covers an admission or
/// a processing wait ended by shutdown, and a wait the caller dropped (a client
/// reset or the RPC deadline). No result is a refusal for size.
pub const UPLOAD_ADMISSION_RESULTS: &[&str] = &[
    ADMISSION_RESULT_ADMITTED,
    ADMISSION_RESULT_REFUSED_SATURATED,
    ADMISSION_RESULT_REFUSED_WAITER_CAP,
    ADMISSION_RESULT_REFUSED_GROWTH,
    ADMISSION_RESULT_CANCELLED,
];

/// `rpc` label value on the upload admission metrics: `CreateLayer`.
pub const UPLOAD_RPC_CREATE_LAYER: &str = "create_layer";
/// `rpc` label value on the upload admission metrics: `UpdateLayer`.
pub const UPLOAD_RPC_UPDATE_LAYER: &str = "update_layer";
/// `rpc` label vocabulary of `live_edit_upload_admission_total`,
/// `live_edit_upload_admission_wait_seconds`,
/// `live_edit_upload_overflow_runs_total` and
/// `live_edit_upload_processing_wait_seconds`.
pub const UPLOAD_RPCS: &[&str] = &[UPLOAD_RPC_CREATE_LAYER, UPLOAD_RPC_UPDATE_LAYER];

/// `phase` label value on `live_edit_uploads_in_flight`: admitted, still
/// receiving its content (holding its receive allowance and session slot).
pub const UPLOAD_PHASE_RECEIVING: &str = "receiving";
/// `phase` label value on `live_edit_uploads_in_flight`: content received,
/// being parsed, stored and committed.
pub const UPLOAD_PHASE_PROCESSING: &str = "processing";
/// `phase` label vocabulary of `live_edit_uploads_in_flight`.
pub const UPLOAD_PHASES: &[&str] = &[UPLOAD_PHASE_RECEIVING, UPLOAD_PHASE_PROCESSING];

/// `outcome` label value on `live_edit_update_layer_chain_wait_seconds`: the
/// chained call's predecessor (or its own operation) was the record, so the
/// call went on to its commit. Also a probe that met a wrong-typed key: the
/// commit refuses it.
pub const CHAIN_OUTCOME_READY: &str = "ready";
/// `outcome` label value on `live_edit_update_layer_chain_wait_seconds`: the
/// object's header was gone or superseded, so the call re-resolved its layer.
pub const CHAIN_OUTCOME_REPLACED: &str = "replaced";
/// `outcome` label value on `live_edit_update_layer_chain_wait_seconds`: a
/// `no_wait` call whose predecessor was not the record, refused at once
/// (`ABORTED PREDECESSOR_NOT_COMMITTED`).
pub const CHAIN_OUTCOME_REFUSED: &str = "refused";
/// `outcome` label value on `live_edit_update_layer_chain_wait_seconds`: the
/// wait reached its deadline cap with the predecessor still not the record
/// (`ABORTED PREDECESSOR_NOT_COMMITTED`).
pub const CHAIN_OUTCOME_TIMEOUT: &str = "timeout";
/// `outcome` label value on `live_edit_update_layer_chain_wait_seconds`: the
/// wait reached its deadline cap without one successful probe
/// (`UNAVAILABLE CHAIN_CHECK_UNAVAILABLE`).
pub const CHAIN_OUTCOME_UNAVAILABLE: &str = "unavailable";
/// `outcome` label value on `live_edit_update_layer_chain_wait_seconds`:
/// graceful shutdown began while the call waited (`UNAVAILABLE "server
/// shutting down"`).
pub const CHAIN_OUTCOME_SHUTDOWN: &str = "shutdown";
/// `outcome` label value on `live_edit_update_layer_chain_wait_seconds`: the
/// call was dropped while it waited — a client cancel, a lost connection or
/// the RPC deadline. Nothing was applied.
pub const CHAIN_OUTCOME_CANCELLED: &str = "cancelled";
/// `outcome` label vocabulary of `live_edit_update_layer_chain_wait_seconds`:
/// every exit of the chain gate (`storage::chain_gate`).
pub const CHAIN_OUTCOMES: &[&str] = &[
    CHAIN_OUTCOME_READY,
    CHAIN_OUTCOME_REPLACED,
    CHAIN_OUTCOME_REFUSED,
    CHAIN_OUTCOME_TIMEOUT,
    CHAIN_OUTCOME_UNAVAILABLE,
    CHAIN_OUTCOME_SHUTDOWN,
    CHAIN_OUTCOME_CANCELLED,
];

/// `result` label value on `live_edit_update_layer_chain_probes_total`: the
/// probe read the writer's record and the object's header.
pub const CHAIN_PROBE_OK: &str = "ok";
/// `result` label value on `live_edit_update_layer_chain_probes_total`: the
/// probe failed; the gate keeps waiting.
pub const CHAIN_PROBE_ERROR: &str = "error";
/// `result` label vocabulary of `live_edit_update_layer_chain_probes_total`.
pub const CHAIN_PROBE_RESULTS: &[&str] = &[CHAIN_PROBE_OK, CHAIN_PROBE_ERROR];

/// `result` label value on `live_edit_subscribe_catchup_wait_seconds`: the
/// catch-up got its concurrent-catch-up slot (including the uncontended fast
/// path, which records ~0).
pub const CATCHUP_WAIT_RESULT_ACQUIRED: &str = "acquired";
/// `result` label value on `live_edit_subscribe_catchup_wait_seconds`: the wait
/// ended without a slot because the gate closed at shutdown, or because the
/// waiting subscription stopped waiting (shutdown, session deletion or client
/// disconnect) — for a shared fenced-read flight's slot wait, because the
/// last catch-up attached to the flight left.
pub const CATCHUP_WAIT_RESULT_CANCELLED: &str = "cancelled";
/// `result` label value on `live_edit_subscribe_catchup_wait_seconds`: no slot
/// freed before the subscription's capacity budget
/// (`LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS`) ran out, so the subscription is
/// refused [`STREAM_EXIT_CATCHUP_CAPACITY`]. Never recorded for a shared
/// fenced-read flight's slot wait, which has no deadline.
pub const CATCHUP_WAIT_RESULT_TIMED_OUT: &str = "timed_out";

/// `class` label value on `live_edit_catchup_snapshot_reads_in_flight`: a
/// catch-up atomic snapshot read holding one of the
/// `LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY` read permits (Spec SDD SR-7.1,
/// SR-8.5).
pub const CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC: &str = "atomic";
/// `class` label value on `live_edit_catchup_snapshot_reads_in_flight`: a
/// shared fenced-read flight's read stage holding a fenced-class read permit
/// (`LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY`; Spec SDD SR-7.1), from before
/// a fence acquire through the fenced read, the fence's release and the
/// memory adjust after it. Pre-registered at zero.
pub const CATCHUP_SNAPSHOT_READ_CLASS_FENCED: &str = "fenced";

/// The complete `class` vocabulary of
/// `live_edit_catchup_snapshot_reads_in_flight`, seeded at zero in
/// [`Metrics::new`], and of `live_edit_catchup_snapshot_read_wait_seconds`.
pub const CATCHUP_SNAPSHOT_READ_CLASSES: &[&str] = &[
    CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC,
    CATCHUP_SNAPSHOT_READ_CLASS_FENCED,
];

/// `result` label value on `live_edit_catchup_build_admission_total`: a
/// catch-up memory reservation (`storage::memory_budget::MemoryBudget`) granted
/// without queueing, a waiting reservation's fast path or a non-blocking grant
/// (Spec SDD SR-7.2, SR-8.5).
pub const CATCHUP_BUILD_ADMISSION_RESULT_IMMEDIATE: &str = "immediate";
/// `result` label value on `live_edit_catchup_build_admission_total`: a
/// reservation granted after queueing for memory to free.
pub const CATCHUP_BUILD_ADMISSION_RESULT_WAITED: &str = "waited";
/// `result` label value on `live_edit_catchup_build_admission_total`: a
/// reservation's deadline passed before memory freed (or had passed on entry
/// with none free), so it was refused.
pub const CATCHUP_BUILD_ADMISSION_RESULT_REFUSED_TIMEOUT: &str = "refused_timeout";
/// `result` label value on `live_edit_catchup_build_admission_total`: a
/// reservation that ended without memory because the budget was closed at
/// shutdown, or because its caller stopped waiting (shutdown, session
/// cancellation, client disconnect).
pub const CATCHUP_BUILD_ADMISSION_RESULT_CANCELLED: &str = "cancelled";

/// The complete `result` vocabulary of
/// `live_edit_catchup_build_admission_total`, seeded at zero in
/// [`Metrics::new`].
pub const CATCHUP_BUILD_ADMISSION_RESULTS: &[&str] = &[
    CATCHUP_BUILD_ADMISSION_RESULT_IMMEDIATE,
    CATCHUP_BUILD_ADMISSION_RESULT_WAITED,
    CATCHUP_BUILD_ADMISSION_RESULT_REFUSED_TIMEOUT,
    CATCHUP_BUILD_ADMISSION_RESULT_CANCELLED,
];

/// `outcome` label value on `live_edit_retirement_wait_seconds`: a retirement
/// reservation (the retirement allowance `R_ref`, a
/// `storage::memory_budget::MemoryBudget` of its own) granted without
/// queueing.
pub const RETIREMENT_WAIT_OUTCOME_IMMEDIATE: &str = "immediate";
/// `outcome` label value on `live_edit_retirement_wait_seconds`: a retirement
/// reservation granted after queueing for the allowance to free.
pub const RETIREMENT_WAIT_OUTCOME_WAITED: &str = "waited";
/// `outcome` label value on `live_edit_retirement_wait_seconds`: a retirement
/// reservation that ended without memory: shutdown ended the allowance's waits
/// while it was queued or would have had to wait, or its caller stopped
/// waiting (a `DeleteLayer` past its deadline or cancelled, the orphan sweep
/// cancelled).
pub const RETIREMENT_WAIT_OUTCOME_CANCELLED: &str = "cancelled";

/// The complete `outcome` vocabulary of `live_edit_retirement_wait_seconds`.
pub const RETIREMENT_WAIT_OUTCOMES: &[&str] = &[
    RETIREMENT_WAIT_OUTCOME_IMMEDIATE,
    RETIREMENT_WAIT_OUTCOME_WAITED,
    RETIREMENT_WAIT_OUTCOME_CANCELLED,
];

/// `role` label value on `live_edit_catchup_snapshot_flight_total`: a
/// catch-up that created a shared snapshot-read flight
/// (`storage::snapshot_flights`) — the first for its object on this replica,
/// or the successor of a flight closed to attach — and so caused one read
/// (Spec SDD SR-8.4).
pub const CATCHUP_SNAPSHOT_FLIGHT_ROLE_LEADER: &str = "leader";
/// `role` label value on `live_edit_catchup_snapshot_flight_total`: a
/// catch-up that attached to a flight another catch-up created, sharing its
/// read instead of reading itself (Spec SDD SR-8.4).
pub const CATCHUP_SNAPSHOT_FLIGHT_ROLE_JOINED_READ: &str = "joined_read";

/// The complete `role` vocabulary of `live_edit_catchup_snapshot_flight_total`,
/// seeded at zero in [`Metrics::new`]. `joined_build` (a catch-up served by a
/// flight's shared build, Spec SDD SR-8.4) is deliberately absent: no build is
/// shared in this binary — each attached catch-up builds its own payload — and
/// a pre-registered value nothing can emit is a permanently-zero series that
/// dashboards bind to as real.
pub const CATCHUP_SNAPSHOT_FLIGHT_ROLES: &[&str] = &[
    CATCHUP_SNAPSHOT_FLIGHT_ROLE_LEADER,
    CATCHUP_SNAPSHOT_FLIGHT_ROLE_JOINED_READ,
];

/// `regime` label vocabulary of `live_subscribe_stable_cut_total`: the
/// catch-up regimes that read a layer's state (`up_to_date` reads none), spelled as on
/// `live_subscribe_regime_total`.
pub const SUBSCRIBE_STABLE_CUT_REGIMES: &[&str] = &["brand_new", "in_window", "out_of_window"];

/// `outcome` label vocabulary of `live_subscribe_stable_cut_total`.
pub const SUBSCRIBE_STABLE_CUT_OUTCOMES: &[&str] = &["verified", "unverified"];

/// `reason` label value: the outer gRPC send could not complete within
/// `OUTER_SEND_DEADLINE` because the client stopped reading (b.f4f).
pub const STREAM_EXIT_SLOW_READER_DEADLINE: &str = "slow_reader_deadline";
/// `reason` label value: the stream reader ended the subscription because the
/// next entry would have taken its undelivered backlog past the hard cap, in
/// entries or bytes, or — once the subscription had started draining — the
/// backlog stayed at or above the soft threshold past `CONSUMER_SEND_WATCHDOG`
/// (b.jr8, b.8z2; which limit is on
/// `live_edit_stream_mux_overflow_kills_total{bound}`). A catch-up's stream is
/// registered at its proven read's cut (its own atomic read's, or the cut of
/// the shared fenced read of the flight it is attached to), after every
/// capacity wait, so a catch-up waiting for capacity is never ended this way.
/// After that read only the hard caps apply until the subscription drains:
/// one ended while the catch-up waits for its build turn in its flight, or
/// during its build, ends before any content, because the catch-up checks for
/// it while it waits for its turn and again before its first message; one
/// ended during the content send itself surfaces after the caught-up ack,
/// which the caps are sized to rule out at the design write rate.
pub const STREAM_EXIT_INNER_WATCHDOG: &str = "inner_watchdog";
/// `reason` label value: the session was deleted — the per-session cancellation
/// token fired for an explicit `DeleteSession`, locally, through the
/// cross-replica fan-out, or through a canceller that found the session
/// missing from the store and carrying `DeleteSession`'s deletion marker (the
/// `session_tokens` sweeper, or a streaming RPC's setup that got there before
/// the fan-out). Graceful shutdown never cancels that token, and a cancel for
/// a session missing from the store without that marker counts under
/// [`STREAM_EXIT_SESSION_ABSENT`], so this value means deletion only. The
/// stream ends `NOT_FOUND "session was deleted"`. It does not count every
/// deletion: a subscriber that reads the deletion's `deleted` sentinel off its
/// layer stream before it sees the token ends `NOT_FOUND "layer '<name>' was
/// deleted"` and counts under [`STREAM_EXIT_ERROR`].
pub const STREAM_EXIT_SESSION_CANCEL: &str = "session_cancel";
/// `reason` label value: the per-session cancellation token fired because the
/// session was found missing from the live store with no `DeleteSession` seen
/// and no deletion marker (or a marker read that failed) — the
/// `session_tokens` sweeper, or a streaming RPC's setup finding it gone even
/// after a pass through the session recovery hook. A keyspace loss that a
/// checkpoint can restore produces it. The stream ends `UNAVAILABLE "session is
/// missing from the live store; retry so it can be recovered"`, which clients
/// retry. The retry's setup runs the session recovery hook on whichever replica
/// serves it, so with checkpointing enabled a session that has a checkpoint is
/// restored, and a session that is really gone is refused `NOT_FOUND` at that
/// retry's setup.
pub const STREAM_EXIT_SESSION_ABSENT: &str = "session_absent";
/// `reason` label value: graceful shutdown — the subscription's enrolled
/// shutdown token fired (SDD §5.6). After the drain and the server's GOAWAY,
/// the paced stream ending cancels each live stream's token in its own batch,
/// in random order, over at most
/// `LIVE_EDIT_SHUTDOWN_STREAM_END_MAX_WINDOW_SECS`: at
/// `LIVE_EDIT_SHUTDOWN_STREAM_END_RATE` per second, or at (live streams) /
/// that window once the cap binds. This label's exits on a draining replica
/// therefore ramp over that window rather than arriving in one step. A
/// subscription still registered when the store pools quit is ended by the
/// stream multiplexer's shutdown, under this label too. The stream ends
/// `UNAVAILABLE`, which clients retry; where the retry lands is the ingress's
/// choice.
pub const STREAM_EXIT_SHUTDOWN: &str = "shutdown";
/// `reason` label value: the response receiver was dropped (client went away).
pub const STREAM_EXIT_CLIENT_DISCONNECT: &str = "client_disconnect";
/// `reason` label value: in cluster mode every stream reader stayed bound to
/// another hash slot for the whole capacity budget
/// (`LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS`), so the subscription could not
/// take a stream reader. The stream ends `UNAVAILABLE`, which clients retry:
/// before any content during catch-up; after the caught-up ack only for a
/// steady-state overwrite re-bootstrap.
pub const STREAM_EXIT_READER_CAPACITY: &str = "reader_capacity";
/// `reason` label value: no concurrent-catch-up slot
/// (`LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS`), or no catch-up snapshot
/// read permit (`LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY`), freed within the
/// subscription's capacity budget (`LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS`).
/// The stream ends `UNAVAILABLE`, which clients retry: before any content during
/// catch-up; after the caught-up ack only for a steady-state overwrite
/// re-bootstrap. A slot wait is recorded as [`CATCHUP_WAIT_RESULT_TIMED_OUT`];
/// a permit wait on `live_edit_catchup_snapshot_read_wait_seconds`, and the two
/// statuses name their own limit. A catch-up attached to a shared fenced-read
/// flight (`storage::snapshot_flights`) is refused under this reason too when
/// its budget runs out while the flight waits for its slot, its fenced read
/// permit (`LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY`) or its memory (the
/// catch-up memory budget, derived from the pod memory limit); the flight's own wait
/// carries no deadline and records no `timed_out`, so that refusal shows only
/// here, and its status names those three waits. A catch-up that leaves its
/// flight because the read's work passed `reader_wait_max(L)` counts under
/// [`STREAM_EXIT_FENCED_READ_WAIT_CAP`] instead.
pub const STREAM_EXIT_CATCHUP_CAPACITY: &str = "catchup_capacity";
/// `reason` label value: a catch-up attached to a shared fenced-read flight
/// (`storage::snapshot_flights`) of a layer above the atomic-read ceilings
/// left it because the flight's work — fence acquires, `RETRY_AFTER` sleeps
/// and the fenced read, counted from the end of any predecessor flight —
/// passed `reader_wait_max(L)` (Spec SDD SR-5.5, SR-11.4), logged at WARN with
/// the time counted. The flight goes on for the catch-ups that stay. The
/// stream ends `UNAVAILABLE`, which clients retry, always before any content.
/// Not a capacity refusal: the flight's capacity waits are not counted
/// against this cap (they are charged to the capacity budget and refused
/// [`STREAM_EXIT_CATCHUP_CAPACITY`]); fence waits stay inside the cap under
/// the design's fence bounds, so any increase means a layer's fence was
/// held off far longer than SR-5.5 bounds.
pub const STREAM_EXIT_FENCED_READ_WAIT_CAP: &str = "fenced_read_wait_cap";
/// `reason` label value: the layer stream trimmed a commit the subscription
/// was owed — one above both its caught-up ack and the last commit it was
/// sent — before its stream reader read it. The stream ends `UNAVAILABLE
/// "layer stream was trimmed past this subscription's position; reconnect to
/// catch up"`, which clients retry with a full catch-up. It is decided when the
/// subscription reads its stream, so it comes after the caught-up ack, and a
/// trim the catch-up covered never ends a subscription this way (the
/// `live_edit_stream_mux_head_checks_total{result="gap"}` count includes
/// those covered trims). The same ending follows a stream that was removed and
/// recreated by a later commit (a restore after a keyspace loss, a stream-key
/// removal). There the reconnect's catch-up is the resync onto the layer's
/// current, restored content, and it is redundant only when the restore lost
/// nothing the client held.
pub const STREAM_EXIT_STREAM_GAP: &str = "stream_gap";
/// `reason` label value: every other terminal exit — catch-up failure, a
/// permanent Valkey error, retry-budget exhaustion, `NOT_FOUND` on a deleted
/// layer.
pub const STREAM_EXIT_ERROR: &str = "error";

/// `reason` label vocabulary for `subscriber_stream_exits_total`.
///
/// Closed and **complete over `run_subscription`'s terminal arms**, which is the
/// property that makes `sum by (reason)` equal total subscription exits — every
/// exit path classifies into exactly one of these before the RPC task returns.
///
/// `budget_exhausted` is deliberately absent, for the same reason
/// `refused_budget` is absent from [`ADMISSION_RESULTS`].
pub const STREAM_EXIT_REASONS: &[&str] = &[
    STREAM_EXIT_SLOW_READER_DEADLINE,
    STREAM_EXIT_INNER_WATCHDOG,
    STREAM_EXIT_SESSION_CANCEL,
    STREAM_EXIT_SESSION_ABSENT,
    STREAM_EXIT_SHUTDOWN,
    STREAM_EXIT_CLIENT_DISCONNECT,
    STREAM_EXIT_READER_CAPACITY,
    STREAM_EXIT_CATCHUP_CAPACITY,
    STREAM_EXIT_FENCED_READ_WAIT_CAP,
    STREAM_EXIT_STREAM_GAP,
    STREAM_EXIT_ERROR,
];

/// `stage` label value on `live_edit_subscriber_delivery_lag_seconds`: commit
/// to the moment the subscription task enqueued the commit's last message on
/// its response channel.
pub const DELIVERY_LAG_STAGE_ENQUEUED: &str = "enqueued";
/// `stage` label value on `live_edit_subscriber_delivery_lag_seconds`: commit
/// to the moment tonic's encoder polled the commit's last message out of the
/// response stream, on its way to the service's HTTP/2 layer.
pub const DELIVERY_LAG_STAGE_TRANSPORT: &str = "transport";

/// The complete `stage` vocabulary of
/// `live_edit_subscriber_delivery_lag_seconds`. A histogram, so nothing is
/// pre-registered (SDD §17.1); a stage's series appears with its first window.
pub const DELIVERY_LAG_STAGES: &[&str] =
    &[DELIVERY_LAG_STAGE_ENQUEUED, DELIVERY_LAG_STAGE_TRANSPORT];

/// `phase` label value on `live_edit_subscriber_send_stalls_total` and
/// `live_edit_subscriber_send_wait_seconds`: a send of catch-up content or of a
/// caught-up ack, including a steady-state re-bootstrap's and a rollback-mode
/// in-window replay's. Multi-chunk catch-ups fill the response channel
/// routinely, so this phase stalls in normal operation.
pub const SEND_PHASE_CATCHUP: &str = "catchup";
/// `phase` label value on `live_edit_subscriber_send_stalls_total` and
/// `live_edit_subscriber_send_wait_seconds`: a send of a steady-state live
/// delivery (a relayed commit or an empty commit's delta).
pub const SEND_PHASE_LIVE: &str = "live";

/// The complete `phase` vocabulary of `live_edit_subscriber_send_stalls_total`
/// (both cells seeded at zero in [`Metrics::new`]) and
/// `live_edit_subscriber_send_wait_seconds` (a histogram, not pre-registered).
pub const SEND_PHASES: &[&str] = &[SEND_PHASE_CATCHUP, SEND_PHASE_LIVE];

/// `site` label value for a `WatchGuard` UNWATCH issued explicitly by a call
/// site on a precondition-failure / early-return path (b.ji8). A failure here
/// is recoverable: the teardown API leaves the guard armed, so `Drop` tries
/// again with the pool mutex still held.
///
/// Lives here, not in `storage::watch_guard`, because the label vocabulary
/// belongs to the counter it labels — and because `observability` must gain no
/// dependency edge into `storage` (CLAUDE.md § Module Dependency Rules).
/// [`Metrics::new`] pre-registers both cells; `watch_guard` imports them.
pub(crate) const UNWATCH_SITE_EXPLICIT: &str = "explicit";
/// `site` label value for the `Drop`-time `WatchGuard` UNWATCH (b.ji8). A
/// failure here is terminal — the connection returns to the pool carrying
/// stale WATCH state — so this is the cell operator alerts fire on. Metric-neutral
/// for the same reason as its sibling above.
pub(crate) const UNWATCH_SITE_DROP: &str = "drop";

// ── snapshot_read_type_error_total `key_family` vocabulary (b.atj) ──────────
//
// One value per key family `read_object_snapshot`'s reads touch — its atomic
// script and its bounded fan-out alike: the header, the four indices, the
// section, tombstone, field and time-sample rows, the children lists, and
// `children_index` (`obj:chs`, on a marked object) — plus `ext_index`
// (`obj:ext`), which no snapshot read touches and only external-value
// clean-up reads (its size read and its reference walk), named for the
// key family (SDD §7.1) rather than for the command, so an operator reading the
// series knows which part of a layer's state is unreadable. Lives here, not in
// `storage::diff_read`, for the reason [`UNWATCH_SITE_EXPLICIT`] gives: the
// label vocabulary belongs to the counter it labels, and `observability` must
// gain no dependency edge into `storage` (CLAUDE.md § Module Dependency Rules).
// `storage::diff_read::SnapshotKeyFamily::label` returns these constants — the
// enum never mints a label string of its own — and three guards named on that
// type keep the two sides in step. Each is weaker than the last, and the ladder
// as a whole is weaker than it looks, so what each one actually enforces:
//
//   1. `label`'s match has no `_` arm, so a new variant must be given a LABEL
//      before the crate compiles. That is the only step the compiler forces
//      unconditionally.
//   2. The `const` length assertion beside `SnapshotKeyFamily::ALL` compares
//      `ALL`'s fixed length against this array's — but it only fires once the
//      new variant has been LISTED in `ALL`. Listing it is a convention the
//      assertion rewards, not one the compiler imposes.
//   3. The unit test beside `ALL` pins the two as the same *set* rather than the
//      same cardinality. It iterates `ALL`, so it too sees only listed variants.
//
// Residual hole, closed by review rather than by the build: a variant given a
// label and a call site but added to NEITHER `ALL` nor this array compiles
// clean and leaves the unit test passing, while emitting a `key_family` value
// that was never registered here — an un-seeded cell, which defeats the zero
// pre-seeding the `increase(...) > 0` alert recipe depends on. The review-time
// check is therefore "every `SnapshotKeyFamily` variant appears in `ALL`". The
// only shape that would close it mechanically is a declarative macro minting the
// enum, `ALL` and these constants from one list — deliberately not built for
// twelve values.

/// `le:{object_id}:obj:hdr` — the object header hash (`HGETALL`). Also carries
/// the `HGET … seq` reads of the same key: the checkpoint assembler's pin and
/// the structural path's proof after its unfenced read.
pub(crate) const SNAPSHOT_KEY_FAMILY_HEADER: &str = "header";
/// `le:{object_id}:obj:secs` — the live-section id index (`SMEMBERS`).
pub(crate) const SNAPSHOT_KEY_FAMILY_SECTIONS_INDEX: &str = "sections_index";
/// `le:{object_id}:obj:dels` — the deleted-section id index (`SMEMBERS`).
pub(crate) const SNAPSHOT_KEY_FAMILY_DELETED_SECTIONS_INDEX: &str = "deleted_sections_index";
/// `le:{object_id}:obj:sec:{section_id}` — one live section's hash (`HGETALL`).
pub(crate) const SNAPSHOT_KEY_FAMILY_SECTION: &str = "section";
/// `le:{object_id}:obj:del:{section_id}` — one tombstone's hash (`HGETALL`).
pub(crate) const SNAPSHOT_KEY_FAMILY_DELETED_SECTION: &str = "deleted_section";
/// `le:{object_id}:obj:ch:{parent_id}` — one parent's children list (`LRANGE`).
/// The b.zvd wedge shape: a foreign value here fails every snapshot consumer
/// whose read covers the list closed until an operator removes the key — on
/// an object carrying the `chs_complete` marker `read_object_snapshot` reads
/// only the lists `obj:chs` names, so its consumers miss a list it does not.
pub(crate) const SNAPSHOT_KEY_FAMILY_CHILDREN: &str = "children";
/// `le:{object_id}:obj:chs` — the children-list parent-id index (`SCARD` /
/// `SMEMBERS`), which `read_object_snapshot` (atomic and unfenced) trusts on a
/// marked object. Never folded
/// into [`SNAPSHOT_KEY_FAMILY_CHILDREN`], which stays the lists themselves.
pub(crate) const SNAPSHOT_KEY_FAMILY_CHILDREN_INDEX: &str = "children_index";
/// `le:{object_id}:obj:flds` — the field-member index (`ZRANGE`).
pub(crate) const SNAPSHOT_KEY_FAMILY_FIELDS_INDEX: &str = "fields_index";
/// `le:{object_id}:obj:tss` — the time-sample-member index (`ZRANGE`).
pub(crate) const SNAPSHOT_KEY_FAMILY_TIME_SAMPLES_INDEX: &str = "time_samples_index";
/// `le:{object_id}:obj:fld:{section_id}:{hex(key_name)}` — one field's hash
/// (`HGETALL`).
pub(crate) const SNAPSHOT_KEY_FAMILY_FIELD: &str = "field";
/// `le:{object_id}:obj:ts:{section_id}:{hex_f64(time)}` — one time sample's
/// hash (`HGETALL`).
pub(crate) const SNAPSHOT_KEY_FAMILY_TIME_SAMPLE: &str = "time_sample";
/// `le:{object_id}:obj:ext` — the object's external-value reference index
/// (`HLEN`, `HSCAN`). No snapshot read touches it: external-value clean-up's
/// size read and reference walk do (`storage::retirement`). A foreign value
/// here stops none of the four snapshot reads; the clean-up skips its release
/// and removes the key, its references freed later by the reference-count
/// reconciler. Edits refuse it on their own
/// (`live_edit_keyspace_type_error_total`).
pub(crate) const SNAPSHOT_KEY_FAMILY_EXT_INDEX: &str = "ext_index";

/// The complete, bounded `key_family` vocabulary of
/// `live_edit_snapshot_read_type_error_total` (b.atj).
///
/// [`Metrics::new`] pre-registers every entry at zero so an
/// `increase(...) > 0` alert has a series to evaluate before the first refusal
/// rather than firing on absent-metric semantics.
///
/// Adding a snapshot read of a new key family (to the bounded fan-out in
/// `storage::diff_read` or to `object_snapshot_read.lua`) means adding a
/// `storage::diff_read::SnapshotKeyFamily` variant — and **nothing in the build
/// drags that variant as far as this array on its own.** The `_`-less `label`
/// match forces a label to exist, no more. The `const` length assertion beside
/// `SnapshotKeyFamily::ALL` does name this array when it fires, but it only
/// fires once the variant is listed in `ALL`; and the set-equality unit test
/// `snapshot_key_family_all_matches_the_registered_label_set` in
/// `storage::diff_read` iterates `ALL`, so it sees only listed variants too. A
/// variant that is labelled and called but listed in neither place compiles
/// clean and emits an unregistered `key_family` — an un-seeded cell, exactly
/// what the pre-registration above exists to prevent.
///
/// So the rule is a review-time one: when you add a variant, add it to `ALL` and
/// to this array in the same edit. From there on the assertion (cardinality, at
/// compile time) and the unit test (element-for-element equality) hold the two
/// in step.
pub const SNAPSHOT_READ_KEY_FAMILIES: &[&str] = &[
    SNAPSHOT_KEY_FAMILY_HEADER,
    SNAPSHOT_KEY_FAMILY_SECTIONS_INDEX,
    SNAPSHOT_KEY_FAMILY_DELETED_SECTIONS_INDEX,
    SNAPSHOT_KEY_FAMILY_SECTION,
    SNAPSHOT_KEY_FAMILY_DELETED_SECTION,
    SNAPSHOT_KEY_FAMILY_CHILDREN,
    SNAPSHOT_KEY_FAMILY_CHILDREN_INDEX,
    SNAPSHOT_KEY_FAMILY_FIELDS_INDEX,
    SNAPSHOT_KEY_FAMILY_TIME_SAMPLES_INDEX,
    SNAPSHOT_KEY_FAMILY_FIELD,
    SNAPSHOT_KEY_FAMILY_TIME_SAMPLE,
    SNAPSHOT_KEY_FAMILY_EXT_INDEX,
];

// ── snapshot_read_refused_total `kind` vocabulary (b.muu) ───────────────────
//
// The sibling vocabulary to `SNAPSHOT_READ_KEY_FAMILIES` above, for the counter
// that answers the *other* half of a refused object-state read: a permanent
// failure that is NOT a wrong-typed key. Lives here for the reason
// [`UNWATCH_SITE_EXPLICIT`] gives — the label vocabulary belongs to the counter
// it labels, and `observability` must gain no dependency edge into `storage`
// (CLAUDE.md § Module Dependency Rules). `storage::diff_read`'s
// `snapshot_refusal_kind_label` is the sole producer; it maps a
// `fred::error::ErrorKind` onto one of these constants and never mints a string.
//
// **These are this crate's own snake_case names, deliberately NOT
// `fred::error::ErrorKind::to_str()`.** That method yields `"Invalid Argument"`
// / `"IO Error"` (`fred-10.1.0/src/error.rs:77-99`) — spaces, capitals, and a
// third-party string a fred release can rename out from under an operator's
// dashboard and alert rules. `valkey_errors_total{kind}` does use `to_str()`
// for the connection events fred reports, and those cells are consequently
// unseedable and unstable across fred bumps; that is the shape this vocabulary
// exists to avoid, not one to copy. (That counter's one service-owned value,
// `VALKEY_ERROR_KIND_COMMAND_TIMEOUT`, is seeded.)
//
// Bounded cardinality is a hard property here, not a convention: the producer's
// match keeps a `_ => SNAPSHOT_READ_REFUSAL_KIND_UNKNOWN` arm precisely because
// `fred::error::ErrorKind` is a third-party enum this crate cannot make
// exhaustive — see that function for why the `_` arm is the guard rather than
// the hole it looks like next to `SnapshotKeyFamily::label`'s `_`-less match.

/// A malformed or unexpected frame from the server — fred `ErrorKind::Protocol`.
/// A service or client-library bug; not a keyspace fault.
pub(crate) const SNAPSHOT_READ_REFUSAL_KIND_PROTOCOL: &str = "protocol";
/// A response this service could not deserialize into the expected type — fred
/// `ErrorKind::Parse`. Same remediation class as `protocol`.
pub(crate) const SNAPSHOT_READ_REFUSAL_KIND_PARSE: &str = "parse";
/// The server rejected the connection's credentials — fred `ErrorKind::Auth`,
/// which fred mints from a `-NOAUTH` / `-WRONGPASS` reply
/// (`fred-10.1.0/src/protocol/utils.rs:112`). A credential / deployment fault:
/// check `LIVE_EDIT_VALKEY_AUTH_MODE` and the token sidecar.
pub(crate) const SNAPSHOT_READ_REFUSAL_KIND_AUTH: &str = "auth";
/// A fatal client-configuration error — fred `ErrorKind::Config` (which fred
/// also mints for a cluster client pointed at a non-cluster server). A
/// deployment fault.
pub(crate) const SNAPSHOT_READ_REFUSAL_KIND_CONFIG: &str = "config";
/// A TLS failure — fred `ErrorKind::Tls`.
///
/// The cell is registered unconditionally while the producer's match arm is
/// `#[cfg(feature = "valkey-tls")]`: the fred variant only exists when fred is
/// built with a TLS backend (`fred-10.1.0/src/error.rs:38-51`, reached here
/// through `Cargo.toml`'s `valkey-tls = ["fred/enable-rustls-ring"]`), so a bare
/// match arm fails the no-default-features lint shape. Gating the arm and not
/// the cell keeps the published series set identical across build shapes — an
/// operator's dashboard must not change because of how the binary was compiled.
pub(crate) const SNAPSHOT_READ_REFUSAL_KIND_TLS: &str = "tls";
/// fred `ErrorKind::InvalidArgument` that is **not** a `WRONGTYPE` reply.
///
/// The `WRONGTYPE` case never reaches this counter: it is claimed by the first
/// arm of `storage::diff_read::count_object_key_refusal` and ticks
/// [`SNAPSHOT_READ_KEY_FAMILIES`]' counter instead, which is what keeps that
/// counter's zero baseline meaning exactly "no foreign value on the `le:`
/// keyspace".
pub(crate) const SNAPSHOT_READ_REFUSAL_KIND_INVALID_ARGUMENT: &str = "invalid_argument";
/// fred `ErrorKind::InvalidCommand` — a command issued in a connection state
/// that forbids it. A service bug.
pub(crate) const SNAPSHOT_READ_REFUSAL_KIND_INVALID_COMMAND: &str = "invalid_command";
/// fred `ErrorKind::Routing` outside the two transient carve-outs
/// (`storage::client::is_router_no_connection_error`,
/// `is_router_retry_budget_exhausted_error`) — those are classified transient
/// and are not counted at all.
pub(crate) const SNAPSHOT_READ_REFUSAL_KIND_ROUTING: &str = "routing";
/// fred `ErrorKind::NotFound` — a `nil` cast into a non-nullable type. A
/// service bug in the read's typing, not a missing key (a missing key reads as
/// an empty collection on every read this counter covers).
pub(crate) const SNAPSHOT_READ_REFUSAL_KIND_NOT_FOUND: &str = "not_found";
/// The backing store answered, but refused a write — a `-MISCONF`, `-READONLY`
/// or `-NOREPLICAS` reply; and, as a regression tripwire only, `-LOADING` or
/// `-BUSY`.
///
/// **Why the cell exists.** fred's `pretty_error` maps every unrecognised
/// leading RESP token onto `ErrorKind::Unknown` with the whole reply preserved
/// in `details` (`fred-10.1.0/src/protocol/utils.rs:96-129`, `_ =>
/// ErrorKind::Unknown` at `:119`), so these replies would otherwise sit in
/// `unknown` and cost the other eleven cells their `increase(...) > 0` alert.
/// Split out, eleven cells keep a true zero baseline and this one takes a
/// sustained-rate rule instead.
///
/// **The live members are the write-gated three.** `-MISCONF`
/// (`stop-writes-on-bgsave-error`), `-READONLY` ("You can't write against a
/// read only replica") and `-NOREPLICAS` (`min-replicas-to-write`) gate
/// **write** commands only, `storage::client::is_transient_fred_error` calls
/// none of them transient, and this counter's touch set is all reads but one —
/// the orphan sweep's retirement fence `HSET obj:hdr superseded 1`
/// (`storage::orphan_sweep::cleanup_orphaned_object`). The cell is therefore
/// reachable from that single write and nowhere else: at most one tick per
/// orphan per 300 s sweep pass, never a request-rate burst. The WARN line's
/// verbatim reply says which of the three.
///
/// **`-LOADING` and `-BUSY` do not move it on a healthy restart.** They refuse
/// every command, reads included, but are transient
/// (`storage::client::is_store_not_serving_reply`): the read is retried and
/// nothing is counted. They stay matched so that a regression making them
/// permanent again lands here, where the sustained-rate alert sees it, rather
/// than in `unknown`; a `-LOADING` or `-BUSY` in this cell's WARN line means
/// that classification has regressed.
///
/// **`-MISCONF` can need operator action.** Under
/// `stop-writes-on-bgsave-error yes` the refusal lasts until a background save
/// succeeds, which on a full disk means until an operator frees space — the
/// same argument that split `-OOM` out below. It stays in this cell because it
/// also arises transiently (the next BGSAVE simply succeeds), and resolving
/// that ambiguity is what the 15 m sustained-rate alert is for: a burst that
/// clears was the transient case, a rate that outlives the window is the one
/// wanting an operator.
///
/// **`-OOM` is deliberately NOT in this family**, and
/// [`SNAPSHOT_READ_REFUSAL_KIND_BACKEND_OUT_OF_MEMORY`] is its cell. What every
/// member above has is a *path back that can close without an operator* — the
/// next background save succeeds, the replica is promoted, the replicas come
/// back (and for the tripwire pair, the replay finishes, the script ends) —
/// which is what licenses the WARN level and the sustained-rate rule. Under
/// the `maxmemory-policy: noeviction` this project mandates (CLAUDE.md
/// § Deployment), `-OOM` has no such path for data in use: nothing is
/// evicted, the only memory the service frees by itself is what released
/// external values hold (the GC's out-of-memory collection, b.pa1), and a
/// store full of data in use refuses every `denyoom` command until an
/// operator raises `maxmemory` or removes data. Even a store that clears was
/// sized too small. Folding it in here would make this cell's contract false
/// for one member and route an operator-action condition to the no-page arm.
/// Full derivation on `storage::diff_read::is_backend_unavailable_reply`.
///
/// A request's write refused the same way is not counted here — a different
/// touch set: the upload write paths answer `-MISCONF`, `-READONLY` and
/// `-NOREPLICAS` (and `-OOM`) `UNAVAILABLE` with nothing applied and count them
/// on `live_edit_store_write_refused_total` ([`STORE_WRITE_REFUSED_REASONS`]),
/// and every other write answers them a plain `UNAVAILABLE`, uncounted
/// (`storage::client::store_write_refusal`).
pub(crate) const SNAPSHOT_READ_REFUSAL_KIND_BACKEND_UNAVAILABLE: &str = "backend_unavailable";
/// The backing store answered, but is full — an `-OOM` reply.
///
/// **What produces it.** Valkey refuses any `denyoom` command with
/// `-OOM command not allowed when used memory > 'maxmemory'.` once
/// `used_memory` exceeds `maxmemory`. Reads are never `denyoom`, so the only
/// emitter reachable on this counter's touch set is the orphan sweep's
/// retirement fence `HSET obj:hdr superseded 1`
/// (`storage::orphan_sweep::cleanup_orphaned_object`) — the one write the
/// counter covers. Full derivation, including the second Valkey origin and why
/// it is unreachable here, on
/// `storage::diff_read::is_backend_out_of_memory_reply`.
///
/// **It is not transient, which is why it is not folded into
/// [`SNAPSHOT_READ_REFUSAL_KIND_BACKEND_UNAVAILABLE`].** This project mandates
/// `maxmemory-policy: noeviction` (CLAUDE.md § Deployment), so the store evicts
/// nothing. The refusal does tell the GC (b.pa1, `storage::store_oom`), whose
/// out-of-memory collection deletes released external values at once, so a
/// store filled by those clears within seconds; a store full of data in use
/// stays refused until an operator acts. Either way the store was sized too
/// small for what it holds, so remediation is operator work, not waiting:
/// raise `maxmemory`, or free memory by removing data, and size the instance
/// against the noeviction sizing model in
/// `docs/service/docs/backing-store.md` § Sizing.
///
/// It is consequently one of the eleven should-never-happen cells, not a
/// sibling of `backend_unavailable`: it logs at **ERROR** and is on the
/// `increase(...) > 0` page, while
/// `backend_unavailable` alone stays at WARN and takes a sustained-rate rule.
///
/// An upload refused the same way is not counted here: the upload write
/// paths count it on `live_edit_store_write_refused_total{reason=
/// "out_of_memory"}` ([`STORE_WRITE_REFUSED_REASON_OUT_OF_MEMORY`]) and answer
/// `UNAVAILABLE`.
pub(crate) const SNAPSHOT_READ_REFUSAL_KIND_BACKEND_OUT_OF_MEMORY: &str = "backend_out_of_memory";
/// Every other fred `ErrorKind::Unknown`, plus `Url`, `Sentinel`, and any kind
/// a future fred adds that this crate has not mapped.
///
/// The catch-all that makes the vocabulary bounded — see
/// `storage::diff_read::snapshot_refusal_kind_label`.
pub(crate) const SNAPSHOT_READ_REFUSAL_KIND_UNKNOWN: &str = "unknown";

/// The complete, bounded `kind` vocabulary of
/// `live_edit_snapshot_read_refused_total` (b.muu).
///
/// [`Metrics::new`] pre-registers every entry at zero — unconditionally,
/// including `tls` under a build that cannot produce it — so an
/// `increase(...) > 0` alert has a series to evaluate from t=0 rather than
/// firing on absent-metric semantics, and so the published series set does not
/// depend on the cargo features the binary was built with.
///
/// Unlike [`SNAPSHOT_READ_KEY_FAMILIES`] this array has **no** enum to hold it
/// in step and needs none: its producer,
/// `storage::diff_read::snapshot_refusal_kind_label`, matches a third-party
/// enum behind a `_` arm that lands on
/// [`SNAPSHOT_READ_REFUSAL_KIND_UNKNOWN`], so no unregistered label can be
/// emitted even by a fred release that adds a kind. What review must still
/// check is the reverse direction — a constant added above and used in a new
/// match arm but never listed here would be an un-seeded cell.
/// `storage::diff_read`'s
/// `snapshot_refusal_kind_label_emits_exactly_the_registered_kind_set` is what
/// checks it: it drives the producer with a hand-built `fred::error::Error` per
/// mapped kind and asserts the emitted set equals this array. Read what that
/// buys honestly, because the two directions are not equally strong. An
/// **unreachable seeded cell** — a label listed here that no input can produce —
/// fails it outright, with no maintenance required. An **unlisted constant** —
/// one added above and used in a new match arm but never added here — is caught
/// only if the test's own case list gains a case driving that arm; the list is
/// hand-written, so a new arm with no new case is simply never exercised and the
/// un-seeded cell slips through. The convention that keeps it honest is stated
/// on that case list and is the thing to preserve when adding a kind: **one case
/// per arm of the producer's match**. (The
/// `snapshot_read_refusal_kind_consts_are_all_registered` test next to it
/// catches the same drift a step earlier and names the constant, but enumerates
/// the constants by hand too, so it carries the same obligation.)
pub const SNAPSHOT_READ_REFUSAL_KINDS: &[&str] = &[
    SNAPSHOT_READ_REFUSAL_KIND_PROTOCOL,
    SNAPSHOT_READ_REFUSAL_KIND_PARSE,
    SNAPSHOT_READ_REFUSAL_KIND_AUTH,
    SNAPSHOT_READ_REFUSAL_KIND_CONFIG,
    SNAPSHOT_READ_REFUSAL_KIND_TLS,
    SNAPSHOT_READ_REFUSAL_KIND_INVALID_ARGUMENT,
    SNAPSHOT_READ_REFUSAL_KIND_INVALID_COMMAND,
    SNAPSHOT_READ_REFUSAL_KIND_ROUTING,
    SNAPSHOT_READ_REFUSAL_KIND_NOT_FOUND,
    SNAPSHOT_READ_REFUSAL_KIND_BACKEND_UNAVAILABLE,
    SNAPSHOT_READ_REFUSAL_KIND_BACKEND_OUT_OF_MEMORY,
    SNAPSHOT_READ_REFUSAL_KIND_UNKNOWN,
];

// ── store_write_refused_total `stage` / `reason` vocabularies ───────────────
//
// The labels of `live_edit_store_write_refused_total`, owned here beside the
// counter for the reason [`UNWATCH_SITE_EXPLICIT`] gives: `observability` must
// gain no dependency edge into `storage`. `storage::client`'s
// `StoreWriteRefusal::reason_label` is the sole producer of a `reason`, and
// the write sites pass one of the `stage` constants; neither mints a string.

/// `stage`: the external-value pre-store every upload runs before its commit
/// (`ValkeyExternalValueStore::store_values`; also the DR restore's pre-store
/// and the read path's lazy rewrite, which share it).
pub(crate) const STORE_WRITE_REFUSED_STAGE_EXT_STORE: &str = "ext_store";
/// `stage`: an `UpdateLayer` commit — the commit script in either mode, or the
/// WATCH path's discarded transaction.
pub(crate) const STORE_WRITE_REFUSED_STAGE_UPDATE_COMMIT: &str = "update_commit";
/// `stage`: a `CreateLayer` write that applied nothing — its object-id
/// allocation, its materialization, the overwrite's `superseded` mark, or a
/// new layer's commit point.
pub(crate) const STORE_WRITE_REFUSED_STAGE_CREATE_COMMIT: &str = "create_commit";

/// The complete `stage` vocabulary of `live_edit_store_write_refused_total`.
pub const STORE_WRITE_REFUSED_STAGES: &[&str] = &[
    STORE_WRITE_REFUSED_STAGE_EXT_STORE,
    STORE_WRITE_REFUSED_STAGE_UPDATE_COMMIT,
    STORE_WRITE_REFUSED_STAGE_CREATE_COMMIT,
];

/// `reason`: `-OOM` — `used_memory` is above `maxmemory`. The required
/// `noeviction` policy evicts nothing. The refusal starts the GC's
/// out-of-memory collection (b.pa1; `live_edit_gc_oom_collections_total`),
/// which frees what released external values hold; a store full of data in
/// use stays refused until an operator raises `maxmemory` or frees memory.
pub(crate) const STORE_WRITE_REFUSED_REASON_OUT_OF_MEMORY: &str = "out_of_memory";
/// `reason`: `-MISCONF` — the store cannot persist (a failing RDB save under
/// `stop-writes-on-bgsave-error`, or a failing AOF write). Lasts until a save
/// succeeds; on a full disk, until an operator frees space.
pub(crate) const STORE_WRITE_REFUSED_REASON_MISCONF: &str = "misconf";
/// `reason`: `-READONLY` — the write reached a read-only replica (a failover in
/// progress, or a mis-routed connection).
pub(crate) const STORE_WRITE_REFUSED_REASON_READONLY: &str = "readonly";
/// `reason`: `-NOREPLICAS` — too few replicas are reachable for
/// `min-replicas-to-write`.
pub(crate) const STORE_WRITE_REFUSED_REASON_NOREPLICAS: &str = "noreplicas";

/// The complete `reason` vocabulary of `live_edit_store_write_refused_total`.
pub const STORE_WRITE_REFUSED_REASONS: &[&str] = &[
    STORE_WRITE_REFUSED_REASON_OUT_OF_MEMORY,
    STORE_WRITE_REFUSED_REASON_MISCONF,
    STORE_WRITE_REFUSED_REASON_READONLY,
    STORE_WRITE_REFUSED_REASON_NOREPLICAS,
];

// ── live_edit_snapshot_read_* vocabularies (Spec SDD SR-8.1, SR-8.2) ─────────
//
// The labels of the per-read series `storage::diff_read::read_object_snapshot`
// records. Closed vocabularies, owned here beside the series they label for the
// reason [`UNWATCH_SITE_EXPLICIT`] gives; `storage::diff_read`'s
// `SnapshotReadConsumer` / `SlimSnapshotReadConsumer` / `SnapshotReadPath`
// enums return these constants and never mint a label of their own.

/// `consumer` label: a `SubscribeReadLayer` catch-up's snapshot read.
pub(crate) const SNAPSHOT_READ_CONSUMER_CATCHUP: &str = "catchup";
/// `consumer` label: a legacy (rollback-mode) in-window replay's per-commit
/// read. An empty commit takes none, and the steady state never reads.
pub(crate) const SNAPSHOT_READ_CONSUMER_CATCHUP_LEGACY_PER_COMMIT: &str =
    "catchup_legacy_per_commit";
/// `consumer` label: the checkpoint assembler's read.
pub(crate) const SNAPSHOT_READ_CONSUMER_CHECKPOINT: &str = "checkpoint";
/// `consumer` label: the structural `UpdateLayer` slow path's read.
pub(crate) const SNAPSHOT_READ_CONSUMER_STRUCTURAL_UPDATE: &str = "structural_update";
/// `consumer` label: external-value clean-up's reference walk of a retiring
/// object (`storage::retirement`, behind `schedule_old_object_ext_gc`), one
/// observation per walk: an unfenced pass that pages the object's value
/// indices and `obj:ext` and reads each value row's external hash, never its
/// whole state.
pub(crate) const SNAPSHOT_READ_CONSUMER_EXT_GC: &str = "ext_gc";
/// `consumer` label: the structural write-planning cache's cold-start read,
/// the plain atomic read or the fenced read
/// (`storage::structural_cache_cold_start`).
pub(crate) const SNAPSHOT_READ_CONSUMER_STRUCTURAL_CACHE: &str = "structural_cache";
/// `consumer` label: the structural write-planning cache's sampled audit read,
/// the plain atomic read or the fenced read
/// (`storage::structural_cache_audit`). Its own label, so
/// `consumer="structural_cache"` counts cold starts only.
pub(crate) const SNAPSHOT_READ_CONSUMER_STRUCTURAL_CACHE_AUDIT: &str = "structural_cache_audit";

/// The complete `consumer` vocabulary of the `live_edit_snapshot_read_*`
/// series.
pub const SNAPSHOT_READ_CONSUMERS: &[&str] = &[
    SNAPSHOT_READ_CONSUMER_CATCHUP,
    SNAPSHOT_READ_CONSUMER_CATCHUP_LEGACY_PER_COMMIT,
    SNAPSHOT_READ_CONSUMER_CHECKPOINT,
    SNAPSHOT_READ_CONSUMER_STRUCTURAL_UPDATE,
    SNAPSHOT_READ_CONSUMER_EXT_GC,
    SNAPSHOT_READ_CONSUMER_STRUCTURAL_CACHE,
    SNAPSHOT_READ_CONSUMER_STRUCTURAL_CACHE_AUDIT,
];

/// `path` label: one `EVALSHA` of `object_snapshot_read.lua`.
pub(crate) const SNAPSHOT_READ_PATH_ATOMIC: &str = "atomic";
/// `path` label: the fan-out read under a per-object snapshot fence.
pub(crate) const SNAPSHOT_READ_PATH_FENCED: &str = "fenced";
/// `path` label: the bounded fan-out read with no fence.
pub(crate) const SNAPSHOT_READ_PATH_UNFENCED: &str = "unfenced";

/// The complete `path` vocabulary of the `live_edit_snapshot_read_*` series.
pub const SNAPSHOT_READ_PATHS: &[&str] = &[
    SNAPSHOT_READ_PATH_ATOMIC,
    SNAPSHOT_READ_PATH_FENCED,
    SNAPSHOT_READ_PATH_UNFENCED,
];

/// `outcome` label: the read returned the object's state.
pub(crate) const SNAPSHOT_READ_OUTCOME_OK: &str = "ok";
/// `outcome` label: the object has no header, or (for every consumer but
/// `ext_gc`) a header without a numeric `seq`.
pub(crate) const SNAPSHOT_READ_OUTCOME_NOT_FOUND: &str = "not_found";
/// `outcome` label: an atomic attempt answered `TOO_LARGE`, early-out or
/// mid-read, so the layer takes its consumer's above-ceiling path. That
/// path's own read is counted under its own `path`, not here.
pub(crate) const SNAPSHOT_READ_OUTCOME_TOO_LARGE: &str = "too_large";
/// `outcome` label: a failure classified transient (`ServiceError::Aborted`,
/// retryable).
pub(crate) const SNAPSHOT_READ_OUTCOME_TRANSIENT_ERROR: &str = "transient_error";
/// `outcome` label: a failure classified permanent (`ServiceError::Internal`).
pub(crate) const SNAPSHOT_READ_OUTCOME_PERMANENT_ERROR: &str = "permanent_error";
/// `outcome` label: a fenced read attempt its release did not prove — its
/// lease was lost, or a restore or cleanup changed the object during the read
/// — so it was retried or given up, never served.
pub(crate) const SNAPSHOT_READ_OUTCOME_FENCE_BROKEN: &str = "fence_broken";

/// The complete `outcome` vocabulary of `live_edit_snapshot_read_total`.
pub const SNAPSHOT_READ_OUTCOMES: &[&str] = &[
    SNAPSHOT_READ_OUTCOME_OK,
    SNAPSHOT_READ_OUTCOME_NOT_FOUND,
    SNAPSHOT_READ_OUTCOME_TOO_LARGE,
    SNAPSHOT_READ_OUTCOME_TRANSIENT_ERROR,
    SNAPSHOT_READ_OUTCOME_PERMANENT_ERROR,
    SNAPSHOT_READ_OUTCOME_FENCE_BROKEN,
];

/// The consumers that take the atomic path (`path="atomic"`). The
/// `structural_cache` and `structural_cache_audit` consumers never fall back
/// to the unfenced fan-out: above their ceilings `too_large` is their own
/// outcome, after which both take the fenced read (counted under
/// `path="fenced"`; the audit only with its above-ceiling switch on). The
/// `structural_update` and ext-GC consumers never take it: the first reads
/// through the unfenced fan-out at every size, the second is the reference
/// walk, an unfenced pass of its own ([`SNAPSHOT_READ_UNFENCED_CONSUMERS`]).
pub const SNAPSHOT_READ_ATOMIC_CONSUMERS: &[&str] = &[
    SNAPSHOT_READ_CONSUMER_CATCHUP,
    SNAPSHOT_READ_CONSUMER_CATCHUP_LEGACY_PER_COMMIT,
    SNAPSHOT_READ_CONSUMER_CHECKPOINT,
    SNAPSHOT_READ_CONSUMER_STRUCTURAL_CACHE,
    SNAPSHOT_READ_CONSUMER_STRUCTURAL_CACHE_AUDIT,
];

/// The outcomes an atomic read can record: every one but `fence_broken`,
/// which only a fenced read can.
pub const SNAPSHOT_READ_ATOMIC_OUTCOMES: &[&str] = &[
    SNAPSHOT_READ_OUTCOME_OK,
    SNAPSHOT_READ_OUTCOME_NOT_FOUND,
    SNAPSHOT_READ_OUTCOME_TOO_LARGE,
    SNAPSHOT_READ_OUTCOME_TRANSIENT_ERROR,
    SNAPSHOT_READ_OUTCOME_PERMANENT_ERROR,
];

/// The consumers that record `path="unfenced"`: the structural `UpdateLayer`
/// WATCH path (`structural_update`), through the unfenced bounded fan-out, and
/// external-value clean-up's reference walk (`ext_gc`), an unfenced pass of its
/// own, each at every size, and no others — not `structural_cache` or
/// `structural_cache_audit`.
pub const SNAPSHOT_READ_UNFENCED_CONSUMERS: &[&str] = &[
    SNAPSHOT_READ_CONSUMER_STRUCTURAL_UPDATE,
    SNAPSHOT_READ_CONSUMER_EXT_GC,
];

/// The outcomes an unfenced read can record. Not `fence_broken`: it holds no
/// fence. `too_large` for `structural_update` only: the fan-out has no
/// ceiling and reads every size, but a structural `UpdateLayer`'s read stops
/// before its next window when its upload charge cannot grow to cover it
/// (the WATCH path's allowance, SDD §5.4), and the attempt waits for that
/// charge with its checkout returned; the ext-GC reference walk carries no
/// allowance, so its `too_large` cell is not registered. `not_found` covers an absent
/// header for both consumers, and for `structural_update` also a header
/// without a numeric `seq`.
pub const SNAPSHOT_READ_UNFENCED_OUTCOMES: &[&str] = &[
    SNAPSHOT_READ_OUTCOME_OK,
    SNAPSHOT_READ_OUTCOME_NOT_FOUND,
    SNAPSHOT_READ_OUTCOME_TOO_LARGE,
    SNAPSHOT_READ_OUTCOME_TRANSIENT_ERROR,
    SNAPSHOT_READ_OUTCOME_PERMANENT_ERROR,
];

/// The consumers whose reads run the fenced read (`path="fenced"`) in this
/// binary, each for a layer above the atomic-read ceilings: a
/// `SubscribeReadLayer` catch-up, through its shared fenced-read flight
/// (`storage::snapshot_flights`); the checkpoint assembler, through a fenced
/// read of its own (`checkpoint::assembler`, Spec SDD SR-11.5); and the
/// structural cache's cold start, through its own hook
/// (`storage::structural_cache_fenced`), for a layer above the plain ceilings
/// or predicted over the plain byte ceiling; and the structural cache's
/// sampled audit, through the same hook, for a layer above the plain ceilings
/// with the above-ceiling switch on (`storage::structural_cache_audit`). A
/// rollback-mode replay's
/// per-commit read is never fenced: on `TOO_LARGE` the replay hands its
/// catch-up to a flight, whose read counts under `catchup`.
pub const SNAPSHOT_READ_FENCED_CONSUMERS: &[&str] = &[
    SNAPSHOT_READ_CONSUMER_CATCHUP,
    SNAPSHOT_READ_CONSUMER_CHECKPOINT,
    SNAPSHOT_READ_CONSUMER_STRUCTURAL_CACHE,
    SNAPSHOT_READ_CONSUMER_STRUCTURAL_CACHE_AUDIT,
];

/// The outcomes a fenced read attempt can record: every one but `too_large`,
/// since the fenced read has no ceiling and reads every size.
pub const SNAPSHOT_READ_FENCED_OUTCOMES: &[&str] = &[
    SNAPSHOT_READ_OUTCOME_OK,
    SNAPSHOT_READ_OUTCOME_NOT_FOUND,
    SNAPSHOT_READ_OUTCOME_TRANSIENT_ERROR,
    SNAPSHOT_READ_OUTCOME_PERMANENT_ERROR,
    SNAPSHOT_READ_OUTCOME_FENCE_BROKEN,
];

// ── live_edit_snapshot_fence_acquire_total `result` vocabulary (SR-8.3) ──────
//
// One value per branch of `snapshot_fence_acquire.lua`'s reply that a reader
// can be given: the two grant kinds and the three retry reasons. The script
// names its branch in the reply, and `storage::snapshot_fence`'s `GrantKind` /
// `RetryReason` map onto these constants; `NOT_FOUND` and a failed call are
// not acquire results and are counted on `live_edit_snapshot_read_total`.

/// `result` label: the reader started a new fence epoch.
pub(crate) const SNAPSHOT_FENCE_ACQUIRE_STARTED_EPOCH: &str = "started_epoch";
/// `result` label: the reader joined a live epoch inside its join window.
pub(crate) const SNAPSHOT_FENCE_ACQUIRE_JOINED_EPOCH: &str = "joined_epoch";
/// `result` label: told to wait, because a live epoch's join window had
/// passed.
pub(crate) const SNAPSHOT_FENCE_ACQUIRE_RETRY_AFTER_EPOCH: &str = "retry_after_epoch";
/// `result` label: told to wait out the writers' gap after an epoch.
pub(crate) const SNAPSHOT_FENCE_ACQUIRE_RETRY_AFTER_GAP: &str = "retry_after_gap";
/// `result` label: told to wait for a WATCH-path writer's live intent.
pub(crate) const SNAPSHOT_FENCE_ACQUIRE_RETRY_AFTER_INTENT: &str = "retry_after_intent";

/// The complete `result` vocabulary of
/// `live_edit_snapshot_fence_acquire_total`.
pub const SNAPSHOT_FENCE_ACQUIRE_RESULTS: &[&str] = &[
    SNAPSHOT_FENCE_ACQUIRE_STARTED_EPOCH,
    SNAPSHOT_FENCE_ACQUIRE_JOINED_EPOCH,
    SNAPSHOT_FENCE_ACQUIRE_RETRY_AFTER_EPOCH,
    SNAPSHOT_FENCE_ACQUIRE_RETRY_AFTER_GAP,
    SNAPSHOT_FENCE_ACQUIRE_RETRY_AFTER_INTENT,
];

// ── live_edit_structural_cold_start_* vocabulary (structural-edits SDD SR-2.5) ─
//
// The `route` label of the structural write-planning cache's cold-start series,
// recorded by `storage::structural_cache_cold_start::cold_start`: the route
// that ended one cold start.

/// `route` label: the plain atomic read, at the shared atomic-read ceilings,
/// produced the entry.
pub(crate) const STRUCTURAL_COLD_START_ROUTE_ATOMIC: &str = "atomic";
/// `route` label: b.y3m's fenced read produced the entry — a layer above the
/// plain ceilings, or predicted over the plain byte ceiling, under the
/// cache's one-at-a-time and plain permits, a fenced-class catch-up read
/// permit and a catch-up memory grant.
pub(crate) const STRUCTURAL_COLD_START_ROUTE_FENCED: &str = "fenced";
/// `route` label: the cold start found the object uncacheable — over the
/// per-entry weight cap, or without `chs_complete` — and recorded its
/// negative entry. A disabled cache never ticks it, and neither does a failed
/// fenced read (`fenced_read_failed`), which is one call's verdict.
pub(crate) const STRUCTURAL_COLD_START_ROUTE_UNCACHEABLE: &str = "uncacheable";

/// The complete `route` vocabulary of
/// `live_edit_structural_cold_start_reads_total` and
/// `live_edit_structural_cold_start_read_duration_seconds`.
pub const STRUCTURAL_COLD_START_ROUTES: &[&str] = &[
    STRUCTURAL_COLD_START_ROUTE_ATOMIC,
    STRUCTURAL_COLD_START_ROUTE_FENCED,
    STRUCTURAL_COLD_START_ROUTE_UNCACHEABLE,
];

// ── live_edit_structural_cache_* vocabulary (structural-edits SDD SR-2.4) ────
//
// Recorded by the structural cache's read side
// (`storage::structural_cache_read`): how a structural call got its entry,
// and why an entry was dropped and read again.

/// `outcome` label: the lookup found an entry and its repair replayed no
/// stream entry (it was already current).
pub(crate) const STRUCTURAL_CACHE_LOOKUP_HIT: &str = "hit";
/// `outcome` label: the lookup found an entry and its repair replayed one or
/// more stream entries to bring it current.
pub(crate) const STRUCTURAL_CACHE_LOOKUP_REPAIRED: &str = "repaired";
/// `outcome` label: the lookup found no entry (first use on this replica, or
/// after eviction, expiry or retirement) and the call cold-started the object.
/// Not a reload.
pub(crate) const STRUCTURAL_CACHE_LOOKUP_COLD: &str = "cold";
/// `outcome` label: the lookup found an entry, its repair found an anomaly,
/// and the call dropped it and read the object again; the reason is on
/// `live_edit_structural_cache_reloads_total`.
pub(crate) const STRUCTURAL_CACHE_LOOKUP_RELOAD: &str = "reload";

/// The complete `outcome` vocabulary of
/// `live_edit_structural_cache_lookups_total`, zero-seeded.
pub const STRUCTURAL_CACHE_LOOKUP_OUTCOMES: &[&str] = &[
    STRUCTURAL_CACHE_LOOKUP_HIT,
    STRUCTURAL_CACHE_LOOKUP_REPAIRED,
    STRUCTURAL_CACHE_LOOKUP_COLD,
    STRUCTURAL_CACHE_LOOKUP_RELOAD,
];

/// `reason` label: the header `seq` equals the entry's but its
/// `commit_chain` differs — the object was re-materialized (a DR restore) or
/// rolled back to the same `seq`.
pub(crate) const STRUCTURAL_CACHE_RELOAD_IDENTITY: &str = "identity";
/// `reason` label: the header `seq` is below the entry's (or fell between two
/// repair reads), or the `commit_chain` folded over the replayed entries does
/// not reach the header's — a re-materialization or a backing-store rollback
/// at another `seq`, or a commit by a writer that does not fold.
pub(crate) const STRUCTURAL_CACHE_RELOAD_REGRESSED: &str = "regressed";
/// `reason` label: stream entries the entry needs are missing — trimmed by
/// the stream's `MAXLEN`, or never written (a DR restore advances `seq`
/// without them).
pub(crate) const STRUCTURAL_CACHE_RELOAD_GAP: &str = "gap";
/// `reason` label: a `deleted` entry sits in the range the repair read.
pub(crate) const STRUCTURAL_CACHE_RELOAD_TOMBSTONE: &str = "tombstone";
/// `reason` label: a stream entry's id, payload or OMPF could not be decoded.
pub(crate) const STRUCTURAL_CACHE_RELOAD_PARSE: &str = "parse";
/// `reason` label: a decoded stream entry did not replay onto the entry.
pub(crate) const STRUCTURAL_CACHE_RELOAD_REPLAY_ERR: &str = "replay_err";
/// `reason` label: more than `STRUCTURAL_REPAIR_MAX_TOTAL` (1,024) stream
/// entries separate the entry from the header. Only a structural call's
/// repair ticks it (the sampled audit's repair ticks no metric), and that
/// repair has no cap on its replayed bytes.
pub(crate) const STRUCTURAL_CACHE_RELOAD_TOO_FAR: &str = "too_far";
/// `reason` label: one pending stream entry alone is larger than the repair
/// reads' hard cap (`STRUCTURAL_REPAIR_MAX_BYTES`, 4 MiB). Only a structural
/// call's repair ticks it (the sampled audit's repair ticks no metric). An
/// entry larger only than a read's current bound (256 KiB at first) grows
/// the next read instead and reloads nothing.
pub(crate) const STRUCTURAL_CACHE_RELOAD_TOO_LARGE: &str = "too_large";
/// `reason` label: a replayed commit — or this replica's own structural
/// commit, whose successor is then dropped instead of swapped in — created a
/// section whose stored children list the entry deliberately did not hold (a
/// dangling parent).
pub(crate) const STRUCTURAL_CACHE_RELOAD_DANGLING_ADOPTED: &str = "dangling_adopted";
/// `reason` label: a structural commit refused with `CACHE_DIVERGED`. Ticked
/// by the structural commit's caller, not by the read side.
pub(crate) const STRUCTURAL_CACHE_RELOAD_DIVERGED: &str = "diverged";
/// `reason` label: the object's entry was dropped because the object became
/// uncacheable — a put-back of a structural call's entry (a commit's
/// successor, or a held entry a repair grew) refused by the per-entry weight
/// cap, which leaves the object an `OverWeight` negative entry. Ticked
/// wherever the structural route puts an entry back explicitly, in the
/// structural `UpdateLayer` attempt and in the read side
/// (`structural_cache_read::count_put_back`); a refusal by the epoch check
/// ticks nothing, nor does the implicit put-back of a dropped call.
pub(crate) const STRUCTURAL_CACHE_RELOAD_UNCACHEABLE: &str = "uncacheable";

/// The complete `reason` vocabulary of
/// `live_edit_structural_cache_reloads_total`, zero-seeded, in the spec's
/// order (SR-2.4).
pub const STRUCTURAL_CACHE_RELOAD_REASONS: &[&str] = &[
    STRUCTURAL_CACHE_RELOAD_IDENTITY,
    STRUCTURAL_CACHE_RELOAD_REGRESSED,
    STRUCTURAL_CACHE_RELOAD_GAP,
    STRUCTURAL_CACHE_RELOAD_TOMBSTONE,
    STRUCTURAL_CACHE_RELOAD_PARSE,
    STRUCTURAL_CACHE_RELOAD_REPLAY_ERR,
    STRUCTURAL_CACHE_RELOAD_TOO_FAR,
    STRUCTURAL_CACHE_RELOAD_TOO_LARGE,
    STRUCTURAL_CACHE_RELOAD_DANGLING_ADOPTED,
    STRUCTURAL_CACHE_RELOAD_DIVERGED,
    STRUCTURAL_CACHE_RELOAD_UNCACHEABLE,
];

// ── live_edit_structural_cache_uncacheable_total vocabulary ──────────────────
//
// Why the structural write-planning cache refused an object, one tick per
// verdict (`storage::structural_cache_cold_start::report_uncacheable`): the
// `storage::structural_cache::UncacheableReason` variants an uncacheable
// verdict can carry. `Disabled` has no value: a disabled cache (a zero
// budget, which only a cache built directly can have) is answered on every
// lookup, never a verdict, so no cell is seeded for it.

/// `reason` label: the object's estimated state weight (its plan indexes not
/// counted) is over the per-entry cap (a quarter of the cache budget) — at a
/// cold start's admission, or at an
/// explicit put-back of an entry that grew past the cap while cached (that
/// one also ticks `live_edit_structural_cache_reloads_total{reason="uncacheable"}`).
pub(crate) const STRUCTURAL_CACHE_UNCACHEABLE_OVER_WEIGHT: &str = "over_weight";
/// `reason` label: the object's header lacks `chs_complete`, so its dangling
/// children lists cannot be enumerated — found by the read side's probe or by
/// a cold start's read.
pub(crate) const STRUCTURAL_CACHE_UNCACHEABLE_NO_CHS_COMPLETE: &str = "no_chs_complete";
/// `reason` label: one structural call reloaded the object as often as a call
/// may and each cold start's entry still failed its repair, so this call
/// alone takes the WATCH path. Never a negative entry, never a cold-start
/// `route="uncacheable"` tick: each tick is one call. Its log line is DEBUG,
/// since it can recur on every edit of a layer (a rolling deploy).
pub(crate) const STRUCTURAL_CACHE_UNCACHEABLE_RELOAD_LIMIT: &str = "reload_limit";
/// `reason` label: one structural call's fenced cold-start read failed — its
/// waiting reached the call's cap, a capacity wait timed out, or three
/// attempts in a row were broken — so this call alone takes the WATCH path.
/// Never a negative entry, never a cold-start `route` tick: each tick is one
/// call, and the next call cold-starts the object afresh. Its log line is
/// DEBUG, since nothing bounds its rate while the layer stays busy.
pub(crate) const STRUCTURAL_CACHE_UNCACHEABLE_FENCED_READ_FAILED: &str = "fenced_read_failed";
/// `reason` label: the layer is too large for its fenced cold-start read to
/// fit the replica's configured RPC deadline (`LIVE_EDIT_RPC_DEADLINE_MS`)
/// with time left for one WATCH attempt, so no call on the replica could
/// load it. Judged on the layer's size and the configuration alone, before
/// any fenced read, and recorded as a negative entry: one tick and one INFO
/// line per verdict, not per edit. The operator's remedies are a longer RPC
/// deadline or a smaller layer.
pub(crate) const STRUCTURAL_CACHE_UNCACHEABLE_OVER_DEADLINE: &str = "over_deadline";

/// The complete `reason` vocabulary of
/// `live_edit_structural_cache_uncacheable_total`, zero-seeded, in
/// `UncacheableReason`'s order.
pub const STRUCTURAL_CACHE_UNCACHEABLE_REASONS: &[&str] = &[
    STRUCTURAL_CACHE_UNCACHEABLE_OVER_WEIGHT,
    STRUCTURAL_CACHE_UNCACHEABLE_NO_CHS_COMPLETE,
    STRUCTURAL_CACHE_UNCACHEABLE_RELOAD_LIMIT,
    STRUCTURAL_CACHE_UNCACHEABLE_FENCED_READ_FAILED,
    STRUCTURAL_CACHE_UNCACHEABLE_OVER_DEADLINE,
];

// ── live_edit_structural_cache_audit_total vocabulary (SR-2.4, SR-3.6) ──────
//
// Recorded by the structural cache's sampled audit
// (`storage::structural_cache_audit`): what one audit that ran to an end
// concluded about the entry it compared.

/// `outcome` label: the audit repaired a copy of the entry to its read's cut
/// and the copy equalled the store's state there.
pub(crate) const STRUCTURAL_CACHE_AUDIT_MATCH: &str = "match";
/// `outcome` label: the repaired copy differed from the store's state at the
/// same cut; the entry was dropped. Expected to stay 0.
pub(crate) const STRUCTURAL_CACHE_AUDIT_DIVERGED: &str = "diverged";
/// `outcome` label: the audit could not compare — the object was above the
/// read ceiling it may use, gone or replaced, unreadable, its cut was not
/// reachable from the entry by stream repair, or the upload budget did not
/// hold the audit's repair (refused while uploads queue or near full, or
/// over the per-audit cap). No label says which.
pub(crate) const STRUCTURAL_CACHE_AUDIT_SKIPPED: &str = "skipped";

/// The complete `outcome` vocabulary of
/// `live_edit_structural_cache_audit_total`, zero-seeded.
pub const STRUCTURAL_CACHE_AUDIT_OUTCOMES: &[&str] = &[
    STRUCTURAL_CACHE_AUDIT_MATCH,
    STRUCTURAL_CACHE_AUDIT_DIVERGED,
    STRUCTURAL_CACHE_AUDIT_SKIPPED,
];

// ── live_edit_structural_cache_audit_fenced_total vocabulary (SR-3.6) ───────
//
// Recorded by the structural cache's sampled audit
// (`storage::structural_cache_audit`) for each audit of a layer above the
// plain atomic-read ceilings with the above-ceiling switch on: whether it took
// the fenced read, and if not, why not.

/// `result` label: the audit's fenced read verified a cut; the audit went on
/// to compare. Each one paused that layer's colliding writers for its hold
/// (`live_edit_snapshot_fence_hold_seconds{consumer="structural_cache_audit"}`).
pub(crate) const STRUCTURAL_CACHE_AUDIT_FENCED_READ: &str = "read";
/// `result` label: no fenced read: one already ran on this replica within the
/// pacing interval (`STRUCTURAL_CACHE_AUDIT_FENCED_MIN_INTERVAL`).
pub(crate) const STRUCTURAL_CACHE_AUDIT_FENCED_PACED: &str = "paced";
/// `result` label: the fenced read ended without waiting because the cache's
/// cold-start permits, the fenced read permit or the catch-up memory were
/// not free at once: the audit never queues ahead of a cold start or a
/// catch-up.
pub(crate) const STRUCTURAL_CACHE_AUDIT_FENCED_BUSY: &str = "busy";
/// `result` label: the fenced read failed: its wait for the fence passed
/// `reader_wait_max(L)`, its attempts were all broken, or a store error.
pub(crate) const STRUCTURAL_CACHE_AUDIT_FENCED_FAILED: &str = "failed";

/// The complete `result` vocabulary of
/// `live_edit_structural_cache_audit_fenced_total`, zero-seeded.
pub const STRUCTURAL_CACHE_AUDIT_FENCED_RESULTS: &[&str] = &[
    STRUCTURAL_CACHE_AUDIT_FENCED_READ,
    STRUCTURAL_CACHE_AUDIT_FENCED_PACED,
    STRUCTURAL_CACHE_AUDIT_FENCED_BUSY,
    STRUCTURAL_CACHE_AUDIT_FENCED_FAILED,
];

/// `backend` label value for `update_layer_commit_already_landed_total` and
/// `update_layer_operation_overtaken_total` on the routes that commit through
/// `update_layer_fast.lua`: the fast path and the cached structural route
/// (whose read-side commit-token check answers under it too).
pub(crate) const UPDATE_LAYER_BACKEND_LUA: &str = "lua";
/// `backend` label value for `update_layer_commit_already_landed_total` and
/// `update_layer_operation_overtaken_total` on the WATCH/MULTI/EXEC slow path.
pub(crate) const UPDATE_LAYER_BACKEND_TXN: &str = "txn";
/// Every `backend` value `update_layer_commit_already_landed_total` and
/// `update_layer_operation_overtaken_total` can carry; both cells of each are
/// seeded at zero in [`Metrics::new`]. The same vocabulary labels
/// `live_edit_update_layer_fence_wait_seconds{backend}` (Spec SDD SR-8.3).
pub const UPDATE_LAYER_BACKENDS: &[&str] = &[UPDATE_LAYER_BACKEND_LUA, UPDATE_LAYER_BACKEND_TXN];

/// `outcome` label value for
/// `live_edit_update_layer_lost_reply_release_lookup_total`: the call's own
/// commit release entry was found and its releases handed to the enqueue
/// (whose own failure is counted on `ext_ref_release_enqueue_failed_total`).
pub const LOST_REPLY_RELEASE_LOOKUP_ENQUEUED: &str = "enqueued";
/// `outcome` label value: the call's token held no entry — no commit of the
/// call landed, it owed no release, or later releasing commits on the object
/// had evicted its entry from the ring.
pub const LOST_REPLY_RELEASE_LOOKUP_ABSENT: &str = "absent";
/// `outcome` label value: the read failed or the entry did not parse; the
/// releases, if any were owed, are left to the reference-count reconciler.
pub const LOST_REPLY_RELEASE_LOOKUP_FAILED: &str = "failed";
/// Every `outcome` value
/// `live_edit_update_layer_lost_reply_release_lookup_total` can carry, each
/// seeded at zero in [`Metrics::new`].
pub const LOST_REPLY_RELEASE_LOOKUP_OUTCOMES: &[&str] = &[
    LOST_REPLY_RELEASE_LOOKUP_ENQUEUED,
    LOST_REPLY_RELEASE_LOOKUP_ABSENT,
    LOST_REPLY_RELEASE_LOOKUP_FAILED,
];

/// `mode` label value for the `{backend="lua", site="update"}` cells of
/// `live_edit_keyspace_type_error_total` and
/// `live_edit_stream_id_collision_total`: the refusal came from the update
/// script's fast mode (`LayerStore::update_layer_once_lua`).
pub const UPDATE_SCRIPT_MODE_FAST: &str = "fast";
/// `mode` label value for the same cells: the refusal came from the update
/// script's structural mode (`storage::structural::commit_structural`).
pub const UPDATE_SCRIPT_MODE_STRUCTURAL: &str = "structural";
/// Every `mode` value the `{backend="lua", site="update"}` cells of
/// `live_edit_keyspace_type_error_total` and
/// `live_edit_stream_id_collision_total` can carry; each is seeded at zero in
/// [`Metrics::new`]. Only those cells carry `mode`: every other cell of the two
/// families is raised by something other than the update script, so it has no
/// script mode and keeps its `{backend, site}` label set.
pub const UPDATE_SCRIPT_MODES: &[&str] = &[UPDATE_SCRIPT_MODE_FAST, UPDATE_SCRIPT_MODE_STRUCTURAL];

/// `path` label value for the `UpdateLayer` commit metrics: the Lua field /
/// time-sample fast path (`update_layer_fast.lua`), taken when the backend is
/// `lua` or `lua-watch-structural` and the delta needs no section-tree read
/// (`storage::lua::is_fast_path`).
pub const UPDATE_LAYER_PATH_FAST: &str = "fast";
/// `path` label value for the `UpdateLayer` commit metrics: the
/// WATCH/MULTI/EXEC slow path, for any delta it commits — structural deltas
/// and resets under `lua-watch-structural`, every delta under `txn`, and under
/// `lua` a structural or reset call the structural cache cannot hold (an
/// uncacheable object), that the size gate sent there because its commit was
/// too large for one structural script call (`storage::structural_size_gate`:
/// a plan over the work limit, or a reset of an uncached layer estimated over
/// it), or that fell back after a repeated `CACHE_DIVERGED`.
pub const UPDATE_LAYER_PATH_WATCH: &str = "watch";
/// `path` label value for the `UpdateLayer` commit metrics: the `seq`-fenced
/// structural commit, under `lua`, whose attempt planned on a cache hit
/// (repaired from the stream or not).
pub const UPDATE_LAYER_PATH_STRUCTURAL: &str = "structural";
/// `path` label value for the `UpdateLayer` commit metrics: a structural commit
/// whose attempt started from a cold-start or reload read of the structural
/// cache.
pub const UPDATE_LAYER_PATH_STRUCTURAL_FULL: &str = "structural_full";
/// `path` label value for the `UpdateLayer` commit metrics: a reset delta
/// committed by the `seq`-fenced structural commit under `lua`, from a cache
/// hit or a cold start alike.
pub const UPDATE_LAYER_PATH_RESET: &str = "reset";
/// Every `path` value the `UpdateLayer` commit metrics can carry —
/// `live_edit_update_layer_retries_total`,
/// `live_edit_update_layer_attempts_histogram`,
/// `live_edit_update_layer_commit_duration_seconds` and
/// `live_edit_update_layer_conflicts_total`. A call's path is that of its
/// committing (or last) attempt.
///
/// Under `lua` every value is emitted; under `lua-watch-structural` only
/// [`UPDATE_LAYER_PATH_FAST`] and [`UPDATE_LAYER_PATH_WATCH`], and under `txn`
/// only [`UPDATE_LAYER_PATH_WATCH`]. The two counters are seeded over the
/// whole list whatever the backend (see the seeding site in
/// [`Metrics::new`]).
pub const UPDATE_LAYER_PATHS: &[&str] = &[
    UPDATE_LAYER_PATH_FAST,
    UPDATE_LAYER_PATH_WATCH,
    UPDATE_LAYER_PATH_STRUCTURAL,
    UPDATE_LAYER_PATH_STRUCTURAL_FULL,
    UPDATE_LAYER_PATH_RESET,
];

/// `kind` label value for `live_edit_update_layer_conflicts_total`: a
/// WATCH-cancelled (nil) `EXEC` on the slow path — a watched key (the layer's
/// header, stream or snapshot fence) changed after the attempt read it, and
/// the `EXEC`'s re-probe found no live fence. Usually another edit's commit,
/// but also a checkpoint cursor stamp on the header
/// (`storage::lua::stamp_obj_header_cursor`), which is no edit at all, or a
/// fence-key write that left no live lease (a release, or the key expiring).
pub const UPDATE_LAYER_CONFLICT_WATCH_CONFLICT: &str = "watch_conflict";
/// `kind` label value for `live_edit_update_layer_conflicts_total`: the WATCH
/// path's structural snapshot read failed its `seq` proof ("object snapshot
/// changed while loading structural update",
/// `LayerStore::read_structural_snapshot`). Counted only when the proof's
/// re-read `seq` is present and moved, not when it is absent (the header
/// deleted or stripped mid-attempt, which is not a lost race).
///
/// Kept apart from [`UPDATE_LAYER_CONFLICT_WATCH_CONFLICT`] because it is
/// today's dominant cost: aborted full reads were 85 % of structural-attempt
/// CPU in the b.2wo spike (Spec SDD SR-2.3).
pub const UPDATE_LAYER_CONFLICT_WATCH_READ_MOVED: &str = "watch_read_moved";
/// `kind` label value for `live_edit_update_layer_conflicts_total`: a
/// `seq`-fenced structural commit refused `STALE` because the header's `seq`
/// or `commit_chain` moved since the cache entry it planned on (Spec SDD
/// SR-6.4), under the attempt's path (`structural`, `structural_full` or
/// `reset`).
pub const UPDATE_LAYER_CONFLICT_SEQ_FENCE: &str = "seq_fence";
/// `kind` label value for `live_edit_update_layer_conflicts_total`: the
/// structural commit's `CACHE_DIVERGED` reply (Spec SDD SR-6.4), under the
/// attempt's path, beside `live_edit_structural_cache_divergence_total`.
pub const UPDATE_LAYER_CONFLICT_CACHE_DIVERGED: &str = "cache_diverged";
/// `kind` label value for `live_edit_update_layer_conflicts_total`: an
/// attempt a fenced snapshot reader's live lease refused, which the call
/// waited out without spending a retry (b.y3m SR-5.4, structural-edits
/// SR-8.1), on any path — the update script's step-1d `FENCED` reply in
/// either mode (`fast`, `structural`, `structural_full`, `reset`), or on the
/// WATCH path (`watch`) its probe, a step-2b intent registration refused
/// `FENCED`, or an `EXEC` cancelled by a fence grant.
pub const UPDATE_LAYER_CONFLICT_SNAPSHOT_FENCE_WAIT: &str = "snapshot_fence_wait";
/// Every `kind` value of `live_edit_update_layer_conflicts_total`.
///
/// A conflict is an attempt abandoned because the layer changed under it: on
/// the WATCH path, a watched key (header, stream or snapshot fence) changed
/// after the read — usually another edit's commit, but also a checkpoint
/// cursor stamp on the header (`watch_conflict`) — or the structural snapshot
/// read failed its `seq` proof (`watch_read_moved`); on the structural route,
/// the commit's `STALE` (`seq_fence`) or `CACHE_DIVERGED` (`cache_diverged`)
/// refusal; and on every path, a live snapshot fence waited out
/// (`snapshot_fence_wait`, which spends no retry attempt). Other retriable
/// failures of an attempt —
/// transient backing-store I/O, a `WATCH` connection found replaced, a
/// stream-collision verdict that could not be disambiguated — are retried the
/// same way but are not counted here.
pub const UPDATE_LAYER_CONFLICT_KINDS: &[&str] = &[
    UPDATE_LAYER_CONFLICT_WATCH_CONFLICT,
    UPDATE_LAYER_CONFLICT_WATCH_READ_MOVED,
    UPDATE_LAYER_CONFLICT_SEQ_FENCE,
    UPDATE_LAYER_CONFLICT_CACHE_DIVERGED,
    UPDATE_LAYER_CONFLICT_SNAPSHOT_FENCE_WAIT,
];

/// `outcome` label value for `live_edit_update_layer_retries_total`: the
/// retries were taken by a call that then committed.
pub const UPDATE_LAYER_RETRY_OUTCOME_COMMITTED: &str = "committed";
/// `outcome` label value for `live_edit_update_layer_retries_total`: the
/// retries were taken by a call that ran out of attempts.
pub const UPDATE_LAYER_RETRY_OUTCOME_EXHAUSTED: &str = "exhausted";
/// Every `outcome` value of `live_edit_update_layer_retries_total`.
pub const UPDATE_LAYER_RETRY_OUTCOMES: &[&str] = &[
    UPDATE_LAYER_RETRY_OUTCOME_COMMITTED,
    UPDATE_LAYER_RETRY_OUTCOME_EXHAUSTED,
];

/// `reason` label value for `live_edit_update_layer_structural_fallbacks_total`:
/// a structural commit refused `CACHE_DIVERGED` again after the call's first
/// `CACHE_DIVERGED` had dropped the cache entry and the next attempt reloaded
/// it, so the call committed through the WATCH path instead of looping.
pub const UPDATE_LAYER_STRUCTURAL_FALLBACK_REPEAT_DIVERGENCE: &str = "repeat_divergence";
/// `reason` label value for `live_edit_update_layer_structural_fallbacks_total`:
/// the plan's work was over the structural commit's work limit
/// (`storage::structural_size_gate::STRUCTURAL_COMMIT_WORK_LIMIT`), so the
/// call committed through the WATCH path rather than hold the shard in one
/// long script.
pub const UPDATE_LAYER_STRUCTURAL_FALLBACK_OVER_WORK_LIMIT: &str = "over_work_limit";
/// `reason` label value for `live_edit_update_layer_structural_fallbacks_total`:
/// a reset of a layer with no cache entry whose work, estimated from the
/// cache probe's index cardinalities and the delta's rows
/// (`storage::structural_size_gate::reset_work_lower_bound`), was over the
/// work limit, so the call went to the WATCH path without the cold start's
/// full read.
pub const UPDATE_LAYER_STRUCTURAL_FALLBACK_RESET_OVER_WORK_LIMIT: &str = "reset_over_work_limit";
/// Every `reason` value of `live_edit_update_layer_structural_fallbacks_total`,
/// each seeded at zero.
pub const UPDATE_LAYER_STRUCTURAL_FALLBACK_REASONS: &[&str] = &[
    UPDATE_LAYER_STRUCTURAL_FALLBACK_REPEAT_DIVERGENCE,
    UPDATE_LAYER_STRUCTURAL_FALLBACK_OVER_WORK_LIMIT,
    UPDATE_LAYER_STRUCTURAL_FALLBACK_RESET_OVER_WORK_LIMIT,
];

/// `outcome` label value for `live_edit_update_layer_commit_duration_seconds`:
/// `update_layer` returned `Ok`.
pub const UPDATE_LAYER_COMMIT_OUTCOME_OK: &str = "ok";
/// `outcome` label value for `live_edit_update_layer_commit_duration_seconds`:
/// `update_layer` returned an error of any kind — a validation refusal, retry
/// exhaustion, or any other failure — or its future was dropped before it
/// returned.
pub const UPDATE_LAYER_COMMIT_OUTCOME_ERROR: &str = "error";
/// Every `outcome` value of `live_edit_update_layer_commit_duration_seconds`.
pub const UPDATE_LAYER_COMMIT_OUTCOMES: &[&str] = &[
    UPDATE_LAYER_COMMIT_OUTCOME_OK,
    UPDATE_LAYER_COMMIT_OUTCOME_ERROR,
];

/// `rpc` label value for `operation_id_match_total` on `UpdateLayer`.
pub const OPERATION_ID_RPC_UPDATE_LAYER: &str = "update_layer";
/// `rpc` label value for `operation_id_match_total` on `CreateLayer`.
pub const OPERATION_ID_RPC_CREATE_LAYER: &str = "create_layer";
/// Every `rpc` value `operation_id_match_total` can carry.
pub const OPERATION_ID_RPCS: &[&str] =
    &[OPERATION_ID_RPC_UPDATE_LAYER, OPERATION_ID_RPC_CREATE_LAYER];
/// `outcome` label value for `operation_id_match_total`: a re-send was
/// answered from the commit an earlier call with the same `operation_id`
/// recorded, and applied nothing.
pub const OPERATION_ID_OUTCOME_ALREADY_COMMITTED: &str = "already_committed";
/// `outcome` label value for `operation_id_match_total`: the recorded
/// operation's request digest differed from this call's, and the call was
/// refused `ALREADY_EXISTS`.
pub const OPERATION_ID_OUTCOME_CONFLICT: &str = "conflict";
/// Every `outcome` value `operation_id_match_total` can carry. All
/// `OPERATION_ID_RPCS × OPERATION_ID_OUTCOMES` cells are reachable and seeded
/// at zero in [`Metrics::new`].
pub const OPERATION_ID_OUTCOMES: &[&str] = &[
    OPERATION_ID_OUTCOME_ALREADY_COMMITTED,
    OPERATION_ID_OUTCOME_CONFLICT,
];

/// `result` label value on `live_edit_update_layer_fence_intent_total`: a
/// WATCH-path attempt that reads a snapshot registered its writer intent
/// (Spec SDD SR-8.3; the cached structural route registers none,
/// structural-edits SR-8.2).
pub(crate) const UPDATE_LAYER_FENCE_INTENT_REGISTERED: &str = "registered";
/// `result` label value on `live_edit_update_layer_fence_intent_total`: one
/// registration was refused because a fenced reader held the layer. It counts
/// the refused registration, not a fence wait: a refusal before `WATCH` is
/// ignored and the attempt falls through to its guards (it may then report an
/// already-landed commit, refuse, or register at step 2b if the fence ended),
/// so one attempt can tick both this and `registered`. Only a refusal at step
/// 2b ends the attempt as a fence wait.
pub(crate) const UPDATE_LAYER_FENCE_INTENT_REFUSED_FENCED: &str = "refused_fenced";
/// Every `result` value `live_edit_update_layer_fence_intent_total` can carry;
/// both are seeded at zero in [`Metrics::new`].
pub const UPDATE_LAYER_FENCE_INTENT_RESULTS: &[&str] = &[
    UPDATE_LAYER_FENCE_INTENT_REGISTERED,
    UPDATE_LAYER_FENCE_INTENT_REFUSED_FENCED,
];

/// `path` label value for `storage_watch_voided_total` on the `UpdateLayer`
/// structural slow path (b.bhb).
pub(crate) const WATCH_VOIDED_PATH_UPDATE_LAYER: &str = "update_layer";
/// `path` label value for `storage_watch_voided_total` on `CreateSession`
/// (b.bhb).
pub(crate) const WATCH_VOIDED_PATH_CREATE_SESSION: &str = "create_session";
/// `path` label value for `storage_watch_voided_total` on the DR restore's
/// pre-write residue wipe (b.f4t).
///
/// The third watched-transaction site, and the only one whose committed effect
/// is an irreversible `UNLINK` of a surviving keyspace: a `committed_unwatched`
/// tick here means the `obj:hdr` fence that was supposed to stop the wipe from
/// deleting a peer replica's freshly landed restore did not actually run.
pub(crate) const WATCH_VOIDED_PATH_RESTORE_RESIDUE_WIPE: &str = "restore_residue_wipe";
/// `path` label value for `storage_watch_voided_total` on the DR restore's
/// object materialization — `LayerStore::write_restored_object` (b.h4m).
///
/// The fourth watched-transaction site, and the sibling of
/// [`WATCH_VOIDED_PATH_RESTORE_RESIDUE_WIPE`] one step later on the same path:
/// where the wipe's `WATCH` stops a peer replica's freshly landed restore from
/// being `UNLINK`ed, this one stops it from being *overwritten*. Its `WATCH` on
/// `le:{oid}:obj:hdr` requires the header to be ABSENT, so the two replicas'
/// `EXEC`s serialize on the `{object_id}` shard and exactly one materialization
/// lands; the loser writes nothing and reports the layer present.
///
/// A `committed_unwatched` tick here means that fence did not actually run: the
/// materialization executed on a connection that never held the `WATCH`, so a
/// peer's rebuild may have been merged with this replica's — the
/// mixture-of-two-backups shape, which is silent content corruption rather than
/// a delete.
pub(crate) const WATCH_VOIDED_PATH_RESTORE_OBJECT_WRITE: &str = "restore_write_object";
/// `disposition` label value for a watched transaction abandoned BEFORE the
/// commit was dispatched, because the pre-dispatch connection-identity check
/// found the `WATCH`'s connection replaced (b.bhb). Nothing was written; the
/// attempt re-enters the caller's `Aborted` retry with a fresh `WATCH`. This is
/// the detection working — a nonzero rate is expected on a fleet that sees
/// failover or proxy recycles.
pub(crate) const WATCH_VOIDED_ABORTED_PRE_EXEC: &str = "aborted_pre_exec";
/// `disposition` label value for a commit PROVEN to have executed on a
/// connection other than the one holding its `WATCH` (b.bhb) — the in-transaction
/// `CLIENT ID` witness disagreed with the baseline sampled at `WATCH` time, or
/// the `EXEC` array came back torn. **This is the alertable cell.** It means an
/// optimistic-concurrency check did not actually run: on `create_session` a
/// duplicate name may have survived, on `update_layer` the commit is torn
/// between materialized state and the live relay stream, on
/// `restore_residue_wipe` a peer replica's freshly landed restore may have been
/// `UNLINK`ed by a wipe its fence should have canceled, and on
/// `restore_write_object` a peer's rebuild may have been merged with this
/// replica's — the mixture-of-two-backups shape, which is silent content
/// corruption rather than a delete.
pub(crate) const WATCH_VOIDED_COMMITTED_UNWATCHED: &str = "committed_unwatched";
/// `disposition` label value for a proven-unwatched commit whose effects were
/// successfully undone (b.bhb) — today only `create_session`, whose two `SADD`s
/// are both uniquely ours and are `SREM`'d back out before the attempt is
/// retried. Consequence-free by itself; read it as the denominator that makes a
/// `committed_unwatched` tick on the same path mean "compensation FAILED".
pub(crate) const WATCH_VOIDED_COMPENSATED: &str = "compensated";
/// `disposition` label value for a watched transaction that ran with the
/// in-transaction `CLIENT ID` witness (L3) **disarmed**, because the backing
/// store would not answer a stable `CLIENT ID` from inside a `MULTI` (b.bhb).
/// Consequence-free per attempt — the commit proceeds normally — but it says
/// the deployment is running on L1 + L2 only, so a commit that executed on a
/// connection which never held its `WATCH` would go undetected there. Expected
/// to be either zero or ~every watched commit: it is a property of the store,
/// not an event, which is exactly what makes it the cell a DR / dev-AMR run
/// reads to learn how a proxied managed store behaves. Not alertable-critical;
/// alert on it only as a deployment-configuration signal.
pub(crate) const WATCH_VOIDED_WITNESS_UNAVAILABLE: &str = "witness_unavailable";
/// `disposition` label value for a watched transaction whose **pre-dispatch
/// identity check (L2) could not be evaluated** (b.bhb): fred is connected but
/// holds no `CLIENT ID` for the connection carrying the `WATCH`, which is what
/// a store that ACL-denies or proxy-rejects `CLIENT ID` at handshake produces
/// on every healthy connection. The attempt DISPATCHES — failing closed here
/// would abort every `CreateSession` and every structural `UpdateLayer` to
/// retry exhaustion — so, like its sibling above, this is a
/// store-property signal rather than an incident. Same reading: zero, or
/// ~every watched commit.
pub(crate) const WATCH_VOIDED_FINGERPRINT_UNAVAILABLE: &str = "fingerprint_unavailable";
/// Every `(path, disposition)` cell of `storage_watch_voided_total`, so
/// [`Metrics::new`] pre-registers the full label set and an `increase(...) > 0`
/// alert on `committed_unwatched` has a series to evaluate before the first
/// void (b.bhb). Not every combination is reachable —
/// `{path="update_layer", disposition="compensated"}` never ticks, because a
/// structural commit cannot be undone from the client, and neither does
/// `{path="restore_residue_wipe", disposition="compensated"}`, because an
/// `UNLINK` cannot be undone at all, and neither does
/// `{path="restore_write_object", disposition="compensated"}`, because a
/// materialization that landed cannot be un-landed from the client — but
/// registering the cross product keeps the alert expression uniform across
/// paths.
pub(crate) const WATCH_VOIDED_PATHS: &[&str] = &[
    WATCH_VOIDED_PATH_UPDATE_LAYER,
    WATCH_VOIDED_PATH_CREATE_SESSION,
    WATCH_VOIDED_PATH_RESTORE_RESIDUE_WIPE,
    WATCH_VOIDED_PATH_RESTORE_OBJECT_WRITE,
];
/// Dispositions of [`WATCH_VOIDED_PATHS`]; see the constants above.
pub(crate) const WATCH_VOIDED_DISPOSITIONS: &[&str] = &[
    WATCH_VOIDED_ABORTED_PRE_EXEC,
    WATCH_VOIDED_COMMITTED_UNWATCHED,
    WATCH_VOIDED_COMPENSATED,
    WATCH_VOIDED_WITNESS_UNAVAILABLE,
    WATCH_VOIDED_FINGERPRINT_UNAVAILABLE,
];

/// `reason` label value for a `txn`-pool checkout whose `DISCARD` probe was
/// **acknowledged** (b.7td). `+OK` from `DISCARD` is only possible when a
/// `MULTI` was already open on the connection, so this cell is the direct
/// evidence that a previous borrower released the slot mid-transaction.
pub(crate) const TXN_DIRTY_REASON_LEAKED_MULTI: &str = "leaked_multi";
/// `reason` label value for a `txn`-pool checkout probe that got a reply of a
/// shape neither verdict allows (b.7td) — e.g. an integer where `+OK` or
/// `-ERR ... without MULTI` was the only possible answer. That means the
/// connection's reply stream is offset against its command stream, which is the
/// unrecoverable state fred cannot resynchronize.
pub(crate) const TXN_DIRTY_REASON_UNEXPECTED_REPLY: &str = "unexpected_reply";
/// `reason` label value for a `txn`-pool checkout probe that hit its
/// per-command deadline (b.7td). `DISCARD` / `UNWATCH` are O(1) argument-free
/// commands, so a timeout cannot mean "big payload on a slow link" — it means
/// the connection is wedged.
pub(crate) const TXN_DIRTY_REASON_PROBE_TIMEOUT: &str = "probe_timeout";
/// `reason` label value for a `txn`-pool checkout probe that failed with a fred
/// error which is neither a timeout nor connection teardown (b.7td). The slot's
/// state is unknown, which is treated as dirty.
pub(crate) const TXN_DIRTY_REASON_PROBE_ERROR: &str = "probe_error";
/// `reason` label value for a `txn`-pool checkout probe answered with a
/// cluster-topology / node-state transition error frame that fred's router did
/// **not** intercept (b.289 sub-task 3).
///
/// The classifier's prefix set is
/// `storage::txn_checkout::TOPOLOGY_TRANSITION_PREFIXES` — all seven of
/// `MOVED`, `ASK`, `CLUSTERDOWN`, `LOADING`, `TRYAGAIN`, `MASTERDOWN`,
/// `READONLY`. `MOVED` / `ASK` / `CLUSTERDOWN` are in it even though fred does
/// intercept their canonical renderings, because every one of those
/// interceptions is a case-**sensitive** prefix test, so a
/// non-canonically-cased rendering reaches this classifier regardless.
/// `MOVED` / `ASK` escape on shape as well (`is_redirection` also requires a
/// three-token payload) and are only tested on a clustered client;
/// `CLUSTERDOWN` has no such escape — it is tested on every read frame — so
/// casing is its only one. The other four are never intercepted on this build
/// at all.
///
/// **The one `reason` cell that is expected to be non-zero in normal
/// operation.** Every other cell means something is wrong; this one ticks
/// during a resharding, a failover, or a replica promotion, and only a
/// *sustained* rate outside such a window is actionable. It is deliberately
/// absent from [`TXN_STATE_PROOF_REASONS`]: the probe learned that the node
/// would not serve the command, not that the connection's reply stream is
/// offset, so the slot's state is unknown and the recycle goes through the
/// streak gate rather than the immediate-repair path that can latch the
/// repair-ineffective breaker. The verified per-reply interception matrix
/// behind the split is on
/// [`storage::txn_checkout::is_state_proof_reason`](crate::storage::txn_checkout::is_state_proof_reason)'s
/// doc; `storage::txn_checkout::is_topology_transition_reply` is the classifier
/// that consumes it.
pub(crate) const TXN_DIRTY_REASON_TOPOLOGY_REPLY: &str = "topology_reply";
/// Every `reason` cell of `valkey_txn_dirty_checkouts_total`, in one place so
/// [`Metrics::new`] pre-registers the full label set and an
/// `increase(...) > 0` alert has a series to evaluate before the first dirty
/// checkout (b.7td). Adding a reason without adding it here leaves an
/// alert blind to it, so the list and the constants above are edited together.
pub(crate) const TXN_DIRTY_REASONS: &[&str] = &[
    TXN_DIRTY_REASON_LEAKED_MULTI,
    TXN_DIRTY_REASON_UNEXPECTED_REPLY,
    TXN_DIRTY_REASON_PROBE_TIMEOUT,
    TXN_DIRTY_REASON_PROBE_ERROR,
    TXN_DIRTY_REASON_TOPOLOGY_REPLY,
];

/// The subset of [`TXN_DIRTY_REASONS`] the probe *proved* from a reply it
/// actually received, rather than inferred from the absence of one (b.7td
/// round 3).
///
/// Single source of truth for two consumers that must not drift:
/// `storage::txn_checkout::is_state_proof_reason` (which verdicts skip the
/// unknown-state streak gate and recycle immediately) and the pre-registration
/// of `valkey_txn_repair_ineffective_total`'s `reason` cells (which verdicts the
/// repair-ineffective breaker can ever report). A reason that is state-proof but
/// missing here would leave the breaker's series absent for it; a reason listed
/// here but not state-proof would advertise a cell nothing can fill.
///
/// [`TXN_DIRTY_REASON_TOPOLOGY_REPLY`] is deliberately **not** here (b.289
/// sub-task 3): its frame proves what the *node* would not do, not what the
/// *connection's reply stream* is doing, and a routine resharding must not be
/// able to latch a breaker whose escalation accuses the classifier.
pub(crate) const TXN_STATE_PROOF_REASONS: &[&str] = &[
    TXN_DIRTY_REASON_LEAKED_MULTI,
    TXN_DIRTY_REASON_UNEXPECTED_REPLY,
];

/// `reason` label value for an `ext_meta` reference released because a commit
/// **superseded** it (b.qpf): the last value row of the object holding that
/// hash was rewritten to a different hash, rewritten to an inline value, or
/// deleted.
pub(crate) const EXT_REF_RELEASE_REASON_REWRITE: &str = "rewrite";
/// `reason` label value for an `ext_meta` reference released because the
/// pre-flight `store_values` increment was **surplus** (b.qpf): the object
/// already held a reference to that hash before the commit, so the commit's
/// `+1` did not represent a new referencing object and is handed straight back.
/// Expected to be the busiest cell on a workload that re-writes the same blob.
pub(crate) const EXT_REF_RELEASE_REASON_SURPLUS_ACQUIRE: &str = "surplus_acquire";
/// `reason` label value for an `ext_meta` reference released because the
/// **object itself retired** (b.qpf) — `DeleteLayer`, a `CreateLayer`
/// overwrite, or the orphan sweep.
pub(crate) const EXT_REF_RELEASE_REASON_RETIREMENT: &str = "retirement";
/// `reason` label value for an `ext_meta` reference released by the
/// **reference-count reconciler** (b.yda): a periodic leader-gated pass found
/// `ref_count` standing above the number of live objects that reference the
/// hash, twice a grace apart at the same surplus amount, and emitted repair
/// members to hand the difference back.
///
/// Distinct from the three commit/retirement reasons because it is the only one
/// not produced by a keyspace event — it is produced by *observation*, so a
/// non-zero rate here says some other release path failed to record an
/// obligation earlier (a crash between `EXEC` and enqueue, a pre-b.qpf object's
/// unseeded index, a suppressed restore surplus). Read it as the drain of the
/// documented leak-direction residuals, not as routine traffic.
pub(crate) const EXT_REF_RELEASE_REASON_RECONCILE: &str = "reconcile";
/// Every `reason` cell of `ext_ref_released_total`, in one place so
/// [`Metrics::new`] pre-registers the full label set (b.qpf). Same discipline
/// as [`TXN_DIRTY_REASONS`]: adding a reason without adding it here leaves the
/// series absent until the first occurrence.
pub(crate) const EXT_REF_RELEASE_REASONS: &[&str] = &[
    EXT_REF_RELEASE_REASON_REWRITE,
    EXT_REF_RELEASE_REASON_SURPLUS_ACQUIRE,
    EXT_REF_RELEASE_REASON_RETIREMENT,
    EXT_REF_RELEASE_REASON_RECONCILE,
];

/// `outcome` label value of `live_edit_gc_decrement_total` (b.yuo): the
/// decrement applied now (`decremented`, `deleted` or `deleted_underflow`).
pub(crate) const GC_DECREMENT_OUTCOME_APPLIED: &str = "applied";
/// `outcome` label value of `live_edit_gc_decrement_total` (b.yuo): the
/// member's decrement had already applied, so this run decremented nothing —
/// a fred re-send, or the in-place re-claim of a stale claim. Expected to tick; it is the
/// at-most-once guarantee working.
pub(crate) const GC_DECREMENT_OUTCOME_REPLAYED: &str = "replayed";
/// `outcome` label value of `live_edit_gc_decrement_total` (b.yuo): the
/// decrement executed more than the claim-age fence after its claim and was
/// refused with no write; the member stays in inflight until it is re-claimed. A
/// sustained rate means a stalled store shard or a primary clock running ahead
/// of the `{gc}` primary's.
pub(crate) const GC_DECREMENT_OUTCOME_STALE: &str = "stale";
/// Every `outcome` cell of `live_edit_gc_decrement_total`, zero-seeded at
/// construction (b.yuo).
pub(crate) const GC_DECREMENT_OUTCOMES: &[&str] = &[
    GC_DECREMENT_OUTCOME_APPLIED,
    GC_DECREMENT_OUTCOME_REPLAYED,
    GC_DECREMENT_OUTCOME_STALE,
];

/// `event` label value of `live_edit_gc_older_build_total` (b.1jt): this
/// build took the GC leader lock `le:gc:leader` over from an older build (a
/// value with no protocol marker, or a lower version). Counted per takeover.
pub(crate) const GC_OLDER_BUILD_EVENT_LOCK_TAKEN_OVER: &str = "lock_taken_over";
/// `event` label value of `live_edit_gc_older_build_total` (b.1jt): the GC
/// loop dropped, without decrementing, a claim an older build left in
/// `inflight`. Counted per member.
pub(crate) const GC_OLDER_BUILD_EVENT_CLAIM_DROPPED: &str = "claim_dropped";
/// `event` label value of `live_edit_gc_older_build_total` (b.1jt): the GC
/// loop stamped an age on a pending release an older build enqueued without
/// one. Counted per member.
pub(crate) const GC_OLDER_BUILD_EVENT_RELEASE_STAMPED: &str = "release_stamped";
/// `event` label value of `live_edit_gc_older_build_total` (b.1jt): the
/// orphan sweep dropped, without paying, an older build's retirement carrier
/// none of whose recorded attempts landed. Counted per release reference on
/// the carrier.
pub(crate) const GC_OLDER_BUILD_EVENT_CARRIER_DROPPED: &str = "carrier_dropped";
/// Every `event` cell of `live_edit_gc_older_build_total`, zero-seeded at
/// construction (b.1jt).
pub(crate) const GC_OLDER_BUILD_EVENTS: &[&str] = &[
    GC_OLDER_BUILD_EVENT_LOCK_TAKEN_OVER,
    GC_OLDER_BUILD_EVENT_CLAIM_DROPPED,
    GC_OLDER_BUILD_EVENT_RELEASE_STAMPED,
    GC_OLDER_BUILD_EVENT_CARRIER_DROPPED,
];

/// `outcome` label value of `live_edit_gc_oom_collections_total` (b.pa1): the
/// GC leader's out-of-memory collection deleted at least one released external
/// value.
pub(crate) const GC_OOM_COLLECTION_OUTCOME_FREED: &str = "freed";
/// `outcome` label value of `live_edit_gc_oom_collections_total` (b.pa1): the
/// collection found released members but deleted none, and none of its own
/// scripts (a decrement, the upload hold) was refused `-OOM`. Either each
/// member it decremented still had another reference — then, if the store
/// keeps refusing writes for memory, it is full of data still in use and
/// needs more memory — or the store refused `-OOM` the re-claims or the age
/// stamp on its full `{gc}` shard, which a pass skips, so owed releases wait
/// in the release ledger for room there (the WARN's `oom_skipped` field counts
/// those skips).
pub(crate) const GC_OOM_COLLECTION_OUTCOME_FREED_NOTHING: &str = "freed_nothing";
/// `outcome` label value of `live_edit_gc_oom_collections_total` (b.pa1): the
/// collection found no released member it could collect at all. Either there
/// is none; or every owed release is clipped by a live read hold (a catch-up
/// re-reading after a miss), which keeps it out of the selection until the
/// hold ends; or releases made while the `{gc}` shard was full were refused at
/// enqueue and wait for the reference-count reconciler (about 1–2 h at the
/// defaults). Only if writes stay refused through all of that is the store
/// full of data in use and in need of more memory.
pub(crate) const GC_OOM_COLLECTION_OUTCOME_NOTHING_TO_COLLECT: &str = "nothing_to_collect";
/// `outcome` label value of `live_edit_gc_oom_collections_total` (b.pa1): the
/// collection's completed passes deleted nothing (the deletions of the pass
/// that failed are not counted), and a pass of it ended in an error. Not taken
/// as futility: the next request may start another a second later.
pub(crate) const GC_OOM_COLLECTION_OUTCOME_FAILED: &str = "failed";
/// `outcome` label value of `live_edit_gc_oom_collections_total` (b.pa1): the
/// collection deleted nothing because the store refused its own scripts `-OOM`
/// — a decrement, or the write of its upload hold. Each of those scripts opens
/// with a write that cannot grow memory so that it runs on a full store; a
/// refusal means the backing store did not let a script continue past its
/// first write on a full store, so the collection cannot free memory there.
/// Distinct from `freed_nothing` and `nothing_to_collect`, where the
/// collection's own scripts ran.
pub(crate) const GC_OOM_COLLECTION_OUTCOME_REFUSED: &str = "refused";
/// Every `outcome` cell of `live_edit_gc_oom_collections_total`, zero-seeded at
/// construction (b.pa1).
pub(crate) const GC_OOM_COLLECTION_OUTCOMES: &[&str] = &[
    GC_OOM_COLLECTION_OUTCOME_FREED,
    GC_OOM_COLLECTION_OUTCOME_FREED_NOTHING,
    GC_OOM_COLLECTION_OUTCOME_NOTHING_TO_COLLECT,
    GC_OOM_COLLECTION_OUTCOME_FAILED,
    GC_OOM_COLLECTION_OUTCOME_REFUSED,
];

/// `outcome` label value of `live_edit_ext_release_gate_total` (b.yuo): an
/// upload carrying external values waited at the release budget and proceeded
/// once the release total dropped below it.
pub(crate) const EXT_RELEASE_GATE_OUTCOME_PASSED_AFTER_WAIT: &str = "passed_after_wait";
/// `outcome` label value of `live_edit_ext_release_gate_total` (b.yuo): the
/// release total stayed at or above the budget for as long as the call's
/// deadline allowed (its remaining deadline less the gate's margin; at once
/// when no more than the margin was left), and the upload was refused
/// `UNAVAILABLE` with nothing applied.
pub(crate) const EXT_RELEASE_GATE_OUTCOME_REFUSED: &str = "refused";
/// `outcome` label value of `live_edit_ext_release_gate_total` (b.yuo): the
/// release total could not be read, and the upload was let through.
pub(crate) const EXT_RELEASE_GATE_OUTCOME_READ_FAILED: &str = "read_failed";
/// Every `outcome` cell of `live_edit_ext_release_gate_total`, zero-seeded at
/// construction (b.yuo).
pub(crate) const EXT_RELEASE_GATE_OUTCOMES: &[&str] = &[
    EXT_RELEASE_GATE_OUTCOME_PASSED_AFTER_WAIT,
    EXT_RELEASE_GATE_OUTCOME_REFUSED,
    EXT_RELEASE_GATE_OUTCOME_READ_FAILED,
];

/// `outcome` label value of `live_edit_catchup_ext_value_miss_total` (b.yuo):
/// a catch-up build found an external-value blob missing from its snapshot,
/// took a `{gc}` read hold and re-read the layer's state. Expected to tick
/// now and then under large-value churn: a commit after the snapshot released
/// the blob and the GC collected it, which the re-read resolves.
pub(crate) const CATCHUP_EXT_VALUE_MISS_OUTCOME_STALE_RETRIED: &str = "stale_retried";
/// `outcome` label value of `live_edit_catchup_ext_value_miss_total` (b.yuo):
/// a blob was still missing when the catch-up rebuilt from its re-read under
/// its read hold, and the store confirmed the hold had stayed live, so it is
/// genuinely missing; the build then took the
/// checkpoint-adapter fetch-on-miss path, and the catch-up ends `INTERNAL`
/// if that cannot restore it. A genuine miss also ticked `stale_retried`
/// first. Any increase is worth a look.
pub(crate) const CATCHUP_EXT_VALUE_MISS_OUTCOME_GENUINE: &str = "genuine";
/// Every `outcome` cell of `live_edit_catchup_ext_value_miss_total`,
/// zero-seeded at construction (b.yuo).
pub(crate) const CATCHUP_EXT_VALUE_MISS_OUTCOMES: &[&str] = &[
    CATCHUP_EXT_VALUE_MISS_OUTCOME_STALE_RETRIED,
    CATCHUP_EXT_VALUE_MISS_OUTCOME_GENUINE,
];

/// `read` label value of `live_edit_catchup_ext_value_miss_total` (b.yuo):
/// the miss was in a snapshot the catch-up read atomically itself, so its
/// re-read is one more atomic read under its own slot — or, when the layer
/// has grown above the atomic-read limits since (the object's flight
/// registered, or the re-read answering `TOO_LARGE`), a read through the
/// layer's fenced-read flight. A `genuine` miss carries the same `read` as
/// the `stale_retried` it follows.
pub(crate) const CATCHUP_EXT_VALUE_MISS_READ_ATOMIC: &str = "atomic";
/// `read` label value of `live_edit_catchup_ext_value_miss_total` (b.yuo):
/// the miss was in a snapshot shared through the layer's fenced-read flight
/// (above the atomic-read limits), so its re-read is a re-attach to a
/// successor flight: one more fenced read of the layer, shared with every
/// other catch-up that attaches to the same successor.
pub(crate) const CATCHUP_EXT_VALUE_MISS_READ_SHARED_FENCED: &str = "shared_fenced";
/// Every `read` cell of `live_edit_catchup_ext_value_miss_total`, zero-seeded
/// against every outcome at construction (b.yuo).
pub(crate) const CATCHUP_EXT_VALUE_MISS_READS: &[&str] = &[
    CATCHUP_EXT_VALUE_MISS_READ_ATOMIC,
    CATCHUP_EXT_VALUE_MISS_READ_SHARED_FENCED,
];

/// `source` label value of `live_edit_ext_release_budget_bytes` (b.yuo): the
/// release budget comes from configuration — `LIVE_EDIT_EXT_RELEASE_MAX_BYTES`,
/// or, when the store reports no usable `maxmemory`, one shard's share of
/// `LIVE_EDIT_VALKEY_MEMORY_BYTES` (the store's total memory) ÷ 8, by the
/// store's cluster map (`storage::store_memory::memory_bytes_budget`, b.qhk).
pub(crate) const EXT_RELEASE_BUDGET_SOURCE_OVERRIDE: &str = "override";
/// `source` label value of `live_edit_ext_release_budget_bytes` (b.yuo): the
/// release budget is the smallest `maxmemory` the store's primaries report in
/// `INFO memory`, ÷ 8.
pub(crate) const EXT_RELEASE_BUDGET_SOURCE_INFO: &str = "info";
/// `source` label value of `live_edit_ext_release_budget_bytes` (b.yuo): the
/// release budget is the fallback, `min(8 × LIVE_EDIT_MAX_LAYER_SIZE, 100 MB)`
/// (b.1jt), because no budget is configured, the store reported no usable
/// `maxmemory`, and `LIVE_EDIT_VALKEY_MEMORY_BYTES` is unset or the store's
/// cluster map could not be read.
pub(crate) const EXT_RELEASE_BUDGET_SOURCE_FALLBACK: &str = "fallback";
/// Every `source` cell of `live_edit_ext_release_budget_bytes` (b.yuo).
pub(crate) const EXT_RELEASE_BUDGET_SOURCES: &[&str] = &[
    EXT_RELEASE_BUDGET_SOURCE_OVERRIDE,
    EXT_RELEASE_BUDGET_SOURCE_INFO,
    EXT_RELEASE_BUDGET_SOURCE_FALLBACK,
];

/// `site` label value for an undecodable release token returned by
/// `update_layer_fast.lua` (b.yda) — `LayerStore::enqueue_ext_releases_from_tokens`.
pub(crate) const EXT_REF_UNDECODABLE_SITE_LUA_TOKEN: &str = "lua_token";
/// `site` label value for an undecodable value-row `hash` column (b.yda) —
/// `LayerStore::read_ext_pre_image`, in its per-chunk row decode
/// (`decode_ext_pre_image_chunk`).
pub(crate) const EXT_REF_UNDECODABLE_SITE_ROW_HASH: &str = "row_hash";
/// `site` label value for an undecodable `le:{oid}:obj:ext` field read while
/// planning a commit's reference bookkeeping (b.yda): a field name that is not
/// a hash in the materializers' whole-index parse (`parse_ext_index`), or a
/// stored count that is not an integer in the per-field count decode
/// (`parse_ext_index_count`), which that parse and the `txn` commit's
/// touched-hash count reads (`LayerStore::read_ext_pre_image`) share.
pub(crate) const EXT_REF_UNDECODABLE_SITE_EXT_INDEX_FIELD: &str = "ext_index_field";
/// `site` label value for an undecodable `le:{oid}:obj:ext` field read while
/// scheduling a retiring object's releases (b.yda) —
/// `schedule_old_object_ext_gc`.
pub(crate) const EXT_REF_UNDECODABLE_SITE_RETIREMENT_INDEX_FIELD: &str = "retirement_index_field";
/// Every `site` cell of `ext_ref_undecodable_total`, zero-seeded at
/// construction (b.yda). Same discipline as [`EXT_REF_RELEASE_REASONS`].
pub(crate) const EXT_REF_UNDECODABLE_SITES: &[&str] = &[
    EXT_REF_UNDECODABLE_SITE_LUA_TOKEN,
    EXT_REF_UNDECODABLE_SITE_ROW_HASH,
    EXT_REF_UNDECODABLE_SITE_EXT_INDEX_FIELD,
    EXT_REF_UNDECODABLE_SITE_RETIREMENT_INDEX_FIELD,
];

/// `outcome`: a meta whose surplus — or **deficit** — this pass OFFERED to
/// `le:ext:{gc}:recon` as a candidate: an observation, nothing repaired and
/// nothing reported (b.yda).
///
/// **Offered, not newly registered.** The pass ticks this by the size of the
/// candidate set it built, and it rebuilds that set from scratch every pass, so
/// a condition that persists is re-offered — and re-ticked — pass after pass.
/// Only the `ZADD NX` inside `recon_claim_expired` distinguishes a first
/// observation from a re-offer, by keeping the original score, and it does so
/// Valkey-side where this counter cannot see it. Read the cell as a level (how
/// many metas are currently anomalous) rather than as a rate of arrival.
///
/// Both directions share the ZSET, the grace and this cell; the member carries
/// the direction in the sign of its amount. A first-observation deficit reaches
/// no gauge — it must survive a grace and confirm before it becomes
/// `deficit_observed`.
pub(crate) const EXT_RECONCILE_OUTCOME_CANDIDATE_REGISTERED: &str = "candidate_registered";
/// `outcome`: a claimed candidate whose surplus still matched a full grace
/// later, so repair members were emitted onto `le:ext:{gc}:pending` (b.yda).
/// Ticked once per repaired *hash*; the references handed back ride
/// `ext_ref_released_total{reason="reconcile"}`.
pub(crate) const EXT_RECONCILE_OUTCOME_REPAIRED: &str = "repaired";
/// `outcome`: a claimed candidate whose surplus no longer matched the amount
/// its member encoded — it healed on its own (a commit landed, a queued release
/// drained), so the member was retired and nothing was repaired (b.yda).
pub(crate) const EXT_RECONCILE_OUTCOME_HEALED: &str = "healed";
/// `outcome`: the pass's claim filled its batch (`RECONCILE_BATCH` members), so
/// more was claimable than one pass acts on and the remainder waits for the next
/// pass (b.yda). Ticked at most once per pass — it is a "the batch was the
/// binding constraint" flag, not a count of what was left behind.
///
/// The pass's wall-clock budget bounds the ENUMERATION phases, which forfeit the
/// whole pass when they overrun; it does not gate phase 5, so it never produces
/// this outcome. The ZSET is durable and every member keeps its
/// first-observation score, so deferring costs latency, never correctness.
pub(crate) const EXT_RECONCILE_OUTCOME_DEFERRED: &str = "deferred";
/// `outcome`: a meta with a positive surplus whose `last_store_ms` pre-flight
/// stamp is newer than the pass's own start minus the grace, so a `store_values`
/// increment may still be awaiting its commit (b.yda, b.rej). Not registered,
/// not repaired.
///
/// This is the pass's normal, healthy ineligibility cell: it climbing means the
/// store is busy and the pass is correctly declining to judge hashes that were
/// written to moments ago.
pub(crate) const EXT_RECONCILE_OUTCOME_SKIPPED_RECENT_STORE: &str = "skipped_recent_store";
/// `outcome`: a meta with a positive surplus whose `last_store_ms` field is
/// PRESENT but could not be parsed as a decimal-millisecond integer (b.yda,
/// b.rej). Not registered, not repaired.
///
/// **Absent is not unreadable, and the two get opposite verdicts.** Since
/// b.rej's writer stamps `last_store_ms` on every pre-flight store, an ABSENT
/// field is positive evidence that nothing has pre-stored the hash since the
/// writer deployed, so it is ELIGIBLE and never reaches this cell. What reaches
/// it is a field that exists and says something the pass cannot read — a
/// hand-edited key, a corrupt value, a build that wrote a different encoding.
/// Folding that into the absent case would read corrupt keyspace as "nothing
/// ever stored this" and license a repair on the strength of a value the pass
/// failed to understand; fail-closed doctrine forbids it, so an unreadable stamp
/// is ineligible and counted here.
///
/// A non-zero rate on this cell is a keyspace-integrity signal, not a
/// throughput one: nothing in the service writes an unparseable stamp
/// (`ext_store_value.lua` formats it with `string.format("%d", ms)`), so a
/// sustained count means something outside the service is writing the field.
pub(crate) const EXT_RECONCILE_OUTCOME_SKIPPED_UNREADABLE_STAMP: &str = "skipped_unreadable_stamp";
/// `outcome`: the pass aborted with NO repair at all because some enumeration
/// did not complete (b.yda). A partial object walk under-counts the referenced
/// side, which would read as surplus and over-release, so any error in the
/// object walk, the owed walk or the meta walk forfeits the whole pass.
pub(crate) const EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED: &str = "enumeration_failed";
/// `outcome`: a meta whose `ref_count` stands BELOW the number of live objects
/// referencing it, **confirmed** by the same two-observation protocol a surplus
/// goes through — the same deficit, at the same amount, still standing a full
/// grace after it was first registered (b.yda). Observed only: incrementing
/// would resurrect a meta whose data the GC may already have deleted.
///
/// **Alert on this cell RECURRING across several reconciler intervals, not on a
/// single tick.** A single pass's negative surplus is not evidence of an
/// over-release at all: the pass's own load-bearing phase order double-counts an
/// obligation that transitions mid-pass (a commit or a retirement dropping a
/// hash's last reference between the object walk and the pending walk yields
/// `-1` on a perfectly healthy store), and a retire carrier whose object is live
/// again is counted on both sides for one orphan-sweep interval. Confirmation
/// removes most of that noise — those shapes usually change or vanish within a
/// grace — but it confirms an AMOUNT, not a persistence: a hash that keeps
/// churning can present the same double-count at the same amount on the
/// confirming pass too. Per-object attribution would settle it; b.1sv tracks
/// that. Until then, recurrence is the signal — a real over-release does not
/// heal.
pub(crate) const EXT_RECONCILE_OUTCOME_DEFICIT_OBSERVED: &str = "deficit_observed";
/// `outcome`: a registered deficit candidate that did NOT survive its grace —
/// the fresh re-read no longer showed the same deficit, so it was one of the
/// benign transients above and the member was retired without alerting (b.yda).
///
/// The deficit-side counterpart of [`EXT_RECONCILE_OUTCOME_HEALED`], kept as its
/// own cell so an operator can see the two directions separately: a busy service
/// is expected to tick this routinely, and it is the denominator that makes an
/// occasional `deficit_observed` readable — a fleet ticking both is a churning
/// store, while `deficit_observed` recurring while this cell stays flat is the
/// shape that warrants a page.
pub(crate) const EXT_RECONCILE_OUTCOME_DEFICIT_CLEARED: &str = "deficit_cleared";
/// `outcome`: a meta with a positive surplus that the pass declined to register
/// or repair because, in this same pass, some live object's per-object
/// reference index (`le:{oid}:obj:ext`) disagreed with its rows for that hash
/// and the per-object recount could not repair it — the object was above the
/// recount's row ceiling, was not live when the recount ran, or carries a
/// `hash` column the recount reports as malformed (whatever it answered: the
/// decoder counts such a row and the recount does not, so its count is not
/// authoritative; it applies raises only and withholds every lowering).
///
/// Draining that surplus could be the second half of an over-release: when a
/// pre-index object's seeding commit under-counted the index, a later commit
/// can release a reference a live row still holds, and the surplus is exactly
/// the increment that would keep the blob alive meanwhile. So the surplus is
/// held — the hold itself leaks rather than loses — until the object is
/// repaired, shrinks below the ceiling or retires. The hold guards only
/// against a loss through a drain this build performs: a surplus a pre-fix or
/// old-build reconciler already drained is not there to hold, and drift from
/// other producers has no surplus behind it; both stay exposed to the commit
/// paths until repaired. Ticked once
/// per gated meta in the meta walk and once per gated claimed candidate.
/// Deficits are never gated.
pub(crate) const EXT_RECONCILE_OUTCOME_SKIPPED_INDEX_DRIFT: &str = "skipped_index_drift";
/// Every `outcome` cell of `ext_reconcile_total`, zero-seeded at construction
/// (b.yda). Same discipline as [`EXT_REF_RELEASE_REASONS`]: an alert on a rare
/// outcome needs its series to exist before the first occurrence.
pub(crate) const EXT_RECONCILE_OUTCOMES: &[&str] = &[
    EXT_RECONCILE_OUTCOME_CANDIDATE_REGISTERED,
    EXT_RECONCILE_OUTCOME_REPAIRED,
    EXT_RECONCILE_OUTCOME_HEALED,
    EXT_RECONCILE_OUTCOME_DEFERRED,
    EXT_RECONCILE_OUTCOME_SKIPPED_RECENT_STORE,
    EXT_RECONCILE_OUTCOME_SKIPPED_UNREADABLE_STAMP,
    EXT_RECONCILE_OUTCOME_ENUMERATION_FAILED,
    EXT_RECONCILE_OUTCOME_DEFICIT_OBSERVED,
    EXT_RECONCILE_OUTCOME_DEFICIT_CLEARED,
    EXT_RECONCILE_OUTCOME_SKIPPED_INDEX_DRIFT,
];

/// `site` label value of the per-object reference-index recount counters: the
/// reference-count reconciler's object walk, which recounts an object whose
/// `le:{oid}:obj:ext` disagrees with its rows (`storage::ext_reconcile`).
pub(crate) const EXT_INDEX_RECOUNT_SITE_RECONCILE: &str = "reconcile";
/// `site` label value of the per-object reference-index recount counters: the
/// txn commit path's torn-`EXEC` arm, which recounts the object whose commit
/// tore (`storage::layer`, `WatchedExec::ExecError`).
pub(crate) const EXT_INDEX_RECOUNT_SITE_TORN_COMMIT: &str = "torn_commit";
/// Every `site` cell of `ext_index_recount_total` and `ext_index_drift_total`,
/// zero-seeded at construction.
pub(crate) const EXT_INDEX_RECOUNT_SITES: &[&str] = &[
    EXT_INDEX_RECOUNT_SITE_RECONCILE,
    EXT_INDEX_RECOUNT_SITE_TORN_COMMIT,
];

/// `site` label value of `ext_index_drift_total` only: the structural commit's
/// `obj:ext` settle (`update_layer_fast.lua` structural mode, step 6b), which
/// finds a touched hash's stored count plus the commit's delta unequal to its
/// recount — the cache's untouched rows plus the touched rows ending with the
/// hash — and writes the raise-only floor. `under` counts the hashes it
/// raised; `over` the hashes it kept above the recount, for the audit to
/// lower. Not a recount site: it runs no recount and has no
/// `ext_index_recount_total` outcome, so it is not in
/// [`EXT_INDEX_RECOUNT_SITES`], whose cells are seeded for both counters.
pub(crate) const EXT_INDEX_DRIFT_SITE_STRUCTURAL_SETTLE: &str = "structural_settle";
/// The `site` cells of `ext_index_drift_total` that have no
/// `ext_index_recount_total` cells, zero-seeded for every direction.
pub(crate) const EXT_INDEX_DRIFT_ONLY_SITES: &[&str] = &[EXT_INDEX_DRIFT_SITE_STRUCTURAL_SETTLE];

/// `outcome`: the recount found drift and repaired the index atomically.
pub(crate) const EXT_INDEX_RECOUNT_OUTCOME_REPAIRED: &str = "repaired";
/// `outcome`: the recount found the index already equal to the rows (for
/// example a legacy dual-shape row, whose stale `hash` column triggers the
/// reconciler's raw comparison but is not a reference). Nothing written.
pub(crate) const EXT_INDEX_RECOUNT_OUTCOME_CLEAN: &str = "clean";
/// `outcome`: the object has more value rows than the recount reads in one
/// atomic call. Nothing written; at the reconcile site the object's
/// mismatched hashes are held from surplus drain (`skipped_index_drift`).
pub(crate) const EXT_INDEX_RECOUNT_OUTCOME_TOO_LARGE: &str = "too_large";
/// `outcome`: the object was not live when the recount ran (header gone or
/// partial, superseded, or its retirement releases scheduled). Nothing
/// written; at the reconcile site the object's mismatched hashes are held
/// from surplus drain (`skipped_index_drift`).
pub(crate) const EXT_INDEX_RECOUNT_OUTCOME_NOT_LIVE: &str = "not_live";
/// `outcome`: the recount call itself failed (transport error, a refused
/// write, a wrong-typed key). Nothing written. At the reconcile site this
/// forfeits the pass; at the torn-commit site the drift is left for the next
/// reconciler pass, and is exposed to the commit paths until then.
pub(crate) const EXT_INDEX_RECOUNT_OUTCOME_FAILED: &str = "failed";
/// `outcome`: the recount saw a value row whose `hash` column it cannot count
/// (not 64 lowercase hex — keyspace corruption), so its count is not
/// authoritative: the decoder may still count that row. Recorded for every
/// such recount, at both sites, whatever the script answered. Over such a row
/// the recount applies only raises and withholds every lowering, since
/// lowering on its count could under-count what the layer serves; any raises
/// still tick `ext_index_drift_total{direction="under"}`. This is NOT a
/// repair, whatever was raised. At the reconcile site the object's mismatched
/// hashes are held from surplus drain (`skipped_index_drift`) on every pass
/// until the malformed rows are corrected or the layer retires; the hold only
/// stops a drain, and an under-count against the decoder's count stays exposed
/// to the commit paths for as long. Needs an operator: the accompanying WARN
/// names the layer.
pub(crate) const EXT_INDEX_RECOUNT_OUTCOME_WITHHELD: &str = "withheld";
/// Every `outcome` cell of `ext_index_recount_total`, zero-seeded at
/// construction for every site.
pub(crate) const EXT_INDEX_RECOUNT_OUTCOMES: &[&str] = &[
    EXT_INDEX_RECOUNT_OUTCOME_REPAIRED,
    EXT_INDEX_RECOUNT_OUTCOME_CLEAN,
    EXT_INDEX_RECOUNT_OUTCOME_TOO_LARGE,
    EXT_INDEX_RECOUNT_OUTCOME_NOT_LIVE,
    EXT_INDEX_RECOUNT_OUTCOME_FAILED,
    EXT_INDEX_RECOUNT_OUTCOME_WITHHELD,
];

/// `direction`: an index field corrected UPWARD — the stored count (an absent
/// field reading as 0) was below the number of rows holding the hash. The
/// dangerous direction: left alone, a later commit can release a reference a
/// live row still holds.
pub(crate) const EXT_INDEX_DRIFT_DIRECTION_UNDER: &str = "under";
/// `direction`: an index field corrected DOWNWARD or removed — the stored
/// count was above the number of rows holding the hash. A leak until repaired.
pub(crate) const EXT_INDEX_DRIFT_DIRECTION_OVER: &str = "over";
/// Every `direction` cell of `ext_index_drift_total`, zero-seeded at
/// construction for every site.
pub(crate) const EXT_INDEX_DRIFT_DIRECTIONS: &[&str] = &[
    EXT_INDEX_DRIFT_DIRECTION_UNDER,
    EXT_INDEX_DRIFT_DIRECTION_OVER,
];

/// `outcome` of `time_sample_signed_zero_merge_total`, per row: a section had
/// both a `+0.0` and a `-0.0` time-sample row, and the merge left one row at
/// `+0.0`, holding the sample the readers already served
/// (`domain::types::resolve_signed_zero_pair`) — or a `+0.0` key that
/// `obj:tss` did not list, which readers never saw, so the merge moved the
/// `-0.0` row over it without consulting that rule.
pub const TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_MERGED: &str = "merged";
/// `outcome`, per row: a section had a `-0.0` row and no `+0.0` row, and the
/// merge moved it to `+0.0` unchanged.
pub const TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_REKEYED: &str = "rekeyed";
/// `outcome`, per row: a commit landed on the layer between the merge's read
/// and its write, so the merge wrote nothing; the next reconciler pass retries.
pub const TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_CHANGED: &str = "changed";
/// `outcome`, per row: a snapshot reader held the layer's fence, so the merge
/// wrote nothing; the next reconciler pass retries.
pub const TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_FENCED: &str = "fenced";
/// `outcome`, per row: the layer was not live (deleted, replaced, or being
/// retired) when the merge ran, so it wrote nothing.
pub const TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_NOT_LIVE: &str = "not_live";
/// `outcome`, per row: one row of the pair could not be decoded, so the merge
/// left the pair alone; readers refuse the same row.
pub const TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_UNDECODABLE: &str = "undecodable";
/// `outcome`, per row: a store command of the merge failed, so it wrote
/// nothing for these rows; the next reconciler pass retries.
pub const TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_FAILED: &str = "failed";
/// Every `outcome` cell of `time_sample_signed_zero_merge_total`, zero-seeded
/// at construction.
pub const TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOMES: &[&str] = &[
    TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_MERGED,
    TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_REKEYED,
    TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_CHANGED,
    TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_FENCED,
    TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_NOT_LIVE,
    TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_UNDECODABLE,
    TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOME_FAILED,
];

/// All gRPC status codes the metrics layer emits as `grpc_code` label values
/// (b.avd item 4). Sourced from `tonic::Code` — keeping the list local lets us
/// pre-init every label permutation at startup so first-scrape misses do not
/// hide a missing endpoint behind an unfamiliar code. The empty-string code
/// from `tonic::Code::Ok` collides with successful-trailer omissions in some
/// transports, so we use the textual `"OK"` slug instead.
pub(crate) const GRPC_CODES: &[&str] = &[
    "OK",
    "CANCELLED",
    "UNKNOWN",
    "INVALID_ARGUMENT",
    "DEADLINE_EXCEEDED",
    "NOT_FOUND",
    "ALREADY_EXISTS",
    "PERMISSION_DENIED",
    "RESOURCE_EXHAUSTED",
    "FAILED_PRECONDITION",
    "ABORTED",
    "OUT_OF_RANGE",
    "UNIMPLEMENTED",
    "INTERNAL",
    "UNAVAILABLE",
    "DATA_LOSS",
    "UNAUTHENTICATED",
];

/// Resolve the OTLP metrics push endpoint. Gated ONLY on the per-signal
/// `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`; there is deliberately no fallback to
/// the generic `OTEL_EXPORTER_OTLP_ENDPOINT`. The standardized Helm schema
/// exposes independent `observability.tracing` and `observability.metrics`
/// blocks: tracing sets the generic endpoint, metrics sets the per-signal one.
/// A fallback would let "tracing enabled, metrics disabled" silently start
/// pushing metrics — so the metrics block stays the sole control. Empty / unset
/// → `Ok(None)` (no OTLP reader attached; the service keeps exporting via the
/// `/metrics` scrape only). A value that is not an `http://` or `https://` URL
/// with a host is an `Err` naming the variable ([`otlp::otlp_endpoint_from`]).
fn otlp_metrics_endpoint() -> Result<Option<otlp::OtlpEndpoint>, String> {
    otlp::otlp_endpoint_from_env(otlp::OTEL_EXPORTER_OTLP_METRICS_ENDPOINT)
}

/// View that pins each histogram's explicit Prometheus bucket boundaries so the
/// `/metrics` output matches the pre-migration `prometheus`-crate buckets
/// exactly. Instruments not named here fall through to the default view.
fn histogram_bucket_view(inst: &Instrument) -> Option<Stream> {
    let boundaries: Vec<f64> = match inst.name() {
        "live_endpoint_duration_microseconds" => DURATION_BUCKETS.to_vec(),
        "live_diff_build_duration_seconds" | "live_subscribe_catch_up_duration_seconds" => {
            DIFF_DURATION_BUCKETS.to_vec()
        }
        "live_subscribe_catch_up_delivered_bytes" => {
            SUBSCRIBE_CATCH_UP_DELIVERED_BYTES_BUCKETS.to_vec()
        }
        "live_edit_join_latency_seconds" => JOIN_LATENCY_BUCKETS.to_vec(),
        "live_edit_presence_sweep_duration_seconds" => PRESENCE_SWEEP_DURATION_BUCKETS.to_vec(),
        "live_edit_presence_stale_removal_delay_seconds" => {
            PRESENCE_STALE_REMOVAL_DELAY_BUCKETS.to_vec()
        }
        "live_edit_auth_call_duration_seconds" => AUTH_CALL_DURATION_BUCKETS.to_vec(),
        "live_edit_update_layer_attempts_histogram" => UPDATE_LAYER_ATTEMPTS_BUCKETS.to_vec(),
        "live_edit_update_layer_relay_bytes" => UPDATE_LAYER_RELAY_BYTES_BUCKETS.to_vec(),
        "live_edit_upload_object_state_bytes" => UPLOAD_OBJECT_STATE_BYTES_BUCKETS.to_vec(),
        "live_edit_update_layer_commit_duration_seconds" => {
            UPDATE_LAYER_COMMIT_DURATION_BUCKETS.to_vec()
        }
        "live_edit_update_layer_structural_commit_seconds" => {
            UPDATE_LAYER_STRUCTURAL_COMMIT_BUCKETS.to_vec()
        }
        "live_edit_checkpoint_pending_age_seconds" => CHECKPOINT_PENDING_AGE_BUCKETS.to_vec(),
        "live_edit_checkpoint_counter_seed_secs" => COUNTER_SEED_BUCKETS.to_vec(),
        "live_edit_valkey_pool_checkout_seconds" => POOL_CHECKOUT_BUCKETS.to_vec(),
        "live_edit_subscriber_admission_wait_seconds"
        | "live_edit_upload_admission_wait_seconds"
        | "live_edit_upload_processing_wait_seconds"
        | "live_edit_subscribe_catchup_wait_seconds"
        | "live_edit_catchup_snapshot_read_wait_seconds"
        | "live_edit_catchup_build_wait_seconds"
        | "live_edit_retirement_wait_seconds" => ADMISSION_WAIT_BUCKETS.to_vec(),
        "live_edit_stream_mux_xread_reply_bytes" => STREAM_MUX_XREAD_REPLY_BYTES_BUCKETS.to_vec(),
        // A cold start is one or two snapshot reads, plus any wait for the
        // cold-start permits, so it shares the read's boundaries: 50 ms at
        // `le=0.05`, and seconds above it for permit queues and fenced reads.
        "live_edit_snapshot_read_duration_seconds"
        | "live_edit_structural_cold_start_read_duration_seconds"
        | "live_edit_update_layer_fence_wait_seconds"
        | "live_edit_snapshot_fence_hold_seconds" => SNAPSHOT_READ_DURATION_BUCKETS.to_vec(),
        // A chain-gate wait or detection lag: `le=0.05` is the gate's added
        // wait target (50 ms p99), and seconds above it for waits that run to
        // a deadline.
        "live_edit_update_layer_chain_wait_seconds"
        | "live_edit_update_layer_chain_detect_lag_seconds" => {
            SNAPSHOT_READ_DURATION_BUCKETS.to_vec()
        }
        "live_edit_snapshot_read_reply_bytes" => SNAPSHOT_READ_REPLY_BYTES_BUCKETS.to_vec(),
        "live_edit_snapshot_read_entries" => SNAPSHOT_READ_ENTRIES_BUCKETS.to_vec(),
        "live_edit_snapshot_fence_reader_wait_seconds" => {
            SNAPSHOT_FENCE_READER_WAIT_BUCKETS.to_vec()
        }
        "live_edit_structural_cache_replayed_entries" => {
            STRUCTURAL_CACHE_REPLAYED_ENTRIES_BUCKETS.to_vec()
        }
        "live_edit_structural_cache_repair_bytes" => STRUCTURAL_CACHE_REPAIR_BYTES_BUCKETS.to_vec(),
        "live_edit_subscriber_delivery_lag_seconds" => SUBSCRIBER_DELIVERY_LAG_BUCKETS.to_vec(),
        "live_edit_subscriber_send_wait_seconds" => SUBSCRIBER_SEND_WAIT_BUCKETS.to_vec(),
        "live_edit_subscriber_backlog_entries" => SUBSCRIBER_BACKLOG_ENTRIES_BUCKETS.to_vec(),
        "live_edit_subscriber_backlog_bytes" => SUBSCRIBER_BACKLOG_BYTES_BUCKETS.to_vec(),
        _ => return None,
    };
    Stream::builder()
        .with_aggregation(Aggregation::ExplicitBucketHistogram {
            boundaries,
            // Prometheus histograms carry no min/max; recording them for
            // cumulative data has no value (SDK docs), so suppress them.
            record_min_max: false,
        })
        .build()
        .ok()
}

/// What [`build_meter_provider`] builds: the provider, the `Meter` every
/// instrument is created from, the scrape exporter, and the push exporter's
/// failure reporter when the OTLP push arm is attached.
type MeterPipeline = (
    SdkMeterProvider,
    Meter,
    PrometheusExporter,
    Option<Arc<MetricExportReporter>>,
);

/// Build the dual-reader `SdkMeterProvider` (Prometheus scrape always, OTLP
/// push when an endpoint is configured), and return it together with the
/// `Meter` used to create every instrument, the `PrometheusExporter` the
/// `/metrics` endpoint renders from, and the push arm's failure reporter. The
/// exporter is both the always-attached scrape `MetricReader` and the
/// text-exposition server (`export()`), so it is handed back for the metrics
/// server (`metrics_server::serve_metrics_with_scaling`) to hold.
///
/// An invalid `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`, or a push exporter that
/// cannot be built for it, is an `Err` naming the variable and the cause. A
/// set compression variable is an `Err` naming that variable
/// ([`otlp`]).
fn build_meter_provider() -> Result<MeterPipeline, String> {
    // Prometheus scrape arm — always attached. The `without_*` knobs keep the
    // emitted wire names byte-identical to the pre-migration output.
    let prometheus_exporter = PrometheusExporter::builder()
        .without_counter_suffixes()
        .without_units()
        .without_scope_info()
        .without_target_info()
        .build();

    let mut builder = SdkMeterProvider::builder()
        .with_reader(prometheus_exporter.clone())
        .with_view(histogram_bucket_view);

    // OTLP push arm — attached ONLY when an endpoint is configured (mirrors the
    // tracing side). Endpoint absent → scrape-only, backward compatible. Each
    // export, the final flush on shutdown included, is capped at
    // `otlp::OTLP_EXPORT_TIMEOUT`, and its failures reach the log through
    // `ReportingMetricExporter` (the SDK logs them only at DEBUG).
    let mut export_reporter = None;
    if let Some(endpoint) = otlp_metrics_endpoint()? {
        let exporter = otlp::metric_exporter_for(&endpoint)?;
        let reporter = Arc::new(MetricExportReporter::new());
        let reader = PeriodicReader::builder(ReportingMetricExporter::new(
            exporter,
            Arc::clone(&reporter),
        ))
        .build();
        builder = builder.with_reader(reader);
        export_reporter = Some(reporter);
        // Deliberately not the endpoint, as on the trace side: the URL may
        // carry credentials.
        tracing::info!(
            transport = endpoint.transport.as_str(),
            "OTLP metrics push exporter enabled"
        );
    }

    let provider = builder.build();
    let meter = provider.meter("live-edit");
    Ok((provider, meter, prometheus_exporter, export_reporter))
}

/// OpenTelemetry metric instruments for the Live Edit service.
///
/// Construct once at startup via [`Metrics::try_new`] (tests, benches and
/// examples use [`Metrics::new`]) and share via [`Arc<Metrics>`].
/// Every instrument is created from one `Meter` backed by the dual-reader
/// `SdkMeterProvider` this struct owns (see module docs). The `_provider` and
/// `meter` handles are retained so observable gauges can be registered after
/// construction (see [`Metrics::register_storage_watch_unwatch_inflight_gauge`])
/// and the OTLP push reader flushed at shutdown ([`Metrics::shutdown`]).
pub struct Metrics {
    /// The dual-reader provider. Held so its OTLP reader can be flushed and
    /// shut down at process exit; dropped without shutdown in tests (the
    /// Prometheus reader has no background thread).
    _provider: SdkMeterProvider,
    /// The OTLP push exporter's failure reporter (`Some` only when the push
    /// arm is attached). The provider's reader owns the exporter; this handle
    /// lets tests read the failure total.
    #[cfg_attr(not(any(test, feature = "test-support")), allow(dead_code))]
    otlp_export_reporter: Option<Arc<MetricExportReporter>>,
    /// The always-attached Prometheus scrape reader. Cloned out via
    /// [`Metrics::prometheus_exporter`] and handed to
    /// `metrics_server::serve_metrics_with_scaling`, which renders `/metrics`
    /// by calling `PrometheusExporter::export`. Cloning shares the same underlying reader
    /// (it is an `Arc`-backed handle), so the served output always reflects the
    /// live pipeline.
    prometheus_exporter: PrometheusExporter,
    /// The meter every instrument was created from. Retained so observable
    /// gauges (e.g. the WatchGuard UNWATCH inflight gauge) can be registered
    /// once the runtime source they observe is available.
    meter: Meter,
    /// The `service` label built once from `Metrics::new`'s `service_label`,
    /// for the per-delivery counters: cloning it bumps an `Arc` refcount
    /// rather than allocating a `String` per recording.
    service_kv: KeyValue,

    /// Total number of endpoint calls, by endpoint name, service label, and
    /// `grpc_code` (b.avd item 4).
    pub endpoint_calls: Counter<u64>,
    /// Endpoint call duration in microseconds, by endpoint name, service
    /// label, and `grpc_code` (b.avd item 4).
    pub endpoint_duration_us: Histogram<f64>,
    /// Outstanding (in-flight) gRPC RPCs, by endpoint and service (b.avd item
    /// 5). A non-monotonic sum, exported to Prometheus as a gauge.
    pub inflight_rpcs: UpDownCounter<i64>,
    /// Steady-state commits delivered to `SubscribeReadLayer` subscribers, by
    /// `endpoint` (always `SubscribeReadLayer`) and `service`. Counts once per
    /// commit per subscriber after every chunk of that commit was enqueued to the
    /// subscriber's stream — the relayed writer OMPF, or, for an empty commit
    /// from a client (an entry without relay bytes), the empty delta at its
    /// sequence, built with no read, which is delivered and counted. Never
    /// counts catch-up content (including a steady-state re-bootstrap's full
    /// state), self-echo, duplicate or object-id-mismatch drops, or a send
    /// that did not complete.
    ///
    /// Recorded in per-subscription batches
    /// (`grpc::subscriber_delivery::DeliveryTally`): a subscription's
    /// deliveries reach the counter normally within about a second of the
    /// delivery, and all of them by the time its stream ends; a delivery
    /// waiting behind a blocked send reaches the counter once the
    /// subscription's loop next goes idle.
    pub subscriber_updates: Counter<u64>,
    /// Total number of diff computations, by case (full_content/incremental/up_to_date).
    pub diff_case_total: Counter<u64>,
    /// Duration of diff build operations in seconds.
    pub diff_build_duration_seconds: Histogram<f64>,
    /// Total number of `SubscribeReadLayer` catch-up phases, by regime and
    /// service (b.qxe). `regime` ∈ {`brand_new`, `in_window`, `out_of_window`,
    /// `up_to_date`}.
    ///
    /// The label records WHY the subscriber needed catch-up, not which code
    /// path served it: under the shipped default
    /// `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=true` the `in_window` and
    /// `out_of_window` regimes CONVERGE on the same coalesced reconstruction
    /// and differ only in this label — the in-window-fast / out-of-window-slow
    /// split survives only under `=false`. What the label still buys is the
    /// distinction between a true bootstrap (`brand_new`) and a reconnect, and
    /// between a reconnector whose gap was still on the stream (`in_window`)
    /// and one whose gap had been trimmed away (`out_of_window`) — all of which
    /// collapse into the coarse `live_diff_case_total{case}` vocabulary.
    pub subscribe_regime_total: Counter<u64>,
    /// Duration of the `SubscribeReadLayer` catch-up phase in seconds, by
    /// `regime` and `service` (b.qxe). Also carries the additive `catchup_mode`
    /// attribute (`snapshot` | `legacy`) as of Plan b.msb. The clock starts when
    /// the spawned subscription task enters catch-up — after auth, session
    /// lookup and admission — so it excludes the admission wait;
    /// [`Self::join_latency_seconds`] covers the whole join.
    pub subscribe_catch_up_duration_seconds: Histogram<f64>,
    /// Time from handler entry to the first delivered state, in seconds, by
    /// `endpoint` (`SubscribeReadLayer` | `JoinSession`) and `service`. Observed
    /// once per RPC that gets there, never on failure:
    ///
    /// - `SubscribeReadLayer`: when the first caught-up ack has been enqueued
    ///   (auth + session lookup + admission wait + catch-up). A steady-state
    ///   re-bootstrap's ack does not observe again.
    /// - `JoinSession`: when `SessionConnected` and the membership replay have
    ///   been enqueued. Includes the wait for the client's header message.
    pub join_latency_seconds: Histogram<f64>,
    /// OMPF bytes delivered during the `SubscribeReadLayer` catch-up phase, by
    /// `regime`, `service`, and `catchup_mode` (Plan b.msb). Byte-scale buckets.
    pub subscribe_catch_up_delivered_bytes: Histogram<f64>,
    /// Outcome of a catch-up's materialized-state read, by `outcome`
    /// ([`SUBSCRIBE_STABLE_CUT_OUTCOMES`]), `regime`
    /// ([`SUBSCRIBE_STABLE_CUT_REGIMES`], the vocabulary of
    /// `subscribe_regime_total`), and `service` (Spec SDD SR-8.6). Incremented
    /// once per catch-up state read that is served or refused; seeded at zero.
    /// A rollback-mode in-window replay's per-commit reads are the exception:
    /// the replay adds to this counter only when a read regresses
    /// (`unverified`, `in_window`), so a replay that succeeds adds +1 to
    /// `subscribe_regime_total{regime="in_window"}` and nothing here. This
    /// counter is therefore not 1:1 with `subscribe_regime_total`; nor is it
    /// for a catch-up that ends on a sequence regression (`unverified`), which
    /// counts here with no regime observation.
    ///
    /// `verified` means the catch-up served a proven cut: at or below the
    /// atomic-read limits, its atomic read's own; above them, the cut of the
    /// shared fenced read of the flight it was attached to
    /// (`storage::snapshot_flights`), once per attached catch-up. Either way
    /// the subscriber's stream starts at the cut and its ack lands there, so
    /// nothing overlaps.
    /// `unverified` means a sequence regression and nothing else: the read's
    /// cut came back below the header sequence the catch-up first read (an
    /// atomic read's, or, per attached catch-up, its flight's shared read's),
    /// or, for a rollback-mode in-window replay's per-commit read, below the
    /// commit it rebuilds — which only a restore with an undersized sequence
    /// band causes — and the catch-up ended `UNAVAILABLE` before any content
    /// or ack. Alert on any increase.
    pub subscribe_stable_cut_total: Counter<u64>,
    /// Live presence events delivered to local `JoinSession` streams, by
    /// `event_type` (`joined` | `left` | `message`) and `service`. Counts once per
    /// event per local recipient, after the event was enqueued to that
    /// recipient's stream. Never counts an event the recipient itself
    /// originated, the membership replay a new joiner receives, a live
    /// `joined` suppressed as a duplicate of that replay (never sent), a failed
    /// send, or events dropped by a lagging receiver (those count on
    /// `presence_broadcast_dropped`, also per recipient).
    ///
    /// Recorded in per-stream batches (`grpc::presence`'s event tally): a
    /// stream's deliveries reach the counter normally within about a second,
    /// and all of them by the time the stream ends.
    pub presence_events_total: Counter<u64>,
    /// Total presence events a local `JoinSession` stream lost because it fell
    /// more than the per-session fan-out capacity
    /// (`LIVE_EDIT_PRESENCE_BROADCAST_CAPACITY`) behind. Incremented by the
    /// number of events the stream missed (`PresenceRecvError::Lagged`'s
    /// `dropped`), not by 1 per lag, whether the stream then continues (only
    /// peer messages were lost) or is ended for a resync (which also adds one
    /// to `presence_resyncs`).
    pub presence_broadcast_dropped: Counter<u64>,
    /// `live_edit_presence_resyncs_total{service}`: local `JoinSession`
    /// streams ended `UNAVAILABLE` because they fell more than the per-session
    /// fan-out capacity behind and missed a participant joining or leaving.
    /// One increment per stream ended; the client's rejoin restores its
    /// participant list, and peers see the connection leave and the rejoin
    /// join. The events the stream missed count on
    /// `presence_broadcast_dropped`. Losses upstream of the per-session queue
    /// (`pubsub_relay_lagged`) never end a stream and are not counted here.
    pub presence_resyncs: Counter<u64>,
    /// `live_edit_presence_keepalive_timeouts_total{service}`: local
    /// `JoinSession` streams ended `UNAVAILABLE CLIENT_KEEPALIVE_TIMEOUT`
    /// because the client opted in to client keepalives and then sent no data
    /// for three keepalive intervals (`LIVE_EDIT_PRESENCE_KEEPALIVE_INTERVAL_MS`).
    /// One increment per stream ended; the participant left (peers saw one
    /// `PeerLeft`) before the stream ended. The ending is logged at INFO, so
    /// this counter is where its rate shows. A client keepalive on the
    /// `JoinSession` stream, unrelated to `stream_mux_keepalives` (the
    /// Valkey stream readers' own keepalives).
    pub presence_keepalive_timeouts: Counter<u64>,
    /// `live_edit_pubsub_relay_lagged_total{relay, service}`: pub/sub messages
    /// a replica-wide relay lost because it fell behind the pub/sub client's
    /// message broadcast channel (`storage::client::PUBSUB_BROADCAST_CHANNEL_CAPACITY`
    /// slots), by `relay` ([`PUBSUB_RELAYS`]). Incremented by the number of
    /// skipped messages, not by 1 per lag: once per drain batch, by the sum of
    /// the batch's `Lagged(n)` counts, so the total is the same as one add per
    /// lag.
    ///
    /// One per replica, upstream of every local recipient: a lost presence
    /// event is missing for every local `JoinSession` stream in its session,
    /// which can leave a client never showing a peer that is present (or
    /// still showing one that left). `presence_broadcast_dropped` counts the
    /// other, per-recipient stage, and does not move here. The relay also logs
    /// one WARN per drain batch that lagged. Every message on the replica's
    /// pub/sub connection reaches every relay, so each relay counts its own
    /// lag, including messages on channels it would have ignored: a
    /// `session_deleted` lag means deletion notices *may* have been missed, and
    /// that relay then requests a prompt re-check of the replica's sessions
    /// (`grpc::session::SessionRevalidation`; re-checks run at least
    /// `grpc::session::REVALIDATION_MIN_GAP` apart).
    pub pubsub_relay_lagged: Counter<u64>,
    /// `live_edit_presence_lease_refresh_failures_total{service}`: failed
    /// steps of a `JoinSession` connection's lease keeper. A step fails when a
    /// store command errors (a timeout included; the timeout is also counted
    /// once on `valkey_errors_total`): the refresh, the store-clock read or
    /// register of a self-rejoin, or the session-index re-add after a renewal
    /// or rejoin. It also fails when a rejoin register reached the store past
    /// its deadline (`Expired`), having written nothing. One increment per
    /// failed keeper step; the keeper logs a WARN and tries again at its next
    /// tick. A sustained rate means leases are not being renewed, or evicted
    /// connections not re-registered, so live connections risk being swept
    /// or staying hidden from their peers.
    pub presence_lease_refresh_failures: Counter<u64>,
    /// `live_edit_presence_rejoins_total{service}`: self-rejoins by
    /// `JoinSession` lease keepers. Counted once when a keeper whose refresh
    /// found its still-open connection evicted re-registers it and the
    /// register re-adds the membership (`Registered { added: true }`). That is
    /// exactly when the register script announced `joined` to peers again,
    /// after the eviction's `left`. Not counted: a register that only repaired
    /// a current member's missing info hash (announced nothing), or any
    /// registration by a join. A rejoin whose re-adding register timed out
    /// but still applied is not counted either, since the keeper never saw its
    /// `added: true`; the counter can undercount rejoins, never overcount
    /// them. A non-zero rate means live connections were
    /// evicted, typically by a store outage or pause longer than the lease TTL.
    pub presence_rejoins: Counter<u64>,
    /// `live_edit_presence_sweep_duration_seconds{outcome, service}`: wall
    /// time of one presence-sweeper leader pass, from just before the leader
    /// probe that won the lock to the pass's end, by how it ended
    /// ([`PRESENCE_SWEEP_OUTCOMES`]). Only the replica that won the lock
    /// records a pass; a replica that lost the probe, or whose probe failed,
    /// records nothing. Buckets 0.01–300 s.
    pub presence_sweep_duration_seconds: Histogram<f64>,
    /// `live_edit_presence_sweep_lease_total{outcome, service}`: the presence
    /// sweeper's leader-lease decisions before a chunk, by
    /// [`PRESENCE_SWEEP_LEASE_OUTCOMES`] — the same meanings as
    /// `live_edit_checkpoint_lease_total`. A check made inside the renewal
    /// window decides nothing and counts nothing, so this measures refreshes,
    /// not chunks.
    ///
    /// - `renewed` — a due refresh extended the lock; the steady-state cell
    ///   of any pass longer than a third of the lock TTL.
    /// - `lost` — the lease could not be proven held (its local deadline had
    ///   passed, the lock lapsed or another process holds it, or the refresh
    ///   errored), so the pass ended before its next script call and logged
    ///   one WARN. A pass renews every third of the TTL, so only a store
    ///   outage or a pause longer than two thirds of the TTL loses it.
    /// - `refresh_error` — the refresh round-trip failed. Additive to `lost`:
    ///   an errored refresh counts both, so `lost` alone is the number of
    ///   passes a lease loss ended.
    pub presence_sweep_lease_total: Counter<u64>,
    /// `live_edit_presence_stale_removed_total{service}`: connections the
    /// presence sweeper removed because their lease had expired — each one
    /// removed, and its `left` published, by one sweep script execution on
    /// this replica. A removal whose script reply was lost (and re-sent by the
    /// client) is not reported back, so it is not counted: the counter can
    /// undercount removals, never overcount them.
    pub presence_stale_removed: Counter<u64>,
    /// `live_edit_presence_stale_removal_delay_seconds{service}`: for each
    /// counted removal whose info hash recorded when its lease expires
    /// (`lease_expires_at_ms`), how long after that expiry the sweep removed
    /// it, both read from the store's clock. The measured stale-removal time:
    /// expect it within one sweep interval plus one pass. Buckets 1–600 s.
    pub presence_stale_removal_delay_seconds: Histogram<f64>,
    /// `live_edit_presence_sweep_oldest_stale_age_seconds{service}`: the
    /// largest removal delay (see
    /// [`Self::presence_stale_removal_delay_seconds`]) in this replica's last
    /// completed leader pass; 0 after a completed pass that removed nothing
    /// with a known delay. A replica that lost the leader probe records 0, so
    /// a demoted leader stops exporting its last maximum; a pass that did not
    /// complete, or a probe that failed, records nothing (unknown is not
    /// zero). Absent until the replica's first tick that publishes — a lost
    /// leader probe (0) or a completed pass. Query it as `max(...)`
    /// across replicas, never `sum(...)` — the same posture as
    /// [`Self::checkpoint_oldest_pending_age_seconds`].
    pub presence_sweep_oldest_stale_age_seconds: Gauge<f64>,
    /// Auth cache hits, by cache kind (`permissions` | `userinfo`) (b.nrt).
    pub auth_cache_hits: Counter<u64>,
    /// Auth cache misses, by cache kind (`permissions` | `userinfo`) (b.nrt).
    pub auth_cache_misses: Counter<u64>,
    /// Auth call duration histogram for the underlying backend, by `backend`
    /// and `protocol` (b.nrt).
    pub auth_call_duration_seconds: Histogram<f64>,
    /// Auth backend errors, by `backend` and `code` (b.nrt).
    pub auth_errors: Counter<u64>,
    /// `live_edit_userinfo_fallbacks_total`, unlabelled: userinfo lookups the
    /// userinfo service answered "no such user" for, resolved instead from the
    /// bearer's JWT claims by [`crate::auth::UserinfoFallbackClient`]. Ticks
    /// once per fallback, which is once per userinfo cache miss for that
    /// bearer; stays 0 unless the userinfo service and authorization are both
    /// on. Seeded at 0 in every configuration.
    pub userinfo_fallbacks_total: Counter<u64>,
    /// Total `update_layer` retry attempts taken, by commit `path`
    /// ([`UPDATE_LAYER_PATHS`]) and eventual `outcome`
    /// ([`UPDATE_LAYER_RETRY_OUTCOMES`]) (b.vi2; `path` per Spec SDD SR-2.1).
    ///
    /// Counts retry attempts, not calls: a commit on attempt N adds N−1, an
    /// exhaustion adds `max_attempts − 1`, and a first-try success adds
    /// nothing. All of a call's retries land on the path of its committing (or
    /// last) attempt. The number of commits per path that needed a retry is
    /// `sum by (path) (rate(live_edit_update_layer_attempts_histogram_count))`
    /// minus
    /// `sum by (path) (rate(live_edit_update_layer_attempts_histogram_bucket{le="1"}))`.
    pub update_layer_retries_total: Counter<u64>,
    /// Total `update_layer` attempts that found their own call's commit
    /// already landed and reported it instead of applying the delta again, by
    /// `backend` ([`UPDATE_LAYER_BACKENDS`]).
    ///
    /// Every `update_layer` call carries one random commit token that its
    /// commit records in `le:{oid}:obj:wtok`, and every attempt checks that
    /// record first. A match means an earlier attempt of the same call
    /// committed but its reply never arrived — a transient error after the
    /// `EVALSHA` ran, an `EXEC` reply lost and not reconciled, or fred
    /// replaying a script that had already run. Each tick is one duplicate
    /// commit the in-process retry loop did NOT make. Non-zero rates track
    /// backing-store reply loss; they are not themselves a fault.
    pub update_layer_commit_already_landed_total: Counter<u64>,
    /// Total lost-reply release lookups, by `outcome`
    /// ([`LOST_REPLY_RELEASE_LOOKUP_OUTCOMES`]).
    ///
    /// An `UpdateLayer` call that ends without proof of its outcome — retry
    /// exhaustion, or, after a counted retry, an overtaken or
    /// `PREDECESSOR_NOT_COMMITTED` refusal or a refusal answered `UNAVAILABLE`
    /// outcome unknown — looks up its own commit's external-value releases
    /// by its commit token, once, and enqueues them if a commit of it landed
    /// and owed any. `enqueued` counts lookups that found one: each is a lost
    /// reply whose releases would otherwise have waited for the
    /// reference-count reconciler. `absent` is the common case (no commit of
    /// the call landed, or it owed nothing). `failed` counts lookups that
    /// could not read or parse the entry, each also logged at WARN; their
    /// releases are left to the reconciler.
    pub update_layer_lost_reply_release_lookup_total: Counter<u64>,
    /// Total `UpdateLayer` / `CreateLayer` calls whose client `operation_id`
    /// matched an operation an EARLIER call had already committed, by `rpc`
    /// ([`OPERATION_ID_RPCS`]) and `outcome` ([`OPERATION_ID_OUTCOMES`]).
    ///
    /// `already_committed` counts client re-sends answered from the recorded
    /// commit instead of being applied again — each tick is one duplicate
    /// commit (or one replaced layer) a client retry did not cause.
    /// `conflict` counts calls refused `ALREADY_EXISTS` because the id was
    /// recorded for a different request: a client reusing an id across
    /// intents, which is a client bug. A match by an attempt of the SAME call
    /// is not counted here; that is
    /// [`Self::update_layer_commit_already_landed_total`].
    pub operation_id_match_total: Counter<u64>,
    /// Total operation-id `UpdateLayer` calls ended because an attempt found
    /// the call's start record no longer matched `le:{oid}:obj:wtok` — the
    /// same `client_id` had committed again on the object since the call
    /// started, or a checkpoint restore had rebuilt the object without the
    /// record — by `backend` ([`UPDATE_LAYER_BACKENDS`]). That attempt applied
    /// nothing and the call answered `UNAVAILABLE` (an earlier attempt of it
    /// may have committed with its reply lost). When another commit of the
    /// `client_id` moved the record, a tick is either a late attempt of a call
    /// the client gave up on (a retry, or a commit the store ran late) that
    /// would otherwise have applied after the client's next write, or that
    /// next call itself, refused because the given-up call's late commit landed
    /// while it was in flight; with writers sharing a `client_id`, another
    /// writer's commit. After a restore that wiped the record, it is an attempt
    /// refused because the call's start record can no longer be checked, not a
    /// late one. Unchained calls only: a chained call has no start record and
    /// is never overtaken — its commit refuses it `PREDECESSOR_NOT_COMMITTED`
    /// instead, which ticks nothing here.
    pub update_layer_operation_overtaken_total: Counter<u64>,
    /// Histogram of attempts per landed `update_layer` commit (1 = first-try
    /// commit), by the committing attempt's `path` ([`UPDATE_LAYER_PATHS`]);
    /// buckets at 1/2/4/8/16/32/64/128 (b.vi2, b.avd item 6; `path` per Spec
    /// SDD SR-2.1). Not pre-registered: a `path` series appears at its first
    /// commit. Re-runs after a snapshot-fence wait are not attempts and do not
    /// count here.
    ///
    /// Observed exactly once per commit whose landing the call observes, by
    /// the attempt that observes it, before the call's post-commit work and
    /// its return (Spec SDD SR-2.6) — so a call whose future is dropped after
    /// that observation (an RPC deadline, a client cancel) is still counted.
    /// That call's [`Self::update_layer_commit_duration_seconds`] sample is
    /// `outcome="error"`, so that histogram's `outcome="ok"` count can read
    /// below this one's count by the number of such calls. It can also read
    /// above it: a call answered `ALREADY_COMMITTED` from an earlier call's
    /// commit of its client `operation_id` returns `Ok` — an `outcome="ok"`
    /// sample there — but made no commit, so it observes nothing here.
    ///
    /// A call dropped before it sees the landing leaves a commit that landed
    /// counted and logged nowhere: one dropped while awaiting the commit's own
    /// `EVALSHA` / `EXEC` reply (the commit can land with no attempt seeing
    /// it), and one dropped during either of the two routes that see the
    /// landing only after further reads — an unwatched commit is accepted
    /// after the retirement re-reads, and an `EXEC` whose reply was lost is
    /// confirmed after the reconciling `XRANGE`.
    pub update_layer_attempts_histogram: Histogram<f64>,
    /// Histogram of `update_layer` wall time, from entry to return, by `path`
    /// ([`UPDATE_LAYER_PATHS`]) and `outcome` ([`UPDATE_LAYER_COMMIT_OUTCOMES`])
    /// (Spec SDD SR-2.2).
    ///
    /// Observed exactly once per call. The time includes the wait for the
    /// per-object admission permit, every attempt and every backoff sleep.
    /// `path` is that of the call's last attempt, or, for a call that ended
    /// before its first attempt (delta validation, a failed start-record read
    /// for a call with a client `operation_id` — both `outcome="error"` — or a
    /// drop while queued for admission), the path the call classified to.
    /// `outcome="ok"` covers every call that returns `Ok`, an earlier call's
    /// commit answered `ALREADY_COMMITTED` included, which made no commit of
    /// its own and records no
    /// [`Self::update_layer_attempts_histogram`] sample. A call whose future is
    /// dropped before it returns — its RPC deadline elapsed, its client went
    /// away — is included, as `outcome="error"`, timed to the drop, even when
    /// its commit's landing had already been observed and counted in
    /// [`Self::update_layer_attempts_histogram`]. Not pre-registered: a series
    /// appears at its first observation.
    pub update_layer_commit_duration_seconds: Histogram<f64>,
    /// Histogram of the relay OMPF bytes of each landed `UpdateLayer` commit —
    /// the bytes its stream entry's `ompf` field carries, `0` for an empty
    /// commit — unlabelled ([`UPDATE_LAYER_RELAY_BYTES_BUCKETS`]).
    ///
    /// One sample per commit the service saw land, the same commits
    /// [`Self::update_layer_attempts_histogram`] counts, so the two `_count`s
    /// agree; a commit no attempt of its call saw land is not sampled. That
    /// covers every route — the fast and structural scripts, the WATCH path,
    /// the WATCH path's two late confirmations (an unwatched commit accepted,
    /// a lost `EXEC` reply reconciled), and a later attempt of the same call
    /// that finds the commit already landed (`ALREADY_LANDED`), whose
    /// committing attempt lost its reply. Never for the cross-call answers (an
    /// earlier call's operation, a conflict, the overtaken answer), which
    /// report no commit of this call. It is what an operator
    /// sizes `LIVE_EDIT_LAYER_STREAM_MAX_BYTES` and the backing store from;
    /// the per-object retained bytes have no gauge (the cardinality would be
    /// unbounded), and are bounded by the budget instead. A histogram, so no
    /// series exists before the first commit.
    pub update_layer_relay_bytes: Histogram<f64>,
    /// Total writes the backing store refused before anything was applied,
    /// by `stage` ([`STORE_WRITE_REFUSED_STAGES`]) and `reason`
    /// ([`STORE_WRITE_REFUSED_REASONS`]): an `-OOM`, `-MISCONF`, `-READONLY`
    /// or `-NOREPLICAS` reply (`storage::client::store_write_refusal`). Mostly
    /// upload writes, which the call answers `UNAVAILABLE` "nothing was
    /// applied, retry"; the stages below name the other writes that share
    /// them.
    ///
    /// `ext_store` is the external-value pre-store every upload runs before
    /// its commit (it also counts the DR restore's pre-store and the read
    /// path's lazy rewrite, which share it); `update_commit` an `UpdateLayer`
    /// commit (either script mode, or the WATCH path's discarded
    /// transaction); `create_commit` a `CreateLayer` object-id allocation
    /// (its counter repair writes included), materialization, overwrite mark
    /// or new-layer commit. The upload paths do not retry a refusal
    /// in-process: none of the four clears within a retry loop's
    /// milliseconds. (The DR restore maps a refused pre-store to its own
    /// retryable error like any other pre-store failure.) All twelve cells are seeded at zero, so the alert
    /// `increase(...{reason="out_of_memory"}[5m]) > 0` has a series from
    /// t=0.
    pub store_write_refused_total: Counter<u64>,
    /// Total `update_layer` attempts abandoned because the layer changed under
    /// them, by the attempt's `path` ([`UPDATE_LAYER_PATHS`]) and `kind`
    /// ([`UPDATE_LAYER_CONFLICT_KINDS`]) (Spec SDD SR-2.3).
    ///
    /// On the WATCH path that is a watched key (header, stream or snapshot
    /// fence) changing after the read — usually another edit's commit, but
    /// also a checkpoint cursor stamp on the header (`watch_conflict`) — or
    /// the structural snapshot read failing its `seq` proof
    /// (`watch_read_moved`). One tick per abandoned attempt, which is retried
    /// while the call has attempts left. Other retriable failures (transient
    /// I/O, a replaced `WATCH` connection) are not counted. On the structural
    /// route it is the commit's `STALE` (`seq_fence`) or `CACHE_DIVERGED`
    /// (`cache_diverged`) refusal, under the attempt's path (`structural`,
    /// `structural_full` or `reset`). On every path, an attempt a live
    /// snapshot fence refused (`snapshot_fence_wait`, waited out without
    /// spending a retry) ticks under that attempt's path. Zero-seeded over
    /// every `(path, kind)` cell, including cells no route can emit.
    pub update_layer_conflicts_total: Counter<u64>,
    /// Total structural `UpdateLayer` attempts that suspended because their
    /// upload charge could not grow without waiting (SDD §5.4), by the
    /// attempt's `path` ([`UPDATE_LAYER_PATHS`]): each released everything
    /// it held — its cache entry, its `WATCH` and transaction checkout, the
    /// per-object permits — waited in the upload gate for the larger charge
    /// and resumed with the same attempt number, spending no retry.
    /// Zero-seeded over every path.
    pub update_layer_recharges_total: Counter<u64>,
    /// Histogram of the object-state bytes an `UpdateLayer` call held in its
    /// upload charge when it returned its own commit — the memory it held for
    /// the object it edits, on top of its processing charge (SDD §5.4) — by
    /// the committing attempt's `path` ([`UPDATE_LAYER_PATHS`])
    /// ([`UPLOAD_OBJECT_STATE_BYTES_BUCKETS`]). One sample per call that
    /// returns its own commit under an upload charge on every path but
    /// `fast`: `structural`, `structural_full`, `reset` and `watch` (a
    /// fast-shaped delta under the `txn` backend included). The Lua fast path
    /// holds no object-state term, so it is not sampled; nor are
    /// storage-level commits with no upload. A histogram, so no series exists
    /// before the first sample.
    pub upload_object_state_bytes: Histogram<f64>,
    /// Total structural or reset `UpdateLayer` calls on the `lua` backend that
    /// left the cached structural route and committed through the WATCH path
    /// for a reason other than an uncacheable object, by `reason`
    /// ([`UPDATE_LAYER_STRUCTURAL_FALLBACK_REASONS`]), once per call:
    ///
    /// - `repeat_divergence`: a second `CACHE_DIVERGED` within one call, after
    ///   the first had reloaded the cache entry — the reload rebuilt the same
    ///   wrong belief, and the WATCH path, which rewrites what it touches,
    ///   commits the call instead of it looping. Expected to stay 0, like
    ///   [`Self::structural_cache_divergence_total`].
    /// - `over_work_limit`: the plan's work was over the structural commit's
    ///   work limit (`storage::structural_size_gate`), so the call committed
    ///   through the WATCH path rather than hold the shard in one long script.
    ///   Expected for the largest edits: resets and deletes of thousands of
    ///   placed assets.
    /// - `reset_over_work_limit`: a reset of a layer with no cache entry whose
    ///   estimated work was over that limit, sent to the WATCH path before
    ///   the cold start's full read.
    ///
    /// Seeded at zero for every reason.
    pub update_layer_structural_fallbacks_total: Counter<u64>,
    /// Histogram of the structural commit's `EVALSHA` round trip, as the
    /// client sees it, in seconds: one sample per structural commit script
    /// call written to the store, whatever it answered (a commit, a refusal,
    /// an error, a client timeout). The script holds the object's shard for
    /// its server-side part, so this is the bound on that hold plus the round
    /// trip. Unlabelled; buckets
    /// [`UPDATE_LAYER_STRUCTURAL_COMMIT_BUCKETS`], whose 2, 2.5 and 5 s
    /// boundaries are the size gate's decision lines. A call over
    /// `storage::structural_size_gate::STRUCTURAL_COMMIT_SLOW_WARN` also logs
    /// a WARN with the plan's work and the layer's size. Not pre-registered.
    pub update_layer_structural_commit_seconds: Histogram<f64>,
    /// How long one `update_layer` call waited on snapshot fences, by
    /// `backend` ([`UPDATE_LAYER_BACKENDS`]): from the start of the first
    /// attempt a fence refused or failed (an attempt an epoch interrupted
    /// counts whole) to the end of its last fence sleep — not to the call's
    /// end, so the attempt that finally commits is not in it — observed once
    /// per call that met a fence — committed, failed, or given up at the cap
    /// (Spec SDD SR-5.4, SR-8.3). Calls that never met a fence are not
    /// observed, so the distribution is that of the writer pause itself
    /// (SR-9.1). Seconds.
    pub update_layer_fence_wait_seconds: Histogram<f64>,
    /// Writer-intent registrations by WATCH-path attempts that read a snapshot,
    /// on layers above the atomic-read ceiling (the cached structural route
    /// registers none, structural-edits SR-8.2), by `result`
    /// ([`UPDATE_LAYER_FENCE_INTENT_RESULTS`]):
    /// `registered`, or `refused_fenced` when a fenced reader held the layer
    /// and the registration was refused (Spec SDD SR-8.3). Each cell counts
    /// registrations, not attempts or waits: a refusal before `WATCH` is
    /// ignored and the attempt falls through to its guards, so one attempt can
    /// tick both cells, and only a refusal at step 2b is a fence wait
    /// ([`UPDATE_LAYER_FENCE_INTENT_REFUSED_FENCED`]).
    pub update_layer_fence_intent_total: Counter<u64>,
    /// `update_layer` calls failed `UNAVAILABLE` because their fence waiting
    /// reached its cap — `LIVE_EDIT_RPC_DEADLINE_MS`, or what was left of the
    /// RPC's deadline (Spec SDD SR-5.4, SR-8.3). Retryable for the client; a
    /// sustained rate means fenced reads on a layer outlast the RPC deadline.
    pub update_layer_fence_timeout_total: Counter<u64>,
    /// Total commits refused because the per-object stream already held an
    /// entry at or above the ID the commit intended to write, by `backend`
    /// (`lua` | `txn`) and `site` (`update` | `create`) (b.ph9). The
    /// `{backend="lua", site="update"}` cell is split by `mode`
    /// ([`UPDATE_SCRIPT_MODES`]): the update script raises it from its fast
    /// mode and from its structural mode, and `mode` says which. No other cell
    /// carries `mode`; see [`UPDATE_SCRIPT_MODES`].
    ///
    /// Named for the condition rather than for one RPC: the `create` cell is
    /// raised by `CreateLayer`, not by `update_layer`, so the original
    /// `update_layer`-prefixed name mis-described a third of its own series.
    ///
    /// **Alertable at any non-zero rate.** A stream top above the header
    /// sequence is not a condition normal operation can produce: both commit
    /// paths pin `{new_seq}-0`, and the two tombstone sites use auto-IDs only
    /// against objects that are being retired. A hit here therefore means
    /// something out-of-band wrote the stream, or a restore seeded the header
    /// below a surviving stream's top. The commit was rejected whole — no
    /// half-committed state, no silently-missed subscriber update — but the
    /// layer is unwritable until an operator clears the stream, so this is a
    /// page, not a dashboard tile.
    ///
    /// Not folded into [`Self::update_layer_retries_total`]: that counter's
    /// `outcome` vocabulary describes how the retry loop ended, and a
    /// collision is not retried at all (see the `Internal` mapping in
    /// `LayerStore::map_update_script_error`, and its txn twin in
    /// `LayerStore::probe_stream_id_collision`).
    pub stream_id_collision_total: Counter<u64>,
    /// Total commits refused because a key the commit was about to mutate held
    /// the wrong Valkey type, by `backend` (`lua` | `txn`) and `site`
    /// (`update` | `create`) (b.459; `create` added by b.zvd, where the
    /// materialization paths gained the same pre-`MULTI` probe). As on
    /// [`Self::stream_id_collision_total`], the `{backend="lua", site="update"}`
    /// cell alone is split by `mode` ([`UPDATE_SCRIPT_MODES`]): the update
    /// script's validation pre-pass refuses in both its fast and its structural
    /// mode. The `create` cell has three producers, and the log line's
    /// `entry_point` field (plus its message) says which:
    ///
    /// - the pre-`MULTI` probe of both materialization entry points —
    ///   `LayerStore::write_object` for a live `CreateLayer` and
    ///   `LayerStore::write_restored_object` for a DR restore — which refuses
    ///   before anything is written;
    /// - the `CreateLayer` handler's read of a client operation's
    ///   `le:{sid}:createop:{operation_id}` record
    ///   (`LayerStore::resolve_create_layer_operation`), which refuses before
    ///   any object is allocated or written;
    /// - the `CreateLayer` commit point (`scripts/create_layer_commit.lua`),
    ///   when the layer mapping, the session's layers index, the new object's
    ///   commit claim (`le:{sid}:createclaim:{object_id}`) or the operation
    ///   record holds a foreign type. The script refuses before its own first
    ///   write, so the layer mapping does not move — but the new object was
    ///   already materialized. On a new layer it is left as an orphan the
    ///   orphan sweep retires. On an overwrite whose foreign key is the layers
    ///   index, the commit claim or the operation record, the old object stays
    ///   mapped and marked superseded, naming the new object as its successor,
    ///   until the orphan sweep completes the replacement and the layer names
    ///   the refused call's object: the sweep retries every pass after its
    ///   grace and completes it once the key is removed. A foreign layer
    ///   mapping maps no object: `UpdateLayer` is refused at its mapping read,
    ///   and every orphan sweep pass stops when it reads that key, stalling
    ///   the sweep's later work across the deployment until it is fixed; once
    ///   the key is removed the layer has no mapping, the sweep retires both
    ///   the old object and the refused call's object, and the layer must be
    ///   created again. For an operation record this is only the narrow case
    ///   of a key retyped between the handler's read and the script; the read
    ///   above catches a standing one first.
    ///
    /// **Alertable at any non-zero rate**, same posture as
    /// [`Self::stream_id_collision_total`] and for the same reason: nothing
    /// this service does can retype a key it owns, so a hit means an
    /// out-of-band writer put a foreign value on the `le:` keyspace. Every
    /// `update` refusal rejects the commit whole — the Lua pre-pass runs before
    /// the `XADD`, the txn probe before `MULTI` — and so do the first two
    /// `create` producers; the commit-point producer leaves the object (and,
    /// on an overwrite, the stranded replacement) described above. Either way
    /// the layer (or, for an operation record, that
    /// `operation_id`) stays refused until an operator removes the offending
    /// key, which the ERROR log line at each arm names.
    ///
    /// Deliberately NOT folded into [`Self::stream_id_collision_total`]: the
    /// stream is healthy here and that counter's remediation ("clear the
    /// per-object stream") would send an operator at the wrong key.
    pub keyspace_type_error_total: Counter<u64>,
    /// Total object-state touches refused because the key they addressed held
    /// the wrong Valkey type, by `key_family` ([`SNAPSHOT_READ_KEY_FAMILIES`])
    /// (b.atj, widened by b.muu).
    ///
    /// "Touches", not "reads": every tick site but one is a read, and the
    /// exception is named here rather than given a counter of its own. The
    /// orphan sweep's retirement fence (`HSET le:{oid}:obj:hdr superseded 1`,
    /// `storage::orphan_sweep::cleanup_orphaned_object`, SDD §5.5 step 6) is a
    /// **write** on `obj:hdr` with exactly the same `WRONGTYPE` exposure as the
    /// reads, and it ticks `key_family="header"` here. A write-side sibling
    /// counter was rejected for b.muu because the operator condition, the zero
    /// baseline, the remediation and the `key_family` value are all identical to
    /// the read side's — a third counter would carry one call site, ten
    /// permanently-zero cells and a third term on an alert that already `or`-
    /// joins two.
    ///
    /// The read-side twin of [`Self::keyspace_type_error_total`], deliberately
    /// a separate counter rather than a third `site` value on it: that counter's
    /// `site` vocabulary is an operation (`update` | `create`), its `backend`
    /// label is meaningless for a read, and its registered description and
    /// remediation both speak of a commit that was "about to mutate" a key.
    /// Here nothing was going to be written.
    ///
    /// **Alertable at any non-zero rate**, same posture and same underlying
    /// condition as its commit-side sibling — nothing in this service retypes a
    /// key it owns, so a hit means an out-of-band writer put a foreign value on
    /// the `le:` keyspace. The blast radius is wider than a refused commit,
    /// which is why it pages: the object-state reads fail closed (b.zvd), so
    /// one wrong-typed key takes down every consumer whose read touches it, up
    /// to **five** for that object, until an operator removes the key, which
    /// the accompanying ERROR log line names. The five, exhaustively: four
    /// through `storage::diff_read::read_object_snapshot`'s key set — the
    /// structural `UpdateLayer` WATCH path
    /// (`LayerStore::update_layer_once_watch`, consumer `structural_update`)
    /// by its unfenced fan-out at every size; catch-up at every size — at or below
    /// the atomic-read limits by its atomic read, above them by the fenced
    /// read's fan-out its shared fenced-read flight runs
    /// (`storage::snapshot_flights`; a legacy replay's coalesced route
    /// included); the checkpoint assembler at every size — by the atomic read
    /// at or below those limits, by a fenced read of its own above them
    /// (`checkpoint::assembler::ValkeySnapshotLoader`); and per-commit
    /// reconstruction (a legacy in-window replay) at or below those limits by
    /// the atomic read. No per-commit reconstruction is left above them: a
    /// legacy in-window replay's first per-commit atomic read answers
    /// `TOO_LARGE`, and the replay then reads through catch-up's coalesced
    /// route (and a live commit is never reconstructed: one without relay bytes
    /// is an empty commit, sent with no read) — and the fifth, external-value
    /// clean-up's reference release (`storage::layer::schedule_old_object_ext_gc`,
    /// reached from `DeleteLayer`, the `CreateLayer` overwrite and the orphan
    /// sweep), whose counted size read (`storage::retirement`: `HGET obj:hdr
    /// ext_gc_scheduled`, `HLEN obj:ext`, `SCARD` of `obj:secs`, `obj:dels`
    /// and `obj:chs`, `ZCARD` of `obj:flds` and `obj:tss`) sizes the
    /// clean-up, and whose reference walk (consumer `ext_gc`) reads only the
    /// header, `obj:flds` and `obj:tss`, their field and time-sample rows,
    /// and `obj:ext`. A wrong-typed header, index or entry hash stops the
    /// four; one on the header or a field or time-sample row also stops the
    /// clean-up's release (the layer is still deleted, its references freed
    /// later by the reference-count reconciler). One on any of the five
    /// indices ticks this counter at the size read, under its own
    /// `key_family` (`obj:flds` and `obj:tss` again at the walk), and the
    /// clean-up goes on at its full page (the orphan sweep runs no size read
    /// and no walk for an orphan whose fence failed or whose id is not
    /// numeric, so there it ticks nothing); the teardown then stops after its
    /// first `UNLINK`, which pages them all (`obj:flds` and `obj:tss` also
    /// stop the release). The five indices and every row not yet removed
    /// stay, and on `DeleteLayer` so does the layers-index member;
    /// `DeleteLayer` answers `INTERNAL` with the mapping already deleted
    /// (b.yz5). The teardown's own index pages count nothing; its failure
    /// names the object and key.
    /// A wrong-typed `obj:ext` (`key_family="ext_index"`) stops none of the
    /// four snapshot reads; the clean-up skips its release and removes the
    /// key. Edits refuse it on their own
    /// ([`Self::keyspace_type_error_total`]). On an object carrying the
    /// `chs_complete` marker the snapshot read reads `obj:chs` and only the
    /// lists it names under a live parent, so a wrong-typed `obj:chs` stops the
    /// four at every size (the atomic read's size check reads `obj:chs` before
    /// it can answer `TOO_LARGE`), and a wrong-typed list `obj:chs` does not
    /// name stops none of them. The per-key split is spelled out on
    /// `storage::diff_read::SNAPSHOT_READ_TYPE_ERROR_REMEDIATION`.
    ///
    /// Structural `UpdateLayer` is on that WATCH path on every call under
    /// `txn` and `lua-watch-structural`, and under `lua` on its uncacheable
    /// and repeat-divergence fallbacks. Under `lua` otherwise it commits
    /// through the structural write-planning cache, which reads the layer
    /// (consumer `structural_cache`, `storage::structural_cache_cold_start`)
    /// only on a cold start or reload, so only those calls stop, through that
    /// read, on a key it touches. At or below the plain ceilings that read is
    /// `read_object_snapshot_with_ceilings`'s atomic read; above them it is
    /// the fenced read, whose fan-out also ticks this counter
    /// (`storage::diff_read::count_object_key_refusal`). A cached call reads
    /// the header, stream and commit token (`structural_cache_sync.lua`),
    /// whose refusals are mapped by `map_transient_fred_err` and counted
    /// nowhere here. Its commit
    /// (`update_layer_fast.lua`, structural mode) still `TYPE`-probes every
    /// record and index key it writes, `obj:chs` included whenever its plan
    /// writes a children list; a foreign value on such a key refuses the
    /// commit with `WRONGTYPE_KEY`, counted on
    /// [`Self::keyspace_type_error_total`] (`backend="lua"`, `site="update"`,
    /// `mode="structural"`), not here. The cache's sampled
    /// audit (`storage::structural_cache_audit`, consumer
    /// `structural_cache_audit`) reads the same way in a background task —
    /// the atomic read at or below the plain ceilings, the fenced read above
    /// them: its refusal ticks this counter, skips the audit and stops no
    /// call.
    ///
    /// **What it does not catch.**
    ///
    /// 1. *Permanent non-`WRONGTYPE` refusals* — a fred `Protocol` / `Parse` /
    ///    `Auth` / `Config` error on the same touches. These wedge the same
    ///    consumers and are deliberately **not** folded in here: the remediation
    ///    is opposite (a service or deployment fault, not a foreign value on the
    ///    keyspace), and folding them in would destroy this counter's zero
    ///    baseline. Since b.muu they are counted next door, on
    ///    [`Self::snapshot_read_refused_total`], from the same helper.
    /// 2. *Four `obj:hdr` touches in `storage::layer` that still bypass the
    ///    count — two pre-reads, one blind read, one write.* They are why
    ///    `key_family="header"` means four different things by consumer:
    ///    **counted on first touch** from checkpoint assembly, from the orphan
    ///    sweep and from external-value clean-up (none has an uncounted touch
    ///    of the header in front of its counted one; the sweep's discovery read
    ///    `?`-propagates, so a refused header ends the pass there, and the
    ///    clean-up's counted size read comes before
    ///    `schedule_old_object_ext_gc`'s own `HGET obj:hdr ext_gc_scheduled`,
    ///    so a refused header skips the release); **race-only** on the two
    ///    `storage::layer` pre-read consumers, each of which meets a wrong-typed
    ///    header first and maps its
    ///    own error, so this family moves only if the key is retyped between the
    ///    pre-read and the counted load; **blind** on `StatLayer`, which has no
    ///    counted load behind its header read at all, so the RPC fails closed
    ///    with no metric, no log line and no key name (`Internal` is scrubbed at
    ///    the gRPC boundary); and one **uncounted write**, the
    ///    CreateLayer-overwrite retirement fence, not race-shaped because a
    ///    write shadows no read. The criterion, stated so the set can be
    ///    re-derived from the key rather than trusted as a count: an `obj:hdr`
    ///    touch there ticked by neither this counter nor
    ///    `keyspace_type_error_total`, type-sensitive enough to answer
    ///    `WRONGTYPE`, and with no counted touch of the same key ahead of it on
    ///    the same path.
    ///
    ///    The follow-up Issue (b.iy5) closes the cell by routing them through
    ///    `storage::diff_read::count_object_key_refusal`, and its scope is all
    ///    four rather than a subset. b.muu routed everything in this counter's
    ///    **declared scope** that sits outside `storage::layer`; the checkpoint
    ///    **restore** lane is same-family and type-sensitive but deliberately
    ///    outside that declared scope rather than an oversight — its reads carry
    ///    their own ERROR line and remediation and tick no counter — and rides
    ///    b.iy5 with the four. Per-site inventory, and the exclusions the
    ///    criterion drops and why: the rustdoc of
    ///    `storage::diff_read::SNAPSHOT_READ_TYPE_ERROR_REMEDIATION`.
    pub snapshot_read_type_error_total: Counter<u64>,
    /// Total object-state touches refused by a **permanent, non-`WRONGTYPE`**
    /// backend failure, by fred error `kind`
    /// ([`SNAPSHOT_READ_REFUSAL_KINDS`]) (b.muu).
    ///
    /// The other half of [`Self::snapshot_read_type_error_total`]: same touch
    /// set, same fail-closed blast radius, ticked from the same helper
    /// (`storage::diff_read::count_object_key_refusal`) on the mutually
    /// exclusive arm. "Permanent" is `storage::client::is_transient_fred_error`
    /// returning `false` and nothing else — the identical predicate
    /// `map_snapshot_load_err` uses to choose `Aborted` — so a tick here is
    /// exactly a refusal that did not reach a caller as a retried `Aborted`:
    /// `ServiceError::Internal`, minus the `WRONGTYPE` case the sibling claims,
    /// or a store write refusal (`storage::client::store_write_refusal`, the
    /// orphan sweep's fence write alone on this touch set). Transient failures
    /// (IO / Timeout / Canceled / Cluster / Backpressure, the two router
    /// carve-outs and the store's not-serving `-LOADING` / `-BUSY`) are retried
    /// and counted nowhere.
    ///
    /// **`kind` only, no `key_family`.** The remediation for `protocol` /
    /// `auth` / `config` is a service or deployment fault and is not
    /// key-specific, so the family would change no operator action; the
    /// 12 × 12 grid would seed 144 permanently-zero series against 12; and the
    /// accompanying log line already carries both the family and the key in
    /// full, which is strictly more than a label.
    ///
    /// **Alerting.** `increase(...) > 0` on the eleven "should never happen"
    /// cells; `backend_unavailable` — the orphan sweep's fence write refused by
    /// a store condition that can clear on its own (`-MISCONF`, `-READONLY`,
    /// `-NOREPLICAS`) — takes a sustained-rate rule instead; a healthy
    /// backing-store restart does not move it, because its `-LOADING` /
    /// `-BUSY` are transient — see
    /// [`SNAPSHOT_READ_REFUSAL_KIND_BACKEND_UNAVAILABLE`]. The log level tracks
    /// that split: the eleven are ERROR, `backend_unavailable` is WARN, so a
    /// condition that clears on its own cannot bury the eleven under ERRORs an
    /// operator then learns to filter away. `backend_out_of_memory` is on the
    /// ERROR / `> 0` side of that split despite also coming from the backing
    /// store, because under the mandated `noeviction` policy it does not clear
    /// on its own — see
    /// [`SNAPSHOT_READ_REFUSAL_KIND_BACKEND_OUT_OF_MEMORY`].
    ///
    /// Counting only: this counter steers nothing, and b.muu changed no
    /// classification, mapping or control flow anywhere.
    pub snapshot_read_refused_total: Counter<u64>,
    /// Structural WATCH-path snapshot reads whose per-window read script
    /// (`storage::lua::FANOUT_WINDOW_READ_SCRIPT`) met a cluster-state
    /// refusal of an undeclared same-slot key — the slot migrating, not
    /// served, or the cluster down — and that finished on plain commands
    /// (SDD §10.5): one tick per read that falls back, when it falls back.
    /// No labels: there is one reason and one consumer. Expected zero
    /// outside a resharding or a cluster failover.
    pub snapshot_read_window_fallback_total: Counter<u64>,
    /// Object-snapshot reads by `consumer` ([`SNAPSHOT_READ_CONSUMERS`]),
    /// `path` ([`SNAPSHOT_READ_PATHS`]) and `outcome`
    /// ([`SNAPSHOT_READ_OUTCOMES`]) (Spec SDD SR-8.1), one tick per read
    /// attempt `storage::diff_read::read_object_snapshot` completes, one per
    /// reference walk external-value clean-up completes
    /// (`storage::retirement`, `consumer="ext_gc"`, `path="unfenced"`), and one
    /// per fenced read attempt
    /// `storage::snapshot_fence::read::read_object_snapshot_fenced` completes
    /// (`path="fenced"`: `ok` when its release verified it, `fence_broken`
    /// when the attempt was broken — its release answered `BROKEN`, a lost
    /// lease stopped its fan-out, or its fan-out found no header that its
    /// release then found (a restore under the read) — and was retried or
    /// given up, `not_found`,
    /// or the error that ended it; an attempt ended by the caller's wait hook,
    /// or cancelled, records nothing).
    ///
    /// `outcome="too_large"` is an atomic attempt that answered `TOO_LARGE`
    /// (early-out or mid-read): the rate at which layers take their consumer's
    /// above-ceiling path. For every consumer with one — catch-up, the
    /// checkpoint, the structural cache's cold start and its audit (with its
    /// above-ceiling switch on) — the read that path then runs is counted
    /// under its own path, the fenced read: a catch-up's
    /// follow-up (including a per-commit replay handed over to the
    /// out-of-window route) is its shared fenced-read flight's read
    /// (`storage::snapshot_flights`), one per flight however many catch-ups
    /// share it, under `consumer="catchup"`; a backup's is the checkpoint
    /// assembler's own fenced read, under `consumer="checkpoint"`; a cold
    /// start's is its own fenced read, under `consumer="structural_cache"`;
    /// an audit's, paced and never waiting for capacity, under
    /// `consumer="structural_cache_audit"`.
    /// A probed cold start reads the plain atomic read only when its route
    /// predictor says the layer fits, so its `too_large` means a predictor
    /// miss or a layer that grew since the probe; an unprobed cold start
    /// (rare: an index key of a foreign type) always reads plain first.
    ///
    /// Seeded at zero for the cells a path that production runs can emit
    /// ([`SNAPSHOT_READ_ATOMIC_CONSUMERS`] × [`SNAPSHOT_READ_ATOMIC_OUTCOMES`]
    /// on `path="atomic"`, [`SNAPSHOT_READ_UNFENCED_CONSUMERS`] ×
    /// [`SNAPSHOT_READ_UNFENCED_OUTCOMES`] on `path="unfenced"`, and
    /// [`SNAPSHOT_READ_FENCED_CONSUMERS`] × [`SNAPSHOT_READ_FENCED_OUTCOMES`]
    /// on `path="fenced"`).
    pub snapshot_read_total: Counter<u64>,
    /// Wall time of one object-snapshot read in seconds, by `consumer` and
    /// `path` (Spec SDD SR-8.2): from issuing the read to its decoded result,
    /// recorded for every outcome. On the atomic path that is the `EVALSHA`
    /// round trip (the server-side script time plus the network) plus the
    /// reply's decode; on the unfenced path, every window of the fan-out plus
    /// the decode; on the fenced path, one attempt's fan-out under its lease
    /// (to the decoded state, or to where it failed or a lost lease stopped
    /// it), not the wait for the lease nor its release — those are
    /// [`Self::snapshot_fence_reader_wait_seconds`] and
    /// [`Self::snapshot_fence_hold_seconds`] — or, for an attempt that ended
    /// at its acquire (`not_found` or an error), that acquire's round trip.
    /// A histogram, so no series exists before the first read.
    pub snapshot_read_duration_seconds: Histogram<f64>,
    /// Reply bytes of one successful object-snapshot read, by `consumer` (Spec
    /// SDD SR-8.2): the length of every string the read read for the object's
    /// indices, entries and children — the same tally the atomic read's byte
    /// limit is checked against. It includes the `obj:chs` members a marked
    /// object's read reads but does not return, and not the header or the
    /// reply framing. For `consumer="ext_gc"`, external-value clean-up's
    /// reference walk (`storage::retirement`): the strings it received — the
    /// value indices' members, each value row's columns but its inline value
    /// (never transferred) and `obj:ext`'s fields and values.
    pub snapshot_read_reply_bytes: Histogram<f64>,
    /// Entries read by one successful object-snapshot read, by `consumer`
    /// (Spec SDD SR-8.2): one per section, tombstone, field and time-sample
    /// hash and one per children list the read issued a command for. For
    /// `consumer="ext_gc"`, the field and time-sample rows the reference walk
    /// read.
    pub snapshot_read_entries: Histogram<f64>,
    /// Cold starts of the structural write-planning cache, by the `route`
    /// that ended each ([`STRUCTURAL_COLD_START_ROUTES`]; structural-edits SDD
    /// SR-2.5), one tick per cold start that produced an entry or an
    /// uncacheable verdict. A cold start that produced neither — a failed or
    /// `NotFound` read, a superseded object, an object retired or dropped
    /// while it read, a disabled cache — ticks nothing
    /// here; its reads show on [`Self::snapshot_read_total`] under
    /// `consumer="structural_cache"`. `fenced` counts one fenced read per
    /// object per replica (plus one per reload) of the layers above the plain
    /// ceilings or predicted over the plain byte ceiling. A fenced read that
    /// failed for its call (`fenced_read_failed` on
    /// [`Self::structural_cache_uncacheable_total`]) ticks nothing here.
    /// Seeded at zero for every route.
    pub structural_cold_start_reads_total: Counter<u64>,
    /// Wall time of one cold start in seconds, by `route` (SR-2.5): from the
    /// first read — its wait for a per-replica plain permit included — to the
    /// uncacheable verdict or the entry's admission (not the commit-token read
    /// after it), so every route includes its permit waits, and a `fenced` or
    /// `uncacheable` cold start also any plain read that came first and the
    /// fenced read's waits for its permits, its memory grant and the snapshot
    /// fence. Recorded exactly when
    /// [`Self::structural_cold_start_reads_total`] ticks. A histogram, so no
    /// series exists before the first cold start.
    pub structural_cold_start_read_duration_seconds: Histogram<f64>,
    /// How each structural `UpdateLayer` attempt that consulted the structural
    /// write-planning cache got its entry, by `outcome`
    /// ([`STRUCTURAL_CACHE_LOOKUP_OUTCOMES`]; structural-edits SDD SR-2.4):
    /// `hit`, `repaired`, `cold` (no entry: a cold start, not a reload) or
    /// `reload` (an entry dropped on a repair anomaly and read again). It
    /// counts attempts, not calls (decider ruling, Q3): at most one tick per
    /// read of the cache, and an attempt reads it once — a retry after `STALE`
    /// or `CACHE_DIVERGED`, and a re-attempt after a fence wait, read it
    /// again, and that repair is real store work. An attempt routed as
    /// uncacheable, or whose repair failed on a store error before its
    /// outcome was known, ticks nothing. Seeded at zero for every outcome.
    pub structural_cache_lookups_total: Counter<u64>,
    /// Entries of the structural write-planning cache dropped and read again,
    /// by `reason` ([`STRUCTURAL_CACHE_RELOAD_REASONS`], SR-2.4). The read
    /// side ticks the repair anomalies; the structural `UpdateLayer` attempt
    /// ticks `diverged` and an own creation's `dangling_adopted`; and every
    /// explicit put-back the per-entry cap refuses — the attempt's or the
    /// read side's — ticks `uncacheable`. Seeded at zero for every reason.
    ///
    /// **Reading the `reason` cells — routine or alertable.** A reload never
    /// produces a wrong edit (the layer is read again); each costs one
    /// whole-layer read.
    ///
    /// | `reason` | Reading |
    /// |---|---|
    /// | `too_far` | Routine. A high rate means layers take many value edits between structural edits on one replica. |
    /// | `gap` | Routine on long-idle entries (the stream window trimmed the entries the repair needs) and after a restore. |
    /// | `too_large` | Routine for layers edited with very large single uploads. |
    /// | `dangling_adopted` | Routine but rare. |
    /// | `identity` | Expected only after a restore or a backing-store failover that lost writes; otherwise investigate. |
    /// | `regressed` | Expected only after a restore, a failover that lost writes, or during a rolling upgrade (an older build's commits); otherwise investigate. |
    /// | `tombstone`, `parse`, `replay_err` | Not expected: no writer produces them; investigate any increase. |
    /// | `diverged` | Alertable: read with [`Self::structural_cache_divergence_total`]. |
    /// | `uncacheable` | Act on it as on [`Self::structural_cache_uncacheable_total`]`{reason="over_weight"}`: layer sizes against the cache budget. |
    pub structural_cache_reloads_total: Counter<u64>,
    /// Uncacheable verdicts of the structural write-planning cache, by
    /// `reason` ([`STRUCTURAL_CACHE_UNCACHEABLE_REASONS`]): one tick per
    /// verdict, where it is reached
    /// (`storage::structural_cache_cold_start::report_uncacheable`), beside
    /// one INFO line naming the session, layer and object and, for a ceiling
    /// or weight verdict, the bound against its ceiling (for `over_deadline`,
    /// the layer's size and the configured RPC deadline). A verdict recorded as
    /// a negative entry ticks exactly once per negative entry, by whoever
    /// first takes it (`StructuralCache::take_unreported`): at once where it
    /// is recorded — by a cold start (`over_weight`, `no_chs_complete`,
    /// `over_deadline`), the read side's probe (`no_chs_complete`) or an explicit put-back the
    /// per-entry cap refused (`over_weight`) — or, for the `over_weight`
    /// verdict the implicit put-back of a dropped call records, at the first
    /// structural lookup it routes. Lookups of an already-reported verdict
    /// tick nothing. `reload_limit` and `fenced_read_failed`, verdicts about
    /// one call that are never recorded, tick once per such call, and their
    /// lines are DEBUG, not INFO: nothing bounds their rate. A disabled cache
    /// ticks nothing.
    /// Unlike `live_edit_structural_cold_start_reads_total{route="uncacheable"}`,
    /// which counts cold starts, this says why. Seeded at zero for every
    /// reason.
    ///
    /// **Reading the `reason` cells.** Any tick means some layer's structural
    /// edits take the slower WATCH path, which reads the whole layer on every
    /// attempt; field edits are unaffected.
    ///
    /// | `reason` | Reading and remedy |
    /// |---|---|
    /// | `over_weight` | The layer's state is over the per-entry limit (a quarter of the cache's budget, 5 % of the pod memory limit). A larger budget, raised with the pod memory limit, helps only a layer whose fenced load also fits the deadline; at the defaults, split the layer. |
    /// | `no_chs_complete` | A layer created before the complete child-ordering index existed; nothing to tune. |
    /// | `reload_limit` | One edit found the layer changing between two loads; expected during a rolling upgrade, investigate a sustained rate outside one. |
    /// | `fenced_read_failed` | One edit's fenced load did not finish in time; expected now and then while the layer's fence or the replica's fenced-read capacity is busy. A sustained rate on one layer with `cause=cap`, `cap_ms=0` acts like `over_deadline`. |
    /// | `over_deadline` | The layer's fenced load plus one slower-path attempt would not fit `LIVE_EDIT_RPC_DEADLINE_MS`; raise the deadline (and the termination grace with it) or split the layer. |
    pub structural_cache_uncacheable_total: Counter<u64>,
    /// Stream entries one repair of a cache entry replayed (SR-2.4), one
    /// observation per repair the read side ran, whatever it ended in. A
    /// histogram, so no series exists before the first repair.
    pub structural_cache_replayed_entries: Histogram<f64>,
    /// Stream bytes one repair of a cache entry read (SR-2.4): the repair
    /// reads' byte tallies (entry ids plus every field name and value), one
    /// observation per repair, recorded with
    /// [`Self::structural_cache_replayed_entries`].
    pub structural_cache_repair_bytes: Histogram<f64>,
    /// Structural commits refused because the cache diverged from the store
    /// (`CACHE_DIVERGED`, SR-2.4). Alertable: expected to stay 0. Ticked by
    /// the structural commit's caller. Seeded at zero.
    pub structural_cache_divergence_total: Counter<u64>,
    /// Sampled audits of the structural write-planning cache, by `outcome`
    /// ([`STRUCTURAL_CACHE_AUDIT_OUTCOMES`]; SR-2.4, SR-3.6), one tick per
    /// audit that ran to an end. An audit not sampled, not started because
    /// another was running, or cancelled by shutdown ticks nothing. The
    /// audit's own reads and repair tick nothing else of the cache's: its
    /// snapshot reads show on [`Self::snapshot_read_total`] under
    /// `consumer="structural_cache_audit"`. `diverged` is alertable (expected
    /// 0). Seeded at zero for every outcome.
    pub structural_cache_audit_total: Counter<u64>,
    /// Audits of layers above the plain atomic-read ceilings, with the
    /// above-ceiling switch on, by whether they took the fenced read
    /// (`result`, [`STRUCTURAL_CACHE_AUDIT_FENCED_RESULTS`]; SR-3.6): `read`,
    /// `paced`, `busy` or `failed`. Each `read` paused the layer's colliding
    /// writers for its hold, which
    /// `live_edit_snapshot_fence_hold_seconds{consumer="structural_cache_audit"}`
    /// records; the audit's verdict still ticks
    /// [`Self::structural_cache_audit_total`]. Seeded at zero for every
    /// result.
    pub structural_cache_audit_fenced_total: Counter<u64>,
    /// How long one fenced read attempt held its snapshot-fence lease, in
    /// seconds, by `consumer` (Spec SDD SR-8.3): from the acquire reply that
    /// granted it to the reply of the release that gave it up — the fan-out,
    /// the renewals beside it and the release — whatever the release
    /// answered. The reader side of the pause writers see on
    /// `live_edit_update_layer_fence_wait_seconds`. An attempt cancelled while
    /// holding the lease is recorded up to its cancellation; its lease is then
    /// released in the background or lapses.
    pub snapshot_fence_hold_seconds: Histogram<f64>,
    /// Replies of `snapshot_fence_acquire.lua` to fenced readers, by `result`
    /// ([`SNAPSHOT_FENCE_ACQUIRE_RESULTS`]; Spec SDD SR-8.3): a grant that
    /// started or joined an epoch, or a `RETRY_AFTER` naming why the reader
    /// must wait. One tick per reply, so a reader that waited ticks once per
    /// retry plus once for its grant. A catch-up flight's read is one reader
    /// however many catch-ups share it. Every cell is seeded at zero.
    pub snapshot_fence_acquire_total: Counter<u64>,
    /// How long one fenced read attempt waited for its lease, in seconds, by
    /// `consumer` (Spec SDD SR-8.3): from its first acquire call to the
    /// acquire reply that granted the lease — the `RETRY_AFTER` sleeps and
    /// whatever the caller's wait hook did around them in between — or to
    /// where the wait ended without a grant (the object gone, an error, the
    /// hook's refusal such as its wait cap, or cancellation). One
    /// observation per attempt that called acquire, so a read retried after a
    /// `BROKEN` release observes once per attempt. SR-5.5 bounds it by one
    /// epoch, one gap, the intents and a polling step; `reader_wait_max(L)`
    /// is the cap callers end it at.
    ///
    /// One meaning only: a read attempt's wait for its lease. A catch-up
    /// flight's read (`storage::snapshot_flights`) is one reader, observed per
    /// attempt like any other, however many catch-ups share it. A catch-up
    /// that leaves its flight at `reader_wait_max(L)` is not observed here
    /// (what it counted — fence acquires, sleeps and the fenced read, maybe
    /// across attempts — is not a lease wait); it is counted on
    /// [`Self::subscriber_stream_exits_total`] under
    /// [`STREAM_EXIT_FENCED_READ_WAIT_CAP`], beside its WARN.
    pub snapshot_fence_reader_wait_seconds: Histogram<f64>,
    /// Total fred reconnect events, by `pool` — one of [`VALKEY_POOLS`]
    /// (`subscriber` | `data` | `txn` | `subscriber_pool` | `maintenance` |
    /// `snapshot`) (b.3z7 H-obs-3).
    pub valkey_reconnects_total: Counter<u64>,
    /// Total fred router tasks that ended other than by `quit()` and were
    /// restarted by `storage::router_supervisor`, by `pool` — one of
    /// [`VALKEY_POOLS`]. Baseline 0: fred marks nothing when a client's router
    /// task ends (the client still reads `Connected`), so without the restart
    /// every later command on that client would time out. Each tick has one
    /// ERROR line beside it naming the pool member and how its router ended.
    pub valkey_router_restarts_total: Counter<u64>,
    /// Total Valkey connections fred closed as unresponsive, by `pool` — one
    /// of [`VALKEY_POOLS`] — and `verdict` — one of
    /// [`VALKEY_UNRESPONSIVE_VERDICTS`]. One tick per verdict of the vendored
    /// fred's unresponsive check, each beside one fred WARN naming the client,
    /// node and verdict: `fast`, no reply progress for 2 s while the same slot
    /// range answered other connections; `slow`, none for 45 s, and also
    /// every expired bounded router wait (a write, a flush, a transaction's
    /// drain and reads, an `ASKING` read): the expiry marks the connection
    /// failed, so fred reports it with the slow verdict's error when it next
    /// polls or drops it. The exception is a cluster-wide command (the
    /// cluster credential refresh's `AUTH`, a cluster `SCRIPT LOAD`): an
    /// expiry in the drain before it, or in its own writes or reply reads,
    /// makes fred close the client's connections itself without reporting the
    /// error, so it is not counted here (its fred WARN is still logged).
    /// Neither is an expiry in the drain before a `QUIT`, which ends the
    /// router. Ticked by `storage::client`'s `spawn_error_metric`
    /// when fred broadcasts a connection error that
    /// `fred::live_edit::liveness::verdict_of` classifies. Seeded at zero per
    /// pool and verdict.
    pub valkey_unresponsive_total: Counter<u64>,
    /// Pool member replacements, by `pool` — one of
    /// [`VALKEY_REPLACEABLE_POOLS`] — and `outcome` — one of
    /// [`VALKEY_MEMBER_REPLACEMENT_OUTCOMES`] (`storage::member_replacement`).
    /// A member is replaced by a fresh fred client after its second fast
    /// unresponsive verdict within an hour: `replaced` when the fresh client
    /// connected and took the member over, `connect_failed` when it did not
    /// connect, `budget_exhausted` when a replacement was due but the pool's
    /// replacement budget was used up (a multiplexed pool's spare cells, or the
    /// `txn` pool's count of `max(size, 4)` replacements). Each `replaced` and
    /// `connect_failed` tick has
    /// one WARN beside it; `budget_exhausted` ticks on every refused
    /// replacement and logs one WARN per pool. Seeded at zero per pool and
    /// outcome.
    pub valkey_member_replacements_total: Counter<u64>,
    /// Valkey errors, by `pool` and `kind` (b.3z7 H-obs-3). Two producers:
    ///
    /// - fred's connection-event hook, one tick per connection-level error with
    ///   `kind` = fred `ErrorKind::to_str()` (connect failures, dropped
    ///   connections, a connect timeout as `"Timeout Error"`);
    /// - [`VALKEY_ERROR_KIND_COMMAND_TIMEOUT`]: one tick each time service code
    ///   receives a fred `Timeout` for a command it issued — fred reports those
    ///   only to the caller. Counted on RPC and background paths alike; not on
    ///   connection establishment, reconnects or shutdown `QUIT`. Each retry
    ///   attempt that times out counts; a fan-out that fails on its first
    ///   timeout counts once.
    ///
    /// Only the `command_timeout` cell is seeded (per pool).
    pub valkey_errors_total: Counter<u64>,
    /// Total `WatchGuard` teardowns that gave up on their slot-pinned
    /// `UNWATCH`, by `site` (`explicit` | `drop`) (b.ji8). One increment per
    /// teardown, never per attempt.
    ///
    /// **Alert on `site="drop"`.** Any non-zero value *there* is actionable,
    /// not a dashboard tile: `drop` is terminal — the last moment the process
    /// held the connection — so the pooled `txn` connection was returned to
    /// the pool still carrying WATCH state the service believes it cleared.
    /// The *next* borrower pays for it — as a spurious `EXEC=nil` → in-process
    /// `Aborted` (a wasted retry, or a client-visible `UNAVAILABLE` where no
    /// retry loop wraps the transaction) on a centralized deployment, or, on a
    /// proxied cluster (Redis Enterprise
    /// / Azure Managed Redis), as an `INTERNAL` carrying a `CROSSSLOT` that
    /// names a key the failing request never touched.
    ///
    /// `site="explicit"` does **not** carry that claim and should not page on
    /// its own. The teardown API leaves the guard armed on failure, so `Drop`
    /// tries again with the pool mutex still held — which is both why the
    /// explicit site does not retry in-line on the RPC request path and why an
    /// explicit tick means only "`Drop` will retry this", not "a connection was
    /// released dirty". It is useful as context next to a `drop` tick (the two
    /// firing together identify a connection that failed teardown twice over)
    /// and as a rate signal, not as a page.
    ///
    /// Neither cell counts a teardown whose failure said the connection itself
    /// was already gone (fred `Canceled` / `IO`, or fred's routing miss when the
    /// router holds no connection; `storage::watch_guard`'s
    /// `watch_died_with_the_connection`): a WATCH cannot outlive
    /// its connection, so there is no residue to inherit. Those are logged at
    /// `debug` and left uncounted (b.ji8 round 3) — without that carve-out,
    /// routine failover / proxy-recycle / reconnect churn on the `txn` pool
    /// would fire the `increase(...) > 0` alert below.
    ///
    /// Deliberately not folded into `valkey_errors_total{pool="txn"}`: that
    /// counter tracks fred connection-level events and command timeouts
    /// (an UNWATCH attempt that timed out counts there too), whereas this one
    /// asserts a specific broken invariant with a specific blast radius on an
    /// unrelated future request.
    pub storage_watch_unwatch_failures_total: Counter<u64>,
    /// Total watched transactions whose `WATCH` was found — or proven — void,
    /// by `path` and `disposition` (b.bhb).
    ///
    /// A `WATCH` dies with its connection, so any reconnect between `WATCH` and
    /// `EXEC` silently turns an optimistic-concurrency commit into an
    /// unconditional one. The service now refuses to replay anything issued
    /// under a `WATCH` (txn-pool `max_command_attempts = 1` +
    /// `storage::watched_txn`'s single pipeline), re-checks the connection
    /// identity before dispatch, and queues `CLIENT ID` inside the `MULTI` so
    /// the `EXEC` reply names the connection that ran it. This counter is what
    /// those three layers report.
    ///
    /// **Alert on `disposition="committed_unwatched"` only.** The other four
    /// dispositions describe the mechanism working, or the store it is running
    /// on: `aborted_pre_exec` is a caught window (nothing written, attempt
    /// retried), `compensated` is an undone one, and `witness_unavailable` /
    /// `fingerprint_unavailable` say the store denies `CLIENT ID` (inside
    /// `MULTI` and at handshake respectively), so the corresponding detection
    /// layer is disarmed for that attempt — a deployment-configuration signal,
    /// expected to read as either zero or ~every watched commit, never an
    /// incident. `committed_unwatched` is the
    /// residual — a commit the server confirms ran on the wrong connection, with
    /// no undo available — and it is the only cell whose consequence reaches a
    /// client. Read it next to
    /// [`Self::storage_watch_unwatch_failures_total`] and
    /// [`Self::valkey_txn_dirty_checkouts_total`]: all three are downstream of
    /// the same event (a `txn`-pool connection changing under an open
    /// transaction), so a correlated rise across the family points at the
    /// backing store or proxy rather than at any one call site.
    pub storage_watch_voided_total: Counter<u64>,
    /// Total `txn`-pool checkouts whose hygiene probe found the slot carrying
    /// leaked transaction state, by `reason` (b.7td).
    ///
    /// **This is the corruption detector the service owns.** Before it, the
    /// only evidence that a pooled connection was carrying an open `MULTI` was
    /// fred's own `Failed to refresh credentials` warning — a *second-order*
    /// symptom raised by the periodic credential re-`AUTH` landing on the
    /// corrupt connection and being rejected by the server with "AUTH is not
    /// allowed in transactions". That made the detector an accident of the auth
    /// cadence: it fired only on the `accessTokenFile` / `iam` modes, only when
    /// a refresh happened to pick the bad slot, and it named the refresh rather
    /// than the leak. Every checkout now probes, so the signal exists on every
    /// auth mode and points at the actual fault.
    ///
    /// **Page on `leaked_multi` / `unexpected_reply` only; alert on the other
    /// three by *sustained rate*.** The reason label is not decoration — it
    /// splits what the probe proved from what it merely could not disprove:
    ///
    /// - `leaked_multi` and `unexpected_reply` rest on a reply that was actually
    ///   received and can only mean one thing, so a healthy fleet reads zero on
    ///   them indefinitely and any increase is actionable — transactions are
    ///   being abandoned mid-flight somewhere upstream of the checkout.
    /// - `probe_timeout` and `probe_error` say only "no answer arrived", which a
    ///   backing store having a bad minute produces without anything being
    ///   structurally wrong. Occasional ticks under store stress are the check
    ///   working; only a *sustained* rate is a signal, and it points at the
    ///   store or the network rather than at the pool. Read them next to
    ///   `valkey_errors_total` and `health_probe_failures_total{backend="valkey"}`.
    /// - `topology_reply` (b.289) is the one cell **expected to be non-zero**,
    ///   and the "healthy fleet reads zero on every cell" reading does not
    ///   apply to it. It ticks while a cluster reshards, a replica is loading,
    ///   or a primary has been demoted — events an operator usually caused —
    ///   because fred's router intercepts only three of the seven prefixes
    ///   this classifier tests, each on a case-**sensitive** match: `MOVED`
    ///   and `ASK` only in their canonical rendering (exactly three
    ///   space-separated tokens) and only on a clustered client, and
    ///   `CLUSTERDOWN` on every read frame with no token-count condition at
    ///   all, so casing is its only escape. The other four are never
    ///   intercepted, so every node-state error frame outside those
    ///   conditions is handed straight through to the probe. Correlate a
    ///   burst with the topology change; only a rate *sustained past* one is
    ///   a signal.
    ///
    /// The operator pages follow this split: `docs/service/docs/monitoring.md`
    /// warns only on a `topology_reply` rate sustained past a cluster change,
    /// and `docs/service/docs/metrics-reference.md` lists every cell; keep them
    /// in step.
    ///
    /// A tick is **not** by itself proof that a connection was rebuilt. The
    /// counter records the *verdict*; the repair is rate-limited on both sides
    /// of the split. For the three unknown-state reasons a pool-wide streak gate
    /// (`storage::txn_checkout::should_recycle_unknown_state`) admits the first
    /// of a streak and every Nth after it, so a store-wide latency event cannot
    /// turn every checkout in the pool into a quit + reconnect storm. For the
    /// two state-proof reasons a **per-node** breaker (b.289 sub-task 4)
    /// (`storage::txn_checkout::state_proof_repair_is_ineffective`) stops
    /// recycling once several rebuilds in a row against THAT node have failed
    /// to change the verdict — see
    /// [`Self::valkey_txn_repair_ineffective_total`]. A suppressed
    /// repair still ticks here; it logs at `debug!` with `suppressed=true`
    /// rather than at `warn!`, because the regime the unknown-state gate exists
    /// for is precisely the one where a per-checkout WARN floods the log
    /// pipeline.
    ///
    /// Read the whole family next to
    /// `storage_watch_unwatch_failures_total{site="drop"}`: that counter is the
    /// disambiguating instrument for the entangled failure mode where a
    /// connection that goes unresponsive mid-transaction never completes its
    /// UNWATCH teardown either.
    ///
    /// A probe failure whose fred error says the connection is already gone
    /// (`Canceled` / `IO`, or fred's routing miss when the router holds no
    /// connection; `storage::txn_checkout`'s
    /// `residue_died_with_the_connection`) is **not** counted, on the same
    /// reasoning as the UNWATCH counter above: a torn-down connection carries
    /// no residue for the next borrower, and counting it would page on routine
    /// failover / proxy-recycle churn.
    pub valkey_txn_dirty_checkouts_total: Counter<u64>,
    /// Total `txn`-pool checkouts whose **state-proof** dirty verdict was left
    /// unrepaired because recycling the slot demonstrably was not clearing it,
    /// by `reason` (b.7td round 3).
    ///
    /// The ordinary `leaked_multi` / `unexpected_reply` path recycles the slot
    /// unconditionally, which is right for real residue: one rebuild clears it
    /// and the next checkout reads `Clean`. It is *self-amplifying* when the
    /// verdict comes from the classifier and the server disagreeing rather than
    /// from residue — a managed proxy that renders the `DISCARD` error without
    /// naming `DISCARD`, say, reads as `unexpected_reply` forever, a freshly
    /// built connection answers identically, and every checkout in the pool
    /// recycles without bound. After several consecutive state-proof verdicts
    /// (`storage::txn_checkout::state_proof_repair_is_ineffective`) the repair
    /// is withheld and this counter ticks instead, on every withheld verdict, so
    /// the rate reads "how often are we handing out a slot we cannot fix".
    ///
    /// **A non-zero rate is a different investigation from its sibling
    /// counter.** `valkey_txn_dirty_checkouts_total` says the probe found slots
    /// unfit; this one says rebuilding them changed nothing, which points at the
    /// *classifier* (or the endpoint's reply rendering) rather than at leaked
    /// transaction state. A `Clean` checkout against the affected node re-arms
    /// the repair, so a genuinely transient burst clears itself and stops
    /// ticking here.
    ///
    /// **The contract is per node, not per pool** (b.289 sub-task 4). The
    /// streak and the breaker it feeds are keyed by the cluster node the
    /// checkout's probe was pinned to, so a tick here means "checkout hygiene
    /// is switched off for one node", not for the whole pool — and a `Clean`
    /// checkout against a *different* node no longer resets it, which is what
    /// makes the breaker able to latch on a single misbehaving primary at all.
    /// **The node is named in the accompanying `error!` log's `node` field,
    /// deliberately NOT as a label on this counter**: node identity is
    /// unbounded cardinality over a deployment's lifetime, and the counter's
    /// question is "how often", not "where".
    ///
    /// Verdicts whose node cannot be named — a centralized or proxied
    /// endpoint, which never has one, and a clustered one transiently while
    /// fred's routing view is being rebuilt — share a single fallback bucket,
    /// and **any** `Clean` checkout re-arms that bucket, whichever node it was
    /// pinned to. So on a single-endpoint deployment the contract reads exactly
    /// as the pre-b.289 pool-wide one did (one bucket, any `Clean` resets it),
    /// and on a cluster a transient unnameable window cannot leave a streak
    /// that nothing is able to clear.
    ///
    /// Only [`TXN_STATE_PROOF_REASONS`] can appear: the unknown-state verdicts
    /// have their own, differently-shaped rate limit (a re-arming pool-wide
    /// streak gate, which never stops repairing) and never reach this breaker.
    pub valkey_txn_repair_ineffective_total: Counter<u64>,
    /// Total `force_reconnection()` calls a stream-multiplexer reader issued
    /// against its own connection, by `phase`
    /// ([`CONSUMER_FORCED_RECONNECT_PHASES`]): `xread`, `wake_ack`, `probe` and
    /// `keepalive` are wedge verdicts (an XREAD timeout, a woken XREAD that
    /// did not return, an idle→active probe that did not answer, an idle
    /// keepalive tick that failed), `permanent` a heal after a whole-reader
    /// failure. The verdict is counted whichever remedy follows: on a cluster,
    /// one that finds the reader's node no longer owns its slot retires the
    /// connection instead of calling `force_reconnection()`, and the reader
    /// retargets; an idle reader whose node is no longer a primary just
    /// retires it.
    /// Distinct from `valkey_reconnects_total`, which counts fred's own
    /// reconnect events across all pools: this one counts only the
    /// reader-initiated self-heals, so a non-zero rate on a wedge phase is a
    /// direct signal that reader connections are being silently invalidated
    /// upstream (e.g. a managed proxy dropping backend sessions at token
    /// rotation). `permanent` instead tracks the whole-reader subset of
    /// `valkey_consumer_permanent_exits_total{phase="xread"|"probe"}` (a key
    /// that fails alone ends its subscriptions without a reconnect).
    ///
    /// Near-zero in steady state on every phase.
    pub valkey_consumer_forced_reconnects_total: Counter<u64>,
    /// Total live layer-stream `XREAD BLOCK`s a reader saw end in `-UNBLOCKED`
    /// (a `CLIENT UNBLOCK … ERROR`) and re-issued from the durable cursor
    /// without charging the transient budget (b.9yf).
    ///
    /// Unlabeled: the only producer is a reader's XREAD.
    ///
    /// Expected to stay at about zero in every auth mode, clustered or not
    /// (b.19i): it ticks only when something outside the service unblocks a
    /// live reader read. No command of this process's interrupts one — fred's
    /// periodic credential refresh no longer reaches a reader connection, and a
    /// reader re-`AUTH`s its own connection between reads (see
    /// `stream_mux_reauths_total`). The one case left for the readers'
    /// `Blocking::Interrupt` (the reader's next command after a client-side
    /// XREAD timeout that did not force a reconnect, while the old read is
    /// still blocked on the server) interrupts that abandoned read, whose
    /// `-UNBLOCKED` fred hands to the timed-out command's dropped caller: it
    /// costs a backchannel connection and never ticks this counter. A non-zero
    /// rate means something outside the service is unblocking reader
    /// connections.
    pub valkey_consumer_unblocked_retries_total: Counter<u64>,
    /// Total stream-reader tear-downs caused by a **permanent** fred error, by
    /// `phase` ([`CONSUMER_PERMANENT_EXIT_PHASES`]) (b.c7a).
    ///
    /// Distinct from the forced-reconnect counter in what it means: a forced
    /// reconnect is a self-heal in progress, this is a reader giving up on the
    /// subscriptions it holds (or, for a key that fails alone, on that key's
    /// subscriptions). Every increment ends subscriptions with
    /// `Code::Unavailable`, so a sustained non-zero rate is a direct signal that
    /// subscribers are being torn down by an error class the transient taxonomy
    /// does not cover (auth, protocol, parse, or an unrecognised fred message) —
    /// which, before b.c7a widened the classifier, is exactly how a
    /// post-token-rotation routing gap silently killed live subscriptions.
    ///
    /// Expected to be zero in steady state. The label cell is pre-registered so
    /// a first scrape shows the zero rather than a missing series.
    pub valkey_consumer_permanent_exits_total: Counter<u64>,
    /// `SubscribeReadLayer` subscriptions holding a stream reader on the XREAD
    /// multiplexer: registered streams, and reader reservations whose stream is
    /// not registered yet (a catch-up that holds its reader through its
    /// capacity waits and registers at its proven read's cut). Unlike
    /// `stream_mux_streams` and `valkey_pool_in_use{pool="subscriber_pool"}`,
    /// it counts a reservation.
    pub stream_mux_subscriptions: UpDownCounter<i64>,
    /// Distinct stream keys the multiplexer's readers are reading.
    pub stream_mux_streams: UpDownCounter<i64>,
    /// Multiplexed `XREAD BLOCK` commands issued.
    pub stream_mux_xreads_total: Counter<u64>,
    /// Reader wake-ups (an XADD on a reader's wake key), by `outcome`.
    pub stream_mux_wakes_total: Counter<u64>,
    /// Idle keepalive ticks run by connected readers with nothing to read, by
    /// `result` ([`KEEPALIVE_RESULTS`]). Every cell is pre-registered.
    pub stream_mux_keepalives_total: Counter<u64>,
    /// Credential re-`AUTH`s a stream reader sent on its own connection, by
    /// `result` ([`REAUTH_RESULTS`]) (b.19i). Every cell is pre-registered.
    ///
    /// Under the passwordless auth modes fred schedules no periodic re-`AUTH`
    /// on a reader client; each connected reader takes the provider's tick
    /// itself, waking an in-flight XREAD when a credential is due and sending
    /// the `AUTH` between reads. `ok`: the `AUTH` was answered. `failed`: the
    /// tick could not fetch a credential (also logged at DEBUG; the pooled
    /// clients' refresh warnings and readiness already report a credential
    /// outage), or the `AUTH` failed or did not answer in time (one WARN per
    /// failure). A failure keeps the connection and charges nothing.
    ///
    /// Under `accessTokenFile` expect one to three `ok` per connected reader per
    /// token rotation, and none between rotations (a gated tick sends
    /// nothing): after a rotation every tick inside the 150 s fan-out window
    /// carries the credential, and a reader ticks every 60–90 s, so it takes
    /// one to three ticks there — the at-most-three bound of SDD §15.4. On a
    /// busy reader each of those is also a credential wake, unless a wake is
    /// already outstanding for that read. Under `iam` expect
    /// one per connected reader per `[300 s, 360 s]`; under `static` none.
    pub stream_mux_reauths_total: Counter<u64>,
    /// Reader connections opened, by `cause` ([`READER_CONNECT_CAUSES`]) and
    /// `result` ([`READER_CONNECT_RESULTS`]). A reader connects when it first
    /// serves a subscription, never at startup; on a cluster it connects only
    /// to the primary owning its bound hash slot, and replaces that connection
    /// (`retarget`) when the slot it serves is owned elsewhere. Every cell is
    /// pre-registered.
    pub stream_mux_reader_connects_total: Counter<u64>,
    /// Readers holding a connection now. Rises as readers first serve a
    /// subscription and falls when one retires its connection (a retarget's
    /// old connection, an idle reader whose node stopped being a primary, and
    /// shutdown).
    pub stream_mux_readers_connected: UpDownCounter<i64>,
    /// Subscriptions refused because no reader could take their hash slot.
    pub stream_mux_bind_refusals_total: Counter<u64>,
    /// Stream keys isolated after failing a multiplexed XREAD on their own.
    pub stream_mux_poison_keys_total: Counter<u64>,
    /// Subscriptions ended because they fell behind, by `bound`
    /// ([`OVERFLOW_BOUNDS`]): the next entry would have taken the undelivered
    /// backlog past the hard cap in entries or bytes (an empty backlog always
    /// takes one entry), or — only once the subscription has started draining
    /// — the backlog stayed at or above the soft threshold in entries or bytes
    /// past the delivery watchdog. `bytes` counts only the endings where a byte limit
    /// was reached and the entry limit was not. Both cells are pre-registered.
    pub stream_mux_overflow_kills_total: Counter<u64>,
    /// Head checks (`XRANGE key - + COUNT 1`) a reader ran because an XREAD
    /// reply's first entry for a key was past the entry after the id the key
    /// was read from, by `result` ([`HEAD_CHECK_RESULTS`]). Every cell is
    /// pre-registered.
    pub stream_mux_head_checks_total: Counter<u64>,
    /// Bytes carried by each multiplexed XREAD reply that returned
    /// layer-stream entries: per entry its id plus its `payload` and `ompf`
    /// field values, summed over the reply's layer-stream keys (the reader's
    /// wake key excluded). Each read's `COUNT` is sized before it is issued so
    /// that this aims at `storage::stream_mux::XREAD_REPLY_TARGET_BYTES`; the
    /// bound is soft, so an entry larger than its key's estimate over-commits
    /// one reply. An empty read (its BLOCK elapsed) and a reply carrying only
    /// the reader's wake entry (a join's wake) record nothing. A histogram, so
    /// no series exists before the first reply.
    pub stream_mux_xread_reply_bytes: Histogram<f64>,
    /// Gauge of Valkey pool use, by pool (`txn` | `subscriber_pool` |
    /// `maintenance` | `snapshot`) (b.3z7 H-obs-3, b.mfn). A non-monotonic sum,
    /// exported to Prometheus as a gauge. Inc'd on acquire, dec'd from the RAII
    /// `PoolUseToken::Drop`; on `subscriber_pool` a stream reader holds one
    /// token while it has at least one stream. The multiplexed `maintenance`
    /// and `snapshot` pools count logical uses, not connections.
    pub valkey_pool_in_use: UpDownCounter<i64>,
    /// Gauge of callers currently parked waiting, by `pool`
    /// ([`CHECKOUT_POOLS`]) (b.f4f).
    ///
    /// On `txn` that is the FIFO permit queue; on `subscriber_pool` it is the
    /// bounded `SubscribeReadLayer` admission queue, whose depth
    /// `LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS` caps and which turns into
    /// `subscriber_admission_total{result="refused_saturated"}` when it times out.
    pub valkey_pool_waiters: UpDownCounter<i64>,
    /// `txn` exclusive-pool checkout latency in seconds, by `pool` (b.f4f).
    ///
    /// Covers permit wait plus the non-blocking next-free scan. A p99 that
    /// climbs is pool saturation, not Valkey latency — the scan itself never
    /// awaits.
    pub valkey_pool_checkout_seconds: Histogram<f64>,
    /// Total `SubscribeReadLayer` admission decisions, by `result`
    /// ([`ADMISSION_RESULTS`]) (b.f4f).
    ///
    /// The only way to tell the two capacity refusals apart: both surface to
    /// the client as `Code::Unavailable`, matching the one saturation code this
    /// codebase already uses (the b.c7a contract). `cancelled` covers a deleted
    /// session (`Code::NotFound`), a shutdown (`Code::Unavailable`) and a
    /// session missing from the live store (`Code::Unavailable`).
    pub subscriber_admission_total: Counter<u64>,
    /// How long a `SubscribeReadLayer` waited for admission, in seconds (b.f4f).
    ///
    /// Unlabelled, and recorded on every outcome including the uncontended fast
    /// path (which records ~0) and the refusals, so the histogram's count equals
    /// `sum(subscriber_admission_total)`.
    pub subscriber_admission_wait_seconds: Histogram<f64>,
    /// Total `CreateLayer` / `UpdateLayer` upload admission and growth
    /// decisions, by `rpc` ([`UPLOAD_RPCS`]) and `result`
    /// ([`UPLOAD_ADMISSION_RESULTS`]). One admission decision per admitted or
    /// refused call; an admitted upload refused receive growth later also
    /// counts its `refused_growth`, and one whose processing wait ends by
    /// shutdown or a dropped call its `cancelled`.
    pub upload_admission_total: Counter<u64>,
    /// Seconds an upload waited for admission, by `rpc`, recorded on every
    /// admission decision including the uncontended fast path (which records
    /// ~0). Not pre-registered: a histogram series appears with its first
    /// observation.
    pub upload_admission_wait_seconds: Histogram<f64>,
    /// Bytes reserved within the upload budget right now, by every admitted
    /// upload except the overflow slot's holder. Never above
    /// `upload_budget_bytes`.
    pub upload_reserved_bytes: UpDownCounter<i64>,
    /// The overflow slot holder's charge in bytes right now: the one upload
    /// running beyond the budget; 0 when the slot is free.
    pub upload_overflow_bytes: Gauge<u64>,
    /// Uploads whose processing step is queued for room in the budget or for
    /// the overflow slot right now.
    pub upload_overflow_waiters: UpDownCounter<i64>,
    /// Total uploads that ran in the overflow slot, beyond the budget, by
    /// `rpc` ([`UPLOAD_RPCS`]). Counted when a processing step takes the slot
    /// — once per "upload runs beyond the upload budget" INFO line — never for
    /// a slot grant a shutdown or a dropped call ended first.
    pub upload_overflow_runs_total: Counter<u64>,
    /// Seconds an upload's processing step waited for room in the budget or
    /// for the overflow slot, by `rpc`, recorded once per waiting step that
    /// grew the reservation — the handler's two steps, a structural commit's
    /// recharge, and the `UpdateLayer` checkpoint-recovery re-issue's hold of
    /// one extra copy of its external values (one per re-issue with external
    /// values) — ~0 when granted at once. A commit's non-waiting
    /// growth of its charge records none. Bounded by the RPC deadline. Not
    /// pre-registered.
    pub upload_processing_wait_seconds: Histogram<f64>,
    /// Admitted uploads right now, by `phase` ([`UPLOAD_PHASES`]).
    pub uploads_in_flight: UpDownCounter<i64>,
    /// Uploads queued for admission right now; bounded by
    /// `LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS`.
    pub upload_admission_waiters: UpDownCounter<i64>,
    /// The upload budget in bytes, derived from the pod memory limit
    /// (`config::memory_plan`), so a dashboard can plot
    /// `upload_reserved_bytes` against it.
    pub upload_budget_bytes: Gauge<u64>,
    /// The pod memory limit every pod-memory budget is derived from, in
    /// bytes, by service and `source` (`config::pod_memory`): every source
    /// series is recorded at startup, the one in use carrying the limit and
    /// the others 0. `source="reference"` carrying the value means the
    /// replica found no readable limit and is sized for the 5 GiB reference.
    pub pod_memory_limit_bytes: Gauge<u64>,
    /// The catch-up memory budget `T` in bytes, derived from the pod memory
    /// limit; recorded at startup. Unlabelled, like
    /// `catchup_build_budget_reserved_bytes`, which is read against it.
    pub catchup_build_budget_bytes: Gauge<u64>,
    /// The retirement memory allowance `R_ref` in bytes (the pod memory plan's
    /// `L / 500`, a margin term), recorded at startup. Unlabelled, like
    /// [`Self::retirement_budget_reserved_bytes`], which is read against it.
    pub retirement_budget_bytes: Gauge<u64>,
    /// The structural write-planning cache's resident budget in bytes,
    /// `C × (1 + φ)`, derived from the pod memory limit, by service; what
    /// `live_edit_structural_cache_weight_bytes` is bounded by. Recorded at
    /// startup.
    pub structural_cache_budget_bytes: Gauge<u64>,
    /// The subscription ceiling `S`, derived from the pod memory limit;
    /// `live_edit_subscriber_admitted` is compared against it. Recorded at
    /// startup. Unlabelled, like `live_edit_subscriber_admitted`.
    pub subscriber_max_subscriptions: Gauge<u64>,
    /// Seconds a chained `UpdateLayer` waited in the chain gate for its
    /// predecessor, by `outcome` ([`CHAIN_OUTCOMES`]): one observation per
    /// chained call that reached the gate, at its exit, the drop of a waiting
    /// call included. Not pre-registered: a histogram series appears with its
    /// first observation.
    pub update_layer_chain_wait_seconds: Histogram<f64>,
    /// Chain-gate probes of a writer's record, by `result`
    /// ([`CHAIN_PROBE_RESULTS`]). Bounded per replica by the gate's probe
    /// ceiling. Every cell is pre-registered.
    pub update_layer_chain_probes_total: Counter<u64>,
    /// Chained `UpdateLayer`s waiting in the chain gate right now.
    pub update_layer_chain_waiters: UpDownCounter<i64>,
    /// The upload-budget bytes the waiting chained `UpdateLayer`s hold right
    /// now: each one's reservation at gate entry (its fixed charge plus its
    /// content). Read against `upload_budget_bytes`.
    pub update_layer_chain_waiting_bytes: UpDownCounter<i64>,
    /// Seconds from the predecessor's commit (its record's server timestamp)
    /// to the gate seeing it, on a `ready` exit after at least one sleep. It
    /// includes clock skew between replicas. Not pre-registered.
    pub update_layer_chain_detect_lag_seconds: Histogram<f64>,
    /// `LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS` slots currently held,
    /// one per unit of catch-up work. A catch-up of its own holds one only
    /// across its backing-store reads and its build — and, in rollback mode
    /// (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`) on a same-object
    /// reconnect, the cluster stream-reader wait. A shared fenced-read flight
    /// (`storage::snapshot_flights`) holds one from its read stage to its
    /// last attached catch-up's build, fence `RETRY_AFTER` sleeps, its
    /// permit and memory waits and its build turns included, and each of its
    /// extra builds holds one more for that build; the catch-ups attached to
    /// it hold none. No slot is held across a send.
    pub subscribe_catchups_inflight: UpDownCounter<i64>,
    /// Catch-ups parked waiting for a slot (a flight's slot wait included).
    pub subscribe_catchups_waiting: UpDownCounter<i64>,
    /// Seconds a catch-up waited for its slot, by `result`
    /// ([`CATCHUP_WAIT_RESULT_ACQUIRED`] | [`CATCHUP_WAIT_RESULT_CANCELLED`] |
    /// [`CATCHUP_WAIT_RESULT_TIMED_OUT`]), recorded on every acquisition
    /// including the uncontended fast path. A catch-up of its own waits until
    /// its capacity budget runs out (`timed_out`). A shared fenced-read
    /// flight's slot wait (one observation per flight) has no deadline — each
    /// catch-up attached to it is charged its share and is refused alone — so
    /// it ends only `acquired` or `cancelled` and can run past the top bucket
    /// into `+Inf`; an extra build's non-blocking slot records nothing. Not
    /// pre-registered: a histogram series appears with its first observation.
    pub subscribe_catchup_wait_seconds: Histogram<f64>,
    /// Catch-up snapshot reads currently holding a read permit, by `class`
    /// ([`CATCHUP_SNAPSHOT_READ_CLASSES`]) (Spec SDD SR-7.1, SR-8.5). The
    /// `atomic` class is bounded by `LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY`:
    /// a read holds its permit only across its atomic read's `EVALSHA` — a
    /// catch-up's, or one commit's of a legacy in-window replay, always inside
    /// the catch-up slot (the steady state never reads layer state). The
    /// `fenced` class is bounded by `LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY`:
    /// a shared fenced-read flight's read stage holds its permit from before a
    /// fence acquire through the fenced read, the fence's release and the
    /// memory adjust after it, inside the flight's slot, and gives it back
    /// during every `RETRY_AFTER` sleep. Both cells are seeded at zero.
    pub catchup_snapshot_reads_in_flight: UpDownCounter<i64>,
    /// Seconds a snapshot read waited for its read permit, by `consumer`
    /// (`catchup` for a catch-up's proven read and for a shared fenced-read
    /// flight's read; `catchup_legacy_per_commit` for a legacy in-window
    /// replay's per-commit read, which an empty commit does not take;
    /// `structural_cache` for the structural cache's fenced cold start;
    /// `structural_cache_audit` for its sampled audit's fenced read) and by
    /// `class` ([`CATCHUP_SNAPSHOT_READ_CLASSES`], the permit's class, so
    /// fenced waits never mix with atomic ones) (Spec SDD SR-7.1, SR-8.5),
    /// recorded on every acquisition whatever its outcome — the permit taken
    /// (the uncontended fast path records ~0), refused at the subscription's
    /// capacity budget, the gate closed at shutdown, or the wait dropped by a
    /// subscription that stopped waiting. An atomic wait is one catch-up's,
    /// ended at its capacity budget. A fenced wait is a flight's: it has no
    /// deadline (each attached catch-up is charged its share and leaves alone
    /// when its own budget runs out), so it ends only acquired, at shutdown,
    /// or dropped when the flight's last catch-up left, and it can run past
    /// the top bucket into `+Inf`. A `structural_cache` wait is a fenced cold
    /// start's (`class="fenced"` only), ended at its waiting cap, which leaves
    /// its call time for a WATCH attempt. A `structural_cache_audit` wait is
    /// the audit's (`class="fenced"` only), which never waits: a permit not
    /// free at once skips the audit. Every other wait is a catch-up's:
    /// the steady state never reads layer state. Not pre-registered: a
    /// histogram series appears with its first observation.
    pub catchup_snapshot_read_wait_seconds: Histogram<f64>,
    /// Bytes of catch-up memory reservations currently granted by the
    /// per-replica `MemoryBudget` (`T`, derived from the pod memory limit and
    /// read against [`Self::catchup_build_budget_bytes`]; Spec SDD SR-7.2,
    /// SR-8.5), in the budget's KiB unit × 1024. Moves with
    /// every grant, shrink and guard drop, so it reads zero once every
    /// reservation is released. Pre-registered at zero.
    pub catchup_build_budget_reserved_bytes: UpDownCounter<i64>,
    /// Bytes of forced, never-blocking top-ups the `MemoryBudget` currently
    /// carries beyond its grants (SR-7.2's over-commit), in the budget's KiB
    /// unit × 1024. Repaid when each top-up guard drops. Unlabelled until
    /// payload guards add a `kind` (Spec SDD SR-8.5). Pre-registered at zero.
    pub catchup_build_budget_overcommit_bytes: UpDownCounter<i64>,
    /// Seconds a waiting `MemoryBudget` reservation took to end, whatever the
    /// outcome (granted at once or after queueing, refused at its deadline,
    /// or ended by a closed budget or a dropped wait). A non-blocking grant and
    /// a top-up never wait and record nothing. Not pre-registered: a histogram
    /// series appears with its first observation.
    pub catchup_build_wait_seconds: Histogram<f64>,
    /// `MemoryBudget` admission decisions by `result`
    /// ([`CATCHUP_BUILD_ADMISSION_RESULTS`]): one per waiting reservation, and
    /// one `immediate` per successful non-blocking grant (a declined one is not
    /// counted). Every cell is seeded at zero.
    pub catchup_build_admission_total: Counter<u64>,
    /// Bytes of retirement reservations currently granted by the retirement
    /// allowance `R_ref` (the `LayerStore`'s own `MemoryBudget`, read against
    /// [`Self::retirement_budget_bytes`]): external-value clean-up of a
    /// retiring object — `DeleteLayer`, the `CreateLayer` overwrite's cleanup
    /// and the orphan sweep — each holding its page (sized from its size
    /// read's largest structural index, at most the page bound `W`, and while
    /// a release is owed at least a row per blob-size read in flight) plus a
    /// charge per reference it releases, then its page alone through its
    /// teardown (`storage::retirement::retirement_charge_bytes`).
    /// In the budget's KiB unit × 1024; reads zero once every reservation is
    /// released. Pre-registered at zero.
    pub retirement_budget_reserved_bytes: UpDownCounter<i64>,
    /// Bytes of forced, never-blocking top-ups the retirement allowance
    /// currently carries beyond its grants: references a walk found beyond
    /// those `HLEN obj:ext` counted, a page widened to the page bound `W`
    /// because an index grew after the size read took its cardinality, and a
    /// teardown's set-index step answered larger than its page. In the budget's KiB unit × 1024; repaid when
    /// each top-up guard drops. Pre-registered at zero.
    pub retirement_budget_overcommit_bytes: UpDownCounter<i64>,
    /// Seconds a retirement reservation waited on the retirement allowance,
    /// by `outcome` ([`RETIREMENT_WAIT_OUTCOMES`]), one observation per
    /// reservation, the uncontended fast path included. Not pre-registered: a
    /// histogram series appears with its first observation.
    pub retirement_wait_seconds: Histogram<f64>,
    /// Catch-up builds currently running (Spec SDD SR-8.5): the builds of
    /// catch-ups attached to a shared fenced-read flight, each counted from
    /// the moment it is given its build turn (the flight's own, or an extra
    /// build's) until it gives the turn back. Pre-registered at zero.
    pub catchup_builds_in_progress: UpDownCounter<i64>,
    /// Catch-ups that took part in a shared snapshot-read flight, by `role`
    /// ([`CATCHUP_SNAPSHOT_FLIGHT_ROLES`]; Spec SDD SR-8.4): `leader` for the
    /// one that created the flight, `joined_read` for each that attached to
    /// it. One per catch-up per flight. The read hit rate is `joined_read` ÷
    /// (`leader` + `joined_read`). Every cell is seeded at zero.
    pub catchup_snapshot_flight_total: Counter<u64>,
    /// `SubscribeReadLayer` subscriptions currently admitted: +1 when a
    /// subscription is admitted, −1 when its admission is dropped at the end
    /// of the RPC (every exit path, task abort included). The capacity signal
    /// to compare against [`Self::subscriber_max_subscriptions`]. Unlike
    /// `stream_mux_subscriptions` it also counts a subscription that holds no
    /// stream reader: one waiting for a stream reader (cluster mode); in
    /// rollback mode (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`), one waiting
    /// for a catch-up slot on a same-object reconnect; and one between
    /// consumers during a re-bootstrap. (A brand-new catch-up, and a
    /// same-object one in the default snapshot mode, reserves its reader
    /// before its slot wait, so both gauges count it while it waits.)
    pub subscriber_admitted: UpDownCounter<i64>,
    /// `SubscribeReadLayer` subscriptions whose first catch-up that reads layer
    /// state — every regime except `up_to_date` — has been selected and whose
    /// first caught-up ack is not yet enqueued: +1 at regime selection, −1 at
    /// that ack or when the subscription ends before it. Counts a catch-up
    /// whatever it is doing — waiting for a slot or a read permit, reading,
    /// building or sending — so it is the queue the autoscaler drains, and it
    /// does not depend on how catch-ups are scheduled. A steady-state
    /// re-bootstrap is not a first catch-up and is not counted.
    pub subscribe_catchups_pending: UpDownCounter<i64>,
    /// In-process mirrors of the counters `/scaling/v1` serves, moved beside
    /// their metrics (see [`ScalingCounters`]).
    scaling_counters: ScalingCounters,
    /// Total `SubscribeReadLayer` subscriptions that ended, by `reason`
    /// ([`STREAM_EXIT_REASONS`]) (b.f4f).
    ///
    /// Complete over the handler's terminal arms, so `sum by (reason)` equals
    /// total exits. Admission refusals are **not** counted here — no stream was
    /// ever established — they are `subscriber_admission_total{result}`.
    pub subscriber_stream_exits_total: Counter<u64>,
    /// Commit-to-hand-off lag of steady-state live deliveries, in seconds, by
    /// `stage` ([`DELIVERY_LAG_STAGES`]). Measures exactly the deliveries
    /// [`Self::subscriber_updates`] counts: never catch-up content, self-echo,
    /// duplicates or an incomplete send.
    ///
    /// **One observation per subscription per window, the window's maximum**
    /// — the worst delivery in the window, not a per-delivery percentile. A
    /// window closes at 64 deliveries, after about a second and at stream end;
    /// an `enqueued` window also before a re-bootstrap
    /// (`grpc::subscriber_delivery`). The lag is the
    /// reading replica's clock minus the writing replica's commit stamp, so
    /// cross-replica clock skew enters it; a negative difference is clamped to
    /// zero, and a commit with no stamp is not measured.
    pub subscriber_delivery_lag_seconds: Histogram<f64>,
    /// `SubscribeReadLayer` sends that found all 32 data slots of the response
    /// channel full, by `phase` ([`SEND_PHASES`]): the server-side signal of an
    /// HTTP/2 flow-control stall, since the service's HTTP/2 layer stops
    /// pulling from the channel once the peer's stream window is used up.
    /// Counted when the stall begins; both cells are seeded at zero.
    pub subscriber_send_stalls_total: Counter<u64>,
    /// How long a stalled send (one [`Self::subscriber_send_stalls_total`]
    /// counted) waited, in seconds, by `phase` ([`SEND_PHASES`]), recorded on
    /// every outcome: sent, the slow-reader deadline, shutdown, session
    /// cancel, client disconnect, or the task being dropped. A send that found
    /// room is not recorded.
    pub subscriber_send_wait_seconds: Histogram<f64>,
    /// Depth of a subscription's stream-multiplexer backlog in entries,
    /// sampled by the subscription task as each live delivery finishes
    /// enqueuing (never by the stream reader); one observation per
    /// subscription per window, the window's maximum (see
    /// [`Self::subscriber_delivery_lag_seconds`]).
    pub subscriber_backlog_entries: Histogram<f64>,
    /// Depth of a subscription's stream-multiplexer backlog in bytes, sampled
    /// and windowed as [`Self::subscriber_backlog_entries`].
    pub subscriber_backlog_bytes: Histogram<f64>,
    /// Total background-task panics observed by the shutdown supervisor, by
    /// task name (b.3z7 H-obs-3).
    ///
    /// It ticks only in the shutdown join (`report_background_task_result` in
    /// `main.rs`), so a task that panics mid-run is not counted until the
    /// process shuts down, and the count exists only for the moments before
    /// exit: `/metrics` serves until `metrics_task.abort()`, so a scrape rarely
    /// lands in that window, though the final OTLP flush may still export it.
    /// No alert uses it; the `background task failed during shutdown` ERROR
    /// line is the signal.
    pub background_task_panics_total: Counter<u64>,
    /// Total external-value GC sweep batch failures, by error `kind`
    /// (b.kf1 medium).
    pub gc_sweep_batch_errors_total: Counter<u64>,
    /// Total health-probe failures, by `backend` label (`valkey`) (b.u96 item
    /// 8; auth-backend labels removed in b.9f4).
    pub health_probe_failures_total: Counter<u64>,
    /// Total per-layer checkpoint write attempts by the periodic loop, by
    /// `result` (`success` | `error` | `skipped`) and `kind` (SDD SR-2.2).
    ///
    /// The `kind` vocabulary is pinned by [`CHECKPOINT_WRITE_KINDS`] —
    /// `layer` | `layer_tail` | `session` | `counters_sidecar` — and the full
    /// `result × kind` grid is zero-seeded at construction.
    pub checkpoint_writes_total: Counter<u64>,
    /// Histogram of per-layer **pending age** — `now - last_checkpoint_ms`,
    /// sampled at the periodic loop's tick time, for layers that have commits
    /// the newest durable generation does not yet carry (SDD SR-2.2, b.jq3).
    /// Buckets `[5s,15s,30s,1m,2m,5m,10m,30m,1h,2h]`.
    ///
    /// Two scoping rules make this a checkpoint-freshness signal rather than an
    /// idleness signal, and both are load-bearing:
    ///
    /// - **Only layers with un-persisted commits** (`seq != last_checkpoint_seq`)
    ///   sample. A fully-checkpointed idle layer's age grows without bound on a
    ///   perfectly healthy fleet, so including it would make a rising p99 mean
    ///   "traffic stopped", which is exactly the false positive that made the
    ///   predecessor metric (`live_edit_checkpoint_lag_seconds`, which measured
    ///   time-since-last-*commit*) unusable for alerting.
    /// - **Only layers with a durable generation** (`last_checkpoint_ms != 0`).
    ///   "Never checkpointed" has no finite age; it is counted on
    ///   [`Self::checkpoint_layers_never_written`] instead. Folding a sentinel
    ///   in would poison `_sum` — `now - 0` is ~1.77e9 s.
    ///
    /// Scoped this way the metric is self-bounding: the loop's own due-gate
    /// guarantees a pending layer is snapshotted within
    /// `LIVE_EDIT_CHECKPOINT_INTERVAL_SECS` + one tick + assembly time, so the
    /// distribution measures precisely the invariant the operator knob controls.
    /// The success arm samples the **pre-write** age, never a post-write ~0.
    pub checkpoint_pending_age_seconds: Histogram<f64>,
    /// Per-tick gauge of the **maximum** pending age (seconds) among the layers
    /// that were still pending when the checkpoint loop's most recent completed
    /// tick ended, by `service` (b.jq3). Recorded once per tick by the leader;
    /// `0` by non-leader replicas.
    ///
    /// Companion to [`Self::checkpoint_pending_age_seconds`], not a substitute:
    /// a p99 over thousands of layers structurally cannot see one layer stuck
    /// behind a poison payload, and that single layer is the DR gap an operator
    /// needs paged on. Query it as `max(...)` across replicas, **never**
    /// `sum(...)` — every replica publishes its own view of the same fleet-wide
    /// maximum, so a sum would multiply the value by the replica count.
    ///
    /// # Expected baseline — read before setting a threshold
    ///
    /// A layer the tick successfully wrote does NOT contribute here (it is no
    /// longer pending), but layers mid-cycle do, so on an active fleet this
    /// gauge **sawtooths between 0 and roughly
    /// `LIVE_EDIT_CHECKPOINT_INTERVAL_SECS`** as layers age toward eligibility
    /// and get written. That is the healthy shape, not a symptom. An alert
    /// threshold must therefore sit well above the interval — the loop's own
    /// worst-case bound for a legitimately-deferred layer is
    /// `interval + quiesce + tick`, which is the same bound the loop WARNs on,
    /// and an alert should allow headroom above even that for assembly time.
    ///
    /// Non-leader replicas record `0` deliberately (see
    /// [`Self::checkpoint_layers_never_written`] for why silence is not an
    /// option). A tick whose keyspace walk was truncated — by shutdown, or by a
    /// SCAN error — publishes nothing at all rather than a partial maximum.
    pub checkpoint_oldest_pending_age_seconds: Gauge<f64>,
    /// Per-tick gauge of how many evaluated layers have un-persisted commits and
    /// **no durable generation at all** (`last_checkpoint_ms == 0`), by
    /// `service` (b.jq3). Recorded once per tick by the leader; `0` by
    /// non-leader replicas.
    ///
    /// This is the honest home for "infinite pending age" — layers that are
    /// outside DR entirely rather than merely stale. A sustained non-zero value
    /// means the loop is enumerating layers it never manages to persist.
    /// Query as `max(...)`, never `sum(...)`, for the same reason as
    /// [`Self::checkpoint_oldest_pending_age_seconds`].
    ///
    /// Both gauges are recorded as `0` on the non-leader path because an OTel
    /// `Gauge` retains its last recorded value indefinitely (see the note on
    /// [`Self::service_token_available`] for the "no series until first
    /// `record()`" half of that contract): a replica that loses the leader lock
    /// would otherwise keep exporting the last maximum it happened to observe,
    /// forever. A *failed* lock probe publishes nothing — an unknown leader
    /// state is not evidence of a healthy fleet.
    pub checkpoint_layers_never_written: Gauge<u64>,
    /// Histogram of one counter-seed routine's wall-clock duration, by `phase`.
    /// Buckets `[1ms..5s]`.
    ///
    /// - `phase="startup"` — the one-shot SR-5.3 startup seed, once per process.
    /// - `phase="repair"` — the b.b2j lazy runtime repair, reached from a
    ///   `CreateLayer` that found `le:counter:object_id` absent. Runs on a
    ///   detached task, so the sample is the repair's own latency and not what
    ///   any one caller waited.
    ///
    /// The label is not cosmetic: the two phases share a body but not a
    /// distribution (startup runs against a cold adapter and a cold token file;
    /// repair runs mid-traffic), and an unlabeled family silently mixed
    /// repair-time samples into the series an operator reads as "how long did
    /// startup take".
    pub checkpoint_counter_seed_secs: Histogram<f64>,
    /// Outcome of the checkpoint assembler's proven read, judged against the
    /// header sequence pinned before it, by `outcome` (`verified` |
    /// `unverified`) and `service` (b.pw6). Incremented once per per-layer
    /// assembly that reached a decision — `Written` or `SkippedNotQuiesced`.
    /// Assemblies that error out are NOT counted, and that includes ones whose
    /// read already proved a cut: a verified cut followed by a failure in a
    /// later assembly step (OMPF build, adapter `put`) spent its read without
    /// landing on this counter, as does a layer deleted mid-read (`NotFound`),
    /// a fenced read that ended without a proof (broken `FENCE_READ_ATTEMPTS`
    /// times, or its fence wait past `reader_wait_max(L)`), and an assembly
    /// abandoned for a lost leader lease. So the sum of both cells is a lower
    /// bound on proven reads, not an exact count; pair it with
    /// `live_edit_checkpoint_writes_total{result="error"}` when reconciling.
    ///
    /// `verified` means the snapshot was proven to be a complete consistent cut
    /// — by the atomic read for a layer at or below the atomic-read limits, by
    /// the fenced read above them — at or above the pin, so the persisted blob
    /// carries every commit at or below its `snap_seq`. `unverified` means the
    /// proven cut came back below the pin: a **header-sequence regression**,
    /// which only a DR restore with an undersized band causes (Spec SDD SR-4.1,
    /// SR-8.6). The loop then deliberately persisted nothing in the snapshot
    /// form for that layer, and the layer's newest durable generation stays
    /// where it was until a later tick succeeds — a DR-exposure signal at any
    /// layer size, so alert on any increase. The assembler's WARN names the pin
    /// and the cut.
    pub checkpoint_stable_cut_total: Counter<u64>,
    /// Outcome of one checkpoint tail-capture attempt, by `outcome` (SDD
    /// SR-8.2). Vocabulary pinned by [`CHECKPOINT_TAIL_OUTCOMES`]; every cell
    /// zero-seeded at construction.
    ///
    /// **Labelled by `outcome` only — deliberately NOT by `service`.** Its
    /// query-pair sibling [`Self::checkpoint_writes_total`] is likewise
    /// unlabelled by `service`, and the healthy-signature query below pairs the
    /// two families, so a label asymmetry would force a `sum without(service)`
    /// rewrite on every such query for no gained dimension. Note the contrast
    /// with [`Self::checkpoint_stable_cut_total`], which IS `service`-labelled
    /// — that is the wrong sibling to copy here.
    ///
    /// # Outcomes
    ///
    /// - `captured` — a capture persisted a tail covering the full sound prefix
    ///   it found.
    /// - `captured_capped` — a capture persisted a *cap-bounded* sound prefix
    ///   (SR-1.4). The claimed range still equals the actual content — claiming
    ///   coverage beyond content is forbidden in every case — so this is not an
    ///   error signal: a sound shorter tail still advances the recovery point.
    /// - `rebased` — a tick that would have exceeded the chain bound composed
    ///   the lineage into a fresh base instead (SR-4.1). Non-zero is the
    ///   feature working, not a fault: it is how the chain of a layer whose
    ///   ticks fell back to tail capture — since every read proves its cut,
    ///   only after header-sequence regressions — gets collapsed so its
    ///   coverage can keep advancing. The re-base's own *write* counts as
    ///   `checkpoint_writes_total{kind="layer"}`, not `layer_tail` — a re-base
    ///   produces a base.
    ///
    ///   **Exactly one producer:** the chain-collapse re-base
    ///   (`checkpoint::rebase::rebase_layer_lineage`, ticked by
    ///   `checkpointer::rebase_on_chain_full`). The SR-3.6 *post-restore*
    ///   re-base is a different operation and deliberately does **not** tick
    ///   this cell — it is a write on the recovery path, not a capture outcome,
    ///   and folding it in would make a DR restore of N layers read as N chain
    ///   collapses. That separation is live and pinned by
    ///   `tests/integration/checkpoint/restore_rebase_test.rs`.
    /// - The seven `refused_*` cells fire **only when the sound prefix is empty**
    ///   (SR-1.3: a non-empty prefix is always persisted, never refused).
    ///   Individually:
    ///   - `refused_trimmed` — the stream head was trimmed past the expected
    ///     first entry ID.
    ///   - `refused_missing_bytes` — no admissible evidence for the next commit.
    ///     Three producer arms: an entry matched neither replayable form
    ///     (defense-in-depth; no reachable producer today); the byte cap
    ///     excluded the very first entry, so the window held no admissible first
    ///     commit; and the capture unit's byte budget derived to zero after the
    ///     envelope-framing reservation (defense-in-depth — `Config::from_env`
    ///     rejects a `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES` below the 66-byte
    ///     one-record floor at startup). The operator lever for all three is
    ///     `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES`, **not** the retention knob.
    ///   - `refused_sentinel` — an auto-ID deleted/overwrite sentinel ended the
    ///     prefix.
    ///   - `refused_seq_gap` — the post-restore fossil boundary. **Three**
    ///     producer arms: the classifier observed an ascending sequence jump in
    ///     the stream, so pre-jump entries belong to an abandoned timeline and
    ///     the whole capture is refused **including any accumulated prefix**;
    ///     the capture unit's pre-read fossil guard refused on a non-zero
    ///     `obj:hdr` `restore_seq_bump` or untrusted cursor columns, with *no*
    ///     jump observed — a jump beyond the bounded read is invisible to the
    ///     slice, so an outstanding restore is refused on the stamp alone; and
    ///     the checkpoint loop's chain-at-K admission gate
    ///     (`checkpointer::rebase_on_chain_full`) declined to start an SR-4.1
    ///     re-base on the same header condition, because the capture unit's
    ///     `ChainFull` return is produced *before* its own fossil guard and so
    ///     carries no fossil check to inherit. All three share one remediation,
    ///     and it is self-healing: a post-restore re-base or the next verified
    ///     snapshot clears the stamp and capture resumes. Each raises its own
    ///     distinct tracing event — that is what makes the three-way fold onto
    ///     one cell defensible, and the separability convention
    ///     `checkpointer_tail_outcomes_test.rs` pins.
    ///   - `refused_no_base` — **this object has no durable generation at
    ///     all**: the checkpoint store's listing for it is empty. Exactly one
    ///     producer arm, and it is the rollout tracker's meaning — a bulk
    ///     non-zero population that drains as each layer gets its first full
    ///     backup is the expected post-enable transition, not a fault. Logged
    ///     at DEBUG for that reason. The store-damage readings that used to
    ///     share this cell are `refused_lineage_unrootable` below, so a
    ///     `refused_no_base` rate now means "these layers have never been
    ///     backed up", full stop.
    ///   - `refused_lineage_unrootable` — generations DO exist for this object,
    ///     but none of them can root a lineage. Two producer arms, both logged
    ///     at WARN: **every** generation in the listing is a tail (the base was
    ///     manually pruned, or the store suffered partial loss), or the lineage
    ///     head sits at sequence 0 (a hand-written store — SR-7.1 roots every
    ///     lineage at `snap_seq = 1`). Each raises its own distinct tracing
    ///     event naming which, per the separability convention.
    ///
    ///     **Expected ZERO forever**, on every deployment and from the first
    ///     scrape: no shipped write path can produce either shape, because a
    ///     tail is only ever written after the base it names. Remediation is
    ///     **checkpoint-store repair** — restore or re-seed the missing base —
    ///     and no `LIVE_EDIT_CHECKPOINT_TAIL_*` knob affects it. That distinct
    ///     remediation is the whole reason it is a separate cell rather than a
    ///     fold onto `refused_no_base`: the two conditions read identically on
    ///     a dashboard but the first drains on its own and the second never
    ///     does. Once lineage-aware retention is live, a tails-only listing is
    ///     also the fingerprint of the PRD AC-7 class — a base deleted while a
    ///     retained tail still references it — which retention forbids by
    ///     design, so a non-zero rate here is alertable.
    ///   - `refused_chain_full` — the chain bound was reached **and the SR-4.1
    ///     re-base that should have collapsed it did not land**. See below.
    ///
    /// # Reading `refused_chain_full` now that re-base is live
    ///
    /// This cell no longer means "a layer hit K and stopped advancing
    /// forever". A tick that finds the chain at K attempts a chain-collapse
    /// re-base; the healthy outcome is `rebased`. `refused_chain_full` is what
    /// remains when that attempt did not land — SR-4.5's refusal arm, one cell
    /// for every cause (gather failure, unverifiable lineage head, compose
    /// rejection, an ext-blob backfill that could not be durably acknowledged,
    /// or the base PUT failing). The distinctions ride the per-arm WARNs, which
    /// is where they are actionable and where they can carry a `Display` no
    /// metric label may.
    ///
    /// So it is **partly an adapter-failure cell**, unlike the other six
    /// `refused_*` values, which are all lineage or stream *decisions*. Two
    /// consequences for whoever reads it:
    ///
    /// - **Remediation is the accompanying WARN plus adapter health**, not the
    ///   chain knob. A sustained non-zero here usually means the checkpoint
    ///   store is failing reads, writes, or both for that layer; the lineage is
    ///   intact and every tick re-attempts, so nothing is lost while it lasts,
    ///   but coverage stops advancing for as long as it does. Raising
    ///   `LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN` does not help a failing adapter —
    ///   it only postpones the collapse.
    /// - **The base-PUT-failure arm also ticks
    ///   [`Self::checkpoint_writes_total`]`{result="error", kind="layer"}`**, so
    ///   that one arm is visible on the adapter-health signal too. The other
    ///   refusal arms deliberately move no `writes_total` cell.
    ///
    /// A single tick may move this cell for one layer and `rebased` for another;
    /// what cannot happen is one layer-tick moving both, which the loop's
    /// interception ordering guarantees.
    ///
    /// # Relationship to the stable-cut counter
    ///
    /// [`Self::checkpoint_stable_cut_total`]`{outcome="unverified"}` keeps its
    /// meaning exactly (assembly could not verify a stable cut); a tail success
    /// does **not** suppress it. The pair `{unverified rising, tail captured
    /// rising}` is the fallback working: since every read proves its cut
    /// (Spec SDD SR-4.1, SR-11.5), `unverified` means a header-sequence
    /// regression at any layer size, and a capture after one is the fallback
    /// covering that rare restore-band fault, so the layer's recovery point
    /// advances anyway.
    pub checkpoint_tail_total: Counter<u64>,
    /// Outcome of one `checkpoint::base_bootstrap::BaseBootstrap::run_once`
    /// invocation, by `outcome` and `site` (SDD SR-7 / SR-7.1 / SR-3.6).
    /// Vocabulary pinned by [`CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES`] and
    /// [`CHECKPOINT_BASE_BOOTSTRAP_SITES`]; the full 10 × 2 grid is zero-seeded
    /// at construction.
    ///
    /// # The identity this family owes its consumers
    ///
    /// **Exactly one increment per `run_once` invocation**, on every arm
    /// including the ones that write nothing and log below WARN. So
    /// `sum by (outcome)` equals the invocation count, and that is not a
    /// nicety — it is what the two intended queries are built on:
    ///
    /// - `shed / sum by (site)` is an **admission-saturation ratio**. A bare
    ///   `shed` count answers "how often", never "how often *relative to
    ///   demand*", and demand here is bursty by nature (a mass restore issues
    ///   invocations as fast as the recovery hook applies).
    /// - `sum(rate(...)) == 0` is a **producer-liveness predicate** — the unit
    ///   ran zero times, distinguishable from "it ran and everything was
    ///   benign", which a refusal-only family cannot express.
    ///
    /// An arm added later that returns without incrementing breaks both. The
    /// benign cells exist to be the denominator, so folding any of them away as
    /// "uninteresting" is the one edit this family cannot absorb.
    ///
    /// # Additive to, never a substitute for, [`Self::checkpoint_writes_total`]
    ///
    /// Scoped to **this producer's own writes** — the pairings below say nothing
    /// about the periodic loop's `kind="layer"` cells, which have their own
    /// producers (a chain-collapse re-base tick moves two of them, so
    /// `writes_total{kind="layer"}` is not one-per-layer-tick in general).
    /// Within one `run_once`:
    ///
    /// - `persisted` pairs 1:1 with `{result="success", kind="layer"}`.
    /// - `failed` pairs with `{result="error", kind="layer"}` — **except** the
    ///   `NotImplemented` arm (the `Disabled` adapter's `NoopCheckpointStore`,
    ///   unreachable behind the three-way gate), which deliberately ticks
    ///   nothing there so "checkpointing is off" manufactures no error noise
    ///   floor, and ticks here only to preserve the one-per-invocation identity.
    /// - `unstamped_write_error` **also** ticks `{result="error", kind="layer"}`.
    ///   It is not a second error family: it splits an existing error tick into
    ///   an actionable sub-case (see below) without removing it, so no landed
    ///   adapter-error-share alert changes shape.
    ///
    /// The other seven cells move no `writes_total` cell at all. That is the
    /// gap this family closes: before it, five of them moved nothing anywhere.
    ///
    /// # The `unstamped_*` grouping
    ///
    /// The prefix is the operator grouping for **"the base blob is durable on
    /// the adapter but the header does not point at it."** All four fire
    /// strictly *after* a successful `put_layer_checkpoint`, at or after the
    /// pre-stamp re-read, so the artifact exists and the cursor columns do not
    /// name it. Query them as `sum by (site) (rate(...{outcome=~"unstamped_.*"}))`
    /// — one grouping, one meaning, one remediation shape.
    ///
    /// `unstamped_header_absent` is the one member of the group whose blob is
    /// not merely un-pointed-at but *unwanted*: the layer it covers no longer
    /// exists, so the artifact is reclaimed rather than adopted by a later tick
    /// (b.vn6). It shares the grouping anyway because the durable-blob /
    /// un-stamped-header shape — the thing the prefix names — is identical.
    ///
    /// Nothing is *lost* in that state: every later reader, the periodic loop
    /// included, sees a layer with un-persisted commits and re-snapshots it on
    /// the next eligible tick. What it costs is the coverage between now and
    /// that tick, plus (at the restore site) a `restore_seq_bump` that stays
    /// outstanding, so tail capture keeps refusing `refused_seq_gap` until it
    /// clears. A sustained non-zero rate is therefore a real, if soft, DR-lag
    /// signal rather than a data-loss one.
    ///
    /// # Outcomes
    ///
    /// - `persisted` — the base landed and the cursor stamp committed. The
    ///   healthy terminal arm.
    /// - `shed` — the per-unit in-flight admission bound
    ///   (`BOOTSTRAP_MAX_INFLIGHT`) was saturated; nothing was read, built,
    ///   written or stamped. At either site it heals at the layer's next
    ///   eligible periodic tick, whose proven read (atomic or fenced, at any
    ///   layer size) lands a base — at `site="restore_rebase"` also clearing
    ///   the restore's `restore_seq_bump`; until then a restore-site shed
    ///   leaves the layer's newest durable copy at the generation it was
    ///   restored from. Where that wait is too long, the remediations
    ///   are re-restoring the affected layers, and raising the pod memory limit
    ///   and `LIVE_EDIT_MAX_LAYER_SIZE` *together* (the bound is derived against
    ///   their product). Read it as the ratio, not the count.
    /// - `skipped_header_absent` — `obj:hdr` was gone at the step-1 read: the
    ///   layer was deleted or overwritten inside the bootstrap window. A routine
    ///   race; non-zero is healthy.
    /// - `skipped_generation_present` — the pre-PUT cursor-regression guard (a)
    ///   found a generation at or above this target already landed, so this
    ///   unit's whole purpose was already served. **Non-zero is healthy** — no
    ///   coverage is lost and no PUT was spent.
    /// - `skipped_untrusted_cursor` — `restore_rebase` only: the header's cursor
    ///   columns were present but unparseable, so a blind stamp (and a blind
    ///   *clearing* of `restore_seq_bump`) could not be justified. Nothing was
    ///   written. Points at a writer putting garbage in `obj:hdr`, not at the
    ///   adapter.
    /// - `unstamped_header_absent` — post-PUT: the conditional cursor stamp was
    ///   REFUSED because `obj:hdr` no longer carries a `seq` field, i.e. the
    ///   layer was deleted (or overwritten) while the base PUT was in flight
    ///   (b.vn6). Deliberately **not** folded onto `skipped_header_absent`: that
    ///   cell is pre-PUT and leaves the adapter untouched, while this one has a
    ///   durable blob behind it that must be reclaimed — the same pre-PUT /
    ///   post-PUT split the `skipped_generation_present` ↔
    ///   `unstamped_generation_present` pair already draws, and the reason
    ///   `skipped_header_absent` stayed flat at zero throughout the dev
    ///   incident that produced this ticket. The accompanying WARN carries
    ///   `reaped = true|false`: `true` means the fast-path reap deleted the
    ///   just-written base immediately, `false` means it did not and the
    ///   reconciliation pass's dead-object phase is the backstop
    ///   (`dead_object_reaped` on
    ///   [`Self::checkpoint_reconcile_total`]). Non-zero is a routine race,
    ///   not a fault; non-zero *with* `reaped = false` is worth following up.
    /// - `unstamped_generation_present` — post-PUT guard (b), sequence half: a
    ///   generation landed while our base PUT was in flight, so every cursor
    ///   column was left untouched. Benign at `creation_base` (some other writer
    ///   covered the layer); actionable at `restore_rebase`, where the
    ///   outstanding bump is what keeps tail capture refusing.
    /// - `unstamped_foreign_restore` — post-PUT guard (b), bump half. See the
    ///   risk-indicator section below; `restore_rebase` only.
    /// - `unstamped_write_error` — the base PUT succeeded and then the Valkey
    ///   side of the stamp failed: the pre-stamp `HMGET` of the cursor columns,
    ///   or the `HSET` itself. Split out from `failed` precisely because the
    ///   artifact is durable here and is not durable there — the same split the
    ///   `unstamped_*` grouping exists to expose. Remediation is Valkey health,
    ///   not adapter health.
    /// - `failed` — every arm that produced no base: the identity read, the
    ///   reset-OMPF build, the ext-refs sidecar serialization, the PUT itself,
    ///   and the unreachable `NotImplemented` return. All but the last also tick
    ///   `writes_total{result="error", kind="layer"}`; the per-arm WARN is what
    ///   separates them, which is the distinct-event discipline every fold in
    ///   this subsystem rests on.
    ///
    /// # `unstamped_foreign_restore` is a RISK INDICATOR, not a harm count
    ///
    /// It counts the double-restore interleavings guard (b) **CAUGHT** — a
    /// re-base that found a `restore_seq_bump` it did not absorb and declined to
    /// clear it. Each one is a correct refusal that cost nothing. What makes the
    /// cell worth having is what it says about the *population*: the guard's
    /// accepted residual is its false negative (two restores that applied the
    /// same band are byte-indistinguishable in the header, so a stale re-base
    /// passes both halves and clears a bump it does not own), and that residual
    /// is undetectable at the decision point — if the code could tell the
    /// harmful case from the benign one it would refuse rather than count. So
    /// this cell is honest evidence that the interleaving class is **live in
    /// this fleet**, which is the only thing an operator can act on; it is
    /// explicitly **not** a count of the residual's harm, and must never be read
    /// or documented as one.
    ///
    /// The guard's bump half blocks in **three** forms, all folded here: a
    /// DIFFERENT band (the named case, a genuine foreign restore), an ABSENT
    /// column, and an UNPARSEABLE one. The latter two normally coincide with the
    /// sequence half and return as `unstamped_generation_present` one arm
    /// earlier — the only writer that clears the bump is a periodic tick, whose
    /// `last_checkpoint_seq` stamp that half already catches. **Reaching this
    /// cell with an absent or unparseable bump therefore means the header was
    /// edited outside the service's write paths**, so a non-zero value must not
    /// be read exclusively as double-restore interleaving; check the
    /// accompanying WARN's `observed_restore_seq_bump` field before concluding
    /// either way.
    ///
    /// # Why the full grid is seeded, inadmissible cells included
    ///
    /// Two cells cannot fire: `skipped_untrusted_cursor` and
    /// `unstamped_foreign_restore` at `site="creation_base"`, both being
    /// `RestoreRebase`-only arms. They are seeded anyway, on the
    /// [`Self::checkpoint_writes_total`] full-grid precedent — a dashboard that
    /// selects the family by name gets a rectangular result, and a permanently
    /// zero cell is cheaper than a query that has to know which pairs are
    /// structurally impossible. (Contrast `stream_id_collision_total`, which
    /// seeds only reachable pairs: there the unreachable pairing is an artifact
    /// of *backend* wiring that could change, so publishing it would assert
    /// something false about the code.)
    ///
    /// # Labelled by `outcome` and `site` only — deliberately NOT by `service`
    ///
    /// Saturation is a per-process property (the admission semaphore is per-unit,
    /// and production constructs one unit per replica), and the absent `service`
    /// label does not hide it — per-pod scrape-target labels already separate the
    /// series. Do not add `service` here: it would break the `writes_total` /
    /// `tail_total` pairing queries this family is designed to sit beside.
    pub checkpoint_base_bootstrap_total: Counter<u64>,
    /// Outcome of the checkpoint loop's mid-tick leader-lease checks, by
    /// `outcome` (`renewed` | `lost` | `refresh_error`) and `service` (b.s5y).
    ///
    /// The checkpoint tick walks its due layers sequentially under
    /// `le:checkpoint:leader`, whose TTL is `LEADER_LOCK_TTL_MS`. A single
    /// adapter call can take minutes (per-request timeout × transient-retry
    /// budget × auth-retry), so a tick genuinely can outlive its lease. The
    /// loop therefore re-proves the lease between layers and aborts the walk
    /// when it cannot.
    ///
    /// - `renewed` — a `PEXPIRE`-style refresh landed; the lease was
    ///   continuously held. This is the steady-state cell and should dominate.
    /// - `lost` — the lease could not be proven held: the local deadline had
    ///   already passed (no round-trip issued), or the refresh reported the key
    ///   lapsed / owned by a peer, or the refresh errored. The tick aborted its
    ///   remaining layers. Sustained non-zero means ticks are outrunning the
    ///   lease, and the tail layers of every tick are being deferred.
    ///
    ///   Remediation, in order: investigate **adapter latency** (one slow
    ///   `put_layer_checkpoint` can consume a whole lease on its own), then
    ///   **reduce the due-set size per tick** (a shorter
    ///   `LIVE_EDIT_CHECKPOINT_INTERVAL_SECS` spreads layers across more ticks
    ///   rather than bunching them), then revisit the lease TTL itself.
    ///   Do **not** reach for a larger `LIVE_EDIT_CHECKPOINT_TICK_SECS`: a
    ///   longer tick period means more layers fall due between ticks, so each
    ///   tick has *more* work to fit inside the same 30 s lease, not less.
    /// - `refresh_error` — a backing-store error on the refresh round-trip.
    ///   Additive to `lost`, not a substitute: an errored refresh also counts
    ///   `lost`, because unknown is not held. Use this cell to tell a Valkey
    ///   fault apart from a genuine lease expiry.
    pub checkpoint_lease_total: Counter<u64>,
    /// Outcome tally of the checkpoint-store reconciliation pass, by
    /// `(outcome, service)` (b.9qd, SDD §22.17.5).
    ///
    /// The pass is hourly by default and mostly reports nothing happening, so
    /// every cell is zero-seeded — an absent series would be indistinguishable
    /// from a pass that never ran, which is the failure this counter has to be
    /// able to show.
    ///
    /// Outcome vocabulary, in the order one pass produces them:
    ///
    /// - `orphan_deferred` — a candidate stem (or temp) registered on
    ///   `le:ckpt:{recon}:orphans` but not yet past its grace window. The
    ///   healthy steady state for a store with a partial publish in flight;
    ///   sustained non-zero with no `orphan_deleted` following means candidates
    ///   keep healing, which is fine, or keep failing to age out, which is not.
    /// - `orphan_deleted` — an orphan stem reaped (its sidecar's hashes
    ///   tombstoned first, then the generation deleted).
    /// - `temp_deleted` — write-temp leftovers reclaimed, counted per stem, not
    ///   per file.
    /// - `dead_object_reaped` — a **dead** object's checkpoint generations were
    ///   reclaimed, counted per generation deleted (b.vn6). Members of
    ///   `le:ckpt:{recon}:dead` are minted only by the delete paths, so this
    ///   cell is the reclamation of layers the service knows are gone — the
    ///   general leak that existed before b.vn6, where `delete_layer_inner`
    ///   touched the checkpoint store nowhere at all. Steady non-zero on a
    ///   fleet with churny create/delete clients is the pass working.
    /// - `dead_deferred` — dead objects claimed but left registered, counted per
    ///   object. Four provenances: a step failed part-way (a sidecar read, a
    ///   tombstone record, a generation delete, or a liveness re-probe); the
    ///   object's header is still standing with `superseded=1` — a
    ///   `CreateLayer` overwrite whose old-object cleanup has not landed, which
    ///   is a dead object with an un-cleaned header rather than a resurrection,
    ///   so it is neither reaped nor retired; the object's own listing was
    ///   empty on a pass whose layers-namespace enumeration listed nothing,
    ///   failed or elapsed, so "already clean" could not be told from a
    ///   misaddressed store (b.oya); or the object's own listing was empty
    ///   although that same pass's enumeration named one of its generations —
    ///   two listings in one pass disagreed, with a WARN saying so (b.oya).
    ///   Durable either way: the next pass re-claims them. Sustained non-zero
    ///   with no `dead_object_reaped` means the same objects are failing every
    ///   pass — check the accompanying WARNs for a stuck adapter or a
    ///   disagreeing listing, or the INFOs for a stuck overwrite cleanup or an
    ///   empty layers namespace.
    /// - `dead_spared` — a member of the dead ZSET whose `obj:hdr` was found
    ///   LIVE and carrying **no** `superseded` flag by a liveness probe, so the
    ///   reap stopped there and the member was retired. **Non-zero is the safety
    ///   re-check working, not a fault**: the reachable provenance is a DR
    ///   restore that resurrected a previously-deleted `object_id`, whose
    ///   generations are exactly the blobs it was restored from. Reaping them
    ///   would be the data loss this phase's liveness re-check exists to
    ///   prevent. Sibling of `ext_spared`, and read the same way.
    ///
    ///   Ticked from either of the two probes, and **what "spared" covers
    ///   differs between them**. The cheap per-object early-out runs before the
    ///   listing, so nothing was deleted at all. The per-generation probe runs
    ///   immediately before each individual delete, so a resurrection it catches
    ///   part-way through an object spares only the generations still standing —
    ///   any already reaped earlier in the same pass stay reaped (and stay
    ///   counted on `dead_object_reaped`), because each of those went while the
    ///   object was provably dead at its own pre-delete probe. The member is
    ///   retired either way: the object is alive now, and a live object's
    ///   remaining generations are not this phase's to reclaim.
    /// - `ext_tombstoned` — hashes recorded on `le:ckpt:{recon}:tomb`. Ticked by
    ///   the reconciliation pass, the retention prune, AND the base bootstrap's
    ///   inline `reap_dead_base` — three producers of one act: recording a hash
    ///   whose last known reference is going away, before the reference is
    ///   deleted. (The four refused-stamp arms are NOT a fourth producer: they
    ///   re-register the object on the dead ZSET and let the reconciliation pass
    ///   do the recording. The bootstrap's arm does both — it re-registers AND
    ///   attempts the inline reap, and only the reap's tombstone step ticks
    ///   here.)
    /// - `ext_collected` — tombstoned hashes confirmed absent from the live set
    ///   after the grace window and `delete_ext_blob`'d. This is the only cell
    ///   that represents reclaimed bytes.
    /// - `ext_spared` — tombstoned hashes found still live at collect time and
    ///   retired without deletion. **Non-zero here is the protocol working, not
    ///   a fault**: it is precisely the over-delete that the pre-b.9qd inline
    ///   cascade would have committed.
    /// - `live_set_failed` — a live-set walk (`list_live_ext_set`) failure aborted the
    ///   collect phase. That includes an ext-refs sidecar the walk listed and
    ///   then read as absent, which the adapter reports as a failure rather
    ///   than skipping (b.oya) — but only while the pass still holds the
    ///   checkpoint lease after the walk. When a peer holds it instead, that
    ///   peer's lease-gated retention prune explains the absence, so the phase
    ///   ends as a lease loss (`live_edit_checkpoint_lease_total`) and does not
    ///   tick here; it deletes nothing either way. Every other live-set failure
    ///   (a store error, a malformed sidecar, the budget elapse) ticks here
    ///   whoever holds the lease. Tombstones survive for the next pass, so this is
    ///   a deferral, not a leak — but sustained non-zero means nothing is ever
    ///   being reclaimed and the backlog gauge will climb. An EMPTY live set
    ///   whose own walk listed no layer artifact is not a failure and does not
    ///   tick here: the phase deletes nothing and logs an INFO instead.
    pub checkpoint_reconcile_total: Counter<u64>,
    /// Total times a checkpoint-adapter wall-clock ceiling elapsed and the work
    /// was abandoned as `Transient`, by `(phase, service)` (b.gg7).
    ///
    /// Vocabulary and per-phase semantics live on
    /// [`CHECKPOINT_ADAPTER_CEILING_PHASES`]; every cell is zero-seeded.
    ///
    /// Read it as a **saturation** signal, not an error rate. Every increment
    /// means a backend stayed slow or unavailable through the adapter's entire
    /// retry ladder, so sustained non-zero `phase="call"` says the storage
    /// backend is browning out, while sustained non-zero `phase="enumeration"`
    /// says the artifact namespace has outgrown what one reconciliation grace
    /// window can walk — a different problem with a different remedy
    /// (retention / prefix sharding, not backend latency).
    ///
    /// Nothing is lost when this ticks: the checkpoint tick, reconciliation
    /// pass, and recovery hook all defer on `Transient` and retry on their next
    /// cadence. What it bounds is *waste* — the pre-b.gg7 code would have kept
    /// the same call alive indefinitely, holding a leader lease or an admission
    /// mutex the whole time.
    pub checkpoint_adapter_ceiling_elapsed_total: Counter<u64>,
    /// Total storage-api client-side redirect events, by `(verb, outcome,
    /// service)` (b.z22).
    ///
    /// Vocabulary and per-cell semantics live on
    /// [`CHECKPOINT_ADAPTER_REDIRECT_CELLS`]; every listed cell is zero-seeded.
    ///
    /// Two different stories share one family because they are the same
    /// mechanism (an HTTP 300 carrying a presigned URL the client must follow)
    /// seen from the two verbs:
    ///
    /// - `verb="put"` is the **expected** path for any artifact at or above the
    ///   deployment's `writeBodyMaxSize`. `followed` + `completed` should track
    ///   each other one-for-one; a gap between them is `complete_failed` and
    ///   means payloads are reaching blob storage but the write is not being
    ///   acknowledged. On a deployment whose upload-options pre-flight answers,
    ///   `preflight_redirect` tracks them too — a `followed` without it is a
    ///   redirect the adapter discovered by streaming a payload first.
    /// - `verb="get"` is the **unexpected** path. Reads declare
    ///   `download_preference=body`, which storage-api honours unconditionally
    ///   at any size, so a healthy deployment leaves every `get` cell at zero.
    ///   Sustained non-zero `("get", "followed")` says the backend is ignoring
    ///   the declared preference and every large restore now depends on
    ///   egress to blob storage that this service otherwise never exercises.
    ///
    /// Deliberately carries no URL: a presigned redirect target is a bearer
    /// credential in URL form (SR-6.2), so neither this counter nor the WARN
    /// beside it may name one.
    pub checkpoint_adapter_redirect_total: Counter<u64>,
    /// Current size of `le:ckpt:{recon}:tomb` — ext-blob hashes awaiting their
    /// grace window or a successful collect (b.9qd).
    ///
    /// Sampled once per reconciliation pass by the leader (a `ZCARD`), and
    /// published as `0` by every non-leader replica for the same reason the
    /// pending-age gauges are: an OTel `Gauge` retains its last value forever,
    /// so a demoted leader would otherwise pin the fleet-wide `max()` at a
    /// backlog that no longer exists.
    ///
    /// Read it as a derivative, not a level. A backlog that rises and falls with
    /// prune activity is the protocol breathing; a backlog that only rises means
    /// the collect phase is not completing — cross-check
    /// `checkpoint_reconcile_total{outcome="live_set_failed"}`. With that cell
    /// flat, look for the collect phase's INFO about an empty live set whose
    /// own listing named no layer artifact (b.oya): that pass deletes nothing
    /// by design — an empty store, a misaddressed one, or a listing that
    /// answered a stale not-found. With neither, check
    /// `live_edit_checkpoint_lease_total{outcome="lost"}` climbing: the collect
    /// ended as a lease loss because its walk outlasted the lease and a peer
    /// claimed it — including a listed ext-refs sidecar that read absent while
    /// the peer led, which ticks no `live_set_failed` and logs only at debug.
    /// That is the routine shape on a large multi-replica store whose walk
    /// outlasts the lease.
    pub checkpoint_ext_tombstone_backlog: Gauge<u64>,
    /// Current size of `le:ckpt:{recon}:dead` — deleted `object_id`s whose
    /// checkpoint generations are still awaiting the reconciliation pass's
    /// dead-object phase (b.vn6).
    ///
    /// Sampled and published exactly like its sibling
    /// [`Self::checkpoint_ext_tombstone_backlog`]: one `ZCARD` per
    /// reconciliation pass by the leader, `0` by every non-leader replica so a
    /// demoted leader cannot fossilise a stale backlog into the fleet-wide
    /// `max()`, and nothing at all when the read itself errored (unknown is not
    /// zero).
    ///
    /// # Why this one needs its own gauge rather than riding the sibling
    ///
    /// The two backlogs have opposite shapes. Tombstones are produced by the
    /// retention prune and the reconciliation pass itself — background work, on
    /// the pass's own cadence. Dead-object members are produced on the
    /// **request path**: every `DeleteLayer` and every CreateLayer overwrite
    /// registers one, at whatever rate the fleet deletes layers, while the drain
    /// is sequential and confined to a quarter of an hourly pass
    /// (`ReconcilePassBudget`). That asymmetry is exactly the kind
    /// that falls behind quietly, and the phase's budget argument rests on this
    /// number being visible when it does.
    ///
    /// Read it as a derivative, not a level, as with the sibling. A backlog that
    /// rises and falls with delete activity is the phase working; one that only
    /// rises means the quarter-budget is no longer enough for the fleet's
    /// deletion rate, or the phase is deferring —  cross-check
    /// `checkpoint_reconcile_total{outcome="dead_deferred"}`. Falling behind
    /// costs storage, never correctness: the ZSET is durable and every member
    /// keeps its original `ZADD NX` first-observation score.
    pub checkpoint_dead_object_backlog: Gauge<u64>,
    /// Total `CommandCreateSection` deltas silently skipped because the named
    /// `parent_id` was not in `state.sections` (b.39x item 7). Unlabeled.
    pub delta_create_section_skipped_missing_parent_total: Counter<u64>,
    /// Total recovery-hook attempts, by `result`, `scope` (`layer` | `session`),
    /// and `service` (SDD SR-2.4 / SR-2.6 / SR-2.7).
    pub checkpoint_recovery_attempts_total: Counter<u64>,
    /// Total layer restores whose surviving per-object stream was refused by the
    /// SR-2.5 high-water plausibility gate and removed, by `service` (b.8y5).
    ///
    /// Deliberately its own instrument rather than a `result` label on
    /// [`Self::checkpoint_recovery_attempts_total`]: that label is one-per-attempt
    /// with a zero-seeded fixed vocabulary, so `sum by (result)` equals the attempt
    /// count, and these restores still count as `restored` there. This counter is
    /// additive and orthogonal — it answers "did a restore land on top of a stream
    /// it could not trust", not "how did the attempt end".
    ///
    /// Alertable. A non-zero rate means DR restored a layer whose stream carried a
    /// foreign high-water — almost always a tombstoned object resurrected through a
    /// stale session checkpoint. The restore is correct and the stream removal
    /// keeps the layer writable (b.ph9 makes the commit paths fail closed on
    /// such a stream rather than half-commit, so the removal is now a
    /// write-availability measure), but the pairing is worth a human look:
    /// subscribers attached to that layer lose their in-window catch-up, and a
    /// recurring signal points at a session-checkpoint/delete ordering problem
    /// upstream rather than at recovery itself.
    pub checkpoint_recovery_stream_refused_total: Counter<u64>,
    /// Total layer restores that found a surviving `obj:*` keyspace under the
    /// object they were about to materialize and wiped it first, by `service`
    /// (b.f4t).
    ///
    /// Additive to [`Self::checkpoint_recovery_attempts_total`] on the same
    /// precedent as [`Self::checkpoint_recovery_stream_refused_total`]: the
    /// restore still reports `result="restored"` there, and this answers a
    /// different question — "did this restore have to replace residue".
    ///
    /// Alertable. A non-zero rate means state loss was PARTIAL: the object's
    /// header was gone (or structurally incomplete) while its indexes, records,
    /// children lists or tombstones survived — the b.kf1 orphan class. The
    /// restore is correct (SR-2.4 restore semantics are an authoritative
    /// replace, so the residue is removed rather than merged into), but the
    /// pairing is worth a human look: a recurring signal points at something
    /// unlinking headers out of band, or at a retirement path that is leaving
    /// its `obj:*` family behind.
    ///
    /// It carries no external-value refcount caveat (b.qpf). The wipe withholds
    /// `obj:ext` from its `UNLINK` inventory precisely so the surviving index is
    /// the materialization's pre-image: `write_restored_object` reads it as
    /// `ext_before` and its absolute rebuild releases `before − after` inside
    /// the same transaction, so the references the wiped records held and the
    /// restored state does not are given up, not stranded. The accompanying WARN
    /// carries `ext_bearing_records` as an upper bound on the size of that
    /// reconciliation — it is not a count of leaked references.
    pub checkpoint_restore_residue_wiped_total: Counter<u64>,
    /// Total layer-restore attempts that found a peer replica had already landed
    /// this object's rebuild and therefore wrote NOTHING, by `stage` and
    /// `service` (b.h4m). Vocabulary pinned by
    /// [`CHECKPOINT_RESTORE_FENCED_STAGES`]; every cell zero-seeded at
    /// construction.
    ///
    /// Additive to [`Self::checkpoint_recovery_attempts_total`], never a
    /// substitute. The fenced attempt still reports `result="restored"` there —
    /// the layer IS restored, just by the other replica — so `sum by (result)`
    /// on that counter still equals its attempt count and this one answers the
    /// separate question "how often did two replicas race for the same layer,
    /// and how late was it noticed".
    ///
    /// `stage` says how late (see [`CHECKPOINT_RESTORE_FENCED_STAGES`] for the
    /// three positions). Read the distribution rather than the total:
    ///
    /// * `presence_probe` alone is unremarkable — lazy per-layer recovery means
    ///   two replicas can both be asked for a lost layer, and the loser paying
    ///   one `HGETALL` is the design working.
    /// * `residue_wipe` and especially `object_write` say the two attempts ran
    ///   concurrently right up to the write. They are the cells that matter,
    ///   because before b.h4m the `object_write` case had no fence at all: the
    ///   loser's content was ADDED to the winner's rebuild rather than replacing
    ///   it, producing a layer that is a MIXTURE of two backup generations with
    ///   no error anywhere and no counter moving. A nonzero rate here means the
    ///   fleet is reaching that window and the fence is what is holding it.
    ///
    /// Alertable only as a rate signal, not per-event: a sustained
    /// `object_write` rate points at a recovery-fan-out problem (many clients
    /// driven at one lost layer across replicas) worth reducing upstream, not at
    /// a correctness failure — the fence makes each such race safe. The
    /// correctness failure has its own cell, and it is
    /// `storage_watch_voided_total{path="restore_write_object",
    /// disposition="committed_unwatched"}`.
    pub checkpoint_restore_fenced_total: Counter<u64>,
    /// Total layer restores that could NOT establish what state the per-object
    /// stream key `le:{oid}:obj:stream` was in, by `reason` and `service`
    /// (b.4et). Vocabulary pinned by
    /// [`CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS`]; every cell
    /// zero-seeded at construction.
    ///
    /// Distinct from [`Self::checkpoint_recovery_stream_refused_total`], which
    /// counts the case where the evidence WAS obtained, was refused on
    /// plausibility, and the stream was verifiably removed — a restore that
    /// landed. This counter is the opposite: the evidence could not be
    /// obtained, or could be obtained but not acted on.
    ///
    /// # Reasons
    ///
    /// - `probe_unreadable` — the stream key exists and is a stream, but its
    ///   `last-generated-id` could not be read after the bounded retry (or the
    ///   `TYPE` round-trip that would classify it could not be read either).
    ///   **The restore was refused** — materializing a header underneath an
    ///   unknown stream top is how a layer silently reuses sequence numbers.
    ///   The caller sees `UNAVAILABLE` and may retry. Remediation: Valkey
    ///   health — this fires on a backend blip that outlasted three attempts.
    /// - `unlink_unverified` — the plausibility gate refused a high-water it
    ///   DID read, but the stale stream could not be verifiably removed: the
    ///   `UNLINK` failed, or it reported deleting nothing and the key was still
    ///   present afterwards, or the identity round-trip that authorizes the
    ///   delete failed, or the b.h4m peer-presence `HGETALL obj:hdr` re-read
    ///   immediately ahead of the `UNLINK` exhausted its retries — that read
    ///   decides whether the stream under this `object_id` is a PEER replica's
    ///   live one, so an unanswered one leaves the removal unauthorized rather
    ///   than merely unconfirmed. **The restore was refused**, for the same
    ///   reason. Remediation: Valkey health first — the identity and
    ///   peer-presence producers are a failed round-trip on the store, the same
    ///   signal as `probe_unreadable`; a repeating signal on one object with the
    ///   store healthy means something is re-creating that stream key out of
    ///   band.
    /// - `wrongtype_key` — `le:{oid}:obj:stream` holds a value that is not a
    ///   stream. Nothing in this service ever retypes a key it owns, so this is
    ///   a foreign value on the `le:` keyspace. **The restore PROCEEDED** (a
    ///   non-stream key carries no surviving high-water, so the guard band
    ///   covers the RPO window exactly as it does on a full wipe), but every
    ///   subsequent write to the layer fails closed on the b.459 wrong-type
    ///   refusal until the key is removed. Remediation: find the out-of-band
    ///   writer, then delete the key — the accompanying WARN names it and its
    ///   actual type.
    ///
    /// Alertable at any increment. The first two are a DR restore that did not
    /// happen; the third is a layer that restored into a permanent write
    /// outage.
    pub checkpoint_recovery_stream_unresolved_total: Counter<u64>,
    /// Outcome of restore-source **lineage selection**, by `outcome` and
    /// `service` (SDD SR-3.1). Vocabulary pinned by
    /// [`CHECKPOINT_RESTORE_LINEAGE_OUTCOMES`]; every cell zero-seeded at
    /// construction.
    ///
    /// **Additive to, never a substitute for,
    /// [`Self::checkpoint_recovery_attempts_total`].** That counter's
    /// `result × scope × service` grid is one-per-attempt with a fixed
    /// four-value vocabulary, and its documented property is that
    /// `sum by (result)` equals the attempt count. A step-down is *intra*-attempt
    /// and can fire several times within one attempt, so it cannot be a `result`
    /// cell without breaking that identity. The same restore still reports
    /// `result="restored"` (or `not_recoverable`) there. This is the deliberate
    /// instrument addition `recovery::select_restore_source`'s observability
    /// note invited, on the [`Self::checkpoint_recovery_stream_refused_total`]
    /// precedent.
    ///
    /// # Outcomes
    ///
    /// The two `gathered_*` cells are the terminal verdict of a selection — at
    /// most one fires per attempt. The `stepped_down_*` cells and `exhausted`
    /// may fire repeatedly within one attempt, once per abandoned candidate.
    ///
    /// - `gathered_base` — selection resolved to a base-only lineage. This is
    ///   the pre-Epic-4 answer and the steady state for a cold layer — and for
    ///   a hot one of any size, whose periodic assembly writes a verified base
    ///   every eligible tick (the atomic read at or below the atomic-read
    ///   limits, the fenced read above them), so tail capture (the fallback
    ///   when no cut is proven) runs for it only on a tick a header-sequence
    ///   regression skipped. A `gathered_base` on such a layer is the feature
    ///   working, not tail capture broken.
    /// - `gathered_composed` — selection resolved to a verified tail-headed
    ///   lineage: a base plus the contiguous tails replayed onto it. Reached
    ///   only while the layer's newest generation is still a tail written
    ///   before its latest eligible tick could land a base: after a
    ///   header-sequence regression skipped a tick, or on a lineage captured by
    ///   a build that predates the proven reads. Only the first leaves a
    ///   regression WARN behind; the second clears at the layer's next eligible
    ///   tick.
    /// - `stepped_down_link_broken` — a link's generation was absent from the
    ///   listing, or present in the listing and gone at fetch (the
    ///   contract-documented concurrent-prune race). Remediation: check
    ///   retention pressure and whether the store lost artifacts.
    /// - `stepped_down_link_mismatch` — a member's authoritative envelope header
    ///   disagreed with the expected link, the lineage root, the addressed kind,
    ///   or the object id. Includes the per-member adapter-contract-violation
    ///   guard. Remediation: suspect a hand-edited artifact or a non-conforming
    ///   adapter, not retention.
    /// - `stepped_down_decode` — `peek_tail_header` rejected the envelope: an
    ///   unknown `format_version` (SR-6.1 — surfaced, never guessed), a bad
    ///   magic, or a truncated artifact. Remediation: version skew across
    ///   replicas, or storage-level corruption.
    /// - `stepped_down_missing_meta` — the generation's artifact is durable but
    ///   its `.meta.json` companion is absent, so the adapter refused it as a
    ///   typed `Permanent` and the walk stepped down (b.hka (d)). Distinct from
    ///   `stepped_down_decode`: the bytes are fine, the index never landed.
    ///   Remediation: a write interrupted between the artifact PUT and its meta
    ///   PUT — check for replica restarts or adapter errors at the write path,
    ///   not for corruption.
    /// - `stepped_down_budget` — the reserve-bounded abandonment: the remaining
    ///   recovery fetch budget could not fund one more member fetch, or a single
    ///   fetch overran its per-member allowance. What the budget can decline is
    ///   asymmetric by member kind: a speculative *tail* is declined (or cut
    ///   short) whenever funding it would eat into the base's reserved fetch,
    ///   while a *base* is declined only when nothing at all remains — so this
    ///   cell firing does not by itself mean the floor answer was missed.
    ///   Remediation: adapter latency — cross-check
    ///   `checkpoint_adapter_ceiling_elapsed_total{phase="call"}`.
    ///   This is a *degraded but correct* signal: the restore still lands, from
    ///   an older generation.
    /// - `stepped_down_compose` — the composition engine classified the
    ///   assembled lineage `Permanent`; the selection stepped down and the
    ///   caller recomposed.
    /// - `exhausted` — no restorable lineage, **conclusively**: the selection
    ///   adjudicated every full (`Base`) generation the listing held. Covers an
    ///   empty listing, a tails-only listing (no reset-decodable root —
    ///   absorbing the arm `select_restore_source` used to WARN about), and a
    ///   listing whose every candidate broke. The attempt ends
    ///   `not_recoverable` and the client sees a terminal `NOT_FOUND`.
    /// - `exhausted_budget` — the selection ran out of budget before it could
    ///   adjudicate every full generation, so it cannot say the layer is
    ///   unrestorable — only that the store did not answer in time. The attempt
    ///   ends `error` (via `CheckpointError::Transient`) and the client sees a
    ///   retriable `UNAVAILABLE`. Remediation: adapter latency — cross-check
    ///   `checkpoint_adapter_ceiling_elapsed_total{phase="call"}` and
    ///   `stepped_down_budget`. Unlike every other cell here this one is not a
    ///   statement about the stored artifacts, so a rise in it does **not**
    ///   indicate checkpoint loss.
    ///
    /// Every increment joins a `warn!`/`debug!` tracing event carrying the same
    /// spelling in its `outcome` field, so a log line and a metric cell join on
    /// one string.
    pub checkpoint_restore_lineage_total: Counter<u64>,
    /// Total session-checkpoint adapter operations whose publish was refused by
    /// the b.443 watermark — i.e. the operation landed on top of a strictly
    /// fresher one — by `result`.
    ///
    /// `result` vocabulary (zero-seeded, so every cell exists from the first
    /// scrape):
    ///
    /// - `write` — clobbered a fresher PUT; a rebuild was scheduled to restore
    ///   the newer payload.
    /// - `delete` — clobbered a fresher DELETE; the DELETE was re-issued.
    /// - `stale_incarnation` — the `le:{sid}:ckpt` key this operation's stamp
    ///   was minted against is gone or has been re-created, so the stamp is not
    ///   comparable to the live watermark and the publish wrote nothing. A
    ///   rebuild was scheduled (for both kinds — a stale incarnation says
    ///   nothing about what else landed, so the information-preserving arm is
    ///   the only safe one). Unlike `write` / `delete` this is NOT the protocol
    ///   working on a normal race: it means the coordination key was lost
    ///   (FLUSH, failover onto a lagging replica, out-of-band DEL, TTL expiry
    ///   on an idle session), which is a keyspace-integrity signal worth
    ///   investigating even though the durable blob self-heals.
    /// - `exhausted` — a supersession was detected but the per-task repair
    ///   budget was spent, so the durable blob is left stale until the next
    ///   lifecycle event on that session.
    ///
    /// Deliberately its own instrument rather than new cells on
    /// [`Self::checkpoint_writes_total`]: that counter's `kind="session"`
    /// vocabulary is pinned by the SDD, and a supersession is not an outcome of
    /// the adapter call (which succeeded, and already ticked `success` there) —
    /// it is a subsequent ordering verdict on it.
    ///
    /// Alertable at `exhausted` only. A steady trickle of `write` / `delete` is
    /// the protocol working: concurrent lifecycle events on one session
    /// genuinely race, and each refusal is a durable-loss that was caught and
    /// repaired. `exhausted` means a repair loop gave up.
    pub checkpoint_session_publish_superseded_total: Counter<u64>,
    /// Total GC decrements whose Lua script returned the `deleted_underflow`
    /// action (post-decrement ref_count < 0) (b.j9j item 4). Unlabeled.
    ///
    /// A decrement with no increment behind it: an older build's replay, a
    /// legacy bare member, or a bug. Since b.yuo a replayed member answers
    /// `replayed` without decrementing, so the baseline is zero and any
    /// increase pages.
    pub gc_underflow_total: Counter<u64>,
    /// Total GC decrement runs by `outcome` (b.yuo) — see
    /// [`GC_DECREMENT_OUTCOMES`]: `applied` (decremented now), `replayed` (the
    /// member's decrement had already applied; nothing decremented) and
    /// `stale` (refused by the claim-age fence; the member waits for the
    /// re-claim).
    pub gc_decrement_total: Counter<u64>,
    /// Total actions the external-value GC took on state an older build left
    /// in the store (b.1jt), by `event` — see [`GC_OLDER_BUILD_EVENTS`]:
    /// `lock_taken_over` (per takeover of `le:gc:leader`), `claim_dropped`
    /// (per inflight claim dropped without a decrement), `release_stamped` (per
    /// un-aged pending release stamped) and `carrier_dropped` (per release
    /// reference on a retirement carrier dropped without paying). Outside a
    /// rollout or rollback across the GC protocol marker, any increase means
    /// an older build is acting on this store.
    pub gc_older_build_total: Counter<u64>,
    /// Total out-of-memory collections the GC leader ran (b.pa1), by `outcome`
    /// — see [`GC_OOM_COLLECTION_OUTCOMES`]: `freed`, `freed_nothing`,
    /// `nothing_to_collect`, `failed` and `refused`. A collection starts when
    /// any replica saw a store write refused `-OOM` (at most one a second
    /// after one that freed or `failed`, one every 30 s after one that found
    /// nothing to free), collects
    /// released external values at any age, and holds uploads carrying
    /// external values while it runs. Zero-seeded; ticked by the leader.
    pub gc_oom_collections_total: Counter<u64>,
    /// Bytes the GC's out-of-memory collections freed (b.pa1): the summed
    /// lengths of the external values they deleted. Ticked by the leader.
    pub gc_oom_freed_bytes_total: Counter<u64>,
    /// Bytes the external-value release ledger accounts over pending ∪
    /// inflight — released blobs awaiting collection (b.yuo), by service.
    /// Leader-published every GC pass; 0 from non-leaders; not published by a
    /// pass that errored. The quantity the release budget
    /// `LIVE_EDIT_EXT_RELEASE_MAX_BYTES` gates on: uploads carrying external
    /// values are held back while it is at or above the budget, but other
    /// sources — `DeleteLayer`, overwrites and edits that drop references,
    /// reference-count repairs, uploads let through on a failed read of the
    /// total, and waiters admitted together on its first reading below the
    /// budget — can take it past the budget.
    pub ext_release_pending_bytes: Gauge<u64>,
    /// Release members owed a decrement — pending plus inflight (b.yuo), by
    /// service. Same leader-published contract as
    /// [`Self::ext_release_pending_bytes`]. A member re-enqueued while its
    /// claim is held sits in both sets and is counted twice, until that claim
    /// settles.
    pub ext_release_pending_count: Gauge<u64>,
    /// Age in seconds of the oldest pending release member (b.yuo), by
    /// service; 0 when none. Same leader-published contract as
    /// [`Self::ext_release_pending_bytes`]. A level that keeps rising past the
    /// GC age floor means collection is not keeping up.
    pub ext_release_oldest_age_seconds: Gauge<f64>,
    /// Entries on the settle carrier `le:ext:{gc}:settled` — release members
    /// whose decrement replay records await teardown (b.yuo), by service.
    /// Same leader-published contract as [`Self::ext_release_pending_bytes`].
    /// Bounded by the release rate times the replay-record retention; growth
    /// without bound means teardown is failing.
    pub ext_release_settled_count: Gauge<u64>,
    /// Live `{gc}` read hold fields at the GC leader's last pass (b.yuo), by
    /// service: catch-ups re-reading after an external-value miss, each of
    /// which stops the collection of releases newer than its start while it
    /// lives. Same leader-published contract as
    /// [`Self::ext_release_pending_bytes`]. Tells a hold apart from a GC
    /// stall when uploads wait at the release budget; it returns to 0 between
    /// miss waves.
    pub ext_read_holds: Gauge<u64>,
    /// Uploads carrying external values that met a full release budget, by
    /// `outcome` (b.yuo) — see [`EXT_RELEASE_GATE_OUTCOMES`]. An upload
    /// admitted at once is not counted.
    pub ext_release_gate_total: Counter<u64>,
    /// The external-value release budget `H` this replica's upload gate and
    /// GC pressure trigger use, in bytes, by service and `source` (b.yuo) — see
    /// [`EXT_RELEASE_BUDGET_SOURCES`]. Recorded at startup and on every change
    /// a store-memory re-read makes; the source in use carries `H`, every
    /// other source 0. `H = 0` (the budget disabled) is recorded under
    /// `override`.
    pub ext_release_budget_bytes: Gauge<u64>,
    /// The value `live_edit_store_memory_used_ratio` reports at each
    /// collection (b.qhk), set by [`Self::record_store_memory_used_ratio`];
    /// `None` emits no sample, so the series is absent.
    store_memory_used_ratio: Arc<std::sync::Mutex<Option<f64>>>,
    /// Catch-up builds that found an external-value blob missing, by
    /// `outcome` and `read` (b.yuo) — see [`CATCHUP_EXT_VALUE_MISS_OUTCOMES`]
    /// and [`CATCHUP_EXT_VALUE_MISS_READS`]:
    /// `stale_retried` (the catch-up took a `{gc}` read hold and re-read)
    /// and `genuine` (still missing under a hold the store confirmed live;
    /// fetch-on-miss, then `INTERNAL` if that fails), each under `read`
    /// `atomic` or `shared_fenced`.
    pub catchup_ext_value_miss_total: Counter<u64>,
    /// Total `ext_meta` references handed back to the GC pending set, by
    /// `reason` (b.qpf) — see [`EXT_REF_RELEASE_REASONS`].
    ///
    /// Each tick is one member enqueued on `le:ext:{gc}:pending`, i.e. one
    /// `ref_count` decrement the GC sweep now owes. Not alertable on its own —
    /// a healthy edit-heavy workload releases constantly. What it buys is the
    /// ability to read the ratio against `ext` blob growth: a store whose
    /// `ext:meta` count climbs while this stays flat is the b.qpf leak
    /// reappearing.
    pub ext_ref_released_total: Counter<u64>,
    /// Total owed `ext_meta` references the service could neither RECORD on
    /// `le:ext:{gc}:pending` nor record on a durable carrier (b.qpf; narrowed
    /// by b.yda and b.yuo Part R).
    ///
    /// The leak signal, and alertable at `increase(...) > 0`. Unlabeled — every
    /// counted site is one terminal point with one meaning.
    ///
    /// **The narrowing.** Before b.yda a retirement-path enqueue failure was
    /// terminal by construction and ticked here. Since b.yuo Part R a
    /// retirement records its owed members on `le:{oid}:obj:extret`, on the
    /// retiring object's OWN slot, BEFORE it enqueues them
    /// (`scripts/ext_retire_begin.lua`), so an enqueue whose outcome is in doubt
    /// — or one not sent at all because the retirement passed its send window —
    /// ticks [`Self::ext_ref_release_deferred_total`] instead and the orphan
    /// sweep settles the carrier. This counter fires on the retirement path only
    /// when that begin itself fails or answers `busy`. The zero baseline and the
    /// `increase(...) > 0` alert are unchanged by the narrowing.
    ///
    /// **Retry amplification survives on exactly those ticks.** A begin that
    /// failed did not write `ext_gc_scheduled`, so the short-circuit that would
    /// suppress a re-derivation is unarmed; `delete_layer`'s retry cascade then
    /// re-runs the whole retirement when the structural UNLINK also fails —
    /// the natural shape when the object's own slot is the one that is down —
    /// re-derives the same members and re-ticks the whole batch, bounded by
    /// `LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS`. (The CreateLayer-overwrite
    /// cleanup task and the orphan sweep visit once and do not loop.) One tick
    /// per object per attempt that reaches that arm, therefore — not one per
    /// object.
    ///
    /// **Unit: references, not enqueue calls.** One failed batch that owed N
    /// releases adds N, matching [`Self::ext_ref_released_total`]'s unit so the
    /// guide's `failed / released` ratio divides like with like, and so "how
    /// many ticked" reads as "how many references leaked".
    ///
    /// **Still an upper bound.** A script is atomic server-side, so a batch the
    /// server *processed* either landed whole or not at all — but the tick is
    /// driven by the *client's* verdict: a fred `Timeout` / `IO` error on a
    /// commit-path enqueue that did in fact land reports N while leaking
    /// nothing, and a retirement begin that ran but whose reply was lost
    /// reports N while its carrier — which the orphan sweep's keyspace backstop
    /// finds — still holds the obligation. Read a tick as "up to one
    /// `ref_count` the service could not confirm it recorded".
    ///
    /// # Exactly what is counted
    ///
    /// - **Commit-path enqueue failures** — `LayerStore::enqueue_ext_releases`,
    ///   after its one in-line retry. Counted per reference in the batch. There
    ///   is no carrier on this path, and retirement does not recover these
    ///   releases (a `rewrite` hash already left `obj:ext` inside the commit; a
    ///   `surplus_acquire` increment is owed by no later event), so only the
    ///   reconciler drains them.
    /// - **Retirement-path obligations that could not be recorded** —
    ///   `storage::layer::schedule_old_object_ext_gc`, when its begin
    ///   (`scripts/ext_retire_begin.lua`) fails, or answers `busy` because the
    ///   `object_id` already has a release record: a carrier from an earlier
    ///   lifecycle (the orphan sweep settles it), or a token HASH with no
    ///   carrier that only an older build leaves (it may belong to this same
    ///   incarnation; nothing in this build removes it — b.9ud sub-task 3 plans
    ///   the only safe remover, the restore's residue wipe extended to void the
    ///   tokens and delete the HASH, not built — so every retirement of that id
    ///   ticks here).
    ///   Counted per reference, once per attempt that reaches that
    ///   arm — see the amplification note above for why `delete_layer`'s
    ///   cascade can reach it more than once for one object.
    ///
    /// # What is NOT counted
    ///
    /// - **Retirement pre-schedule read failures** — a failed
    ///   `HGET obj:hdr ext_gc_scheduled` or reference walk (the value
    ///   indices, their rows, `HSCAN obj:ext`; `storage::retirement`)
    ///   aborts before the release set exists, so the owed N is unknown and no
    ///   count can honestly be derived. (The three retirement callers log a
    ///   WARN and proceed either way, so this shape leaks too — it is simply
    ///   not measurable here. The reconciler drains it.)
    /// - **A retirement enqueue whose outcome is in doubt, or that was not sent
    ///   because the retirement passed its send window** — see
    ///   [`Self::ext_ref_release_deferred_total`]; the obligation is on its
    ///   carrier, so it is not a leak.
    /// - **The orphan sweep's re-drive failing, or not sent because the sweep
    ///   passed its send window** — the carrier keeps every member and the
    ///   attempt's token, and the next tick resolves it.
    /// - **A recorded obligation the retire-retry settle CANCELLED** because the
    ///   object id was live again — see
    ///   [`Self::ext_ref_release_cancelled_total`]. That one does leak, but it
    ///   is a deliberate policy drop rather than a failure to record, and it
    ///   carries its own counter so the two are not read as one signal.
    /// - **A process dying between a commit's `EXEC` and its enqueue** — the
    ///   accepted residual; nothing observes it from inside.
    /// - **Undecodable release tokens, value-row `hash` columns and `obj:ext`
    ///   fields** — dropped at the point of decode with an ERROR (Lua release
    ///   tokens) or a WARN (row `hash` columns, `obj:ext` fields) rather than
    ///   released as an unidentified hash. They have their own counter,
    ///   [`Self::ext_ref_undecodable_total`], because no recovery path is
    ///   expressible for them: the hash is unidentifiable at the drop site.
    ///
    /// # Why a tick still means a leak
    ///
    /// A commit whose enqueue failed has already landed and is never failed for
    /// this, and no later event owes its decrement: retirement releases only the
    /// hashes the object still references. A *retirement* obligation that
    /// reached this counter is one no carrier recorded, so the structural
    /// teardown removes `obj:hdr`, `obj:ext` and every value row with nothing
    /// left to re-derive the hashes from. Read a tick as UP TO one
    /// `ref_count` that will not be decremented — until the reference-count
    /// reconciler's next pass observes the surplus and drains it (b.yda), which
    /// is a repair on a scale of hours, not the seconds a `pending` member takes.
    /// Residual hardening of that carrier is planned on b.9ud and not built.
    pub ext_ref_release_enqueue_failed_total: Counter<u64>,
    /// Total owed `ext_meta` references whose retirement-path enqueue ended in
    /// doubt — both attempts failed, so whether either ran is unknown — or was
    /// NOT SENT at all because the retirement passed its send window
    /// (`external_values::RETIRE_INTENT_SEND_WINDOW`), or whose first attempt
    /// failed in doubt and whose retry was then not sent because the window
    /// had passed, and was left on the retiring object's own carrier
    /// (`le:{oid}:obj:extret`, written before the enqueue) for the orphan sweep
    /// to settle (b.yda; b.yuo Part R). In the pure not-sent case nothing was
    /// enqueued; in the mixed case the first attempt may have run. Either way
    /// the sweep settles the recorded releases, resolving every attempt before
    /// it sends anything.
    ///
    /// The recoverable half of what
    /// [`Self::ext_ref_release_enqueue_failed_total`] used to absorb whole. Not
    /// a leak on its own — read it against
    /// [`Self::ext_ref_release_recovered_total`],
    /// [`Self::ext_ref_release_already_enqueued_total`] and
    /// [`Self::ext_ref_release_cancelled_total`], which between them should
    /// track it with a lag of about one orphan-sweep interval. A `deferred`
    /// rate that none of them follows means the drain is not running (no
    /// leader, a wedged sweep) or is failing, and the backlog gauge
    /// [`Self::ext_retire_retry_backlog`] confirms which.
    ///
    /// Unit: references, matching its siblings — one in-doubt enqueue owing N
    /// adds N.
    ///
    /// Its completion signal has THREE arms: the sweep found an attempt had
    /// landed after all ([`Self::ext_ref_release_already_enqueued_total`]), or
    /// re-drove the members ([`Self::ext_ref_release_recovered_total`]), or
    /// cancelled them ([`Self::ext_ref_release_cancelled_total`]) because the
    /// object id came back to life. `deferred ≈ recovered + cancelled +
    /// already_enqueued` is the identity to read these four by. The
    /// completions can run ABOVE `deferred`: the sweep also settles carriers
    /// that were never deferred — a begin whose reply was lost, a crash between
    /// a begin and its clear, a failed or superseded clear, a slow retirement
    /// the sweep overtook.
    pub ext_ref_release_deferred_total: Counter<u64>,
    /// Total recorded `ext_meta` references the orphan sweep's retire-retry
    /// phase re-drove onto `le:ext:{gc}:pending` (b.yda; b.yuo Part R), each
    /// only after every recorded earlier attempt of the carrier resolved void,
    /// and only from this build's carriers, which record every attempt (an
    /// older build's carrier is never re-driven: it is cleared or dropped,
    /// b.1jt).
    ///
    /// One completion arm of [`Self::ext_ref_release_deferred_total`]. Unit:
    /// references. A recovered reference is one the GC now owes exactly as if
    /// the original retirement enqueue had succeeded — the member is
    /// byte-identical (`{hash_hex}:{object_id}:ret`).
    pub ext_ref_release_recovered_total: Counter<u64>,
    /// Total recorded `ext_meta` references the orphan sweep's retire-retry
    /// phase found ALREADY enqueued — an earlier attempt of the carrier had
    /// landed although its caller never learned it — and therefore cleared
    /// without enqueuing again (b.yuo Part R). Unit: references.
    ///
    /// One completion arm of [`Self::ext_ref_release_deferred_total`]. Each
    /// tick is a second decrement that did NOT happen: before Part R the sweep
    /// re-enqueued such members, and once their replay records were torn down
    /// the GC decremented them twice.
    pub ext_ref_release_already_enqueued_total: Counter<u64>,
    /// Total recorded `ext_meta` references the orphan sweep's retire-retry
    /// phase **dropped** because the object id was live again when every
    /// recorded earlier attempt had resolved void (b.yda). Unit: references.
    ///
    /// The other completion arms of [`Self::ext_ref_release_deferred_total`]
    /// are [`Self::ext_ref_release_recovered_total`] and
    /// [`Self::ext_ref_release_already_enqueued_total`]. A recorded obligation
    /// is owed only while its `object_id` is dead; an id that is live again —
    /// re-materialised by a checkpoint restore, or never fully torn down — has
    /// inherited the standing, unreleased `+1` per hash those members stood
    /// for, so paying them would be an unmatched decrement that deletes
    /// `ext_meta` + `ext_data` under a live reference. The settle script
    /// therefore cancels them (`scripts/ext_retire_settle.lua`). A retirement
    /// slower than the sweep's in-progress age that a sweep overtakes can be
    /// cancelled while its object is still live, and its teardown then leaks
    /// the references too.
    ///
    /// **Leak direction, and read together with
    /// [`Self::ext_ref_deficit_total`].** Each cancelled reference is at most
    /// one `ref_count` the retirement path will now never decrement, so it
    /// lands — if it was genuinely owed at all — in
    /// [`Self::ext_ref_surplus_total`], which the reference-count reconciler
    /// drains on a scale of hours. What it is NOT is a deficit: paying a
    /// cancelled obligation is exactly the over-release
    /// [`Self::ext_ref_deficit_total`] alerts on, so a non-zero rate here
    /// alongside a flat zero deficit is the gate working as designed, and is
    /// the pair to look at together. A rate that climbs steadily means restores
    /// (or unfinished teardowns) keep landing on ids that had deferred
    /// obligations, and the reconciler is carrying that work.
    pub ext_ref_release_cancelled_total: Counter<u64>,
    /// Total external-value references dropped because the hash could not be
    /// decoded at the point it was read, by `site` (b.yda) — see
    /// [`EXT_REF_UNDECODABLE_SITES`].
    ///
    /// These are the corrupt-keyspace drop sites: a Lua release token, a value
    /// row's `hash` column, an `obj:ext` field read while planning a commit, and
    /// an `obj:ext` field read while scheduling a retirement. Each drops rather
    /// than guessing, because a mis-decoded hash would decrement the WRONG
    /// blob — data loss — while dropping leaks one reference, which the
    /// reconciler drains.
    ///
    /// **No dedicated recovery path exists, and none can**: the hash is
    /// unidentifiable at the drop site, so nothing can name the blob to release
    /// it. The reconciler drains these as a matter of course — an object that
    /// cannot name a hash simply does not count toward that hash's referenced
    /// total, so the surplus shows up in the ordinary way.
    ///
    /// Expected to be zero. Any non-zero value is keyspace corruption and the
    /// accompanying ERROR/WARN log names the undecodable text.
    pub ext_ref_undecodable_total: Counter<u64>,
    /// Number of `le:ext:meta:*` keys the reference-count reconciler enumerated
    /// on its last pass (b.yda), by service.
    ///
    /// Leader-published: sampled once per pass by the replica that held
    /// `le:gc:leader`, published as `0` by every non-leader for the same reason
    /// [`Self::checkpoint_ext_tombstone_backlog`] is, and **not published at
    /// all** by a pass that errored — unknown is not zero.
    ///
    /// Read as a pair with [`Self::ext_referenced_hashes_total`]: the gap
    /// between them is the population of blobs no live object names, which is
    /// normal transiently (a release awaiting its GC tick) and a leak when it
    /// grows monotonically.
    pub ext_meta_total: Gauge<u64>,
    /// Number of distinct external hashes named by at least one live object,
    /// as counted by the reconciler's object walk (b.yda), by service.
    ///
    /// Same leader-published contract as [`Self::ext_meta_total`]. Derived from
    /// the union of every object's value rows and its `obj:ext` index — the same
    /// union `schedule_old_object_ext_gc` releases at retirement, so the two can
    /// never disagree about what an object owes.
    pub ext_referenced_hashes_total: Gauge<u64>,
    /// Summed positive surplus across every `ext:meta` — how many `ref_count`
    /// units stand above the live-object reference count, after subtracting the
    /// releases already owed on `pending` ∪ `inflight` (b.yda), by service.
    ///
    /// Same leader-published contract as [`Self::ext_meta_total`]. This is the
    /// leak, measured. In a healthy fleet it hovers near zero and any excursion
    /// is drained within two passes plus a grace; a level that ratchets upward
    /// pass over pass means some release path is failing faster than the
    /// reconciler repairs.
    pub ext_ref_surplus_total: Gauge<u64>,
    /// Summed **confirmed** deficit across every `ext:meta` — how many
    /// `ref_count` units stand BELOW the live-object reference count and were
    /// still standing, at the same amount, a full grace after the pass first
    /// observed them (b.yda), by service.
    ///
    /// Same leader-published contract as [`Self::ext_meta_total`]. The
    /// confirmation protocol filters the single-pass noise — a single pass's
    /// negative surplus is routine on a busy store (see
    /// [`EXT_RECONCILE_OUTCOME_DEFICIT_OBSERVED`] for the benign sources), so
    /// only the confirmed figure reaches this gauge. First observations are
    /// visible as `ext_reconcile_total{outcome="candidate_registered"}` and as
    /// backlog on `le:ext:{gc}:recon`; the ones that turn out benign tick
    /// `deficit_cleared`.
    ///
    /// **Alert on a deficit that RECURS**, not on one confirmed report:
    /// `max_over_time(...[N intervals])` non-zero across several reconciler
    /// intervals. Confirmation proves the amount held still across one grace; it
    /// does not prove the deficit is a persistent one. A hash that churns — its
    /// last reference dropped and re-acquired at a comparable rate — can present
    /// the same mid-pass double-count at the same amount on the confirming pass
    /// as on the registering one, and confirm a transient. Distinguishing those
    /// needs the per-hash obligation to be attributed to the object that owes
    /// it, which this pass does not do; b.1sv tracks that. Recurrence is the
    /// available proxy: a churn coincidence does not repeat window after window,
    /// and a real over-release does not go away.
    ///
    /// A confirmed deficit is an over-release that already happened. It is
    /// confirmed only on a live meta, so the value is still stored when it is
    /// reported (meta and data are created and deleted together); but once the
    /// count reaches 0 while live objects still reference the hash, the GC
    /// collects the value under a live reference — and it may already have been
    /// collected once and stored again at too low a count (an upload of the same
    /// content, or fetch-on-miss). The reconciler deliberately never repairs
    /// this direction — an increment racing the GC's delete would resurrect a
    /// meta whose data is gone — so the gauge is the whole response.
    ///
    /// `max_over_time` is also what makes the series readable at all, quite
    /// apart from the alert shape above: a confirmed candidate is retired once
    /// it has been counted, so a standing deficit re-registers, waits out
    /// another grace and reports again, publishing zero on the passes in
    /// between.
    pub ext_ref_deficit_total: Gauge<u64>,
    /// Current size of `le:ext:{gc}:retire-retry` — retiring objects whose
    /// release enqueue failed and whose obligations are waiting on the orphan
    /// sweep's retire-retry phase (b.yda), by service.
    ///
    /// Sampled once per orphan-sweep tick by the leader and published as `0` by
    /// non-leaders, for the same non-decaying-gauge reason as
    /// [`Self::checkpoint_ext_tombstone_backlog`]. Read as a derivative: a
    /// backlog that rises and falls with store hiccups is the carrier working; a
    /// backlog that only rises means the drain is failing and the deferred
    /// obligations are aging.
    pub ext_retire_retry_backlog: Gauge<u64>,
    /// Total reference-count reconciler outcomes, by `outcome` (b.yda) — see
    /// [`EXT_RECONCILE_OUTCOMES`].
    ///
    /// The pass's own health signal, distinct from the gauges it publishes:
    /// `enumeration_failed` is alertable per occurrence, `deficit_observed` is
    /// alertable on RECURRENCE across several intervals (it confirms an amount,
    /// not a persistence — see [`EXT_RECONCILE_OUTCOME_DEFICIT_OBSERVED`]),
    /// sustained `deferred` means one batch no longer covers the fleet's
    /// candidate backlog, and `repaired` is the drain doing its job.
    ///
    /// Read `candidate_registered` as **candidates OFFERED**, not new ones: the
    /// pass ticks it by the size of the candidate set it computed, and a
    /// candidate is re-offered on every pass for as long as its condition holds
    /// (only the `ZADD NX` behind it distinguishes first observation from
    /// re-offer, and that happens Valkey-side). So a surplus parked on a
    /// `skipped_*` arm's neighbour, or a standing confirmed deficit that
    /// re-registers after each report, re-ticks this cell every pass — a level,
    /// not a rate of arrival. `candidate_registered` and `healed` in balance
    /// with no `repaired` is the two-observation guard correctly declining to
    /// act on transients; `candidate_registered` and `deficit_cleared` in
    /// balance with no `deficit_observed` is the same guard doing the same job
    /// on the deficit side, and is the expected shape on a busy store.
    ///
    /// `skipped_recent_store` moving is the expected shape on a busy store — the
    /// pass declining to judge hashes something wrote to inside the grace.
    /// `skipped_unreadable_stamp` moving is different in kind: nothing in the
    /// service can write an unparseable `last_store_ms`, so a sustained count
    /// there is a keyspace-integrity signal, not a load one.
    ///
    /// `skipped_index_drift` moving means a surplus is being HELD because some
    /// object's per-object reference index disagrees with its rows and could
    /// not be repaired this pass (see [`Self::ext_index_recount_total`]'s
    /// `too_large` / `not_live` / `withheld` cells). Holding the surplus leaks
    /// rather than
    /// loses; it does not stop a commit path releasing on the unrepaired
    /// index.
    pub ext_reconcile_total: Counter<u64>,
    /// Total per-object reference-index recounts, by `site` and `outcome` — see
    /// [`EXT_INDEX_RECOUNT_SITES`] and [`EXT_INDEX_RECOUNT_OUTCOMES`].
    ///
    /// A recount runs when the reconciler's object walk finds an object whose
    /// `le:{oid}:obj:ext` disagrees with its rows (`site="reconcile"`), and
    /// after a torn txn commit (`site="torn_commit"`). A recount that finds
    /// drift on a live object within its row ceiling repairs the index
    /// atomically; none releases anything. `clean` at the reconcile site is the
    /// expected answer for a legacy dual-shape row; `too_large` / `not_live`
    /// there hold the object's hashes from surplus drain. On an object with a
    /// malformed `hash` column a recount applies raises only and withholds
    /// every lowering; any recount that saw such a row records `withheld`
    /// (never `clean` or `repaired`, whatever it raised), at both sites, and
    /// logs at WARN. At the reconcile site it also holds the object's
    /// mismatched hashes like `too_large`.
    pub ext_index_recount_total: Counter<u64>,
    /// Total per-object reference-index fields corrected by a recount, by
    /// `site` and `direction` (`under` | `over`) — see
    /// [`EXT_INDEX_DRIFT_DIRECTIONS`].
    ///
    /// A burst on the first reconciler pass after upgrading a deployment that
    /// holds layers from before 0.1.0 (when the index did not yet exist) is
    /// migration. `direction="under"` recurring after
    /// that means something still produces per-object drift in the dangerous
    /// direction.
    pub ext_index_drift_total: Counter<u64>,
    /// Total time-sample rows the reference-count reconciler's signed-zero
    /// merge handled, by `outcome` — see
    /// [`TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOMES`].
    ///
    /// `-0.0` and `0.0` are one USD time code. This build stores every time
    /// at `+0.0`; a `-0.0` row is one an older build wrote. Each reconciler
    /// pass merges such rows into one row at `+0.0` (`merged` for a pair,
    /// `rekeyed` for a lone `-0.0` row), changing nothing a reader serves.
    /// A burst on the first passes after an upgrade is the migration;
    /// `merged` / `rekeyed` recurring after the whole fleet runs this build
    /// means something still writes `-0.0`. `changed` and `fenced` are retried
    /// on the next pass; `undecodable` and `failed` recurring need an operator.
    pub time_sample_signed_zero_merge_total: Counter<u64>,
    /// Total object-id-counter repairs triggered by a `CreateLayer` that found
    /// `le:counter:object_id` absent (b.b2j), by `outcome`.
    ///
    /// An absent counter is evidence of keyspace loss **only when something
    /// durable could collide with a fresh one** — nothing in the service ever
    /// deletes the counter, but a store that has never checkpointed one has
    /// nothing to protect. The `outcome` carries that verdict, so read the
    /// cells rather than the family total. The loss-evidence cells are this
    /// service's primary signal for the surviving-pods keyspace-loss shape — a
    /// Valkey restart without a pod restart, which readiness alone does not
    /// surface (readiness recovers as soon as Valkey answers pings again, empty
    /// keyspace or not).
    ///
    /// `outcome` values — the page-on-first-occurrence alert covers exactly the
    /// four ERROR-level cells (`repaired_seeded`, `repaired_noop`, `failed`,
    /// `failed_after_repair`). The other four are deliberately off the page:
    /// `wait_timeout` is a WARN-level dashboard signal with its own
    /// sustained-rate (`for: 10m`) warning rule, the two `*_initialized` cells
    /// are expected once on a fresh deployment, and `already_repaired` is
    /// bookkeeping. Operator readings (`docs/service/docs/metrics-reference.md`;
    /// alert rules in `docs/service/docs/monitoring.md`):
    /// - `repaired_seeded` — re-seeded from the checkpoint counters sidecar.
    ///   The sidecar's existence is what makes the absence real loss.
    ///   **Alertable**; logged at ERROR.
    /// - `repaired_noop` — sidecar read, but a peer had already pushed the
    ///   counter at or above the computed seed. Same evidence, no write of our
    ///   own. **Alertable**; logged at ERROR.
    /// - `failed` — the repair attempt ran to completion without leaving a
    ///   usable counter behind. A **catch-all, not a checkpoint-store
    ///   reachability signal**: six distinct arms land here, so triage starts at
    ///   the accompanying ERROR log line, which names which fired. The counter
    ///   re-check errored; the sidecar read errored or exceeded its fetch
    ///   budget; the sidecar was present but corrupt or unparseable
    ///   (`SeedError::ParseFailed`); the Valkey-side conditional seed `EVAL`
    ///   failed (`SeedError::Seed`); the establish-at-zero write failed on a
    ///   store with no sidecar; or the attempt task ended without reporting (a
    ///   panic, or runtime shutdown mid-attempt). The `CreateLayer` surfaced
    ///   `UNAVAILABLE`. **Alertable**; logged at ERROR.
    /// - `wait_timeout` — a `CreateLayer` stopped waiting on the in-flight
    ///   repair and surfaced `UNAVAILABLE`. The effective budget is
    ///   `Config::object_id_counter_repair_wait_budget_within`: ⅔ × min(
    ///   `LIVE_EDIT_RPC_DEADLINE_MS`, what is actually left of *this* request's
    ///   deadline — which includes any shorter `grpc-timeout` the client sent).
    ///   It is re-derived per call, so two concurrent `CreateLayer`s waiting on
    ///   the same attempt can give up at different times. This is a caller-side
    ///   give-up, **not** a verdict on the counter: the attempt keeps running on
    ///   its detached task and lands on one of the cells above when it finishes,
    ///   and the client's retry joins that same attempt. Logged at WARN.
    ///   **Not paged** — a single occurrence says only that one caller stopped
    ///   waiting while the attempt was still running, and paging here would
    ///   page twice for one event. It gets a WARN-level sustained-rate rule
    ///   (`for: 10m`) instead. A sustained rate has **two** causes and they need
    ///   different remediation, so triage must distinguish them: either the
    ///   checkpoint store is slow enough that no `CreateLayer` can outlast a
    ///   repair (fix the store — correlate with the adapter latency/error
    ///   families and with a `failed` cell arriving late), **or** clients are
    ///   setting `grpc-timeout`s short enough that ⅔ of what remains at
    ///   allocation is less than a healthy repair takes (fix the callers'
    ///   deadlines — here the repair itself completes normally, so the cells
    ///   above show `repaired_*` rather than `failed`, and the same clients'
    ///   retries eventually succeed).
    /// - `failed_after_repair` — the repair reported success and the counter
    ///   was absent *again* on the retry one round trip later. The most severe
    ///   verdict here: the backing store is losing writes, not merely empty.
    ///   The `CreateLayer` surfaced `UNAVAILABLE` without retrying further.
    ///   **Alertable**; logged at ERROR from `LayerStore::allocate_object_id`,
    ///   the only site that can observe it (both the hooked-repair and
    ///   unhooked-initialize paths returned `Ok` before it).
    /// - `repaired_initialized` — no sidecar exists on the checkpoint store, so
    ///   no durable object id can collide; the counter was established at `0`.
    ///   **Not by itself evidence of loss** — this is the expected once-per-
    ///   deployment outcome on a fresh store with an adapter configured whose
    ///   counters-sync loop has not yet run. Logged at WARN. On a store that is
    ///   *not* fresh it means the sidecar itself went missing, which is a
    ///   separate checkpoint-store investigation.
    /// - `unhooked_initialized` — no repair hook was wired
    ///   (`LIVE_EDIT_CHECKPOINT_ADAPTER=Disabled`), so the counter was
    ///   established at `0` without consulting any checkpoint state. Same
    ///   fresh-store reading as above. Logged at WARN.
    /// - `already_repaired` — a **new** attempt's opening re-check found the
    ///   counter already above zero, so a peer task or peer replica had
    ///   repaired it first. Counts the *detection*, not a second repair.
    ///   Rarer than the label suggests: callers arriving while an attempt is
    ///   in flight join that attempt's `watch` channel and receive its real
    ///   outcome, so they land on whichever cell it recorded — this cell fires
    ///   only for a caller that arrives after the previous attempt finished and
    ///   spawns a fresh one. Logged at WARN.
    ///
    /// Deliberately unlabeled by `service`: the scrape target already
    /// attributes the series to a pod, and the same series must be writable
    /// from the `LayerStore` fallback path, which holds no service label.
    pub object_id_counter_repair_total: Counter<u64>,
    /// Total service-identity token file parse failures observed by the
    /// `ServiceTokenProvider` polling task (Epic `t1.fgj.dp`, SDD SR-6.2).
    /// Unlabeled.
    pub service_token_parse_errors_total: Counter<u64>,
    /// Total external values whose backup copy the mirror dropped, by `reason`
    /// ([`EXT_MIRROR_DROP_REASONS`]): its channel full, its byte budget full,
    /// the value larger than the whole budget, or still queued at shutdown
    /// (SDD §22.17). The backup repair re-sends a dropped copy while Valkey
    /// holds the value, except an oversized one. Every cell is pre-registered.
    pub checkpoint_ext_mirror_dropped_total: Counter<u64>,
    /// Bytes charged to the mirror's byte budget right now, by `state`
    /// ([`EXT_MIRROR_STATES`]), the per-item overhead allowance included.
    /// Their sum never exceeds `checkpoint_ext_mirror_budget_bytes`.
    pub checkpoint_ext_mirror_bytes: UpDownCounter<i64>,
    /// The mirror's byte budget, derived at startup from the pod memory limit
    /// (`config::memory_plan`); 0 when the mirror is not wired.
    pub checkpoint_ext_mirror_budget_bytes: Gauge<u64>,
    /// Total worker-side `put_ext_blob` failures during ext-value mirroring
    /// (Epic `t1.fgj.zc` / SDD SR-3.1). Unlabeled.
    pub checkpoint_ext_mirror_errors_total: Counter<u64>,
    /// External values the backup repair handled, by `outcome`
    /// ([`EXT_BACKUP_REPAIR_OUTCOMES`]; `checkpoint::ext_backup_repair`, SDD
    /// §22.17). Every cell is pre-registered.
    pub checkpoint_ext_backup_repair_total: Counter<u64>,
    /// External values older than the repair's age gate with no confirmed
    /// backup copy at the end of the leader's last complete repair pass, by
    /// `service`. Non-leaders publish 0; a pass that did not complete
    /// publishes nothing.
    pub checkpoint_ext_backup_owed: Gauge<u64>,
    /// Failed writes of a value's backup confirmation, by `op`
    /// ([`EXT_BACKUP_MARK_OPS`]; `storage::ext_backup`). Every cell is
    /// pre-registered.
    pub checkpoint_ext_backup_mark_failed_total: Counter<u64>,
    /// Total external-value `read_batch` miss recoveries via the checkpoint
    /// adapter, by `result` (`recovered` | `adapter_miss` | `adapter_transient`
    /// | `adapter_permanent` | `adapter_other`) (SDD SR-3.2).
    pub checkpoint_ext_recovery_total: Counter<u64>,
    /// External-value references checked by the CreateLayer / UpdateLayer
    /// upload content check, by `result` ([`EXT_VALUE_VERIFY_RESULTS`]), once
    /// per checked reference in both `LIVE_EDIT_EXT_VALUE_VERIFY` modes (SDD
    /// §11.1). Every cell is pre-registered.
    pub ext_value_verify_total: Counter<u64>,
    /// Per-replica gauge of service-identity token availability (Task
    /// `t2.fgj.dp.s7`). `1` when the sidecar-written token file is currently
    /// readable and parsed into a non-expired token, `0` otherwise.
    ///
    /// Unconditional instrument, but recorded ONLY by the health supervisor and
    /// ONLY when a [`ServiceTokenProvider`] exists (i.e. when
    /// `LIVE_EDIT_SERVICE_TOKEN_FILE` is set). Because an OTel gauge emits no
    /// Prometheus series until first `record()`, the `/metrics` family is
    /// absent on replicas without the env var — preserving the pre-migration
    /// "only present when a provider is constructed" contract without a
    /// conditional-registration seam.
    ///
    /// [`ServiceTokenProvider`]: crate::auth::service_token::ServiceTokenProvider
    pub service_token_available: Gauge<i64>,
    /// Seconds until the currently cached service-identity token expires, by
    /// `role` (`valkey` | `storage-api` | `unspecified`). Signed: negative once
    /// the cached token has expired, so the series crosses zero exactly when
    /// that role's credential dies (b.d6b).
    ///
    /// Recorded by each [`ServiceTokenProvider`]'s polling task at its loop
    /// head, from values that loop already computes — so it costs nothing extra
    /// and stays live between phase transitions rather than only at them.
    /// Labeled by `role` rather than unlabeled because the two production
    /// instances have wholly different severities: expiry on the `valkey`
    /// instance means a crash-looping pod, on `storage-api` a degraded
    /// checkpoint path.
    ///
    /// Carries no secret — a derived duration at exactly the granularity the
    /// stale-phase INFO already logs as `secs_to_expiry`, never the
    /// `expires_on` timestamp and never token bytes (SR-6.2).
    ///
    /// Like [`Self::service_token_available`], the OTel gauge emits no
    /// Prometheus series until first `record()` — and the recorder only fires
    /// once a token has actually been cached (it is inside the polling task's
    /// `if let Some(expires_on) = cached_expiry`). So the series for a role is
    /// absent not only on replicas that construct no provider, but for any
    /// provider that has never once parsed a token: a token file that never
    /// appears, or that has only ever failed to parse, produces **no series at
    /// all** — never a zero, and never a negative.
    ///
    /// Absence is therefore ambiguous by construction and must not be alerted
    /// on as if it meant "expired". Alert on the *value* crossing zero for the
    /// credential-dying condition; for the never-had-a-token condition use
    /// [`Self::service_token_available`], which the health supervisor records
    /// on every tick — including `0` — for as long as a provider is wired to
    /// it.
    pub service_token_expires_in_seconds: Gauge<i64>,
    /// Prolongation headroom of the most recent observed token rotation, by
    /// `role` (`valkey` | `storage-api` | `unspecified`): seconds between the
    /// moment the polling task caches a *replacement* token and the moment the
    /// token it replaced would have expired (b.is1).
    ///
    /// This is the quantity the deployment recommendation is actually about.
    /// A live connection AUTHed with the old token is dropped by the backing
    /// store at that token's expiry, so a periodic re-AUTH can only carry the
    /// fresh credential onto an existing connection if it fires inside this
    /// headroom. Recording the *observed* headroom — pickup delay included,
    /// since it is measured from the pickup instant — is strictly tighter than
    /// any static comparison of the configured leads, and it is the only form
    /// available to the service at all: the sidecar's refresh lead is not a
    /// live-edit config value and cannot be validated at startup.
    ///
    /// Signed and unclamped: negative means the replacement landed *after* the
    /// old token had already died, i.e. every pooled connection was dropped and
    /// re-established rather than prolonged.
    ///
    /// Recorded once per observed rotation (a `Some(prev) → Some(curr)`,
    /// `curr != prev` transition of the cached expiry), so a healthy `valkey`
    /// instance emits one point per ~24 h token lifetime. Like
    /// [`Self::service_token_expires_in_seconds`], the series is absent until
    /// the first record — a provider that has cached at most one token has by
    /// definition observed no rotation, so absence means "no rotation seen yet",
    /// never "headroom was zero".
    ///
    /// Carries a derived duration only; never the `expires_on` timestamp and
    /// never token bytes (SR-6.2).
    pub service_token_refresh_headroom_seconds: Gauge<i64>,
}

impl Metrics {
    /// [`Metrics::try_new`] for tests, benches and examples, which keep the
    /// OTLP endpoint variables unset or valid.
    ///
    /// # Panics
    ///
    /// When [`Metrics::try_new`] fails: an invalid
    /// `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`, or a push exporter that cannot
    /// be built for it.
    pub fn new(service_label: &str) -> Self {
        Self::try_new(service_label).unwrap_or_else(|error| panic!("{error}"))
    }

    /// Create the dual-reader meter provider and all metric instruments.
    ///
    /// `service_label` — the value for the `service` label (pod name /
    /// hostname), pre-seeded on the label-bearing families below. The
    /// `/metrics` scrape reader is owned internally; retrieve it via
    /// [`Metrics::prometheus_exporter`] to wire the HTTP endpoint.
    ///
    /// An invalid `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`, or a push exporter
    /// that cannot be built for it, is an `Err` naming the variable and the
    /// cause; `main` reports it as a configuration error and exits.
    pub fn try_new(service_label: &str) -> Result<Self, String> {
        let (provider, meter, prometheus_exporter, otlp_export_reporter) = build_meter_provider()?;

        let endpoint_calls = meter
            .u64_counter("live_endpoint_calls_total")
            .with_description(
                "Total number of gRPC endpoint calls, by endpoint, service, and grpc_code",
            )
            .build();
        let endpoint_duration_us = meter
            .f64_histogram("live_endpoint_duration_microseconds")
            .with_description(
                "gRPC endpoint call duration in microseconds, by endpoint, service, and grpc_code",
            )
            .build();
        let inflight_rpcs = meter
            .i64_up_down_counter("live_edit_inflight_rpcs")
            .with_description("Number of in-flight gRPC RPCs, by endpoint and service")
            .build();
        let subscriber_updates = meter
            .u64_counter("live_subscriber_updates_total")
            .with_description(
                "Steady-state commits delivered to SubscribeReadLayer subscribers, once per commit \
                 per subscriber after the whole commit was enqueued to its stream. Excludes \
                 catch-up content (including a re-bootstrap's full state), self-echo, duplicate \
                 drops, object-id mismatches, and sends that did not complete. An empty commit \
                 from a client is delivered (as an empty delta at its sequence) and counted",
            )
            .build();
        let diff_case_total = meter
            .u64_counter("live_diff_case_total")
            .with_description("Total number of diff computations by case")
            .build();
        let diff_build_duration_seconds = meter
            .f64_histogram("live_diff_build_duration_seconds")
            .with_description("Duration of diff build operations in seconds")
            .build();
        let subscribe_regime_total = meter
            .u64_counter("live_subscribe_regime_total")
            .with_description("Total number of SubscribeReadLayer catch-up phases by regime")
            .build();
        let subscribe_catch_up_duration_seconds = meter
            .f64_histogram("live_subscribe_catch_up_duration_seconds")
            .with_description(
                "Duration of the SubscribeReadLayer catch-up phase in seconds, from catch-up \
                 entry after auth and admission to the catch-up content being sent",
            )
            .build();
        let join_latency_seconds = meter
            .f64_histogram("live_edit_join_latency_seconds")
            .with_description(
                "Seconds from RPC handler entry to the first delivered state, by endpoint \
                 (SubscribeReadLayer: first caught-up ack enqueued, including auth, admission \
                 wait and catch-up; JoinSession: SessionConnected and membership replay \
                 enqueued, including the wait for the client's header). Successful joins only",
            )
            .build();
        let subscribe_catch_up_delivered_bytes = meter
            .f64_histogram("live_subscribe_catch_up_delivered_bytes")
            .with_description(
                "OMPF bytes delivered during a SubscribeReadLayer catch-up phase, by regime, \
                 service, and catchup_mode (snapshot|legacy)",
            )
            .build();
        let subscribe_stable_cut_total = meter
            .u64_counter("live_subscribe_stable_cut_total")
            .with_description(
                "SubscribeReadLayer catch-up state reads by cut outcome (verified|unverified), \
                 regime, and service. verified: the catch-up served a proven cut (its atomic \
                 read's, or above the atomic-read limits its shared fenced read's, counted once \
                 per catch-up sharing it). unverified: the cut regressed below the catch-up's \
                 first header read, or a rollback-mode in-window replay's per-commit read \
                 regressed below the commit it rebuilds (the only replay read counted here), and \
                 the catch-up ended UNAVAILABLE (alert on any increase)",
            )
            .build();
        let presence_events_total = meter
            .u64_counter("live_presence_events_total")
            .with_description(
                "Live presence events delivered to local JoinSession streams, by event_type \
                 (joined|left|message), once per event per recipient after it was enqueued. \
                 Excludes a recipient's own events, the membership replay, live joined events \
                 suppressed as duplicates of that replay, failed sends and lagged drops (see \
                 live_edit_presence_broadcast_dropped_total)",
            )
            .build();
        let presence_broadcast_dropped = meter
            .u64_counter("live_edit_presence_broadcast_dropped_total")
            .with_description(
                "Total number of presence events a local JoinSession stream lost because it fell \
                 more than the per-session presence capacity \
                 (LIVE_EDIT_PRESENCE_BROADCAST_CAPACITY) behind",
            )
            .build();
        let presence_resyncs = meter
            .u64_counter("live_edit_presence_resyncs_total")
            .with_description(
                "Local JoinSession streams ended UNAVAILABLE because they fell more than the \
                 per-session presence capacity behind and missed a participant joining or \
                 leaving; the client's rejoin restores its participant list",
            )
            .build();
        let presence_keepalive_timeouts = meter
            .u64_counter("live_edit_presence_keepalive_timeouts_total")
            .with_description(
                "Local JoinSession streams ended UNAVAILABLE CLIENT_KEEPALIVE_TIMEOUT because the \
                 client opted in to client keepalives and then sent no data for three keepalive \
                 intervals; the participant left before the stream ended",
            )
            .build();
        let pubsub_relay_lagged = meter
            .u64_counter("live_edit_pubsub_relay_lagged_total")
            .with_description(
                "Pub/sub messages a replica-wide relay lost because it fell behind the pub/sub \
                 client's message broadcast channel, by relay (presence|session_deleted); a lost \
                 presence event is missing for every local JoinSession stream in its session",
            )
            .build();
        let presence_lease_refresh_failures = meter
            .u64_counter("live_edit_presence_lease_refresh_failures_total")
            .with_description(
                "Failed steps of a JoinSession connection's presence lease keeper (a store error \
                 on the lease refresh, a self-rejoin's clock read or register, or the \
                 session-index re-add; or a rejoin register past its deadline), once per failed \
                 keeper step; the keeper retries at its next tick",
            )
            .build();
        let presence_rejoins = meter
            .u64_counter("live_edit_presence_rejoins_total")
            .with_description(
                "Self-rejoins by JoinSession presence lease keepers: an evicted, still-open \
                 connection re-registered by its own keeper with the membership re-added, which \
                 announces joined to peers again. An info-hash repair is not counted",
            )
            .build();
        let presence_sweep_duration_seconds = meter
            .f64_histogram("live_edit_presence_sweep_duration_seconds")
            .with_description(
                "Wall time in seconds of one presence-sweeper leader pass, by outcome \
                 (completed|lease_lost|cancelled|error); recorded only by the replica that won \
                 the leader lock",
            )
            .build();
        let presence_sweep_lease_total = meter
            .u64_counter("live_edit_presence_sweep_lease_total")
            .with_description(
                "Presence-sweeper leader-lease refresh decisions before a chunk, by outcome \
                 (renewed|lost|refresh_error); lost means the pass ended before its next chunk; \
                 refresh_error is additive to lost",
            )
            .build();
        let presence_stale_removed = meter
            .u64_counter("live_edit_presence_stale_removed_total")
            .with_description(
                "Connections the presence sweeper removed, and announced as left, because their \
                 lease had expired",
            )
            .build();
        let presence_stale_removal_delay_seconds = meter
            .f64_histogram("live_edit_presence_stale_removal_delay_seconds")
            .with_description(
                "Seconds from a connection's lease expiry to its removal by the presence \
                 sweeper, on the store's clock, per removal with a recorded expiry",
            )
            .build();
        let presence_sweep_oldest_stale_age_seconds = meter
            .f64_gauge("live_edit_presence_sweep_oldest_stale_age_seconds")
            .with_description(
                "Largest stale-removal delay (seconds) in this replica's last completed \
                 presence-sweeper leader pass; 0 after a pass with no removals and on replicas \
                 that are not the leader. Aggregate with max(), never sum().",
            )
            .build();
        let auth_cache_hits = meter
            .u64_counter("live_edit_auth_cache_hits_total")
            .with_description("Total number of auth cache hits, by kind (permissions|userinfo)")
            .build();
        let auth_cache_misses = meter
            .u64_counter("live_edit_auth_cache_misses_total")
            .with_description("Total number of auth cache misses, by kind (permissions|userinfo)")
            .build();
        let auth_call_duration_seconds = meter
            .f64_histogram("live_edit_auth_call_duration_seconds")
            .with_description("Duration of external auth calls in seconds, by backend and protocol")
            .build();
        let auth_errors = meter
            .u64_counter("live_edit_auth_errors_total")
            .with_description("Total number of auth backend errors, by backend and code")
            .build();
        let userinfo_fallbacks_total = meter
            .u64_counter("live_edit_userinfo_fallbacks_total")
            .with_description(
                "Total userinfo lookups the userinfo service answered with no such user, \
                 resolved instead from the bearer's JWT claims; ticks once per fallback, which \
                 is once per userinfo cache miss, and stays 0 unless the userinfo service and \
                 authorization are both on",
            )
            .build();
        let update_layer_retries_total = meter
            .u64_counter("live_edit_update_layer_retries_total")
            .with_description(
                "Total number of update_layer retry attempts taken, by commit path \
                 (fast|watch|structural|structural_full|reset) and eventual outcome \
                 (committed|exhausted)",
            )
            .build();
        let update_layer_commit_already_landed_total = meter
            .u64_counter("live_edit_update_layer_commit_already_landed_total")
            .with_description(
                "Total update_layer attempts that found their own call's commit already landed \
                 and reported it instead of applying it again, by backend (lua|txn)",
            )
            .build();
        let update_layer_lost_reply_release_lookup_total = meter
            .u64_counter("live_edit_update_layer_lost_reply_release_lookup_total")
            .with_description(
                "Total lookups by an UpdateLayer call that ended without proof of its outcome \
                 for the external-value releases its own landed commit owed, by outcome \
                 (enqueued|absent|failed): enqueued found and enqueued them, absent found none, \
                 failed could not read them and left them to the reference-count reconciler",
            )
            .build();
        let update_layer_operation_overtaken_total = meter
            .u64_counter("live_edit_update_layer_operation_overtaken_total")
            .with_description(
                "Total unchained operation-id UpdateLayer calls ended because their client_id's \
                 commit record on the object changed after the call started (another UpdateLayer \
                 from that client_id committed, or a checkpoint restore), by backend (lua|txn); \
                 a chained call is never counted: its commit refuses it PREDECESSOR_NOT_COMMITTED",
            )
            .build();
        let operation_id_match_total = meter
            .u64_counter("live_edit_operation_id_match_total")
            .with_description(
                "Total UpdateLayer/CreateLayer calls whose operation_id matched an operation an \
                 earlier call already committed, by rpc (update_layer|create_layer) and outcome \
                 (already_committed|conflict)",
            )
            .build();
        let update_layer_attempts_histogram = meter
            .f64_histogram("live_edit_update_layer_attempts_histogram")
            .with_description(
                "Number of attempts per update_layer commit whose landing the call observed \
                 (1 = first-try commit), counted once when the landing is observed, by the \
                 committing attempt's path",
            )
            .build();
        let update_layer_commit_duration_seconds = meter
            .f64_histogram("live_edit_update_layer_commit_duration_seconds")
            .with_description(
                "Duration of update_layer calls in seconds, from entry (admission wait included) \
                 to return, or to the drop of a call abandoned first (outcome=error), by the last \
                 attempt's path and outcome (ok|error)",
            )
            .build();
        let update_layer_relay_bytes = meter
            .f64_histogram("live_edit_update_layer_relay_bytes")
            .with_description(
                "Relay OMPF bytes of each landed UpdateLayer commit (the bytes its stream entry's \
                 ompf field carries, 0 for an empty commit): one sample per commit the service saw \
                 land, the same commits live_edit_update_layer_attempts_histogram counts; a commit \
                 no attempt of its call saw land is not sampled. Size \
                 LIVE_EDIT_LAYER_STREAM_MAX_BYTES and the backing store from it",
            )
            .build();
        let store_write_refused_total = meter
            .u64_counter("live_edit_store_write_refused_total")
            .with_description(
                "Total writes the backing store refused before anything was applied, by stage \
                 (ext_store|update_commit|create_commit) and reason \
                 (out_of_memory|misconf|readonly|noreplicas). Mostly uploads, answered UNAVAILABLE \
                 and not retried in-process; ext_store also counts the DR restore's pre-store and \
                 read-path rewrites of recovered external values, and create_commit the object-id \
                 counter repair writes. out_of_memory starts the external-value GC's \
                 out-of-memory collection, which frees only released external values; a store \
                 full of data in use stays refused under the required noeviction policy: raise \
                 maxmemory or free memory",
            )
            .build();
        let update_layer_conflicts_total = meter
            .u64_counter("live_edit_update_layer_conflicts_total")
            .with_description(
                "Total update_layer attempts abandoned because the layer changed under them \
                 (usually another edit's commit; on the watch path also a checkpoint cursor \
                 stamp on the header), by path and kind (watch_conflict|watch_read_moved|\
                 seq_fence|cache_diverged|snapshot_fence_wait)",
            )
            .build();
        let update_layer_recharges_total = meter
            .u64_counter("live_edit_update_layer_recharges_total")
            .with_description(
                "Total structural UpdateLayer attempts that released what they held (cache \
                 entry, WATCH and transaction checkout, per-object permits) and waited in the \
                 upload gate for a larger upload charge, then resumed without spending a retry, \
                 by path",
            )
            .build();
        let upload_object_state_bytes = meter
            .f64_histogram("live_edit_upload_object_state_bytes")
            .with_description(
                "Object-state bytes an UpdateLayer call held in its upload charge, on top of its \
                 processing charge, when it returned its own commit: the memory it held for the \
                 object it edits, by the committing attempt's path. Observed on the structural, \
                 structural_full, reset and watch paths; never on fast, which holds none",
            )
            .build();
        let update_layer_structural_fallbacks_total = meter
            .u64_counter("live_edit_update_layer_structural_fallbacks_total")
            .with_description(
                "Total structural UpdateLayer calls that committed through the WATCH path \
                 instead of the structural cache for a reason other than an uncacheable \
                 object, by reason (repeat_divergence: CACHE_DIVERGED again after the cache \
                 entry was reloaded, expected to stay 0; over_work_limit: the plan was over \
                 the structural commit's work limit; reset_over_work_limit: a reset of an \
                 uncached layer estimated over that limit before its cold start)",
            )
            .build();
        let update_layer_structural_commit_seconds = meter
            .f64_histogram("live_edit_update_layer_structural_commit_seconds")
            .with_description(
                "Round trip of each structural commit's EVALSHA as the client sees it, in \
                 seconds, whatever it answered: the bound on the shard hold of one structural \
                 UpdateLayer commit",
            )
            .build();
        let update_layer_fence_wait_seconds = meter
            .f64_histogram("live_edit_update_layer_fence_wait_seconds")
            .with_description(
                "Seconds one update_layer call waited on snapshot fences, from the start of the \
                 first attempt a fence refused or failed to the end of its last fence sleep, by \
                 backend (lua|txn); observed once per call that met a fence, including a call \
                 the RPC deadline, a disconnect or shutdown cancelled (its wait up to the \
                 cancellation)",
            )
            .build();
        let update_layer_fence_intent_total = meter
            .u64_counter("live_edit_update_layer_fence_intent_total")
            .with_description(
                "Writer-intent registrations by WATCH-path update_layer attempts that read a \
                 snapshot, on layers above the atomic-read ceiling, by result \
                 (registered|refused_fenced)",
            )
            .build();
        let update_layer_fence_timeout_total = meter
            .u64_counter("live_edit_update_layer_fence_timeout_total")
            .with_description(
                "update_layer calls failed UNAVAILABLE because their snapshot-fence waiting \
                 reached its cap (LIVE_EDIT_RPC_DEADLINE_MS or the RPC's remaining deadline)",
            )
            .build();
        let stream_id_collision_total = meter
            .u64_counter("live_edit_stream_id_collision_total")
            .with_description(
                "Total commits refused because the per-object stream already held an entry at or \
                 above the stream ID the commit intended to write, by backend (lua|txn) and site \
                 (update|create); the backend=lua, site=update cell is further split by the \
                 update script's mode (fast|structural). The commit was rejected whole; on \
                 site=update the layer is \
                 unwritable until the stream is cleared, and on site=create the layer was not \
                 created because its freshly-minted object_id already owned a stream",
            )
            .build();
        let keyspace_type_error_total = meter
            .u64_counter("live_edit_keyspace_type_error_total")
            .with_description(
                "Total commits refused because a key the commit was about to mutate held the \
                 wrong Valkey type, by backend (lua|txn) and site (update|create); the \
                 backend=lua, site=update cell is further split by the update script's mode \
                 (fast|structural). The commit was \
                 rejected whole; on site=update the layer stays unwritable and on site=create the \
                 layer was not materialized, until the offending key (named in the accompanying \
                 ERROR log line) is removed",
            )
            .build();
        let snapshot_read_type_error_total = meter
            .u64_counter("live_edit_snapshot_read_type_error_total")
            .with_description(
                "Total object-state touches refused because the key they addressed held the wrong \
                 Valkey type, by key_family (which part of the layer's state was unreadable). \
                 Object-snapshot reads, plus the one obj:hdr fence write the orphan sweep issues \
                 on the same key before retiring an object. The touch fails closed, so one such \
                 key takes down, for that object, every consumer whose read touches it - \
                 structural UpdateLayer, subscribe per-commit reconstruction, catch-up and the \
                 checkpoint assembler for the header, an index or an entry row; on an object \
                 marked chs_complete=1, obj:chs stops all four snapshot \
                 consumers at every layer size and a children list obj:chs does not name stops \
                 none of them - until the offending key (named in the accompanying ERROR \
                 log line) is removed. A deleted or replaced layer's clean-up skips its \
                 external-value release for the header or a field or time-sample row (the layer \
                 is still deleted). A foreign value on any of the five indices is counted at \
                 the clean-up's size read, under that index's key_family (obj:flds and obj:tss \
                 once more at its release), and the clean-up goes on (an orphan whose \
                 retirement fence failed, or whose id is not numeric, gets no size read, so \
                 there it is not counted); it stops the key \
                 deletion after the header, and on obj:flds or obj:tss the release too: the five \
                 indices and every row not yet removed stay, and on a DeleteLayer so does the \
                 layer's session entry. The DeleteLayer answers INTERNAL with the layer already \
                 unmapped. The key deletion's own failure log names the object and the index. A \
                 foreign obj:ext \
                 (key_family ext_index) stops none of \
                 the four; the clean-up skips its release and removes the key. Structural \
                 UpdateLayer reads the layer on every call on \
                 its WATCH path (backends txn and lua-watch-structural, and under lua its \
                 uncacheable objects and fallbacks), but under lua's structural cache only on a \
                 cold start or reload; a cached call's own reads (header, stream and commit \
                 token, then its commit's probes and digest reads of the keys it rewrites) are \
                 not counted here. A cached commit still TYPE-probes every record and index key \
                 it writes, obj:chs included whenever its plan writes a children list, and a \
                 foreign value on such a key refuses the commit with WRONGTYPE_KEY, counted on \
                 live_edit_keyspace_type_error_total (backend=lua, site=update, \
                 mode=structural), not here. The cache's sampled audit reads the layer too; a \
                 refusal there ticks this counter and stops no call. When op in \
                 that log line is HGETALL obj:hdr (orphan discovery) the radius is wider than \
                 per-layer: that read aborts the rest of the orphan-sweep pass, so orphan cleanup \
                 stalls for every object the pass had not yet reached, and orphaned-mapping \
                 cleanup and the deferred-release drain stall entirely, on every 300s pass until \
                 that key is removed",
            )
            .build();
        let snapshot_read_window_fallback_total = meter
            .u64_counter("live_edit_snapshot_read_window_fallback_total")
            .with_description(
                "Total structural WATCH-path snapshot reads whose per-window read script met a \
                 cluster-state refusal (slot migrating, not served, or cluster down) and that \
                 finished on plain commands; expected zero outside a resharding or cluster \
                 failover.",
            )
            .build();
        let snapshot_read_refused_total = meter
            .u64_counter("live_edit_snapshot_read_refused_total")
            .with_description(
                "Total object-state touches refused by a permanent, non-WRONGTYPE backend \
                 failure, by fred error kind. Object-snapshot reads, plus the one obj:hdr fence \
                 write the orphan sweep issues on the same key before retiring an object. The \
                 touch fails closed, so every consumer whose read issues the command stops on \
                 that object until it stops failing - structural UpdateLayer, subscribe \
                 per-commit reconstruction, catch-up and the checkpoint assembler for the \
                 header, an index or an entry row; on an object marked \
                 chs_complete=1, obj:chs stops all four snapshot consumers at every layer size \
                 and a children list obj:chs does not name stops none of them. A deleted or \
                 replaced layer's clean-up stops whole, before it changes anything, for any \
                 command of its size read (the header, obj:ext, or the size of any of the five \
                 indices), and skips only its external-value release for a field or time-sample \
                 row (the layer is still deleted). A persistent failure on any of the five \
                 indices that the size read's count did not meet (its count answered) stops \
                 the key deletion after the header, and on obj:flds or obj:tss the release \
                 too: the five indices and every \
                 row not yet removed stay, and on a DeleteLayer so does the layer's session \
                 entry. The DeleteLayer fails with the layer already unmapped. The key \
                 deletion's own failure log names the object and the index. Structural \
                 UpdateLayer reads the \
                 layer on every call on its \
                 WATCH path (backends txn and lua-watch-structural, and under lua its \
                 uncacheable objects and fallbacks), but under lua's structural cache only on a \
                 cold start or reload; a cached call's own reads (header, stream and commit \
                 token, then its commit's probes and digest reads of the keys it rewrites) are \
                 not counted here, though a permanent failure on them still fails the call. The \
                 cache's sampled audit reads the layer too; a refusal there ticks this counter \
                 and stops no call. When op in the \
                 accompanying log line is HGETALL obj:hdr (orphan discovery) the radius is wider \
                 than per-layer: \
                 that read aborts the rest of the orphan-sweep pass, so orphan cleanup stalls for \
                 every object the pass had not yet reached, and orphaned-mapping cleanup and the \
                 deferred-release drain stall entirely, on every 300s pass until that command \
                 stops failing. WRONGTYPE is claimed by \
                 live_edit_snapshot_read_type_error_total instead; transient failures are retried \
                 and counted on neither, including the backing store's -LOADING and -BUSY while \
                 it restarts or runs a long script. Eleven kinds should never occur and are \
                 alertable at any non-zero increase; backend_unavailable (-MISCONF / -READONLY / \
                 -NOREPLICAS refusing the orphan sweep's header fence write, which can clear \
                 without an operator) takes a sustained-rate rule instead, and a -LOADING or \
                 -BUSY in it means their transient classification has regressed; \
                 backend_out_of_memory is one of those eleven and \
                 also comes from the backing store, but under the required noeviction policy it \
                 clears without an operator only when released external values filled the store \
                 (the GC's out-of-memory collection frees those). Each tick carries a log line \
                 naming the object, the \
                 key and the failing command — ERROR for the eleven, WARN for backend_unavailable",
            )
            .build();
        let snapshot_read_total = meter
            .u64_counter("live_edit_snapshot_read_total")
            .with_description(
                "Total object-snapshot reads, by consumer (catchup|catchup_legacy_per_commit|\
                 checkpoint|structural_update|ext_gc|structural_cache|structural_cache_audit), \
                 path (atomic|fenced|unfenced) and outcome \
                 (ok|not_found|too_large|transient_error|permanent_error|fence_broken). \
                 too_large counts an atomic read that answered TOO_LARGE (the layer is above the \
                 atomic-read item or byte limit), so the layer took its consumer's above-ceiling \
                 path; that path's own read is counted under its own path",
            )
            .build();
        let snapshot_read_duration_seconds = meter
            .f64_histogram("live_edit_snapshot_read_duration_seconds")
            .with_description(
                "Wall time of one object-snapshot read in seconds, from issue to decoded result, \
                 by consumer and path, recorded for every outcome",
            )
            .build();
        let snapshot_read_reply_bytes = meter
            .f64_histogram("live_edit_snapshot_read_reply_bytes")
            .with_description(
                "Bytes of the strings one successful object-snapshot read read for the object's \
                 indices, entries and children, including a marked object's obj:chs members, \
                 which are read but not returned (the tally the atomic read's byte limit is \
                 checked against), by consumer. A slim atomic read, which returns each value \
                 row's shape and external hash instead of its inline value, counts the values \
                 it read but did not transfer, so it records what a full read of the same \
                 object would. For consumer ext_gc, external-value clean-up's reference walk: \
                 the strings it received (index members, value rows' columns but their inline \
                 values, obj:ext entries)",
            )
            .build();
        let snapshot_read_entries = meter
            .f64_histogram("live_edit_snapshot_read_entries")
            .with_description(
                "Entries one successful object-snapshot read read (section, tombstone, field and \
                 time-sample hashes, plus children lists), by consumer; for consumer ext_gc, the \
                 field and time-sample rows external-value clean-up's reference walk read",
            )
            .build();
        let structural_cold_start_reads_total = meter
            .u64_counter("live_edit_structural_cold_start_reads_total")
            .with_description(
                "Total cold starts of the structural write-planning cache, by the route that \
                 ended each (atomic|fenced|uncacheable): atomic produced a cache entry from the \
                 plain atomic read, fenced from the snapshot-fenced read of a layer above the \
                 plain read's limits; uncacheable found the object over the per-entry weight \
                 cap or without chs_complete. A cold start that failed before its verdict or \
                 admission (a fenced read that failed for its edit included), found the object \
                 gone or superseded, had its loaded entry refused because the object was \
                 retired or dropped by an audit while it read, or ran on a disabled cache is \
                 not counted",
            )
            .build();
        let structural_cold_start_read_duration_seconds = meter
            .f64_histogram("live_edit_structural_cold_start_read_duration_seconds")
            .with_description(
                "Wall time of one structural-cache cold start in seconds, by route, from its \
                 first read to its uncacheable verdict or its entry's admission, including any \
                 plain read before a fenced one and every wait for its permits, memory and the \
                 snapshot fence; recorded only for the cold starts \
                 live_edit_structural_cold_start_reads_total counts",
            )
            .build();
        let structural_cache_lookups_total = meter
            .u64_counter("live_edit_structural_cache_lookups_total")
            .with_description(
                "Structural UpdateLayer attempts that consulted the structural write-planning \
                 cache, one per attempt (a call that retries counts again), by how each got its \
                 entry (hit|repaired|cold|reload): hit found it current, repaired replayed stream \
                 entries to bring it current, cold found no entry and read the object, reload \
                 dropped an entry on a repair anomaly and read the object again",
            )
            .build();
        let structural_cache_reloads_total = meter
            .u64_counter("live_edit_structural_cache_reloads_total")
            .with_description(
                "Structural write-planning cache entries dropped and read again, by reason \
                 (identity|regressed|gap|tombstone|parse|replay_err|too_far|too_large|\
                 dangling_adopted|diverged|uncacheable); a cold miss is not a reload",
            )
            .build();
        let structural_cache_uncacheable_total = meter
            .u64_counter("live_edit_structural_cache_uncacheable_total")
            .with_description(
                "Uncacheable verdicts of the structural write-planning cache, one per verdict \
                 (not per edit it routes), by reason \
                 (over_weight|no_chs_complete|reload_limit|fenced_read_failed|over_deadline): \
                 over_weight is a layer over the per-entry weight cap, no_chs_complete one \
                 without a complete children-list index, reload_limit one edit whose cold \
                 starts kept failing their repair, fenced_read_failed one edit whose fenced \
                 cold-start read ran out of waiting time or was broken three times, \
                 over_deadline a layer too large for its fenced cold-start read to fit the \
                 configured RPC deadline; the layer's structural edits take the WATCH path \
                 while the verdict lasts (reload_limit and fenced_read_failed: that edit only)",
            )
            .build();
        let structural_cache_replayed_entries = meter
            .f64_histogram("live_edit_structural_cache_replayed_entries")
            .with_description(
                "Stream entries one repair of a structural write-planning cache entry replayed",
            )
            .build();
        let structural_cache_repair_bytes = meter
            .f64_histogram("live_edit_structural_cache_repair_bytes")
            .with_description(
                "Stream bytes one repair of a structural write-planning cache entry read \
                 (entry ids plus every field name and value)",
            )
            .build();
        let structural_cache_divergence_total = meter
            .u64_counter("live_edit_structural_cache_divergence_total")
            .with_description(
                "Structural commits refused because the structural write-planning cache \
                 diverged from the store (CACHE_DIVERGED); expected to stay 0",
            )
            .build();
        let structural_cache_audit_total = meter
            .u64_counter("live_edit_structural_cache_audit_total")
            .with_description(
                "Sampled audits of the structural write-planning cache that ran to an end, by \
                 outcome (match|diverged|skipped): match found a repaired copy of the entry equal \
                 to the store at the same cut, diverged found it different and dropped the entry \
                 (expected to stay 0), skipped could not compare",
            )
            .build();
        let structural_cache_audit_fenced_total = meter
            .u64_counter("live_edit_structural_cache_audit_fenced_total")
            .with_description(
                "Sampled audits of structural write-planning cache entries whose layer is above \
                 the atomic-read limits, by result (read|paced|busy|failed): read took the \
                 snapshot-fenced read, which paused that layer's colliding writers for its hold \
                 (live_edit_snapshot_fence_hold_seconds{consumer=\"structural_cache_audit\"}); \
                 paced skipped it because another ran on this replica within the pacing \
                 interval; busy skipped it because the replica's cold-start permits, fenced \
                 read permit or catch-up memory were not free at once; failed gave up waiting \
                 for the fence, was broken on every attempt, or met a store error",
            )
            .build();
        let snapshot_fence_hold_seconds = meter
            .f64_histogram("live_edit_snapshot_fence_hold_seconds")
            .with_description(
                "Seconds one fenced snapshot read attempt held its snapshot-fence lease, from the \
                 grant to the release's reply, by consumer; writers of the layer wait while a \
                 lease is held",
            )
            .build();
        let snapshot_fence_acquire_total = meter
            .u64_counter("live_edit_snapshot_fence_acquire_total")
            .with_description(
                "Snapshot-fence acquire replies to fenced readers, by result \
                 (started_epoch|joined_epoch|retry_after_epoch|retry_after_gap|\
                 retry_after_intent); one per reply. A shared fenced-read flight's read is one \
                 reader however many catch-ups share it",
            )
            .build();
        let snapshot_fence_reader_wait_seconds = meter
            .f64_histogram("live_edit_snapshot_fence_reader_wait_seconds")
            .with_description(
                "Seconds a fenced snapshot read waited for its snapshot-fence lease, by consumer: \
                 one observation per read attempt, from its first acquire call to the grant or \
                 to where the wait ended without one. A shared fenced-read flight's read is one \
                 reader however many catch-ups share it; a catch-up that leaves its flight at \
                 reader_wait_max(L) is counted on live_edit_subscriber_stream_exits_total \
                 {reason=fenced_read_wait_cap}, not here.",
            )
            .build();
        let valkey_reconnects_total = meter
            .u64_counter("live_edit_valkey_reconnects_total")
            .with_description(
                "Total number of fred reconnect events, by pool \
                 (subscriber|data|txn|subscriber_pool|maintenance|snapshot)",
            )
            .build();
        let valkey_router_restarts_total = meter
            .u64_counter("live_edit_valkey_router_restarts_total")
            .with_description(
                "Total Valkey client router tasks that ended other than by quit and were \
                 restarted, by pool (subscriber|data|txn|subscriber_pool|maintenance|snapshot). \
                 Baseline 0; each restart also logs one ERROR line naming the pool member",
            )
            .build();
        let valkey_unresponsive_total = meter
            .u64_counter("live_edit_valkey_unresponsive_total")
            .with_description(
                "Total Valkey connections closed as unresponsive, by pool \
                 (subscriber|data|txn|subscriber_pool|maintenance|snapshot) and verdict (fast: no \
                 reply progress for 2 s while the same slot range answered other connections; \
                 slow: none for the 45 s slow bound, or a bounded router wait expired). Each also \
                 logs one fred WARN naming the client, node and verdict",
            )
            .build();
        let valkey_member_replacements_total = meter
            .u64_counter("live_edit_valkey_member_replacements_total")
            .with_description(
                "Pool member replacements by a fresh Valkey client, after a member's second fast \
                 unresponsive verdict within an hour, by pool (data|txn|maintenance|snapshot) and \
                 outcome (replaced|connect_failed|budget_exhausted)",
            )
            .build();
        let valkey_errors_total = meter
            .u64_counter("live_edit_valkey_errors_total")
            .with_description(
                "Total number of Valkey errors, by pool and kind: fred connection-level errors \
                 under fred's ErrorKind name, plus kind=command_timeout for each command the \
                 service issued whose client-side timeout expired",
            )
            .build();
        let storage_watch_unwatch_failures_total = meter
            .u64_counter("live_edit_storage_watch_unwatch_failures_total")
            .with_description(
                "Total WatchGuard teardowns that gave up on their slot-pinned UNWATCH, by site \
                 (explicit|drop). Non-zero means a txn-pool connection was released carrying \
                 stale WATCH state, which punishes the next borrower with a spurious \
                 EXEC=nil (an in-process transaction abort and retry) or, on a proxied \
                 cluster, a CROSSSLOT naming a key that request never touched.",
            )
            .build();
        let storage_watch_voided_total = meter
            .u64_counter("live_edit_storage_watch_voided_total")
            .with_description(
                "Total watched transactions whose WATCH was found or proven void, or whose \
                 void-detection layers could not be armed, by path \
                 (update_layer|create_session) and disposition \
                 (aborted_pre_exec|committed_unwatched|compensated|witness_unavailable|\
                 fingerprint_unavailable). A WATCH dies with its connection, so a reconnect \
                 between WATCH and EXEC turns an optimistic commit into an unconditional one. \
                 Alert on committed_unwatched: it is the residual where the server confirms the \
                 commit ran on a connection that never held the WATCH, so a duplicate session \
                 name (create_session) or a commit torn between materialized state and the live \
                 stream (update_layer) may have landed. The two *_unavailable dispositions are \
                 not incidents: they report that the backing store does not answer CLIENT ID in \
                 the way one detection layer needs, so that layer is disarmed and the commit ran \
                 on the remaining ones. They are near-zero or near-total by construction, and \
                 are what a DR / dev-AMR run reads to characterise a proxied managed store.",
            )
            .build();
        let valkey_txn_dirty_checkouts_total = meter
            .u64_counter("live_edit_valkey_txn_dirty_checkouts_total")
            .with_description(
                "Total txn-pool checkouts whose DISCARD/UNWATCH hygiene probe found the slot \
                 unfit for a transaction, by reason \
                 (leaked_multi|unexpected_reply|probe_timeout|probe_error|topology_reply). A tick \
                 records the VERDICT, not a repair: the repair for \
                 probe_timeout/probe_error/topology_reply is streak-gated \
                 (first of a streak, then every Nth) so a slow store is not answered with a \
                 pool-wide reconnect storm, and the repair for leaked_multi/unexpected_reply \
                 stops once several rebuilds in a row against the same node have not changed the \
                 verdict (see live_edit_valkey_txn_repair_ineffective_total). A sustained rate on \
                 leaked_multi/unexpected_reply means transactions are being abandoned \
                 mid-flight; on probe_timeout/probe_error it points at the store or the network. \
                 topology_reply is the one cell EXPECTED to be non-zero: it ticks while the \
                 cluster reshards, a replica loads, or a primary is demoted, because fred \
                 intercepts only three of the seven prefixes this classifier tests, each \
                 case-sensitively: MOVED/ASK only when canonical (exactly 3 space-separated \
                 tokens) and only on a clustered client, and CLUSTERDOWN on every read frame with \
                 no token-count condition, so casing is its only escape. The other four are never \
                 intercepted, so every node-state error frame outside those conditions passes \
                 through. Correlate a burst with the topology change; \
                 only a rate sustained past one is a signal.",
            )
            .build();
        let valkey_txn_repair_ineffective_total = meter
            .u64_counter("live_edit_valkey_txn_repair_ineffective_total")
            .with_description(
                "Total txn-pool state-proof dirty verdicts (leaked_multi|unexpected_reply) whose \
                 slot was deliberately NOT recycled because several consecutive rebuilds had \
                 already failed to change the verdict, by reason. Non-zero means the probe's \
                 classifier and the server disagree about a reply shape rather than a connection \
                 carrying leaked transaction state: recycling cannot fix that and would recycle \
                 without bound, so the repair is withheld and the slot is handed \
                 over as-is. The streak and this breaker are PER CLUSTER NODE, so a tick means \
                 hygiene is off for one node rather than for the whole pool; the node is named in \
                 the accompanying ERROR log's `node` field, not as a label here (cardinality). \
                 A clean checkout against that node re-arms the repair. Verdicts whose node \
                 cannot be named (a centralized or proxied endpoint, or a cluster whose routing \
                 view is momentarily unavailable) share one fallback bucket that ANY clean \
                 checkout re-arms, so a single-endpoint deployment reads exactly as the earlier \
                 pool-wide contract did.",
            )
            .build();
        let valkey_consumer_forced_reconnects_total = meter
            .u64_counter("live_edit_valkey_consumer_forced_reconnects_total")
            .with_description(
                "Total number of force_reconnection() calls issued by a stream-multiplexer reader \
                 against its own connection, by phase (xread|wake_ack|probe|keepalive|permanent). \
                 xread, wake_ack, probe and keepalive are wedge verdicts: an XREAD timeout, a woken \
                 XREAD that did not return in time, a PING before the first XREAD after the reader \
                 had nothing to read (or, on a cluster, bound to a different hash slot) that did \
                 not answer in time, and an idle keepalive PING that \
                 failed. permanent is a heal after a failure that ended every subscription on the \
                 reader. On a cluster, a verdict that finds the reader's node no longer owns its \
                 hash slot retires the connection instead and the reader retargets \
                 (live_edit_stream_mux_reader_connects_total cause=retarget); an idle reader \
                 whose node is no longer a primary just retires it.",
            )
            .build();
        let valkey_consumer_unblocked_retries_total = meter
            .u64_counter("live_edit_valkey_consumer_unblocked_retries_total")
            .with_description(
                "Total number of live layer-stream XREAD BLOCKs a stream reader saw end in \
                 -UNBLOCKED (a CLIENT UNBLOCK) and re-issued from the durable cursor without \
                 charging the transient-failure budget. Expected about zero in every auth mode: \
                 no command of the service interrupts a live reader read (readers re-AUTH between \
                 their reads, live_edit_stream_mux_reauths_total), so it ticks only when \
                 something outside the service unblocks one. An unblock of a read the reader \
                 already abandoned after a client-side timeout is never counted.",
            )
            .build();
        let valkey_consumer_permanent_exits_total = meter
            .u64_counter("live_edit_valkey_consumer_permanent_exits_total")
            .with_description(
                "Total number of stream-multiplexer reader give-ups caused by a permanent fred \
                 error, by phase (xread|probe): the XREAD, or the PING a reader sends before its \
                 first XREAD after it had nothing to read (or, on a cluster, bound to a different \
                 hash slot). Each one ends the affected \
                 subscriptions with Code::Unavailable.",
            )
            .build();
        let stream_mux_subscriptions = meter
            .i64_up_down_counter("live_edit_stream_mux_subscriptions")
            .with_description(
                "SubscribeReadLayer subscriptions holding a stream reader on the XREAD \
                 multiplexer: registered streams, plus reader reservations whose stream is not \
                 registered yet (a catch-up waiting for capacity, or for its proven read — its \
                 own, or its layer's shared fenced read)",
            )
            .build();
        let stream_mux_streams = meter
            .i64_up_down_counter("live_edit_stream_mux_streams")
            .with_description(
                "Distinct layer stream keys currently read by the XREAD multiplexer's readers",
            )
            .build();
        let stream_mux_xreads_total = meter
            .u64_counter("live_edit_stream_mux_xreads_total")
            .with_description(
                "Total multiplexed XREAD BLOCK commands issued by the stream multiplexer",
            )
            .build();
        let stream_mux_wakes_total = meter
            .u64_counter("live_edit_stream_mux_wakes_total")
            .with_description(
                "Total reader wake-ups (an XADD on a reader's wake key so its blocked XREAD \
                 returns), by outcome (ok|error). Counts four kinds: a coverage wake, for a \
                 joining subscription the in-flight XREAD does not cover; a liveness wake, for a \
                 joining subscription when the reader's connection has not answered in the last \
                 3 s or has failed since, so the join gets a connection check; a credential \
                 wake, so the reader can send a due credential re-AUTH between reads; and the \
                 shutdown wake. An error on a coverage wake delays that subscription's first \
                 live update by up to one XREAD BLOCK period; on a liveness wake it delays \
                 nothing, but that join gets no connection check from it; on a credential wake \
                 the re-AUTH waits for the in-flight XREAD to return (up to one XREAD BLOCK \
                 period); on the shutdown wake the reader's QUIT may wait for the in-flight XREAD \
                 to return.",
            )
            .build();
        let stream_mux_keepalives_total = meter
            .u64_counter("live_edit_stream_mux_keepalives_total")
            .with_description(
                "Total idle keepalive ticks run by connected stream-multiplexer readers with \
                 nothing to read (no subscription, no XREAD in flight), by result (ok|failed). A \
                 tick is one PING on the reader's connection; failed means it failed or did not \
                 answer in time, and the reader force-reconnects \
                 (live_edit_valkey_consumer_forced_reconnects_total phase=keepalive), or on a \
                 cluster retires the connection when its node is no longer a primary.",
            )
            .build();
        let stream_mux_reauths_total = meter
            .u64_counter("live_edit_stream_mux_reauths_total")
            .with_description(
                "Total credential re-AUTHs stream-multiplexer readers sent on their own \
                 connections between reads, by result (ok|failed). Under the passwordless auth \
                 modes fred schedules no periodic re-AUTH on a reader; each connected reader \
                 takes the credential provider's tick itself and, when a credential is due, \
                 wakes its in-flight XREAD (unless a wake is already outstanding for that read) \
                 and sends AUTH before its next read. ok: the AUTH \
                 was answered. failed: no credential could be fetched, or the AUTH failed or did \
                 not answer in time; the connection is kept and nothing is reconnected. Expected \
                 rate: under accessTokenFile one to three per connected reader per token rotation \
                 (every tick inside the 150 s fan-out window sends; a reader ticks every 60-90 s), \
                 none between rotations; under iam one per connected reader per 300-360 s; zero \
                 under static auth.",
            )
            .build();
        let stream_mux_reader_connects_total = meter
            .u64_counter("live_edit_stream_mux_reader_connects_total")
            .with_description(
                "Total stream-multiplexer reader connections opened, by cause (bind|retarget) and \
                 result (ok|failed). Readers connect when they first serve a subscription, not at \
                 startup; on a cluster a reader connects only to the primary that owns its hash \
                 slot. bind: a reader without a connection connected. retarget: a cluster reader \
                 replaced its connection with one to another primary (a slot owned elsewhere, a \
                 MOVED, or a failover). failed: the connect failed or timed out; the reader's \
                 subscriptions retry it with backoff, or end with a retryable UNAVAILABLE on an \
                 auth, TLS or configuration error.",
            )
            .build();
        let stream_mux_readers_connected = meter
            .i64_up_down_counter("live_edit_stream_mux_readers_connected")
            .with_description(
                "Stream-multiplexer readers holding a backing-store connection now. At most \
                 LIVE_EDIT_SUBSCRIBER_POOL_SIZE; zero after startup until the first subscription.",
            )
            .build();
        let stream_mux_bind_refusals_total = meter
            .u64_counter("live_edit_stream_mux_bind_refusals_total")
            .with_description(
                "Total subscriptions refused because every multiplexer reader was bound to another \
                 hash slot (cluster mode). Raise LIVE_EDIT_SUBSCRIBER_POOL_SIZE if non-zero.",
            )
            .build();
        let stream_mux_poison_keys_total = meter
            .u64_counter("live_edit_stream_mux_poison_keys_total")
            .with_description(
                "Total stream keys whose subscriptions were ended because the key alone failed a \
                 multiplexed XREAD (for example WRONGTYPE)",
            )
            .build();
        let stream_mux_overflow_kills_total = meter
            .u64_counter("live_edit_stream_mux_overflow_kills_total")
            .with_description(
                "Total subscriptions ended because they fell behind, by bound (entries|bytes): \
                 the next entry would have taken their undelivered backlog past the hard cap \
                 (16384 entries or 32 MiB), or, once the subscription had started draining, the \
                 backlog stayed at or above 4096 entries or 8 MiB for longer than the delivery \
                 watchdog. bytes counts only endings where a byte limit was reached and the entry \
                 limit was not.",
            )
            .build();
        let stream_mux_head_checks_total = meter
            .u64_counter("live_edit_stream_mux_head_checks_total")
            .with_description(
                "Total head checks (XRANGE key - + COUNT 1) stream multiplexer readers ran because \
                 an XREAD reply's first entry for a layer stream was past the entry after the id \
                 the stream was read from, by result (gap|no_gap|error). gap: at the check the \
                 stream's head was past the read position, so the reply's first entry sits above \
                 ids the stream no longer holds. Each subscription more than one id below that \
                 entry is queued a trim marker and ends stream_gap unless its catch-up covered \
                 those ids. no_gap: the skipped ids were never \
                 written (an id jump, such as a restore band) or the stream was empty. error: the \
                 check failed or timed out; the stream is re-read by the next XREAD.",
            )
            .build();
        let stream_mux_xread_reply_bytes = meter
            .f64_histogram("live_edit_stream_mux_xread_reply_bytes")
            .with_description(
                "Bytes carried by each multiplexed XREAD reply that returned layer-stream entries \
                 (per entry: its id plus its payload and ompf field values; the reader's wake key \
                 excluded, and a reply carrying only the wake entry records nothing). \
                 Each read's COUNT is sized so a reply aims at 1 MiB; the bound is soft, so an \
                 entry larger than its stream's estimate over-commits one reply.",
            )
            .build();
        let valkey_pool_in_use = meter
            .i64_up_down_counter("live_edit_valkey_pool_in_use")
            .with_description(
                "Valkey pool use, by pool (txn|subscriber_pool|maintenance|snapshot). txn counts \
                 exclusive slot acquisitions; subscriber_pool counts stream readers with at least \
                 one stream; the multiplexed maintenance and snapshot pools count logical uses.",
            )
            .build();
        let valkey_pool_waiters = meter
            .i64_up_down_counter("live_edit_valkey_pool_waiters")
            .with_description(
                "Callers currently parked waiting, by pool (txn|subscriber_pool). On txn this is \
                 the FIFO permit queue; on subscriber_pool it is the bounded SubscribeReadLayer \
                 admission queue.",
            )
            .build();
        let valkey_pool_checkout_seconds = meter
            .f64_histogram("live_edit_valkey_pool_checkout_seconds")
            .with_description(
                "txn exclusive-pool checkout latency in seconds, by pool: permit wait plus the \
                 non-blocking next-free scan.",
            )
            .build();
        let subscriber_admission_total = meter
            .u64_counter("live_edit_subscriber_admission_total")
            .with_description(
                "Total SubscribeReadLayer admission decisions, by result \
                 (admitted|refused_saturated|refused_waiter_cap|cancelled). Both capacity \
                 refusals surface to the client as Code::Unavailable, so this is the only way to \
                 tell them apart. cancelled means the request went away, not the \
                 capacity: the session was deleted (Code::NotFound), the service is shutting \
                 down (Code::Unavailable), or the session went missing from the live store \
                 without a deletion (Code::Unavailable).",
            )
            .build();
        let subscriber_admission_wait_seconds = meter
            .f64_histogram("live_edit_subscriber_admission_wait_seconds")
            .with_description(
                "Seconds a SubscribeReadLayer waited for its subscription admission \
                 (live_edit_subscriber_max_subscriptions), recorded on every outcome including the \
                 uncontended fast path.",
            )
            .build();
        let upload_admission_total = meter
            .u64_counter("live_edit_upload_admission_total")
            .with_description(
                "Total CreateLayer/UpdateLayer upload admission and receive-growth decisions, by \
                 rpc (create_layer|update_layer) and result \
                 (admitted|refused_saturated|refused_waiter_cap|refused_growth|cancelled). The \
                 capacity refusals answer Code::Unavailable with nothing applied; no upload under \
                 LIVE_EDIT_MAX_LAYER_SIZE is refused for its byte size. cancelled is an \
                 admission or processing wait ended by shutdown or by a dropped call.",
            )
            .build();
        let upload_admission_wait_seconds = meter
            .f64_histogram("live_edit_upload_admission_wait_seconds")
            .with_description(
                "Seconds a CreateLayer/UpdateLayer waited for upload admission \
                 (live_edit_upload_budget_bytes), by rpc, recorded on every admission \
                 decision including the uncontended fast path.",
            )
            .build();
        let upload_reserved_bytes = meter
            .i64_up_down_counter("live_edit_upload_reserved_bytes")
            .with_description(
                "Bytes reserved within the upload memory budget by admitted \
                 CreateLayer/UpdateLayer uploads, the overflow slot's holder excluded. Never \
                 above live_edit_upload_budget_bytes.",
            )
            .build();
        let upload_overflow_bytes = meter
            .u64_gauge("live_edit_upload_overflow_bytes")
            .with_description(
                "Processing memory charged to the one CreateLayer/UpdateLayer upload running in \
                 the overflow slot, beyond live_edit_upload_budget_bytes; 0 when the slot is \
                 free.",
            )
            .build();
        let upload_overflow_waiters = meter
            .i64_up_down_counter("live_edit_upload_overflow_waiters")
            .with_description(
                "CreateLayer/UpdateLayer uploads whose processing step is waiting for room in the \
                 upload budget or for the overflow slot.",
            )
            .build();
        let upload_overflow_runs_total = meter
            .u64_counter("live_edit_upload_overflow_runs_total")
            .with_description(
                "Total CreateLayer/UpdateLayer uploads that ran in the overflow slot, one at a \
                 time beyond the upload budget, by rpc (create_layer|update_layer).",
            )
            .build();
        let upload_processing_wait_seconds = meter
            .f64_histogram("live_edit_upload_processing_wait_seconds")
            .with_description(
                "Seconds a CreateLayer/UpdateLayer processing step, a structural UpdateLayer \
                 commit's recharge, or an UpdateLayer checkpoint-recovery re-issue's hold of one \
                 extra copy of its external values waited for room in the upload budget or for \
                 the overflow slot, by rpc, bounded by the RPC deadline; ~0 when granted at \
                 once. A commit's non-waiting growth of its charge records none.",
            )
            .build();
        let uploads_in_flight = meter
            .i64_up_down_counter("live_edit_uploads_in_flight")
            .with_description(
                "Admitted CreateLayer/UpdateLayer uploads, by phase (receiving|processing).",
            )
            .build();
        let upload_admission_waiters = meter
            .i64_up_down_counter("live_edit_upload_admission_waiters")
            .with_description(
                "CreateLayer/UpdateLayer uploads queued for admission, bounded by \
                 LIVE_EDIT_UPLOAD_ADMISSION_MAX_WAITERS.",
            )
            .build();
        let upload_budget_bytes = meter
            .u64_gauge("live_edit_upload_budget_bytes")
            .with_description(
                "The upload memory budget in bytes, derived at startup from the pod memory \
                 limit (live_edit_pod_memory_limit_bytes).",
            )
            .build();
        let pod_memory_limit_bytes = meter
            .u64_gauge("live_edit_pod_memory_limit_bytes")
            .with_description(
                "The pod memory limit every memory budget of this replica is derived from, in \
                 bytes, by service and source: cgroup_v2 or cgroup_v1 = the container's cgroup \
                 memory limit; host_memory = the host's physical memory, below the cgroup limit; \
                 reference = no readable limit, so the replica is sized as a 5 GiB pod. Recorded \
                 at startup; the source in use carries the limit, the others 0",
            )
            .build();
        let catchup_build_budget_bytes = meter
            .u64_gauge("live_edit_catchup_build_budget_bytes")
            .with_description(
                "The catch-up memory budget in bytes, derived at startup from the pod memory \
                 limit; read live_edit_catchup_build_budget_reserved_bytes against it.",
            )
            .build();
        let retirement_budget_bytes = meter
            .u64_gauge("live_edit_retirement_budget_bytes")
            .with_description(
                "The retirement memory allowance in bytes, derived at startup from the pod \
                 memory limit, that layer deletes, replaced layers' clean-up and the orphan \
                 sweep share; read live_edit_retirement_budget_reserved_bytes against it.",
            )
            .build();
        let structural_cache_budget_bytes = meter
            .u64_gauge("live_edit_structural_cache_budget_bytes")
            .with_description(
                "The structural write-planning cache's resident budget in bytes (its budget x \
                 1.21, for its entries' plan indexes), derived at startup from the pod memory \
                 limit; live_edit_structural_cache_weight_bytes stays at or below it.",
            )
            .build();
        let subscriber_max_subscriptions = meter
            .u64_gauge("live_edit_subscriber_max_subscriptions")
            .with_description(
                "SubscribeReadLayer subscriptions this replica admits at once, derived at \
                 startup from the pod memory limit; compare live_edit_subscriber_admitted \
                 against it.",
            )
            .build();
        let update_layer_chain_wait_seconds = meter
            .f64_histogram("live_edit_update_layer_chain_wait_seconds")
            .with_description(
                "Seconds a chained UpdateLayer waited for its predecessor in the chain gate, by \
                 outcome (ready|replaced|refused|timeout|unavailable|shutdown|cancelled), one \
                 observation per chained call at its gate exit.",
            )
            .build();
        let update_layer_chain_probes_total = meter
            .u64_counter("live_edit_update_layer_chain_probes_total")
            .with_description(
                "Total chain-gate probes of a chained UpdateLayer writer's record, by result \
                 (ok|error).",
            )
            .build();
        let update_layer_chain_waiters = meter
            .i64_up_down_counter("live_edit_update_layer_chain_waiters")
            .with_description("Chained UpdateLayer calls waiting in the chain gate.")
            .build();
        let update_layer_chain_waiting_bytes = meter
            .i64_up_down_counter("live_edit_update_layer_chain_waiting_bytes")
            .with_description(
                "Upload-budget bytes held by the chained UpdateLayer calls waiting in the chain \
                 gate; read against live_edit_upload_budget_bytes.",
            )
            .build();
        let update_layer_chain_detect_lag_seconds = meter
            .f64_histogram("live_edit_update_layer_chain_detect_lag_seconds")
            .with_description(
                "Seconds from a chained UpdateLayer's predecessor commit to the chain gate seeing \
                 it, on a ready exit after at least one sleep; includes clock skew between \
                 replicas.",
            )
            .build();
        let subscribe_catchups_inflight = meter
            .i64_up_down_counter("live_edit_subscribe_catchups_inflight")
            .with_description(
                "SubscribeReadLayer catch-up slots currently held, bounded by \
                 LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS, never across a send: a catch-up of \
                 its own holds one for its backing-store reads and build (in rollback mode, \
                 LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false, a same-object reconnect also across the \
                 cluster stream-reader wait); a shared fenced-read flight holds one from its read \
                 through its fence RETRY_AFTER sleeps, permit and memory waits and every attached \
                 catch-up's build turn; each of its extra builds holds one more.",
            )
            .build();
        let subscribe_catchups_waiting = meter
            .i64_up_down_counter("live_edit_subscribe_catchups_waiting")
            .with_description(
                "SubscribeReadLayer catch-ups, and shared fenced-read flights, waiting for a \
                 LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS slot",
            )
            .build();
        let subscribe_catchup_wait_seconds = meter
            .f64_histogram("live_edit_subscribe_catchup_wait_seconds")
            .with_description(
                "Seconds a SubscribeReadLayer catch-up waited for a concurrent-catch-up slot, by \
                 result (acquired|cancelled|timed_out), recorded on every acquisition including the \
                 uncontended fast path. timed_out: the subscription's capacity budget \
                 (LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS) ran out and it was refused \
                 catchup_capacity. A shared fenced-read flight's slot wait (one per flight) has \
                 no deadline - each attached catch-up is charged its share and refused alone - so \
                 it ends acquired or cancelled only and can run past the top bucket.",
            )
            .build();
        let catchup_snapshot_reads_in_flight = meter
            .i64_up_down_counter("live_edit_catchup_snapshot_reads_in_flight")
            .with_description(
                "Catch-up snapshot reads currently holding a read permit, by class \
                 (atomic|fenced). atomic is bounded by \
                 LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY: a read holds its permit only across \
                 its atomic read - a catch-up's, or one commit's of a legacy in-window replay, \
                 always inside the catch-up slot. fenced is bounded by \
                 LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY: a shared fenced-read flight holds its \
                 permit from before a fence acquire through the fenced read, the fence's release \
                 and the memory adjust after it, and gives it back during fence RETRY_AFTER \
                 sleeps.",
            )
            .build();
        let catchup_snapshot_read_wait_seconds = meter
            .f64_histogram("live_edit_catchup_snapshot_read_wait_seconds")
            .with_description(
                "Seconds a snapshot read waited for its read permit, by consumer \
                 (catchup|catchup_legacy_per_commit|structural_cache|structural_cache_audit) and class \
                 (atomic|fenced), \
                 recorded on every \
                 acquisition whatever its outcome, including the uncontended fast path. atomic \
                 (LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY): one catch-up's wait - a legacy \
                 in-window replay's per-commit waits included (an empty commit takes none) - \
                 charged to its capacity budget (LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS); one \
                 that runs out is refused catchup_capacity. fenced \
                 (LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY): a shared fenced-read flight's wait, \
                 which has no deadline - each catch-up attached to the flight is charged for the \
                 time it is attached during it and is refused catchup_capacity alone when its own \
                 budget runs out - so it ends acquired, at shutdown, or when the flight's last \
                 catch-up leaves, and can run past the top bucket. structural_cache (fenced only): \
                 a structural-cache cold start's wait, ended at its waiting cap. \
                 structural_cache_audit (fenced only): the sampled audit's, which never waits - a \
                 permit that is not free at once skips the audit.",
            )
            .build();
        let catchup_build_budget_reserved_bytes = meter
            .i64_up_down_counter("live_edit_catchup_build_budget_reserved_bytes")
            .with_description(
                "Bytes of catch-up memory reservations currently granted by the per-replica \
                 memory budget (live_edit_catchup_build_budget_bytes), in whole KiB.",
            )
            .build();
        let catchup_build_budget_overcommit_bytes = meter
            .i64_up_down_counter("live_edit_catchup_build_budget_overcommit_bytes")
            .with_description(
                "Bytes of never-blocking top-ups the catch-up memory budget currently carries \
                 beyond its grants (over-commit), in whole KiB; repaid when the memory that \
                 caused them is freed.",
            )
            .build();
        let catchup_build_wait_seconds = meter
            .f64_histogram("live_edit_catchup_build_wait_seconds")
            .with_description(
                "Seconds a catch-up memory reservation waited on the memory budget \
                 (live_edit_catchup_build_budget_bytes), recorded on every waiting \
                 reservation whatever its outcome, including the uncontended fast path.",
            )
            .build();
        let catchup_build_admission_total = meter
            .u64_counter("live_edit_catchup_build_admission_total")
            .with_description(
                "Catch-up memory budget admission decisions, by result \
                 (immediate|waited|refused_timeout|cancelled). cancelled: the budget closed at \
                 shutdown or the waiter stopped waiting.",
            )
            .build();
        let retirement_budget_reserved_bytes = meter
            .i64_up_down_counter("live_edit_retirement_budget_reserved_bytes")
            .with_description(
                "Bytes of retirement reservations currently granted by the retirement memory \
                 allowance (live_edit_retirement_budget_bytes), in whole KiB: a layer delete's, a \
                 replaced layer's or an orphan's clean-up, each holding a page sized from its \
                 largest structural index (up to a page bound) plus a charge per external-value \
                 reference it releases.",
            )
            .build();
        let retirement_budget_overcommit_bytes = meter
            .i64_up_down_counter("live_edit_retirement_budget_overcommit_bytes")
            .with_description(
                "Bytes of never-blocking top-ups the retirement memory allowance currently \
                 carries beyond its grants (over-commit), in whole KiB; repaid when the clean-up \
                 step that caused them ends.",
            )
            .build();
        let retirement_wait_seconds = meter
            .f64_histogram("live_edit_retirement_wait_seconds")
            .with_description(
                "Seconds a clean-up's reservation waited on the retirement memory allowance \
                 (live_edit_retirement_budget_bytes), by outcome (immediate|waited|cancelled), \
                 recorded on every reservation including the uncontended fast path. cancelled: \
                 shutdown ended a wait that was queued or would have had to queue, or the waiter \
                 stopped waiting.",
            )
            .build();
        let catchup_builds_in_progress = meter
            .i64_up_down_counter("live_edit_catchup_builds_in_progress")
            .with_description(
                "Catch-up builds currently running on this replica: the builds of catch-ups \
                 attached to a shared fenced-read flight, each from its build turn to the end of \
                 its build.",
            )
            .build();
        let catchup_snapshot_flight_total = meter
            .u64_counter("live_edit_catchup_snapshot_flight_total")
            .with_description(
                "Catch-ups that took part in a shared snapshot-read flight, by role \
                 (leader|joined_read): leader created the flight and caused its read, joined_read \
                 attached to one and shared its read. Read hit rate = joined_read / (leader + \
                 joined_read).",
            )
            .build();
        let subscriber_admitted = meter
            .i64_up_down_counter("live_edit_subscriber_admitted")
            .with_description(
                "SubscribeReadLayer subscriptions currently admitted, from admission to the end of \
                 the RPC. Compare against live_edit_subscriber_max_subscriptions.",
            )
            .build();
        let subscribe_catchups_pending = meter
            .i64_up_down_counter("live_edit_subscribe_catchups_pending")
            .with_description(
                "SubscribeReadLayer subscriptions whose first catch-up that reads layer state \
                 (every regime except up_to_date) has started and whose first caught-up ack is \
                 not yet sent: waiting for a catch-up slot or read permit, reading, building or \
                 sending. Falls when the ack is enqueued or the subscription ends before it. A \
                 steady-state re-bootstrap is not counted. The catch-up queue the autoscaler \
                 sizes the fleet to drain.",
            )
            .build();
        let subscriber_stream_exits_total = meter
            .u64_counter("live_edit_subscriber_stream_exits_total")
            .with_description(
                "Total SubscribeReadLayer subscriptions that ended, by reason \
                 (slow_reader_deadline|inner_watchdog|session_cancel|session_absent|shutdown|client_disconnect|reader_capacity|catchup_capacity|fenced_read_wait_cap|stream_gap|error). \
                 Complete over the handler's terminal arms, so sum by (reason) equals total \
                 exits.",
            )
            .build();
        let subscriber_delivery_lag_seconds = meter
            .f64_histogram("live_edit_subscriber_delivery_lag_seconds")
            .with_description(
                "Seconds from commit to hand-off of SubscribeReadLayer steady-state live \
                 deliveries, by stage: enqueued (the commit's last message put on the response \
                 channel) or transport (that message polled out by the gRPC encoder toward the \
                 service's HTTP/2 layer). One observation per subscription per window of up to 64 \
                 deliveries or about 1 s: the worst delivery in the window, not a per-delivery \
                 percentile. Measures exactly what live_subscriber_updates_total counts: no \
                 catch-up content, self-echo, duplicate or incomplete send. The commit time is the \
                 writing replica's clock, so cross-replica skew enters; negative values are clamped \
                 to 0. Time past the service's HTTP/2 layer (a proxy, the client) is not measured.",
            )
            .build();
        let subscriber_send_stalls_total = meter
            .u64_counter("live_edit_subscriber_send_stalls_total")
            .with_description(
                "Total SubscribeReadLayer sends that found all 32 data slots of the response \
                 channel full, by phase (catchup|live): the server-side HTTP/2 flow-control stall \
                 signal. Counted when the stall begins. catchup stalls are routine for multi-chunk \
                 catch-ups.",
            )
            .build();
        let subscriber_send_wait_seconds = meter
            .f64_histogram("live_edit_subscriber_send_wait_seconds")
            .with_description(
                "Seconds a stalled SubscribeReadLayer send (one counted by \
                 live_edit_subscriber_send_stalls_total) waited for a response-channel slot, by \
                 phase (catchup|live), recorded on every outcome: sent, slow-reader deadline (above \
                 the 5 s bucket), shutdown, session cancel, disconnect. Sends that found room are \
                 not recorded.",
            )
            .build();
        let subscriber_backlog_entries = meter
            .f64_histogram("live_edit_subscriber_backlog_entries")
            .with_description(
                "Entries queued in a SubscribeReadLayer subscription's stream-multiplexer backlog, \
                 sampled as each steady-state live delivery finishes enqueuing. One observation \
                 per subscription per window of up to 64 deliveries or about 1 s: the deepest \
                 sample in the window, not a per-delivery percentile.",
            )
            .build();
        let subscriber_backlog_bytes = meter
            .f64_histogram("live_edit_subscriber_backlog_bytes")
            .with_description(
                "Bytes queued in a SubscribeReadLayer subscription's stream-multiplexer backlog, \
                 sampled as each steady-state live delivery finishes enqueuing. One observation \
                 per subscription per window of up to 64 deliveries or about 1 s: the deepest \
                 sample in the window, not a per-delivery percentile.",
            )
            .build();
        let background_task_panics_total = meter
            .u64_counter("live_edit_background_task_panics_total")
            .with_description(
                "Total number of spawned background tasks observed to have panicked during \
                 shutdown, by task name",
            )
            .build();
        let gc_sweep_batch_errors_total = meter
            .u64_counter("live_edit_gc_sweep_batch_errors_total")
            .with_description(
                "Total number of external-value GC sweep batch failures, by error kind",
            )
            .build();
        let health_probe_failures_total = meter
            .u64_counter("live_edit_health_probe_failures_total")
            .with_description(
                "Total number of HealthSupervisor probe failures, by backend (valkey)",
            )
            .build();
        let checkpoint_writes_total = meter
            .u64_counter("live_edit_checkpoint_writes_total")
            .with_description(
                "Total per-layer checkpoint write attempts by the periodic loop, by result \
                 (success|error|skipped) and kind (layer|layer_tail|session|counters_sidecar). \
                 layer covers full-state bases, including the base a re-base composes; layer_tail \
                 covers stream-replay tail artifacts",
            )
            .build();
        let checkpoint_pending_age_seconds = meter
            .f64_histogram("live_edit_checkpoint_pending_age_seconds")
            .with_description(
                "Age in seconds of the newest durable checkpoint generation (now - \
                 last_checkpoint_ms), sampled at periodic-loop tick time for layers that have \
                 commits that generation does not carry. Fully-checkpointed idle \
                 layers and layers with no durable generation at all do NOT sample. p99 above \
                 LIVE_EDIT_CHECKPOINT_INTERVAL_SECS + tick slack + assembly time means the loop \
                 fell behind.",
            )
            .build();
        let checkpoint_oldest_pending_age_seconds = meter
            .f64_gauge("live_edit_checkpoint_oldest_pending_age_seconds")
            .with_description(
                "Maximum pending age (seconds) among layers still pending at the end of one \
                 completed checkpoint-loop tick, by service. Sawtooths between 0 and roughly \
                 LIVE_EDIT_CHECKPOINT_INTERVAL_SECS on a healthy active fleet; alert well above \
                 interval + quiesce + tick. Zero on non-leader replicas. Aggregate with max(), \
                 never sum().",
            )
            .build();
        let checkpoint_layers_never_written = meter
            .u64_gauge("live_edit_checkpoint_layers_never_written")
            .with_description(
                "Number of layers evaluated in one completed checkpoint-loop tick that have \
                 un-persisted commits and no durable checkpoint generation at all \
                 (last_checkpoint_ms == 0), by service. Zero on non-leader replicas. Aggregate \
                 with max(), never sum().",
            )
            .build();
        let checkpoint_counter_seed_secs = meter
            .f64_histogram("live_edit_checkpoint_counter_seed_secs")
            .with_description(
                "Duration of one counter-seed routine invocation by phase: startup = the \
                 one-shot startup recovery routine, repair = the lazy runtime object-id \
                 counter repair reached from CreateLayer.",
            )
            .build();
        let checkpoint_stable_cut_total = meter
            .u64_counter("live_edit_checkpoint_stable_cut_total")
            .with_description(
                "Checkpoint-assembler proven state reads by stable-cut outcome \
                 (verified|unverified) and service; unverified means the tick deliberately \
                 persisted nothing for that layer because the read's cut came back below the \
                 header sequence pinned before it (a header-sequence regression)",
            )
            .build();
        let checkpoint_tail_total = meter
            .u64_counter("live_edit_checkpoint_tail_total")
            .with_description(
                "Checkpoint tail-capture attempts by outcome (captured|captured_capped|rebased|\
                 refused_trimmed|refused_missing_bytes|refused_sentinel|refused_seq_gap|\
                 refused_no_base|refused_chain_full|refused_lineage_unrootable). Refusal outcomes \
                 fire only when the sound replayable prefix is empty; captured_capped marks a \
                 cap-bounded capture whose claimed range still equals its content. \
                 refused_no_base means the object has no durable generation at all (the \
                 post-enable rollout population); refused_lineage_unrootable means generations \
                 exist but none can root a lineage, which is checkpoint-store damage no tail knob \
                 clears. Pairs with \
                 live_edit_checkpoint_stable_cut_total{outcome=\"unverified\"} — both rising is the \
                 healthy signature.",
            )
            .build();
        let checkpoint_base_bootstrap_total = meter
            .u64_counter("live_edit_checkpoint_base_bootstrap_total")
            .with_description(
                "Base-bootstrap invocations by outcome (persisted|shed|skipped_header_absent|\
                 skipped_generation_present|skipped_untrusted_cursor|\
                 unstamped_generation_present|unstamped_foreign_restore|unstamped_write_error|\
                 failed) and site (creation_base|restore_rebase). Exactly one increment per \
                 BaseBootstrap::run_once, so sum by (outcome) equals the invocation count — \
                 which is what makes shed/total an admission-saturation ratio and a zero total \
                 rate a producer-liveness predicate. The unstamped_* group means the base blob \
                 is durable but the header does not point at it. Additive to \
                 live_edit_checkpoint_writes_total, never a substitute.",
            )
            .build();
        let checkpoint_lease_total = meter
            .u64_counter("live_edit_checkpoint_lease_total")
            .with_description(
                "Checkpoint-loop mid-tick leader-lease checks by outcome (renewed|lost|\
                 refresh_error) and service; lost means the tick aborted its remaining layers \
                 because it could not prove the lease was continuously held",
            )
            .build();
        let delta_create_section_skipped_missing_parent_total = meter
            .u64_counter("live_edit_delta_create_section_skipped_missing_parent_total")
            .with_description(
                "Total CommandCreateSection deltas silently skipped because the named parent_id \
                 was not in state.sections",
            )
            .build();
        let checkpoint_reconcile_total = meter
            .u64_counter("live_edit_checkpoint_reconcile_total")
            .with_description(
                "Total checkpoint-store reconciliation-pass outcomes, by outcome \
                 (orphan_deferred|orphan_deleted|temp_deleted|ext_tombstoned|ext_collected|\
                 ext_spared|live_set_failed) and service.",
            )
            .build();
        let checkpoint_adapter_ceiling_elapsed_total = meter
            .u64_counter("live_edit_checkpoint_adapter_ceiling_elapsed_total")
            .with_description(
                "Total times a checkpoint-adapter wall-clock ceiling elapsed and the work was \
                 abandoned as Transient, by phase (call|enumeration) and service. A saturation \
                 signal, not a loss signal — every caller defers and retries on its next \
                 cadence.",
            )
            .build();
        let checkpoint_adapter_redirect_total = meter
            .u64_counter("live_edit_checkpoint_adapter_redirect_total")
            .with_description(
                "Total storage-api client-side redirect events on the checkpoint adapter, by verb \
                 (get|put), outcome (followed|completed|target_failed|complete_failed|\
                 multipart_reissued|malformed|preflight_redirect|preflight_unavailable|\
                 preflight_stale) and service. verb=put is the expected path for \
                 artifacts at or above the deployment's writeBodyMaxSize; verb=get should stay at \
                 zero because reads declare download_preference=body.",
            )
            .build();
        let checkpoint_ext_tombstone_backlog = meter
            .u64_gauge("live_edit_checkpoint_ext_tombstone_backlog")
            .with_description(
                "Number of ext-blob hashes currently tombstoned on le:ckpt:{recon}:tomb awaiting \
                 their grace window or a successful collect, by service. Sampled once per \
                 reconciliation pass by the leader; published as 0 by non-leaders so a demoted \
                 leader does not fossilise a stale backlog into the fleet-wide max().",
            )
            .build();
        let checkpoint_dead_object_backlog = meter
            .u64_gauge("live_edit_checkpoint_dead_object_backlog")
            .with_description(
                "Number of deleted object_ids currently registered on le:ckpt:{recon}:dead \
                 awaiting the reconciliation pass's dead-object phase, by service. Producers are \
                 on the request path (DeleteLayer, CreateLayer overwrite); the drain runs once \
                 per reconciliation pass. Sampled once per pass by the leader; published as 0 by \
                 non-leaders so a demoted leader does not fossilise a stale backlog into the \
                 fleet-wide max().",
            )
            .build();
        let checkpoint_recovery_attempts_total = meter
            .u64_counter("live_edit_checkpoint_recovery_attempts_total")
            .with_description(
                "Total recovery-hook attempts, by result (restored|not_recoverable|error|\
                 already_present), scope (layer|session), and service",
            )
            .build();
        let checkpoint_recovery_stream_refused_total = meter
            .u64_counter("live_edit_checkpoint_recovery_stream_refused_total")
            .with_description(
                "Total layer restores whose surviving per-object stream carried a high-water ID \
                 the restore's plausibility gate refused (tombstone auto-ID or other \
                 non-sequence writer), and which therefore had that stream removed, by service",
            )
            .build();
        let checkpoint_restore_residue_wiped_total = meter
            .u64_counter("live_edit_checkpoint_restore_residue_wiped_total")
            .with_description(
                "Total layer restores that found a surviving obj:* keyspace under the object being \
                 materialized (header gone or structurally incomplete, indexes/records/children/\
                 tombstones alive) and wiped it before writing, by service. Restore semantics are \
                 an authoritative replace, so the residue is removed rather than merged into",
            )
            .build();
        let checkpoint_restore_fenced_total = meter
            .u64_counter("live_edit_checkpoint_restore_fenced_total")
            .with_description(
                "Total layer-restore attempts that found a peer replica had already landed this \
                 object's rebuild and therefore wrote nothing, by stage (presence_probe = seen \
                 before any work; residue_wipe = the peer's header landed under the residue \
                 wipe's WATCH; object_write = it landed under the materialization's own WATCH, \
                 the innermost fence and the only one of the two WATCH-held fences reachable on \
                 a total-loss restore) and service. Additive to \
                 live_edit_checkpoint_recovery_attempts_total, never a \
                 substitute: the fenced attempt still reports result=\"restored\" there",
            )
            .build();
        let checkpoint_recovery_stream_unresolved_total = meter
            .u64_counter("live_edit_checkpoint_recovery_stream_unresolved_total")
            .with_description(
                "Total layer restores that could not establish the state of the per-object stream \
                 key, by reason (probe_unreadable | unlink_unverified = evidence unobtainable or \
                 unactionable, restore REFUSED; wrongtype_key = a foreign value sits on the stream \
                 key, restore proceeded and writes to the layer fail closed) and service",
            )
            .build();
        let checkpoint_restore_lineage_total = meter
            .u64_counter("live_edit_checkpoint_restore_lineage_total")
            .with_description(
                "Restore-source lineage-selection outcomes by outcome (gathered_base|\
                 gathered_composed|stepped_down_link_broken|stepped_down_link_mismatch|\
                 stepped_down_decode|stepped_down_missing_meta|stepped_down_budget|\
                 stepped_down_compose|exhausted|exhausted_budget) and \
                 service. Additive to live_edit_checkpoint_recovery_attempts_total, never a \
                 substitute: a step-down is intra-attempt and may fire several times per attempt.",
            )
            .build();
        let checkpoint_session_publish_superseded_total = meter
            .u64_counter("live_edit_checkpoint_session_publish_superseded_total")
            .with_description(
                "Total session-checkpoint adapter operations whose publish was refused by the \
                 publish watermark, by result (write | delete = landed on top of a \
                 strictly fresher operation of that kind, repair scheduled; stale_incarnation = \
                 the le:{sid}:ckpt key the stamp was minted against was lost or re-created, \
                 rebuild scheduled; exhausted = repair budget spent)",
            )
            .build();
        let gc_underflow_total = meter
            .u64_counter("live_edit_gc_underflow_total")
            .with_description(
                "Total external-value GC decrement operations whose post-decrement ref_count went \
                 below zero: a decrement with no increment behind it — an older \
                 build's replay, a legacy bare release member, or a bug. Replays of this build's \
                 release members answer replayed without decrementing, so the baseline is \
                 zero and any increase pages",
            )
            .build();
        let gc_decrement_total = meter
            .u64_counter("live_edit_gc_decrement_total")
            .with_description(
                "Total external-value GC decrement runs, by outcome: applied = the \
                 release member's decrement applied now; replayed = it had already applied (a \
                 fred re-send or the re-claim of a stale claim), so nothing was decremented; \
                 stale = the run executed more than the claim-age fence after its claim and wrote \
                 nothing (the member waits to be re-claimed). A sustained stale rate means a \
                 stalled store shard or a primary clock running ahead of the {gc} primary's",
            )
            .build();
        let gc_older_build_total = meter
            .u64_counter("live_edit_gc_older_build_total")
            .with_description(
                "Total actions the external-value GC took on state an older build (one without \
                 the GC leader-lock protocol marker, or with a lower version) left in the \
                 store, by event: lock_taken_over = took the GC leader lock over from it; \
                 claim_dropped = dropped one of its inflight claims without decrementing; \
                 release_stamped = stamped an age on a pending release it enqueued without one; \
                 carrier_dropped = dropped, without paying, one release reference on a \
                 retirement carrier none of whose recorded attempts landed. Outside a rollout or \
                 rollback across the marker, any increase means an older build is acting on \
                 this store",
            )
            .build();
        let gc_oom_collections_total = meter
            .u64_counter("live_edit_gc_oom_collections_total")
            .with_description(
                "Total out-of-memory collections the external-value GC leader ran after the \
                 backing store refused a write for lack of memory, by outcome: freed = a completed \
                 pass deleted released external values; freed_nothing = found released ones but deleted none \
                 (each still referenced, or the store refused the re-claims or age stamps on its \
                 full {gc} shard, so owed releases wait for room there; the collection's WARN \
                 line tells which); nothing_to_collect = found none to collect (none owed, every \
                 owed release held by a reader's read hold, or releases refused at enqueue on a \
                 full {gc} shard that wait for the reference-count reconciler); failed = no completed \
                 pass deleted any (a failing pass's deletions are not counted) and a pass ended in \
                 an error; refused = deleted none because the store refused the \
                 collection's own scripts for lack of memory (the store did not run them past \
                 their first write on a full store). If writes stay refused, repeated \
                 nothing_to_collect, or freed_nothing with nothing skipped, means the store is \
                 full of data in use and needs more memory",
            )
            .build();
        let gc_oom_freed_bytes_total = meter
            .u64_counter("live_edit_gc_oom_freed_bytes_total")
            .with_description(
                "Total bytes of released external values the external-value GC's \
                 out-of-memory collections deleted",
            )
            .build();
        let ext_release_pending_bytes = meter
            .u64_gauge("live_edit_ext_release_pending_bytes")
            .with_description(
                "Bytes the external-value release ledger accounts over pending and inflight \
                 release members: released blobs awaiting garbage collection, by service. \
                 Published every GC pass by the le:gc:leader holder; 0 from \
                 non-leaders; not published by a pass that errored. The release budget \
                 LIVE_EDIT_EXT_RELEASE_MAX_BYTES holds back uploads carrying external values \
                 while this total is at or above it; other releases (DeleteLayer, overwrites \
                 and edits that drop references, reference-count repairs) and uploads let \
                 through on a failed read of the total or admitted together on its first \
                 reading below the budget can take it past the budget",
            )
            .build();
        let ext_release_pending_count = meter
            .u64_gauge("live_edit_ext_release_pending_count")
            .with_description(
                "External-value release members owed a decrement, pending plus inflight, by \
                 service. Same leader-published contract as \
                 live_edit_ext_release_pending_bytes. A member re-enqueued while its claim is \
                 held is counted twice, transiently, until that claim settles",
            )
            .build();
        let ext_release_oldest_age_seconds = meter
            .f64_gauge("live_edit_ext_release_oldest_age_seconds")
            .with_description(
                "Age in seconds of the oldest pending external-value release member, 0 when \
                 none, by service. Same leader-published contract as \
                 live_edit_ext_release_pending_bytes. A level that keeps rising past the GC age \
                 floor means collection is not keeping up",
            )
            .build();
        let ext_release_settled_count = meter
            .u64_gauge("live_edit_ext_release_settled_count")
            .with_description(
                "External-value release members settled out of the GC sets whose decrement \
                 replay records await teardown, by service. Same leader-published \
                 contract as live_edit_ext_release_pending_bytes. Bounded by the release rate \
                 times the replay-record retention; growth without bound means teardown is \
                 failing",
            )
            .build();
        let ext_release_gate_total = meter
            .u64_counter("live_edit_ext_release_gate_total")
            .with_description(
                "Uploads carrying external values that met a full external-value release budget \
                 (LIVE_EDIT_EXT_RELEASE_MAX_BYTES), by outcome: passed_after_wait = \
                 proceeded once the release total dropped below the budget; refused = the \
                 budget stayed full for as long as the call's deadline allowed (its remaining \
                 deadline less a margin) and the upload was refused UNAVAILABLE with nothing \
                 applied; read_failed = the release total could not be \
                 read and the upload was let through. An upload admitted at once is not counted",
            )
            .build();
        let ext_release_budget_bytes = meter
            .u64_gauge("live_edit_ext_release_budget_bytes")
            .with_description(
                "The external-value release budget (LIVE_EDIT_EXT_RELEASE_MAX_BYTES) this \
                 replica's upload gate and GC pressure trigger use, in bytes, by service and \
                 source: \
                 override = from configuration (LIVE_EDIT_EXT_RELEASE_MAX_BYTES, 0 = disabled; \
                 or, when the store reports no usable maxmemory, one shard's share of \
                 LIVE_EDIT_VALKEY_MEMORY_BYTES, the store's total memory, x 0.8 / 8 by the \
                 cluster map's slot ranges, re-read every 60 s); info = the smallest maxmemory \
                 the store's primaries report in INFO memory, / 8, re-read every 60 s; fallback \
                 = min(8 x LIVE_EDIT_MAX_LAYER_SIZE, 100 MB) because nothing else applies. The \
                 source in use carries the budget, the others 0",
            )
            .build();
        let ext_read_holds = meter
            .u64_gauge("live_edit_ext_read_holds")
            .with_description(
                "Live GC read hold fields at the GC leader's last pass, by service: catch-ups \
                 re-reading after an external-value miss, each stopping the collection of \
                 releases newer than its start while it lives. Same leader-published contract \
                 as live_edit_ext_release_pending_bytes. Returns to 0 between miss waves; a \
                 level that stays above 0 while uploads wait at the release budget is a hold, \
                 not a GC stall",
            )
            .build();
        let catchup_ext_value_miss_total = meter
            .u64_counter("live_edit_catchup_ext_value_miss_total")
            .with_description(
                "Catch-up builds that found an external-value blob missing, by outcome and \
                 read: stale_retried = the catch-up took a GC read hold and re-read the layer's \
                 state (a commit after its snapshot released the blob and the GC collected it); \
                 genuine = still missing when rebuilt under a hold the store confirmed live, so \
                 the build took the checkpoint-adapter fetch-on-miss path and the catch-up ends \
                 INTERNAL if that cannot restore it. A genuine miss also counts stale_retried \
                 first, under the same read. read = where the miss was found: atomic (the \
                 catch-up's own atomic read; its re-read is one more atomic read, or a read \
                 through the layer's fenced-read flight if the layer grew above the atomic-read \
                 limits) or shared_fenced (a snapshot shared through the layer's fenced-read \
                 flight; its re-read re-attaches to a successor flight, one more fenced read \
                 shared with every catch-up attached to it)",
            )
            .build();
        let ext_ref_released_total = meter
            .u64_counter("live_edit_ext_ref_released_total")
            .with_description(
                "Total external-value ext_meta references handed back to the GC pending set, \
                 by reason. rewrite = a commit superseded the object's last row holding \
                 that hash (rewritten to another hash, rewritten inline, or deleted); \
                 surplus_acquire = the object already referenced that hash, so the pre-flight \
                 store_values increment was surplus; retirement = the object itself retired; \
                 reconcile = the reference-count reconciler observed ref_count standing above the \
                 live-object reference count twice a grace apart and handed the difference back",
            )
            .build();
        let ext_ref_release_enqueue_failed_total = meter
            .u64_counter("live_edit_ext_ref_release_enqueue_failed_total")
            .with_description(
                "Total owed ext_meta references the service could neither RECORD on \
                 le:ext:{gc}:pending nor record on a durable carrier — the leak signal, and an \
                 UPPER BOUND on the references actually leaked (a client-side timeout or I/O \
                 error on a script the server did run ticks N while leaking nothing). Counted \
                 per reference, not per call: one failed batch owing N adds N. RETRY \
                 AMPLIFICATION survives on the retirement ticks: a failed begin did not write \
                 ext_gc_scheduled, so a delete_layer cascade re-run \
                 re-derives the same members and re-ticks the whole batch (bounded by \
                 LIVE_EDIT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS; the other two callers do not loop). \
                 Counted = commit-path enqueue failures after one in-line retry; and \
                 retirement-path obligations whose begin (recording them on the object's \
                 le:{oid}:obj:extret carrier before the enqueue) failed or found a release \
                 record already on the object id — an earlier lifecycle's carrier (the orphan \
                 sweep settles it) or, only after a rolling upgrade, attempt tokens with no \
                 carrier, which this release does not remove, so every retirement of that id \
                 ticks here. NOT counted = retirement \
                 pre-schedule \
                 read failures (HGET ext_gc_scheduled, the reference walk of the value rows and \
                 obj:ext), where the \
                 owed count is not yet known; a process dying between a commit's EXEC and its \
                 enqueue; undecodable release tokens, value-row hash columns or obj:ext fields \
                 (see live_edit_ext_ref_undecodable_total); a retirement enqueue whose outcome \
                 is in doubt (see live_edit_ext_ref_release_deferred_total); and a failed \
                 orphan-sweep re-drive, whose carrier keeps every member. Read a tick as UP TO \
                 one ref_count that will not be decremented until the reference-count \
                 reconciler drains it",
            )
            .build();
        let ext_ref_release_deferred_total = meter
            .u64_counter("live_edit_ext_ref_release_deferred_total")
            .with_description(
                "Total owed ext_meta references whose retirement-path enqueue ended in doubt \
                 (both attempts failed, so whether either ran is unknown), or was not sent within \
                 the retirement's send window (nothing was sent), or whose first attempt failed \
                 in doubt and whose retry was then not sent within the window (the first \
                 attempt may have run), and were left on the retiring \
                 object's own le:{oid}:obj:extret carrier, written before the enqueue, for the \
                 orphan sweep's retire-retry phase to settle. Counted per \
                 reference. Not a leak on its own — read against \
                 live_edit_ext_ref_release_recovered_total, \
                 live_edit_ext_ref_release_already_enqueued_total and \
                 live_edit_ext_ref_release_cancelled_total, which between them should follow it \
                 within about one orphan-sweep interval (deferred is approximately recovered + \
                 cancelled + already_enqueued); a gap means the drain is not running or is \
                 failing, and live_edit_ext_retire_retry_backlog says which",
            )
            .build();
        let ext_ref_release_recovered_total = meter
            .u64_counter("live_edit_ext_ref_release_recovered_total")
            .with_description(
                "Total recorded ext_meta references the orphan sweep's retire-retry phase \
                 re-drove onto le:ext:{gc}:pending, each only after every recorded \
                 earlier enqueue attempt of the carrier was decided void; only carriers \
                 this build wrote, which record every attempt, are re-driven — an older \
                 build's carrier is cleared or dropped, never re-driven. Counted \
                 per reference. One completion arm of live_edit_ext_ref_release_deferred_total; \
                 the recovered members are byte-identical to the ones the original retirement \
                 would have written",
            )
            .build();
        let ext_ref_release_already_enqueued_total = meter
            .u64_counter("live_edit_ext_ref_release_already_enqueued_total")
            .with_description(
                "Total recorded ext_meta references the orphan sweep's retire-retry phase found \
                 already enqueued — an earlier attempt of the carrier had landed although its \
                 caller never learned it — and cleared without enqueuing again. Counted \
                 per reference. One completion arm of live_edit_ext_ref_release_deferred_total \
                 (deferred is approximately recovered + cancelled + already_enqueued); each tick \
                 is a second decrement that did not happen",
            )
            .build();
        let ext_ref_release_cancelled_total = meter
            .u64_counter("live_edit_ext_ref_release_cancelled_total")
            .with_description(
                "Total recorded ext_meta references the orphan sweep's retire-retry phase \
                 DROPPED because the object id was live again when every recorded earlier \
                 enqueue attempt had been decided void. Counted per reference. One \
                 completion arm of live_edit_ext_ref_release_deferred_total (deferred is \
                 approximately recovered + cancelled + already_enqueued). A live id has \
                 inherited the standing unreleased +1 per hash those members stood for, so \
                 paying them would delete ext_meta+ext_data under a live reference; cancelling \
                 leaks instead, and the reference-count reconciler drains the leak. Read together with live_edit_ext_ref_deficit_total: a non-zero rate \
                 here with a flat zero deficit is the gate working as designed",
            )
            .build();
        let ext_ref_undecodable_total = meter
            .u64_counter("live_edit_ext_ref_undecodable_total")
            .with_description(
                "Total external-value references dropped because the hash could not be decoded \
                 where it was read, by site (lua_token|row_hash|ext_index_field|\
                 retirement_index_field). Dropping rather than guessing is deliberate: a \
                 mis-decoded hash would decrement the WRONG blob and can delete live data, while \
                 a drop leaks one reference. No dedicated recovery path exists or can — the hash \
                 is unidentifiable at the drop site — so these are drained by the reference-count \
                 reconciler as ordinary surplus. Expected to be zero; any tick is keyspace \
                 corruption and the accompanying log line names the undecodable text",
            )
            .build();
        let ext_meta_total = meter
            .u64_gauge("live_edit_ext_meta_total")
            .with_description(
                "Number of le:ext:meta:* keys enumerated by the last reference-count reconciler \
                 pass, by service. Sampled once per pass by the leader; published as 0 by \
                 non-leaders so a demoted leader does not fossilise a stale level into the \
                 fleet-wide max(); published not at all by a pass that errored (unknown is not \
                 zero). Read as a pair with live_edit_ext_referenced_hashes_total",
            )
            .build();
        let ext_referenced_hashes_total = meter
            .u64_gauge("live_edit_ext_referenced_hashes_total")
            .with_description(
                "Number of distinct external hashes named by at least one live object, as counted \
                 by the reference-count reconciler's object walk (union of each object's value \
                 rows and its obj:ext index), by service. Same leader-published contract \
                 as live_edit_ext_meta_total",
            )
            .build();
        let ext_ref_surplus_total = meter
            .u64_gauge("live_edit_ext_ref_surplus_total")
            .with_description(
                "Summed positive surplus across every ext:meta — ref_count units standing above \
                 the live-object reference count after subtracting the releases already owed on \
                 le:ext:{gc}:pending and :inflight, by service. The leak, measured. Same \
                 leader-published contract as live_edit_ext_meta_total. A level that ratchets \
                 upward pass over pass means a release path is failing faster than the reconciler \
                 repairs",
            )
            .build();
        let ext_ref_deficit_total = meter
            .u64_gauge("live_edit_ext_ref_deficit_total")
            .with_description(
                "Summed CONFIRMED deficit across every ext:meta — ref_count units standing BELOW \
                 the live-object reference count that were still standing, at the same amount, a \
                 full grace after the pass first observed them, by service. ALERTABLE ON \
                 RECURRENCE, not on a single confirmed report: alert when \
                 max_over_time(...[N intervals]) is non-zero across several reconciler intervals. \
                 Confirmation proves an amount held still for one grace, not that the deficit \
                 persists — a hash whose references churn at a rate comparable to a pass can \
                 present the same mid-pass double-count on the confirming pass and confirm a \
                 transient (per-object attribution, which would settle it, is not implemented). A \
                 recurring deficit is an over-release that already happened: the value is still \
                 stored when reported, but will be collected under a live reference once the \
                 count reaches 0, and may already have been collected once and stored again. A \
                 single pass's negative surplus is not \
                 even a confirmation (the pass's phase order double-counts an obligation that \
                 transitions mid-pass, and a retire carrier whose object is live again is counted \
                 twice for one sweep interval), which is why only the confirmed figure is \
                 published here; first observations show up as \
                 live_edit_ext_reconcile_total{outcome=\"candidate_registered\"} and the ones that \
                 turn out benign as outcome=\"deficit_cleared\". Deliberately never repaired — \
                 an increment racing the GC's delete could resurrect a meta whose data is gone — \
                 so this gauge is the whole response. max_over_time is also what makes the series readable at \
                 all: a confirmed candidate is retired once counted, so a standing deficit reports \
                 every other pass. Same leader-published contract as live_edit_ext_meta_total",
            )
            .build();
        let ext_retire_retry_backlog = meter
            .u64_gauge("live_edit_ext_retire_retry_backlog")
            .with_description(
                "Number of retiring objects registered on le:ext:{gc}:retire-retry whose deferred \
                 external-value release obligations are awaiting the orphan sweep's retire-retry \
                 phase, by service. Sampled once per orphan-sweep tick by the leader; \
                 published as 0 by non-leaders for the same non-decaying-gauge reason as \
                 live_edit_checkpoint_ext_tombstone_backlog. Read as a derivative: a backlog that \
                 only rises means the drain is failing",
            )
            .build();
        let ext_reconcile_total = meter
            .u64_counter("live_edit_ext_reconcile_total")
            .with_description(
                "Total reference-count reconciler outcomes, by outcome \
                 (candidate_registered|repaired|healed|deferred|skipped_recent_store|\
                 skipped_unreadable_stamp|enumeration_failed|deficit_observed|deficit_cleared|\
                 skipped_index_drift). \
                 skipped_index_drift is a surplus held because an object's per-object \
                 reference index disagreed with its rows and could not be repaired this pass \
                 (too large, not live, or a malformed hash column the recount cannot count): \
                 the hold leaks rather than drains, and does not \
                 stop a commit path releasing on the unrepaired index. \
                 enumeration_failed is alertable per occurrence. deficit_observed counts only \
                 deficits CONFIRMED a grace apart — a benign mid-pass double-count ticks \
                 deficit_cleared instead — but confirmation proves an amount held still for one \
                 grace, not a persistent deficit, so alert it ON RECURRENCE across several \
                 reconciler intervals rather than on a single report, matching \
                 live_edit_ext_ref_deficit_total. Sustained deferred means the per-pass budget no \
                 longer covers the backlog; candidate_registered and healed in balance with no \
                 repaired \
                 is the two-observation guard correctly declining to act on transients",
            )
            .build();
        let ext_index_recount_total = meter
            .u64_counter("live_edit_ext_index_recount_total")
            .with_description(
                "Total atomic recounts of a layer's per-object external-value reference index \
                 against its value rows, by site (reconcile|torn_commit) and outcome \
                 (repaired|clean|too_large|not_live|failed|withheld). The reference-count \
                 reconciler \
                 recounts an object whose index disagrees with its rows; the txn commit path \
                 recounts after a torn commit. A recount that finds drift on a live object \
                 within its row ceiling repairs the index atomically; none releases anything. \
                 clean at site=reconcile is expected for legacy rows carrying both an inline \
                 value and a stale hash. too_large|not_live at site=reconcile hold that object's \
                 hashes from surplus drain (live_edit_ext_reconcile_total\
                 {outcome=\"skipped_index_drift\"}): the hold leaks rather than drains, while \
                 the unrepaired index stays exposed to the commit paths. withheld: the recount \
                 saw a value row whose hash column it cannot count (keyspace corruption), so its \
                 count is not authoritative; recorded at both sites for every such recount, \
                 whatever it raised. It applies raises only and withholds every lowering, so \
                 this is not a repair and needs an operator (the WARN names the layer). At \
                 site=reconcile it holds the layer's hashes like too_large until the rows are \
                 corrected. failed at \
                 site=reconcile forfeits the pass; at site=torn_commit the drift is left to the \
                 next reconciler pass's audit, which repairs it, or holds the hash from surplus \
                 drain when it cannot, and until then a commit releasing on the drifted count \
                 can over-release. Not self-healing in every case: a wrong-typed key that \
                 fails the recount also forfeits every reconciler pass until it is cleared, and \
                 LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0 disables the audit and the drain",
            )
            .build();
        let ext_index_drift_total = meter
            .u64_counter("live_edit_ext_index_drift_total")
            .with_description(
                "Total per-object external-value reference-index fields corrected by a recount, \
                 by site (reconcile|torn_commit) and direction (under|over). under: the index \
                 counted fewer rows than hold the hash, the direction that can release a value a \
                 live row still references; over: more, a leak. A burst on the first reconciler \
                 pass after upgrading a deployment that holds layers from before 0.1.0 is \
                 migration; \
                 direction=under recurring after that means something still produces drift",
            )
            .build();
        let time_sample_signed_zero_merge_total = meter
            .u64_counter("live_edit_time_sample_signed_zero_merge_total")
            .with_description(
                "Total time-sample rows the reference-count reconciler's signed-zero merge \
                 handled, by outcome (merged|rekeyed|changed|fenced|not_live|undecodable|failed). \
                 -0.0 and 0.0 are one USD time code; this build stores every time at 0.0, so a \
                 -0.0 row was written by an older build. merged: a section's -0.0 and 0.0 rows \
                 became one row at 0.0 (including a 0.0 row missing from the layer's time-sample \
                 index, which readers never saw); rekeyed: a lone -0.0 row moved to 0.0. Neither \
                 changes what readers serve. A burst after an upgrade is the migration; merged or \
                 rekeyed recurring once every replica runs this build means something still \
                 writes -0.0. changed (a commit landed meanwhile), fenced (a snapshot reader held \
                 the layer) and not_live wrote nothing and are retried on the next pass. \
                 undecodable (a row readers refuse too) and failed (a store error) recurring \
                 need an operator; the WARN names the layer",
            )
            .build();
        let object_id_counter_repair_total = meter
            .u64_counter("live_edit_object_id_counter_repair_total")
            .with_description(
                "Total object-id counter repairs triggered by a CreateLayer that found \
                 le:counter:object_id absent, by outcome. repaired_seeded / \
                 repaired_noop: a counters sidecar exists, so the absence is evidence the Valkey \
                 keyspace was lost underneath a running pod — alert. failed / \
                 failed_after_repair: the repair could not complete, or the counter was absent \
                 again right after it, and CreateLayer returned UNAVAILABLE — alert. \
                 repaired_initialized / unhooked_initialized: counter established at 0 against a \
                 store holding no counters sidecar — expected once on a fresh deployment. \
                 already_repaired: benign singleflight dedup.",
            )
            .build();
        let service_token_parse_errors_total = meter
            .u64_counter("live_edit_service_token_parse_errors_total")
            .with_description(
                "Total service-identity token file parse failures observed by the \
                 ServiceTokenProvider polling task. Unlabeled; \
                 per-failure context is logged at WARN with no token bytes attached.",
            )
            .build();
        let checkpoint_ext_mirror_dropped_total = meter
            .u64_counter("live_edit_checkpoint_ext_mirror_dropped_total")
            .with_description(
                "Total external values whose backup copy the checkpoint mirror dropped, by reason: \
                 queue_full (the channel held its item capacity), budget_full (no room left in \
                 the byte budget), oversized (larger than the whole byte budget), shutdown \
                 (still queued when the replica stopped). The value stays stored, and the \
                 backup repair re-sends its copy within minutes while the store holds it \
                 (an oversized one only once the upload size limit admits it again, or once \
                 resources.limits.memory is raised until the copy memory \
                 (live_edit_checkpoint_ext_mirror_budget_bytes) exceeds the value's size); until \
                 then a restore after the store loses it cannot bring it back.",
            )
            .build();
        let checkpoint_ext_mirror_bytes = meter
            .i64_up_down_counter("live_edit_checkpoint_ext_mirror_bytes")
            .with_description(
                "Bytes of external values held for the checkpoint mirror right now, charged \
                 against live_edit_checkpoint_ext_mirror_budget_bytes, by state (queued|in_flight); \
                 each value is charged its size rounded up to a KiB, plus 1 KiB.",
            )
            .build();
        let checkpoint_ext_mirror_budget_bytes = meter
            .u64_gauge("live_edit_checkpoint_ext_mirror_budget_bytes")
            .with_description(
                "The checkpoint mirror's byte budget, derived at startup from the pod memory limit \
                 (live_edit_pod_memory_limit_bytes): 1/40 of it, and never less than one \
                 maximum-size value. 0 when the mirror is not wired (checkpointing off or no \
                 adapter).",
            )
            .build();
        let checkpoint_ext_mirror_errors_total = meter
            .u64_counter("live_edit_checkpoint_ext_mirror_errors_total")
            .with_description("Total worker-side put_ext_blob failures during ext-value mirroring.")
            .build();
        let checkpoint_ext_backup_repair_total = meter
            .u64_counter("live_edit_checkpoint_ext_backup_repair_total")
            .with_description(
                "External values the backup repair handled, by outcome: already_stored (the \
                 backup store held the bytes; confirmed with no upload; includes the daily \
                 audit's re-checks), resent (uploaded from the store), resend_failed (the value \
                 stays owed; a failed upload also stops the pass re-sending, a failed store read \
                 fails that value alone), \
                 probe_failed (an existence check failed; the pass stops checking and \
                 re-sending), value_gone (collected before it could be read), budget_busy (no \
                 room in the mirror's byte budget; the pass stops re-sending), oversized (larger \
                 than the whole byte budget; stays owed), confirmed_copy_missing (the daily audit \
                 found a confirmed copy missing; re-sent unless re-sending stopped for the pass).",
            )
            .build();
        let checkpoint_ext_backup_owed = meter
            .u64_gauge("live_edit_checkpoint_ext_backup_owed")
            .with_description(
                "External values older than the backup repair's age gate with no confirmed copy \
                 in the backup store, at the end of the leader's last complete repair pass. \
                 Non-leader replicas publish 0; a pass that did not complete publishes nothing. \
                 Read it with max() across replicas.",
            )
            .build();
        let checkpoint_ext_backup_mark_failed_total = meter
            .u64_counter("live_edit_checkpoint_ext_backup_mark_failed_total")
            .with_description(
                "Failed writes of an external value's backup confirmation, by op: confirm \
                 (after a successful upload, or the repair's existence check that found the \
                 bytes; the repair checks the value again) or clear (after the checkpoint \
                 collect deleted its backup copy).",
            )
            .build();
        let checkpoint_ext_recovery_total = meter
            .u64_counter("live_edit_checkpoint_ext_recovery_total")
            .with_description(
                "Total external-value read_batch miss recoveries via CheckpointStore::get_ext_blob, \
                 by result (recovered|adapter_miss|adapter_transient|adapter_permanent|\
                 adapter_other)",
            )
            .build();
        let ext_value_verify_total = meter
            .u64_counter("live_edit_ext_value_verify_total")
            .with_description(
                "External-value references checked by the CreateLayer / UpdateLayer content \
                 check, by result (ok|hash_mismatch|size_mismatch), in both \
                 LIVE_EDIT_EXT_VALUE_VERIFY modes",
            )
            .build();
        let service_token_available = meter
            .i64_gauge("live_edit_service_token_available")
            .with_description(
                "1 while the service-identity token cached from the sidecar-written token file is \
                 unexpired, 0 when ServiceTokenProvider::cached_bearer_if_unexpired() returns \
                 None — i.e. no token has ever been cached, or the cached one has passed its \
                 expiry. A valid cached token still reads 1 inside the pre-expiry refresh lead \
                 and while the file is momentarily unreadable. Recorded only when \
                 LIVE_EDIT_SERVICE_TOKEN_FILE is set; absent from /metrics otherwise.",
            )
            .build();
        let service_token_expires_in_seconds = meter
            .i64_gauge("live_edit_service_token_expires_in_seconds")
            .with_description(
                "Seconds until the currently cached service-identity token expires, by role \
                 (valkey|storage-api|unspecified). Negative once the cached token has expired, so \
                 the series crosses zero exactly when that role's credential dies. Recorded by the \
                 ServiceTokenProvider polling task; absent from /metrics on replicas that \
                 construct no provider. Carries a derived duration only — never token bytes.",
            )
            .build();
        let service_token_refresh_headroom_seconds = meter
            .i64_gauge("live_edit_service_token_refresh_headroom_seconds")
            .with_description(
                "Seconds between the pickup of a replacement service-identity token and the \
                 expiry of the token it replaced, by role (valkey|storage-api|unspecified) — the \
                 window in which a periodic re-AUTH can prolong a live connection across the \
                 rotation. Negative when the replacement landed after the old token had already \
                 expired. Recorded once per observed rotation by the ServiceTokenProvider polling \
                 task. Carries a derived duration only — never token bytes.",
            )
            .build();

        // ── Pre-initialise label cells at zero ──────────────────────────────
        //
        // Recording `0` on a counter / up-down-counter materialises the series
        // so it renders at zero on the first scrape — operator dashboards keying
        // off `rate(... [5m])` then see a continuous derivative from boot rather
        // than a step-change on the first real event. (Histograms cannot be
        // pre-materialised on the OTel API without recording a sample, so their
        // families first appear on first observation.)
        let svc = |v: &str| KeyValue::new("service", v.to_string());
        let service_kv = KeyValue::new("service", Arc::<str>::from(service_label));

        presence_broadcast_dropped.add(0, &[svc(service_label)]);
        presence_resyncs.add(0, std::slice::from_ref(&service_kv));
        presence_keepalive_timeouts.add(0, std::slice::from_ref(&service_kv));
        for relay in PUBSUB_RELAYS {
            pubsub_relay_lagged.add(0, &[KeyValue::new("relay", *relay), service_kv.clone()]);
        }
        presence_lease_refresh_failures.add(0, std::slice::from_ref(&service_kv));
        presence_rejoins.add(0, std::slice::from_ref(&service_kv));
        for outcome in PRESENCE_SWEEP_LEASE_OUTCOMES {
            presence_sweep_lease_total
                .add(0, &[KeyValue::new("outcome", *outcome), service_kv.clone()]);
        }
        presence_stale_removed.add(0, std::slice::from_ref(&service_kv));
        subscriber_updates.add(
            0,
            &[
                KeyValue::new("endpoint", ENDPOINT_SUBSCRIBE_READ_LAYER),
                service_kv.clone(),
            ],
        );
        for event_type in PRESENCE_EVENT_TYPES {
            presence_events_total.add(
                0,
                &[KeyValue::new("event_type", *event_type), service_kv.clone()],
            );
        }
        for kind in ["permissions", "userinfo"] {
            auth_cache_hits.add(0, &[KeyValue::new("kind", kind)]);
            auth_cache_misses.add(0, &[KeyValue::new("kind", kind)]);
        }
        userinfo_fallbacks_total.add(0, &[]);
        health_probe_failures_total.add(0, &[KeyValue::new("backend", "valkey")]);
        for result in ["success", "error", "skipped"] {
            for kind in CHECKPOINT_WRITE_KINDS {
                checkpoint_writes_total.add(
                    0,
                    &[
                        KeyValue::new("result", result),
                        KeyValue::new("kind", *kind),
                    ],
                );
            }
        }
        for outcome in ["verified", "unverified"] {
            checkpoint_stable_cut_total
                .add(0, &[KeyValue::new("outcome", outcome), svc(service_label)]);
        }
        // Deliberately unlabeled by `service` — matches the increment site and
        // the query-pair sibling `checkpoint_writes_total`, which is likewise
        // service-free. Adding `svc(...)` here would both materialise a second,
        // permanently-zero series per outcome and break the pairing query.
        for outcome in CHECKPOINT_TAIL_OUTCOMES {
            checkpoint_tail_total.add(0, &[KeyValue::new("outcome", *outcome)]);
        }
        // The FULL `outcome × site` grid, inadmissible cells included — the two
        // `RestoreRebase`-only arms are seeded at `creation_base` too, on the
        // `checkpoint_writes_total` full-grid precedent (argued on the field
        // doc). Deliberately unlabeled by `service`, for the same reason as the
        // tail family directly above: it pairs with the service-free
        // `checkpoint_tail_total` and `checkpoint_writes_total`, and the
        // asymmetry would force a `sum without(service)` rewrite on every
        // pairing query while materialising a second, permanently-zero series
        // per cell.
        for outcome in CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES {
            for site in CHECKPOINT_BASE_BOOTSTRAP_SITES {
                checkpoint_base_bootstrap_total.add(
                    0,
                    &[
                        KeyValue::new("outcome", *outcome),
                        KeyValue::new("site", *site),
                    ],
                );
            }
        }
        for outcome in ["renewed", "lost", "refresh_error"] {
            checkpoint_lease_total.add(0, &[KeyValue::new("outcome", outcome), svc(service_label)]);
        }
        for outcome in CHECKPOINT_RECONCILE_OUTCOMES {
            checkpoint_reconcile_total
                .add(0, &[KeyValue::new("outcome", *outcome), svc(service_label)]);
        }
        for phase in CHECKPOINT_ADAPTER_CEILING_PHASES {
            checkpoint_adapter_ceiling_elapsed_total
                .add(0, &[KeyValue::new("phase", *phase), svc(service_label)]);
        }
        for (verb, outcome) in CHECKPOINT_ADAPTER_REDIRECT_CELLS {
            checkpoint_adapter_redirect_total.add(
                0,
                &[
                    KeyValue::new("verb", *verb),
                    KeyValue::new("outcome", *outcome),
                    svc(service_label),
                ],
            );
        }
        // Deliberately unlabeled by `service` — matches the increment site,
        // which is also reachable from the `LayerStore` fallback path that holds
        // no service label. Adding `svc(...)` here would materialise a second,
        // permanently-zero series per outcome.
        for outcome in OBJECT_ID_COUNTER_REPAIR_OUTCOMES {
            object_id_counter_repair_total.add(0, &[KeyValue::new("outcome", *outcome)]);
        }
        for result in ["restored", "not_recoverable", "error", "already_present"] {
            for scope in ["layer", "session"] {
                checkpoint_recovery_attempts_total.add(
                    0,
                    &[
                        KeyValue::new("result", result),
                        KeyValue::new("scope", scope),
                        svc(service_label),
                    ],
                );
            }
        }
        checkpoint_recovery_stream_refused_total.add(0, &[svc(service_label)]);
        checkpoint_restore_residue_wiped_total.add(0, &[svc(service_label)]);
        for stage in CHECKPOINT_RESTORE_FENCED_STAGES {
            checkpoint_restore_fenced_total
                .add(0, &[KeyValue::new("stage", *stage), svc(service_label)]);
        }
        for reason in CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS {
            checkpoint_recovery_stream_unresolved_total
                .add(0, &[KeyValue::new("reason", *reason), svc(service_label)]);
        }
        for outcome in CHECKPOINT_RESTORE_LINEAGE_OUTCOMES {
            checkpoint_restore_lineage_total
                .add(0, &[KeyValue::new("outcome", *outcome), svc(service_label)]);
        }
        for result in CHECKPOINT_SESSION_PUBLISH_SUPERSEDED_RESULTS {
            checkpoint_session_publish_superseded_total.add(0, &[KeyValue::new("result", *result)]);
        }
        for result in [
            "recovered",
            "adapter_miss",
            "adapter_transient",
            "adapter_permanent",
            "adapter_other",
        ] {
            checkpoint_ext_recovery_total.add(0, &[KeyValue::new("result", result)]);
        }
        for reason in EXT_MIRROR_DROP_REASONS {
            checkpoint_ext_mirror_dropped_total.add(0, &[KeyValue::new("reason", *reason)]);
        }
        for state in EXT_MIRROR_STATES {
            checkpoint_ext_mirror_bytes.add(0, &[KeyValue::new("state", *state)]);
        }
        for outcome in EXT_BACKUP_REPAIR_OUTCOMES {
            checkpoint_ext_backup_repair_total.add(0, &[KeyValue::new("outcome", *outcome)]);
        }
        for op in EXT_BACKUP_MARK_OPS {
            checkpoint_ext_backup_mark_failed_total.add(0, &[KeyValue::new("op", *op)]);
        }
        // 0 until `record_pod_memory_plan` sets it; stays 0 when the mirror is
        // not wired.
        checkpoint_ext_mirror_budget_bytes.record(0, &[]);
        // Zero-seeded: its baseline is zero and any increase pages, and an
        // alert on `increase()` misses the first tick of a series that does not
        // exist yet.
        gc_underflow_total.add(0, &[]);
        for outcome in GC_DECREMENT_OUTCOMES {
            gc_decrement_total.add(0, &[KeyValue::new("outcome", *outcome)]);
        }
        // Zero-seeded: outside a rollout its baseline is zero, and an alert on
        // `increase()` misses the first tick of a series that does not exist.
        for event in GC_OLDER_BUILD_EVENTS {
            gc_older_build_total.add(0, &[KeyValue::new("event", *event)]);
        }
        // Zero-seeded: an alert on `increase()` misses the first tick of a
        // series that does not exist yet.
        for outcome in GC_OOM_COLLECTION_OUTCOMES {
            gc_oom_collections_total.add(0, &[KeyValue::new("outcome", *outcome)]);
        }
        gc_oom_freed_bytes_total.add(0, &[]);
        for outcome in EXT_RELEASE_GATE_OUTCOMES {
            ext_release_gate_total.add(0, &[KeyValue::new("outcome", *outcome)]);
        }
        for outcome in CATCHUP_EXT_VALUE_MISS_OUTCOMES {
            for read in CATCHUP_EXT_VALUE_MISS_READS {
                catchup_ext_value_miss_total.add(
                    0,
                    &[
                        KeyValue::new("outcome", *outcome),
                        KeyValue::new("read", *read),
                    ],
                );
            }
        }
        for result in EXT_VALUE_VERIFY_RESULTS {
            ext_value_verify_total.add(0, &[KeyValue::new("result", *result)]);
        }
        // Spec SDD SR-2.1 / SR-2.3: both `UpdateLayer` commit counters are
        // seeded over their FULL label cross product, including cells no route
        // can emit (the WATCH path's conflict kinds under a structural path,
        // and the structural route's under `watch` or `fast`). That is a
        // deliberate exception to the SDD §17.1 rule that no cell is
        // pre-registered for a value this binary cannot emit: the per-path
        // baseline and the builds that add those paths compare like-for-like
        // series, so the vocabulary is fixed once.
        // The two histograms of the same family are not pre-registered — an
        // OTel histogram cannot be seeded without recording a sample.
        for path in UPDATE_LAYER_PATHS {
            for outcome in UPDATE_LAYER_RETRY_OUTCOMES {
                update_layer_retries_total.add(
                    0,
                    &[
                        KeyValue::new("path", *path),
                        KeyValue::new("outcome", *outcome),
                    ],
                );
            }
            for kind in UPDATE_LAYER_CONFLICT_KINDS {
                update_layer_conflicts_total.add(
                    0,
                    &[KeyValue::new("path", *path), KeyValue::new("kind", *kind)],
                );
            }
            // The structural paths and `watch` can suspend for a recharge;
            // `fast` cannot, and is seeded for the same like-for-like reason
            // as the two counters above.
            update_layer_recharges_total.add(0, &[KeyValue::new("path", *path)]);
        }
        for reason in UPDATE_LAYER_STRUCTURAL_FALLBACK_REASONS {
            update_layer_structural_fallbacks_total.add(0, &[KeyValue::new("reason", *reason)]);
        }
        // Both backends run the commit-token check, so both cells are
        // reachable and seeded beside the retry counter they explain.
        for backend in UPDATE_LAYER_BACKENDS {
            update_layer_commit_already_landed_total.add(0, &[KeyValue::new("backend", *backend)]);
        }
        // Every outcome is reachable on any store: a lookup finds an entry,
        // finds none, or fails to read it.
        for outcome in LOST_REPLY_RELEASE_LOOKUP_OUTCOMES {
            update_layer_lost_reply_release_lookup_total
                .add(0, &[KeyValue::new("outcome", *outcome)]);
        }
        // Both backends run the start-record check, so both cells are
        // reachable.
        for backend in UPDATE_LAYER_BACKENDS {
            update_layer_operation_overtaken_total.add(0, &[KeyValue::new("backend", *backend)]);
        }
        // Every (rpc, outcome) cell is reachable: both RPCs answer a matched
        // re-send and refuse a mismatched one.
        for rpc in OPERATION_ID_RPCS {
            for outcome in OPERATION_ID_OUTCOMES {
                operation_id_match_total.add(
                    0,
                    &[
                        KeyValue::new("rpc", *rpc),
                        KeyValue::new("outcome", *outcome),
                    ],
                );
            }
        }
        // Spec SDD SR-8.3: both intent results and the fence-timeout counter
        // exist from the first scrape. (The fence-wait histogram has no zero
        // observation to seed; it appears with the first fenced commit.)
        for result in UPDATE_LAYER_FENCE_INTENT_RESULTS {
            update_layer_fence_intent_total.add(0, &[KeyValue::new("result", *result)]);
        }
        update_layer_fence_timeout_total.add(0, &[]);
        // b.ph9: seed only the reachable (backend, site) cells. `create` is
        // served exclusively by `LayerStore::write_object`'s MULTI/EXEC, so
        // there is no `lua`/`create` pairing to seed — emitting one would
        // publish a series that can never move. Unlike
        // `keyspace_type_error_total` below, the sibling materialization entry
        // point `LayerStore::write_restored_object` adds no producer here: the
        // restore path queues no `XADD` at all (the per-object stream is
        // adjudicated separately by `checkpoint::recovery`), so it has no `1-0`
        // seed entry that could collide.
        //
        // The `lua`/`update` cell is seeded once per update-script `mode`
        // (`UPDATE_SCRIPT_MODES`): both the fast and the structural mode reach
        // the collision arm, through the one shared
        // `LayerStore::map_update_script_error`. The `txn` cells are raised by
        // no script mode and keep their two-label set.
        for mode in UPDATE_SCRIPT_MODES {
            stream_id_collision_total.add(
                0,
                &[
                    KeyValue::new("backend", UPDATE_LAYER_BACKEND_LUA),
                    KeyValue::new("site", "update"),
                    KeyValue::new("mode", *mode),
                ],
            );
        }
        for (backend, site) in [("txn", "update"), ("txn", "create")] {
            stream_id_collision_total.add(
                0,
                &[
                    KeyValue::new("backend", backend),
                    KeyValue::new("site", site),
                ],
            );
        }
        // Seed only the reachable (backend, site) cells. Both `update_layer`
        // backends refuse a wrong-typed mutated key before writing anything
        // (b.459), and b.zvd gave both materialization entry points
        // (`LayerStore::write_object` for `CreateLayer`,
        // `LayerStore::write_restored_object` for the DR restore) the same
        // pre-`MULTI` probe under `ProbeSite::Create`, so `txn`/`create` is
        // reachable too — from either of them, and from the two `CreateLayer`
        // operation-record producers (the handler's record read, and the
        // commit script's type check), into this one cell, with `entry_point`
        // carried as a log field rather than a second label value. There is no
        // `lua`/`create` pairing to seed: layer creation has no Lua commit
        // backend, exactly as for `stream_id_collision_total` above — the
        // commit-point script is counted under `txn`, the creation path it
        // belongs to.
        //
        // As for `stream_id_collision_total` above, the `lua`/`update` cell is
        // seeded once per update-script `mode`: the script's validation
        // pre-pass refuses in both modes.
        for mode in UPDATE_SCRIPT_MODES {
            keyspace_type_error_total.add(
                0,
                &[
                    KeyValue::new("backend", UPDATE_LAYER_BACKEND_LUA),
                    KeyValue::new("site", "update"),
                    KeyValue::new("mode", *mode),
                ],
            );
        }
        for (backend, site) in [("txn", "update"), ("txn", "create")] {
            keyspace_type_error_total.add(
                0,
                &[
                    KeyValue::new("backend", backend),
                    KeyValue::new("site", site),
                ],
            );
        }
        // b.atj: every `key_family` is reachable — a snapshot load reads the
        // header and all four indices unconditionally and reaches the per-entry
        // families for whatever the object holds, and `read_object_snapshot`
        // (atomic and unfenced) reads `obj:chs` (`children_index`) on every
        // marked object, and external-value clean-up's size read and
        // reference walk read `obj:ext` (`ext_index`), so any of them
        // can be the one that meets a foreign value. Unlike the `(backend, site)` grids
        // above there is no unreachable pairing to skip. Seeding the whole
        // vocabulary keeps an `increase(...) > 0` alert evaluating a present
        // series from t=0 rather than firing on absent-metric semantics.
        for family in SNAPSHOT_READ_KEY_FAMILIES {
            snapshot_read_type_error_total.add(0, &[KeyValue::new("key_family", *family)]);
        }
        // b.muu: the sibling counter's vocabulary is seeded whole for the same
        // alert-recipe reason, and unconditionally — `tls` is registered even
        // under a build whose fred has no `ErrorKind::Tls` to produce it, so the
        // published series set does not vary with the cargo features the binary
        // was compiled with. `backend_unavailable` is seeded like the rest,
        // though its alert is a sustained-rate rule rather than a page, and so
        // is `backend_out_of_memory`, whose `increase(...) > 0` page needs a
        // series present from t=0.
        for kind in SNAPSHOT_READ_REFUSAL_KINDS {
            snapshot_read_refused_total.add(0, &[KeyValue::new("kind", *kind)]);
        }
        // Seeded so an `increase(...) > 0` alert evaluates a present series
        // from t=0.
        snapshot_read_window_fallback_total.add(0, &[]);
        // All twelve store-write-refusal cells, so the out_of_memory page and
        // the sustained-rate rule over the other three have a series from t=0.
        for stage in STORE_WRITE_REFUSED_STAGES {
            for reason in STORE_WRITE_REFUSED_REASONS {
                store_write_refused_total.add(
                    0,
                    &[
                        KeyValue::new("stage", *stage),
                        KeyValue::new("reason", *reason),
                    ],
                );
            }
        }
        // Spec SDD SR-8.1 (and SDD §17.1's seeding rule): the cells the three
        // paths can emit, and no others — see `SNAPSHOT_READ_ATOMIC_CONSUMERS`
        // for why the `structural_update` and ext-GC consumers have no atomic
        // cell and `structural_cache` and `structural_cache_audit` no unfenced
        // one,
        // `SNAPSHOT_READ_UNFENCED_OUTCOMES` for why the unfenced path has no
        // `fence_broken` cell and only `structural_update` a `too_large` one
        // (skipped below for ext-GC, whose read carries no allowance), and
        // `SNAPSHOT_READ_FENCED_CONSUMERS` for which consumers run the fenced
        // read.
        for (path, consumers, outcomes) in [
            (
                SNAPSHOT_READ_PATH_ATOMIC,
                SNAPSHOT_READ_ATOMIC_CONSUMERS,
                SNAPSHOT_READ_ATOMIC_OUTCOMES,
            ),
            (
                SNAPSHOT_READ_PATH_UNFENCED,
                SNAPSHOT_READ_UNFENCED_CONSUMERS,
                SNAPSHOT_READ_UNFENCED_OUTCOMES,
            ),
            (
                SNAPSHOT_READ_PATH_FENCED,
                SNAPSHOT_READ_FENCED_CONSUMERS,
                SNAPSHOT_READ_FENCED_OUTCOMES,
            ),
        ] {
            for consumer in consumers {
                for outcome in outcomes {
                    if path == SNAPSHOT_READ_PATH_UNFENCED
                        && *outcome == SNAPSHOT_READ_OUTCOME_TOO_LARGE
                        && *consumer != SNAPSHOT_READ_CONSUMER_STRUCTURAL_UPDATE
                    {
                        continue;
                    }
                    snapshot_read_total.add(
                        0,
                        &[
                            KeyValue::new("consumer", *consumer),
                            KeyValue::new("path", path),
                            KeyValue::new("outcome", *outcome),
                        ],
                    );
                }
            }
        }
        // Structural-edits SDD SR-2.5: every cold-start route, so a dashboard
        // row and an `uncacheable > 0` rule (policy decision 14's trigger)
        // evaluate a present series from t=0.
        for route in STRUCTURAL_COLD_START_ROUTES {
            structural_cold_start_reads_total.add(0, &[KeyValue::new("route", *route)]);
        }
        // Structural-edits SDD SR-2.4: every lookup outcome and every reload
        // reason, so a per-reason rate and the `divergence_total > 0` alert
        // evaluate a present series from t=0.
        for outcome in STRUCTURAL_CACHE_LOOKUP_OUTCOMES {
            structural_cache_lookups_total.add(0, &[KeyValue::new("outcome", *outcome)]);
        }
        for reason in STRUCTURAL_CACHE_RELOAD_REASONS {
            structural_cache_reloads_total.add(0, &[KeyValue::new("reason", *reason)]);
        }
        // Every uncacheable reason, so a per-reason `increase(...) > 0` rule
        // evaluates a present series from t=0.
        for reason in STRUCTURAL_CACHE_UNCACHEABLE_REASONS {
            structural_cache_uncacheable_total.add(0, &[KeyValue::new("reason", *reason)]);
        }
        structural_cache_divergence_total.add(0, &[]);
        // SR-2.4: every audit outcome, so a `diverged > 0` alert evaluates a
        // present series from t=0.
        for outcome in STRUCTURAL_CACHE_AUDIT_OUTCOMES {
            structural_cache_audit_total.add(0, &[KeyValue::new("outcome", *outcome)]);
        }
        for result in STRUCTURAL_CACHE_AUDIT_FENCED_RESULTS {
            structural_cache_audit_fenced_total.add(0, &[KeyValue::new("result", *result)]);
        }
        // Spec SDD SR-8.3: every acquire reply a fenced reader can be given,
        // now that a production read (a catch-up flight's) takes the fence.
        for result in SNAPSHOT_FENCE_ACQUIRE_RESULTS {
            snapshot_fence_acquire_total.add(0, &[KeyValue::new("result", *result)]);
        }
        // Every pool the service opens: reconnects, router restarts and
        // unresponsive closes per pool, and the one service-owned
        // `valkey_errors_total` kind per pool.
        // The fred-named kinds on that counter stay unseeded (their vocabulary
        // is fred's).
        for pool in VALKEY_POOLS {
            valkey_reconnects_total.add(0, &[KeyValue::new("pool", *pool)]);
            valkey_router_restarts_total.add(0, &[KeyValue::new("pool", *pool)]);
            for verdict in VALKEY_UNRESPONSIVE_VERDICTS {
                valkey_unresponsive_total.add(
                    0,
                    &[
                        KeyValue::new("pool", *pool),
                        KeyValue::new("verdict", *verdict),
                    ],
                );
            }
            valkey_errors_total.add(
                0,
                &[
                    KeyValue::new("pool", *pool),
                    KeyValue::new("kind", VALKEY_ERROR_KIND_COMMAND_TIMEOUT),
                ],
            );
        }
        for pool in VALKEY_REPLACEABLE_POOLS {
            for outcome in VALKEY_MEMBER_REPLACEMENT_OUTCOMES {
                valkey_member_replacements_total.add(
                    0,
                    &[
                        KeyValue::new("pool", *pool),
                        KeyValue::new("outcome", *outcome),
                    ],
                );
            }
        }
        for pool in [
            VALKEY_POOL_TXN,
            VALKEY_POOL_SUBSCRIBER_POOL,
            VALKEY_POOL_MAINTENANCE,
            VALKEY_POOL_SNAPSHOT,
        ] {
            valkey_pool_in_use.add(0, &[KeyValue::new("pool", pool)]);
        }
        // b.f4f: same pre-initialisation posture as `valkey_pool_in_use` above —
        // an operator dashboard that plots waiter depth must see the zero from
        // the first scrape, not an absent series that only appears once the pool
        // has already saturated. Only the two queues `CHECKOUT_POOLS` names: the
        // `txn` `ExclusivePool`'s permit queue and the `SubscribeReadLayer`
        // admission queue labelled `subscriber_pool`. The multiplexed
        // maintenance pool has no queue to wait in.
        for pool in CHECKOUT_POOLS {
            valkey_pool_waiters.add(0, &[KeyValue::new("pool", *pool)]);
        }
        // Every value the shipped binary can emit, and no others — a
        // pre-registered value that can never tick becomes a permanently-zero
        // series dashboards bind to as real (b.f4f).
        for result in ADMISSION_RESULTS {
            subscriber_admission_total.add(0, &[KeyValue::new("result", *result)]);
        }
        // Upload admission: every (rpc, result) cell, every overflow-run cell,
        // both phases, and the unlabelled series, so an alert on a refusal
        // rate, an overflow rate or a reserved-bytes ratio has a series from
        // the first scrape. The two wait histograms are not pre-registered
        // (SDD §17.1); the gate records the configured budget over the seeded
        // zero when it is built.
        for rpc in UPLOAD_RPCS {
            for result in UPLOAD_ADMISSION_RESULTS {
                upload_admission_total.add(
                    0,
                    &[KeyValue::new("rpc", *rpc), KeyValue::new("result", *result)],
                );
            }
            upload_overflow_runs_total.add(0, &[KeyValue::new("rpc", *rpc)]);
        }
        for phase in UPLOAD_PHASES {
            uploads_in_flight.add(0, &[KeyValue::new("phase", *phase)]);
        }
        upload_reserved_bytes.add(0, &[]);
        upload_admission_waiters.add(0, &[]);
        upload_budget_bytes.record(0, &[]);
        upload_overflow_bytes.record(0, &[]);
        upload_overflow_waiters.add(0, &[]);
        // The chain gate: both probe results and the two waiter series, so a
        // probe-error rate or a waiting-bytes ratio has a series from the
        // first scrape. The two histograms are not pre-registered (SDD §17.1).
        for result in CHAIN_PROBE_RESULTS {
            update_layer_chain_probes_total.add(0, &[KeyValue::new("result", *result)]);
        }
        update_layer_chain_waiters.add(0, &[]);
        update_layer_chain_waiting_bytes.add(0, &[]);
        for reason in STREAM_EXIT_REASONS {
            subscriber_stream_exits_total.add(0, &[KeyValue::new("reason", *reason)]);
        }
        // Both phases, so an alert on the `live` cell's rate has a series to
        // evaluate before the first stall. The companion histograms are not
        // pre-registered (SDD §17.1).
        for phase in SEND_PHASES {
            subscriber_send_stalls_total.add(0, &[KeyValue::new("phase", *phase)]);
        }
        // b.ji8: pre-init both teardown sites so an alert on
        // `increase(...) > 0` has a series to evaluate before the first
        // failure rather than firing on absent-metric semantics. That alert
        // shape belongs on `{site="drop"}` only — `{site="explicit"}` means
        // "Drop will retry this teardown", not "a connection was released
        // carrying stale WATCH state" (see the field's doc comment).
        for site in [UNWATCH_SITE_EXPLICIT, UNWATCH_SITE_DROP] {
            storage_watch_unwatch_failures_total.add(0, &[KeyValue::new("site", site)]);
        }
        // b.bhb: same reasoning — the `committed_unwatched` alert must have a
        // series before the first void. Registered as the full cross product
        // (see `WATCH_VOIDED_PATHS`), including the unreachable `compensated`
        // cells — three of them, one per path that cannot compensate
        // (`update_layer`, `restore_residue_wipe`, `restore_write_object`;
        // only `create_session` can) — so the alert expression is identical
        // across all four paths.
        for path in WATCH_VOIDED_PATHS {
            for disposition in WATCH_VOIDED_DISPOSITIONS {
                storage_watch_voided_total.add(
                    0,
                    &[
                        KeyValue::new("path", *path),
                        KeyValue::new("disposition", *disposition),
                    ],
                );
            }
        }
        // b.7td: same reasoning one metric over — an alert on
        // `increase(...) > 0` needs a series before the first dirty checkout.
        // Unlike the UNWATCH sites there is no "context only" cell here: every
        // reason means the checkout probe returned a dirty verdict (whether or
        // not the streak gate / ineffective-repair breaker admitted a recycle).
        for reason in TXN_DIRTY_REASONS {
            valkey_txn_dirty_checkouts_total.add(0, &[KeyValue::new("reason", *reason)]);
        }
        // b.7td round 3: the repair-ineffective breaker can only ever report a
        // state-proof reason, so its label set is the strict subset above — not
        // the full `TXN_DIRTY_REASONS`, which would advertise three cells nothing
        // can fill. Same zero-seed reasoning: this is the series an operator
        // alerts on to learn that recycling has stopped working, and it must
        // exist before that happens.
        for reason in TXN_STATE_PROOF_REASONS {
            valkey_txn_repair_ineffective_total.add(0, &[KeyValue::new("reason", *reason)]);
        }
        // b.qpf: the release counters are read as a pair — a rate of releases
        // against a rate of failed enqueues — so both need a series before the
        // first event, and the reason cells must all exist so an operator can
        // tell "no rewrites happened" from "the rewrite cell is missing".
        for reason in EXT_REF_RELEASE_REASONS {
            ext_ref_released_total.add(0, &[KeyValue::new("reason", *reason)]);
        }
        ext_ref_release_enqueue_failed_total.add(0, &[]);
        // b.yda: the deferred / recovered / cancelled triple is read as a ratio
        // in exactly the same way, and the whole point of it is that a
        // `deferred` with no matching `recovered` or `cancelled` is the
        // alertable shape — which needs all three series to exist before the
        // first deferral, not after it. `cancelled` especially: it is the rarest
        // of the three and the one an operator is least likely to have seen, so
        // an absent series is exactly what would read as "this never happens".
        ext_ref_release_deferred_total.add(0, &[]);
        ext_ref_release_recovered_total.add(0, &[]);
        ext_ref_release_cancelled_total.add(0, &[]);
        // b.yuo Part R: the fourth arm of the same identity, seeded for the same
        // reason.
        ext_ref_release_already_enqueued_total.add(0, &[]);
        // b.yda: every drop site, so "no corruption at this site" is a visible
        // zero rather than an absent series. These are expected-zero cells, which
        // is exactly the case where a missing series and a healthy one look the
        // same on a dashboard.
        for site in EXT_REF_UNDECODABLE_SITES {
            ext_ref_undecodable_total.add(0, &[KeyValue::new("site", *site)]);
        }
        // b.yda: the reconciler's outcome vocabulary. `enumeration_failed` and
        // `deficit_observed` are the alertable cells and are the rarest, so they
        // are precisely the ones an alert rule must have been watching before
        // the first occurrence.
        for outcome in EXT_RECONCILE_OUTCOMES {
            ext_reconcile_total.add(0, &[KeyValue::new("outcome", *outcome)]);
        }
        // The per-object reference-index recount: every (site, outcome) and
        // (site, direction) cell, so `direction="under"` and `failed` are
        // visible zeros before their first occurrence.
        for site in EXT_INDEX_RECOUNT_SITES {
            for outcome in EXT_INDEX_RECOUNT_OUTCOMES {
                ext_index_recount_total.add(
                    0,
                    &[
                        KeyValue::new("site", *site),
                        KeyValue::new("outcome", *outcome),
                    ],
                );
            }
            for direction in EXT_INDEX_DRIFT_DIRECTIONS {
                ext_index_drift_total.add(
                    0,
                    &[
                        KeyValue::new("site", *site),
                        KeyValue::new("direction", *direction),
                    ],
                );
            }
        }
        for site in EXT_INDEX_DRIFT_ONLY_SITES {
            for direction in EXT_INDEX_DRIFT_DIRECTIONS {
                ext_index_drift_total.add(
                    0,
                    &[
                        KeyValue::new("site", *site),
                        KeyValue::new("direction", *direction),
                    ],
                );
            }
        }
        // The signed-zero merge: every outcome, so `failed` and `undecodable`
        // are visible zeros before their first occurrence.
        for outcome in TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOMES {
            time_sample_signed_zero_merge_total.add(0, &[KeyValue::new("outcome", *outcome)]);
        }
        // Pre-initialised so the series exist before the first (rare) event.
        for phase in CONSUMER_FORCED_RECONNECT_PHASES {
            valkey_consumer_forced_reconnects_total.add(0, &[KeyValue::new("phase", *phase)]);
        }
        for phase in CONSUMER_PERMANENT_EXIT_PHASES {
            valkey_consumer_permanent_exits_total.add(0, &[KeyValue::new("phase", *phase)]);
        }
        // Unlabeled — one cell, pre-initialized so the series exists before the
        // first interrupt (b.9yf), which may never come (b.19i).
        valkey_consumer_unblocked_retries_total.add(0, &[]);
        stream_mux_subscriptions.add(0, &[]);
        subscriber_admitted.add(0, &[]);
        subscribe_catchups_pending.add(0, &[]);
        subscribe_catchups_inflight.add(0, &[]);
        subscribe_catchups_waiting.add(0, &[]);
        // Spec SDD SR-8.5: both permit classes, `fenced` included, so a
        // dashboard sees the zero before the first read (and before the fenced
        // read ships).
        for class in CATCHUP_SNAPSHOT_READ_CLASSES {
            catchup_snapshot_reads_in_flight.add(0, &[KeyValue::new("class", *class)]);
        }
        // Spec SDD SR-8.5: the memory budget's gauges, the builds gauge and
        // every admission result, so dashboards and alerts see the zeros
        // before the first catch-up that reserves memory.
        catchup_build_budget_reserved_bytes.add(0, &[]);
        catchup_build_budget_overcommit_bytes.add(0, &[]);
        catchup_builds_in_progress.add(0, &[]);
        // The retirement allowance's gauges, so its zeros are there before the
        // first clean-up reserves.
        retirement_budget_reserved_bytes.add(0, &[]);
        retirement_budget_overcommit_bytes.add(0, &[]);
        for result in CATCHUP_BUILD_ADMISSION_RESULTS {
            catchup_build_admission_total.add(0, &[KeyValue::new("result", *result)]);
        }
        // Spec SDD SR-8.4: both flight roles, so the read hit-rate recipe
        // divides present series from the first scrape.
        for role in CATCHUP_SNAPSHOT_FLIGHT_ROLES {
            catchup_snapshot_flight_total.add(0, &[KeyValue::new("role", *role)]);
        }
        // Spec SDD SR-8.6: the stable-cut outcome grid, so an alert on
        // `increase({outcome="unverified"}[...]) > 0` evaluates a present
        // series from the first scrape.
        for outcome in SUBSCRIBE_STABLE_CUT_OUTCOMES {
            for regime in SUBSCRIBE_STABLE_CUT_REGIMES {
                subscribe_stable_cut_total.add(
                    0,
                    &[
                        KeyValue::new("outcome", *outcome),
                        svc(service_label),
                        KeyValue::new("regime", *regime),
                    ],
                );
            }
        }
        stream_mux_streams.add(0, &[]);
        stream_mux_xreads_total.add(0, &[]);
        for outcome in WAKE_OUTCOMES {
            stream_mux_wakes_total.add(0, &[KeyValue::new("outcome", *outcome)]);
        }
        for result in KEEPALIVE_RESULTS {
            stream_mux_keepalives_total.add(0, &[KeyValue::new("result", *result)]);
        }
        for result in REAUTH_RESULTS {
            stream_mux_reauths_total.add(0, &[KeyValue::new("result", *result)]);
        }
        for cause in READER_CONNECT_CAUSES {
            for result in READER_CONNECT_RESULTS {
                stream_mux_reader_connects_total.add(
                    0,
                    &[
                        KeyValue::new("cause", *cause),
                        KeyValue::new("result", *result),
                    ],
                );
            }
        }
        stream_mux_readers_connected.add(0, &[]);
        stream_mux_bind_refusals_total.add(0, &[]);
        stream_mux_poison_keys_total.add(0, &[]);
        for bound in OVERFLOW_BOUNDS {
            stream_mux_overflow_kills_total.add(0, &[KeyValue::new("bound", *bound)]);
        }
        for result in HEAD_CHECK_RESULTS {
            stream_mux_head_checks_total.add(0, &[KeyValue::new("result", *result)]);
        }
        // Pre-init every (endpoint, service, grpc_code) call-counter cell and
        // (endpoint, service) inflight-gauge cell (b.avd item 4/5).
        let endpoint_label_iter = crate::grpc::metrics_layer::KNOWN_ENDPOINTS
            .iter()
            .copied()
            .chain(std::iter::once(crate::grpc::metrics_layer::OTHER_ENDPOINT));
        for endpoint in endpoint_label_iter {
            for code in GRPC_CODES {
                endpoint_calls.add(
                    0,
                    &[
                        KeyValue::new("endpoint", endpoint),
                        svc(service_label),
                        KeyValue::new("grpc_code", *code),
                    ],
                );
            }
            inflight_rpcs.add(
                0,
                &[KeyValue::new("endpoint", endpoint), svc(service_label)],
            );
        }

        // b.qhk: how full the backing store is. An observable gauge over a
        // cell the store-memory reads set, never zero-seeded: a collection
        // that finds the cell empty observes nothing, and the delta
        // temporality the SDK gives observable gauges then drops the series,
        // so "no figure" reads as absent, never as an empty store.
        let store_memory_used_ratio: Arc<std::sync::Mutex<Option<f64>>> =
            Arc::new(std::sync::Mutex::new(None));
        {
            let cell = Arc::clone(&store_memory_used_ratio);
            let _gauge = meter
                .f64_observable_gauge("live_edit_store_memory_used_ratio")
                .with_description(
                    "How full the backing store is, re-read every 60 s: normally 0 to 1, and \
                     above 1 when the store is past its limit. On a store whose primaries \
                     report maxmemory, the highest used_memory / maxmemory across primaries. On \
                     a store none of whose primaries reports maxmemory (Azure Managed Redis) \
                     with LIVE_EDIT_VALKEY_MEMORY_BYTES set, the whole-database used_memory / \
                     copies / (LIVE_EDIT_VALKEY_MEMORY_BYTES x 0.8), where copies is 2 when the \
                     store's replica acknowledgement count (cmdstat_replconf) rose since the \
                     previous read and 1 when it did not move, so there the series appears at \
                     the second read. Absent when neither applies, the latest read failed, or \
                     the copies are not known",
                )
                .with_callback(move |observer| {
                    let ratio = *cell
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if let Some(ratio) = ratio {
                        observer.observe(ratio, &[]);
                    }
                })
                .build();
        }

        Ok(Metrics {
            _provider: provider,
            otlp_export_reporter,
            prometheus_exporter,
            meter,
            service_kv,
            endpoint_calls,
            endpoint_duration_us,
            inflight_rpcs,
            subscriber_updates,
            diff_case_total,
            diff_build_duration_seconds,
            subscribe_regime_total,
            subscribe_catch_up_duration_seconds,
            join_latency_seconds,
            subscribe_catch_up_delivered_bytes,
            subscribe_stable_cut_total,
            presence_events_total,
            presence_broadcast_dropped,
            presence_resyncs,
            presence_keepalive_timeouts,
            pubsub_relay_lagged,
            presence_lease_refresh_failures,
            presence_rejoins,
            presence_sweep_duration_seconds,
            presence_sweep_lease_total,
            presence_stale_removed,
            presence_stale_removal_delay_seconds,
            presence_sweep_oldest_stale_age_seconds,
            auth_cache_hits,
            auth_cache_misses,
            auth_call_duration_seconds,
            auth_errors,
            userinfo_fallbacks_total,
            update_layer_retries_total,
            update_layer_commit_already_landed_total,
            update_layer_lost_reply_release_lookup_total,
            operation_id_match_total,
            update_layer_operation_overtaken_total,
            update_layer_attempts_histogram,
            update_layer_commit_duration_seconds,
            update_layer_relay_bytes,
            store_write_refused_total,
            update_layer_conflicts_total,
            update_layer_recharges_total,
            upload_object_state_bytes,
            update_layer_structural_fallbacks_total,
            update_layer_structural_commit_seconds,
            update_layer_fence_wait_seconds,
            update_layer_fence_intent_total,
            update_layer_fence_timeout_total,
            stream_id_collision_total,
            keyspace_type_error_total,
            snapshot_read_type_error_total,
            snapshot_read_refused_total,
            snapshot_read_window_fallback_total,
            snapshot_read_total,
            snapshot_read_duration_seconds,
            snapshot_read_reply_bytes,
            snapshot_read_entries,
            structural_cold_start_reads_total,
            structural_cold_start_read_duration_seconds,
            structural_cache_lookups_total,
            structural_cache_reloads_total,
            structural_cache_uncacheable_total,
            structural_cache_replayed_entries,
            structural_cache_repair_bytes,
            structural_cache_divergence_total,
            structural_cache_audit_total,
            structural_cache_audit_fenced_total,
            snapshot_fence_hold_seconds,
            snapshot_fence_acquire_total,
            snapshot_fence_reader_wait_seconds,
            valkey_reconnects_total,
            valkey_router_restarts_total,
            valkey_unresponsive_total,
            valkey_member_replacements_total,
            valkey_errors_total,
            storage_watch_unwatch_failures_total,
            storage_watch_voided_total,
            valkey_txn_dirty_checkouts_total,
            valkey_txn_repair_ineffective_total,
            valkey_consumer_forced_reconnects_total,
            valkey_consumer_unblocked_retries_total,
            valkey_consumer_permanent_exits_total,
            stream_mux_subscriptions,
            stream_mux_streams,
            stream_mux_xreads_total,
            stream_mux_wakes_total,
            stream_mux_keepalives_total,
            stream_mux_reauths_total,
            stream_mux_reader_connects_total,
            stream_mux_readers_connected,
            stream_mux_bind_refusals_total,
            stream_mux_poison_keys_total,
            stream_mux_overflow_kills_total,
            stream_mux_head_checks_total,
            stream_mux_xread_reply_bytes,
            valkey_pool_in_use,
            valkey_pool_waiters,
            valkey_pool_checkout_seconds,
            subscriber_admission_total,
            subscriber_admission_wait_seconds,
            upload_admission_total,
            upload_admission_wait_seconds,
            upload_reserved_bytes,
            uploads_in_flight,
            upload_admission_waiters,
            upload_budget_bytes,
            pod_memory_limit_bytes,
            catchup_build_budget_bytes,
            retirement_budget_bytes,
            structural_cache_budget_bytes,
            subscriber_max_subscriptions,
            upload_overflow_bytes,
            upload_overflow_waiters,
            upload_overflow_runs_total,
            upload_processing_wait_seconds,
            update_layer_chain_wait_seconds,
            update_layer_chain_probes_total,
            update_layer_chain_waiters,
            update_layer_chain_waiting_bytes,
            update_layer_chain_detect_lag_seconds,
            subscribe_catchups_inflight,
            subscribe_catchups_waiting,
            subscribe_catchup_wait_seconds,
            catchup_snapshot_reads_in_flight,
            catchup_snapshot_read_wait_seconds,
            catchup_build_budget_reserved_bytes,
            catchup_build_budget_overcommit_bytes,
            catchup_build_wait_seconds,
            catchup_build_admission_total,
            retirement_budget_reserved_bytes,
            retirement_budget_overcommit_bytes,
            retirement_wait_seconds,
            catchup_builds_in_progress,
            catchup_snapshot_flight_total,
            subscriber_admitted,
            subscribe_catchups_pending,
            scaling_counters: ScalingCounters::default(),
            subscriber_stream_exits_total,
            subscriber_delivery_lag_seconds,
            subscriber_send_stalls_total,
            subscriber_send_wait_seconds,
            subscriber_backlog_entries,
            subscriber_backlog_bytes,
            background_task_panics_total,
            gc_sweep_batch_errors_total,
            health_probe_failures_total,
            checkpoint_writes_total,
            checkpoint_pending_age_seconds,
            checkpoint_oldest_pending_age_seconds,
            checkpoint_layers_never_written,
            checkpoint_counter_seed_secs,
            checkpoint_stable_cut_total,
            checkpoint_tail_total,
            checkpoint_base_bootstrap_total,
            checkpoint_lease_total,
            checkpoint_reconcile_total,
            checkpoint_adapter_ceiling_elapsed_total,
            checkpoint_adapter_redirect_total,
            checkpoint_ext_tombstone_backlog,
            checkpoint_dead_object_backlog,
            delta_create_section_skipped_missing_parent_total,
            checkpoint_recovery_attempts_total,
            checkpoint_recovery_stream_refused_total,
            checkpoint_restore_residue_wiped_total,
            checkpoint_restore_fenced_total,
            checkpoint_recovery_stream_unresolved_total,
            checkpoint_restore_lineage_total,
            checkpoint_session_publish_superseded_total,
            gc_underflow_total,
            gc_decrement_total,
            gc_older_build_total,
            gc_oom_collections_total,
            gc_oom_freed_bytes_total,
            ext_release_pending_bytes,
            ext_release_pending_count,
            ext_release_oldest_age_seconds,
            ext_release_settled_count,
            ext_read_holds,
            ext_release_gate_total,
            ext_release_budget_bytes,
            store_memory_used_ratio,
            catchup_ext_value_miss_total,
            ext_ref_released_total,
            ext_ref_release_enqueue_failed_total,
            ext_ref_release_deferred_total,
            ext_ref_release_recovered_total,
            ext_ref_release_already_enqueued_total,
            ext_ref_release_cancelled_total,
            ext_ref_undecodable_total,
            ext_meta_total,
            ext_referenced_hashes_total,
            ext_ref_surplus_total,
            ext_ref_deficit_total,
            ext_retire_retry_backlog,
            ext_reconcile_total,
            ext_index_recount_total,
            ext_index_drift_total,
            time_sample_signed_zero_merge_total,
            object_id_counter_repair_total,
            service_token_parse_errors_total,
            checkpoint_ext_mirror_dropped_total,
            checkpoint_ext_mirror_bytes,
            checkpoint_ext_mirror_budget_bytes,
            checkpoint_ext_mirror_errors_total,
            checkpoint_ext_backup_repair_total,
            checkpoint_ext_backup_owed,
            checkpoint_ext_backup_mark_failed_total,
            checkpoint_ext_recovery_total,
            ext_value_verify_total,
            service_token_available,
            service_token_expires_in_seconds,
            service_token_refresh_headroom_seconds,
        })
    }

    /// The OTLP push exporter's failure accounting, when the push arm is
    /// attached: its total rises with every failed push.
    #[cfg(any(test, feature = "test-support"))]
    pub fn otlp_export_reporter(&self) -> Option<&Arc<MetricExportReporter>> {
        self.otlp_export_reporter.as_ref()
    }

    /// Clone the always-attached Prometheus scrape exporter for the `/metrics`
    /// HTTP server. The clone shares the underlying `MetricReader`, so
    /// `PrometheusExporter::export` on the returned handle renders a live read
    /// of the same pipeline these instruments feed.
    pub fn prometheus_exporter(&self) -> PrometheusExporter {
        self.prometheus_exporter.clone()
    }

    /// Register the `live_edit_storage_watch_unwatch_inflight` observable gauge
    /// (b.k1r H-stor-2). The `source` closure is invoked at each metric
    /// collection (every `/metrics` scrape and every OTLP push tick) so the
    /// gauge always reflects the live `WatchGuard::Drop`-spawned UNWATCH
    /// tracker count rather than a stale last-set value.
    ///
    /// Called once at startup from `main.rs` with a closure that reads
    /// `ValkeyClient::watch_unwatch_inflight()`. On replicas / tests where it
    /// is never called the family is simply absent from `/metrics`.
    pub fn register_storage_watch_unwatch_inflight_gauge<F>(&self, source: F)
    where
        F: Fn() -> i64 + Send + Sync + 'static,
    {
        let _gauge = self
            .meter
            .i64_observable_gauge("live_edit_storage_watch_unwatch_inflight")
            .with_description(
                "Number of WatchGuard::Drop-spawned UNWATCH cleanup tasks currently in flight on \
                 the txn pool",
            )
            .with_callback(move |observer| observer.observe(source(), &[]))
            .build();
    }

    /// Register the per-replica load gauges. Each `source` closure is invoked
    /// at every metric collection (each `/metrics` scrape and each OTLP push
    /// tick), so the gauges always reflect live in-process state:
    ///
    /// | Metric | Source | Meaning |
    /// |---|---|---|
    /// | `live_active_sessions{service}` | `LocalPresenceRouter::session_count()` | sessions with ≥ 1 local JoinSession connection on this replica, or whose last connection's presence-channel `UNSUBSCRIBE` is still in flight — each holds one presence fan-out queue, sized by its undelivered events (SDD §13.4) |
    /// | `live_active_connections{service}` | `LocalPresenceRouter::connection_count()` | JoinSession connections currently open on this replica |
    /// | `live_edit_update_layer_admission_entries{service}` | `LayerStore::semaphore_count()` | per-object `update_layer` admission entries currently held in memory (SDD §7.5 *Semaphore-map lifecycle*) |
    ///
    /// Observable gauges rather than paired inc/dec counters on purpose: an
    /// entry that outlives its owner (a JoinSession task that unwound past
    /// its cleanup, an object that was never retired) then shows up as a
    /// gauge that does not return to zero once load stops, instead of being
    /// hidden behind a decrement that never ran. Called once at startup from
    /// `main.rs`; on replicas / tests that never call it the three families
    /// are simply absent from `/metrics`.
    pub fn register_replica_load_gauges<S, C, A>(
        &self,
        service: &str,
        sessions: S,
        connections: C,
        admission_entries: A,
    ) where
        S: Fn() -> i64 + Send + Sync + 'static,
        C: Fn() -> i64 + Send + Sync + 'static,
        A: Fn() -> i64 + Send + Sync + 'static,
    {
        let labels = [KeyValue::new("service", service.to_string())];
        let register = |name: &'static str,
                        description: &'static str,
                        source: Box<dyn Fn() -> i64 + Send + Sync>| {
            let labels = labels.clone();
            let _gauge = self
                .meter
                .i64_observable_gauge(name)
                .with_description(description)
                .with_callback(move |observer| observer.observe(source(), &labels))
                .build();
        };
        register(
            "live_active_sessions",
            "Sessions with at least one JoinSession connection open on this replica, or whose \
             last connection's presence-channel UNSUBSCRIBE is still in flight; each holds one \
             presence fan-out queue whose memory follows its undelivered events, not \
             LIVE_EDIT_PRESENCE_BROADCAST_CAPACITY",
            Box::new(sessions),
        );
        register(
            "live_active_connections",
            "JoinSession connections currently open on this replica",
            Box::new(connections),
        );
        register(
            "live_edit_update_layer_admission_entries",
            "Per-object UpdateLayer admission entries currently held in memory on this \
             replica; grows with distinct objects written and shrinks only when an object is \
             deleted or overwritten",
            Box::new(admission_entries),
        );
    }

    /// Register the structural-planning cache gauges (structural-edits spec
    /// SR-2.4), each labelled `service`. Each `source` closure is invoked at
    /// every metric collection:
    ///
    /// | Metric | Source | Meaning |
    /// |---|---|---|
    /// | `live_edit_structural_cache_entries{service}` | `StructuralCache::entry_count()` | objects with an entry in this replica's structural write-planning cache; a checked-out entry whose key stays resident still counts |
    /// | `live_edit_structural_cache_weight_bytes{service}` | `StructuralCache::weight_bytes()` | the estimated weight of those entries, state and plan indexes, within the cache's resident budget, its budget (5 % of the pod memory limit) × (1 + φ), φ = 0.21, `live_edit_structural_cache_budget_bytes` (`storage::structural_cache::resident_budget_bytes`), whatever an entry's indexes weigh: a heavier-indexed entry is charged in full and other entries are evicted sooner |
    ///
    /// Called once at startup from `main.rs`; on replicas / tests that never
    /// call it the two families are absent from `/metrics`.
    pub fn register_structural_cache_gauges<E, W>(&self, service: &str, entries: E, weight_bytes: W)
    where
        E: Fn() -> u64 + Send + Sync + 'static,
        W: Fn() -> u64 + Send + Sync + 'static,
    {
        let labels = [KeyValue::new("service", service.to_string())];
        let register = |name: &'static str,
                        description: &'static str,
                        source: Box<dyn Fn() -> u64 + Send + Sync>| {
            let labels = labels.clone();
            let _gauge = self
                .meter
                .u64_observable_gauge(name)
                .with_description(description)
                .with_callback(move |observer| observer.observe(source(), &labels))
                .build();
        };
        register(
            "live_edit_structural_cache_entries",
            "Objects with an entry in this replica's structural write-planning cache",
            Box::new(entries),
        );
        register(
            "live_edit_structural_cache_weight_bytes",
            "Estimated in-memory weight, in bytes, of the entries in this replica's structural \
             write-planning cache, their plan indexes included. An entry holds the layer's \
             structure in a compact form: its sections and children lists, and per field and \
             time sample only its key and external-value hash, never the value. \
             The cache's budget, derived from the pod memory limit, budgets the structures; \
             with their indexes the entries stay within 1.21 times it \
             (live_edit_structural_cache_budget_bytes), whatever an entry's indexes weigh: a \
             heavier-indexed entry is charged in full and other entries are dropped sooner",
            Box::new(weight_bytes),
        );
    }

    /// Register the `process_resident_memory_bytes` observable gauge (b.u5k):
    /// the process's resident set size in bytes. The `source` closure is
    /// invoked at each metric collection (every `/metrics` scrape and every
    /// OTLP push tick); a `None` from it emits no sample for that collection
    /// rather than a misleading `0`.
    ///
    /// Named after the Prometheus standard process metric and unlabelled like
    /// it, the one documented exception to this module's prefix convention
    /// (see the module header). Called once at startup from `main.rs` with
    /// [`crate::observability::process_memory::resident_memory_bytes`]; on
    /// replicas / tests that never call it the family is absent.
    pub fn register_process_memory_gauge<F>(&self, source: F)
    where
        F: Fn() -> Option<u64> + Send + Sync + 'static,
    {
        let _gauge = self
            .meter
            .u64_observable_gauge("process_resident_memory_bytes")
            .with_description("Resident memory size in bytes.")
            .with_callback(move |observer| {
                if let Some(bytes) = source() {
                    observer.observe(bytes, &[]);
                }
            })
            .build();
    }

    /// Register the jemalloc allocator gauges (b.u5k), each labelled `service`:
    ///
    /// | Metric | Source | Meaning |
    /// |---|---|---|
    /// | `live_edit_jemalloc_allocated_bytes{service}` | `stats.allocated` | bytes the application currently holds allocated through jemalloc |
    /// | `live_edit_jemalloc_resident_bytes{service}` | `stats.resident` | an upper bound, not a precise count, on bytes in resident pages jemalloc has mapped (metadata, live allocations and unpurged dirty pages); jemalloc counts pages that are demand-zeroed but never touched, so it can read above process RSS |
    ///
    /// **Memory returned to the OS is judged from `process_resident_memory_bytes`,
    /// not from this pair.** The resident gauge is jemalloc's own accounting: it
    /// can sit above the process RSS at idle, and it cannot see pages the kernel
    /// makes resident behind jemalloc's back — with transparent huge pages
    /// `always` and jemalloc's default `thp` option, khugepaged re-fills purged
    /// ranges and RSS stays high while this gauge still looks healthy. What the
    /// pair does show is how the heap splits between live allocations and
    /// jemalloc's own overhead and unpurged pages. Each `source` closure is
    /// invoked at each metric collection; a `None` emits no sample rather than
    /// a misleading `0`. Called once at startup from `main.rs`, and only by a
    /// binary whose global allocator is jemalloc — elsewhere the two families
    /// are absent.
    pub fn register_allocator_memory_gauges<A, R>(&self, service: &str, allocated: A, resident: R)
    where
        A: Fn() -> Option<u64> + Send + Sync + 'static,
        R: Fn() -> Option<u64> + Send + Sync + 'static,
    {
        let labels = [KeyValue::new("service", service.to_string())];
        let register = |name: &'static str,
                        description: &'static str,
                        source: Box<dyn Fn() -> Option<u64> + Send + Sync>| {
            let labels = labels.clone();
            let _gauge = self
                .meter
                .u64_observable_gauge(name)
                .with_description(description)
                .with_callback(move |observer| {
                    if let Some(bytes) = source() {
                        observer.observe(bytes, &labels);
                    }
                })
                .build();
        };
        register(
            "live_edit_jemalloc_allocated_bytes",
            "Bytes currently allocated by the service through jemalloc (jemalloc stats.allocated)",
            Box::new(allocated),
        );
        register(
            "live_edit_jemalloc_resident_bytes",
            "Upper bound on bytes in physically resident pages mapped by jemalloc, including \
             freed pages not yet purged (jemalloc stats.resident); not a precise count and can \
             read above process RSS. Judge memory returned to the OS from \
             process_resident_memory_bytes",
            Box::new(resident),
        );
    }

    /// `arenas` label value on `live_edit_jemalloc_dirty_bytes`: every jemalloc
    /// arena but the snapshot fan-out's (the automatic arenas the service's
    /// workers allocate in, and jemalloc's huge arena).
    pub const JEMALLOC_DIRTY_ARENAS_AUTOMATIC: &'static str = "automatic";
    /// `arenas` label value on `live_edit_jemalloc_dirty_bytes`: the snapshot
    /// fan-out's own arena (`storage::fanout_thread`, *One arena*).
    pub const JEMALLOC_DIRTY_ARENAS_FANOUT: &'static str = "fanout";

    /// Register `live_edit_jemalloc_dirty_bytes{service, arenas}` (b.fsz d1):
    /// bytes in dirty pages — freed by the service and not yet purged back to
    /// the OS — by `arenas`: `automatic` (every arena but the fan-out arena)
    /// and `fanout` (the snapshot fan-out's arena; absent when it was not set
    /// up). `source` is invoked at each metric collection and returns
    /// `(automatic, fanout)` (jemalloc `stats.arenas.<i>.pdirty` × page size);
    /// `None` emits no sample. What it shows beside the allocated and resident
    /// pair is where the slack between live heap and resident memory sits, and
    /// how fast each arena's decay returns it. Called once at startup from
    /// `main.rs`, and only by a binary whose global allocator is jemalloc —
    /// elsewhere the family is absent.
    pub fn register_allocator_dirty_gauge<F>(&self, service: &str, source: F)
    where
        F: Fn() -> Option<(u64, Option<u64>)> + Send + Sync + 'static,
    {
        let service_kv = KeyValue::new("service", service.to_string());
        let _gauge = self
            .meter
            .u64_observable_gauge("live_edit_jemalloc_dirty_bytes")
            .with_description(
                "Bytes in dirty pages jemalloc holds - freed by the service, not yet purged back \
                 to the OS - by arenas: automatic (every arena but the snapshot fan-out's) or \
                 fanout (the snapshot fan-out's own arena)",
            )
            .with_callback(move |observer| {
                if let Some((automatic, fanout)) = source() {
                    observer.observe(
                        automatic,
                        &[
                            service_kv.clone(),
                            KeyValue::new("arenas", Self::JEMALLOC_DIRTY_ARENAS_AUTOMATIC),
                        ],
                    );
                    if let Some(fanout) = fanout {
                        observer.observe(
                            fanout,
                            &[
                                service_kv.clone(),
                                KeyValue::new("arenas", Self::JEMALLOC_DIRTY_ARENAS_FANOUT),
                            ],
                        );
                    }
                }
            })
            .build();
    }

    /// `reason` label value on `live_edit_trace_spans_dropped_total`: the span
    /// was refused because the export queue already held
    /// `OTEL_BSP_MAX_QUEUE_SIZE` spans.
    pub const TRACE_SPANS_DROPPED_REASON_QUEUE_FULL: &'static str = "queue_full";
    /// `reason` label value on `live_edit_trace_spans_dropped_total`: the span
    /// was in a batch whose export failed.
    pub const TRACE_SPANS_DROPPED_REASON_EXPORT_FAILED: &'static str = "export_failed";

    /// Register the OTLP trace-export counters, read from `source` at each
    /// metric collection:
    ///
    /// | Metric | Meaning |
    /// |---|---|
    /// | `live_edit_trace_spans_exported_total` | spans the exporter delivered |
    /// | `live_edit_trace_spans_dropped_total{reason}` | spans lost: `queue_full` (refused at the exporter's queue bound) or `export_failed` (in a batch whose export failed) |
    ///
    /// Both `reason` cells are reported on every collection, at zero until a
    /// span is lost, so a rate over either is defined from the first scrape.
    /// Every span that reaches the export pipeline ends in exactly one of the
    /// three series (`crate::observability::trace_export`).
    ///
    /// Called once at startup from `main.rs`, and only when trace export is
    /// enabled — the families' presence is that signal, so a replica without
    /// `OTEL_EXPORTER_OTLP_ENDPOINT` must not report them.
    pub fn register_trace_export_counters(
        &self,
        source: Arc<crate::observability::trace_export::TraceExportState>,
    ) {
        let exported = Arc::clone(&source);
        let _exported = self
            .meter
            .u64_observable_counter("live_edit_trace_spans_exported_total")
            .with_description("Trace spans the OTLP exporter delivered to the collector")
            .with_callback(move |observer| observer.observe(exported.exported_total(), &[]))
            .build();

        let queue_full = [KeyValue::new(
            "reason",
            Self::TRACE_SPANS_DROPPED_REASON_QUEUE_FULL,
        )];
        let export_failed = [KeyValue::new(
            "reason",
            Self::TRACE_SPANS_DROPPED_REASON_EXPORT_FAILED,
        )];
        let _dropped = self
            .meter
            .u64_observable_counter("live_edit_trace_spans_dropped_total")
            .with_description(
                "Trace spans lost before reaching the collector, by reason: queue_full (the \
                 export queue already held OTEL_BSP_MAX_QUEUE_SIZE spans) or export_failed \
                 (the span's batch failed to export)",
            )
            .with_callback(move |observer| {
                observer.observe(source.queue_full_total(), &queue_full);
                observer.observe(source.export_failed_total(), &export_failed);
            })
            .build();
    }

    /// Register the `live_edit_checkpoint_config_*_seconds` config-echo gauge
    /// family (b.ijf) so alert rules can express checkpoint thresholds
    /// *relative to this replica's pacing configuration* rather than
    /// hardcoding the numbers that happen to hold at the defaults.
    ///
    /// Six `f64` observable gauges, each labelled `service`:
    ///
    /// | Metric | Value |
    /// |---|---|
    /// | `live_edit_checkpoint_config_interval_seconds` | `LIVE_EDIT_CHECKPOINT_INTERVAL_SECS` |
    /// | `live_edit_checkpoint_config_quiesce_seconds` | `LIVE_EDIT_CHECKPOINT_QUIESCE_SECS` |
    /// | `live_edit_checkpoint_config_tick_seconds` | `LIVE_EDIT_CHECKPOINT_TICK_SECS` |
    /// | `live_edit_checkpoint_config_stuck_threshold_seconds` | `interval + quiesce + tick` |
    /// | `live_edit_checkpoint_config_adapter_call_ceiling_seconds` | the configured adapter's enforced per-call ceiling |
    /// | `live_edit_checkpoint_config_reconcile_grace_seconds` | `reconcile_grace(adapter_call_ceiling)` |
    ///
    /// The last two are the b.gg7 additions, and they are what let an operator
    /// write the *other* half of the alert recipes this family exists for: the
    /// pacing gauges say how often work is attempted, these say how long one
    /// attempt may take before it is abandoned. A "checkpoints are stalling"
    /// rule that does not know the adapter ceiling cannot distinguish a backend
    /// brownout absorbed inside one call from a loop that is not running.
    ///
    /// The `_config_` infix is load-bearing: it separates a process constant
    /// echoed for arithmetic from a measurement of the running system. An
    /// info-style single metric cannot serve this purpose — PromQL has no
    /// stable way to turn a label value into a sample value, and a threshold
    /// recipe needs the sample.
    ///
    /// **Two of the six are *derived here*, not passed in**, which is why this
    /// takes four values instead of six. The checkpoint loop computes the same
    /// `interval + quiesce + tick` for its stuck-pending WARN bound, and both
    /// read it out of [`stuck_threshold_secs`]; the reconciliation pass derives
    /// its grace from the adapter's own ceiling, and both read it out of
    /// [`crate::checkpoint::store::reconcile_grace`]. Because neither derived
    /// value is a parameter, no caller can hand a gauge a bound that differs
    /// from the one the running code applies — the single-expression invariant
    /// is structural rather than a convention a future caller has to remember.
    ///
    /// [`Duration`] rather than `f64` seconds for the same reason: four
    /// adjacent bare `f64`s are silently transposable, and interval/quiesce/tick
    /// are exactly the kind of same-typed neighbours that get swapped. The
    /// conversion to seconds happens once, inside.
    ///
    /// Observable gauges (callback per SDK collect) rather than a set-once
    /// `Gauge`: the values are process constants, so re-observing them on
    /// every scrape costs nothing and removes any question of a delta-temporality
    /// reader dropping a value that is only ever written at startup.
    ///
    /// Called once at startup, and only for a replica that actually
    /// checkpoints — the family's *presence* is itself that signal, so a
    /// replica with `LIVE_EDIT_CHECKPOINT_ENABLED=false` must not advertise
    /// pacing config (the same conditional-presence contract as
    /// `live_edit_service_token_available`). That predicate is owned and
    /// tested at
    /// [`crate::checkpoint::register_checkpoint_pacing_gauges_if_enabled`],
    /// which is what `main.rs` calls; this method registers unconditionally
    /// once asked.
    ///
    /// The `service` label matches
    /// [`Self::record_checkpoint_pending_aggregates`], so the `max()`-never-`sum()`
    /// idiom operators already use for the pending-age gauges carries over to
    /// the recipes that multiply by these.
    pub fn register_checkpoint_pacing_gauges(
        &self,
        interval: Duration,
        quiesce: Duration,
        tick: Duration,
        adapter_call_ceiling: Duration,
        service: &str,
    ) {
        let labels = [KeyValue::new("service", service.to_string())];
        let register = |name: &'static str, description: &'static str, value: f64| {
            let labels = labels.clone();
            let _gauge = self
                .meter
                .f64_observable_gauge(name)
                .with_description(description)
                .with_callback(move |observer| observer.observe(value, &labels))
                .build();
        };

        register(
            "live_edit_checkpoint_config_interval_seconds",
            "Configured checkpoint cadence upper bound (LIVE_EDIT_CHECKPOINT_INTERVAL_SECS), \
             echoed so alert rules can be written relative to this replica's pacing",
            interval.as_secs_f64(),
        );
        register(
            "live_edit_checkpoint_config_quiesce_seconds",
            "Configured checkpoint quiesce window (LIVE_EDIT_CHECKPOINT_QUIESCE_SECS), echoed \
             so alert rules can be written relative to this replica's pacing",
            quiesce.as_secs_f64(),
        );
        register(
            "live_edit_checkpoint_config_tick_seconds",
            "Configured checkpoint loop tick cadence (LIVE_EDIT_CHECKPOINT_TICK_SECS), echoed \
             so alert rules can be written relative to this replica's pacing",
            tick.as_secs_f64(),
        );
        register(
            "live_edit_checkpoint_config_stuck_threshold_seconds",
            "Longest pending age any configured checkpoint deferral can explain \
             (interval + quiesce + tick); the same bound the loop uses for its stuck-pending WARN",
            stuck_threshold_secs(interval, quiesce, tick),
        );
        register(
            "live_edit_checkpoint_config_adapter_call_ceiling_seconds",
            "Enforced wall-clock ceiling on one checkpoint-adapter call for the configured \
             adapter (StorageApi: LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS × transient retries × \
             auth attempts), echoed so alert rules can tell a call absorbed inside the ceiling \
             apart from a loop that is not running",
            adapter_call_ceiling.as_secs_f64(),
        );
        register(
            "live_edit_checkpoint_config_reconcile_grace_seconds",
            "Reconciliation grace window derived from the adapter call ceiling; an artifact is \
             only acted on once it has been observed unchanged for this long, so it is also the \
             floor on how stale a reclamation can be",
            crate::checkpoint::store::reconcile_grace(adapter_call_ceiling).as_secs_f64(),
        );
    }

    /// Flush and shut down the dual-reader meter provider. Must be called during
    /// graceful shutdown so the OTLP `PeriodicReader` performs a final export
    /// before the process exits. Idempotent (the SDK guards re-entry); a no-op
    /// beyond the first call.
    pub fn shutdown(&self) {
        if let Err(e) = self._provider.shutdown() {
            tracing::warn!(error = %e, "failed to shut down metrics meter provider");
        }
    }

    /// Set the `live_edit_service_token_available` gauge to `1`
    /// (`available = true`) or `0` (`available = false`). Recorded by the health
    /// supervisor each tick — and only when a `ServiceTokenProvider` exists, so
    /// the family stays absent from `/metrics` on replicas without one. Task
    /// `t2.fgj.dp.s7`.
    pub fn set_service_token_available(&self, available: bool) {
        self.service_token_available
            .record(if available { 1 } else { 0 }, &[]);
    }

    /// Record `live_edit_service_token_expires_in_seconds{role}` (b.d6b).
    ///
    /// `role` is one of the `ServiceTokenProvider::ROLE_*` constants
    /// (`valkey` | `storage-api` | `unspecified`). `secs` is signed and goes
    /// negative once the cached token has expired — that zero crossing is the
    /// operator-meaningful part, so it must NOT be clamped.
    ///
    /// A derived duration only; never the `expires_on` timestamp and never
    /// token bytes (SDD SR-6.2).
    pub fn set_service_token_expires_in_seconds(&self, role: &str, secs: i64) {
        self.service_token_expires_in_seconds
            .record(secs, &[KeyValue::new("role", role.to_string())]);
    }

    /// Record `live_edit_service_token_refresh_headroom_seconds{role}` (b.is1).
    ///
    /// `role` is one of the `ServiceTokenProvider::ROLE_*` constants
    /// (`valkey` | `storage-api` | `unspecified`). `secs` is signed and goes
    /// negative when the replacement token was picked up only after the token
    /// it replaced had already expired — that sign change is the
    /// operator-meaningful part, so it must NOT be clamped.
    ///
    /// Called once per observed rotation, not per poll tick.
    ///
    /// A derived duration only; never the `expires_on` timestamp and never
    /// token bytes (SDD SR-6.2).
    pub fn set_service_token_refresh_headroom_seconds(&self, role: &str, secs: i64) {
        self.service_token_refresh_headroom_seconds
            .record(secs, &[KeyValue::new("role", role.to_string())]);
    }

    /// Increment the health probe failure counter for the given backend label
    /// (b.u96 item 8). Allowed values: `valkey`.
    pub fn inc_health_probe_failure(&self, backend: &'static str) {
        self.health_probe_failures_total
            .add(1, &[KeyValue::new("backend", backend)]);
    }

    /// Increment the periodic-checkpoint-loop per-write counter for the given
    /// `(result, kind)` cell (SDD SR-2.2).
    pub fn inc_checkpoint_write(&self, result: &str, kind: &str) {
        self.checkpoint_writes_total.add(
            1,
            &[
                KeyValue::new("result", result.to_string()),
                KeyValue::new("kind", kind.to_string()),
            ],
        );
    }

    /// Record one checkpoint tail-capture outcome (SDD SR-8.2).
    ///
    /// `outcome` must be a member of [`CHECKPOINT_TAIL_OUTCOMES`], the
    /// vocabulary zero-seeded at construction — an unlisted value would create
    /// a series that first appears only once that outcome occurs, defeating the
    /// "no tail has ever been refused" assertion the zero-seeding exists to
    /// support.
    ///
    /// Increments by one rather than taking a count: unlike the reconciliation
    /// outcomes this is never set-shaped — one capture attempt produces exactly
    /// one outcome.
    pub fn inc_checkpoint_tail(&self, outcome: &str) {
        self.checkpoint_tail_total
            .add(1, &[KeyValue::new("outcome", outcome.to_string())]);
    }

    /// Record one base-bootstrap invocation's outcome (SDD SR-7).
    ///
    /// `outcome` must be a member of [`CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES`] and
    /// `site` a member of [`CHECKPOINT_BASE_BOOTSTRAP_SITES`] — the grid
    /// zero-seeded at construction. An unlisted value would create a series that
    /// first appears only once that arm is taken, defeating the "no bootstrap
    /// has ever been shed" assertion the zero-seeding exists to support. Callers
    /// spell `site` through `BootstrapTarget::label()` rather than as a literal.
    ///
    /// Increments by one rather than taking a count, and must be called on
    /// **every** exit path of
    /// `checkpoint::base_bootstrap::BaseBootstrap::run_once`: one invocation
    /// produces exactly one outcome, and
    /// [`Self::checkpoint_base_bootstrap_total`]'s documented identity
    /// (`sum by (outcome)` equals the invocation count) is what the
    /// admission-saturation ratio and the producer-liveness predicate are built
    /// on. An arm that returns without calling this breaks both.
    pub fn inc_checkpoint_base_bootstrap(&self, outcome: &str, site: &str) {
        self.checkpoint_base_bootstrap_total.add(
            1,
            &[
                KeyValue::new("outcome", outcome.to_string()),
                KeyValue::new("site", site.to_string()),
            ],
        );
    }

    /// Count `count` external values whose backup copy the mirror dropped for
    /// `reason`, one of [`EXT_MIRROR_DROP_REASONS`] (SDD §22.17).
    pub fn add_checkpoint_ext_mirror_dropped(&self, reason: &'static str, count: u64) {
        self.checkpoint_ext_mirror_dropped_total
            .add(count, &[KeyValue::new("reason", reason)]);
    }

    /// Move the mirror's charged bytes in `state`, one of
    /// [`EXT_MIRROR_STATES`], by `delta` (SDD §22.17).
    pub fn add_checkpoint_ext_mirror_bytes(&self, state: &'static str, delta: i64) {
        self.checkpoint_ext_mirror_bytes
            .add(delta, &[KeyValue::new("state", state)]);
    }

    /// Increment the worker-side ext-value mirror failure counter (SDD SR-3.1).
    pub fn inc_checkpoint_ext_mirror_error(&self) {
        self.checkpoint_ext_mirror_errors_total.add(1, &[]);
    }

    /// Add `n` to the backup repair's `outcome` cell, one of
    /// [`EXT_BACKUP_REPAIR_OUTCOMES`] (SDD §22.17). No-op on `n == 0`.
    pub fn inc_checkpoint_ext_backup_repair(&self, outcome: &'static str, n: u64) {
        if n == 0 {
            return;
        }
        self.checkpoint_ext_backup_repair_total
            .add(n, &[KeyValue::new("outcome", outcome)]);
    }

    /// Publish the backup repair's owed count under `service`: the leader's
    /// reading from a complete pass, or 0 from a non-leader (an OTel `Gauge`
    /// never decays, so a demoted leader would otherwise pin the fleet-wide
    /// `max()`). A pass that did not complete calls nothing: unknown is not
    /// zero.
    pub fn record_checkpoint_ext_backup_owed(&self, owed: u64, service: &str) {
        self.checkpoint_ext_backup_owed
            .record(owed, &[KeyValue::new("service", service.to_string())]);
    }

    /// Count one failed write of a value's backup confirmation for `op`, one
    /// of [`EXT_BACKUP_MARK_OPS`].
    pub fn inc_checkpoint_ext_backup_mark_failed(&self, op: &'static str) {
        self.checkpoint_ext_backup_mark_failed_total
            .add(1, &[KeyValue::new("op", op)]);
    }

    /// Increment the service-identity token parse-error counter (SDD SR-6.2).
    pub fn inc_service_token_parse_error(&self) {
        self.service_token_parse_errors_total.add(1, &[]);
    }

    /// Increment the ext-value recovery-via-adapter outcome counter (SDD SR-3.2).
    /// `result` must be one of `recovered`, `adapter_miss`, `adapter_transient`,
    /// `adapter_permanent`, `adapter_other`.
    pub fn inc_ext_recovery(&self, result: &str) {
        self.checkpoint_ext_recovery_total
            .add(1, &[KeyValue::new("result", result.to_string())]);
    }

    /// Add `n` checked external-value references to
    /// `live_edit_ext_value_verify_total` (SDD §11.1). `result` is one of
    /// [`EXT_VALUE_VERIFY_RESULTS`].
    pub fn add_ext_value_verify(&self, result: &'static str, n: u64) {
        self.ext_value_verify_total
            .add(n, &[KeyValue::new("result", result)]);
    }

    /// Increment the recovery-hook attempts counter for the given outcome
    /// (SR-2.4 / SR-2.6 / SR-2.7). `scope` is `layer` or `session`.
    pub fn inc_checkpoint_recovery_attempt(&self, result: &str, scope: &str, service: &str) {
        self.checkpoint_recovery_attempts_total.add(
            1,
            &[
                KeyValue::new("result", result.to_string()),
                KeyValue::new("scope", scope.to_string()),
                KeyValue::new("service", service.to_string()),
            ],
        );
    }

    /// Record one layer restore whose surviving per-object stream was refused by
    /// the SR-2.5 high-water plausibility gate and removed (b.8y5).
    ///
    /// Additive to [`Self::inc_checkpoint_recovery_attempt`], never a substitute:
    /// the same restore still reports `result="restored"` there.
    pub fn inc_checkpoint_recovery_stream_refused(&self, service: &str) {
        self.checkpoint_recovery_stream_refused_total
            .add(1, &[KeyValue::new("service", service.to_string())]);
    }

    /// Record one layer restore that wiped a surviving `obj:*` keyspace before
    /// materializing over it (b.f4t).
    ///
    /// Additive to [`Self::inc_checkpoint_recovery_attempt`], never a
    /// substitute: the same restore still reports `result="restored"` there.
    pub fn inc_checkpoint_restore_residue_wiped(&self, service: &str) {
        self.checkpoint_restore_residue_wiped_total
            .add(1, &[KeyValue::new("service", service.to_string())]);
    }

    /// Record one layer-restore attempt that found a peer replica had already
    /// landed this object's rebuild, and therefore wrote nothing (b.h4m).
    ///
    /// `stage` MUST be a member of [`CHECKPOINT_RESTORE_FENCED_STAGES`] — an
    /// unlisted value materialises a cell that was never zero-seeded, so a
    /// dashboard sees it only after it first fires, which is indistinguishable
    /// from a fence that was never wired.
    ///
    /// Additive to [`Self::inc_checkpoint_recovery_attempt`], never a
    /// substitute: the fenced attempt still reports `result="restored"` there,
    /// because the layer really is restored — by the peer.
    pub fn inc_checkpoint_restore_fenced(&self, stage: &str, service: &str) {
        self.checkpoint_restore_fenced_total.add(
            1,
            &[
                KeyValue::new("stage", stage.to_string()),
                KeyValue::new("service", service.to_string()),
            ],
        );
    }

    /// Record one layer restore that could not establish the state of the
    /// per-object stream key (b.4et).
    ///
    /// `reason` MUST be a member of
    /// [`CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS`] — an unlisted value
    /// materialises a cell that was never zero-seeded, so a dashboard sees it
    /// only after it first fires, which is indistinguishable from a code path
    /// that was never wired.
    ///
    /// Additive to [`Self::inc_checkpoint_recovery_attempt`], never a
    /// substitute: on the two refusing reasons the same attempt also reports
    /// `result="error"` there, and on `wrongtype_key` it reports
    /// `result="restored"`.
    pub fn inc_checkpoint_recovery_stream_unresolved(&self, reason: &str, service: &str) {
        self.checkpoint_recovery_stream_unresolved_total.add(
            1,
            &[
                KeyValue::new("reason", reason.to_string()),
                KeyValue::new("service", service.to_string()),
            ],
        );
    }

    /// Record one restore-source lineage-selection outcome (SDD SR-3.1).
    ///
    /// `outcome` must be a member of [`CHECKPOINT_RESTORE_LINEAGE_OUTCOMES`],
    /// the vocabulary zero-seeded at construction — an unlisted value would
    /// create a series that first appears only once that outcome occurs,
    /// defeating the "DR has never stepped down" assertion the zero-seeding
    /// exists to support. Callers spell it through
    /// `checkpoint::lineage`'s decision types rather than as a literal.
    ///
    /// Additive to [`Self::inc_checkpoint_recovery_attempt`], never a
    /// substitute: the same restore still reports its one-per-attempt `result`
    /// there. Unlike that counter this one is **not** one-per-attempt — a
    /// single attempt that steps down twice before resolving increments three
    /// times here. The two counters are not independent on the exhaustion arms,
    /// though: `exhausted` accompanies `result="not_recoverable"` and
    /// `exhausted_budget` accompanies `result="error"`, because the exhaustion
    /// variant is what decides whether the attempt ends terminal or retriable.
    pub fn inc_checkpoint_restore_lineage(&self, outcome: &str, service: &str) {
        self.checkpoint_restore_lineage_total.add(
            1,
            &[
                KeyValue::new("outcome", outcome.to_string()),
                KeyValue::new("service", service.to_string()),
            ],
        );
    }

    /// Record one session-checkpoint adapter operation whose publish the b.443
    /// watermark refused.
    ///
    /// `result` MUST be a member of
    /// [`CHECKPOINT_SESSION_PUBLISH_SUPERSEDED_RESULTS`] — an unlisted value
    /// materialises a cell that was never zero-seeded, so it appears to a
    /// dashboard only after it first fires.
    ///
    /// Additive to [`Self::inc_checkpoint_write`], never a substitute: the
    /// clobbering operation succeeded at the adapter and already ticked
    /// `result="success", kind="session"` there.
    pub fn inc_checkpoint_session_publish_superseded(&self, result: &str) {
        self.checkpoint_session_publish_superseded_total
            .add(1, &[KeyValue::new("result", result.to_string())]);
    }

    /// Observe one per-layer checkpoint **pending-age** sample in seconds
    /// (SDD SR-2.2, b.jq3).
    ///
    /// Callers must have already established both scoping conditions
    /// documented on [`Self::checkpoint_pending_age_seconds`]: the layer has
    /// commits its newest durable generation does not carry, and that
    /// generation exists (`last_checkpoint_ms != 0`). The checkpoint loop's
    /// `PendingAgeAccumulator` is the only caller and enforces both.
    pub fn observe_checkpoint_pending_age(&self, secs: f64) {
        self.checkpoint_pending_age_seconds.record(secs, &[]);
    }

    /// Publish one checkpoint tick's two pending-age aggregates (b.jq3).
    ///
    /// Called exactly once per tick per replica: by the leader after a tick
    /// that ran to completion (a tick cut short by shutdown publishes nothing —
    /// a partial SCAN pass understates the maximum), and by every non-leader
    /// replica with `(0.0, 0)` so a demoted leader stops exporting the last
    /// maximum it observed. A replica whose leader-lock probe *failed*
    /// publishes nothing at all: unknown is not zero.
    pub fn record_checkpoint_pending_aggregates(
        &self,
        oldest_pending_age_secs: f64,
        layers_never_written: u64,
        service: &str,
    ) {
        let labels = [KeyValue::new("service", service.to_string())];
        self.checkpoint_oldest_pending_age_seconds
            .record(oldest_pending_age_secs, &labels);
        self.checkpoint_layers_never_written
            .record(layers_never_written, &labels);
    }

    /// Record one checkpoint-assembly attempt's stable-cut outcome (b.pw6).
    ///
    /// Modelled on [`Self::observe_stable_cut`], the catch-up sibling, and
    /// deliberately a separate instrument family rather than a `regime` label on
    /// that one: an `unverified` cut means different things on the two paths
    /// (a catch-up that ends `UNAVAILABLE` before any content vs. a skipped
    /// durable generation), with different operator responses, so joining them under one counter would blur a
    /// subscriber-correctness signal into a DR-coverage signal.
    ///
    /// One counter add per call: the proven read is one pass, atomic or fenced
    /// (a fenced read's broken attempts are counted on the snapshot-read
    /// metrics). Called from the periodic loop — the assembler itself stays
    /// metrics-free.
    pub fn observe_checkpoint_stable_cut(&self, verified: bool, service: &str) {
        let outcome = if verified { "verified" } else { "unverified" };
        self.checkpoint_stable_cut_total.add(
            1,
            &[
                KeyValue::new("outcome", outcome),
                KeyValue::new("service", service.to_string()),
            ],
        );
    }

    /// Record one checkpoint-loop mid-tick leader-lease check (b.s5y).
    ///
    /// `outcome` must be one of `renewed`, `lost`, `refresh_error` — the pinned
    /// vocabulary zero-seeded at construction. `refresh_error` is additive: an
    /// errored refresh records `refresh_error` AND `lost`, because a refresh
    /// whose outcome is unknown is not a lease this replica can prove it holds.
    ///
    /// Called only from `CheckpointLease::ensure_held`; the checks that resolve
    /// locally without a round-trip (the lease is comfortably inside its
    /// renewal window) record nothing, so this counter measures refresh
    /// decisions, not iteration count.
    pub fn inc_checkpoint_lease(&self, outcome: &str, service: &str) {
        self.checkpoint_lease_total.add(
            1,
            &[
                KeyValue::new("outcome", outcome.to_string()),
                KeyValue::new("service", service.to_string()),
            ],
        );
    }

    /// Add `n` to one reconciliation-pass outcome cell (b.9qd).
    ///
    /// `outcome` must be a member of [`CHECKPOINT_RECONCILE_OUTCOMES`], the
    /// vocabulary zero-seeded at construction — an unlisted value would create a
    /// series that appears only once something goes wrong, which is the opposite
    /// of what this counter is for.
    ///
    /// Takes a count rather than incrementing by one because most of the pass's
    /// arms are set-shaped (n hashes tombstoned, n candidates deferred) and
    /// looping to add ones would just be a slower way to say the same thing.
    /// No-op on `n == 0` so callers stay branch-free.
    pub fn add_checkpoint_reconcile(&self, outcome: &str, n: u64, service: &str) {
        if n == 0 {
            return;
        }
        self.checkpoint_reconcile_total.add(
            n,
            &[
                KeyValue::new("outcome", outcome.to_string()),
                KeyValue::new("service", service.to_string()),
            ],
        );
    }

    /// Record one checkpoint-adapter ceiling elapse (b.gg7).
    ///
    /// `phase` must be a member of [`CHECKPOINT_ADAPTER_CEILING_PHASES`], the
    /// vocabulary zero-seeded at construction.
    ///
    /// Increments by one rather than taking a count: unlike the reconciliation
    /// outcomes this is never set-shaped — one elapse is one abandoned call.
    pub fn inc_checkpoint_adapter_ceiling_elapsed(&self, phase: &str, service: &str) {
        self.checkpoint_adapter_ceiling_elapsed_total.add(
            1,
            &[
                KeyValue::new("phase", phase.to_string()),
                KeyValue::new("service", service.to_string()),
            ],
        );
    }

    /// Record one storage-api client-side redirect event (b.z22).
    ///
    /// `(verb, outcome)` must be a member of
    /// [`CHECKPOINT_ADAPTER_REDIRECT_CELLS`], the vocabulary zero-seeded at
    /// construction. Increments by one — one call site, one event.
    ///
    /// Callers MUST NOT pass a redirect target URL in any argument. A presigned
    /// URL is a bearer credential (SR-6.2) and this family is deliberately
    /// URL-free; the labels are a fixed low-cardinality vocabulary, not a place
    /// to smuggle per-request detail.
    pub fn inc_checkpoint_adapter_redirect(&self, verb: &str, outcome: &str, service: &str) {
        self.checkpoint_adapter_redirect_total.add(
            1,
            &[
                KeyValue::new("verb", verb.to_string()),
                KeyValue::new("outcome", outcome.to_string()),
                KeyValue::new("service", service.to_string()),
            ],
        );
    }

    /// Publish the ext-blob tombstone backlog observed by one reconciliation
    /// pass (b.9qd).
    ///
    /// Called once per pass by the leader with the live `ZCARD`, and by every
    /// non-leader replica with `0`. The zero publish is required, not cosmetic:
    /// an OTel `Gauge` never decays, so a replica that held the lock, saw a
    /// large backlog, and then lost the lock would keep exporting that backlog
    /// forever and pin the fleet-wide `max()`. A replica whose leader probe
    /// *errored* publishes nothing — unknown is not zero.
    pub fn record_checkpoint_ext_tombstone_backlog(&self, backlog: u64, service: &str) {
        self.checkpoint_ext_tombstone_backlog
            .record(backlog, &[KeyValue::new("service", service.to_string())]);
    }

    /// Publish the dead-object backlog observed by one reconciliation pass
    /// (b.vn6).
    ///
    /// Same contract as [`Self::record_checkpoint_ext_tombstone_backlog`],
    /// deliberately: called once per pass by the leader with the live `ZCARD`
    /// and by every non-leader replica with `0`, because an OTel `Gauge` never
    /// decays and a demoted leader would otherwise pin the fleet-wide `max()`
    /// at a backlog that no longer exists. A replica whose leader probe
    /// *errored* publishes nothing — unknown is not zero.
    pub fn record_checkpoint_dead_object_backlog(&self, backlog: u64, service: &str) {
        self.checkpoint_dead_object_backlog
            .record(backlog, &[KeyValue::new("service", service.to_string())]);
    }

    /// Observe the wall-clock duration in seconds of one counter-seed routine
    /// invocation (SDD SR-5.3), tagged with the call site's `phase`
    /// (`startup` | `repair` — `SeedTrigger::label`).
    pub fn observe_checkpoint_counter_seed(&self, secs: f64, phase: &str) {
        self.checkpoint_counter_seed_secs
            .record(secs, &[KeyValue::new("phase", phase.to_string())]);
    }

    /// Increment the `CreateSection`-with-missing-parent skip counter by `n`
    /// (b.39x item 7). No-op on `n == 0` to keep callers branch-free.
    pub fn inc_delta_create_section_skipped_missing_parent(&self, n: u64) {
        if n == 0 {
            return;
        }
        self.delta_create_section_skipped_missing_parent_total
            .add(n, &[]);
    }

    /// Increment the GC sweep batch error counter for a given error kind
    /// (b.kf1 medium).
    pub fn inc_gc_sweep_batch_error(&self, kind: &str) {
        self.gc_sweep_batch_errors_total
            .add(1, &[KeyValue::new("kind", kind.to_string())]);
    }

    /// Increment the GC `ref_count` underflow counter (b.j9j item 4).
    pub fn inc_gc_underflow(&self) {
        self.gc_underflow_total.add(1, &[]);
    }

    /// Count one GC decrement run with `outcome` (b.yuo), one of
    /// [`GC_DECREMENT_OUTCOMES`].
    pub fn inc_gc_decrement(&self, outcome: &'static str) {
        self.gc_decrement_total
            .add(1, &[KeyValue::new("outcome", outcome)]);
    }

    /// Count one GC action on an older build's state (b.1jt) with `event`,
    /// one of [`GC_OLDER_BUILD_EVENTS`].
    pub fn inc_gc_older_build(&self, event: &'static str) {
        self.inc_gc_older_build_by(event, 1);
    }

    /// Count `n` GC actions on an older build's state (b.1jt) with `event`,
    /// one of [`GC_OLDER_BUILD_EVENTS`] — the per-member and per-reference
    /// events of one batch.
    pub fn inc_gc_older_build_by(&self, event: &'static str, n: u64) {
        self.gc_older_build_total
            .add(n, &[KeyValue::new("event", event)]);
    }

    /// Count one GC out-of-memory collection (b.pa1) with `outcome`, one of
    /// [`GC_OOM_COLLECTION_OUTCOMES`], and add the bytes it freed to
    /// `live_edit_gc_oom_freed_bytes_total`.
    pub fn record_gc_oom_collection(&self, outcome: &'static str, freed_bytes: u64) {
        debug_assert!(
            GC_OOM_COLLECTION_OUTCOMES.contains(&outcome),
            "unknown gc_oom_collections outcome label {outcome:?}"
        );
        self.gc_oom_collections_total
            .add(1, &[KeyValue::new("outcome", outcome)]);
        if freed_bytes > 0 {
            self.gc_oom_freed_bytes_total.add(freed_bytes, &[]);
        }
    }

    /// Count one release-budget gate decision with `outcome` (b.yuo), one of
    /// [`EXT_RELEASE_GATE_OUTCOMES`].
    pub fn inc_ext_release_gate(&self, outcome: &'static str) {
        self.ext_release_gate_total
            .add(1, &[KeyValue::new("outcome", outcome)]);
    }

    /// Count one catch-up external-value miss with `outcome` (b.yuo), one of
    /// [`CATCHUP_EXT_VALUE_MISS_OUTCOMES`], in a snapshot read as `read`, one
    /// of [`CATCHUP_EXT_VALUE_MISS_READS`].
    pub fn inc_catchup_ext_value_miss(&self, outcome: &'static str, read: &'static str) {
        self.catchup_ext_value_miss_total.add(
            1,
            &[
                KeyValue::new("outcome", outcome),
                KeyValue::new("read", read),
            ],
        );
    }

    /// Record the external-value release budget in use, `bytes`, under
    /// `source` (b.yuo), one of [`EXT_RELEASE_BUDGET_SOURCES`], and 0 under
    /// every other source, so exactly one source carries the budget.
    pub fn record_ext_release_budget(&self, source: &'static str, bytes: u64) {
        for cell in EXT_RELEASE_BUDGET_SOURCES {
            let value = if *cell == source { bytes } else { 0 };
            self.ext_release_budget_bytes.record(
                value,
                &[self.service_kv.clone(), KeyValue::new("source", *cell)],
            );
        }
    }

    /// Record `plan`'s gauges once at startup: its limit under its source and
    /// 0 under every other source of
    /// [`crate::config::PodMemoryLimitSource::ALL`], so every source series
    /// exists and exactly one carries the limit, and the catch-up budget, the
    /// retirement allowance, the structural cache's resident budget, the
    /// subscription ceiling and the checkpoint mirror's byte budget (0 when the
    /// plan does not fund the mirror). Each budget carries the labels of the
    /// usage gauge it is read against: the structural cache's `service`, as
    /// its weight gauge does; the catch-up budget, the retirement allowance and
    /// the subscription ceiling none, as
    /// `live_edit_catchup_build_budget_reserved_bytes`,
    /// `live_edit_retirement_budget_reserved_bytes` and
    /// `live_edit_subscriber_admitted` carry none. The mirror's budget is
    /// unlabelled too, and is read against the sum over `state` of
    /// `live_edit_checkpoint_ext_mirror_bytes`.
    pub fn record_pod_memory_plan(&self, plan: &crate::config::PodMemoryPlan) {
        let limit = plan.limit();
        for cell in crate::config::PodMemoryLimitSource::ALL {
            let value = if cell == limit.source { limit.bytes } else { 0 };
            self.pod_memory_limit_bytes.record(
                value,
                &[
                    self.service_kv.clone(),
                    KeyValue::new("source", cell.label()),
                ],
            );
        }
        self.catchup_build_budget_bytes
            .record(plan.catchup_budget_bytes(), &[]);
        self.retirement_budget_bytes
            .record(plan.retirement_budget_bytes(), &[]);
        self.structural_cache_budget_bytes.record(
            plan.structural_cache_resident_budget_bytes(),
            std::slice::from_ref(&self.service_kv),
        );
        self.subscriber_max_subscriptions
            .record(plan.max_subscriptions(), &[]);
        // The mirror's byte budget, unlabelled, read against the sum over
        // `state` of `checkpoint_ext_mirror_bytes`: 0 when the plan does not
        // fund the mirror, which is exactly when `main.rs` does not wire it.
        self.checkpoint_ext_mirror_budget_bytes
            .record(plan.ext_mirror_bytes().unwrap_or(0), &[]);
    }

    /// Set `live_edit_store_memory_used_ratio` (b.qhk) to `ratio`, reported
    /// at every later collection until the next call; `None` removes the
    /// series from the next collection on. Unclamped: above 1 means the store
    /// is past its limit. Written by the store-memory reads at startup and on
    /// every refresh (`storage::store_memory`).
    pub fn record_store_memory_used_ratio(&self, ratio: Option<f64>) {
        *self
            .store_memory_used_ratio
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = ratio;
    }

    /// Publish the five external-value release-ledger gauges from one GC pass
    /// (b.yuo), or the all-zero non-leader publish. Published together because
    /// they are readings of one selection. Same leader contract as
    /// [`Self::record_ext_reconcile_gauges`]: the leader publishes its pass's
    /// readings, every non-leader publishes zeros, and a pass that errored
    /// publishes nothing.
    pub fn record_ext_release_gauges(
        &self,
        pending_bytes: u64,
        pending_count: u64,
        oldest_age_seconds: f64,
        settled_count: u64,
        read_holds: u64,
    ) {
        let labels = [self.service_kv.clone()];
        self.ext_release_pending_bytes
            .record(pending_bytes, &labels);
        self.ext_release_pending_count
            .record(pending_count, &labels);
        self.ext_release_oldest_age_seconds
            .record(oldest_age_seconds, &labels);
        self.ext_release_settled_count
            .record(settled_count, &labels);
        self.ext_read_holds.record(read_holds, &labels);
    }

    /// Add `n` to the external-value release counter for one `reason` (b.qpf).
    /// Takes a count rather than incrementing by one because a single commit
    /// enqueues a whole batch. No-op on `n == 0` so callers stay branch-free.
    /// `reason` must be one of [`EXT_REF_RELEASE_REASONS`].
    ///
    /// `&'static str`, not `&str`, for the same reason as
    /// [`Self::inc_snapshot_read_type_error`]: the vocabulary is a fixed set of
    /// `const` label values, so tightening the parameter to the lifetime they
    /// actually have turns "callers pass a registered constant" from a
    /// doc-comment into a signature — a runtime-computed `String` cannot be
    /// passed at all — and lets the label ride into `KeyValue` without the
    /// per-tick allocation a `&str` parameter forced.
    pub fn inc_ext_ref_released(&self, reason: &'static str, n: u64) {
        if n == 0 {
            return;
        }
        self.ext_ref_released_total
            .add(n, &[KeyValue::new("reason", reason)]);
    }

    /// Add `n` to the failed-release-enqueue counter (b.qpf). Unlabeled; see
    /// [`Self::ext_ref_release_enqueue_failed_total`].
    ///
    /// Takes a count, like its sibling [`Self::inc_ext_ref_released`] and for
    /// the same reason: a single failed enqueue owes one decrement per
    /// `ExtRelease` in the batch, so the caller passes the batch length. The
    /// counter's unit is **references**, not enqueue calls — the alert recipe
    /// reads a tick as one leaked `ref_count`, and the guide's
    /// `failed / released` ratio query divides it by a per-reference counter.
    /// No-op on `n == 0` so callers stay branch-free.
    ///
    /// Call this only where the owed batch is in hand, so `n` is the batch
    /// length: the two counted sites are `LayerStore::enqueue_ext_releases`
    /// (commit path, after its one in-line retry, so the retry itself never
    /// doubles a tick) and `storage::layer::schedule_old_object_ext_gc`
    /// (retirement path, only when its begin failed or answered `busy`). A
    /// caller that re-runs the whole retirement — `delete_layer`'s retry
    /// cascade — re-derives the batch and ticks again, which is the residual
    /// amplification the field doc describes. A failure earlier than either of
    /// these — the retirement path's pre-schedule reads — has no honest `n`,
    /// and is deliberately left uncounted rather than ticked as one. See the field doc for the full
    /// coverage statement and for why the total is an upper bound.
    pub fn inc_ext_ref_release_enqueue_failed(&self, n: u64) {
        if n == 0 {
            return;
        }
        self.ext_ref_release_enqueue_failed_total.add(n, &[]);
    }

    /// Add `n` to the deferred-release counter (b.yda). Unlabeled; see
    /// [`Self::ext_ref_release_deferred_total`].
    ///
    /// Called at exactly one site — `schedule_old_object_ext_gc`, when both of
    /// its attempt-mode enqueues failed, or its first attempt was not sent
    /// because the retirement passed its send window (nothing was sent), or its
    /// first attempt failed in doubt and the retry was then not sent (the first
    /// attempt may have run), and the owed
    /// members are left on the retiring object's `le:{oid}:obj:extret` carrier,
    /// written by its begin, for the orphan sweep to settle. Unit: references,
    /// like every counter in this family. No-op on `n == 0`.
    pub fn inc_ext_ref_release_deferred(&self, n: u64) {
        if n == 0 {
            return;
        }
        self.ext_ref_release_deferred_total.add(n, &[]);
    }

    /// Add `n` to the recovered-release counter (b.yda). Unlabeled; see
    /// [`Self::ext_ref_release_recovered_total`].
    ///
    /// Called at exactly one site — the orphan sweep's retire-retry phase, after
    /// its re-drive of a carrier's members landed on `le:ext:{gc}:pending`.
    /// Unit: references. No-op on `n == 0`.
    pub fn inc_ext_ref_release_recovered(&self, n: u64) {
        if n == 0 {
            return;
        }
        self.ext_ref_release_recovered_total.add(n, &[]);
    }

    /// Add `n` to the already-enqueued counter (b.yuo Part R). Unlabeled; see
    /// [`Self::ext_ref_release_already_enqueued_total`].
    ///
    /// Called at exactly one site — the orphan sweep's retire-retry phase, after
    /// it resolved an earlier attempt of a carrier as `landed` and cleared the
    /// carrier without enqueuing its members again. Unit: references. No-op on
    /// `n == 0`.
    pub fn inc_ext_ref_release_already_enqueued(&self, n: u64) {
        if n == 0 {
            return;
        }
        self.ext_ref_release_already_enqueued_total.add(n, &[]);
    }

    /// Add `n` to the cancelled-release counter (b.yda). Unlabeled; see
    /// [`Self::ext_ref_release_cancelled_total`].
    ///
    /// Called at exactly one site — the orphan sweep's retire-retry phase, when
    /// the settle script answered `cancelled` because the object id was live
    /// again and the carrier's members were therefore dropped rather than
    /// enqueued. Unit: references. No-op on `n == 0`.
    pub fn inc_ext_ref_release_cancelled(&self, n: u64) {
        if n == 0 {
            return;
        }
        self.ext_ref_release_cancelled_total.add(n, &[]);
    }

    /// Increment the undecodable-reference counter for one `site` (b.yda).
    ///
    /// `&'static str` for the same reason as [`Self::inc_ext_ref_released`]: the
    /// vocabulary is the fixed [`EXT_REF_UNDECODABLE_SITES`] set of `const`
    /// label values, so the signature enforces what a doc-comment would only
    /// ask, and the label rides into `KeyValue` with no per-tick allocation.
    pub fn inc_ext_ref_undecodable(&self, site: &'static str) {
        self.ext_ref_undecodable_total
            .add(1, &[KeyValue::new("site", site)]);
    }

    /// Add `n` to the reconciler outcome counter for one `outcome` (b.yda).
    ///
    /// Takes a count because the pass reports whole phases at once (candidates
    /// registered, candidates healed). `&'static str` over the fixed
    /// [`EXT_RECONCILE_OUTCOMES`] vocabulary, as above. No-op on `n == 0` so the
    /// pass can report every outcome unconditionally at its end without a guard
    /// per cell.
    pub fn inc_ext_reconcile(&self, outcome: &'static str, n: u64) {
        if n == 0 {
            return;
        }
        self.ext_reconcile_total
            .add(n, &[KeyValue::new("outcome", outcome)]);
    }

    /// Count one per-object reference-index recount at `site` with `outcome`.
    ///
    /// `&'static str` over the fixed [`EXT_INDEX_RECOUNT_SITES`] and
    /// [`EXT_INDEX_RECOUNT_OUTCOMES`] vocabularies, as above.
    pub fn inc_ext_index_recount(&self, site: &'static str, outcome: &'static str) {
        self.ext_index_recount_total.add(
            1,
            &[
                KeyValue::new("site", site),
                KeyValue::new("outcome", outcome),
            ],
        );
    }

    /// Add `n` time-sample rows to the signed-zero merge's `outcome` cell.
    /// `&'static str` over the fixed [`TIME_SAMPLE_SIGNED_ZERO_MERGE_OUTCOMES`]
    /// vocabulary. No-op on `n == 0`.
    pub fn inc_time_sample_signed_zero_merge(&self, outcome: &'static str, n: u64) {
        if n == 0 {
            return;
        }
        self.time_sample_signed_zero_merge_total
            .add(n, &[KeyValue::new("outcome", outcome)]);
    }

    /// Add `n` index fields corrected at `site` in `direction`. No-op on
    /// `n == 0`, so a repair can report both directions unconditionally.
    /// `site` is one of [`EXT_INDEX_RECOUNT_SITES`] or
    /// [`EXT_INDEX_DRIFT_ONLY_SITES`].
    pub fn inc_ext_index_drift(&self, site: &'static str, direction: &'static str, n: u64) {
        if n == 0 {
            return;
        }
        self.ext_index_drift_total.add(
            n,
            &[
                KeyValue::new("site", site),
                KeyValue::new("direction", direction),
            ],
        );
    }

    /// Publish the four reference-count reconciler gauges from one completed
    /// pass (b.yda), or the all-zero non-leader publish.
    ///
    /// Published together, deliberately: they are four readings of ONE
    /// observation and are only meaningful against each other (`surplus` is
    /// derived from `meta` minus `referenced` minus what is already owed). A
    /// caller that could publish three of the four would let an operator read a
    /// surplus against a stale meta count from a different pass.
    ///
    /// The two directions are NOT symmetric and the caller must not pass them
    /// symmetrically: `surplus` is every positive surplus this pass measured,
    /// while `confirmed_deficit` is only the deficit that survived a full grace
    /// at the same amount (see [`Self::ext_ref_deficit_total`]). Publishing a raw
    /// per-pass negative surplus here would put routine mid-pass double-counting
    /// onto an alerted gauge — one whose alert shape already has to tolerate the
    /// residue of that double-counting that survives confirmation.
    ///
    /// Same leader contract as [`Self::record_checkpoint_ext_tombstone_backlog`]:
    /// the leader publishes its live readings, every non-leader publishes zeros
    /// so a demoted leader cannot pin the fleet-wide `max()` at a level that no
    /// longer exists, and a pass whose enumeration FAILED publishes nothing at
    /// all — unknown is not zero, and a partial walk's surplus is exactly the
    /// number that must never be believed.
    pub fn record_ext_reconcile_gauges(
        &self,
        meta_total: u64,
        referenced_hashes: u64,
        surplus: u64,
        confirmed_deficit: u64,
        service: &str,
    ) {
        let labels = [KeyValue::new("service", service.to_string())];
        self.ext_meta_total.record(meta_total, &labels);
        self.ext_referenced_hashes_total
            .record(referenced_hashes, &labels);
        self.ext_ref_surplus_total.record(surplus, &labels);
        self.ext_ref_deficit_total
            .record(confirmed_deficit, &labels);
    }

    /// Publish the retire-retry backlog observed by one orphan-sweep tick
    /// (b.yda).
    ///
    /// Same contract as [`Self::record_checkpoint_ext_tombstone_backlog`]: the
    /// leader publishes the live `ZCARD`, every non-leader publishes `0` because
    /// an OTel `Gauge` never decays, and a replica whose leader probe errored
    /// publishes nothing.
    pub fn record_ext_retire_retry_backlog(&self, backlog: u64, service: &str) {
        self.ext_retire_retry_backlog
            .record(backlog, &[KeyValue::new("service", service.to_string())]);
    }

    /// Increment the object-id-counter repair counter for one `outcome`
    /// (b.b2j). See [`Self::object_id_counter_repair_total`] for the outcome
    /// vocabulary and which cells are alertable.
    ///
    /// `outcome` must be a member of [`OBJECT_ID_COUNTER_REPAIR_OUTCOMES`], the
    /// vocabulary zero-seeded at construction — an unlisted value would create a
    /// series that first appears at the moment of the keyspace loss it reports,
    /// leaving nothing for an alert rule to have been watching beforehand.
    pub fn inc_object_id_counter_repair(&self, outcome: &str) {
        self.object_id_counter_repair_total
            .add(1, &[KeyValue::new("outcome", outcome.to_string())]);
    }

    /// Increment the endpoint call counter for the given endpoint (b.avd item
    /// 4). `grpc_code` is the textual slug for the eventual `tonic::Code`; use
    /// [`grpc_code_slug`] to translate.
    pub fn inc_calls(&self, endpoint: &str, service: &str, grpc_code: &str) {
        self.endpoint_calls.add(
            1,
            &[
                KeyValue::new("endpoint", endpoint.to_string()),
                KeyValue::new("service", service.to_string()),
                KeyValue::new("grpc_code", grpc_code.to_string()),
            ],
        );
    }

    /// Record an endpoint call duration in microseconds (b.avd item 4).
    pub fn observe_duration_us(
        &self,
        endpoint: &str,
        service: &str,
        grpc_code: &str,
        duration_us: f64,
    ) {
        self.endpoint_duration_us.record(
            duration_us,
            &[
                KeyValue::new("endpoint", endpoint.to_string()),
                KeyValue::new("service", service.to_string()),
                KeyValue::new("grpc_code", grpc_code.to_string()),
            ],
        );
    }

    /// Increment the in-flight RPC gauge (b.avd item 5). Paired with
    /// [`Self::dec_inflight_rpc`].
    pub fn inc_inflight_rpc(&self, endpoint: &str, service: &str) {
        self.inflight_rpcs.add(
            1,
            &[
                KeyValue::new("endpoint", endpoint.to_string()),
                KeyValue::new("service", service.to_string()),
            ],
        );
    }

    /// Decrement the in-flight RPC gauge (b.avd item 5).
    pub fn dec_inflight_rpc(&self, endpoint: &str, service: &str) {
        self.inflight_rpcs.add(
            -1,
            &[
                KeyValue::new("endpoint", endpoint.to_string()),
                KeyValue::new("service", service.to_string()),
            ],
        );
    }

    /// Add `n` steady-state commits delivered to `SubscribeReadLayer`
    /// subscribers (see [`Self::subscriber_updates`] for what counts).
    ///
    /// Callers batch: each subscription accumulates its deliveries and flushes
    /// them here, because a per-delivery `add(1)` contends one counter cell
    /// across all subscriber tasks under fan-out; batching removes that, and
    /// the interleaved comparison showed the batched build at parity or better
    /// with the uncounted baseline. A delivery therefore reaches the counter
    /// normally within about a second, and all of them by the time the stream
    /// ends; one waiting behind a blocked send reaches it once the
    /// subscription's loop next goes idle. Allocates nothing: the `endpoint` value is a
    /// `&'static str` and the `service` value is the `Arc`-backed label built
    /// once in [`Metrics::new`].
    pub fn add_subscriber_updates(&self, n: u64) {
        self.subscriber_updates.add(
            n,
            &[
                KeyValue::new("endpoint", ENDPOINT_SUBSCRIBE_READ_LAYER),
                self.service_kv.clone(),
            ],
        );
    }

    /// Observe one join latency (see [`Self::join_latency_seconds`]).
    /// `endpoint` is [`ENDPOINT_SUBSCRIBE_READ_LAYER`] or
    /// [`ENDPOINT_JOIN_SESSION`].
    pub fn observe_join_latency(&self, endpoint: &'static str, elapsed: Duration) {
        self.join_latency_seconds.record(
            elapsed.as_secs_f64(),
            &[KeyValue::new("endpoint", endpoint), self.service_kv.clone()],
        );
    }

    /// Record a diff build: increment the case counter and observe the duration.
    pub fn observe_diff_build(&self, case: &str, service: &str, duration: Duration) {
        let attrs = [
            KeyValue::new("case", case.to_string()),
            KeyValue::new("service", service.to_string()),
        ];
        self.diff_case_total.add(1, &attrs);
        self.diff_build_duration_seconds
            .record(duration.as_secs_f64(), &attrs);
    }

    /// Record a `SubscribeReadLayer` catch-up: increment the regime counter,
    /// observe the catch-up phase duration (b.qxe), and observe the delivered
    /// OMPF byte count (Plan b.msb). `catchup_mode` (`snapshot` | `legacy`)
    /// lands as an additive attribute on ALL three instruments alongside the
    /// pre-existing `regime` + `service` labels.
    pub fn observe_subscribe_regime(
        &self,
        regime: &str,
        service: &str,
        duration: Duration,
        catchup_mode: &str,
        delivered_bytes: u64,
    ) {
        let attrs = [
            KeyValue::new("regime", regime.to_string()),
            KeyValue::new("service", service.to_string()),
            KeyValue::new("catchup_mode", catchup_mode.to_string()),
        ];
        self.subscribe_regime_total.add(1, &attrs);
        self.subscribe_catch_up_duration_seconds
            .record(duration.as_secs_f64(), &attrs);
        self.subscribe_catch_up_delivered_bytes
            .record(delivered_bytes as f64, &attrs);
    }

    /// Record one catch-up materialized-state read's cut outcome (b.vd8, Spec
    /// SDD SR-8.6; see [`Self::subscribe_stable_cut_total`] for what each
    /// outcome means, and for the rollback-mode in-window replay, which
    /// records only a regressed per-commit read). One counter add per call:
    /// every catch-up read is one pass — an atomic read, or a catch-up's share
    /// of its flight's fenced read.
    ///
    /// `regime` uses the same label vocabulary as
    /// [`Self::observe_subscribe_regime`] (`brand_new` | `in_window` |
    /// `out_of_window`; `up_to_date` never reads materialized state, so it never
    /// reaches here), which is what makes the two instruments joinable — an
    /// `unverified` rate is only actionable once you know which catch-up shape
    /// hit the regression.
    pub fn observe_stable_cut(&self, verified: bool, service: &str, regime: &'static str) {
        let outcome = if verified { "verified" } else { "unverified" };
        self.subscribe_stable_cut_total.add(
            1,
            &[
                KeyValue::new("outcome", outcome),
                KeyValue::new("service", service.to_string()),
                KeyValue::new("regime", regime),
            ],
        );
    }

    /// Add `n` live presence events of `event_type` delivered to local
    /// `JoinSession` streams (see [`Self::presence_events_total`] for what
    /// counts). `event_type` is one of [`PRESENCE_EVENT_TYPES`].
    ///
    /// Callers batch: each `JoinSession` stream accumulates its deliveries and
    /// flushes them here. The counter is per recipient per event, so a
    /// per-delivery `add(1)` puts every recipient task on the same counter
    /// cell. In an interleaved A/B on a 400-member session (~425k presence
    /// events/s) per-delivery counting cost about 4.5 % more service CPU per
    /// delivered event than an uncounted baseline; the batched form measured
    /// within noise of the baseline. Allocates nothing: the same
    /// `&'static str` label and `Arc`-backed `service` label as
    /// [`Self::add_subscriber_updates`].
    pub fn add_presence_events(&self, event_type: &'static str, n: u64) {
        self.presence_events_total.add(
            n,
            &[
                KeyValue::new("event_type", event_type),
                self.service_kv.clone(),
            ],
        );
    }

    /// Record `n` presence events a local JoinSession stream lost by falling
    /// more than the per-session fan-out capacity behind.
    pub fn inc_presence_broadcast_dropped(&self, service: &str, n: u64) {
        self.presence_broadcast_dropped
            .add(n, &[KeyValue::new("service", service.to_string())]);
    }

    /// Record one JoinSession stream ended for a resync on
    /// `live_edit_presence_resyncs_total`: it lagged and missed a
    /// participant's join or leave. Carries this process's `service` label;
    /// allocates nothing.
    pub fn inc_presence_resync(&self) {
        self.presence_resyncs
            .add(1, std::slice::from_ref(&self.service_kv));
    }

    /// Record one JoinSession stream ended for client silence on
    /// `live_edit_presence_keepalive_timeouts_total`: it opted in to client
    /// keepalives and sent no data for three intervals. Carries this
    /// process's `service` label; allocates nothing.
    pub fn inc_presence_keepalive_timeout(&self) {
        self.presence_keepalive_timeouts
            .add(1, std::slice::from_ref(&self.service_kv));
    }

    /// Add `n` pub/sub messages the replica-wide `relay` lost to a lag of its
    /// receiver on the pub/sub client's message broadcast channel, on
    /// `live_edit_pubsub_relay_lagged_total`. `relay` is one of
    /// [`PUBSUB_RELAYS`]. Carries this process's `service` label; allocates
    /// nothing.
    pub fn add_pubsub_relay_lagged(&self, relay: &'static str, n: u64) {
        self.pubsub_relay_lagged
            .add(n, &[KeyValue::new("relay", relay), self.service_kv.clone()]);
    }

    /// Record one failed presence lease-keeper step on
    /// `live_edit_presence_lease_refresh_failures_total`. Carries this
    /// process's `service` label; allocates nothing.
    pub fn inc_presence_lease_refresh_failure(&self) {
        self.presence_lease_refresh_failures
            .add(1, std::slice::from_ref(&self.service_kv));
    }

    /// Record one self-rejoin on `live_edit_presence_rejoins_total`: a lease
    /// keeper's register re-added its evicted connection's membership
    /// (`Registered { added: true }`). Carries this process's `service` label;
    /// allocates nothing.
    pub fn inc_presence_rejoin(&self) {
        self.presence_rejoins
            .add(1, std::slice::from_ref(&self.service_kv));
    }

    /// Observe one presence-sweeper leader pass's wall time on
    /// `live_edit_presence_sweep_duration_seconds`. `outcome` is one of
    /// [`PRESENCE_SWEEP_OUTCOMES`]. Carries this process's `service` label.
    pub fn observe_presence_sweep_duration(&self, outcome: &'static str, secs: f64) {
        self.presence_sweep_duration_seconds.record(
            secs,
            &[KeyValue::new("outcome", outcome), self.service_kv.clone()],
        );
    }

    /// Record one presence-sweeper lease decision on
    /// `live_edit_presence_sweep_lease_total`. `outcome` is one of
    /// [`PRESENCE_SWEEP_LEASE_OUTCOMES`]; a caller recording
    /// [`PRESENCE_SWEEP_LEASE_REFRESH_ERROR`] also records
    /// [`PRESENCE_SWEEP_LEASE_LOST`]. Carries this process's `service` label;
    /// allocates nothing.
    pub fn inc_presence_sweep_lease(&self, outcome: &'static str) {
        self.presence_sweep_lease_total.add(
            1,
            &[KeyValue::new("outcome", outcome), self.service_kv.clone()],
        );
    }

    /// Add `n` sweeper removals to `live_edit_presence_stale_removed_total`.
    /// Carries this process's `service` label; allocates nothing.
    pub fn add_presence_stale_removed(&self, n: u64) {
        self.presence_stale_removed
            .add(n, std::slice::from_ref(&self.service_kv));
    }

    /// Observe one sweeper removal's delay past its lease expiry on
    /// `live_edit_presence_stale_removal_delay_seconds`. Called only for a
    /// removal whose expiry was recorded. Carries this process's `service`
    /// label.
    pub fn observe_presence_stale_removal_delay(&self, secs: f64) {
        self.presence_stale_removal_delay_seconds
            .record(secs, std::slice::from_ref(&self.service_kv));
    }

    /// Publish `live_edit_presence_sweep_oldest_stale_age_seconds`: the
    /// largest removal delay of a completed leader pass, or 0 from a replica
    /// that lost the leader probe. Never called for a pass that did not
    /// complete or a probe that failed (see the field doc). Carries this
    /// process's `service` label.
    pub fn record_presence_sweep_oldest_stale_age(&self, secs: f64) {
        self.presence_sweep_oldest_stale_age_seconds
            .record(secs, std::slice::from_ref(&self.service_kv));
    }

    /// Increment auth cache hits. `kind` should be `permissions` or `userinfo`.
    pub fn inc_auth_cache_hit(&self, kind: &str) {
        self.auth_cache_hits
            .add(1, &[KeyValue::new("kind", kind.to_string())]);
    }

    /// Increment auth cache misses. `kind` should be `permissions` or `userinfo`.
    pub fn inc_auth_cache_miss(&self, kind: &str) {
        self.auth_cache_misses
            .add(1, &[KeyValue::new("kind", kind.to_string())]);
    }

    /// Observe the duration of an auth backend call. `backend` is `storage` or
    /// `userinfo`, `protocol` is `rest` or `grpc`.
    pub fn observe_auth_call(&self, backend: &str, protocol: &str, duration: Duration) {
        self.auth_call_duration_seconds.record(
            duration.as_secs_f64(),
            &[
                KeyValue::new("backend", backend.to_string()),
                KeyValue::new("protocol", protocol.to_string()),
            ],
        );
    }

    /// Increment auth errors. `code` classifies the `AuthError` variant
    /// (`timeout`, `unreachable`, `invalid_response`).
    pub fn inc_auth_error(&self, backend: &str, code: &str) {
        self.auth_errors.add(
            1,
            &[
                KeyValue::new("backend", backend.to_string()),
                KeyValue::new("code", code.to_string()),
            ],
        );
    }

    /// Count one userinfo fallback: the userinfo service answered "no such
    /// user" and the display name came from the bearer's JWT claims instead.
    pub fn inc_userinfo_fallback(&self) {
        self.userinfo_fallbacks_total.add(1, &[]);
    }

    /// Add `count` to the `update_layer` retry counter for a given path and
    /// outcome. `count == 0` is a no-op (first-try successes / single-attempt
    /// exhaustions leave the counter untouched). `path` is one of
    /// [`UPDATE_LAYER_PATHS`] — the committing (or last) attempt's — and
    /// `outcome` one of [`UPDATE_LAYER_RETRY_OUTCOMES`].
    pub fn add_update_layer_retries(&self, path: &'static str, outcome: &'static str, count: u64) {
        debug_assert!(
            UPDATE_LAYER_PATHS.contains(&path),
            "unknown update_layer path label {path:?}"
        );
        debug_assert!(
            UPDATE_LAYER_RETRY_OUTCOMES.contains(&outcome),
            "unknown update_layer retry outcome label {outcome:?}"
        );
        self.update_layer_retries_total.add(
            count,
            &[
                KeyValue::new("path", path),
                KeyValue::new("outcome", outcome),
            ],
        );
    }

    /// Record one `update_layer` attempt that found its own call's commit
    /// already landed and reported it. `backend` is one of
    /// [`UPDATE_LAYER_BACKENDS`].
    pub fn inc_update_layer_commit_already_landed(&self, backend: &'static str) {
        debug_assert!(
            UPDATE_LAYER_BACKENDS.contains(&backend),
            "unknown update_layer backend label {backend:?}"
        );
        self.update_layer_commit_already_landed_total
            .add(1, &[KeyValue::new("backend", backend)]);
    }

    /// Record one lost-reply release lookup by an `UpdateLayer` call that
    /// ended without proof of its outcome. `outcome` is one of
    /// [`LOST_REPLY_RELEASE_LOOKUP_OUTCOMES`].
    pub fn inc_update_layer_lost_reply_release_lookup(&self, outcome: &'static str) {
        debug_assert!(
            LOST_REPLY_RELEASE_LOOKUP_OUTCOMES.contains(&outcome),
            "unknown lost-reply release lookup outcome label {outcome:?}"
        );
        self.update_layer_lost_reply_release_lookup_total
            .add(1, &[KeyValue::new("outcome", outcome)]);
    }

    /// Record one unchained operation-id `UpdateLayer` call overtaken: its
    /// `client_id`'s commit record on the object changed while it was in flight (another
    /// `UpdateLayer` from that `client_id` committed, or a checkpoint restore).
    /// `backend` is one of [`UPDATE_LAYER_BACKENDS`].
    pub fn inc_update_layer_operation_overtaken(&self, backend: &'static str) {
        debug_assert!(
            UPDATE_LAYER_BACKENDS.contains(&backend),
            "unknown update_layer backend label {backend:?}"
        );
        self.update_layer_operation_overtaken_total
            .add(1, &[KeyValue::new("backend", backend)]);
    }

    /// Record one call whose client `operation_id` matched an operation an
    /// earlier call already committed. `rpc` is one of [`OPERATION_ID_RPCS`],
    /// `outcome` one of [`OPERATION_ID_OUTCOMES`].
    pub fn inc_operation_id_match(&self, rpc: &'static str, outcome: &'static str) {
        debug_assert!(
            OPERATION_ID_RPCS.contains(&rpc),
            "unknown operation_id rpc label {rpc:?}"
        );
        debug_assert!(
            OPERATION_ID_OUTCOMES.contains(&outcome),
            "unknown operation_id outcome label {outcome:?}"
        );
        self.operation_id_match_total.add(
            1,
            &[KeyValue::new("rpc", rpc), KeyValue::new("outcome", outcome)],
        );
    }

    /// Observe the number of attempts one `update_layer` commit took (1 =
    /// first-try commit), once, where the call observes it land. `path` is the
    /// committing attempt's, one of [`UPDATE_LAYER_PATHS`].
    pub fn observe_update_layer_attempts(&self, path: &'static str, attempts: u32) {
        debug_assert!(
            UPDATE_LAYER_PATHS.contains(&path),
            "unknown update_layer path label {path:?}"
        );
        self.update_layer_attempts_histogram
            .record(attempts as f64, &[KeyValue::new("path", path)]);
    }

    /// Observe the relay OMPF bytes of one landed `UpdateLayer` commit on
    /// `live_edit_update_layer_relay_bytes` — once per commit, beside its
    /// [`Self::observe_update_layer_attempts`] sample, where the call counts
    /// the commit landed (see [`Self::update_layer_relay_bytes`]).
    pub fn observe_update_layer_relay_bytes(&self, relay_bytes: u64) {
        self.update_layer_relay_bytes
            .record(relay_bytes as f64, &[]);
    }

    /// Tick `live_edit_store_write_refused_total{stage, reason}`: one write
    /// the backing store refused before anything was applied — an upload
    /// write, or one sharing its stage (the DR restore's pre-store and a
    /// read's recovery rewrite at `ext_store`, the object-id counter repair
    /// writes at `create_commit`). `stage` is
    /// one of [`STORE_WRITE_REFUSED_STAGES`] and `reason` one of
    /// [`STORE_WRITE_REFUSED_REASONS`].
    pub fn inc_store_write_refused(&self, stage: &'static str, reason: &'static str) {
        debug_assert!(
            STORE_WRITE_REFUSED_STAGES.contains(&stage),
            "unknown store_write_refused stage label {stage:?}"
        );
        debug_assert!(
            STORE_WRITE_REFUSED_REASONS.contains(&reason),
            "unknown store_write_refused reason label {reason:?}"
        );
        self.store_write_refused_total.add(
            1,
            &[
                KeyValue::new("stage", stage),
                KeyValue::new("reason", reason),
            ],
        );
    }

    /// Observe one `update_layer` call's wall time, entry to return, drop or
    /// unwind. `path` is one of [`UPDATE_LAYER_PATHS`] (the last attempt's, or
    /// the call's classification when it ended — returned, was dropped or
    /// unwound — before any attempt) and `outcome` one of
    /// [`UPDATE_LAYER_COMMIT_OUTCOMES`]; a dropped or unwound call is `error`.
    pub fn observe_update_layer_commit_duration(
        &self,
        path: &'static str,
        outcome: &'static str,
        seconds: f64,
    ) {
        debug_assert!(
            UPDATE_LAYER_PATHS.contains(&path),
            "unknown update_layer path label {path:?}"
        );
        debug_assert!(
            UPDATE_LAYER_COMMIT_OUTCOMES.contains(&outcome),
            "unknown update_layer commit outcome label {outcome:?}"
        );
        self.update_layer_commit_duration_seconds.record(
            seconds,
            &[
                KeyValue::new("path", path),
                KeyValue::new("outcome", outcome),
            ],
        );
    }

    /// Record one `update_layer` attempt abandoned because the layer changed
    /// under it (see [`UPDATE_LAYER_CONFLICT_KINDS`]). `path` is the
    /// attempt's, one of [`UPDATE_LAYER_PATHS`]; `kind` one of
    /// [`UPDATE_LAYER_CONFLICT_KINDS`].
    pub fn inc_update_layer_conflict(&self, path: &'static str, kind: &'static str) {
        debug_assert!(
            UPDATE_LAYER_PATHS.contains(&path),
            "unknown update_layer path label {path:?}"
        );
        debug_assert!(
            UPDATE_LAYER_CONFLICT_KINDS.contains(&kind),
            "unknown update_layer conflict kind label {kind:?}"
        );
        self.update_layer_conflicts_total.add(
            1,
            &[KeyValue::new("path", path), KeyValue::new("kind", kind)],
        );
    }

    /// Record one structural `UpdateLayer` attempt that suspended to wait for
    /// a larger upload charge (`live_edit_update_layer_recharges_total`).
    /// `path` is the attempt's, one of [`UPDATE_LAYER_PATHS`].
    pub fn inc_update_layer_recharge(&self, path: &'static str) {
        debug_assert!(
            UPDATE_LAYER_PATHS.contains(&path),
            "unknown update_layer path label {path:?}"
        );
        self.update_layer_recharges_total
            .add(1, &[KeyValue::new("path", path)]);
    }

    /// Observe the object-state bytes one `UpdateLayer` call held when it
    /// returned its own commit (`live_edit_upload_object_state_bytes`).
    /// `path` is the committing attempt's, one of [`UPDATE_LAYER_PATHS`];
    /// the caller never observes `fast`, which holds no object-state term.
    pub fn observe_upload_object_state_bytes(&self, path: &'static str, bytes: u64) {
        debug_assert!(
            UPDATE_LAYER_PATHS.contains(&path),
            "unknown update_layer path label {path:?}"
        );
        self.upload_object_state_bytes
            .record(bytes as f64, &[KeyValue::new("path", path)]);
    }

    /// Record one structural `UpdateLayer` call that committed through the
    /// WATCH path instead of the structural cache. `reason` is one of
    /// [`UPDATE_LAYER_STRUCTURAL_FALLBACK_REASONS`].
    pub fn inc_update_layer_structural_fallback(&self, reason: &'static str) {
        debug_assert!(
            UPDATE_LAYER_STRUCTURAL_FALLBACK_REASONS.contains(&reason),
            "unknown update_layer structural fallback reason label {reason:?}"
        );
        self.update_layer_structural_fallbacks_total
            .add(1, &[KeyValue::new("reason", reason)]);
    }

    /// Observe one structural commit `EVALSHA`'s client-observed round trip
    /// ([`Self::update_layer_structural_commit_seconds`]).
    pub fn observe_update_layer_structural_commit(&self, elapsed: Duration) {
        self.update_layer_structural_commit_seconds
            .record(elapsed.as_secs_f64(), &[]);
    }

    /// Observe one `update_layer` call's total snapshot-fence wait. `backend`
    /// is one of [`UPDATE_LAYER_BACKENDS`].
    pub fn observe_update_layer_fence_wait(&self, backend: &'static str, waited: Duration) {
        debug_assert!(
            UPDATE_LAYER_BACKENDS.contains(&backend),
            "unknown update_layer backend label {backend:?}"
        );
        self.update_layer_fence_wait_seconds
            .record(waited.as_secs_f64(), &[KeyValue::new("backend", backend)]);
    }

    /// Record one writer-intent registration outcome. `result` is one of
    /// [`UPDATE_LAYER_FENCE_INTENT_RESULTS`].
    pub fn inc_update_layer_fence_intent(&self, result: &'static str) {
        debug_assert!(
            UPDATE_LAYER_FENCE_INTENT_RESULTS.contains(&result),
            "unknown fence-intent result label {result:?}"
        );
        self.update_layer_fence_intent_total
            .add(1, &[KeyValue::new("result", result)]);
    }

    /// Record one `update_layer` call failed at its fence-wait cap.
    pub fn inc_update_layer_fence_timeout(&self) {
        self.update_layer_fence_timeout_total.add(1, &[]);
    }

    /// Record one commit refused by the b.ph9 stream-ID collision guard.
    ///
    /// For the cells the update script does not raise: `backend` is `txn` and
    /// `site` is `update` (the WATCH/MULTI/EXEC commit path's probe) or
    /// `create` (`write_object`'s initial `1-0` entry). The update script's
    /// `{backend="lua", site="update"}` cell carries a `mode` label, so it is
    /// raised through [`Self::inc_update_script_stream_id_collision`] instead,
    /// never here.
    ///
    /// Not called for a collision verdict that
    /// `LayerStore::disambiguate_stream_collision` traces to a concurrent
    /// layer retirement: that is ordinary concurrency, and this counter's
    /// value rests on its baseline being zero.
    pub fn inc_stream_id_collision(&self, backend: &str, site: &str) {
        debug_assert!(
            !(backend == UPDATE_LAYER_BACKEND_LUA && site == "update"),
            "the lua/update stream_id_collision cell carries `mode`: use \
             inc_update_script_stream_id_collision"
        );
        self.stream_id_collision_total.add(
            1,
            &[
                KeyValue::new("backend", backend.to_string()),
                KeyValue::new("site", site.to_string()),
            ],
        );
    }

    /// Record one update-script commit refused by the b.ph9 stream-ID collision
    /// guard: the `{backend="lua", site="update", mode}` cell.
    ///
    /// `mode` is one of [`UPDATE_SCRIPT_MODES`] — the script mode whose call
    /// was refused (`storage::lua::UpdateScriptMode::metric_label`). The single
    /// call site is `LayerStore::map_update_script_error`, which both modes
    /// share. Same retirement carve-out as [`Self::inc_stream_id_collision`].
    pub fn inc_update_script_stream_id_collision(&self, mode: &'static str) {
        debug_assert!(
            UPDATE_SCRIPT_MODES.contains(&mode),
            "unknown update script mode label {mode:?}"
        );
        self.stream_id_collision_total.add(
            1,
            &[
                KeyValue::new("backend", UPDATE_LAYER_BACKEND_LUA),
                KeyValue::new("site", "update"),
                KeyValue::new("mode", mode),
            ],
        );
    }

    /// Record one commit refused because a key it was about to mutate — or, for
    /// a `CreateLayer` operation record, about to read — held the wrong Valkey
    /// type (b.459), on a cell the update script does not raise.
    ///
    /// `backend` is `txn` (every producer but the update script: the
    /// pre-`MULTI` keyspace type probe, the `CreateLayer` operation-record
    /// read, and the `CreateLayer` commit script); `site` is `update` or
    /// `create`. The update script's validation pre-pass raises the
    /// `{backend="lua", site="update"}` cell, which carries a `mode` label,
    /// through [`Self::inc_update_script_keyspace_type_error`] instead, never
    /// here. Every arm also logs at ERROR with the offending key, which is the
    /// remediation. See [`Self::keyspace_type_error_total`] for which producers
    /// write nothing and which one leaves an orphaned object.
    pub fn inc_keyspace_type_error(&self, backend: &str, site: &str) {
        debug_assert!(
            !(backend == UPDATE_LAYER_BACKEND_LUA && site == "update"),
            "the lua/update keyspace_type_error cell carries `mode`: use \
             inc_update_script_keyspace_type_error"
        );
        self.keyspace_type_error_total.add(
            1,
            &[
                KeyValue::new("backend", backend.to_string()),
                KeyValue::new("site", site.to_string()),
            ],
        );
    }

    /// Record one update-script commit refused because a key it was about to
    /// mutate held the wrong Valkey type (b.459): the
    /// `{backend="lua", site="update", mode}` cell.
    ///
    /// `mode` is one of [`UPDATE_SCRIPT_MODES`] — the script mode whose call
    /// was refused (`storage::lua::UpdateScriptMode::metric_label`). The single
    /// call site is `LayerStore::map_update_script_error`'s `WRONGTYPE_KEY`
    /// arm, which both modes share; it also logs at ERROR with the offending
    /// key.
    pub fn inc_update_script_keyspace_type_error(&self, mode: &'static str) {
        debug_assert!(
            UPDATE_SCRIPT_MODES.contains(&mode),
            "unknown update script mode label {mode:?}"
        );
        self.keyspace_type_error_total.add(
            1,
            &[
                KeyValue::new("backend", UPDATE_LAYER_BACKEND_LUA),
                KeyValue::new("site", "update"),
                KeyValue::new("mode", mode),
            ],
        );
    }

    /// Record one object-state touch refused because the key it addressed held
    /// the wrong Valkey type (b.atj).
    ///
    /// `key_family` is one of [`SNAPSHOT_READ_KEY_FAMILIES`]. The single call
    /// site (`storage::diff_read::count_object_key_refusal`, reached both from
    /// `classify_snapshot_read_err` and directly from the orphan sweep's two
    /// `obj:hdr` touches — its `HGETALL` discovery read and its retirement-fence
    /// write — since b.muu) passes `SnapshotKeyFamily::label()`;
    /// it also logs at ERROR with the offending key, which is the remediation.
    ///
    /// `&'static str`, not `&str` as on [`Self::inc_keyspace_type_error`]:
    /// tightening the parameter to the lifetime the bounded vocabulary actually
    /// has turns "callers pass a registered constant" from a doc-comment into a
    /// signature — a runtime-computed `String` cannot be passed at all. Same
    /// posture as [`Self::inc_valkey_consumer_forced_reconnect`], and it lets
    /// the label ride into `KeyValue` without an allocation.
    ///
    /// Distinct from [`Self::inc_keyspace_type_error`] on purpose: that one
    /// counts a refused *commit*, this one a refused *read*, and the service's
    /// own `WRONGTYPE_KEY:`-tagged Lua verdict never reaches here (different
    /// fred `ErrorKind` — see `storage::client::is_server_wrongtype_error`), so
    /// one refusal never ticks both.
    pub fn inc_snapshot_read_type_error(&self, key_family: &'static str) {
        self.snapshot_read_type_error_total
            .add(1, &[KeyValue::new("key_family", key_family)]);
    }

    /// Record one object-state touch refused by a permanent, non-`WRONGTYPE`
    /// backend failure (b.muu).
    ///
    /// `kind` is one of [`SNAPSHOT_READ_REFUSAL_KINDS`]. The single call site
    /// (`storage::diff_read::count_object_key_refusal`) passes
    /// `snapshot_refusal_kind_label`'s return, which cannot produce an
    /// unregistered value: its match over the third-party `fred::error::
    /// ErrorKind` ends in a `_` arm landing on
    /// [`SNAPSHOT_READ_REFUSAL_KIND_UNKNOWN`]. That call site also logs the
    /// object, the key and the failing command — at ERROR for the eleven
    /// should-never-happen kinds, at WARN for
    /// [`SNAPSHOT_READ_REFUSAL_KIND_BACKEND_UNAVAILABLE`], a store condition
    /// refusing the orphan sweep's fence write that can clear on its own.
    ///
    /// `&'static str` for the reason [`Self::inc_snapshot_read_type_error`]
    /// gives: the bounded vocabulary's lifetime in the signature means a
    /// runtime-computed `String` cannot be passed at all, and the label rides
    /// into `KeyValue` without an allocation.
    ///
    /// Never ticks for the same refusal as its sibling — the two arms at the
    /// call site are mutually exclusive and ordered, `WRONGTYPE` first.
    pub fn inc_snapshot_read_refused(&self, kind: &'static str) {
        self.snapshot_read_refused_total
            .add(1, &[KeyValue::new("kind", kind)]);
    }

    /// Record one structural WATCH-path snapshot read that fell back from its
    /// per-window read script to plain commands on a cluster-state refusal
    /// (`live_edit_snapshot_read_window_fallback_total`). Called once per
    /// read, by `storage::diff_read` when the read falls back.
    pub fn inc_snapshot_read_window_fallback(&self) {
        self.snapshot_read_window_fallback_total.add(1, &[]);
    }

    /// Record one completed object-snapshot read attempt (Spec SDD SR-8.1,
    /// SR-8.2): ticks `live_edit_snapshot_read_total{consumer, path, outcome}`
    /// and records its wall time on
    /// `live_edit_snapshot_read_duration_seconds{consumer, path}`.
    ///
    /// `consumer`, `path` and `outcome` are [`SNAPSHOT_READ_CONSUMERS`],
    /// [`SNAPSHOT_READ_PATHS`] and [`SNAPSHOT_READ_OUTCOMES`] values; the
    /// `&'static str` parameters keep a runtime-computed label out, as on
    /// [`Self::inc_snapshot_read_type_error`]. The callers are
    /// `storage::diff_read::read_object_snapshot_with_ceilings` (which
    /// `read_object_snapshot` delegates to), the fenced read,
    /// `storage::snapshot_fence::read::read_object_snapshot_fenced`, which
    /// pass `storage::diff_read`'s enums' labels, and external-value
    /// clean-up's reference walk (`storage::retirement`), which passes
    /// `ext_gc` and `unfenced`.
    pub fn observe_snapshot_read(
        &self,
        consumer: &'static str,
        path: &'static str,
        outcome: &'static str,
        duration: std::time::Duration,
    ) {
        let consumer_kv = KeyValue::new("consumer", consumer);
        let path_kv = KeyValue::new("path", path);
        self.snapshot_read_total.add(
            1,
            &[
                consumer_kv.clone(),
                path_kv.clone(),
                KeyValue::new("outcome", outcome),
            ],
        );
        self.snapshot_read_duration_seconds
            .record(duration.as_secs_f64(), &[consumer_kv, path_kv]);
    }

    /// Record the size of one successful object-snapshot read on
    /// `live_edit_snapshot_read_reply_bytes{consumer}` and
    /// `live_edit_snapshot_read_entries{consumer}` (Spec SDD SR-8.2).
    pub fn observe_snapshot_read_size(
        &self,
        consumer: &'static str,
        reply_bytes: u64,
        entries: u64,
    ) {
        let consumer_kv = [KeyValue::new("consumer", consumer)];
        self.snapshot_read_reply_bytes
            .record(reply_bytes as f64, &consumer_kv);
        self.snapshot_read_entries
            .record(entries as f64, &consumer_kv);
    }

    /// Record one cold start of the structural write-planning cache
    /// (structural-edits SDD SR-2.5): ticks
    /// `live_edit_structural_cold_start_reads_total{route}` and records its
    /// wall time on `live_edit_structural_cold_start_read_duration_seconds{route}`.
    ///
    /// `route` is a [`STRUCTURAL_COLD_START_ROUTES`] value. The single caller
    /// is `storage::structural_cache_cold_start`, which passes its
    /// `ColdStartRoute` enum's labels (for the read side's probe verdict too).
    pub fn observe_structural_cold_start(
        &self,
        route: &'static str,
        duration: std::time::Duration,
    ) {
        let route_kv = [KeyValue::new("route", route)];
        self.structural_cold_start_reads_total.add(1, &route_kv);
        self.structural_cold_start_read_duration_seconds
            .record(duration.as_secs_f64(), &route_kv);
    }

    /// Tick `live_edit_structural_cache_lookups_total{outcome}` (SR-2.4).
    /// `outcome` is a [`STRUCTURAL_CACHE_LOOKUP_OUTCOMES`] value.
    pub fn inc_structural_cache_lookup(&self, outcome: &'static str) {
        self.structural_cache_lookups_total
            .add(1, &[KeyValue::new("outcome", outcome)]);
    }

    /// Tick `live_edit_structural_cache_reloads_total{reason}` (SR-2.4).
    /// `reason` is a [`STRUCTURAL_CACHE_RELOAD_REASONS`] value.
    pub fn inc_structural_cache_reload(&self, reason: &'static str) {
        self.structural_cache_reloads_total
            .add(1, &[KeyValue::new("reason", reason)]);
    }

    /// Tick `live_edit_structural_cache_uncacheable_total{reason}`: one
    /// uncacheable verdict. `reason` is a
    /// [`STRUCTURAL_CACHE_UNCACHEABLE_REASONS`] value. The single caller is
    /// `storage::structural_cache_cold_start::report_uncacheable`, which also
    /// writes the verdict line (INFO; DEBUG for `reload_limit` and
    /// `fenced_read_failed`).
    pub fn inc_structural_cache_uncacheable(&self, reason: &'static str) {
        debug_assert!(
            STRUCTURAL_CACHE_UNCACHEABLE_REASONS.contains(&reason),
            "unknown structural cache uncacheable reason label {reason:?}"
        );
        self.structural_cache_uncacheable_total
            .add(1, &[KeyValue::new("reason", reason)]);
    }

    /// Record one repair of a structural cache entry (SR-2.4): the stream
    /// entries it replayed on `live_edit_structural_cache_replayed_entries`
    /// and the stream bytes it read on `live_edit_structural_cache_repair_bytes`.
    pub fn observe_structural_cache_repair(&self, replayed_entries: u64, bytes: u64) {
        self.structural_cache_replayed_entries
            .record(replayed_entries as f64, &[]);
        self.structural_cache_repair_bytes.record(bytes as f64, &[]);
    }

    /// Tick `live_edit_structural_cache_divergence_total` (SR-2.4): one
    /// structural commit refused with `CACHE_DIVERGED`.
    pub fn inc_structural_cache_divergence(&self) {
        self.structural_cache_divergence_total.add(1, &[]);
    }

    /// Tick `live_edit_structural_cache_audit_total{outcome}` (SR-2.4,
    /// SR-3.6): one sampled audit that ran to an end. `outcome` is a
    /// [`STRUCTURAL_CACHE_AUDIT_OUTCOMES`] value.
    pub fn inc_structural_cache_audit(&self, outcome: &'static str) {
        self.structural_cache_audit_total
            .add(1, &[KeyValue::new("outcome", outcome)]);
    }

    /// Tick `live_edit_structural_cache_audit_fenced_total{result}` (SR-3.6):
    /// one audit of a layer above the plain ceilings. `result` is a
    /// [`STRUCTURAL_CACHE_AUDIT_FENCED_RESULTS`] value.
    pub fn inc_structural_cache_audit_fenced(&self, result: &'static str) {
        debug_assert!(
            STRUCTURAL_CACHE_AUDIT_FENCED_RESULTS.contains(&result),
            "unknown audit fenced result {result:?}"
        );
        self.structural_cache_audit_fenced_total
            .add(1, &[KeyValue::new("result", result)]);
    }

    /// Record one fenced read attempt's lease hold on
    /// `live_edit_snapshot_fence_hold_seconds{consumer}` (Spec SDD SR-8.3).
    /// `consumer` is a [`SNAPSHOT_READ_CONSUMERS`] value.
    pub fn observe_snapshot_fence_hold(&self, consumer: &'static str, held: Duration) {
        self.snapshot_fence_hold_seconds
            .record(held.as_secs_f64(), &[KeyValue::new("consumer", consumer)]);
    }

    /// Count one `snapshot_fence_acquire.lua` reply to a fenced reader on
    /// `live_edit_snapshot_fence_acquire_total{result}` (Spec SDD SR-8.3).
    /// `result` is one of [`SNAPSHOT_FENCE_ACQUIRE_RESULTS`].
    pub fn inc_snapshot_fence_acquire(&self, result: &'static str) {
        debug_assert!(
            SNAPSHOT_FENCE_ACQUIRE_RESULTS.contains(&result),
            "unknown fence-acquire result label {result:?}"
        );
        self.snapshot_fence_acquire_total
            .add(1, &[KeyValue::new("result", result)]);
    }

    /// Record one fenced read attempt's wait for its lease on
    /// `live_edit_snapshot_fence_reader_wait_seconds{consumer}` (Spec SDD
    /// SR-8.3). `consumer` is a [`SNAPSHOT_READ_CONSUMERS`] value.
    pub fn observe_snapshot_fence_reader_wait(&self, consumer: &'static str, waited: Duration) {
        self.snapshot_fence_reader_wait_seconds
            .record(waited.as_secs_f64(), &[KeyValue::new("consumer", consumer)]);
    }

    /// Increment `valkey_reconnects_total{pool}`. `pool` is `subscriber` or
    /// `data`.
    pub fn inc_valkey_reconnect(&self, pool: &str) {
        self.valkey_reconnects_total
            .add(1, &[KeyValue::new("pool", pool.to_string())]);
    }

    /// Increment `valkey_router_restarts_total{pool}`. `pool` is one of
    /// [`VALKEY_POOLS`]. Called only by `storage::router_supervisor`, once per
    /// router task it restarts.
    pub fn inc_valkey_router_restart(&self, pool: &'static str) {
        debug_assert!(
            VALKEY_POOLS.contains(&pool),
            "unknown Valkey pool label {pool:?}"
        );
        self.valkey_router_restarts_total
            .add(1, &[KeyValue::new("pool", pool)]);
    }

    /// Increment `valkey_unresponsive_total{pool,verdict}`. `pool` is one of
    /// [`VALKEY_POOLS`], `verdict` one of [`VALKEY_UNRESPONSIVE_VERDICTS`].
    /// Called only by `storage::client`'s error hook, once per connection fred
    /// closed as unresponsive.
    pub fn inc_valkey_unresponsive(&self, pool: &str, verdict: &'static str) {
        debug_assert!(
            VALKEY_POOLS.contains(&pool),
            "unknown Valkey pool label {pool:?}"
        );
        debug_assert!(
            VALKEY_UNRESPONSIVE_VERDICTS.contains(&verdict),
            "unknown unresponsive verdict label {verdict:?}"
        );
        self.valkey_unresponsive_total.add(
            1,
            &[
                KeyValue::new("pool", pool.to_string()),
                KeyValue::new("verdict", verdict),
            ],
        );
    }

    /// Increment `valkey_member_replacements_total{pool,outcome}`. `pool` is
    /// one of [`VALKEY_REPLACEABLE_POOLS`], `outcome` one of
    /// [`VALKEY_MEMBER_REPLACEMENT_OUTCOMES`]. Called only by
    /// `storage::member_replacement`.
    pub fn inc_valkey_member_replacement(&self, pool: &str, outcome: &'static str) {
        debug_assert!(
            VALKEY_REPLACEABLE_POOLS.contains(&pool),
            "unknown replaceable Valkey pool label {pool:?}"
        );
        debug_assert!(
            VALKEY_MEMBER_REPLACEMENT_OUTCOMES.contains(&outcome),
            "unknown member replacement outcome label {outcome:?}"
        );
        self.valkey_member_replacements_total.add(
            1,
            &[
                KeyValue::new("pool", pool.to_string()),
                KeyValue::new("outcome", outcome),
            ],
        );
    }

    /// Increment `valkey_consumer_forced_reconnects_total{phase}` (b.9uq).
    /// `phase` is one of [`CONSUMER_FORCED_RECONNECT_PHASES`].
    pub fn inc_valkey_consumer_forced_reconnect(&self, phase: &'static str) {
        self.valkey_consumer_forced_reconnects_total
            .add(1, &[KeyValue::new("phase", phase)]);
    }

    /// Increment `valkey_consumer_unblocked_retries_total` (b.9yf) — one
    /// live `XREAD BLOCK` a reader saw end in `-UNBLOCKED` (since b.19i, only
    /// an unblock from outside the service) and re-planned from the durable
    /// cursors.
    pub fn inc_valkey_consumer_unblocked_retry(&self) {
        self.valkey_consumer_unblocked_retries_total.add(1, &[]);
    }

    /// Increment `valkey_consumer_permanent_exits_total{phase}` (b.c7a) — one
    /// stream-multiplexer reader give-up on a permanent fred error. `phase` is
    /// one of [`CONSUMER_PERMANENT_EXIT_PHASES`].
    pub fn inc_valkey_consumer_permanent_exit(&self, phase: &'static str) {
        self.valkey_consumer_permanent_exits_total
            .add(1, &[KeyValue::new("phase", phase)]);
    }

    pub fn add_subscribe_catchups_inflight(&self, delta: i64) {
        self.subscribe_catchups_inflight.add(delta, &[]);
    }

    pub fn add_subscribe_catchups_waiting(&self, delta: i64) {
        self.subscribe_catchups_waiting.add(delta, &[]);
    }

    /// `result` is [`CATCHUP_WAIT_RESULT_ACQUIRED`],
    /// [`CATCHUP_WAIT_RESULT_CANCELLED`] or [`CATCHUP_WAIT_RESULT_TIMED_OUT`].
    pub fn observe_subscribe_catchup_wait(&self, result: &'static str, waited: Duration) {
        self.subscribe_catchup_wait_seconds
            .record(waited.as_secs_f64(), &[KeyValue::new("result", result)]);
    }

    /// Move `live_edit_catchup_snapshot_reads_in_flight{class}` by `delta`.
    /// `class` is one of [`CATCHUP_SNAPSHOT_READ_CLASSES`].
    pub fn add_catchup_snapshot_reads_in_flight(&self, class: &'static str, delta: i64) {
        self.catchup_snapshot_reads_in_flight
            .add(delta, &[KeyValue::new("class", class)]);
    }

    /// Record one read-permit wait on
    /// `live_edit_catchup_snapshot_read_wait_seconds{consumer, class}`.
    /// `consumer` is a [`SNAPSHOT_READ_CONSUMERS`] value, `class` one of
    /// [`CATCHUP_SNAPSHOT_READ_CLASSES`] (the gate's permit class).
    pub fn observe_catchup_snapshot_read_wait(
        &self,
        consumer: &'static str,
        class: &'static str,
        waited: Duration,
    ) {
        debug_assert!(
            CATCHUP_SNAPSHOT_READ_CLASSES.contains(&class),
            "unknown read-permit class label {class:?}"
        );
        self.catchup_snapshot_read_wait_seconds.record(
            waited.as_secs_f64(),
            &[
                KeyValue::new("consumer", consumer),
                KeyValue::new("class", class),
            ],
        );
    }

    /// Move `live_edit_catchup_build_budget_reserved_bytes` by `delta` bytes.
    pub fn add_catchup_build_budget_reserved_bytes(&self, delta: i64) {
        self.catchup_build_budget_reserved_bytes.add(delta, &[]);
    }

    /// Move `live_edit_catchup_build_budget_overcommit_bytes` by `delta` bytes.
    pub fn add_catchup_build_budget_overcommit_bytes(&self, delta: i64) {
        self.catchup_build_budget_overcommit_bytes.add(delta, &[]);
    }

    /// Record one waiting memory reservation's wait on
    /// `live_edit_catchup_build_wait_seconds`.
    pub fn observe_catchup_build_wait(&self, waited: Duration) {
        self.catchup_build_wait_seconds
            .record(waited.as_secs_f64(), &[]);
    }

    /// Count one memory-budget admission decision on
    /// `live_edit_catchup_build_admission_total{result}`. `result` is one of
    /// [`CATCHUP_BUILD_ADMISSION_RESULTS`].
    pub fn inc_catchup_build_admission(&self, result: &'static str) {
        self.catchup_build_admission_total
            .add(1, &[KeyValue::new("result", result)]);
    }

    /// Move `live_edit_retirement_budget_reserved_bytes` by `delta` bytes.
    pub fn add_retirement_budget_reserved_bytes(&self, delta: i64) {
        self.retirement_budget_reserved_bytes.add(delta, &[]);
    }

    /// Move `live_edit_retirement_budget_overcommit_bytes` by `delta` bytes.
    pub fn add_retirement_budget_overcommit_bytes(&self, delta: i64) {
        self.retirement_budget_overcommit_bytes.add(delta, &[]);
    }

    /// Record one retirement reservation's wait on
    /// `live_edit_retirement_wait_seconds{outcome}`. `outcome` is one of
    /// [`RETIREMENT_WAIT_OUTCOMES`].
    pub fn observe_retirement_wait(&self, waited: Duration, outcome: &'static str) {
        self.retirement_wait_seconds
            .record(waited.as_secs_f64(), &[KeyValue::new("outcome", outcome)]);
    }

    /// Move `live_edit_catchup_builds_in_progress` by `delta`.
    pub fn add_catchup_builds_in_progress(&self, delta: i64) {
        self.catchup_builds_in_progress.add(delta, &[]);
    }

    /// Count one catch-up's part in a shared snapshot-read flight on
    /// `live_edit_catchup_snapshot_flight_total{role}` (Spec SDD SR-8.4).
    /// `role` is one of [`CATCHUP_SNAPSHOT_FLIGHT_ROLES`].
    pub fn inc_catchup_snapshot_flight(&self, role: &'static str) {
        debug_assert!(
            CATCHUP_SNAPSHOT_FLIGHT_ROLES.contains(&role),
            "unknown snapshot-flight role label {role:?}"
        );
        self.catchup_snapshot_flight_total
            .add(1, &[KeyValue::new("role", role)]);
    }

    /// Move `live_edit_subscriber_admitted` and its `/scaling/v1` mirror by
    /// `delta`. `SubscriberAdmission`'s creation and drop are the only callers,
    /// so the two move together on every exit path.
    pub fn add_subscriber_admitted(&self, delta: i64) {
        self.subscriber_admitted.add(delta, &[]);
        self.scaling_counters
            .subscriptions_admitted
            .fetch_add(delta, std::sync::atomic::Ordering::AcqRel);
    }

    /// A subscription's first state-reading catch-up was selected: +1 on
    /// `live_edit_subscribe_catchups_pending` and its `/scaling/v1` mirror.
    /// Pair every call with exactly one [`Self::end_pending_catchup`].
    pub fn begin_pending_catchup(&self) {
        self.subscribe_catchups_pending.add(1, &[]);
        self.scaling_counters
            .catchups_pending
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }

    /// A pending first catch-up ended: −1 on the pending gauge and its mirror,
    /// and, when it ended with its caught-up ack enqueued (`completed`), +1 on
    /// the completed count `/scaling/v1` serves.
    pub fn end_pending_catchup(&self, completed: bool) {
        self.subscribe_catchups_pending.add(-1, &[]);
        self.scaling_counters
            .catchups_pending
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        if completed {
            self.scaling_counters
                .catchups_completed
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
    }

    /// The counters `/scaling/v1` serves.
    pub fn scaling_counters(&self) -> &ScalingCounters {
        &self.scaling_counters
    }

    pub fn add_stream_mux_subscriptions(&self, delta: i64) {
        self.stream_mux_subscriptions.add(delta, &[]);
    }

    pub fn add_stream_mux_streams(&self, delta: i64) {
        self.stream_mux_streams.add(delta, &[]);
    }

    pub fn inc_stream_mux_xread(&self) {
        self.stream_mux_xreads_total.add(1, &[]);
    }

    /// `outcome` is one of [`WAKE_OUTCOMES`].
    pub fn inc_stream_mux_wake(&self, outcome: &'static str) {
        self.stream_mux_wakes_total
            .add(1, &[KeyValue::new("outcome", outcome)]);
    }

    /// `result` is one of [`KEEPALIVE_RESULTS`].
    pub fn inc_stream_mux_keepalive(&self, result: &'static str) {
        self.stream_mux_keepalives_total
            .add(1, &[KeyValue::new("result", result)]);
    }

    /// Increment `stream_mux_reauths_total{result}` (b.19i). `result` is one
    /// of [`REAUTH_RESULTS`].
    pub fn inc_stream_mux_reauth(&self, result: &'static str) {
        self.stream_mux_reauths_total
            .add(1, &[KeyValue::new("result", result)]);
    }

    /// `cause` is one of [`READER_CONNECT_CAUSES`], `result` one of
    /// [`READER_CONNECT_RESULTS`].
    pub fn inc_stream_mux_reader_connect(&self, cause: &'static str, result: &'static str) {
        self.stream_mux_reader_connects_total.add(
            1,
            &[
                KeyValue::new("cause", cause),
                KeyValue::new("result", result),
            ],
        );
    }

    pub fn add_stream_mux_readers_connected(&self, delta: i64) {
        self.stream_mux_readers_connected.add(delta, &[]);
    }

    pub fn inc_stream_mux_bind_refusal(&self) {
        self.stream_mux_bind_refusals_total.add(1, &[]);
    }

    pub fn add_stream_mux_poison_keys(&self, n: u64) {
        self.stream_mux_poison_keys_total.add(n, &[]);
    }

    /// `bound` is one of [`OVERFLOW_BOUNDS`].
    pub fn add_stream_mux_overflow_kills(&self, bound: &'static str, n: u64) {
        self.stream_mux_overflow_kills_total
            .add(n, &[KeyValue::new("bound", bound)]);
    }

    /// `result` is one of [`HEAD_CHECK_RESULTS`].
    pub fn inc_stream_mux_head_check(&self, result: &'static str) {
        self.stream_mux_head_checks_total
            .add(1, &[KeyValue::new("result", result)]);
    }

    /// Records one multiplexed XREAD reply's size on
    /// `live_edit_stream_mux_xread_reply_bytes`.
    pub fn observe_stream_mux_xread_reply_bytes(&self, bytes: u64) {
        self.stream_mux_xread_reply_bytes.record(bytes as f64, &[]);
    }

    /// Increment `valkey_errors_total{pool, kind}` for a fred connection-level
    /// error. `kind` is the fred `ErrorKind::to_str()` value.
    pub fn inc_valkey_error(&self, pool: &str, kind: &str) {
        self.valkey_errors_total.add(
            1,
            &[
                KeyValue::new("pool", pool.to_string()),
                KeyValue::new("kind", kind.to_string()),
            ],
        );
    }

    /// Increment `valkey_errors_total{pool, kind="command_timeout"}` — one
    /// command whose client-side timeout expired, observed by service code.
    /// `pool` is one of [`VALKEY_POOLS`]. Called only from
    /// `storage::client::FredErrOrigin::observe_error`.
    pub fn inc_valkey_command_timeout(&self, pool: &'static str) {
        self.valkey_errors_total.add(
            1,
            &[
                KeyValue::new("pool", pool),
                KeyValue::new("kind", VALKEY_ERROR_KIND_COMMAND_TIMEOUT),
            ],
        );
    }

    /// Increment `storage_watch_unwatch_failures_total{site}` (b.ji8) — one
    /// `WatchGuard` teardown that gave up on its slot-pinned UNWATCH, releasing
    /// a `txn`-pool connection with stale WATCH state. `site` is one of
    /// [`UNWATCH_SITE_EXPLICIT`] or [`UNWATCH_SITE_DROP`].
    pub fn inc_storage_watch_unwatch_failure(&self, site: &'static str) {
        self.storage_watch_unwatch_failures_total
            .add(1, &[KeyValue::new("site", site)]);
    }

    /// Increment `storage_watch_voided_total{path, disposition}` (b.bhb) — one
    /// watched transaction whose `WATCH` was found void before dispatch, or
    /// proven void afterwards by the in-transaction `CLIENT ID` witness.
    ///
    /// `path` is one of [`WATCH_VOIDED_PATHS`], `disposition` one of
    /// [`WATCH_VOIDED_DISPOSITIONS`]. Only `committed_unwatched` describes a
    /// consequence that reached a client — see the field's doc comment.
    pub fn inc_storage_watch_voided(&self, path: &'static str, disposition: &'static str) {
        self.storage_watch_voided_total.add(
            1,
            &[
                KeyValue::new("path", path),
                KeyValue::new("disposition", disposition),
            ],
        );
    }

    /// Increment `valkey_txn_dirty_checkouts_total{reason}` (b.7td) — one
    /// `txn`-pool checkout whose hygiene probe found the slot unfit for a
    /// transaction. `reason` is one of [`TXN_DIRTY_REASONS`].
    ///
    /// Records the **verdict**, not the repair: it ticks whether or not the
    /// recycle was admitted, because "the probe found this slot unfit" is true
    /// either way. Whether a repair actually ran is carried by the log event's
    /// `suppressed` field and, for the state-proof breaker, by
    /// [`Self::inc_valkey_txn_repair_ineffective`].
    pub fn inc_valkey_txn_dirty_checkout(&self, reason: &'static str) {
        self.valkey_txn_dirty_checkouts_total
            .add(1, &[KeyValue::new("reason", reason)]);
    }

    /// Increment `valkey_txn_repair_ineffective_total{reason}` (b.7td round 3)
    /// — one state-proof dirty verdict whose recycle was withheld because
    /// rebuilding the connection had already been shown not to clear it.
    ///
    /// `reason` is one of [`TXN_STATE_PROOF_REASONS`]; the unknown-state
    /// verdicts have their own rate limit and never reach this breaker.
    pub fn inc_valkey_txn_repair_ineffective(&self, reason: &'static str) {
        self.valkey_txn_repair_ineffective_total
            .add(1, &[KeyValue::new("reason", reason)]);
    }

    /// Increment the in-use gauge for `pool` (`txn` | `subscriber_pool` |
    /// `maintenance` | `snapshot`). Paired with [`Self::dec_valkey_pool_in_use`]; both are
    /// driven by `storage::client::PoolUseToken`.
    pub fn inc_valkey_pool_in_use(&self, pool: &str) {
        self.valkey_pool_in_use
            .add(1, &[KeyValue::new("pool", pool.to_string())]);
    }

    /// Decrement the in-use gauge for `pool`.
    pub fn dec_valkey_pool_in_use(&self, pool: &str) {
        self.valkey_pool_in_use
            .add(-1, &[KeyValue::new("pool", pool.to_string())]);
    }

    /// Increment the waiter gauge for one of [`CHECKOUT_POOLS`] (b.f4f): the
    /// `txn` `ExclusivePool`'s permit queue, or the `SubscribeReadLayer`
    /// admission queue (`subscriber_pool`). Paired with
    /// [`Self::dec_valkey_pool_waiters`]; each queue drives both from one RAII
    /// guard (`storage::checkout`, `storage::subscriber_admission`), so a
    /// cancelled RPC's dropped future still decrements.
    pub fn inc_valkey_pool_waiters(&self, pool: &'static str) {
        self.valkey_pool_waiters
            .add(1, &[KeyValue::new("pool", pool)]);
    }

    /// Decrement the waiter gauge for one of [`CHECKOUT_POOLS`] (b.f4f).
    pub fn dec_valkey_pool_waiters(&self, pool: &'static str) {
        self.valkey_pool_waiters
            .add(-1, &[KeyValue::new("pool", pool)]);
    }

    /// Record one exclusive-pool checkout's latency (b.f4f).
    pub fn observe_valkey_pool_checkout(&self, pool: &'static str, elapsed: Duration) {
        self.valkey_pool_checkout_seconds
            .record(elapsed.as_secs_f64(), &[KeyValue::new("pool", pool)]);
    }

    /// Record one `SubscribeReadLayer` admission decision and how long it waited
    /// (b.f4f). `result` must come from [`ADMISSION_RESULTS`].
    pub fn observe_subscriber_admission(&self, result: &'static str, waited: Duration) {
        self.subscriber_admission_total
            .add(1, &[KeyValue::new("result", result)]);
        self.subscriber_admission_wait_seconds
            .record(waited.as_secs_f64(), &[]);
    }

    /// Record one upload admission decision and how long it waited. `rpc` must
    /// come from [`UPLOAD_RPCS`] and `result` from [`UPLOAD_ADMISSION_RESULTS`].
    pub fn observe_upload_admission(
        &self,
        rpc: &'static str,
        result: &'static str,
        waited: Duration,
    ) {
        self.upload_admission_total.add(
            1,
            &[KeyValue::new("rpc", rpc), KeyValue::new("result", result)],
        );
        self.upload_admission_wait_seconds
            .record(waited.as_secs_f64(), &[KeyValue::new("rpc", rpc)]);
    }

    /// Record one decision on an admitted upload with no admission wait to
    /// observe: a refused receive growth (`refused_growth`), or a processing
    /// wait ended by shutdown or a dropped call (`cancelled`).
    pub fn inc_upload_admission_refusal(&self, rpc: &'static str, result: &'static str) {
        self.upload_admission_total.add(
            1,
            &[KeyValue::new("rpc", rpc), KeyValue::new("result", result)],
        );
    }

    pub fn add_upload_reserved_bytes(&self, delta: i64) {
        self.upload_reserved_bytes.add(delta, &[]);
    }

    /// `phase` must come from [`UPLOAD_PHASES`].
    pub fn add_uploads_in_flight(&self, phase: &'static str, delta: i64) {
        self.uploads_in_flight
            .add(delta, &[KeyValue::new("phase", phase)]);
    }

    pub fn add_upload_admission_waiters(&self, delta: i64) {
        self.upload_admission_waiters.add(delta, &[]);
    }

    pub fn set_upload_budget_bytes(&self, bytes: u64) {
        self.upload_budget_bytes.record(bytes, &[]);
    }

    /// The overflow slot holder's charge, or 0 when the slot frees.
    pub fn set_upload_overflow_bytes(&self, bytes: u64) {
        self.upload_overflow_bytes.record(bytes, &[]);
    }

    pub fn add_upload_overflow_waiters(&self, delta: i64) {
        self.upload_overflow_waiters.add(delta, &[]);
    }

    /// One upload's processing step took the overflow slot. `rpc` must come
    /// from [`UPLOAD_RPCS`].
    pub fn inc_upload_overflow_runs(&self, rpc: &'static str) {
        self.upload_overflow_runs_total
            .add(1, &[KeyValue::new("rpc", rpc)]);
    }

    /// Record one processing step's wait. `rpc` must come from
    /// [`UPLOAD_RPCS`].
    pub fn observe_upload_processing_wait(&self, rpc: &'static str, waited: Duration) {
        self.upload_processing_wait_seconds
            .record(waited.as_secs_f64(), &[KeyValue::new("rpc", rpc)]);
    }

    /// Record one chained `UpdateLayer`'s chain-gate wait at its exit.
    /// `outcome` must come from [`CHAIN_OUTCOMES`].
    pub fn observe_update_layer_chain_wait(&self, outcome: &'static str, waited: Duration) {
        self.update_layer_chain_wait_seconds
            .record(waited.as_secs_f64(), &[KeyValue::new("outcome", outcome)]);
    }

    /// Count one chain-gate probe. `result` must come from
    /// [`CHAIN_PROBE_RESULTS`].
    pub fn inc_update_layer_chain_probe(&self, result: &'static str) {
        self.update_layer_chain_probes_total
            .add(1, &[KeyValue::new("result", result)]);
    }

    /// A chained `UpdateLayer` entered (`+1`, `+bytes`) or left (`-1`,
    /// `-bytes`) the chain gate.
    pub fn add_update_layer_chain_waiters(&self, delta: i64, bytes: i64) {
        self.update_layer_chain_waiters.add(delta, &[]);
        self.update_layer_chain_waiting_bytes.add(bytes, &[]);
    }

    /// Record one chain-gate detection lag (a `ready` exit after a sleep).
    pub fn observe_update_layer_chain_detect_lag(&self, lag: Duration) {
        self.update_layer_chain_detect_lag_seconds
            .record(lag.as_secs_f64(), &[]);
    }

    /// Record one `SubscribeReadLayer` subscription ending (b.f4f). `reason`
    /// must come from [`STREAM_EXIT_REASONS`], and exactly one increment fires
    /// per established subscription — that is what makes `sum by (reason)` the
    /// total.
    pub fn inc_subscriber_stream_exit(&self, reason: &'static str) {
        self.subscriber_stream_exits_total
            .add(1, &[KeyValue::new("reason", reason)]);
    }

    /// Record one window's maximum commit-to-hand-off lag (see
    /// [`Self::subscriber_delivery_lag_seconds`]). `stage` must come from
    /// [`DELIVERY_LAG_STAGES`]. Callers record one window maximum per
    /// subscription, never every delivery, so subscriber tasks under fan-out do
    /// not contend on one histogram cell per commit.
    pub fn observe_subscriber_delivery_lag(&self, stage: &'static str, lag: Duration) {
        self.subscriber_delivery_lag_seconds
            .record(lag.as_secs_f64(), &[KeyValue::new("stage", stage)]);
    }

    /// Count one send that found the response channel's data slots full
    /// ([`Self::subscriber_send_stalls_total`]). `phase` must come from
    /// [`SEND_PHASES`].
    pub fn inc_subscriber_send_stall(&self, phase: &'static str) {
        self.subscriber_send_stalls_total
            .add(1, &[KeyValue::new("phase", phase)]);
    }

    /// Record how long one stalled send waited
    /// ([`Self::subscriber_send_wait_seconds`]). `phase` must come from
    /// [`SEND_PHASES`].
    pub fn observe_subscriber_send_wait(&self, phase: &'static str, waited: Duration) {
        self.subscriber_send_wait_seconds
            .record(waited.as_secs_f64(), &[KeyValue::new("phase", phase)]);
    }

    /// Record one window's deepest stream-multiplexer backlog sample, in
    /// entries and in bytes ([`Self::subscriber_backlog_entries`],
    /// [`Self::subscriber_backlog_bytes`]).
    pub fn observe_subscriber_backlog(&self, entries: u64, bytes: u64) {
        self.subscriber_backlog_entries.record(entries as f64, &[]);
        self.subscriber_backlog_bytes.record(bytes as f64, &[]);
    }

    /// Increment the background-task panic counter for a given task name.
    pub fn inc_background_task_panic(&self, task: &str) {
        self.background_task_panics_total
            .add(1, &[KeyValue::new("task", task.to_string())]);
    }
}

/// Convenience alias used throughout the codebase.
pub type SharedMetrics = Arc<Metrics>;

/// Translate a `grpc-status` header value to the textual code slug used as
/// the `grpc_code` metric label (b.avd item 4). Absence (`None`) means the
/// RPC completed with `OK` — tonic omits the trailer on success. Any value
/// outside the documented 0..=16 range collapses to `"UNKNOWN"` so the
/// label set stays bounded by `GRPC_CODES`.
pub fn grpc_code_slug(grpc_status: Option<&str>) -> &'static str {
    let raw = match grpc_status {
        None => return "OK",
        Some(s) => s,
    };
    match raw.trim().parse::<i32>() {
        Ok(0) => "OK",
        Ok(1) => "CANCELLED",
        Ok(2) => "UNKNOWN",
        Ok(3) => "INVALID_ARGUMENT",
        Ok(4) => "DEADLINE_EXCEEDED",
        Ok(5) => "NOT_FOUND",
        Ok(6) => "ALREADY_EXISTS",
        Ok(7) => "PERMISSION_DENIED",
        Ok(8) => "RESOURCE_EXHAUSTED",
        Ok(9) => "FAILED_PRECONDITION",
        Ok(10) => "ABORTED",
        Ok(11) => "OUT_OF_RANGE",
        Ok(12) => "UNIMPLEMENTED",
        Ok(13) => "INTERNAL",
        Ok(14) => "UNAVAILABLE",
        Ok(15) => "DATA_LOSS",
        Ok(16) => "UNAUTHENTICATED",
        _ => "UNKNOWN",
    }
}

/// Test-only helpers for asserting against the rendered Prometheus scrape
/// exposition produced by the OpenTelemetry `PrometheusExporter` (OMPE-98174).
///
/// Available to in-crate unit tests (`cfg(test)`) and to external
/// integration-test crates via the `test-support` feature — the same gating
/// idiom `storage::test_support` uses. Post-migration the metric internals are
/// OTel instruments with no `prometheus::Registry` to `gather()`, so tests that
/// need to read a metric value render `/metrics` text through [`render`] (the
/// exact path the HTTP handler drives) and parse a sample out of it with
/// [`sample_value`] / [`histogram_bucket`] instead of poking the old typed
/// metric fields.
#[cfg(any(test, feature = "test-support"))]
pub mod test_util {
    use super::Metrics;

    /// Render the current Prometheus text exposition for `metrics` by driving
    /// the always-attached scrape exporter's `export()` — the exact code path
    /// the `/metrics` HTTP handler uses. Firing `export()` runs a fresh SDK
    /// collect, so observable-gauge callbacks execute and the returned text
    /// reflects live instrument state.
    pub fn render(metrics: &Metrics) -> String {
        let mut buf = Vec::new();
        metrics
            .prometheus_exporter()
            .export(&mut buf)
            .expect("PrometheusExporter::export must succeed");
        String::from_utf8(buf).expect("Prometheus exposition must be valid UTF-8")
    }

    /// Every sample line of rendered exposition `text` whose metric name is
    /// exactly `name` and whose label set contains every `(key, value)` pair
    /// in `labels` (subset match, order-independent), in exposition order,
    /// each as its parsed value (`None` for a value that does not parse).
    /// Handles both labelled (`name{...} v`) and bare (`name v`) series. The
    /// one matcher [`sample_value`] and [`sample_sum`] share.
    fn matching_samples<'a>(
        text: &'a str,
        name: &'a str,
        labels: &'a [(&'a str, &'a str)],
    ) -> impl Iterator<Item = Option<f64>> + 'a {
        text.lines().filter_map(move |line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let (series, value) = line.rsplit_once(char::is_whitespace)?;
            let (series_name, label_body) = match series.split_once('{') {
                Some((n, rest)) => (n, rest.strip_suffix('}').unwrap_or(rest)),
                None => (series, ""),
            };
            if series_name != name {
                return None;
            }
            labels
                .iter()
                .all(|(k, v)| label_body.contains(&format!("{k}=\"{v}\"")))
                .then(|| value.trim().parse::<f64>().ok())
        })
    }

    /// Parse the value of a single sample line out of rendered exposition
    /// `text`. Matches the first series line whose metric name is exactly
    /// `name` and whose label set contains every `(key, value)` pair in
    /// `labels` (subset match, order-independent). Returns `None` when no such
    /// line exists. Handles both labelled (`name{...} v`) and bare (`name v`)
    /// series.
    pub fn sample_value(text: &str, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
        matching_samples(text, name, labels).next().flatten()
    }

    /// The sum of every sample line in `text` that [`sample_value`]'s subset
    /// match accepts, rather than the first one. `None` when no series
    /// matches (or none of the matching values parses).
    ///
    /// For reading a cell that a finer label splits into several series, as
    /// a `sum` over that label would in PromQL: e.g. the
    /// `{backend="lua", site="update"}` total of
    /// `live_edit_keyspace_type_error_total` across its `mode` series.
    pub fn sample_sum(text: &str, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
        matching_samples(text, name, labels)
            .flatten()
            .fold(None, |total, value| Some(total.unwrap_or(0.0) + value))
    }

    /// The cumulative count in a histogram bucket with the given `le`
    /// upper-bound string (e.g. `"128"`, `"+Inf"`), optionally further
    /// constrained by `labels`. Convenience wrapper over [`sample_value`]
    /// against the `<name>_bucket` series.
    pub fn histogram_bucket(
        text: &str,
        name: &str,
        le: &str,
        labels: &[(&str, &str)],
    ) -> Option<f64> {
        let mut all: Vec<(&str, &str)> = labels.to_vec();
        all.push(("le", le));
        sample_value(text, &format!("{name}_bucket"), &all)
    }

    /// The `# TYPE` declared for `name` — `"counter"`, `"gauge"`, or
    /// `"histogram"`. `None` when the family is absent from the exposition
    /// (an OTel instrument with no recorded measurement emits no family).
    pub fn metric_type(text: &str, name: &str) -> Option<String> {
        let needle = format!("# TYPE {name} ");
        text.lines()
            .find_map(|l| l.strip_prefix(&needle))
            .map(|t| t.trim().to_string())
    }

    /// Whether any family for `name` is present in the exposition.
    pub fn has_metric(text: &str, name: &str) -> bool {
        metric_type(text, name).is_some()
    }
}

/// Build a [`Metrics`] with both OTLP endpoint env vars pinned to `None` for the
/// duration of construction — the env-guarded path every in-crate unit test must
/// use to build a `Metrics`.
///
/// `Metrics::new` consults `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` (and, on the
/// tracing side, `OTEL_EXPORTER_OTLP_ENDPOINT`) at construction. If a sibling
/// test or a leaked CI env var has either set, an unguarded `Metrics::new` in a
/// sync unit test would build a live tonic OTLP push exporter — panicking for
/// lack of a tokio reactor, or silently attaching a real OTLP push to a
/// collector during the test. Pinning both vars to `None` inside a `temp_env`
/// guard is the single sanctioned construction path; every `#[cfg(test)]` call
/// site in `src/auth`, `src/checkpoint`, `src/grpc`, and `src/observability`
/// routes through it rather than calling `Metrics::new` directly. Mirrors the
/// tracing-side guard (b.5yt) and the `tests/common` integration helper.
///
/// `#[cfg(test)]`-only (not `test-support`): `temp-env` is a dev-dependency, so
/// this helper is compiled solely for the crate's own test build.
#[cfg(test)]
pub(crate) fn metrics_scrape_only(label: &str) -> Metrics {
    temp_env::with_vars(
        [
            ("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", None::<&str>),
            ("OTEL_EXPORTER_OTLP_ENDPOINT", None::<&str>),
        ],
        || Metrics::new(label),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
    use std::sync::Arc;

    use super::test_util::{has_metric, histogram_bucket, metric_type, render, sample_value};
    use super::*;

    /// Env var names for the OTLP-metrics-push gating tests.
    const METRICS_ENDPOINT_VAR: &str = "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT";
    const GENERIC_ENDPOINT_VAR: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";

    // ── grpc_code_slug (pure) ───────────────────────────────────────────

    #[test]
    fn grpc_code_slug_none_is_ok() {
        assert_eq!(grpc_code_slug(None), "OK");
    }

    #[test]
    fn grpc_code_slug_known_numeric_codes_map_correctly() {
        let cases = [
            ("0", "OK"),
            ("1", "CANCELLED"),
            ("2", "UNKNOWN"),
            ("3", "INVALID_ARGUMENT"),
            ("4", "DEADLINE_EXCEEDED"),
            ("5", "NOT_FOUND"),
            ("6", "ALREADY_EXISTS"),
            ("7", "PERMISSION_DENIED"),
            ("8", "RESOURCE_EXHAUSTED"),
            ("9", "FAILED_PRECONDITION"),
            ("10", "ABORTED"),
            ("11", "OUT_OF_RANGE"),
            ("12", "UNIMPLEMENTED"),
            ("13", "INTERNAL"),
            ("14", "UNAVAILABLE"),
            ("15", "DATA_LOSS"),
            ("16", "UNAUTHENTICATED"),
        ];
        for (raw, expected) in cases {
            assert_eq!(grpc_code_slug(Some(raw)), expected, "slug for {raw:?}");
        }
    }

    #[test]
    fn grpc_code_slug_out_of_range_collapses_to_unknown() {
        for raw in ["999", "-1", "garbage", "", "1.0"] {
            assert_eq!(grpc_code_slug(Some(raw)), "UNKNOWN", "slug for {raw:?}");
        }
    }

    #[test]
    fn grpc_code_slug_only_returns_strings_in_grpc_codes_list() {
        for raw in [None, Some("0"), Some("5"), Some("13"), Some("999")] {
            let slug = grpc_code_slug(raw);
            assert!(
                GRPC_CODES.contains(&slug),
                "grpc_code_slug({raw:?}) returned {slug:?} which is not in GRPC_CODES"
            );
        }
    }

    // ── Load-bearing parity guard: names / types / labels / buckets ─────
    //
    // This is the single most important regression test for the OMPE-98174
    // migration: it renders the scrape output and pins the wire shape the
    // pre-migration `prometheus`-crate path produced. It fails if the exporter
    // is ever built without the `without_*` name-mangling knobs (which would
    // reintroduce `otel_scope_*`, `target_info`, `_total_total`, or a unit
    // suffix) or if the histogram bucket view is dropped.

    #[test]
    fn scrape_output_preserves_pre_migration_names_types_and_buckets() {
        let metrics = metrics_scrape_only("test-pod");
        // Exercise one instrument of each kind so every family surfaces.
        metrics.inc_calls("Ping", "test-pod", "OK"); // Counter
        metrics.inc_calls("Ping", "test-pod", "OK");
        metrics.inc_inflight_rpc("Ping", "test-pod"); // UpDownCounter → gauge
        metrics.observe_duration_us("test_rpc", "test-pod", "OK", 999.0); // Histogram
        metrics.register_replica_load_gauges("test-pod", || 1, || 1, || 0); // ObservableGauge → gauge

        let text = render(&metrics);

        // 1) No OTel-synthetic series / label pollution.
        assert!(
            !text.contains("otel_scope_"),
            "scrape must not carry otel_scope_* labels (without_scope_info); got:\n{text}"
        );
        assert!(
            !text.contains("target_info"),
            "scrape must not carry the target_info series (without_target_info)"
        );
        assert!(
            !text.contains("_total_total"),
            "counter names already end in _total; without_counter_suffixes must prevent double-suffixing"
        );
        // 2) Unit suffix must not be mangled onto the microseconds histogram.
        assert!(
            !text.contains("live_endpoint_duration_microseconds_microseconds"),
            "without_units must prevent unit-suffix mangling"
        );

        // 3) Metric names + TYPE lines match the pre-migration contract.
        assert_eq!(
            metric_type(&text, "live_endpoint_calls_total").as_deref(),
            Some("counter"),
            "endpoint_calls must render as a counter under its exact pre-migration name"
        );
        assert_eq!(
            metric_type(&text, "live_edit_inflight_rpcs").as_deref(),
            Some("gauge"),
            "inflight_rpcs (UpDownCounter) must export as a Prometheus gauge"
        );
        assert_eq!(
            metric_type(&text, "live_active_sessions").as_deref(),
            Some("gauge"),
            "live_active_sessions (ObservableGauge) must export as a Prometheus gauge under its \
             exact pre-migration name"
        );
        assert_eq!(
            metric_type(&text, "live_endpoint_duration_microseconds").as_deref(),
            Some("histogram"),
        );

        // 4) Counter value round-trips through the render/parse path.
        assert_eq!(
            sample_value(
                &text,
                "live_endpoint_calls_total",
                &[
                    ("endpoint", "Ping"),
                    ("service", "test-pod"),
                    ("grpc_code", "OK"),
                ],
            ),
            Some(2.0),
            "two inc_calls must render as 2"
        );

        // 5) The observed histogram's `le=` boundaries match DURATION_BUCKETS
        //    exactly (plus the synthetic +Inf), rendered as bare integers /
        //    decimals — not `100.0`-style or seconds buckets.
        let series_le: Vec<String> = text
            .lines()
            .filter(|l| {
                l.starts_with("live_endpoint_duration_microseconds_bucket")
                    && l.contains("endpoint=\"test_rpc\"")
            })
            .filter_map(|l| {
                let start = l.find("le=\"")? + 4;
                let rest = &l[start..];
                let end = rest.find('"')?;
                Some(rest[..end].to_string())
            })
            .collect();
        let mut expected_le: Vec<String> = DURATION_BUCKETS.iter().map(|b| b.to_string()).collect();
        expected_le.push("+Inf".to_string());
        assert_eq!(
            series_le, expected_le,
            "histogram le= boundaries must match DURATION_BUCKETS + Inf exactly"
        );
    }

    #[test]
    fn fractional_second_histogram_le_boundaries_render_in_exact_wire_string_form() {
        // Load-bearing parity guard, strongest form (OMPE-98174): `le` is a
        // STRING label, so any float-formatting divergence between the
        // OTel `PrometheusExporter` and the historical `prometheus`-crate output
        // would fork a fractional-second bucket series and silently break
        // dashboards across the migration. The sibling parity test only pins the
        // INTEGER-valued microseconds histogram; this one pins a FRACTIONAL-second
        // family (`live_diff_build_duration_seconds`, backed by
        // DIFF_DURATION_BUCKETS) in exact wire string form.
        //
        // Expected strings hardcoded (not derived via `f64::to_string`) so the
        // assertion pins the reality the exporter emits: sub-decimal bounds keep
        // their decimal form (`0.0001`, `0.0005`, …) and integer-valued bounds
        // drop the trailing `.0` (`1.0` → `1`, `5.0` → `5`) — byte-identical to
        // the pre-migration prometheus-crate rendering.
        //
        // Every boundary in `stable_boundaries` below must stay in the list:
        // boundaries may be added (`2.5`, `10`, `30`, `60` are additions) but
        // never dropped, so a dashboard keyed on an existing `le` series keeps
        // finding it.
        let metrics = metrics_scrape_only("test-pod");
        metrics.observe_diff_build("full_content", "test-pod", Duration::from_millis(2));
        let text = render(&metrics);

        let series_le = bucket_le_strings(
            &text,
            "live_diff_build_duration_seconds",
            "case=\"full_content\"",
        );

        assert_eq!(
            series_le, DIFF_DURATION_LE_WIRE,
            "fractional-second histogram le= labels must render in exact pre-migration wire \
             string form (decimal bounds verbatim, integer-valued bounds without a trailing .0)"
        );
        let stable_boundaries = [
            "0.0001", "0.0005", "0.001", "0.005", "0.01", "0.05", "0.1", "0.5", "1", "5",
        ];
        assert!(
            stable_boundaries
                .iter()
                .all(|le| series_le.iter().any(|s| s == le)),
            "every stable `le` boundary must still render, or an existing dashboard or recording \
             rule keyed on it silently loses its series; got {series_le:?}"
        );

        // Cross-check via the `histogram_bucket` helper that the exact strings
        // resolve to real bucket series (a 2 ms observation lands in `le=0.005`).
        assert_eq!(
            histogram_bucket(
                &text,
                "live_diff_build_duration_seconds",
                "0.0001",
                &[("case", "full_content"), ("service", "test-pod")],
            ),
            Some(0.0),
            "a 2 ms observation must NOT be counted in the le=0.0001 bucket"
        );
        assert_eq!(
            histogram_bucket(
                &text,
                "live_diff_build_duration_seconds",
                "0.005",
                &[("case", "full_content"), ("service", "test-pod")],
            ),
            Some(1.0),
            "a 2 ms observation must land inside the le=0.005 bucket"
        );
    }

    /// The `le` label values of every `<family>_bucket` line whose label body
    /// contains `label_filter`, in exposition order.
    fn bucket_le_strings(text: &str, family: &str, label_filter: &str) -> Vec<String> {
        let prefix = format!("{family}_bucket");
        text.lines()
            .filter(|l| l.starts_with(&prefix) && l.contains(label_filter))
            .filter_map(|l| {
                let start = l.find("le=\"")? + 4;
                let rest = &l[start..];
                let end = rest.find('"')?;
                Some(rest[..end].to_string())
            })
            .collect()
    }

    /// `DIFF_DURATION_BUCKETS` in exact wire string form, plus `+Inf`. Shared
    /// by the diff-build and catch-up duration families, which must stay
    /// matched (SDD §17.1).
    const DIFF_DURATION_LE_WIRE: [&str; 15] = [
        "0.0001", "0.0005", "0.001", "0.005", "0.01", "0.05", "0.1", "0.5", "1", "2.5", "5", "10",
        "30", "60", "+Inf",
    ];

    /// The catch-up histogram renders the same buckets as the diff-build one
    /// (the two share `DIFF_DURATION_BUCKETS`), and the dev load test's 7.8 s
    /// joins land in a finite bucket: inside `le="10"`, outside `le="5"`. A
    /// catch-up visible only in `+Inf` would leave `histogram_quantile`
    /// returning a bucket edge rather than a real estimate for it.
    #[test]
    fn catch_up_duration_histogram_renders_the_extended_buckets_and_bins_a_7_8_s_catch_up_under_10()
    {
        let metrics = metrics_scrape_only("test-pod");
        metrics.observe_subscribe_regime(
            "out_of_window",
            "test-pod",
            Duration::from_millis(7_800),
            "legacy",
            0,
        );
        let text = render(&metrics);

        assert_eq!(
            bucket_le_strings(
                &text,
                "live_subscribe_catch_up_duration_seconds",
                "regime=\"out_of_window\""
            ),
            DIFF_DURATION_LE_WIRE,
            "the catch-up histogram must render DIFF_DURATION_BUCKETS in exact wire form — the SDD \
             states the two duration histograms share one bucket list"
        );
        let labels = [("regime", "out_of_window"), ("service", "test-pod")];
        let family = "live_subscribe_catch_up_duration_seconds";
        assert_eq!(
            histogram_bucket(&text, family, "5", &labels),
            Some(0.0),
            "a 7.8 s catch-up must NOT be counted in le=\"5\""
        );
        assert_eq!(
            histogram_bucket(&text, family, "10", &labels),
            Some(1.0),
            "a 7.8 s catch-up must land in the finite le=\"10\" bucket, not only in +Inf"
        );
    }

    /// `live_edit_join_latency_seconds` renders `JOIN_LATENCY_BUCKETS` in exact
    /// wire form. Without its arm in `histogram_bucket_view` the family would
    /// fall back to OTel's default buckets, which are sized for milliseconds.
    #[test]
    fn join_latency_histogram_le_boundaries_render_in_exact_wire_string_form() {
        let metrics = metrics_scrape_only("test-pod");
        metrics.observe_join_latency(ENDPOINT_SUBSCRIBE_READ_LAYER, Duration::from_millis(300));
        let text = render(&metrics);

        assert_eq!(
            metric_type(&text, "live_edit_join_latency_seconds").as_deref(),
            Some("histogram"),
            "the family must export as a histogram under its exact name"
        );
        assert_eq!(
            bucket_le_strings(
                &text,
                "live_edit_join_latency_seconds",
                "endpoint=\"SubscribeReadLayer\""
            ),
            [
                "0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1", "2.5", "5", "10",
                "20", "30", "60", "+Inf",
            ],
            "join-latency `le` labels must be JOIN_LATENCY_BUCKETS in exact wire form"
        );
        let labels = [("endpoint", "SubscribeReadLayer"), ("service", "test-pod")];
        assert_eq!(
            histogram_bucket(&text, "live_edit_join_latency_seconds", "0.25", &labels),
            Some(0.0),
            "a 300 ms join must not be counted in le=\"0.25\""
        );
        assert_eq!(
            histogram_bucket(&text, "live_edit_join_latency_seconds", "0.5", &labels),
            Some(1.0),
            "a 300 ms join must land in le=\"0.5\""
        );
        assert_eq!(
            sample_value(&text, "live_edit_join_latency_seconds_count", &labels),
            Some(1.0)
        );
    }

    // ── SubscribeReadLayer delivery lag, send stalls, backlog depth ─────

    /// The `stage` and `phase` vocabularies are wire strings that dashboards,
    /// alerts and `docs/service/docs/metrics-reference.md` key on, so they are
    /// pinned literally: a
    /// renamed constant must break here, not silently fork a series.
    #[test]
    fn subscriber_delivery_label_vocabularies_are_the_documented_wire_strings() {
        assert_eq!(DELIVERY_LAG_STAGES, ["enqueued", "transport"]);
        assert_eq!(SEND_PHASES, ["catchup", "live"]);
    }

    /// Both `live_edit_subscriber_send_stalls_total{phase}` cells export at
    /// zero before any stall, so an alert on the `live` cell's rate has a
    /// series to evaluate from the pod's first scrape.
    #[test]
    fn subscriber_send_stall_phase_cells_export_at_zero_before_any_stall() {
        let text = render(&metrics_scrape_only("test-pod"));
        assert_eq!(
            metric_type(&text, "live_edit_subscriber_send_stalls_total").as_deref(),
            Some("counter")
        );
        for phase in SEND_PHASES {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_subscriber_send_stalls_total",
                    &[("phase", phase)]
                ),
                Some(0.0),
                "send_stalls_total{{phase=\"{phase}\"}} must be seeded at zero"
            );
        }
    }

    /// The four new histograms render their own bucket lists in exact wire
    /// form. Without its arm in `histogram_bucket_view` a family falls back to
    /// OTel's default buckets, which would put a 30 s delivery lag, a 4096-entry
    /// backlog or an 8 MiB one into the wrong bucket or `+Inf`.
    ///
    /// The placement checks pin the boundaries the guide and the
    /// `storage::notifications` limits rely on: an empty backlog lands in
    /// `le="0"`, and a lag of tens of seconds stays finite.
    #[test]
    fn subscriber_delivery_histograms_render_their_buckets_in_exact_wire_string_form() {
        let metrics = metrics_scrape_only("test-pod");
        metrics
            .observe_subscriber_delivery_lag(DELIVERY_LAG_STAGE_ENQUEUED, Duration::from_secs(25));
        metrics.observe_subscriber_send_wait(SEND_PHASE_LIVE, Duration::from_millis(2));
        metrics.observe_subscriber_backlog(0, 2_048);
        let text = render(&metrics);

        assert_eq!(
            bucket_le_strings(
                &text,
                "live_edit_subscriber_delivery_lag_seconds",
                "stage=\"enqueued\""
            ),
            [
                "0.001", "0.0025", "0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1",
                "2.5", "5", "10", "20", "30", "60", "120", "+Inf",
            ],
            "delivery-lag `le` labels must be SUBSCRIBER_DELIVERY_LAG_BUCKETS in exact wire form"
        );
        assert_eq!(
            bucket_le_strings(
                &text,
                "live_edit_subscriber_send_wait_seconds",
                "phase=\"live\""
            ),
            [
                "0.0001", "0.001", "0.005", "0.01", "0.05", "0.1", "0.25", "0.5", "1", "2.5", "5",
                "+Inf",
            ],
            "send-wait `le` labels must be SUBSCRIBER_SEND_WAIT_BUCKETS in exact wire form, \
             topped by the 5 s OUTER_SEND_DEADLINE"
        );
        assert_eq!(
            bucket_le_strings(&text, "live_edit_subscriber_backlog_entries", ""),
            ["0", "1", "4", "16", "64", "256", "1024", "4096", "16384", "+Inf"],
            "backlog-entries `le` labels must be SUBSCRIBER_BACKLOG_ENTRIES_BUCKETS in exact wire \
             form, bracketing the 4096 soft threshold and the 16384 hard cap"
        );
        assert_eq!(
            bucket_le_strings(&text, "live_edit_subscriber_backlog_bytes", ""),
            [
                "0", "1024", "16384", "131072", "1048576", "4194304", "8388608", "16777216",
                "33554432", "+Inf",
            ],
            "backlog-bytes `le` labels must be SUBSCRIBER_BACKLOG_BYTES_BUCKETS in exact wire \
             form, bracketing the 8 MiB soft threshold and the 32 MiB hard cap"
        );

        let lag = "live_edit_subscriber_delivery_lag_seconds";
        let enqueued = [("stage", "enqueued")];
        assert_eq!(
            histogram_bucket(&text, lag, "20", &enqueued),
            Some(0.0),
            "a 25 s lag is not counted in le=\"20\""
        );
        assert_eq!(
            histogram_bucket(&text, lag, "30", &enqueued),
            Some(1.0),
            "a 25 s lag lands in the finite le=\"30\" bucket, not only in +Inf"
        );
        assert_eq!(
            histogram_bucket(&text, "live_edit_subscriber_backlog_entries", "0", &[]),
            Some(1.0),
            "an empty backlog lands in le=\"0\", so `le=\"0\"` alone means the queue was empty"
        );
        assert_eq!(
            histogram_bucket(&text, "live_edit_subscriber_backlog_bytes", "1024", &[]),
            Some(0.0),
            "a 2048-byte backlog is above le=\"1024\""
        );
    }

    // ── Customer metrics reference guards ────────────────────────────────
    //
    // The six guard tests below read `docs/service/docs/metrics-reference.md`,
    // the customer page that documents the metrics an operator alerts on,
    // dashboards or diagnoses with. Every metric there has exactly one table
    // row whose first cell is its backtick-wrapped name, and that row's
    // Labels cell lists every label value the metric emits. A row is the
    // tightest scope a value can be checked in: a backtick-wrapped value such
    // as `` `ok` `` or `` `failed` `` appears on many rows, so a whole-page
    // match would pass for a value written nowhere near its own metric.
    //
    // Metrics the reference deliberately leaves out (support-only internals)
    // are not checked here; their vocabularies stay pinned by this crate's
    // zero-seeding and vocabulary tests and documented in their rustdoc.

    /// The customer metrics reference, compiled in so that a moved or
    /// deleted page is a compile error rather than a skipped test.
    const METRICS_REFERENCE: &str = include_str!("../../docs/service/docs/metrics-reference.md");

    /// The one table row of the metrics reference documenting `family`: the
    /// line whose first cell is the backtick-wrapped name.
    fn metrics_reference_row(family: &str) -> &'static str {
        let first_cell = format!("| `{family}` |");
        let rows: Vec<&'static str> = METRICS_REFERENCE
            .lines()
            .filter(|line| line.starts_with(&first_cell))
            .collect();
        assert_eq!(
            rows.len(),
            1,
            "docs/service/docs/metrics-reference.md must document `{family}` in exactly one \
             table row whose first cell is the backtick-wrapped name; found {}",
            rows.len()
        );
        rows[0]
    }

    /// Asserts that `family`'s row in the metrics reference names every value
    /// in `values` as a backtick-wrapped code span.
    fn assert_reference_row_names(family: &str, label: &str, values: &[&str]) {
        let row = metrics_reference_row(family);
        for value in values {
            let quoted = format!("`{value}`");
            assert!(
                row.contains(&quoted),
                "`{family}` mints {label}={value:?}; its row in \
                 docs/service/docs/metrics-reference.md must name it as the backtick-wrapped \
                 code span {quoted} in its Labels cell. A label value with no reference entry \
                 is a series operators can scrape but cannot interpret, and no other test in \
                 this module can see it: they all read the same constant the value was added to.\n\
                 row: {row}"
            );
        }
    }

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    /// Every `stage` and `phase` value of the live-delivery families, every
    /// `result` of `live_edit_ext_value_verify_total` ([`EXT_VALUE_VERIFY_RESULTS`]),
    /// every `outcome` of the chained-write wait
    /// `live_edit_update_layer_chain_wait_seconds` ([`CHAIN_OUTCOMES`]), and
    /// every `reason` and `state` of the external-value mirror's dropped
    /// counter and byte gauge ([`EXT_MIRROR_DROP_REASONS`],
    /// [`EXT_MIRROR_STATES`]) appears in that family's own row of the
    /// customer metrics reference.
    ///
    /// **Why scoped to the family's row.** `` `catchup` ``, `` `ok` ``,
    /// `` `ready` ``, `` `timeout` `` and the like are label values of other
    /// families too, so a whole-page match would pass with nothing written
    /// about this family. See the module note above
    /// [`metrics_reference_row`].
    ///
    /// The chain-probe counter `live_edit_update_layer_chain_probes_total`
    /// is a support-only internal the reference omits; its
    /// [`CHAIN_PROBE_RESULTS`] stay pinned in
    /// `tests/integration/grpc/chained_writes_test.rs`.
    ///
    /// Presence only: no prose, ordering or counts.
    #[test]
    fn every_subscriber_delivery_label_value_appears_in_its_family_section_of_the_metrics_guide() {
        for (family, label, values) in [
            (
                "live_edit_subscriber_delivery_lag_seconds",
                "stage",
                DELIVERY_LAG_STAGES,
            ),
            (
                "live_edit_subscriber_send_stalls_total",
                "phase",
                SEND_PHASES,
            ),
            (
                "live_edit_ext_value_verify_total",
                "result",
                EXT_VALUE_VERIFY_RESULTS,
            ),
            (
                "live_edit_update_layer_chain_wait_seconds",
                "outcome",
                CHAIN_OUTCOMES,
            ),
            (
                "live_edit_checkpoint_ext_mirror_dropped_total",
                "reason",
                EXT_MIRROR_DROP_REASONS,
            ),
            (
                "live_edit_checkpoint_ext_mirror_bytes",
                "state",
                EXT_MIRROR_STATES,
            ),
        ] {
            assert_reference_row_names(family, label, values);
        }
    }

    /// The two per-recipient delivery counters render their whole bounded label set at
    /// zero on the first scrape, so `rate()` has a series to evaluate before
    /// the first delivery — and "never delivered" is distinguishable from
    /// "never wired". Also pins the `event_type` vocabulary: one value per
    /// `PresenceEvent` variant.
    #[test]
    fn subscriber_update_and_presence_event_cells_are_zero_seeded() {
        let metrics = metrics_scrape_only("zero-seed-test");
        let text = render(&metrics);

        assert_eq!(
            sample_value(
                &text,
                "live_subscriber_updates_total",
                &[
                    ("endpoint", "SubscribeReadLayer"),
                    ("service", "zero-seed-test")
                ],
            ),
            Some(0.0),
            "live_subscriber_updates_total{{endpoint=\"SubscribeReadLayer\"}} must be seeded"
        );
        assert_eq!(
            PRESENCE_EVENT_TYPES,
            ["joined", "left", "message"],
            "the event_type vocabulary is one value per PresenceEvent variant"
        );
        for event_type in PRESENCE_EVENT_TYPES {
            assert_eq!(
                sample_value(
                    &text,
                    "live_presence_events_total",
                    &[("event_type", event_type), ("service", "zero-seed-test")],
                ),
                Some(0.0),
                "live_presence_events_total{{event_type=\"{event_type}\"}} must be seeded"
            );
        }
    }

    /// `add_subscriber_updates(n)` adds `n` to the one seeded cell, and
    /// `add_presence_events` adds to exactly the named `event_type` cell. Both
    /// carry the `service` label `Metrics::new` was built with — the cached
    /// `Arc<str>` label must render identically to the `String` one the seed
    /// and every other family use, or the batched adds would land in a second,
    /// unseeded series.
    #[test]
    fn delivery_counters_add_to_their_seeded_cells() {
        let metrics = metrics_scrape_only("svc-a");
        metrics.add_subscriber_updates(64);
        metrics.add_subscriber_updates(3);
        metrics.add_presence_events(PRESENCE_EVENT_LEFT, 1);
        let text = render(&metrics);

        let update_lines = text
            .lines()
            .filter(|l| l.starts_with("live_subscriber_updates_total{"))
            .count();
        assert_eq!(
            update_lines, 1,
            "the batched adds must land in the seeded cell, not a second series"
        );
        assert_eq!(
            sample_value(
                &text,
                "live_subscriber_updates_total",
                &[("endpoint", "SubscribeReadLayer"), ("service", "svc-a")],
            ),
            Some(67.0)
        );
        for (event_type, expected) in [("joined", 0.0), ("left", 1.0), ("message", 0.0)] {
            assert_eq!(
                sample_value(
                    &text,
                    "live_presence_events_total",
                    &[("event_type", event_type), ("service", "svc-a")],
                ),
                Some(expected),
                "event_type=\"{event_type}\""
            );
        }
    }

    /// `valkey_errors_total{kind="command_timeout"}` is seeded for every pool
    /// in `VALKEY_POOLS`, and `VALKEY_POOLS` is the six pools the service
    /// opens. `inc_valkey_command_timeout` ticks exactly one pool's cell; the
    /// tick lands on the snapshot lane, the newest label, so its cell is shown
    /// to be its own rather than an alias of the data pool's.
    #[test]
    fn valkey_command_timeout_cells_are_zero_seeded_per_pool_and_tick_one_pool() {
        assert_eq!(
            VALKEY_POOLS,
            [
                "subscriber",
                "data",
                "txn",
                "subscriber_pool",
                "maintenance",
                "snapshot"
            ],
            "the pool vocabulary is every Valkey connection pool the service opens"
        );
        assert_eq!(VALKEY_ERROR_KIND_COMMAND_TIMEOUT, "command_timeout");

        let metrics = metrics_scrape_only("test-pod");
        let text = render(&metrics);
        for pool in VALKEY_POOLS {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_valkey_errors_total",
                    &[("pool", pool), ("kind", "command_timeout")],
                ),
                Some(0.0),
                "valkey_errors_total{{pool=\"{pool}\", kind=\"command_timeout\"}} must be seeded"
            );
        }

        metrics.inc_valkey_command_timeout(VALKEY_POOL_SNAPSHOT);
        let text = render(&metrics);
        for pool in VALKEY_POOLS {
            let expected = if *pool == VALKEY_POOL_SNAPSHOT {
                1.0
            } else {
                0.0
            };
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_valkey_errors_total",
                    &[("pool", pool), ("kind", "command_timeout")],
                ),
                Some(expected),
                "pool=\"{pool}\""
            );
        }
    }

    /// A `service`-labelled counter `family` renders at zero under this
    /// process's `service` label (`pod`) on the first scrape, so `rate()` has a
    /// series before the first increment. `inc` adds one to that seeded cell:
    /// its cached `service` label must render identically to the seed's, or the
    /// increment would open a second, unseeded series.
    fn assert_service_counter_is_zero_seeded_and_ticks_by_one(
        family: &str,
        pod: &str,
        inc: fn(&Metrics),
    ) {
        let metrics = metrics_scrape_only(pod);
        let labels = [("service", pod)];

        assert_eq!(
            sample_value(&render(&metrics), family, &labels),
            Some(0.0),
            "{family} must be seeded at zero"
        );

        inc(&metrics);
        let text = render(&metrics);
        assert_eq!(
            sample_value(&text, family, &labels),
            Some(1.0),
            "one increment of {family} must add exactly one"
        );
        assert_eq!(
            text.lines()
                .filter(|l| l.starts_with(&format!("{family}{{")))
                .count(),
            1,
            "the {family} increment must land in the seeded cell, not a second series"
        );
    }

    /// `live_edit_presence_lease_refresh_failures_total`: one failed keeper
    /// step (`inc_presence_lease_refresh_failure`).
    #[test]
    fn presence_lease_refresh_failures_cell_is_zero_seeded_and_ticks_by_one() {
        assert_service_counter_is_zero_seeded_and_ticks_by_one(
            "live_edit_presence_lease_refresh_failures_total",
            "lease-keeper-pod",
            Metrics::inc_presence_lease_refresh_failure,
        );
    }

    /// `live_edit_presence_keepalive_timeouts_total`: one JoinSession ended
    /// at its client keepalive deadline (`inc_presence_keepalive_timeout`).
    #[test]
    fn presence_keepalive_timeouts_cell_is_zero_seeded_and_ticks_by_one() {
        assert_service_counter_is_zero_seeded_and_ticks_by_one(
            "live_edit_presence_keepalive_timeouts_total",
            "keepalive-pod",
            Metrics::inc_presence_keepalive_timeout,
        );
    }

    /// `live_edit_presence_rejoins_total`: one self-rejoin
    /// (`inc_presence_rejoin`).
    #[test]
    fn presence_rejoins_cell_is_zero_seeded_and_ticks_by_one() {
        assert_service_counter_is_zero_seeded_and_ticks_by_one(
            "live_edit_presence_rejoins_total",
            "rejoin-pod",
            Metrics::inc_presence_rejoin,
        );
    }

    /// `live_edit_presence_stale_removed_total`: one sweeper removal
    /// (`add_presence_stale_removed(1)`).
    #[test]
    fn presence_stale_removed_cell_is_zero_seeded_and_ticks_by_one() {
        assert_service_counter_is_zero_seeded_and_ticks_by_one(
            "live_edit_presence_stale_removed_total",
            "sweeper-pod",
            |m| m.add_presence_stale_removed(1),
        );
    }

    /// `live_edit_presence_sweep_lease_total`: the outcome vocabulary is pinned
    /// (the checkpoint lease counter's, `renewed|lost|refresh_error`), every
    /// cell renders at zero on the first scrape, and one increment of a cell
    /// lands in its seeded series and nowhere else.
    #[test]
    fn presence_sweep_lease_cells_are_zero_seeded_per_outcome_and_tick_by_one() {
        const FAMILY: &str = "live_edit_presence_sweep_lease_total";
        const POD: &str = "sweep-lease-pod";
        assert_eq!(
            PRESENCE_SWEEP_LEASE_OUTCOMES,
            &["renewed", "lost", "refresh_error"],
            "the presence sweep lease outcome vocabulary is the checkpoint lease counter's"
        );
        let metrics = metrics_scrape_only(POD);
        let text = render(&metrics);
        for outcome in PRESENCE_SWEEP_LEASE_OUTCOMES {
            assert_eq!(
                sample_value(&text, FAMILY, &[("outcome", outcome), ("service", POD)]),
                Some(0.0),
                "{FAMILY}{{outcome=\"{outcome}\"}} must be seeded at zero"
            );
        }

        for (i, outcome) in PRESENCE_SWEEP_LEASE_OUTCOMES.iter().enumerate() {
            metrics.inc_presence_sweep_lease(outcome);
            let text = render(&metrics);
            for (j, other) in PRESENCE_SWEEP_LEASE_OUTCOMES.iter().enumerate() {
                assert_eq!(
                    sample_value(&text, FAMILY, &[("outcome", other), ("service", POD)]),
                    Some(if j <= i { 1.0 } else { 0.0 }),
                    "after incrementing {:?} once each, {FAMILY}{{outcome=\"{other}\"}}",
                    &PRESENCE_SWEEP_LEASE_OUTCOMES[..=i]
                );
            }
            assert_eq!(
                text.lines()
                    .filter(|l| l.starts_with(&format!("{FAMILY}{{")))
                    .count(),
                PRESENCE_SWEEP_LEASE_OUTCOMES.len(),
                "an increment must land in its seeded cell, not open another series"
            );
        }
    }

    /// `live_edit_presence_sweep_duration_seconds`: the outcome vocabulary is
    /// pinned; the family is absent until a pass is observed (a histogram
    /// cannot be zero-seeded); an observation lands in its `outcome` cell and
    /// in the registered 0.01–300 s buckets, not the SDK defaults.
    #[test]
    fn presence_sweep_duration_appears_on_first_pass_in_its_buckets() {
        const FAMILY: &str = "live_edit_presence_sweep_duration_seconds";
        const POD: &str = "sweep-duration-pod";
        assert_eq!(
            PRESENCE_SWEEP_OUTCOMES,
            &["completed", "lease_lost", "cancelled", "error"],
            "the presence sweep pass outcome vocabulary"
        );
        let metrics = metrics_scrape_only(POD);
        assert!(
            !has_metric(&render(&metrics), FAMILY),
            "{FAMILY} is a histogram: absent until the first pass is observed"
        );

        metrics.observe_presence_sweep_duration(PRESENCE_SWEEP_OUTCOME_CANCELLED, 0.02);
        let text = render(&metrics);
        assert_eq!(metric_type(&text, FAMILY).as_deref(), Some("histogram"));
        let cancelled = [("outcome", "cancelled"), ("service", POD)];
        for (le, want) in [("0.01", 0.0), ("0.025", 1.0), ("15", 1.0), ("300", 1.0)] {
            assert_eq!(
                histogram_bucket(&text, FAMILY, le, &cancelled),
                Some(want),
                "{FAMILY}{{outcome=\"cancelled\", le=\"{le}\"}}"
            );
        }
        assert_eq!(
            histogram_bucket(&text, FAMILY, "+Inf", &cancelled),
            Some(1.0)
        );
        assert_eq!(
            sample_value(
                &text,
                &format!("{FAMILY}_count"),
                &[("outcome", "completed")]
            ),
            None,
            "an observation opens only its own outcome's series"
        );
    }

    /// `live_edit_presence_stale_removal_delay_seconds`: absent until the
    /// first removal with a known delay, then observed in the registered
    /// 1–600 s buckets.
    #[test]
    fn presence_stale_removal_delay_appears_on_first_removal_in_its_buckets() {
        const FAMILY: &str = "live_edit_presence_stale_removal_delay_seconds";
        const POD: &str = "removal-delay-pod";
        let metrics = metrics_scrape_only(POD);
        assert!(
            !has_metric(&render(&metrics), FAMILY),
            "{FAMILY} is a histogram: absent until the first removal is observed"
        );

        metrics.observe_presence_stale_removal_delay(12.0);
        let text = render(&metrics);
        let labels = [("service", POD)];
        for (le, want) in [("1", 0.0), ("10", 0.0), ("15", 1.0), ("600", 1.0)] {
            assert_eq!(
                histogram_bucket(&text, FAMILY, le, &labels),
                Some(want),
                "{FAMILY}{{le=\"{le}\"}}"
            );
        }
        assert_eq!(
            sample_value(&text, &format!("{FAMILY}_count"), &labels),
            Some(1.0)
        );
    }

    /// `live_edit_presence_sweep_oldest_stale_age_seconds`: not seeded (it
    /// first appears with the replica's first sweeper tick that publishes — a
    /// lost leader probe, which publishes 0, or a completed pass), then the
    /// last value recorded — a later 0, as a non-leader publishes, replaces an
    /// earlier maximum.
    #[test]
    fn presence_sweep_oldest_stale_age_is_absent_until_recorded_then_the_last_value() {
        const FAMILY: &str = "live_edit_presence_sweep_oldest_stale_age_seconds";
        const POD: &str = "oldest-age-pod";
        let metrics = metrics_scrape_only(POD);
        assert!(
            !has_metric(&render(&metrics), FAMILY),
            "{FAMILY} is not seeded: absent until a sweeper tick publishes it (a lost leader \
             probe or a completed pass)"
        );

        metrics.record_presence_sweep_oldest_stale_age(42.5);
        let text = render(&metrics);
        assert_eq!(metric_type(&text, FAMILY).as_deref(), Some("gauge"));
        assert_eq!(sample_value(&text, FAMILY, &[("service", POD)]), Some(42.5));

        metrics.record_presence_sweep_oldest_stale_age(0.0);
        assert_eq!(
            sample_value(&render(&metrics), FAMILY, &[("service", POD)]),
            Some(0.0),
            "a later 0 replaces the earlier maximum"
        );
    }

    #[test]
    fn pre_initialised_families_render_at_zero_on_first_scrape() {
        // Counters / up-down-counters are pre-seeded with `.add(0, ..)` so
        // `rate()` alerts see a continuous derivative from boot. Assert a
        // representative pre-init cell is present at zero on the very first
        // render (no measurement recorded).
        let metrics = metrics_scrape_only("test-pod");
        let text = render(&metrics);
        assert_eq!(
            sample_value(
                &text,
                "live_edit_presence_broadcast_dropped_total",
                &[("service", "test-pod")],
            ),
            Some(0.0),
            "presence_broadcast_dropped must be pre-initialised at zero on first scrape"
        );
        assert_eq!(
            sample_value(
                &text,
                "live_edit_health_probe_failures_total",
                &[("backend", "valkey")],
            ),
            Some(0.0),
            "health_probe_failures{{backend=valkey}} must be pre-initialised at zero"
        );
    }

    /// `live_edit_userinfo_fallbacks_total` is one unlabelled series, at 0 on
    /// the first scrape in every configuration (it stays 0 unless the
    /// userinfo service and authorization are both on), and one
    /// `inc_userinfo_fallback` adds one to that same series.
    #[test]
    fn userinfo_fallbacks_total_is_one_unlabelled_series_seeded_at_zero() {
        const FAMILY: &str = "live_edit_userinfo_fallbacks_total";
        let samples = |text: &str| -> Vec<String> {
            text.lines()
                .filter(|l| !l.starts_with('#'))
                .filter(|l| l.split(['{', ' ']).next() == Some(FAMILY))
                .map(str::to_owned)
                .collect()
        };
        let metrics = metrics_scrape_only("fallback-pod");

        let first = samples(&render(&metrics));
        assert_eq!(
            first,
            vec![format!("{FAMILY} 0")],
            "{FAMILY} must render as one unlabelled series at zero on first scrape"
        );

        metrics.inc_userinfo_fallback();
        let after = samples(&render(&metrics));
        assert_eq!(
            after,
            vec![format!("{FAMILY} 1")],
            "one fallback must add one to the seeded series, not open another"
        );
    }

    /// Every `phase` cell of the b.gg7 ceiling counter is zero-seeded, and the
    /// vocabulary is pinned.
    ///
    /// Sibling of `checkpointer::reconcile_tests::reconcile_outcome_cells_are_
    /// zero_seeded`, and for the sharper version of the same reason: this
    /// counter is expected to read **zero forever** on a healthy deployment.
    /// Without seeding, "the adapter ceilings have never fired" and "the metric
    /// was never wired" render identically — and the first is the thing an
    /// operator is actually trying to confirm.
    ///
    /// The length assertion is the other half. A phase added at a new elapse
    /// site but not to `CHECKPOINT_ADAPTER_CEILING_PHASES` materialises its
    /// series only on first occurrence, which is precisely the failure mode the
    /// seeding exists to prevent — and it would go unnoticed, because the
    /// series does eventually appear.
    #[test]
    fn checkpoint_adapter_ceiling_phase_cells_are_zero_seeded() {
        let metrics = metrics_scrape_only("zero-seed-test");
        let text = render(&metrics);
        for phase in CHECKPOINT_ADAPTER_CEILING_PHASES {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_checkpoint_adapter_ceiling_elapsed_total",
                    &[("phase", phase), ("service", "zero-seed-test")],
                ),
                Some(0.0),
                "phase `{phase}` must be zero-seeded — a ceiling that has never elapsed is \
                 indistinguishable from an unwired counter otherwise"
            );
        }
        assert_eq!(
            CHECKPOINT_ADAPTER_CEILING_PHASES,
            ["call", "enumeration"],
            "the vocabulary is pinned: `call` is one logical request (backend brownout), \
             `enumeration` is a multi-request walk (a namespace that has outgrown one grace \
             window). They have different remedies, so a new elapse site needs its own \
             deliberate label rather than reuse of whichever is closest"
        );
    }

    /// Every `live_edit_ext_value_verify_total{result}` cell renders at zero on
    /// the first scrape, and the vocabulary is pinned.
    ///
    /// Same reason as [`checkpoint_adapter_ceiling_phase_cells_are_zero_seeded`]:
    /// the rollout check for `LIVE_EDIT_EXT_VALUE_VERIFY=observe` is that both
    /// mismatch cells stay flat while `ok` climbs, and an unseeded mismatch
    /// cell reads the same as an unwired counter. A result added to the code
    /// but not to [`EXT_VALUE_VERIFY_RESULTS`] would materialise only on its
    /// first occurrence; the length pin makes that a deliberate change here.
    #[test]
    fn ext_value_verify_result_cells_are_zero_seeded() {
        let metrics = metrics_scrape_only("zero-seed-test");
        let text = render(&metrics);
        for result in EXT_VALUE_VERIFY_RESULTS {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_ext_value_verify_total",
                    &[("result", result)],
                ),
                Some(0.0),
                "result `{result}` must be zero-seeded, or a mismatch cell that has never moved \
                 is indistinguishable from an unwired counter"
            );
        }
        assert_eq!(
            EXT_VALUE_VERIFY_RESULTS.len(),
            3,
            "the vocabulary is pinned at three results"
        );
        assert_eq!(
            EXT_VALUE_VERIFY_RESULTS,
            ["ok", "hash_mismatch", "size_mismatch"],
            "the result vocabulary is pinned: the published metrics reference names these values"
        );
    }

    /// U2 (b.pa1): every `outcome` cell of
    /// `live_edit_gc_oom_collections_total` — all five — and the unlabelled
    /// `live_edit_gc_oom_freed_bytes_total` exist at zero before the first
    /// collection, so an `increase()` alert on `outcome="refused"` or
    /// `"freed_nothing"` sees its first tick. One recorded collection ticks
    /// its own cell only, and adds the bytes it freed.
    #[test]
    fn live_edit_gc_oom_collections_total_seeds_every_outcome_at_zero() {
        let metrics = metrics_scrape_only("zero-seed-test");
        let text = render(&metrics);
        for outcome in GC_OOM_COLLECTION_OUTCOMES {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_gc_oom_collections_total",
                    &[("outcome", outcome)],
                ),
                Some(0.0),
                "outcome `{outcome}` must be zero-seeded"
            );
        }
        assert_eq!(
            sample_value(&text, "live_edit_gc_oom_freed_bytes_total", &[]),
            Some(0.0),
            "the freed-bytes counter must be zero-seeded"
        );
        assert_eq!(
            GC_OOM_COLLECTION_OUTCOMES,
            [
                "freed",
                "freed_nothing",
                "nothing_to_collect",
                "failed",
                "refused"
            ],
            "the outcome vocabulary is pinned at five: the metrics guide and its alerts name \
             these values"
        );

        metrics.record_gc_oom_collection(GC_OOM_COLLECTION_OUTCOME_FREED, 4_096);
        let text = render(&metrics);
        for outcome in GC_OOM_COLLECTION_OUTCOMES {
            let want = if *outcome == GC_OOM_COLLECTION_OUTCOME_FREED {
                1.0
            } else {
                0.0
            };
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_gc_oom_collections_total",
                    &[("outcome", outcome)],
                ),
                Some(want),
                "outcome `{outcome}` after one `freed` collection"
            );
        }
        assert_eq!(
            sample_value(&text, "live_edit_gc_oom_freed_bytes_total", &[]),
            Some(4_096.0)
        );
    }

    // ══ b.z22: storage-api client-side redirect surface ═════════════════════

    /// Every `(verb, outcome)` cell of the b.z22 redirect counter is
    /// zero-seeded, and the vocabulary is pinned as a list of PAIRS.
    ///
    /// Sharper than its ceiling sibling in one respect. This counter is *not*
    /// expected to read zero forever: `verb="put"` is the normal path for any
    /// artifact at or above the deployment's `writeBodyMaxSize`, while
    /// `verb="get"` should stay at zero because reads declare
    /// `download_preference=body`. So the operator's question is per-arm — "is
    /// the read fallback engaging?" — and an absent `get` series cannot answer
    /// it, because "never fired" and "never wired" render identically.
    ///
    /// The pairs assertion is the other half, and it is why the vocabulary is a
    /// list of tuples rather than a cross product: several outcomes are
    /// structurally reachable on only one verb (there is no completion call on
    /// a read, and no multipart alternative on a write's read path), so seeding
    /// the 2 × 6 grid would publish six cells that can never fire — noise that
    /// looks exactly like coverage.
    #[test]
    fn checkpoint_adapter_redirect_cells_are_zero_seeded() {
        let metrics = metrics_scrape_only("zero-seed-test");
        let text = render(&metrics);
        for (verb, outcome) in CHECKPOINT_ADAPTER_REDIRECT_CELLS {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_checkpoint_adapter_redirect_total",
                    &[
                        ("verb", verb),
                        ("outcome", outcome),
                        ("service", "zero-seed-test")
                    ],
                ),
                Some(0.0),
                "cell {{verb={verb}, outcome={outcome}}} must be zero-seeded — on the `get` arm \
                 a cell that has never fired is the HEALTHY reading, and it is unreadable from \
                 an absent series"
            );
        }
        assert_eq!(
            CHECKPOINT_ADAPTER_REDIRECT_CELLS,
            [
                ("get", "followed"),
                ("get", "target_failed"),
                ("get", "malformed"),
                ("put", "followed"),
                ("put", "completed"),
                ("put", "target_failed"),
                ("put", "complete_failed"),
                ("put", "multipart_reissued"),
                ("put", "malformed"),
                ("put", "preflight_redirect"),
                ("put", "preflight_unavailable"),
                ("put", "preflight_stale"),
            ],
            "the vocabulary is pinned as PAIRS, not a cross product: `completed` / \
             `complete_failed` / `multipart_reissued` and the three `preflight_*` outcomes exist \
             only on the write handshake, so a `get` cell bearing one of them could never fire \
             and would publish a series that looks like coverage and is not"
        );
    }

    /// Driving every pinned pair through `inc_checkpoint_adapter_redirect`
    /// opens exactly as many label cells as the vocabulary has.
    ///
    /// The zero-seeding test cannot catch this: seeding proves what
    /// *construction* opens, this proves what *increments* open. An increment
    /// carrying a label spelled at the call site instead of through the
    /// `REDIRECT_OUTCOME_*` constants adds a series on top of the seeded ones,
    /// every per-cell assertion keeps passing, and `sum by (outcome)` quietly
    /// stops equalling the event count. The adapter has nine outcome constants
    /// and two verb constants spread across twenty-three increment sites, which
    /// is exactly the shape that produces a typo nobody notices.
    #[test]
    fn inc_checkpoint_adapter_redirect_opens_no_series_outside_the_pinned_vocabulary() {
        let metrics = metrics_scrape_only("typo-guard-test");
        for (verb, outcome) in CHECKPOINT_ADAPTER_REDIRECT_CELLS {
            metrics.inc_checkpoint_adapter_redirect(verb, outcome, "typo-guard-test");
        }
        let text = render(&metrics);

        let series: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("live_edit_checkpoint_adapter_redirect_total{"))
            .collect();
        assert_eq!(
            series.len(),
            CHECKPOINT_ADAPTER_REDIRECT_CELLS.len(),
            "driving every pinned pair must leave exactly {} label cells; got {} series: \
             {series:?}. An extra key is an out-of-band label materialised on its first \
             occurrence — the regression no per-cell probe can see",
            CHECKPOINT_ADAPTER_REDIRECT_CELLS.len(),
            series.len()
        );

        for (verb, outcome) in CHECKPOINT_ADAPTER_REDIRECT_CELLS {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_checkpoint_adapter_redirect_total",
                    &[
                        ("verb", verb),
                        ("outcome", outcome),
                        ("service", "typo-guard-test")
                    ],
                ),
                Some(1.0),
                "cell {{verb={verb}, outcome={outcome}}} must read exactly 1 after one \
                 increment — a 2 would mean two pairs collapsed onto one cell, which breaks the \
                 one-tick-per-event identity the `followed` vs `completed` gap is read through"
            );
        }
    }

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    /// Every redirect label value appears in the
    /// `live_edit_checkpoint_adapter_redirect_total` row of the customer
    /// metrics reference.
    ///
    /// Same shape as
    /// [`every_adjudicated_label_value_appears_in_the_metrics_guide`]: presence
    /// only, backtick-wrapped, scoped to the family's row.
    ///
    /// It gets its own test rather than an extension of that one because the
    /// two vocabularies would fail for different reasons.
    ///
    /// The `verb` values are checked alongside the outcomes: this family is the
    /// only checkpoint-adapter counter carrying a `verb` label, so `get` / `put`
    /// are not vocabulary an operator can infer from a neighbouring metric.
    /// `verb="get"` climbing is the operator's only signal that the backup
    /// store ignores the declared inline-read preference, so that restores
    /// depend on blob-storage egress the deployment may not permit.
    #[test]
    fn every_checkpoint_adapter_redirect_label_value_appears_in_the_metrics_guide() {
        let verbs: Vec<&str> = CHECKPOINT_ADAPTER_REDIRECT_CELLS
            .iter()
            .map(|(verb, _)| *verb)
            .collect();
        let outcomes: Vec<&str> = CHECKPOINT_ADAPTER_REDIRECT_CELLS
            .iter()
            .map(|(_, outcome)| *outcome)
            .collect();
        assert_reference_row_names(
            "live_edit_checkpoint_adapter_redirect_total",
            "verb",
            &verbs,
        );
        assert_reference_row_names(
            "live_edit_checkpoint_adapter_redirect_total",
            "outcome",
            &outcomes,
        );
    }

    // ══ b.dfh Epic 3 / SDD SR-8.2: tail-capture observability surface ═══════

    /// Every `outcome` cell of `live_edit_checkpoint_tail_total` is zero-seeded,
    /// the vocabulary is pinned, and the family carries no `service` label.
    ///
    /// Same argument as `checkpoint_adapter_ceiling_phase_cells_are_zero_seeded`,
    /// sharpened: on a healthy enabled deployment every `refused_*` cell reads
    /// ZERO forever, and "no tail has ever been refused" is exactly what an
    /// operator wants to assert from a *first* scrape, before any tick has run.
    /// A missing series cannot support that assertion; a zero one can.
    #[test]
    fn checkpoint_tail_outcome_cells_are_zero_seeded() {
        let metrics = metrics_scrape_only("zero-seed-test");
        let text = render(&metrics);

        for outcome in CHECKPOINT_TAIL_OUTCOMES {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_checkpoint_tail_total",
                    // The increment site (`inc_checkpoint_tail`) emits `outcome`
                    // and nothing else — this tuple must mirror it exactly, or
                    // the test would pass against a seeding loop whose label set
                    // no longer matches what increments actually open.
                    &[("outcome", outcome)],
                ),
                Some(0.0),
                "outcome `{outcome}` must be zero-seeded — a refusal that has never \
                 fired is otherwise indistinguishable from an unwired counter, and \
                 that distinction is the whole point of the family"
            );
        }

        // `sample_value` label-matches by subset, so the loop above would also
        // be satisfied by cells that carry an extra `service` label. Pin the
        // deliberate absence directly: this family is `outcome`-only so it pairs
        // with the likewise service-free `checkpoint_writes_total` without a
        // `sum without(service)` rewrite on every dashboard query.
        let tail_series: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("live_edit_checkpoint_tail_total{"))
            .collect();
        assert!(
            !tail_series.is_empty(),
            "the tail family must render on a first scrape; exposition was: {text}"
        );
        for line in &tail_series {
            assert!(
                !line.contains("service=\""),
                "live_edit_checkpoint_tail_total must NOT carry a `service` label — it \
                 pairs with the service-free checkpoint_writes_total, and the asymmetry \
                 would force a `sum without(service)` on every pairing query. Got: {line}"
            );
        }

        // Vocabulary pin. SDD SR-8.2 enumerates EIGHT outcomes; the ninth is
        // `refused_chain_full`, the chain-bound refusal (Epic ticket
        // t1.dfh.57). Both it and `rebased` have live producers: a tick that
        // finds the chain at K attempts an SR-4.1 chain-collapse re-base, which
        // ticks `rebased` when it lands and folds every SR-4.5 refusal cause
        // back onto `refused_chain_full` when it does not. Seeding both from the
        // outset is why the dashboard vocabulary never had to migrate when that
        // producer arrived.
        //
        // The TENTH is `refused_lineage_unrootable` (Epic 5's SR-7
        // adjudication, t2.dfh.be.s8): the two checkpoint-store-DAMAGE arms —
        // a listing that holds only tails, and a lineage head at sequence 0 —
        // split out of `refused_no_base`, which is left meaning exactly "this
        // object has no durable generation at all", the rollout tracker's
        // reading. The split is what keeps a bulk-then-draining rollout
        // population from hiding a store that needs manual repair. Appended
        // rather than inserted beside its sibling, and zero-seeded like every
        // other member, so the addition is invisible to dashboards keyed on the
        // nine older cells.
        assert_eq!(
            CHECKPOINT_TAIL_OUTCOMES,
            [
                "captured",
                "captured_capped",
                "rebased",
                "refused_trimmed",
                "refused_missing_bytes",
                "refused_sentinel",
                "refused_seq_gap",
                "refused_no_base",
                "refused_chain_full",
                "refused_lineage_unrootable",
            ],
            "the outcome vocabulary is pinned: sibling Tasks in this Epic map their \
             classification results onto it 1:1, so an outcome invented at a call site \
             rather than added here would render only on first occurrence"
        );
        assert_eq!(
            CHECKPOINT_TAIL_OUTCOMES.len(),
            10,
            "ten outcomes: SR-8.2's eight, plus the chain-bound `refused_chain_full`, plus \
             SR-7's `refused_lineage_unrootable` store-damage split"
        );
    }

    /// `checkpoint_writes_total` seeds the full `result x kind` grid including
    /// the new `layer_tail` kind — and still seeds the three pre-existing kinds.
    ///
    /// The second half is the regression that matters: a change that *replaced*
    /// the inner kind list rather than extending it would satisfy a
    /// `layer_tail`-only assertion while silently un-seeding `layer`.
    #[test]
    fn checkpoint_writes_total_layer_tail_cells_are_zero_seeded() {
        let metrics = metrics_scrape_only("zero-seed-test");
        let text = render(&metrics);

        for result in ["success", "error", "skipped"] {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_checkpoint_writes_total",
                    &[("result", result), ("kind", "layer_tail")],
                ),
                Some(0.0),
                "{{result={result}, kind=layer_tail}} must be zero-seeded (SR-8.2)"
            );
            for kind in ["layer", "session", "counters_sidecar"] {
                assert_eq!(
                    sample_value(
                        &text,
                        "live_edit_checkpoint_writes_total",
                        &[("result", result), ("kind", kind)],
                    ),
                    Some(0.0),
                    "pre-existing cell {{result={result}, kind={kind}}} must still be \
                     zero-seeded — adding `layer_tail` must extend the kind list, not \
                     replace it"
                );
            }
        }
    }

    /// The `kind` vocabulary of `live_edit_checkpoint_writes_total` is pinned.
    ///
    /// This closes the const-drift hole at the unit tier. The integration-tier
    /// cardinality guard deliberately restates the vocabulary independently, so
    /// it cannot catch an edit to the const itself — the two would move
    /// together. This pin is what fails on such an edit.
    #[test]
    fn checkpoint_write_kinds_vocabulary_is_pinned() {
        assert_eq!(
            CHECKPOINT_WRITE_KINDS,
            ["layer", "layer_tail", "session", "counters_sidecar"],
            "`layer` is a base (including the fresh base a re-base composes), \
             `layer_tail` is a stream-replay tail, and the two coordination blobs \
             keep their own kinds; a re-base deliberately gets no kind of its own \
             because what it writes IS a base"
        );
        assert_eq!(
            CHECKPOINT_WRITE_KINDS.len(),
            4,
            "four write kinds; a fifth added at an increment site but not here \
             materialises its series only once that write path first fires"
        );
    }

    /// Driving `inc_checkpoint_tail` once per vocabulary member opens exactly
    /// as many label cells as the vocabulary has members — no extra series from
    /// a hardcoded label at a call site.
    ///
    /// Zero-seeding alone cannot catch that: an increment with an unlisted
    /// outcome adds a series on top of the seeded ones, and every per-cell
    /// assertion above keeps passing.
    #[test]
    fn inc_checkpoint_tail_opens_no_series_outside_the_pinned_vocabulary() {
        let metrics = metrics_scrape_only("typo-guard-test");
        for outcome in CHECKPOINT_TAIL_OUTCOMES {
            metrics.inc_checkpoint_tail(outcome);
        }
        let text = render(&metrics);

        let tail_series: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("live_edit_checkpoint_tail_total{"))
            .collect();
        assert_eq!(
            tail_series.len(),
            CHECKPOINT_TAIL_OUTCOMES.len(),
            "driving every vocabulary member must leave exactly {} label cells; got \
             {} series: {tail_series:?}",
            CHECKPOINT_TAIL_OUTCOMES.len(),
            tail_series.len()
        );

        for outcome in CHECKPOINT_TAIL_OUTCOMES {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_checkpoint_tail_total",
                    &[("outcome", outcome)],
                ),
                Some(1.0),
                "outcome `{outcome}` must read exactly 1 after one increment — a value \
                 of 2 would mean two vocabulary members collapsed onto one cell"
            );
        }
    }

    // ══ b.dfh Epic 5 / SDD SR-7: base-bootstrap observability surface ═══════

    /// Every `outcome × site` cell of `live_edit_checkpoint_base_bootstrap_total`
    /// is zero-seeded, the family renders exactly the 9 × 2 grid, and it carries
    /// no `service` label.
    ///
    /// Modelled on `checkpoint_tail_outcome_cells_are_zero_seeded` and for the
    /// sharpest form of that argument yet. Every `shed`, `unstamped_*` and
    /// `failed` cell reads ZERO forever on a healthy deployment, and **"no
    /// bootstrap has ever been shed"** is exactly the assertion an operator needs
    /// from a *first* scrape — before any layer has been created and before any
    /// restore has run. A missing series cannot support that assertion; a zero
    /// one can. The `shed` concentration case is a mass restore against a
    /// degraded adapter, i.e. the moment that first scrape is being read under
    /// pressure.
    ///
    /// Three halves, each load-bearing and none implied by the others:
    ///
    /// - The per-cell loop covers the FULL grid, the two `RestoreRebase`-only
    ///   arms at `creation_base` included. Those cells are inadmissible by
    ///   construction and stay zero forever, which is precisely why an
    ///   admissible-pairs-only seeding would be indistinguishable from a
    ///   regression that stopped seeding them.
    /// - The rendered-line count is the OTHER direction: `sample_value` matches
    ///   labels as a subset, so the loop alone passes against a seeding that
    ///   opened extra out-of-band cells too. Eighteen lines, exactly.
    /// - The `service` absence is pinned directly off the rendered text for the
    ///   same reason as the tail family's: this counter pairs with the
    ///   likewise service-free `checkpoint_writes_total` and
    ///   `checkpoint_tail_total`, and the asymmetry would force a
    ///   `sum without(service)` rewrite on every pairing query.
    #[test]
    fn checkpoint_base_bootstrap_cells_are_zero_seeded() {
        let metrics = metrics_scrape_only("zero-seed-test");
        let text = render(&metrics);

        for outcome in CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES {
            for site in CHECKPOINT_BASE_BOOTSTRAP_SITES {
                assert_eq!(
                    sample_value(
                        &text,
                        "live_edit_checkpoint_base_bootstrap_total",
                        // Mirrors `inc_checkpoint_base_bootstrap`'s label tuple
                        // exactly — `outcome` and `site`, nothing else. A seeding
                        // loop whose label set drifted from the increment site
                        // would open a second, permanently-zero series per cell
                        // and this tuple is what refuses to match it.
                        &[("outcome", outcome), ("site", site)],
                    ),
                    Some(0.0),
                    "cell {{outcome={outcome}, site={site}}} must be zero-seeded — an arm that \
                     has never been taken is otherwise indistinguishable from a producer nobody \
                     wired, and for `shed` that distinction is the whole point of the family"
                );
            }
        }

        let series: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("live_edit_checkpoint_base_bootstrap_total{"))
            .collect();
        assert_eq!(
            series.len(),
            20,
            "the family must render exactly 20 series on a FIRST scrape — the full 10 x 2 grid, \
             the two RestoreRebase-only arms seeded at `creation_base` included. Fewer means the \
             seeding loop skipped inadmissible pairs (making a stopped-seeding regression \
             invisible); more means an out-of-band cell the per-cell subset matcher above cannot \
             see. Got {} series: {series:?}",
            series.len()
        );

        for line in &series {
            assert!(
                !line.contains("service=\""),
                "live_edit_checkpoint_base_bootstrap_total must NOT carry a `service` label — it \
                 pairs with the service-free checkpoint_writes_total and checkpoint_tail_total, \
                 and the asymmetry would force a `sum without(service)` on every pairing query \
                 while materialising a second, permanently-zero series per cell. Got: {line}"
            );
        }
    }

    /// Both vocabularies of `live_edit_checkpoint_base_bootstrap_total` are
    /// pinned to their exact spellings, in order, with their lengths.
    ///
    /// The exact-slice comparison is the half a `len()` check cannot do: a member
    /// *renamed* at the constant and left un-renamed at the mint
    /// (`checkpoint::base_bootstrap`'s `BootstrapOutcome::outcome_label` /
    /// `BootstrapTarget::label`) keeps the length and opens an unseeded series.
    /// The length assertions are the half the slice comparison cannot do on its
    /// own for a *reader*: they state the grid size the 20-series count above
    /// rides on, so the two move together or the pair goes red.
    ///
    /// Order is pinned, not just membership, because `outcome_label` INDEXES this
    /// constant — a reordering would leave every arm compiling and every label
    /// still a member while silently re-attributing each arm to the wrong cell.
    /// `base_bootstrap`'s `BOOTSTRAP_OUTCOME_LABEL_PINS` makes that a compile
    /// error; this is the same fact stated where the vocabulary lives.
    #[test]
    fn checkpoint_base_bootstrap_vocabulary_is_pinned() {
        assert_eq!(
            CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES,
            [
                "persisted",
                "shed",
                "skipped_header_absent",
                "skipped_generation_present",
                "skipped_untrusted_cursor",
                "unstamped_header_absent",
                "unstamped_generation_present",
                "unstamped_foreign_restore",
                "unstamped_write_error",
                "failed",
            ],
            "the base-bootstrap outcome vocabulary is pinned, in order: `outcome_label` indexes \
             this constant, so a reordering re-attributes every arm from that index on, and an \
             outcome invented at a call site rather than added here renders only on first \
             occurrence — the exact failure the zero-seeding exists to prevent"
        );
        assert_eq!(
            CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES.len(),
            10,
            "ten outcomes: `persisted`, the admission `shed`, three pre-PUT `skipped_*`, four \
             post-PUT `unstamped_*` (b.vn6 added `unstamped_header_absent`, the post-PUT half of \
             the deleted-header race whose pre-PUT half is `skipped_header_absent`), and \
             `failed`. The benign members are the DENOMINATOR that turns `shed` into a \
             saturation ratio, so none of them may be folded away"
        );

        assert_eq!(
            CHECKPOINT_BASE_BOOTSTRAP_SITES,
            ["creation_base", "restore_rebase"],
            "the site vocabulary is pinned: `BootstrapTarget::label()` mints exactly these two \
             spellings and they double as the `target` field on this unit's log lines, so a \
             re-spelling on either side silently breaks the cell-to-log join"
        );
        assert_eq!(
            CHECKPOINT_BASE_BOOTSTRAP_SITES.len(),
            2,
            "two call sites: SR-7.1's creation-time first base and SR-3.6's post-restore \
             re-base. A third spawn site needs its own deliberate label here, not reuse of \
             whichever is closest"
        );
    }

    /// Driving `inc_checkpoint_base_bootstrap` once per grid cell opens exactly
    /// as many label cells as the grid has — no extra series from a hardcoded or
    /// misspelled label at a call site.
    ///
    /// The typo-guard `checkpoint_base_bootstrap_cells_are_zero_seeded` cannot
    /// be: seeding proves what construction opens, this proves what INCREMENTS
    /// open. An increment carrying an unlisted `outcome` (or a `site` spelled at
    /// the call site instead of through `BootstrapTarget::label()`) adds a series
    /// on top of the seeded ones, and every per-cell assertion keeps passing
    /// while `sum by (outcome)` quietly stops equalling the invocation count.
    #[test]
    fn inc_checkpoint_base_bootstrap_opens_no_series_outside_the_pinned_vocabulary() {
        let metrics = metrics_scrape_only("typo-guard-test");
        for outcome in CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES {
            for site in CHECKPOINT_BASE_BOOTSTRAP_SITES {
                metrics.inc_checkpoint_base_bootstrap(outcome, site);
            }
        }
        let text = render(&metrics);

        let series: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("live_edit_checkpoint_base_bootstrap_total{"))
            .collect();
        let expected =
            CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES.len() * CHECKPOINT_BASE_BOOTSTRAP_SITES.len();
        assert_eq!(
            series.len(),
            expected,
            "driving every grid cell must leave exactly {expected} label cells; got {} series: \
             {series:?}. An extra key is an out-of-band label materialised on its first \
             occurrence — the regression no per-cell probe can see",
            series.len()
        );

        for outcome in CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES {
            for site in CHECKPOINT_BASE_BOOTSTRAP_SITES {
                assert_eq!(
                    sample_value(
                        &text,
                        "live_edit_checkpoint_base_bootstrap_total",
                        &[("outcome", outcome), ("site", site)],
                    ),
                    Some(1.0),
                    "cell {{outcome={outcome}, site={site}}} must read exactly 1 after one \
                     increment — a value of 2 would mean two grid cells collapsed onto one, \
                     which breaks the one-tick-per-invocation identity `sum by (outcome)` rides"
                );
            }
        }
    }

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    /// Every value of the checkpoint full-backup and incremental-backup
    /// vocabularies, and every upload content-check result, appears in its
    /// family's row of the customer metrics reference.
    ///
    /// **Presence only, and deliberately so.** It asserts that each value is
    /// written, backtick-wrapped, in the family's row — no prose matching, no
    /// ordering, no counts, no claim about what the page *says* about a value.
    /// A prose-matching test would go red on every unrelated rewording;
    /// presence cannot be satisfied by accident and cannot be broken by an
    /// editor.
    ///
    /// # Why the backtick-wrapped form, scoped to the row
    ///
    /// `persisted`, `shed` and `failed` are ordinary English or substrings of
    /// it, and `` `failed` `` is a label value of other families too, so a
    /// bare or whole-page match would be vacuous for them. The backtick-wrapped
    /// value in the family's own row can only come from a deliberate listing
    /// of that family's identifier.
    ///
    /// # What it is a fence against
    ///
    /// A future cell shipping undocumented. The vocabulary constants are the
    /// mint; the reference is the only place an operator learns that a cell
    /// exists. Adding a member to a constant without a reference entry
    /// produces a series that scrapes, alerts and dashboards can see and
    /// nobody can interpret — and nothing else in the suite notices, because
    /// every other pin in this module reads the constant and is therefore
    /// satisfied by the addition itself.
    ///
    /// # Scope
    ///
    /// [`CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES`] × [`CHECKPOINT_BASE_BOOTSTRAP_SITES`],
    /// `refused_lineage_unrootable` (the newest [`CHECKPOINT_TAIL_OUTCOMES`]
    /// member, the one with a standing alert), and
    /// [`EXT_VALUE_VERIFY_RESULTS`] of the upload content check.
    #[test]
    fn every_adjudicated_label_value_appears_in_the_metrics_guide() {
        // The newest tail member only, by position — see the assertion's
        // message for why the position is pinned.
        let tail_member = CHECKPOINT_TAIL_OUTCOMES
            .last()
            .expect("CHECKPOINT_TAIL_OUTCOMES is never empty");
        assert_eq!(
            *tail_member, "refused_lineage_unrootable",
            "this guard checks the tail vocabulary's NEWEST member by position, so an eleventh \
             member appended later inherits the check automatically — but a member inserted \
             ahead of `refused_lineage_unrootable` would silently move the target. Re-point this \
             assertion deliberately if the vocabulary grows"
        );

        assert_reference_row_names(
            "live_edit_checkpoint_base_bootstrap_total",
            "outcome",
            CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES,
        );
        assert_reference_row_names(
            "live_edit_checkpoint_base_bootstrap_total",
            "site",
            CHECKPOINT_BASE_BOOTSTRAP_SITES,
        );
        assert_reference_row_names("live_edit_checkpoint_tail_total", "outcome", &[tail_member]);
        // The mismatch cells are the operator's rollout check for
        // LIVE_EDIT_EXT_VALUE_VERIFY=observe, so an undocumented one cannot
        // be read.
        assert_reference_row_names(
            "live_edit_ext_value_verify_total",
            "result",
            EXT_VALUE_VERIFY_RESULTS,
        );
    }

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    /// [`every_adjudicated_label_value_appears_in_the_metrics_guide`]'s sibling
    /// for the unresolved-stream `reason` vocabulary.
    ///
    /// Same property, same mechanism, same reasoning: presence only, matched on
    /// the **backtick-wrapped** value in the family's own row of the customer
    /// metrics reference. Read that test's rustdoc for the full argument.
    ///
    /// A separate test rather than more values chained into it: the two
    /// vocabularies would fail for different reasons, and this one sweeps its
    /// constant WHOLE rather than pinning a newest-member-by-position.
    ///
    /// # What it is a fence against
    ///
    /// A fourth `reason` shipping undocumented. Each of today's three names a
    /// different operator situation with a different remediation (two refuse
    /// the restore, one lets it proceed onto a key the service will not
    /// touch), so a cell an operator can scrape but cannot interpret is the
    /// failure this catches.
    #[test]
    fn every_unresolved_stream_reason_appears_in_the_metrics_guide() {
        assert_reference_row_names(
            "live_edit_checkpoint_recovery_stream_unresolved_total",
            "reason",
            CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS,
        );
    }

    /// The [`CHECKPOINT_TAIL_OUTCOMES`] guard, for the restore-lineage family.
    ///
    /// Driving `inc_checkpoint_restore_lineage` once per vocabulary member must
    /// open exactly one label cell per member — no extra series from a
    /// hardcoded label at a call site. Zero-seeding alone cannot catch that: an
    /// increment with an unlisted outcome adds a series on top of the seeded
    /// ones, and every per-cell assertion keeps passing.
    #[test]
    fn inc_checkpoint_restore_lineage_opens_no_series_outside_the_pinned_vocabulary() {
        let metrics = metrics_scrape_only("typo-guard-test");
        for outcome in CHECKPOINT_RESTORE_LINEAGE_OUTCOMES {
            metrics.inc_checkpoint_restore_lineage(outcome, "typo-guard-test");
        }
        let text = render(&metrics);

        let series: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("live_edit_checkpoint_restore_lineage_total{"))
            .collect();
        assert_eq!(
            series.len(),
            CHECKPOINT_RESTORE_LINEAGE_OUTCOMES.len(),
            "driving every vocabulary member must leave exactly {} label cells; got {} series: \
             {series:?}",
            CHECKPOINT_RESTORE_LINEAGE_OUTCOMES.len(),
            series.len()
        );

        for outcome in CHECKPOINT_RESTORE_LINEAGE_OUTCOMES {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_checkpoint_restore_lineage_total",
                    &[("outcome", outcome), ("service", "typo-guard-test")],
                ),
                Some(1.0),
                "outcome `{outcome}` must read exactly 1 after one increment — a value of 2 \
                 would mean two vocabulary members collapsed onto one cell"
            );
        }
    }

    /// Every cell of the new family reads ZERO from the FIRST scrape, before
    /// any restore has run — the assertion the zero-seeding exists to support.
    #[test]
    fn checkpoint_restore_lineage_cells_are_zero_seeded() {
        let metrics = metrics_scrape_only("seed-test");
        let text = render(&metrics);
        for outcome in CHECKPOINT_RESTORE_LINEAGE_OUTCOMES {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_checkpoint_restore_lineage_total",
                    &[("outcome", outcome), ("service", "seed-test")],
                ),
                Some(0.0),
                "outcome `{outcome}` must exist and read 0 on the first scrape; a missing series \
                 cannot support the \"DR has never stepped down\" assertion"
            );
        }
        assert_eq!(
            CHECKPOINT_RESTORE_LINEAGE_OUTCOMES.len(),
            10,
            "ten selection outcomes; one added at an increment site but not here materialises \
             its series only once that step-down first fires"
        );
    }

    /// The two halves `checkpoint_restore_lineage_cells_are_zero_seeded` leaves
    /// open, modelled on `checkpoint_tail_outcome_cells_are_zero_seeded`: the
    /// **exact** vocabulary slice, and the label set pinned directly.
    ///
    /// Why each is load-bearing:
    ///
    /// - A `len() == 10` check passes against a list where a member was *renamed*
    ///   at the const and left un-renamed at the minting site
    ///   (`checkpoint::lineage`'s `outcome_label` / `reason_label`). The
    ///   exact-slice comparison is what makes a reason invented at a call site
    ///   rather than added here a red test, instead of a series that
    ///   materialises only on first occurrence — the precise failure the
    ///   zero-seeding exists to prevent.
    /// - `test_util::sample_value` matches labels as a SUBSET, so every
    ///   per-cell assertion elsewhere would also pass against cells carrying
    ///   extra labels. This family's decision — unlike the deliberately
    ///   service-free `checkpoint_tail_total` — is that it DOES carry `service`,
    ///   because a step-down is attributable to the replica that served the
    ///   restore, exactly as `checkpoint_recovery_attempts_total` is. Pin that
    ///   choice, and the absence of anything else, directly off the rendered
    ///   series.
    #[test]
    fn checkpoint_restore_lineage_vocabulary_and_label_set_are_pinned() {
        // Vocabulary pin. `gathered_*` are the two resolved arms, the six
        // `stepped_down_*` are `LineageBreak`'s variants one-for-one (no two
        // merged — each names a distinct operator remediation), and the two
        // `exhausted*` members split `Exhaustion`'s four variants by whether
        // the selection reached a CONCLUSIVE verdict: `exhausted` folds the
        // three that did (`EmptyListing` / `NoBase` / `AllBroken`, whose
        // remaining split is diagnostic rather than dashboard-visible), and
        // `exhausted_budget` is `BudgetInconclusive` alone.
        //
        // `exhausted_budget` is NOT foldable into `exhausted` (b.hxk): the two
        // differ in the answer the client got — terminal `NOT_FOUND` for
        // `exhausted`, retriable `UNAVAILABLE` for `exhausted_budget`, because
        // only the conclusive arm holds evidence that the layer is
        // unrestorable. Folding them would hide "this layer has no usable
        // backup" inside "the store was too slow to say".
        //
        // `stepped_down_missing_meta` sits beside `stepped_down_decode` and is
        // deliberately NOT folded into it (b.hka (d)): a durable artifact whose
        // meta companion never landed is a torn put, remediated by letting the
        // loop write a current generation, while a decode failure is bit-rot or
        // version skew. Folding them would report a routine write-path
        // interruption as storage corruption.
        assert_eq!(
            CHECKPOINT_RESTORE_LINEAGE_OUTCOMES,
            [
                "gathered_base",
                "gathered_composed",
                "stepped_down_link_broken",
                "stepped_down_link_mismatch",
                "stepped_down_decode",
                "stepped_down_missing_meta",
                "stepped_down_budget",
                "stepped_down_compose",
                "exhausted",
                "exhausted_budget",
            ],
            "the restore-lineage outcome vocabulary is pinned; a member added at an increment \
             site but not here renders only once that step-down first fires, and a member \
             renamed here but not at `checkpoint::lineage`'s label methods opens an unseeded \
             series"
        );

        let metrics = metrics_scrape_only("label-set-test");
        let text = render(&metrics);
        let series: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("live_edit_checkpoint_restore_lineage_total{"))
            .collect();
        assert_eq!(
            series.len(),
            CHECKPOINT_RESTORE_LINEAGE_OUTCOMES.len(),
            "the family must render one seeded cell per vocabulary member on a FIRST scrape, \
             before any restore has run; got {series:?}"
        );

        for line in &series {
            let body = line
                .split_once('{')
                .and_then(|(_, rest)| rest.split_once('}'))
                .map(|(labels, _)| labels)
                .unwrap_or_else(|| panic!("malformed series line: {line}"));
            let mut keys: Vec<&str> = body
                .split(',')
                .filter_map(|pair| pair.split_once('='))
                .map(|(key, _)| key.trim())
                .collect();
            keys.sort_unstable();
            assert_eq!(
                keys,
                ["outcome", "service"],
                "the label set must mirror the increment site (`inc_checkpoint_restore_lineage` \
                 emits `outcome` + `service` and nothing else). An extra label would make every \
                 subset matcher in this file read the wrong cell; a missing `service` would \
                 break attribution to the replica that served the restore. Got: {line}"
            );
        }
    }

    /// b.4et: the `reason` vocabulary of
    /// `live_edit_checkpoint_recovery_stream_unresolved_total`, pinned as an
    /// exact slice.
    ///
    /// A `len()` check would pass against a list where a member was *renamed*
    /// at the const and left un-renamed at the increment site
    /// (`classify_unreadable_stream` / `refuse_restore` / `remove_stale_stream`
    /// in `src/checkpoint/recovery.rs`), which opens an unseeded series at the
    /// exact moment of an incident. The exact-slice comparison is what makes
    /// that a red test rather than a silently-late dashboard.
    ///
    /// **No two members may be merged.** `probe_unreadable` and
    /// `unlink_unverified` both END the restore, but at different steps and
    /// with different remediations — the first is "Valkey health, the probe
    /// could not read the key", the second is "something is re-creating that
    /// stream key out of band". `wrongtype_key` is the odd one out entirely:
    /// the restore PROCEEDED, and what the operator must act on is a layer that
    /// came back readable and permanently unwritable. Folding any two of the
    /// three would send an operator at the wrong key.
    #[test]
    fn checkpoint_recovery_stream_unresolved_vocabulary_is_pinned() {
        assert_eq!(
            CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS,
            ["probe_unreadable", "unlink_unverified", "wrongtype_key"],
            "the b.4et unresolved-reason vocabulary is pinned; a reason added at an increment \
             site but not here renders only once that arm first fires, and a reason renamed here \
             but not at the increment site opens an unseeded series"
        );
    }

    /// Every cell of the b.4et family reads ZERO from the FIRST scrape, and the
    /// label set is exactly `{reason, service}`.
    ///
    /// The label-set half is not redundant with the zero-seed half:
    /// `test_util::sample_value` matches labels as a SUBSET, so every per-cell
    /// assertion in this file and in the integration suite would also pass
    /// against cells carrying an extra label. Pin the choice directly off the
    /// rendered series. `service` is present for the same reason
    /// `checkpoint_recovery_attempts_total` carries it: a refused restore is
    /// attributable to the replica that served it.
    #[test]
    fn checkpoint_recovery_stream_unresolved_cells_are_zero_seeded() {
        let metrics = metrics_scrape_only("seed-test");
        let text = render(&metrics);

        for reason in CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_checkpoint_recovery_stream_unresolved_total",
                    &[("reason", reason), ("service", "seed-test")],
                ),
                Some(0.0),
                "reason `{reason}` must exist and read 0 on the first scrape; a missing series \
                 cannot support the \"DR has never failed to establish a stream's state\" \
                 assertion, which is the only reason to publish this family at all"
            );
        }

        let series: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("live_edit_checkpoint_recovery_stream_unresolved_total{"))
            .collect();
        assert_eq!(
            series.len(),
            CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS.len(),
            "the family must render one seeded cell per vocabulary member on a FIRST scrape, \
             before any restore has run; got {series:?}"
        );
        for line in &series {
            let body = line
                .split_once('{')
                .and_then(|(_, rest)| rest.split_once('}'))
                .map(|(labels, _)| labels)
                .unwrap_or_else(|| panic!("malformed series line: {line}"));
            let mut keys: Vec<&str> = body
                .split(',')
                .filter_map(|pair| pair.split_once('='))
                .map(|(key, _)| key.trim())
                .collect();
            keys.sort_unstable();
            assert_eq!(
                keys,
                ["reason", "service"],
                "the label set must mirror `inc_checkpoint_recovery_stream_unresolved`, which \
                 emits `reason` + `service` and nothing else. An extra label would make every \
                 subset matcher elsewhere read the wrong cell; a missing `service` would break \
                 attribution to the replica that refused the restore. Got: {line}"
            );
        }
    }

    /// The typo guard the zero-seed test structurally cannot provide: driving
    /// `inc_checkpoint_recovery_stream_unresolved` once per vocabulary member
    /// must open exactly one cell per member and no more.
    ///
    /// An increment with an unlisted reason ADDS a series on top of the seeded
    /// ones, so every per-cell assertion keeps passing while a dashboard grows
    /// a row nobody documented. Only a series count can see it.
    #[test]
    fn inc_checkpoint_recovery_stream_unresolved_opens_no_series_outside_the_vocabulary() {
        let metrics = metrics_scrape_only("typo-guard-test");
        for reason in CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS {
            metrics.inc_checkpoint_recovery_stream_unresolved(reason, "typo-guard-test");
        }
        let text = render(&metrics);

        let series: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("live_edit_checkpoint_recovery_stream_unresolved_total{"))
            .collect();
        assert_eq!(
            series.len(),
            CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS.len(),
            "driving every vocabulary member must leave exactly {} label cells; got {} series: \
             {series:?}",
            CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS.len(),
            series.len()
        );
        for reason in CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_checkpoint_recovery_stream_unresolved_total",
                    &[("reason", reason), ("service", "typo-guard-test")],
                ),
                Some(1.0),
                "reason `{reason}` must read exactly 1 after one increment — a value of 2 would \
                 mean two vocabulary members collapsed onto one cell"
            );
        }
    }

    /// The identity `checkpoint_recovery_attempts_total` owes its consumers is
    /// untouched by the new family: four `result` values, two `scope` values,
    /// and `sum by (result)` still equals the attempt count.
    #[test]
    fn the_recovery_attempts_grid_is_unchanged_by_the_restore_lineage_family() {
        let metrics = metrics_scrape_only("grid-test");
        // One attempt, which also steps down twice on its way to a verdict.
        metrics.inc_checkpoint_recovery_attempt("restored", "layer", "grid-test");
        metrics.inc_checkpoint_restore_lineage("stepped_down_link_broken", "grid-test");
        metrics.inc_checkpoint_restore_lineage("stepped_down_budget", "grid-test");
        metrics.inc_checkpoint_restore_lineage("gathered_base", "grid-test");
        let text = render(&metrics);

        let attempts: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("live_edit_checkpoint_recovery_attempts_total{"))
            .collect();
        assert_eq!(
            attempts.len(),
            8,
            "four results x two scopes; the step-downs must NOT have opened a fifth result cell: \
             {attempts:?}"
        );
        let total: f64 = ["restored", "not_recoverable", "error", "already_present"]
            .iter()
            .filter_map(|result| {
                sample_value(
                    &text,
                    "live_edit_checkpoint_recovery_attempts_total",
                    &[
                        ("result", result),
                        ("scope", "layer"),
                        ("service", "grid-test"),
                    ],
                )
            })
            .sum();
        assert_eq!(
            total, 1.0,
            "`sum by (result)` must still equal the attempt count — three lineage increments \
             belong to ONE attempt"
        );
    }

    #[test]
    fn update_layer_attempts_histogram_pins_64_and_128_buckets() {
        // b.avd item 6 regression: a commit landing on attempt 100 must fall
        // inside the 128 bucket, not the +Inf overflow — proving the explicit
        // bucket view survived the migration for this histogram.
        let metrics = metrics_scrape_only("test-pod");
        metrics.observe_update_layer_attempts(UPDATE_LAYER_PATH_WATCH, 100);
        let text = render(&metrics);
        let name = "live_edit_update_layer_attempts_histogram";
        let watch = [("path", UPDATE_LAYER_PATH_WATCH)];
        assert_eq!(
            histogram_bucket(&text, name, "64", &watch),
            Some(0.0),
            "a 100-attempt observation must NOT be counted in the 64 bucket"
        );
        assert_eq!(
            histogram_bucket(&text, name, "128", &watch),
            Some(1.0),
            "a 100-attempt observation must land inside the 128 bucket"
        );
        assert_eq!(
            histogram_bucket(&text, name, "+Inf", &watch),
            Some(1.0),
            "the sample must be present in the +Inf bucket total"
        );
        assert_eq!(
            sample_value(
                &text,
                &format!("{name}_count"),
                &[("path", UPDATE_LAYER_PATH_FAST)]
            ),
            None,
            "the observation is labelled with the path it was given — no `path=\"fast\"` series \
             may appear"
        );
    }

    /// `live_edit_update_layer_commit_duration_seconds` renders its explicit
    /// bucket view, `UPDATE_LAYER_COMMIT_DURATION_BUCKETS` (Spec SDD SR-2.2),
    /// not OTel's defaults: the full rendered boundary list is pinned, and so
    /// are the two ends that matter. A 2 ms call lands above `le="0.001"`,
    /// a boundary the defaults lack; a 20 s call — one that ran most of the
    /// 30 s default RPC deadline — lands in the finite (16, 32] band rather
    /// than only in `+Inf`. Re-sizing the list deliberately means updating this
    /// test in the same commit: the boundaries are the dashboard's x-axis.
    #[test]
    fn update_layer_commit_duration_histogram_renders_its_explicit_buckets_and_bins_a_20_s_call_under_32(
    ) {
        let name = "live_edit_update_layer_commit_duration_seconds";
        let labels = [
            ("path", UPDATE_LAYER_PATH_WATCH),
            ("outcome", UPDATE_LAYER_COMMIT_OUTCOME_OK),
        ];

        let metrics = metrics_scrape_only("test-pod");
        metrics.observe_update_layer_commit_duration(
            UPDATE_LAYER_PATH_WATCH,
            UPDATE_LAYER_COMMIT_OUTCOME_OK,
            0.002,
        );
        let text = render(&metrics);
        assert_eq!(
            bucket_le_strings(&text, name, ""),
            [
                "0.001", "0.0025", "0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1",
                "2", "4", "8", "16", "32", "+Inf",
            ],
            "the commit-duration histogram must render UPDATE_LAYER_COMMIT_DURATION_BUCKETS \
             ({UPDATE_LAYER_COMMIT_DURATION_BUCKETS:?})"
        );
        assert_eq!(
            histogram_bucket(&text, name, "0.001", &labels),
            Some(0.0),
            "a 2 ms call must NOT be counted in the 1 ms bucket"
        );
        assert_eq!(
            histogram_bucket(&text, name, "0.0025", &labels),
            Some(1.0),
            "a 2 ms call must land inside the 2.5 ms bucket"
        );

        let metrics = metrics_scrape_only("test-pod");
        metrics.observe_update_layer_commit_duration(
            UPDATE_LAYER_PATH_WATCH,
            UPDATE_LAYER_COMMIT_OUTCOME_OK,
            20.0,
        );
        let text = render(&metrics);
        assert_eq!(
            histogram_bucket(&text, name, "16", &labels),
            Some(0.0),
            "a 20 s call is above le=\"16\""
        );
        assert_eq!(
            histogram_bucket(&text, name, "32", &labels),
            Some(1.0),
            "a 20 s call must land in the finite (16, 32] band, not only in +Inf"
        );
    }

    /// Both b.f4f histograms must keep their EXPLICIT bucket views.
    ///
    /// `histogram_bucket_boundaries` maps each family name to its boundary list;
    /// delete either arm and the instrument silently falls back to OTel's default
    /// boundaries — which start at 0 and step 5, 10, 25, 50 SECONDS. Both of these
    /// measure microseconds-to-seconds, so the fallback would put every real
    /// observation in the first bucket and every pathological one in `+Inf`,
    /// leaving a histogram that renders, scrapes, and answers no question. There
    /// is no error and no absent series to notice it by; only the boundaries
    /// themselves show it.
    ///
    /// Each half asserts a boundary that the OTel defaults do **not** contain, so
    /// the fallback cannot satisfy it, plus the full rendered boundary list
    /// against the constant — which is what makes a deliberate re-sizing fail
    /// here. **That is intended: re-sizing either list is an operator-visible
    /// change to a dashboard's shape, and it must update this test in the same
    /// commit.**
    #[test]
    fn the_b_f4f_histograms_keep_their_explicit_bucket_views() {
        /// `le` labels as Prometheus renders them: decimal bounds verbatim,
        /// integer-valued bounds without a trailing `.0`.
        fn rendered_boundaries(text: &str, name: &str) -> Vec<String> {
            text.lines()
                .filter(|l| l.trim_start().starts_with(&format!("{name}_bucket")))
                .filter_map(|l| {
                    let start = l.find("le=\"")? + 4;
                    let rest = &l[start..];
                    let end = rest.find('"')?;
                    Some(rest[..end].to_string())
                })
                .collect()
        }

        let metrics = metrics_scrape_only("test-pod");
        metrics.observe_valkey_pool_checkout("txn", Duration::from_micros(200));
        metrics.observe_subscriber_admission(ADMISSION_RESULT_ADMITTED, Duration::from_millis(30));
        let text = render(&metrics);

        // ── Checkout latency ────────────────────────────────────────────────
        let checkout_name = "live_edit_valkey_pool_checkout_seconds";
        assert_eq!(
            rendered_boundaries(&text, checkout_name),
            [
                "0.00005", "0.0001", "0.0005", "0.001", "0.005", "0.01", "0.05", "0.1", "0.5", "1",
                "5", "+Inf",
            ],
            "the checkout histogram's bucket view must match POOL_CHECKOUT_BUCKETS \
             ({POOL_CHECKOUT_BUCKETS:?}). If you re-sized it deliberately, update this list in \
             the same commit — the boundaries are the dashboard's x-axis",
        );
        assert_eq!(
            histogram_bucket(&text, checkout_name, "0.00005", &[("pool", "txn")]),
            Some(0.0),
            "a 200 us checkout must NOT be counted in the 50 us bucket. This boundary is absent \
             from OTel's defaults, so its mere presence proves the explicit view survived"
        );
        assert_eq!(
            histogram_bucket(&text, checkout_name, "0.0005", &[("pool", "txn")]),
            Some(1.0),
            "a 200 us checkout must land inside the 500 us bucket"
        );

        // ── Admission wait ──────────────────────────────────────────────────
        let admission_name = "live_edit_subscriber_admission_wait_seconds";
        assert_eq!(
            rendered_boundaries(&text, admission_name),
            [
                "0.0001", "0.001", "0.01", "0.05", "0.1", "0.25", "0.5", "1", "2", "5", "10", "20",
                "30", "+Inf",
            ],
            "the admission-wait histogram's bucket view must match ADMISSION_WAIT_BUCKETS \
             ({ADMISSION_WAIT_BUCKETS:?}). The `20` boundary in particular brackets the 20000 ms \
             default LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS, so a refusal at the wait limit is \
             visible rather than buried in +Inf — re-sizing deliberately means updating this \
             list here too",
        );
        assert_eq!(
            histogram_bucket(&text, admission_name, "0.01", &[]),
            Some(0.0),
            "a 30 ms wait must NOT be counted in the 10 ms bucket. Like the checkout half, this \
             boundary is absent from OTel's defaults"
        );
        assert_eq!(
            histogram_bucket(&text, admission_name, "0.05", &[]),
            Some(1.0),
            "a 30 ms wait must land inside the 50 ms bucket"
        );
    }

    /// Both capacity-wait histograms — the admission wait and the catch-up
    /// slot wait — share `ADMISSION_WAIT_BUCKETS`, whose `20` and `30`
    /// boundaries bracket the 20 s default capacity budget: a 19 s wait lands
    /// in `le="20"`, and a 20.5 s one (a wait that ran out the whole budget)
    /// only in the (20, 30] band. Without the two boundaries both would sit in
    /// `+Inf`, indistinguishable from each other and from an unbounded wait.
    #[test]
    fn both_capacity_wait_histograms_bracket_the_20_s_budget_with_their_20_and_30_buckets() {
        /// One observation of `waited` on `family`, in a fresh registry so the
        /// family carries exactly one series: the admission wait is unlabelled,
        /// the catch-up wait is labelled by `result`.
        fn scrape_one(family: &str, waited: Duration) -> String {
            let metrics = metrics_scrape_only("test-pod");
            match family {
                "live_edit_subscriber_admission_wait_seconds" => {
                    metrics.observe_subscriber_admission(ADMISSION_RESULT_ADMITTED, waited)
                }
                "live_edit_subscribe_catchup_wait_seconds" => {
                    metrics.observe_subscribe_catchup_wait(CATCHUP_WAIT_RESULT_TIMED_OUT, waited)
                }
                other => panic!("not a capacity-wait histogram: {other}"),
            }
            render(&metrics)
        }

        let expected_le = [
            "0.0001", "0.001", "0.01", "0.05", "0.1", "0.25", "0.5", "1", "2", "5", "10", "20",
            "30", "+Inf",
        ];
        for family in [
            "live_edit_subscriber_admission_wait_seconds",
            "live_edit_subscribe_catchup_wait_seconds",
        ] {
            let within = scrape_one(family, Duration::from_secs(19));
            assert_eq!(
                bucket_le_strings(&within, family, ""),
                expected_le,
                "{family} must render ADMISSION_WAIT_BUCKETS, the one list both capacity-wait \
                 histograms share"
            );
            assert_eq!(
                histogram_bucket(&within, family, "10", &[]),
                Some(0.0),
                "{family}: a 19 s wait is above le=\"10\""
            );
            assert_eq!(
                histogram_bucket(&within, family, "20", &[]),
                Some(1.0),
                "{family}: a 19 s wait lands in the finite le=\"20\" bucket"
            );

            let ran_out = scrape_one(family, Duration::from_millis(20_500));
            assert_eq!(
                histogram_bucket(&ran_out, family, "20", &[]),
                Some(0.0),
                "{family}: a 20.5 s wait ran out the 20 s budget and is above le=\"20\""
            );
            assert_eq!(
                histogram_bucket(&ran_out, family, "30", &[]),
                Some(1.0),
                "{family}: a 20.5 s wait lands in the (20, 30] band, not only in +Inf"
            );
        }
    }

    /// The b.qpf leak signal counts REFERENCES, not enqueue calls: a failed
    /// enqueue of a commit's N owed releases adds N.
    ///
    /// The documented reading (`CHANGELOG.md`; the
    /// `docs/service/docs/metrics-reference.md` row counts releases that could
    /// not be recorded) is "every tick is one reference the service owed and
    /// could not record", and a `failed / released` ratio divides this by
    /// `live_edit_ext_ref_released_total`, which is per reference. A
    /// per-call `add(1, ..)` would make a three-field rewrite report one tick
    /// while leaking three, breaking both readings — and it would look healthy
    /// doing it, which is why the arithmetic is pinned here.
    #[test]
    fn ext_ref_release_enqueue_failed_counts_references_not_calls() {
        let metrics = metrics_scrape_only("test-pod");
        let name = "live_edit_ext_ref_release_enqueue_failed_total";
        assert_eq!(
            sample_value(&render(&metrics), name, &[]),
            Some(0.0),
            "the leak signal is zero-seeded so `increase(...) > 0` has a baseline from boot"
        );

        // One failed enqueue, three owed references.
        metrics.inc_ext_ref_release_enqueue_failed(3);
        assert_eq!(
            sample_value(&render(&metrics), name, &[]),
            Some(3.0),
            "a single failed enqueue of three releases must add three, not one"
        );

        // A second failure accumulates in the same unit.
        metrics.inc_ext_ref_release_enqueue_failed(2);
        assert_eq!(
            sample_value(&render(&metrics), name, &[]),
            Some(5.0),
            "ticks accumulate per reference across failures"
        );

        // Zero is a no-op so the call site stays branch-free.
        metrics.inc_ext_ref_release_enqueue_failed(0);
        assert_eq!(
            sample_value(&render(&metrics), name, &[]),
            Some(5.0),
            "adding zero owed references must not move the counter"
        );
    }

    // ══ b.yda: the reference-count reconciler's observability surface ═══════

    /// The `reason` vocabulary of `ext_ref_released_total` grew a fourth cell.
    ///
    /// Pinned as a literal here rather than compared against itself: this is the
    /// unit-tier half of the const-drift guard, and the integration-tier
    /// cardinality pin in `tests/integration/observability_metrics_test.rs`
    /// restates the same four values independently. Only one of the two can
    /// catch an edit to the const, and this is it.
    #[test]
    fn ext_ref_release_reasons_vocabulary_is_pinned() {
        assert_eq!(
            EXT_REF_RELEASE_REASONS,
            ["rewrite", "surplus_acquire", "retirement", "reconcile"],
            "three commit/retirement reasons plus b.yda's out-of-band `reconcile`. The fourth is \
             deliberately distinct from the other three: it names a reference handed back by a \
             reconciler that could not attribute it to any object, so a non-zero rate on it is \
             the documented leak-direction residuals draining, never routine traffic"
        );
    }

    /// The `site` vocabulary of `ext_ref_undecodable_total` is pinned.
    ///
    /// Four corrupt-keyspace drop sites, each naming a place a reference is
    /// stranded because its hash could not be decoded. The values are read off
    /// a dashboard to tell "the Lua backend returned a garbled token" from "a
    /// value row's `hash` column is corrupt", so a rename is an operator-facing
    /// break.
    #[test]
    fn ext_ref_undecodable_sites_vocabulary_is_pinned() {
        assert_eq!(
            EXT_REF_UNDECODABLE_SITES,
            [
                "lua_token",
                "row_hash",
                "ext_index_field",
                "retirement_index_field"
            ],
            "the four sites at which a reference is dropped because nothing can name the blob it \
             points at. `retirement_index_field` is the worst of the four — the object's keys are \
             about to be UNLINKed, so nothing can ever re-derive the hash"
        );
    }

    /// The `outcome` vocabulary of `ext_reconcile_total` is pinned at ten.
    ///
    /// The three `skipped_*` cells and the two deficit cells are the ones a
    /// rename would quietly break: `skipped_recent_store` is the pass's normal
    /// busy-store ineligibility, `skipped_unreadable_stamp` is a
    /// keyspace-integrity signal that must not be confused with it (b.rej
    /// renamed it from `skipped_no_stamp` when an ABSENT stamp became eligible),
    /// `skipped_index_drift` is a surplus held because an object's per-object
    /// reference index could not be repaired, and `deficit_observed` is the
    /// alertable one whose recipe reads recurrence.
    #[test]
    fn ext_reconcile_outcomes_vocabulary_is_pinned() {
        assert_eq!(
            EXT_RECONCILE_OUTCOMES,
            [
                "candidate_registered",
                "repaired",
                "healed",
                "deferred",
                "skipped_recent_store",
                "skipped_unreadable_stamp",
                "enumeration_failed",
                "deficit_observed",
                "deficit_cleared",
                "skipped_index_drift",
            ],
            "ten outcomes. An eleventh added at an increment site but not here materialises its \
             series only once that path first fires, which is exactly when an alert most needs a \
             baseline to have existed"
        );
        assert_eq!(EXT_RECONCILE_OUTCOMES.len(), 10);
    }

    /// The `site`, `outcome` and `direction` vocabularies of the per-object
    /// reference-index recount counters are pinned.
    ///
    /// Literals rather than a comparison against the consts themselves, for the
    /// reason [`ext_ref_release_reasons_vocabulary_is_pinned`] gives: the label
    /// values are the operator-facing contract (`direction="under"` is the
    /// documented "something still produces drift" signal), and only a literal
    /// can catch a rename of the const.
    #[test]
    fn ext_index_recount_vocabularies_are_pinned() {
        assert_eq!(
            EXT_INDEX_RECOUNT_SITES,
            ["reconcile", "torn_commit"],
            "two call sites: the reconciler's object walk and the txn path's torn-EXEC arm"
        );
        assert_eq!(
            EXT_INDEX_RECOUNT_OUTCOMES,
            [
                "repaired",
                "clean",
                "too_large",
                "not_live",
                "failed",
                "withheld"
            ],
            "six outcomes. `too_large`, `not_live` and `withheld` at site=reconcile are the three \
             that hold a surplus from drain, so a rename of any of them breaks the \
             `skipped_index_drift` reading; `withheld` is also the operator-facing signal of a \
             malformed hash column"
        );
        assert_eq!(
            EXT_INDEX_DRIFT_ONLY_SITES,
            ["structural_settle"],
            "one drift-only site: the structural commit's raise-only `obj:ext` settle"
        );
        assert_eq!(
            EXT_INDEX_DRIFT_DIRECTIONS,
            ["under", "over"],
            "`under` is the dangerous direction (a later commit can release a reference a live \
             row holds); `over` is a leak"
        );
    }

    /// Both recount counters zero-seed EXACTLY the (site, outcome) and
    /// (site, direction) products, and nothing else.
    ///
    /// Same argument as the `ext_reconcile_total` seeding: an alert on
    /// `direction="under"` or `outcome="failed"` needs its series to exist at
    /// zero before the first occurrence, and a two-label family can seed every
    /// value of each label while still missing a PAIR.
    #[test]
    fn ext_index_recount_counters_zero_seed_every_label_pair() {
        let metrics = metrics_scrape_only("test-pod");
        let text = render(&metrics);

        let count_series = |name: &str| {
            text.lines()
                .map(str::trim)
                .filter(|l| l.starts_with(&format!("{name}{{")))
                .count()
        };
        assert_eq!(
            count_series("live_edit_ext_index_recount_total"),
            EXT_INDEX_RECOUNT_SITES.len() * EXT_INDEX_RECOUNT_OUTCOMES.len(),
            "one seeded series per (site, outcome) pair and no others"
        );
        assert_eq!(
            count_series("live_edit_ext_index_drift_total"),
            (EXT_INDEX_RECOUNT_SITES.len() + EXT_INDEX_DRIFT_ONLY_SITES.len())
                * EXT_INDEX_DRIFT_DIRECTIONS.len(),
            "one seeded series per (site, direction) pair and no others"
        );
        for site in EXT_INDEX_DRIFT_ONLY_SITES {
            for direction in EXT_INDEX_DRIFT_DIRECTIONS {
                assert_eq!(
                    sample_value(
                        &text,
                        "live_edit_ext_index_drift_total",
                        &[("site", site), ("direction", direction)]
                    ),
                    Some(0.0),
                    "drift-only site=\"{site}\", direction=\"{direction}\" must be seeded at zero"
                );
            }
        }
        for site in EXT_INDEX_RECOUNT_SITES {
            for outcome in EXT_INDEX_RECOUNT_OUTCOMES {
                assert_eq!(
                    sample_value(
                        &text,
                        "live_edit_ext_index_recount_total",
                        &[("site", site), ("outcome", outcome)]
                    ),
                    Some(0.0),
                    "site=\"{site}\", outcome=\"{outcome}\" must be seeded at zero"
                );
            }
            for direction in EXT_INDEX_DRIFT_DIRECTIONS {
                assert_eq!(
                    sample_value(
                        &text,
                        "live_edit_ext_index_drift_total",
                        &[("site", site), ("direction", direction)]
                    ),
                    Some(0.0),
                    "site=\"{site}\", direction=\"{direction}\" must be seeded at zero"
                );
            }
        }
        // Named, not only iterated: `withheld` is the cell an operator alerts
        // on for a malformed hash column, and the loop above reads its list
        // from the same const the seeding reads, so a cell dropped from both
        // would pass it.
        for site in ["reconcile", "torn_commit"] {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_ext_index_recount_total",
                    &[("site", site), ("outcome", "withheld")]
                ),
                Some(0.0),
                "site=\"{site}\", outcome=\"withheld\" must be seeded at zero"
            );
        }
    }

    /// `inc_ext_index_recount` adds one to exactly the named (site, outcome)
    /// cell, and `inc_ext_index_drift` adds its count to exactly the named
    /// (site, direction) cell and no-ops on zero.
    ///
    /// The no-op is what lets a repair report both directions unconditionally;
    /// the per-cell isolation is what keeps a site or direction swap from
    /// passing — every other cell must still read zero.
    #[test]
    fn ext_index_recount_increments_land_on_exactly_the_named_cell() {
        let metrics = metrics_scrape_only("test-pod");

        metrics.inc_ext_index_recount(
            EXT_INDEX_RECOUNT_SITE_TORN_COMMIT,
            EXT_INDEX_RECOUNT_OUTCOME_REPAIRED,
        );
        metrics.inc_ext_index_drift(
            EXT_INDEX_RECOUNT_SITE_RECONCILE,
            EXT_INDEX_DRIFT_DIRECTION_UNDER,
            3,
        );
        metrics.inc_ext_index_drift(
            EXT_INDEX_RECOUNT_SITE_RECONCILE,
            EXT_INDEX_DRIFT_DIRECTION_OVER,
            0,
        );
        let text = render(&metrics);

        for site in EXT_INDEX_RECOUNT_SITES {
            for outcome in EXT_INDEX_RECOUNT_OUTCOMES {
                let want = if (*site, *outcome) == ("torn_commit", "repaired") {
                    1.0
                } else {
                    0.0
                };
                assert_eq!(
                    sample_value(
                        &text,
                        "live_edit_ext_index_recount_total",
                        &[("site", site), ("outcome", outcome)]
                    ),
                    Some(want),
                    "site=\"{site}\", outcome=\"{outcome}\""
                );
            }
            for direction in EXT_INDEX_DRIFT_DIRECTIONS {
                let want = if (*site, *direction) == ("reconcile", "under") {
                    3.0
                } else {
                    0.0
                };
                assert_eq!(
                    sample_value(
                        &text,
                        "live_edit_ext_index_drift_total",
                        &[("site", site), ("direction", direction)]
                    ),
                    Some(want),
                    "site=\"{site}\", direction=\"{direction}\": one call reporting three fields \
                     adds three, and a zero-count call moves nothing"
                );
            }
        }
    }

    /// Driving `inc_ext_reconcile` once per vocabulary member opens exactly as
    /// many label cells as the vocabulary has members.
    ///
    /// Zero-seeding alone cannot catch a hardcoded outcome string at a call
    /// site: an unlisted value adds a series on top of the seeded ones and every
    /// per-cell assertion keeps passing.
    #[test]
    fn inc_ext_reconcile_opens_no_series_outside_the_pinned_vocabulary() {
        let metrics = metrics_scrape_only("typo-guard-test");
        for outcome in EXT_RECONCILE_OUTCOMES {
            metrics.inc_ext_reconcile(outcome, 1);
        }
        let text = render(&metrics);

        let series: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("live_edit_ext_reconcile_total{"))
            .collect();
        assert_eq!(
            series.len(),
            EXT_RECONCILE_OUTCOMES.len(),
            "driving every vocabulary member must leave exactly {} label cells; got {} series: \
             {series:?}",
            EXT_RECONCILE_OUTCOMES.len(),
            series.len()
        );

        for outcome in EXT_RECONCILE_OUTCOMES {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_ext_reconcile_total",
                    &[("outcome", outcome)]
                ),
                Some(1.0),
                "outcome `{outcome}` must read exactly 1 after one increment — a 2 would mean two \
                 vocabulary members collapsed onto one cell"
            );
        }
    }

    /// The same guard for `ext_ref_undecodable_total{site}`.
    #[test]
    fn inc_ext_ref_undecodable_opens_no_series_outside_the_pinned_vocabulary() {
        let metrics = metrics_scrape_only("typo-guard-test");
        for site in EXT_REF_UNDECODABLE_SITES {
            metrics.inc_ext_ref_undecodable(site);
        }
        let text = render(&metrics);

        let series: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with("live_edit_ext_ref_undecodable_total{"))
            .collect();
        assert_eq!(
            series.len(),
            EXT_REF_UNDECODABLE_SITES.len(),
            "driving every vocabulary member must leave exactly {} label cells; got {} series: \
             {series:?}",
            EXT_REF_UNDECODABLE_SITES.len(),
            series.len()
        );

        for site in EXT_REF_UNDECODABLE_SITES {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_ext_ref_undecodable_total",
                    &[("site", site)]
                ),
                Some(1.0),
                "site `{site}` must read exactly 1 after one increment"
            );
        }
    }

    /// The `inc_ext_reconcile` counter takes a COUNT, and `0` is a no-op.
    ///
    /// The pass reports whole phases at once (`candidates_registered`, and both
    /// `skipped_*` tallies) and reports every cell unconditionally at the end of
    /// a phase, so the no-op on zero is what keeps the call sites branch-free.
    #[test]
    fn inc_ext_reconcile_adds_the_count_and_no_ops_on_zero() {
        let metrics = metrics_scrape_only("test-pod");
        let cell = &[("outcome", EXT_RECONCILE_OUTCOME_CANDIDATE_REGISTERED)];

        metrics.inc_ext_reconcile(EXT_RECONCILE_OUTCOME_CANDIDATE_REGISTERED, 7);
        assert_eq!(
            sample_value(&render(&metrics), "live_edit_ext_reconcile_total", cell),
            Some(7.0),
            "one call reporting seven candidates must add seven, not one"
        );

        metrics.inc_ext_reconcile(EXT_RECONCILE_OUTCOME_CANDIDATE_REGISTERED, 0);
        assert_eq!(
            sample_value(&render(&metrics), "live_edit_ext_reconcile_total", cell),
            Some(7.0),
            "reporting zero of an outcome must not move the cell — the pass ticks every cell \
             unconditionally at the end of a phase"
        );
    }

    /// The four retire-carrier lifecycle counters are unlabelled, zero-seeded,
    /// and count REFERENCES.
    ///
    /// They are read together: `deferred` is approximately `recovered +
    /// cancelled + already_enqueued` (b.yuo Part R added the last: a carrier
    /// whose earlier attempt is found landed is cleared, not paid), and a
    /// persistent gap is an obligation stuck on a carrier. All four must
    /// therefore share one unit, and all four must exist at zero from boot or
    /// the arithmetic is undefined on a fresh replica.
    #[test]
    fn ext_ref_retire_carrier_counters_are_unlabelled_and_count_references() {
        let metrics = metrics_scrape_only("test-pod");
        let names = [
            "live_edit_ext_ref_release_deferred_total",
            "live_edit_ext_ref_release_recovered_total",
            "live_edit_ext_ref_release_cancelled_total",
            "live_edit_ext_ref_release_already_enqueued_total",
        ];
        let text = render(&metrics);
        for name in names {
            assert_eq!(
                sample_value(&text, name, &[]),
                Some(0.0),
                "{name} must be zero-seeded: the four are read as `deferred ~= recovered + \
                 cancelled + already_enqueued`, and an absent term makes that arithmetic \
                 undefined on a fresh replica"
            );
        }

        metrics.inc_ext_ref_release_deferred(4);
        metrics.inc_ext_ref_release_recovered(2);
        metrics.inc_ext_ref_release_cancelled(1);
        metrics.inc_ext_ref_release_already_enqueued(3);
        let text = render(&metrics);
        assert_eq!(
            sample_value(&text, names[0], &[]),
            Some(4.0),
            "a deferral of four owed references must add four — one tick per REFERENCE, the \
             unit `ext_ref_released_total` and the leak signal already use"
        );
        assert_eq!(sample_value(&text, names[1], &[]), Some(2.0));
        assert_eq!(sample_value(&text, names[2], &[]), Some(1.0));
        assert_eq!(
            sample_value(&text, names[3], &[]),
            Some(3.0),
            "a carrier of three references found already paid must add three, in the same unit"
        );

        for (name, add) in names.iter().zip([4.0, 2.0, 1.0, 3.0]) {
            metrics.inc_ext_ref_release_deferred(0);
            metrics.inc_ext_ref_release_recovered(0);
            metrics.inc_ext_ref_release_cancelled(0);
            metrics.inc_ext_ref_release_already_enqueued(0);
            assert_eq!(
                sample_value(&render(&metrics), name, &[]),
                Some(add),
                "{name} must no-op on zero so the call sites stay branch-free"
            );
        }
    }

    /// The four reconciler gauges publish together, under a `service` label.
    ///
    /// Published as one call because they are four readings of ONE observation
    /// and are only meaningful against each other — `surplus` is `meta` minus
    /// `referenced` minus what is already owed. A caller able to publish three
    /// of the four would let an operator read a surplus against a stale meta
    /// count from a different pass, which is why the API takes all four.
    #[test]
    fn record_ext_reconcile_gauges_publishes_all_four_under_one_service_label() {
        let metrics = metrics_scrape_only("test-pod");
        metrics.record_ext_reconcile_gauges(12, 9, 3, 1, "replica-a");
        let text = render(&metrics);

        for (name, expected) in [
            ("live_edit_ext_meta_total", 12.0),
            ("live_edit_ext_referenced_hashes_total", 9.0),
            ("live_edit_ext_ref_surplus_total", 3.0),
            ("live_edit_ext_ref_deficit_total", 1.0),
        ] {
            assert_eq!(
                sample_value(&text, name, &[("service", "replica-a")]),
                Some(expected),
                "{name} must carry the publishing replica's `service` label — a demoted leader \
                 zero-publishes under its OWN label, and a fleet-wide max() over a shared label \
                 could not tell the two apart"
            );
        }

        // The non-leader publish: all four to zero, so an OTel gauge that never
        // decays cannot pin the fleet max at a level that no longer exists.
        metrics.record_ext_reconcile_gauges(0, 0, 0, 0, "replica-a");
        let text = render(&metrics);
        for name in [
            "live_edit_ext_meta_total",
            "live_edit_ext_referenced_hashes_total",
            "live_edit_ext_ref_surplus_total",
            "live_edit_ext_ref_deficit_total",
        ] {
            assert_eq!(
                sample_value(&text, name, &[("service", "replica-a")]),
                Some(0.0),
                "{name} must fall back to zero on the non-leader publish"
            );
        }
    }

    /// The retire-retry backlog gauge is service-labelled, like its
    /// leader-published siblings.
    #[test]
    fn record_ext_retire_retry_backlog_publishes_under_the_service_label() {
        let metrics = metrics_scrape_only("test-pod");
        metrics.record_ext_retire_retry_backlog(5, "replica-b");
        assert_eq!(
            sample_value(
                &render(&metrics),
                "live_edit_ext_retire_retry_backlog",
                &[("service", "replica-b")]
            ),
            Some(5.0),
            "the backlog is a ZCARD of `le:ext:{{gc}}:retire-retry` read by the sweep leader; a \
             non-leader publishes 0 under its own label"
        );
    }

    // ── per-replica load gauges: observable, read live state at collect ──

    const REPLICA_LOAD_GAUGES: [&str; 3] = [
        "live_active_sessions",
        "live_active_connections",
        "live_edit_update_layer_admission_entries",
    ];

    #[test]
    fn replica_load_gauges_absent_until_registered() {
        let metrics = metrics_scrape_only("test-pod");
        let text = render(&metrics);
        for name in REPLICA_LOAD_GAUGES {
            assert!(
                !has_metric(&text, name),
                "{name} must be absent before register_replica_load_gauges is called"
            );
        }
    }

    #[test]
    fn replica_load_gauges_reflect_their_sources_at_each_collection() {
        let metrics = metrics_scrape_only("test-pod");
        let sessions = Arc::new(AtomicI64::new(2));
        let connections = Arc::new(AtomicI64::new(5));
        let admission = Arc::new(AtomicI64::new(11));
        {
            let (s, c, a) = (
                Arc::clone(&sessions),
                Arc::clone(&connections),
                Arc::clone(&admission),
            );
            metrics.register_replica_load_gauges(
                "test-pod",
                move || s.load(Ordering::SeqCst),
                move || c.load(Ordering::SeqCst),
                move || a.load(Ordering::SeqCst),
            );
        }
        let svc = [("service", "test-pod")];
        let text = render(&metrics);
        assert_eq!(sample_value(&text, "live_active_sessions", &svc), Some(2.0));
        assert_eq!(
            sample_value(&text, "live_active_connections", &svc),
            Some(5.0)
        );
        assert_eq!(
            sample_value(&text, "live_edit_update_layer_admission_entries", &svc),
            Some(11.0)
        );

        // Load stops: sessions/connections return to zero, the admission map
        // does not (it only shrinks on delete/overwrite). The next collect
        // must show exactly that — proving the callbacks re-read the sources
        // rather than replaying a stale last-set value.
        sessions.store(0, Ordering::SeqCst);
        connections.store(0, Ordering::SeqCst);
        admission.store(12, Ordering::SeqCst);
        let text = render(&metrics);
        assert_eq!(sample_value(&text, "live_active_sessions", &svc), Some(0.0));
        assert_eq!(
            sample_value(&text, "live_active_connections", &svc),
            Some(0.0)
        );
        assert_eq!(
            sample_value(&text, "live_edit_update_layer_admission_entries", &svc),
            Some(12.0)
        );
    }

    // ── structural-planning cache gauges: observable, read the cache at collect ──

    const STRUCTURAL_CACHE_GAUGES: [&str; 2] = [
        "live_edit_structural_cache_entries",
        "live_edit_structural_cache_weight_bytes",
    ];

    #[test]
    fn structural_cache_gauges_absent_until_registered() {
        let metrics = metrics_scrape_only("test-pod");
        let text = render(&metrics);
        for name in STRUCTURAL_CACHE_GAUGES {
            assert!(
                !has_metric(&text, name),
                "{name} must be absent before register_structural_cache_gauges is called"
            );
        }
    }

    #[test]
    fn structural_cache_gauges_reflect_their_sources_at_each_collection() {
        let metrics = metrics_scrape_only("test-pod");
        let entries = Arc::new(AtomicU64::new(3));
        let weight = Arc::new(AtomicU64::new(7 << 20));
        {
            let (e, w) = (Arc::clone(&entries), Arc::clone(&weight));
            metrics.register_structural_cache_gauges(
                "test-pod",
                move || e.load(Ordering::SeqCst),
                move || w.load(Ordering::SeqCst),
            );
        }
        let svc = [("service", "test-pod")];
        let text = render(&metrics);
        for name in STRUCTURAL_CACHE_GAUGES {
            assert_eq!(
                metric_type(&text, name).as_deref(),
                Some("gauge"),
                "{name} renders as a gauge"
            );
        }
        assert_eq!(
            sample_value(&text, "live_edit_structural_cache_entries", &svc),
            Some(3.0)
        );
        assert_eq!(
            sample_value(&text, "live_edit_structural_cache_weight_bytes", &svc),
            Some((7u64 << 20) as f64)
        );

        // The cache shrinks (an eviction, a retirement): the next collect must
        // show the new values, proving the callbacks re-read the cache rather
        // than replaying the last one.
        entries.store(1, Ordering::SeqCst);
        weight.store(4_096, Ordering::SeqCst);
        let text = render(&metrics);
        assert_eq!(
            sample_value(&text, "live_edit_structural_cache_entries", &svc),
            Some(1.0)
        );
        assert_eq!(
            sample_value(&text, "live_edit_structural_cache_weight_bytes", &svc),
            Some(4_096.0)
        );
    }

    // ── service_token_available: unconditional instrument, lazy family ──

    #[test]
    fn service_token_available_absent_until_recorded_then_present() {
        let metrics = metrics_scrape_only("test-pod");
        // Never recorded → the OTel gauge emits no family, so the /metrics
        // output omits it entirely (the pre-migration "only present when a
        // ServiceTokenProvider exists" contract, now without conditional
        // registration).
        let before = render(&metrics);
        assert!(
            !has_metric(&before, "live_edit_service_token_available"),
            "service_token_available must be absent from /metrics until first record()"
        );
        metrics.set_service_token_available(true);
        let after = render(&metrics);
        assert_eq!(
            sample_value(&after, "live_edit_service_token_available", &[]),
            Some(1.0),
            "after set_service_token_available(true) the gauge must render as 1"
        );
        metrics.set_service_token_available(false);
        let after_false = render(&metrics);
        assert_eq!(
            sample_value(&after_false, "live_edit_service_token_available", &[]),
            Some(0.0),
        );
    }

    // ── Lazy signed per-role service-token gauges (b.d6b, b.is1) ──────────────

    /// Assert the three-part contract every `service_token_*_seconds{role}`
    /// gauge shares, given the gauge's exposition `name` and the `Metrics`
    /// method that records it.
    ///
    /// One helper rather than a test apiece per gauge because the contract is
    /// structural, not per-metric: these gauges are all lazily-registered,
    /// role-labelled, and signed, and the tests for them were otherwise
    /// identical but for the name and the recorder — the
    /// copy-paste-with-one-line-differences shape the test-review guide flags,
    /// and the shape in which the next gauge of this family arrives with a
    /// fourth copy and one of the three properties quietly dropped.
    ///
    /// The three properties, in the order they are checked:
    ///
    /// 1. **Absent until first `record()`.** A replica that constructs no
    ///    `ServiceTokenProvider` exposes no series at all, so an absent series
    ///    is readable as "nothing has reported here" and never as a recorded
    ///    zero. (What the *absence* means differs per gauge — see the call
    ///    sites — but that it must hold does not.)
    /// 2. **Split by role.** The production instances (`valkey`,
    ///    `storage-api`) differ in severity, so a second record() under a
    ///    different role must add a series rather than overwrite one.
    /// 3. **Signed, unclamped.** The zero crossing is the operator-meaningful
    ///    part of each of these series, and a re-record must not disturb the
    ///    sibling role.
    fn assert_lazy_signed_role_gauge(name: &str, record: impl Fn(&Metrics, &str, i64)) {
        let metrics = metrics_scrape_only("test-pod");
        assert!(
            !has_metric(&render(&metrics), name),
            "{name} must be absent from /metrics until its first record(); an absent series is \
             how an operator reads 'nothing has reported here', which a pre-initialised zero \
             would destroy"
        );

        record(&metrics, "valkey", 900);
        record(&metrics, "storage-api", 30);
        let after = render(&metrics);
        assert_eq!(
            sample_value(&after, name, &[("role", "valkey")]),
            Some(900.0),
            "{name} must render the value recorded under role=valkey"
        );
        assert_eq!(
            sample_value(&after, name, &[("role", "storage-api")]),
            Some(30.0),
            "{name}: each role must carry its own series; a second record() must not overwrite \
             the first"
        );

        record(&metrics, "valkey", -12);
        let after_negative = render(&metrics);
        assert_eq!(
            sample_value(&after_negative, name, &[("role", "valkey")]),
            Some(-12.0),
            "{name} must render a negative value unmodified — the zero crossing is the \
             operator-meaningful part of the series, and clamping erases it"
        );
        assert_eq!(
            sample_value(&after_negative, name, &[("role", "storage-api")]),
            Some(30.0),
            "{name}: re-recording one role must leave the other role's series untouched"
        );
    }

    #[test]
    fn service_token_expires_in_seconds_is_a_lazy_signed_per_role_gauge() {
        // b.d6b. Absence here means "no provider is wired on this replica";
        // negative means that role's credential has already died, and how long
        // ago — expiry on `valkey` is a crash-looping pod, on `storage-api` a
        // degraded checkpoint path.
        assert_lazy_signed_role_gauge(
            "live_edit_service_token_expires_in_seconds",
            |metrics, role, secs| metrics.set_service_token_expires_in_seconds(role, secs),
        );
    }

    #[test]
    fn service_token_refresh_headroom_seconds_is_a_lazy_signed_per_role_gauge() {
        // b.is1. Same structural contract, stronger reading of each part: this
        // series is written once per observed *rotation*, so absence means "no
        // rotation seen yet" (a provider that has cached at most one token has
        // by definition observed none), and negative means the replacement
        // landed after the old token had already expired — every pooled
        // connection dropped and re-established rather than prolonged.
        assert_lazy_signed_role_gauge(
            "live_edit_service_token_refresh_headroom_seconds",
            |metrics, role, secs| metrics.set_service_token_refresh_headroom_seconds(role, secs),
        );
    }

    // ── Observable gauge (watch/unwatch inflight) reflects the closure ──

    #[test]
    fn storage_watch_unwatch_inflight_observable_gauge_reflects_closure_at_collect() {
        let metrics = metrics_scrape_only("test-pod");
        let name = "live_edit_storage_watch_unwatch_inflight";
        // Absent until the observable gauge is registered.
        assert!(
            !has_metric(&render(&metrics), name),
            "observable gauge must be absent before registration"
        );
        let source = Arc::new(AtomicI64::new(3));
        let handle = Arc::clone(&source);
        metrics
            .register_storage_watch_unwatch_inflight_gauge(move || handle.load(Ordering::SeqCst));
        // First collect must observe the current closure value (3).
        assert_eq!(
            sample_value(&render(&metrics), name, &[]),
            Some(3.0),
            "observable gauge must reflect the closure value at collection time"
        );
        // Mutate the source; the next collect must pick up the new value,
        // proving the callback runs on every scrape (not a stale last-set).
        source.store(7, Ordering::SeqCst);
        assert_eq!(
            sample_value(&render(&metrics), name, &[]),
            Some(7.0),
            "observable gauge must re-read the closure on each collection"
        );
    }

    // ── Process and allocator memory gauges (b.u5k) ─────────────────────

    /// The sample lines of exactly the family `name` — not its `#` metadata,
    /// and not a longer name that merely starts with `name`.
    fn series_lines<'a>(text: &'a str, name: &str) -> Vec<&'a str> {
        text.lines()
            .map(str::trim)
            .filter(|l| {
                l.strip_prefix(name)
                    .is_some_and(|rest| rest.starts_with(' ') || rest.starts_with('{'))
            })
            .collect()
    }

    /// A gauge source backed by a shared cell the test mutates between
    /// collections. `u64::MAX` stands for "the source has no reading" (`None`).
    fn cell_source(cell: &Arc<AtomicU64>) -> impl Fn() -> Option<u64> + Send + Sync + 'static {
        let cell = Arc::clone(cell);
        move || Some(cell.load(Ordering::SeqCst)).filter(|v| *v != u64::MAX)
    }

    const PROCESS_RSS: &str = "process_resident_memory_bytes";
    const JEMALLOC_ALLOCATED: &str = "live_edit_jemalloc_allocated_bytes";
    const JEMALLOC_RESIDENT: &str = "live_edit_jemalloc_resident_bytes";

    #[test]
    fn process_resident_memory_gauge_is_an_unlabelled_gauge_read_at_each_collection() {
        let metrics = metrics_scrape_only("test-pod");
        assert!(
            !has_metric(&render(&metrics), PROCESS_RSS),
            "the family must be absent until registered"
        );
        let rss = Arc::new(AtomicU64::new(123_456_789));
        metrics.register_process_memory_gauge(cell_source(&rss));

        let text = render(&metrics);
        assert_eq!(metric_type(&text, PROCESS_RSS).as_deref(), Some("gauge"));
        let series = series_lines(&text, PROCESS_RSS);
        assert_eq!(
            series.len(),
            1,
            "exactly one series expected; got {series:?}"
        );
        assert!(
            series[0].starts_with(&format!("{PROCESS_RSS} ")),
            "`{PROCESS_RSS}` must carry no labels, like the Prometheus standard process metric \
             whose name it reuses, so dashboards written against that name match it; got {:?}",
            series[0]
        );
        assert_eq!(sample_value(&text, PROCESS_RSS, &[]), Some(123_456_789.0));

        rss.store(42 << 20, Ordering::SeqCst);
        assert_eq!(
            sample_value(&render(&metrics), PROCESS_RSS, &[]),
            Some((42u64 << 20) as f64),
            "the gauge must re-read its source at every collection"
        );
    }

    #[test]
    fn process_resident_memory_gauge_with_no_reading_emits_no_sample() {
        let metrics = metrics_scrape_only("test-pod");
        metrics.register_process_memory_gauge(|| None);

        let text = render(&metrics);
        let series = series_lines(&text, PROCESS_RSS);
        assert!(
            series.is_empty(),
            "a source with no reading (off Linux, or /proc unreadable) must emit no sample \
             rather than a misleading 0; got {series:?}"
        );
    }

    #[test]
    fn process_resident_memory_gauge_losing_its_reading_stops_emitting_the_last_value() {
        let metrics = metrics_scrape_only("test-pod");
        let rss = Arc::new(AtomicU64::new(64 << 20));
        metrics.register_process_memory_gauge(cell_source(&rss));
        assert_eq!(
            sample_value(&render(&metrics), PROCESS_RSS, &[]),
            Some((64u64 << 20) as f64)
        );

        rss.store(u64::MAX, Ordering::SeqCst);
        let text = render(&metrics);
        let series = series_lines(&text, PROCESS_RSS);
        assert!(
            series.is_empty(),
            "a collection whose source has no reading must emit no sample, not repeat the \
             previous one: a stale resident size would read as memory that never moves; got \
             {series:?}"
        );
    }

    const STORE_MEMORY_USED_RATIO: &str = "live_edit_store_memory_used_ratio";

    /// T7 (b.qhk): `live_edit_store_memory_used_ratio` is absent until a
    /// store-memory read records a figure; then one unlabelled gauge sample
    /// carrying it, unclamped above 1; absent again from the collection after
    /// a read records none — never a 0 and never the stale figure — and
    /// present again at the next figure.
    #[test]
    fn store_memory_used_ratio_is_absent_until_recorded_and_again_when_a_read_gives_none() {
        let metrics = metrics_scrape_only("test-pod");
        let series = |metrics: &Metrics| {
            series_lines(&render(metrics), STORE_MEMORY_USED_RATIO)
                .into_iter()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            series(&metrics),
            Vec::<String>::new(),
            "absent before any read"
        );

        metrics.record_store_memory_used_ratio(Some(0.25));
        let text = render(&metrics);
        assert_eq!(
            metric_type(&text, STORE_MEMORY_USED_RATIO).as_deref(),
            Some("gauge")
        );
        let present = series(&metrics);
        assert_eq!(present.len(), 1, "one series; got {present:?}");
        assert!(
            present[0].starts_with(&format!("{STORE_MEMORY_USED_RATIO} ")),
            "no labels; got {:?}",
            present[0]
        );
        assert_eq!(
            sample_value(&text, STORE_MEMORY_USED_RATIO, &[]),
            Some(0.25)
        );

        metrics.record_store_memory_used_ratio(Some(1.5));
        assert_eq!(
            sample_value(&render(&metrics), STORE_MEMORY_USED_RATIO, &[]),
            Some(1.5),
            "past the store's limit, unclamped"
        );

        metrics.record_store_memory_used_ratio(None);
        assert_eq!(
            series(&metrics),
            Vec::<String>::new(),
            "a read with no figure drops the series, not repeat 1.5 or report 0"
        );

        metrics.record_store_memory_used_ratio(Some(0.5));
        assert_eq!(
            sample_value(&render(&metrics), STORE_MEMORY_USED_RATIO, &[]),
            Some(0.5),
            "the next figure brings it back"
        );
    }

    /// T7 (b.qhk): the help text names the two arms, the copies rule with its
    /// second-read absence on a store that reports no `maxmemory`, and when
    /// the series is absent.
    #[test]
    fn store_memory_used_ratio_help_names_its_arms_and_its_absence() {
        let metrics = metrics_scrape_only("test-pod");
        metrics.record_store_memory_used_ratio(Some(0.1));
        let text = render(&metrics);
        let help = text
            .lines()
            .find_map(|l| l.strip_prefix(&format!("# HELP {STORE_MEMORY_USED_RATIO} ")))
            .unwrap_or_else(|| panic!("a HELP line for {STORE_MEMORY_USED_RATIO}; got {text}"));
        for phrase in [
            "the highest used_memory / maxmemory across primaries",
            "LIVE_EDIT_VALKEY_MEMORY_BYTES x 0.8",
            "cmdstat_replconf",
            "appears at the second read",
            "above 1 when the store is past its limit",
            "Absent when",
        ] {
            assert!(help.contains(phrase), "{phrase:?} in {help:?}");
        }
    }

    #[test]
    fn allocator_memory_gauges_are_service_labelled_gauges_read_at_each_collection() {
        let metrics = metrics_scrape_only("test-pod");
        let text = render(&metrics);
        for name in [JEMALLOC_ALLOCATED, JEMALLOC_RESIDENT] {
            assert!(
                !has_metric(&text, name),
                "{name} must be absent until registered"
            );
        }
        let allocated = Arc::new(AtomicU64::new(10 << 20));
        let resident = Arc::new(AtomicU64::new(30 << 20));
        metrics.register_allocator_memory_gauges(
            "replica-a",
            cell_source(&allocated),
            cell_source(&resident),
        );

        let text = render(&metrics);
        for (name, expected) in [
            (JEMALLOC_ALLOCATED, 10u64 << 20),
            (JEMALLOC_RESIDENT, 30u64 << 20),
        ] {
            assert_eq!(metric_type(&text, name).as_deref(), Some("gauge"), "{name}");
            let series = series_lines(&text, name);
            assert_eq!(
                series,
                vec![format!("{name}{{service=\"replica-a\"}} {expected}")],
                "{name} must be one series labelled only `service`, carrying its own source's \
                 value (a swapped pair would read allocated as resident)"
            );
        }

        allocated.store(11 << 20, Ordering::SeqCst);
        resident.store(12 << 20, Ordering::SeqCst);
        let text = render(&metrics);
        let svc = [("service", "replica-a")];
        assert_eq!(
            sample_value(&text, JEMALLOC_ALLOCATED, &svc),
            Some((11u64 << 20) as f64),
            "the gauges must re-read their sources at every collection"
        );
        assert_eq!(
            sample_value(&text, JEMALLOC_RESIDENT, &svc),
            Some((12u64 << 20) as f64)
        );
    }

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    /// Each memory family has its own row in the customer metrics reference,
    /// the only place an operator learns how to read them: memory returned to
    /// the OS is judged from `process_resident_memory_bytes`, while
    /// `stats.resident` is only an upper bound beside `stats.allocated`; and
    /// the pod memory plan's gauges (the limit every budget is derived from,
    /// by source, and the budgets read against it) are how an operator sees
    /// what a replica sized itself for.
    #[test]
    fn every_memory_gauge_family_appears_in_the_metrics_guide() {
        for name in [
            PROCESS_RSS,
            JEMALLOC_ALLOCATED,
            JEMALLOC_RESIDENT,
            POD_MEMORY_LIMIT,
            CATCHUP_BUILD_BUDGET,
            STRUCTURAL_CACHE_BUDGET,
            SUBSCRIBER_MAX_SUBSCRIPTIONS,
            EXT_MIRROR_BUDGET,
            EXT_MIRROR_BYTES,
            JEMALLOC_DIRTY,
            RETIREMENT_BUDGET,
            RETIREMENT_RESERVED,
            RETIREMENT_OVERCOMMIT,
            RETIREMENT_WAIT,
        ] {
            metrics_reference_row(name);
        }
    }

    /// b.fsz MR-D4: the label values operators select on for the families
    /// MR-D4 added or widened appear in their rows of the metrics reference:
    /// the dirty-pages gauge's `arenas`, the retirement wait's `outcome`, and
    /// every `key_family` of `live_edit_snapshot_read_type_error_total`
    /// (`ext_index` among them).
    #[test]
    fn every_retirement_dirty_and_key_family_label_value_appears_in_the_metrics_guide() {
        assert_reference_row_names(
            JEMALLOC_DIRTY,
            "arenas",
            &[
                Metrics::JEMALLOC_DIRTY_ARENAS_AUTOMATIC,
                Metrics::JEMALLOC_DIRTY_ARENAS_FANOUT,
            ],
        );
        assert_reference_row_names(RETIREMENT_WAIT, "outcome", RETIREMENT_WAIT_OUTCOMES);
        assert_reference_row_names(
            "live_edit_snapshot_read_type_error_total",
            "key_family",
            SNAPSHOT_READ_KEY_FAMILIES,
        );
    }

    // ── The pod memory plan's gauges (b.fj4) ──────────────────────────────

    const POD_MEMORY_LIMIT: &str = "live_edit_pod_memory_limit_bytes";
    const CATCHUP_BUILD_BUDGET: &str = "live_edit_catchup_build_budget_bytes";
    const STRUCTURAL_CACHE_BUDGET: &str = "live_edit_structural_cache_budget_bytes";
    const SUBSCRIBER_MAX_SUBSCRIPTIONS: &str = "live_edit_subscriber_max_subscriptions";
    /// The external-value mirror's byte budget, `M_ext` (b.mhq).
    const EXT_MIRROR_BUDGET: &str = "live_edit_checkpoint_ext_mirror_budget_bytes";
    /// The bytes charged to it, by `state` (b.mhq).
    const EXT_MIRROR_BYTES: &str = "live_edit_checkpoint_ext_mirror_bytes";
    /// The copies the mirror dropped, by `reason` (b.mhq).
    const EXT_MIRROR_DROPPED: &str = "live_edit_checkpoint_ext_mirror_dropped_total";

    /// The external-value mirror's series exist from the first scrape,
    /// before any value is mirrored and whether or not a mirror is wired:
    /// every `reason` of the dropped counter and every `state` of the byte
    /// gauge at 0, so an `increase()` alert on `budget_full` sees its first
    /// tick and a dashboard reads "nothing held" rather than "no data"; and
    /// the budget gauge, unlabelled, at 0 until `record_pod_memory_plan`
    /// sets it. The vocabularies are pinned: the metrics reference names
    /// these values.
    #[test]
    fn ext_mirror_series_are_zero_seeded_and_their_vocabularies_pinned() {
        let metrics = metrics_scrape_only("zero-seed-test");
        let text = render(&metrics);
        assert_eq!(
            EXT_MIRROR_DROP_REASONS,
            ["queue_full", "budget_full", "oversized", "shutdown"]
        );
        assert_eq!(EXT_MIRROR_STATES, ["queued", "in_flight"]);
        for reason in EXT_MIRROR_DROP_REASONS {
            assert_eq!(
                sample_value(&text, EXT_MIRROR_DROPPED, &[("reason", reason)]),
                Some(0.0),
                "reason `{reason}` is zero-seeded"
            );
        }
        for state in EXT_MIRROR_STATES {
            assert_eq!(
                sample_value(&text, EXT_MIRROR_BYTES, &[("state", state)]),
                Some(0.0),
                "state `{state}` is zero-seeded"
            );
        }
        assert_eq!(
            metric_type(&text, EXT_MIRROR_DROPPED).as_deref(),
            Some("counter")
        );
        assert_eq!(
            metric_type(&text, EXT_MIRROR_BYTES).as_deref(),
            Some("gauge")
        );
        assert_eq!(
            series_lines(&text, EXT_MIRROR_BUDGET),
            vec![format!("{EXT_MIRROR_BUDGET} 0")],
            "the budget is one unlabelled series at 0 before the plan is recorded"
        );
    }

    /// The backup repair's outcomes (b.wsx).
    const EXT_BACKUP_REPAIR: &str = "live_edit_checkpoint_ext_backup_repair_total";
    /// Its failed confirmation writes, by `op` (b.wsx).
    const EXT_BACKUP_MARK_FAILED: &str = "live_edit_checkpoint_ext_backup_mark_failed_total";
    /// Its owed gauge (b.wsx).
    const EXT_BACKUP_OWED: &str = "live_edit_checkpoint_ext_backup_owed";

    /// The backup repair's counters exist from the first scrape, every
    /// `outcome` and every `op` at 0, so an `increase()` alert sees the first
    /// tick; the vocabularies are pinned (eight outcomes, two ops: the clean-up
    /// fence and its outcome are not part of this build). The owed gauge has
    /// no series until a pass or a non-leader publishes one: unknown is not 0.
    /// Publishing replaces the reading per `service`, and the outcome adder
    /// ignores a 0.
    #[test]
    fn ext_backup_repair_series_are_zero_seeded_and_their_vocabularies_pinned() {
        let metrics = metrics_scrape_only("zero-seed-test");
        let text = render(&metrics);
        assert_eq!(
            EXT_BACKUP_REPAIR_OUTCOMES,
            [
                "already_stored",
                "resent",
                "resend_failed",
                "probe_failed",
                "value_gone",
                "budget_busy",
                "oversized",
                "confirmed_copy_missing",
            ]
        );
        assert_eq!(EXT_BACKUP_MARK_OPS, ["confirm", "clear"]);
        for outcome in EXT_BACKUP_REPAIR_OUTCOMES {
            assert_eq!(
                sample_value(&text, EXT_BACKUP_REPAIR, &[("outcome", outcome)]),
                Some(0.0),
                "outcome `{outcome}` is zero-seeded"
            );
        }
        for op in EXT_BACKUP_MARK_OPS {
            assert_eq!(
                sample_value(&text, EXT_BACKUP_MARK_FAILED, &[("op", op)]),
                Some(0.0),
                "op `{op}` is zero-seeded"
            );
        }
        assert_eq!(
            metric_type(&text, EXT_BACKUP_REPAIR).as_deref(),
            Some("counter")
        );
        assert_eq!(
            metric_type(&text, EXT_BACKUP_MARK_FAILED).as_deref(),
            Some("counter")
        );
        assert!(
            series_lines(&text, EXT_BACKUP_OWED).is_empty(),
            "the owed gauge has no reading before one is published: {text}"
        );

        metrics.inc_checkpoint_ext_backup_repair(EXT_BACKUP_REPAIR_RESENT, 0);
        metrics.inc_checkpoint_ext_backup_repair(EXT_BACKUP_REPAIR_RESENT, 2);
        metrics.inc_checkpoint_ext_backup_mark_failed(EXT_BACKUP_MARK_OP_CLEAR);
        metrics.record_checkpoint_ext_backup_owed(7, "replica-a");
        metrics.record_checkpoint_ext_backup_owed(3, "replica-a");
        metrics.record_checkpoint_ext_backup_owed(0, "replica-b");
        let text = render(&metrics);
        assert_eq!(
            sample_value(&text, EXT_BACKUP_REPAIR, &[("outcome", "resent")]),
            Some(2.0)
        );
        assert_eq!(
            sample_value(&text, EXT_BACKUP_MARK_FAILED, &[("op", "clear")]),
            Some(1.0)
        );
        assert_eq!(
            sample_value(&text, EXT_BACKUP_MARK_FAILED, &[("op", "confirm")]),
            Some(0.0)
        );
        assert_eq!(
            metric_type(&text, EXT_BACKUP_OWED).as_deref(),
            Some("gauge")
        );
        assert_eq!(
            sample_value(&text, EXT_BACKUP_OWED, &[("service", "replica-a")]),
            Some(3.0),
            "the last reading replaces the earlier one"
        );
        assert_eq!(
            sample_value(&text, EXT_BACKUP_OWED, &[("service", "replica-b")]),
            Some(0.0)
        );
    }

    /// `record_pod_memory_plan` creates every `source` series of the limit
    /// gauge, the source in use carrying the limit and the other three 0, so a
    /// dashboard can select on the label from the first scrape whichever
    /// source a replica read; and the budget gauges carry their values, each
    /// its own (a swapped pair would read the cache's budget as the catch-up
    /// budget). Each is a gauge; the limit is labelled by service and source,
    /// the structural cache's budget by service, and the catch-up budget, the
    /// retirement allowance and the subscription ceiling not at all, like the
    /// usage gauges they are read against. The external-value mirror's budget is 0 for a
    /// plan that funds no mirror (no backup store) and `M_ext` for one that
    /// does, one unlabelled series either way: the sum over `state` of the
    /// bytes it is read against carries no other label.
    #[test]
    fn record_pod_memory_plan_creates_every_source_series_and_each_budget_gauge() {
        use crate::config::{PlanInputs, PodMemoryLimit, PodMemoryLimitSource, PodMemoryPlan};
        for in_use in PodMemoryLimitSource::ALL {
            let metrics = metrics_scrape_only("test-pod");
            // The chart's configuration at 8 GiB: T 1,607 MiB, C 409 MiB
            // (resident 518,929,777 B), S 6,553.
            let plan = PodMemoryPlan::derive(
                PodMemoryLimit {
                    bytes: 8 << 30,
                    source: in_use,
                    reason: None,
                },
                &PlanInputs::reference(),
            )
            .expect("the chart's configuration closes at 8 GiB");
            metrics.record_pod_memory_plan(&plan);
            let text = render(&metrics);
            for name in [
                POD_MEMORY_LIMIT,
                CATCHUP_BUILD_BUDGET,
                STRUCTURAL_CACHE_BUDGET,
                SUBSCRIBER_MAX_SUBSCRIPTIONS,
            ] {
                assert_eq!(metric_type(&text, name).as_deref(), Some("gauge"), "{name}");
            }
            assert_eq!(
                series_lines(&text, POD_MEMORY_LIMIT).len(),
                PodMemoryLimitSource::ALL.len(),
                "one series per source: {text}"
            );
            for source in PodMemoryLimitSource::ALL {
                let expected = if source == in_use {
                    (8u64 << 30) as f64
                } else {
                    0.0
                };
                assert_eq!(
                    sample_value(
                        &text,
                        POD_MEMORY_LIMIT,
                        &[("service", "test-pod"), ("source", source.label())]
                    ),
                    Some(expected),
                    "source={} with {} in use",
                    source.label(),
                    in_use.label()
                );
            }
            // The exact series, label sets included: `sample_value` matches
            // a subset of the labels, so it cannot tell an unlabelled gauge
            // from one that carries `service`.
            let mut limit_series = series_lines(&text, POD_MEMORY_LIMIT);
            limit_series.sort_unstable();
            let mut expected_limit_series: Vec<String> = PodMemoryLimitSource::ALL
                .iter()
                .map(|source| {
                    let value = if *source == in_use { 8u64 << 30 } else { 0 };
                    format!(
                        "{POD_MEMORY_LIMIT}{{service=\"test-pod\",source=\"{}\"}} {value}",
                        source.label()
                    )
                })
                .collect();
            expected_limit_series.sort_unstable();
            assert_eq!(limit_series, expected_limit_series, "{text}");
            assert_eq!(
                series_lines(&text, CATCHUP_BUILD_BUDGET),
                vec![format!("{CATCHUP_BUILD_BUDGET} {}", 1_607u64 << 20)],
                "the catch-up budget is unlabelled, the plan's T"
            );
            assert_eq!(
                series_lines(&text, STRUCTURAL_CACHE_BUDGET),
                vec![format!(
                    "{STRUCTURAL_CACHE_BUDGET}{{service=\"test-pod\"}} {}",
                    plan.structural_cache_resident_budget_bytes()
                )],
                "the structural cache's budget carries only `service`, at C's resident budget"
            );
            assert_eq!(
                plan.structural_cache_resident_budget_bytes(),
                518_929_777,
                "C × (1 + φ) at 8 GiB, not C"
            );
            assert_eq!(
                series_lines(&text, SUBSCRIBER_MAX_SUBSCRIPTIONS),
                vec![format!("{SUBSCRIBER_MAX_SUBSCRIPTIONS} 6553")],
                "the subscription ceiling is unlabelled, the plan's S"
            );
            assert_eq!(
                metric_type(&text, RETIREMENT_BUDGET).as_deref(),
                Some("gauge")
            );
            assert_eq!(
                series_lines(&text, RETIREMENT_BUDGET),
                vec![format!(
                    "{RETIREMENT_BUDGET} {}",
                    plan.retirement_budget_bytes()
                )],
                "the retirement allowance is unlabelled, like the reserved gauge it is read \
                 against, at the plan's R_ref (b.fsz MR-D4)"
            );
            assert_eq!(
                plan.retirement_budget_bytes(),
                16 << 20,
                "R_ref = L / 500 at 8 GiB, whole MiB"
            );
            assert_eq!(
                series_lines(&text, EXT_MIRROR_BUDGET),
                vec![format!("{EXT_MIRROR_BUDGET} 0")],
                "the chart's configuration has no backup store: the plan funds no mirror"
            );
        }

        // With a backup store the plan funds the mirror: its budget is
        // M_ext, 204 MiB at 8 GiB, one unlabelled gauge series.
        let metrics = metrics_scrape_only("test-pod");
        let plan = PodMemoryPlan::derive(
            PodMemoryLimit {
                bytes: 8 << 30,
                source: PodMemoryLimitSource::CgroupV2,
                reason: None,
            },
            &PlanInputs {
                base_bootstrap: true,
                ext_mirror: true,
                ..PlanInputs::reference()
            },
        )
        .expect("the chart's configuration with a backup store closes at 8 GiB");
        assert_eq!(plan.ext_mirror_bytes(), Some(204 << 20));
        metrics.record_pod_memory_plan(&plan);
        let text = render(&metrics);
        assert_eq!(
            metric_type(&text, EXT_MIRROR_BUDGET).as_deref(),
            Some("gauge")
        );
        assert_eq!(
            series_lines(&text, EXT_MIRROR_BUDGET),
            vec![format!("{EXT_MIRROR_BUDGET} {}", 204u64 << 20)],
            "the mirror's budget is unlabelled, the plan's M_ext"
        );
    }

    // ── Trace-export counters ────────────────────────────────────────────

    const TRACE_SPANS_EXPORTED: &str = "live_edit_trace_spans_exported_total";
    const TRACE_SPANS_DROPPED: &str = "live_edit_trace_spans_dropped_total";

    /// Both families are counters, and both `reason` cells of the dropped
    /// family are reported at zero before any span is lost, so a rate over
    /// either is defined from the first scrape.
    #[test]
    fn trace_export_counters_render_as_counters_with_both_reasons_at_zero() {
        let metrics = metrics_scrape_only("test-pod");
        let state = Arc::new(crate::observability::trace_export::TraceExportState::new(4));
        metrics.register_trace_export_counters(Arc::clone(&state));

        let text = render(&metrics);
        for name in [TRACE_SPANS_EXPORTED, TRACE_SPANS_DROPPED] {
            assert_eq!(
                metric_type(&text, name).as_deref(),
                Some("counter"),
                "{name} must be a counter"
            );
        }
        assert_eq!(sample_value(&text, TRACE_SPANS_EXPORTED, &[]), Some(0.0));
        for reason in [
            Metrics::TRACE_SPANS_DROPPED_REASON_QUEUE_FULL,
            Metrics::TRACE_SPANS_DROPPED_REASON_EXPORT_FAILED,
        ] {
            assert_eq!(
                sample_value(&text, TRACE_SPANS_DROPPED, &[("reason", reason)]),
                Some(0.0),
                "reason={reason} must be reported at 0 before any loss"
            );
        }
    }

    /// Each series reads its own counter at every collection: distinct
    /// amounts on the three counters must land on the three matching series,
    /// so a swapped pair of `reason` cells cannot pass.
    #[test]
    fn trace_export_counters_track_their_source_at_each_collection() {
        let metrics = metrics_scrape_only("test-pod");
        let state = Arc::new(crate::observability::trace_export::TraceExportState::new(4));
        metrics.register_trace_export_counters(Arc::clone(&state));

        crate::observability::trace_export::test_support::advance_counters(&state, 7, 3, 5);
        let text = render(&metrics);
        assert_eq!(sample_value(&text, TRACE_SPANS_EXPORTED, &[]), Some(7.0));
        assert_eq!(
            sample_value(&text, TRACE_SPANS_DROPPED, &[("reason", "queue_full")]),
            Some(3.0)
        );
        assert_eq!(
            sample_value(&text, TRACE_SPANS_DROPPED, &[("reason", "export_failed")]),
            Some(5.0)
        );

        crate::observability::trace_export::test_support::advance_counters(&state, 1, 1, 1);
        let text = render(&metrics);
        assert_eq!(
            sample_value(&text, TRACE_SPANS_EXPORTED, &[]),
            Some(8.0),
            "the counters must be re-read at every collection"
        );
        assert_eq!(
            sample_value(&text, TRACE_SPANS_DROPPED, &[("reason", "queue_full")]),
            Some(4.0)
        );
        assert_eq!(
            sample_value(&text, TRACE_SPANS_DROPPED, &[("reason", "export_failed")]),
            Some(6.0)
        );
    }

    /// Neither family is reported unless registered: their presence on
    /// `/metrics` is the signal that trace export is enabled.
    #[test]
    fn trace_export_counters_are_absent_unless_registered() {
        let metrics = metrics_scrape_only("test-pod");
        let text = render(&metrics);
        for name in [TRACE_SPANS_EXPORTED, TRACE_SPANS_DROPPED] {
            assert!(
                !has_metric(&text, name),
                "{name} must be absent while trace export is not enabled"
            );
        }
    }

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    /// The customer metrics reference documents both trace-export families
    /// and both `reason` values, in the `##` section that holds the dropped
    /// family's row, so the reason values only count where an operator
    /// reading about trace-span drops will find them.
    #[test]
    fn trace_export_counters_and_reasons_appear_in_their_metrics_guide_section() {
        let row = metrics_reference_row(TRACE_SPANS_DROPPED);
        let at = METRICS_REFERENCE
            .find(row)
            .expect("the row was found in this same text");
        let start = METRICS_REFERENCE[..at].rfind("\n## ").map_or(0, |i| i + 1);
        let end = METRICS_REFERENCE[at..]
            .find("\n## ")
            .map_or(METRICS_REFERENCE.len(), |i| at + i);
        let section = &METRICS_REFERENCE[start..end];

        for needle in [
            format!("| `{TRACE_SPANS_EXPORTED}` |"),
            format!("`{}`", Metrics::TRACE_SPANS_DROPPED_REASON_QUEUE_FULL),
            format!("`{}`", Metrics::TRACE_SPANS_DROPPED_REASON_EXPORT_FAILED),
        ] {
            assert!(
                section.contains(&needle),
                "the metrics-reference section documenting `{TRACE_SPANS_DROPPED}` must name \
                 {needle}; section:\n{section}"
            );
        }
    }

    const JEMALLOC_DIRTY: &str = "live_edit_jemalloc_dirty_bytes";

    /// b.fsz MR-D4 (D4-T10): the dirty-pages gauge is absent until
    /// registered; then one gauge series per arena family the source reads,
    /// labelled `service` and `arenas` (`automatic` and `fanout`), each its
    /// own value, re-read at every collection; the `fanout` series absent
    /// while the source reports no fan-out arena, and no series at all while
    /// it has no reading. The source itself (jemalloc's `pdirty`) is the bin
    /// target's, `allocator::tests`.
    #[test]
    fn allocator_dirty_gauge_reports_each_arena_family_under_service_and_arenas() {
        let metrics = metrics_scrape_only("test-pod");
        assert!(
            !has_metric(&render(&metrics), JEMALLOC_DIRTY),
            "the family must be absent until registered"
        );
        /// What the source reads: `(automatic, fanout)`, `None` on a failed
        /// read.
        type DirtyReading = Option<(u64, Option<u64>)>;
        let reading: Arc<std::sync::Mutex<DirtyReading>> =
            Arc::new(std::sync::Mutex::new(Some((5 << 20, Some(3 << 20)))));
        let source = Arc::clone(&reading);
        metrics.register_allocator_dirty_gauge("replica-a", move || *source.lock().unwrap());

        let text = render(&metrics);
        assert_eq!(metric_type(&text, JEMALLOC_DIRTY).as_deref(), Some("gauge"));
        let dirty = |text: &str, arenas: &str| {
            sample_value(
                text,
                JEMALLOC_DIRTY,
                &[("service", "replica-a"), ("arenas", arenas)],
            )
        };
        let series = series_lines(&text, JEMALLOC_DIRTY);
        assert_eq!(series.len(), 2, "one series per arena family: {series:?}");
        assert!(
            series.iter().all(|line| line.matches('=').count() == 2),
            "each series carries exactly `service` and `arenas`: {series:?}"
        );
        assert_eq!(
            (dirty(&text, "automatic"), dirty(&text, "fanout")),
            (Some((5u64 << 20) as f64), Some((3u64 << 20) as f64)),
            "each arena family its own value (a swapped pair would read the fan-out arena's \
             slack as everyone else's)"
        );

        *reading.lock().unwrap() = Some((7 << 20, None));
        let text = render(&metrics);
        assert_eq!(
            series_lines(&text, JEMALLOC_DIRTY).len(),
            1,
            "no fan-out arena, no `fanout` series"
        );
        assert_eq!(
            dirty(&text, "automatic"),
            Some((7u64 << 20) as f64),
            "re-read at each collection"
        );

        *reading.lock().unwrap() = None;
        assert!(
            series_lines(&render(&metrics), JEMALLOC_DIRTY).is_empty(),
            "a failed read emits no sample rather than a misleading 0"
        );
    }

    const RETIREMENT_BUDGET: &str = "live_edit_retirement_budget_bytes";
    const RETIREMENT_RESERVED: &str = "live_edit_retirement_budget_reserved_bytes";
    const RETIREMENT_OVERCOMMIT: &str = "live_edit_retirement_budget_overcommit_bytes";
    const RETIREMENT_WAIT: &str = "live_edit_retirement_wait_seconds";

    /// b.fsz MR-D4: the retirement allowance's usage gauges are unlabelled
    /// and present at 0 from the first scrape, before any retirement
    /// reserves; its wait histogram's `outcome` vocabulary is pinned (the
    /// metrics reference names it) and it shares the admission waits' bucket
    /// boundaries; an observation lands under its own outcome only.
    #[test]
    fn retirement_allowance_series_are_zero_seeded_and_its_wait_outcomes_pinned() {
        let metrics = metrics_scrape_only("zero-seed-test");
        let text = render(&metrics);
        for gauge in [RETIREMENT_RESERVED, RETIREMENT_OVERCOMMIT] {
            assert_eq!(
                series_lines(&text, gauge),
                vec![format!("{gauge} 0")],
                "{gauge}: one unlabelled series at 0, like the catch-up budget's"
            );
        }
        assert_eq!(
            RETIREMENT_WAIT_OUTCOMES,
            ["immediate", "waited", "cancelled"]
        );
        assert!(
            !has_metric(&text, RETIREMENT_WAIT),
            "a histogram series appears with its first observation"
        );

        metrics.observe_retirement_wait(Duration::from_millis(30), RETIREMENT_WAIT_OUTCOME_WAITED);
        metrics.observe_catchup_build_wait(Duration::from_millis(30));
        let text = render(&metrics);
        let count = |outcome: &str| {
            sample_value(
                &text,
                &format!("{RETIREMENT_WAIT}_count"),
                &[("outcome", outcome)],
            )
        };
        assert_eq!(
            (count("waited"), count("immediate"), count("cancelled")),
            (Some(1.0), None, None),
            "the observation lands under its own outcome"
        );
        let boundaries = |family: &str| -> Vec<String> {
            text.lines()
                .filter(|l| l.starts_with(&format!("{family}_bucket{{")))
                .filter_map(|l| l.split("le=\"").nth(1))
                .filter_map(|rest| rest.split('"').next())
                .map(str::to_owned)
                .collect()
        };
        let retirement = boundaries(RETIREMENT_WAIT);
        assert!(!retirement.is_empty(), "the histogram renders its buckets");
        assert_eq!(
            retirement,
            boundaries("live_edit_catchup_build_wait_seconds"),
            "the retirement wait shares the admission waits' boundaries"
        );
    }

    #[test]
    fn allocator_memory_gauge_with_no_reading_emits_no_sample_for_that_family_only() {
        let metrics = metrics_scrape_only("test-pod");
        let allocated = Arc::new(AtomicU64::new(u64::MAX));
        let resident = Arc::new(AtomicU64::new(30 << 20));
        metrics.register_allocator_memory_gauges(
            "replica-a",
            cell_source(&allocated),
            cell_source(&resident),
        );

        let text = render(&metrics);
        let series = series_lines(&text, JEMALLOC_ALLOCATED);
        assert!(
            series.is_empty(),
            "a failed `stats.allocated` read must emit no sample rather than a misleading 0; \
             got {series:?}"
        );
        assert_eq!(
            sample_value(&text, JEMALLOC_RESIDENT, &[("service", "replica-a")]),
            Some((30u64 << 20) as f64),
            "one source having no reading must not suppress the other family"
        );
    }

    // ── checkpoint pacing config-echo family (b.ijf) ────────────────────
    //
    // The family exists so an alert rule can divide by this replica's own
    // pacing configuration instead of hardcoding the numbers that only hold
    // at the defaults. Two properties carry that: the family is absent
    // unless registered, and each gauge renders the exact value it was given
    // under a `service` label an operator can `max()` over.
    //
    // These cases pin *this method* only: asked to register, it registers,
    // and it echoes what it was handed. WHO asks — the "iff this replica
    // checkpoints" predicate that makes the family's presence a signal — is
    // owned and tested at
    // `checkpoint::register_checkpoint_pacing_gauges_if_enabled`.

    /// The six members of the config-echo family, in argument order followed by
    /// the derived pair. Members 1-3 and 5 are the durations
    /// `register_checkpoint_pacing_gauges` takes; member 4 is derived inside it
    /// from the first three and member 6 from the fifth, which is why the
    /// method takes four values and publishes six gauges.
    const CHECKPOINT_PACING_GAUGES: [&str; 6] = [
        "live_edit_checkpoint_config_interval_seconds",
        "live_edit_checkpoint_config_quiesce_seconds",
        "live_edit_checkpoint_config_tick_seconds",
        "live_edit_checkpoint_config_stuck_threshold_seconds",
        "live_edit_checkpoint_config_adapter_call_ceiling_seconds",
        "live_edit_checkpoint_config_reconcile_grace_seconds",
    ];

    #[test]
    fn checkpoint_pacing_gauges_absent_until_registered() {
        // Registration must be what materialises the family. A
        // pre-registered instrument would emit pacing config on every
        // replica, which is what would rob the family's presence of its
        // meaning — but the decision to call this at all is not made here,
        // so this case claims nothing about when it is called. That
        // predicate lives in
        // `checkpoint::register_checkpoint_pacing_gauges_if_enabled`.
        let metrics = metrics_scrape_only("test-pod");
        let before = render(&metrics);
        for name in CHECKPOINT_PACING_GAUGES {
            assert!(
                !has_metric(&before, name),
                "{name} must be absent from /metrics until \
                 register_checkpoint_pacing_gauges is called — registration is \
                 what materialises the family, so a caller that never asks \
                 publishes nothing"
            );
        }

        metrics.register_checkpoint_pacing_gauges(
            Duration::from_secs(60),
            Duration::from_secs(30),
            Duration::from_secs(10),
            Duration::from_secs(180),
            "test-pod",
        );
        let after = render(&metrics);
        for name in CHECKPOINT_PACING_GAUGES {
            assert_eq!(
                metric_type(&after, name).as_deref(),
                Some("gauge"),
                "{name} must render as a gauge after registration — a counter \
                 or untyped family would break `max()` and any recipe that \
                 divides by it"
            );
        }
    }

    #[test]
    fn checkpoint_pacing_gauges_echo_each_configured_value_under_the_service_label() {
        // Two rows, neither at the defaults and all six values distinct
        // within a row: a registration that transposed two arguments, or that
        // echoed a constant instead of its input, passes at 60/30/10/100 by
        // coincidence far too easily.
        //
        // The second row is sub-second on every *pacing* term. The inputs are
        // `Duration`s, so fractional seconds are representable end to end;
        // an integer-typed instrument would floor 0.5 to 0 and turn a divisor
        // into a division by zero in an operator's recipe. It also pins that
        // the derived stuck-threshold gauge does not round through an integer
        // on the way to `f64` — 1.5 + 0.25 + 0.5 must come back as 2.25, not 2.
        //
        // The two b.gg7 members are the adapter call ceiling (echoed) and the
        // reconciliation grace (derived from it via `reconcile_grace`). Row 1's
        // 180 s ceiling exercises the multiply arm (× RECONCILE_GRACE_MULTIPLE
        // = 540 s); row 2's sub-second ceiling exercises the
        // RECONCILE_GRACE_FLOOR arm (0.75 × 3 = 2.25 s, floored to 60 s), which
        // is the case an operator on a fast local adapter actually sees.
        let rows: [(Duration, Duration, Duration, Duration, [f64; 6]); 2] = [
            (
                Duration::from_secs(300),
                Duration::from_secs(45),
                Duration::from_secs(5),
                Duration::from_secs(180),
                [300.0, 45.0, 5.0, 350.0, 180.0, 540.0],
            ),
            (
                Duration::from_millis(1_500),
                Duration::from_millis(250),
                Duration::from_millis(500),
                Duration::from_millis(750),
                [1.5, 0.25, 0.5, 2.25, 0.75, 60.0],
            ),
        ];

        for (interval, quiesce, tick, ceiling, expected) in rows {
            // A fresh `Metrics` per row: the instruments are registered by
            // name, so reusing one meter would leave the previous row's
            // series in the exposition alongside this row's.
            let metrics = metrics_scrape_only("ckpt-replica-7");
            metrics.register_checkpoint_pacing_gauges(
                interval,
                quiesce,
                tick,
                ceiling,
                "ckpt-replica-7",
            );
            let text = render(&metrics);
            let service = [("service", "ckpt-replica-7")];

            for (name, expected) in CHECKPOINT_PACING_GAUGES.iter().zip(expected) {
                assert_eq!(
                    sample_value(&text, name, &service),
                    Some(expected),
                    "{name} must echo the value it was registered with, under \
                     the `service` label, for pacing ({interval:?}, {quiesce:?}, \
                     {tick:?}, {ceiling:?}) — the whole point of the family is that \
                     an alert rule can read the number back out"
                );
            }
        }
    }

    #[test]
    fn checkpoint_pacing_gauges_carry_the_service_label_argument_not_the_meter_label() {
        // The `service` label is the argument, matching
        // `record_checkpoint_pending_aggregates` so the `max()`-never-`sum()`
        // idiom carries across a recipe that joins the two families. A series
        // emitted bare (or under some other label) cannot be joined at all.
        let metrics = metrics_scrape_only("test-pod");
        metrics.register_checkpoint_pacing_gauges(
            Duration::from_secs(60),
            Duration::from_secs(30),
            Duration::from_secs(10),
            Duration::from_secs(180),
            "labelled-replica",
        );
        let text = render(&metrics);
        let name = "live_edit_checkpoint_config_interval_seconds";
        assert_eq!(
            sample_value(&text, name, &[("service", "labelled-replica")]),
            Some(60.0),
            "the series must be labelled with the `service` argument"
        );
        assert_eq!(
            sample_value(&text, name, &[("service", "some-other-replica")]),
            None,
            "and must not match a different service value"
        );
    }

    #[test]
    fn checkpoint_pacing_gauges_re_observe_on_every_scrape() {
        // Observable gauges (callback per collect) rather than a set-once
        // `Gauge` — so a second scrape still carries the values. A set-once
        // gauge read by a delta-temporality reader can drop a value only ever
        // written at startup, which would make these silently disappear.
        let metrics = metrics_scrape_only("test-pod");
        metrics.register_checkpoint_pacing_gauges(
            Duration::from_secs(60),
            Duration::from_secs(30),
            Duration::from_secs(10),
            Duration::from_secs(180),
            "test-pod",
        );
        let service = [("service", "test-pod")];
        let _first = render(&metrics);
        let second = render(&metrics);
        assert_eq!(
            sample_value(
                &second,
                "live_edit_checkpoint_config_stuck_threshold_seconds",
                &service
            ),
            Some(100.0),
            "the family must survive a second collect — these are process \
             constants and must be present on every scrape, not just the first"
        );
    }

    // ── OTLP-reader endpoint gating (mirror of b.5yt build_otel_layer) ──
    //
    // `otlp_metrics_endpoint()` is the sole gate deciding whether the OTLP
    // push reader is attached to the dual-reader provider. These tests pin the
    // refinement-2 independence guarantee: the per-signal
    // `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` is the ONLY control — there is no
    // fallback to the generic `OTEL_EXPORTER_OTLP_ENDPOINT`, so enabling
    // tracing alone must never silently start pushing metrics.

    #[test]
    fn otlp_metrics_endpoint_none_when_both_unset() {
        temp_env::with_vars(
            [
                (METRICS_ENDPOINT_VAR, None::<&str>),
                (GENERIC_ENDPOINT_VAR, None::<&str>),
            ],
            || {
                assert_eq!(
                    otlp_metrics_endpoint(),
                    Ok(None),
                    "no OTLP reader when neither endpoint var is set"
                );
            },
        );
    }

    #[test]
    fn otlp_metrics_endpoint_some_when_metrics_var_set() {
        temp_env::with_vars(
            [
                (METRICS_ENDPOINT_VAR, Some("http://collector:4317")),
                (GENERIC_ENDPOINT_VAR, None::<&str>),
            ],
            || {
                assert_eq!(
                    otlp_metrics_endpoint()
                        .expect("a valid endpoint is accepted")
                        .map(|endpoint| endpoint.url)
                        .as_deref(),
                    Some("http://collector:4317"),
                    "OTLP reader attaches when the per-signal metrics endpoint is set"
                );
            },
        );
    }

    #[test]
    fn otlp_metrics_endpoint_ignores_generic_trace_endpoint() {
        // Refinement-2 independence: only the generic (tracing) endpoint set →
        // metrics push must stay OFF.
        temp_env::with_vars(
            [
                (METRICS_ENDPOINT_VAR, None::<&str>),
                (GENERIC_ENDPOINT_VAR, Some("http://collector:4317")),
            ],
            || {
                assert_eq!(
                    otlp_metrics_endpoint(),
                    Ok(None),
                    "generic OTEL_EXPORTER_OTLP_ENDPOINT must NOT enable metrics push"
                );
            },
        );
    }

    #[test]
    fn otlp_metrics_endpoint_blank_is_none() {
        temp_env::with_vars(
            [
                (METRICS_ENDPOINT_VAR, Some("   ")),
                (GENERIC_ENDPOINT_VAR, None::<&str>),
            ],
            || {
                assert_eq!(
                    otlp_metrics_endpoint(),
                    Ok(None),
                    "a whitespace-only endpoint must be treated as unset"
                );
            },
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn meter_provider_builds_with_otlp_reader_attached_and_scrape_still_works() {
        // Mirror of the b.5yt `build_otel_layer_with_endpoint…` test: with the
        // metrics endpoint set, the dual-reader provider must build (OTLP push
        // arm attached) AND the always-on scrape arm must still render. The
        // set-endpoint window is scoped to the sync build under `temp_env`.
        let metrics = temp_env::with_vars(
            [
                (METRICS_ENDPOINT_VAR, Some("http://127.0.0.1:0")),
                (GENERIC_ENDPOINT_VAR, None::<&str>),
            ],
            || Metrics::new("otlp-attached-test"),
        );
        metrics.inc_calls("Ping", "otlp-attached-test", "OK");
        let text = render(&metrics);
        assert!(
            has_metric(&text, "live_endpoint_calls_total"),
            "scrape arm must keep working when the OTLP push arm is also attached"
        );
        metrics.shutdown();
    }
}
