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

//! Exact accounting of every span handed to the OTLP trace exporter.
//!
//! The SDK's `BatchSpanProcessor` drops a span when its queue is full, but it
//! keeps that count private: it logs one WARN at the first drop and the total
//! only at shutdown, and it logs an ERROR for every batch whose export fails.
//! Neither is something an operator can graph or alert on. This module wraps
//! the processor and its exporter so every span that reaches the processor
//! ends in exactly one of three counters — `exported`, `export_failed` or
//! `queue_full` — and the two loss counters surface through a rate-limited
//! WARN ([`TRACE_SPANS_DROPPED_MSG`]) and the
//! `live_edit_trace_spans_{exported,dropped}_total` metrics.
//!
//! **Why the wrapper bounds the queue itself.** [`CountingSpanProcessor`]
//! admits a span only while fewer than `capacity` admitted spans are still
//! un-exported (`pending`), and [`ReleasingExporter`] releases a batch from
//! `pending` when the processor hands it to the exporter. The processor's own
//! queue is built with the same capacity, and a span stays counted in
//! `pending` from admission until its batch is exported — through the whole
//! time it sits in the processor's queue — so the processor's queue can never
//! be full when a span is admitted. A drop therefore always happens here,
//! where it is counted, and never inside the SDK, where it would not be.
//!
//! **Why a failed export returns `Ok`.** [`ReleasingExporter`] counts the
//! batch as `export_failed`, keeps the error text for the WARN, and returns
//! `Ok(())` to the processor. The failure is still reported, once per WARN
//! interval rather than as one SDK ERROR line per batch; an unreachable
//! collector would otherwise log an ERROR every few seconds.
//!
//! **What is not counted.** Spans still queued when the processor's shutdown
//! timeout (5 s) or the service's post-budget flush cap runs out, and spans
//! ended during or after shutdown. Most of the latter the SDK discards before
//! any processor sees them, once the provider reports itself shut down. A span
//! that passed that check just before the flag flipped can still reach
//! [`CountingSpanProcessor`] after the batch worker has drained or exited: it
//! is admitted into `pending`, then either fails to enqueue (the worker's
//! channel is gone) or sits in the queue unexported, and neither counter
//! records it.

use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use opentelemetry::Context;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{
    BatchConfigBuilder, BatchSpanProcessor, SdkTracerProvider, Span, SpanData, SpanExporter,
    SpanProcessor,
};
use opentelemetry_sdk::Resource;

/// Message of the rate-limited WARN that reports lost spans.
pub const TRACE_SPANS_DROPPED_MSG: &str = "trace spans dropped";

/// Minimum interval between two [`TRACE_SPANS_DROPPED_MSG`] lines. The final
/// line at exporter shutdown ignores it.
///
/// Shared with the metrics push WARN
/// ([`super::metrics_export::OTLP_METRICS_EXPORT_FAILED_MSG`]), which uses
/// the same interval, so retuning it here changes both lines.
pub const DROP_WARN_INTERVAL: Duration = Duration::from_secs(60);

/// Queue capacity when `OTEL_BSP_MAX_QUEUE_SIZE` is unset or unparseable — the
/// SDK's own default.
pub const DEFAULT_MAX_QUEUE_SIZE: usize = 2_048;

/// The SDK variable the queue capacity is read from.
pub const OTEL_BSP_MAX_QUEUE_SIZE: &str = "OTEL_BSP_MAX_QUEUE_SIZE";

/// Parse a raw `OTEL_BSP_MAX_QUEUE_SIZE` value exactly as the SDK does
/// (`usize::from_str` of the untrimmed value): `None` or an unparseable value
/// yields [`DEFAULT_MAX_QUEUE_SIZE`].
///
/// The capacity is passed to the batch processor explicitly, so this and the
/// SDK can never disagree; parsing it the SDK's way keeps an operator's value
/// meaning what the SDK documentation says it means.
pub fn max_queue_size_from(raw: Option<&str>) -> usize {
    raw.and_then(|value| usize::from_str(value).ok())
        .unwrap_or(DEFAULT_MAX_QUEUE_SIZE)
}

/// [`max_queue_size_from`] applied to the process environment.
pub fn max_queue_size_from_env() -> usize {
    max_queue_size_from(std::env::var(OTEL_BSP_MAX_QUEUE_SIZE).ok().as_deref())
}

/// Clock the WARN rate limit reads. Injectable so a test can step time.
pub type Clock = Arc<dyn Fn() -> Instant + Send + Sync>;

/// Shared accounting for one tracer provider's span pipeline.
///
/// Held by [`CountingSpanProcessor`] (admission), [`ReleasingExporter`]
/// (release, outcome, WARN) and, through `main`, by the metrics registry
/// (`Metrics::register_trace_export_counters`).
pub struct TraceExportState {
    capacity: usize,
    pending: AtomicUsize,
    queue_full: AtomicU64,
    export_failed: AtomicU64,
    exported: AtomicU64,
    reporter: Mutex<DropReporter>,
    clock: Clock,
}

/// What the previous WARN already reported, and when.
#[derive(Default)]
struct DropReporter {
    reported_queue_full: u64,
    reported_export_failed: u64,
    last_warn: Option<Instant>,
    /// Text of the most recent failed export not yet carried by a WARN.
    last_export_error: Option<String>,
}

/// One WARN's worth of figures, built under the reporter lock and emitted
/// after it is released.
struct DropLine {
    queue_full: u64,
    export_failed: u64,
    queue_full_total: u64,
    export_failed_total: u64,
    last_export_error: Option<String>,
}

impl fmt::Debug for TraceExportState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TraceExportState")
            .field("capacity", &self.capacity)
            .field("pending", &self.pending())
            .field("exported", &self.exported_total())
            .field("queue_full", &self.queue_full_total())
            .field("export_failed", &self.export_failed_total())
            .finish_non_exhaustive()
    }
}

impl TraceExportState {
    /// State for a pipeline whose processor queue holds `capacity` spans.
    pub fn new(capacity: usize) -> Self {
        Self::with_clock(capacity, Arc::new(Instant::now))
    }

    /// [`Self::new`] with an injected clock for the WARN rate limit.
    pub fn with_clock(capacity: usize, clock: Clock) -> Self {
        Self {
            capacity,
            pending: AtomicUsize::new(0),
            queue_full: AtomicU64::new(0),
            export_failed: AtomicU64::new(0),
            exported: AtomicU64::new(0),
            reporter: Mutex::new(DropReporter::default()),
            clock,
        }
    }

    /// The queue capacity, `OTEL_BSP_MAX_QUEUE_SIZE`.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Spans admitted and not yet handed to the exporter.
    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::Relaxed)
    }

    /// Spans the exporter delivered, since start.
    pub fn exported_total(&self) -> u64 {
        self.exported.load(Ordering::Relaxed)
    }

    /// Spans dropped because `capacity` spans were already pending, since
    /// start.
    pub fn queue_full_total(&self) -> u64 {
        self.queue_full.load(Ordering::Relaxed)
    }

    /// Spans in batches whose export failed, since start.
    pub fn export_failed_total(&self) -> u64 {
        self.export_failed.load(Ordering::Relaxed)
    }

    /// Admit one span if fewer than `capacity` are pending; otherwise count it
    /// `queue_full`. Returns whether it was admitted.
    fn try_admit(&self) -> bool {
        let admitted = self
            .pending
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |pending| {
                (pending < self.capacity).then_some(pending + 1)
            })
            .is_ok();
        if !admitted {
            self.queue_full.fetch_add(1, Ordering::Relaxed);
        }
        admitted
    }

    /// Release `n` spans the processor has taken out of its queue.
    fn release(&self, n: usize) {
        // Saturating: a release can never exceed what was admitted, because
        // every span the processor exports went through `try_admit`. The
        // saturation only keeps a broken invariant from wrapping the counter.
        let _ = self
            .pending
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |pending| {
                Some(pending.saturating_sub(n))
            });
    }

    fn record_exported(&self, n: usize) {
        self.exported.fetch_add(n as u64, Ordering::Relaxed);
    }

    fn record_export_failed(&self, n: usize, error: String) {
        self.export_failed.fetch_add(n as u64, Ordering::Relaxed);
        self.lock_reporter().last_export_error = Some(error);
    }

    fn lock_reporter(&self) -> std::sync::MutexGuard<'_, DropReporter> {
        // A panic while holding the lock leaves plain counters behind; they
        // are still consistent enough to report from.
        self.reporter.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Emit [`TRACE_SPANS_DROPPED_MSG`] if spans were lost since the previous
    /// line and, unless `force`, at least [`DROP_WARN_INTERVAL`] has passed
    /// since it. The first line after start is never held back.
    fn report_drops(&self, force: bool) {
        let now = (self.clock)();
        let line = {
            let mut reporter = self.lock_reporter();
            let queue_full_total = self.queue_full_total();
            let export_failed_total = self.export_failed_total();
            let queue_full = queue_full_total.saturating_sub(reporter.reported_queue_full);
            let export_failed = export_failed_total.saturating_sub(reporter.reported_export_failed);
            if queue_full == 0 && export_failed == 0 {
                return;
            }
            if !force {
                if let Some(last) = reporter.last_warn {
                    if now.saturating_duration_since(last) < DROP_WARN_INTERVAL {
                        return;
                    }
                }
            }
            reporter.reported_queue_full = queue_full_total;
            reporter.reported_export_failed = export_failed_total;
            reporter.last_warn = Some(now);
            DropLine {
                queue_full,
                export_failed,
                queue_full_total,
                export_failed_total,
                last_export_error: reporter.last_export_error.take(),
            }
        };
        tracing::warn!(
            queue_full = line.queue_full,
            export_failed = line.export_failed,
            queue_full_total = line.queue_full_total,
            export_failed_total = line.export_failed_total,
            max_queue_size = self.capacity,
            last_export_error = line.last_export_error.as_deref(),
            "{}",
            TRACE_SPANS_DROPPED_MSG
        );
    }
}

/// Span processor that admits a span into `inner` only while the shared
/// state has room, counting every refused span as `queue_full`.
///
/// Every other call passes straight through to `inner`.
#[derive(Debug)]
pub struct CountingSpanProcessor<P> {
    inner: P,
    state: Arc<TraceExportState>,
}

impl<P> CountingSpanProcessor<P> {
    pub fn new(inner: P, state: Arc<TraceExportState>) -> Self {
        Self { inner, state }
    }
}

impl<P: SpanProcessor> SpanProcessor for CountingSpanProcessor<P> {
    fn on_start(&self, span: &mut Span, cx: &Context) {
        self.inner.on_start(span, cx);
    }

    fn on_end(&self, span: SpanData) {
        if self.state.try_admit() {
            self.inner.on_end(span);
        }
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }

    fn shutdown(&self) -> OTelSdkResult {
        self.inner.shutdown()
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}

/// Exporter that releases each batch from `pending`, counts its outcome, and
/// emits the rate-limited drop WARN.
///
/// A failed export is counted `export_failed` and answered `Ok(())` (see the
/// module doc). `set_resource` must reach the inner exporter — without it the
/// exported spans lose `service.name` — and both shutdown entry points pass
/// through because the OTLP client overrides `shutdown` itself. Shutdown ends
/// with a final WARN for any loss not yet reported.
#[derive(Debug)]
pub struct ReleasingExporter<E> {
    inner: E,
    state: Arc<TraceExportState>,
}

impl<E> ReleasingExporter<E> {
    pub fn new(inner: E, state: Arc<TraceExportState>) -> Self {
        Self { inner, state }
    }
}

impl<E: SpanExporter> SpanExporter for ReleasingExporter<E> {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        let n = batch.len();
        self.state.release(n);
        match self.inner.export(batch).await {
            Ok(()) => self.state.record_exported(n),
            Err(error) => self.state.record_export_failed(n, error.to_string()),
        }
        self.state.report_drops(false);
        Ok(())
    }

    fn shutdown_with_timeout(&mut self, timeout: Duration) -> OTelSdkResult {
        let result = self.inner.shutdown_with_timeout(timeout);
        self.state.report_drops(true);
        result
    }

    fn shutdown(&mut self) -> OTelSdkResult {
        let result = self.inner.shutdown();
        self.state.report_drops(true);
        result
    }

    fn force_flush(&mut self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}

/// Build a tracer provider whose single processor is a
/// [`CountingSpanProcessor`] around a `BatchSpanProcessor` exporting through
/// a [`ReleasingExporter`] around `exporter`.
///
/// The batch processor's queue capacity is `state.capacity()`, so the
/// wrapper's bound and the processor's are the same number. The processor's
/// other `OTEL_BSP_*` settings, and the provider's sampler and resource, are
/// still read from the environment by the SDK.
pub fn build_tracer_provider<E>(exporter: E, state: Arc<TraceExportState>) -> SdkTracerProvider
where
    E: SpanExporter + 'static,
{
    let batch = BatchSpanProcessor::builder(ReleasingExporter::new(exporter, Arc::clone(&state)))
        .with_batch_config(
            BatchConfigBuilder::default()
                .with_max_queue_size(state.capacity())
                .build(),
        )
        .build();
    SdkTracerProvider::builder()
        .with_span_processor(CountingSpanProcessor::new(batch, state))
        .build()
}

/// The startup line `main` logs when trace export is enabled: the transport
/// the endpoint's scheme selected, the trace filter, the queue capacity, and
/// the sampler variables exactly as set. Deliberately not the endpoint.
pub fn log_trace_export_enabled(
    trace_filter: &str,
    state: &TraceExportState,
    transport: super::otlp::OtlpTransport,
) {
    let sampler = std::env::var("OTEL_TRACES_SAMPLER").ok();
    let sampler_arg = std::env::var("OTEL_TRACES_SAMPLER_ARG").ok();
    tracing::info!(
        transport = transport.as_str(),
        trace_filter,
        max_queue_size = state.capacity(),
        otel_traces_sampler = sampler.as_deref(),
        otel_traces_sampler_arg = sampler_arg.as_deref(),
        "OpenTelemetry trace export enabled"
    );
}

/// Test doubles for tests that capture what tracing and trace export carried,
/// shared by the crate's unit tests (this module's, the subscriber registry's
/// in `observability`, the metrics registration's, and the per-RPC server
/// span's in `grpc::metrics_layer`) and the integration tests (the log
/// capture and the server-span test under `tests/integration/grpc/`).
///
/// Compiled only under `test` or the `test-support` feature, like
/// `storage::test_support`, so none of it reaches the release binary. Every
/// item uses normal dependencies only, except `collecting_tracer`, which is
/// unit-test-only — see its doc.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex, OnceLock, PoisonError};

    use opentelemetry_sdk::error::OTelSdkResult;
    use opentelemetry_sdk::trace::{SpanData, SpanExporter};

    use super::TraceExportState;

    /// Exporter that keeps every span it is handed, in arrival order.
    #[derive(Clone, Debug, Default)]
    pub struct CollectingExporter {
        spans: Arc<Mutex<Vec<SpanData>>>,
    }

    impl CollectingExporter {
        /// Every span exported so far.
        pub fn spans(&self) -> Vec<SpanData> {
            self.spans
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }
    }

    impl SpanExporter for CollectingExporter {
        async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
            self.spans
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .extend(batch);
            Ok(())
        }
    }

    /// A tracer whose spans are exported synchronously as each one ends — a
    /// simple processor, no batch thread — into the returned
    /// [`CollectingExporter`], so a span is readable the moment it closes.
    ///
    /// The sampler variables are pinned unset while the provider reads them,
    /// so the SDK default `ParentBased(AlwaysOn)` applies whatever the ambient
    /// environment holds. Keep the provider alive for the test's duration.
    ///
    /// Unit tests only (`#[cfg(test)]`): the pinning uses `temp-env`, a
    /// dev-dependency, which a `test-support` build of the library cannot
    /// reach. An integration test builds the same provider inline around a
    /// [`CollectingExporter`].
    #[cfg(test)]
    pub(crate) fn collecting_tracer() -> (
        opentelemetry_sdk::trace::SdkTracerProvider,
        opentelemetry_sdk::trace::SdkTracer,
        CollectingExporter,
    ) {
        use opentelemetry::trace::TracerProvider as _;
        use opentelemetry_sdk::trace::SdkTracerProvider;

        let exporter = CollectingExporter::default();
        let provider = temp_env::with_vars(
            [
                ("OTEL_TRACES_SAMPLER", None::<&str>),
                ("OTEL_TRACES_SAMPLER_ARG", None),
            ],
            || {
                SdkTracerProvider::builder()
                    .with_simple_exporter(exporter.clone())
                    .build()
            },
        );
        let tracer = provider.tracer("live-edit");
        (provider, tracer, exporter)
    }

    /// Keep a second dispatcher registered for the rest of the process, so a
    /// test's scoped subscriber is never the only one. Call it before
    /// installing any scoped capture.
    ///
    /// While at most one dispatcher is registered, tracing-core caches a
    /// callsite's interest from the default dispatcher of whichever thread
    /// first hits it (`Rebuilder::JustOne`, tracing-core 0.1.36
    /// `callsite.rs:544-566`). A scoped capture is the default only inside its
    /// own scope, so a callsite first hit on another thread — a sibling test,
    /// or a spawned task logging the same line while the capture is live — is
    /// cached as `Interest::never` for the whole process, and the captured
    /// scope's own span or event at that callsite is then dropped. With two
    /// registered, interest is combined across both (`never` and `always` make
    /// `sometimes`) and every span or event is checked against the dispatcher
    /// in scope, so a capture sees exactly what is emitted inside it. The
    /// second dispatcher is a `NoSubscriber`: it records nothing.
    pub fn keep_a_second_dispatcher_registered() {
        static SECOND: OnceLock<tracing::Dispatch> = OnceLock::new();
        SECOND.get_or_init(|| tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default()));
    }

    /// A `MakeWriter` that appends everything written through it to one
    /// shared buffer, for tests that read what a fmt layer wrote.
    #[derive(Clone, Default)]
    pub struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

    impl CapturingWriter {
        /// Everything written so far, as text.
        pub fn text(&self) -> String {
            let raw = self
                .0
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone();
            String::from_utf8(raw).expect("captured output must be UTF-8")
        }
    }

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
                .unwrap_or_else(PoisonError::into_inner)
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Advance `state`'s three counters by the given amounts through the same
    /// recorders the pipeline uses, for tests of code that only reads them (the
    /// metrics registration).
    pub fn advance_counters(
        state: &TraceExportState,
        exported: usize,
        queue_full: u64,
        export_failed: usize,
    ) {
        state.record_exported(exported);
        state.queue_full.fetch_add(queue_full, Ordering::Relaxed);
        state.record_export_failed(export_failed, "test export failure".to_string());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::sync::{mpsc, Condvar, MutexGuard};

    use futures::FutureExt as _;
    use opentelemetry::trace::{Span as _, Tracer as _, TracerProvider as _};
    use opentelemetry::Key;
    use opentelemetry_sdk::error::OTelSdkError;

    use super::test_support::{
        collecting_tracer, keep_a_second_dispatcher_registered, CapturingWriter,
    };
    use super::*;

    /// Upper bound on any wait for the batch worker. Unreachable on the
    /// healthy path, where every wait returns in milliseconds.
    const SETTLE: Duration = Duration::from_secs(10);

    /// `service.name` the pipeline tests pin, to recognise the provider's
    /// resource at the inner exporter.
    const SERVICE_NAME: &str = "trace-export-test-service";

    /// SDK environment pinned while a pipeline is built:
    ///
    /// - a one-hour schedule delay, so the batch worker exports only when a
    ///   full batch or a flush asks it to and the tests decide when exports
    ///   happen;
    /// - every other `OTEL_BSP_*` variable unset, so the queue and batch sizes
    ///   are exactly what `build_tracer_provider` gives the processor;
    /// - the sampler unset (always on), and a fixed service name.
    const PINNED_SDK_ENV: [(&str, Option<&str>); 9] = [
        ("OTEL_BSP_SCHEDULE_DELAY", Some("3600000")),
        (OTEL_BSP_MAX_QUEUE_SIZE, None),
        ("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", None),
        ("OTEL_BSP_EXPORT_TIMEOUT", None),
        ("OTEL_BSP_MAX_CONCURRENT_EXPORTS", None),
        ("OTEL_TRACES_SAMPLER", None),
        ("OTEL_TRACES_SAMPLER_ARG", None),
        ("OTEL_SERVICE_NAME", Some(SERVICE_NAME)),
        ("OTEL_RESOURCE_ATTRIBUTES", None),
    ];

    fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(PoisonError::into_inner)
    }

    // ── Inner exporter double ────────────────────────────────────────────────

    /// What a [`ProbeExporter`] saw, shared with the test.
    #[derive(Debug, Default)]
    struct Probe {
        /// Sizes of the batches it exported successfully, in order.
        batches: Mutex<Vec<usize>>,
        /// While `false`, every export waits inside `export` — the batch
        /// worker is then held mid-export, as behind a slow collector.
        open: Mutex<bool>,
        opened: Condvar,
        /// Told each batch's size as an export begins, before the gate.
        entered: Mutex<Option<mpsc::Sender<usize>>>,
        /// Fail every export with this message instead of exporting.
        fail_with: Option<&'static str>,
        resource: Mutex<Option<Resource>>,
        shut_down: AtomicBool,
    }

    /// The exporter `ReleasingExporter` wraps in these tests.
    #[derive(Debug, Clone)]
    struct ProbeExporter(Arc<Probe>);

    impl ProbeExporter {
        fn passing() -> Self {
            Self(Arc::new(Probe {
                open: Mutex::new(true),
                ..Probe::default()
            }))
        }

        fn failing(message: &'static str) -> Self {
            Self(Arc::new(Probe {
                open: Mutex::new(true),
                fail_with: Some(message),
                ..Probe::default()
            }))
        }

        /// An exporter that holds every export until [`Self::open`], and
        /// reports each batch's size on the returned receiver as it arrives.
        fn gated() -> (Self, mpsc::Receiver<usize>) {
            let (tx, rx) = mpsc::channel();
            let exporter = Self(Arc::new(Probe {
                entered: Mutex::new(Some(tx)),
                ..Probe::default()
            }));
            (exporter, rx)
        }

        fn open(&self) {
            *lock(&self.0.open) = true;
            self.0.opened.notify_all();
        }

        fn exported_spans(&self) -> usize {
            lock(&self.0.batches).iter().sum()
        }

        fn resource(&self) -> Option<Resource> {
            lock(&self.0.resource).clone()
        }

        fn was_shut_down(&self) -> bool {
            self.0.shut_down.load(Ordering::SeqCst)
        }
    }

    impl SpanExporter for ProbeExporter {
        async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
            let n = batch.len();
            if let Some(entered) = lock(&self.0.entered).as_ref() {
                let _ = entered.send(n);
            }
            let mut open = lock(&self.0.open);
            while !*open {
                open = self
                    .0
                    .opened
                    .wait(open)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            drop(open);
            if let Some(message) = self.0.fail_with {
                return Err(OTelSdkError::InternalFailure(message.to_string()));
            }
            lock(&self.0.batches).push(n);
            Ok(())
        }

        fn shutdown_with_timeout(&mut self, _timeout: Duration) -> OTelSdkResult {
            self.0.shut_down.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn set_resource(&mut self, resource: &Resource) {
            *lock(&self.0.resource) = Some(resource.clone());
        }
    }

    // ── Pipeline helpers ─────────────────────────────────────────────────────

    /// The production pipeline (`build_tracer_provider`) over `exporter`, with
    /// a queue of `capacity` spans and the SDK environment pinned.
    fn pipeline(
        capacity: usize,
        exporter: ProbeExporter,
    ) -> (SdkTracerProvider, Arc<TraceExportState>) {
        let state = Arc::new(TraceExportState::new(capacity));
        let provider = temp_env::with_vars(PINNED_SDK_ENV, || {
            build_tracer_provider(exporter, Arc::clone(&state))
        });
        (provider, state)
    }

    /// Start and end `n` spans on the calling thread.
    fn end_spans(provider: &SdkTracerProvider, n: usize) {
        let tracer = provider.tracer("trace-export-test");
        for _ in 0..n {
            let mut span = tracer.start("flood");
            span.end();
        }
    }

    /// Every span the pipeline has accounted for, across the three outcomes.
    fn accounted(state: &TraceExportState) -> u64 {
        state.exported_total() + state.export_failed_total() + state.queue_full_total()
    }

    // ── T3: exact loss accounting ────────────────────────────────────────────

    /// With the exporter held mid-export, the wrapper admits exactly
    /// `capacity` spans and counts every further one `queue_full` — and once
    /// the exporter is released, every span is in exactly one of the three
    /// counters.
    ///
    /// The phases are ordered by the exporter's entry signal, not by timing:
    /// the first `CAPACITY` spans fill a batch, whose export the test waits to
    /// see begin (releasing them from `pending` at the handoff); the next
    /// `CAPACITY` are admitted; the `REFUSED` after them find `pending` full.
    #[test]
    fn counting_pipeline_with_exporter_held_counts_every_refused_span_queue_full() {
        const CAPACITY: usize = 4;
        const REFUSED: usize = 10;
        const FLOODED: usize = CAPACITY + CAPACITY + REFUSED;

        let (exporter, entered) = ProbeExporter::gated();
        let (provider, state) = pipeline(CAPACITY, exporter.clone());

        end_spans(&provider, CAPACITY);
        assert_eq!(
            entered.recv_timeout(SETTLE),
            Ok(CAPACITY),
            "precondition: a full batch must reach the exporter and be held there"
        );
        assert_eq!(
            state.pending(),
            0,
            "a batch handed to the exporter must leave `pending` at the handoff, so the queue \
             can refill while a slow export runs"
        );

        end_spans(&provider, CAPACITY + REFUSED);
        assert_eq!(
            state.queue_full_total(),
            REFUSED as u64,
            "exactly the spans arriving while {CAPACITY} were pending must be counted \
             queue_full — no more (an admitted span counted lost) and no fewer (a span dropped \
             uncounted)"
        );
        assert_eq!(
            state.pending(),
            CAPACITY,
            "the admitted spans wait in the queue"
        );

        exporter.open();
        provider
            .force_flush()
            .expect("the flush must drain the queue once the exporter is released");

        assert_eq!(state.exported_total(), (CAPACITY * 2) as u64);
        assert_eq!(state.export_failed_total(), 0);
        assert_eq!(
            accounted(&state),
            FLOODED as u64,
            "every span that reached the pipeline must end in exactly one of exported, \
             export_failed and queue_full"
        );
        assert_eq!(
            exporter.exported_spans(),
            CAPACITY * 2,
            "`exported` must count what the inner exporter actually delivered"
        );
        provider.shutdown().expect("shutdown must succeed");
    }

    /// A failed export counts its whole batch `export_failed` and is answered
    /// `Ok` to the processor — the flush that triggered it succeeds — so the
    /// SDK logs no per-batch ERROR and the loss is reported by the WARN and
    /// the counter instead.
    #[test]
    fn counting_pipeline_with_failing_exporter_counts_whole_batches_and_answers_ok() {
        let (provider, state) = pipeline(4, ProbeExporter::failing("collector unreachable"));

        end_spans(&provider, 3);
        provider
            .force_flush()
            .expect("a failed export must be answered Ok to the processor");
        assert_eq!(
            state.export_failed_total(),
            3,
            "the whole batch of 3 failed"
        );

        end_spans(&provider, 2);
        provider
            .force_flush()
            .expect("a failed export must be answered Ok to the processor");
        assert_eq!(
            state.export_failed_total(),
            5,
            "each failed batch adds its own size"
        );
        assert_eq!(state.exported_total(), 0);
        assert_eq!(state.queue_full_total(), 0);
        assert_eq!(
            state.pending(),
            0,
            "a failed batch is released like any other"
        );
        provider.shutdown().expect("shutdown must succeed");
    }

    /// The provider's resource reaches the inner exporter through both
    /// wrappers; without it every exported span would lose `service.name`.
    #[test]
    fn counting_pipeline_passes_the_provider_resource_to_the_inner_exporter() {
        let exporter = ProbeExporter::passing();
        let (provider, _state) = pipeline(4, exporter.clone());

        // The batch worker applies the resource from its control channel in
        // order, so once a later flush has returned it has been applied.
        provider.force_flush().expect("flush must succeed");

        let resource = exporter
            .resource()
            .expect("the inner exporter must be given the provider's resource");
        assert_eq!(
            resource
                .get(&Key::new("service.name"))
                .map(|v| v.as_str().into_owned()),
            Some(SERVICE_NAME.to_string())
        );
        provider.shutdown().expect("shutdown must succeed");
    }

    /// Shutting the provider down shuts the inner exporter down, so the OTLP
    /// client's own shutdown runs.
    #[test]
    fn counting_pipeline_shutdown_reaches_the_inner_exporter() {
        let exporter = ProbeExporter::passing();
        let (provider, _state) = pipeline(4, exporter.clone());

        provider.shutdown().expect("shutdown must succeed");

        assert!(
            exporter.was_shut_down(),
            "the provider's shutdown must reach the inner exporter's shutdown"
        );
    }

    // ── T8: queue capacity ───────────────────────────────────────────────────

    #[test]
    fn max_queue_size_from_unset_is_the_sdk_default() {
        assert_eq!(DEFAULT_MAX_QUEUE_SIZE, 2_048);
        assert_eq!(max_queue_size_from(None), 2_048);
    }

    #[test]
    fn max_queue_size_from_a_number_is_that_number() {
        assert_eq!(max_queue_size_from(Some("4096")), 4_096);
    }

    /// An unparseable value falls back to the default, exactly as the SDK
    /// does — including an untrimmed number, which the SDK's
    /// `usize::from_str` also refuses.
    #[test]
    fn max_queue_size_from_an_unparseable_value_is_the_sdk_default() {
        for raw in ["abc", "", "-1", "4096.0", " 4096"] {
            assert_eq!(
                max_queue_size_from(Some(raw)),
                DEFAULT_MAX_QUEUE_SIZE,
                "OTEL_BSP_MAX_QUEUE_SIZE={raw:?}"
            );
        }
    }

    #[test]
    fn max_queue_size_from_env_reads_otel_bsp_max_queue_size() {
        temp_env::with_var(OTEL_BSP_MAX_QUEUE_SIZE, Some("4096"), || {
            assert_eq!(max_queue_size_from_env(), 4_096);
        });
        temp_env::with_var(OTEL_BSP_MAX_QUEUE_SIZE, None::<&str>, || {
            assert_eq!(max_queue_size_from_env(), DEFAULT_MAX_QUEUE_SIZE);
        });
    }

    /// The batch processor's own queue holds exactly the admission capacity.
    ///
    /// The capacity here is above the SDK's 2,048 default, so a processor
    /// built without the explicit capacity would hold fewer spans than the
    /// wrapper admits and drop the difference silently inside the SDK. With
    /// the first batch held mid-export, `CAPACITY` more spans are admitted;
    /// every one of them must then be exported.
    #[test]
    fn counting_pipeline_batch_queue_holds_the_whole_admission_capacity() {
        /// Above the SDK default, so a default-sized processor queue is too small.
        const CAPACITY: usize = DEFAULT_MAX_QUEUE_SIZE + 8;
        /// The SDK's default export batch size, below `CAPACITY`.
        const FIRST_BATCH: usize = 512;

        let (exporter, entered) = ProbeExporter::gated();
        let (provider, state) = pipeline(CAPACITY, exporter.clone());

        end_spans(&provider, FIRST_BATCH);
        assert_eq!(
            entered.recv_timeout(SETTLE),
            Ok(FIRST_BATCH),
            "precondition: the first full batch must be held in the exporter"
        );

        end_spans(&provider, CAPACITY);
        assert_eq!(
            state.queue_full_total(),
            0,
            "precondition: the wrapper admits a full capacity of spans"
        );

        exporter.open();
        provider.force_flush().expect("flush must succeed");

        assert_eq!(
            state.exported_total(),
            (FIRST_BATCH + CAPACITY) as u64,
            "every admitted span must be exported: a processor queue smaller than the \
             admission capacity drops the excess inside the SDK, uncounted"
        );
        assert_eq!(accounted(&state), (FIRST_BATCH + CAPACITY) as u64);
        provider.shutdown().expect("shutdown must succeed");
    }

    // ── T4: rate-limited drop WARN ───────────────────────────────────────────

    /// A clock the test steps by hand.
    struct ManualClock(Arc<Mutex<Instant>>);

    impl ManualClock {
        fn start() -> (Self, Clock) {
            let now = Arc::new(Mutex::new(Instant::now()));
            let read = Arc::clone(&now);
            (Self(now), Arc::new(move || *lock(&read)))
        }

        fn advance(&self, by: Duration) {
            *lock(&self.0) += by;
        }
    }

    /// A processor that discards whatever it is handed.
    #[derive(Debug)]
    struct NullProcessor;

    impl SpanProcessor for NullProcessor {
        fn on_start(&self, _span: &mut Span, _cx: &Context) {}
        fn on_end(&self, _span: SpanData) {}
        fn force_flush(&self) -> OTelSdkResult {
            Ok(())
        }
        fn shutdown_with_timeout(&self, _timeout: Duration) -> OTelSdkResult {
            Ok(())
        }
    }

    /// A provider whose spans all reach `CountingSpanProcessor` on the calling
    /// thread. Over a capacity-0 `state`, every span ended through it is a
    /// `queue_full` drop.
    fn counting_provider(state: &Arc<TraceExportState>) -> SdkTracerProvider {
        temp_env::with_vars(PINNED_SDK_ENV, || {
            SdkTracerProvider::builder()
                .with_span_processor(CountingSpanProcessor::new(NullProcessor, Arc::clone(state)))
                .build()
        })
    }

    /// Export `batch` through `exporter` on the calling thread, completing on
    /// the first poll.
    ///
    /// `ReleasingExporter::export` awaits only its inner exporter, and every
    /// caller passes an open [`ProbeExporter`] (`passing` or `failing`), which
    /// never suspends — so the export finishes, WARN included, on this thread
    /// inside [`drop_lines`]'s capture. Polled once rather than run on an
    /// executor so no runtime bridge appears under `src/` (the
    /// `stream_consumer_test` source fence cannot tell test code from
    /// production code).
    fn export_now(exporter: &ReleasingExporter<ProbeExporter>, batch: Vec<SpanData>) {
        exporter
            .export(batch)
            .now_or_never()
            .expect("an export through an open probe exporter completes on its first poll")
            .expect("the releasing exporter always answers Ok");
    }

    /// Run `f` on this thread under a JSON capture and return the
    /// [`TRACE_SPANS_DROPPED_MSG`] lines it emitted. The WARN fires inside the
    /// exporter, which the tests here call directly on the test thread.
    fn drop_lines<T>(f: impl FnOnce() -> T) -> (T, Vec<serde_json::Value>) {
        keep_a_second_dispatcher_registered();
        let writer = CapturingWriter::default();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(writer.clone())
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .finish();
        let value = tracing::subscriber::with_default(subscriber, f);
        let lines = writer
            .text()
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("JSON line"))
            .filter(|event| event["fields"]["message"] == TRACE_SPANS_DROPPED_MSG)
            .collect();
        (value, lines)
    }

    /// Assert one drop line's level and figures.
    fn assert_drop_line(
        line: &serde_json::Value,
        queue_full: u64,
        export_failed: u64,
        queue_full_total: u64,
        export_failed_total: u64,
    ) {
        assert_eq!(line["level"], "WARN", "{line}");
        let fields = &line["fields"];
        assert_eq!(fields["queue_full"], queue_full, "{line}");
        assert_eq!(fields["export_failed"], export_failed, "{line}");
        assert_eq!(fields["queue_full_total"], queue_full_total, "{line}");
        assert_eq!(fields["export_failed_total"], export_failed_total, "{line}");
    }

    /// The first drop is reported once, at the next export — not where the
    /// span is dropped, which is the hot path of whatever ended it.
    #[test]
    fn drop_warn_first_drop_is_reported_once_at_the_next_export() {
        let (_clock, now) = ManualClock::start();
        let state = Arc::new(TraceExportState::with_clock(0, now));
        let provider = counting_provider(&state);
        let exporter = ReleasingExporter::new(ProbeExporter::passing(), Arc::clone(&state));

        let ((), at_drop) = drop_lines(|| end_spans(&provider, 2));
        assert!(
            at_drop.is_empty(),
            "a drop is reported at the next export, not where it happens; got {at_drop:?}"
        );
        assert_eq!(
            state.queue_full_total(),
            2,
            "precondition: both spans dropped"
        );

        let ((), lines) = drop_lines(|| export_now(&exporter, Vec::new()));
        assert_eq!(
            lines.len(),
            1,
            "the first drop must be reported at once: {lines:?}"
        );
        assert_drop_line(&lines[0], 2, 0, 2, 0);
        assert_eq!(lines[0]["fields"]["max_queue_size"], 0);
        assert!(
            lines[0]["fields"].get("last_export_error").is_none(),
            "no export failed, so the line carries no error text: {}",
            lines[0]
        );
    }

    /// Drops inside the interval are counted but not logged; once the
    /// interval has passed the next export logs one line carrying the drops
    /// since the previous line and the running totals.
    ///
    /// The SDK's own reporting — one WARN at the first drop, then nothing
    /// until shutdown — fails the line at 60 s.
    #[test]
    fn drop_warn_holds_later_drops_until_the_interval_then_reports_deltas_and_totals() {
        let (clock, now) = ManualClock::start();
        let state = Arc::new(TraceExportState::with_clock(0, now));
        let provider = counting_provider(&state);
        let exporter = ReleasingExporter::new(ProbeExporter::passing(), Arc::clone(&state));

        end_spans(&provider, 1);
        let ((), first) = drop_lines(|| export_now(&exporter, Vec::new()));
        assert_eq!(first.len(), 1, "precondition: the first drop is reported");

        clock.advance(Duration::from_secs(30));
        end_spans(&provider, 3);
        let ((), held) = drop_lines(|| export_now(&exporter, Vec::new()));
        assert!(
            held.is_empty(),
            "inside the interval nothing is logged: {held:?}"
        );
        assert_eq!(
            state.queue_full_total(),
            4,
            "drops inside the interval are still counted"
        );

        clock.advance(Duration::from_secs(29));
        let ((), still_held) = drop_lines(|| export_now(&exporter, Vec::new()));
        assert!(
            still_held.is_empty(),
            "59 s after the last line is still inside the interval: {still_held:?}"
        );

        clock.advance(Duration::from_secs(1));
        let ((), second) = drop_lines(|| export_now(&exporter, Vec::new()));
        assert_eq!(
            second.len(),
            1,
            "once {DROP_WARN_INTERVAL:?} has passed, the held drops are reported: {second:?}"
        );
        assert_drop_line(&second[0], 3, 0, 4, 0);

        let ((), nothing_new) = drop_lines(|| export_now(&exporter, Vec::new()));
        assert!(
            nothing_new.is_empty(),
            "with no drop since the last line there is nothing to report: {nothing_new:?}"
        );
    }

    /// Shutdown reports what the interval is still holding back, without
    /// waiting for the interval.
    #[test]
    fn drop_warn_shutdown_reports_unreported_drops_inside_the_interval() {
        let (clock, now) = ManualClock::start();
        let state = Arc::new(TraceExportState::with_clock(0, now));
        let provider = counting_provider(&state);
        let mut exporter = ReleasingExporter::new(ProbeExporter::passing(), Arc::clone(&state));

        end_spans(&provider, 1);
        let ((), first) = drop_lines(|| export_now(&exporter, Vec::new()));
        assert_eq!(first.len(), 1, "precondition: the first drop is reported");

        clock.advance(Duration::from_secs(1));
        end_spans(&provider, 2);
        let (result, last) = drop_lines(|| exporter.shutdown());
        result.expect("shutdown must succeed");
        assert_eq!(
            last.len(),
            1,
            "shutdown must report the drops the interval was holding: {last:?}"
        );
        assert_drop_line(&last[0], 2, 0, 3, 0);
    }

    /// With nothing lost, neither an export nor shutdown logs a drop line,
    /// however much time passes.
    #[test]
    fn drop_warn_without_drops_never_logs() {
        let (clock, now) = ManualClock::start();
        let state = Arc::new(TraceExportState::with_clock(4, now));
        let mut exporter = ReleasingExporter::new(ProbeExporter::passing(), Arc::clone(&state));

        let (result, lines) = drop_lines(|| {
            export_now(&exporter, Vec::new());
            clock.advance(DROP_WARN_INTERVAL * 2);
            export_now(&exporter, Vec::new());
            exporter.shutdown()
        });
        result.expect("shutdown must succeed");
        assert!(lines.is_empty(), "no loss, no line: {lines:?}");
    }

    /// A failed export is reported by the same line, with its span count and
    /// the error text, and the exporter still answers `Ok`.
    #[test]
    fn drop_warn_failed_export_reports_its_spans_and_error_text() {
        let (_provider, tracer, collected) = collecting_tracer();
        for _ in 0..2 {
            let mut span = tracer.start("batch-member");
            span.end();
        }
        let batch = collected.spans();
        assert_eq!(batch.len(), 2, "precondition: a batch of two spans");

        let (_clock, now) = ManualClock::start();
        let state = Arc::new(TraceExportState::with_clock(4, now));
        let exporter = ReleasingExporter::new(
            ProbeExporter::failing("collector unreachable"),
            Arc::clone(&state),
        );

        let ((), lines) = drop_lines(|| export_now(&exporter, batch));
        assert_eq!(
            lines.len(),
            1,
            "a failed export is a loss to report: {lines:?}"
        );
        assert_drop_line(&lines[0], 0, 2, 0, 2);
        let error = lines[0]["fields"]["last_export_error"]
            .as_str()
            .unwrap_or_default();
        assert!(
            error.contains("collector unreachable"),
            "the line must carry the failed export's error text; got {}",
            lines[0]
        );
        assert_eq!(state.export_failed_total(), 2);
        assert_eq!(state.exported_total(), 0);
    }
}
