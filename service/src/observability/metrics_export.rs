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

//! A visible WARN for failed OTLP metrics pushes.
//!
//! The SDK logs a failed metrics export only at DEBUG, both in the OTLP
//! client and in the `PeriodicReader` that drives it, so at the default log
//! level a push that never arrives leaves no trace. [`ReportingMetricExporter`]
//! wraps the push exporter, counts each failed export, keeps the last error
//! text, and emits the rate-limited WARN [`OTLP_METRICS_EXPORT_FAILED_MSG`]:
//! the first failure at once, then at most once per [`DROP_WARN_INTERVAL`] —
//! the interval the trace side's `trace spans dropped` WARN uses — plus a
//! final line at exporter shutdown for any failure not yet reported.
//!
//! **Why the result passes through.** Unlike the trace side's
//! `ReleasingExporter`, the wrapper returns the inner result unchanged: there
//! is no per-export SDK ERROR to suppress, and an `Ok` in place of a failure
//! would make the provider's flush and shutdown report success falsely.
//!
//! The WARN is emitted on the `PeriodicReader`'s own thread, where the SDK
//! suppresses its own telemetry; the log layer does not honour that, so the
//! line still reaches the log.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::metrics::data::ResourceMetrics;
use opentelemetry_sdk::metrics::exporter::PushMetricExporter;
use opentelemetry_sdk::metrics::Temporality;

use super::trace_export::{Clock, DROP_WARN_INTERVAL};

/// Message of the rate-limited WARN that reports failed metrics pushes.
pub const OTLP_METRICS_EXPORT_FAILED_MSG: &str = "OTLP metrics export failed";

/// Failure accounting for one metrics push exporter, and the WARN it emits.
pub struct MetricExportReporter {
    failed: AtomicU64,
    state: Mutex<ReportState>,
    clock: Clock,
}

/// What the previous WARN already reported, and when.
#[derive(Default)]
struct ReportState {
    reported_failed: u64,
    last_warn: Option<Instant>,
    /// Text of the most recent failed export not yet carried by a WARN.
    last_export_error: Option<String>,
}

impl std::fmt::Debug for MetricExportReporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetricExportReporter")
            .field("failed_exports_total", &self.failed_exports_total())
            .finish_non_exhaustive()
    }
}

impl Default for MetricExportReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricExportReporter {
    /// A reporter on the real clock.
    pub fn new() -> Self {
        Self::with_clock(Arc::new(Instant::now))
    }

    /// [`Self::new`] with an injected clock for the WARN rate limit.
    pub fn with_clock(clock: Clock) -> Self {
        Self {
            failed: AtomicU64::new(0),
            state: Mutex::new(ReportState::default()),
            clock,
        }
    }

    /// Failed exports since start.
    pub fn failed_exports_total(&self) -> u64 {
        self.failed.load(Ordering::Relaxed)
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, ReportState> {
        // A panic while holding the lock leaves plain counters behind; they
        // are still consistent enough to report from.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn record_failure(&self, error: String) {
        self.failed.fetch_add(1, Ordering::Relaxed);
        self.lock_state().last_export_error = Some(error);
    }

    /// Emit [`OTLP_METRICS_EXPORT_FAILED_MSG`] if exports failed since the
    /// previous line and, unless `force`, at least [`DROP_WARN_INTERVAL`] has
    /// passed since it. The first line after start is never held back.
    fn report(&self, force: bool) {
        let now = (self.clock)();
        let (failed_exports, failed_exports_total, last_export_error) = {
            let mut state = self.lock_state();
            let total = self.failed_exports_total();
            let failed = total.saturating_sub(state.reported_failed);
            if failed == 0 {
                return;
            }
            if !force {
                if let Some(last) = state.last_warn {
                    if now.saturating_duration_since(last) < DROP_WARN_INTERVAL {
                        return;
                    }
                }
            }
            state.reported_failed = total;
            state.last_warn = Some(now);
            (failed, total, state.last_export_error.take())
        };
        tracing::warn!(
            failed_exports,
            failed_exports_total,
            last_export_error = last_export_error.as_deref(),
            "{}",
            OTLP_METRICS_EXPORT_FAILED_MSG
        );
    }
}

/// Push exporter that reports `inner`'s failed exports through a
/// [`MetricExportReporter`] and otherwise passes every call through.
///
/// Both shutdown entry points pass through (the OTLP exporter overrides
/// both) and then force a final WARN for any failure not yet reported.
#[derive(Debug)]
pub struct ReportingMetricExporter<E> {
    inner: E,
    reporter: Arc<MetricExportReporter>,
}

impl<E> ReportingMetricExporter<E> {
    pub fn new(inner: E, reporter: Arc<MetricExportReporter>) -> Self {
        Self { inner, reporter }
    }
}

impl<E: PushMetricExporter> PushMetricExporter for ReportingMetricExporter<E> {
    async fn export(&self, metrics: &ResourceMetrics) -> OTelSdkResult {
        let result = self.inner.export(metrics).await;
        if let Err(error) = &result {
            self.reporter.record_failure(error.to_string());
        }
        // After a success too, as on the trace side: failures held back by
        // the interval are then reported once it passes, even when the
        // collector has since recovered.
        self.reporter.report(false);
        result
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        let result = self.inner.shutdown_with_timeout(timeout);
        self.reporter.report(true);
        result
    }

    fn shutdown(&self) -> OTelSdkResult {
        let result = self.inner.shutdown();
        self.reporter.report(true);
        result
    }

    fn temporality(&self) -> Temporality {
        self.inner.temporality()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::MutexGuard;

    use futures::FutureExt as _;
    use opentelemetry_sdk::error::OTelSdkError;

    use super::*;
    use crate::observability::trace_export::test_support::{
        keep_a_second_dispatcher_registered, CapturingWriter,
    };

    /// Error text of every failed export the probe answers.
    const COLLECTOR_DOWN: &str = "collector unreachable";

    fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
        mutex.lock().unwrap_or_else(PoisonError::into_inner)
    }

    // ── Inner exporter double ────────────────────────────────────────────────

    /// Which inner shutdown entry point the wrapper called.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ShutdownCall {
        Shutdown,
        WithTimeout(Duration),
    }

    /// What a [`ProbeExporter`] answers and saw, shared with the test.
    #[derive(Debug, Default)]
    struct Probe {
        /// While `true`, every export fails with [`COLLECTOR_DOWN`].
        failing: Mutex<bool>,
        exports: AtomicU64,
        flushes: AtomicU64,
        shutdowns: Mutex<Vec<ShutdownCall>>,
    }

    /// The push exporter `ReportingMetricExporter` wraps in these tests. Its
    /// temporality is `Delta`, not the default, and its flush and shutdown
    /// answer distinct errors, so a wrapper answering anything of its own is
    /// told apart from one passing the inner answer through.
    #[derive(Clone, Debug)]
    struct ProbeExporter(Arc<Probe>);

    impl ProbeExporter {
        fn new(failing: bool) -> Self {
            let probe = Probe::default();
            *lock(&probe.failing) = failing;
            Self(Arc::new(probe))
        }

        fn set_failing(&self, failing: bool) {
            *lock(&self.0.failing) = failing;
        }

        fn shutdowns(&self) -> Vec<ShutdownCall> {
            lock(&self.0.shutdowns).clone()
        }
    }

    impl PushMetricExporter for ProbeExporter {
        async fn export(&self, _metrics: &ResourceMetrics) -> OTelSdkResult {
            self.0.exports.fetch_add(1, Ordering::SeqCst);
            if *lock(&self.0.failing) {
                Err(OTelSdkError::InternalFailure(COLLECTOR_DOWN.to_string()))
            } else {
                Ok(())
            }
        }

        fn force_flush(&self) -> OTelSdkResult {
            self.0.flushes.fetch_add(1, Ordering::SeqCst);
            Err(OTelSdkError::InternalFailure("probe flush".to_string()))
        }

        fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
            lock(&self.0.shutdowns).push(ShutdownCall::WithTimeout(timeout));
            Err(OTelSdkError::InternalFailure(
                "probe shutdown_with_timeout".to_string(),
            ))
        }

        fn shutdown(&self) -> OTelSdkResult {
            lock(&self.0.shutdowns).push(ShutdownCall::Shutdown);
            Err(OTelSdkError::InternalFailure("probe shutdown".to_string()))
        }

        fn temporality(&self) -> Temporality {
            Temporality::Delta
        }
    }

    // ── Harness ──────────────────────────────────────────────────────────────

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

    /// A wrapper around `probe` on a manual clock.
    fn reporting(
        probe: &ProbeExporter,
    ) -> (
        ManualClock,
        Arc<MetricExportReporter>,
        ReportingMetricExporter<ProbeExporter>,
    ) {
        let (clock, now) = ManualClock::start();
        let reporter = Arc::new(MetricExportReporter::with_clock(now));
        let exporter = ReportingMetricExporter::new(probe.clone(), Arc::clone(&reporter));
        (clock, reporter, exporter)
    }

    /// Export through `exporter` on the calling thread, completing on the
    /// first poll: the probe never suspends, so the WARN fires on this thread
    /// inside [`failure_lines`]'s capture. Polled once rather than run on an
    /// executor so no runtime bridge appears under `src/` (the
    /// `stream_consumer_test` source fence cannot tell test code from
    /// production code).
    fn export_now(exporter: &ReportingMetricExporter<ProbeExporter>) -> OTelSdkResult {
        exporter
            .export(&ResourceMetrics::default())
            .now_or_never()
            .expect("an export through the probe completes on its first poll")
    }

    /// Run `f` on this thread under a JSON capture and return the
    /// [`OTLP_METRICS_EXPORT_FAILED_MSG`] lines it emitted.
    fn failure_lines<T>(f: impl FnOnce() -> T) -> (T, Vec<serde_json::Value>) {
        keep_a_second_dispatcher_registered();
        let writer = CapturingWriter::default();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(writer.clone())
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .finish();
        let value = tracing::subscriber::with_default(subscriber, f);
        let lines = writer
            .text()
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("JSON line"))
            .filter(|event| event["fields"]["message"] == OTLP_METRICS_EXPORT_FAILED_MSG)
            .collect();
        (value, lines)
    }

    /// Assert one failure line's level, figures, and error text.
    fn assert_failure_line(line: &serde_json::Value, failed_exports: u64, total: u64) {
        assert_eq!(line["level"], "WARN", "{line}");
        let fields = &line["fields"];
        assert_eq!(fields["failed_exports"], failed_exports, "{line}");
        assert_eq!(fields["failed_exports_total"], total, "{line}");
        assert!(
            fields["last_export_error"]
                .as_str()
                .is_some_and(|error| error.contains(COLLECTOR_DOWN)),
            "the line must carry the last failed export's error text: {line}"
        );
    }

    // ── The WARN ─────────────────────────────────────────────────────────────

    #[test]
    fn otlp_metrics_export_failed_message_is_stable() {
        assert_eq!(OTLP_METRICS_EXPORT_FAILED_MSG, "OTLP metrics export failed");
    }

    /// The first failed export is logged at once, with its error text, and
    /// the failure still reaches the caller.
    #[test]
    fn reporting_exporter_first_failure_is_logged_at_once() {
        let probe = ProbeExporter::new(true);
        let (_clock, reporter, exporter) = reporting(&probe);

        let (result, lines) = failure_lines(|| export_now(&exporter));

        assert!(result.is_err(), "the failure passes through");
        assert_eq!(lines.len(), 1, "the first failure is logged: {lines:?}");
        assert_failure_line(&lines[0], 1, 1);
        assert_eq!(reporter.failed_exports_total(), 1);
    }

    /// Failures inside the interval are counted but not logged; once the
    /// interval has passed the next failure logs one line with the failures
    /// since the previous line and the running total.
    #[test]
    fn reporting_exporter_holds_failures_inside_the_interval_then_reports_deltas_and_totals() {
        let probe = ProbeExporter::new(true);
        let (clock, reporter, exporter) = reporting(&probe);

        let (_, first) = failure_lines(|| export_now(&exporter));
        assert_eq!(first.len(), 1, "precondition: the first failure is logged");

        clock.advance(Duration::from_secs(30));
        let (_, held) = failure_lines(|| {
            for _ in 0..3 {
                let _ = export_now(&exporter);
            }
        });
        assert!(
            held.is_empty(),
            "inside the interval nothing is logged: {held:?}"
        );
        assert_eq!(
            reporter.failed_exports_total(),
            4,
            "failures inside the interval are still counted"
        );

        clock.advance(Duration::from_secs(29));
        let (_, still_held) = failure_lines(|| export_now(&exporter));
        assert!(
            still_held.is_empty(),
            "59 s after the last line is still inside the interval: {still_held:?}"
        );

        clock.advance(Duration::from_secs(1));
        let (_, second) = failure_lines(|| export_now(&exporter));
        assert_eq!(
            second.len(),
            1,
            "once {DROP_WARN_INTERVAL:?} has passed, the held failures are reported: {second:?}"
        );
        assert_failure_line(&second[0], 5, 6);
    }

    /// Failures held back by the interval are reported by the first export
    /// after it passes, even when that export succeeds: a collector that
    /// recovered inside the interval does not hide the failures before it.
    /// Successes before the interval passes log nothing.
    #[test]
    fn reporting_exporter_reports_held_failures_after_the_interval_even_once_recovered() {
        let probe = ProbeExporter::new(true);
        let (clock, reporter, exporter) = reporting(&probe);

        let (_, first) = failure_lines(|| export_now(&exporter));
        assert_eq!(first.len(), 1, "precondition: the first failure is logged");

        clock.advance(Duration::from_secs(10));
        let (_, held) = failure_lines(|| export_now(&exporter));
        assert!(
            held.is_empty(),
            "precondition: the second failure is held: {held:?}"
        );

        probe.set_failing(false);
        let (result, recovered) = failure_lines(|| export_now(&exporter));
        assert!(result.is_ok(), "precondition: the collector recovered");
        assert!(
            recovered.is_empty(),
            "a success inside the interval logs nothing: {recovered:?}"
        );

        clock.advance(DROP_WARN_INTERVAL);
        let (result, late) = failure_lines(|| export_now(&exporter));
        assert!(result.is_ok());
        assert_eq!(
            late.len(),
            1,
            "the first export after the interval reports the held failure: {late:?}"
        );
        assert_failure_line(&late[0], 1, 2);

        let (_, nothing_new) = failure_lines(|| export_now(&exporter));
        assert!(
            nothing_new.is_empty(),
            "with no failure since the last line there is nothing to report: {nothing_new:?}"
        );
        assert_eq!(reporter.failed_exports_total(), 2);
    }

    /// With every export succeeding, neither an export nor either shutdown
    /// logs a line, however much time passes.
    #[test]
    fn reporting_exporter_success_alone_never_logs() {
        let probe = ProbeExporter::new(false);
        let (clock, reporter, exporter) = reporting(&probe);

        let (_, lines) = failure_lines(|| {
            export_now(&exporter).expect("the probe succeeds");
            clock.advance(DROP_WARN_INTERVAL * 2);
            export_now(&exporter).expect("the probe succeeds");
            let _ = exporter.shutdown_with_timeout(Duration::from_secs(1));
            let _ = exporter.shutdown();
        });

        assert!(lines.is_empty(), "no failure, no line: {lines:?}");
        assert_eq!(reporter.failed_exports_total(), 0);
    }

    /// Each shutdown entry point reaches the same inner entry point, answers
    /// its result, and then reports the failures the interval was holding
    /// back, without waiting for the interval.
    #[test]
    fn reporting_exporter_both_shutdowns_report_failures_held_inside_the_interval() {
        type Shutdown = fn(&ReportingMetricExporter<ProbeExporter>) -> OTelSdkResult;
        let cases: [(&str, Shutdown, ShutdownCall, &str); 2] = [
            (
                "shutdown",
                |exporter| exporter.shutdown(),
                ShutdownCall::Shutdown,
                "probe shutdown",
            ),
            (
                "shutdown_with_timeout",
                |exporter| exporter.shutdown_with_timeout(Duration::from_secs(7)),
                ShutdownCall::WithTimeout(Duration::from_secs(7)),
                "probe shutdown_with_timeout",
            ),
        ];
        for (name, shutdown, expected_call, expected_error) in cases {
            let probe = ProbeExporter::new(true);
            let (clock, _reporter, exporter) = reporting(&probe);
            let (_, first) = failure_lines(|| export_now(&exporter));
            assert_eq!(
                first.len(),
                1,
                "{name}: precondition: the first failure is logged"
            );
            clock.advance(Duration::from_secs(1));
            let (_, held) = failure_lines(|| export_now(&exporter));
            assert!(
                held.is_empty(),
                "{name}: precondition: the second failure is held"
            );

            let (result, last) = failure_lines(|| shutdown(&exporter));

            assert_eq!(
                last.len(),
                1,
                "{name} must report the failure the interval was holding: {last:?}"
            );
            assert_failure_line(&last[0], 1, 2);
            assert_eq!(probe.shutdowns(), vec![expected_call], "{name}");
            let error = result.expect_err("the inner shutdown's error passes through");
            assert!(
                error.to_string().contains(expected_error),
                "{name}: got {error}"
            );
        }
    }

    /// Temporality, flush and every export result are the inner exporter's.
    #[test]
    fn reporting_exporter_passes_temporality_flush_and_results_through() {
        let probe = ProbeExporter::new(false);
        let (_clock, _reporter, exporter) = reporting(&probe);

        assert_eq!(exporter.temporality(), Temporality::Delta);

        let flush = exporter.force_flush();
        assert_eq!(probe.0.flushes.load(Ordering::SeqCst), 1);
        assert!(
            flush.is_err_and(|error| error.to_string().contains("probe flush")),
            "the inner flush's answer passes through"
        );

        assert!(export_now(&exporter).is_ok(), "a success passes through");
        probe.set_failing(true);
        let (failed, _) = failure_lines(|| export_now(&exporter));
        assert!(
            failed.is_err_and(|error| error.to_string().contains(COLLECTOR_DOWN)),
            "a failure passes through with the inner error, not `Ok`"
        );
        assert_eq!(probe.0.exports.load(Ordering::SeqCst), 2);
    }
}
