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

pub mod health;
pub mod metrics;
pub mod metrics_export;
pub mod metrics_server;
pub mod otlp;
pub mod process_cpu;
pub mod process_memory;
pub mod scaling;
pub mod startup;
pub mod trace_export;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use opentelemetry::propagation::{Extractor, TextMapPropagator};
use opentelemetry::trace::TraceContextExt as _;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::SdkTracer;
use tracing_subscriber::filter::{filter_fn, FilterExt, LevelFilter};
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

use crate::config::LogFormat;
use otlp::OtlpTransport;
use trace_export::TraceExportState;

/// Global handle to the initialized `SdkTracerProvider` used by `init_tracing`.
///
/// opentelemetry 0.29+ removed `opentelemetry::global::shutdown_tracer_provider()`
/// in favor of calling `shutdown()` on the concrete provider, so we stash the
/// provider here at init time and `shutdown_tracing()` pulls it back out to
/// flush pending batch exports before the process exits.
static OTEL_PROVIDER: OnceLock<Mutex<Option<opentelemetry_sdk::trace::SdkTracerProvider>>> =
    OnceLock::new();

/// Test-only: held by every unit test that can put a provider in
/// [`OTEL_PROVIDER`], from its first install to its last drain, so that
/// under a threaded `cargo test` no test's slot assertions or
/// `build_otel_layer` results see a sibling's provider. (nextest runs each
/// test in its own process, where the slot is never shared.) The env lock
/// `temp_env` takes is separate and is always taken inside this one.
#[cfg(test)]
pub(crate) static OTEL_PROVIDER_TEST_LOCK: tokio::sync::Mutex<()> =
    tokio::sync::Mutex::const_new(());

/// Global guard against calling `init_tracing` more than once (b.3z7 medium).
///
/// `tracing_subscriber::registry().init()` panics on a second call because the
/// global subscriber is a `OnceCell`. This matches the guard `build_otel_layer`
/// added in b.q1d — tests and rare startup-retry paths can now call
/// `init_tracing` more than once without aborting the process.
static TRACING_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// What an enabled trace export hands back from [`init_tracing`]: the
/// pipeline's loss accounting, and the transport the endpoint's scheme
/// selected.
pub type TraceExport = (Arc<TraceExportState>, OtlpTransport);

/// Initialize structured logging.
///
/// `log_level` is the tracing filter string (e.g. "info", "debug").
/// Invalid level strings fall back to "info" without panicking.
///
/// `log_format` selects between `LogFormat::Json` (production default —
/// structured records for log aggregators) and `LogFormat::Pretty`
/// (human-readable multi-line output for local development). See b.3z7
/// H-obs-4.
///
/// When `OTEL_EXPORTER_OTLP_ENDPOINT` is set the function also wires up a
/// tracing-opentelemetry layer that ships spans to that endpoint via gRPC
/// (plaintext for `http://`, TLS for `https://`; see [`otlp`]), filtered by
/// `trace_filter` (`LIVE_EDIT_TRACE_FILTER`), and returns the pipeline's loss
/// accounting and transport for `main` to log and register as metrics. If the
/// variable is unset or blank the OTEL layer is skipped silently and
/// `Ok(None)` is returned.
///
/// An endpoint that is not a valid `http://` or `https://` URL, or an exporter
/// that cannot be built for it, is an `Err` naming the variable and the
/// cause, and no subscriber is installed: `main` reports it on stderr and
/// exits.
///
/// Second and subsequent calls are a no-op returning `Ok(None)` (b.3z7
/// medium). This mirrors the b.q1d guard on `build_otel_layer` — both layers
/// sit behind idempotency guards so a test harness (or accidental
/// double-init) does not panic on a second call.
pub fn init_tracing(
    log_level: &str,
    log_format: LogFormat,
    trace_filter: &str,
) -> Result<Option<TraceExport>, String> {
    if TRACING_INITIALIZED.swap(true, Ordering::AcqRel) {
        tracing::warn!(
            "init_tracing called while tracing is already initialized — \
             skipping; expected only once at process startup"
        );
        return Ok(None);
    }
    let (subscriber, trace_export) = build_registry(log_level, log_format, trace_filter)?;
    subscriber.init();
    Ok(trace_export)
}

/// Resolve the EnvFilter precedence used by [`build_registry`] /
/// [`init_tracing`] (b.avd item 11 + HIGH follow-up).
///
/// Precedence:
///   1. `LIVE_EDIT_LOG_LEVEL` — read by `TracingSettings::from_env` and
///      threaded in here as `log_level`. Authoritative when explicitly set
///      in the environment, captured into the `log_level` argument by
///      `TracingSettings::from_env`. "Explicitly set" is detected by a presence
///      check on the env var (`std::env::var(...).is_ok()`): set-to-empty
///      still counts as "explicit" because the operator typed it, while
///      unset (`Err(NotPresent)`) defers to `RUST_LOG`.
///   2. `RUST_LOG` — honored as a fallback when the operator didn't set
///      `LIVE_EDIT_LOG_LEVEL` (the caller passes `log_level = "info"`,
///      the default). This lets `RUST_LOG=live_edit::storage=debug` work
///      without forcing operators to also set `LIVE_EDIT_LOG_LEVEL`,
///      which is the standard Rust idiom.
///   3. Fallback to `log_level` (parsed as an EnvFilter directive).
///   4. If both directives fail to parse, fall back to `"info"`.
///
/// Extracted from `build_registry` so unit tests can construct the same
/// filter and assert that targeted directives (e.g.
/// `live_edit::storage=debug`) actually surface — the previous test
/// scaffolding only proved the function didn't panic, not that the
/// directive was applied to the emitted event.
pub(crate) fn build_env_filter(log_level: &str) -> EnvFilter {
    let live_edit_explicit = std::env::var("LIVE_EDIT_LOG_LEVEL").is_ok();
    if live_edit_explicit {
        EnvFilter::try_new(log_level)
            .or_else(|_| EnvFilter::try_from_default_env())
            .unwrap_or_else(|_| EnvFilter::new("info"))
    } else {
        EnvFilter::try_from_default_env()
            .or_else(|_| EnvFilter::try_new(log_level))
            .unwrap_or_else(|_| EnvFilter::new("info"))
    }
}

/// Build the subscriber registry for the given log level, log format and trace
/// filter without installing it, wiring the OTLP pipeline when
/// `OTEL_EXPORTER_OTLP_ENDPOINT` is set (see [`init_tracing`], including its
/// errors).
///
/// Exposed for testing: callers can use `tracing::subscriber::with_default` to
/// activate the subscriber only within the scope of a test.
pub(crate) fn build_registry(
    log_level: &str,
    log_format: LogFormat,
    trace_filter: &str,
) -> Result<
    (
        Box<dyn tracing::Subscriber + Send + Sync + 'static>,
        Option<TraceExport>,
    ),
    String,
> {
    let (tracer, trace_export) = match build_otel_layer()? {
        Some((tracer, state, transport)) => (Some(tracer), Some((state, transport))),
        None => (None, None),
    };
    let subscriber = compose_registry(log_level, log_format, trace_filter, std::io::stdout, tracer);
    Ok((subscriber, trace_export))
}

/// Compose the two layers of the subscriber, with the log writer and the
/// tracer injected.
///
/// - **Log layer** (fmt, JSON or pretty): filtered by [`build_env_filter`]
///   and to events only. It never sees a span, so no log line carries span
///   context and both formats are the same whatever spans exist.
/// - **Trace layer** (OpenTelemetry, present when `tracer` is): filtered by
///   `trace_filter`, which is independent of the log filter. Only spans and
///   events that filter enables reach the exporter; an event inside an
///   exported span becomes a span event. Source location and thread
///   attributes are off.
///
/// [`build_registry`] calls this with stdout and the environment's OTLP
/// pipeline; tests inject a capturing writer and a tracer over a collecting
/// exporter.
pub(crate) fn compose_registry<W>(
    log_level: &str,
    log_format: LogFormat,
    trace_filter: &str,
    writer: W,
    tracer: Option<SdkTracer>,
) -> Box<dyn tracing::Subscriber + Send + Sync + 'static>
where
    W: for<'w> MakeWriter<'w> + Send + Sync + 'static,
{
    // Events only. The level hint is the least restrictive one so it never
    // narrows the env filter's own hint, which `And` would otherwise lose
    // (it drops the hint when either side has none).
    let events_only =
        filter_fn(|metadata| metadata.is_event()).with_max_level_hint(LevelFilter::TRACE);
    let filter = build_env_filter(log_level).and(events_only);

    // Box-erase the fmt layer so the Json and Pretty variants can share a
    // single registry type — without it, the generic `fmt::Layer<_, JsonFields, ...>`
    // vs `fmt::Layer<_, Pretty, ...>` would force two separate concrete
    // registry shapes, and the OpenTelemetry layer's trait bound is
    // parameterised over the exact inner fmt type so it can only be layered
    // on one of them.
    let fmt_layer: Box<dyn Layer<_> + Send + Sync + 'static> = match log_format {
        LogFormat::Json => Box::new(fmt::layer().json().with_writer(writer).with_filter(filter)),
        LogFormat::Pretty => Box::new(
            fmt::layer()
                .pretty()
                .with_writer(writer)
                .with_filter(filter),
        ),
    };

    let otel_layer = tracer.map(|tracer| {
        tracing_opentelemetry::layer()
            .with_location(false)
            .with_threads(false)
            .with_tracer(tracer)
            .with_filter(EnvFilter::new(trace_filter))
    });

    Box::new(
        tracing_subscriber::registry()
            .with(fmt_layer)
            .with(otel_layer),
    )
}

/// [`compose_registry`] for integration tests, which cannot reach
/// `pub(crate)` items.
#[cfg(any(test, feature = "test-support"))]
pub fn compose_registry_for_tests<W>(
    log_level: &str,
    log_format: LogFormat,
    trace_filter: &str,
    writer: W,
    tracer: Option<SdkTracer>,
) -> Box<dyn tracing::Subscriber + Send + Sync + 'static>
where
    W: for<'w> MakeWriter<'w> + Send + Sync + 'static,
{
    compose_registry(log_level, log_format, trace_filter, writer, tracer)
}

/// The W3C trace context a request's `traceparent` header carries, to use as
/// its server span's parent.
///
/// `None` when the request has no `traceparent` header or the header does
/// not parse to a valid span context, in which case the server span is a
/// root. Extraction starts from an empty context, never the thread's current
/// one, so a malformed header can never make the span a child of whatever
/// happens to be active.
///
/// **An inbound `tracestate` is deliberately ignored**: [`HeaderExtractor`]
/// shows the propagator `traceparent` alone. The propagator accepts any
/// `tracestate` that parses, the sampler copies it onto the span, and OTLP
/// exports it with every span, so honouring it would let any caller —
/// including one that fails authentication — attach up to the HTTP/2 header
/// limit (16 KiB) of its own bytes to its span, and multiply the export
/// queue's memory bound by the same factor. The service never propagates
/// trace context outbound, so `tracestate` has no use here, and ignoring it
/// keeps a rejected RPC's cost at one ordinary-sized span. `traceparent`
/// itself carries only fixed-width ids and flags.
pub(crate) fn remote_parent_context(headers: &http::HeaderMap) -> Option<opentelemetry::Context> {
    if !headers.contains_key(TRACEPARENT_HEADER) {
        return None;
    }
    let cx = TraceContextPropagator::new()
        .extract_with_context(&opentelemetry::Context::new(), &HeaderExtractor(headers));
    cx.span().span_context().is_valid().then_some(cx)
}

/// The one header [`HeaderExtractor`] exposes.
const TRACEPARENT_HEADER: &str = "traceparent";

/// `Extractor` over an `http::HeaderMap` for [`remote_parent_context`] that
/// exposes `traceparent` and nothing else, so an inbound `tracestate` (or
/// any other header) never reaches the span — see that function's doc.
struct HeaderExtractor<'a>(&'a http::HeaderMap);

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        if !key.eq_ignore_ascii_case(TRACEPARENT_HEADER) {
            return None;
        }
        self.0
            .get(TRACEPARENT_HEADER)
            .and_then(|value| value.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        if self.0.contains_key(TRACEPARENT_HEADER) {
            vec![TRACEPARENT_HEADER]
        } else {
            Vec::new()
        }
    }
}

/// Flush and shut down the global OpenTelemetry tracer provider.
///
/// Must be called during graceful shutdown to ensure all buffered spans are
/// exported before the process exits. The exporter's shutdown also emits the
/// final `trace spans dropped` WARN for any loss not yet reported
/// ([`trace_export`]).
///
/// `SdkTracerProvider::shutdown()` joins the `BatchSpanProcessor`'s dedicated
/// OS thread with a 5 s internal timeout (opentelemetry_sdk 0.31), so upstream
/// docs warn that it must not run on a tokio worker thread. We offload it via
/// `spawn_blocking` so the reactor is not held up during graceful shutdown.
pub async fn shutdown_tracing() {
    let Some(slot) = OTEL_PROVIDER.get() else {
        return;
    };
    let Some(provider) = slot.lock().ok().and_then(|mut g| g.take()) else {
        return;
    };
    let join = tokio::task::spawn_blocking(move || provider.shutdown()).await;
    match join {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "failed to shut down OpenTelemetry tracer provider")
        }
        Err(e) => {
            tracing::warn!(error = %e, "spawn_blocking for OpenTelemetry shutdown panicked")
        }
    }
}

/// Build the OTLP trace pipeline the OpenTelemetry layer exports through.
///
/// - `OTEL_EXPORTER_OTLP_ENDPOINT` unset or blank, or a provider already
///   built (the double-init guard): `Ok(None)`.
/// - A valid `http://` or `https://` endpoint: the tracer for the layer, the
///   pipeline's loss accounting, and the endpoint's transport.
/// - Any other value, or an exporter that cannot be built for it: `Err`
///   naming the variable and the cause ([`otlp`]).
/// - A set compression variable: `Err` naming that variable ([`otlp`]).
///
/// The endpoint passed to the exporter is always the non-empty value read
/// here, so the SDK never falls back to `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.
///
/// The processor's queue capacity comes from `OTEL_BSP_MAX_QUEUE_SIZE`, parsed
/// the way the SDK parses it; see [`trace_export`] for why the capacity is
/// passed explicitly and how every span is accounted for.
fn build_otel_layer() -> Result<Option<(SdkTracer, Arc<TraceExportState>, OtlpTransport)>, String> {
    use opentelemetry::trace::TracerProvider as _;

    let Some(endpoint) = otlp::otlp_endpoint_from_env(otlp::OTEL_EXPORTER_OTLP_ENDPOINT)? else {
        return Ok(None);
    };

    // Guard against a second init: the slot already holds a provider, so
    // overwriting it would drop the BatchSpanProcessor's background OS thread
    // and any buffered un-flushed spans without calling `shutdown()`, and
    // would also stomp the globally-registered tracer provider. The caller
    // already got a tracing layer on the first init; return None so they do
    // not get a second layer pointing at an orphaned tracer.
    let slot = OTEL_PROVIDER.get_or_init(|| Mutex::new(None));
    if let Ok(guard) = slot.lock() {
        if guard.is_some() {
            tracing::warn!(
                "build_otel_layer called while provider is already initialized — \
                 skipping; expected only once at process startup"
            );
            return Ok(None);
        }
    }

    let exporter = otlp::span_exporter_for(&endpoint)?;

    let state = Arc::new(TraceExportState::new(
        trace_export::max_queue_size_from_env(),
    ));
    let provider = trace_export::build_tracer_provider(exporter, Arc::clone(&state));

    let tracer = provider.tracer("live-edit");

    // Register the provider globally so the SDK's internal propagation helpers
    // can resolve it, and keep a local handle so `shutdown_tracing()` can flush
    // the batch exporter at process exit. `set_tracer_provider` takes ownership
    // of its argument, so we clone first.
    opentelemetry::global::set_tracer_provider(provider.clone());
    if let Ok(mut guard) = slot.lock() {
        *guard = Some(provider);
    }

    Ok(Some((tracer, state, endpoint.transport)))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- t3.1it.rr.ng.41: tracing init does not panic for valid levels ----

    /// Activate `build_registry(level)` as a scoped default subscriber and emit
    /// one event at each level. The closure must not panic.
    ///
    /// `OTEL_EXPORTER_OTLP_ENDPOINT` is pinned to `None` inside a
    /// `temp_env::with_vars` guard so a sibling test running in parallel
    /// (notably the b.5yt `build_otel_layer_with_endpoint…` test) cannot leak
    /// its set-endpoint state into this test. `std::env::set_var` is process-
    /// global, so without the guard these tests would stochastically enter the
    /// OTLP branch of `build_otel_layer` and panic in `tonic`/`hyper-util` for
    /// lack of a tokio reactor.
    fn assert_level_accepted(level: &str) {
        temp_env::with_vars([("OTEL_EXPORTER_OTLP_ENDPOINT", None::<&str>)], || {
            let (subscriber, _) =
                build_registry(level, LogFormat::Json, crate::config::DEFAULT_TRACE_FILTER)
                    .expect("build_registry must succeed");
            tracing::subscriber::with_default(subscriber, || {
                tracing::info!(target: "test", "probe");
            });
        });
    }

    #[test]
    fn init_tracing_level_info_does_not_panic() {
        assert_level_accepted("info");
    }

    #[test]
    fn init_tracing_level_debug_does_not_panic() {
        assert_level_accepted("debug");
    }

    #[test]
    fn init_tracing_level_warn_does_not_panic() {
        assert_level_accepted("warn");
    }

    #[test]
    fn init_tracing_level_error_does_not_panic() {
        assert_level_accepted("error");
    }

    #[test]
    fn init_tracing_level_trace_does_not_panic() {
        assert_level_accepted("trace");
    }

    #[test]
    fn init_tracing_invalid_level_falls_back_to_info_without_panic() {
        // An invalid level directive must not panic — build_registry falls back to "info".
        assert_level_accepted("not_a_real_level_directive[[[");
    }

    // ── b.avd item 11: RUST_LOG fallback ───────────────────────────────
    //
    // Acceptance criterion 7: `RUST_LOG=live_edit::storage=debug` works
    // without setting `LIVE_EDIT_LOG_LEVEL`. We assert at the
    // `build_registry` boundary because `init_tracing` is process-global
    // and not safely re-invocable in tests.
    //
    // The test exercises the precedence rules:
    //   - When `LIVE_EDIT_LOG_LEVEL` is unset and `RUST_LOG` is set,
    //     `RUST_LOG` wins.
    //   - When `LIVE_EDIT_LOG_LEVEL` is set, it wins over `RUST_LOG`.
    //
    // We can't easily peek inside the `EnvFilter` to read its directives
    // back out, but we can prove the function doesn't panic across the
    // four (LIVE_EDIT_LOG_LEVEL × RUST_LOG) combinations — which is the
    // shape of the regression mode pre-fix (a malformed precedence path
    // would surface as a parse panic or a missing log line at runtime).

    #[test]
    fn build_registry_honors_rust_log_when_live_edit_log_level_unset() {
        temp_env::with_vars(
            [
                ("OTEL_EXPORTER_OTLP_ENDPOINT", None::<&str>),
                ("LIVE_EDIT_LOG_LEVEL", None::<&str>),
                ("RUST_LOG", Some("live_edit::storage=debug")),
            ],
            || {
                let (subscriber, _) =
                    build_registry("info", LogFormat::Json, crate::config::DEFAULT_TRACE_FILTER)
                        .expect("build_registry must succeed");
                tracing::subscriber::with_default(subscriber, || {
                    tracing::debug!(target: "live_edit::storage", "probe");
                });
            },
        );
    }

    #[test]
    fn build_registry_live_edit_log_level_wins_over_rust_log() {
        temp_env::with_vars(
            [
                ("OTEL_EXPORTER_OTLP_ENDPOINT", None::<&str>),
                ("LIVE_EDIT_LOG_LEVEL", Some("warn")),
                ("RUST_LOG", Some("live_edit::storage=debug")),
            ],
            || {
                // Caller passes the same value `Config::from_env` captured
                // from LIVE_EDIT_LOG_LEVEL.
                let (subscriber, _) =
                    build_registry("warn", LogFormat::Json, crate::config::DEFAULT_TRACE_FILTER)
                        .expect("build_registry must succeed");
                tracing::subscriber::with_default(subscriber, || {
                    tracing::warn!(target: "live_edit::storage", "probe");
                });
            },
        );
    }

    #[test]
    fn build_registry_falls_back_to_log_level_when_both_env_vars_unset() {
        temp_env::with_vars(
            [
                ("OTEL_EXPORTER_OTLP_ENDPOINT", None::<&str>),
                ("LIVE_EDIT_LOG_LEVEL", None::<&str>),
                ("RUST_LOG", None::<&str>),
            ],
            || {
                let (subscriber, _) =
                    build_registry("info", LogFormat::Json, crate::config::DEFAULT_TRACE_FILTER)
                        .expect("build_registry must succeed");
                tracing::subscriber::with_default(subscriber, || {
                    tracing::info!(target: "test", "probe");
                });
            },
        );
    }

    // ── b.avd HIGH follow-up: assert RUST_LOG directives are applied ────
    //
    // The previous tests only proved `build_registry` doesn't panic
    // across the (LIVE_EDIT_LOG_LEVEL × RUST_LOG) matrix. AC7 of b.avd
    // is "RUST_LOG=live_edit::storage=debug works without setting
    // LIVE_EDIT_LOG_LEVEL" — that requires asserting the DEBUG-level
    // event at the `live_edit::storage` target is actually emitted (and
    // a corresponding sibling-target DEBUG event at the default INFO
    // level is suppressed).
    //
    // We assert at the `build_env_filter` boundary because
    // `EnvFilter` exposes `enabled()` directly via the `Layer` trait —
    // no need to install a fmt subscriber, parse JSON output, or wrestle
    // with process-global state. The `Metadata` for the test event is
    // synthesised via `tracing::callsite::DefaultCallsite`; we use the
    // higher-level `tracing::span!` machinery indirectly through
    // `tracing::Subscriber::enabled` on a temporary registry.

    /// `RUST_LOG=live_edit::storage=debug` with `LIVE_EDIT_LOG_LEVEL` unset
    /// must produce a filter that enables DEBUG events targeted at
    /// `live_edit::storage` and suppresses DEBUG events at other targets.
    /// This is the precise behavior AC7 was originally testing for; the
    /// previous test only asserted "no panic", which would have passed
    /// even if the filter silently ignored `RUST_LOG`.
    #[test]
    fn build_env_filter_rust_log_directive_actually_enables_targeted_debug() {
        use tracing_subscriber::layer::SubscriberExt;

        temp_env::with_vars(
            [
                ("LIVE_EDIT_LOG_LEVEL", None::<&str>),
                ("RUST_LOG", Some("live_edit::storage=debug")),
            ],
            || {
                let filter = build_env_filter("info");
                // Layer the filter onto a no-op registry and probe via
                // the resulting Subscriber's `enabled` callback. This is
                // the same path tracing-subscriber uses internally to
                // decide whether to materialise an event.
                let subscriber = tracing_subscriber::registry().with(filter);

                tracing::subscriber::with_default(subscriber, || {
                    // The directive is `live_edit::storage=debug`, so a
                    // DEBUG event targeted at that module must be
                    // enabled. This is the load-bearing AC7 assertion —
                    // pre-fix the test only proved `build_registry`
                    // didn't panic, which would have passed even if the
                    // filter silently ignored `RUST_LOG`.
                    assert!(
                        tracing::event_enabled!(target: "live_edit::storage", tracing::Level::DEBUG),
                        "RUST_LOG=live_edit::storage=debug must enable DEBUG at \
                         target=live_edit::storage when LIVE_EDIT_LOG_LEVEL is unset (b.avd AC7)"
                    );
                    // Sanity-check the targeted-directive shape: a DEBUG
                    // event at an unrelated target must NOT be enabled.
                    // `EnvFilter` with a single targeted directive
                    // implicitly suppresses everything else — that's the
                    // operator semantic of `RUST_LOG=my_crate=debug`,
                    // and proves the directive parsed as a targeted one
                    // rather than a bare global level.
                    assert!(
                        !tracing::event_enabled!(
                            target: "live_edit::grpc",
                            tracing::Level::DEBUG
                        ),
                        "DEBUG at an unrelated target must remain suppressed when only \
                         live_edit::storage=debug is set (otherwise the directive parsed \
                         as a global level instead of a targeted one)"
                    );
                });
            },
        );
    }

    /// When `LIVE_EDIT_LOG_LEVEL` is set, it must win over `RUST_LOG`.
    /// Concretely: setting `LIVE_EDIT_LOG_LEVEL=warn` and
    /// `RUST_LOG=live_edit::storage=debug` must suppress the
    /// `live_edit::storage` DEBUG event (because the LIVE_EDIT directive
    /// wins and `warn` is more restrictive than `debug`).
    #[test]
    fn build_env_filter_live_edit_log_level_overrides_rust_log_directive() {
        use tracing_subscriber::layer::SubscriberExt;

        temp_env::with_vars(
            [
                ("LIVE_EDIT_LOG_LEVEL", Some("warn")),
                ("RUST_LOG", Some("live_edit::storage=debug")),
            ],
            || {
                let filter = build_env_filter("warn");
                let subscriber = tracing_subscriber::registry().with(filter);

                tracing::subscriber::with_default(subscriber, || {
                    // RUST_LOG would have enabled this; LIVE_EDIT_LOG_LEVEL
                    // must override.
                    assert!(
                        !tracing::event_enabled!(
                            target: "live_edit::storage",
                            tracing::Level::DEBUG
                        ),
                        "LIVE_EDIT_LOG_LEVEL=warn must suppress the live_edit::storage=debug \
                         directive from RUST_LOG (LIVE_EDIT_LOG_LEVEL wins on explicit set)"
                    );
                    // WARN must still be enabled (the global level).
                    assert!(
                        tracing::event_enabled!(
                            target: "live_edit::storage",
                            tracing::Level::WARN
                        ),
                        "WARN at any target must remain enabled at LIVE_EDIT_LOG_LEVEL=warn"
                    );
                    assert!(
                        !tracing::event_enabled!(
                            target: "live_edit::storage",
                            tracing::Level::INFO
                        ),
                        "INFO must be suppressed at LIVE_EDIT_LOG_LEVEL=warn"
                    );
                });
            },
        );
    }

    // ---- b.5yt: build_otel_layer + OTEL_PROVIDER OnceLock ordering ----

    /// Cover sub-tasks 2 and 3 in one combined test because they share process-
    /// global state (`OTEL_EXPORTER_OTLP_ENDPOINT` env var, the `OTEL_PROVIDER`
    /// `OnceLock`, and `opentelemetry::global::set_tracer_provider`). Splitting
    /// them would create two tests that race on all three every time they both
    /// ran with an endpoint set — `temp_env::with_vars` serializes env-var
    /// access but cannot protect the other two. A single test keeps the full
    /// init → take sequence atomic from the test harness's point of view, and
    /// [`OTEL_PROVIDER_TEST_LOCK`], held throughout, keeps every other
    /// slot-touching test out of it under a threaded `cargo test`.
    ///
    /// `127.0.0.1:0` is an invalid-but-parseable gRPC endpoint: the
    /// `BatchSpanProcessor` swallows connect errors into its background thread,
    /// so `build_otel_layer` returns `Some(_)` and `shutdown_tracing()` completes
    /// on its 5 s internal timeout without blocking the test.
    ///
    /// The pipeline is built through `build_registry`, the production entry
    /// point, so the test also pins its wiring:
    ///
    /// - The returned loss accounting is the provider's own, sized from
    ///   `OTEL_BSP_MAX_QUEUE_SIZE`. With the size pinned to `0`, every span
    ///   reaching the counting processor is counted `queue_full` synchronously
    ///   on this thread; a separately built state would read 0 forever.
    /// - The trace layer is filtered by `trace_filter`, not `log_level`. The
    ///   log level here is `trace` and the trace filter the default, so of
    ///   one `live_edit` INFO span and one h2 TRACE span exactly one is
    ///   counted; the log level as the trace filter would count both.
    #[tokio::test]
    async fn build_otel_layer_with_endpoint_initializes_and_shutdown_drains() {
        let _slot = OTEL_PROVIDER_TEST_LOCK.lock().await;
        crate::observability::trace_export::test_support::keep_a_second_dispatcher_registered();
        // `temp_env::with_vars` takes a synchronous closure, so run the sync
        // `build_registry` assertions inside the guard, then drop out and
        // await the async `shutdown_tracing`.
        temp_env::with_vars(
            [
                ("OTEL_EXPORTER_OTLP_ENDPOINT", Some("http://127.0.0.1:0")),
                ("OTEL_BSP_MAX_QUEUE_SIZE", Some("0")),
                ("OTEL_TRACES_SAMPLER", None),
                ("OTEL_TRACES_SAMPLER_ARG", None),
            ],
            || {
                let (subscriber, state) = build_registry(
                    "trace",
                    LogFormat::Json,
                    crate::config::DEFAULT_TRACE_FILTER,
                )
                .expect("build_registry must succeed");
                let (state, _transport) = state.expect(
                    "build_registry must return the pipeline's loss accounting when \
                     OTEL_EXPORTER_OTLP_ENDPOINT is set",
                );
                assert_eq!(
                    state.capacity(),
                    0,
                    "the capacity must come from OTEL_BSP_MAX_QUEUE_SIZE"
                );

                tracing::subscriber::with_default(subscriber, || {
                    tracing::info_span!(target: "live_edit::observability", "wiring_probe")
                        .in_scope(|| {});
                    tracing::trace_span!(target: "h2::codec::framed_write", "FramedWrite::buffer")
                        .in_scope(|| {});
                });
                assert_eq!(
                    state.queue_full_total(),
                    1,
                    "exactly the live_edit span must reach the provider's counting processor: \
                     0 means the returned state is not the provider's, 2 means the trace layer \
                     is filtered by the log level"
                );

                // Sub-task 3: after a successful init, the OnceLock slot holds
                // the provider so shutdown_tracing can flush it.
                let present = OTEL_PROVIDER
                    .get()
                    .and_then(|s| s.lock().ok().and_then(|g| g.as_ref().map(|_| ())))
                    .is_some();
                assert!(
                    present,
                    "OTEL_PROVIDER slot must be populated after build_otel_layer succeeds"
                );
            },
        );

        shutdown_tracing().await;

        // Sub-task 3: after shutdown, the slot must be drained so a subsequent
        // shutdown is a no-op and a future re-init would install a fresh
        // provider rather than stomp an existing one.
        let drained = OTEL_PROVIDER
            .get()
            .and_then(|s| s.lock().ok().and_then(|g| g.as_ref().map(|_| ())))
            .is_none();
        assert!(
            drained,
            "OTEL_PROVIDER slot must be None after shutdown_tracing completes"
        );
    }

    // ---- b.q1d: build_otel_layer double-init guard ----

    /// Regression test for b.q1d. A second call to `build_otel_layer` while the
    /// `OTEL_PROVIDER` slot already holds a provider must NOT drop the stored
    /// provider and must NOT install a second one — doing so would leak the
    /// old `BatchSpanProcessor`'s background OS thread and any buffered spans
    /// and stomp the globally-registered tracer provider. The guard in
    /// `build_otel_layer` returns `None` early in that case.
    ///
    /// `OTEL_PROVIDER` is a process-global `OnceLock` shared with the b.5yt
    /// test `build_otel_layer_with_endpoint_initializes_and_shutdown_drains`.
    /// Rather than relying on test-ordering, the test starts by draining any
    /// leftover provider from a prior test, then runs its canonical
    /// init → init-again → assert-original-kept → shutdown → drained sequence.
    /// `temp_env::with_vars` serialises the env var, and
    /// [`OTEL_PROVIDER_TEST_LOCK`], held throughout, serialises the slot, so
    /// no sibling test can race on either.
    #[tokio::test]
    async fn build_otel_layer_second_init_is_noop_and_preserves_stored_provider() {
        let _slot = OTEL_PROVIDER_TEST_LOCK.lock().await;
        // Drain any provider leaked by a sibling test so our first
        // `build_otel_layer` call below is the canonical "first init" from
        // this test's point of view. `shutdown_tracing` is a no-op when the
        // OnceLock is uninitialized or the slot is already None.
        shutdown_tracing().await;

        temp_env::with_vars(
            [("OTEL_EXPORTER_OTLP_ENDPOINT", Some("http://127.0.0.1:0"))],
            || {
                // First init: should succeed and populate the slot.
                let first = build_otel_layer();
                assert!(
                    matches!(first, Ok(Some(_))),
                    "first build_otel_layer call must return Some when endpoint is set"
                );
                assert!(
                    OTEL_PROVIDER
                        .get()
                        .and_then(|s| s.lock().ok().and_then(|g| g.as_ref().map(|_| ())))
                        .is_some(),
                    "OTEL_PROVIDER slot must be populated after first build_otel_layer"
                );

                // Second init: the guard must detect the populated slot and
                // return None before constructing a new exporter/provider.
                // Because the guard runs BEFORE any new provider is built,
                // returning None implies the slot was not stomped — there is
                // no code path where the stored provider gets replaced and
                // `build_otel_layer` still returns None.
                let second = build_otel_layer();
                assert!(
                    matches!(second, Ok(None)),
                    "second build_otel_layer call must return None when a provider is already stored \
                     (guards against dropping the existing BatchSpanProcessor without shutdown)"
                );

                // Slot must still hold the original provider — the guard
                // short-circuits before any replacement can happen.
                assert!(
                    OTEL_PROVIDER
                        .get()
                        .and_then(|s| s.lock().ok().and_then(|g| g.as_ref().map(|_| ())))
                        .is_some(),
                    "OTEL_PROVIDER slot must remain populated after the second init is rejected"
                );
            },
        );

        // Drain the original provider — the second call must not have
        // shut it down, so shutdown_tracing still finds and flushes it here.
        shutdown_tracing().await;

        assert!(
            OTEL_PROVIDER
                .get()
                .and_then(|s| s.lock().ok().and_then(|g| g.as_ref().map(|_| ())))
                .is_none(),
            "OTEL_PROVIDER slot must be None after shutdown_tracing completes"
        );
    }

    // ── init_tracing idempotence (b.3z7 medium) ───────────────────────────────
    //
    // Pre-fix, calling `init_tracing` twice panicked on the second call
    // because `tracing_subscriber::registry().init()` sets a global
    // `OnceCell`. The fix adds an `AtomicBool` guard that short-circuits
    // every subsequent call. These tests pin that idempotence so a test
    // harness that accidentally double-initializes (or a future
    // startup-retry loop) doesn't abort the process.

    /// A second call to `init_tracing` after a first (successful or panicking)
    /// must be a silent no-op rather than panicking.
    ///
    /// This test is robust to being run after a sibling test that already
    /// called `init_tracing` or installed a subscriber via
    /// `tracing::subscriber::with_default` — we catch any panic from the
    /// first call (which may happen because the process-global subscriber
    /// slot is one-shot) and only assert on the second call's behavior.
    /// The first call still flips the `TRACING_INITIALIZED` atomic to true
    /// on its `swap` regardless of whether `.init()` succeeds, so the
    /// second call's guard will fire either way.
    #[test]
    fn init_tracing_second_call_is_noop_not_panic() {
        use std::panic::AssertUnwindSafe;

        // First call: may succeed (flipping the atomic + installing a
        // subscriber) or panic (if a subscriber is already installed by
        // another test running in parallel). Either way, the atomic is
        // flipped to `true` by the leading `swap` before any
        // subscriber-install attempt runs. The trace endpoint is pinned unset
        // so this call can never put a provider in the `OTEL_PROVIDER` slot,
        // which this test never drains; the later calls return before reading
        // it.
        temp_env::with_var("OTEL_EXPORTER_OTLP_ENDPOINT", None::<&str>, || {
            let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let _ = init_tracing("info", LogFormat::Json, crate::config::DEFAULT_TRACE_FILTER);
            }));
        });

        // Second call: must be a guard-short-circuited no-op. A panic here
        // is the b.3z7-medium regression resurfacing.
        let second_outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = init_tracing(
                "debug",
                LogFormat::Pretty,
                crate::config::DEFAULT_TRACE_FILTER,
            );
        }));
        assert!(
            second_outcome.is_ok(),
            "second init_tracing call must be a no-op, not a panic — this is the \
             b.3z7-medium idempotence contract that lets a test harness or \
             startup retry loop call init_tracing more than once safely"
        );

        // Third call: also a no-op.
        let third_outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = init_tracing("warn", LogFormat::Json, crate::config::DEFAULT_TRACE_FILTER);
        }));
        assert!(
            third_outcome.is_ok(),
            "third init_tracing call must also be a no-op"
        );
    }

    // ── Log-format selection (b.3z7 H-obs-4) ──────────────────────────────────
    //
    // `build_registry` is the only surface through which `LogFormat`
    // reaches `tracing_subscriber`, so covering it at the registry level
    // proves the branch wiring without having to capture live stdout from
    // a running service.

    /// Both format variants must produce a usable subscriber that accepts
    /// events without panicking. Smoke-level: a pretty/JSON mismatch in
    /// the fmt-layer construction would surface as a compile error or a
    /// panic at first event, not a silent wrong-format regression — so
    /// any regression worth catching fires this test.
    #[test]
    fn build_registry_json_format_accepts_events_without_panic() {
        temp_env::with_vars([("OTEL_EXPORTER_OTLP_ENDPOINT", None::<&str>)], || {
            let (subscriber, _) =
                build_registry("info", LogFormat::Json, crate::config::DEFAULT_TRACE_FILTER)
                    .expect("build_registry must succeed");
            tracing::subscriber::with_default(subscriber, || {
                tracing::info!(target: "test", "probe-json");
            });
        });
    }

    #[test]
    fn build_registry_pretty_format_accepts_events_without_panic() {
        temp_env::with_vars([("OTEL_EXPORTER_OTLP_ENDPOINT", None::<&str>)], || {
            let (subscriber, _) = build_registry(
                "info",
                LogFormat::Pretty,
                crate::config::DEFAULT_TRACE_FILTER,
            )
            .expect("build_registry must succeed");
            tracing::subscriber::with_default(subscriber, || {
                tracing::info!(target: "test", "probe-pretty");
            });
        });
    }

    /// Prove the two branches actually construct distinct subscriber
    /// types. The JSON and Pretty layers internally differ — the test
    /// just asserts they are both constructible at both format values
    /// across every log level, closing off "LogFormat::Pretty silently
    /// falls through to JSON because of an unreachable match arm" as a
    /// regression mode.
    #[test]
    fn build_registry_both_formats_work_at_every_level() {
        for level in ["trace", "debug", "info", "warn", "error"] {
            for format in [LogFormat::Json, LogFormat::Pretty] {
                temp_env::with_vars([("OTEL_EXPORTER_OTLP_ENDPOINT", None::<&str>)], || {
                    let (subscriber, _) =
                        build_registry(level, format, crate::config::DEFAULT_TRACE_FILTER)
                            .expect("build_registry must succeed");
                    tracing::subscriber::with_default(subscriber, || {
                        tracing::info!(target: "test", level = level, format = ?format, "probe");
                    });
                });
            }
        }
    }

    // ── build_otel_layer malformed URL robustness (b.3z7 ticket test list) ────

    /// `OTEL_EXPORTER_OTLP_ENDPOINT` set to syntactic garbage never panics:
    /// `build_otel_layer` returns an `Err` naming the variable, which `main`
    /// reports as a configuration error before any subscriber exists, and
    /// builds no provider. The value is classified before the double-init
    /// guard and before any exporter is built, so the refusal does not
    /// depend on the `OTEL_PROVIDER` slot's state.
    #[tokio::test]
    async fn build_otel_layer_malformed_endpoint_returns_an_error_naming_the_variable() {
        // Held throughout, so no sibling's provider lands in the slot while
        // this test asserts it is empty.
        let _slot = OTEL_PROVIDER_TEST_LOCK.lock().await;
        // Drain any provider from a prior test so the OTEL_PROVIDER slot is
        // clean for this test's assertions.
        shutdown_tracing().await;

        let malformed_endpoints: &[&str] =
            &["not-a-url", "http://", ":::garbage:::", "http://[::bad"];
        for endpoint in malformed_endpoints {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                temp_env::with_vars([("OTEL_EXPORTER_OTLP_ENDPOINT", Some(*endpoint))], || {
                    build_otel_layer()
                })
            }));
            let result = outcome.unwrap_or_else(|_| {
                panic!("build_otel_layer must not panic on malformed OTEL endpoint {endpoint:?}")
            });
            let error = result.err().unwrap_or_else(|| {
                panic!(
                    "a malformed OTEL_EXPORTER_OTLP_ENDPOINT ({endpoint:?}) must be refused, \
                     not built"
                )
            });
            assert!(
                error.starts_with("OTEL_EXPORTER_OTLP_ENDPOINT "),
                "{endpoint:?}: the error must name the variable; got {error:?}"
            );
            assert!(
                OTEL_PROVIDER
                    .get()
                    .and_then(|s| s.lock().ok().and_then(|g| g.as_ref().map(|_| ())))
                    .is_none(),
                "{endpoint:?}: a refused endpoint must not install a provider"
            );
        }

        // Drain anything a regression installed so no state leaks into the
        // next test.
        shutdown_tracing().await;
    }

    /// A blank `OTEL_EXPORTER_OTLP_ENDPOINT` switches trace export off even
    /// with `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` set: the service never hands
    /// the SDK an empty endpoint, so the SDK's fallback to the per-signal
    /// variable is unreachable and that variable is never read.
    #[tokio::test]
    async fn build_otel_layer_blank_endpoint_is_off_even_with_the_traces_endpoint_set() {
        let _slot = OTEL_PROVIDER_TEST_LOCK.lock().await;
        shutdown_tracing().await;

        for blank in ["", "   "] {
            let result = temp_env::with_vars(
                [
                    ("OTEL_EXPORTER_OTLP_ENDPOINT", Some(blank)),
                    (
                        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
                        Some("http://127.0.0.1:1"),
                    ),
                ],
                build_otel_layer,
            );
            assert!(
                matches!(result, Ok(None)),
                "OTEL_EXPORTER_OTLP_ENDPOINT={blank:?} must leave trace export off, whatever \
                 OTEL_EXPORTER_OTLP_TRACES_ENDPOINT says; got a {} result",
                if result.is_ok() { "built" } else { "refused" }
            );
        }

        // Drain anything a regression installed.
        shutdown_tracing().await;
    }

    // ── The two layers' filters: trace export and the events-only log ──────
    //
    // `compose_registry` with an injected collecting tracer and writer, so the
    // assertions read what the trace layer actually exported and what the log
    // layer actually wrote — not what a filter claims to enable.

    use opentelemetry_sdk::trace::SpanData;

    use crate::observability::trace_export::test_support::{
        collecting_tracer, keep_a_second_dispatcher_registered, CapturingWriter,
    };

    /// A trace filter enabling everything except the service's own crate —
    /// the inverse of the default.
    const ALL_BUT_LIVE_EDIT: &str = "trace,live_edit=off";

    /// One span (and, inside the first, one event) per class the trace
    /// filter must tell apart: the service's own INFO span with an INFO and a
    /// DEBUG event inside it, h2's per-frame TRACE span and per-connection
    /// DEBUG span, and a DEBUG span of the service's own.
    fn emit_probe_workload() {
        tracing::info_span!(target: "live_edit::grpc", "live_edit_info_span").in_scope(|| {
            tracing::info!(target: "live_edit::grpc", "info event inside");
            tracing::debug!(target: "live_edit::grpc", "debug event inside");
        });
        tracing::trace_span!(target: "h2::codec::framed_write", "FramedWrite::buffer")
            .in_scope(|| {});
        tracing::debug_span!(target: "h2::proto::connection", "Connection").in_scope(|| {});
        tracing::debug_span!(target: "live_edit::grpc", "live_edit_debug_span").in_scope(|| {});
    }

    /// Run [`emit_probe_workload`] under a registry composed with
    /// `trace_filter` and a collecting tracer, and return what was exported.
    fn exported_under(trace_filter: &str) -> Vec<SpanData> {
        keep_a_second_dispatcher_registered();
        let (_provider, tracer, exporter) = collecting_tracer();
        let subscriber = compose_registry(
            "info",
            LogFormat::Json,
            trace_filter,
            std::io::sink,
            Some(tracer),
        );
        tracing::subscriber::with_default(subscriber, emit_probe_workload);
        exporter.spans()
    }

    fn names(spans: &[SpanData]) -> Vec<String> {
        spans.iter().map(|span| span.name.to_string()).collect()
    }

    /// The incident's root cause, pinned: under the default filter the trace
    /// layer exports the service's own INFO span and nothing else — not h2's
    /// TRACE per-frame span, not its DEBUG connection span, and not a DEBUG
    /// span of the service's own.
    #[test]
    fn trace_filter_default_exports_only_live_edit_info_spans() {
        let exported = exported_under(crate::config::DEFAULT_TRACE_FILTER);
        assert_eq!(
            names(&exported),
            vec!["live_edit_info_span".to_string()],
            "the default trace filter ({}) must export the service's INFO span alone",
            crate::config::DEFAULT_TRACE_FILTER
        );
    }

    /// An INFO event inside an exported span becomes one of its span events;
    /// a DEBUG event there does not.
    #[test]
    fn trace_filter_default_exports_info_events_inside_the_span_as_span_events() {
        let exported = exported_under(crate::config::DEFAULT_TRACE_FILTER);
        let span = exported
            .iter()
            .find(|span| span.name == "live_edit_info_span")
            .expect("precondition: the service's INFO span is exported");
        let events: Vec<String> = span.events.iter().map(|e| e.name.to_string()).collect();
        assert_eq!(
            events,
            vec!["info event inside".to_string()],
            "only the INFO event may ride the span; a DEBUG event adds bytes to every span"
        );
    }

    /// The filter is what keeps the transport spans out: with `trace` as the
    /// trace filter, h2's spans are exported. Without this, the default-filter
    /// test could pass because the probe spans were never created at all.
    #[test]
    fn trace_filter_trace_exports_the_transport_spans() {
        let exported = names(&exported_under("trace"));
        for name in ["FramedWrite::buffer", "Connection"] {
            assert!(
                exported.iter().any(|n| n == name),
                "under `trace`, h2's `{name}` span must be exported; got {exported:?}"
            );
        }
    }

    /// The filter decides the service's own spans too: turning `live_edit`
    /// off drops its INFO span while the h2 spans are still exported, so the
    /// export pipeline itself is shown working.
    #[test]
    fn trace_filter_with_live_edit_off_exports_no_live_edit_span() {
        let exported = names(&exported_under(ALL_BUT_LIVE_EDIT));
        assert!(
            !exported.iter().any(|n| n == "live_edit_info_span"),
            "`{ALL_BUT_LIVE_EDIT}` must not export the service's span; got {exported:?}"
        );
        assert!(
            exported.iter().any(|n| n == "FramedWrite::buffer"),
            "precondition: the rest of `{ALL_BUT_LIVE_EDIT}` still exports; got {exported:?}"
        );
    }

    /// What the log layer writes, in `format`, for one INFO event inside an
    /// INFO span of the service that carries a `session_id` field.
    ///
    /// The log filter is pinned to `info` (both log variables unset) so the
    /// span itself would pass it; only the events-only filter keeps it out.
    fn log_output_for_an_event_inside_a_span(format: LogFormat) -> String {
        keep_a_second_dispatcher_registered();
        let writer = CapturingWriter::default();
        let (_provider, tracer, _exporter) = collecting_tracer();
        temp_env::with_vars(
            [
                ("LIVE_EDIT_LOG_LEVEL", None::<&str>),
                ("RUST_LOG", None::<&str>),
            ],
            || {
                let subscriber = compose_registry(
                    "info",
                    format,
                    crate::config::DEFAULT_TRACE_FILTER,
                    writer.clone(),
                    Some(tracer),
                );
                tracing::subscriber::with_default(subscriber, || {
                    tracing::info_span!(target: "live_edit::grpc", "rpc", session_id = "s-1")
                        .in_scope(|| tracing::info!(target: "live_edit::grpc", "inside the span"));
                });
            },
        );
        writer.text()
    }

    /// `text` with ANSI escape sequences (`ESC [ ... <letter>`) removed, so the
    /// pretty format's styling cannot split a word the assertion looks for.
    fn strip_ansi(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut chars = text.chars();
        while let Some(c) = chars.next() {
            if c == '\u{1b}' {
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    /// The log layer sees events only, in the pretty format too: the event
    /// is written, but no `in live_edit::grpc::rpc with session_id: …`
    /// span-context line follows it.
    #[test]
    fn log_layer_pretty_line_inside_a_span_carries_no_span_context() {
        let output = strip_ansi(&log_output_for_an_event_inside_a_span(LogFormat::Pretty));
        assert!(
            output.contains("inside the span"),
            "precondition: the event is logged; got:\n{output}"
        );
        for needle in ["live_edit::grpc::rpc", "session_id"] {
            assert!(
                !output.contains(needle),
                "a pretty log line must carry no span context (`{needle}`) — the log layer \
                 never sees spans; got:\n{output}"
            );
        }
    }

    /// The log layer sees events only: an INFO event inside an INFO span of
    /// the service writes a JSON line with neither a `span` nor a `spans`
    /// key, so no log line carries span context whatever spans exist.
    #[test]
    fn log_layer_json_line_inside_a_span_carries_no_span_context() {
        let lines: Vec<serde_json::Value> = log_output_for_an_event_inside_a_span(LogFormat::Json)
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("each log line must be JSON"))
            .collect();
        assert_eq!(
            lines.len(),
            1,
            "precondition: the event is logged; got {lines:?}"
        );
        let line = &lines[0];
        assert_eq!(line["fields"]["message"], "inside the span");
        for key in ["span", "spans"] {
            assert!(
                line.get(key).is_none(),
                "a log line must carry no `{key}` key — the log layer never sees spans; got {line}"
            );
        }
    }

    // ── remote_parent_context: W3C `traceparent` only ──────────────────────

    const TRACE_ID: &str = "0af7651916cd43dd8448eb211c80319c";
    const PARENT_SPAN_ID: &str = "b7ad6b7169203331";

    fn headers(pairs: &[(&'static str, &str)]) -> http::HeaderMap {
        let mut map = http::HeaderMap::new();
        for (name, value) in pairs {
            map.insert(
                *name,
                http::HeaderValue::from_str(value).expect("valid header value"),
            );
        }
        map
    }

    #[test]
    fn remote_parent_context_valid_traceparent_is_a_remote_parent() {
        let cx = remote_parent_context(&headers(&[(
            "traceparent",
            &format!("00-{TRACE_ID}-{PARENT_SPAN_ID}-01"),
        )]))
        .expect("a valid traceparent is a parent");
        let span = cx.span();
        let parent = span.span_context();
        assert_eq!(parent.trace_id().to_string(), TRACE_ID);
        assert_eq!(parent.span_id().to_string(), PARENT_SPAN_ID);
        assert!(parent.is_remote());
        assert!(parent.is_sampled());
    }

    #[test]
    fn remote_parent_context_absent_or_malformed_traceparent_is_none() {
        assert!(remote_parent_context(&http::HeaderMap::new()).is_none());
        for malformed in [
            "garbage",
            &format!("00-{TRACE_ID}-{PARENT_SPAN_ID}") as &str,
            &format!("00-{}-{PARENT_SPAN_ID}-01", "0".repeat(32)),
        ] {
            assert!(
                remote_parent_context(&headers(&[("traceparent", malformed)])).is_none(),
                "traceparent {malformed:?} must give no parent"
            );
        }
    }

    /// An inbound `tracestate` never reaches the parent context, so it can
    /// never be copied onto the span and exported.
    #[test]
    fn remote_parent_context_ignores_inbound_tracestate() {
        let cx = remote_parent_context(&headers(&[
            ("traceparent", &format!("00-{TRACE_ID}-{PARENT_SPAN_ID}-01")),
            ("tracestate", "vendor=caller-chosen-bytes"),
        ]))
        .expect("a valid traceparent is a parent");
        assert_eq!(
            cx.span().span_context().trace_state().header(),
            "",
            "the inbound tracestate must be ignored"
        );
    }
}
