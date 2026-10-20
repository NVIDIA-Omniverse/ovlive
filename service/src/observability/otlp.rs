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

//! The OTLP collector endpoints: how each endpoint variable is read, and how
//! the trace and metrics exporters are built for it.
//!
//! **Endpoint values.** Each variable (`OTEL_EXPORTER_OTLP_ENDPOINT` for
//! traces, `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` for the metrics push) is
//! either unset or blank, which switches that signal's export off, or an
//! absolute `http://` or `https://` URL with a host. The scheme alone picks the
//! transport: `http` is plaintext gRPC, `https` is gRPC over TLS. Any other
//! value stops startup with an error naming the variable
//! ([`otlp_endpoint_from`]). The scheme is judged by `http::Uri`, the parser
//! the exporter's own `Channel::from_shared` uses, and the same
//! case-insensitive `https` test as tonic's connector, so TLS is attached
//! exactly when tonic would demand it.
//!
//! **Why TLS is attached here.** opentelemetry-otlp builds its channel with
//! `Channel::from_shared`, which never gets tonic's automatic TLS for an
//! `https` URI; without a TLS config the connector refuses every connection
//! ("Connecting to HTTPS without TLS enabled"). So for an `https://` endpoint
//! the builders below pass [`otlp_tls_config`] through `with_tls_config`. A
//! plaintext endpoint gets no TLS config and reads no CA store.
//!
//! **Trust roots.** The platform's native roots: the image's CA bundle, or
//! the files `SSL_CERT_FILE` / `SSL_CERT_DIR` name, loaded once when the
//! exporter is built, so a changed bundle takes effect only after a restart.
//! No client certificate is sent. The OTel `OTEL_EXPORTER_OTLP_CERTIFICATE`,
//! `_CLIENT_CERTIFICATE`, `_CLIENT_KEY` and `_INSECURE` variables are not
//! read, here or by the exporter.
//!
//! **Build errors keep their cause.** tonic's transport error displays only
//! its kind ("transport error") and the exporter stringifies it, losing why
//! the TLS setup failed (for example "no native certs found"). So before
//! handing the TLS settings to the exporter, [`check_tls_connector`] builds
//! the same connector through tonic directly and renders the whole error
//! chain.
//!
//! **No compression.** Neither of opentelemetry-otlp's tonic compression
//! features (`gzip-tonic`, `zstd-tonic`) is built, so the exporter cannot
//! honour any compression variable it reads:
//! - `gzip` and `zstd` fail the build for the missing feature;
//! - every other value, `none` and the empty value included, fails to parse,
//!   since the SDK knows only those two;
//! - nothing set in code can make it skip the variable: the code-set option
//!   holds only `gzip` or `zstd`, and the variable is read whenever it is
//!   absent.
//!
//! So a set compression variable can only ever stop the exporter, and
//! [`check_compression_unset`] refuses it first, with an error naming the
//! variable rather than the endpoint.
//!
//! **Handshake bound.** The TLS handshake runs inside the channel's connect,
//! before the request timeout starts, so without its own bound a peer that
//! accepts TCP and then stalls the handshake would hold an export forever.
//! [`otlp_tls_config`] bounds it by [`OTLP_EXPORT_TIMEOUT`].

use std::error::Error as StdError;
use std::time::Duration;

use http::uri::Scheme;
use opentelemetry_otlp::{Protocol, WithExportConfig as _, WithTonicConfig};
use tonic::transport::ClientTlsConfig;

/// The trace endpoint variable.
pub const OTEL_EXPORTER_OTLP_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";

/// The metrics push endpoint variable. The only switch for the push: there is
/// no fallback to [`OTEL_EXPORTER_OTLP_ENDPOINT`].
pub const OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT";

/// The compression variable every exporter reads when its signal's own
/// variable is unset.
pub const OTEL_EXPORTER_OTLP_COMPRESSION: &str = "OTEL_EXPORTER_OTLP_COMPRESSION";

/// The trace exporter's own compression variable, read before
/// [`OTEL_EXPORTER_OTLP_COMPRESSION`].
pub const OTEL_EXPORTER_OTLP_TRACES_COMPRESSION: &str = "OTEL_EXPORTER_OTLP_TRACES_COMPRESSION";

/// The metrics exporter's own compression variable, read before
/// [`OTEL_EXPORTER_OTLP_COMPRESSION`].
pub const OTEL_EXPORTER_OTLP_METRICS_COMPRESSION: &str = "OTEL_EXPORTER_OTLP_METRICS_COMPRESSION";

/// Cap on each OTLP metrics export request, and on the TLS handshake of
/// either exporter's connection.
///
/// The OTLP default request timeout (10 s) is far too long for the final
/// flush at shutdown, which runs outside the service's shutdown budget and
/// must fit inside the termination grace period (`main.rs`).
pub const OTLP_EXPORT_TIMEOUT: Duration = Duration::from_secs(3);

/// The transport an endpoint's scheme selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OtlpTransport {
    /// `http://`: plaintext gRPC.
    Plaintext,
    /// `https://`: gRPC over TLS.
    Tls,
}

impl OtlpTransport {
    /// The value of the `transport` field on the export-enabled startup lines.
    pub fn as_str(self) -> &'static str {
        match self {
            OtlpTransport::Plaintext => "plaintext",
            OtlpTransport::Tls => "tls",
        }
    }
}

/// A set, valid OTLP endpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OtlpEndpoint {
    /// The variable the endpoint came from, for error messages.
    pub var: &'static str,
    /// The trimmed value: what the exporter receives.
    pub url: String,
    /// The transport its scheme selects.
    pub transport: OtlpTransport,
}

/// The remedy every endpoint error ends with.
const ENDPOINT_REMEDY: &str =
    "use an http:// URL for plaintext or an https:// URL for TLS, or leave it unset to turn \
     export off";

/// Classify a raw endpoint value read from `var`.
///
/// - Unset, empty or whitespace-only: `Ok(None)`, export off.
/// - Otherwise the trimmed value must parse as an absolute URL with a host and
///   the scheme `http` (plaintext) or `https` (TLS), case-insensitively.
/// - Anything else is an error that names `var` and gives the remedy. It
///   echoes at most the scheme, never the value, which may carry credentials.
pub fn otlp_endpoint_from(
    var: &'static str,
    raw: Option<&str>,
) -> Result<Option<OtlpEndpoint>, String> {
    let Some(value) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let uri: http::Uri = value
        .parse()
        .map_err(|error| format!("{var} is not a valid URL ({error}); {ENDPOINT_REMEDY}"))?;
    let transport = match uri.scheme() {
        Some(scheme) if *scheme == Scheme::HTTP => OtlpTransport::Plaintext,
        Some(scheme) if *scheme == Scheme::HTTPS => OtlpTransport::Tls,
        Some(scheme) => {
            return Err(format!(
                "{var} has the unsupported scheme `{scheme}`; {ENDPOINT_REMEDY}"
            ))
        }
        None => return Err(format!("{var} has no scheme; {ENDPOINT_REMEDY}")),
    };
    if uri.host().is_none_or(str::is_empty) {
        return Err(format!("{var} has no host; {ENDPOINT_REMEDY}"));
    }
    Ok(Some(OtlpEndpoint {
        var,
        url: value.to_string(),
        transport,
    }))
}

/// [`otlp_endpoint_from`] applied to the process environment. A value that
/// is not valid UTF-8 is an error naming `var`.
pub fn otlp_endpoint_from_env(var: &'static str) -> Result<Option<OtlpEndpoint>, String> {
    match std::env::var(var) {
        Ok(value) => otlp_endpoint_from(var, Some(&value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(format!("{var} is not valid UTF-8; {ENDPOINT_REMEDY}"))
        }
    }
}

/// The TLS settings of an `https://` endpoint: the platform's native trust
/// roots, and the handshake bounded by [`OTLP_EXPORT_TIMEOUT`].
///
/// Native roots, not `with_enabled_roots`: that would also add the webpki
/// roots if tonic's `tls-webpki-roots` ever entered the graph, silently
/// widening what the service trusts.
pub fn otlp_tls_config() -> ClientTlsConfig {
    ClientTlsConfig::new()
        .with_native_roots()
        .timeout(OTLP_EXPORT_TIMEOUT)
}

/// Build the OTLP trace exporter for `endpoint`.
///
/// The request timeout and headers stay the SDK's
/// (`OTEL_EXPORTER_OTLP_TIMEOUT` and friends); a set compression variable is
/// refused ([`check_compression_unset`]). An error names the variable and
/// carries the cause.
pub fn span_exporter_for(
    endpoint: &OtlpEndpoint,
) -> Result<opentelemetry_otlp::SpanExporter, String> {
    check_compression_unset(OTEL_EXPORTER_OTLP_TRACES_COMPRESSION)?;
    let builder = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint.url.clone());
    with_transport(builder, endpoint)?
        .build()
        .map_err(|error| build_error(endpoint, "trace", &error))
}

/// Build the OTLP metrics exporter for `endpoint`, each export capped at
/// [`OTLP_EXPORT_TIMEOUT`]; a set compression variable is refused
/// ([`check_compression_unset`]). An error names the variable and carries
/// the cause.
pub fn metric_exporter_for(
    endpoint: &OtlpEndpoint,
) -> Result<opentelemetry_otlp::MetricExporter, String> {
    check_compression_unset(OTEL_EXPORTER_OTLP_METRICS_COMPRESSION)?;
    let builder = opentelemetry_otlp::MetricExporter::builder()
        .with_tonic()
        .with_protocol(Protocol::Grpc)
        .with_endpoint(endpoint.url.clone())
        .with_timeout(OTLP_EXPORT_TIMEOUT);
    with_transport(builder, endpoint)?
        .build()
        .map_err(|error| build_error(endpoint, "metrics", &error))
}

/// Refuse the compression variable the exporter would read: `signal_var` if
/// it is set, otherwise [`OTEL_EXPORTER_OTLP_COMPRESSION`] if that is. Any
/// value is refused, `none` and the empty value included, because the SDK
/// reads the variable itself and accepts no value this build can honour (see
/// the module doc). The read follows the SDK's: a value that is not valid
/// UTF-8 is skipped, as the SDK skips it.
fn check_compression_unset(signal_var: &'static str) -> Result<(), String> {
    for var in [signal_var, OTEL_EXPORTER_OTLP_COMPRESSION] {
        if let Ok(value) = std::env::var(var) {
            return Err(format!(
                "{var} is set to `{value}`, but this build sends OTLP exports uncompressed \
                 and accepts no compression setting, `none` included; unset {var}"
            ));
        }
    }
    Ok(())
}

/// Attach [`otlp_tls_config`] to `builder` for a TLS endpoint, after
/// [`check_tls_connector`] has proven it builds; a plaintext endpoint's
/// builder is returned unchanged.
fn with_transport<B: WithTonicConfig>(builder: B, endpoint: &OtlpEndpoint) -> Result<B, String> {
    match endpoint.transport {
        OtlpTransport::Plaintext => Ok(builder),
        OtlpTransport::Tls => {
            check_tls_connector(endpoint)?;
            Ok(builder.with_tls_config(otlp_tls_config()))
        }
    }
}

/// Build the TLS connector for `endpoint` through tonic directly, so a
/// failure (for example no readable CA certificates) is reported with its
/// cause; the exporter's own error would say only "transport error".
fn check_tls_connector(endpoint: &OtlpEndpoint) -> Result<(), String> {
    tonic::transport::Endpoint::from_shared(endpoint.url.clone())
        .and_then(|channel| channel.tls_config(otlp_tls_config()))
        .map(drop)
        .map_err(|error| {
            format!(
                "{}: TLS to the collector could not be set up: {}. An https:// endpoint \
                 verifies the collector's certificate against the system CA store, which \
                 SSL_CERT_FILE or SSL_CERT_DIR can replace",
                endpoint.var,
                error_chain(&error)
            )
        })
}

/// The error for an exporter that could not be built.
fn build_error(endpoint: &OtlpEndpoint, signal: &str, error: &(dyn StdError + 'static)) -> String {
    format!(
        "{}: the OTLP {signal} exporter could not be built: {}",
        endpoint.var,
        error_chain(error)
    )
}

/// `error` and each of its sources, joined with `: `. A source whose text the
/// line already carries is skipped, since some errors repeat their source in
/// their own display.
fn error_chain(error: &(dyn StdError + 'static)) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        let cause_text = cause.to_string();
        if !text.contains(&cause_text) {
            text.push_str(": ");
            text.push_str(&cause_text);
        }
        source = cause.source();
    }
    text
}

#[cfg(test)]
mod tests {
    use std::net::SocketAddr;
    use std::sync::Arc;

    use opentelemetry::trace::{Span as _, Tracer as _};
    use opentelemetry_sdk::metrics::data::ResourceMetrics;
    use opentelemetry_sdk::metrics::exporter::PushMetricExporter as _;
    use opentelemetry_sdk::trace::SpanExporter as _;
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::sync::oneshot;
    use tracing::instrument::WithSubscriber as _;

    use super::*;
    use crate::config::{LogFormat, DEFAULT_TRACE_FILTER};
    use crate::observability::metrics::Metrics;
    use crate::observability::metrics_export::{
        MetricExportReporter, ReportingMetricExporter, OTLP_METRICS_EXPORT_FAILED_MSG,
    };
    use crate::observability::trace_export::test_support::{
        collecting_tracer, keep_a_second_dispatcher_registered, CapturingWriter,
    };
    use crate::observability::trace_export::{
        ReleasingExporter, TraceExportState, TRACE_SPANS_DROPPED_MSG,
    };
    use crate::observability::{build_registry, shutdown_tracing, OTEL_PROVIDER_TEST_LOCK};

    // ── T5: endpoint classification ──────────────────────────────────────────

    /// The text every endpoint error ends with.
    const REMEDY_FRAGMENT: &str = "use an http:// URL for plaintext or an https:// URL for TLS";

    /// Unset, empty and whitespace-only values switch export off.
    #[test]
    fn otlp_endpoint_from_unset_or_blank_is_off() {
        for raw in [None, Some(""), Some("   "), Some("\t\n ")] {
            assert_eq!(
                otlp_endpoint_from(OTEL_EXPORTER_OTLP_ENDPOINT, raw),
                Ok(None),
                "{raw:?}"
            );
        }
    }

    /// The scheme alone picks the transport, case-insensitively, and the
    /// exporter receives the trimmed value.
    #[test]
    fn otlp_endpoint_from_http_or_https_url_picks_the_transport_and_trims() {
        let cases = [
            (
                "http://collector.example.com:4317",
                "http://collector.example.com:4317",
                OtlpTransport::Plaintext,
            ),
            (
                "HTTP://collector.example.com:4317",
                "HTTP://collector.example.com:4317",
                OtlpTransport::Plaintext,
            ),
            (
                "https://collector.example.com:4317",
                "https://collector.example.com:4317",
                OtlpTransport::Tls,
            ),
            (
                "HTTPS://collector.example.com:4317",
                "HTTPS://collector.example.com:4317",
                OtlpTransport::Tls,
            ),
            (
                "  https://collector.example.com:4317/\n",
                "https://collector.example.com:4317/",
                OtlpTransport::Tls,
            ),
        ];
        for (raw, url, transport) in cases {
            assert_eq!(
                otlp_endpoint_from(OTEL_EXPORTER_OTLP_METRICS_ENDPOINT, Some(raw)),
                Ok(Some(OtlpEndpoint {
                    var: OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
                    url: url.to_string(),
                    transport,
                })),
                "{raw:?}"
            );
        }
    }

    /// Every value that is not blank or an `http://` / `https://` URL with a
    /// host is refused with an error naming the variable it came from and
    /// giving the remedy.
    #[test]
    fn otlp_endpoint_from_any_other_value_is_an_error_naming_the_variable() {
        let refused = [
            "collector.example.com:4317",
            "grpc://collector.example.com:4317",
            "grpcs://collector.example.com:4317",
            "unix:/var/run/otel-collector.sock",
            "http://",
            "https://",
            "not a url",
            "http://[::bad",
        ];
        for var in [
            OTEL_EXPORTER_OTLP_ENDPOINT,
            OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
        ] {
            for raw in refused {
                let error = otlp_endpoint_from(var, Some(raw))
                    .expect_err(&format!("{var}={raw:?} must be refused"));
                assert!(
                    error.starts_with(&format!("{var} ")) && error.contains(REMEDY_FRAGMENT),
                    "{var}={raw:?}: the error must name the variable and give the remedy; \
                     got {error:?}"
                );
            }
        }
    }

    /// A refused value is never echoed whole: it may carry credentials.
    #[test]
    fn otlp_endpoint_from_refusal_does_not_echo_the_value() {
        let error = otlp_endpoint_from(
            OTEL_EXPORTER_OTLP_ENDPOINT,
            Some("grpc://otel-user:s3cret-token@collector.example.com:4317"),
        )
        .expect_err("grpc:// is refused");
        assert!(
            !error.contains("s3cret-token") && !error.contains("collector.example.com"),
            "the error may name the scheme, never the value; got {error:?}"
        );
    }

    /// The environment variant reads the named variable: unset is off, a
    /// value is classified, and a value that is not UTF-8 is refused naming
    /// the variable.
    #[cfg(unix)]
    #[test]
    fn otlp_endpoint_from_env_reads_and_classifies_the_variable() {
        use std::os::unix::ffi::OsStringExt as _;

        temp_env::with_var(OTEL_EXPORTER_OTLP_METRICS_ENDPOINT, None::<&str>, || {
            assert_eq!(
                otlp_endpoint_from_env(OTEL_EXPORTER_OTLP_METRICS_ENDPOINT),
                Ok(None)
            );
        });
        temp_env::with_var(
            OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
            Some("https://collector.example.com:4317"),
            || {
                let endpoint = otlp_endpoint_from_env(OTEL_EXPORTER_OTLP_METRICS_ENDPOINT)
                    .expect("a valid endpoint")
                    .expect("set");
                assert_eq!(endpoint.transport, OtlpTransport::Tls);
            },
        );
        temp_env::with_var(
            OTEL_EXPORTER_OTLP_METRICS_ENDPOINT,
            Some(std::ffi::OsString::from_vec(vec![b'h', 0xff])),
            || {
                let error = otlp_endpoint_from_env(OTEL_EXPORTER_OTLP_METRICS_ENDPOINT)
                    .expect_err("a non-UTF-8 value is refused");
                assert!(
                    error.starts_with(OTEL_EXPORTER_OTLP_METRICS_ENDPOINT)
                        && error.contains("not valid UTF-8"),
                    "got {error:?}"
                );
            },
        );
    }

    #[test]
    fn otlp_transport_as_str_names_the_startup_line_values() {
        assert_eq!(OtlpTransport::Plaintext.as_str(), "plaintext");
        assert_eq!(OtlpTransport::Tls.as_str(), "tls");
    }

    // ── Compression refusal ──────────────────────────────────────────────────

    /// The three compression variables the exporters read, every one unset.
    const NO_COMPRESSION: [(&str, Option<&str>); 3] = [
        (OTEL_EXPORTER_OTLP_COMPRESSION, None),
        (OTEL_EXPORTER_OTLP_TRACES_COMPRESSION, None),
        (OTEL_EXPORTER_OTLP_METRICS_COMPRESSION, None),
    ];

    /// Run `f` with every compression variable pinned unset, so an exporter
    /// built inside it never meets one a sibling test set.
    fn without_compression<T>(f: impl FnOnce() -> T) -> T {
        temp_env::with_vars(NO_COMPRESSION, f)
    }

    /// `check_compression_unset(signal_var)` with exactly `set` among the
    /// compression variables.
    fn compression_check_with(
        signal_var: &'static str,
        set: &[(&'static str, &str)],
    ) -> Result<(), String> {
        let vars: Vec<(&str, Option<&str>)> = NO_COMPRESSION
            .iter()
            .map(|(var, _)| {
                let value = set.iter().find(|(name, _)| name == var).map(|(_, v)| *v);
                (*var, value)
            })
            .collect();
        temp_env::with_vars(vars, || check_compression_unset(signal_var))
    }

    /// With no compression variable set, both exporters' checks pass.
    #[test]
    fn check_compression_unset_with_nothing_set_passes() {
        for signal_var in [
            OTEL_EXPORTER_OTLP_TRACES_COMPRESSION,
            OTEL_EXPORTER_OTLP_METRICS_COMPRESSION,
        ] {
            assert_eq!(
                compression_check_with(signal_var, &[]),
                Ok(()),
                "{signal_var}"
            );
        }
    }

    /// Every set value of the generic variable is refused — `none` and the
    /// empty value included — with an error that names the variable and its
    /// value and says to unset it.
    #[test]
    fn check_compression_unset_refuses_every_generic_value_naming_the_variable_to_unset() {
        for value in ["gzip", "none", ""] {
            for signal_var in [
                OTEL_EXPORTER_OTLP_TRACES_COMPRESSION,
                OTEL_EXPORTER_OTLP_METRICS_COMPRESSION,
            ] {
                let error =
                    compression_check_with(signal_var, &[(OTEL_EXPORTER_OTLP_COMPRESSION, value)])
                        .expect_err(&format!("{signal_var}: `{value}` must be refused"));
                assert!(
                    error.starts_with(&format!(
                        "{OTEL_EXPORTER_OTLP_COMPRESSION} is set to `{value}`"
                    )) && error.ends_with(&format!("unset {OTEL_EXPORTER_OTLP_COMPRESSION}")),
                    "{signal_var}, `{value}`: got {error:?}"
                );
            }
        }
    }

    /// A signal's own variable is read before the generic one, as the SDK
    /// reads it, so the error names the per-signal variable even when the
    /// generic one is set too; another signal's variable is not read.
    #[test]
    fn check_compression_unset_per_signal_variable_wins_over_the_generic_one() {
        for (signal_var, other_var) in [
            (
                OTEL_EXPORTER_OTLP_TRACES_COMPRESSION,
                OTEL_EXPORTER_OTLP_METRICS_COMPRESSION,
            ),
            (
                OTEL_EXPORTER_OTLP_METRICS_COMPRESSION,
                OTEL_EXPORTER_OTLP_TRACES_COMPRESSION,
            ),
        ] {
            let error = compression_check_with(
                signal_var,
                &[
                    (signal_var, "none"),
                    (OTEL_EXPORTER_OTLP_COMPRESSION, "gzip"),
                ],
            )
            .expect_err("a set per-signal value is refused");
            assert!(
                error.starts_with(&format!("{signal_var} is set to `none`"))
                    && error.ends_with(&format!("unset {signal_var}"))
                    && !error.contains(OTEL_EXPORTER_OTLP_COMPRESSION),
                "{signal_var}: the error must name the per-signal variable only; got {error:?}"
            );

            assert_eq!(
                compression_check_with(signal_var, &[(other_var, "gzip")]),
                Ok(()),
                "{signal_var}: {other_var} belongs to the other exporter"
            );
        }
    }

    // ── T1, T2, T4: the exporters against a TLS-refusing listener ────────────
    //
    // `spawn_tls_alert_listener` records the first bytes the client sends on
    // its first connection, then answers every connection with a fatal TLS
    // alert. An exporter that attaches TLS opens with a ClientHello, a TLS
    // handshake record; one that does not (an `https://` endpoint without a
    // TLS config, which tonic's connector refuses after opening the TCP
    // connection) sends nothing at all. Building the TLS settings loads the
    // runner's native trust roots, so the runner needs a CA bundle;
    // `SSL_CERT_FILE` is never changed in-process, since other TLS tests in
    // this binary read the process environment too. The `https://` path that
    // completes a handshake is the binary-level `otlp_tls_export_binary`
    // test, which can give its child process a CA of its own.

    /// Upper bound on each wait for an export to reach the listener or
    /// finish. Unreachable on the healthy path, which takes milliseconds:
    /// the listener answers at once, and each export is bounded anyway (the
    /// TLS handshake by [`OTLP_EXPORT_TIMEOUT`], the request by the exporter's
    /// own timeout, a provider shutdown by the SDK's 5 s).
    const EXPORT_BUDGET: Duration = Duration::from_secs(20);

    /// TLS record content type `handshake` (22), the first byte of a
    /// ClientHello.
    const TLS_HANDSHAKE_RECORD: u8 = 0x16;

    /// Major protocol version byte of every TLS record header (`0x03`).
    const TLS_MAJOR_VERSION: u8 = 0x03;

    /// A TLS record header is five bytes: type, version (2), length (2).
    const TLS_RECORD_HEADER_LEN: usize = 5;

    /// A fatal `handshake_failure` alert record: the listener's answer.
    const TLS_HANDSHAKE_FAILURE_ALERT: [u8; 7] = [0x15, 0x03, 0x03, 0x00, 0x02, 0x02, 0x28];

    /// What the exporters' error text carries for the alert above. tonic
    /// keeps rustls' cause in the export error, so the operator reads why the
    /// handshake failed.
    const HANDSHAKE_FAILURE_FRAGMENT: &str = "HandshakeFailure";

    /// Spawn the listener described above; the receiver yields the first
    /// connection's opening bytes (up to one TLS record header, fewer if the
    /// client closes first).
    async fn spawn_tls_alert_listener() -> (SocketAddr, oneshot::Receiver<Vec<u8>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind the TLS alert listener");
        let addr = listener.local_addr().expect("listener address");
        let (first_tx, first_rx) = oneshot::channel();
        tokio::spawn(async move {
            let mut first_tx = Some(first_tx);
            while let Ok((mut socket, _)) = listener.accept().await {
                let record = first_tx.take();
                tokio::spawn(async move {
                    let mut head = vec![0u8; TLS_RECORD_HEADER_LEN];
                    let mut read = 0;
                    while read < TLS_RECORD_HEADER_LEN {
                        match socket.read(&mut head[read..]).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => read += n,
                        }
                    }
                    head.truncate(read);
                    if let Some(record) = record {
                        let _ = record.send(head);
                    }
                    let _ = socket.write_all(&TLS_HANDSHAKE_FAILURE_ALERT).await;
                    let _ = socket.shutdown().await;
                });
            }
        });
        (addr, first_rx)
    }

    fn https_url(addr: SocketAddr) -> String {
        format!("https://127.0.0.1:{}", addr.port())
    }

    /// Wait for the listener's record of the first connection and assert it
    /// opened with a TLS handshake record.
    async fn assert_first_connection_opened_with_tls(
        first: oneshot::Receiver<Vec<u8>>,
        signal: &str,
    ) {
        let head = tokio::time::timeout(EXPORT_BUDGET, first)
            .await
            .unwrap_or_else(|_| {
                panic!("the {signal} exporter never connected to the https:// endpoint")
            })
            .expect("the listener records its first connection");
        assert!(
            head.len() >= 2 && head[0] == TLS_HANDSHAKE_RECORD && head[1] == TLS_MAJOR_VERSION,
            "an https:// {signal} endpoint must be spoken to over TLS: the connection must open \
             with a ClientHello (0x16 0x03 ...), got {head:02x?} (empty: the exporter refused \
             the connection without TLS)"
        );
    }

    /// The JSON events with `message` emitted while `future` runs, captured
    /// on the future itself so a sibling test's events never mix in.
    async fn lines_with_message<F: std::future::Future>(
        message: &str,
        future: F,
    ) -> (F::Output, Vec<serde_json::Value>) {
        keep_a_second_dispatcher_registered();
        let writer = CapturingWriter::default();
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(writer.clone())
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .finish();
        let output = tokio::time::timeout(EXPORT_BUDGET, future.with_subscriber(subscriber))
            .await
            .expect("the export finishes inside its own timeout");
        let lines = writer
            .text()
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("JSON line"))
            .filter(|event| event["fields"]["message"] == message)
            .collect();
        (output, lines)
    }

    /// T1, traces: with `OTEL_EXPORTER_OTLP_ENDPOINT` set to an `https://`
    /// URL, the pipeline `build_registry` wires reports the TLS transport and
    /// exports over TLS.
    ///
    /// Shares the `OTEL_PROVIDER` slot with the `build_otel_layer` tests in
    /// `observability`, so it holds [`OTEL_PROVIDER_TEST_LOCK`] throughout and
    /// drains the slot first and last, as they do.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn https_trace_endpoint_exports_over_tls_through_build_registry() {
        let _slot = OTEL_PROVIDER_TEST_LOCK.lock().await;
        keep_a_second_dispatcher_registered();
        shutdown_tracing().await;
        let (addr, first) = spawn_tls_alert_listener().await;
        let url = https_url(addr);

        temp_env::with_vars(
            [
                (OTEL_EXPORTER_OTLP_ENDPOINT, Some(url.as_str())),
                ("OTEL_BSP_MAX_QUEUE_SIZE", None),
                ("OTEL_TRACES_SAMPLER", None),
                ("OTEL_TRACES_SAMPLER_ARG", None),
                (OTEL_EXPORTER_OTLP_TRACES_COMPRESSION, None),
                (OTEL_EXPORTER_OTLP_COMPRESSION, None),
            ],
            || {
                let (subscriber, export) =
                    build_registry("info", LogFormat::Json, DEFAULT_TRACE_FILTER)
                        .expect("an https:// endpoint builds");
                let (_state, transport) = export.expect("trace export is enabled");
                assert_eq!(transport, OtlpTransport::Tls);
                tracing::subscriber::with_default(subscriber, || {
                    tracing::info_span!(target: "live_edit::observability", "tls_probe")
                        .in_scope(|| {});
                });
            },
        );
        // Flushes the span: the export connects to the listener.
        shutdown_tracing().await;

        assert_first_connection_opened_with_tls(first, "trace").await;
    }

    /// Build `Metrics` with the push endpoint at `url` (the periodic export
    /// pushed an hour out, so only the shutdown's final export runs), shut it
    /// down off the runtime's workers as `main` does, and return the push
    /// arm's failure reporter.
    async fn metrics_pushed_once_to(url: &str) -> Arc<MetricExportReporter> {
        let metrics = temp_env::with_vars(
            [
                (OTEL_EXPORTER_OTLP_METRICS_ENDPOINT, Some(url)),
                (OTEL_EXPORTER_OTLP_ENDPOINT, None),
                ("OTEL_METRIC_EXPORT_INTERVAL", Some("3600000")),
                (OTEL_EXPORTER_OTLP_METRICS_COMPRESSION, None),
                (OTEL_EXPORTER_OTLP_COMPRESSION, None),
            ],
            || Metrics::try_new("otlp-tls-test"),
        )
        .expect("an https:// metrics endpoint builds");
        let reporter = Arc::clone(
            metrics
                .otlp_export_reporter()
                .expect("the push arm is attached"),
        );
        tokio::time::timeout(
            EXPORT_BUDGET,
            tokio::task::spawn_blocking(move || metrics.shutdown()),
        )
        .await
        .expect("the final export finishes inside its own timeout")
        .expect("the shutdown thread does not panic");
        reporter
    }

    /// T1, metrics: with `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` set to an
    /// `https://` URL, the push arm `Metrics::try_new` attaches exports over
    /// TLS.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn https_metrics_endpoint_exports_over_tls_through_metrics_try_new() {
        let (addr, first) = spawn_tls_alert_listener().await;

        metrics_pushed_once_to(&https_url(addr)).await;

        assert_first_connection_opened_with_tls(first, "metrics").await;
    }

    /// T4: the push arm `Metrics::try_new` attaches reports through the
    /// failure reporter it hands back: the shutdown's final export, refused
    /// by the listener, is counted there.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metrics_try_new_push_arm_counts_a_failed_push_in_its_reporter() {
        let (addr, _first) = spawn_tls_alert_listener().await;

        let reporter = metrics_pushed_once_to(&https_url(addr)).await;

        assert_eq!(
            reporter.failed_exports_total(),
            1,
            "the shutdown's one export failed and must be counted by the reporter the push \
             arm exports through"
        );
    }

    /// T2, traces: an `https://` export the collector refuses at the TLS
    /// handshake is reported by the `trace spans dropped` WARN, whose
    /// `last_export_error` names the handshake failure.
    #[tokio::test]
    async fn https_trace_export_refused_at_the_handshake_names_it_in_the_drop_warn() {
        let (addr, _first) = spawn_tls_alert_listener().await;
        let endpoint = otlp_endpoint_from(OTEL_EXPORTER_OTLP_ENDPOINT, Some(&https_url(addr)))
            .expect("valid")
            .expect("set");
        let state = Arc::new(TraceExportState::new(4));
        let exporter = ReleasingExporter::new(
            without_compression(|| span_exporter_for(&endpoint)).expect("the exporter builds"),
            Arc::clone(&state),
        );
        let (_provider, tracer, collected) = collecting_tracer();
        tracer.start("tls_probe").end();
        let batch = collected.spans();
        assert_eq!(batch.len(), 1, "precondition: a batch of one span");

        let (result, lines) =
            lines_with_message(TRACE_SPANS_DROPPED_MSG, exporter.export(batch)).await;

        result.expect("the releasing exporter answers Ok");
        assert_eq!(
            state.export_failed_total(),
            1,
            "precondition: the export failed"
        );
        assert_eq!(lines.len(), 1, "one drop line: {lines:?}");
        assert_eq!(lines[0]["level"], "WARN", "{}", lines[0]);
        let error = lines[0]["fields"]["last_export_error"]
            .as_str()
            .unwrap_or_default();
        assert!(
            error.contains(HANDSHAKE_FAILURE_FRAGMENT),
            "last_export_error must name the TLS handshake failure; got {}",
            lines[0]
        );
    }

    /// An `https://` export to a peer that accepts the TCP connection and
    /// then never answers the ClientHello fails once the handshake bound
    /// ([`OTLP_EXPORT_TIMEOUT`] in [`otlp_tls_config`]) passes, instead of
    /// hanging. The request timeout cannot stand in for it: it starts only
    /// once the channel has connected, and the handshake is part of the
    /// connect.
    ///
    /// The peer is a listener that is never accepted from: the kernel
    /// completes the TCP handshake in its backlog and nothing is ever
    /// written. Slow by design (> 1 s): the test waits out the 3 s bound,
    /// which is the behaviour under test; [`EXPORT_BUDGET`] is the backstop
    /// that turns a hang into a failure.
    #[tokio::test]
    async fn https_metrics_export_to_a_peer_stalling_the_handshake_fails_within_the_bound() {
        let silent_peer = std::net::TcpListener::bind("127.0.0.1:0").expect("bind the silent peer");
        let addr = silent_peer.local_addr().expect("silent peer address");
        let endpoint =
            otlp_endpoint_from(OTEL_EXPORTER_OTLP_METRICS_ENDPOINT, Some(&https_url(addr)))
                .expect("valid")
                .expect("set");
        let exporter =
            without_compression(|| metric_exporter_for(&endpoint)).expect("the exporter builds");
        let resource_metrics = ResourceMetrics::default();

        let result = tokio::time::timeout(EXPORT_BUDGET, exporter.export(&resource_metrics))
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "an export whose TLS handshake stalls must fail within the handshake bound \
                     ({OTLP_EXPORT_TIMEOUT:?}), not hang past {EXPORT_BUDGET:?}"
                )
            });

        assert!(
            result.is_err(),
            "a handshake that never completes cannot deliver the export"
        );
        drop(silent_peer);
    }

    /// T2, metrics: the same refusal on the push arm's exporter is reported
    /// by one `OTLP metrics export failed` WARN naming the handshake failure.
    #[tokio::test]
    async fn https_metrics_export_refused_at_the_handshake_names_it_in_the_failure_warn() {
        let (addr, _first) = spawn_tls_alert_listener().await;
        let endpoint =
            otlp_endpoint_from(OTEL_EXPORTER_OTLP_METRICS_ENDPOINT, Some(&https_url(addr)))
                .expect("valid")
                .expect("set");
        let reporter = Arc::new(MetricExportReporter::new());
        let exporter = ReportingMetricExporter::new(
            without_compression(|| metric_exporter_for(&endpoint)).expect("the exporter builds"),
            Arc::clone(&reporter),
        );
        let resource_metrics = ResourceMetrics::default();

        let (result, lines) = lines_with_message(
            OTLP_METRICS_EXPORT_FAILED_MSG,
            exporter.export(&resource_metrics),
        )
        .await;

        assert!(result.is_err(), "precondition: the export failed");
        assert_eq!(lines.len(), 1, "one failure line: {lines:?}");
        let line = &lines[0];
        assert_eq!(line["level"], "WARN", "{line}");
        assert_eq!(line["fields"]["failed_exports"], 1, "{line}");
        assert_eq!(line["fields"]["failed_exports_total"], 1, "{line}");
        let error = line["fields"]["last_export_error"]
            .as_str()
            .unwrap_or_default();
        assert!(
            error.contains(HANDSHAKE_FAILURE_FRAGMENT),
            "last_export_error must name the TLS handshake failure; got {line}"
        );
    }
}
