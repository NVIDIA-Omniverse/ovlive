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

//! The suite's configuration: every `CONFORMANCE_*` variable, validated once.
//!
//! Variables are read through a lookup function ([`Settings::from_lookup`]),
//! never by changing the process environment: nothing in this crate calls
//! `std::env::set_var`, and tests pass a map instead. The provisioned base
//! layer replaces `CONFORMANCE_BASE_PATH` through [`Settings::with_base_path`].
//! A value is trimmed. An empty or whitespace-only value counts as unset,
//! because the chart renders its optional variables (the peer endpoint, the
//! denied path, the Storage API endpoint) empty when they are not configured;
//! the exceptions are below.
//!
//! Rules (each refused with a [`ConfigError::Invalid`] naming the variable):
//! - `CONFORMANCE_GRPC_ENDPOINT`, `CONFORMANCE_TIMEOUT` and
//!   `CONFORMANCE_READY_TIMEOUT` may be unset (taking their defaults) but not
//!   set and blank: a blank endpoint would otherwise test
//!   `http://localhost:50001` and fail only after the readiness wait;
//! - an endpoint is `http[s]://host[:port]` with no credentials, path, query
//!   or fragment; the port defaults to 443 for `https` and 50001 for `http`
//!   (also for an explicit port 0); an IPv6 host is bracketed ([`target`]);
//! - exactly one of `CONFORMANCE_AUTH_TOKEN` and `CONFORMANCE_AUTH_TOKEN_FILE`;
//!   the file is re-read for every RPC, may hold JSON `{"access_token": …}`,
//!   and the token must be non-empty printable ASCII with no whitespace;
//! - `CONFORMANCE_BASE_PATH` is required;
//! - `CONFORMANCE_TIMEOUT` (default 15) and `CONFORMANCE_READY_TIMEOUT`
//!   (default 120) are finite, positive seconds, at most one day
//!   ([`MAX_TIMEOUT`]);
//! - `CONFORMANCE_CA_FILE` replaces the default trust roots of every TLS
//!   connection the run makes (the gRPC endpoints and the Storage API), and is
//!   refused only when none of them uses TLS.

use std::fmt;
use std::net::Ipv6Addr;
use std::path::PathBuf;
use std::time::Duration;

/// Reads one variable: `None` when unset. [`process_env`] in production, a map
/// in tests.
pub type Lookup<'a> = &'a dyn Fn(&str) -> Option<String>;

/// The process environment as a [`Lookup`]. A variable that is not valid
/// Unicode reads as unset.
pub fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// A refused configuration. Its text names the variable and the rule and never
/// contains a credential.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    /// `CONFORMANCE_AUTH_TOKEN_FILE` names a file that does not exist yet. A
    /// sidecar may still be writing it: [`Settings::wait_for_token`] waits for
    /// it up to the readiness timeout.
    TokenFileMissing,
    /// Any other refusal, with its message.
    Invalid(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenFileMissing => f.write_str("CONFORMANCE_AUTH_TOKEN_FILE does not exist"),
            Self::Invalid(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ConfigError {}

/// A validated gRPC endpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    /// `host:port`, the host bracketed when it is an IPv6 address
    /// (`[::1]:443`).
    pub address: String,
    /// Whether the endpoint is `https` (TLS) rather than `http` (h2c).
    pub tls: bool,
}

/// The refusal of a malformed endpoint. It never repeats the endpoint, which
/// may carry credentials.
const ENDPOINT_RULE: &str =
    "endpoint must be http[s]://host:port without credentials, path, query or fragment";

/// Validates an `http[s]://host[:port]` endpoint (see the module rules). The
/// host is lowercased. The error never repeats `url`.
pub fn target(url: &str) -> Result<Target, ConfigError> {
    let refused = || ConfigError::Invalid(ENDPOINT_RULE.to_string());
    let (scheme, rest) = url.split_once("://").ok_or_else(refused)?;
    let tls = if scheme.eq_ignore_ascii_case("https") {
        true
    } else if scheme.eq_ignore_ascii_case("http") {
        false
    } else {
        return Err(refused());
    };

    // As a URL splits: the authority ends at the first `/`, `?` or `#`; the
    // fragment follows the first `#`, the query the first `?` before it.
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(authority_end);
    let (before_fragment, fragment) = tail.split_once('#').unwrap_or((tail, ""));
    let (path, query) = before_fragment
        .split_once('?')
        .unwrap_or((before_fragment, ""));
    if !(path.is_empty() || path == "/")
        || !query.is_empty()
        || !fragment.is_empty()
        || authority.contains('@')
    {
        return Err(refused());
    }

    let (host, port) = match authority.strip_prefix('[') {
        Some(bracketed) => {
            let (host, after) = bracketed.split_once(']').ok_or_else(refused)?;
            host.parse::<Ipv6Addr>().map_err(|_| refused())?;
            let port = if after.is_empty() {
                ""
            } else {
                after.strip_prefix(':').ok_or_else(refused)?
            };
            (host, port)
        }
        None if authority.contains(['[', ']']) => return Err(refused()),
        None => authority.split_once(':').unwrap_or((authority, "")),
    };
    if host.is_empty() {
        return Err(refused());
    }
    let port = match port {
        "" => 0,
        digits if digits.bytes().all(|b| b.is_ascii_digit()) => digits
            .parse::<u32>()
            .ok()
            .filter(|port| *port <= 65535)
            .ok_or_else(|| {
                ConfigError::Invalid("endpoint port must be in the range 0-65535".to_string())
            })?,
        _ => return Err(refused()),
    };
    // An absent port, or port 0, takes the scheme's default.
    let port = match port {
        0 if tls => 443,
        0 => 50001,
        port => port,
    };
    let host = host.to_ascii_lowercase();
    let host = if host.contains(':') {
        format!("[{host}]")
    } else {
        host
    };
    let address = format!("{host}:{port}");
    // What the gRPC transport will be given must parse as a URI authority.
    if format!("http://{address}").parse::<http::Uri>().is_err() {
        return Err(refused());
    }
    Ok(Target { address, tls })
}

/// Where the bearer token comes from. Its `Debug` never shows the token.
///
/// Private, so `Settings` can only be built by [`Settings::from_lookup`].
#[derive(Clone)]
enum Credential {
    /// `CONFORMANCE_AUTH_TOKEN`.
    Token(String),
    /// `CONFORMANCE_AUTH_TOKEN_FILE`, re-read for every RPC.
    File(PathBuf),
}

impl fmt::Debug for Credential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Token(_) => f.write_str("Token(<redacted>)"),
            Self::File(path) => f.debug_tuple("File").field(path).finish(),
        }
    }
}

/// The validated configuration of one run.
#[derive(Clone, Debug)]
pub struct Settings {
    /// `CONFORMANCE_GRPC_ENDPOINT` (default `http://localhost:50001`),
    /// validated by [`target`].
    pub endpoint: String,
    /// The base layer path sessions are created for: `CONFORMANCE_BASE_PATH`,
    /// or the provisioned file ([`Settings::with_base_path`]).
    pub base_path: String,
    credential: Credential,
    /// `CONFORMANCE_CA_FILE`: replaces the default roots of every TLS
    /// connection. `None`: gRPC trusts the system store and the Storage API
    /// trusts reqwest's built-in roots, as the service's own clients do.
    pub ca_file: Option<PathBuf>,
    /// `CONFORMANCE_TIMEOUT`: the deadline of each unary RPC, and of one
    /// complete streamed delivery or awaited stream message.
    pub timeout: Duration,
    /// `CONFORMANCE_READY_TIMEOUT`: how long the preflight waits for the token
    /// file and for the endpoints to become ready.
    pub ready_timeout: Duration,
    /// `CONFORMANCE_REPORT_DIR` (default `reports`), as [`report_dir`] reads it.
    pub report_dir: PathBuf,
    /// `CONFORMANCE_PEER_ENDPOINT`: a second replica, for the `multi_replica`
    /// profile; validated by [`target`] when set.
    pub peer_endpoint: Option<String>,
    /// `CONFORMANCE_DENIED_PATH`: a base path the credential may not read, for
    /// the `permissions` profile.
    pub denied_path: Option<String>,
    /// `CONFORMANCE_STORAGE_ENDPOINT`: when set, the run provisions its own
    /// base layer through this Storage API (`crate::deployment`).
    pub storage_endpoint: Option<String>,
}

/// The variable `name`, trimmed; `None` when it is unset, empty or only
/// whitespace.
fn var(lookup: Lookup<'_>, name: &str) -> Option<String> {
    lookup(name)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// The variable `name`, trimmed; `None` when it is unset. Set but empty or
/// only whitespace is refused, naming `name`: a blank value here is a
/// mistake (the chart's `service.endpoint: ""`, say), not a request for the
/// default.
fn var_not_blank(lookup: Lookup<'_>, name: &str) -> Result<Option<String>, ConfigError> {
    match lookup(name) {
        None => Ok(None),
        Some(value) if value.trim().is_empty() => Err(ConfigError::Invalid(format!(
            "{name} is set but empty; unset it to use the default"
        ))),
        Some(value) => Ok(Some(value.trim().to_string())),
    }
}

/// [`target`] of the variable `name`'s value, the refusal naming `name`.
fn endpoint_of(name: &str, url: &str) -> Result<Target, ConfigError> {
    target(url).map_err(|error| ConfigError::Invalid(format!("{name}: {error}")))
}

/// The largest `CONFORMANCE_TIMEOUT` or `CONFORMANCE_READY_TIMEOUT`: one day.
///
/// The timeouts are sent as `grpc-timeout` headers, which tonic panics on
/// above 99,999,999 hours, and added to `Instant`s, which panic on overflow;
/// a readiness panic would end the run with no reports. One day is far below
/// both limits and still far above any useful wait: the chart's Job is
/// stopped after 900 s by default, and the defaults are 15 s and 120 s.
pub const MAX_TIMEOUT: Duration = Duration::from_secs(86_400);

/// The seconds in the variable `name` (or `default` when it is unset), which
/// must be finite, positive and at most [`MAX_TIMEOUT`]; a blank value is
/// refused.
fn positive(lookup: Lookup<'_>, name: &str, default: u64) -> Result<Duration, ConfigError> {
    let Some(value) = var_not_blank(lookup, name)? else {
        return Ok(Duration::from_secs(default));
    };
    let seconds = value
        .parse::<f64>()
        .ok()
        .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
        .ok_or_else(|| ConfigError::Invalid(format!("{name} must be finite and positive")))?;
    // Checked on the seconds, before the conversion: a value too large for a
    // `Duration` still gets this message, not the one above.
    if seconds > MAX_TIMEOUT.as_secs_f64() {
        return Err(ConfigError::Invalid(format!(
            "{name} must be at most {} seconds (one day)",
            MAX_TIMEOUT.as_secs()
        )));
    }
    Duration::try_from_secs_f64(seconds)
        .ok()
        .filter(|duration| !duration.is_zero())
        .ok_or_else(|| ConfigError::Invalid(format!("{name} must be finite and positive")))
}

/// Whether a Storage API endpoint uses TLS: its scheme is `https`.
fn is_https(url: &str) -> bool {
    url.split_once("://")
        .is_some_and(|(scheme, _)| scheme.eq_ignore_ascii_case("https"))
}

/// Checks a bearer token: non-empty printable ASCII (`!` to `~`), so no
/// whitespace or control character.
fn valid_token(token: &str) -> bool {
    !token.is_empty() && token.bytes().all(|b| (33..=126).contains(&b))
}

impl Settings {
    /// Reads and validates the process environment.
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_lookup(&process_env)
    }

    /// Reads and validates the variables `lookup` returns. An empty or
    /// whitespace-only value is refused for `CONFORMANCE_GRPC_ENDPOINT`,
    /// `CONFORMANCE_TIMEOUT` and `CONFORMANCE_READY_TIMEOUT`, and counts as
    /// unset for every other variable (see the module doc).
    pub fn from_lookup(lookup: Lookup<'_>) -> Result<Self, ConfigError> {
        let endpoint = var_not_blank(lookup, "CONFORMANCE_GRPC_ENDPOINT")?
            .unwrap_or_else(|| "http://localhost:50001".to_string());
        let grpc = endpoint_of("CONFORMANCE_GRPC_ENDPOINT", &endpoint)?;

        let credential = match (
            var(lookup, "CONFORMANCE_AUTH_TOKEN"),
            var(lookup, "CONFORMANCE_AUTH_TOKEN_FILE"),
        ) {
            (Some(token), None) => Credential::Token(token),
            (None, Some(file)) => Credential::File(PathBuf::from(file)),
            _ => {
                return Err(ConfigError::Invalid(
                    "Set exactly one of CONFORMANCE_AUTH_TOKEN or CONFORMANCE_AUTH_TOKEN_FILE"
                        .to_string(),
                ))
            }
        };

        let base_path = var(lookup, "CONFORMANCE_BASE_PATH").ok_or_else(|| {
            ConfigError::Invalid(
                "CONFORMANCE_BASE_PATH must name an existing authorized base layer".to_string(),
            )
        })?;

        let peer_endpoint = var(lookup, "CONFORMANCE_PEER_ENDPOINT");
        let peer_tls = match &peer_endpoint {
            Some(peer) => endpoint_of("CONFORMANCE_PEER_ENDPOINT", peer)?.tls,
            None => false,
        };
        let storage_endpoint = var(lookup, "CONFORMANCE_STORAGE_ENDPOINT");
        let storage_tls = storage_endpoint.as_deref().is_some_and(is_https);

        let ca_file = var(lookup, "CONFORMANCE_CA_FILE").map(PathBuf::from);
        if ca_file.is_some() && !(grpc.tls || peer_tls || storage_tls) {
            return Err(ConfigError::Invalid(
                "CONFORMANCE_CA_FILE requires a TLS connection: an https \
                 CONFORMANCE_GRPC_ENDPOINT, CONFORMANCE_PEER_ENDPOINT or \
                 CONFORMANCE_STORAGE_ENDPOINT"
                    .to_string(),
            ));
        }

        Ok(Settings {
            endpoint,
            base_path,
            credential,
            ca_file,
            timeout: positive(lookup, "CONFORMANCE_TIMEOUT", 15)?,
            ready_timeout: positive(lookup, "CONFORMANCE_READY_TIMEOUT", 120)?,
            report_dir: report_dir(lookup),
            peer_endpoint,
            denied_path: var(lookup, "CONFORMANCE_DENIED_PATH"),
            storage_endpoint,
        })
    }

    /// These settings with `base_path` as the base layer: the file the run
    /// provisioned.
    pub fn with_base_path(mut self, base_path: impl Into<String>) -> Self {
        self.base_path = base_path.into();
        self
    }

    /// The `authorization` header value for one RPC or Storage API request,
    /// `Bearer <token>`. The token file is read again on every call, so a
    /// rotated token is picked up. [`ConfigError::TokenFileMissing`] when the
    /// file does not exist.
    pub fn authorization(&self) -> Result<String, ConfigError> {
        let token = match &self.credential {
            Credential::Token(token) => token.clone(),
            Credential::File(path) => match std::fs::read_to_string(path) {
                Ok(contents) => contents.trim().to_string(),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Err(ConfigError::TokenFileMissing)
                }
                Err(_) => {
                    return Err(ConfigError::Invalid(
                        "CONFORMANCE_AUTH_TOKEN_FILE could not be read as text".to_string(),
                    ))
                }
            },
        };
        // A token that starts like a JSON object is one: its access_token.
        let token = if token.starts_with('{') {
            let refused =
                || ConfigError::Invalid("Token JSON must contain a string access_token".into());
            let value: serde_json::Value = serde_json::from_str(&token).map_err(|_| refused())?;
            value
                .get("access_token")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(refused)?
                .to_string()
        } else {
            token
        };
        if !valid_token(&token) {
            return Err(ConfigError::Invalid(
                "Bearer token must be nonempty printable ASCII without whitespace".to_string(),
            ));
        }
        Ok(format!("Bearer {token}"))
    }

    /// Waits up to [`Settings::ready_timeout`] for the token file to appear
    /// (a sidecar may write it after the suite starts), then checks the token
    /// is valid. Returns at once with a token from `CONFORMANCE_AUTH_TOKEN`.
    /// [`ConfigError::TokenFileMissing`] when the file never appeared.
    pub async fn wait_for_token(&self) -> Result<(), ConfigError> {
        let deadline = tokio::time::Instant::now() + self.ready_timeout;
        loop {
            match self.authorization() {
                Err(ConfigError::TokenFileMissing) => {
                    if tokio::time::Instant::now() >= deadline {
                        return Err(ConfigError::TokenFileMissing);
                    }
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
                other => return other.map(|_| ()),
            }
        }
    }

    /// The CA bundle to trust instead of the default roots, read from
    /// [`Settings::ca_file`]; `None` when it is unset.
    pub fn ca_pem(&self) -> Result<Option<Vec<u8>>, ConfigError> {
        match &self.ca_file {
            None => Ok(None),
            Some(path) => std::fs::read(path).map(Some).map_err(|_| {
                ConfigError::Invalid("CONFORMANCE_CA_FILE could not be read".to_string())
            }),
        }
    }
}

/// `CONFORMANCE_REPORT_DIR`, default `reports` (relative to the working
/// directory). Read on its own because reports are written even when the rest
/// of the configuration is refused.
pub fn report_dir(lookup: Lookup<'_>) -> PathBuf {
    PathBuf::from(var(lookup, "CONFORMANCE_REPORT_DIR").unwrap_or_else(|| "reports".to_string()))
}

/// Whether `CONFORMANCE_REPORT_STDOUT` is exactly `true`: print each report
/// as a `CONFORMANCE_REPORT <name> <base64>` line too.
pub fn report_stdout(lookup: Lookup<'_>) -> bool {
    lookup("CONFORMANCE_REPORT_STDOUT").as_deref() == Some("true")
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;

    use super::*;

    /// The variables every valid configuration needs.
    const REQUIRED: [(&str, &str); 2] = [
        ("CONFORMANCE_AUTH_TOKEN", "local-token"),
        ("CONFORMANCE_BASE_PATH", "/container/base.usda"),
    ];

    /// [`Settings::from_lookup`] over [`REQUIRED`] with `vars` on top (a
    /// later entry replaces an earlier one; an empty value is a set-but-empty
    /// variable, not an unset one).
    fn settings(vars: &[(&str, &str)]) -> Result<Settings, ConfigError> {
        settings_without(&[], vars)
    }

    /// As [`settings`], with the [`REQUIRED`] variables named in `unset` left
    /// out.
    fn settings_without(unset: &[&str], vars: &[(&str, &str)]) -> Result<Settings, ConfigError> {
        let mut map: HashMap<String, String> = REQUIRED
            .iter()
            .filter(|(name, _)| !unset.contains(name))
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect();
        map.extend(
            vars.iter()
                .map(|(name, value)| (name.to_string(), value.to_string())),
        );
        Settings::from_lookup(&|name| map.get(name).cloned())
    }

    fn invalid_message(result: Result<Settings, ConfigError>) -> String {
        match result {
            Err(ConfigError::Invalid(message)) => message,
            Err(other) => panic!("expected an Invalid refusal, got {other:?}"),
            Ok(settings) => panic!("expected a refusal, got {settings:?}"),
        }
    }

    #[test]
    fn target_malformed_endpoint_is_refused_without_repeating_it() {
        for url in [
            // The Python suite's six.
            "localhost:50001",
            "ftp://host",
            "https://user:pass@host",
            "http://host/path",
            "https://host?token=x",
            "http://host:99999",
            // A fragment, an empty host, a non-numeric port, broken brackets.
            "http://host#fragment",
            "http://:50001",
            "http://host:port",
            "https://[::1",
            "http://[not-an-address]:1",
            "http://[::1]x",
            "http://host]:1",
        ] {
            match target(url) {
                Err(ConfigError::Invalid(message)) => {
                    assert!(!message.contains(url), "{url:?} is repeated in {message:?}");
                    assert!(
                        !message.contains("pass"),
                        "{message:?} carries a credential"
                    );
                }
                other => panic!("{url:?} must be refused, got {other:?}"),
            }
        }
    }

    #[test]
    fn target_ipv6_host_is_bracketed_and_https_is_tls() {
        assert_eq!(
            target("https://[::1]:443"),
            Ok(Target {
                address: "[::1]:443".into(),
                tls: true
            })
        );
        assert_eq!(
            target("http://service:50001"),
            Ok(Target {
                address: "service:50001".into(),
                tls: false
            })
        );
        assert_eq!(
            target("https://[2001:DB8::1]"),
            Ok(Target {
                address: "[2001:db8::1]:443".into(),
                tls: true
            })
        );
    }

    #[test]
    fn target_absent_or_zero_port_takes_the_scheme_default() {
        // 50001 for http, not the 80 a URL parser defaults to.
        for (url, address, tls) in [
            ("http://service", "service:50001", false),
            ("http://service:0", "service:50001", false),
            ("https://service", "service:443", true),
            ("https://service:0", "service:443", true),
            (
                "HTTPS://Service.Example.com/",
                "service.example.com:443",
                true,
            ),
            ("http://[::1]", "[::1]:50001", false),
        ] {
            assert_eq!(
                target(url),
                Ok(Target {
                    address: address.into(),
                    tls
                }),
                "{url}"
            );
        }
    }

    #[test]
    fn from_lookup_unset_optional_variables_take_their_defaults() {
        let settings = settings(&[]).expect("valid");
        assert_eq!(settings.endpoint, "http://localhost:50001");
        assert_eq!(settings.base_path, "/container/base.usda");
        assert_eq!(settings.timeout, Duration::from_secs(15));
        assert_eq!(settings.ready_timeout, Duration::from_secs(120));
        assert_eq!(settings.report_dir, PathBuf::from("reports"));
        assert_eq!(settings.ca_file, None);
        assert_eq!(settings.peer_endpoint, None);
        assert_eq!(settings.denied_path, None);
        assert_eq!(settings.storage_endpoint, None);
    }

    #[test]
    fn from_lookup_timeout_that_is_not_finite_and_positive_is_refused() {
        // `1e-10` parses as a positive float but rounds to a zero Duration.
        for name in ["CONFORMANCE_TIMEOUT", "CONFORMANCE_READY_TIMEOUT"] {
            for value in ["0", "-1", "nan", "inf", "1e-10", "-0", "fifteen"] {
                let message = invalid_message(settings(&[(name, value)]));
                assert_eq!(
                    message,
                    format!("{name} must be finite and positive"),
                    "{name}={value}"
                );
            }
        }
    }

    #[test]
    fn from_lookup_timeout_of_exactly_one_day_is_accepted() {
        assert_eq!(MAX_TIMEOUT, Duration::from_secs(86_400));
        let settings = settings(&[
            ("CONFORMANCE_TIMEOUT", "86400"),
            ("CONFORMANCE_READY_TIMEOUT", "86400"),
        ])
        .expect("one day is within the cap");
        assert_eq!(settings.timeout, Duration::from_secs(86_400));
        assert_eq!(settings.ready_timeout, Duration::from_secs(86_400));
    }

    #[test]
    fn from_lookup_timeout_over_one_day_is_refused_with_the_cap_message() {
        // `1e20` seconds does not fit a `Duration`: it still gets the cap
        // message, not "finite and positive".
        for name in ["CONFORMANCE_TIMEOUT", "CONFORMANCE_READY_TIMEOUT"] {
            for value in ["86400.001", "86401", "1e12", "1e20"] {
                let message = invalid_message(settings(&[(name, value)]));
                assert_eq!(
                    message,
                    format!("{name} must be at most 86400 seconds (one day)"),
                    "{name}={value}"
                );
            }
        }
    }

    #[test]
    fn from_lookup_fractional_timeouts_are_kept() {
        let settings = settings(&[
            ("CONFORMANCE_TIMEOUT", "0.5"),
            ("CONFORMANCE_READY_TIMEOUT", "2.25"),
        ])
        .expect("valid");
        assert_eq!(settings.timeout, Duration::from_millis(500));
        assert_eq!(settings.ready_timeout, Duration::from_millis(2250));
    }

    #[test]
    fn from_lookup_both_credentials_set_is_refused() {
        let message = invalid_message(settings(&[(
            "CONFORMANCE_AUTH_TOKEN_FILE",
            "/var/run/token",
        )]));
        assert!(message.contains("exactly one"), "{message}");
    }

    #[test]
    fn from_lookup_no_credential_set_is_refused() {
        let message = invalid_message(settings_without(&["CONFORMANCE_AUTH_TOKEN"], &[]));
        assert!(message.contains("exactly one"), "{message}");
    }

    #[test]
    fn from_lookup_missing_base_path_is_refused() {
        let message = invalid_message(settings_without(&["CONFORMANCE_BASE_PATH"], &[]));
        assert!(message.contains("CONFORMANCE_BASE_PATH"), "{message}");
    }

    /// The variables a set-but-blank value is refused for, rather than read
    /// as unset.
    const NOT_BLANK: [&str; 3] = [
        "CONFORMANCE_GRPC_ENDPOINT",
        "CONFORMANCE_TIMEOUT",
        "CONFORMANCE_READY_TIMEOUT",
    ];

    /// Empty and whitespace-only values.
    const BLANKS: [&str; 4] = ["", " ", "\t", " \n "];

    #[test]
    fn from_lookup_blank_optional_values_count_as_unset() {
        // A blank token beside a token file is no second credential, and a
        // blank CA file beside plaintext endpoints is no CA file to refuse.
        let blank = settings(&[
            ("CONFORMANCE_AUTH_TOKEN", "   "),
            ("CONFORMANCE_AUTH_TOKEN_FILE", "/var/run/token"),
            ("CONFORMANCE_CA_FILE", "\t"),
            ("CONFORMANCE_PEER_ENDPOINT", " "),
            ("CONFORMANCE_DENIED_PATH", ""),
            ("CONFORMANCE_STORAGE_ENDPOINT", "  "),
            ("CONFORMANCE_REPORT_DIR", " "),
        ])
        .expect("blank values are unset");
        assert!(
            matches!(&blank.credential, Credential::File(path) if path == Path::new("/var/run/token"))
        );
        assert_eq!(blank.ca_file, None);
        assert_eq!(blank.peer_endpoint, None);
        assert_eq!(blank.denied_path, None);
        assert_eq!(blank.storage_endpoint, None);
        assert_eq!(blank.report_dir, PathBuf::from("reports"));

        // The other way round: a blank token file beside a token.
        let blank_file =
            settings(&[("CONFORMANCE_AUTH_TOKEN_FILE", " ")]).expect("a blank token file is unset");
        assert!(
            matches!(&blank_file.credential, Credential::Token(token) if token == "local-token")
        );
    }

    #[test]
    fn from_lookup_blank_base_path_is_refused_as_missing() {
        for blank in BLANKS {
            let message = invalid_message(settings(&[("CONFORMANCE_BASE_PATH", blank)]));
            assert_eq!(
                message, "CONFORMANCE_BASE_PATH must name an existing authorized base layer",
                "{blank:?}"
            );
        }
    }

    #[test]
    fn from_lookup_blank_endpoint_or_timeout_is_refused_naming_the_variable() {
        // Unset takes the default; set but blank is a mistake, refused.
        for name in NOT_BLANK {
            for blank in BLANKS {
                let message = invalid_message(settings(&[(name, blank)]));
                assert_eq!(
                    message,
                    format!("{name} is set but empty; unset it to use the default"),
                    "{name}={blank:?}"
                );
            }
        }
    }

    #[test]
    fn from_lookup_chart_default_environment_is_accepted() {
        // What `helm/live-edit-conformance-tests` renders with its default
        // values (and a base path, which it requires): the optional
        // variables are set but empty, the timeouts quoted numbers.
        let settings = settings_without(
            &["CONFORMANCE_AUTH_TOKEN"],
            &[
                (
                    "CONFORMANCE_GRPC_ENDPOINT",
                    "http://live-edit-service:50001",
                ),
                ("CONFORMANCE_PEER_ENDPOINT", ""),
                ("CONFORMANCE_DENIED_PATH", ""),
                ("CONFORMANCE_STORAGE_ENDPOINT", ""),
                ("CONFORMANCE_REPORT_STDOUT", "true"),
                ("CONFORMANCE_AUTH_TOKEN_FILE", "/auth/token"),
                ("CONFORMANCE_TIMEOUT", "15"),
                ("CONFORMANCE_READY_TIMEOUT", "120"),
            ],
        )
        .expect("the chart's default environment is valid");
        assert_eq!(settings.endpoint, "http://live-edit-service:50001");
        assert!(
            matches!(&settings.credential, Credential::File(path) if path == Path::new("/auth/token"))
        );
        assert_eq!(settings.timeout, Duration::from_secs(15));
        assert_eq!(settings.ready_timeout, Duration::from_secs(120));
        assert_eq!(settings.peer_endpoint, None);
        assert_eq!(settings.denied_path, None);
        assert_eq!(settings.storage_endpoint, None);
        assert_eq!(settings.ca_file, None);
    }

    #[test]
    fn from_lookup_values_are_trimmed() {
        let settings = settings(&[
            ("CONFORMANCE_BASE_PATH", "  /container/base.usda\n"),
            ("CONFORMANCE_GRPC_ENDPOINT", " http://service:50001 "),
        ])
        .expect("valid");
        assert_eq!(settings.base_path, "/container/base.usda");
        assert_eq!(settings.endpoint, "http://service:50001");
    }

    #[test]
    fn from_lookup_malformed_endpoints_are_refused_naming_their_variable() {
        for name in ["CONFORMANCE_GRPC_ENDPOINT", "CONFORMANCE_PEER_ENDPOINT"] {
            let message = invalid_message(settings(&[(name, "service:50001")]));
            assert!(message.starts_with(name), "{message}");
        }
    }

    #[test]
    fn from_lookup_ca_file_with_plaintext_grpc_and_https_storage_is_accepted() {
        // b.cva: the CA file is for whichever connection uses TLS.
        let settings = settings(&[
            ("CONFORMANCE_GRPC_ENDPOINT", "http://service:50001"),
            (
                "CONFORMANCE_STORAGE_ENDPOINT",
                "https://storage.example.com",
            ),
            ("CONFORMANCE_CA_FILE", "/etc/conformance/ca.pem"),
        ])
        .expect("a CA file is accepted when the Storage API uses TLS");
        assert_eq!(
            settings.ca_file,
            Some(PathBuf::from("/etc/conformance/ca.pem"))
        );
    }

    #[test]
    fn from_lookup_ca_file_with_any_tls_connection_is_accepted() {
        for tls in [
            [
                ("CONFORMANCE_GRPC_ENDPOINT", "https://service:443"),
                ("CONFORMANCE_PEER_ENDPOINT", "http://peer:50001"),
            ],
            [
                ("CONFORMANCE_GRPC_ENDPOINT", "http://service:50001"),
                ("CONFORMANCE_PEER_ENDPOINT", "https://peer:443"),
            ],
        ] {
            let mut vars = tls.to_vec();
            vars.push(("CONFORMANCE_CA_FILE", "/etc/conformance/ca.pem"));
            assert!(settings(&vars).is_ok(), "{tls:?}");
        }
    }

    #[test]
    fn from_lookup_ca_file_with_plaintext_everywhere_is_refused() {
        for vars in [
            vec![("CONFORMANCE_GRPC_ENDPOINT", "http://service:50001")],
            vec![
                ("CONFORMANCE_GRPC_ENDPOINT", "http://service:50001"),
                ("CONFORMANCE_PEER_ENDPOINT", "http://peer:50001"),
                ("CONFORMANCE_STORAGE_ENDPOINT", "http://storage.example.com"),
            ],
        ] {
            let mut vars = vars;
            vars.push(("CONFORMANCE_CA_FILE", "/etc/conformance/ca.pem"));
            let message = invalid_message(settings(&vars));
            assert!(message.contains("CONFORMANCE_CA_FILE"), "{message}");
        }
    }

    #[test]
    fn settings_debug_never_shows_the_token() {
        let settings = settings(&[("CONFORMANCE_AUTH_TOKEN", "secret-token-value")]).unwrap();
        assert!(!format!("{settings:?}").contains("secret-token-value"));
    }

    #[test]
    fn authorization_rotating_token_file_is_read_on_every_call() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        std::fs::write(&path, "first").unwrap();
        let settings = settings_without(
            &["CONFORMANCE_AUTH_TOKEN"],
            &[("CONFORMANCE_AUTH_TOKEN_FILE", path.to_str().unwrap())],
        )
        .unwrap();
        assert_eq!(settings.authorization(), Ok("Bearer first".to_string()));

        std::fs::write(&path, "second\n").unwrap();
        assert_eq!(settings.authorization(), Ok("Bearer second".to_string()));

        std::fs::write(&path, "bad\nheader").unwrap();
        assert!(matches!(
            settings.authorization(),
            Err(ConfigError::Invalid(_))
        ));
    }

    #[test]
    fn authorization_json_token_file_yields_its_access_token() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token.json");
        let settings = settings_without(
            &["CONFORMANCE_AUTH_TOKEN"],
            &[("CONFORMANCE_AUTH_TOKEN_FILE", path.to_str().unwrap())],
        )
        .unwrap();

        std::fs::write(&path, r#"{"access_token":"rotating"}"#).unwrap();
        assert_eq!(settings.authorization(), Ok("Bearer rotating".to_string()));

        for refused in [
            r#"{"token":"rotating"}"#,
            r#"{"access_token":5}"#,
            "{not json",
        ] {
            std::fs::write(&path, refused).unwrap();
            assert!(
                matches!(settings.authorization(), Err(ConfigError::Invalid(_))),
                "{refused}"
            );
        }
    }

    #[test]
    fn authorization_missing_token_file_is_token_file_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-yet-written");
        let settings = settings_without(
            &["CONFORMANCE_AUTH_TOKEN"],
            &[("CONFORMANCE_AUTH_TOKEN_FILE", path.to_str().unwrap())],
        )
        .unwrap();
        assert_eq!(settings.authorization(), Err(ConfigError::TokenFileMissing));
    }

    #[test]
    fn authorization_token_with_whitespace_or_non_ascii_is_refused() {
        for token in ["two words", "tab\tinside", "café"] {
            let settings = settings(&[("CONFORMANCE_AUTH_TOKEN", token)]).unwrap();
            assert!(
                matches!(settings.authorization(), Err(ConfigError::Invalid(_))),
                "{token:?}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn wait_for_token_file_that_never_appears_fails_at_the_ready_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        let settings = settings_without(
            &["CONFORMANCE_AUTH_TOKEN"],
            &[
                ("CONFORMANCE_AUTH_TOKEN_FILE", path.to_str().unwrap()),
                ("CONFORMANCE_READY_TIMEOUT", "3"),
            ],
        )
        .unwrap();
        let started = tokio::time::Instant::now();
        assert_eq!(
            settings.wait_for_token().await,
            Err(ConfigError::TokenFileMissing)
        );
        assert!(started.elapsed() >= Duration::from_secs(3));
    }

    #[tokio::test(start_paused = true)]
    async fn wait_for_token_file_written_late_is_waited_for() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("token");
        let settings = settings_without(
            &["CONFORMANCE_AUTH_TOKEN"],
            &[
                ("CONFORMANCE_AUTH_TOKEN_FILE", path.to_str().unwrap()),
                ("CONFORMANCE_READY_TIMEOUT", "30"),
            ],
        )
        .unwrap();
        // A sidecar writing the token 2 s (of paused time) after the start.
        let writer = tokio::spawn({
            let path = path.clone();
            async move {
                tokio::time::sleep(Duration::from_secs(2)).await;
                std::fs::write(path, "late-token").unwrap();
            }
        });
        assert_eq!(settings.wait_for_token().await, Ok(()));
        writer.await.unwrap();
        assert_eq!(
            settings.authorization(),
            Ok("Bearer late-token".to_string())
        );
    }

    #[test]
    fn report_stdout_is_enabled_only_by_exactly_true() {
        for (value, expected) in [
            (Some("true"), true),
            (Some("TRUE"), false),
            (Some("1"), false),
            (Some(" true"), false),
            (None, false),
        ] {
            let lookup = |name: &str| {
                (name == "CONFORMANCE_REPORT_STDOUT")
                    .then(|| value.map(str::to_string))
                    .flatten()
            };
            assert_eq!(report_stdout(&lookup), expected, "{value:?}");
        }
    }
}
