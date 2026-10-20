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

//! Integration test for `Config::from_env`'s fail-closed endpoint
//! validation (b.39x item 6 / test gap H8).
//!
//! Two posture rules are exercised end-to-end through `Config::from_env`
//! rather than through the internal `normalize_endpoint` helper alone:
//!
//! 1. **Bare hostnames are rejected.** Pre-b.ohw the path silently rewrote
//!    `storage.prod.internal` → `http://storage.prod.internal`, downgrading
//!    the auth channel to plaintext and leaking bearer tokens on the wire.
//!    The fix (b.ohw C-6 / H-auth-1) requires an explicit scheme; this
//!    test pins that posture at the public config-loading boundary so a
//!    future refactor that loses the `validate_auth_endpoint_uri` call
//!    inside `from_env` is caught immediately.
//!
//! 2. **The `:443` substring trap is closed.** Pre-b.ohw classification
//!    used `value.contains(":443")` which incorrectly flipped every port
//!    in the range 4430..=4439 (and 44300..=44399, …) to TLS. The fix
//!    parses the right-most `:` and matches the port string by exact
//!    equality. The lower-level test in
//!    `src/auth/mod.rs::normalize_endpoint_tests` covers the helper; this
//!    integration test confirms that an endpoint with `host:4430`
//!    survives `Config::from_env` (i.e. the substring-trap fix is
//!    actually reachable through the public config-loading API).
//!
//! Uses `temp_env::with_vars` (same pattern as the in-tree unit tests
//! that already cover `from_env`'s scheme validation) so env-var mutation
//! is process-scoped and the test doesn't leak state across the binary.

use live_edit::config::{Config, PodMemorySource};

/// Each entry is one bad endpoint value that must trip
/// `validate_auth_endpoint_uri` inside `Config::from_env`. The `var`
/// column names the offending env var; the `value` column is the
/// rejected input; the `reason_fragment` column is a substring the error
/// message must contain so a regression in error wording is caught here
/// rather than only at the unit-test layer.
struct BadEndpointCase {
    var: &'static str,
    value: &'static str,
    reason_fragment: &'static str,
}

const BAD_ENDPOINT_CASES: &[BadEndpointCase] = &[
    BadEndpointCase {
        var: "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
        value: "storage.prod.internal",
        reason_fragment: "explicit scheme",
    },
    BadEndpointCase {
        var: "LIVE_EDIT_USERINFO_ENDPOINT",
        value: "userinfo.prod.internal",
        reason_fragment: "explicit scheme",
    },
    // IP-only bare-host shape (no scheme) — same family of typo, separate
    // input shape so a future regex-style validator that handles
    // dot-separated hostnames but not IPv4 literals is still caught.
    BadEndpointCase {
        var: "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
        value: "10.0.0.1:8080",
        reason_fragment: "explicit scheme",
    },
];

/// Build a sane env-var map and overlay the per-case override. Returning
/// `Vec<(&'static str, Option<&'static str>)>` keeps the signature
/// compatible with `temp_env::with_vars` without a lifetime juggle.
///
/// `LIVE_EDIT_USERINFO_SERVICE_ENABLED=true` is pinned explicitly so the
/// scheme-validation branch on `LIVE_EDIT_USERINFO_ENDPOINT` actually runs
/// — the b.g3z change makes that validation conditional on userinfo being
/// enabled, and the default flipped to `false`, so the
/// `bare_hostname_endpoints_rejected_for_each_var` test would silently
/// lose coverage on the userinfo arm without this pin.
fn base_env() -> Vec<(&'static str, Option<&'static str>)> {
    vec![
        ("LIVE_EDIT_AUTHZ_ENABLED", Some("true")),
        ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("true")),
        (
            "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
            Some("https://storage.prod.internal"),
        ),
        (
            "LIVE_EDIT_USERINFO_ENDPOINT",
            Some("https://userinfo.prod.internal"),
        ),
    ]
}

#[test]
fn bare_hostname_endpoints_rejected_for_each_var() {
    // Loop over the case table inside the test body — one
    // `temp_env::with_vars` scope per case so env-var mutations cannot
    // leak across cases.
    for case in BAD_ENDPOINT_CASES {
        let mut env = base_env();
        // Replace the existing entry for `case.var` with the bad value.
        for entry in env.iter_mut() {
            if entry.0 == case.var {
                entry.1 = Some(case.value);
            }
        }
        temp_env::with_vars(env, || {
            let err = Config::from_env_with(PodMemorySource::Reference).expect_err(
                "Config::from_env must fail-closed on a bare hostname endpoint (b.ohw C-6 / H-auth-1)",
            );
            assert!(
                err.contains(case.var),
                "rejection message must name the offending env var so operators \
                 can fix the right config; case={case_var:?} value={value:?} expected {expected:?} in {err:?}",
                case_var = case.var,
                value = case.value,
                expected = case.var,
            );
            assert!(
                err.contains(case.reason_fragment),
                "rejection message must surface the validator reason so \
                 logs/CI failures point at the right diagnostic; case={case_var:?} value={value:?} expected {expected:?} in {err:?}",
                case_var = case.var,
                value = case.value,
                expected = case.reason_fragment,
            );
        });
    }
}

/// b.ohw H-auth-1: the pre-fix `value.contains(":443")` substring check
/// silently TLS-flipped every port in the range 4430..=4439 (and
/// 44300..=44399, …). Pin the post-fix behavior at the public-config
/// boundary so a future refactor of `validate_auth_endpoint_uri` that
/// regresses to substring matching is caught by an integration test, not
/// just the existing unit tests in `src/auth/mod.rs`.
///
/// An endpoint `http://host:4430` must survive `Config::from_env` —
/// pre-fix it would have silently flipped to `https://` and tonic would
/// have failed the handshake with a confusing SNI mismatch instead of a
/// config-layer rejection. The endpoint round-trips through the Config
/// unchanged because `Config::from_env` retains the literal env value;
/// the URI normalization (https-flip on exact port 443) happens later
/// inside the auth-client constructors.
#[test]
fn port_4430_passes_config_validation_substring_trap() {
    temp_env::with_vars(
        [
            ("LIVE_EDIT_AUTHZ_ENABLED", Some("true")),
            // userinfo_service_enabled=true so scheme validation runs on the
            // userinfo endpoint — without this the b.g3z parse path would
            // skip the validator entirely and the round-trip check would
            // silently lose coverage.
            ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("true")),
            (
                "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                Some("http://storage.prod.internal:4430"),
            ),
            (
                "LIVE_EDIT_USERINFO_ENDPOINT",
                Some("http://userinfo.prod.internal:4435"),
            ),
        ],
        || {
            let cfg = Config::from_env_with(PodMemorySource::Reference).expect(
                "http://host:4430 / :4435 must pass config validation — the b.ohw H-auth-1 \
                 fix uses exact-equality port parsing instead of the pre-fix substring check",
            );
            assert_eq!(
                cfg.storage_service_endpoint.as_deref(),
                Some("http://storage.prod.internal:4430"),
                "Config::from_env must round-trip the literal endpoint; URI flipping \
                 (e.g. https on exact :443) happens later in the auth-client constructor"
            );
            assert_eq!(
                cfg.userinfo_endpoint.as_deref(),
                Some("http://userinfo.prod.internal:4435"),
                "userinfo endpoint must round-trip on the same posture"
            );
        },
    );
}

/// Companion positive case: exact port `:443` on an `http://` scheme is
/// still accepted because the documented carve-out flips it to TLS at
/// connect time. Pinning this prevents an over-eager future fix from
/// rejecting the legitimate `http://host:443` shape.
#[test]
fn exact_port_443_accepted_through_config() {
    temp_env::with_vars(
        [
            ("LIVE_EDIT_AUTHZ_ENABLED", Some("true")),
            // userinfo_service_enabled=true so the userinfo endpoint also
            // flows through scheme validation (parity with the sibling
            // substring-trap test).
            ("LIVE_EDIT_USERINFO_SERVICE_ENABLED", Some("true")),
            (
                "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                Some("http://storage.prod.internal:443"),
            ),
            (
                "LIVE_EDIT_USERINFO_ENDPOINT",
                Some("https://userinfo.prod.internal:443"),
            ),
        ],
        || {
            let cfg = Config::from_env_with(PodMemorySource::Reference)
                .expect("http://host:443 must pass config validation (TLS-flipped at connect)");
            assert_eq!(
                cfg.storage_service_endpoint.as_deref(),
                Some("http://storage.prod.internal:443"),
                "Config::from_env preserves the literal env value; the https flip \
                 happens later inside `normalize_endpoint`"
            );
        },
    );
}

/// Standalone unit-style assertion that the `validate_auth_endpoint_uri`
/// helper itself is reachable from the public `live_edit::auth` re-export
/// surface. If a future module-restructure hides this entry point, the
/// `Config::from_env` path that calls it would still compile (because
/// `from_env` lives in the same crate) but external integration tests
/// would silently lose coverage. This test fails fast if that happens.
#[test]
fn validate_auth_endpoint_uri_is_reachable_from_integration_tests() {
    // The helper accepts a `&str` and returns `Result<(), String>`.
    let ok = live_edit::auth::validate_auth_endpoint_uri("https://host.example.com");
    assert!(ok.is_ok(), "https endpoint must pass: {ok:?}");

    let err = live_edit::auth::validate_auth_endpoint_uri("bare.host.example.com")
        .expect_err("bare hostnames must be rejected via the public helper");
    assert!(
        err.contains("explicit scheme"),
        "error must mention the contract violation: {err}"
    );
}
