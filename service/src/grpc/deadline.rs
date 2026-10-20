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

//! Ambient per-request deadline, readable from anywhere inside a handler
//! future (b.hka (f)).
//!
//! # Why this exists
//!
//! Several request-path waits are sized as a fraction of
//! `LIVE_EDIT_RPC_DEADLINE_MS` so that *they*, and not the tower deadline
//! layer, are what ends the wait — the difference between a deliberate
//! `UNAVAILABLE` that names what is pending and a bare `DEADLINE_EXCEEDED`
//! that says nothing. That ordering only holds if the wait starts at RPC
//! t=0. It does not: a `CreateLayer` reaches object-id allocation only after
//! auth, the upload admission wait, the content read, userinfo, OMPF parse
//! and ext-value storage, and
//! a client may have set a `grpc-timeout` shorter than the server deadline in
//! the first place. A budget derived from the *configured* deadline can
//! therefore outlast what is actually left.
//!
//! This module publishes the one fact those call sites are missing: how much
//! of this request's deadline remains. [`remaining`] answers it;
//! [`Config::object_id_counter_repair_wait_budget_within`] is the first
//! consumer.
//!
//! [`Config::object_id_counter_repair_wait_budget_within`]:
//!     crate::config::Config::object_id_counter_repair_wait_budget_within
//!
//! # Where the deadline comes from
//!
//! [`crate::grpc::metrics_layer::MetricsService`] scopes every
//! non-deadline-exempt call, using `min(server deadline, client
//! `grpc-timeout`)` — the same pair tonic's own `GrpcTimeout` layer takes the
//! minimum of. That layer sits OUTSIDE our stack (`tonic
//! 0.14 transport/server/mod.rs`, the `layer_fn(|s| GrpcTimeout::new(s,
//! timeout))` rung wraps the routes) and passes the request through
//! unmodified, so the `grpc-timeout` header is still on the request when it
//! reaches us and re-parsing it here observes exactly what tonic enforces.
//!
//! The two long-lived streaming RPCs, `JoinSession` and `SubscribeReadLayer`,
//! are deliberately NOT scoped: they are exempt from the RPC deadline
//! entirely, so there is no deadline to report and [`remaining`] answers
//! `None` — which every consumer must read as "no bound", not "no time left".
//! The client-streaming uploads `CreateLayer` and `UpdateLayer` are NOT
//! exempt; they are scoped like any unary call.
//!
//! # Not inherited by `tokio::spawn`
//!
//! A `task_local!` belongs to one task. A future moved onto a new task with
//! `tokio::spawn` does NOT inherit it, and that is load-bearing here rather
//! than incidental: the object-id counter repair's detached attempt
//! (`checkpoint::counter_seed`) deliberately outlives the `CreateLayer` that
//! started it, so that its verdict is recorded even when the caller gives up.
//! It must keep its own budget, bounded by the adapter fetch budget, not by
//! the RPC that happened to trigger it. Nothing needs to be done to get that
//! behaviour — but anything that later wraps a spawn in [`scope`] to
//! "propagate" the deadline would break it.

use std::time::Duration;

use tokio::time::Instant;

// The instant this request's deadline elapses.
//
// `tokio::time::Instant` rather than `std::time::Instant` so the value shares
// a clock with `tokio::time::timeout` (and with `tokio::time::pause()` in
// tests) — a budget computed against one clock and enforced against another
// would diverge under a paused runtime.
tokio::task_local! {
    static REQUEST_DEADLINE: Instant;
}

/// How much of the current request's deadline is left.
///
/// `None` means no deadline is in scope at all — one of the two
/// deadline-exempt streaming RPCs, a background task, or a unit test calling
/// the code directly. Consumers MUST treat that as "unbounded", never as
/// "expired".
///
/// `Some(Duration::ZERO)` is reachable and means the deadline has already
/// passed; the tower layer is about to drop this future. Callers that feed the
/// value into a timeout should keep their own non-zero floor
/// (`tokio::time::timeout` treats a zero duration as already elapsed).
pub fn remaining() -> Option<Duration> {
    REQUEST_DEADLINE
        .try_with(|deadline| deadline.saturating_duration_since(Instant::now()))
        .ok()
}

/// Run `fut` with `deadline` installed as the ambient request deadline.
///
/// Scoping (rather than setting) is what keeps the value honest: it is visible
/// exactly for the dynamic extent of the wrapped future and cannot leak onto a
/// pooled worker thread after the request ends.
pub async fn scope<F: std::future::Future>(deadline: Instant, fut: F) -> F::Output {
    REQUEST_DEADLINE.scope(deadline, fut).await
}

/// Parse the gRPC `grpc-timeout` header per the gRPC-over-HTTP/2 wire format:
/// 1–8 ASCII digits followed by a unit character — `H` hours, `M` minutes,
/// `S` seconds, `m` milliseconds, `u` microseconds, `n` nanoseconds.
///
/// Returns `None` for absent, empty, non-ASCII, over-long (> 8 digits, the
/// spec's own cap, which also rules out the overflow), non-numeric, and
/// unknown-unit values. **Fail-open is deliberate**: this header is only used
/// to make a wait budget *tighter*, and tonic's `GrpcTimeout` layer — which
/// applies the same 8-digit cap and unit set, and is the thing that actually
/// enforces the timeout — treats a malformed value as absent too
/// (`try_parse_grpc_timeout`, `tonic 0.14
/// transport/service/grpc_timeout.rs`). Rejecting the request here instead
/// would disagree with the layer that owns the decision.
///
/// Re-implemented rather than reused because tonic's parser is `pub(crate)`.
/// It is a closed, versioned wire format, so the duplication is a stable one;
/// keep the two in step if tonic's ever widens.
pub fn parse_grpc_timeout(headers: &http::HeaderMap) -> Option<Duration> {
    const SECONDS_IN_HOUR: u64 = 60 * 60;
    const SECONDS_IN_MINUTE: u64 = 60;

    let raw = headers.get("grpc-timeout")?.to_str().ok()?;
    // `split_at` on the last byte is safe: `to_str` succeeded, so the value is
    // ASCII and every byte is a char boundary.
    if raw.is_empty() {
        return None;
    }
    let (value, unit) = raw.split_at(raw.len() - 1);
    // The spec caps TimeoutValue at 8 digits; enforcing it also makes the
    // multiplications below unable to overflow a u64.
    if value.is_empty() || value.len() > 8 {
        return None;
    }
    let value: u64 = value.parse().ok()?;
    Some(match unit {
        "H" => Duration::from_secs(value * SECONDS_IN_HOUR),
        "M" => Duration::from_secs(value * SECONDS_IN_MINUTE),
        "S" => Duration::from_secs(value),
        "m" => Duration::from_millis(value),
        "u" => Duration::from_micros(value),
        "n" => Duration::from_nanos(value),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    //! Unit coverage for the ambient-deadline surface (b.hka (f)).
    //!
    //! Three properties, kept apart because they fail for different reasons:
    //! the wire-format parse ([`parse_grpc_timeout`]), the scope/read pair
    //! ([`scope`] / [`remaining`]), and the deliberate NON-propagation across
    //! `tokio::spawn`.
    //!
    //! No Valkey, no server: everything here is in-process. The composed claim
    //! — that `MetricsService` installs the scope for non-exempt RPCs and not
    //! for exempt ones — lives in `src/grpc/metrics_layer.rs`'s tests, and the
    //! end-to-end payoff in
    //! `tests/integration/checkpoint/counter_repair_test.rs`.
    use super::*;

    /// Build a `HeaderMap` carrying exactly one `grpc-timeout` value.
    fn headers_with(timeout: &str) -> http::HeaderMap {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            "grpc-timeout",
            http::HeaderValue::from_str(timeout).expect("test header value must be valid ASCII"),
        );
        headers
    }

    /// Every unit in the gRPC wire format is understood, and scaled correctly.
    ///
    /// The whole set is walked rather than a representative sample because a
    /// dropped or mis-scaled unit is silent in exactly the direction that
    /// hurts: an unrecognised `"u"` degrades to `None` (fail-open → the budget
    /// stays at the configured deadline, which is the pre-fix behaviour this
    /// module exists to end), and a mis-scaled `"M"` produces a budget 60×
    /// wrong in whichever direction the bug went.
    #[test]
    fn parse_grpc_timeout_understands_every_unit() {
        for (raw, expected) in [
            ("2H", Duration::from_secs(2 * 60 * 60)),
            ("3M", Duration::from_secs(3 * 60)),
            ("5S", Duration::from_secs(5)),
            ("250m", Duration::from_millis(250)),
            ("400u", Duration::from_micros(400)),
            ("900n", Duration::from_nanos(900)),
        ] {
            assert_eq!(
                parse_grpc_timeout(&headers_with(raw)),
                Some(expected),
                "grpc-timeout: {raw} must parse as {expected:?}"
            );
        }
    }

    /// The spec's 8-digit `TimeoutValue` cap is enforced at the boundary.
    ///
    /// Enforcing it is not pedantry: it is what makes the `value * 3600`
    /// multiplication in the `H` arm unable to overflow a `u64`, and tonic's
    /// own parser — the thing that actually enforces the timeout — applies the
    /// same cap, so accepting a 9-digit value here would compute a budget
    /// against a bound nobody enforces.
    #[test]
    fn parse_grpc_timeout_enforces_the_eight_digit_cap() {
        assert_eq!(
            parse_grpc_timeout(&headers_with("99999999m")),
            Some(Duration::from_millis(99_999_999)),
            "8 digits is the spec's maximum and MUST be accepted"
        );
        assert_eq!(
            parse_grpc_timeout(&headers_with("100000000m")),
            None,
            "9 digits is outside the wire format; it MUST be treated as absent, matching \
             tonic's parser rather than inventing a bound tonic will not enforce"
        );
    }

    /// Everything malformed degrades to `None` — deliberately fail-open.
    ///
    /// `None` means "no client bound", which leaves the wait budget at the
    /// configured server deadline: the pre-b.hka behaviour, and strictly safe.
    /// Rejecting the request instead would disagree with tonic's `GrpcTimeout`
    /// layer, which owns the decision and treats a malformed value as absent.
    #[test]
    fn parse_grpc_timeout_treats_malformed_values_as_absent() {
        for raw in [
            "100X",  // unknown unit
            "100",   // no unit at all — the trailing `0` is read as the unit
            "m",     // unit with no value
            "",      // empty
            "abcm",  // non-numeric value
            "-100m", // negative
            "10.5m", // fractional
            "100 m", // embedded space
        ] {
            assert_eq!(
                parse_grpc_timeout(&headers_with(raw)),
                None,
                "grpc-timeout: {raw:?} is malformed and MUST read as absent, not as some \
                 salvaged duration"
            );
        }

        assert_eq!(
            parse_grpc_timeout(&http::HeaderMap::new()),
            None,
            "no header at all is the common case (most clients set no deadline) and MUST be \
             `None` rather than a zero duration"
        );
    }

    /// Outside a [`scope`] there is no deadline, and `None` means *unbounded*.
    ///
    /// The distinction matters at every consumer: a deadline-exempt streaming
    /// RPC, a background task and a unit test all read `None`, and reading that
    /// as "expired" would collapse their waits to the 1 ms floor.
    #[tokio::test]
    async fn remaining_is_none_outside_a_scope() {
        assert_eq!(
            remaining(),
            None,
            "no scope means no bound to report; `Some(ZERO)` here would be read by every \
             consumer as an already-expired deadline"
        );
    }

    /// Inside a scope the reported remainder is positive and never exceeds the
    /// budget the scope was opened with.
    ///
    /// The upper bound is the load-bearing half — a consumer sizes a wait as a
    /// fraction of this value, so over-reporting would put the wait back past
    /// the deadline, which is the bug (f) fixes. The lower bound is asserted
    /// only as "positive": anything tighter would be a wall-clock assertion,
    /// which this suite does not gate on.
    #[tokio::test]
    async fn remaining_inside_a_scope_is_positive_and_within_the_budget() {
        const BUDGET: Duration = Duration::from_secs(30);

        let observed = scope(Instant::now() + BUDGET, async { remaining() }).await;

        let observed = observed.expect("a scoped future MUST see the deadline it was scoped with");
        assert!(
            observed <= BUDGET,
            "the remainder ({observed:?}) MUST NOT exceed the budget the scope was opened with \
             ({BUDGET:?}) — a consumer sizes its wait as a fraction of this number, so \
             over-reporting puts the wait back past the deadline"
        );
        assert!(
            !observed.is_zero(),
            "and it must not already read as expired at the instant the scope opens"
        );
    }

    /// A `tokio::spawn`ed task does NOT inherit the scope.
    ///
    /// Load-bearing rather than incidental. The object-id counter repair's
    /// attempt is spawned precisely so it outlives the `CreateLayer` that
    /// triggered it and records its own verdict; inheriting the caller's
    /// deadline would re-couple the attempt to a request that has already given
    /// up, which is the design (f) was built on top of, not something it may
    /// quietly undo. Pinned here because "propagate the deadline to spawned
    /// work" is an easy-looking improvement.
    #[tokio::test]
    async fn a_spawned_task_does_not_inherit_the_scope() {
        let seen = scope(Instant::now() + Duration::from_secs(30), async {
            let inside = remaining();
            let spawned = tokio::spawn(async { remaining() })
                .await
                .expect("the probe task must not panic");
            (inside, spawned)
        })
        .await;

        assert!(
            seen.0.is_some(),
            "precondition: the scope really was installed for the calling task"
        );
        assert_eq!(
            seen.1, None,
            "a spawned task MUST NOT inherit the request deadline — the detached counter-repair \
             attempt is spawned exactly so it survives its caller giving up, and it must keep \
             its own adapter-fetch budget rather than a deadline that is about to elapse"
        );
    }
}
