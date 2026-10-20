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

//! Shared TLS-error chain walker for the REST + gRPC auth client paths
//! (b.j5t follow-up — DRY).
//!
//! Both [`crate::auth::rest`] and [`crate::auth::mod`] need to classify a
//! transport error as "TLS handshake failure" so it short-circuits to
//! `AuthError::InvalidResponse(...)` instead of being retried as transient
//! [`AuthError::Unreachable`] (b.ohw H-auth-4 / b.j5t item 1). The walker
//! logic is bit-for-bit identical across the two paths: both descend the
//! `Error::source()` chain *and* the hidden `std::io::Error::get_ref()`
//! link (which `source()` skips), downcasting at every level for
//! `rustls::Error`.
//!
//! Keeping two copies of the walker invited drift the moment a future
//! reqwest/hyper bump rearranged how `rustls::Error` appears in the source
//! chain — one path would be fixed, the other silently regress to
//! `Unreachable`-and-retried for the rest of the retry budget. Lift the
//! walker into a single shared helper that both protocol paths call.
//!
//! Version-bound assumption: works against `reqwest = "0.12"` + tonic /
//! hyper-rustls paired with `rustls = "0.23"`. A rustls major bump that
//! changes how the error is wrapped may require updating this helper.
//! Downcasting (rather than string-matching) means a brittle log-format
//! change can't silently break classification — but the tradeoff is that
//! detection is implicitly pinned to the rustls major version we depend on.
//!
//! The walker descends through *two* chain shapes because hyper + rustls
//! produce errors that don't link cleanly via `Error::source()`:
//!
//! 1. **`Error::source()`** — the standard chain. We walk it and downcast
//!    each link.
//! 2. **`std::io::Error::get_ref()`** — `std::io::Error::Custom` wraps a
//!    `Box<dyn Error>`, but its `source()` impl forwards to *that inner
//!    error's* `source()` (skipping the wrapped error itself). reqwest /
//!    hyper wrap `rustls::Error` inside `io::Error` exactly this way, so
//!    without `get_ref()` the walker hops over the link that actually holds
//!    the rustls type and the downcast never fires.
//!
//! Walking both means the function is correct against either chain shape
//! and remains correct if a future reqwest/hyper bump rearranges them.

/// Walk an error-source chain looking for a `rustls::Error`. Returns `true`
/// if any link in the chain — including links reachable only through
/// `std::io::Error::get_ref()` — is a `rustls::Error`.
///
/// Both the REST and gRPC auth paths call this helper to classify TLS
/// handshake failures as permanent (so the retry loop doesn't burn its
/// budget on a cert / trust mismatch). See the module-level docs for the
/// reasoning behind walking both chain shapes.
pub(crate) fn has_rustls_error_in_chain(err: &(dyn std::error::Error + 'static)) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(err);
    while let Some(e) = current {
        if e.downcast_ref::<rustls::Error>().is_some() {
            return true;
        }
        // `std::io::Error::Custom` hides its wrapped error from `source()`
        // — descend through `get_ref()` to reach it. The wrapped value can
        // itself be a `rustls::Error` (the common shape) or another
        // `io::Error` that wraps one (rare but possible during reconnect),
        // so recurse via [`rustls_in_dyn_error`].
        if let Some(io_err) = e.downcast_ref::<std::io::Error>() {
            if let Some(inner) = io_err.get_ref() {
                if rustls_in_dyn_error(inner) {
                    return true;
                }
            }
        }
        current = e.source();
    }
    false
}

/// Inner helper for [`has_rustls_error_in_chain`]: given a `&dyn Error`
/// reached via `io::Error::get_ref()`, walk its own `source()` chain *and*
/// its nested `io::Error::get_ref()` links looking for a `rustls::Error`.
fn rustls_in_dyn_error(err: &(dyn std::error::Error + 'static)) -> bool {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(err);
    while let Some(e) = current {
        if e.downcast_ref::<rustls::Error>().is_some() {
            return true;
        }
        if let Some(io_err) = e.downcast_ref::<std::io::Error>() {
            if let Some(inner) = io_err.get_ref() {
                if inner.downcast_ref::<rustls::Error>().is_some() {
                    return true;
                }
                current = inner.source();
                continue;
            }
        }
        current = e.source();
    }
    false
}
