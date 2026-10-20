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

//! Integration tests for `ServiceTokenProvider` rotation, force-refresh,
//! absent-file tolerance, cancellation, and parse-error recovery.
//!
//! Subtask `t3.fgj.dp.io.t8` under Task `t2.fgj.dp.io` (Epic `t1.fgj.dp`).
//! These cases exercise the same logic the sibling unit subtask covers, but
//! from outside the crate — they import only the `live_edit::auth::service_token`
//! public surface. Pinning the integration boundary here catches accidental
//! `pub(crate)` regressions that the inline unit tests would miss.
//!
//! These tests do NOT require Valkey — the surface is pure on-disk + in-memory.
//!
//! ## What these tests are observing through (b.e49, b.mq8)
//!
//! The assertions below read [`ServiceTokenProvider::cached_bearer_if_unexpired`],
//! the provider's sole bearer accessor and the one every production consumer
//! reads: the Valkey credential path (b.d6b), the storage-api checkpoint
//! adapter, and the health supervisor (b.e49). It ignores the per-instance
//! refresh lead and serves anything not actually expired.
//!
//! What this file pins is the *polling-task lifecycle* — rotation pickup,
//! force-refresh, absent-file tolerance, cancellation, parse-error recovery —
//! observed through that accessor, which is simply the cheapest window onto
//! "what does the cache hold now". Every fixture uses a comfortably-future
//! expiry so the accessor never answers `None` for a reason other than the
//! lifecycle step under test.
//!
//! These assertions used to run through a second, *withholding* accessor,
//! which is why the fixtures clear the 60 s `DEFAULT_REFRESH_LEAD` by an hour.
//! That accessor lost its last production consumer in b.e49 and was deleted in
//! b.mq8; the hour-long expiry is kept because it keeps the lifecycle
//! assertions isolated, not because any lead is still consulted.
//!
//! ## File-write semantics
//!
//! The sidecar writes the token file using temp-file + rename, so this test
//! exercises the same atomic-rename path via `tempfile::NamedTempFile::persist`
//! (or a sibling-file + `std::fs::rename`). A plain `std::fs::write` to the
//! live path would mask a race the production code is specifically designed
//! to tolerate.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use tempfile::TempDir;
use tokio_util::sync::CancellationToken;

use live_edit::auth::service_token::{ServiceTokenError, ServiceTokenProvider};
use live_edit::observability::metrics::Metrics;

/// File name used inside each per-test `TempDir`. Matches the conventional
/// sidecar mount path (`/var/run/oauth/access_token`) basename.
const TOKEN_FILE_NAME: &str = "access_token";

/// Build a JSON token document with `expires_on = now + 1 hour`. Returns the
/// serialized JSON string.
///
/// An hour puts every fixture comfortably clear of its own expiry, so
/// [`ServiceTokenProvider::cached_bearer_if_unexpired`] — the accessor these
/// tests observe through — can only answer `None` for the lifecycle reason each
/// test is about (nothing read yet, nothing cached, task cancelled), never
/// because a fixture quietly died mid-test.
///
/// It also clears the provider's 60 s `DEFAULT_REFRESH_LEAD`, which used to be
/// load-bearing: these assertions once ran through a withholding accessor that
/// returned `None` inside the lead. That accessor is gone (b.e49, b.mq8) and no
/// accessor consults the lead any more, so the clearance is now incidental.
fn fresh_token_json(access_token: &str) -> String {
    let expires_on = Utc::now() + chrono::Duration::hours(1);
    // Format as RFC3339 — chrono's default `Serialize` impl for `DateTime<Utc>`
    // emits RFC3339, which is what the provider parses on the read side.
    format!(
        "{{\"expires_on\":\"{}\",\"access_token\":\"{}\"}}",
        expires_on.to_rfc3339(),
        access_token
    )
}

/// Atomically write `contents` to `path` via a sibling tempfile + rename.
///
/// Mirrors the credential-provisioner sidecar's write contract: a partial
/// reader on the other side never observes a torn file. Using a sibling in
/// the same directory keeps `rename` cross-filesystem-safe (rename across
/// filesystems is `EXDEV`).
fn atomic_write(path: &Path, contents: &str) {
    let parent = path
        .parent()
        .expect("token file path must have a parent directory");
    let mut tmp = tempfile::NamedTempFile::new_in(parent)
        .expect("NamedTempFile::new_in must succeed in test tempdir");
    use std::io::Write;
    tmp.write_all(contents.as_bytes())
        .expect("write to tempfile must succeed");
    tmp.persist(path)
        .expect("persist (atomic rename) must succeed");
}

/// Build a fresh `Arc<Metrics>` rooted in a private `prometheus::Registry`
/// so concurrent tests don't collide on global metric registration.
fn fresh_metrics() -> Arc<Metrics> {
    Arc::new(crate::common::metrics_scrape_only(
        "service-token-rotation-test",
    ))
}

// ---------------------------------------------------------------------------
// AC #3 — rotation picked up within one poll interval + jitter
// ---------------------------------------------------------------------------

/// Pins parent Task acceptance criterion #3: a sidecar-driven token rewrite
/// is reflected by `cached_bearer_if_unexpired()` within roughly one poll interval.
///
/// Sequence: write token-A → construct provider (50 ms poll) → wait 150 ms →
/// assert token-A → atomic rewrite to token-B → wait 150 ms → assert token-B
/// → cancel and exit cleanly.
#[tokio::test]
async fn token_rotation_picked_up_within_one_poll_interval_plus_jitter() {
    let dir = TempDir::new().expect("TempDir::new must succeed");
    let token_path = dir.path().join(TOKEN_FILE_NAME);

    // Initial write — direct write is fine for the first creation because
    // there is no concurrent reader yet (provider has not been constructed).
    std::fs::write(&token_path, fresh_token_json("token-A"))
        .expect("initial token file write must succeed");

    let cancel = CancellationToken::new();
    let provider = ServiceTokenProvider::new(
        token_path.clone(),
        Duration::from_millis(50),
        fresh_metrics(),
        cancel.clone(),
    );

    // Allow the polling task at least 2-3 ticks to complete its first
    // read. 150 ms > 3 × 50 ms covers scheduling jitter on a loaded CI
    // worker without ballooning test runtime.
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(
        provider.cached_bearer_if_unexpired(),
        Some("token-A".to_string()),
        "provider must observe token-A after initial poll"
    );

    // Rotation: atomic rename so the polling task never sees a partial file.
    atomic_write(&token_path, &fresh_token_json("token-B"));

    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(
        provider.cached_bearer_if_unexpired(),
        Some("token-B".to_string()),
        "provider must observe token-B within one poll interval + jitter \
         after atomic rewrite"
    );

    // Cancellation must let the test exit cleanly — the polling task
    // observes the cancel signal in the `select!` and returns.
    cancel.cancel();
    // Yield once so the polling task gets a chance to observe the cancel
    // signal before TempDir's Drop tears down the file.
    tokio::time::sleep(Duration::from_millis(50)).await;
}

// ---------------------------------------------------------------------------
// AC #4 — force_refresh short-circuits the polling interval
// ---------------------------------------------------------------------------

/// Pins parent Task acceptance criterion #4: `force_refresh()` returns the
/// freshly-read token WITHOUT waiting for the next poll tick.
///
/// The poll interval is deliberately set to 60 s so the polling task cannot
/// possibly help within the test's wall-clock window. If the assertion holds,
/// `force_refresh()` is the only path that could have refreshed the cache.
#[tokio::test]
async fn force_refresh_returns_new_token_immediately_without_polling_tick() {
    let dir = TempDir::new().expect("TempDir::new must succeed");
    let token_path = dir.path().join(TOKEN_FILE_NAME);

    std::fs::write(&token_path, fresh_token_json("token-A"))
        .expect("initial token file write must succeed");

    let cancel = CancellationToken::new();
    // 60 s poll interval — long enough that the polling task's first tick
    // (which fires immediately on construction) is the ONLY tick within the
    // test's wall-clock window; the next would be 60 s later.
    let provider = ServiceTokenProvider::new(
        token_path.clone(),
        Duration::from_secs(60),
        fresh_metrics(),
        cancel.clone(),
    );

    // The polling task's first tick fires immediately on `interval()`
    // construction, so token-A should be cached. If the engineer changes
    // the interval-startup semantics later, this `force_refresh` also
    // bootstraps so the test stays robust either way.
    provider
        .force_refresh()
        .await
        .expect("force_refresh must succeed on a present, valid file");
    assert_eq!(
        provider.cached_bearer_if_unexpired(),
        Some("token-A".to_string()),
        "provider must observe token-A after the bootstrap force_refresh"
    );

    // Rewrite the file. The polling task is parked on a 60 s timer; only
    // `force_refresh()` can possibly pick this up before the test ends.
    atomic_write(&token_path, &fresh_token_json("token-B"));

    provider
        .force_refresh()
        .await
        .expect("force_refresh on the rewritten file must succeed");

    // IMMEDIATE assertion — no sleep — proves the refresh path didn't rely
    // on the polling interval.
    assert_eq!(
        provider.cached_bearer_if_unexpired(),
        Some("token-B".to_string()),
        "force_refresh must surface token-B without waiting for the next \
         poll tick"
    );

    cancel.cancel();
}

// ---------------------------------------------------------------------------
// AC #5 — constructor tolerates an absent file
// ---------------------------------------------------------------------------

/// Pins parent Task acceptance criterion #5: constructing the provider when
/// the token file does not yet exist returns a valid provider that yields
/// `None` from `cached_bearer_if_unexpired()` until the sidecar writes the file. This is
/// the SR-6.2 "startup-absent" path.
#[tokio::test]
async fn provider_constructor_tolerates_absent_file() {
    let dir = TempDir::new().expect("TempDir::new must succeed");
    let token_path = dir.path().join(TOKEN_FILE_NAME);
    // File deliberately NOT created.

    let cancel = CancellationToken::new();
    let provider = ServiceTokenProvider::new(
        token_path.clone(),
        Duration::from_millis(50),
        fresh_metrics(),
        cancel.clone(),
    );

    // Immediately after construction, cache must be empty.
    assert_eq!(
        provider.cached_bearer_if_unexpired(),
        None,
        "cache must be empty immediately after construction"
    );

    // Several poll ticks against a missing file must keep cache empty —
    // the polling task swallows NotFound silently (SR-6.2).
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        provider.cached_bearer_if_unexpired(),
        None,
        "cache must remain empty across multiple polls of an absent file"
    );

    // Now sidecar appears: write the file. Use atomic rename so the
    // production-shaped write semantics are exercised end-to-end.
    atomic_write(&token_path, &fresh_token_json("token-late"));

    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(
        provider.cached_bearer_if_unexpired(),
        Some("token-late".to_string()),
        "provider must pick up the token once the sidecar writes the file"
    );

    cancel.cancel();
}

// ---------------------------------------------------------------------------
// Cancellation shutdown — defense-in-depth on the polling-task lifetime
// ---------------------------------------------------------------------------

/// Defense-in-depth on the shutdown contract: once `cancel()` fires, the
/// polling task stops reading the file. Observable from the public API:
/// post-cancel rewrites are NOT picked up by `cached_bearer_if_unexpired()`.
///
/// This avoids reaching into internal task handles — the assertion uses only
/// the public surface, matching what an Epic 5 caller could verify.
#[tokio::test]
async fn cancellation_token_shutdown_terminates_polling_task() {
    let dir = TempDir::new().expect("TempDir::new must succeed");
    let token_path = dir.path().join(TOKEN_FILE_NAME);

    std::fs::write(&token_path, fresh_token_json("token-pre-cancel"))
        .expect("initial token file write must succeed");

    let cancel = CancellationToken::new();
    let provider = ServiceTokenProvider::new(
        token_path.clone(),
        Duration::from_millis(50),
        fresh_metrics(),
        cancel.clone(),
    );

    // Confirm the polling task is running by waiting for the initial read.
    tokio::time::sleep(Duration::from_millis(150)).await;
    assert_eq!(
        provider.cached_bearer_if_unexpired(),
        Some("token-pre-cancel".to_string()),
        "polling task must have read token-pre-cancel before cancel"
    );

    // Cancel the polling task.
    cancel.cancel();
    // Yield generously so the polling task observes the cancel signal and
    // returns from its `select!` before we rewrite the file. The cancel
    // branch is `biased`, so the task should exit on its very next loop
    // iteration; 100 ms is well past one poll interval.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Rewrite the file. A still-alive polling task would observe this
    // within ~50 ms; a cancelled task will not.
    atomic_write(&token_path, &fresh_token_json("token-post-cancel"));

    // Wait several poll intervals: if the task were still running, it would
    // have refreshed the cache to "token-post-cancel" by now.
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert_eq!(
        provider.cached_bearer_if_unexpired(),
        Some("token-pre-cancel".to_string()),
        "polling task must NOT pick up post-cancel writes — cache pinned to \
         pre-cancel value"
    );
}

// ---------------------------------------------------------------------------
// Concurrent force_refresh — defense-in-depth on the request channel
// ---------------------------------------------------------------------------

/// Defense-in-depth: many concurrent `force_refresh()` futures from one
/// shared `Arc<ServiceTokenProvider>` must all complete (each either
/// `Ok(())` or a benign `Cancelled` if the channel saturates) without
/// panicking or deadlocking, and the final `cached_bearer_if_unexpired()` must reflect
/// the file contents.
///
/// Catches a future regression where the `RwLock` strategy degrades into a
/// deadlock when multiple 401-retries race through `force_refresh()` at the
/// same time. The current implementation funnels requests through a bounded
/// `mpsc::channel(8)`; this test fans in 16 futures > channel capacity to
/// also exercise the channel's full-backpressure path.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn force_refresh_can_be_called_concurrently_without_panic() {
    let dir = TempDir::new().expect("TempDir::new must succeed");
    let token_path = dir.path().join(TOKEN_FILE_NAME);

    std::fs::write(&token_path, fresh_token_json("token-concurrent"))
        .expect("initial token file write must succeed");

    let cancel = CancellationToken::new();
    let provider = Arc::new(ServiceTokenProvider::new(
        token_path.clone(),
        // Long-ish interval so the polling task isn't competing with the
        // burst of force_refresh requests for the read_and_update slot.
        Duration::from_secs(60),
        fresh_metrics(),
        cancel.clone(),
    ));

    // Fan out 16 concurrent force_refresh calls — > the current
    // mpsc::channel(8) capacity so the test exercises both the
    // immediate-send and the await-on-full-channel paths.
    const N: usize = 16;
    let mut handles = Vec::with_capacity(N);
    for _ in 0..N {
        let p = Arc::clone(&provider);
        handles.push(tokio::spawn(async move { p.force_refresh().await }));
    }

    let results = futures::future::join_all(handles).await;

    for (i, joined) in results.into_iter().enumerate() {
        let inner = joined.expect("force_refresh task must not panic");
        // Every call must complete: either Ok (the typical case) or
        // — if the channel ever saturates and the polling task hasn't
        // drained yet — a benign error variant. Panics or deadlocks are
        // the only outcomes this test rules out.
        match inner {
            Ok(()) => {}
            Err(ServiceTokenError::Cancelled) => {
                panic!(
                    "force_refresh #{i} returned Cancelled while the \
                     CancellationToken was untouched — this indicates the \
                     polling task exited unexpectedly"
                );
            }
            Err(e) => {
                panic!("force_refresh #{i} returned unexpected error: {e:?}");
            }
        }
    }

    // After the storm of refreshes, the cache must hold the only token the
    // file ever contained.
    assert_eq!(
        provider.cached_bearer_if_unexpired(),
        Some("token-concurrent".to_string()),
        "final cache state must match the file's contents after N concurrent \
         force_refresh calls"
    );

    cancel.cancel();
}

// ---------------------------------------------------------------------------
// SC #6 — malformed JSON then valid: recovery via public API + metric
// ---------------------------------------------------------------------------

/// Pins parent Task Success Criterion #6 at the integration tier: a malformed
/// JSON file does NOT poison the provider, and a subsequent valid write is
/// picked up without restart. Also asserts the
/// `service_token_parse_errors_total` metric ticks exactly once per
/// parse failure.
///
/// Note on the previous-good-cache invariant: this test starts with NO prior
/// good token (cold start + first bad write), so the relevant assertion is
/// "cache stays None across the bad poll, then transitions to Some on the
/// good poll". The sibling unit subtask covers the case where a previously
/// good token survives a subsequent malformed write.
#[tokio::test]
async fn malformed_json_then_valid_recovers_via_public_api() {
    let dir = TempDir::new().expect("TempDir::new must succeed");
    let token_path = dir.path().join(TOKEN_FILE_NAME);

    // Real metrics — we want to observe the counter from the public surface
    // (`live_edit::observability::metrics::test_util::sample_value(&live_edit::observability::metrics::test_util::render(&metrics), "live_edit_service_token_parse_errors_total", &[]).unwrap_or(0.0)`).
    let metrics = Arc::new(crate::common::metrics_scrape_only(
        "service-token-rotation-test",
    ));

    // Start with malformed JSON. Cold start: cache has never held a value.
    std::fs::write(&token_path, "not-json-at-all")
        .expect("malformed token file write must succeed");

    let cancel = CancellationToken::new();
    let provider = ServiceTokenProvider::new(
        token_path.clone(),
        Duration::from_millis(50),
        Arc::clone(&metrics),
        cancel.clone(),
    );

    // Wait for the first poll tick + slack so the parse-error path has run
    // at least once.
    tokio::time::sleep(Duration::from_millis(150)).await;

    assert_eq!(
        provider.cached_bearer_if_unexpired(),
        None,
        "malformed JSON must NOT populate the cache (cold start)"
    );

    // Capture the parse-error metric AFTER the read but BEFORE the recovery
    // write. The polling task may have ticked multiple times against the
    // malformed file — assert at least one parse failure was recorded, and
    // remember the count so we can verify the recovery write does NOT tick
    // it further.
    let parse_errors_after_malformed = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_service_token_parse_errors_total",
        &[],
    )
    .unwrap_or(0.0);
    assert!(
        parse_errors_after_malformed >= 1.0,
        "expected at least one parse error after polling a malformed file; \
         got {parse_errors_after_malformed}"
    );

    // Recover: write a valid JSON file via atomic rename.
    atomic_write(&token_path, &fresh_token_json("token-recovered"));

    tokio::time::sleep(Duration::from_millis(150)).await;

    assert_eq!(
        provider.cached_bearer_if_unexpired(),
        Some("token-recovered".to_string()),
        "provider must recover and observe token-recovered after the file is \
         rewritten with valid JSON — no restart required"
    );

    // The valid read must NOT increment the parse-error metric.
    let parse_errors_after_recovery = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_service_token_parse_errors_total",
        &[],
    )
    .unwrap_or(0.0);
    assert_eq!(
        parse_errors_after_recovery, parse_errors_after_malformed,
        "service_token_parse_errors_total must NOT tick on a successful read; \
         expected {parse_errors_after_malformed}, got {parse_errors_after_recovery}"
    );

    cancel.cancel();
}
