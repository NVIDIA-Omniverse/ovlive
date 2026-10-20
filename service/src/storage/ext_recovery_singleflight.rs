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

//! Per-hash singleflight primitive for ext-value recovery fetches.
//!
//! SDD SR-3.2: when a diff-build read finds an ext-value blob missing from
//! Valkey, the recovery layer falls back to
//! [`CheckpointStore::get_ext_blob(hash_hex)`](crate::checkpoint::CheckpointStore::get_ext_blob).
//! That fallback MUST be single-flight per hash — concurrent diff-builds that
//! miss on the same hash issue exactly one adapter fetch + Valkey rewrite;
//! subsequent waiters share the result. Without coalescing, a thundering-herd
//! subscribe-storm against a freshly-recovered layer would stampede the
//! checkpoint adapter with N redundant fetches of the same multi-MB blob.
//!
//! The implementation mirrors the canonical singleflight pattern in
//! [`crate::auth::cache`] (`AuthCache::check_permission` /
//! `AuthCache::get_current_user`):
//!
//! - `DashMap<String, Shared<BoxFuture<...>>>` keyed by `hash_hex`. The
//!   `Shared` future is the cloneable join handle every coalesced waiter
//!   awaits.
//! - `DashMap::entry(...).or_insert_with(...)` atomically inserts the new
//!   shared future only when no peer is already in flight, so `make_future`
//!   runs at most once per (key, generation).
//! - `AssertUnwindSafe(...).catch_unwind()` wraps the inner future so a panic
//!   in the adapter implementation does not cascade through `futures::Shared`
//!   to every coalesced waiter. Caught panics surface as
//!   [`CheckpointError::Permanent`] to every waiter (b.ohw H-auth-3
//!   equivalent).
//! - `scopeguard::guard(...)` removes the inflight entry on every exit path
//!   — normal return, error return, or future-drop on cancellation
//!   (b.j5t item 4 equivalent). The removal is keyed on `Shared::ptr_eq` so a
//!   follow-up caller that installed a fresh future at the same key after
//!   ours resolved is never evicted.

use std::error::Error as StdError;
use std::fmt;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use bytes::Bytes;
use dashmap::DashMap;
use futures::future::{BoxFuture, FutureExt, Shared};

use crate::checkpoint::CheckpointError;

/// Cloneable view of a `CheckpointError` carried inside the shared future.
///
/// `CheckpointError` is not `Clone` because its `Transient` / `Permanent`
/// variants own a `Box<dyn Error + Send + Sync>` source (which is not
/// cloneable in general). `futures::Shared::shared()` requires the future's
/// `Output: Clone`, so we cannot store `Result<_, CheckpointError>` directly.
///
/// `SharedCheckpointError` wraps the original error in `Arc` for the
/// transient and permanent variants, and copies the unit-like variants
/// verbatim. Coalesced waiters all observe the same underlying error: the
/// `Arc` keeps the original boxed source live and every waiter constructs a
/// fresh `CheckpointError` whose `source` is a small wrapper around the same
/// `Arc<CheckpointError>` (so `error.source()` chains back to the original).
#[derive(Clone)]
enum SharedCheckpointError {
    Transient(Arc<CheckpointError>),
    Permanent(Arc<CheckpointError>),
    NotFound,
    Conflict,
    NotImplemented { method: &'static str },
}

impl SharedCheckpointError {
    fn from_owned(e: CheckpointError) -> Self {
        match e {
            CheckpointError::Transient { .. } => Self::Transient(Arc::new(e)),
            CheckpointError::Permanent { .. } => Self::Permanent(Arc::new(e)),
            CheckpointError::NotFound => Self::NotFound,
            CheckpointError::Conflict => Self::Conflict,
            CheckpointError::NotImplemented { method } => Self::NotImplemented { method },
        }
    }

    /// Reconstruct a fresh `CheckpointError` for one coalesced waiter. The
    /// source field of `Transient` / `Permanent` is a thin `Arc`-backed
    /// wrapper so `error.source()` chains back to the original `dyn Error`.
    fn into_checkpoint_error(self) -> CheckpointError {
        match self {
            Self::Transient(arc) => CheckpointError::Transient {
                source: Box::new(ArcChainedError(arc)),
            },
            Self::Permanent(arc) => CheckpointError::Permanent {
                source: Box::new(ArcChainedError(arc)),
            },
            Self::NotFound => CheckpointError::NotFound,
            Self::Conflict => CheckpointError::Conflict,
            Self::NotImplemented { method } => CheckpointError::NotImplemented { method },
        }
    }
}

/// `Error` wrapper around an `Arc<CheckpointError>` so it can be stored as
/// the `source: Box<dyn Error + Send + Sync>` of a reconstructed
/// `CheckpointError::Transient` / `Permanent`. Display/Source delegate to the
/// inner `CheckpointError` so the message chain matches the original.
struct ArcChainedError(Arc<CheckpointError>);

impl fmt::Debug for ArcChainedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

impl fmt::Display for ArcChainedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&*self.0, f)
    }
}

impl StdError for ArcChainedError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.0.source()
    }
}

/// Singleflight map value: a cloneable, shared future that resolves to the
/// same `Result<Option<Bytes>, SharedCheckpointError>` for every concurrent
/// waiter. The error half is the cloneable wrapper above; `fetch_once`
/// converts it back to a fresh `CheckpointError` per waiter.
type SharedRecoveryFuture =
    Shared<BoxFuture<'static, Result<Option<Bytes>, SharedCheckpointError>>>;

/// Per-hash singleflight for ext-value recovery fetches.
///
/// See the module-level docs for the design rationale. Construct once and
/// share across diff-build readers (typically held inside
/// `ValkeyExternalValueStore` and reused across `read_batch` calls).
pub struct ExtRecoverySingleflight {
    inflight: DashMap<String, SharedRecoveryFuture>,
}

impl ExtRecoverySingleflight {
    /// Construct an empty singleflight map.
    pub fn new() -> Self {
        Self {
            inflight: DashMap::new(),
        }
    }

    /// Run `make_future` at most once per concurrent `hash_hex`, awaiting the
    /// result. Concurrent callers for the same `hash_hex` share the single
    /// inner future and all observe the same resolved value. Concurrent
    /// callers for different `hash_hex` values invoke their closures
    /// independently.
    ///
    /// Cancellation safety: dropping the returned future before it resolves
    /// will not leak the inflight map entry. The scope guard removes the
    /// entry on every exit path, keyed on `Shared::ptr_eq` so a follow-up
    /// caller's freshly installed future is never evicted.
    ///
    /// Panic isolation: if the future produced by `make_future` panics on
    /// poll, every coalesced waiter sees `Err(CheckpointError::Permanent)`
    /// rather than a cascading panic.
    ///
    /// **Layout invariant**: the FIRST `.await` in this function MUST be the
    /// `shared.await` at the bottom of the body; the `coalesces_same_hash`
    /// test's noop-poll rendezvous depends on this. The test uses a manual
    /// `noop_waker` single-poll to prove that a joining caller has executed
    /// the synchronous prelude (inflight-map lookup / `Shared` clone / scope
    /// guard install) and is parked on the shared future — if a future
    /// refactor inserts an earlier `.await` (e.g., an `async` lookup, a
    /// metrics tick, a tracing span enter) the test's "parked at shared"
    /// invariant silently degrades into "parked somewhere earlier".
    pub async fn fetch_once<F, Fut>(
        &self,
        hash_hex: String,
        make_future: F,
    ) -> Result<Option<Bytes>, CheckpointError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Option<Bytes>, CheckpointError>> + Send + 'static,
    {
        let shared = self.get_or_create_inflight(&hash_hex, make_future);
        // Clone for the post-await `remove_if` check; `shared.await` consumes
        // the original via `IntoFuture`.
        let shared_for_check = shared.clone();

        // Cancellation-safe inflight cleanup (b.j5t item 4 equivalent).
        // A caller dropped mid-`.await` (subscriber cancellation, deadline
        // expiry) would otherwise leak the inflight entry indefinitely —
        // DashMap has no TTL eviction. The guard removes the entry on every
        // exit path including the Drop path. `Shared::ptr_eq` ensures we
        // never evict a fresh entry installed by a follow-up caller after
        // ours resolved.
        let cleanup_inflight = &self.inflight;
        let cleanup_key = hash_hex.clone();
        let cleanup_shared = shared.clone();
        let cleanup = scopeguard::guard((), move |()| {
            cleanup_inflight.remove_if(&cleanup_key, |_k, v| v.ptr_eq(&cleanup_shared));
        });

        let shared_result = shared.await;

        // Defuse the scope guard so we own the explicit removal sequence
        // below (it runs only on the happy path; cancellation goes through
        // the guard). Mirrors `AuthCache`'s pattern even though we have no
        // post-resolve cache write to interleave — keeping the structure
        // identical makes future maintenance easier.
        scopeguard::ScopeGuard::into_inner(cleanup);
        self.inflight
            .remove_if(&hash_hex, |_k, v| v.ptr_eq(&shared_for_check));

        // Convert the cloneable `SharedCheckpointError` back to a fresh
        // `CheckpointError` per waiter. Every coalesced waiter receives an
        // error of the same variant whose `source()` chains back to the
        // original `Box<dyn Error>` carried by the first caller's failure.
        match shared_result {
            Ok(opt) => Ok(opt),
            Err(shared_err) => Err(shared_err.into_checkpoint_error()),
        }
    }

    fn get_or_create_inflight<F, Fut>(&self, hash_hex: &str, make_future: F) -> SharedRecoveryFuture
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Option<Bytes>, CheckpointError>> + Send + 'static,
    {
        // DashMap::entry::or_insert_with atomically inserts when absent and
        // returns a reference either way — the new future is only built when
        // no other caller has already registered one for this hash.
        let entry = self.inflight.entry(hash_hex.to_owned()).or_insert_with(|| {
            let inner = make_future();
            // b.ohw H-auth-3 equivalent: wrap the inner future in
            // `AssertUnwindSafe(...).catch_unwind()` so a panic in the
            // adapter implementation does not propagate through
            // `futures::Shared` to every coalesced waiter as a panic
            // each. Caught panics surface as
            // `CheckpointError::Permanent` so callers observe a normal
            // error return.
            //
            // The output is mapped from `Result<_, CheckpointError>` to
            // `Result<_, SharedCheckpointError>` so the future's
            // `Output` is `Clone` (required by `futures::Shared`).
            AssertUnwindSafe(inner)
                .catch_unwind()
                .map(|r| match r {
                    Ok(Ok(opt)) => Ok(opt),
                    Ok(Err(e)) => Err(SharedCheckpointError::from_owned(e)),
                    Err(_) => Err(SharedCheckpointError::from_owned(
                        CheckpointError::Permanent {
                            source: "internal panic in checkpoint adapter recovery fetch".into(),
                        },
                    )),
                })
                .boxed()
                .shared()
        });
        entry.clone()
    }

    /// Test-only inspector: report the number of in-flight singleflight
    /// entries currently registered. Used by the SR-3.2 wiring tests in
    /// [`crate::storage::external_values`] to assert the no-miss happy
    /// path never touches the singleflight map (Subtask
    /// `t3.fgj.zc.p6.rc` Success Criterion #5).
    #[cfg(test)]
    pub(crate) fn inflight_len(&self) -> usize {
        self.inflight.len()
    }
}

impl Default for ExtRecoverySingleflight {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    use tokio::sync::oneshot;
    use tokio::sync::Notify;

    use super::*;

    /// Concurrent callers for the same `hash_hex` invoke `make_future` exactly
    /// once and all observe the same `Ok(Some(bytes))` value.
    ///
    /// Deterministic synchronization — see the inline rationale on the test
    /// body for the full design; the short story is a TWO-STAGE rendezvous:
    /// (1) a pre-installer task calls `fetch_once` first and parks its
    /// inner future on a `oneshot::Receiver`; (2) joiner tasks each
    /// MANUALLY POLL their `fetch_once` future ONCE with a noop_waker
    /// — a `Poll::Pending` return proves the joiner has installed onto
    /// the existing `Shared`. Main then waits on an `installed_count`
    /// atomic counter before firing `release_tx`. The previous 50ms
    /// `tokio::time::sleep` was a wall-clock kludge that masked the
    /// entry-removed-after-resolve race; this rendezvous closes it
    /// deterministically without any wall-clock sleep.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn coalesces_same_hash() {
        // The total coalesced-caller count is `1 (pre-installer) + 16
        // (post-rendezvous joiners) = 17`. The pre-installer's role
        // is explained below: it ensures the inflight entry already
        // exists by the time the 16 joiner workers race in, which is
        // load-bearing for joiner-coalescence (see the long comment
        // on the test rationale).
        const JOINER_COUNT: usize = 16;
        const TOTAL_CALLERS: usize = JOINER_COUNT + 1;

        // Pre-install rationale (race that the previous sleep-based
        // test was implicitly working around):
        //
        // `ExtRecoverySingleflight::fetch_once` removes the inflight
        // entry from the DashMap as soon as the inner future resolves
        // (the "entry removed after resolve" invariant —
        // `entry_removed_after_resolve` test pins this). If 16
        // workers race onto `fetch_once` simultaneously, there is NO
        // guarantee that all 16 hit `get_or_create_inflight` BEFORE
        // the first one's `shared.await` resolves; with 4 tokio
        // worker threads, a fast first caller can complete the entire
        // round-trip before slower callers reach the DashMap shard
        // lock — leaving later callers to install a SECOND inflight
        // entry and run the closure a second time. The original 50ms
        // sleep was a wall-clock kludge that masked this race by
        // delaying inner-future resolution.
        //
        // The deterministic fix is a TWO-STAGE rendezvous:
        //
        //   1. A "pre-installer" task calls `fetch_once` FIRST and
        //      parks the inner future on a oneshot receiver. The
        //      `preinstaller_started` Notify fires from inside the
        //      closure body after the inflight entry is fully
        //      installed and the closure is about to `rx.await`.
        //      Main awaits that notification before spawning joiners,
        //      so by the time joiners run the inflight entry exists
        //      and the inner future is mid-poll.
        //
        //   2. Each joiner increments `installed_count` AFTER its
        //      `fetch_once` future has been driven past
        //      `get_or_create_inflight` (i.e., the joiner has
        //      installed onto the existing `Shared`). The "driven
        //      past" guarantee comes from a `noop_waker`-driven
        //      single poll of the boxed future — `Poll::Pending`
        //      proves the future has executed step 1 of
        //      `fetch_once` (synchronously, before any `.await`) and
        //      is now parked.
        //
        //   3. After all joiners have signaled installation,
        //      `installed_count == JOINER_COUNT`. Main observes that
        //      via a bounded poll, then fires `release_tx` to
        //      resolve the pre-installer's inner future. Every
        //      coalesced waiter (pre-installer + 16 joiners) sees
        //      the same `Bytes::from_static(b"hello")`.
        let sf = Arc::new(ExtRecoverySingleflight::new());
        let call_count = Arc::new(AtomicUsize::new(0));
        let installed_count = Arc::new(AtomicUsize::new(0));
        let preinstaller_started = Arc::new(Notify::new());
        let (release_tx, release_rx) = oneshot::channel::<()>();
        let release_rx = std::sync::Mutex::new(Some(release_rx));

        let sf_pre = Arc::clone(&sf);
        let call_count_pre = Arc::clone(&call_count);
        let preinstaller_started_pre = Arc::clone(&preinstaller_started);
        let preinstaller = tokio::spawn(async move {
            sf_pre
                .fetch_once("abc".to_owned(), move || async move {
                    call_count_pre.fetch_add(1, Ordering::SeqCst);
                    let rx = release_rx
                        .lock()
                        .expect("release_rx mutex must not be poisoned")
                        .take()
                        .expect(
                            "release_rx must be present on the singleflight winner's \
                             run; a None here means the closure ran twice for the same \
                             key — singleflight regression",
                        );
                    preinstaller_started_pre.notify_one();
                    rx.await.expect("release_tx must not be dropped");
                    Ok(Some(Bytes::from_static(b"hello")))
                })
                .await
        });

        // Wait for the pre-installer to install + park. From this
        // point onward, the inflight entry for key "abc" exists and
        // the inner future is parked on `rx.await`.
        preinstaller_started.notified().await;

        // Spawn joiners. Each joiner manually polls its fetch_once
        // future ONCE with a noop_waker — that single poll executes
        // `fetch_once`'s synchronous step 1
        // (`get_or_create_inflight`) and then suspends at
        // `shared.await`. After `Poll::Pending` is observed, we are
        // certain the joiner has installed onto the existing
        // `Shared`; we then increment `installed_count` and proceed
        // to await the future normally.
        let mut handles = Vec::new();
        for _ in 0..JOINER_COUNT {
            let sf = Arc::clone(&sf);
            let call_count = Arc::clone(&call_count);
            let installed_count = Arc::clone(&installed_count);
            handles.push(tokio::spawn(async move {
                let fut = sf.fetch_once("abc".to_owned(), move || async move {
                    // This closure must NOT run if the pre-installer
                    // is correctly holding the inflight entry.
                    call_count.fetch_add(1, Ordering::SeqCst);
                    panic!(
                        "joiner closure body ran — singleflight regression \
                         (pre-installer's inflight entry was evicted before this \
                         joiner installed)"
                    );
                });
                let mut fut = Box::pin(fut);

                // Manual single poll with a noop_waker. This drives
                // `fetch_once`'s synchronous prelude (the
                // `get_or_create_inflight` step) and parks at
                // `shared.await`. `Poll::Pending` is the expected
                // and load-bearing outcome — the joiner has now
                // installed onto the pre-existing `Shared`.
                //
                // The `Context`/`Waker` are scoped to this block and
                // dropped before the subsequent `.await` so the
                // outer async block remains `Send`-able.
                let early_ready = {
                    let noop_waker = futures::task::noop_waker();
                    let mut cx = std::task::Context::from_waker(&noop_waker);
                    matches!(fut.as_mut().poll(&mut cx), std::task::Poll::Ready(_))
                };
                if early_ready {
                    panic!(
                        "joiner fetch_once future resolved on the FIRST poll — \
                         this can only happen if the pre-installer's inner future \
                         had already resolved (which means it was never properly \
                         parked) OR if the inflight entry never existed when this \
                         joiner ran (an installation regression in fetch_once)."
                    );
                }

                // Layout invariant witness: inflight map must contain the
                // entry by the time the joiner observes Pending, proving
                // `get_or_create_inflight` synchronous prelude executed
                // before `shared.await` parked the future.
                assert!(
                    sf.inflight.contains_key("abc"),
                    "inflight entry for \"abc\" must be present immediately \
                     after this joiner's noop-poll returned Pending; absence \
                     here means either the pre-installer's entry was already \
                     evicted (it should still be parked on `rx.await`) or \
                     `get_or_create_inflight` failed to atomically install \
                     onto the existing `Shared` — both indicate a regression \
                     in `fetch_once`'s synchronous prelude (b.m52 layout-\
                     invariant: FIRST `.await` MUST be `shared.await`)"
                );

                installed_count.fetch_add(1, Ordering::Release);

                // Now await the future normally. The pre-installer's
                // resolution will wake all coalesced waiters,
                // including this one.
                fut.await
            }));
        }

        // Wait until all joiners have installed.
        let poll_deadline = std::time::Instant::now() + Duration::from_secs(5);
        while installed_count.load(Ordering::Acquire) < JOINER_COUNT {
            if std::time::Instant::now() >= poll_deadline {
                panic!(
                    "joiner installation timed out: only {} of {} joiners installed \
                     within 5s",
                    installed_count.load(Ordering::Acquire),
                    JOINER_COUNT
                );
            }
            tokio::task::yield_now().await;
        }

        // All joiners installed. Release the pre-installer's inner
        // future. The oneshot send stores the value; the receiver
        // consumes it on its next poll — no lost-wakeup window.
        release_tx
            .send(())
            .expect("release_rx must still be live (held inside the inner future)");

        // Pre-installer must succeed and return the canonical value.
        let pre_result = preinstaller
            .await
            .expect("pre-installer task must not panic")
            .expect("pre-installer fetch_once must succeed");
        assert_eq!(pre_result, Some(Bytes::from_static(b"hello")));

        // All 16 joiners must succeed and return the same value.
        for h in handles {
            let result = h.await.unwrap().expect("joiner fetch_once should succeed");
            assert_eq!(result, Some(Bytes::from_static(b"hello")));
        }

        // The load-bearing assertion: exactly ONE call to
        // `make_future`'s closure body across all 17 callers.
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "make_future must be invoked exactly once for {TOTAL_CALLERS} coalesced callers"
        );
    }

    /// Concurrent calls for DIFFERENT `hash_hex` values invoke their closures
    /// independently.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn independent_for_different_hashes() {
        let sf = Arc::new(ExtRecoverySingleflight::new());
        let call_count = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for i in 0..8u32 {
            let sf = Arc::clone(&sf);
            let call_count = Arc::clone(&call_count);
            handles.push(tokio::spawn(async move {
                let bytes = Bytes::copy_from_slice(&i.to_le_bytes());
                let key = format!("hash-{i}");
                let returned_bytes = bytes.clone();
                sf.fetch_once(key, move || async move {
                    call_count.fetch_add(1, Ordering::SeqCst);
                    Ok(Some(returned_bytes))
                })
                .await
                .map(|opt| (i, opt))
            }));
        }

        for h in handles {
            let (i, opt) = h.await.unwrap().expect("fetch_once should succeed");
            assert_eq!(opt, Some(Bytes::copy_from_slice(&i.to_le_bytes())));
        }
        assert_eq!(
            call_count.load(Ordering::SeqCst),
            8,
            "each distinct hash must invoke its own make_future exactly once"
        );
    }

    /// After the shared future resolves the inflight entry is removed; the
    /// next call for the same hash invokes `make_future` again (no stale
    /// joining onto a resolved future).
    #[tokio::test]
    async fn entry_removed_after_resolve() {
        let sf = ExtRecoverySingleflight::new();
        let calls = Arc::new(AtomicUsize::new(0));

        for _ in 0..3 {
            let calls = Arc::clone(&calls);
            let out = sf
                .fetch_once("k".to_owned(), move || async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(Some(Bytes::from_static(b"v")))
                })
                .await
                .unwrap();
            assert_eq!(out, Some(Bytes::from_static(b"v")));
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "sequential calls must each invoke make_future (entry is removed after resolve)"
        );
        assert_eq!(sf.inflight.len(), 0, "no leftover inflight entries");
    }

    /// Cancellation safety: a caller dropped mid-`.await` does NOT leak the
    /// inflight entry. The scope guard removes the entry via `Shared::ptr_eq`
    /// match on the Drop path.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cancellation_does_not_leak_entry() {
        let sf = Arc::new(ExtRecoverySingleflight::new());
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());

        let sf_for_task = Arc::clone(&sf);
        let started_for_task = Arc::clone(&started);
        let release_for_task = Arc::clone(&release);
        let task = tokio::spawn(async move {
            sf_for_task
                .fetch_once("cancel-me".to_owned(), move || async move {
                    started_for_task.notify_waiters();
                    release_for_task.notified().await;
                    Ok(Some(Bytes::from_static(b"unreached")))
                })
                .await
        });

        // Wait until the inner future starts so we know the entry has been
        // installed.
        started.notified().await;
        assert_eq!(sf.inflight.len(), 1, "inflight entry should be installed");

        // Cancel the only waiter; the inner future has not resolved.
        task.abort();
        let _ = task.await;

        // Give DashMap entry removal (via the scope guard's Drop) time to
        // propagate.
        for _ in 0..20 {
            if sf.inflight.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            sf.inflight.len(),
            0,
            "inflight entry must not leak after cancellation"
        );

        // Confirm a fresh call for the same key still works (no zombie entry).
        let out = sf
            .fetch_once("cancel-me".to_owned(), || async {
                Ok(Some(Bytes::from_static(b"fresh")))
            })
            .await
            .unwrap();
        assert_eq!(out, Some(Bytes::from_static(b"fresh")));
    }

    /// Panic isolation: a panic in the inner future surfaces as
    /// `Err(CheckpointError::Permanent)` to every coalesced waiter (never a
    /// cascading panic).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn panic_isolated_as_permanent_error() {
        let sf = Arc::new(ExtRecoverySingleflight::new());
        let release = Arc::new(Notify::new());

        let mut handles = Vec::new();
        for _ in 0..8 {
            let sf = Arc::clone(&sf);
            let release = Arc::clone(&release);
            handles.push(tokio::spawn(async move {
                sf.fetch_once("boom".to_owned(), move || async move {
                    release.notified().await;
                    panic!("simulated adapter panic");
                })
                .await
            }));
        }

        tokio::time::sleep(Duration::from_millis(50)).await;
        release.notify_waiters();

        for h in handles {
            let result = h.await.expect("join handle should not propagate panic");
            match result {
                Err(CheckpointError::Permanent { .. }) => {}
                other => panic!("expected Permanent error, got {other:?}"),
            }
        }
        // Inflight map drains after the shared future resolves (with the
        // panic-caught error).
        assert_eq!(sf.inflight.len(), 0, "no leftover inflight entries");
    }

    /// `None` results (NotFound-equivalent) propagate to every coalesced
    /// waiter unchanged.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn coalesces_none_result() {
        let sf = Arc::new(ExtRecoverySingleflight::new());
        let calls = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..4 {
            let sf = Arc::clone(&sf);
            let calls = Arc::clone(&calls);
            handles.push(tokio::spawn(async move {
                sf.fetch_once("missing".to_owned(), move || async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(20)).await;
                    Ok(None)
                })
                .await
            }));
        }

        for h in handles {
            let result = h.await.unwrap().expect("fetch_once should succeed");
            assert!(result.is_none(), "expected None result for missing blob");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1, "coalesced into one call");
    }

    /// Error results (Transient/Permanent) propagate to every coalesced
    /// waiter. We assert each waiter observes an error of the same variant
    /// (the inner `dyn Error` source is not cloneable, so we cannot assert
    /// pointer equality — variant equality is the contract).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn coalesces_error_result() {
        let sf = Arc::new(ExtRecoverySingleflight::new());
        let release = Arc::new(Notify::new());

        let mut handles = Vec::new();
        for _ in 0..4 {
            let sf = Arc::clone(&sf);
            let release = Arc::clone(&release);
            handles.push(tokio::spawn(async move {
                sf.fetch_once("err".to_owned(), move || async move {
                    release.notified().await;
                    Err(CheckpointError::Transient {
                        source: "simulated transient".into(),
                    })
                })
                .await
            }));
        }

        tokio::time::sleep(Duration::from_millis(50)).await;
        release.notify_waiters();

        for h in handles {
            let result = h.await.unwrap();
            assert!(
                matches!(result, Err(CheckpointError::Transient { .. })),
                "expected Transient error, got {result:?}"
            );
        }
    }
}
