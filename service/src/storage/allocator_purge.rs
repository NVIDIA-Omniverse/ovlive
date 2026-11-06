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

//! A hook that returns the allocator's unused dirty pages to the OS on
//! request ([`AllocatorPurge`]), for a long task that frees large transients
//! as it goes and must not leave them resident until the allocator's own
//! decay runs. Its one caller today is the accept-loss Job's value removal
//! (`dr::job::execute::discard`, D5), around each of its transactions; no
//! serving process installs one.
//!
//! **No allocator dependency.** The library cannot call the allocator: it is
//! the binary's alone (`src/allocator.rs`, declared only from `main.rs`). So
//! this module declares the hook, and whoever builds the process installs
//! the function: the binary's DR Job (`main.rs`'s `run_dr_job`, linux-gnu
//! only, purging every jemalloc arena), a test (a recording function), or a
//! measurement example. A process that installs none passes `None` where a
//! hook is taken, and its caller charges as if nothing were purged. The
//! module needs only std and tokio. The same pattern as the fan-out thread
//! start hook (`storage::fanout_thread`), but a cloneable handle carried
//! explicitly rather than a once-set registration, so each test holds its
//! own.
//!
//! **The contract.**
//! - The function returns every arena's unused dirty pages to the OS now,
//!   and answers `true` when the allocator accepted the request.
//! - [`AllocatorPurge::purge`] runs it on tokio's blocking pool
//!   (`spawn_blocking`): a purge blocks its thread while it walks the
//!   arenas, and must not stall the other work polled on the caller's
//!   worker. It writes nothing but the allocator's own state; a purge
//!   already running when its caller's future is dropped finishes, harmless.
//! - A `JoinError` (the function panicked, or the runtime is shutting down)
//!   counts as not purged.
//! - The first purge that did not happen sets the handle's failure flag
//!   ([`AllocatorPurge::has_failed`], shared by every clone) and logs one
//!   WARN; later failures log nothing more. A caller reads the flag to stop
//!   pricing its memory as purged.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// The purge function a process installs: return every arena's unused dirty
/// pages to the OS now; `true` when the allocator accepted the request.
pub type PurgeFn = dyn Fn() -> bool + Send + Sync;

/// A handle on the process's allocator purge (module docs). Cheap to clone;
/// every clone calls the same function and shares one failure flag.
#[derive(Clone)]
pub struct AllocatorPurge {
    purge: Arc<PurgeFn>,
    failed: Arc<AtomicBool>,
}

impl fmt::Debug for AllocatorPurge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AllocatorPurge")
            .field("failed", &self.has_failed())
            .finish_non_exhaustive()
    }
}

impl AllocatorPurge {
    /// A handle on `purge`, which must keep the contract of [`PurgeFn`].
    pub fn new(purge: impl Fn() -> bool + Send + Sync + 'static) -> Self {
        Self {
            purge: Arc::new(purge),
            failed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Run the purge on the blocking pool and wait for it: `true` when the
    /// allocator accepted it; `false` when it refused, or when the function
    /// panicked or the runtime is shutting down (a `JoinError`). The first
    /// `false` sets [`Self::has_failed`] and logs the handle's one WARN.
    pub async fn purge(&self) -> bool {
        let purge = Arc::clone(&self.purge);
        let purged = tokio::task::spawn_blocking(move || purge())
            .await
            .unwrap_or(false);
        if !purged && !self.failed.swap(true, Ordering::SeqCst) {
            tracing::warn!(
                "the allocator purge did not run (the allocator refused it, the purge \
                 function panicked, or the runtime is shutting down): \
                 memory freed between purges stays resident until the allocator's own decay \
                 returns it, and the accept-loss Job's value removal charges each removal its \
                 un-purged admission from now on (logged once)"
            );
        }
        purged
    }

    /// Whether any purge through this handle, or a clone of it, did not run.
    pub fn has_failed(&self) -> bool {
        self.failed.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    //! The handle's contract on recording functions (W-T5): where the
    //! function runs, how a panic and a refusal read, the one WARN, and the
    //! flag every clone shares. The real purge is `src/allocator.rs`'s
    //! tests (the binary's test target); D5's use of the handle is
    //! `dr::job::execute::discard`'s tests.

    use std::sync::Mutex;
    use std::thread::ThreadId;

    use super::*;

    /// Everything logged while installed (the thread-local default), as
    /// plain text lines.
    #[derive(Clone, Default)]
    struct Lines(Arc<Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Lines {
        type Writer = Lines;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for Lines {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Lines {
        /// The lines logged at WARN.
        fn warns(&self) -> Vec<String> {
            let text = String::from_utf8(
                self.0
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .clone(),
            )
            .expect("UTF-8 log output");
            text.lines()
                .filter(|line| line.contains("WARN"))
                .map(str::to_owned)
                .collect()
        }
    }

    /// The function runs off the caller's thread (on the blocking pool), so
    /// a purge never stalls the work polled beside its caller: on a
    /// current-thread runtime the recorded thread is not the test's. An
    /// accepted purge answers `true` and leaves the flag clear. Fails with
    /// the function called inline (in-place perturbation of
    /// [`AllocatorPurge::purge`]): the same thread.
    #[tokio::test]
    async fn purge_runs_the_function_off_the_callers_thread() {
        let ran_on: Arc<Mutex<Option<ThreadId>>> = Arc::new(Mutex::new(None));
        let purge = AllocatorPurge::new({
            let ran_on = Arc::clone(&ran_on);
            move || {
                *ran_on.lock().expect("not poisoned") = Some(std::thread::current().id());
                true
            }
        });

        assert!(purge.purge().await, "an accepted purge answers true");

        let ran_on = ran_on
            .lock()
            .expect("not poisoned")
            .expect("the function ran");
        assert_ne!(
            ran_on,
            std::thread::current().id(),
            "the function runs on another thread than its caller's"
        );
        assert!(!purge.has_failed(), "an accepted purge sets no failure");
    }

    /// A function that panics reads as not purged: `false`, and the failure
    /// flag set, on the handle and on a clone made before. Fails with a
    /// `JoinError` read as purged (in-place perturbation of
    /// [`AllocatorPurge::purge`], `unwrap_or(true)`).
    #[tokio::test]
    async fn a_purge_whose_function_panics_reads_false_and_sets_the_failure_flag() {
        let purge = AllocatorPurge::new(|| panic!("the purge function panicked (test)"));
        let clone = purge.clone();

        assert!(!purge.purge().await, "a panic is not a purge");
        assert!(
            purge.has_failed() && clone.has_failed(),
            "the flag is set, and shared by the clone"
        );
    }

    /// Two refused purges, one through a clone, log one WARN between them,
    /// and both handles read the failure; a handle on another function
    /// shares nothing with them. Fails with the WARN logged at every
    /// refusal (in-place perturbation of [`AllocatorPurge::purge`], the
    /// flag's swap no longer guarding it): two.
    #[tokio::test]
    async fn two_refused_purges_log_one_warn_and_every_clone_shares_the_flag() {
        let lines = Lines::default();
        let _guard = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_writer(lines.clone())
                .with_ansi(false)
                .with_max_level(tracing::Level::INFO)
                .finish(),
        );
        let purge = AllocatorPurge::new(|| false);
        let clone = purge.clone();
        let other = AllocatorPurge::new(|| true);

        assert!(!purge.purge().await, "the allocator refused");
        assert!(!clone.purge().await, "and refused again, through the clone");

        let warns = lines.warns();
        assert_eq!(warns.len(), 1, "one WARN for two refusals: {warns:?}");
        assert!(
            warns[0].contains("the allocator purge did not run"),
            "{warns:?}"
        );
        assert!(
            purge.has_failed() && clone.has_failed(),
            "both handles read the failure"
        );
        assert!(!other.has_failed(), "another handle's flag is its own");
    }
}
