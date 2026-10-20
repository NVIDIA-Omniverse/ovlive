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

// Shared full-jitter retry helper for storage operations whose retriable
// failures surface as `ServiceError::Aborted`.
//
// Factored out of `LayerStore::update_layer` (b.vi2) so that `SessionStore::create`
// can reuse the same backoff shape under WATCH contention (b.91j H-stor-3). The
// helper itself is pool-agnostic — callers supply an `FnMut(attempt)` closure
// that returns the single-attempt future, and the helper handles the loop,
// the full-jitter sleep on `Aborted`, and the exhaustion message.
//
// Non-retriable errors (any `ServiceError` variant other than `Aborted`) are
// surfaced immediately without consuming the retry budget. A caller whose
// attempts can also ask to wait without failing (`update_layer` under a
// snapshot fence) or to stop and be resumed (`update_layer` needing a larger
// upload charge) uses `retry_aborted_with_jitter_and_uncounted_waits`, the
// same loop with those two extra outcomes.

use std::convert::Infallible;
use std::future::Future;
use std::time::Duration;

use crate::error::ServiceError;

/// Parameters for [`retry_aborted_with_jitter`].
#[derive(Debug, Clone, Copy)]
pub struct RetryConfig {
    /// Total attempts (first try + retries). Values below 1 are clamped to 1
    /// by the helper — zero would disable the first attempt.
    pub max_attempts: u32,
    /// Starting full-jitter backoff window, in milliseconds. The actual sleep
    /// before retry `N` is drawn uniformly from `[0, min(2^(N-1) * base, cap)]`.
    pub initial_delay_ms: u64,
    /// Upper bound on the backoff window (ms). Once the doubling window reaches
    /// this cap, further attempts keep sampling from `[0, cap]`.
    pub cap_delay_ms: u64,
}

/// One attempt's verdict under [`retry_aborted_with_jitter_and_uncounted_waits`].
#[derive(Debug)]
pub enum AttemptOutcome<T, W, S = Infallible> {
    /// The operation finished; return `T`.
    Done(T),
    /// The operation could not proceed for a reason that is not a failure —
    /// something it must wait out (for `update_layer`, a live snapshot fence,
    /// Spec SDD SR-5.4). Re-run it after a wait the caller sizes from `W`,
    /// without spending a retry attempt.
    WaitUncounted(W),
    /// The operation stopped, holding nothing, for something only the
    /// helper's caller can do between attempts (for `update_layer`, wait on
    /// the upload gate for a larger charge with its per-object permits given
    /// back, SDD §5.4). The helper returns [`RetryExit::Suspended`] with the
    /// attempt number, so the caller can resume the same attempt, spending
    /// no retry.
    Suspend(S),
}

/// How [`retry_aborted_with_jitter_and_uncounted_waits`] ended without an
/// error.
#[derive(Debug)]
pub enum RetryExit<T, S> {
    /// An attempt finished with `T`.
    Done(T),
    /// Attempt `attempt` suspended with `payload`
    /// ([`AttemptOutcome::Suspend`]). Pass `attempt` back as the start
    /// attempt to resume it: the suspension spent nothing of the `Aborted`
    /// budget.
    Suspended { attempt: u32, payload: S },
}

/// Run `op` up to `config.max_attempts` times, retrying on
/// [`ServiceError::Aborted`] with full-jitter exponential backoff.
///
/// Full jitter (b.vi2, mirrors `src/auth/mod.rs`): sleep a uniform random delay
/// in `[0, min(2^(attempt-1) * base, cap)]` before each retry. `attempt` is
/// 1-based — the first retry samples from `[0, base]` and doubles thereafter.
///
/// On exhaustion the helper wraps the final `Aborted` message with the
/// attempt count, so operators can distinguish retry-exhaustion from a
/// single-shot conflict. `op_name` is the label that prefixes the exhaustion
/// message (e.g. "create_session").
///
/// Non-`Aborted` failures (Internal, InvalidArgument, AlreadyExists, NotFound,
/// PermissionDenied, Unauthenticated, FailedPrecondition, Cancelled) propagate
/// immediately — only `Aborted` is treated as retriable. `update_layer` runs
/// the same loop through [`retry_aborted_with_jitter_and_uncounted_waits`].
pub async fn retry_aborted_with_jitter<F, Fut, T>(
    op_name: &'static str,
    config: RetryConfig,
    mut op: F,
) -> Result<T, ServiceError>
where
    F: FnMut(u32) -> Fut,
    Fut: Future<Output = Result<T, ServiceError>>,
{
    let exit = retry_aborted_with_jitter_and_uncounted_waits(
        op_name,
        config,
        1,
        move |attempt| {
            let attempt_fut = op(attempt);
            async move {
                attempt_fut
                    .await
                    .map(AttemptOutcome::<T, Infallible, Infallible>::Done)
            }
        },
        |never: Infallible| match never {},
    )
    .await?;
    match exit {
        RetryExit::Done(val) => Ok(val),
        RetryExit::Suspended { payload, .. } => match payload {},
    }
}

/// [`retry_aborted_with_jitter`], plus attempts that ask to wait rather than
/// fail, and attempts that suspend:
///
/// - an `Ok(`[`AttemptOutcome::WaitUncounted`]`(w))` is handed to
///   `on_wait`, which answers how long to sleep — or an error that ends the
///   call — and `op` is then re-run with the **same** attempt number. Such a
///   wait spends nothing of the `Aborted` budget and is not a backoff step:
///   the backoff window of the next `Aborted` retry is the one it would have
///   had without the wait;
/// - an `Ok(`[`AttemptOutcome::Suspend`]`(s))` ends the loop with
///   [`RetryExit::Suspended`], carrying `s` and the attempt's number. The
///   caller does what the attempt asked, then resumes it by calling again
///   with that number as `start_attempt`: like a wait, a suspension spends
///   nothing of the `Aborted` budget.
///
/// The first attempt run is `start_attempt` (1 for a fresh call); the
/// budget is still `config.max_attempts` in all.
///
/// `on_wait` is synchronous so it can keep its own state (a cap on the total
/// wait, say) across the call; the helper does the sleeping.
pub async fn retry_aborted_with_jitter_and_uncounted_waits<F, Fut, T, W, S, G>(
    op_name: &'static str,
    config: RetryConfig,
    start_attempt: u32,
    mut op: F,
    mut on_wait: G,
) -> Result<RetryExit<T, S>, ServiceError>
where
    F: FnMut(u32) -> Fut,
    Fut: Future<Output = Result<AttemptOutcome<T, W, S>, ServiceError>>,
    G: FnMut(W) -> Result<Duration, ServiceError>,
{
    let max_attempts = config.max_attempts.max(1);
    let base_ms = config.initial_delay_ms.max(1);
    let cap_ms = config.cap_delay_ms.max(base_ms);

    let mut attempt = start_attempt.clamp(1, max_attempts);
    while attempt <= max_attempts {
        match op(attempt).await {
            Ok(AttemptOutcome::Done(val)) => return Ok(RetryExit::Done(val)),
            Ok(AttemptOutcome::Suspend(payload)) => {
                return Ok(RetryExit::Suspended { attempt, payload });
            }
            Ok(AttemptOutcome::WaitUncounted(wait)) => {
                let sleep = on_wait(wait)?;
                tracing::debug!(
                    op = op_name,
                    attempt,
                    sleep_ms = sleep.as_millis() as u64,
                    "attempt must wait; re-running it after the wait without spending a retry"
                );
                tokio::time::sleep(sleep).await;
                // Same `attempt`: the wait is not a retry.
                continue;
            }
            Err(ServiceError::Aborted(msg)) if attempt < max_attempts => {
                // Full jitter: delay ∈ [0, min(2^(attempt-1) * base, cap)].
                // Clamp the shift to 31 before multiplying to avoid u64
                // overflow on large attempt counts — the `min(cap_ms)` below
                // makes any shift past ~log2(cap/base) functionally equivalent.
                let shift = (attempt - 1).min(31);
                let window_ms = base_ms.checked_shl(shift).unwrap_or(cap_ms).min(cap_ms);
                let sleep_ms = {
                    use rand::Rng;
                    rand::rng().random_range(0..=window_ms)
                };
                tracing::debug!(
                    op = op_name,
                    attempt,
                    max_attempts,
                    backoff_window_ms = window_ms,
                    sleep_ms,
                    error = %msg,
                    "retriable failure, retrying with full-jitter backoff"
                );
                tokio::time::sleep(Duration::from_millis(sleep_ms)).await;
                attempt += 1;
            }
            Err(ServiceError::Aborted(msg)) => {
                tracing::warn!(
                    op = op_name,
                    attempts = max_attempts,
                    last_cause = %msg,
                    "retry budget exhausted"
                );
                return Err(ServiceError::Aborted(format!(
                    "{op_name} retriable failures, {max_attempts} attempts exhausted (last cause: {msg})"
                )));
            }
            Err(other) => return Err(other),
        }
    }

    // Unreachable: the loop either returns on success, on exhaustion, or
    // on a non-retriable error. Surface a defensive Internal if somehow
    // reached so the caller doesn't silently hang.
    Err(ServiceError::Internal(format!(
        "{op_name} retry loop exited without result"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    fn cfg(max: u32) -> RetryConfig {
        RetryConfig {
            max_attempts: max,
            initial_delay_ms: 1,
            cap_delay_ms: 2,
        }
    }

    #[tokio::test]
    async fn returns_success_on_first_attempt_without_delay() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = calls.clone();
        let result: Result<u32, _> = retry_aborted_with_jitter("op", cfg(5), move |_| {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(42)
            }
        })
        .await;
        assert_eq!(result.unwrap(), 42);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retries_on_aborted_and_eventually_succeeds() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = calls.clone();
        let result: Result<u32, _> = retry_aborted_with_jitter("op", cfg(5), move |attempt| {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                if attempt < 3 {
                    Err(ServiceError::Aborted(format!("conflict {attempt}")))
                } else {
                    Ok(7)
                }
            }
        })
        .await;
        assert_eq!(result.unwrap(), 7);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn exhausts_budget_and_wraps_last_cause() {
        let result: Result<u32, _> = retry_aborted_with_jitter("op", cfg(3), |_| async {
            Err::<u32, _>(ServiceError::Aborted("conflict".to_owned()))
        })
        .await;
        match result.unwrap_err() {
            ServiceError::Aborted(msg) => {
                assert!(msg.contains("op"), "missing op name: {msg}");
                assert!(msg.contains("3 attempts"), "missing attempt count: {msg}");
                assert!(msg.contains("conflict"), "missing last cause: {msg}");
            }
            other => panic!("expected Aborted, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_aborted_error_bypasses_retries() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = calls.clone();
        let result: Result<u32, _> = retry_aborted_with_jitter("op", cfg(5), move |_| {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err::<u32, _>(ServiceError::Internal("boom".to_owned()))
            }
        })
        .await;
        assert!(matches!(result, Err(ServiceError::Internal(_))));
        assert_eq!(calls.load(Ordering::SeqCst), 1, "must not retry");
    }

    #[tokio::test]
    async fn zero_max_attempts_clamped_to_one() {
        let result: Result<u32, _> =
            retry_aborted_with_jitter("op", cfg(0), |_| async { Ok(99) }).await;
        assert_eq!(result.unwrap(), 99);
    }

    // ── `retry_aborted_with_jitter_and_uncounted_waits` ───────────────────
    //
    // The loop `update_layer` runs (Spec SDD SR-5.4): a `WaitUncounted`
    // outcome — a live snapshot fence — re-runs the same attempt after the
    // wait `on_wait` sizes, spending nothing of the `Aborted` budget.

    /// One scripted outcome per call of `op`, in order.
    enum Step {
        Done(u32),
        Wait(u64),
        Abort,
    }

    /// Drive the helper through `steps`; returns its result, the attempt
    /// number `op` saw on each call, and every value handed to `on_wait`.
    async fn run_steps(
        max_attempts: u32,
        steps: Vec<Step>,
        on_wait: impl FnMut(u64) -> Result<Duration, ServiceError>,
    ) -> (Result<u32, ServiceError>, Vec<u32>, Vec<u64>) {
        let steps = Arc::new(std::sync::Mutex::new(
            steps.into_iter().collect::<std::collections::VecDeque<_>>(),
        ));
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let waits = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut on_wait = on_wait;
        let result = {
            let steps = Arc::clone(&steps);
            let seen = Arc::clone(&seen);
            let waits = Arc::clone(&waits);
            retry_aborted_with_jitter_and_uncounted_waits(
                "op",
                cfg(max_attempts),
                1,
                move |attempt| {
                    seen.lock().unwrap().push(attempt);
                    let step = steps
                        .lock()
                        .unwrap()
                        .pop_front()
                        .expect("op called more often than the test scripted");
                    async move {
                        match step {
                            Step::Done(v) => Ok(AttemptOutcome::<u32, u64>::Done(v)),
                            Step::Wait(w) => Ok(AttemptOutcome::WaitUncounted(w)),
                            Step::Abort => Err(ServiceError::Aborted("conflict".to_owned())),
                        }
                    }
                },
                move |w: u64| {
                    waits.lock().unwrap().push(w);
                    on_wait(w)
                },
            )
            .await
            .map(|exit| match exit {
                RetryExit::Done(v) => v,
                RetryExit::Suspended { payload, .. } => match payload {},
            })
        };
        let seen = seen.lock().unwrap().clone();
        let waits = waits.lock().unwrap().clone();
        (result, seen, waits)
    }

    /// One scripted outcome of a call that can suspend.
    enum SuspendStep {
        Done(u32),
        Abort,
        Suspend(&'static str),
    }

    /// Drive the helper from `start_attempt` through `steps`, whose
    /// `Suspend` ends the call with its payload; returns the exit and the
    /// attempt number `op` saw on each call.
    async fn run_suspendable(
        max_attempts: u32,
        start_attempt: u32,
        steps: Vec<SuspendStep>,
    ) -> (Result<RetryExit<u32, &'static str>, ServiceError>, Vec<u32>) {
        let steps = Arc::new(std::sync::Mutex::new(
            steps.into_iter().collect::<std::collections::VecDeque<_>>(),
        ));
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let result = {
            let seen = Arc::clone(&seen);
            retry_aborted_with_jitter_and_uncounted_waits(
                "op",
                cfg(max_attempts),
                start_attempt,
                move |attempt| {
                    seen.lock().unwrap().push(attempt);
                    let step = steps
                        .lock()
                        .unwrap()
                        .pop_front()
                        .expect("op called more often than the test scripted");
                    async move {
                        match step {
                            SuspendStep::Done(v) => {
                                Ok(AttemptOutcome::<u32, u64, &'static str>::Done(v))
                            }
                            SuspendStep::Abort => Err(ServiceError::Aborted("conflict".to_owned())),
                            SuspendStep::Suspend(s) => Ok(AttemptOutcome::Suspend(s)),
                        }
                    }
                },
                |_: u64| -> Result<Duration, ServiceError> { panic!("no step waits") },
            )
            .await
        };
        let seen = seen.lock().unwrap().clone();
        (result, seen)
    }

    /// A suspension at attempt k ends the call at once with
    /// `Suspended { attempt: k }` and its payload: no sleep and nothing of the
    /// budget spent, and the attempt is not re-run.
    #[tokio::test(start_paused = true)]
    async fn a_suspension_ends_the_call_at_its_attempt_without_sleeping() {
        let started = tokio::time::Instant::now();
        let (result, seen) = run_suspendable(5, 1, vec![SuspendStep::Suspend("grow")]).await;
        assert!(
            matches!(
                result,
                Ok(RetryExit::Suspended {
                    attempt: 1,
                    payload: "grow"
                })
            ),
            "a first-attempt suspension: {result:?}"
        );
        assert_eq!(seen, [1]);
        assert_eq!(
            tokio::time::Instant::now(),
            started,
            "a suspension sleeps nothing"
        );

        let (result, seen) = run_suspendable(
            5,
            1,
            vec![
                SuspendStep::Abort,
                SuspendStep::Abort,
                SuspendStep::Suspend("grow"),
            ],
        )
        .await;
        assert!(
            matches!(
                result,
                Ok(RetryExit::Suspended {
                    attempt: 3,
                    payload: "grow"
                })
            ),
            "a suspension after two aborts carries attempt 3: {result:?}"
        );
        assert_eq!(seen, [1, 2, 3]);
    }

    /// Resumed with `start_attempt` k after k − 1 aborted attempts, the call
    /// has exactly `max − k + 1` attempts left — the suspended attempt itself
    /// re-runs, and the suspension spent none — then wraps the last abort with
    /// the whole budget's count.
    #[tokio::test]
    async fn a_resumed_call_keeps_the_budget_its_aborted_attempts_left() {
        let (result, seen) = run_suspendable(
            5,
            3,
            vec![SuspendStep::Abort, SuspendStep::Abort, SuspendStep::Abort],
        )
        .await;
        assert_eq!(seen, [3, 4, 5], "max − k + 1 = 3 attempts, numbered on");
        match result {
            Err(ServiceError::Aborted(msg)) => {
                assert!(msg.contains("5 attempts"), "the whole budget: {msg}");
                assert!(msg.contains("conflict"), "the last cause: {msg}");
            }
            other => panic!("expected the wrapped abort, got {other:?}"),
        }

        let (result, seen) =
            run_suspendable(5, 3, vec![SuspendStep::Abort, SuspendStep::Done(9)]).await;
        assert!(matches!(result, Ok(RetryExit::Done(9))), "{result:?}");
        assert_eq!(seen, [3, 4]);
    }

    /// `start_attempt` 0 is taken as 1, and one above the budget as the
    /// budget's last attempt.
    #[tokio::test]
    async fn start_attempt_is_clamped_to_the_budget() {
        let (result, seen) = run_suspendable(5, 0, vec![SuspendStep::Done(1)]).await;
        assert!(matches!(result, Ok(RetryExit::Done(1))), "{result:?}");
        assert_eq!(seen, [1], "start 0 runs attempt 1");

        let (result, seen) = run_suspendable(5, 9, vec![SuspendStep::Abort]).await;
        assert_eq!(seen, [5], "start 9 runs the last attempt, 5, alone");
        assert!(
            matches!(&result, Err(ServiceError::Aborted(msg)) if msg.contains("5 attempts")),
            "the last attempt's abort exhausts the budget: {result:?}"
        );
    }

    /// Waits re-run the SAME attempt: a one-attempt budget survives any number
    /// of them, and the wait value reaches `on_wait` verbatim.
    #[tokio::test]
    async fn uncounted_waits_rerun_the_same_attempt_without_spending_the_budget() {
        let (result, seen, waits) = run_steps(
            1,
            vec![Step::Wait(7), Step::Wait(8), Step::Wait(9), Step::Done(42)],
            |_| Ok(Duration::ZERO),
        )
        .await;

        assert_eq!(
            result.unwrap(),
            42,
            "a wait is not a failed attempt: a budget of one must still reach the commit"
        );
        assert_eq!(
            seen,
            vec![1, 1, 1, 1],
            "every re-run keeps attempt number 1"
        );
        assert_eq!(waits, vec![7, 8, 9], "each wait value is handed to on_wait");
    }

    /// Waits interleaved with `Aborted` retries leave the attempt numbering
    /// and the exhaustion point exactly where the `Aborted`s alone put them.
    #[tokio::test]
    async fn uncounted_waits_between_aborts_leave_the_attempt_count_unchanged() {
        let (result, seen, _) = run_steps(
            3,
            vec![
                Step::Abort,
                Step::Wait(1),
                Step::Wait(1),
                Step::Abort,
                Step::Wait(1),
                Step::Done(5),
            ],
            |_| Ok(Duration::ZERO),
        )
        .await;
        assert_eq!(result.unwrap(), 5);
        assert_eq!(seen, vec![1, 2, 2, 2, 3, 3]);

        let (result, seen, _) = run_steps(2, vec![Step::Abort, Step::Wait(1), Step::Abort], |_| {
            Ok(Duration::ZERO)
        })
        .await;
        match result {
            Err(ServiceError::Aborted(msg)) => assert!(
                msg.contains("2 attempts exhausted"),
                "two Aborted outcomes exhaust a budget of two whatever waits sit between them: \
                 {msg}"
            ),
            other => panic!("expected the exhausted Aborted, got {other:?}"),
        }
        assert_eq!(seen, vec![1, 2, 2]);
    }

    /// `on_wait` answering an error ends the call with that error, without
    /// another run of `op` (`update_layer`'s fence-wait cap).
    #[tokio::test]
    async fn an_on_wait_error_ends_the_call_with_that_error() {
        let (result, seen, waits) = run_steps(5, vec![Step::Wait(3)], |_| {
            Err(ServiceError::Unavailable("cap reached".to_owned()))
        })
        .await;
        match result {
            Err(ServiceError::Unavailable(msg)) => assert_eq!(msg, "cap reached"),
            other => panic!("expected on_wait's own error, got {other:?}"),
        }
        assert_eq!(seen, vec![1], "op must not run again after on_wait refused");
        assert_eq!(waits, vec![3]);
    }

    /// The helper sleeps exactly what `on_wait` answered before re-running.
    #[tokio::test(start_paused = true)]
    async fn uncounted_wait_sleeps_the_duration_on_wait_answered() {
        let start = tokio::time::Instant::now();
        let (result, _, _) = run_steps(1, vec![Step::Wait(1), Step::Done(1)], |_| {
            Ok(Duration::from_millis(1_234))
        })
        .await;
        assert_eq!(result.unwrap(), 1);
        assert_eq!(
            tokio::time::Instant::now() - start,
            Duration::from_millis(1_234),
            "on a paused clock the only elapsed time is the one wait"
        );
    }
}
