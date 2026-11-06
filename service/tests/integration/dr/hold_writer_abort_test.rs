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

//! What aborting one inventory writer's task handle, alone, stops (SDD
//! SR-6.2): the hold entry routine aborts every registered handle before it
//! cancels any token, so a writer's handle must stop the writer's in-flight
//! work by itself, not leave it to the token.
//!
//! - **The external-value GC task**
//!   ([`aborting_the_gc_task_handle_drops_a_parked_pass_and_stops_its_lease_refresh`]).
//!   The GC loop checks its cancel token only between passes and awaits
//!   `process_pending` without racing it against the token, so a pass running
//!   at entry would finish and write if only the token were cancelled. The
//!   handle `spawn_gc_task` returns holds the loop and the `le:gc:leader`
//!   refresh ticker as abort-on-drop handles, so aborting it — and leaving the
//!   token alone — drops the parked pass before its closing write, and the
//!   lease stops being refreshed.
//!
//! An aborted writer's store call queued behind a refused reconnect — the
//! reference-count reconciler's lease refresh — is never written once the
//! store is back: `storage::abandoned_write_test`'s T7.
//!
//! # Waits
//!
//! [`SETTLE`] (10 s) bounds each awaited event: the pass parking, the handle
//! ending, the pass's drop. [`REFRESH_WINDOW`] (3 s) is the span over which a
//! live 1 s refresh ticker would have refreshed the lease at least twice.
//! Worst case about 35 s; a healthy run takes about 3 s.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::KeysInterface;
use tokio_util::sync::CancellationToken;

use live_edit::storage::external_values::{spawn_gc_task, GcPolicy, ValkeyExternalValueStore};
use live_edit::storage::test_support::TestValkey;

use super::hold_support::{GcPark, ShimExtStore};

/// Bound on each awaited event. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// How long the test watches the lease after the abort: the GC refresh ticker
/// renews every second, so a ticker still running would have reset the TTL
/// to 30 s at least twice in this span.
const REFRESH_WINDOW: Duration = Duration::from_secs(3);

/// Poll `cond` until it holds, within [`SETTLE`].
async fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while !cond() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} did not happen within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

async fn pttl(tv: &TestValkey, key: &str) -> i64 {
    tv.client().pool.pttl(key).await.expect("PTTL must succeed")
}

/// Aborting only the handle `spawn_gc_task` returns, with its token left
/// uncancelled, while a GC pass is parked mid-pass: the pass is dropped and
/// never reaches its closing write, and `le:gc:leader` is no longer refreshed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn aborting_the_gc_task_handle_drops_a_parked_pass_and_stops_its_lease_refresh() {
    let tv = TestValkey::setup().await;
    let finished = format!("{}hold-test:gc-pass-finished", tv.prefix());
    let store = Arc::new(
        ShimExtStore::new(ValkeyExternalValueStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ))
        .with_gc_park(GcPark::new(tv.client().clone(), finished.clone())),
    );
    // Never cancelled before the assertions: only the handle is aborted.
    let token = CancellationToken::new();
    let handle = spawn_gc_task(
        Arc::clone(&store) as _,
        tv.client().clone(),
        tv.key_schema().clone(),
        crate::common::test_leader("hold-gc-abort"),
        GcPolicy::PRODUCTION,
        token.clone(),
        None,
    );
    store.gc_park().park.entered("the first GC pass").await;
    let lease = tv.key_schema().gc_leader_lock();
    assert!(
        pttl(&tv, &lease).await > 0,
        "precondition: the GC task holds its lease"
    );

    handle.abort();
    let ended = tokio::time::timeout(SETTLE, handle)
        .await
        .expect("the aborted handle must end");
    assert!(
        ended
            .expect_err("an aborted handle ends with a JoinError")
            .is_cancelled(),
        "the handle ends aborted"
    );
    let dropped = Arc::clone(&store.gc_park().pass_dropped);
    wait_until("the parked GC pass to be dropped", || {
        dropped.load(Ordering::SeqCst)
    })
    .await;
    let ttl_at_abort = pttl(&tv, &lease).await;

    // A pass still alive would now run to its closing write, and a ticker
    // still alive would renew the lease.
    store.gc_park().park.release();
    tokio::time::sleep(REFRESH_WINDOW).await;

    let finished_written: i64 = tv
        .client()
        .pool
        .exists(&finished)
        .await
        .expect("EXISTS must succeed");
    assert_eq!(
        finished_written, 0,
        "the pass parked at the abort must never reach its write"
    );
    let ttl_after = pttl(&tv, &lease).await;
    assert!(
        ttl_after < 0 || ttl_after <= ttl_at_abort - 2_000,
        "the lease must not be refreshed after the abort: {ttl_at_abort} ms at the abort, \
         {ttl_after} ms {REFRESH_WINDOW:?} later"
    );
    assert!(
        !token.is_cancelled(),
        "the token was never involved: the abort alone stopped the task"
    );
    token.cancel();
}
