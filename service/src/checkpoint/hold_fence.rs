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

//! The hold write fence of the on-access repair and restore paths (SDD
//! SR-6.2): one `CancellationToken`, cancelled at a disaster-recovery hold's
//! entry and at a stale hold verdict's (SR-6.5), that stops the object-id
//! counter repair, the layer restore, the session restore and the
//! external-value fetch-on-miss rewrite from sending any new store write
//! after it.
//!
//! These paths run inside calls that were already past the LiveService
//! gate when it closed — unary calls such as `CreateLayer`, and the setup or
//! catch-up of the deadline-exempt `SubscribeReadLayer` / `JoinSession` — or on
//! a detached task such a call started. The gate does not stop them, and on
//! the streaming and detached paths no RPC deadline bounds them either.
//! Each path checks the fence immediately before each store write it would
//! send ([`is_closed`]); the three restore paths also race it against their
//! backup-store fetch ([`closed`]). The fetch-on-miss rewrite
//! (`storage::external_values`, the store's `recovery_hold_fence`) does not:
//! its fetch is single-flighted with other reads and writes nothing.
//!
//! A closed fence refuses the write: the path returns the failure outcome it
//! already has for a transient failure, carrying [`HOLD_FENCE_REFUSAL`] or,
//! from a storage-side primitive (the fetch-on-miss rewrite's among them),
//! `storage::client::HOLD_FENCE_UNSENT`.
//!
//! # The bound after hold entry
//!
//! Every store write these paths would send after entry is refused: each
//! checks the fence immediately before each write it sends. The fence does
//! not stop a write already handed to fred at entry; fred's give-up rule
//! bounds it (below). It is one of:
//!
//! - one command: the counter repair's conditional seed (`EVALSHA`) or its
//!   establish-at-zero `SET NX`, the layer restore's stale-stream `UNLINK`,
//!   the `EVALSHA` of `ext_release_enqueue.lua` that enqueues a committed
//!   materialisation's owed external-value releases, or the session
//!   restore's record step (the `EVALSHA` of `session_record_restore.lua`);
//! - one `EXEC`: the layer restore's residue wipe or materialisation, or the
//!   session restore's path-index step (Step A). The check sits after the
//!   step's reads and queueing, and `MULTI`, the queued commands and `EXEC`
//!   are handed to fred as one unit after it;
//! - one external-value pre-store chunk of the layer restore: at most
//!   `STORE_VALUES_MAX_INFLIGHT` single-command `EVALSHA`s, sent together;
//! - one fetch-on-miss rewrite: one value's single-command `EVALSHA` of
//!   `ext_store_value.lua`, per read that missed the value.
//!
//! A re-send is a new send and is checked like one: each attempt of the
//! counter writes' transient retry (`CounterStore`'s `*_unless_held`
//! primitives), each retry of the stale-stream `UNLINK` and of the release
//! enqueue, and — for the conditional seed, each pre-store value, the
//! fetch-on-miss rewrite, the release enqueue and the session restore's
//! record step — a `NOSCRIPT` reload's
//! `SCRIPT LOAD` and re-sent
//! `EVALSHA` (`storage::client::evalsha_with_fenced_reload`; a store that lost
//! its keyspace has usually lost its script cache too).
//!
//! Each runs under the configured Valkey command timeout T
//! (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`; none of these writes uses the
//! size-scaled `txn_timeout`). fred writes a write already handed to it at
//! entry — one command or a `MULTI … EXEC` — only while its caller still
//! waits: until its timer T fires, fred answers it with an error, or its
//! future is dropped (the RPC deadline, or a stream the hold ending ended).
//! Once the caller gives up, fred writes no frame of it not yet written,
//! whether it waits behind a reconnect, in the retry buffer or behind a
//! redirect, its own re-issue on a dropped connection included; a cut
//! `MULTI` is discarded, by a `DISCARD` for a pipeline (`WatchedTxn`) or by
//! closing the connection for a fred `Transaction` (SDD §15.3 *Vendored
//! fred* change 9). So a write handed over at entry is written within T of
//! entry, or never. The bound needs T > 0 (at 0 the client bounds none of
//! these writes), which `Config::from_env` enforces at startup.
//!
//! **The residual.** A frame fred had already written before its caller
//! gave up is not recalled: it reaches the socket within one of fred's
//! bounded write or flush waits (`UNRESPONSIVE_MAX_TIMEOUT`, 45 s), or is
//! dropped with its connection. From the kernel's send queue on, when the
//! store runs it is the network's and the store's, with no client-side bound
//! (SDD §15.3 *Vendored fred* change 9).
//!
//! The release enqueue is skipped, not delayed: once the fence is closed —
//! before the enqueue starts, between its attempts, after its reads, or
//! before a `NOSCRIPT` reload — its script is not sent, and the releases leak
//! their references (the safe
//! direction, which the disaster-recovery restore Job's recount or the
//! reference-count reconciler corrects), counted on
//! `live_edit_ext_ref_release_enqueue_failed_total` with an INFO line naming
//! the object (`LayerStore::enqueue_ext_releases_unless_held`).
//!
//! Each path reads its fence through one accessor (`hold_fence()` on
//! `CheckpointObjectIdCounterRepair`, `RecoveryHook` and
//! `SessionRecoveryHook`), so every check of a path goes through one site:
//! the layer restore hands that accessor's value to `apply_restore`, which
//! passes it to the residue wipe, the pre-store and the materialisation (and
//! on to its release enqueue); the counter repair hands it to `CounterStore`'s
//! fenced primitives; and the session restore hands it to
//! `apply_session_checkpoint`. The fetch-on-miss rewrite has no such hook:
//! `main.rs` hands the same token to `ValkeyExternalValueStore` once
//! (`with_recovery_hold_fence`), and its `store_recovered_values` passes it
//! to the store path. In `storage`, every check reads the one helper
//! `storage::client::hold_fence_closed`.

use std::future::Future;

use tokio_util::sync::CancellationToken;

/// The message every refusal by the hold write fence carries. It names both
/// causes that close the fence: a fleet hold, and a stale hold verdict.
pub(crate) const HOLD_FENCE_REFUSAL: &str =
    "refused: this replica entered a disaster-recovery hold or lost contact with the store's \
     hold state, which stops every on-access repair and restore write";

/// Whether `fence` is closed: `false` when there is no fence.
pub(crate) fn is_closed(fence: Option<&CancellationToken>) -> bool {
    fence.is_some_and(CancellationToken::is_cancelled)
}

/// Completes when `fence` closes, at once if it already is closed, and never
/// when there is no fence.
///
/// For racing a future that sends no store write against the fence: the
/// caller writes the race as its own `tokio::select!`, `biased`, with this
/// branch first, so a fence closed before the future finishes (or already
/// closed) wins and the future is dropped:
///
/// ```ignore
/// let fetch = tokio::select! {
///     biased;
///     () = hold_fence::closed(fence) => return refused(),
///     fetch = the_fetch() => fetch,
/// };
/// ```
///
/// Not a wrapper that takes the future: an `async fn` keeps a future passed
/// to it as an argument as well as the future it awaits, so the race would
/// hold the fetch twice in the caller's future (the layer restore's sits in
/// every `UpdateLayer` call's). The refusal returns from inside the race
/// rather than through an `Option` around the fetch's output, which in an
/// unoptimised build is one more copy of that output held in the caller's
/// future.
pub(crate) fn closed(fence: Option<&CancellationToken>) -> impl Future<Output = ()> + '_ {
    match fence {
        Some(fence) => futures::future::Either::Left(fence.cancelled()),
        None => futures::future::Either::Right(std::future::pending()),
    }
}
