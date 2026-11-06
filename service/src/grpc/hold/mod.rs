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

//! The server's reactions to the disaster-recovery hold verdict (SDD SR-5,
//! SR-6): on a serving replica, the gate in front of the LiveService, hold
//! entry, stale-verdict entry, and the exits; and held mode, what a replica
//! that finds the fleet held at startup runs instead.
//!
//! **Held mode** ([`run_held_mode`]). When the startup hold evaluation
//! answers `Held`, `main.rs` runs held mode in place of the rest of the
//! normal startup: a refuse-all LiveService stub ([`held_live_service`], a
//! [`HoldGate`] closed for the hold) with health, reflection and the
//! CapabilitiesService; its own hold poll, health supervisor and scaling
//! source (`held` true), stopping on the process shutdown request (the first
//! SIGTERM or SIGINT); and its own reactor
//! ([`spawn_held_mode_reactor`]), which exits 0 after the jitter when the
//! hold clears, reports readiness `NOT_SERVING` while its verdict is stale
//! and keeps reading, with no time limit, until a read succeeds. It never
//! builds the
//! service, a store or any store- or backup-writing task. The rest of this
//! doc is about normal mode.
//!
//! In normal mode the verdict comes from the hold poll (`crate::dr::hold`),
//! which `main.rs` spawns right after the startup hold evaluation, before any
//! other background task. Its channel feeds two consumers here:
//!
//! - **[`HoldGate`]**, inside
//!   [`live_service_server`](crate::grpc::transport::live_service_server):
//!   open while serving; closed at hold entry, after which it refuses every
//!   LiveService RPC `UNAVAILABLE` `SERVICE_HELD`, naming the latest
//!   verdict's incident, before the auth interceptor runs; or closed at
//!   stale-verdict entry, after which it refuses every one `UNAVAILABLE`
//!   `UPSTREAM_UNAVAILABLE`. Health, reflection and the CapabilitiesService
//!   are not behind it.
//! - **The hold reactor** ([`spawn_hold_reactor`]), which acts on each
//!   verdict:
//!   - `Held` before the serving point ([`HoldServingPoint`]): the process
//!     exits 0 at once ([`exit_held_before_serving`]), so it restarts into
//!     held mode.
//!     Nothing that the hold entry routine stops exists yet, and nothing
//!     spawned after it could be stopped, so the routine never runs before
//!     the serving point.
//!   - `Held` on a serving replica: the hold entry routine ([`enter_hold`]),
//!     in order — close the gate; end every live stream `SERVICE_HELD`
//!     (`GracefulShutdown::end_streams_for_hold`); abort every store- or
//!     backup-writing task ([`HoldWriterAborts`], the inventory in
//!     `crate::dr::hold::HOLD_WRITERS`); set `live_edit_dr_held` and tick
//!     `live_edit_dr_hold_entered_total{reason}`; log one ERROR naming the
//!     incident, reason and evidence. Then a WARN every
//!     [`HOLD_HEARTBEAT_INTERVAL`] while held.
//!   - `Serving` after a hold entry: a random 0–5 s wait
//!     (`crate::dr::hold::unhold_exit_jitter`), then exit 0 through the
//!     [`HoldExit`] seam, restarting into normal service. The gate never
//!     reopens in place.
//!
//!   And on the latest verdict's age, on its own timer (SDD SR-6.5):
//!   - 30 s with no successful read on a serving replica that is not held:
//!     the stale-verdict entry routine ([`enter_stale_verdict`]) — the hold
//!     entry routine's steps with the cause `StaleVerdict` and the reason
//!     `UPSTREAM_UNAVAILABLE`, `live_edit_dr_hold_verdict_stale` set, one
//!     WARN, and no marker written — and readiness `NOT_SERVING` through the
//!     reactor's `verdict_stale` signal. The next successful read exits 0
//!     after the jitter. A held replica keeps its hold and only reports
//!     itself stale. Before the serving point, the replica exits
//!     [`NO_VERDICT_EXIT_CODE`] instead.
//!   - No successful read for any longer, held or serving: nothing more. The
//!     replica keeps reading, with no time limit, through the data pool,
//!     whose connections are rebuilt beneath it. A serving replica restarts
//!     once a read succeeds; a held one acts on that read as above.
//!
//!   A hold poll that ends before shutdown (its channel closes without the
//!   shutdown token) exits [`NO_VERDICT_EXIT_CODE`] at once, held mode
//!   included, and so does a hold reactor that fails: either way the
//!   replica can no longer act on the verdict, and its restart brings a new
//!   poll and reactor.
//!
//! Calls already past the gate when it closes are bounded by when their
//! callers give up. Once a call's caller gives up — its timer fires, the
//! Valkey client (fred) answers it with an error, or its future is dropped —
//! fred writes no frame of that call it has not already written, whether it
//! waits behind a reconnect, in the retry buffer or behind a redirect (SDD
//! §15.3 *Vendored fred* change 9). A deadline-bound call, an upload's commit
//! included, is dropped at the RPC deadline D (`LIVE_EDIT_RPC_DEADLINE_MS`)
//! after entry, and the drop is a give-up, so the size-scaled `txn_timeout`
//! of `write_object` (`CreateLayer`'s object write) and of the WATCH-route
//! `UpdateLayer` commit no longer reaches past D. A clean-up a drop leaves
//! (the inventory's audit lists them) is cut at the Valkey command timeout T
//! (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`) after its spawn, its `NOSCRIPT`
//! reload included, and the store out-of-memory request is behind the hold
//! write fence, so none is sent after entry. A deadline-exempt stream the
//! hold ending ends stops its own store work at entry — a join's
//! registration steps and `PeerMessage` publish and a subscription's
//! catch-up build race the stream's token — and runs its teardown as it
//! ends: a join's is a short sequence of writes, each cut at T.
//!
//! The on-access repairs and restores those calls run (the object-id counter
//! repair, the layer and session restores, and the external-value
//! fetch-on-miss rewrite) send nothing new after entry: their hold write
//! fence (`crate::checkpoint` `hold_fence`) is checked immediately before each
//! write they send, and every write they would send after entry is refused —
//! re-sends included (a transient retry, a `NOSCRIPT` reload) and a committed
//! restore's release enqueue included (skipped, leaking its references). The
//! fence does not stop a write already handed to fred at entry (one command,
//! one value of an external-value pre-store chunk, or one `MULTI … EXEC`):
//! fred writes it only while its caller still waits, at most T, and a cut
//! `MULTI` is discarded. So such a write is written within T of entry, or
//! never (the module doc of `hold_fence` has the exact bound).
//!
//! A frame fred had already written before its caller gave up is not
//! recalled: it reaches the socket within one of fred's bounded write or
//! flush waits, 45 s (`UNRESPONSIVE_MAX_TIMEOUT`, `crate::storage::client`),
//! or is dropped with its connection. The bounds need T > 0 (at 0 fred
//! bounds none of these writes), which `Config::from_env` enforces at
//! startup. So, apart from a join's teardown, nothing on this replica sends
//! a store write from hold entry + D + T + 45 s on — the window
//! [`hold_write_bound`](crate::dr::hold::hold_write_bound) computes, from the
//! constant itself; a join's teardown ends within 5T + 1 s + 45 s
//! (`crate::dr::hold::join_teardown_write_bound`). From the kernel's send
//! queue on, when the store runs a frame is the network's and the store's:
//! a recorded residual outside every bound (SDD §15.3 *Vendored fred* change
//! 9).
//!
//! The stale-verdict entry is bounded the same way: it closes the gate, ends
//! the streams and aborts the same inventory, so nothing on the replica
//! sends a store write from stale entry + D + T + 45 s on, apart from the
//! same exceptions.
//!
//! **Not a shutdown.** A hold or a stale verdict cancels no per-session token
//! and never touches the graceful-shutdown handle's `begun` or parent token
//! (so the scaling snapshot's `shutting_down` stays false). A hold tells the
//! health supervisor nothing: readiness and `""` stay SERVING while held
//! (SDD SR-5.4, P1). A stale verdict, held or not, turns readiness
//! NOT_SERVING through the reactor's `verdict_stale` signal: a replica that
//! has lost the store's counter shard should leave the load balancer, which a
//! fleet hold must not do. A SIGTERM while held or stale runs the normal
//! paced shutdown.
//!
//! **Shutting down is read from the shutdown token only.** The reactor and its
//! heartbeat stop when the token `main.rs` cancels on SIGTERM or SIGINT is
//! cancelled, and act on nothing after it; the poll stops on the same token,
//! so a paced shutdown that outlasts the stale threshold never reads as a
//! stale verdict. A closed verdict channel (the poll ended, or panicked,
//! without the token) is never read as a shutdown: the replica can no longer
//! see the verdict, so the reactor logs one ERROR and exits
//! [`NO_VERDICT_EXIT_CODE`] at once, before or after the serving point.

mod gate;
mod held_mode;
mod reaction;

/// Re-exported from `crate::dr::hold`, where the restore Job, which cannot
/// depend on `grpc`, reaches it too.
pub use crate::dr::hold::hold_write_bound;
pub use gate::{HoldGate, HoldGateFuture, HoldGateResponse, HoldGateService, HoldGateState};
pub use held_mode::{
    held_live_service, run_held_mode, HeldLiveService, HeldMode, HeldModeError,
    RefuseAllLiveService, HELD_MODE_SERVER_STOP_BUDGET,
};
pub use reaction::{
    enter_hold, enter_stale_verdict, exit_held_before_serving, spawn_held_mode_reactor,
    spawn_hold_reactor, HoldEntryParts, HoldExit, HoldReactorSettings, HoldServingPoint,
    HoldWriterAborts, ProcessExit, HOLD_HEARTBEAT_INTERVAL, NO_VERDICT_EXIT_CODE,
};
