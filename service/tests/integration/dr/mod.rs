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

//! Disaster recovery for a lost backing store (`live_edit::dr`): the fleet
//! hold, against a real store, and the server's reactions to it
//! (`live_edit::grpc::hold`).

/// SDD SR-10, PRD FR7 — the accept-loss Job end to end: its restore phase,
/// its lost set, its discards, the recount and the un-hold, with kills
/// converging.
mod accept_loss_test;
/// SDD SR-5.1–5.6, SR-6.4, SR-6.5, SR-7a — held-mode startup: the held
/// server in process over a real store, and its reactor on a paused clock.
mod held_mode_test;
/// SDD SR-6.1–6.4, SR-7 — a hold end to end: the marker, the real poll, the
/// reactor and the production server stack over TCP.
mod hold_entry_test;
/// SDD SR-6.2 — the hold write fence one check site at a time, through the
/// real repair and restore hooks.
mod hold_fence_sites_test;
/// SDD SR-6.2 — the hold write fence of the on-access repair and restore
/// paths: one test per path, per re-send mechanism, and the release-enqueue
/// skip.
mod hold_fence_test;
/// SDD SR-6.5, §22.21 — a dead read path comes back by itself: every store
/// connection cut and reconnects refused, then accepted again; stale with no
/// exit, then exit 0 once a read succeeds, with no client built by the hold.
mod hold_poll_reconnect_test;
mod hold_poll_test;
/// SDD SR-6.2 — nothing writes the store or the backup store over a 30 s
/// window from the hold's write bound on, with the busy writers of the
/// inventory (the checkpoint loop, the mirror workers, GC, the reconciler and
/// the presence sweeper) working before entry.
mod hold_quiescence_test;
/// SDD SR-6.1–6.4, SR-7 — the hold reactor, the entry routine, the writer
/// abort set and the gate's refusal, driven through a verdict channel the
/// test owns on a paused clock.
mod hold_reactor_test;
/// The checkpoint and external-value stores the hold tests park, slow and
/// count (`first_base_bootstrap_test.rs` uses them too).
pub(crate) mod hold_support;
/// SDD SR-6.2 — aborting an inventory writer's task handle alone stops its
/// in-flight work.
mod hold_writer_abort_test;
/// SDD SR-8.1, SR-9.1, SR-9.3, SR-10 — the DR Job frame in process: every
/// precondition exit, the adapter branch, the lease, the quiet period, the
/// absence rule and the body this build carries.
mod job_frame_test;
/// SDD SR-9.1 — the DR Job lease's refresh: a transient error retried, a
/// stalled refresh lost on time, a lease found gone reported at once.
mod job_lease_test;
/// SDD SR-9.8, SR-9.9, SR-14.6 — the restore Job's finish end to
/// end: the authoritative recount, the summary and the un-hold, one test
/// per Epic acceptance criterion, with kills converging.
mod restore_finish_test;
/// SDD SR-12 — the restore Job's memory: layers held at once within its
/// budget and its parallelism, and one at a time with no readable limit.
mod restore_memory_test;
/// SDD SR-9.2 to SR-9.6 steps 1–2 — the restore planner over a real store
/// and a scripted backup store: the planning rules end to end, the absence
/// rule at `T_safe`, the coverage check and its report, the session-listing
/// cross-check, and the quiet period (defer-39).
mod restore_plan_test;
/// SDD SR-9.4, SR-9.5, SR-9.10 — the restore writer's session writes: moves,
/// abandoned deletes, session restores and defer-51, with kills converging.
mod restore_sessions_test;
/// SDD SR-9.5, SR-9.6, SR-9.10 — the restore writer's object writes: a full
/// wipe, batches, the band, the backfill, flag clears, no releases, and a
/// kill at each step converging.
mod restore_write_test;
/// The scaffolding the hold tests share: verdict fixtures, the recording exit
/// seam, log and metric readers, readiness waits and keyspace snapshots.
pub(crate) mod support;
/// The scaffolding the restore writer's tests share: a scripted backup
/// store, backup and store fixtures, the Job run with the writer's seams,
/// and a restored object read back.
pub(crate) mod writer_support;
