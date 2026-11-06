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

//! The writer inventory: every background task that writes to the store or
//! to the backup store, and every on-access repair or restore path that does,
//! which a serving replica therefore stops when it enters a hold (SDD
//! SR-6.2). The tasks are aborted; the on-access paths, which have no task of
//! their own, are stopped by a hold write fence.
//!
//! This is data only. The abort routine lives with the server
//! (`crate::grpc::hold`), which registers each entry's tasks and tokens where
//! they are spawned and stops every entry at hold entry. A fleet hold and a
//! stale verdict (SDD SR-6.5) both stop exactly this set.
//!
//! # Adding a writer
//!
//! A new store- or backup-writing background task is one more [`HoldWriter`]
//! variant, with its name in [`HoldWriter::name`] and its place in
//! [`HOLD_WRITERS`]; its spawn site then registers its handle and token on the
//! server's abort set. Until it does, the server logs the entry as
//! unregistered when it reaches its serving point, and
//! `tests/misc/hold_wiring_scan.rs` fails: it requires a registration in
//! `main.rs` for every variant, and it classifies every task `main.rs` spawns
//! as a registered writer or a listed non-writer, so a new spawn there fails
//! until it is one or the other. A task that a library function or another
//! task spawns is outside that scan: the audit below is its only record.
//!
//! # The audit: every other background task
//!
//! Every task a serving replica spawns, background and per-call, and whether
//! it writes, so the inventory's completeness can be checked against the
//! code. "Writes" means a keyspace change on the store or a write to the
//! backup store.
//!
//! **Aborted at hold entry** (the variants of [`HoldWriter`]): the external-
//! value GC loop and its `le:gc:leader` refresh ticker; the orphan sweep; the
//! external-value `ref_count` reconciler; the checkpoint loop (with it the
//! `le:checkpoint:leader` refreshes); the session-checkpoint dispatch and the
//! base bootstraps (one tracker); the external-value mirror workers; the
//! external-value backup repair (its `le:gc:leader` lease, its
//! `ext_backup_confirm.lua` confirmations on the value's meta, and its
//! re-sent copies to the backup store); the checkpoint clean-up (its
//! `le:gc:leader` lease, its `{recon}` fence scripts and meta writes, its
//! backup-store deletes and its collect's heal re-puts); the layer
//! store's tracked cleanups (the `CreateLayer` commit finisher, the overwrite
//! cleanup bodies and the torn-commit record retract); the presence and
//! session tasks — the presence sweeper (and its `le:presence:sweeper:leader`
//! lease), the presence subscriber, the session-deleted relay, the
//! `session_tokens` sweeper, the base-path cache sweeper and the structural
//! cache's running audits, all on one token; and the on-access repairs and
//! restores — the `CreateLayer` object-id counter repair (a detached task),
//! the layer and session restores run inline by the calls that need them,
//! and the external-value fetch-on-miss rewrite (`ValkeyExternalValueStore`'s
//! `store_recovered_values`, which writes a value the backup store served
//! back into the store from inside a read, such as a deadline-exempt
//! `SubscribeReadLayer` catch-up) — which have no task to abort, so a hold
//! write fence, checked immediately
//! before each write they send and before every re-send (a transient retry,
//! a `NOSCRIPT` reload), refuses every write they have not sent yet.
//! The fence does not stop a write of theirs already handed to fred at entry
//! (one command, one value of an external-value pre-store chunk or of a
//! fetch-on-miss rewrite, or one `MULTI … EXEC`): fred writes it only while
//! its caller still waits, and once the caller gives up it writes no frame
//! of it not yet written and discards a cut `MULTI` (SDD §15.3 *Vendored
//! fred* change 9). So it is written within one Valkey command timeout of
//! entry, or never, which needs `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` > 0;
//! `Config::from_env` refuses 0 at startup. A frame fred had already
//! written is not recalled: it reaches the socket within fred's 45 s write
//! wait or is dropped with its connection, and from the socket on it is the
//! network's and the store's, a residual. A
//! committed layer materialisation's release enqueue that has not sent its
//! script by entry is skipped (`crate::checkpoint` `hold_fence` has the exact
//! bound).
//! The two subscribers, the cache sweeper and the
//! audits write nothing, but they share the token with the sweeper, and the
//! `session_tokens` sweeper, which writes nothing either, would otherwise
//! cancel per-session tokens on a store whose sessions are gone.
//!
//! **The exception: a backup-store write already under way.** Aborting a
//! task drops its future, which stops the store writes it has not sent, but
//! not a LocalDisk filesystem write already running on the blocking pool:
//! "a LocalDisk `put_ext_blob` runs in one blocking task that finishes its
//! write and rename even when its caller is dropped. So a hold that stops the
//! mirror workers or the backup repair does not stop an ext-blob write
//! already under way: it can land after hold entry. At most (mirror
//! workers + 1) such writes can be in flight" — one per mirror worker
//! (`LIVE_EDIT_CHECKPOINT_EXT_MIRROR_WORKERS`) and one for the backup repair,
//! which re-sends one value at a time. It is harmless: it writes "the same
//! bytes under the same content hash, which only adds to what the restore
//! Job's coverage check sees. StorageApi requests already sent have the same
//! property."
//!
//! The checkpoint loop is one more such writer: its chain-collapse re-base
//! (`checkpointer::rebase_on_chain_full`, the loop's only production caller
//! of `checkpoint::rebase`) backfills a re-based layer's external values to
//! the backup store through `put_ext_blob` one value at a time (in
//! `checkpoint::rebase`'s backfill, under `rebase_layer_lineage`), and on
//! LocalDisk that call is the same single blocking write-and-rename
//! (`LocalDiskCheckpointStore::put_ext_blob`). An
//! aborted checkpoint loop can therefore also land one blob write after
//! entry. It is equally harmless: the same bytes under the same content
//! hash. The checkpoint clean-up is the last: its collect's heal re-puts a
//! value through the same `put_ext_blob`, one value at a time. So at most
//! (mirror workers + 3) such blob writes can land after entry: one per
//! mirror worker, one for the backup repair, one for the checkpoint loop's
//! re-base and one for the clean-up's heal.
//!
//! **The checkpoint clean-up's pass, aborted mid-pass.** The pass runs one
//! step at a time, with no task of its own beyond the registered one, so
//! the abort drops it at its current step, and no step after it starts:
//! after entry it starts no `{recon}` script, meta write, lease probe,
//! backup-store delete or heal re-put. What can still land after entry is
//! the one step in flight at entry, bounded as follows. A store command it
//! had handed to fred is written within one Valkey command timeout of
//! entry, or never (fred change 9, as for the on-access writes above). A
//! LocalDisk call it had dispatched to the blocking pool completes after
//! entry: one `remove_file` (of one layer artifact, one temp file or one
//! external-value copy, a delete its `{recon}` script licensed before
//! entry) or the heal's one `put_ext_blob`, the blob write counted above.
//! A StorageApi request already sent is bounded as the other writers' are:
//! Q's term C covers it, and only a backend applying it later than C is the
//! residual (`crate::dr::job::quiet`). Cutting the pass between a delete and the `{recon}` step after
//! it is the cut a process death makes there, which the clean-up's work
//! lists already resume from (`checkpoint::cleanup`): the next pass, after
//! the hold clears, finishes it.
//!
//! The LocalDisk adapter's other writes — the layer and session checkpoint
//! puts and deletes and the counters sidecar — run each step of the shared
//! temp-then-rename helper (`atomic_write`), and each remove, as its own
//! `tokio::fs` call on the blocking pool. The one step already on the
//! blocking pool at entry finishes after it, and the aborted task sends no
//! next step. A late rename completes a publish whose earlier steps had
//! already finished, so publish order is preserved; a late remove is a
//! checkpoint-GC delete decided before entry (or a failed publish's rollback
//! or temp-file cleanup, also decided before entry). Each is bounded by the
//! completion of the one blocking call already dispatched; a filesystem
//! stalled longer than Q (the restore Job's quiet period) is outside the
//! bound and is a recorded residual, the same standing as `put_ext_blob`'s.
//!
//! **Library-internal spawns.** `tests/misc/hold_wiring_scan.rs`'s spawn
//! check covers spawns written in `main.rs` only; this audit is the record for
//! every other spawn. Each task an inventory entry's code spawns on its own,
//! and how the hold stops it:
//!
//! - GC: the loop (`spawn_gc_loop`) and its lease ticker
//!   (`spawn_gc_leader_refresh`) are `AbortOnDropHandle`s inside the handle
//!   `storage::external_values::spawn_gc_task` returns, the registered one.
//! - Session-checkpoint dispatch and base bootstraps, including
//!   `spawn_restore_rebase`: each tracker task selects on the registered
//!   abort token first (`SessionCheckpointWriter::enqueue`,
//!   `BaseBootstrap::spawn`).
//! - The counter repair's detached task
//!   (`CheckpointObjectIdCounterRepair::join_or_spawn_attempt`): it carries
//!   the on-access fence.
//! - The join lease keepers (`grpc::presence`'s `spawn_lease_keeper`): they
//!   stop on the enrolled shutdown child that `end_streams_for_hold` cancels.
//! - The structural-cache audits: reads only.
//! - The backup repair, the mirror workers, the orphan sweep, the reconciler,
//!   the checkpoint loop and the checkpoint clean-up: no further non-test
//!   spawns.
//! - The drop-spawned clean-ups (`ReadHold`'s `HDEL`, `OomUploadHold`'s
//!   `DEL`, the fence-guard cleanups — a writer intent's `HDEL` and a fence
//!   lease's release, the release cut at one command timeout, its `NOSCRIPT`
//!   reload included — and `IntentRenewal`'s tail, its one `HDEL` cut the
//!   same way) and the `store_oom` request, which the hold write fence
//!   stops: in the bounded list below.
//!
//! The only late writes among them are the LocalDisk calls already
//! dispatched, above: the blob write of the checkpoint loop's re-base
//! backfill, and the checkpoint clean-up's one remove or heal blob write.
//!
//! **Not writing, so left running:**
//!
//! - the hold poll: one `EVALSHA` that reads the marker;
//! - the hold reactor, its heartbeat and, in normal mode, the watcher that
//!   exits the process if the reactor fails (`crate::grpc::hold`): they act
//!   on the gate, the streams, the abort set and the two signals they
//!   publish (readiness while the verdict is stale, and the scaling
//!   snapshot's `held`), and log;
//! - the health supervisor: `PING`s; readiness stays as it is while held
//!   (SDD SR-5.4) and turns `NOT_SERVING` while the verdict is stale;
//! - the store-memory re-read behind the external-value release budget and
//!   `live_edit_store_memory_used_ratio`: `INFO` and `CLUSTER SLOTS` reads
//!   only;
//! - the service-token providers: they read token files;
//! - the metrics server and the gRPC servers: they serve, and the LiveService
//!   gate refuses every call;
//! - the startup signal watcher (`observability::startup::StartupClaim::watch`):
//!   on the first `SIGTERM` or `SIGINT` it cancels the process shutdown
//!   request and, until the serving path claims startup, sets readiness and `""`
//!   `NOT_SERVING`, stops the bootstrap and metrics servers, flushes and
//!   exits; it sends no store command;
//! - the Valkey client's own tasks: the reconnect and error hooks (metrics
//!   only), the `txn_pool` slot repairs and member replacements (connection
//!   state: `UNWATCH`, routing syncs, reconnects), the router supervisor, the
//!   stream multiplexer's `XREAD BLOCK` readers and the snapshot fan-out
//!   threads (reads), and the `WatchGuard` `UNWATCH` cleanups (connection
//!   state, no keyspace change);
//! - the external-value release gate's poll: reads;
//! - the per-call tasks that write nothing: an `UpdateLayer`'s start-record
//!   read (`LayerStore::spawn_start_record_read`, one `HGET`), the restore
//!   compositions on the blocking pool (`compose_from_artifacts`, from
//!   `checkpoint::recovery` and `checkpoint::rebase`; CPU only, no store
//!   call), and graceful shutdown's paced stream ending
//!   (`grpc::transport::end_streams_after_drain`).
//!
//! **Writing, but bounded and not in the inventory:** each is owned by a call
//! already past the gate or by a task that is ending, does one command or a
//! short sequence, and is bounded as its entry says. None is a recurring
//! writer, and none has an abort path of its own; aborting the cleanups among
//! them would only leave a lease, an intent or a hold in place until its TTL.
//!
//! - the snapshot-fence cleanups on the client's fence-guard tracker
//!   (`ValkeyClient::spawn_fence_guard_cleanup`): a dropped fenced read's
//!   lease release, or a dropped writer intent's `HDEL`, spawned when its
//!   owner ends. An aborted checkpoint loop or GC pass spawns them as it is
//!   dropped, and leaving them would hold other replicas' writers of that
//!   layer off until the lease lapses — wrong for a stale-verdict abort, where
//!   only this replica stops. The intent's `HDEL` is one command; the lease's
//!   release is cut at one command timeout, its `NOSCRIPT` reload included
//!   (`storage::snapshot_fence`, # Guards);
//! - the `UpdateLayer` attempts' writer-intent renewals (`IntentRenewal`, on
//!   the same tracker): each belongs to an attempt in a call already past the
//!   gate and ends with that call (bounded by `LIVE_EDIT_RPC_DEADLINE_MS`):
//!   the stop interrupts a renew in flight, and the tail's one `HDEL` is cut
//!   at one command timeout, with no second `HDEL` after a failed one;
//! - a dropped read hold's `HDEL` and a dropped out-of-memory upload hold's
//!   `DEL` (`storage::external_values`): one command each, spawned when its
//!   owner ends;
//! - the store out-of-memory signal's collection request
//!   (`storage::store_oom`), spawned whenever a write is refused `-OOM`:
//!   behind the hold write fence, so one in flight at entry gives up within
//!   one command timeout, with no `NOSCRIPT` reload, and none is sent after
//!   entry;
//! - the per-call streaming tasks: each `JoinSession` task and its lease
//!   keeper, and each `SubscribeReadLayer` task. The hold ending ends them
//!   (SDD SR-6.3), and their own store work ends at that ending: a join's
//!   registration steps and `PeerMessage` publish, and a subscription's
//!   catch-up build, race the stream's token. A join's teardown then runs
//!   its unregister (cut at one command timeout, its reload included) and
//!   index writes, each given up within one command timeout
//!   (`dr::hold::join_teardown_write_bound`).

/// A store- or backup-writing background task, or group of tasks sharing one
/// token or tracker, that hold entry aborts — or, for
/// [`HoldWriter::OnAccessRepairs`], a group of on-access write paths with no
/// task of their own, which hold entry stops by closing their hold write fence
/// (SDD SR-6.2). See the module docs for what each covers and for every task
/// that is left running.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HoldWriter {
    /// The external-value GC loop and its `le:gc:leader` refresh ticker
    /// (`storage::external_values::spawn_gc_task`). The lease lapses within
    /// its TTL of the ticker's last refresh reaching the store.
    ExtGc,
    /// The orphan sweep (`storage::orphan_sweep::spawn_orphan_sweep`), on
    /// `le:gc:leader`.
    OrphanSweep,
    /// The external-value `ref_count` reconciler
    /// (`storage::ext_reconcile::spawn_ext_ref_reconcile`), on
    /// `le:gc:leader`. Not spawned when its interval is 0.
    ExtRefReconcile,
    /// The leader-elected checkpoint loop (`checkpoint::spawn_checkpoint_loop`),
    /// which also refreshes `le:checkpoint:leader`. Not spawned when
    /// checkpointing is off.
    CheckpointLoop,
    /// The session-checkpoint dispatch (`checkpoint::SessionCheckpointWriter`)
    /// and the base bootstraps (`checkpoint::BaseBootstrap`), on their shared
    /// tracker's abort token.
    SessionCheckpoints,
    /// The external-value mirror workers (`checkpoint::spawn_ext_mirror_workers`),
    /// which `PUT` values to the backup store. Not spawned without a backup
    /// store or with checkpointing off.
    ExtMirrorWorkers,
    /// The external-value backup repair (`checkpoint::spawn_ext_backup_repair`),
    /// on `le:gc:leader` every 5 min: it confirms backed-up values on their
    /// meta (`ext_backup_confirm.lua`) and re-sends the copies the backup
    /// store lacks. Spawned with the mirror workers, under the same gate.
    ExtBackupRepair,
    /// The checkpoint clean-up (`checkpoint::spawn_checkpoint_cleanup`), on
    /// `le:gc:leader` every `LIVE_EDIT_CHECKPOINT_RECONCILE_INTERVAL_SECS`: it
    /// takes the lease, runs the `{recon}` fence scripts and the meta writes,
    /// deletes orphaned partial publishes, dead layers' generations and
    /// collected external-value copies from the backup store, and re-puts a
    /// value its collect heals. Spawned with the mirror workers, under the
    /// same gate. Aborted mid-pass, it sends no further store command and
    /// starts no further backup-store call; its last write lands as the
    /// module docs' bounds state: a store command already handed to fred
    /// within one command timeout of entry or never, and one LocalDisk call
    /// already on the blocking pool (one file remove, or the heal's one
    /// `put_ext_blob`) when that call completes.
    CheckpointCleanup,
    /// The layer store's tracked cleanups (`storage::layer::LayerStore`'s
    /// cleanup tracker): the `CreateLayer` commit finisher, the overwrite
    /// cleanup bodies and the torn-commit record retract, on the tracker's
    /// abort token.
    LayerCleanups,
    /// The presence and session tasks on one token: the presence sweeper (and
    /// its `le:presence:sweeper:leader` lease), the presence subscriber, the
    /// session-deleted relay, the `session_tokens` sweeper, the base-path
    /// cache sweeper and the structural cache's running audits.
    PresenceAndSessionTasks,
    /// The on-access repairs and restores, behind one hold write fence
    /// (`checkpoint::hold_fence`): the `CreateLayer` object-id counter repair
    /// (`checkpoint::CheckpointObjectIdCounterRepair`), the layer restore
    /// (`checkpoint::RecoveryHook`), the session restore
    /// (`checkpoint::SessionRecoveryHook`) and the external-value fetch-on-miss
    /// rewrite (`storage::external_values::ValkeyExternalValueStore`'s
    /// recovery fence). They run inside calls already past
    /// the gate, or on a detached task such a call started, so there is no
    /// task to abort: the fence, a token, is checked immediately before each
    /// write they send and each re-send of one, so after entry none sends a
    /// write it has not sent yet
    /// and no new counter repair is spawned. A write already sent runs to its
    /// end (`checkpoint::hold_fence` has the bound).
    OnAccessRepairs,
}

/// The writer inventory: every [`HoldWriter`], in the order the abort routine
/// stops them (aborting each task entry, closing the on-access fence). One
/// place, so a hold and a stale verdict stop the same set.
pub const HOLD_WRITERS: [HoldWriter; 11] = [
    HoldWriter::ExtGc,
    HoldWriter::OrphanSweep,
    HoldWriter::ExtRefReconcile,
    HoldWriter::CheckpointLoop,
    HoldWriter::SessionCheckpoints,
    HoldWriter::ExtMirrorWorkers,
    HoldWriter::ExtBackupRepair,
    HoldWriter::CheckpointCleanup,
    HoldWriter::LayerCleanups,
    HoldWriter::PresenceAndSessionTasks,
    HoldWriter::OnAccessRepairs,
];

impl HoldWriter {
    /// A stable lowercase name for logs.
    pub fn name(self) -> &'static str {
        match self {
            Self::ExtGc => "ext_gc",
            Self::OrphanSweep => "orphan_sweep",
            Self::ExtRefReconcile => "ext_ref_reconcile",
            Self::CheckpointLoop => "checkpoint_loop",
            Self::SessionCheckpoints => "session_checkpoints",
            Self::ExtMirrorWorkers => "ext_mirror_workers",
            Self::ExtBackupRepair => "ext_backup_repair",
            Self::CheckpointCleanup => "checkpoint_cleanup",
            Self::LayerCleanups => "layer_cleanups",
            Self::PresenceAndSessionTasks => "presence_and_session_tasks",
            Self::OnAccessRepairs => "on_access_repairs",
        }
    }
}
