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

// External value store: deduplication by content hash (`ext_content_hash`),
// reference counting, GC.
//
// Implements SDD §11: write path (§11.1), read path (§11.2), garbage
// collection (§11.3), and the GC loop's per-pass handling of stale in-flight
// claims (§11.4).
//
// External value operations run OUTSIDE the object-state WATCH/MULTI/EXEC —
// external keys share a Cluster slot with each other (per content hash) but
// NOT with object keys (external values are shared across objects by design).
// Within a single content hash, `ext_meta:{hash}` and `ext_data:{hash}` share
// a slot via the `{hash_hex}` hash tag, so one single-slot Lua script can
// atomically create both and another can atomically delete both.
//
// Crash-safety protocol:
//
// * Content gate (`verify_external_values`): before the writer below is
//   reached on the live CreateLayer / UpdateLayer path,
//   `prepare_external_values` (the check half of `store_external_values`)
//   checks, once per upload, that every external-value reference's raw
//   claim is exactly 32 bytes, that the OMPF part it names is `ext_value_size`
//   bytes long, and that the part's content hash (`ext_content_hash`: SHA-256
//   over the SHA-256 digests of its 1 MiB blocks) equals the claim. Under the
//   default `LIVE_EDIT_EXT_VALUE_VERIFY=enforce`, one bad reference refuses the
//   whole upload `INVALID_ARGUMENT` before the first store, so nothing is
//   written, incremented, mirrored or committed for it; under `observe` a size
//   or hash mismatch is counted and logged and the upload is stored anyway.
//
// * Writer (`store_values`): for each hash, issue one `EVALSHA` of
//   `scripts/ext_store_value.lua` — `SET ext_data{hash} NX` +
//   `HINCRBY ext_meta{hash} ref_count 1` + `HSET ext_meta{hash} last_store_ms
//   <Valkey clock>` — on the multiplexed data pool. Both keys live in the same
//   slot and Valkey runs Lua atomically under its single server thread, so a
//   crash before the script runs leaves neither key written and after it both
//   are written consistently. b.rej replaced a per-hash `MULTI/EXEC` on the
//   exclusive `txn_pool` with this: the atomicity is the same, the b.7td
//   checkout hygiene probe and the txn-pool contention are gone, and the
//   increment gained the server-side `last_store_ms` stamp that arms the
//   reference-count reconciler (see the script header).
//   The DATA half is set-if-absent: an existing key (of any type) is not
//   overwritten, so it is idempotent under retry and a later unverified store
//   of the same hash cannot replace it. The one exception is the verified
//   store (`store_verified_values`), which the live path uses in `enforce`
//   mode after the gate accepted every part: it replaces an existing data key
//   that is not a string or whose length differs from the verified bytes
//   (no byte compare), so a damaged copy is healed by a live re-upload. The
//   writer itself verifies nothing: the live path runs the gate above first
//   (and in `observe` mode stores what fails it, never healing), and the
//   restore and recovery callers carry bytes a gated upload stored (or that
//   predate the gate) and never heal. The
//   `HINCRBY` half is not idempotent: it is a
//   PRE-FLIGHT increment, paid before the object commit that would justify it,
//   so a retry — or a
//   commit that never lands (WATCH conflict, or any other refusal after the
//   pre-store) — leaves `ref_count` too high by one. The GC sweep cannot see
//   that on its own: it only performs decrements someone owed it, so an
//   increment nobody owes a decrement for is invisible to it. The direction is
//   always leak, never over-release. Such a surplus is reclaimed at the
//   retirement of an object that does end up referencing the hash; where no
//   reference ever lands, the reference-count reconciler
//   (`storage::ext_reconcile`, b.yda) drains it — a leader-gated periodic pass
//   that compares each `ext:meta`'s `ref_count` against the live objects that
//   actually reference the hash and hands back a surplus that survives two
//   observations a grace apart. Same residual class as the crash-window
//   leftovers in §11.3 — see SDD §11.1 and §11.3.
//
// * Release ledger (b.yuo): beside the unchanged pending Set, three `{gc}`
//   companion keys account every owed release — `pending:age` (first-enqueue
//   time), `pending:bytes` (the blob bytes the member accounts for) and
//   `pending:total` (their sum over pending ∪ inflight). Every enqueue goes
//   through one script, `ext_release_enqueue.lua`, which adds the member and
//   its ledger entries atomically — or, for a replay of a member already
//   claimed, settled within the replay window or still accounted, writes
//   nothing (b.1jt); the bytes are the blob's `STRLEN`, read just
//   before (0 for a commit's `surplus_acquire` release). The total is what the
//   release budget `H` (`LIVE_EDIT_EXT_RELEASE_MAX_BYTES`) bounds: uploads
//   carrying external values wait at `H`, and are refused `UNAVAILABLE`
//   "nothing applied" if it stays full (`storage::ext_release_gate`).
//
// * GC (`process_pending`, one pass of a continuous, age-gated loop): one
//   read-only `gc_select.lua` returns the ledger's levels and up to a batch of
//   members, oldest first, no younger than the policy's age floor — while the
//   total is at or above `H / 2`, also younger ones down to the pressure
//   floor, each added only while the bytes of every member selected so far,
//   those past the age floor included, do not exceed the total's excess over
//   `H / 2` — and never one enqueued after the earliest live read hold began (a
//   catch-up whose build found a blob missing takes a per-holder hold,
//   `ReadHold`, re-reads under it and releases it when its rebuild ends).
//   Only a member with an age is ever selected: an un-aged one (an older
//   build's bare `SADD`) is sampled every pass and stamped with an age
//   (`gc_stamp_unaged.lua`), and collected like any other from then on.
//   Each member is then claimed —
//   moved from `ext:pending` into the `ext:inflight` sorted set, scored with
//   the Valkey-clock claim time plus `GC_CLAIM_SCORE_OFFSET_MS`, by
//   `gc_claim_pending.lua` (atomic; a 0 reply means another worker owns
//   it) — decremented by
//   `gc_decrement_and_collect.lua` (one atomic `EVALSHA` that decrements
//   `ref_count` and deletes `meta` and `data` at `<= 0`, b.kf1 C-5), and
//   completed by `gc_complete.lua`, which removes it from `inflight`, takes its
//   bytes out of the total and settles it. At most `GC_MEMBER_CONCURRENCY`
//   members are in flight.
//
// * At most one decrement per release member (b.yuo): the decrement is NOT
//   safe to re-run — fred re-sends a command whose reply was lost, and a
//   member left in `inflight` is re-claimed by a later pass — and a second
//   decrement of a still-referenced hash deletes a blob a layer still points
//   at. So the decrement script records every applied member on the hash's own
//   slot (`le:ext:applied:{hash_hex}`) and answers a re-run `replayed`, and
//   refuses (`stale`, no write) a run executing more than the fence after its
//   claim. The record is torn down through the `{gc}` settle carrier
//   (`le:ext:{gc}:settled`) once the member has been settled for the replay
//   retention, which is longer than the fence plus clock skew, so no copy of
//   any claim can act after its record is gone. Leak, never over-release.
//   The leader lock only sheds load among builds of this protocol: every step
//   above is safe with two such sweepers, because anything unsafe under two
//   leaders must be safe by construction (a 30 s lease cannot exclude a sweep
//   whose holder's refreshes stopped reaching the store: a peer takes the
//   lock over while that sweep may still run).
//
// * Older builds sharing the store (b.1jt). Every build before the GC
//   protocol marker (0.1.0 through 0.2.0-rc.48) writes an unmarked lock
//   value, scores its claims with the bare claim time, drains claims scored
//   in `[0, now − 30 s]` back to `pending`, and has its sweep and reconciler
//   read retirement carriers as SETs. 0.1.0 through 0.2.0-rc.47 also run a
//   record-blind GC: their decrement has no replay record (0.2.0-rc.48 has
//   one). This build keeps every obligation it may have acted on out of that
//   code's reach, and never finishes one an older build started:
//   - the GC lock `le:gc:leader` carries a protocol-marked value; this build
//     takes an unmarked (or lower-version) lock over, and every process of
//     this build refreshes it once a second from a ticker independent of its
//     passes, renewing only while its GC loop is running and making progress
//     (`spawn_gc_task`, `GC_PROGRESS_STALL_AFTER`), so the key never lapses
//     while any of them with a running, progressing loop can reach the store,
//     and no older build starts a new pass meanwhile
//     (`try_acquire_gc_leader`). The lock is not what makes this safe: an
//     older pass already running when the key was taken over runs on until it
//     ends, and if every process of this build loses the store for the 30 s
//     TTL (or every one's loop dies or stalls) the key lapses and an older
//     build can run a pass. The progress gate is in-process only — the key and
//     value are unchanged — so it adds no mixed-window behaviour beyond when a
//     lapse can happen, which this paragraph covers. Safety there rests
//     on the points below, which hold without the lock (`pending` holds only
//     never-attempted obligations, the enqueue refuses a replay, carriers are
//     HASHes), so such a pass's exposure is that it ignores this build's read
//     holds (a catch-up can fail `INTERNAL`), never a double decrement. The
//     same holds one version up: a newer marked leader's key lapses to this
//     build the same way, so a future protocol bump needs the same
//     store-state analysis, not just the lock;
//   - its claims are scored claim ms + 10^13, outside every older drain's
//     range, and never go back to `pending`: a stale one is re-claimed in
//     place (`gc_reclaim_stale.lua`). So `pending` only ever holds obligations
//     no decrement was attempted for, which an older GC that inherits it
//     decrements correctly (invariant I1);
//   - a claim scored below the offset is an older build's: never decremented
//     here, dropped once older than the fence (`gc_drop_foreign_claims.lua`);
//   - the enqueue refuses a replay of a member already claimed, settled or
//     accounted, so a fred re-send cannot hand a collected member to an older
//     sweep still running;
//   - retirement carriers are HASHes, which an older carrier claim cannot
//     read; an older build's SET carrier is never paid
//     (`storage::orphan_sweep`).
//   Each action on an older build's state ticks
//   `live_edit_gc_older_build_total{event}`. What is left is leak-direction
//   (an older claim or carrier dropped, a rollback's stranded claims) or
//   the older builds' own defects; SDD §11.3 states the residuals.
//
// * Counting unit (b.qpf): `ref_count` counts LIVE OBJECTS referencing the
//   hash — one increment per object per distinct hash, not one per referencing
//   value row. Every transition of an object's reference set (a rewrite that
//   supersedes a hash, a delete of the last row holding it, the object's own
//   retirement) emits exactly one release event, and every release event
//   decrements exactly once. The per-object bookkeeping that makes a backend
//   able to decide "was that the object's last row holding this hash?" locally
//   is `le:{object_id}:obj:ext` (`KeySchema::obj_ext_index`), maintained in the
//   same atomic unit as the value rows.
//
// * Release channel identity (b.qpf): a counted decrement cannot ride a lossy
//   channel, and `SADD` of a bare `hash_hex` is lossy — two genuinely-distinct
//   owed decrements collapsed into one member and the second reference leaked
//   forever. Members are now `{hash_hex}:{object_id}:{event}` (`ExtRelease`),
//   so distinct events stay distinct while a REPLAY of one event still
//   collapses, which is what `schedule_old_object_ext_gc`'s `ext_gc_scheduled`
//   guard and `delete_layer`'s retry depend on. Legacy bare-`hash_hex` members
//   parse to themselves, so a rolling deploy needs no migration.
//
// * Stale claims (§11.4): every pass, the selection reports this build's own
//   `ext:inflight` claims older than the re-claim age D (the `le:gc:leader`
//   TTL in production) — a crash victim, or a member a transient error or a
//   `stale` decrement left behind, whose decrement is uncertain. The pass
//   re-claims each in place, only while its score is unchanged
//   (`gc_reclaim_stale.lua`, so of two concurrent re-claims one wins), and
//   decrements and completes it in the same pass; it never returns to
//   `ext:pending`. A fresher claim belongs to a peer's running sweep and is
//   left alone. The retry is safe because the decrement script answers a
//   member whose decrement already applied `replayed`, and refuses a late copy
//   of the old claim `stale` (the fence is below D).
//
// * The out-of-memory collection (b.pa1): under the required `noeviction`
//   policy a full shard refuses every write that can grow memory. When
//   released blobs filled it, the GC frees them at once:
//   - Trigger: any store write refused `-OOM`, anywhere in the process, is
//     noted on the client's `StoreOomSignal` (`storage::store_oom`), which,
//     at most once a second, sets the `{gc}` request key
//     `le:ext:{gc}:oom:request` (`PX` 10 s) and then wakes this replica's GC
//     loop. The wake shares the request's rate limit: the GC's own refused
//     steps note too, and a wake per note would keep the loop from sleeping
//     while the shard stays full.
//   - The leader takes the request with a `GETDEL` at the start of every pass
//     and starts a collection when the last one ended at least 1 s ago and
//     freed something or ended `failed` (a pass error), or at least 30 s ago
//     (`OOM_COLLECTION_RETRY_AFTER_*`).
//     The collection sets the upload hold `le:ext:{gc}:oom:upload_hold`
//     (`PX` 15 s; only with a release budget), runs passes under
//     `GcPolicy::OOM_COLLECTION` — every released member, at any age, still
//     clipped by live read holds — while a pass reports more work, for at most
//     `OOM_COLLECTION_MAX_DURATION` (10 s), then deletes the hold on every
//     exit (a drop guard when cancelled; the TTL when the leader dies).
//   - Every replica's release-budget gate holds uploads carrying external
//     values while the hold exists, within the wait the budget already has;
//     the hold never refuses an upload on its own (`storage::ext_release_gate`).
//   - The scripts that must work on a full shard — the decrement, the read
//     hold's take and renew, the leader acquisition and the signal keys —
//     open with a write that cannot grow memory, after which Valkey lets every
//     later command of the script run (`gc_decrement_and_collect.lua`, "The
//     out-of-memory opener"). A pass also skips, rather than ends at, a `{gc}`
//     stamp or re-claim refused `-OOM`. Still refused, by design: on a full
//     `{gc}` shard the release enqueue (a refused release falls to the
//     reference-count reconciler), the stamp and the re-claim; on any full
//     shard the external-value store and every request write.
//   - It frees only owed releases and never reads or moves the release
//     budget. Observable as `live_edit_gc_oom_collections_total{outcome}` and
//     `live_edit_gc_oom_freed_bytes_total`, with one WARN per collection
//     (rate-limited; it also carries an upload hold that could not be set)
//     and one WARN per process the first time the store
//     refuses a collection's own scripts (`outcome="refused"`).

use std::sync::Arc;
use std::sync::LazyLock;

use async_trait::async_trait;
use fred::interfaces::{HashesInterface, KeysInterface, SortedSetsInterface};
use fred::types::scripts::Script;
use oodf_ffi::OwnedDelta;

use crate::checkpoint::{CheckpointError, CheckpointStore};
use crate::config::ExtValueVerifyMode;
use crate::domain::delta::DeltaError;
use crate::domain::types::{FieldValue, ObjectState};
use crate::error::{ExternalValueError, ServiceError};
use crate::observability::metrics::{
    SharedMetrics, EXT_VALUE_VERIFY_HASH_MISMATCH, EXT_VALUE_VERIFY_OK,
    EXT_VALUE_VERIFY_SIZE_MISMATCH, GC_DECREMENT_OUTCOME_APPLIED, GC_DECREMENT_OUTCOME_REPLAYED,
    GC_DECREMENT_OUTCOME_STALE, GC_OLDER_BUILD_EVENT_CLAIM_DROPPED,
    GC_OLDER_BUILD_EVENT_LOCK_TAKEN_OVER, GC_OLDER_BUILD_EVENT_RELEASE_STAMPED,
    GC_OOM_COLLECTION_OUTCOME_FAILED, GC_OOM_COLLECTION_OUTCOME_FREED,
    GC_OOM_COLLECTION_OUTCOME_FREED_NOTHING, GC_OOM_COLLECTION_OUTCOME_NOTHING_TO_COLLECT,
    GC_OOM_COLLECTION_OUTCOME_REFUSED, STORE_WRITE_REFUSED_STAGE_EXT_STORE,
};
use crate::storage::client::{
    map_fred_err_for_grpc, map_transient_fred_err, refuse_store_write, store_write_refusal,
    FredErrOrigin, StoreWriteRefusal, ValkeyClient, POOL_LABEL_DATA, POOL_LABEL_MAINTENANCE,
};
use crate::storage::ext_recovery_singleflight::ExtRecoverySingleflight;
use crate::storage::ext_release_gate::ExtReleaseGate;
use crate::storage::keys::{sha256_to_hex, KeySchema};
use crate::storage::leader_lease::LeaderIdentity;
use crate::storage::log_gate::LogGate;
use crate::storage::retirement::RELEASE_SIZE_READS_IN_FLIGHT;

/// Maximum number of concurrent per-hash blob stores in flight inside
/// `ValkeyExternalValueStore::store_values` (b.avd HIGH follow-up; rationale
/// re-argued for the data pool by b.rej).
///
/// **What the bound protects, post-b.rej.** Each in-flight store is one
/// `EVALSHA` of `ext_store_value.lua` on the MULTIPLEXED data pool — it no
/// longer holds an exclusive `txn_pool` slot, so the original starvation
/// argument (N concurrent `acquire_txn` waiters pinning every slot of the pool
/// and queueing `UpdateLayer` slow-path commits and `CreateSession` behind
/// them) no longer applies. The bound is not vestigial, because multiplexing is
/// not free for THIS command: a multi-MB blob occupies the shared socket it
/// was routed onto for the whole of its transfer, and every read and every Lua
/// fast-path commit that fred assigned to that same connection waits behind it.
/// A wave of `STORE_VALUES_MAX_INFLIGHT` concurrent blobs therefore
/// head-of-line-blocks up to that many of the pool's connections, and the
/// invariant that keeps the effect bounded rather than unbounded is that the
/// pool is AT LEAST AS WIDE AS the wave — so a wave can never exceed the pool,
/// starve the txn/subscriber tokens, or wedge fred's queue with blobs that have
/// no connection to land on. That is a floor, not a comfort margin: at exactly
/// the floor a full wave can momentarily occupy EVERY data-pool connection, and
/// concurrent reads and Lua fast-path commits queue behind blob transfers until
/// it drains.
///
/// `8` is the chosen bound: well under the default 32-slot data pool (~75% of
/// connections left clear for concurrent readers and committers), generous
/// enough that a typical 5-10-hash layer stores in one or two waves, and
/// constant rather than configurable because the value's only job is to stay
/// much smaller than `LIVE_EDIT_VALKEY_POOL_SIZE` — re-tuning it would not
/// change steady-state throughput, only the worst-case blocking profile.
///
/// Exposed `pub(crate)` so the storage-module unit tests can pin the
/// contract and `Config::validate_pool_size_vs_store_values_inflight`
/// (b.avd second follow-up) can reject a `LIVE_EDIT_VALKEY_POOL_SIZE`
/// configured below this floor — below it, a wave is wider than the pool it
/// has to fit into. Equality is accepted, and deliberately so, but it is the
/// worst accepted case rather than a healthy one, which is why production
/// (Helm) sets `LIVE_EDIT_VALKEY_POOL_SIZE=64` and the guidance is "well above
/// the floor". Not exposed in the public crate API because no out-of-crate
/// caller has any business depending on the bound; tests outside the crate
/// read it as `storage::test_support::STORE_VALUES_MAX_INFLIGHT` (under the
/// `test-support` feature).
pub(crate) const STORE_VALUES_MAX_INFLIGHT: usize = 8;

/// Compiled Lua script that applies one release member's decrement at most
/// once: on `ext_meta:{hash_hex}`, `ext_data:{hash_hex}` and the replay record
/// `ext:applied:{hash_hex}` (one `{hash_hex}` slot), it answers `replayed` for
/// a member already recorded, `stale` for a claim older than the fence, and
/// otherwise decrements `ref_count`, records the member, and deletes `meta` and
/// `data` at `<= 0`, answering the data's length as the bytes freed (b.kf1
/// C-5; b.yuo; b.pa1). Each writing branch opens with a write that cannot grow
/// memory, so it runs on a store over `maxmemory` too. See
/// `scripts/gc_decrement_and_collect.lua` for the contract, why a re-run is
/// not harmless, the clock assumption the fence makes, and the out-of-memory
/// opener. Sent with fred's default write attempts (`gc_member`): a re-send
/// answers `replayed`.
///
/// `Script::from_lua` is infallible — it just hashes the source. The first
/// `evalsha_with_reload` call sends `SCRIPT LOAD` to the server on
/// `NOSCRIPT` before retrying.
pub static GC_DECREMENT_AND_COLLECT_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/gc_decrement_and_collect.lua")));

/// Compiled Lua script that enqueues owed release members onto
/// `le:ext:{gc}:pending` together with their release-ledger entries (age,
/// bytes, total) in one atomic `{gc}`-slot step (b.yuo), refusing — writing
/// nothing for — a member that is not in `pending` but is still in
/// `inflight`, in `settled`, or holds a ledger field: a replay of an enqueue
/// that already landed (b.1jt). The single production
/// enqueue script, behind both the plain enqueue ([`enqueue_ext_releases`]) and
/// the retirement intent's attempt mode ([`enqueue_ext_release_attempt`]) —
/// see `scripts/ext_release_enqueue.lua`.
pub static EXT_RELEASE_ENQUEUE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/ext_release_enqueue.lua")));

/// Compiled Lua script that releases a GC claim after its decrement was
/// decided and settles the member in the release ledger: `ZREM inflight`, and
/// only when that removed it and the member is not back in `pending`, take its
/// bytes out of the total and `ZADD` it to `le:ext:{gc}:settled` (b.yuo).
/// Replaces the bare `ZREM inflight`. See `scripts/gc_complete.lua`.
pub static GC_COMPLETE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/gc_complete.lua")));

/// Compiled read-only Lua script that opens every GC loop pass: the release
/// ledger's levels, the oldest age, the settled replay records due
/// for teardown, the members to claim, oldest first, under the policy's
/// age floors and the read hold (b.yuo), a sample of un-aged members, and the
/// stale claims — this build's to re-claim, an older build's to drop
/// (b.1jt). See `scripts/gc_select.lua`.
pub static GC_SELECT_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/gc_select.lua")));

/// Compiled Lua script that tears down one settled member's decrement replay
/// record on its hash's own slot, only while the record is no newer than the
/// settle (b.yuo). See `scripts/gc_unmark.lua`.
pub static GC_UNMARK_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/gc_unmark.lua")));

/// Compiled Lua script that retires `le:ext:{gc}:settled` entries whose replay
/// records were torn down, each only while its score is still the one read
/// (b.yuo). See `scripts/gc_settled_retire.lua`.
pub static GC_SETTLED_RETIRE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/gc_settled_retire.lua")));

/// Compiled Lua script that runs one `HSCAN` page of the release-ledger audit,
/// settling every ledger entry whose member left both `pending` and `inflight`
/// without being settled (b.yuo). Its seventh key is the `{gc}` migration
/// sentinel ([`KeySchema::migration_sentinel`], b.cyu), which keeps a page
/// from being cut short mid-script while the `{gc}` slot migrates. See
/// `scripts/gc_ledger_audit.lua`.
pub static GC_LEDGER_AUDIT_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/gc_ledger_audit.lua")));

/// Compiled Lua script that takes, or renews, one holder's field of the
/// `{gc}` read hold (`le:ext:{gc}:read_hold`, a HASH) a catch-up holds while it
/// re-reads after an external-value miss (b.yuo): `take` starts (or, for a
/// re-sent take, extends) the holder's field, then prunes the other expired
/// fields;
/// `renew` extends a live field and answers 0 — never reviving it — for a
/// lapsed one. Its writes open with one that cannot grow memory, so a hold is
/// taken and renewed on a full `{gc}` shard too (b.pa1). See
/// `scripts/ext_read_hold.lua` and [`ReadHold`].
pub static EXT_READ_HOLD_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/ext_read_hold.lua")));

/// Compiled Lua script that performs the external-value PRE-FLIGHT store in one
/// atomic step on the multiplexed data pool (b.rej): `SET ext_data:{hash_hex}
/// NX`, `HINCRBY ext_meta:{hash_hex} ref_count 1`, and `HSET` the Valkey-side
/// `last_store_ms` stamp the reference-count reconciler reads. The exact
/// inverse of [`GC_DECREMENT_AND_COLLECT_SCRIPT`], and its mirror image in
/// shape — same two `{hash_hex}`-tagged keys, same meta-first `KEYS` order.
///
/// Replaces the per-hash `MULTI { SET data; HINCRBY meta ref_count 1 } EXEC` on
/// the exclusive `txn_pool` that [`ValkeyExternalValueStore::store_values`] ran
/// until b.rej. See `scripts/ext_store_value.lua` for why the `SET` is `NX`
/// (an existing data key, of any type, is kept against an unverified store,
/// while the increment and the stamp are paid on every call), the one heal its
/// ARGV[2] verified flag arms (a wrong-typed or wrong-length existing key is
/// replaced with verified bytes; no byte compare), what that gives up (a
/// corrupt blob of the right length, or one only an unverified path ever
/// re-stores, is not replaced), why the stamp is derived from
/// `redis.call("TIME")` rather than supplied by the caller (NTP skew across
/// replicas), why it is `HSET` and not `HSETNX`, and the invariant that the GC
/// decrement must never touch the stamp.
///
/// Two keys, so it routes by them under fred's `ClusterHash::Custom` and
/// satisfies the b.289 no-zero-key-`EVALSHA` fence.
///
/// Same lazy-`SHA-1` / `NOSCRIPT`-reload story as
/// [`GC_DECREMENT_AND_COLLECT_SCRIPT`].
pub static EXT_STORE_VALUE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/ext_store_value.lua")));

/// Compiled Lua script for atomic CAS acquire-or-refresh of a plain leader
/// lock — `le:checkpoint:leader` or `le:presence:sweeper:leader` (b.kf1
/// H-stor-4 follow-up). Replaces the GET-then-SET
/// pair that had a sub-30 s clobber window where another replica's NX
/// claim could land between our GET and our refresh-SET. Builds before the
/// GC protocol marker also took `le:gc:leader` with it; this build takes that
/// lock only through [`GC_LEADER_ACQUIRE_SCRIPT`]. See
/// `scripts/leader_acquire_or_refresh.lua` for the full contract.
pub static LEADER_ACQUIRE_OR_REFRESH_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/leader_acquire_or_refresh.lua")));

/// Compiled Lua script that acquires, refreshes or takes over the
/// protocol-marked GC leader lock `le:gc:leader` (b.1jt): it takes over a
/// value with no marker or a lower version (an older build's), a same-version
/// peer whose TTL is at or below [`GC_LEADER_TAKEOVER_BELOW_MS`] and a marked
/// value of any version with no TTL, and leaves any other marked holder alone
/// (a higher version at any TTL). Each `SET` follows a `PEXPIRE` that cannot
/// grow memory, so the lock is acquired and taken over on a full shard too
/// (b.pa1). Used only through [`acquire_gc_leader_lock`]. See
/// `scripts/gc_leader_acquire.lua`.
pub static GC_LEADER_ACQUIRE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/gc_leader_acquire.lua")));

/// Compiled Lua script that sets one of the GC's out-of-memory signal keys on
/// the `{gc}` slot (b.pa1): the collection request `le:ext:{gc}:oom:request`
/// (`storage::store_oom::StoreOomSignal`) and the upload hold
/// `le:ext:{gc}:oom:upload_hold` (the leader's out-of-memory collection). A
/// `SET … PX` after a `PEXPIRE` opener, so it writes on a store over
/// `maxmemory` too. See `scripts/gc_oom_signal.lua`.
pub static GC_OOM_SIGNAL_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/gc_oom_signal.lua")));

/// Compiled Lua script for the REFRESH-ONLY variant of the leader-lock CAS
/// (b.s5y). Extends the TTL iff the lock still holds the calling process's
/// `LeaderIdentity::lock_value`, and — unlike
/// [`LEADER_ACQUIRE_OR_REFRESH_SCRIPT`] — never claims an unset key. Used by
/// a long-running tick to prove mid-flight that its lease has been
/// continuously held; see
/// `scripts/leader_refresh_if_held.lua` for why re-acquiring in that
/// position would hide a duplicate-leadership window rather than close it.
pub static LEADER_REFRESH_IF_HELD_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/leader_refresh_if_held.lua")));

/// Compiled Lua script that atomically claims a pending GC member by
/// SREMing it from `ext_pending` and ZADDing it to `ext_inflight` with
/// the Valkey-side claim time (b.j9j item 2, migrated to
/// `redis.call("TIME")` by b.u96 item 7), encoded as claim ms +
/// [`GC_CLAIM_SCORE_OFFSET_MS`] (b.1jt) so an older build's drain never
/// selects it. It also maintains the release ledger (removes the member's
/// age, `HSETNX`es its bytes field), self-heals a member that left both sets
/// with ledger entries behind, and replies with the claim time (unencoded),
/// which the decrement's fence checks. See `scripts/gc_claim_pending.lua` for
/// the full contract.
pub static GC_CLAIM_PENDING_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/gc_claim_pending.lua")));

/// Compiled Lua script that re-claims one of this build's own `inflight`
/// claims in place (b.1jt): only while its score is still the one the pass's
/// selection read, it re-scores the member with a fresh encoded claim time and
/// replies with that time, so the pass decrements it (at most once, through
/// the replay record) and completes it — the member never goes back to
/// `pending`. Replaces the drain to `pending`. See
/// `scripts/gc_reclaim_stale.lua`.
pub static GC_RECLAIM_STALE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/gc_reclaim_stale.lua")));

/// Compiled Lua script that drops an older build's stranded `inflight`
/// claims (b.1jt): each claim scored below [`GC_CLAIM_SCORE_OFFSET_MS`] and
/// older than the decrement fence, only while its score is still the one the
/// pass's selection read, is removed from `inflight` and settled in the
/// release ledger WITHOUT a decrement — leak, never over-release. Its seventh
/// key is the `{gc}` migration sentinel ([`KeySchema::migration_sentinel`],
/// b.cyu), which keeps a run from being cut short mid-script while the `{gc}`
/// slot migrates. See `scripts/gc_drop_foreign_claims.lua`.
pub static GC_DROP_FOREIGN_CLAIMS_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/gc_drop_foreign_claims.lua")));

/// Compiled Lua script that stamps an age (`ZADD pending:age NX <now>`) on
/// each un-aged `pending` member the pass's selection sampled — an older
/// build's bare `SADD` (b.1jt) — so it is collected only past the age floor
/// and the read-hold clip like any other. See `scripts/gc_stamp_unaged.lua`.
pub static GC_STAMP_UNAGED_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/gc_stamp_unaged.lua")));

/// Compiled Lua script backing the checkpoint-store reconciliation protocol
/// (b.9qd): registers candidates on a `{recon}` ZSET with `ZADD NX` at the
/// Valkey-side clock, then returns those whose FIRST-observed score has aged
/// past a caller-supplied grace window.
///
/// Sibling of [`GC_CLAIM_PENDING_SCRIPT`] and shares its `redis.call("TIME")`
/// discipline for the same reason — a caller-supplied timestamp would let a
/// clock-ahead replica expire a candidate its peers had only just registered,
/// which here means deleting an artifact whose publish is still in flight. See
/// `scripts/recon_claim_expired.lua` for the full contract and the
/// two-observations-a-grace-apart safety argument.
pub static RECON_CLAIM_EXPIRED_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/recon_claim_expired.lua")));

// ── The retirement intent's scripts (b.yda; write-ahead since b.yuo Part R) ──
//
// A retirement that owes external-value releases records them on its own
// `{object_id}` slot BEFORE it enqueues them, and every enqueue attempt carries
// a token recorded there first, so the orphan sweep never re-drives a member
// until the `{gc}` slot has decided, once, that every recorded earlier attempt
// is void.
//
// This build's carriers are HASHes (member → 1, b.1jt), at the same key an
// older build writes a SET. An older build's carrier claim and reconciler read
// the key with `SCARD` / `SMEMBERS` and fail `WRONGTYPE` before writing
// anything, so they can neither pay nor delete a carrier this build wrote. A
// SET carrier is therefore an older build's, and this build never pays one —
// most older builds record no attempt token at all, and no token is not
// evidence that nothing ran (rule E4 below): the orphan sweep clears it if a
// recorded attempt landed and otherwise drops it, a leak the reference-count
// reconciler repairs (unless it is turned off,
// `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0`, when the leak is permanent).
//
// A member is enqueued twice only if (i) two different tokens land for one
// obligation, or (ii) a send under one token reaches `{gc}` after that token's
// record is gone.
//
// (i) cannot happen within an incarnation: each attempt is decided once on
// `{gc}` (the attempt-mode enqueue's check-then-set against the resolve's
// `HSETNX`, on one slot); a token is recorded on the carrier before its send;
// a drain sends its new token only after every recorded prior token resolved
// void and its settle confirmed the token set unchanged and the carrier
// present; and a second retirement of the incarnation is stopped by
// `ext_gc_scheduled`, or by `busy`. (Across a DR restore of the same object
// id, a begin still in transit can land on the restored header — a residual
// b.9ud plans to close with an incarnation stamp; not built.)
//
// (ii) is confined by three rules:
//
// - **F1.** A `landed` record is removed at once only by its token's own
//   sender, and only when its single send under the token answered `enqueued`
//   or `already` (no attempt under the token errored) and its clear then
//   answered `cleared` — so no carrier names the token and no service send
//   under it remains ([`forget_landed_attempts`]).
// - **F2.** Every other record is removed only by the GC loop's attempt audit,
//   at or after V ([`ENQUEUE_ATTEMPT_VOID_RETENTION`]); a `landed` record only
//   while no carrier holds its token.
// - **F3.** No service send is decided more than Y
//   ([`RETIRE_INTENT_SEND_WINDOW`]) after the command that recorded its token
//   was issued ([`ReleaseAttemptOutcome::NotSent`]).
//
// So no decision the service makes depends on timing — not the sweep's
// in-progress age, not the sweep interval, not how long `{gc}` is down. What is
// left is how long a command's bytes take to reach the store: more than V − Y
// after the service decided the send (policy 14); or, for an earlier copy of a
// command whose caller already got its reply, more than settle + U (the
// commit path's accepted class, policy 15). No design with bounded `{gc}`
// state removes that transit bound: a command already handed to the network
// cannot be recalled.
//
// Three rules bind every classification of an outcome here — code, comments
// and operator text alike:
//
// - **E — what a reply proves.** (E1) A script verdict proves only the state
//   the executing copy found: not WHO produced it (an earlier copy of the same
//   call may have), and not that no other copy is still in transit. (E2) An
//   error of any kind — `NOSCRIPT`, a timeout, "channel closed", a connection,
//   routing or cluster error, OOM, an unparseable reply — is outcome-unknown,
//   for the command and for every earlier copy of it: fred makes up to three
//   attempts per command on the data and maintenance pools and re-sends after
//   a reconnect or a redirect. No code branches on an error's kind to conclude
//   "did not run"; the one allowed branch is "reload and re-send", safe because
//   a re-send under the same token is decided by that token's `{gc}` record.
//   (E3) [`ReleaseAttemptOutcome::NotSent`] means this call transmitted nothing
//   under its token: only the window check before the call's first
//   transmission returns it, and a window check that fails after any
//   transmission returns an outcome-unknown `Err`. (E4) No token is not
//   evidence that nothing ran: an older build's untokened carrier recorded
//   none of its attempts.
// - **H — one helper.** Every transmission under an attempt token is decided
//   inside [`enqueue_ext_release_attempt`], the only builder of attempt-mode
//   arguments; it alone owns the window check before each transmission, the
//   `NOSCRIPT` reload and the classification under E. One call is one
//   decision: fred's re-sends, and the helper's own post-`NOSCRIPT` re-send,
//   are copies of it, so F1's "single send" means one call that returned
//   `enqueued` or `already`.
// - **T — present tense.** Operator and contract text (warnings, returned
//   error strings, metric descriptions, rustdoc, script headers) states only
//   what this build does. A planned remover or fence is named in comments as
//   planned and not built, and operator text never promises it.
//
// Every script below is idempotent under a replay of the same command (fred
// re-sends an in-flight command after its connection is lost): a begin re-send
// answers `begun` for its own token; a re-token re-send re-adds the same token
// or answers `in_progress`; a resolve returns the recorded verdicts; an
// attempt-mode enqueue re-send answers `already`; a settle or clear re-run
// finds the state it left. Clear and settle also write nothing when the carrier
// is absent, so a token HASH is deleted only in the same step as a carrier
// that exists. So a killed or retried command never does more than a replay
// would. Each script's header carries its own contract.

/// Compiled Lua script that begins a retirement intent (b.yuo Part R): in one
/// step on the object's slot, `ext_gc_scheduled=1` on the header, the owed
/// members onto the HASH carrier `le:{oid}:obj:extret` (b.1jt), and the first attempt
/// token onto `le:{oid}:obj:extret:attempts` — all before the enqueue is sent.
/// Answers `begun` / `already` / `busy` / `gone` ([`RetireBegin`]).
///
/// It replaces both the b.yda deferral script (which wrote the carrier only
/// after both enqueue attempts had failed) and the post-enqueue `EXISTS` +
/// `HSET ext_gc_scheduled`, which together let a landed-but-unacknowledged
/// enqueue, a crash between the enqueue and the flag, or two concurrent
/// retirements pay the same members twice. See `scripts/ext_retire_begin.lua`.
///
/// Three `{object_id}`-tagged keys: single-slot, and the b.289 no-zero-key
/// fence holds structurally.
pub static EXT_RETIRE_BEGIN_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/ext_retire_begin.lua")));

/// Compiled Lua script for the orphan sweep's re-token of a retirement
/// intent's carrier (b.yda; rewritten as non-destructive by b.yuo Part R).
///
/// It adds the drain's own attempt token to `le:{oid}:obj:extret:attempts` and
/// returns the carrier's members with the tokens recorded before it — or
/// answers `empty`, writing nothing (a token HASH beside no carrier is left in
/// place: only the incarnation boundary may remove it, and nothing in this
/// build does — b.9ud sub-task 3 plans that remover, not built), or
/// `in_progress` (the newest token is younger than the in-progress age, so a
/// retirement or another drain still owns the carrier). It reads the carrier
/// by its type (b.1jt): this build's HASH, or an older build's SET, which it
/// reports as such so the drain never pays it. It removes nothing
/// from the carrier: the carrier goes only once an attempt is known landed
/// ([`EXT_RETIRE_CLEAR_SCRIPT`]), by a cancel ([`EXT_RETIRE_SETTLE_SCRIPT`]),
/// or, for an older build's SET carrier, by a drop,
/// so a lost reply here leaves every member on the carrier. See
/// `scripts/ext_retire_retry_claim.lua`.
///
/// Two `{object_id}`-tagged keys, the carrier and the token HASH.
pub static EXT_RETIRE_RETRY_CLAIM_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/ext_retire_retry_claim.lua")));

/// Compiled Lua script that settles a retirement intent once every recorded
/// earlier attempt resolved void (b.yuo Part R, carrying b.yda's liveness
/// gate): the token set must still be exactly the re-token's `prior` plus its
/// new token and the carrier must still exist (else `changed`), and then the
/// id is either live again — `obj:hdr` or `obj:ext` exists, so the carrier and
/// its tokens are deleted and
/// `ext_gc_scheduled` cleared (`cancelled`) — or dead, and the drain pays the
/// members under its new token (`pay`, nothing written).
///
/// A live id has inherited the standing unreleased `+1` per hash the members
/// represent, so paying them would delete `ext_meta` + `ext_data` under a live
/// reference; cancelling leaks at worst, and the reference-count reconciler
/// drains a leak. The clear of `ext_gc_scheduled` hands the object's eventual
/// real retirement its re-derivation back, safe because every attempt of this
/// obligation is recorded and resolved void — the drain settles only this
/// build's HASH carriers, never an older build's SET carrier (b.1jt). See
/// `scripts/ext_retire_settle.lua` for the full
/// invariant, the `obj:ext`-in-the-gate argument, and the in-flight-retirement
/// residual.
///
/// Four `{object_id}`-tagged keys.
pub static EXT_RETIRE_SETTLE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/ext_retire_settle.lua")));

/// Compiled Lua script that deletes a retirement intent's carrier and token
/// HASH once one of its attempts is known landed (b.yuo Part R), or — under
/// the orphan sweep's own token — to drop an older build's SET carrier none
/// of whose recorded attempts landed (b.1jt): `cleared`, or
/// `superseded` when a drain re-tokened the carrier since that attempt's token
/// was recorded, or `gone` (the token or the carrier is already absent;
/// nothing written). It never touches `ext_gc_scheduled`: a paid
/// obligation keeps the flag even when the id is live again (policy 12). See
/// `scripts/ext_retire_clear.lua`.
///
/// Two `{object_id}`-tagged keys.
pub static EXT_RETIRE_CLEAR_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/ext_retire_clear.lua")));

/// Compiled Lua script that decides, once and for good, whether each of a
/// carrier's recorded attempts ran (b.yuo Part R): `HSETNX token void` on
/// `le:ext:{gc}:enqueue_attempt` and the recorded verdict back, `landed` or
/// `void`. Against the attempt-mode enqueue's own check-then-set on the same
/// slot, this means an attempt either landed before the resolve or never lands
/// — a copy that arrives later finds `void` and writes nothing, for as long as
/// the record stands (F2 keeps it at least V; see the section header above).
/// See `scripts/ext_enqueue_attempt_resolve.lua`.
///
/// One `{gc}`-tagged key.
pub static EXT_ENQUEUE_ATTEMPT_RESOLVE_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/ext_enqueue_attempt_resolve.lua")));

/// Compiled Lua script for the GC loop's audit of the `{gc}` attempt records
/// (b.yuo Part R): one explicit-cursor `HSCAN` page per call that removes the
/// `void` records older than [`ENQUEUE_ATTEMPT_VOID_RETENTION`] (V) and returns
/// the `landed` ones older than the same V for the loop to check against their
/// carriers (F2); and, in `forget` mode, removes a landed record only while its
/// value is unchanged. See `scripts/gc_enqueue_attempt_audit.lua`.
///
/// One `{gc}`-tagged key.
pub static GC_ENQUEUE_ATTEMPT_AUDIT_SCRIPT: LazyLock<Script> =
    LazyLock::new(|| Script::from_lua(include_str!("scripts/gc_enqueue_attempt_audit.lua")));

/// How many candidate members ride on one `recon_claim_expired` EVALSHA.
///
/// Bounds the ARGV vector per round-trip. Matches the order of magnitude of
/// [`GC_BATCH_SIZE`]; the pass registers in chunks and then issues one final
/// claim call, so this is a network-framing bound, not a work bound.
const RECON_REGISTER_CHUNK: usize = 256;

/// Leader lock TTL in milliseconds — the `ttl_ms` every production caller of
/// [`try_acquire_leader_lock`], [`refresh_leader_lock_if_held`] and
/// [`try_acquire_gc_leader`] passes (b.kf1 H-stor-4). 30 s matches the
/// original `EX 30` semantics:
/// long enough that a normal sweep tick never expires the lock under the
/// leader, short enough that a crashed leader's lock is reclaimable on
/// the next non-leader tick.
///
/// Exported so the checkpoint loop's leader lease can reuse it (its renewal
/// period is a third of it, `checkpointer::LEASE_RENEW_INTERVAL`) and tests
/// can derive drift guards from it rather than duplicating the literal (the
/// checkpoint loop's lease tests check a refresh against it).
pub const LEADER_LOCK_TTL_MS: u64 = 30_000;

/// The TTL, in ms, at or below which a GC loop takes `le:gc:leader` over from
/// a peer of its own protocol version (b.1jt; `scripts/gc_leader_acquire.lua`;
/// a higher version is never taken over while it has a TTL): a third of
/// [`LEADER_LOCK_TTL_MS`], so the holder has missed refreshes for 20 s —
/// about twenty of its once-a-second refreshes — before a peer takes over.
/// The holder refreshes once a second from a ticker independent of its passes
/// ([`spawn_gc_task`]) while its GC loop is running and making progress
/// ([`GC_PROGRESS_STALL_AFTER`]), so a long but progressing pass does not stop
/// its refreshes; what triggers a takeover is a holder that stopped, crashed,
/// lost the store, or whose GC loop died or stalled while the process lives on
/// (the ticker stops renewing with it). While any process of this build with a
/// running, progressing GC loop can reach the store (and no newer marked build
/// holds the key), its refreshes or its takeover keep the key from lapsing,
/// and an older build, which only acquires an absent key, does not lead. The
/// key lapses only when every such process loses the store for the TTL; see
/// this module's header comment ("Older builds sharing the store") for why
/// that window is still safe. Taking over a holder whose refreshes were only
/// stalled briefly runs two leaders, which every leader-gated step is safe
/// under by construction.
pub const GC_LEADER_TAKEOVER_BELOW_MS: u64 = LEADER_LOCK_TTL_MS / 3;

/// `S`, how long the GC loop may go without marking progress before its
/// lease refresh ticker stops renewing `le:gc:leader` (b.1jt): 20 s.
///
/// The loop marks progress (`GcProgress`) at every wake-up — at least once a
/// second between passes, leader or not — and, inside a pass, after every
/// step, every member's claim → decrement → complete, every replay-record
/// unmark, and every `HEXISTS` of the enqueue-attempt audit. So a healthy
/// loop, even one in a long pass, marks progress within a few store round
/// trips: the longest unmarked stretch is one member's claim → decrement →
/// complete, three round trips (nine if each needs a `NOSCRIPT` reload:
/// EVALSHA, SCRIPT LOAD, EVALSHA). Each round trip is bounded by the command
/// timeout (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`, 5 s by default), and one
/// that hits it fails and ends the member early (a failed claim leaves it
/// unclaimed, a failed decrement or completion leaves it in `inflight` or ends
/// the pass). At ordinary latencies (milliseconds per round trip) a
/// progressing pass stays far below 20 s between marks and never trips it; it
/// can trip only when one member's round trips each run close to the command
/// timeout yet still succeed. Even then the lock is lost only if the gap
/// outlasts `S` plus the 20 s a peer waits before taking over, about 40 s,
/// which takes nearly all of one member's nine reloading round trips each
/// running close to the timeout and succeeding; renewal resumes at the next
/// mark.
///
/// It is below [`LEADER_LOCK_TTL_MS`] so a loop that is wedged — a pass that
/// awaits forever without ending, which the child-token stop in
/// [`spawn_gc_task`] does not see — gives the lock up: the ticker stops
/// renewing within `S` (plus one tick) of the last progress, and a peer takes
/// the key over once its TTL falls to [`GC_LEADER_TAKEOVER_BELOW_MS`], about
/// `S + 20 s` after the last progress. Before the ticker existed the same
/// wedge lost the lock 30 s after the pass began; the gate restores a bounded
/// handover of that kind, never holding the lock indefinitely. With the command timeout
/// disabled (`0`) a single round trip can exceed `S`; the ticker then pauses
/// until progress resumes, which at worst lets the key lapse during that pass
/// — exactly the behaviour before the ticker existed.
///
/// The gate keys on the GC loop alone. The orphan sweep, the reconciler and
/// the external-value backup repair
/// ride the same key and identity, and keep it through a long pass of their
/// own because the GC loop beside them keeps marking progress; when that loop
/// is wedged, their passes lose the key mid-pass as they did before the
/// ticker existed, which they are built for (each is safe under two leaders).
pub const GC_PROGRESS_STALL_AFTER: std::time::Duration = std::time::Duration::from_millis(20_000);
const _: () = assert!(
    GC_PROGRESS_STALL_AFTER.as_millis() < LEADER_LOCK_TTL_MS as u128,
    "the progress bound must be below the lock TTL"
);

/// The GC loop's progress heartbeat (b.1jt): the time the loop last marked
/// progress, shared between the loop (which marks it) and the lease refresh
/// ticker (which renews `le:gc:leader` only while the last mark is younger
/// than [`GC_PROGRESS_STALL_AFTER`]). In-process only: it writes nothing to
/// the store, and the lock's key and value are unchanged, so it adds no
/// behaviour an older build sharing the store can observe beyond the lock's
/// renewal timing.
///
/// Cheap to clone (one `Arc`); every clone marks and reads the same time. A
/// fresh heartbeat counts as marked at its creation.
#[derive(Debug, Clone)]
pub(crate) struct GcProgress {
    base: tokio::time::Instant,
    last_ms: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl GcProgress {
    /// A heartbeat marked now.
    pub(crate) fn new() -> Self {
        Self {
            base: tokio::time::Instant::now(),
            last_ms: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Mark progress now. Monotonic: a mark never moves the time back.
    pub(crate) fn mark(&self) {
        let now = u64::try_from(self.base.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.last_ms
            .fetch_max(now, std::sync::atomic::Ordering::Relaxed);
    }

    /// How long ago progress was last marked.
    pub(crate) fn since_last(&self) -> std::time::Duration {
        let now = u64::try_from(self.base.elapsed().as_millis()).unwrap_or(u64::MAX);
        let last = self.last_ms.load(std::sync::atomic::Ordering::Relaxed);
        std::time::Duration::from_millis(now.saturating_sub(last))
    }
}

/// The offset this build adds to the claim time it scores an `inflight` claim
/// with (b.1jt): a claim made at Valkey ms `t` is scored `t + 10^13`.
///
/// Every older build scores its claims with the bare claim time and drains
/// back to `pending` only claims scored in `[0, now − 30 s]`; an encoded score
/// is never in that range, so no older drain re-queues a member this build
/// claimed (and possibly decremented), while an older reconciler, which
/// `ZSCAN`s `inflight` and ignores the score, still counts it as owed. The
/// offset is positive so an encoded claim sorts after every bare one, and an
/// older build's `ZRANGE inflight 0 0` still finds its own oldest claim. A
/// score below the offset is therefore an older build's claim: this build
/// never decrements one, and drops it once it is older than the decrement
/// fence ([`GC_DROP_FOREIGN_CLAIMS_SCRIPT`]).
///
/// `scripts/gc_claim_pending.lua`, the only writer of a fresh claim, carries
/// the same value as a literal (so it still reads only its member argument);
/// every script that reads the score — `gc_select.lua`,
/// `gc_reclaim_stale.lua`, `gc_drop_foreign_claims.lua` — is handed this
/// constant. The two must stay equal: a mismatch would read this build's
/// claims as an older build's (dropped, leaking) or the reverse (decremented
/// a second time).
///
/// `t + 10^13` is an integer far below 2^53, so the score is an exact double
/// and age order is preserved. The scripts hand it to `ZADD` as a Lua number,
/// never through string concatenation (whose 14-significant-digit format
/// would round it). The claim script's reply, the decrement's fence and the
/// replay record all keep the unencoded claim time; only the `inflight` score
/// carries the offset.
pub const GC_CLAIM_SCORE_OFFSET_MS: u64 = 10_000_000_000_000;

/// The most members one GC pass selects to claim (`gc_select.lua`'s candidate
/// limit). A pass that selects a full batch and collects at least one member
/// of it runs the next pass at once.
const GC_BATCH_SIZE: usize = 100;

/// The most settled replay records one GC pass tears down.
const GC_SETTLED_BATCH_SIZE: usize = 100;

/// How many pending members `gc_select.lua` samples (`SRANDMEMBER`) on every
/// pass while `pending` is non-empty, looking for un-aged ones (pending with
/// no `pending:age` entry — an older build's bare `SADD`), which the pass
/// then stamps ([`GC_STAMP_UNAGED_SCRIPT`]).
const GC_UNAGED_SAMPLE: usize = 100;

/// The most stale claims of each kind one GC pass acts on: this build's own
/// claims older than [`GcPolicy::reclaim_after`], re-claimed in place, and an
/// older build's claims older than the decrement fence, dropped
/// (`gc_select.lua`'s `stale_claim_limit`).
const GC_STALE_CLAIM_BATCH_SIZE: usize = 100;

/// The most members one GC pass has between claim and completion at once, on
/// the maintenance pool (b.yuo). Bounds the pass's command fan-out; the
/// members of one pass are independent of each other, each on its own hash
/// slot.
const GC_MEMBER_CONCURRENCY: usize = 16;

/// `P`, the GC loop's longest sleep (b.yuo): a leader sleeps at most this
/// long between passes — less when the oldest member falls due sooner, not at
/// all after a pass that found more work than one batch — and a non-leader
/// re-checks the lease this often. It is also the period of the lease refresh
/// ticker [`spawn_gc_task`] runs beside the loop, independent of any pass
/// (b.1jt). One second keeps collection within about a second of a member
/// falling due, at the cost of about two lease calls per replica per second
/// (the loop's and the ticker's).
pub const GC_PASS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// `F`, the decrement's claim-age fence, in milliseconds (b.yuo): a GC
/// decrement executing more than this long after its claim, by its own
/// primary's clock, is refused `stale` and writes nothing
/// (`scripts/gc_decrement_and_collect.lua`).
///
/// Normal claim-to-decrement latency is milliseconds, so 15 s refuses only a
/// copy written late — by fred after its caller gave up, or by a sweeper
/// paused after its claim — and a store frozen for longer than this only
/// delays that member's collection by the re-claim age. It must stay below the
/// re-claim age ([`GcPolicy::reclaim_after`]), so a member is never re-claimed
/// while a copy of its old claim can still act, and below the replay-record
/// retention minus the tolerated clock skew ([`GC_REPLAY_RECORD_RETENTION`]).
///
/// It is also the age past which a GC pass drops an older build's stranded
/// claim (b.1jt, [`GC_DROP_FOREIGN_CLAIMS_SCRIPT`]): 15 s is past that build's
/// own claim-to-decrement latency, and inside the 30 s an older build's
/// startup drain waits before re-queueing a claim, so while this build leads
/// no older drain re-queues a claim it could otherwise have found.
pub const GC_DECREMENT_FENCE_MS: u64 = 15_000;

/// `U`, the replay-record retention (b.yuo): a member's
/// `le:ext:applied:{hash_hex}` record is torn down only once the member has
/// been settled — out of both `pending` and `inflight` — this long.
///
/// Must exceed [`GC_DECREMENT_FENCE_MS`] plus the cluster clock skew it
/// tolerates: every claim of a member precedes its settle, and a copy of a
/// claim can act only until claim time + F (+ skew), so a teardown at settle +
/// U comes after any copy's last chance. 120 s tolerates a `{hash_hex}`
/// primary running up to U − F = 105 s behind the `{gc}` primary. The cost is
/// that an identical member enqueued within U of its previous decrement is
/// treated as a replay; a genuinely repeated identical member inside the
/// window leaks one reference to the reconciler.
///
/// **The replay record is one layer of the at-most-once decrement, and it is
/// time-bounded.** It answers a duplicate enqueue `replayed` only until U
/// after the member settled, which covers a fred re-send of a commit-path
/// enqueue (seconds). The service's own re-presentations of a member — a
/// retirement's in-doubt enqueue re-driven off its carrier, or re-derived
/// after a crash — can come any time later, so they do not lean on U: the
/// retirement intent decides each enqueue attempt once on the `{gc}` slot (see
/// [`EXT_ENQUEUE_ATTEMPT_RESOLVE_SCRIPT`] and [`ENQUEUE_ATTEMPT_VOID_RETENTION`]).
/// What still leans on U there is the commit path's own class: an earlier copy
/// of a retirement or drain enqueue whose caller got a clean reply (and so
/// removed its token's record at once, F1) that reaches the store more than
/// settle + U later (policy 15).
pub const GC_REPLAY_RECORD_RETENTION: std::time::Duration = std::time::Duration::from_secs(120);

/// The margin below the earliest live read hold's start that `gc_select.lua`
/// clips the candidate cutoff to (b.yuo): no member enqueued within this long
/// before that hold began, or later, is collected while any hold field lives.
/// Absorbs the gap between a reader's snapshot and its hold write. A test can
/// override it per store (`ValkeyExternalValueStore::with_hold_margin_for_test`,
/// `test-support` only).
const GC_HOLD_MARGIN_MS: u64 = 1_000;

/// How long one write of a holder's `{gc}` read hold field keeps it live
/// (b.yuo): the `expires_ms` `ext_read_hold.lua` writes, and the key's
/// `PEXPIRE`. A catch-up that missed a blob holds its field across its re-read
/// and rebuild, renewing it every [`EXT_READ_HOLD_RENEW_INTERVAL`], and
/// releases it (`HDEL`) when the rebuild ends; the TTL matters only when the
/// release never runs — the holder died, or the `HDEL` failed — and bounds how
/// long such a field keeps stopping the collection of newer releases (which,
/// while the release total is at the budget, also keeps uploads waiting at
/// the upload gate). 15 s leaves room for two late renewals.
/// [`ReadHoldTiming::PRODUCTION`] carries it; tests may override it.
pub const EXT_READ_HOLD_TTL: std::time::Duration = std::time::Duration::from_secs(15);

/// How often a catch-up renews its `{gc}` read hold field while its re-read
/// and rebuild still run (b.yuo): a third of [`EXT_READ_HOLD_TTL`], so two
/// renewals can be late before the field could lapse. A renewal that finds
/// the field lapsed ends the catch-up `UNAVAILABLE`.
pub const EXT_READ_HOLD_RENEW_INTERVAL: std::time::Duration = std::time::Duration::from_secs(5);

/// How often the GC loop starts a release-ledger audit cycle (b.yuo); a cycle
/// also starts at the first pass of a new leadership. A cycle runs one
/// `gc_ledger_audit.lua` page per pass until its cursor wraps. The same cycle
/// starts the enqueue-attempt audit (`gc_enqueue_attempt_audit.lua`), which
/// runs one page per pass on its own cursor.
const GC_LEDGER_AUDIT_INTERVAL: std::time::Duration = std::time::Duration::from_secs(300);

/// `V`, how long the GC loop's attempt audit keeps every record on
/// `le:ext:{gc}:enqueue_attempt` it removes (b.yuo Part R, policy 14 as
/// amended): 24 h, for `void` records and for `landed` ones alike (F2).
///
/// A record is the only guard against a send under its token: a `void` record
/// makes a late copy of a voided attempt write nothing, and a `landed` record
/// makes a second send under the same token (the retirement's second attempt,
/// or a copy fred writes late) answer `already`. Only the token's own sender
/// may remove a `landed` record at once, after a single clean send and a
/// cleared carrier (F1, [`forget_landed_attempts`]); every other record — a
/// void, a landed record of an in-doubt send (an errored attempt, a prior
/// token a drain found, a failed eager removal) — waits V, and a landed one
/// goes only while no carrier holds its token.
///
/// Because no service send is decided more than Y
/// ([`RETIRE_INTENT_SEND_WINDOW`]) after its token was recorded (F3), **the
/// residual is transit only: a send whose bytes reach the store more than
/// V − Y after the service decided it can still enqueue the members once more,
/// and its decrement double-releases.** That is an accepted bound: after b.zt3
/// it needs a replica alive with a router stuck for most of a day. The cost is
/// about 100 bytes per in-doubt retirement or drain, bounded by failures rather
/// than by traffic; keeping records forever would close the bound but grow by
/// one field per failure for the deployment's life.
pub const ENQUEUE_ATTEMPT_VOID_RETENTION: std::time::Duration =
    std::time::Duration::from_secs(24 * 60 * 60);

/// `Y`, the retirement intent's send window (b.yuo Part R, F3): 30 s.
///
/// A retirement, and the orphan sweep's drain, compute `send_by` as now + Y
/// just before the command that records their token (the begin, or the
/// re-token), and every transmission an attempt-mode enqueue would make is
/// checked against it, in two cases. Past it before the call's first
/// transmission: nothing is sent, and the answer is
/// [`ReleaseAttemptOutcome::NotSent`] — the token stays on the carrier, never
/// sent by this call, and the next sweep resolves it. Past it before the
/// re-send that follows a `NOSCRIPT` reload: the re-send is not made, and the
/// answer is an outcome-unknown `Err`, because an earlier copy of the call may
/// have run (rule E). Without this bound a supported
/// setting (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS=0`) or a frozen process could
/// let the service decide a send after any finite record retention, so this is
/// what makes every service decision independent of V.
///
/// Correctness needs only Y ≪ V. Y below the sweep's in-progress age X
/// (`orphan_sweep::RETIRE_INTENT_IN_PROGRESS_AGE`, 60 s) keeps a sender from
/// deciding a send after a sweep may already have taken its carrier, which is
/// efficiency. Tests override it per client
/// (`ValkeyClient::set_retire_intent_send_window_for_test`, `test-support`
/// only).
pub const RETIRE_INTENT_SEND_WINDOW: std::time::Duration = std::time::Duration::from_secs(30);

/// The `HSCAN` COUNT hint of one enqueue-attempt audit page.
const ENQUEUE_ATTEMPT_AUDIT_PAGE: usize = 256;

/// How long the upload hold of an out-of-memory collection lives when the
/// leader does not delete it (b.pa1): the `PX` of
/// `le:ext:{gc}:oom:upload_hold`. Above [`OOM_COLLECTION_MAX_DURATION`], so the
/// hold stands for the whole collection, and short enough that a leader that
/// dies mid-collection holds uploads back for no longer than this.
pub const OOM_UPLOAD_HOLD_TTL: std::time::Duration = std::time::Duration::from_secs(15);

/// The longest one out-of-memory collection runs passes (b.pa1): it repeats
/// while a pass reports more work ([`GcBatchResult::more`]), and stops after
/// the pass that ends past this, so the uploads it holds wait at most about
/// this long for it.
pub const OOM_COLLECTION_MAX_DURATION: std::time::Duration = std::time::Duration::from_secs(10);

/// How soon after an out-of-memory collection that freed something, or that
/// ended `failed`, the leader starts the next one on a new request (b.pa1): a
/// store still refusing writes after a collection that freed bytes gets
/// another within a second, and so does one whose collection a pass error cut
/// off — a transient `{gc}` error is no evidence the collection cannot help.
/// Only the GC leader runs collections, so this is at most one attempt a
/// second per deployment (two, briefly, while two leaders overlap after a
/// takeover). The collection's own WARN stays rate-limited
/// ([`OOM_COLLECTION_LOG_INTERVAL`]), so a persistent error does not repeat
/// that line once a second. A `failed` collection still returns the pass's
/// error, so the GC loop's per-pass "GC pass failed" ERROR fires once per
/// errored pass, as it does for any other errored pass; this cadence does not
/// change that.
pub const OOM_COLLECTION_RETRY_AFTER_FREED: std::time::Duration = std::time::Duration::from_secs(1);

/// How soon after an out-of-memory collection that found nothing to free
/// (`freed_nothing`, `nothing_to_collect`, `refused`) the leader starts the
/// next one on a new request (b.pa1): a store full of data in use keeps
/// refusing writes and keeps requesting, and a collection that cannot help
/// then runs (and holds uploads) at most this often.
pub const OOM_COLLECTION_RETRY_AFTER_FUTILE: std::time::Duration =
    std::time::Duration::from_secs(30);

/// The least time between two of the leader's per-collection WARN lines
/// (b.pa1); the lines held back are counted on the next one (`suppressed`).
const OOM_COLLECTION_LOG_INTERVAL: std::time::Duration = std::time::Duration::from_secs(30);

/// The least time between two WARN lines of a pass skipping a `{gc}` step
/// (the stamp, or a re-claim) the store refused `-OOM` (b.pa1).
const GC_OOM_SKIP_LOG_INTERVAL: std::time::Duration = std::time::Duration::from_secs(60);

/// The external-value GC loop's timing policy (b.yuo).
///
/// Production runs [`GcPolicy::PRODUCTION`], and, for an out-of-memory
/// collection only, [`GcPolicy::OOM_COLLECTION`] (b.pa1); tests inject other
/// values through [`process_pending_gc`], [`spawn_gc_task`] and
/// [`spawn_gc_task_with_leader`] (and `GcPolicy::IMMEDIATE` under the
/// `test-support` feature). So a production build collects younger than the
/// production floor only while the store refuses writes for lack of memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GcPolicy {
    /// `G`: no member is collected younger than this, measured from its first
    /// enqueue, while the release total is below the pressure threshold
    /// (`H / 2`).
    pub min_age: std::time::Duration,
    /// The age floor while the release total is at or above the pressure
    /// threshold. A pass then collects only the excess over the threshold,
    /// oldest first: every member past `min_age`, then younger ones down to
    /// this age, each added only while the release-ledger bytes of every
    /// member selected so far, those past `min_age` included, do not exceed
    /// the total's excess over the threshold (the younger member that takes
    /// them past it is the last).
    /// Not younger than `min_age`, it is simply the cutoff, and the excess
    /// plays no part.
    pub pressure_min_age: std::time::Duration,
    /// `U`: how long a settled member's replay record is kept before teardown.
    /// See [`GC_REPLAY_RECORD_RETENTION`].
    pub replay_record_retention: std::time::Duration,
    /// `D`: one of this build's own claims in `inflight` older than this is
    /// re-claimed in place by the next leader pass and decremented (at most
    /// once, through the replay record) and completed there; it never goes
    /// back to `pending`. Must exceed [`GC_DECREMENT_FENCE_MS`], so the old
    /// claim's late copies are refused `stale` by then.
    pub reclaim_after: std::time::Duration,
    /// Whether `gc_select.lua` clips its cutoff below a live read hold
    /// (`le:ext:{gc}:read_hold`).
    pub hold_clip: bool,
}

impl GcPolicy {
    /// The production policy: `min_age` 5 s, `pressure_min_age` 0, `U` 120 s,
    /// `D` 30 s (the leader-lock TTL; the re-claim age), hold clipping on.
    ///
    /// A release is collected from 5 s after it while the release total is
    /// below the pressure threshold `T = H / 2` — except during an
    /// out-of-memory collection, which runs [`Self::OOM_COLLECTION`] and
    /// collects at any age (b.pa1) — so the garbage awaiting
    /// collection stays near the release rate × (5 s + one pass). At or above
    /// `T` a pass also collects the excess over `T`, oldest first, at any age,
    /// so the total holds near `T`, higher only while a live read hold keeps
    /// the releases newer than its start. Collecting this soon is safe for
    /// readers only because no catch-up treats a blob collected under its
    /// snapshot as missing: its build probes, and on a miss it takes a `{gc}`
    /// read hold and reads the layer again — its own atomic read, or, above
    /// the atomic-read limits, the object's fenced-read flight (a successor
    /// of the stale one, when the miss was in a flight's snapshot) — which
    /// `hold_clip` keeps clear of collection
    /// (`grpc::subscribe::build_catch_up`). A reader of blobs that
    /// does without that protocol must not be added while these floors stand.
    /// A unit test pins these values.
    pub const PRODUCTION: GcPolicy = GcPolicy {
        min_age: std::time::Duration::from_secs(5),
        pressure_min_age: std::time::Duration::ZERO,
        replay_record_retention: GC_REPLAY_RECORD_RETENTION,
        reclaim_after: std::time::Duration::from_millis(LEADER_LOCK_TTL_MS),
        hold_clip: true,
    };

    /// The out-of-memory collection's policy (b.pa1): both age floors 0, so a
    /// pass selects every released member, oldest first, at any age, up to a
    /// batch; `U` and `D` and hold clipping as in [`Self::PRODUCTION`]. Run by
    /// the leader only for an out-of-memory collection (module header), never
    /// as the loop's own policy.
    ///
    /// Collecting at any age is safe for readers for the reason
    /// [`Self::PRODUCTION`]'s floors are: a catch-up that misses a blob takes a
    /// `{gc}` read hold and reads again, and `hold_clip` keeps every release
    /// enqueued after the earliest live hold began out of the selection. The
    /// hold's take and renew run on a full `{gc}` shard too (their scripts'
    /// out-of-memory opener), so the protocol holds while the store refuses
    /// other writes. A unit test pins these values.
    pub const OOM_COLLECTION: GcPolicy = GcPolicy {
        min_age: std::time::Duration::ZERO,
        pressure_min_age: std::time::Duration::ZERO,
        replay_record_retention: GC_REPLAY_RECORD_RETENTION,
        reclaim_after: std::time::Duration::from_millis(LEADER_LOCK_TTL_MS),
        hold_clip: true,
    };

    /// A test policy that collects everything due at once: both age floors and
    /// `U` are 0, so a pass claims every pending member and the next tears
    /// down their replay records — the pre-b.yuo "drain now" behaviour existing
    /// tests and the reconciler's test helpers rely on. `D` stays 30 s.
    /// Compiled only for tests and the `test-support` feature, so production
    /// builds cannot use it.
    #[cfg(any(test, feature = "test-support"))]
    pub const IMMEDIATE: GcPolicy = GcPolicy {
        min_age: std::time::Duration::ZERO,
        pressure_min_age: std::time::Duration::ZERO,
        replay_record_retention: std::time::Duration::ZERO,
        reclaim_after: std::time::Duration::from_millis(LEADER_LOCK_TTL_MS),
        hold_clip: true,
    };
}

/// A duration in whole milliseconds, saturating, as a decimal Lua argument.
fn duration_ms(d: std::time::Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// Current Unix-epoch time in milliseconds. Used only by
/// `MockExternalValueStore` to simulate enqueue and claim timestamps in the
/// in-memory mock; production code samples `redis.call("TIME")` Valkey-side
/// inside the GC scripts, so the wall-clock NTP-skew concern does not apply on
/// the production path.
fn now_unix_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        // Pre-1970 wall clock is implausible in production; if a
        // machine ever runs with a wildly-misconfigured clock we
        // fall back to 0, which makes the mock's re-claim treat every claim
        // as stale and re-claim it — the mock's applied set then answers
        // the re-run as a replay, so nothing is decremented twice.
        .unwrap_or(0)
}

/// Test whether a `ServiceError` is the `Aborted("transient I/O: ...")`
/// produced by [`map_transient_fred_err`] (b.j9j item 3). Used inside
/// `process_pending` to swallow a single transient member failure and
/// continue with the rest of the pass — the member stays in `inflight`
/// (or `pending`, depending on which sub-op failed) for a later pass, or its
/// re-claim, to retry.
///
/// The prefix match is intentional and matches the well-known contract
/// at [`map_transient_fred_err`]'s call sites — see
/// `client.rs`'s `transient_classification_test` for the integration
/// pin that prevents this prefix from drifting.
fn is_transient_aborted(err: &ServiceError) -> bool {
    matches!(err, ServiceError::Aborted(msg) if msg.starts_with("transient I/O:"))
}

/// Internal helper used by `read_batch` to distinguish "meta absent" (unknown
/// hash) from "meta present, data missing" (integrity error). See the `match`
/// in `read_batch` for how this disambiguates the two error variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetaState {
    Present,
    Missing,
}

// ── Release events (b.qpf) ─────────────────────────────────────────────────

/// What produced an owed `ext_meta` `ref_count` decrement (b.qpf).
///
/// Rides the `le:ext:{gc}:pending` member so two independently-owed decrements
/// for the same hash cannot collapse into one. See [`ExtRelease`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReleaseEvent {
    /// A commit superseded the reference: the committing `seq`.
    Commit(u64),
    /// The object itself is being retired (`DeleteLayer`, `CreateLayer`
    /// overwrite, orphan sweep).
    Retirement,
    /// The reference-count reconciler observed a surplus of `n` on this hash,
    /// twice a grace apart at the same amount, and is handing it back (b.yda).
    ///
    /// Carries the surplus amount for the same reason the reconciler's candidate
    /// member does: it is what distinguishes two repairs of one hash made at
    /// different amounts, and it makes a replayed repair of the SAME amount
    /// collapse in the pending Set instead of double-decrementing.
    Reconcile(u32),
    /// A call that pre-stored this hash was answered from an operation an
    /// EARLIER call had already committed (an `UpdateLayer` or `CreateLayer`
    /// re-send carrying the same client `operation_id`), or was refused
    /// because the id was recorded for a different request — so it committed
    /// nothing, and the pre-flight `+1` it paid is owed back. Carries the
    /// call's random 128-bit commit token.
    ///
    /// The token, not the object or the commit, is the identity: nothing this
    /// call wrote can name the release, and two calls answered from the same
    /// record each paid their own `+1`. A replayed enqueue of the SAME call
    /// collapses in the pending Set.
    Resend(u128),
}

impl ReleaseEvent {
    /// The event's spelling inside a pending-set member. Stable wire text —
    /// see [`ExtRelease::member`].
    fn token(self) -> String {
        match self {
            ReleaseEvent::Commit(seq) => seq.to_string(),
            // Not a decimal, so it can never collide with a `Commit` token.
            ReleaseEvent::Retirement => RETIREMENT_EVENT_TOKEN.to_owned(),
            // Prefixed, so it collides with neither a `Commit` decimal nor the
            // `ret` token.
            ReleaseEvent::Reconcile(n) => format!("{RECONCILE_EVENT_TOKEN_PREFIX}{n}"),
            // Prefixed, so it collides with neither a `Commit` decimal, the
            // `ret` token, nor a `rec{n}` repair.
            ReleaseEvent::Resend(token) => format!("{RESEND_EVENT_TOKEN_PREFIX}{token:032x}"),
        }
    }
}

/// The `event` token for [`ReleaseEvent::Retirement`].
const RETIREMENT_EVENT_TOKEN: &str = "ret";

/// The `event` token prefix for [`ReleaseEvent::Reconcile`] (b.yda). The
/// surplus amount follows it, so the full token is `rec{n}` — not a decimal, so
/// it cannot collide with a `Commit` token, and distinct from
/// [`RETIREMENT_EVENT_TOKEN`], so it cannot collide with a retirement.
const RECONCILE_EVENT_TOKEN_PREFIX: &str = "rec";

/// The `event` token prefix for [`ReleaseEvent::Resend`]. The call's commit
/// token follows it as 32 lowercase hex digits, so the full token is
/// `rsd<32hex>` — not a decimal, and distinct from both `ret` and `rec{n}`.
const RESEND_EVENT_TOKEN_PREFIX: &str = "rsd";

/// The `object_id` a reconciler repair member carries (b.yda) — see
/// [`ExtRelease::reconcile`]. `0` is expressible only as a sentinel: object ids
/// come from `INCR le:counter:object_id`, which returns 1 first.
pub(crate) const RECONCILE_NO_OBJECT_SENTINEL: u64 = 0;

/// One owed `ext_meta` `ref_count` decrement, with the identity of the event
/// that owes it (b.qpf).
///
/// # Why the event is part of the member
///
/// `ref_count` counts **live objects referencing the hash** — one increment per
/// object per distinct hash — so a decrement is *counted*, not idempotent. The
/// release channel is `SADD le:ext:{gc}:pending <member>`, and a Set collapses
/// equal members. Pre-b.qpf the member was the bare `hash_hex`, so two
/// genuinely-distinct owed decrements — two objects retiring before a GC tick,
/// or a rewrite and a retirement of the same blob — collapsed into ONE, and the
/// second reference leaked permanently.
///
/// Naming the event restores exactly-once *per event* while keeping SADD's
/// idempotence *within* an event, which is what the replay-safety of
/// `schedule_old_object_ext_gc`'s `ext_gc_scheduled` guard and `delete_layer`'s
/// retry both depend on. A retiring object takes no further commits
/// (`superseded=1`) and an `object_id` is never re-issued, so `ret` is a stable
/// per-object event; a commit's `seq` is unique per object by construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExtRelease {
    /// The content hash whose `ref_count` is owed a decrement.
    pub hash: [u8; 32],
    /// The object that is giving up its reference.
    pub object_id: u64,
    /// The event that owes the decrement.
    pub event: ReleaseEvent,
}

impl ExtRelease {
    /// Convenience constructor for a commit-derived release.
    pub fn commit(hash: [u8; 32], object_id: u64, seq: u64) -> Self {
        Self {
            hash,
            object_id,
            event: ReleaseEvent::Commit(seq),
        }
    }

    /// Convenience constructor for a retirement-derived release.
    pub fn retirement(hash: [u8; 32], object_id: u64) -> Self {
        Self {
            hash,
            object_id,
            event: ReleaseEvent::Retirement,
        }
    }

    /// Convenience constructor for a reconciler-derived repair (b.yda).
    ///
    /// `object_id` is `0`, the documented **no-object sentinel**: this release
    /// belongs to no object, because it is handing back an increment whose
    /// owning object is unknown (it crashed before enqueuing, it predates the
    /// `obj:ext` index, or it was a restore's suppressed surplus). `0` is safe
    /// as a sentinel because object ids are minted by `INCR` from 1 and are
    /// never re-issued, so no real object can ever spell one.
    ///
    /// The resulting member grants exactly one decrement, subject to the same
    /// claim / underflow accounting as any other member — the GC scripts are
    /// member-opaque and do not know or care that this one names no object.
    pub fn reconcile(hash: [u8; 32], surplus: u32) -> Self {
        Self {
            hash,
            object_id: RECONCILE_NO_OBJECT_SENTINEL,
            event: ReleaseEvent::Reconcile(surplus),
        }
    }

    /// Convenience constructor for the pre-flight `+1` of a call that
    /// committed nothing because its `operation_id` was already recorded
    /// ([`ReleaseEvent::Resend`]). `object_id` is the object the call was
    /// answered about.
    pub fn resend(hash: [u8; 32], object_id: u64, call_token: u128) -> Self {
        Self {
            hash,
            object_id,
            event: ReleaseEvent::Resend(call_token),
        }
    }

    /// The `le:ext:{gc}:pending` member for this release:
    /// `{hash_hex}:{object_id}:{event}`.
    ///
    /// The hash is the prefix up to the FIRST `:`, which is what
    /// [`pending_member_hash_hex`] relies on and what keeps a legacy bare
    /// `{hash_hex}` member (written by a pre-b.qpf replica, or already sitting
    /// in `pending` at upgrade time) parsing to itself — a rolling deploy needs
    /// no migration.
    ///
    /// b.yda: a reconciler repair spells `{hash_hex}:0:rec{n}`. The `0` is
    /// [`Self::reconcile`]'s no-object sentinel and the `rec{n}` token is
    /// unambiguous against both the decimal `Commit` tokens and the `ret`
    /// retirement token, so the member format stayed a pure extension —
    /// `pending_member_hash_hex` parses it unchanged and all three GC scripts
    /// needed no edit. A re-send release ([`Self::resend`]) spells
    /// `{hash_hex}:{object_id}:rsd<32hex>` and extends it the same way.
    pub fn member(&self) -> String {
        format!(
            "{}:{}:{}",
            sha256_to_hex(&self.hash),
            self.object_id,
            self.event.token()
        )
    }
}

/// The `hash_hex` carried by a `le:ext:{gc}:pending` member.
///
/// Members are `{hash_hex}` (legacy, pre-b.qpf) or
/// `{hash_hex}:{object_id}:{event}` ([`ExtRelease::member`]); a `hash_hex` is
/// 64 hex characters and can never contain a `:`, so the prefix up to the first
/// `:` is the hash under both spellings — including b.yda's reconciler repair
/// spelling `{hash_hex}:0:rec{n}`, which is why that addition needed no change
/// here. Every GC script (`ext_release_enqueue.lua`, `gc_claim_pending.lua`,
/// `gc_reclaim_stale.lua`, `gc_decrement_and_collect.lua`,
/// `gc_complete.lua` and the rest of the ledger scripts) is
/// member-opaque — it stores and compares the member string and never parses
/// it; this is the single place the member is interpreted.
pub(crate) fn pending_member_hash_hex(member: &str) -> &str {
    member.split(':').next().unwrap_or(member)
}

/// Strict inverse of [`sha256_to_hex`]: decode exactly 64 lowercase-or-
/// uppercase hex characters into a `[u8; 32]`, or `None`.
///
/// Deliberately strict rather than lenient. Its callers turn a hex string read
/// back OUT of the keyspace (a value row's `hash` column, a `HKEYS
/// le:{oid}:obj:ext` field, a hash returned by `update_layer_fast.lua`) into an
/// [`ExtRelease`], and a silently zero-padded decode of a corrupt string would
/// enqueue a decrement against the WRONG blob. `None` is the fail-closed
/// answer: the caller skips the release, which leaks one reference rather than
/// deleting live data. The snapshot readers' row decoder
/// (`storage::diff_read::decode_field_value`) uses it too and fails the read
/// `Internal` on `None`, rather than hand a reader a hash naming the wrong
/// blob.
pub(crate) fn hash_from_hex(hex: &str) -> Option<[u8; 32]> {
    // The digit check is load-bearing, not redundant with `from_str_radix`:
    // that accepts a leading `+`, so a pair like `"+a"` would decode to 0x0a
    // and a corrupt 64-character string would alias some real hash.
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = u8::from_str_radix(hex.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(out)
}

/// Enqueue a batch of owed decrements onto `le:ext:{gc}:pending` and the
/// release ledger (b.qpf; ledger b.yuo). Every release accounts its blob's
/// full size; see [`enqueue_ext_releases_accounted`] for the commit path's
/// zero-byte `surplus_acquire` releases.
///
/// The plain (non-attempt) enqueue surface, shared by the three commit paths in
/// `storage::layer` and by the reference-count reconciler's repairs. A
/// retirement does not use it: it enqueues through
/// [`ExternalValueStore::enqueue_retirement_attempt`], in attempt mode, so its
/// in-doubt enqueues can be re-driven without a second decrement
/// ([`enqueue_ext_release_attempt`]). `LayerStore` holds no external-value
/// store, and threading one into the materialization entry points would change
/// a signature `src/checkpoint/apply.rs` calls (it calls
/// `LayerStore::write_restored_object`, the DR-restore half of what used to be
/// `write_object`); exposing the enqueue as a free function over
/// `(&ValkeyClient, &KeySchema)` — which every commit path already has — keeps
/// one implementation of the member protocol rather than two.
///
/// Runs on the data pool: the commit paths call it on the request path
/// immediately after their commit lands, so it must not queue behind
/// maintenance-pool sweep traffic.
///
/// b.fd9: transient fred errors (the store's not-serving `-LOADING` / `-BUSY`
/// included) are classified so callers inside a retry loop retry; a store
/// write refusal of the enqueue script's write (`-OOM`, `-MISCONF`, …) is a
/// plain `Unavailable` that no loop retries (b.xd1); every other error is
/// `Internal`.
///
/// A no-op on an empty slice, so callers need no guard.
pub(crate) async fn enqueue_ext_releases(
    client: &ValkeyClient,
    keys: &KeySchema,
    releases: &[ExtRelease],
) -> Result<(), ServiceError> {
    let entries: Vec<(String, bool)> = releases.iter().map(|r| (r.member(), true)).collect();
    enqueue_ext_release_entries(client, keys, &entries).await
}

/// [`enqueue_ext_releases`] with a per-release byte-accounting choice
/// (b.yuo): `true` accounts the blob's full size in the release ledger,
/// `false` accounts 0.
///
/// The commit paths pass `false` for a `surplus_acquire` release: it hands back
/// the pre-flight `+1` of a hash the object still references, so its decrement
/// cannot free the blob, and counting the blob's size against the release
/// budget would hold uploads back for garbage that is not garbage. Every other
/// release — `rewrite`, retirement, a re-send's hand-back, a reconciler repair
/// — accounts the full size, an upper bound on what its decrement can free.
pub(crate) async fn enqueue_ext_releases_accounted(
    client: &ValkeyClient,
    keys: &KeySchema,
    releases: &[(ExtRelease, bool)],
) -> Result<(), ServiceError> {
    let entries: Vec<(String, bool)> = releases
        .iter()
        .map(|(r, accounts_blob)| (r.member(), *accounts_blob))
        .collect();
    enqueue_ext_release_entries(client, keys, &entries).await
}

/// The ARGV pairs `member, bytes` behind every enqueue (b.yuo): read each
/// accounted hash's blob size, then pair every member with its bytes.
///
/// **Bytes.** One `STRLEN le:ext:data:{hash_hex}` per distinct hash with at
/// least one accounted member, on the data pool just before the script (each
/// hash is on its own slot, so they cannot share a script), at most
/// [`RELEASE_SIZE_READS_IN_FLIGHT`] in flight at once on every release path, so
/// the reads' transient memory is a constant, not a term that grows with the
/// reference count (b.fsz). A member whose hash was not accounted, or whose
/// `STRLEN` failed, accounts 0; a failed `STRLEN` logs a WARN and never blocks
/// the enqueue — under-counting the budget is the lesser harm next to losing a
/// release. A blob that is already gone reads 0, correctly: there is nothing
/// left for its decrement to free.
///
/// `entries` is borrowed from the caller and walked twice (the distinct
/// hashes, then the pairs), so no copy of the member list is built before the
/// ARGV itself.
async fn release_entry_argv(
    client: &ValkeyClient,
    keys: &KeySchema,
    entries: ReleaseEntries<'_>,
) -> Vec<String> {
    use futures::stream::{FuturesUnordered, StreamExt};

    // Distinct accounted hashes, in first-occurrence order. The set makes the
    // dedup linear in the member count: an object's member list is unbounded in
    // principle, and on the retirement intent's sends this build is counted
    // against the send window (`enqueue_ext_release_attempt`).
    let mut accounted: Vec<&str> = Vec::new();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for i in 0..entries.len() {
        let (member, accounts_blob) = entries.get(i);
        let hash_hex = pending_member_hash_hex(member);
        if accounts_blob && seen.insert(hash_hex) {
            accounted.push(hash_hex);
        }
    }
    drop(seen);

    // The size reads, at most `RELEASE_SIZE_READS_IN_FLIGHT` at once: the
    // shape of `buffer_unordered`, written out so the reads are plain named
    // futures (a closure building them defeats the `Send` proof of the
    // callers' futures).
    let mut sizes: std::collections::HashMap<&str, u64> =
        std::collections::HashMap::with_capacity(accounted.len());
    let mut to_read = accounted.into_iter();
    let mut in_flight = FuturesUnordered::new();
    loop {
        while in_flight.len() < RELEASE_SIZE_READS_IN_FLIGHT {
            match to_read.next() {
                Some(hash_hex) => in_flight.push(read_release_blob_size(client, keys, hash_hex)),
                None => break,
            }
        }
        match in_flight.next().await {
            Some((hash_hex, bytes)) => {
                sizes.insert(hash_hex, bytes);
            }
            None => break,
        }
    }

    let mut argv: Vec<String> = Vec::with_capacity(entries.len() * 2);
    for i in 0..entries.len() {
        let (member, accounts_blob) = entries.get(i);
        let bytes = if accounts_blob {
            sizes
                .get(pending_member_hash_hex(member))
                .copied()
                .unwrap_or(0)
        } else {
            0
        };
        argv.push(member.to_owned());
        argv.push(bytes.to_string());
    }
    argv
}

/// The members [`release_entry_argv`] pairs with their bytes, borrowed from
/// the caller in place.
#[derive(Clone, Copy)]
enum ReleaseEntries<'e> {
    /// Each member with whether it accounts its blob (the plain enqueue).
    Pairs(&'e [(String, bool)]),
    /// Every member accounts its blob (the attempt-mode enqueue).
    Accounted(&'e [String]),
}

impl<'e> ReleaseEntries<'e> {
    fn len(self) -> usize {
        match self {
            Self::Pairs(pairs) => pairs.len(),
            Self::Accounted(members) => members.len(),
        }
    }

    /// Entry `i`: its member and whether it accounts its blob.
    fn get(self, i: usize) -> (&'e str, bool) {
        match self {
            Self::Pairs(pairs) => (pairs[i].0.as_str(), pairs[i].1),
            Self::Accounted(members) => (members[i].as_str(), true),
        }
    }
}

/// One release's blob-size read: `STRLEN le:ext:data:{hash_hex}` on the data
/// pool, 0 on a failure, which is WARNed and never blocks the enqueue
/// ([`release_entry_argv`]).
async fn read_release_blob_size<'h>(
    client: &ValkeyClient,
    keys: &KeySchema,
    hash_hex: &'h str,
) -> (&'h str, u64) {
    let data_key = keys.ext_data(hash_hex);
    let len: Result<i64, fred::error::Error> = client.pool.strlen(&data_key).await;
    let bytes = match len {
        Ok(n) => u64::try_from(n).unwrap_or(0),
        Err(e) => {
            FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(&e);
            tracing::warn!(
                hash = %hash_hex,
                error = %e,
                "could not read an external value's size for the release ledger; \
                 accounting its release as 0 bytes (the enqueue proceeds)"
            );
            0
        }
    };
    (hash_hex, bytes)
}

/// The `{gc}` keys of every `scripts/ext_release_enqueue.lua` call, in its
/// `KEYS` order; attempt mode appends the attempt record.
fn release_enqueue_keys(keys: &KeySchema) -> Vec<String> {
    vec![
        keys.ext_pending(),
        keys.ext_pending_age(),
        keys.ext_pending_bytes(),
        keys.ext_pending_total(),
        keys.ext_inflight(),
        keys.ext_settled(),
    ]
}

/// Log the members an enqueue refused as replays (b.1jt), at DEBUG: the
/// script found each one out of `pending` but still claimed in `inflight`,
/// settled within the replay window, or holding a ledger field, so it was
/// already enqueued once and its decrement was or is being decided. The
/// expected source is a re-send of an enqueue that already landed.
fn log_refused_replays(refused: i64, context: &'static str) {
    if refused > 0 {
        tracing::debug!(
            refused,
            context,
            "release enqueue refused members as replays of an enqueue that already landed \
             (claimed, settled within the replay window, or still accounted); nothing was \
             written for them"
        );
    }
}

/// The plain enqueue (b.yuo): [`release_entry_argv`], then
/// `scripts/ext_release_enqueue.lua` without attempt mode, which adds every
/// member to `le:ext:{gc}:pending` and, atomically with it, records its
/// first-enqueue age and its bytes and adds those bytes to the ledger total —
/// except a member the script refuses as a replay (b.1jt: not in `pending`,
/// but in `inflight`, in `settled`, or holding a ledger field), for which it
/// writes nothing.
///
/// The script's own failure is the enqueue's failure, classified like the bare
/// `SADD` it replaces: transient fred errors as the in-loop `Aborted`, so
/// callers inside a retry loop retry; permanent ones as `Internal`.
async fn enqueue_ext_release_entries(
    client: &ValkeyClient,
    keys: &KeySchema,
    entries: &[(String, bool)],
) -> Result<(), ServiceError> {
    if entries.is_empty() {
        return Ok(());
    }
    let argv = release_entry_argv(client, keys, ReleaseEntries::Pairs(entries)).await;
    let (_added, refused): (i64, i64) = EXT_RELEASE_ENQUEUE_SCRIPT
        .evalsha_with_reload(client.pool.next(), release_enqueue_keys(keys), argv)
        .await
        .map_err(|e| {
            map_transient_fred_err(
                FredErrOrigin::client(client, POOL_LABEL_DATA),
                "EVALSHA ext_release_enqueue",
                e,
            )
        })?;
    log_refused_replays(refused, "plain enqueue");
    Ok(())
}

/// What an attempt-mode enqueue answered (b.yuo Part R; see
/// `scripts/ext_release_enqueue.lua`, "Attempt mode").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseAttemptOutcome {
    /// This attempt ran now: its members are on `le:ext:{gc}:pending` and its
    /// token is recorded `landed`.
    Enqueued,
    /// This attempt had already run (a re-send): nothing was written again.
    AlreadyLanded,
    /// The orphan sweep's resolve decided this attempt never ran before it
    /// arrived: nothing was written, and a re-drive under a newer token owns
    /// the members.
    Voided,
    /// This call sent nothing under the token: the window check before its
    /// first transmission found the caller's send window
    /// ([`RETIRE_INTENT_SEND_WINDOW`], F3) passed (rule E3). The token stays on
    /// the carrier, not sent by this call, and the orphan sweep resolves it and
    /// settles the recorded releases. A window that passes after any
    /// transmission in the call is never reported this way: that is an
    /// outcome-unknown `Err`.
    NotSent,
}

/// The token an attempt-mode enqueue carries (b.yuo Part R): 32 lowercase hex
/// characters from `rand::random`, the same thread-local CSPRNG the commit
/// token is drawn from, so two attempts never share one.
pub(crate) fn mint_retire_attempt_token() -> String {
    format!("{:032x}", rand::random::<u128>())
}

/// Enqueue a retirement intent's members in ATTEMPT MODE (b.yuo Part R): the
/// same enqueue as [`enqueue_ext_releases`], every member accounting its blob's
/// full size, but checked against and recorded on the `{gc}` attempt record
/// `le:ext:{gc}:enqueue_attempt` under `token` in the same script.
///
/// `token` must already be recorded on the object's
/// `le:{oid}:obj:extret:attempts` — by the retirement's begin or the orphan
/// sweep's re-token — before this is called: that is what lets the drain find
/// and resolve every attempt that may have run. `object_id` is written into the
/// record so the GC loop's attempt audit can find that HASH.
///
/// The two callers: [`ExternalValueStore::enqueue_retirement_attempt`] (the
/// retirement, `storage::layer::schedule_old_object_ext_gc`) and the orphan
/// sweep's re-drive (`storage::orphan_sweep`), which holds formatted members
/// rather than [`ExtRelease`] values and re-enqueues the bytes verbatim.
///
/// Data pool, the same transient classification as [`enqueue_ext_releases`].
/// An `Err` means the outcome is unknown: the script may have run. A reply
/// this build cannot read is an `Internal` error, the same unknown outcome.
///
/// **The send window (F3).** `send_by` is the instant past which the caller
/// may no longer decide a send under `token`: now + [`RETIRE_INTENT_SEND_WINDOW`]
/// taken just before the command that recorded the token. It is checked before
/// every transmission this call makes:
///
/// - **past it before the first transmission** (checked after the blob-size
///   reads, which can themselves stall): nothing is sent, and the answer is
///   [`ReleaseAttemptOutcome::NotSent`];
/// - **past it before the re-send that follows a `NOSCRIPT` reload** (the
///   `SCRIPT LOAD` can stall too): the re-send is not made, and the answer is
///   an outcome-unknown `Err` — `NOSCRIPT` proves only that the copy answering
///   it did not run, and an earlier copy fred sent may have (rule E).
///
/// That bound is what lets the `{gc}` attempt audit remove an in-doubt record
/// after V without a later service send under its token ever finding it gone.
/// (fred's own re-send of an in-flight command after a lost connection is a
/// copy of this call's decision, not a new one; it is the transit residual.)
///
/// This is the only function that transmits under an attempt token (rule H).
///
/// Consults the test-only [`ReleaseEnqueueFaults`] seam on `client`, after the
/// first window check.
pub(crate) async fn enqueue_ext_release_attempt(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    token: &str,
    members: &[String],
    send_by: std::time::Instant,
) -> Result<ReleaseAttemptOutcome, ServiceError> {
    // Every member accounts its blob; read in place, never copied into a
    // second list before the ARGV.
    let pairs = release_entry_argv(client, keys, ReleaseEntries::Accounted(members)).await;
    if std::time::Instant::now() >= send_by {
        return Ok(ReleaseAttemptOutcome::NotSent);
    }
    let mut argv: Vec<String> = Vec::with_capacity(pairs.len() + 3);
    argv.push(EXT_RELEASE_ATTEMPT_MARK.to_owned());
    argv.push(token.to_owned());
    argv.push(object_id.to_owned());
    argv.extend(pairs);
    let mut script_keys = release_enqueue_keys(keys);
    script_keys.push(keys.ext_enqueue_attempts());
    let map = |e: fred::error::Error| {
        map_transient_fred_err(
            FredErrOrigin::client(client, POOL_LABEL_DATA),
            "EVALSHA ext_release_enqueue (attempt)",
            e,
        )
    };

    let conn = client.pool.next();

    // The test seam may answer this call; or stand in for the first
    // transmission's reply — with `NOSCRIPT`, stalling the reload past the
    // window (`NoscriptReloadPastWindow`), or with the real reply of a send
    // it made itself, the helper's own way, before running a test hook
    // (`ExecuteThenRun`). The release shape has none of these.
    #[cfg(any(test, feature = "test-fault-injection"))]
    let (seam_first, seam_stall_reload): (Option<AttemptEnqueueReply>, bool) =
        match client.release_enqueue_faults.take() {
            Some((_, ReleaseEnqueueFault::Fail(e))) => return Err(map(e)),
            Some((_, ReleaseEnqueueFault::ExecuteThenFail(e))) => {
                let _dropped_reply: AttemptEnqueueReply = EXT_RELEASE_ENQUEUE_SCRIPT
                    .evalsha_with_reload(client.pool.next(), script_keys, argv)
                    .await;
                return Err(map(e));
            }
            Some((injector, ReleaseEnqueueFault::Hold(e))) => {
                injector.hold(script_keys, argv);
                return Err(map(e));
            }
            Some((_, ReleaseEnqueueFault::ExecuteThenRun(after))) => {
                // Sent for real, as the helper's own first transmission: the
                // script is loaded first (a `SCRIPT LOAD`, which carries no
                // token) so the plain `EVALSHA` runs on a cold script cache
                // too, which makes this exactly one transmission under the
                // token (rule H). The hook runs once that send's reply or
                // error has come back, whatever it is; the reply then takes
                // the helper's normal handling below, so a `NOSCRIPT` (a cache
                // emptied between the load and the send) goes through the
                // reload branch and its send-window re-check like any other.
                let _ = EXT_RELEASE_ENQUEUE_SCRIPT.load(conn).await;
                let reply: AttemptEnqueueReply = EXT_RELEASE_ENQUEUE_SCRIPT
                    .evalsha(conn, script_keys.clone(), argv.clone())
                    .await;
                after.run().await;
                (Some(reply), false)
            }
            Some((_, ReleaseEnqueueFault::NoscriptReloadPastWindow { first_runs })) => {
                if first_runs {
                    // The copy fred sent to a primary that ran it before a
                    // failover emptied the script cache: sent, reply dropped.
                    // The script is loaded first so the copy runs on a cold
                    // script cache too (a plain `EVALSHA` never reloads); the
                    // `SCRIPT LOAD` carries no token, so this arm still makes
                    // exactly one transmission under it before the stand-in
                    // `NOSCRIPT` (rule H). If the load fails the copy may not
                    // run, and a test that expects it landed sees that.
                    let _ = EXT_RELEASE_ENQUEUE_SCRIPT.load(conn).await;
                    let _dropped_reply: AttemptEnqueueReply = EXT_RELEASE_ENQUEUE_SCRIPT
                        .evalsha(conn, script_keys.clone(), argv.clone())
                        .await;
                }
                // Exactly what fred builds from the server's `NOSCRIPT` reply.
                let noscript = fred::error::Error::new(
                    fred::error::ErrorKind::Unknown,
                    "NOSCRIPT No matching script. Please use EVAL.",
                );
                (Some(Err(noscript)), true)
            }
            None => (None, false),
        };
    // Release builds have no seam; naming the handle keeps the `ValkeyClient`
    // field used in both cfg shapes (the release lint shape would otherwise
    // flag a field only the test shape reads).
    #[cfg(not(any(test, feature = "test-fault-injection")))]
    let _ = &client.release_enqueue_faults;
    #[cfg(not(any(test, feature = "test-fault-injection")))]
    let (seam_first, seam_stall_reload): (Option<AttemptEnqueueReply>, bool) = (None, false);

    // Not `evalsha_with_reload`: on `NOSCRIPT` it sends `SCRIPT LOAD` and then
    // a second `EVALSHA` with no window check, and with the command timeout
    // disabled the load can stall without bound — a re-send decided past
    // `send_by`. So the reload is done here, and the window is checked again
    // before the re-send. `NOSCRIPT` proves only that the copy answering it
    // did not run: fred may have sent an earlier copy of this EVALSHA to a
    // primary that ran it before a failover or restart emptied the script
    // cache. So a window that has passed by then means outcome unknown, never
    // `NotSent` (rule E3); and the re-send, when it is made, is decided by the
    // token's `{gc}` record like any copy (rule E2's one allowed branch).
    let first: AttemptEnqueueReply = match seam_first {
        Some(injected) => injected,
        None => {
            EXT_RELEASE_ENQUEUE_SCRIPT
                .evalsha(conn, script_keys.clone(), argv.clone())
                .await
        }
    };
    let (verdict, _added, refused): (String, i64, i64) = match first {
        Err(e) if e.details().starts_with("NOSCRIPT") => {
            EXT_RELEASE_ENQUEUE_SCRIPT.load(conn).await.map_err(map)?;
            if seam_stall_reload {
                // Test seam: a `SCRIPT LOAD` that stalled past the window.
                tokio::time::sleep_until(tokio::time::Instant::from_std(send_by)).await;
            }
            if std::time::Instant::now() >= send_by {
                return Err(ServiceError::Unavailable(
                    "ext_release_enqueue (attempt): the send window passed while the script was \
                     reloaded after NOSCRIPT, so it was not re-sent; whether an earlier copy of \
                     this attempt ran is unknown"
                        .to_owned(),
                ));
            }
            EXT_RELEASE_ENQUEUE_SCRIPT
                .evalsha(conn, script_keys, argv)
                .await
                .map_err(map)?
        }
        other => other.map_err(map)?,
    };
    log_refused_replays(refused, "attempt-mode enqueue");
    match verdict.as_str() {
        "enqueued" => Ok(ReleaseAttemptOutcome::Enqueued),
        "already" => Ok(ReleaseAttemptOutcome::AlreadyLanded),
        "void" => Ok(ReleaseAttemptOutcome::Voided),
        other => Err(ServiceError::Internal(format!(
            "ext_release_enqueue answered an unknown attempt verdict {other:?}"
        ))),
    }
}

/// The ARGV marker that puts `scripts/ext_release_enqueue.lua` in attempt mode.
const EXT_RELEASE_ATTEMPT_MARK: &str = "@attempt";

/// The raw reply of one attempt-mode `EVALSHA ext_release_enqueue`:
/// `(verdict, added, refused)`, or fred's error.
type AttemptEnqueueReply = Result<(String, i64, i64), fred::error::Error>;

// ── Release enqueue fault seam (test-only) ──────────────────────────────────
//
// Compiled only under `#[cfg(any(test, feature = "test-fault-injection"))]`,
// modelled on `storage::diff_read::ObjectSnapshotReadFaults`. The release
// library has neither the injector nor the branch that consults it:
// `ReleaseEnqueueFaults` degenerates to a zero-sized unit struct, and
// `enqueue_ext_release_attempt` sends its script and nothing else.
//
// **Why a seam is needed.** The retirement intent exists for one shape: an
// attempt-mode enqueue whose outcome the caller cannot know — it ran and its
// reply was lost, or it never ran. Neither is reachable from the keyspace
// alone: a planted wrong-typed key fails the script before it runs, never
// after. The seam answers one attempt with a canned fred error, either
// instead of sending it (`Fail`), after sending it and dropping the reply
// (`ExecuteThenFail`), or instead of sending it while keeping the command for
// the test to send later (`Hold`, then
// [`ReleaseEnqueueFaultInjector::send_held`] — a copy fred writes late), so the
// error takes the real classification and logging path. A fourth answer,
// `NoscriptReloadPastWindow { first_runs }`, drives the helper's REAL
// `NOSCRIPT` branch: optionally the script is loaded (an untokened
// `SCRIPT LOAD`, so the copy runs on a cold script cache) and the first copy
// is sent and its reply dropped, then the first transmission is answered with
// fred's `NOSCRIPT` error, the
// real `SCRIPT LOAD` runs, the call sleeps until `send_by`, and the real second
// window check decides — the only deterministic way to reach that check on a
// shared test server. A fifth, `ExecuteThenRun(hook)`, injects no error: the
// enqueue is sent for real (an untokened `SCRIPT LOAD`, then the plain
// `EVALSHA` — one transmission under the token) and its real reply or error
// takes the helper's normal handling, `NOSCRIPT` reload and send-window
// re-check included, but a test-supplied async hook runs
// in between, so a test can change what the caller's next
// step meets (the drain's `clear` after its own enqueue, which a test cannot
// otherwise reach between the two). The seam covers attempt-mode enqueues
// only — the retirement's and the orphan sweep's re-drive — never the commit
// paths' plain enqueue. The
// first send-window check runs before it, so a `NotSent` attempt consumes no
// queued answer. Production paths are unchanged by every answer.
//
// Per-instance (installed on a `ValkeyClient`, shared behind an `Arc`) rather
// than process-global: a process-global queue is shared-static state that
// leaks across concurrently-running tests.

/// One canned answer for an attempt-mode release enqueue (b.yuo Part R).
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug)]
pub enum ReleaseEnqueueFault {
    /// Do not send the enqueue; answer this error. The attempt never ran.
    Fail(fred::error::Error),
    /// Send the enqueue, drop its reply, and answer this error. The attempt
    /// ran, and the caller cannot know it.
    ExecuteThenFail(fred::error::Error),
    /// Do not send the enqueue yet: keep its keys and arguments on the
    /// injector and answer this error. The attempt has not run when the caller
    /// sees the error; the test sends it later, exactly as built, with
    /// [`ReleaseEnqueueFaultInjector::send_held`] — the copy of a stuck
    /// command that fred writes after its caller has moved on.
    Hold(fred::error::Error),
    /// Drive the helper's real `NOSCRIPT` branch with the reload stalled past
    /// the send window. If `first_runs`, first load the script (a `SCRIPT
    /// LOAD`, which carries no token, so the copy runs on a cold script cache
    /// too), then send the enqueue with the helper's own plain `EVALSHA` and
    /// drop the reply — the copy fred sent to a primary that ran it before a
    /// failover emptied the script cache. Then,
    /// either way, answer the first transmission with the `NOSCRIPT` error
    /// fred builds from the server's reply, run the real `SCRIPT LOAD`, sleep
    /// until `send_by`, and run the real second window check, which answers an
    /// outcome-unknown `Err` without re-sending.
    NoscriptReloadPastWindow { first_runs: bool },
    /// Send the enqueue for real as the helper's own first transmission —
    /// load the script (a `SCRIPT LOAD`, which carries no token, so the send
    /// runs on a cold script cache too), then the plain `EVALSHA`, exactly one
    /// transmission under the token (rule H) — run the hook once after that
    /// send's reply or error came back, then hand the reply to the helper's
    /// normal handling of a first transmission's reply, which maps a verdict
    /// to its [`ReleaseAttemptOutcome`] and an error to the call's `Err`. A
    /// `NOSCRIPT` reply (a script cache emptied between the load and the
    /// send) takes the helper's own reload branch, after the hook, with its
    /// send-window re-check before any re-send, as in production. fred has
    /// already converted the reply to `(verdict, added)` before the hook
    /// runs; only that handling happens after it. For staging what another
    /// actor does in the gap between the send and the call's return — for
    /// example re-tokening the intent's carrier, so the caller's next step
    /// meets a newer token.
    ///
    /// The hook runs whatever the send answered, so it does not prove the
    /// enqueue ran: on a transport error or a `NOSCRIPT` reply it still runs,
    /// before the caller sees the error or the reload branch runs.
    ///
    /// Arm it like any answer, with the hook built by
    /// [`ReleaseEnqueueHook::new`]:
    ///
    /// ```ignore
    /// let tv2 = tv.clone();
    /// faults.push(ReleaseEnqueueFault::ExecuteThenRun(ReleaseEnqueueHook::new(
    ///     move || async move {
    ///         // e.g. re-token the carrier through `tv2`'s client
    ///     },
    /// )));
    /// ```
    ///
    /// The hook is awaited inside the enqueue call, so an enqueue that is
    /// itself awaited by the code under test does not return until the hook
    /// completes; it must not wait on that code's progress.
    ExecuteThenRun(ReleaseEnqueueHook),
}

/// The async hook a [`ReleaseEnqueueFault::ExecuteThenRun`] answer runs once,
/// after its send's reply or error came back and before the enqueue call
/// returns.
#[cfg(any(test, feature = "test-fault-injection"))]
pub struct ReleaseEnqueueHook(
    Box<dyn FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send>,
);

#[cfg(any(test, feature = "test-fault-injection"))]
impl ReleaseEnqueueHook {
    /// Wrap `hook`, an async closure the enqueue call runs and awaits once.
    /// `Send + 'static` because the enqueue may run on any runtime thread,
    /// inside a task the code under test spawned.
    pub fn new<F, Fut>(hook: F) -> Self
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        Self(Box::new(move || Box::pin(hook())))
    }

    /// Run the hook to completion.
    async fn run(self) {
        (self.0)().await;
    }
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl std::fmt::Debug for ReleaseEnqueueHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ReleaseEnqueueHook(..)")
    }
}

/// One attempt-mode enqueue a [`ReleaseEnqueueFault::Hold`] kept back: the
/// script's keys and arguments, exactly as the caller built them.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug)]
struct HeldReleaseEnqueue {
    keys: Vec<String>,
    argv: Vec<String>,
}

/// Queued canned answers for attempt-mode release enqueues (b.yuo Part R).
///
/// Consumed front-to-back, one per attempt-mode enqueue on the client it is
/// installed on, whichever object it is for; an empty queue falls through to
/// the real round trip, so the seam is invisible to every other test.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Default)]
pub struct ReleaseEnqueueFaultInjector {
    queue: std::sync::Mutex<std::collections::VecDeque<ReleaseEnqueueFault>>,
    held: std::sync::Mutex<Vec<HeldReleaseEnqueue>>,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl ReleaseEnqueueFaultInjector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue one answer for the next attempt-mode enqueue.
    pub fn push(&self, fault: ReleaseEnqueueFault) -> &Self {
        if let Ok(mut q) = self.queue.lock() {
            q.push_back(fault);
        }
        self
    }

    /// How many canned answers are still queued. Zero means every subsequent
    /// attempt-mode enqueue takes the real path.
    pub fn remaining(&self) -> usize {
        self.queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// How many [`ReleaseEnqueueFault::Hold`] commands are kept, not yet sent.
    pub fn held(&self) -> usize {
        self.held.lock().map(|h| h.len()).unwrap_or(0)
    }

    /// Send every command a [`ReleaseEnqueueFault::Hold`] kept, in the order
    /// they were held, on `client`'s data pool, and return each raw reply —
    /// `(verdict, added, refused)`, the attempt-mode answer of
    /// `scripts/ext_release_enqueue.lua` (`"enqueued"`, `"already"` or
    /// `"void"`). The held list is emptied first, so a second call sends
    /// nothing. Models a copy of a stuck command that fred writes late.
    pub async fn send_held(
        &self,
        client: &ValkeyClient,
    ) -> Vec<Result<(String, i64, i64), fred::error::Error>> {
        let held: Vec<HeldReleaseEnqueue> = self
            .held
            .lock()
            .map(|mut h| std::mem::take(&mut *h))
            .unwrap_or_default();
        let mut replies = Vec::with_capacity(held.len());
        for command in held {
            replies.push(
                EXT_RELEASE_ENQUEUE_SCRIPT
                    .evalsha_with_reload(client.pool.next(), command.keys, command.argv)
                    .await,
            );
        }
        replies
    }

    fn take(&self) -> Option<ReleaseEnqueueFault> {
        self.queue.lock().ok().and_then(|mut q| q.pop_front())
    }

    fn hold(&self, keys: Vec<String>, argv: Vec<String>) {
        if let Ok(mut h) = self.held.lock() {
            h.push(HeldReleaseEnqueue { keys, argv });
        }
    }
}

/// The handle [`ValkeyClient`] carries.
///
/// Shared behind an `Arc` so an injector installed on one clone is visible to
/// every other clone — the orphan sweep and the retirement cleanup tasks are
/// handed a `ValkeyClient` by value, so an injector installed after that
/// hand-off must still be seen.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Clone, Debug, Default)]
pub struct ReleaseEnqueueFaults {
    slot: Arc<std::sync::Mutex<Option<Arc<ReleaseEnqueueFaultInjector>>>>,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl ReleaseEnqueueFaults {
    /// The inert seam — no injector installed. Named rather than
    /// `Default::default()` because the construction site in `ValkeyClient` is
    /// shared by both cfg shapes, and the release shape's unit struct makes
    /// `::default()` a clippy error (`default_constructed_unit_structs`).
    pub fn none() -> Self {
        Self::default()
    }

    /// Install (or replace) the injector every attempt-mode enqueue on this
    /// client consults.
    pub fn install(&self, injector: Arc<ReleaseEnqueueFaultInjector>) {
        if let Ok(mut slot) = self.slot.lock() {
            *slot = Some(injector);
        }
    }

    /// The next queued answer, with the injector it came from (a `Hold` keeps
    /// its command there).
    fn take(&self) -> Option<(Arc<ReleaseEnqueueFaultInjector>, ReleaseEnqueueFault)> {
        let injector = self.slot.lock().ok()?.clone()?;
        let fault = injector.take()?;
        Some((injector, fault))
    }
}

/// Production stand-in for the fault seam: a zero-sized unit struct, so the
/// field on `ValkeyClient` costs nothing and needs no `#[cfg]` at its
/// construction site.
#[cfg(not(any(test, feature = "test-fault-injection")))]
#[derive(Clone, Debug, Default)]
pub struct ReleaseEnqueueFaults;

#[cfg(not(any(test, feature = "test-fault-injection")))]
impl ReleaseEnqueueFaults {
    /// Shape-independent counterpart of the test-shape constructor, so
    /// `ValkeyClient`'s construction site needs no `#[cfg]`.
    pub fn none() -> Self {
        Self
    }
}

// ── Retirement intent steps (b.yuo Part R) ──────────────────────────────────
//
// Thin typed wrappers over the intent's `{object_id}` and `{gc}` scripts, so
// the retirement (`storage::layer`) and the orphan sweep's drain
// (`storage::orphan_sweep`) share one decoding of every verdict. Each takes
// the connection to send on — the retirement uses the data pool, the drain a
// maintenance connection — and returns fred's error unclassified, because the
// two callers count and log failures differently. A verdict this build does
// not know comes back as a `Parse` error: the caller then does nothing it
// cannot undo, which is the safe reading of an unknown answer.

fn unknown_verdict(script: &str, verdict: &str) -> fred::error::Error {
    fred::error::Error::new(
        fred::error::ErrorKind::Parse,
        format!("{script} answered an unknown verdict {verdict:?}"),
    )
}

/// What `scripts/ext_retire_begin.lua` answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetireBegin {
    /// The intent is recorded (or this call's own earlier run recorded it):
    /// the flag is set, and with members the carrier and the token are too.
    Begun,
    /// `ext_gc_scheduled` was already set by an earlier retirement of this
    /// incarnation, which owns the obligation; nothing was written.
    Already,
    /// The flag is not set, but a carrier or a token HASH is already there;
    /// nothing was written. Either a carrier from an earlier lifecycle of this
    /// id that the orphan sweep settles on its own, or a token HASH with NO
    /// carrier — only an older build leaves one — which may belong to this
    /// same incarnation and is the only evidence that its obligation may have
    /// been paid. Nothing in this build removes that one (b.9ud sub-task 3
    /// plans the only safe remover, the restore's residue wipe extended to
    /// void the tokens and delete the HASH — not built), so every retirement of
    /// the id answers `Busy`.
    Busy,
    /// The header is gone; nothing was written.
    Gone,
}

/// Begin a retirement intent: `scripts/ext_retire_begin.lua` with `token` and
/// the owed `members` (may be empty).
pub(crate) async fn retire_begin(
    conn: &fred::clients::Client,
    keys: &KeySchema,
    object_id: &str,
    token: &str,
    members: &[String],
) -> Result<RetireBegin, fred::error::Error> {
    let mut argv: Vec<String> = Vec::with_capacity(members.len() + 1);
    argv.push(token.to_owned());
    argv.extend(members.iter().cloned());
    let verdict: String = EXT_RETIRE_BEGIN_SCRIPT
        .evalsha_with_reload(
            conn,
            vec![
                keys.obj_header(object_id),
                keys.obj_ext_retire_carrier(object_id),
                keys.obj_ext_retire_attempts(object_id),
            ],
            argv,
        )
        .await?;
    match verdict.as_str() {
        "begun" => Ok(RetireBegin::Begun),
        "already" => Ok(RetireBegin::Already),
        "busy" => Ok(RetireBegin::Busy),
        "gone" => Ok(RetireBegin::Gone),
        other => Err(unknown_verdict("ext_retire_begin", other)),
    }
}

/// What `scripts/ext_retire_retry_claim.lua` (the re-token) answered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RetireRetoken {
    /// The carrier is empty or absent; nothing was written. A token HASH
    /// beside no carrier is left in place (nothing in this build removes it;
    /// b.9ud sub-task 3 plans the remover, not built).
    Empty,
    /// The newest token is younger than the in-progress age; nothing written.
    InProgress,
    /// The new token is recorded on this build's HASH carrier: the carrier's
    /// members, and the tokens that were recorded before it.
    Claimed {
        members: Vec<String>,
        prior: Vec<String>,
    },
    /// The new token is recorded on a SET carrier — an older build's
    /// (b.1jt): the carrier's members, and the tokens that were recorded
    /// before it (none for an untokened one). Such a carrier is cleared if
    /// one of its recorded attempts landed and otherwise dropped, never paid:
    /// no recorded token is not proof that nothing ran (rule E4).
    Legacy {
        members: Vec<String>,
        prior: Vec<String>,
    },
}

/// Re-token a carrier: `scripts/ext_retire_retry_claim.lua` with `new_token`
/// and the in-progress age.
pub(crate) async fn retire_retoken(
    conn: &fred::clients::Client,
    keys: &KeySchema,
    object_id: &str,
    new_token: &str,
    in_progress: std::time::Duration,
) -> Result<RetireRetoken, fred::error::Error> {
    let (verdict, members, prior, kind): (String, Vec<String>, Vec<String>, String) =
        EXT_RETIRE_RETRY_CLAIM_SCRIPT
            .evalsha_with_reload(
                conn,
                vec![
                    keys.obj_ext_retire_carrier(object_id),
                    keys.obj_ext_retire_attempts(object_id),
                ],
                vec![new_token.to_owned(), duration_ms(in_progress).to_string()],
            )
            .await?;
    match (verdict.as_str(), kind.as_str()) {
        ("empty", _) => Ok(RetireRetoken::Empty),
        ("in_progress", _) => Ok(RetireRetoken::InProgress),
        ("claimed", "hash") => Ok(RetireRetoken::Claimed { members, prior }),
        ("claimed", "set") => Ok(RetireRetoken::Legacy { members, prior }),
        ("claimed", other) => Err(unknown_verdict(
            "ext_retire_retry_claim carrier kind",
            other,
        )),
        (other, _) => Err(unknown_verdict("ext_retire_retry_claim", other)),
    }
}

/// What one token resolved to on the `{gc}` attempt record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttemptVerdict {
    /// The attempt ran before the resolve.
    Landed,
    /// The attempt had not run, and now never will.
    Void,
}

/// Resolve `tokens` (at least one) on `le:ext:{gc}:enqueue_attempt`:
/// `scripts/ext_enqueue_attempt_resolve.lua`. One verdict per token, in order.
/// A recorded value this build cannot read is a `Parse` error.
pub(crate) async fn resolve_enqueue_attempts(
    conn: &fred::clients::Client,
    keys: &KeySchema,
    object_id: &str,
    tokens: &[String],
) -> Result<Vec<AttemptVerdict>, fred::error::Error> {
    let mut argv: Vec<String> = Vec::with_capacity(tokens.len() + 1);
    argv.push(object_id.to_owned());
    argv.extend(tokens.iter().cloned());
    let verdicts: Vec<String> = EXT_ENQUEUE_ATTEMPT_RESOLVE_SCRIPT
        .evalsha_with_reload(conn, vec![keys.ext_enqueue_attempts()], argv)
        .await?;
    if verdicts.len() != tokens.len() {
        return Err(fred::error::Error::new(
            fred::error::ErrorKind::Parse,
            format!(
                "ext_enqueue_attempt_resolve answered {} verdicts for {} tokens",
                verdicts.len(),
                tokens.len()
            ),
        ));
    }
    verdicts
        .iter()
        .map(|v| match v.as_str() {
            "landed" => Ok(AttemptVerdict::Landed),
            "void" => Ok(AttemptVerdict::Void),
            other => Err(unknown_verdict("ext_enqueue_attempt_resolve", other)),
        })
        .collect()
}

/// What `scripts/ext_retire_settle.lua` answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetireSettle {
    /// The token set moved since the re-token, or the carrier is gone while
    /// the tokens are still recorded (an older build's claim took it);
    /// nothing written.
    Changed,
    /// The id is live: carrier and tokens deleted, the flag cleared; the
    /// number of members dropped.
    Cancelled(u64),
    /// The id is dead: the members are owed, nothing written.
    Pay,
}

/// Settle a carrier whose prior tokens all resolved void:
/// `scripts/ext_retire_settle.lua` with the re-token's `new_token` and `prior`.
pub(crate) async fn retire_settle(
    conn: &fred::clients::Client,
    keys: &KeySchema,
    object_id: &str,
    new_token: &str,
    prior: &[String],
) -> Result<RetireSettle, fred::error::Error> {
    let mut argv: Vec<String> = Vec::with_capacity(prior.len() + 1);
    argv.push(new_token.to_owned());
    argv.extend(prior.iter().cloned());
    let (verdict, count): (String, i64) = EXT_RETIRE_SETTLE_SCRIPT
        .evalsha_with_reload(
            conn,
            vec![
                keys.obj_ext_retire_carrier(object_id),
                keys.obj_ext_retire_attempts(object_id),
                keys.obj_header(object_id),
                keys.obj_ext_index(object_id),
            ],
            argv,
        )
        .await?;
    match verdict.as_str() {
        "changed" => Ok(RetireSettle::Changed),
        "cancelled" => Ok(RetireSettle::Cancelled(u64::try_from(count).unwrap_or(0))),
        "pay" => Ok(RetireSettle::Pay),
        other => Err(unknown_verdict("ext_retire_settle", other)),
    }
}

/// What `scripts/ext_retire_clear.lua` answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetireClear {
    /// Carrier and tokens deleted.
    Cleared,
    /// A drain re-tokened the carrier since this token was recorded; nothing
    /// written, the drain owns it.
    Superseded,
    /// The token or the carrier is already gone — removed by a drain, by an
    /// older build's claim, or by an earlier copy of this same clear (rule E1:
    /// the verdict does not say who). A token HASH left with no carrier stays
    /// in place (nothing in this build removes it; b.9ud sub-task 3 plans the
    /// remover, not built). Nothing written.
    Gone,
}

/// Clear a carrier one of whose attempts landed:
/// `scripts/ext_retire_clear.lua` with that attempt's `token` and the tokens
/// the caller knows were recorded `older` than it (none for a retirement's own
/// begin token; the re-token's `prior` for a drain's).
///
/// The orphan sweep's drain also uses it, under its own new token, to DROP an
/// older build's SET carrier none of whose recorded attempts landed (b.1jt):
/// the same deletion of the carrier and its tokens, under the same "no newer
/// token" check, with nothing enqueued and `ext_gc_scheduled` untouched.
pub(crate) async fn retire_clear(
    conn: &fred::clients::Client,
    keys: &KeySchema,
    object_id: &str,
    token: &str,
    older: &[String],
) -> Result<RetireClear, fred::error::Error> {
    let mut argv: Vec<String> = Vec::with_capacity(older.len() + 1);
    argv.push(token.to_owned());
    argv.extend(older.iter().cloned());
    let verdict: String = EXT_RETIRE_CLEAR_SCRIPT
        .evalsha_with_reload(
            conn,
            vec![
                keys.obj_ext_retire_carrier(object_id),
                keys.obj_ext_retire_attempts(object_id),
            ],
            argv,
        )
        .await?;
    match verdict.as_str() {
        "cleared" => Ok(RetireClear::Cleared),
        "superseded" => Ok(RetireClear::Superseded),
        "gone" => Ok(RetireClear::Gone),
        other => Err(unknown_verdict("ext_retire_clear", other)),
    }
}

/// Remove the `{gc}` records of `tokens` from `le:ext:{gc}:enqueue_attempt`
/// with one `HDEL` (b.yuo Part R). The ONLY eager removal of a record, and it
/// is F1's: call it only as a token's own sender, only when that sender's
/// single send under the token answered `enqueued` or `already` with no
/// earlier attempt under the token having errored, and only after its clear
/// then answered `cleared`. Then no carrier names the token and no service
/// send under it remains. A record is also the only guard against a SEND
/// under its token — a send that finds it absent enqueues again — so any
/// other caller (a drain that found an earlier token landed, a retirement
/// whose first attempt errored) must leave the record to the GC loop's
/// attempt audit, which removes it after [`ENQUEUE_ATTEMPT_VOID_RETENTION`]
/// (F2). Best effort: a record left behind is removed by the audit too.
pub(crate) async fn forget_landed_attempts(
    conn: &fred::clients::Client,
    keys: &KeySchema,
    tokens: &[String],
) -> Result<(), fred::error::Error> {
    if tokens.is_empty() {
        return Ok(());
    }
    let _removed: i64 = conn
        .hdel(keys.ext_enqueue_attempts(), tokens.to_vec())
        .await?;
    Ok(())
}

/// Result of one GC pass ([`ExternalValueStore::process_pending`]).
#[derive(Debug, Clone, Default)]
pub struct GcBatchResult {
    /// Members this pass claimed: moved from pending to inflight, or — this
    /// build's own stale claims — re-claimed in place in inflight
    /// ([`Self::reclaimed`] counts those too).
    pub processed: usize,
    /// Number of external values actually deleted (refcount reached zero),
    /// counted when the decrement's reply reports the delete, whether or not
    /// the member's completion then lands. A lower bound, not an exact count:
    /// a delete whose decrement reply was lost (the member's re-claim then
    /// answers `replayed`) is not counted, and a pass that ends `Err` returns
    /// no counts.
    pub deleted: usize,
    /// The bytes the deletes in [`Self::deleted`] freed: the summed lengths
    /// of the deleted external values, as the decrement read them just before
    /// its `DEL` (b.pa1). A lower bound for the same reasons.
    pub freed_bytes: u64,
    /// Claimed members whose decrement the store refused `-OOM` (b.pa1): the
    /// store did not run the decrement past its opening write on a full
    /// store, so it could free nothing there. Each stays in `inflight` for a
    /// later pass to re-claim. Zero wherever the store honours the opener.
    pub oom_refused: usize,
    /// `{gc}` steps this pass skipped because the store refused them `-OOM`
    /// (b.pa1): the stamp of the sampled un-aged members (one step), and each
    /// re-claim of one of this build's own stale claims. Each is retried by a
    /// later pass.
    pub oom_skipped: usize,
    /// Claimed members whose decrement had already applied, answered
    /// `replayed` without decrementing.
    pub replayed: usize,
    /// Claimed members whose decrement was refused `stale` by the claim-age
    /// fence; they stay in inflight, to be re-claimed in place.
    pub stale: usize,
    /// Settled members whose replay records this pass tore down.
    pub torn_down: usize,
    /// This build's own inflight claims, older than the re-claim age, that
    /// this pass re-claimed in place (b.1jt); included in [`Self::processed`].
    pub reclaimed: usize,
    /// An older build's inflight claims, older than the decrement fence, that
    /// this pass dropped without decrementing (b.1jt).
    pub foreign_dropped: usize,
    /// Un-aged pending members — an older build's bare `SADD` — that this pass
    /// stamped with an age (b.1jt).
    pub stamped: usize,
    /// Members the pass's release-ledger audit page settled.
    pub audit_repaired: usize,
    /// `{gc}` enqueue-attempt records the pass's attempt-audit page removed
    /// (b.yuo Part R): `void` records older than
    /// [`ENQUEUE_ATTEMPT_VOID_RETENTION`], and `landed` records older than the
    /// same retention that no carrier holds any more.
    pub attempt_records_retired: usize,
    /// The pass made progress on a full batch — it selected a full batch of
    /// pending members, or of its own stale claims, and collected at least one
    /// of them; tore down a full batch of replay records; or dropped a full
    /// batch of an older build's claims — so the loop runs the next pass
    /// without sleeping. Never set by work that was only selected: a pass
    /// whose decrements or unmarks all fail sleeps like an idle one.
    pub more: bool,
    /// The release ledger as the pass's selection read it; `None` from a store
    /// with no ledger (test doubles).
    pub ledger: Option<GcLedgerSnapshot>,
}

/// The release ledger's levels as one GC pass read them (b.yuo), for the
/// leader-published gauges and the loop's sleep.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GcLedgerSnapshot {
    /// `le:ext:{gc}:pending:total`: the bytes accounted over pending ∪
    /// inflight.
    pub pending_bytes: u64,
    /// Members in pending plus members in inflight: a member an older build
    /// re-enqueued (a bare `SADD`) while its claim is held is in both and
    /// counted twice, transiently.
    pub pending_count: u64,
    /// The age of the oldest aged pending member; `None` when there is none.
    pub oldest_age: Option<std::time::Duration>,
    /// Entries on the settle carrier `le:ext:{gc}:settled`.
    pub settled_count: u64,
    /// Live read hold fields on `le:ext:{gc}:read_hold`: catch-ups re-reading
    /// after an external-value miss, each stopping the collection of newer
    /// releases while it lives.
    pub read_holds: u64,
    /// How long until the oldest pending member falls due under the pass's
    /// age floor; `None` when none is waiting to fall due.
    pub next_due_in: Option<std::time::Duration>,
}

/// State a GC loop carries from one pass to the next (b.yuo): the
/// release-ledger audit's cursor and cycle schedule, the enqueue-attempt
/// audit's cursor, and the out-of-memory collection's rate limit (b.pa1). A
/// new leadership starts from a fresh state, so its first pass starts an audit
/// cycle at cursor 0, and it may start an out-of-memory collection at once.
#[derive(Debug, Default)]
pub struct GcPassState {
    /// `Some` while an audit cycle runs: the `HSCAN` cursor of its next page.
    audit_cursor: Option<String>,
    /// `Some` while the enqueue-attempt audit (b.yuo Part R) has pages left in
    /// the current cycle: the `HSCAN` cursor of its next page. Started with the
    /// release-ledger audit's cycle, it walks on its own cursor and ends only
    /// when a page returns "0".
    attempt_audit_cursor: Option<String>,
    /// When the next audit cycle may start; `None` until the first one has.
    next_audit_cycle: Option<tokio::time::Instant>,
    /// The loop's progress heartbeat (b.1jt), marked after every step and
    /// every member of a pass (and every enqueue-attempt audit `HEXISTS`), so
    /// a long pass that keeps making progress keeps the lease refresh
    /// ticker renewing ([`GC_PROGRESS_STALL_AFTER`]). `None` outside
    /// [`spawn_gc_task`]'s loop: nothing then reads it.
    progress: Option<GcProgress>,
    /// The out-of-memory collection's rate limit and log gates (b.pa1). Reset
    /// with the rest of the state at every new leadership, so a new leader may
    /// start a collection at once.
    oom: OomCollectionState,
    /// The identity the out-of-memory collection's log line names
    /// (`replica_id`, `lock_value`). `None` outside [`spawn_gc_task`]'s loop.
    leader: Option<LeaderIdentity>,
}

/// What the leader carries from one out-of-memory collection to the next
/// (b.pa1), inside [`GcPassState`].
#[derive(Debug)]
struct OomCollectionState {
    /// When the last collection ended, and whether the next may follow it
    /// soon — it freed something, or ended `failed` (a pass error, which is
    /// no evidence of futility); `None` before the first.
    last: Option<(tokio::time::Instant, bool)>,
    /// Spaces the per-collection WARN ([`OOM_COLLECTION_LOG_INTERVAL`]).
    log_gate: LogGate,
    /// Collections whose WARN the gate held back since the last one logged.
    suppressed: u64,
    /// Spaces the WARN of a pass skipping a `{gc}` step refused `-OOM`
    /// ([`GC_OOM_SKIP_LOG_INTERVAL`]).
    skip_log_gate: LogGate,
}

impl Default for OomCollectionState {
    fn default() -> Self {
        Self {
            last: None,
            log_gate: LogGate::new(OOM_COLLECTION_LOG_INTERVAL),
            suppressed: 0,
            skip_log_gate: LogGate::new(GC_OOM_SKIP_LOG_INTERVAL),
        }
    }
}

impl OomCollectionState {
    /// Whether a collection requested now may start: none ran before, or the
    /// last one ended at least [`OOM_COLLECTION_RETRY_AFTER_FREED`] ago and
    /// freed something or ended `failed`, or at least
    /// [`OOM_COLLECTION_RETRY_AFTER_FUTILE`] ago.
    fn may_start(&self, now: tokio::time::Instant) -> bool {
        match self.last {
            None => true,
            Some((ended, retry_soon)) => {
                let since = now.saturating_duration_since(ended);
                since
                    >= if retry_soon {
                        OOM_COLLECTION_RETRY_AFTER_FREED
                    } else {
                        OOM_COLLECTION_RETRY_AFTER_FUTILE
                    }
            }
        }
    }
}

impl GcPassState {
    /// A fresh state: the next pass starts an audit cycle.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark progress on the loop's heartbeat, if it has one.
    fn mark_progress(&self) {
        if let Some(progress) = &self.progress {
            progress.mark();
        }
    }
}

// ── Trait ───────────────────────────────────────────────────────────────────

/// What [`ExternalValueStore::probe_values`] found (b.yuo).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ExtValueProbe {
    /// The bytes of every probed hash that is stored.
    pub found: std::collections::HashMap<[u8; 32], Vec<u8>>,
    /// Every probed hash that is not stored, in the order it was asked for.
    pub missing: Vec<[u8; 32]>,
}

/// Async trait for external value storage with dedup, refcounting, and GC.
///
/// Implementations are expected to be wrapped in `Arc<dyn ExternalValueStore>`.
///
/// # Blob readers
///
/// The methods that return blob bytes ([`Self::load_values`],
/// [`Self::read_batch`], [`Self::read_batch_with_recovery_counter`],
/// [`Self::probe_values`]) have one production reader, the catch-up build
/// (`diff_read::resolve_external_values`). **A new reader of blob bytes must
/// follow the GC miss protocol** (probe; on a miss take a `{gc}` read hold
/// ([`ReadHold`]), re-read the layer and rebuild; SDD §11.2, §11.3): the GC
/// may collect a released blob 5 s after its release, so a blob a reader's
/// snapshot references can be gone, and a reader without the protocol can
/// hand a client a delta whose blob is missing. Every production call site of
/// these methods, and of the key builder `KeySchema::ext_data`, is allowlisted
/// with the reason it is safe by the guard test
/// `storage::ext_blob_reader_guard_test::every_external_value_blob_read_site_under_src_is_allowlisted`
/// (`tests/integration/storage/ext_blob_reader_guard_test.rs`); a new method
/// returning bytes joins its scan.
#[async_trait]
pub trait ExternalValueStore: Send + Sync {
    /// Store external value(s). For each `(hash, data)` pair:
    /// - If data is already stored under the hash, leave it exactly as it is
    ///   (set-if-absent: this method never overwrites an existing
    ///   `ext_data:{hash_hex}` key, of any type) and increment the reference
    ///   count.
    /// - If not, store the data under the content-addressed key
    ///   `ext_data:{hash_hex}` and set ref_count = 1.
    ///
    /// **Precondition: the caller has verified the bytes.** This method does
    /// not hash `data` and trusts that `hash` is its content hash
    /// ([`ext_content_hash`]). The live CreateLayer / UpdateLayer path meets
    /// this through [`store_external_values`] (on UpdateLayer, its two halves
    /// [`prepare_external_values`] and [`PreparedExternalValues::store`]),
    /// which in `enforce` mode refuses
    /// any part whose length or content hash does not match its reference
    /// before storing (and then stores through [`Self::store_verified_values`]
    /// instead); the restore and recovery callers pass bytes a verified upload
    /// stored, or that predate the check (SDD §11.1). With
    /// `LIVE_EDIT_EXT_VALUE_VERIFY=observe` the live path deliberately calls
    /// here with bytes that failed the check (counted and logged). Because the
    /// write is set-if-absent, the first copy stored under a hash is the one
    /// every reader sees, so an unverified store fixes wrong bytes in place
    /// for every layer that shares the hash; a later verified store replaces
    /// it only if its type or length differs (see
    /// [`Self::store_verified_values`]).
    ///
    /// Each hash is written by a single `EVALSHA` of
    /// `scripts/ext_store_value.lua` (meta + data share a Cluster slot via the
    /// `{hash_hex}` hash tag, and Valkey runs Lua atomically), so the meta hash
    /// and the data blob always appear atomically together. The same script
    /// stamps `last_store_ms` on the meta from Valkey's own clock — the
    /// pre-flight timestamp `storage::ext_reconcile` reads.
    ///
    /// SDD §11.1.
    async fn store_values(&self, values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError>;

    /// [`Self::store_values`] for bytes the caller has VERIFIED against their
    /// hash, plus one heal. When a pair's data key already exists and is not
    /// a string, or is a string whose length differs from `data`'s, it is
    /// replaced with `data`. A stored string of the same length is left as it
    /// is: there is no byte compare. Everything else — set-if-absent on an
    /// absent key, the `+1` on every pair, the `last_store_ms` stamp, the
    /// mirror enqueue — is exactly [`Self::store_values`].
    ///
    /// Verified bytes are the content of the hash, so the heal can only
    /// replace damaged bytes with correct ones. It repairs a blob stored
    /// corrupt before the content check existed in the shape that bug left (a
    /// truncated prefix under the full value's hash) and a wrong-typed data
    /// key. The checkpoint adapter's mirror copy is not healed: the adapter
    /// keeps the first copy it was given.
    ///
    /// **Precondition: every pair passed the content check.** Its `hash` is
    /// exactly the 32-byte claim and [`ext_content_hash`] of its `data` equals
    /// it. The only caller is [`PreparedExternalValues::store`] (the store
    /// step of [`store_external_values`]) in `enforce` mode, after
    /// [`verify_external_values`] accepted every reference. Calling this with
    /// unchecked bytes would let them replace a good blob of a different
    /// length, which is the corruption the set-if-absent rule exists to
    /// prevent. The `observe` mode and every restore and recovery path call
    /// [`Self::store_values`], and so never heal.
    ///
    /// Required, with no default, so every implementation states what it does
    /// here: a store heals as above, a wrapping implementation delegates to
    /// its inner store's `store_verified_values` (a wrapper that fell back to
    /// `store_values` would silently stop the live path healing), and a test
    /// double records or fails the call as its `store_values` does.
    ///
    /// SDD §11.1.
    async fn store_verified_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), ServiceError>;

    /// The upload content-check policy [`store_external_values`] (its
    /// [`prepare_external_values`] half) applies before it stores: the `LIVE_EDIT_EXT_VALUE_VERIFY` mode (which also
    /// picks [`Self::store_verified_values`] under `enforce` or
    /// [`Self::store_values`] under `observe`), and the metrics handle
    /// `live_edit_ext_value_verify_total` is counted on.
    ///
    /// Carried by the store because the store is what both upload handlers
    /// already hand to the pre-store ([`store_external_values`] or
    /// [`prepare_external_values`]), and it is where startup
    /// wiring already delivers configuration and metrics to storage
    /// (`with_metrics`, `with_checkpoint_store`).
    ///
    /// Required, with no default, so every implementation states its policy
    /// and a wrapper cannot silently drop `observe` mode or the counter: a
    /// store returns its configured mode and metrics handle, a wrapping
    /// implementation delegates to its inner store, and a test double with no
    /// policy of its own returns [`ExtValueVerify::ENFORCE_UNCOUNTED`] — enforce,
    /// counted nowhere, the production default's refusal behaviour.
    fn ext_value_verify(&self) -> ExtValueVerify<'_>;

    /// Load external value data by hash(es). Returns `(hash, data)` pairs.
    ///
    /// Missing hashes are silently skipped — callers should check the count
    /// if they need strict presence guarantees.
    ///
    /// SDD §11.2 — single round trip: `GET ext_data:{hash_hex}` is
    /// pipelined over all requested hashes (data keys are content-addressed,
    /// so no meta lookup is needed).
    ///
    /// A new caller must follow the GC miss protocol: see the trait's *Blob
    /// readers*.
    async fn load_values(
        &self,
        hashes: &[&[u8; 32]],
    ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError>;

    /// Batch-read external values by hash, returning a map of hash → data.
    ///
    /// Unlike `load_values`, this method returns an error if any hash is
    /// missing (`MetaNotFound` or `DataNotFound`), because missing data
    /// during diff reconstruction indicates a data integrity issue — once a
    /// stale read is ruled out: a catch-up build probes first
    /// ([`Self::probe_values`]) and reads through this only for a hash still
    /// missing after it re-read the layer under the `{gc}` read hold (b.yuo).
    ///
    /// SDD §11.2 — pipelined: for every hash, `GET ext_data` and
    /// `HEXISTS ext_meta ref_count` go in one pipeline, one round trip per
    /// hash, with the hashes read concurrently. The `HEXISTS` answer is used
    /// only for a hash whose data is missing, to tell "unknown hash" from
    /// "dangling meta". Empty input returns an empty map immediately (no
    /// Valkey calls).
    ///
    /// A new caller must follow the GC miss protocol: see the trait's *Blob
    /// readers*.
    async fn read_batch(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError>;

    /// Variant of [`Self::read_batch`] that additionally accepts an
    /// optional `&AtomicUsize` recovery counter. When the SR-3.2 fallback
    /// successfully recovers a missing blob from the configured
    /// `CheckpointStore`, the counter is incremented once per recovered
    /// hash. Other outcomes (hot-path hit, `adapter_miss`,
    /// `adapter_transient`, `adapter_permanent`, `adapter_other`) leave
    /// the counter untouched — the count reflects exactly the number of
    /// hashes whose data was reconstructed via the adapter during this
    /// call.
    ///
    /// Used by the version-diff builds behind
    /// [`crate::storage::diff_read::compute_diff_at_cut`] and
    /// [`crate::storage::diff_read::compute_per_commit_diff_at_cut`] (Subtask
    /// `t3.fgj.zc.p6.78`) so an operator-readable INFO log line can
    /// surface `(object_id, recovered_count)` together — `object_id` is
    /// not in scope inside this method, so the count travels up to the
    /// caller via the shared counter.
    ///
    /// Default implementation delegates to [`Self::read_batch`] with no
    /// counter participation, so the previous contract is preserved for
    /// any impl that does not override.
    ///
    /// A new caller must follow the GC miss protocol: see the trait's *Blob
    /// readers*.
    async fn read_batch_with_recovery_counter(
        &self,
        hashes: &[[u8; 32]],
        _recovery_counter: Option<&std::sync::atomic::AtomicUsize>,
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        self.read_batch(hashes).await
    }

    /// Probe read of external values by hash, with no recovery (b.yuo, the
    /// catch-up miss protocol; SDD §11.2): the bytes of every hash that is
    /// stored, and the hashes that are not.
    ///
    /// Unlike [`Self::read_batch`], a missing hash is an answer here, not an
    /// error: nothing is logged and the checkpoint adapter is never consulted,
    /// so no blob is fetched or re-stored. A catch-up build reads through this
    /// first, because a blob can be missing for a reason that is no damage at
    /// all: a commit after the catch-up's snapshot released it and the GC
    /// collected it. Such a catch-up re-reads its state under the `{gc}` read
    /// hold ([`ReadHold`]) and builds again; only a hash still
    /// missing then is genuinely missing, and [`Self::read_batch`]
    /// (fetch-on-miss through the adapter, then `INTERNAL`) serves it.
    ///
    /// [`ExtValueProbe::missing`] keeps the order of `hashes`. A failed read
    /// is an `Err`, classified as the store's other reads classify it. Empty
    /// input answers an empty probe with no store call.
    ///
    /// Required, with no default, so every implementation states it: a store
    /// answers from its keyspace, a wrapping implementation delegates to its
    /// inner store, and a test double answers from what it holds (or fails as
    /// its other reads fail).
    ///
    /// The probe is the protocol's first step, not the whole of it: a new
    /// caller must follow the GC miss protocol, see the trait's *Blob readers*.
    async fn probe_values(&self, hashes: &[[u8; 32]]) -> Result<ExtValueProbe, ServiceError>;

    /// Enqueue a retiring object's owed `ref_count` decrements onto the pending
    /// release set for background GC, as ONE ATTEMPT of its retirement intent
    /// (b.yuo Part R).
    ///
    /// Each [`ExtRelease`] names the hash, the object giving up its reference,
    /// and the event that owes the decrement, so two distinct owed decrements
    /// for one hash cannot collapse into one pending member while a *replay* of
    /// the same event still does (b.qpf — see [`ExtRelease`]).
    ///
    /// `attempt` is the attempt token the caller has ALREADY recorded on the
    /// object's `le:{oid}:obj:extret:attempts` (with the members on its carrier)
    /// through `scripts/ext_retire_begin.lua`. The enqueue is checked against
    /// and recorded on the `{gc}` attempt record under that token in the same
    /// step ([`enqueue_ext_release_attempt`]), so an attempt whose reply is
    /// lost can later be resolved — landed, or never to land — before anything
    /// enqueues the members again. Calling this twice with one token enqueues at
    /// most once ([`ReleaseAttemptOutcome::AlreadyLanded`]) while the token's
    /// record stands; an `Err` means the outcome is unknown.
    ///
    /// `send_by` is the caller's send window (F3): now +
    /// [`RETIRE_INTENT_SEND_WINDOW`], taken just before the begin that recorded
    /// `attempt`. A send decided at or after it is not made. Past it before
    /// the call's first transmission, the answer is
    /// [`ReleaseAttemptOutcome::NotSent`]; past it before a later transmission
    /// in the same call (the re-send after a `NOSCRIPT` reload), the answer is
    /// an outcome-unknown `Err`, because an earlier copy may have run (rule E3
    /// in [`enqueue_ext_release_attempt`]). Every implementation that sends
    /// (or models sending) must honour it, and a wrapping double passes it
    /// through.
    ///
    /// Replaces the plain `schedule_release`, whose only production caller was
    /// the retirement (`storage::layer::schedule_old_object_ext_gc`).
    ///
    /// SDD §11.3, §11.3.1.
    async fn enqueue_retirement_attempt(
        &self,
        object_id: &str,
        attempt: &str,
        releases: &[ExtRelease],
        send_by: std::time::Instant,
    ) -> Result<ReleaseAttemptOutcome, ServiceError>;

    /// Run one GC pass over the pending release set under `policy` (b.yuo).
    ///
    /// On a store with a release ledger, in order: one read-only selection of
    /// the ledger's levels and of up to a batch of members, oldest first, no
    /// younger than `policy`'s age floor (while the ledger total is at or above
    /// the pressure threshold, also younger ones down to the pressure floor,
    /// each added only while the bytes of every member selected so far, those
    /// past the age floor included, do not exceed the excess over the
    /// threshold) and none
    /// enqueued after a live read hold began, plus a sample of un-aged
    /// members and the stale claims in `inflight`; an age stamp on each
    /// sampled un-aged member (an older build's bare `SADD`); a drop, with no
    /// decrement, of every older build's claim older than the decrement fence;
    /// teardown of the replay
    /// records of members settled at least `policy.replay_record_retention`
    /// ago; then, for each selected member — and each of this build's own
    /// claims older than `policy.reclaim_after` — at most
    /// `GC_MEMBER_CONCURRENCY` at a time: claim
    /// it (`gc_claim_pending.lua`; an own stale claim is re-claimed in place
    /// by `gc_reclaim_stale.lua`), apply its decrement at most once
    /// (`gc_decrement_and_collect.lua`: decrement, delete at zero, or answer
    /// `replayed` / `stale`), and complete it (`gc_complete.lua`); one page
    /// of the release-ledger audit while a cycle runs; and one page of the
    /// enqueue-attempt audit (`gc_enqueue_attempt_audit.lua`, b.yuo Part R)
    /// while its walk of the same cycle runs (`state` carries both cursors).
    ///
    /// A member left in `inflight` — a transient error, a `stale` refusal, a
    /// decrement error on its own hash's keys, or a crash — is re-claimed in
    /// place by a later pass once its claim is older than
    /// `policy.reclaim_after`; no claim ever goes back to `pending` (b.1jt). An
    /// error from a `{gc}`-slot step ends the pass (`Err`); the next pass
    /// retries. The exception is the store refusing the age stamp or a
    /// re-claim `-OOM` (a full `{gc}` shard): that step or member is skipped
    /// and the pass goes on (b.pa1).
    ///
    /// On a store that keeps the out-of-memory collection's signal keys
    /// (b.pa1), the pass first takes a pending collection request
    /// (`GETDEL le:ext:{gc}:oom:request`); when there is one and the
    /// collection's rate limit allows, the call runs an out-of-memory
    /// collection instead — passes under `GcPolicy::OOM_COLLECTION` while one
    /// reports more work, for at most `OOM_COLLECTION_MAX_DURATION`, with the
    /// upload hold set — and returns their combined result. Only the GC leader
    /// calls this in production.
    ///
    /// SDD §11.3, §11.4.
    async fn process_pending(
        &self,
        policy: &GcPolicy,
        state: &mut GcPassState,
    ) -> Result<GcBatchResult, ServiceError>;

    /// The release-budget gate uploads carrying external values wait at
    /// (`LIVE_EDIT_EXT_RELEASE_MAX_BYTES`, b.yuo), or `None` when there is
    /// none: the budget is disabled (`0`), or the store keeps no release ledger.
    ///
    /// Carried by the store for the reason [`Self::ext_value_verify`] is: both
    /// upload handlers already hold it, and startup wiring delivers storage
    /// configuration to it. Required, with no default, so every implementation
    /// states it: a store returns its configured gate, a wrapping
    /// implementation delegates to its inner store, and a test double with no
    /// ledger returns `None`.
    fn ext_release_gate(&self) -> Option<&Arc<ExtReleaseGate>>;
}

// ── Valkey implementation ──────────────────────────────────────────────────

/// Test-only fault-injection queue alias for
/// `ValkeyExternalValueStore::process_pending_fault_injector_for_test`.
/// Spelled as a type alias rather than inline so `cargo clippy
/// --all-targets` does not trip on `clippy::type_complexity` for the
/// nested `Option<Arc<Mutex<VecDeque<(usize, PendingFault)>>>>` shape.
#[cfg(any(test, feature = "test-fault-injection"))]
pub type ProcessPendingFaultInjector =
    std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<(usize, PendingFault)>>>;

/// What a [`ProcessPendingFaultInjector`] entry does at the round trip it
/// targets (b.yuo). Test-only.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug)]
pub enum PendingFault {
    /// The round trip is not sent; this error is its result. Valid at every
    /// site.
    Fail(fred::error::Error),
    /// The round trip is sent and runs on the server, then its reply is
    /// replaced by this error: a lost reply. Valid at the decrement and
    /// complete sites.
    ExecuteThenFail(fred::error::Error),
    /// The round trip is sent twice with identical arguments and the second
    /// reply is returned: fred's transparent re-send of a command whose
    /// connection dropped after the server ran it. Valid at the decrement and
    /// complete sites.
    ExecuteTwice,
}

/// Production external value store backed by Valkey.
#[derive(Clone)]
pub struct ValkeyExternalValueStore {
    client: ValkeyClient,
    keys: KeySchema,
    /// Optional shared metrics handle so `process_pending` can bump
    /// `live_edit_gc_underflow_total` when the Lua script returns the
    /// `"deleted_underflow"` action (b.j9j item 4). Optional because
    /// unit tests construct the store without a metrics registry; the
    /// production wiring in `main.rs` always attaches it.
    metrics: Option<SharedMetrics>,
    /// Optional producer-side handle into the SR-3.1 ext-blob mirror
    /// pipeline (b.fgj Epic 6 Task `t2.fgj.zc.nx`). When `Some`, every
    /// successful per-hash `EVALSHA` inside `store_values` is followed
    /// by one `ExtMirrorSender::send(hash_hex, copy, last_store_ms)` call so a
    /// dedicated worker can mirror the blob to the checkpoint adapter
    /// outside the hot path — except in the fetch-on-miss rewrite
    /// (`ExtRecoveryRewrite::store_recovered_values`), whose bytes came
    /// from the adapter. When `None` — checkpoint disabled or the
    /// adapter is `Disabled` — the mirror path is a complete no-op (no
    /// enqueue attempt, no clone, no metric, no log).
    ///
    /// Production wiring attaches this via `with_ext_mirror_sender(..)`
    /// only when the pod memory plan funds the mirror's byte budget, which
    /// is exactly when checkpoint is enabled with a non-`Disabled` adapter;
    /// unit tests construct the store without it.
    ext_mirror_sender: Option<crate::checkpoint::ExtMirrorSender>,
    /// Optional `CheckpointStore` adapter handle used as the recovery
    /// fallback on `read_batch` misses (Epic `t1.fgj.zc` Task
    /// `t2.fgj.zc.p6`, SDD SR-3.2). `None` when checkpoint is disabled
    /// or the adapter is `Disabled`; on a miss the original
    /// `ExternalValueError` surfaces straight to the caller with no
    /// adapter consult (byte-identical pre-Task behaviour). `Some(adapter)`
    /// engages the singleflight-coalesced `get_ext_blob` →
    /// `store_recovered_values` rewrite path. See `try_recover_via_checkpoint`.
    checkpoint_store: Option<Arc<dyn CheckpointStore>>,
    /// Per-hash singleflight guard for the SR-3.2 recovery path. Always
    /// allocated so the `Some(checkpoint_store)` branch can rely on it
    /// without a runtime `Option` check on the hot path; when
    /// `checkpoint_store` is `None`, the singleflight is simply never
    /// consulted. Shared across `read_batch` calls inside a single store
    /// instance so concurrent diff-builds for the same hash coalesce.
    recovery_singleflight: Arc<ExtRecoverySingleflight>,
    /// `LIVE_EDIT_EXT_VALUE_VERIFY`, handed to [`prepare_external_values`]
    /// through [`ExternalValueStore::ext_value_verify`] together with
    /// `metrics`. `Enforce` unless `with_ext_value_verify` set it.
    ext_value_verify: ExtValueVerifyMode,
    /// The release-budget gate (`LIVE_EDIT_EXT_RELEASE_MAX_BYTES`, b.yuo),
    /// handed to the upload handlers through
    /// [`ExternalValueStore::ext_release_gate`]. Its budget `H` also sets the
    /// GC loop's pressure threshold `H / 2`. `None` (the default) disables
    /// both: no upload waits, and no pass is ever under pressure.
    ext_release_gate: Option<Arc<ExtReleaseGate>>,
    /// Test-only fault-injection queue for `process_pending`. Each entry
    /// is a `(visit_index, PendingFault)` tuple where `visit_index`
    /// names one `(member_index_in_pass, site)` pair. Three sites live in
    /// each member's processing (b.j9j item 3, expanded by b.u96 item 2,
    /// b.yuo):
    ///
    /// * site 0 — `EVALSHA gc_claim_pending` (atomic SREM pending + ZADD
    ///   inflight), or, for one of this build's own stale claims,
    ///   `EVALSHA gc_reclaim_stale` (the in-place re-claim, b.1jt)
    /// * site 1 — `EVALSHA gc_decrement_and_collect` (the at-most-once decrement)
    /// * site 2 — `EVALSHA gc_complete` (release of ownership after the decrement)
    ///
    /// Visit index `3 * member + site` selects the unique (member, site)
    /// pair, where `member` is the member's position in the pass's work list:
    /// the selected pending members first (oldest first), then the own stale
    /// claims it re-claims. A site that is never reached (a skipped member's later
    /// sites) consumes nothing. At each site, `process_pending` looks for an
    /// entry with that visit index anywhere in the queue and, if there is one,
    /// removes it and applies its [`PendingFault`]: an injected error is fed
    /// through `map_transient_fred_err` exactly as a real fred error would be.
    /// Members run concurrently, so the lookup does not depend on the order
    /// their sites are reached. Non-matching entries stay in the queue (e.g.,
    /// installing `[(4, Fail(Timeout))]` on a 3-member pass tags member 1's
    /// site 1 — `gc_decrement_and_collect` — leaving members 0 and 2 to run
    /// normally; member 1 stays in `ext_inflight` until a later pass
    /// re-claims it).
    ///
    /// **Test-only.** Production call sites never invoke this — it is
    /// exposed only through `with_process_pending_fault_injector_for_test`.
    /// Mirrors the `fred_error_injector_for_test` pattern in
    /// `LayerStore`. Compiled only under
    /// `#[cfg(any(test, feature = "test-fault-injection"))]` so release
    /// builds do not carry the seam.
    #[cfg(any(test, feature = "test-fault-injection"))]
    process_pending_fault_injector_for_test: Option<ProcessPendingFaultInjector>,
    /// Test-only override of the decrement's claim-age fence
    /// ([`GC_DECREMENT_FENCE_MS`]); see
    /// [`Self::with_decrement_fence_for_test`]. Not compiled into production
    /// builds, which always pass the constant.
    #[cfg(any(test, feature = "test-support"))]
    decrement_fence_ms_for_test: Option<u64>,
    /// Test-only override of the read hold margin `gc_select.lua` clips by
    /// ([`GC_HOLD_MARGIN_MS`]); see [`Self::with_hold_margin_for_test`].
    #[cfg(any(test, feature = "test-support"))]
    hold_margin_ms_for_test: Option<u64>,
}

impl ValkeyExternalValueStore {
    pub fn new(client: ValkeyClient, keys: KeySchema) -> Self {
        Self {
            client,
            keys,
            metrics: None,
            ext_mirror_sender: None,
            checkpoint_store: None,
            recovery_singleflight: Arc::new(ExtRecoverySingleflight::new()),
            ext_value_verify: ExtValueVerifyMode::Enforce,
            ext_release_gate: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            process_pending_fault_injector_for_test: None,
            #[cfg(any(test, feature = "test-support"))]
            decrement_fence_ms_for_test: None,
            #[cfg(any(test, feature = "test-support"))]
            hold_margin_ms_for_test: None,
        }
    }

    /// **Test-only.** Enqueue `releases` with the plain (non-attempt) enqueue
    /// — what the trait's `schedule_release` did before b.yuo Part R moved the
    /// retirement onto [`ExternalValueStore::enqueue_retirement_attempt`]. Kept
    /// for tests that seed `le:ext:{gc}:pending` directly; no production path
    /// enqueues a retirement this way. Compiled only for tests and the
    /// `test-support` feature.
    #[cfg(any(test, feature = "test-support"))]
    pub async fn schedule_release(&self, releases: &[ExtRelease]) -> Result<(), ServiceError> {
        enqueue_ext_releases(&self.client, &self.keys, releases).await
    }

    /// Attach the release-budget gate (b.yuo). Builder-style beside
    /// `with_metrics`. Only call this with a gate whose budget is above zero;
    /// without it no upload waits on the release budget and the GC loop never
    /// collects under pressure.
    pub fn with_ext_release_gate(mut self, gate: Arc<ExtReleaseGate>) -> Self {
        self.ext_release_gate = Some(gate);
        self
    }

    /// `T`, the release-ledger total at or above which a GC pass collects
    /// under pressure: half the release budget, or 0 (never) without a gate.
    fn pressure_threshold_bytes(&self) -> u64 {
        self.ext_release_gate
            .as_ref()
            .map_or(0, |gate| gate.budget_bytes() / 2)
    }

    /// Set the upload content-check mode (`LIVE_EDIT_EXT_VALUE_VERIFY`, SDD
    /// §11.1) that [`prepare_external_values`] applies for this store; the check
    /// is counted on the [`Self::with_metrics`] handle. Builder-style beside
    /// `with_metrics` so the production wiring stays one chained expression.
    /// Without it the store enforces.
    pub fn with_ext_value_verify(mut self, mode: ExtValueVerifyMode) -> Self {
        self.ext_value_verify = mode;
        self
    }

    /// Attach a shared metrics handle. Builder-style so the production
    /// wiring stays a single chained expression in `main.rs` and unit
    /// tests can construct the store without metrics. b.j9j item 4.
    pub fn with_metrics(mut self, metrics: SharedMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Attach the SR-3.1 ext-blob mirror sender (b.fgj Epic 6 Task
    /// `t2.fgj.zc.nx`). Builder-style alongside `with_metrics` so the
    /// production wiring stays a single chained expression. Only call
    /// this when checkpoint is enabled with a non-`Disabled` adapter —
    /// when the field is `None`, `store_values` performs zero mirror-side
    /// work (no enqueue, no clone, no metric, no log), matching the
    /// pre-Task behaviour.
    pub fn with_ext_mirror_sender(mut self, sender: crate::checkpoint::ExtMirrorSender) -> Self {
        self.ext_mirror_sender = Some(sender);
        self
    }

    /// Attach the SR-3.2 recovery `CheckpointStore` adapter handle
    /// (Epic `t1.fgj.zc` Task `t2.fgj.zc.p6`). Builder-style alongside
    /// `with_metrics` / `with_ext_mirror_sender` so the production
    /// wiring stays a single chained expression. Only call this when
    /// checkpoint is enabled with a non-`Disabled` adapter — when the
    /// field is `None`, `read_batch` misses surface straight to the
    /// caller with no adapter consult (byte-identical pre-Task
    /// behaviour). When `Some(adapter)`, a missing-meta / missing-data
    /// hash drives a singleflight-coalesced `get_ext_blob` →
    /// `store_recovered_values` rewrite that returns the recovered bytes
    /// inline. See `try_recover_via_checkpoint`.
    pub fn with_checkpoint_store(mut self, store: Arc<dyn CheckpointStore>) -> Self {
        self.checkpoint_store = Some(store);
        self
    }

    /// Attempt to recover a missing ext-value blob via the configured
    /// `CheckpointStore` adapter (SDD SR-3.2). Called from the
    /// `read_batch` miss arms BEFORE the original `ExternalValueError`
    /// surfaces, so a successful recovery transparently fills the hash
    /// without the caller observing the miss.
    ///
    /// Returns `Ok(bytes)` when the adapter served the blob; the bytes
    /// have been rewritten into Valkey via `store_recovered_values` (a
    /// `store_values` that sends no mirror copy) before this returns, so a
    /// follow-up `read_batch` for the same hash hits Valkey alone with no
    /// adapter consult.
    ///
    /// Returns `Err(original_error)` (preserving the wire contract)
    /// when:
    /// - The store was constructed without `with_checkpoint_store`
    ///   (`self.checkpoint_store == None`).
    /// - The adapter returned `Ok(None)` — the hash is unknown.
    /// - The adapter returned `Err(CheckpointError::Permanent)` — no
    ///   retry could help; surface the original error.
    /// - The adapter returned an unexpected variant (`NotFound`,
    ///   `Conflict`, `NotImplemented`) — `get_ext_blob`'s trait contract
    ///   reserves these to other methods; treat as permanent.
    ///
    /// Returns `Err(ServiceError::Unavailable(...))` when the adapter
    /// returned `Err(CheckpointError::Transient)` — the caller's gRPC
    /// status becomes `UNAVAILABLE` so the client retries against a
    /// healthy replica.
    ///
    /// The adapter fetch is single-flight per `hash_hex` via
    /// `recovery_singleflight` so a thundering-herd subscribe-storm
    /// against a freshly-recovered layer issues exactly one adapter
    /// call + Valkey rewrite per hash; subsequent waiters share the
    /// result. Every outcome ticks the
    /// `live_edit_checkpoint_ext_recovery_total{result=...}` counter
    /// for operator visibility.
    ///
    /// The policy itself is [`recover_ext_value_via_checkpoint`], shared with
    /// [`MockExternalValueStore`]; this store supplies only its
    /// [`ExtRecoveryRewrite::store_recovered_values`].
    async fn try_recover_via_checkpoint(
        &self,
        hash_hex: &str,
        hash: &[u8; 32],
        original_error: ServiceError,
        recovery_counter: Option<&std::sync::atomic::AtomicUsize>,
    ) -> Result<Vec<u8>, ServiceError> {
        recover_ext_value_via_checkpoint(
            self,
            self.checkpoint_store.clone(),
            &self.recovery_singleflight,
            self.metrics.as_ref(),
            hash_hex,
            hash,
            original_error,
            recovery_counter,
        )
        .await
    }

    /// Install a queue of synthetic `(visit_index, PendingFault)` entries
    /// applied at the three per-member round trips inside
    /// `process_pending` — the claim, the decrement and the completion.
    /// Each site looks up its visit index (`3 * member + site`) anywhere in
    /// the queue; a matching entry is removed and applied (see
    /// [`PendingFault`]), and an injected error is fed through
    /// `map_transient_fred_err` exactly as a real fred error would be, so the
    /// real error path runs end-to-end. Non-matching entries stay in the
    /// queue.
    ///
    /// **Test-only.** Production call sites never invoke this — it
    /// exists so that regression coverage can prove the per-member error
    /// branches each leave the member where the next pass or its re-claim
    /// retries it without aborting the whole pass (b.u96 item 2), and that a
    /// lost reply or a fred re-send of the decrement or the completion
    /// applies the decrement once (b.yuo).
    ///
    /// Compiled only under `#[cfg(any(test, feature = "test-fault-injection"))]`
    /// so release builds do not carry the seam.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn with_process_pending_fault_injector_for_test(
        mut self,
        queue: ProcessPendingFaultInjector,
    ) -> Self {
        self.process_pending_fault_injector_for_test = Some(queue);
        self
    }

    /// **Test-only.** Pass `fence` instead of [`GC_DECREMENT_FENCE_MS`] as the
    /// claim-age fence of every GC decrement this store runs, so a test can
    /// drive the `stale` refusal in milliseconds rather than waiting out the
    /// production 15 s. Compiled only for tests and the `test-support`
    /// feature; a production build always passes the constant.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_decrement_fence_for_test(mut self, fence: std::time::Duration) -> Self {
        self.decrement_fence_ms_for_test = Some(duration_ms(fence));
        self
    }

    /// **Test-only.** Clip GC selection `margin` below the earliest live read
    /// hold's start instead of the production [`GC_HOLD_MARGIN_MS`], so a test
    /// can place a release just before or after a hold's start without
    /// waiting out a second. Compiled only for tests and the `test-support`
    /// feature; a production build always passes the constant.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_hold_margin_for_test(mut self, margin: std::time::Duration) -> Self {
        self.hold_margin_ms_for_test = Some(duration_ms(margin));
        self
    }

    /// The read hold margin this store's GC selection clips by, in ms:
    /// [`GC_HOLD_MARGIN_MS`], or a test's override.
    fn hold_margin_ms(&self) -> u64 {
        #[cfg(any(test, feature = "test-support"))]
        if let Some(margin) = self.hold_margin_ms_for_test {
            return margin;
        }
        GC_HOLD_MARGIN_MS
    }

    /// The claim-age fence this store's GC decrements pass, in ms:
    /// [`GC_DECREMENT_FENCE_MS`], or a test's override.
    fn decrement_fence_ms(&self) -> u64 {
        #[cfg(any(test, feature = "test-support"))]
        if let Some(fence) = self.decrement_fence_ms_for_test {
            return fence;
        }
        GC_DECREMENT_FENCE_MS
    }

    /// Test-only helper: if the fault-injection queue is installed and
    /// holds an entry whose `visit_index` equals `visit_index`, remove the
    /// first such entry and return its fault. Otherwise `None`.
    #[cfg(any(test, feature = "test-fault-injection"))]
    fn take_process_pending_fault(&self, visit_index: usize) -> Option<PendingFault> {
        let queue = self.process_pending_fault_injector_for_test.as_ref()?;
        let mut guard = queue.lock().ok()?;
        let at = guard.iter().position(|(vi, _)| *vi == visit_index)?;
        guard.remove(at).map(|(_, fault)| fault)
    }

    /// Run one of `process_pending`'s per-member round trips, applying an
    /// injected fault for `visit_index` when the test-only queue holds one.
    /// Without the `test-fault-injection` feature this is `op()`.
    async fn gc_round_trip<T, F, Fut>(
        &self,
        visit_index: usize,
        op: F,
    ) -> Result<T, fred::error::Error>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, fred::error::Error>>,
    {
        #[cfg(any(test, feature = "test-fault-injection"))]
        if let Some(fault) = self.take_process_pending_fault(visit_index) {
            return match fault {
                PendingFault::Fail(e) => Err(e),
                PendingFault::ExecuteThenFail(e) => {
                    let _ = op().await;
                    Err(e)
                }
                PendingFault::ExecuteTwice => {
                    let _ = op().await;
                    op().await
                }
            };
        }
        #[cfg(not(any(test, feature = "test-fault-injection")))]
        let _ = visit_index;
        op().await
    }
}

impl ValkeyExternalValueStore {
    /// The body of [`ExternalValueStore::store_values`] (`verified` false)
    /// and [`ExternalValueStore::store_verified_values`] (`verified` true),
    /// both [`ExtMirroring::Send`], and of the recovery rewrite
    /// [`ExtRecoveryRewrite::store_recovered_values`] (`verified` false,
    /// [`ExtMirroring::Skip`]).
    /// `verified` is passed to `ext_store_value.lua` as its ARGV[2] flag, which
    /// arms the script's type / length heal of an existing data key.
    /// `mirroring` decides only whether a committed value's copy goes to the
    /// wired ext-mirror sender: [`ExtMirroring::Skip`] runs the path a store
    /// with no sender runs. Nothing else in the store differs between the
    /// three.
    async fn store_values_with(
        &self,
        values: &[(&[u8; 32], &[u8])],
        verified: bool,
        mirroring: ExtMirroring,
    ) -> Result<(), ServiceError> {
        if values.is_empty() {
            return Ok(());
        }

        // Atomic write per hash: one `EVALSHA` of `ext_store_value.lua`
        // (`SET data NX` — plus, when `verified`, the script's type / length
        // heal — + `HINCRBY meta ref_count 1` + `HSET meta last_store_ms`)
        // on the MULTIPLEXED DATA POOL. Both keys share a Cluster slot via the
        // `{hash_hex}` hash tag on ext_meta/ext_data, so the single-slot script
        // is legal on Valkey Cluster and routes by its own KEYS.
        //
        // b.rej replaced the per-hash `MULTI { SET; HINCRBY } EXEC` this used to
        // run on an exclusive `txn_pool` slot. Valkey runs Lua atomically under
        // its single server thread, so the exclusivity a `MULTI` needed a
        // dedicated session for (b.avd: the multiplexed pool can interleave a
        // sibling caller's commands between our MULTI and EXEC and corrupt both
        // transactions) is free here — there is no MULTI to interleave into.
        // Two things fall out, and both are the point of the change:
        //
        //   * No `txn_pool` checkout, so no b.7td checkout hygiene probe
        //     (`DISCARD` + `UNWATCH`, ≈ 63 µs and two extra round-trips per
        //     blob) and no contention with the `update_layer` WATCH/MULTI/EXEC
        //     slow path or `CreateSession` for exclusive connections. A blob
        //     store is one round trip again.
        //   * The pre-flight increment carries a server-side `last_store_ms`
        //     stamp, which is what arms the reference-count reconciler
        //     (`storage::ext_reconcile`) — see the script header.
        //
        // Each hash is a separate call because different hashes may live on
        // different slots, and parallelism stays bounded by
        // `STORE_VALUES_MAX_INFLIGHT`. That bound survives the move off the txn
        // pool with a DIFFERENT justification, and it is not vestigial: on the
        // multiplexed data pool a multi-MB blob occupies a shared socket for
        // its whole transfer, so a wave of concurrent blobs head-of-line-blocks
        // the reads and Lua fast-path commits that share those connections. The
        // enforced floor keeps the pool at least as wide as the wave; a pool
        // WIDER than the wave is what keeps some connections clear while one
        // drains. See the module-level constant for the full rationale and the
        // unit test that pins it.

        // One future per blob, each making its blob's one owned copy only when
        // it is first polled. `buffered(STORE_VALUES_MAX_INFLIGHT)` keeps at
        // most that many stores in flight, and a store makes its copy only as
        // it starts — not one per blob in the upload, all at once.
        //
        // The copies alive are not strictly capped at
        // `STORE_VALUES_MAX_INFLIGHT`: the script's ARGV holds the copy inside
        // fred's `Command`, and fred's router task drops that command only
        // after it has sent the reply (`protocol::responders::respond_to_caller`
        // takes it by value, sends, then returns). A store woken by that reply
        // on another thread can finish, and the next store make its copy,
        // before the router reaches the drop. The gap is synchronous — no
        // await between the send and the drop — so each thread holds at most
        // one such copy at a time: the copies the store path holds are at most
        // `STORE_VALUES_MAX_INFLIGHT` plus one per other runtime thread
        // running a router at that moment. (Measured: the T7 census, its
        // handler on the test thread and fred's routers on one worker, peaked
        // at 9, the lingering copy freed inside `respond_to_caller` on the
        // worker.) Not avoidable here short of an upload permit tied to the
        // copy's last drop, which the mirror's share of the copy would then
        // hold too. A copy handed to the ext-mirror is not in that count:
        // `mirror_copy` is the same allocation, moves into the mirror channel
        // at `ExtMirrorSender::send` and stays alive until a mirror worker's
        // `put_ext_blob` and every clone the adapter took of it are done, so
        // with a mirror sender wired (checkpointing on) mirrored copies
        // outlive their stores. Those are bounded in bytes by the mirror's own
        // budget, `M_ext` (SDD §22.17): the charge it takes after the commit
        // sits in the copy's owner and is released at the copy's last drop,
        // wherever that is. The upload charge does not rest on the store
        // path's bound above (`STORE_VALUES_MAX_INFLIGHT` plus one per router
        // thread): one store copies each value once, so its copies alive
        // never exceed the external-value bytes, which `e_ext` charges at
        // more than one per byte. An `UpdateLayer` RPC can store its values
        // twice — its checkpoint-recovery re-issue pre-stores them again
        // (`PreparedExternalValues::store`) — and a first-store copy the
        // mirror still holds can then be alive beside the second store's copy
        // and its write buffer, past what `e_ext` prices. The handler charges
        // that second copy to the upload budget itself, for the second store
        // only (`grpc::layer`'s re-issue arm).
        //
        // The futures are collected into a `Vec` before they are streamed,
        // rather than built lazily by a `map` closure inside the stream: a
        // closure borrowing the input slice, held in the stream's type, does
        // not satisfy the higher-rank lifetime bounds this `async_trait`
        // future is checked against. Each future borrows only its own
        // `(hash, data)` pair.
        use futures::stream::{StreamExt, TryStreamExt};
        let store_futures: Vec<_> = values
            .iter()
            .map(|&(hash, data)| {
                let hash_hex = sha256_to_hex(hash);
                let meta_key = self.keys.ext_meta(&hash_hex);
                let data_key = self.keys.ext_data(&hash_hex);
                let client = self.client.clone();
                // Clone the sender into the async move so the future is fully
                // owned; `ExtMirrorSender` is cheap to clone (an mpsc Sender,
                // the `SharedMetrics` `Arc` and the byte budget's
                // `Arc<Semaphore>`). When checkpoint is disabled or the
                // adapter is `Disabled` this is `None` and the mirror path
                // below is a complete no-op; `ExtMirroring::Skip` (the
                // recovery rewrite) makes it `None` for this call too.
                let ext_mirror_sender = match mirroring {
                    ExtMirroring::Send => self.ext_mirror_sender.clone(),
                    ExtMirroring::Skip => None,
                };
                async move {
                    // The blob's one owned copy, exact-size, shared by the
                    // script's ARGV and the SR-3.1 ext-mirror send below. With a
                    // sender wired it is built by `ExtMirrorSender::copy_value`,
                    // whose owner can carry the mirror's byte charge, so the
                    // charge taken after the commit lives exactly as long as this
                    // memory; the ARGV's handle is a refcount bump, not a second
                    // copy. With no sender it is a plain copy and the path is the
                    // pre-Task hot path verbatim.
                    //
                    // `benches/ext_mirror_enqueue.rs` (`enqueue_once`) measures
                    // the same two `ExtMirrorSender` helpers this path calls.
                    let mirror_copy: Option<crate::checkpoint::ExtMirrorCopy> = ext_mirror_sender
                        .as_ref()
                        .map(|sender| sender.copy_value(data));
                    let data_owned = match &mirror_copy {
                        Some(copy) => copy.bytes().clone(),
                        None => bytes::Bytes::copy_from_slice(data),
                    };
                    // The payload rides as `Value::Bytes`, never a `String`. Two
                    // reasons, both load-bearing: the blob is arbitrary binary (NUL
                    // bytes, non-UTF-8) and must round-trip unchanged, and
                    // `Script::evalsha_with_reload` CLONES its args before the first
                    // attempt so the `NOSCRIPT` reload can re-send them
                    // (`fred-10.1.0/src/types/scripts.rs:232-236`) — `Bytes` makes
                    // that clone a refcount bump rather than a copy of a multi-MB
                    // payload.
                    // ARGV[2] is the script's verified flag: `"1"` arms its heal
                    // of a wrong-typed or wrong-length existing data key.
                    let argv: Vec<fred::types::Value> = vec![
                        fred::types::Value::Bytes(data_owned),
                        fred::types::Value::from(if verified { "1" } else { "0" }),
                    ];
                    // b.bhb: classified, not wrapped `Internal` directly. This is a
                    // real round-trip and `store_values` itself runs no retry loop
                    // of its own — an error that reaches this `map_err` reaches the
                    // CreateLayer / UpdateLayer handler directly (via
                    // `store_external_values`), so the transient answer is
                    // `Unavailable` rather than the in-loop `Aborted` retry signal
                    // (both reach the client as `UNAVAILABLE`; the variant says
                    // no loop retries it). b.rej: the two queue-time `Internal` wraps that
                    // used to sit above this (`queue SET ext_data failed`, `queue
                    // HINCRBY ext_meta ref_count failed`) are gone with the
                    // `Transaction` — a script has no local queueing step whose
                    // argument/slot check could fail separately from the call.
                    //
                    // **Pool retry posture, and why it is accepted.** "No retry
                    // loop here" is not "no retry": this call rides the multiplexed
                    // DATA pool, which runs fred's DEFAULT `max_command_attempts`
                    // (3 — see the `max_command_attempts` discussion at
                    // `client.rs:~385`). A connection that drops after the script
                    // has already executed server-side but before its reply lands
                    // is re-issued by fred, the `HINCRBY` runs a second time, and
                    // the caller sees a plain `Ok` with a `ref_count` one higher
                    // than the object graph justifies. The direction is LEAK only,
                    // never over-release — a re-executed store can only add a
                    // surplus `+1`, never consume a decrement someone else was
                    // owed. The exclusive `txn_pool` this used to run on pinned
                    // `max_command_attempts = 1` and so surfaced the same shape as
                    // an `UNAVAILABLE` instead; that was never exactly-once either,
                    // because the client's own RPC retry then re-ran the upload and
                    // paid the identical second `+1` one layer up. What actually
                    // drains it — and what this change arms — is b.yda's
                    // reference-count reconciler: the duplicate increment carries
                    // the script's `last_store_ms` stamp, so it becomes eligible
                    // for repair once that instant is a full grace old. Deliberately
                    // NOT `with_options(max_attempts: 1)`: that converts routine
                    // connection blips into user-visible errors without preventing
                    // the duplicate.
                    // The reply is `(ref_count, last_store_ms)`. Nothing on the
                    // write path branches on the count. The stamp goes to the
                    // ext-mirror with the copy: the worker's confirmation of the
                    // copy (`ext_backup_confirm.lua`) writes only while the meta
                    // still carries it, so a store of the same hash after this
                    // one leaves the confirmation to its own copy.
                    // b.zt3 re-send audit: fred keeps its default write attempts here, so a lost
                    // connection can run this twice: a second `ref_count` increment, a leak only,
                    // which the reference-count reconciler drains (`ext_store_value.lua`). The
                    // stamp returned is then the second run's, the one the meta holds.
                    let (_ref_count, last_store_ms): (i64, String) = EXT_STORE_VALUE_SCRIPT
                        .evalsha_with_reload::<(i64, String), _, _>(
                            client.pool.next(),
                            vec![meta_key, data_key],
                            argv,
                        )
                        .await
                        .map_err(|e| {
                            // A store write refusal (`-OOM` and its three
                            // siblings) refused this script's first write, so
                            // this value was not stored: `UNAVAILABLE` "nothing
                            // was applied", with its retry hint, counted at
                            // `stage="ext_store"`, before the mapper would answer
                            // it a plain, uncounted `UNAVAILABLE`. See
                            // `storage::client::store_write_refusal`.
                            refuse_store_write(
                                FredErrOrigin::metrics(
                                    self.metrics.as_ref(),
                                    POOL_LABEL_DATA,
                                    self.client.store_oom_signal(),
                                ),
                                STORE_WRITE_REFUSED_STAGE_EXT_STORE,
                                &e,
                            )
                            .unwrap_or_else(|| {
                                map_fred_err_for_grpc(
                                    FredErrOrigin::client(&self.client, POOL_LABEL_DATA),
                                    "EVALSHA ext store_values",
                                    e,
                                )
                            })
                        })?;
                    // SR-3.1 ext-blob mirror send. By the time we reach
                    // here the Valkey script has committed — the hot
                    // path is done. The cross-Task contract (b.fgj Task
                    // `t2.fgj.zc.nx`) is that `ExtMirrorSender::send` is the
                    // SOLE enqueue surface and MUST NOT block, fail, or panic
                    // the hot path; a value larger than the mirror's byte
                    // budget, a full budget, a full or closed channel are all
                    // absorbed there and surface only via
                    // `checkpoint_ext_mirror_dropped_total{reason}`.
                    // The `if let` keeps the `None` path a complete no-op
                    // (no enqueue attempt, no metric, no log), preserving
                    // the pre-Task behaviour when checkpoint is disabled or
                    // the adapter is `Disabled`. Sending only AFTER the
                    // `EVALSHA` succeeds is load-bearing twice over: a queued
                    // `(hash, bytes)` for a store that never committed would
                    // corrupt the mirror copy, and a store that fails (an
                    // `-OOM` refusal included, which returned above) takes no
                    // charge from the byte budget. `hash_hex` is already in
                    // lowercase 64-char form from `sha256_to_hex` (line ~444)
                    // — pass it through as-is.
                    if let (Some(sender), Some(copy)) = (ext_mirror_sender.as_ref(), mirror_copy) {
                        sender.send(hash_hex.clone(), copy, last_store_ms);
                    }
                    Ok::<_, ServiceError>(())
                }
            })
            .collect();
        futures::stream::iter(store_futures)
            .buffered(STORE_VALUES_MAX_INFLIGHT)
            .try_collect::<Vec<()>>()
            .await?;

        Ok(())
    }
}

#[async_trait]
impl ExternalValueStore for ValkeyExternalValueStore {
    async fn store_values(&self, values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
        self.store_values_with(values, false, ExtMirroring::Send)
            .await
    }

    async fn store_verified_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), ServiceError> {
        self.store_values_with(values, true, ExtMirroring::Send)
            .await
    }

    async fn load_values(
        &self,
        hashes: &[&[u8; 32]],
    ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
        if hashes.is_empty() {
            return Ok(Vec::new());
        }

        // One round trip: pipeline `GET ext_data:{hash_hex}` for each hash.
        // Data keys are content-addressed (SDD §11.1), so no meta lookup is
        // needed. Missing hashes are silently skipped.
        let mut data_futures = Vec::with_capacity(hashes.len());
        let origin = FredErrOrigin::client(&self.client, POOL_LABEL_DATA);
        for hash in hashes {
            let hash_hex = sha256_to_hex(hash);
            let data_key = self.keys.ext_data(&hash_hex);
            let hash_copy = **hash;
            let pool = self.client.pool.clone();
            data_futures.push(async move {
                let data: Option<bytes::Bytes> = pool
                    .get(&data_key)
                    .await
                    .map_err(|e| map_fred_err_for_grpc(origin, "GET ext_data", e))?;
                Ok::<_, ServiceError>((hash_copy, data))
            });
        }

        let data_results = futures::future::try_join_all(data_futures).await?;
        let results: Vec<([u8; 32], Vec<u8>)> = data_results
            .into_iter()
            .filter_map(|(hash, data)| data.map(|d| (hash, d.to_vec())))
            .collect();

        Ok(results)
    }

    async fn read_batch(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        // Forward to the counter-aware path with `None` so existing
        // callers (and the trait default implementation) see byte-
        // identical behaviour — the recovery counter is purely additive.
        self.read_batch_with_recovery_counter(hashes, None).await
    }

    async fn read_batch_with_recovery_counter(
        &self,
        hashes: &[[u8; 32]],
        recovery_counter: Option<&std::sync::atomic::AtomicUsize>,
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        if hashes.is_empty() {
            return Ok(std::collections::HashMap::new());
        }

        // Pipeline `GET ext_data` + `HEXISTS ext_meta` for each hash in a
        // single round-trip per hash (b.kf1 medium). Pre-b.kf1 the miss
        // path issued a follow-up `HEXISTS` after observing a missing
        // GET — two RTTs per miss. Pipelining both into one fred
        // `Pipeline` collapses that to a single round-trip, and because
        // `ext_meta`/`ext_data` share the `{hash_hex}` Cluster slot the
        // pipeline is slot-safe.
        //
        // The per-hash futures still run concurrently across hashes via
        // `try_join_all`, so a wide read fans out across the data pool's
        // multiplexed connections rather than serializing on one client.
        let mut fetch_futures = Vec::with_capacity(hashes.len());
        let origin = FredErrOrigin::client(&self.client, POOL_LABEL_DATA);
        for hash in hashes {
            let hash_hex = sha256_to_hex(hash);
            let meta_key = self.keys.ext_meta(&hash_hex);
            let data_key = self.keys.ext_data(&hash_hex);
            let hash_copy = *hash;
            let pool = self.client.pool.clone();
            fetch_futures.push(async move {
                let pipeline = pool.next().pipeline();
                let _: () = pipeline.get(&data_key).await.map_err(|e| {
                    ServiceError::Internal(format!("queue GET ext_data failed: {e}"))
                })?;
                let _: () = pipeline
                    .hexists(&meta_key, "ref_count")
                    .await
                    .map_err(|e| {
                        ServiceError::Internal(format!(
                            "queue HEXISTS ext_meta ref_count failed: {e}"
                        ))
                    })?;
                // The round-trip (the two queue calls above only buffer
                // locally). No in-process retry wraps this read, so a
                // transient Valkey failure answers `UNAVAILABLE` and the
                // client retries; only a permanent one answers `INTERNAL`.
                let (data, meta_exists): (Option<bytes::Bytes>, bool) =
                    pipeline.all().await.map_err(|e| {
                        map_fred_err_for_grpc(origin, "pipeline GET ext_data + HEXISTS ext_meta", e)
                    })?;
                let state = if meta_exists {
                    MetaState::Present
                } else {
                    MetaState::Missing
                };
                Ok::<_, ServiceError>((hash_copy, hash_hex, state, data))
            });
        }

        let fetch_results = futures::future::try_join_all(fetch_futures).await?;

        let mut result = std::collections::HashMap::with_capacity(fetch_results.len());
        for (hash, hash_hex, meta_state, data) in fetch_results {
            match (data, meta_state) {
                (Some(d), _) => {
                    // Hot path — zero adapter / singleflight work. SDD
                    // SR-3.2 explicitly bounds the recovery work to the
                    // miss arms below.
                    result.insert(hash, d.to_vec());
                }
                (None, MetaState::Missing) => {
                    tracing::error!(
                        hash = %hash_hex,
                        "external value meta not found during diff reconstruction"
                    );
                    let original = ServiceError::from(ExternalValueError::MetaNotFound {
                        hash_hex: hash_hex.clone(),
                    });
                    let recovered = self
                        .try_recover_via_checkpoint(&hash_hex, &hash, original, recovery_counter)
                        .await?;
                    result.insert(hash, recovered);
                }
                (None, MetaState::Present) => {
                    tracing::error!(
                        hash = %hash_hex,
                        "external value data not found during diff reconstruction"
                    );
                    let original = ServiceError::from(ExternalValueError::DataNotFound {
                        hash_hex: hash_hex.clone(),
                    });
                    let recovered = self
                        .try_recover_via_checkpoint(&hash_hex, &hash, original, recovery_counter)
                        .await?;
                    result.insert(hash, recovered);
                }
            }
        }

        Ok(result)
    }

    async fn probe_values(&self, hashes: &[[u8; 32]]) -> Result<ExtValueProbe, ServiceError> {
        // `load_values` is exactly the probe's read: one pipelined
        // `GET ext_data:{hash_hex}` per hash on the data pool, a missing key
        // skipped silently, a failed `GET` classified by
        // `map_fred_err_for_grpc`, and no meta read, log or adapter consult.
        let refs: Vec<&[u8; 32]> = hashes.iter().collect();
        let found: std::collections::HashMap<[u8; 32], Vec<u8>> =
            self.load_values(&refs).await?.into_iter().collect();
        let missing = hashes
            .iter()
            .filter(|hash| !found.contains_key(*hash))
            .copied()
            .collect();
        Ok(ExtValueProbe { found, missing })
    }

    async fn enqueue_retirement_attempt(
        &self,
        object_id: &str,
        attempt: &str,
        releases: &[ExtRelease],
        send_by: std::time::Instant,
    ) -> Result<ReleaseAttemptOutcome, ServiceError> {
        let members: Vec<String> = releases.iter().map(ExtRelease::member).collect();
        enqueue_ext_release_attempt(
            &self.client,
            &self.keys,
            object_id,
            attempt,
            &members,
            send_by,
        )
        .await
    }

    async fn process_pending(
        &self,
        policy: &GcPolicy,
        state: &mut GcPassState,
    ) -> Result<GcBatchResult, ServiceError> {
        // The out-of-memory collection (b.pa1; module header): a request a
        // replica left since the last pass, taken with `GETDEL`, makes this
        // pass a collection under `GcPolicy::OOM_COLLECTION` when its rate
        // limit allows.
        if let Some(requested_by) = self.take_oom_collection_request(state).await {
            return self.run_oom_collection(&requested_by, state).await;
        }
        self.gc_pass(policy, state).await
    }

    fn ext_release_gate(&self) -> Option<&Arc<ExtReleaseGate>> {
        self.ext_release_gate.as_ref()
    }

    fn ext_value_verify(&self) -> ExtValueVerify<'_> {
        ExtValueVerify {
            mode: self.ext_value_verify,
            metrics: self.metrics.as_ref(),
        }
    }
}

impl ValkeyExternalValueStore {
    /// One GC pass under `policy`: the steps [`ExternalValueStore::process_pending`]
    /// documents, without the out-of-memory collection's request check.
    /// [`Self::run_oom_collection`] runs its passes through here.
    async fn gc_pass(
        &self,
        policy: &GcPolicy,
        state: &mut GcPassState,
    ) -> Result<GcBatchResult, ServiceError> {
        // All maintenance traffic flows through the dedicated maintenance
        // pool (b.kf1 H-stor-5) so a long pass cannot queue behind
        // writers / readers on the multiplexed data pool. `acquire_maintenance`
        // also bumps `valkey_pool_in_use{pool="maintenance"}` for the duration
        // of this pass (b.mfn).
        let (maint_pool, _maint_token) = self.client.acquire_maintenance();

        // 1. One read-only selection: the ledger's levels, the oldest age, the
        //    settled records due for teardown, the members to claim, a sample
        //    of un-aged members, and the stale claims in `inflight` — this
        //    build's own older than D, an older build's older than F. An error
        //    here (every `{gc}`-slot step, but for the stamp and a re-claim
        //    refused `-OOM`, below) ends the pass; the next pass retries.
        //
        //    Progress is marked on the loop's heartbeat after every step and
        //    every member below (`GcPassState::mark_progress`), so a long pass
        //    that keeps making progress keeps the lease refresh ticker renewing
        //    (b.1jt, `GC_PROGRESS_STALL_AFTER`).
        let selection = self.gc_select(maint_pool, policy).await?;
        state.mark_progress();
        let mut result = GcBatchResult {
            ledger: Some(selection.snapshot(policy)),
            ..GcBatchResult::default()
        };

        // 2. Stamp an age on every sampled un-aged member (an older build's
        //    bare `SADD`), so it is collected only past the age floor and the
        //    read-hold clip, measured from the stamp (b.1jt). Never collected
        //    unstamped: the selection does not return it as a candidate. A
        //    stamp the store refuses `-OOM` (a full `{gc}` shard) is skipped,
        //    not the end of the pass, so the pass still collects (b.pa1); the
        //    members are stamped by a later pass.
        match self.stamp_unaged(maint_pool, &selection.unaged).await? {
            Some(stamped) => result.stamped = stamped,
            None => {
                result.oom_skipped += 1;
                log_oom_skip(state, "the age stamp of un-aged pending releases");
            }
        }
        state.mark_progress();

        // 3. Drop an older build's claims older than F without a decrement
        //    (b.1jt): this build never decrements a claim it did not make.
        result.foreign_dropped = self
            .drop_foreign_claims(maint_pool, &selection.foreign_claims)
            .await?;
        state.mark_progress();

        // 4. Tear down the replay records of members settled at least U ago.
        result.torn_down = self
            .tear_down_settled(maint_pool, &selection.settled, state.progress.as_ref())
            .await?;
        state.mark_progress();

        // 5. Claim → decrement → complete, at most GC_MEMBER_CONCURRENCY
        //    members at a time: the selected pending members first, then this
        //    build's own claims older than D, each re-claimed in place (never
        //    moved back to `pending`, b.1jt). The futures are collected before
        //    they are streamed, for the same higher-rank-lifetime reason as in
        //    `store_values_with`. A `{gc}`-slot error ends the pass and drops
        //    the members still in flight — each step is crash-safe, so that is
        //    a crash of those members, which a later pass re-claims — except a
        //    re-claim refused `-OOM`, which skips only its member (b.pa1).
        //    `completed` counts the members whose decrement was decided AND whose
        //    completion landed (`Applied`, `Replayed`); a `Stale`, `Left` or
        //    `DecrementRefusedOom` member did not complete. A `Left` member whose
        //    decrement already deleted its blob still counts in `deleted` (and
        //    its bytes in `freed_bytes`): the delete happened.
        let mut completed = 0usize;
        let mut reclaims_completed = 0usize;
        {
            use futures::stream::StreamExt;
            let pending_members = selection
                .candidates
                .iter()
                .map(|member| (member.as_str(), GcClaim::Pending));
            let own_stale = selection
                .own_stale_claims
                .iter()
                .map(|(member, score)| (member.as_str(), GcClaim::Reclaim { score }));
            let member_futures: Vec<_> = pending_members
                .chain(own_stale)
                .enumerate()
                .map(|(index, (member, claim))| async move {
                    let outcome = self.gc_member(maint_pool, index, member, claim).await;
                    (claim, outcome)
                })
                .collect();
            let mut outcomes =
                futures::stream::iter(member_futures).buffer_unordered(GC_MEMBER_CONCURRENCY);
            while let Some((claim, outcome)) = outcomes.next().await {
                state.mark_progress();
                let outcome = outcome?;
                let reclaim = matches!(claim, GcClaim::Reclaim { .. });
                if reclaim
                    && !matches!(
                        outcome,
                        GcMemberOutcome::NotClaimed | GcMemberOutcome::ReclaimRefusedOom
                    )
                {
                    result.reclaimed += 1;
                }
                match outcome {
                    GcMemberOutcome::NotClaimed => {}
                    GcMemberOutcome::ReclaimRefusedOom => {
                        result.oom_skipped += 1;
                        log_oom_skip(state, "the re-claim of a stale GC claim");
                    }
                    GcMemberOutcome::Applied {
                        deleted,
                        freed_bytes,
                    } => {
                        result.processed += 1;
                        completed += 1;
                        reclaims_completed += usize::from(reclaim);
                        if deleted {
                            result.deleted += 1;
                            result.freed_bytes = result.freed_bytes.saturating_add(freed_bytes);
                        }
                    }
                    GcMemberOutcome::Replayed => {
                        result.processed += 1;
                        completed += 1;
                        reclaims_completed += usize::from(reclaim);
                        result.replayed += 1;
                    }
                    GcMemberOutcome::Stale => {
                        result.processed += 1;
                        result.stale += 1;
                    }
                    GcMemberOutcome::Left {
                        deleted,
                        freed_bytes,
                    } => {
                        result.processed += 1;
                        if deleted {
                            result.deleted += 1;
                            result.freed_bytes = result.freed_bytes.saturating_add(freed_bytes);
                        }
                    }
                    GcMemberOutcome::DecrementRefusedOom => {
                        result.processed += 1;
                        result.oom_refused += 1;
                    }
                }
            }
        }

        // 6. One page of the release-ledger audit while a cycle runs (H5).
        result.audit_repaired = self.gc_audit_page(maint_pool, state).await?;
        state.mark_progress();

        // 7. One page of the enqueue-attempt audit while its walk of the same
        //    cycle runs (b.yuo Part R).
        result.attempt_records_retired = self.gc_attempt_audit_page(maint_pool, state).await?;
        state.mark_progress();

        // Run the next pass at once only on progress this pass actually made:
        // a full batch (of pending members, or of own stale claims) with at
        // least one member collected, a full batch of replay records actually
        // torn down, or a full batch of an older build's claims dropped. Never
        // on work merely selected — while a shard is down, the same settled
        // pairs (whose unmarks keep failing) or a batch whose decrements all
        // fail would otherwise be re-run with no sleep.
        let pending_completed = completed - reclaims_completed;
        result.more = (selection.candidates.len() >= GC_BATCH_SIZE && pending_completed > 0)
            || (selection.own_stale_claims.len() >= GC_STALE_CLAIM_BATCH_SIZE
                && reclaims_completed > 0)
            || result.torn_down >= GC_SETTLED_BATCH_SIZE
            || result.foreign_dropped >= GC_STALE_CLAIM_BATCH_SIZE;
        Ok(result)
    }
}

// ── The out-of-memory collection (b.pa1) ────────────────────────────────────

/// Set the first time this process's leader ends an out-of-memory collection
/// `refused`, so the WARN naming the likely cause is logged once per process.
static OOM_COLLECTION_REFUSED_WARNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// The upload hold of one running out-of-memory collection
/// (`le:ext:{gc}:oom:upload_hold`), deleted on every exit: by
/// [`Self::clear`] when the collection ends, or by a `DEL` its drop spawns
/// when the collection's future is dropped first; a leader that dies leaves it
/// to its [`OOM_UPLOAD_HOLD_TTL`]. A plain `DEL`: of two leaders briefly
/// overlapping after a takeover, one may end the other's hold early, which
/// only lets uploads through sooner.
struct OomUploadHold {
    /// Its `DEL`s run on the maintenance pool, through
    /// [`ValkeyClient::acquire_maintenance`] so they hold the
    /// `valkey_pool_in_use{pool="maintenance"}` token like every other GC
    /// command.
    client: ValkeyClient,
    key: String,
    /// The `DEL` returned, so the drop has nothing to do.
    cleared: bool,
}

impl OomUploadHold {
    /// Delete the hold now, so waiting uploads proceed at once. A failed
    /// `DEL` is observed on the maintenance pool's origin (a command timeout
    /// counted) and logged at DEBUG: the hold lapses by its TTL.
    async fn clear(mut self) {
        let deleted: Result<i64, fred::error::Error> = {
            let (maint_pool, _maint_token) = self.client.acquire_maintenance();
            maint_pool.del(&self.key).await
        };
        // Marked only once the `DEL` returned: a clear cancelled mid-flight
        // still leaves the drop's spawned `DEL` behind it.
        self.cleared = true;
        if let Err(error) = deleted {
            FredErrOrigin::client(&self.client, POOL_LABEL_MAINTENANCE).observe_error(&error);
            tracing::debug!(
                error = %error,
                "could not delete the out-of-memory collection's upload hold; it lapses by its TTL"
            );
        }
    }
}

impl Drop for OomUploadHold {
    fn drop(&mut self) {
        if self.cleared {
            return;
        }
        // No runtime (a drop during late shutdown): spawn nothing; the hold
        // lapses on its own expiry.
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let client = self.client.clone();
        let key = std::mem::take(&mut self.key);
        runtime.spawn(async move {
            let deleted: Result<i64, fred::error::Error> = {
                let (maint_pool, _maint_token) = client.acquire_maintenance();
                maint_pool.del(&key).await
            };
            if let Err(error) = deleted {
                FredErrOrigin::client(&client, POOL_LABEL_MAINTENANCE).observe_error(&error);
                tracing::debug!(
                    error = %error,
                    "could not delete the out-of-memory collection's upload hold on drop; it \
                     lapses by its TTL"
                );
            }
        });
    }
}

impl ValkeyExternalValueStore {
    /// Take a pending out-of-memory collection request: `GETDEL
    /// le:ext:{gc}:oom:request` on the maintenance pool (a command the store
    /// never refuses for memory). Returns the requesting replica's label when
    /// there was a request and [`OomCollectionState::may_start`] allows a
    /// collection now; a request the rate limit turns away is consumed all the
    /// same (the replicas still refused re-request it once a second). A failed
    /// read is logged at DEBUG and starts nothing: the pass that follows reads
    /// the same store and reports its own error.
    async fn take_oom_collection_request(&self, state: &GcPassState) -> Option<String> {
        let (maint_pool, _maint_token) = self.client.acquire_maintenance();
        let taken: Result<Option<String>, fred::error::Error> =
            maint_pool.getdel(self.keys.ext_oom_request()).await;
        let requested_by = match taken {
            Ok(Some(requested_by)) => requested_by,
            Ok(None) => return None,
            Err(error) => {
                FredErrOrigin::client(&self.client, POOL_LABEL_MAINTENANCE).observe_error(&error);
                tracing::debug!(
                    error = %error,
                    "could not read the GC's out-of-memory collection request; running an \
                     ordinary pass"
                );
                return None;
            }
        };
        if !state.oom.may_start(tokio::time::Instant::now()) {
            tracing::debug!(
                requested_by = %requested_by,
                "out-of-memory collection requested too soon after the last one; not starting \
                 another yet"
            );
            return None;
        }
        Some(requested_by)
    }

    /// Set the upload hold for one out-of-memory collection: one `EVALSHA` of
    /// `gc_oom_signal.lua` on `le:ext:{gc}:oom:upload_hold`, a fresh random
    /// token, `PX` [`OOM_UPLOAD_HOLD_TTL`]. The guard exists before the send,
    /// so a hold whose reply was lost is deleted too.
    async fn set_oom_upload_hold(&self) -> (OomUploadHold, Result<(), fred::error::Error>) {
        let hold = OomUploadHold {
            client: self.client.clone(),
            key: self.keys.ext_oom_upload_hold(),
            cleared: false,
        };
        let (maint_pool, _maint_token) = self.client.acquire_maintenance();
        let set: Result<i64, fred::error::Error> = GC_OOM_SIGNAL_SCRIPT
            .evalsha_with_reload(
                maint_pool.next(),
                vec![hold.key.clone()],
                vec![
                    format!("{:032x}", rand::random::<u128>()),
                    duration_ms(OOM_UPLOAD_HOLD_TTL).to_string(),
                ],
            )
            .await;
        (hold, set.map(|_| ()))
    }

    /// One out-of-memory collection (module header): set the upload hold
    /// (only when the release budget's gate exists), run passes under
    /// [`GcPolicy::OOM_COLLECTION`] while a pass reports more work and the
    /// collection is younger than [`OOM_COLLECTION_MAX_DURATION`], delete the
    /// hold, then count the collection on
    /// `live_edit_gc_oom_collections_total{outcome}` (and the bytes it freed)
    /// and log it on the one per-collection WARN (rate-limited), which also
    /// carries a hold that could not be set (`held_uploads=false`,
    /// `hold_error`); the collection then runs without it. Returns the passes'
    /// combined result, or the error of the pass that ended it.
    ///
    /// Deletions are counted from completed passes only: a pass that ends in
    /// an error returns no counts. Outcomes, in order: `freed` (a completed
    /// pass deleted something); `refused` (nothing
    /// deleted, and the store refused a decrement or the hold `-OOM`: it did
    /// not run the scripts past their opening write on a full store; also
    /// logged once per process); `failed` (no completed pass deleted anything,
    /// and a pass ended in an error);
    /// `nothing_to_collect` (no member claimed and no step skipped);
    /// `freed_nothing` (otherwise). After `freed` or `failed` the next
    /// collection may start a second later; after any other, 30 s later
    /// ([`OomCollectionState::may_start`]).
    async fn run_oom_collection(
        &self,
        requested_by: &str,
        state: &mut GcPassState,
    ) -> Result<GcBatchResult, ServiceError> {
        let started = tokio::time::Instant::now();
        let origin = FredErrOrigin::client(&self.client, POOL_LABEL_MAINTENANCE);

        // 1. The upload hold, while a release budget gates uploads at all. A
        //    hold that cannot be set is not logged on its own: the collection
        //    runs without it, and the per-collection WARN (rate-limited)
        //    carries the error as `hold_error`, beside `held_uploads=false`.
        let mut hold_error: Option<fred::error::Error> = None;
        let hold = if self.ext_release_gate.is_some() {
            let (hold, set) = self.set_oom_upload_hold().await;
            match set {
                Ok(()) => Some(hold),
                Err(error) => {
                    origin.observe_error(&error);
                    hold_error = Some(error);
                    // Dropped here: its spawned `DEL` covers a set whose
                    // reply was lost.
                    drop(hold);
                    None
                }
            }
        } else {
            None
        };
        let held_uploads = hold.is_some();
        let hold_refused = hold_error.as_ref().is_some_and(is_store_out_of_memory);

        // 2. Passes at any age, while there is more and time is left.
        let mut total = GcBatchResult::default();
        let mut passes: u64 = 0;
        let mut cut_short = false;
        let mut failure: Option<ServiceError> = None;
        loop {
            let pass = self.gc_pass(&GcPolicy::OOM_COLLECTION, state).await;
            passes += 1;
            match pass {
                Ok(pass) => {
                    let more = pass.more;
                    accumulate_gc_batch(&mut total, pass);
                    if !more {
                        break;
                    }
                    if started.elapsed() >= OOM_COLLECTION_MAX_DURATION {
                        cut_short = true;
                        break;
                    }
                }
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }

        // 3. Let uploads through at once.
        if let Some(hold) = hold {
            hold.clear().await;
        }

        // 4. Count and log it.
        let freed = total.deleted > 0 || total.freed_bytes > 0;
        let outcome = if freed {
            GC_OOM_COLLECTION_OUTCOME_FREED
        } else if total.oom_refused > 0 || hold_refused {
            GC_OOM_COLLECTION_OUTCOME_REFUSED
        } else if failure.is_some() {
            GC_OOM_COLLECTION_OUTCOME_FAILED
        } else if total.processed == 0 && total.oom_skipped == 0 {
            GC_OOM_COLLECTION_OUTCOME_NOTHING_TO_COLLECT
        } else {
            GC_OOM_COLLECTION_OUTCOME_FREED_NOTHING
        };
        // A collection that freed something, or that a pass error cut off
        // (a `{gc}` error is no evidence the store cannot be helped), may be
        // followed by the next one a second later; any other outcome found
        // nothing to free and backs off (`OomCollectionState::may_start`).
        let retry_soon = freed || outcome == GC_OOM_COLLECTION_OUTCOME_FAILED;
        state.oom.last = Some((tokio::time::Instant::now(), retry_soon));
        if let Some(m) = self.metrics.as_ref() {
            m.record_gc_oom_collection(outcome, total.freed_bytes);
        }
        self.log_oom_collection(
            state,
            &OomCollectionReport {
                outcome,
                requested_by,
                passes,
                result: &total,
                duration: started.elapsed(),
                held_uploads,
                hold_error: hold_error.as_ref(),
                cut_short,
                failure: failure.as_ref(),
            },
        );
        match failure {
            Some(error) => Err(error),
            None => {
                total.more = cut_short;
                Ok(total)
            }
        }
    }

    /// The per-collection WARN, at most once per
    /// [`OOM_COLLECTION_LOG_INTERVAL`] (the lines held back are counted on the
    /// next one as `suppressed`), and, the first time this process sees a
    /// `refused` collection, the WARN naming the likely cause.
    fn log_oom_collection(&self, state: &mut GcPassState, report: &OomCollectionReport<'_>) {
        if report.outcome == GC_OOM_COLLECTION_OUTCOME_REFUSED
            && !OOM_COLLECTION_REFUSED_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            tracing::warn!(
                "the backing store refused the GC's own clean-up scripts for lack of memory: it \
                 did not run a script past its first write on a full store, so the service \
                 cannot free released large values while the store is full. Writes there stay \
                 refused until an operator raises maxmemory or frees memory. Logged once per \
                 process; live_edit_gc_oom_collections_total{{outcome=\"refused\"}} counts every \
                 such collection"
            );
        }
        if !state.oom.log_gate.allow() {
            state.oom.suppressed += 1;
            return;
        }
        let suppressed = std::mem::take(&mut state.oom.suppressed);
        let (replica_id, lock_value) = state.leader.as_ref().map_or_else(
            || (String::new(), String::new()),
            |leader| (leader.service_label.clone(), leader.gc_lock_value()),
        );
        let what = match report.outcome {
            GC_OOM_COLLECTION_OUTCOME_FREED => {
                "the backing store refused a write for lack of memory; the GC collected released \
                 large values at once"
            }
            GC_OOM_COLLECTION_OUTCOME_REFUSED => {
                "the backing store refused a write for lack of memory; the GC tried to collect \
                 released large values at once, but the store also refused the collection's own \
                 scripts for lack of memory, so it freed nothing"
            }
            GC_OOM_COLLECTION_OUTCOME_FAILED => {
                "the backing store refused a write for lack of memory; the GC's collection of \
                 released large values failed and freed nothing; another runs on the next \
                 request a second or more after this one"
            }
            GC_OOM_COLLECTION_OUTCOME_NOTHING_TO_COLLECT => {
                "the backing store refused a write for lack of memory; the GC found no released \
                 large values it could collect: none are owed, or a reader's read hold keeps \
                 them until it ends, or releases refused on the full {gc} shard wait for the \
                 reference-count reconciler (about 1-2 h). If writes stay refused, the store is \
                 full of data still in use and needs more memory"
            }
            // `freed_nothing` with skipped steps: the owed releases are stuck
            // behind refused re-claims or stamps, not live data.
            _ if report.result.oom_skipped > 0 => {
                "the backing store refused a write for lack of memory; the GC collected released \
                 large values at once and freed nothing: the store also refused re-claims or age \
                 stamps on its full {gc} shard, so owed releases wait for room there"
            }
            _ => {
                "the backing store refused a write for lack of memory; the GC collected released \
                 large values at once and freed nothing: each released value it reached is still \
                 referenced or was already settled. If writes stay refused, the store is full of \
                 data still in use and needs more memory"
            }
        };
        tracing::warn!(
            outcome = report.outcome,
            requested_by = %report.requested_by,
            passes = report.passes,
            members = report.result.processed,
            deleted = report.result.deleted,
            oom_skipped = report.result.oom_skipped,
            freed_bytes = report.result.freed_bytes,
            duration_ms = duration_ms(report.duration),
            held_uploads = report.held_uploads,
            hold_error = report.hold_error.map(tracing::field::display),
            cut_short = report.cut_short,
            error = report.failure.map(tracing::field::display),
            suppressed,
            replica_id = %replica_id,
            lock_value = %lock_value,
            "{what}"
        );
    }
}

/// What [`ValkeyExternalValueStore::log_oom_collection`] reports about one
/// out-of-memory collection.
struct OomCollectionReport<'a> {
    outcome: &'static str,
    requested_by: &'a str,
    passes: u64,
    result: &'a GcBatchResult,
    duration: std::time::Duration,
    /// The upload hold was set and stood for the collection.
    held_uploads: bool,
    /// Why the upload hold could not be set, when it could not; the
    /// collection then ran without it (`held_uploads` is false).
    hold_error: Option<&'a fred::error::Error>,
    cut_short: bool,
    failure: Option<&'a ServiceError>,
}

/// Add one pass's counts into an out-of-memory collection's combined result;
/// the ledger and `more` are the latest pass's.
fn accumulate_gc_batch(total: &mut GcBatchResult, pass: GcBatchResult) {
    total.processed += pass.processed;
    total.deleted += pass.deleted;
    total.freed_bytes = total.freed_bytes.saturating_add(pass.freed_bytes);
    total.oom_refused += pass.oom_refused;
    total.oom_skipped += pass.oom_skipped;
    total.replayed += pass.replayed;
    total.stale += pass.stale;
    total.torn_down += pass.torn_down;
    total.reclaimed += pass.reclaimed;
    total.foreign_dropped += pass.foreign_dropped;
    total.stamped += pass.stamped;
    total.audit_repaired += pass.audit_repaired;
    total.attempt_records_retired += pass.attempt_records_retired;
    total.more = pass.more;
    total.ledger = pass.ledger;
}

// ── GC pass steps (b.yuo) ───────────────────────────────────────────────────

/// What one member's claim → decrement → complete came to in a GC pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GcMemberOutcome {
    /// Not claimed: another sweeper owns it, it already left `pending` (or,
    /// for a re-claim, its claim changed since the selection read it), or a
    /// transient error refused the claim. Nothing changed for it.
    NotClaimed,
    /// One of this build's own stale claims whose re-claim the store refused
    /// `-OOM` (b.pa1: a full `{gc}` shard). Nothing changed for it; the pass
    /// goes on, and a later pass re-claims it.
    ReclaimRefusedOom,
    /// Its decrement applied now; `deleted` when that took `ref_count` to zero
    /// (or below) and deleted the blob, whose length `freed_bytes` is (0
    /// otherwise).
    Applied { deleted: bool, freed_bytes: u64 },
    /// Its decrement had already applied (a replay); nothing was decremented.
    Replayed,
    /// Its decrement was refused `stale` by the claim-age fence; the member
    /// stays in `inflight` until a later pass re-claims it.
    Stale,
    /// Claimed, but its decrement or completion failed on a transient error,
    /// or its decrement failed on its own hash's keys; the member stays in
    /// `inflight` until a later pass re-claims it. `deleted` when the decrement
    /// had already applied and deleted the blob before the completion failed:
    /// the delete happened and is counted (with its `freed_bytes`), and the
    /// re-claim's decrement answers `replayed`, which never counts it again.
    Left { deleted: bool, freed_bytes: u64 },
    /// Claimed, but the store refused its decrement `-OOM` (b.pa1): it did not
    /// run the decrement past its opening write on a full store. Counted on
    /// [`GcBatchResult::oom_refused`]; otherwise as [`Self::Left`] with
    /// nothing deleted.
    DecrementRefusedOom,
}

/// Whether a fred error is the store refusing a write `-OOM`
/// (`storage::client::store_write_refusal`), bare or after an `EXEC`'s
/// discard prefix.
fn is_store_out_of_memory(e: &fred::error::Error) -> bool {
    store_write_refusal(e) == Some(StoreWriteRefusal::OutOfMemory)
}

/// Log, at most once per [`GC_OOM_SKIP_LOG_INTERVAL`] per leadership, that a
/// pass skipped a `{gc}` step the store refused `-OOM` (b.pa1). `step` names
/// it.
fn log_oom_skip(state: &GcPassState, step: &'static str) {
    if state.oom.skip_log_gate.allow() {
        tracing::warn!(
            step,
            "the backing store refused a GC step for lack of memory (its {{gc}} shard is \
             full); skipped it and went on with the pass, which still collects released \
             external values. A later pass retries the step"
        );
    }
}

/// How a GC pass takes ownership of one member before its decrement.
#[derive(Debug, Clone, Copy)]
enum GcClaim<'a> {
    /// A member the selection picked from `pending`: claimed by
    /// `gc_claim_pending.lua`.
    Pending,
    /// One of this build's own `inflight` claims older than the re-claim age:
    /// re-claimed in place by `gc_reclaim_stale.lua`, only while its score is
    /// still `score`, the one the selection read (b.1jt).
    Reclaim { score: &'a str },
}

/// `gc_select.lua`'s reply, decoded.
#[derive(Debug, Clone, Default)]
struct GcSelection {
    /// The `{gc}` primary's `TIME`, ms.
    now_ms: u64,
    /// Live read hold fields (`le:ext:{gc}:read_hold`).
    holds_live: u64,
    /// `pending:total`; a negative total (a ledger that lost track) reads 0.
    pending_total: u64,
    /// `SCARD pending`.
    pending_count: u64,
    /// `ZCARD inflight`.
    inflight_count: u64,
    /// The oldest `pending:age` score.
    oldest_age_ms: Option<u64>,
    /// `ZCARD settled`.
    settled_count: u64,
    /// The total was at or above the pressure threshold.
    under_pressure: bool,
    /// Settled `(member, score)` pairs due for teardown.
    settled: Vec<(String, String)>,
    /// Members to claim, oldest first.
    candidates: Vec<String>,
    /// Sampled `pending` members with no `pending:age` entry, to stamp.
    unaged: Vec<String>,
    /// This build's own `inflight` claims older than the re-claim age, as
    /// `(member, score)` with the score as the store reported it.
    own_stale_claims: Vec<(String, String)>,
    /// An older build's `inflight` claims (scored below
    /// [`GC_CLAIM_SCORE_OFFSET_MS`]) older than the decrement fence, as
    /// `(member, score)` with the score as the store reported it.
    foreign_claims: Vec<(String, String)>,
}

impl GcSelection {
    /// Decode `gc_select.lua`'s flat reply (see the script's "Return value").
    fn decode(reply: Vec<fred::types::Value>) -> Result<Self, ServiceError> {
        // Not a fred error: the reply arrived and does not have the script's
        // shape, so a retry would read the same shape. It ends the pass, and
        // the loop counts it as `internal`.
        fn bad(what: &str) -> ServiceError {
            ServiceError::Internal(["gc_select reply malformed: ", what].concat())
        }
        let mut it = reply.into_iter();
        let mut int = |what: &str| -> Result<i64, ServiceError> {
            it.next().and_then(|v| v.as_i64()).ok_or_else(|| bad(what))
        };
        let now_ms = int("now_ms")?;
        let _hold_start_ms = int("hold_start_ms")?;
        let holds_live = int("holds_live")?;
        let pending_total = int("pending_total")?;
        let pending_count = int("pending_count")?;
        let inflight_count = int("inflight_count")?;
        let oldest_age_ms = int("oldest_age_ms")?;
        let settled_count = int("settled_count")?;
        let under_pressure = int("under_pressure")?;
        let settled_n = int("settled_n")?;
        let candidates_n = int("candidates_n")?;
        let unaged_n = int("unaged_n")?;
        let own_stale_n = int("own_stale_n")?;
        let foreign_n = int("foreign_n")?;
        let non_negative = |n: i64| u64::try_from(n).unwrap_or(0);
        let present = |n: i64| u64::try_from(n).ok();
        // `width` strings per entry, `n` entries: a member, or a
        // `(member, score)` pair.
        let mut entries = |n: i64, width: usize, what: &str| -> Result<Vec<String>, ServiceError> {
            let count = usize::try_from(n.max(0)).unwrap_or(0) * width;
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                out.push(
                    it.next()
                        .and_then(fred::types::Value::into_string)
                        .ok_or_else(|| bad(what))?,
                );
            }
            Ok(out)
        };
        let into_pairs = |flat: Vec<String>| -> Vec<(String, String)> {
            let mut out = Vec::with_capacity(flat.len() / 2);
            let mut flat = flat.into_iter();
            while let (Some(member), Some(score)) = (flat.next(), flat.next()) {
                out.push((member, score));
            }
            out
        };
        let settled = into_pairs(entries(settled_n, 2, "settled pair")?);
        let candidates = entries(candidates_n, 1, "candidate")?;
        let unaged = entries(unaged_n, 1, "un-aged member")?;
        let own_stale_claims = into_pairs(entries(own_stale_n, 2, "own stale claim")?);
        let foreign_claims = into_pairs(entries(foreign_n, 2, "foreign claim")?);
        Ok(Self {
            now_ms: non_negative(now_ms),
            holds_live: non_negative(holds_live),
            pending_total: non_negative(pending_total),
            pending_count: non_negative(pending_count),
            inflight_count: non_negative(inflight_count),
            oldest_age_ms: present(oldest_age_ms),
            settled_count: non_negative(settled_count),
            under_pressure: under_pressure == 1,
            settled,
            candidates,
            unaged,
            own_stale_claims,
            foreign_claims,
        })
    }

    /// The ledger snapshot this selection read, with the time until the oldest
    /// member falls due under the floor that applied to this pass.
    fn snapshot(&self, policy: &GcPolicy) -> GcLedgerSnapshot {
        let floor_ms = duration_ms(if self.under_pressure {
            policy.pressure_min_age
        } else {
            policy.min_age
        });
        let oldest_age = self
            .oldest_age_ms
            .map(|age| std::time::Duration::from_millis(self.now_ms.saturating_sub(age)));
        let next_due_in = self.oldest_age_ms.and_then(|age| {
            let due_at = age.saturating_add(floor_ms);
            (due_at > self.now_ms).then(|| std::time::Duration::from_millis(due_at - self.now_ms))
        });
        GcLedgerSnapshot {
            pending_bytes: self.pending_total,
            pending_count: self.pending_count.saturating_add(self.inflight_count),
            oldest_age,
            settled_count: self.settled_count,
            read_holds: self.holds_live,
            next_due_in,
        }
    }
}

impl ValkeyExternalValueStore {
    /// The pass's selection: one `EVALSHA` of `gc_select.lua` on `{gc}`.
    async fn gc_select(
        &self,
        pool: &fred::clients::Pool,
        policy: &GcPolicy,
    ) -> Result<GcSelection, ServiceError> {
        let keys = vec![
            self.keys.ext_pending(),
            self.keys.ext_inflight(),
            self.keys.ext_pending_age(),
            self.keys.ext_pending_total(),
            self.keys.ext_settled(),
            self.keys.ext_read_hold(),
            self.keys.ext_pending_bytes(),
        ];
        let argv = vec![
            duration_ms(policy.min_age).to_string(),
            duration_ms(policy.pressure_min_age).to_string(),
            self.pressure_threshold_bytes().to_string(),
            duration_ms(policy.replay_record_retention).to_string(),
            if policy.hold_clip { "1" } else { "0" }.to_owned(),
            self.hold_margin_ms().to_string(),
            GC_BATCH_SIZE.to_string(),
            GC_SETTLED_BATCH_SIZE.to_string(),
            GC_UNAGED_SAMPLE.to_string(),
            GC_CLAIM_SCORE_OFFSET_MS.to_string(),
            duration_ms(policy.reclaim_after).to_string(),
            self.decrement_fence_ms().to_string(),
            GC_STALE_CLAIM_BATCH_SIZE.to_string(),
        ];
        let reply: Vec<fred::types::Value> = GC_SELECT_SCRIPT
            .evalsha_with_reload(pool.next(), keys, argv)
            .await
            .map_err(|e| {
                map_transient_fred_err(
                    FredErrOrigin::client(&self.client, POOL_LABEL_MAINTENANCE),
                    "EVALSHA gc_select",
                    e,
                )
            })?;
        GcSelection::decode(reply)
    }

    /// Stamp an age on each sampled un-aged `pending` member (b.1jt): one
    /// `EVALSHA` of `gc_stamp_unaged.lua` on `{gc}`, which `ZADD NX`es the
    /// `{gc}` primary's `TIME` into `pending:age` for every member still in
    /// `pending` with no age. Ticks
    /// `live_edit_gc_older_build_total{event="release_stamped"}` per member
    /// stamped — an older build's bare `SADD`, since every enqueue this build
    /// makes records the age with the member. Returns how many members were
    /// stamped, or `None` when the store refused the stamp `-OOM` (b.pa1: a
    /// full `{gc}` shard; noted on the client's store out-of-memory signal),
    /// which the pass skips. Any other `{gc}`-slot error ends the pass.
    async fn stamp_unaged(
        &self,
        pool: &fred::clients::Pool,
        unaged: &[String],
    ) -> Result<Option<usize>, ServiceError> {
        if unaged.is_empty() {
            return Ok(Some(0));
        }
        let origin = FredErrOrigin::client(&self.client, POOL_LABEL_MAINTENANCE);
        let stamped: i64 = match GC_STAMP_UNAGED_SCRIPT
            .evalsha_with_reload(
                pool.next(),
                vec![self.keys.ext_pending(), self.keys.ext_pending_age()],
                unaged.to_vec(),
            )
            .await
        {
            Ok(stamped) => stamped,
            Err(e) if is_store_out_of_memory(&e) => {
                origin.observe_error(&e);
                return Ok(None);
            }
            Err(e) => return Err(map_transient_fred_err(origin, "EVALSHA gc_stamp_unaged", e)),
        };
        let stamped = usize::try_from(stamped.max(0)).unwrap_or(0);
        if stamped > 0 {
            if let Some(m) = self.metrics.as_ref() {
                m.inc_gc_older_build_by(GC_OLDER_BUILD_EVENT_RELEASE_STAMPED, stamped as u64);
            }
            // DEBUG, not INFO: an older build beside this one enqueues
            // continuously, so this would log on most passes for the whole
            // mixed window; the counter carries the signal.
            tracing::debug!(
                stamped,
                "stamped an age on pending external-value releases that had none (an older \
                 build's enqueue); each is collected only once it is past the age floor from \
                 the stamp"
            );
        }
        Ok(Some(stamped))
    }

    /// Drop an older build's stranded `inflight` claims the selection found
    /// (b.1jt): one `EVALSHA` of `gc_drop_foreign_claims.lua` on `{gc}`, which,
    /// for each `(member, score)` whose score is still the one read, is below
    /// [`GC_CLAIM_SCORE_OFFSET_MS`] and is older than the decrement fence,
    /// removes the claim and settles the member in the release ledger without
    /// any decrement. The older build's own decrement either applied (then
    /// nothing more is owed) or did not (then `ref_count` is one too high,
    /// which the reference-count reconciler repairs, unless it is turned off
    /// with `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0`, when the leak is
    /// permanent): leak, never over-release. Logs one WARN naming the dropped
    /// members and ticks
    /// `live_edit_gc_older_build_total{event="claim_dropped"}` per member. A
    /// `{gc}`-slot error ends the pass. Returns how many claims were dropped.
    async fn drop_foreign_claims(
        &self,
        pool: &fred::clients::Pool,
        claims: &[(String, String)],
    ) -> Result<usize, ServiceError> {
        if claims.is_empty() {
            return Ok(0);
        }
        let mut argv: Vec<String> = Vec::with_capacity(2 * claims.len() + 2);
        argv.push(GC_CLAIM_SCORE_OFFSET_MS.to_string());
        argv.push(self.decrement_fence_ms().to_string());
        for (member, score) in claims {
            argv.push(member.clone());
            argv.push(score.clone());
        }
        let dropped: Vec<String> = GC_DROP_FOREIGN_CLAIMS_SCRIPT
            .evalsha_with_reload(
                pool.next(),
                vec![
                    self.keys.ext_inflight(),
                    self.keys.ext_pending(),
                    self.keys.ext_pending_age(),
                    self.keys.ext_pending_bytes(),
                    self.keys.ext_pending_total(),
                    self.keys.ext_settled(),
                    // Never written: makes the store refuse the script while
                    // the `{gc}` slot migrates (see the script's "Migration
                    // sentinel").
                    self.keys.migration_sentinel("gc"),
                ],
                argv,
            )
            .await
            .map_err(|e| {
                map_transient_fred_err(
                    FredErrOrigin::client(&self.client, POOL_LABEL_MAINTENANCE),
                    "EVALSHA gc_drop_foreign_claims",
                    e,
                )
            })?;
        if !dropped.is_empty() {
            if let Some(m) = self.metrics.as_ref() {
                m.inc_gc_older_build_by(GC_OLDER_BUILD_EVENT_CLAIM_DROPPED, dropped.len() as u64);
            }
            tracing::warn!(
                dropped = dropped.len(),
                members = %dropped.join(","),
                fence_ms = self.decrement_fence_ms(),
                "dropped external-value GC claims an older build left in inflight, without \
                 decrementing them: this build never decrements a claim it did not make. If the \
                 older build's decrement had not applied, the reference-count reconciler \
                 repairs the leaked reference, unless the reconciler is turned off \
                 (LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0), in which case the leak is permanent"
            );
        }
        Ok(dropped.len())
    }

    /// Tear down the replay records of the settled `(member, score)` pairs
    /// the selection returned (H4): `gc_unmark.lua` on each member's hash slot
    /// (at most [`GC_MEMBER_CONCURRENCY`] at a time), then one
    /// `gc_settled_retire.lua` on `{gc}` for the pairs whose unmark succeeded.
    ///
    /// A failed unmark leaves its pair for the next pass (WARN; it is on the
    /// member's own slot, so it does not end the pass); a failed retire is a
    /// `{gc}`-slot error and ends the pass. Returns how many settle entries
    /// were retired. Marks `progress` as each unmark returns, so a long
    /// teardown keeps the lease refresh ticker renewing (b.1jt).
    async fn tear_down_settled(
        &self,
        pool: &fred::clients::Pool,
        settled: &[(String, String)],
        progress: Option<&GcProgress>,
    ) -> Result<usize, ServiceError> {
        if settled.is_empty() {
            return Ok(0);
        }
        use futures::stream::StreamExt;
        let unmarks: Vec<_> = settled
            .iter()
            .map(|(member, score)| {
                let applied_key = self.keys.ext_applied(pending_member_hash_hex(member));
                async move {
                    let unmarked: Result<i64, fred::error::Error> = GC_UNMARK_SCRIPT
                        .evalsha_with_reload(
                            pool.next(),
                            vec![applied_key],
                            vec![member.clone(), score.clone()],
                        )
                        .await;
                    match unmarked {
                        Ok(_) => Some((member, score)),
                        Err(e) => {
                            let mapped = map_transient_fred_err(
                                FredErrOrigin::client(&self.client, POOL_LABEL_MAINTENANCE),
                                "EVALSHA gc_unmark",
                                e,
                            );
                            tracing::warn!(
                                member = %member,
                                error = %mapped,
                                "could not tear down a settled GC replay record; leaving it for \
                                 the next pass"
                            );
                            None
                        }
                    }
                }
            })
            .collect();
        let retire: Vec<(&String, &String)> = futures::stream::iter(unmarks)
            .buffer_unordered(GC_MEMBER_CONCURRENCY)
            .inspect(|_| {
                if let Some(progress) = progress {
                    progress.mark();
                }
            })
            .filter_map(|pair| async move { pair })
            .collect()
            .await;
        if retire.is_empty() {
            return Ok(0);
        }
        let argv: Vec<String> = retire
            .into_iter()
            .flat_map(|(member, score)| [member.clone(), score.clone()])
            .collect();
        let retired: i64 = GC_SETTLED_RETIRE_SCRIPT
            .evalsha_with_reload(pool.next(), vec![self.keys.ext_settled()], argv)
            .await
            .map_err(|e| {
                map_transient_fred_err(
                    FredErrOrigin::client(&self.client, POOL_LABEL_MAINTENANCE),
                    "EVALSHA gc_settled_retire",
                    e,
                )
            })?;
        Ok(retired.max(0) as usize)
    }

    /// Claim, decrement at most once, and complete one member of the pass's
    /// work list (`index` is its position there — the selected pending
    /// members, then the own stale claims — for the test-only fault injector's
    /// visit index `3 * index + site`).
    ///
    /// - **Claim** — for a [`GcClaim::Pending`] member `gc_claim_pending.lua`,
    ///   for a [`GcClaim::Reclaim`] one `gc_reclaim_stale.lua` (b.1jt), both on
    ///   `{gc}`: a 0 reply, or a transient error, is
    ///   [`GcMemberOutcome::NotClaimed`]; a re-claim the store refuses `-OOM`
    ///   is [`GcMemberOutcome::ReclaimRefusedOom`] (b.pa1); any other
    ///   non-transient error ends the pass. Both reply with the (unencoded)
    ///   claim time the decrement fences on.
    /// - **Decrement** (`gc_decrement_and_collect.lua`, the member's hash
    ///   slot): `decremented` / `deleted` / `deleted_underflow` /
    ///   `replayed` go on to the completion; `stale` stays in `inflight` with a
    ///   WARN. Any error leaves the member in `inflight` for a later pass to
    ///   re-claim: a
    ///   transient one with a WARN, a non-transient one — this member's own
    ///   keys, in practice a foreign value of the wrong type — with an ERROR
    ///   naming the member and `gc_sweep_batch_errors_total{kind}`, ending only
    ///   this member, because the pass selects oldest first and one member that
    ///   always fails would otherwise end every pass (H7). A `-OOM` refusal is
    ///   one of the non-transient ones, answered
    ///   [`GcMemberOutcome::DecrementRefusedOom`] so the out-of-memory
    ///   collection can tell it apart (b.pa1).
    /// - **Complete** (`gc_complete.lua`, `{gc}`): a transient error leaves the
    ///   member in `inflight` (its decrement is recorded, so its re-claim's
    ///   decrement answers `replayed`); a non-transient one ends the pass.
    ///
    /// Ticks `live_edit_gc_decrement_total{outcome}` once per decided
    /// decrement.
    async fn gc_member(
        &self,
        pool: &fred::clients::Pool,
        index: usize,
        member: &str,
        claim: GcClaim<'_>,
    ) -> Result<GcMemberOutcome, ServiceError> {
        let hash_hex = pending_member_hash_hex(member);
        let origin = FredErrOrigin::client(&self.client, POOL_LABEL_MAINTENANCE);

        // Site 0: claim, or re-claim in place.
        let (claimed, claim_script) = match claim {
            GcClaim::Pending => {
                let claim_keys = vec![
                    self.keys.ext_pending(),
                    self.keys.ext_inflight(),
                    self.keys.ext_pending_age(),
                    self.keys.ext_pending_bytes(),
                    self.keys.ext_pending_total(),
                    self.keys.ext_settled(),
                ];
                let argv = vec![member.to_owned()];
                let claimed = self
                    .gc_round_trip(3 * index, || {
                        GC_CLAIM_PENDING_SCRIPT.evalsha_with_reload::<i64, _, _>(
                            pool.next(),
                            claim_keys.clone(),
                            argv.clone(),
                        )
                    })
                    .await;
                (claimed, "EVALSHA gc_claim_pending")
            }
            GcClaim::Reclaim { score } => {
                let argv = vec![
                    member.to_owned(),
                    score.to_owned(),
                    GC_CLAIM_SCORE_OFFSET_MS.to_string(),
                ];
                let claimed = self
                    .gc_round_trip(3 * index, || {
                        GC_RECLAIM_STALE_SCRIPT.evalsha_with_reload::<i64, _, _>(
                            pool.next(),
                            vec![self.keys.ext_inflight()],
                            argv.clone(),
                        )
                    })
                    .await;
                (claimed, "EVALSHA gc_reclaim_stale")
            }
        };
        let claim_ms = match claimed {
            Ok(ms) if ms > 0 => ms,
            Ok(_) => return Ok(GcMemberOutcome::NotClaimed),
            // A re-claim's first write is a `ZADD XX`, which a full `{gc}`
            // shard refuses: skip this member, not the pass, which still
            // collects the rest (b.pa1). Noted on the store out-of-memory
            // signal like any other `-OOM`.
            Err(e) if matches!(claim, GcClaim::Reclaim { .. }) && is_store_out_of_memory(&e) => {
                origin.observe_error(&e);
                return Ok(GcMemberOutcome::ReclaimRefusedOom);
            }
            Err(e) => {
                let mapped = map_transient_fred_err(origin, claim_script, e);
                if is_transient_aborted(&mapped) {
                    tracing::warn!(
                        hash = %hash_hex,
                        member = %member,
                        error = %mapped,
                        script = claim_script,
                        "transient claim failure during a GC pass; skipping the member this pass"
                    );
                    return Ok(GcMemberOutcome::NotClaimed);
                }
                return Err(mapped);
            }
        };
        if matches!(claim, GcClaim::Reclaim { .. }) {
            tracing::debug!(
                hash = %hash_hex,
                member = %member,
                "re-claimed a stale GC claim in place; its decrement is decided once through the \
                 replay record"
            );
        }

        // Site 1: the at-most-once decrement, on the member's hash slot.
        //
        // Sent with fred's default write attempts, deliberately NOT through
        // `lua::evalsha_once_with_reload`. fred re-sends an in-flight command
        // after its connection is lost, so this script can run twice; the second
        // run finds the member in the replay record and answers `replayed`
        // without decrementing, and a copy that runs later than the fence after
        // the claim answers `stale` and writes nothing. Keeping the retries
        // strands fewer members in `inflight` than one attempt would, since
        // every connection blip would otherwise leave the member to be
        // re-claimed a re-claim age later (b.yuo H8).
        let decrement_keys = vec![
            self.keys.ext_meta(hash_hex),
            self.keys.ext_data(hash_hex),
            self.keys.ext_applied(hash_hex),
        ];
        let decrement_argv = vec![
            member.to_owned(),
            claim_ms.to_string(),
            self.decrement_fence_ms().to_string(),
        ];
        let decrement = self
            .gc_round_trip(3 * index + 1, || {
                GC_DECREMENT_AND_COLLECT_SCRIPT.evalsha_with_reload::<(String, i64, i64), _, _>(
                    pool.next(),
                    decrement_keys.clone(),
                    decrement_argv.clone(),
                )
            })
            .await;
        let (action, n, freed_bytes) = match decrement {
            Ok(reply) => reply,
            Err(e) => {
                // The script opens every writing branch with a write that
                // cannot grow memory, after which Valkey runs the rest on a
                // full store; an `-OOM` here means the store did not, so the
                // out-of-memory collection cannot free this member's blob
                // (`GcMemberOutcome::DecrementRefusedOom`). Mapped, counted and
                // logged below like any other error on its own keys.
                let refused_oom = is_store_out_of_memory(&e);
                let mapped = map_transient_fred_err(origin, "EVALSHA gc_decrement_and_collect", e);
                if is_transient_aborted(&mapped) {
                    tracing::warn!(
                        hash = %hash_hex,
                        member = %member,
                        error = %mapped,
                        "transient gc_decrement_and_collect failure during a GC pass; leaving the \
                         member in inflight for a later pass to re-claim"
                    );
                } else {
                    let kind = service_error_kind_label(&mapped);
                    if let Some(m) = self.metrics.as_ref() {
                        m.inc_gc_sweep_batch_error(kind);
                    }
                    tracing::error!(
                        hash = %hash_hex,
                        member = %member,
                        error = %mapped,
                        kind = %kind,
                        "GC decrement of one release member failed on its own keys; leaving it in \
                         inflight for a later pass to re-claim and continuing the pass"
                    );
                }
                return Ok(if refused_oom {
                    GcMemberOutcome::DecrementRefusedOom
                } else {
                    GcMemberOutcome::Left {
                        deleted: false,
                        freed_bytes: 0,
                    }
                });
            }
        };
        let freed_bytes = u64::try_from(freed_bytes).unwrap_or(0);
        let outcome = match action.as_str() {
            "decremented" => GcMemberOutcome::Applied {
                deleted: false,
                freed_bytes: 0,
            },
            "deleted" => {
                tracing::info!(
                    hash = %hash_hex,
                    ref_count = n,
                    bytes = freed_bytes,
                    "GC deleted external value"
                );
                GcMemberOutcome::Applied {
                    deleted: true,
                    freed_bytes,
                }
            }
            "deleted_underflow" => {
                // A decrement with no increment behind it: an older build's
                // replay, a legacy bare member, or a bug. Replays answer
                // `replayed` now, so the baseline is zero.
                tracing::warn!(
                    hash = %hash_hex,
                    member = %member,
                    ref_count = n,
                    bytes = freed_bytes,
                    "GC deleted external value with ref_count underflow"
                );
                if let Some(m) = self.metrics.as_ref() {
                    m.inc_gc_underflow();
                }
                GcMemberOutcome::Applied {
                    deleted: true,
                    freed_bytes,
                }
            }
            "replayed" => {
                tracing::debug!(
                    hash = %hash_hex,
                    member = %member,
                    ref_count = n,
                    "GC decrement already applied for this release member (replay); not \
                     decrementing again"
                );
                GcMemberOutcome::Replayed
            }
            "stale" => {
                if let Some(m) = self.metrics.as_ref() {
                    m.inc_gc_decrement(GC_DECREMENT_OUTCOME_STALE);
                }
                tracing::warn!(
                    hash = %hash_hex,
                    member = %member,
                    claim_age_ms = n,
                    fence_ms = self.decrement_fence_ms(),
                    "GC decrement refused: it executed more than the fence after its claim (a \
                     stalled store, or this hash's primary clock running ahead of the {{gc}} \
                     primary's); the member stays in inflight for a later pass to re-claim"
                );
                return Ok(GcMemberOutcome::Stale);
            }
            other => {
                if let Some(m) = self.metrics.as_ref() {
                    m.inc_gc_sweep_batch_error("internal");
                }
                tracing::error!(
                    hash = %hash_hex,
                    member = %member,
                    action = %other,
                    "gc_decrement_and_collect returned an unknown action; leaving the member in \
                     inflight"
                );
                return Ok(GcMemberOutcome::Left {
                    deleted: false,
                    freed_bytes: 0,
                });
            }
        };
        if let Some(m) = self.metrics.as_ref() {
            m.inc_gc_decrement(if outcome == GcMemberOutcome::Replayed {
                GC_DECREMENT_OUTCOME_REPLAYED
            } else {
                GC_DECREMENT_OUTCOME_APPLIED
            });
        }

        // Site 2: complete — leave inflight, settle, take the bytes out.
        let complete_keys = vec![
            self.keys.ext_inflight(),
            self.keys.ext_pending(),
            self.keys.ext_pending_bytes(),
            self.keys.ext_pending_total(),
            self.keys.ext_settled(),
        ];
        let complete = self
            .gc_round_trip(3 * index + 2, || {
                GC_COMPLETE_SCRIPT.evalsha_with_reload::<i64, _, _>(
                    pool.next(),
                    complete_keys.clone(),
                    vec![member.to_owned()],
                )
            })
            .await;
        if let Err(e) = complete {
            let mapped = map_transient_fred_err(origin, "EVALSHA gc_complete", e);
            if is_transient_aborted(&mapped) {
                tracing::warn!(
                    hash = %hash_hex,
                    member = %member,
                    error = %mapped,
                    "transient gc_complete failure during a GC pass; leaving the member in \
                     inflight (its decrement is recorded, so its re-claim replays it)"
                );
                return Ok(match outcome {
                    GcMemberOutcome::Applied {
                        deleted: true,
                        freed_bytes,
                    } => GcMemberOutcome::Left {
                        deleted: true,
                        freed_bytes,
                    },
                    _ => GcMemberOutcome::Left {
                        deleted: false,
                        freed_bytes: 0,
                    },
                });
            }
            return Err(mapped);
        }
        Ok(outcome)
    }

    /// One page of the release-ledger audit while a cycle runs (H5):
    /// `gc_ledger_audit.lua` settles every ledger entry whose member left both
    /// sets unsettled. A cycle starts at the first pass of a fresh `state` and
    /// every [`GC_LEDGER_AUDIT_INTERVAL`] after, and runs one page per pass
    /// until its cursor wraps. Starting a cycle also starts the
    /// enqueue-attempt audit's walk ([`Self::gc_attempt_audit_page`]) unless
    /// the previous one is still running. WARNs when a page repairs anything — expected
    /// only around a rolling upgrade. Returns how many members the page
    /// settled; an error ends the pass and the cycle resumes from the same
    /// cursor.
    async fn gc_audit_page(
        &self,
        pool: &fred::clients::Pool,
        state: &mut GcPassState,
    ) -> Result<usize, ServiceError> {
        let now = tokio::time::Instant::now();
        if state.audit_cursor.is_none() && state.next_audit_cycle.is_none_or(|at| now >= at) {
            state.audit_cursor = Some("0".to_owned());
            state.next_audit_cycle = Some(now + GC_LEDGER_AUDIT_INTERVAL);
            // The enqueue-attempt audit walks once per cycle too, on its own
            // cursor; a walk still unfinished from the last cycle continues.
            if state.attempt_audit_cursor.is_none() {
                state.attempt_audit_cursor = Some("0".to_owned());
            }
        }
        let Some(cursor) = state.audit_cursor.clone() else {
            return Ok(0);
        };
        let keys = vec![
            self.keys.ext_pending(),
            self.keys.ext_inflight(),
            self.keys.ext_pending_age(),
            self.keys.ext_pending_bytes(),
            self.keys.ext_pending_total(),
            self.keys.ext_settled(),
            // Never written: makes the store refuse the script while the
            // `{gc}` slot migrates (see the script's "Migration sentinel").
            self.keys.migration_sentinel("gc"),
        ];
        let (next, repaired): (String, i64) = GC_LEDGER_AUDIT_SCRIPT
            .evalsha_with_reload(pool.next(), keys, vec![cursor])
            .await
            .map_err(|e| {
                map_transient_fred_err(
                    FredErrOrigin::client(&self.client, POOL_LABEL_MAINTENANCE),
                    "EVALSHA gc_ledger_audit",
                    e,
                )
            })?;
        state.audit_cursor = (next != "0").then_some(next);
        let repaired = repaired.max(0) as usize;
        if repaired > 0 {
            tracing::warn!(
                repaired,
                "release-ledger audit settled ledger entries whose members left the GC sets \
                 unsettled (expected only while an older build runs beside this one)"
            );
        }
        Ok(repaired)
    }

    /// One page of the enqueue-attempt audit while its walk runs (b.yuo Part
    /// R): `gc_enqueue_attempt_audit.lua` removes the `void` records older than
    /// [`ENQUEUE_ATTEMPT_VOID_RETENTION`] (V) on the page and returns its
    /// `landed` records older than the same V. For each of those, an
    /// `HEXISTS le:{oid}:obj:extret:attempts <token>` on this maintenance pool
    /// decides it: absent → the record is removed (`forget` mode, only while its
    /// value is unchanged); present → kept, because a carrier still names the
    /// token and a later resolve must find it `landed`. This is F2.
    ///
    /// Why V for both kinds, and not a shorter age for a landed record no
    /// carrier holds: a record guards SENDS under its token as well as
    /// resolves, and a token missing from its carrier is never resolved again
    /// but can still be sent under — the retirement's second attempt, or a
    /// copy fred writes late. No service send is decided more than Y after its
    /// token was recorded (F3), so past V only a command still in transit more
    /// than V − Y after it was decided can find the record gone (policy 14).
    ///
    /// The walk is an explicit cursor, complete only on a page that returns
    /// "0" (SDD §11.3.2); nothing is inferred from completeness. A `{gc}`-slot
    /// error ends the pass and the walk resumes from the same cursor; a failed
    /// `HEXISTS` (the object's own slot) keeps that record for the next cycle.
    /// Returns how many records the page removed.
    async fn gc_attempt_audit_page(
        &self,
        pool: &fred::clients::Pool,
        state: &mut GcPassState,
    ) -> Result<usize, ServiceError> {
        let Some(cursor) = state.attempt_audit_cursor.clone() else {
            return Ok(0);
        };
        let origin = FredErrOrigin::client(&self.client, POOL_LABEL_MAINTENANCE);
        let attempts_key = self.keys.ext_enqueue_attempts();
        let (next, void_removed, landed): (String, i64, Vec<String>) =
            GC_ENQUEUE_ATTEMPT_AUDIT_SCRIPT
                .evalsha_with_reload(
                    pool.next(),
                    vec![attempts_key.clone()],
                    vec![
                        "page".to_owned(),
                        cursor,
                        duration_ms(ENQUEUE_ATTEMPT_VOID_RETENTION).to_string(),
                        ENQUEUE_ATTEMPT_AUDIT_PAGE.to_string(),
                    ],
                )
                .await
                .map_err(|e| {
                    map_transient_fred_err(origin, "EVALSHA gc_enqueue_attempt_audit", e)
                })?;
        state.mark_progress();

        // `landed` is flat triples `token, oid, value`. Up to a page of
        // sequential `HEXISTS` follow; progress is marked after each, so a
        // long but healthy page keeps the lease refresh ticker renewing
        // (b.1jt, `GC_PROGRESS_STALL_AFTER`).
        let mut forget: Vec<String> = Vec::new();
        for triple in landed.chunks(3) {
            let [token, oid, value] = triple else {
                return Err(ServiceError::Internal(
                    "gc_enqueue_attempt_audit reply malformed: landed is not token, oid, value \
                     triples"
                        .to_owned(),
                ));
            };
            let held: Result<bool, fred::error::Error> = pool
                .next()
                .hexists(self.keys.obj_ext_retire_attempts(oid), token.as_str())
                .await;
            state.mark_progress();
            match held {
                Ok(false) => {
                    forget.push(token.clone());
                    forget.push(value.clone());
                }
                Ok(true) => {}
                Err(e) => {
                    origin.observe_error(&e);
                    tracing::warn!(
                        object_id = %oid,
                        token = %token,
                        error = %e,
                        "could not check a landed enqueue-attempt record against its carrier; \
                         keeping it for the next audit cycle"
                    );
                }
            }
        }
        let mut forgotten: i64 = 0;
        if !forget.is_empty() {
            let mut argv = Vec::with_capacity(forget.len() + 1);
            argv.push("forget".to_owned());
            argv.extend(forget);
            forgotten = GC_ENQUEUE_ATTEMPT_AUDIT_SCRIPT
                .evalsha_with_reload(pool.next(), vec![attempts_key], argv)
                .await
                .map_err(|e| {
                    map_transient_fred_err(origin, "EVALSHA gc_enqueue_attempt_audit (forget)", e)
                })?;
            state.mark_progress();
        }

        state.attempt_audit_cursor = (next != "0").then_some(next);
        let retired = usize::try_from(void_removed.max(0) + forgotten.max(0)).unwrap_or(0);
        if retired > 0 {
            tracing::debug!(
                void_removed,
                landed_removed = forgotten,
                "enqueue-attempt audit removed {{gc}} attempt records no carrier needs"
            );
        }
        Ok(retired)
    }
}

// ── GC sweep and background task ──────────────────────────────────────────

/// Statistics returned by [`process_pending_gc`].
#[derive(Debug, Clone, Default)]
pub struct GcStats {
    /// Members claimed across every pass.
    pub processed: usize,
    /// Number of external values deleted (refcount reached zero), summed over
    /// the passes' [`GcBatchResult::deleted`] — a lower bound for the same
    /// reasons: a delete whose decrement reply was lost is not counted, and a
    /// pass that ends `Err` contributes nothing.
    pub deleted: usize,
    /// Claimed members answered `replayed` (their decrement had already
    /// applied).
    pub replayed: usize,
    /// Claimed members refused `stale` by the decrement's claim-age fence.
    pub stale: usize,
    /// Settled members whose replay records were torn down.
    pub torn_down: usize,
}

/// Run GC passes under `policy` until the pending set is quiet: until a pass
/// claims nothing, tears down nothing and stamps nothing (b.yuo).
///
/// Calls [`ExternalValueStore::process_pending`] in a loop from a fresh
/// [`GcPassState`]. Under `GcPolicy::IMMEDIATE` that collects every pending
/// member and then tears down their replay records — the "drain now" helper
/// tests and the reconciler's test helpers use. The background loop
/// ([`spawn_gc_task`]) does not call this; it runs one pass per wake-up.
///
/// A pass error is logged AND counted via `gc_sweep_batch_errors_total{kind}`
/// (b.kf1 medium) and stops the run, so a persistent fault surfaces fast
/// rather than retrying the same broken Valkey call. The ledger gauges are
/// published from the last successful pass.
///
/// Returns aggregate [`GcStats`] for the whole run.
///
/// SDD §11.3.
pub async fn process_pending_gc(
    store: &dyn ExternalValueStore,
    metrics: Option<&SharedMetrics>,
    policy: GcPolicy,
) -> GcStats {
    let mut stats = GcStats::default();
    let mut state = GcPassState::new();

    loop {
        match store.process_pending(&policy, &mut state).await {
            Ok(pass) => {
                stats.processed += pass.processed;
                stats.deleted += pass.deleted;
                stats.replayed += pass.replayed;
                stats.stale += pass.stale;
                stats.torn_down += pass.torn_down;
                if let (Some(m), Some(ledger)) = (metrics, pass.ledger.as_ref()) {
                    publish_ext_release_gauges(m, Some(ledger));
                }
                // A member stamped this pass is a candidate from the next one
                // (at once under a zero age floor), so stamping keeps the run
                // going too.
                if pass.processed == 0 && pass.torn_down == 0 && pass.stamped == 0 {
                    break;
                }
            }
            Err(e) => {
                let kind = service_error_kind_label(&e);
                if let Some(m) = metrics {
                    m.inc_gc_sweep_batch_error(kind);
                }
                tracing::error!(error = %e, kind = %kind, "GC pass failed");
                break;
            }
        }
    }

    tracing::debug!(
        processed = stats.processed,
        deleted = stats.deleted,
        replayed = stats.replayed,
        stale = stats.stale,
        torn_down = stats.torn_down,
        "GC run complete"
    );

    stats
}

/// Publish the five release-ledger gauges from `ledger`, or zeros for `None`
/// — the non-leader publish, so a demoted leader does not fossilise a stale
/// level into the fleet-wide `max()`.
fn publish_ext_release_gauges(metrics: &SharedMetrics, ledger: Option<&GcLedgerSnapshot>) {
    let ledger = ledger.copied().unwrap_or_default();
    metrics.record_ext_release_gauges(
        ledger.pending_bytes,
        ledger.pending_count,
        ledger.oldest_age.map_or(0.0, |age| age.as_secs_f64()),
        ledger.settled_count,
        ledger.read_holds,
    );
}

/// Classify a `ServiceError` raised by `process_pending` into a short
/// `kind` slug suitable for the `gc_sweep_batch_errors_total{kind}`
/// metric (b.kf1 medium). The label cardinality stays bounded at four:
/// `transient_io` for an `Aborted` carrying [`map_snapshot_load_err`]'s
/// `"transient I/O:"` prefix (retriable fred failures, the store's
/// `-LOADING` / `-BUSY` included), `aborted` for any other `Aborted`,
/// `internal` for `Internal`, and `other` for every other variant — which
/// includes a store write refusal the mapper answers as a plain
/// `Unavailable` (`storage::client::store_write_refusal`, b.xd1). Operators
/// who need per-fred-`ErrorKind` resolution can cross-reference
/// `live_edit_valkey_errors_total{kind}` which already breaks down at the
/// fred layer.
fn service_error_kind_label(e: &ServiceError) -> &'static str {
    match e {
        ServiceError::Aborted(msg) if msg.starts_with("transient I/O") => "transient_io",
        ServiceError::Aborted(_) => "aborted",
        ServiceError::Internal(_) => "internal",
        _ => "other",
    }
}

/// Spawn the external-value GC loop: a background tokio task that runs one
/// GC pass ([`ExternalValueStore::process_pending`]) per wake-up under
/// `policy`, and exits cleanly when `cancel` is triggered (b.yuo).
///
/// **Leader election (b.kf1 H-stor-4; marked since b.1jt):** before every
/// pass, the task calls [`try_acquire_gc_leader`], which acquires, refreshes
/// or takes over `le:gc:leader` with `leader`'s protocol-marked value
/// ([`LeaderIdentity::gc_lock_value`], TTL [`LEADER_LOCK_TTL_MS`]). Only the
/// replica that holds the lock runs the pass; other replicas skip it.
///
/// **Refresh ticker (b.1jt).** Beside the loop, a second task calls
/// [`try_acquire_gc_leader`] once every [`GC_PASS_INTERVAL`] on every
/// replica, leader or not, independent of the passes, while the loop is
/// running and making progress, until `cancel` or until the loop ends. The
/// loop marks progress at every wake-up and after every step and member of a
/// pass, and the ticker renews only while the last mark is younger than
/// [`GC_PROGRESS_STALL_AFTER`] (20 s). So the leader refreshes the key
/// mid-pass, however long a progressing GC pass, orphan sweep, reconciler or
/// backup-repair pass runs (all four gate on the same key with the same
/// `leader`, and the GC loop keeps marking progress beside the other three),
/// and a non-leader
/// takes a same-version lock over once its TTL is at or below
/// [`GC_LEADER_TAKEOVER_BELOW_MS`] (a higher version only when it has no TTL):
/// failover takes about 20 s after the leader's last refresh. That holds for
/// a leader whose loop died while the process lives on (a panic): the ticker
/// stops with the loop, the end is logged at ERROR at once, and the orphan
/// sweep's, reconciler's and backup repair's own calls come minutes apart, too
/// rarely to keep
/// the key from a peer. It holds for a loop that is wedged without ending (a
/// pass that awaits forever) too: the ticker stops renewing
/// [`GC_PROGRESS_STALL_AFTER`] after the last progress (one WARN), so a peer
/// takes over about 40 s after it; the wedged replica's orphan sweep,
/// reconciler and backup-repair passes then lose the key mid-pass as they did before the
/// ticker existed. An older build's lock is taken over at once. The key
/// therefore never lapses while any process of this build with a running,
/// progressing GC loop can reach the store; the only lapse is every such
/// process losing the store for the TTL (or every one's loop dying or
/// stalling), and that window's safety rests on the store state, not the lock
/// (module header, "Older builds sharing the store"). The progress gate
/// writes nothing and leaves the key and value unchanged, so an older build
/// sees no new behaviour from it. The lock only sheds load among marked
/// builds: every step of a pass is safe under two sweepers, so a refresh that
/// finds a peer holding the key changes nothing about a pass already running.
///
/// `leader` must be the process-wide identity the orphan sweep, the
/// reconciler and the backup repair also hold: the four tasks share `le:gc:leader` and ride one
/// leadership through the script's refresh branch.
///
/// **Out-of-memory wake-up (b.pa1).** The loop also ends its sleep when
/// `client`'s store out-of-memory signal wakes it — at most once per
/// `store_oom::STORE_OOM_REQUEST_MIN_INTERVAL`, after the note's request is
/// sent (`storage::store_oom`) — so a leader takes a collection request soon
/// after a refused write rather than a sleep later; its out-of-memory
/// collections log `leader`'s identity.
///
/// See [`spawn_gc_task_with_leader`] for the loop's cadence, gauges and error
/// handling.
///
/// Returns a `JoinHandle` that can be awaited during shutdown: it awaits the
/// loop, stops the refresh ticker as soon as the loop ends, completes once
/// both have stopped, and re-raises a panic from either.
pub fn spawn_gc_task(
    store: std::sync::Arc<dyn ExternalValueStore>,
    client: ValkeyClient,
    keys: KeySchema,
    leader: LeaderIdentity,
    policy: GcPolicy,
    cancel: tokio_util::sync::CancellationToken,
    metrics: Option<SharedMetrics>,
) -> tokio::task::JoinHandle<()> {
    // Logged here, not in `spawn_gc_task_with_leader`: the start line carries
    // this process's identity, and only this wrapper holds it (the lower-level
    // spawn sees an opaque gate closure).
    tracing::info!(
        pass_interval_ms = duration_ms(GC_PASS_INTERVAL),
        min_age_ms = duration_ms(policy.min_age),
        pressure_min_age_ms = duration_ms(policy.pressure_min_age),
        replay_record_retention_ms = duration_ms(policy.replay_record_retention),
        reclaim_after_ms = duration_ms(policy.reclaim_after),
        replica_id = %leader.service_label,
        lock_value = %leader.gc_lock_value(),
        "GC background task started"
    );
    // The ticker runs under a child token so it stops when the loop ends for
    // any reason, not only on shutdown: a ticker outliving a dead loop would
    // keep this replica's lock alive with no GC pass behind it, and no peer
    // would ever take it over. And it renews only while the loop marks
    // progress on this heartbeat, so a loop that hangs without ending gives
    // the lock up too.
    let refresh_cancel = cancel.child_token();
    let shutdown = cancel.clone();
    let progress = GcProgress::new();
    let refresh = spawn_gc_leader_refresh(
        client.clone(),
        keys.clone(),
        leader.clone(),
        progress.clone(),
        GC_PROGRESS_STALL_AFTER,
        refresh_cancel.clone(),
        metrics.clone(),
    );
    let ended_replica_id = leader.service_label.clone();
    let ended_lock_value = leader.gc_lock_value();
    // b.pa1: the loop wakes on the client's store out-of-memory signal, and
    // its out-of-memory collections log this identity.
    let store_oom = client.store_oom_signal().clone();
    let log_leader = leader.clone();
    let leader_metrics = metrics.clone();
    let gate = move || {
        let client = client.clone();
        let keys = keys.clone();
        let leader = leader.clone();
        let metrics = leader_metrics.clone();
        async move {
            match try_acquire_gc_leader(&client, &keys, &leader, metrics.as_ref()).await {
                Ok(true) => true,
                Ok(false) => {
                    tracing::trace!(
                        replica_id = %leader.service_label,
                        lock_value = %leader.gc_lock_value(),
                        "GC pass skipped: another replica holds the leader lock"
                    );
                    false
                }
                Err(e) => {
                    if let Some(m) = metrics.as_ref() {
                        m.inc_gc_sweep_batch_error("leader_lock_error");
                    }
                    tracing::warn!(
                        replica_id = %leader.service_label,
                        lock_value = %leader.gc_lock_value(),
                        error = %e,
                        "GC leader lock acquisition failed"
                    );
                    false
                }
            }
        }
    };
    let gc_loop = spawn_gc_loop(
        store,
        policy,
        cancel,
        metrics,
        gate,
        GcLoopExtras {
            progress: Some(progress),
            store_oom: Some(store_oom),
            leader: Some(log_leader),
        },
    );
    // One handle for shutdown. The loop is awaited first and the ticker is
    // stopped as soon as it ends — on `cancel`, or early if it panicked — so
    // this replica stops refreshing the lock and a peer takes it over once its
    // TTL falls to the takeover threshold. A panic in either task is re-raised
    // here so the shutdown join still reports it.
    tokio::spawn(async move {
        let gc_loop = gc_loop.await;
        refresh_cancel.cancel();
        if !shutdown.is_cancelled() {
            // The loop only returns on `cancel`, so this is a panic (or an
            // abort). Logged now: the re-raised panic below reaches the
            // shutdown join only when the process shuts down.
            tracing::error!(
                replica_id = %ended_replica_id,
                lock_value = %ended_lock_value,
                panicked = gc_loop.as_ref().is_err_and(|e| e.is_panic()),
                "GC loop ended before shutdown; stopped refreshing the GC leader lock so a peer \
                 takes it over. This replica runs no GC pass until it restarts"
            );
        }
        let refresh = refresh.await;
        for joined in [gc_loop, refresh] {
            if let Err(e) = joined {
                if e.is_panic() {
                    std::panic::resume_unwind(e.into_panic());
                }
            }
        }
    })
}

/// The GC leader-lock refresh ticker [`spawn_gc_task`] runs beside its loop
/// (b.1jt): one [`try_acquire_gc_leader`] call every [`GC_PASS_INTERVAL`],
/// independent of any pass, while the loop is making progress, until
/// `cancel`. [`spawn_gc_task`] passes a child of its shutdown token and
/// cancels it as soon as the loop ends, so the ticker never outlives the loop
/// it keeps the lock for.
///
/// It exists because the loop's own call runs only between passes: without
/// it, a leader whose pass outlived [`LEADER_LOCK_TTL_MS`] would let
/// `le:gc:leader` lapse, and an older build, which acquires only an absent
/// key, could then lead. The verdict is not used here — the loop, the orphan
/// sweep, the reconciler and the backup repair each still gate their next pass
/// on their own call
/// — and a takeover is logged and counted inside [`try_acquire_gc_leader`]
/// like any other.
///
/// **Progress-gated.** Each tick first reads `progress`: when the loop last
/// marked it `stall_after` ago or more ([`GC_PROGRESS_STALL_AFTER`] in
/// production), the tick skips the call, so a loop that is wedged — hung, not
/// ended — stops renewing the key and a peer takes it over once its TTL falls
/// to [`GC_LEADER_TAKEOVER_BELOW_MS`]. A store slow enough that a few round
/// trips between marks reach the bound pauses renewal the same way. Renewing
/// resumes on the first tick after the loop marks progress again, so the key
/// is kept if that happens before a peer takes it over. The key and value written are unchanged, so
/// the gate changes only when this replica renews, never what an older build
/// sharing the store sees.
///
/// **Logging on a change of state only.** A failed call is counted under
/// `gc_sweep_batch_errors_total{kind="leader_lock_error"}` every time, as the
/// gated tasks do, but logged only on the edges: WARN when calls start
/// failing, INFO when one succeeds again, WARN once when the ticker stops
/// renewing for want of progress, and INFO when it resumes. The next tick
/// retries either way.
fn spawn_gc_leader_refresh(
    client: ValkeyClient,
    keys: KeySchema,
    leader: LeaderIdentity,
    progress: GcProgress,
    stall_after: std::time::Duration,
    cancel: tokio_util::sync::CancellationToken,
    metrics: Option<SharedMetrics>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut failing = false;
        let mut stalled = false;
        loop {
            tokio::select! {
                () = tokio::time::sleep(GC_PASS_INTERVAL) => {}
                () = cancel.cancelled() => return,
            }
            let since_progress = progress.since_last();
            if since_progress >= stall_after {
                if !stalled {
                    stalled = true;
                    tracing::warn!(
                        replica_id = %leader.service_label,
                        lock_value = %leader.gc_lock_value(),
                        since_progress_ms = duration_ms(since_progress),
                        stall_after_ms = duration_ms(stall_after),
                        "the GC loop has made no progress for too long (a pass that is hung, or \
                         store round trips that are very slow); stopped refreshing the GC leader \
                         lock so a peer takes it over once its TTL runs low. Refreshing resumes \
                         when the loop makes progress again"
                    );
                }
                continue;
            }
            if stalled {
                stalled = false;
                tracing::info!(
                    replica_id = %leader.service_label,
                    lock_value = %leader.gc_lock_value(),
                    "the GC loop is making progress again; resumed refreshing the GC leader lock"
                );
            }
            let result = tokio::select! {
                result = try_acquire_gc_leader(&client, &keys, &leader, metrics.as_ref()) => result,
                () = cancel.cancelled() => return,
            };
            match result {
                Ok(_) => {
                    if failing {
                        failing = false;
                        tracing::info!(
                            replica_id = %leader.service_label,
                            lock_value = %leader.gc_lock_value(),
                            "GC leader lock refresh succeeded again"
                        );
                    }
                }
                Err(e) => {
                    if let Some(m) = metrics.as_ref() {
                        m.inc_gc_sweep_batch_error("leader_lock_error");
                    }
                    if !failing {
                        failing = true;
                        tracing::warn!(
                            replica_id = %leader.service_label,
                            lock_value = %leader.gc_lock_value(),
                            error = %e,
                            "GC leader lock refresh failed; logged again only when a refresh \
                             succeeds. The key lapses only if every process of this build fails \
                             to refresh it for the lock TTL"
                        );
                    }
                }
            }
        }
    })
}

/// Lower-level spawn that decouples the leader-gate from the GC loop
/// (b.kf1 H-stor-4). Production callers go through [`spawn_gc_task`] which
/// supplies a Valkey-backed gate; unit tests pin the lifecycle behavior
/// without a real ValkeyClient by passing a closure that always returns
/// `true`. The closure is `FnMut` so callers that need state (counters,
/// alternating leader assignment in cluster tests) can keep it.
///
/// **The loop (b.yuo).** Every wake-up asks the gate. A leader runs one pass
/// and publishes the release-ledger gauges from it; a non-leader publishes
/// zeros (an OTel gauge never decays, so a demoted leader must not leave its
/// last level standing). A pass that errored publishes nothing — unknown is
/// not zero — and is logged at ERROR and counted via
/// `gc_sweep_batch_errors_total{kind}`; the loop always continues. After a
/// pass that made progress on a full batch ([`GcBatchResult::more`]) the next
/// pass runs at once;
/// otherwise the loop sleeps [`GC_PASS_INTERVAL`], or less when the oldest
/// pending member falls due sooner, or until the client's store out-of-memory
/// signal wakes it, at most once a second (b.pa1; [`spawn_gc_task`] only). The
/// pass state (the audit
/// cursor, and the out-of-memory collection's rate limit) is
/// reset at the start of every new leadership. The gate is asked only between
/// passes; the mid-pass lease refresh is [`spawn_gc_task`]'s ticker, not part
/// of this loop, and so is the progress heartbeat the ticker is gated on:
/// this entry point runs the loop with none, and with no out-of-memory
/// wake-up.
///
/// Emits no start line: it holds no [`LeaderIdentity`], and a leader-lock
/// log line carries one. [`spawn_gc_task`] logs it.
pub fn spawn_gc_task_with_leader<F, Fut>(
    store: std::sync::Arc<dyn ExternalValueStore>,
    policy: GcPolicy,
    cancel: tokio_util::sync::CancellationToken,
    metrics: Option<SharedMetrics>,
    leader_gate: F,
) -> tokio::task::JoinHandle<()>
where
    F: FnMut() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = bool> + Send + 'static,
{
    spawn_gc_loop(
        store,
        policy,
        cancel,
        metrics,
        leader_gate,
        GcLoopExtras::default(),
    )
}

/// What [`spawn_gc_task`] gives its loop beyond [`spawn_gc_task_with_leader`]'s
/// arguments; all `None` there.
#[derive(Default)]
struct GcLoopExtras {
    /// The loop's progress heartbeat (b.1jt): the loop marks it at every
    /// wake-up and hands it to every pass ([`GcPassState`]), which marks it
    /// after every step and member; the lease refresh ticker reads it.
    progress: Option<GcProgress>,
    /// The client's store out-of-memory signal (b.pa1): the note that takes
    /// its turn — at most one per `store_oom::STORE_OOM_REQUEST_MIN_INTERVAL`,
    /// after its request is sent — ends the loop's sleep, so a leader takes a
    /// collection request without waiting out [`GC_PASS_INTERVAL`]. Every
    /// other note leaves the sleep alone, so the GC's own refused steps, which
    /// note too, never keep the loop from sleeping.
    store_oom: Option<crate::storage::store_oom::StoreOomSignal>,
    /// The process's leader identity, which the out-of-memory collection's
    /// log line names (b.pa1).
    leader: Option<LeaderIdentity>,
}

/// [`spawn_gc_task_with_leader`] with [`GcLoopExtras`]: the progress
/// heartbeat, the out-of-memory wake-up and the identity [`spawn_gc_task`]
/// passes.
fn spawn_gc_loop<F, Fut>(
    store: std::sync::Arc<dyn ExternalValueStore>,
    policy: GcPolicy,
    cancel: tokio_util::sync::CancellationToken,
    metrics: Option<SharedMetrics>,
    mut leader_gate: F,
    extras: GcLoopExtras,
) -> tokio::task::JoinHandle<()>
where
    F: FnMut() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = bool> + Send + 'static,
{
    let GcLoopExtras {
        progress,
        store_oom,
        leader,
    } = extras;
    tokio::spawn(async move {
        let fresh_state = || GcPassState {
            progress: progress.clone(),
            leader: leader.clone(),
            ..GcPassState::new()
        };
        // The store out-of-memory signal's wake-up — at most one a second,
        // after a note's request is sent — ends the sleep below; without a
        // signal the branch never fires.
        let woken = || {
            let store_oom = store_oom.clone();
            async move {
                match store_oom {
                    Some(signal) => signal.woken().await,
                    None => std::future::pending().await,
                }
            }
        };
        let mut state = fresh_state();
        let mut was_leader = false;
        loop {
            state.mark_progress();
            let mut sleep = GC_PASS_INTERVAL;
            let is_leader = leader_gate().await;
            if is_leader {
                if !was_leader {
                    state = fresh_state();
                }
                match store.process_pending(&policy, &mut state).await {
                    Ok(pass) => {
                        if let Some(m) = metrics.as_ref() {
                            if let Some(ledger) = pass.ledger.as_ref() {
                                publish_ext_release_gauges(m, Some(ledger));
                            }
                        }
                        tracing::debug!(
                            processed = pass.processed,
                            deleted = pass.deleted,
                            replayed = pass.replayed,
                            stale = pass.stale,
                            torn_down = pass.torn_down,
                            reclaimed = pass.reclaimed,
                            foreign_dropped = pass.foreign_dropped,
                            stamped = pass.stamped,
                            "GC pass complete"
                        );
                        if pass.more {
                            sleep = std::time::Duration::ZERO;
                        } else if let Some(due) = pass.ledger.and_then(|l| l.next_due_in) {
                            sleep = sleep.min(due);
                        }
                    }
                    Err(e) => {
                        let kind = service_error_kind_label(&e);
                        if let Some(m) = metrics.as_ref() {
                            m.inc_gc_sweep_batch_error(kind);
                        }
                        tracing::error!(error = %e, kind = %kind, "GC pass failed");
                    }
                }
            } else if let Some(m) = metrics.as_ref() {
                publish_ext_release_gauges(m, None);
            }
            was_leader = is_leader;

            if sleep.is_zero() {
                if cancel.is_cancelled() {
                    tracing::info!("GC task shutting down");
                    return;
                }
                tokio::task::yield_now().await;
                continue;
            }
            tokio::select! {
                () = tokio::time::sleep(sleep) => {}
                () = woken() => {}
                () = cancel.cancelled() => {
                    tracing::info!("GC task shutting down");
                    return;
                }
            }
        }
    })
}

/// What [`acquire_gc_leader_lock`] found and did on `le:gc:leader` — the
/// verdicts of `scripts/gc_leader_acquire.lua`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GcLeaderAcquire {
    /// The key was absent; this process now holds it.
    Acquired,
    /// This process already held it; its TTL was extended.
    Refreshed,
    /// The key held an older build's value — no protocol marker, or a lower
    /// version — and this process took it over. `displaced` is that value.
    TookOverOlder { displaced: String },
    /// The key held a marked value with no TTL (any version), or a
    /// same-version peer's value whose TTL was at or below
    /// [`GC_LEADER_TAKEOVER_BELOW_MS`], and this process took it over.
    /// `displaced` is that value.
    TookOverExpiring { displaced: String },
    /// Another marked process holds it — a higher version with a TTL, at any
    /// TTL, or a same-version peer above the threshold; nothing was written.
    /// `holder` is its value.
    Held { holder: String },
}

impl GcLeaderAcquire {
    /// Whether this process holds the lock after the call.
    pub fn is_leader(&self) -> bool {
        !matches!(self, GcLeaderAcquire::Held { .. })
    }
}

/// One round trip of `scripts/gc_leader_acquire.lua` against `lock_key` with
/// `leader`'s marked value ([`LeaderIdentity::gc_lock_value`]), TTL `ttl_ms`
/// and takeover threshold `takeover_below_ms`, on the maintenance pool.
/// Production goes through [`try_acquire_gc_leader`] (30 s TTL, 10 s
/// threshold), which also logs and counts a takeover; this lower level logs
/// nothing and exists so tests can drive the script with shorter times.
///
/// A command timeout is counted here; callers only log the returned error. A
/// verdict this build does not know is a `Parse` error: the caller then does
/// not lead, the safe reading.
pub async fn acquire_gc_leader_lock(
    client: &ValkeyClient,
    lock_key: &str,
    leader: &LeaderIdentity,
    ttl_ms: u64,
    takeover_below_ms: u64,
) -> Result<GcLeaderAcquire, fred::error::Error> {
    use crate::storage::leader_lease::{GC_LEADER_MARKER_PREFIX, GC_LEADER_PROTOCOL_VERSION};
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    // b.zt3 re-send audit: fred keeps its default write attempts here, so a lost
    // connection can run this twice: the second run finds this process's own
    // value and answers `refreshed`, harmless (script header).
    let (verdict, found): (String, String) = GC_LEADER_ACQUIRE_SCRIPT
        .evalsha_with_reload(
            maint_pool.next(),
            vec![lock_key.to_owned()],
            vec![
                leader.gc_lock_value(),
                ttl_ms.to_string(),
                GC_LEADER_MARKER_PREFIX.to_owned(),
                GC_LEADER_PROTOCOL_VERSION.to_string(),
                takeover_below_ms.to_string(),
            ],
        )
        .await
        .inspect_err(|e| FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE).observe_error(e))?;
    match verdict.as_str() {
        "acquired" => Ok(GcLeaderAcquire::Acquired),
        "refreshed" => Ok(GcLeaderAcquire::Refreshed),
        "took_over_older" => Ok(GcLeaderAcquire::TookOverOlder { displaced: found }),
        "took_over_expiring" => Ok(GcLeaderAcquire::TookOverExpiring { displaced: found }),
        "held" => Ok(GcLeaderAcquire::Held { holder: found }),
        other => Err(unknown_verdict("gc_leader_acquire", other)),
    }
}

/// Try to acquire, refresh or take over the GC leader lock `le:gc:leader`
/// (SDD §11.3) with the [`LEADER_LOCK_TTL_MS`] (30 s) TTL. Returns
/// `Ok(true)` if `leader` now holds the lock, `Ok(false)` if another marked
/// process does. The GC loop, the orphan sweep, the reconciler and the
/// external-value backup repair (`checkpoint::ext_backup_repair`) all gate on
/// it, with one process-wide `leader`.
///
/// The value is protocol-marked ([`LeaderIdentity::gc_lock_value`]), and the
/// script ([`acquire_gc_leader_lock`], `scripts/gc_leader_acquire.lua`) takes
/// the lock over from an older build (no marker or a lower version), from a
/// same-version peer whose TTL is at or below [`GC_LEADER_TAKEOVER_BELOW_MS`],
/// and from a marked value of any version with no TTL; a higher version with a
/// TTL is left to lapse. Every process calls this once every
/// [`GC_PASS_INTERVAL`] from the refresh ticker [`spawn_gc_task`] runs beside
/// its GC loop, independent of any pass and only while that loop runs and
/// makes progress ([`GC_PROGRESS_STALL_AFTER`]), and again before each GC,
/// orphan-sweep, reconciler and backup-repair pass. So the leader refreshes the key
/// mid-pass, and the key never lapses while any process of this build with a
/// running, progressing GC loop can reach the store: while it stays marked an
/// older build, which only acquires an absent key, starts no new GC,
/// orphan-sweep or reconciler pass (one already running when the lock was
/// taken over runs on until it ends on its own). The only lapse is every such
/// process losing the store for the TTL (or every one's loop dying or
/// stalling); an older
/// build — unmarked, or a lower marked version, which answers `held` to a
/// higher version at any TTL and so never keeps the key alive for it — can then
/// acquire the key until a process of this build reaches the store and takes it
/// back; safety in that window rests on the store state, not the lock (module
/// header, "Older builds sharing the store").
///
/// A takeover from an older build logs one WARN naming the displaced value
/// and ticks `live_edit_gc_older_build_total{event="lock_taken_over"}`; a
/// takeover from an expiring marked peer logs INFO. Both lines carry
/// `lock_value`, the marked value written.
pub(crate) async fn try_acquire_gc_leader(
    client: &ValkeyClient,
    keys: &KeySchema,
    leader: &LeaderIdentity,
    metrics: Option<&SharedMetrics>,
) -> Result<bool, fred::error::Error> {
    let verdict = acquire_gc_leader_lock(
        client,
        &keys.gc_leader_lock(),
        leader,
        LEADER_LOCK_TTL_MS,
        GC_LEADER_TAKEOVER_BELOW_MS,
    )
    .await?;
    match &verdict {
        GcLeaderAcquire::TookOverOlder { displaced } => {
            if let Some(m) = metrics {
                m.inc_gc_older_build(GC_OLDER_BUILD_EVENT_LOCK_TAKEN_OVER);
            }
            tracing::warn!(
                replica_id = %leader.service_label,
                lock_value = %leader.gc_lock_value(),
                displaced = %displaced,
                "took over the GC leader lock from an older build (its value carries no \
                 protocol marker, or a lower version); that build's GC, orphan sweep and \
                 reconciler start no new pass while the key stays marked. Expected once per \
                 rollout or rollback across the marker"
            );
        }
        GcLeaderAcquire::TookOverExpiring { displaced } => {
            tracing::info!(
                replica_id = %leader.service_label,
                lock_value = %leader.gc_lock_value(),
                displaced = %displaced,
                "took over the GC leader lock from a peer that stopped refreshing it (stopped, \
                 restarting, or stalled)"
            );
        }
        GcLeaderAcquire::Acquired | GcLeaderAcquire::Refreshed | GcLeaderAcquire::Held { .. } => {}
    }
    Ok(verdict.is_leader())
}

/// Generic leader-lock acquire/refresh against an arbitrary lock key
/// (b.w29). Wraps [`LEADER_ACQUIRE_OR_REFRESH_SCRIPT`], the plain CAS of
/// `le:checkpoint:leader` and `le:presence:sweeper:leader`. `le:gc:leader`
/// does not use it: it carries a protocol-marked value and its own takeover
/// script ([`try_acquire_gc_leader`]). The locks are
/// deliberately separate keys so the maintenance loops can land on different
/// replicas in steady state. See [`KeySchema::presence_sweeper_leader_lock`]
/// and [`KeySchema::checkpoint_leader_lock`].
///
/// Writes, and recognises as "already ours", `leader`'s
/// [`LeaderIdentity::lock_value`] — never the bare service label, so two
/// processes sharing a label never both lead. On a claim or a refresh the
/// key's TTL becomes `ttl_ms`; every production caller passes
/// [`LEADER_LOCK_TTL_MS`], tests a shorter one.
///
/// `lock_key` must be a fully-qualified Valkey key (i.e. the
/// `KeySchema::*_leader_lock` accessor output, NOT a bare suffix).
pub async fn try_acquire_leader_lock(
    client: &ValkeyClient,
    lock_key: &str,
    leader: &LeaderIdentity,
    ttl_ms: u64,
) -> Result<bool, fred::error::Error> {
    // `acquire_maintenance` bumps `valkey_pool_in_use{pool="maintenance"}`
    // for the duration of this leader probe (b.mfn).
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let lua_client = maint_pool.next();
    let lua_keys = vec![lock_key.to_owned()];
    let lua_argv = vec![leader.lock_value(), ttl_ms.to_string()];

    // A command timeout is counted here; callers only log the returned error.
    // b.zt3 re-send audit: fred keeps its default write attempts here, so a lost
    // connection can run this twice: it re-asserts the same holder's lease,
    // harmless (script header).
    let result: i64 = LEADER_ACQUIRE_OR_REFRESH_SCRIPT
        .evalsha_with_reload::<i64, _, _>(lua_client, lua_keys, lua_argv)
        .await
        .inspect_err(|e| FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE).observe_error(e))?;
    Ok(result == 1)
}

/// Refresh-only extension of a leader lock this process already holds
/// (b.s5y). Returns `true` iff the lock's value is still `leader`'s
/// [`LeaderIdentity::lock_value`] and its TTL was extended to `ttl_ms`;
/// `false` if the key has lapsed or is owned by another process — in which
/// case the key is NOT touched. Every production caller passes
/// [`LEADER_LOCK_TTL_MS`], tests a shorter one.
///
/// Deliberately distinct from [`try_acquire_leader_lock`]: this one never
/// claims an unset key. The caller is a tick already mutating shared durable
/// state under the lock, and re-acquiring a lapsed lock there would report
/// "still held" across a gap in which a peer replica may have acquired the
/// lock and begun the same work. See `scripts/leader_refresh_if_held.lua`.
///
/// `lock_key` must be a fully-qualified Valkey key (i.e. the
/// `KeySchema::*_leader_lock` accessor output, NOT a bare suffix).
pub async fn refresh_leader_lock_if_held(
    client: &ValkeyClient,
    lock_key: &str,
    leader: &LeaderIdentity,
    ttl_ms: u64,
) -> Result<bool, fred::error::Error> {
    // Same maintenance-pool routing as the acquire probe: the refresh is
    // maintenance traffic and must not queue behind data-pool writers.
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let lua_client = maint_pool.next();
    let lua_keys = vec![lock_key.to_owned()];
    let lua_argv = vec![leader.lock_value(), ttl_ms.to_string()];

    // A command timeout is counted here; callers only log the returned error.
    // b.zt3 re-send audit: fred keeps its default write attempts here, so a lost
    // connection can run this twice: it re-asserts the same holder's lease,
    // harmless (script header).
    let result: i64 = LEADER_REFRESH_IF_HELD_SCRIPT
        .evalsha_with_reload::<i64, _, _>(lua_client, lua_keys, lua_argv)
        .await
        .inspect_err(|e| FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE).observe_error(e))?;
    Ok(result == 1)
}

// ── Checkpoint-store reconciliation ZSETs (b.9qd) ──────────────────────────

/// Register `candidates` on `zset_key` and return the members whose FIRST
/// observation has aged past `grace`.
///
/// Two phases, deliberately split across round-trips rather than crammed into
/// one: the registrations go out in [`RECON_REGISTER_CHUNK`]-sized batches with
/// `max_return = 0` (register-only), then one final call with no candidates
/// fetches the claim. Batching the ARGV vector keeps a fleet-sized candidate
/// list off a single command, and doing the fetch last means the returned set
/// reflects every registration this pass made.
///
/// `max_return` bounds one pass's destructive batch. Anything above it simply
/// waits for the next pass — the ZSET is durable, so nothing is lost by
/// deferring.
///
/// Returns members oldest-first. The script does not remove them; the caller
/// calls [`recon_retire`] once it has finished acting on a member, so an action
/// that failed leaves the member claimable next pass. See
/// `scripts/recon_claim_expired.lua`.
pub(crate) async fn recon_claim_expired(
    client: &ValkeyClient,
    zset_key: &str,
    grace: std::time::Duration,
    max_return: usize,
    candidates: &[String],
) -> Result<Vec<String>, fred::error::Error> {
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let grace_ms = u64::try_from(grace.as_millis()).unwrap_or(u64::MAX);
    // A command timeout is counted here; callers only log the returned error.
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);

    for chunk in candidates.chunks(RECON_REGISTER_CHUNK) {
        let mut argv: Vec<String> = Vec::with_capacity(chunk.len() + 2);
        argv.push(grace_ms.to_string());
        // `0` = register-only; the script returns nothing for these calls.
        argv.push("0".to_owned());
        argv.extend(chunk.iter().cloned());
        let _: Vec<String> = RECON_CLAIM_EXPIRED_SCRIPT
            .evalsha_with_reload::<Vec<String>, _, _>(
                maint_pool.next(),
                vec![zset_key.to_owned()],
                argv,
            )
            .await
            .inspect_err(|e| origin.observe_error(e))?;
    }

    RECON_CLAIM_EXPIRED_SCRIPT
        .evalsha_with_reload::<Vec<String>, _, _>(
            maint_pool.next(),
            vec![zset_key.to_owned()],
            vec![grace_ms.to_string(), max_return.to_string()],
        )
        .await
        .inspect_err(|e| origin.observe_error(e))
}

/// Drop members from a `{recon}` ZSET — the caller's acknowledgement that it
/// finished acting on them (deleted the orphan, collected or spared the blob),
/// or that they healed and are no longer candidates.
///
/// A no-op on an empty slice so callers need no guard. `ZREM` is
/// idempotent-on-missing, so a duplicate retire from two replicas costs
/// nothing.
pub(crate) async fn recon_retire(
    client: &ValkeyClient,
    zset_key: &str,
    members: &[String],
) -> Result<(), fred::error::Error> {
    if members.is_empty() {
        return Ok(());
    }
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    // A command timeout is counted here; callers only log the returned error.
    let origin = FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE);
    for chunk in members.chunks(RECON_REGISTER_CHUNK) {
        let _: i64 = maint_pool
            .next()
            .zrem(zset_key, chunk.to_vec())
            .await
            .inspect_err(|e| origin.observe_error(e))?;
    }
    Ok(())
}

/// Register `members` on a `{recon}` ZSET without claiming anything (b.vn6) —
/// the register-only entry point every dead-object producer outside the
/// reconciliation pass itself goes through.
///
/// Same `ZADD NX` first-observation semantics as [`recon_claim_expired`]'s
/// register phase — it is literally the same script, driven register-only —
/// and it exists as a separate entry point for two reasons, the **pool** and
/// keeping those semantics spelled out in one place.
///
/// # Why the data pool
///
/// `recon_claim_expired` routes through `acquire_maintenance()`, which is
/// correct for the hourly reconciliation pass and wrong for every caller here.
/// `LIVE_EDIT_MAINTENANCE_POOL_SIZE` defaults to **1**, so anything sharing it
/// queues behind a sweep's enumeration traffic. The callers are:
///
/// - **The delete paths** — `LayerStore::delete_layer_inner` (the DeleteLayer
///   RPC) and the CreateLayer-overwrite cleanup. Squarely request-path: a
///   user-facing delete stalled head-of-line behind a sweep would be
///   self-inflicted latency, and every other command in that cascade already
///   takes the data pool.
/// - **The four refused-stamp arms** in the checkpoint machinery — the loop's
///   periodic snapshot, tail capture and chain-collapse re-base, plus the
///   creation-time base bootstrap (b.vn6 review round 2). These are background
///   callers, so the head-of-line argument above does not apply to them; the
///   data pool is still right, for the adjacent reason. Each is a single `ZADD
///   NX` on an arm that is *abandoning* a layer's tick after discovering the
///   layer is gone — the cheapest possible write, on a path with nothing left
///   to do — and putting it on the size-1 maintenance pool would make an
///   abandoning arm contend with the very reconciliation pass that consumes
///   what it is registering. The bootstrap arm is also, in the worst case, on a
///   CreateLayer's heels.
///
/// The common shape: a caller that registers and never claims has no business
/// on the maintenance pool, whichever side of the request boundary it sits on.
///
/// A no-op on an empty slice, so callers need no guard.
pub(crate) async fn recon_register(
    client: &ValkeyClient,
    zset_key: &str,
    members: &[String],
) -> Result<(), fred::error::Error> {
    if members.is_empty() {
        return Ok(());
    }
    for chunk in members.chunks(RECON_REGISTER_CHUNK) {
        let mut argv: Vec<String> = Vec::with_capacity(chunk.len() + 2);
        // The grace is inert on a register-only call, and `0` = register-only:
        // this caller must never claim, because claiming implies it is about to
        // act on what it claims and it is not.
        argv.push("0".to_owned());
        argv.push("0".to_owned());
        argv.extend(chunk.iter().cloned());
        // A command timeout is counted here; callers only log the returned
        // error.
        let _: Vec<String> = RECON_CLAIM_EXPIRED_SCRIPT
            .evalsha_with_reload::<Vec<String>, _, _>(
                client.pool.next(),
                vec![zset_key.to_owned()],
                argv,
            )
            .await
            .inspect_err(|e| FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(e))?;
    }
    Ok(())
}

/// `ZCARD` of a `{recon}` ZSET — the backlog gauge's input.
pub(crate) async fn recon_backlog(
    client: &ValkeyClient,
    zset_key: &str,
) -> Result<u64, fred::error::Error> {
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    // A command timeout is counted here; callers only log the returned error.
    let n: i64 =
        maint_pool.next().zcard(zset_key).await.inspect_err(|e| {
            FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE).observe_error(e)
        })?;
    Ok(u64::try_from(n).unwrap_or(0))
}

// ── The `{gc}` read hold (b.yuo) ─────────────────────────────────────────────

// ── Read hold take seam (test-only) ──────────────────────────────────────────
//
// Compiled only under `#[cfg(any(test, feature = "test-fault-injection"))]`,
// in the style of `storage::diff_read::SnapshotReadHold`. The release library
// has neither the hold nor the branch that consults it: `ReadHoldTakeHolds`
// degenerates to a zero-sized unit struct, and `ReadHold::take` sends its
// script and nothing else.
//
// **Why a seam is needed.** The catch-up miss protocol's soundness rests on
// its hold being acknowledged before it attaches to a fenced-read flight. A
// test proves that order fails-on-old only if it can hold the take itself
// back — with no hold field on `{gc}` yet — while a flight fixes its cut,
// releases its fence, a commit releases a blob the cut references and the GC
// collects it. A park placed after the store answered would not do: the field
// would already be live, and the GC's hold clip would protect the blob.
//
// Per-instance (on a `ValkeyClient`, shared behind an `Arc`) and keyed by
// object id, so concurrently-running tests never share an armed hold.

/// One armed hold of [`ValkeyClient::arm_read_hold_take_hold_for_test`]: the
/// handle a test waits on and releases.
///
/// The take for the armed object parks at the start of [`ReadHold::take`],
/// **before** its `EVALSHA ext_read_hold` is sent and before its guard is
/// built: no field is written on `{gc}`, the store has answered nothing, and
/// the caller has no hold. On [`Self::release`] the take goes on exactly as
/// an unheld one. A take dropped while parked (its catch-up cancelled) sends
/// nothing and leaves nothing to release.
///
/// Two level latches, each set once and never cleared: **parked**, set when
/// the take reaches the park, and **released**, set by the test. Both are
/// levels, so the order of the two sides does not matter: [`Self::parked`]
/// awaited after the take parked returns at once, and a [`Self::release`]
/// made before the take arrives lets it pass without parking (it still sets
/// parked as it passes). The park is an async wait: it blocks no OS thread.
/// One hold parks one take. A take the test never releases stays parked until
/// its caller is dropped, so a test that arms one must release it on every
/// path, and must bound a [`Self::parked`] wait (for example with
/// `tokio::time::timeout`), since it never returns if no take for the object
/// comes.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Default)]
pub struct ReadHoldTakeHold {
    parked: std::sync::atomic::AtomicBool,
    parked_notify: tokio::sync::Notify,
    released: std::sync::atomic::AtomicBool,
    released_notify: tokio::sync::Notify,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl ReadHoldTakeHold {
    /// Wait until the armed object's take has parked, nothing sent. Returns
    /// at once if it already has; never returns if no take for the object
    /// comes, so bound the wait.
    pub async fn parked(&self) {
        crate::storage::diff_read::wait_for_latch(&self.parked, &self.parked_notify).await;
    }

    /// Whether the armed object's take has reached the park.
    pub fn is_parked(&self) -> bool {
        self.parked.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Let the parked take send its `EVALSHA` and go on. Idempotent; made
    /// before the take arrives, the take does not park at all.
    pub fn release(&self) {
        crate::storage::diff_read::set_latch(&self.released, &self.released_notify);
    }

    /// The take's side: mark parked, then wait for the release.
    async fn park(&self) {
        crate::storage::diff_read::set_latch(&self.parked, &self.parked_notify);
        crate::storage::diff_read::wait_for_latch(&self.released, &self.released_notify).await;
    }
}

/// The armed [`ReadHoldTakeHold`]s a [`ValkeyClient`] carries, one per
/// object id at most.
///
/// Shared behind an `Arc` so a hold armed on one clone is seen by every
/// other clone: the catch-up takes its hold through the service's client,
/// which the test arms after building the service.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Clone, Debug, Default)]
pub struct ReadHoldTakeHolds {
    armed: Arc<std::sync::Mutex<std::collections::HashMap<u64, Arc<ReadHoldTakeHold>>>>,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl ReadHoldTakeHolds {
    /// No hold armed. Named rather than `Default::default()` because the
    /// construction site in `ValkeyClient` is shared by both cfg shapes, and
    /// the release shape's unit struct makes `::default()` a clippy error
    /// (`default_constructed_unit_structs`).
    pub fn none() -> Self {
        Self::default()
    }

    /// Arm (or replace) the hold for `object_id`'s next take.
    pub(crate) fn arm(&self, object_id: u64) -> Arc<ReadHoldTakeHold> {
        let hold = Arc::new(ReadHoldTakeHold::default());
        if let Ok(mut armed) = self.armed.lock() {
            armed.insert(object_id, Arc::clone(&hold));
        }
        hold
    }

    /// Disarm and return the hold armed for `object_id`, if any: one take
    /// consumes it.
    fn take(&self, object_id: u64) -> Option<Arc<ReadHoldTakeHold>> {
        self.armed.lock().ok()?.remove(&object_id)
    }
}

/// Production stand-in for the take seam: a zero-sized unit struct, so the
/// field on `ValkeyClient` costs nothing and needs no `#[cfg]` at its
/// construction site.
#[cfg(not(any(test, feature = "test-fault-injection")))]
#[derive(Clone, Debug, Default)]
pub struct ReadHoldTakeHolds;

#[cfg(not(any(test, feature = "test-fault-injection")))]
impl ReadHoldTakeHolds {
    /// Shape-independent counterpart of the test-shape constructor, so
    /// `ValkeyClient`'s construction site needs no `#[cfg]`.
    pub fn none() -> Self {
        Self
    }
}

/// The `{gc}` read hold's timing (b.yuo): how long one write keeps a holder's
/// field live, and how often a holder renews it while it still needs it.
/// Production runs [`Self::PRODUCTION`]; a test can override it per client
/// (`ValkeyClient::set_read_hold_timing_for_test`, `test-support` only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadHoldTiming {
    /// A field's lifetime from its last write ([`EXT_READ_HOLD_TTL`]).
    pub ttl: std::time::Duration,
    /// The renewal interval ([`EXT_READ_HOLD_RENEW_INTERVAL`]); must be well
    /// below `ttl`.
    pub renew_every: std::time::Duration,
}

impl ReadHoldTiming {
    /// 15 s TTL, renewed every 5 s.
    pub const PRODUCTION: ReadHoldTiming = ReadHoldTiming {
        ttl: EXT_READ_HOLD_TTL,
        renew_every: EXT_READ_HOLD_RENEW_INTERVAL,
    };
}

/// What [`ReadHold::renew`] found on the `{gc}` primary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadHoldRenewal {
    /// The holder's field was live and is extended: the hold has protected
    /// the holder without a gap since it was taken.
    Live,
    /// The field had expired or was gone (a renewal that came too late, or a
    /// field lost to a `{gc}` failover). It was not revived: the holder's
    /// protection may have had a gap, so nothing it read since can be trusted
    /// to be complete.
    Lapsed,
}

/// Proof that the `{gc}` primary answered a [`ReadHold`]'s field live on a
/// genuine-miss check ([`ReadHold::confirm_live`]): the hold had protected
/// its holder without a gap since it was taken. The capability that builds
/// the fetch-on-miss read mode
/// ([`ExtValueReads::recover`](crate::storage::diff_read::ExtValueReads::recover)),
/// so a build reads external values through the checkpoint adapter only
/// after the store confirmed a live hold (b.yuo, SDD §11.2).
///
/// Its private field means only this module builds one, and only
/// [`ReadHold::confirm_live`] does; it borrows the hold, so it cannot outlive
/// the [`ReadHold`] (whose [`ReadHold::release`] consumes it). It proves the
/// answer at the time of the check, not that the field is still live later:
/// the caller builds under the hold's renewal.
#[derive(Debug)]
pub struct LiveHoldProof<'h> {
    _hold: std::marker::PhantomData<&'h ReadHold>,
}

/// One holder's `{gc}` read hold (b.yuo, the catch-up miss protocol; SDD
/// §11.2, §11.3): a field `<object_id>:<token>` on `le:ext:{gc}:read_hold`
/// ([`KeySchema::ext_read_hold`]) whose value is `<start_ms>:<expires_ms>`.
///
/// While the field is live, the GC loop (`gc_select.lua`) collects no release
/// enqueued within its margin of the earliest live hold's start, or later, so
/// every blob a read taken after the hold began references stays stored.
///
/// A catch-up whose build found a blob missing [`Self::take`]s one before it
/// reads the layer's state again, [`Self::renew`]s it every
/// [`ReadHoldTiming::renew_every`] while the re-read and rebuild run — the
/// renewal also proves, on the server, that the hold never lapsed — and
/// [`Self::release`]s it on every return. A guard dropped unreleased (a
/// cancelled future) spawns a best-effort `HDEL`; a field whose holder died
/// lapses one TTL after its last write. Every failure direction protects
/// longer, never less: a late `take` adds a field that lapses, a late
/// `renew` answers [`ReadHoldRenewal::Lapsed`], a failed release lapses by
/// TTL.
pub struct ReadHold {
    client: ValkeyClient,
    key: String,
    field: String,
    timing: ReadHoldTiming,
    /// The field is gone (released), so the drop has nothing to clean up.
    released: bool,
}

impl std::fmt::Debug for ReadHold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadHold")
            .field("field", &self.field)
            .field("timing", &self.timing)
            .field("released", &self.released)
            .finish_non_exhaustive()
    }
}

impl ReadHold {
    /// Take a hold for `object_id`: one `EVALSHA` of
    /// `scripts/ext_read_hold.lua` in `take` mode on the data pool, under a
    /// fresh random 128-bit token, with the client's TTL
    /// (`ValkeyClient::read_hold_timing`). The script writes this field,
    /// prunes the other expired ones and returns its start.
    ///
    /// No retry of its own: the `EVALSHA` keeps the data pool's fred retries
    /// (its default command attempts, and a `NOSCRIPT` reload), and the caller
    /// ends its catch-up `UNAVAILABLE` on the error that survives them. A
    /// re-send is harmless: it rewrites the same field, and the script keeps
    /// a live field's start. A transient failure is classified by
    /// [`map_fred_err_for_grpc`] for its log line and timeout count. A take
    /// whose reply was lost may still have written the field; the guard it
    /// would have returned is dropped unreleased, which spawns the `HDEL`.
    pub async fn take(
        client: &ValkeyClient,
        keys: &KeySchema,
        object_id: u64,
    ) -> Result<ReadHold, ServiceError> {
        // Test-only: park here — nothing sent, no guard built — when a test
        // armed a take hold for this object (`ReadHoldTakeHold`). Gated behind
        // `test-fault-injection` so release builds never see this branch.
        #[cfg(any(test, feature = "test-fault-injection"))]
        if let Some(held) = client.read_hold_take_holds.take(object_id) {
            held.park().await;
        }
        // The release shape's unit seam is read here only, so the field it
        // lives in is not dead code there.
        #[cfg(not(any(test, feature = "test-fault-injection")))]
        let _ = &client.read_hold_take_holds;
        let hold = ReadHold {
            client: client.clone(),
            key: keys.ext_read_hold(),
            field: [
                object_id.to_string().as_str(),
                ":",
                &format!("{:032x}", rand::random::<u128>()),
            ]
            .concat(),
            timing: client.read_hold_timing(),
            released: false,
        };
        hold.run(READ_HOLD_MODE_TAKE).await?;
        Ok(hold)
    }

    /// Renew this hold: `ext_read_hold.lua` in `renew` mode. A live field is
    /// extended by the TTL ([`ReadHoldRenewal::Live`]); an expired or missing
    /// one is deleted and NOT revived ([`ReadHoldRenewal::Lapsed`]). Errors
    /// as on [`Self::take`].
    pub async fn renew(&self) -> Result<ReadHoldRenewal, ServiceError> {
        Ok(if self.run(READ_HOLD_MODE_RENEW).await? > 0 {
            ReadHoldRenewal::Live
        } else {
            ReadHoldRenewal::Lapsed
        })
    }

    /// The genuine-miss check (SDD §11.2): one [`Self::renew`], and, when the
    /// store answers the field live, the [`LiveHoldProof`] that answer is —
    /// the only way to obtain one, and so the only way to build the
    /// fetch-on-miss read mode
    /// ([`ExtValueReads::recover`](crate::storage::diff_read::ExtValueReads::recover)).
    /// `None` when the
    /// store answered the field lapsed or gone. Errors as on [`Self::take`].
    pub async fn confirm_live(&self) -> Result<Option<LiveHoldProof<'_>>, ServiceError> {
        Ok(match self.renew().await? {
            ReadHoldRenewal::Live => Some(LiveHoldProof {
                _hold: std::marker::PhantomData,
            }),
            ReadHoldRenewal::Lapsed => None,
        })
    }

    /// Release this hold: `HDEL` its field, at once, so the GC stops clipping
    /// for it. A failed `HDEL` is only logged (DEBUG): the field lapses one
    /// TTL after its last write.
    pub async fn release(mut self) {
        let deleted: Result<i64, _> = self.client.pool.hdel(&self.key, &self.field).await;
        // Marked only once the `HDEL` returned: a release cancelled mid-flight
        // still leaves the drop's spawned `HDEL` behind it.
        self.released = true;
        if let Err(error) = deleted {
            FredErrOrigin::client(&self.client, POOL_LABEL_DATA).observe_error(&error);
            tracing::debug!(
                field = %self.field,
                error = %error,
                "GC read hold release failed; the hold lapses by its TTL"
            );
        }
    }

    /// The renewal interval this hold's holder renews it at.
    pub fn renew_every(&self) -> std::time::Duration {
        self.timing.renew_every
    }

    /// This hold's field on `le:ext:{gc}:read_hold`, `<object_id>:<token>`.
    pub fn field(&self) -> &str {
        &self.field
    }

    async fn run(&self, mode: &str) -> Result<u64, ServiceError> {
        let start: i64 = EXT_READ_HOLD_SCRIPT
            .evalsha_with_reload(
                self.client.pool.next(),
                vec![self.key.clone()],
                vec![
                    mode.to_owned(),
                    self.field.clone(),
                    duration_ms(self.timing.ttl).to_string(),
                ],
            )
            .await
            .map_err(|e| {
                map_fred_err_for_grpc(
                    FredErrOrigin::client(&self.client, POOL_LABEL_DATA),
                    "EVALSHA ext_read_hold",
                    e,
                )
            })?;
        u64::try_from(start).map_err(|_| {
            ServiceError::Internal(
                [
                    "ext_read_hold answered a negative hold start: ",
                    &start.to_string(),
                ]
                .concat(),
            )
        })
    }
}

impl Drop for ReadHold {
    fn drop(&mut self) {
        if self.released {
            return;
        }
        // No runtime (a drop during late shutdown): spawn nothing; the field
        // lapses on its own expiry.
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let pool = self.client.pool.clone();
        let key = std::mem::take(&mut self.key);
        let field = std::mem::take(&mut self.field);
        runtime.spawn(async move {
            let deleted: Result<i64, fred::error::Error> = pool.hdel(&key, &field).await;
            if let Err(error) = deleted {
                tracing::debug!(
                    field = %field,
                    error = %error,
                    "GC read hold release on drop failed; the hold lapses by its TTL"
                );
            }
        });
    }
}

/// `ext_read_hold.lua`'s ARGV[1] for [`ReadHold::take`].
const READ_HOLD_MODE_TAKE: &str = "take";
/// `ext_read_hold.lua`'s ARGV[1] for [`ReadHold::renew`].
const READ_HOLD_MODE_RENEW: &str = "renew";

// ── Pre-flight: store external values before object transaction ────────────

/// Scan an OODF delta for external value references, resolve each to its
/// OMPF data blob via `ext_value_index`, check every reference's part against
/// its declared size and content hash, and store them via the
/// [`ExternalValueStore`]. Deduplicates within a single upload so that the
/// same hash referenced by multiple fields/time samples results in only one
/// entry in the store call.
///
/// Must be called BEFORE the WATCH/MULTI/EXEC object transaction (SDD §11.1).
/// This — or its two halves, below — is the one place a CreateLayer /
/// UpdateLayer upload's untrusted external-value bytes enter the store, and it
/// runs in this order:
///
/// 1. [`collect_external_values`] — hash length, index resolution and dedup.
///    A claim that is not exactly 32 bytes, and a zero or out-of-range
///    `ext_value_index`, are refused here, before any hashing.
/// 2. Early return when nothing was collected.
/// 3. The content check ([`verify_external_values`]) under the store's
///    [`ExternalValueStore::ext_value_verify`] policy: every reference's raw
///    claim must be exactly 32 bytes, its part's length must equal its
///    `ext_value_size`, and the part's [`ext_content_hash`] must equal the
///    claim.
/// 4. The store: [`ExternalValueStore::store_verified_values`] in `enforce`
///    mode, [`ExternalValueStore::store_values`] in `observe` mode.
///
/// In `enforce` mode (the default, and the client contract) every refusal in
/// steps 1 and 3 returns before the first store, so a refused upload writes no
/// `ext_data` / `ext_meta`, pays no `+1`, enqueues nothing on the mirror, and —
/// because both handlers run steps 1–3 before they allocate an object id or
/// commit — allocates and commits nothing. An upload that passes is stored as
/// verified, so an existing data key under one of its hashes that is not a
/// string, or whose length differs from the verified bytes, is replaced with
/// them. In `observe` mode a size or hash mismatch is counted and logged and
/// the upload is stored anyway, set-if-absent and never healing; a
/// wrong-length claim and a bad index are still refused.
///
/// Steps 1–3 are [`prepare_external_values`] and step 4 is
/// [`PreparedExternalValues::store`]; this is the two in sequence, for an RPC
/// that stores once (`CreateLayer`). `UpdateLayer` calls the two halves
/// itself, because its checkpoint-recovery re-issue stores the same values a
/// second time without checking them again.
///
/// # Errors
///
/// [`ServiceError::InvalidArgument`] from steps 1 and 3; whatever the store
/// returns from step 4. No-ops when `delta` is `None` or contains no external
/// value references.
pub async fn store_external_values(
    delta: Option<&OwnedDelta>,
    ompf_external_values: &[&[u8]],
    ext_store: &dyn ExternalValueStore,
) -> Result<(), ServiceError> {
    prepare_external_values(delta, ompf_external_values, ext_store)?
        .store(ext_store)
        .await
}

/// Steps 1–3 of [`store_external_values`] — collect, then the content check
/// under the store's [`ExternalValueStore::ext_value_verify`] policy — run
/// once, returning what step 4 stores ([`PreparedExternalValues::store`]).
/// Synchronous: nothing is stored, incremented or mirrored here.
///
/// The check ticks `live_edit_ext_value_verify_total` once per checked value
/// reference and, in `observe` mode, logs its one WARN, here and only here; a
/// later store of the result checks nothing again.
///
/// # Errors
///
/// [`ServiceError::InvalidArgument`] from steps 1 and 3, before anything is
/// stored. A `delta` of `None`, or one with no external value references,
/// prepares an empty set.
pub fn prepare_external_values<'a>(
    delta: Option<&OwnedDelta>,
    ompf_external_values: &[&'a [u8]],
    ext_store: &dyn ExternalValueStore,
) -> Result<PreparedExternalValues<'a>, ServiceError> {
    let verify = ext_store.ext_value_verify();
    let empty = PreparedExternalValues {
        values: Vec::new(),
        mode: verify.mode,
    };
    let Some(delta) = delta else {
        return Ok(empty);
    };

    let values = collect_external_values(delta, ompf_external_values)?;
    if values.is_empty() {
        return Ok(empty);
    }

    verify_external_values(delta, ompf_external_values, verify)?;

    Ok(PreparedExternalValues {
        values,
        mode: verify.mode,
    })
}

/// An upload's external values that passed [`prepare_external_values`]'s
/// collect and content check, with the verify mode that check ran under:
/// checked once, stored any number of times ([`Self::store`]).
///
/// It exists for the `UpdateLayer` checkpoint-recovery re-issue, which must
/// pre-store the RPC's values a second time, as every `update_layer` call of
/// the RPC is preceded by its own pre-store (SDD §11.1, §11.3). Running
/// [`store_external_values`] again would count every reference twice on
/// `live_edit_ext_value_verify_total`, whose contract is one count per checked
/// value reference, WARN twice in `observe` mode, and hash up to the whole
/// upload's external bytes again. The fields are private, so only
/// [`prepare_external_values`] builds one: a value cannot be stored as
/// verified without having passed the check.
///
/// Borrows the OMPF parts it resolved; no value bytes are copied until a
/// store copies them.
pub struct PreparedExternalValues<'a> {
    /// The distinct `(hash, part)` pairs, deduplicated by hash in first
    /// appearance order ([`collect_external_values`]).
    values: Vec<ExtValueRef<'a>>,
    /// The mode the content check ran under, which picks step 4's store.
    mode: ExtValueVerifyMode,
}

impl PreparedExternalValues<'_> {
    /// Whether there is nothing to store: no delta, or no external value
    /// references.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The bytes one store copies: the sum of the distinct stored parts'
    /// lengths, one per hash ([`ValkeyExternalValueStore`] makes one owned
    /// copy per stored value).
    pub fn stored_bytes(&self) -> u64 {
        self.values
            .iter()
            .map(|(_, data)| data.len() as u64)
            .fold(0u64, u64::saturating_add)
    }

    /// Step 4 of [`store_external_values`], in the verify mode the check ran
    /// under: [`ExternalValueStore::store_verified_values`] in `enforce`
    /// mode, [`ExternalValueStore::store_values`] in `observe` mode. Each call
    /// is a full store — one `+1` per distinct hash, one mirror enqueue per
    /// value — and checks nothing again. A no-op when [`Self::is_empty`].
    ///
    /// # Errors
    ///
    /// Whatever the store returns.
    pub async fn store(&self, ext_store: &dyn ExternalValueStore) -> Result<(), ServiceError> {
        if self.values.is_empty() {
            return Ok(());
        }
        let pairs: Vec<(&[u8; 32], &[u8])> = self.values.iter().map(|(h, d)| (h, *d)).collect();
        match self.mode {
            // Every pair passed the check (the gate refuses otherwise), so the
            // store may heal a damaged existing copy with these bytes.
            ExtValueVerifyMode::Enforce => ext_store.store_verified_values(&pairs).await,
            // The check may have failed and been let through: never heal.
            ExtValueVerifyMode::Observe => ext_store.store_values(&pairs).await,
        }
    }
}

/// Block size of the external-value content hash ([`ext_content_hash`]):
/// 1,048,576 bytes. A fixed protocol constant, not a setting — clients hash
/// with it and no client negotiates it.
pub const EXT_CONTENT_HASH_BLOCK_BYTES: usize = 1 << 20;

/// The content hash an external value's `valueOrExtHash` carries: SHA-256 over
/// the concatenated SHA-256 digests of the value's consecutive
/// [`EXT_CONTENT_HASH_BLOCK_BYTES`]-byte blocks, the last block holding the
/// remainder.
///
/// This is the clients' definition, not a server choice, and it is NOT plain
/// SHA-256 of the bytes: a non-empty value of at most one block (1 to
/// 1,048,576 bytes) hashes to `SHA-256(SHA-256(value))`. The shape (one outer
/// hasher fed each block's digest) is the Rust client library's own.
///
/// Two edge cases follow from `chunks` never yielding an empty chunk:
///
/// - a value whose length is an exact multiple of the block size has no empty
///   final block;
/// - an empty value has no blocks and hashes to SHA-256 of the empty string
///   (`e3b0c442…b855`), as the clients' zero-block path does.
///
/// Streams: the outer hasher takes one 32-byte digest per block, so the cost
/// is one SHA-256 pass over the value plus one 32-byte update per MiB, with no
/// allocation and no copy. SDD §11.1.
pub fn ext_content_hash(bytes: &[u8]) -> [u8; 32] {
    use sha2::{Digest, Sha256};

    let mut outer = Sha256::new();
    for block in bytes.chunks(EXT_CONTENT_HASH_BLOCK_BYTES) {
        outer.update(Sha256::digest(block));
    }
    outer.finalize().into()
}

/// How the upload content check treats a reference that fails it, and where
/// the check is counted. [`prepare_external_values`] reads it from the store
/// through [`ExternalValueStore::ext_value_verify`].
#[derive(Clone, Copy)]
pub struct ExtValueVerify<'a> {
    /// `LIVE_EDIT_EXT_VALUE_VERIFY`: refuse (`Enforce`) or count, log and
    /// store anyway (`Observe`).
    pub mode: ExtValueVerifyMode,
    /// Where `live_edit_ext_value_verify_total` is counted. `None` counts
    /// nothing.
    pub metrics: Option<&'a SharedMetrics>,
}

impl ExtValueVerify<'_> {
    /// Enforce, counted nowhere: the policy a test double with none of its
    /// own states in [`ExternalValueStore::ext_value_verify`].
    pub const ENFORCE_UNCOUNTED: ExtValueVerify<'static> = ExtValueVerify {
        mode: ExtValueVerifyMode::Enforce,
        metrics: None,
    };
}

/// The content check on a CreateLayer / UpdateLayer upload's external values:
/// every reference must name a part whose length is its `ext_value_size` and
/// whose [`ext_content_hash`] is exactly the hash it claims. Runs under
/// `verify`'s mode and is counted on `verify`'s metrics
/// ([`ExtValueVerify::ENFORCE_UNCOUNTED`] for a plain enforcing check).
///
/// Walks `delta.set_fields`, then `delta.time_samples`, and for **every**
/// reference with `ext_value_size > 0` — not only the first appearance of each
/// hash — in this order:
///
/// 1. takes the **raw** `value_or_ext_hash` and refuses one that is not exactly
///    32 bytes, in both modes, because it breaks the protocol's 32-byte rule.
///    The refusal is the same [`DeltaError::InvalidExternalHashLength`]
///    `apply_delta` produces. [`store_external_values`]'s collector refuses
///    such a claim first ([`collect_external_values`]); this check refuses it
///    for any other caller, and `observe` deliberately does not relax it;
/// 2. resolves the part with `resolve_ext_index` (a bad index is refused in
///    both modes);
/// 3. requires the part's length to equal `ext_value_size`, the value's size
///    in bytes (a `size_mismatch`);
/// 4. requires [`ext_content_hash`] of the part to equal the claim (a
///    `hash_mismatch`).
///
/// **Modes.** In `Enforce` the first size or hash mismatch is refused
/// `INVALID_ARGUMENT`, with text naming the rule so a client that computed
/// something else can diagnose itself. In `Observe` nothing is refused for a
/// size or hash mismatch and the walk goes on to the next reference. When the
/// whole walk passes (so the call proceeds to the store) and found any
/// mismatch, one WARN per call carries the mismatch counts and the first
/// failing reference's `ext_value_index`, claimed hash, kind and sizes — never
/// value bytes. An observe-mode call refused by a later wrong-length claim or
/// bad index stores nothing and is not WARNed; its refusal surfaces on the RPC
/// failure line. The WARN is emitted inside the RPC's `grpc.server` span,
/// which names the RPC; the session and layer are not in scope here.
///
/// **Counting.** `live_edit_ext_value_verify_total{result}` gains one count
/// per checked reference, in both modes: `ok`, `size_mismatch` (the hash is
/// then not checked) or `hash_mismatch`. In `Enforce` the references after the
/// refused one are not checked and not counted. Wrong-length claims and bad
/// indexes are refusals, not results, and are not counted.
///
/// Every reference is checked, not only each hash's first appearance, because
/// UpdateLayer relays the uploaded OMPF container verbatim to steady-state
/// subscribers (SDD §12): a second reference to an already-checked hash that
/// names a different part would otherwise put a part that fails its claim on
/// the wire. Hashes are memoized per part index, so each part is hashed at
/// most once — CPU is at most one pass over the parts, memory is one 32-byte
/// digest per distinct referenced index, and no part is copied. Hashing runs
/// inline and synchronously, as `parse_delta` does on the same task.
///
/// Deliberately NOT folded into [`collect_external_values`]: that collector is
/// shared with `checkpoint::compose`, which replays only records of commits
/// that landed and must not re-hash a whole lineage on every restore and
/// re-base.
///
/// # Errors
///
/// [`ServiceError::InvalidArgument`] for the first refused reference, in walk
/// order: a wrong-length claim or a zero or out-of-range `ext_value_index` in
/// either mode, and in `Enforce` a size or hash mismatch.
pub(crate) fn verify_external_values(
    delta: &OwnedDelta,
    ompf_external_values: &[&[u8]],
    verify: ExtValueVerify<'_>,
) -> Result<(), ServiceError> {
    let mut tally = VerifyTally::default();
    let outcome = check_external_values(delta, ompf_external_values, verify.mode, &mut tally);
    tally.count(verify.metrics);
    // Only a call that proceeds to the store is WARNed: an observe-mode call
    // refused for another reason (a wrong-length claim later in the walk)
    // stores nothing, and its refusal already surfaces on the RPC failure line.
    if verify.mode == ExtValueVerifyMode::Observe && outcome.is_ok() {
        if let Some(first) = tally.first_mismatch.as_ref() {
            tracing::warn!(
                ext_value_index = first.ext_value_index,
                claimed_hash = %sha256_to_hex(&first.claim),
                mismatch = first.kind,
                part_len = first.part_len,
                ext_value_size = first.ext_value_size,
                size_mismatches = tally.size_mismatch,
                hash_mismatches = tally.hash_mismatch,
                "external value failed the upload content check; the upload proceeds to the \
                 store anyway because LIVE_EDIT_EXT_VALUE_VERIFY=observe. Fields \
                 describe the first failing reference"
            );
        }
    }
    outcome
}

/// The checked-reference counts of one content check, and the first failing
/// reference for the `observe` WARN.
#[derive(Default)]
struct VerifyTally {
    ok: u64,
    size_mismatch: u64,
    hash_mismatch: u64,
    first_mismatch: Option<VerifyMismatch>,
}

/// What the `observe` WARN says about a failing reference. No value bytes.
struct VerifyMismatch {
    kind: &'static str,
    ext_value_index: u32,
    claim: [u8; 32],
    part_len: usize,
    ext_value_size: u64,
}

impl VerifyTally {
    fn mismatch(&mut self, mismatch: VerifyMismatch) {
        if mismatch.kind == EXT_VALUE_VERIFY_SIZE_MISMATCH {
            self.size_mismatch += 1;
        } else {
            self.hash_mismatch += 1;
        }
        self.first_mismatch.get_or_insert(mismatch);
    }

    /// One `add` per non-zero result, so a call with many references costs at
    /// most three counter updates.
    fn count(&self, metrics: Option<&SharedMetrics>) {
        let Some(m) = metrics else { return };
        for (result, n) in [
            (EXT_VALUE_VERIFY_OK, self.ok),
            (EXT_VALUE_VERIFY_SIZE_MISMATCH, self.size_mismatch),
            (EXT_VALUE_VERIFY_HASH_MISMATCH, self.hash_mismatch),
        ] {
            if n > 0 {
                m.add_ext_value_verify(result, n);
            }
        }
    }
}

/// The walk behind [`verify_external_values`]; see it for the order and
/// the modes. Records every checked reference in `tally`.
fn check_external_values(
    delta: &OwnedDelta,
    ompf_external_values: &[&[u8]],
    mode: ExtValueVerifyMode,
    tally: &mut VerifyTally,
) -> Result<(), ServiceError> {
    use std::collections::HashMap;

    let mut hashes: HashMap<u32, [u8; 32]> = HashMap::new();
    let refs = delta
        .set_fields
        .iter()
        .filter(|sf| sf.ext_value_size > 0)
        .map(|sf| {
            (
                sf.value_or_ext_hash.as_slice(),
                sf.ext_value_size,
                sf.ext_value_index,
            )
        })
        .chain(
            delta
                .time_samples
                .iter()
                .filter(|ts| ts.ext_value_size > 0)
                .map(|ts| {
                    (
                        ts.value_or_ext_hash.as_slice(),
                        ts.ext_value_size,
                        ts.ext_value_index,
                    )
                }),
        );
    for (raw_claim, ext_value_size, ext_value_index) in refs {
        // The RAW claim, never a padded or truncated form of it.
        let Ok(claim) = <&[u8; 32]>::try_from(raw_claim) else {
            return Err(ServiceError::from(DeltaError::InvalidExternalHashLength {
                got: raw_claim.len(),
                expected: 32,
            }));
        };
        let part = resolve_ext_index(ext_value_index, ompf_external_values)?;

        if part.len() as u64 != ext_value_size {
            tally.mismatch(VerifyMismatch {
                kind: EXT_VALUE_VERIFY_SIZE_MISMATCH,
                ext_value_index,
                claim: *claim,
                part_len: part.len(),
                ext_value_size,
            });
            if mode == ExtValueVerifyMode::Enforce {
                return Err(ServiceError::InvalidArgument(format!(
                    "external value at ext_value_index {ext_value_index} is {} bytes, but its \
                     extValueSize is {ext_value_size}",
                    part.len()
                )));
            }
            continue;
        }

        let hash = *hashes
            .entry(ext_value_index)
            .or_insert_with(|| ext_content_hash(part));
        if &hash != claim {
            tally.mismatch(VerifyMismatch {
                kind: EXT_VALUE_VERIFY_HASH_MISMATCH,
                ext_value_index,
                claim: *claim,
                part_len: part.len(),
                ext_value_size,
            });
            if mode == ExtValueVerifyMode::Enforce {
                return Err(ServiceError::InvalidArgument(format!(
                    "external value at ext_value_index {ext_value_index} does not match its \
                     valueOrExtHash {} (expected SHA-256 over the SHA-256 digests of the \
                     value's 1 MiB blocks)",
                    sha256_to_hex(claim)
                )));
            }
            continue;
        }

        tally.ok += 1;
    }
    Ok(())
}

/// One resolved external-value reference: the 32-byte content hash, and a
/// borrow of the OMPF part carrying its bytes. Spelled as an alias so the
/// collector's return type stays readable (and inside clippy's
/// `type_complexity` budget) at every call site.
pub(crate) type ExtValueRef<'a> = ([u8; 32], &'a [u8]);

/// Pure, synchronous collection step of [`store_external_values`]: scan an
/// OODF delta for external value references and resolve each to its OMPF data
/// blob. It hashes nothing — the content check is [`verify_external_values`],
/// which the live path runs after this and which compose deliberately does not.
///
/// Deduplicates by hash — the same hash referenced by several fields and/or
/// time samples yields exactly one pair, carrying the data slice resolved at
/// its *first* appearance. Index resolution still runs for every referencing
/// entry (including later duplicates), so a bad `ext_value_index` is rejected
/// wherever it appears.
///
/// **Output order is deterministic**: first-appearance order over
/// `delta.set_fields`, then `delta.time_samples`. No `HashMap` iteration is on
/// the output path; the `HashSet` is membership-only. The live path does not
/// care (per-hash writes are independent), but `checkpoint::compose` composes
/// this into a lineage-wide list whose equality is asserted, so the order is
/// part of the contract.
///
/// # Errors
///
/// [`ServiceError::InvalidArgument`] when an external entry's
/// `value_or_ext_hash` is not exactly 32 bytes (see `external_hash`), or when
/// an `ext_value_index` is `0` or is out of range for `ompf_external_values`
/// (see `resolve_ext_index`). Never panics on either: `checkpoint::compose`
/// runs this over stored stream records, which may predate the ingest-time
/// hash-length check.
pub(crate) fn collect_external_values<'a>(
    delta: &OwnedDelta,
    ompf_external_values: &[&'a [u8]],
) -> Result<Vec<ExtValueRef<'a>>, ServiceError> {
    use std::collections::HashSet;

    let mut seen: HashSet<[u8; 32]> = HashSet::new();
    let mut to_store: Vec<ExtValueRef<'a>> = Vec::new();

    // Scan SetField entries for external values.
    for sf in &delta.set_fields {
        if sf.ext_value_size > 0 {
            let hash = external_hash(&sf.value_or_ext_hash)?;
            let data = resolve_ext_index(sf.ext_value_index, ompf_external_values)?;
            if seen.insert(hash) {
                to_store.push((hash, data));
            }
        }
    }

    // Scan TimeSample entries for external values.
    for ts in &delta.time_samples {
        if ts.ext_value_size > 0 {
            let hash = external_hash(&ts.value_or_ext_hash)?;
            let data = resolve_ext_index(ts.ext_value_index, ompf_external_values)?;
            if seen.insert(hash) {
                to_store.push((hash, data));
            }
        }
    }

    Ok(to_store)
}

/// The distinct external-value hashes an OODF delta references — exactly the
/// hashes [`store_external_values`] pre-stores (and so pre-increments) for it,
/// in the same first-appearance order as [`collect_external_values`].
///
/// Needs no OMPF parts: by the time a caller asks, the delta has already been
/// through [`store_external_values`], which rejected any bad
/// `ext_value_index`. Used to hand back a call's pre-flight increments when it
/// commits nothing because its client `operation_id` was already recorded
/// ([`ReleaseEvent::Resend`]).
pub(crate) fn delta_external_hashes(delta: &OwnedDelta) -> Vec<[u8; 32]> {
    use std::collections::HashSet;

    let mut seen: HashSet<[u8; 32]> = HashSet::new();
    let mut hashes = Vec::new();
    let refs = delta
        .set_fields
        .iter()
        .filter(|sf| sf.ext_value_size > 0)
        .map(|sf| sf.value_or_ext_hash.as_slice())
        .chain(
            delta
                .time_samples
                .iter()
                .filter(|ts| ts.ext_value_size > 0)
                .map(|ts| ts.value_or_ext_hash.as_slice()),
        );
    for raw in refs {
        // [`external_hash`], the rule [`collect_external_values`] applies, so
        // the two never disagree about which hash a reference pre-stored. A
        // reference that is not a 32-byte hash was refused before anything was
        // pre-stored (`store_external_values`, `validate_delta_bounds`), so it
        // pre-stored nothing and is owed nothing back.
        let Ok(hash) = external_hash(raw) else {
            continue;
        };
        if seen.insert(hash) {
            hashes.push(hash);
        }
    }
    hashes
}

/// Collect all external value hashes from an [`ObjectState`].
///
/// Used by the CreateLayer overwrite path to schedule GC for the old object's
/// external values (SDD §11.3).
pub fn collect_external_hashes(state: &ObjectState) -> Vec<[u8; 32]> {
    use std::collections::HashSet;
    let mut hashes = HashSet::new();

    for field in state.fields.values() {
        if let FieldValue::External { hash, .. } = &field.value {
            hashes.insert(*hash);
        }
    }
    for sample in state.time_samples.values() {
        if let FieldValue::External { hash, .. } = &sample.value {
            hashes.insert(*hash);
        }
    }

    hashes.into_iter().collect()
}

/// An external value's `value_or_ext_hash` as its 32-byte SHA-256 hash —
/// shared by [`collect_external_values`] and [`delta_external_hashes`].
/// Returns `InvalidArgument` — the same status and message
/// `validate_delta_bounds` produces — for any other length; the bytes are
/// never padded or truncated into a hash the client did not send.
fn external_hash(value_or_ext_hash: &[u8]) -> Result<[u8; 32], ServiceError> {
    <[u8; 32]>::try_from(value_or_ext_hash).map_err(|_| {
        ServiceError::from(DeltaError::InvalidExternalHashLength {
            got: value_or_ext_hash.len(),
            expected: 32,
        })
    })
}

/// Resolve a 1-based `ext_value_index` to the corresponding OMPF external
/// value slice. Returns `InvalidArgument` if the index is 0 or out of range.
fn resolve_ext_index<'a>(
    ext_value_index: u32,
    ompf_external_values: &[&'a [u8]],
) -> Result<&'a [u8], ServiceError> {
    if ext_value_index == 0 {
        return Err(ServiceError::InvalidArgument(
            "ext_value_index must be non-zero for external values".to_owned(),
        ));
    }
    let zero_based = (ext_value_index - 1) as usize;
    ompf_external_values
        .get(zero_based)
        .copied()
        .ok_or_else(|| {
            ServiceError::InvalidArgument(format!(
                "ext_value_index {ext_value_index} out of range (OMPF has {} external parts)",
                ompf_external_values.len()
            ))
        })
}

// ── Checkpoint-adapter ext-value recovery (both stores) ────────────────────

/// Whether `ValkeyExternalValueStore::store_values_with` sends a committed
/// value's copy to the wired ext-mirror sender (SDD §22.17).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExtMirroring {
    /// Send it, as every store path does: `store_values`,
    /// `store_verified_values`, and through them the upload pre-store and the
    /// composed-restore pre-store (`checkpoint::apply::ExtPrestore`).
    Send,
    /// Send nothing and take no byte charge: only
    /// [`ExtRecoveryRewrite::store_recovered_values`].
    Skip,
}

/// The store-specific step of [`recover_ext_value_via_checkpoint`], which
/// [`ValkeyExternalValueStore`] and [`MockExternalValueStore`] implement.
#[async_trait]
trait ExtRecoveryRewrite: Send + Sync {
    /// Rewrite bytes fetch-on-miss recovery just read from the checkpoint
    /// adapter. Every store effect is exactly [`ExternalValueStore::store_values`]'s
    /// — the set-if-absent data write, the `+1` on the count, the
    /// `last_store_ms` stamp, the `-OOM` refusal and the error mapping — but
    /// no copy goes to the ext mirror and no `M_ext` byte charge is taken.
    ///
    /// Why: the bytes came from the backup store, which already holds them, so
    /// a mirror copy would be a redundant PUT the adapter answers 412, while
    /// its charge holds `M_ext` — about one maximum-size value at the smallest
    /// pods — away from live uploads' copies, which would then be dropped
    /// `budget_full`. Every other store path keeps mirroring, the
    /// composed-restore pre-store included. Its values need not be among the
    /// backup store's external-value blobs: the upload's mirror copy may have
    /// been dropped, or the backup store may have deleted the blob since,
    /// which is why the tail carries the bytes. The pre-store's own mirror copy
    /// is what makes sure one exists (`checkpoint::apply::ExtPrestore`), and
    /// the backup repair re-sends it when that copy is lost. A recovered value
    /// gets no worker confirmation; the repair's existence check confirms it.
    /// SDD §22.17.
    async fn store_recovered_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), ServiceError>;
}

#[async_trait]
impl ExtRecoveryRewrite for ValkeyExternalValueStore {
    async fn store_recovered_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), ServiceError> {
        self.store_values_with(values, false, ExtMirroring::Skip)
            .await
    }
}

#[async_trait]
impl ExtRecoveryRewrite for MockExternalValueStore {
    /// The mock has no mirror, so this is its `store_values`.
    async fn store_recovered_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), ServiceError> {
        self.store_values_with(values, false);
        Ok(())
    }
}

/// Recover one missing ext-value blob through the checkpoint adapter (SDD
/// SR-3.2): the policy [`ValkeyExternalValueStore`] and
/// [`MockExternalValueStore`] share. `store`'s
/// [`ExtRecoveryRewrite::store_recovered_values`] — the rewrite of the
/// recovered bytes, which sends no mirror copy — is the only store-specific
/// step.
///
/// - `checkpoint_store` `None` → `Err(original_error)`, no adapter consult and
///   no counter tick.
/// - The adapter fetch is single-flight per `hash_hex` through `singleflight`,
///   so a subscribe storm against a freshly recovered layer costs one adapter
///   call and one rewrite per hash.
/// - `Ok(Some(bytes))` → rewrite through `store.store_recovered_values`, tick
///   `recovered`, bump `recovery_counter`, return the bytes.
/// - `Ok(None)` (`adapter_miss`), `Permanent` (`adapter_permanent`) and any
///   other variant (`adapter_other` — `get_ext_blob` never returns them) →
///   `Err(original_error)`.
/// - `Transient` (`adapter_transient`) → `Err(ServiceError::Unavailable)`, so
///   the client retries.
///
/// Every consulted outcome ticks `live_edit_checkpoint_ext_recovery_total` with
/// the label in parentheses above and logs it at debug.
#[allow(clippy::too_many_arguments)]
async fn recover_ext_value_via_checkpoint(
    store: &(dyn ExtRecoveryRewrite + '_),
    checkpoint_store: Option<Arc<dyn CheckpointStore>>,
    singleflight: &ExtRecoverySingleflight,
    metrics: Option<&SharedMetrics>,
    hash_hex: &str,
    hash: &[u8; 32],
    original_error: ServiceError,
    recovery_counter: Option<&std::sync::atomic::AtomicUsize>,
) -> Result<Vec<u8>, ServiceError> {
    let Some(checkpoint_store) = checkpoint_store else {
        // No adapter wired: the original error surfaces with no counter tick
        // (the metric counts "consulted the adapter" outcomes only, so
        // checkpoint-disabled replicas publish no permanent baseline).
        return Err(original_error);
    };

    let key = hash_hex.to_owned();
    let key_for_closure = key.clone();
    let fetch_result = singleflight
        .fetch_once(key, move || async move {
            checkpoint_store.get_ext_blob(&key_for_closure).await
        })
        .await;

    match fetch_result {
        Ok(Some(bytes)) => {
            // Rewrite into Valkey before returning so a follow-up
            // `read_batch` for the same hash hits Valkey alone. The data half
            // of the store's per-hash script is `SET NX`, so whichever
            // coalesced waiter (or concurrent upload) stores first keeps its
            // copy and the rest write nothing — except a concurrent verified
            // upload, which replaces it only if its length differs. This is
            // an unverified store, so it never heals. This path is entered
            // only from the `(None, _)` arms of `read_batch`, i.e. only when `GET
            // ext_data` found no key at all, so the `NX` writes here exactly as
            // an unconditional `SET` would. The bytes are not re-hashed: the
            // adapter holds the first copy a store mirrored, which a gated
            // upload verified, or which predates the gate or was admitted
            // under `LIVE_EDIT_EXT_VALUE_VERIFY=observe` — see
            // `scripts/ext_store_value.lua`. Its `ref_count` increment is a
            // plain `+1`, and what that does to the count depends on which
            // half was missing (SDD §22.17.4):
            //
            // - Meta gone too (`MetaNotFound`, the normal case: the GC deletes
            //   meta and data together when the count reaches 0). The meta is
            //   recreated at 1 while the L live objects that reference the
            //   hash each still owe one release: exact for L = 1, a deficit of
            //   L − 1 for L ≥ 2 — an over-release still standing, which the
            //   reference-count reconciler confirms and reports but never
            //   repairs (`storage::ext_reconcile`, the deficit arm).
            // - Meta present, data gone (`DataNotFound`, which atomic create
            //   and delete make impossible; logged as an integrity error).
            //   The `+1` lands on a live count and no release is ever owed for
            //   it — a leak, in the same class as the failed-commit increment
            //   (SDD §11.1, §11.3). The reconciler's surplus arm drains it out
            //   of band — hours later, and only while the pass is enabled, so
            //   it is a backstop rather than this path's release accounting.
            //
            // The rewrite also re-stamps `last_store_ms`, which is what keeps
            // the reconciler from repairing against a recovery still in flight.
            //
            // Not `store_values`: these bytes came from the backup store, so
            // the rewrite sends no mirror copy and takes no `M_ext` charge.
            let pair: [(&[u8; 32], &[u8]); 1] = [(hash, bytes.as_ref())];
            store.store_recovered_values(&pair).await?;
            tracing::debug!(
                hash = %hash_hex,
                adapter_status = "recovered",
                "ext-value recovery via checkpoint adapter succeeded"
            );
            if let Some(m) = metrics {
                m.inc_ext_recovery("recovered");
            }
            // Subtask t3.fgj.zc.p6.78: bump the caller-provided recovery
            // counter so the version-diff build can emit a single INFO log line per
            // batch including the object_id (not in scope here). Every other
            // arm leaves it untouched, so the count is exactly the number of
            // hashes reconstructed via the adapter during this call.
            if let Some(c) = recovery_counter {
                c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            Ok(bytes.to_vec())
        }
        Ok(None) => {
            tracing::debug!(
                hash = %hash_hex,
                adapter_status = "adapter_miss",
                "ext-value recovery via checkpoint adapter returned None"
            );
            if let Some(m) = metrics {
                m.inc_ext_recovery("adapter_miss");
            }
            Err(original_error)
        }
        Err(CheckpointError::Transient { source }) => {
            tracing::debug!(
                hash = %hash_hex,
                adapter_status = "adapter_transient",
                error = %source,
                "ext-value recovery via checkpoint adapter returned Transient"
            );
            if let Some(m) = metrics {
                m.inc_ext_recovery("adapter_transient");
            }
            Err(ServiceError::Unavailable(format!(
                "checkpoint adapter transient failure during ext-value recovery: {source}"
            )))
        }
        Err(CheckpointError::Permanent { source }) => {
            tracing::debug!(
                hash = %hash_hex,
                adapter_status = "adapter_permanent",
                error = %source,
                "ext-value recovery via checkpoint adapter returned Permanent"
            );
            if let Some(m) = metrics {
                m.inc_ext_recovery("adapter_permanent");
            }
            Err(original_error)
        }
        Err(other) => {
            // `get_ext_blob`'s trait contract reserves `NotFound`, `Conflict`,
            // and `NotImplemented` to other methods — observing them here
            // means the adapter has drifted from the contract. Treat as
            // permanent (preserve the wire contract) but bucket separately so
            // operators can spot the drift.
            tracing::debug!(
                hash = %hash_hex,
                adapter_status = "adapter_other",
                error = %other,
                "ext-value recovery via checkpoint adapter returned unexpected variant"
            );
            if let Some(m) = metrics {
                m.inc_ext_recovery("adapter_other");
            }
            Err(original_error)
        }
    }
}

// ── Mock implementation ────────────────────────────────────────────────────

/// In-memory mock for unit testing components that depend on external value
/// storage. Behaves identically to [`ValkeyExternalValueStore`] in terms of
/// dedup, refcounting, and pending GC logic.
///
/// Like the real store, the mock is content-addressed: the "data key" is the
/// hash itself, with no secondary `value_id` indirection (SDD §11.1).
pub struct MockExternalValueStore {
    inner: std::sync::Mutex<MockState>,
    /// Per-hash singleflight guard mirroring the production store's
    /// SR-3.2 recovery primitive. Always allocated so unit-test
    /// fixtures can exercise the singleflight coalescing semantics
    /// against the mock adapter without a real Valkey backend.
    recovery_singleflight: Arc<ExtRecoverySingleflight>,
    /// Optional shared metrics handle. When `Some`, recovery outcomes
    /// tick `live_edit_checkpoint_ext_recovery_total{result=...}` —
    /// matching the production wiring. Also where the upload content check
    /// is counted.
    metrics: Option<SharedMetrics>,
    /// Upload content-check mode, mirroring
    /// [`ValkeyExternalValueStore::with_ext_value_verify`]. `Enforce` unless
    /// set.
    ext_value_verify: ExtValueVerifyMode,
}

struct MockState {
    /// hash → (data, ref_count)
    store: std::collections::HashMap<[u8; 32], (Vec<u8>, i64)>,
    /// Pending release set. Members are `ExtRelease::member()` strings, not
    /// bare hashes (b.qpf) — that is what makes two owed decrements for one
    /// hash distinct here exactly as they are in Valkey. Each maps to its
    /// first-enqueue time in wall-clock ms (the release ledger's age, b.yuo),
    /// which the mock's `process_pending` gates on `GcPolicy::min_age`.
    pending: std::collections::HashMap<String, u64>,
    /// In-flight GC set with claim timestamps (b.j9j item 2): the
    /// production store backs this with a sorted set scored by the
    /// (encoded) claim time, so a pass re-claims only stale entries. The mock
    /// mirrors the same shape via a HashMap — with plain claim times: it
    /// models only this build's own claims — so tests can pin both the
    /// "fresh claim left alone" and the "stale claim re-claimed" branches
    /// without a real Valkey ZSET.
    /// Key = pending member (b.qpf), value = wall-clock millis at claim /
    /// simulated crash.
    inflight: std::collections::HashMap<String, u64>,
    /// The mock's replay records (b.yuo): every member whose decrement has
    /// applied → the wall-clock ms it settled (left both sets), or `None`
    /// while it has not. A member found here is completed without
    /// decrementing, as `gc_decrement_and_collect.lua` answers `replayed`;
    /// an entry settled at least `GcPolicy::replay_record_retention` ago is
    /// torn down at the start of the next pass. No claim-age fence: the mock
    /// has no late copies of a decrement.
    applied: std::collections::HashMap<String, Option<u64>>,
    /// The mock's `{gc}` attempt record (b.yuo Part R): every retirement
    /// attempt token whose enqueue ran. A token found here answers
    /// `AlreadyLanded` without enqueuing again, as the attempt-mode
    /// `ext_release_enqueue.lua` answers `already`. The mock has no resolve,
    /// so nothing is ever recorded `void`.
    landed_attempts: std::collections::HashSet<String>,
    /// Optional `CheckpointStore` adapter that mirrors the SR-3.2
    /// recovery fallback `ValkeyExternalValueStore` engages on
    /// `read_batch` misses (Task `t2.fgj.zc.p6`). `None` keeps the
    /// pre-Task mock behaviour (miss → `MetaNotFound` straight to the
    /// caller); `Some(adapter)` enables the same singleflight-coalesced
    /// `get_ext_blob` → `store_recovered_values` rewrite path used in
    /// production.
    checkpoint_store: Option<Arc<dyn CheckpointStore>>,
}

impl MockExternalValueStore {
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(MockState {
                store: std::collections::HashMap::new(),
                pending: std::collections::HashMap::new(),
                inflight: std::collections::HashMap::new(),
                applied: std::collections::HashMap::new(),
                landed_attempts: std::collections::HashSet::new(),
                checkpoint_store: None,
            }),
            recovery_singleflight: Arc::new(ExtRecoverySingleflight::new()),
            metrics: None,
            ext_value_verify: ExtValueVerifyMode::Enforce,
        }
    }

    /// Builder: set the upload content-check mode. Mirrors
    /// [`ValkeyExternalValueStore::with_ext_value_verify`]; the check is
    /// counted on the [`Self::with_metrics`] handle.
    pub fn with_ext_value_verify(mut self, mode: ExtValueVerifyMode) -> Self {
        self.ext_value_verify = mode;
        self
    }

    /// **Test-only.** Enqueue `releases` onto the mock's pending set with no
    /// attempt token — the plain enqueue the trait's `schedule_release` used to
    /// be before b.yuo Part R moved the retirement onto
    /// [`ExternalValueStore::enqueue_retirement_attempt`]. Kept for tests that
    /// seed pending members directly. Compiled only for tests and the
    /// `test-support` feature.
    #[cfg(any(test, feature = "test-support"))]
    pub async fn schedule_release(&self, releases: &[ExtRelease]) -> Result<(), ServiceError> {
        let mut state = self.inner.lock().unwrap();
        Self::enqueue_members(&mut state, releases);
        Ok(())
    }

    /// Add `releases` to the mock's pending set. Member-keyed, exactly like the
    /// production `SADD` (b.qpf): a replay of one event collapses, two distinct
    /// events do not. A replayed enqueue keeps its first age, as `ZADD NX`
    /// does.
    fn enqueue_members(state: &mut MockState, releases: &[ExtRelease]) {
        for release in releases {
            state
                .pending
                .entry(release.member())
                .or_insert_with(now_unix_millis);
        }
    }

    /// Builder: attach a `CheckpointStore` adapter so `read_batch`
    /// misses fall back to the same SR-3.2 recovery path the
    /// production `ValkeyExternalValueStore` engages. Mirrors
    /// [`ValkeyExternalValueStore::with_checkpoint_store`] so unit
    /// tests can drive the fallback without a real Valkey backend.
    pub fn with_checkpoint_store(self, store: Arc<dyn CheckpointStore>) -> Self {
        {
            let mut state = self.inner.lock().unwrap();
            state.checkpoint_store = Some(store);
        }
        self
    }

    /// Builder: attach a shared metrics handle so recovery outcomes
    /// tick `live_edit_checkpoint_ext_recovery_total{result=...}`.
    /// Mirrors [`ValkeyExternalValueStore::with_metrics`] so tests
    /// that need to assert the counter ticked have a constructor that
    /// gives them a handle.
    pub fn with_metrics(mut self, metrics: SharedMetrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Mock-side twin of [`ValkeyExternalValueStore::try_recover_via_checkpoint`]:
    /// both delegate to [`recover_ext_value_via_checkpoint`], so the recovery
    /// policy — singleflight, rewrite, error classification, logs, metric
    /// labels, recovery counting — is one body. `checkpoint_store` is passed in
    /// rather than read here: the caller clones it out under the mock's mutex
    /// and drops the guard first, because the rewrite re-enters the mock's
    /// store body (`store_recovered_values`), which takes the same mutex.
    async fn try_recover_via_checkpoint(
        &self,
        hash_hex: &str,
        hash: &[u8; 32],
        original_error: ServiceError,
        checkpoint_store: Option<&Arc<dyn CheckpointStore>>,
        recovery_counter: Option<&std::sync::atomic::AtomicUsize>,
    ) -> Result<Vec<u8>, ServiceError> {
        recover_ext_value_via_checkpoint(
            self,
            checkpoint_store.cloned(),
            &self.recovery_singleflight,
            self.metrics.as_ref(),
            hash_hex,
            hash,
            original_error,
            recovery_counter,
        )
        .await
    }

    /// Inspect the current ref_count for a hash. Returns `None` if not stored.
    pub fn ref_count(&self, hash: &[u8; 32]) -> Option<i64> {
        let state = self.inner.lock().unwrap();
        state.store.get(hash).map(|(_, rc)| *rc)
    }

    /// Test-only inspector: report the number of in-flight per-hash
    /// singleflight entries on the mock's recovery guard. The no-miss
    /// happy-path test (`t3.fgj.zc.p6.rc`) asserts this stays at `0` so
    /// the hot path provably never touches the singleflight map.
    #[cfg(test)]
    pub fn recovery_singleflight_inflight_len(&self) -> usize {
        self.recovery_singleflight.inflight_len()
    }

    /// Check whether a hash has at least one owed decrement in the pending
    /// release set. Hash-scoped rather than member-scoped (b.qpf) so the
    /// pre-existing contract — "is this blob queued for release?" — is
    /// unchanged for callers that do not care which event owes it.
    pub fn is_pending(&self, hash: &[u8; 32]) -> bool {
        let hash_hex = sha256_to_hex(hash);
        let state = self.inner.lock().unwrap();
        state
            .pending
            .keys()
            .any(|m| pending_member_hash_hex(m) == hash_hex)
    }

    /// Number of distinct hashes currently stored.
    pub fn entry_count(&self) -> usize {
        let state = self.inner.lock().unwrap();
        state.store.len()
    }

    /// Number of owed decrements in the pending release set (b.qpf: members,
    /// so two events owing a decrement on the same hash count as two).
    pub fn pending_count(&self) -> usize {
        let state = self.inner.lock().unwrap();
        state.pending.len()
    }

    /// Number of hashes currently in the in-flight GC set.
    ///
    /// b.kf1 medium: gated alongside [`Self::simulate_gc_crash_after_smove`]
    /// under `cfg(any(test, feature = "test-fault-injection"))` so the two
    /// halves of the crash-simulation API are visible together. Pre-b.kf1
    /// `inflight_count` was unconditionally `pub` while
    /// `simulate_gc_crash_after_smove` was `cfg(test)`-gated, an asymmetry
    /// that left release builds carrying an inspector for state they could
    /// never put the mock into.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn inflight_count(&self) -> usize {
        let state = self.inner.lock().unwrap();
        state.inflight.len()
    }

    /// Test hook: simulate a crash *after* a hash has been moved to the
    /// in-flight set but *before* the GC has decremented or deleted. Leaves
    /// the hash in `inflight` and its ref_count/data untouched — exactly the
    /// state produced by a real crash between the `gc_claim_pending.lua`
    /// EVALSHA and the EVAL'd `HINCRBY -1` or `DEL` calls in
    /// [`ValkeyExternalValueStore::process_pending`].
    ///
    /// The claim is stamped with `now_unix_millis() - LEADER_LOCK_TTL_MS - 1`
    /// so the simulated crash is unambiguously older than the re-claim age
    /// — the next `process_pending` re-claims it in place. To
    /// simulate a fresh, still-active claim instead, use
    /// [`Self::simulate_gc_inflight_fresh_claim`] (b.j9j item 2).
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn simulate_gc_crash_after_smove(&self, hash: &[u8; 32]) {
        let stale = now_unix_millis().saturating_sub(LEADER_LOCK_TTL_MS + 1);
        self.move_pending_members_to_inflight(hash, stale);
    }

    /// Test hook: simulate a fresh in-flight claim (b.j9j item 2). The
    /// claim timestamp is `now_unix_millis()` — well inside the
    /// re-claim age — so `process_pending` must NOT re-claim this entry.
    /// Used by the b.j9j item 2 regression tests to pin the "a pass leaves
    /// another replica's active sweep alone" contract.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn simulate_gc_inflight_fresh_claim(&self, hash: &[u8; 32]) {
        self.move_pending_members_to_inflight(hash, now_unix_millis());
    }

    /// Shared body of the two crash-simulation hooks (b.qpf): move every
    /// pending member owing a decrement on `hash` into `inflight` at
    /// `claim_ms`. Hash-scoped rather than member-scoped so both hooks keep
    /// their pre-b.qpf `&[u8; 32]` signatures.
    #[cfg(any(test, feature = "test-fault-injection"))]
    fn move_pending_members_to_inflight(&self, hash: &[u8; 32], claim_ms: u64) {
        let hash_hex = sha256_to_hex(hash);
        let mut state = self.inner.lock().unwrap();
        let members: Vec<String> = state
            .pending
            .keys()
            .filter(|m| pending_member_hash_hex(m) == hash_hex)
            .cloned()
            .collect();
        for member in members {
            state.pending.remove(&member);
            state.inflight.insert(member, claim_ms);
        }
    }

    /// Test hook (b.yuo): simulate a sweeper that applied the decrement of
    /// every pending member owing `hash` and then crashed before completing —
    /// the decrement is applied (and recorded, as the real decrement script
    /// records it) and the members are left in `inflight`, with a claim older
    /// than the re-claim age so the next `process_pending` re-claims them.
    /// It must complete them without decrementing again.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn simulate_gc_crash_after_decrement(&self, hash: &[u8; 32]) {
        let stale = now_unix_millis().saturating_sub(LEADER_LOCK_TTL_MS + 1);
        self.move_pending_members_to_inflight(hash, stale);
        let hash_hex = sha256_to_hex(hash);
        let mut state = self.inner.lock().unwrap();
        let members: Vec<String> = state
            .inflight
            .keys()
            .filter(|m| pending_member_hash_hex(m) == hash_hex)
            .filter(|m| !state.applied.contains_key(*m))
            .cloned()
            .collect();
        for member in members {
            if let Some(entry) = state.store.get_mut(hash) {
                entry.1 -= 1;
                if entry.1 <= 0 {
                    state.store.remove(hash);
                }
            }
            state.applied.insert(member, None);
        }
    }

    /// The body of the mock's `store_values` and `store_recovered_values`
    /// (`verified` false) and `store_verified_values` (`verified` true),
    /// modelling the real store's
    /// `ext_store_value.lua` (`SET data NX` + `HINCRBY meta ref_count 1`): an
    /// existing entry keeps its first-stored bytes and only its count is
    /// incremented; a new entry stores `data` with a count of 1. When
    /// `verified`, an existing entry whose bytes differ in LENGTH from `data`
    /// is replaced with `data` (still one increment) — the script's verified
    /// heal. Same-length bytes are kept: the script does no byte compare and
    /// neither does the mock. The script's other heal trigger, a data key that
    /// is not a string, has no counterpart here: the mock holds only bytes.
    ///
    /// The mock deliberately does NOT model the script's third effect, the
    /// `last_store_ms` stamp. Nothing reads it through this trait: the
    /// reference-count reconciler (`storage::ext_reconcile`) is
    /// `ValkeyClient`-only — it walks the real keyspace with `SCAN` / `HMGET`
    /// and never goes through `ExternalValueStore` at all — so a stamp here
    /// would be state no test could observe and no production reader could
    /// reach. Coverage for the stamp lives against a real Valkey, not against
    /// this mock.
    fn store_values_with(&self, values: &[(&[u8; 32], &[u8])], verified: bool) {
        let mut state = self.inner.lock().unwrap();
        for &(hash, data) in values {
            state
                .store
                .entry(*hash)
                .and_modify(|(stored, rc)| {
                    if verified && stored.len() != data.len() {
                        *stored = data.to_vec();
                    }
                    *rc += 1;
                })
                .or_insert_with(|| (data.to_vec(), 1));
        }
    }
}

impl Default for MockExternalValueStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ExternalValueStore for MockExternalValueStore {
    async fn store_values(&self, values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
        self.store_values_with(values, false);
        Ok(())
    }

    async fn store_verified_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), ServiceError> {
        self.store_values_with(values, true);
        Ok(())
    }

    async fn load_values(
        &self,
        hashes: &[&[u8; 32]],
    ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
        let state = self.inner.lock().unwrap();
        let mut results = Vec::with_capacity(hashes.len());
        for hash in hashes {
            if let Some((data, _)) = state.store.get(*hash) {
                results.push((**hash, data.clone()));
            }
        }
        Ok(results)
    }

    async fn read_batch(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        // Forward to the counter-aware path with `None` so the mock
        // mirrors the production `ValkeyExternalValueStore` shape — the
        // recovery counter is purely additive and existing test callers
        // see byte-identical behaviour.
        self.read_batch_with_recovery_counter(hashes, None).await
    }

    async fn read_batch_with_recovery_counter(
        &self,
        hashes: &[[u8; 32]],
        recovery_counter: Option<&std::sync::atomic::AtomicUsize>,
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        if hashes.is_empty() {
            return Ok(std::collections::HashMap::new());
        }

        // Gather hits + misses under one lock, then drop the guard
        // BEFORE consulting the SR-3.2 recovery adapter — the recovery
        // path calls back into `store_recovered_values` which re-acquires the
        // same mutex, so holding it across the async fallback would
        // deadlock the test.
        let (mut result, misses, checkpoint_store) = {
            let state = self.inner.lock().unwrap();
            let mut hits = std::collections::HashMap::with_capacity(hashes.len());
            let mut misses: Vec<[u8; 32]> = Vec::new();
            for hash in hashes {
                if let Some((data, _)) = state.store.get(hash) {
                    hits.insert(*hash, data.clone());
                } else {
                    misses.push(*hash);
                }
            }
            (hits, misses, state.checkpoint_store.clone())
        };

        for hash in misses {
            let hash_hex = sha256_to_hex(&hash);
            // In the mock, missing from store means meta not found.
            let original = ServiceError::from(ExternalValueError::MetaNotFound {
                hash_hex: hash_hex.clone(),
            });
            let recovered = self
                .try_recover_via_checkpoint(
                    &hash_hex,
                    &hash,
                    original,
                    checkpoint_store.as_ref(),
                    recovery_counter,
                )
                .await?;
            result.insert(hash, recovered);
        }

        Ok(result)
    }

    async fn probe_values(&self, hashes: &[[u8; 32]]) -> Result<ExtValueProbe, ServiceError> {
        // The production probe's contract on the in-memory map: present bytes
        // are returned, a missing hash is listed, and the recovery adapter is
        // never consulted.
        let state = self.inner.lock().unwrap();
        let mut probe = ExtValueProbe::default();
        for hash in hashes {
            match state.store.get(hash) {
                Some((data, _)) => {
                    probe.found.insert(*hash, data.clone());
                }
                None => probe.missing.push(*hash),
            }
        }
        Ok(probe)
    }

    async fn enqueue_retirement_attempt(
        &self,
        _object_id: &str,
        attempt: &str,
        releases: &[ExtRelease],
        send_by: std::time::Instant,
    ) -> Result<ReleaseAttemptOutcome, ServiceError> {
        // The send window (F3), as the real enqueue checks it: past `send_by`
        // nothing is sent.
        if std::time::Instant::now() >= send_by {
            return Ok(ReleaseAttemptOutcome::NotSent);
        }
        let mut state = self.inner.lock().unwrap();
        if !state.landed_attempts.insert(attempt.to_owned()) {
            return Ok(ReleaseAttemptOutcome::AlreadyLanded);
        }
        Self::enqueue_members(&mut state, releases);
        Ok(ReleaseAttemptOutcome::Enqueued)
    }

    async fn process_pending(
        &self,
        policy: &GcPolicy,
        _pass_state: &mut GcPassState,
    ) -> Result<GcBatchResult, ServiceError> {
        let mut state = self.inner.lock().unwrap();
        // Mirror the real pass on wall-clock time: tear down the replay
        // records settled at least U ago, select the members at least
        // `min_age` old (oldest first, up to a batch) and the inflight claims
        // older than `reclaim_after` (re-claimed in place, never moved back to
        // pending), then claim the selected members, apply each one's
        // decrement at most once, and complete it — settling it unless it is
        // back in pending. The mock keeps no byte ledger, no read hold and no
        // older build's state (no un-aged members, no foreign claims), so it
        // never collects under pressure, never clips, never stamps and never
        // drops. In the mock the
        // "crash window" is zero unless the test uses
        // [`simulate_gc_crash_after_smove`] /
        // [`simulate_gc_crash_after_decrement`] /
        // [`simulate_gc_inflight_fresh_claim`].
        let now = now_unix_millis();
        let retention = duration_ms(policy.replay_record_retention);
        let before = state.applied.len();
        state
            .applied
            .retain(|_, settled| settled.is_none_or(|at| at.saturating_add(retention) > now));
        let torn_down = before - state.applied.len();

        let floor = duration_ms(policy.min_age);
        let mut due: Vec<(String, u64)> = state
            .pending
            .iter()
            .filter(|(_, enqueued)| enqueued.saturating_add(floor) <= now)
            .map(|(m, enqueued)| (m.clone(), *enqueued))
            .collect();
        due.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        due.truncate(GC_BATCH_SIZE);
        let mut batch: Vec<String> = due.into_iter().map(|(m, _)| m).collect();

        // The in-place re-claim (b.1jt): a claim older than `reclaim_after`
        // is claimed again by this pass and decided below with the rest,
        // as `gc_reclaim_stale.lua` re-scores it; it never goes back to
        // pending. A re-claimed member whose decrement already applied is
        // answered as a replay.
        let reclaim_cutoff = now.saturating_sub(duration_ms(policy.reclaim_after));
        let mut stale: Vec<(String, u64)> = state
            .inflight
            .iter()
            .filter(|(_, claim_ms)| **claim_ms < reclaim_cutoff)
            .map(|(m, claim_ms)| (m.clone(), *claim_ms))
            .collect();
        stale.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        stale.truncate(GC_STALE_CLAIM_BATCH_SIZE);
        let reclaimed = stale.len();

        for member in &batch {
            state.pending.remove(member);
            state.inflight.insert(member.clone(), now);
        }
        for (member, _) in stale {
            state.inflight.insert(member.clone(), now);
            batch.push(member);
        }

        let mut result = GcBatchResult {
            processed: batch.len(),
            torn_down,
            reclaimed,
            ..GcBatchResult::default()
        };

        for member in &batch {
            if state.applied.contains_key(member) {
                result.replayed += 1;
            } else {
                // Only the `hash_hex` prefix names the value the decrement
                // acts on (b.qpf); the whole member is the claim identity.
                if let Some(hash) = hash_from_hex(pending_member_hash_hex(member)) {
                    if let Some(entry) = state.store.get_mut(&hash) {
                        entry.1 -= 1;
                        if entry.1 <= 0 {
                            state.store.remove(&hash);
                            result.deleted += 1;
                        }
                    }
                }
                state.applied.insert(member.clone(), None);
            }
            state.inflight.remove(member);
            if !state.pending.contains_key(member) {
                state.applied.insert(member.clone(), Some(now));
            }
        }

        Ok(result)
    }

    fn ext_release_gate(&self) -> Option<&Arc<ExtReleaseGate>> {
        // The mock keeps no release ledger, so there is no budget to wait at.
        None
    }

    fn ext_value_verify(&self) -> ExtValueVerify<'_> {
        ExtValueVerify {
            mode: self.ext_value_verify,
            metrics: self.metrics.as_ref(),
        }
    }
}

// ─── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{Field, TimeSample};
    use std::sync::Arc;
    use std::time::Duration;

    /// Helper: create a distinct 32-byte hash from a single byte value.
    fn test_hash(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    /// Helper (b.qpf): wrap a bare hash as one owed decrement for a fixture
    /// object's retirement. The tests below predate the event-carrying member
    /// and only care that *one* decrement is owed, which
    /// [`ExtRelease::retirement`] on a fixed `object_id` expresses exactly.
    fn rel(hash: [u8; 32]) -> ExtRelease {
        ExtRelease::retirement(hash, 1)
    }

    // ── the pending-member wire format ────────────────────────────────────
    //
    // Stable wire text, written by this process and read back by the GC scripts
    // and by `storage::ext_reconcile`'s owed walk. The three event spellings are
    // asserted as LITERALS rather than round-tripped through `token()`, because
    // a round-trip agrees with itself under any rename: the point is that a
    // member written by yesterday's replica still parses today.

    /// `ExtRelease::member` spells all three events, and every spelling parses
    /// back to its hash.
    ///
    /// The parse is the load-bearing half. `pending_member_hash_hex` takes the
    /// prefix up to the FIRST `:`, which works only because a `hash_hex` is 64
    /// hex characters and can never contain one — so every spelling the protocol
    /// has ever written parses with the same one-line rule, including the legacy
    /// bare `{hash_hex}` a pre-b.qpf replica may still have sitting in `pending`
    /// at upgrade time. If that ever stopped holding, a member would be claimed
    /// and then decrement the WRONG key, or none.
    #[test]
    fn every_pending_member_spelling_parses_back_to_its_hash() {
        let hash = test_hash(0x5A);
        let hex = sha256_to_hex(&hash);

        for (member, expected, what) in [
            (
                ExtRelease::commit(hash, 42, 7).member(),
                format!("{hex}:42:7"),
                "a commit's event token is its decimal sequence",
            ),
            (
                ExtRelease::retirement(hash, 42).member(),
                format!("{hex}:42:ret"),
                "a retirement's token is `ret` — NOT a decimal, so it can never collide with a \
                 commit sequence",
            ),
            (
                ExtRelease::reconcile(hash, 3).member(),
                format!("{hex}:0:rec3"),
                "a reconciler repair spells `{hash_hex}:0:rec{n}` (b.yda): the `0` is the \
                 no-object sentinel and `rec3` collides with neither a decimal nor `ret`",
            ),
        ] {
            assert_eq!(member, expected, "{what}");
            assert_eq!(
                pending_member_hash_hex(&member),
                hex,
                "and it must parse back to the hash it names, or the GC claims it and then \
                 decrements the wrong key: {member}"
            );
        }

        assert_eq!(
            pending_member_hash_hex(&hex),
            hex,
            "the legacy bare `{{hash_hex}}` member — written by a pre-b.qpf replica, or already \
             sitting in `pending` at upgrade time — must still parse to itself. That is the \
             whole of the no-migration guarantee"
        );
    }

    /// A reconciler repair names NO object, and that sentinel is `0`.
    ///
    /// `ref_count` is per live object, so every other release is attributable to
    /// one. A repair is not: it hands back an increment whose owning object is
    /// unknown — it crashed before enqueuing, it predates the `obj:ext` index,
    /// or it was a restore's suppressed surplus. `0` is safe as the sentinel
    /// because object ids are minted by `INCR le:counter:object_id`, which
    /// returns 1 first and never re-issues, so no real object can spell one.
    #[test]
    fn a_reconciler_repair_carries_the_no_object_sentinel() {
        let hash = test_hash(0x5B);
        let release = ExtRelease::reconcile(hash, 1);

        assert_eq!(release.object_id, RECONCILE_NO_OBJECT_SENTINEL);
        assert_eq!(
            RECONCILE_NO_OBJECT_SENTINEL, 0,
            "the sentinel is `0` precisely because `INCR` mints ids from 1, so it is \
             unreachable by any real object"
        );
        assert_eq!(release.event, ReleaseEvent::Reconcile(1));
        assert_eq!(release.hash, hash);
    }

    /// The repair's AMOUNT rides in the member, and it is what makes a replay
    /// idempotent without making two genuinely different repairs collapse.
    ///
    /// The release channel is a Set. Emitting `n` members that all spelled the
    /// same thing would collapse to ONE and hand back one reference against a
    /// surplus of `n`; emitting an index instead of the amount would make a
    /// replayed repair of the SAME surplus produce fresh members and
    /// double-decrement. The amount counting DOWN from the claimed surplus gives
    /// both: `n` distinct members within one repair, and byte-identical members
    /// across a replay of it.
    ///
    /// # This test is NOT the authority on what the reconciler emits
    ///
    /// The `emit` closure below reproduces `act_on_candidate`'s arithmetic rather
    /// than calling it — `act_on_candidate` is private to `ext_reconcile`, needs a
    /// claimed candidate and a live keyspace, and reaching it from here would be a
    /// second copy of that module's fixture. So this arm agrees with itself under
    /// any change made to BOTH: what it actually pins is the property the
    /// spelling has to have (distinct within a repair, identical across a replay,
    /// distinct across amounts) and the exact `{hash}:0:rec{n}` wire text, which
    /// `pending_member_hash_hex` and all three GC scripts have to keep parsing.
    ///
    /// The authority on what the reconciler really emits is
    /// `ext_reconcile::tests::a_standing_surplus_is_repaired_in_capped_batches`,
    /// which drives the real pass and reads the members off `le:ext:{gc}:pending`
    /// at a surplus of 3 and of 70 — the second above
    /// `EXT_RECON_MAX_REPAIR_PER_HASH`, where the batching this closure knows
    /// nothing about takes over. Keep the two in agreement: a change to the
    /// spelling that updated only this file would pass here and fail there, which
    /// is the correct direction.
    #[test]
    fn a_repair_of_n_references_spells_n_distinct_members_that_replay_identically() {
        let hash = test_hash(0x5C);
        let hex = sha256_to_hex(&hash);

        let emit = |surplus: u32| -> Vec<String> {
            (0..surplus)
                .map(|i| ExtRelease::reconcile(hash, surplus - i).member())
                .collect()
        };

        let first = emit(3);
        assert_eq!(
            first,
            vec![
                format!("{hex}:0:rec3"),
                format!("{hex}:0:rec2"),
                format!("{hex}:0:rec1"),
            ],
            "three owed decrements must be three DISTINCT members — a Set collapses equal ones, \
             so one repeated spelling would hand back one reference against a surplus of three"
        );
        assert_eq!(
            emit(3),
            first,
            "and a replay of the same repair must produce byte-identical members, so the Set \
             collapses it to a no-op rather than owing a second round of decrements"
        );

        let smaller = emit(2);
        assert!(
            !smaller.contains(&format!("{hex}:0:rec3")),
            "while a repair at a DIFFERENT amount stays distinct at its top member, so a genuine \
             second repair of the same hash is not silently swallowed: {smaller:?}"
        );
    }

    // ── store_values ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn store_new_hash_creates_entry_with_refcount_one() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xAA);

        mock.store_values(&[(&hash, b"hello")])
            .await
            .expect("store must succeed");

        assert_eq!(mock.ref_count(&hash), Some(1));
        assert_eq!(mock.entry_count(), 1);
    }

    #[tokio::test]
    async fn store_same_hash_twice_deduplicates_and_increments_refcount() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xBB);

        mock.store_values(&[(&hash, b"data")]).await.unwrap();
        mock.store_values(&[(&hash, b"data")]).await.unwrap();

        assert_eq!(mock.ref_count(&hash), Some(2));
        assert_eq!(mock.entry_count(), 1, "dedup: still one entry");
    }

    #[tokio::test]
    async fn store_same_hash_twice_preserves_data_bytes() {
        // Content-addressed: the data key is derived from the hash, so
        // repeated stores of the same hash must return the same bytes.
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xCC);

        mock.store_values(&[(&hash, b"data")]).await.unwrap();
        mock.store_values(&[(&hash, b"data")]).await.unwrap();

        let results = mock.load_values(&[&hash]).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, b"data");
        assert_eq!(mock.entry_count(), 1, "dedup: still one stored entry");
    }

    #[tokio::test]
    async fn store_two_different_hashes_stores_both_independently() {
        let mock = MockExternalValueStore::new();
        let h1 = test_hash(0x01);
        let h2 = test_hash(0x02);

        mock.store_values(&[(&h1, b"a"), (&h2, b"b")])
            .await
            .unwrap();

        assert_eq!(mock.entry_count(), 2);
        assert_eq!(mock.ref_count(&h1), Some(1));
        assert_eq!(mock.ref_count(&h2), Some(1));

        let mut loaded: std::collections::HashMap<[u8; 32], Vec<u8>> = mock
            .load_values(&[&h1, &h2])
            .await
            .unwrap()
            .into_iter()
            .collect();
        assert_eq!(loaded.remove(&h1).unwrap(), b"a");
        assert_eq!(loaded.remove(&h2).unwrap(), b"b");
    }

    // ── load_values ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn load_returns_stored_data() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xDD);
        let data = b"payload bytes";

        mock.store_values(&[(&hash, data)]).await.unwrap();

        let results = mock.load_values(&[&hash]).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, hash);
        assert_eq!(results[0].1, data);
    }

    #[tokio::test]
    async fn load_unknown_hash_returns_empty() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xFF);

        let results = mock.load_values(&[&hash]).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn load_batch_returns_all_stored_and_skips_missing() {
        let mock = MockExternalValueStore::new();
        let h1 = test_hash(0x01);
        let h2 = test_hash(0x02);
        let h_missing = test_hash(0xFF);

        mock.store_values(&[(&h1, b"data1"), (&h2, b"data2")])
            .await
            .unwrap();

        let results = mock.load_values(&[&h1, &h_missing, &h2]).await.unwrap();
        assert_eq!(results.len(), 2);

        let result_map: std::collections::HashMap<[u8; 32], Vec<u8>> =
            results.into_iter().collect();
        assert_eq!(result_map[&h1], b"data1");
        assert_eq!(result_map[&h2], b"data2");
    }

    #[tokio::test]
    async fn load_empty_input_returns_empty() {
        let mock = MockExternalValueStore::new();
        let results = mock.load_values(&[]).await.unwrap();
        assert!(results.is_empty());
    }

    // ── schedule_release ─────────────────────────────────────────────────

    #[tokio::test]
    async fn schedule_release_adds_hashes_to_pending() {
        let mock = MockExternalValueStore::new();
        let h1 = test_hash(0x01);
        let h2 = test_hash(0x02);

        mock.schedule_release(&[rel(h1), rel(h2)]).await.unwrap();

        assert!(mock.is_pending(&h1));
        assert!(mock.is_pending(&h2));
        assert_eq!(mock.pending_count(), 2);
    }

    #[tokio::test]
    async fn schedule_release_is_idempotent() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0x01);

        mock.schedule_release(&[rel(hash)]).await.unwrap();
        mock.schedule_release(&[rel(hash)]).await.unwrap();

        assert_eq!(mock.pending_count(), 1, "set semantics — no duplicates");
    }

    #[tokio::test]
    async fn schedule_release_empty_input_is_noop() {
        let mock = MockExternalValueStore::new();
        mock.schedule_release(&[]).await.unwrap();
        assert_eq!(mock.pending_count(), 0);
    }

    // ── process_pending ──────────────────────────────────────────────────

    #[tokio::test]
    async fn process_pending_deletes_when_refcount_reaches_zero() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xEE);

        mock.store_values(&[(&hash, b"ephemeral")]).await.unwrap();
        assert_eq!(mock.ref_count(&hash), Some(1));

        mock.schedule_release(&[rel(hash)]).await.unwrap();
        let batch = mock
            .process_pending(&GcPolicy::IMMEDIATE, &mut GcPassState::new())
            .await
            .unwrap();

        assert_eq!(batch.processed, 1);
        assert_eq!(batch.deleted, 1);
        assert_eq!(mock.entry_count(), 0, "entry deleted");
        assert!(!mock.is_pending(&hash), "removed from pending");

        // Data should be gone — load returns empty.
        let results = mock.load_values(&[&hash]).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn process_pending_decrements_but_keeps_when_refcount_above_zero() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xEE);

        // Store twice → ref_count = 2.
        mock.store_values(&[(&hash, b"shared")]).await.unwrap();
        mock.store_values(&[(&hash, b"shared")]).await.unwrap();
        assert_eq!(mock.ref_count(&hash), Some(2));

        mock.schedule_release(&[rel(hash)]).await.unwrap();
        let batch = mock
            .process_pending(&GcPolicy::IMMEDIATE, &mut GcPassState::new())
            .await
            .unwrap();

        assert_eq!(batch.processed, 1);
        assert_eq!(batch.deleted, 0);
        assert_eq!(mock.ref_count(&hash), Some(1));
        assert_eq!(mock.entry_count(), 1, "entry still present");

        // Data should still be readable.
        let results = mock.load_values(&[&hash]).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, b"shared");
    }

    #[tokio::test]
    async fn process_pending_on_empty_set_returns_zero() {
        let mock = MockExternalValueStore::new();
        let batch = mock
            .process_pending(&GcPolicy::IMMEDIATE, &mut GcPassState::new())
            .await
            .unwrap();
        assert_eq!(batch.processed, 0);
    }

    #[tokio::test]
    async fn process_pending_handles_already_deleted_metadata() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xFF);

        // IMMEDIATE, except that replay records outlive the run (b.yuo).
        let policy = GcPolicy {
            replay_record_retention: GC_REPLAY_RECORD_RETENTION,
            ..GcPolicy::IMMEDIATE
        };

        // Add to pending without storing — simulates metadata already deleted.
        mock.schedule_release(&[rel(hash)]).await.unwrap();
        let batch = mock
            .process_pending(&policy, &mut GcPassState::new())
            .await
            .unwrap();

        assert_eq!(batch.processed, 1);
        assert!(!mock.is_pending(&hash), "removed from pending");

        // b.yuo: the decrement was recorded even though there was nothing to
        // decrement, so a replay of the same member is answered as one and
        // never reaches the store.
        mock.store_values(&[(&hash, b"stored-after")])
            .await
            .unwrap();
        mock.schedule_release(&[rel(hash)]).await.unwrap();
        let replay = mock
            .process_pending(&policy, &mut GcPassState::new())
            .await
            .unwrap();
        assert_eq!((replay.processed, replay.replayed), (1, 1));
        assert_eq!(mock.ref_count(&hash), Some(1));
    }

    #[tokio::test]
    async fn process_pending_clears_pending_set() {
        let mock = MockExternalValueStore::new();
        let h1 = test_hash(0x01);
        let h2 = test_hash(0x02);

        mock.store_values(&[(&h1, b"a"), (&h2, b"b")])
            .await
            .unwrap();
        mock.schedule_release(&[rel(h1), rel(h2)]).await.unwrap();
        assert_eq!(mock.pending_count(), 2);

        mock.process_pending(&GcPolicy::IMMEDIATE, &mut GcPassState::new())
            .await
            .unwrap();
        assert_eq!(mock.pending_count(), 0, "pending set empty after GC");
    }

    // ── crash-recovery (b.i8x Bug 2) ─────────────────────────────────────

    #[tokio::test]
    async fn gc_end_to_end_after_crash_between_smove_and_decrement() {
        // b.i8x Bug 2 regression: a crash after the claim but before the
        // decrement must not leak. The next pass re-claims the stale claim in
        // place and collects it (b.1jt; see
        // `a_pass_reclaims_a_stale_claim_in_place_and_collects_it_in_the_same_pass`).
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xB2);

        mock.store_values(&[(&hash, b"data")]).await.unwrap();
        mock.schedule_release(&[rel(hash)]).await.unwrap();
        mock.simulate_gc_crash_after_smove(&hash);

        let batch = mock
            .process_pending(&GcPolicy::IMMEDIATE, &mut GcPassState::new())
            .await
            .unwrap();
        assert_eq!(batch.processed, 1);
        assert_eq!(batch.deleted, 1);
        assert_eq!(mock.entry_count(), 0);
        assert_eq!(mock.inflight_count(), 0);
        assert_eq!(mock.pending_count(), 0);
    }

    // ── b.yuo: the GC policy, the per-pass re-claim, at most one decrement ──

    /// `GcPolicy::PRODUCTION` is pinned at its exact values (b.yuo part D):
    /// `min_age` 5 s, `pressure_min_age` 0, `U` 120 s, `D` 30 s, hold clipping
    /// on. Floors this low are safe only because every catch-up build reads
    /// its blobs under the read-hold miss protocol, re-reading (atomically, or
    /// through the object's fenced-read flight) instead of failing on a blob
    /// collected under its snapshot — and that protocol's protection is the
    /// hold clip, so the clip is pinned with the floors. A change to any value
    /// is a deliberate decision this test makes visible.
    ///
    /// The timing relations the at-most-once decrement rests on are pinned
    /// too: the decrement fence `F` below the re-claim age `D`, and `U − F`, the
    /// clock skew between a hash's primary and the `{gc}` primary the replay
    /// record tolerates, at least 105 s.
    #[test]
    fn production_gc_policy_is_5_s_with_no_pressure_floor() {
        let p = GcPolicy::PRODUCTION;
        assert_eq!(
            (
                p.min_age,
                p.pressure_min_age,
                p.replay_record_retention,
                p.reclaim_after
            ),
            (
                Duration::from_secs(5),
                Duration::ZERO,
                Duration::from_secs(120),
                Duration::from_secs(30)
            ),
            "PRODUCTION is (min_age, pressure_min_age, U, D) = (5 s, 0, 120 s, 30 s)"
        );
        assert!(
            p.hold_clip,
            "PRODUCTION must clip its selection below a live read hold: the catch-up miss \
             protocol's re-read relies on it"
        );
        let fence = Duration::from_millis(GC_DECREMENT_FENCE_MS);
        assert!(
            fence < p.reclaim_after,
            "the decrement fence F must stay below the re-claim age D, so a member is never \
             re-claimed while a copy of its old claim can still act"
        );
        assert!(
            p.replay_record_retention.saturating_sub(fence) >= Duration::from_secs(105),
            "U − F must stay at least 105 s, so a record is never torn down while a copy of the \
             claim it records can still act on a primary up to that far behind the {{gc}} one"
        );
    }

    /// U1 (b.pa1): `GcPolicy::OOM_COLLECTION` collects at any age — both age
    /// floors 0 — and keeps everything else `PRODUCTION`'s: `U`, `D`, and
    /// hold clipping on. Collecting at any age is safe for readers only
    /// through the read-hold miss protocol, whose protection is the hold
    /// clip, so the clip is pinned with the floors; `U` and `D` are what the
    /// at-most-once decrement rests on, so they may not differ from the
    /// loop's. The collection's own timings are pinned too: the upload hold
    /// outlives the longest collection, and a collection that freed nothing
    /// backs off longer than one that freed.
    #[test]
    fn oom_collection_gc_policy_collects_at_any_age_and_honours_read_holds() {
        let p = GcPolicy::OOM_COLLECTION;
        assert_eq!(
            (p.min_age, p.pressure_min_age),
            (Duration::ZERO, Duration::ZERO),
            "an out-of-memory collection collects every released member at any age"
        );
        assert!(
            p.hold_clip,
            "OOM_COLLECTION must clip its selection below a live read hold: collecting at any \
             age is safe for a catch-up only through the hold"
        );
        assert_eq!(
            (p.replay_record_retention, p.reclaim_after),
            (
                GcPolicy::PRODUCTION.replay_record_retention,
                GcPolicy::PRODUCTION.reclaim_after
            ),
            "OOM_COLLECTION keeps PRODUCTION's U and D, which the at-most-once decrement rests on"
        );
        assert!(
            OOM_UPLOAD_HOLD_TTL > OOM_COLLECTION_MAX_DURATION,
            "the upload hold must stand for the whole collection"
        );
        assert!(
            OOM_COLLECTION_RETRY_AFTER_FREED < OOM_COLLECTION_RETRY_AFTER_FUTILE,
            "a collection that freed nothing must back off longer than one that freed"
        );
    }

    /// The out-of-memory collection's rate limit (b.pa1): after a collection
    /// that may be followed soon (it freed, or ended `failed`) the next may
    /// start [`OOM_COLLECTION_RETRY_AFTER_FREED`] after it ended, and not
    /// before; after any other, only [`OOM_COLLECTION_RETRY_AFTER_FUTILE`]
    /// after it. Before the first collection one may start at once.
    #[test]
    fn oom_collection_may_start_a_second_after_a_soon_one_and_30_s_after_any_other() {
        const JUST_UNDER: Duration = Duration::from_millis(1);
        let ended = tokio::time::Instant::now();
        assert!(
            OomCollectionState::default().may_start(ended),
            "the first collection of a leadership may start at once"
        );

        let soon = OomCollectionState {
            last: Some((ended, true)),
            ..OomCollectionState::default()
        };
        assert!(!soon.may_start(ended + OOM_COLLECTION_RETRY_AFTER_FREED - JUST_UNDER));
        assert!(
            soon.may_start(ended + OOM_COLLECTION_RETRY_AFTER_FREED),
            "a second after a collection that freed or failed, the next may start"
        );

        let futile = OomCollectionState {
            last: Some((ended, false)),
            ..OomCollectionState::default()
        };
        assert!(
            !futile.may_start(ended + OOM_COLLECTION_RETRY_AFTER_FREED),
            "a second after a collection that found nothing to free, the next is turned away"
        );
        assert!(!futile.may_start(ended + OOM_COLLECTION_RETRY_AFTER_FUTILE - JUST_UNDER));
        assert!(futile.may_start(ended + OOM_COLLECTION_RETRY_AFTER_FUTILE));
    }

    /// Plant an out-of-memory collection request, as a replica that saw
    /// `-OOM` does: `gc_oom_signal.lua` on the request key, labelled `by`.
    async fn request_oom_collection(tv: &crate::storage::test_support::TestValkey, by: &str) {
        let _: i64 = GC_OOM_SIGNAL_SCRIPT
            .evalsha_with_reload(
                tv.client().pool.next(),
                vec![tv.key_schema().ext_oom_request()],
                vec![by.to_owned(), "10000".to_owned()],
            )
            .await
            .expect("gc_oom_signal must set the request");
    }

    /// The out-of-memory collection's outcome drives its retry cadence
    /// (b.pa1), against a real Valkey, through the leader's request check:
    ///
    /// 1. A collection that ends `failed` (a pass error: the completion of
    ///    its one member fails, fault-injection site 2) is no evidence
    ///    of futility: a request a second later starts the next collection.
    /// 2. A collection that ends `freed_nothing` (it decrements a value with
    ///    two references and deletes nothing) backs off: a request a second
    ///    later is turned away, and only one 30 s after it ended may start.
    ///
    /// The rate limit reads the tokio clock; the one-second waits are real.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_oom_collection_that_failed_retries_after_a_second_and_one_that_freed_nothing_waits_30_s(
    ) {
        use crate::observability::metrics::test_util::{render, sample_value};

        let tv = crate::storage::test_support::TestValkey::setup().await;
        let keys = tv.key_schema().clone();
        let m = Arc::new(crate::observability::metrics::metrics_scrape_only(
            "b-pa1-oom-cadence",
        ));
        let collections = |outcome: &str| {
            sample_value(
                &render(&m),
                "live_edit_gc_oom_collections_total",
                &[("outcome", outcome)],
            )
            .unwrap_or_else(|| panic!("outcome={outcome} is zero-seeded")) as u64
        };
        let plain = ValkeyExternalValueStore::new(tv.client().clone(), keys.clone())
            .with_metrics(Arc::clone(&m));
        let a_second_later = OOM_COLLECTION_RETRY_AFTER_FREED + Duration::from_millis(100);

        // 1. `failed`.
        let failing_hash = test_hash(0xE1);
        plain
            .store_values(&[(&failing_hash, b"failed-collection".as_slice())])
            .await
            .expect("store_values");
        plain
            .schedule_release(&[rel(failing_hash)])
            .await
            .expect("schedule_release");
        let faults: ProcessPendingFaultInjector =
            Arc::new(std::sync::Mutex::new(std::collections::VecDeque::from([(
                2,
                PendingFault::Fail(fred::error::Error::new(
                    fred::error::ErrorKind::Protocol,
                    "synthetic gc_complete failure",
                )),
            )])));
        let failing = ValkeyExternalValueStore::new(tv.client().clone(), keys.clone())
            .with_metrics(Arc::clone(&m))
            .with_process_pending_fault_injector_for_test(Arc::clone(&faults));
        let mut state = GcPassState::new();
        request_oom_collection(&tv, "replica-failed").await;
        let failed = failing
            .process_pending(&GcPolicy::PRODUCTION, &mut state)
            .await;
        assert!(
            faults.lock().unwrap().is_empty(),
            "precondition: the completion's failure was taken"
        );
        assert!(
            failed.is_err(),
            "precondition: the collection ends in the pass's error; got {failed:?}"
        );
        assert_eq!(collections("failed"), 1, "precondition: counted `failed`");
        let (failed_ended, _) = state
            .oom
            .last
            .expect("the failed collection recorded its end");

        tokio::time::sleep(a_second_later).await;
        request_oom_collection(&tv, "replica-failed-again").await;
        failing
            .process_pending(&GcPolicy::PRODUCTION, &mut state)
            .await
            .expect("the next collection must succeed");
        let (next_ended, _) = state.oom.last.expect("a collection ended");
        assert!(
            next_ended > failed_ended,
            "a request a second after a `failed` collection must start the next collection"
        );
        let counted: u64 = crate::observability::metrics::GC_OOM_COLLECTION_OUTCOMES
            .iter()
            .map(|outcome| collections(outcome))
            .sum();
        assert_eq!(counted, 2, "the second collection is counted");

        // 2. `freed_nothing`.
        let shared_hash = test_hash(0xE2);
        for _ in 0..2 {
            plain
                .store_values(&[(&shared_hash, b"shared-value".as_slice())])
                .await
                .expect("store_values");
        }
        plain
            .schedule_release(&[rel(shared_hash)])
            .await
            .expect("schedule_release");
        let mut state = GcPassState::new();
        request_oom_collection(&tv, "replica-futile").await;
        let futile = plain
            .process_pending(&GcPolicy::PRODUCTION, &mut state)
            .await
            .expect("the collection must succeed");
        assert_eq!(
            (futile.processed, futile.deleted, futile.freed_bytes),
            (1, 0, 0),
            "precondition: the collection decremented the value and freed nothing"
        );
        assert_eq!(
            collections("freed_nothing"),
            1,
            "precondition: counted `freed_nothing`"
        );
        let (futile_ended, _) = state
            .oom
            .last
            .expect("the futile collection recorded its end");

        tokio::time::sleep(a_second_later).await;
        request_oom_collection(&tv, "replica-futile-again").await;
        plain
            .process_pending(&GcPolicy::PRODUCTION, &mut state)
            .await
            .expect("the ordinary pass must succeed");
        assert_eq!(
            state.oom.last.map(|(ended, _)| ended),
            Some(futile_ended),
            "a request a second after a `freed_nothing` collection must be turned away"
        );
        assert_eq!(
            collections("freed_nothing"),
            1,
            "no collection ran to count"
        );
        assert!(
            !state.oom.may_start(
                futile_ended + OOM_COLLECTION_RETRY_AFTER_FUTILE - Duration::from_millis(1)
            ),
            "the next waits the full 30 s"
        );
        assert!(state
            .oom
            .may_start(futile_ended + OOM_COLLECTION_RETRY_AFTER_FUTILE));
    }

    /// A store whose `process_pending` only counts its calls: the GC loop's
    /// passes, with nothing behind them (every other method is unused).
    struct PassCountingStore {
        passes: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl ExternalValueStore for PassCountingStore {
        async fn store_values(&self, _values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
            unimplemented!("the GC loop arms only run passes")
        }

        async fn store_verified_values(
            &self,
            _values: &[(&[u8; 32], &[u8])],
        ) -> Result<(), ServiceError> {
            unimplemented!("the GC loop arms only run passes")
        }

        fn ext_value_verify(&self) -> ExtValueVerify<'_> {
            ExtValueVerify::ENFORCE_UNCOUNTED
        }

        async fn load_values(
            &self,
            _hashes: &[&[u8; 32]],
        ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
            unimplemented!("the GC loop arms only run passes")
        }

        async fn read_batch(
            &self,
            _hashes: &[[u8; 32]],
        ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
            unimplemented!("the GC loop arms only run passes")
        }

        async fn probe_values(&self, _hashes: &[[u8; 32]]) -> Result<ExtValueProbe, ServiceError> {
            unimplemented!("the GC loop arms only run passes")
        }

        async fn enqueue_retirement_attempt(
            &self,
            _object_id: &str,
            _attempt: &str,
            _releases: &[ExtRelease],
            _send_by: std::time::Instant,
        ) -> Result<ReleaseAttemptOutcome, ServiceError> {
            unimplemented!("the GC loop arms only run passes")
        }

        async fn process_pending(
            &self,
            _policy: &GcPolicy,
            _state: &mut GcPassState,
        ) -> Result<GcBatchResult, ServiceError> {
            self.passes
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(GcBatchResult::default())
        }

        fn ext_release_gate(&self) -> Option<&Arc<ExtReleaseGate>> {
            None
        }
    }

    /// The GC loop's out-of-memory wake-up arm (b.pa1): after a pass that
    /// found nothing to do the loop sleeps `GC_PASS_INTERVAL`, and a note on
    /// the store out-of-memory signal it was given starts the next pass at
    /// once — with the tokio clock paused and never advanced. Without a note
    /// the loop stays asleep. The test task only yields, so the paused clock
    /// does not auto-advance under it.
    #[tokio::test(start_paused = true)]
    async fn an_out_of_memory_note_starts_the_next_gc_pass_without_waiting_out_the_interval() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        async fn yield_until(passes: &AtomicUsize, at_least: usize) -> usize {
            for _ in 0..1_000 {
                if passes.load(Ordering::SeqCst) >= at_least {
                    break;
                }
                tokio::task::yield_now().await;
            }
            passes.load(Ordering::SeqCst)
        }

        let passes = Arc::new(AtomicUsize::new(0));
        let store: Arc<dyn ExternalValueStore> = Arc::new(PassCountingStore {
            passes: Arc::clone(&passes),
        });
        let signal = crate::storage::store_oom::StoreOomSignal::new(
            fred::types::Builder::default_centralized()
                .build_pool(1)
                .expect("an unconnected pool builds"),
        );
        let cancel = tokio_util::sync::CancellationToken::new();
        let gc_loop = spawn_gc_loop(
            store,
            GcPolicy::IMMEDIATE,
            cancel.clone(),
            None,
            || std::future::ready(true),
            GcLoopExtras {
                progress: None,
                store_oom: Some(signal.clone()),
                leader: None,
            },
        );
        let started = tokio::time::Instant::now();

        assert_eq!(
            yield_until(&passes, 1).await,
            1,
            "the loop runs its first pass"
        );
        assert_eq!(
            yield_until(&passes, 2).await,
            1,
            "with no note the loop sleeps out GC_PASS_INTERVAL"
        );
        signal.note();
        assert_eq!(
            yield_until(&passes, 2).await,
            2,
            "a note wakes the loop into its next pass"
        );
        assert_eq!(
            tokio::time::Instant::now(),
            started,
            "the pass ran with no time passing: the wake-up, not the interval, started it"
        );

        cancel.cancel();
        gc_loop.await.expect("the GC loop must not panic");
    }

    /// The claim-score offset `gc_claim_pending.lua` adds as a literal equals
    /// [`GC_CLAIM_SCORE_OFFSET_MS`], the value every reader of the score is
    /// handed (b.1jt). A mismatch reads this build's own claims as an older
    /// build's — dropped, never decremented: a leak — or an older build's as
    /// this build's — re-claimed and decremented a second time. The encoded
    /// score must also stay an exact double for any claim time this service
    /// will see, so age order survives the encoding.
    #[test]
    fn the_claim_scripts_offset_literal_equals_the_rust_constant() {
        const SCRIPT: &str = include_str!("scripts/gc_claim_pending.lua");
        let literal: Vec<u64> = SCRIPT
            .lines()
            .filter(|line| !line.trim_start().starts_with("--"))
            .filter_map(|line| {
                let rest = line.trim().strip_prefix("local CLAIM_SCORE_OFFSET_MS =")?;
                Some(
                    rest.trim()
                        .parse()
                        .unwrap_or_else(|e| panic!("the offset literal {rest:?} must parse: {e}")),
                )
            })
            .collect();
        assert_eq!(
            literal,
            vec![GC_CLAIM_SCORE_OFFSET_MS],
            "gc_claim_pending.lua must define `local CLAIM_SCORE_OFFSET_MS` once, equal to \
             GC_CLAIM_SCORE_OFFSET_MS"
        );
        assert!(
            SCRIPT.contains("now_ms + CLAIM_SCORE_OFFSET_MS"),
            "the claim's ZADD must score with the offset added to the claim time"
        );
        // A bare claim time (an older build's score) stays below the offset
        // for at least a century from now, and an encoded one — below twice
        // the offset while that holds — is an exact integer double.
        const CENTURY_MS: u64 = 100 * 365 * 24 * 3_600 * 1_000;
        assert!(
            now_unix_millis() + CENTURY_MS < GC_CLAIM_SCORE_OFFSET_MS,
            "a bare claim time must sort below the offset"
        );
        const { assert!(2 * GC_CLAIM_SCORE_OFFSET_MS < 1u64 << 53) };
    }

    /// A pass re-claims an `inflight` claim older than the policy's re-claim
    /// age IN PLACE and collects it in the same pass — the mock's twin of the
    /// real loop's per-pass re-claim (b.1jt). The member never goes back to
    /// pending, where a record-blind GC (0.1.0 through 0.2.0-rc.47) could
    /// decrement it a second time. A fresh claim (a peer's live sweep) is left
    /// alone.
    #[tokio::test]
    async fn a_pass_reclaims_a_stale_claim_in_place_and_collects_it_in_the_same_pass() {
        let mock = MockExternalValueStore::new();
        let stale = test_hash(0xD1);
        let fresh = test_hash(0xD2);
        mock.store_values(&[(&stale, b"stale-claim"), (&fresh, b"fresh-claim")])
            .await
            .unwrap();
        mock.schedule_release(&[rel(stale), rel(fresh)])
            .await
            .unwrap();
        mock.simulate_gc_crash_after_smove(&stale);
        mock.simulate_gc_inflight_fresh_claim(&fresh);
        assert_eq!((mock.inflight_count(), mock.pending_count()), (2, 0));

        let mut state = GcPassState::new();
        let pass = mock
            .process_pending(&GcPolicy::IMMEDIATE, &mut state)
            .await
            .unwrap();
        assert_eq!(
            (pass.reclaimed, pass.processed, pass.deleted),
            (1, 1, 1),
            "the stale claim is re-claimed and collected by the pass that found it"
        );
        assert!(
            !mock.is_pending(&stale),
            "a re-claimed member never goes back to pending"
        );
        assert_eq!(
            (mock.inflight_count(), mock.pending_count()),
            (1, 0),
            "only the fresh claim is left, in inflight"
        );
        assert_eq!(
            mock.ref_count(&stale),
            None,
            "the stale claim was collected"
        );
        assert_eq!(
            mock.ref_count(&fresh),
            Some(1),
            "the fresh claim's decrement belongs to its own sweeper"
        );
    }

    /// A member whose decrement applied before its sweeper crashed (b.yuo
    /// H11) is re-claimed by a later pass and completed WITHOUT a second
    /// decrement: a shared blob stays at the one reference its other holder
    /// owns.
    #[tokio::test]
    async fn a_member_decremented_before_a_crash_is_not_decremented_again() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xD3);
        mock.store_values(&[(&hash, b"shared")]).await.unwrap();
        mock.store_values(&[(&hash, b"shared")]).await.unwrap();
        mock.schedule_release(&[rel(hash)]).await.unwrap();
        mock.simulate_gc_crash_after_decrement(&hash);
        assert_eq!(
            (mock.ref_count(&hash), mock.inflight_count()),
            (Some(1), 1),
            "precondition: the crashed sweeper's decrement applied and the member is stuck in \
             inflight"
        );

        let stats = process_pending_gc(&mock, None, GcPolicy::IMMEDIATE).await;
        assert_eq!(
            (stats.processed, stats.replayed, stats.deleted),
            (1, 1, 0),
            "the re-claimed member is decided once and answered as a replay"
        );
        assert_eq!(
            mock.ref_count(&hash),
            Some(1),
            "a second decrement here deletes a blob its other holder still references"
        );
        assert_eq!((mock.inflight_count(), mock.pending_count()), (0, 0));
    }

    /// Under a non-zero age floor nothing younger is collected (b.yuo test
    /// 1, the mock's twin of `gc_select.lua`'s cutoff).
    #[tokio::test]
    async fn a_member_younger_than_the_age_floor_is_not_collected() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xD4);
        mock.store_values(&[(&hash, b"young")]).await.unwrap();
        mock.schedule_release(&[rel(hash)]).await.unwrap();
        let policy = GcPolicy {
            min_age: Duration::from_secs(3600),
            ..GcPolicy::IMMEDIATE
        };
        let pass = mock
            .process_pending(&policy, &mut GcPassState::new())
            .await
            .unwrap();
        assert_eq!(pass.processed, 0);
        assert_eq!(mock.ref_count(&hash), Some(1));
        assert!(mock.is_pending(&hash));
    }

    // ── data integrity ───────────────────────────────────────────────────

    #[tokio::test]
    async fn stored_bytes_match_retrieved_bytes() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xAB);
        let data: Vec<u8> = (0..=255).collect();

        mock.store_values(&[(&hash, &data)]).await.unwrap();

        let results = mock.load_values(&[&hash]).await.unwrap();
        assert_eq!(results[0].1, data);
    }

    // ── store_external_values pre-flight ─────────────────────────────────

    /// Build an OwnedDelta with one external SetField.
    fn delta_with_ext_field(hash_bytes: &[u8; 32], ext_index: u32) -> oodf_ffi::OwnedDelta {
        oodf_ffi::OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "tex".to_string(),
                value_or_ext_hash: hash_bytes.to_vec(),
                ext_value_size: 1024,
                ext_value_index: ext_index,
                set_order: 0,
            }],
            ..Default::default()
        }
    }

    /// Build an OwnedDelta with an inline-only SetField (no external values).
    fn delta_inline_only() -> oodf_ffi::OwnedDelta {
        oodf_ffi::OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "color".to_string(),
                value_or_ext_hash: vec![0xFFu8, 0x00, 0x80],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn store_external_values_single_ext_value_stored() {
        let mock = MockExternalValueStore::new();
        let ext_data = b"external-blob";
        // A claim and a size the content check admits: the part's content hash
        // and its length.
        let hash = ext_content_hash(ext_data);
        let mut delta = delta_with_ext_field(&hash, 1);
        delta.set_fields[0].ext_value_size = ext_data.len() as u64;

        store_external_values(Some(&delta), &[ext_data.as_slice()], &mock)
            .await
            .unwrap();

        assert_eq!(mock.ref_count(&hash), Some(1));
        assert_eq!(mock.entry_count(), 1);
    }

    #[tokio::test]
    async fn store_external_values_dedup_same_hash_one_store_call() {
        // Two SetField entries with the same hash → only one store call (dedup).
        let mock = MockExternalValueStore::new();
        let ext_data = b"shared-blob";
        let hash = ext_content_hash(ext_data);

        let delta = oodf_ffi::OwnedDelta {
            is_diff: false,
            set_fields: vec![
                oodf_ffi::SetField {
                    section_id: 1,
                    key_name: "tex1".to_string(),
                    value_or_ext_hash: hash.to_vec(),
                    ext_value_size: ext_data.len() as u64,
                    ext_value_index: 1,
                    set_order: 0,
                },
                oodf_ffi::SetField {
                    section_id: 2,
                    key_name: "tex2".to_string(),
                    value_or_ext_hash: hash.to_vec(),
                    ext_value_size: ext_data.len() as u64,
                    ext_value_index: 1,
                    set_order: 0,
                },
            ],
            ..Default::default()
        };

        store_external_values(Some(&delta), &[ext_data.as_slice()], &mock)
            .await
            .unwrap();

        // Dedup: same hash → ref_count=1 (only one store call).
        assert_eq!(mock.ref_count(&hash), Some(1));
        assert_eq!(mock.entry_count(), 1);
    }

    #[tokio::test]
    async fn store_external_values_inline_only_no_calls() {
        let mock = MockExternalValueStore::new();
        let delta = delta_inline_only();

        store_external_values(Some(&delta), &[], &mock)
            .await
            .unwrap();

        assert_eq!(
            mock.entry_count(),
            0,
            "no store calls for inline-only delta"
        );
    }

    #[tokio::test]
    async fn store_external_values_empty_delta_no_calls() {
        let mock = MockExternalValueStore::new();

        store_external_values(None, &[], &mock).await.unwrap();

        assert_eq!(mock.entry_count(), 0);
    }

    #[tokio::test]
    async fn store_external_values_out_of_range_index_returns_error() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xCC);
        // ext_value_index=5 but only 1 external part
        let delta = delta_with_ext_field(&hash, 5);

        let result = store_external_values(Some(&delta), &[b"only-one"], &mock).await;

        assert!(result.is_err(), "out-of-range index must return error");
        assert_eq!(mock.entry_count(), 0, "no store calls on error");
    }

    #[tokio::test]
    async fn store_external_values_zero_index_returns_error() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xDD);
        // ext_value_index=0 is invalid for external values
        let delta = delta_with_ext_field(&hash, 0);

        let result = store_external_values(Some(&delta), &[b"data"], &mock).await;

        assert!(result.is_err(), "zero ext_value_index must return error");
    }

    // ── collect_external_values + the store_external_values wrapper ──────
    //
    // `collect_external_values` is the pure collection step
    // [`store_external_values`] runs first (the content check, further down, is
    // its other pure step). Two of its guarantees are contract rather than incidental,
    // because `checkpoint::compose` folds its output into a lineage-wide list
    // whose *equality* it asserts: first-appearance ORDER (`set_fields`, then
    // `time_samples`; no `HashMap` iteration on the output path) and dedup
    // spanning BOTH maps. The six `store_external_values_*` arms above fence
    // neither — they assert ref-counts and `is_err()` — which is exactly the
    // gap these arms close.

    /// A `SetField` carrying an external reference at `ext_index`.
    fn ext_set_field(key: &str, hash: &[u8; 32], ext_index: u32) -> oodf_ffi::SetField {
        oodf_ffi::SetField {
            section_id: 1,
            key_name: key.to_string(),
            value_or_ext_hash: hash.to_vec(),
            ext_value_size: 512,
            ext_value_index: ext_index,
            set_order: 0,
        }
    }

    /// A `TimeSample` carrying an external reference at `ext_index`.
    fn ext_time_sample(time: f64, hash: &[u8; 32], ext_index: u32) -> oodf_ffi::TimeSample {
        oodf_ffi::TimeSample {
            section_id: 1,
            time,
            value_or_ext_hash: hash.to_vec(),
            ext_value_size: 512,
            ext_value_index: ext_index,
        }
    }

    #[test]
    fn collect_external_values_orders_by_first_appearance_across_both_maps() {
        // Four distinct hashes, two per map. Their byte values are deliberately
        // NOT in the expected output order, so neither a sorted collection nor
        // `HashMap` iteration order can match by accident.
        let h_f1 = test_hash(0x33);
        let h_f2 = test_hash(0x11);
        let h_t1 = test_hash(0x44);
        let h_t2 = test_hash(0x22);

        // Distinct lengths so a part mix-up cannot pass a length-blind compare.
        let parts: [&[u8]; 4] = [b"f1", b"f2-", b"t1--", b"t2---"];

        let delta = oodf_ffi::OwnedDelta {
            is_diff: false,
            set_fields: vec![ext_set_field("a", &h_f1, 1), ext_set_field("b", &h_f2, 2)],
            time_samples: vec![
                ext_time_sample(0.5, &h_t1, 3),
                ext_time_sample(1.5, &h_t2, 4),
            ],
            ..Default::default()
        };

        let collected = collect_external_values(&delta, &parts).expect("collector must succeed");

        assert_eq!(
            collected,
            vec![
                (h_f1, parts[0]),
                (h_f2, parts[1]),
                (h_t1, parts[2]),
                (h_t2, parts[3]),
            ],
            "output must be first-appearance order over set_fields, THEN time_samples — \
             `checkpoint::compose` asserts equality of the list this feeds"
        );
    }

    #[test]
    fn collect_external_values_dedups_a_hash_shared_by_a_field_and_a_time_sample() {
        // The cross-map case: the dedup set spans both loops, so one hash
        // referenced from a field AND from a time sample yields ONE pair. A
        // second, field-only hash brackets it so a "stop after the first map"
        // bug cannot pass this arm.
        let shared = test_hash(0x77);
        let field_only = test_hash(0x88);
        // Part 3 is the same blob the field resolved, addressed by a different
        // index: whichever pair survives names which appearance won.
        let parts: [&[u8]; 3] = [b"shared-via-field", b"field-only", b"shared-via-sample"];

        let delta = oodf_ffi::OwnedDelta {
            is_diff: false,
            set_fields: vec![
                ext_set_field("shared", &shared, 1),
                ext_set_field("solo", &field_only, 2),
            ],
            time_samples: vec![ext_time_sample(0.5, &shared, 3)],
            ..Default::default()
        };

        let collected = collect_external_values(&delta, &parts).expect("collector must succeed");

        assert_eq!(
            collected,
            vec![(shared, parts[0]), (field_only, parts[1])],
            "a hash referenced from both maps must yield exactly one pair, carrying the bytes \
             resolved at its FIRST appearance"
        );
    }

    #[test]
    fn collect_external_values_zero_ext_index_returns_invalid_argument() {
        let delta = delta_with_ext_field(&test_hash(0x01), 0);

        let err = collect_external_values(&delta, &[b"data".as_slice()])
            .expect_err("a zero ext_value_index must be rejected");

        match err {
            ServiceError::InvalidArgument(msg) => assert_eq!(
                msg, "ext_value_index must be non-zero for external values",
                "the message is the operator's only signal for which knob was wrong"
            ),
            other => panic!("expected ServiceError::InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn collect_external_values_out_of_range_ext_index_returns_invalid_argument() {
        let delta = delta_with_ext_field(&test_hash(0x02), 5);

        let err = collect_external_values(&delta, &[b"only-one".as_slice()])
            .expect_err("an out-of-range ext_value_index must be rejected");

        match err {
            ServiceError::InvalidArgument(msg) => assert_eq!(
                msg, "ext_value_index 5 out of range (OMPF has 1 external parts)",
                "the message must name BOTH the offending index and the part count available"
            ),
            other => panic!("expected ServiceError::InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn collect_external_values_rejects_a_bad_index_behind_an_already_seen_hash() {
        // Dedup is a membership test, not a short-circuit: index resolution
        // runs for EVERY referencing entry, so a bad index hiding behind an
        // already-collected hash is still rejected. The obvious "skip the
        // duplicate before resolving" optimization would let it through
        // silently, and no other arm here would notice.
        let shared = test_hash(0x99);
        let delta = oodf_ffi::OwnedDelta {
            is_diff: false,
            set_fields: vec![
                ext_set_field("good", &shared, 1),
                ext_set_field("bad", &shared, 9),
            ],
            ..Default::default()
        };

        let err = collect_external_values(&delta, &[b"only-one".as_slice()])
            .expect_err("a duplicate hash must not mask an out-of-range index");

        match err {
            ServiceError::InvalidArgument(msg) => assert_eq!(
                msg, "ext_value_index 9 out of range (OMPF has 1 external parts)",
                "the rejection must name the duplicate entry's own index"
            ),
            other => panic!("expected ServiceError::InvalidArgument, got {other:?}"),
        }
    }

    /// One recorded `store_values` or `store_verified_values` call: the
    /// `(hash, data)` pairs it carried, owned. Spelled as an alias to stay
    /// inside clippy's `type_complexity` budget — the same reason
    /// [`ExtValueRef`] is an alias.
    type RecordedStoreCall = Vec<([u8; 32], Vec<u8>)>;

    /// Records every `store_values` call (and every `store_verified_values`
    /// call, in the same log) verbatim, so the wrapper's delegation
    /// can be asserted exactly: which pairs, in which order, in how many calls.
    ///
    /// [`MockExternalValueStore`] cannot carry these arms: it folds calls into
    /// a ref-counted map, which makes "one call carrying zero pairs"
    /// indistinguishable from "no call at all" — precisely the distinction the
    /// wrapper's two no-op short-circuits are about.
    #[derive(Default)]
    struct RecordingExtStore {
        calls: std::sync::Mutex<Vec<RecordedStoreCall>>,
    }

    #[async_trait]
    impl ExternalValueStore for RecordingExtStore {
        async fn store_values(&self, values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
            self.calls
                .lock()
                .expect("recording mutex must not be poisoned")
                .push(values.iter().map(|(h, d)| (**h, d.to_vec())).collect());
            Ok(())
        }

        // Recorded in the same log as `store_values`: the arms assert which
        // pairs reached the store, not which of the two store calls carried them.
        async fn store_verified_values(
            &self,
            values: &[(&[u8; 32], &[u8])],
        ) -> Result<(), ServiceError> {
            self.store_values(values).await
        }

        fn ext_value_verify(&self) -> ExtValueVerify<'_> {
            ExtValueVerify::ENFORCE_UNCOUNTED
        }

        async fn load_values(
            &self,
            _hashes: &[&[u8; 32]],
        ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
            unimplemented!("the wrapper-equivalence arms only exercise the store calls")
        }

        async fn read_batch(
            &self,
            _hashes: &[[u8; 32]],
        ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
            unimplemented!("the wrapper-equivalence arms only exercise the store calls")
        }

        async fn probe_values(&self, _hashes: &[[u8; 32]]) -> Result<ExtValueProbe, ServiceError> {
            unimplemented!("the wrapper-equivalence arms only exercise the store calls")
        }

        async fn enqueue_retirement_attempt(
            &self,
            _object_id: &str,
            _attempt: &str,
            _releases: &[ExtRelease],
            _send_by: std::time::Instant,
        ) -> Result<ReleaseAttemptOutcome, ServiceError> {
            unimplemented!("the wrapper-equivalence arms only exercise the store calls")
        }

        async fn process_pending(
            &self,
            _policy: &GcPolicy,
            _state: &mut GcPassState,
        ) -> Result<GcBatchResult, ServiceError> {
            unimplemented!("the wrapper-equivalence arms only exercise the store calls")
        }

        fn ext_release_gate(&self) -> Option<&Arc<ExtReleaseGate>> {
            None
        }
    }

    impl RecordingExtStore {
        fn calls(&self) -> Vec<RecordedStoreCall> {
            self.calls
                .lock()
                .expect("recording mutex must not be poisoned")
                .clone()
        }
    }

    #[tokio::test]
    async fn store_external_values_hands_the_store_exactly_the_collectors_pairs() {
        // Wrapper equivalence: the wrapper gates (the content check) and then
        // delegates. For an upload the check admits, the pairs that reach the
        // store must be the collector's output — same set, same order, one
        // call. Part 1 is named by two references (a field and a time sample),
        // so a part shared by several references is admitted, and stored once.
        let parts: [&[u8]; 3] = [b"one", b"two-", b"three"];

        let delta = oodf_ffi::OwnedDelta {
            is_diff: false,
            set_fields: vec![
                checked_set_field("a", parts[0], 1),
                checked_set_field("b", parts[1], 2),
            ],
            time_samples: vec![
                checked_time_sample(0.5, parts[2], 3),
                checked_time_sample(1.5, parts[0], 1),
            ],
            ..Default::default()
        };

        let expected: RecordedStoreCall = collect_external_values(&delta, &parts)
            .expect("collector must succeed")
            .into_iter()
            .map(|(hash, data)| (hash, data.to_vec()))
            .collect();

        let store = RecordingExtStore::default();
        store_external_values(Some(&delta), &parts, &store)
            .await
            .expect("wrapper must succeed");

        let calls = store.calls();
        assert_eq!(
            calls.len(),
            1,
            "the wrapper must make exactly one store call"
        );
        assert_eq!(
            calls[0], expected,
            "the wrapper must hand the store the collector's pairs verbatim, in order"
        );
    }

    #[tokio::test]
    async fn store_external_values_none_delta_never_touches_the_store() {
        // Non-empty parts on purpose: the `None` short-circuit must win before
        // `ompf_external_values` is read at all.
        let store = RecordingExtStore::default();

        store_external_values(None, &[b"unused".as_slice()], &store)
            .await
            .expect("a None delta must be a no-op, not an error");

        assert!(
            store.calls().is_empty(),
            "a None delta must not reach the store at all"
        );
    }

    #[tokio::test]
    async fn store_external_values_empty_collector_result_never_touches_the_store() {
        // An inline-only delta WITH parts present: the collector returns empty,
        // and the wrapper must not issue a zero-pair call on the back of it.
        let store = RecordingExtStore::default();
        let delta = delta_inline_only();
        assert!(
            collect_external_values(&delta, &[b"unused".as_slice()])
                .expect("collector must succeed")
                .is_empty(),
            "the fixture must collect nothing, or the arm proves nothing"
        );

        store_external_values(Some(&delta), &[b"unused".as_slice()], &store)
            .await
            .expect("an empty collection must be a no-op, not an error");

        assert!(
            store.calls().is_empty(),
            "an empty collector result must not reach the store"
        );
    }

    // ── G3: wrong-length external hashes are refused, never padded ───────
    //
    // The collector used to zero-pad a short `value_or_ext_hash` and truncate
    // a long one into a 32-byte hash the client never sent, so the gRPC
    // handlers stored a blob under that invented hash before validation
    // refused the delta. It now refuses with the same `InvalidArgument` that
    // `validate_delta_bounds` produces, and never panics:
    // `checkpoint::compose` runs it over stored stream records that may
    // predate the ingest-time check.

    /// Hash lengths the collector must refuse: empty, one short, one long, and
    /// the 64-byte hex-string length.
    const WRONG_HASH_LENS: [usize; 4] = [0, 31, 33, 64];

    /// The client-facing message for a `len`-byte external hash — the
    /// `DeltaError::InvalidExternalHashLength` text through `From<DeltaError>`.
    fn wrong_hash_len_message(len: usize) -> String {
        format!("delta error: invalid external-value hash length: got {len} bytes, expected 32")
    }

    fn assert_wrong_hash_len_refused(
        result: Result<Vec<ExtValueRef<'_>>, ServiceError>,
        len: usize,
    ) {
        match result {
            Err(ServiceError::InvalidArgument(msg)) => assert_eq!(
                msg,
                wrong_hash_len_message(len),
                "a {len}-byte external hash must be refused with the hash-length message, not \
                 an index error"
            ),
            Ok(pairs) => panic!(
                "a {len}-byte external hash must be refused, not padded or truncated into \
                 {pairs:?}"
            ),
            Err(other) => panic!("expected ServiceError::InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn collect_external_values_set_field_wrong_length_hash_returns_invalid_argument() {
        for len in WRONG_HASH_LENS {
            let delta = oodf_ffi::OwnedDelta {
                set_fields: vec![oodf_ffi::SetField {
                    value_or_ext_hash: vec![0xAB; len],
                    ..ext_set_field("tex", &test_hash(0), 1)
                }],
                ..Default::default()
            };
            assert_wrong_hash_len_refused(
                collect_external_values(&delta, &[b"blob".as_slice()]),
                len,
            );
        }
    }

    // ── the upload content check (`verify_external_values`) ──────────────
    //
    // `store_external_values` runs the content check between the collector and
    // the store: every reference's raw claim must be 32 bytes, its part must be
    // `ext_value_size` long, and the part's `ext_content_hash` must equal the
    // claim. These arms pin the hash against the clients' own vectors, the
    // verify-all-then-store order, the per-reference walk, both
    // `LIVE_EDIT_EXT_VALUE_VERIFY` modes, the `live_edit_ext_value_verify_total`
    // counter and the `observe` WARN.

    /// `len` bytes of `i % 251`: the input both client vector sets hash (the
    /// C++ client library's and the Rust client library's own test suites).
    fn client_vector_input(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    /// `(length, expected content hash of client_vector_input(length), source)`.
    /// The lengths straddle every block edge: empty, one byte, a sub-block
    /// value, one block minus one / exactly / plus one byte, a 31-byte
    /// remainder, and exactly two blocks.
    const CLIENT_CONTENT_HASH_VECTORS: [(usize, &str, &str); 8] = [
        (
            0,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "hashlib",
        ),
        (
            1,
            "1406e05881e299367766d313e26c05564ec91bf721d31726bd6e46e60689539a",
            "C++ client vector",
        ),
        (
            65_536,
            "90df369a7383e1c6da72aa68c8f7fb6ab1ba311fad0f8bae602fee725c9f6596",
            "Rust client vector",
        ),
        (
            1_048_575,
            "64b38a8ab1b18fce95a4fce9994720477a2928c1af722d355c343a700c0cf164",
            "hashlib",
        ),
        (
            1_048_576,
            "004fbf44b4c0a58632164fc9ec00e8f35e330037fd4425dd947cf65d930762c2",
            "C++ client vector",
        ),
        (
            1_048_577,
            "a9c574ce937d2371daf87cdd0e75396b096d7f74a66c764e4c656bd12b2e7bf9",
            "Rust client vector",
        ),
        (
            1_048_607,
            "f5c9e6694fcbe188c96f8a1945622f16b6ef5808bc40e4cd9f1abb085f3d7e2b",
            "C++ client vector",
        ),
        (
            2_097_152,
            "bf4dff263ae686b64403208c35a3767e38453d6477e841947588d13d04e1d816",
            "Rust client vector",
        ),
    ];

    /// Plain SHA-256 of the single byte `0x00`: what a server that hashed
    /// external values the obvious way would compute.
    const PLAIN_SHA256_OF_ONE_ZERO_BYTE: &str =
        "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d";

    /// `ext_content_hash` is the clients' 1 MiB block hash, bit for bit.
    ///
    /// The expected values were computed by the clients' own test suites and by
    /// hashlib, independently of this crate, so the function cannot agree with
    /// itself here. The gate admits exactly what this function computes, so a
    /// drift here refuses every external value real clients send.
    #[test]
    fn ext_content_hash_matches_the_clients_pinned_vectors() {
        assert_eq!(
            EXT_CONTENT_HASH_BLOCK_BYTES, 1_048_576,
            "the block size is a fixed protocol constant the clients hash with"
        );
        for (len, expected, source) in CLIENT_CONTENT_HASH_VECTORS {
            assert_eq!(
                sha256_to_hex(&ext_content_hash(&client_vector_input(len))),
                expected,
                "content hash of {len} bytes of i % 251 must match the vector from {source}"
            );
        }
    }

    #[test]
    fn collect_external_values_time_sample_wrong_length_hash_returns_invalid_argument() {
        for len in WRONG_HASH_LENS {
            let delta = oodf_ffi::OwnedDelta {
                time_samples: vec![oodf_ffi::TimeSample {
                    value_or_ext_hash: vec![0xAB; len],
                    ..ext_time_sample(0.5, &test_hash(0), 1)
                }],
                ..Default::default()
            };
            assert_wrong_hash_len_refused(
                collect_external_values(&delta, &[b"blob".as_slice()]),
                len,
            );
        }
    }

    #[tokio::test]
    async fn store_external_values_wrong_length_hash_never_touches_the_store() {
        // A valid entry FIRST, so a collector that stored as it went (or
        // returned the pairs it had so far) would reach the store with it.
        // The 31-byte hash is the pre-fix zero-pad shape.
        let store = RecordingExtStore::default();
        let delta = oodf_ffi::OwnedDelta {
            set_fields: vec![
                ext_set_field("good", &test_hash(0x42), 1),
                oodf_ffi::SetField {
                    value_or_ext_hash: vec![0xAB; 31],
                    ..ext_set_field("bad", &test_hash(0), 2)
                },
            ],
            ..Default::default()
        };

        let err = store_external_values(
            Some(&delta),
            &[b"good-blob".as_slice(), b"bad-blob".as_slice()],
            &store,
        )
        .await
        .expect_err("a 31-byte external hash must be refused");

        assert!(
            matches!(&err, ServiceError::InvalidArgument(msg) if *msg == wrong_hash_len_message(31)),
            "expected the hash-length InvalidArgument, got {err:?}"
        );
        assert!(
            store.calls().is_empty(),
            "no blob may be stored for a refused delta — not even its valid entry's: {:?}",
            store.calls()
        );
    }

    // ── hash_from_hex ────────────────────────────────────────────────────
    //
    // Strict: exactly 64 ASCII hex digits, or `None`. Two readers depend on
    // that: the GC's pending-member parse, and the snapshot row decoder, which
    // fails a read `Internal` on `None` rather than hand out a hash naming the
    // wrong blob.

    #[test]
    fn hash_from_hex_64_hex_digits_either_case_decodes() {
        let hash: [u8; 32] = std::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xA5);
        let lower = sha256_to_hex(&hash);
        assert_eq!(hash_from_hex(&lower), Some(hash), "lowercase: {lower}");
        let upper = lower.to_ascii_uppercase();
        assert_ne!(
            upper, lower,
            "fixture must contain hex letters, or case is not exercised"
        );
        assert_eq!(hash_from_hex(&upper), Some(hash), "uppercase: {upper}");
    }

    #[test]
    fn hash_from_hex_malformed_input_returns_none() {
        let valid = "ab".repeat(32);
        // A leading `+` passes `u8::from_str_radix` ("+a" -> 0x0a), so a
        // 64-character string with one would decode to a real-looking hash
        // without the digit check.
        let plus = format!("+a{}", "ab".repeat(31));
        // 64 BYTES of non-ASCII: slicing it at byte pairs splits a code point.
        let non_ascii = format!("{}a", "\u{20AC}".repeat(21));
        assert_eq!(plus.len(), 64);
        assert_eq!(non_ascii.len(), 64);

        for (what, input) in [
            ("empty", String::new()),
            ("odd length (63)", valid[..63].to_string()),
            ("one byte short (62)", valid[..62].to_string()),
            ("too long (66)", format!("{valid}ab")),
            ("non-hex digit", format!("zz{}", &valid[2..])),
            ("64 characters with a `+`", plus),
            ("64 bytes of non-ASCII", non_ascii),
        ] {
            assert_eq!(
                hash_from_hex(&input),
                None,
                "{what}: {input:?} must be refused, never decoded to a padded or aliased hash"
            );
        }
    }

    /// The content hash is not plain SHA-256: for one block it is
    /// SHA-256(SHA-256(bytes)).
    #[test]
    fn ext_content_hash_of_one_byte_is_not_its_plain_sha256() {
        use sha2::{Digest, Sha256};
        let plain: [u8; 32] = Sha256::digest([0u8]).into();
        assert_eq!(
            sha256_to_hex(&plain),
            PLAIN_SHA256_OF_ONE_ZERO_BYTE,
            "fixture self-check: the constant is plain SHA-256 of [0x00]"
        );
        assert_ne!(
            sha256_to_hex(&ext_content_hash(&[0u8])),
            PLAIN_SHA256_OF_ONE_ZERO_BYTE,
            "the content hash of [0x00] must not be its plain SHA-256: every client sends \
             SHA-256 over the per-block digests, so a plain-SHA-256 gate refuses all of them"
        );
    }

    /// A `SetField` the content check admits for `part`: the claim is the
    /// part's content hash and the size is the part's length.
    fn checked_set_field(key: &str, part: &[u8], ext_index: u32) -> oodf_ffi::SetField {
        oodf_ffi::SetField {
            section_id: 1,
            key_name: key.to_string(),
            value_or_ext_hash: ext_content_hash(part).to_vec(),
            ext_value_size: part.len() as u64,
            ext_value_index: ext_index,
            set_order: 0,
        }
    }

    /// A `TimeSample` the content check admits for `part`.
    fn checked_time_sample(time: f64, part: &[u8], ext_index: u32) -> oodf_ffi::TimeSample {
        oodf_ffi::TimeSample {
            section_id: 1,
            time,
            value_or_ext_hash: ext_content_hash(part).to_vec(),
            ext_value_size: part.len() as u64,
            ext_value_index: ext_index,
        }
    }

    /// The text of an `InvalidArgument`; any other error panics.
    fn invalid_argument_text(err: ServiceError, context: &str) -> String {
        match err {
            ServiceError::InvalidArgument(msg) => msg,
            other => panic!("{context}: expected ServiceError::InvalidArgument, got {other:?}"),
        }
    }

    /// The three `live_edit_ext_value_verify_total{result}` cells.
    #[derive(Debug, PartialEq, Eq)]
    struct VerifyCells {
        ok: u64,
        size_mismatch: u64,
        hash_mismatch: u64,
    }

    /// Read the counter's cells from the rendered scrape. The label values are
    /// spelled out rather than read from `EXT_VALUE_VERIFY_RESULTS`, so a
    /// renamed label fails here instead of agreeing with itself. Every cell is
    /// seeded by `Metrics::new`, so an absent one panics.
    fn verify_cells(metrics: &crate::observability::metrics::Metrics) -> VerifyCells {
        use crate::observability::metrics::test_util::{render, sample_value};
        let text = render(metrics);
        let cell = |result: &str| {
            sample_value(
                &text,
                "live_edit_ext_value_verify_total",
                &[("result", result)],
            )
            .unwrap_or_else(|| panic!("result={result} must be seeded in Metrics::new"))
                as u64
        };
        VerifyCells {
            ok: cell("ok"),
            size_mismatch: cell("size_mismatch"),
            hash_mismatch: cell("hash_mismatch"),
        }
    }

    /// No checked reference counted at all.
    const NO_CELLS: VerifyCells = VerifyCells {
        ok: 0,
        size_mismatch: 0,
        hash_mismatch: 0,
    };

    /// A [`RecordingExtStore`] that answers `ext_value_verify` with a chosen
    /// mode and its own metrics handle, as a store built with
    /// `with_ext_value_verify(..)` and `with_metrics(..)` does.
    struct PolicyRecordingStore {
        recorder: RecordingExtStore,
        mode: crate::config::ExtValueVerifyMode,
        metrics: SharedMetrics,
    }

    impl PolicyRecordingStore {
        fn new(mode: crate::config::ExtValueVerifyMode) -> Self {
            Self {
                recorder: RecordingExtStore::default(),
                mode,
                metrics: Arc::new(crate::observability::metrics::metrics_scrape_only(
                    "content-check-test",
                )),
            }
        }

        fn calls(&self) -> Vec<RecordedStoreCall> {
            self.recorder.calls()
        }

        fn cells(&self) -> VerifyCells {
            verify_cells(&self.metrics)
        }
    }

    #[async_trait]
    impl ExternalValueStore for PolicyRecordingStore {
        async fn store_values(&self, values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
            self.recorder.store_values(values).await
        }

        async fn store_verified_values(
            &self,
            values: &[(&[u8; 32], &[u8])],
        ) -> Result<(), ServiceError> {
            self.recorder.store_verified_values(values).await
        }

        async fn load_values(
            &self,
            _hashes: &[&[u8; 32]],
        ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
            unimplemented!("the content-check arms only exercise the store calls")
        }

        async fn read_batch(
            &self,
            _hashes: &[[u8; 32]],
        ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
            unimplemented!("the content-check arms only exercise the store calls")
        }

        async fn probe_values(&self, _hashes: &[[u8; 32]]) -> Result<ExtValueProbe, ServiceError> {
            unimplemented!("the content-check arms only exercise the store calls")
        }

        async fn enqueue_retirement_attempt(
            &self,
            _object_id: &str,
            _attempt: &str,
            _releases: &[ExtRelease],
            _send_by: std::time::Instant,
        ) -> Result<ReleaseAttemptOutcome, ServiceError> {
            unimplemented!("the content-check arms only exercise the store calls")
        }

        async fn process_pending(
            &self,
            _policy: &GcPolicy,
            _state: &mut GcPassState,
        ) -> Result<GcBatchResult, ServiceError> {
            unimplemented!("the content-check arms only exercise the store calls")
        }

        fn ext_release_gate(&self) -> Option<&Arc<ExtReleaseGate>> {
            None
        }

        fn ext_value_verify(&self) -> ExtValueVerify<'_> {
            ExtValueVerify {
                mode: self.mode,
                metrics: Some(&self.metrics),
            }
        }
    }

    /// Verify all, then store: a mismatch in the LAST of three distinct hashes
    /// refuses the upload before the first store. A check made per hash while
    /// storing would already have stored the first two.
    #[tokio::test]
    async fn store_external_values_refuses_a_mismatch_in_the_last_of_three_hashes_before_storing_any(
    ) {
        let parts: [&[u8]; 3] = [b"first part", b"second part", b"third part"];
        let mut last = checked_time_sample(0.5, parts[2], 3);
        last.value_or_ext_hash = ext_content_hash(b"not the third part").to_vec();
        let delta = oodf_ffi::OwnedDelta {
            set_fields: vec![
                checked_set_field("a", parts[0], 1),
                checked_set_field("b", parts[1], 2),
            ],
            time_samples: vec![last],
            ..Default::default()
        };
        let store = RecordingExtStore::default();

        let err = store_external_values(Some(&delta), &parts, &store)
            .await
            .expect_err("a part that does not hash to its claim must be refused");

        let msg = invalid_argument_text(err, "hash mismatch on the last hash");
        assert!(
            msg.contains("ext_value_index 3") && msg.contains("does not match its valueOrExtHash"),
            "the refusal must name the failing reference and the rule; got: {msg}"
        );
        assert!(
            store.calls().is_empty(),
            "nothing may be stored when any reference fails: the two good hashes ahead of the bad \
             one must not have been written"
        );
    }

    /// Every reference is checked, not only each hash's first appearance. The
    /// second reference below claims an already-collected hash but names a
    /// DIFFERENT part whose bytes do not hash to it. The collector dedups it
    /// away, so a check over the deduplicated pairs alone passes it, and
    /// UpdateLayer relays the uploaded OMPF verbatim to subscribers.
    #[tokio::test]
    async fn store_external_values_refuses_a_second_reference_naming_a_part_that_does_not_hash_to_its_claim(
    ) {
        let parts: [&[u8]; 2] = [b"the claimed part", b"a different part"];
        let first = checked_set_field("a", parts[0], 1);
        let mut second = checked_set_field("b", parts[1], 2);
        second.value_or_ext_hash = first.value_or_ext_hash.clone();
        let delta = oodf_ffi::OwnedDelta {
            set_fields: vec![first, second],
            ..Default::default()
        };
        assert_eq!(
            collect_external_values(&delta, &parts)
                .expect("collector must succeed")
                .len(),
            1,
            "fixture self-check: the collector dedups the second reference away, so only a \
             per-reference check can see it"
        );
        let store = RecordingExtStore::default();

        let err = store_external_values(Some(&delta), &parts, &store)
            .await
            .expect_err("a reference naming a part that does not hash to its claim is refused");

        let msg = invalid_argument_text(err, "second reference");
        assert!(
            msg.contains("ext_value_index 2") && msg.contains("does not match its valueOrExtHash"),
            "the refusal must name the second reference's own index; got: {msg}"
        );
        assert!(store.calls().is_empty(), "nothing may be stored");
    }

    /// The one part every [`two_references_to_one_part`] delta names.
    const SHARED_PART: &[u8] = b"one part named by two references";

    /// Two references naming the SAME part (`ext_value_index` 1): the first
    /// claims the part's content hash; the second claims it too when
    /// `second_claim_is_correct`, and otherwise claims other bytes' hash.
    ///
    /// The content check memoizes each part's hash by index, so the second
    /// reference reaches an index already checked. Skipping such a reference
    /// would pass a wrong second claim, and because the collector dedups by
    /// claim, not by index, it would store the part under that wrong claim.
    fn two_references_to_one_part(second_claim_is_correct: bool) -> oodf_ffi::OwnedDelta {
        let first = checked_set_field("a", SHARED_PART, 1);
        let mut second = checked_set_field("b", SHARED_PART, 1);
        if !second_claim_is_correct {
            second.value_or_ext_hash = ext_content_hash(b"bytes the part does not hold").to_vec();
        }
        oodf_ffi::OwnedDelta {
            set_fields: vec![first, second],
            ..Default::default()
        }
    }

    /// `enforce`: a second reference to an already-checked index is checked
    /// against its OWN claim, so a wrong one refuses the call and nothing is
    /// stored.
    #[tokio::test]
    async fn store_external_values_refuses_a_wrong_claim_on_a_second_reference_to_an_already_checked_index(
    ) {
        let delta = two_references_to_one_part(false);
        assert_eq!(
            collect_external_values(&delta, &[SHARED_PART])
                .expect("collector must succeed")
                .len(),
            2,
            "fixture self-check: the two claims differ, so the collector keeps both pairs and a \
             skipped check would store the part under the wrong claim"
        );
        let store = PolicyRecordingStore::new(crate::config::ExtValueVerifyMode::Enforce);

        let err = store_external_values(Some(&delta), &[SHARED_PART], &store)
            .await
            .expect_err("the wrong second claim must be refused");

        let msg = invalid_argument_text(err, "second reference to the same index");
        assert!(
            msg.contains("does not match its valueOrExtHash"),
            "the refusal must be the hash check; got: {msg}"
        );
        assert!(store.calls().is_empty(), "nothing may be stored");
    }

    /// `observe`: both references to the one part are checked and counted, one
    /// `ok` and one `hash_mismatch`.
    #[tokio::test]
    async fn observe_mode_counts_each_reference_to_an_already_checked_index() {
        let delta = two_references_to_one_part(false);
        let store = PolicyRecordingStore::new(crate::config::ExtValueVerifyMode::Observe);

        store_external_values(Some(&delta), &[SHARED_PART], &store)
            .await
            .expect("observe stores a hash mismatch");

        assert_eq!(
            store.cells(),
            VerifyCells {
                ok: 1,
                size_mismatch: 0,
                hash_mismatch: 1,
            },
            "a part named twice is counted once per reference"
        );
    }

    /// Two correct references to the one part count `ok` twice: the count is
    /// per reference, not per part or per claim.
    #[tokio::test]
    async fn two_correct_references_to_one_part_count_ok_twice() {
        let delta = two_references_to_one_part(true);
        let store = PolicyRecordingStore::new(crate::config::ExtValueVerifyMode::Enforce);

        store_external_values(Some(&delta), &[SHARED_PART], &store)
            .await
            .expect("two correct references are admitted");

        assert_eq!(
            store.cells(),
            VerifyCells {
                ok: 2,
                size_mismatch: 0,
                hash_mismatch: 0,
            },
            "one count per checked reference"
        );
        assert_eq!(
            store.calls(),
            vec![vec![(ext_content_hash(SHARED_PART), SHARED_PART.to_vec())]],
            "and the one claim is stored once"
        );
    }

    /// The RAW claim is compared, never a padded or truncated form of it. A 33-byte claim whose first 32 bytes ARE the part's content hash
    /// passes a truncating compare; it must be refused, before any store, with
    /// the protocol's 32-byte error. So must a 31-byte claim.
    #[tokio::test]
    async fn store_external_values_refuses_a_33_or_31_byte_claim_before_any_store() {
        let part: &[u8] = b"a part whose claim is the wrong length";
        let hash = ext_content_hash(part);
        let mut long = hash.to_vec();
        long.push(0x00);
        let short = hash[..31].to_vec();

        for (claim, len) in [(long, 33), (short, 31)] {
            let mut field = checked_set_field("a", part, 1);
            field.value_or_ext_hash = claim;
            let delta = oodf_ffi::OwnedDelta {
                set_fields: vec![field],
                ..Default::default()
            };
            let store = RecordingExtStore::default();

            let err = store_external_values(Some(&delta), &[part], &store)
                .await
                .expect_err("a claim that is not 32 bytes must be refused");

            let msg = invalid_argument_text(err, "wrong-length claim");
            assert!(
                msg.contains(&format!("got {len} bytes, expected 32")),
                "a {len}-byte claim must be refused for its length, not compared; got: {msg}"
            );
            assert!(
                store.calls().is_empty(),
                "a {len}-byte claim must be refused before anything is stored"
            );
        }
    }

    /// `ext_value_size` must equal the part's length. A part that matches its
    /// claim but declares one byte more or less is refused before any store.
    #[tokio::test]
    async fn store_external_values_refuses_an_ext_value_size_one_off_the_part_length_before_any_store(
    ) {
        let part: &[u8] = b"a part whose declared size is off by one";
        let len = part.len() as u64;

        for size in [len - 1, len + 1] {
            let mut field = checked_set_field("a", part, 1);
            field.ext_value_size = size;
            let delta = oodf_ffi::OwnedDelta {
                set_fields: vec![field],
                ..Default::default()
            };
            let store = RecordingExtStore::default();

            let err = store_external_values(Some(&delta), &[part], &store)
                .await
                .expect_err("an extValueSize other than the part's length must be refused");

            let msg = invalid_argument_text(err, "size off by one");
            assert!(
                msg.contains(&format!("is {len} bytes, but its extValueSize is {size}")),
                "the refusal must name both lengths; got: {msg}"
            );
            assert!(
                store.calls().is_empty(),
                "extValueSize {size} for a {len}-byte part must be refused before any store"
            );
        }
    }

    /// Mock parity with `ext_store_value.lua`'s `SET … NX`: a second store of a
    /// hash with DIFFERENT bytes keeps the first bytes and still increments.
    #[tokio::test]
    async fn mock_store_keeps_the_first_bytes_when_a_hash_is_stored_again_with_different_bytes() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xC1);

        mock.store_values(&[(&hash, b"the first copy".as_slice())])
            .await
            .unwrap();
        mock.store_values(&[(&hash, b"TRUNC".as_slice())])
            .await
            .unwrap();

        let loaded = mock.load_values(&[&hash]).await.unwrap();
        assert_eq!(
            loaded,
            vec![(hash, b"the first copy".to_vec())],
            "the first copy stored under a hash stays, as the real store's set-if-absent write"
        );
        assert_eq!(
            mock.ref_count(&hash),
            Some(2),
            "and the second store still pays its increment"
        );
    }

    /// Mock parity with `ext_store_value.lua`'s verified heal:
    /// `store_verified_values` replaces an existing entry whose LENGTH differs
    /// from the verified bytes, whether shorter or longer. It keeps one of the
    /// same length, because the script does no byte compare. It pays one
    /// increment on each.
    #[tokio::test]
    async fn mock_store_verified_values_replaces_a_different_length_entry_and_keeps_a_same_length_one(
    ) {
        let mock = MockExternalValueStore::new();
        let truncated_hash = test_hash(0xC2);
        let same_length_hash = test_hash(0xC3);
        let longer_hash = test_hash(0xC4);
        let verified: &[u8] = b"the verified bytes";
        let truncated = &verified[..7];
        // Every byte differs, the length does not.
        let same_length_wrong: &[u8] = b"THE VERIFIED BYTES";
        assert_eq!(
            same_length_wrong.len(),
            verified.len(),
            "staged: same length"
        );
        let longer: &[u8] = b"the verified bytes, and bytes appended past them";

        mock.store_values(&[
            (&truncated_hash, truncated),
            (&same_length_hash, same_length_wrong),
            (&longer_hash, longer),
        ])
        .await
        .unwrap();
        mock.store_verified_values(&[
            (&truncated_hash, verified),
            (&same_length_hash, verified),
            (&longer_hash, verified),
        ])
        .await
        .unwrap();

        for (entry, hash) in [("shorter", truncated_hash), ("longer", longer_hash)] {
            assert_eq!(
                mock.load_values(&[&hash]).await.unwrap(),
                vec![(hash, verified.to_vec())],
                "a verified store replaces an entry whose length differs from the verified \
                 bytes in either direction; the {entry} entry was kept"
            );
        }
        assert_eq!(
            mock.load_values(&[&same_length_hash]).await.unwrap(),
            vec![(same_length_hash, same_length_wrong.to_vec())],
            "and keeps an entry of the same length: like the script, the mock compares lengths, \
             never bytes"
        );
        for hash in [truncated_hash, same_length_hash, longer_hash] {
            assert_eq!(
                mock.ref_count(&hash),
                Some(2),
                "the verified store pays exactly one increment per pair, healed or not"
            );
        }
    }

    /// `store_external_values` stores through `store_verified_values` under
    /// `enforce` and through `store_values` under `observe`: a correct upload
    /// over a truncated entry under its hash replaces the entry in `enforce`
    /// mode and leaves it in `observe` mode. Each mode still pays one increment.
    #[tokio::test]
    async fn store_external_values_heals_a_truncated_entry_under_enforce_and_leaves_it_under_observe(
    ) {
        let good: &[u8] = b"a part that passes its content check";
        let truncated = &good[..good.len() - 5];
        let hash = ext_content_hash(good);
        let delta = oodf_ffi::OwnedDelta {
            set_fields: vec![checked_set_field("a", good, 1)],
            ..Default::default()
        };

        for (mode, expected) in [
            (crate::config::ExtValueVerifyMode::Enforce, good),
            (crate::config::ExtValueVerifyMode::Observe, truncated),
        ] {
            let mock = MockExternalValueStore::new().with_ext_value_verify(mode);
            mock.store_values(&[(&hash, truncated)]).await.unwrap();

            store_external_values(Some(&delta), &[good], &mock)
                .await
                .unwrap_or_else(|e| panic!("{mode:?}: a correct upload must be admitted: {e:?}"));

            assert_eq!(
                mock.load_values(&[&hash]).await.unwrap(),
                vec![(hash, expected.to_vec())],
                "{mode:?}: enforce stores the checked upload as verified (healing the truncated \
                 entry); observe stores it unverified (leaving the entry)"
            );
            assert_eq!(
                mock.ref_count(&hash),
                Some(2),
                "{mode:?}: the upload pays its one increment either way"
            );
        }
    }

    /// `observe`: a size mismatch and a hash mismatch are each counted, nothing
    /// is refused, and exactly one store call carries the collector's pairs,
    /// failing parts included.
    #[tokio::test]
    async fn observe_mode_counts_each_mismatch_and_stores_the_collectors_pairs() {
        let parts: [&[u8]; 3] = [
            b"an admitted part",
            b"a part whose size is misdeclared",
            b"a part whose claim is wrong",
        ];
        let ok = checked_set_field("ok", parts[0], 1);
        // Only the size is wrong: its claim is the part's own content hash.
        let mut size_off = checked_set_field("size", parts[1], 2);
        size_off.ext_value_size += 1;
        let mut hash_off = checked_time_sample(0.5, parts[2], 3);
        hash_off.value_or_ext_hash = ext_content_hash(b"some other bytes").to_vec();
        let delta = oodf_ffi::OwnedDelta {
            set_fields: vec![ok, size_off],
            time_samples: vec![hash_off],
            ..Default::default()
        };
        let expected: RecordedStoreCall = collect_external_values(&delta, &parts)
            .expect("collector must succeed")
            .into_iter()
            .map(|(hash, data)| (hash, data.to_vec()))
            .collect();
        let store = PolicyRecordingStore::new(crate::config::ExtValueVerifyMode::Observe);

        store_external_values(Some(&delta), &parts, &store)
            .await
            .expect("observe must not refuse a size or hash mismatch");

        assert_eq!(
            store.calls(),
            vec![expected],
            "observe stores the upload as sent: one call carrying the collector's pairs, the \
             failing parts included"
        );
        assert_eq!(
            store.cells(),
            VerifyCells {
                ok: 1,
                size_mismatch: 1,
                hash_mismatch: 1,
            },
            "one count per checked reference, by result"
        );
    }

    /// The mock's `with_ext_value_verify` reaches the gate: built `observe`, it
    /// stores a part that fails its claim and counts the mismatch on its
    /// `with_metrics` handle.
    #[tokio::test]
    async fn mock_store_built_observe_stores_a_part_that_fails_its_claim_and_counts_it() {
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only(
            "content-check-mock",
        ));
        let mock = MockExternalValueStore::new()
            .with_ext_value_verify(crate::config::ExtValueVerifyMode::Observe)
            .with_metrics(Arc::clone(&metrics));
        let part: &[u8] = b"bytes that do not hash to the claim";
        let claim = ext_content_hash(b"the bytes the claim names");
        let mut field = checked_set_field("a", part, 1);
        field.value_or_ext_hash = claim.to_vec();
        let delta = oodf_ffi::OwnedDelta {
            set_fields: vec![field],
            ..Default::default()
        };

        store_external_values(Some(&delta), &[part], &mock)
            .await
            .expect("a mock built observe must store a hash mismatch");

        assert_eq!(
            mock.load_values(&[&claim]).await.unwrap(),
            vec![(claim, part.to_vec())],
            "observe stores the bytes as sent under the claim"
        );
        assert_eq!(
            verify_cells(&metrics),
            VerifyCells {
                ok: 0,
                size_mismatch: 0,
                hash_mismatch: 1,
            },
            "and counts the mismatch on the mock's metrics handle"
        );
    }

    /// `observe` does not relax the protocol's 32-byte rule or index
    /// resolution: each of these is refused, stores nothing and counts
    /// nothing.
    #[tokio::test]
    async fn observe_mode_still_refuses_a_wrong_length_claim_or_a_bad_index_and_counts_nothing() {
        let part: &[u8] = b"a part whose reference is malformed";
        let hash = ext_content_hash(part);
        let with = |f: &dyn Fn(&mut oodf_ffi::SetField)| {
            let mut field = checked_set_field("a", part, 1);
            f(&mut field);
            field
        };
        let cases = [
            ("33-byte claim", with(&|f| f.value_or_ext_hash.push(0x00))),
            (
                "31-byte claim",
                with(&|f| f.value_or_ext_hash = hash[..31].to_vec()),
            ),
            ("zero ext_value_index", with(&|f| f.ext_value_index = 0)),
            (
                "out-of-range ext_value_index",
                with(&|f| f.ext_value_index = 2),
            ),
        ];

        for (case, field) in cases {
            let delta = oodf_ffi::OwnedDelta {
                set_fields: vec![field],
                ..Default::default()
            };
            let store = PolicyRecordingStore::new(crate::config::ExtValueVerifyMode::Observe);

            let err = store_external_values(Some(&delta), &[part], &store)
                .await
                .expect_err("observe must still refuse a malformed reference");

            invalid_argument_text(err, case);
            assert!(store.calls().is_empty(), "{case}: nothing may be stored");
            assert_eq!(
                store.cells(),
                NO_CELLS,
                "{case}: a refusal is not a check result and is not counted"
            );
        }
    }

    /// `enforce` counts the references it checked, up to and including the
    /// refused one, and none after it.
    #[tokio::test]
    async fn enforce_mode_counts_up_to_the_refused_reference_and_no_further() {
        let parts: [&[u8]; 3] = [b"first, admitted", b"second, refused", b"third, unchecked"];
        let mut refused = checked_set_field("b", parts[1], 2);
        refused.value_or_ext_hash = ext_content_hash(b"not the second part").to_vec();
        let delta = oodf_ffi::OwnedDelta {
            set_fields: vec![
                checked_set_field("a", parts[0], 1),
                refused,
                checked_set_field("c", parts[2], 3),
            ],
            ..Default::default()
        };
        let store = PolicyRecordingStore::new(crate::config::ExtValueVerifyMode::Enforce);

        store_external_values(Some(&delta), &parts, &store)
            .await
            .expect_err("enforce refuses the hash mismatch");

        assert_eq!(
            store.cells(),
            VerifyCells {
                ok: 1,
                size_mismatch: 0,
                hash_mismatch: 1,
            },
            "the first reference and the refused one are counted; the third is never checked"
        );
        assert!(store.calls().is_empty(), "and nothing is stored");
    }

    /// A size mismatch is counted as such and its part is never hashed, in
    /// either mode. The reference's claim is ALSO wrong, so hashing it after
    /// the size check would add a `hash_mismatch` count. Under `observe` the
    /// walk goes on past it, so that arm is what shows it is skipped rather
    /// than merely returned from.
    #[tokio::test]
    async fn a_size_mismatch_is_counted_as_such_and_its_part_is_never_hashed() {
        let part: &[u8] = b"a part with a wrong size and a wrong claim";
        let mut field = checked_set_field("a", part, 1);
        field.ext_value_size += 1;
        field.value_or_ext_hash = ext_content_hash(b"other bytes").to_vec();
        let delta = oodf_ffi::OwnedDelta {
            set_fields: vec![field],
            ..Default::default()
        };
        let only_size = VerifyCells {
            ok: 0,
            size_mismatch: 1,
            hash_mismatch: 0,
        };

        let enforcing = PolicyRecordingStore::new(crate::config::ExtValueVerifyMode::Enforce);
        store_external_values(Some(&delta), &[part], &enforcing)
            .await
            .expect_err("enforce refuses the size mismatch");
        assert_eq!(enforcing.cells(), only_size, "enforce");

        let observing = PolicyRecordingStore::new(crate::config::ExtValueVerifyMode::Observe);
        store_external_values(Some(&delta), &[part], &observing)
            .await
            .expect("observe stores the size mismatch");
        assert_eq!(observing.cells(), only_size, "observe");
    }

    // ── the `observe` WARN ───────────────────────────────────────────────
    //
    // The WARN is the operator's per-call record of an unverified store, so
    // its fields are the contract (test-writing guide, "when the log line is
    // the deliverable"). Captured as JSON so an absent field is a missing key.

    /// A `MakeWriter` that captures everything written into a shared buffer.
    /// Duplicated from `config.rs`'s tests for the same reason given there: a
    /// shared helper would have to live outside test code.
    #[derive(Clone)]
    struct CapturingWriter(Arc<std::sync::Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturingWriter {
        type Writer = CapturingWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for CapturingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Run `store_external_values` under a JSON WARN subscriber scoped to the
    /// call, returning its result and the `fields` object of every content-check
    /// WARN it logged.
    async fn content_check_warns(
        delta: &oodf_ffi::OwnedDelta,
        parts: &[&[u8]],
        store: &dyn ExternalValueStore,
    ) -> (Result<(), ServiceError>, Vec<serde_json::Value>) {
        use tracing::instrument::WithSubscriber;
        let buf = Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(CapturingWriter(Arc::clone(&buf)))
            .with_max_level(tracing::Level::WARN)
            .finish();
        let result = store_external_values(Some(delta), parts, store)
            .with_subscriber(subscriber)
            .await;
        let text = String::from_utf8(buf.lock().unwrap().clone()).expect("UTF-8 log output");
        let warns = text
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("one JSON event"))
            .filter(|event| {
                event["level"] == "WARN"
                    && event["fields"]["message"]
                        .as_str()
                        .is_some_and(|m| m.contains("upload content check"))
            })
            .map(|event| event["fields"].clone())
            .collect();
        (result, warns)
    }

    /// One `observe` call with several mismatches logs exactly one WARN, which
    /// carries both mismatch counts and describes the FIRST failing reference.
    #[tokio::test]
    async fn observe_mode_logs_one_warn_per_storing_call_with_both_counts_and_the_first_failure() {
        let parts: [&[u8]; 4] = [
            b"admitted",
            b"hash mismatch, first failure",
            b"size mismatch",
            b"hash mismatch, second",
        ];
        let mut first_hash = checked_set_field("b", parts[1], 2);
        first_hash.value_or_ext_hash = ext_content_hash(b"x").to_vec();
        let mut size_off = checked_set_field("c", parts[2], 3);
        size_off.ext_value_size -= 1;
        let mut second_hash = checked_set_field("d", parts[3], 4);
        second_hash.value_or_ext_hash = ext_content_hash(b"y").to_vec();
        let delta = oodf_ffi::OwnedDelta {
            set_fields: vec![
                checked_set_field("a", parts[0], 1),
                first_hash,
                size_off,
                second_hash,
            ],
            ..Default::default()
        };
        let store = PolicyRecordingStore::new(crate::config::ExtValueVerifyMode::Observe);

        let (result, warns) = content_check_warns(&delta, &parts, &store).await;

        result.expect("observe stores");
        assert_eq!(
            warns.len(),
            1,
            "one WARN per call, however many references fail: {warns:?}"
        );
        let fields = &warns[0];
        assert_eq!(fields["size_mismatches"], 1, "{fields}");
        assert_eq!(fields["hash_mismatches"], 2, "{fields}");
        assert_eq!(
            fields["ext_value_index"], 2,
            "the WARN describes the first failing reference: {fields}"
        );
        assert_eq!(fields["mismatch"], "hash_mismatch", "{fields}");
        assert_eq!(
            fields["claimed_hash"],
            sha256_to_hex(&ext_content_hash(b"x")),
            "{fields}"
        );
        // The first failing reference's part is its declared size: only its
        // hash is wrong.
        assert_eq!(fields["part_len"], parts[1].len(), "{fields}");
        assert_eq!(fields["ext_value_size"], parts[1].len(), "{fields}");
    }

    /// No WARN for an `observe` call whose every reference passes.
    #[tokio::test]
    async fn observe_mode_logs_no_warn_for_an_all_ok_call() {
        let parts: [&[u8]; 2] = [b"one", b"two"];
        let delta = oodf_ffi::OwnedDelta {
            set_fields: vec![
                checked_set_field("a", parts[0], 1),
                checked_set_field("b", parts[1], 2),
            ],
            ..Default::default()
        };
        let store = PolicyRecordingStore::new(crate::config::ExtValueVerifyMode::Observe);

        let (result, warns) = content_check_warns(&delta, &parts, &store).await;

        result.expect("an all-ok call stores");
        assert!(warns.is_empty(), "no mismatch, no WARN: {warns:?}");
    }

    /// `enforce` refuses a mismatch and logs no content-check WARN: the refusal
    /// surfaces on the RPC failure line instead.
    #[tokio::test]
    async fn enforce_mode_logs_no_content_check_warn() {
        let part: &[u8] = b"a part whose claim is wrong";
        let mut field = checked_set_field("a", part, 1);
        field.value_or_ext_hash = ext_content_hash(b"other").to_vec();
        let delta = oodf_ffi::OwnedDelta {
            set_fields: vec![field],
            ..Default::default()
        };
        let store = PolicyRecordingStore::new(crate::config::ExtValueVerifyMode::Enforce);

        let (result, warns) = content_check_warns(&delta, &[part], &store).await;

        result.expect_err("enforce refuses");
        assert!(
            warns.is_empty(),
            "enforce logs no content-check WARN: {warns:?}"
        );
    }

    /// An `observe` call with a mismatch and a LATER wrong-length claim is
    /// refused by the collector, which checks every claim's length before the
    /// content check runs: nothing is stored, nothing is WARNed, and nothing is
    /// counted.
    #[tokio::test]
    async fn observe_mode_refused_by_a_later_wrong_length_claim_logs_no_warn_and_counts_nothing() {
        let parts: [&[u8]; 2] = [b"hash mismatch first", b"then a 33-byte claim"];
        let mut mismatch = checked_set_field("a", parts[0], 1);
        mismatch.value_or_ext_hash = ext_content_hash(b"z").to_vec();
        let mut long = checked_set_field("b", parts[1], 2);
        long.value_or_ext_hash.push(0x00);
        let delta = oodf_ffi::OwnedDelta {
            set_fields: vec![mismatch, long],
            ..Default::default()
        };
        let store = PolicyRecordingStore::new(crate::config::ExtValueVerifyMode::Observe);

        let (result, warns) = content_check_warns(&delta, &parts, &store).await;

        invalid_argument_text(
            result.expect_err("the 33-byte claim refuses the call"),
            "later wrong-length claim",
        );
        assert!(
            warns.is_empty(),
            "a refused call stores nothing and is not WARNed: {warns:?}"
        );
        assert!(store.calls().is_empty(), "nothing is stored");
        assert_eq!(
            store.cells(),
            VerifyCells {
                ok: 0,
                size_mismatch: 0,
                hash_mismatch: 0,
            },
            "the collector refuses before the content check, so nothing was checked"
        );
    }

    /// An `observe` call with a mismatch and a LATER bad `ext_value_index` is
    /// refused by the collector, which resolves every index before the content
    /// check runs: nothing is stored, nothing is WARNed, and nothing is counted.
    #[tokio::test]
    async fn observe_mode_refused_by_a_later_bad_index_logs_no_warn_and_counts_nothing() {
        let part: &[u8] = b"hash mismatch first";
        let mut mismatch = checked_set_field("a", part, 1);
        mismatch.value_or_ext_hash = ext_content_hash(b"z").to_vec();
        let mut bad_index = checked_set_field("b", part, 9);
        bad_index.value_or_ext_hash = ext_content_hash(b"w").to_vec();
        let delta = oodf_ffi::OwnedDelta {
            set_fields: vec![mismatch, bad_index],
            ..Default::default()
        };
        let store = PolicyRecordingStore::new(crate::config::ExtValueVerifyMode::Observe);

        let (result, warns) = content_check_warns(&delta, &[part], &store).await;

        let msg = invalid_argument_text(
            result.expect_err("the bad index refuses the call"),
            "later bad index",
        );
        assert!(msg.contains("ext_value_index 9 out of range"), "{msg}");
        assert!(warns.is_empty(), "a refused call is not WARNed: {warns:?}");
        assert!(store.calls().is_empty(), "nothing is stored");
        assert_eq!(
            store.cells(),
            NO_CELLS,
            "the collector refuses before the content check, so nothing was checked"
        );
    }

    // ── collect_external_hashes ──────────────────────────────────────────

    #[test]
    fn collect_external_hashes_finds_fields_and_time_samples() {
        use ordered_float::OrderedFloat;

        let h1 = test_hash(0x01);
        let h2 = test_hash(0x02);

        let mut state = ObjectState::default();
        state.fields.insert(
            (1, "ext-field".to_string()),
            Field {
                section_id: 1,
                key_name: "ext-field".to_string(),
                value: FieldValue::External {
                    hash: h1,
                    size: 100,
                    ext_value_index: 1,
                },
                set_order: 0,
                authoring_set_order: 0,
                version: 1,
            },
        );
        state.time_samples.insert(
            (2, OrderedFloat(1.0)),
            TimeSample {
                section_id: 2,
                time: OrderedFloat(1.0),
                value: FieldValue::External {
                    hash: h2,
                    size: 200,
                    ext_value_index: 2,
                },
                version: 1,
            },
        );
        // Inline field should not appear.
        state.fields.insert(
            (3, "inline".to_string()),
            Field {
                section_id: 3,
                key_name: "inline".to_string(),
                value: FieldValue::Inline(vec![0xFF]),
                set_order: 1,
                authoring_set_order: 1,
                version: 1,
            },
        );

        let hashes = collect_external_hashes(&state);
        assert_eq!(hashes.len(), 2);
        assert!(hashes.contains(&h1));
        assert!(hashes.contains(&h2));
    }

    #[test]
    fn collect_external_hashes_empty_state_returns_empty() {
        let state = ObjectState::default();
        let hashes = collect_external_hashes(&state);
        assert!(hashes.is_empty());
    }

    #[test]
    fn collect_external_hashes_deduplicates() {
        let hash = test_hash(0xAA);
        let mut state = ObjectState::default();
        state.fields.insert(
            (1, "f1".to_string()),
            Field {
                section_id: 1,
                key_name: "f1".to_string(),
                value: FieldValue::External {
                    hash,
                    size: 100,
                    ext_value_index: 1,
                },
                set_order: 0,
                authoring_set_order: 0,
                version: 1,
            },
        );
        state.fields.insert(
            (2, "f2".to_string()),
            Field {
                section_id: 2,
                key_name: "f2".to_string(),
                value: FieldValue::External {
                    hash,
                    size: 100,
                    ext_value_index: 1,
                },
                set_order: 1,
                authoring_set_order: 1,
                version: 1,
            },
        );

        let hashes = collect_external_hashes(&state);
        assert_eq!(hashes.len(), 1, "same hash referenced twice → dedup to 1");
    }

    // ── read_batch ──────────────────────────────────────────────────────

    #[tokio::test]
    async fn read_batch_empty_returns_empty_no_errors() {
        let mock = MockExternalValueStore::new();
        let result = mock.read_batch(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn read_batch_single_success() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xDD);
        let data = b"payload bytes";

        mock.store_values(&[(&hash, data)]).await.unwrap();

        let result = mock.read_batch(&[hash]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[&hash], data);
    }

    #[tokio::test]
    async fn read_batch_multiple_all_present() {
        let mock = MockExternalValueStore::new();
        let h1 = test_hash(0x01);
        let h2 = test_hash(0x02);
        let h3 = test_hash(0x03);

        mock.store_values(&[(&h1, b"d1"), (&h2, b"d2"), (&h3, b"d3")])
            .await
            .unwrap();

        let result = mock.read_batch(&[h1, h2, h3]).await.unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[&h1], b"d1");
        assert_eq!(result[&h2], b"d2");
        assert_eq!(result[&h3], b"d3");
    }

    #[tokio::test]
    async fn read_batch_missing_meta_returns_error() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xFF);

        let result = mock.read_batch(&[hash]).await;
        assert!(result.is_err(), "missing hash must return error");

        match result.unwrap_err() {
            ServiceError::Internal(msg) => {
                assert!(
                    msg.contains("meta not found"),
                    "error should mention meta not found, got: {msg}"
                );
            }
            other => panic!("expected Internal error, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn read_batch_partial_missing_returns_error() {
        let mock = MockExternalValueStore::new();
        let h1 = test_hash(0x01);
        let h_missing = test_hash(0xFF);

        mock.store_values(&[(&h1, b"data1")]).await.unwrap();

        let result = mock.read_batch(&[h1, h_missing]).await;
        assert!(result.is_err(), "batch with missing hash must return error");
    }

    // ── read_batch SR-3.2 recovery wiring ───────────────────────────────
    //
    // The tests below exercise the SR-3.2 recovery contract Task
    // `t2.fgj.zc.p6` delivers end-to-end through `MockExternalValueStore::
    // read_batch` → `try_recover_via_checkpoint` → `ExtRecoverySingleflight::
    // fetch_once` → `CheckpointStore::get_ext_blob`. The singleflight
    // primitive itself is covered in isolation by
    // `crate::storage::ext_recovery_singleflight::tests`; what these tests
    // add is the wiring: that the mock's `read_batch` consults the
    // singleflight only on miss, that the singleflight coalesces
    // concurrent same-hash misses into one adapter call, that a
    // post-success rewrite into Valkey lets the second batch skip the
    // adapter entirely, and that dropping some concurrent waiters does
    // not prevent the remaining waiters from resolving.

    /// Test-only `CheckpointStore` that counts `get_ext_blob` calls
    /// and sleeps a fixed duration before returning the configured
    /// payload. The sleep models real adapter slowness (multi-MB blob
    /// fetch) so all N concurrent same-hash misses have time to
    /// register on the singleflight before the inner future resolves;
    /// it is NOT a synchronization primitive (no test coordinates on
    /// the sleep). Barrier-based rendezvous between the inner future
    /// and the test driver deadlocks under `cargo test`'s parallel
    /// test-threads model when many `multi_thread` tokio runtimes
    /// contend for host CPU, whereas `tokio::time::sleep` resolves
    /// deterministically when its duration elapses regardless of
    /// scheduler pressure. Every non-`get_ext_blob` trait method is a
    /// `NotImplemented` error so an accidental call from the wiring
    /// surfaces as a loud test failure.
    struct SleepyCountingCheckpointStore {
        get_calls: Arc<std::sync::atomic::AtomicUsize>,
        sleep_duration: std::time::Duration,
        payload: bytes::Bytes,
    }

    #[async_trait::async_trait]
    impl crate::checkpoint::CheckpointStore for SleepyCountingCheckpointStore {
        async fn put_layer_checkpoint(
            &self,
            _object_id: u64,
            _snap_seq: u64,
            _ompf: bytes::Bytes,
            _meta: crate::checkpoint::CheckpointMeta,
        ) -> Result<crate::checkpoint::CheckpointHandle, CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "put_layer_checkpoint",
            })
        }
        async fn get_latest_layer_checkpoint(
            &self,
            _object_id: u64,
        ) -> Result<Option<crate::checkpoint::CheckpointBlob>, CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "get_latest_layer_checkpoint",
            })
        }
        async fn list_layer_checkpoints(
            &self,
            _object_id: u64,
        ) -> Result<Vec<crate::checkpoint::CheckpointHandle>, CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "list_layer_checkpoints",
            })
        }
        async fn delete_layer_checkpoint(
            &self,
            _handle: crate::checkpoint::CheckpointHandle,
        ) -> Result<(), CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "delete_layer_checkpoint",
            })
        }
        async fn put_session_checkpoint(
            &self,
            _session_id: &str,
            _body: bytes::Bytes,
            _meta: crate::checkpoint::CheckpointMeta,
        ) -> Result<crate::checkpoint::CheckpointHandle, CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "put_session_checkpoint",
            })
        }
        async fn get_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<Option<crate::checkpoint::CheckpointBlob>, CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "get_session_checkpoint",
            })
        }
        async fn delete_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<(), CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "delete_session_checkpoint",
            })
        }
        async fn put_ext_blob(
            &self,
            _hash_hex: &str,
            _bytes: bytes::Bytes,
        ) -> Result<(), CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "put_ext_blob",
            })
        }
        async fn get_ext_blob(
            &self,
            _hash_hex: &str,
        ) -> Result<Option<bytes::Bytes>, CheckpointError> {
            self.get_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            tokio::time::sleep(self.sleep_duration).await;
            Ok(Some(self.payload.clone()))
        }
        async fn delete_ext_blob(&self, _hash_hex: &str) -> Result<(), CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "delete_ext_blob",
            })
        }
        async fn put_counters_sidecar(&self, _body: bytes::Bytes) -> Result<(), CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "put_counters_sidecar",
            })
        }
        async fn get_counters_sidecar(&self) -> Result<Option<bytes::Bytes>, CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "get_counters_sidecar",
            })
        }
        async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "list_all_object_ids",
            })
        }
    }

    /// Test-only `CheckpointStore` that increments a counter on **every**
    /// trait method call. Used by the no-miss happy-path test to assert
    /// the adapter is never consulted when the hot path hits.
    ///
    /// Every trait method must be written out by hand and must `tick()`,
    /// including the ones that carry a provided default on the trait
    /// (`get_layer_checkpoint`, `list_session_checkpoint_ids`,
    /// `ext_blob_exists`, `list_referenced_ext_hashes`,
    /// `read_ext_refs_sidecar`, `list_layer_artifacts`,
    /// `delete_layer_temp_artifacts`, `publish_worst_case`). Inheriting a
    /// default here would leave the
    /// counter blind to that method: an implementation that started calling
    /// it on the no-miss path would still read `any_call_count == 0` and the
    /// "MUST NOT invoke any CheckpointStore trait method" assertion below
    /// would silently pass. Anything added to the trait needs an arm here.
    struct ProbeCountingCheckpointStore {
        any_call_count: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ProbeCountingCheckpointStore {
        fn tick(&self) {
            self.any_call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[async_trait::async_trait]
    impl crate::checkpoint::CheckpointStore for ProbeCountingCheckpointStore {
        async fn put_layer_checkpoint(
            &self,
            _object_id: u64,
            _snap_seq: u64,
            _ompf: bytes::Bytes,
            _meta: crate::checkpoint::CheckpointMeta,
        ) -> Result<crate::checkpoint::CheckpointHandle, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "put_layer_checkpoint",
            })
        }
        async fn get_latest_layer_checkpoint(
            &self,
            _object_id: u64,
        ) -> Result<Option<crate::checkpoint::CheckpointBlob>, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "get_latest_layer_checkpoint",
            })
        }
        async fn get_layer_checkpoint(
            &self,
            _handle: &crate::checkpoint::CheckpointHandle,
        ) -> Result<Option<crate::checkpoint::CheckpointBlob>, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "get_layer_checkpoint",
            })
        }
        async fn list_layer_checkpoints(
            &self,
            _object_id: u64,
        ) -> Result<Vec<crate::checkpoint::CheckpointHandle>, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "list_layer_checkpoints",
            })
        }
        async fn delete_layer_checkpoint(
            &self,
            _handle: crate::checkpoint::CheckpointHandle,
        ) -> Result<(), CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "delete_layer_checkpoint",
            })
        }
        async fn put_session_checkpoint(
            &self,
            _session_id: &str,
            _body: bytes::Bytes,
            _meta: crate::checkpoint::CheckpointMeta,
        ) -> Result<crate::checkpoint::CheckpointHandle, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "put_session_checkpoint",
            })
        }
        async fn get_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<Option<crate::checkpoint::CheckpointBlob>, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "get_session_checkpoint",
            })
        }
        async fn delete_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<(), CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "delete_session_checkpoint",
            })
        }
        async fn put_ext_blob(
            &self,
            _hash_hex: &str,
            _bytes: bytes::Bytes,
        ) -> Result<(), CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "put_ext_blob",
            })
        }
        async fn get_ext_blob(
            &self,
            _hash_hex: &str,
        ) -> Result<Option<bytes::Bytes>, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "get_ext_blob",
            })
        }
        async fn delete_ext_blob(&self, _hash_hex: &str) -> Result<(), CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "delete_ext_blob",
            })
        }
        async fn put_counters_sidecar(&self, _body: bytes::Bytes) -> Result<(), CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "put_counters_sidecar",
            })
        }
        async fn get_counters_sidecar(&self) -> Result<Option<bytes::Bytes>, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "get_counters_sidecar",
            })
        }
        async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "list_all_object_ids",
            })
        }
        async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "list_session_checkpoint_ids",
            })
        }
        async fn ext_blob_exists(&self, _hash_hex: &str) -> Result<bool, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "ext_blob_exists",
            })
        }
        async fn list_referenced_ext_hashes(
            &self,
        ) -> Result<std::collections::HashSet<String>, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "list_referenced_ext_hashes",
            })
        }
        async fn read_ext_refs_sidecar(
            &self,
            _handle: &crate::checkpoint::CheckpointHandle,
        ) -> Result<std::collections::HashSet<String>, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "read_ext_refs_sidecar",
            })
        }
        async fn list_layer_artifacts(
            &self,
        ) -> Result<Vec<crate::checkpoint::LayerArtifact>, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "list_layer_artifacts",
            })
        }
        async fn delete_layer_temp_artifacts(
            &self,
            _object_id: u64,
            _snap_seq: u64,
        ) -> Result<usize, CheckpointError> {
            self.tick();
            Err(CheckpointError::NotImplemented {
                method: "delete_layer_temp_artifacts",
            })
        }
        // Synchronous and infallible on the trait, so it cannot answer
        // `NotImplemented`; it ticks and returns the trait default's value.
        fn publish_worst_case(&self) -> std::time::Duration {
            self.tick();
            crate::checkpoint::store::LOCAL_PUBLISH_WORST_CASE
        }
    }

    /// Concurrent `read_batch` calls that all miss on the same hash MUST
    /// coalesce through `ExtRecoverySingleflight` into exactly one
    /// `CheckpointStore::get_ext_blob` invocation; every spawned caller
    /// must observe the same recovered bytes. This is Task
    /// `t2.fgj.zc.p6` Success Criterion #3 — the thundering-herd
    /// protection contract.
    ///
    /// The slow-adapter model uses `tokio::time::sleep` to give all
    /// N spawned tasks ample time to register on the singleflight
    /// before the inner future resolves. The sleep is the adapter's
    /// internal latency model, NOT a synchronization primitive — see
    /// the `SleepyCountingCheckpointStore` doc-comment for why
    /// barrier-rendezvous patterns are unsafe under `cargo test`'s
    /// parallel test-threads load.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn read_batch_singleflight_coalesces_concurrent_same_hash_misses() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;
        use tokio::sync::Barrier;

        let payload = bytes::Bytes::from_static(b"recovered-payload-bytes");
        let get_calls = Arc::new(AtomicUsize::new(0));

        let store: Arc<dyn crate::checkpoint::CheckpointStore> =
            Arc::new(SleepyCountingCheckpointStore {
                get_calls: Arc::clone(&get_calls),
                // 200 ms is generous on the high end so all 64 spawned
                // tasks register on the singleflight before the inner
                // future resolves, even under heavy `cargo test`
                // parallel load.
                sleep_duration: Duration::from_millis(200),
                payload: payload.clone(),
            });

        let mock = Arc::new(MockExternalValueStore::new().with_checkpoint_store(store));
        let hash = test_hash(0xA1);

        // Hash is intentionally NOT pre-stored so every concurrent
        // `read_batch` enters the miss arm and consults the singleflight.
        const N: usize = 64;
        let task_barrier = Arc::new(Barrier::new(N));

        let mut handles = Vec::with_capacity(N);
        for _ in 0..N {
            let mock = Arc::clone(&mock);
            let task_barrier = Arc::clone(&task_barrier);
            handles.push(tokio::spawn(async move {
                task_barrier.wait().await;
                mock.read_batch(&[hash]).await
            }));
        }

        for h in handles {
            let map = h
                .await
                .expect("spawned task panicked")
                .expect("read_batch via recovery must succeed");
            assert_eq!(map.len(), 1);
            assert_eq!(
                map[&hash],
                payload.as_ref(),
                "every coalesced waiter must observe the same recovered bytes"
            );
        }

        assert_eq!(
            get_calls.load(Ordering::SeqCst),
            1,
            "64 concurrent same-hash misses must coalesce into exactly one adapter fetch"
        );
    }

    /// After the singleflight resolves and the recovered bytes have been
    /// rewritten into Valkey (via `store_recovered_values`), a second batch of
    /// concurrent `read_batch` calls for the same hash MUST hit the
    /// `Some(d)` hot path. Zero additional adapter calls; zero
    /// singleflight participation. Mirrors the cross-Task contract that
    /// the singleflight entry is released on success so post-recovery
    /// reads are byte-identical to pre-recovery hot-path reads.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn read_batch_singleflight_cleanup_after_success_skips_adapter_on_second_pass() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;
        use tokio::sync::Barrier;

        let payload = bytes::Bytes::from_static(b"recovered-once-then-cached");
        let get_calls = Arc::new(AtomicUsize::new(0));

        let store: Arc<dyn crate::checkpoint::CheckpointStore> =
            Arc::new(SleepyCountingCheckpointStore {
                get_calls: Arc::clone(&get_calls),
                sleep_duration: Duration::from_millis(200),
                payload: payload.clone(),
            });

        let mock = Arc::new(MockExternalValueStore::new().with_checkpoint_store(store));
        let hash = test_hash(0xA2);

        // ── Phase 1: 64 concurrent misses coalesce into one adapter call.
        const N: usize = 64;
        let phase1_barrier = Arc::new(Barrier::new(N));
        let mut phase1 = Vec::with_capacity(N);
        for _ in 0..N {
            let mock = Arc::clone(&mock);
            let phase1_barrier = Arc::clone(&phase1_barrier);
            phase1.push(tokio::spawn(async move {
                phase1_barrier.wait().await;
                mock.read_batch(&[hash]).await
            }));
        }

        for h in phase1 {
            let map = h.await.expect("spawned task panicked").expect("recovery");
            assert_eq!(map[&hash], payload.as_ref());
        }
        assert_eq!(
            get_calls.load(Ordering::SeqCst),
            1,
            "phase 1 must coalesce to exactly one adapter fetch"
        );
        assert_eq!(
            mock.recovery_singleflight_inflight_len(),
            0,
            "singleflight entry must be released on success"
        );

        // ── Phase 2: 64 more concurrent reads on the same hash must
        // skip the adapter entirely (Valkey-rehydrate hot path). If
        // Phase 2 incorrectly re-entered the adapter, the counter
        // would increment again; the assertion below would fail.
        let phase2_barrier = Arc::new(Barrier::new(N));
        let mut phase2 = Vec::with_capacity(N);
        for _ in 0..N {
            let mock = Arc::clone(&mock);
            let phase2_barrier = Arc::clone(&phase2_barrier);
            phase2.push(tokio::spawn(async move {
                phase2_barrier.wait().await;
                mock.read_batch(&[hash]).await
            }));
        }

        for h in phase2 {
            let map = h
                .await
                .expect("spawned task panicked")
                .expect("post-recovery read must succeed");
            assert_eq!(
                map[&hash],
                payload.as_ref(),
                "post-recovery hot-path read returns the rewritten bytes"
            );
        }

        assert_eq!(
            get_calls.load(Ordering::SeqCst),
            1,
            "post-recovery batch must NOT consult the adapter (singleflight cleanup contract)"
        );
        assert_eq!(
            mock.recovery_singleflight_inflight_len(),
            0,
            "no leftover singleflight entries after Phase 2"
        );
    }

    /// When every requested hash is present in the store, `read_batch`
    /// MUST take the `Some(d)` hot-path arm only — never invoking any
    /// adapter method and never touching the singleflight map. This is
    /// Task `t2.fgj.zc.p6` Success Criterion #5: the singleflight is
    /// consulted **only on a Valkey miss**.
    #[tokio::test]
    async fn read_batch_no_miss_path_skips_adapter_and_keeps_singleflight_map_empty() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let any_call_count = Arc::new(AtomicUsize::new(0));
        let probe: Arc<dyn crate::checkpoint::CheckpointStore> =
            Arc::new(ProbeCountingCheckpointStore {
                any_call_count: Arc::clone(&any_call_count),
            });

        let mock = MockExternalValueStore::new().with_checkpoint_store(probe);
        let h1 = test_hash(0xB1);
        let h2 = test_hash(0xB2);
        let h3 = test_hash(0xB3);
        mock.store_values(&[(&h1, b"d1"), (&h2, b"d2"), (&h3, b"d3")])
            .await
            .unwrap();

        let result = mock
            .read_batch(&[h1, h2, h3])
            .await
            .expect("hot path must succeed");
        assert_eq!(result.len(), 3);
        assert_eq!(result[&h1], b"d1");
        assert_eq!(result[&h2], b"d2");
        assert_eq!(result[&h3], b"d3");

        assert_eq!(
            any_call_count.load(Ordering::SeqCst),
            0,
            "no-miss hot path MUST NOT invoke any CheckpointStore trait method"
        );
        assert_eq!(
            mock.recovery_singleflight_inflight_len(),
            0,
            "no-miss hot path MUST NOT register any singleflight entry"
        );
    }

    /// Cancellation safety: when some concurrent same-hash misses abort
    /// before the singleflight inner future resolves, the remaining
    /// (non-cancelled) waiters MUST still observe the recovered bytes.
    /// This pins the `scopeguard::guard` cleanup pattern in
    /// `ExtRecoverySingleflight::fetch_once`: a dropped waiter does not
    /// poison the shared future for its peers (b.j5t item 4 equivalent).
    ///
    /// The slow-adapter model uses `tokio::time::sleep` to simulate a
    /// multi-MB blob fetch latency rather than barrier coordination —
    /// barrier-based rendezvous between main and the inner future
    /// fights with `cargo test`'s parallel test-threads model under
    /// heavy CPU contention (the runtime scheduler can starve the
    /// barrier participant), whereas `sleep` does not coordinate
    /// tasks at all (it models adapter slowness only) and resolves
    /// deterministically when its duration elapses.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn read_batch_singleflight_survives_concurrent_caller_cancellation() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;
        use tokio::sync::Barrier;

        let payload = bytes::Bytes::from_static(b"survive-cancellation");
        let get_calls = Arc::new(AtomicUsize::new(0));

        let store: Arc<dyn crate::checkpoint::CheckpointStore> =
            Arc::new(SleepyCountingCheckpointStore {
                get_calls: Arc::clone(&get_calls),
                sleep_duration: Duration::from_millis(200),
                payload: payload.clone(),
            });

        let mock = Arc::new(MockExternalValueStore::new().with_checkpoint_store(store));
        let hash = test_hash(0xC1);

        // Spawn 4 concurrent same-hash misses. The task_barrier
        // synchronizes their start so they all enter the singleflight
        // window before the abort fires; the adapter's 200 ms sleep
        // window then gives main ample time to issue the aborts before
        // the inner future would otherwise resolve.
        const N: usize = 4;
        let task_barrier = Arc::new(Barrier::new(N));
        let mut handles = Vec::with_capacity(N);
        for _ in 0..N {
            let mock = Arc::clone(&mock);
            let task_barrier = Arc::clone(&task_barrier);
            handles.push(tokio::spawn(async move {
                task_barrier.wait().await;
                mock.read_batch(&[hash]).await
            }));
        }

        // Wait until the inner adapter call has started (counter ticked)
        // — but bound by the adapter sleep budget so a stuck poll loop
        // surfaces as a deterministic test timeout, not a forever-spin.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while get_calls.load(Ordering::SeqCst) == 0 {
            if std::time::Instant::now() >= deadline {
                panic!("adapter never ticked get_calls — inner future was not polled");
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        // Abort 2 of the 4 spawned tasks while the adapter is still
        // sleeping. The remaining 2 must still resolve.
        let cancelled_a = handles.swap_remove(0);
        let cancelled_b = handles.swap_remove(0);
        cancelled_a.abort();
        cancelled_b.abort();
        let _ = cancelled_a.await;
        let _ = cancelled_b.await;

        for h in handles {
            // Generous budget covers `cargo test`'s parallel test-thread
            // CPU contention; in isolation this test resolves in
            // ~200 ms (the adapter sleep). A real hang surfaces as a
            // timeout, not an indefinite cargo-test wait.
            let map = tokio::time::timeout(Duration::from_secs(30), h)
                .await
                .expect("surviving waiter timed out")
                .expect("surviving waiter panicked")
                .expect("surviving waiter must observe Ok");
            assert_eq!(
                map[&hash],
                payload.as_ref(),
                "surviving waiter must see the recovered bytes (cancellation-safety contract)"
            );
        }

        // The load-bearing post-condition: at least one adapter fetch
        // succeeded (otherwise the survivors could not have observed
        // bytes), and the singleflight map drains after survivors
        // resolve. The exact adapter-call count is NOT asserted as
        // strictly == 1: the singleflight's ptr_eq cleanup is keyed
        // on the dropped caller's Shared, so a registering task that
        // drops mid-flight legitimately permits a follow-up caller
        // to install a fresh entry — pinning that count to 1 would
        // test an implementation detail rather than the contract.
        assert!(
            get_calls.load(Ordering::SeqCst) >= 1,
            "at least one adapter fetch must have landed for survivors to resolve"
        );
        // Allow the survivors' scopeguard cleanup to settle; DashMap
        // entry removal happens after `shared.await` returns and may
        // briefly race with this assertion on a fast multi-threaded
        // runtime.
        for _ in 0..50 {
            if mock.recovery_singleflight_inflight_len() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            mock.recovery_singleflight_inflight_len(),
            0,
            "no leftover singleflight entries after survivors resolve"
        );
    }

    // ── GC via mock: missing metadata handled gracefully ────────────────

    #[tokio::test]
    async fn process_pending_gc_missing_metadata_skips_gracefully() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xFF);

        // Add to pending without storing data -- simulates metadata already deleted.
        mock.schedule_release(&[rel(hash)]).await.unwrap();
        assert_eq!(mock.pending_count(), 1);

        let stats = process_pending_gc(&mock, None, GcPolicy::IMMEDIATE).await;
        assert_eq!(stats.processed, 1);
        assert_eq!(mock.pending_count(), 0, "removed from pending");
    }

    /// A release member enqueued again after its decrement deleted the blob
    /// — a fred re-send of the enqueue, say — is a replay (b.yuo): within
    /// the replay-record retention it is answered `replayed` and decrements
    /// nothing, even when the same content was re-stored in between. Before
    /// b.yuo the replay decremented the re-stored blob's only reference and
    /// deleted it under the layer that had just stored it.
    #[tokio::test]
    async fn process_pending_gc_replay_after_delete_and_restore_keeps_the_new_blob() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xEE);
        // IMMEDIATE, except that replay records outlive the run.
        let policy = GcPolicy {
            replay_record_retention: GC_REPLAY_RECORD_RETENTION,
            ..GcPolicy::IMMEDIATE
        };

        mock.store_values(&[(&hash, b"data")]).await.unwrap();
        mock.schedule_release(&[rel(hash)]).await.unwrap();
        let stats = process_pending_gc(&mock, None, policy).await;
        assert_eq!((stats.deleted, stats.replayed), (1, 0));
        assert_eq!(mock.entry_count(), 0, "entry deleted");

        // The same content is stored again by a new reference, then the
        // identical release member arrives a second time.
        mock.store_values(&[(&hash, b"data")]).await.unwrap();
        mock.schedule_release(&[rel(hash)]).await.unwrap();
        let replay = process_pending_gc(&mock, None, policy).await;
        assert_eq!(
            (replay.processed, replay.replayed, replay.deleted),
            (1, 1, 0),
            "the replayed member is claimed and answered as a replay"
        );
        assert_eq!(
            mock.ref_count(&hash),
            Some(1),
            "the re-stored blob keeps its one reference"
        );
    }

    #[tokio::test]
    async fn process_pending_gc_empty_pending_completes_without_error() {
        let mock = MockExternalValueStore::new();
        let stats = process_pending_gc(&mock, None, GcPolicy::IMMEDIATE).await;
        assert_eq!(stats.processed, 0);
        assert_eq!(stats.deleted, 0);
    }

    #[tokio::test]
    async fn process_pending_gc_shared_refcount_preserves_value() {
        let mock = MockExternalValueStore::new();
        let hash = test_hash(0xAB);

        // Store twice -> ref_count=2.
        mock.store_values(&[(&hash, b"shared")]).await.unwrap();
        mock.store_values(&[(&hash, b"shared")]).await.unwrap();
        assert_eq!(mock.ref_count(&hash), Some(2));

        mock.schedule_release(&[rel(hash)]).await.unwrap();
        let stats = process_pending_gc(&mock, None, GcPolicy::IMMEDIATE).await;

        assert_eq!(stats.processed, 1);
        assert_eq!(stats.deleted, 0, "value still has references");
        assert_eq!(mock.ref_count(&hash), Some(1));
        assert_eq!(mock.entry_count(), 1, "entry preserved");
    }

    #[tokio::test]
    async fn process_pending_gc_drains_all_entries() {
        let mock = MockExternalValueStore::new();
        let h1 = test_hash(0x01);
        let h2 = test_hash(0x02);

        mock.store_values(&[(&h1, b"a"), (&h2, b"b")])
            .await
            .unwrap();
        mock.schedule_release(&[rel(h1), rel(h2)]).await.unwrap();

        let stats = process_pending_gc(&mock, None, GcPolicy::IMMEDIATE).await;
        assert_eq!(stats.processed, 2);
        assert_eq!(stats.deleted, 2);
        assert_eq!(mock.pending_count(), 0);
        assert_eq!(mock.entry_count(), 0);
    }

    // ── spawn_gc_task lifecycle tests ───────────────────────────────────

    #[tokio::test]
    async fn gc_task_is_non_blocking() {
        let mock = Arc::new(MockExternalValueStore::new());
        let cancel = tokio_util::sync::CancellationToken::new();

        // spawn_gc_task is a sync function that returns a JoinHandle immediately.
        // It must not block the calling thread.
        let handle = spawn_gc_task_with_leader(
            mock as Arc<dyn ExternalValueStore>,
            GcPolicy::IMMEDIATE,
            cancel.clone(),
            None,
            || async { true },
        );

        // The handle must be valid (task is running).
        assert!(
            !handle.is_finished(),
            "task should be running, not finished"
        );

        // Clean up.
        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_millis(500), handle).await;
    }

    #[tokio::test]
    async fn gc_task_cancels_cleanly() {
        let mock = Arc::new(MockExternalValueStore::new());
        let cancel = tokio_util::sync::CancellationToken::new();

        let handle = spawn_gc_task_with_leader(
            mock as Arc<dyn ExternalValueStore>,
            GcPolicy::IMMEDIATE,
            cancel.clone(),
            None,
            || async { true },
        );

        // Give the task a moment to start, then cancel.
        tokio::time::sleep(Duration::from_millis(10)).await;
        cancel.cancel();

        // Task should exit within 100ms of cancellation.
        let result = tokio::time::timeout(Duration::from_millis(100), handle).await;
        assert!(
            result.is_ok(),
            "GC task must exit within 100ms of cancellation"
        );
        assert!(
            result.unwrap().is_ok(),
            "GC task must not panic on cancellation"
        );
    }

    // ── b.avd HIGH follow-up: store_values inflight bound ──────────────
    //
    // `ValkeyExternalValueStore::store_values` uses
    // `futures::stream::iter(...).buffered(STORE_VALUES_MAX_INFLIGHT)` so
    // a CreateLayer with hundreds of distinct external-value hashes
    // cannot put more concurrent blob writes on the wire than the pool
    // they ride has connections. Pre-fix this used `try_join_all` over
    // every hash, which spawned `values.len()` concurrent waiters.
    //
    // b.rej re-argued WHAT the bound protects without changing the
    // bound. The wave no longer takes exclusive `txn_pool` slots at all
    // — each store is one `EVALSHA` of `ext_store_value.lua` on the
    // MULTIPLEXED data pool — so the original claim ("N concurrent
    // `acquire_txn` waiters pin every slot and starve `UpdateLayer` /
    // `CreateSession`") no longer describes anything. What binds now is
    // head-of-line blocking: a multi-MB blob occupies the connection
    // fred routed it onto for the whole of its transfer, and every read
    // and Lua fast-path commit multiplexed onto that connection waits
    // behind it. The invariant that keeps the effect bounded is that the
    // pool is AT LEAST AS WIDE AS the wave — enforced at startup by
    // `Config::validate_pool_size_vs_store_values_inflight` and, on the
    // auto path, by `effective_pool_size`'s floor.
    //
    // Neither pool is mockable at unit-test scale, but the `buffered(N)`
    // semantics the production call relies on are: feed `buffered` N+M
    // synthetic futures, gate each on a per-future barrier, observe the
    // inflight count at the moment all N initial waves are stuck, and
    // assert the bound holds. This pins the regression mode "someone
    // replaces `buffered(N)` with `try_join_all` or raises N past the
    // pool size the startup gate enforces".

    #[tokio::test]
    async fn store_values_buffered_bound_caps_inflight_to_const() {
        use futures::stream::{StreamExt, TryStreamExt};
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::Notify;

        let inflight = Arc::new(AtomicUsize::new(0));
        let max_observed = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(Notify::new());

        // Feed 4 × bound futures so the bound is exercised even if every
        // wave drains instantly.
        let n_futures: usize = STORE_VALUES_MAX_INFLIGHT * 4;
        let inflight_for_closure = Arc::clone(&inflight);
        let max_observed_for_closure = Arc::clone(&max_observed);
        let release_for_closure = Arc::clone(&release);
        let futures = (0..n_futures).map(move |_| {
            let inflight = Arc::clone(&inflight_for_closure);
            let max_observed = Arc::clone(&max_observed_for_closure);
            let release = Arc::clone(&release_for_closure);
            async move {
                let now = inflight.fetch_add(1, Ordering::SeqCst) + 1;
                // CAS-style max tracker — no `fetch_max` on stable
                // AtomicUsize before 1.45; this loop is canonically safe.
                let mut prev = max_observed.load(Ordering::SeqCst);
                while now > prev {
                    match max_observed.compare_exchange_weak(
                        prev,
                        now,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => break,
                        Err(actual) => prev = actual,
                    }
                }
                // Park until the test releases us, then drain.
                release.notified().await;
                inflight.fetch_sub(1, Ordering::SeqCst);
                Ok::<_, ServiceError>(())
            }
        });

        // Background driver that mirrors the production call shape.
        let driver = tokio::spawn({
            let release = Arc::clone(&release);
            let stream = futures::stream::iter(futures).buffered(STORE_VALUES_MAX_INFLIGHT);
            async move {
                // Release every parked future. `notify_waiters` only wakes
                // waiters currently parked, so a steady stream of new
                // arrivals needs repeated wakes — drive a fan-out in
                // parallel with the stream's progress.
                tokio::spawn(async move {
                    for _ in 0..n_futures + STORE_VALUES_MAX_INFLIGHT {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                        release.notify_waiters();
                    }
                });
                stream.try_collect::<Vec<()>>().await
            }
        });

        // Wait for the driver to finish all N+M futures.
        let outcome = tokio::time::timeout(Duration::from_secs(5), driver).await;
        let result = outcome
            .expect("test timed out — driver did not drain")
            .expect("driver task panicked")
            .expect("a synthetic future errored");
        assert_eq!(result.len(), n_futures, "driver must complete every future");

        // Concurrency bound is the load-bearing assertion: at no point
        // did the inflight count exceed STORE_VALUES_MAX_INFLIGHT, which
        // is exactly the head-of-line-blocking contract `store_values`
        // relies on post-b.rej.
        let observed = max_observed.load(Ordering::SeqCst);
        assert!(
            observed <= STORE_VALUES_MAX_INFLIGHT,
            "buffered({STORE_VALUES_MAX_INFLIGHT}) must cap inflight count at \
             {STORE_VALUES_MAX_INFLIGHT}; observed peak was {observed}. If this assertion \
             fails because someone replaced buffered() with try_join_all or with a \
             higher bound, audit the call site against the DATA pool size first — the \
             wave rides the multiplexed data pool (b.rej), where each concurrent blob \
             occupies the connection it was routed onto for its whole transfer and \
             head-of-line-blocks the reads and Lua fast-path commits sharing it. \
             `STORE_VALUES_MAX_INFLIGHT` must stay at most \
             `LIVE_EDIT_VALKEY_POOL_SIZE` — the pool has to be at least as wide as the \
             wave, which `Config::validate_pool_size_vs_store_values_inflight` enforces \
             at startup — and should stay well below it (b.avd HIGH follow-up)."
        );
        // And it must have actually reached the cap (otherwise the test
        // is trivially satisfied by a degenerate stream).
        assert_eq!(
            observed, STORE_VALUES_MAX_INFLIGHT,
            "expected the bound to be hit (driver feeds 4× the cap); got {observed}"
        );
    }
}

/// b.9qd — the `recon_claim_expired.lua` protocol, against a live Valkey.
///
/// These sit in-crate rather than under `tests/integration/` because
/// [`recon_claim_expired`], [`recon_retire`] and [`recon_backlog`] are
/// `pub(crate)`: the ZSETs are an internal coordination detail of the
/// checkpoint reconciliation pass, not a storage-layer API, and widening their
/// visibility to reach them from an integration binary would be a worse trade
/// than putting the tests where the functions are. `TestValkey` gives the same
/// live server either way.
///
/// The script's two behaviours are what the whole reconciliation safety
/// argument rests on, so each gets a dedicated test: `ZADD NX` must pin the
/// FIRST observation (a refreshed score means nothing ever ages out), and the
/// expiry cutoff must be computed from the SAME Valkey-side `TIME` sample the
/// registration used (a caller clock would let a clock-ahead replica expire
/// what its peers had only just registered).
#[cfg(test)]
mod recon_zset_protocol_tests {
    use std::time::Duration;

    use fred::interfaces::SortedSetsInterface;

    use super::{recon_backlog, recon_claim_expired, recon_retire};
    use crate::storage::test_support::TestValkey;

    /// The `{recon}`-tagged key under test. Any key works — the script takes
    /// it from `KEYS[1]` precisely so one implementation serves both the
    /// orphan-candidate and ext-tombstone ZSETs — so the tests use the real
    /// tombstone key to keep the fixture honest about hash-tag placement.
    fn key(tv: &TestValkey) -> String {
        tv.key_schema().checkpoint_reconcile_tombstones()
    }

    async fn members(tv: &TestValkey, key: &str) -> Vec<String> {
        tv.client()
            .pool
            .zrange(key, 0, -1, None, false, None, false)
            .await
            .expect("ZRANGE must succeed")
    }

    async fn score(tv: &TestValkey, key: &str, member: &str) -> Option<f64> {
        tv.client()
            .pool
            .zscore(key, member)
            .await
            .expect("ZSCORE must succeed")
    }

    /// Register-only mode (`max_return = 0`) records candidates and returns
    /// nothing, even when the set already holds a member long past its grace.
    ///
    /// This is the mode both *writers* use — the retention prune and the
    /// orphan reap — and returning anything here would hand a caller that only
    /// meant to record something a list of things to delete.
    #[tokio::test]
    async fn register_only_mode_records_without_returning_anything() {
        let tv = TestValkey::setup().await;
        let k = key(&tv);

        // A pre-existing member at the epoch: unambiguously expired.
        let _: () = tv
            .client()
            .pool
            .zadd(&k, None, None, false, false, (0.0, "ancient"))
            .await
            .expect("seeding must succeed");

        let returned = recon_claim_expired(
            tv.client(),
            &k,
            Duration::from_millis(1),
            0,
            &["fresh".to_string()],
        )
        .await
        .expect("register-only claim must succeed");

        assert!(
            returned.is_empty(),
            "max_return = 0 means register-only; got {returned:?}"
        );
        let mut got = members(&tv, &k).await;
        got.sort();
        assert_eq!(got, vec!["ancient".to_string(), "fresh".to_string()]);
    }

    /// `ZADD NX` pins the first observation: re-registering an existing member
    /// must NOT move its score.
    ///
    /// The load-bearing property. With an `XX`/plain `ZADD` every pass would
    /// refresh the score, nothing would ever cross the expiry cutoff, and the
    /// sweep would enumerate the whole namespace every hour and reclaim
    /// nothing — with no error anywhere.
    #[tokio::test]
    async fn re_registering_does_not_refresh_the_first_observed_score() {
        let tv = TestValkey::setup().await;
        let k = key(&tv);

        recon_claim_expired(tv.client(), &k, Duration::ZERO, 0, &["m".to_string()])
            .await
            .expect("first registration must succeed");
        let first = score(&tv, &k, "m")
            .await
            .expect("member must carry a score");

        tokio::time::sleep(Duration::from_millis(50)).await;
        recon_claim_expired(tv.client(), &k, Duration::ZERO, 0, &["m".to_string()])
            .await
            .expect("second registration must succeed");

        assert_eq!(
            score(&tv, &k, "m").await,
            Some(first),
            "ZADD NX must pin the FIRST observation — a refreshed score makes the grace \
             window unreachable and the sweep a silent no-op"
        );
    }

    /// The score is a Valkey-side millisecond timestamp, not a caller clock
    /// and not a seconds value.
    ///
    /// A seconds-resolution score would compare against a millisecond grace
    /// and expire everything instantly; a zero score would do the same. Both
    /// failure modes are "reap a publish that is still in flight".
    #[tokio::test]
    async fn registration_score_is_a_valkey_side_millisecond_timestamp() {
        let tv = TestValkey::setup().await;
        let k = key(&tv);

        recon_claim_expired(tv.client(), &k, Duration::ZERO, 0, &["m".to_string()])
            .await
            .expect("registration must succeed");

        let s = score(&tv, &k, "m")
            .await
            .expect("member must carry a score");
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock must be past the epoch")
            .as_millis() as f64;
        assert!(
            (s - now_ms).abs() < 60_000.0,
            "score must be a millisecond epoch timestamp sampled by Valkey; got {s} against a \
             local now of {now_ms}. A seconds-resolution or zero score expires every candidate \
             on its first sighting, which is exactly the reap-in-flight failure the grace \
             exists to prevent"
        );
    }

    /// A member registered by a call cannot be claimed by that same call, for
    /// any positive grace — the register and the claim share one `TIME`
    /// sample, so `now_ms > now_ms - grace_ms`.
    #[tokio::test]
    async fn a_member_registered_now_is_never_expired_now() {
        let tv = TestValkey::setup().await;
        let k = key(&tv);

        let returned = recon_claim_expired(
            tv.client(),
            &k,
            Duration::from_secs(60),
            100,
            &["m".to_string()],
        )
        .await
        .expect("claim must succeed");

        assert!(
            returned.is_empty(),
            "a candidate must be observed twice, a grace apart, before it can be acted on; \
             got {returned:?}"
        );
        assert_eq!(members(&tv, &k).await, vec!["m".to_string()]);
    }

    /// Expired members come back oldest-first and bounded by `max_return`; the
    /// remainder simply waits for the next pass.
    ///
    /// The bound is what stops a store that has accumulated candidates for
    /// months from turning one pass into an unbounded delete storm while
    /// holding the checkpoint leader lease.
    #[tokio::test]
    async fn expired_members_return_oldest_first_and_respect_the_limit() {
        let tv = TestValkey::setup().await;
        let k = key(&tv);

        // Ancient scores (ms since the epoch) so any positive grace puts them
        // past the cutoff regardless of when the test runs.
        for (score_ms, member) in [(3_000.0, "third"), (1_000.0, "first"), (2_000.0, "second")] {
            let _: () = tv
                .client()
                .pool
                .zadd(&k, None, None, false, false, (score_ms, member))
                .await
                .expect("seeding must succeed");
        }

        let claimed = recon_claim_expired(tv.client(), &k, Duration::from_millis(1), 2, &[])
            .await
            .expect("claim must succeed");
        assert_eq!(
            claimed,
            vec!["first".to_string(), "second".to_string()],
            "oldest first, at most max_return of them"
        );

        let all = recon_claim_expired(tv.client(), &k, Duration::from_millis(1), 100, &[])
            .await
            .expect("claim must succeed");
        assert_eq!(
            all,
            vec![
                "first".to_string(),
                "second".to_string(),
                "third".to_string()
            ],
            "the script must NOT remove what it returns — a member left behind by a failed \
             action has to stay claimable next pass, which is the protocol's retry behaviour"
        );
    }

    /// [`recon_retire`] is the caller's acknowledgement; it is a no-op on an
    /// empty slice and idempotent on a member that is already gone.
    #[tokio::test]
    async fn retire_removes_members_and_is_idempotent_and_empty_safe() {
        let tv = TestValkey::setup().await;
        let k = key(&tv);

        recon_claim_expired(
            tv.client(),
            &k,
            Duration::ZERO,
            0,
            &["a".to_string(), "b".to_string()],
        )
        .await
        .expect("registration must succeed");

        recon_retire(tv.client(), &k, &[])
            .await
            .expect("empty retire must be a no-op, not an error");
        assert_eq!(members(&tv, &k).await.len(), 2);

        recon_retire(tv.client(), &k, &["a".to_string()])
            .await
            .expect("retire must succeed");
        assert_eq!(members(&tv, &k).await, vec!["b".to_string()]);

        recon_retire(tv.client(), &k, &["a".to_string()])
            .await
            .expect("retiring an already-absent member must succeed: two replicas acting on the same member is idempotent by design");
        assert_eq!(members(&tv, &k).await, vec!["b".to_string()]);
    }

    /// [`recon_backlog`] is the gauge's input: the live `ZCARD`, and `0` on a
    /// key that does not exist yet.
    #[tokio::test]
    async fn backlog_reports_zcard_and_zero_on_an_absent_key() {
        let tv = TestValkey::setup().await;
        let k = key(&tv);

        assert_eq!(
            recon_backlog(tv.client(), &k)
                .await
                .expect("backlog on an absent key must succeed"),
            0
        );

        recon_claim_expired(
            tv.client(),
            &k,
            Duration::ZERO,
            0,
            &["a".to_string(), "b".to_string(), "c".to_string()],
        )
        .await
        .expect("registration must succeed");

        assert_eq!(
            recon_backlog(tv.client(), &k)
                .await
                .expect("backlog must succeed"),
            3
        );
    }
}

/// The SR-3.2 ext-value recovery policy both stores share through
/// [`recover_ext_value_via_checkpoint`].
///
/// There is ONE policy body, so a mock-only suite would be proof
/// about the policy but not that `ValkeyExternalValueStore` still reaches it
/// with its own rewrite. Every adapter outcome is therefore driven through BOTH
/// stores: the mock, and the real store against Valkey — including the real
/// store's `ref_count` effect of the recovered rewrite, which the mock's
/// in-memory refcount cannot stand in for.
#[cfg(test)]
mod ext_recovery_policy_tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use fred::interfaces::{HashesInterface, KeysInterface};

    use super::*;
    use crate::checkpoint::CheckpointError;
    use crate::observability::metrics::test_util::{render, sample_value};
    use crate::observability::metrics::{metrics_scrape_only, Metrics};
    use crate::storage::test_support::TestValkey;

    /// What the scripted adapter's `get_ext_blob` answers.
    #[derive(Clone, Copy, Debug)]
    enum Adapter {
        /// `Ok(Some(RECOVERED_BYTES))`.
        Hit,
        /// `Ok(None)`.
        Miss,
        Transient,
        Permanent,
        /// A variant `get_ext_blob`'s contract never returns.
        Other,
    }

    const RECOVERED_BYTES: &[u8] = b"b21r-recovered-ext-value";

    /// The five `result` labels, in [`Adapter`] order.
    const LABELS: [&str; 5] = [
        "recovered",
        "adapter_miss",
        "adapter_transient",
        "adapter_permanent",
        "adapter_other",
    ];

    fn label(adapter: Adapter) -> &'static str {
        match adapter {
            Adapter::Hit => "recovered",
            Adapter::Miss => "adapter_miss",
            Adapter::Transient => "adapter_transient",
            Adapter::Permanent => "adapter_permanent",
            Adapter::Other => "adapter_other",
        }
    }

    /// A `CheckpointStore` whose only working method is `get_ext_blob`, which
    /// answers `self.0`. Every other method is `NotImplemented`, so a recovery
    /// path that touched anything else would fail loudly.
    struct ScriptedExtBlobStore(Adapter);

    #[async_trait::async_trait]
    impl crate::checkpoint::CheckpointStore for ScriptedExtBlobStore {
        async fn put_layer_checkpoint(
            &self,
            _object_id: u64,
            _snap_seq: u64,
            _ompf: bytes::Bytes,
            _meta: crate::checkpoint::CheckpointMeta,
        ) -> Result<crate::checkpoint::CheckpointHandle, CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "put_layer_checkpoint",
            })
        }
        async fn get_latest_layer_checkpoint(
            &self,
            _object_id: u64,
        ) -> Result<Option<crate::checkpoint::CheckpointBlob>, CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "get_latest_layer_checkpoint",
            })
        }
        async fn list_layer_checkpoints(
            &self,
            _object_id: u64,
        ) -> Result<Vec<crate::checkpoint::CheckpointHandle>, CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "list_layer_checkpoints",
            })
        }
        async fn delete_layer_checkpoint(
            &self,
            _handle: crate::checkpoint::CheckpointHandle,
        ) -> Result<(), CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "delete_layer_checkpoint",
            })
        }
        async fn put_session_checkpoint(
            &self,
            _session_id: &str,
            _body: bytes::Bytes,
            _meta: crate::checkpoint::CheckpointMeta,
        ) -> Result<crate::checkpoint::CheckpointHandle, CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "put_session_checkpoint",
            })
        }
        async fn get_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<Option<crate::checkpoint::CheckpointBlob>, CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "get_session_checkpoint",
            })
        }
        async fn delete_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<(), CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "delete_session_checkpoint",
            })
        }
        async fn put_ext_blob(
            &self,
            _hash_hex: &str,
            _bytes: bytes::Bytes,
        ) -> Result<(), CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "put_ext_blob",
            })
        }
        async fn get_ext_blob(
            &self,
            _hash_hex: &str,
        ) -> Result<Option<bytes::Bytes>, CheckpointError> {
            match self.0 {
                Adapter::Hit => Ok(Some(bytes::Bytes::from_static(RECOVERED_BYTES))),
                Adapter::Miss => Ok(None),
                Adapter::Transient => Err(CheckpointError::Transient {
                    source: "scripted transient".into(),
                }),
                Adapter::Permanent => Err(CheckpointError::Permanent {
                    source: "scripted permanent".into(),
                }),
                Adapter::Other => Err(CheckpointError::NotImplemented {
                    method: "get_ext_blob",
                }),
            }
        }
        async fn delete_ext_blob(&self, _hash_hex: &str) -> Result<(), CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "delete_ext_blob",
            })
        }
        async fn put_counters_sidecar(&self, _body: bytes::Bytes) -> Result<(), CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "put_counters_sidecar",
            })
        }
        async fn get_counters_sidecar(&self) -> Result<Option<bytes::Bytes>, CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "get_counters_sidecar",
            })
        }
        async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
            Err(CheckpointError::NotImplemented {
                method: "list_all_object_ids",
            })
        }
    }

    fn scripted(adapter: Adapter) -> Arc<dyn crate::checkpoint::CheckpointStore> {
        Arc::new(ScriptedExtBlobStore(adapter))
    }

    fn test_hash(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    /// Every `live_edit_checkpoint_ext_recovery_total{result}` cell, in
    /// [`LABELS`] order. All five are seeded, so absence panics.
    fn recovery_cells(metrics: &Metrics) -> Vec<(&'static str, u64)> {
        let text = render(metrics);
        LABELS
            .iter()
            .map(|result| {
                let v = sample_value(
                    &text,
                    "live_edit_checkpoint_ext_recovery_total",
                    &[("result", result)],
                )
                .expect("every ext-recovery result is seeded in Metrics::new");
                (*result, v as u64)
            })
            .collect()
    }

    fn only_cell(result: &str) -> Vec<(&'static str, u64)> {
        LABELS
            .iter()
            .map(|r| (*r, u64::from(*r == result)))
            .collect()
    }

    /// The contract for one adapter outcome, checked against what a store's
    /// `read_batch_with_recovery_counter` returned for a hash it had no data
    /// for: `original` is the error the store would have surfaced with no
    /// adapter wired.
    fn assert_outcome(
        store: &str,
        adapter: Adapter,
        hash: &[u8; 32],
        result: Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError>,
        original: &ServiceError,
        recovery_counter: &AtomicUsize,
        metrics: &Metrics,
    ) {
        match adapter {
            Adapter::Hit => {
                let map = result.unwrap_or_else(|e| {
                    panic!("{store}/{adapter:?}: a recovered read must succeed, got {e:?}")
                });
                assert_eq!(
                    map.get(hash).map(Vec::as_slice),
                    Some(RECOVERED_BYTES),
                    "{store}/{adapter:?}: the recovered bytes are returned"
                );
            }
            Adapter::Transient => match result {
                Err(ServiceError::Unavailable(_)) => {}
                other => panic!(
                    "{store}/{adapter:?}: an adapter Transient must surface UNAVAILABLE so the \
                     client retries; got {other:?}"
                ),
            },
            Adapter::Miss | Adapter::Permanent | Adapter::Other => {
                let err = result.expect_err("a non-recovering outcome must fail the read");
                assert_eq!(
                    format!("{err:?}"),
                    format!("{original:?}"),
                    "{store}/{adapter:?}: the ORIGINAL miss error surfaces unchanged"
                );
            }
        }
        assert_eq!(
            recovery_counter.load(Ordering::Relaxed),
            usize::from(matches!(adapter, Adapter::Hit)),
            "{store}/{adapter:?}: the caller's recovery counter moves on success only"
        );
        assert_eq!(
            recovery_cells(metrics),
            only_cell(label(adapter)),
            "{store}/{adapter:?}: exactly one `{}` tick and nothing else",
            label(adapter)
        );
    }

    const ADAPTERS: [Adapter; 5] = [
        Adapter::Hit,
        Adapter::Miss,
        Adapter::Transient,
        Adapter::Permanent,
        Adapter::Other,
    ];

    /// The mock store, every adapter outcome: result, error mapping, recovery
    /// counter, and exactly one tick on the outcome's `result` label.
    #[tokio::test]
    async fn mock_store_recovery_maps_every_adapter_outcome_and_ticks_its_one_label() {
        for adapter in ADAPTERS {
            let metrics = Arc::new(metrics_scrape_only("b21r-mock"));
            let mock = MockExternalValueStore::new()
                .with_checkpoint_store(scripted(adapter))
                .with_metrics(Arc::clone(&metrics));
            let hash = test_hash(0x51);
            let original = ServiceError::from(ExternalValueError::MetaNotFound {
                hash_hex: sha256_to_hex(&hash),
            });
            let counter = AtomicUsize::new(0);

            let result = mock
                .read_batch_with_recovery_counter(&[hash], Some(&counter))
                .await;
            assert_outcome(
                "mock", adapter, &hash, result, &original, &counter, &metrics,
            );
            if matches!(adapter, Adapter::Hit) {
                assert_eq!(
                    mock.ref_count(&hash),
                    Some(1),
                    "mock/Hit: the recovered bytes were rewritten through store_recovered_values"
                );
            }
        }
    }

    /// With no adapter wired, a miss surfaces its original error and no
    /// `result` cell moves: the counter counts adapter consultations only.
    #[tokio::test]
    async fn mock_store_without_an_adapter_surfaces_the_miss_and_ticks_nothing() {
        let metrics = Arc::new(metrics_scrape_only("b21r-mock-no-adapter"));
        let mock = MockExternalValueStore::new().with_metrics(Arc::clone(&metrics));
        let hash = test_hash(0x52);
        let original = ServiceError::from(ExternalValueError::MetaNotFound {
            hash_hex: sha256_to_hex(&hash),
        });

        let err = mock
            .read_batch(&[hash])
            .await
            .expect_err("a miss with no adapter must fail the read");
        assert_eq!(format!("{err:?}"), format!("{original:?}"));
        assert_eq!(
            recovery_cells(&metrics),
            LABELS.iter().map(|r| (*r, 0)).collect::<Vec<_>>()
        );
    }

    /// The real store, every adapter outcome, against Valkey: the same
    /// contract as the mock's, reached through `ValkeyExternalValueStore`'s own
    /// `read_batch` and its own rewrite. The hash was never stored, so its
    /// miss is `MetaNotFound`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn valkey_store_recovery_maps_every_adapter_outcome_and_ticks_its_one_label() {
        for adapter in ADAPTERS {
            let tv = TestValkey::setup().await;
            let metrics = Arc::new(metrics_scrape_only("b21r-valkey"));
            let store = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
                .with_checkpoint_store(scripted(adapter))
                .with_metrics(Arc::clone(&metrics));
            let hash = test_hash(0x61);
            let original = ServiceError::from(ExternalValueError::MetaNotFound {
                hash_hex: sha256_to_hex(&hash),
            });
            let counter = AtomicUsize::new(0);

            let result = store
                .read_batch_with_recovery_counter(&[hash], Some(&counter))
                .await;
            assert_outcome(
                "valkey", adapter, &hash, result, &original, &counter, &metrics,
            );
        }
    }

    /// A recovered rewrite on the real store goes through
    /// `store_recovered_values`, whose per-hash script (the one `store_values`
    /// runs) increments `ref_count` — so recovering a hash whose
    /// `ext_meta` survived (only `ext_data` was lost) leaves `ref_count` one
    /// higher than before. That `+1` is the known, reconciler-drained leak the
    /// recovery path documents; this pins that it is exactly one, and that the
    /// data is back.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn valkey_store_recovered_rewrite_adds_exactly_one_ref_count() {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only("b21r-valkey-refcount"));
        let store = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_checkpoint_store(scripted(Adapter::Hit))
            .with_metrics(Arc::clone(&metrics));
        let hash = test_hash(0x62);
        let hash_hex = sha256_to_hex(&hash);
        store
            .store_values(&[(&hash, RECOVERED_BYTES)])
            .await
            .expect("store_values must succeed for setup");
        let ref_count = |tv: &TestValkey| {
            let key = tv.key_schema().ext_meta(&hash_hex);
            let pool = tv.client().pool.clone();
            async move {
                let v: Option<String> = pool
                    .hget(&key, "ref_count")
                    .await
                    .expect("HGET ref_count must succeed");
                v.map(|s| s.parse::<i64>().expect("ref_count is numeric"))
            }
        };
        assert_eq!(ref_count(&tv).await, Some(1), "precondition: one reference");

        let _: () = tv
            .client()
            .pool
            .del(tv.key_schema().ext_data(&hash_hex))
            .await
            .expect("DEL ext_data must succeed");
        let map = store
            .read_batch(&[hash])
            .await
            .expect("the DataNotFound miss must recover");
        assert_eq!(map.get(&hash).map(Vec::as_slice), Some(RECOVERED_BYTES));

        assert_eq!(
            ref_count(&tv).await,
            Some(2),
            "the recovered rewrite increments ref_count exactly once"
        );
        assert_eq!(recovery_cells(&metrics), only_cell("recovered"));
        let data: Option<Vec<u8>> = tv
            .client()
            .pool
            .get(tv.key_schema().ext_data(&hash_hex))
            .await
            .expect("GET ext_data must succeed");
        assert_eq!(
            data.as_deref(),
            Some(RECOVERED_BYTES),
            "the rewrite restored ext_data, so the next read is a Valkey hit"
        );
    }

    /// `checkpoint_ext_mirror_bytes{state}` as `(queued, in_flight)`; both
    /// states are pre-registered, so an absent cell is a failure, not 0.
    fn mirror_bytes(metrics: &Metrics) -> (i64, i64) {
        let text = render(metrics);
        let state = |state: &str| {
            sample_value(
                &text,
                "live_edit_checkpoint_ext_mirror_bytes",
                &[("state", state)],
            )
            .unwrap_or_else(|| panic!("checkpoint_ext_mirror_bytes{{state={state:?}}} is seeded"))
                as i64
        };
        (state("queued"), state("in_flight"))
    }

    /// `checkpoint_ext_mirror_dropped_total` summed over every `reason`
    /// (absent → 0): any drop moves it.
    fn mirror_dropped(metrics: &Metrics) -> u64 {
        crate::observability::metrics::test_util::sample_sum(
            &render(metrics),
            "live_edit_checkpoint_ext_mirror_dropped_total",
            &[],
        )
        .unwrap_or(0.0) as u64
    }

    /// Fetch-on-miss recovery re-stores bytes it just read from the backup
    /// store, so its rewrite (`ExtRecoveryRewrite::store_recovered_values`)
    /// sends the ext mirror no copy and takes no `M_ext` charge (SDD §22.17):
    /// with a mirror sender wired on the real store, a recovered read leaves
    /// the mirror's channel empty, its byte budget whole, its `queued` and
    /// `in_flight` bytes at zero, and no drop counted. The contrast, an
    /// ordinary `store_values` on the same store, sends a charged copy — so
    /// the empty channel is the rewrite sending nothing, not the harness
    /// failing to see a copy.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn valkey_store_recovered_rewrite_sends_no_mirror_copy_and_takes_no_charge() {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only("mhq-recovery-no-mirror"));
        // No worker drains the receiver, so a copy sent stays queued here,
        // charged, where the assertions below can see it.
        let (sender, mut receiver) = crate::checkpoint::ext_mirror_channel(
            8,
            std::num::NonZeroU64::new(1 << 20).expect("non-zero"),
            Arc::clone(&metrics),
        );
        let store = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_checkpoint_store(scripted(Adapter::Hit))
            .with_metrics(Arc::clone(&metrics))
            .with_ext_mirror_sender(sender.clone());
        let charge =
            crate::checkpoint::ext_mirror::item_charge_kib(RECOVERED_BYTES.len() as u64) * 1024;

        // A hash never stored: its miss recovers through the adapter.
        let recovered = test_hash(0x63);
        let map = store
            .read_batch(&[recovered])
            .await
            .expect("the MetaNotFound miss must recover");
        assert_eq!(
            map.get(&recovered).map(Vec::as_slice),
            Some(RECOVERED_BYTES),
            "fixture self-check: the read recovered the adapter's bytes"
        );
        assert_eq!(
            recovery_cells(&metrics),
            only_cell("recovered"),
            "fixture self-check: the read went through the recovery rewrite"
        );
        assert!(
            matches!(
                receiver.try_recv(),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty)
            ),
            "the recovery rewrite sent the ext mirror a copy of bytes the backup store holds"
        );
        assert_eq!(
            sender.available_budget_bytes(),
            sender.budget_bytes(),
            "the recovery rewrite took an M_ext charge"
        );
        assert_eq!(
            mirror_bytes(&metrics),
            (0, 0),
            "no mirror bytes are queued or in flight after a recovery"
        );
        assert_eq!(
            mirror_dropped(&metrics),
            0,
            "no copy was offered, so none was dropped"
        );

        // The contrast: an ordinary store on the same store sends one copy,
        // charged.
        let stored = test_hash(0x64);
        store
            .store_values(&[(&stored, RECOVERED_BYTES)])
            .await
            .expect("store_values must succeed");
        assert_eq!(
            sender.budget_bytes() - sender.available_budget_bytes(),
            charge,
            "store_values took one value's charge"
        );
        assert_eq!(
            mirror_bytes(&metrics),
            (charge as i64, 0),
            "store_values' copy is queued"
        );
        let item = receiver
            .try_recv()
            .expect("store_values sends the ext mirror a copy");
        assert_eq!(
            (item.hash_hex(), item.bytes().as_ref()),
            (sha256_to_hex(&stored).as_str(), RECOVERED_BYTES),
            "the copy is the stored value's"
        );
        assert!(
            matches!(
                receiver.try_recv(),
                Err(tokio::sync::mpsc::error::TryRecvError::Empty)
            ),
            "one store, one copy"
        );
        assert_eq!(mirror_dropped(&metrics), 0, "nothing was dropped");
    }
}
