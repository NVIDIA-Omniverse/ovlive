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

//! Pre-write residue wipe for the DR restore path (b.f4t sub-task 1).
//!
//! # What this exists to fix
//!
//! `apply_restore` materializes a restored [`ObjectState`](crate::domain::types::ObjectState)
//! through [`LayerStore::write_restored_object`](crate::storage::layer::LayerStore::write_restored_object),
//! which queues the same body `CreateLayer`'s materializer does
//! ([`queue_object_state`](crate::storage::layer)). That body's preconditions
//! assume a *freshly minted* `object_id` that owns nothing: it `SADD`s /
//! `ZADD`s the restored members into `obj:secs` / `obj:flds` / `obj:tss` rather
//! than replacing those indexes, it writes no tombstones at all, and (since
//! b.zvd) it `DEL`s a children list only for the parents the restored state
//! actually carries.
//!
//! On the restore path that assumption does not hold. The DR entry point runs
//! at a KNOWN `object_id`, deliberately over a keyspace that may have partially
//! survived — the b.kf1 orphan class is exactly "header gone, everything else
//! alive". Written into such a keyspace, that body **merges**: stale index
//! members union with the restored set, their per-record hashes read back as
//! live content the checkpoint never contained, a surviving children list under
//! a parent the restore does not carry survives intact (and its `obj:chs`
//! member with it), and a surviving `obj:dels` resurrects deletions the
//! checkpoint does not carry.
//!
//! Policy (ratified, b.f4t): **restore is an authoritative REPLACE.** After
//! `apply_restore` returns `Applied`, the object's index-reachable `obj:*`
//! keyspace equals exactly the restored state. This module is how that is
//! reached — a conditional pre-write wipe of the object's surviving key
//! families, run from `apply_restore` after the SR-2.4 identity guard and
//! before the SR-3.4 ext pre-store.
//!
//! # Why the wipe lives here and not in the materializer
//!
//! The create path must NEVER wipe. A freshly minted `object_id` that owns
//! anything is an anomaly — precisely the anomaly the b.ph9 stream `EXISTS`
//! guard and the b.zvd keyspace type probe exist to *report* — and turning that
//! into a silent delete would destroy the evidence. That argument survived
//! b.h4m splitting the two entry points apart: the wipe could now sit inside
//! `write_restored_object` without touching the create path at all, and it
//! still does not, because folding it in would mean one transaction that both
//! deletes and rebuilds, which is a strictly larger blast radius for a step
//! whose whole safety argument is that it is separately idempotent. Keeping it
//! in the checkpoint lane also leaves `ProbeSite`'s "no third call site" fence
//! intact.
//!
//! # What the wipe deliberately does NOT do
//!
//! **It does not remove the per-object stream** (the shared key inventory
//! never carries it). The stream is owned by the b.4et adjudication
//! (`recovery::probe_stream_high_water` / `recovery::unlink_stale_stream`),
//! which has already run by the time this module is reached and whose
//! high-water read is the restore floor. Wiping it here would destroy that
//! evidence and the `STREAM_DELETE_GRACE_MS` window that lets lagging
//! subscribers still observe a `deleted` sentinel.
//!
//! **It does not remove the per-object external-value reference index**
//! (`obj:ext`), and it releases no external-value reference counts itself.
//! `obj:ext` is a member of the shared [`KeySchema::object_cleanup_keys`]
//! inventory (b.qpf), so this module removes it from the `UNLINK` list
//! explicitly — see step 4. That is not a nicety: `obj:ext` is the **pre-image**
//! `write_restored_object` reads (`HGETALL`, as `ext_before`) immediately
//! before it materializes the restored state, and the releases it
//! owes are exactly `before − after` over that index — the hashes the residue's
//! value rows held and the restored state does not, emitted as `rewrite`
//! releases under `SurplusPolicy::Suppress`. Wiping the index first would make
//! `ext_before` empty, so `before − after` would be empty too and every restore
//! over residue would leak a reference per dropped hash.
//!
//! Nor may this module release them itself. The residue's hashes are the set
//! the *old* rows held; the restored state generally holds most of them again,
//! so releasing all of them would **over**-release — an unmatched decrement
//! against a blob the restored object still references, and with GC behind it,
//! meta+data deleted under a live reference. `before − after`, computed by
//! `write_restored_object` from the `obj:ext` pre-image and enqueued once its
//! `EXEC` lands, is the only correct set, and that function is where it is
//! computed.
//!
//! The withholding has a **second** consumer since b.yda: the orphan sweep's
//! settle of a retirement intent's carrier (`scripts/ext_retire_settle.lua`,
//! b.yuo Part R) gates on `EXISTS obj:hdr OR EXISTS obj:ext`, and the
//! `obj:ext` leg is what covers the wipe→materialize window this module opens.
//! See the `retain` at step 4 for why removing the index would turn a settle
//! landing in that window into a release the restored object still needs.
//!
//! `schedule_old_object_ext_gc` is likewise not an option here, for an
//! independent reason: it calls `read_object_snapshot`, which answers
//! `NotFound` on an empty header (`src/storage/diff_read.rs`), i.e. a no-op
//! over exactly the header-less residue this path faces.
//!
//! # Exit paths
//!
//! | outcome | keyspace | caller |
//! |---|---|---|
//! | [`ResidueOutcome::Clean`] | untouched (no residue existed) | proceeds to the materialization |
//! | [`ResidueOutcome::Wiped`] | residue unlinked | proceeds to the materialization |
//! | [`ResidueOutcome::PeerLandedHeader`] | untouched | returns `ApplyOutcome::AlreadyPresent` |
//! | `Err(Permanent)` | untouched | refuses the restore; an operator must remove a wrong-typed key |
//! | `Err(Transient)` | possibly partially wiped | the recovery hook retries; see idempotence below |
//! | `Err(<as injected>)` | untouched (`WATCH` released, nothing dispatched) | the seam below refused; classification is the injector's |
//!
//! The last row is reachable only under `test-checkpoint-fault-injection`. The
//! seam that produces it,
//! [`RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`](crate::checkpoint::apply::RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC),
//! fires at the one position no other seam in the repo occupies: **inside** the
//! `WATCH` … `EXEC` window, after the post-`WATCH` header re-read has passed the
//! structural predicate and after the five enumerations have completed,
//! immediately before the `EXEC` is dispatched. That position is what makes the
//! tripped-`EXEC` arm's two verdicts reachable on purpose rather than by timing:
//! an injector that lands a structurally-complete `obj:hdr` from there and
//! returns `Ok(())` drives `PeerLandedHeader` (⇒ `ApplyOutcome::AlreadyPresent`,
//! nothing wiped), one that lands a single field — a racing `CreateLayer`
//! overwrite's `HSET superseded=1`, most concretely — drives `Err(Transient)`
//! (⇒ the wipe is re-driven), and one that returns `Err` drives the row above.
//! Production compiles the field, the builder, and the check out entirely.
//!
//! `PeerLandedHeader` is claimed only on POSITIVE evidence — a structurally
//! complete `obj:hdr`, read either before the `WATCH` armed or after the `EXEC`
//! tripped on it. A tripped `EXEC` alone does not qualify: a single-field
//! `HSET` (a racing `CreateLayer` overwrite's `superseded=1`, say) trips it too,
//! and reporting `AlreadyPresent` there would tick
//! `checkpoint_recovery_attempts_total{result="restored"}` for a restore that
//! never ran.
//!
//! `Err(Transient)` additionally covers the two watched-transaction guard arms
//! this site shares with `create_session` and the structural `update_layer` slow
//! path: an L2 pre-dispatch `Refuse` (the connection carrying the `WATCH` was
//! replaced, so the fence would not have held) aborts before anything reaches
//! the wire, and the wipe is re-driven. Those arms, plus the L3 witness and its
//! `witness_unavailable` / `fingerprint_unavailable` / `committed_unwatched`
//! dispositions, report on `storage_watch_voided_total{path="restore_residue_wipe"}`.
//!
//! **Idempotent under cancel / abort.** Every arm that mutates runs while the
//! object is header-less (or header-incomplete), so a partially-wiped residue
//! is indistinguishable from the pre-wipe state: the next attempt re-probes,
//! finds whatever is left, and converges. Nothing here is ordered against
//! anything a later attempt reads.

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface};
use fred::types::ClusterHash;
use std::collections::HashMap;

#[cfg(feature = "test-checkpoint-fault-injection")]
use crate::checkpoint::adapters::CheckpointFaultInjector;
#[cfg(feature = "test-checkpoint-fault-injection")]
use crate::checkpoint::apply::RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC;
use crate::checkpoint::apply::{
    classify_object_key_read_error, header_fields_are_structurally_complete,
};
use crate::checkpoint::store::CheckpointError;
use crate::config::Config;
use crate::observability::metrics::{
    WATCH_VOIDED_ABORTED_PRE_EXEC, WATCH_VOIDED_COMMITTED_UNWATCHED,
    WATCH_VOIDED_FINGERPRINT_UNAVAILABLE, WATCH_VOIDED_PATH_RESTORE_RESIDUE_WIPE,
    WATCH_VOIDED_WITNESS_UNAVAILABLE,
};
use crate::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_DATA, POOL_LABEL_TXN};
use crate::storage::keys::{KeySchema, ROOT_PARENT_ID};
use crate::storage::layer::{LayerStore, UNLINK_CHUNK_SIZE};
use crate::storage::watch_guard::WatchGuard;
use crate::storage::watched_txn::{
    dispatch_verdict, watch_with_baseline, DispatchVerdict, WatchFingerprint, WatchedExec,
    WatchedTxn,
};

/// Everything [`wipe_restore_residue`] needs beyond the object it is wiping:
/// the [`Config`] whose service label the counter is keyed on, and — only under
/// `test-checkpoint-fault-injection` — the injector whose seam fires inside the
/// `WATCH` … `EXEC` window.
///
/// A struct rather than a second positional parameter *because* the injector
/// exists only under that feature. Bundling it here keeps
/// [`wipe_restore_residue`]'s arity identical in every feature configuration —
/// a cfg-gated parameter would make the same call site compile in one shape and
/// not the other — and the `From<&Config>` below lets a caller with no seam to
/// install pass the `&Config` it already holds, unchanged.
#[derive(Clone, Copy)]
pub(crate) struct ResidueWipeCtx<'a> {
    config: &'a Config,
    /// Fires once, between the post-`WATCH` header re-read and the `EXEC`. See
    /// [`crate::checkpoint::apply::RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`] for
    /// the two verdicts it exists to make reachable.
    #[cfg(feature = "test-checkpoint-fault-injection")]
    injector: Option<&'a dyn CheckpointFaultInjector>,
}

impl<'a> From<&'a Config> for ResidueWipeCtx<'a> {
    fn from(config: &'a Config) -> Self {
        Self {
            config,
            #[cfg(feature = "test-checkpoint-fault-injection")]
            injector: None,
        }
    }
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl<'a> ResidueWipeCtx<'a> {
    /// Install the
    /// [`RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`](crate::checkpoint::apply::RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC)
    /// seam.
    ///
    /// Takes an `Option` rather than a bare reference so the one production
    /// call site can hand over whatever `ExtPrestore` is carrying without a
    /// second `match`.
    ///
    /// Gated on `test-checkpoint-fault-injection` **exactly** — matching
    /// [`CheckpointFaultInjector`]'s own cfg, as `ExtPrestore::with_fault_injector`
    /// does — so the field, this builder, and the seam-check site all compile
    /// out of production builds entirely.
    pub(crate) fn with_fault_injector(
        mut self,
        injector: Option<&'a dyn CheckpointFaultInjector>,
    ) -> Self {
        self.injector = injector;
        self
    }
}

/// What [`wipe_restore_residue`] found and did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidueOutcome {
    /// No object-level residue key existed. Nothing was enumerated, nothing was
    /// watched, nothing was unlinked, and the counter did not tick. This is the
    /// whole-keyspace-loss path — the overwhelmingly common DR case — and it
    /// costs exactly one `EXISTS` round trip.
    Clean,
    /// Residue existed and was unlinked under the header `WATCH`.
    Wiped {
        /// Size of the [`KeySchema::object_cleanup_keys`] inventory that was
        /// sent to `UNLINK`, **less the entry this path withholds**: the
        /// per-object external-value reference index `obj:ext`, which is
        /// retained as `write_restored_object`'s ext pre-image (see the module
        /// docs and step 4). The inventory never carries the per-object stream.
        /// Not the number of keys that actually existed — `UNLINK` on an absent
        /// key is a no-op and the reply count is not read back, for the same
        /// reason the two existing retirement paths do not read it.
        keys_unlinked: usize,
    },
    /// A peer replica landed a structurally-complete header while we held the
    /// `WATCH`. Nothing was wiped, and the caller must report `AlreadyPresent`
    /// rather than writing over the peer's restore.
    ///
    /// Claimed on positive evidence only — a structurally-complete `obj:hdr`
    /// observed either by the post-`WATCH` re-read or by the re-read that
    /// follows a tripped `EXEC`. A tripped `EXEC` on its own is NOT evidence: it
    /// fires on any mutation of the key, and the wipe answers that with
    /// `Transient` instead.
    PeerLandedHeader,
}

/// Wipe the surviving `obj:*` residue for `object_id` so the restore that
/// follows is a REPLACE rather than a merge (b.f4t).
///
/// Called from `apply_restore` after the SR-2.4 identity guard has proved the
/// live mapping still resolves to this `object_id` — that proof is what makes
/// the keys *ours* to delete — and after the payload has decoded, so an
/// undecodable blob refuses before anything is removed.
///
/// # Algorithm
///
/// 1. **One-round-trip residue probe.** `EXISTS` over the eleven object-level
///    keys (`obj:hdr`, `obj:secs`, `obj:flds`, `obj:tss`, `obj:dels`,
///    `obj:chs`, `obj:ch:0`, `obj:wtok`, `obj:wrel`, `obj:snapfence`,
///    `obj:snapintent`) in a single command. All eleven carry the
///    `{object_id}` hash tag, so the multi-key form is slot-safe on Cluster. A
///    reply of `0` is [`ResidueOutcome::Clean`] and the function returns without
///    a `WATCH`, an `UNLINK`, or a counter tick. `obj:hdr` is in the set because
///    a structurally-incomplete header that outlived every index is residue too
///    — see the probe's own comment for the two stale fields that survive
///    `write_restored_object`'s HSET otherwise.
/// 2. **`WATCH obj:hdr` on a slot-pinned `txn_pool` connection**, then re-read
///    the header **on that same connection** and require it to be *not*
///    structurally complete. See the fencing note below. The L3 witness
///    baseline `watch_with_baseline` returns is reported when unavailable, and
///    the L2 connection fingerprint is taken here for the pre-dispatch check in
///    step 4.
/// 3. **Enumerate** the five indexes on the data pool. A `WRONGTYPE` here is
///    [`CheckpointError::Permanent`], never `Transient` — see below. Every
///    other error is `Transient`; nothing is ever `unwrap_or_default`ed (the
///    b.u96 argument: an empty Vec substituted for a failed enumeration drops
///    per-record keys off the inventory and strands them with no header left
///    for the orphan sweep to rediscover them from).
/// 4. **`UNLINK`** the [`KeySchema::object_cleanup_keys`] inventory — which
///    never carries the per-object stream — less `obj:ext`, which is retained
///    as `write_restored_object`'s ext pre-image (module docs), in
///    [`UNLINK_CHUNK_SIZE`] chunks through a
///    [`WatchedTxn`](crate::storage::watched_txn::WatchedTxn) on the watched
///    connection, gated immediately before the `EXEC` by the L2 pre-dispatch
///    identity check (`dispatch_verdict`): a `Refuse` there unwatches and
///    returns `Transient` rather than dispatching an unfenced delete. Under
///    `test-checkpoint-fault-injection` the
///    [`RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`](crate::checkpoint::apply::RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC)
///    seam fires between that check and the `EXEC` — see the module docs for
///    the two tripped-`EXEC` verdicts it makes reachable.
/// 5. **Observe.** WARN with the per-family breakdown; tick
///    `live_edit_checkpoint_restore_residue_wiped_total`, and
///    `storage_watch_voided_total{path="restore_residue_wipe"}` on each of the
///    watched-transaction dispositions.
///
/// # Why the `WATCH` (orchestrator refinement R1)
///
/// Before this wipe existed, the only cross-replica guard on the restore path
/// was `apply_restore`'s step-1 header predicate, and that predicate is a
/// point-in-time read: it closes the race only up to the instant it answers.
/// The `WATCH` below is what holds across a write. A `WATCH` on `obj:hdr` is
/// exactly the fence: a peer replica's restore lands its 12 canonical header
/// fields in one `HSET` inside its own MULTI/EXEC, and any live commit likewise
/// bumps `seq` on that key, so either one trips the `EXEC` and the wipe deletes
/// nothing.
///
/// The materialization that follows now holds an equivalent fence of its own
/// (b.h4m): `LayerStore::write_restored_object` `WATCH`es the same key and
/// requires it to be ABSENT, so a peer landing a complete header in the gap
/// between this `EXEC` and that one trips there instead and this replica writes
/// nothing. That closes the residual TOCTOU this paragraph used to name as a
/// storage-lane follow-up: the window between this `EXEC` and the
/// materialization's, in which the unfenced materializer would have `HSET` its
/// own generation over a peer's freshly landed one.
///
/// The trip says *something* wrote `obj:hdr`, not *what*, so the tripped arm
/// re-reads the key and reports [`ResidueOutcome::PeerLandedHeader`] only when
/// the header is structurally complete; anything else (a single-field `HSET`,
/// most concretely a racing `CreateLayer` overwrite's `superseded=1`) is
/// `Transient` and the idempotent wipe is re-driven.
///
/// The five index keys are NOT watched, and do not need to be: with no
/// structurally-complete header there is no writer that can legitimately mutate
/// them (both commit paths fail closed on an absent or partial header), so the
/// header is the only key whose change can invalidate the decision to wipe.
///
/// Because this is the only watched site whose committed effect is an
/// irreversible `UNLINK`, it runs the same three layers the three storage-lane
/// watched sites do — `update_layer_once_watch`, `create_once`, and (since b.h4m)
/// `write_restored_object` — L1 `WATCH`, L2 pre-dispatch identity check, L3
/// in-transaction `CLIENT ID` witness — and reports every voided disposition on
/// `storage_watch_voided_total{path="restore_residue_wipe"}`.
///
/// # Why a `WRONGTYPE` must be `Permanent`
///
/// This is defect (2) of the same ticket, re-created in a new place if it were
/// classified `Transient`: a wrong-typed `obj:secs` cannot be enumerated — nor a
/// wrong-typed `obj:hdr` `HGETALL`'d — by any retry, so the recovery hook would
/// re-drive a restore that cannot succeed, on every attempt, forever. Every read
/// on this path (both header re-reads and all five enumerations) therefore
/// routes through the one shared classifier,
/// [`classify_object_key_read_error`](crate::checkpoint::apply::classify_object_key_read_error),
/// whose refusal names the offending key so an operator can remove it — the
/// remediation shape `LayerStore::classify_keyspace_probe_error` uses.
pub(crate) async fn wipe_restore_residue<'a>(
    layer_store: &LayerStore,
    object_id: u64,
    ctx: impl Into<ResidueWipeCtx<'a>>,
) -> Result<ResidueOutcome, CheckpointError> {
    // `impl Into<..>`, so a caller with nothing to install passes the `&Config`
    // it already holds and a caller installing the WATCH…EXEC seam passes a
    // built [`ResidueWipeCtx`]. See that type for why the seam does not get a
    // parameter of its own.
    let ctx: ResidueWipeCtx<'a> = ctx.into();
    let config = ctx.config;
    let client = layer_store.client();
    let keys = layer_store.keys();
    let oid_str = object_id.to_string();

    // ── Step 1: the one-round-trip residue probe ──────────────────────────
    //
    // Eleven object-level keys, one `EXISTS`. The reply is the COUNT of those
    // that exist, so any non-zero value means residue survived under this id.
    // Every one carries the `{object_id}` hash tag, so this is a single-slot
    // command on Cluster.
    //
    // `obj:hdr` is one of the eleven, and that is not decoration: the
    // structurally-INCOMPLETE header with no surviving index key is a
    // first-class input class on this path (`apply_restore`'s step-1 predicate
    // now lets it through), and omitting the header from the probe would
    // short-circuit it as `Clean` and leave the partial header in place. It
    // must not be left in place, because `write_restored_object`'s single `HSET` stamps
    // the 12 canonical fields WITHOUT `DEL`ing the key first, so any
    // non-canonical field on the survivor outlives the restore underneath the
    // restored header — two of them with teeth:
    //
    //   * a stale `ext_gc_scheduled = "1"` makes the restored object's later
    //     ext-GC schedule a silent no-op, so its external-value refcounts are
    //     never released;
    //   * a stale `superseded = "1"` makes the restored layer permanently
    //     UNWRITABLE — both commit backends fail closed on it (the Lua fast
    //     path and the WATCH/MULTI/EXEC slow path read the same flag).
    //
    // Probing the header cannot cause a false wipe. `apply_restore`'s step 1
    // has already proved this header is not structurally complete, and a peer
    // replica landing a complete one in the gap is caught by the post-`WATCH`
    // re-read below, which wipes nothing and reports `PeerLandedHeader`.
    let header_key = keys.obj_header(&oid_str);
    let probe_keys: Vec<String> = vec![
        header_key.clone(),
        keys.obj_sections_index(&oid_str),
        keys.obj_fields_index(&oid_str),
        keys.obj_time_samples_index(&oid_str),
        keys.obj_deleted_sections_index(&oid_str),
        keys.obj_children_index(&oid_str),
        keys.obj_children(&oid_str, ROOT_PARENT_ID),
        // The `UpdateLayer` commit-token record. A survivor of it alone is
        // residue too: it names commits of the incarnation being replaced, so
        // an in-flight call whose earlier attempt landed before the loss —
        // or a client re-send carrying the operation id of such a commit —
        // would find its token or `op.` key on the restored object and report
        // a sequence the restore may not contain. The wipe below removes it
        // with the rest of the `object_cleanup_keys` inventory, and that call
        // or re-send then re-applies onto the restored state instead.
        keys.obj_write_token(&oid_str),
        // The `UpdateLayer` commit release ring, written in the same atomic
        // unit as the commit-token record. A survivor of it alone is residue
        // for the same reason: its entries are commits of the incarnation
        // being replaced, so a call still in flight across the restore could
        // find its own pre-loss entry — on an own-call verdict, or through
        // the lost-reply lookup — and enqueue releases for a commit the
        // restored object may not contain, an over-release that can delete a
        // still-referenced blob; an older build's per-`client_id` field could
        // likewise match a restored commit's sequence. The wipe below removes
        // it with the rest of the `object_cleanup_keys` inventory.
        keys.obj_write_releases(&oid_str),
        // The snapshot fence and the writer-intent key (Spec SDD SR-5.1). A
        // survivor of either alone is residue too: a fence's epoch, gap and
        // stored size belong to the incarnation being replaced, and a live
        // `r:*` or `w:*` would hold the restored layer's writers or readers to
        // an epoch or intent no live process owns until its lease or PEXPIRE
        // ran out. The wipe below removes both with the rest of the
        // `object_cleanup_keys` inventory.
        keys.obj_snapshot_fence(&oid_str),
        keys.obj_snapshot_intent(&oid_str),
    ];
    let present: u64 = client.pool.exists(probe_keys).await.map_err(|e| {
        FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(&e);
        CheckpointError::Transient {
            source: format!(
                "EXISTS residue probe failed during restore of object {object_id}: {e}"
            )
            .into(),
        }
    })?;
    if present == 0 {
        // The whole-keyspace-loss path. Nothing survived, so the restore is a
        // replace by construction and there is nothing to pay for it.
        return Ok(ResidueOutcome::Clean);
    }

    // ── Step 2: WATCH obj:hdr on a slot-pinned txn connection ─────────────
    //
    // The guard is what routes the teardown UNWATCH back to the node holding
    // the WATCH (b.ji8); `acquire_txn_for` pins the checkout hygiene probe to
    // the same slot (b.7td).
    let checkout = client.acquire_txn_for(&header_key).await;
    let mut conn = WatchGuard::new(checkout, client.watch_unwatch_tracker())
        .with_metrics(client.metrics_snapshot());

    let watch_hash = ClusterHash::from(header_key.as_str());
    // `watch_with_baseline`, not a bare `WATCH`: it is the only sanctioned way
    // to arm a `WatchedTxn`, and it costs no extra round trip. The baseline it
    // returns decides whether the transaction queues its L3 `CLIENT ID`
    // witness; on a store that refuses `CLIENT ID` inside `MULTI`, queueing one
    // would `EXECABORT` the whole wipe.
    let witness_baseline = watch_with_baseline(&conn, &watch_hash, &[header_key.as_str()])
        .await
        .map_err(|e| {
            FredErrOrigin::client(client, POOL_LABEL_TXN).observe_error(&e);
            CheckpointError::Transient {
                source: format!(
                    "WATCH obj:hdr failed before the restore residue wipe for object \
                     {object_id}: {e}"
                )
                .into(),
            }
        })?;
    conn.mark_watched(&header_key, &[]);

    // The store would not give a stable in-`MULTI` `CLIENT ID`, so L3 is
    // disarmed for this wipe: the transaction queues no witness (queueing one
    // on such a store would `EXECABORT` the whole wipe) and nothing is ever
    // accused. Not a failure — record it and continue on L1 + L2, exactly as
    // `create_session` and the structural `update_layer` slow path do. Reported
    // rather than assumed away because on a proxied managed store this is the
    // difference between a fenced destructive `UNLINK` and one running on L1
    // alone.
    if let Some(reason) = witness_baseline.unavailable_reason() {
        if let Some(metrics) = client.metrics_snapshot() {
            metrics.inc_storage_watch_voided(
                WATCH_VOIDED_PATH_RESTORE_RESIDUE_WIPE,
                WATCH_VOIDED_WITNESS_UNAVAILABLE,
            );
        }
        crate::storage::watched_txn::warn_witness_unavailable(
            WATCH_VOIDED_PATH_RESTORE_RESIDUE_WIPE,
            reason,
        );
    }

    // The identity of the connection now carrying the WATCH, for the L2
    // pre-dispatch check below. A `WATCH` dies with its connection, so a
    // replacement between here and the dispatch voids the watch set — and this
    // is the one watched site whose dispatched effect cannot be undone. Reading
    // it is a local map lookup, so it does not widen the window it measures.
    let watch_fingerprint = WatchFingerprint::take(&conn, &watch_hash);

    // Re-read the header ON THE WATCHED CONNECTION. Two cases reach here and
    // both must be handled by the same predicate:
    //
    //   * header ABSENT — the b.kf1 orphan class. Wipe-eligible.
    //   * header PRESENT but structurally incomplete — the sub-task 3 class,
    //     which `apply_restore`'s step-1 predicate now lets through. Also
    //     wipe-eligible, and `KeySchema::object_cleanup_keys` puts `obj:hdr`
    //     at element 0 unconditionally, so the partial header is removed by
    //     the same UNLINK. That is load-bearing rather than incidental:
    //     `write_restored_object`'s single HSET stamps the 12 canonical fields but does
    //     not DEL the key first, so any field a partial header carries that is
    //     not one of the 12 would survive underneath the restored header and be
    //     read back as ours — see the step-1 probe comment for the two with
    //     teeth (`superseded`, `ext_gc_scheduled`).
    //
    // A structurally-COMPLETE header means a peer replica landed the restore
    // between `apply_restore`'s step-1 read and this one. Wipe nothing.
    //
    // A `WRONGTYPE` here is `Permanent`, not `Transient`, and via the same
    // shared classifier every other restore-path read uses: no retry can
    // `HGETALL` a key that is not a hash, so a `Transient` verdict would spin
    // the recovery loop forever — the defect this ticket exists to remove,
    // re-created at a brand-new gate.
    let header_fields: HashMap<String, String> = match conn.hgetall(&header_key).await {
        Ok(fields) => fields,
        Err(e) => {
            let _ = conn.unwatch_pinned().await;
            return Err(classify_object_key_read_error(
                FredErrOrigin::client(client, POOL_LABEL_TXN),
                "HGETALL obj:hdr (residue-wipe WATCH re-read)",
                &header_key,
                object_id,
                e,
            ));
        }
    };
    if header_fields_are_structurally_complete(&header_fields) {
        let _ = conn.unwatch_pinned().await;
        tracing::debug!(
            object_id,
            "restore residue wipe: obj:hdr became structurally complete under the WATCH (a peer \
             replica landed this recovery); wiping nothing and reporting the layer present"
        );
        return Ok(ResidueOutcome::PeerLandedHeader);
    }

    // ── Step 3: enumerate the five indexes ────────────────────────────────
    //
    // On the data pool: the WATCH protects the UNLINK, not these reads, and a
    // read that raced a mutation cannot make the inventory wrong in a way the
    // fence would catch anyway (nothing may mutate these keys while the header
    // is absent or partial).
    let enumerated = enumerate_residue(client, keys, object_id, &oid_str).await;
    let residue = match enumerated {
        Ok(residue) => residue,
        Err(err) => {
            let _ = conn.unwatch_pinned().await;
            return Err(err);
        }
    };

    // ── Step 4: build the inventory and UNLINK it under the WATCH ─────────
    //
    // `KeySchema::object_cleanup_keys` rather than a hand-rolled list, and
    // that choice is the b.9fn invariant rather than a convenience: this is the
    // THIRD route by which an object's `obj:*` family is retired (after
    // `layer::cleanup_old_object` / `cleanup_old_object_retrying` and
    // `orphan_sweep::cleanup_orphaned_object`),
    // and a hand-rolled list is how the root children list `obj:ch:0` and the
    // dangling-parent lists recorded in `obj:chs` got missed the first two
    // times.
    //
    // The per-object stream is never in the inventory — see the module docs.
    // It belongs to the b.4et adjudication that already ran.
    let mut to_unlink = keys.object_cleanup_keys(
        &oid_str,
        &residue.section_ids,
        &residue.field_members,
        &residue.ts_members,
        &residue.tombstone_ids,
        &residue.children_parent_ids,
    );

    // `obj:ext` is withheld from the inventory (b.qpf × b.f4t). It is a member
    // of `object_cleanup_keys` because the two RETIREMENT routes must remove it
    // — there the object is dying and `schedule_old_object_ext_gc` has already
    // consumed the index to release the object's references. This route is not
    // a retirement: the object is about to be re-materialized at the same
    // `object_id`, and `write_restored_object` reads this very key
    // as `ext_before` to compute the `before − after` rewrite releases it owes
    // (`src/storage/layer.rs`, `SurplusPolicy::Suppress`). UNLINKing it here
    // would hand that read an empty pre-image, so no release would be emitted
    // and every restore over residue would leak one reference per hash the
    // residue held and the restored state does not. Releasing them from this
    // lane instead is NOT the alternative — the residue's hash set is not the
    // set being given up, and releasing all of it would over-release hashes the
    // restored state still references. See the module docs.
    //
    // SECOND load-bearing consumer of the withholding (b.yda; b.yuo Part R).
    // A retirement intent's carrier `le:{object_id}:obj:extret` whose recorded
    // attempts all resolved void is settled by
    // `scripts/ext_retire_settle.lua` (this build's HASH carriers only; an
    // older build's SET carrier is dropped, never paid — b.1jt),
    // whose liveness gate cancels it when `EXISTS obj:hdr OR EXISTS obj:ext`.
    // The `obj:ext` leg of that OR exists precisely for THIS window: between
    // this UNLINK (which removes `obj:hdr`) and the materialization, the header
    // is absent but the index still stands, and a settle landing there must
    // cancel — the restored object inherits the standing `+1` (a base-only
    // restore pre-stores nothing). Removing `obj:ext` here would delete that
    // leg's evidence, so such a settle would re-drive a release the restored
    // object still needs: the LOSS direction, ending in meta+data deleted under
    // a live reference.
    //
    // Any future change that does want the index gone must therefore also close
    // that window rather than just deleting the key — add the carrier and its
    // attempt tokens (`obj:extret:attempts`) to the probe list and `DEL` them
    // inside this same WATCHed MULTI, voiding the carrier's attempt tokens on
    // the `{gc}` slot first (b.9ud sub-task 3 describes that hardening).
    // Withholding is the cheaper half of that pair and is why it stands.
    let ext_index_key = keys.obj_ext_index(&oid_str);
    to_unlink.retain(|k| k != &ext_index_key);

    let keys_unlinked = to_unlink.len();

    let txn = match WatchedTxn::open(&conn, &watch_hash, &witness_baseline).await {
        Ok(txn) => txn,
        Err(e) => {
            FredErrOrigin::client(client, POOL_LABEL_TXN).observe_error(&e);
            let _ = conn.unwatch_pinned().await;
            return Err(CheckpointError::Transient {
                source: format!(
                    "MULTI queue failed for the restore residue wipe of object {object_id}: {e}"
                )
                .into(),
            });
        }
    };
    for chunk in to_unlink.chunks(UNLINK_CHUNK_SIZE) {
        // fred's `unlink` takes something convertible into `MultipleKeys`; a
        // `Vec<String>` is, a `&[String]` is not. Same per-chunk clone the
        // paged teardown's `UNLINK`s pay (`storage::retirement::Teardown`).
        let chunk_vec: Vec<String> = chunk.to_vec();
        if let Err(e) = txn.commands().unlink::<(), _>(chunk_vec).await {
            let _ = conn.unwatch_pinned().await;
            return Err(CheckpointError::Transient {
                source: format!(
                    "queueing UNLINK for the restore residue wipe of object {object_id} failed: \
                     {e}"
                )
                .into(),
            });
        }
    }

    // ── L2 pre-dispatch identity check (b.bhb), the last free moment ───────
    //
    // Nothing has reached the wire yet — queueing into the pipeline is a local
    // buffer push — so this is the last point at which noticing that the
    // connection carrying the WATCH has been replaced, or is being replaced
    // right now, costs nothing. Without it the EXEC below would commit
    // UNCONDITIONALLY, which on this site means an irreversible `UNLINK` of a
    // keyspace the fence was supposed to protect.
    //
    // Refusing is free here in a way it is not on the two *commit* sites
    // (`update_layer_once_watch`, `create_once`), where a false refusal costs the
    // caller's write: the wipe is idempotent by construction, so `Transient`
    // re-drives it from the probe with a fresh checkout and a fresh WATCH, and a
    // false refusal costs one extra recovery attempt and nothing else. The
    // restore materialization that follows shares that property rather than the
    // commit sites' — it too is idempotent under retry, an attempt that wrote
    // nothing leaving the object header-less, exactly the state the next attempt
    // expects (`LayerStore::write_restored_object`) — so the contrast here is
    // commit-vs-restore, not checkpoint-lane-vs-storage-lane.
    match dispatch_verdict(&conn, &watch_fingerprint, &watch_hash) {
        DispatchVerdict::Dispatch => {}
        // fred is connected but holds no id for this connection, so L2 has
        // nothing to compare. Dispatch anyway — refusing here would stall every
        // residue wipe on a store that does not answer `CLIENT ID` at handshake,
        // i.e. would make DR restores over a surviving keyspace impossible
        // there — and record that the layer is off.
        DispatchVerdict::DispatchUncomparable => {
            if let Some(metrics) = client.metrics_snapshot() {
                metrics.inc_storage_watch_voided(
                    WATCH_VOIDED_PATH_RESTORE_RESIDUE_WIPE,
                    WATCH_VOIDED_FINGERPRINT_UNAVAILABLE,
                );
            }
            crate::storage::watched_txn::warn_fingerprint_unavailable(
                WATCH_VOIDED_PATH_RESTORE_RESIDUE_WIPE,
            );
        }
        DispatchVerdict::Refuse => {
            let _ = conn.unwatch_pinned().await;
            if let Some(metrics) = client.metrics_snapshot() {
                metrics.inc_storage_watch_voided(
                    WATCH_VOIDED_PATH_RESTORE_RESIDUE_WIPE,
                    WATCH_VOIDED_ABORTED_PRE_EXEC,
                );
            }
            return Err(CheckpointError::Transient {
                source: format!(
                    "restore residue wipe for object {object_id} aborted before EXEC: the WATCH \
                     connection was replaced, so the obj:hdr fence would not have held; the next \
                     attempt re-probes and converges"
                )
                .into(),
            });
        }
    }

    // ── The WATCH…EXEC fault seam (test-only, b.f4t) ──────────────────────
    //
    // The last point at which anything can happen *inside* the fenced window,
    // and the only reason it exists: the tripped-`EXEC` arm below has two
    // verdicts — `PeerLandedHeader` and `Transient` — separated solely by which
    // write landed on `obj:hdr` between the re-read above and the `EXEC` here,
    // and that window is made of real round-trips no test can interleave
    // against without a sleep. An injector that mutates the header from this
    // seam picks the verdict deterministically.
    //
    // AFTER the L2 pre-dispatch check on purpose: L2 asks whether the
    // connection carrying the WATCH is still the one about to dispatch, which
    // is a question about *this* replica's client state and must be answered
    // from the same unperturbed position it is answered from in production.
    // An injected write to `obj:hdr` lands on the data pool, not on this
    // connection, so it cannot change that answer either way.
    //
    // An `Err` unwatches and propagates verbatim, wiping nothing — the same
    // disposition as the L2 `Refuse` arm, and for the same reason: the wipe is
    // idempotent by construction, so the next attempt re-probes and converges.
    #[cfg(feature = "test-checkpoint-fault-injection")]
    if let Some(injector) = ctx.injector {
        if let Err(e) = injector.maybe_inject(RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC) {
            let _ = conn.unwatch_pinned().await;
            return Err(e);
        }
    }

    match txn
        .exec(FredErrOrigin::client(client, POOL_LABEL_TXN))
        .await
    {
        WatchedExec::Conflict => {
            // `obj:hdr` changed between the WATCH and the EXEC, so nothing was
            // wiped. WHAT changed decides the verdict, and a tripped EXEC does
            // not say: any single-field `HSET` trips it too — most concretely a
            // racing `CreateLayer` overwrite's `HSET superseded=1`. Inferring
            // `PeerLandedHeader` from the trip alone would report
            // `ApplyOutcome::AlreadyPresent` for a restore that never ran,
            // ticking `checkpoint_recovery_attempts_total{result="restored"}` —
            // exactly the metric dishonesty this ticket exists to remove,
            // relocated to a new arm.
            //
            // So re-read the header and apply the same structural predicate as
            // the pre-WATCH read. Complete ⇒ a peer replica really did land the
            // restore, and this is a *fence*, not a retry condition: re-running
            // the wipe would delete the peer's freshly restored state.
            // Otherwise the trip was some other mutation and the wipe is simply
            // undone-and-unstarted, so `Transient` re-drives it — at no cost,
            // because it is idempotent by construction.
            conn.mark_unwatched();
            let header_fields: HashMap<String, String> = match conn.hgetall(&header_key).await {
                Ok(fields) => fields,
                Err(e) => {
                    return Err(classify_object_key_read_error(
                        FredErrOrigin::client(client, POOL_LABEL_TXN),
                        "HGETALL obj:hdr (residue-wipe post-conflict re-read)",
                        &header_key,
                        object_id,
                        e,
                    ));
                }
            };
            if header_fields_are_structurally_complete(&header_fields) {
                tracing::debug!(
                    object_id,
                    "restore residue wipe: EXEC tripped on the obj:hdr WATCH and the header is now \
                     structurally complete (a peer replica landed this recovery); wiped nothing \
                     and reporting the layer present"
                );
                return Ok(ResidueOutcome::PeerLandedHeader);
            }
            tracing::debug!(
                object_id,
                "restore residue wipe: EXEC tripped on the obj:hdr WATCH but the header is still \
                 not structurally complete, so no peer landed this recovery — some other write \
                 touched obj:hdr. Wiped nothing; the next attempt re-probes and converges"
            );
            Err(CheckpointError::Transient {
                source: format!(
                    "restore residue wipe for object {object_id} lost its obj:hdr WATCH to a write \
                     that did not land a complete header; nothing was wiped and the next attempt \
                     re-probes"
                )
                .into(),
            })
        }
        WatchedExec::Committed { witness } => {
            conn.mark_unwatched();
            if witness_baseline.contradicted_by(witness) {
                // The transaction's own `CLIENT ID` disagrees with the baseline
                // sampled on the WATCH's connection: the UNLINK provably ran
                // somewhere that never held the WATCH, so the peer-landed fence
                // did not hold for this wipe. There is nothing to compensate —
                // a delete cannot be undone — and the restore that follows
                // replaces whatever was removed wholesale with THIS replica's
                // resolved lineage head. That is a convergence only when the
                // peer resolved the same head: `select_restore_source` steps
                // down per replica, so two concurrent restores straddling a
                // retention prune or a chain collapse can land on different
                // generations, and then the layer comes back internally
                // consistent but silently stepped back to this replica's
                // (possibly older) generation — a recovery-point regression
                // whose only evidence is this WARN plus the `stepped_down_*`
                // cells of `live_edit_checkpoint_restore_lineage_total`. WARN so
                // the condition is visible rather than silently assumed away —
                // and tick the alertable cell, because this is the one watched
                // site whose unfenced effect is an irreversible delete.
                if let Some(metrics) = client.metrics_snapshot() {
                    metrics.inc_storage_watch_voided(
                        WATCH_VOIDED_PATH_RESTORE_RESIDUE_WIPE,
                        WATCH_VOIDED_COMMITTED_UNWATCHED,
                    );
                }
                tracing::warn!(
                    object_id,
                    "restore residue wipe committed on a connection that never held its WATCH; \
                     the peer-landed fence did not hold for this wipe. If a peer replica landed \
                     this recovery concurrently, its freshly restored state was removed and this \
                     replica's restore replaces it wholesale. The object converges only if both \
                     replicas resolved the same checkpoint lineage; restore-source selection steps \
                     down per replica, so if they did not, this layer has silently stepped back to \
                     this replica's (possibly older) generation — internally consistent, no client \
                     error, but a recovery-point regression on that one layer. Correlate with \
                     live_edit_checkpoint_restore_lineage_total{{outcome=~\"stepped_down_.*\"}} in \
                     the same window to tell the two apart. A repeating signal means this backing \
                     store is replaying transactions onto other connections"
                );
            }
            if let Some(metrics) = client.metrics_snapshot() {
                metrics.inc_checkpoint_restore_residue_wiped(&config.service_label);
            }
            // WARN, not INFO: reaching this arm means a keyspace survived under
            // an object the DR path is restoring, which is a state an operator
            // should know about even though recovery handles it correctly. The
            // per-family breakdown is what says WHICH survived, and
            // `ext_bearing_records` sizes the reconciliation `write_restored_object` is
            // about to perform against the retained `obj:ext` pre-image (see the
            // module docs) — an upper bound on the records whose hashes are in
            // that pre-image, not a residual.
            tracing::warn!(
                object_id,
                keys_unlinked,
                sections = residue.section_ids.len(),
                fields = residue.field_members.len(),
                time_samples = residue.ts_members.len(),
                tombstones = residue.tombstone_ids.len(),
                children_lists = residue.children_parent_ids.len(),
                ext_bearing_records = residue.field_members.len() + residue.ts_members.len(),
                "checkpoint restore found a surviving keyspace under this object and wiped it \
                 before materializing: the restore is an authoritative REPLACE, so stale members \
                 must not union with the restored set. The per-object external-value reference \
                 index (obj:ext) was deliberately LEFT in place as the materialization's \
                 pre-image; it releases the references the restored state gives up \
                 (before − after) inside its own transaction"
            );
            Ok(ResidueOutcome::Wiped { keys_unlinked })
        }
        // A discarded transaction (`EXECABORT`, or the cluster check's
        // `-TRYAGAIN` to the `EXEC`: nothing ran) keeps the same `Transient`
        // answer, from its `EXEC` reply.
        WatchedExec::ExecError(e) | WatchedExec::Discarded { exec: e, .. } => {
            // Not a command timeout by construction: `ExecError` and
            // `Discarded` need a full reply set, so `e` is a server error
            // frame (see `WatchedTxn::exec`).
            //
            // The guard is left ARMED on purpose: an errored `EXEC` does not
            // prove the transaction executed, so the watch set may still be
            // live and `Drop`'s slot-pinned UNWATCH is what clears it (the same
            // posture `SessionStore::create_once` documents).
            //
            // `Transient`, and not by omission: `UNLINK` is type-agnostic, so
            // this arm cannot carry the wrong-typed-key refusal that step 3
            // classifies `Permanent`. What it can carry is an `EXECABORT` or a
            // transient server frame, and both are cleared by re-driving a wipe
            // that is idempotent by construction.
            Err(CheckpointError::Transient {
                source: format!(
                    "EXEC failed for the restore residue wipe of object {object_id}: {e}"
                )
                .into(),
            })
        }
        WatchedExec::InDoubt(e) => {
            // A timed-out EXEC was already counted by `WatchedTxn::exec`.
            //
            // Whether the UNLINK ran is unknown, and — unlike the commit paths
            // that reconcile — this needs no reconciliation: both answers are
            // states the next attempt handles. Ran ⇒ the re-probe finds less
            // residue (or none) and converges; did not run ⇒ the re-probe finds
            // the same residue and wipes it. The guard stays armed because the
            // WATCH may still be live.
            Err(CheckpointError::Transient {
                source: format!(
                    "restore residue wipe for object {object_id} was left in doubt (the EXEC reply \
                     was lost); the next attempt re-probes and converges: {e}"
                )
                .into(),
            })
        }
    }
}

/// The five index enumerations [`KeySchema::object_cleanup_keys`] consumes.
struct ResidueInventory {
    section_ids: Vec<String>,
    field_members: Vec<String>,
    ts_members: Vec<String>,
    tombstone_ids: Vec<String>,
    children_parent_ids: Vec<String>,
}

/// Read the five indexes concurrently, routing every failure through
/// [`classify_object_key_read_error`] — the one shared classifier the restore
/// path's header reads use too, so a `WRONGTYPE` is
/// [`CheckpointError::Permanent`] and everything else `Transient` wherever the
/// wrong-typed key is first touched. Nothing is ever `unwrap_or_default`ed (the
/// b.u96 posture: a failed enumeration is never silently substituted with an
/// empty result).
///
/// Concurrent rather than serial for the same reason
/// `LayerStore::probe_keyspace_types` fans out: these are independent reads and
/// the caller is holding an exclusive `txn_pool` slot while they run.
async fn enumerate_residue(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    oid_str: &str,
) -> Result<ResidueInventory, CheckpointError> {
    let sections = async {
        let key = keys.obj_sections_index(oid_str);
        let out: Vec<String> = client.pool.smembers(&key).await.map_err(|e| {
            classify_object_key_read_error(
                FredErrOrigin::client(client, POOL_LABEL_DATA),
                "SMEMBERS obj:secs",
                &key,
                object_id,
                e,
            )
        })?;
        Ok::<_, CheckpointError>(out)
    };
    let fields = async {
        let key = keys.obj_fields_index(oid_str);
        let out: Vec<String> = client
            .pool
            .zrange(&key, 0, -1, None, false, None, false)
            .await
            .map_err(|e| {
                classify_object_key_read_error(
                    FredErrOrigin::client(client, POOL_LABEL_DATA),
                    "ZRANGE obj:flds",
                    &key,
                    object_id,
                    e,
                )
            })?;
        Ok::<_, CheckpointError>(out)
    };
    let time_samples = async {
        let key = keys.obj_time_samples_index(oid_str);
        let out: Vec<String> = client
            .pool
            .zrange(&key, 0, -1, None, false, None, false)
            .await
            .map_err(|e| {
                classify_object_key_read_error(
                    FredErrOrigin::client(client, POOL_LABEL_DATA),
                    "ZRANGE obj:tss",
                    &key,
                    object_id,
                    e,
                )
            })?;
        Ok::<_, CheckpointError>(out)
    };
    let tombstones = async {
        let key = keys.obj_deleted_sections_index(oid_str);
        let out: Vec<String> = client.pool.smembers(&key).await.map_err(|e| {
            classify_object_key_read_error(
                FredErrOrigin::client(client, POOL_LABEL_DATA),
                "SMEMBERS obj:dels",
                &key,
                object_id,
                e,
            )
        })?;
        Ok::<_, CheckpointError>(out)
    };
    let children = async {
        let key = keys.obj_children_index(oid_str);
        let out: Vec<String> = client.pool.smembers(&key).await.map_err(|e| {
            classify_object_key_read_error(
                FredErrOrigin::client(client, POOL_LABEL_DATA),
                "SMEMBERS obj:chs",
                &key,
                object_id,
                e,
            )
        })?;
        Ok::<_, CheckpointError>(out)
    };

    let (section_ids, field_members, ts_members, tombstone_ids, children_parent_ids) =
        tokio::try_join!(sections, fields, time_samples, tombstones, children)?;

    Ok(ResidueInventory {
        section_ids,
        field_members,
        ts_members,
        tombstone_ids,
        children_parent_ids,
    })
}

#[cfg(test)]
mod tests {
    //! Unit lane for [`wipe_restore_residue`]'s three [`ResidueOutcome`] arms
    //! (b.f4t).
    //!
    //! The function is `pub(crate)`, so these tests can observe the outcome
    //! ENUM directly — something the integration lane cannot do, because
    //! `apply_checkpoint` folds `AlreadyPresent` and `Applied` into the same
    //! `Ok(())`. That distinction is the whole reason this lane exists:
    //! `Clean` vs `Wiped` is what says whether the counter ticked, and
    //! `PeerLandedHeader` is what says a peer's freshly landed restore was
    //! fenced rather than deleted.
    //!
    //! # What is covered elsewhere, and why it is not here
    //!
    //! What this lane covers is the arm that decides the peer-landed question
    //! on the post-`WATCH` re-read — one round trip BEFORE the `EXEC` — plus
    //! the `Clean` / `Wiped` outcomes and the predicate itself
    //! (`crate::storage::layer::tests::header_fields_are_structurally_complete_*`,
    //! beside the declaration — the predicate is storage-owned and
    //! `checkpoint::apply` only re-exports it).
    //!
    //! The **tripped-`EXEC`** producers of the same two verdicts need a write
    //! to land on `obj:hdr` inside the `WATCH` … `EXEC` window, which is made
    //! of five concurrent enumerations and a `MULTI` queue — real round trips a
    //! test cannot interleave against without a sleep (forbidden as a
    //! synchronisation primitive), and a racing writer would land at an
    //! arbitrary point and yield `PeerLandedHeader` OR `Transient` OR `Wiped`
    //! by luck.
    //! [`RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`](crate::checkpoint::apply::RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC)
    //! is what makes the choice deterministic, and all three of its exits are covered by the
    //! composed, handler-level arms in
    //! `tests/integration/checkpoint/restore_fault_injection_matrix_test.rs`:
    //!
    //! * `a_peer_landing_a_complete_header_inside_the_wipe_window_fences_the_restore`
    //!   — the seam lands the 12 canonical fields, the `EXEC` trips, and the
    //!   post-conflict re-read reports `PeerLandedHeader` (⇒ `AlreadyPresent`,
    //!   nothing wiped, nothing written);
    //! * `a_partial_header_write_inside_the_wipe_window_is_transient_and_converges`
    //!   — the seam lands `superseded=1`, the `EXEC` trips on the same `WATCH`,
    //!   and the still-incomplete header yields `Transient` rather than a
    //!   `restored` tick for a restore that never ran;
    //! * `a_refusal_inside_the_wipe_window_wipes_nothing_and_converges` — the
    //!   seam returns `Err`, the `WATCH` is released, and the error propagates
    //!   verbatim with nothing wiped.
    //!
    //! They live there rather than here because the interesting consequences —
    //! the recovery-attempt result label, the surfaced gRPC code, and the
    //! converging re-drive — are only observable through the hook and the
    //! handler, which this lane does not run.

    use super::*;
    use crate::checkpoint::apply::tests::{seed_complete_header, test_config};
    use crate::storage::test_support::TestValkey;
    use fred::interfaces::{KeysInterface, SetsInterface};

    /// Seed one surviving index member + its per-record hash, i.e. the b.kf1
    /// orphan shape at its smallest: a section in `obj:secs` whose
    /// `obj:sec:{id}` hash is alive, under an object with no header at all.
    async fn seed_orphan_section(tv: &TestValkey, object_id: u64, section_id: &str) {
        let oid = object_id.to_string();
        let _: () = tv
            .client()
            .pool
            .sadd(tv.key_schema().obj_sections_index(&oid), section_id)
            .await
            .expect("SADD obj:secs");
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema().obj_section(&oid, section_id),
                [("name", "stale"), ("version", "7")],
            )
            .await
            .expect("HSET obj:sec");
    }

    /// Nothing survived under this `object_id` — the whole-keyspace-loss case,
    /// which is the overwhelmingly common DR shape. `Clean`, and by contract
    /// nothing was watched, unlinked, or counted.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wipe_restore_residue_empty_keyspace_reports_clean() {
        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);

        let outcome = wipe_restore_residue(&layer_store, 9_301, &test_config(1000))
            .await
            .expect("a clean keyspace must not error");

        assert_eq!(
            outcome,
            ResidueOutcome::Clean,
            "an empty keyspace costs exactly one EXISTS and reports Clean — a `Wiped` here \
             would mean the common DR path pays a WATCH + txn checkout it does not need, and \
             would tick the residue counter on every ordinary restore"
        );
    }

    /// Residue under a header-less object is `Wiped`, and the inventory it
    /// reports is the one [`KeySchema::object_cleanup_keys`] built — never an
    /// ad-hoc list.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wipe_restore_residue_surviving_index_reports_wiped_and_removes_it() {
        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let oid: u64 = 9_302;
        seed_orphan_section(&tv, oid, "7").await;

        let outcome = wipe_restore_residue(&layer_store, oid, &test_config(1000))
            .await
            .expect("a header-less residue must be wipeable");

        match outcome {
            ResidueOutcome::Wiped { keys_unlinked } => assert!(
                keys_unlinked >= 2,
                "the inventory must at least carry the index and its one per-record hash; got \
                 {keys_unlinked}"
            ),
            other => panic!("surviving residue must report Wiped; got {other:?}"),
        }

        let oid_str = oid.to_string();
        for key in [
            tv.key_schema().obj_sections_index(&oid_str),
            tv.key_schema().obj_section(&oid_str, "7"),
        ] {
            let alive: u64 = tv
                .client()
                .pool
                .exists(&key)
                .await
                .expect("EXISTS after the wipe");
            assert_eq!(alive, 0, "`{key}` must not survive the wipe");
        }
    }

    /// A structurally-complete `obj:hdr` observed on the `WATCH`ed connection
    /// means a peer replica landed this recovery in the gap. Wipe NOTHING and
    /// report [`ResidueOutcome::PeerLandedHeader`].
    ///
    /// The residue seeded alongside the header is what makes the assertion
    /// bite: if the wipe reached its `UNLINK` it would have taken the peer's
    /// index with it, and "nothing was deleted" is the only observation that
    /// distinguishes a fence from a race the peer happened to win.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wipe_restore_residue_complete_header_reports_peer_landed_and_deletes_nothing() {
        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let oid: u64 = 9_303;
        seed_orphan_section(&tv, oid, "7").await;
        seed_complete_header(&tv, oid, "500").await;

        let outcome = wipe_restore_residue(&layer_store, oid, &test_config(1000))
            .await
            .expect("a peer-landed header is a fence, not an error");

        assert_eq!(
            outcome,
            ResidueOutcome::PeerLandedHeader,
            "a structurally-complete obj:hdr under the WATCH means a peer landed the restore; \
             wiping there would delete its freshly restored state"
        );

        let oid_str = oid.to_string();
        for key in [
            tv.key_schema().obj_header(&oid_str),
            tv.key_schema().obj_sections_index(&oid_str),
            tv.key_schema().obj_section(&oid_str, "7"),
        ] {
            let alive: u64 = tv
                .client()
                .pool
                .exists(&key)
                .await
                .expect("EXISTS after the peer-landed fence");
            assert_eq!(
                alive, 1,
                "`{key}` must survive untouched — the PeerLandedHeader arm deletes nothing"
            );
        }
    }

    /// `obj:ext` is withheld from the `UNLINK` inventory — and it is the ONLY
    /// member withheld, with `obj:hdr` still at element 0.
    ///
    /// Both halves are load-bearing and neither is observable from the
    /// integration lane, which sees only the composed restore:
    ///
    /// * **`obj:ext` survives.** It is `write_restored_object`'s ext pre-image:
    ///   the restore materializer `HGETALL`s it as `ext_before` and owes exactly
    ///   `before − after` in `rewrite` releases. Unlinking it here hands that
    ///   read an empty pre-image, so nothing is released and every restore over
    ///   residue leaks one `ref_count` per dropped hash. The two RETIREMENT
    ///   routes DO unlink it (pinned in
    ///   `tests/integration/storage/b_qpf_tests.rs`) — there the object is
    ///   dying and its references have already been released — which is why
    ///   the key is in the shared inventory at all and why this route has to
    ///   subtract it rather than the inventory omitting it.
    /// * **`obj:hdr` is still element 0.** The subtraction is a `retain` over
    ///   the inventory, and `object_cleanup_keys`' header-first ordering is
    ///   what leaves the header ABSENT for `write_restored_object`'s fence,
    ///   which since b.h4m refuses to write over any header it finds rather
    ///   than `HSET`ing through it. A `retain` that dropped the wrong entry
    ///   would be invisible to every other test here, all of which assert on
    ///   keys the wipe removes.
    ///
    /// The count assertion is exact (`len() - 1`) rather than a bound, so a
    /// second withheld member — or a stream that crept back in — fails here
    /// rather than being absorbed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wipe_restore_residue_withholds_obj_ext_from_the_unlink_inventory() {
        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let oid: u64 = 9_305;
        let oid_str = oid.to_string();
        seed_orphan_section(&tv, oid, "7").await;

        // The pre-image the restore is about to consume: one hash held by one
        // of the object's value rows.
        let ext_index = tv.key_schema().obj_ext_index(&oid_str);
        let _: () = tv
            .client()
            .pool
            .hset(&ext_index, [("00".repeat(32), "1")])
            .await
            .expect("HSET obj:ext pre-image");

        let outcome = wipe_restore_residue(&layer_store, oid, &test_config(1000))
            .await
            .expect("a header-less residue must be wipeable");

        // The inventory the wipe built, reconstructed from the same enumerations
        // this fixture seeded: one section, nothing else, no stream.
        let inventory =
            tv.key_schema()
                .object_cleanup_keys(&oid_str, &["7".to_owned()], &[], &[], &[], &[]);
        assert_eq!(
            inventory.first().map(String::as_str),
            Some(tv.key_schema().obj_header(&oid_str).as_str()),
            "the ordering invariant this wipe depends on: obj:hdr is element 0, so the partial \
             header is removed before `write_restored_object` evaluates its header fence — which \
             refuses outright on a header that is present but incomplete (b.h4m)"
        );
        assert!(
            inventory.contains(&ext_index),
            "non-vacuity: obj:ext must BE in the shared inventory, or `keys_unlinked == len() - 1` \
             below would hold for the wrong reason and this test would prove nothing about the \
             carve-out"
        );

        match outcome {
            ResidueOutcome::Wiped { keys_unlinked } => assert_eq!(
                keys_unlinked,
                inventory.len() - 1,
                "exactly ONE member of the inventory is withheld — obj:ext. A count of \
                 {} means nothing was withheld and the ext pre-image was destroyed; anything \
                 lower means a second key silently fell out of the teardown",
                inventory.len()
            ),
            other => panic!("surviving residue must report Wiped; got {other:?}"),
        }

        let alive: u64 = tv
            .client()
            .pool
            .exists(&ext_index)
            .await
            .expect("EXISTS obj:ext after the wipe");
        assert_eq!(
            alive, 1,
            "`{ext_index}` must SURVIVE the wipe: `write_restored_object` reads it as \
             `ext_before` a few statements later, and an empty pre-image means the releases the \
             restore owes are never computed — a permanent refcount leak with no reaper"
        );
    }

    /// A header present but structurally INCOMPLETE is residue too, and it is
    /// one of the probe keys: with no other `obj:*` key alive, omitting
    /// `obj:hdr` from the probe would short-circuit this case as `Clean` and
    /// leave the partial header in place — which since b.h4m does not merely
    /// get `HSET` over, it makes the restore REFUSE (`write_restored_object`
    /// answers `Unavailable` on any header it finds), so the layer would never
    /// come back at all until something else removed the key.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wipe_restore_residue_partial_header_alone_is_wiped() {
        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let oid: u64 = 9_304;
        let header_key = tv.key_schema().obj_header(&oid.to_string());

        // `seq` present, `created_by` absent — plus the stale flag with the
        // sharpest teeth, the one whose survival underneath a restored header
        // would permanently un-write the layer. That hazard is why
        // `write_restored_object` refuses a present-but-incomplete header
        // outright rather than HSETing through it, and why this wipe removing
        // the key is what lets the next attempt converge.
        let _: () = tv
            .client()
            .pool
            .hset(&header_key, [("seq", "100"), ("superseded", "1")])
            .await
            .expect("HSET partial obj:hdr");

        let outcome = wipe_restore_residue(&layer_store, oid, &test_config(1000))
            .await
            .expect("a partial header must be wipeable");

        assert!(
            matches!(outcome, ResidueOutcome::Wiped { .. }),
            "a lone partial header must report Wiped — a `Clean` here means obj:hdr fell out of \
             the residue probe and the stale header survives the restore; got {outcome:?}"
        );

        let alive: u64 = tv
            .client()
            .pool
            .exists(&header_key)
            .await
            .expect("EXISTS obj:hdr after the wipe");
        assert_eq!(
            alive, 0,
            "the partial header must be gone; `object_cleanup_keys` puts obj:hdr at element 0 \
             precisely so `write_restored_object`'s fence finds an ABSENT key and materializes"
        );
    }

    /// A surviving `UpdateLayer` commit-token record (`obj:wtok`, b.75i) with
    /// nothing else alive is residue too, and the wipe removes it.
    ///
    /// The record names commits of the incarnation being replaced. Left in
    /// place under the restored object, a call whose earlier attempt landed
    /// before the loss would find its own token there and report a sequence
    /// the restore may not contain — instead of re-applying onto the restored
    /// state. No other probe key is alive here, so a `Clean` means `obj:wtok`
    /// fell out of the residue probe and the record survives the restore.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wipe_restore_residue_commit_token_record_alone_is_wiped() {
        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let oid: u64 = 9_306;
        let wtok_key = tv.key_schema().obj_write_token(&oid.to_string());

        let _: () = tv
            .client()
            .pool
            .hset(
                &wtok_key,
                [(
                    "writer",
                    "0123456789abcdef0123456789abcdef:41:1700000000000000",
                )],
            )
            .await
            .expect("HSET obj:wtok");

        let outcome = wipe_restore_residue(&layer_store, oid, &test_config(1000))
            .await
            .expect("a lone commit-token record must be wipeable");

        assert!(
            matches!(outcome, ResidueOutcome::Wiped { .. }),
            "a lone commit-token record must report Wiped — a `Clean` here means obj:wtok fell \
             out of the residue probe and a pre-loss commit token survives the restore; got \
             {outcome:?}"
        );
        let alive: u64 = tv
            .client()
            .pool
            .exists(&wtok_key)
            .await
            .expect("EXISTS obj:wtok after the wipe");
        assert_eq!(
            alive, 0,
            "the record must be gone — `object_cleanup_keys` names it, and unlike `obj:ext` the \
             wipe does not withhold it"
        );
    }

    /// A surviving `UpdateLayer` commit release ring (`obj:wrel`) with nothing
    /// else alive is residue too, and the wipe removes it — the ring's fields
    /// and an older build's per-`client_id` field alike, since the key goes
    /// whole.
    ///
    /// The ring's entries are commits of the incarnation being replaced. Left
    /// in place under the restored object, a call still in flight across the
    /// restore could find its own pre-loss entry by its token — on an own-call
    /// verdict, or through the lost-reply lookup — and enqueue releases for a
    /// commit the restored object may not contain; an older build's
    /// per-`client_id` field could likewise match a restored commit's
    /// sequence. Either is an over-release that can delete a still-referenced
    /// blob. No other probe key is alive here, so a `Clean` means `obj:wrel`
    /// fell out of the residue probe and the ring survives the restore.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wipe_restore_residue_commit_release_record_alone_is_wiped() {
        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let oid: u64 = 9_309;
        let wrel_key = tv.key_schema().obj_write_releases(&oid.to_string());
        const TOKEN: &str = "0123456789abcdef0123456789abcdef";
        const ENTRY: &str =
            "41 rewrite:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

        let _: () = tv
            .client()
            .pool
            .hset(
                &wrel_key,
                [
                    // One ring entry, its start slot and the next slot.
                    (format!("|t|{TOKEN}").as_str(), ENTRY),
                    ("|s|0", TOKEN),
                    ("|n|", "1"),
                    // An older build's per-`client_id` field.
                    ("writer", ENTRY),
                ],
            )
            .await
            .expect("HSET obj:wrel");

        let outcome = wipe_restore_residue(&layer_store, oid, &test_config(1000))
            .await
            .expect("a lone commit release ring must be wipeable");

        assert!(
            matches!(outcome, ResidueOutcome::Wiped { .. }),
            "a lone commit release ring must report Wiped — a `Clean` here means obj:wrel fell \
             out of the residue probe and a pre-loss ring survives the restore; got {outcome:?}"
        );
        let alive: u64 = tv
            .client()
            .pool
            .exists(&wrel_key)
            .await
            .expect("EXISTS obj:wrel after the wipe");
        assert_eq!(
            alive, 0,
            "the ring must be gone — `object_cleanup_keys` names it"
        );
    }

    /// A surviving snapshot fence (`obj:snapfence`) or writer-intent key
    /// (`obj:snapintent`, Spec SDD SR-5.1), each alone with nothing else alive,
    /// is residue too, and the wipe removes it.
    ///
    /// Both describe the incarnation being replaced: a live `r:*` lease would
    /// hold the restored layer's writers to an epoch no live reader owns, a
    /// live `w:*` intent would hold its readers off, and a stored gap and size
    /// would size the restored layer's limits from the old one — each until
    /// the key's `PEXPIRE` ran out. No other probe key is alive in either case,
    /// so a `Clean` means the key fell out of the residue probe.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wipe_restore_residue_lone_snapshot_fence_or_intent_key_is_wiped() {
        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let fence_oid: u64 = 9_307;
        let intent_oid: u64 = 9_308;
        let fence_key = tv.key_schema().obj_snapshot_fence(&fence_oid.to_string());
        let intent_key = tv.key_schema().obj_snapshot_intent(&intent_oid.to_string());
        let _: () = tv
            .client()
            .pool
            .hset(
                &fence_key,
                [
                    ("r:0123456789abcdef0123456789abcdef", "99999999999999"),
                    ("epoch_start", "1700000000000"),
                    ("size_e", "150216"),
                    ("size_b", "1799752"),
                ],
            )
            .await
            .expect("HSET obj:snapfence");
        let _: () = tv
            .client()
            .pool
            .hset(
                &intent_key,
                [("w:0123456789abcdef0123456789abcdef", "99999999999999")],
            )
            .await
            .expect("HSET obj:snapintent");

        for (oid, key, what) in [
            (fence_oid, &fence_key, "obj:snapfence"),
            (intent_oid, &intent_key, "obj:snapintent"),
        ] {
            let outcome = wipe_restore_residue(&layer_store, oid, &test_config(1000))
                .await
                .unwrap_or_else(|e| panic!("a lone {what} must be wipeable: {e:?}"));

            assert!(
                matches!(outcome, ResidueOutcome::Wiped { .. }),
                "a lone {what} must report Wiped — a `Clean` here means it fell out of the \
                 residue probe and the old incarnation's fence state survives the restore; got \
                 {outcome:?}"
            );
            let alive: u64 = tv
                .client()
                .pool
                .exists(key)
                .await
                .expect("EXISTS after the wipe");
            assert_eq!(
                alive, 0,
                "{what} must be gone — `object_cleanup_keys` names it, and unlike `obj:ext` the \
                 wipe does not withhold it"
            );
        }
    }

    /// A residue probe `EXISTS` that gets no reply within the command timeout
    /// counts exactly one `valkey_errors_total{pool="data",
    /// kind="command_timeout"}` and reports `Transient`, wiping nothing. The
    /// probe is held behind a server-side `BLPOP` on every data-pool
    /// connection, so the `Timeout` is fred's real one.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wipe_restore_residue_timed_out_probe_counts_one_data_command_timeout() {
        use crate::storage::test_support::{
            command_timeout_row, command_timeouts_only_on, release_all,
        };
        let tv = TestValkey::setup().await;
        let metrics = std::sync::Arc::new(crate::observability::metrics::metrics_scrape_only(
            "residue",
        ));
        tv.client()
            .install_observability_hooks(std::sync::Arc::clone(&metrics))
            .await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);

        let stalls = tv.stall_data_pool().await;
        let outcome = wipe_restore_residue(&layer_store, 9_311, &test_config(1000)).await;
        release_all(stalls).await;

        assert!(
            matches!(outcome, Err(CheckpointError::Transient { .. })),
            "a probe that timed out is transient; got {outcome:?}"
        );
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on("data", 1),
            "the timed-out EXISTS must count once, on the data pool"
        );
    }
}
