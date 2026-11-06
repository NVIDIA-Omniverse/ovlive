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

//! The two pre-PUT gates every durable layer generation passes: the liveness
//! guard — don't write a generation for an object that is already provably
//! dead (b.hka, deferred from b.vn6) — and the clean-up fence — record the
//! re-arm the checkpoint clean-up's delete rule depends on, and don't write
//! while the clean-up is deleting something the generation depends on (b.g65;
//! see "The clean-up fence" below).
//!
//! # Why it lives here and not in an adapter
//!
//! The natural home for a "refuse this put" rule is beside
//! [`validate_tail_put`](crate::checkpoint::adapters::validate_tail_put) and
//! [`validate_generation_identity`](crate::checkpoint::adapters::validate_generation_identity)
//! in `adapters/mod.rs`, and this ticket's own coordination fence proposed
//! exactly that. It cannot go there: those two are **pure predicates over the
//! meta**, while this one is a question about the *live Valkey keyspace*, and
//! no adapter holds a Valkey handle. `LocalDiskCheckpointStore` holds paths;
//! `StorageApiCheckpointStore` holds an HTTP client, a base path and a token
//! provider. Handing every adapter a `ValkeyClient` so a store could ask
//! whether a layer still exists would invert the dependency direction the
//! whole checkpoint subsystem is built on (`checkpoint → storage`, never a
//! store reaching back into the live keyspace). So the predicate lives one
//! layer up, at the four call sites that *do* hold a client, and the two
//! meta-only predicates stay where they are.
//!
//! # The race
//!
//! `DeleteLayer` is a sequence of independent commands inside a retry helper,
//! not a transaction. A producer that started building a generation before the
//! delete landed will finish and PUT for an object that no longer exists. The
//! PUT itself is not harmful — it is a durable artifact nobody will read — but
//! it costs a multi-MB write, a retention slot, and a round trip through the
//! reconciliation dead-object phase before it is reclaimed.
//!
//! # What this is, precisely
//!
//! **Advisory, and deliberately so.** The predicate is
//! `HEXISTS le:{object_id}:obj:hdr seq` — the same predicate
//! `storage/scripts/stamp_cursor_if_live.lua` uses, for the reasons its header
//! gives: on a live layer it is equivalent to `EXISTS` (`create_layer` and both
//! materialization entry points — `write_object` for CreateLayer,
//! `write_restored_object` for the DR restore — write `seq` inside the one
//! MULTI/EXEC that creates the
//! header), and against a header that is *already* a cursor-only phantom it is
//! strictly stronger, so the check is self-healing over pre-existing garbage
//! rather than perpetuating it.
//!
//! Three postures, and the asymmetry is the whole design:
//!
//! - a definite `false` **skips the PUT** — the object is provably dead;
//! - a definite `true` proceeds;
//! - a Valkey **error proceeds too**. This probe is an optimisation, so
//!   failing a real generation on a transient blip would trade something
//!   valuable for nothing. The same posture the b.vn6 bootstrap-local probe
//!   took, whose body this module absorbed.
//!
//! # It is not the only header question a producer asks
//!
//! Three of the four producers — the snapshot build (`assembler`), the
//! bootstrap base (`base_bootstrap`) and the tail capture (`tail_capture`) —
//! also run
//! [`SnapshotLoader::read_header_identity`](crate::checkpoint::assembler::SnapshotLoader::read_header_identity),
//! and the two predicates can disagree: `HEXISTS seq` answers `true` for a
//! header that exists with an empty `session_id` / `layer_name`, while the
//! identity read reports that same header as absent (b.hka review round 2). For
//! those three the disagreement is resolved by **ordering, not precedence** —
//! each reads identity first and returns before it ever reaches this guard, so
//! the degenerate header is refused by the identity read and this probe is
//! never consulted about it.
//!
//! The fourth producer, `rebase::rebase_layer_lineage_with_probe`, never calls
//! `read_header_identity` at all and is **structurally exempt** rather than
//! merely ordered. Two independent reasons:
//!
//! - its identity is **base-meta-sourced**, not header-sourced —
//!   `adjudicate_composed_meta(base_meta, ..)` carries the four identity fields
//!   forward from the base generation it is collapsing, so no live `obj:hdr`
//!   read is involved in producing them;
//! - that base already passed [`validate_generation_identity`](crate::checkpoint::adapters::validate_generation_identity)
//!   at its own put gate, so the fields it forwards are already non-empty by
//!   construction — the degenerate-header case this section is about cannot be
//!   what a re-base stamps.
//!
//! (It is also unreachable from the degenerate header for a third, incidental
//! reason: a re-base runs only on the tail capture's `ChainFull` return, which
//! happens *before* that capture's own identity read.)
//!
//! So nothing here needs to re-check identity. The bug a future producer would
//! introduce is not "PUT before reading identity" — it is **PUT while neither
//! reading identity nor sourcing it from a base that a put gate already
//! validated**.
//!
//! # Residual — read this before treating it as a guarantee
//!
//! It **cannot** catch a delete that lands after the probe and before (or
//! during) the PUT. The window is narrowed, never closed, and no amount of
//! re-probing closes it: the store write is not transactional with the Valkey
//! keyspace and cannot be made so. This therefore **complements and never
//! replaces** the three mechanisms that actually make the outcome safe:
//!
//! 1. the b.vn6 **atomic conditional cursor stamp** at all four stamp sites
//!    (`stamp_cursor_if_live.lua`), which is the real guard — it is what stops
//!    a post-PUT `HSET` from resurrecting a deleted header as a phantom;
//! 2. the **just-written-generation reap** on a refused stamp;
//! 3. the `le:ckpt:{recon}:dead` **reconcile backstop**, which reclaims
//!    artifacts written for objects that died in any window at all.
//!
//! Deleting any of those three because this probe exists would be a
//! correctness regression, not a simplification.
//!
//! # The clean-up fence (b.g65, SDD §22.17.3, §22.17.5)
//!
//! The checkpoint clean-up deletes the backup copy of a large value whose
//! tombstone has aged and that no retained generation's ext-refs sidecar
//! names, and it reaps orphaned stems. It holds no lock that keeps writers
//! out; its deletes are licensed in Valkey instead, and that license depends
//! on every producer of a generation running one script immediately before
//! its adapter PUT: `storage::recon_fence::recon_ext_rearm`, on the data pool,
//! after the liveness probe. **A producer that publishes without going
//! through [`put_layer_checkpoint_if_live`] with a probe reopens the race the
//! clean-up's delete rule closes.**
//!
//! - **Re-arm.** Every hash the generation's sidecar names that is tombstoned
//!   is recorded on `le:ckpt:{recon}:rearm` at the Valkey clock. Before the
//!   PUT, because the sidecar is the first artifact the adapter publishes: a
//!   collect whose snapshot predates the sidecar then finds the re-arm after
//!   its threshold and refuses the delete.
//! - **Stem.** The generation's stem (`<object_id>.<snap_seq>`, or
//!   `<object_id>.<to_seq>.tail` for a tail) is removed from
//!   `le:ckpt:{recon}:orphans` — the publish heals its own orphan candidate —
//!   and from `le:ckpt:{recon}:deleting`, unless the publish is refused
//!   (below): a refused publish writes nothing for its stem.
//! - **Refusal.** No PUT when the stem is announced within the grace (the
//!   clean-up is deleting an earlier copy of this same generation), or when
//!   the collect licensed the delete of a named hash's backup copy within the
//!   producer's reliance window: the time since the producer's reliance start
//!   (when it began reading the state it publishes) plus the store's
//!   `publish_worst_case`. A window longer than the collect's 24 h record
//!   refuses too, since it cannot be answered. A Valkey error refuses as well
//!   (fail closed), with a WARN rather than the other refusals' INFO. The
//!   refusal is a typed `Transient`,
//!   [`CleanupInFlight`](super::store::CleanupInFlight), which every producer
//!   routes to its own deferral; the layer stays due and a later tick
//!   retries.
//!
//! A `None` probe (the unit-test lanes with no Valkey) skips the fence along
//! with the liveness probe.

use std::time::Instant;

use bytes::Bytes;
use fred::interfaces::HashesInterface;

use crate::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_DATA};
use crate::storage::keys::KeySchema;
use crate::storage::recon_fence::{recon_ext_rearm, RECON_FENCE_HORIZON};

use super::assembler::ext_refs_sidecar_hashes;
use super::checkpointer::reconcile_stem_member;
use super::store::{
    reconcile_grace, CheckpointError, CheckpointHandle, CheckpointMeta, CheckpointStore,
    CleanupInFlight,
};

/// The two borrowed handles the liveness probe needs, bundled so a producer
/// that is several frames away from the checkpoint loop can carry "can I ask
/// Valkey?" as one optional argument.
///
/// `Option<ObjectLivenessProbe>` is the shape the deeper producers take:
/// `None` means "no live keyspace is reachable from here", which is the
/// unit-test lane driving a fake `CheckpointStore` with no Valkey at all. A
/// `None` probe means the PUT is unguarded — exactly today's behaviour, which
/// is what makes those tests' expectations unchanged.
#[derive(Clone, Copy)]
pub(crate) struct ObjectLivenessProbe<'a> {
    client: &'a ValkeyClient,
    keys: &'a KeySchema,
}

impl<'a> ObjectLivenessProbe<'a> {
    pub(crate) fn new(client: &'a ValkeyClient, keys: &'a KeySchema) -> Self {
        Self { client, keys }
    }
}

/// What the probe learned. `Unknown` and `Live` are handled identically by
/// every caller (proceed) but are kept distinct so the tracing at the call site
/// can say which happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ObjectLiveness {
    /// `obj:hdr` carries a `seq` — the layer is live.
    Live,
    /// `obj:hdr` carries no `seq` — the layer is deleted, or its header is a
    /// pre-b.vn6 cursor-only phantom. Either way nothing should be written for
    /// it.
    Dead,
    /// The probe could not answer (Valkey error, or no probe was supplied).
    /// Callers proceed; see the module header.
    Unknown,
}

/// Run the advisory `HEXISTS obj:hdr seq` probe.
///
/// `context` names the producer (`"base bootstrap"`, `"assembler"`, …) and
/// appears in the debug event on the error arm; it is never used for control
/// flow.
///
/// # Pool: the data pool, deliberately — a documented exception
///
/// This probe issues its `HEXISTS` on the **data** pool (`client.pool`), while
/// `read_live_header_seq` (`src/checkpoint/checkpointer.rs`) — the retention
/// read that runs in the same checkpoint-loop tick, against the same `obj:hdr`
/// key — takes the **maintenance** pool. SDD §22.8's convention is that checkpoint-loop reads
/// use the maintenance pool so a slow retention pass cannot queue behind
/// request-path writers. This call site does not follow it, and that is
/// recorded here rather than silently diverging.
///
/// Why it is acceptable: the probe is one O(1) `HEXISTS` per **durable write**,
/// not per tick per layer, and it runs on a path that has just performed — and
/// is about to perform more — data-pool work of its own. The producer reached
/// this point by reading live state off the data pool (`read_header_identity`
/// for three of the four producers) and, on the base bootstrap, will follow the
/// PUT with the conditional cursor stamp (`stamp_cursor_if_live.lua`) on the
/// data pool too (the loop's own stamps use the maintenance pool). Routing the
/// one probe in the middle of that sequence to a different pool would buy no
/// isolation the surrounding reads do not already forfeit, and on the bootstrap
/// path it would cost the probe the property that actually matters to it:
/// observing the same connection's view of the keyspace as the stamp that
/// follows.
///
/// Why it is not simply changed: [`ObjectLivenessProbe`] is threaded through
/// four producers several frames from the loop, and switching pools would be an
/// API change (the maintenance pool is acquired with a guard token, not
/// borrowed), not a one-line substitution. If SDD §22.8's convention is ever
/// tightened into a rule, this is the call site to revisit.
pub(crate) async fn probe_object_liveness(
    probe: Option<ObjectLivenessProbe<'_>>,
    object_id: u64,
    context: &str,
) -> ObjectLiveness {
    let Some(probe) = probe else {
        return ObjectLiveness::Unknown;
    };
    let header_key = probe.keys.obj_header(&object_id.to_string());
    match probe
        .client
        .pool
        .hexists::<bool, _, _>(&header_key, "seq")
        .await
    {
        Ok(true) => ObjectLiveness::Live,
        Ok(false) => ObjectLiveness::Dead,
        Err(e) => {
            FredErrOrigin::client(probe.client, POOL_LABEL_DATA).observe_error(&e);
            tracing::debug!(
                object_id,
                context,
                error = %e,
                "pre-PUT obj:hdr liveness probe failed; proceeding to the PUT anyway — this \
                 probe is an optimisation and the conditional cursor stamp is the actual guard"
            );
            ObjectLiveness::Unknown
        }
    }
}

/// [`CheckpointStore::put_layer_checkpoint`], skipped when the target object is
/// provably dead, and fenced against the checkpoint clean-up (module docs,
/// "The clean-up fence").
///
/// `Ok(None)` means **nothing was written** — the store was never called. Each
/// caller maps that onto whatever it already does for the b.vn6 conditional
/// stamp's refusal, which for three of the four is
/// [`CheckpointError::NotFound`] (the checkpoint loop's benign-and-terminal
/// arm: `debug!`, no metric tick, no cursor stamp) and for the creation-time
/// bootstrap is its own `SkippedHeaderAbsent` outcome cell. No caller invents a
/// new metric for it: the object is gone, and "a layer we did not check-point
/// because it no longer exists" is not an adapter failure.
///
/// `reliance_start` is when the producer began reading the state it is about
/// to publish; the clean-up fence's lookback is measured from it. An
/// `Err` carrying [`CleanupInFlight`] (test with
/// [`CheckpointError::is_cleanup_in_flight`]) means the fence refused and
/// nothing was written; the producer defers.
///
/// `Ok(Some(handle))` and every other `Err` are the underlying call's,
/// verbatim.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn put_layer_checkpoint_if_live(
    probe: Option<ObjectLivenessProbe<'_>>,
    store: &dyn CheckpointStore,
    object_id: u64,
    snap_seq: u64,
    bytes: Bytes,
    meta: CheckpointMeta,
    reliance_start: Instant,
    context: &str,
) -> Result<Option<CheckpointHandle>, CheckpointError> {
    if probe_object_liveness(probe, object_id, context).await == ObjectLiveness::Dead {
        tracing::debug!(
            object_id,
            snap_seq,
            context,
            "obj:hdr vanished while this generation was being built (layer deleted or \
             overwritten within the build window); skipping before the PUT so no artifact is \
             written"
        );
        return Ok(None);
    }

    if let Some(probe) = probe {
        cleanup_fence(
            probe,
            store,
            object_id,
            snap_seq,
            &meta,
            reliance_start,
            context,
        )
        .await?;
    }

    store
        .put_layer_checkpoint(object_id, snap_seq, bytes, meta)
        .await
        .map(Some)
}

/// The clean-up fence for one generation (module docs): `Ok(())` lets the PUT
/// proceed; an `Err` carrying [`CleanupInFlight`] refuses it. A sidecar that
/// does not parse is the `Permanent` from
/// [`ext_refs_sidecar_hashes`](super::assembler::ext_refs_sidecar_hashes).
async fn cleanup_fence(
    probe: ObjectLivenessProbe<'_>,
    store: &dyn CheckpointStore,
    object_id: u64,
    snap_seq: u64,
    meta: &CheckpointMeta,
    reliance_start: Instant,
    context: &str,
) -> Result<(), CheckpointError> {
    let stem = reconcile_stem_member(object_id, snap_seq, meta.kind);
    let lookback = reliance_start.elapsed() + store.publish_worst_case();
    let refuse = |reason: String| {
        tracing::info!(
            object_id,
            snap_seq,
            context,
            stem = %stem,
            reason = %reason,
            "layer backup deferred by the checkpoint clean-up fence; nothing was written and a \
             later tick retries"
        );
        Err(CheckpointError::cleanup_in_flight(CleanupInFlight {
            reason,
        }))
    };
    if lookback > RECON_FENCE_HORIZON {
        return refuse(format!(
            "its reliance window ({} s) is longer than the clean-up's {} s record of deletes",
            lookback.as_secs(),
            RECON_FENCE_HORIZON.as_secs()
        ));
    }
    let hashes = match meta.sidecar_bytes.as_deref() {
        Some(sidecar) => ext_refs_sidecar_hashes(sidecar)?,
        None => Vec::new(),
    };
    let grace = reconcile_grace(store.publish_worst_case());
    match recon_ext_rearm(
        probe.client,
        probe.keys,
        &hashes,
        lookback,
        grace,
        Some(&stem),
    )
    .await
    {
        Ok(reply) if reply.stem_refused => refuse(format!(
            "the clean-up is deleting an earlier copy of generation {stem}"
        )),
        Ok(reply) if !reply.collected.is_empty() => refuse(format!(
            "the clean-up deleted the backup copy of {} large value(s) this generation names \
             within its reliance window",
            reply.collected.len()
        )),
        Ok(_) => Ok(()),
        // A Valkey error fails closed (P3). It is a store fault rather than a
        // clean-up in progress, so it logs at WARN, as the backup repair's
        // fence error does.
        Err(e) => {
            let reason = format!(
                "the clean-up bookkeeping that must precede the write could not be recorded in \
                 Valkey: {e}"
            );
            tracing::warn!(
                object_id,
                snap_seq,
                context,
                stem = %stem,
                reason = %reason,
                "layer backup deferred because the checkpoint clean-up fence could not be \
                 checked in Valkey; nothing was written and a later tick retries"
            );
            Err(CheckpointError::cleanup_in_flight(CleanupInFlight {
                reason,
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    //! The three probe verdicts, each driven against a REAL Valkey `obj:hdr`
    //! rather than a stub, because the whole predicate is a single Valkey
    //! command and a stub of it would be a test of the stub.
    //!
    //! "The store was never called" is read off an
    //! [`InMemoryCheckpointStore`](crate::checkpoint::adapters::InMemoryCheckpointStore)'s
    //! own listing rather than a call counter: a generation that was put is
    //! listed, one that was not is not, and the adapter is the same one the
    //! rest of this subsystem's unit lane uses. A counter would need a
    //! decorator implementing all eighteen trait methods to observe strictly
    //! less.

    use super::*;
    use crate::checkpoint::adapters::InMemoryCheckpointStore;
    use crate::checkpoint::store::CheckpointKind;
    use crate::storage::test_support::TestValkey;
    use fred::interfaces::KeysInterface;

    const OBJECT_ID: u64 = 4242;
    const SNAP_SEQ: u64 = 7;

    fn meta() -> CheckpointMeta {
        CheckpointMeta {
            snap_seq: Some(SNAP_SEQ),
            kind: CheckpointKind::Base,
            ..crate::checkpoint::adapters::identity_default()
        }
    }

    /// Write a live-looking `obj:hdr` — just the one field the probe reads.
    ///
    /// `seq` and not some other column: `stamp_cursor_if_live.lua` keys on the
    /// same field, and this helper exists so the two cannot drift.
    async fn plant_live_header(tv: &TestValkey) {
        use fred::interfaces::HashesInterface;
        let key = tv.key_schema().obj_header(&OBJECT_ID.to_string());
        let _: () = tv
            .client()
            .pool
            .hset(&key, ("seq", "1"))
            .await
            .expect("planting obj:hdr seq must succeed");
    }

    async fn generations(store: &InMemoryCheckpointStore) -> usize {
        store
            .list_layer_checkpoints(OBJECT_ID)
            .await
            .expect("listing an in-memory store cannot fail")
            .len()
    }

    /// `Live` → the PUT runs and the handle is the underlying call's.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_live_header_lets_the_put_through() {
        let tv = TestValkey::setup().await;
        plant_live_header(&tv).await;
        let store = InMemoryCheckpointStore::new();

        let handle = put_layer_checkpoint_if_live(
            Some(ObjectLivenessProbe::new(tv.client(), tv.key_schema())),
            &store,
            OBJECT_ID,
            SNAP_SEQ,
            Bytes::from_static(b"ompf"),
            meta(),
            Instant::now(),
            "unit-test",
        )
        .await
        .expect("a live object must not fail the guard")
        .expect("a live object must produce a handle");

        assert_eq!(handle.object_id, OBJECT_ID);
        assert_eq!(handle.snap_seq, SNAP_SEQ);
        assert_eq!(generations(&store).await, 1, "the generation is durable");
    }

    /// `Dead` → `Ok(None)`, and the store is never called at all.
    ///
    /// The second half is the point of the whole module: the value is the PUT
    /// that does not happen — a multi-MB write, a retention slot, and a round
    /// trip through the reconcile dead-object phase, all avoided. A `Ok(None)`
    /// that still wrote would satisfy the first assertion and none of the
    /// reason the guard exists.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dead_header_skips_the_put_entirely() {
        let tv = TestValkey::setup().await;
        // No header planted at all — the DeleteLayer / cleanup / orphan-sweep
        // outcome.
        let store = InMemoryCheckpointStore::new();

        let outcome = put_layer_checkpoint_if_live(
            Some(ObjectLivenessProbe::new(tv.client(), tv.key_schema())),
            &store,
            OBJECT_ID,
            SNAP_SEQ,
            Bytes::from_static(b"ompf"),
            meta(),
            Instant::now(),
            "unit-test",
        )
        .await
        .expect("a provably dead object is not an error — it is a skip");

        assert!(
            outcome.is_none(),
            "`Ok(None)` is the caller's signal to route onto its own \
             benign-and-terminal arm; `Ok(Some(..))` would mean an artifact landed"
        );
        assert_eq!(
            generations(&store).await,
            0,
            "the store MUST NOT have been called: the skipped write is the entire value of the \
             guard, and an artifact here would be durable garbage nobody ever reads"
        );
    }

    /// A Valkey ERROR on the probe → the PUT proceeds anyway.
    ///
    /// The asymmetry is deliberate and is the design (module header): the probe
    /// is an optimisation and the b.vn6 conditional cursor stamp is the actual
    /// guard, so failing a real generation on a transient blip would trade
    /// something valuable for nothing.
    ///
    /// The error is forced with a WRONGTYPE rather than by pointing at an
    /// unreachable Valkey: `obj:hdr` is planted as a STRING, so `HEXISTS`
    /// against it is a server-side error on a reachable connection. That keeps
    /// the case fast and deterministic, and it exercises the arm the way a real
    /// keyspace fault would rather than the way a connection outage would.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_probe_error_proceeds_to_the_put() {
        let tv = TestValkey::setup().await;
        let key = tv.key_schema().obj_header(&OBJECT_ID.to_string());
        let _: () = tv
            .client()
            .pool
            .set(&key, "not-a-hash", None, None, false)
            .await
            .expect("planting the wrong-typed key must succeed");

        assert_eq!(
            probe_object_liveness(
                Some(ObjectLivenessProbe::new(tv.client(), tv.key_schema())),
                OBJECT_ID,
                "unit-test",
            )
            .await,
            ObjectLiveness::Unknown,
            "a probe that could not answer must report `Unknown` — reporting `Dead` on an error \
             would let a blip silently drop a real generation"
        );

        let store = InMemoryCheckpointStore::new();
        let handle = put_layer_checkpoint_if_live(
            Some(ObjectLivenessProbe::new(tv.client(), tv.key_schema())),
            &store,
            OBJECT_ID,
            SNAP_SEQ,
            Bytes::from_static(b"ompf"),
            meta(),
            Instant::now(),
            "unit-test",
        )
        .await
        .expect("an unanswerable probe must not fail the put")
        .expect("an unanswerable probe must not skip the put either");
        assert_eq!(handle.object_id, OBJECT_ID);
        assert_eq!(generations(&store).await, 1);
    }

    // ── The clean-up fence (b.g65) ───────────────────────────────────────

    /// What [`FenceObservingStore`] read from Valkey at its `put` — i.e. at
    /// the moment the adapter PUT was called.
    #[derive(Debug, Clone, PartialEq)]
    struct SeenAtPut {
        /// `ZSCORE le:ckpt:{recon}:rearm <hash>` per observed hash.
        rearm: Vec<Option<f64>>,
        /// `ZSCORE le:ckpt:{recon}:orphans <stem>`.
        orphan: Option<f64>,
    }

    /// An in-memory store that, inside `put_layer_checkpoint` and before
    /// storing, reads the fence's Valkey writes for `hashes` and `stem` — the
    /// direct observation that the fence ran BEFORE the adapter PUT (decider
    /// obligation (a)) — and whose `publish_worst_case` is `pwc`.
    struct FenceObservingStore {
        inner: InMemoryCheckpointStore,
        client: ValkeyClient,
        keys: KeySchema,
        hashes: Vec<String>,
        stem: String,
        pwc: std::time::Duration,
        seen: std::sync::Mutex<Vec<SeenAtPut>>,
    }

    impl FenceObservingStore {
        fn new(tv: &TestValkey, hashes: &[&str], stem: &str) -> Self {
            Self {
                inner: InMemoryCheckpointStore::new(),
                client: tv.client().clone(),
                keys: tv.key_schema().clone(),
                hashes: hashes.iter().map(|h| (*h).to_owned()).collect(),
                stem: stem.to_owned(),
                pwc: crate::checkpoint::store::LOCAL_PUBLISH_WORST_CASE,
                seen: std::sync::Mutex::new(Vec::new()),
            }
        }

        fn seen(&self) -> Vec<SeenAtPut> {
            self.seen.lock().expect("seen mutex").clone()
        }
    }

    #[async_trait::async_trait]
    impl CheckpointStore for FenceObservingStore {
        async fn put_layer_checkpoint(
            &self,
            object_id: u64,
            snap_seq: u64,
            ompf: Bytes,
            meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            use fred::interfaces::SortedSetsInterface;
            let mut rearm = Vec::new();
            for hash in &self.hashes {
                let score: Option<f64> = self
                    .client
                    .pool
                    .zscore(self.keys.checkpoint_reconcile_rearm(), hash.as_str())
                    .await
                    .expect("ZSCORE :rearm");
                rearm.push(score);
            }
            let orphan: Option<f64> = self
                .client
                .pool
                .zscore(self.keys.checkpoint_reconcile_orphans(), self.stem.as_str())
                .await
                .expect("ZSCORE :orphans");
            self.seen
                .lock()
                .expect("seen mutex")
                .push(SeenAtPut { rearm, orphan });
            self.inner
                .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
                .await
        }
        async fn get_latest_layer_checkpoint(
            &self,
            object_id: u64,
        ) -> Result<Option<crate::checkpoint::store::CheckpointBlob>, CheckpointError> {
            self.inner.get_latest_layer_checkpoint(object_id).await
        }
        async fn list_layer_checkpoints(
            &self,
            object_id: u64,
        ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
            self.inner.list_layer_checkpoints(object_id).await
        }
        async fn delete_layer_checkpoint(
            &self,
            handle: CheckpointHandle,
        ) -> Result<(), CheckpointError> {
            self.inner.delete_layer_checkpoint(handle).await
        }
        async fn put_session_checkpoint(
            &self,
            session_id: &str,
            body: Bytes,
            meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            self.inner
                .put_session_checkpoint(session_id, body, meta)
                .await
        }
        async fn get_session_checkpoint(
            &self,
            session_id: &str,
        ) -> Result<Option<crate::checkpoint::store::CheckpointBlob>, CheckpointError> {
            self.inner.get_session_checkpoint(session_id).await
        }
        async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
            self.inner.delete_session_checkpoint(session_id).await
        }
        async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
            self.inner.put_ext_blob(hash_hex, bytes).await
        }
        async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
            self.inner.get_ext_blob(hash_hex).await
        }
        async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
            self.inner.delete_ext_blob(hash_hex).await
        }
        async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
            self.inner.put_counters_sidecar(body).await
        }
        async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
            self.inner.get_counters_sidecar().await
        }
        async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
            self.inner.list_all_object_ids().await
        }
        fn publish_worst_case(&self) -> std::time::Duration {
            self.pwc
        }
    }

    /// A base meta whose ext-refs sidecar names `hashes`.
    fn base_meta_naming(hashes: &[&str]) -> CheckpointMeta {
        CheckpointMeta {
            sidecar_bytes: Some(Bytes::from(
                serde_json::to_vec(&serde_json::json!({ "ext_hashes": hashes }))
                    .expect("sidecar JSON encodes"),
            )),
            ..meta()
        }
    }

    /// `members` on `key` at `score_ms`.
    async fn zadd_at(tv: &TestValkey, key: &str, score_ms: u64, members: &[&str]) {
        use fred::interfaces::SortedSetsInterface;
        for member in members {
            let _: () = tv
                .client()
                .pool
                .zadd(key, None, None, false, false, (score_ms as f64, *member))
                .await
                .expect("ZADD must succeed");
        }
    }

    async fn zscore(tv: &TestValkey, key: &str, member: &str) -> Option<f64> {
        use fred::interfaces::SortedSetsInterface;
        tv.client()
            .pool
            .zscore(key, member)
            .await
            .expect("ZSCORE must succeed")
    }

    async fn now_ms(tv: &TestValkey) -> u64 {
        crate::storage::ext_backup::valkey_time_ms(tv.client())
            .await
            .expect("TIME must succeed")
    }

    async fn publish(
        tv: &TestValkey,
        store: &dyn CheckpointStore,
        meta: CheckpointMeta,
    ) -> Result<Option<CheckpointHandle>, CheckpointError> {
        put_layer_checkpoint_if_live(
            Some(ObjectLivenessProbe::new(tv.client(), tv.key_schema())),
            store,
            OBJECT_ID,
            SNAP_SEQ,
            Bytes::from_static(b"ompf"),
            meta,
            Instant::now(),
            "unit-test",
        )
        .await
    }

    /// R2 at the guard (b.g65; decider obligation (a) for the one choke point
    /// all four producers publish through): BEFORE the adapter PUT is called,
    /// every tombstoned hash the sidecar names is re-armed on
    /// `le:ckpt:{recon}:rearm` — and only tombstoned ones — and the
    /// generation's own orphan member is gone from `le:ckpt:{recon}:orphans`
    /// (a publish heals its orphan candidate). The store reads both inside its
    /// `put`. A re-arm after the PUT would let a collect whose snapshot
    /// predates the sidecar delete a value the new backup names.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_fence_re_arms_the_tombstoned_hashes_and_heals_the_orphan_member_before_the_put() {
        let tv = TestValkey::setup().await;
        plant_live_header(&tv).await;
        let keys = tv.key_schema();
        zadd_at(
            &tv,
            &keys.checkpoint_reconcile_tombstones(),
            1,
            &["tombstoned"],
        )
        .await;
        let stem = format!("{OBJECT_ID}.{SNAP_SEQ}");
        zadd_at(
            &tv,
            &keys.checkpoint_reconcile_orphans(),
            1,
            &[stem.as_str()],
        )
        .await;
        let store = FenceObservingStore::new(&tv, &["tombstoned", "untombstoned"], &stem);
        let before = now_ms(&tv).await;

        publish(
            &tv,
            &store,
            base_meta_naming(&["tombstoned", "untombstoned"]),
        )
        .await
        .expect("an unfenced publish is not refused")
        .expect("and it PUTs");

        let seen = store.seen();
        assert_eq!(seen.len(), 1, "one PUT");
        let rearm = &seen[0].rearm;
        assert!(
            rearm[0].is_some_and(|s| s >= before as f64),
            "the tombstoned hash is re-armed at the Valkey clock before the PUT: {seen:?}"
        );
        assert_eq!(
            rearm[1], None,
            "an untombstoned hash cannot be collected, so it is not re-armed"
        );
        assert_eq!(
            seen[0].orphan, None,
            "the publish removed its own orphan member before the PUT"
        );
        assert_eq!(
            zscore(&tv, &keys.checkpoint_reconcile_tombstones(), "tombstoned").await,
            Some(1.0),
            "the fence never re-scores the tombstone itself"
        );
    }

    /// A tail publish heals its own TAIL orphan member (`<oid>.<seq>.tail`)
    /// and re-arms nothing: a tail names no ext hashes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_tail_publish_heals_its_tail_orphan_member() {
        let tv = TestValkey::setup().await;
        plant_live_header(&tv).await;
        let keys = tv.key_schema();
        let tail_stem = format!("{OBJECT_ID}.{SNAP_SEQ}.tail");
        let base_stem = format!("{OBJECT_ID}.{SNAP_SEQ}");
        zadd_at(
            &tv,
            &keys.checkpoint_reconcile_orphans(),
            1,
            &[tail_stem.as_str(), base_stem.as_str()],
        )
        .await;
        let store = FenceObservingStore::new(&tv, &[], &tail_stem);
        let tail = CheckpointMeta {
            kind: CheckpointKind::Tail,
            base_snap_seq: Some(1),
            from_seq: Some(2),
            snap_seq: Some(SNAP_SEQ),
            ..crate::checkpoint::adapters::identity_default()
        };

        publish(&tv, &store, tail)
            .await
            .expect("a tail publish")
            .expect("and it PUTs");

        assert_eq!(
            store.seen()[0].orphan,
            None,
            "the tail member is healed before the PUT"
        );
        assert!(
            zscore(&tv, &keys.checkpoint_reconcile_orphans(), &base_stem)
                .await
                .is_some(),
            "and the base member at the same sequence is another candidate, left alone"
        );
    }

    /// R7 at the guard: a hash the collect licensed for deletion inside the
    /// producer's reliance window (`:collected` scored now) refuses the PUT
    /// with the typed `CleanupInFlight` deferral and writes nothing; once the
    /// record is older than the window, the same publish goes through.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_fence_refuses_a_put_naming_a_value_collected_inside_its_reliance_window() {
        let tv = TestValkey::setup().await;
        plant_live_header(&tv).await;
        let keys = tv.key_schema();
        let now = now_ms(&tv).await;
        zadd_at(
            &tv,
            &keys.checkpoint_reconcile_collected(),
            now,
            &["collected"],
        )
        .await;
        let stem = format!("{OBJECT_ID}.{SNAP_SEQ}");
        let store = FenceObservingStore::new(&tv, &[], &stem);

        let refused = publish(&tv, &store, base_meta_naming(&["collected"])).await;
        let err = refused.expect_err("a value collected inside the window refuses the PUT");
        assert!(err.is_cleanup_in_flight(), "the typed deferral: {err:?}");
        assert!(store.seen().is_empty(), "the store was never called");

        // Ten minutes ago: past the window (the time since the reliance start
        // plus this store's publish worst case).
        zadd_at(
            &tv,
            &keys.checkpoint_reconcile_collected(),
            now - 600_000,
            &["collected"],
        )
        .await;
        publish(&tv, &store, base_meta_naming(&["collected"]))
            .await
            .expect("past the window the publish is not refused")
            .expect("and it PUTs");
        assert_eq!(store.seen().len(), 1);
    }

    /// N3 at the guard: a stem a deleter announced on
    /// `le:ckpt:{recon}:deleting` within the grace refuses the PUT and writes
    /// nothing for the stem (its orphan member stays); a stale announcement,
    /// older than the grace, lets the PUT through and is removed with the
    /// orphan member, so a later not-found of the new copy is never
    /// explained away as an announced deletion.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_fence_refuses_a_stem_announced_within_the_grace_and_clears_a_stale_announcement() {
        let tv = TestValkey::setup().await;
        plant_live_header(&tv).await;
        let keys = tv.key_schema();
        let stem = format!("{OBJECT_ID}.{SNAP_SEQ}");
        let store = FenceObservingStore::new(&tv, &[], &stem);
        let grace_ms = crate::checkpoint::store::reconcile_grace(store.publish_worst_case())
            .as_millis() as u64;
        let now = now_ms(&tv).await;
        zadd_at(
            &tv,
            &keys.checkpoint_reconcile_orphans(),
            1,
            &[stem.as_str()],
        )
        .await;
        zadd_at(
            &tv,
            &keys.checkpoint_reconcile_deleting(),
            now,
            &[stem.as_str()],
        )
        .await;

        let err = publish(&tv, &store, base_meta_naming(&[]))
            .await
            .expect_err("an announced stem refuses the PUT");
        assert!(err.is_cleanup_in_flight(), "{err:?}");
        assert!(store.seen().is_empty(), "the store was never called");
        assert!(
            zscore(&tv, &keys.checkpoint_reconcile_orphans(), &stem)
                .await
                .is_some(),
            "a refused stem writes nothing for the stem: its orphan member stays"
        );

        zadd_at(
            &tv,
            &keys.checkpoint_reconcile_deleting(),
            now - grace_ms - 1_000,
            &[stem.as_str()],
        )
        .await;
        publish(&tv, &store, base_meta_naming(&[]))
            .await
            .expect("a stale announcement does not refuse")
            .expect("and it PUTs");
        assert_eq!(
            zscore(&tv, &keys.checkpoint_reconcile_deleting(), &stem).await,
            None,
            "the stale announcement is removed"
        );
        assert_eq!(
            zscore(&tv, &keys.checkpoint_reconcile_orphans(), &stem).await,
            None,
            "and the orphan member healed"
        );
    }

    /// P3 at the guard: a fence that cannot record its re-arm refuses the PUT
    /// (fail closed) — here `le:ckpt:{recon}:rearm` holds a string, so the
    /// script's opener answers a real `WRONGTYPE`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_fence_that_cannot_record_its_re_arm_refuses_the_put() {
        let tv = TestValkey::setup().await;
        plant_live_header(&tv).await;
        let _: () = tv
            .client()
            .pool
            .set(
                tv.key_schema().checkpoint_reconcile_rearm(),
                "not-a-zset",
                None,
                None,
                false,
            )
            .await
            .expect("poisoning :rearm must succeed");
        let store = FenceObservingStore::new(&tv, &[], "unused");

        let err = publish(&tv, &store, base_meta_naming(&["h"]))
            .await
            .expect_err("an unrecordable re-arm refuses the PUT");
        assert!(err.is_cleanup_in_flight(), "{err:?}");
        assert!(store.seen().is_empty(), "nothing was written unprotected");
    }

    /// R8 at the guard: a reliance window longer than the collect's 24 h
    /// record of deletes cannot be answered, so the publish refuses rather
    /// than guess. The store's publish worst case alone is 25 h here.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_reliance_window_longer_than_the_collect_record_refuses_the_put() {
        let tv = TestValkey::setup().await;
        plant_live_header(&tv).await;
        let mut store = FenceObservingStore::new(&tv, &[], "unused");
        store.pwc = std::time::Duration::from_secs(25 * 60 * 60);

        let err = publish(&tv, &store, base_meta_naming(&[]))
            .await
            .expect_err("an unanswerable window refuses");
        assert!(err.is_cleanup_in_flight(), "{err:?}");
        assert!(store.seen().is_empty());
    }

    /// No probe at all → `Unknown`, and the PUT proceeds.
    ///
    /// This is the shape every fake-store unit lane passes
    /// (`capture_layer_tail_with`, `rebase_layer_lineage`), so it is what makes
    /// those tests' expectations unchanged by this module's existence. Asserted
    /// rather than assumed, because a `None` that defaulted to `Dead` would
    /// silently turn every one of them into a no-op that still passed its
    /// `Ok(_)` assertions.
    #[tokio::test]
    async fn an_absent_probe_is_unknown_and_never_skips() {
        assert_eq!(
            probe_object_liveness(None, OBJECT_ID, "unit-test").await,
            ObjectLiveness::Unknown
        );

        let store = InMemoryCheckpointStore::new();
        let handle = put_layer_checkpoint_if_live(
            None,
            &store,
            OBJECT_ID,
            SNAP_SEQ,
            Bytes::from_static(b"ompf"),
            meta(),
            Instant::now(),
            "unit-test",
        )
        .await
        .expect("no probe means no guard")
        .expect("no probe means the put runs");
        assert_eq!(handle.object_id, OBJECT_ID);
        assert_eq!(generations(&store).await, 1);
    }
}
