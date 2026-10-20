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

//! Checkpoint store value types, error enum, adapter-selection enum, and the
//! [`CheckpointStore`] async trait.
//!
//! These are the abstractions that `Config::from_env` parsing of
//! `LIVE_EDIT_CHECKPOINT_ADAPTER`, the periodic checkpoint loop, the recovery
//! hook, and the SR-1.3 conformance harness all build on. No I/O and no
//! adapter implementations live here — see SDD SR-1.1 / SR-1.3 for the
//! contract.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;
use std::str::FromStr;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error contract for every `CheckpointStore` operation.
///
/// The four classified variants (`Transient`, `Permanent`, `NotFound`,
/// `Conflict`) are the contract that the SR-1.3 conformance harness verifies
/// (tests #19-#22). `NotImplemented` is the off-mode / unimplemented-override
/// signal: the `Disabled` adapter returns it from every method, and the trait
/// defaults of [`CheckpointStore::get_layer_checkpoint`],
/// [`CheckpointStore::list_session_checkpoint_ids`] and
/// [`CheckpointStore::ext_blob_exists`] return it so an impl that forgot to
/// override is loud rather than silently absent. See the variant's own docs
/// and the trait-level error contract.
#[derive(Debug, Error)]
pub enum CheckpointError {
    /// Transient backend failure — callers SHOULD retry with backoff.
    #[error("transient checkpoint-store error: {source}")]
    Transient {
        source: Box<dyn Error + Send + Sync>,
    },

    /// Permanent backend failure — callers MUST NOT retry; surface to the
    /// operator.
    #[error("permanent checkpoint-store error: {source}")]
    Permanent {
        source: Box<dyn Error + Send + Sync>,
    },

    /// Requested handle does not exist in the backing store.
    #[error("checkpoint not found")]
    NotFound,

    /// Optimistic-concurrency conflict — a competing writer raced and won.
    #[error("checkpoint conflict")]
    Conflict,

    /// This store does not implement this method. Two shipped kinds of
    /// producer, and the enumeration is exhaustive:
    ///
    /// - every method of the `Disabled` (no-op) adapter, so an off-mode caller
    ///   that drives a checkpoint path fails loudly instead of appearing to
    ///   succeed;
    /// - the default impls of [`CheckpointStore::get_layer_checkpoint`],
    ///   [`CheckpointStore::list_session_checkpoint_ids`] and
    ///   [`CheckpointStore::ext_blob_exists`], which every real adapter and
    ///   every forwarding decorator is expected to override — inheriting one
    ///   is a bug, and this variant is what makes that bug loud instead of an
    ///   `Ok(None)`, an empty list or an `Ok(false)` that reads as "absent".
    ///
    /// Generation kind is **not** a producer. All three adapters (`InMemory`,
    /// `LocalDisk`, `StorageApi`) accept a [`CheckpointKind::Tail`]
    /// `put_layer_checkpoint` and store what they were given, so no adapter
    /// answers a tail put with this variant.
    ///
    /// `method` is always the **real** method name, never a synthetic
    /// context-qualified label; the surrounding tracing carries the context.
    #[error("checkpoint method {method} not implemented for this adapter")]
    NotImplemented { method: &'static str },
}

impl CheckpointError {
    /// True iff the error is `Transient`. Callers SHOULD retry with backoff.
    pub fn is_transient(&self) -> bool {
        matches!(self, CheckpointError::Transient { .. })
    }

    /// True iff the error is `Permanent`. Callers MUST NOT retry.
    pub fn is_permanent(&self) -> bool {
        matches!(self, CheckpointError::Permanent { .. })
    }

    /// True iff the error is `NotFound`.
    pub fn is_not_found(&self) -> bool {
        matches!(self, CheckpointError::NotFound)
    }

    /// True iff the error is `Conflict`.
    pub fn is_conflict(&self) -> bool {
        matches!(self, CheckpointError::Conflict)
    }

    /// Build the [`Permanent`](CheckpointError::Permanent) that every adapter
    /// returns when a generation's artifact is present but its meta companion
    /// is absent. See [`MissingMetaCompanion`] for why this is `Permanent` and
    /// why it is typed.
    pub fn missing_meta_companion(marker: MissingMetaCompanion) -> Self {
        CheckpointError::Permanent {
            source: Box::new(marker),
        }
    }

    /// True iff this is the artifact-present / meta-companion-absent
    /// `Permanent` built by [`Self::missing_meta_companion`].
    ///
    /// This is what lets `checkpoint::lineage`'s restore-lineage gather single
    /// out *this one* `Permanent` and step down to the next-older restorable
    /// generation, while every other `Permanent` — auth, malformed response,
    /// store outage — still propagates unchanged. Matching on the error
    /// *message* would be the alternative and is exactly the coupling this
    /// exists to avoid: a message reworded for the operator would silently
    /// convert a step-down into a hard restore failure, or (worse) an auth
    /// outage into a "layer not found".
    pub fn is_missing_meta_companion(&self) -> bool {
        match self {
            CheckpointError::Permanent { source } => {
                source.downcast_ref::<MissingMetaCompanion>().is_some()
            }
            _ => false,
        }
    }

    /// Build the [`Transient`](CheckpointError::Transient) that
    /// `list_referenced_ext_hashes` returns when an ext-refs sidecar its own
    /// enumeration named reads absent by address. See [`ListedSidecarAbsent`]
    /// for why this is `Transient` and why it is typed.
    pub fn listed_sidecar_absent(marker: ListedSidecarAbsent) -> Self {
        CheckpointError::Transient {
            source: Box::new(marker),
        }
    }

    /// True iff this is the listed-sidecar-absent `Transient` built by
    /// [`Self::listed_sidecar_absent`].
    ///
    /// This is what lets the reconciliation collect phase
    /// (`checkpointer::run_reconcile_collect`) qualify this one live-set
    /// failure by who holds the checkpoint lease, while every other live-set
    /// failure still counts unconditionally. Same reason as
    /// [`Self::is_missing_meta_companion`]: code must not branch on
    /// operator-facing wording.
    pub fn is_listed_sidecar_absent(&self) -> bool {
        match self {
            CheckpointError::Transient { source } => {
                source.downcast_ref::<ListedSidecarAbsent>().is_some()
            }
            _ => false,
        }
    }
}

/// What [`CheckpointStore::list_live_ext_set`] returns: the live ext-hash set
/// and the evidence, from the same enumeration, that licenses deleting
/// against it (b.oya).
///
/// Built only by the adapters' live-set walks and the trait's fail-closed
/// default, read only by the reconciliation collect phase, never mutated,
/// never persisted.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LiveExtSet {
    /// Every hash a stored base's ext-refs sidecar names — exactly what
    /// [`CheckpointStore::list_referenced_ext_hashes`] returns.
    pub hashes: HashSet<String>,
    /// Layer artifacts (names [`parse_layer_artifact_filename`] accepts, any
    /// kind) that this walk's OWN enumeration named — the same request
    /// `hashes` came from. 0 when it named none, or when the implementation
    /// cannot vouch for one (the trait default). An empty `hashes` licenses a
    /// delete only when this is non-zero.
    pub listed_layer_artifacts: usize,
}

/// Typed marker carried as the `source` of the
/// [`CheckpointError::Transient`] that
/// [`CheckpointStore::list_referenced_ext_hashes`] and
/// [`CheckpointStore::list_live_ext_set`] (one walk) return when an ext-refs
/// sidecar its own enumeration named reads absent by address.
///
/// # Why an error, and why `Transient`
///
/// Skipping the sidecar would drop its hashes from the live set, and an
/// under-counted live set is an over-delete: the reconciliation collect phase
/// deletes every tombstoned blob the set does not name. So the walk returns an
/// error and never a partial set. It is `Transient` because the next pass
/// lists again and reads a consistent store.
///
/// # What explains it
///
/// While another replica holds the checkpoint lease, that replica's
/// lease-gated retention prune deletes sidecars as a matter of course, so a
/// sidecar listed and then pruned under the walk is routine. While this
/// replica still holds the lease, the absence is explained only by a deleter
/// outside the lease (the base bootstrap's reap on a create/delete race, or
/// `rollback_partial_publish`), a peer that led during the walk and then
/// stopped refreshing the lock, or the store answering inconsistently (a stale
/// not-found answer, or a listing that lags deletes).
///
/// # Why it is a type and not just a message
///
/// [`CheckpointError::is_listed_sidecar_absent`] downcasts to this, which is
/// how the collect phase tells this failure apart from every other live-set
/// failure and qualifies it by lease ownership, without string-matching an
/// operator-facing message. `checkpointer::checkpoint_error_kind` still reports
/// it as `transient`.
#[derive(Debug, Clone)]
pub struct ListedSidecarAbsent {
    /// Adapter-native address of the sidecar that was listed and then read
    /// absent — a filesystem path on LocalDisk, an absolute URL on StorageApi.
    pub artifact: String,
}

impl fmt::Display for ListedSidecarAbsent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ext-refs sidecar {} was named by the layers listing but read as absent; the live \
             ext-hash set is not trustworthy this pass, so it is not returned",
            self.artifact
        )
    }
}

impl Error for ListedSidecarAbsent {}

/// Typed marker carried as the `source` of the
/// [`CheckpointError::Permanent`] an adapter returns when a generation's
/// artifact is durable but its meta companion (`<stem>.meta.json` /
/// `<stem>.tail.meta.json`) is absent.
///
/// # Why `Permanent` rather than a placeholder meta
///
/// Both the LocalDisk and StorageApi read paths used to synthesize
/// `CheckpointMeta::default()` here and hand back `Ok(Some(..))`. That meta
/// carries an empty `session_id` / `layer_name`, and since b.b2j the apply
/// path's identity guard (`checkpoint::apply::live_mapping_resolves_blob_identity`)
/// refuses a blob it cannot identify — so the placeholder bought nothing but a
/// generic `NotFound` raised three layers above the adapter that already knew
/// exactly what was wrong. Refusing here puts the diagnosis at the source.
///
/// The "legacy pre-meta-companion generation" the placeholder existed for is an
/// empty population: no release tag predates the meta companion (all tags are
/// pre-release), so no deployed generation predates it either. Both kinds are refused for the same
/// reason (the b.hka (d) decision that supersedes Epic 2's
/// mirror-the-base-arm pin): under the shipped meta-first publish and
/// artifact-first delete ordering, artifact-present / meta-absent is
/// unreachable in both windows on both kinds.
///
/// One further producer was **narrowed** in the same review that made this a
/// `Permanent` (b.hka round 2): a tail publish whose envelope PUT failed used to
/// DELETE its meta companion unconditionally, which under duplicate leadership
/// could take the companion of a *peer's* already-durable tail at the same
/// `to_seq`. Both tail-rollback sites now probe for a durable envelope first and
/// skip the DELETE when one exists
/// (`StorageApiCheckpointStore::rollback_partial_tail_publish`, and the
/// LocalDisk arm's equivalent ordering). That matters because the shape is a
/// lost recovery point, not a degraded read: `checkpoint::lineage` steps PAST a
/// generation that reports this marker.
///
/// **Narrowed, not closed** — the same honesty the `checkpoint::put_guard`
/// module header carries in its own residual section about its own probe
/// applies verbatim here, and for the same structural reason. Each probe
/// is one advisory read followed by a *separate* DELETE, with no atomicity
/// between them, so this interleaving survives: leader B probes and sees no
/// peer envelope (A's PUT is not durable yet) → B DELETEs the shared meta
/// address → A's envelope PUT lands. The result is artifact-present /
/// meta-absent for a tail that is otherwise perfectly good. Re-probing
/// immediately before the DELETE shrinks the window again but cannot close it;
/// only a compare-and-delete the store does not offer would.
///
/// What remains, therefore, is: store corruption; a torn rolled-back put that
/// left an artifact with no durable envelope to protect it (the probe skips only
/// when the artifact IS durable); out-of-band removal or tampering; or — rarely,
/// and only under duplicate leadership — the probe-then-DELETE interleaving
/// above. None of them is improved by a fabricated identity-empty meta.
///
/// It is `Permanent` and not `Transient`: the companion's absence is a durable
/// fact about the store, and a retry re-reads the same absence.
///
/// # Why it is a type and not just a message
///
/// [`CheckpointError::is_missing_meta_companion`] downcasts to this, which is
/// how `checkpoint::lineage` recognises the one `Permanent` it is allowed to
/// treat as "this generation is unusable, try the next-older one" without
/// string-matching an operator-facing message.
#[derive(Debug, Clone)]
pub struct MissingMetaCompanion {
    /// Adapter-native address of the artifact that IS present — a filesystem
    /// path on LocalDisk, an absolute URL on StorageApi.
    pub artifact: String,
    /// Adapter-native address of the meta companion that is NOT present.
    pub companion: String,
    /// The addressed generation's object id.
    pub object_id: u64,
    /// The addressed generation's sequence (`snap_seq` for a base, `to_seq` for
    /// a tail).
    pub sequence: u64,
    /// The addressed generation's kind.
    pub kind: CheckpointKind,
}

impl fmt::Display for MissingMetaCompanion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "checkpoint generation object_id={} sequence={} kind={} is unrestorable: its \
             artifact {} is present but its meta companion {} is absent, so the generation \
             carries no session_id/layer_name and the restore path cannot verify which layer \
             it belongs to. Publish order writes the meta before the artifact and delete order \
             removes the artifact before the meta, so this state is unreachable in normal \
             operation — it indicates store corruption, a torn rolled-back put, or out-of-band \
             removal. The fix is to let the checkpoint loop write a current generation WITH its \
             meta sidecar; do not relax the guard",
            self.object_id,
            self.sequence,
            self.kind.as_str(),
            self.artifact,
            self.companion,
        )
    }
}

impl Error for MissingMetaCompanion {}

/// Which kind of generation a checkpoint artifact is (SR-2.3).
///
/// Before tails existed, `(object_id, snap_seq)` was the whole identity of a
/// generation and every artifact was a full-state snapshot. A lineage now
/// interleaves two kinds under one object id, so the kind is what tells a
/// reader whether an artifact stands alone ([`CheckpointKind::Base`]) or must be
/// replayed on top of the base it names ([`CheckpointKind::Tail`]).
///
/// `Base` is the [`Default`], which is what makes the persisted form
/// backward-compatible: a `<oid>.<snap_seq>.meta.json` written before this
/// feature has no `kind` key at all and deserializes as `Base` with no lineage
/// (PRD A-6 / SR-2.5). No migration, no version bump on the meta sidecar.
///
/// The serialized form is the lowercase strings `"base"` / `"tail"`. An
/// **unknown** kind string is a `serde` parse failure, not a guess: both
/// adapters' meta-read arms already classify a `CheckpointMeta` parse error as
/// [`CheckpointError::Permanent`] (`local_disk::get_latest_layer_checkpoint`'s
/// `serde_json::from_slice` arm, `storage_api::get_latest_layer_checkpoint`'s
/// equivalent), so an artifact written by a *newer* build surfaces to the
/// operator instead of being silently downgraded to a base and replayed wrong.
/// That matches SR-6.1's posture for the tail envelope's `format_version`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointKind {
    /// A full-state snapshot. Restorable on its own, with no lineage.
    #[default]
    Base,
    /// A tail envelope carrying the commits captured after a base. Restorable
    /// only by replaying it onto the base named by its lineage fields.
    Tail,
}

impl CheckpointKind {
    /// The persisted / tracing spelling — `"base"` or `"tail"`.
    ///
    /// This is the same string the `Serialize` impl emits, and the one Tasks 3
    /// and 4 map onto adapter filename suffixes, so kind spellings cannot drift
    /// between the meta JSON, a tracing field, and an artifact name.
    pub fn as_str(&self) -> &'static str {
        match self {
            CheckpointKind::Base => "base",
            CheckpointKind::Tail => "tail",
        }
    }
}

/// Provenance metadata stamped on every checkpoint blob, plus the optional
/// paired ext-refs sidecar bytes the assembler emits alongside the OMPF.
///
/// `creator_replica_id` identifies the writer (typically `LIVE_EDIT_SERVICE_LABEL`
/// or the pod hostname); `written_at_unix_ms` is the wall-clock write time;
/// `snap_seq` is the per-object snapshot sequence number when the blob is a
/// layer-state snapshot, and `None` for non-layer blobs (e.g. future session
/// checkpoints in Epic 2).
///
/// `session_id`, `layer_name`, `created_by`, and `server_ts` are the
/// session-level identity fields the apply / recovery path needs to rebuild
/// `obj:hdr` on `LayerStore::apply_checkpoint`, and `observed_commit_span` is
/// the measured DR guard band `apply_checkpoint` consumes to size its sequence
/// advance (b.vnw). All five are read from `obj:hdr` by the assembler at write
/// time and persisted alongside the OMPF (in the `InMemory` adapter as
/// `LayerEntry` fields, in the `LocalDisk` adapter inside the
/// `<oid>.<snap_seq>.meta.json` sidecar) so `get_latest_layer_checkpoint` can
/// round-trip them verbatim. Adapters MUST persist all five: dropping an
/// identity field forces the apply path to invent session-level metadata, and
/// dropping `observed_commit_span` silently degrades every restore of a hot
/// layer back to the configured margin (a *smaller* band — the b.8y5 hazard
/// direction). The session-checkpoint and counters-sidecar puts ignore them
/// entirely; only `put_layer_checkpoint` consults them.
///
/// `sidecar_bytes` carries the ext-refs sidecar JSON paired with the OMPF when
/// `put_layer_checkpoint` is the call site. Adapters that observe `Some(_)` MUST
/// write the sidecar atomically alongside the OMPF so a partial-write failure
/// cannot leave a checkpoint visible without its referenced-blob index. `None`
/// (the default) leaves the put as OMPF-only — callers that pair the OMPF and
/// sidecar via a later adapter-specific `set_layer_sidecar` (the legacy path)
/// continue to work unchanged. The session-checkpoint and counters-sidecar puts
/// ignore this field entirely; only `put_layer_checkpoint` consults it.
///
/// # Lockstep with the tail-envelope header — a cross-module invariant
///
/// The lineage fields below (`kind`, `base_snap_seq`, `from_seq`, and `snap_seq`
/// in its tail role) are one half of a coupling whose other half is
/// `TailHeader` in **`src/checkpoint/tail_codec.rs`**. Named in prose rather
/// than as an intra-doc link on purpose: the two modules deliberately do not
/// import each other, and a link would make one a build-time dependency of the
/// other's docs.
///
/// This is the same shape of invariant SDD §22.11 records for
/// `SessionCheckpoint` / `SessionStore::create_once` — a contract that lives in
/// neither module alone, and is therefore written down on **both** sides so a
/// reader arriving at either one sees it. Changing one side without the other is
/// a contract violation, not a refactor.
///
/// The split of duty:
///
/// - **This meta serves listing and retention.** It is what
///   [`CheckpointStore::list_layer_checkpoints`] and the retention prune read, so
///   a generation's kind and lineage are answerable without fetching and
///   decoding the artifact's bytes.
/// - **The envelope header is authoritative at replay time.** Replay reads
///   lineage out of the bytes it is about to apply, so a meta that disagrees with
///   its envelope loses — the meta is an index, not a source of truth.
/// - **Both carry the same identity**, and the field names match deliberately so
///   the correspondence is checkable by eye.
///
/// ## The one name that differs: `to_seq` ⇄ `snap_seq`
///
/// `TailHeader` carries `to_seq` — the inclusive upper bound of the commit span
/// the envelope holds. There is deliberately **no** `to_seq` field on this
/// struct: that number lands in `snap_seq` here and in
/// [`CheckpointHandle::snap_seq`] on the handle, so both kinds address, sort, and
/// name their artifacts through one field (see [`CheckpointHandle::snap_seq`] for
/// why). Read the mapping as:
///
/// ```text
/// TailHeader::object_id      ⇄  CheckpointHandle::object_id
/// TailHeader::base_snap_seq  ⇄  CheckpointMeta::base_snap_seq  (Some for a tail)
/// TailHeader::from_seq       ⇄  CheckpointMeta::from_seq       (Some for a tail)
/// TailHeader::to_seq         ⇄  CheckpointMeta::snap_seq  ==  CheckpointHandle::snap_seq
/// ```
///
/// ## Tail metas are NOT identity-empty
///
/// A tail's meta carries the **same `session_id` / `layer_name` identity fields
/// as the base it is rooted at** — they are not `None`-in-spirit just because the
/// envelope header does not repeat them. `TailHeader` deliberately carries no
/// identity; identity at replay time is **base-sourced**, read off the base
/// generation's meta.
///
/// This is pinned rather than left to convention because
/// `apply::live_mapping_resolves_blob_identity` refuses any blob with an empty
/// `session_id` or `layer_name` outright — it cannot verify such a blob against
/// the live layer mapping, so it declines to materialize an identity-less object
/// header. Epic 4's composed restore must therefore never route an
/// identity-empty tail meta into that guard: it would surface as a flat refusal
/// to restore, attributed to the guard rather than to the tail writer that left
/// the fields blank.
///
/// `Serialize` / `Deserialize` are derived so the `LocalDisk` adapter can
/// persist the meta as a JSON sidecar (`<oid>.<snap_seq>.meta.json`).
/// `sidecar_bytes` is skipped on the wire: the ext-refs sidecar is itself a
/// separate `<oid>.<snap_seq>.ext-refs.json` file on the LocalDisk layout, so
/// round-tripping it through the meta JSON would duplicate the on-disk
/// payload. `InMemory` doesn't serialize at all — it keeps `CheckpointMeta`
/// in-memory verbatim, `sidecar_bytes` included.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub creator_replica_id: String,
    pub written_at_unix_ms: u64,
    pub snap_seq: Option<u64>,
    /// `obj:hdr session_id` — the session that owned the layer at checkpoint
    /// time. Required to reconstruct `obj:hdr` on apply.
    #[serde(default)]
    pub session_id: String,
    /// `obj:hdr layer_name` — the human-visible layer name at checkpoint time.
    /// Required to reconstruct `obj:hdr` on apply.
    #[serde(default)]
    pub layer_name: String,
    /// `obj:hdr created_by` — the user / service identity that authored the
    /// original CreateLayer. Required to reconstruct `obj:hdr` on apply.
    #[serde(default)]
    pub created_by: String,
    /// `obj:hdr last_server_ts` — the most recent server-side timestamp the
    /// header carried at checkpoint time. Re-used as `server_ts` for every
    /// rebuilt `obj:hdr` field on apply (created_at / modified_at /
    /// last_server_ts collapse to this single value; the apply path does NOT
    /// invent a fresh wall-clock so a restored checkpoint stays observably
    /// identical to the pre-loss state). Unlike these timestamps, the
    /// `commit_chain` column (`storage::layer::HEADER_FIELD_COMMIT_CHAIN`) is
    /// deliberately NOT reproduced: every materialization, a restore
    /// included, writes a fresh random value, so two restores of this same
    /// generation are distinguishable from each other and from the pre-loss
    /// object.
    #[serde(default)]
    pub server_ts: u64,
    /// Commits this generation covers that the *previous* generation did not:
    /// `seq − last_checkpoint_seq` measured off `obj:hdr` at assembly time,
    /// with any restore-applied sequence bump subtracted out (b.vnw). It is a
    /// per-object, per-interval measurement of this layer's real commit rate,
    /// stamped with no extra Valkey round-trip — the assembler already reads
    /// both columns off the header for the `CheckpointMeta` identity fields.
    ///
    /// Consumed by `apply_checkpoint`, which sizes the DR-restore guard band as
    /// `max(LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN, observed_commit_span)`. The
    /// configured margin stays a floor, so a cold layer restores exactly as it
    /// did before this field existed; a layer that demonstrably commits faster
    /// than the margin carries its own measured band instead of relying on the
    /// operator having sized the knob against peak traffic.
    ///
    /// `0` is the **no-measurement sentinel** and folds the band back to the
    /// configured margin alone — i.e. exactly the pre-b.vnw restore. It is
    /// stamped whenever the assembler cannot produce an honest per-interval
    /// reading, and every pre-b.vnw blob deserializes to it
    /// (`#[serde(default)]`). The cases, per
    /// `assembler::measure_observed_commit_span`:
    ///
    /// - a layer's **first** generation (`last_checkpoint_seq == 0`), which
    ///   would otherwise report commits-since-**creation** — a different
    ///   quantity in the same units as the one the band consumes, and for a
    ///   layer predating the checkpoint loop potentially orders of magnitude
    ///   larger;
    /// - a **fabricated baseline**: `obj:hdr` vanished before the identity read,
    ///   or a cursor column was present but unparseable and folded to 0;
    /// - a post-restore generation whose restore-bump correction consumed the
    ///   whole raw difference (fewer real commits than the band it was seeded
    ///   with).
    ///
    /// A **non-zero** value is always commits accumulated since the previous
    /// *successful* generation. That is normally one checkpoint interval, but it
    /// spans however many intervals a checkpoint outage covered — and that is
    /// deliberate and safety-relevant, not an over-estimate to be clamped away:
    /// after such an outage the span approximates the commit range a restore
    /// actually lost, which is precisely what the guard band must cover.
    /// `apply_checkpoint` WARNs above
    /// `apply::OUTSIZED_BAND_MARGIN_MULTIPLE × safety_margin` but applies the
    /// value unclamped; the `F64_SAFE_INT_MAX` check there remains the only hard
    /// bound on what a span can do to a restore.
    #[serde(default)]
    pub observed_commit_span: u64,
    /// Which kind of generation this meta describes (SR-2.3).
    ///
    /// `#[serde(default)]` is load-bearing: a pre-feature `meta.json` has no
    /// `kind` key and deserializes as [`CheckpointKind::Base`], which is exactly
    /// what those blobs are (PRD A-6). Deliberately NOT `skip_serializing_if` —
    /// a base persisting `"kind":"base"` explicitly is worth the handful of
    /// bytes, both for an operator reading the sidecar and for a forward reader
    /// that wants to distinguish "written by a kind-aware build" from "predates
    /// the field".
    #[serde(default)]
    pub kind: CheckpointKind,
    /// For a [`CheckpointKind::Tail`], the `snap_seq` of the base its lineage
    /// is rooted at; `None` for a [`CheckpointKind::Base`].
    ///
    /// Mirrors `TailHeader::base_snap_seq`, which is authoritative at replay
    /// time. The copy here is what listing and retention read, so a generation's
    /// lineage is answerable without fetching and decoding the envelope.
    #[serde(default)]
    pub base_snap_seq: Option<u64>,
    /// For a [`CheckpointKind::Tail`], the **exclusive** lower bound of the
    /// commit span this envelope carries — the first tail in a lineage has
    /// `from_seq == base_snap_seq`, each later tail has `from_seq ==
    /// previous.to_seq`; `None` for a [`CheckpointKind::Base`].
    ///
    /// The span is `(from_seq, snap_seq]`. There is deliberately **no** separate
    /// `to_seq` field: the upper bound is `snap_seq`, one number with one home,
    /// matching [`CheckpointHandle::snap_seq`] so both kinds address and order
    /// identically. Mirrors `TailHeader::from_seq`.
    ///
    /// Invariant across the three fields: [`CheckpointKind::Base`] ⇒
    /// `base_snap_seq == None && from_seq == None`; [`CheckpointKind::Tail`] ⇒
    /// both `Some`.
    #[serde(default)]
    pub from_seq: Option<u64>,
    /// Optional ext-refs sidecar JSON bytes paired with the OMPF. See the
    /// type-level doc above for the put-time atomicity contract.
    ///
    /// Skipped on serialize / deserialize because the ext-refs sidecar is a
    /// distinct on-disk file in the `LocalDisk` adapter (`<oid>.<snap_seq>.
    /// ext-refs.json`); persisting it again inside the meta JSON would
    /// duplicate the payload.
    #[serde(default, skip_serializing, skip_deserializing)]
    pub sidecar_bytes: Option<Bytes>,
}

impl Default for CheckpointMeta {
    /// Default-constructs a `CheckpointMeta` with empty identity fields and
    /// zero timestamps. Tests that don't care about meta content use this via
    /// `CheckpointMeta::default()` or `..CheckpointMeta::default()` so they
    /// don't have to enumerate every field.
    fn default() -> Self {
        Self {
            creator_replica_id: String::new(),
            written_at_unix_ms: 0,
            snap_seq: None,
            session_id: String::new(),
            layer_name: String::new(),
            created_by: String::new(),
            server_ts: 0,
            observed_commit_span: 0,
            kind: CheckpointKind::Base,
            base_snap_seq: None,
            from_seq: None,
            sidecar_bytes: None,
        }
    }
}

/// Stable identifier for a checkpoint blob in the backing store.
///
/// `adapter_key` is the adapter-specific opaque key (e.g. a relative file path
/// for `LocalDisk`, an in-memory map key for `InMemory`, a blob URL for the
/// `StorageApi` adapter). `object_id` + `snap_seq` + `kind` together form the
/// logical identity used by callers; `adapter_key` is what the adapter needs
/// to round-trip back to the bytes.
///
/// Construct through [`CheckpointHandle::base`] / [`CheckpointHandle::tail`]
/// rather than a struct literal. The fields stay public — every caller reads
/// `handle.object_id` / `handle.snap_seq`, and `#[non_exhaustive]` would forbid
/// the `..` update syntax the integration-test crate uses for no pre-release
/// benefit (single workspace, no external consumers) — but routing construction
/// through the two constructors keeps a future lineage field a one-file change
/// instead of another crate-wide sweep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointHandle {
    pub object_id: u64,
    /// The generation's **ordering** sequence: the snapshot sequence for a
    /// [`CheckpointKind::Base`], the tail envelope's `to_seq` for a
    /// [`CheckpointKind::Tail`] (SR-2.3 orders a lineage newest-first by
    /// exactly this number, whichever kind supplies it).
    ///
    /// Deliberately NOT renamed to something kind-neutral: it is also the
    /// LocalDisk path component (`<oid>.<snap_seq>.ompf`, and a tail's
    /// `<oid>.<to_seq>.tail`) and the addressing key for
    /// `delete_layer_checkpoint` / `read_ext_refs_sidecar`, so one field
    /// carrying the kind's sequence is what keeps addressing uniform across
    /// kinds. Read it through [`CheckpointHandle::seq`] in kind-neutral code.
    pub snap_seq: u64,
    /// Whether this handle names a base snapshot or a tail envelope.
    ///
    /// Meaningful only for **layer** generations. Session-checkpoint handles
    /// (`object_id: 0`, `snap_seq: 0`, `adapter_key: "session:<id>"` in all
    /// three adapters) carry the inert [`CheckpointKind::Base`] default — a
    /// session checkpoint has no lineage and is never replayed onto anything.
    pub kind: CheckpointKind,
    pub adapter_key: String,
}

impl CheckpointHandle {
    /// A handle naming a full-state snapshot.
    ///
    /// Also the constructor for the inert session-checkpoint handles, whose
    /// kind is not consulted by anything.
    pub fn base(object_id: u64, snap_seq: u64, adapter_key: String) -> Self {
        Self {
            object_id,
            snap_seq,
            kind: CheckpointKind::Base,
            adapter_key,
        }
    }

    /// A handle naming a tail envelope. `to_seq` — the envelope's upper,
    /// inclusive commit bound — is what lands in `snap_seq`; see that field's
    /// doc for why the two kinds share one ordering number.
    pub fn tail(object_id: u64, to_seq: u64, adapter_key: String) -> Self {
        Self {
            object_id,
            snap_seq: to_seq,
            kind: CheckpointKind::Tail,
            adapter_key,
        }
    }

    /// The generation's ordering sequence, kind-neutrally: a base's `snap_seq`
    /// or a tail's `to_seq`. Prefer this over reading `snap_seq` directly
    /// wherever the code handles both kinds.
    pub fn seq(&self) -> u64 {
        self.snap_seq
    }
}

/// A checkpoint blob plus its handle and provenance metadata.
#[derive(Debug, Clone)]
pub struct CheckpointBlob {
    pub handle: CheckpointHandle,
    pub bytes: Bytes,
    pub meta: CheckpointMeta,
}

/// Which artifact of a layer-checkpoint generation a [`LayerArtifact`] names.
///
/// Two generation families share the `layers/` namespace, each published as its
/// own stem:
///
/// - a **base** at `<object_id>.<snap_seq>`, carrying up to three durable
///   artifacts ([`Ompf`](Self::Ompf), [`ExtRefs`](Self::ExtRefs),
///   [`Meta`](Self::Meta));
/// - a **tail** at `<object_id>.<to_seq>.tail`, carrying exactly two
///   ([`TailEnvelope`](Self::TailEnvelope), [`TailMeta`](Self::TailMeta)) — a
///   tail never carries an ext-refs sidecar (SR-2.4, enforced at put time by
///   `validate_tail_put`).
///
/// Plus, on adapters that write-temp-then-rename, transient [`Temp`](Self::Temp)
/// files of either family.
///
/// Only the family's **generation artifact** — `*.ompf` for a base, `*.tail` for
/// a tail — makes its stem visible to `list_layer_checkpoints` and to every
/// other generation-stem-driven enumeration in the system; those enumerations
/// key on a generation artifact, never on a companion. That is exactly why the
/// reconciliation pass enumerates artifacts separately (SDD §22.17.5): a
/// companion whose generation artifact never landed is unreachable from every
/// handle-producing path. [`Self::is_generation_artifact`] is the predicate,
/// and [`Self::generation_kind`] names the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerArtifactKind {
    /// `<oid>.<snap_seq>.ompf` — the base generation's OMPF payload. Its
    /// presence is what makes the base stem a *generation* rather than an
    /// orphan.
    Ompf,
    /// `<oid>.<snap_seq>.ext-refs.json` — the ext-refs sidecar. An orphaned one
    /// pins its whole referenced hash set in the retention cascade's live set
    /// indefinitely (SDD §22.17.3).
    ExtRefs,
    /// `<oid>.<snap_seq>.meta.json` — the base's `CheckpointMeta` companion. An
    /// orphaned one is inert for reads but still occupies the store.
    Meta,
    /// `<oid>.<to_seq>.tail` — the tail generation's envelope payload. Its
    /// presence is what makes the tail stem a *generation* rather than an
    /// orphan; the exact analogue of [`Ompf`](Self::Ompf) for the base stem.
    TailEnvelope,
    /// `<oid>.<to_seq>.tail.meta.json` — the tail's kind-qualified
    /// `CheckpointMeta` companion.
    ///
    /// An orphaned one is the ONLY torn-tail durable residual: the tail publish
    /// writes meta first, and a tail carries no ext-refs sidecar (SR-2.4), so a
    /// publish that dies between the two writes can leave nothing else behind.
    /// It is inert to the retention cascade's live set for the same reason — it
    /// references no ext hashes — so unlike an orphaned
    /// [`ExtRefs`](Self::ExtRefs) it costs storage but pins nothing.
    TailMeta,
    /// A write-temp-then-rename leftover — `<final-name>.<uuid>.tmp` on the
    /// LocalDisk adapter. Never published, never enumerable, and (unlike the
    /// durable kinds) not addressable from `(object_id, snap_seq)` alone, so it
    /// is reclaimed through [`CheckpointStore::delete_layer_temp_artifacts`]
    /// rather than through `delete_layer_checkpoint`.
    ///
    /// **Deliberately single and kind-agnostic** — there is no `TailTemp`. A
    /// temp is never a live artifact of *either* family, so it can never
    /// satisfy a stem or be confused for one that does; it is not addressable
    /// from `(object_id, snap_seq)` in either family's case; and
    /// [`CheckpointStore::delete_layer_temp_artifacts`] reclaims every `.tmp`
    /// at a stem across both families in one call. Splitting the variant would
    /// buy no decision anywhere and would add a family axis to a candidate that
    /// belongs to neither family.
    Temp,
}

impl LayerArtifactKind {
    /// Which generation family this artifact belongs to, or `None` for a
    /// [`Temp`](Self::Temp) — a temp belongs to no family; it is its own
    /// reconciliation candidate.
    ///
    /// Exhaustive on purpose (no `_` arm): a future
    /// [`LayerArtifactKind`] variant — the type this match scrutinizes — must
    /// fail the build here rather than silently inherit a family. A future
    /// third [`CheckpointKind`] is caught elsewhere: by the two exhaustive
    /// `match kind` sites in `checkpoint::checkpointer`'s reconciliation pass
    /// (the orphan-candidate member encoding and the act-phase pre-delete
    /// obligation), not by this match, whose arms only *produce*
    /// `CheckpointKind` values.
    pub fn generation_kind(self) -> Option<CheckpointKind> {
        match self {
            LayerArtifactKind::Ompf | LayerArtifactKind::ExtRefs | LayerArtifactKind::Meta => {
                Some(CheckpointKind::Base)
            }
            LayerArtifactKind::TailEnvelope | LayerArtifactKind::TailMeta => {
                Some(CheckpointKind::Tail)
            }
            LayerArtifactKind::Temp => None,
        }
    }

    /// Whether this artifact is the one that makes its stem a *generation*
    /// rather than an orphan — `*.ompf` for a base, `*.tail` for a tail.
    ///
    /// This is the "does this member satisfy its stem" predicate the
    /// reconciliation grouping keys on. A companion never satisfies: it is
    /// precisely a companion without its generation artifact that is the orphan
    /// the pass exists to find. Exhaustive on [`LayerArtifactKind`] for the
    /// same reason as [`Self::generation_kind`], and with the same division of
    /// labour: a new variant of *this* enum fails the build here; a third
    /// [`CheckpointKind`] does not reach this match at all.
    pub fn is_generation_artifact(self) -> bool {
        match self {
            LayerArtifactKind::Ompf | LayerArtifactKind::TailEnvelope => true,
            LayerArtifactKind::ExtRefs
            | LayerArtifactKind::Meta
            | LayerArtifactKind::TailMeta
            | LayerArtifactKind::Temp => false,
        }
    }
}

/// One artifact observed by [`CheckpointStore::list_layer_artifacts`].
///
/// Plain owned scalars only — the type crosses the object-safe trait boundary
/// and carries no adapter-specific state.
///
/// Callers group by `(object_id, snap_seq, generation kind)` — see
/// [`LayerArtifactKind::generation_kind`] — to recover a generation's stem, and
/// a stem with no [`LayerArtifactKind::is_generation_artifact`] member is an
/// orphan candidate **within its own family**. The kind is part of the key
/// because a base at `S` and a tail whose `to_seq` is `S` share the numeric
/// pair: a kind-blind grouping would let the base's `*.ompf` satisfy the tail's
/// torn stem, or vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerArtifact {
    pub object_id: u64,
    /// The generation's **ordering** sequence, kind-neutrally — a base's
    /// `snap_seq`, a tail's `to_seq` — mirroring [`CheckpointHandle::snap_seq`],
    /// and for a [`LayerArtifactKind::Temp`] the sequence of the final filename
    /// the temp wraps.
    ///
    /// Deliberately NOT renamed to something kind-neutral: the name is a
    /// crate-wide sweep for no benefit, and the field carries the same number
    /// under the same name as on the handle it pairs with.
    pub snap_seq: u64,
    pub kind: LayerArtifactKind,
}

/// Suffix of the OMPF payload artifact in the shared layer-artifact naming
/// convention.
pub const OMPF_SUFFIX: &str = ".ompf";
/// Suffix of the ext-refs sidecar artifact.
pub const EXT_REFS_SUFFIX: &str = ".ext-refs.json";
/// Suffix of the `CheckpointMeta` companion artifact of a **base** generation.
pub const META_SUFFIX: &str = ".meta.json";
/// Suffix of the tail envelope artifact — the tail family's generation
/// artifact, the exact analogue of [`OMPF_SUFFIX`] for a base.
pub const TAIL_SUFFIX: &str = ".tail";
/// Suffix of the kind-qualified `CheckpointMeta` companion of a **tail**
/// generation.
///
/// Note this is *not* [`TAIL_SUFFIX`] + [`META_SUFFIX`] by construction — it is
/// spelled out so the parser can test it as one unit, which is what the
/// ordering constraint in [`parse_layer_artifact_filename`] depends on.
pub const TAIL_META_SUFFIX: &str = ".tail.meta.json";
/// Suffix of a write-temp-then-rename leftover. Kind-agnostic: it wraps a
/// complete final filename of either generation family.
pub const TEMP_SUFFIX: &str = ".tmp";
/// Suffix of a session-state checkpoint, `sessions/<session_id>.json`, in the
/// same cross-adapter layout as the layer artifacts above. Both adapters build
/// the session address and strip it back off in
/// [`CheckpointStore::list_session_checkpoint_ids`] with this one constant, so
/// the name a put writes is the name the listing recognises. A session write
/// temp, `<session_id>.json.<uuid>.tmp`, ends in [`TEMP_SUFFIX`] instead and so
/// never matches it.
pub const SESSION_CHECKPOINT_SUFFIX: &str = ".json";

/// Parse one `layers/` artifact filename into a [`LayerArtifact`].
///
/// The naming convention (`<object_id>.<snap_seq>.<kind-suffix>`, plus
/// `<final-name>.<uuid>.tmp` for temps) is a **cross-adapter contract**, not an
/// adapter-local choice: the LocalDisk directory layout and the StorageApi
/// `by-address` layout are deliberately identical so an operator switching
/// adapters does not have to reshape durable state (SDD §22.17). One parser
/// here keeps the two adapters from drifting.
///
/// Returns `None` for anything that is not a layer artifact — a strict
/// two-component numeric stem is required, so a hostile or malformed name
/// (`12.3.4.ompf`) is rejected rather than silently truncated to `(12, 3)`.
///
/// Both generation families are recognized: `12.34.ompf` /
/// `12.34.ext-refs.json` / `12.34.meta.json` for a base, and `12.34.tail` /
/// `12.34.tail.meta.json` for a tail, plus the `.<uuid>.tmp` form of any of
/// them.
///
/// `12.34.tail.ext-refs.json` deliberately parses to `None`: SR-2.4 forbids a
/// tail sidecar, so that name is not a shape this codebase produces. Nothing
/// special-cases it — stripping [`EXT_REFS_SUFFIX`] leaves the residual stem
/// `12.34.tail`, which the strict two-component canonical-`u64` check in
/// [`parse_layer_artifact_stem`] rejects. Admitting it would put a name no
/// writer produces onto a deletion path, and would alias the tail's own stem.
pub fn parse_layer_artifact_filename(name: &str) -> Option<LayerArtifact> {
    // Temps first: the temp suffix wraps a complete final filename, so
    // `12.3.ompf.<uuid>.tmp` must not be mistaken for a durable artifact. Tail
    // temps need no branch of their own — `12.34.tail.<uuid>.tmp` and
    // `12.34.tail.meta.json.<uuid>.tmp` recurse into the tail kinds below and
    // fold to the same kind-agnostic `Temp { 12, 34 }`.
    if let Some(without_tmp) = name.strip_suffix(TEMP_SUFFIX) {
        // `<final-name>.<uuid>` — the disambiguator is a UUID (hyphens, no
        // dots), so the last dot separates it from the final name.
        let (final_name, _uuid) = without_tmp.rsplit_once('.')?;
        let inner = parse_layer_artifact_filename(final_name)?;
        // A temp *of* a temp is not a shape this codebase produces.
        if inner.kind == LayerArtifactKind::Temp {
            return None;
        }
        return Some(LayerArtifact {
            kind: LayerArtifactKind::Temp,
            ..inner
        });
    }

    // Durable suffixes. ORDER IS LOAD-BEARING: `TAIL_META_SUFFIX` must be
    // tested BEFORE `META_SUFFIX`, because `12.34.tail.meta.json` ends in both.
    // Testing `META_SUFFIX` first would leave the residual stem `12.34.tail`,
    // fail the strict stem parse, and return `None` — i.e. silently reproduce
    // the pre-tail behaviour and leak every torn tail's meta companion.
    // `TAIL_SUFFIX` is likewise tested after the two `.json` suffixes: it is a
    // proper substring of `TAIL_META_SUFFIX`, and only the longest match is the
    // right one.
    let (stem, kind) = if let Some(stem) = name.strip_suffix(TAIL_META_SUFFIX) {
        (stem, LayerArtifactKind::TailMeta)
    } else if let Some(stem) = name.strip_suffix(EXT_REFS_SUFFIX) {
        (stem, LayerArtifactKind::ExtRefs)
    } else if let Some(stem) = name.strip_suffix(META_SUFFIX) {
        (stem, LayerArtifactKind::Meta)
    } else if let Some(stem) = name.strip_suffix(TAIL_SUFFIX) {
        (stem, LayerArtifactKind::TailEnvelope)
    } else if let Some(stem) = name.strip_suffix(OMPF_SUFFIX) {
        (stem, LayerArtifactKind::Ompf)
    } else {
        return None;
    };

    let (object_id, snap_seq) = parse_layer_artifact_stem(stem)?;
    Some(LayerArtifact {
        object_id,
        snap_seq,
        kind,
    })
}

/// Parse the shared `<object_id>.<snap_seq>` stem. Strict two-component split
/// (no extra dots) — see [`parse_layer_artifact_filename`].
///
/// Each component must be the **canonical** rendering of its `u64`: ASCII
/// digits only, no sign, and no leading zero on a multi-digit number. `u64`'s
/// own `FromStr` is laxer than that — it accepts `+12` and `0000012` — and both
/// forms alias the canonical `12`. That aliasing is not cosmetic on this path:
/// the reconciliation pass encodes candidates as `<object_id>.<snap_seq>` —
/// plus a suffix distinguishing the other member shapes it mints at the same
/// numeric stem (a temp candidate, and a tail stem's `.tail`-suffixed member) —
/// and dates them on a ZSET keyed by that exact string, so a `+`-prefixed or
/// zero-padded artifact name would register a *second*, permanent member for a
/// stem that already has one. It would never heal (nothing writes that
/// spelling, so the generation artifact it is waiting for never appears under
/// it) and it would never be actionable either way, i.e. it would churn a claim
/// slot on every pass for the lifetime of the deployment. The strictness here
/// is what makes that argument cover every member shape, present and future:
/// every one of them is built from these two canonical components.
pub fn parse_layer_artifact_stem(stem: &str) -> Option<(u64, u64)> {
    // Split at the FIRST dot only, and deliberately do not guard against a
    // third component: such a guard is unreachable by construction, because
    // everything past the first dot stays in the second half. What actually
    // rejects an extra component is `parse_canonical_u64` — `12.3.4` splits to
    // `12` and `3.4`, and `.` is not an ASCII digit, so `3.4` fails the
    // canonical check. Same for a suffix that survived a strip (`34.tail`).
    let (oid_str, snap_str) = stem.split_once('.')?;
    Some((
        parse_canonical_u64(oid_str)?,
        parse_canonical_u64(snap_str)?,
    ))
}

/// Parse a `u64` written exactly the way this codebase writes one.
///
/// Stricter than `u64::from_str` on purpose — see
/// [`parse_layer_artifact_stem`] for why an alias is a durable-state problem
/// rather than a formatting nit. `0` itself is canonical; `00`, `007`, `+7`,
/// `-7`, `7 ` and `` are not.
fn parse_canonical_u64(s: &str) -> Option<u64> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if s.len() > 1 && s.starts_with('0') {
        return None;
    }
    s.parse::<u64>().ok()
}

/// Conservative bound on one artifact publish for adapters whose writes are
/// local and un-retried (LocalDisk, InMemory).
///
/// A `rename(2)` on a healthy filesystem is sub-millisecond; five seconds is a
/// deliberately loose ceiling covering a stalled writeback or a contended
/// volume. It exists to be multiplied into the reconciliation grace, and a
/// grace that is generous costs only reclamation latency, whereas a grace that
/// is tight risks deleting an artifact a concurrent publish is about to pair
/// with an OMPF.
pub const LOCAL_PUBLISH_WORST_CASE: Duration = Duration::from_secs(5);

/// Artifacts one [`CheckpointStore::put_layer_checkpoint`] publishes: the meta
/// JSON, the ext-refs sidecar, and the OMPF itself.
///
/// Named because it is the multiplier between the adapter's *per-call* ceiling
/// and the wall-clock cost of publishing one layer — the number the startup
/// INFO line reports as `layer_publish_ceiling_secs` (b.gg7). It is
/// deliberately **not** applied to [`reconcile_grace`]: the grace protects
/// against a peer sampling the live set while a publish is in flight, and the
/// live set is enumerated from the *sidecar*, so what matters there is the
/// bound on one artifact becoming visible, not on all three landing.
pub const LAYER_PUBLISH_ARTIFACTS: u64 = 3;

/// Multiple applied to [`CheckpointStore::publish_worst_case`] to derive the
/// reconciliation grace window.
///
/// Three, not one: the grace is what separates "an artifact whose sibling
/// publish is still in flight" from "an artifact whose publish died". One
/// worst-case publish is the bound on the *former*, so the grace has to sit
/// clear of it with margin for a publish that started just before the pass's
/// first observation and one that is itself the slowest the adapter admits.
pub const RECONCILE_GRACE_MULTIPLE: u32 = 3;

/// Floor on the derived reconciliation grace, regardless of how fast the
/// adapter claims a publish can be. Keeps an adapter with an optimistic
/// `publish_worst_case` from collapsing the two-observation safety argument
/// into a window a single slow tick could span.
pub const RECONCILE_GRACE_FLOOR: Duration = Duration::from_secs(60);

/// Derive the reconciliation grace from an adapter's own publish bound.
///
/// Deliberately **derived, not configured**: the quantity being bounded is a
/// property of the adapter's retry arithmetic, not an operator preference, and
/// an operator who set it too low would get silent data loss (a half-published
/// generation reaped out from under its own writer) with no signal.
pub fn reconcile_grace(publish_worst_case: Duration) -> Duration {
    publish_worst_case
        .saturating_mul(RECONCILE_GRACE_MULTIPLE)
        .max(RECONCILE_GRACE_FLOOR)
}

/// Operator-facing adapter selector parsed from `LIVE_EDIT_CHECKPOINT_ADAPTER`.
///
/// - `Disabled`: checkpointing is off entirely.
/// - `InMemory`: process-local map; tests and single-replica dev only.
/// - `LocalDisk`: pod-local filesystem; the Epic 1 default-on adapter.
/// - `StorageApi`: external blob-storage backend, reached over the storage
///   API. Selecting it requires `LIVE_EDIT_CHECKPOINT_STORAGE_API_ENDPOINT`,
///   `LIVE_EDIT_CHECKPOINT_BASE_PATH` and `LIVE_EDIT_SERVICE_TOKEN_FILE`;
///   `Config::from_env` refuses the selection when any of them is missing
///   (enforced there, not here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointAdapter {
    Disabled,
    InMemory,
    LocalDisk,
    StorageApi,
}

/// Error returned when `LIVE_EDIT_CHECKPOINT_ADAPTER` contains a value that is
/// none of `disabled`, `inmemory`, `localdisk`, `storageapi` (case-insensitive).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseCheckpointAdapterError {
    input: String,
}

impl fmt::Display for ParseCheckpointAdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid LIVE_EDIT_CHECKPOINT_ADAPTER value {:?}: expected one of disabled, inmemory, localdisk, storageapi (case-insensitive)",
            self.input
        )
    }
}

impl std::error::Error for ParseCheckpointAdapterError {}

impl FromStr for CheckpointAdapter {
    type Err = ParseCheckpointAdapterError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "disabled" => Ok(CheckpointAdapter::Disabled),
            "inmemory" => Ok(CheckpointAdapter::InMemory),
            "localdisk" => Ok(CheckpointAdapter::LocalDisk),
            "storageapi" => Ok(CheckpointAdapter::StorageApi),
            _ => Err(ParseCheckpointAdapterError {
                input: s.to_string(),
            }),
        }
    }
}

/// Backend-agnostic durable-checkpoint store.
///
/// Locked per SDD SR-1.1: the periodic checkpoint loop, the recovery hook, the
/// SR-1.3 conformance harness, and every adapter (in-memory, local-disk, future
/// storage-api) target this exact surface. Trait-object form
/// (`Arc<dyn CheckpointStore>`) is plumbed from `main.rs` into the loop and
/// into `LayerStore`, so the trait MUST stay object-safe (no generic methods,
/// no `Self`-returning methods, no associated types).
///
/// # Type discipline
///
/// Only plain owned types cross the trait boundary — `u64`, `bytes::Bytes`,
/// `&str`, `String`, the leaf types in this module, and `std` collections.
/// No fred / Valkey / cxx / OMPF / FlatBuffers types appear anywhere in the
/// trait signature. Adapters treat the blob payloads as opaque bytes (SR-1.5);
/// OMPF parsing happens above the trait, never inside an adapter.
///
/// # Error contract (asserted by SR-1.3 #19-#22)
///
/// Every method MUST classify failures into the four observable
/// [`CheckpointError`] variants:
///
/// - [`CheckpointError::Transient`] — retryable infrastructure failure (I/O
///   blip, network reset, backend `UNAVAILABLE`). Callers SHOULD retry with
///   backoff.
/// - [`CheckpointError::Permanent`] — non-retryable failure (corrupted blob,
///   schema mismatch, unrecoverable adapter state). Callers MUST NOT retry.
/// - [`CheckpointError::NotFound`] — a `delete_*` or other path-keyed method
///   was handed a handle that does not resolve. `get_*` methods MUST NOT
///   return `NotFound`; they return `Ok(None)` for absent objects so the
///   common "no checkpoint yet" path stays infallible.
/// - [`CheckpointError::Conflict`] — optimistic-concurrency race; a competing
///   writer won. Currently produced by the put paths when two writers race on
///   the same `(object_id, snap_seq)` tuple under an adapter that surfaces the
///   race (LocalDisk's rename-into-place can; InMemory's mutex serializes).
///
/// [`CheckpointError::NotImplemented`] has exactly two kinds of producer on the
/// shipped surface, and neither is a phasing gate on the session-checkpoint
/// methods:
///
/// - **The `Disabled` adapter.** `NoopCheckpointStore` (`checkpoint/builder.rs`)
///   answers *every* method with it, so a caller that drives a checkpoint path
///   while checkpointing is switched off surfaces a classified error rather
///   than a silent success or a fabricated `Ok(None)`.
/// - **Three trait defaults:** [`CheckpointStore::get_layer_checkpoint`],
///   [`CheckpointStore::list_session_checkpoint_ids`] and
///   [`CheckpointStore::ext_blob_exists`]. They exist only to keep non-adapter
///   impls compiling across each method's introduction. Every real adapter and
///   every forwarding decorator overrides all three; inheriting one is a bug,
///   and this variant is what makes that bug loud.
///
/// A real adapter MUST NOT return it from any other method — the
/// session-checkpoint methods **included**. All three adapters implement
/// `put_session_checkpoint` / `get_session_checkpoint` /
/// `delete_session_checkpoint` / `list_session_checkpoint_ids` for real, and
/// `get_session_checkpoint` is on the production session-recovery path
/// (`checkpoint/session_recovery.rs`), so there is no caller-side prohibition
/// on invoking them.
///
/// There is no third kind of producer, and generation kind is not one. All
/// three adapters (`InMemory`, `LocalDisk`, `StorageApi`) accept a
/// [`CheckpointKind::Tail`] [`Self::put_layer_checkpoint`], persist the
/// envelope, and list what they stored, so the producers above are the whole
/// shipped surface: a `NotImplemented` from any other method is a bug in that
/// adapter, never a phasing gate to wait out.
///
/// # Idempotency
///
/// - `put_ext_blob` is idempotent on `hash_hex`: re-putting the same content
///   is a no-op; the bytes themselves are content-addressed.
/// - `delete_layer_checkpoint`, `delete_session_checkpoint`, `delete_ext_blob`
///   are idempotent on missing keys — deleting an already-absent object
///   returns `Ok(())`, never `NotFound`.
/// - `put_layer_checkpoint` and `put_session_checkpoint` are NOT idempotent
///   across distinct `(object_id, snap_seq)` / `session_id` writes; they are
///   the authoritative versioning surface.
///
/// # Ordering
///
/// - `list_layer_checkpoints` returns handles **newest first**, where "newest"
///   is the ordering implied by [`CheckpointHandle::snap_seq`] (higher =
///   newer). Ties on `snap_seq` are not expected in steady state (the writer
///   holds a per-object lease) but adapters MUST still return a stable total
///   order.
/// - **Mixed-kind listings — MUST (SR-2.3).** A lineage interleaves
///   [`CheckpointKind::Base`] and [`CheckpointKind::Tail`] generations under one
///   `object_id`, and the listing is **one strict descending order over the
///   sequence**, not two segregated runs: the sequence is `snap_seq` for a base
///   and the envelope's `to_seq` for a tail, and *both* are carried in
///   `handle.snap_seq` (see that field's doc for why one number serves both
///   kinds). Read it kind-neutrally through [`CheckpointHandle::seq`].
/// - **Kind exposure — MUST.** Every handle a listing mints MUST carry the
///   generation's true `kind`. It is the only thing distinguishing a
///   restorable-on-its-own artifact from one that is meaningless without the
///   base it names, and the SR-2.5 restore-selection invariant is implemented by
///   filtering on it. An adapter that defaulted `kind` to `Base` on a tail would
///   hand the recovery path an unrestorable blob addressed as a full snapshot.
/// - **Same-sequence tie-break — pinned to Base before Tail.** When a base and a
///   tail collide on the same sequence value, the base sorts first. This is not
///   an arbitrary stabilizer: at an equal sequence the base is the artifact that
///   restores on its own, so "first" stays the more useful answer for every
///   kind-blind reader, including [`Self::get_latest_layer_checkpoint`] below.
///   Adapters MUST produce this order via the shared `generation_sort_key`
///   helper in `src/checkpoint/adapters/mod.rs` rather than re-deriving it, so
///   the three adapters cannot drift; Task 6's conformance cases pin it
///   cross-adapter.
/// - `get_latest_layer_checkpoint` returns the same blob whose handle is
///   `list_layer_checkpoints(object_id).await?.first()`, when any exist.
///
///   **This contract is retained verbatim in the tail era — deliberately.** It
///   is tempting to redefine `get_latest` as "newest *base*" so that the sole
///   restore-path caller gets a restorable blob without changing, and that
///   temptation is exactly what this paragraph exists to refuse, for two
///   reasons. First, the SR-1.3 conformance harness asserts the
///   `get_latest` == `list().first()` identity (`conformance.rs`); redefining
///   `get_latest` would silently invalidate that assertion for mixed-kind
///   lineages — the tie between the two enumerations is the only thing keeping
///   an adapter's two listing paths honest with each other. Second, "which
///   generation does a restore start from?" is a **restore-path policy**
///   decision, and pushing it adapter-side reimplements it three times where it
///   can only be tested three times. SR-2.5's interim invariant (newest *base*
///   wins) is therefore implemented once, by kind-aware selection at the
///   recovery hook (Task `t2.dfh.gy.4o`): list → newest [`CheckpointKind::Base`]
///   handle → [`Self::get_layer_checkpoint`]. That is what
///   [`Self::get_layer_checkpoint`] exists for.
/// - `get_layer_checkpoint(h)` returns the blob for whichever element of
///   `list_layer_checkpoints(h.object_id)` matches `h` on
///   `(object_id, snap_seq, kind)`, and `Ok(None)` when none does. `kind` is
///   part of the match key, not a hint: at an equal sequence a base and a tail
///   are two different artifacts.
///
/// # Send + Sync + 'static
///
/// Required so the trait composes with `tokio::spawn`, `Arc<dyn ...>`, and the
/// recovery hook which crosses task boundaries.
#[async_trait]
pub trait CheckpointStore: Send + Sync + 'static {
    // ---- Layer checkpoints --------------------------------------------------

    /// Persist one layer-checkpoint generation keyed by
    /// `(object_id, snap_seq, meta.kind)`.
    ///
    /// `ompf` is the opaque payload; the adapter MUST NOT parse it. For a
    /// [`CheckpointKind::Base`] it is an OMPF container, for a
    /// [`CheckpointKind::Tail`] a tail envelope — the adapter cannot tell the
    /// difference and MUST NOT try to. `meta` carries provenance —
    /// `creator_replica_id`, `written_at_unix_ms`, and `snap_seq` (the adapter
    /// MUST persist these alongside the bytes so
    /// `get_latest_layer_checkpoint` can return them).
    ///
    /// Returns the [`CheckpointHandle`] the adapter assigned. The handle's
    /// `object_id` and `snap_seq` MUST match the arguments; `adapter_key` is
    /// adapter-specific opaque storage state.
    ///
    /// # Generation kind and lineage — MUST (SR-2.3)
    ///
    /// **The put learns the kind from `meta.kind`.** The signature stays
    /// `(object_id, snap_seq, ompf, meta)`: no `put_tail_checkpoint` sibling, no
    /// extra argument. `meta` is already this trait's established carrier for
    /// put-time policy the signature does not name — `sidecar_bytes` works
    /// exactly this way — so routing the kind through it keeps one put path, one
    /// publish-order argument, and one set of conformance cases instead of two
    /// that must be kept in step.
    ///
    /// Caller obligations, all MUST:
    ///
    /// - For a [`CheckpointKind::Tail`], pass the envelope's **`to_seq`** as the
    ///   `snap_seq` argument. That is the tail's ordering sequence, and it is
    ///   what the returned handle's `snap_seq` carries (see
    ///   [`CheckpointHandle::snap_seq`]), so both kinds address and order
    ///   identically.
    /// - Set `meta.kind = CheckpointKind::Tail`, and set **both**
    ///   `meta.base_snap_seq` and `meta.from_seq` to `Some`.
    /// - For a [`CheckpointKind::Base`], leave both lineage fields `None`. This
    ///   is the [`Default`], so a base caller writes nothing kind-related at all.
    ///
    /// Adapter obligations, all MUST:
    ///
    /// - **Never infer lineage.** A `Tail` meta missing either lineage field is a
    ///   caller bug, not a shape to repair by guessing — a fabricated
    ///   `base_snap_seq` names a base that replay will happily but wrongly layer
    ///   onto. Adapters MUST reject it as [`CheckpointError::Permanent`], and
    ///   all three shipped adapters do; a retry cannot fix a caller's meta.
    /// - **Reject a `Tail` meta carrying `sidecar_bytes` as
    ///   [`CheckpointError::Permanent`].** This is strictly stronger than
    ///   SR-2.4's "no sidecar is written for a tail" — silently dropping the
    ///   field would satisfy the letter of that rule while leaving the caller
    ///   believing an ext-refs index was published. It is a caller-visible
    ///   contract precisely so the bug surfaces at its author rather than as a
    ///   dangling-ext-refs restore months later. See the tail-sidecar MUST below
    ///   for why no sidecar can exist for a tail in the first place.
    /// - **The returned handle's `kind` MUST equal `meta.kind`.** Callers use the
    ///   handle to address the generation back (`get_layer_checkpoint`,
    ///   `delete_layer_checkpoint`, `read_ext_refs_sidecar`), all of which match
    ///   on kind.
    ///
    /// # Publish-order obligation (SDD §22.17.3) — MUST
    ///
    /// When `meta.sidecar_bytes` is `Some`, the adapter **MUST make the
    /// ext-refs sidecar visible to [`Self::list_referenced_ext_hashes`] no
    /// later than it makes the OMPF visible to
    /// [`Self::list_layer_checkpoints`]**. Sidecar-first, or atomically with
    /// the OMPF; never OMPF-first.
    ///
    /// This is a data-loss constraint, not a tidiness one. The retention-prune
    /// cascade deletes ext blobs by computing `pruned_hashes − live_hashes`,
    /// where the live set comes from `list_referenced_ext_hashes`. If a
    /// generation's OMPF can become visible while its sidecar is not yet, then
    /// a concurrent prune sees a durable-looking checkpoint whose hashes are
    /// absent from the live set, classifies them prunable, and deletes blobs
    /// that checkpoint points at. The result restores with dangling ext-refs —
    /// silent, unrecoverable loss on a real DR restore, not a retryable error.
    ///
    /// The adapter **SHOULD** also roll back a published sidecar when a later
    /// step of the same put fails, best-effort and without masking the original
    /// error. A sidecar whose OMPF never landed is unreachable from every
    /// deletion path, all of which are generation-stem-driven — a companion is
    /// only ever deleted alongside the generation that owns it — so it pins that
    /// snapshot's entire ext hash set in the live set indefinitely and degrades
    /// the cascade toward deleting nothing.
    ///
    /// # Tail publish order (SR-2.4) — MUST
    ///
    /// A [`CheckpointKind::Tail`] artifact **MUST NOT become visible to
    /// [`Self::list_layer_checkpoints`] before its bytes are fully written** —
    /// write-temp-then-rename, or address-after-complete-PUT, the same two
    /// mechanisms the base path already uses.
    ///
    /// Same shape as the base rule, different reason, so neither one implies the
    /// other and both are stated:
    ///
    /// - A **torn tail is a decode failure on the restore path.** The envelope is
    ///   a length-prefixed record stream with a magic and a record count, so a
    ///   half-written one does not read as a shorter valid tail — it reads as a
    ///   `Permanent` error at exactly the moment DR is trying to recover, on a
    ///   generation the listing claimed was there.
    /// - **SR-9.2's duplicate-leader idempotency argument assumes a listed tail
    ///   is a complete tail.** Two replicas capturing the same immutable stream
    ///   slice at the same `(object_id, to_seq)` produce byte-identical
    ///   artifacts, so the racing puts are safely idempotent — but only if a peer
    ///   that *lists* the artifact and skips its own capture is looking at
    ///   finished bytes. A visible-while-partial tail turns that optimisation
    ///   into silent data loss for the commit range it claimed to cover.
    ///
    /// # Tails carry NO ext-refs sidecar (SR-2.4) — MUST
    ///
    /// An adapter **MUST NOT write any ext-refs artifact for a tail put**, and
    /// **MUST NOT weaken the base path's sidecar-first rule above to accommodate
    /// tails.** The two halves are separate obligations: the first is about what
    /// a tail put produces, the second is a fence around the rule immediately
    /// preceding it — "tails need no sidecar, so the ordering constraint can be
    /// relaxed" is the reasoning that must not be followed, because bases still
    /// need it and the two kinds share this one put path.
    ///
    /// The failure mode is the same silent one the base rule already guards,
    /// arrived at from the other direction. The live set is **sidecar-driven**
    /// ([`Self::list_referenced_ext_hashes`]), while every deletion path is
    /// **generation-stem-driven** — it reaches a companion only through the
    /// generation that owns it, and the tail branch of
    /// [`Self::delete_layer_checkpoint`] removes exactly the two tail files
    /// (envelope + kind-qualified meta), no sidecar among them. So a sidecar
    /// written under a tail's stem is unreachable from
    /// [`Self::delete_layer_checkpoint`] and from the retention cascade alike.
    /// It is an unreapable orphan: it pins that snapshot's entire ext hash set
    /// in the live set forever, and the cascade's `pruned − live` difference
    /// degrades toward the empty set, i.e. toward deleting nothing at all. That
    /// is precisely the residual the LocalDisk adapter already describes for the
    /// base path (`local_disk.rs`, the sidecar-publish comment inside
    /// `put_layer_checkpoint` and `rollback_partial_publish`'s doc) — except
    /// that there it is a bounded consequence of a *failed* put that the
    /// reconciliation pass eventually reaps, whereas a tail-authored sidecar
    /// would be produced by every *successful* one.
    ///
    /// How long it then survives depends on the stem, and "forever" above is
    /// the worse of the two cases. Where a `base@S` and a `tail@to_seq == S`
    /// collide, the shared stem owns a live [`CheckpointKind::Base`] artifact,
    /// so reconciliation reads the sidecar as a live generation's companion and
    /// never classifies it as an orphan — unreapable for as long as that base is
    /// retained. Where the tail's stem carries no base, the stem is
    /// companion-only, reconciliation *does* classify the sidecar as an orphan
    /// candidate, and the pass reaps it — but only after the reconciliation
    /// grace (≥ 1 h), so the hashes stay pinned for at least that long. The MUST
    /// holds either way: one case pins forever, the other pins for the grace
    /// period, and neither is a cost a tail has any reason to incur.
    ///
    /// A tail has nothing to index in any case: it stores the writer's raw
    /// inbound bytes for a commit range, and the ext blobs those commits
    /// reference are already pinned by the base's sidecar. There is no
    /// under-counting to compensate for — see [`Self::read_ext_refs_sidecar`],
    /// which returns `Ok(empty)` for a tail handle as **final** behaviour.
    ///
    /// # Non-empty identity — MUST
    ///
    /// An adapter **MUST reject a put whose `meta.session_id` or
    /// `meta.layer_name` is empty**, with a [`CheckpointError::Permanent`],
    /// before performing any I/O.
    /// [`validate_generation_identity`](crate::checkpoint::adapters::validate_generation_identity)
    /// is the shared implementation; call it as the first statement, beside
    /// [`validate_tail_put`](crate::checkpoint::adapters::validate_tail_put).
    ///
    /// It is a store-side backstop, not the primary guard. A generation with no
    /// identity is **permanently unrestorable**: the apply path
    /// (`checkpoint::apply`) refuses a blob whose layer it cannot resolve, so
    /// writing one spends a PUT, consumes a retention slot, and leaves a
    /// generation on the stem that can only ever fail a restore. Every producer
    /// therefore declines earlier and more cheaply — `read_header_identity`
    /// reports a header with empty identity columns as absent, and each producer
    /// routes that onto its existing "the object is gone" arm — and this rule is
    /// what makes the invariant hold for a producer that has not yet been
    /// written.
    ///
    /// Refusing loudly rather than defaulting is the point: an adapter that
    /// substituted a placeholder identity would convert an unrestorable
    /// generation into a *listed* unrestorable generation, which the restore
    /// path selects and then fails on.
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] — backend write failed but a retry may
    ///   succeed.
    /// - [`CheckpointError::Permanent`] — write rejected unrecoverably (disk
    ///   full on a permanently-too-small volume, schema-version mismatch), **or
    ///   caller-side validation refused the put before any I/O**: empty
    ///   `session_id` / `layer_name` (the MUST above), or a tail put that
    ///   violates the tail-shape rules `validate_tail_put` enforces (a
    ///   `Some(sidecar_bytes)`, a missing sequence, and so on). Both are
    ///   programming errors in the producer, not backend conditions, and no
    ///   retry can change them.
    /// - [`CheckpointError::Conflict`] — a competing writer landed
    ///   `(object_id, snap_seq)` first; the caller is racing and SHOULD bump
    ///   `snap_seq` or abort.
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError>;

    /// Fetch the newest layer-state checkpoint for `object_id`, if any exists.
    ///
    /// Returns `Ok(None)` when no checkpoints exist for the object — the
    /// "first run, nothing to recover" path. Errors are reserved for actual
    /// backend failures.
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] —
    ///   classified per the trait-level error contract. MUST NOT return
    ///   [`CheckpointError::NotFound`] for an absent object; that is an
    ///   `Ok(None)`.
    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError>;

    /// Fetch one **specific** generation of a layer-state checkpoint,
    /// addressed by a handle [`Self::list_layer_checkpoints`] minted (SR-2.3).
    ///
    /// [`Self::get_latest_layer_checkpoint`] structurally cannot express this.
    /// Lineage gather (SR-3.1) and re-base (SR-4.1) list a lineage, pick a
    /// generation by [`CheckpointKind`] and sequence — typically the newest
    /// [`CheckpointKind::Base`], which is by construction *not* the newest
    /// generation — and fetch exactly that one. Hence the handle, not
    /// `object_id`, is the addressing argument.
    ///
    /// The handle is taken **by reference**. Unlike
    /// [`Self::delete_layer_checkpoint`], a get does not consume the
    /// addressing, and a caller walking a lineage still holds the handle after
    /// the fetch — the same stance [`Self::read_ext_refs_sidecar`] already
    /// takes on the read path.
    ///
    /// # Addressing
    ///
    /// Adapters MUST address off `(object_id, snap_seq, kind)` and MAY ignore
    /// `adapter_key`, exactly as [`Self::read_ext_refs_sidecar`] documents:
    /// `adapter_key` is opaque carrier state, so a handle reconstructed by a
    /// caller that never saw the adapter's key must still resolve.
    ///
    /// # The returned blob — MUST
    ///
    /// - `blob.handle`'s `(object_id, snap_seq, kind)` MUST equal the
    ///   addressed handle's.
    /// - `blob.handle.adapter_key` MUST be the adapter's **own canonical key**
    ///   for that generation — the identical string
    ///   [`Self::list_layer_checkpoints`] mints — never an echo of the
    ///   caller's. A handle fabricated with a junk `adapter_key` therefore
    ///   round-trips into a usable one.
    /// - `blob.bytes` are opaque (SR-1.5); the adapter MUST NOT parse them.
    /// - `blob.meta` round-trips what the put persisted — `kind`,
    ///   `base_snap_seq`, and `from_seq` included — so a caller can read a
    ///   generation's lineage off the fetched blob without re-listing.
    ///
    /// # Default impl — treat inheriting it as a bug in any real adapter
    ///
    /// The default returns [`CheckpointError::NotImplemented`] with
    /// `method: "get_layer_checkpoint"`. It exists only so the trait's many
    /// test doubles and decorators keep compiling across this method's
    /// introduction; every shipped adapter overrides it, and **any decorator /
    /// wrapper `CheckpointStore` MUST forward this method to its inner store**
    /// wherever it forwards the other layer methods.
    ///
    /// `Ok(None)` would be the wrong default precisely because a decorator
    /// wrapping a working store would then report "no such generation" for a
    /// generation that exists — silently, to the one caller whose entire job
    /// is deciding what exists. `NotImplemented` fails loudly instead.
    ///
    /// # Errors
    ///
    /// - `Ok(None)` — the addressed generation is absent, **including** the
    ///   concurrent-prune race where a [`Self::list_layer_checkpoints`]
    ///   observed the handle and retention pruned it before this call landed.
    ///   Absence MUST NOT surface as [`CheckpointError::NotFound`]; that is the
    ///   trait-level get-path rule.
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] —
    ///   classified per the trait-level error contract. A malformed persisted
    ///   meta is [`CheckpointError::Permanent`], mirroring
    ///   [`Self::get_latest_layer_checkpoint`]'s arm: the bytes are durable, so
    ///   a retry cannot fix them.
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        let _ = handle;
        Err(CheckpointError::NotImplemented {
            method: "get_layer_checkpoint",
        })
    }

    /// List every layer-state checkpoint handle for `object_id`,
    /// **newest first** (descending `snap_seq`).
    ///
    /// Returns an empty vector when no checkpoints exist; absence is not an
    /// error. The conformance harness asserts that the first element matches
    /// the handle returned by `get_latest_layer_checkpoint`.
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] —
    ///   classified per the trait-level error contract.
    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError>;

    /// Delete the layer-state checkpoint identified by `handle`.
    ///
    /// **Idempotent on missing handles** — if the handle has already been
    /// pruned (or never existed), returns `Ok(())`, NEVER
    /// [`CheckpointError::NotFound`]. The retention pruner relies on this so
    /// concurrent prune passes do not race on identical handles.
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] —
    ///   classified per the trait-level error contract.
    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError>;

    // ---- Session checkpoints -------------------------------------------------

    /// Persist a session-state checkpoint blob keyed by `session_id`.
    ///
    /// Wired and callable: the session-checkpoint writer
    /// (`checkpoint/session_writer.rs`) drives this on the live path, and all
    /// three real adapters implement it. Only the `Disabled` adapter answers
    /// with [`CheckpointError::NotImplemented`], which that writer treats as a
    /// silent off-mode signal rather than a failure.
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] /
    ///   [`CheckpointError::Conflict`] — classified per the trait-level error
    ///   contract.
    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError>;

    /// Fetch the session-state checkpoint for `session_id`, if any exists.
    ///
    /// Wired and callable — this is on the production session-recovery path
    /// (`checkpoint/session_recovery.rs`), under that hook's fetch budget.
    ///
    /// Returns `Ok(None)` for an absent session; absence is not an error, per
    /// the trait-level get-path rule. Only the `Disabled` adapter returns
    /// [`CheckpointError::NotImplemented`].
    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError>;

    /// Delete the session-state checkpoint for `session_id`.
    ///
    /// Wired and callable. **Idempotent on missing session ids** — deleting an
    /// already-absent session returns `Ok(())`, never
    /// [`CheckpointError::NotFound`]. Only the `Disabled` adapter returns
    /// [`CheckpointError::NotImplemented`].
    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError>;

    /// List the id of every session that has a stored session-state
    /// checkpoint (SR-13.1).
    ///
    /// The DR restore Job lists the store once (SR-9.2) and uses this as its
    /// set of restorable sessions; no other caller exists.
    ///
    /// # The returned set — MUST
    ///
    /// - It is **exactly** the session ids for which
    ///   [`Self::get_session_checkpoint`] would return a blob: never a write
    ///   temp, a layer artifact, an ext blob, the counters sidecar, or any
    ///   other file the backend happens to hold beside the session backups.
    /// - An empty store answers `Ok(vec![])`; absence is not an error.
    /// - Order is unspecified.
    ///
    /// # Default impl — treat inheriting it as a bug in any real adapter
    ///
    /// The default returns [`CheckpointError::NotImplemented`] with
    /// `method: "list_session_checkpoint_ids"`. Every shipped adapter
    /// overrides it, and **any decorator / wrapper `CheckpointStore` MUST
    /// forward this method to its inner store** wherever it forwards the other
    /// session methods.
    ///
    /// `Ok(vec![])` would be the wrong default precisely because a decorator
    /// wrapping a working store would then report "no session backups" for a
    /// store full of them — silently, to the one caller whose job is deciding
    /// which sessions are restorable. `NotImplemented` fails loudly instead.
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] —
    ///   classified per the trait-level error contract.
    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        Err(CheckpointError::NotImplemented {
            method: "list_session_checkpoint_ids",
        })
    }

    // ---- External-value blobs -----------------------------------------------

    /// Persist an external-value blob under its content-address `hash_hex`.
    ///
    /// **Idempotent on `hash_hex`** — re-putting bytes that already exist
    /// under the same hash MUST succeed without rewriting. External values
    /// are content-addressed (the lowercase hex of the value's 32-byte content
    /// hash, its `valueOrExtHash`), so duplicate puts under the same
    /// hash imply identical bytes; the adapter MAY skip the write entirely.
    ///
    /// **Keeping `bytes` keeps the mirror's charge.** The ext-mirror and the
    /// backup repair send each value's one owned copy here, charged to the
    /// mirror's byte budget until the last handle on that memory drops
    /// (`checkpoint::ext_mirror`). An adapter writes from these bytes, never
    /// from a copy of them: a copy made for the write is memory the budget
    /// does not count. Clones held for the call's own retries are covered by
    /// that charge and end with the call. A write that outlives a dropped
    /// caller (the LocalDisk adapter's blocking write task finishes after its
    /// caller stops waiting) keeps the bytes, and so the charge, until it
    /// ends. An adapter that keeps the bytes past its return (in a map, a
    /// cache, a background task) holds the charge for as long and shrinks the
    /// budget left for everything else; to keep the bytes without the charge,
    /// store a copy of them, as the InMemory adapter's map does.
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] —
    ///   classified per the trait-level error contract.
    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError>;

    /// Fetch an external-value blob by content-address.
    ///
    /// Returns `Ok(None)` when the hash is unknown; absence is not an error.
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] —
    ///   classified per the trait-level error contract.
    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError>;

    /// Report whether the bytes of the external-value blob at `hash_hex` are
    /// stored, without fetching them (SR-13.2).
    ///
    /// Callers: the external-value backup repair
    /// (`checkpoint::ext_backup_repair`), for each value Valkey holds with no
    /// confirmed copy and for its daily audit of confirmed ones; and the DR
    /// restore Job's coverage check (SR-9.3), for every referenced hash Valkey
    /// and the restored tails do not already cover.
    ///
    /// # The answer — MUST
    ///
    /// - `Ok(true)` only when the bytes are stored.
    /// - `Ok(false)` only when the store answered that they are not (no such
    ///   key, file or object). That answer is not proof of absence:
    ///   `Ok(false)` may be stale for up to the storage cache TTL. The
    ///   storage-api backend can cache a not-found answer per pod (15 minutes
    ///   per pod, measured on RC; whether it caches not-found, and for how
    ///   long, is the deployer's setting), and a write clears that entry only
    ///   on the pod that served the write, so bytes that are stored can still
    ///   read absent on another pod. Callers therefore never act on absence
    ///   alone in a way a wrong answer would make lossy: the backup repair
    ///   answers it with an upload of the same bytes, which at worst repeats
    ///   one already stored; the DR restore Job that will call it carries the
    ///   rule as t1.tv3.s9 defer-6(d) (the sidecar reads, as t1.tv3.gk
    ///   defer-12).
    /// - An error in every other case. The Job counts an error as "unknown",
    ///   never as "absent", so an adapter MUST NOT fold an indeterminate
    ///   backend answer into `Ok(false)` — that would let the Job strip a
    ///   value whose bytes may well exist.
    ///
    /// # Default impl — treat inheriting it as a bug in any real adapter
    ///
    /// The default returns [`CheckpointError::NotImplemented`] with
    /// `method: "ext_blob_exists"`. Every shipped adapter overrides it, and
    /// **any decorator / wrapper `CheckpointStore` MUST forward this method to
    /// its inner store** wherever it forwards the other ext-blob methods.
    /// `Ok(false)` would be the wrong default for the same reason an empty
    /// list is wrong for [`Self::list_session_checkpoint_ids`]: a decorator
    /// that forgot to forward would report every stored value as lost.
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] —
    ///   classified per the trait-level error contract. MUST NOT return
    ///   [`CheckpointError::NotFound`] for an absent blob; that is an
    ///   `Ok(false)`.
    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        let _ = hash_hex;
        Err(CheckpointError::NotImplemented {
            method: "ext_blob_exists",
        })
    }

    /// Delete the external-value blob at `hash_hex`.
    ///
    /// **Idempotent on missing hashes** — deleting an absent blob returns
    /// `Ok(())`, never [`CheckpointError::NotFound`]. The orphan-sweep loop
    /// depends on this so multiple sweepers do not race on identical hashes.
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] —
    ///   classified per the trait-level error contract.
    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError>;

    // ---- Counters sidecar ---------------------------------------------------

    /// Persist the singleton counters-sidecar blob.
    ///
    /// The counters sidecar carries the atomic-id-counter floor used by the
    /// recovery hook to reseed `le:counter:*` after a Valkey wipe. There is
    /// exactly one sidecar per checkpoint-store instance. The adapter stores
    /// `body` as given, replacing any earlier sidecar; monotonicity is the
    /// caller's rule. The checkpoint leader's counters tick reads the stored
    /// sidecar before every write and never writes a `max_object_id_observed`
    /// at or below the stored one (SDD SR-2.4). A write already in flight can
    /// still land after a higher one and lower the stored value to that
    /// earlier reading; the next counters tick then raises it again.
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] —
    ///   classified per the trait-level error contract.
    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError>;

    /// Fetch the singleton counters-sidecar blob.
    ///
    /// Returns `Ok(None)` when the store answers that no sidecar exists — the
    /// pre-first-checkpoint recovery path. That answer is not proof the
    /// sidecar was never written: it may be stale for up to the storage cache
    /// TTL. The storage-api backend can cache a not-found answer per pod (15
    /// minutes per pod, measured on RC; whether it caches not-found, and for
    /// how long, is the deployer's setting), and a write clears that entry
    /// only on the pod that served the write. The rule for callers is
    /// therefore never to act on absence alone. The counters tick keeps it
    /// once its process has seen the sidecar: it then treats `Ok(None)` as an
    /// error (`checkpointer::run_counters_sync_tick`). The readers that still
    /// take `Ok(None)` at its word — that tick before its process has seen the
    /// sidecar, and the counter seed (`checkpoint::counter_seed`) — are
    /// carried as t1.tv3.gk defer-12, beside t1.tv3.s9 defer-6(d) for the DR
    /// restore Job.
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] —
    ///   classified per the trait-level error contract.
    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError>;

    // ---- Cross-cutting enumeration ------------------------------------------

    /// List every `object_id` for which at least one layer-state checkpoint
    /// exists.
    ///
    /// **This method has no production caller.** Its one remaining consumer is
    /// the *default* implementation of
    /// [`list_referenced_ext_hashes`](Self::list_referenced_ext_hashes), which
    /// all three shipped adapters override — and the orphan-ext-blob sweep,
    /// which this doc used to name, no longer routes through it: b.s5y made the
    /// live set sidecar-driven and b.9qd moved orphan reclamation onto
    /// [`Self::list_layer_artifacts`], both precisely *because* an
    /// object-id-then-OMPF walk is the data-loss pattern
    /// [`list_referenced_ext_hashes`](Self::list_referenced_ext_hashes)
    /// documents. Do not reintroduce it as a caller.
    ///
    /// It stays on the trait because it is the honest surface for "which objects
    /// does this store hold anything for" — adapters with cheap directory
    /// listings (LocalDisk, InMemory) implement it directly, and it is
    /// exercised by the conformance harness and by diagnostics.
    ///
    /// Order is unspecified. An object whose only surviving artifact is a
    /// [`CheckpointKind::Tail`] **does** enumerate, on every adapter: all three
    /// (`InMemory`, `LocalDisk`, `StorageApi`) are **generation-stem-driven** and
    /// recognize both the base and the tail stem, and a tail-only object left
    /// unreported would be invisible to the recovery hook's working set. What
    /// no adapter reports is a stem carrying only companions (`.meta.json`,
    /// `.ext-refs.json`, `.tail.meta.json`) or an in-flight `.tmp`: none of those
    /// parses as a generation, which is the point — this method's consumer wants
    /// "layers that have a restorable payload".
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] —
    ///   classified per the trait-level error contract.
    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError>;

    /// Collect the set of external-value hashes still referenced by any
    /// live layer-state checkpoint.
    ///
    /// The reconciliation pass's collect phase (`checkpointer::run_reconcile_collect`)
    /// is the only ext-blob deleter: it deletes a tombstoned blob only when this
    /// set does not name it, so the set is the **safety input** to every
    /// ext-blob deletion in the system, not a lookup optimization. The collect
    /// reads it through [`Self::list_live_ext_set`], which carries this same
    /// set plus the evidence that the walk's own listing named something; an
    /// adapter overriding this method MUST override that one too (see its
    /// doc), and this method is then its projection.
    ///
    /// # Enumeration obligation (SDD §22.17.3) — MUST
    ///
    /// Any adapter that stores ext blobs **MUST** override this, and the
    /// override **MUST be sidecar-driven**: enumerate the ext-refs sidecars
    /// themselves (a directory walk, a prefix LIST, a map scan — whatever the
    /// backend offers), never the OMPF artifacts.
    ///
    /// The tempting generic composition —
    /// `list_all_object_ids` → [`Self::list_layer_checkpoints`] →
    /// [`Self::read_ext_refs_sidecar`] — **is exactly the data-loss pattern**
    /// and MUST NOT be used. It discovers a generation's sidecar by way of its
    /// OMPF, so any generation whose sidecar has been published but whose OMPF
    /// has not is invisible to the live set. A concurrent prune then classifies
    /// that generation's hashes prunable and deletes blobs it points at, and
    /// the checkpoint restores with dangling ext-refs. Paired with
    /// [`Self::put_layer_checkpoint`]'s sidecar-first obligation, sidecar-driven
    /// enumeration is what shrinks that window to the publish gap.
    ///
    /// # A listed sidecar that reads absent is an error — MUST
    ///
    /// An adapter that reads a sidecar its own enumeration named, and then
    /// finds it absent, **MUST** return the typed error
    /// [`CheckpointError::listed_sidecar_absent`] (a `Transient` carrying a
    /// [`ListedSidecarAbsent`] naming the artifact) and **MUST NOT** skip it.
    /// Skipping drops that sidecar's hashes from the live set, and an
    /// under-counted live set is an over-delete: the reconciliation collect
    /// phase deletes every tombstoned blob the set does not name, which can be
    /// a blob a retained backup still references. A partial set is never
    /// returned. The marker must be the typed one, not just an equivalent
    /// message: the collect phase recognises it with
    /// [`CheckpointError::is_listed_sidecar_absent`] and qualifies it by who
    /// holds the checkpoint lease. It always deletes nothing that pass; it
    /// counts the abort as a live-set failure only while this replica still
    /// holds the lease, because while a peer holds it that peer's lease-gated
    /// retention prune explains the absence routinely. The next pass re-lists.
    /// The rule binds [`Self::list_live_ext_set`] equally — it is the same
    /// walk. Forwarding decorators pass the error through untouched. An adapter
    /// that enumerates and reads under one atomic view (InMemory's single
    /// lock) cannot observe the condition and needs no arm for it. This is not
    /// [`Self::read_ext_refs_sidecar`]'s rule: that per-handle read is not
    /// driven by a listing, so an absent sidecar there is `Ok` and empty.
    ///
    /// # Tails are structurally invisible here — and that is load-bearing
    ///
    /// The sidecar-driven obligation has a second consequence in the tail era:
    /// because [`Self::put_layer_checkpoint`] forbids a tail from ever writing an
    /// ext-refs artifact, a sidecar walk **cannot** observe a
    /// [`CheckpointKind::Tail`] generation. No kind predicate is needed and none
    /// should be added — the enumeration shape produces the right answer on its
    /// own.
    ///
    /// That invisibility is now **load-bearing rather than incidental**, so it
    /// must not be "fixed". A tail's referenced blobs are pinned by its base's
    /// sidecar, and the base outlives every tail rooted at it: retention's unit
    /// is the whole lineage, and it deletes a unit's tails **before** its base
    /// (SR-5.2's dependency-safe order, executed by
    /// `checkpointer::prune_old_generations`), so a live base names those blobs
    /// for the tail's entire lifetime. Counting a tail would therefore add
    /// nothing to the live set. What it *would* do is make the live set depend
    /// on the tail-capture path's write ordering as well as the assembler's — a
    /// second publish race against the same deletion decision, for zero
    /// coverage. An adapter that grows a tail-aware branch here has re-opened
    /// the data-loss window this section exists to close.
    ///
    /// # Default-impl contract — deprecated for new adapters
    ///
    /// The default impl returns `Ok(HashSet::new())`. It does NOT parse OMPF
    /// blobs to discover ext-refs — that requires the (not-yet-built) OODF
    /// reassembler and would pull cxx / FlatBuffers types into this module,
    /// breaking the opaque-bytes-only adapter contract (SR-1.5). Nor does it
    /// use the generic composition above, for the reason just given.
    ///
    /// Treat inheriting it as a bug in any new adapter. It survives only so the
    /// trait compiles in isolation and so an ext-blob-less adapter need not
    /// implement it; all three shipped adapters override it.
    ///
    /// **Override this, [`Self::list_live_ext_set`] and
    /// [`Self::read_ext_refs_sidecar`] together, or none of them.** The
    /// retention prune tombstones the hashes [`Self::read_ext_refs_sidecar`]
    /// reports, and the collect deletes every tombstoned hash the live set
    /// does not name. An adapter that inherits all three defaults tombstones
    /// nothing and so deletes nothing. One that reports real per-handle hashes
    /// while inheriting both live-set defaults reads an empty live set that
    /// vouches for no listing, so the collect defers: blobs leak, nothing is
    /// lost. One that overrides [`Self::list_live_ext_set`] to vouch for a
    /// listing while its hash set misses sidecars that listing named is an
    /// over-delete, silently.
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] —
    ///   classified per the trait-level error contract.
    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        // Drive the companion enumeration to keep adapter implementations
        // honest — overriders that depend on `list_all_object_ids` returning
        // sensibly will hit the same surface here. The default impl
        // intentionally drops the result and returns an empty set; see the
        // doc-comment above for why we don't parse OMPF blobs here.
        let _object_ids = self.list_all_object_ids().await?;
        Ok(HashSet::new())
    }

    /// The live ext-hash set together with the evidence that licenses
    /// deleting against it: how many layer artifacts the walk's **own**
    /// enumeration named (b.oya, SDD §22.17.3 obligation 5, §22.17.5).
    ///
    /// `hashes` is exactly what [`Self::list_referenced_ext_hashes`] returns,
    /// and every obligation on that method — sidecar-driven, a listed sidecar
    /// that reads absent is the typed [`ListedSidecarAbsent`] error, tails
    /// structurally invisible, never a partial set — binds this one equally.
    /// `listed_layer_artifacts` is the number of names in that same
    /// enumeration that [`parse_layer_artifact_filename`] accepts, of any
    /// kind.
    ///
    /// # Why the count must come from the same request
    ///
    /// The reconciliation collect deletes every aged tombstoned blob the live
    /// set does not name. An EMPTY live set is the trait's answer for an empty
    /// store, and also what a store at the wrong address, or a listing that
    /// answers a cached or lagging not-found, returns — the last would delete
    /// blobs retained backups still name, an unrecoverable DR loss. So the
    /// collect deletes against an empty set only when this count is non-zero.
    /// A count from any other request cannot vouch for this one: on
    /// storage-api two listings in one pass can answer differently.
    ///
    /// # Adapters — MUST
    ///
    /// An adapter that overrides [`Self::list_referenced_ext_hashes`] **MUST**
    /// override this method too, producing both fields from **one**
    /// enumeration, and its `list_referenced_ext_hashes` then becomes
    /// `Ok(self.list_live_ext_set().await?.hashes)`. The count is positive
    /// evidence: a path that cannot produce it from the actual listing MUST
    /// report 0.
    ///
    /// # Forwarding decorators — MUST
    ///
    /// A decorator that forwards [`Self::list_referenced_ext_hashes`] **MUST**
    /// forward this method too, applying whatever it does to that one (fault
    /// injection, recording, stalls). One that inherits the default instead
    /// only ever turns an empty live set into a deferral; it never causes a
    /// wrong delete.
    ///
    /// # Default — fail-closed
    ///
    /// The default wraps [`Self::list_referenced_ext_hashes`] with
    /// `listed_layer_artifacts: 0`: it vouches for no listing, so an empty
    /// live set read through it never licenses a delete.
    ///
    /// Errors: as [`Self::list_referenced_ext_hashes`].
    async fn list_live_ext_set(&self) -> Result<LiveExtSet, CheckpointError> {
        Ok(LiveExtSet {
            hashes: self.list_referenced_ext_hashes().await?,
            listed_layer_artifacts: 0,
        })
    }

    /// Read the set of external-value hashes referenced by a single
    /// layer-state checkpoint generation, addressed by its
    /// [`CheckpointHandle`].
    ///
    /// This is the per-handle companion to
    /// [`list_referenced_ext_hashes`](Self::list_referenced_ext_hashes). The
    /// retention prune, the reconciliation pass's orphan and dead-object
    /// phases, and the base bootstrap's inline reap read the about-to-be-deleted
    /// generation's ext-hashes via this method *before* its
    /// `delete_layer_checkpoint` lands, and record them as tombstones. The
    /// reconciliation collect later deletes a tombstoned blob only if the live
    /// set ([`Self::list_live_ext_set`]) no longer names it. Without
    /// per-handle resolution nothing could tell "what hashes did *this*
    /// generation reference?" apart from "what hashes does the union still
    /// reference?".
    ///
    /// The handle's `(object_id, snap_seq)` pair is the addressing primitive;
    /// `adapter_key` is opaque carrier state and adapters MAY ignore it on
    /// the read path.
    ///
    /// # Tail handles ⇒ `Ok(HashSet::new())` — MUST, and final (SR-2.4)
    ///
    /// On a [`CheckpointKind::Tail`] handle, implementations MUST return
    /// `Ok(HashSet::new())`.
    ///
    /// This is not a new rule so much as the absence contract below, stated for
    /// a kind that **structurally never has a sidecar**:
    /// [`Self::put_layer_checkpoint`] forbids writing one for a tail, so "absent"
    /// is the only truthful answer and it is knowable from the handle without
    /// touching the backend. Adapters SHOULD short-circuit on `handle.kind`
    /// rather than probing, because the addressing helpers are kind-blind — a
    /// tail handle falling through to a `<oid>.<seq>` sidecar lookup would return
    /// the *base* generation's hashes as if they were the tail's, and the
    /// retention prune would then tombstone them when the tail is pruned —
    /// hashes the base still names, so every one of them comes back to the
    /// reconciliation collect as a false alarm.
    ///
    /// **This is final behaviour, not a placeholder.** Tail storage has landed
    /// on all three adapters and nothing filled this in, because nothing can: a
    /// tail's referenced ext blobs are pinned by its base's sidecar, so an empty
    /// set here is the correct steady-state answer forever.
    ///
    /// # Default-impl contract
    ///
    /// The default impl returns `Ok(HashSet::new())`. Returning empty
    /// tombstones nothing for that handle, so its blobs are never collected
    /// rather than deleted wrongly — the safe direction. All three shipped
    /// adapters override it.
    ///
    /// **Override this together with
    /// [`list_referenced_ext_hashes`](Self::list_referenced_ext_hashes) and
    /// [`list_live_ext_set`](Self::list_live_ext_set), never alone.** This
    /// method decides what gets tombstoned; the live set decides which
    /// tombstoned blobs the reconciliation collect may delete. Overriding this
    /// one while inheriting the live-set defaults tombstones real hashes
    /// against a live set that vouches for no listing, so the collect defers
    /// forever and the blobs leak. Pairing the other way tombstones nothing
    /// and so deletes nothing. The over-delete is a different fault: a
    /// live-set walk that vouches for a listing while missing sidecars it
    /// named (see `list_referenced_ext_hashes`). The pairing is the
    /// invariant, not any one method on its own.
    ///
    /// # Errors
    ///
    /// - When the sidecar is absent for the addressed handle (file not
    ///   found / no inline sidecar / never written) implementations MUST
    ///   return `Ok(HashSet::new())`. Absence is not an error here: a
    ///   generation may legitimately carry no sidecar — a legacy generation
    ///   written before the assembler started emitting sidecars, or one that
    ///   never wrote a sidecar (`sidecar_bytes` was `None` at put time; a tail
    ///   never does) — and an empty answer keeps the cascade well-behaved on
    ///   them.
    ///   [`list_referenced_ext_hashes`](Self::list_referenced_ext_hashes)
    ///   differs on purpose: it reads only sidecars its own enumeration
    ///   listed, so an absent one there is an inconsistency and MUST be an
    ///   error (see that method's contract).
    /// - Malformed sidecar bytes MUST classify as
    ///   [`CheckpointError::Permanent`] — the bytes are durable, retrying
    ///   the read does not fix them; surface so the operator can investigate
    ///   rather than silently dropping references.
    /// - Other failures classified per the trait-level error contract
    ///   (`Transient` for infrastructure blips, `Permanent` for unrecoverable
    ///   backend conditions).
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        let _ = handle;
        Ok(HashSet::new())
    }

    // ---- Orphan reconciliation (b.9qd) --------------------------------------

    /// **Raw** enumeration of every artifact in the adapter's `layers/`
    /// namespace — of **both** generation families: a base's OMPF, ext-refs
    /// sidecar and meta companion, a tail's envelope and kind-qualified meta
    /// companion, and write-temp leftovers of either, alike.
    ///
    /// This is the one enumeration on the trait that is deliberately **not**
    /// filtered by anything. `list_layer_checkpoints`, `list_all_object_ids`,
    /// and `get_latest_layer_checkpoint` are all **generation-stem-driven** on
    /// all three adapters (`InMemory`, `LocalDisk`, `StorageApi`), which
    /// recognize both the base and the tail stem; `list_referenced_ext_hashes`
    /// is sidecar-driven; between them, an artifact whose stem never acquired a
    /// generation artifact is invisible to every handle-producing path in the
    /// system and therefore unreachable from
    /// [`Self::delete_layer_checkpoint`]. That is not merely untidy: an orphaned
    /// `.ext-refs.json` keeps counting toward the live set forever, so the
    /// reconciliation collect spares every tombstoned blob it names, for good
    /// (SDD §22.17.3).
    ///
    /// # Both generation families are reported
    ///
    /// The reconciliation pass groups the result by
    /// `(object_id, snap_seq, generation kind)` — the kind belongs in the key
    /// because a base at `S` and a tail whose `to_seq` is `S` share the numeric
    /// pair, and a kind-blind grouping would let one family's generation
    /// artifact satisfy the other family's torn stem. Within a family, a stem
    /// with no [`LayerArtifactKind::is_generation_artifact`] member is an orphan
    /// candidate: for a base that means an `.ext-refs.json` or `.meta.json` with
    /// no `.ompf`; for a tail it means a [`LayerArtifactKind::TailMeta`] with no
    /// [`LayerArtifactKind::TailEnvelope`] — the meta-first tail publish's only
    /// durable torn residual (a tail carries no ext-refs sidecar, SR-2.4). A
    /// [`LayerArtifactKind::Temp`] artifact is a candidate on its own, of
    /// neither family, and tail temps are reclaimable exactly as base temps
    /// are. Order is unspecified and duplicates are permitted (the caller sets).
    ///
    /// # Default-impl contract — degrades to no sweep
    ///
    /// The default returns `Ok(Vec::new())`, which makes the reconciliation pass
    /// a no-op for that adapter. Unlike the paired
    /// [`Self::list_referenced_ext_hashes`] / [`Self::read_ext_refs_sidecar`]
    /// defaults — where inheriting one and overriding the other is a silent
    /// over-delete — this default is **standalone-safe**: an empty enumeration
    /// yields no candidates, so nothing is deleted. A new adapter that inherits
    /// it leaks orphans (a storage-cost bug) rather than losing data.
    ///
    /// Errors:
    /// - [`CheckpointError::Transient`] / [`CheckpointError::Permanent`] —
    ///   classified per the trait-level error contract. The pass aborts on
    ///   either; the next pass re-enumerates.
    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        Ok(Vec::new())
    }

    /// Delete every write-temp leftover belonging to
    /// `(object_id, snap_seq)`, returning how many were removed.
    ///
    /// Separate from [`Self::delete_layer_checkpoint`] because a temp file is
    /// **not addressable from `(object_id, snap_seq)` alone** — its name carries
    /// a random disambiguator the caller cannot reconstruct — and because a temp
    /// can be garbage while its stem's generation is perfectly live, in which
    /// case deleting the generation would be catastrophic and deleting the temp
    /// is correct.
    ///
    /// # Kind-agnostic, deliberately
    ///
    /// One call reclaims **every** `.tmp` at that numeric stem, across both
    /// generation families — a base's `<oid>.<snap_seq>.ompf.<uuid>.tmp` and a
    /// tail's `<oid>.<to_seq>.tail.<uuid>.tmp` alike (hence the single
    /// kind-agnostic [`LayerArtifactKind::Temp`], which carries no family).
    /// That is safe precisely because a temp is never a live artifact of either
    /// family: it is unpublished by construction, so no reader can reach it and
    /// no generation of either kind depends on it. The over-broad-key worry that
    /// makes kind part of the *grouping* key does not apply here — collecting a
    /// same-sequence sibling family's temp is not a mis-delete, it is the other
    /// half of the same reclamation.
    ///
    /// **Idempotent**: an adapter with no temps for that stem returns `Ok(0)`,
    /// never [`CheckpointError::NotFound`].
    ///
    /// # Default-impl contract
    ///
    /// `Ok(0)` — correct only for an adapter that can neither *produce* a temp
    /// nor *enumerate* one, i.e. one whose [`Self::list_layer_artifacts`] can
    /// never emit [`LayerArtifactKind::Temp`]. InMemory is the only such
    /// adapter today (it inserts generations under a mutex and enumerates that
    /// same map).
    ///
    /// **Inheriting this default while emitting `Temp` is a livelock, not a
    /// degradation.** The reconciliation pass registers every enumerated temp
    /// as a candidate, waits out the grace, calls this method, sees `Ok(0)`,
    /// counts a reclamation, retires the candidate — and re-registers the very
    /// same temp on the next pass, forever, logging a reclamation that never
    /// happened. Both StorageApi and LocalDisk parse the `<final>.<uuid>.tmp`
    /// shape out of their namespaces (an operator or a foreign tool can leave
    /// one there even where the adapter's own writes never do), so both
    /// override it.
    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        let _ = (object_id, snap_seq);
        Ok(0)
    }

    /// This adapter's own upper bound on the wall-clock cost of publishing
    /// **one** artifact, including every retry and auth attempt it may make
    /// internally.
    ///
    /// Feeds [`reconcile_grace`], which is what separates "this artifact's
    /// sibling publish is still in flight" from "this artifact's publish died".
    /// Only the adapter knows its own retry arithmetic, so only the adapter can
    /// answer — a caller-side constant would be wrong for whichever adapter it
    /// was not written against.
    ///
    /// Synchronous and infallible on purpose: it is a property of the adapter's
    /// configuration, not an observation, and the reconciliation pass consults
    /// it before doing any I/O.
    ///
    /// The default is [`LOCAL_PUBLISH_WORST_CASE`] — appropriate for any adapter
    /// whose publish is a local, un-retried operation.
    fn publish_worst_case(&self) -> Duration {
        LOCAL_PUBLISH_WORST_CASE
    }
}

// Compile-time object-safety guard. If `CheckpointStore` ever grows a
// non-object-safe method (generic, `Self`-by-value, `async fn` without
// `#[async_trait]`, etc.), this function fails to typecheck and the build
// breaks here rather than at the `Arc<dyn CheckpointStore>` plumbing site.
#[allow(dead_code)]
fn _assert_checkpoint_store_object_safe(_store: &dyn CheckpointStore) {}

#[allow(dead_code)]
fn _assert_checkpoint_store_arc_dyn(_store: std::sync::Arc<dyn CheckpointStore>) {}

/// b.9qd — the shared layer-artifact filename grammar and the derived
/// reconciliation grace.
///
/// Both are pure functions consumed by *both* adapters, and both sit directly
/// upstream of a deletion decision, so the cases below are the hostile-input
/// half of the reconciliation pass's safety story: a filename that
/// mis-classifies puts an artifact on the wrong deletion path, and a grace
/// that collapses removes the window that separates "publish in flight" from
/// "publish died".
///
/// The grammar spans two generation families whose names overlap by
/// construction: a tail's `.tail.meta.json` ends in a base's `.meta.json`, and a
/// tail's kind marker sits mid-name in `12.34.tail.<uuid>.tmp`. Suffix-strip
/// order and the strict canonical stem are both load-bearing because of that
/// overlap, and both are what `a_tail_name_never_classifies_as_a_base_kind`
/// pins: a name that crosses families is not a formatting nit but a live
/// artifact of one family sitting on the other family's deletion path.
#[cfg(test)]
mod layer_artifact_tests {
    use super::*;

    fn artifact(object_id: u64, snap_seq: u64, kind: LayerArtifactKind) -> Option<LayerArtifact> {
        Some(LayerArtifact {
            object_id,
            snap_seq,
            kind,
        })
    }

    #[test]
    fn parses_each_durable_artifact_kind() {
        assert_eq!(
            parse_layer_artifact_filename("12.34.ompf"),
            artifact(12, 34, LayerArtifactKind::Ompf)
        );
        assert_eq!(
            parse_layer_artifact_filename("12.34.ext-refs.json"),
            artifact(12, 34, LayerArtifactKind::ExtRefs)
        );
        assert_eq!(
            parse_layer_artifact_filename("12.34.meta.json"),
            artifact(12, 34, LayerArtifactKind::Meta)
        );
    }

    /// A temp wraps a COMPLETE final filename, so the `.tmp` suffix has to be
    /// stripped first.
    ///
    /// If it were not, `12.34.ompf.<uuid>.tmp` would fail the strict
    /// two-component stem check and be skipped — leaking every temp forever —
    /// or, worse, a looser parser would read it as a durable OMPF and make the
    /// stem look like a complete generation it is not.
    #[test]
    fn parses_a_temp_of_each_durable_kind() {
        let uuid = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9";
        for final_name in ["12.34.ompf", "12.34.ext-refs.json", "12.34.meta.json"] {
            assert_eq!(
                parse_layer_artifact_filename(&format!("{final_name}.{uuid}.tmp")),
                artifact(12, 34, LayerArtifactKind::Temp),
                "`{final_name}.<uuid>.tmp` must classify as Temp — it is reclaimed through \
                 delete_layer_temp_artifacts, never through delete_layer_checkpoint"
            );
        }
    }

    /// Anything that is not a layer artifact is skipped rather than guessed
    /// at. `12.3.4.ompf` is the pointed case: a lenient split would truncate
    /// it to `(12, 3)` and hand a real generation's stem to a deletion path.
    #[test]
    fn rejects_malformed_and_hostile_names() {
        for name in [
            "",
            "readme.txt",
            "12.ompf",
            ".34.ompf",
            "12..ompf",
            "12.3.4.ompf",
            "twelve.34.ompf",
            "-1.34.ompf",
            "12.34.OMPF",
            "12.34.ext-refs.JSON",
            "12.34.tmp",
            "12.34.ompf.tmp",
        ] {
            assert_eq!(
                parse_layer_artifact_filename(name),
                None,
                "`{name}` must not classify as a layer artifact — everything this parser \
                 accepts becomes reachable from a delete"
            );
        }
    }

    /// A temp of a temp is not a shape this codebase produces, and admitting
    /// one would let a crafted name nest arbitrarily.
    #[test]
    fn rejects_a_temp_of_a_temp() {
        let uuid = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9";
        assert_eq!(
            parse_layer_artifact_filename(&format!("12.34.ompf.{uuid}.tmp.{uuid}.tmp")),
            None
        );
    }

    /// Both durable tail shapes parse, and a tail's `to_seq` lands in the same
    /// ordering field a base's `snap_seq` lands in.
    ///
    /// The sibling of `parses_each_durable_artifact_kind` for the tail family.
    /// Sharing [`LayerArtifact::snap_seq`] across the families is deliberate,
    /// and it is exactly why the kind has to travel alongside: `(12, 34)` on its
    /// own cannot say whether it names a base at `snap_seq` 34 or a tail whose
    /// `to_seq` is 34, and those are two different generations.
    #[test]
    fn parses_each_durable_tail_artifact_kind() {
        assert_eq!(
            parse_layer_artifact_filename("12.34.tail"),
            artifact(12, 34, LayerArtifactKind::TailEnvelope)
        );
        assert_eq!(
            parse_layer_artifact_filename("12.34.tail.meta.json"),
            artifact(12, 34, LayerArtifactKind::TailMeta)
        );
    }

    /// A tail temp is a temp: `.tmp` is stripped first, the wrapped final name
    /// recurses into a tail kind, and the result folds to the kind-agnostic
    /// [`LayerArtifactKind::Temp`].
    ///
    /// The fold is the claim, not an implementation accident. A temp belongs to
    /// no generation family — there is no `TailTemp` — so a tail temp is the
    /// same reconciliation candidate a base temp at the same numeric stem is,
    /// and one `delete_layer_temp_artifacts` call reclaims both. What it must
    /// never be is a durable tail artifact: that would let a half-written
    /// envelope satisfy the tail stem it is named after.
    #[test]
    fn parses_a_temp_of_each_durable_tail_kind() {
        let uuid = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9";
        for final_name in ["12.34.tail", "12.34.tail.meta.json"] {
            assert_eq!(
                parse_layer_artifact_filename(&format!("{final_name}.{uuid}.tmp")),
                artifact(12, 34, LayerArtifactKind::Temp),
                "`{final_name}.<uuid>.tmp` must classify as the kind-agnostic Temp — it is \
                 reclaimed through delete_layer_temp_artifacts, never through \
                 delete_layer_checkpoint, and it must not satisfy the tail stem it names"
            );
        }
        assert_eq!(
            parse_layer_artifact_filename(&format!("12.34.tail.{uuid}.tmp")),
            parse_layer_artifact_filename(&format!("12.34.ompf.{uuid}.tmp")),
            "a tail temp and a base temp at the same numeric stem are deliberately \
             indistinguishable — the family axis does not exist on a temp candidate"
        );
    }

    /// **The suffix-strip-order test.** `12.34.tail.meta.json` ends in
    /// [`META_SUFFIX`] as well as [`TAIL_META_SUFFIX`], so it is precisely the
    /// name a durable-suffix chain in the wrong order mis-files; `12.34.tail` is
    /// a name the pre-tail grammar rejected outright, so it is the one a
    /// too-eager base arm could swallow.
    ///
    /// Two independent guards stand between those names and a base kind: the
    /// strip order, and the strict canonical stem that refuses the `34.tail`
    /// component a base-first strip leaves behind. Both failure modes are
    /// covered here — a wrong kind trips the `assert_ne!`, and the `None` that a
    /// base-first strip actually produces today (i.e. a silent return to the
    /// pre-tail behaviour, leaking every torn tail's companion) trips the
    /// `expect`.
    ///
    /// Kept out of the parse tables on purpose: in a table a wrong-*kind* result
    /// still reads as "parsed fine", and a wrong kind here is the whole hazard.
    #[test]
    fn a_tail_name_never_classifies_as_a_base_kind() {
        let tail_meta = parse_layer_artifact_filename("12.34.tail.meta.json").expect(
            "a tail's meta companion must parse — returning None here is the pre-tail posture, \
             under which every torn tail's companion leaks permanently",
        );
        assert_ne!(
            tail_meta.kind,
            LayerArtifactKind::Meta,
            "`12.34.tail.meta.json` classified as the BASE Meta kind. TAIL_META_SUFFIX must be \
             stripped before META_SUFFIX, and the stem parser must refuse the `34.tail` \
             component a base-first strip leaves behind; losing both files a live tail's \
             companion against the base stem (12, 34) — a companion of a base generation that \
             need not exist, sitting on the base stem's deletion path"
        );
        assert_eq!(tail_meta.kind, LayerArtifactKind::TailMeta);
        assert_eq!(tail_meta.kind.generation_kind(), Some(CheckpointKind::Tail));

        let tail_envelope = parse_layer_artifact_filename("12.34.tail").expect(
            "a tail envelope must parse — returning None here leaves a durable generation \
             artifact invisible to the reconciliation pass",
        );
        assert_ne!(
            tail_envelope.kind,
            LayerArtifactKind::Ompf,
            "`12.34.tail` classified as the BASE Ompf kind — a tail envelope would then satisfy \
             the base stem (12, 34), telling the reconciliation pass a base generation is \
             complete when not one of its artifacts exists"
        );
        assert_eq!(tail_envelope.kind, LayerArtifactKind::TailEnvelope);
        assert_eq!(
            tail_envelope.kind.generation_kind(),
            Some(CheckpointKind::Tail)
        );
    }

    /// Tail-shaped names that are not tail artifacts, under the same contract as
    /// `rejects_malformed_and_hostile_names`: everything the parser accepts
    /// becomes reachable from a delete.
    ///
    /// The tail family widens the surface twice over. Its kind marker is not the
    /// end of the name (`.tail.meta.json`), so near-miss spellings — `.tail.meta`,
    /// `.tail.json` — must fall out rather than partially match; and its stem is
    /// built from the same two canonical `u64` components as a base's, so the
    /// zero-padded and `+`-prefixed spellings alias a stem that already has a
    /// reconciliation member and would churn a claim slot forever.
    #[test]
    fn rejects_malformed_and_hostile_tail_names() {
        for name in [
            // Stem shape.
            "12.tail",
            ".34.tail",
            "12..tail",
            "12.3.4.tail",
            "twelve.34.tail",
            // Canonical-`u64` aliases of the stem `12.34`.
            "-1.34.tail",
            "+12.34.tail",
            "012.34.tail",
            // Suffix spelling: case-sensitive, and no partial match.
            "12.34.TAIL",
            "12.34.tail.meta",
            "12.34.tail.json",
            "12.34.tail.meta.JSON",
            // A temp carries a UUID disambiguator; a bare `.tmp` is not a shape
            // any adapter writes.
            "12.34.tail.tmp",
            "12.34.tail.meta.json.tmp",
            // The kind marker alone is not a name.
            "tail",
            ".tail",
        ] {
            assert_eq!(
                parse_layer_artifact_filename(name),
                None,
                "`{name}` must not classify as a layer artifact — everything this parser \
                 accepts becomes reachable from a delete, and a non-canonical stem spelling \
                 additionally registers a second, permanent reconciliation member for a stem \
                 that already has one"
            );
        }
    }

    /// SR-2.4 forbids a tail sidecar, so `12.34.tail.ext-refs.json` is a name no
    /// writer in this codebase produces.
    ///
    /// Its rejection is structural rather than special-cased: stripping
    /// [`EXT_REFS_SUFFIX`] leaves the residual stem `12.34.tail`, which the
    /// strict two-component canonical check refuses. Admitting it would both put
    /// an unproduced name on a deletion path and alias the tail's own stem —
    /// a second member at `(12, 34)` in the tail family that nothing will ever
    /// complete.
    #[test]
    fn rejects_the_tail_ext_refs_name_no_writer_produces() {
        assert_eq!(
            parse_layer_artifact_filename("12.34.tail.ext-refs.json"),
            None,
            "a tail carries no ext-refs sidecar (SR-2.4); admitting the name would alias the \
             tail stem (12, 34) with a member no publish ever satisfies"
        );
    }

    /// The tail family's half of `rejects_a_temp_of_a_temp` — nesting must not
    /// become reachable through the tail kinds, which the temp arm reaches by
    /// recursion.
    #[test]
    fn rejects_a_temp_of_a_temp_on_the_tail_family() {
        let uuid = "0a1b2c3d-4e5f-6071-8293-a4b5c6d7e8f9";
        for final_name in ["12.34.tail", "12.34.tail.meta.json"] {
            assert_eq!(
                parse_layer_artifact_filename(&format!("{final_name}.{uuid}.tmp.{uuid}.tmp")),
                None,
                "`{final_name}.<uuid>.tmp.<uuid>.tmp` is not a shape this codebase produces"
            );
        }
    }

    /// The full-`u64` round-trip on a tail stem, for the same reason the base
    /// stem carries one: object ids are minted from a counter with a 2^53
    /// ceiling, but the parser is what a hostile name meets first.
    #[test]
    fn parses_a_full_u64_tail_stem() {
        assert_eq!(
            parse_layer_artifact_filename("18446744073709551615.0.tail"),
            artifact(u64::MAX, 0, LayerArtifactKind::TailEnvelope)
        );
    }

    /// The family axis the reconciliation grouping key is built from.
    ///
    /// A kind mapped to the wrong family would group with the other family's
    /// stem, which is the one mistake the kind was added to the key to prevent:
    /// a base at `S` and a tail whose `to_seq` is `S` share the numeric pair, so
    /// a mis-mapped member lets one family's generation artifact satisfy the
    /// other family's torn stem. `Temp` maps to no family at all — it is its own
    /// candidate and satisfies nothing.
    #[test]
    fn generation_kind_names_each_artifacts_family() {
        for (kind, family) in [
            (LayerArtifactKind::Ompf, Some(CheckpointKind::Base)),
            (LayerArtifactKind::ExtRefs, Some(CheckpointKind::Base)),
            (LayerArtifactKind::Meta, Some(CheckpointKind::Base)),
            (LayerArtifactKind::TailEnvelope, Some(CheckpointKind::Tail)),
            (LayerArtifactKind::TailMeta, Some(CheckpointKind::Tail)),
            (LayerArtifactKind::Temp, None),
        ] {
            assert_eq!(
                kind.generation_kind(),
                family,
                "{kind:?} must report family {family:?} — the grouping key is \
                 (object_id, sequence, generation kind), so a mis-mapped kind lands in the \
                 wrong family's stem group"
            );
        }
    }

    /// The "does this member satisfy its stem" predicate, exactly one artifact
    /// per family.
    ///
    /// Both error directions are damaging and neither is visible from the
    /// grouping code: a companion that satisfied its stem would hide a torn
    /// publish's orphan from the pass forever, and a generation artifact that
    /// did not would make the pass classify a complete, live generation as an
    /// orphan and delete it after the grace.
    #[test]
    fn is_generation_artifact_holds_only_for_a_familys_generation_artifact() {
        for (kind, satisfies_its_stem) in [
            (LayerArtifactKind::Ompf, true),
            (LayerArtifactKind::TailEnvelope, true),
            (LayerArtifactKind::ExtRefs, false),
            (LayerArtifactKind::Meta, false),
            (LayerArtifactKind::TailMeta, false),
            (LayerArtifactKind::Temp, false),
        ] {
            assert_eq!(
                kind.is_generation_artifact(),
                satisfies_its_stem,
                "{kind:?} must{} satisfy its stem",
                if satisfies_its_stem { "" } else { " not" }
            );
        }
    }

    #[test]
    fn stem_parser_is_strictly_two_numeric_components() {
        assert_eq!(parse_layer_artifact_stem("1.2"), Some((1, 2)));
        assert_eq!(
            parse_layer_artifact_stem("18446744073709551615.0"),
            Some((u64::MAX, 0)),
            "the stem must round-trip the full u64 range — object ids are minted from a \
             counter with a 2^53 ceiling, but the parser is what a hostile name meets first"
        );
        for stem in ["1", "1.", ".2", "1.2.3", "a.2", "1.b", "", " 1.2"] {
            assert_eq!(parse_layer_artifact_stem(stem), None, "stem `{stem}`");
        }
    }

    /// The grace is `3 ×` the adapter's own publish bound, floored.
    ///
    /// Both terms matter and are asserted separately: the multiple is what
    /// keeps the window clear of a publish that started just before the first
    /// observation, and the floor is what stops an adapter with an optimistic
    /// `publish_worst_case` from collapsing the two-observation argument into
    /// a window one slow tick could span.
    #[test]
    fn reconcile_grace_is_the_multiple_or_the_floor_whichever_is_larger() {
        assert_eq!(
            reconcile_grace(Duration::from_secs(1)),
            RECONCILE_GRACE_FLOOR,
            "an optimistic adapter bound must not shrink the grace below the floor"
        );
        assert_eq!(
            reconcile_grace(Duration::ZERO),
            RECONCILE_GRACE_FLOOR,
            "and a zero bound certainly must not produce a zero grace — that would act on \
             every candidate the first time it was seen"
        );
        assert_eq!(
            reconcile_grace(Duration::from_secs(30)),
            Duration::from_secs(30 * u64::from(RECONCILE_GRACE_MULTIPLE)),
            "above the floor the grace tracks the adapter's own arithmetic"
        );
        assert_eq!(
            reconcile_grace(Duration::MAX),
            Duration::MAX,
            "saturating, not wrapping: an overflow here would produce a tiny grace"
        );
    }

    /// The local-adapter publish bound feeds the floor branch, so the two
    /// constants are pinned together — raising `LOCAL_PUBLISH_WORST_CASE`
    /// above `RECONCILE_GRACE_FLOOR / 3` would silently start moving the
    /// LocalDisk and InMemory grace off the floor.
    #[test]
    fn local_publish_worst_case_stays_inside_the_floor() {
        assert_eq!(
            reconcile_grace(LOCAL_PUBLISH_WORST_CASE),
            RECONCILE_GRACE_FLOOR
        );
    }
}

/// SR-2.3 — the generation-kind discriminator and the persisted shape of
/// [`CheckpointMeta`].
///
/// Two obligations live here, both of which a symmetric round-trip test would
/// pass while violating:
///
/// 1. **On-disk back-compat (PRD A-6 / SR-2.5).** Every `<oid>.<snap_seq>.
///    meta.json` already sitting in a dev/rc store — LocalDisk's sidecar and
///    the StorageApi adapter's `layer_meta_url` body are the same JSON document
///    — predates `kind`, and several predate `observed_commit_span` and the
///    identity fields before it. Those stores must stay readable with no
///    migration, which is what every `#[serde(default)]` on this struct buys.
///    The literals below are hand-written on purpose: serializing an older
///    struct would only prove the type is self-consistent, whereas a literal
///    pins the bytes independently of the current definition, so dropping a
///    `default` (or adding `deny_unknown_fields`) fails here instead of on an
///    operator's restore.
/// 2. **Kind is part of a generation's identity.** Once a base and a tail can
///    share `(object_id, snap_seq)`, anything that addresses a generation by
///    handle — retention prune, `delete_layer_checkpoint`,
///    `get_layer_checkpoint` — must not conflate them.
///
/// Pure value-type behavior: no filesystem, no network, no Valkey, no runtime.
#[cfg(test)]
mod kind_and_meta_tests {
    use super::*;

    /// A meta with every persisted field set to a distinct, non-default value,
    /// so a field that fails to survive a round-trip cannot hide behind a
    /// coincidental default.
    fn fully_populated_base_meta() -> CheckpointMeta {
        CheckpointMeta {
            creator_replica_id: "live-edit-7c9f".to_string(),
            written_at_unix_ms: 1_717_000_000_000,
            snap_seq: Some(4211),
            session_id: "sess-abc".to_string(),
            layer_name: "root.live".to_string(),
            created_by: "user@example.com".to_string(),
            server_ts: 1_716_999_999_000,
            observed_commit_span: 37,
            kind: CheckpointKind::Base,
            base_snap_seq: None,
            from_seq: None,
            sidecar_bytes: None,
        }
    }

    /// The tail sibling of [`fully_populated_base_meta`]: same identity fields,
    /// lineage populated, `snap_seq` carrying the envelope's `to_seq`.
    fn fully_populated_tail_meta() -> CheckpointMeta {
        CheckpointMeta {
            snap_seq: Some(4290),
            kind: CheckpointKind::Tail,
            base_snap_seq: Some(4211),
            from_seq: Some(4260),
            ..fully_populated_base_meta()
        }
    }

    fn to_json_value(meta: &CheckpointMeta) -> serde_json::Value {
        serde_json::to_value(meta)
            .expect("CheckpointMeta must serialize — both shipped adapters persist it as JSON")
    }

    fn round_trip(meta: &CheckpointMeta) -> CheckpointMeta {
        let json = serde_json::to_string(meta).expect("CheckpointMeta must serialize");
        serde_json::from_str(&json).expect("a meta this crate just serialized must deserialize")
    }

    fn parse_meta(json: &str) -> CheckpointMeta {
        serde_json::from_str(json).expect("meta JSON must deserialize")
    }

    // ── Defaults ─────────────────────────────────────────────────────────────

    /// The whole mechanical compile-through of Task 2 rests on this: ~30
    /// construction sites use `..CheckpointMeta::default()` and never mention
    /// `kind`, so if the default ever flipped, every one of them would silently
    /// start writing tails.
    #[test]
    fn checkpoint_kind_default_is_base() {
        assert_eq!(
            CheckpointKind::default(),
            CheckpointKind::Base,
            "an unannotated construction site must mean 'full-state snapshot' — a default of \
             Tail would make every legacy call site claim a lineage it does not have"
        );
    }

    #[test]
    fn checkpoint_meta_default_is_base_kind_with_absent_lineage() {
        let meta = CheckpointMeta::default();

        assert_eq!(meta.kind, CheckpointKind::Base, "default meta kind");
        assert_eq!(
            meta.base_snap_seq, None,
            "a base has no base to be rooted at"
        );
        assert_eq!(meta.from_seq, None, "a base has no commit-span lower bound");
    }

    #[test]
    fn checkpoint_handle_base_constructor_yields_base_kind() {
        let handle = CheckpointHandle::base(77, 4211, "layers/77.4211.ompf".to_string());

        assert_eq!(handle.kind, CheckpointKind::Base, "handle kind");
        assert_eq!(
            handle.seq(),
            4211,
            "seq() is the kind-neutral read of snap_seq"
        );
    }

    #[test]
    fn checkpoint_handle_tail_constructor_yields_tail_kind_with_to_seq_as_the_ordering_sequence() {
        let handle = CheckpointHandle::tail(77, 4290, "layers/77.4290.tail".to_string());

        assert_eq!(handle.kind, CheckpointKind::Tail, "handle kind");
        assert_eq!(
            handle.snap_seq, 4290,
            "a tail's `to_seq` lands in `snap_seq` so both kinds order and address identically"
        );
        assert_eq!(handle.seq(), 4290, "seq() reads the same number");
    }

    // ── Persisted wire values ────────────────────────────────────────────────

    /// **Persisted-wire-value contract.** `"base"` / `"tail"` are durable bytes
    /// inside every kind-aware `meta.json`, and the same two tokens name Task
    /// 1's tail-envelope header field values, so the spellings move in lockstep
    /// across the meta sidecar, the envelope header, and adapter filename
    /// suffixes.
    ///
    /// Renaming a token is a data-compat break, not a refactor: an existing
    /// `"kind":"tail"` on disk would stop parsing, and the generation would be
    /// stranded. Asserted against exact strings rather than round-tripped —
    /// a round-trip passes under any renaming.
    #[test]
    fn checkpoint_kind_serializes_to_the_pinned_lowercase_tokens() {
        assert_eq!(
            serde_json::to_value(CheckpointKind::Base).unwrap(),
            serde_json::json!("base"),
            "the persisted token for a base generation"
        );
        assert_eq!(
            serde_json::to_value(CheckpointKind::Tail).unwrap(),
            serde_json::json!("tail"),
            "the persisted token for a tail generation"
        );
    }

    /// `as_str` is what tracing fields and (Tasks 3/4) adapter filename
    /// suffixes render, so it must not be allowed to drift away from the
    /// `Serialize` impl — that drift would put one spelling in the meta JSON
    /// and a different one in the artifact name for the same generation.
    #[test]
    fn checkpoint_kind_as_str_matches_the_serialized_token() {
        for kind in [CheckpointKind::Base, CheckpointKind::Tail] {
            let serialized = serde_json::to_value(kind).unwrap();
            assert_eq!(
                serde_json::json!(kind.as_str()),
                serialized,
                "as_str() and Serialize must render the same token for {kind:?}"
            );
        }
    }

    /// The document-level half of the same contract: the *key* is `kind`, and
    /// the value is the pinned token. A `#[serde(rename)]` on the field would
    /// orphan every meta file already on disk exactly as a token rename would.
    #[test]
    fn checkpoint_meta_persists_the_kind_token_under_the_kind_key() {
        let base = to_json_value(&fully_populated_base_meta());
        assert_eq!(
            base.get("kind").and_then(serde_json::Value::as_str),
            Some("base"),
            "base meta JSON must carry `\"kind\":\"base\"` explicitly — the field is \
             deliberately not `skip_serializing_if`, so a forward reader can tell a \
             kind-aware writer from a pre-feature one"
        );

        let tail = to_json_value(&fully_populated_tail_meta());
        assert_eq!(
            tail.get("kind").and_then(serde_json::Value::as_str),
            Some("tail"),
            "tail meta JSON must carry `\"kind\":\"tail\"`"
        );
    }

    // ── Round-trips ──────────────────────────────────────────────────────────

    /// Every persisted field of a base meta survives serialize → deserialize.
    ///
    /// The destructuring is deliberate: a new `CheckpointMeta` field breaks
    /// this pattern at compile time, so it cannot be added without a decision
    /// about whether it round-trips.
    #[test]
    fn checkpoint_meta_base_round_trips_every_persisted_field() {
        let original = fully_populated_base_meta();
        let back = round_trip(&original);

        let CheckpointMeta {
            creator_replica_id,
            written_at_unix_ms,
            snap_seq,
            session_id,
            layer_name,
            created_by,
            server_ts,
            observed_commit_span,
            kind,
            base_snap_seq,
            from_seq,
            sidecar_bytes: _, // asserted separately — deliberately not persisted
        } = back;

        assert_eq!(creator_replica_id, original.creator_replica_id);
        assert_eq!(written_at_unix_ms, original.written_at_unix_ms);
        assert_eq!(snap_seq, original.snap_seq);
        assert_eq!(session_id, original.session_id, "obj:hdr identity field");
        assert_eq!(layer_name, original.layer_name, "obj:hdr identity field");
        assert_eq!(created_by, original.created_by, "obj:hdr identity field");
        assert_eq!(server_ts, original.server_ts, "obj:hdr identity field");
        assert_eq!(
            observed_commit_span, original.observed_commit_span,
            "the measured DR guard band — dropping it silently shrinks every restore of a \
             hot layer back to the configured margin"
        );
        assert_eq!(kind, CheckpointKind::Base, "kind");
        assert_eq!(base_snap_seq, None, "a base carries no lineage");
        assert_eq!(from_seq, None, "a base carries no lineage");
    }

    /// A tail's lineage fields are what listing and retention read to answer
    /// "which base is this rooted at" without fetching and decoding the
    /// envelope, so losing them on the wire strands the generation.
    #[test]
    fn checkpoint_meta_tail_round_trips_its_lineage_fields() {
        let original = fully_populated_tail_meta();
        let back = round_trip(&original);

        assert_eq!(back.kind, CheckpointKind::Tail, "kind");
        assert_eq!(
            back.base_snap_seq,
            Some(4211),
            "the snap_seq of the base this lineage is rooted at"
        );
        assert_eq!(
            back.from_seq,
            Some(4260),
            "the exclusive lower bound of the span this envelope carries"
        );
        assert_eq!(
            back.snap_seq,
            Some(4290),
            "the inclusive upper bound rides snap_seq — there is no separate to_seq field"
        );
    }

    /// `sidecar_bytes` is `skip_serializing` / `skip_deserializing` because the
    /// ext-refs sidecar is its own `<oid>.<snap_seq>.ext-refs.json` file;
    /// persisting it inside the meta too would double-store the payload.
    /// Asserted explicitly so the skip stays a deliberate choice rather than
    /// something a later edit can quietly reverse.
    #[test]
    fn checkpoint_meta_sidecar_bytes_is_not_persisted() {
        let meta = CheckpointMeta {
            sidecar_bytes: Some(Bytes::from_static(b"{\"hashes\":[]}")),
            ..fully_populated_base_meta()
        };

        let json = to_json_value(&meta);
        assert!(
            json.get("sidecar_bytes").is_none(),
            "the ext-refs sidecar must not be duplicated inside the meta JSON, got: {json}"
        );

        let back = round_trip(&meta);
        assert_eq!(
            back.sidecar_bytes, None,
            "and it must come back absent, not empty-but-present"
        );
    }

    // ── On-disk back-compat ──────────────────────────────────────────────────

    /// Every historical on-disk `meta.json` shape must still parse, and must
    /// parse as a base with no lineage — which is exactly what those blobs are
    /// (PRD A-6 / SR-2.5, no migration).
    ///
    /// The three literals are the three shapes a real dev/rc store contains:
    /// the current writer minus `kind`, that minus `observed_commit_span`
    /// (pre-b.vnw), and the original three-field meta written before the
    /// `obj:hdr` identity fields were persisted.
    #[test]
    fn checkpoint_meta_every_pre_kind_on_disk_shape_deserializes_as_base() {
        const PRE_KIND: &str = r#"{
            "creator_replica_id": "live-edit-7c9f",
            "written_at_unix_ms": 1717000000000,
            "snap_seq": 4211,
            "session_id": "sess-abc",
            "layer_name": "root.live",
            "created_by": "user@example.com",
            "server_ts": 1716999999000,
            "observed_commit_span": 37
        }"#;
        const PRE_SPAN: &str = r#"{
            "creator_replica_id": "live-edit-7c9f",
            "written_at_unix_ms": 1717000000000,
            "snap_seq": 4211,
            "session_id": "sess-abc",
            "layer_name": "root.live",
            "created_by": "user@example.com",
            "server_ts": 1716999999000
        }"#;
        const PRE_IDENTITY: &str = r#"{
            "creator_replica_id": "live-edit-7c9f",
            "written_at_unix_ms": 1717000000000,
            "snap_seq": 4211
        }"#;

        for (label, json) in [
            ("pre-kind (current writer minus the kind fields)", PRE_KIND),
            ("pre-observed_commit_span and pre-kind", PRE_SPAN),
            ("earliest three-field meta", PRE_IDENTITY),
        ] {
            let meta = serde_json::from_str::<CheckpointMeta>(json).unwrap_or_else(|e| {
                panic!(
                    "the {label} shape must still deserialize — stores holding it are read \
                     without migration, and this fails the moment a #[serde(default)] is \
                     dropped or #[serde(deny_unknown_fields)] is added: {e}"
                )
            });

            assert_eq!(
                meta.kind,
                CheckpointKind::Base,
                "{label}: a document with no `kind` key is a full-state snapshot"
            );
            assert_eq!(
                meta.base_snap_seq, None,
                "{label}: lineage must stay absent"
            );
            assert_eq!(meta.from_seq, None, "{label}: lineage must stay absent");
        }
    }

    /// The fields a pre-kind document *does* carry survive verbatim, and the
    /// ones it omits fall back to their documented defaults — in particular
    /// `observed_commit_span: 0`, the no-measurement sentinel that folds the
    /// DR guard band back to the configured margin alone.
    #[test]
    fn checkpoint_meta_pre_kind_json_preserves_persisted_fields_and_defaults_absent_ones() {
        let meta = parse_meta(
            r#"{
                "creator_replica_id": "live-edit-7c9f",
                "written_at_unix_ms": 1717000000000,
                "snap_seq": 4211,
                "session_id": "sess-abc",
                "layer_name": "root.live",
                "created_by": "user@example.com",
                "server_ts": 1716999999000
            }"#,
        );

        assert_eq!(meta.creator_replica_id, "live-edit-7c9f");
        assert_eq!(meta.written_at_unix_ms, 1_717_000_000_000);
        assert_eq!(meta.snap_seq, Some(4211));
        assert_eq!(meta.session_id, "sess-abc");
        assert_eq!(meta.layer_name, "root.live");
        assert_eq!(meta.created_by, "user@example.com");
        assert_eq!(meta.server_ts, 1_716_999_999_000);
        assert_eq!(
            meta.observed_commit_span, 0,
            "absent means no measurement, which folds the restore band back to the \
             configured margin — the pre-b.vnw behavior"
        );
    }

    /// `kind` never *requires* the lineage fields on the wire: a base written
    /// by a kind-aware build carries `"kind":"base"` and nothing else, and the
    /// base ⇒ no-lineage invariant is a writer obligation, not a serde one.
    /// A tail missing its lineage likewise parses — validation is a caller
    /// concern (the replay path), so the reader must not fail here.
    #[test]
    fn checkpoint_meta_kind_without_lineage_fields_deserializes_ok() {
        let base = parse_meta(
            r#"{
                "creator_replica_id": "r1",
                "written_at_unix_ms": 1,
                "snap_seq": 9,
                "kind": "base"
            }"#,
        );
        assert_eq!(base.kind, CheckpointKind::Base);
        assert_eq!(base.base_snap_seq, None);
        assert_eq!(base.from_seq, None);

        let tail = parse_meta(
            r#"{
                "creator_replica_id": "r1",
                "written_at_unix_ms": 1,
                "snap_seq": 9,
                "kind": "tail"
            }"#,
        );
        assert_eq!(
            tail.kind,
            CheckpointKind::Tail,
            "the lineage invariant is enforced by the replay path, not by serde — the \
             reader's job is to report the kind that was written"
        );
        assert_eq!(tail.base_snap_seq, None);
        assert_eq!(tail.from_seq, None);
    }

    /// An unrecognized `kind` token MUST fail deserialization (SR-6.1 — readers
    /// never guess).
    ///
    /// This is a decided contract, not an observed behavior. A tolerant
    /// fallback (`#[serde(other)]` → `Base`) would mean an older replica reads
    /// a newer replica's *tail* as a self-sufficient base and restores it
    /// alone: the mis-restore / stranded-generation class SR-2.5 forbids.
    /// Failing instead surfaces as a `CheckpointMeta` parse error, which both
    /// adapters' meta-read arms already classify as
    /// [`CheckpointError::Permanent`] — durable bytes, no retry, operator sees
    /// it. Same posture as the tail envelope's `format_version`.
    #[test]
    fn checkpoint_meta_unknown_kind_token_fails_deserialization() {
        let err = serde_json::from_str::<CheckpointMeta>(
            r#"{
                "creator_replica_id": "r1",
                "written_at_unix_ms": 1,
                "snap_seq": 9,
                "kind": "delta_v2"
            }"#,
        )
        .expect_err("an unknown kind token must not silently degrade to a base");

        assert_eq!(
            err.classify(),
            serde_json::error::Category::Data,
            "a well-formed document carrying an unmodelled kind is a data error — that is \
             what the adapters map to CheckpointError::Permanent"
        );
        assert!(
            err.to_string().contains("unknown variant"),
            "the error must name the offending token so an operator can act on it, got: {err}"
        );
    }

    // ── Handle identity ──────────────────────────────────────────────────────

    /// Kind participates in handle equality.
    ///
    /// A base and a tail can share `(object_id, snap_seq)` — the tail's
    /// ordering sequence is its `to_seq`, and nothing stops it from landing on
    /// a base's snapshot sequence. Retention prune, `delete_layer_checkpoint`,
    /// and `get_layer_checkpoint` all address a generation *by handle*, so if
    /// kind dropped out of `PartialEq` those paths would match the wrong
    /// generation: a wrong-generation delete, not a cosmetic mismatch.
    #[test]
    fn checkpoint_handle_equality_is_kind_sensitive() {
        let base = CheckpointHandle::base(77, 4290, "layers/77.4290".to_string());
        let tail = CheckpointHandle {
            kind: CheckpointKind::Tail,
            ..base.clone()
        };

        assert_ne!(
            base, tail,
            "handles identical in object_id, snap_seq and adapter_key but differing in kind \
             name two distinct generations — conflating them is a wrong-generation delete"
        );
    }

    /// The other half of the same property: kind is not the *only* thing that
    /// distinguishes handles, and two handles naming the same generation still
    /// compare equal.
    #[test]
    fn checkpoint_handle_equality_holds_for_the_same_generation() {
        let a = CheckpointHandle::tail(77, 4290, "layers/77.4290.tail".to_string());
        let b = CheckpointHandle::tail(77, 4290, "layers/77.4290.tail".to_string());

        assert_eq!(a, b, "same object, same sequence, same kind, same key");
        assert_ne!(
            a,
            CheckpointHandle::tail(77, 4291, "layers/77.4291.tail".to_string()),
            "a different ordering sequence is a different generation"
        );
    }
}
