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

//! Checkpoint apply: the restore core `apply_restore`, its bytes-in wrapper
//! `apply_checkpoint`, and the OMPF reset-delta decoder they share (SR-2.5).
//!
//! # How one layer restore flows across files
//!
//! The rules each step must keep, and why, are SDD §22.6; this is the map.
//!
//! 1. `recovery::RecoveryHook::try_restore_layer` takes the per-`object_id`
//!    K=1 admission mutex (per-process only), selects the newest restorable
//!    lineage (`checkpoint::lineage`; a composed one is replayed by
//!    `checkpoint::compose`), and adjudicates the surviving per-object stream:
//!    it probes the stream's high-water, and removes a stream whose top is
//!    implausible only under this module's identity guard. The probe's result
//!    becomes `apply_restore`'s `max_seq_observed`.
//! 2. `apply_restore` (this module) runs, in order: the structural-header
//!    short-circuit, the identity guard, the decode (base-only payloads), the
//!    sequence advance and the step-4a version flattening, then the residue
//!    wipe (`checkpoint::residue`, step 3b), the external-value pre-store of a
//!    composed lineage (step 4b), and the single fenced materialization
//!    `LayerStore::write_restored_object` (`storage::layer`, step 5), which
//!    also stamps the checkpoint cursor columns. Its rustdoc lists the steps
//!    and the error mapping.
//! 3. On `ApplyOutcome::Applied`, the hook hands the in-memory state to
//!    `base_bootstrap::BaseBootstrap::spawn_restore_rebase`, which (when the
//!    bootstrap unit is enabled, i.e. with tail capture on) writes a fresh
//!    base at the seeded sequence and clears `restore_seq_bump`.
//!
//! Cross-replica races are fenced by the two `WATCH`es on `obj:hdr` (the
//! residue wipe's and the materialization's), not by the admission mutex; see
//! `recovery`'s module docs.
//!
//! # The decoder
//!
//! Pure sync primitive: takes the OMPF bytes a `CheckpointStore` returned and
//! reconstructs the in-memory [`ObjectState`] that
//! [`LayerStore::write_restored_object`](crate::storage::layer::LayerStore::write_restored_object)
//! consumes. Zero Valkey, zero async, zero `oodf_ffi::bridge::ffi`.
//! `apply_restore` chains this primitive in front of `write_restored_object`.
//!
//! # Round-trip contract with the assembler
//!
//! OMPF Part 0 is an OODF delta in `is_diff=true, base_version=0` (the reset
//! delta shape `DiffEngine::plan` emits for `DiffPlan::FullContent`). Any
//! other shape (`is_diff=false`, or `is_diff=true` with `base_version != 0`)
//! is a wire-protocol violation and surfaces as
//! [`CheckpointError::Permanent`] — re-reading the same bytes can never
//! convert it to a reset delta, so retry would loop. A base over one delta's
//! entry limit is instead a chunked base (`checkpoint::chunked_base`) whose
//! every chunk must have that same reset shape; the decoder merges the chunks
//! into the one reset delta an unchunked base would have carried.
//!
//! # Why the decoder is pure (no `CheckpointMeta` parameter)
//!
//! The decoder deliberately returns `ObjectState` alone, not
//! `(ObjectState, CheckpointMeta)`: the meta lives in the
//! [`CheckpointBlob`](super::store::CheckpointBlob) returned by
//! [`CheckpointStore::get_layer_checkpoint`] (the recovery hook's
//! restore-source fetch), and the apply path already holds it — there's no
//! benefit to threading it through a function whose only job is parsing the
//! OMPF bytes. Coupling the
//! decoder to `CheckpointMeta` would also conflate "decode bytes" with
//! "fuse OMPF + meta" into a single helper, which is harder to unit-test
//! against the wire format alone. `apply_restore` combines the two at its
//! call site.
//!
//! # Reuse of [`crate::domain::delta::apply_delta_into_mode`]
//!
//! `apply_delta` already implements the reset-path logic the decoder
//! needs: clear `sections` / `children` / `fields` / `time_samples`, set
//! `source_format`, then run the struct commands / set fields / time
//! samples. Re-implementing that here would duplicate the depth check, the
//! name-conflict cascade, the `next_set_order` ceiling, and every other
//! invariant `apply_delta` enforces — and would drift over time. We call
//! `apply_delta` directly, then reset `sequence_num` to 0 (`apply_restore`
//! sets the canonical value per SR-2.5) and `first_version` to 0
//! (likewise — the assembler stored it on `obj:hdr`, not in the delta).
//! The decoder rebuilds state from scratch, not from Valkey, so it wants no
//! persistence mutations at all: it goes through
//! [`crate::domain::delta::apply_delta_into_mode`] with a
//! [`crate::domain::delta::MutationSink::Discard`] sink (b.ot3), which is
//! the same engine with nothing built rather than a list built and dropped,
//! in [`crate::domain::delta::ApplyMode::RestoreBase`], which skips only the
//! per-delta entry cap a whole-layer base can exceed.

#![allow(dead_code)]

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface};
use std::collections::HashMap;

#[cfg(feature = "test-checkpoint-fault-injection")]
use crate::checkpoint::adapters::CheckpointFaultInjector;
use crate::checkpoint::chunked_base;
use crate::checkpoint::compose::ExtValuePart;
use crate::checkpoint::store::{CheckpointError, CheckpointMeta};
use crate::config::Config;
use crate::domain::delta::{apply_delta_into_mode, ApplyDeltaStats, ApplyMode, MutationSink};
use crate::domain::ompf::OmpfContainer;
use crate::domain::types::ObjectState;
use crate::error::ServiceError;
use oodf_ffi::{OodfError, OwnedDelta};
// Named only by this module's rustdoc (intra-doc links resolve through a `use`)
// and by its unit tests, never by its production code — the permanence classifier
// below is deliberately kind-agnostic. Hence the `allow`; dropping the name would
// silently break those links rather than fail the build. Same reasoning as the
// `HEADER_STRUCTURAL_FIELDS` re-export further down.
#[allow(unused_imports)]
use crate::error::KeyspaceRefusal;
use crate::observability::metrics::{
    CHECKPOINT_RESTORE_FENCED_STAGE_OBJECT_WRITE, CHECKPOINT_RESTORE_FENCED_STAGE_PRESENCE_PROBE,
    CHECKPOINT_RESTORE_FENCED_STAGE_RESIDUE_WIPE,
};
use crate::storage::client::{is_wrongtype_reply, FredErrOrigin, ValkeyClient, POOL_LABEL_DATA};
use crate::storage::external_values::ExternalValueStore;
use crate::storage::keys::KeySchema;
use crate::storage::layer::{CheckpointCursorStamp, LayerStore, RestoreWriteOutcome};

/// Lua / JavaScript safe integer ceiling (2^53 − 1).
///
/// The `update_layer_fast.lua` script (and the legacy WATCH/MULTI/EXEC
/// path) reject any commit that would push `seq` or `next_set_order` at or
/// above this value, because Lua's number type is f64 and integers above
/// 2^53 lose precision. `apply_checkpoint`'s SR-2.5 seq advance MUST stay
/// strictly below this ceiling so the very next `UpdateLayer` against the
/// restored object can run; otherwise the script would reject every
/// subsequent commit with a "sequence number overflow" error.
///
/// Pre-flighting the rejection here is purely defensive — a checkpoint
/// that already encodes a `snap_seq` close to 2^53 indicates a runaway
/// writer or a corrupted checkpoint, and the operator surface is the
/// resulting `Permanent` error rather than a silent reject on the next
/// write.
const F64_SAFE_INT_MAX: u64 = 9_007_199_254_740_991; // 2^53 - 1

/// Multiple of `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN` above which a measured
/// `observed_commit_span` earns an operator WARN at restore time (b.vnw).
///
/// `8` is chosen to be roughly an order of magnitude: a layer running 2–4× the
/// configured margin is a routinely hot layer and the whole point of the
/// measured band is to serve it silently, so warning there would be noise on
/// every DR drill. A span eight times the margin is a different statement — the
/// knob is mis-sized against this fleet's real traffic, or (the more likely
/// cause after the sentinel-0 fixes) this generation is a post-outage one
/// spanning many checkpoint intervals rather than one.
///
/// The band is deliberately **not** clamped to this multiple. After the
/// fabricated-baseline cases are sentinelled to 0 at assembly time, a surviving
/// large span is an honest measurement — in the post-outage case it approximates
/// exactly the commit range that was lost — and clamping it would re-open the
/// silent sequence-reuse window (b.8y5 / b.dy5-class staleness) that the
/// measured band exists to close. `F64_SAFE_INT_MAX` stays the only hard bound.
const OUTSIZED_BAND_MARGIN_MULTIPLE: u64 = 8;

/// The OODF parser's refusal of a plain OMPF base's delta, carried as the
/// boxed source of [`decode_checkpoint_ompf`]'s [`CheckpointError::Permanent`].
///
/// Typed rather than rendered so the parser's verdict survives the
/// `Permanent`: [`is_over_entry_limit_base`] reads it to tell a base over the
/// per-delta entry limit from a corrupt one.
#[derive(Debug, thiserror::Error)]
#[error("OODF parse failed during checkpoint apply: {source}")]
pub(crate) struct BaseDeltaParseError {
    /// The parser's error.
    #[source]
    pub(crate) source: OodfError,
}

/// Whether `err` is a base decode failure because the base is a plain OMPF
/// delta over the per-delta entry limit: a `Permanent` carrying a
/// [`BaseDeltaParseError`] whose parser error is
/// [`OodfError::TooManyTables`].
///
/// Only an earlier release writes such a base; this release writes a chunked
/// base above the limit instead. On the composed arm, the layer recovery hook
/// uses this check to refuse that base `Permanent` with no step-down
/// (SR-14.7). Every other decode failure, a corrupt base included, answers
/// `false`.
pub(crate) fn is_over_entry_limit_base(err: &CheckpointError) -> bool {
    match err {
        CheckpointError::Permanent { source } => source
            .downcast_ref::<BaseDeltaParseError>()
            .is_some_and(|e| matches!(e.source, OodfError::TooManyTables { .. })),
        _ => false,
    }
}

/// Decode a checkpoint OMPF blob into an in-memory [`ObjectState`].
///
/// Reads both base artifact shapes: the single OMPF-wrapped reset delta, and
/// the chunked base written for a layer over one delta's entry limit
/// ([`chunked_base`], dispatched on its magic). Both yield the same state for
/// the same layer. This is the one entry point for every reader of base
/// bytes (the restore decode and compose alike).
///
/// A base over the per-delta entry limit is never restored on access
/// (SR-14.7). The layer recovery hook refuses it `Permanent` on both arms,
/// with no step-down, unless its identity verdict finds the layer gone, which
/// is answered not-found like [`apply_restore`]'s own identity guard. A
/// chunked base is refused before it reaches
/// this decoder. A plain OMPF base over the limit, which only an earlier
/// release writes, reaches it and fails here with a typed
/// [`BaseDeltaParseError`]. The hook recognises that error through
/// [`is_over_entry_limit_base`]. The base-only arm surfaces it as
/// `Permanent` from `apply_restore`'s decode, before the residue wipe, the
/// ext pre-store and the write. The composed arm refuses it rather than
/// stepping down.
///
/// `object_id` is the logical identity the caller intends to materialize
/// under; it MUST match the `object_id` argument the caller will pass to
/// [`crate::storage::layer::LayerStore::write_restored_object`]. The OMPF payload
/// does not carry an object id — that lives on `obj:hdr` and on the
/// `CheckpointHandle`. The caller is responsible for passing the same
/// value to both.
///
/// `sequence_num` and `first_version` on the returned state are zeroed —
/// the apply Subtask (per SR-2.5) sets them from `CheckpointMeta::snap_seq`
/// + the configured safety margin before calling `write_restored_object`.
///
/// # Errors
///
/// - [`CheckpointError::Permanent`] when:
///   - the OMPF wire bytes fail to parse (bad magic, truncated, etc.), or a
///     chunked base's framing is malformed
///     ([`chunked_base::ChunkedBaseError`])
///   - the OODF parser rejects the delta payload (a [`BaseDeltaParseError`]
///     on the OMPF path) or any chunk's payload (a
///     [`chunked_base::ChunkedBaseError`] on the chunked path)
///   - the delta (or any chunk) is not a reset delta (`is_diff != true` OR
///     `base_version != 0`), or a chunk after the first carries a
///     `source_format`
///   - `apply_delta` rejects the delta (e.g. tree depth exceeded, name
///     conflict left over from a previously-recovered torn state)
pub(crate) fn decode_checkpoint_ompf(
    object_id: u64,
    ompf: &[u8],
) -> Result<ObjectState, CheckpointError> {
    // A base over one delta's entry limit is a chunked base, not an OMPF. Its
    // decoder parses and shape-checks every chunk and merges them into the one
    // reset delta an unchunked base would have carried (`chunked_base`).
    // Everything else — including bytes that are neither — takes the OMPF
    // path below, exactly as before the chunked format existed.
    if chunked_base::is_chunked_base(ompf) {
        let delta = chunked_base::decode_chunked_reset_delta(ompf)?;
        return apply_reset_base(object_id, delta);
    }

    // Step 1: OMPF parse. Map every parse-error variant onto `Permanent` —
    // the bytes on the wire are what they are; re-reading the adapter's
    // copy of the same bytes will produce the same error.
    let container = OmpfContainer::parse(ompf).map_err(|e| CheckpointError::Permanent {
        source: format!("OMPF parse failed during checkpoint apply: {e}").into(),
    })?;

    // Step 2: OODF parse over Part 0. An OMPF with `num_parts == 0` is
    // explicitly rejected by `OmpfContainer::parse` (it returns
    // `OmpfError::NoParts`), so reaching here means we have a delta-bytes
    // slice; whether it's the empty slice (the "OMPF with zero parts but
    // empty data" degenerate case) is handled by `oodf_ffi::parse_delta`'s
    // `EmptyInput` rejection below. The parser's error stays typed inside the
    // `Permanent` (`BaseDeltaParseError`), so the recovery hook can tell an
    // over-limit base from a corrupt one (`is_over_entry_limit_base`).
    let delta = oodf_ffi::parse_delta(container.delta_bytes).map_err(|source| {
        CheckpointError::Permanent {
            source: Box::new(BaseDeltaParseError { source }),
        }
    })?;

    // Step 3: shape validation. Reset deltas have exactly `is_diff=true`
    // and `base_version=0` — both knobs at the wire level must match.
    // Anything else is a wire-protocol violation (e.g. someone fed an
    // incremental diff into the checkpoint path).
    if !(delta.is_diff && delta.base_version == 0) {
        return Err(CheckpointError::Permanent {
            source: format!(
                "checkpoint OMPF carries a non-reset delta (is_diff={}, base_version={}); \
                 reset deltas require is_diff=true AND base_version=0",
                delta.is_diff, delta.base_version
            )
            .into(),
        });
    }

    apply_reset_base(object_id, delta)
}

/// Step 4 of [`decode_checkpoint_ompf`], shared by its OMPF and chunked-base
/// paths: rebuild an [`ObjectState`] from a validated reset delta.
fn apply_reset_base(object_id: u64, delta: OwnedDelta) -> Result<ObjectState, CheckpointError> {
    // Step 4: reconstruct ObjectState by applying the reset delta against
    // a fresh empty state with the caller-supplied object_id pre-stamped.
    // `apply_delta` will:
    //   - Run `apply_reset` (because is_diff && base_version == 0) which
    //     wipes the (already-empty) state.
    //   - Bump `state.sequence_num` from 0 to 1 and `state.first_version`
    //     from 0 to 1. We reset both back to 0 below so the caller's SR-2.5
    //     `restore_floor + band` seed is the authoritative sequence value.
    //   - Apply every struct command / set field / time sample exactly as
    //     it would on the live write path — including depth-check and
    //     name-conflict resolution. Any rejection here means the
    //     checkpoint bytes encode an invalid state (corruption / cross-
    //     version drift); surfacing as `Permanent` is correct because the
    //     bytes can't be re-parsed into a valid state by retry.
    //   - Run in `ApplyMode::RestoreBase`: every per-item rule and the depth
    //     check, but not the per-delta entry cap — a base describes a whole
    //     layer, and a chunked base merges into one delta over that cap. A
    //     plain OMPF base is at or below the cap by construction, so the mode
    //     changes nothing for it.
    let mut state = ObjectState {
        object_id,
        ..ObjectState::default()
    };
    // b.ot3: the decoder rebuilds state from the checkpoint bytes and writes
    // it out itself, so the mutation list was always dropped — `Discard`
    // makes that true by construction instead of by convention. This is the
    // largest discarded list in the service (a whole-document base
    // checkpoint, decoded inside `spawn_blocking`).
    let mut sink = MutationSink::Discard;
    let mut stats = ApplyDeltaStats::default();
    apply_delta_into_mode(
        &mut state,
        Some(delta),
        &mut sink,
        &mut stats,
        ApplyMode::RestoreBase,
    )
    .map_err(|e| CheckpointError::Permanent {
        source: format!("apply_delta rejected reset delta during checkpoint apply: {e}").into(),
    })?;

    // Per `decode_checkpoint_ompf`'s doc, the caller (the apply Subtask) owns the
    // authoritative sequence / first-version values from `CheckpointMeta`.
    // Zero them here so the decoder's surface is "structural state only".
    state.sequence_num = 0;
    state.first_version = 0;

    Ok(state)
}

/// SR-2.4 guard: return `Ok(true)` iff the live layer mapping named by the
/// blob's **own** identity — `le:{meta.session_id}:layer:{hex(meta.layer_name)}`
/// — currently resolves to `object_id`.
///
/// Three failure modes drive this guard:
///
/// 1. **DeleteLayer-checkpoint-delete race.** A checkpoint was written, then
///    a `DeleteLayer` RPC reaped both the live layer mapping and the
///    `obj:hdr`. Re-materializing the object on recovery would resurrect
///    deleted-by-the-user state — surfacing as a phantom layer in the
///    next `ListLayers`. The mapping is gone, so the GET misses and the
///    apply is refused.
///
/// 2. **CreateLayer-overwrite during state-loss.** A checkpoint was written
///    against `object_id = N`, the keyspace was lost, and on recovery a
///    new `CreateLayer` for the same `(session, layer_name)` minted a fresh
///    `object_id` before the apply ran. The live mapping now points at the
///    new id, so the value comparison fails and `N` is not resurrected as
///    an orphan shadowing post-recovery state.
///
/// 3. **Object-id reuse across a keyspace loss (b.b2j).** A wipe under
///    surviving pods used to restart allocation at `1`, so a *different*
///    layer could hold the id a surviving blob was written under. The
///    allocator now fails closed, but this guard is the second line: it
///    proves the blob's own `(session_id, layer_name)` is what resolves to
///    the id, so one layer's content can never be materialized under
///    another layer's identity.
///
/// # Why a targeted GET rather than a keyspace scan
///
/// The pre-b.b2j guard SCANned every `le:{*}:layer:*` key looking for *any*
/// value equal to `object_id` — which answers "is this id live somewhere",
/// not "is this id live as *this blob's* layer", and so could not detect
/// case 3 at all. There is no RenameLayer RPC, so `(session_id, layer_name)`
/// is immutable for a given `object_id`: the targeted read is *equivalent* to
/// the scan in the honest case and *strictly stronger* under reuse. It is
/// also one single-key `{session_id}`-tagged GET instead of a cluster-wide
/// enumeration, which removes the old 100_000-key scan cap — a large enough
/// keyspace used to make the guard silently refuse legitimate restores.
///
/// # Empty blob identity
///
/// A blob whose `meta.session_id` or `meta.layer_name` is empty carries no
/// identity to verify. Such a blob is refused here, at ERROR: writing an
/// identity-less `obj:hdr` — content restored under an id nothing can prove
/// belongs to it — is a worse outcome than a refused restore.
///
/// **As of b.hka this guard has no shipped producer left**, and the check is
/// retained as the last line rather than as the working refusal it used to be.
/// The three that existed were each closed at their source, which is strictly
/// better than refusing three layers up with a verdict that names neither the
/// artifact nor the reason:
///
/// - both adapters' missing-meta-companion read fallback — the
///   `is_not_found()` arm of the meta GET in `read_generation_at`
///   (`local_disk.rs`, `storage_api.rs`), for BOTH kinds — now returns
///   [`CheckpointError::Permanent`] carrying a typed
///   [`MissingMetaCompanion`](crate::checkpoint::store::MissingMetaCompanion)
///   instead of synthesizing `CheckpointMeta::default()`. That is also what
///   lets `checkpoint::lineage` recognise the one unusable generation and step
///   down to the next-older restorable one;
/// - the assembler's fabricated-baseline arm (taken when `obj:hdr` vanished
///   mid-assembly) now declines to assemble at all, returning
///   [`CheckpointError::NotFound`] into the checkpoint loop's
///   benign-and-terminal arm rather than writing a permanently unrestorable
///   generation;
/// - and [`crate::checkpoint::adapters::validate_generation_identity`] rejects
///   an identity-empty put at every adapter, so a future producer cannot
///   reintroduce one without failing loudly at the write.
///
/// Epic 4 keeps that true concretely: the composed restore path hands
/// [`apply_restore`] a meta built as `base.meta.clone()` — so every identity and
/// header column (`session_id`, `layer_name`, `created_by`, `server_ts`) is
/// **base-sourced**, and this guard only ever reads base-sourced fields. Exactly
/// two **non-identity** columns are head-generation-sourced —
/// `observed_commit_span` (from the head tail's *envelope header*, which no meta
/// can contradict) and `written_at_unix_ms` (the head tail's own write time,
/// falling back to the base's on the `0` sentinel). Neither is read here; the
/// sourcing itself is done by `RecoveryHook::select_restore_source`.
///
/// **This is a deliberate behaviour change, not a restatement of the old
/// guard.** The pre-b.b2j scan guard never read `meta` at all: it asked only
/// whether *some* live mapping resolved to the `object_id`. A blob whose meta
/// sidecar is missing used to restore successfully; since b.b2j it was refused
/// here as `NotFound`, and since b.hka it is refused one layer lower still, at
/// the adapter, with an error that names the artifact and the missing
/// companion. On whichever adapter is configured, the fix for an operator
/// facing that refusal is unchanged: let the checkpoint loop write a current
/// generation *with* its meta sidecar; do not relax the guard.
///
/// Per SR-2.6, the caller holds the per-`object_id` admission lock around this
/// guard + the subsequent `write_restored_object` call, so concurrent writers cannot
/// race in between this check and the write. This function acquires NO locks
/// of its own.
///
/// # Errors
///
/// Every Valkey error on the mapping `GET` is classified by
/// [`classify_object_key_read_error`], the one predicate every restore-path
/// read shares:
///
/// - `WRONGTYPE` on `le:{sid}:layer:{name}` → [`CheckpointError::Permanent`].
///   No retry re-types a key, so folding it to `Transient` would make the
///   recovery hook re-drive a restore that cannot succeed, forever (b.f4t
///   sub-task 2).
/// - Every other Valkey error → [`CheckpointError::Transient`]. The apply
///   path's retry loop is the right place to handle these; the guard does not
///   retry internally.
pub(crate) async fn live_mapping_resolves_blob_identity(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    meta: &CheckpointMeta,
) -> Result<bool, CheckpointError> {
    if meta.session_id.is_empty() || meta.layer_name.is_empty() {
        tracing::error!(
            object_id,
            session_id = %meta.session_id,
            layer_name = %meta.layer_name,
            "apply_checkpoint identity guard: checkpoint blob carries an empty session_id or \
             layer_name, so its identity cannot be verified against a live layer mapping; \
             refusing to materialize an identity-less object header"
        );
        return Ok(false);
    }

    let mapping_key = keys.layer_mapping(&meta.session_id, &meta.layer_name);
    let mapped: Option<String> = client.pool.get(&mapping_key).await.map_err(|e| {
        // b.f4t sub-task 2, the last restore-path read that still folded a
        // `WRONGTYPE` to `Transient`: that verdict becomes
        // `RecoveryOutcome::Error` → a retriable `UNAVAILABLE`, and no retry
        // re-types a key, so the client spins on the identical error forever.
        // Note this key is under the *session* hash tag, not the object's —
        // which is why the shared classifier takes the key it read rather than
        // asserting an `le:{object_id}:obj:*` family.
        classify_object_key_read_error(
            FredErrOrigin::client(client, POOL_LABEL_DATA),
            "GET layer mapping (restore identity guard)",
            &mapping_key,
            object_id,
            e,
        )
    })?;

    let expected = object_id.to_string();
    match mapped {
        Some(oid_str) if oid_str == expected => Ok(true),
        Some(oid_str) => {
            tracing::warn!(
                object_id,
                session_id = %meta.session_id,
                layer_name = %meta.layer_name,
                live_object_id = %oid_str,
                "apply_checkpoint identity guard: the blob's layer resolves to a DIFFERENT \
                 object_id than the blob was written under; refusing the restore"
            );
            Ok(false)
        }
        None => Ok(false),
    }
}

/// The structural-header predicate and the field list behind it (b.f4t
/// sub-task 3): the `obj:hdr` fields whose presence the live read paths require
/// before they will serve an object.
///
/// Both are `contains_key` checks in `storage::layer`, the list is the
/// **union** of the two, and it is declared in the `storage` module beside the
/// read paths whose guards it unions — see
/// [`crate::storage::layer::HEADER_STRUCTURAL_FIELDS`] for the per-field
/// argument.
///
/// Re-exported here rather than restated (b.6de sub-task 6, landed with b.h4m):
/// this lane's three gates ([`header_is_structurally_complete`], the residue
/// wipe's `WATCH`ed re-read, and the materialization fence's) must read the
/// same list the storage read paths enforce, and a hand-mirrored copy is how a
/// third structural field added storage-side silently re-creates the b.f4t
/// defect — a partial header reported as `Restored`.
///
/// The field list itself is re-exported alongside the predicate even though no
/// code in this module names it: this module's rustdoc — and its test lane's —
/// document their fixtures in terms of "the two fields
/// [`HEADER_STRUCTURAL_FIELDS`] names", and an intra-doc link resolves through
/// the `use`. Hence the `allow`; dropping the name would silently break those
/// links rather than fail the build.
#[allow(unused_imports)]
pub(crate) use crate::storage::layer::{
    header_fields_are_structurally_complete, HEADER_STRUCTURAL_FIELDS,
};

/// Is `le:{object_id}:obj:hdr` present **and structurally complete**?
///
/// The presence predicate for the whole recovery path (b.f4t sub-task 3),
/// replacing the bare `EXISTS obj:hdr` that both
/// [`apply_restore`]'s step 1 and `RecoveryHook::is_layer_present` used to run.
///
/// A header that exists but lacks one of [`HEADER_STRUCTURAL_FIELDS`] is
/// reported ABSENT, because that is what its readers already report: a partial
/// header makes `read_object_header` / `stat_layer_for_object_id` answer
/// `NotFound`, `StatLayer` and `SubscribeReadLayer` route that into the
/// recovery hook, and a bare `EXISTS` there answered "already restored" for a
/// restore that never ran — ticking `result="restored"` and handing the handler
/// its retry back with the identical error.
///
/// **Policy (ratified, b.f4t): on a structurally-partial header, recovery
/// RESTORES.** It does not refuse. The SR-2.4 identity guard and the b.4et
/// stream adjudication still run, and `write_restored_object`'s single `HSET` stamps all
/// 12 canonical fields, so the restore *repairs* the header; the b.4et
/// high-water probe supplies `max_seq_observed`, so the seeded cursor cannot
/// rewind below what the surviving stream proves was issued.
///
/// One `HGETALL` where the old check was one `EXISTS` — same round-trip count,
/// on the same multiplexed data pool.
pub(crate) async fn header_is_structurally_complete(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
) -> Result<bool, CheckpointError> {
    let header_key = keys.obj_header(&object_id.to_string());
    let fields: HashMap<String, String> = client.pool.hgetall(&header_key).await.map_err(|e| {
        // Never a bare `Transient`: a wrong-typed `obj:hdr` cannot be HGETALL'd
        // by any retry, and this gate runs ahead of the `Permanent` arm in
        // `write_object_err_to_checkpoint_err` that would otherwise have caught
        // it (the materializers' shared keyspace probe — `ProbeSite::Create`,
        // which the restore runs too — already classifies `obj:hdr` as
        // `ProbedType::Hash`). See `classify_object_key_read_error`.
        classify_object_key_read_error(
            FredErrOrigin::client(client, POOL_LABEL_DATA),
            "HGETALL obj:hdr (structural presence check)",
            &header_key,
            object_id,
            e,
        )
    })?;
    Ok(header_fields_are_structurally_complete(&fields))
}

/// Operator-facing remediation carried on every wrong-typed-key refusal the
/// restore path raises.
///
/// Deliberately its own sentence rather than a reach into the storage lane's
/// private `KEYSPACE_TYPE_ERROR_REMEDIATION_CREATE`: this refusal names a
/// different actor (a *read* of a key the restore path already depends on — an
/// `le:{object_id}:obj:*` key, or the `le:{sid}:layer:{name}` mapping that
/// names the object — not a probe of keys a commit is about to write) and a
/// different consequence (the restore did not run at all, and nothing was
/// deleted).
///
/// It points at the emitted `key` field rather than describing the key's
/// family, because not every routed read is under the `object_id`: the SR-2.4
/// identity guard reads the layer mapping, which is under the *session* hash
/// tag. Valkey's `WRONGTYPE Operation against a key holding the wrong kind of
/// value` reply carries no key name of its own, so the `key` and `command`
/// fields on the ERROR line are the operator's only locators.
pub(crate) const OBJECT_KEY_WRONGTYPE_REMEDIATION: &str =
    "the restore did not run and nothing was deleted; the key named on this line's `key` field \
     holds a foreign Valkey type, so it cannot be read and the restore therefore cannot proceed. \
     Inspect that key, establish what wrote it, remove it, then let recovery \
     retry. Retrying before that cannot succeed — this refusal is permanent by construction";

/// Classify one failed read on the restore path — an `le:{object_id}:obj:*`
/// key, or the `le:{sid}:layer:{name}` mapping the identity guard resolves:
/// `WRONGTYPE` → [`CheckpointError::Permanent`], everything else →
/// [`CheckpointError::Transient`].
///
/// **One predicate, every restore-path read THIS lane owns.** The structural-header
/// gates it serves ([`header_is_structurally_complete`] here,
/// `RecoveryHook::is_layer_present` which delegates to it, and the residue wipe's
/// post-`WATCH` re-read and its tripped-`EXEC` re-read in [`super::residue`]),
/// the SR-2.4 identity guard's layer-mapping `GET`
/// ([`live_mapping_resolves_blob_identity`]), and the wipe's five index
/// enumerations all route through this function, so a wrong-typed key gets the
/// same verdict wherever it is first touched. A `Transient` at any one of them
/// re-creates this ticket's own sub-task 2 defect at a new gate: the recovery
/// hook maps `Transient` to a retriable `UNAVAILABLE`, so the client retries
/// into the identical error forever while
/// `checkpoint_recovery_attempts_total{result="error"}` ticks each time, and no
/// retry can ever change a key's Valkey type.
///
/// **Two restore-path reads are deliberately NOT routed here** (b.h4m): the
/// header fence inside `LayerStore::write_restored_object` reads `obj:hdr` twice
/// on its watched connection — once before the `MULTI`, once after a tripped
/// `EXEC` — and both classify through that module's own
/// `classify_restore_header_read_error` instead. Not an omission, and not a
/// second predicate: the storage-side helper applies the same `WRONGTYPE` test
/// (`storage::client::is_wrongtype_reply`, shared by both) and reaches the
/// identical verdict, but it returns `ServiceError` rather than
/// [`CheckpointError`]: calling this function from `storage` would add a
/// `storage → checkpoint` call edge, which the inversion rule for new edges out
/// of `storage` keeps out (SDD §3.1). It answers
/// [`ServiceError::PermanentKeyspaceRefusal`] carrying
/// [`KeyspaceRefusal::WrongType`], which [`write_object_err_to_checkpoint_err`]
/// maps to the same `Permanent` this function would have produced — that mapping
/// is the join point. Edit either verdict and edit both.
///
/// Kind-gated AND prefix-anchored, never a bare substring search — the
/// anti-creep rule `storage::client`'s error predicates follow, here through
/// `storage::client::is_wrongtype_reply`. fred's `pretty_error` maps a leading
/// `WRONGTYPE` token to `ErrorKind::InvalidArgument` and stores the whole
/// server reply as `details` (`fred-10.1.0/src/protocol/utils.rs:95-129`), so
/// both halves are available and each rules out a different false positive:
/// the kind alone would catch fred's own local argument-validation errors, and
/// the prefix alone would catch any message that merely quotes the word.
///
/// `label` names the command and the key family, the way the wipe's enumeration
/// already does (`"SMEMBERS obj:secs"`); `key` is the *actual* key that failed,
/// verbatim. Both are emitted, because the label alone is not a locator on a
/// path whose reads are not all under one hash tag, and Valkey's `WRONGTYPE`
/// reply names no key at all. Together they are what an operator reads to know
/// which key to inspect.
///
/// `origin` counts a command timeout (see `storage::client::FredErrOrigin`); it
/// never changes the verdict.
pub(crate) fn classify_object_key_read_error(
    origin: FredErrOrigin<'_>,
    label: &'static str,
    key: &str,
    object_id: u64,
    e: fred::error::Error,
) -> CheckpointError {
    origin.observe_error(&e);
    if is_wrongtype_reply(&e) {
        tracing::error!(
            object_id,
            command = label,
            key,
            detail = %e.details(),
            remediation = OBJECT_KEY_WRONGTYPE_REMEDIATION,
            "checkpoint restore REFUSED: a key this restore must read holds the wrong Valkey \
             type, so it cannot be read and the restore cannot proceed. Nothing was deleted and \
             nothing was written"
        );
        return CheckpointError::Permanent {
            source: format!(
                "restore path refused for object {object_id}: {label} on key '{key}': {}",
                e.details()
            )
            .into(),
        };
    }
    CheckpointError::Transient {
        source: format!(
            "restore path read failed for object {object_id}: {label} on key '{key}': {e}"
        )
        .into(),
    }
}

/// Map a `ServiceError` produced by [`LayerStore::write_restored_object`] into
/// the checkpoint-apply error contract.
///
/// Named for `write_object` because that is where the restore materialization
/// lived until b.h4m split the two entry points apart; the emitted variants and
/// their meanings are unchanged, and the fence's own refusals were deliberately
/// spelled in the same variants so this mapping did not have to grow an
/// arm (see that function's exit table).
///
/// # What the materializer actually emits
///
/// Four variants, not one (the pre-b.f4t doc here claimed `Internal` only,
/// which stopped being true when b.zvd added a transient arm at the create
/// site; b.6de then lifted the structural refusals out of `Internal` into a
/// variant of their own):
///
/// * [`ServiceError::PermanentKeyspaceRefusal`] — the two structural refusals
///   detailed under *The permanent arm* below: the pre-`MULTI` keyspace type
///   probe's `WRONGTYPE` under `ProbeSite::Create` (which the restore runs too),
///   and the header fence's own wrong-typed `obj:hdr`.
/// * `Internal` — everything queue-time (`HSET section failed: …`), and only
///   that: since b.6de no structural refusal reachable from this materializer
///   rides this variant.
/// * `Unavailable` — a *transient* failure of the pre-`MULTI` keyspace type
///   probe or the `obj:ext` pre-image read, via `map_fred_err_for_grpc`
///   (`storage/layer.rs`), plus every one of the header fence's own retriable
///   refusals (b.h4m: dirty header, L2 refuse, tripped `EXEC`, `EXEC` error
///   frame, in-doubt `EXEC`). The materializer runs no retry loop of its own, so
///   its transient posture is `Unavailable`, never `Aborted`.
/// * `NotFound` — never produced today; forwarded verbatim for forward-compat.
///
/// # The permanent arm (b.f4t sub-task 2; typed in b.6de sub-task 2)
///
/// Some of the materializer's refusals are permanent **by construction** — an
/// operator has to remove a key before any retry can succeed — and those arrive
/// as [`ServiceError::PermanentKeyspaceRefusal`], carrying a
/// [`KeyspaceRefusal`] discriminant:
///
/// * [`KeyspaceRefusal::WrongType`] — a key this materialization would write, or
///   the `obj:hdr` its fence must read, holds a foreign Valkey type (b.zvd's
///   keyspace type probe, and since b.h4m the fence's own `HGETALL obj:hdr`,
///   which reuses the same classification rather than minting a second one).
/// * [`KeyspaceRefusal::StreamIdCollision`] — the per-object stream already
///   holds an entry at or above the ID a materialization would write (b.ph9).
///
/// Folding those into `Transient` (the pre-b.f4t behaviour) makes the recovery
/// hook re-drive a restore that cannot succeed on every attempt: the
/// keyspace-type-error metric re-ticks, the ERROR line re-emits, and the client
/// sees a retriable `UNAVAILABLE` for a condition no retry clears. `Permanent`
/// routes it to `INTERNAL` through `ServiceError::from_recovery_error`, which is
/// the honest answer.
///
/// **The match is kind-agnostic on purpose.** Both kinds are permanent for the
/// same reason and carry the same remediation shape ("remove or clear the key
/// this names"), so branching on the discriminant here would buy nothing and
/// would make a third kind a silent fall-through into `Transient` — precisely
/// the failure this arm exists to prevent. The discriminant is carried for
/// consumers that need to *name* the condition, not for this one.
///
/// [`KeyspaceRefusal::StreamIdCollision`] is **unreachable from the restore path
/// by construction**, not merely unreached today: the stream `EXISTS` guard
/// whose condition this kind names lives in `LayerStore::write_object`, the
/// create-only materializer (and still raises an untyped `Internal` there —
/// b.p4t), and `apply_restore` calls [`LayerStore::write_restored_object`] — a different
/// function, which queues no `XADD` at all (b.h4m). Covering it anyway costs one
/// word of match, and means a future restore path that does seed a stream entry
/// cannot silently regress into the misclassification this arm exists to fix.
///
/// Everything else stays `Transient` so the recovery hook errs on the side of
/// retry rather than dropping a recoverable layer — including a plain
/// [`ServiceError::Internal`] whose *text* happens to quote one of the reply
/// tags. That is not a gap: five emitters not reachable from
/// [`LayerStore::write_restored_object`] still raise tagged `Internal`s (see the
/// variant's rustdoc — the predicate is reachability from *this* materializer,
/// not "outside the materialization path": `write_object`'s create-path stream
/// `EXISTS` guard is itself in a materializer, just not in this one), and the
/// point of the typed variant is that this classifier no longer has to guess
/// which.
fn write_object_err_to_checkpoint_err(err: ServiceError) -> CheckpointError {
    match err {
        ServiceError::NotFound(_) => CheckpointError::NotFound,
        // `detail`, not the variant's `Display`: it is the emitter's message
        // verbatim, so this `source` string stays byte-identical to the one the
        // pre-variant tag match produced from `ServiceError::Internal(msg)`.
        ServiceError::PermanentKeyspaceRefusal { detail, .. } => CheckpointError::Permanent {
            source: format!(
                "write_restored_object refused the restore with a structural keyspace refusal no \
                 retry can clear: {detail}"
            )
            .into(),
        },
        other => CheckpointError::Transient {
            source: format!("write_restored_object failed during apply_checkpoint: {other}").into(),
        },
    }
}

/// Map a `ServiceError` produced by
/// [`ExternalValueStore::store_values`](crate::storage::external_values::ExternalValueStore::store_values)
/// during the SR-3.4 ext pre-store into the checkpoint-apply error contract.
///
/// `Transient` for everything: the pre-store is content-addressed and
/// idempotent, so a Valkey blip mid-store is exactly the case the recovery
/// hook's next attempt re-runs cleanly. Classifying it `Permanent` would strand
/// a restorable lineage on a backend hiccup — the failure class SR-2.5 and FR-6
/// forbid.
fn ext_prestore_err_to_checkpoint_err(err: ServiceError) -> CheckpointError {
    CheckpointError::Transient {
        source: format!("ext pre-store failed during composed checkpoint restore: {err}").into(),
    }
}

/// Op label for the SR-3.4 fault seam: fires AFTER the ext pre-store
/// `store_values` returned `Ok` and BEFORE the single `write_restored_object`
/// MULTI/EXEC. Task 6's AC-8 matrix drives it to prove no partially-
/// applied object is observable and that the retry converges.
///
/// Spelled **once**, here, so no consumer can misspell it. The seam fires only
/// on the arm where a pre-store actually ran (a composed payload with a
/// non-empty ext set and an [`ExtPrestore`] wired); a lineage carrying no
/// external values has no pre-store to sit behind and therefore no seam.
pub const RESTORE_SEAM_AFTER_EXT_PRESTORE: &str =
    "restore_after_ext_prestore_before_write_restored_object";

/// Op label for the b.f4t fault seam **inside** the step-3b residue wipe's
/// `WATCH` … `EXEC` window: fires after the post-`WATCH` `obj:hdr` re-read has
/// passed the structural predicate and after the five index enumerations have
/// completed, immediately before the wipe's `EXEC` is dispatched.
///
/// This is the only seam in the repo that sits *between* a `WATCH` and its
/// `EXEC`, and that position is the whole point: it is what makes the wipe's
/// tripped-`EXEC` arm deterministically reachable. Its two verdicts — a
/// structurally-complete header ⇒ [`ApplyOutcome::AlreadyPresent`], a header
/// that changed but is still partial (a racing `CreateLayer` overwrite's
/// `HSET superseded=1`, most concretely) ⇒ [`CheckpointError::Transient`] —
/// are otherwise separated only by which write happened to land in a window
/// made of real round-trips, so no test could pick one without a sleep.
///
/// Delivery precondition: the injector rides [`ExtPrestore`], so a recovery hook
/// that calls `with_restore_fault_injector_for_test` must ALSO wire
/// `with_ext_value_store` — the injector is installed only inside the
/// `ext_value_store.as_ref().map(..)` arm, so without an ext store this seam is
/// inert and never fires.
///
/// An injector reaching this seam therefore has two ways to drive it:
///
/// - **Return `Ok(())` after mutating `obj:hdr`** (from a blocking context —
///   `tokio::task::block_in_place` + `Handle::current().block_on`, since
///   `CheckpointFaultInjector::maybe_inject` is synchronous). `block_in_place`
///   **panics on a `current_thread` runtime**, so an arm using this route must
///   run under `#[tokio::test(flavor = "multi_thread")]` — every arm in the
///   matrix does today, and a new one that forgets fails loudly rather than
///   silently skipping the injection. The wipe's `EXEC` then trips on the
///   `WATCH`, re-reads the header, and reaches whichever of the two verdicts the
///   injected write selected.
/// - **Return `Err(..)`.** The wipe releases its `WATCH` (`unwatch_pinned`,
///   the same disposition as the L2 pre-dispatch `Refuse` arm) and propagates
///   the error verbatim, having wiped nothing.
///
/// Spelled **once**, here, beside its siblings, so no consumer can misspell it.
pub const RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC: &str =
    "restore_residue_wipe_after_watch_reread_before_exec";

/// Op label for the b.h4m fault seam on the **stale-stream removal's
/// peer-presence re-read**: fires inside `RecoveryHook::remove_stale_stream`,
/// after the SR-2.4 identity gate authorized the removal and immediately
/// **before** the structural `HGETALL obj:hdr` re-read that now guards the
/// verified `UNLINK`.
///
/// # What it exists to make reachable
///
/// The window this seam opens is the one b.h4m narrowed. Step 5 of
/// `try_restore_layer` (the checkpoint-store blob fetch) is multi-second I/O; a
/// peer replica can land the whole rebuild inside it, so by the time this
/// replica reaches step 6 the header EXISTS, writers are admitted again, and a
/// client's `XADD` can be sitting on the winner's *live* stream — which the
/// plausibility gate reads as stale and the identity gate (same `object_id`,
/// same mapping) happily authorizes destroying. The re-read this seam sits in
/// front of is what turns that into a skip.
///
/// Reproducing it without a seam means winning a race against a real blob
/// fetch, so this is the only deterministic way to drive the two arms apart:
///
/// - **Land a peer's complete `obj:hdr` from inside the injector, then return
///   `Ok(())`.** `CheckpointFaultInjector::maybe_inject` is synchronous, so a
///   test doing Valkey work here needs `tokio::task::block_in_place` +
///   `Handle::current().block_on` and therefore
///   `#[tokio::test(flavor = "multi_thread")]` (`block_in_place` panics on a
///   `current_thread` runtime) — the same route
///   [`RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`] documents. The re-read then sees
///   the peer's header, the `UNLINK` is skipped, and `apply_restore`
///   short-circuits at its step-1 presence probe.
/// - **Return `Err(..)`.** The seam fires inside the same
///   `RECOVERY_STREAM_PROBE_RETRY` driver its `RECOVERY_SEAM_*` neighbours use,
///   at the top of every attempt, so an injected
///   [`CheckpointError::Transient`] is a retriable blip (fail N attempts, let
///   the (N+1)th through) while any other injected kind is terminal on the
///   first attempt. Either way, once the attempts are spent the restore is
///   refused having removed nothing.
///
/// Reachable only on the plausibility-refusal branch: a test must first stage a
/// stale stream top (an auto-ID tombstone, or any ID more than
/// `MAX_PLAUSIBLE_HIGH_WATER_GAP` above the checkpoint's `snap_seq`) on a layer
/// whose live mapping still resolves to the same `object_id`.
///
/// Delivery precondition, and it is WEAKER than
/// [`RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`]'s: this seam is fired straight off
/// the hook's own injector, not off one riding [`ExtPrestore`], so
/// `with_restore_fault_injector_for_test` alone arms it — no
/// `with_ext_value_store` is required.
///
/// Spelled **once**, here, beside its siblings, so no consumer can misspell it.
/// It is the **injector** op only: the retry driver wrapping that re-read logs
/// under `RECOVERY_STREAM_PEER_PRESENCE_REREAD_OP` instead, so renaming this
/// seam cannot silently retarget an operator-facing log label.
///
/// Named for the `UNLINK` it precedes, not for the adjudication: it fires well
/// INSIDE that adjudication — after the high-water probe, the `TYPE`
/// classification and the SR-2.4 identity gate, and only on the
/// plausibility-refusal branch that reaches the removal at all.
pub const RESTORE_SEAM_BEFORE_STALE_STREAM_UNLINK: &str =
    "restore_before_stale_stream_unlink_peer_presence_reread";

/// Op label for the b.4et fault seam on the **stream high-water probe**: fires
/// at the top of EVERY probe attempt, inside the bounded retry, before the
/// `EVALSHA` goes on the wire.
///
/// Firing per attempt rather than once per restore is the point: a test drives
/// "fail N attempts, then let the (N+1)th through" to prove the retry budget is
/// spent as documented, and "fail every attempt" to prove exhaustion refuses the
/// restore rather than materializing a header under an unknown stream top.
///
/// An injected [`CheckpointError::Transient`] is treated as a retriable probe
/// failure (it maps onto the same `ServiceError::Aborted` in-band signal a
/// transient fred error produces); any other injected kind terminates the retry
/// immediately, exactly as a non-transient backend error would.
///
/// Spelled **once**, here, beside its sibling, so no consumer can misspell it.
pub const RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE: &str = "recovery_stream_high_water_probe";

/// Op label for the b.4et fault seam on the **stale-stream `UNLINK`**: fires at
/// the top of every `UNLINK` attempt on the plausibility-refusal branch, inside
/// the same bounded retry, before the command goes on the wire.
///
/// Reachable only after the probe read a high-water the plausibility gate
/// refused AND the SR-2.4 identity guard authorized the delete, so a test
/// exercising it must first stage a stale stream (an auto-ID tombstone top, or
/// any ID more than `MAX_PLAUSIBLE_HIGH_WATER_GAP` above the checkpoint's
/// `snap_seq`) on a layer whose mapping still resolves.
///
/// Same `Transient`-is-retriable convention as
/// [`RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE`]. Exhausting the budget here is the
/// `unlink_unverified` refusal: the restore does not run.
pub const RECOVERY_SEAM_STALE_STREAM_UNLINK: &str = "recovery_stale_stream_unlink";

/// Op label for the b.4et fault seam on the **stream-key `TYPE`
/// classification**: fires at the top of every `TYPE` attempt, before the
/// command goes on the wire.
///
/// The `TYPE` round-trip is reached from **two** places, and which one is live
/// is decided entirely by how the test stages the stream — so one op label
/// drives two distinct, mutually exclusive refusal reasons:
///
/// - **`probe_unreadable`.** Stage a stream whose high-water the probe cannot
///   read (or fail the probe outright at
///   [`RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE`]). The classification `TYPE` in
///   `classify_unreadable_stream` then runs; a terminal injected fault there
///   takes its `Err(type_err)` arm — "we could not even learn what the key is"
///   — which refuses with reason `probe_unreadable`.
/// - **`unlink_unverified`.** Stage an *implausible* stream top (an auto-ID
///   tombstone, or any ID more than `MAX_PLAUSIBLE_HIGH_WATER_GAP` above
///   `snap_seq`) on a layer whose mapping still resolves, **and** make the
///   `UNLINK` report deleting nothing. The probe then takes the
///   plausibility-refusal branch, and the only `TYPE` on that path is the
///   post-`UNLINK` confirmation inside `unlink_stale_stream_once` — which is
///   reached **only** when the `UNLINK` returned `0`, since a reply of `>= 1`
///   is already the verification and returns before any `TYPE`. So the
///   implausible top alone does not reach this seam; the staging that does is
///   to clear the stream key from *inside* the
///   [`RECOVERY_SEAM_STALE_STREAM_UNLINK`] injector, so the key is gone by the
///   time the `UNLINK` runs (the b.4et test suite's `VanishingStreamInjector`
///   in `tests/integration/checkpoint/e2e_object_id_preservation_test.rs`
///   does exactly this). With the confirmation `TYPE` reached, a terminal
///   injected fault there makes the removal unverifiable, which refuses with
///   reason `unlink_unverified`.
///
/// Same `Transient`-is-retriable convention as its two siblings: an injected
/// [`CheckpointError::Transient`] is retried within the enclosing budget
/// (`RECOVERY_STREAM_PROBE_RETRY` for the classification `TYPE`; the `UNLINK`'s
/// own budget for the confirmation `TYPE`, which re-runs the whole `UNLINK`
/// attempt), any other kind is terminal.
pub const RECOVERY_SEAM_STREAM_KEY_TYPE: &str = "recovery_stream_key_type";

/// What [`apply_restore`] is asked to materialize.
///
/// `pub(crate)` rather than private: both Epic-4 consumers — the recovery
/// hook's plumbing and its lineage swap — construct these variants from
/// `src/checkpoint/recovery.rs`.
// `large_enum_variant` measures a cost this type does not pay. A
// `RestorePayload` is constructed exactly once per recovery attempt and moved
// exactly once — into `apply_restore` — on a cold path that already spends
// several adapter round-trips and a Valkey MULTI/EXEC. None is ever stored in a
// collection, cloned, or passed in a loop, so the per-move cost of the size
// delta is unmeasurable against that. Boxing would buy nothing and would change
// the payload shape the Epic-4 breakdown adjudicated across three Subtasks.
#[allow(clippy::large_enum_variant)]
pub(crate) enum RestorePayload {
    /// Bytes-in: decode with [`decode_checkpoint_ompf`] **after** the guards
    /// (today's ordering, preserved exactly).
    Ompf(Bytes),
    /// Already composed by
    /// [`compose_from_artifacts`](crate::checkpoint::compose::compose_from_artifacts);
    /// no decode.
    ///
    /// The payload carries its own external-value set rather than leaving it to
    /// [`ExtPrestore`]. That is what makes the fail-closed guard in
    /// [`apply_restore`] *reachable*: the core can see both "this restore needs
    /// values pre-stored" and "no store was wired", and refuse. With the values
    /// hanging off `ExtPrestore` instead, `ext: None` would withhold the only
    /// carrier and the mismatch could not be detected at all.
    Composed {
        /// The composed lineage head state.
        state: ObjectState,
        /// Parts 1..N of every replayed record's OMPF, deduplicated by hash
        /// across the whole lineage — [`ComposedLineage::ext_values`] verbatim.
        ///
        /// [`ComposedLineage::ext_values`]: crate::checkpoint::compose::ComposedLineage::ext_values
        ext_values: Vec<ExtValuePart>,
    },
}

/// The external-value store handle the SR-3.4 pre-store writes through, plus
/// the cfg-gated fault seam that sits immediately after it.
///
/// # Where the pre-store runs, and why exactly there
///
/// After [`live_mapping_resolves_blob_identity`] and **strictly before** the
/// single `write_restored_object` MULTI/EXEC. All three legs are load-bearing:
///
/// - **Strictly before the write (SR-3.4).** [`apply_restore`]'s step-1
///   structural-header short-circuit makes post-write repair *impossible*: a
///   crash between the object write and the ext store would strand a
///   complete-looking object whose tail-carried values exist nowhere reachable,
///   and every retry would short-circuit on that header (a `write_restored_object` that
///   committed stamped all 12 canonical fields, so the header it left behind is
///   structurally complete by construction) and never repair it. With pre-write
///   order a crash
///   between the two leaves only orphaned content-addressed values — existing GC
///   / orphan-sweep territory — and the retry re-stores idempotently.
/// - **After the guards.** A restore the identity guard refuses (deleted or
///   overwritten layer) must not first write up to
///   `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES × LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN`
///   = 512 MiB of blobs into a `noeviction` Valkey for nothing.
/// - **One `write_restored_object`.** Nothing but the seam may sit between the pre-store
///   and the write; persistence stays exactly one MULTI/EXEC, which is the whole
///   of SR-3.3's atomicity argument.
///
/// # Reference-count posture — ACCEPTED and documented, and leak-only
///
/// [`ValkeyExternalValueStore::store_values`](crate::storage::external_values::ValkeyExternalValueStore)
/// is content-idempotent but ref-count-**incrementing** (one `EVALSHA` of
/// `ext_store_value.lua`: `SET data NX` + `HINCRBY meta ref_count 1` + the
/// `last_store_ms` stamp). Composed restore therefore leaves
/// `ref_count` standing above the live-object count in three shapes, and **all
/// three are accepted as-is**. All three leak; none loses data.
///
/// - **Over-increment on retry.** A restore that is retried (a transient
///   `write_restored_object` failure, a crash after the pre-store) re-stores the same
///   hashes and over-counts their refs — leaked-but-never-GC'd blobs. This is
///   *not new*: it is the pre-existing behaviour of any retried live upload.
/// - **Over-increment on losing the b.h4m header fence.** Same leak, but a
///   *routine* outcome rather than a crash-or-retry one: step 5's
///   `write_restored_object` reports [`RestoreWriteOutcome::PeerLandedHeader`]
///   — folded here to [`ApplyOutcome::AlreadyPresent`] — after this pre-store has
///   already incremented, so a replica that loses the race leaves increments no
///   `obj:ext` entry of the winner's generation matches. Symmetric release on
///   that exit is tracked as follow-up **b.oj9**.
/// - **Pre-stored blobs the final state does not reference.**
///   `ComposedLineage::ext_values` is deduplicated across the whole lineage and
///   keeps a hash that a later tail record superseded. `write_restored_object` rebuilds
///   `obj:ext` absolutely from the restored state and takes
///   `SurplusPolicy::Suppress`, so such a hash is never released by the
///   commit, never appears in `obj:ext`, and is never released at retirement.
///   Its `+1` stands until the reference-count reconciler drains it (b.yda).
///
/// **There is no early-collect direction.** An earlier revision of this comment
/// warned that lineage-wide dedup under-counts — that a hash referenced by N
/// tail commits restores at `ref_count = 1` where live would have reached N, so
/// an overwrite of the first referencing field could decrement to zero while
/// other restored fields still pointed at the blob. That reasoned from the
/// pre-b.qpf **per-commit** counting unit. `ref_count` now counts LIVE OBJECTS
/// (one increment per object per distinct hash), so live also reaches 1 for that
/// case and there is nothing to under-count; and the `le:{oid}:obj:ext` index
/// makes the collect structurally unreachable regardless, because a commit emits
/// a `rewrite` release only when the object's LAST row holding the hash gives it
/// up. See `docs-internal/sdd.md` §11.3 and
/// [`ComposedLineage::ext_values`](crate::checkpoint::compose::ComposedLineage).
///
/// Leak in all three shapes, bounded, reconciled, and no silent data loss in any.
///
/// # Ext-mirror interaction — ACCEPTED, and the store path deliberately unchanged
///
/// The pre-store does not bypass the SR-3.1 ext-blob mirror.
/// `ValkeyExternalValueStore::store_values` sends **every** blob it is
/// handed (whether or not its set-if-absent data write landed) to the
/// `ExtMirrorSender` whenever a checkpoint adapter is
/// wired — unconditionally, with no "restore, so skip it" arm — and a production
/// restore is by definition the case where an adapter *is* wired. A composed
/// restore of a large lineage therefore emits up to
/// `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES × LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN`
/// worth of mirror sends into a channel bounded both by
/// `LIVE_EDIT_CHECKPOINT_EXT_MIRROR_QUEUE_CAPACITY` items and by the mirror's
/// byte budget `M_ext`, **drop-on-full** on either, that live `CreateLayer` /
/// `UpdateLayer` writers are feeding at the same time. Restore work and
/// live-path work compete there, and either side's entries can be the ones
/// dropped. A value larger than the whole budget — possible here only for a
/// tail-carried value uploaded under an earlier, larger
/// `LIVE_EDIT_MAX_LAYER_SIZE` — is always dropped (`reason="oversized"`).
///
/// Accepted, on three grounds:
///
/// - **A restored blob is legitimately worth mirroring.** It is a real external
///   value that a real object now references; the mirror's job is blob-level DR
///   for exactly such values. Suppressing restore-path enqueues would leave a
///   restored object's blobs uncopied until the backup repair reached them —
///   minutes later, and only while Valkey still holds them.
/// - **The mirror is best-effort by design, and repaired.** Drop-on-full is
///   its contract (`checkpoint_ext_mirror_dropped_total{reason}` is the
///   signal). A dropped copy is never confirmed on the value's meta, so the
///   backup repair (`checkpoint::ext_backup_repair`) re-sends it from Valkey,
///   which the pre-store just wrote, within its age gate and one interval.
///   Until then a restored tail-carried value whose copy is dropped survives
///   outside Valkey only in the restored lineage's tail files, and a value
///   larger than the mirror's whole budget is never re-sent. Fetch-on-miss
///   cannot cover either — it reads *from* the backup copy that was never
///   written.
/// - **No loss of the restore itself.** The pre-store's own Valkey write is
///   the one the restore depends on and is unaffected by what the mirror does;
///   a dropped live-path send displaced by restore traffic degrades
///   identically to one displaced by other live traffic.
///
/// Consequently the pre-store's store path is **not** to be changed for this —
/// no restore-only bypass, no second sender, no non-mirroring `store_values`
/// variant for the pre-store. Its values need not be among the backup store's
/// external-value blobs: the upload's mirror copy may have been dropped
/// (above), or the backup store may have deleted the blob since, which is why
/// the tail carries the bytes. The pre-store's own mirror copy, and the backup
/// repair when that copy is lost, are what make sure one exists. The one
/// non-mirroring store is fetch-on-miss recovery's
/// rewrite (`storage::external_values`'
/// `ExtRecoveryRewrite::store_recovered_values`), whose bytes were just read
/// from those blobs, so a copy of them would be a redundant PUT. The
/// interaction is recorded here so a future reader finds it adjudicated rather
/// than undiscovered.
pub struct ExtPrestore<'a> {
    store: &'a dyn ExternalValueStore,
    #[cfg(feature = "test-checkpoint-fault-injection")]
    fault_injector: Option<&'a dyn CheckpointFaultInjector>,
}

impl<'a> ExtPrestore<'a> {
    /// Wrap the external-value store the composed restore pre-stores through.
    ///
    /// Declared explicitly (rather than left to a struct literal) because the
    /// type is re-exported from `crate::checkpoint` and its fields are private.
    pub fn new(store: &'a dyn ExternalValueStore) -> Self {
        Self {
            store,
            #[cfg(feature = "test-checkpoint-fault-injection")]
            fault_injector: None,
        }
    }

    /// Install the [`RESTORE_SEAM_AFTER_EXT_PRESTORE`] fault seam.
    ///
    /// Gated on `test-checkpoint-fault-injection` **exactly** — matching
    /// [`CheckpointFaultInjector`]'s own cfg, not `any(test, feature = ...)`
    /// — so the field, this builder, and the seam-check site all compile out of
    /// production builds entirely. Mirrors
    /// `LocalDiskCheckpointStore::with_fault_injector`.
    #[cfg(feature = "test-checkpoint-fault-injection")]
    pub fn with_fault_injector(mut self, injector: &'a dyn CheckpointFaultInjector) -> Self {
        self.fault_injector = Some(injector);
        self
    }
}

/// What one [`apply_restore`] attempt did.
// Same reasoning as `RestorePayload` above: one value per recovery attempt,
// moved once, on a cold path. `Applied(AppliedCheckpoint)` is additionally the
// shape Task 4's SR-3.6 re-base pattern-matches on as an adjudicated cross-Task
// contract, so boxing it would churn a consumer that does not exist yet in
// exchange for nothing measurable.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum ApplyOutcome {
    /// A structurally-complete `obj:hdr` was observed — a sibling replica (or
    /// the previous holder of this object's admission mutex) landed the
    /// recovery. Nothing was decoded, composed, pre-stored, wiped, or written.
    ///
    /// Two producers since b.f4t: the step-1 short-circuit, and the step-3b
    /// residue wipe observing a structurally-complete header under its `WATCH`
    /// — either on the post-`WATCH` re-read or on the re-read that follows a
    /// tripped `EXEC`. Both mean the same thing to every consumer: this attempt
    /// wrote nothing and the layer is present.
    ///
    /// **Positive evidence only.** The wipe does NOT report this arm merely
    /// because its `EXEC` tripped — any single-field `HSET` on `obj:hdr` trips
    /// it, and a racing `CreateLayer` overwrite's `superseded=1` is the concrete
    /// case. Doing so would tick
    /// `checkpoint_recovery_attempts_total{result="restored"}` for a restore
    /// that never ran; the wipe answers that with `Transient` and is re-driven.
    ///
    /// **Not a re-base trigger.** Task 4's SR-3.6 gap re-base must NOT run on
    /// this arm: nothing was composed here, and the peer that actually landed
    /// the restore owns its own re-base.
    AlreadyPresent,
    /// The restore ran to completion and the single `write_restored_object` MULTI/EXEC
    /// committed.
    Applied(AppliedCheckpoint),
}

/// The outputs of a completed restore, handed to Task 4's SR-3.6 re-base so it
/// can work from the already-in-memory state **without re-fetching** and
/// **without duplicating** the seq-advance formula (duplicating it is guaranteed
/// drift against b.8y5 / b.vnw).
///
/// This shape is THE cross-Task handoff, adjudicated at breakdown: Task 4 does
/// not define a parallel `RestoredState`. The `observed_commit_span` its re-base
/// target needs is *not* recoverable from `band` (`band = max(margin, span)` is
/// not invertible) — it comes from the meta the recovery hook already holds and
/// keeps in scope at the `restored` arm.
#[derive(Debug)]
pub struct AppliedCheckpoint {
    /// The state actually persisted. `state.sequence_num == seeded_seq`.
    ///
    /// Returned by value at no cost: `write_restored_object` takes `&ObjectState`, so the
    /// state is simply moved out after the write rather than cloned.
    pub state: ObjectState,
    /// The seeded writer cursor: `restore_floor + band`.
    pub seeded_seq: u64,
    /// The guard band applied — `max(config.checkpoint_seq_safety_margin,
    /// meta.observed_commit_span)` — i.e. the `restore_seq_bump` stamped on the
    /// [`CheckpointCursorStamp`].
    pub band: u64,
    /// The snapshot sequence this restore was seeded from: the composed head `S`
    /// on a composed lineage, the base's own `snap_seq` on a base-only one.
    pub snap_seq: u64,
}

/// Apply a checkpoint payload — raw OMPF bytes **or** an already-composed
/// lineage state — to live Valkey state for `object_id` (SR-2.4 / SR-2.5 /
/// SR-2.6, and SR-3.3 / SR-3.4 on the composed arm).
///
/// This is the load-bearing recovery primitive: it composes the
/// [`decode_checkpoint_ompf`] decoder, the [`live_mapping_resolves_blob_identity`]
/// guard, the SR-3.4 external-value pre-store, [`LayerStore::write_restored_object`], and
/// the SR-2.5 sequence-number
/// advance. The `last_checkpoint_seq` audit-trail stamp is folded into
/// `write_restored_object`'s MULTI/EXEC (b.4fh — eliminates the partial-header
/// window that a separate trailing HSET previously opened). Caller (the
/// recovery hook in the sibling Task `t2.fgj.ya.qm`) holds the per-
/// `object_id` admission lock around this call; **this function acquires
/// NO locks of its own**.
///
/// # Algorithm (ordered)
///
/// 1. **`HGETALL le:{oid}:obj:hdr`** on the data pool, through
///    [`header_is_structurally_complete`]. Present AND carrying `seq` +
///    `created_by` → `Ok(ApplyOutcome::AlreadyPresent)` immediately (a sibling
///    replica already landed the recovery for this object; nothing to do). A
///    header that exists but lacks either field reads as absent and the restore
///    PROCEEDS (b.f4t) — its readers already report it `NotFound`, so treating
///    it as present made recovery a no-op that handed the caller its own error
///    back. Step 3b removes the partial header before the write.
/// 2. **Identity guard.** Run [`live_mapping_resolves_blob_identity`]
///    against the data pool: a single `GET` of the mapping key named by the
///    blob's own `(session_id, layer_name)`, required to equal `object_id`.
///    `false` → [`CheckpointError::NotFound`] without writing — the layer was
///    deleted, the live mapping was overwritten by a newer `CreateLayer`
///    minting a fresh `object_id`, the `object_id` now belongs to a different
///    layer entirely (b.b2j reuse), or the blob carries no identity at all.
/// 3. **Take the state.** On [`RestorePayload::Ompf`],
///    [`decode_checkpoint_ompf`] parses the OMPF bytes into an `ObjectState`;
///    decoder errors propagate as [`CheckpointError::Permanent`] verbatim. On
///    [`RestorePayload::Composed`] the state is taken as-is — the lineage was
///    already replayed by the composition engine and re-decoding the base would
///    throw that replay away.
///
///    The decode stays **after** steps 1 and 2 on the `Ompf` arm, deliberately:
///    hoisting it would flip a corrupt blob for a deleted layer from `NotFound`
///    to `Permanent`, i.e. from the `not_recoverable` metric cell to `error`.
/// 4. **Seq advance (SR-2.5).** Compute
///    `restore_floor = max(snap_seq, max_seq_observed.unwrap_or(0))` —
///    the highest sequence number this restore can *prove* was issued —
///    then `seeded_seq = restore_floor + band` where `band =
///    max(config.checkpoint_seq_safety_margin, meta.observed_commit_span)`,
///    and stamp it on `state.sequence_num`. The band is a **guard band
///    above the restore floor**, not an alternative floor: commits that
///    landed after the checkpoint (inside the RPO window) are lost with
///    the crashed state, so their sequence numbers are unprovable and
///    resuming at exactly `restore_floor` would re-issue numbers that
///    pre-restore subscribers may already hold attached to *different*
///    (now-lost) content — the b.dy5-class contiguity/staleness hazard.
///    The margin is therefore additive at every input, including
///    `snap_seq > margin`, which the pre-b.8y5 `max(snap_seq, margin)`
///    form silently failed to do (the margin was inert for every layer
///    past its first thousand commits). **Reject** if `seeded_seq`
///    reaches the f64 safe-integer ceiling (`2^53 − 1`) — the Lua
///    fast-path and slow-path both reject sequence numbers at or above
///    this boundary, so seeding to the ceiling would make the next
///    UpdateLayer fail with `SEQ_OVERFLOW`. The defensive check exists
///    to surface this as `Permanent` at apply time rather than as a
///    Lua-error UpdateLayer. `max_seq_observed` is the caller-supplied
///    high-water hint: the production recovery hook derives it from an
///    `XINFO STREAM` on the surviving per-object stream
///    (`le:{oid}:obj:stream`), reading `last-generated-id` — the exact ID
///    the server compares an incoming `XADD` against. Since b.4et that
///    read is **not** best-effort, so `None` here carries a narrower
///    meaning than it used to. It is exactly one of: the stream key is
///    absent (positively confirmed by a `TYPE` round-trip); a
///    foreign-typed value was left in place; the identity guard declined
///    to touch a stream it may not delete; or — mirroring row 2 of
///    `probe_stream_high_water`'s arm table — the probe DID read the top,
///    the plausibility gate refused it (a tombstone's auto-ID, or any ID
///    more than `MAX_PLAUSIBLE_HIGH_WATER_GAP` above `snap_seq`), and the
///    adjudication then verifiably deleted that stream under the SR-2.4
///    identity gate. So a *present* stream can reach here as `None` — on
///    that last arm, deliberately, having been cleared. What cannot reach
///    here is a stream whose top could be neither read nor cleared: that
///    refuses the restore outright and `apply_restore` is never called, so
///    `None` never means "unreadable" any more. Every arm folds the floor
///    down to `snap_seq` alone.
///    `meta.observed_commit_span` (b.vnw) is the
///    blob's own measurement of the commits accumulated over the interval
///    it covers; taking the `max` keeps the configured margin a floor —
///    a cold layer measures fewer commits than the margin, and a
///    pre-b.vnw blob measures `0`, so both restore exactly as they did
///    before — while letting a layer that commits faster than the margin
///    carry a band sized from its observed traffic. That matters most
///    wherever `max_seq_observed` is `None` and the band is therefore the
///    only thing standing between the restore and sequence reuse: on a
///    full-keyspace loss, where the stream is gone with everything else,
///    and equally on the arm above where a known-implausible top was
///    deliberately cleared — there a stream demonstrably existed, and the
///    one thing we know about its top is that it is not a number we may
///    resume from.
///
/// 3b. **Residue wipe (b.f4t).** Restore is an authoritative REPLACE, so any
///    `obj:*` keyspace that survived under this `object_id` — indexes,
///    per-record hashes, children lists, tombstones, and a structurally
///    incomplete header — is removed before the materialization, under a
///    `WATCH` on `obj:hdr` that fences a peer replica's concurrent restore.
///    One `EXISTS` when nothing survived (the common whole-loss case) and
///    nothing else. `super::residue::wipe_restore_residue` owns the algorithm,
///    the refusal classification, and the exit paths; a
///    `PeerLandedHeader` outcome returns [`ApplyOutcome::AlreadyPresent`] from
///    here without writing. Placed after step 3 so an undecodable payload
///    refuses before anything is deleted. The window between that `WATCH` and
///    its `EXEC` carries the [`RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`] fault
///    seam, delivered through the same injector [`ExtPrestore`] carries.
///
/// 4b. **Ext pre-store (SR-3.4).** When the payload is
///    [`RestorePayload::Composed`] with a non-empty `ext_values`, write those
///    content-addressed pairs through `ext.store.store_values(..)`, then fire
///    the [`RESTORE_SEAM_AFTER_EXT_PRESTORE`] seam, then — and only then — the
///    single `write_restored_object`. Nothing else sits between the seam and the
///    write. The full order argument, and the ACCEPTED leak-only ref-count
///    posture, live on [`ExtPrestore`]. The write is set-if-absent: the residue
///    wipe does not cover `le:ext:data:{hash_hex}`, and a data key that
///    survived the incident is left as it stands rather than replaced — this
///    is an unverified `store_values`, so the verified store's type / length
///    heal does not apply (SDD §11.1 records the residual for a survivor stored
///    corrupt before uploads were hash-verified). The pre-store does not
///    re-hash: tail records carry only bytes of commits that landed.
///
///    **Fail closed.** A `Composed` payload with a non-empty `ext_values` and
///    `ext: None` is a wiring defect and returns
///    [`CheckpointError::Permanent`] with an ERROR log. Never a silent skip:
///    skipping is the exact silent-loss shape SR-3.4 exists to forbid.
/// 5. **Materialize.** Delegate to
///    [`LayerStore::write_restored_object`](crate::storage::layer::LayerStore::write_restored_object)
///    with the meta fields from [`CheckpointMeta`] (session_id, layer_name,
///    created_by, server_ts) and a
///    `CheckpointCursorStamp { snap_seq, written_at_ms, restore_seq_bump }`,
///    where `restore_seq_bump` is the `band` step 4 applied — the next
///    assembly subtracts it so the artificial restore jump is not measured
///    as commit traffic, and the post-restore re-base (normally first) or the
///    periodic loop's next successful write for this object clears the
///    column.
///    `write_restored_object` folds those columns into the same single HSET that stamps
///    the 12 canonical header fields, all inside one MULTI/EXEC on the txn
///    pool. `snap_seq` is the snapshot's authoritative version, NOT
///    `seeded_seq` (the safety-bumped writer cursor): operators reading
///    `HGET le:{oid}:obj:hdr last_checkpoint_seq` need to see which
///    checkpoint generation seeded the live state, not where the writer
///    cursor was advanced to. This is the load-bearing observability
///    invariant for the recovery audit trail. `written_at_ms` is the
///    restored blob's own [`CheckpointMeta::written_at_unix_ms`] (falling
///    back to `now_unix_ms()` when the adapter recorded none), so the
///    periodic loop's interval arm measures cadence from when the newest
///    durable copy actually landed (b.yut). `apply_checkpoint` does NOT need
///    to hold a transactional connection at this layer.
///
///    That materialization runs under a `WATCH` on `obj:hdr` requiring the
///    header to be ABSENT (b.h4m) — the innermost of this path's two
///    `WATCH`-held cross-replica fences (steps 3b and 5), and the only fence a
///    total-loss restore reaches, since such a restore skips step 3b for want
///    of residue to wipe. Step 1 is not a fence — it is a point-in-time
///    presence probe, and every restore runs it, total loss included. A
///    `RestoreWriteOutcome::PeerLandedHeader` returns
///    [`ApplyOutcome::AlreadyPresent`] from here without writing, exactly as
///    steps 1 and 3b do; all three stages tick
///    `live_edit_checkpoint_restore_fenced_total` with the `stage` that says
///    which one caught it.
///
/// # Pool selection
///
/// The step-1 header read and the identity guard use the data pool
/// (multiplexed, round-robin). The two data-pool round-trips at this layer share
/// no transactional semantics with what follows; the per-`object_id` admission
/// lock the caller holds is what serializes the read / guard / write sequence
/// against concurrent recovery attempts on the same object *on this replica*.
///
/// Steps 3b and 5 each take their own slot-pinned `txn_pool` connection, and
/// **sequentially, never overlapping**: the residue wipe releases its checkout
/// before [`LayerStore::write_restored_object`] asks for one. Both hold a
/// `WATCH` on `le:{oid}:obj:hdr` for the duration of their transaction, and
/// those two `WATCH`es are the *cross-replica* fences on this path — the
/// admission lock is per-process. The wipe's stops it from deleting a peer
/// replica's freshly landed restore; the materialization's stops it from being
/// **merged with** one (b.h4m), which is the shape that produced a layer made of
/// two backup generations. See `crate::checkpoint::recovery`'s module header for
/// all three guards in order.
///
/// # Error mapping
///
/// - Structurally-complete header (step 1) → `Ok(ApplyOutcome::AlreadyPresent)`.
/// - Identity guard false (step 2) → [`CheckpointError::NotFound`]. That
///   includes the empty-identity blob: `NotFound`, never `Permanent`, so
///   `RecoveryOutcome::NotRecoverable` and the recovery-attempt metric label
///   vocabulary are unchanged.
/// - Decode failure (step 3) → [`CheckpointError::Permanent`] verbatim.
/// - Residue wipe (step 3b): a wrong-typed key →
///   [`CheckpointError::Permanent`] (nothing wiped, nothing written — no retry
///   can read it); a peer landing a structurally-complete header under the
///   wipe's `WATCH` → `Ok(ApplyOutcome::AlreadyPresent)`; every other backend
///   failure, including an L2 pre-dispatch refusal and an `EXEC` tripped by a
///   write that did NOT land a complete header →
///   [`CheckpointError::Transient`], and the wipe is idempotent so the next
///   attempt converges.
/// - Missing ext store for a composed payload that needs one (step 4b) →
///   [`CheckpointError::Permanent`].
/// - Ext pre-store failure (step 4b) → [`CheckpointError::Transient`]
///   (via [`ext_prestore_err_to_checkpoint_err`]); an injected seam error
///   returns verbatim as the injector classified it, before any `write_restored_object`.
/// - Step 1 / 2 Valkey errors → [`CheckpointError::Transient`], except a
///   `WRONGTYPE` — on step 1's `HGETALL obj:hdr` or on step 2's
///   `GET le:{sid}:layer:{name}` — which is [`CheckpointError::Permanent`] via
///   [`classify_object_key_read_error`]. Both run ahead of the `write_restored_object`
///   keyspace probe that would otherwise have caught the header (and the
///   mapping key it never probes at all), and no retry changes a key's type.
/// - Materialization (step 5): a peer landing a structurally-complete header
///   under [`LayerStore::write_restored_object`]'s own `WATCH` →
///   `Ok(ApplyOutcome::AlreadyPresent)`, nothing written (b.h4m); a header
///   present but structurally INCOMPLETE, an L2 pre-dispatch refusal, a tripped
///   `EXEC` that did not leave a complete header, an `EXEC` error frame, and an
///   in-doubt `EXEC` are all [`CheckpointError::Transient`] and the next attempt
///   converges. A `WRONGTYPE` on that fence's `HGETALL obj:hdr` →
///   [`CheckpointError::Permanent`], like every other wrong-typed read here.
/// - Any other materialization failure (step 5) →
///   [`CheckpointError::Permanent`] for the two structural keyspace refusals no
///   retry can clear, `NotFound` verbatim, [`CheckpointError::Transient`] for
///   everything else (via [`write_object_err_to_checkpoint_err`], which
///   enumerates them).
///
/// # Visibility
///
/// `pub(crate)`: this is the in-crate core. Its bytes-in, `Result<(), _>`-out
/// compatibility wrapper [`apply_checkpoint`] is what stays `pub`, so the
/// ~25-call-site integration suite in
/// `tests/integration/storage/checkpoint_apply_test.rs` — the regression fence
/// for "existing base-only restore behaviour unchanged" — keeps compiling
/// against an unchanged call shape.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn apply_restore(
    layer_store: &LayerStore,
    object_id: u64,
    payload: RestorePayload,
    snap_seq: u64,
    meta: &CheckpointMeta,
    max_seq_observed: Option<u64>,
    config: &Config,
    ext: Option<ExtPrestore<'_>>,
) -> Result<ApplyOutcome, CheckpointError> {
    let oid_str = object_id.to_string();
    let client = layer_store.client();
    let keys = layer_store.keys();

    // Step 1: structural-presence short-circuit. If a sibling replica already
    // landed the recovery for this object, this round-trip is the cheapest way
    // to detect it — and skipping the rest of the steps avoids a wasted
    // MULTI/EXEC plus a wasted ext-blob GC schedule downstream (`write_restored_object`
    // overwrites the header unconditionally if we proceed).
    //
    // b.f4t: the predicate is `header_is_structurally_complete`, not a bare
    // `EXISTS`. A header that exists but lacks `seq` / `created_by` reads as
    // ABSENT here and the restore proceeds — see that helper for why (its
    // consumers already report such a header as `NotFound`, so answering
    // "already present" made recovery a no-op that returned the caller's own
    // error straight back). The wipe in step 3b removes the partial header so
    // `write_restored_object`'s HSET lands on a clean key.
    if header_is_structurally_complete(client, keys, object_id).await? {
        tracing::debug!(
            object_id,
            "apply_checkpoint: obj:hdr already present and structurally complete (sibling replica \
             landed recovery); skip"
        );
        // b.h4m: the outermost of this path's three cross-replica *guards*, and
        // the only one that is not `WATCH`-held — a point-in-time probe, and the
        // cheapest: one `HGETALL` before any work. (The two `WATCH`-held fences
        // are step 3b's residue wipe and step 5's materialization.) Counted so
        // the three stages together say how tight the race was; see
        // [`CHECKPOINT_RESTORE_FENCED_STAGES`].
        if let Some(metrics) = client.metrics_snapshot() {
            metrics.inc_checkpoint_restore_fenced(
                CHECKPOINT_RESTORE_FENCED_STAGE_PRESENCE_PROBE,
                &config.service_label,
            );
        }
        return Ok(ApplyOutcome::AlreadyPresent);
    }

    // Step 2: identity guard. SR-2.4 + b.b2j. The blob's OWN
    // `(session_id, layer_name)` must still resolve to this `object_id` —
    // otherwise the layer was deleted (DeleteLayer-checkpoint-delete race),
    // overwritten by a newer CreateLayer minting a fresh object_id
    // (CreateLayer-overwrite during state-loss race), or the id now belongs
    // to an entirely different layer (object-id reuse after a keyspace loss).
    // In all three cases restoring the OMPF would write dead — or worse,
    // someone else's — state.
    let mapping_live = live_mapping_resolves_blob_identity(client, keys, object_id, meta).await?;
    if !mapping_live {
        tracing::warn!(
            object_id,
            session_id = %meta.session_id,
            layer_name = %meta.layer_name,
            "apply_checkpoint: the checkpoint blob's layer identity does not resolve to this \
             object; skipping recovery (layer deleted, overwritten, or the object_id was reused \
             by a different layer)"
        );
        return Err(CheckpointError::NotFound);
    }

    // Step 3: take the `ObjectState` the payload names.
    //
    // `Ompf` — decode the bytes. The decoder is a pure sync primitive; any
    // failure here is a wire-protocol or content-level violation that retrying
    // cannot fix. The decode stays HERE, after steps 1 and 2, for the reason
    // stated in the algorithm doc: hoisting it flips a corrupt blob for a
    // deleted layer from `NotFound` to `Permanent`.
    //
    // `Composed` — the lineage was already replayed by the composition engine;
    // re-decoding the base would throw that replay away.
    let (mut state, ext_values) = match payload {
        RestorePayload::Ompf(ompf) => (decode_checkpoint_ompf(object_id, &ompf)?, Vec::new()),
        RestorePayload::Composed {
            mut state,
            ext_values,
        } => {
            // Zero `first_version` for the same reason `decode_checkpoint_ompf`
            // does (`apply.rs`'s "Per the function-level doc" block): a lineage
            // containing a replayed reset arrives with a `first_version` that
            // `apply_reset` set to that reset's `new_version`
            // (`src/domain/delta.rs`) — a number from the PRE-CRASH timeline
            // with no relation to the `seeded_seq` computed below. Today's
            // base-only restore writes `0` here and lets the first
            // post-recovery write populate the column; the composed path must
            // do the same, or a restored layer would advertise a
            // `first_version` above its own `sequence_num`.
            state.first_version = 0;
            (state, ext_values)
        }
    };

    // Step 4: SR-2.5 seq advance. `restore_floor` is the highest sequence
    // number this restore can PROVE was issued: the checkpoint's own
    // `snap_seq`, raised to `max_seq_observed` when the caller could read a
    // higher water mark out of surviving state (the recovery hook probes
    // the per-object stream's last entry ID). A guard `band` is then added ON
    // TOP, covering the commits that are provably NOT representable here —
    // everything that landed between the checkpoint and the crash, which died
    // with the live state and whose sequence numbers pre-restore subscribers
    // may still hold attached to different content. Resuming at exactly
    // `restore_floor` would re-issue those numbers against new content
    // (b.dy5-class staleness). The band is `max(configured margin, measured
    // span)` — see the b.vnw paragraph below; `safety_margin` is its floor, not
    // the whole of it.
    //
    // b.8y5: this term is additive at EVERY input. The prior
    // `max(snap_seq, max_seq_observed + margin)` form anchored the additive
    // term at zero, so for any layer with `snap_seq >= margin` the margin
    // was inert and the restore resumed at exactly `snap_seq` — the guard
    // band the knob advertises did not exist past a layer's first
    // `margin` commits.
    //
    // b.vnw: the band is `max(configured margin, observed_commit_span)`. The
    // configured margin remains a floor — a cold layer measures fewer commits
    // than the margin and a pre-b.vnw blob measures nothing at all (`0`), so
    // both restore exactly as they did before — but a layer that demonstrably
    // commits faster than the margin now carries its own measured band. That closes
    // the residual reuse window on a full-keyspace loss, where the stream is
    // gone, the probe contributes nothing, and the band is the whole defense:
    // pre-b.vnw its adequacy rested entirely on the operator having sized
    // `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN >= peak_commits_per_sec ×
    // LIVE_EDIT_CHECKPOINT_INTERVAL_SECS` against a peak they had to predict.
    // `observed_commit_span` is that same quantity measured rather than
    // predicted: commits accumulated between the two newest generations, i.e.
    // over one checkpoint interval of this layer's real traffic.
    let restore_floor = snap_seq.max(max_seq_observed.unwrap_or(0));
    let band = config
        .checkpoint_seq_safety_margin
        .max(meta.observed_commit_span);
    // Outsized-band signal. The band is applied as measured — see
    // `OUTSIZED_BAND_MARGIN_MULTIPLE` for why it is not clamped — but a span
    // this far above the configured margin is worth an operator's attention,
    // because the guard band is not free downstream: every sequence number in
    // it is skipped, and each skipped sequence is one every reconnecting
    // subscriber synthesizes through client-side gap-fill before it can serve
    // live traffic again. A band of a few thousand is unremarkable; a band of
    // millions makes post-restore reconnects observably slow.
    let outsized_threshold = config
        .checkpoint_seq_safety_margin
        .saturating_mul(OUTSIZED_BAND_MARGIN_MULTIPLE);
    if meta.observed_commit_span > outsized_threshold {
        tracing::warn!(
            object_id,
            snap_seq,
            observed_commit_span = meta.observed_commit_span,
            safety_margin = config.checkpoint_seq_safety_margin,
            margin_multiple = OUTSIZED_BAND_MARGIN_MULTIPLE,
            band,
            "apply_checkpoint: measured commit span exceeds \
             LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN by more than the expected \
             multiple; applying the measured band unclamped (clamping would \
             re-open the sequence-reuse window). Either the margin is sized \
             below this layer's real per-interval commit rate, or this \
             generation spans a checkpoint outage. Note every sequence in the \
             band is skipped and gap-filled client-side by each reconnecting \
             subscriber"
        );
    }
    let seeded_seq = restore_floor.saturating_add(band);
    if seeded_seq >= F64_SAFE_INT_MAX {
        // The Lua fast path and slow path both reject writes once
        // `seq >= 2^53 - 1`, so seeding at or above the ceiling would
        // make every subsequent `UpdateLayer` against the restored
        // object fail with a sequence-overflow error. Surface a
        // `Permanent` here rather than materializing unwritable state —
        // a checkpoint whose snap_seq + safety_margin reaches 2^53 - 1
        // indicates either a runaway writer or a corrupted checkpoint,
        // and retrying the apply against the same checkpoint bytes will
        // hit the same ceiling.
        tracing::error!(
            object_id,
            snap_seq,
            // `?` not `.unwrap_or(0)`: renders `None` / `Some(N)`, so a
            // declined probe is distinguishable from a genuine zero.
            ?max_seq_observed,
            restore_floor,
            safety_margin = config.checkpoint_seq_safety_margin,
            observed_commit_span = meta.observed_commit_span,
            band,
            seeded_seq,
            f64_safe_int_max = F64_SAFE_INT_MAX,
            "apply_checkpoint: seeded_seq exceeds f64 safe-integer ceiling; \
             refusing to materialize unwritable state"
        );
        return Err(CheckpointError::Permanent {
            source: format!(
                "apply_checkpoint seq advance reaches f64 safe-integer ceiling: \
                 snap_seq={snap_seq}, restore_floor={restore_floor}, safety_margin={}, \
                 observed_commit_span={}, band={band}, seeded_seq={seeded_seq} >= 2^53-1",
                config.checkpoint_seq_safety_margin, meta.observed_commit_span
            )
            .into(),
        });
    }
    state.sequence_num = seeded_seq;
    // `first_version` was zeroed by the decoder (and by the `Composed` arm of
    // step 3); leave it at 0. The
    // assembler intentionally drops `first_version` from the OMPF payload
    // because it is a per-update bookkeeping field that has no meaning on
    // a restored snapshot — the first post-recovery write will populate
    // it as part of its `SetObjectHeader` mutation.

    // Step 4a: flatten the per-entry versions a diff builder FILTERS on —
    // `Section::last_version`, `Field::version`, `TimeSample::version` — onto
    // `seeded_seq` (b.2ti). `Section::first_version` is deliberately exempt;
    // see the last paragraph.
    //
    // A restore is ONE write of the WHOLE object at `seeded_seq`, so every
    // entry it materializes must be visible as of that sequence. The restore
    // preserves `object_id` (the SR-2.4 identity guard above), so a subscriber
    // that survives it does NOT re-bootstrap — it reconnects and takes the
    // incremental path, where `src/domain/diff.rs` (`plan_incremental` /
    // `plan_faithful_coalesced`) emits only entries with `version >
    // have_version` (the latter also restates the other children of a parent
    // with such a child, and nothing else). Carrying the pre-restore stamps
    // would therefore withhold the restored content from exactly the
    // subscribers that need correcting:
    // the base decode leaves everything at `version = 1`, and a composed
    // lineage leaves base entries below its tail entries.
    //
    // What the guard band does and does not buy. `restore_floor` (step 4)
    // provably dominates every sequence this server can PROVE the object
    // issued — the checkpoint's `snap_seq`, and the surviving stream top when
    // the probe returned one. The commits lost in the RPO window are by
    // definition not provable; the band above the floor
    // (`max(configured margin, measured commit span)`) is what COVERS them. So
    // `seeded_seq` dominates a reconnector's `have_version` only insofar as
    // that band is sized for the layer's real per-interval commit rate. If the
    // band is undersized a subscriber can hold `have_version >= seeded_seq`,
    // and for that subscriber this stamp does not help — its diff still
    // filters the restored entries out. Closing that residual is the band's
    // job, not this step's (SDD §22.6).
    //
    // Determinism, not order preservation. A constant `version` keeps
    // `sort_fields_by_authoring`'s key `(version, authoring_set_order,
    // section_id, key_name)` a TOTAL order, so the emit sequence stays
    // deterministic — but it is a DIFFERENT sequence. With `version` constant
    // the primary key collapses to `authoring_set_order`, which resets to 0
    // per delta, so on the `Composed` arm base-era and tail-era fields
    // interleave where composition had kept the eras separated. That
    // reordering propagates into `dense_reindex_by_authoring`'s emitted
    // indices and into the bytes of the re-base artifact named below.
    //
    // The time-sample comparator loses its primary key the same way, and in
    // all three builders that use it: `plan_full_content`,
    // `plan_faithful_snapshot` and `plan_faithful_coalesced` all sort on
    // `(version, section_id, time)`, so with `version` constant the key
    // collapses to `(section_id, time)` and samples authored in different eras
    // interleave. For the re-base artifact that is not cosmetic — per
    // `plan_full_content`'s rustdoc, `collect_external_value_hashes` scans
    // `fields` THEN `time_samples` into ONE first-appearance pool, and
    // `build_reset_ompf_from_state` forces that arm, so the artifact's ext-refs
    // sidecar hash order (`ExternalValueScan::ordered_hashes`) moves with the
    // time-sample order too. Still total, still deterministic — a different
    // total order.
    //
    // Scope. `compose_lineage`'s own per-record stamping is left alone — it is
    // the in-order-replay witness. But this flattening does not stop at what
    // Valkey persists: `AppliedCheckpoint::state` is moved into
    // `BaseBootstrap::spawn_restore_rebase` (`src/checkpoint/base_bootstrap.rs`;
    // the recovery path calls it from `src/checkpoint/recovery.rs`) and
    // rebuilt by `build_reset_ompf_from_state` (`src/checkpoint/assembler.rs`),
    // so the SR-3.6 restore-minted re-base artifact is assembled from these
    // flattened versions too.
    //
    // Why `Section::first_version` is exempt. `plan_faithful_coalesced`'s
    // abandoned-prim retraction arm fires on `first_version <= have_version &&
    // !live_prim_ids.contains(id)`. Raising `first_version` to `seeded_seq`
    // would put it above every reconnector's `have_version`, so the orphan
    // sections a restored base carries (`plan_full_content` emits unreachable
    // sections into the base OMPF) could never be retracted. Left at the value
    // the decoder produced (`1`) or the composition engine replayed (the real
    // tail sequence, for a tail-created section), that arm keeps working, and
    // new-vs-modified classification stays truthful for tail-created sections
    // on the `Composed` arm — base-era sections read as pre-existing by
    // construction, since the decoder stamps `first_version = 1` (on the
    // `Ompf` arm that is EVERY section, so the retraction arm above matches
    // every base-carried orphan for every reconnector, including orphans
    // created after its `have_version` that it never held). The collapsed
    // classification is harmless because `build_delta`
    // (`src/storage/diff_read.rs`) emits the same `DiffChangeSection` upsert
    // for both classes, so the wire shape is unchanged either way — the
    // load-bearing reason to preserve `first_version` is the retraction arm,
    // not the classification. `first_version <= last_version` still holds
    // everywhere, and not by luck: `snap_seq` arrives here as the recovery
    // path's `head_seq` (`src/checkpoint/recovery.rs`), which is cross-checked
    // equal to `composed.composed_seq` — the terminal replay cursor — and so
    // bounds every version composition stamped, while a base-only decode
    // stamps `1`; with `restore_floor >= snap_seq` (step 4) and `band >= 1`
    // (`checkpoint_seq_safety_margin` is zero-rejected in `Config::from_env`,
    // and the band is a `max` over it), `seeded_seq = restore_floor + band` is
    // strictly above the composed head and at least `1`, hence at or above
    // every preserved `first_version`. SDD §22.6 (guard band), §12.10
    // (catch-up / reconnect).
    for section in state.sections.values_mut() {
        section.last_version = seeded_seq;
    }
    for field in state.fields.values_mut() {
        field.version = seeded_seq;
    }
    for sample in state.time_samples.values_mut() {
        sample.version = seeded_seq;
    }
    // The exemption's ordering derivation above rests on step 4's floor/band
    // arithmetic and on `snap_seq == composed_seq` upstream; pin it here so a
    // change to either trips in-crate instead of shipping sections whose
    // `first_version` outruns their `last_version`. `debug_assert!` rather than
    // `assert!` because a restore runs on a request path — a release build must
    // not abort a recovery it can still serve over an ordering invariant.
    debug_assert!(
        state
            .sections
            .values()
            .all(|section| section.first_version <= section.last_version),
        "restore must leave first_version <= last_version: seeded_seq is \
         restore_floor + band, with restore_floor >= snap_seq (the composed \
         head, which bounds every replayed version) and band >= 1, so it \
         dominates every preserved first_version"
    );

    // Step 3b: residue wipe (b.f4t). Restore is an authoritative REPLACE, and
    // `write_restored_object` — which queues the same body the CreateLayer
    // materializer does — cannot deliver that on
    // its own: it unions the restored members into the surviving indexes,
    // writes no tombstones at all, and replaces a children list only under a
    // parent the restored state carries. So the surviving `obj:*` families are
    // removed here, first. See `super::residue` for the whole argument,
    // including why the per-object stream and the per-object external-value
    // reference index `obj:ext` are deliberately NOT part of the inventory —
    // the latter because `write_restored_object` reads it as its `ext_before` pre-image
    // and is what emits the `before − after` releases (b.qpf).
    //
    // Placed HERE — after the decode and the seq arithmetic, before the ext
    // pre-store — on purpose. The decode is the last step that can refuse the
    // restore for a reason intrinsic to the payload, so running it first means
    // an undecodable blob (or a `seeded_seq` over the f64 ceiling) refuses
    // without having deleted anything. Everything from this point on either
    // materializes or fails on the backend.
    //
    // The wipe takes a `ResidueWipeCtx` rather than the bare `&Config` so that
    // the [`RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC`] injector — the only way to
    // reach INSIDE its `WATCH` … `EXEC` window — can ride along without the
    // wipe's parameter list changing shape between feature configurations.
    // `From<&Config>` is what keeps every seam-less caller passing exactly what
    // it passed before.
    let wipe_ctx = crate::checkpoint::residue::ResidueWipeCtx::from(config);
    // The injector is read off the `ExtPrestore`, which is where the recovery
    // hook's `with_restore_fault_injector_for_test` already installs it: every
    // arm of the AC-8 matrix restores through a hook that wires
    // `with_ext_value_store`, so `ext` is `Some` there and no second delivery
    // route is needed. A restore with no ext store wired simply has no seam —
    // which is the production shape, where the whole field is compiled out.
    #[cfg(feature = "test-checkpoint-fault-injection")]
    let wipe_ctx = wipe_ctx.with_fault_injector(ext.as_ref().and_then(|e| e.fault_injector));
    // `Box::pin`, not a bare `.await`. An `async fn`'s state machine is inlined
    // into its caller's, and the wipe is a big one — a `WatchGuard`, a
    // `WatchedTxn`, five concurrent enumerations under `try_join!`, and the
    // chunked UNLINK inventory. Left inline it lands in `apply_restore`'s
    // future, which lands in `RecoveryHook::try_restore_layer`'s, which lands in
    // the `SubscribeReadLayer` catch-up chain — the deepest caller of this path,
    // and one that runs on a 2 MiB tokio worker thread. In a debug build,
    // which gives each inlined future its own stack slot, that stack overflowed
    // outright without the box. Boxing moves the state to the heap and leaves a
    // pointer behind. The debug-build margin is the one the note at
    // `try_restore_layer`'s `Box::pin` describes. The product check is the
    // release build: the release stack guard's subscribe layer-recovery case
    // (`tests/release_stack_guard/`) measures this chain's depth on a 2 MiB
    // worker-shaped thread. (The matching test-side note is on
    // `setup_wipe_window_arm` in
    // `tests/integration/checkpoint/restore_fault_injection_matrix_test.rs`.)
    match Box::pin(crate::checkpoint::residue::wipe_restore_residue(
        layer_store,
        object_id,
        wipe_ctx,
    ))
    .await?
    {
        crate::checkpoint::residue::ResidueOutcome::Clean
        | crate::checkpoint::residue::ResidueOutcome::Wiped { .. } => {}
        // A peer replica landed a structurally-complete header while the wipe
        // held its `WATCH`, so the wipe deleted nothing. Reporting
        // `AlreadyPresent` is what stops this attempt from writing over the
        // peer's restore — the same verdict step 1 would have reached had the
        // peer been a round-trip earlier.
        crate::checkpoint::residue::ResidueOutcome::PeerLandedHeader => {
            // b.h4m: the middle fence. Counted on the same counter as its two
            // siblings, with `stage` saying which one caught it.
            if let Some(metrics) = client.metrics_snapshot() {
                metrics.inc_checkpoint_restore_fenced(
                    CHECKPOINT_RESTORE_FENCED_STAGE_RESIDUE_WIPE,
                    &config.service_label,
                );
            }
            return Ok(ApplyOutcome::AlreadyPresent);
        }
    }

    // Step 4b: SR-3.4 external-value pre-store. Runs AFTER the identity guard
    // and STRICTLY BEFORE the single `write_restored_object` MULTI/EXEC — see
    // `ExtPrestore`'s rustdoc for all three legs of that ordering argument and
    // for the accepted leak-only ref-count posture.
    if !ext_values.is_empty() {
        let Some(ext) = ext.as_ref() else {
            // Fail closed. A composed payload that references external values
            // with no store wired is a wiring defect, and the ONLY safe verdict
            // is to refuse: skipping the pre-store would materialize a
            // complete-looking object whose values exist nowhere reachable,
            // which is precisely the silent-loss shape SR-3.4 exists to forbid.
            // `Permanent`, not `Transient` — no retry re-wires a hook.
            tracing::error!(
                object_id,
                ext_values = ext_values.len(),
                session_id = %meta.session_id,
                layer_name = %meta.layer_name,
                "apply_restore: composed payload carries external values but no \
                 ExternalValueStore was wired into the restore path; refusing to write an \
                 object whose tail-carried values would exist nowhere reachable. This is a \
                 wiring defect — RecoveryHook::with_ext_value_store is not set"
            );
            return Err(CheckpointError::Permanent {
                source: format!(
                    "composed restore for object {object_id} carries {} external value(s) but \
                     no ExternalValueStore was wired (the external-value pre-store cannot run)",
                    ext_values.len()
                )
                .into(),
            });
        };

        let pairs: Vec<(&[u8; 32], &[u8])> = ext_values
            .iter()
            .map(|(hash, data)| (hash, data.as_ref()))
            .collect();
        ext.store
            .store_values(&pairs)
            .await
            .map_err(ext_prestore_err_to_checkpoint_err)?;
        tracing::debug!(
            object_id,
            ext_values = pairs.len(),
            "apply_restore: tail-carried external values pre-stored ahead of the single \
             write_restored_object"
        );

        // The seam. Nothing else may sit between here and `write_restored_object`.
        #[cfg(feature = "test-checkpoint-fault-injection")]
        if let Some(injector) = ext.fault_injector {
            injector.maybe_inject(RESTORE_SEAM_AFTER_EXT_PRESTORE)?;
        }
    }

    // Release the tail-carried value bytes the instant the pre-store is done
    // with them, BEFORE the write. This is a move, so the compiler — not a
    // comment — is what stops them living across `write_restored_object`.
    //
    // It matters because these `Bytes` are zero-copy views over the tail
    // *artifact* buffers, so each pinned pair keeps a whole envelope alive by
    // refcount. Under gather's split cache the resident-artifact ceiling is one
    // lineage (`1 base + K tails`; 512 MiB at stock knobs), and holding it
    // across the MULTI/EXEC would double a large restore's peak footprint on a
    // `noeviction` deployment for no benefit — the values are already durable in
    // Valkey by this point.
    drop(ext_values);

    // Step 5: materialize. `write_restored_object` runs its own WATCHed
    // MULTI/EXEC and owns the txn pool for its window. It queues no stream
    // entry at all: the recovery path deliberately does NOT XADD a `1-0` entry
    // on the restored stream — the stream key uses the same `{object_id}` hash
    // tag, so the per-object stream is independently recoverable (it
    // either survived the loss, in which case appending `1-0` would
    // collide, or it didn't, in which case the next live write will
    // initialize it). The SR-2.4 recovery contract is "make the static
    // layer state visible", not "synthesize a synthetic update event".
    //
    // The [`CheckpointCursorStamp`] (b.4fh): folded into the same MULTI/EXEC
    // that lands the 12 canonical header fields. Not an `Option` on this entry
    // point — a restore ALWAYS stamps its cursor, and the type says so.
    // Previously this was a separate trailing HSET, which opened a window
    // where a crash or concurrent header wipe between EXEC and the trailing
    // HSET could leave a structurally-invalid 1-field skeletal header (only
    // `last_checkpoint_seq` set). The value is `snap_seq` (the snapshot's
    // authoritative version), NOT `seeded_seq` (the safety-bumped writer
    // cursor) — operators reading `HGET le:{oid}:obj:hdr last_checkpoint_seq`
    // need to see which checkpoint generation seeded the live state, not
    // where the writer cursor was advanced to.
    //
    // `written_at_ms` (b.yut) — stamp `last_checkpoint_ms` with the restored
    // blob's OWN recorded write time (`CheckpointMeta::written_at_unix_ms`),
    // not `None` and not an unconditional `now`. That column is the periodic
    // loop's cadence reference — "when did the newest durable copy of this
    // layer land" — and post-restore the newest durable copy IS this blob,
    // written when it says it was. Consequences, both intended:
    //   - blob older than `LIVE_EDIT_CHECKPOINT_INTERVAL_SECS` → the interval
    //     arm fires on the first eligible tick, which is CORRECT: step 4 above
    //     seeds `seq = restore_floor + band`, so the live header
    //     already diverges from `last_checkpoint_seq` and the seeded bump is
    //     itself state worth persisting.
    //   - blob fresh → the restored layer observes normal cadence instead of
    //     an immediate redundant re-checkpoint of bytes we just read.
    // `written_at_unix_ms == 0` means the adapter did not record a write time.
    // Since b.hka the meta-sidecar-missing read paths are refused at the
    // adapter rather than serving `CheckpointMeta::default()`, so no shipped
    // producer reaches this; it stays as the defensive fold, falling back to
    // `now_unix_ms()` rather than writing a literal 0, which the gate would
    // read as "never checkpointed".
    let restored_checkpoint_ms = if meta.written_at_unix_ms == 0 {
        crate::checkpoint::now_unix_ms()
    } else {
        meta.written_at_unix_ms
    };
    //
    // b.h4m: `write_restored_object`, not `LayerStore::write_object`. The two split when the
    // restore path gained its own `WATCH` on `obj:hdr` requiring the header to
    // be ABSENT — the innermost of the two `WATCH`-held fences (steps 3b and
    // 5), and the only fence reachable on a total-loss restore, which skips
    // step 3b for want of residue to wipe. Before it, two replicas
    // restoring the same layer concurrently could have one rebuild ADDED to the
    // other's (a MIXTURE of two backup generations, silent and uncounted); now
    // Valkey serializes the two `EXEC`s on the `{object_id}` shard and the loser
    // writes nothing. See that function for the whole argument.
    //
    // `Box::pin`, not a bare `.await`, and for the same reason the step-3b wipe
    // is boxed (b.9qr): an `async fn`'s state machine is inlined into its
    // caller's, and this one now carries a `WatchGuard`, a `WatchedTxn` and the
    // whole queueing body. Left inline it lands in `apply_restore`'s future,
    // which lands in `RecoveryHook::try_restore_layer`'s, which lands in the
    // `SubscribeReadLayer` catch-up chain — the deepest caller of this path, on
    // a 2 MiB tokio worker thread.
    let write_outcome = Box::pin(layer_store.write_restored_object(
        &oid_str,
        &meta.session_id,
        &meta.layer_name,
        &meta.created_by,
        &state,
        meta.server_ts,
        CheckpointCursorStamp {
            snap_seq,
            written_at_ms: restored_checkpoint_ms,
            // `restore_seq_bump` (b.vnw) — the band this restore applied,
            // recorded so the next assembly can subtract it and measure
            // only real commits. `band`, not `seeded_seq - snap_seq`: the
            // stream-probe term (`restore_floor - snap_seq`) IS real
            // commit traffic the probe proved had landed past the
            // checkpoint, so it belongs in the next span rather than being
            // corrected away. The ceiling check above rules out the
            // saturating arm, so `seeded_seq == restore_floor + band`
            // exactly on this path.
            restore_seq_bump: band,
        },
    ))
    .await
    .map_err(write_object_err_to_checkpoint_err)?;

    // The fence answered "a peer replica already landed this object's rebuild",
    // so nothing was written (b.h4m). Same verdict step 1 and step 3b reach one
    // and two round trips earlier respectively — this one is the innermost, and
    // the only *fence* a total-loss restore can reach: step 3b's `WATCH` rides
    // the residue wipe, which a total loss skips for want of fragments to
    // clear. Step 1 is not ruled out — it is a point-in-time presence probe,
    // not a fence, and every restore runs it, total loss included.
    if write_outcome == RestoreWriteOutcome::PeerLandedHeader {
        if let Some(metrics) = client.metrics_snapshot() {
            metrics.inc_checkpoint_restore_fenced(
                CHECKPOINT_RESTORE_FENCED_STAGE_OBJECT_WRITE,
                &config.service_label,
            );
        }
        return Ok(ApplyOutcome::AlreadyPresent);
    }

    // `max_seq_observed` / `restore_floor` alongside `snap_seq` and
    // `seeded_seq`: without them an operator reading this line cannot tell
    // whether a seeded_seq above snap_seq came from the stream high-water
    // probe or from the safety margin alone, which is the first question
    // asked when a restored layer's sequence looks surprising. The
    // ceiling-rejection log above already carries both. `?` rather than
    // `.unwrap_or(0)` so a declined probe renders `None` instead of
    // masquerading as an affirmative high-water of zero.
    tracing::info!(
        object_id,
        snap_seq,
        ?max_seq_observed,
        restore_floor,
        // Which of the two band inputs won (b.vnw): a `band` above
        // `safety_margin` means the blob's measured commit span sized the
        // guard, which is the answer to "why did this restore jump further
        // than the configured margin".
        safety_margin = config.checkpoint_seq_safety_margin,
        observed_commit_span = meta.observed_commit_span,
        band,
        seeded_seq,
        session_id = %meta.session_id,
        layer_name = %meta.layer_name,
        "apply_checkpoint: object materialized from checkpoint"
    );

    // `state` is moved out here at no cost: `write_restored_object` took it by
    // reference, so nothing was cloned to produce this handoff.
    Ok(ApplyOutcome::Applied(AppliedCheckpoint {
        state,
        seeded_seq,
        band,
        snap_seq,
    }))
}

/// Bytes-in, `Result<(), _>`-out compatibility wrapper over [`apply_restore`]
/// (SR-2.4 / SR-2.5 / SR-2.6).
///
/// Behaviourally identical to the pre-Epic-4 function of the same name: it feeds
/// [`RestorePayload::Ompf`] and no [`ExtPrestore`] into the shared core and
/// discards the [`ApplyOutcome`]. Both `Ok` arms —
/// [`ApplyOutcome::AlreadyPresent`] and a completed apply — fold to `Ok(())`,
/// exactly as before.
///
/// # Visibility
///
/// Surfaced as `pub` (rather than `pub(crate)`) so the integration-test
/// binary in `tests/integration/storage/checkpoint_apply_test.rs` can call
/// it directly against a live `TestValkey` fixture. The Epic-1 production
/// caller (the recovery hook in sibling Task `t2.fgj.ya.qm`) is in-crate
/// and could just as well consume the `pub(crate)` form; the broader
/// visibility is purely a test-affordance and carries no runtime cost.
///
/// **The wrapper exists to keep that suite's call shape stable.** Epic 4 moved
/// the body into [`apply_restore`], which takes a [`RestorePayload`] and returns
/// an [`ApplyOutcome`]; ~25 call sites in `checkpoint_apply_test.rs` (plus this
/// module's own in-file tests) pass the exact
/// `(&ls, oid, ompf, snap_seq, &meta, max_seq_observed, &config)` tuple, and
/// that suite IS the regression fence for "existing base-only restore behaviour
/// unchanged". Churning it would delete the fence in the same commit that needs
/// it most.
#[allow(clippy::too_many_arguments)]
pub async fn apply_checkpoint(
    layer_store: &LayerStore,
    object_id: u64,
    ompf: Bytes,
    snap_seq: u64,
    meta: &CheckpointMeta,
    max_seq_observed: Option<u64>,
    config: &Config,
) -> Result<(), CheckpointError> {
    apply_restore(
        layer_store,
        object_id,
        RestorePayload::Ompf(ompf),
        snap_seq,
        meta,
        max_seq_observed,
        config,
        None,
    )
    .await
    .map(|_| ())
}

#[cfg(test)]
// `pub(crate)`, on the same precedent as `checkpoint::compose::tests`: the
// sibling `checkpoint::residue` unit lane reuses this module's `test_config` /
// `install_mapping` / `seed_complete_header` fixtures rather than re-stating a
// 90-field `Config` literal that would then have to be kept in sync twice
// (b.f4t).
pub(crate) mod tests {
    use super::*;
    use crate::checkpoint::compose::tests as compose_tests;
    // The collecting wrapper — fixtures build a reference state and ignore
    // the mutations; production code here goes through `apply_delta_into_mode`.
    use crate::checkpoint::tail_codec::{decode_tail_envelope, TAIL_ENVELOPE_MAGIC};
    use crate::domain::delta::apply_delta;
    use crate::domain::ompf::OmpfContainer;
    use bytes::Bytes;
    use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand, TimeSample};

    /// Build a reset-delta OMPF carrying the supplied delta.
    ///
    /// Helper for the round-trip tests — wraps `build_delta` + `OmpfContainer::serialize`
    /// so each test focuses on its specific invariant.
    fn build_reset_ompf(delta: &OwnedDelta) -> Bytes {
        let delta_bytes =
            oodf_ffi::build_delta(delta).expect("build_delta must succeed for valid OwnedDelta");
        let ompf = OmpfContainer::serialize(&delta_bytes, &[])
            .expect("OMPF serialize must succeed for non-empty delta");
        Bytes::from(ompf)
    }

    #[test]
    fn decode_valid_empty_reset_delta_returns_empty_state() {
        // Smallest valid reset delta: no commands, no fields, no time
        // samples. `is_diff=true, base_version=0` is the reset marker the
        // assembler emits even for an empty object.
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            ..OwnedDelta::default()
        };
        let ompf = build_reset_ompf(&delta);

        let state = decode_checkpoint_ompf(42, &ompf).expect("decode of empty reset delta");
        assert_eq!(
            state.object_id, 42,
            "object_id must be the caller-supplied value"
        );
        assert_eq!(
            state.sequence_num, 0,
            "decoder must reset sequence_num to 0 (apply Subtask sets the SR-2.5 value)"
        );
        assert_eq!(
            state.first_version, 0,
            "decoder must reset first_version to 0 (apply Subtask sets it from meta)"
        );
        assert!(state.sections.is_empty());
        assert!(state.fields.is_empty());
        assert!(state.time_samples.is_empty());
    }

    #[test]
    fn decode_reset_delta_with_sections_and_fields_reconstructs_state() {
        // A reset delta carrying a single top-level section + one field
        // and one time sample on it. The reconstructed state must contain
        // exactly those entries.
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: 7,
                section_name: "alpha".to_string(),
                section_type: 1,
            })],
            set_fields: vec![SetField {
                section_id: 7,
                key_name: "k".to_string(),
                value_or_ext_hash: vec![0xde, 0xad, 0xbe, 0xef],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            time_samples: vec![TimeSample {
                section_id: 7,
                time: 1.5,
                value_or_ext_hash: vec![0x01, 0x02],
                ext_value_size: 0,
                ext_value_index: 0,
            }],
            source_format: "usda".to_string(),
        };
        let ompf = build_reset_ompf(&delta);

        let state = decode_checkpoint_ompf(99, &ompf).expect("decode");

        assert_eq!(state.object_id, 99);
        assert_eq!(state.source_format, "usda");
        assert_eq!(state.sections.len(), 1, "exactly one section");
        let section = state.sections.get(&7).expect("section 7 present");
        assert_eq!(section.name, "alpha");
        assert_eq!(section.section_type, 1);

        assert_eq!(state.fields.len(), 1, "exactly one field");
        let field = state
            .fields
            .get(&(7, "k".to_string()))
            .expect("field present");
        assert_eq!(field.section_id, 7);

        assert_eq!(state.time_samples.len(), 1, "exactly one time sample");
    }

    #[test]
    fn decode_empty_ompf_bytes_returns_permanent() {
        // `OmpfContainer::parse` accepts an empty slice (returns an empty
        // container). `oodf_ffi::parse_delta` then sees a zero-length
        // delta-bytes slice and returns `EmptyInput`. We surface that as
        // `Permanent`.
        let err = decode_checkpoint_ompf(1, &[])
            .expect_err("empty OMPF bytes must Err, not silently produce empty state");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "empty OMPF must classify Permanent; got {:?}",
            err
        );
    }

    #[test]
    fn decode_garbage_ompf_bytes_returns_permanent() {
        // 16 bytes of garbage that won't parse as a valid OMPF magic +
        // header. The OMPF layer rejects with `BadMagic` and we surface
        // Permanent.
        let garbage: Vec<u8> = (0..16).map(|i| i as u8).collect();
        let err = decode_checkpoint_ompf(1, &garbage).expect_err("garbage bytes must Err");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "garbage OMPF must classify Permanent; got {:?}",
            err
        );
        // The error message MUST mention OMPF parse so operators can
        // locate the failure layer in logs.
        let msg = err.to_string();
        assert!(
            msg.contains("OMPF") || msg.contains("ompf"),
            "Permanent error must name OMPF parse layer, got: {msg}"
        );
    }

    #[test]
    fn decode_checkpoint_ompf_rejects_a_tail_envelope_as_permanent() {
        // Two decoders now coexist in one restore path: this one owns BASE
        // blobs, `compose.rs`'s `decode_tail_record_ompf` owns tail records.
        // SDD §22.19 records that the `LETL` magic exists so cross-decoder
        // misuse "fails loudly and self-describingly instead of producing a
        // confusing `Permanent` from a foreign decoder" — a claim nothing
        // observed until tail bytes could reach this call graph at all.
        //
        // Real encoder output, never hand-assembled bytes: the composition
        // lane's `tail()` fixture drives `encode_tail_envelope` over the
        // canonical history.
        let (history, _base) = compose_tests::canonical();
        let records = history.records_in(compose_tests::BASE_SNAP_SEQ, compose_tests::HEAD_SEQ);
        let envelope_bytes = compose_tests::tail(
            compose_tests::BASE_SNAP_SEQ,
            compose_tests::BASE_SNAP_SEQ,
            &records,
        );

        // Non-vacuity: these bytes are a *well-formed* tail. The rejection
        // below is therefore the decoder boundary refusing a valid artifact of
        // the wrong kind — not a malformed-input rejection, which would prove
        // nothing about the boundary.
        assert_eq!(
            envelope_bytes[..4],
            TAIL_ENVELOPE_MAGIC[..],
            "the fixture must carry the LETL envelope magic"
        );
        assert!(
            decode_tail_envelope(&envelope_bytes).is_ok(),
            "the fixture must decode as a well-formed tail envelope"
        );

        let err = decode_checkpoint_ompf(compose_tests::OBJECT_ID, &envelope_bytes)
            .expect_err("a tail envelope must never decode as a checkpoint base blob");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "a foreign artifact must classify Permanent — re-reading identical bytes cannot \
             change the verdict; got {err:?}"
        );
        // Same convention as `decode_garbage_ompf_bytes_returns_permanent`
        // above: the message names the layer that refused, so an operator
        // reading a step-down log sees the OMPF parse boundary rather than a
        // shape complaint invented by a decoder that should never have been
        // handed these bytes.
        let msg = err.to_string();
        assert!(
            msg.contains("OMPF") || msg.contains("ompf"),
            "Permanent error must name OMPF parse layer, got: {msg}"
        );
    }

    // Reset-shape gate for the BASE path only: compose.rs's tail records deliberately bypass it.
    #[test]
    fn decode_non_reset_delta_is_diff_false_returns_permanent() {
        // An `is_diff=false` delta (e.g. a CreateLayer initial upload) is
        // NOT a reset delta. The checkpoint round-trip contract requires
        // the assembler to emit `is_diff=true, base_version=0`; anything
        // else surfaces Permanent because retrying won't fix it.
        let delta = OwnedDelta {
            is_diff: false,
            base_version: 0,
            ..OwnedDelta::default()
        };
        let ompf = build_reset_ompf(&delta);
        let err = decode_checkpoint_ompf(1, &ompf)
            .expect_err("is_diff=false delta must be rejected by checkpoint apply");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "non-reset delta must classify Permanent; got {:?}",
            err
        );
        // The error message MUST name both the observed `is_diff` and
        // `base_version` so an operator can diagnose without re-running
        // with verbose logging.
        let msg = err.to_string();
        assert!(
            msg.contains("is_diff=false"),
            "error message must name observed is_diff value, got: {msg}"
        );
    }

    #[test]
    fn decode_non_reset_delta_base_version_nonzero_returns_permanent() {
        // `is_diff=true` but `base_version != 0` is an incremental diff,
        // not a reset. The assembler never emits this shape; reject.
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 5,
            ..OwnedDelta::default()
        };
        let ompf = build_reset_ompf(&delta);
        let err =
            decode_checkpoint_ompf(1, &ompf).expect_err("base_version != 0 delta must be rejected");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "incremental delta must classify Permanent; got {:?}",
            err
        );
        let msg = err.to_string();
        assert!(
            msg.contains("base_version=5"),
            "error message must name observed base_version value, got: {msg}"
        );
    }

    // ── SR-14.7: telling an over-limit plain base from a corrupt one ────────
    //
    // An earlier release wrote a base over the per-delta entry limit as one
    // plain OMPF delta. Its decode fails like a corrupt base's, but the
    // recovery hook must refuse it rather than step down, so the parser's
    // verdict travels typed (`BaseDeltaParseError`) and
    // `is_over_entry_limit_base` reads it.

    /// A plain OMPF base one entry over the cap fails to decode with a
    /// `Permanent` carrying a typed `BaseDeltaParseError` of
    /// `OodfError::TooManyTables`, rendered with the same message as before
    /// the type existed, and `is_over_entry_limit_base` answers `true`.
    #[test]
    fn is_over_entry_limit_base_is_true_for_a_plain_base_over_the_entry_limit() {
        use crate::checkpoint::chunked_base::tests::plain_over_cap_base;

        let err = decode_checkpoint_ompf(1, &plain_over_cap_base())
            .expect_err("a plain base over the entry limit must not decode");

        assert!(
            is_over_entry_limit_base(&err),
            "the over-limit decode failure must be recognised; got {err:?}"
        );
        let CheckpointError::Permanent { source } = &err else {
            panic!("expected Permanent, got {err:?}");
        };
        let parse = source
            .downcast_ref::<BaseDeltaParseError>()
            .expect("the parser's verdict must travel typed");
        assert!(
            matches!(
                parse.source,
                OodfError::TooManyTables { max_entries, .. }
                    if max_entries == crate::domain::limits::MAX_DELTA_ENTRIES
            ),
            "{:?}",
            parse.source
        );
        assert!(
            err.to_string()
                .contains("OODF parse failed during checkpoint apply: delta carries"),
            "the rendered message is unchanged: {err}"
        );
    }

    /// Every other failure answers `false`: bytes that are not an OMPF, an
    /// OMPF whose delta is garbage (the same `BaseDeltaParseError` type with a
    /// different parser error), a non-reset delta, a `BaseDeltaParseError` of
    /// each other parser error, and the non-`Permanent` classes.
    #[test]
    fn is_over_entry_limit_base_is_false_for_every_other_failure() {
        let garbage_ompf: Vec<u8> = (0..16).map(|i| i as u8).collect();
        let garbage_delta = OmpfContainer::serialize(&b"definitely not an OODF delta"[..], &[])
            .expect("OMPF serialize");
        let non_reset = build_reset_ompf(&OwnedDelta {
            is_diff: false,
            ..OwnedDelta::default()
        });

        for (label, bytes) in [
            ("bytes that are not an OMPF", garbage_ompf.as_slice()),
            ("an OMPF carrying a garbage delta", garbage_delta.as_slice()),
            ("a non-reset delta", &non_reset[..]),
        ] {
            let err = decode_checkpoint_ompf(1, bytes).expect_err(label);
            assert!(
                matches!(err, CheckpointError::Permanent { .. }),
                "{label}: precondition, a Permanent decode failure; got {err:?}"
            );
            assert!(!is_over_entry_limit_base(&err), "{label}: {err:?}");
        }
        assert!(
            decode_checkpoint_ompf(1, &garbage_delta)
                .err()
                .and_then(|err| match err {
                    CheckpointError::Permanent { source } => source
                        .downcast_ref::<BaseDeltaParseError>()
                        .map(|e| !matches!(e.source, OodfError::TooManyTables { .. })),
                    _ => None,
                })
                .unwrap_or(false),
            "precondition: a garbage delta is the same typed error with another parser verdict"
        );

        for other in [
            OodfError::EmptyInput,
            OodfError::TooShort,
            OodfError::BadMagic,
            OodfError::VerifierFailed("bad offset".to_string()),
        ] {
            let label = format!("{other:?}");
            let err = CheckpointError::Permanent {
                source: Box::new(BaseDeltaParseError { source: other }),
            };
            assert!(!is_over_entry_limit_base(&err), "{label}");
        }
        assert!(!is_over_entry_limit_base(&CheckpointError::Transient {
            source: "transient".into(),
        }));
        assert!(!is_over_entry_limit_base(&CheckpointError::NotFound));
        assert!(!is_over_entry_limit_base(&CheckpointError::Conflict));
    }

    #[test]
    fn decode_round_trip_state_matches_apply_delta_baseline() {
        // Verify that decoding a reset-delta OMPF produces the SAME state
        // an in-process `apply_delta` against a fresh empty state would
        // produce. This is the "reconstructed state matches encoded delta
        // bit-for-bit" acceptance criterion.
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            struct_commands: vec![
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 0,
                    section_id: 1,
                    section_name: "root".to_string(),
                    section_type: 0,
                }),
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 1,
                    section_id: 2,
                    section_name: "child".to_string(),
                    section_type: 0,
                }),
            ],
            set_fields: vec![SetField {
                section_id: 1,
                key_name: "name".to_string(),
                value_or_ext_hash: b"value".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            time_samples: vec![],
            source_format: "usda".to_string(),
        };

        // Baseline: apply against a fresh ObjectState directly.
        let mut baseline = ObjectState {
            object_id: 42,
            ..ObjectState::default()
        };
        apply_delta(&mut baseline, Some(delta.clone()))
            .expect("baseline apply must succeed for a well-formed reset delta");
        baseline.sequence_num = 0; // match decoder's normalization.
        baseline.first_version = 0;

        // Decoder path.
        let ompf = build_reset_ompf(&delta);
        let decoded = decode_checkpoint_ompf(42, &ompf).expect("decode");

        // The two paths must produce structurally-identical state. We
        // compare the load-bearing maps explicitly rather than via the
        // derived `PartialEq` so a future field addition surfaces here
        // first.
        assert_eq!(decoded.object_id, baseline.object_id);
        assert_eq!(decoded.source_format, baseline.source_format);
        assert_eq!(decoded.next_set_order, baseline.next_set_order);
        assert_eq!(decoded.sections, baseline.sections);
        assert_eq!(decoded.children, baseline.children);
        assert_eq!(decoded.fields, baseline.fields);
        assert_eq!(decoded.time_samples, baseline.time_samples);
    }

    // ── b.8jc — a base whose bytes carry a `-0.0` time ──────────────────────
    //
    // A base is bytes read back from the checkpoint adapter, and restore must
    // decode whatever a build wrote there. A base whose time sample carries
    // `-0.0` must restore, with the sample at `+0.0` (the one stored spelling
    // of time code zero), never be refused. Every base path parses through
    // `oodf_ffi::parse_delta`, which canonicalizes the time before
    // `ApplyMode::RestoreBase` runs `validate_delta_content`; that function's
    // `TimeSampleNegativeZeroTime` backstop would otherwise refuse the base.
    // The chunked-base twin is in `chunked_base::tests`.

    /// A sample time no checkpoint fixture uses for anything else; a sample
    /// built at it is patched to `-0.0` by
    /// [`build_delta_with_negative_zero_time`].
    pub(crate) const NEGATIVE_ZERO_SENTINEL: f64 = 2_718.281_828_459;

    /// The OODF bytes of `delta` with the sample time `sentinel` rewritten to
    /// `-0.0` in place.
    ///
    /// `oodf_ffi::build_delta` cannot write `-0.0`: FlatBuffers drops a field
    /// equal to its default, and `-0.0 == 0.0`, so a sample built at `-0.0`
    /// parses back at `+0.0` and a test of it would prove nothing. The sample
    /// is built at `sentinel` instead, which FlatBuffers stores, and its 8
    /// little-endian bytes are overwritten with `-0.0`'s. Panics unless the
    /// built bytes carry the `-0.0` pattern nowhere and `sentinel` exactly
    /// once, and `-0.0` exactly once after patching, so a collision with
    /// another byte run fails the fixture rather than patching the wrong
    /// bytes. `tests/common/owned_delta.rs`'s
    /// `build_delta_with_negative_zero_times` is the integration-test twin,
    /// which unit tests cannot reach.
    pub(crate) fn build_delta_with_negative_zero_time(
        delta: &OwnedDelta,
        sentinel: f64,
    ) -> Vec<u8> {
        fn positions(bytes: &[u8], pattern: [u8; 8]) -> Vec<usize> {
            bytes
                .windows(8)
                .enumerate()
                .filter(|(_, w)| *w == pattern)
                .map(|(at, _)| at)
                .collect()
        }
        let negative_zero = (-0.0f64).to_le_bytes();
        let mut bytes = oodf_ffi::build_delta(delta).expect("the fixture delta must build");
        assert!(
            positions(&bytes, negative_zero).is_empty(),
            "fixture: the built delta must not carry the -0.0 byte pattern before patching"
        );
        let at = positions(&bytes, sentinel.to_le_bytes());
        assert_eq!(
            at.len(),
            1,
            "fixture: sentinel time {sentinel} must occur exactly once in the built delta"
        );
        bytes[at[0]..at[0] + 8].copy_from_slice(&negative_zero);
        assert_eq!(
            positions(&bytes, negative_zero).len(),
            1,
            "fixture: the patched delta carries -0.0 exactly once"
        );
        bytes
    }

    /// Asserts that `state` holds `section_id`'s time-zero sample at `+0.0`,
    /// its key and its stored time both bit-exact `0`, with the inline
    /// `value`, and that no sample anywhere in `state` sits at `-0.0`.
    pub(crate) fn assert_sample_restored_at_positive_zero(
        state: &ObjectState,
        section_id: u64,
        value: &[u8],
    ) {
        use crate::domain::types::FieldValue;
        use ordered_float::OrderedFloat;

        // `OrderedFloat` hashes and compares `0.0` and `-0.0` as one key, so
        // this lookup finds the row under either spelling; the bit checks
        // tell the two apart.
        let (key, sample) = state
            .time_samples
            .get_key_value(&(section_id, OrderedFloat(0.0)))
            .unwrap_or_else(|| {
                panic!("the restored state must hold section {section_id}'s time-0 sample")
            });
        assert_eq!(
            (
                key.1.into_inner().to_bits(),
                sample.time.into_inner().to_bits()
            ),
            (0, 0),
            "section {section_id}: the restored sample must sit at +0.0, key and time alike"
        );
        assert_eq!(
            sample.value,
            FieldValue::Inline(value.to_vec()),
            "section {section_id}: the sample's value must survive the restore intact"
        );
        let negative_zero = (-0.0f64).to_bits();
        for (key, sample) in &state.time_samples {
            assert!(
                key.1.into_inner().to_bits() != negative_zero
                    && sample.time.into_inner().to_bits() != negative_zero,
                "no restored sample may sit at -0.0; found one under key {key:?}"
            );
        }
    }

    /// A plain OMPF base whose bytes physically carry a `-0.0` sample time
    /// decodes, and the restored state holds the sample at `+0.0` with its
    /// value intact, beside an untouched sample at another time.
    ///
    /// Without `parse_delta`'s canonicalization the `-0.0` reaches
    /// `validate_delta_content` under `ApplyMode::RestoreBase`, and the base is
    /// refused `Permanent` with `TimeSampleNegativeZeroTime`.
    #[test]
    fn decode_checkpoint_ompf_base_carrying_negative_zero_time_restores_the_sample_at_positive_zero(
    ) {
        const ZERO_VALUE: [u8; 2] = [0x0d, 0x0e];
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: 7,
                section_name: "alpha".to_string(),
                section_type: 1,
            })],
            set_fields: vec![],
            time_samples: vec![
                TimeSample {
                    section_id: 7,
                    time: NEGATIVE_ZERO_SENTINEL,
                    value_or_ext_hash: ZERO_VALUE.to_vec(),
                    ext_value_size: 0,
                    ext_value_index: 0,
                },
                TimeSample {
                    section_id: 7,
                    time: 2.5,
                    value_or_ext_hash: vec![0x02],
                    ext_value_size: 0,
                    ext_value_index: 0,
                },
            ],
            source_format: "usda".to_string(),
        };
        let delta_bytes = build_delta_with_negative_zero_time(&delta, NEGATIVE_ZERO_SENTINEL);
        let ompf = OmpfContainer::serialize(&delta_bytes, &[]).expect("OMPF serialize");
        assert!(
            !chunked_base::is_chunked_base(&ompf),
            "precondition: this is the plain OMPF base path"
        );

        let state = decode_checkpoint_ompf(42, &ompf)
            .expect("a base whose bytes carry a -0.0 time must restore, not be refused");

        assert_eq!(
            state.time_samples.len(),
            2,
            "both samples are restored, neither dropped nor merged"
        );
        assert_sample_restored_at_positive_zero(&state, 7, &ZERO_VALUE);
        let other = state
            .time_samples
            .get(&(7, ordered_float::OrderedFloat(2.5)))
            .expect("the sample at 2.5 is restored");
        assert_eq!(
            other.value,
            crate::domain::types::FieldValue::Inline(vec![0x02])
        );
    }

    // -------------------------------------------------------------------
    // apply_checkpoint smoke tests (structural-header short-circuit +
    // guard-false arms).
    //
    // These two arms are the load-bearing "fast-fail" paths of the
    // recovery primitive and they MUST exercise live Valkey state (the
    // header round-trip / the SCAN-based layer-mapping guard cannot be
    // mocked without rebuilding fred). The Subtask body intentionally
    // scopes us to two unit smoke tests; broader integration coverage
    // (decode success path, error mapping across all six steps, the
    // last_checkpoint_seq HSET) belongs to the sibling test-writer
    // Subtask.
    //
    // Both tests use [`TestValkey::setup`] for an isolated key namespace.
    // They will skip silently with a `#[ignore]` ladder if Valkey is
    // unreachable — but they are NOT marked `#[ignore]` here because the
    // CI `test:` job spins up Valkey via `scripts/dev-valkey.sh up`
    // before running the full suite, so the smoke arms execute on every
    // PR.
    // -------------------------------------------------------------------

    use crate::storage::layer::LayerStore;
    use crate::storage::test_support::TestValkey;
    use fred::interfaces::HashesInterface;
    use std::collections::{HashMap, HashSet};

    /// Build a `Config` carrying just the fields `apply_checkpoint` reads.
    /// All other fields take their `LIVE_EDIT_*` defaults so the test does
    /// not have to track every future field addition (the apply path only
    /// consumes `checkpoint_seq_safety_margin`; everything else is filler).
    pub(crate) fn test_config(safety_margin: u64) -> crate::config::Config {
        crate::config::Config {
            catchup_snapshot_enabled: false,
            port: 0,
            metrics_port: 0,
            valkey_url: String::new(),
            valkey_pool_size: 1,
            subscriber_pool_size: 1,
            subscriber_admission_wait_ms: 2_000,
            subscriber_max_subscriptions: 4096,
            subscriber_max_concurrent_catchups: 64,
            subscriber_admission_max_waiters: None,
            maintenance_pool_size: 1,
            catchup_snapshot_read_concurrency:
                crate::config::CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT,
            catchup_fenced_read_concurrency: crate::config::CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT,
            catchup_build_memory_budget_mib: crate::config::CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
            snapshot_pool_size: 1,
            snapshot_atomic_read_max_items: crate::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
            snapshot_atomic_read_max_bytes: crate::config::SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
            structural_cache_max_mib: crate::config::STRUCTURAL_CACHE_MAX_MIB_DEFAULT,
            structural_cache_audit_rate: crate::config::STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT,
            structural_cache_audit_above_ceiling: false,
            valkey_connect_timeout_secs: 5,
            valkey_command_timeout_ms: 5000,
            valkey_reconnect_max_delay_ms: 30_000,
            valkey_auth_mode: crate::config::ValkeyAuthMode::Static,
            valkey_token_file: None,
            valkey_token_username: None,
            valkey_token_refresh_lead_secs: 180,
            valkey_iam_username: None,
            valkey_iam_region: None,
            valkey_iam_cluster_host: None,
            valkey_iam_service: crate::config::ValkeyIamService::MemoryDb,
            storage_service_endpoint: None,
            auth_timeout_ms: 1000,
            auth_cache_ttl_secs: 60,
            auth_cache_negative_ttl_secs: 30,
            auth_userinfo_negative_ttl_secs: 5,
            auth_cache_max_entries: 1_000,
            userinfo_endpoint: None,
            max_layer_size: 1_048_576,
            upload_memory_budget_mib:
                crate::grpc::upload_admission::UPLOAD_MEMORY_BUDGET_MIB_DEFAULT,
            upload_admission_wait_ms:
                crate::grpc::upload_admission::UPLOAD_ADMISSION_WAIT_MS_DEFAULT,
            upload_admission_max_waiters:
                crate::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
            layer_stream_maxlen: 4096,
            layer_stream_max_bytes: crate::config::LAYER_STREAM_MAX_BYTES_DEFAULT,
            presence_broadcast_capacity: 4096,
            presence_lease_ttl_secs: crate::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
            presence_refresh_interval_secs: crate::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
            presence_keepalive_interval_ms: crate::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
            shutdown_timeout_secs: 1,
            shutdown_overall_budget_secs: 5,
            shutdown_stream_end_rate: 25,
            shutdown_stream_end_max_window_secs: 20,
            log_level: "warn".to_string(),
            log_format: crate::config::LogFormat::Json,
            trace_filter: crate::config::DEFAULT_TRACE_FILTER.to_string(),
            sweeper_interval_secs: 15,
            session_list_timeout_ms: 1000,
            rpc_deadline_ms: 5000,
            service_label: "test".to_string(),
            valkey_memory_bytes: None,
            ext_release_max_bytes: None,
            ext_reconcile_interval_secs: 3600,
            auth_max_attempts: 3,
            auth_retry_initial_delay_ms: 100,
            authn_enabled: true,
            authz_enabled: false,
            userinfo_service_enabled: true,
            test_only_dangerously_allow_unvalidated_tokens: false,
            health_probe_interval_ms: 5_000,
            health_probe_timeout_ms: 2_000,
            auth_protocol: crate::config::AuthProtocol::Rest,
            update_layer_retry_max_attempts: 128,
            update_layer_per_object_concurrency: 16,
            update_layer_backend: crate::config::UpdateLayerBackend::Lua,
            ext_value_verify: crate::config::ExtValueVerifyMode::Enforce,
            session_retry_max_attempts: 32,
            session_retry_initial_delay_ms: 1,
            pod_memory: crate::config::PodMemoryPlan::reference(),
            session_tokens_sweep_interval_secs:
                crate::grpc::session::SESSION_TOKENS_SWEEP_INTERVAL_SECS,
            checkpoint_enabled: false,
            checkpoint_adapter: crate::checkpoint::CheckpointAdapter::Disabled,
            checkpoint_interval_secs: 60,
            checkpoint_quiesce_secs: 30,
            checkpoint_tick_secs: 10,
            checkpoint_retain_generations: 2,
            checkpoint_seq_safety_margin: safety_margin,
            checkpoint_counters_sync_secs: 60,
            checkpoint_reconcile_interval_secs: 3600,
            checkpoint_counter_safety_margin: 1000,
            checkpoint_tail_enabled: false,
            checkpoint_tail_max_chain: 8,
            checkpoint_tail_max_entries: 4096,
            checkpoint_tail_max_bytes: 67_108_864,
            checkpoint_local_disk_path: None,
            checkpoint_storage_api_endpoint: None,
            checkpoint_base_path: None,
            checkpoint_connect_timeout_ms: 5_000,
            checkpoint_request_timeout_ms: 30_000,
            checkpoint_ext_mirror_workers: 2,
            checkpoint_ext_mirror_queue_capacity: 1_000,
            service_token_file: None,
            service_token_poll_interval_secs: 15,
            service_token_client_credentials_enabled: false,
        }
    }

    /// Step-1 short-circuit: if `obj:hdr` is already present AND
    /// structurally complete for the target `object_id` (a sibling replica
    /// raced ahead with recovery, or the per-`object_id` admission lock let
    /// two attempts through on the same replica), `apply_checkpoint` MUST
    /// short-circuit with `Ok(())` without invoking the decoder, the guard,
    /// the residue wipe, or `write_restored_object`.
    ///
    /// We assert the short-circuit by passing deliberately invalid OMPF
    /// bytes (`&[]`) and an empty meta: if any step beyond step 1 ran,
    /// the test would surface `CheckpointError::Permanent` from the
    /// decoder. `Ok(())` proves the short-circuit fired and the rest of
    /// the body was skipped.
    ///
    /// b.f4t: the fixture seeds a **structurally complete** header. A
    /// `seq`-only header — what this test used to install — is now
    /// deliberately NOT a short-circuit (see the two sibling tests below),
    /// so seeding one here would prove the opposite of what the name claims.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_checkpoint_structurally_complete_header_short_circuits() {
        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);

        // Pre-populate obj:hdr for object 42 so the step-1 check trips.
        let header_key = tv.key_schema().obj_header("42");
        seed_complete_header(&tv, 42, "100").await;

        // Deliberately broken inputs: empty OMPF bytes would surface
        // `Permanent` from the decoder if step 3 ran.
        let meta = CheckpointMeta::default();
        let config = test_config(1000);

        let result = apply_checkpoint(
            &layer_store,
            42,
            Bytes::new(), // would fail decode
            0,
            &meta,
            None,
            &config,
        )
        .await;

        assert!(
            result.is_ok(),
            "the structurally-complete-header arm must short-circuit Ok(()); got {:?}",
            result.err()
        );

        // Verify the short-circuit did NOT clobber the pre-existing header
        // (i.e. we really skipped write_restored_object). The pre-installed `seq`
        // field MUST still read `100`.
        let seq: Option<String> = tv
            .client()
            .pool
            .hget(&header_key, "seq")
            .await
            .expect("HGET seq after apply_checkpoint short-circuit");
        assert_eq!(
            seq.as_deref(),
            Some("100"),
            "the short-circuit must not invoke write_restored_object (obj:hdr seq was clobbered)"
        );

        // Verify `last_checkpoint_seq` was NOT written either — the
        // short-circuit path skips step 6 by design.
        let lcs: Option<String> = tv
            .client()
            .pool
            .hget(&header_key, "last_checkpoint_seq")
            .await
            .expect("HGET last_checkpoint_seq after short-circuit");
        assert!(
            lcs.is_none(),
            "the short-circuit must not run the post-write HSET; got last_checkpoint_seq={lcs:?}"
        );
    }

    /// Shared body for the two structurally-PARTIAL header fixtures (b.f4t).
    ///
    /// `partial_pairs` is the header the restore finds in place: it always
    /// carries the two stale non-canonical fields with teeth, and it always
    /// omits exactly one of [`HEADER_STRUCTURAL_FIELDS`]. Both shapes must
    /// (a) fall THROUGH the step-1 short-circuit rather than being reported
    /// already-present, and (b) come out of the restore with a repaired,
    /// structurally complete header carrying neither stale field — which is only true
    /// because step 3b's wipe removes `obj:hdr` before `write_restored_object`
    /// evaluates its header fence. That function does not `DEL` the key first
    /// and never HSETs through a header it finds — since b.h4m a
    /// present-but-incomplete header is refused outright — so the wipe is what
    /// makes this shape converge at all. Pre-b.h4m it merely made the stale
    /// fields disappear; without it they survived underneath the restored
    /// header, where `ext_gc_scheduled=1` makes the restored object's later
    /// ext-GC a silent no-op and `superseded=1` makes the restored layer
    /// permanently unwritable on both commit backends.
    async fn assert_partial_header_is_restored_and_repaired(
        object_id: u64,
        partial_pairs: Vec<(&'static str, String)>,
        omitted_field: &str,
    ) {
        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        install_mapping(&tv, object_id).await;

        let header_key = tv.key_schema().obj_header(&object_id.to_string());
        let _: () = tv
            .client()
            .pool
            .hset(&header_key, partial_pairs)
            .await
            .expect("HSET obj:hdr to install the partial header");

        // A valid reset-delta OMPF: if the restore is (wrongly) short-
        // circuited, nothing below changes; if it runs, it lands a full
        // header.
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            ..OwnedDelta::default()
        };
        let ompf = build_reset_ompf(&delta);
        let config = test_config(1000);
        let snap_seq = 30u64;

        apply_checkpoint(
            &layer_store,
            object_id,
            ompf,
            snap_seq,
            &composed_meta(),
            None,
            &config,
        )
        .await
        .expect("a structurally-partial header must NOT short-circuit; the restore must run");

        let header: HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(&header_key)
            .await
            .expect("HGETALL obj:hdr after the restore");

        assert!(
            header_fields_are_structurally_complete(&header),
            "the restore MUST repair the header (the `{omitted_field}` that was missing is what \
             made its live readers answer NotFound); got {header:?}"
        );
        assert_eq!(
            header.get("seq").map(String::as_str),
            Some((snap_seq + config.checkpoint_seq_safety_margin).to_string()).as_deref(),
            "the repaired header must carry the seeded sequence, not the partial header's old \
             value — a restore that ran is the only way to get here"
        );

        // The load-bearing half: NO stale non-canonical field survives.
        for stale in ["superseded", "ext_gc_scheduled"] {
            assert!(
                !header.contains_key(stale),
                "`{stale}` survived the restore. `write_restored_object` HSETs the 12 canonical \
                 fields WITHOUT DELing the key, so the only thing that can remove a non-canonical \
                 field is step 3b's residue wipe — this is exactly the b.f4t defect. A \
                 surviving `superseded=1` makes the restored layer permanently unwritable on \
                 both commit backends; a surviving `ext_gc_scheduled=1` makes its later \
                 ext-GC a silent no-op. Got {header:?}"
            );
        }
    }

    /// A header present but missing `created_by` — what
    /// `stat_layer_for_object_id` already answers `NotFound` on — must drive a
    /// REAL restore, not a short-circuit.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_checkpoint_partial_header_missing_created_by_is_restored() {
        assert_partial_header_is_restored_and_repaired(
            8_120,
            vec![
                ("seq", "100".to_owned()),
                ("first_ver", "0".to_owned()),
                ("session_id", "s1".to_owned()),
                ("layer_name", "l".to_owned()),
                // The two stale fields with teeth (see the shared body).
                ("superseded", "1".to_owned()),
                ("ext_gc_scheduled", "1".to_owned()),
            ],
            "created_by",
        )
        .await;
    }

    /// A header present but missing `seq` — what `read_object_header` already
    /// answers `NotFound` on — must drive a REAL restore too. The two
    /// structural fields belong to different consumers, so both halves of the
    /// union need their own fixture.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_checkpoint_partial_header_missing_seq_is_restored() {
        assert_partial_header_is_restored_and_repaired(
            8_121,
            vec![
                ("created_by", "stale-user".to_owned()),
                ("first_ver", "0".to_owned()),
                ("session_id", "s1".to_owned()),
                ("layer_name", "l".to_owned()),
                ("superseded", "1".to_owned()),
                ("ext_gc_scheduled", "1".to_owned()),
            ],
            "seq",
        )
        .await;
    }

    /// Guard-false arm: if the live `le:{sid}:layer:{hex(name)}` mapping
    /// named by the blob's own identity does not resolve to the target
    /// `object_id`, `apply_checkpoint` MUST return
    /// `CheckpointError::NotFound` WITHOUT writing anything.
    ///
    /// We assert the guard fires (and that no state was written) by
    /// inspecting the absence of `obj:hdr` after the call —
    /// `write_restored_object` would have created it if it had run.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_checkpoint_guard_false_returns_not_found() {
        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);

        // Do NOT install any layer mapping. The keyspace is empty, so the
        // targeted GET inside `live_mapping_resolves_blob_identity` misses
        // and the guard returns Ok(false).

        // Build a valid reset-delta OMPF so step 3 would succeed if
        // step 2 didn't bail. This proves the guard is the layer that
        // rejected the apply (not the decoder).
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            ..OwnedDelta::default()
        };
        let ompf = build_reset_ompf(&delta);

        let meta = CheckpointMeta {
            session_id: "s1".to_string(),
            layer_name: "l".to_string(),
            created_by: "u".to_string(),
            server_ts: 123,
            ..CheckpointMeta::default()
        };
        let config = test_config(1000);

        let result = apply_checkpoint(&layer_store, 7777, ompf, 0, &meta, None, &config).await;

        match result {
            Err(CheckpointError::NotFound) => {}
            other => panic!("guard-false arm must return NotFound; got {other:?}"),
        }

        // Confirm nothing was written for object 7777.
        let header_key = tv.key_schema().obj_header("7777");
        let exists: bool = tv
            .client()
            .pool
            .exists(&header_key)
            .await
            .expect("EXISTS obj:hdr after guard-false return");
        assert!(
            !exists,
            "guard-false arm must NOT call write_restored_object; obj:hdr was created"
        );
    }

    // ── b.f4t: the structural-presence predicate and its two consumers ────

    /// `HSET` a structurally-complete `obj:hdr` — one carrying BOTH fields
    /// [`HEADER_STRUCTURAL_FIELDS`] names.
    ///
    /// These fixtures stand in for "a peer replica already landed the restore",
    /// and seed the 12 canonical fields both materialization entry points stamp
    /// (the shared `queue_object_state` body) — deliberately only those. All
    /// twelve rather than the two-field minimum: a fixture seeding only the two
    /// fields the predicate reads would pass whatever the predicate's field list
    /// happened to be, which is the thing under test. No more than the twelve,
    /// because the header only has to look structurally complete. A real peer's
    /// header is a superset: its restore also writes the children-index marker
    /// `chs_complete` and the three checkpoint-cursor columns
    /// (`last_checkpoint_seq`, `last_checkpoint_ms`, `restore_seq_bump`). So this
    /// fixture header is unmarked, and once the atomic snapshot read trusts
    /// `obj:chs` only on marked objects (b.y3m Epic 2, Task 4) it takes the
    /// unmarked fallback path where a real peer's restored object would not.
    pub(crate) async fn seed_complete_header(tv: &TestValkey, object_id: u64, seq: &str) {
        let header_key = tv.key_schema().obj_header(&object_id.to_string());
        let pairs: Vec<(&'static str, String)> = vec![
            ("seq", seq.to_owned()),
            ("first_ver", "0".to_owned()),
            ("src_fmt", "usda".to_owned()),
            ("next_set_order", "0".to_owned()),
            ("session_id", "s1".to_owned()),
            ("layer_name", "l".to_owned()),
            ("created_by", "peer-replica-user".to_owned()),
            ("created_at", "1700000000000".to_owned()),
            ("modified_at", "1700000000000".to_owned()),
            ("last_client_id", String::new()),
            ("last_client_ts", "0".to_owned()),
            ("last_server_ts", "1700000000000".to_owned()),
        ];
        let _: () = tv
            .client()
            .pool
            .hset(&header_key, pairs)
            .await
            .expect("HSET obj:hdr to install a structurally-complete header");
    }

    // The predicate's own three unit tests are NOT here. It is declared in
    // `crate::storage::layer` (b.6de sub-task 6, landed with b.h4m) and this
    // module only re-exports it, so its tests live beside the declaration:
    // `crate::storage::layer::tests::header_fields_are_structurally_complete_*`.
    // What this module's lane owns is the CONSUMER behaviour — the
    // short-circuit arms above, which drive a real `apply_checkpoint` against a
    // seeded header.

    /// A fred `WRONGTYPE` reply → `Permanent`, naming the key an operator has
    /// to remove.
    ///
    /// Kind-gated AND prefix-anchored, so this test builds the error the way
    /// fred's `pretty_error` does (`ErrorKind::InvalidArgument` + a `details`
    /// string starting `WRONGTYPE`) rather than asserting on a substring
    /// search that would also match a message merely quoting the word.
    #[test]
    fn classify_object_key_read_error_wrongtype_is_permanent_and_names_the_key() {
        let err = fred::error::Error::new(
            fred::error::ErrorKind::InvalidArgument,
            "WRONGTYPE Operation against a key holding the wrong kind of value",
        );
        let classified = classify_object_key_read_error(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "SMEMBERS obj:secs",
            "le:{9}:obj:secs",
            9,
            err,
        );
        assert!(
            matches!(classified, CheckpointError::Permanent { .. }),
            "a WRONGTYPE cannot be re-read by any retry, so folding it to Transient makes the \
             recovery hook re-drive a restore that can never succeed; got {classified:?}"
        );
        let msg = classified.to_string();
        assert!(
            msg.contains("le:{9}:obj:secs"),
            "the refusal MUST name the offending key verbatim — Valkey's WRONGTYPE reply names \
             no key, so this is the operator's only locator. Got: {msg}"
        );
        assert!(
            msg.contains("SMEMBERS obj:secs"),
            "the refusal MUST name the command / key-family label too. Got: {msg}"
        );
    }

    #[test]
    fn classify_object_key_read_error_other_kinds_are_transient() {
        // A backend blip is exactly what the recovery loop's retry exists for.
        let timeout = fred::error::Error::new(fred::error::ErrorKind::Timeout, "timed out");
        assert!(
            matches!(
                classify_object_key_read_error(
                    FredErrOrigin::metrics(
                        None,
                        POOL_LABEL_DATA,
                        crate::storage::store_oom::StoreOomSignal::unwired(),
                    ),
                    "HGETALL obj:hdr",
                    "le:{9}:obj:hdr",
                    9,
                    timeout
                ),
                CheckpointError::Transient { .. }
            ),
            "a Timeout must stay Transient"
        );

        let io = fred::error::Error::new(fred::error::ErrorKind::IO, "connection reset");
        assert!(
            matches!(
                classify_object_key_read_error(
                    FredErrOrigin::metrics(
                        None,
                        POOL_LABEL_DATA,
                        crate::storage::store_oom::StoreOomSignal::unwired(),
                    ),
                    "HGETALL obj:hdr",
                    "le:{9}:obj:hdr",
                    9,
                    io
                ),
                CheckpointError::Transient { .. }
            ),
            "an IO error must stay Transient"
        );
    }

    /// The kind gate alone is not the predicate: fred raises
    /// `InvalidArgument` for its own local argument validation too, and those
    /// are not wrong-typed keys.
    #[test]
    fn classify_object_key_read_error_non_wrongtype_invalid_argument_is_transient() {
        let err = fred::error::Error::new(
            fred::error::ErrorKind::InvalidArgument,
            "invalid argument: expected a non-empty key",
        );
        let classified = classify_object_key_read_error(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_DATA,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "HGETALL obj:hdr",
            "le:{9}:obj:hdr",
            9,
            err,
        );
        assert!(
            matches!(classified, CheckpointError::Transient { .. }),
            "an InvalidArgument whose details do NOT start with `WRONGTYPE` must stay Transient \
             — the prefix anchor is what keeps the kind gate from swallowing fred's own local \
             validation errors; got {classified:?}"
        );
    }

    /// The structural keyspace refusals the restore materialization can surface
    /// are permanent by construction: an operator has to remove a key before any
    /// retry can succeed.
    ///
    /// The classification is keyed on the VARIANT, not on the message text
    /// (b.6de sub-task 2), and both kinds are covered because the arm is
    /// deliberately kind-agnostic. Today the restore path reaches
    /// [`KeyspaceRefusal::WrongType`] from two sites — its pre-`MULTI` keyspace
    /// probe and, since b.h4m, its own `HGETALL obj:hdr` fence via
    /// `classify_restore_header_read_error` — and cannot reach
    /// [`KeyspaceRefusal::StreamIdCollision`] at all, since `write_restored_object`
    /// contains no stream guard and queues no `XADD`. The kind is covered anyway:
    /// a condition that stops being reachable must not silently stop being
    /// classified, or re-introducing the guard later would hand the recovery hook
    /// a retriable verdict for a condition no retry clears.
    #[test]
    fn write_object_err_to_checkpoint_err_structural_refusals_are_permanent() {
        for refusal in [
            KeyspaceRefusal::WrongType,
            KeyspaceRefusal::StreamIdCollision,
        ] {
            let err = ServiceError::PermanentKeyspaceRefusal {
                refusal,
                detail: "create_layer keyspace type error: WRONGTYPE_KEY: le:{9}:obj:secs holds \
                         a string, expected zset"
                    .to_owned(),
            };
            let classified = write_object_err_to_checkpoint_err(err);
            let CheckpointError::Permanent { source } = &classified else {
                panic!(
                    "`{refusal:?}` must classify Permanent — folding it to Transient makes the \
                     recovery hook re-drive a restore that cannot succeed, re-ticking \
                     keyspace_type_error_total and surfacing a retriable UNAVAILABLE for a \
                     condition no retry clears; got {classified:?}"
                );
            };
            assert!(
                source
                    .to_string()
                    .contains("WRONGTYPE_KEY: le:{9}:obj:secs"),
                "the emitter's message must survive verbatim into `source` — it names the key an \
                 operator has to remove, and it is the only locator the recovery log carries. \
                 Got: {source}"
            );
        }
    }

    /// The point of the typed variant: a plain [`ServiceError::Internal`] whose
    /// *text* quotes a reply tag is **Transient**.
    ///
    /// Pre-b.6de this classifier matched `msg.contains(KEYSPACE_WRONGTYPE_TAG)`,
    /// so any `Internal` that merely mentioned the tag — including the five
    /// emitters not reachable from [`LayerStore::write_restored_object`] that
    /// still raise one, and any future message that quotes a tag while reporting
    /// something else entirely — was misclassified Permanent and abandoned a
    /// recoverable layer. Text is no longer evidence; the variant is.
    #[test]
    fn write_object_err_to_checkpoint_err_tag_text_alone_is_transient() {
        for tag in ["WRONGTYPE_KEY", "STREAM_ID_COLLISION"] {
            let err = ServiceError::Internal(format!(
                "HSET section failed while queueing (unrelated to {tag})"
            ));
            let classified = write_object_err_to_checkpoint_err(err);
            assert!(
                matches!(classified, CheckpointError::Transient { .. }),
                "an untyped `Internal` quoting `{tag}` must stay Transient — permanence is now \
                 carried by `ServiceError::PermanentKeyspaceRefusal`, and re-deriving it from \
                 message text is exactly what this variant replaced; got {classified:?}"
            );
        }
    }

    #[test]
    fn write_object_err_to_checkpoint_err_other_variants_keep_their_classification() {
        assert!(
            matches!(
                write_object_err_to_checkpoint_err(ServiceError::NotFound("gone".to_owned())),
                CheckpointError::NotFound
            ),
            "NotFound is forwarded verbatim"
        );
        assert!(
            matches!(
                write_object_err_to_checkpoint_err(ServiceError::Unavailable(
                    "valkey blip".to_owned()
                )),
                CheckpointError::Transient { .. }
            ),
            "the b.zvd create-site transient arm must stay retriable"
        );
        assert!(
            matches!(
                write_object_err_to_checkpoint_err(ServiceError::Internal(
                    "HSET section failed: connection reset".to_owned()
                )),
                CheckpointError::Transient { .. }
            ),
            "a plain Internal (queue-time / EXEC frame) must stay Transient — only the two \
             tagged structural refusals are permanent"
        );
    }

    /// SR-2.5 f64 safe-integer boundary: `seeded_seq` reaching exactly
    /// `2^53 − 1` MUST be rejected as `Permanent` BEFORE any write lands.
    ///
    /// The Lua fast-path (`update_layer_fast.lua`) and the slow path
    /// (`apply_delta` in `src/domain/delta.rs`) both reject sequence
    /// numbers at or above `2^53 − 1` (`SEQUENCE_CEILING`), so seeding to
    /// the ceiling would materialize an unwritable layer — every
    /// subsequent `UpdateLayer` would fail with `SEQ_OVERFLOW`. The
    /// defensive check exists exactly to prevent that outcome; this test
    /// pins the boundary so a future `>` / `>=` slip would surface here.
    ///
    /// The boundary is driven through the REAL formula rather than around
    /// it: `max_seq_observed = None` and `snap_seq = 2^53 − 2` make the
    /// restore floor `2^53 − 2`, and the smallest margin `Config::from_env`
    /// accepts (`1`, b.8y5) lands `seeded_seq` on exactly `2^53 − 1`. The
    /// earlier form of this test used `safety_margin = 0` to collapse the
    /// formula to `snap_seq`, which is no longer a reachable configuration
    /// — a boundary pinned at a value production can never produce is not
    /// a boundary.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_checkpoint_seeded_seq_at_safe_int_max_returns_permanent() {
        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);

        // Install a live layer mapping so the SR-2.4 identity guard (step
        // 2) passes — we want the boundary check (step 4) to be the layer
        // that rejects, not the guard. The mapping key must be the one
        // named by the blob's own `(session_id, layer_name)` ("s1" / "l",
        // matching the `meta` below) and its value must equal
        // `object_id.to_string()` for `live_mapping_resolves_blob_identity`
        // to return true.
        let mapping_key = tv.key_schema().layer_mapping("s1", "l");
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key, "424242", None, None, false)
            .await
            .expect("SET layer mapping to make SR-2.4 guard pass");

        // Build a valid reset-delta OMPF so decoding (step 3) also
        // succeeds — only step 4 should reject.
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            ..OwnedDelta::default()
        };
        let ompf = build_reset_ompf(&delta);

        let meta = CheckpointMeta {
            session_id: "s1".to_string(),
            layer_name: "l".to_string(),
            created_by: "u".to_string(),
            server_ts: 123,
            ..CheckpointMeta::default()
        };
        // The smallest margin the config surface accepts (b.8y5 rejects 0).
        let config = test_config(1);

        // restore_floor = max(snap_seq, 0) = 2^53 - 2, so
        // seeded_seq = 2^53 - 2 + 1 = 2^53 - 1 — the exact f64
        // safe-integer ceiling. The Lua fast path rejects
        // `seq >= 2^53 - 1`, so this MUST be rejected at apply time.
        let snap_seq: u64 = 9_007_199_254_740_990;

        let result =
            apply_checkpoint(&layer_store, 424_242, ompf, snap_seq, &meta, None, &config).await;

        match result {
            Err(CheckpointError::Permanent { .. }) => {}
            other => panic!(
                "seeded_seq at f64 safe-integer ceiling must classify Permanent; got {other:?}"
            ),
        }

        // Confirm no `obj:hdr` was written for the target object — the
        // boundary check MUST fire before `write_restored_object` runs.
        let header_key = tv.key_schema().obj_header("424242");
        let exists: bool = tv
            .client()
            .pool
            .exists(&header_key)
            .await
            .expect("EXISTS obj:hdr after boundary rejection");
        assert!(
            !exists,
            "boundary rejection must NOT call write_restored_object; obj:hdr was created"
        );
    }

    // ── SR-3.3 / SR-3.4: the composed-restore core ───────────────────────
    //
    // The four tests below pin `apply_restore`'s Epic-4 surface. They are
    // deliberately in-file and minimal — the composed round-trip's fidelity
    // proof is the composition engine's own lane, and the end-to-end restore
    // matrix is Task 6's.

    /// A `("s1", "l")`-addressed meta whose identity the tests install a live
    /// mapping for, so the SR-2.4 guard passes and later steps are reached.
    pub(crate) fn composed_meta() -> CheckpointMeta {
        CheckpointMeta {
            session_id: "s1".to_string(),
            layer_name: "l".to_string(),
            created_by: "u".to_string(),
            server_ts: 123,
            ..CheckpointMeta::default()
        }
    }

    /// A composed state carrying nothing but the target `object_id` — the
    /// content is irrelevant to every assertion below, which are all about
    /// ordering and the returned outcome.
    fn composed_state(object_id: u64) -> ObjectState {
        ObjectState {
            object_id,
            // Deliberately non-zero: the `Composed` arm MUST zero this before
            // the seq advance (a replayed reset leaves a pre-crash value here).
            first_version: 4_242,
            sequence_num: 77,
            ..ObjectState::default()
        }
    }

    pub(crate) async fn install_mapping(tv: &TestValkey, object_id: u64) {
        let mapping_key = tv.key_schema().layer_mapping("s1", "l");
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key, object_id.to_string(), None, None, false)
            .await
            .expect("SET layer mapping so the SR-2.4 identity guard passes");
    }

    /// The C1 fail-closed guard: a `Composed` payload that references external
    /// values with **no** `ExtPrestore` wired MUST return `Permanent` and write
    /// nothing.
    ///
    /// This is the arm SR-3.4 exists for. Skipping the pre-store would
    /// materialize a complete-looking object whose tail-carried values live
    /// nowhere reachable, and `apply_restore`'s own step-1 structural-header
    /// short-circuit makes
    /// that unrepairable by retry — so "silently skip" is not an option and the
    /// guard must be observable.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_restore_composed_with_ext_values_and_no_ext_store_returns_permanent() {
        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let oid: u64 = 8_101;
        install_mapping(&tv, oid).await;

        let result = apply_restore(
            &layer_store,
            oid,
            RestorePayload::Composed {
                state: composed_state(oid),
                ext_values: vec![([0xab; 32], Bytes::from_static(b"tail-carried-value"))],
            },
            30,
            &composed_meta(),
            None,
            &test_config(1000),
            // The wiring defect under test.
            None,
        )
        .await;

        match result {
            Err(CheckpointError::Permanent { .. }) => {}
            other => panic!(
                "a composed payload with a non-empty ext set and no ExternalValueStore MUST \
                 fail closed as Permanent — never a silent skip; got {other:?}"
            ),
        }

        let header_key = tv.key_schema().obj_header(&oid.to_string());
        let exists: bool = tv
            .client()
            .pool
            .exists(&header_key)
            .await
            .expect("EXISTS obj:hdr after the fail-closed refusal");
        assert!(
            !exists,
            "the fail-closed arm MUST fire before write_restored_object; obj:hdr was created"
        );
    }

    /// The happy composed arm: the ext values land, the object is written, and
    /// the returned [`AppliedCheckpoint`] carries what Task 4's re-base needs.
    ///
    /// Pins four things at once, because they are one commit's worth of
    /// contract: (a) the values are present in the live ext store *and* were
    /// written before the header (asserted by both being present on the success
    /// path — the fail-closed sibling above pins the ordering's other half),
    /// (b) `state.sequence_num == seeded_seq`, (c) `band` equals the
    /// `restore_seq_bump` stamped on the header, and (d) `first_version` was
    /// zeroed despite the payload arriving with a pre-crash value.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_restore_composed_pre_stores_ext_values_and_returns_applied() {
        use crate::storage::external_values::ValkeyExternalValueStore;

        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let ext_store = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone());
        let oid: u64 = 8_102;
        install_mapping(&tv, oid).await;

        let hash = [0xcd; 32];
        let config = test_config(1000);
        let snap_seq = 30u64;

        let outcome = apply_restore(
            &layer_store,
            oid,
            RestorePayload::Composed {
                state: composed_state(oid),
                ext_values: vec![(hash, Bytes::from_static(b"tail-carried-value"))],
            },
            snap_seq,
            &composed_meta(),
            None,
            &config,
            Some(ExtPrestore::new(&ext_store)),
        )
        .await
        .expect("a composed restore with its ext store wired must apply");

        let applied = match outcome {
            ApplyOutcome::Applied(applied) => applied,
            ApplyOutcome::AlreadyPresent => {
                panic!(
                    "no obj:hdr existed, so the step-1 structural-header short-circuit must not \
                     have fired"
                )
            }
        };

        assert_eq!(
            applied.seeded_seq,
            snap_seq + config.checkpoint_seq_safety_margin,
            "seeded_seq is restore_floor + band with the default no-measurement span"
        );
        assert_eq!(applied.band, config.checkpoint_seq_safety_margin);
        assert_eq!(applied.snap_seq, snap_seq);
        assert_eq!(
            applied.state.sequence_num, applied.seeded_seq,
            "the returned state MUST be the one persisted, stamped at seeded_seq"
        );
        assert_eq!(
            applied.state.first_version, 0,
            "the Composed arm MUST zero first_version — a replayed reset leaves a pre-crash \
             value there that has no relation to seeded_seq"
        );

        // The pre-stored value is readable through the live ext store.
        let loaded = ext_store
            .load_values(&[&hash])
            .await
            .expect("load_values after the SR-3.4 pre-store");
        assert_eq!(
            loaded.len(),
            1,
            "the tail-carried external value MUST be present in the live ext store after the \
             restore; got {loaded:?}"
        );

        // And the header carries the guard band the outcome reports.
        let bump: Option<String> = tv
            .client()
            .pool
            .hget(
                &tv.key_schema().obj_header(&oid.to_string()),
                "restore_seq_bump",
            )
            .await
            .expect("HGET restore_seq_bump");
        assert_eq!(
            bump.as_deref(),
            Some(applied.band.to_string().as_str()),
            "AppliedCheckpoint::band MUST equal the restore_seq_bump stamped on the cursor"
        );
    }

    /// The three versions a diff builder FILTERS on — `Section::last_version`,
    /// `Field::version`, `TimeSample::version` — must read exactly
    /// `seeded_seq` in the restored state, with no arm-dependent survivors,
    /// while every `Section::first_version` must still read the value the
    /// decoder / composition engine produced.
    ///
    /// `expected_first_versions` maps `section_id → the pre-restore
    /// first_version`, captured by the caller before `apply_restore` ran. Its
    /// key set must match the restored state's exactly, so a section that
    /// silently appears or vanishes fails here rather than going unchecked.
    fn assert_filtered_versions_flattened_preserving_first(
        state: &ObjectState,
        seeded_seq: u64,
        expected_first_versions: &HashMap<u64, u64>,
        arm: &str,
    ) {
        // Non-vacuity: an arm whose fixture carries no entries of a kind would
        // pass the loops below without proving anything about that kind.
        assert!(
            !state.sections.is_empty(),
            "{arm}: the fixture must carry sections"
        );
        assert!(
            !state.fields.is_empty(),
            "{arm}: the fixture must carry fields"
        );
        assert!(
            !state.time_samples.is_empty(),
            "{arm}: the fixture must carry time samples"
        );
        let restored_ids: HashSet<u64> = state.sections.keys().copied().collect();
        let expected_ids: HashSet<u64> = expected_first_versions.keys().copied().collect();
        assert_eq!(
            restored_ids, expected_ids,
            "{arm}: the captured pre-restore first_version map must cover exactly the \
             restored section set"
        );

        for (id, section) in &state.sections {
            assert_eq!(
                section.last_version, seeded_seq,
                "{arm}: section {id} last_version must be re-stamped at seeded_seq"
            );
            let expected_first = expected_first_versions[id];
            assert_eq!(
                section.first_version, expected_first,
                "{arm}: section {id} first_version must be PRESERVED at its pre-restore \
                 value — `plan_faithful_coalesced`'s abandoned-prim retraction arm fires on \
                 `first_version <= have_version`, so raising it strands orphan sections \
                 permanently unretractable"
            );
            assert!(
                section.first_version <= section.last_version,
                "{arm}: section {id} must keep first_version <= last_version"
            );
        }
        for (key, field) in &state.fields {
            assert_eq!(
                field.version, seeded_seq,
                "{arm}: field {key:?} version must be re-stamped at seeded_seq"
            );
        }
        for (key, sample) in &state.time_samples {
            assert_eq!(
                sample.version, seeded_seq,
                "{arm}: time sample {key:?} version must be re-stamped at seeded_seq"
            );
        }
    }

    /// b.2ti: a restore is ONE write of the WHOLE object at `seeded_seq`, so
    /// every version a diff builder filters on MUST read that value afterwards
    /// — on both payload arms — while `Section::first_version` must survive
    /// untouched.
    ///
    /// The restore preserves `object_id` (the SR-2.4 identity guard), so a
    /// subscriber that survives it does not re-bootstrap: it reconnects onto
    /// the incremental path, where `src/domain/diff.rs` emits only entries with
    /// `version > have_version` (`plan_faithful_coalesced` also restates the
    /// other children of a parent with such a child, and nothing else). Any
    /// stamp below the subscriber's tracked version withholds the restored
    /// content from exactly the subscribers the
    /// restore exists to correct. The two pre-restore shapes are both wrong in
    /// that way and both asserted non-vacuously below: the base decode leaves
    /// everything at `version = 1`, and a composed lineage leaves base-era
    /// entries below its tail-era ones.
    ///
    /// The `first_version` half is the other direction of the same argument:
    /// `plan_faithful_coalesced`'s abandoned-prim arm keys on `first_version <=
    /// have_version`, so flattening it would make the orphan sections a
    /// restored base carries permanently unretractable. The `Composed` arm's
    /// fixture therefore carries a tail-created section whose `first_version`
    /// sits in the tail range, to pin that a non-base-era value is preserved
    /// too rather than merely "left at 1".
    ///
    /// Also pins the two neighbouring header columns the re-stamp must not
    /// disturb: `sequence_num == seeded_seq` and `first_version == 0`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_restore_flattens_diff_filtered_versions_at_seeded_seq_but_preserves_section_first_version(
    ) {
        use crate::domain::types::{Field, FieldValue, Section, TimeSample as DomainTimeSample};
        use ordered_float::OrderedFloat;

        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let config = test_config(1000);
        // The lineage head S, shared by both arms. It is the `Composed` arm's
        // OWN head (`state.sequence_num`, below) rather than some smaller
        // number, because that is what production passes: `snap_seq` reaches
        // `apply_restore` as the recovery path's `head_seq`, which is
        // cross-checked equal to `composed.composed_seq == state.sequence_num`.
        // Understating it here would let the 1000-margin dominate the tail-era
        // versions on its own, so the `first_version <= last_version`
        // assertion would witness the margin rather than the real
        // `seeded_seq > S >= first_version` argument.
        let snap_seq = 77u64;
        // `max_seq_observed = None` and no measured span, so
        // restore_floor = snap_seq and band = the configured margin.
        let expected_seeded = snap_seq + config.checkpoint_seq_safety_margin;

        // ── Arm 1: RestorePayload::Ompf (the base-only restore at HEAD) ────
        let oid_ompf: u64 = 8_105;
        install_mapping(&tv, oid_ompf).await;

        let delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: 7,
                section_name: "alpha".to_string(),
                section_type: 1,
            })],
            set_fields: vec![SetField {
                section_id: 7,
                key_name: "k".to_string(),
                value_or_ext_hash: vec![0xde, 0xad, 0xbe, 0xef],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            time_samples: vec![TimeSample {
                section_id: 7,
                time: 1.5,
                value_or_ext_hash: vec![0x01, 0x02],
                ext_value_size: 0,
                ext_value_index: 0,
            }],
            source_format: "usda".to_string(),
        };
        let ompf = build_reset_ompf(&delta);

        // Non-vacuity for this arm: the decoder alone stamps everything at
        // `1` — the exact value the `version > have_version` filter hides from
        // every reconnector — and `1 != expected_seeded`. The same decode
        // supplies the `first_version` baseline the restore must preserve,
        // taken from the decoder rather than restated by hand so the
        // preservation claim cannot drift from what `apply_restore` decodes.
        let decoded = decode_checkpoint_ompf(oid_ompf, &ompf).expect("decode the arm-1 fixture");
        assert_eq!(
            decoded.sections.get(&7).expect("section 7").first_version,
            1,
            "the pre-restore decode must stamp the base era at 1, or this arm proves nothing"
        );
        assert_ne!(
            1, expected_seeded,
            "the fixture's pre-restore stamp must differ from seeded_seq"
        );
        let ompf_first_versions: HashMap<u64, u64> = decoded
            .sections
            .iter()
            .map(|(id, section)| (*id, section.first_version))
            .collect();

        let outcome = apply_restore(
            &layer_store,
            oid_ompf,
            RestorePayload::Ompf(ompf),
            snap_seq,
            &composed_meta(),
            None,
            &config,
            None,
        )
        .await
        .expect("the Ompf arm must apply");
        let applied = match outcome {
            ApplyOutcome::Applied(applied) => applied,
            ApplyOutcome::AlreadyPresent => panic!("no obj:hdr existed for the Ompf arm"),
        };
        assert_eq!(applied.seeded_seq, expected_seeded);
        assert_eq!(
            applied.state.sequence_num, applied.seeded_seq,
            "Ompf arm: the returned state is the one persisted, stamped at seeded_seq"
        );
        assert_eq!(
            applied.state.first_version, 0,
            "Ompf arm: `first_version` stays 0 — no diff builder reads it, and the first \
             post-recovery write populates it"
        );
        assert_filtered_versions_flattened_preserving_first(
            &applied.state,
            applied.seeded_seq,
            &ompf_first_versions,
            "Ompf arm",
        );

        // ── Arm 2: RestorePayload::Composed (base era + tail era) ──────────
        let oid_composed: u64 = 8_106;
        install_mapping(&tv, oid_composed).await;

        let mut state = ObjectState {
            object_id: oid_composed,
            // Deliberately non-zero: a replayed reset leaves a pre-crash value.
            first_version: 4_242,
            // == `snap_seq`, as production guarantees: the composed cursor IS
            // the state's own sequence, and it is what travels as `head_seq`.
            sequence_num: snap_seq,
            ..ObjectState::default()
        };
        state.sections.insert(
            1,
            Section {
                section_id: 1,
                parent_id: 0,
                name: "base-era".to_string(),
                section_type: 0,
                section_order: 0,
                first_version: 1,
                last_version: 1,
            },
        );
        state.sections.insert(
            2,
            Section {
                section_id: 2,
                parent_id: 0,
                name: "tail-era".to_string(),
                section_type: 0,
                section_order: 1,
                first_version: 41,
                last_version: 43,
            },
        );
        state.children.insert(0, vec![1, 2]);
        state.fields.insert(
            (1, "base_k".to_string()),
            Field {
                section_id: 1,
                key_name: "base_k".to_string(),
                value: FieldValue::Inline(vec![0x01]),
                set_order: 1,
                authoring_set_order: 0,
                version: 1,
            },
        );
        state.fields.insert(
            (2, "tail_k".to_string()),
            Field {
                section_id: 2,
                key_name: "tail_k".to_string(),
                value: FieldValue::Inline(vec![0x02]),
                set_order: 2,
                authoring_set_order: 0,
                version: 43,
            },
        );
        state.time_samples.insert(
            (1, OrderedFloat(1.0)),
            DomainTimeSample {
                section_id: 1,
                time: OrderedFloat(1.0),
                value: FieldValue::Inline(vec![0x03]),
                version: 1,
            },
        );
        state.time_samples.insert(
            (2, OrderedFloat(2.0)),
            DomainTimeSample {
                section_id: 2,
                time: OrderedFloat(2.0),
                value: FieldValue::Inline(vec![0x04]),
                version: 42,
            },
        );

        // Non-vacuity for this arm: the composed input carries two distinct
        // eras, neither at `seeded_seq`. Re-stamping only the base era (the
        // rejected alternative) would leave the tail era at 41..=43 and land
        // the base ABOVE it — this loop's `assert_ne` is what makes the
        // post-condition a flattening claim rather than a bump claim. Every
        // entry sits at or below the head `snap_seq`, as composition
        // guarantees; the 43..77 gap is the §8.4 empty commits, which advance
        // the cursor without stamping an entry.
        for section in state.sections.values() {
            assert!(section.last_version <= snap_seq);
            assert_ne!(section.first_version, expected_seeded);
            assert_ne!(section.last_version, expected_seeded);
        }
        for field in state.fields.values() {
            assert_ne!(field.version, expected_seeded);
        }
        for sample in state.time_samples.values() {
            assert_ne!(sample.version, expected_seeded);
        }

        // The `first_version` baseline the restore must preserve, captured
        // before `state` is moved into the payload. Section 2 is tail-created,
        // so its `first_version` sits in the tail range rather than at the
        // base era's `1` — asserting it survives pins PRESERVATION, not merely
        // "everything happens to already be 1".
        let composed_first_versions: HashMap<u64, u64> = state
            .sections
            .iter()
            .map(|(id, section)| (*id, section.first_version))
            .collect();
        assert!(
            composed_first_versions.values().any(|v| *v > 1),
            "the Composed fixture must carry a tail-created section whose first_version is \
             above the base era, or the preservation claim is untestable on this arm"
        );

        let outcome = apply_restore(
            &layer_store,
            oid_composed,
            RestorePayload::Composed {
                state,
                ext_values: Vec::new(),
            },
            snap_seq,
            &composed_meta(),
            None,
            &config,
            None,
        )
        .await
        .expect("the Composed arm must apply");
        let applied = match outcome {
            ApplyOutcome::Applied(applied) => applied,
            ApplyOutcome::AlreadyPresent => panic!("no obj:hdr existed for the Composed arm"),
        };
        assert_eq!(applied.seeded_seq, expected_seeded);
        assert_eq!(
            applied.state.sequence_num, applied.seeded_seq,
            "Composed arm: the returned state is the one persisted, stamped at seeded_seq"
        );
        assert_eq!(
            applied.state.first_version, 0,
            "Composed arm: `first_version` stays 0"
        );
        assert_filtered_versions_flattened_preserving_first(
            &applied.state,
            applied.seeded_seq,
            &composed_first_versions,
            "Composed arm",
        );
    }

    /// The step-1 short-circuit reports [`ApplyOutcome::AlreadyPresent`],
    /// which is the arm Task 4 must NOT re-base on.
    ///
    /// b.f4t: the fixture header must be structurally COMPLETE. A `seq`-only
    /// header now falls through to the restore
    /// (`apply_checkpoint_partial_header_missing_created_by_is_restored`), so
    /// seeding one here would assert the opposite of the name.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_restore_structurally_complete_header_reports_already_present() {
        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let oid: u64 = 8_103;

        seed_complete_header(&tv, oid, "100").await;

        let outcome = apply_restore(
            &layer_store,
            oid,
            // Deliberately a payload that would fail closed if step 4b ran.
            RestorePayload::Composed {
                state: composed_state(oid),
                ext_values: vec![([0xef; 32], Bytes::from_static(b"never-stored"))],
            },
            30,
            &composed_meta(),
            None,
            &test_config(1000),
            None,
        )
        .await
        .expect("the step-1 short-circuit must precede every other step");

        assert!(
            matches!(outcome, ApplyOutcome::AlreadyPresent),
            "a present, structurally-complete obj:hdr MUST report AlreadyPresent, not Applied"
        );
    }

    /// The SR-3.4 seam: an error injected at
    /// [`RESTORE_SEAM_AFTER_EXT_PRESTORE`] returns **after** the ext pre-store
    /// landed and **before** `write_restored_object` ran.
    ///
    /// That is precisely the crash window Task 6's AC-8 matrix drives: the
    /// values are durable, the object is not, nothing partial is observable,
    /// and the retry converges because the pre-store is content-idempotent.
    #[cfg(feature = "test-checkpoint-fault-injection")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn apply_restore_seam_fires_after_ext_prestore_and_before_write_restored_object() {
        use crate::checkpoint::adapters::CheckpointFaultInjector;
        use crate::storage::external_values::ValkeyExternalValueStore;

        struct SeamInjector;
        impl CheckpointFaultInjector for SeamInjector {
            fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
                if op == RESTORE_SEAM_AFTER_EXT_PRESTORE {
                    return Err(CheckpointError::Transient {
                        source: "injected at the SR-3.4 restore seam".into(),
                    });
                }
                Ok(())
            }
        }

        let tv = TestValkey::setup().await;
        let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
        let ext_store = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone());
        let oid: u64 = 8_104;
        install_mapping(&tv, oid).await;

        let hash = [0x5a; 32];
        let injector = SeamInjector;
        let result = apply_restore(
            &layer_store,
            oid,
            RestorePayload::Composed {
                state: composed_state(oid),
                ext_values: vec![(hash, Bytes::from_static(b"durable-before-the-crash"))],
            },
            30,
            &composed_meta(),
            None,
            &test_config(1000),
            Some(ExtPrestore::new(&ext_store).with_fault_injector(&injector)),
        )
        .await;

        assert!(
            matches!(result, Err(CheckpointError::Transient { .. })),
            "the injected seam error MUST return verbatim as the injector classified it; \
             got {result:?}"
        );

        // BEFORE the write: no object.
        let exists: bool = tv
            .client()
            .pool
            .exists(&tv.key_schema().obj_header(&oid.to_string()))
            .await
            .expect("EXISTS obj:hdr after the seam fired");
        assert!(
            !exists,
            "the seam MUST fire before write_restored_object — no partially-applied object may be \
             observable (SR-3.3 / SR-3.5)"
        );

        // AFTER the pre-store: the values are durable, so the retry converges.
        let loaded = ext_store
            .load_values(&[&hash])
            .await
            .expect("load_values after the seam fired");
        assert_eq!(
            loaded.len(),
            1,
            "the seam MUST fire AFTER store_values returned Ok; an orphaned \
             content-addressed value is existing GC territory, an absent one would mean the \
             seam sat on the wrong side of the pre-store"
        );
    }
}
