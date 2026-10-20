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

// Error types and gRPC status mapping.

use tonic::Status;

use crate::domain::delta::DeltaError;
use crate::domain::diff::DiffError;

/// Error during external value resolution (read path).
///
/// Both variants indicate data integrity issues and map to gRPC INTERNAL.
/// Data keys are content-addressed (SDD §11.1), so there is no secondary
/// `value_id` — the hash hex uniquely names both meta and data.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ExternalValueError {
    #[error("external value meta not found for hash {hash_hex}")]
    MetaNotFound { hash_hex: String },

    #[error("external value data not found for hash {hash_hex}")]
    DataNotFound { hash_hex: String },
}

impl From<ExternalValueError> for ServiceError {
    fn from(e: ExternalValueError) -> Self {
        ServiceError::Internal(e.to_string())
    }
}

impl From<DeltaError> for ServiceError {
    fn from(e: DeltaError) -> Self {
        // b.fd9: the pre-fix arm `DeltaError::Oodf(oodf) => ServiceError::from(oodf)`
        // is removed because the `Oodf` variant was never constructed by
        // domain code — gRPC handlers call `parse_delta` and map its
        // `OodfError` straight via `ServiceError::from(OodfError)` (see
        // `src/grpc/layer.rs`), bypassing `DeltaError` entirely.
        match e {
            // b.qw6 item 2: `SetOrderOverflow` joins `SequenceOverflow` in
            // the Internal bucket because the live-trigger condition is
            // identical — a u64 cursor reached the Lua f64 integer-precision
            // ceiling (2^53 - 1), beyond which no retry can help and no
            // input can avoid (`delta_set_order = 0` would still fail if
            // `next_set_order` already hit the ceiling). The Lua fast path
            // and the slow path both surface this as Internal so the
            // backend toggle (`LIVE_EDIT_UPDATE_LAYER_BACKEND`) yields
            // identical wire status at the boundary — AC #2 parity. Pre-fix
            // this routed to InvalidArgument with a u64::MAX-based message
            // that the runtime could never actually reach (the Lua ceiling
            // tripped first when crossed via the fast path); the message
            // also misled operators into thinking the *client's* delta was
            // at fault, when the root cause is a long-lived object that
            // accumulated 2^53 set_order advancements.
            DeltaError::InconsistentState { .. }
            | DeltaError::SequenceOverflow
            | DeltaError::SetOrderOverflow { .. } => {
                ServiceError::Internal(format!("delta error: {e}"))
            }
            // The two size refusals carry their reason (SDD §18.1): built by
            // the same functions the parser's refusals are, so a client sees
            // one refusal whichever check caught the delta.
            DeltaError::TooManyEntries {
                struct_commands,
                set_fields,
                time_samples,
                max_entries,
            } => ServiceError::delta_too_large(
                struct_commands,
                set_fields,
                time_samples,
                max_entries,
            ),
            DeltaError::ChildrenListTooLong { limit, actual } => {
                ServiceError::children_list_too_long(limit, actual)
            }
            DeltaError::InvalidExternalHashLength { .. }
            | DeltaError::DuplicateSection { .. }
            | DeltaError::DeltaBoundsExceeded { .. }
            | DeltaError::NameContainsControlChar { .. }
            | DeltaError::TimeSampleNonFiniteTime { .. }
            | DeltaError::TimeSampleNegativeZeroTime { .. }
            | DeltaError::SelfParenting { .. }
            | DeltaError::DuplicateSiblingName { .. }
            | DeltaError::SectionTreeTooDeep { .. } => {
                ServiceError::InvalidArgument(format!("delta error: {e}"))
            }
        }
    }
}

impl From<DiffError> for ServiceError {
    fn from(e: DiffError) -> Self {
        match e {
            // A section-tree depth breach is a malformed-input condition: the
            // stored state that produced it was either built from a malicious
            // delta (producer bug) or has been corrupted post-write. Either way
            // the subscriber cannot be served — surface as an internal error
            // so the client retries against a healthy replica rather than
            // concluding "my layer is invalid".
            DiffError::SectionTreeTooDeep { .. } => {
                ServiceError::Internal(format!("diff error: {e}"))
            }
        }
    }
}

/// Map an [`AuthError`] from an external auth-backend call into the
/// service-level error code (b.ohw H-grpc-1).
///
/// Transient outages (`Timeout`, `Unreachable`) become
/// `ServiceError::Unavailable` so standard gRPC client retry policies kick in
/// — `Code::Unavailable` (14) is retriable, `Code::Internal` (13) is not.
/// Protocol-level / contract violations (`InvalidResponse`) keep the
/// `Internal` mapping because they signal a real bug, not a recoverable blip.
/// `UserNotFound` reaches this mapping only with authorization off: with it
/// on, the userinfo fallback decorator turns that answer into a claim-derived
/// name before any handler sees it. With nothing to fall back to, it ends the
/// RPC `Internal` as it always has. It shares the `InvalidResponse` arm so the
/// `"auth backend: {e}"` message stays one site.
///
/// `kind` is a short label (e.g. `"permissions service"` /
/// `"userinfo service"`) prefixed onto the message so operators can tell
/// which backend produced the error from logs alone.
impl From<crate::auth::AuthError> for ServiceError {
    fn from(e: crate::auth::AuthError) -> Self {
        use crate::auth::AuthError;
        match e {
            AuthError::Timeout | AuthError::Unreachable => {
                ServiceError::Unavailable(format!("auth backend: {e}"))
            }
            AuthError::InvalidResponse(_) | AuthError::UserNotFound => {
                ServiceError::Internal(format!("auth backend: {e}"))
            }
        }
    }
}

/// Checkpoint-recovery error classification (b.33e).
///
/// `RecoveryOutcome::Error` / `SessionRecoveryOutcome::Error` carry the
/// classified [`crate::checkpoint::store::CheckpointError`] verbatim precisely so the gRPC handlers can
/// branch on its kind — see the outcome contract in
/// `src/checkpoint/recovery.rs` (`Error(_)`: "The caller MAY retry later
/// (transient) or SHOULD surface an internal error to its client
/// (permanent)"). These two associated functions are the ONE place that
/// decision is made, so §18.1's transient-error transcript stays in this file
/// alongside the `AuthError` mapping above rather than being re-derived at each
/// of the six handler arms that consume it.
impl ServiceError {
    /// SDD §18.1: `Transient` is the ONLY retriable checkpoint-recovery class.
    ///
    /// `NotFound` / `Conflict` / `NotImplemented` reaching an `Error` arm means
    /// an inconsistent store or contract drift — `CheckpointStore::get_*` must
    /// answer `Ok(None)` for plain absence, and absence is routed to
    /// `NotRecoverable` well before this point. Retrying cannot fix any of
    /// them, so they are non-retriable.
    ///
    /// Exhaustive on purpose — a new
    /// [`crate::checkpoint::store::CheckpointError`] variant must be a
    /// compile error here, at the one classification site, not a silent
    /// default at six call sites.
    pub fn recovery_error_is_retriable(err: &crate::checkpoint::store::CheckpointError) -> bool {
        use crate::checkpoint::store::CheckpointError;
        match err {
            CheckpointError::Transient { .. } => true,
            CheckpointError::Permanent { .. }
            | CheckpointError::NotFound
            | CheckpointError::Conflict
            | CheckpointError::NotImplemented { .. } => false,
        }
    }

    /// Wrap `message` in the `ServiceError` variant matching `err`'s
    /// retriability: retriable → [`ServiceError::Unavailable`] (gRPC
    /// `UNAVAILABLE`, 14, retriable), otherwise [`ServiceError::Internal`]
    /// (gRPC `INTERNAL`, 13, non-retriable — logged once at ERROR with the full
    /// message (see [`InternalFault`]), while the wire message is redacted).
    ///
    /// Each call site supplies its own `message` so the existing operator-facing
    /// prefixes (`"checkpoint recovery failed for object …"` on the layer path,
    /// `"layer-state recovery failed for object …"` on the subscribe path) stay
    /// byte-identical.
    pub fn from_recovery_error(
        err: &crate::checkpoint::store::CheckpointError,
        message: String,
    ) -> Self {
        if Self::recovery_error_is_retriable(err) {
            ServiceError::Unavailable(message)
        } else {
            ServiceError::Internal(message)
        }
    }
}

impl From<oodf_ffi::OodfError> for ServiceError {
    fn from(e: oodf_ffi::OodfError) -> Self {
        use oodf_ffi::OodfError;
        match e {
            OodfError::EmptyInput
            | OodfError::TooShort
            | OodfError::BadMagic
            | OodfError::VerifierFailed(_)
            | OodfError::UnknownCommandType(_)
            | OodfError::UnknownCommandTagAt { .. }
            | OodfError::InvalidUtf8(_) => {
                ServiceError::InvalidArgument(format!("OODF error: {e}"))
            }
            // The parser's two size refusals, answered exactly as the domain
            // check answers the same delta (`From<DeltaError>` above): the
            // parser's children-list cap is above the domain's, so a list it
            // refuses is over the domain's limit too, and is reported against
            // that limit.
            OodfError::TooManyTables {
                struct_commands,
                set_fields,
                time_samples,
                max_entries,
            } => ServiceError::delta_too_large(struct_commands, set_fields, time_samples, max_entries),
            OodfError::ChildrenListTooLong { len } => ServiceError::children_list_too_long(
                crate::domain::limits::MAX_CHILDREN_LIST_LEN as u64,
                len,
            ),
            OodfError::AllocationFailed
            | OodfError::InternalError(_)
            | OodfError::Ffi(_)
            // `BuildUnknownCommandTag` can only arise from a bridge caller
            // smuggling a raw out-of-range CommandTag through the cxx layer;
            // it cannot be produced by a user-supplied gRPC payload (which
            // crosses through `parse_delta`, not `build_delta`). Map it to
            // `Internal` because the only reachable origin is a Rust-side
            // contract break, not a malformed wire payload (b.vvs item 8).
            | OodfError::BuildUnknownCommandTag(_) => {
                ServiceError::Internal(format!("OODF error: {e}"))
            }
        }
    }
}

/// Which structural keyspace condition refused an object materialization.
///
/// Both are permanent by construction: only an operator can clear them. See
/// [`ServiceError::PermanentKeyspaceRefusal`] for why the kind is carried as a
/// typed discriminant rather than recovered from the message text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyspaceRefusal {
    /// A key the materialization would write, or the restore fence must read,
    /// holds a foreign Valkey type. Remediation: remove the key.
    WrongType,
    /// The per-object stream already holds an entry at or above the ID the
    /// materialization would write. Remediation: clear the stream.
    ///
    /// Declared, and constructed by no production site today (b.6de sub-task
    /// 2). **Three** emitters raise this refusal, all still as untyped
    /// [`ServiceError::Internal`] carrying `STREAM_ID_COLLISION`: the
    /// create-path stream `EXISTS` guard in `LayerStore::write_object`, the
    /// txn update path's `probe_stream_id_collision` arm, and the Lua update
    /// path's `STREAM_ID_COLLISION` arm (all `src/storage/layer.rs`; the
    /// authoritative list is the "# What is NOT converted yet" section of
    /// [`ServiceError::PermanentKeyspaceRefusal`] below). Converting either
    /// update-path arm alone would split the §7.5 lua-vs-txn parity pair, and
    /// all three are pinned by integration-test helpers outside this lane, so
    /// **b.p4t** carries them as one change. The variant is declared
    /// now so the checkpoint lane's classifier can be kind-agnostic, and so
    /// re-pointing those emitters later is a mechanical change rather than a
    /// re-litigation of the variant's shape.
    StreamIdCollision,
}

/// Top-level service error. Every variant maps to a gRPC status code.
#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// [`ServiceError::InvalidArgument`] carrying a refusal detail: the same
    /// `INVALID_ARGUMENT` code and the same message, plus a [`RefusalDetail`]
    /// (`google.rpc.ErrorInfo`, no `google.rpc.RetryInfo`).
    ///
    /// Built by three constructors:
    /// - [`ServiceError::delta_too_large`] and
    ///   [`ServiceError::children_list_too_long`]: a delta refused for its
    ///   size ([`DELTA_TOO_LARGE`](refusal_reason::DELTA_TOO_LARGE),
    ///   [`CHILDREN_LIST_TOO_LONG`](refusal_reason::CHILDREN_LIST_TOO_LONG)),
    ///   on any `CreateLayer` / `UpdateLayer`, chained or not;
    /// - [`ServiceError::invalid_update`], at the `UpdateLayer` handler's one
    ///   choke point for a chained call (`UpdateLayerRequest.chained_header`):
    ///   every `INVALID_ARGUMENT` answering a chained call carries
    ///   [`INVALID_UPDATE`](refusal_reason::INVALID_UPDATE) unless it already
    ///   carries one of the size reasons.
    ///
    /// Any other `INVALID_ARGUMENT` of an unchained call stays a plain
    /// [`ServiceError::InvalidArgument`]. `Display` equals `InvalidArgument`'s.
    #[error("invalid argument: {message}")]
    InvalidArgumentWithDetail {
        /// The wire message, as [`ServiceError::InvalidArgument`] would carry
        /// it.
        message: String,
        /// The refusal detail the status carries. Boxed, as
        /// [`ServiceError::UnavailableWithDetail`]'s is.
        detail: Box<RefusalDetail>,
    },

    #[error("not found: {0}")]
    NotFound(String),

    #[error("already exists: {0}")]
    AlreadyExists(String),

    #[error("permission denied")]
    PermissionDenied,

    #[error("unauthenticated")]
    Unauthenticated,

    #[error("failed precondition: {0}")]
    FailedPrecondition(String),

    /// In-process retry signal consumed by
    /// [`crate::storage::retry::retry_aborted_with_jitter`]. Never reaches the
    /// wire as `ABORTED`: when it escapes a handler (single-attempt site or
    /// loop exhaustion) it surfaces as `UNAVAILABLE`, with its message
    /// verbatim — see the `From<ServiceError> for tonic::Status` arm below.
    /// (DeleteSession's cascade alone rewraps a layer's `Aborted` first, as an
    /// `Unavailable` carrying progress counts; the wire code is the same.)
    #[error("aborted: {0}")]
    Aborted(String),

    /// A chained `UpdateLayer` call (`UpdateLayerRequest.chained_header`)
    /// refused because the latest operation recorded for its
    /// `(object_id, client_id)` is not its predecessor: gRPC `ABORTED`, with
    /// a [`RefusalDetail`] of reason
    /// [`PREDECESSOR_NOT_COMMITTED`](refusal_reason::PREDECESSOR_NOT_COMMITTED),
    /// metadata [`LAST_RECORD`](refusal_metadata::LAST_RECORD) (and
    /// [`LAST_OPERATION_ID`](refusal_metadata::LAST_OPERATION_ID) with
    /// [`OPERATION`](refusal_metadata::OPERATION)) and no `RetryInfo`. The
    /// call applied nothing.
    ///
    /// The service's only `ABORTED` (SDD §18.1): built only by
    /// [`ServiceError::predecessor_not_committed`], which has two producers,
    /// both of which promise that the call applied nothing: the chain gate
    /// (`storage::chain_gate`), before any commit attempt, and the commit
    /// path's predecessor guard (`LayerStore::update_layer_with_summary`), on
    /// the call's first attempt only — after a counted retry that refusal is
    /// a plain `UNAVAILABLE`, outcome unknown, instead, in storage and on the
    /// handler's recovery re-issue. Never an in-process retry signal:
    /// [`ServiceError::Aborted`] is that, and never reaches the wire as
    /// `ABORTED`.
    #[error("predecessor not committed: {message}")]
    PredecessorNotCommitted {
        /// The wire message.
        message: String,
        /// The refusal detail the status carries. Boxed, as
        /// [`ServiceError::UnavailableWithDetail`]'s is.
        detail: Box<RefusalDetail>,
    },

    /// Mapped to gRPC `CANCELLED`. Use for mid-RPC transport errors that
    /// reflect client/transport disconnects rather than internal faults —
    /// e.g. a `tonic::Streaming::message()` failure after the client has
    /// begun uploading chunks (b.jr8 mediums), or an upload stream that ended
    /// without END_STREAM (reset, lost or ended early), which applies
    /// nothing. Keeps `Internal` reserved for conditions that warrant
    /// operator attention.
    #[error("cancelled: {0}")]
    Cancelled(String),

    /// Mapped to gRPC `UNAVAILABLE`. Used when an upstream dependency
    /// (auth backend, storage backend) is reachable-but-failing, has timed
    /// out, or is otherwise transiently broken. Standard gRPC client retry
    /// policies treat `UNAVAILABLE` as retriable, so callers can re-issue
    /// the RPC during a brief outage instead of bubbling up a fatal error.
    /// Reserve `Internal` for conditions that warrant operator attention
    /// (b.ohw H-grpc-1).
    #[error("unavailable: {0}")]
    Unavailable(String),

    /// [`ServiceError::Unavailable`] carrying a retry hint: the same
    /// `UNAVAILABLE` code and the same message, plus a [`RefusalDetail`]
    /// (`google.rpc.ErrorInfo` + `google.rpc.RetryInfo` in
    /// `grpc-status-details-bin` — every producer gives the detail a delay
    /// range; see [`refusal_status`]).
    ///
    /// Used only by refusals that applied nothing; each listed refusal site
    /// builds it. A plain
    /// [`ServiceError::Unavailable`] is converted into a detailed refusal in
    /// exactly one place: the `CreateLayer` / `UpdateLayer` pre-commit choke
    /// point (`grpc::layer`'s `refused_before_commit`), which gives an
    /// `UNAVAILABLE` status with no detail, raised before the call's commit
    /// started, [`refusal_reason::UPSTREAM_UNAVAILABLE`]
    /// ([`RefusalDetail::upstream_unavailable`]) with its message unchanged.
    /// It works on the outgoing `Status`, not on this enum, so no in-process
    /// consumer ever sees the converted value. `Display` equals
    /// `Unavailable`'s, so every log line and every wrapping message
    /// (DeleteSession's cascade) reads byte for byte as before. Every
    /// in-process consumer that branches on `Unavailable` treats it the same,
    /// except the in-process retry loops, which do not retry it: each of its
    /// producers is a refusal that is not retried in-process.
    #[error("unavailable: {message}")]
    UnavailableWithDetail {
        /// The wire message, as [`ServiceError::Unavailable`] would carry it.
        message: String,
        /// The retry hint the status carries. Boxed so the variant stays as
        /// small as the others: every `Result<_, ServiceError>` carries its
        /// largest variant's size.
        detail: Box<RefusalDetail>,
    },

    #[error("internal error: {0}")]
    Internal(String),

    /// A structural keyspace condition refused an object materialization, and
    /// no retry can clear it — an operator has to remove or re-type a key
    /// first (b.6de sub-task 2).
    ///
    /// Maps to gRPC `INTERNAL` exactly as [`ServiceError::Internal`] does, and
    /// produces the same single ERROR line (see [`InternalFault`]). It is a
    /// *separate* variant only so the checkpoint lane can tell the permanent
    /// refusals apart from the far larger transient `Internal` population
    /// **by type**: `write_object_err_to_checkpoint_err`
    /// (`src/checkpoint/apply.rs`) classifies this one `CheckpointError::Permanent`
    /// and everything else `Transient`, and before this variant existed it did
    /// that with `msg.contains(KEYSPACE_WRONGTYPE_TAG)` — a text match that any
    /// reworded message silently disarms, turning a restore that cannot succeed
    /// back into one the recovery hook re-drives forever.
    ///
    /// # Shape
    ///
    /// **`{ refusal, detail }`, not `{ key, actual, expected }`.** The
    /// key/actual/expected triple exists only *inside* the Lua probe script's
    /// reply string (`<key> holds a <actual>, expected <expected>`); parsing it
    /// back out in Rust would re-introduce, in a worse place, exactly the
    /// text-matching this variant removes. `detail` therefore carries the
    /// emitter's string verbatim — tag prefix included — so every operator-facing
    /// message and every prefix assertion is byte-identical to the pre-variant
    /// `Internal` it replaces, while `refusal` is the part consumers branch on.
    ///
    /// **Not a bespoke `RestoreWriteError`.** The wrong-type refusal is raised
    /// inside `LayerStore::classify_keyspace_probe_error`, which is shared by the
    /// `CreateLayer` and DR-restore materializations and whose return type is
    /// `ServiceError`; a checkpoint-only error type would have to be recovered
    /// from a `ServiceError` at the lane boundary, i.e. by matching text again.
    ///
    /// # What is NOT converted yet
    ///
    /// Five emitters still raise plain [`ServiceError::Internal`] carrying the
    /// same tags — `write_object`'s create-path stream `EXISTS` guard, the Lua
    /// update path's `WRONGTYPE_KEY` arm, the txn update path's `WRONGTYPE_KEY`
    /// arm (`LayerStore::classify_keyspace_probe_error`'s `ProbeSite::Update`
    /// arm, `src/storage/layer.rs` — the direct sibling of the
    /// `ProbeSite::Create` arm that mints this variant, three lines away), and
    /// both update paths' `STREAM_ID_COLLISION` arms. Their conversion is
    /// carried by the **b.p4t** follow-up: each is pinned by integration-test
    /// helpers outside the lane that landed this variant, and `update_layer`'s
    /// lua-vs-txn parity contract (SDD §7.5) is asserted on those sites' error
    /// shape. Consumers must therefore treat this variant as *sufficient*
    /// evidence of a permanent keyspace refusal, never as *necessary*.
    #[error("permanent keyspace refusal: {detail}")]
    PermanentKeyspaceRefusal {
        /// Which of the two structural conditions fired.
        refusal: KeyspaceRefusal,
        /// The emitting site's message, verbatim — including its tag prefix.
        detail: String,
    },
}

impl From<ServiceError> for tonic::Status {
    fn from(e: ServiceError) -> tonic::Status {
        match e {
            ServiceError::InvalidArgument(msg) => Status::invalid_argument(msg),
            ServiceError::InvalidArgumentWithDetail { message, detail } => {
                detail.status(tonic::Code::InvalidArgument, message)
            }
            ServiceError::NotFound(msg) => Status::not_found(msg),
            ServiceError::AlreadyExists(msg) => Status::already_exists(msg),
            ServiceError::PermissionDenied => Status::permission_denied(""),
            ServiceError::Unauthenticated => Status::unauthenticated(""),
            ServiceError::FailedPrecondition(msg) => Status::failed_precondition(msg),
            // `Aborted` never reaches the wire as gRPC `ABORTED`. It is the
            // in-process "retry me" signal; one that escapes a handler — from a
            // single-attempt site returning the in-loop signal, or a retry loop
            // that ran out of attempts — is a transient backend condition, so
            // it goes out as `UNAVAILABLE`, which the Kit client retries (it
            // treats `ABORTED` as terminal). The message travels verbatim.
            // This arm is the only place `Aborted` becomes a wire code; the
            // one handler that rewraps an `Aborted` first is DeleteSession's
            // cascade (`delete_session_inner`, `src/grpc/session.rs`), which
            // turns it into an `Unavailable` carrying progress counts — the
            // same `UNAVAILABLE` on the wire.
            ServiceError::Aborted(msg) => Status::unavailable(msg),
            // The service's one gRPC `ABORTED`: a chained `UpdateLayer`'s
            // predecessor refusal, which only a client that sent
            // `chained_header` can receive (Kit never does). The guard
            // `no_source_file_constructs_a_grpc_aborted_status` allows exactly
            // this arm.
            ServiceError::PredecessorNotCommitted { message, detail } => {
                detail.status(tonic::Code::Aborted, message)
            }
            ServiceError::Cancelled(msg) => Status::cancelled(msg),
            ServiceError::Unavailable(msg) => Status::unavailable(msg),
            ServiceError::UnavailableWithDetail { message, detail } => detail.unavailable(message),
            ServiceError::Internal(msg) => internal_status(msg),
            // Identical treatment to `Internal`: same code, same redacted wire
            // message, same single ERROR line. The typed variant exists for
            // in-process consumers (see its rustdoc), not to change anything a
            // client or a log reader observes.
            ServiceError::PermanentKeyspaceRefusal { detail, .. } => internal_status(detail),
        }
    }
}

/// **Client contract.** The `domain` of the `google.rpc.ErrorInfo` every
/// refusal detail this service attaches carries (see [`refusal_status`]).
///
/// A client reads an `UNAVAILABLE` carrying a refusal detail as follows.
/// `retry_delay` (the `google.rpc.RetryInfo`) is the **minimum** wait before
/// the next attempt of the refused call, already randomized per response.
/// Across consecutive refusals of the same call the client keeps its own
/// backoff and waits max(`retry_delay`, backoff), with no extra jitter on
/// top. An unknown `reason` on such an `UNAVAILABLE` means "retry after that
/// wait". On `CreateLayer` and `UpdateLayer` an `UNAVAILABLE` carrying this
/// `ErrorInfo` means: this call applied nothing; a re-send is safe unless an
/// earlier attempt of the same write is still in doubt — then use
/// `operation_id`. Every `UNAVAILABLE` such a call is refused with before its
/// commit started carries a detail —
/// [`UPSTREAM_UNAVAILABLE`](refusal_reason::UPSTREAM_UNAVAILABLE) when no more
/// specific reason applies — so one a replica of this build answers without it
/// comes from a write whose commit had started, and the client cannot assume
/// it applied nothing. During a rolling upgrade a replica of an earlier build
/// answers some pre-commit refusals without the detail too; read as outcome
/// unknown, that answer is weaker but never wrong (SDD §18.1, *Rolling
/// upgrade*). On every other RPC an `UNAVAILABLE` without the detail keeps its
/// plain meaning. This contract covers `UNAVAILABLE` only; a refusal detail on
/// any other code means what that code's refusal states, and carries no
/// `RetryInfo` — today:
/// - the two refusals of a chained `UpdateLayer`
///   (`UpdateLayerRequest.chained_header`), `ABORTED`
///   [`PREDECESSOR_NOT_COMMITTED`](refusal_reason::PREDECESSOR_NOT_COMMITTED)
///   and `INVALID_ARGUMENT` [`INVALID_UPDATE`](refusal_reason::INVALID_UPDATE),
///   each of which means this call applied nothing;
/// - the two size refusals of `CreateLayer` / `UpdateLayer`, `INVALID_ARGUMENT`
///   [`DELTA_TOO_LARGE`](refusal_reason::DELTA_TOO_LARGE) and
///   [`CHILDREN_LIST_TOO_LONG`](refusal_reason::CHILDREN_LIST_TOO_LONG): this
///   call applied nothing, and no re-send of the same delta can succeed. On
///   `DELTA_TOO_LARGE`, split the content into several smaller deltas (for a new
///   layer, create it with the first and add the rest with `UpdateLayer`; a
///   refused `UpdateLayer` becomes several smaller `UpdateLayer`s); on
///   `CHILDREN_LIST_TOO_LONG`, name at most 100,000 children in one
///   `ReorderChildren`.
pub const REFUSAL_DETAIL_DOMAIN: &str = "nvidia.omniverse.live.v1alpha";

/// **Client contract.** The `google.rpc.ErrorInfo` `reason` of each refusal
/// that carries a refusal detail ([`REFUSAL_DETAIL_DOMAIN`]). Each is a stable
/// name; on an `UNAVAILABLE`, a client treats one it does not know as "retry
/// after max(`retry_delay`, its own backoff)". That rule is the `UNAVAILABLE`
/// contract only: a reason on any other code carries no `RetryInfo`, and
/// promises only what its own entry states.
pub mod refusal_reason {
    /// Graceful shutdown ended the stream, or refused the call, on this
    /// replica. Every `UNAVAILABLE "server shutting down"` carries it.
    pub const SERVER_SHUTTING_DOWN: &str = "SERVER_SHUTTING_DOWN";
    /// `SubscribeReadLayer` admission refused: the replica's concurrent
    /// subscription ceiling, or its admission queue
    /// (metadata [`cause`](super::refusal_metadata::CAUSE)).
    pub const SUBSCRIPTION_CAPACITY: &str = "SUBSCRIPTION_CAPACITY";
    /// `SubscribeReadLayer` ran out of its capacity budget waiting for a
    /// stream reader.
    pub const READER_CAPACITY: &str = "READER_CAPACITY";
    /// `SubscribeReadLayer` ran out of its capacity budget waiting for a
    /// catch-up slot or a snapshot read permit
    /// (metadata [`wait`](super::refusal_metadata::WAIT)), or while the shared
    /// snapshot read it joined waited for catch-up capacity (no `wait`).
    pub const CATCHUP_CAPACITY: &str = "CATCHUP_CAPACITY";
    /// `CreateLayer` / `UpdateLayer` refused by the replica's upload capacity
    /// (metadata [`cause`](super::refusal_metadata::CAUSE)).
    pub const UPLOAD_CAPACITY: &str = "UPLOAD_CAPACITY";
    /// The backing store refused a write the call needed, before applying
    /// anything (metadata [`condition`](super::refusal_metadata::CONDITION)).
    pub const STORE_WRITE_REFUSED: &str = "STORE_WRITE_REFUSED";
    /// `UpdateLayer` reached its cap on waiting for a snapshot fence; nothing
    /// was written.
    pub const SNAPSHOT_FENCE_TIMEOUT: &str = "SNAPSHOT_FENCE_TIMEOUT";
    /// **Client contract.** On `ABORTED`, and only there: a chained
    /// `UpdateLayer` (`UpdateLayerRequest.chained_header`) whose predecessor
    /// was not the latest operation recorded for its `(object_id, client_id)`.
    /// This call applied nothing. Metadata
    /// [`last_record`](super::refusal_metadata::LAST_RECORD), and
    /// [`last_operation_id`](super::refusal_metadata::LAST_OPERATION_ID) when
    /// that is [`operation`](super::refusal_metadata::OPERATION): at the
    /// refusal every write of the chain up to and including
    /// `last_operation_id` had committed and none after it. No `RetryInfo`:
    /// it is not a capacity refusal. A chained refusal after the server
    /// retried the call internally is instead a plain `UNAVAILABLE`, outcome
    /// unknown.
    pub const PREDECESSOR_NOT_COMMITTED: &str = "PREDECESSOR_NOT_COMMITTED";
    /// **Client contract.** On `INVALID_ARGUMENT`, and only there: every
    /// `INVALID_ARGUMENT` answering a chained `UpdateLayer` carries it unless
    /// another reason applies — [`DELTA_TOO_LARGE`] or
    /// [`CHILDREN_LIST_TOO_LONG`], which a chained call carries in its place.
    /// Either way this call applied nothing. No metadata, no `RetryInfo`. A
    /// chained call refused `INVALID_ARGUMENT` without an `ErrorInfo` of
    /// [`REFUSAL_DETAIL_DOMAIN`](super::REFUSAL_DETAIL_DOMAIN) was answered
    /// by a replica that does not implement chained writes, which read no
    /// content and applied nothing.
    pub const INVALID_UPDATE: &str = "INVALID_UPDATE";
    /// **Client contract.** On `INVALID_ARGUMENT`, and only there: a
    /// `CreateLayer` / `UpdateLayer` whose delta carries more entries than one
    /// delta may hold — its struct commands, set fields and time samples
    /// summed are over [`MAX_DELTA_ENTRIES`](crate::domain::limits::MAX_DELTA_ENTRIES)
    /// (99,999). This call applied nothing, and no re-send of the same delta
    /// can succeed: split the content into several smaller deltas. Metadata
    /// [`struct_commands`](super::refusal_metadata::STRUCT_COMMANDS),
    /// [`set_fields`](super::refusal_metadata::SET_FIELDS),
    /// [`time_samples`](super::refusal_metadata::TIME_SAMPLES) and
    /// [`max_entries`](super::refusal_metadata::MAX_ENTRIES), each a decimal
    /// integer. No `RetryInfo`. On a chained call it takes the place of
    /// [`INVALID_UPDATE`].
    pub const DELTA_TOO_LARGE: &str = "DELTA_TOO_LARGE";
    /// **Client contract.** On `INVALID_ARGUMENT`, and only there: a
    /// `CreateLayer` / `UpdateLayer` whose delta carries a `ReorderChildren`
    /// command naming more children than one list may hold
    /// ([`MAX_CHILDREN_LIST_LEN`](crate::domain::limits::MAX_CHILDREN_LIST_LEN),
    /// 100,000). This call applied nothing, and no re-send of the same delta
    /// can succeed: name at most 100,000 children in one `ReorderChildren`.
    /// Metadata [`limit`](super::refusal_metadata::LIMIT) and
    /// [`actual`](super::refusal_metadata::ACTUAL), each a decimal integer.
    /// No `RetryInfo`. On a chained call it takes the place of
    /// [`INVALID_UPDATE`].
    pub const CHILDREN_LIST_TOO_LONG: &str = "CHILDREN_LIST_TOO_LONG";
    /// **Client contract.** On `UNAVAILABLE`: a chained `UpdateLayer` waiting
    /// for its predecessor reached its deadline cap without one successful
    /// read of its writer's record, so the server could not tell whether the
    /// predecessor committed. Nothing was applied; retry after the
    /// `retry_delay`.
    pub const CHAIN_CHECK_UNAVAILABLE: &str = "CHAIN_CHECK_UNAVAILABLE";
    /// `CreateLayer` / `UpdateLayer` carrying external values refused by the
    /// backing store's external-value release budget
    /// (`LIVE_EDIT_EXT_RELEASE_MAX_BYTES`): released external values awaiting
    /// garbage collection stayed at or above it for as long as the call's
    /// deadline allowed. Refused before anything was stored or committed. No
    /// metadata.
    pub const EXT_RELEASE_BUDGET_FULL: &str = "EXT_RELEASE_BUDGET_FULL";
    /// `JoinSession` ended because a join that opted in to client keepalives
    /// (`JoinSessionHeader.client_keepalive`, with a non-zero
    /// `SessionConnected.keepalive_interval_ms`) sent no data for three
    /// keepalive intervals. The participant has already left, and peers saw
    /// one `PeerLeft`; a client that is still alive rejoins.
    pub const CLIENT_KEEPALIVE_TIMEOUT: &str = "CLIENT_KEEPALIVE_TIMEOUT";
    /// **Client contract.** On `UNAVAILABLE`, and only on `CreateLayer` /
    /// `UpdateLayer`: the call was refused before its commit was sent,
    /// because a dependency it needed was unavailable — the backing store not
    /// serving (still loading its data after a restart, busy running a long
    /// script) or failing at the connection level, the auth backend, or the
    /// backup store during a lazy restore. This call applied nothing. No
    /// metadata; `RetryInfo` drawn from
    /// [`UPSTREAM_UNAVAILABLE_RETRY_DELAY`](super::UPSTREAM_UNAVAILABLE_RETRY_DELAY),
    /// a minimum wait — the client keeps its own backoff. A write that fails
    /// after its commit started (retry exhaustion, overtaken, a `CreateLayer`
    /// commit-stage refusal, an outcome-unknown answer after an internal
    /// retry) never carries it.
    pub const UPSTREAM_UNAVAILABLE: &str = "UPSTREAM_UNAVAILABLE";
}

/// **Client contract.** The `google.rpc.ErrorInfo` `metadata` keys a refusal
/// detail may carry, and their values.
pub mod refusal_metadata {
    /// On [`SUBSCRIPTION_CAPACITY`](super::refusal_reason::SUBSCRIPTION_CAPACITY)
    /// and [`UPLOAD_CAPACITY`](super::refusal_reason::UPLOAD_CAPACITY): what
    /// was full — [`SATURATED`], [`WAITER_CAP`] or (uploads only) [`GROWTH`].
    pub const CAUSE: &str = "cause";
    /// Capacity stayed full for the whole admission wait.
    pub const SATURATED: &str = "saturated";
    /// The admission queue is full.
    pub const WAITER_CAP: &str = "waiter_cap";
    /// The upload budget cannot hold the upload's content right now.
    pub const GROWTH: &str = "growth";
    /// On [`CATCHUP_CAPACITY`](super::refusal_reason::CATCHUP_CAPACITY): what
    /// the catch-up waited for — [`SLOT`] or [`READ_PERMIT`]. Absent when the
    /// catch-up waited on a shared snapshot read's capacity waits.
    pub const WAIT: &str = "wait";
    /// A catch-up slot.
    pub const SLOT: &str = "slot";
    /// A snapshot read permit.
    pub const READ_PERMIT: &str = "read_permit";
    /// On [`STORE_WRITE_REFUSED`](super::refusal_reason::STORE_WRITE_REFUSED):
    /// the store's refusal — `out_of_memory`, `misconf`, `readonly` or
    /// `noreplicas` (the `reason` labels of
    /// `live_edit_store_write_refused_total`).
    pub const CONDITION: &str = "condition";
    /// **Client contract.** On
    /// [`PREDECESSOR_NOT_COMMITTED`](super::refusal_reason::PREDECESSOR_NOT_COMMITTED):
    /// what the server found as the latest record for the call's
    /// `(object_id, client_id)` — [`ABSENT`], [`OPERATION`] or
    /// [`NO_OPERATION_ID`]. Always present.
    pub const LAST_RECORD: &str = "last_record";
    /// No record: none was made under this `client_id` on the object, or the
    /// object was restored from a checkpoint, which loses the records.
    pub const ABSENT: &str = "absent";
    /// The latest commit under this `client_id` carried an `operation_id`,
    /// which [`LAST_OPERATION_ID`] holds.
    pub const OPERATION: &str = "operation";
    /// The latest commit under this `client_id` carried no `operation_id`.
    pub const NO_OPERATION_ID: &str = "no_operation_id";
    /// **Client contract.** On
    /// [`PREDECESSOR_NOT_COMMITTED`](super::refusal_reason::PREDECESSOR_NOT_COMMITTED),
    /// present exactly when [`LAST_RECORD`] is [`OPERATION`]: the
    /// `operation_id` of the latest operation recorded for the call's
    /// `(object_id, client_id)`, as the client sent it. Not sensitive: a
    /// caller that can name the object and the `client_id` can already
    /// commit under it.
    pub const LAST_OPERATION_ID: &str = "last_operation_id";
    /// **Client contract.** On
    /// [`DELTA_TOO_LARGE`](super::refusal_reason::DELTA_TOO_LARGE): the
    /// delta's struct commands, a decimal integer. Always present, as are
    /// [`SET_FIELDS`], [`TIME_SAMPLES`] and [`MAX_ENTRIES`].
    pub const STRUCT_COMMANDS: &str = "struct_commands";
    /// **Client contract.** On
    /// [`DELTA_TOO_LARGE`](super::refusal_reason::DELTA_TOO_LARGE): the
    /// delta's set fields, a decimal integer.
    pub const SET_FIELDS: &str = "set_fields";
    /// **Client contract.** On
    /// [`DELTA_TOO_LARGE`](super::refusal_reason::DELTA_TOO_LARGE): the
    /// delta's time samples, a decimal integer.
    pub const TIME_SAMPLES: &str = "time_samples";
    /// **Client contract.** On
    /// [`DELTA_TOO_LARGE`](super::refusal_reason::DELTA_TOO_LARGE): the most
    /// entries one delta may carry, the three kinds summed, a decimal integer.
    pub const MAX_ENTRIES: &str = "max_entries";
    /// **Client contract.** On
    /// [`CHILDREN_LIST_TOO_LONG`](super::refusal_reason::CHILDREN_LIST_TOO_LONG):
    /// the most children one list may name, a decimal integer. Always present,
    /// as is [`ACTUAL`].
    pub const LIMIT: &str = "limit";
    /// **Client contract.** On
    /// [`CHILDREN_LIST_TOO_LONG`](super::refusal_reason::CHILDREN_LIST_TOO_LONG):
    /// the children the refused list names, a decimal integer.
    pub const ACTUAL: &str = "actual";
}

/// The latest record a chained `UpdateLayer` found for its
/// `(object_id, client_id)`, as a
/// [`PREDECESSOR_NOT_COMMITTED`](refusal_reason::PREDECESSOR_NOT_COMMITTED)
/// refusal reports it ([`refusal_metadata::LAST_RECORD`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LastRecord<'a> {
    /// [`refusal_metadata::ABSENT`].
    Absent,
    /// [`refusal_metadata::OPERATION`], with the recorded `operation_id`.
    Operation(&'a str),
    /// [`refusal_metadata::NO_OPERATION_ID`].
    NoOperationId,
}

/// A half-open range `[min, max)` a refusal's `retry_delay` is drawn from,
/// uniformly and afresh for every response, so the clients one event refused
/// do not all come back at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryDelayRange {
    /// The shortest delay a response can carry.
    pub min: std::time::Duration,
    /// The bound every delay stays below. Equal to `min` makes the delay
    /// fixed.
    pub max: std::time::Duration,
}

impl RetryDelayRange {
    /// `[min, max)`.
    pub const fn new(min: std::time::Duration, max: std::time::Duration) -> Self {
        Self { min, max }
    }

    /// One delay from the range, drawn uniformly; `min` when the range is
    /// empty.
    pub fn sample(&self) -> std::time::Duration {
        use rand::Rng;
        let span = self.max.saturating_sub(self.min);
        let span_nanos = u64::try_from(span.as_nanos()).unwrap_or(u64::MAX);
        if span_nanos == 0 {
            return self.min;
        }
        self.min + std::time::Duration::from_nanos(rand::rng().random_range(0..span_nanos))
    }
}

/// The retry delay of [`refusal_reason::SERVER_SHUTTING_DOWN`]: `[0, 100 ms)`.
/// Near zero because the server already paced its shutdown endings, and the
/// client's retry reaches another replica. As a minimum it barely changes a
/// client's first reconnect: max(`retry_delay`, the client's first backoff)
/// stays under 100 ms for a client whose first backoff is that short, and is
/// the client's own first backoff otherwise.
pub const SHUTDOWN_RETRY_DELAY: RetryDelayRange = RetryDelayRange::new(
    std::time::Duration::ZERO,
    std::time::Duration::from_millis(100),
);

/// The retry delay of the capacity refusals
/// ([`refusal_reason::SUBSCRIPTION_CAPACITY`],
/// [`refusal_reason::READER_CAPACITY`], [`refusal_reason::CATCHUP_CAPACITY`],
/// [`refusal_reason::UPLOAD_CAPACITY`],
/// [`refusal_reason::EXT_RELEASE_BUDGET_FULL`]) and of
/// [`refusal_reason::SNAPSHOT_FENCE_TIMEOUT`] and
/// [`refusal_reason::CHAIN_CHECK_UNAVAILABLE`]: `[500 ms, 1500 ms)`.
pub const CAPACITY_RETRY_DELAY: RetryDelayRange = RetryDelayRange::new(
    std::time::Duration::from_millis(500),
    std::time::Duration::from_millis(1500),
);

/// The retry delay of [`refusal_reason::CLIENT_KEEPALIVE_TIMEOUT`]:
/// `[0, 100 ms)`. Near zero because the ending is about this one client, not
/// a load the rejoin could add to: a live client that was ended falsely
/// should rejoin at once. As a minimum it barely changes the rejoin: the
/// client still waits max(`retry_delay`, its own backoff). Its own constant,
/// not [`SHUTDOWN_RETRY_DELAY`], so the two reasons' hints can change apart.
pub const CLIENT_KEEPALIVE_TIMEOUT_RETRY_DELAY: RetryDelayRange = RetryDelayRange::new(
    std::time::Duration::ZERO,
    std::time::Duration::from_millis(100),
);

/// The retry delay of [`refusal_reason::STORE_WRITE_REFUSED`]: `[2 s, 6 s)`.
/// Longer, because a refusing store does not recover in a few hundred
/// milliseconds.
pub const STORE_WRITE_REFUSED_RETRY_DELAY: RetryDelayRange = RetryDelayRange::new(
    std::time::Duration::from_secs(2),
    std::time::Duration::from_secs(6),
);

/// The retry delay of [`refusal_reason::UPSTREAM_UNAVAILABLE`]: `[1 s, 3 s)`.
/// A minimum wait; the client keeps its own backoff and waits
/// max(`retry_delay`, backoff).
///
/// The floor sits above the backing-store client's own reconnect ramp
/// (100 + 200 + 400 ms ≈ 0.7 s, `storage::client::shared_reconnect_policy`) and
/// above the capacity refusals' 0.5 s ([`CAPACITY_RETRY_DELAY`]), so the first
/// retry does not land on a connection still being rebuilt. The 2 s span
/// spreads the clients one store event refused (about 1,000 per deployment)
/// over two seconds; the ceiling stays below [`STORE_WRITE_REFUSED_RETRY_DELAY`]'s
/// 2–6 s, since most of these conditions clear faster than a full store, and
/// keeps the first retry within one health-probe interval (the default
/// `LIVE_EDIT_HEALTH_PROBE_INTERVAL_MS`, 5 s). Its own constant, not
/// [`CAPACITY_RETRY_DELAY`], so the hints can change apart.
pub const UPSTREAM_UNAVAILABLE_RETRY_DELAY: RetryDelayRange = RetryDelayRange::new(
    std::time::Duration::from_secs(1),
    std::time::Duration::from_secs(3),
);

/// A refusal's detail: the `google.rpc.ErrorInfo` reason and metadata, and,
/// when the refusal carries a retry hint, the range its `google.rpc.RetryInfo`
/// delay is drawn from when the status is built.
///
/// The status code is chosen where the status is built ([`Self::status`]), not
/// stored here, so one value type serves a refusal of any code, with or
/// without a `RetryInfo`. The client contract of [`REFUSAL_DETAIL_DOMAIN`]
/// (the minimum wait, "applied nothing" on the uploads) is that of an
/// `UNAVAILABLE` carrying the detail; a detail on another code promises only
/// what that refusal states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefusalDetail {
    /// One of [`refusal_reason`].
    pub reason: &'static str,
    /// `ErrorInfo.metadata`, keys from [`refusal_metadata`].
    pub metadata: Vec<(&'static str, String)>,
    /// What `RetryInfo.retry_delay` is drawn from; `None` puts no
    /// `RetryInfo` on the status.
    pub retry_delay: Option<RetryDelayRange>,
}

impl RefusalDetail {
    /// A detail with `reason`, no metadata, and, when `retry_delay` is
    /// `Some`, a `RetryInfo` delay drawn from it.
    pub fn new(reason: &'static str, retry_delay: Option<RetryDelayRange>) -> Self {
        Self {
            reason,
            metadata: Vec::new(),
            retry_delay,
        }
    }

    /// This detail with one more `ErrorInfo.metadata` entry.
    pub fn with_metadata(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.metadata.push((key, value.into()));
        self
    }

    /// The detail of every shutdown ending and refusal:
    /// [`refusal_reason::SERVER_SHUTTING_DOWN`], [`SHUTDOWN_RETRY_DELAY`].
    pub fn server_shutting_down() -> Self {
        Self::new(
            refusal_reason::SERVER_SHUTTING_DOWN,
            Some(SHUTDOWN_RETRY_DELAY),
        )
    }

    /// The detail of a chained `UpdateLayer`'s gate refusal when no read of
    /// its writer's record succeeded before its deadline cap:
    /// [`refusal_reason::CHAIN_CHECK_UNAVAILABLE`], [`CAPACITY_RETRY_DELAY`].
    pub fn chain_check_unavailable() -> Self {
        Self::new(
            refusal_reason::CHAIN_CHECK_UNAVAILABLE,
            Some(CAPACITY_RETRY_DELAY),
        )
    }

    /// The detail of a `CreateLayer` / `UpdateLayer` refused before its commit
    /// because a dependency was unavailable:
    /// [`refusal_reason::UPSTREAM_UNAVAILABLE`],
    /// [`UPSTREAM_UNAVAILABLE_RETRY_DELAY`]. Attached only by the uploads'
    /// pre-commit choke point (`grpc::layer`'s `refused_before_commit`; see
    /// [`ServiceError::UnavailableWithDetail`]).
    pub fn upstream_unavailable() -> Self {
        Self::new(
            refusal_reason::UPSTREAM_UNAVAILABLE,
            Some(UPSTREAM_UNAVAILABLE_RETRY_DELAY),
        )
    }

    /// The detail of a delta refused for carrying more entries than one delta
    /// may hold: [`refusal_reason::DELTA_TOO_LARGE`] with its three counts and
    /// the limit, no `RetryInfo`.
    pub fn delta_too_large(
        struct_commands: u64,
        set_fields: u64,
        time_samples: u64,
        max_entries: u64,
    ) -> Self {
        Self::new(refusal_reason::DELTA_TOO_LARGE, None)
            .with_metadata(
                refusal_metadata::STRUCT_COMMANDS,
                struct_commands.to_string(),
            )
            .with_metadata(refusal_metadata::SET_FIELDS, set_fields.to_string())
            .with_metadata(refusal_metadata::TIME_SAMPLES, time_samples.to_string())
            .with_metadata(refusal_metadata::MAX_ENTRIES, max_entries.to_string())
    }

    /// The detail of a delta refused for a `ReorderChildren` list longer than
    /// one list may be: [`refusal_reason::CHILDREN_LIST_TOO_LONG`] with the
    /// limit and the list's length, no `RetryInfo`.
    pub fn children_list_too_long(limit: u64, actual: u64) -> Self {
        Self::new(refusal_reason::CHILDREN_LIST_TOO_LONG, None)
            .with_metadata(refusal_metadata::LIMIT, limit.to_string())
            .with_metadata(refusal_metadata::ACTUAL, actual.to_string())
    }

    /// [`ServiceError::UnavailableWithDetail`] with `message` and this detail,
    /// for a refusal decided where a `ServiceError` is returned.
    pub fn into_unavailable_error(self, message: impl Into<String>) -> ServiceError {
        ServiceError::UnavailableWithDetail {
            message: message.into(),
            detail: Box::new(self),
        }
    }

    /// `UNAVAILABLE` `message`, carrying this detail: [`Self::status`] with
    /// [`tonic::Code::Unavailable`].
    pub fn unavailable(&self, message: impl Into<String>) -> Status {
        self.status(tonic::Code::Unavailable, message)
    }

    /// `code` `message`, carrying this detail, with a `RetryInfo` delay freshly
    /// drawn from [`Self::retry_delay`] when it is `Some`.
    pub fn status(&self, code: tonic::Code, message: impl Into<String>) -> Status {
        refusal_status(
            code,
            message,
            self.reason,
            self.metadata
                .iter()
                .map(|(key, value)| (*key, value.as_str())),
            self.retry_delay.as_ref().map(RetryDelayRange::sample),
        )
    }
}

impl ServiceError {
    /// A chained `UpdateLayer`'s predecessor refusal
    /// ([`ServiceError::PredecessorNotCommitted`]): `ABORTED` `message`,
    /// carrying [`refusal_reason::PREDECESSOR_NOT_COMMITTED`] with
    /// [`refusal_metadata::LAST_RECORD`] for `last` (and
    /// [`refusal_metadata::LAST_OPERATION_ID`] for an operation), and no
    /// `RetryInfo`. Called by the chain gate and by the commit path on a
    /// call's first attempt — never after a counted retry, whose refusal
    /// cannot promise the call applied nothing.
    pub fn predecessor_not_committed(message: impl Into<String>, last: LastRecord<'_>) -> Self {
        let detail = RefusalDetail::new(refusal_reason::PREDECESSOR_NOT_COMMITTED, None);
        let detail = match last {
            LastRecord::Absent => {
                detail.with_metadata(refusal_metadata::LAST_RECORD, refusal_metadata::ABSENT)
            }
            LastRecord::Operation(operation_id) => detail
                .with_metadata(refusal_metadata::LAST_RECORD, refusal_metadata::OPERATION)
                .with_metadata(refusal_metadata::LAST_OPERATION_ID, operation_id),
            LastRecord::NoOperationId => detail.with_metadata(
                refusal_metadata::LAST_RECORD,
                refusal_metadata::NO_OPERATION_ID,
            ),
        };
        ServiceError::PredecessorNotCommitted {
            message: message.into(),
            detail: Box::new(detail),
        }
    }

    /// The `INVALID_ARGUMENT` a chained `UpdateLayer` is answered with
    /// ([`ServiceError::InvalidArgumentWithDetail`]): `message`, carrying
    /// [`refusal_reason::INVALID_UPDATE`] with no metadata and no
    /// `RetryInfo`.
    pub fn invalid_update(message: impl Into<String>) -> Self {
        ServiceError::InvalidArgumentWithDetail {
            message: message.into(),
            detail: Box::new(RefusalDetail::new(refusal_reason::INVALID_UPDATE, None)),
        }
    }

    /// The `INVALID_ARGUMENT` a delta of more entries than one delta may hold
    /// is answered with, whichever check caught it — the parser's entry-count
    /// probe (`OodfError::TooManyTables`) or the domain bound
    /// (`DeltaError::TooManyEntries`): a message naming the counts, the limit
    /// and the remedy, and [`RefusalDetail::delta_too_large`].
    pub fn delta_too_large(
        struct_commands: u64,
        set_fields: u64,
        time_samples: u64,
        max_entries: u64,
    ) -> Self {
        let total = struct_commands
            .saturating_add(set_fields)
            .saturating_add(time_samples);
        ServiceError::InvalidArgumentWithDetail {
            message: format!(
                "delta too large: {total} entries ({struct_commands} struct commands, \
                 {set_fields} set fields, {time_samples} time samples); one delta may carry at \
                 most {max_entries} entries in total; split it into smaller deltas"
            ),
            detail: Box::new(RefusalDetail::delta_too_large(
                struct_commands,
                set_fields,
                time_samples,
                max_entries,
            )),
        }
    }

    /// The `INVALID_ARGUMENT` a `ReorderChildren` list longer than `limit` is
    /// answered with, whichever check caught it — the domain bound
    /// (`DeltaError::ChildrenListTooLong`) or the parser's higher defensive
    /// cap (`OodfError::ChildrenListTooLong`): the domain refusal's message,
    /// and [`RefusalDetail::children_list_too_long`].
    pub fn children_list_too_long(limit: u64, actual: u64) -> Self {
        ServiceError::InvalidArgumentWithDetail {
            message: format!(
                "delta error: {}",
                crate::domain::delta::DeltaError::ChildrenListTooLong { limit, actual }
            ),
            detail: Box::new(RefusalDetail::children_list_too_long(limit, actual)),
        }
    }
}

/// The status of a refusal that carries a refusal detail: `code` and `message`
/// exactly as given, and in `grpc-status-details-bin` a `google.rpc.Status`
/// whose details are a `google.rpc.ErrorInfo` {`reason`, domain
/// [`REFUSAL_DETAIL_DOMAIN`], `metadata`} and, when `retry_delay` is `Some`, a
/// `google.rpc.RetryInfo` {`retry_delay`}.
///
/// The one builder of the service's refusal details: every refusal that
/// carries one is built here, so the wire shape cannot drift between them.
/// The caller draws `retry_delay` per response
/// ([`RetryDelayRange::sample`]).
pub fn refusal_status<'a>(
    code: tonic::Code,
    message: impl Into<String>,
    reason: &str,
    metadata: impl IntoIterator<Item = (&'a str, &'a str)>,
    retry_delay: Option<std::time::Duration>,
) -> Status {
    use tonic_types::{ErrorDetails, StatusExt};
    let metadata: std::collections::HashMap<String, String> = metadata
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect();
    let mut details = ErrorDetails::with_error_info(reason, REFUSAL_DETAIL_DOMAIN, metadata);
    if let Some(delay) = retry_delay {
        details.set_retry_info(Some(delay));
    }
    Status::with_error_details(code, message, details)
}

/// The message of every server-fault log line, and the redacted `INTERNAL`
/// wire message's text.
pub(crate) const INTERNAL_ERROR_MSG: &str = "internal error";

/// Build the redacted `INTERNAL` status for a server fault, carrying `detail`
/// as an [`InternalFault`] source.
///
/// The conversion does not log. The detail travels with the `Status` to the
/// completion point that has the request context —
/// `grpc::rpc_log::log_rpc_failure`, called by a handler boundary or a
/// streaming tail — which emits the one ERROR line with the request fields and
/// marks the fault reported. A fault nothing reports logs itself on drop, so
/// there is exactly one ERROR line per fault either way.
///
/// The wire bytes are unchanged by the source: `Status::add_header` writes the
/// code, message, details and metadata only, never the source.
fn internal_status(detail: String) -> Status {
    let mut status = Status::internal(INTERNAL_ERROR_MSG); // don't leak details to client
    status.set_source(std::sync::Arc::new(InternalFault::new(detail)));
    status
}

/// The un-redacted detail of a server fault, attached as the `source` of the
/// `INTERNAL` status `From<ServiceError> for Status` builds.
///
/// **Exactly one ERROR line per fault.** `grpc::rpc_log::log_rpc_failure`
/// recognises it by downcast, logs `internal error` with the detail and the
/// request fields, and calls [`Self::mark_reported`]. If nothing does — the
/// status was built and discarded, or it reached a path with no completion
/// point — `Drop` emits the line itself, without request context, when the
/// last clone of the status goes away.
///
/// `Display` is the redacted text; `Debug` carries the detail, which is
/// server-side only (it appears in the `Status`'s own `Display` as its source,
/// never on the wire).
pub(crate) struct InternalFault {
    detail: String,
    reported: std::sync::atomic::AtomicBool,
}

impl InternalFault {
    pub(crate) fn new(detail: String) -> Self {
        Self {
            detail,
            reported: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// The un-redacted detail. Server-side log output only.
    pub(crate) fn detail(&self) -> &str {
        &self.detail
    }

    /// Record that a completion point logged this fault, which silences the
    /// `Drop` fallback. Release pairs with the Acquire in `Drop`: the last
    /// clone of the status can drop on another thread.
    pub(crate) fn mark_reported(&self) {
        self.reported
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

impl std::fmt::Display for InternalFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(INTERNAL_ERROR_MSG)
    }
}

impl std::fmt::Debug for InternalFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InternalFault")
            .field("detail", &self.detail)
            .finish_non_exhaustive()
    }
}

impl std::error::Error for InternalFault {}

impl Drop for InternalFault {
    /// The fallback: the line the conversion itself emitted before request
    /// context was attached, for a fault no completion point reported.
    fn drop(&mut self) {
        if !self.reported.load(std::sync::atomic::Ordering::Acquire) {
            tracing::error!(error = %self.detail, "{}", INTERNAL_ERROR_MSG);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tonic::Code;

    #[test]
    fn invalid_argument_maps_to_correct_code_and_message() {
        let status = tonic::Status::from(ServiceError::InvalidArgument("bad field".to_string()));
        assert_eq!(status.code(), Code::InvalidArgument);
        assert_eq!(status.message(), "bad field");
    }

    #[test]
    fn not_found_maps_to_correct_code_and_message() {
        let status = tonic::Status::from(ServiceError::NotFound("session xyz".to_string()));
        assert_eq!(status.code(), Code::NotFound);
        assert_eq!(status.message(), "session xyz");
    }

    #[test]
    fn already_exists_maps_to_correct_code_and_message() {
        let status = tonic::Status::from(ServiceError::AlreadyExists("layer abc".to_string()));
        assert_eq!(status.code(), Code::AlreadyExists);
        assert_eq!(status.message(), "layer abc");
    }

    #[test]
    fn permission_denied_maps_to_correct_code_and_empty_message() {
        let status = tonic::Status::from(ServiceError::PermissionDenied);
        assert_eq!(status.code(), Code::PermissionDenied);
        assert_eq!(status.message(), "");
    }

    #[test]
    fn unauthenticated_maps_to_correct_code_and_empty_message() {
        let status = tonic::Status::from(ServiceError::Unauthenticated);
        assert_eq!(status.code(), Code::Unauthenticated);
        assert_eq!(status.message(), "");
    }

    #[test]
    fn failed_precondition_maps_to_correct_code_and_message() {
        let status = tonic::Status::from(ServiceError::FailedPrecondition(
            "object_id mismatch".to_string(),
        ));
        assert_eq!(status.code(), Code::FailedPrecondition);
        assert_eq!(status.message(), "object_id mismatch");
    }

    /// The in-process retry signal never reaches the wire as gRPC `ABORTED`
    /// (the service's one `ABORTED` is a chained `UpdateLayer`'s
    /// `PredecessorNotCommitted`): an `Aborted` that escapes a handler — a
    /// retry loop's exhaustion, or a single-attempt read reusing the in-loop
    /// mapper (e.g. the SubscribeReadLayer catch-up snapshot read) — reaches
    /// the client as `UNAVAILABLE`, which the Kit client retries (it treats
    /// `ABORTED` as terminal). The message travels verbatim, unredacted.
    #[test]
    fn aborted_maps_to_unavailable_with_message_verbatim() {
        let msg = "transient I/O: LRANGE children list (parent 0) failed (Timeout): stall";
        let status = tonic::Status::from(ServiceError::Aborted(msg.to_string()));
        assert_eq!(
            status.code(),
            Code::Unavailable,
            "an escaped `Aborted` must surface as retriable UNAVAILABLE, never ABORTED"
        );
        assert_eq!(status.message(), msg);
        // The wire pin for an `UpdateLayer` retry exhaustion: its commit may
        // have been sent, so it must not claim it applied nothing — no
        // `ErrorInfo`, no `RetryInfo` (SDD §18.1's never-list).
        assert!(
            status.details().is_empty(),
            "an escaped `Aborted` carries no refusal detail; got {status:?}"
        );
    }

    #[test]
    fn cancelled_maps_to_correct_code_and_message() {
        // b.jr8 mediums: a mid-upload transport error on `read_upload_body`'s
        // chunk loop is surfaced as `ServiceError::Cancelled(...)` — the
        // status code must be `CANCELLED` and the message must be preserved
        // verbatim (unlike `Internal`, we do not redact, because the
        // message is client-transport noise, not a service-side detail).
        let status = tonic::Status::from(ServiceError::Cancelled(
            "stream receive error: transport closed".to_string(),
        ));
        assert_eq!(status.code(), Code::Cancelled);
        assert_eq!(status.message(), "stream receive error: transport closed");
    }

    #[test]
    fn cancelled_display_includes_message() {
        let err = ServiceError::Cancelled("peer gone".to_string());
        assert_eq!(err.to_string(), "cancelled: peer gone");
    }

    #[test]
    fn internal_maps_to_correct_code_and_sanitized_message() {
        let status = tonic::Status::from(ServiceError::Internal("secret db password".to_string()));
        assert_eq!(status.code(), Code::Internal);
        assert_eq!(status.message(), "internal error");
        assert!(!status.message().contains("secret"));
        assert!(!status.message().contains("db password"));
    }

    #[test]
    fn display_invalid_argument_includes_message() {
        let err = ServiceError::InvalidArgument("missing name".to_string());
        assert_eq!(err.to_string(), "invalid argument: missing name");
    }

    #[test]
    fn display_not_found_includes_message() {
        let err = ServiceError::NotFound("session 42".to_string());
        assert_eq!(err.to_string(), "not found: session 42");
    }

    #[test]
    fn display_internal_includes_original_message() {
        let err = ServiceError::Internal("db connection failed".to_string());
        assert_eq!(err.to_string(), "internal error: db connection failed");
    }

    #[test]
    fn display_permission_denied_has_fixed_message() {
        let err = ServiceError::PermissionDenied;
        assert_eq!(err.to_string(), "permission denied");
    }

    #[test]
    fn display_unauthenticated_has_fixed_message() {
        let err = ServiceError::Unauthenticated;
        assert_eq!(err.to_string(), "unauthenticated");
    }

    // ── ExternalValueError → gRPC INTERNAL ──────────────────────────────

    #[test]
    fn external_value_meta_not_found_maps_to_internal() {
        let err = ExternalValueError::MetaNotFound {
            hash_hex: "abcd".to_string(),
        };
        let svc_err: ServiceError = err.into();
        let status = tonic::Status::from(svc_err);
        assert_eq!(status.code(), Code::Internal);
    }

    #[test]
    fn external_value_data_not_found_maps_to_internal() {
        let err = ExternalValueError::DataNotFound {
            hash_hex: "abcd".to_string(),
        };
        let svc_err: ServiceError = err.into();
        let status = tonic::Status::from(svc_err);
        assert_eq!(status.code(), Code::Internal);
    }

    #[test]
    fn external_value_meta_not_found_display_includes_hash() {
        let err = ExternalValueError::MetaNotFound {
            hash_hex: "deadbeef".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("deadbeef"),
            "should include hash hex, got: {msg}"
        );
    }

    #[test]
    fn external_value_data_not_found_display_includes_hash() {
        let err = ExternalValueError::DataNotFound {
            hash_hex: "cafebabe".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("cafebabe"),
            "should include hash hex, got: {msg}"
        );
    }

    // ── DeltaError → ServiceError ──────────────────────────────────────────

    #[test]
    fn delta_inconsistent_state_maps_to_internal() {
        let err = DeltaError::InconsistentState {
            section_id: 42,
            operation: "update_field",
        };
        let svc_err: ServiceError = err.into();
        let status = tonic::Status::from(svc_err);
        assert_eq!(status.code(), Code::Internal);
    }

    // b.qw6 item 2: `SetOrderOverflow` now maps to Internal (was
    // InvalidArgument pre-fix). Joins `SequenceOverflow` in the bucket
    // of "u64 cursor reached the Lua f64 ceiling — no retry, no
    // client-input fix" — and pins the cross-backend parity guarantee
    // from the AC: the Lua fast path's SET_ORDER_OVERFLOW reply and
    // the slow path's `DeltaError::SetOrderOverflow` both surface as
    // gRPC Internal.
    #[test]
    fn delta_set_order_overflow_maps_to_internal() {
        let err = DeltaError::SetOrderOverflow {
            next: (1u64 << 53) - 1,
            delta: 1,
        };
        let svc_err: ServiceError = err.into();
        let status = tonic::Status::from(svc_err);
        assert_eq!(
            status.code(),
            Code::Internal,
            "SetOrderOverflow must map to Internal so the slow path matches \
             the Lua fast path's SET_ORDER_OVERFLOW → Internal route (b.qw6 \
             item 2 parity)"
        );
    }

    #[test]
    fn delta_sequence_overflow_maps_to_internal() {
        // Pin the existing SequenceOverflow contract alongside the new
        // SetOrderOverflow → Internal mapping. Both share the same
        // bucket; any future refactor that moves one out of `Internal`
        // should fail loudly on the other for parity.
        let err = DeltaError::SequenceOverflow;
        let svc_err: ServiceError = err.into();
        let status = tonic::Status::from(svc_err);
        assert_eq!(status.code(), Code::Internal);
    }

    // ── OodfError → ServiceError ───────────────────────────────────────────

    #[test]
    fn oodf_empty_input_maps_to_invalid_argument() {
        let err: ServiceError = oodf_ffi::OodfError::EmptyInput.into();
        assert_eq!(tonic::Status::from(err).code(), Code::InvalidArgument);
    }

    #[test]
    fn oodf_too_short_maps_to_invalid_argument() {
        let err: ServiceError = oodf_ffi::OodfError::TooShort.into();
        assert_eq!(tonic::Status::from(err).code(), Code::InvalidArgument);
    }

    #[test]
    fn oodf_bad_magic_maps_to_invalid_argument() {
        let err: ServiceError = oodf_ffi::OodfError::BadMagic.into();
        assert_eq!(tonic::Status::from(err).code(), Code::InvalidArgument);
    }

    #[test]
    fn oodf_verifier_failed_maps_to_invalid_argument() {
        let err: ServiceError =
            oodf_ffi::OodfError::VerifierFailed("bad root table".to_string()).into();
        assert_eq!(tonic::Status::from(err).code(), Code::InvalidArgument);
    }

    #[test]
    fn oodf_unknown_command_type_maps_to_invalid_argument() {
        let err: ServiceError = oodf_ffi::OodfError::UnknownCommandType(42).into();
        assert_eq!(tonic::Status::from(err).code(), Code::InvalidArgument);
    }

    #[test]
    fn oodf_unknown_command_tag_maps_to_invalid_argument() {
        let err: ServiceError =
            oodf_ffi::OodfError::UnknownCommandTagAt { index: 4, tag: 9 }.into();
        assert_eq!(tonic::Status::from(err).code(), Code::InvalidArgument);
    }

    #[test]
    fn oodf_allocation_failed_maps_to_internal() {
        let err: ServiceError = oodf_ffi::OodfError::AllocationFailed.into();
        assert_eq!(tonic::Status::from(err).code(), Code::Internal);
    }

    #[test]
    fn oodf_internal_error_maps_to_internal() {
        let err: ServiceError = oodf_ffi::OodfError::InternalError("boom".to_string()).into();
        assert_eq!(tonic::Status::from(err).code(), Code::Internal);
    }

    #[test]
    fn oodf_ffi_error_maps_to_internal() {
        let err: ServiceError = oodf_ffi::OodfError::Ffi("bridge weirdness".to_string()).into();
        assert_eq!(tonic::Status::from(err).code(), Code::Internal);
    }

    // b.fd9: the pre-fix tests `delta_oodf_invalid_argument_variant_maps_to_invalid_argument`
    // and `delta_oodf_internal_variant_maps_to_internal` constructed
    // `DeltaError::Oodf(...)` to assert the `From<DeltaError>` impl
    // delegated to `From<OodfError>`. Now that the `Oodf` variant is
    // removed (no domain call site ever constructed it; gRPC handlers map
    // `OodfError` directly), the round-trip lives entirely on the
    // `oodf_*_maps_to_*` tests above, which exercise the same
    // `From<OodfError>` impl.

    // ── DiffError → ServiceError ───────────────────────────────────────────
    //
    // b.rsw: depth-cap breaches surface ONLY through `DiffError`, not the
    // pre-b.rsw redundant `DeltaError::SectionTreeTooDeep` (which had no
    // constructor and would have routed to `InvalidArgument` had it ever
    // been hit). The live path maps to `Internal` because a malformed
    // section tree at diff time means the *stored* state is corrupted —
    // not that the subscriber's read is bad — so the client should retry
    // against a healthy replica rather than treating the layer as invalid.

    #[test]
    fn diff_section_tree_too_deep_maps_to_internal() {
        let err = crate::domain::diff::DiffError::SectionTreeTooDeep {
            depth: 10_001,
            cap: 10_000,
        };
        let svc_err: ServiceError = err.into();
        let status = tonic::Status::from(svc_err);
        assert_eq!(status.code(), Code::Internal);
    }

    // ── b.ohw H-grpc-1: AuthError → ServiceError mapping ──────────────────
    //
    // Pre-fix every AuthError variant routed to `ServiceError::Internal`,
    // which surfaced as gRPC `Code::Internal` (13) — non-retriable per
    // standard client policies. Transient outages
    // (Timeout / Unreachable) now map to `ServiceError::Unavailable`
    // (Code::Unavailable, 14, retriable). Permanent contract violations
    // (`InvalidResponse`) keep the Internal mapping with a redacted
    // client-facing message, and so does the userinfo service's
    // `UserNotFound` answer when nothing falls back from it.

    #[test]
    fn auth_error_timeout_maps_to_unavailable_status() {
        let svc: ServiceError = crate::auth::AuthError::Timeout.into();
        assert!(
            matches!(svc, ServiceError::Unavailable(_)),
            "AuthError::Timeout must map to ServiceError::Unavailable so gRPC retries kick in (b.ohw H-grpc-1)"
        );
        let status: tonic::Status = svc.into();
        assert_eq!(
            status.code(),
            Code::Unavailable,
            "Status::unavailable's code must be 14 — retriable per gRPC client retry policies"
        );
        // Unavailable's message is operator-facing (not security-sensitive)
        // so it preserves the inner detail. Verify the AuthError display
        // round-trips so log readers can identify the source.
        assert!(
            status.message().contains("auth backend"),
            "message should preserve the auth-backend tag, got: {}",
            status.message()
        );
    }

    #[test]
    fn auth_error_unreachable_maps_to_unavailable_status() {
        let svc: ServiceError = crate::auth::AuthError::Unreachable.into();
        assert!(matches!(svc, ServiceError::Unavailable(_)));
        let status: tonic::Status = svc.into();
        assert_eq!(status.code(), Code::Unavailable);
    }

    #[test]
    fn auth_error_invalid_response_maps_to_internal_status_with_redacted_message() {
        // `InvalidResponse` carries the wrapped backend description (e.g.
        // "auth backend returned 14"). It must NOT propagate through to the
        // Status message body — `ServiceError::Internal`'s conversion
        // redacts to a generic "internal error" string. Any future
        // regression that lets the inner message escape would violate the
        // status.message() scrubbing posture (H-auth-5).
        let auth_err =
            crate::auth::AuthError::InvalidResponse("sensitive backend detail".to_owned());
        let svc: ServiceError = auth_err.into();
        assert!(
            matches!(svc, ServiceError::Internal(_)),
            "AuthError::InvalidResponse must keep the Internal mapping — it signals a real bug"
        );
        let status: tonic::Status = svc.into();
        assert_eq!(status.code(), Code::Internal);
        assert!(
            !status.message().contains("sensitive backend detail"),
            "Internal status must redact the wrapped detail, got: {}",
            status.message()
        );
        assert_eq!(status.message(), "internal error");
    }

    /// The userinfo service's "no such user" answer reaches this mapping only
    /// with authorization off (with it on, the fallback decorator has turned
    /// it into a claim-derived name). There it still ends the RPC `INTERNAL`
    /// with the redacted message — not the retryable `UNAVAILABLE` a client
    /// would loop on.
    #[test]
    fn auth_error_user_not_found_maps_to_internal_status_with_redacted_message() {
        let svc: ServiceError = crate::auth::AuthError::UserNotFound.into();
        assert!(
            matches!(svc, ServiceError::Internal(_)),
            "AuthError::UserNotFound must map to ServiceError::Internal, got {svc:?}"
        );
        let status: tonic::Status = svc.into();
        assert_eq!(status.code(), Code::Internal);
        assert_eq!(status.message(), "internal error");
    }

    /// `ServiceError::Unavailable` round-trips faithfully to a
    /// `Status::unavailable`. Distinct from the `AuthError → Unavailable`
    /// test above because callers also produce `Unavailable` directly from
    /// other code paths (e.g. storage-backend outages); this guards the
    /// generic mapping.
    #[test]
    fn service_error_unavailable_maps_to_unavailable_code_with_message_preserved() {
        let svc = ServiceError::Unavailable("backend X is unreachable".to_owned());
        let status = tonic::Status::from(svc);
        assert_eq!(status.code(), Code::Unavailable);
        assert_eq!(
            status.message(),
            "backend X is unreachable",
            "Unavailable's message body is operator/log noise — preserve it verbatim"
        );
    }

    /// Display formatting for the new `Unavailable` variant. Mirrors the
    /// existing `Cancelled` / `Internal` display tests.
    #[test]
    fn service_error_unavailable_display_includes_message() {
        let err = ServiceError::Unavailable("auth backend".to_owned());
        assert_eq!(err.to_string(), "unavailable: auth backend");
    }

    // ── b.33e: CheckpointError → ServiceError recovery classification ─────
    //
    // Pre-fix, all four layer/subscribe `RecoveryOutcome::Error` arms
    // (`grpc::layer` x3, `grpc::subscribe` x1 — `grpc::session`'s arm matches
    // the distinct `SessionRecoveryOutcome::Error` and is not counted here)
    // constructed `ServiceError::Unavailable` unconditionally, so a
    // `Permanent` corrupt-blob failure invited the client's retry policy to
    // hammer a failure that cannot succeed. Only `Transient` is retriable per
    // SDD §18.1; every other variant means an inconsistent store or contract
    // drift and maps to `Internal`.

    #[test]
    fn recovery_error_transient_is_retriable_and_maps_to_unavailable() {
        let err = crate::checkpoint::store::CheckpointError::Transient {
            source: "backend blip".into(),
        };
        assert!(
            ServiceError::recovery_error_is_retriable(&err),
            "Transient is the only retriable checkpoint-recovery class (SDD §18.1)"
        );
        let svc =
            ServiceError::from_recovery_error(&err, "recovery failed for object 7".to_owned());
        assert!(matches!(svc, ServiceError::Unavailable(_)));
        let status = tonic::Status::from(svc);
        assert_eq!(status.code(), Code::Unavailable);
        assert_eq!(
            status.message(),
            "recovery failed for object 7",
            "Unavailable preserves the caller's message verbatim"
        );
    }

    #[test]
    fn recovery_error_permanent_is_not_retriable_and_maps_to_internal() {
        let err = crate::checkpoint::store::CheckpointError::Permanent {
            source: "corrupt checkpoint blob".into(),
        };
        assert!(!ServiceError::recovery_error_is_retriable(&err));
        let svc = ServiceError::from_recovery_error(&err, format!("recovery failed: {err}"));
        assert!(
            matches!(svc, ServiceError::Internal(_)),
            "Permanent must NOT present as retriable UNAVAILABLE (b.33e)"
        );
        let status = tonic::Status::from(svc);
        assert_eq!(status.code(), Code::Internal);
        assert_eq!(
            status.message(),
            "internal error",
            "Internal redacts the backend detail from the wire; the ERROR log carries it"
        );
    }

    #[test]
    fn recovery_error_not_found_is_not_retriable_and_maps_to_internal() {
        // Plain absence never reaches an `Error` arm — `CheckpointStore::get_*`
        // answers `Ok(None)`, routed to `NotRecoverable`. A `NotFound` here is
        // an inconsistent store, not something a retry heals.
        let err = crate::checkpoint::store::CheckpointError::NotFound;
        assert!(!ServiceError::recovery_error_is_retriable(&err));
        let svc = ServiceError::from_recovery_error(&err, "recovery failed".to_owned());
        assert!(matches!(svc, ServiceError::Internal(_)));
    }

    #[test]
    fn recovery_error_conflict_is_not_retriable_and_maps_to_internal() {
        let err = crate::checkpoint::store::CheckpointError::Conflict;
        assert!(!ServiceError::recovery_error_is_retriable(&err));
        let svc = ServiceError::from_recovery_error(&err, "recovery failed".to_owned());
        assert!(matches!(svc, ServiceError::Internal(_)));
    }

    #[test]
    fn recovery_error_not_implemented_is_not_retriable_and_maps_to_internal() {
        let err = crate::checkpoint::store::CheckpointError::NotImplemented {
            method: "get_layer_checkpoint",
        };
        assert!(!ServiceError::recovery_error_is_retriable(&err));
        let svc = ServiceError::from_recovery_error(&err, "recovery failed".to_owned());
        assert!(matches!(svc, ServiceError::Internal(_)));
    }

    // ── b.6de sub-task 2: PermanentKeyspaceRefusal ────────────────────────
    //
    // The variant's whole purpose is to be distinguishable BY TYPE from the
    // transient `Internal` population while being indistinguishable from it on
    // the wire. Both halves are pinned here; the classification half (this
    // variant → `CheckpointError::Permanent`, a tag-carrying `Internal` →
    // `Transient`) is pinned in `checkpoint::apply`'s own unit tests.

    #[test]
    fn permanent_keyspace_refusal_maps_to_internal_with_sanitized_message() {
        let status = tonic::Status::from(ServiceError::PermanentKeyspaceRefusal {
            refusal: KeyspaceRefusal::WrongType,
            detail: "WRONGTYPE_KEY: le:{9}:obj:secs holds a string, expected zset".to_owned(),
        });
        assert_eq!(
            status.code(),
            Code::Internal,
            "the refusal is an operator-remediable server-side condition — same code as \
             `Internal`, so no client retry policy treats it as retriable"
        );
        assert_eq!(
            status.message(),
            "internal error",
            "the detail names live keyspace keys and must not reach the client; the ERROR log \
             carries it"
        );
        assert!(!status.message().contains("obj:secs"));
    }

    #[test]
    fn permanent_keyspace_refusal_stream_collision_kind_maps_to_internal() {
        // Constructed by no production site today (the create-path stream
        // guard's conversion is a follow-up), so this test is the kind's only
        // constructor in-crate. It exists so the kind cannot rot into a
        // non-compiling shape while unreachable.
        let status = tonic::Status::from(ServiceError::PermanentKeyspaceRefusal {
            refusal: KeyspaceRefusal::StreamIdCollision,
            detail: "create_layer stream-id collision: STREAM_ID_COLLISION: stream not empty"
                .to_owned(),
        });
        assert_eq!(status.code(), Code::Internal);
        assert_eq!(status.message(), "internal error");
    }

    #[test]
    fn permanent_keyspace_refusal_display_renders_detail_verbatim() {
        // `detail` carries the emitter's string unmodified, tag prefix and all,
        // so the operator-facing text is byte-identical to the `Internal` this
        // variant replaced and every prefix assertion downstream still holds.
        let detail = "create_layer keyspace type error: WRONGTYPE_KEY: le:{9}:obj:flds holds a \
                      string, expected zset";
        let err = ServiceError::PermanentKeyspaceRefusal {
            refusal: KeyspaceRefusal::WrongType,
            detail: detail.to_owned(),
        };
        assert_eq!(
            err.to_string(),
            format!("permanent keyspace refusal: {detail}")
        );
        assert!(
            err.to_string().contains("WRONGTYPE_KEY"),
            "the tag must survive into the rendered message — the Lua reply contract, the \
             remediation log line and the storage-lane assertions all spell it"
        );
    }

    // ── InternalFault: the one ERROR line per server fault ────────────────
    //
    // The conversion no longer logs. It attaches the un-redacted detail to the
    // `Status` as an `InternalFault` source, so the completion point that has
    // the request context can log it; a fault nobody reports logs itself on
    // drop. These tests pin the conversion half and the drop half. The
    // reporting half (`log_rpc_failure`'s fault arm) is pinned in
    // `grpc::rpc_log`'s unit tests.

    /// Capture sink for [`capture_json`]. Local copy of the tree-wide
    /// `CapturingWriter` idiom (`src/checkpoint/assembler.rs::tests`).
    #[derive(Clone)]
    struct CapturingWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturingWriter {
        type Writer = CapturingWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for CapturingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Run `f` under a thread-local JSON capture subscriber and return every
    /// event it emitted. `set_default` suffices: everything here is
    /// synchronous, so `Drop` runs on this thread inside the scope.
    fn capture_json(f: impl FnOnce()) -> Vec<serde_json::Value> {
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(CapturingWriter(std::sync::Arc::clone(&buf)))
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .finish();
        {
            let _guard = tracing::subscriber::set_default(subscriber);
            f();
        }
        let raw = buf.lock().unwrap_or_else(|p| p.into_inner()).clone();
        String::from_utf8(raw)
            .expect("captured tracing output must be UTF-8")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("each captured line must be JSON"))
            .collect()
    }

    /// One `ServiceError` per variant the conversion maps to `INTERNAL`, with
    /// the detail it carries. "Both variants behave the same" is asserted by
    /// running every test below over this list.
    fn server_faults() -> Vec<(&'static str, ServiceError)> {
        const INTERNAL_DETAIL: &str = "fault-detail-internal: HGETALL le:{3}:obj:hdr WRONGTYPE";
        const REFUSAL_DETAIL: &str = "fault-detail-refusal: WRONGTYPE_KEY: le:{3}:obj:secs";
        vec![
            (
                INTERNAL_DETAIL,
                ServiceError::Internal(INTERNAL_DETAIL.to_owned()),
            ),
            (
                REFUSAL_DETAIL,
                ServiceError::PermanentKeyspaceRefusal {
                    refusal: KeyspaceRefusal::WrongType,
                    detail: REFUSAL_DETAIL.to_owned(),
                },
            ),
        ]
    }

    /// The fault behind a status, by downcast — the same route
    /// `grpc::rpc_log::log_rpc_failure` takes.
    fn fault_of(status: &Status) -> &InternalFault {
        use std::error::Error as _;
        status
            .source()
            .and_then(|source| source.downcast_ref::<InternalFault>())
            .expect("a server-fault INTERNAL must carry an InternalFault source")
    }

    /// The conversion keeps the redacted wire shape and carries the detail on
    /// the side, as the status's source.
    #[test]
    fn server_fault_status_is_redacted_internal_carrying_the_detail_as_its_source() {
        for (detail, err) in server_faults() {
            let status = Status::from(err);
            assert_eq!(status.code(), Code::Internal);
            assert_eq!(status.message(), INTERNAL_ERROR_MSG);
            assert!(
                !status.message().contains(detail),
                "the detail must never reach the wire message"
            );
            let fault = fault_of(&status);
            assert_eq!(fault.detail(), detail);
            fault.mark_reported(); // keep this test's drop silent
        }
    }

    /// The source never reaches the wire: the headers tonic writes for the
    /// status carry the code and the redacted message, and no byte of the
    /// detail.
    ///
    /// `add_header` is what tonic's response path calls, so this is the wire
    /// check without a server. A source that leaked into `grpc-message` or
    /// `grpc-status-details-bin` would expose live keyspace key names to any
    /// client.
    #[test]
    fn server_fault_wire_headers_never_carry_the_detail() {
        for (detail, err) in server_faults() {
            let status = Status::from(err);
            let mut headers = http::HeaderMap::new();
            status
                .add_header(&mut headers)
                .expect("a redacted INTERNAL must encode");
            assert_eq!(
                headers.get("grpc-status").map(|v| v.as_bytes()),
                Some(&b"13"[..])
            );
            for (name, value) in &headers {
                let value = String::from_utf8_lossy(value.as_bytes());
                assert!(
                    !value.contains(detail) && !value.contains("WRONGTYPE"),
                    "header `{name}` carries fault detail: {value}"
                );
            }
            fault_of(&status).mark_reported();
        }
    }

    /// The conversion itself logs nothing, and a fault no completion point
    /// reported logs exactly one ERROR — today's line, `internal error` with the
    /// detail as `error` — when its status drops.
    ///
    /// The fallback is what keeps "exactly one ERROR per fault" true on paths
    /// with no completion point (a status built and discarded, or one that
    /// reached a path nothing logs). Without it such a fault would vanish.
    #[test]
    fn unreported_server_fault_logs_one_error_with_the_detail_when_its_status_drops() {
        for (detail, err) in server_faults() {
            let events = capture_json(|| drop(Status::from(err)));
            assert_eq!(
                events.len(),
                1,
                "exactly one line for an unreported fault — none from the conversion, one from \
                 the drop; got {events:?}"
            );
            assert_eq!(events[0]["level"], "ERROR");
            assert_eq!(events[0]["fields"]["message"], INTERNAL_ERROR_MSG);
            assert_eq!(events[0]["fields"]["error"], detail);
        }
    }

    /// The conversion alone emits nothing: building the status and holding it
    /// is silent. Separated from the drop test so a conversion that logged AND
    /// a drop that stayed silent could not net out to the same one line.
    #[test]
    fn server_fault_conversion_emits_no_line_before_the_status_drops() {
        for (_detail, err) in server_faults() {
            let mut held = None;
            let events = capture_json(|| held = Some(Status::from(err)));
            assert!(
                events.is_empty(),
                "the conversion must not log; the completion point that has the request context \
                 does. Got {events:?}"
            );
            let status = held.expect("the closure ran");
            fault_of(&status).mark_reported();
        }
    }

    /// A reported fault is silent on drop: the completion point already
    /// emitted the one ERROR, with context.
    #[test]
    fn reported_server_fault_logs_nothing_when_its_status_drops() {
        for (_detail, err) in server_faults() {
            let events = capture_json(|| {
                let status = Status::from(err);
                fault_of(&status).mark_reported();
                drop(status);
            });
            assert!(
                events.is_empty(),
                "a reported fault must not also log its fallback; got {events:?}"
            );
        }
    }

    /// The fallback fires once, when the LAST clone of the status drops.
    ///
    /// `Status: Clone` shares the source `Arc`, and the streaming paths clone
    /// statuses (a trailer sent to the client while the task keeps its own
    /// copy). A fallback that fired per clone would log a fault twice; one that
    /// fired on the first drop would log it before a later clone was reported.
    #[test]
    fn server_fault_fallback_fires_once_when_the_last_clone_drops() {
        for (detail, err) in server_faults() {
            let mut survivor = None;
            let first_drop = capture_json(|| {
                let status = Status::from(err);
                survivor = Some(status.clone());
                drop(status);
            });
            assert!(
                first_drop.is_empty(),
                "dropping one clone while another lives must not log; got {first_drop:?}"
            );
            let survivor = survivor.expect("the closure ran");
            let last_drop = capture_json(|| drop(survivor));
            assert_eq!(
                last_drop.len(),
                1,
                "the last clone's drop logs the one fallback line; got {last_drop:?}"
            );
            assert_eq!(last_drop[0]["fields"]["error"], detail);
        }
    }

    /// A fault marked through one clone silences the fallback for every clone —
    /// the tail logs through its copy and the client's copy drops later.
    #[test]
    fn server_fault_reported_through_one_clone_is_silent_for_all() {
        for (_detail, err) in server_faults() {
            let events = capture_json(|| {
                let status = Status::from(err);
                let clone = status.clone();
                fault_of(&clone).mark_reported();
                drop(clone);
                drop(status);
            });
            assert!(events.is_empty(), "got {events:?}");
        }
    }

    /// `Display` is the redacted text; the detail is only in `Debug`.
    ///
    /// `Status`'s own `Display` renders its source, so a `Display` carrying the
    /// detail would put it anywhere a status is formatted with `{}` — including
    /// status messages that do reach a client when a caller re-wraps one.
    #[test]
    fn internal_fault_display_is_redacted_and_debug_carries_the_detail() {
        let fault = InternalFault::new("secret-key-name".to_owned());
        assert_eq!(fault.to_string(), INTERNAL_ERROR_MSG);
        assert!(format!("{fault:?}").contains("secret-key-name"));
        fault.mark_reported();
    }

    #[test]
    fn keyspace_refusal_kinds_are_distinct_and_copy() {
        // `Copy + PartialEq` is what lets consumers match the kind without
        // taking the error apart; a derive removed here would surface as a
        // borrow error far from this file.
        let wrong = KeyspaceRefusal::WrongType;
        let collision = KeyspaceRefusal::StreamIdCollision;
        assert_ne!(wrong, collision);
        let copied = wrong;
        assert_eq!(copied, wrong);
    }

    #[test]
    fn delta_inconsistent_state_message_includes_section_id() {
        let err = DeltaError::InconsistentState {
            section_id: 99,
            operation: "delete_section",
        };
        let svc_err: ServiceError = err.into();
        let msg = svc_err.to_string();
        assert!(msg.contains("99"), "should include section_id, got: {msg}");
        assert!(
            msg.contains("delete_section"),
            "should include operation, got: {msg}"
        );
    }

    // ── The refusal detail (client contract) ────────────────────────────────

    use std::time::Duration;
    use tonic_types::StatusExt;

    /// The `ErrorInfo` a client decodes from `status`, as
    /// `(reason, domain, metadata)`.
    fn error_info(status: &Status) -> (String, String, Vec<(String, String)>) {
        let info = status
            .get_details_error_info()
            .unwrap_or_else(|| panic!("the status must carry an ErrorInfo: {status:?}"));
        let mut metadata: Vec<(String, String)> = info.metadata.into_iter().collect();
        metadata.sort();
        (info.reason, info.domain, metadata)
    }

    /// The domain and reasons are a client contract: their spellings are
    /// pinned here, so a rename is a red test rather than a silent wire break.
    #[test]
    fn refusal_detail_domain_reasons_and_metadata_keys_are_the_documented_spellings() {
        assert_eq!(REFUSAL_DETAIL_DOMAIN, "nvidia.omniverse.live.v1alpha");
        assert_eq!(refusal_reason::SERVER_SHUTTING_DOWN, "SERVER_SHUTTING_DOWN");
        assert_eq!(
            refusal_reason::SUBSCRIPTION_CAPACITY,
            "SUBSCRIPTION_CAPACITY"
        );
        assert_eq!(refusal_reason::READER_CAPACITY, "READER_CAPACITY");
        assert_eq!(refusal_reason::CATCHUP_CAPACITY, "CATCHUP_CAPACITY");
        assert_eq!(refusal_reason::UPLOAD_CAPACITY, "UPLOAD_CAPACITY");
        assert_eq!(refusal_reason::STORE_WRITE_REFUSED, "STORE_WRITE_REFUSED");
        assert_eq!(
            refusal_reason::SNAPSHOT_FENCE_TIMEOUT,
            "SNAPSHOT_FENCE_TIMEOUT"
        );
        assert_eq!(
            refusal_reason::EXT_RELEASE_BUDGET_FULL,
            "EXT_RELEASE_BUDGET_FULL"
        );
        assert_eq!(
            refusal_reason::CLIENT_KEEPALIVE_TIMEOUT,
            "CLIENT_KEEPALIVE_TIMEOUT"
        );
        assert_eq!(refusal_reason::UPSTREAM_UNAVAILABLE, "UPSTREAM_UNAVAILABLE");
        assert_eq!(
            [
                refusal_reason::PREDECESSOR_NOT_COMMITTED,
                refusal_reason::INVALID_UPDATE,
                refusal_reason::CHAIN_CHECK_UNAVAILABLE,
            ],
            [
                "PREDECESSOR_NOT_COMMITTED",
                "INVALID_UPDATE",
                "CHAIN_CHECK_UNAVAILABLE"
            ]
        );
        // The size refusals' reasons and keys, as the C++ client library reads
        // them.
        assert_eq!(
            [
                refusal_reason::DELTA_TOO_LARGE,
                refusal_reason::CHILDREN_LIST_TOO_LONG,
            ],
            ["DELTA_TOO_LARGE", "CHILDREN_LIST_TOO_LONG"]
        );
        assert_eq!(
            [
                refusal_metadata::STRUCT_COMMANDS,
                refusal_metadata::SET_FIELDS,
                refusal_metadata::TIME_SAMPLES,
                refusal_metadata::MAX_ENTRIES,
                refusal_metadata::LIMIT,
                refusal_metadata::ACTUAL,
            ],
            [
                "struct_commands",
                "set_fields",
                "time_samples",
                "max_entries",
                "limit",
                "actual"
            ]
        );
        assert_eq!(
            [
                refusal_metadata::CAUSE,
                refusal_metadata::SATURATED,
                refusal_metadata::WAITER_CAP,
                refusal_metadata::GROWTH,
                refusal_metadata::WAIT,
                refusal_metadata::SLOT,
                refusal_metadata::READ_PERMIT,
                refusal_metadata::CONDITION,
                refusal_metadata::LAST_RECORD,
                refusal_metadata::ABSENT,
                refusal_metadata::OPERATION,
                refusal_metadata::NO_OPERATION_ID,
                refusal_metadata::LAST_OPERATION_ID,
            ],
            [
                "cause",
                "saturated",
                "waiter_cap",
                "growth",
                "wait",
                "slot",
                "read_permit",
                "condition",
                "last_record",
                "absent",
                "operation",
                "no_operation_id",
                "last_operation_id"
            ]
        );
        assert_eq!(
            (
                SHUTDOWN_RETRY_DELAY,
                CAPACITY_RETRY_DELAY,
                STORE_WRITE_REFUSED_RETRY_DELAY,
                CLIENT_KEEPALIVE_TIMEOUT_RETRY_DELAY,
                UPSTREAM_UNAVAILABLE_RETRY_DELAY,
            ),
            (
                RetryDelayRange::new(Duration::ZERO, Duration::from_millis(100)),
                RetryDelayRange::new(Duration::from_millis(500), Duration::from_millis(1_500)),
                RetryDelayRange::new(Duration::from_secs(2), Duration::from_secs(6)),
                RetryDelayRange::new(Duration::ZERO, Duration::from_millis(100)),
                RetryDelayRange::new(Duration::from_secs(1), Duration::from_secs(3)),
            ),
            "the documented retry_delay ranges"
        );
    }

    /// Every draw lies in `[min, max)` and the draws vary across both halves
    /// of the range, and an empty range always yields `min` (a fixed delay).
    #[test]
    fn retry_delay_range_sample_stays_in_its_half_open_range() {
        for range in [
            SHUTDOWN_RETRY_DELAY,
            CAPACITY_RETRY_DELAY,
            STORE_WRITE_REFUSED_RETRY_DELAY,
            CLIENT_KEEPALIVE_TIMEOUT_RETRY_DELAY,
            UPSTREAM_UNAVAILABLE_RETRY_DELAY,
        ] {
            let mut distinct = std::collections::BTreeSet::new();
            let midpoint = range.min + (range.max - range.min) / 2;
            let (mut lower, mut upper) = (false, false);
            for _ in 0..1_000 {
                let delay = range.sample();
                assert!(
                    range.min <= delay && delay < range.max,
                    "{delay:?} outside {range:?}"
                );
                distinct.insert(delay);
                if delay < midpoint {
                    lower = true;
                } else {
                    upper = true;
                }
            }
            // The contract promises a delay "already randomized per
            // response": a fixed delay would send every refused client back
            // at once.
            assert!(
                distinct.len() > 1 && lower && upper,
                "1,000 draws from {range:?} must vary and reach both halves of the range; got \
                 {} distinct values (lower half hit: {lower}, upper half hit: {upper})",
                distinct.len()
            );
        }
        let fixed = RetryDelayRange::new(Duration::from_millis(250), Duration::from_millis(250));
        assert_eq!(fixed.sample(), Duration::from_millis(250));
        let inverted = RetryDelayRange::new(Duration::from_secs(2), Duration::from_secs(1));
        assert_eq!(inverted.sample(), Duration::from_secs(2));
    }

    /// `UnavailableWithDetail` reaches the wire as `UNAVAILABLE` with its
    /// message verbatim — byte for byte what `Unavailable` sends — plus, in
    /// `grpc-status-details-bin`, the `ErrorInfo` {reason, the service's
    /// domain, metadata} and a `RetryInfo` delay drawn from the detail's
    /// range. Its `Display` equals `Unavailable`'s, so every log line and
    /// wrapping message reads as before.
    #[test]
    fn unavailable_with_detail_maps_to_unavailable_with_the_message_and_the_detail() {
        const MSG: &str =
            "this replica's upload admission queue is full; nothing was applied, retry";
        let detail =
            RefusalDetail::new(refusal_reason::UPLOAD_CAPACITY, Some(CAPACITY_RETRY_DELAY))
                .with_metadata(refusal_metadata::CAUSE, refusal_metadata::WAITER_CAP);
        let err = detail.clone().into_unavailable_error(MSG);
        assert_eq!(
            err.to_string(),
            ServiceError::Unavailable(MSG.to_owned()).to_string(),
            "Display equals Unavailable's"
        );
        let status = Status::from(err);
        assert_eq!(status.code(), Code::Unavailable);
        assert_eq!(status.message(), MSG);
        assert_eq!(
            error_info(&status),
            (
                "UPLOAD_CAPACITY".to_owned(),
                REFUSAL_DETAIL_DOMAIN.to_owned(),
                vec![("cause".to_owned(), "waiter_cap".to_owned())]
            )
        );
        let delay = status
            .get_details_retry_info()
            .and_then(|r| r.retry_delay)
            .expect("a detail with a delay range carries a RetryInfo delay");
        assert!(
            CAPACITY_RETRY_DELAY.min <= delay && delay < CAPACITY_RETRY_DELAY.max,
            "{delay:?}"
        );
        assert_eq!(
            detail.unavailable(MSG).code(),
            Code::Unavailable,
            "`unavailable` is `status` with UNAVAILABLE"
        );
    }

    /// A plain `Unavailable` — a transient, an auth outage, an overtaken
    /// write — converts to a status with no detail at all: the conversion
    /// never infers a refusal detail. The one place a plain `UNAVAILABLE`
    /// gains one is outside this enum: the `CreateLayer` / `UpdateLayer`
    /// pre-commit choke point (`grpc::layer`'s `refused_before_commit`), which
    /// adds `UPSTREAM_UNAVAILABLE` to such a status raised before the call's
    /// commit started.
    #[test]
    fn plain_unavailable_carries_no_refusal_detail() {
        let status = Status::from(ServiceError::Unavailable("backend X".to_owned()));
        assert!(status.details().is_empty(), "{status:?}");
        assert!(status.get_details_error_info().is_none());
        assert!(status.get_details_retry_info().is_none());
    }

    /// The `None` branch of the shared builder: a detail without a delay
    /// range carries its `ErrorInfo` (reason, domain, every metadata entry)
    /// and no `RetryInfo` at all. The size refusals (`DELTA_TOO_LARGE`,
    /// `CHILDREN_LIST_TOO_LONG`) and the chained `UpdateLayer` refusals use
    /// it; the shape is pinned here on `UNAVAILABLE`, and on
    /// `INVALID_ARGUMENT` by the size-refusal tests below.
    #[test]
    fn refusal_detail_without_a_retry_delay_carries_error_info_and_no_retry_info() {
        let detail = RefusalDetail::new("SOME_REFUSAL", None)
            .with_metadata("limit", "16")
            .with_metadata("size", "32");
        let status = detail.unavailable("refused");
        assert_eq!(status.code(), Code::Unavailable);
        assert_eq!(status.message(), "refused");
        assert_eq!(
            error_info(&status),
            (
                "SOME_REFUSAL".to_owned(),
                REFUSAL_DETAIL_DOMAIN.to_owned(),
                vec![
                    ("limit".to_owned(), "16".to_owned()),
                    ("size".to_owned(), "32".to_owned())
                ]
            )
        );
        assert!(
            status.get_details_retry_info().is_none(),
            "no delay range, no RetryInfo; got {:?}",
            status.get_details_retry_info()
        );
        let err = detail.into_unavailable_error("refused");
        let status = Status::from(err);
        assert!(status.get_details_error_info().is_some());
        assert!(status.get_details_retry_info().is_none());
    }

    /// The builder keeps whatever code it is given, with or without a
    /// `RetryInfo`: a refusal of another code (such as `INVALID_ARGUMENT`
    /// `DELTA_TOO_LARGE`) reuses it and gets the same detail shape.
    #[test]
    fn refusal_detail_status_keeps_a_non_unavailable_code() {
        for (code, retry_delay) in [
            (Code::InvalidArgument, None),
            (
                Code::ResourceExhausted,
                Some(STORE_WRITE_REFUSED_RETRY_DELAY),
            ),
        ] {
            let status = RefusalDetail::new("SOME_REFUSAL", retry_delay)
                .with_metadata("limit", "16")
                .status(code, "too large");
            assert_eq!(status.code(), code);
            assert_eq!(status.message(), "too large");
            assert_eq!(
                error_info(&status),
                (
                    "SOME_REFUSAL".to_owned(),
                    REFUSAL_DETAIL_DOMAIN.to_owned(),
                    vec![("limit".to_owned(), "16".to_owned())]
                ),
                "{code:?}"
            );
            let delay = status.get_details_retry_info().and_then(|r| r.retry_delay);
            match retry_delay {
                None => assert_eq!(delay, None, "{code:?}: no RetryInfo"),
                Some(range) => {
                    let delay = delay.expect("a RetryInfo delay");
                    assert!(range.min <= delay && delay < range.max, "{delay:?}");
                }
            }
        }
        let direct = refusal_status(
            Code::FailedPrecondition,
            "direct",
            "SOME_REFUSAL",
            [("k", "v")],
            Some(Duration::from_millis(1_234)),
        );
        assert_eq!(direct.code(), Code::FailedPrecondition);
        assert_eq!(
            direct.get_details_retry_info().and_then(|r| r.retry_delay),
            Some(Duration::from_millis(1_234)),
            "`refusal_status` carries the delay it is given, as given"
        );
    }

    /// The shutdown detail: `SERVER_SHUTTING_DOWN`, no metadata, the
    /// shutdown range.
    #[test]
    fn refusal_detail_server_shutting_down_is_the_shutdown_reason_and_range() {
        assert_eq!(
            RefusalDetail::server_shutting_down(),
            RefusalDetail {
                reason: refusal_reason::SERVER_SHUTTING_DOWN,
                metadata: Vec::new(),
                retry_delay: Some(SHUTDOWN_RETRY_DELAY),
            }
        );
    }

    /// The pre-commit detail: `UPSTREAM_UNAVAILABLE`, no metadata, its own
    /// `[1 s, 3 s)` range — not the capacity range, so the two hints can
    /// change apart.
    #[test]
    fn refusal_detail_upstream_unavailable_is_the_upstream_reason_and_range() {
        assert_eq!(
            RefusalDetail::upstream_unavailable(),
            RefusalDetail {
                reason: refusal_reason::UPSTREAM_UNAVAILABLE,
                metadata: Vec::new(),
                retry_delay: Some(UPSTREAM_UNAVAILABLE_RETRY_DELAY),
            }
        );
    }

    // ── The size refusals (b.tbw) ───────────────────────────────────────────
    //
    // A delta over the entry limit is caught by the parser's entry-count
    // probe (`OodfError::TooManyTables`) or, for a delta that did not come
    // through the parser, by the domain bound (`DeltaError::TooManyEntries`);
    // an over-long children list by the domain bound
    // (`DeltaError::ChildrenListTooLong`) or the parser's higher defensive cap
    // (`OodfError::ChildrenListTooLong`). Whichever check caught it, the
    // client must read one refusal: the same code, message and `ErrorInfo`.

    /// `(reason, domain, metadata)` of an `ErrorInfo`, as [`error_info`]
    /// returns it.
    type WireErrorInfo = (String, String, Vec<(String, String)>);

    /// The wire form a client reads: code, message, `ErrorInfo` and whether
    /// a `RetryInfo` is present.
    fn wire(err: ServiceError) -> (Code, String, WireErrorInfo, bool) {
        let status = Status::from(err);
        let info = error_info(&status);
        let retry = status.get_details_retry_info().is_some();
        (status.code(), status.message().to_owned(), info, retry)
    }

    fn pairs(entries: &[(&str, &str)]) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = entries
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        out.sort();
        out
    }

    /// The parser's and the domain's over-limit refusals are one refusal on
    /// the wire: `INVALID_ARGUMENT`, a message naming the counts, the limit
    /// and the remedy, `DELTA_TOO_LARGE` with the four counts as decimal
    /// strings, and no `RetryInfo`.
    #[test]
    fn delta_too_large_from_the_parser_and_from_the_domain_is_one_refusal() {
        let from_parser = wire(
            oodf_ffi::OodfError::TooManyTables {
                struct_commands: 60_000,
                set_fields: 60_000,
                time_samples: 0,
                max_entries: 99_999,
            }
            .into(),
        );
        let from_domain = wire(
            DeltaError::TooManyEntries {
                struct_commands: 60_000,
                set_fields: 60_000,
                time_samples: 0,
                max_entries: 99_999,
            }
            .into(),
        );
        assert_eq!(
            from_parser, from_domain,
            "one refusal, whichever check caught it"
        );

        let (code, message, info, retry) = from_parser;
        assert_eq!(code, Code::InvalidArgument);
        assert_eq!(
            message,
            "delta too large: 120000 entries (60000 struct commands, 60000 set fields, 0 time \
             samples); one delta may carry at most 99999 entries in total; split it into \
             smaller deltas"
        );
        assert_eq!(
            info,
            (
                "DELTA_TOO_LARGE".to_owned(),
                REFUSAL_DETAIL_DOMAIN.to_owned(),
                pairs(&[
                    ("struct_commands", "60000"),
                    ("set_fields", "60000"),
                    ("time_samples", "0"),
                    ("max_entries", "99999"),
                ])
            )
        );
        assert!(
            !retry,
            "a size refusal carries no RetryInfo: no re-send can succeed"
        );
    }

    /// The domain's children-list refusal and the parser's higher defensive
    /// cap answer alike: `INVALID_ARGUMENT`, the domain refusal's text,
    /// `CHILDREN_LIST_TOO_LONG` with the service's 100,000-id limit (not the
    /// parser's 1,000,000) and the list's length, no `RetryInfo`.
    #[test]
    fn children_list_too_long_from_the_domain_and_from_the_parser_reports_the_service_limit() {
        for (err, actual) in [
            (
                ServiceError::from(DeltaError::ChildrenListTooLong {
                    limit: 100_000,
                    actual: 100_001,
                }),
                "100001",
            ),
            (
                ServiceError::from(oodf_ffi::OodfError::ChildrenListTooLong { len: 1_000_001 }),
                "1000001",
            ),
        ] {
            let (code, message, info, retry) = wire(err);
            assert_eq!(code, Code::InvalidArgument);
            assert_eq!(
                message,
                format!(
                    "delta error: delta bounds exceeded: ReorderChildren.children_list.len \
                     limit=100000, actual={actual}"
                )
            );
            assert_eq!(
                info,
                (
                    "CHILDREN_LIST_TOO_LONG".to_owned(),
                    REFUSAL_DETAIL_DOMAIN.to_owned(),
                    pairs(&[("limit", "100000"), ("actual", actual)])
                )
            );
            assert!(!retry, "{actual}: no RetryInfo");
        }
    }

    /// Only the two size refusals carry a detail: every other delta bound, and
    /// every other parse refusal, stays a plain `INVALID_ARGUMENT`.
    #[test]
    fn other_delta_bounds_and_parse_refusals_carry_no_refusal_detail() {
        for err in [
            ServiceError::from(DeltaError::DeltaBoundsExceeded {
                field: "SetField.key_name",
                limit: 255,
                actual: 256,
            }),
            ServiceError::from(oodf_ffi::OodfError::VerifierFailed(
                "FlatBuffers verifier rejected a 9-byte OODF buffer".to_owned(),
            )),
            ServiceError::from(oodf_ffi::OodfError::BadMagic),
        ] {
            assert!(
                matches!(err, ServiceError::InvalidArgument(_)),
                "a plain InvalidArgument; got {err:?}"
            );
            let status = Status::from(err);
            assert_eq!(status.code(), Code::InvalidArgument);
            assert!(status.details().is_empty(), "{status:?}");
        }
    }

    /// A size refusal's `Display` is `InvalidArgument`'s, so log lines and
    /// wrapping messages read like every other invalid argument.
    #[test]
    fn size_refusal_display_equals_invalid_argument_display() {
        let err = ServiceError::delta_too_large(1, 2, 3, 4);
        let ServiceError::InvalidArgumentWithDetail { message, .. } = &err else {
            panic!("expected InvalidArgumentWithDetail, got {err:?}");
        };
        assert_eq!(
            err.to_string(),
            ServiceError::InvalidArgument(message.clone()).to_string()
        );
    }
}
