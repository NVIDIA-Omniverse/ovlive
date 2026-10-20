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

//! Checkpoint-based disaster recovery: durable, out-of-Valkey copies of layer
//! and session state, restored lazily when an RPC finds that state gone.
//! Guarantees, decisions and the data-flow overview: SDD §22.
//!
//! Module map:
//!
//! - **Store surface** — `store` (the `CheckpointStore` trait, whose rustdoc is
//!   the adapter contract, plus value types and `CheckpointError`), `adapters`
//!   (InMemory, LocalDisk, StorageApi) and `builder` (adapter selection; the
//!   `Disabled` no-op). Adapters move opaque bytes and hold no Valkey handle.
//! - **Write side** — `checkpointer` (the leader-elected periodic loop) drives
//!   `assembler` (base snapshot at a proven cut), `tail_classify` /
//!   `tail_capture` / `tail_codec` (tail generations), `chunked_base` (the
//!   base artifact for a layer over one delta's entry limit, and its decoder),
//!   `rebase` (chain collapse), `retention`, the DR-store reconciliation pass (in
//!   `checkpointer`), and the counters sidecar (`counters` is its wire shape);
//!   `base_bootstrap` writes the creation-time base and the post-restore
//!   re-base; `put_guard` skips PUTs for provably dead objects; `ext_mirror`
//!   mirrors external-value blobs within a byte budget, and
//!   `ext_backup_repair` re-sends the copies it never confirmed; `session` / `session_payload` /
//!   `session_writer` write session checkpoints on lifecycle events.
//! - **Read side** — `recovery` (the layer recovery hook) selects a lineage via
//!   `lineage`, composes tails onto their base via `compose`, and materializes
//!   the result via `apply` (with `residue` wiping surviving partial state);
//!   `session_recovery` is the session twin, applying through
//!   `session::apply_session_checkpoint`.
//! - **Startup** — `durability_guard` refuses a configuration with no
//!   durability layer; `counter_seed` seeds the object-id counter from the
//!   counters sidecar (and repairs it lazily after a keyspace loss).

pub mod adapters;
pub mod apply;
pub mod assembler;
pub mod base_bootstrap;
pub mod builder;
pub mod checkpointer;
pub mod chunked_base;
pub mod compose;
pub mod counter_seed;
pub mod counters;
pub mod durability_guard;
pub mod ext_backup_repair;
pub mod ext_mirror;
pub mod lineage;
pub(crate) mod put_guard;
pub mod rebase;
pub mod recovery;
pub(crate) mod residue;
pub mod retention;
pub mod session;
pub mod session_payload;
pub mod session_recovery;
pub mod session_writer;
pub mod store;
pub mod tail_capture;
pub mod tail_classify;
pub mod tail_codec;

/// The stuck-pending bound lives in the leaf [`crate::pacing`] module so
/// `observability` can derive the exported gauge from it without an edge back
/// into `checkpoint` (b.ijf). Conceptual ownership stays here, so the
/// canonical path callers use is unchanged: `live_edit::checkpoint::stuck_threshold_secs`.
pub use crate::pacing::stuck_threshold_secs;
/// `pub(crate)`, not `pub`: [`apply::RestorePayload`] is itself `pub(crate)`, and
/// re-exporting a `pub(crate)` item behind a `pub use` is `E0365`. Both Epic-4
/// consumers live in `src/checkpoint/recovery.rs`, so crate visibility is the
/// whole requirement.
pub(crate) use apply::RestorePayload;
pub use apply::{
    apply_checkpoint, AppliedCheckpoint, ApplyOutcome, ExtPrestore,
    RECOVERY_SEAM_STALE_STREAM_UNLINK, RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE,
    RECOVERY_SEAM_STREAM_KEY_TYPE, RESTORE_SEAM_AFTER_EXT_PRESTORE,
    RESTORE_SEAM_BEFORE_STALE_STREAM_UNLINK, RESTORE_SEAM_RESIDUE_WIPE_BEFORE_EXEC,
};
pub use base_bootstrap::{
    BaseBootstrap, BootstrapOutcome, BootstrapTarget, BOOTSTRAP_MAX_INFLIGHT,
};
pub use builder::build_checkpoint_store;
pub use checkpointer::{
    register_checkpoint_pacing_gauges_if_enabled, spawn_checkpoint_loop, CheckpointLoopConfig,
    TailCaptureConfig,
};
pub use compose::{
    compose_from_artifacts, compose_lineage, ComposeError, ComposedLineage, ExtValuePart,
    RecordSite, TailSite,
};
pub use counter_seed::{
    seed_object_id_counter, seed_object_id_counter_with_retry, CheckpointObjectIdCounterRepair,
    SeedError, SeedOutcome, SeedTrigger, COUNTER_SEED_RETRY_DELAY, COUNTER_SEED_RETRY_MAX_ATTEMPTS,
};
pub use counters::CountersSidecar;
pub use durability_guard::{validate_durability_mode, AofProbeOutcome, DurabilityGuardError};
pub use ext_backup_repair::{spawn_ext_backup_repair, ExtBackupRepair, ExtBackupRepairTiming};
pub use ext_mirror::{
    ext_mirror_channel, ext_mirror_channel_for_plan, spawn_ext_mirror_workers, ExtMirrorCopy,
    ExtMirrorItem, ExtMirrorReceiver, ExtMirrorReservation, ExtMirrorSender, ReserveRefusal,
};
pub use lineage::{
    gather_restore_lineage, no_deadline, Exhaustion, GatheredLineage, GatheredTail, LineageBreak,
    LineageSelection, StepDownRecord, NO_DEADLINE_HORIZON,
};
/// `pub`, not `pub(crate)`: SR-9.2's duplicate-leader test drives two
/// CONCURRENT re-bases from the integration-test crate, which two checkpoint
/// loops could not do (they would serialize on `le:checkpoint:leader`). Same
/// reason `capture_layer_tail` is `pub`.
pub use rebase::{rebase_layer_lineage, RebaseDecision, RebaseRefusal};
pub use recovery::{RecoveryHook, RecoveryOutcome};
pub use session_payload::SessionCheckpoint;
pub use session_recovery::{SessionRecoveryHook, SessionRecoveryOutcome};
pub use session_writer::SessionCheckpointWriter;
pub use store::{
    reconcile_grace, CheckpointAdapter, CheckpointBlob, CheckpointError, CheckpointHandle,
    CheckpointKind, CheckpointMeta, CheckpointStore, LayerArtifact, LayerArtifactKind, LiveExtSet,
    ParseCheckpointAdapterError, LAYER_PUBLISH_ARTIFACTS,
};
pub use tail_codec::{
    decode_tail_envelope, encode_tail_envelope, peek_tail_header, TailCodecError, TailEnvelope,
    TailHeader, TailRecord, TailRecordKind, TAIL_ENVELOPE_MAGIC, TAIL_FORMAT_VERSION,
    TAIL_FORMAT_VERSION_OFFSET, TAIL_HEADER_SIZE, TAIL_MAX_SEQUENCE, TAIL_RECORD_COUNT_OFFSET,
    TAIL_RECORD_PREFIX_SIZE,
};

/// Current wall-clock time as milliseconds since the Unix epoch.
///
/// Centralized so the per-tick checkpoint loop, the session-checkpoint writer,
/// and (eventually) the layer-checkpoint assembler share one clamp policy that
/// downstream recovery diagnostics consume. A pre-epoch clock (impossible in
/// practice for any sane container) folds to `0` rather than panicking;
/// `Duration::as_millis()` overflowing `u64` saturates to `u64::MAX` (likewise
/// impossible without a multi-billion-year-future clock, but cheap to be
/// explicit about).
pub(crate) fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}
