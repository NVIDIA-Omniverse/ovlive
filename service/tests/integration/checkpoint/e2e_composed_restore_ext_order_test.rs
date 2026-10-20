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

//! SR-3.4 external-value **ordering** on a composed restore — Task
//! `t2.dfh.yj.b7`, Subtask `t3.dfh.yj.b7.74`.
//!
//! ## The claim, and why it is load-bearing
//!
//! A composed restore pre-stores the tail-carried external values **strictly
//! before** the single `write_object` MULTI/EXEC that materializes the object
//! (`src/checkpoint/apply.rs`, step 4b then step 5). The order is not cosmetic,
//! and the reason is `apply_checkpoint`'s EXISTS short-circuit: once
//! `le:{oid}:obj:hdr` exists, every subsequent recovery attempt for that
//! `object_id` answers `AlreadyPresent` and returns without touching the ext
//! store. Post-write repair is therefore *impossible* — an interruption between
//! the object write and the value store would strand a complete-looking object
//! whose tail-carried values exist nowhere reachable, and no retry could ever
//! notice. Pre-write order downgrades that failure to orphaned
//! content-addressed blobs, which is ordinary GC territory, and the retry
//! re-stores the same content idempotently.
//!
//! This file makes that order observable in two independent forms, and pins the
//! one accepted-by-decision wart it carries:
//!
//! 1. **Happy path, at the store.** A recording decorator over the real
//!    `ValkeyExternalValueStore` `EXISTS`-probes `le:{oid}:obj:hdr` at the
//!    instant `store_values` is entered. The pre-store must have fired at all
//!    (non-vacuity) AND every probe must have read the header ABSENT. An
//!    implementation that wrote the object first fails the second assertion; one
//!    that never pre-stored fails the first.
//! 2. **At the seam.** `RESTORE_SEAM_AFTER_EXT_PRESTORE` is fired once, and the
//!    durable state at that instant is asserted to have the SR-3.4 shape: no
//!    `obj:hdr`, but the values already landed.
//! 3. **The wart.** `store_values` is content-idempotent but
//!    ref-count-INCREMENTING, so a retried composed restore over-counts. That is
//!    an accepted posture (breakdown amendment 2), documented here rather than
//!    silently fixed — see the in-body comment on Test 3.
//!
//! ## Scope fence — do NOT grow this file
//!
//! Task 6 (`t2.dfh.yj.u3`) owns the AC-8 fault-injection **matrix** (injection
//! during gather, during composition, between the pre-store and the write,
//! during the SR-3.6 re-base) plus **retry convergence**, and the AC-3 **ext
//! matrix** (base-referenced-present / base-referenced-missing / inline-only,
//! and the fetch-on-miss surface for an unrecoverable base-referenced value).
//! This file fires the between-pre-store-and-write seam **exactly once**, as
//! evidence that the seam exists and that durable state at that instant has the
//! SR-3.4 shape. It does not enumerate injection points, does not assert that
//! the retry converges, and does not build a mirror-presence matrix. A reader
//! who wants any of those wants Task 6's files.
//!
//! ## Wiring: the harness owns it
//!
//! Nothing here builds a `RecoveryHook` or swaps a service field. The shared
//! harness resolves ONE real `ValkeyExternalValueStore` over its own
//! `TestValkey` and hands that single `Arc` to both `svc.ext_value_store` and
//! the recovery hook (`helpers.rs`: `default_ext_value_store_for_harness` plus
//! `E2eRecoveryHookWiring`), which is exactly what `main.rs` does — so a
//! composed lineage restored through the harness pre-stores its
//! tail-carried values for real, against the same keyspace this file asserts
//! on. The two tests that need something extra on the hook chain it through
//! [`CheckpointHarness::install_recovery_hook`]: Test 1 redirects the hook's
//! ext store (and ONLY the hook's) through the recording decorator, Test 2 adds
//! the one-shot fault injector. Test 3 needs no install at all.
//!
//! The wiring claims themselves — that the hook and the service hold the same
//! store, that the harness hook carries what `main.rs` carries — are asserted
//! in `harness_parity_test.rs` and are deliberately not re-argued here.
//!
//! ## Synchronization budget
//!
//! No `tokio::time::sleep` is used for coordination anywhere in this file. The
//! two waits are:
//!
//! | Wait | Bound | Why that number |
//! |---|---|---|
//! | first base lands | [`LOOP_BASE_DEADLINE_SECS`] (bounded poll on the store listing) | the loop ticks at [`TICK_SECS`]; `wait_for_checkpoint_at_least` additionally clamps to its own load-tolerant floor |
//! | loop task drains after `cancel()` | [`LOOP_DRAIN_BUDGET`] | a cancelled loop returns at its next `select!`; the budget only stops a wedged one from hanging the binary |
//!
//! ## Tail flag
//!
//! Every harness here is built through [`setup_checkpoint_harness_tails_off`] —
//! `LIVE_EDIT_CHECKPOINT_TAIL_ENABLED` pinned explicitly OFF, not inherited from
//! the binary default (`true` since b.3n1). The subject is the SR-3.4 PRE-STORE:
//! ordering against the object write, durable state at the interruption instant,
//! and the per-restore ext `ref_count` DELTA. With the flag on, a successful
//! restore additionally spawns the SR-3.6 re-base, which is itself an ext-aware
//! writer (`rebase_ext_backfill_test`) — so the count observed after a restore
//! stops being "what the restore's pre-store did" and becomes a sum over two
//! writers. Test 3's delta assertion is the one that says so out loud: it must
//! stay a statement about the pre-store, not an absorbed total.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; see CLAUDE.md.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface};
use tonic::Request;

// The seam constant exists only to be driven, and the only driver in this file
// is the feature-gated Test 2 — so the import carries the same gate as its one
// consumer rather than an `#[allow(unused_imports)]` that would also hide a
// genuinely dead import later.
#[cfg(feature = "test-checkpoint-fault-injection")]
use live_edit::checkpoint::RESTORE_SEAM_AFTER_EXT_PRESTORE;
use live_edit::error::ServiceError;
use live_edit::grpc::layer::{handle_create_layer, handle_stat_layer, handle_update_layer};
use live_edit::grpc::proto::StatLayerRequest;
use live_edit::grpc::BearerToken;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::external_values::{
    ext_content_hash, ExtRelease, ExternalValueStore, GcBatchResult,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::session::SessionStore;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, SetField, StructCommand};

use super::helpers::*;
// Reused rather than re-implemented, deliberately: the surgical ext-namespace
// wipe is a correctness contract (explicit hashes only — a wildcard would also
// catch the `le:ext:{gc}:*` bookkeeping sets and change what the test means),
// and one home for it is what keeps the two files from drifting apart on it.
// It is `pub(super)` in that module precisely so siblings can consume it; this
// is a read-only dependency and does not modify that file.
use super::e2e_ext_value_recovery_test::wipe_ext_namespace_for_hashes;
use crate::common::owned_delta::delta_to_ompf_bytes;

// ─────────────────────────────────────────────────────────────────────────────
// Named waits and fixture constants — see the module header's budget table
// ─────────────────────────────────────────────────────────────────────────────

/// Checkpoint-loop tick cadence. `1` is the value every e2e recovery file in
/// this module uses: the harness sets `interval = quiesce = 0`, so a
/// just-written layer is due on the very next tick.
const TICK_SECS: u64 = 1;

/// Deadline handed to `wait_for_checkpoint_at_least`. The helper clamps it up
/// to its own load-tolerant floor, so this is the *lower* bound on how long a
/// contended Valkey may take before the wait is called a failure.
const LOOP_BASE_DEADLINE_SECS: u64 = 10;

/// Bound on the post-`cancel()` drain of the checkpoint loop's `JoinHandle`.
/// The loop returns at its next `select!` on the cancellation token, so this is
/// never the wait actually observed — it exists so a wedged loop fails the test
/// promptly instead of hanging the binary.
const LOOP_DRAIN_BUDGET: Duration = Duration::from_secs(5);

/// `client_id` on every UpdateLayer header. Only needs to be stable and
/// non-empty; self-echo suppression is not under test.
const CLIENT_ID: &str = "ext-order-client";

const SESSION_BASE_PATH: &str = "/stage";
const CREATED_BY: &str = "ext-order-user";

/// The section every delta in this file writes into.
const ROOT_SECTION_ID: u64 = 42;

/// Section minted by the first tail-range commit, so the replayed range carries
/// structural work and not only field writes.
const TAIL_SECTION_ID: u64 = 1_000;

/// Authoring rank stamped on every `SetField`. Every field in this file is
/// written exactly once, so the value only has to be stable.
const AUTHORING_RANK: u64 = 0;

/// Byte length of the external value the tail carries. Large enough that the
/// blob is unambiguously an out-of-band OMPF part rather than something that
/// could be mistaken for an inline value, small enough to stay free.
const EXT_BLOB_LEN: usize = 4 * 1024;

/// The one-based OMPF part index the ext-carrying `SetField` points at. Part 0
/// is the OODF delta; `delta_to_ompf_bytes(delta, &[blob])` puts `blob` at
/// Part 1.
const EXT_VALUE_INDEX: u32 = 1;

// ─────────────────────────────────────────────────────────────────────────────
// Payload builders
// ─────────────────────────────────────────────────────────────────────────────

/// CreateLayer payload: one section plus one inline field, so the loop's base
/// carries real state.
fn populated_initial_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_string(),
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: ROOT_SECTION_ID,
            section_name: "ext-order-root".to_string(),
            section_type: 1,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: "initial-field".to_string(),
            value_or_ext_hash: vec![0xAA, 0xBB, 0xCC],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// A pre-base commit: one new inline field on the root section.
///
/// `base_version: 1` (any non-zero) marks this an INCREMENTAL diff. With
/// `base_version == 0` and `is_diff == true` the domain engine takes the reset
/// branch and wipes prior state, which would empty the fixture.
fn pre_base_commit_delta(index: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: format!("pre-base-field-{index}"),
            value_or_ext_hash: vec![0x10 + index as u8, 0x11, 0x12],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// The first tail-range commit: a new section plus an inline field inside it.
/// Structural work in the replayed range, carrying no external value — so the
/// ext set the pre-store receives is provably sourced from the OTHER commit.
fn tail_structural_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: ROOT_SECTION_ID,
            section_id: TAIL_SECTION_ID,
            section_name: "ext-order-tail-section".to_string(),
            section_type: 2,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: TAIL_SECTION_ID,
            key_name: "tail-inline-field".to_string(),
            value_or_ext_hash: vec![0xC0, 0xDE, 0x01],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// The second tail-range commit: ONE external-value `SetField`.
///
/// This is what puts the value bytes inside the tail record's OMPF Parts 1..N,
/// which is the only thing SR-3.4's pre-store has to work with:
/// `compose_from_artifacts` collects `(hash, Bytes)` pairs off exactly these
/// parts into `ComposedLineage::ext_values`, and the pre-store writes that list
/// through `ExternalValueStore::store_values`.
fn tail_ext_value_delta(hash: &[u8; 32], blob_len: usize) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: TAIL_SECTION_ID,
            key_name: "tail-ext-field".to_string(),
            value_or_ext_hash: hash.to_vec(),
            ext_value_size: blob_len as u64,
            ext_value_index: EXT_VALUE_INDEX,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// Deterministic external-value payload. An RNG would make a failing run
/// unreproducible for no benefit — the bytes only have to be non-trivial and
/// stable so `ext_content_hash` of them is the fixture's own contract.
fn ext_blob() -> Vec<u8> {
    (0..EXT_BLOB_LEN).map(|i| (i % 251) as u8).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// The recording decorator
// ─────────────────────────────────────────────────────────────────────────────

/// One observation of a `store_values` call: what the object header looked like
/// at the instant the pre-store entered, and which hashes it carried.
#[derive(Debug, Clone)]
struct StoreValuesProbe {
    /// `EXISTS le:{object_id}:obj:hdr`, read BEFORE delegating.
    header_present: bool,
    hashes: Vec<[u8; 32]>,
}

/// A newtype over a real [`ExternalValueStore`] that records every
/// `store_values` call together with an `EXISTS le:{oid}:obj:hdr` probe taken
/// immediately before the delegation.
///
/// **Every trait method is forwarded, including the defaulted
/// `read_batch_with_recovery_counter`.** The warning on
/// `helpers::DelegatingCheckpointStore` applies in spirit and in letter: an
/// omitted arm on a *defaulted* method is not a missing pass-through, it is a
/// silent behaviour substitution — `read_batch_with_recovery_counter`'s default
/// body drops the SR-3.2 recovery counter on the floor, so a decorator that
/// forgot it would quietly disable the counter for every caller under test.
struct RecordingExtValueStore {
    inner: Arc<dyn ExternalValueStore>,
    client: ValkeyClient,
    /// Fully-qualified `le:{object_id}:obj:hdr` for the object under test.
    header_key: String,
    probes: Mutex<Vec<StoreValuesProbe>>,
}

impl RecordingExtValueStore {
    fn new(inner: Arc<dyn ExternalValueStore>, client: ValkeyClient, header_key: String) -> Self {
        Self {
            inner,
            client,
            header_key,
            probes: Mutex::new(Vec::new()),
        }
    }

    /// Snapshot of everything recorded so far, in call order.
    fn probes(&self) -> Vec<StoreValuesProbe> {
        self.probes
            .lock()
            .expect("probe log mutex is never poisoned — no panic runs while it is held")
            .clone()
    }
}

#[async_trait::async_trait]
impl ExternalValueStore for RecordingExtValueStore {
    async fn store_values(&self, values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
        // The probe. Read BEFORE delegating, so a `true` here would mean the
        // object was already materialized when the pre-store ran — the exact
        // ordering inversion this file exists to exclude.
        let header_present: bool = self
            .client
            .pool
            .exists(&self.header_key)
            .await
            .expect("EXISTS obj:hdr must succeed while probing the pre-store ordering");
        let probe = StoreValuesProbe {
            header_present,
            hashes: values.iter().map(|(hash, _)| **hash).collect(),
        };
        // Scoped so the std mutex is never held across the `.await` below.
        {
            self.probes
                .lock()
                .expect("probe log mutex is never poisoned")
                .push(probe);
        }
        self.inner.store_values(values).await
    }

    // Forwarded, not probed: the probe log is the restore pre-store's
    // `store_values` calls, and only the live upload path stores verified.
    async fn store_verified_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), ServiceError> {
        self.inner.store_verified_values(values).await
    }

    fn ext_value_verify(&self) -> live_edit::storage::external_values::ExtValueVerify<'_> {
        self.inner.ext_value_verify()
    }

    async fn load_values(
        &self,
        hashes: &[&[u8; 32]],
    ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
        self.inner.load_values(hashes).await
    }

    async fn read_batch(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        self.inner.read_batch(hashes).await
    }

    // Forwarded per the type's doc comment: this method has a TRAIT DEFAULT that
    // drops the recovery counter, so omitting the arm would substitute
    // behaviour rather than pass through.
    async fn read_batch_with_recovery_counter(
        &self,
        hashes: &[[u8; 32]],
        recovery_counter: Option<&std::sync::atomic::AtomicUsize>,
    ) -> Result<HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        self.inner
            .read_batch_with_recovery_counter(hashes, recovery_counter)
            .await
    }

    async fn probe_values(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<live_edit::storage::external_values::ExtValueProbe, ServiceError> {
        self.inner.probe_values(hashes).await
    }

    async fn enqueue_retirement_attempt(
        &self,
        object_id: &str,
        attempt: &str,
        releases: &[ExtRelease],
        send_by: std::time::Instant,
    ) -> Result<live_edit::storage::external_values::ReleaseAttemptOutcome, ServiceError> {
        self.inner
            .enqueue_retirement_attempt(object_id, attempt, releases, send_by)
            .await
    }

    async fn process_pending(
        &self,
        policy: &live_edit::storage::external_values::GcPolicy,
        state: &mut live_edit::storage::external_values::GcPassState,
    ) -> Result<GcBatchResult, ServiceError> {
        self.inner.process_pending(policy, state).await
    }

    fn ext_release_gate(
        &self,
    ) -> Option<&std::sync::Arc<live_edit::storage::ext_release_gate::ExtReleaseGate>> {
        self.inner.ext_release_gate()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Drive + read-back helpers (the wiring lives in the harness — see the header)
// ─────────────────────────────────────────────────────────────────────────────

/// A `StatLayerRequest` carrying the stub bearer token the harness's
/// `AllowAllPermissions` / `StubUserinfo` wiring expects.
fn stat_request(session_id: &str, layer_name: &str) -> Request<StatLayerRequest> {
    let mut request = Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// Issue one `UpdateLayer` and return `(sequence_num, the OMPF bytes it sent)`.
///
/// Returning the sent bytes is the point: they are what
/// `put_tail_generation_from_records` encodes into the tail envelope, so the
/// artifact carries exactly what the live commit carried — including, for the
/// ext-value commit, the blob as OMPF Part 1.
async fn drive_commit(
    harness: &CheckpointHarness,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    delta: &OwnedDelta,
    external_values: &[&[u8]],
    client_timestamp: u64,
) -> (u64, Bytes) {
    let ompf = delta_to_ompf_bytes(delta, external_values);
    let stream = streaming_update(vec![
        make_update_header(
            session_id,
            layer_name,
            object_id,
            client_timestamp,
            CLIENT_ID,
        ),
        make_update_data(ompf.clone()),
    ]);
    let sequence_num = handle_update_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("UpdateLayer must succeed while building the lineage fixture")
        .into_inner()
        .sequence_num;
    (sequence_num, Bytes::from(ompf))
}

/// `EXISTS` on a fully-qualified key.
async fn key_exists(harness: &CheckpointHarness, key: &str) -> bool {
    harness
        .tv
        .client()
        .pool
        .exists(key)
        .await
        .expect("EXISTS must succeed against the live test Valkey")
}

/// `HGET le:ext:meta:{hash_hex} ref_count`, parsed. Panics when the field is
/// absent — every caller has just asserted the meta key exists, so an absent
/// `ref_count` means the shape of `store_values`' per-hash `ext_store_value.lua`
/// script (`SET data NX` + `HINCRBY meta ref_count 1`) moved, which is exactly
/// what Test 3 characterizes.
async fn ext_ref_count(harness: &CheckpointHarness, hash_hex: &str) -> i64 {
    let meta_key = harness.tv.key_schema().ext_meta(hash_hex);
    let raw: Option<String> = harness
        .tv
        .client()
        .pool
        .hget(&meta_key, "ref_count")
        .await
        .expect("HGET ext_meta ref_count must succeed");
    let raw = raw.unwrap_or_else(|| {
        panic!("`{meta_key}` MUST carry a `ref_count` field after a store_values landed")
    });
    raw.parse::<i64>()
        .unwrap_or_else(|e| panic!("ext_meta ref_count MUST parse as an integer; got {raw:?}: {e}"))
}

/// Assert the object's `obj:hdr` is absent — the half of the SR-3.4 durable
/// pair that only the seam test can observe (on the happy path the object is
/// materialized by the time the restore returns), so this carries the same gate
/// as its one consumer.
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn assert_object_absent(harness: &CheckpointHarness, object_id: u64, context: &str) {
    let header_key = harness.tv.key_schema().obj_header(&object_id.to_string());
    assert!(
        !key_exists(harness, &header_key).await,
        "{context}: `{header_key}` MUST be absent. Once it exists, apply_checkpoint's EXISTS \
         short-circuit answers AlreadyPresent for every later attempt and no retry can ever \
         repair the external values — which is the whole reason the pre-store runs first"
    );
}

/// Assert both content-addressed keys for `hash_hex` exist.
async fn assert_ext_keys_present(harness: &CheckpointHarness, hash_hex: &str, context: &str) {
    let meta_key = harness.tv.key_schema().ext_meta(hash_hex);
    let data_key = harness.tv.key_schema().ext_data(hash_hex);
    assert!(
        key_exists(harness, &meta_key).await,
        "{context}: `{meta_key}` MUST exist"
    );
    assert!(
        key_exists(harness, &data_key).await,
        "{context}: `{data_key}` MUST exist"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The fixture: a real-commit lineage whose tail carries an external value
// ─────────────────────────────────────────────────────────────────────────────

/// Everything the three tests share.
struct ExtLineageFixture {
    session_id: String,
    layer_name: String,
    object_id: u64,
    // `S0` (the loop-written base's `snap_seq`) is deliberately NOT carried out
    // of the fixture: it is asserted against the committed sequence inside
    // `setup_ext_lineage_fixture`, and no test here reasons about sequence
    // numbers — that is `e2e_composed_restore_test`'s subject, not this file's.
    /// The content hash of the external value the tail-range commit carried,
    /// and its hex spelling (the ext key-schema input).
    ext_hash: [u8; 32],
    ext_hash_hex: String,
    /// The blob bytes, kept for the byte-identical readback assertion.
    ext_blob: Vec<u8>,
}

/// CreateSession → CreateLayer → two pre-base commits → wait for the loop's
/// base → **fence the loop** → two tail-range commits, the second carrying an
/// external value → hand-put the tail from the exact OMPF bytes those commits
/// sent.
///
/// The fence (`cancel()` + a bounded join) is the shape
/// `e2e_composed_restore_test` uses: after it, no tick exists to land a
/// generation, so retention cannot prune the fixture and no fresh base can
/// appear at a sequence the hand-put tail does not link onto.
///
/// The base is written BEFORE the ext-carrying commit on purpose. The assembler
/// serializes base OMPFs with zero external parts, so a base written after it
/// would carry the field but not the bytes; keeping the ext commit strictly
/// inside the replayed range is what makes `ComposedLineage::ext_values`
/// non-empty and the pre-store reachable at all.
async fn setup_ext_lineage_fixture(
    harness: &mut CheckpointHarness,
    name: &str,
) -> ExtLineageFixture {
    let session_store =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
    let session_id = session_store
        .create(&format!("{name}-session"), SESSION_BASE_PATH, CREATED_BY)
        .await
        .expect("session creation must succeed");
    let layer_name = format!("{name}-layer");

    let ompf = delta_to_ompf_bytes(&populated_initial_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, &layer_name),
        make_create_data(ompf),
    ]);
    let object_id = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed while building the lineage fixture")
        .into_inner()
        .object_id;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );

    let mut committed_seq = 0;
    for index in 0..2 {
        let (seq, _) = drive_commit(
            harness,
            &session_id,
            &layer_name,
            object_id,
            &pre_base_commit_delta(index),
            &[],
            1_000 + index,
        )
        .await;
        committed_seq = seq;
    }

    // Gate on the committed sequence, not merely "any base": with
    // `interval = quiesce = 0` the loop can fire between CreateLayer and the
    // commits and land a generation that predates them, which would leave `S0`
    // non-deterministic and the tail's link arithmetic meaningless.
    let base_handle = wait_for_checkpoint_at_least(
        &harness.store,
        object_id,
        committed_seq,
        LOOP_BASE_DEADLINE_SECS,
    )
    .await;

    // ── The fence ────────────────────────────────────────────────────────
    harness.cancel.cancel();
    let loop_task = std::mem::replace(&mut harness.loop_handle, tokio::spawn(async {}));
    let _ = tokio::time::timeout(LOOP_DRAIN_BUDGET, loop_task).await;

    let s0 = base_handle.snap_seq;
    assert_eq!(
        s0, committed_seq,
        "fixture precondition: the loop's base MUST have been assembled at the layer's live \
         sequence ({committed_seq}). A higher value means a commit landed that this fixture does \
         not know about, and the hand-put tail's `from_seq = S0` link would name a predecessor \
         covering commits nothing replays"
    );

    // ── The replayed range: structural commit, then the ext-value commit ──
    let blob = ext_blob();
    let ext_hash = ext_content_hash(&blob);
    let ext_hash_hex = sha256_to_hex(&ext_hash);

    let structural = drive_commit(
        harness,
        &session_id,
        &layer_name,
        object_id,
        &tail_structural_delta(),
        &[],
        2_000,
    )
    .await;
    assert_eq!(
        structural.0,
        s0 + 1,
        "fixture precondition: the first tail-range commit must land at S0 + 1; a gap would be \
         rejected downstream as a RecordSeqGap and the test would assert the step-down path"
    );

    let ext_commit = drive_commit(
        harness,
        &session_id,
        &layer_name,
        object_id,
        &tail_ext_value_delta(&ext_hash, blob.len()),
        &[&blob],
        2_001,
    )
    .await;
    assert_eq!(
        ext_commit.0,
        s0 + 2,
        "fixture precondition: commit sequences must be contiguous"
    );

    // The live UpdateLayer path pre-flighted the same value through
    // `store_external_values`, so the ext keys exist right now. Every test
    // wipes them before restoring — otherwise a post-restore presence assertion
    // would pass on this residue and prove nothing about the pre-store.
    assert_ext_keys_present(harness, &ext_hash_hex, "fixture precondition").await;

    let records = vec![structural, ext_commit];
    put_tail_generation_from_records(
        &harness.store,
        object_id,
        &session_id,
        &layer_name,
        s0,
        s0,
        &records,
        HAND_PUT_WRITTEN_AT_UNIX_MS,
        0,
    )
    .await;

    ExtLineageFixture {
        session_id,
        layer_name,
        object_id,
        ext_hash,
        ext_hash_hex,
        ext_blob: blob,
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 1 — the ordering claim, observed at the store
// ═════════════════════════════════════════════════════════════════════════════

/// SR-3.4: on a composed restore the external-value pre-store runs, and runs
/// while `le:{oid}:obj:hdr` is still ABSENT.
///
/// Three independent assertions, each excluding a different regression:
///
/// - **Non-vacuity** — at least one recorded `store_values` carried the
///   tail-borne hash. An implementation that never pre-stored (or that lost the
///   ext set somewhere between composition and apply) fails here, and without
///   this line the ordering assertion below would be vacuously true.
/// - **Order** — EVERY recorded probe read the header absent. An implementation
///   that wrote the object first fails here. This is "strictly before the single
///   `write_object`" made observable on the happy path, with no dependence on
///   abort semantics or on any fault seam.
/// - **Reachability** — after the restore, both content-addressed keys exist and
///   `read_batch` returns the original bytes byte-identically. Storing a value
///   nobody can read back would satisfy the first two and still lose the data,
///   and the ext namespace is wiped beforehand so this cannot pass on the live
///   upload's residue.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ext_pre_store_runs_before_the_object_write_on_a_composed_restore() {
    let mut harness = setup_checkpoint_harness_tails_off(TICK_SECS).await;
    let fixture = setup_ext_lineage_fixture(&mut harness, "ext-order").await;

    // The hook's ext store is the recording decorator, wrapping the harness's
    // own store so both write the same `le:ext:*` keyspace. The decorator is
    // installed ONLY on the hook, never on the service, so the probe log
    // contains restore-time pre-stores and nothing else — the fixture's live
    // upload cannot pollute it.
    let recorder = Arc::new(RecordingExtValueStore::new(
        Arc::clone(&harness.svc.ext_value_store),
        harness.tv.client().clone(),
        harness
            .tv
            .key_schema()
            .obj_header(&fixture.object_id.to_string()),
    ));
    // The chained `with_ext_value_store` is the ONE hook override in this file
    // and it is deliberately narrow: `install_recovery_hook` rebuilds from the
    // harness's own store / layer store / config / metrics / base bootstrap, so
    // only the ext store is redirected and nothing else can drift.
    harness.install_recovery_hook(|hook| {
        hook.with_ext_value_store(Arc::clone(&recorder) as Arc<dyn ExternalValueStore>)
    });

    // Wipe BOTH surfaces: the object keys (so recovery fires at all) and the
    // ext namespace for the recorded hash (surgical, explicit hash only — a
    // wildcard would also catch the `le:ext:{gc}:*` bookkeeping sets).
    wipe_layer_state(&harness.tv, fixture.object_id).await;
    wipe_ext_namespace_for_hashes(&harness.tv, std::slice::from_ref(&fixture.ext_hash_hex)).await;
    assert!(
        !key_exists(
            &harness,
            &harness.tv.key_schema().ext_data(&fixture.ext_hash_hex)
        )
        .await,
        "wipe precondition: the ext data key must be gone before the restore, or the \
         post-restore presence assertion would pass on the live upload's residue"
    );

    let restored_pre = layer_recovery_metric(&harness, "restored");
    let error_pre = layer_recovery_metric(&harness, "error");
    let composed_pre = read_restore_lineage_counter(&harness.metrics, "gathered_composed");

    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "StatLayer after the wipe MUST restore the composed lineage and return Ok, got \
             {:?}: {}. An UNAVAILABLE with a `no ExternalValueStore was wired` message means the \
             recorder install above did not take",
            status.code(),
            status.message()
        )
    });

    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        1,
        "exactly ONE `restored` tick"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        0,
        "`error` MUST stay flat — a tick here means the restore failed rather than pre-storing"
    );
    assert_eq!(
        read_restore_lineage_counter(&harness.metrics, "gathered_composed") - composed_pre,
        1,
        "the lineage MUST resolve COMPOSED. A base-only restore carries no external values at \
         all (the assembler serializes base OMPFs with zero ext parts), so every assertion below \
         would be vacuous on one"
    );

    // ── Non-vacuity ──────────────────────────────────────────────────────
    let probes = recorder.probes();
    assert!(
        probes
            .iter()
            .any(|probe| probe.hashes.contains(&fixture.ext_hash)),
        "the SR-3.4 pre-store MUST have run and MUST have carried the tail-borne hash {}. \
         Recorded calls: {:?}. Without this the ordering assertion below is vacuously true",
        fixture.ext_hash_hex,
        probes
            .iter()
            .map(|probe| probe.hashes.iter().map(sha256_to_hex).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    );

    // ── Order ────────────────────────────────────────────────────────────
    for (index, probe) in probes.iter().enumerate() {
        assert!(
            !probe.header_present,
            "store_values call {index} observed `obj:hdr` ALREADY PRESENT. The pre-store must \
             run strictly BEFORE the single write_object MULTI/EXEC: once the header exists, \
             apply_checkpoint's EXISTS short-circuit makes every later attempt answer \
             AlreadyPresent, so a value that failed to land here can never be repaired"
        );
    }

    // ── Reachability ─────────────────────────────────────────────────────
    assert_ext_keys_present(
        &harness,
        &fixture.ext_hash_hex,
        "after the composed restore",
    )
    .await;
    let loaded = harness
        .svc
        .ext_value_store
        .read_batch(&[fixture.ext_hash])
        .await
        .expect("read_batch of the pre-stored hash must succeed after the restore");
    let bytes = loaded.get(&fixture.ext_hash).unwrap_or_else(|| {
        panic!(
            "read_batch MUST return the tail-carried hash {}; got {} entries",
            fixture.ext_hash_hex,
            loaded.len()
        )
    });
    assert_eq!(
        bytes.as_slice(),
        fixture.ext_blob.as_slice(),
        "the pre-stored value MUST come back byte-identical to what the live commit uploaded — \
         storing bytes nobody can read back loses the data just as thoroughly as not storing them"
    );

    // The object itself landed, after the values.
    let header_key = harness
        .tv
        .key_schema()
        .obj_header(&fixture.object_id.to_string());
    assert!(
        key_exists(&harness, &header_key).await,
        "the restore MUST have materialized the object after pre-storing the values"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 2 — the seam: durable state between the pre-store and the write
// ═════════════════════════════════════════════════════════════════════════════

/// Op-targeted ONE-SHOT fault injector.
///
/// One-shot rather than always-on because this file's contract is "fire the
/// seam exactly once". The matrix over the other injection points, and the
/// retry-convergence claim that a re-armed injector would be needed for, belong
/// to Task 6 (`t2.dfh.yj.u3`, AC-8) and must not be grown here.
///
/// `error_factory` is a closure rather than a stored `CheckpointError` because
/// `Transient` / `Permanent` carry a boxed `Send + Sync` source that is not
/// `Clone`. Mirrors the `OpTargetedFaultInjector` shape in
/// `e2e_ext_value_recovery_test.rs` and `conformance.rs`.
#[cfg(feature = "test-checkpoint-fault-injection")]
struct OneShotSeamInjector {
    target_op: &'static str,
    armed: std::sync::atomic::AtomicBool,
    fired: std::sync::atomic::AtomicUsize,
    error_factory: Box<dyn Fn() -> live_edit::checkpoint::CheckpointError + Send + Sync>,
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl OneShotSeamInjector {
    fn new<EF>(target_op: &'static str, error_factory: EF) -> Self
    where
        EF: Fn() -> live_edit::checkpoint::CheckpointError + Send + Sync + 'static,
    {
        Self {
            target_op,
            armed: std::sync::atomic::AtomicBool::new(true),
            fired: std::sync::atomic::AtomicUsize::new(0),
            error_factory: Box::new(error_factory),
        }
    }

    fn fire_count(&self) -> usize {
        self.fired.load(std::sync::atomic::Ordering::Acquire)
    }
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl live_edit::checkpoint::adapters::CheckpointFaultInjector for OneShotSeamInjector {
    fn maybe_inject(&self, op: &str) -> Result<(), live_edit::checkpoint::CheckpointError> {
        if op == self.target_op && self.armed.swap(false, std::sync::atomic::Ordering::AcqRel) {
            self.fired.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
            return Err((self.error_factory)());
        }
        Ok(())
    }
}

/// Stable marker carried on the injected error's boxed source, so the status
/// message assertion below is pinned to THIS injection rather than to any
/// UNAVAILABLE the restore path could produce for another reason.
#[cfg(feature = "test-checkpoint-fault-injection")]
const SEAM_ERROR_MARKER: &str = "e2e-ext-order-seam-interrupt";

#[cfg(feature = "test-checkpoint-fault-injection")]
#[derive(Debug)]
struct SeamInterrupt;

#[cfg(feature = "test-checkpoint-fault-injection")]
impl std::fmt::Display for SeamInterrupt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{SEAM_ERROR_MARKER}")
    }
}

#[cfg(feature = "test-checkpoint-fault-injection")]
impl std::error::Error for SeamInterrupt {}

/// SR-3.4 / SR-3.5 at the seam: interrupting a composed restore between the ext
/// pre-store and the single `write_object` leaves the values durable and NO
/// object.
///
/// That pair is the whole point of the ordering. The values being present is
/// what makes the failure recoverable — they are content-addressed, so the retry
/// re-stores the identical content; the object being absent is what keeps
/// `apply_checkpoint`'s EXISTS short-circuit from locking in a complete-looking
/// object whose values never landed. Reverse the order and the same interruption
/// produces exactly that unrepairable state.
///
/// The seam is Task 3's own named constant
/// [`RESTORE_SEAM_AFTER_EXT_PRESTORE`], driven through the
/// `CheckpointFaultInjector` trait — this test does not invent a second one, and
/// spelling the label from the constant means it cannot drift.
///
/// Fired exactly ONCE. The AC-8 injection matrix (gather, composition, the
/// SR-3.6 re-base) and the retry-convergence claim are Task 6's
/// (`t2.dfh.yj.u3`) and must not be grown here.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_interrupt_between_the_ext_pre_store_and_the_object_write_leaves_values_but_no_object() {
    use live_edit::checkpoint::adapters::CheckpointFaultInjector;
    use live_edit::checkpoint::CheckpointError;

    let mut harness = setup_checkpoint_harness_tails_off(TICK_SECS).await;
    let fixture = setup_ext_lineage_fixture(&mut harness, "ext-order-seam").await;

    let injector = Arc::new(OneShotSeamInjector::new(
        RESTORE_SEAM_AFTER_EXT_PRESTORE,
        || CheckpointError::Transient {
            source: Box::new(SeamInterrupt),
        },
    ));
    // The injector is the ONLY addition: the rebuilt hook keeps the harness's
    // ext store, so the pre-store this seam interrupts writes the same keyspace
    // the assertions below read back through `harness.svc.ext_value_store`.
    harness.install_recovery_hook(|hook| {
        hook.with_restore_fault_injector_for_test(
            Arc::clone(&injector) as Arc<dyn CheckpointFaultInjector>
        )
    });

    wipe_layer_state(&harness.tv, fixture.object_id).await;
    wipe_ext_namespace_for_hashes(&harness.tv, std::slice::from_ref(&fixture.ext_hash_hex)).await;

    let error_pre = layer_recovery_metric(&harness, "error");
    let restored_pre = layer_recovery_metric(&harness, "restored");
    let not_recoverable_pre = layer_recovery_metric(&harness, "not_recoverable");

    let status = handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .expect_err("a restore interrupted at the SR-3.4 seam MUST surface an error, never Ok");

    // ── Observable failure, specifically classified ──────────────────────
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "the handler maps RecoveryOutcome::Error onto UNAVAILABLE (retriable). Got {:?}: {}",
        status.code(),
        status.message()
    );
    assert!(
        status.message().contains(SEAM_ERROR_MARKER),
        "the surfaced status MUST carry THIS injection's source ({SEAM_ERROR_MARKER}), so the \
         assertion cannot pass on some other UNAVAILABLE the restore path produced; got: {}",
        status.message()
    );
    assert_eq!(
        injector.fire_count(),
        1,
        "the seam MUST fire exactly once — the one-shot latch is what keeps this test from \
         drifting into Task 6's retry-convergence territory"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error") - error_pre,
        1,
        "the interrupted restore MUST land on the `error` recovery-attempt cell exactly once"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored") - restored_pre,
        0,
        "`restored` MUST stay flat — nothing was materialized"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "not_recoverable") - not_recoverable_pre,
        0,
        "`not_recoverable` MUST stay flat — a restorable lineage is on disk; the failure is the \
         injected interruption, not an absent lineage"
    );

    // ── The durable pair at the interruption point ───────────────────────
    assert_object_absent(
        &harness,
        fixture.object_id,
        "interrupted at the SR-3.4 seam",
    )
    .await;
    assert_ext_keys_present(
        &harness,
        &fixture.ext_hash_hex,
        "interrupted at the SR-3.4 seam",
    )
    .await;
    let loaded = harness
        .svc
        .ext_value_store
        .read_batch(&[fixture.ext_hash])
        .await
        .expect("read_batch of the pre-stored hash must succeed after the interruption");
    assert_eq!(
        loaded.get(&fixture.ext_hash).map(Vec::as_slice),
        Some(fixture.ext_blob.as_slice()),
        "the values the pre-store landed before the seam MUST be durable and byte-identical — \
         that is what makes this failure recoverable rather than a silent loss"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 3 — characterization: the accepted ref-count posture
// ═════════════════════════════════════════════════════════════════════════════

/// A retried composed restore over-increments the external value's `ref_count`,
/// by exactly one per restore.
///
/// This is a CHARACTERIZATION test of an ACCEPTED posture, not a defect report.
/// See the in-body comment for the full statement; the short version is that
/// `store_values` is content-idempotent but ref-count-INCREMENTING, the live
/// upload path has the identical property on any retried upload, and the
/// consequence is a blob that is never GC'd — leaked, not lost.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_retried_composed_restore_over_increments_ext_ref_counts() {
    let mut harness = setup_checkpoint_harness_tails_off(TICK_SECS).await;
    let fixture = setup_ext_lineage_fixture(&mut harness, "ext-order-refcount").await;
    // No hook install: this test needs nothing beyond the ext store the stock
    // harness already wires into `svc.recovery`, which is the same `Arc` the
    // service writes through.

    // First restore. The ext keys are deliberately NOT wiped here: this test's
    // subject is what a SECOND pre-store over the SAME content does, and the
    // live upload's ref_count is the honest starting point (it is what a real
    // deployment's restore would find).
    wipe_layer_state(&harness.tv, fixture.object_id).await;
    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .expect("the first composed restore must succeed");

    let after_first = ext_ref_count(&harness, &fixture.ext_hash_hex).await;

    // Wipe the OBJECT keys only, leaving the ext namespace intact — the shape a
    // second recovery attempt over the same lineage sees.
    wipe_layer_state(&harness.tv, fixture.object_id).await;
    assert_ext_keys_present(
        &harness,
        &fixture.ext_hash_hex,
        "between the two restores the ext keys must survive",
    )
    .await;

    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .expect("the retried composed restore must succeed");

    let after_second = ext_ref_count(&harness, &fixture.ext_hash_hex).await;

    // ── ACCEPTED POSTURE — do NOT "fix" this test into asserting idempotence.
    //
    // `ValkeyExternalValueStore::store_values` writes each hash with one
    // `EVALSHA` of `ext_store_value.lua` (`SET data NX` + `HINCRBY meta
    // ref_count 1`). It is content-idempotent (a copy already stored under the
    // content-addressed key is kept) but ref-count-INCREMENTING, so a composed restore
    // that runs twice over the same lineage counts the tail-carried value
    // twice. The reference the count is supposed to track — the field in the
    // restored object — exists once.
    //
    // This is a KNOWN, PRE-EXISTING property, adjudicated as accepted
    // (breakdown amendment 2), and it is the SAME property the live upload path
    // has: any client that retries an UpdateLayer carrying an external value
    // increments the count identically. The consequence is an over-counted blob
    // that GC will never reclaim — leaked, never lost — which is strictly the
    // safer direction of the two errors available here. Under-counting would
    // let GC delete a value a live object still references.
    //
    // A future reader who "fixes" this assertion into `after_second ==
    // after_first` will be encoding a behaviour the code does not have, and the
    // failing test that results is the accepted wart drifting into an
    // unaccepted one with nobody noticing. If the posture is ever revisited,
    // change the CODE first and this test second.
    assert_eq!(
        after_second - after_first,
        1,
        "each composed restore MUST add exactly one ref_count to the tail-carried value \
         (accepted posture, see the comment above): first restore left {after_first}, the \
         retried restore left {after_second}. A delta of 0 means the increment stopped \
         happening — a real behaviour change that must be adjudicated, not absorbed. A delta \
         above 1 means the pre-store ran more than once per restore"
    );
    // The other half of the characterization, and the reason the over-count is
    // survivable: the SECOND pre-store is content-idempotent. It re-`SET`s the
    // same bytes under the same content-addressed key, so the value a reader
    // gets after the retry is the original one — the drift is confined to the
    // count, never to the content.
    let loaded = harness
        .svc
        .ext_value_store
        .read_batch(&[fixture.ext_hash])
        .await
        .expect("read_batch after the retried restore must succeed");
    assert_eq!(
        loaded.get(&fixture.ext_hash).map(Vec::as_slice),
        Some(fixture.ext_blob.as_slice()),
        "the retried pre-store MUST leave the value byte-identical — the accepted wart is a \
         ref_count over-count and NOTHING else; a content change here would be a real defect"
    );
}
