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

//! The restore writer's object writes end to end (`live_edit::dr::job::execute`;
//! SDD §22.22 *The restore writer*, SR-9.5, SR-9.6, SR-9.10): the restore Job
//! run through its frame against a real store and a scripted backup store,
//! with the writer's test seams (kill points, the batch size), and what it
//! left read back.
//!
//! - **W1, a full wipe**: every layer's content equals its backup's, every
//!   version at the seeded `seq`, the header last; the summary line; after
//!   the hold lifts, ListSessions lists each session and each layer resolves.
//!   (This is where the I1 plan test's writes are checked.)
//! - **W2, a layer over one delta's entry limit** (a chunked base): its rows
//!   in several batches, each within the production command limit (its
//!   50,000-child list cut across batches), each with the timeout its
//!   commands earn, and its content equal to its backup's.
//! - **W3, the band**: a plausible high-water raises the floor; a `deleted`
//!   top, no stream and a foreign value at the stream key do not; in every
//!   case the stream key is gone afterwards and the cursor columns hold the
//!   restored head and the band.
//! - **W4, a kill at each step of a restore converges**: the kill leaves
//!   exactly the steps before it done (checked at each point), and the
//!   re-run writes what an uninterrupted run writes, banded above the
//!   surviving high-water (PM r2 N4), with the tail-carried value in the
//!   backup store.
//! - **W5, values only the tails carry** are stored in Valkey and PUT to the
//!   backup store. The planner settles a tail-carried value without probing
//!   the backup store (the tail that carries it is its source), so none is
//!   probed. Each PUT is recorded as the run's own, so a value two restored
//!   layers carry is PUT once.
//! - **W6, a flag clear**: the stream, the retirement carrier and the flags
//!   go, `seq` is banded and `restore_seq_bump` accumulates; an `UpdateLayer`
//!   commits after the hold lifts; a kill at each step leaves the flags and
//!   the stream (checked at each point), and the re-run converges.
//! - **W8, no releases**: a replaced survivor's external-value index is
//!   rewritten from the backup, with no release queued or counted.
//! - **W12**: an object held with no session or layer writes nothing.
//! - **W13, a layer with no root section** is restored as backed up, with no
//!   root section supplied, and then accepts a structural `UpdateLayer`.
//! - **W13b, a fully empty layer** (no section, field or time sample) is
//!   restored as backed up: still empty, with no root section supplied.
//! - **defer-45**: a backup the plan selected that now reads absent stops
//!   the Job (exit 1) naming the object, before any write to it.
//!
//! Every run uses a per-test key prefix (`TestValkey`).
//!
//! # Time
//!
//! Every test but W2 runs under `writer_support`'s one-test budget (its
//! module docs, *Time*): 90 s plus the store clean-up, inside nextest's
//! 120 s kill. W2 restores a layer of 100,001 entries, so it has budgets of
//! its own and a nextest override (`.config/nextest.toml`, 60 s × 5 =
//! 300 s). Its phases, each with what a healthy debug-build run took alone
//! on the remote build host (the `W2:` lines it prints on stderr), measured
//! 2026-10-20 at 100,000-command batches, before the batches' limit fell to
//! 10,000 (not re-measured since), and its bound:
//!
//! | Phase | Measured (2026-10-20, 100,000-command batches) | Bound |
//! |---|---|---|
//! | building the chunked base, CPU only | 0.2 s | none |
//! | the Job's run: plan, decode twice, 300,009 commands then (300,014 now: five `LTRIM`s) | 30 s | [`BIG_RUN_BUDGET`], 160 s |
//! | the read-back of every row | 16 s | [`BIG_READ_BACK_BUDGET`], 80 s |
//! | `TestValkey`'s clean-up of every key it wrote | 0.8 s | each step by `CLEANUP_STEP_BUDGET` |
//!
//! Every wait at its full budget is 240 s, about five times the measured
//! run and read-back, and leaves 60 s of the 300 s kill for the build and
//! the clean-up (about 1 s measured), so a runner several times slower
//! still fits.

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface, SortedSetsInterface};
use tonic::Request;

use live_edit::checkpoint::chunked_base::encode_chunked_base;
use live_edit::checkpoint::CheckpointStore;
use live_edit::domain::limits::MAX_DELTA_ENTRIES;
use live_edit::domain::types::ObjectState;
use live_edit::dr::job::execute::{BackfillReport, KillPoint, WriteKind};
use live_edit::dr::job::plan::model::{HeldItem, HeldReason};
use live_edit::dr::job::JobExit;
use live_edit::grpc::layer::{handle_stat_layer, handle_update_layer};
use live_edit::grpc::proto::{ListSessionsRequest, StatLayerRequest};
use live_edit::grpc::session::handle_list_sessions;
use live_edit::grpc::BearerToken;
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};

use super::writer_support::*;
use crate::grpc::helpers::{
    build_service, create_session_and_layer, make_request, make_update_data, make_update_header,
    streaming_update, AllowAllPermissions,
};

/// The test configuration's Valkey command timeout, `T`.
const COMMAND_TIMEOUT: Duration = Duration::from_millis(5_000);

/// A transaction's client timeout as `storage::client::txn_timeout` gives
/// it at `T` = [`COMMAND_TIMEOUT`]: `T` plus 120 µs per command (the
/// M1′-measured constant), capped at 45 s. Restated, not imported: the
/// function is crate-private.
fn txn_timeout(commands: u64) -> Duration {
    (COMMAND_TIMEOUT + Duration::from_micros(120) * u32::try_from(commands).expect("fits"))
        .min(Duration::from_secs(45))
}

/// A unary request carrying the test bearer.
fn bearer<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ═════════════════════════════════════════════════════════════════════════════
// W1: a full wipe
// ═════════════════════════════════════════════════════════════════════════════

/// W1 (and the writes of the I1 plan test). After a full wipe, with every
/// session and layer backed up, the Job writes every layer — two base-only,
/// one composed from a base and a tail carrying a value — as its backup
/// decodes, every version at the seeded `seq` (`head + band`, no stream
/// surviving), with its cursor columns, each in one transaction; it
/// restores both sessions; it logs its summary, recounts, and stops just
/// before its un-hold (exit 1, `run_to_the_end`). Once the hold is lifted,
/// ListSessions lists each session under its base path and every layer
/// resolves.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_after_a_full_wipe_writes_every_layer_and_session_as_backed_up() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let (value_hash, value) = carried(b"a value only b1's tail carries");
        let a1 = base(3, 0xA1, &[]);
        let a2 = base(1, 0xA2, &[]);
        let b1 = base(2, 0xB1, &[]);
        let (b1_tail, b1_head) = tail(201, 9, &[(value_hash, value)]);
        put_session(
            store.as_ref(),
            &session_backup("s-a", &[("a1.usda", 101), ("a2.usda", 102)], &[]),
        )
        .await;
        put_session(
            store.as_ref(),
            &session_backup("s-b", &[("b1.usda", 201)], &[]),
        )
        .await;
        put_base(store.as_ref(), 101, "s-a", "a1.usda", 5, a1.clone()).await;
        put_base(store.as_ref(), 102, "s-a", "a2.usda", 7, a2.clone()).await;
        put_base(store.as_ref(), 201, "s-b", "b1.usda", 9, b1.clone()).await;
        put_tail(
            store.as_ref(),
            201,
            "s-b",
            "b1.usda",
            9,
            b1_head,
            b1_tail.clone(),
        )
        .await;
        put_counters(store.as_ref()).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (plan, report, captured) = run_to_the_end(&tv, &config, &store).await;

        /// One restored layer: its object, session, layer, base, base seq,
        /// tails and head.
        type Restored<'a> = (u64, &'a str, &'a str, &'a Bytes, u64, Vec<Bytes>, u64);
        let layers: [Restored<'_>; 3] = [
            (101, "s-a", "a1.usda", &a1, 5, vec![], 5),
            (102, "s-a", "a2.usda", &a2, 7, vec![], 7),
            (
                201,
                "s-b",
                "b1.usda",
                &b1,
                9,
                vec![b1_tail.clone()],
                b1_head,
            ),
        ];
        for (oid, session, layer, base_bytes, base_seq, tails, head) in &layers {
            let (oid, head) = (*oid, *head);
            assert_eq!(restored_lineage(&plan, oid), (head, *base_seq, tails.len()));
            let write = &report.objects[&oid];
            assert_eq!(
                write.kind,
                WriteKind::Restored {
                    head_seq: head,
                    base_seq: *base_seq,
                    tails: tails.len(),
                }
            );
            assert_eq!(
                (write.floor, write.high_water, write.band, write.seeded_seq),
                (head, None, BAND, head + BAND),
                "object {oid}: no stream survives, so the floor is the head"
            );
            assert_eq!(write.batches.len(), 1, "object {oid}: one transaction");
            assert_eq!(
                write.batches[0].timeout,
                txn_timeout(write.batches[0].commands),
                "object {oid}"
            );
            assert_restored(
                &tv,
                oid,
                &expected_state(oid, *base_seq, base_bytes, tails, head + BAND),
            )
            .await;
            let h = header(&tv, oid).await;
            assert_eq!(h["last_checkpoint_seq"], head.to_string(), "object {oid}");
            assert_eq!(h["restore_seq_bump"], BAND.to_string(), "object {oid}");
            assert_eq!(
                (h["session_id"].as_str(), h["layer_name"].as_str()),
                (*session, *layer),
                "object {oid}"
            );
            assert!(!exists(&tv, tv.key_schema().layer_stream(oid)).await);
        }
        assert_eq!(report.sessions_restored, 2);
        assert_eq!(
            report.backfill,
            BackfillReport {
                valkey_values: 1,
                backup_puts: 1,
            }
        );
        assert_eq!(
            lines(
                &captured,
                "the restore Job restored 3 layers in 3 batches, cleared retirement flags on 0, \
                 moved 0 layers to their successor, restored 2 sessions, finished 0 deletes, and \
                 stored 1 values carried only by backups in Valkey and 1 in the backup store"
            )
            .len(),
            1,
            "the summary; capture:\n{}",
            captured.text
        );

        assert!(tv.remove_hold_marker().await, "the hold marker stayed");
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        for (session, layers) in [
            ("s-a", vec!["a1.usda", "a2.usda"]),
            ("s-b", vec!["b1.usda"]),
        ] {
            let listed = handle_list_sessions(
                &svc,
                bearer(ListSessionsRequest {
                    path: base_path_of(session),
                }),
            )
            .await
            .unwrap_or_else(|s| panic!("{session}: ListSessions answers: {s}"))
            .into_inner();
            let ids: Vec<&str> = listed
                .sessions
                .iter()
                .map(|s| s.session_id.as_str())
                .collect();
            assert_eq!(
                ids,
                vec![session],
                "ListSessions under {session}'s base path"
            );
            for layer in layers {
                handle_stat_layer(
                    &svc,
                    bearer(StatLayerRequest {
                        session_id: session.to_owned(),
                        layer_name: layer.to_owned(),
                    }),
                )
                .await
                .unwrap_or_else(|s| panic!("{session}/{layer} resolves: {s}"));
            }
        }
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// W2: a layer over one delta's entry limit
// ═════════════════════════════════════════════════════════════════════════════

/// The attributes of W2's layer: with the prim, `2 × 50,000 + 1` entries,
/// two over one delta's limit.
const BIG_ATTRIBUTES: u64 = 50_000;

/// W2's hang bound on its one Job run, which plans and writes about 300,000
/// commands in a debug build (module docs, *Time*).
const BIG_RUN_BUDGET: Duration = Duration::from_secs(160);

/// W2's hang bound on reading back every row it restored (module docs,
/// *Time*).
const BIG_READ_BACK_BUDGET: Duration = Duration::from_secs(80);

/// A chunked base (LEBC) of the prim `/World` and [`BIG_ATTRIBUTES`]
/// attributes with one inline `default` each, cut into reset deltas of at
/// most `MAX_DELTA_ENTRIES` entries in the plan's order (struct commands,
/// then fields), as the assembler cuts one.
fn big_base() -> Bytes {
    let mut struct_commands = vec![StructCommand::CreateSection(CommandCreateSection {
        parent_id: 0,
        section_id: 1,
        section_name: "World".to_owned(),
        section_type: 6,
    })];
    let mut set_fields = Vec::new();
    for i in 0..BIG_ATTRIBUTES {
        struct_commands.push(StructCommand::CreateSection(CommandCreateSection {
            parent_id: 1,
            section_id: 2 + i,
            section_name: format!("a{i}"),
            section_type: 1,
        }));
        set_fields.push(SetField {
            section_id: 2 + i,
            key_name: "default".to_owned(),
            value_or_ext_hash: i.to_le_bytes().to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: i,
        });
    }
    assert!(
        (struct_commands.len() + set_fields.len()) as u64 > MAX_DELTA_ENTRIES,
        "fixture: over one delta's entry limit"
    );
    let cap = usize::try_from(MAX_DELTA_ENTRIES).expect("fits");
    let in_first = cap - struct_commands.len();
    let rest = set_fields.split_off(in_first);
    let chunk = |struct_commands, set_fields, source_format: &str| {
        oodf_ffi::build_delta(&OwnedDelta {
            is_diff: true,
            base_version: 0,
            struct_commands,
            set_fields,
            time_samples: Vec::new(),
            source_format: source_format.to_owned(),
        })
        .expect("a chunk builds")
    };
    let chunks = vec![
        chunk(struct_commands, set_fields, "usda"),
        chunk(Vec::new(), rest, ""),
    ];
    Bytes::from(encode_chunked_base(&chunks).expect("the chunked base encodes"))
}

/// The production restore batch's command limit,
/// `storage::layer::dr_restore::RESTORE_BATCH_COMMANDS`: restated, not
/// imported (the constant is crate-private), and checked against the
/// restore memory record that derives it where the record is in the tree
/// ([`restore_batch_commands`]).
const RESTORE_BATCH_COMMANDS: u64 = 10_000;

/// The restore memory envelope's record, which derives the restore batch's
/// limits (`writer_batch_limits`).
const RESTORE_ENVELOPE_DERIVATION_JSON: &str =
    "docs-internal/validation/restore-memory-envelope-tv3-20261022/derivation.json";

/// [`RESTORE_BATCH_COMMANDS`], first checked equal to the record's
/// `writer_batch_limits.restore_batch_commands` when the tree has
/// `docs-internal/` (the published source tree does not).
fn restore_batch_commands() -> u64 {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    if root.join("docs-internal").is_dir() {
        let raw = std::fs::read_to_string(root.join(RESTORE_ENVELOPE_DERIVATION_JSON))
            .unwrap_or_else(|e| {
                panic!("the committed record {RESTORE_ENVELOPE_DERIVATION_JSON}: {e}")
            });
        let record: serde_json::Value =
            serde_json::from_str(&raw).expect("the record is valid JSON");
        assert_eq!(
            record
                .pointer("/writer_batch_limits/restore_batch_commands")
                .and_then(serde_json::Value::as_u64),
            Some(RESTORE_BATCH_COMMANDS),
            "the restated command limit is the record's"
        );
    }
    RESTORE_BATCH_COMMANDS
}

/// W2. A layer of 100,001 entries, restorable only from a chunked base: its
/// rows go in several batches, every one within the production command
/// limit of 10,000 with no exemption — its prim's 50,000-child list is cut
/// across batches, its first slice and five later ones, each later slice
/// one `LTRIM` more — the header transaction last; each batch's timeout is
/// `txn_timeout(T, its commands)`; and the restored content, the cut list
/// whole and in order among it, equals the backup's. Logs each phase's
/// duration on stderr (`W2:`), the measurement behind the module docs'
/// *Time*.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_writes_a_layer_over_one_deltas_entry_limit_in_batches_each_with_its_timeout() {
    let tv = TestValkey::setup().await;
    let store = WriterStore::new();
    let started = std::time::Instant::now();
    let big = big_base();
    eprintln!("W2: building the chunked base took {:?}", started.elapsed());
    put_session(
        store.as_ref(),
        &session_backup("s-big", &[("big.usda", 251)], &[]),
    )
    .await;
    put_base(store.as_ref(), 251, "s-big", "big.usda", 40, big.clone()).await;
    put_counters(store.as_ref()).await;
    let config = writer_config();
    hold(&tv, &config).await;

    let started = std::time::Instant::now();
    let (_plan, report, _captured) =
        run_to_the_end_with(BIG_RUN_BUDGET, &tv, &config, &store, |b| b).await;
    eprintln!("W2: the Job's run took {:?}", started.elapsed());

    let write = &report.objects[&251];
    let limit = restore_batch_commands();
    let (rows, header_txn) = write.batches.split_at(write.batches.len() - 1);
    assert!(rows.len() >= 2, "several row batches: {:?}", write.batches);
    for (i, batch) in write.batches.iter().enumerate() {
        assert!(batch.commands <= limit, "batch {i}: {batch:?}");
        assert_eq!(batch.timeout, txn_timeout(batch.commands), "batch {i}");
    }
    assert_eq!(
        header_txn[0].commands, 2,
        "the header transaction: UNLINK + HSET"
    );
    let total: u64 = write.batches.iter().map(|b| b.commands).sum();
    // Per attribute a section (2) and a field (3); the prim's section (2);
    // the prim's children list (50,000 + 2) and the root's (1 + 2); one
    // `LTRIM` per later slice of the prim's list, five (its first slice
    // fills the batch holding the sections' last two commands, then four
    // later slices fill a batch each and a fifth carries the rest, in
    // either order of the two lists); the header and the UNLINK.
    const LTRIMS: u64 = 5;
    assert_eq!(
        total,
        5 * BIG_ATTRIBUTES + 2 + (BIG_ATTRIBUTES + 2) + 3 + LTRIMS + 2
    );
    let started = std::time::Instant::now();
    tokio::time::timeout(
        BIG_READ_BACK_BUDGET,
        assert_restored(&tv, 251, &expected_state(251, 40, &big, &[], 40 + BAND)),
    )
    .await
    .expect("the read-back ends within its budget");
    eprintln!("W2: the read-back took {:?}", started.elapsed());

    let started = std::time::Instant::now();
    drop(tv);
    eprintln!("W2: the store clean-up took {:?}", started.elapsed());
}

// ═════════════════════════════════════════════════════════════════════════════
// W3: the band
// ═════════════════════════════════════════════════════════════════════════════

/// W3. Four restored layers of one session, each with its stream key in a
/// different state: a plausible high-water above the head (the floor is
/// the high-water), a `deleted` tombstone's auto-ID top (the floor is the
/// head), no stream, and a string at the stream key (the floor is the head;
/// a WARN names its type). In every case `seq` and every version are the
/// floor plus the band, `last_checkpoint_seq` is the head,
/// `restore_seq_bump` is the band, and the stream key is gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_bands_above_a_plausible_high_water_only_and_removes_every_stream_key() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let head = 10;
        let layers = [
            ("plausible.usda", 301),
            ("deleted.usda", 302),
            ("nostream.usda", 303),
            ("foreign.usda", 304),
        ];
        put_session(store.as_ref(), &session_backup("s3", &layers, &[])).await;
        let bytes = base(2, 0x30, &[]);
        for (layer, oid) in layers {
            put_base(store.as_ref(), oid, "s3", layer, head, bytes.clone()).await;
        }
        put_counters(store.as_ref()).await;
        xadd(&tv, 301, "40-0").await;
        xadd(&tv, 302, "*").await;
        let _: () = tv
            .client()
            .pool
            .set(
                tv.key_schema().layer_stream(304),
                "foreign",
                None,
                None,
                false,
            )
            .await
            .expect("SET a string at the stream key");
        let config = writer_config();
        hold(&tv, &config).await;

        let (_plan, report, captured) = run_to_the_end(&tv, &config, &store).await;

        for (oid, floor, high_water) in [
            (301, 40, Some(40)),
            (302, head, None),
            (303, head, None),
            (304, head, None),
        ] {
            let write = &report.objects[&oid];
            assert_eq!(
                (write.floor, write.high_water, write.band, write.seeded_seq),
                (floor, high_water, BAND, floor + BAND),
                "object {oid}"
            );
            assert_restored(
                &tv,
                oid,
                &expected_state(oid, head, &bytes, &[], floor + BAND),
            )
            .await;
            let h = header(&tv, oid).await;
            assert_eq!(h["seq"], (floor + BAND).to_string(), "object {oid}");
            assert_eq!(h["last_checkpoint_seq"], head.to_string(), "object {oid}");
            assert_eq!(h["restore_seq_bump"], BAND.to_string(), "object {oid}");
            assert!(
                !exists(&tv, tv.key_schema().layer_stream(oid)).await,
                "object {oid}: the stream key is gone"
            );
        }
        assert_eq!(
            lines(
                &captured,
                "object 304's stream key held a string, not a stream; the restore removes it \
                 with the header write so the restored layer can be written to"
            )
            .len(),
            1,
            "the foreign-stream WARN; capture:\n{}",
            captured.text
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// W4: a kill at each step of a restore converges
// ═════════════════════════════════════════════════════════════════════════════

/// W4's object.
const KILLED: u64 = 401;
/// Its lineage: a base at 20 and a tail to 21.
const KILLED_BASE_SEQ: u64 = 20;
/// The high-water of its surviving stream, plausible and above the head.
const KILLED_HIGH_WATER: u64 = 55;
/// The batch size W4 runs at, so the object takes several row batches.
const SMALL_BATCHES: u64 = 12;

/// W4's fixture: object [`KILLED`], lost (no header) but with residue of an
/// earlier object and a surviving stream whose high-water is above the
/// lineage's head, restored from a base and a tail carrying a value the
/// backup store does not hold. Returns the base, the tail, the head and the
/// carried value's hash.
async fn killed_fixture(
    tv: &TestValkey,
    store: &Arc<WriterStore>,
) -> (Bytes, Bytes, u64, [u8; 32]) {
    let (value_hash, value) = carried(b"the killed layer's tail-carried value");
    let bytes = base(6, 0x4A, &[]);
    let (envelope, head) = tail(KILLED, KILLED_BASE_SEQ, &[(value_hash, value)]);
    put_session(
        store.as_ref(),
        &session_backup("s4", &[("k.usda", KILLED)], &[]),
    )
    .await;
    put_base(
        store.as_ref(),
        KILLED,
        "s4",
        "k.usda",
        KILLED_BASE_SEQ,
        bytes.clone(),
    )
    .await;
    put_tail(
        store.as_ref(),
        KILLED,
        "s4",
        "k.usda",
        KILLED_BASE_SEQ,
        head,
        envelope.clone(),
    )
    .await;
    put_counters(store.as_ref()).await;
    seed_residue(tv, KILLED).await;
    xadd(tv, KILLED, &format!("{KILLED_HIGH_WATER}-0")).await;
    (bytes, envelope, head, value_hash)
}

/// The run after a kill (or the only run) wrote what an uninterrupted run
/// writes: the tail-carried value in the backup store and in Valkey, the
/// band above the surviving high-water, the content of the lineage (the
/// residue gone), in several batches with the header last, and no stream.
async fn assert_killed_object_restored(
    tv: &TestValkey,
    store: &Arc<WriterStore>,
    report: &live_edit::dr::job::ExecutionReport,
    fixture: &(Bytes, Bytes, u64, [u8; 32]),
) {
    let (bytes, envelope, head, value_hash) = fixture;
    assert!(
        store.holds_value(value_hash).await,
        "the tail-carried value is in the backup store"
    );
    assert!(
        exists(tv, tv.key_schema().ext_data(&sha256_to_hex(value_hash))).await,
        "the tail-carried value is in Valkey"
    );
    let seeded = KILLED_HIGH_WATER + BAND;
    assert_eq!(
        header(tv, KILLED).await.get("seq").map(String::as_str),
        Some(seeded.to_string().as_str()),
        "banded above the surviving high-water"
    );
    let write = &report.objects[&KILLED];
    assert_eq!(
        (write.high_water, write.seeded_seq),
        (Some(KILLED_HIGH_WATER), seeded)
    );
    let (rows, last) = write.batches.split_at(write.batches.len() - 1);
    assert!(rows.len() >= 2, "several row batches: {:?}", write.batches);
    assert!(rows.iter().all(|b| b.commands <= SMALL_BATCHES));
    assert_eq!(last[0].commands, 2, "the header transaction last");
    assert_restored(
        tv,
        KILLED,
        &expected_state(
            KILLED,
            KILLED_BASE_SEQ,
            bytes,
            std::slice::from_ref(envelope),
            seeded,
        ),
    )
    .await;
    assert_eq!(
        header(tv, KILLED).await["last_checkpoint_seq"],
        head.to_string()
    );
    assert!(!exists(tv, tv.key_schema().layer_stream(KILLED)).await);
}

/// Which of W4's restored rows had landed at a kill.
#[derive(Debug, PartialEq, Eq)]
enum RowsLanded {
    /// None: no restored section, children list, field, time sample or
    /// external-value index.
    None,
    /// Exactly row batches 0 and 1's units. The eight restored sections come
    /// first (in the state's own order), then the children lists, fields,
    /// time samples and external-value index writes. At [`SMALL_BATCHES`]
    /// (12 commands): batch 0 is six sections (2 commands each); batch 1 the
    /// other two and, when the decode keeps one, the root's one-child list
    /// (3 commands); the `/World` prim's list of seven children (9 commands)
    /// fits batch 1 in no order, so it, and every field, time sample and
    /// external-value index write after it, is in a later batch. So: every
    /// restored section, and no `/World` list, field, time sample or index.
    /// (After batch 0 alone, two sections are missing; after batch 2,
    /// `/World`'s list is there.)
    FirstTwoBatches,
    /// Every unit: every section, `/World`'s children list, every field and
    /// time sample, and the external-value index.
    All,
    /// Anything else, as read.
    Other(String),
}

/// Read which of W4's restored rows had landed, against `expected`, the
/// state the restore writes.
async fn rows_landed(tv: &TestValkey, expected: &ObjectState) -> RowsLanded {
    let oid = KILLED.to_string();
    let sections: BTreeSet<String> = smembers(tv, tv.key_schema().obj_sections_index(&oid))
        .await
        .into_iter()
        .filter(|s| s != "777")
        .collect();
    let all_sections: BTreeSet<String> = expected.sections.keys().map(u64::to_string).collect();
    let world_children = exists(tv, tv.key_schema().obj_children(&oid, "1")).await;
    let fields: u64 = tv
        .client()
        .pool
        .zcard(tv.key_schema().obj_fields_index(&oid))
        .await
        .expect("ZCARD fields");
    let samples: u64 = tv
        .client()
        .pool
        .zcard(tv.key_schema().obj_time_samples_index(&oid))
        .await
        .expect("ZCARD time samples");
    let ext_index = exists(tv, tv.key_schema().obj_ext_index(&oid)).await;
    let after_sections = (world_children, fields, samples, ext_index);
    if sections.is_empty() && after_sections == (false, 0, 0, false) {
        RowsLanded::None
    } else if sections == all_sections && after_sections == (false, 0, 0, false) {
        RowsLanded::FirstTwoBatches
    } else if sections == all_sections
        && after_sections
            == (
                true,
                expected.fields.len() as u64,
                expected.time_samples.len() as u64,
                true,
            )
    {
        RowsLanded::All
    } else {
        RowsLanded::Other(format!(
            "sections {sections:?} of {all_sections:?}; (/World's children list, fields, time \
             samples, external-value index) {after_sections:?}"
        ))
    }
}

/// What a kill at `point` left of W4's object, before the re-run: every
/// step before the point done and none after it. `(the residue's row
/// survives, the value is in Valkey, the value is in the backup store, the
/// restored rows landed)`; the stream survives every kill (it goes with the
/// header).
async fn assert_killed_at(
    tv: &TestValkey,
    store: &Arc<WriterStore>,
    point: &KillPoint,
    fixture: &(Bytes, Bytes, u64, [u8; 32]),
) {
    let (bytes, envelope, _, value_hash) = fixture;
    let oid = KILLED.to_string();
    let expected = expected_state(
        KILLED,
        KILLED_BASE_SEQ,
        bytes,
        std::slice::from_ref(envelope),
        KILLED_HIGH_WATER + BAND,
    );
    let state = (
        exists(tv, tv.key_schema().obj_section(&oid, "777")).await,
        exists(tv, tv.key_schema().ext_data(&sha256_to_hex(value_hash))).await,
        store.holds_value(value_hash).await,
        rows_landed(tv, &expected).await,
    );
    let want = match point {
        KillPoint::ObjectAfterHighWater(_) => (true, false, false, RowsLanded::None),
        KillPoint::ObjectAfterWipe(_) => (false, false, false, RowsLanded::None),
        KillPoint::ObjectAfterBackfillValkey(_) => (false, true, false, RowsLanded::None),
        KillPoint::ObjectAfterBackfillBackup(_) => (false, true, true, RowsLanded::None),
        KillPoint::ObjectAfterBatch { batch: 1, .. } => {
            (false, true, true, RowsLanded::FirstTwoBatches)
        }
        KillPoint::ObjectBeforeHeader(_) => (false, true, true, RowsLanded::All),
        other => panic!("not a restore's kill point this fixture describes: {other:?}"),
    };
    assert_eq!(
        state, want,
        "{point:?}: (residue row, value in Valkey, value in the backup store, restored rows) at \
         the kill"
    );
    assert!(
        exists(tv, tv.key_schema().layer_stream(KILLED)).await,
        "{point:?}: the stream survives until the header transaction"
    );
}

/// Kill the writer at `point` on W4's fixture, check what the kill left,
/// run it again, and check the re-run converged on what an uninterrupted
/// run writes.
async fn kill_then_converge(point: KillPoint) {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let fixture = killed_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;

        run_to_the_kill(&tv, &config, &store, |b| {
            b.with_batch_commands_for_test(SMALL_BATCHES)
                .with_kill_point_for_test(point.clone())
        })
        .await;
        assert!(
            header(&tv, KILLED).await.is_empty(),
            "the header is written last, after the backfills and every row batch: none landed \
             before the writer's kill point"
        );
        assert_killed_at(&tv, &store, &point, &fixture).await;
        let (_plan, report, _captured) =
            run_to_the_end_with(RUN_BUDGET, &tv, &config, &store, |b| {
                b.with_batch_commands_for_test(SMALL_BATCHES)
            })
            .await;

        assert_killed_object_restored(&tv, &store, &report, &fixture).await;
    })
    .await;
}

/// The reference the kill tests compare with: an uninterrupted run.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_uninterrupted_bands_above_the_surviving_high_water_and_wipes_the_residue() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let fixture = killed_fixture(&tv, &store).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let (_plan, report, _captured) =
            run_to_the_end_with(RUN_BUDGET, &tv, &config, &store, |b| {
                b.with_batch_commands_for_test(SMALL_BATCHES)
            })
            .await;
        assert_killed_object_restored(&tv, &store, &report, &fixture).await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_killed_after_the_high_water_read_converges() {
    kill_then_converge(KillPoint::ObjectAfterHighWater(Some(KILLED))).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_killed_after_the_wipe_converges() {
    kill_then_converge(KillPoint::ObjectAfterWipe(Some(KILLED))).await;
}

/// Also the header-before-backfill pin: killed between the Valkey and the
/// backup-store backfill, the re-run must still PUT the value, which it can
/// only if no header landed before the backfill.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_killed_after_the_valkey_backfill_converges() {
    kill_then_converge(KillPoint::ObjectAfterBackfillValkey(Some(KILLED))).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_killed_after_the_backup_backfill_converges() {
    kill_then_converge(KillPoint::ObjectAfterBackfillBackup(Some(KILLED))).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_killed_after_a_row_batch_converges() {
    kill_then_converge(KillPoint::ObjectAfterBatch {
        object: Some(KILLED),
        batch: 1,
    })
    .await;
}

/// PM r2 N4: killed after every row batch, before the header transaction,
/// the surviving stream still carries its high-water, so the re-run bands
/// above it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_killed_before_the_header_still_bands_above_the_high_water() {
    kill_then_converge(KillPoint::ObjectBeforeHeader(Some(KILLED))).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// W5: values only the tails carry
// ═════════════════════════════════════════════════════════════════════════════

/// W5. Two values only a restored lineage's tail carries (the mirror never
/// copied them to the backup store) are stored in Valkey and PUT to the
/// backup store, once each, and counted. The planner settles a value a
/// restored tail carries without probing the backup store, so neither is
/// probed. A value the base references and the backup store already holds
/// is probed and not PUT.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_backfills_tail_carried_values_once_and_the_coverage_check_never_probes_them() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let first = carried(b"first value the backup store lacks");
        let second = carried(b"second value the backup store lacks");
        let in_backup = carried(b"a value the backup store holds");
        store
            .put_ext_blob(
                &sha256_to_hex(&in_backup.0),
                Bytes::from_static(in_backup.1),
            )
            .await
            .expect("the value put");
        let bytes = base(1, 0x50, &[in_backup.0]);
        let (envelope, head) = tail(501, 30, &[first, second]);
        put_session(
            store.as_ref(),
            &session_backup("s5", &[("v.usda", 501)], &[]),
        )
        .await;
        put_base(store.as_ref(), 501, "s5", "v.usda", 30, bytes).await;
        put_tail(store.as_ref(), 501, "s5", "v.usda", 30, head, envelope).await;
        put_counters(store.as_ref()).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        assert_eq!(
            report.backfill,
            BackfillReport {
                valkey_values: 2,
                backup_puts: 2,
            }
        );
        for (hash, _) in [first, second] {
            assert!(store.holds_value(&hash).await, "PUT to the backup store");
            assert_eq!(store.puts_of(&hash), 1, "PUT once");
            assert!(
                exists(&tv, tv.key_schema().ext_data(&sha256_to_hex(&hash))).await,
                "stored in Valkey"
            );
            assert_eq!(
                store.probes_of(&hash),
                0,
                "a tail-carried value is never probed"
            );
        }
        assert_eq!(store.puts_of(&in_backup.0), 1, "the fixture's PUT alone");
        assert_eq!(
            store.probes_of(&in_backup.0),
            1,
            "the base's value is probed"
        );
    })
    .await;
}

/// W5, the run's own PUTs (defer-45 (2)). Two restored layers whose tails
/// carry the same value, written one at a time
/// (`LIVE_EDIT_DR_RESTORE_PARALLELISM` = 1): the first layer PUTs the value
/// and records the PUT as the run's own, so the second, finding it
/// recorded, does not PUT it again. Each stores it in Valkey. Only the
/// serial order is pinned: at a parallelism above 1 both layers may PUT the
/// value before either records it, which a content-addressed PUT makes
/// harmless.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_puts_a_value_two_restored_layers_carry_once_as_the_runs_own() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let shared = carried(b"a value both layers' tails carry");
        let layers = [("p.usda", 511, 0x51), ("q.usda", 512, 0x52)];
        put_session(
            store.as_ref(),
            &session_backup("s5b", &[("p.usda", 511), ("q.usda", 512)], &[]),
        )
        .await;
        for (layer, oid, tag) in layers {
            put_base(store.as_ref(), oid, "s5b", layer, 30, base(1, tag, &[])).await;
            let (envelope, head) = tail(oid, 30, &[shared]);
            put_tail(store.as_ref(), oid, "s5b", layer, 30, head, envelope).await;
        }
        put_counters(store.as_ref()).await;
        let mut config = writer_config();
        config.dr_restore_parallelism = NonZeroUsize::new(1).expect("non-zero");
        hold(&tv, &config).await;

        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        for (_, oid, _) in layers {
            assert!(
                matches!(
                    report.objects[&oid].kind,
                    WriteKind::Restored { tails: 1, .. }
                ),
                "object {oid} restored from its base and tail"
            );
        }
        assert_eq!(
            report.backfill,
            BackfillReport {
                valkey_values: 2,
                backup_puts: 1,
            },
            "each layer stores the value in Valkey; only the first PUTs it"
        );
        assert_eq!(store.puts_of(&shared.0), 1, "PUT once over the run");
        assert!(store.holds_value(&shared.0).await);
        assert!(exists(&tv, tv.key_schema().ext_data(&sha256_to_hex(&shared.0))).await);
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// W6: a flag clear
// ═════════════════════════════════════════════════════════════════════════════

/// The `restore_seq_bump` W6's survivor carries before its flags are cleared.
const PRIOR_BUMP: u64 = 7;

/// The retirement flags a crash left on W6's survivor.
const FLAGS: [&str; 3] = ["superseded", "superseded_by", "ext_gc_scheduled"];

/// W6's fixture: a layer created through the service, then left retiring
/// as a crash would leave it: `superseded` naming an object that does not
/// exist, `ext_gc_scheduled`, a prior `restore_seq_bump`, the retirement
/// carrier and its attempts, and a `deleted` tombstone's auto-ID on top of
/// its stream. Returns `(session, object, seq)`.
async fn retiring_fixture(tv: &TestValkey, store: &Arc<WriterStore>) -> (String, u64, u64) {
    let svc = build_service(tv, Arc::new(AllowAllPermissions));
    let (session, oid) = create_session_and_layer(tv, &svc, "clear-flags", "c.usda").await;
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let seq: u64 = tv
        .client()
        .pool
        .hget::<String, _, _>(&header_key, "seq")
        .await
        .expect("HGET seq")
        .parse()
        .expect("a seq");
    let prior = PRIOR_BUMP.to_string();
    hset(
        tv,
        header_key,
        &[
            ("superseded", "1"),
            ("superseded_by", "999999"),
            ("ext_gc_scheduled", "1"),
            ("restore_seq_bump", prior.as_str()),
        ],
    )
    .await;
    let oid_str = oid.to_string();
    hset(
        tv,
        tv.key_schema().obj_ext_retire_carrier(&oid_str),
        &[("h", "1")],
    )
    .await;
    hset(
        tv,
        tv.key_schema().obj_ext_retire_attempts(&oid_str),
        &[("n", "1")],
    )
    .await;
    xadd(tv, oid, "*").await;
    put_session(
        store.as_ref(),
        &session_backup(&session, &[("c.usda", oid)], &[]),
    )
    .await;
    put_counters(store.as_ref()).await;
    (session, oid, seq)
}

/// W6's survivor after its flags were cleared: `seq` banded at the
/// survivor's `seq` (the stream's top is a tombstone), the band added to the
/// prior `restore_seq_bump`, a fresh `commit_chain`, every flag, the stream
/// and the carrier gone; then, with the hold lifted, an `UpdateLayer`
/// commits on it.
async fn assert_flags_cleared(
    tv: &TestValkey,
    (session, oid, seq): &(String, u64, u64),
    chain_before: &str,
) {
    let h = header(tv, *oid).await;
    assert_eq!(h["seq"], (seq + BAND).to_string(), "seq banded");
    assert_eq!(h["restore_seq_bump"], (PRIOR_BUMP + BAND).to_string());
    assert_ne!(h["commit_chain"], chain_before, "a fresh commit_chain");
    for flag in FLAGS {
        assert!(!h.contains_key(flag), "{flag} cleared: {h:?}");
    }
    let oid_str = oid.to_string();
    for key in [
        tv.key_schema().layer_stream(*oid),
        tv.key_schema().obj_ext_retire_carrier(&oid_str),
        tv.key_schema().obj_ext_retire_attempts(&oid_str),
    ] {
        assert!(!exists(tv, key.clone()).await, "{key} removed");
    }

    assert!(tv.remove_hold_marker().await);
    let svc = build_service(tv, Arc::new(AllowAllPermissions));
    let ompf = crate::common::owned_delta::delta_to_ompf_bytes(
        &crate::common::owned_delta::empty_delta(),
        &[],
    );
    handle_update_layer(
        &svc,
        make_request(streaming_update(vec![
            make_update_header(session, "c.usda", *oid),
            make_update_data(ompf),
        ])),
    )
    .await
    .unwrap_or_else(|s| panic!("an UpdateLayer commits on the cleared survivor: {s}"));
    assert_eq!(
        header(tv, *oid).await["seq"],
        (seq + BAND + 1).to_string(),
        "the update committed at the next seq"
    );
}

/// W6. A kept survivor whose retirement flags a crash left set has its
/// flags cleared (one transaction of 3 commands): see
/// [`assert_flags_cleared`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_clears_a_survivors_flags_with_a_banded_seq_and_it_takes_updates() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let fixture = retiring_fixture(&tv, &store).await;
        let chain_before = header(&tv, fixture.1).await["commit_chain"].clone();
        let config = writer_config();
        hold(&tv, &config).await;

        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        let write = &report.objects[&fixture.1];
        assert_eq!(write.kind, WriteKind::FlagsCleared);
        assert_eq!(
            (write.floor, write.high_water, write.band, write.seeded_seq),
            (fixture.2, None, BAND, fixture.2 + BAND)
        );
        assert_eq!(write.batches.len(), 1);
        assert_eq!(
            (write.batches[0].commands, write.batches[0].timeout),
            (3, txn_timeout(3))
        );
        assert_flags_cleared(&tv, &fixture, &chain_before).await;
    })
    .await;
}

/// What a flag clear killed at `point` left: the survivor's header as it
/// was (`seq`, every flag) and its stream; the retirement carrier and its
/// attempts are still there after the high-water read and gone after the
/// carrier's removal.
async fn assert_clear_killed_at(
    tv: &TestValkey,
    (_, oid, seq): &(String, u64, u64),
    point: &KillPoint,
) {
    let h = header(tv, *oid).await;
    assert_eq!(h["seq"], seq.to_string(), "{point:?}: seq not yet banded");
    for flag in FLAGS {
        assert!(h.contains_key(flag), "{point:?}: {flag} still set: {h:?}");
    }
    assert!(
        exists(tv, tv.key_schema().layer_stream(*oid)).await,
        "{point:?}: the stream stays until the last transaction"
    );
    let oid_str = oid.to_string();
    let carrier_left = match point {
        KillPoint::ClearFlagsAfterHighWater(_) => true,
        KillPoint::ClearFlagsAfterCarrier(_) => false,
        other => panic!("not a flag clear's kill point: {other:?}"),
    };
    for key in [
        tv.key_schema().obj_ext_retire_carrier(&oid_str),
        tv.key_schema().obj_ext_retire_attempts(&oid_str),
    ] {
        assert_eq!(
            exists(tv, key.clone()).await,
            carrier_left,
            "{point:?}: {key}"
        );
    }
}

/// A flag clear killed at `point` leaves the flags and the stream, so the
/// re-run plans and writes the same clear (PM B2's order).
async fn clear_flags_kill_then_converge(point: fn(u64) -> KillPoint) {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let fixture = retiring_fixture(&tv, &store).await;
        let chain_before = header(&tv, fixture.1).await["commit_chain"].clone();
        let config = writer_config();
        hold(&tv, &config).await;
        let point = point(fixture.1);

        run_to_the_kill(&tv, &config, &store, |b| {
            b.with_kill_point_for_test(point.clone())
        })
        .await;
        assert_clear_killed_at(&tv, &fixture, &point).await;
        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        assert_flags_cleared(&tv, &fixture, &chain_before).await;
        assert_eq!(report.objects[&fixture.1].kind, WriteKind::FlagsCleared);
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_clear_flags_killed_after_the_high_water_read_converges() {
    clear_flags_kill_then_converge(|oid| KillPoint::ClearFlagsAfterHighWater(Some(oid))).await;
}

/// Also the flags-last pin: killed after the carrier's removal, the re-run
/// finds the flags still set, so it bands the survivor; a clear that removed
/// the flags first would leave it kept, unbanded and unwritable.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_clear_flags_killed_after_the_carrier_removal_converges() {
    clear_flags_kill_then_converge(|oid| KillPoint::ClearFlagsAfterCarrier(Some(oid))).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// W8: no releases
// ═════════════════════════════════════════════════════════════════════════════

/// W8. A survivor replaced by its newer backup: its external-value index is
/// rewritten to exactly the backup's hashes, the stale hash's meta is left
/// as it was (no release decremented it), and no release is queued.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_rewrites_a_replaced_survivors_value_index_and_releases_nothing() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let stale = [0x80; 32];
        let (h1, h2) = ([0x81; 32], [0x82; 32]);
        for hash in [h1, h2] {
            store
                .put_ext_blob(&sha256_to_hex(&hash), Bytes::from_static(b"in the backup"))
                .await
                .expect("the value put");
        }
        put_session(
            store.as_ref(),
            &session_backup("s8", &[("r.usda", 801)], &[]),
        )
        .await;
        put_base(
            store.as_ref(),
            801,
            "s8",
            "r.usda",
            20,
            base(1, 0x80, &[h1, h2]),
        )
        .await;
        put_counters(store.as_ref()).await;
        seed_session_record(&tv, "s8").await;
        seed_mapping(&tv, "s8", "r.usda", 801).await;
        seed_complete(&tv, 801, "s8", "r.usda", 10, &[]).await;
        hset(
            &tv,
            tv.key_schema().obj_ext_index("801"),
            &[(sha256_to_hex(&stale).as_str(), "1")],
        )
        .await;
        let stale_meta = tv.key_schema().ext_meta(&sha256_to_hex(&stale));
        hset(&tv, stale_meta.clone(), &[("refcount", "1")]).await;
        let stale_meta_before: std::collections::HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(&stale_meta)
            .await
            .expect("HGETALL");
        let config = writer_config();
        hold(&tv, &config).await;

        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        assert!(matches!(
            report.objects[&801].kind,
            WriteKind::Restored { head_seq: 20, .. }
        ));
        let index: std::collections::HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(tv.key_schema().obj_ext_index("801"))
            .await
            .expect("HGETALL obj:ext");
        let mut named: Vec<String> = index.keys().cloned().collect();
        named.sort();
        assert_eq!(named, vec![sha256_to_hex(&h1), sha256_to_hex(&h2)]);
        let stale_meta_after: std::collections::HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(&stale_meta)
            .await
            .expect("HGETALL");
        assert_eq!(stale_meta_after, stale_meta_before, "no release counted");
        assert!(
            !exists(&tv, tv.key_schema().ext_pending()).await,
            "no release queued"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// W12 and defer-45
// ═════════════════════════════════════════════════════════════════════════════

/// W12. An object held with no session or layer (its only listed backup
/// cannot be read to learn them, under an empty session listing) is in the
/// plan's held items; the writer finishes and writes nothing for it, and the
/// run stays held (exit 3) after its recount.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_writes_nothing_for_an_object_held_with_no_session_or_layer() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        put_base(
            store.as_ref(),
            1201,
            "s12",
            "u.usda",
            30,
            base(1, 0x12, &[]),
        )
        .await;
        store.permanent(1201, 30);
        put_counters(store.as_ref()).await;
        let config = writer_config();
        hold(&tv, &config).await;

        let (plan, report, _captured) = run_to_the_end_held(&tv, &config, &store).await;

        assert!(
            matches!(
                plan.held.as_slice(),
                [HeldItem {
                    session_id,
                    layer_name,
                    object_id: Some(1201),
                    reason: HeldReason::Unknown { .. },
                }] if session_id.is_empty() && layer_name.is_empty()
            ),
            "{:?}",
            plan.held
        );
        assert!(report.objects.is_empty(), "{:?}", report.objects);
        assert!(object_keys(&tv, 1201).await.is_empty());
    })
    .await;
}

/// defer-45 (1). A backup generation the plan selected that reads absent
/// when the writer fetches it again contradicts the plan: the Job logs one
/// ERROR naming the layer, the session, the object and the generation, and
/// exits 1 without stepping down to another backup or writing the object.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_exits_1_naming_the_object_when_a_planned_backup_reads_absent_on_refetch() {
    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        put_session(
            store.as_ref(),
            &session_backup("s13", &[("x.usda", 1301)], &[]),
        )
        .await;
        put_base(store.as_ref(), 1301, "s13", "x.usda", 4, base(1, 0x13, &[])).await;
        put_base(store.as_ref(), 1301, "s13", "x.usda", 5, base(2, 0x13, &[])).await;
        store.absent_after(1301, 5, 1);
        put_counters(store.as_ref()).await;
        let config = writer_config();
        hold(&tv, &config).await;
        let (body, observed) = observed_body(|b| b);

        let (exit, captured) = run(&tv, &config, &store, body).await;

        assert_eq!(exit, JobExit::Failed);
        assert_eq!(
            store.gets_of(1301, 5),
            2,
            "fixture: the plan's read, then the writer's"
        );
        assert_eq!(
            store.gets_of(1301, 4),
            0,
            "no step-down to the older backup"
        );
        assert_eq!(
            lines(
                &captured,
                "the restore Job could not restore layer x.usda of session s13 (object 1301): \
                 backup generation 5 reads as absent; its plan selected generation 5, so the Job \
                 does not fall back to another backup; it stops writing and exits 1 so \
                 Kubernetes retries it, and the service stays held"
            )
            .len(),
            1,
            "capture:\n{}",
            captured.text
        );
        assert!(
            observed.lock().unwrap().is_empty(),
            "the writer did not finish"
        );
        assert_eq!(
            lines_starting(&captured, "the restore Job finished writing"),
            0,
            "the finish never ran"
        );
        assert!(
            object_keys(&tv, 1301).await.is_empty(),
            "nothing written for it"
        );
    })
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// W13, W13b: a layer with no root section
// ═════════════════════════════════════════════════════════════════════════════

/// A base backup holding exactly `delta`'s content, with no external values.
fn base_of(delta: &OwnedDelta) -> Bytes {
    Bytes::from(
        live_edit::domain::ompf::OmpfContainer::serialize(
            &oodf_ffi::build_delta(delta).expect("the base delta builds"),
            &[],
        )
        .expect("the base serializes"),
    )
}

/// W13. A layer backed up with no top-level section (no section whose
/// `parent_id` is 0; SDD §9.7) is restored exactly as backed up: the Job adds
/// no root section, because a restore reproduces the backup as stored. Once
/// the hold lifts, the restored layer accepts a structural `UpdateLayer` that
/// leaves it without one, since it had none to lose.
///
/// Fail-on-old: a writer that applied the `CreateLayer` root rule
/// (`domain::layer_root::build_creation_state`) would refuse this content or
/// write a supplied root, and the restored content would differ from the
/// backup's.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_restores_a_layer_with_no_root_section_as_backed_up_and_it_accepts_updates() {
    use live_edit::domain::layer_root::has_top_level_section;
    use oodf_ffi::CommandDiffChangeSection;

    /// A section stored under parent 42, which has no row: never top-level.
    fn orphan(section_id: u64) -> StructCommand {
        StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 42,
            section_id,
            section_name: format!("orphan{section_id}"),
            section_type: SPEC_PRIM,
            section_order: 0,
        })
    }

    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let rootless = OwnedDelta {
            is_diff: true,
            base_version: 0,
            struct_commands: vec![orphan(5)],
            set_fields: vec![SetField {
                section_id: 5,
                key_name: "default".to_owned(),
                value_or_ext_hash: b"rootless".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            }],
            time_samples: Vec::new(),
            source_format: "usda".to_owned(),
        };
        let base_bytes = base_of(&rootless);
        put_session(
            store.as_ref(),
            &session_backup("s-r", &[("r.usda", 1401)], &[]),
        )
        .await;
        put_base(store.as_ref(), 1401, "s-r", "r.usda", 6, base_bytes.clone()).await;
        put_counters(store.as_ref()).await;
        let expected = expected_state(1401, 6, &base_bytes, &[], 6 + BAND);
        assert!(
            !expected.sections.is_empty() && !has_top_level_section(&expected),
            "fixture: the backup holds sections but no top-level section"
        );
        let config = writer_config();
        hold(&tv, &config).await;

        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        assert!(
            matches!(report.objects[&1401].kind, WriteKind::Restored { .. }),
            "{:?}",
            report.objects
        );
        assert_restored(&tv, 1401, &expected).await;

        assert!(tv.remove_hold_marker().await, "the hold marker stayed");
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let update = OwnedDelta {
            struct_commands: vec![orphan(6)],
            ..OwnedDelta::default()
        };
        handle_update_layer(
            &svc,
            make_request(streaming_update(vec![
                make_update_header("s-r", "r.usda", 1401),
                make_update_data(crate::common::owned_delta::delta_to_ompf_bytes(
                    &update,
                    &[],
                )),
            ])),
        )
        .await
        .unwrap_or_else(|s| panic!("a structural UpdateLayer commits on the rootless layer: {s}"));
        let after = stored_state(&tv, 1401).await;
        assert_eq!(
            after.sequence_num,
            6 + BAND + 1,
            "the update committed at the next seq"
        );
        assert!(after.sections.contains_key(&6), "the update's section");
        assert!(
            !has_top_level_section(&after),
            "the layer still has no top-level section: nothing was supplied"
        );
    })
    .await;
}

/// W13b. A fully empty layer, backed up with no section, field or time
/// sample (as a replica created one before `CreateLayer` supplied a root
/// section), is restored exactly as backed up: still empty, with no root
/// section added.
///
/// Fail-on-old: a writer that supplied a root section to a layer lacking
/// one would write a section the backup does not hold.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_job_restores_a_fully_empty_layer_as_backed_up_with_no_root_section() {
    use live_edit::domain::layer_root::has_top_level_section;

    within_test_budget(async {
        let tv = TestValkey::setup().await;
        let store = WriterStore::new();
        let empty = OwnedDelta {
            is_diff: true,
            source_format: "usda".to_owned(),
            ..OwnedDelta::default()
        };
        let base_bytes = base_of(&empty);
        put_session(
            store.as_ref(),
            &session_backup("s-e", &[("e.usda", 1402)], &[]),
        )
        .await;
        put_base(store.as_ref(), 1402, "s-e", "e.usda", 6, base_bytes.clone()).await;
        put_counters(store.as_ref()).await;
        let expected = expected_state(1402, 6, &base_bytes, &[], 6 + BAND);
        assert!(
            expected.sections.is_empty()
                && expected.children.values().all(Vec::is_empty)
                && expected.fields.is_empty()
                && expected.time_samples.is_empty(),
            "fixture: the backup holds no section, field or time sample: {expected:?}"
        );
        let config = writer_config();
        hold(&tv, &config).await;

        let (_plan, report, _captured) = run_to_the_end(&tv, &config, &store).await;

        assert!(
            matches!(report.objects[&1402].kind, WriteKind::Restored { .. }),
            "{:?}",
            report.objects
        );
        assert_restored(&tv, 1402, &expected).await;
        assert!(
            !has_top_level_section(&stored_state(&tv, 1402).await),
            "the restored layer has no top-level section: nothing was supplied"
        );
    })
    .await;
}
