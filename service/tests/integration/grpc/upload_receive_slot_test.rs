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

//! A receiving upload whose content the upload budget cannot hold beside
//! another upload's in-budget processing charge runs in the free overflow slot
//! instead of being refused partway through its body (SDD §5.4).
//!
//! An upload's reservation grows by twice each chunk while its body is read,
//! and that growth never waits. It used to be granted in the budget `B` or
//! refused `UNAVAILABLE` (`cause` = `growth`) — so a large upload that fits an
//! empty budget was refused mid-body whenever another upload's processing
//! charge held part of `B`. Now a growth `B` cannot hold takes the overflow
//! slot at once when the slot is free, no processing step is queued for it,
//! and the upload is not a chained `UpdateLayer`; otherwise it is still
//! refused.
//!
//! Each handler case runs against a gate at `4A` and parks a dense
//! `CreateLayer` at the `CreateLayer` pre-commit hook, holding a processing
//! charge `P` in budget, sized (and self-checked) so that a large upload's whole receive
//! phase, `F + A + 2 × L`, fits the empty budget but not `B − P`. The large
//! upload carries one external value of about 9 MiB.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`create_layer_whose_content_does_not_fit_beside_a_processing_charge_commits_through_the_free_slot`] | the large `CreateLayer` commits; one overflow run, no `refused_growth`, one "upload runs beyond the upload budget" line at phase `receiving` with the charge the slot was booked at and `waited_ms` 0; the in-budget peak stays within `B` |
//! | [`create_layer_that_takes_the_slot_while_receiving_and_then_ends_early_logs_its_entry`] | the entry line is logged at entry: an upload that takes the slot and is then refused for a chunk past the size cap still logs its one `receiving` line, paired with its one overflow run |
//! | [`create_layer_whose_content_does_not_fit_beside_a_processing_charge_is_refused_growth_while_the_slot_is_held`] | with the slot held, and then with a processing step queued for it too, the same upload is refused `UNAVAILABLE` `growth` at stage `body`, counted `refused_growth`, nothing created; the slot goes to the queued step, which logs its entry at phase `processing` |
//! | [`chained_update_layer_whose_content_does_not_fit_is_refused_growth_with_the_slot_free`] | a chained `UpdateLayer` never takes the slot: refused `growth` at stage `body` with the slot free; its unchained twin takes the slot and commits |
//!
//! Recorded red runs. With the receive growth denied the slot
//! (`read_upload_body` passing `may_enter_overflow = false` for every
//! upload), the first test's upload and the unchained twin are refused
//! `UNAVAILABLE` with `cause` `growth`, while the slot-held control and the
//! chained call's refusal still pass. With the `UpdateLayer` handler letting
//! a chained call take the slot, the chained call is not refused. With the
//! entry line's log call removed from `read_upload_body`, the upload that ends
//! early logs no entry line. That shows the tests catch those three changes;
//! it is not a claim about any other.
//!
//! A processing step is only ever queued for the slot while the slot is held
//! — a free slot goes to the head of the queue at once — so the queued-step
//! case cannot be separated from the held slot here; the gate's unit tests
//! (`grpc::upload_admission`) cover the rule itself.
//!
//! The parked holders need `--features test-fault-injection`; without it
//! this module is compiled out.
//!
//! # Wait budget
//!
//! Each test takes ONE deadline when it starts ([`TEST_BUDGET`], 60 s) and
//! bounds every wait — each handler result, each poll for a condition, each
//! holder's arrival at its commit point — by that instant. Waits inside a
//! handler (the 10 s admission wait, which no case reaches) are inside its
//! bounded result. All are under nextest's 120 s kill.

use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::KeysInterface;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tonic::{Code, Status};

use live_edit::error::{refusal_metadata, refusal_reason, CAPACITY_RETRY_DELAY};
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    update_layer_request, ChainedUpdateLayerHeader, CreateLayerRequest, UpdateLayerHeader,
    UpdateLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::upload_admission::{
    admission_charge, processing_charge, CommitPath, DeltaFootprint, UploadAdmissionConfig,
    UploadAdmissionGate, RECEIVE_ALLOWANCE_BYTES, RECEIVE_CHARGE_FACTOR,
};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::storage::external_values::ext_content_hash;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use live_edit::storage::watched_txn::WatchWindowHook;

use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::common::refusal_detail::assert_refusal_detail;

use super::helpers::*;
use super::log_capture::{assert_common_fields, assert_field, capture, RPC_FAILURE_MSG};

/// One test's whole wait budget. Unreachable on a healthy run.
const TEST_BUDGET: Duration = Duration::from_secs(60);

/// `F + A`: what every admission reserves.
const C: u64 = admission_charge();

/// `B` = `4A`, floored for a 0-byte size cap: three admissions by bytes and
/// a per-session receiving cap of one.
const BUDGET: u64 = 4 * RECEIVE_ALLOWANCE_BYTES;

/// The section every fixture field hangs off.
const SECTION: u64 = 1;

/// Fields in the in-budget holder's `CreateLayer`: a processing charge of
/// about 12 MB, which `B` holds beside one admission but not beside the
/// large upload's receive phase (checked by [`park_in_budget_holder`]).
const HOLDER_FIELDS: usize = 1_500;

/// Fields in an over-budget `CreateLayer`: a processing charge above `B`, so
/// it runs in the overflow slot or waits for it.
const OVER_BUDGET_FIELDS: usize = 6_000;

/// The large upload's external value: with its delta, an OMPF body of about
/// 9 MiB, under `build_service`'s 10 MB size cap.
const BLOB_BYTES: usize = 9 * 1024 * 1024 - 4_096;

/// The chunks the large upload's body is sent in: each message stays under
/// tonic's 4 MiB decode limit.
const LARGE_UPLOAD_CHUNKS: usize = 4;

/// The INFO line an upload logs when it enters the overflow slot.
const SLOT_RUN_MSG: &str = "upload runs beyond the upload budget";

fn deadline() -> tokio::time::Instant {
    tokio::time::Instant::now() + TEST_BUDGET
}

/// Poll `cond` until it holds, failing at `deadline`.
async fn until(deadline: tokio::time::Instant, what: &str, mut cond: impl FnMut() -> bool) {
    while !cond() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} did not happen within the test's {TEST_BUDGET:?} budget"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// `fut`, bounded by `deadline`.
async fn settled<T>(
    deadline: tokio::time::Instant,
    what: &str,
    fut: impl std::future::Future<Output = T>,
) -> T {
    tokio::time::timeout_at(deadline, fut)
        .await
        .unwrap_or_else(|_| panic!("{what} did not finish within the test's {TEST_BUDGET:?}"))
}

fn create_section(parent_id: u64, section_id: u64) -> StructCommand {
    StructCommand::CreateSection(CommandCreateSection {
        parent_id,
        section_id,
        section_name: format!("s{section_id}"),
        section_type: 0,
    })
}

/// A `CreateLayer` delta: `SECTION`, then `n` inline fields on it.
fn dense_create(n: usize) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![create_section(0, SECTION)],
        set_fields: (0..n)
            .map(|i| SetField {
                section_id: SECTION,
                key_name: format!("f{i}"),
                value_or_ext_hash: (i as u64).to_le_bytes().to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            })
            .collect(),
        ..Default::default()
    }
}

/// One field on `SECTION` referencing `blob` as external value 1.
fn blob_field(blob: &[u8]) -> SetField {
    SetField {
        section_id: SECTION,
        key_name: "blob".to_owned(),
        value_or_ext_hash: ext_content_hash(blob).to_vec(),
        ext_value_size: blob.len() as u64,
        ext_value_index: 1,
        set_order: 0,
    }
}

/// The large upload's external value.
fn blob() -> Vec<u8> {
    (0..BLOB_BYTES).map(|i| (i % 251) as u8).collect()
}

/// The large `CreateLayer` body: `SECTION` holding one field whose value is
/// [`blob`].
fn large_create_ompf() -> Vec<u8> {
    let blob = blob();
    let delta = OwnedDelta {
        struct_commands: vec![create_section(0, SECTION)],
        set_fields: vec![blob_field(&blob)],
        ..Default::default()
    };
    delta_to_ompf_bytes(&delta, &[&blob])
}

/// The large `UpdateLayer` body: one field on the existing `SECTION` whose
/// value is [`blob`].
fn large_update_ompf() -> Vec<u8> {
    let blob = blob();
    let delta = OwnedDelta {
        set_fields: vec![blob_field(&blob)],
        ..Default::default()
    };
    delta_to_ompf_bytes(&delta, &[&blob])
}

/// `bytes` split into `parts` contiguous chunks (the last takes the rest).
fn split(bytes: &[u8], parts: usize) -> Vec<Vec<u8>> {
    let size = bytes.len().div_ceil(parts).max(1);
    bytes.chunks(size).map(<[u8]>::to_vec).collect()
}

/// A `CreateLayer` of `layer` on `session_id` carrying `ompf` in `parts`
/// chunks, its stream ended with END_STREAM.
fn create_request(
    session_id: &str,
    layer: &str,
    ompf: &[u8],
    parts: usize,
) -> tonic::Request<tonic::Streaming<CreateLayerRequest>> {
    let messages: Vec<CreateLayerRequest> = std::iter::once(make_create_header(session_id, layer))
        .chain(split(ompf, parts).into_iter().map(make_create_data))
        .collect();
    make_request(streaming_create(messages))
}

/// The receive charge at which an upload of `ompf`, sent in `parts` chunks,
/// first needs more than `room` in budget: `F + A + 2 ×` the bytes received
/// through the first chunk that does not fit. `None` when every chunk fits.
fn first_growth_past(room: u64, ompf: &[u8], parts: usize) -> Option<u64> {
    let mut received = 0u64;
    split(ompf, parts).iter().find_map(|chunk| {
        received += chunk.len() as u64;
        let target = C + RECEIVE_CHARGE_FACTOR * received;
        (target > room).then_some(target)
    })
}

/// `live_edit_upload_admission_total{rpc, result="refused_growth"}`.
fn refused_growth(svc: &LiveEditService, rpc: &str) -> f64 {
    sample_value(
        &render(&svc.metrics),
        "live_edit_upload_admission_total",
        &[("rpc", rpc), ("result", "refused_growth")],
    )
    .unwrap_or(f64::NAN)
}

/// `live_edit_upload_overflow_runs_total{rpc}`.
fn overflow_runs(svc: &LiveEditService, rpc: &str) -> f64 {
    sample_value(
        &render(&svc.metrics),
        "live_edit_upload_overflow_runs_total",
        &[("rpc", rpc)],
    )
    .unwrap_or(f64::NAN)
}

/// The growth refusal: `UNAVAILABLE` with the `UPLOAD_CAPACITY` retry hint,
/// `cause` = `growth`.
fn assert_growth_refusal(status: &Status, what: &str) {
    assert_eq!(status.code(), Code::Unavailable, "{what}: {status:?}");
    assert_refusal_detail(
        status,
        refusal_reason::UPLOAD_CAPACITY,
        &[(refusal_metadata::CAUSE, refusal_metadata::GROWTH)],
        CAPACITY_RETRY_DELAY,
    );
}

async fn new_session(tv: &TestValkey, name: &str) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, "/stage", "test-user")
        .await
        .expect("session creation must succeed")
}

/// `CreateLayer` pre-commit hooks that park the first `parks` calls to reach
/// them until [`Self::release`], and let every later one through. A parked
/// call keeps everything it holds of the gate: its processing charge in
/// budget, or the overflow slot.
struct ParkedHolders {
    hook: WatchWindowHook,
    entered: mpsc::UnboundedReceiver<usize>,
    release: CancellationToken,
}

impl ParkedHolders {
    fn new(parks: usize) -> Self {
        let (entered_tx, entered) = mpsc::unbounded_channel();
        let release = CancellationToken::new();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let hook: WatchWindowHook = Arc::new({
            let release = release.clone();
            move |_client| {
                let call = calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let entered_tx = entered_tx.clone();
                let release = release.clone();
                Box::pin(async move {
                    if call < parks {
                        let _ = entered_tx.send(call);
                        release.cancelled().await;
                    }
                })
            }
        });
        Self {
            hook,
            entered,
            release,
        }
    }

    /// A service whose `CreateLayer` commits pass the hook, with its upload
    /// gate at [`BUDGET`].
    fn service(&self, tv: &TestValkey) -> LiveEditService {
        let store = Arc::new(
            LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
                .with_create_layer_pre_commit_hook_for_test(Arc::clone(&self.hook)),
        );
        let base = build_service_with_layer_store(tv, store);
        LiveEditService {
            upload_admission: Arc::new(UploadAdmissionGate::new(
                UploadAdmissionConfig {
                    budget_bytes: BUDGET,
                    wait: Duration::from_secs(10),
                    max_waiters: 8,
                    // No size-cap floor, so the budget can stay small.
                    max_upload_bytes: 0,
                },
                Some(Arc::clone(&base.metrics)),
            )),
            ..base
        }
    }

    /// Start a `CreateLayer` of `fields` dense fields on a new session and
    /// wait, until `deadline`, for it to park at its commit point.
    async fn park(
        &mut self,
        tv: &TestValkey,
        svc: &Arc<LiveEditService>,
        fields: usize,
        deadline: tokio::time::Instant,
    ) -> tokio::task::JoinHandle<Result<(), Status>> {
        let ompf = delta_to_ompf_bytes(&dense_create(fields), &[]);
        let session_id = new_session(tv, &format!("parked-{fields}")).await;
        let request = create_request(&session_id, "layer", &ompf, 1);
        let task = tokio::spawn({
            let svc = Arc::clone(svc);
            async move { handle_create_layer(&svc, request).await.map(|_| ()) }
        });
        settled(
            deadline,
            "a holder parked at its commit point",
            self.entered.recv(),
        )
        .await
        .expect("the hook's sender lives in the service");
        task
    }

    fn release(&self) {
        self.release.cancel();
    }
}

/// Park the in-budget holder — a dense `CreateLayer` of [`HOLDER_FIELDS`]
/// fields — and return its task and `P`, the processing charge it holds in
/// budget. Checks the fixture's arithmetic against `P` and the large upload's
/// body: one admission fits beside `P`; the large upload's whole receive phase
/// fits the empty budget but not `B − P`.
async fn park_in_budget_holder(
    parked: &mut ParkedHolders,
    tv: &TestValkey,
    svc: &Arc<LiveEditService>,
    large: &[u8],
    deadline: tokio::time::Instant,
) -> (tokio::task::JoinHandle<Result<(), Status>>, u64) {
    let task = parked.park(tv, svc, HOLDER_FIELDS, deadline).await;
    let gate = &svc.upload_admission;
    assert_eq!(
        gate.overflow_bytes(),
        0,
        "fixture self-check: the holder runs in budget"
    );
    let held = gate.reserved_bytes();
    let large_receive = C + RECEIVE_CHARGE_FACTOR * large.len() as u64;
    assert!(
        held + C <= BUDGET && large_receive <= BUDGET && held + large_receive > BUDGET,
        "fixture self-check: with P = {held} held in the {BUDGET}-byte budget, one admission \
         ({C}) must fit beside it, and the large upload's receive phase ({large_receive}) must \
         fit the empty budget but not beside P"
    );
    (task, held)
}

/// A dense `CreateLayer` whose content the budget cannot hold beside another
/// upload's in-budget processing charge `P`, while the overflow slot is free
/// and no processing step is queued, takes the slot at once while it receives
/// and commits.
///
/// It is counted once in `overflow_runs_total{rpc="create_layer"}` (the
/// in-budget holder never entered the slot), counts no `refused_growth`, and
/// logs exactly one INFO "upload runs beyond the upload budget" line: phase
/// `receiving`, `rpc` `create_layer`, `waited_ms` 0, and `charge_bytes` the
/// receive charge of the first chunk `B − P` could not hold — what the slot
/// was booked at. The in-budget peak stays within `B`. Refused `UNAVAILABLE`
/// with `cause` `growth` when the receive growth is denied the slot (recorded
/// red run).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_layer_whose_content_does_not_fit_beside_a_processing_charge_commits_through_the_free_slot(
) {
    let deadline = deadline();
    let tv = TestValkey::setup().await;
    let mut parked = ParkedHolders::new(1);
    let svc = Arc::new(parked.service(&tv));
    let gate = Arc::clone(&svc.upload_admission);
    let large = large_create_ompf();
    let (holder, held) = park_in_budget_holder(&mut parked, &tv, &svc, &large, deadline).await;
    let entry = first_growth_past(BUDGET - held, &large, LARGE_UPLOAD_CHUNKS)
        .expect("fixture self-check: a chunk does not fit beside P");

    let session_id = new_session(&tv, "large-create").await;
    let request = create_request(&session_id, "layer", &large, LARGE_UPLOAD_CHUNKS);
    let (result, captured) = settled(
        deadline,
        "the large CreateLayer",
        capture(handle_create_layer(&svc, request)),
    )
    .await;

    let created = result
        .expect(
            "a growth the budget cannot hold beside P takes the free slot, and the upload commits",
        )
        .into_inner();
    assert_ne!(created.object_id, 0);
    assert_eq!(overflow_runs(&svc, "create_layer"), 1.0);
    assert_eq!(refused_growth(&svc, "create_layer"), 0.0);
    let line = captured.only(SLOT_RUN_MSG);
    assert_eq!(line["level"], "INFO", "{line}");
    assert_field(line, "phase", "receiving".into());
    assert_field(line, "rpc", "create_layer".into());
    assert_field(line, "charge_bytes", entry.into());
    assert_field(line, "upload_budget_bytes", BUDGET.into());
    assert_field(line, "waited_ms", 0.into());
    assert!(
        gate.high_water_bytes() <= BUDGET,
        "in-budget peak {} past the budget",
        gate.high_water_bytes()
    );
    assert!(gate.overflow_high_water_bytes() >= entry);

    parked.release();
    settled(deadline, "the in-budget holder", holder)
        .await
        .expect("no panic")
        .expect("the in-budget holder commits");
    assert_eq!(gate.reserved_bytes(), 0);
    assert_eq!(gate.overflow_bytes(), 0);
}

/// A chunk sent after the large upload's content, taking it past
/// `build_service`'s 10 MB size cap.
const PAST_THE_CAP_CHUNK_BYTES: usize = 2 * 1024 * 1024;

/// The slot entry is logged when it happens, not when the body completes: a
/// `CreateLayer` whose receive growth takes the free slot and whose body then
/// ends early — a later chunk past the size cap, refused `INVALID_ARGUMENT` —
/// logs exactly one INFO "upload runs beyond the upload budget" line, phase
/// `receiving`, `charge_bytes` the charge the slot was booked at and
/// `waited_ms` 0, paired with its one `overflow_runs_total{create_layer}`.
/// Nothing is created, and the slot is free again once the call ends.
/// Recorded red run: with the entry line not logged at entry (the log call
/// in `read_upload_body` removed), the call logs no entry line.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_layer_that_takes_the_slot_while_receiving_and_then_ends_early_logs_its_entry() {
    let deadline = deadline();
    let tv = TestValkey::setup().await;
    let mut parked = ParkedHolders::new(1);
    let svc = Arc::new(parked.service(&tv));
    let gate = Arc::clone(&svc.upload_admission);
    let large = large_create_ompf();
    let (holder, held) = park_in_budget_holder(&mut parked, &tv, &svc, &large, deadline).await;
    let entry = first_growth_past(BUDGET - held, &large, LARGE_UPLOAD_CHUNKS)
        .expect("fixture self-check: a chunk does not fit beside P");
    assert!(
        large.len() <= svc.max_layer_size
            && large.len() + PAST_THE_CAP_CHUNK_BYTES > svc.max_layer_size,
        "fixture self-check: the content fits the size cap and the extra chunk does not"
    );

    let session_id = new_session(&tv, "ends-early").await;
    let messages: Vec<CreateLayerRequest> =
        std::iter::once(make_create_header(&session_id, "layer"))
            .chain(
                split(&large, LARGE_UPLOAD_CHUNKS)
                    .into_iter()
                    .map(make_create_data),
            )
            .chain(std::iter::once(make_create_data(vec![
                0;
                PAST_THE_CAP_CHUNK_BYTES
            ])))
            .collect();
    let (result, captured) = settled(
        deadline,
        "the CreateLayer that ends early",
        capture(handle_create_layer(
            &svc,
            make_request(streaming_create(messages)),
        )),
    )
    .await;

    let status = result.expect_err("a chunk past the size cap ends the upload");
    assert_eq!(status.code(), Code::InvalidArgument, "{status:?}");
    let line = captured.only(SLOT_RUN_MSG);
    assert_eq!(line["level"], "INFO", "{line}");
    assert_field(line, "phase", "receiving".into());
    assert_field(line, "rpc", "create_layer".into());
    assert_field(line, "charge_bytes", entry.into());
    assert_field(line, "waited_ms", 0.into());
    assert_eq!(overflow_runs(&svc, "create_layer"), 1.0);
    assert!(
        gate.overflow_high_water_bytes() >= entry,
        "the upload ran in the slot"
    );
    assert_eq!(
        gate.overflow_bytes(),
        0,
        "the slot is free once the call ends"
    );
    let mapping: Option<String> = tv
        .client()
        .pool
        .get(tv.key_schema().layer_mapping(&session_id, "layer"))
        .await
        .expect("GET the mapping");
    assert_eq!(mapping, None, "nothing created");

    parked.release();
    settled(deadline, "the in-budget holder", holder)
        .await
        .expect("no panic")
        .expect("the in-budget holder commits");
    assert_eq!(gate.reserved_bytes(), 0);
}

/// The same large `CreateLayer` is refused at once, `UNAVAILABLE` `growth` at
/// stage `body`, when the slot is held — by an over-budget `CreateLayer`
/// parked at its commit point — and again when a processing step is also
/// queued for the slot. Each refusal counts one `refused_growth`, no overflow
/// run, creates nothing, and leaves the slot to its holder; once the holders
/// are released the queued step takes the slot and all three commit, the
/// queued upload logging its entry once at phase `processing`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_layer_whose_content_does_not_fit_beside_a_processing_charge_is_refused_growth_while_the_slot_is_held(
) {
    let deadline = deadline();
    let tv = TestValkey::setup().await;
    let mut parked = ParkedHolders::new(2);
    let svc = Arc::new(parked.service(&tv));
    let gate = Arc::clone(&svc.upload_admission);
    let large = large_create_ompf();
    let (in_budget, _held) = park_in_budget_holder(&mut parked, &tv, &svc, &large, deadline).await;
    let slot_holder = parked.park(&tv, &svc, OVER_BUDGET_FIELDS, deadline).await;
    let slot = gate.overflow_bytes();
    assert!(
        slot > BUDGET,
        "fixture self-check: the over-budget holder holds the slot"
    );

    let refused_large_create = |what: &'static str| {
        let svc = Arc::clone(&svc);
        let tv = &tv;
        let large = &large;
        async move {
            let session_id = new_session(tv, what).await;
            let request = create_request(&session_id, "layer", large, LARGE_UPLOAD_CHUNKS);
            let (result, captured) =
                settled(deadline, what, capture(handle_create_layer(&svc, request))).await;
            let status = result.expect_err("the slot is held: the growth must be refused");
            assert_growth_refusal(&status, what);
            let event = captured.only(RPC_FAILURE_MSG);
            assert_common_fields(event, "WARN", "CreateLayer", "body", "UNAVAILABLE");
            assert!(
                captured.with_message(SLOT_RUN_MSG).is_empty(),
                "{what}: a refused upload never ran in the slot"
            );
            let mapping: Option<String> = tv
                .client()
                .pool
                .get(tv.key_schema().layer_mapping(&session_id, "layer"))
                .await
                .expect("GET the mapping");
            assert_eq!(mapping, None, "{what}: nothing created");
        }
    };

    refused_large_create("slot-held").await;
    assert_eq!(refused_growth(&svc, "create_layer"), 1.0);
    assert_eq!(gate.overflow_bytes(), slot, "the holder keeps the slot");
    assert_eq!(gate.processing_waiters(), 0, "a refusal never queues");

    // A third upload's processing step queues for the slot.
    let ompf = delta_to_ompf_bytes(&dense_create(OVER_BUDGET_FIELDS), &[]);
    let session_id = new_session(&tv, "queued").await;
    let request = create_request(&session_id, "layer", &ompf, 1);
    let queued = tokio::spawn({
        let svc = Arc::clone(&svc);
        async move {
            let (result, captured) = capture(handle_create_layer(&svc, request)).await;
            (result.map(|_| ()), captured)
        }
    });
    until(deadline, "a processing step queued for the slot", || {
        gate.processing_waiters() == 1
    })
    .await;

    refused_large_create("slot-held-and-queued").await;
    assert_eq!(refused_growth(&svc, "create_layer"), 2.0);
    assert_eq!(gate.overflow_bytes(), slot);
    assert_eq!(gate.processing_waiters(), 1, "still only the queued step");
    assert_eq!(
        overflow_runs(&svc, "create_layer"),
        1.0,
        "only the slot holder's entry"
    );

    parked.release();
    for (what, task) in [
        ("the in-budget holder", in_budget),
        ("the slot holder", slot_holder),
    ] {
        settled(deadline, what, task)
            .await
            .expect("no panic")
            .unwrap_or_else(|s| panic!("{what} commits: {s:?}"));
    }
    let (result, captured) = settled(deadline, "the queued upload", queued)
        .await
        .expect("no panic");
    result.unwrap_or_else(|s| panic!("the queued upload commits: {s:?}"));
    assert_eq!(
        overflow_runs(&svc, "create_layer"),
        2.0,
        "the slot went to the queued step"
    );
    let line = captured.only(SLOT_RUN_MSG);
    assert_field(line, "phase", "processing".into());
    assert_field(line, "rpc", "create_layer".into());
    assert_eq!(gate.reserved_bytes(), 0);
    assert_eq!(gate.overflow_bytes(), 0);
}

/// An `UpdateLayer` header for `object_id` of `session_id`'s `layer` from
/// `client`, chained to the chain's start when `chained`.
fn update_header(session_id: &str, object_id: u64, chained: bool) -> UpdateLayerRequest {
    let header = UpdateLayerHeader {
        session_id: session_id.to_owned(),
        layer_name: "layer".to_owned(),
        object_id,
        timestamp_microseconds_since_epoch: 1_000,
        client_id: "writer".to_owned(),
        operation_id: if chained {
            "chained-1".to_owned()
        } else {
            String::new()
        },
    };
    let message_type = if chained {
        update_layer_request::MessageType::ChainedHeader(ChainedUpdateLayerHeader {
            header: Some(header),
            previous_operation_id: String::new(),
            no_wait: false,
        })
    } else {
        update_layer_request::MessageType::Header(header)
    };
    UpdateLayerRequest {
        message_type: Some(message_type),
    }
}

/// A chained `UpdateLayer` never takes the overflow slot — it waits in the
/// chain gate after its body and must hold no slot there. With the slot free
/// and an in-budget processing charge `P` held, its large body is refused
/// `UNAVAILABLE` `growth` at stage `body`: one `refused_growth`, no overflow
/// run, the slot still free, nothing committed. Its unchained twin, the same
/// body under a plain header, takes the free slot and commits, logging its
/// entry at phase `receiving`. Recorded red runs: with the receive growth
/// denied the slot the twin is refused too; with the handler letting a
/// chained call take the slot, the chained call is not refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chained_update_layer_whose_content_does_not_fit_is_refused_growth_with_the_slot_free() {
    let deadline = deadline();
    let tv = TestValkey::setup().await;
    let mut parked = ParkedHolders::new(1);
    let svc = Arc::new(parked.service(&tv));
    let gate = Arc::clone(&svc.upload_admission);
    let setup = build_service(&tv, Arc::new(AllowAllPermissions));
    let base = delta_to_ompf_bytes(
        &OwnedDelta {
            struct_commands: vec![create_section(0, SECTION)],
            ..Default::default()
        },
        &[],
    );
    let session_id = new_session(&tv, "large-update").await;
    let object_id = handle_create_layer(&setup, create_request(&session_id, "layer", &base, 1))
        .await
        .expect("the base layer")
        .into_inner()
        .object_id;
    let large = large_update_ompf();
    let (holder, _held) = park_in_budget_holder(&mut parked, &tv, &svc, &large, deadline).await;
    let update = |chained: bool| {
        let messages: Vec<UpdateLayerRequest> =
            std::iter::once(update_header(&session_id, object_id, chained))
                .chain(
                    split(&large, LARGE_UPLOAD_CHUNKS)
                        .into_iter()
                        .map(make_update_data),
                )
                .collect();
        make_request(streaming_update(messages))
    };

    let (result, captured) = settled(
        deadline,
        "the chained UpdateLayer",
        capture(handle_update_layer(&svc, update(true))),
    )
    .await;
    let status = result.expect_err("a chained call's growth never takes the slot");
    assert_growth_refusal(&status, "the chained UpdateLayer");
    let event = captured.only(RPC_FAILURE_MSG);
    assert_common_fields(event, "WARN", "UpdateLayer", "body", "UNAVAILABLE");
    assert_eq!(refused_growth(&svc, "update_layer"), 1.0);
    assert_eq!(overflow_runs(&svc, "update_layer"), 0.0);
    assert_eq!(
        gate.overflow_high_water_bytes(),
        0,
        "the slot was never taken"
    );
    assert_eq!(gate.processing_waiters(), 0);

    let (result, captured) = settled(
        deadline,
        "the unchained twin",
        capture(handle_update_layer(&svc, update(false))),
    )
    .await;
    let updated = result
        .expect("the unchained twin takes the free slot and commits")
        .into_inner();
    assert_eq!(updated.sequence_num, 2, "only the twin committed");
    assert_eq!(overflow_runs(&svc, "update_layer"), 1.0);
    assert_eq!(refused_growth(&svc, "update_layer"), 1.0);
    let line = captured.only(SLOT_RUN_MSG);
    assert_field(line, "phase", "receiving".into());
    assert_field(line, "rpc", "update_layer".into());

    parked.release();
    settled(deadline, "the in-budget holder", holder)
        .await
        .expect("no panic")
        .expect("the in-budget holder commits");
    assert_eq!(gate.reserved_bytes(), 0);
    assert_eq!(gate.overflow_bytes(), 0);
}

/// Fixture self-check: the over-budget holder's processing charge is above
/// the whole budget, so it can only run in the slot.
#[test]
fn over_budget_fixture_is_charged_more_than_the_budget() {
    let delta = dense_create(OVER_BUDGET_FIELDS);
    let ompf = delta_to_ompf_bytes(&delta, &[]);
    let delta_len = oodf_ffi::build_delta(&delta).expect("builds").len() as u64;
    let charge = processing_charge(
        CommitPath::Create,
        ompf.len() as u64,
        delta_len,
        0,
        DeltaFootprint::of(Some(&delta)),
    );
    assert!(
        charge > BUDGET,
        "the over-budget fixture charges {charge}, within the {BUDGET}-byte budget"
    );
}
