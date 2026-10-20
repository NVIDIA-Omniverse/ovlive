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

//! b.yuo part C — the external-value release budget's upload gate, through
//! the real `CreateLayer` / `UpdateLayer` / `DeleteLayer` handlers and a real
//! `ValkeyExternalValueStore` carrying an `ExtReleaseGate`.
//!
//! The release total is planted directly on `le:ext:{gc}:pending:total`,
//! except in the shutdown and call-deadline tests, whose gate reads a
//! test-chosen total (`ExtReleaseGate::with_total_reader_for_test`) so they can
//! see each read the gate makes. The handlers are called directly, so no gRPC
//! deadline is in scope unless a test installs one with
//! `grpc::deadline::scope`, and the gate then waits `LIVE_EDIT_RPC_DEADLINE_MS`
//! less its 5 s margin; each test picks its RPC deadline to set that wait. The
//! gate's own unit tests (its wait bound, its end point, its counters) are in
//! `src/storage/ext_release_gate.rs`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fred::interfaces::{KeysInterface, StreamsInterface};
use tonic::Request;

use live_edit::error::{refusal_reason, CAPACITY_RETRY_DELAY};
use live_edit::grpc::layer::{handle_create_layer, handle_delete_layer, handle_update_layer};
use live_edit::grpc::proto::DeleteLayerRequest;
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::ext_release_gate::{
    ExtReleaseGate, ExtReleaseRefusal, TotalReaderForTest, EXT_RELEASE_BUDGET_FULL_MESSAGE,
    EXT_RELEASE_GATE_DEADLINE_MARGIN,
};
use live_edit::storage::external_values::{ext_content_hash, ValkeyExternalValueStore};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};

use super::helpers::*;
use super::log_capture::{assert_common_fields, capture, RPC_FAILURE_MSG};
use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::common::refusal_detail::{
    assert_refusal_detail, assert_refusal_error, assert_shutdown_status,
};

/// The release budget `H` every gated service here runs at.
const BUDGET: u64 = 1_000;

/// The section the fields below live in.
const SECTION_ID: u64 = 7;

/// A service whose external-value store carries a release gate at
/// [`BUDGET`], with an RPC deadline of `margin + wait` so an upload waits
/// `wait` at a full budget.
fn gated_service(tv: &TestValkey, wait: Duration) -> (LiveEditService, Arc<Metrics>) {
    let metrics = Arc::new(crate::common::metrics_scrape_only("ext-release-gate"));
    let gate = ExtReleaseGate::new(
        tv.client().clone(),
        tv.key_schema(),
        BUDGET,
        EXT_RELEASE_GATE_DEADLINE_MARGIN + wait,
        Some(Arc::clone(&metrics)),
    );
    (service_around(tv, gate, &metrics), metrics)
}

/// A service whose gate, at [`BUDGET`] and an RPC deadline of
/// `rpc_deadline`, reads its total from `reader` instead of the store.
fn service_with_reader(
    tv: &TestValkey,
    reader: TotalReaderForTest,
    rpc_deadline: Duration,
) -> (LiveEditService, Arc<Metrics>) {
    let metrics = Arc::new(crate::common::metrics_scrape_only("ext-release-gate"));
    let gate = ExtReleaseGate::with_total_reader_for_test(
        reader,
        BUDGET,
        rpc_deadline,
        Some(Arc::clone(&metrics)),
    );
    (service_around(tv, gate, &metrics), metrics)
}

fn service_around(
    tv: &TestValkey,
    gate: ExtReleaseGate,
    metrics: &Arc<Metrics>,
) -> LiveEditService {
    let ext_value_store =
        ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_metrics(Arc::clone(metrics))
            .with_ext_release_gate(Arc::new(gate));
    LiveEditService {
        ext_value_store: Arc::new(ext_value_store),
        ..build_service(tv, Arc::new(AllowAllPermissions))
    }
}

/// A gate reader that counts its reads in `reads` and answers a full budget
/// until `drops_at` holds an instant that has passed, then one byte below it.
fn counting_reader(
    reads: Arc<AtomicUsize>,
    drops_at: Arc<Mutex<Option<tokio::time::Instant>>>,
) -> TotalReaderForTest {
    Arc::new(
        move || -> futures::future::BoxFuture<'static, Option<u64>> {
            reads.fetch_add(1, Ordering::SeqCst);
            let drops_at: Option<tokio::time::Instant> = *drops_at.lock().unwrap();
            let full = drops_at.is_none_or(|at| tokio::time::Instant::now() < at);
            Box::pin(async move { Some(if full { BUDGET } else { BUDGET - 1 }) })
        },
    )
}

async fn set_total(tv: &TestValkey, total: u64) {
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().ext_pending_total(),
            total.to_string(),
            None,
            None,
            false,
        )
        .await
        .expect("SET pending:total");
}

/// `(passed_after_wait, refused, read_failed)`.
fn gate_cells(metrics: &Metrics) -> (u64, u64, u64) {
    let text = render(metrics);
    let cell = |outcome: &str| {
        sample_value(
            &text,
            "live_edit_ext_release_gate_total",
            &[("outcome", outcome)],
        )
        .unwrap_or_else(|| panic!("outcome={outcome} is zero-seeded")) as u64
    };
    (
        cell("passed_after_wait"),
        cell("refused"),
        cell("read_failed"),
    )
}

/// Assert `status` is the release-budget refusal as a client decodes it:
/// `UNAVAILABLE` with the unchanged "nothing was applied" message naming the
/// setting, carrying the retry hint `EXT_RELEASE_BUDGET_FULL` — no metadata, a
/// delay in `[500 ms, 1500 ms)` — and it is the very refusal the gate's
/// `Full` answers (`assert_refusal_error` on its `ServiceError`).
#[track_caller]
fn assert_budget_refusal(status: &tonic::Status) {
    assert_eq!(
        status.message(),
        EXT_RELEASE_BUDGET_FULL_MESSAGE,
        "{status:?}"
    );
    for fragment in ["nothing was applied", "LIVE_EDIT_EXT_RELEASE_MAX_BYTES"] {
        assert!(
            status.message().contains(fragment),
            "the refusal must say {fragment:?}; got {:?}",
            status.message()
        );
    }
    assert_refusal_detail(
        status,
        refusal_reason::EXT_RELEASE_BUDGET_FULL,
        &[],
        CAPACITY_RETRY_DELAY,
    );
    let gate_error = ExtReleaseRefusal::Full.into_service_error();
    assert_eq!(
        assert_refusal_error(
            &gate_error,
            refusal_reason::EXT_RELEASE_BUDGET_FULL,
            &[],
            CAPACITY_RETRY_DELAY,
        ),
        status.message(),
        "the handler answers the gate's own refusal"
    );
}

/// The store-write refusal counter, summed: a gate refusal must not tick it.
fn store_write_refusals(metrics: &Metrics) -> f64 {
    live_edit::observability::metrics::test_util::sample_sum(
        &render(metrics),
        "live_edit_store_write_refused_total",
        &[],
    )
    .unwrap_or(0.0)
}

fn create_section() -> StructCommand {
    StructCommand::CreateSection(CommandCreateSection {
        parent_id: 0,
        section_id: SECTION_ID,
        section_name: "Asset".to_owned(),
        section_type: 1,
    })
}

/// A part and the delta referencing it as an external value.
fn external_upload(seed: u8) -> (OwnedDelta, Vec<u8>) {
    let part: Vec<u8> = (0..2_048).map(|i| (i as u8).wrapping_add(seed)).collect();
    let delta = OwnedDelta {
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: "v".to_owned(),
            value_or_ext_hash: ext_content_hash(&part).to_vec(),
            ext_value_size: part.len() as u64,
            ext_value_index: 1,
            set_order: 0,
        }],
        ..Default::default()
    };
    (delta, part)
}

fn inline_upload(v: u8) -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: "inline".to_owned(),
            value_or_ext_hash: vec![v],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    }
}

/// A session and a layer whose first commit creates [`SECTION_ID`].
async fn layer(tv: &TestValkey, svc: &LiveEditService, name: &str) -> (String, u64) {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(&format!("{name}-session"), "/stage", "test-user")
        .await
        .expect("session");
    let delta = OwnedDelta {
        struct_commands: vec![create_section()],
        ..Default::default()
    };
    let object_id = handle_create_layer(
        svc,
        make_request(streaming_create(vec![
            make_create_header(&session_id, name),
            make_create_data(delta_to_ompf_bytes(&delta, &[])),
        ])),
    )
    .await
    .expect("the setup CreateLayer must succeed")
    .into_inner()
    .object_id;
    (session_id, object_id)
}

async fn update(
    svc: &LiveEditService,
    session_id: &str,
    name: &str,
    object_id: u64,
    delta: &OwnedDelta,
    parts: &[&[u8]],
) -> Result<(), tonic::Status> {
    handle_update_layer(
        svc,
        make_request(streaming_update(vec![
            make_update_header(session_id, name, object_id),
            make_update_data(delta_to_ompf_bytes(delta, parts)),
        ])),
    )
    .await
    .map(|_| ())
}

async fn seq_and_stream_len(tv: &TestValkey, object_id: u64) -> (u64, u64) {
    let seq = crate::common::read_header_u64(tv, object_id, "seq")
        .await
        .expect("seq");
    let len: u64 = tv
        .client()
        .pool
        .xlen(tv.key_schema().layer_stream(object_id))
        .await
        .expect("XLEN");
    (seq, len)
}

async fn ext_keys_present(tv: &TestValkey, part: &[u8]) -> i64 {
    let hex = sha256_to_hex(&ext_content_hash(part));
    tv.client()
        .pool
        .exists(vec![
            tv.key_schema().ext_meta(&hex),
            tv.key_schema().ext_data(&hex),
        ])
        .await
        .expect("EXISTS")
}

/// A gate reader of the stored release total (`le:ext:{gc}:pending:total`,
/// absent reading 0, as the store source reads it) that counts its reads in
/// `reads`: the gate behaves as over the store, and a test can see each read.
fn counting_store_reader(tv: &TestValkey, reads: Arc<AtomicUsize>) -> TotalReaderForTest {
    let (client, key) = (tv.client().clone(), tv.key_schema().ext_pending_total());
    Arc::new(
        move || -> futures::future::BoxFuture<'static, Option<u64>> {
            let (client, key, reads) = (client.clone(), key.clone(), Arc::clone(&reads));
            Box::pin(async move {
                let total: Result<Option<u64>, _> = client.pool.get(&key).await;
                reads.fetch_add(1, Ordering::SeqCst);
                total.ok().map(|total| total.unwrap_or(0))
            })
        },
    )
}

/// At a full budget an `UpdateLayer` carrying an external value waits, and
/// proceeds once the total drops below the budget within its wait: counted
/// `passed_after_wait`, the value stored and the commit landed. The gate reads
/// the stored total through [`counting_store_reader`], so the total is drained
/// only once the call is known to wait: the gate's shared poller, which only a
/// waiting call starts, has read it after the call's own first read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_update_with_external_values_waits_at_a_full_budget_and_proceeds_when_it_drains() {
    let tv = TestValkey::setup().await;
    let reads = Arc::new(AtomicUsize::new(0));
    let (svc, metrics) = service_with_reader(
        &tv,
        counting_store_reader(&tv, Arc::clone(&reads)),
        EXT_RELEASE_GATE_DEADLINE_MARGIN + Duration::from_secs(60),
    );
    let name = "gate-wait-proceed";
    let (sid, oid) = layer(&tv, &svc, name).await;
    let (seq_before, _) = seq_and_stream_len(&tv, oid).await;
    set_total(&tv, BUDGET).await;
    let (delta, part) = external_upload(1);

    let svc = Arc::new(svc);
    let call = {
        let (svc, sid, delta, part) = (svc.clone(), sid.clone(), delta.clone(), part.clone());
        tokio::spawn(async move { update(&svc, &sid, name, oid, &delta, &[&part]).await })
    };
    // Evidence of waiting: a second read is the shared poller's. The sleep is
    // the probe interval, not a coordination primitive.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while reads.load(Ordering::SeqCst) < 2 {
        assert!(
            !call.is_finished(),
            "the upload ended before it waited at the full budget"
        );
        assert!(
            tokio::time::Instant::now() < deadline,
            "the upload never waited at the full budget"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(
        ext_keys_present(&tv, &part).await,
        0,
        "nothing stored while waiting"
    );
    set_total(&tv, BUDGET - 1).await;

    tokio::time::timeout(Duration::from_secs(30), call)
        .await
        .expect("the upload must proceed once the budget drains")
        .expect("no panic")
        .expect("the upload must succeed");
    assert_eq!(
        ext_keys_present(&tv, &part).await,
        2,
        "the value was stored"
    );
    assert_eq!(
        seq_and_stream_len(&tv, oid).await.0,
        seq_before + 1,
        "the commit landed"
    );
    assert_eq!(gate_cells(&metrics), (1, 0, 0));
}

/// An `UpdateLayer` and a `CreateLayer` carrying external values, at a budget
/// that stays full, are refused `UNAVAILABLE` "nothing applied" with the
/// `EXT_RELEASE_BUDGET_FULL` retry hint: no value stored or incremented, no
/// commit, no stream entry, no new layer — counted `refused`, and not as a
/// store-write refusal (nothing reached the store).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_upload_with_external_values_at_a_budget_that_stays_full_applies_nothing() {
    let tv = TestValkey::setup().await;
    let (svc, metrics) = gated_service(&tv, Duration::from_millis(500));
    let name = "gate-refuse";
    let (sid, oid) = layer(&tv, &svc, name).await;
    set_total(&tv, BUDGET + 10).await;
    let before = seq_and_stream_len(&tv, oid).await;
    let refusals_before = store_write_refusals(&metrics);

    let (delta, part) = external_upload(2);
    let status = update(&svc, &sid, name, oid, &delta, &[&part])
        .await
        .expect_err("a budget that stays full must refuse");
    assert_budget_refusal(&status);
    assert_eq!(
        ext_keys_present(&tv, &part).await,
        0,
        "no value stored or incremented"
    );
    assert_eq!(
        seq_and_stream_len(&tv, oid).await,
        before,
        "no commit, no stream entry"
    );

    // CreateLayer: refused the same way, and no layer is created.
    let (create_delta, create_part) = {
        let (mut d, p) = external_upload(3);
        d.struct_commands = vec![create_section()];
        (d, p)
    };
    let status = handle_create_layer(
        &svc,
        make_request(streaming_create(vec![
            make_create_header(&sid, "gate-refuse-create"),
            make_create_data(delta_to_ompf_bytes(&create_delta, &[&create_part])),
        ])),
    )
    .await
    .expect_err("a create carrying external values at a full budget must refuse");
    assert_budget_refusal(&status);
    assert_eq!(ext_keys_present(&tv, &create_part).await, 0);
    let mapped: bool = tv
        .client()
        .pool
        .exists(tv.key_schema().layer_mapping(&sid, "gate-refuse-create"))
        .await
        .expect("EXISTS layer mapping");
    assert!(!mapped, "no layer was created");

    assert_eq!(gate_cells(&metrics), (0, 2, 0));
    assert_eq!(
        store_write_refusals(&metrics),
        refusals_before,
        "a gate refusal is not a store-write refusal"
    );
}

/// Inline-only uploads and `DeleteLayer` are never gated: at a full budget,
/// with a 20 s wait at the gate, they proceed and the gate counts nothing —
/// not `refused`, not `passed_after_wait`, which a call that waited at the
/// gate and then went through would count.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inline_only_uploads_and_delete_layer_are_never_gated() {
    let tv = TestValkey::setup().await;
    let (svc, metrics) = gated_service(&tv, Duration::from_secs(20));
    let name = "gate-inline";
    let (sid, oid) = layer(&tv, &svc, name).await;
    set_total(&tv, BUDGET * 10).await;

    update(&svc, &sid, name, oid, &inline_upload(9), &[])
        .await
        .expect("an inline-only upload is never gated");
    let mut request = Request::new(DeleteLayerRequest {
        session_id: sid.clone(),
        layer_name: name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    handle_delete_layer(&svc, request)
        .await
        .expect("DeleteLayer is never gated");
    assert_eq!(
        gate_cells(&metrics),
        (0, 0, 0),
        "neither call reached the gate: no outcome counted"
    );
}

/// A total the gate cannot read lets an external upload through, counted
/// `read_failed`; and a store with no gate (a budget of 0 attaches none)
/// admits it at any total. The planted total is a string that is not an
/// integer: the gate reads it with `MGET` (beside the upload hold), which
/// answers nil for a key of another type, so a wrong-typed key would read as
/// 0 and be admitted uncounted instead (pinned in `ext_release_gate`'s unit
/// tests).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unreadable_total_or_a_disabled_budget_lets_the_upload_through() {
    let tv = TestValkey::setup().await;
    let (svc, metrics) = gated_service(&tv, Duration::from_secs(20));
    let name = "gate-read-failed";
    let (sid, oid) = layer(&tv, &svc, name).await;
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().ext_pending_total(),
            "not-an-integer",
            None,
            None,
            false,
        )
        .await
        .expect("plant a total that is not an integer");
    let (delta, part) = external_upload(4);
    update(&svc, &sid, name, oid, &delta, &[&part])
        .await
        .expect("an unreadable total lets the upload through");
    assert_eq!(gate_cells(&metrics), (0, 0, 1));

    let tv2 = TestValkey::setup().await;
    let ungated = LiveEditService {
        ext_value_store: Arc::new(ValkeyExternalValueStore::new(
            tv2.client().clone(),
            tv2.key_schema().clone(),
        )),
        ..build_service(&tv2, Arc::new(AllowAllPermissions))
    };
    assert!(
        ungated.ext_value_store.ext_release_gate().is_none(),
        "precondition: a disabled budget attaches no gate"
    );
    let (sid2, oid2) = layer(&tv2, &ungated, "gate-disabled").await;
    set_total(&tv2, u64::MAX / 2).await;
    let (delta, part) = external_upload(5);
    update(&ungated, &sid2, "gate-disabled", oid2, &delta, &[&part])
        .await
        .expect("without a gate nothing waits");
    assert_eq!(
        gate_cells(&ungated.metrics),
        (0, 0, 0),
        "no gate outcome is counted without a gate"
    );
    assert_eq!(
        ext_keys_present(&tv2, &part).await,
        2,
        "the value was stored at a total far above any budget"
    );
}

/// A graceful shutdown while an `UpdateLayer` waits at a full budget ends the
/// wait with the shared shutdown status, `UNAVAILABLE "server shutting
/// down"` with its `SERVER_SHUTTING_DOWN` retry hint, and its completion line is the shutdown-caused one — INFO, not the
/// WARN of a budget refusal — with nothing applied and no gate outcome
/// counted. The call is known to be waiting once the gate's shared poller has
/// read the total (only a waiting call starts it).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_shutdown_while_an_upload_waits_at_the_gate_answers_the_shutdown_status() {
    let tv = TestValkey::setup().await;
    let reads = Arc::new(AtomicUsize::new(0));
    let (svc, metrics) = service_with_reader(
        &tv,
        counting_reader(Arc::clone(&reads), Arc::new(Mutex::new(None))),
        Duration::from_secs(600),
    );
    let name = "gate-shutdown";
    let (sid, oid) = layer(&tv, &svc, name).await;
    let before = seq_and_stream_len(&tv, oid).await;
    let (delta, part) = external_upload(6);

    let shutdown = svc.shutdown.clone();
    let cancel_once_waiting = async {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
        while reads.load(Ordering::SeqCst) < 2 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the upload never waited at the gate"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        shutdown.cancel();
    };
    let ((result, captured), ()) = tokio::join!(
        async {
            tokio::time::timeout(
                Duration::from_secs(60),
                capture(update(&svc, &sid, name, oid, &delta, &[&part])),
            )
            .await
            .expect("the shutdown must end the wait (the gate would otherwise wait 595 s)")
        },
        cancel_once_waiting
    );
    let status = result.expect_err("a shutdown must end the gate's wait");
    assert_eq!(status.code(), tonic::Code::Unavailable, "{status:?}");
    assert_shutdown_status(&status);
    let event = captured.only(RPC_FAILURE_MSG);
    assert_common_fields(event, "INFO", "UpdateLayer", "parse", "UNAVAILABLE");
    assert_eq!(
        ext_keys_present(&tv, &part).await,
        0,
        "no value stored or incremented"
    );
    assert_eq!(
        seq_and_stream_len(&tv, oid).await,
        before,
        "no commit, no stream entry"
    );
    assert_eq!(
        gate_cells(&metrics),
        (0, 0, 0),
        "a shutdown is not a budget refusal"
    );
}

/// The gate's wait follows the call's own remaining deadline: inside a
/// deadline scope of `margin + 800 ms` — far shorter than the RPC deadline
/// (600 s, a 595 s wait) — an upload at a full budget waits about 800 ms, then
/// is refused `UNAVAILABLE` "nothing applied", with the
/// `EXT_RELEASE_BUDGET_FULL` retry hint. The reader drops the total
/// below the budget at 800 ms + half the margin: a wait bounded by the RPC
/// deadline instead (the handler passing no remaining deadline), or by the
/// whole remaining deadline (no margin), would see that drop and let the
/// upload through. The call waited: the shared poller read the total.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_gate_waits_the_calls_remaining_deadline_less_the_margin() {
    let tv = TestValkey::setup().await;
    let reads = Arc::new(AtomicUsize::new(0));
    let drops_at = Arc::new(Mutex::new(None));
    let (svc, metrics) = service_with_reader(
        &tv,
        counting_reader(Arc::clone(&reads), Arc::clone(&drops_at)),
        Duration::from_secs(600),
    );
    let name = "gate-call-deadline";
    let (sid, oid) = layer(&tv, &svc, name).await;
    let before = seq_and_stream_len(&tv, oid).await;
    let (delta, part) = external_upload(7);
    let wait = Duration::from_millis(800);

    let started = tokio::time::Instant::now();
    *drops_at.lock().unwrap() = Some(started + wait + EXT_RELEASE_GATE_DEADLINE_MARGIN / 2);
    let call_deadline = started + EXT_RELEASE_GATE_DEADLINE_MARGIN + wait;
    let status = tokio::time::timeout(
        Duration::from_secs(60),
        live_edit::grpc::deadline::scope(
            call_deadline,
            update(&svc, &sid, name, oid, &delta, &[&part]),
        ),
    )
    .await
    .expect("the wait must end by the call's deadline, not the RPC deadline's 595 s")
    .expect_err("the budget stayed full for the whole wait the call's deadline allows");
    // A budget refusal, not a shutdown.
    assert_budget_refusal(&status);
    assert!(
        reads.load(Ordering::SeqCst) >= 2,
        "the call waited at the gate (the poller read the total) rather than refusing at once"
    );
    assert_eq!(
        gate_cells(&metrics),
        (0, 1, 0),
        "refused, never passed after the drop"
    );
    assert_eq!(ext_keys_present(&tv, &part).await, 0);
    assert_eq!(seq_and_stream_len(&tv, oid).await, before);
}
