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

//! b.yuo part D — the catch-up external-value miss protocol, through
//! `handle_subscribe_read_layer` against a real Valkey and a real
//! `ValkeyExternalValueStore` (SDD §11.2, §11.3): on a catch-up's own atomic
//! read, and on a flight's shared fenced snapshot, whose late waiter
//! re-attaches to a successor flight.
//!
//! A blob a catch-up's snapshot references can be collected before the build
//! reads it: a commit after the snapshot released it and the GC took it. The
//! protocol treats that as a stale read — probe, give back the stale
//! snapshot, take a per-holder `{gc}` read hold, re-read, rebuild — and only a
//! blob still missing under a hold the store confirms live takes fetch-on-miss
//! through the checkpoint adapter.
//!
//! | Area | Tests |
//! |---|---|
//! | Test 6, the stale-miss regression, per regime the helper serves | [`a_brand_new_stale_miss_rebuilds_fresh_state_at_the_new_cut`], [`a_brand_new_legacy_mode_stale_miss_rebuilds_fresh_state_at_the_new_cut`], [`an_in_window_stale_miss_rebuilds_fresh_state_at_the_new_cut`], [`an_out_of_window_stale_miss_rebuilds_fresh_state_at_the_new_cut`], [`an_out_of_window_legacy_mode_stale_miss_rebuilds_fresh_state_at_the_new_cut`], [`a_rollback_replay_stale_miss_rebuilds_its_commit_from_a_fresh_read`] |
//! | Test 7, the genuine miss | [`a_genuine_miss_under_a_live_hold_recovers_through_the_adapter`], [`a_genuine_miss_the_adapter_cannot_restore_ends_internal`], [`a_hold_lost_before_the_genuine_check_ends_unavailable_never_internal`], [`a_stale_miss_whose_layer_was_overwritten_rebootstraps_on_the_new_object`], [`a_stale_miss_whose_layer_was_deleted_ends_not_found`] |
//! | The protocol's other endings, each releasing the hold or never holding one | [`a_hold_that_cannot_be_taken_ends_unavailable_never_internal`], [`a_genuine_check_whose_renewal_fails_ends_unavailable_never_internal`], [`a_re_read_whose_cut_regressed_ends_unavailable_before_content`], [`a_per_commit_re_read_handed_to_a_flight_releases_its_hold_first`], [`a_per_commit_re_read_too_large_hands_back_to_a_flight_with_its_hold_released`], [`a_per_commit_re_read_whose_cut_regressed_ends_unavailable_before_content`], [`a_re_read_out_of_capacity_budget_ends_with_its_capacity_ending_and_releases_the_hold`], [`a_re_read_waiting_for_its_permit_at_shutdown_ends_as_is_and_releases_the_hold`] (the unreleased guard's drop is pinned at the store, `b_yuo_gc_tests::a_hold_dropped_unreleased_deletes_its_field`); the cut-based re-read's three endings here also leave `live_edit_stream_mux_subscriptions` at its pre-subscribe level |
//! | The re-read finding the layer overwritten, per regime, the stream registrations back at their pre-subscribe level | [`a_stale_miss_whose_layer_was_overwritten_rebootstraps_on_the_new_object`], [`an_in_window_stale_miss_whose_layer_was_overwritten_rebootstraps`], [`an_out_of_window_stale_miss_whose_layer_was_overwritten_rebootstraps`], [`an_out_of_window_legacy_mode_stale_miss_whose_layer_was_overwritten_rebootstraps`], [`a_rollback_replay_stale_miss_whose_layer_was_overwritten_rebootstraps`] |
//! | The re-read finding the layer's state gone (lost, or overwritten for the rollback replay), with checkpointing on: at a cut-based site the recovery runs once the hold is released, and the restored layer is caught up again (or the recovery's failure ends UNAVAILABLE); the rollback replay never asks the recovery | [`a_brand_new_re_read_that_finds_the_state_lost_restores_it_and_delivers_it`], [`an_in_window_re_read_that_finds_the_state_lost_restores_it_and_delivers_it`], [`an_out_of_window_re_read_that_finds_the_state_lost_restores_it_and_delivers_it`], [`a_re_read_whose_recovery_fails_transiently_ends_unavailable`], [`re_read_through_a_flight::a_fenced_re_attach_that_finds_the_state_lost_restores_it_and_delivers_it`]; a restored re-run charged to the budget the first read left, ending with the capacity refusal when it runs out, [`a_restored_re_run_out_of_its_capacity_budget_ends_with_the_capacity_refusal`]; the rollback replay's per-commit re-read re-bootstrapping without asking the recovery, [`a_rollback_replay_re_read_that_finds_the_state_gone_rebootstraps_without_recovery`] |
//! | H-b, renewal through the rebuild | [`a_rebuild_parked_across_hold_renewals_stays_protected`] |
//! | H-c, a lapse the store reports | [`a_renewal_that_finds_the_hold_lapsed_ends_the_catch_up_unavailable`] |
//! | 6f-a, a stale fenced miss re-attaching to one successor flight, with the order pin (the old flight's turn and place given back before the hold is taken) | [`re_read_through_a_flight::a_stale_fenced_miss_re_attaches_to_one_successor_flight_and_serves_its_cut`] |
//! | 6f-b, two late waiters sharing one successor read | [`re_read_through_a_flight::two_late_waiters_that_miss_share_one_successor_read`] |
//! | 6f-c, a cut-based atomic re-read routed to a flight under its hold | [`re_read_through_a_flight::a_cut_based_re_read_answering_too_large_hands_over_to_a_flight_under_its_hold`], [`re_read_through_a_flight::a_cut_based_re_read_finding_the_objects_flight_registered_reads_through_it_under_its_hold`] |
//! | 6f-d, the hold covering the re-attach to the last blob read | [`re_read_through_a_flight::the_hold_covers_the_re_attach_from_the_successors_cut_to_its_last_blob_read`] |
//! | A successor whose cut is fixed before the late waiter's hold, its take held back: no attach before the take; joined and every blob delivered; closed and never shipped a collected blob | [`re_read_through_a_flight::a_successor_whose_cut_was_fixed_before_the_late_waiters_hold_is_joined_and_delivers_every_blob_it_references`], [`re_read_through_a_flight::a_successor_that_closes_before_the_late_waiters_hold_is_not_joined_and_no_collected_blob_is_shipped`] |
//! | 7f, the genuine fenced miss, the lost hold, and `NotFound` on the re-attach | [`re_read_through_a_flight::a_genuine_fenced_miss_under_a_live_hold_recovers_through_the_adapter`], [`re_read_through_a_flight::a_genuine_fenced_miss_the_adapter_cannot_restore_ends_internal`], [`re_read_through_a_flight::a_fenced_re_attach_whose_hold_is_lost_ends_read_hold_refused`], [`re_read_through_a_flight::a_fenced_re_attach_whose_layer_was_overwritten_rebootstraps_on_the_new_object`], [`re_read_through_a_flight::a_fenced_re_attach_that_finds_the_state_lost_restores_it_and_delivers_it`] |
//! | H-f, the hold and every stream registration released on each exit of the re-attach, and 6f-e, the regression pin | [`re_read_through_a_flight::a_re_attach_ended_by_its_client_releases_its_hold_and_registrations`], [`re_read_through_a_flight::a_re_attach_ended_by_shutdown_releases_its_hold_and_registrations`], [`re_read_through_a_flight::a_re_attach_whose_successor_read_fails_releases_its_hold_and_registrations`], [`re_read_through_a_flight::a_re_attach_out_of_capacity_budget_releases_its_hold_and_registrations`], [`re_read_through_a_flight::a_re_attach_past_the_reader_wait_cap_releases_its_hold_and_registrations`], [`re_read_through_a_flight::a_re_attach_whose_registration_hits_the_hard_backlog_cap_releases_its_hold_and_registrations`], [`re_read_through_a_flight::a_re_attach_waiting_for_its_build_turn_whose_registration_hits_the_hard_backlog_cap_releases_its_hold_and_registration`], [`re_read_through_a_flight::a_re_attach_whose_successor_cut_regressed_ends_unavailable_and_releases_its_hold`]; the re-read's `NotFound` and `read_hold_refused` in the 7f row |
//! | 6-swap, the registration swapped before the re-read waits | [`re_read_through_a_flight::a_re_read_waiting_for_its_permit_keeps_no_registration_at_the_first_cut`] |
//! | The D-helper pin (a source scan) | [`the_read_mode_and_every_builder_call_stay_inside_the_miss_protocol`], [`build_catch_up_once_is_reached_only_through_the_miss_protocol`], [`every_catch_up_regime_builds_through_build_catch_up`] |
//! | The miss counter's cells | [`the_miss_counter_is_zero_seeded_over_outcome_and_read`] |
//!
//! The hold's own lifecycle (no chaining, re-send, crash lapse, the `renew`
//! script's answers) and `gc_select.lua`'s reading of it are pinned at the
//! store, in `tests/integration/storage/b_yuo_gc_tests.rs`.
//!
//! # How a miss is staged
//!
//! A real one: the layer's field references blob `old`; the catch-up is parked
//! after its snapshot and before its first blob read; a commit rewrites the
//! field to `fresh`, which releases `old`; a GC pass under
//! `GcPolicy::IMMEDIATE` collects it; the catch-up is released. Test 6 parks
//! on the service's own seam (`CatchUpStateReadSeam::new_before_ext_probe`,
//! `test-fault-injection`). The other tests park, or act, inside the external
//! value store's `probe_values` ([`ScriptedProbeStore`]), which the build
//! calls at the same point and again after the re-read — the only place a
//! test can stand between the re-read and the rebuild's blob read. A genuine
//! miss is staged by deleting the blob's keys out of band, so the re-read
//! misses it too. A miss on a flight's shared snapshot is staged the same way
//! on a catch-up client whose atomic-read limits are below the layer (the
//! [`re_read_through_a_flight`] module docs).
//!
//! "Zero adapter calls" is read off `live_edit_checkpoint_ext_recovery_total`:
//! the store carries an `InMemoryCheckpointStore`, and every consult of it
//! ticks one of that counter's `result` cells.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::{ClientLike, HashesInterface, KeysInterface, StreamsInterface};
use fred::types::streams::XCap;
use futures::{Stream, StreamExt};
use tonic::{Code, Request, Status};

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta, CheckpointStore,
    LayerArtifact, RecoveryHook,
};
use live_edit::domain::ompf::OmpfContainer;
use live_edit::error::ServiceError;
use live_edit::grpc::layer::{handle_create_layer, handle_delete_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, DeleteLayerRequest, SubscribeReadLayerHeader,
    SubscribeReadLayerRequest, SubscribeReadLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::ext_release_gate::ExtReleaseGate;
use live_edit::storage::external_values::{
    ext_content_hash, ExtRelease, ExtValueProbe, ExtValueVerify, ExternalValueStore, GcBatchResult,
    GcPassState, GcPolicy, ReadHoldTiming, ValkeyExternalValueStore,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::snapshot_fence::LayerSize;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{
    CommandCreateSection, CommandDiffChangeSection, OwnedDelta, SetField, StructCommand,
};

use super::helpers::*;
use crate::common::owned_delta::delta_to_ompf_bytes;

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// Bound on every wait for the subscriber, the GC or a park: generous, since
/// a deadline costs nothing on the healthy path.
const DEADLINE: Duration = Duration::from_secs(30);

/// How long a stream must stay silent for "nothing else arrives".
#[cfg(feature = "test-fault-injection")]
const QUIET: Duration = Duration::from_secs(2);

/// The section every field here lives in.
const SECTION: u64 = 7;

/// The field whose external value the commits rewrite.
const FIELD: &str = "mesh:points";

/// A distinct 2 KiB external value per `seed`.
fn blob(seed: u8) -> Vec<u8> {
    (0..2_048u32)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

/// A service on `tv` whose external values live in a real
/// `ValkeyExternalValueStore` — with the service's metrics, and an empty
/// `InMemoryCheckpointStore` as its adapter (returned, so a test can seed
/// it) — wrapped by `wrap`, in catch-up snapshot mode `snapshot_mode`.
fn service_with(
    tv: &TestValkey,
    snapshot_mode: bool,
    wrap: impl FnOnce(ValkeyExternalValueStore) -> Arc<dyn ExternalValueStore>,
) -> (LiveEditService, Arc<InMemoryCheckpointStore>) {
    let base = build_service(tv, Arc::new(AllowAllPermissions));
    let adapter = Arc::new(InMemoryCheckpointStore::new());
    let store = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_metrics(Arc::clone(&base.metrics))
        .with_checkpoint_store(Arc::clone(&adapter) as Arc<dyn CheckpointStore>);
    let svc = LiveEditService {
        ext_value_store: wrap(store),
        catchup_snapshot_enabled: snapshot_mode,
        ..base
    };
    (svc, adapter)
}

/// The commit that writes [`FIELD`] as the external value `value`, creating
/// [`SECTION`] first when `create_section`.
async fn commit_ext(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    object_id: u64,
    create_section: bool,
    value: &[u8],
) {
    let delta = OwnedDelta {
        struct_commands: if create_section {
            vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: SECTION,
                section_name: "Mesh".to_owned(),
                section_type: 1,
            })]
        } else {
            Vec::new()
        },
        set_fields: vec![SetField {
            section_id: SECTION,
            key_name: FIELD.to_owned(),
            value_or_ext_hash: ext_content_hash(value).to_vec(),
            ext_value_size: value.len() as u64,
            ext_value_index: 1,
            set_order: 0,
        }],
        ..Default::default()
    };
    upload(
        svc,
        session_id,
        layer,
        object_id,
        delta_to_ompf_bytes(&delta, &[value]),
    )
    .await;
}

/// A commit carrying no external value: an inline `marker` field.
async fn commit_inline(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    object_id: u64,
    marker: u8,
) {
    let delta = OwnedDelta {
        set_fields: vec![SetField {
            section_id: SECTION,
            key_name: "marker".to_owned(),
            value_or_ext_hash: vec![marker],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    };
    upload(
        svc,
        session_id,
        layer,
        object_id,
        delta_to_ompf_bytes(&delta, &[]),
    )
    .await;
}

async fn upload(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    object_id: u64,
    ompf: Vec<u8>,
) {
    let request = make_request(streaming_update(vec![
        make_update_header(session_id, layer, object_id),
        make_update_data(ompf),
    ]));
    handle_update_layer(svc, request)
        .await
        .unwrap_or_else(|e| panic!("UpdateLayer on {layer} failed: {e}"));
}

/// A `CreateLayer` of `layer` with no content: an overwrite when it exists.
async fn create_layer(svc: &LiveEditService, session_id: &str, layer: &str) -> u64 {
    let ompf = delta_to_ompf_bytes(&crate::common::owned_delta::empty_delta(), &[]);
    let request = make_request(streaming_create(vec![
        make_create_header(session_id, layer),
        make_create_data(ompf),
    ]));
    handle_create_layer(svc, request)
        .await
        .expect("CreateLayer must succeed")
        .into_inner()
        .object_id
}

fn subscribe_request(
    session_id: &str,
    layer: &str,
    object_id: u64,
    sequence_num: u64,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer.to_owned(),
        object_id,
        sequence_num,
        client_id: "b-yuo-subscriber".to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

type SubscribeStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>;

async fn subscribe(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    have_object_id: u64,
    have_version: u64,
) -> SubscribeStream {
    let response = handle_subscribe_read_layer(
        svc,
        subscribe_request(session_id, layer, have_object_id, have_version),
    )
    .await
    .expect("SubscribeReadLayer must open");
    Box::pin(response.into_inner())
}

/// The next message — a header and its reassembled body — or the status the
/// stream ended with in its place.
async fn next_message(
    stream: &mut SubscribeStream,
) -> Result<(SubscribeReadLayerHeader, Vec<u8>), Status> {
    let first = tokio::time::timeout(DEADLINE, stream.next())
        .await
        .expect("the subscriber sent nothing within the deadline")
        .expect("the stream closed without a message or a status")?;
    let header = match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(header)) => header,
        other => panic!("expected a header, got {other:?}"),
    };
    let mut body = Vec::new();
    while (body.len() as u64) < header.total_size {
        let next = tokio::time::timeout(DEADLINE, stream.next())
            .await
            .expect("a body chunk did not arrive within the deadline")
            .expect("the stream closed inside a message")?;
        match next.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(chunk)) => {
                body.extend_from_slice(&chunk.data)
            }
            other => panic!("expected a body chunk, got {other:?}"),
        }
    }
    Ok((header, body))
}

/// The next message, which must be content (not a status, not an ack).
async fn next_content(
    stream: &mut SubscribeStream,
    what: &str,
) -> (SubscribeReadLayerHeader, Vec<u8>) {
    let (header, body) = next_message(stream)
        .await
        .unwrap_or_else(|status| panic!("{what}: the stream ended {status:?} instead"));
    assert!(
        header.total_size > 0,
        "{what}: expected content, got an empty header {header:?}"
    );
    (header, body)
}

/// The caught-up ack after a cut-based catch-up's content: an empty header
/// echoing the content's object and sequence.
async fn expect_caught_up_ack(stream: &mut SubscribeStream, object_id: u64, sequence_num: u64) {
    let (ack, body) = next_message(stream)
        .await
        .unwrap_or_else(|status| panic!("the caught-up ack: the stream ended {status:?}"));
    assert_eq!(
        (ack.object_id, ack.sequence_num, ack.total_size, body.len()),
        (object_id, sequence_num, 0, 0),
        "the caught-up ack echoes the content's cut with no body"
    );
}

/// The stream's ending status, expected before any message.
async fn expect_ended_before_content(stream: &mut SubscribeStream, what: &str) -> Status {
    match next_message(stream).await {
        Err(status) => status,
        Ok((header, _)) => panic!("{what}: expected the stream to end, got {header:?}"),
    }
}

/// Nothing more arrives for [`QUIET`].
#[cfg(feature = "test-fault-injection")]
async fn assert_quiet(stream: &mut SubscribeStream) {
    match tokio::time::timeout(QUIET, stream.next()).await {
        Err(_) | Ok(None) => {}
        Ok(Some(item)) => panic!("expected nothing more, got {item:?}"),
    }
}

/// The external-value parts a payload carries.
fn carried(body: &[u8]) -> Vec<Vec<u8>> {
    OmpfContainer::parse(body)
        .expect("a payload parses as OMPF")
        .external_values
        .iter()
        .map(|part| part.to_vec())
        .collect()
}

/// The content hash [`FIELD`]'s `SetField` in `body` references, if any.
#[cfg(feature = "test-fault-injection")]
fn field_hash(body: &[u8]) -> Option<Vec<u8>> {
    let container = OmpfContainer::parse(body).expect("a payload parses as OMPF");
    if container.delta_bytes.is_empty() {
        return None;
    }
    oodf_ffi::parse_delta(container.delta_bytes)
        .expect("a payload's delta parses")
        .set_fields
        .into_iter()
        .find(|field| field.key_name == FIELD)
        .map(|field| field.value_or_ext_hash)
}

/// Collect `value` with GC passes under `GcPolicy::IMMEDIATE` until its data
/// key is gone: the GC taking a blob a later commit released.
async fn collect(tv: &TestValkey, value: &[u8]) {
    collect_with(tv.client(), tv.key_schema(), value).await;
}

/// [`collect`] from a client and key schema, for a task that holds no
/// `TestValkey`.
async fn collect_with(
    client: &ValkeyClient,
    keys: &live_edit::storage::keys::KeySchema,
    value: &[u8],
) {
    let gc = ValkeyExternalValueStore::new(client.clone(), keys.clone());
    let data_key = keys.ext_data(&sha256_to_hex(&ext_content_hash(value)));
    let deadline = tokio::time::Instant::now() + DEADLINE;
    loop {
        let stored: bool = client
            .pool
            .exists(&data_key)
            .await
            .expect("EXISTS ext:data");
        if !stored {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the GC never collected the released blob"
        );
        gc_pass(&gc).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

async fn gc_pass(gc: &ValkeyExternalValueStore) -> GcBatchResult {
    gc.process_pending(&GcPolicy::IMMEDIATE, &mut GcPassState::new())
        .await
        .expect("a GC pass must succeed")
}

async fn blob_stored(tv: &TestValkey, value: &[u8]) -> bool {
    tv.client()
        .pool
        .exists(
            tv.key_schema()
                .ext_data(&sha256_to_hex(&ext_content_hash(value))),
        )
        .await
        .expect("EXISTS ext:data")
}

/// Delete `value`'s keys out of band, as a keyspace loss would: the layer
/// still references it, and nothing will release it.
async fn wipe(tv: &TestValkey, value: &[u8]) {
    let hex = sha256_to_hex(&ext_content_hash(value));
    let removed: i64 = tv
        .client()
        .pool
        .unlink(vec![
            tv.key_schema().ext_meta(&hex),
            tv.key_schema().ext_data(&hex),
        ])
        .await
        .expect("UNLINK ext keys");
    assert_eq!(removed, 2, "precondition: the blob was stored");
}

/// Fields on the `{gc}` read hold.
async fn hold_fields(tv: &TestValkey) -> u64 {
    tv.client()
        .pool
        .hlen(tv.key_schema().ext_read_hold())
        .await
        .expect("HLEN read_hold")
}

/// `live_edit_stream_mux_subscriptions`: stream reservations and
/// registrations held now on the catch-up client whose observability hooks
/// carry `metrics` ([`count_subscriptions`] on the atomic-path services; the
/// fenced fixture installs them itself).
fn subscriptions(metrics: &Metrics) -> i64 {
    sample_value(&render(metrics), "live_edit_stream_mux_subscriptions", &[])
        .expect("live_edit_stream_mux_subscriptions is pre-initialised") as i64
}

/// Count `svc`'s stream reservations and registrations on its own metrics
/// ([`subscriptions`]): install the service's metrics as its catch-up
/// client's observability hooks, which the multiplexer reports through.
/// Returns the gauge's level before any subscribe.
async fn count_subscriptions(svc: &LiveEditService) -> i64 {
    svc.valkey
        .install_observability_hooks(Arc::clone(&svc.metrics))
        .await;
    subscriptions(&svc.metrics)
}

/// Wait until [`subscriptions`] is at `level`, within [`DEADLINE`]: with
/// `level` the pre-subscribe level, neither a re-read's reservation nor a
/// registration outlives its subscription. The sleep is the probe interval,
/// not a coordination primitive.
async fn until_subscriptions(metrics: &Metrics, level: i64, what: &str) {
    let deadline = tokio::time::Instant::now() + DEADLINE;
    loop {
        let now = subscriptions(metrics);
        if now == level {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what}: live_edit_stream_mux_subscriptions is {now}, expected {level}, after \
             {DEADLINE:?}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// `live_edit_catchup_ext_value_miss_total{outcome, read}`, zero-seeded.
fn misses(metrics: &Metrics, outcome: &str, read: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_catchup_ext_value_miss_total",
        &[("outcome", outcome), ("read", read)],
    )
    .unwrap_or_else(|| panic!("miss cell {{outcome={outcome}, read={read}}} is zero-seeded"))
        as u64
}

/// `(stale_retried, genuine)` on the atomic read, with the shared-fenced
/// cells checked to be zero: every miss a caller of this counts is found in
/// a catch-up's own atomic snapshot (a re-read may then read through a
/// flight, and its build from the flight's snapshot finds nothing missing).
fn atomic_misses(metrics: &Metrics) -> (u64, u64) {
    for outcome in ["stale_retried", "genuine"] {
        assert_eq!(
            misses(metrics, outcome, "shared_fenced"),
            0,
            "no miss here is found in a flight's shared snapshot"
        );
    }
    (
        misses(metrics, "stale_retried", "atomic"),
        misses(metrics, "genuine", "atomic"),
    )
}

/// `live_edit_checkpoint_ext_recovery_total{result}`, zero-seeded.
fn adapter_result(metrics: &Metrics, result: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_checkpoint_ext_recovery_total",
        &[("result", result)],
    )
    .unwrap_or_else(|| panic!("recovery cell {result} is zero-seeded")) as u64
}

/// Every consult of the checkpoint adapter (`get_ext_blob`), whatever it
/// answered.
fn adapter_consults(metrics: &Metrics) -> u64 {
    [
        "recovered",
        "adapter_miss",
        "adapter_transient",
        "adapter_permanent",
        "adapter_other",
    ]
    .into_iter()
    .map(|result| adapter_result(metrics, result))
    .sum()
}

// ── An external-value store that acts inside `probe_values` ──────────────────

/// What [`ScriptedProbeStore`] does on one `probe_values` call before it
/// delegates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OnProbe {
    /// Nothing.
    Pass,
    /// Announce the park and wait until the test releases it.
    Park,
    /// Delete the `{gc}` read hold key — every holder's field — as a `{gc}`
    /// failover that lost an unreplicated hold write would leave it.
    DropReadHolds,
    /// Replace the `{gc}` read hold key with a string, so every later
    /// `ext_read_hold.lua` call on it fails with `WRONGTYPE`.
    PoisonReadHolds,
}

/// A real store whose `probe_values` calls run `script` by call index (then
/// [`OnProbe::Pass`]). A catch-up build probes after its snapshot, before any
/// blob read, and again after the miss protocol's re-read; so call 0 stands
/// where the service's pre-ext-probe seam does, and call 1 between the re-read
/// and the rebuild's blob read, under the hold.
///
/// Every trait method is forwarded, the defaulted ones included: an omitted
/// arm would substitute the default body, not pass through.
struct ScriptedProbeStore {
    inner: ValkeyExternalValueStore,
    client: ValkeyClient,
    hold_key: String,
    script: Vec<OnProbe>,
    calls: AtomicUsize,
    parked: tokio::sync::Notify,
    released: tokio::sync::Notify,
}

impl ScriptedProbeStore {
    fn new(inner: ValkeyExternalValueStore, tv: &TestValkey, script: Vec<OnProbe>) -> Self {
        Self {
            inner,
            client: tv.client().clone(),
            hold_key: tv.key_schema().ext_read_hold(),
            script,
            calls: AtomicUsize::new(0),
            parked: tokio::sync::Notify::new(),
            released: tokio::sync::Notify::new(),
        }
    }

    /// Wait until a [`OnProbe::Park`] call has parked.
    async fn wait_parked(&self) {
        tokio::time::timeout(DEADLINE, self.parked.notified())
            .await
            .expect("the catch-up never reached the parking probe");
    }

    /// Let the parked call go on.
    fn release(&self) {
        self.released.notify_one();
    }
}

#[async_trait::async_trait]
impl ExternalValueStore for ScriptedProbeStore {
    async fn store_values(&self, values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
        self.inner.store_values(values).await
    }

    async fn store_verified_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), ServiceError> {
        self.inner.store_verified_values(values).await
    }

    fn ext_value_verify(&self) -> ExtValueVerify<'_> {
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
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        self.inner.read_batch(hashes).await
    }

    async fn read_batch_with_recovery_counter(
        &self,
        hashes: &[[u8; 32]],
        recovery_counter: Option<&AtomicUsize>,
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        self.inner
            .read_batch_with_recovery_counter(hashes, recovery_counter)
            .await
    }

    async fn probe_values(&self, hashes: &[[u8; 32]]) -> Result<ExtValueProbe, ServiceError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        match self.script.get(call).copied().unwrap_or(OnProbe::Pass) {
            OnProbe::Pass => {}
            OnProbe::Park => {
                self.parked.notify_one();
                self.released.notified().await;
            }
            OnProbe::DropReadHolds => {
                let _: i64 = self
                    .client
                    .pool
                    .del(&self.hold_key)
                    .await
                    .expect("DEL read_hold");
            }
            OnProbe::PoisonReadHolds => poison_read_holds(&self.client, &self.hold_key).await,
        }
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
        policy: &GcPolicy,
        state: &mut GcPassState,
    ) -> Result<GcBatchResult, ServiceError> {
        self.inner.process_pending(policy, state).await
    }

    fn ext_release_gate(&self) -> Option<&Arc<ExtReleaseGate>> {
        self.inner.ext_release_gate()
    }
}

/// Replace the `{gc}` read hold key with a string: every `ext_read_hold.lua`
/// call on it then fails with `WRONGTYPE`, as on a store whose key was
/// clobbered.
async fn poison_read_holds(client: &ValkeyClient, hold_key: &str) {
    let _: i64 = client.pool.del(hold_key).await.expect("DEL read_hold");
    let _: () = client
        .pool
        .set(hold_key, "not-a-hash", None, None, false)
        .await
        .expect("SET read_hold");
}

/// A service whose store runs `script` in `probe_values`, in snapshot mode,
/// with the store kept for the test to drive.
fn scripted_service(
    tv: &TestValkey,
    script: Vec<OnProbe>,
) -> (Arc<LiveEditService>, Arc<ScriptedProbeStore>) {
    scripted_service_in(tv, true, script)
}

/// [`scripted_service`] in catch-up snapshot mode `snapshot_mode`.
fn scripted_service_in(
    tv: &TestValkey,
    snapshot_mode: bool,
    script: Vec<OnProbe>,
) -> (Arc<LiveEditService>, Arc<ScriptedProbeStore>) {
    let mut scripted = None;
    let (svc, _adapter) = service_with(tv, snapshot_mode, |store| {
        let store = Arc::new(ScriptedProbeStore::new(store, tv, script));
        scripted = Some(Arc::clone(&store));
        store
    });
    (Arc::new(svc), scripted.expect("the wrap closure ran"))
}

/// [`scripted_service_in`] whose catch-ups run on `valkey` — a client
/// connected with a test's own `configure`d settings (atomic-read limits,
/// capacity budget) on `TEST_VALKEY_URL` — instead of the fixture's. The
/// catch-up's reads, read permits, capacity budget, stream registration and
/// read holds all go through `svc.valkey`; the layer, session and
/// external-value stores and the flights keep the fixture's client, on the
/// same database, so writes and GC are unchanged.
async fn scripted_service_on(
    tv: &TestValkey,
    snapshot_mode: bool,
    script: Vec<OnProbe>,
    configure: impl FnOnce(&mut live_edit::config::Config),
) -> (Arc<LiveEditService>, Arc<ScriptedProbeStore>) {
    let mut config = crate::common::test_config();
    configure(&mut config);
    let valkey = live_edit::storage::test_support::connect_test_client(&config)
        .await
        .expect("a catch-up client with test settings connects");
    let mut scripted = None;
    let (svc, _adapter) = service_with(tv, snapshot_mode, |store| {
        let store = Arc::new(ScriptedProbeStore::new(store, tv, script));
        scripted = Some(Arc::clone(&store));
        store
    });
    let svc = LiveEditService { valkey, ..svc };
    (Arc::new(svc), scripted.expect("the wrap closure ran"))
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 6 — the stale-miss regression, on every atomic regime
// ═════════════════════════════════════════════════════════════════════════════
//
// Each case parks the first build at the pre-ext-probe seam with its snapshot
// at cut C1 referencing `old`, lands the commit C1 + 1 that rewrites the field
// to `fresh` (releasing `old`), lets the GC collect `old`, and releases the
// build. The probe misses `old`; the catch-up re-reads under a hold at C2 =
// C1 + 1 and builds from it.
//
// Before part D the build read its blobs with fetch-on-miss on the stale
// snapshot, so the collected blob went to the adapter, which does not hold it,
// and the catch-up ended INTERNAL. In-place proof (recorded in the change's
// evidence): building the first attempt with fetch-on-miss instead of the
// probe (today that is `ExtValueReads::recover_for_test()` in place of
// `ExtValueReads::PROBE`) fails every case here on its "no INTERNAL"
// assertion, and leaking
// the hold guard — no release and no drop, so neither the explicit `HDEL` nor
// the drop's spawned one runs — fails each on its "no hold field" assertion.
// A release skipped but the guard still dropped is NOT detected here: the
// drop's `HDEL` can land before the content does (the drop path itself is
// pinned at the store, `b_yuo_gc_tests::a_hold_dropped_unreleased_deletes_its_field`).

/// Which catch-up a case drives (test 6, and the re-read endings below).
#[derive(Clone, Copy, Debug)]
enum Regime {
    /// Regime 1, `have = (0, 0)`: the full state.
    BrandNew,
    /// Regime 2, `have = (object, 1)` with `2-0` retained: in snapshot mode
    /// the coalesced diff; in rollback mode the per-commit replay.
    InWindow,
    /// Regime 3, `have = (object, 1)` with `2-0` trimmed: the coalesced diff
    /// (snapshot mode) or the legacy version diff (rollback mode).
    OutOfWindow,
}

/// Seed `layer` (created at seq 1) for `regime`: `value` written at seq 2;
/// for Regime 3 an inline commit at seq 3 and a deterministic `XTRIM MINID
/// 3-0`, which trims the reconnector's start id `2-0` away. Returns the
/// subscriber's `(have_object_id, have_version)` and the cut its first read
/// will take (C1).
async fn seed_regime(
    tv: &TestValkey,
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    object_id: u64,
    regime: Regime,
    value: &[u8],
) -> ((u64, u64), u64) {
    commit_ext(svc, session_id, layer, object_id, true, value).await; // seq 2
    let first_cut = if matches!(regime, Regime::OutOfWindow) {
        commit_inline(svc, session_id, layer, object_id, 1).await; // seq 3
        let cap = XCap::try_from(("MINID", "=", "3-0")).expect("MINID cap");
        let _: () = tv
            .client()
            .pool
            .xtrim(tv.key_schema().layer_stream(object_id), cap)
            .await
            .expect("XTRIM");
        3
    } else {
        2
    };
    let have = match regime {
        Regime::BrandNew => (0, 0),
        _ => (object_id, 1),
    };
    (have, first_cut)
}

/// Drive one cut-based test-6 case: see the section header. Asserts the
/// payload is fresh and carries C2, the ack is C2, the next message is the
/// next commit, the miss counted one `stale_retried` on the atomic read and
/// nothing genuine, the adapter was never consulted, and no hold field is
/// left once the content arrives.
#[cfg(feature = "test-fault-injection")]
async fn assert_cut_based_stale_miss_rebuilds_fresh(regime: Regime, snapshot_mode: bool) {
    let tv = TestValkey::setup().await;
    let (mut svc, _adapter) = service_with(&tv, snapshot_mode, |store| Arc::new(store));
    let seam = Arc::new(live_edit::grpc::subscribe::CatchUpStateReadSeam::new_before_ext_probe());
    svc.catch_up_state_read_seam_for_test = Some(Arc::clone(&seam));
    let svc = Arc::new(svc);
    let layer = format!("stale-{regime:?}-{snapshot_mode}");
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, "b-yuo-stale", &layer).await;
    let (old, fresh) = (blob(0xA1), blob(0xB2));
    let (have, first_cut) =
        seed_regime(&tv, &svc, &session_id, &layer, object_id, regime, &old).await;
    let new_cut = first_cut + 1;

    let driver = {
        let (svc, seam, session_id, layer, fresh, old) = (
            Arc::clone(&svc),
            Arc::clone(&seam),
            session_id.clone(),
            layer.clone(),
            fresh.clone(),
            old.clone(),
        );
        let tv_client = tv.client().clone();
        let keys = tv.key_schema().clone();
        tokio::spawn(async move {
            seam.wait_until_parked_pre_ext_probe().await;
            commit_ext(&svc, &session_id, &layer, object_id, false, &fresh).await;
            collect_with(&tv_client, &keys, &old).await;
            seam.release_pre_ext_probe().await;
        })
    };
    let mut stream = subscribe(&svc, &session_id, &layer, have.0, have.1).await;

    let (content, body) = next_message(&mut stream).await.unwrap_or_else(|status| {
        panic!(
            "[{regime:?} / snapshot_mode={snapshot_mode}] a blob collected after the snapshot \
             must be re-read, not end the catch-up: got {status:?}"
        )
    });
    tokio::time::timeout(DEADLINE, driver)
        .await
        .expect("the driver finished")
        .expect("the driver did not panic");
    assert_eq!(
        hold_fields(&tv).await,
        0,
        "[{regime:?} / snapshot_mode={snapshot_mode}] no hold field is left once the content \
         arrives"
    );
    assert_eq!(
        (content.object_id, content.sequence_num),
        (object_id, new_cut),
        "[{regime:?} / snapshot_mode={snapshot_mode}] the payload carries the re-read's cut"
    );
    let parts = carried(&body);
    assert!(
        parts.contains(&fresh) && !parts.contains(&old),
        "[{regime:?} / snapshot_mode={snapshot_mode}] the payload carries the fresh value, not \
         the collected one"
    );
    assert_eq!(
        field_hash(&body),
        Some(ext_content_hash(&fresh).to_vec()),
        "[{regime:?} / snapshot_mode={snapshot_mode}] the field references the fresh value"
    );
    expect_caught_up_ack(&mut stream, object_id, new_cut).await;
    // The rewrite is at the ack, so the next message must be the next commit:
    // a re-delivered rewrite would arrive first.
    commit_inline(&svc, &session_id, &layer, object_id, 2).await;
    let (live, _) = next_content(&mut stream, "the next commit").await;
    assert_eq!(
        live.sequence_num,
        new_cut + 1,
        "[{regime:?} / snapshot_mode={snapshot_mode}] the next commit arrives next: none lost, \
         none repeated"
    );

    assert_eq!(
        atomic_misses(&svc.metrics),
        (1, 0),
        "[{regime:?} / snapshot_mode={snapshot_mode}] one stale_retried miss, nothing genuine"
    );
    assert_eq!(
        adapter_consults(&svc.metrics),
        0,
        "[{regime:?} / snapshot_mode={snapshot_mode}] the checkpoint adapter is never consulted"
    );
}

/// Regime 1 in snapshot mode (`compute_faithful_full_content_at_cut`).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_brand_new_stale_miss_rebuilds_fresh_state_at_the_new_cut() {
    assert_cut_based_stale_miss_rebuilds_fresh(Regime::BrandNew, true).await;
}

/// Regime 1 in rollback mode (the legacy full-content `compute_diff_at_cut`).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_brand_new_legacy_mode_stale_miss_rebuilds_fresh_state_at_the_new_cut() {
    assert_cut_based_stale_miss_rebuilds_fresh(Regime::BrandNew, false).await;
}

/// Regime 2's in-window snapshot (`compute_faithful_coalesced_at_cut`).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_in_window_stale_miss_rebuilds_fresh_state_at_the_new_cut() {
    assert_cut_based_stale_miss_rebuilds_fresh(Regime::InWindow, true).await;
}

/// Regime 3 in snapshot mode (`compute_faithful_coalesced_at_cut`).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_out_of_window_stale_miss_rebuilds_fresh_state_at_the_new_cut() {
    assert_cut_based_stale_miss_rebuilds_fresh(Regime::OutOfWindow, true).await;
}

/// Regime 3 in rollback mode (the legacy version diff, `compute_diff_at_cut`).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_out_of_window_legacy_mode_stale_miss_rebuilds_fresh_state_at_the_new_cut() {
    assert_cut_based_stale_miss_rebuilds_fresh(Regime::OutOfWindow, false).await;
}

/// Regime 2 in rollback mode: the per-commit replay's first commit (seq 2,
/// which wrote `old`) is built from its own atomic read and parks; the
/// rewrite (seq 3) lands and `old` is collected. The commit's re-read is
/// taken at cut 3 and its diff stays bounded by its own sequence, so the
/// rewrite is not folded into it: the replay delivers seq 2 (without `old`),
/// then seq 3 live (with `fresh`), each once, and nothing ends INTERNAL.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rollback_replay_stale_miss_rebuilds_its_commit_from_a_fresh_read() {
    let tv = TestValkey::setup().await;
    let (mut svc, _adapter) = service_with(&tv, false, |store| Arc::new(store));
    let seam = Arc::new(live_edit::grpc::subscribe::CatchUpStateReadSeam::new_before_ext_probe());
    svc.catch_up_state_read_seam_for_test = Some(Arc::clone(&seam));
    let svc = Arc::new(svc);
    let layer = "stale-rollback-replay";
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, "b-yuo-stale", layer).await;
    let (old, fresh) = (blob(0xA3), blob(0xB4));
    commit_ext(&svc, &session_id, layer, object_id, true, &old).await; // seq 2

    let mut stream = subscribe(&svc, &session_id, layer, object_id, 1).await;
    seam.wait_until_parked_pre_ext_probe().await;
    commit_ext(&svc, &session_id, layer, object_id, false, &fresh).await; // seq 3
    collect(&tv, &old).await;
    seam.release_pre_ext_probe().await;

    let (replayed, replayed_body) = next_message(&mut stream).await.unwrap_or_else(|status| {
        panic!("a replayed commit whose blob was collected must be re-read: got {status:?}")
    });
    assert_eq!(
        hold_fields(&tv).await,
        0,
        "no hold field is left once the replay's content arrives"
    );
    assert_eq!(
        (replayed.object_id, replayed.sequence_num),
        (object_id, 2),
        "the replay delivers the commit it was rebuilding, under its own sequence"
    );
    let replayed_parts = carried(&replayed_body);
    assert!(
        !replayed_parts.contains(&old),
        "the collected value is never delivered"
    );
    assert!(
        !replayed_parts.contains(&fresh),
        "the rewrite (seq 3) is not folded into seq 2's diff: it is delivered once, as itself"
    );
    // Then the rewrite, once. An empty header between them (an ack) is
    // allowed; a second copy of seq 2 or a skip to seq 4 is not.
    let mut sequences = vec![replayed.sequence_num];
    let live_body = loop {
        let (header, body) = next_message(&mut stream)
            .await
            .unwrap_or_else(|status| panic!("the replay's tail ended {status:?}"));
        if header.total_size == 0 {
            continue;
        }
        sequences.push(header.sequence_num);
        if header.sequence_num >= 3 {
            break body;
        }
    };
    assert_eq!(sequences, [2, 3], "each commit is delivered once, in order");
    assert!(
        carried(&live_body).contains(&fresh),
        "the rewrite delivers the fresh value"
    );
    assert_quiet(&mut stream).await;

    assert_eq!(atomic_misses(&svc.metrics), (1, 0));
    assert_eq!(adapter_consults(&svc.metrics), 0);
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 7 — the genuine miss
// ═════════════════════════════════════════════════════════════════════════════

/// A layer whose field references `value`, which is then deleted out of
/// band — so the probe misses it and the re-read misses it again — with the
/// adapter holding it when `adapter_holds_it`. Returns the service, the
/// session, the layer and its object.
async fn genuinely_missing(
    tv: &TestValkey,
    value: &[u8],
    adapter_holds_it: bool,
) -> (Arc<LiveEditService>, String, String, u64) {
    let (svc, adapter) = service_with(tv, true, |store| Arc::new(store));
    let svc = Arc::new(svc);
    let layer = format!("genuine-{adapter_holds_it}");
    let (session_id, object_id) = create_session_and_layer(tv, &svc, "b-yuo-genuine", &layer).await;
    commit_ext(&svc, &session_id, &layer, object_id, true, value).await; // seq 2
    if adapter_holds_it {
        adapter
            .put_ext_blob(
                &sha256_to_hex(&ext_content_hash(value)),
                Bytes::from(value.to_vec()),
            )
            .await
            .expect("seed the adapter");
    }
    wipe(tv, value).await;
    (svc, session_id, layer, object_id)
}

/// A hash still missing after the re-read, under a hold the store confirms
/// live, is genuine: it takes fetch-on-miss, and an adapter that holds the
/// blob restores it and the catch-up delivers it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_genuine_miss_under_a_live_hold_recovers_through_the_adapter() {
    let tv = TestValkey::setup().await;
    let value = blob(0xC1);
    let (svc, session_id, layer, object_id) = genuinely_missing(&tv, &value, true).await;

    let mut stream = subscribe(&svc, &session_id, &layer, 0, 0).await;
    let (content, body) = next_content(&mut stream, "a recovered genuine miss").await;

    assert_eq!((content.object_id, content.sequence_num), (object_id, 2));
    assert!(
        carried(&body).contains(&value),
        "the adapter's copy is delivered"
    );
    expect_caught_up_ack(&mut stream, object_id, 2).await;
    assert_eq!(
        atomic_misses(&svc.metrics),
        (1, 1),
        "the first miss is retried as stale; the re-read's miss is genuine"
    );
    assert_eq!(adapter_result(&svc.metrics, "recovered"), 1);
    assert!(
        blob_stored(&tv, &value).await,
        "the recovery stored it again"
    );
    assert_eq!(hold_fields(&tv).await, 0, "the hold is released");
}

/// A genuine miss the adapter cannot restore ends the catch-up INTERNAL,
/// before content — the one ending that says the layer is damaged.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_genuine_miss_the_adapter_cannot_restore_ends_internal() {
    let tv = TestValkey::setup().await;
    let value = blob(0xC2);
    let (svc, session_id, layer, _object_id) = genuinely_missing(&tv, &value, false).await;

    let mut stream = subscribe(&svc, &session_id, &layer, 0, 0).await;
    let status = expect_ended_before_content(&mut stream, "an unrecoverable miss").await;

    assert_eq!(status.code(), Code::Internal, "{status:?}");
    assert_eq!(atomic_misses(&svc.metrics), (1, 1));
    assert_eq!(adapter_result(&svc.metrics, "adapter_miss"), 1);
    assert_eq!(hold_fields(&tv).await, 0, "the hold is released");
}

/// The in-stream message of `read_hold_refused` names this phrase.
const READ_HOLD_REFUSED_PHRASE: &str = "could not hold garbage collection off";

/// The hold's field deleted between the re-read and the genuine check — a
/// `{gc}` failover that lost the unreplicated hold write — makes the check's
/// renewal answer LAPSED, and the catch-up ends UNAVAILABLE before content,
/// retryable, never INTERNAL: the re-read's miss may be a collection the
/// lost hold failed to stop. The adapter does not hold the blob, so a check
/// that trusted the hold anyway would end INTERNAL.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hold_lost_before_the_genuine_check_ends_unavailable_never_internal() {
    let tv = TestValkey::setup().await;
    let value = blob(0xC3);
    let (svc, store) = scripted_service(&tv, vec![OnProbe::Pass, OnProbe::DropReadHolds]);
    let layer = "hold-lost";
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, "b-yuo-genuine", layer).await;
    commit_ext(&svc, &session_id, layer, object_id, true, &value).await;
    wipe(&tv, &value).await;

    let mut stream = subscribe(&svc, &session_id, layer, 0, 0).await;
    let status = expect_ended_before_content(&mut stream, "a lost hold").await;

    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert!(
        status.message().contains(READ_HOLD_REFUSED_PHRASE),
        "the read-hold refusal: {status:?}"
    );
    assert_eq!(
        store.calls.load(Ordering::SeqCst),
        2,
        "precondition: the hold was dropped at the re-read's probe"
    );
    assert_eq!(
        atomic_misses(&svc.metrics),
        (1, 0),
        "a miss under a lost hold is not counted genuine"
    );
    assert_eq!(adapter_consults(&svc.metrics), 0, "no fetch-on-miss");
}

/// The layer overwritten between the stale snapshot and the re-read: the
/// re-read answers NotFound for the old object, and the catch-up re-bootstraps
/// onto the new one before any content, with the hold released. One case per
/// regime, because each maps the re-read's NotFound itself: the cut-based
/// regimes through `CatchUpBuilt::RereadNotFound` and the recovery that
/// follows the hold's release (`recover_after_reread_not_found`; with
/// checkpointing off here it answers not recoverable, and the re-bootstrap
/// follows), the rollback replay's per-commit build through its own
/// `NotFound` arm. One stream registration is left, the re-bootstrap's, and
/// `live_edit_stream_mux_subscriptions` is back at its pre-subscribe level
/// once the subscription goes: a cut-based re-read's swap reservation goes
/// with its `RereadNotFound` answer.
async fn assert_overwrite_between_reads_rebootstraps(
    regime: Regime,
    snapshot_mode: bool,
    seed: u8,
) {
    let tv = TestValkey::setup().await;
    let value = blob(seed);
    let (svc, store) = scripted_service_in(&tv, snapshot_mode, vec![OnProbe::Park]);
    let before = count_subscriptions(&svc).await;
    let layer = format!("overwritten-{regime:?}-{snapshot_mode}");
    let layer = layer.as_str();
    let (session_id, old_object) = create_session_and_layer(&tv, &svc, "b-yuo-gone", layer).await;
    let (have, _) = seed_regime(&tv, &svc, &session_id, layer, old_object, regime, &value).await;

    let mut stream = subscribe(&svc, &session_id, layer, have.0, have.1).await;
    store.wait_parked().await;
    let new_object = create_layer(&svc, &session_id, layer).await;
    assert_ne!(new_object, old_object, "precondition: an overwrite");
    svc.layer_store.wait_for_cleanup(DEADLINE).await;
    // The old object's blob goes too, so the parked build misses it.
    wipe(&tv, &value).await;
    store.release();

    let (content, _) = next_content(&mut stream, "the re-bootstrap").await;
    assert_eq!(
        (content.object_id, content.sequence_num),
        (new_object, 1),
        "[{regime:?} / snapshot_mode={snapshot_mode}] the catch-up re-bootstraps onto the new \
         object"
    );
    expect_caught_up_ack(&mut stream, new_object, 1).await;
    assert_eq!(atomic_misses(&svc.metrics), (1, 0));
    assert_eq!(
        hold_fields(&tv).await,
        0,
        "[{regime:?} / snapshot_mode={snapshot_mode}] the hold is released"
    );
    let what = format!("[{regime:?} / snapshot_mode={snapshot_mode}] the re-bootstrap");
    assert_eq!(
        subscriptions(&svc.metrics),
        before + 1,
        "{what}: one stream registration, the re-bootstrap's"
    );
    drop(stream);
    until_subscriptions(&svc.metrics, before, &what).await;
}

/// Regime 1 (`catch_up_brand_new`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stale_miss_whose_layer_was_overwritten_rebootstraps_on_the_new_object() {
    assert_overwrite_between_reads_rebootstraps(Regime::BrandNew, true, 0xC4).await;
}

/// Regime 2's snapshot arm (`catch_up_in_window_snapshot`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_in_window_stale_miss_whose_layer_was_overwritten_rebootstraps() {
    assert_overwrite_between_reads_rebootstraps(Regime::InWindow, true, 0xCB).await;
}

/// Regime 3 (`catch_up_out_of_window`), in snapshot mode.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_out_of_window_stale_miss_whose_layer_was_overwritten_rebootstraps() {
    assert_overwrite_between_reads_rebootstraps(Regime::OutOfWindow, true, 0xCC).await;
}

/// Regime 3 in rollback mode.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_out_of_window_legacy_mode_stale_miss_whose_layer_was_overwritten_rebootstraps() {
    assert_overwrite_between_reads_rebootstraps(Regime::OutOfWindow, false, 0xCD).await;
}

/// The rollback replay's per-commit build (`replay_commit_diff`'s own
/// `NotFound` arm).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rollback_replay_stale_miss_whose_layer_was_overwritten_rebootstraps() {
    assert_overwrite_between_reads_rebootstraps(Regime::InWindow, false, 0xCE).await;
}

/// The layer deleted between the stale snapshot and the re-read: the
/// re-read's NotFound finds no mapping either, and the catch-up ends
/// NOT_FOUND before content — never INTERNAL.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stale_miss_whose_layer_was_deleted_ends_not_found() {
    let tv = TestValkey::setup().await;
    let value = blob(0xC5);
    let (svc, store) = scripted_service(&tv, vec![OnProbe::Park]);
    let layer = "deleted";
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, "b-yuo-gone", layer).await;
    commit_ext(&svc, &session_id, layer, object_id, true, &value).await;

    let mut stream = subscribe(&svc, &session_id, layer, 0, 0).await;
    store.wait_parked().await;
    let mut delete = Request::new(DeleteLayerRequest {
        session_id: session_id.clone(),
        layer_name: layer.to_owned(),
    });
    delete
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    handle_delete_layer(&svc, delete)
        .await
        .expect("DeleteLayer must succeed");
    svc.layer_store.wait_for_cleanup(DEADLINE).await;
    if blob_stored(&tv, &value).await {
        wipe(&tv, &value).await;
    }
    store.release();

    let status = expect_ended_before_content(&mut stream, "a deleted layer").await;
    assert_eq!(status.code(), Code::NotFound, "{status:?}");
    assert_eq!(atomic_misses(&svc.metrics), (1, 0));
    assert_eq!(hold_fields(&tv).await, 0, "the hold is released");
}

// ═════════════════════════════════════════════════════════════════════════════
// A cut-based site's re-read NotFound runs the layer recovery, once the hold
// is released
// ═════════════════════════════════════════════════════════════════════════════
//
// At a cut-based site (Regime 1, Regime 2's snapshot arm, Regime 3, and the
// re-attach to a flight), a stale miss whose re-read finds the layer's state
// gone — every `le:{oid}:obj:*` key lost, the session's mapping kept, as a
// keyspace loss leaves it — is the first read's `NotFound` met later: with
// checkpointing on and a restorable checkpoint, the catch-up restores the
// layer and delivers it, instead of ending `NOT_FOUND` "…could not be
// recovered". The restore runs only once the `{gc}` read hold is released
// (`recover_after_reread_not_found`, SDD §11.2).
//
// In-place proof (fail-on-old): with `recover_after_reread_not_found`'s body
// replaced by the pre-fix `handle_deleted_and_rebootstrap(…, StateGone)`,
// every cut-based case here ends `NOT_FOUND` (the Transient case too,
// instead of `UNAVAILABLE`).
//
// The rollback replay's per-commit re-read is not a cut-based site: its
// `NotFound` goes to the replay's own `NotFound` arm, which re-bootstraps
// without asking the recovery and never enters
// `recover_after_reread_not_found`. Its case here
// (`a_rollback_replay_re_read_that_finds_the_state_gone_rebootstraps_without_recovery`)
// is a guard on that behaviour, which the old code had too, so it passes
// on the old code as well.

/// The inline field a restorable checkpoint holds in [`SECTION`]: only the
/// restore can have written it.
const RESTORED_FIELD: &str = "restored:marker";

/// The checkpoint the layer recovery restores from, and what each restore
/// attempt found: an `InMemoryCheckpointStore` whose `list_layer_checkpoints`
/// — the first store call a restore attempt makes (its restore-source
/// selection lists, then reads) — records how many `{gc}` read hold fields
/// the store holds at that moment, then, built
/// [`holding_listing`](Self::holding_listing), waits for the test, and,
/// with `fail_listing`, answers `Transient`. Every other method is forwarded,
/// the defaulted ones included:
/// an omitted arm would substitute the trait's default, not pass through.
struct RestoreObservingStore {
    inner: Arc<InMemoryCheckpointStore>,
    client: ValkeyClient,
    hold_key: String,
    fail_listing: bool,
    holds_at_listing: std::sync::Mutex<Vec<u64>>,
    /// With [`Self::holding_listing`], each listing waits here, after it has
    /// recorded the hold count, until the test releases it: a slow restore.
    hold_listing: bool,
    listing_parked: tokio::sync::Notify,
    listing_released: tokio::sync::Notify,
}

impl RestoreObservingStore {
    fn new(tv: &TestValkey, fail_listing: bool) -> Self {
        Self {
            inner: Arc::new(InMemoryCheckpointStore::new()),
            client: tv.client().clone(),
            hold_key: tv.key_schema().ext_read_hold(),
            fail_listing,
            holds_at_listing: std::sync::Mutex::new(Vec::new()),
            hold_listing: false,
            listing_parked: tokio::sync::Notify::new(),
            listing_released: tokio::sync::Notify::new(),
        }
    }

    /// [`Self::new`], whose listings park until [`Self::release_listing`].
    #[cfg(feature = "test-fault-injection")]
    fn holding_listing(tv: &TestValkey) -> Self {
        Self {
            hold_listing: true,
            ..Self::new(tv, false)
        }
    }

    /// Wait until a listing has parked.
    #[cfg(feature = "test-fault-injection")]
    async fn listing_parked(&self) {
        tokio::time::timeout(DEADLINE, self.listing_parked.notified())
            .await
            .expect("no restore listed the checkpoints");
    }

    /// Let the parked listing go on.
    #[cfg(feature = "test-fault-injection")]
    fn release_listing(&self) {
        self.listing_released.notify_one();
    }

    /// The read hold's field count at each restore attempt's listing.
    fn holds_at_listing(&self) -> Vec<u64> {
        self.holds_at_listing.lock().unwrap().clone()
    }

    /// Put a checkpoint of `object_id` at `snap_seq` whose state is one
    /// section, [`SECTION`], holding the inline field [`RESTORED_FIELD`].
    /// The meta names the live mapping, as the restore's SR-2.4 identity
    /// guard requires.
    async fn seed(&self, object_id: u64, session_id: &str, layer: &str, snap_seq: u64) {
        let state = OwnedDelta {
            is_diff: true,
            base_version: 0,
            struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 0,
                section_id: SECTION,
                section_name: "Mesh".to_owned(),
                section_type: 1,
                section_order: 0,
            })],
            set_fields: vec![SetField {
                section_id: SECTION,
                key_name: RESTORED_FIELD.to_owned(),
                value_or_ext_hash: vec![0x5E, 0x57],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            ..Default::default()
        };
        let meta = CheckpointMeta {
            creator_replica_id: crate::common::RECOVERY_SERVICE_LABEL.to_owned(),
            written_at_unix_ms: 1_700_000_000_000,
            snap_seq: Some(snap_seq),
            session_id: session_id.to_owned(),
            layer_name: layer.to_owned(),
            created_by: "test-user".to_owned(),
            server_ts: 1_700_000_000_000,
            ..CheckpointMeta::default()
        };
        self.inner
            .put_layer_checkpoint(
                object_id,
                snap_seq,
                Bytes::from(delta_to_ompf_bytes(&state, &[])),
                meta,
            )
            .await
            .expect("seeding the layer checkpoint must succeed");
    }
}

#[async_trait::async_trait]
impl CheckpointStore for RestoreObservingStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_latest_layer_checkpoint(object_id).await
    }

    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        let holds: u64 = self
            .client
            .pool
            .hlen(&self.hold_key)
            .await
            .expect("HLEN read_hold");
        self.holds_at_listing.lock().unwrap().push(holds);
        if self.hold_listing {
            self.listing_parked.notify_one();
            self.listing_released.notified().await;
        }
        if self.fail_listing {
            return Err(CheckpointError::Transient {
                source: "synthetic transient listing failure (b.yuo re-read test)".into(),
            });
        }
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
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.inner.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.inner.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.inner.ext_blob_exists(hash_hex).await
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

    async fn list_referenced_ext_hashes(
        &self,
    ) -> Result<std::collections::HashSet<String>, CheckpointError> {
        self.inner.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.inner.list_live_ext_set().await
    }

    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<std::collections::HashSet<String>, CheckpointError> {
        self.inner.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.inner.list_layer_artifacts().await
    }

    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.inner
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> Duration {
        self.inner.publish_worst_case()
    }
}

/// Install on `svc` an enabled layer recovery hook restoring from
/// `checkpoints`, on the service's layer store and metrics.
fn install_recovery(svc: &mut LiveEditService, checkpoints: &Arc<RestoreObservingStore>) {
    let mut config = crate::common::test_config();
    config.checkpoint_enabled = true;
    config.service_label = crate::common::RECOVERY_SERVICE_LABEL.to_owned();
    svc.recovery = Arc::new(RecoveryHook::new(
        Arc::clone(checkpoints) as Arc<dyn CheckpointStore>,
        Arc::clone(&svc.layer_store),
        Arc::new(config),
        Arc::clone(&svc.metrics),
    ));
}

/// The inline value of [`RESTORED_FIELD`] in `body`'s delta, and the delta's
/// `(is_diff, base_version)`.
fn restored_field(body: &[u8]) -> (Option<Vec<u8>>, (bool, u64)) {
    let container = OmpfContainer::parse(body).expect("a payload parses as OMPF");
    let delta = oodf_ffi::parse_delta(container.delta_bytes).expect("a payload's delta parses");
    let field = delta
        .set_fields
        .iter()
        .find(|field| field.key_name == RESTORED_FIELD)
        .map(|field| field.value_or_ext_hash.clone());
    (field, (delta.is_diff, delta.base_version))
}

/// The state a keyspace loss leaves while a catch-up is parked between its
/// first read and its blob read: every `le:{oid}:obj:*` key gone, the
/// mapping kept, and the blob the parked snapshot references gone too, so
/// the build misses it and the re-read finds no state.
async fn lose_the_layers_state(tv: &TestValkey, object_id: u64, value: &[u8]) {
    crate::common::wipe_object_keys(tv, object_id).await;
    if blob_stored(tv, value).await {
        wipe(tv, value).await;
    }
}

/// A service in snapshot mode whose store runs `script` in `probe_values`,
/// with an enabled layer recovery hook over a [`RestoreObservingStore`]
/// (listing fails `Transient` when `fail_listing`).
fn restoring_service(
    tv: &TestValkey,
    script: Vec<OnProbe>,
    fail_listing: bool,
) -> (
    Arc<LiveEditService>,
    Arc<ScriptedProbeStore>,
    Arc<RestoreObservingStore>,
) {
    restoring_service_in(tv, true, script, fail_listing)
}

/// [`restoring_service`] in catch-up snapshot mode `snapshot_mode`.
fn restoring_service_in(
    tv: &TestValkey,
    snapshot_mode: bool,
    script: Vec<OnProbe>,
    fail_listing: bool,
) -> (
    Arc<LiveEditService>,
    Arc<ScriptedProbeStore>,
    Arc<RestoreObservingStore>,
) {
    let mut scripted = None;
    let (mut svc, _adapter) = service_with(tv, snapshot_mode, |store| {
        let store = Arc::new(ScriptedProbeStore::new(store, tv, script));
        scripted = Some(Arc::clone(&store));
        store
    });
    let checkpoints = Arc::new(RestoreObservingStore::new(tv, fail_listing));
    install_recovery(&mut svc, &checkpoints);
    (
        Arc::new(svc),
        scripted.expect("the wrap closure ran"),
        checkpoints,
    )
}

/// One cut-based regime's site: the first build parks at its probe with its
/// snapshot referencing `value`; the layer's state is lost; the build misses
/// `value`, takes its hold and re-reads, which answers `NotFound`. Then, the
/// hold released (zero fields when the restore lists the checkpoints), the
/// layer is restored from its checkpoint (one `restored` attempt) and caught
/// up again from its start: the brand-new site sends the restored state from
/// empty (`base_version = 0`), a reconnector's site the coalesced diff at its
/// `have_version` — never a reset an advanced reconnector would refuse — each
/// carrying the restored field, at the restored header's sequence, then the
/// caught-up ack, then a commit on the restored layer live. One stale miss,
/// nothing genuine, no fetch-on-miss; the hold released, and one stream
/// registration left, released with the subscription.
async fn assert_reread_not_found_restores(regime: Regime, seed: u8) {
    let tv = TestValkey::setup().await;
    let value = blob(seed);
    let (svc, store, checkpoints) = restoring_service(&tv, vec![OnProbe::Park], false);
    let before = count_subscriptions(&svc).await;
    let layer = format!("restored-{regime:?}");
    let layer = layer.as_str();
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, "b-yuo-restore", layer).await;
    let (have, first_cut) =
        seed_regime(&tv, &svc, &session_id, layer, object_id, regime, &value).await;
    checkpoints
        .seed(object_id, &session_id, layer, first_cut)
        .await;

    let mut stream = subscribe(&svc, &session_id, layer, have.0, have.1).await;
    store.wait_parked().await;
    lose_the_layers_state(&tv, object_id, &value).await;
    store.release();

    let what = format!("[{regime:?}] a re-read that met the lost state");
    let (content, body) = next_content(&mut stream, &what).await;
    let restored_seq = crate::common::read_header_u64(&tv, object_id, "seq")
        .await
        .expect("the restore wrote the header");
    assert!(
        restored_seq >= first_cut,
        "{what}: precondition: the restored sequence {restored_seq} is not below the first \
         read's cut {first_cut}"
    );
    assert_eq!(
        (content.object_id, content.sequence_num),
        (object_id, restored_seq),
        "{what}: the restored layer is delivered, at its restored sequence"
    );
    let (field, shape) = restored_field(&body);
    assert_eq!(
        field,
        Some(vec![0x5E, 0x57]),
        "{what}: the payload carries the state only the restore wrote"
    );
    let expected_shape = match regime {
        Regime::BrandNew => (true, 0),
        Regime::InWindow | Regime::OutOfWindow => (true, have.1),
    };
    assert_eq!(
        shape, expected_shape,
        "{what}: `(is_diff, base_version)` — the brand-new site sends from empty, a \
         reconnector's site the coalesced diff at its have_version"
    );
    expect_caught_up_ack(&mut stream, object_id, restored_seq).await;
    commit_inline(&svc, &session_id, layer, object_id, 9).await;
    let (live, _) = next_content(&mut stream, "the commit after the restore").await;
    assert_eq!(
        live.sequence_num,
        restored_seq + 1,
        "{what}: the restored layer's next commit arrives live"
    );

    assert_eq!(
        checkpoints.holds_at_listing(),
        vec![0],
        "{what}: one restore attempt, run with no read hold field on the store"
    );
    assert_eq!(
        crate::common::recovery_metric(&svc.metrics, "restored"),
        1,
        "{what}: the layer recovery restored the layer once"
    );
    assert_eq!(atomic_misses(&svc.metrics), (1, 0), "{what}");
    assert_eq!(
        adapter_consults(&svc.metrics),
        0,
        "{what}: no fetch-on-miss"
    );
    assert_eq!(hold_fields(&tv).await, 0, "{what}: the hold is released");
    assert_eq!(
        subscriptions(&svc.metrics),
        before + 1,
        "{what}: one stream registration, the re-run's"
    );
    drop(stream);
    until_subscriptions(&svc.metrics, before, &what).await;
}

/// Regime 1 (`catch_up_brand_new`): re-run as a brand-new catch-up.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_brand_new_re_read_that_finds_the_state_lost_restores_it_and_delivers_it() {
    assert_reread_not_found_restores(Regime::BrandNew, 0xB5).await;
}

/// Regime 2's snapshot arm (`catch_up_in_window_snapshot`): re-run through
/// Regime 3, whose coalesced diff is the one this arm builds.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_in_window_re_read_that_finds_the_state_lost_restores_it_and_delivers_it() {
    assert_reread_not_found_restores(Regime::InWindow, 0xB6).await;
}

/// Regime 3 (`catch_up_out_of_window`): re-run through Regime 3.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_out_of_window_re_read_that_finds_the_state_lost_restores_it_and_delivers_it() {
    assert_reread_not_found_restores(Regime::OutOfWindow, 0xB7).await;
}

/// The recovery's own failure (`Transient` from the checkpoint store, after
/// the hold is released) ends the catch-up `UNAVAILABLE` before content —
/// retryable, as the first read's recovery failure does — never `NOT_FOUND`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_re_read_whose_recovery_fails_transiently_ends_unavailable() {
    let tv = TestValkey::setup().await;
    let value = blob(0xB8);
    let (svc, store, checkpoints) = restoring_service(&tv, vec![OnProbe::Park], true);
    let layer = "restore-transient";
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, "b-yuo-restore", layer).await;
    commit_ext(&svc, &session_id, layer, object_id, true, &value).await; // seq 2
    checkpoints.seed(object_id, &session_id, layer, 2).await;

    let mut stream = subscribe(&svc, &session_id, layer, 0, 0).await;
    store.wait_parked().await;
    lose_the_layers_state(&tv, object_id, &value).await;
    store.release();

    let status = expect_ended_before_content(&mut stream, "a failed recovery").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_eq!(
        checkpoints.holds_at_listing(),
        vec![0],
        "the recovery ran once, with no read hold field on the store"
    );
    assert_eq!(crate::common::recovery_metric(&svc.metrics, "error"), 1);
    assert_eq!(crate::common::recovery_metric(&svc.metrics, "restored"), 0);
    assert_eq!(atomic_misses(&svc.metrics), (1, 0));
    assert_eq!(hold_fields(&tv).await, 0, "the hold is released");
}

/// The rollback replay's per-commit re-read is not a cut-based site: its
/// `NotFound` goes to the replay's own `NotFound` arm, which re-bootstraps at
/// once and never asks the layer recovery — even with checkpointing on and a
/// checkpoint of the old object stored. Here the layer is overwritten at the
/// park, so the re-bootstrap lands on the new object; no restore attempt
/// lists the checkpoints and the recovery counts nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rollback_replay_re_read_that_finds_the_state_gone_rebootstraps_without_recovery() {
    let tv = TestValkey::setup().await;
    let value = blob(0xB9);
    let (svc, store, checkpoints) = restoring_service_in(&tv, false, vec![OnProbe::Park], false);
    let layer = "replay-no-recovery";
    let (session_id, old_object) =
        create_session_and_layer(&tv, &svc, "b-yuo-restore", layer).await;
    let (have, first_cut) = seed_regime(
        &tv,
        &svc,
        &session_id,
        layer,
        old_object,
        Regime::InWindow,
        &value,
    )
    .await;
    checkpoints
        .seed(old_object, &session_id, layer, first_cut)
        .await;

    let mut stream = subscribe(&svc, &session_id, layer, have.0, have.1).await;
    store.wait_parked().await;
    let new_object = create_layer(&svc, &session_id, layer).await;
    assert_ne!(new_object, old_object, "precondition: an overwrite");
    svc.layer_store.wait_for_cleanup(DEADLINE).await;
    if blob_stored(&tv, &value).await {
        wipe(&tv, &value).await;
    }
    store.release();

    let (content, _) = next_content(&mut stream, "the re-bootstrap").await;
    assert_eq!(
        (content.object_id, content.sequence_num),
        (new_object, 1),
        "the replay re-bootstraps onto the new object"
    );
    expect_caught_up_ack(&mut stream, new_object, 1).await;
    assert_eq!(
        checkpoints.holds_at_listing(),
        Vec::<u64>::new(),
        "no restore attempt listed the checkpoints"
    );
    for result in ["restored", "not_recoverable", "error"] {
        assert_eq!(
            crate::common::recovery_metric(&svc.metrics, result),
            0,
            "the layer recovery was not asked: `{result}`"
        );
    }
    assert_eq!(atomic_misses(&svc.metrics), (1, 0));
    assert_eq!(hold_fields(&tv).await, 0, "the hold is released");
}

/// A restored re-run is charged to the capacity budget the first read left —
/// a restore is not an overwrite, so the budget is not refilled — and a
/// re-run that cannot get capacity within it ends with the retryable
/// capacity refusal, not content.
///
/// The catch-up client has one catch-up slot and a 1 s budget. The
/// re-read finds the state lost; while the restore is held at its listing
/// (no slot or hold is held then), the test takes the only slot, then lets
/// the restore finish. The re-run waits for the slot until what is left of
/// the budget runs out and ends `catchup_capacity` (`wait=slot`). Its charges
/// (`CapacityBudgetSeam`) chain from the first read's — each wait starts with
/// what the one before left, never the full budget again — and its slot wait
/// leaves nothing.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restored_re_run_out_of_its_capacity_budget_ends_with_the_capacity_refusal() {
    use live_edit::error::{refusal_metadata, refusal_reason, CAPACITY_RETRY_DELAY};
    use live_edit::grpc::subscribe::{BudgetWait, CapacityBudgetSeam};

    const BUDGET_MS: u64 = 1_000;
    let tv = TestValkey::setup().await;
    let mut config = crate::common::test_config();
    config.subscriber_max_concurrent_catchups = 1;
    config.subscriber_admission_wait_ms = BUDGET_MS;
    let valkey = live_edit::storage::test_support::connect_test_client(&config)
        .await
        .expect("a catch-up client with one slot connects");
    let mut scripted = None;
    let (svc, _adapter) = service_with(&tv, true, |store| {
        let store = Arc::new(ScriptedProbeStore::new(store, &tv, vec![OnProbe::Park]));
        scripted = Some(Arc::clone(&store));
        store
    });
    let store = scripted.expect("the wrap closure ran");
    let seam = Arc::new(CapacityBudgetSeam::new());
    let mut svc = LiveEditService {
        valkey,
        capacity_budget_seam_for_test: Some(Arc::clone(&seam)),
        ..svc
    };
    let checkpoints = Arc::new(RestoreObservingStore::holding_listing(&tv));
    install_recovery(&mut svc, &checkpoints);
    let svc = Arc::new(svc);
    let value = blob(0xBA);
    let layer = "restored-out-of-budget";
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, "b-yuo-restore", layer).await;
    commit_ext(&svc, &session_id, layer, object_id, true, &value).await; // seq 2
    checkpoints.seed(object_id, &session_id, layer, 2).await;

    let mut stream = subscribe(&svc, &session_id, layer, 0, 0).await;
    store.wait_parked().await;
    lose_the_layers_state(&tv, object_id, &value).await;
    store.release();
    checkpoints.listing_parked().await;
    let slot = svc
        .valkey
        .catchup_gate
        .acquire(tokio::time::Instant::now() + Duration::from_secs(3_600))
        .await
        .expect("the only catch-up slot is free while the restore runs");
    checkpoints.release_listing();

    let status = expect_ended_before_content(&mut stream, "a re-run out of its budget").await;
    crate::common::refusal_detail::assert_refusal_detail(
        &status,
        refusal_reason::CATCHUP_CAPACITY,
        &[(refusal_metadata::WAIT, refusal_metadata::SLOT)],
        CAPACITY_RETRY_DELAY,
    );
    assert_eq!(
        crate::common::recovery_metric(&svc.metrics, "restored"),
        1,
        "the restore finished; only the re-run's capacity ran out"
    );
    assert_eq!(checkpoints.holds_at_listing(), vec![0]);
    assert_eq!(hold_fields(&tv).await, 0, "the hold is released");

    let charges = seam.charges_for("b-yuo-subscriber");
    assert!(
        charges.len() >= 3,
        "admission, the first read's waits and the re-run's: {charges:#?}"
    );
    let total = Duration::from_millis(BUDGET_MS);
    assert_eq!(charges[0].wait, BudgetWait::Admission, "{charges:#?}");
    for pair in charges.windows(2) {
        assert_eq!(
            pair[1].budget_before, pair[0].budget_after,
            "each wait starts with what the one before left — the restored re-run's \
             included, so no refill: {charges:#?}"
        );
    }
    let last = charges.last().expect("charges");
    assert_eq!(
        (last.wait, last.budget_after),
        (BudgetWait::CatchUpSlot, Duration::ZERO),
        "the re-run's slot wait ran out the budget: {charges:#?}"
    );
    assert!(
        last.budget_before < total,
        "the re-run's slot wait started with less than the full budget: {charges:#?}"
    );
    drop(slot);
}

/// Assert the stream ended `read_hold_refused`: UNAVAILABLE before content,
/// retryable, never INTERNAL, with no refusal detail (no `ErrorInfo`, no
/// retry hint: a fault ending, not load-shedding); one stale miss on the
/// `read` it was found in and nothing genuine; no fetch-on-miss.
async fn assert_read_hold_refused(
    svc: &LiveEditService,
    stream: &mut SubscribeStream,
    what: &str,
    read: &str,
) {
    let status = expect_ended_before_content(stream, what).await;
    assert_eq!(status.code(), Code::Unavailable, "{what}: {status:?}");
    assert!(
        status.message().contains(READ_HOLD_REFUSED_PHRASE),
        "{what}: the read-hold refusal, got {status:?}"
    );
    crate::common::refusal_detail::assert_no_refusal_detail(&status);
    let other = if read == "atomic" {
        "shared_fenced"
    } else {
        "atomic"
    };
    assert_eq!(
        (
            misses(&svc.metrics, "stale_retried", read),
            misses(&svc.metrics, "genuine", read),
            misses(&svc.metrics, "stale_retried", other),
            misses(&svc.metrics, "genuine", other),
        ),
        (1, 0, 0, 0),
        "{what}: one stale miss on the {read} read, nothing genuine"
    );
    assert_eq!(
        adapter_consults(&svc.metrics),
        0,
        "{what}: no fetch-on-miss"
    );
}

/// A hold that cannot be taken — the `take` fails, here on a hold key of the
/// wrong type — ends the catch-up `read_hold_refused` before any re-read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hold_that_cannot_be_taken_ends_unavailable_never_internal() {
    let tv = TestValkey::setup().await;
    let value = blob(0xC6);
    let (svc, session_id, layer, _object_id) = genuinely_missing(&tv, &value, false).await;
    poison_read_holds(tv.client(), &tv.key_schema().ext_read_hold()).await;

    let mut stream = subscribe(&svc, &session_id, &layer, 0, 0).await;
    assert_read_hold_refused(&svc, &mut stream, "a failed take", "atomic").await;
}

/// The genuine check's renewal failing (here on a hold key turned to the
/// wrong type between the re-read and the check) is not a confirmed live
/// hold: the catch-up ends `read_hold_refused`, never INTERNAL, and counts
/// nothing genuine. The adapter does not hold the blob, so a check that
/// treated the failure as live would end INTERNAL.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_genuine_check_whose_renewal_fails_ends_unavailable_never_internal() {
    let tv = TestValkey::setup().await;
    let value = blob(0xC7);
    let (svc, store) = scripted_service(&tv, vec![OnProbe::Pass, OnProbe::PoisonReadHolds]);
    let layer = "renewal-fails";
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, "b-yuo-genuine", layer).await;
    commit_ext(&svc, &session_id, layer, object_id, true, &value).await;
    wipe(&tv, &value).await;

    let mut stream = subscribe(&svc, &session_id, layer, 0, 0).await;
    assert_read_hold_refused(
        &svc,
        &mut stream,
        "a failed genuine-check renewal",
        "atomic",
    )
    .await;
    assert_eq!(
        store.calls.load(Ordering::SeqCst),
        2,
        "precondition: the key was poisoned at the re-read's probe"
    );
}

/// The re-read's cut below the first read's — the layer's sequence moved
/// backwards between the two, as a restore with an undersized sequence band
/// leaves it — ends the catch-up UNAVAILABLE before content (the
/// sequence-regression ending, not the miss protocol's), with the hold
/// released, and leaves `live_edit_stream_mux_subscriptions` back at its
/// pre-subscribe level: the swap's reservation goes with the ending. The
/// ending's one WARN is the shared regression line (`catch_up_regressed`),
/// naming the re-read it was found on, `read="atomic"`, its `regime`, the pin
/// (the first read's cut) and the cut, beside one `unverified` stable cut
/// under the regime.
///
/// On a current-thread runtime, because the catch-up runs in a task the
/// handler spawns, which only a thread-local log capture sees there
/// (`log_capture::ThreadCapture`); `TestValkey`'s cleanup skips on one, so
/// the test purges its keys itself.
#[tokio::test]
async fn a_re_read_whose_cut_regressed_ends_unavailable_before_content() {
    let logs = super::log_capture::ThreadCapture::start();
    let tv = TestValkey::setup().await;
    let value = blob(0xC8);
    let (svc, store) = scripted_service(&tv, vec![OnProbe::Park]);
    let before = count_subscriptions(&svc).await;
    let layer = "regressed";
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, "b-yuo-regress", layer).await;
    commit_ext(&svc, &session_id, layer, object_id, true, &value).await; // seq 2

    let mut stream = subscribe(&svc, &session_id, layer, 0, 0).await;
    store.wait_parked().await;
    // The first read's cut is 2; the re-read will see 1.
    let _: () = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_header(&object_id.to_string()),
            ("seq", "1"),
        )
        .await
        .expect("rewind the header sequence");
    wipe(&tv, &value).await;
    store.release();

    let status = expect_ended_before_content(&mut stream, "a regressed re-read").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert!(
        status.message().contains("sequence moved backwards"),
        "the sequence-regression ending: {status:?}"
    );
    assert_eq!(atomic_misses(&svc.metrics), (1, 0));
    assert_eq!(adapter_consults(&svc.metrics), 0);
    assert_eq!(hold_fields(&tv).await, 0, "the hold is released");
    assert_eq!(
        stable_cuts(&svc.metrics, "unverified", "brand_new"),
        1,
        "the regression ending records its unverified cut under the regime"
    );
    let captured = logs.snapshot();
    let regressed: Vec<&serde_json::Value> = captured
        .events
        .iter()
        .filter(|event| {
            event["fields"]["message"]
                .as_str()
                .is_some_and(|m| m.starts_with("catch-up snapshot read's cut is below its pin"))
        })
        .collect();
    assert_eq!(
        regressed.len(),
        1,
        "one regression WARN; full capture:\n{}",
        captured.text
    );
    let warn = regressed[0];
    assert_eq!(warn["level"], "WARN", "{warn}");
    for (field, want) in [
        ("read", serde_json::json!("atomic")),
        ("regime", serde_json::json!("brand_new")),
        ("pin", serde_json::json!(2)),
        ("cut", serde_json::json!(1)),
        ("object_id", serde_json::json!(object_id)),
    ] {
        assert_eq!(
            warn["fields"][field], want,
            "the regression WARN's `{field}`: {warn}"
        );
    }
    drop(stream);
    until_subscriptions(&svc.metrics, before, "a regressed re-read").await;
    super::log_capture::purge_keys(&tv).await;
}

/// A rollback replay's per-commit re-read that finds the object's shared
/// fenced-read flight registered hands the commit back as above the
/// atomic-read limits, and the replay re-routes through the flight. The hold
/// is released before that: while the replay builds from the flight's
/// snapshot no field is left — the flight's read and build take no hold.
///
/// The flight is registered by the test (`SnapshotFlights::attach_or_create`)
/// after the first read, so only the re-read sees it. The replay's waiter
/// drives the flight's read; the test's own waiter, which never reads, holds
/// the flight registered until then. The replay is parked at its flight build's
/// probe (`probe_values` call 1), inside its build turn and still attached,
/// so the test sees it in the flight however fast the read ran — a poll for
/// the join could miss a replay that joined, built and left between polls.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_per_commit_re_read_handed_to_a_flight_releases_its_hold_first() {
    let tv = TestValkey::setup().await;
    let (old, fresh) = (blob(0xC9), blob(0xCA));
    let (svc, store) = scripted_service_in(&tv, false, vec![OnProbe::Park, OnProbe::Park]);
    let layer = "handed-to-a-flight";
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, "b-yuo-flight", layer).await;
    commit_ext(&svc, &session_id, layer, object_id, true, &old).await; // seq 2

    let mut stream = subscribe(&svc, &session_id, layer, object_id, 1).await;
    store.wait_parked().await;
    commit_ext(&svc, &session_id, layer, object_id, false, &fresh).await; // seq 3
    collect(&tv, &old).await;
    let ours = svc.snapshot_flights.attach_or_create(
        object_id,
        LayerSize {
            items: 1_000,
            bytes: 1_000_000,
        },
    );
    store.release();

    // Parked again: the replay has joined the flight, read through it, and
    // is building from its snapshot.
    store.wait_parked().await;
    assert!(
        ours.flight().waiters() >= 2,
        "precondition: the replay is attached to the test's flight"
    );
    assert_eq!(
        hold_fields(&tv).await,
        0,
        "the hold is released before the replay re-routes through the flight"
    );
    drop(ours);
    store.release();

    let (content, body) = next_content(&mut stream, "the flight's catch-up").await;
    assert_eq!((content.object_id, content.sequence_num), (object_id, 3));
    assert!(carried(&body).contains(&fresh));
    expect_caught_up_ack(&mut stream, object_id, 3).await;
    assert_eq!(atomic_misses(&svc.metrics), (1, 0));
    assert_eq!(adapter_consults(&svc.metrics), 0);
}

/// `live_subscribe_stable_cut_total{outcome, regime}`, 0 while absent.
fn stable_cuts(metrics: &Metrics, outcome: &str, regime: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_subscribe_stable_cut_total",
        &[("outcome", outcome), ("regime", regime)],
    )
    .unwrap_or(0.0) as u64
}

/// A rollback replay's per-commit re-read whose cut is below the commit it
/// rebuilds — the header's sequence rewound at the park — ends the catch-up
/// through the per-commit regression ending (`per_commit_regressed`:
/// UNAVAILABLE before content, one `unverified` stable cut under
/// `regime="in_window"`), never the cut-based regression check, which a
/// per-commit build skips; the hold is released.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_per_commit_re_read_whose_cut_regressed_ends_unavailable_before_content() {
    let tv = TestValkey::setup().await;
    let value = blob(0xD6);
    let (svc, store) = scripted_service_in(&tv, false, vec![OnProbe::Park]);
    let layer = "per-commit-regressed";
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, "b-yuo-regress", layer).await;
    commit_ext(&svc, &session_id, layer, object_id, true, &value).await; // seq 2

    let mut stream = subscribe(&svc, &session_id, layer, object_id, 1).await;
    store.wait_parked().await;
    // Commit 2's first read had cut 2; its re-read will see 1.
    let _: () = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_header(&object_id.to_string()),
            ("seq", "1"),
        )
        .await
        .expect("rewind the header sequence");
    wipe(&tv, &value).await;
    store.release();

    let status = expect_ended_before_content(&mut stream, "a regressed per-commit re-read").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert!(
        status.message().contains("sequence moved backwards"),
        "the sequence-regression ending: {status:?}"
    );
    assert_eq!(
        stable_cuts(&svc.metrics, "unverified", "in_window"),
        1,
        "the per-commit regression ending records its unverified cut"
    );
    assert_eq!(atomic_misses(&svc.metrics), (1, 0));
    assert_eq!(adapter_consults(&svc.metrics), 0);
    assert_eq!(hold_fields(&tv).await, 0, "the hold is released");
}

/// Atomic-read item limit of the catch-up client in
/// [`a_per_commit_re_read_too_large_hands_back_to_a_flight_with_its_hold_released`]
/// and in 6f-c's `TOO_LARGE` case
/// (`re_read_through_a_flight::a_cut_based_re_read_answering_too_large_hands_over_to_a_flight_under_its_hold`):
/// above the layer as its first read sees it (one section, one field, and the
/// header's own items), below it once [`GROWTH_SECTIONS`] more sections land.
const SMALL_ATOMIC_ITEMS: u64 = 100;

/// Sections the growth commit adds at the park, in both cases
/// [`SMALL_ATOMIC_ITEMS`] names.
const GROWTH_SECTIONS: u64 = 300;

/// A rollback replay's per-commit re-read that finds the layer above the
/// atomic-read limits (`TOO_LARGE`; the layer grew past them at the park)
/// hands the commit back as above the ceiling, and the replay re-routes
/// through the object's shared fenced-read flight. The hold is released
/// before that: while the replay builds from the flight's snapshot
/// (`probe_values` call 1, parked), a flight is registered and no field is
/// left. The catch-up client carries a small item limit
/// ([`SMALL_ATOMIC_ITEMS`]); the flight reads fenced, whatever the limit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_per_commit_re_read_too_large_hands_back_to_a_flight_with_its_hold_released() {
    let tv = TestValkey::setup().await;
    let (old, fresh) = (blob(0xD7), blob(0xD8));
    let (svc, store) =
        scripted_service_on(&tv, false, vec![OnProbe::Park, OnProbe::Park], |config| {
            config.snapshot_atomic_read_max_items = SMALL_ATOMIC_ITEMS
        })
        .await;
    let layer = "too-large-at-the-re-read";
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, "b-yuo-flight", layer).await;
    commit_ext(&svc, &session_id, layer, object_id, true, &old).await; // seq 2

    let mut stream = subscribe(&svc, &session_id, layer, object_id, 1).await;
    store.wait_parked().await;
    assert!(
        svc.snapshot_flights.lookup(object_id).is_none(),
        "precondition: the first read was the replay's own atomic read, not a flight's"
    );
    commit_ext(&svc, &session_id, layer, object_id, false, &fresh).await; // seq 3
    let growth = OwnedDelta {
        struct_commands: (0..GROWTH_SECTIONS)
            .map(|i| {
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 0,
                    section_id: 1_000 + i,
                    section_name: format!("Growth{i}"),
                    section_type: 1,
                })
            })
            .collect(),
        ..Default::default()
    };
    upload(
        &svc,
        &session_id,
        layer,
        object_id,
        delta_to_ompf_bytes(&growth, &[]),
    )
    .await; // seq 4
    collect(&tv, &old).await;
    // A write path may have run a flight of its own for the grown layer; the
    // re-read must find none, so that it takes its atomic read (and answers
    // TOO_LARGE) instead of the registered-flight hand-back.
    let deadline = tokio::time::Instant::now() + DEADLINE;
    while svc.snapshot_flights.lookup(object_id).is_some() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "precondition: no flight is registered before the re-read"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    store.release();

    // Parked again: the re-read answered TOO_LARGE, the replay handed over,
    // and it is building from the flight's snapshot.
    store.wait_parked().await;
    assert!(
        svc.snapshot_flights.lookup(object_id).is_some(),
        "the replay re-routed through a flight"
    );
    assert_eq!(
        hold_fields(&tv).await,
        0,
        "the hold is released before the replay re-routes through the flight"
    );
    store.release();

    let (content, body) = next_content(&mut stream, "the flight's catch-up").await;
    assert_eq!((content.object_id, content.sequence_num), (object_id, 4));
    assert!(carried(&body).contains(&fresh));
    expect_caught_up_ack(&mut stream, object_id, 4).await;
    assert_eq!(atomic_misses(&svc.metrics), (1, 0));
    assert_eq!(adapter_consults(&svc.metrics), 0);
}

/// Stage an atomic stale miss on `svc` (brand-new subscribe, parked at
/// `probe_values` call 0), take the replica's only atomic read permit, and
/// release the park: the re-read under the hold queues for that permit.
/// Returns the stream and the held permit.
async fn re_read_waiting_for_the_permit(
    tv: &TestValkey,
    svc: &LiveEditService,
    store: &ScriptedProbeStore,
    layer: &str,
    seeds: (u8, u8),
) -> (
    SubscribeStream,
    live_edit::storage::subscriber_admission::CatchUpReadPermit,
) {
    let (old, fresh) = (blob(seeds.0), blob(seeds.1));
    let (session_id, object_id) = create_session_and_layer(tv, svc, "b-yuo-permit", layer).await;
    commit_ext(svc, &session_id, layer, object_id, true, &old).await;
    let stream = subscribe(svc, &session_id, layer, 0, 0).await;
    store.wait_parked().await;
    commit_ext(svc, &session_id, layer, object_id, false, &fresh).await;
    collect(tv, &old).await;
    let permit = svc
        .valkey
        .catchup_read_gate
        .acquire(
            tokio::time::Instant::now() + Duration::from_secs(3_600),
            live_edit::storage::diff_read::SnapshotReadConsumer::Catchup,
        )
        .await
        .expect("the test takes the only atomic read permit");
    store.release();
    (stream, permit)
}

/// S11: the re-read's own read-permit wait running out of the subscribe's
/// capacity budget (300 ms here, the replica's only atomic read permit held
/// by the test) ends the catch-up with that capacity ending as-is —
/// UNAVAILABLE "snapshot-read limit", not `read_hold_refused` and not
/// INTERNAL — with the hold released and `live_edit_stream_mux_subscriptions`
/// back at its pre-subscribe level: the swap's reservation goes with the
/// ending.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_re_read_out_of_capacity_budget_ends_with_its_capacity_ending_and_releases_the_hold() {
    let tv = TestValkey::setup().await;
    let (svc, store) = scripted_service_on(&tv, true, vec![OnProbe::Park], |config| {
        config.subscriber_admission_wait_ms = 300;
    })
    .await;
    let before = count_subscriptions(&svc).await;
    let (mut stream, permit) =
        re_read_waiting_for_the_permit(&tv, &svc, &store, "out-of-budget", (0xD9, 0xDA)).await;

    let status = expect_ended_before_content(&mut stream, "an exhausted capacity budget").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert!(
        status.message().contains("snapshot-read limit")
            && !status.message().contains(READ_HOLD_REFUSED_PHRASE),
        "the permit wait's own capacity ending, passed through: {status:?}"
    );
    assert_eq!(atomic_misses(&svc.metrics), (1, 0));
    assert_eq!(hold_fields(&tv).await, 0, "the hold is released");
    drop(permit);
    drop(stream);
    until_subscriptions(&svc.metrics, before, "an exhausted capacity budget").await;
}

/// S11: shutdown while the re-read waits for its read permit ends the
/// catch-up "server shutting down" (UNAVAILABLE), as-is, with the hold
/// released and `live_edit_stream_mux_subscriptions` back at its
/// pre-subscribe level (the swap's reservation goes with the ending). The
/// budget is long (30 s) so only the shutdown can end the wait; the test
/// cancels once the hold field is live, i.e. once the catch-up has taken its
/// hold and is in the re-read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_re_read_waiting_for_its_permit_at_shutdown_ends_as_is_and_releases_the_hold() {
    let tv = TestValkey::setup().await;
    let (svc, store) = scripted_service_on(&tv, true, vec![OnProbe::Park], |config| {
        config.subscriber_admission_wait_ms = 30_000;
    })
    .await;
    let before = count_subscriptions(&svc).await;
    let (mut stream, permit) =
        re_read_waiting_for_the_permit(&tv, &svc, &store, "shutdown-in-re-read", (0xDB, 0xDC))
            .await;
    let deadline = tokio::time::Instant::now() + DEADLINE;
    while hold_fields(&tv).await == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the catch-up never took its hold"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    svc.shutdown.cancel();

    let status = expect_ended_before_content(&mut stream, "a shutdown in the re-read").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_eq!(status.message(), "server shutting down", "{status:?}");
    assert_eq!(atomic_misses(&svc.metrics), (1, 0));
    // Released by the catch-up's own return, or — should shutdown drop the
    // catch-up's future first — by the guard's drop, whose `HDEL` is spawned:
    // either way within moments, far inside the 15 s TTL.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while hold_fields(&tv).await != 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the hold was not released"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    drop(permit);
    drop(stream);
    until_subscriptions(&svc.metrics, before, "a shutdown in the re-read").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// H-b / H-c — the hold across a slow rebuild
// ═════════════════════════════════════════════════════════════════════════════

/// Stage a stale miss on a scripted service whose store parks at both
/// probes: commit `old` (seq 2), subscribe brand new, and at the first park
/// rewrite to `fresh` (seq 3) and collect `old`; then release it and wait at
/// the second park — the re-read at cut 3 done, under the hold, before the
/// rebuild reads `fresh`.
async fn park_under_the_hold(
    tv: &TestValkey,
    layer: &str,
    old: &[u8],
    fresh: &[u8],
) -> (
    Arc<LiveEditService>,
    Arc<ScriptedProbeStore>,
    String,
    u64,
    SubscribeStream,
) {
    let (svc, store) = scripted_service(tv, vec![OnProbe::Park, OnProbe::Park]);
    let (session_id, object_id) = create_session_and_layer(tv, &svc, "b-yuo-renew", layer).await;
    commit_ext(&svc, &session_id, layer, object_id, true, old).await;
    let stream = subscribe(&svc, &session_id, layer, 0, 0).await;
    store.wait_parked().await;
    commit_ext(&svc, &session_id, layer, object_id, false, fresh).await;
    collect(tv, old).await;
    store.release();
    store.wait_parked().await;
    (svc, store, session_id, object_id, stream)
}

/// How often a renewing hold here is renewed: often, so a test sees its
/// renewals within moments. Its TTL ([`renewing_hold_timing`]) is what keeps
/// a late renewal from lapsing it.
const RENEW_EVERY: Duration = Duration::from_millis(100);

/// The `{gc}` read hold timing of a test whose subject is a hold kept live
/// by its renewals (the test-writing guide's rule (c)): a TTL of twice the
/// command timeout of `client` — the client the renewals run on — so
/// 0.75·TTL exceeds it, and a renewal can let the field lapse only by running
/// later than that client's own commands may; renewed every
/// [`RENEW_EVERY`]. The test asserts the deadlines renewals write, never
/// that the store's clock passed one.
fn renewing_hold_timing(client: &ValkeyClient) -> ReadHoldTiming {
    let command_timeout = client.pool.next().perf_config().default_command_timeout;
    assert!(
        command_timeout > Duration::ZERO,
        "precondition: the renewals' client bounds its commands"
    );
    ReadHoldTiming {
        ttl: command_timeout * 2,
        renew_every: RENEW_EVERY,
    }
}

/// The one `{gc}` read hold field of a holder of `object_id`: its field name
/// and its `(start_ms, expires_ms)`.
async fn the_hold_of(tv: &TestValkey, object_id: u64) -> (String, (u64, u64)) {
    let fields: std::collections::HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().ext_read_hold())
        .await
        .expect("HGETALL read_hold");
    let prefix = format!("{object_id}:");
    let mine: Vec<(String, String)> = fields
        .into_iter()
        .filter(|(field, _)| field.starts_with(&prefix))
        .collect();
    assert_eq!(mine.len(), 1, "one hold field for the object: {mine:?}");
    let (field, value) = mine.into_iter().next().expect("one field");
    let (start, expires) = value
        .split_once(':')
        .unwrap_or_else(|| panic!("a hold field's value is <start>:<expires>; got {value}"));
    (
        field,
        (
            start.parse().expect("a numeric start"),
            expires.parse().expect("a numeric expiry"),
        ),
    )
}

/// The store's clock, in ms (`TIME`), as the hold script reads it.
async fn store_now_ms(tv: &TestValkey) -> u64 {
    let reply: Vec<String> = tv
        .client()
        .pool
        .custom(
            fred::types::CustomCommand::new_static(
                "TIME",
                fred::types::ClusterHash::FirstKey,
                false,
            ),
            Vec::<fred::types::Value>::new(),
        )
        .await
        .expect("TIME must succeed");
    let secs: u64 = reply[0].parse().expect("TIME seconds");
    let micros: u64 = reply[1].parse().expect("TIME microseconds");
    secs * 1000 + micros / 1000
}

/// Wait until a renewal of `object_id`'s hold field moves its deadline past
/// `expires`, within [`DEADLINE`], and check what it wrote: the field's start
/// is `start`, and the new deadline is `ttl` past the store time the renewal
/// ran at — after the write that set `expires`, and not after now. Returns
/// the new deadline. The sleep is the probe interval, not a coordination
/// primitive: the wait ends on the renewal itself.
async fn observe_a_renewal(
    tv: &TestValkey,
    object_id: u64,
    (start, expires): (u64, u64),
    ttl: Duration,
    what: &str,
) -> u64 {
    let ttl_ms = ttl.as_millis() as u64;
    let deadline = tokio::time::Instant::now() + DEADLINE;
    loop {
        let (_, (now_start, now_expires)) = the_hold_of(tv, object_id).await;
        if now_expires > expires {
            let now = store_now_ms(tv).await;
            let renewed_at = now_expires - ttl_ms;
            assert_eq!(
                now_start, start,
                "{what}: a renewal keeps the field's start"
            );
            assert!(
                expires - ttl_ms < renewed_at && renewed_at <= now,
                "{what}: the renewal's deadline is one TTL past the store time it ran at, after \
                 the previous write and not after now: ran at {renewed_at}, previous write at {}, \
                 now {now}",
                expires - ttl_ms
            );
            return now_expires;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what}: no renewal of the hold within {DEADLINE:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// H-b: a rebuild parked under its read hold stays protected because the
/// hold is renewed. The hold's TTL is sized per the test-writing guide's rule
/// (c) ([`renewing_hold_timing`]); while the rebuild stays parked, two
/// renewals are observed, each keeping the field's start and writing a
/// deadline one TTL past the store time it ran at, and GC passes around them
/// leave stored a blob the re-read references whose release was enqueued
/// after the hold began; released, the rebuild delivers that blob at the
/// re-read's cut. Without renewal no renewal is ever observed and the test
/// fails at that wait: the TTL is never waited out.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rebuild_parked_across_hold_renewals_stays_protected() {
    let tv = TestValkey::setup().await;
    let timing = renewing_hold_timing(tv.client());
    tv.client().set_read_hold_timing_for_test(timing);
    let (old, fresh, newest) = (blob(0xD1), blob(0xD2), blob(0xD3));
    let layer = "renewed";
    let (svc, store, session_id, object_id, mut stream) =
        park_under_the_hold(&tv, layer, &old, &fresh).await;
    assert_eq!(hold_fields(&tv).await, 1, "precondition: the rebuild holds");
    // Releases `fresh`, after the hold began.
    commit_ext(&svc, &session_id, layer, object_id, false, &newest).await; // seq 4

    let gc = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone());
    let (_, (start, mut expires)) = the_hold_of(&tv, object_id).await;
    for renewal in 1..=2 {
        gc_pass(&gc).await;
        assert!(
            blob_stored(&tv, &fresh).await,
            "a blob the parked rebuild needs was collected: the hold did not protect it"
        );
        expires = observe_a_renewal(
            &tv,
            object_id,
            (start, expires),
            timing.ttl,
            &format!("renewal {renewal} of the parked rebuild's hold"),
        )
        .await;
    }
    gc_pass(&gc).await;
    assert!(
        blob_stored(&tv, &fresh).await,
        "a blob the parked rebuild needs was collected after the renewals"
    );
    store.release();

    let (content, body) = next_content(&mut stream, "the renewed rebuild").await;
    assert_eq!((content.object_id, content.sequence_num), (object_id, 3));
    assert!(carried(&body).contains(&fresh));
    expect_caught_up_ack(&mut stream, object_id, 3).await;
    let (live, live_body) = next_content(&mut stream, "the commit after the cut").await;
    assert_eq!(live.sequence_num, 4);
    assert!(carried(&live_body).contains(&newest));
    assert_eq!(atomic_misses(&svc.metrics), (1, 0));
    assert_eq!(hold_fields(&tv).await, 0, "the hold is released");
}

/// H-c: with a 200 ms hold whose first renewal comes at 600 ms, the renewal
/// finds the field lapsed and the store answers LAPSED; the catch-up ends
/// UNAVAILABLE before any content — its re-read's blobs may have been
/// collected in the gap — never INTERNAL, and leaves no field behind.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_renewal_that_finds_the_hold_lapsed_ends_the_catch_up_unavailable() {
    let tv = TestValkey::setup().await;
    tv.client().set_read_hold_timing_for_test(ReadHoldTiming {
        ttl: Duration::from_millis(200),
        renew_every: Duration::from_millis(600),
    });
    let (old, fresh) = (blob(0xD4), blob(0xD5));
    let (svc, _store, _session_id, _object_id, mut stream) =
        park_under_the_hold(&tv, "lapsed", &old, &fresh).await;

    // The rebuild stays parked; the renewal ends it.
    let status = expect_ended_before_content(&mut stream, "a lapsed hold").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert!(
        status.message().contains(READ_HOLD_REFUSED_PHRASE),
        "the read-hold refusal: {status:?}"
    );
    assert_eq!(atomic_misses(&svc.metrics), (1, 0));
    assert_eq!(adapter_consults(&svc.metrics), 0);
    assert_eq!(hold_fields(&tv).await, 0, "no field is left or revived");
}

// ═════════════════════════════════════════════════════════════════════════════
// The re-read through a flight: the fenced re-attach, the cut-based routing,
// and the registration swap
// ═════════════════════════════════════════════════════════════════════════════

/// Tests 6f, 7f, H-f and 6-swap, and the hold-before-attach cases: a re-read
/// after an external-value miss that reads through the object's shared
/// fenced-read flight, and the stream-registration swap every cut-based
/// re-read makes. Every case drives a test seam, so the module exists only
/// under `test-fault-injection`.
///
/// # How a fenced miss is staged
///
/// The fixture ([`fixture`]) runs every catch-up on a client whose atomic-read
/// limits ([`LOW_ITEMS`] / [`LOW_BYTES`]) are far below the padded layer
/// ([`PADDING_SECTIONS`]), so a catch-up's own atomic read answers
/// `TOO_LARGE` and it attaches to the layer's flight; that client also reads
/// the flights, takes the `{gc}` read holds and carries the service's metrics
/// (flight roles, fenced reads, the multiplexer's subscriptions). The first
/// build — in its build turn in the first flight, its snapshot at the first
/// cut C1 referencing `old` — parks before its first blob read
/// (`CatchUpStateReadSeam::new_before_ext_probe`); the test lands the rewrite
/// to `fresh` at C1 + 1, collects `old`, and releases it. Its probe misses
/// `old` (`stale_retried{read="shared_fenced"}`), it gives back its turn and
/// its place in the first flight, takes its hold and re-attaches to a
/// successor flight, whose cut is C2 = C1 + 1.
///
/// # Waits
///
/// Each test body runs under [`TEST_BUDGET`] (100 s), inside nextest's 120 s
/// hang-killer, so a stuck test fails with its message; inside it every wait
/// is a deadline ([`DEADLINE`] per message, park or polled condition,
/// [`RELEASED_WITHIN`] for a hold field to go, [`CAP_BUDGET`] for the
/// reader-wait cap). The healthy path pays none of them. Paid on purpose: a
/// renewed hold's park across two observed renewals (a few
/// [`RENEW_EVERY`]s), the [`SHORT_BUDGET_MS`] capacity budget the capacity
/// case runs out, and the real `reader_wait_max(L)` (about 27 s) the
/// reader-cap case waits out.
#[cfg(feature = "test-fault-injection")]
mod re_read_through_a_flight {
    use std::future::Future;

    use fred::types::scripts::Script;

    use live_edit::error::{refusal_reason, CAPACITY_RETRY_DELAY};
    use live_edit::grpc::subscribe::CatchUpStateReadSeam;
    use live_edit::observability::metrics::{
        STREAM_EXIT_CATCHUP_CAPACITY, STREAM_EXIT_CLIENT_DISCONNECT, STREAM_EXIT_ERROR,
        STREAM_EXIT_FENCED_READ_WAIT_CAP, STREAM_EXIT_INNER_WATCHDOG, STREAM_EXIT_SHUTDOWN,
    };
    use live_edit::storage::diff_read::{
        ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget, ObjectSnapshotReader,
        SnapshotReadConsumer, SnapshotReadHold,
    };
    use live_edit::storage::external_values::ReadHoldTakeHold;
    use live_edit::storage::snapshot_flights::{FlightLeaveSeam, FlightState, SnapshotFlights};
    use live_edit::storage::subscriber_admission::{CapacityBudget, CatchUpReadPermit};

    use super::*;
    use crate::common::refusal_detail::{
        assert_no_refusal_detail, assert_refusal_detail, assert_shutdown_status,
    };

    // ── Constants ───────────────────────────────────────────────────────────

    /// The fenced fixture's atomic-read item ceiling: far below its padded
    /// layer, so every catch-up of it reads through the layer's flight. The
    /// ceiling of `subscribe_fenced_catch_up_test`.
    const LOW_ITEMS: u64 = 16;

    /// The fenced fixture's atomic-read byte ceiling.
    const LOW_BYTES: u64 = 256;

    /// Sections a fenced layer is padded with before its external value, so it
    /// is above [`LOW_ITEMS`] and [`LOW_BYTES`].
    const PADDING_SECTIONS: u64 = 6;

    /// A capacity budget no wait here runs out of.
    const ROOMY_BUDGET_MS: u64 = 60_000;

    /// The capacity budget of the case whose successor flight's capacity wait
    /// runs it out.
    const SHORT_BUDGET_MS: u64 = 1_000;

    /// The one deadline over a whole test body; see the module's waits.
    const TEST_BUDGET: Duration = Duration::from_secs(100);

    /// Bound on the ending of a catch-up whose successor flight sleeps past
    /// `reader_wait_max(L)` (about 27 s at these sizes).
    const CAP_BUDGET: Duration = Duration::from_secs(60);

    /// How soon an ended catch-up's hold field must be gone: far inside the
    /// hold's 15 s TTL, so a field left to lapse cannot pass for a release.
    const RELEASED_WITHIN: Duration = Duration::from_secs(5);

    /// The shrunk per-subscription backlog hard cap.
    const TINY_HARD_CAP: usize = 8;

    /// Inline commits written past [`TINY_HARD_CAP`].
    const OVERFLOW_COMMITS: u8 = 12;

    /// A staged writers' gap past every `reader_wait_max(L)` here.
    const FAR_MS: i64 = 600_000;

    /// The layer size a staged writers' gap names.
    const GAP_SIZE: LayerSize = LayerSize {
        items: 50,
        bytes: 5_000,
    };

    /// The size a flight the test registers itself is created with.
    const TEST_FLIGHT_SIZE: LayerSize = LayerSize {
        items: 1_000,
        bytes: 100_000,
    };

    /// The fence scripts' clock: `TIME` in whole ms, as they read it.
    static CLOCK_SCRIPT: std::sync::LazyLock<Script> = std::sync::LazyLock::new(|| {
        Script::from_lua(
            "local t = redis.call('TIME') \
             return tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)",
        )
    });

    /// Run a test's body under [`TEST_BUDGET`], boxed so its future lives on
    /// the heap.
    async fn within_test_budget(body: impl Future<Output = ()>) {
        tokio::time::timeout(TEST_BUDGET, Box::pin(body))
            .await
            .unwrap_or_else(|_| panic!("the test did not finish within {TEST_BUDGET:?}"));
    }

    // ── Fixture ─────────────────────────────────────────────────────────────

    /// The capacity and atomic-read limits a fixture's catch-up client is
    /// built with.
    #[derive(Clone, Copy)]
    struct Shape {
        /// `LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS`.
        slots: usize,
        /// `LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS`: each subscription's
        /// capacity budget.
        budget_ms: u64,
        /// The atomic-read item ceiling; `None` keeps the test config's.
        items: Option<u64>,
        /// The atomic-read byte ceiling; `None` keeps the test config's.
        bytes: Option<u64>,
    }

    impl Shape {
        /// Limits far below a padded layer: every catch-up takes a flight.
        const fn fenced(slots: usize, budget_ms: u64) -> Self {
            Self {
                slots,
                budget_ms,
                items: Some(LOW_ITEMS),
                bytes: Some(LOW_BYTES),
            }
        }

        /// An unpadded layer's catch-up reads atomically, at an item ceiling
        /// of `items` (`None`: the test config's).
        const fn atomic(items: Option<u64>) -> Self {
            Self {
                slots: 2,
                budget_ms: ROOMY_BUDGET_MS,
                items,
                bytes: None,
            }
        }
    }

    /// A service in snapshot mode whose external values live in a
    /// [`ScriptedProbeStore`] running `script`, with an empty checkpoint
    /// adapter. Its catch-ups, its flights and its read holds run on one
    /// client of `shape` with the service's metrics installed (flight roles,
    /// fenced reads, the multiplexer's gauges); the layer, session and
    /// external-value stores keep the fixture's client, on the same database.
    struct Fixture {
        svc: LiveEditService,
        store: Arc<ScriptedProbeStore>,
        adapter: Arc<InMemoryCheckpointStore>,
    }

    async fn fixture(tv: &TestValkey, shape: Shape, script: Vec<OnProbe>) -> Fixture {
        live_edit::storage::test_support::raise_fd_limit_for_tests();
        let mut config = crate::common::test_config();
        if let Some(items) = shape.items {
            config.snapshot_atomic_read_max_items = items;
        }
        if let Some(bytes) = shape.bytes {
            config.snapshot_atomic_read_max_bytes = bytes;
        }
        config.subscriber_max_concurrent_catchups = shape.slots;
        config.subscriber_admission_wait_ms = shape.budget_ms;
        config.subscriber_admission_max_waiters = None;
        let client = live_edit::storage::test_support::connect_test_client(&config)
            .await
            .expect("the catch-up client connects");
        let mut scripted = None;
        let (svc, adapter) = service_with(tv, true, |store| {
            let store = Arc::new(ScriptedProbeStore::new(store, tv, script));
            scripted = Some(Arc::clone(&store));
            store
        });
        client
            .install_observability_hooks(Arc::clone(&svc.metrics))
            .await;
        let svc = LiveEditService {
            snapshot_flights: SnapshotFlights::new(client.clone(), tv.key_schema().clone()),
            valkey: client,
            ..svc
        };
        Fixture {
            svc,
            store: scripted.expect("the wrap closure ran"),
            adapter,
        }
    }

    /// A seeded layer: its session, name and object, and the cut a first read
    /// of it takes.
    struct Seeded {
        session_id: String,
        layer: String,
        object_id: u64,
        first_cut: u64,
    }

    /// Create `layer` (seq 1) and write [`FIELD`] as the external value
    /// `value`, in [`SECTION`]. `padded` first adds [`PADDING_SECTIONS`]
    /// sections (seq 2), which puts the layer above the fenced fixture's
    /// atomic-read limits.
    async fn seed(
        tv: &TestValkey,
        svc: &LiveEditService,
        layer: &str,
        padded: bool,
        value: &[u8],
    ) -> Seeded {
        let (session_id, object_id) =
            create_session_and_layer(tv, svc, "b-yuo-re-read", layer).await;
        if padded {
            let padding = OwnedDelta {
                struct_commands: (0..PADDING_SECTIONS)
                    .map(|i| {
                        StructCommand::CreateSection(CommandCreateSection {
                            parent_id: 0,
                            section_id: 100 + i,
                            section_name: format!("Padding{i}"),
                            section_type: 1,
                        })
                    })
                    .collect(),
                ..Default::default()
            };
            upload(
                svc,
                &session_id,
                layer,
                object_id,
                delta_to_ompf_bytes(&padding, &[]),
            )
            .await;
        }
        commit_ext(svc, &session_id, layer, object_id, true, value).await;
        Seeded {
            session_id,
            layer: layer.to_owned(),
            object_id,
            first_cut: if padded { 3 } else { 2 },
        }
    }

    // ── Park guards: every park is released on every path ───────────────────

    /// The service's state-read seam, parking the first build before its
    /// first blob read, released on drop if the test has not released it (by
    /// a spawned task: the release is an async rendezvous), so a failing test
    /// never leaves a subscription parked.
    struct Parks {
        seam: Arc<CatchUpStateReadSeam>,
        ext_probe_pending: bool,
    }

    impl Parks {
        /// Install the seam on `svc`.
        fn install(svc: &mut LiveEditService) -> Self {
            let seam = Arc::new(CatchUpStateReadSeam::new_before_ext_probe());
            svc.catch_up_state_read_seam_for_test = Some(Arc::clone(&seam));
            Self {
                seam,
                ext_probe_pending: true,
            }
        }

        async fn parked_pre_ext_probe(&self) {
            tokio::time::timeout(DEADLINE, self.seam.wait_until_parked_pre_ext_probe())
                .await
                .expect("the first build never parked before its first blob read");
        }

        async fn release_pre_ext_probe(&mut self) {
            self.ext_probe_pending = false;
            tokio::time::timeout(DEADLINE, self.seam.release_pre_ext_probe())
                .await
                .expect("the parked build accepts its release");
        }
    }

    impl Drop for Parks {
        fn drop(&mut self) {
            let Ok(runtime) = tokio::runtime::Handle::try_current() else {
                return;
            };
            if self.ext_probe_pending {
                let seam = Arc::clone(&self.seam);
                runtime.spawn(async move { seam.release_pre_ext_probe().await });
            }
        }
    }

    /// Releases a `{gc}` read-hold take held back by
    /// `ValkeyClient::arm_read_hold_take_hold_for_test`, on drop.
    struct ReleaseTake(Arc<ReadHoldTakeHold>);

    impl Drop for ReleaseTake {
        fn drop(&mut self) {
            self.0.release();
        }
    }

    /// Releases a flight leave parked on the seam, on drop.
    struct ReleaseLeave(Arc<FlightLeaveSeam>);

    impl Drop for ReleaseLeave {
        fn drop(&mut self) {
            self.0.release();
        }
    }

    /// Releases a held snapshot-read command, on drop.
    struct ReleaseRead(Arc<SnapshotReadHold>);

    impl Drop for ReleaseRead {
        fn drop(&mut self) {
            self.0.release();
        }
    }

    /// Releases a [`OnProbe::Park`] of the store, on drop.
    struct ReleaseProbe(Arc<ScriptedProbeStore>);

    impl Drop for ReleaseProbe {
        fn drop(&mut self) {
            self.0.release();
        }
    }

    // ── Staging ─────────────────────────────────────────────────────────────

    /// Subscribe brand new and wait until its first build — on a fenced
    /// fixture, in its build turn in the layer's first flight — parks before
    /// its first blob read, its snapshot at the first cut referencing `old`;
    /// then land the rewrite of [`FIELD`] to `fresh` (the first cut + 1),
    /// which releases `old`, and let the GC collect `old`. The caller releases
    /// the park.
    async fn subscribe_and_stale_its_first_build(
        tv: &TestValkey,
        svc: &LiveEditService,
        parks: &Parks,
        seeded: &Seeded,
        old: &[u8],
        fresh: &[u8],
    ) -> SubscribeStream {
        let stream = subscribe(svc, &seeded.session_id, &seeded.layer, 0, 0).await;
        parks.parked_pre_ext_probe().await;
        commit_ext(
            svc,
            &seeded.session_id,
            &seeded.layer,
            seeded.object_id,
            false,
            fresh,
        )
        .await;
        collect(tv, old).await;
        stream
    }

    /// Wait until a build parks in the store (its next [`OnProbe::Park`]),
    /// failing at once, with what the stream gave, if the stream ends or
    /// sends anything first.
    async fn parked_in_build(store: &ScriptedProbeStore, stream: &mut SubscribeStream, what: &str) {
        tokio::select! {
            () = store.wait_parked() => {}
            item = stream.next() => {
                panic!("{what}: the stream gave {item:?} before the build parked")
            }
        }
    }

    /// Fields on the `{gc}` read hold that belong to holders of `object_id`
    /// (`<object_id>:<token>`).
    async fn hold_fields_for(tv: &TestValkey, object_id: u64) -> usize {
        let fields: Vec<String> = tv
            .client()
            .pool
            .hkeys(tv.key_schema().ext_read_hold())
            .await
            .expect("HKEYS read_hold");
        let prefix = format!("{object_id}:");
        fields
            .iter()
            .filter(|field| field.starts_with(&prefix))
            .count()
    }

    /// Wait, from a plain OS thread, until a flight leave parks on `seam`,
    /// and read the `{gc}` read-hold HASH there while it stays parked, with a
    /// runtime and a client of the thread's own. Returns whether the parked
    /// leave counted out its flight's last waiter, and the fields that belong
    /// to holders of `object_id`; the leave stays parked.
    ///
    /// Off the test's runtime because the parked leave blocks the runtime
    /// worker its catch-up runs on (it parks inside the turn's drop), and a
    /// worker blocked that way can hold up the runtime's I/O and timer
    /// driver: a wait or a Valkey read on the test's runtime could then stall
    /// until the park ends. The report comes back on a oneshot channel,
    /// whose wake reaches the test without that driver.
    async fn while_the_leave_is_parked(
        seam: Arc<FlightLeaveSeam>,
        tv: &TestValkey,
        object_id: u64,
    ) -> (Option<bool>, usize) {
        let key = tv.key_schema().ext_read_hold();
        let (report, reported) = tokio::sync::oneshot::channel();
        std::thread::spawn(move || {
            let parked = seam.wait_until_parked(DEADLINE);
            let was_last = seam.parked_was_last();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("the observer thread builds a runtime of its own");
            let fields: Vec<String> = runtime.block_on(async move {
                let client = live_edit::storage::test_support::connect_test_client(
                    &crate::common::test_config(),
                )
                .await
                .expect("the observer's client connects");
                let fields: Vec<String> = client.pool.hkeys(key).await.expect("HKEYS read_hold");
                client.quit().await;
                fields
            });
            report
                .send((parked, was_last, fields))
                .expect("the test awaits the observer's report");
        });
        let (parked, was_last, fields) = reported.await.expect("the observer thread reports");
        assert!(
            parked,
            "the late waiter's leave of its old flight never parked within {DEADLINE:?}"
        );
        let prefix = format!("{object_id}:");
        let held = fields
            .iter()
            .filter(|field| field.starts_with(&prefix))
            .count();
        (was_last, held)
    }

    /// Wait until `object_id`'s holders hold `count` fields, within `within`.
    async fn until_hold_fields(
        tv: &TestValkey,
        object_id: u64,
        count: usize,
        within: Duration,
        what: &str,
    ) {
        let deadline = tokio::time::Instant::now() + within;
        loop {
            let held = hold_fields_for(tv, object_id).await;
            if held == count {
                return;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "{what}: {held} hold field(s) for the object after {within:?}, expected {count}"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Poll `condition` until it holds, within [`DEADLINE`]. The sleep is the
    /// probe interval, not a coordination primitive.
    async fn until(what: &str, mut condition: impl FnMut() -> bool) {
        let deadline = tokio::time::Instant::now() + DEADLINE;
        while !condition() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "{what} did not hold within {DEADLINE:?}"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// Run GC passes under `GcPolicy::IMMEDIATE` for at least `span` (and at
    /// least three), asserting after each that `value` is still stored. The
    /// sleep between passes spends the span; it coordinates nothing.
    async fn gc_passes_keep(tv: &TestValkey, value: &[u8], span: Duration, what: &str) {
        let gc = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone());
        let until = tokio::time::Instant::now() + span;
        let mut passes = 0;
        while passes < 3 || tokio::time::Instant::now() < until {
            gc_pass(&gc).await;
            passes += 1;
            assert!(
                blob_stored(tv, value).await,
                "{what}: the blob was collected by GC pass {passes}"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Install a snapshot-read fault injector on `client` and return it.
    fn injector(client: &ValkeyClient) -> Arc<ObjectSnapshotReadFaultInjector> {
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        client.install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
        injector
    }

    /// The first command of the next fenced fan-out read on the client.
    fn next_fenced_read() -> ObjectSnapshotReadFaultTarget {
        ObjectSnapshotReadFaultTarget::nth_command(ObjectSnapshotReader::FencedFanout, 0)
    }

    /// Take the client's one fenced read permit: every flight then waits for
    /// it, attachable.
    async fn hold_fenced_permit(client: &ValkeyClient) -> CatchUpReadPermit {
        client
            .catchup_fenced_read_gate
            .acquire(
                tokio::time::Instant::now() + Duration::from_secs(3_600),
                SnapshotReadConsumer::Catchup,
            )
            .await
            .unwrap_or_else(|r| panic!("the test's fenced read permit must be free: {r:?}"))
    }

    /// Stage a writers' gap on `object_id`'s fence key ending `ahead_ms` after
    /// the store's clock, naming [`GAP_SIZE`]: a fence acquire answers
    /// `RETRY_AFTER gap` until then.
    async fn stage_writers_gap(tv: &TestValkey, object_id: u64, ahead_ms: i64) {
        let fence_key = tv.key_schema().obj_snapshot_fence(&object_id.to_string());
        let now: i64 = CLOCK_SCRIPT
            .evalsha_with_reload::<i64, _, _>(
                tv.client().pool.next(),
                vec![fence_key.clone()],
                Vec::<String>::new(),
            )
            .await
            .expect("TIME must answer");
        let _: () = tv
            .client()
            .pool
            .hset(
                &fence_key,
                [
                    ("gap_until", (now + ahead_ms).to_string()),
                    ("size_e", GAP_SIZE.items.to_string()),
                    ("size_b", GAP_SIZE.bytes.to_string()),
                ],
            )
            .await
            .expect("staging the gap must succeed");
        let _: i64 = tv
            .client()
            .pool
            .pexpire(&fence_key, ahead_ms * 2, None)
            .await
            .expect("staging the key expiry must succeed");
    }

    /// Shrink every new stream registration's sink to a hard cap of
    /// [`TINY_HARD_CAP`] entries, with the soft watchdog out of reach, so only
    /// the hard cap can end it. Applies to registrations made after the call.
    fn install_tiny_hard_cap(svc: &LiveEditService) {
        let knobs = svc.valkey.stream_mux.knobs();
        svc.valkey
            .stream_mux
            .set_knobs_for_test(live_edit::storage::stream_mux::StreamMuxKnobs {
                channel_capacity: TINY_HARD_CAP / 2,
                backlog_hard_cap: TINY_HARD_CAP,
                send_watchdog: Duration::from_secs(600),
                ..knobs
            });
    }

    /// The stream's ending status, expected before any message, within
    /// `bound`.
    async fn ended_within(stream: &mut SubscribeStream, bound: Duration, what: &str) -> Status {
        let item = tokio::time::timeout(bound, stream.next())
            .await
            .unwrap_or_else(|_| panic!("{what}: the stream did not end within {bound:?}"))
            .unwrap_or_else(|| panic!("{what}: the stream closed without a status"));
        match item {
            Err(status) => status,
            Ok(message) => panic!("{what}: expected the stream to end, got {message:?}"),
        }
    }

    // ── Metrics ─────────────────────────────────────────────────────────────

    fn metric(metrics: &Metrics, name: &str, labels: &[(&str, &str)]) -> i64 {
        sample_value(&render(metrics), name, labels).unwrap_or(0.0) as i64
    }

    /// `live_edit_catchup_snapshot_flight_total{role}`: flights created
    /// (`leader`) and joined (`joined_read`).
    fn roles(metrics: &Metrics, role: &str) -> i64 {
        metric(
            metrics,
            "live_edit_catchup_snapshot_flight_total",
            &[("role", role)],
        )
    }

    /// Verified fenced catch-up reads: one per flight read.
    fn fenced_reads(metrics: &Metrics) -> i64 {
        metric(
            metrics,
            "live_edit_snapshot_read_total",
            &[
                ("consumer", "catchup"),
                ("path", "fenced"),
                ("outcome", "ok"),
            ],
        )
    }

    /// `live_edit_stream_mux_streams`: registered streams the multiplexer
    /// reads; a reservation is not one.
    fn streams(metrics: &Metrics) -> i64 {
        metric(metrics, "live_edit_stream_mux_streams", &[])
    }

    fn exits(metrics: &Metrics, reason: &str) -> i64 {
        metric(
            metrics,
            "live_edit_subscriber_stream_exits_total",
            &[("reason", reason)],
        )
    }

    /// `live_edit_stream_mux_overflow_kills_total{bound="entries"}`.
    fn overflow_kills(metrics: &Metrics) -> i64 {
        metric(
            metrics,
            "live_edit_stream_mux_overflow_kills_total",
            &[("bound", "entries")],
        )
    }

    /// `(stale_retried, genuine)` on the shared fenced read, with the atomic
    /// cells checked to be zero: every miss here is found in a flight's
    /// snapshot.
    fn fenced_misses(metrics: &Metrics) -> (u64, u64) {
        for outcome in ["stale_retried", "genuine"] {
            assert_eq!(
                misses(metrics, outcome, "atomic"),
                0,
                "no miss here is found in a catch-up's own atomic snapshot"
            );
        }
        (
            misses(metrics, "stale_retried", "shared_fenced"),
            misses(metrics, "genuine", "shared_fenced"),
        )
    }

    // ── 6f-a and the order pin ──────────────────────────────────────────────

    /// 6f-a, with the order pin. A late waiter whose build from its flight's
    /// shared snapshot misses a blob collected after that snapshot gives back
    /// its build turn, snapshot handle and place in the old flight BEFORE it
    /// takes its `{gc}` read hold: while that leave is parked
    /// (`SnapshotFlights::arm_leave_seam_for_test`), the read-hold HASH holds
    /// no field for the object. It then takes the hold and re-attaches to one
    /// successor flight, whose cut C2 = C1 + 1 its payload and ack carry: the
    /// fresh value and no collected one, the checkpoint adapter never
    /// consulted (no fetch-on-miss, no INTERNAL), the rewrite between the cuts
    /// delivered once (folded into the payload; the next message is the commit
    /// after it), `stale_retried{read="shared_fenced"}` once, and the hold gone
    /// once the content arrives. One successor read: two flights created, none
    /// joined, two fenced reads; one stream registration, none left behind.
    ///
    /// The parked leave blocks the runtime worker the catch-up runs on (it
    /// parks inside the turn's drop), so the test observes the park, and reads
    /// the HASH, from a thread of its own ([`while_the_leave_is_parked`]).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_stale_fenced_miss_re_attaches_to_one_successor_flight_and_serves_its_cut() {
        within_test_budget(async {
            let tv = TestValkey::setup().await;
            let mut svc = fixture(&tv, Shape::fenced(2, ROOMY_BUDGET_MS), Vec::new())
                .await
                .svc;
            let (old, fresh) = (blob(0xE1), blob(0xE2));
            let seeded = seed(&tv, &svc, "re-attach", true, &old).await;
            let object_id = seeded.object_id;
            let mut parks = Parks::install(&mut svc);
            let svc = Arc::new(svc);
            let before = subscriptions(&svc.metrics);
            let mut stream =
                subscribe_and_stale_its_first_build(&tv, &svc, &parks, &seeded, &old, &fresh).await;
            let leave = ReleaseLeave(svc.snapshot_flights.arm_leave_seam_for_test(object_id));
            parks.release_pre_ext_probe().await;

            let (was_last, held) =
                while_the_leave_is_parked(Arc::clone(&leave.0), &tv, object_id).await;
            assert_eq!(
                was_last,
                Some(true),
                "precondition: the late waiter is its old flight's last waiter"
            );
            assert_eq!(
                held, 0,
                "the order pin: the old flight's turn, snapshot handle and place go back before \
                 the read hold is taken, so no hold field exists while that leave is parked"
            );
            leave.0.release();

            let new_cut = seeded.first_cut + 1;
            let (content, body) = next_content(&mut stream, "the successor's catch-up").await;
            assert_eq!(
                hold_fields_for(&tv, object_id).await,
                0,
                "no hold field is left once the content arrives"
            );
            assert_eq!(
                (content.object_id, content.sequence_num),
                (object_id, new_cut),
                "the payload carries the successor's cut"
            );
            let parts = carried(&body);
            assert!(
                parts.contains(&fresh) && !parts.contains(&old),
                "the payload carries the fresh value, not the collected one"
            );
            assert_eq!(field_hash(&body), Some(ext_content_hash(&fresh).to_vec()));
            expect_caught_up_ack(&mut stream, object_id, new_cut).await;
            commit_inline(&svc, &seeded.session_id, &seeded.layer, object_id, 2).await;
            let (live, _) = next_content(&mut stream, "the next commit").await;
            assert_eq!(
                live.sequence_num,
                new_cut + 1,
                "the rewrite is at the ack, so the next message is the next commit"
            );

            assert_eq!(fenced_misses(&svc.metrics), (1, 0));
            assert_eq!(
                adapter_consults(&svc.metrics),
                0,
                "no fetch-on-miss: the checkpoint adapter is never consulted"
            );
            assert_eq!(
                (
                    roles(&svc.metrics, "leader"),
                    roles(&svc.metrics, "joined_read"),
                    fenced_reads(&svc.metrics),
                ),
                (2, 0, 2),
                "the first flight and one successor, each read once"
            );
            assert_eq!(
                subscriptions(&svc.metrics),
                before + 1,
                "one stream registration: the swap leaves none behind"
            );
            drop(stream);
            until("the subscription's registration is released", || {
                subscriptions(&svc.metrics) == before
            })
            .await;
        })
        .await;
    }

    // ── 6f-b ────────────────────────────────────────────────────────────────

    /// 6f-b. Two late waiters of one flight that both miss share exactly one
    /// successor read. The replica's one catch-up slot keeps the second
    /// waiter's build queued for the flight's turn behind the first's (no
    /// extra build), so both build after the collection and both miss. The
    /// test holds the client's fenced read permit while they re-attach, so
    /// the successor waits for it, attachable, until both are attached — one
    /// `leader` and one `joined_read` on it; the late waiters' own places no
    /// longer hold it back. Released, it reads once and both are served at
    /// its cut, each with its own hold, released once its content arrives.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_late_waiters_that_miss_share_one_successor_read() {
        within_test_budget(async {
            let tv = TestValkey::setup().await;
            let mut svc = fixture(&tv, Shape::fenced(1, ROOMY_BUDGET_MS), Vec::new())
                .await
                .svc;
            let (old, fresh) = (blob(0xE3), blob(0xE4));
            let seeded = seed(&tv, &svc, "two-late-waiters", true, &old).await;
            let object_id = seeded.object_id;
            let mut parks = Parks::install(&mut svc);
            let svc = Arc::new(svc);

            let permit = hold_fenced_permit(&svc.valkey).await;
            let mut first = subscribe(&svc, &seeded.session_id, &seeded.layer, 0, 0).await;
            until("the first waiter creates the first flight", || {
                roles(&svc.metrics, "leader") == 1
            })
            .await;
            let mut second = subscribe(&svc, &seeded.session_id, &seeded.layer, 0, 0).await;
            until("the second waiter joins it", || {
                roles(&svc.metrics, "joined_read") == 1
            })
            .await;
            drop(permit);
            parks.parked_pre_ext_probe().await;
            commit_ext(
                &svc,
                &seeded.session_id,
                &seeded.layer,
                object_id,
                false,
                &fresh,
            )
            .await;
            collect(&tv, &old).await;
            let permit = hold_fenced_permit(&svc.valkey).await;
            parks.release_pre_ext_probe().await;

            until(
                "both late waiters attach to one successor, held open by the fenced permit",
                || roles(&svc.metrics, "leader") == 2 && roles(&svc.metrics, "joined_read") == 2,
            )
            .await;
            until_hold_fields(&tv, object_id, 2, DEADLINE, "each late waiter's own hold").await;
            let found = svc
                .snapshot_flights
                .lookup(object_id)
                .expect("the successor is registered");
            assert_eq!(
                (found.attachable, found.state),
                (true, FlightState::Queued),
                "the successor waits for the held permit, attachable"
            );
            drop(permit);

            let new_cut = seeded.first_cut + 1;
            for (who, stream) in [("first", &mut first), ("second", &mut second)] {
                let (content, body) = next_content(stream, who).await;
                assert_eq!(
                    (content.object_id, content.sequence_num),
                    (object_id, new_cut),
                    "{who}: served at the successor's cut"
                );
                assert!(carried(&body).contains(&fresh), "{who}");
                expect_caught_up_ack(stream, object_id, new_cut).await;
            }
            assert_eq!(
                fenced_reads(&svc.metrics),
                2,
                "one read for the first flight and exactly one for the successor"
            );
            assert_eq!(fenced_misses(&svc.metrics), (2, 0));
            assert_eq!(adapter_consults(&svc.metrics), 0);
            assert_eq!(hold_fields_for(&tv, object_id).await, 0);
        })
        .await;
    }

    // ── 6f-c: a cut-based atomic re-read's routing ──────────────────────────

    /// 6f-c. A brand-new catch-up's atomic re-read after a stale miss that
    /// answers `TOO_LARGE` — the layer grew past the atomic-read limits at the
    /// park — hands over to the object's flight, under its hold: no
    /// UNAVAILABLE, the payload and ack at the grown layer's cut with the
    /// fresh value, and the hold live while the build reads from the flight's
    /// snapshot (`probe_values` call 1, parked), released once the content
    /// arrives. The catch-up client carries a small item limit
    /// ([`SMALL_ATOMIC_ITEMS`]); the flight reads fenced, whatever the limit.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_cut_based_re_read_answering_too_large_hands_over_to_a_flight_under_its_hold() {
        within_test_budget(async {
            let tv = TestValkey::setup().await;
            let fx = fixture(
                &tv,
                Shape::atomic(Some(SMALL_ATOMIC_ITEMS)),
                vec![OnProbe::Park, OnProbe::Park],
            )
            .await;
            let (svc, store) = (Arc::new(fx.svc), fx.store);
            let _release = ReleaseProbe(Arc::clone(&store));
            let (old, fresh) = (blob(0xE5), blob(0xE6));
            let seeded = seed(&tv, &svc, "too-large-cut-based", false, &old).await;
            let object_id = seeded.object_id;

            let mut stream = subscribe(&svc, &seeded.session_id, &seeded.layer, 0, 0).await;
            store.wait_parked().await;
            assert!(
                svc.snapshot_flights.lookup(object_id).is_none(),
                "precondition: the first read was the catch-up's own atomic read"
            );
            commit_ext(
                &svc,
                &seeded.session_id,
                &seeded.layer,
                object_id,
                false,
                &fresh,
            )
            .await; // seq 3
            let growth = OwnedDelta {
                struct_commands: (0..GROWTH_SECTIONS)
                    .map(|i| {
                        StructCommand::CreateSection(CommandCreateSection {
                            parent_id: 0,
                            section_id: 1_000 + i,
                            section_name: format!("Growth{i}"),
                            section_type: 1,
                        })
                    })
                    .collect(),
                ..Default::default()
            };
            upload(
                &svc,
                &seeded.session_id,
                &seeded.layer,
                object_id,
                delta_to_ompf_bytes(&growth, &[]),
            )
            .await; // seq 4
            collect(&tv, &old).await;
            until(
                "precondition: no flight is registered before the re-read, so it takes its \
                 atomic read",
                || svc.snapshot_flights.lookup(object_id).is_none(),
            )
            .await;
            store.release();

            parked_in_build(&store, &mut stream, "the re-read's hand-over to a flight").await;
            assert!(
                svc.snapshot_flights.lookup(object_id).is_some(),
                "the re-read handed over to a flight"
            );
            assert_eq!(
                hold_fields_for(&tv, object_id).await,
                1,
                "the hold covers the flight's read and the build from it"
            );
            store.release();

            let (content, body) = next_content(&mut stream, "the flight's catch-up").await;
            assert_eq!((content.object_id, content.sequence_num), (object_id, 4));
            assert!(carried(&body).contains(&fresh));
            expect_caught_up_ack(&mut stream, object_id, 4).await;
            assert_eq!(atomic_misses(&svc.metrics), (1, 0));
            assert_eq!(adapter_consults(&svc.metrics), 0);
            assert_eq!(hold_fields_for(&tv, object_id).await, 0);
        })
        .await;
    }

    /// 6f-c, the other route. A brand-new catch-up's re-read after a stale
    /// miss on its own atomic snapshot that finds the object's flight
    /// registered — here by the test, after the first read — gives its slot
    /// back and reads through that flight, under its hold, sized as the
    /// flight knows the layer: the build from the flight's snapshot
    /// (`probe_values` call 1, parked) runs attached to the test's flight with
    /// the hold live, and the payload and ack carry the flight's cut.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_cut_based_re_read_finding_the_objects_flight_registered_reads_through_it_under_its_hold(
    ) {
        within_test_budget(async {
            let tv = TestValkey::setup().await;
            let fx = fixture(&tv, Shape::atomic(None), vec![OnProbe::Park, OnProbe::Park]).await;
            let (svc, store) = (Arc::new(fx.svc), fx.store);
            let _release = ReleaseProbe(Arc::clone(&store));
            let (old, fresh) = (blob(0xEA), blob(0xEB));
            let seeded = seed(&tv, &svc, "flight-registered-cut-based", false, &old).await;
            let object_id = seeded.object_id;

            let mut stream = subscribe(&svc, &seeded.session_id, &seeded.layer, 0, 0).await;
            store.wait_parked().await;
            commit_ext(
                &svc,
                &seeded.session_id,
                &seeded.layer,
                object_id,
                false,
                &fresh,
            )
            .await; // seq 3
            collect(&tv, &old).await;
            let ours = svc
                .snapshot_flights
                .attach_or_create(object_id, TEST_FLIGHT_SIZE);
            store.release();

            parked_in_build(
                &store,
                &mut stream,
                "the re-read through the registered flight",
            )
            .await;
            assert!(
                ours.flight().waiters() >= 2,
                "the re-read attached to the registered flight"
            );
            assert_eq!(
                hold_fields_for(&tv, object_id).await,
                1,
                "the hold covers the flight's read and the build from it"
            );
            drop(ours);
            store.release();

            let (content, body) = next_content(&mut stream, "the flight's catch-up").await;
            assert_eq!((content.object_id, content.sequence_num), (object_id, 3));
            assert!(carried(&body).contains(&fresh));
            expect_caught_up_ack(&mut stream, object_id, 3).await;
            assert_eq!(atomic_misses(&svc.metrics), (1, 0));
            assert_eq!(adapter_consults(&svc.metrics), 0);
            assert_eq!(hold_fields_for(&tv, object_id).await, 0);
        })
        .await;
    }

    // ── 6f-d: the hold covers the re-attach ─────────────────────────────────

    /// 6f-d. The hold covers the re-attach, from the successor's cut to the
    /// last blob read. The late waiter is parked inside its successor build
    /// turn (its build's `probe_values` call 1), its hold live; a commit then
    /// releases `fresh`, which the successor's cut references. The hold's TTL
    /// is sized per the test-writing guide's rule (c)
    /// ([`renewing_hold_timing`], on the catch-up client its renewals run
    /// on); two renewals are observed while the build stays parked, each
    /// keeping the field's start and writing a deadline one TTL past the store
    /// time it ran at, and GC passes under `GcPolicy::IMMEDIATE` around them
    /// collect nothing. Released, the build delivers `fresh` at the
    /// successor's cut, and the release commit arrives live after it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_hold_covers_the_re_attach_from_the_successors_cut_to_its_last_blob_read() {
        within_test_budget(async {
            let tv = TestValkey::setup().await;
            let fx = fixture(
                &tv,
                Shape::fenced(2, ROOMY_BUDGET_MS),
                vec![OnProbe::Pass, OnProbe::Park],
            )
            .await;
            let (mut svc, store) = (fx.svc, fx.store);
            let _release = ReleaseProbe(Arc::clone(&store));
            let timing = renewing_hold_timing(&svc.valkey);
            svc.valkey.set_read_hold_timing_for_test(timing);
            let (old, fresh, newest) = (blob(0xEC), blob(0xED), blob(0xEE));
            let seeded = seed(&tv, &svc, "hold-covers-re-attach", true, &old).await;
            let object_id = seeded.object_id;
            let mut parks = Parks::install(&mut svc);
            let svc = Arc::new(svc);
            let mut stream =
                subscribe_and_stale_its_first_build(&tv, &svc, &parks, &seeded, &old, &fresh).await;
            parks.release_pre_ext_probe().await;

            parked_in_build(&store, &mut stream, "the build in the successor's turn").await;
            assert_eq!(
                roles(&svc.metrics, "leader"),
                2,
                "precondition: the late waiter builds in a successor flight"
            );
            let new_cut = seeded.first_cut + 1;
            // Releases `fresh`, which the successor's cut references.
            commit_ext(
                &svc,
                &seeded.session_id,
                &seeded.layer,
                object_id,
                false,
                &newest,
            )
            .await;
            let what = "a blob the successor's cut references, released after the cut";
            let (_, (start, mut expires)) = the_hold_of(&tv, object_id).await;
            for renewal in 1..=2 {
                gc_passes_keep(&tv, &fresh, Duration::ZERO, what).await;
                expires = observe_a_renewal(
                    &tv,
                    object_id,
                    (start, expires),
                    timing.ttl,
                    &format!("renewal {renewal} of the late waiter's hold"),
                )
                .await;
            }
            gc_passes_keep(&tv, &fresh, Duration::ZERO, what).await;
            store.release();

            let (content, body) = next_content(&mut stream, "the successor's catch-up").await;
            assert_eq!(
                (content.object_id, content.sequence_num),
                (object_id, new_cut)
            );
            assert!(carried(&body).contains(&fresh));
            expect_caught_up_ack(&mut stream, object_id, new_cut).await;
            let (live, live_body) = next_content(&mut stream, "the release commit").await;
            assert_eq!(live.sequence_num, new_cut + 1);
            assert!(carried(&live_body).contains(&newest));
            assert_eq!(fenced_misses(&svc.metrics), (1, 0));
            assert_eq!(adapter_consults(&svc.metrics), 0);
            assert_eq!(hold_fields_for(&tv, object_id).await, 0);
        })
        .await;
    }

    // ── The successor whose cut is fixed before the hold ────────────────────

    /// The setup of the two cases below. The late waiter's first build parks
    /// before its first blob read; the rewrite to `fresh` lands and `old` is
    /// collected. The test then registers the successor itself
    /// (`attach_or_create` while the late waiter still holds its place in the
    /// first flight, so it succeeds that flight) and drives its read from a
    /// task, with the read's first fenced command held: once the first flight
    /// is done, the successor's fence is granted and its cut — C2, referencing
    /// `fresh` — is fixed, and it stays attachable. The late waiter's `{gc}`
    /// read-hold take is held back at its start, before anything is sent
    /// (`ValkeyClient::arm_read_hold_take_hold_for_test`, in `take`), so no
    /// hold field exists while it is held. The late waiter, released, misses
    /// `old` and comes to that take. Returns once the successor's read is held
    /// and the late waiter's take is held — or the late waiter has joined a
    /// flight instead, which the joined case's order assertion reports and a
    /// waiter that reaches its take only after its flight read would do. GC
    /// passes under `GcPolicy::IMMEDIATE` run in that gap and keep `fresh`.
    struct CutFixedBeforeTheHold {
        tv: TestValkey,
        svc: Arc<LiveEditService>,
        store: Arc<ScriptedProbeStore>,
        seeded: Seeded,
        stream: SubscribeStream,
        take: ReleaseTake,
        successor_read: ReleaseRead,
        driver: tokio::task::JoinHandle<
            Result<
                live_edit::storage::snapshot_flights::FlightRead,
                live_edit::storage::snapshot_flights::FlightWaitError,
            >,
        >,
        fresh: Vec<u8>,
        newest: Vec<u8>,
    }

    async fn cut_fixed_before_the_hold(
        layer: &str,
        script: Vec<OnProbe>,
        seeds: (u8, u8, u8),
    ) -> CutFixedBeforeTheHold {
        let tv = TestValkey::setup().await;
        let fx = fixture(&tv, Shape::fenced(2, ROOMY_BUDGET_MS), script).await;
        let (mut svc, store) = (fx.svc, fx.store);
        let (old, fresh, newest) = (blob(seeds.0), blob(seeds.1), blob(seeds.2));
        let seeded = seed(&tv, &svc, layer, true, &old).await;
        let object_id = seeded.object_id;
        let mut parks = Parks::install(&mut svc);
        let take = ReleaseTake(svc.valkey.arm_read_hold_take_hold_for_test(object_id));
        let svc = Arc::new(svc);
        let stream =
            subscribe_and_stale_its_first_build(&tv, &svc, &parks, &seeded, &old, &fresh).await;

        let successor_read = ReleaseRead(injector(&svc.valkey).push_hold_at(next_fenced_read()));
        let ours = svc
            .snapshot_flights
            .attach_or_create(object_id, TEST_FLIGHT_SIZE);
        let driver = tokio::spawn(async move {
            let budget = CapacityBudget::new(DEADLINE);
            ours.read(&budget).await
        });
        parks.release_pre_ext_probe().await;

        tokio::time::timeout(DEADLINE, successor_read.0.parked())
            .await
            .expect("the successor's fence is granted and its read reaches its held command");
        until(
            "the late waiter's hold take is held back (or it joined a flight first)",
            || take.0.is_parked() || roles(&svc.metrics, "joined_read") > 0,
        )
        .await;
        let found = svc
            .snapshot_flights
            .lookup(object_id)
            .expect("the successor is registered");
        assert_eq!(
            (found.attachable, found.state),
            (true, FlightState::Reading),
            "the successor's cut is fixed, and it is still attachable"
        );
        assert_eq!(
            hold_fields_for(&tv, object_id).await,
            0,
            "the late waiter's hold is not taken yet"
        );
        gc_passes_keep(
            &tv,
            &fresh,
            Duration::ZERO,
            "the gap: the successor's cut references it, and no commit lands while its fence \
             is held",
        )
        .await;
        CutFixedBeforeTheHold {
            tv,
            svc,
            store,
            seeded,
            stream,
            take,
            successor_read,
            driver,
            fresh,
            newest,
        }
    }

    /// The decider's case. A successor flight that fixed its cut BEFORE the
    /// late waiter's hold was acknowledged, with the GC run in that gap, is
    /// joined, and its read still delivers every external value it
    /// references: no missing blob, no NOT_FOUND or INTERNAL to the
    /// subscriber. The order rule, observed while the late waiter's hold
    /// take is held back with nothing sent to `{gc}`: it has joined no flight
    /// (the first flight's creator and the successor's, the test's waiter,
    /// are the only flight roles counted). So it does not attach before its
    /// take — neither ahead of the take nor by reaching the take only after
    /// its flight read. Released, it takes its hold and joins the successor;
    /// the successor's read is let go, the waiter parks in its build from that
    /// snapshot (`probe_values` call 1), a commit releases `fresh` (which the
    /// joined cut references) and GC passes run: nothing is collected, and
    /// the payload at C2 carries `fresh`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_successor_whose_cut_was_fixed_before_the_late_waiters_hold_is_joined_and_delivers_every_blob_it_references(
    ) {
        within_test_budget(async {
            let mut case = cut_fixed_before_the_hold(
                "cut-fixed-before-the-hold-joined",
                vec![OnProbe::Pass, OnProbe::Park],
                (0xF4, 0xF5, 0xF6),
            )
            .await;
            let _release = ReleaseProbe(Arc::clone(&case.store));
            let (tv, svc) = (&case.tv, Arc::clone(&case.svc));
            let object_id = case.seeded.object_id;
            assert_eq!(
                (
                    roles(&svc.metrics, "leader"),
                    roles(&svc.metrics, "joined_read")
                ),
                (2, 0),
                "the order rule: while its hold take is held, nothing sent to {{gc}}, the late \
                 waiter has joined no flight (the two flights' creators are the first waiter and \
                 the test)"
            );
            assert!(
                case.take.0.is_parked(),
                "precondition: the late waiter's take is held back"
            );

            case.take.0.release();
            until("the late waiter joins the successor", || {
                roles(&svc.metrics, "joined_read") == 1
            })
            .await;
            assert_eq!(
                hold_fields_for(tv, object_id).await,
                1,
                "the late waiter attached holding its hold"
            );
            case.successor_read.0.release();
            let ours = tokio::time::timeout(DEADLINE, &mut case.driver)
                .await
                .expect("the successor's read resolves")
                .expect("the test's waiter did not panic")
                .expect("the test's waiter reads the successor");
            let new_cut = case.seeded.first_cut + 1;
            assert_eq!(ours.snapshot().cut, new_cut, "the successor's cut");

            parked_in_build(
                &case.store,
                &mut case.stream,
                "the build from the joined successor's snapshot",
            )
            .await;
            // Releases `fresh`, which the joined cut references.
            commit_ext(
                &svc,
                &case.seeded.session_id,
                &case.seeded.layer,
                object_id,
                false,
                &case.newest,
            )
            .await;
            gc_passes_keep(
                tv,
                &case.fresh,
                Duration::ZERO,
                "the hold, taken before the attach, keeps every blob the joined cut references",
            )
            .await;
            case.store.release();

            let (content, body) =
                next_content(&mut case.stream, "the joined read's catch-up").await;
            assert_eq!(
                (content.object_id, content.sequence_num),
                (object_id, new_cut),
                "served from the joined successor's cut"
            );
            assert!(
                carried(&body).contains(&case.fresh),
                "every external value the joined read references is delivered"
            );
            expect_caught_up_ack(&mut case.stream, object_id, new_cut).await;
            let (live, live_body) = next_content(&mut case.stream, "the release commit").await;
            assert_eq!(live.sequence_num, new_cut + 1);
            assert!(carried(&live_body).contains(&case.newest));
            assert_eq!(fenced_misses(&svc.metrics), (1, 0));
            assert_eq!(adapter_consults(&svc.metrics), 0);
            assert_eq!(hold_fields_for(tv, object_id).await, 0);
            drop(ours);
        })
        .await;
    }

    /// The decider's case, its other half. The same successor, cut fixed
    /// before the late waiter's hold; this time the successor's read is let go
    /// while the late waiter's hold take is still held back, nothing sent to
    /// `{gc}`, so the successor closes and releases its fence, a commit
    /// releases `fresh` (which the successor's cut references) and, with no
    /// hold live, the GC collects it. Only then is the take let go: the late
    /// waiter takes its hold and re-reads at a new flight's cut, past the
    /// release; its payload carries the newest value, never the collected one,
    /// and nothing ends NOT_FOUND or INTERNAL. A waiter that attached to the
    /// successor before its take — ahead of it, or by reaching it only after
    /// its flight read — would build from the successor's cut and miss
    /// `fresh`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_successor_that_closes_before_the_late_waiters_hold_is_not_joined_and_no_collected_blob_is_shipped(
    ) {
        within_test_budget(async {
            let mut case = cut_fixed_before_the_hold(
                "cut-fixed-before-the-hold-closed",
                Vec::new(),
                (0xF7, 0xF8, 0xF9),
            )
            .await;
            let (tv, svc) = (&case.tv, Arc::clone(&case.svc));
            let object_id = case.seeded.object_id;

            case.successor_read.0.release();
            let ours = tokio::time::timeout(DEADLINE, &mut case.driver)
                .await
                .expect("the successor's read resolves")
                .expect("the test's waiter did not panic")
                .expect("the test's waiter reads the successor");
            let successor_cut = case.seeded.first_cut + 1;
            assert_eq!(ours.snapshot().cut, successor_cut, "the successor's cut");
            drop(ours);
            tokio::time::timeout(DEADLINE, case.take.0.parked())
                .await
                .expect("the late waiter's hold take is held back");
            // Releases `fresh`, which the successor's cut references; no hold
            // is live, so the GC takes it.
            commit_ext(
                &svc,
                &case.seeded.session_id,
                &case.seeded.layer,
                object_id,
                false,
                &case.newest,
            )
            .await;
            collect(tv, &case.fresh).await;
            case.take.0.release();

            let (content, body) = next_content(&mut case.stream, "the late waiter's re-read").await;
            assert_eq!(
                (content.object_id, content.sequence_num),
                (object_id, successor_cut + 1),
                "the late waiter re-reads past the release, not at the closed successor's cut"
            );
            let parts = carried(&body);
            assert!(
                parts.contains(&case.newest) && !parts.contains(&case.fresh),
                "the payload carries the newest value and never the collected one"
            );
            expect_caught_up_ack(&mut case.stream, object_id, successor_cut + 1).await;
            assert_eq!(fenced_misses(&svc.metrics), (1, 0));
            assert_eq!(adapter_consults(&svc.metrics), 0);
            assert_eq!(hold_fields_for(tv, object_id).await, 0);
            assert_eq!(
                roles(&svc.metrics, "leader"),
                3,
                "the first flight, the test's successor, and a new flight for the re-read"
            );
        })
        .await;
    }

    // ── 7f: the genuine fenced miss and the re-read's other endings ─────────

    /// 7f, a genuine miss on the fenced path: `value` is deleted out of band,
    /// so the first flight's build misses it (`stale_retried`) and the
    /// successor's build misses it again under a hold the store confirms live
    /// (`genuine`, both under `read="shared_fenced"`), which takes
    /// fetch-on-miss through the checkpoint adapter. An adapter that holds the
    /// blob restores it and the catch-up delivers it; one that does not ends
    /// the catch-up INTERNAL, before content. The hold is released either way.
    async fn assert_genuine_fenced_miss(adapter_holds_it: bool, seed_byte: u8) {
        let tv = TestValkey::setup().await;
        let fx = fixture(&tv, Shape::fenced(2, ROOMY_BUDGET_MS), Vec::new()).await;
        let (svc, adapter) = (Arc::new(fx.svc), fx.adapter);
        let value = blob(seed_byte);
        let seeded = seed(
            &tv,
            &svc,
            &format!("genuine-fenced-{adapter_holds_it}"),
            true,
            &value,
        )
        .await;
        if adapter_holds_it {
            adapter
                .put_ext_blob(
                    &sha256_to_hex(&ext_content_hash(&value)),
                    Bytes::from(value.clone()),
                )
                .await
                .expect("seed the adapter");
        }
        wipe(&tv, &value).await;

        let mut stream = subscribe(&svc, &seeded.session_id, &seeded.layer, 0, 0).await;
        if adapter_holds_it {
            let (content, body) = next_content(&mut stream, "a recovered genuine miss").await;
            assert_eq!(
                (content.object_id, content.sequence_num),
                (seeded.object_id, seeded.first_cut)
            );
            assert!(carried(&body).contains(&value), "the adapter's copy");
            expect_caught_up_ack(&mut stream, seeded.object_id, seeded.first_cut).await;
            assert_eq!(adapter_result(&svc.metrics, "recovered"), 1);
            assert!(
                blob_stored(&tv, &value).await,
                "the recovery stored it again"
            );
        } else {
            let status = expect_ended_before_content(&mut stream, "an unrecoverable miss").await;
            assert_eq!(status.code(), Code::Internal, "{status:?}");
            assert_eq!(adapter_result(&svc.metrics, "adapter_miss"), 1);
        }
        assert_eq!(
            fenced_misses(&svc.metrics),
            (1, 1),
            "the first miss is retried as stale; the successor's is genuine"
        );
        assert_eq!(
            roles(&svc.metrics, "leader"),
            2,
            "the genuine check runs on the successor's snapshot"
        );
        assert_eq!(hold_fields_for(&tv, seeded.object_id).await, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_genuine_fenced_miss_under_a_live_hold_recovers_through_the_adapter() {
        within_test_budget(assert_genuine_fenced_miss(true, 0xD0)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_genuine_fenced_miss_the_adapter_cannot_restore_ends_internal() {
        within_test_budget(assert_genuine_fenced_miss(false, 0xCF)).await;
    }

    /// 7f, a lost hold (H-f's `read_hold_refused` exit). The `{gc}` read hold
    /// key deleted at the successor build's probe — a failover that lost the
    /// hold write — makes the genuine check's renewal answer LAPSED: the
    /// catch-up ends `read_hold_refused` (UNAVAILABLE before content, no
    /// refusal detail, never INTERNAL), counts nothing genuine and takes no
    /// fetch-on-miss; no hold field is left, and its stream registration is
    /// released with the subscription.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_fenced_re_attach_whose_hold_is_lost_ends_read_hold_refused() {
        within_test_budget(async {
            let tv = TestValkey::setup().await;
            let fx = fixture(
                &tv,
                Shape::fenced(2, ROOMY_BUDGET_MS),
                vec![OnProbe::Pass, OnProbe::DropReadHolds],
            )
            .await;
            let (svc, store) = (Arc::new(fx.svc), fx.store);
            let value = blob(0xD1);
            let seeded = seed(&tv, &svc, "lost-hold-fenced", true, &value).await;
            wipe(&tv, &value).await;
            let before = subscriptions(&svc.metrics);

            let mut stream = subscribe(&svc, &seeded.session_id, &seeded.layer, 0, 0).await;
            assert_read_hold_refused(&svc, &mut stream, "a lost hold", "shared_fenced").await;
            assert_eq!(
                store.calls.load(Ordering::SeqCst),
                2,
                "precondition: the hold was dropped at the successor build's probe"
            );
            assert_eq!(hold_fields_for(&tv, seeded.object_id).await, 0);
            drop(stream);
            until("the ended subscription's registration is released", || {
                subscriptions(&svc.metrics) == before
            })
            .await;
        })
        .await;
    }

    /// 7f, `NotFound` (H-f's re-read `NotFound` exit, checkpointing off). The
    /// layer overwritten between the first flight's snapshot and the
    /// re-attach: the successor's read answers NotFound for the old object,
    /// the recovery finds nothing to restore, and the catch-up re-bootstraps
    /// onto the new one before any content, with the hold released and one
    /// stream registration — the re-bootstrap's — left, released with the
    /// subscription.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_fenced_re_attach_whose_layer_was_overwritten_rebootstraps_on_the_new_object() {
        within_test_budget(async {
            let tv = TestValkey::setup().await;
            let mut svc = fixture(&tv, Shape::fenced(2, ROOMY_BUDGET_MS), Vec::new())
                .await
                .svc;
            let value = blob(0xD2);
            let seeded = seed(&tv, &svc, "overwritten-fenced", true, &value).await;
            let mut parks = Parks::install(&mut svc);
            let svc = Arc::new(svc);
            let before = subscriptions(&svc.metrics);

            let mut stream = subscribe(&svc, &seeded.session_id, &seeded.layer, 0, 0).await;
            parks.parked_pre_ext_probe().await;
            let new_object = create_layer(&svc, &seeded.session_id, &seeded.layer).await;
            assert_ne!(new_object, seeded.object_id, "precondition: an overwrite");
            svc.layer_store.wait_for_cleanup(DEADLINE).await;
            // The old object's blob goes too, so the parked build misses it.
            if blob_stored(&tv, &value).await {
                wipe(&tv, &value).await;
            }
            parks.release_pre_ext_probe().await;

            let (content, _) = next_content(&mut stream, "the re-bootstrap").await;
            assert_eq!(
                (content.object_id, content.sequence_num),
                (new_object, 1),
                "the catch-up re-bootstraps onto the new object"
            );
            expect_caught_up_ack(&mut stream, new_object, 1).await;
            assert_eq!(fenced_misses(&svc.metrics), (1, 0));
            assert_eq!(hold_fields_for(&tv, seeded.object_id).await, 0);
            assert_eq!(
                subscriptions(&svc.metrics),
                before + 1,
                "one registration: the re-bootstrap's"
            );
            drop(stream);
            until("the subscription's registration is released", || {
                subscriptions(&svc.metrics) == before
            })
            .await;
        })
        .await;
    }

    /// 7f, `NotFound` on the re-attach with checkpointing on: the layer's
    /// state lost between the first flight's snapshot and the re-attach (every
    /// `le:{oid}:obj:*` key gone, the mapping kept) makes the successor's read
    /// answer `NotFound`; once the hold is released (no field on the store when
    /// the restore lists the checkpoints) the layer recovery restores the
    /// layer, and the brand-new catch-up runs again and delivers the restored
    /// state from empty at the restored sequence — not `NOT_FOUND` "…could not
    /// be recovered". One stale fenced miss, nothing genuine, no fetch-on-miss.
    /// Fails on the pre-fix re-bootstrap without recovery (the section header
    /// of the atomic cases, `assert_reread_not_found_restores`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_fenced_re_attach_that_finds_the_state_lost_restores_it_and_delivers_it() {
        within_test_budget(async {
            let tv = TestValkey::setup().await;
            let mut svc = fixture(&tv, Shape::fenced(2, ROOMY_BUDGET_MS), Vec::new())
                .await
                .svc;
            let checkpoints = Arc::new(RestoreObservingStore::new(&tv, false));
            install_recovery(&mut svc, &checkpoints);
            let value = blob(0xD3);
            let seeded = seed(&tv, &svc, "restored-fenced", true, &value).await;
            let object_id = seeded.object_id;
            checkpoints
                .seed(
                    object_id,
                    &seeded.session_id,
                    &seeded.layer,
                    seeded.first_cut,
                )
                .await;
            let mut parks = Parks::install(&mut svc);
            let svc = Arc::new(svc);

            let mut stream = subscribe(&svc, &seeded.session_id, &seeded.layer, 0, 0).await;
            parks.parked_pre_ext_probe().await;
            lose_the_layers_state(&tv, object_id, &value).await;
            parks.release_pre_ext_probe().await;

            let (content, body) = next_content(&mut stream, "the restored layer").await;
            let restored_seq = crate::common::read_header_u64(&tv, object_id, "seq")
                .await
                .expect("the restore wrote the header");
            assert_eq!(
                (content.object_id, content.sequence_num),
                (object_id, restored_seq),
                "the restored layer is delivered, at its restored sequence"
            );
            assert_eq!(
                restored_field(&body),
                (Some(vec![0x5E, 0x57]), (true, 0)),
                "the brand-new re-run sends the restored state from empty"
            );
            expect_caught_up_ack(&mut stream, object_id, restored_seq).await;
            assert_eq!(
                checkpoints.holds_at_listing(),
                vec![0],
                "one restore attempt, run with no read hold field on the store"
            );
            assert_eq!(crate::common::recovery_metric(&svc.metrics, "restored"), 1);
            assert_eq!(fenced_misses(&svc.metrics), (1, 0));
            assert_eq!(adapter_consults(&svc.metrics), 0);
            assert_eq!(hold_fields_for(&tv, object_id).await, 0);
        })
        .await;
    }

    // ── H-f and 6f-e: the hold released on every exit of the re-attach ──────

    /// How a re-attach after a stale fenced miss ends, the hold taken.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Exit {
        /// The client goes away while the successor waits for the fenced read
        /// permit the test holds.
        ClientDisconnect,
        /// Graceful shutdown, in the same wait.
        Shutdown,
        /// The successor's shared read fails (a transient fault injected on
        /// its first fenced command).
        FlightReadFailure,
        /// The subscription's capacity budget ([`SHORT_BUDGET_MS`]) runs out
        /// while the successor waits for the held fenced permit.
        CapacityBudget,
        /// The successor sleeps out a staged writers' gap past
        /// `reader_wait_max(L)`.
        ReaderWaitCap,
        /// The new registration, at the successor's cut, reaches a tiny hard
        /// backlog cap while the build from the successor's snapshot is parked.
        BacklogCap,
        /// 6f-e: the successor's cut is below the first read's — the header's
        /// sequence rewound at the park.
        Regression,
    }

    /// H-f: a catch-up re-attaching after a stale fenced miss, ended by `exit`
    /// after taking its hold, ends before content by that exit's own status
    /// (a disconnect sees none) and its own stream-exit reason (the regression
    /// its `unverified` stable cut instead), releases its hold field (within
    /// [`RELEASED_WITHIN`], far
    /// inside the TTL), takes no fetch-on-miss, and leaves
    /// `live_edit_stream_mux_subscriptions` back at its pre-subscribe level:
    /// neither the swap's reservation nor either registration outlives it.
    async fn a_re_attach_ended_by(exit: Exit) {
        let tv = TestValkey::setup().await;
        let budget_ms = if exit == Exit::CapacityBudget {
            SHORT_BUDGET_MS
        } else {
            ROOMY_BUDGET_MS
        };
        let script = if exit == Exit::BacklogCap {
            vec![OnProbe::Pass, OnProbe::Park]
        } else {
            Vec::new()
        };
        let fx = fixture(&tv, Shape::fenced(2, budget_ms), script).await;
        let (mut svc, store) = (fx.svc, fx.store);
        let _release = ReleaseProbe(Arc::clone(&store));
        let (old, fresh) = (blob(0xF1), blob(0xF2));
        let seeded = seed(&tv, &svc, &format!("re-attach-{exit:?}"), true, &old).await;
        let object_id = seeded.object_id;
        let mut parks = Parks::install(&mut svc);
        let svc = Arc::new(svc);
        let before = subscriptions(&svc.metrics);
        let what = format!("{exit:?}");

        let mut stream = Some(subscribe(&svc, &seeded.session_id, &seeded.layer, 0, 0).await);
        parks.parked_pre_ext_probe().await;
        if exit == Exit::Regression {
            // The first read's cut is 3; the successor's will be 1.
            let _: () = tv
                .client()
                .pool
                .hset(
                    tv.key_schema().obj_header(&object_id.to_string()),
                    ("seq", "1"),
                )
                .await
                .expect("rewind the header sequence");
            wipe(&tv, &old).await;
        } else {
            commit_ext(
                &svc,
                &seeded.session_id,
                &seeded.layer,
                object_id,
                false,
                &fresh,
            )
            .await;
            collect(&tv, &old).await;
        }
        let mut permit = None;
        match exit {
            Exit::ClientDisconnect | Exit::Shutdown | Exit::CapacityBudget => {
                permit = Some(hold_fenced_permit(&svc.valkey).await)
            }
            Exit::FlightReadFailure => {
                injector(&svc.valkey).push_at(
                    next_fenced_read(),
                    fred::error::Error::new(
                        fred::error::ErrorKind::Timeout,
                        "injected fenced fan-out timeout",
                    ),
                );
            }
            Exit::ReaderWaitCap => stage_writers_gap(&tv, object_id, FAR_MS).await,
            Exit::BacklogCap => install_tiny_hard_cap(&svc),
            Exit::Regression => {}
        }
        parks.release_pre_ext_probe().await;

        if matches!(exit, Exit::ClientDisconnect | Exit::Shutdown) {
            until_hold_fields(&tv, object_id, 1, DEADLINE, &what).await;
            until(
                &format!("{what}: the successor waits for the held fenced permit"),
                || roles(&svc.metrics, "leader") == 2,
            )
            .await;
        }
        let status = match exit {
            Exit::ClientDisconnect => {
                drop(stream.take());
                until(
                    &format!("{what}: the ending is counted under client_disconnect"),
                    || exits(&svc.metrics, STREAM_EXIT_CLIENT_DISCONNECT) == 1,
                )
                .await;
                None
            }
            Exit::Shutdown => {
                svc.shutdown.cancel();
                Some(expect_ended_before_content(stream.as_mut().expect("subscribed"), &what).await)
            }
            Exit::ReaderWaitCap => {
                Some(ended_within(stream.as_mut().expect("subscribed"), CAP_BUDGET, &what).await)
            }
            Exit::BacklogCap => {
                parked_in_build(&store, stream.as_mut().expect("subscribed"), &what).await;
                for marker in 0..OVERFLOW_COMMITS {
                    commit_inline(&svc, &seeded.session_id, &seeded.layer, object_id, marker).await;
                }
                until(
                    &format!("{what}: the new registration's backlog hits the hard cap"),
                    || overflow_kills(&svc.metrics) == 1,
                )
                .await;
                store.release();
                Some(expect_ended_before_content(stream.as_mut().expect("subscribed"), &what).await)
            }
            Exit::FlightReadFailure | Exit::CapacityBudget | Exit::Regression => {
                Some(expect_ended_before_content(stream.as_mut().expect("subscribed"), &what).await)
            }
        };

        let counted = |reason: &str| exits(&svc.metrics, reason) == 1;
        match (exit, status) {
            (Exit::ClientDisconnect, None) => {}
            (Exit::Shutdown, Some(status)) => {
                assert_shutdown_status(&status);
                until(&format!("{what}: counted under shutdown"), || {
                    counted(STREAM_EXIT_SHUTDOWN)
                })
                .await;
            }
            (Exit::FlightReadFailure, Some(status)) => {
                assert_eq!(status.code(), Code::Unavailable, "{what}: {status:?}");
                until(&format!("{what}: counted under error"), || {
                    counted(STREAM_EXIT_ERROR)
                })
                .await;
            }
            (Exit::CapacityBudget, Some(status)) => {
                assert!(
                    status.message().contains(
                        "the shared snapshot read this catch-up joined was still waiting"
                    ),
                    "{what}: the flight's capacity refusal, got {status:?}"
                );
                assert_refusal_detail(
                    &status,
                    refusal_reason::CATCHUP_CAPACITY,
                    &[],
                    CAPACITY_RETRY_DELAY,
                );
                until(&format!("{what}: counted under catchup_capacity"), || {
                    counted(STREAM_EXIT_CATCHUP_CAPACITY)
                })
                .await;
            }
            (Exit::ReaderWaitCap, Some(status)) => {
                assert_eq!(status.code(), Code::Unavailable, "{what}: {status:?}");
                assert!(
                    status.message().contains("reader_wait_max"),
                    "{what}: the reader-wait cap's ending, got {status:?}"
                );
                assert_no_refusal_detail(&status);
                until(
                    &format!("{what}: counted under fenced_read_wait_cap"),
                    || counted(STREAM_EXIT_FENCED_READ_WAIT_CAP),
                )
                .await;
            }
            (Exit::BacklogCap, Some(status)) => {
                assert_eq!(status.code(), Code::Unavailable, "{what}: {status:?}");
                assert!(
                    status
                        .message()
                        .contains("fell behind its delivery watchdog"),
                    "{what}: the hard cap's ending, got {status:?}"
                );
                until(&format!("{what}: counted under inner_watchdog"), || {
                    counted(STREAM_EXIT_INNER_WATCHDOG)
                })
                .await;
            }
            (Exit::Regression, Some(status)) => {
                assert_eq!(status.code(), Code::Unavailable, "{what}: {status:?}");
                assert!(
                    status.message().contains("sequence moved backwards"),
                    "{what}: the sequence-regression ending, got {status:?}"
                );
                assert_eq!(
                    stable_cuts(&svc.metrics, "unverified", "brand_new"),
                    1,
                    "{what}: the regression records its unverified cut"
                );
            }
            (exit, status) => panic!("{exit:?}: unexpected ending {status:?}"),
        }

        until_hold_fields(&tv, object_id, 0, RELEASED_WITHIN, &what).await;
        assert_eq!(fenced_misses(&svc.metrics), (1, 0), "{what}");
        assert_eq!(adapter_consults(&svc.metrics), 0, "{what}");
        drop(permit);
        drop(stream);
        until(
            &format!("{what}: the subscriptions gauge is back at its pre-subscribe level"),
            || subscriptions(&svc.metrics) == before,
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_re_attach_ended_by_its_client_releases_its_hold_and_registrations() {
        within_test_budget(a_re_attach_ended_by(Exit::ClientDisconnect)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_re_attach_ended_by_shutdown_releases_its_hold_and_registrations() {
        within_test_budget(a_re_attach_ended_by(Exit::Shutdown)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_re_attach_whose_successor_read_fails_releases_its_hold_and_registrations() {
        within_test_budget(a_re_attach_ended_by(Exit::FlightReadFailure)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_re_attach_out_of_capacity_budget_releases_its_hold_and_registrations() {
        within_test_budget(a_re_attach_ended_by(Exit::CapacityBudget)).await;
    }

    /// Slow (about 27 s): the successor sleeps out the real
    /// `reader_wait_max(L)`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_re_attach_past_the_reader_wait_cap_releases_its_hold_and_registrations() {
        within_test_budget(a_re_attach_ended_by(Exit::ReaderWaitCap)).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_re_attach_whose_registration_hits_the_hard_backlog_cap_releases_its_hold_and_registrations(
    ) {
        within_test_budget(a_re_attach_ended_by(Exit::BacklogCap)).await;
    }

    /// 6f-e, the regression pin: the successor's cut below the first read's
    /// ends the catch-up `sequence_regressed` before content
    /// (`catch_up_regressed`, `read="shared_fenced"`, pinned at the first
    /// read's cut), never serving the lower cut.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_re_attach_whose_successor_cut_regressed_ends_unavailable_and_releases_its_hold() {
        within_test_budget(a_re_attach_ended_by(Exit::Regression)).await;
    }

    /// H-f, the hard backlog cap reached while the re-attach waits for its
    /// build turn (`take_build_turn`'s stream check), not while it builds.
    /// Two late waiters of one flight both miss and re-attach to one
    /// successor (as in 6f-b), each registered at its cut C2 under a
    /// [`TINY_HARD_CAP`]-entry hard cap. The replica's one catch-up slot is
    /// the successor's, so no extra build can start: whichever waiter gets
    /// the successor's turn parks in its build (`probe_values` call 2), and
    /// the other waits for the turn. [`OVERFLOW_COMMITS`] commits overflow
    /// both registrations. The waiter queued for the turn — the only one that
    /// can end while the build is parked — ends before content by the hard
    /// cap's ending, releases its hold (the parked builder's alone is left)
    /// and its registration at C2 (the parked builder's alone counts on
    /// `live_edit_stream_mux_subscriptions`). Released, the parked builder
    /// ends the same way at its delivery check; then no hold field is left
    /// and the gauge is back at its pre-subscribe level.
    ///
    /// The queued waiter is a second late waiter rather than a waiter the
    /// test drives itself: the test's waiter and the late waiter would ask
    /// for the turn as one shared read resolves, so which of them queued
    /// would be a race.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_re_attach_waiting_for_its_build_turn_whose_registration_hits_the_hard_backlog_cap_releases_its_hold_and_registration(
    ) {
        within_test_budget(async {
            let tv = TestValkey::setup().await;
            let fx = fixture(
                &tv,
                Shape::fenced(1, ROOMY_BUDGET_MS),
                vec![OnProbe::Pass, OnProbe::Pass, OnProbe::Park],
            )
            .await;
            let (mut svc, store) = (fx.svc, fx.store);
            let _release = ReleaseProbe(Arc::clone(&store));
            let (old, fresh) = (blob(0xFB), blob(0xFC));
            let seeded = seed(&tv, &svc, "turn-wait-hard-cap", true, &old).await;
            let object_id = seeded.object_id;
            let mut parks = Parks::install(&mut svc);
            let svc = Arc::new(svc);
            let before = subscriptions(&svc.metrics);

            let permit = hold_fenced_permit(&svc.valkey).await;
            let first = subscribe(&svc, &seeded.session_id, &seeded.layer, 0, 0).await;
            until("the first waiter creates the first flight", || {
                roles(&svc.metrics, "leader") == 1
            })
            .await;
            let second = subscribe(&svc, &seeded.session_id, &seeded.layer, 0, 0).await;
            until("the second waiter joins it", || {
                roles(&svc.metrics, "joined_read") == 1
            })
            .await;
            drop(permit);
            parks.parked_pre_ext_probe().await;
            commit_ext(
                &svc,
                &seeded.session_id,
                &seeded.layer,
                object_id,
                false,
                &fresh,
            )
            .await;
            collect(&tv, &old).await;
            let permit = hold_fenced_permit(&svc.valkey).await;
            parks.release_pre_ext_probe().await;
            until(
                "both late waiters attach to one successor, held open by the fenced permit",
                || roles(&svc.metrics, "leader") == 2 && roles(&svc.metrics, "joined_read") == 2,
            )
            .await;
            until_hold_fields(&tv, object_id, 2, DEADLINE, "each late waiter's own hold").await;
            install_tiny_hard_cap(&svc);
            drop(permit);

            tokio::time::timeout(DEADLINE, store.wait_parked())
                .await
                .expect("the successor's turn holder never parked in its build");
            for marker in 0..OVERFLOW_COMMITS {
                commit_inline(&svc, &seeded.session_id, &seeded.layer, object_id, marker).await;
            }
            until("both registrations at C2 hit the hard cap", || {
                overflow_kills(&svc.metrics) == 2
            })
            .await;

            let mut streams = [first, second];
            let (queued, ended) = {
                let [a, b] = &mut streams;
                tokio::time::timeout(DEADLINE, async {
                    tokio::select! {
                        item = a.next() => (0, item),
                        item = b.next() => (1, item),
                    }
                })
                .await
                .expect("the waiter queued for the turn did not end within the deadline")
            };
            let status = match ended {
                Some(Err(status)) => status,
                other => {
                    panic!("the waiter queued for the turn: expected an ending, got {other:?}")
                }
            };
            assert_eq!(status.code(), Code::Unavailable, "{status:?}");
            assert!(
                status
                    .message()
                    .contains("fell behind its delivery watchdog"),
                "the hard cap's ending, got {status:?}"
            );
            until_hold_fields(
                &tv,
                object_id,
                1,
                RELEASED_WITHIN,
                "the turn-wait ending releases its hold; the parked builder's is left",
            )
            .await;
            until_subscriptions(
                &svc.metrics,
                before + 1,
                "the turn-wait ending releases its registration at C2; the parked builder's \
                 alone counts",
            )
            .await;

            store.release();
            let parked = &mut streams[1 - queued];
            let status = expect_ended_before_content(parked, "the parked builder").await;
            assert!(
                status
                    .message()
                    .contains("fell behind its delivery watchdog"),
                "the parked builder ends at its delivery check: {status:?}"
            );
            until("both endings counted under inner_watchdog", || {
                exits(&svc.metrics, STREAM_EXIT_INNER_WATCHDOG) == 2
            })
            .await;
            until_hold_fields(&tv, object_id, 0, RELEASED_WITHIN, "both holds released").await;
            assert_eq!(fenced_misses(&svc.metrics), (2, 0));
            assert_eq!(adapter_consults(&svc.metrics), 0);
            drop(streams);
            until_subscriptions(&svc.metrics, before, "both late waiters gone").await;
        })
        .await;
    }

    // ── 6-swap: the registration swap before the re-read waits ─────────────

    /// 6-swap. A cut-based re-read swaps its stream registration for a
    /// reservation before it waits for anything, so no backlog queues behind
    /// the first read's cut while it waits: with a [`TINY_HARD_CAP`]-entry
    /// hard cap, the replica's only atomic read permit held by the test, and
    /// [`OVERFLOW_COMMITS`] commits landing during the re-read's permit wait,
    /// no registration is killed. The swap is observed before the commits:
    /// the hold is taken and no stream is registered
    /// (`live_edit_stream_mux_streams` = 0) — a registration kept at the first
    /// cut would still count, and would queue every commit. Released, the
    /// re-read registers at its own cut, past every commit, and serves them
    /// folded into its payload.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_re_read_waiting_for_its_permit_keeps_no_registration_at_the_first_cut() {
        within_test_budget(async {
            let tv = TestValkey::setup().await;
            let fx = fixture(&tv, Shape::atomic(None), vec![OnProbe::Park]).await;
            let (svc, store) = (Arc::new(fx.svc), fx.store);
            let _release = ReleaseProbe(Arc::clone(&store));
            install_tiny_hard_cap(&svc);
            let (old, fresh) = (blob(0xF3), blob(0xFA));
            let seeded = seed(&tv, &svc, "swap-before-the-wait", false, &old).await;
            let object_id = seeded.object_id;

            let mut stream = subscribe(&svc, &seeded.session_id, &seeded.layer, 0, 0).await;
            store.wait_parked().await;
            assert_eq!(
                streams(&svc.metrics),
                1,
                "precondition: registered at the first read's cut"
            );
            commit_ext(
                &svc,
                &seeded.session_id,
                &seeded.layer,
                object_id,
                false,
                &fresh,
            )
            .await; // seq 3
            collect(&tv, &old).await;
            let permit = svc
                .valkey
                .catchup_read_gate
                .acquire(
                    tokio::time::Instant::now() + Duration::from_secs(3_600),
                    SnapshotReadConsumer::Catchup,
                )
                .await
                .expect("the test takes the only atomic read permit");
            store.release();

            until_hold_fields(&tv, object_id, 1, DEADLINE, "the re-read's hold").await;
            until(
                "the swap releases the first read's registration before the re-read waits for \
                 its permit (a registration kept at the first cut would still count)",
                || streams(&svc.metrics) == 0,
            )
            .await;
            for marker in 0..OVERFLOW_COMMITS {
                commit_inline(&svc, &seeded.session_id, &seeded.layer, object_id, marker).await;
            }
            drop(permit);

            let last = seeded.first_cut + 1 + u64::from(OVERFLOW_COMMITS);
            let (content, body) = next_content(&mut stream, "the re-read's catch-up").await;
            assert_eq!(
                (content.object_id, content.sequence_num),
                (object_id, last),
                "the re-read's cut is past every commit made while it waited"
            );
            assert!(carried(&body).contains(&fresh));
            expect_caught_up_ack(&mut stream, object_id, last).await;
            assert_eq!(
                overflow_kills(&svc.metrics),
                0,
                "nothing queued behind the first cut while the re-read waited"
            );
            assert_eq!(atomic_misses(&svc.metrics), (1, 0));
            assert_eq!(hold_fields_for(&tv, object_id).await, 0);
        })
        .await;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The D-helper pin: every catch-up build goes through `build_catch_up`
// ═════════════════════════════════════════════════════════════════════════════
//
// A source scan, because the property is structural: a new build site that
// called a `diff_read` builder directly would skip the miss protocol — with
// no live read hold it can build only with the probe, and nothing would
// re-read on a stale miss — and only a test of that exact site would notice.
// The fetch-on-miss mode itself is guarded by its type (only a live hold's
// proof builds it); the scan pins the structure that type relies on, and the
// call chain, which no type shows. Each call site, or each occurrence of a
// name, is attributed to the top-level item it sits in — the column-0 `fn`,
// `mod` or `impl` that opened last, or `<word>` for any other column-0 item
// (`<use>`, `<struct>`, …; a `const` item is labelled by its name, `<FOO>`,
// because the `const ` prefix is stripped as for a `const fn`) — and comment
// lines are skipped. It is a scan of text: what it cannot see is said where
// it applies.

/// The `storage::diff_read` builders that read a snapshot's external values;
/// checked against the module's own signatures.
const BUILDERS: [&str; 4] = [
    "compute_diff_at_cut",
    "compute_faithful_coalesced_at_cut",
    "compute_faithful_full_content_at_cut",
    "compute_per_commit_diff_at_cut",
];

/// The catch-up module, relative to the crate root.
const SUBSCRIBE_RS: &str = "src/grpc/subscribe.rs";

/// The builders' own module, where [`BUILDERS`] is derived from.
const DIFF_READ_RS: &str = "src/storage/diff_read.rs";

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_source(relative: &str) -> String {
    std::fs::read_to_string(crate_root().join(relative))
        .unwrap_or_else(|e| panic!("read {relative}: {e}"))
}

/// The top-level item a column-0 line opens, if it opens one: a function's
/// name, `mod …` / `impl …` as written, or `<word>` for any other item a
/// column-0 identifier starts (`<use>`, `<static>`, `<struct>`, …; a `const`
/// item's word is its name, since `const ` is stripped first), so nothing at
/// column 0 is attributed to the function before it.
fn top_level_item(line: &str) -> Option<String> {
    if !line.starts_with(|c: char| c.is_ascii_alphabetic()) {
        return None;
    }
    let mut rest = line;
    for prefix in [
        "pub(crate) ",
        "pub(super) ",
        "pub ",
        "const ",
        "async ",
        "unsafe ",
    ] {
        rest = rest.strip_prefix(prefix).unwrap_or(rest);
    }
    if let Some(after) = rest.strip_prefix("fn ") {
        let name: String = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        return Some(name);
    }
    if rest.starts_with("mod ") || rest.starts_with("impl") {
        return Some(rest.trim_end_matches('{').trim().to_owned());
    }
    let word: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '!')
        .collect();
    Some(format!("<{word}>"))
}

/// Every occurrence of `needle` in `source` not on a comment line and not
/// preceded by an identifier character, as `(line number, enclosing
/// top-level item)`. With `skip_definitions`, an occurrence right after `fn`
/// (a definition) is left out.
fn occurrences(source: &str, needle: &str, skip_definitions: bool) -> Vec<(usize, String)> {
    let mut item = String::from("<file start>");
    let mut sites = Vec::new();
    for (index, line) in source.lines().enumerate() {
        if let Some(opened) = top_level_item(line) {
            item = opened;
        }
        if line.trim_start().starts_with("//") {
            continue;
        }
        let mut from = 0;
        while let Some(at) = line[from..].find(needle) {
            let start = from + at;
            let before = &line[..start];
            let identifier_continues = before
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
            let definition = skip_definitions && before.trim_end().ends_with("fn");
            if !identifier_continues && !definition {
                sites.push((index + 1, item.clone()));
            }
            from = start + needle.len();
        }
    }
    sites
}

/// Every call of `name(` in `source` — not its definition, not on a comment
/// line — as `(line number, enclosing top-level item)`.
fn call_sites(source: &str, name: &str) -> Vec<(usize, String)> {
    occurrences(source, &format!("{name}("), true)
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}")) {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// `external_values.rs`, where the live-hold proof is defined and built.
const EXTERNAL_VALUES_RS: &str = "src/storage/external_values.rs";

/// The read mode's type, exactly: a tuple struct whose one field is private.
const READ_MODE_TYPE: &str = "pub struct ExtValueReads(ExtReadMode);";

/// The private enum behind it, at column 0 with no visibility.
const READ_MODE_ENUM: &str = "enum ExtReadMode {";

/// The `#[derive(…)]` line directly above both, pinned: with `Default`
/// added, `Default::default()` would choose a mode without a constructor.
const READ_MODE_DERIVE: &str = "#[derive(Debug, Clone, Copy, PartialEq, Eq)]";

/// The fetch-on-miss mode's only constructors in `diff_read.rs`, as
/// `(signature line, gate line above it)`: one taking the live-hold proof,
/// and the test-support one, compiled out of a production build. Each one's
/// body is the single line [`RECOVER_BODY`].
const RECOVER_CONSTRUCTORS: [(&str, Option<&str>); 2] = [
    ("pub fn recover(_live: &LiveHoldProof<'_>) -> Self {", None),
    (
        "pub fn recover_for_test() -> Self {",
        Some("#[cfg(any(test, feature = \"test-support\"))]"),
    ),
];

/// The body of each of [`RECOVER_CONSTRUCTORS`].
const RECOVER_BODY: &str = "ExtValueReads(ExtReadMode::Recover)";

/// The live-hold proof's definition line; the line after it is its one
/// field, which must stay private.
const PROOF_TYPE: &str = "pub struct LiveHoldProof<'h> {";

/// The non-comment lines of `source`, as `(line number, line)`.
fn code_lines(source: &str) -> Vec<(usize, &str)> {
    source
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
        .map(|(index, line)| (index + 1, line))
        .collect()
}

/// The code line right above line `number` in `lines` (from
/// [`code_lines`]), trimmed.
fn code_line_above<'s>(lines: &[(usize, &'s str)], number: usize) -> &'s str {
    let at = lines
        .iter()
        .position(|(n, _)| *n == number)
        .expect("the line is a code line");
    assert!(at > 0, "line {number} has a code line above it");
    lines[at - 1].1.trim()
}

/// The nearest line at or above line `number` of `source` that declares a
/// function (`fn <name>(`), trimmed: the function an indented line sits in,
/// for a function with no nested `fn`.
fn enclosing_fn_line(source: &str, number: usize) -> String {
    let lines: Vec<&str> = source.lines().take(number).collect();
    lines
        .into_iter()
        .rev()
        .find(|line| {
            let code = line.trim_start();
            !code.starts_with("//") && (code.contains(" fn ") || code.starts_with("fn "))
        })
        .unwrap_or("<none>")
        .trim()
        .to_owned()
}

/// The top-level `async fn`s of `source` visible outside it — `pub`,
/// `pub(crate)` or `pub(super)` — whose signature takes `ExtValueReads`: the
/// builders whose caller chooses the probe read or fetch-on-miss. A signature
/// runs from its `fn` line to the line that opens the body.
fn ext_reading_builders(source: &str) -> std::collections::BTreeSet<String> {
    let mut found = std::collections::BTreeSet::new();
    let mut lines = source.lines();
    while let Some(line) = lines.next() {
        let Some(visible) = ["pub(crate) ", "pub(super) ", "pub "]
            .into_iter()
            .find_map(|prefix| line.strip_prefix(prefix))
        else {
            continue;
        };
        let Some(after) = visible.strip_prefix("async fn ") else {
            continue;
        };
        let name: String = after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let mut signature = line.to_owned();
        let mut last = line;
        while !last.trim_end().ends_with('{') {
            let Some(next) = lines.next() else { break };
            signature.push_str(next);
            last = next;
        }
        if signature.contains("ExtValueReads") {
            found.insert(name);
        }
    }
    found
}

/// Three rules over every file under `src/`, comment lines skipped:
///
/// 1. **The fetch-on-miss mode is built only from a live hold.** The compiler
///    enforces most of it: `ExtValueReads` is a struct whose mode field is
///    private, so outside `diff_read` the mode is chosen only through its
///    constructors, and `ExtValueReads::recover` takes a `LiveHoldProof`,
///    which only `ReadHold::confirm_live`'s live answer builds (its field is
///    private to `external_values`). This rule pins the structure the
///    compiler cannot, so that the type keeps meaning that:
///    - the type is exactly [`READ_MODE_TYPE`], its enum [`READ_MODE_ENUM`]
///      (no visibility, named nowhere else under `src/`), each derived
///      exactly [`READ_MODE_DERIVE`] — no `Default`;
///    - `ExtReadMode::Recover` is written in exactly three places: the body of
///      each of [`RECOVER_CONSTRUCTORS`] ([`RECOVER_BODY`], the test-support
///      one gated out of production), and `resolve_external_values`' match
///      arm — so a forged constructor (a `const`, a second `fn`, a `Default`)
///      fails here;
///    - the literal `LiveHoldProof {` is written once under `src/`, in
///      `confirm_live`, and the proof's field stays private — a second
///      construction fails here;
///    - `ExtValueReads::recover(` and `confirm_live(` are called only in
///      `subscribe.rs`'s `genuine_miss_rebuild` (and unit-test modules), and
///      `recover_for_test` is called nowhere under `src/`;
///    - non-vacuity: `ExtValueReads::PROBE` builds at `build_catch_up` and at
///      `reread_and_rebuild`.
///
///    What the type does not show — that a build probes first and that the
///    Recover build runs under the hold's renewal — is the call chain, pinned
///    by [`build_catch_up_once_is_reached_only_through_the_miss_protocol`] and
///    [`every_catch_up_regime_builds_through_build_catch_up`], and the
///    structure of `under_read_hold`.
/// 2. **[`BUILDERS`] is exactly the set of `diff_read` builders that take
///    the mode from a caller outside the module**: its `pub`, `pub(crate)`
///    and `pub(super)` `async fn`s whose signature names `ExtValueReads`.
/// 3. **Only `subscribe.rs`'s `build_catch_up_once` calls a builder** —
///    beside `diff_read`'s own test module and a builder's own body — and it
///    calls each one.
///
/// Not seen, because this is a scan of text: a macro that assembles an
/// identifier from pieces.
#[test]
fn the_read_mode_and_every_builder_call_stay_inside_the_miss_protocol() {
    let mut files = Vec::new();
    rust_sources(&crate_root().join("src"), &mut files);

    // Rule 1: the type and its private enum.
    let diff_read_source = read_source(DIFF_READ_RS);
    let diff_read_lines = code_lines(&diff_read_source);
    for (definition, at_column_0) in [(READ_MODE_TYPE, false), (READ_MODE_ENUM, true)] {
        let found: Vec<usize> = diff_read_lines
            .iter()
            .filter(|(_, line)| {
                if at_column_0 {
                    *line == definition
                } else {
                    line.trim() == definition
                }
            })
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(
            found.len(),
            1,
            "{DIFF_READ_RS} defines `{definition}` exactly once (a `pub` field, a `pub` enum or \
             a second definition would let code outside the module choose a mode): {found:?}"
        );
        assert_eq!(
            code_line_above(&diff_read_lines, found[0]),
            READ_MODE_DERIVE,
            "the derive above `{definition}` is pinned: a `Default` would choose a mode with \
             no constructor"
        );
    }
    for file in &files {
        if file.ends_with(DIFF_READ_RS) {
            continue;
        }
        let source = std::fs::read_to_string(file).expect("read a source file");
        let named = occurrences(&source, "ExtReadMode", false);
        assert!(
            named.is_empty(),
            "{}: the private `ExtReadMode` is named outside {DIFF_READ_RS}: {named:?}",
            file.display()
        );
    }

    // Rule 1: where the fetch-on-miss mode is written.
    let recover_sites: Vec<(usize, &str)> = diff_read_lines
        .iter()
        .filter(|(_, line)| line.contains("ExtReadMode::Recover"))
        .copied()
        .collect();
    assert_eq!(
        recover_sites.len(),
        3,
        "`ExtReadMode::Recover` is written in exactly three places — the two constructors' \
         bodies and `resolve_external_values`' match arm — so no other code can build the \
         fetch-on-miss mode: {recover_sites:#?}"
    );
    for (signature, gate) in RECOVER_CONSTRUCTORS {
        let (number, _) = diff_read_lines
            .iter()
            .find(|(_, line)| line.trim() == signature)
            .unwrap_or_else(|| panic!("{DIFF_READ_RS} defines `{signature}`"));
        let body = diff_read_lines
            .iter()
            .position(|(n, _)| n == number)
            .map(|at| diff_read_lines[at + 1].1.trim())
            .expect("a body line");
        assert_eq!(
            body, RECOVER_BODY,
            "`{signature}`'s body is the fetch-on-miss mode, alone"
        );
        if let Some(gate) = gate {
            assert_eq!(
                code_line_above(&diff_read_lines, *number),
                gate,
                "`{signature}` is compiled out of a production build"
            );
        }
    }
    let arm: Vec<(usize, String)> =
        occurrences(&diff_read_source, "ExtReadMode::Recover =>", false);
    assert_eq!(
        arm.iter()
            .map(|(_, item)| item.as_str())
            .collect::<Vec<_>>(),
        ["resolve_external_values"],
        "the third is `resolve_external_values`' match arm: {arm:?}"
    );

    // Rule 1: the live-hold proof is built once, in `confirm_live`.
    let mut proofs = Vec::new();
    for file in &files {
        let source = std::fs::read_to_string(file).expect("read a source file");
        for (line, _) in occurrences(&source, "LiveHoldProof {", false) {
            proofs.push((file.clone(), line, enclosing_fn_line(&source, line)));
        }
    }
    assert_eq!(
        proofs.len(),
        1,
        "the literal `LiveHoldProof {{` is written once under src/: {proofs:#?}"
    );
    let (proof_file, _, proof_fn) = &proofs[0];
    assert!(
        proof_file.ends_with(EXTERNAL_VALUES_RS)
            && proof_fn.starts_with("pub async fn confirm_live(&self)"),
        "the proof is built in `ReadHold::confirm_live`: {proofs:#?}"
    );
    let external_values = read_source(EXTERNAL_VALUES_RS);
    let external_values_lines = code_lines(&external_values);
    let definition = external_values_lines
        .iter()
        .position(|(_, line)| line.trim() == PROOF_TYPE)
        .unwrap_or_else(|| panic!("{EXTERNAL_VALUES_RS} defines `{PROOF_TYPE}`"));
    let field = external_values_lines[definition + 1].1.trim();
    assert!(
        field.starts_with("_hold:"),
        "the proof's one field stays private, so only its module builds one: `{field}`"
    );

    // Rule 1: who asks for the proof and the mode.
    let subscribe = read_source(SUBSCRIBE_RS);
    for name in ["ExtValueReads::recover", "confirm_live"] {
        let mut callers = Vec::new();
        for file in &files {
            let source = std::fs::read_to_string(file).expect("read a source file");
            for (line, item) in call_sites(&source, name) {
                if item != "mod tests" {
                    callers.push((file.clone(), line, item));
                }
            }
        }
        assert!(
            !callers.is_empty()
                && callers.iter().all(|(file, _, item)| {
                    file.ends_with(SUBSCRIBE_RS) && item == "genuine_miss_rebuild"
                }),
            "`{name}(` is called in production only by `genuine_miss_rebuild`: {callers:#?}"
        );
    }
    for file in &files {
        let source = std::fs::read_to_string(file).expect("read a source file");
        let calls = call_sites(&source, "recover_for_test");
        assert!(
            calls.is_empty(),
            "{}: `recover_for_test` is the integration tests' constructor, never called under \
             src/: {calls:?}",
            file.display()
        );
    }
    let probe_items: std::collections::BTreeSet<String> =
        occurrences(&subscribe, "ExtValueReads::PROBE", false)
            .into_iter()
            .map(|(_, item)| item)
            .collect();
    for item in ["build_catch_up", "reread_and_rebuild"] {
        assert!(
            probe_items.contains(item),
            "non-vacuity: `{item}` builds with `ExtValueReads::PROBE`: {probe_items:?}"
        );
    }

    // Rule 2.
    let derived = ext_reading_builders(&read_source(DIFF_READ_RS));
    let listed: std::collections::BTreeSet<String> =
        BUILDERS.iter().map(|b| (*b).to_owned()).collect();
    assert_eq!(
        derived, listed,
        "the visible `async fn`s of {DIFF_READ_RS} taking `ExtValueReads` must be exactly \
         BUILDERS; a new builder must be added there so this scan covers it"
    );

    // Rule 3.
    let diff_read = crate_root().join(DIFF_READ_RS);
    let mut inside = 0;
    let mut outside = Vec::new();
    for file in &files {
        let source = std::fs::read_to_string(file).expect("read a source file");
        let in_subscribe = file.ends_with(SUBSCRIBE_RS);
        let in_diff_read = *file == diff_read;
        for builder in BUILDERS {
            for (line, item) in call_sites(&source, builder) {
                if in_subscribe && item == "build_catch_up_once" {
                    inside += 1;
                } else if in_diff_read && (item == "mod tests" || item == builder) {
                    // `diff_read`'s own tests, or a builder's own body.
                } else {
                    outside.push(format!(
                        "{}:{line} ({item}) calls {builder}",
                        file.display()
                    ));
                }
            }
        }
    }
    assert!(
        outside.is_empty(),
        "a catch-up builder is called outside `build_catch_up_once`, so it builds without the \
         external-value miss protocol: {outside:#?}"
    );
    let subscribe = read_source(SUBSCRIBE_RS);
    for builder in BUILDERS {
        assert!(
            !call_sites(&subscribe, builder).is_empty(),
            "non-vacuity: `build_catch_up_once` calls {builder}"
        );
    }
    assert!(inside >= BUILDERS.len());
}

/// `build_catch_up_once` is called only from the miss protocol, and the
/// protocol's inner steps only from each other, down from `build_catch_up`:
/// nothing reaches a build around it.
#[test]
fn build_catch_up_once_is_reached_only_through_the_miss_protocol() {
    let subscribe = read_source(SUBSCRIBE_RS);
    let callers = |name: &str| -> std::collections::BTreeSet<String> {
        let sites = call_sites(&subscribe, name);
        assert!(!sites.is_empty(), "non-vacuity: {name} is called");
        sites.into_iter().map(|(_, item)| item).collect()
    };
    let expect = |name: &str, allowed: &[&str]| {
        let found = callers(name);
        let allowed: std::collections::BTreeSet<String> =
            allowed.iter().map(|s| (*s).to_owned()).collect();
        assert!(
            found.is_subset(&allowed),
            "{name} is called from {found:?}; only {allowed:?} may call it"
        );
    };
    expect(
        "build_catch_up_once",
        &[
            "build_catch_up",
            "reread_and_rebuild",
            "genuine_miss_rebuild",
        ],
    );
    expect("reread_and_rebuild", &["rebuild_and_check"]);
    expect("genuine_miss_rebuild", &["rebuild_and_check"]);
    expect("rebuild_and_check", &["rebuild_under_read_hold"]);
    expect("rebuild_under_read_hold", &["build_catch_up"]);
}

/// Every catch-up regime builds through `build_catch_up`: Regime 1 (and
/// every re-bootstrap, which enters it), Regime 2's snapshot arm, Regime 3,
/// and the rollback replay's per-commit build.
#[test]
fn every_catch_up_regime_builds_through_build_catch_up() {
    let subscribe = read_source(SUBSCRIBE_RS);
    let sites: std::collections::BTreeSet<String> = call_sites(&subscribe, "build_catch_up")
        .into_iter()
        .map(|(_, item)| item)
        .collect();
    let expected: std::collections::BTreeSet<String> = [
        "catch_up_brand_new",
        "catch_up_in_window_snapshot",
        "catch_up_out_of_window",
        "replay_commit_diff",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    assert_eq!(sites, expected);
}

// ═════════════════════════════════════════════════════════════════════════════
// The miss counter
// ═════════════════════════════════════════════════════════════════════════════

/// `live_edit_catchup_ext_value_miss_total` exports all four
/// `{outcome, read}` cells at zero before any miss, so a rate or `increase()`
/// alert sees the first one.
#[test]
fn the_miss_counter_is_zero_seeded_over_outcome_and_read() {
    let metrics = crate::common::metrics_scrape_only("b-yuo-miss");
    let text = render(&metrics);
    for outcome in ["stale_retried", "genuine"] {
        for read in ["atomic", "shared_fenced"] {
            assert_eq!(
                sample_value(
                    &text,
                    "live_edit_catchup_ext_value_miss_total",
                    &[("outcome", outcome), ("read", read)],
                ),
                Some(0.0),
                "{{outcome={outcome}, read={read}}} is exported at zero"
            );
        }
    }
}
