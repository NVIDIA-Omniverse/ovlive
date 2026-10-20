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

//! b.tbw part one — the per-delta size limits through the real `CreateLayer`
//! / `UpdateLayer` handlers: one delta carries at most 99,999 entries
//! (struct commands + set fields + time samples, summed;
//! `MAX_DELTA_ENTRIES`), and one `ReorderChildren` list at most 100,000 ids
//! (`MAX_CHILDREN_LIST_LEN`). A delta over either is refused
//! `INVALID_ARGUMENT` with a reason a client can act on — `DELTA_TOO_LARGE`
//! with the three counts and the limit, or `CHILDREN_LIST_TOO_LONG` with the
//! limit and the list's length — and no `RetryInfo`, having applied nothing.
//! Every refusal is decoded as the C++ client library decodes it
//! (`common::refusal_detail::assert_delta_too_large` /
//! `assert_children_list_too_long`): exactly one `ErrorInfo`, its `Any` type
//! URL ending `/google.rpc.ErrorInfo`, every metadata value an unsigned
//! decimal.
//!
//! | Test | Pins |
//! |---|---|
//! | [`create_layer_of_sixty_thousand_prims_and_sixty_thousand_fields_is_refused_delta_too_large_and_applies_nothing`] / [`update_layer_of_sixty_thousand_prims_and_sixty_thousand_fields_is_refused_delta_too_large_and_applies_nothing`] | T1: each kind under the old per-kind limit of 100,000, the sum over 99,999: `DELTA_TOO_LARGE` {60000, 60000, 0, 99999}; no layer mapped, the sequence and stream unchanged. Before b.tbw this was the verifier's opaque `INVALID_ARGUMENT` with no details |
//! | [`create_layer_at_exactly_the_entry_limit_commits_and_its_catch_up_returns_every_entry`] | T2: 99,999 entries mixed across the three kinds parse, validate, commit (a create at the limit, which under the flat command timeout timed out in an unoptimised build) and come back whole in the catch-up |
//! | [`one_entry_over_the_limit_and_one_hundred_thousand_set_fields_alone_are_refused_on_both_rpcs`] | T2: 100,000 entries mixed, and 100,000 set fields alone (which the old documentation allowed), are refused with the exact counts |
//! | [`children_list_one_past_the_limit_is_refused_children_list_too_long_on_both_rpcs`] | T4: a list of 100,001 ids, the domain bound: `CHILDREN_LIST_TOO_LONG` {100000, 100001} with the bound's message unchanged |
//! | [`children_list_past_the_parsers_cap_is_refused_children_list_too_long_on_both_rpcs`] | T4: a list of 1,000,001 ids, the parser's defensive cap, answered against the service's 100,000 limit |
//!
//! The chained-call precedence of these refusals over `INVALID_UPDATE` is in
//! `chained_writes_test` (Group 13).
//!
//! # Size and time
//!
//! The deltas are built in code, never committed as fixtures: a few MB each
//! (the 1,000,001-id list about 8 MB), sent in 1 MiB chunks, under the 10 MB
//! size cap `build_service` sets. The refused ones never reach the store. The
//! at-limit create writes about 300,000 commands in one transaction and reads
//! the layer back, 35–45 s unoptimised on a shared store.
//!
//! Its every step, the fixture's setup included, runs under one deadline,
//! [`AT_LIMIT_BUDGET`] (150 s) from its start. After it comes only
//! `TestValkey`'s drop:
//!
//! - its purge (`cleanup_keys`) runs `SCAN MATCH <prefix>*` with a page hint
//!   of 100 and `UNLINK`s the matches in batches of 100, each page and each
//!   `UNLINK` one round trip, and fails at the first step unanswered for 5 s.
//!   `SCAN` walks the whole shared keyspace, not just this test's keys, so
//!   the pages number about the database's key count / 100: no fixed bound.
//!   The `UNLINK`s number this test's keys / 100. The at-limit object
//!   (`queue_object_state`, plus the create's stream) is one hash per
//!   section, per field and per time sample (33,333 + 33,333 + 33,333 =
//!   99,999), plus the header, the sections index, the root's children list,
//!   the children index, the fields and time-samples indexes and the stream
//!   (7), so 99,999 + 7 = about 100,000 keys (plus the session's few): about
//!   1,000 `UNLINK`s;
//! - then it quits its client, bounded at 10 s.
//!
//! So its worst case is 150 + 5 + 10 = 165 s plus the purge's answered round
//! trips, under the 240 s its `.config/nextest.toml` override kills it at,
//! which leaves 75 s for those.

use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface, StreamsInterface};
use futures::StreamExt;
use tonic::{Request, Status};

use live_edit::domain::ompf::OmpfContainer;
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, CreateLayerRequest, CreateLayerResponse,
    SubscribeReadLayerRequest, UpdateLayerRequest, UpdateLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{
    CommandCreateSection, CommandReorderChildren, OwnedDelta, SetField, StructCommand, TimeSample,
};

use super::helpers::*;
use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::common::refusal_detail::{assert_children_list_too_long, assert_delta_too_large};

/// The layer every test writes.
const LAYER: &str = "size-limit-layer";

/// The chunk each body is sent in: well under tonic's 4 MiB message limit.
const CHUNK_BYTES: usize = 1 << 20;

/// The most entries one delta carries, as a count.
const MAX_ENTRIES: usize = 99_999;

/// The service's limit on one children list.
const MAX_CHILDREN: usize = 100_000;

/// The parser's defensive cap on one children list.
const PARSER_CHILDREN_CAP: usize = 1_000_000;

/// The at-limit test's one deadline, from its first line to its last
/// assertion: far past its healthy 35–45 s (see the module's "Size and
/// time").
const AT_LIMIT_BUDGET: std::time::Duration = std::time::Duration::from_secs(150);

/// A delta of `prims` sections created under the root (ids 1..=prims), then
/// `fields` one-byte inline fields and `samples` one-byte inline time samples
/// on section 1 — legal content of each kind, so an at-limit delta commits.
fn entries_delta(prims: usize, fields: usize, samples: usize) -> OwnedDelta {
    OwnedDelta {
        struct_commands: (1..=prims as u64)
            .map(|id| {
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 0,
                    section_id: id,
                    section_name: format!("s{id}"),
                    section_type: 1,
                })
            })
            .collect(),
        set_fields: (0..fields)
            .map(|i| SetField {
                section_id: 1,
                key_name: format!("k{i}"),
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            })
            .collect(),
        time_samples: (0..samples)
            .map(|i| TimeSample {
                section_id: 1,
                time: i as f64,
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
            })
            .collect(),
        ..Default::default()
    }
}

/// A delta whose one `ReorderChildren` command lists `n` ids under section 1.
/// The size checks run before the delta is applied, so the ids need not
/// exist.
fn reorder_delta(n: usize) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::ReorderChildren(CommandReorderChildren {
            section_id: 1,
            children_list_id: 0,
            children_list: (1..=n as u64).collect(),
        })],
        ..Default::default()
    }
}

/// `delta`'s OMPF body in [`CHUNK_BYTES`] pieces.
fn body_chunks(delta: &OwnedDelta) -> Vec<Vec<u8>> {
    delta_to_ompf_bytes(delta, &[])
        .chunks(CHUNK_BYTES)
        .map(<[u8]>::to_vec)
        .collect()
}

/// A session with no layer yet.
async fn new_session(tv: &TestValkey, name: &str) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, "/stage", "test-user")
        .await
        .expect("session creation must succeed")
}

/// `delta` as one `CreateLayer` of [`LAYER`].
async fn create(
    svc: &LiveEditService,
    session_id: &str,
    delta: &OwnedDelta,
) -> Result<CreateLayerResponse, Status> {
    let messages: Vec<CreateLayerRequest> = std::iter::once(make_create_header(session_id, LAYER))
        .chain(body_chunks(delta).into_iter().map(make_create_data))
        .collect();
    handle_create_layer(svc, make_request(streaming_create(messages)))
        .await
        .map(tonic::Response::into_inner)
}

/// `delta` as one `UpdateLayer` of [`LAYER`].
async fn update(
    svc: &LiveEditService,
    session_id: &str,
    object_id: u64,
    delta: &OwnedDelta,
) -> Result<UpdateLayerResponse, Status> {
    let messages: Vec<UpdateLayerRequest> =
        std::iter::once(make_update_header(session_id, LAYER, object_id))
            .chain(body_chunks(delta).into_iter().map(make_update_data))
            .collect();
    handle_update_layer(svc, make_request(streaming_update(messages)))
        .await
        .map(tonic::Response::into_inner)
}

/// Whether [`LAYER`] is mapped to an object in `session_id`.
async fn mapped(tv: &TestValkey, session_id: &str) -> bool {
    let exists: u64 = tv
        .client()
        .pool
        .next()
        .exists(tv.key_schema().layer_mapping(session_id, LAYER))
        .await
        .expect("EXISTS layer mapping");
    exists > 0
}

/// `(sequence number, stream length)` of `object_id`.
async fn seq_and_stream(tv: &TestValkey, object_id: u64) -> (u64, u64) {
    let client = tv.client().pool.next();
    let seq: Option<String> = client
        .hget(tv.key_schema().obj_header(&object_id.to_string()), "seq")
        .await
        .expect("HGET seq");
    let stream: u64 = client
        .xlen(tv.key_schema().layer_stream(object_id))
        .await
        .expect("XLEN layer stream");
    (
        seq.expect("the object has a header")
            .parse()
            .expect("seq is an integer"),
        stream,
    )
}

/// A session with [`LAYER`] created empty, at sequence 1.
async fn session_with_layer(tv: &TestValkey, svc: &LiveEditService, name: &str) -> (String, u64) {
    create_session_and_layer(tv, svc, name, LAYER).await
}

// ═════════════════════════════════════════════════════════════════════════════
// T1 — each kind under the old per-kind limit, the sum over the new one
// ═════════════════════════════════════════════════════════════════════════════

/// 60,000 prims and 60,000 fields as one `CreateLayer`: `DELTA_TOO_LARGE`
/// naming both counts and the limit, and no layer exists afterwards.
///
/// Fails before b.tbw: the parser's verifier refused the buffer's 120,001
/// tables, answered `INVALID_ARGUMENT "OODF error: FlatBuffers verifier
/// failed: …"` with no status details, which `assert_delta_too_large`'s
/// message and `ErrorInfo` checks reject.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_of_sixty_thousand_prims_and_sixty_thousand_fields_is_refused_delta_too_large_and_applies_nothing(
) {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = new_session(&tv, "t1-create").await;

    let status = create(&svc, &session_id, &entries_delta(60_000, 60_000, 0))
        .await
        .expect_err("120,000 entries must be refused");

    assert_delta_too_large(&status, 60_000, 60_000, 0);
    assert!(!mapped(&tv, &session_id).await, "no layer was created");
}

/// The same delta as an `UpdateLayer` of an existing layer: the same refusal,
/// and the layer's sequence number and stream are unchanged.
///
/// Fails before b.tbw, as the `CreateLayer` twin does.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_of_sixty_thousand_prims_and_sixty_thousand_fields_is_refused_delta_too_large_and_applies_nothing(
) {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = session_with_layer(&tv, &svc, "t1-update").await;
    let before = seq_and_stream(&tv, object_id).await;

    let status = update(
        &svc,
        &session_id,
        object_id,
        &entries_delta(60_000, 60_000, 0),
    )
    .await
    .expect_err("120,000 entries must be refused");

    assert_delta_too_large(&status, 60_000, 60_000, 0);
    assert_eq!(
        seq_and_stream(&tv, object_id).await,
        before,
        "nothing applied: the sequence number and the stream are unchanged"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T2 — the boundary
// ═════════════════════════════════════════════════════════════════════════════

/// The catch-up of [`LAYER`], read whole through `SubscribeReadLayer` as a
/// new subscriber (object id 0), and decoded.
async fn catch_up(svc: &LiveEditService, session_id: &str) -> (u64, OwnedDelta) {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: LAYER.to_owned(),
        object_id: 0,
        sequence_num: 0,
        client_id: "size-limit-reader".to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    let mut stream = handle_subscribe_read_layer(svc, request)
        .await
        .expect("the subscription must start")
        .into_inner();
    let header = match stream.next().await {
        Some(Ok(message)) => match message.response_type {
            Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
            other => panic!("expected the catch-up's header, got {other:?}"),
        },
        other => panic!("expected the catch-up's header, got {other:?}"),
    };
    let mut body = Vec::with_capacity(header.total_size as usize);
    while (body.len() as u64) < header.total_size {
        match stream.next().await {
            Some(Ok(message)) => match message.response_type {
                Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                    body.extend_from_slice(&c.data)
                }
                other => panic!("expected a catch-up chunk, got {other:?}"),
            },
            other => panic!("the catch-up ended early: {other:?}"),
        }
    }
    let container = OmpfContainer::parse(&body).expect("the catch-up is an OMPF");
    let delta = oodf_ffi::parse_delta(container.delta_bytes)
        .expect("a catch-up of exactly the limit is one delta the parser accepts");
    (header.sequence_num, delta)
}

/// Exactly 99,999 entries, a third of each kind, as one `CreateLayer`: the
/// parse, the bounds check and the commit all accept it, and a new
/// subscriber's catch-up returns every prim, field and time sample.
///
/// The create's one transaction queues about 300,000 commands; under the flat
/// 5 s command timeout its `EXEC` reply could outlive the timeout in an
/// unoptimised build (`UNAVAILABLE "EXEC object write failed (Timeout
/// Error)"`), where its size-scaled timeout now holds it.
///
/// Every step, the fixture's setup included, runs under one deadline,
/// [`AT_LIMIT_BUDGET`] from the test's start (see the module's "Size and
/// time").
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_at_exactly_the_entry_limit_commits_and_its_catch_up_returns_every_entry() {
    let deadline = tokio::time::Instant::now() + AT_LIMIT_BUDGET;
    let tv = tokio::time::timeout_at(deadline, TestValkey::setup())
        .await
        .expect("the fixture did not set up within the test's budget");
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let third = MAX_ENTRIES / 3;
    assert_eq!(3 * third, MAX_ENTRIES);

    let (created, mapped_after, (sequence_num, delta)) = tokio::time::timeout_at(deadline, async {
        let session_id = new_session(&tv, "t2-at-limit").await;
        let created = create(&svc, &session_id, &entries_delta(third, third, third))
            .await
            .unwrap_or_else(|s| panic!("exactly {MAX_ENTRIES} entries must commit; got {s:?}"));
        let mapped_after = mapped(&tv, &session_id).await;
        (created, mapped_after, catch_up(&svc, &session_id).await)
    })
    .await
    .expect("the create and its catch-up did not finish within the test's budget");

    assert!(created.object_id > 0);
    assert!(mapped_after, "the layer is mapped");
    assert_eq!(sequence_num, 1, "the created layer's first version");
    assert_eq!(
        (
            delta.struct_commands.len(),
            delta.set_fields.len(),
            delta.time_samples.len()
        ),
        (third, third, third),
        "the catch-up returns every prim, field and time sample"
    );
}

/// One entry over the limit, mixed across the kinds, and 100,000 set fields
/// alone — a delta the old documentation (100,000 of each kind) allowed —
/// are refused on both RPCs with their exact counts; nothing is applied.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_entry_over_the_limit_and_one_hundred_thousand_set_fields_alone_are_refused_on_both_rpcs(
) {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = session_with_layer(&tv, &svc, "t2-over").await;
    let fresh = new_session(&tv, "t2-over-create").await;
    let before = seq_and_stream(&tv, object_id).await;
    let third = MAX_ENTRIES / 3;

    for (prims, fields, samples) in [(third + 1, third, third), (0, MAX_ENTRIES + 1, 0)] {
        let delta = entries_delta(prims, fields, samples);
        let counts = (prims as u64, fields as u64, samples as u64);

        let status = create(&svc, &fresh, &delta)
            .await
            .expect_err("one entry over the limit must be refused by CreateLayer");
        assert_delta_too_large(&status, counts.0, counts.1, counts.2);

        let status = update(&svc, &session_id, object_id, &delta)
            .await
            .expect_err("one entry over the limit must be refused by UpdateLayer");
        assert_delta_too_large(&status, counts.0, counts.1, counts.2);
    }
    assert!(!mapped(&tv, &fresh).await, "no layer was created");
    assert_eq!(
        seq_and_stream(&tv, object_id).await,
        before,
        "nothing applied"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T4 — CHILDREN_LIST_TOO_LONG, unchained
// ═════════════════════════════════════════════════════════════════════════════

/// A `ReorderChildren` of 100,001 ids — one over the service's limit — is
/// refused `CHILDREN_LIST_TOO_LONG` {limit 100000, actual 100001} on both RPCs,
/// with the domain bound's message as it was before the refusal carried a
/// reason. Nothing is applied.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn children_list_one_past_the_limit_is_refused_children_list_too_long_on_both_rpcs() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = session_with_layer(&tv, &svc, "t4-domain").await;
    let fresh = new_session(&tv, "t4-domain-create").await;
    let before = seq_and_stream(&tv, object_id).await;
    let delta = reorder_delta(MAX_CHILDREN + 1);

    let status = create(&svc, &fresh, &delta)
        .await
        .expect_err("CreateLayer must refuse a list of 100,001 ids");
    assert_children_list_too_long(&status, MAX_CHILDREN as u64 + 1);

    let status = update(&svc, &session_id, object_id, &delta)
        .await
        .expect_err("UpdateLayer must refuse a list of 100,001 ids");
    assert_children_list_too_long(&status, MAX_CHILDREN as u64 + 1);

    assert!(!mapped(&tv, &fresh).await, "no layer was created");
    assert_eq!(
        seq_and_stream(&tv, object_id).await,
        before,
        "nothing applied"
    );
}

/// A `ReorderChildren` of 1,000,001 ids — past the parser's own defensive cap
/// of 1,000,000, which refuses it before the domain sees it — is answered
/// exactly as the domain answers an over-long list: `CHILDREN_LIST_TOO_LONG`
/// against the service's 100,000 limit, with the list's real length. Before
/// b.tbw the parser refused it `INVALID_ARGUMENT` naming its 1,000,000 cap,
/// with no details.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn children_list_past_the_parsers_cap_is_refused_children_list_too_long_on_both_rpcs() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = session_with_layer(&tv, &svc, "t4-parser").await;
    let fresh = new_session(&tv, "t4-parser-create").await;
    let before = seq_and_stream(&tv, object_id).await;
    let delta = reorder_delta(PARSER_CHILDREN_CAP + 1);

    let status = create(&svc, &fresh, &delta)
        .await
        .expect_err("CreateLayer must refuse a list of 1,000,001 ids");
    assert_children_list_too_long(&status, PARSER_CHILDREN_CAP as u64 + 1);

    let status = update(&svc, &session_id, object_id, &delta)
        .await
        .expect_err("UpdateLayer must refuse a list of 1,000,001 ids");
    assert_children_list_too_long(&status, PARSER_CHILDREN_CAP as u64 + 1);

    assert!(!mapped(&tv, &fresh).await, "no layer was created");
    assert_eq!(
        seq_and_stream(&tv, object_id).await,
        before,
        "nothing applied"
    );
}
