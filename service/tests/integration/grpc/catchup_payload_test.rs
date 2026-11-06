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

//! b.ntm — a catch-up's payload, end to end through `handle_subscribe_read_layer`
//! and a real `ValkeyExternalValueStore` (SDD §11.2, §12.3, §12.6):
//!
//! - **The reads** (T2): a catch-up of a layer carrying external values sends
//!   exactly the container `OmpfContainer::serialize` makes of its delta and
//!   values, in 1 MiB chunks, having read each value only with `STRLEN`
//!   (`EXISTS` for one that reads 0) and `GETRANGE`s of at most 65,472 B that
//!   cover it once — never a whole-value `GET` — as the store's `MONITOR`
//!   feed records. A value stored at another length than its delta declares,
//!   which `LIVE_EDIT_EXT_VALUE_VERIFY=observe` admits, is served as stored
//!   (T2b).
//! - **The catch-up payload budget `D_c`** (T3, T4): a catch-up reserves its
//!   payload from the replica's budget after its values' lengths are probed
//!   and before their first byte is read. With the budget held by another, a
//!   catch-up is refused `catchup_capacity` before any content, having read
//!   no value byte; one waiting for it ends at shutdown's close of the budget
//!   and at its client's disconnect; and what a catch-up holds returns to the
//!   budget page by page as its chunks are taken, and all at once when its
//!   client goes away.
//! - **Stalled holders** (T10, P5a): while a catch-up waits for the budget, a
//!   subscription whose response queue holds catch-up pages and has had
//!   nothing taken for the outer send deadline is ended
//!   `slow_reader_deadline`, its queue dropped, and the waiter granted; with
//!   no catch-up waiting, such a subscription is left alone.
//!
//! The layout, the ranges' bounds and window, the miss rules and the page
//! grants are pinned against a scripted store in `storage::ompf_payload`'s
//! unit tests; the stalled-holder rule's clock in
//! `storage::stream_mux::delivery_ledger`'s.
//!
//! **Wait budget.** Every wait is a deadline, not a measurement: `SETTLE`
//! (10 s) per awaited message or polled condition, `DRAIN` (20 s) per
//! catch-up drained. Paid in full, once each: the TestValkey capacity budget
//! (2 s), which the refused catch-up runs out, and `QUIET` (four outer send
//! deadlines, 1.2 s), the window in which an unchallenged holder of pages
//! must not be ended. The heaviest test, the stalled holder's, is about 70 s
//! at every budget's full value; the healthy path takes a few hundred
//! milliseconds a test plus what it pays.
//!
//! Tests that shrink the outer send deadline (`outer_send_deadline_for_test`)
//! exist only under the `test-fault-injection` feature and are `#[cfg]`-gated
//! on it.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::KeysInterface;
use futures::{Stream, StreamExt};
use tonic::{Code, Request, Status};

use live_edit::config::ExtValueVerifyMode;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::error::{refusal_reason, CAPACITY_RETRY_DELAY};
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, SubscribeReadLayerHeader, SubscribeReadLayerRequest,
    SubscribeReadLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
#[cfg(feature = "test-fault-injection")]
use live_edit::observability::metrics::STREAM_EXIT_SLOW_READER_DEADLINE;
use live_edit::observability::metrics::{
    STREAM_EXIT_CATCHUP_CAPACITY, STREAM_EXIT_CLIENT_DISCONNECT, STREAM_EXIT_SHUTDOWN,
};
use live_edit::storage::external_values::{ext_content_hash, ValkeyExternalValueStore};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::memory_budget::{MemoryBudget, CATCHUP_PAYLOAD_SERIES};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};

use super::helpers::*;
use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::common::refusal_detail::{assert_refusal_detail, assert_shutdown_status};
use crate::storage::snapshot_read_support::{
    monitor_fence, test_valkey_addr, MonitorTap, MonitoredCommand,
};

/// Bound on one awaited message or polled condition; unreachable on the
/// healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// One budget over a whole catch-up drain, whatever its message count.
const DRAIN: Duration = Duration::from_secs(20);

/// The section every field here lives in.
const SECTION_ID: u64 = 7;

/// The largest range a catch-up may ask for: 64 B under the connection's
/// counted 64 KiB read buffer, so no reply frame outgrows it (SDD §15.3). The
/// decided bound, written out on purpose rather than read from
/// `ext_backup::EXT_VALUE_READ_CHUNK_BYTES`, so a change to that constant
/// fails the T2 test that checks it rather than moving the bound silently.
const RANGE_MAX_BYTES: u64 = 65_472;

/// A chunk's and a payload page's size.
const PAGE_BYTES: usize = 1 << 20;

/// The refusal a catch-up whose payload memory is held ends with.
const CATCHUP_PAYLOAD_MSG: &str = "this replica's catch-up payload memory, sized from its pod \
     memory limit, is in use by other catch-ups; retry";

type SubStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>;

// ── Fixture ─────────────────────────────────────────────────────────────────

/// A service over `tv` with a real external-value store checking uploads in
/// `mode`.
fn service(tv: &TestValkey, mode: ExtValueVerifyMode) -> LiveEditService {
    let ext_value_store =
        ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_ext_value_verify(mode);
    LiveEditService {
        ext_value_store: Arc::new(ext_value_store),
        ..build_service(tv, Arc::new(AllowAllPermissions))
    }
}

/// Replace `svc`'s catch-up payload budget with one of `total` bytes, on its
/// own series: the test seam a test sizes `D_c` to its payloads with.
fn with_payload_budget(svc: &mut LiveEditService, total: u64) -> Arc<MemoryBudget> {
    let budget = Arc::new(MemoryBudget::with_series(
        total / 1024,
        Arc::new(Mutex::new(None)),
        &CATCHUP_PAYLOAD_SERIES,
    ));
    svc.valkey.catchup_payload_budget = Arc::clone(&budget);
    budget
}

/// A deterministic value of `len` bytes, distinct per `seed`.
fn value(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(29).wrapping_add(seed))
        .collect()
}

fn ext_field(key: &str, claim: &[u8; 32], size: usize, index: u32) -> SetField {
    SetField {
        section_id: SECTION_ID,
        key_name: key.to_owned(),
        value_or_ext_hash: claim.to_vec(),
        ext_value_size: size as u64,
        ext_value_index: index,
        set_order: u64::from(index),
    }
}

/// A session and a layer holding [`SECTION_ID`], then one commit setting a
/// field per entry of `parts` to that external value, each declared at its
/// `declared` size. Returns `(session_id, object_id)`.
async fn layer_with_external_values(
    tv: &TestValkey,
    svc: &LiveEditService,
    layer: &str,
    parts: &[(Vec<u8>, usize)],
) -> (String, u64) {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(&format!("{layer}-session"), "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let create = OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: SECTION_ID,
            section_name: "Asset".to_owned(),
            section_type: 1,
        })],
        ..Default::default()
    };
    let stream = streaming_create(vec![
        make_create_header(&session_id, layer),
        make_create_data(delta_to_ompf_bytes(&create, &[])),
    ]);
    let object_id = Box::pin(handle_create_layer(svc, make_request(stream)))
        .await
        .expect("the setup CreateLayer must succeed")
        .into_inner()
        .object_id;
    let fields = parts
        .iter()
        .enumerate()
        .map(|(i, (part, declared))| {
            ext_field(
                &format!("v{i}"),
                &ext_content_hash(part),
                *declared,
                i as u32 + 1,
            )
        })
        .collect();
    let delta = OwnedDelta {
        set_fields: fields,
        ..Default::default()
    };
    let refs: Vec<&[u8]> = parts.iter().map(|(p, _)| p.as_slice()).collect();
    let stream = streaming_update(vec![
        make_update_header(&session_id, layer, object_id),
        make_update_data(delta_to_ompf_bytes(&delta, &refs)),
    ]);
    Box::pin(handle_update_layer(svc, make_request(stream)))
        .await
        .expect("the setup UpdateLayer must succeed");
    (session_id, object_id)
}

/// A brand-new subscription to `layer`.
async fn subscribe(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    client: &str,
) -> SubStream {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer.to_owned(),
        object_id: 0,
        sequence_num: 0,
        client_id: client.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    Box::pin(handle_subscribe_read_layer(svc, request))
        .await
        .expect("the subscription must be admitted")
        .into_inner()
}

async fn next_item(
    stream: &mut SubStream,
    what: &str,
) -> Result<SubscribeReadLayerResponse, Status> {
    tokio::time::timeout(SETTLE, stream.next())
        .await
        .unwrap_or_else(|_| panic!("{what} did not arrive within {SETTLE:?}"))
        .unwrap_or_else(|| panic!("the stream ended before {what}"))
}

/// A catch-up drained: its content header, its chunks' messages (held, so a
/// test decides when their pages are released), and its caught-up ack.
struct CatchUp {
    header: SubscribeReadLayerHeader,
    chunks: Vec<SubscribeReadLayerResponse>,
    ack: SubscribeReadLayerHeader,
}

impl CatchUp {
    fn chunk_data(&self) -> Vec<&Bytes> {
        self.chunks
            .iter()
            .map(|m| match &m.response_type {
                Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => &c.data,
                other => panic!("a chunk message: {other:?}"),
            })
            .collect()
    }

    fn payload(&self) -> Vec<u8> {
        self.chunk_data()
            .into_iter()
            .flat_map(|d| d.iter().copied())
            .collect()
    }
}

/// Drain a catch-up of one content message: the header, its chunks, the ack.
async fn drain_catch_up(stream: &mut SubStream) -> CatchUp {
    tokio::time::timeout(DRAIN, async {
        let header = header_of(next_item(stream, "the content header").await);
        assert!(header.total_size > 0, "the catch-up sends content");
        let mut chunks = Vec::new();
        let mut seen = 0u64;
        while seen < header.total_size {
            let msg = next_item(stream, "a chunk")
                .await
                .unwrap_or_else(|s| panic!("a chunk, not {s:?}"));
            match &msg.response_type {
                Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                    seen += c.data.len() as u64
                }
                other => panic!("a chunk, not {other:?}"),
            }
            chunks.push(msg);
        }
        let ack = header_of(next_item(stream, "the caught-up ack").await);
        assert_eq!(ack.total_size, 0, "the ack");
        CatchUp {
            header,
            chunks,
            ack,
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the catch-up did not drain within {DRAIN:?}"))
}

fn header_of(item: Result<SubscribeReadLayerResponse, Status>) -> SubscribeReadLayerHeader {
    match item
        .unwrap_or_else(|s| panic!("a header, not {s:?}"))
        .response_type
    {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
        other => panic!("a header, not {other:?}"),
    }
}

/// The first item of `stream`, which must be its ending status.
async fn first_status(stream: &mut SubStream, what: &str) -> Status {
    match next_item(stream, what).await {
        Err(status) => status,
        Ok(msg) => panic!("{what}: the first item is the status, not {msg:?}"),
    }
}

/// Everything `stream` yields to its end, under one deadline: its messages
/// and its status.
#[cfg(feature = "test-fault-injection")]
async fn drain_to_end(stream: &mut SubStream) -> (Vec<SubscribeReadLayerResponse>, Option<Status>) {
    tokio::time::timeout(DRAIN, async {
        let mut messages = Vec::new();
        let mut status = None;
        while let Some(item) = stream.next().await {
            match item {
                Ok(m) => messages.push(m),
                Err(s) => status = Some(s),
            }
        }
        (messages, status)
    })
    .await
    .unwrap_or_else(|_| panic!("the stream did not end within {DRAIN:?}"))
}

fn exits(svc: &LiveEditService, reason: &str) -> u64 {
    sample_value(
        &render(&svc.metrics),
        "live_edit_subscriber_stream_exits_total",
        &[("reason", reason)],
    )
    .expect("every exit reason is seeded") as u64
}

async fn await_condition<F: FnMut() -> bool>(mut condition: F, what: &str) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while !condition() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "{what} did not hold within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The `MONITOR` feed's commands on a blob key of `tv` (`ext:data:{hash}`)
/// since the tap started, up to a fence issued now, keyed by blob key.
async fn blob_commands(
    tv: &TestValkey,
    tap: &mut MonitorTap,
) -> BTreeMap<String, Vec<Vec<String>>> {
    let fence = monitor_fence(tv.prefix());
    let _: i64 = tv
        .client()
        .pool
        .exists(fence.as_str())
        .await
        .expect("the fence EXISTS must succeed");
    // Every blob key starts so: the key builder's spelling up to the hash's
    // `{` (`le:ext:data:{hash}`, the hash its hash tag).
    let any_blob = tv.key_schema().ext_data("x");
    let blob_prefix = &any_blob[..any_blob.len() - "x}".len()];
    let mut out: BTreeMap<String, Vec<Vec<String>>> = BTreeMap::new();
    for cmd in tap.drain_until(&fence).await {
        let MonitoredCommand { args, .. } = &cmd;
        for key in args.iter().skip(1).filter(|a| a.starts_with(blob_prefix)) {
            out.entry(key.clone()).or_default().push(args.clone());
        }
    }
    out
}

/// The blob key of `part`.
fn blob_key(tv: &TestValkey, part: &[u8]) -> String {
    tv.key_schema()
        .ext_data(&sha256_to_hex(&ext_content_hash(part)))
}

// ═══════════════════════════════════════════════════════════════════════════
// T2: the reads and the container
// ═══════════════════════════════════════════════════════════════════════════

/// T2 (fails before b.ntm, whose catch-up read each value whole with a
/// `GET`): a brand-new catch-up of a layer carrying three external values —
/// a few ranges' worth, one larger than a chunk, and a tiny one — sends the
/// container `OmpfContainer::serialize` makes of its delta and values, every
/// chunk 1 MiB but the last; and the store saw, on each value's blob key,
/// only `STRLEN` (and `EXISTS` for a value that reads 0) and `GETRANGE`s of
/// at most 65,472 B covering the value exactly once — no `GET`. The chunks'
/// pages hold the catch-up payload budget until the client drops them, and
/// return it all.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_reads_its_external_values_only_in_ranges_and_sends_the_serialized_container() {
    assert_eq!(
        live_edit::storage::ext_backup::EXT_VALUE_READ_CHUNK_BYTES as u64,
        RANGE_MAX_BYTES,
        "the ranged read's bound is the decided 65,472 B"
    );
    let tv = TestValkey::setup().await;
    let svc = service(&tv, ExtValueVerifyMode::Enforce);
    let layer = "payload-ranges";
    let parts = vec![
        value(70_000, 1),
        value(PAGE_BYTES + PAGE_BYTES / 2, 2),
        value(10, 3),
    ];
    let declared: Vec<(Vec<u8>, usize)> = parts.iter().map(|p| (p.clone(), p.len())).collect();
    let (session_id, _) = layer_with_external_values(&tv, &svc, layer, &declared).await;

    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let mut stream = subscribe(&svc, &session_id, layer, "ranged-reader").await;
    let catch_up = drain_catch_up(&mut stream).await;
    assert_eq!(
        catch_up.ack.sequence_num, 2,
        "caught up to the layer's create and its update"
    );

    let payload = catch_up.payload();
    assert_eq!(payload.len() as u64, catch_up.header.total_size);
    let container = OmpfContainer::parse(&payload).expect("the payload is an OMPF container");
    let mut stored: Vec<&[u8]> = container.external_values.clone();
    stored.sort();
    let mut sent: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    sent.sort();
    assert_eq!(
        stored, sent,
        "the container carries the three stored values"
    );
    assert!(
        OmpfContainer::serialize(container.delta_bytes, &container.external_values)
            .expect("serializes")
            == payload,
        "the payload is byte-identical to serialize of its delta and values"
    );
    let chunks = catch_up.chunk_data();
    for (i, chunk) in chunks.iter().enumerate() {
        if i + 1 < chunks.len() {
            assert_eq!(chunk.len(), PAGE_BYTES, "chunk {i}");
        }
    }

    let commands = blob_commands(&tv, &mut tap).await;
    for part in &parts {
        let key = blob_key(&tv, part);
        let on_key = commands
            .get(&key)
            .unwrap_or_else(|| panic!("the catch-up read {key}"));
        let names: Vec<String> = on_key.iter().map(|a| a[0].to_ascii_uppercase()).collect();
        for name in &names {
            assert!(
                ["STRLEN", "EXISTS", "GETRANGE"].contains(&name.as_str()),
                "{key}: only STRLEN, EXISTS and GETRANGE, never a whole-value read: {names:?}"
            );
        }
        assert!(names.iter().any(|n| n == "STRLEN"), "{key}: a length probe");
        let mut ranges: Vec<(u64, u64)> = on_key
            .iter()
            .filter(|a| a[0].eq_ignore_ascii_case("GETRANGE"))
            .map(|a| {
                let start: u64 = a[2].parse().expect("a start");
                let end: u64 = a[3].parse().expect("an end");
                (start, end)
            })
            .collect();
        ranges.sort_unstable();
        let mut covered = 0u64;
        for (start, end) in ranges {
            let len = end - start + 1;
            assert!(
                len <= RANGE_MAX_BYTES,
                "{key}: a range of {len} B, past {RANGE_MAX_BYTES}"
            );
            assert_eq!(
                start, covered,
                "{key}: ranges cover the value once, in order"
            );
            covered += len;
        }
        assert_eq!(covered, part.len() as u64, "{key}: every byte read");
    }

    let budget = &svc.valkey.catchup_payload_budget;
    let page_charges: u64 = chunks
        .iter()
        .map(|c| (c.len() as u64).div_ceil(1024) * 1024)
        .sum();
    assert_eq!(
        budget.reserved_bytes(),
        page_charges,
        "every page the client holds still holds its share of the catch-up payload budget"
    );
    drop(catch_up);
    assert_eq!(
        (budget.reserved_bytes(), budget.overcommit_bytes()),
        (0, 0),
        "dropping the pages returns the budget"
    );
}

/// T2b: under `LIVE_EDIT_EXT_VALUE_VERIFY=observe` a value whose stored length
/// is not the size its delta declared (here 3,096 B declared as 4,096) is
/// admitted and stored as sent; a catch-up of the layer then serves it as
/// stored, the container byte-identical to `serialize` of the stored value —
/// never `INTERNAL`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_observe_mode_value_stored_at_another_length_than_declared_is_served_as_stored() {
    let tv = TestValkey::setup().await;
    let svc = service(&tv, ExtValueVerifyMode::Observe);
    let layer = "payload-observe";
    let part = value(3_096, 9);
    let (session_id, _) =
        layer_with_external_values(&tv, &svc, layer, &[(part.clone(), 4_096)]).await;

    let mut stream = subscribe(&svc, &session_id, layer, "observe-reader").await;
    let catch_up = drain_catch_up(&mut stream).await;
    let payload = catch_up.payload();
    let container = OmpfContainer::parse(&payload).expect("an OMPF container");
    assert_eq!(container.external_values, vec![part.as_slice()]);
    assert!(
        OmpfContainer::serialize(container.delta_bytes, &[part.as_slice()]).expect("serializes")
            == payload
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// T3, T4: the catch-up payload budget
// ═══════════════════════════════════════════════════════════════════════════

/// T3 (fails before b.ntm, which had no payload budget): with the replica's
/// catch-up payload budget held by another catch-up (a test grant of all of
/// it), a catch-up of a layer with an external value waits its capacity
/// budget out and is refused `catchup_capacity` before any content —
/// `UNAVAILABLE` with the retry hint and no `wait` metadata — having probed
/// the value's length and read none of its bytes: no `GETRANGE`, no `GET`. Its
/// wait is counted `refused_timeout`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_catch_up_whose_payload_memory_is_held_is_refused_before_content_reading_no_value_byte() {
    let tv = TestValkey::setup().await;
    let svc = service(&tv, ExtValueVerifyMode::Enforce);
    let layer = "payload-refused";
    let part = value(200_000, 4);
    let (session_id, _) =
        layer_with_external_values(&tv, &svc, layer, &[(part.clone(), part.len())]).await;
    let budget = Arc::clone(&svc.valkey.catchup_payload_budget);
    let held = budget
        .reserve(budget.total_bytes(), None)
        .await
        .expect("the whole budget is free");

    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let mut stream = subscribe(&svc, &session_id, layer, "refused").await;
    let status = first_status(&mut stream, "the refusal").await;
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert_eq!(status.message(), CATCHUP_PAYLOAD_MSG);
    assert_refusal_detail(
        &status,
        refusal_reason::CATCHUP_CAPACITY,
        &[],
        CAPACITY_RETRY_DELAY,
    );
    await_condition(
        || exits(&svc, STREAM_EXIT_CATCHUP_CAPACITY) == 1,
        "the refusal is counted catchup_capacity",
    )
    .await;

    let commands = blob_commands(&tv, &mut tap).await;
    let names: Vec<String> = commands
        .get(&blob_key(&tv, &part))
        .map(|c| c.iter().map(|a| a[0].to_ascii_uppercase()).collect())
        .unwrap_or_default();
    assert!(
        names.iter().any(|n| n == "STRLEN"),
        "fixture self-check: the catch-up laid its payload out first: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n == "GETRANGE" || n == "GET"),
        "a catch-up refused its payload memory reads no value byte: {names:?}"
    );
    drop(held);
    assert_eq!(budget.reserved_bytes(), 0);
}

/// Refused catch-ups [`while_payload_memory_is_held_refused_catch_ups_leave_nothing_and_a_live_stream_keeps_delivering`]
/// retries: enough that an accumulation would show, few enough that their
/// capacity budgets (2 s each, paid in full) stay well inside the hang-killer.
const REFUSED_RETRIES: u64 = 3;

/// T11 (b.f4f's admission properties, with the catch-up payload budget
/// binding and catch-up slots ample): with the whole budget held, each of a
/// client's retried catch-ups is refused within its capacity budget —
/// admission stays finite — and leaves nothing behind it: no reservation
/// queued, no live-delivery hold, so retries do not accumulate; and a
/// subscription already caught up keeps receiving live updates throughout,
/// as a relay needs no payload memory.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn while_payload_memory_is_held_refused_catch_ups_leave_nothing_and_a_live_stream_keeps_delivering(
) {
    let tv = TestValkey::setup().await;
    let svc = service(&tv, ExtValueVerifyMode::Enforce);
    let layer = "payload-retried";
    let part = value(50_000, 7);
    let (session_id, object_id) =
        layer_with_external_values(&tv, &svc, layer, &[(part.clone(), part.len())]).await;
    let mut established = subscribe(&svc, &session_id, layer, "established").await;
    drop(drain_catch_up(&mut established).await);
    let budget = Arc::clone(&svc.valkey.catchup_payload_budget);
    let held = budget
        .reserve(budget.total_bytes(), None)
        .await
        .expect("the whole budget is free");
    let ledger = svc.valkey.stream_mux.delivery_ledger();

    for retry in 1..=REFUSED_RETRIES {
        let mut refused = subscribe(&svc, &session_id, layer, "retrying").await;
        let status = first_status(&mut refused, "the refusal").await;
        assert_eq!(status.message(), CATCHUP_PAYLOAD_MSG, "retry {retry}");
        await_condition(
            || {
                exits(&svc, STREAM_EXIT_CATCHUP_CAPACITY) == retry
                    && budget.waiters() == 0
                    && ledger.holds() == 1
            },
            "the refused catch-up left no reservation queued and no hold",
        )
        .await;

        let update = OwnedDelta {
            set_fields: vec![SetField {
                section_id: SECTION_ID,
                key_name: "live".to_owned(),
                value_or_ext_hash: vec![retry as u8; 16],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 100 + retry,
            }],
            ..Default::default()
        };
        let stream = streaming_update(vec![
            make_update_header(&session_id, layer, object_id),
            make_update_data(delta_to_ompf_bytes(&update, &[])),
        ]);
        Box::pin(handle_update_layer(&svc, make_request(stream)))
            .await
            .expect("the live commit succeeds");
        let live = header_of(next_item(&mut established, "the live update").await);
        assert_eq!(
            live.sequence_num,
            2 + retry,
            "the established subscription keeps delivering"
        );
        let _ = next_item(&mut established, "the live update's chunk")
            .await
            .expect("a chunk");
    }
    drop(held);
}

/// T4, the waits' exits: a catch-up waiting for the catch-up payload budget
/// whose client disconnects leaves the queue (`client_disconnect`); one
/// waiting when the budget is closed — `ValkeyClient::quit`'s shutdown step —
/// ends `UNAVAILABLE "server shutting down"` (`shutdown`) before any content.
/// Neither holds anything of the budget afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_catch_up_waiting_for_payload_memory_ends_on_its_disconnect_and_on_the_budgets_close() {
    let tv = TestValkey::setup().await;
    let mut svc = service(&tv, ExtValueVerifyMode::Enforce);
    let budget = with_payload_budget(&mut svc, 1 << 20);
    let layer = "payload-wait-exits";
    let part = value(10_000, 5);
    let (session_id, _) =
        layer_with_external_values(&tv, &svc, layer, &[(part.clone(), part.len())]).await;
    let held = budget
        .reserve(budget.total_bytes(), None)
        .await
        .expect("the whole budget is free");

    let leaving = subscribe(&svc, &session_id, layer, "leaving").await;
    await_condition(
        || budget.waiters() == 1,
        "the catch-up queues for the budget",
    )
    .await;
    drop(leaving);
    await_condition(
        || budget.waiters() == 0 && exits(&svc, STREAM_EXIT_CLIENT_DISCONNECT) == 1,
        "the disconnected catch-up leaves the queue",
    )
    .await;

    let mut closed = subscribe(&svc, &session_id, layer, "closed").await;
    await_condition(
        || budget.waiters() == 1,
        "the catch-up queues for the budget",
    )
    .await;
    budget.close();
    let status = first_status(&mut closed, "the shutdown ending").await;
    assert_shutdown_status(&status);
    await_condition(
        || exits(&svc, STREAM_EXIT_SHUTDOWN) == 1,
        "the ending is counted shutdown",
    )
    .await;
    drop(held);
    assert_eq!(
        (
            budget.reserved_bytes(),
            budget.overcommit_bytes(),
            budget.waiters()
        ),
        (0, 0, 0)
    );
}

/// T4, a disconnect after content: a client that goes away with its
/// catch-up's pages still queued, unread, returns their share of the catch-up
/// payload budget with its queue — and so does one that took only some.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_gone_with_its_catch_up_pages_queued_returns_their_payload_memory() {
    let tv = TestValkey::setup().await;
    let svc = service(&tv, ExtValueVerifyMode::Enforce);
    let layer = "payload-disconnect";
    let part = value(2 * PAGE_BYTES, 6);
    let (session_id, _) =
        layer_with_external_values(&tv, &svc, layer, &[(part.clone(), part.len())]).await;
    let budget = Arc::clone(&svc.valkey.catchup_payload_budget);

    for taken in [0, 2] {
        let mut stream = subscribe(&svc, &session_id, layer, &format!("gone-{taken}")).await;
        await_condition(
            || budget.reserved_bytes() >= 2 * PAGE_BYTES as u64,
            "the catch-up's pages are queued, holding the budget",
        )
        .await;
        for _ in 0..taken {
            let _ = next_item(&mut stream, "a message the client takes").await;
        }
        drop(stream);
        await_condition(
            || budget.reserved_bytes() == 0 && budget.overcommit_bytes() == 0,
            "the queued pages are dropped with the client's queue",
        )
        .await;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// T10: stalled holders under contention (P5a)
// ═══════════════════════════════════════════════════════════════════════════

/// The shrunk outer send deadline: how long a holder of catch-up pages may go
/// with nothing taken, and how often a waiting catch-up looks.
#[cfg(feature = "test-fault-injection")]
const TEST_OUTER_SEND_DEADLINE: Duration = Duration::from_millis(300);

/// A catch-up payload budget of 3 MiB: one catch-up of a 2.5 MiB value
/// (reserving its payload and a window of replies, clamped to the total) runs
/// alone, and a second must wait for its pages.
#[cfg(feature = "test-fault-injection")]
const SMALL_PAYLOAD_BUDGET: u64 = 3 << 20;

/// The value the contended layer carries: two and a half pages.
#[cfg(feature = "test-fault-injection")]
const CONTENDED_VALUE_BYTES: usize = 5 * PAGE_BYTES / 2;

/// A service with the shrunk outer deadline and the small payload budget,
/// and a layer carrying one [`CONTENDED_VALUE_BYTES`] value.
#[cfg(feature = "test-fault-injection")]
async fn contended(tv: &TestValkey, layer: &str) -> (LiveEditService, Arc<MemoryBudget>, String) {
    let mut svc = service(tv, ExtValueVerifyMode::Enforce);
    svc.outer_send_deadline_for_test = Some(TEST_OUTER_SEND_DEADLINE);
    let budget = with_payload_budget(&mut svc, SMALL_PAYLOAD_BUDGET);
    let part = value(CONTENDED_VALUE_BYTES, 8);
    let (session_id, _) =
        layer_with_external_values(tv, &svc, layer, &[(part.clone(), part.len())]).await;
    (svc, budget, session_id)
}

/// Subscribe `client` and wait until its catch-up is over and its pages are
/// queued: it holds its payload's share of the budget, and its slot is free.
#[cfg(feature = "test-fault-injection")]
async fn stalled_holder(
    svc: &LiveEditService,
    budget: &MemoryBudget,
    session_id: &str,
    layer: &str,
) -> SubStream {
    let stream = subscribe(svc, session_id, layer, "stalled").await;
    await_condition(
        || {
            budget.reserved_bytes() >= CONTENDED_VALUE_BYTES as u64
                && sample_value(
                    &render(&svc.metrics),
                    "live_edit_subscribe_catchups_inflight",
                    &[],
                )
                .unwrap_or(0.0)
                    == 0.0
        },
        "the stalled subscriber's catch-up has queued its pages and released its slot",
    )
    .await;
    stream
}

/// T10 (P5a): a subscriber whose catch-up pages sit in its response queue
/// with nothing taken holds the catch-up payload budget; a second catch-up
/// that cannot be granted waits, and while it waits the first is ended
/// `slow_reader_deadline` once it has had nothing taken for the outer send
/// deadline: its queue is dropped — a client resuming reads only the status —
/// its pages' memory comes back, and the waiter is granted and catches up.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stalled_holder_of_catch_up_pages_is_ended_when_a_catch_up_waits_and_the_waiter_is_granted(
) {
    let tv = TestValkey::setup().await;
    let layer = "payload-contended";
    let (svc, budget, session_id) = contended(&tv, layer).await;
    let mut stalled = stalled_holder(&svc, &budget, &session_id, layer).await;

    let mut waiter = subscribe(&svc, &session_id, layer, "waiter").await;
    let catch_up = drain_catch_up(&mut waiter).await;
    assert_eq!(catch_up.header.total_size, catch_up.payload().len() as u64);
    assert_eq!(
        exits(&svc, STREAM_EXIT_SLOW_READER_DEADLINE),
        1,
        "the stalled holder was ended slow_reader_deadline for the waiter"
    );

    let (messages, status) = drain_to_end(&mut stalled).await;
    assert!(
        messages.is_empty(),
        "the ending dropped the stalled subscriber's queued catch-up: {} message(s) left",
        messages.len()
    );
    let status = status.expect("the stalled subscriber's status");
    assert_eq!(status.code(), Code::Unavailable, "{status:?}");
    assert!(
        status.message().contains("did not read its stream"),
        "the slow-reader status: {status:?}"
    );
    drop(catch_up);
    assert_eq!(budget.reserved_bytes(), 0);
}

/// How long an unchallenged holder of catch-up pages is watched: several
/// outer send deadlines, any one of which a wrongly armed ending would use.
#[cfg(feature = "test-fault-injection")]
const QUIET: Duration = Duration::from_millis(1_200);

/// T10 (P5a), without contention: a subscriber whose catch-up pages sit
/// unread for several outer send deadlines is not ended while no catch-up
/// waits for the budget, and still receives its whole catch-up when it
/// reads.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_holder_of_catch_up_pages_is_not_ended_while_no_catch_up_waits() {
    let tv = TestValkey::setup().await;
    let layer = "payload-quiet";
    let (svc, budget, session_id) = contended(&tv, layer).await;
    let mut stalled = stalled_holder(&svc, &budget, &session_id, layer).await;

    tokio::time::sleep(QUIET).await;
    assert_eq!(
        exits(&svc, STREAM_EXIT_SLOW_READER_DEADLINE),
        0,
        "no catch-up waited, so nothing ended the holder"
    );
    let catch_up = drain_catch_up(&mut stalled).await;
    assert_eq!(
        catch_up.payload().len(),
        catch_up.header.total_size as usize
    );
}
