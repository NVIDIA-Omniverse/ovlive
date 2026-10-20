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

//! The CreateLayer / UpdateLayer upload content check (SDD §11.1), end to end
//! through the real handlers and a real `ValkeyExternalValueStore`.
//!
//! Every external-value reference in an upload must carry a 32-byte claim equal
//! to its part's content hash (SHA-256 over the SHA-256 digests of the part's
//! 1 MiB blocks) and an `extValueSize` equal to the part's length. Under the
//! default `LIVE_EDIT_EXT_VALUE_VERIFY=enforce` an upload that breaks either
//! rule is refused `INVALID_ARGUMENT` and nothing is applied: no `ext:data` /
//! `ext:meta` key, no sequence advance, no stream entry, no layer mapping, no
//! object id. An `enforce` upload that passes is stored as verified, which
//! replaces a damaged copy already stored under its hash (a key that is not a
//! string, or a string of the wrong length). Under `observe` the mismatch is
//! counted and the upload is stored unverified, so the set-if-absent data
//! write keeps any copy already stored.
//!
//! The unit-level arms (`storage::external_values::tests`) pin the walk, the
//! modes, the counter and the WARN against test doubles. This file pins what a
//! client observes: the status code, and the keyspace a refused or relaxed
//! call leaves behind.
//!
//! `build_service` wires a `MockExternalValueStore`; every test here replaces
//! it with a `ValkeyExternalValueStore` built `with_ext_value_verify(mode)`,
//! because `Config.ext_value_verify` configures nothing in tests.

use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface, StreamsInterface};

use live_edit::config::ExtValueVerifyMode;
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::service::LiveEditService;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::{
    ext_content_hash, ExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};

use super::helpers::*;
use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::common::read_header_u64;

/// The section every external field below lives in. Created with the layer so
/// the uploads under test carry only `SetField`s and can be repeated.
const SECTION_ID: u64 = 7;

/// Length of [`good_blob`]: big enough that a truncation is unmistakable,
/// small enough to stay one content-hash block.
const GOOD_BLOB_LEN: usize = 4096;

/// Length of [`truncated_blob`]: the 3096-of-4096 prefix the b.682 reproducer
/// stored under the full value's hash.
const TRUNCATED_LEN: usize = 3096;

/// Content hash of the single byte `0x00`, from the C++ client library's own
/// test vectors: what a real client sends.
const CLIENT_HASH_OF_ONE_ZERO_BYTE: &str =
    "1406e05881e299367766d313e26c05564ec91bf721d31726bd6e46e60689539a";

/// Plain SHA-256 of the single byte `0x00`: what a client that hashed the
/// obvious way would send. Not the content hash.
const PLAIN_SHA256_OF_ONE_ZERO_BYTE: &str =
    "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d";

/// A deterministic external value.
fn good_blob() -> Vec<u8> {
    (0..GOOD_BLOB_LEN).map(|i| (i % 251) as u8).collect()
}

/// The first [`TRUNCATED_LEN`] bytes of [`good_blob`]: the shape a cut-off
/// upload leaves, which OMPF still parses because it records only where each
/// part starts.
fn truncated_blob() -> Vec<u8> {
    good_blob()[..TRUNCATED_LEN].to_vec()
}

fn hash_from_hex(hex: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("a 64-char hex hash");
    }
    out
}

/// `build_service` with its mock external-value store replaced by a real
/// `ValkeyExternalValueStore` running the content check in `mode`, counted on
/// the returned metrics.
fn service_with_valkey_ext_store(
    tv: &TestValkey,
    mode: ExtValueVerifyMode,
) -> (LiveEditService, Arc<Metrics>) {
    let metrics = Arc::new(crate::common::metrics_scrape_only("ext-value-integrity"));
    let ext_value_store =
        ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_metrics(Arc::clone(&metrics))
            .with_ext_value_verify(mode);
    let svc = LiveEditService {
        ext_value_store: Arc::new(ext_value_store),
        ..build_service(tv, Arc::new(AllowAllPermissions))
    };
    (svc, metrics)
}

fn create_section() -> StructCommand {
    StructCommand::CreateSection(CommandCreateSection {
        parent_id: 0,
        section_id: SECTION_ID,
        section_name: "Asset".to_owned(),
        section_type: 1,
    })
}

/// An external-value `SetField` in [`SECTION_ID`].
fn ext_field(key: &str, claim: &[u8], ext_value_size: usize, ext_value_index: u32) -> SetField {
    SetField {
        section_id: SECTION_ID,
        key_name: key.to_owned(),
        value_or_ext_hash: claim.to_vec(),
        ext_value_size: ext_value_size as u64,
        ext_value_index,
        set_order: 0,
    }
}

/// A delta carrying only `fields`.
fn fields_delta(fields: Vec<SetField>) -> OwnedDelta {
    OwnedDelta {
        set_fields: fields,
        ..Default::default()
    }
}

/// Create a session and a layer whose first commit creates [`SECTION_ID`].
/// Returns `(session_id, object_id)`.
async fn create_layer_with_section(
    tv: &TestValkey,
    svc: &LiveEditService,
    layer: &str,
) -> (String, u64) {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(&format!("{layer}-session"), "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let delta = OwnedDelta {
        struct_commands: vec![create_section()],
        ..Default::default()
    };
    let stream = streaming_create(vec![
        make_create_header(&session_id, layer),
        make_create_data(delta_to_ompf_bytes(&delta, &[])),
    ]);
    let object_id = handle_create_layer(svc, make_request(stream))
        .await
        .expect("the setup CreateLayer must succeed")
        .into_inner()
        .object_id;
    (session_id, object_id)
}

/// One UpdateLayer carrying `delta` and `parts` as its OMPF.
async fn update(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    object_id: u64,
    delta: &OwnedDelta,
    parts: &[&[u8]],
) -> Result<(), tonic::Status> {
    let stream = streaming_update(vec![
        make_update_header(session_id, layer, object_id),
        make_update_data(delta_to_ompf_bytes(delta, parts)),
    ]);
    handle_update_layer(svc, make_request(stream))
        .await
        .map(|_| ())
}

/// What a commit moves on the object: its sequence and its stream length.
#[derive(Debug, PartialEq, Eq)]
struct ObjectMark {
    seq: u64,
    stream_len: u64,
}

async fn object_mark(tv: &TestValkey, object_id: u64) -> ObjectMark {
    let stream_len: u64 = tv
        .client()
        .pool
        .xlen(tv.key_schema().layer_stream(object_id))
        .await
        .expect("XLEN of the object stream must succeed");
    ObjectMark {
        seq: read_header_u64(tv, object_id, "seq")
            .await
            .expect("the object header carries its sequence"),
        stream_len,
    }
}

/// How many of `ext:meta:{hash}` / `ext:data:{hash}` exist (0, 1 or 2).
async fn ext_keys_present(tv: &TestValkey, hash: &[u8; 32]) -> i64 {
    let hex = sha256_to_hex(hash);
    tv.client()
        .pool
        .exists(vec![
            tv.key_schema().ext_meta(&hex),
            tv.key_schema().ext_data(&hex),
        ])
        .await
        .expect("EXISTS must succeed")
}

async fn ext_data(tv: &TestValkey, hash: &[u8; 32]) -> Option<Vec<u8>> {
    tv.client()
        .pool
        .get(tv.key_schema().ext_data(&sha256_to_hex(hash)))
        .await
        .expect("GET ext:data must succeed")
}

async fn ext_ref_count(tv: &TestValkey, hash: &[u8; 32]) -> Option<i64> {
    let raw: Option<String> = tv
        .client()
        .pool
        .hget(tv.key_schema().ext_meta(&sha256_to_hex(hash)), "ref_count")
        .await
        .expect("HGET ext:meta ref_count must succeed");
    raw.map(|v| v.parse().expect("ref_count is an integer"))
}

/// `live_edit_ext_value_verify_total{result}` as `(ok, size_mismatch,
/// hash_mismatch)`.
fn verify_cells(metrics: &Metrics) -> (u64, u64, u64) {
    let text = render(metrics);
    let cell = |result: &str| {
        sample_value(
            &text,
            "live_edit_ext_value_verify_total",
            &[("result", result)],
        )
        .unwrap_or_else(|| panic!("result={result} is seeded")) as u64
    };
    (cell("ok"), cell("size_mismatch"), cell("hash_mismatch"))
}

fn expect_invalid_argument(result: Result<(), tonic::Status>, context: &str) -> tonic::Status {
    let status = result.expect_err(context);
    assert_eq!(
        status.code(),
        tonic::Code::InvalidArgument,
        "{context}: expected INVALID_ARGUMENT, got {status:?}"
    );
    status
}

// ── the hash definition, through the handler ─────────────────────────────────

/// The client's block hash of a one-byte value is admitted and keys the stored
/// value. A plain-SHA-256 gate refuses it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_admits_a_one_byte_value_under_the_clients_content_hash() {
    let tv = TestValkey::setup().await;
    let (svc, _) = service_with_valkey_ext_store(&tv, ExtValueVerifyMode::Enforce);
    let layer = "content-hash-one-byte";
    let (sid, oid) = create_layer_with_section(&tv, &svc, layer).await;
    let part: &[u8] = &[0x00];
    let claim = hash_from_hex(CLIENT_HASH_OF_ONE_ZERO_BYTE);

    update(
        &svc,
        &sid,
        layer,
        oid,
        &fields_delta(vec![ext_field("v", &claim, 1, 1)]),
        &[part],
    )
    .await
    .expect("the client's content hash of [0x00] must be admitted");

    assert_eq!(
        ext_data(&tv, &claim).await.as_deref(),
        Some(part),
        "the value is stored under the client's content hash"
    );
}

/// A plain-SHA-256 claim for the same one-byte value is refused, and nothing is
/// stored or committed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_refuses_a_one_byte_value_under_its_plain_sha256() {
    let tv = TestValkey::setup().await;
    let (svc, _) = service_with_valkey_ext_store(&tv, ExtValueVerifyMode::Enforce);
    let layer = "content-hash-plain-sha256";
    let (sid, oid) = create_layer_with_section(&tv, &svc, layer).await;
    let part: &[u8] = &[0x00];
    let claim = hash_from_hex(PLAIN_SHA256_OF_ONE_ZERO_BYTE);
    let before = object_mark(&tv, oid).await;

    let result = update(
        &svc,
        &sid,
        layer,
        oid,
        &fields_delta(vec![ext_field("v", &claim, 1, 1)]),
        &[part],
    )
    .await;

    expect_invalid_argument(result, "a plain-SHA-256 claim must be refused");
    assert_eq!(ext_keys_present(&tv, &claim).await, 0, "nothing stored");
    assert_eq!(object_mark(&tv, oid).await, before, "nothing committed");
}

// ── enforce: a refused upload applies nothing ────────────────────────────────

/// An UpdateLayer whose well-formed OMPF carries a part that fails its claim is
/// refused, in each way a part can fail, and leaves the claim's keys absent,
/// the sequence and the object stream untouched.
///
/// The second and third cases are the truncation the b.682 reproducer stored:
/// a 3096-byte prefix under the full value's hash. Declaring the full size
/// fails the size check first; declaring the prefix's own size reaches the
/// hash check. The status text tells which check refused it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_refuses_a_part_that_fails_its_claim_and_applies_nothing() {
    let tv = TestValkey::setup().await;
    let (svc, _) = service_with_valkey_ext_store(&tv, ExtValueVerifyMode::Enforce);
    let layer = "content-check-refusal";
    let (sid, oid) = create_layer_with_section(&tv, &svc, layer).await;
    let good = good_blob();
    let truncated = truncated_blob();
    let good_hash = ext_content_hash(&good);
    let wrong_hash = ext_content_hash(b"bytes this upload does not carry");

    /// One way a part can fail its reference, and the rule the refusal names.
    struct Case<'a> {
        case: &'a str,
        part: &'a [u8],
        claim: [u8; 32],
        size: usize,
        rule: &'a str,
    }
    let cases = [
        Case {
            case: "a wrong hash",
            part: &good,
            claim: wrong_hash,
            size: good.len(),
            rule: "valueOrExtHash",
        },
        Case {
            case: "a truncated prefix declaring the full size",
            part: &truncated,
            claim: good_hash,
            size: good.len(),
            rule: "extValueSize",
        },
        Case {
            case: "a truncated prefix declaring its own size",
            part: &truncated,
            claim: good_hash,
            size: truncated.len(),
            rule: "valueOrExtHash",
        },
    ];

    for Case {
        case,
        part,
        claim,
        size,
        rule,
    } in cases
    {
        let before = object_mark(&tv, oid).await;

        let result = update(
            &svc,
            &sid,
            layer,
            oid,
            &fields_delta(vec![ext_field("v", &claim, size, 1)]),
            &[part],
        )
        .await;

        let status = expect_invalid_argument(result, case);
        assert!(
            status.message().contains(rule),
            "{case}: the refusal must name the rule it broke ({rule}); got: {}",
            status.message()
        );
        assert_eq!(
            ext_keys_present(&tv, &claim).await,
            0,
            "{case}: no ext:meta / ext:data may exist for the claim"
        );
        assert_eq!(
            object_mark(&tv, oid).await,
            before,
            "{case}: the sequence and the object stream must not move"
        );
    }
}

/// A valid value travelling with a bad one is not stored either: the whole
/// upload is checked before the first store.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_refuses_a_valid_value_alongside_a_bad_one_and_stores_neither() {
    let tv = TestValkey::setup().await;
    let (svc, _) = service_with_valkey_ext_store(&tv, ExtValueVerifyMode::Enforce);
    let layer = "content-check-valid-then-bad";
    let (sid, oid) = create_layer_with_section(&tv, &svc, layer).await;
    let valid = good_blob();
    let valid_hash = ext_content_hash(&valid);
    let bad = truncated_blob();
    let bad_claim = ext_content_hash(b"the bytes the bad claim names");
    let before = object_mark(&tv, oid).await;

    let result = update(
        &svc,
        &sid,
        layer,
        oid,
        &fields_delta(vec![
            ext_field("valid", &valid_hash, valid.len(), 1),
            ext_field("bad", &bad_claim, bad.len(), 2),
        ]),
        &[valid.as_slice(), bad.as_slice()],
    )
    .await;

    expect_invalid_argument(result, "the bad value refuses the upload");
    assert_eq!(
        ext_keys_present(&tv, &valid_hash).await,
        0,
        "the valid value ahead of the bad one must not have been stored"
    );
    assert_eq!(
        ext_keys_present(&tv, &bad_claim).await,
        0,
        "nor the bad one"
    );
    assert_eq!(object_mark(&tv, oid).await, before, "nothing committed");
}

/// A part that matches its content hash but declares one byte more or less
/// is refused, and nothing is stored or committed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_refuses_an_ext_value_size_one_off_the_part_length_and_applies_nothing() {
    let tv = TestValkey::setup().await;
    let (svc, _) = service_with_valkey_ext_store(&tv, ExtValueVerifyMode::Enforce);
    let layer = "content-check-size-off-by-one";
    let (sid, oid) = create_layer_with_section(&tv, &svc, layer).await;
    let part = good_blob();
    let claim = ext_content_hash(&part);

    for size in [part.len() - 1, part.len() + 1] {
        let before = object_mark(&tv, oid).await;

        let result = update(
            &svc,
            &sid,
            layer,
            oid,
            &fields_delta(vec![ext_field("v", &claim, size, 1)]),
            &[part.as_slice()],
        )
        .await;

        let status = expect_invalid_argument(result, "extValueSize off by one");
        assert!(
            status.message().contains("extValueSize"),
            "size {size}: {}",
            status.message()
        );
        assert_eq!(
            ext_keys_present(&tv, &claim).await,
            0,
            "size {size}: nothing stored"
        );
        assert_eq!(
            object_mark(&tv, oid).await,
            before,
            "size {size}: nothing committed"
        );
    }
}

/// A 33-byte claim whose first 32 bytes ARE the part's content hash is refused
/// on the default Lua fast path. That path never runs `apply_delta` (which
/// refuses a wrong-length claim on the other paths) and its encoder truncates
/// the claim, so before the content check it stored the value and committed
/// the truncated claim.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_refuses_a_33_byte_claim_on_the_lua_fast_path_and_applies_nothing() {
    let tv = TestValkey::setup().await;
    let (svc, _) = service_with_valkey_ext_store(&tv, ExtValueVerifyMode::Enforce);
    let layer = "content-check-33-byte-claim";
    let (sid, oid) = create_layer_with_section(&tv, &svc, layer).await;
    let part = good_blob();
    let hash = ext_content_hash(&part);
    let mut claim = hash.to_vec();
    claim.push(0x00);
    let before = object_mark(&tv, oid).await;

    let result = update(
        &svc,
        &sid,
        layer,
        oid,
        &fields_delta(vec![ext_field("v", &claim, part.len(), 1)]),
        &[part.as_slice()],
    )
    .await;

    expect_invalid_argument(result, "a 33-byte claim must be refused");
    assert_eq!(
        ext_keys_present(&tv, &hash).await,
        0,
        "nothing stored under the claim's first 32 bytes"
    );
    assert_eq!(object_mark(&tv, oid).await, before, "nothing committed");
}

/// A CreateLayer whose upload fails the check is refused and creates nothing:
/// no layer mapping, no object id allocated, no external-value keys.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_refuses_a_part_that_fails_its_claim_and_creates_nothing() {
    let tv = TestValkey::setup().await;
    let (svc, _) = service_with_valkey_ext_store(&tv, ExtValueVerifyMode::Enforce);
    let layer = "content-check-create";
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("content-check-create-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let part = good_blob();
    let truthful_hash = ext_content_hash(&part);
    let claim = ext_content_hash(b"the bytes the claim names");
    let counter_key = tv.key_schema().counter_object_id();
    let counter_before: Option<String> = tv
        .client()
        .pool
        .get(&counter_key)
        .await
        .expect("GET the object-id counter must succeed");

    let delta = OwnedDelta {
        struct_commands: vec![create_section()],
        set_fields: vec![ext_field("v", &claim, part.len(), 1)],
        ..Default::default()
    };
    let stream = streaming_create(vec![
        make_create_header(&session_id, layer),
        make_create_data(delta_to_ompf_bytes(&delta, &[part.as_slice()])),
    ]);
    let result = handle_create_layer(&svc, make_request(stream))
        .await
        .map(|_| ());

    expect_invalid_argument(result, "a CreateLayer that fails the check must be refused");
    let mapped: i64 = tv
        .client()
        .pool
        .exists(tv.key_schema().layer_mapping(&session_id, layer))
        .await
        .expect("EXISTS the layer mapping must succeed");
    assert_eq!(mapped, 0, "no layer mapping may be created");
    let counter_after: Option<String> = tv
        .client()
        .pool
        .get(&counter_key)
        .await
        .expect("GET the object-id counter must succeed");
    assert_eq!(
        counter_after, counter_before,
        "no object id may be allocated for a refused CreateLayer"
    );
    assert_eq!(
        ext_keys_present(&tv, &claim).await,
        0,
        "nothing stored under the claim"
    );
    assert_eq!(
        ext_keys_present(&tv, &truthful_hash).await,
        0,
        "nor under the part's true content hash"
    );
}

// ── (c) end to end: a stored blob survives a later bad upload of its hash ─────

/// A good blob stored under `h` survives a later UpdateLayer that sends a
/// truncated value under `h`: the later upload is refused, the stored bytes
/// are byte for byte the good ones, and the refused call pays no increment.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_truncated_upload_leaves_the_stored_blob_and_its_ref_count_unchanged() {
    let tv = TestValkey::setup().await;
    let (svc, _) = service_with_valkey_ext_store(&tv, ExtValueVerifyMode::Enforce);
    let layer = "content-check-survivor";
    let (sid, oid) = create_layer_with_section(&tv, &svc, layer).await;
    let good = good_blob();
    let truncated = truncated_blob();
    let h = ext_content_hash(&good);

    update(
        &svc,
        &sid,
        layer,
        oid,
        &fields_delta(vec![ext_field("v", &h, good.len(), 1)]),
        &[good.as_slice()],
    )
    .await
    .expect("the good upload must be admitted");
    let ref_count_before = ext_ref_count(&tv, &h)
        .await
        .expect("the good upload stored a meta");
    let before = object_mark(&tv, oid).await;

    let result = update(
        &svc,
        &sid,
        layer,
        oid,
        &fields_delta(vec![ext_field("w", &h, truncated.len(), 1)]),
        &[truncated.as_slice()],
    )
    .await;

    expect_invalid_argument(result, "the truncated upload must be refused");
    assert_eq!(
        ext_data(&tv, &h).await.as_deref(),
        Some(good.as_slice()),
        "the stored blob must still be the good one, byte for byte"
    );
    assert_eq!(
        ext_ref_count(&tv, &h).await,
        Some(ref_count_before),
        "the refused call must pay no pre-flight increment"
    );
    assert_eq!(object_mark(&tv, oid).await, before, "nothing committed");
}

// ── the verified heal: a correct upload repairs a truncated blob ─────────────

/// A truncated prefix stored under the full value's hash before the content
/// check existed is replaced by the next correct UpdateLayer of that hash
/// under `enforce`. The commit lands, the stored bytes are the good ones byte
/// for byte, and the upload pays one pre-flight increment.
///
/// The b.682 reproducer, end to end. The truncated copy is planted through an
/// unverified `store_values` on a second store, standing in for a store made
/// before the gate existed. The handler then reaches the heal through
/// `store_external_values`, which stores an admitted `enforce` upload with
/// `store_verified_values`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_heals_a_truncated_blob_stored_before_the_gate_with_a_correct_upload() {
    let tv = TestValkey::setup().await;
    let (svc, _) = service_with_valkey_ext_store(&tv, ExtValueVerifyMode::Enforce);
    let layer = "content-check-heal";
    let (sid, oid) = create_layer_with_section(&tv, &svc, layer).await;
    let good = good_blob();
    let truncated = truncated_blob();
    let h = ext_content_hash(&good);
    ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
        .store_values(&[(&h, truncated.as_slice())])
        .await
        .expect("planting the pre-gate truncated copy must succeed");
    let ref_count_before = ext_ref_count(&tv, &h)
        .await
        .expect("staged: the planted copy has a meta");
    let before = object_mark(&tv, oid).await;

    update(
        &svc,
        &sid,
        layer,
        oid,
        &fields_delta(vec![ext_field("v", &h, good.len(), 1)]),
        &[good.as_slice()],
    )
    .await
    .expect("a correct upload must be admitted");

    // Length first, then a plain `==`: an `assert_eq!` on 4 KiB values would
    // print both in full.
    let stored = ext_data(&tv, &h)
        .await
        .expect("a data key must be stored under the hash");
    assert_eq!(
        stored.len(),
        good.len(),
        "the truncated copy must be replaced by the verified bytes"
    );
    assert!(stored == good, "byte for byte");
    assert_eq!(
        ext_ref_count(&tv, &h).await,
        Some(ref_count_before + 1),
        "the upload pays its one pre-flight increment"
    );
    assert_eq!(
        read_header_u64(&tv, oid, "seq").await,
        Some(before.seq + 1),
        "and the commit lands"
    );
}

// ── observe: counted, logged, stored ─────────────────────────────────────────

/// Under `observe` a truncated upload claiming an already-stored hash is
/// admitted and counted, and the set-if-absent write keeps the stored blob.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn observe_mode_admits_a_truncated_upload_over_a_stored_blob_and_keeps_the_blob() {
    let tv = TestValkey::setup().await;
    let (svc, metrics) = service_with_valkey_ext_store(&tv, ExtValueVerifyMode::Observe);
    let layer = "content-check-observe-survivor";
    let (sid, oid) = create_layer_with_section(&tv, &svc, layer).await;
    let good = good_blob();
    let truncated = truncated_blob();
    let h = ext_content_hash(&good);

    update(
        &svc,
        &sid,
        layer,
        oid,
        &fields_delta(vec![ext_field("v", &h, good.len(), 1)]),
        &[good.as_slice()],
    )
    .await
    .expect("the good upload must be admitted");
    let ref_count_before = ext_ref_count(&tv, &h)
        .await
        .expect("the good upload stored a meta");
    let cells_before = verify_cells(&metrics);

    update(
        &svc,
        &sid,
        layer,
        oid,
        &fields_delta(vec![ext_field("w", &h, truncated.len(), 1)]),
        &[truncated.as_slice()],
    )
    .await
    .expect("observe must admit a hash mismatch");

    assert_eq!(
        ext_data(&tv, &h).await.as_deref(),
        Some(good.as_slice()),
        "the stored blob must still be the good one: observe stores, but never over an \
         existing copy"
    );
    assert_eq!(
        ext_ref_count(&tv, &h).await,
        Some(ref_count_before + 1),
        "the admitted call pays its one pre-flight increment"
    );
    let (ok, size_mismatch, hash_mismatch) = verify_cells(&metrics);
    assert_eq!(
        (ok, size_mismatch, hash_mismatch),
        (cells_before.0, cells_before.1, cells_before.2 + 1),
        "the truncated value declaring its own size is one hash mismatch"
    );
}

/// Under `observe` a mismatched value under a hash nothing has stored yet is
/// stored as sent, the commit lands, and the mismatch is counted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn observe_mode_stores_a_mismatched_first_copy_as_sent_and_counts_it() {
    let tv = TestValkey::setup().await;
    let (svc, metrics) = service_with_valkey_ext_store(&tv, ExtValueVerifyMode::Observe);
    let layer = "content-check-observe-first-copy";
    let (sid, oid) = create_layer_with_section(&tv, &svc, layer).await;
    let part = good_blob();
    let h2 = ext_content_hash(b"the bytes this claim names");
    assert_eq!(ext_keys_present(&tv, &h2).await, 0, "staged: h2 is absent");
    let before = object_mark(&tv, oid).await;

    update(
        &svc,
        &sid,
        layer,
        oid,
        &fields_delta(vec![ext_field("v", &h2, part.len(), 1)]),
        &[part.as_slice()],
    )
    .await
    .expect("observe must admit a hash mismatch");

    assert_eq!(
        ext_data(&tv, &h2).await.as_deref(),
        Some(part.as_slice()),
        "observe stores the first copy as sent"
    );
    assert_eq!(
        read_header_u64(&tv, oid, "seq").await,
        Some(before.seq + 1),
        "and the commit lands"
    );
    assert_eq!(
        verify_cells(&metrics),
        (0, 0, 1),
        "one hash mismatch counted"
    );
}

// ── check once, store any number of times (b.5wb) ────────────────────────────

/// A Valkey external-value store checking in `mode`, counted on the returned
/// metrics.
fn valkey_ext_store(
    tv: &TestValkey,
    mode: ExtValueVerifyMode,
) -> (ValkeyExternalValueStore, Arc<Metrics>) {
    let metrics = Arc::new(crate::common::metrics_scrape_only("ext-value-prepare"));
    let store = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_metrics(Arc::clone(&metrics))
        .with_ext_value_verify(mode);
    (store, metrics)
}

/// `prepare_external_values` checks an upload's references once and
/// `PreparedExternalValues::store` stores the result any number of times —
/// what the `UpdateLayer` checkpoint-recovery re-issue's own pre-store
/// relies on. Two references to one good value: preparing counts both
/// (`ok` +2) and stores nothing; it holds one distinct value of
/// [`GOOD_BLOB_LEN`] bytes; each store adds one reference (two stores, a
/// `ref_count` of 2) and checks nothing again (the counter stays at 2).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_external_values_are_checked_once_and_stored_on_every_store() {
    use live_edit::storage::external_values::prepare_external_values;

    let tv = TestValkey::setup().await;
    let (store, metrics) = valkey_ext_store(&tv, ExtValueVerifyMode::Enforce);
    let blob = good_blob();
    let hash = ext_content_hash(&blob);
    let delta = fields_delta(vec![
        ext_field("a", &hash, blob.len(), 1),
        ext_field("b", &hash, blob.len(), 1),
    ]);
    let parts: [&[u8]; 1] = [blob.as_slice()];

    let prepared =
        prepare_external_values(Some(&delta), &parts, &store).expect("a good upload prepares");

    assert_eq!(verify_cells(&metrics), (2, 0, 0), "both references checked");
    assert!(!prepared.is_empty());
    assert_eq!(
        prepared.stored_bytes(),
        GOOD_BLOB_LEN as u64,
        "one distinct value: its bytes once"
    );
    assert_eq!(
        ext_keys_present(&tv, &hash).await,
        0,
        "preparing stores nothing"
    );

    for stores in 1..=2 {
        prepared.store(&store).await.expect("the store succeeds");
        assert_eq!(
            ext_ref_count(&tv, &hash).await,
            Some(stores),
            "store {stores} adds one reference to the one distinct value"
        );
    }
    assert_eq!(ext_data(&tv, &hash).await.as_deref(), Some(blob.as_slice()));
    assert_eq!(
        verify_cells(&metrics),
        (2, 0, 0),
        "storing checks nothing again"
    );
}

/// The store step runs in the mode the check ran under. In `observe` a part
/// failing its claim is counted once when prepared and stored unverified —
/// set-if-absent, never healing — on every store: over a good copy already
/// stored under the claim, two stores of the truncated part keep the good
/// bytes and add two references. (An `enforce` store of verified bytes would
/// replace the stored copy.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_external_values_store_in_the_mode_their_check_ran_under() {
    use live_edit::storage::external_values::prepare_external_values;

    let tv = TestValkey::setup().await;
    let (store, metrics) = valkey_ext_store(&tv, ExtValueVerifyMode::Observe);
    let good = good_blob();
    let hash = ext_content_hash(&good);
    store
        .store_values(&[(&hash, good.as_slice())])
        .await
        .expect("the good copy is stored first");
    let part = truncated_blob();
    let delta = fields_delta(vec![ext_field("v", &hash, part.len(), 1)]);
    let parts: [&[u8]; 1] = [part.as_slice()];

    let prepared = prepare_external_values(Some(&delta), &parts, &store)
        .expect("observe prepares a part failing its claim");
    prepared.store(&store).await.expect("the first store");
    prepared.store(&store).await.expect("the second store");

    assert_eq!(
        verify_cells(&metrics),
        (0, 0, 1),
        "the mismatch counted once, when prepared"
    );
    assert_eq!(
        ext_data(&tv, &hash).await.as_deref(),
        Some(good.as_slice()),
        "an unverified store never replaces the stored copy"
    );
    assert_eq!(ext_ref_count(&tv, &hash).await, Some(3));
}

/// An upload with no external-value references — no delta, or a delta
/// without any — prepares an empty set, which stores nothing and charges
/// nothing (`stored_bytes` 0), and counts nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_external_values_of_an_upload_without_references_are_empty() {
    use live_edit::storage::external_values::prepare_external_values;

    let tv = TestValkey::setup().await;
    let (store, metrics) = valkey_ext_store(&tv, ExtValueVerifyMode::Enforce);
    let inline = OwnedDelta {
        struct_commands: vec![create_section()],
        ..Default::default()
    };
    for delta in [None, Some(&inline)] {
        let prepared =
            prepare_external_values(delta, &[], &store).expect("nothing to check prepares");
        assert!(prepared.is_empty(), "{delta:?}");
        assert_eq!(prepared.stored_bytes(), 0, "{delta:?}");
        prepared
            .store(&store)
            .await
            .expect("an empty store is a no-op");
    }
    assert_eq!(verify_cells(&metrics), (0, 0, 0), "nothing checked");
    let pattern = format!("{}*", tv.prefix());
    let keys = crate::common::scan_keys(tv.client(), &pattern).await;
    assert!(
        keys.iter().all(|k| !format!("{k:?}").contains("ext:")),
        "nothing stored: {keys:?}"
    );
}
