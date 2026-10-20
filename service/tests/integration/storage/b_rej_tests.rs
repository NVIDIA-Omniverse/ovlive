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

// b.rej — the external-value PRE-FLIGHT store moved off the txn pool: the
// per-hash `MULTI { SET ext_data; HINCRBY ext_meta ref_count 1 } EXEC` that
// `ValkeyExternalValueStore::store_values` ran on an exclusive `txn_pool` slot
// is now one `EVALSHA` of `scripts/ext_store_value.lua` on the multiplexed DATA
// pool, and the increment gained a server-side `last_store_ms` stamp.
//
// ── What this file is for ────────────────────────────────────────────────────
//
// The change is billed as "same semantics, different transport, plus one new
// field". Everything here drives the REAL `ValkeyExternalValueStore` against a
// real Valkey and checks that bill line by line, because a Lua transcription of
// a `MULTI` can differ from it in ways no type checks:
//
//   * the count (`HINCRBY` vs a re-`SET`, one `+1` per call per DISTINCT hash),
//   * the bytes (`Value::Bytes` argv vs a `String` — a NUL byte in a multi-MB
//     blob is not a theoretical payload shape),
//   * the write mode (`SET … NX` since b.682: `store_values` never overwrites
//     an existing `le:ext:data:{hash_hex}` key, of any type, so a later
//     truncated or mismatched store cannot replace the copy every referencing
//     layer reads, while the increment and the stamp are still paid; § 4
//     below. The one exception is the VERIFIED store the live path makes under
//     `enforce`, which replaces an existing key that is not a string or whose
//     length differs from the verified bytes; § 5 below),
//   * the stamp (present, decimal, milliseconds, and written by the store
//     rather than by the GC),
//   * and the pool (a store must no longer wait on a `txn_pool` slot at all).
//
// ── Division of labour with the neighbouring files ───────────────────────────
//
// * `external_value_test.rs` owns the pre-existing `store_values` contract —
//   dedup, retry additivity, the exact `ext_meta` FIELD SET, the GC race. b.rej
//   widened its field-set assertion to `{ref_count, last_store_ms}`; the VALUE
//   of the stamp is pinned here.
// * `b_qpf_tests.rs` / `b_yda_tests.rs` own `ref_count`'s meaning (one per live
//   object per distinct hash) across the commit backends. Neither is touched by
//   b.rej and both must keep passing unchanged — that is the parity argument,
//   and it is a run, not a test.
// * `src/storage/ext_reconcile.rs`'s in-crate module owns the READER of the
//   stamp, including the arm that pins the writer's own stamp end to end
//   (`the_writers_own_stamp_gates_the_pass_from_skip_to_repair`).
// * `b_289_tests.rs` owns the two source-shape fences the new script had to
//   satisfy: `.multi()` in `src/storage` dropped 4 → 3, and the new `EVALSHA`
//   site must pass keys (it passes both `{hash_hex}`-tagged ones).
// * The first-appearance ORDER of `collect_external_values` is a pure function's
//   contract and is pinned by its own in-crate unit tests
//   (`collect_external_values_orders_by_first_appearance_across_both_maps`,
//   `..._dedups_a_hash_shared_by_a_field_and_a_time_sample`); it is `pub(crate)`
//   and unreachable from here. What this file pins instead is the OBSERVABLE
//   consequence through the public `store_external_values` door: one `+1` per
//   distinct hash, dedup spanning both maps.
//
// ── One deliberately-pinned defect ───────────────────────────────────────────
//
// The wrong-typed-meta arm asserts a HALF-WRITE: the data key stands while the
// increment is refused. That is not an endorsement — it is the pre-existing
// `MULTI` behaviour transcribed faithfully (Valkey rolls back neither a script
// nor a `MULTI`), pinned so the fix has something to change deliberately. It is
// filed as **b.5yz**.

use std::time::Duration;

use fred::interfaces::{ClientLike, HashesInterface, KeysInterface, ListInterface};
use fred::types::{ClusterHash, CustomCommand, Value};

use live_edit::config::ExtValueVerifyMode;
use live_edit::error::ServiceError;
use live_edit::storage::external_values::{
    ext_content_hash, process_pending_gc, store_external_values, ExtRelease, ExternalValueStore,
    ValkeyExternalValueStore,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{OwnedDelta, SetField, TimeSample};

/// The `ext_meta` field carrying the live-object reference count.
///
/// Spelled out rather than imported: both field names are a cross-file contract
/// between a Lua writer that names them literally and a Rust reader that names
/// them in a `const`, so a test reading either from the emitter's own constant
/// could only ever agree with itself.
const REF_COUNT_FIELD: &str = "ref_count";
/// The `ext_meta` field carrying the pre-flight store's Valkey-side timestamp.
/// See [`REF_COUNT_FIELD`] for why it is restated here.
const LAST_STORE_MS_FIELD: &str = "last_store_ms";

/// Sanity floor for a millisecond epoch stamp: 2020-09-13T12:26:40Z.
///
/// Its job is to catch a stamp written in SECONDS rather than milliseconds,
/// which no bracket against a same-host clock reading would necessarily catch on
/// its own and which would be quietly catastrophic — every meta would date to
/// 1970 and read as ancient, so the reconciler would judge every hash eligible
/// the first time it saw one.
const PLAUSIBLE_MS_FLOOR: u64 = 1_600_000_000_000;

/// How long a `store_values` may take while another task holds every `txn_pool`
/// slot. Generous: this is a deadline that the healthy path returns in
/// milliseconds from, not a latency assertion. Pre-b.rej the call would wait on
/// the pool forever, so any finite budget separates the two outcomes.
const TXN_STARVED_STORE_BUDGET: Duration = Duration::from_secs(10);

fn ext_store(tv: &TestValkey) -> ValkeyExternalValueStore {
    ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
}

fn test_hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

/// Valkey's own clock in milliseconds, read the same way
/// `storage::ext_reconcile` reads it.
async fn valkey_now_ms(tv: &TestValkey) -> u64 {
    let reply: Vec<String> = tv
        .client()
        .pool
        .custom(
            CustomCommand::new_static("TIME", ClusterHash::FirstKey, false),
            Vec::<Value>::new(),
        )
        .await
        .expect("TIME must succeed");
    let secs: u64 = reply[0].parse().expect("TIME seconds must parse");
    let usecs: u64 = reply[1].parse().expect("TIME microseconds must parse");
    secs * 1000 + usecs / 1000
}

/// The raw `last_store_ms` string, or `None` when the field is absent.
async fn raw_stamp(tv: &TestValkey, hash_hex: &str) -> Option<String> {
    tv.client()
        .pool
        .hget(tv.key_schema().ext_meta(hash_hex), LAST_STORE_MS_FIELD)
        .await
        .expect("HGET last_store_ms must succeed")
}

async fn ref_count(tv: &TestValkey, hash_hex: &str) -> Option<i64> {
    let raw: Option<String> = tv
        .client()
        .pool
        .hget(tv.key_schema().ext_meta(hash_hex), REF_COUNT_FIELD)
        .await
        .expect("HGET ref_count must succeed");
    raw.map(|v| v.parse().expect("ref_count must be an integer"))
}

async fn data_bytes(tv: &TestValkey, hash_hex: &str) -> Option<Vec<u8>> {
    tv.client()
        .pool
        .get(tv.key_schema().ext_data(hash_hex))
        .await
        .expect("GET ext_data must succeed")
}

/// Valkey's own `TYPE` verdict for a key, e.g. `"string"` / `"list"` / `"none"`.
///
/// Read raw rather than inferred from whether a typed accessor succeeded: an
/// arm that only ever `GET`s reports a surviving wrong-typed key as a
/// `WRONGTYPE` panic from inside a helper, which reads as a broken test rather
/// than as the finding it is.
async fn key_type(tv: &TestValkey, key: &str) -> String {
    tv.client()
        .pool
        .custom(
            CustomCommand::new_static("TYPE", ClusterHash::FirstKey, false),
            vec![Value::from(key.to_owned())],
        )
        .await
        .expect("TYPE must succeed")
}

/// Assert a raw stamp is the shape the reconciler's `parse::<u64>()` requires,
/// and return it parsed.
///
/// The digit check is not pedantry: Lua numbers are doubles and `tostring` on a
/// millisecond epoch yields `1.7654e+12`, which parses as a float everywhere
/// except in the one place it has to — the reader classifies it `Unreadable` and
/// then declines to judge that hash for as long as it holds that value.
/// `string.format("%d", ms)` is what prevents it, and this is the assertion that
/// notices if that formatting is ever dropped.
fn parsed_stamp(raw: &str) -> u64 {
    assert!(
        !raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()),
        "last_store_ms must be a plain decimal integer with no exponent, sign or decimal point \
         — the reconciler reads it with `parse::<u64>()` and classifies anything else as \
         corrupt keyspace it must fail closed on; got {raw:?}"
    );
    raw.parse::<u64>()
        .unwrap_or_else(|e| panic!("last_store_ms {raw:?} must parse as u64: {e}"))
}

// =============================================================================
// 1 — the fresh store: data, count, stamp
// =============================================================================

/// A first store of a hash writes the blob, sets `ref_count = 1`, and stamps
/// `last_store_ms` with a plausible millisecond timestamp bracketed by Valkey's
/// own clock either side of the call.
///
/// The bracket cannot, on a single-node fixture, distinguish the server's clock
/// from the caller's — they are the same host. What it does establish is that
/// the stamp is a real epoch-MILLISECOND value taken during this call rather
/// than a constant, a seconds value, or a build stamp; the server-side
/// derivation itself is a property of the script (`redis.call("TIME")`), which
/// exists so replicas cannot disagree by their NTP skew.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn store_values_writes_the_blob_sets_ref_count_one_and_stamps_last_store_ms() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let hash = test_hash(0x10);
    let hex = sha256_to_hex(&hash);
    let payload = b"b.rej fresh payload".as_slice();

    let before = valkey_now_ms(&tv).await;
    store
        .store_values(&[(&hash, payload)])
        .await
        .expect("the pre-flight store must succeed");
    let after = valkey_now_ms(&tv).await;

    assert_eq!(
        data_bytes(&tv, &hex).await.as_deref(),
        Some(payload),
        "the content-addressed blob must land under ext_data"
    );
    assert_eq!(
        ref_count(&tv, &hex).await,
        Some(1),
        "a first store is exactly one pre-flight increment"
    );

    let raw = raw_stamp(&tv, &hex).await.expect(
        "every pre-flight store must stamp last_store_ms — the reconciler treats an \
                 ABSENT stamp as eligible for repair, so a writer that stops stamping does not \
                 fail loudly, it makes live references repairable",
    );
    let stamp = parsed_stamp(&raw);
    assert!(
        stamp >= PLAUSIBLE_MS_FLOOR,
        "the stamp must be MILLISECONDS since the epoch, not seconds: {stamp} is below the \
         {PLAUSIBLE_MS_FLOOR} floor. A seconds-resolution stamp dates every meta to 1970, which \
         reads as `nothing has stored this for decades` and makes every hash eligible on its \
         first pass"
    );
    assert!(
        (before..=after).contains(&stamp),
        "the stamp must be taken DURING the call: expected {before} <= {stamp} <= {after}"
    );
}

/// Storing the SAME bytes under the same hash again increments the count and
/// re-stamps; the blob is untouched.
///
/// Two contracts in one arm because they are the same event seen from two
/// sides. The count is `HINCRBY`, not a re-`SET`: the pre-flight `+1` is paid
/// per call and handed back by the commit that justifies it (b.qpf), so a second
/// upload that collapsed to `1` would leave the first reference unaccounted.
/// The stamp is plain `HSET`, not `HSETNX`: the reconciler wants the MOST RECENT
/// pre-store, and an `HSETNX` would pin the first store a hash ever saw and let
/// a hash re-uploaded a thousand times since read as ancient.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_store_of_the_same_hash_increments_the_count_and_re_stamps() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let hash = test_hash(0x11);
    let hex = sha256_to_hex(&hash);
    let payload = b"b.rej repeated payload".as_slice();

    store
        .store_values(&[(&hash, payload)])
        .await
        .expect("the first store must succeed");
    let first_stamp = parsed_stamp(&raw_stamp(&tv, &hex).await.expect("the first store stamps"));

    store
        .store_values(&[(&hash, payload)])
        .await
        .expect("the second store must succeed");
    let second_stamp = parsed_stamp(
        &raw_stamp(&tv, &hex)
            .await
            .expect("the second store re-stamps"),
    );

    assert_eq!(
        ref_count(&tv, &hex).await,
        Some(2),
        "the pre-flight increment is per CALL — at-least-once, not exactly-once. A second store \
         collapsing back to 1 would drop a reference a commit is entitled to hand back"
    );
    assert!(
        second_stamp >= first_stamp,
        "the stamp must move forward (or stand still within one millisecond), never stick at the \
         first store's value: `HSET`, never `HSETNX`. Got {first_stamp} then {second_stamp}"
    );
    assert_eq!(
        data_bytes(&tv, &hex).await.as_deref(),
        Some(payload),
        "and the content-addressed blob is unchanged"
    );
}

/// Two DISTINCT hashes in one call each get exactly one increment and one stamp
/// — and a hash referenced twice within the same delta still gets exactly one.
///
/// Driven through the public `store_external_values` door rather than
/// `store_values` directly, so the dedup half of the contract is exercised by
/// the collector that production actually uses: `ref_count` counts LIVE OBJECTS
/// (b.qpf), so one upload referencing a hash from two rows owes ONE pre-flight
/// increment, and the commit that materialises those rows hands back nothing.
/// The field/time-sample split matters because the collector's dedup set has to
/// span both maps. That door also runs the upload content check, so each
/// reference's claim is its part's content hash and its size the part's length.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_upload_pays_exactly_one_increment_per_distinct_hash_and_stamps_each() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    // Distinct lengths so a part mix-up cannot pass a length-blind compare.
    let parts: [&[u8]; 2] = [b"shared-blob", b"solo-blob-longer"];

    let shared = ext_content_hash(parts[0]);
    let solo = ext_content_hash(parts[1]);
    let shared_hex = sha256_to_hex(&shared);
    let solo_hex = sha256_to_hex(&solo);

    // `shared` is referenced TWICE — once from a field, once from a time sample
    // — and `solo` once. Three references, two distinct hashes.
    let delta = OwnedDelta {
        set_fields: vec![SetField {
            section_id: 1,
            key_name: "color".to_string(),
            value_or_ext_hash: shared.to_vec(),
            ext_value_size: parts[0].len() as u64,
            ext_value_index: 1,
            set_order: 0,
        }],
        time_samples: vec![
            TimeSample {
                section_id: 1,
                time: 1.0,
                value_or_ext_hash: shared.to_vec(),
                ext_value_size: parts[0].len() as u64,
                ext_value_index: 1,
            },
            TimeSample {
                section_id: 1,
                time: 2.0,
                value_or_ext_hash: solo.to_vec(),
                ext_value_size: parts[1].len() as u64,
                ext_value_index: 2,
            },
        ],
        ..Default::default()
    };

    let before = valkey_now_ms(&tv).await;
    store_external_values(Some(&delta), &parts, &store)
        .await
        .expect("the pre-flight store must succeed");
    let after = valkey_now_ms(&tv).await;

    assert_eq!(
        ref_count(&tv, &shared_hex).await,
        Some(1),
        "the hash referenced from BOTH maps still owes exactly one pre-flight increment — the \
         dedup spans set_fields and time_samples, and a 2 here would be a surplus no commit \
         hands back"
    );
    assert_eq!(
        ref_count(&tv, &solo_hex).await,
        Some(1),
        "and the second distinct hash gets its own single increment"
    );

    for (hex, expected) in [(&shared_hex, parts[0]), (&solo_hex, parts[1])] {
        assert_eq!(
            data_bytes(&tv, hex).await.as_deref(),
            Some(expected),
            "each hash's own blob must land under its own key — a per-hash mix-up would store \
             one payload twice"
        );
        let stamp = parsed_stamp(&raw_stamp(&tv, hex).await.unwrap_or_else(|| {
            panic!("every hash in a multi-hash wave must be stamped; {hex} was not")
        }));
        assert!(
            (before..=after).contains(&stamp),
            "and each stamp is taken during the call: expected {before} <= {stamp} <= {after}"
        );
    }
}

/// A payload containing NUL bytes and invalid UTF-8 round-trips byte for byte.
///
/// The script's argv rides as `fred::types::Value::Bytes`. A `String` there
/// would be a correctness bug, not a typing preference: external values are
/// arbitrary binary (compressed USD, image data), and a lossy conversion would
/// corrupt the blob while leaving its content-hash key claiming otherwise —
/// content-addressed storage whose content does not match its address. It is
/// also why the payload cannot simply be `.to_string()`-ed at the call site.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_binary_payload_with_nul_and_invalid_utf8_round_trips_unchanged() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let hash = test_hash(0x30);
    let hex = sha256_to_hex(&hash);
    // A NUL, a lone continuation byte, a bare 0xFF (never valid UTF-8), a CR/LF
    // pair (the RESP framing bytes), and a high byte run.
    let payload: Vec<u8> = vec![
        0x00, 0x80, 0xFF, b'\r', b'\n', 0xC3, 0x28, 0x00, 0xFE, 0xFD, b'x',
    ];

    store
        .store_values(&[(&hash, payload.as_slice())])
        .await
        .expect("a binary payload must store");

    assert_eq!(
        data_bytes(&tv, &hex).await.as_deref(),
        Some(payload.as_slice()),
        "the blob must come back byte for byte — a UTF-8 lossy conversion anywhere on the write \
         path silently rewrites the content under its own content hash"
    );
    let loaded = store
        .load_values(&[&hash])
        .await
        .expect("the read path must return the blob");
    assert_eq!(loaded.len(), 1);
    assert_eq!(
        loaded[0].1, payload,
        "and the store's own reader must agree with the raw GET"
    );
}

// =============================================================================
// 2 — the GC boundary: the collect takes the stamp, a decrement never writes it
// =============================================================================

/// A decrement that reaches zero collects BOTH meta fields and the data key.
///
/// The stamp must not survive its meta. It is a per-hash fact about the last
/// pre-store, so a stamp left behind on a collected hash would date a meta that
/// a later store re-creates from scratch, and the reconciler would judge that
/// brand-new meta against an instant belonging to a previous life of the key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_gc_collect_removes_the_stamp_along_with_the_rest_of_the_meta() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let hash = test_hash(0x40);
    let hex = sha256_to_hex(&hash);

    store
        .store_values(&[(&hash, b"collect-me".as_slice())])
        .await
        .expect("the pre-flight store must succeed");
    assert!(raw_stamp(&tv, &hex).await.is_some(), "staged: meta stamped");

    store
        .schedule_release(&[ExtRelease::retirement(hash, 8001)])
        .await
        .expect("scheduling the release must succeed");
    let stats = process_pending_gc(
        &store,
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.processed, 1, "one owed decrement, one claim");
    assert_eq!(stats.deleted, 1, "and it reached zero, so it collected");

    let meta_exists: bool = tv
        .client()
        .pool
        .exists(tv.key_schema().ext_meta(&hex))
        .await
        .expect("EXISTS ext_meta must succeed");
    assert!(
        !meta_exists,
        "the whole meta goes at zero — `last_store_ms` included. The collect is the ONLY \
         interaction the GC script may have with the stamp"
    );
    assert_eq!(
        data_bytes(&tv, &hex).await,
        None,
        "and the blob goes with it"
    );
}

/// A decrement that does NOT reach zero leaves `last_store_ms` byte-for-byte
/// unchanged.
///
/// **This is the "GC never stamps" invariant, and it is load-bearing rather than
/// tidy.** The reconciler will only judge a meta whose stamp is a full grace
/// old. A decrement that re-stamped would refresh the field on every GC pass
/// over a busy hash, so the surplus the reconciler exists to drain would read as
/// perpetually too fresh to judge and would stand forever — the pass would
/// enumerate it, decline it, and never repair anything, with every other signal
/// looking healthy.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_decrement_short_of_zero_leaves_the_stamp_untouched() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let hash = test_hash(0x41);
    let hex = sha256_to_hex(&hash);

    // Two references, so the single release below cannot collect.
    for _ in 0..2 {
        store
            .store_values(&[(&hash, b"survives-the-decrement".as_slice())])
            .await
            .expect("the pre-flight store must succeed");
    }
    let before = raw_stamp(&tv, &hex).await.expect("staged: meta stamped");

    store
        .schedule_release(&[ExtRelease::retirement(hash, 8002)])
        .await
        .expect("scheduling the release must succeed");
    let stats = process_pending_gc(
        &store,
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.processed, 1);
    assert_eq!(
        stats.deleted, 0,
        "one reference still stands, so nothing may be collected"
    );

    assert_eq!(ref_count(&tv, &hex).await, Some(1), "the count went down");
    assert_eq!(
        raw_stamp(&tv, &hex).await.as_deref(),
        Some(before.as_str()),
        "and the stamp did NOT move. `gc_decrement_and_collect.lua` must never write \
         `last_store_ms`: a decrement that re-stamped would keep a busy hash perpetually too \
         fresh for the reconciler to judge, and its surplus would stand forever"
    );
}

// =============================================================================
// 3 — the failure shape, and the pool it no longer touches
// =============================================================================

/// A wrong-typed `ext_meta` key surfaces a permanent `Internal` refusal — and
/// the data write STANDS.
///
/// Two claims, and the second is a **pinned defect**, filed as **b.5yz**.
///
/// Valkey rolls back neither a script nor a `MULTI`, so a `WRONGTYPE` raised by
/// the `HINCRBY` aborts with the preceding `SET` already applied. That was true
/// of the transaction this replaced and is true of the script, which is exactly
/// why b.rej transcribed it rather than fixing it here: adding a refusal arm
/// would have been new behaviour smuggled into a transport change. The residue
/// is a standing `ext_data` with no `ext_meta`, invisible to the GC (which only
/// performs owed decrements) and to the reconciler (which enumerates
/// `ext:meta:*`) — a leak no pass can reach. b.5yz adds the in-script `TYPE`
/// pre-check; when it lands, this arm's second half is what it changes.
///
/// The classification is asserted as `Internal` because `WRONGTYPE` is not
/// transient: no retry fixes a foreign value sitting on the key, and answering
/// `UNAVAILABLE` would invite the client to re-upload the whole layer into the
/// same refusal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wrong_typed_meta_refuses_the_increment_and_leaves_the_data_write_standing() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let hash = test_hash(0x50);
    let hex = sha256_to_hex(&hash);
    let payload = b"half-written payload".as_slice();

    // A foreign STRING where the meta hash belongs.
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().ext_meta(&hex),
            "not-a-hash",
            None,
            None,
            false,
        )
        .await
        .expect("planting the wrong-typed key must succeed");

    let err = store
        .store_values(&[(&hash, payload)])
        .await
        .expect_err("a wrong-typed meta key must not be reported as a successful store");

    match err {
        ServiceError::Internal(msg) => {
            assert!(
                msg.contains("ext store_values"),
                "the refusal must name the operation so an operator can locate it; got: {msg}"
            );
            assert!(
                msg.contains("WRONGTYPE"),
                "and must carry Valkey's own diagnosis rather than flattening it to a generic \
                 failure — the remediation is `look at what is sitting on that key`; got: {msg}"
            );
        }
        other => panic!(
            "a wrong-typed key is PERMANENT: no retry can move a foreign value off it, so the \
             answer must be Internal rather than a retriable Unavailable/Aborted. Got: {other:?}"
        ),
    }

    // The pinned defect. b.5yz.
    assert_eq!(
        data_bytes(&tv, &hex).await.as_deref(),
        Some(payload),
        "PINNED CURRENT BEHAVIOUR (b.5yz), not an endorsement: the `SET` runs before the \
         `HINCRBY`, and Valkey rolls back neither a script nor the `MULTI` this replaced, so the \
         blob stands with no meta to account for it. When b.5yz's in-script TYPE pre-check lands \
         this becomes `None` — change it there, deliberately"
    );
}

/// `store_values` completes while ANOTHER task holds every `txn_pool` slot.
///
/// **This is the regression test for the move itself.** Until b.rej each
/// per-hash write took an exclusive `txn_pool` slot via `acquire_txn_for`, so
/// with the pool fully held this call would wait on the mutex for as long as the
/// holder lived — the whole starvation dynamic the ticket is about, and the
/// reason a `CreateLayer` carrying N blobs could queue every concurrent
/// `UpdateLayer` slow-path commit and `CreateSession` behind it. On the data
/// pool there is no slot to wait for.
///
/// The fixture is a pool of ONE, which makes "every slot" a single
/// `acquire_txn()` and removes any doubt about whether the wave merely fitted:
/// the store must not take a slot at all, because there is none to take. The
/// budget is a deadline, not a latency assertion — the healthy path returns in
/// milliseconds and the failing one never returns.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_store_completes_while_every_txn_pool_slot_is_held() {
    // A one-slot txn pool: `valkey_pool_size` sizes it (see
    // `ValkeyClient::connect`), so holding one checkout holds the pool.
    let tv = TestValkey::setup_with_pool_size(1).await;
    let store = ext_store(&tv);

    let hash = test_hash(0x60);
    let hex = sha256_to_hex(&hash);

    // Held for the whole of the store below. `acquire_txn` is the sanctioned
    // test-only form for taking a raw slot with no transaction of its own.
    let _held_slot = tv.client().acquire_txn().await;

    tokio::time::timeout(
        TXN_STARVED_STORE_BUDGET,
        store.store_values(&[(&hash, b"unblocked payload".as_slice())]),
    )
    .await
    .expect(
        "`store_values` waited on the txn pool. Post-b.rej the per-hash write is an EVALSHA on \
         the MULTIPLEXED data pool and takes no exclusive slot at all, so a fully-held txn pool \
         must not delay it by so much as a round trip — this is the starvation the ticket \
         removed",
    )
    .expect("and the store itself must succeed");

    assert_eq!(
        ref_count(&tv, &hex).await,
        Some(1),
        "and it really wrote: a call that returned early without storing would satisfy the \
         deadline above while doing nothing"
    );
    assert!(
        raw_stamp(&tv, &hex).await.is_some(),
        "including the stamp — the whole write happened on the data pool"
    );
}

// =============================================================================
// 4 — the unverified data write is SET-IF-ABSENT: `store_values` never
//     overwrites a stored copy
// =============================================================================

/// A store whose data key already holds a blob keeps that blob byte for byte
/// when handed DIFFERENT bytes for the same hash, and still pays exactly one
/// increment and one re-stamp.
///
/// **This is the regression test for b.682 sub-task (c), and the arm that
/// discriminates `SET … NX` from `SET`.** The script's data write used to be an
/// unconditional `SET`, so one truncated or mismatched upload of a hash
/// replaced the good blob every other layer referencing that hash reads. Every
/// other store test re-stores byte-IDENTICAL content, where the two forms are
/// indistinguishable, so a revert to the unconditional `SET` turns only this
/// arm red.
///
/// The second store calls `store_values` DIRECTLY, bypassing the upload content
/// check (`verify_external_values`) on purpose: the check keeps such bytes off
/// the default live path, but `LIVE_EDIT_EXT_VALUE_VERIFY=observe` hands them
/// to the store by design and the DR paths reach it ungated, so `store_values`
/// itself must not overwrite. The bytes are a truncated prefix of the good
/// value under the good value's own content hash, the shape a cut-off upload
/// produces. Their LENGTH differs from the stored blob's, so this arm is also
/// what turns red if the verified store's length heal (§ 5) ever fires for an
/// unverified `store_values`.
///
/// The count and the stamp are asserted alongside because a skipped write is
/// not a skipped store: `ref_count` counts live objects, not stored copies, so
/// the call still owes its pre-flight `+1`, and the stamp still refreshes so
/// the reconciler does not judge a hash that is being re-uploaded.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_store_over_an_existing_data_blob_keeps_it_byte_for_byte_and_increments_once() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let good: &[u8] = b"b.682 the first, verified copy of this external value";
    // A strict prefix: different LENGTH as well as different bytes, so neither
    // a length-blind compare nor a partial overwrite could pass this.
    let truncated = &good[..good.len() - 9];
    let hash = ext_content_hash(good);
    let hex = sha256_to_hex(&hash);

    store
        .store_values(&[(&hash, good)])
        .await
        .expect("the first store must succeed");
    let count_before = ref_count(&tv, &hex)
        .await
        .expect("staged: the meta carries a count");
    let stamp_before = parsed_stamp(&raw_stamp(&tv, &hex).await.expect("staged: meta stamped"));

    store
        .store_values(&[(&hash, truncated)])
        .await
        .expect("a store over an existing blob must succeed, writing nothing to the blob");

    assert_eq!(
        data_bytes(&tv, &hex).await.as_deref(),
        Some(good),
        "the unverified data write is `SET … NX`: a later `store_values` of the same hash, here \
         with truncated bytes, must leave the first copy byte for byte. An unconditional `SET` lets one bad upload \
         replace the blob every layer referencing this hash reads"
    );
    assert_eq!(
        ref_count(&tv, &hex).await,
        Some(count_before + 1),
        "and the store still pays exactly one `+1`: not zero (the skipped write is not a skipped \
         store) and not two"
    );
    let stamp_after = parsed_stamp(&raw_stamp(&tv, &hex).await.expect("the store must re-stamp"));
    assert!(
        stamp_after >= stamp_before,
        "and it re-stamps, so the reconciler cannot judge a hash that is being re-uploaded. Got \
         {stamp_before} then {stamp_after}"
    );
}

/// A WRONG-TYPED data key — a list where the blob belongs — is left standing
/// by an unverified `store_values`, and the store still succeeds and pays its
/// one increment and stamp.
///
/// **Observed behaviour, pinned as observed.** `SET … NX` treats a key of any
/// type as present: it writes nothing and raises no `WRONGTYPE`, and the
/// `HINCRBY` that follows addresses the (untouched) meta key. So unverified
/// stores of this hash keep succeeding, and reads of it surface `WRONGTYPE`
/// until an operator deletes the key or a verified store of the hash replaces
/// it (§ 5; `scripts/ext_store_value.lua`, SDD §11.1). This arm is what turns
/// red if the verified store's type heal ever fires for `store_values`.
///
/// Contrast the wrong-typed **meta** arm in § 3: there the offending key is the
/// one `HINCRBY` addresses, and `HINCRBY` does raise `WRONGTYPE`, so that call
/// fails. The asymmetry is a property of which command touches which key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_store_over_a_wrong_typed_data_key_leaves_it_standing_and_increments_once() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let payload = b"b.682 payload over a list".as_slice();
    let hash = ext_content_hash(payload);
    let hex = sha256_to_hex(&hash);
    let data_key = tv.key_schema().ext_data(&hex);
    let planted = vec!["not", "a", "blob"];

    // A LIST where the content-addressed blob belongs.
    tv.client()
        .pool
        .rpush::<(), _, _>(&data_key, planted.clone())
        .await
        .expect("planting the wrong-typed data key must succeed");

    store.store_values(&[(&hash, payload)]).await.expect(
        "a wrong-typed DATA key must not fail the store: `SET … NX` sees a key of any type as \
         present and writes nothing, and the `HINCRBY` that follows addresses the (untouched) \
         meta key. If this ever fails instead, every upload of the hash fails with it, and the \
         write-mode rationale in `scripts/ext_store_value.lua` is the thing to change",
    );

    assert_eq!(
        key_type(&tv, &data_key).await,
        "list",
        "the list must still be standing: an unverified set-if-absent write never replaces an \
         existing key, whatever its type"
    );
    let still: Vec<String> = tv
        .client()
        .pool
        .lrange(&data_key, 0, -1)
        .await
        .expect("LRANGE of the planted list must succeed");
    assert_eq!(still, planted, "and it is untouched");
    assert_eq!(
        ref_count(&tv, &hex).await,
        Some(1),
        "with the ordinary single pre-flight increment"
    );
    assert!(
        raw_stamp(&tv, &hex).await.is_some(),
        "and the ordinary stamp — the skipped write takes no special path"
    );
}

// =============================================================================
// 5 — the VERIFIED store heals a damaged copy; nothing else does
// =============================================================================
//
// Under `enforce`, `store_external_values` stores an upload that passed the
// content check through `store_verified_values`, which sets the script's
// ARGV[2] verified flag. When the `SET … NX` was skipped, that flag replaces an
// existing data key that is not a string, or whose length differs from the
// verified bytes. A same-length string is never replaced, because the script
// compares no bytes. Under `observe` the same call goes through `store_values`
// and never heals. § 4's two arms pin that `store_values` never heals when
// called directly.
//
// Every arm here enters through `store_external_values` with a correct upload,
// as the live path does, on a store built with an explicit
// `with_ext_value_verify` mode. The damaged copy is planted either by an
// unverified `store_values`, which stands in for a store made before the gate
// existed, or by a raw write of another key type.

/// Length of [`good_value`]: one content-hash block.
const HEAL_GOOD_LEN: usize = 4096;

/// The truncated prefix length: 3096 of 4096 bytes, the shape the b.682
/// reproducer stored under the full value's hash.
const HEAL_TRUNCATED_LEN: usize = 3096;

/// A deterministic [`HEAL_GOOD_LEN`]-byte value. Values with different `seed`s
/// differ in every byte, so their content hashes differ.
fn good_value(seed: u8) -> Vec<u8> {
    (0..HEAL_GOOD_LEN)
        .map(|i| ((i % 251) as u8) ^ seed)
        .collect()
}

/// An upload referencing each of `parts` once, as a conforming client sends
/// it. Each claim is the part's content hash and each size is the part's
/// length, so the upload passes the content check in either mode.
fn checked_upload(parts: &[&[u8]]) -> OwnedDelta {
    OwnedDelta {
        set_fields: parts
            .iter()
            .enumerate()
            .map(|(i, part)| SetField {
                section_id: 1,
                key_name: format!("ext{i}"),
                value_or_ext_hash: ext_content_hash(part).to_vec(),
                ext_value_size: part.len() as u64,
                ext_value_index: u32::try_from(i + 1).expect("a handful of parts"),
                set_order: 0,
            })
            .collect(),
        ..Default::default()
    }
}

/// A store whose upload content check runs in `mode`.
fn ext_store_in(tv: &TestValkey, mode: ExtValueVerifyMode) -> ValkeyExternalValueStore {
    ext_store(tv).with_ext_value_verify(mode)
}

/// Assert the string under `ext_data:{hex}` is exactly `expected`.
///
/// Compares by hand rather than with `assert_eq!` on the bytes: the values here
/// are 4 KiB, and a failing `assert_eq!` would print both in full. This
/// reports the two lengths and the first differing offset instead.
async fn assert_stored(tv: &TestValkey, hex: &str, expected: &[u8], why: &str) {
    let stored = data_bytes(tv, hex)
        .await
        .unwrap_or_else(|| panic!("{why}: no data key is stored under {hex}"));
    let first_difference = stored.iter().zip(expected).position(|(s, e)| s != e);
    assert!(
        stored == expected,
        "{why}: stored {} bytes, expected {} bytes; first differing offset {first_difference:?}",
        stored.len(),
        expected.len()
    );
}

/// Under `enforce`, a correct upload of a hash whose stored blob has the WRONG
/// LENGTH replaces the blob with the correct bytes, pays exactly one increment
/// and refreshes the stamp. Two plants, each under the good value's hash in its
/// own fixture: a truncated prefix (shorter) and the good value with bytes
/// appended (longer).
///
/// **This is the regression test for the verified heal (b.682 post-completion
/// finding 2).** Before the heal, `SET … NX` kept the damaged copy for good,
/// and every layer referencing the hash read it, however often a correct
/// upload arrived. The truncated prefix is the shape the pre-gate bug left.
/// The longer copy is the shape an `observe` upload can leave, since `observe`
/// stores bytes of any length as the first copy under a hash they do not
/// match. Both directions are planted because the heal replaces on any length
/// difference: a length test narrowed to "shorter than" passes the truncated
/// plant and fails only the longer one. This arm turns red if the script never
/// heals on ARGV[2] or if `store_external_values` stores an `enforce` upload
/// unverified.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_verified_enforce_upload_replaces_a_wrong_length_blob_and_increments_once() {
    let good = good_value(0x00);
    let hash = ext_content_hash(&good);
    let hex = sha256_to_hex(&hash);
    let mut longer = good.clone();
    longer.extend_from_slice(b"bytes appended past the verified length");

    for (plant, damaged) in [
        ("shorter", &good[..HEAL_TRUNCATED_LEN]),
        ("longer", longer.as_slice()),
    ] {
        // A fixture per plant: both copies sit under `good`'s hash, so each
        // needs its own key prefix.
        let tv = TestValkey::setup().await;
        let store = ext_store_in(&tv, ExtValueVerifyMode::Enforce);

        store
            .store_values(&[(&hash, damaged)])
            .await
            .unwrap_or_else(|e| panic!("planting the {plant} copy must succeed: {e:?}"));
        let count_before = ref_count(&tv, &hex)
            .await
            .unwrap_or_else(|| panic!("staged ({plant}): the meta carries a count"));
        let stamp_before = parsed_stamp(
            &raw_stamp(&tv, &hex)
                .await
                .unwrap_or_else(|| panic!("staged ({plant}): meta stamped")),
        );

        store_external_values(Some(&checked_upload(&[&good])), &[&good], &store)
            .await
            .unwrap_or_else(|e| {
                panic!("a correct upload must be admitted under enforce ({plant}): {e:?}")
            });

        assert_stored(
            &tv,
            &hex,
            &good,
            &format!(
                "a verified store must replace a stored blob whose length differs from the \
                 verified bytes in either direction: the {plant} copy is damage, and the checked \
                 upload is the hash's content"
            ),
        )
        .await;
        assert_eq!(
            ref_count(&tv, &hex).await,
            Some(count_before + 1),
            "the heal pays exactly the one pre-flight `+1` every store pays ({plant})"
        );
        let stamp_after = parsed_stamp(
            &raw_stamp(&tv, &hex)
                .await
                .unwrap_or_else(|| panic!("the store must re-stamp ({plant})")),
        );
        assert!(
            stamp_after >= stamp_before,
            "and it re-stamps as every store does ({plant}). Got {stamp_before} then \
             {stamp_after}"
        );
    }
}

/// Under `enforce`, a correct upload of a hash whose data key holds a LIST or
/// a HASH replaces that key with the correct bytes as a string, pays exactly
/// one increment and stamps.
///
/// **This is the regression test for the type half of the verified heal.**
/// Before the heal, § 4's wrong-typed arm held for every caller: the key stood
/// and every read of the hash surfaced `WRONGTYPE` until an operator deleted
/// it. Both non-string types are planted to show the check is "not a string"
/// rather than one specific type. No meta is planted, so the count goes from
/// absent to 1.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_verified_enforce_upload_replaces_a_wrong_typed_data_key_with_the_blob() {
    let tv = TestValkey::setup().await;
    let store = ext_store_in(&tv, ExtValueVerifyMode::Enforce);
    let under_list = good_value(0x01);
    let under_hash = good_value(0x02);
    let list_hex = sha256_to_hex(&ext_content_hash(&under_list));
    let hash_hex = sha256_to_hex(&ext_content_hash(&under_hash));
    let list_key = tv.key_schema().ext_data(&list_hex);
    let hash_key = tv.key_schema().ext_data(&hash_hex);

    tv.client()
        .pool
        .rpush::<(), _, _>(&list_key, vec!["not", "a", "blob"])
        .await
        .expect("planting the list must succeed");
    tv.client()
        .pool
        .hset::<(), _, _>(&hash_key, [("not", "a blob")])
        .await
        .expect("planting the hash must succeed");
    assert_eq!(key_type(&tv, &list_key).await, "list", "staged: a list");
    assert_eq!(key_type(&tv, &hash_key).await, "hash", "staged: a hash");

    let parts: [&[u8]; 2] = [&under_list, &under_hash];
    store_external_values(Some(&checked_upload(&parts)), &parts, &store)
        .await
        .expect("a correct upload must be admitted under enforce");

    for (planted, hex, key, good) in [
        ("list", &list_hex, &list_key, &under_list),
        ("hash", &hash_hex, &hash_key, &under_hash),
    ] {
        assert_eq!(
            key_type(&tv, key).await,
            "string",
            "a verified store must replace a data key that is not a string; the planted \
             {planted} is still standing"
        );
        assert_stored(
            &tv,
            hex,
            good,
            &format!("the replacement must be the verified bytes (planted {planted})"),
        )
        .await;
        assert_eq!(
            ref_count(&tv, hex).await,
            Some(1),
            "with exactly one pre-flight increment (planted {planted})"
        );
        assert!(
            raw_stamp(&tv, hex).await.is_some(),
            "and the stamp (planted {planted})"
        );
    }
}

/// Under `observe`, a correct upload heals nothing: a truncated blob and a
/// wrong-typed data key under its hashes both stay as they were, and each hash
/// still pays one increment.
///
/// `observe` stores through `store_values` because the check may have failed
/// and been let through. A verified flag on that path would let a mismatched
/// upload replace a good blob of a different length. These bytes happen to be
/// correct, but `observe` does not use that to heal. This arm turns red if
/// `store_external_values` stores an `observe` upload verified.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_observe_upload_leaves_a_truncated_blob_and_a_wrong_typed_key_standing() {
    let tv = TestValkey::setup().await;
    let store = ext_store_in(&tv, ExtValueVerifyMode::Observe);
    let over_truncated = good_value(0x03);
    let over_list = good_value(0x04);
    let truncated = &over_truncated[..HEAL_TRUNCATED_LEN];
    let truncated_hash = ext_content_hash(&over_truncated);
    let truncated_hex = sha256_to_hex(&truncated_hash);
    let list_hex = sha256_to_hex(&ext_content_hash(&over_list));
    let list_key = tv.key_schema().ext_data(&list_hex);
    let planted = vec!["not", "a", "blob"];

    store
        .store_values(&[(&truncated_hash, truncated)])
        .await
        .expect("planting the truncated copy must succeed");
    tv.client()
        .pool
        .rpush::<(), _, _>(&list_key, planted.clone())
        .await
        .expect("planting the list must succeed");
    let truncated_count_before = ref_count(&tv, &truncated_hex)
        .await
        .expect("staged: the meta carries a count");

    let parts: [&[u8]; 2] = [&over_truncated, &over_list];
    store_external_values(Some(&checked_upload(&parts)), &parts, &store)
        .await
        .expect("observe must admit a correct upload");

    assert_stored(
        &tv,
        &truncated_hex,
        truncated,
        "observe stores unverified, so the truncated copy must stand even under correct bytes",
    )
    .await;
    assert_eq!(
        ref_count(&tv, &truncated_hex).await,
        Some(truncated_count_before + 1),
        "and the hash still pays its one pre-flight increment"
    );
    assert_eq!(
        key_type(&tv, &list_key).await,
        "list",
        "observe stores unverified, so the wrong-typed key must stand too"
    );
    let still: Vec<String> = tv
        .client()
        .pool
        .lrange(&list_key, 0, -1)
        .await
        .expect("LRANGE of the planted list must succeed");
    assert_eq!(still, planted, "and it is untouched");
    assert_eq!(
        ref_count(&tv, &list_hex).await,
        Some(1),
        "and that hash still pays its one pre-flight increment"
    );
}

/// Under `enforce`, a correct upload over a WRONG blob of the SAME length
/// leaves the wrong blob standing and pays one increment.
///
/// This is a deliberate gap, not a defect. The heal checks type and length
/// only, because a byte compare would read the whole stored blob on Valkey's
/// command thread on every store of an existing hash. A same-length string is
/// trusted (`scripts/ext_store_value.lua`, SDD §11.1). This arm turns red if
/// the heal starts comparing bytes or overwriting unconditionally.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_verified_enforce_upload_leaves_a_same_length_wrong_blob_standing() {
    let tv = TestValkey::setup().await;
    let store = ext_store_in(&tv, ExtValueVerifyMode::Enforce);
    let good = good_value(0x05);
    // Every byte differs, the length does not.
    let wrong: Vec<u8> = good.iter().map(|b| !b).collect();
    let hash = ext_content_hash(&good);
    let hex = sha256_to_hex(&hash);

    store
        .store_values(&[(&hash, wrong.as_slice())])
        .await
        .expect("planting the same-length wrong copy must succeed");
    let count_before = ref_count(&tv, &hex)
        .await
        .expect("staged: the meta carries a count");

    store_external_values(Some(&checked_upload(&[&good])), &[&good], &store)
        .await
        .expect("a correct upload must be admitted under enforce");

    assert_stored(
        &tv,
        &hex,
        &wrong,
        "the verified heal compares type and length only: a stored string of the verified \
         length is left as it stands",
    )
    .await;
    assert_eq!(
        ref_count(&tv, &hex).await,
        Some(count_before + 1),
        "and the store pays its one pre-flight increment"
    );
}
