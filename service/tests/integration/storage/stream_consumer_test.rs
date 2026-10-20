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

// Integration tests for the layer stream consumer (src/storage/notifications.rs).
//
// These tests verify delivery through the stream multiplexer end-to-end
// against a real Valkey: XADD on the layer stream → consume_layer_stream → recv
// → assert the (stream_id, LayerNotification) pair comes back with the expected
// content.
//
// Requires a running Valkey. Address is read from TEST_VALKEY_URL, defaulting
// to redis://127.0.0.1:6379.

use std::env;
use std::time::{Duration, Instant};

use fred::interfaces::StreamsInterface;

use live_edit::config::{AuthProtocol, Config};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::notifications::{consume_layer_stream, LayerNotification, StreamCursor};
use live_edit::storage::test_support::TestValkey;

/// XADD a `payload=<value>` entry on the layer stream for `object_id` and
/// return the stream ID Valkey assigned.
async fn xadd_payload(tv: &TestValkey, object_id: u64, id: &str, payload: &str) -> String {
    let stream_key = tv.key_schema().layer_stream(object_id);
    tv.client()
        .pool
        .xadd::<String, _, _, _, _>(stream_key, false, None, id, vec![("payload", payload)])
        .await
        .expect("XADD must succeed")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consume_from_tail_yields_xadd_entry_with_stream_id_and_notification() {
    let tv = TestValkey::setup().await;
    let object_id: u64 = 42;

    // Start the consumer BEFORE XADD so the entry lands while the consumer's
    // XREAD BLOCK is outstanding. This exercises the wake-up path. `0-0` is the
    // cursor a production caller derives for a stream with nothing in it yet
    // (`grpc::subscribe::stream_cursor_at`); it replays from the beginning,
    // which on an empty stream means "everything from here".
    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("0-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consume_layer_stream must succeed");

    // Give the reader a moment to issue its XREAD so that the subsequent XADD
    // exercises the blocking wake-up path.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Append an entry with a deterministic ID — same format the write path
    // uses for Update entries: `{seq}-0`. It is `1-0`, the commit right after
    // the `0-0` cursor: a first entry further above it, on a stream holding
    // nothing below that entry, is a trim past the cursor and would end the
    // subscription `stream_gap` instead.
    let payload = "update|42|1|client-abc|1000000|2000000";
    let expected_id = xadd_payload(&tv, object_id, "1-0", payload).await;
    assert_eq!(expected_id, "1-0", "XADD must honor the explicit ID");

    // Expect recv to return within a reasonable bound; the XREAD BLOCK
    // wake-up from XADD should be near-instant.
    let pair = tokio::time::timeout(Duration::from_secs(5), consumer.recv())
        .await
        .expect("recv must not time out")
        .expect("consumer must yield the notification");

    let (stream_id, notification) = pair;
    assert_eq!(stream_id, "1-0", "stream_id must match the XADD id");
    assert_eq!(
        notification,
        LayerNotification::Update {
            object_id: 42,
            sequence_num: 1,
            client_id: "client-abc".to_string(),
            client_ts: 1_000_000,
            server_ts: 2_000_000,
            relay_ompf: None,
        },
    );

    consumer.unsubscribe().await;
}

/// A header-derived cursor is DURABLE: a commit that lands before the consumer
/// is even opened is still delivered, because `{sequence_num}-0` means the same
/// thing whenever the XREAD executes.
///
/// This is the loss window b.9uq round 4 closed, in its smallest reproducible
/// form. Production ordering is: read the layer header (`sequence_num = 1`) →
/// derive `From("1-0")` (`grpc::subscribe::stream_cursor_at`) → send full state
/// → open the consumer. Any commit landing anywhere in that window has a stream
/// ID above the cursor and is owed to the subscriber. The pre-round-4 `$` open
/// resolved at XREAD-execution time instead, so a commit landing before the
/// first (or any retried) XREAD was skipped permanently and the subscriber went
/// silent with no error anywhere.
///
/// The fixture writes `2-0` BEFORE opening the consumer, so no timing window is
/// being raced: `$` semantics would lose this entry deterministically, a durable
/// `{seq}-0` cursor delivers it deterministically.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consume_from_cursor_starts_strictly_after_given_id() {
    let tv = TestValkey::setup().await;
    let object_id: u64 = 99;

    // Pre-populate the stream with two entries before starting the consumer:
    // `1-0` is the commit the (hypothetical) header pin was taken at, `2-0` is
    // the commit that lands before the consumer's first XREAD is ever issued.
    let _ = xadd_payload(&tv, object_id, "1-0", "update|99|1||0|0").await;
    let _ = xadd_payload(&tv, object_id, "2-0", "update|99|2||0|0").await;

    // Resume from "1-0" — XREAD treats the given ID as exclusive, so we
    // should get entry 2-0 but not entry 1-0.
    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("1-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consume_layer_stream must succeed");

    let (stream_id, notification) = tokio::time::timeout(Duration::from_secs(5), consumer.recv())
        .await
        .expect(
            "recv must not time out — a commit appended BEFORE the consumer's first XREAD \
             must still be delivered from a durable `{seq}-0` cursor",
        )
        .expect("consumer must yield the notification at 2-0");

    assert_eq!(stream_id, "2-0");
    assert!(
        matches!(
            notification,
            LayerNotification::Update {
                sequence_num: 2,
                ..
            },
        ),
        "expected entry 2-0 (sequence_num=2), got: {notification:?}",
    );

    // …and the cursor is exclusive at its own ID: the commit the pin was taken
    // at is NOT re-delivered. Over-delivery here is the mirror failure — it is
    // what the steady-state `sequence_num <= tracked_version` filter would then
    // have to absorb, and the coalesced-snapshot regime has no such filter for
    // entries its diff already folded in (`catch_up_in_window_snapshot`).
    let extra = consumer.try_recv().await;
    assert!(
        extra.is_none(),
        "the pinned commit `1-0` must not be replayed — a header-derived cursor resumes \
         STRICTLY after the sequence the caller already served: {extra:?}"
    );

    consumer.unsubscribe().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consume_from_tail_does_not_replay_preexisting_entries() {
    // A concrete `From(id)` cursor is exclusive on XREAD. Per Valkey semantics
    // delivers only entries appended after the XREAD call reaches the server,
    // so any entries already on the stream at subscription time must not be
    // replayed. This test pre-populates two entries, starts a consumer pinned
    // at the tail (`2-0` — exactly what a production caller derives from a
    // header at `sequence_num = 2`), appends a third, and asserts that only the
    // third arrives.
    let tv = TestValkey::setup().await;
    let object_id: u64 = 123;

    // Pre-populate the stream with two entries BEFORE the consumer starts.
    let _ = xadd_payload(&tv, object_id, "1-0", "update|123|1||0|0").await;
    let _ = xadd_payload(&tv, object_id, "2-0", "update|123|2||0|0").await;

    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("2-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consume_layer_stream must succeed");

    // Settle before appending so the post-subscription entry exercises the
    // blocking wake-up path rather than being picked up by the first XREAD's
    // immediate return. Not load-bearing for the assertion: the cursor is a
    // durable `2-0`, so `3-0` is owed to the consumer whenever it lands.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Append the only entry the consumer should observe.
    let expected_id = xadd_payload(&tv, object_id, "3-0", "update|123|3||0|0").await;
    assert_eq!(expected_id, "3-0");

    let (stream_id, notification) = tokio::time::timeout(Duration::from_secs(5), consumer.recv())
        .await
        .expect("recv must not time out")
        .expect("consumer must yield the post-subscription notification");

    // First yielded entry must be 3-0 — the two pre-existing entries were not
    // replayed despite being on the stream when the consumer started.
    assert_eq!(
        stream_id, "3-0",
        "a cursor pinned at the header sequence must skip everything at or below it"
    );
    assert!(
        matches!(
            notification,
            LayerNotification::Update {
                sequence_num: 3,
                ..
            },
        ),
        "expected sequence_num=3, got: {notification:?}",
    );

    // Drain anything else that might be buffered. There should be nothing: the
    // cursor is exclusive at `2-0` and only one entry was appended after it.
    let extra = consumer.try_recv().await;
    assert!(
        extra.is_none(),
        "the header-pinned cursor leaked a pre-existing entry: {extra:?}",
    );

    consumer.unsubscribe().await;
}

/// Shared body for the `deleted`-sentinel cases below.
///
/// `seeded_commits` are XADDed with explicit `{seq}-0` IDs before the consumer
/// opens at `cursor`; the sentinel is then appended with an auto ID (`*`), which
/// is what DeleteLayer and the CreateLayer-overwrite path actually use. The
/// sentinel is XADDed AFTER the consumer starts (the pre-existing seeded
/// commits are what prove the cursor was honoured), so the assertion covers the
/// blocking wake-up path as well as the ID ordering.
async fn assert_deleted_sentinel_reaches_consumer(
    object_id: u64,
    seeded_commits: &[u64],
    cursor: &str,
    case: &str,
) {
    let tv = TestValkey::setup().await;

    for seq in seeded_commits {
        let _ = xadd_payload(
            &tv,
            object_id,
            &format!("{seq}-0"),
            &format!("update|{object_id}|{seq}||0|0"),
        )
        .await;
    }

    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From(cursor.to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consume_layer_stream must succeed");

    tokio::time::sleep(Duration::from_millis(100)).await;

    // DeleteLayer / the CreateLayer-overwrite path write a `deleted` sentinel
    // with an auto ID (`*`), i.e. `<wall-clock-ms>-<n>`.
    let stream_id = xadd_payload(&tv, object_id, "*", "deleted").await;

    let (got_id, notification) = tokio::time::timeout(Duration::from_secs(5), consumer.recv())
        .await
        .unwrap_or_else(|_| {
            panic!(
                "{case}: the `deleted` sentinel never reached a consumer pinned at \
                 `{cursor}`. Subscribers detect DeleteLayer and CreateLayer-overwrite \
                 through this entry — if a header-derived cursor could sort above the \
                 auto-assigned ID, an overwritten layer's subscribers would never \
                 re-bootstrap onto the new generation and would silently tail a dead \
                 object (b.9uq round 4)"
            )
        })
        .unwrap_or_else(|| panic!("{case}: the subscription ended instead of delivering"));

    assert_eq!(
        got_id, stream_id,
        "{case}: stream_id round-trips through consumer"
    );
    assert_eq!(notification, LayerNotification::Deleted, "{case}");

    consumer.unsubscribe().await;
}

/// Brand-new / empty-stream case: cursor `0-0`, nothing seeded.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consume_deleted_sentinel_returns_deleted_variant() {
    assert_deleted_sentinel_reaches_consumer(7, &[], "0-0", "empty-stream cursor `0-0`").await;
}

/// Header-pinned case (b.9uq round 4): the consumer resumes at `{seq}-0`
/// derived from a layer header, and the sentinel still arrives.
///
/// This is the one ordering assumption the round-4 design carries that the
/// commit path does not: commit entries get explicit `{sequence_num}-0` IDs,
/// but the `deleted` sentinel gets an auto ID, so "the sentinel is above every
/// header pin" rests on Valkey's auto IDs being wall-clock-ms based (~1.7e12
/// today) and sequence numbers being far smaller. It holds by many orders of
/// magnitude in practice, and it is precisely the kind of implicit assumption
/// that is worth an explicit, cheap regression guard — a header pin that sorted
/// ABOVE the sentinel would leave overwritten layers' subscribers tailing a dead
/// object forever, with no error surfaced anywhere.
///
/// End-to-end coverage of the same property through the gRPC path exists
/// (`grpc::subscribe_regimes_test::overwrite_during_subscription_rebootstraps_on_new_object`,
/// whose subscriber's consumer is header-pinned post-round-4). This storage-level
/// case is deliberately kept as well: it isolates the cursor-vs-sentinel ID
/// ordering from the whole rebootstrap chain, so a failure names the cause
/// directly instead of surfacing as "the subscriber never re-bootstrapped".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consume_deleted_sentinel_reaches_a_header_pinned_consumer() {
    assert_deleted_sentinel_reaches_consumer(
        8,
        &[1, 2, 3],
        "3-0",
        "header-pinned cursor `3-0` (auto-ID sentinel must sort above it)",
    )
    .await;
}

/// b.gze regression, re-homed on the XREAD path (b.9uq round 4).
///
/// Every real update entry carries the writer's raw inbound OMPF on a separate
/// binary `ompf` field alongside the UTF-8 `payload`. The consumer must decode
/// stream field VALUES as `Bytes`: a `String` decode fails the WHOLE row on the
/// OMPF bytes' invalid UTF-8, so the entry — payload and all — is dropped and
/// the subscriber silently stalls.
///
/// The fixture uses bytes that are DELIBERATELY invalid UTF-8 (a lone 0x80
/// continuation byte and a bare 0xFF, neither legal in any UTF-8 sequence)
/// rather than a real OMPF buffer, so the test cannot start passing vacuously
/// if the container format's byte distribution changes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consumer_delivers_an_entry_whose_ompf_field_is_not_valid_utf8() {
    use bytes::Bytes;

    let tv = TestValkey::setup().await;
    let object_id: u64 = 4242;

    let ompf_bytes: Vec<u8> = vec![0x4f, 0x4d, 0x50, 0x46, 0x00, 0x80, 0xff, 0xfe, 0x01];
    assert!(
        std::str::from_utf8(&ompf_bytes).is_err(),
        "fixture precondition: the relay bytes must NOT be valid UTF-8, otherwise a \
         String-decoding consumer would pass this test"
    );

    let stream_key = tv.key_schema().layer_stream(object_id);
    let fields: Vec<(&str, fred::types::Value)> = vec![
        (
            "payload",
            format!("update|{object_id}|5|writer|10|20").into(),
        ),
        (
            "ompf",
            fred::types::Value::Bytes(Bytes::from(ompf_bytes.clone())),
        ),
    ];
    let _: String = tv
        .client()
        .pool
        .xadd(&stream_key, false, None, "5-0", fields)
        .await
        .expect("XADD with a binary ompf field must succeed");

    // From `4-0`, so the entry is the commit right after the cursor: a `0-0`
    // cursor under a stream whose first entry is `5-0` reads as a trim past
    // it, and the subscription would end `stream_gap` before the entry.
    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("4-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consume_layer_stream must succeed");

    let (stream_id, notification) = tokio::time::timeout(Duration::from_secs(5), consumer.recv())
        .await
        .expect(
            "recv must not time out — an entry carrying a non-UTF-8 `ompf` field must still \
             be delivered; a String field decode would fail the whole row and the \
             subscriber would stall with nothing logged at the entry level (b.gze)",
        )
        .expect("consumer must yield the notification");

    assert_eq!(stream_id, "5-0");
    assert_eq!(
        notification,
        LayerNotification::Update {
            object_id,
            sequence_num: 5,
            client_id: "writer".to_string(),
            client_ts: 10,
            server_ts: 20,
            relay_ompf: Some(Bytes::from(ompf_bytes)),
        },
        "the binary `ompf` field must be attached to the notification verbatim — \
         steady-state subscribers relay exactly these bytes, so any lossy round-trip \
         here corrupts the payload the Kit client applies"
    );

    consumer.unsubscribe().await;
}

// ── Unregistration, churn and admission (the multiplexer's lifecycle) ──────
//
// A subscription is an in-memory registration on a shared reader. Dropping the
// consumer unregisters it synchronously (the reader applies the removal before
// its next XREAD) and must never block a runtime worker; the admission permit
// the RPC holds is independent of it.

/// Build a ValkeyClient with caller-chosen reader and admission sizes.
async fn connect_with_subscriber_pool(
    readers: u32,
    max_subscriptions: usize,
) -> (ValkeyClient, KeySchema, String) {
    let valkey_url =
        env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let prefix = format!("test_{}:", uuid::Uuid::new_v4());
    let key_schema = KeySchema::new(&prefix);

    let config = Config {
        catchup_snapshot_enabled: false,
        port: 0,
        metrics_port: 0,
        valkey_url,
        valkey_pool_size: 2,
        subscriber_pool_size: readers,
        subscriber_admission_wait_ms: 2_000,
        subscriber_max_subscriptions: max_subscriptions,
        subscriber_max_concurrent_catchups: 64,
        subscriber_admission_max_waiters: None,
        maintenance_pool_size: 1,
        catchup_snapshot_read_concurrency:
            live_edit::config::CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT,
        catchup_fenced_read_concurrency: live_edit::config::CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT,
        catchup_build_memory_budget_mib: live_edit::config::CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
        snapshot_pool_size: 1,
        snapshot_atomic_read_max_items: live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
        snapshot_atomic_read_max_bytes: live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
        structural_cache_max_mib: live_edit::config::STRUCTURAL_CACHE_MAX_MIB_DEFAULT,
        structural_cache_audit_rate: live_edit::config::STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT,
        structural_cache_audit_above_ceiling: false,
        valkey_connect_timeout_secs: 30,
        valkey_command_timeout_ms: 5000,
        valkey_reconnect_max_delay_ms: 30_000,
        valkey_auth_mode: live_edit::config::ValkeyAuthMode::Static,
        valkey_token_file: None,
        valkey_token_username: None,
        valkey_token_refresh_lead_secs: 180,
        valkey_iam_username: None,
        valkey_iam_region: None,
        valkey_iam_cluster_host: None,
        valkey_iam_service: live_edit::config::ValkeyIamService::MemoryDb,
        storage_service_endpoint: Some("http://unused.test".to_string()),
        auth_timeout_ms: 1000,
        auth_cache_ttl_secs: 60,
        auth_cache_negative_ttl_secs: 30,
        auth_userinfo_negative_ttl_secs: 5,
        auth_cache_max_entries: 1_000,
        userinfo_endpoint: Some("http://unused.test".to_string()),
        max_layer_size: 1_048_576,
        upload_memory_budget_mib:
            live_edit::grpc::upload_admission::UPLOAD_MEMORY_BUDGET_MIB_DEFAULT,
        upload_admission_wait_ms:
            live_edit::grpc::upload_admission::UPLOAD_ADMISSION_WAIT_MS_DEFAULT,
        upload_admission_max_waiters:
            live_edit::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
        layer_stream_maxlen: 4096,
        layer_stream_max_bytes: live_edit::config::LAYER_STREAM_MAX_BYTES_DEFAULT,
        presence_broadcast_capacity: 4096,
        presence_lease_ttl_secs: live_edit::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
        presence_refresh_interval_secs: live_edit::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
        presence_keepalive_interval_ms: live_edit::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
        shutdown_timeout_secs: 1,
        shutdown_overall_budget_secs: 5,
        shutdown_stream_end_rate: 25,
        shutdown_stream_end_max_window_secs: 20,
        log_level: "warn".to_string(),
        log_format: live_edit::config::LogFormat::Json,
        trace_filter: live_edit::config::DEFAULT_TRACE_FILTER.to_string(),
        sweeper_interval_secs: 15,
        session_list_timeout_ms: 1000,
        rpc_deadline_ms: 5000,
        service_label: "test".to_string(),
        valkey_memory_bytes: None,
        ext_release_max_bytes: None,
        ext_reconcile_interval_secs: 3600,
        auth_max_attempts: 3,
        auth_retry_initial_delay_ms: 100,
        authn_enabled: true,
        authz_enabled: false,
        userinfo_service_enabled: true,
        test_only_dangerously_allow_unvalidated_tokens: false,
        health_probe_interval_ms: 5_000,
        health_probe_timeout_ms: 2_000,
        auth_protocol: AuthProtocol::Rest,
        update_layer_retry_max_attempts: 128,
        update_layer_per_object_concurrency: 16,
        update_layer_backend: live_edit::config::UpdateLayerBackend::Lua,
        ext_value_verify: live_edit::config::ExtValueVerifyMode::Enforce,
        session_retry_max_attempts: 32,
        session_retry_initial_delay_ms: 1,
        pod_memory: live_edit::config::PodMemoryPlan::reference(),
        session_tokens_sweep_interval_secs: 60,
        checkpoint_enabled: false,
        checkpoint_adapter: live_edit::checkpoint::CheckpointAdapter::Disabled,
        checkpoint_interval_secs: 60,
        checkpoint_quiesce_secs: 30,
        checkpoint_tick_secs: 10,
        checkpoint_retain_generations: 2,
        checkpoint_seq_safety_margin: 1000,
        checkpoint_counters_sync_secs: 60,
        checkpoint_reconcile_interval_secs: 3600,
        checkpoint_counter_safety_margin: 1000,
        checkpoint_tail_enabled: false,
        checkpoint_tail_max_chain: 8,
        checkpoint_tail_max_entries: 4096,
        checkpoint_tail_max_bytes: 67_108_864,
        checkpoint_local_disk_path: None,
        checkpoint_storage_api_endpoint: None,
        checkpoint_base_path: None,
        checkpoint_connect_timeout_ms: 5_000,
        checkpoint_request_timeout_ms: 30_000,
        checkpoint_ext_mirror_workers: 2,
        checkpoint_ext_mirror_queue_capacity: 1_000,
        service_token_file: None,
        service_token_poll_interval_secs: 15,
        service_token_client_credentials_enabled: false,
    };

    let client = live_edit::storage::test_support::connect_test_client(&config)
        .await
        .expect("Valkey connect must succeed");
    (client, key_schema, prefix)
}

/// Subscriptions opened for the mass-drop test.
const MASS_DROP_SUBSCRIPTIONS: usize = 64;

/// Watchdog for the scenario's setup stage (connect + open every consumer).
const MASS_DROP_SETUP_WATCHDOG: Duration = Duration::from_secs(60);

/// The scenario's own deadline for the drop phase, checked from inside the
/// runtime under test.
const MASS_DROP_PHASE_DEADLINE: Duration = Duration::from_secs(15);

/// Outer watchdog for the drop phase, checked from a plain OS thread: if the
/// runtime deadlocks, the in-runtime deadline above cannot fire.
const MASS_DROP_WATCHDOG: Duration = Duration::from_secs(25);

/// Dropping many consumers (and their admissions) at once must keep the
/// runtime responsive and return every registration and admission permit.
///
/// Runs on a dedicated thread with a capped runtime (2 workers, 1 blocking
/// thread) so a blocking wait in the drop path would starve the runtime and be
/// caught by the outer watchdog rather than hanging the test binary.
#[test]
fn mass_consumer_drop_keeps_the_runtime_responsive_and_returns_every_slot() {
    let (progress_tx, progress_rx) = std::sync::mpsc::channel::<&'static str>();
    let mut scenario = Some(
        std::thread::Builder::new()
            .name("mass-drop".to_string())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .max_blocking_threads(1)
                    .enable_all()
                    .build()
                    .expect("building the capped multi-thread runtime must succeed");
                runtime.block_on(mass_drop_scenario(progress_tx));
                runtime.shutdown_timeout(Duration::from_secs(5));
            })
            .expect("spawning the scenario thread must succeed"),
    );

    await_mass_drop_stage(
        &progress_rx,
        &mut scenario,
        MASS_DROP_SETUP_WATCHDOG,
        "set up",
        "the fixture did not finish connecting and opening its consumers — a setup failure \
         (Valkey unreachable or overloaded), not a verdict on teardown",
    );
    await_mass_drop_stage(
        &progress_rx,
        &mut scenario,
        MASS_DROP_WATCHDOG,
        "finished",
        "dropping the consumers at once DEADLOCKED the runtime: not even its own phase \
         deadline could fire. `LayerStreamConsumer::Drop` must unregister without a blocking \
         wait (`block_in_place` / `block_on`)",
    );
    scenario
        .take()
        .expect("the scenario thread is joined exactly once")
        .join()
        .unwrap_or_else(|payload| std::panic::resume_unwind(payload));
}

fn await_mass_drop_stage(
    progress: &std::sync::mpsc::Receiver<&'static str>,
    scenario: &mut Option<std::thread::JoinHandle<()>>,
    budget: Duration,
    stage: &'static str,
    on_timeout: &str,
) {
    match progress.recv_timeout(budget) {
        Ok(reported) => assert_eq!(
            reported, stage,
            "the scenario reported its stages out of order"
        ),
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("no `{stage}` from the mass-drop scenario within {budget:?}: {on_timeout}")
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            match scenario
                .take()
                .expect("the scenario thread is joined exactly once")
                .join()
            {
                Err(payload) => std::panic::resume_unwind(payload),
                Ok(()) => panic!("the mass-drop scenario exited without reporting `{stage}`"),
            }
        }
    }
}

async fn mass_drop_scenario(progress: std::sync::mpsc::Sender<&'static str>) {
    use std::sync::Arc;
    use tokio::time::timeout_at;

    const N: usize = MASS_DROP_SUBSCRIPTIONS;
    // Two readers for N subscriptions: the registrations are what is dropped,
    // not connections. The admission cap is exactly N so the gate is saturated.
    let (client, key_schema, _prefix) = connect_with_subscriber_pool(2, N).await;
    let object_id: u64 = 970_201;

    let barrier = Arc::new(tokio::sync::Barrier::new(N + 1));
    let mut droppers = Vec::with_capacity(N);
    for i in 0..N {
        let admission = client
            .admit_subscriber(None)
            .await
            .unwrap_or_else(|refusal| {
                panic!("subscription {i} must be admitted under a cap of {N}: {refusal:?}")
            });
        let consumer = consume_layer_stream(
            &client,
            &key_schema,
            object_id + (i as u64 % 8),
            StreamCursor::From("0-0".to_string()),
            tokio::time::Instant::now() + std::time::Duration::from_secs(2),
        )
        .await
        .unwrap_or_else(|e| panic!("consumer {i} must start: {e:?}"));
        let barrier = Arc::clone(&barrier);
        droppers.push(tokio::spawn(async move {
            barrier.wait().await;
            drop(consumer);
            drop(admission);
        }));
    }
    assert_eq!(
        client.subscriber_admission.available_permits(),
        0,
        "fixture self-check: all {N} admission permits must be held before the drop"
    );
    assert_eq!(client.stream_mux.snapshot().subscriptions(), N);

    progress
        .send("set up")
        .expect("the test thread is waiting for this stage");
    let deadline = tokio::time::Instant::now() + MASS_DROP_PHASE_DEADLINE;
    barrier.wait().await;

    // A timer on a separate task must still fire while the drops run.
    let timer = tokio::spawn(timeout_at(
        deadline,
        tokio::time::sleep(Duration::from_millis(10)),
    ));
    for (i, dropper) in droppers.into_iter().enumerate() {
        timeout_at(deadline, dropper)
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "dropping subscription {i} did not complete within the phase deadline — \
                     `LayerStreamConsumer::Drop` must never wait"
                )
            })
            .unwrap_or_else(|e| panic!("dropping subscription {i} panicked: {e}"));
    }
    timer
        .await
        .expect("the timer task must not panic")
        .expect("a 10 ms sleep on a separate task must complete while consumers are torn down");

    // Unregistration is synchronous; the permits return with the admissions.
    assert_eq!(client.stream_mux.snapshot().subscriptions(), 0);
    assert_eq!(client.subscriber_admission.available_permits(), N);

    // The multiplexer is fully usable again: N fresh subscriptions all deliver.
    let mut consumers = Vec::with_capacity(N);
    for i in 0..N {
        consumers.push(
            timeout_at(
                deadline,
                consume_layer_stream(
                    &client,
                    &key_schema,
                    object_id + 100,
                    StreamCursor::From("0-0".to_string()),
                    tokio::time::Instant::now() + std::time::Duration::from_secs(2),
                ),
            )
            .await
            .unwrap_or_else(|_| panic!("fresh consumer {i} did not open in time"))
            .unwrap_or_else(|e| panic!("fresh consumer {i} failed to open: {e:?}")),
        );
    }
    let stream_key = key_schema.layer_stream(object_id + 100);
    let _: String = client
        .pool
        .xadd(
            &stream_key,
            false,
            None,
            "1-0",
            vec![("payload", format!("update|{}|1|w|0|0", object_id + 100))],
        )
        .await
        .expect("XADD must succeed");
    for (i, consumer) in consumers.iter().enumerate() {
        let got = timeout_at(deadline, consumer.recv())
            .await
            .unwrap_or_else(|_| panic!("fresh consumer {i} got nothing after the mass drop"));
        assert!(
            got.is_some(),
            "fresh consumer {i} ended instead of delivering"
        );
    }
    drop(consumers);
    client.quit().await;

    progress
        .send("finished")
        .expect("the test thread is waiting for this stage");
}

/// A reader still blocked on a stream whose only subscription ended must pick
/// up a NEW stream promptly: the new key is not in the in-flight XREAD, so the
/// reader wakes it (XADD on its wake key) instead of waiting out the 30 s
/// BLOCK — and that wake is not scored as a wedge.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reader_blocked_on_an_ended_stream_picks_up_a_new_stream_promptly() {
    /// A deadline, not a measurement: anything well inside the 30 s BLOCK
    /// still tells a woken read from one that waited the BLOCK out.
    const FIRST_DELIVERY_BUDGET: Duration = Duration::from_secs(10);

    let (client, key_schema, _prefix) = connect_with_subscriber_pool(1, 16).await;
    let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("test"));
    client
        .install_observability_hooks(std::sync::Arc::clone(&metrics))
        .await;

    let object_a: u64 = 970_001;
    let object_b: u64 = 970_002;

    let a = consume_layer_stream(
        &client,
        &key_schema,
        object_a,
        StreamCursor::From("0-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consumer A must start");
    // A warm-up delivery proves A's XREAD reached the server; the reader then
    // re-plans and parks in a fresh 30 s BLOCK on A's key.
    let a_stream_key = key_schema.layer_stream(object_a);
    let _: String = client
        .pool
        .xadd(
            &a_stream_key,
            false,
            None,
            "1-0",
            vec![("payload", format!("update|{object_a}|1|warmup|0|0"))],
        )
        .await
        .expect("warm-up XADD on object A must succeed");
    tokio::time::timeout(Duration::from_secs(10), a.recv())
        .await
        .expect("consumer A must deliver the warm-up entry")
        .expect("consumer A must still be registered");
    await_blocked_client(&client).await;

    drop(a);

    let b = consume_layer_stream(
        &client,
        &key_schema,
        object_b,
        StreamCursor::From("0-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consumer B must start");
    let b_stream_key = key_schema.layer_stream(object_b);
    let _: String = client
        .pool
        .xadd(
            &b_stream_key,
            false,
            None,
            "1-0",
            vec![("payload", format!("update|{object_b}|1|writer|0|0"))],
        )
        .await
        .expect("XADD on object B must succeed");

    let received = tokio::time::timeout(FIRST_DELIVERY_BUDGET, b.recv()).await;
    match received {
        Ok(Some((stream_id, notification))) => {
            assert_eq!(stream_id, "1-0");
            assert!(
                matches!(
                    notification,
                    LayerNotification::Update {
                        sequence_num: 1,
                        ..
                    }
                ),
                "expected the entry written to object B, got: {notification:?}"
            );
        }
        Ok(None) => panic!("consumer B ended instead of delivering"),
        Err(_) => panic!(
            "consumer B took longer than {FIRST_DELIVERY_BUDGET:?} for its first delivery: the \
             reader did not wake its in-flight XREAD for the new key and waited out the BLOCK"
        ),
    }
    assert!(client.stream_mux.snapshot().readers[0].wakes >= 1);
    assert_eq!(
        crate::storage::mux_test_helpers::forced_reconnect_row(&metrics),
        crate::storage::mux_test_helpers::forced_reconnects_only_on(
            live_edit::observability::metrics::CONSUMER_PHASE_WAKE_ACK,
            0
        ),
        "a routine wake must not be scored as a wedge, in any phase"
    );

    drop(b);
    client.quit().await;
}

/// Wait until Valkey reports at least one client parked in a blocking command.
async fn await_blocked_client(client: &ValkeyClient) {
    use fred::interfaces::ClientLike;
    use fred::types::InfoKind;

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let info: String = client
            .pool
            .info(Some(InfoKind::Clients))
            .await
            .expect("INFO clients must succeed");
        let blocked: u64 = info
            .lines()
            .find_map(|line: &str| line.trim().strip_prefix("blocked_clients:"))
            .and_then(|v| v.trim().parse::<u64>().ok())
            .expect("INFO clients must report a blocked_clients field");
        if blocked >= 1 {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "no client registered as blocked within 10 s — the reader's XREAD BLOCK should be \
             pending server-side by now"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[derive(Clone, Copy, Debug)]
enum Churn {
    /// Each cycle admits, subscribes, and drops both — a stream of short
    /// subscriptions on a saturated admission gate.
    EndOfSubscription,
    /// Admissions are held across cycles and each cycle re-opens consumers
    /// under them via `shutdown_and_release`, as a re-bootstrap does.
    Rebootstrap,
}

async fn churn_saturated_subscriber_pool(churn: Churn) {
    const CHURN_CAP: usize = 4;
    const CHURN_CYCLES: usize = 16;
    const CHURN_DEADLINE: Duration = Duration::from_secs(60);

    let (client, key_schema, _prefix) = connect_with_subscriber_pool(2, CHURN_CAP).await;
    let object_id: u64 = 99;

    let admit_all =
        |cycle: usize| {
            let client = client.clone();
            async move {
                let mut admissions = Vec::with_capacity(CHURN_CAP);
                for _ in 0..CHURN_CAP {
                    admissions.push(client.admit_subscriber(None).await.unwrap_or_else(
                        |refusal| {
                            panic!(
                        "{churn:?} cycle {cycle}: a subscription was refused ({refusal:?}) \
                         although every previous one had ended"
                    )
                        },
                    ));
                }
                admissions
            }
        };

    let run = async {
        let mut held = match churn {
            Churn::Rebootstrap => admit_all(0).await,
            Churn::EndOfSubscription => Vec::new(),
        };
        for cycle in 0..CHURN_CYCLES {
            if let Churn::EndOfSubscription = churn {
                held = admit_all(cycle).await;
            }
            let mut consumers = Vec::with_capacity(CHURN_CAP);
            for _ in &held {
                consumers.push(
                    consume_layer_stream(
                        &client,
                        &key_schema,
                        object_id,
                        StreamCursor::From("0-0".to_string()),
                        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
                    )
                    .await
                    .unwrap_or_else(|e| panic!("{churn:?} cycle {cycle}: open failed: {e:?}")),
                );
            }
            assert_eq!(client.stream_mux.snapshot().subscriptions(), CHURN_CAP);
            match churn {
                Churn::EndOfSubscription => {
                    drop(consumers);
                    held.clear();
                }
                Churn::Rebootstrap => {
                    for consumer in &mut consumers {
                        consumer.shutdown_and_release().await;
                    }
                }
            }
            assert_eq!(
                client.stream_mux.snapshot().subscriptions(),
                0,
                "{churn:?} cycle {cycle}: closed consumers must unregister"
            );
        }
    };
    tokio::time::timeout(CHURN_DEADLINE, run)
        .await
        .unwrap_or_else(|_| {
            panic!("{churn:?} churn did not finish within {CHURN_DEADLINE:?} — a close is hanging")
        });
    client.quit().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn drop_churn_readmits_and_reopens_every_cycle_on_a_saturated_pool() {
    churn_saturated_subscriber_pool(Churn::EndOfSubscription).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rebootstrap_churn_reopens_under_the_same_admissions_every_cycle() {
    churn_saturated_subscriber_pool(Churn::Rebootstrap).await;
}

/// A subscription opened right after another on the same object ended still
/// receives new entries.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fresh_consumer_after_drop_still_receives_notifications() {
    let tv = TestValkey::setup().await;
    let object_id: u64 = 2024;

    let first = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("0-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consume_layer_stream must succeed");
    tokio::time::sleep(Duration::from_millis(100)).await;
    drop(first);

    let second = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("0-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("replacement consume_layer_stream must succeed");
    tokio::time::sleep(Duration::from_millis(100)).await;

    // `1-0`, the commit right after both consumers' `0-0` cursor (a first
    // entry further above it would read as a trim past the cursor).
    let payload = "update|2024|1|client-after-drop|111|222";
    let expected_id = xadd_payload(&tv, object_id, "1-0", payload).await;

    let (stream_id, notification) = tokio::time::timeout(Duration::from_secs(5), second.recv())
        .await
        .expect("replacement recv must not time out")
        .expect("replacement consumer must yield the notification");

    assert_eq!(stream_id, expected_id);
    assert_eq!(
        notification,
        LayerNotification::Update {
            object_id: 2024,
            sequence_num: 1,
            client_id: "client-after-drop".to_string(),
            client_ts: 111,
            server_ts: 222,
            relay_ompf: None,
        },
    );

    second.unsubscribe().await;
}

// =============================================================================
// b.tmx — no production code bridges back into the runtime by blocking
// =============================================================================

/// `tokio::task::block_in_place`, the call that turned a mass disconnect into a
/// process-wide deadlock (b.tmx).
const BLOCK_IN_PLACE_NEEDLE: &str = "block_in_place";

/// Any `block_on(` — `Handle::block_on`, `Runtime::block_on`,
/// `futures::executor::block_on`. Each parks the calling thread until a future
/// completes; on a runtime worker, or inside `block_in_place`, that is the same
/// hazard.
const BLOCK_ON_NEEDLE: &str = "block_on(";

/// The one file under `src/` allowed to hold either needle in code.
/// `TestValkey`'s `Drop` bridges into the runtime to UNLINK the test's keys,
/// and the module is compiled only under `test-support` — never into the
/// release binary.
const BLOCKING_BRIDGE_CARVE_OUT: &str = "src/storage/test_support.rs";

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
/// No production code under `src/` may call `block_in_place` or `block_on(`
/// (b.tmx).
///
/// `LayerStreamConsumer::Drop` used `block_in_place` + `block_on` to wait up to
/// 250 ms for its task. Each call hands the worker's core to a replacement
/// thread from the blocking pool, so when hundreds of consumers dropped at once
/// the pool ran out, every core queued behind a thread that never came, and the
/// process deadlocked — gRPC, `/metrics` and logging all stopped. The
/// behavioural regression
/// (`mass_consumer_drop_keeps_the_runtime_responsive_and_returns_every_slot`)
/// covers that one `Drop`; this fence covers the next `Drop` or cancellation
/// path that reaches for the same bridge, because the failure it causes shows
/// up only under a burst no ordinary test produces.
///
/// Comments are stripped (all three shapes, by the scanner shared with
/// `txn_checkout_test.rs`), so prose explaining the rule — including the
/// doc-only mentions in `src/checkpoint/apply.rs` — does not trip it. String
/// literals are kept, deliberately: a false positive there is a visible red
/// naming the line, never a silent miss.
///
/// If you are here because the fence failed on test-only code inside `src/` (a
/// `#[cfg(test)]` module, say), move the helper into
/// [`BLOCKING_BRIDGE_CARVE_OUT`] or under `tests/` rather than widening the
/// carve-out: the fence cannot tell a `cfg`-gated line from a production one.
#[test]
fn no_production_code_under_src_blocks_a_runtime_worker() {
    use std::path::{Path, PathBuf};

    use super::txn_checkout_test::{is_needle_declaration, needle_hits, rust_sources_under};

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    rust_sources_under(&root, &root.join("src"), &mut files);
    files.sort();

    // Non-vacuity, part 1: the walk reaches the file that held the defect, the
    // pool it starved, and the carve-out (so the exclusion below is excluding
    // something that exists).
    for expected in [
        "src/storage/notifications.rs",
        "src/storage/checkout.rs",
        BLOCKING_BRIDGE_CARVE_OUT,
    ] {
        assert!(
            files.iter().any(|f| f == Path::new(expected)),
            "the source scan did not visit `{expected}`; it walked {} files. Either the file \
             moved (update this list) or the walk is not reaching where it should",
            files.len()
        );
    }

    let mut violations = Vec::new();
    for file in files
        .iter()
        .filter(|f| *f != Path::new(BLOCKING_BRIDGE_CARVE_OUT))
    {
        let source = std::fs::read_to_string(root.join(file))
            .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", file.display()));
        for needle in [BLOCK_IN_PLACE_NEEDLE, BLOCK_ON_NEEDLE] {
            for (line_no, code) in needle_hits(&source, needle) {
                violations.push(format!("  {}:{line_no}\n    {code}", file.display()));
            }
        }
    }

    // Non-vacuity, part 2: with no allowed production site, "found nothing" is
    // the passing state, so a misspelt needle would pass silently. Each needle
    // must still select real CODE somewhere under tests/, where the test-only
    // bridges legitimately live — excluding this file's own needle
    // declarations, which would otherwise satisfy the check by themselves.
    let mut test_files = Vec::new();
    rust_sources_under(&root, &root.join("tests"), &mut test_files);
    for needle in [BLOCK_IN_PLACE_NEEDLE, BLOCK_ON_NEEDLE] {
        let real_sites = test_files
            .iter()
            .filter(|file| {
                let source = std::fs::read_to_string(root.join(file))
                    .unwrap_or_else(|e| panic!("reading {} must succeed: {e}", file.display()));
                needle_hits(&source, needle)
                    .iter()
                    .any(|(_, code)| !is_needle_declaration(code))
            })
            .count();
        assert!(
            real_sites > 0,
            "`{needle}` matches no code anywhere under tests/, so it has stopped selecting real \
             calls and the src/ scan proves nothing. Fix the needle rather than deleting this \
             fence"
        );
    }

    assert!(
        violations.is_empty(),
        "blocking runtime bridge(s) in production code under src/:\n\n{}\n\n\
         `block_in_place` hands the worker's core to a new blocking-pool thread, and `block_on` \
         parks the calling thread until a future completes. On a hot path — a `Drop`, a \
         cancellation, anything many tasks can hit at once — enough concurrent callers exhaust \
         the blocking pool, every worker core queues behind a thread that never arrives, and the \
         process deadlocks with no timer running to break it (b.tmx: a mass subscriber \
         disconnect wedged the whole process in a local reproduction). Make the path \
         asynchronous instead: abort or signal \
         the task and let an owned value's own `Drop` do the release, or await an explicit async \
         close before dropping. Only `{BLOCKING_BRIDGE_CARVE_OUT}` is exempt, because it is \
         compiled only under `test-support`.",
        violations.join("\n")
    );
}

// =============================================================================
// b.9yf — `is_client_unblocked_error` vs. the error fred ACTUALLY returns
// =============================================================================
//
// Every other test of this predicate — the classifier unit tests in
// `client.rs` and `stream_mux/reader.rs` and the fault-injected mux tests —
// feeds the predicate a hand-built
// `(ErrorKind::Unknown, "UNBLOCKED …")` value. That constant restates the exact
// assumption `is_client_unblocked_error` makes, so the whole set is circular:
// if fred's `pretty_error` ever grows an `UNBLOCKED` arm (giving the reply a
// kind of its own), or Valkey reworks the reply text, every one of those tests
// stays green while production silently reclassifies every `-UNBLOCKED` a live
// read receives as PERMANENT and tears the subscription down — the b.9yf
// symptom. Since b.19i no credential refresh interrupts a reader's read (a
// reader re-AUTHs between its reads), so the production trigger is narrower:
// anything outside the service that `CLIENT UNBLOCK`s a reader's live read
// (an operator, a managed-store maintenance action). The reader must still
// ride that out.
//
// The test below closes that loop by provoking a real `-UNBLOCKED` from the
// server and asserting the predicate accepts whatever fred hands back.

/// Bounded poll of `CLIENT LIST` for the connection carrying `name`, returning
/// its server-assigned connection id.
///
/// `CLIENT LIST` is server-global, so on the shared test Valkey it also lists
/// every other agent's connections — filtering on a UUID-unique `CLIENT
/// SETNAME` is what makes the lookup deterministic under concurrent load. The
/// poll exists only to absorb the connection's own registration latency; a
/// miss past the deadline means the `SETNAME` did not land on the connection
/// the caller thinks it did.
async fn connection_id_by_name(observer: &ValkeyClient, name: &str) -> i64 {
    use fred::interfaces::ClientLike;
    use fred::types::CustomCommand;

    let want_name = format!("name={name}");
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let listing: String = observer
            .pool
            .next()
            .custom::<String, String>(
                CustomCommand::new_static("CLIENT LIST", None::<u16>, false),
                vec![],
            )
            .await
            .expect("CLIENT LIST must succeed");

        let found = listing.lines().find_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if !fields.contains(&want_name.as_str()) {
                return None;
            }
            fields
                .iter()
                .find_map(|f| f.strip_prefix("id="))
                .and_then(|id| id.parse::<i64>().ok())
        });
        if let Some(id) = found {
            return id;
        }

        assert!(
            Instant::now() < deadline,
            "no connection named {name} appeared in CLIENT LIST within 10 s, although \
             CLIENT SETNAME succeeded on it — the harness cannot identify the connection \
             it is about to unblock, so the test would prove nothing"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// The predicate the consumer's cheap re-issue path keys on must accept the
/// error fred REALLY produces for a server-side `-UNBLOCKED` (b.9yf).
///
/// The fixture reproduces, by hand, what a `CLIENT UNBLOCK` from outside the
/// service does to a reader's live read — the same server reply fred's
/// `Blocking::Interrupt` provokes. (Before b.19i a queued credential-refresh
/// `AUTH` provoked it on every rotation; since b.19i a reader re-AUTHs between
/// its reads, so in production only an outside unblock reaches a live read.)
///
///   1. Take an idle stream reader client — the same pool, and so the same fred
///      client configuration, the multiplexed XREADs run on — with the
///      readers' idle keepalive turned off, so the reader itself sends nothing
///      on it.
///   2. Name it with a UUID so it is findable on a shared Valkey.
///   3. Park it in an `XREAD BLOCK` on a stream nothing ever writes to, under
///      the reader's own per-command `Options` (`max_attempts = 1`, timeout
///      `block + 5 s`), so nothing but the unblock can end the read.
///   4. From a SECOND connection, `CLIENT UNBLOCK <id> ERROR`.
///
/// The assertion is deliberately only `is_client_unblocked_error(&err)` — not a
/// re-check of the kind or the reply text. Re-asserting those here would
/// reintroduce the circularity this test exists to break; the actual kind and
/// details are interpolated into the failure message instead, which is what a
/// future engineer needs when fred or Valkey changes the shape.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn is_client_unblocked_error_accepts_the_real_server_unblocked_reply() {
    use fred::interfaces::ClientLike;
    use fred::types::config::Options;
    use fred::types::CustomCommand;
    use live_edit::storage::client::is_client_unblocked_error;

    // Long enough that the BLOCK cannot lapse on its own inside the test
    // window: the read must end because of the unblock, not because it expired.
    const BLOCK_MS: u64 = 30_000;

    let tv = TestValkey::setup().await;
    // An idle reader `PING`s its own connection on its keepalive, and a
    // command on this client while the XREAD below is parked would `CLIENT
    // UNBLOCK` it before the test's own unblock could. The knob seam exists
    // only under `test-fault-injection` (which CI's runs enable); without it
    // the production keepalive (every 200 s, from a random offset) leaves a
    // small chance of that early unblock.
    #[cfg(feature = "test-fault-injection")]
    tv.client()
        .stream_mux
        .set_knobs_for_test(live_edit::storage::stream_mux::StreamMuxKnobs {
            keepalive_interval: None,
            ..tv.client().stream_mux.knobs()
        });

    // A stream key nothing in this test (or any other — the prefix is
    // per-fixture) ever XADDs to.
    let quiet_stream = tv.key_schema().layer_stream(970_101);
    let name = format!("b9yf-unblock-{}", uuid::Uuid::new_v4());

    // Readers connect lazily, the first time they serve a subscription:
    // connect reader 0 (standalone routing sends object 0 to it), then let the
    // subscription go. Its read is left blocked on the server, and the command
    // below interrupts it (`Blocking::Interrupt`); with no subscription left
    // the reader then has nothing to read, so the client is ours.
    super::mux_test_helpers::connect_reader_zero(tv.client()).await;
    let exclusive = tv
        .client()
        .stream_mux
        .reader_client_for_test(0)
        .expect("reader 0 holds a connection");
    let _: String = exclusive
        .custom(
            CustomCommand::new_static("CLIENT SETNAME", None::<u16>, false),
            vec![name.clone()],
        )
        .await
        .expect("CLIENT SETNAME on the reader client must succeed");

    // Same per-command options the reader's XREAD uses, so the
    // error travels the same path through fred: one attempt (no silent retry
    // that could swallow the reply), and a fred timeout comfortably past the
    // server-side BLOCK so Valkey's answer always wins.
    let xread_options = Options {
        timeout: Some(Duration::from_millis(BLOCK_MS + 5_000)),
        max_attempts: Some(1),
        ..Default::default()
    };
    let reader = tokio::spawn(async move {
        exclusive
            .with_options(&xread_options)
            .xread::<fred::types::Value, _, _>(None, Some(BLOCK_MS), quiet_stream.as_str(), "0-0")
            .await
    });

    let id = connection_id_by_name(tv.client(), &name).await;

    // `CLIENT UNBLOCK` answers 1 only when the target was actually parked in a
    // blocking command, so this loop doubles as the barrier proving the XREAD
    // reached the server — a 0 just means the spawned task has not got there
    // yet. A deadline without a single 1 means the fixture never established
    // the precondition, which must not be read as a predicate failure.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let unblocked: i64 = tv
            .client()
            .pool
            .next()
            .custom(
                CustomCommand::new_static("CLIENT UNBLOCK", None::<u16>, false),
                vec![id.to_string(), "ERROR".to_string()],
            )
            .await
            .expect("CLIENT UNBLOCK must succeed");
        if unblocked == 1 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "connection {id} ({name}) never registered as blocked within 10 s, so \
             `CLIENT UNBLOCK ... ERROR` had nothing to interrupt and no `-UNBLOCKED` reply \
             was ever produced — a harness precondition failure, not a classifier result"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let result = tokio::time::timeout(Duration::from_secs(20), reader)
        .await
        .expect("the unblocked XREAD must resolve promptly once the server answered it")
        .expect("the XREAD task must not panic");

    let err = result.expect_err(
        "XREAD returned Ok after the server unblocked it with an ERROR flag — the reply \
         should be an error frame, so the fixture is not producing the condition under test",
    );

    assert!(
        is_client_unblocked_error(&err),
        "`is_client_unblocked_error` REJECTED the error fred actually returns for a real \
         server-side `-UNBLOCKED` (kind={:?}, details={:?}). Every other b.9yf test builds \
         that error by hand from the same assumption the predicate encodes, so they all stay \
         green while production regresses: with this predicate returning false, a stream \
         reader classifies each `-UNBLOCKED` a live read receives (an outside `CLIENT \
         UNBLOCK`) as a PERMANENT failure and ends every subscription it holds. Update the \
         predicate \
         in `src/storage/client.rs` to match the shape above, and the hand-built constants \
         in the unit tests alongside it",
        err.kind(),
        err.details()
    );
}
