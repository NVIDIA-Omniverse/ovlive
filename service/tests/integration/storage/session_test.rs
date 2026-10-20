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

// Integration tests for SessionStore: CreateSession, DeleteSession, ListSessions.
//
// Tests run against a real Valkey instance via the TestValkey fixture. Each test
// gets an isolated key namespace so tests can run in parallel safely.
//
// Requires a running Valkey. Address is read from TEST_VALKEY_URL, defaulting to
// redis://127.0.0.1:6379. Tests skip gracefully if Valkey is unreachable.

use std::collections::HashMap;
use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};

use live_edit::config::{AuthProtocol, Config};
use live_edit::error::ServiceError;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::{connect_test_client, TestValkey};

/// Helper: build a SessionStore from a TestValkey fixture.
fn store(tv: &TestValkey) -> SessionStore {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
}

// ═══════════════════════════════════════════════════════════════════════════════
// CreateSession tests (t3.1it.ra.13.r6)
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_round_trips_via_list() {
    let tv = TestValkey::setup().await;
    let ss = store(&tv);

    let sid = ss
        .create("my-session", "/stage", "alice")
        .await
        .expect("create must succeed");

    let sessions = ss.list("/stage").await.expect("list must succeed").present;
    assert_eq!(sessions.len(), 1, "list must return exactly 1 session");
    assert_eq!(sessions[0].session_id, sid);
    assert_eq!(sessions[0].name, "my-session");
    assert_eq!(sessions[0].base_path, "/stage");
    assert_eq!(sessions[0].created_by, "alice");
    assert!(sessions[0].created_at > 0, "created_at must be non-zero");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_returns_valid_uuid4() {
    let tv = TestValkey::setup().await;
    let ss = store(&tv);

    let sid = ss
        .create("uuid-test", "/stage", "bob")
        .await
        .expect("create must succeed");

    let parsed = uuid::Uuid::parse_str(&sid).expect("session_id must be valid UUID");
    assert_eq!(
        parsed.get_version(),
        Some(uuid::Version::Random),
        "session_id must be UUID v4 (random)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_duplicate_name_returns_already_exists() {
    let tv = TestValkey::setup().await;
    let ss = store(&tv);

    ss.create("dup-name", "/stage", "alice")
        .await
        .expect("first create must succeed");

    let err = ss
        .create("dup-name", "/stage", "bob")
        .await
        .expect_err("second create with same name must fail");

    assert!(
        matches!(err, ServiceError::AlreadyExists(_)),
        "expected AlreadyExists, got: {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_case_sensitive_names_are_distinct() {
    let tv = TestValkey::setup().await;
    let ss = store(&tv);

    let sid_upper = ss
        .create("Foo", "/stage", "alice")
        .await
        .expect("create 'Foo' must succeed");

    let sid_lower = ss
        .create("foo", "/stage", "alice")
        .await
        .expect("create 'foo' must succeed");

    assert_ne!(
        sid_upper, sid_lower,
        "'Foo' and 'foo' must create distinct sessions"
    );

    let sessions = ss.list("/stage").await.expect("list must succeed").present;
    assert_eq!(sessions.len(), 2, "both sessions must appear in list");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_session_concurrent_same_name_exactly_one_succeeds() {
    // WATCH/MULTI/EXEC requires separate connections to detect contention.
    // Build a ValkeyClient with pool_size=10 so each concurrent task gets
    // its own connection.
    let valkey_url =
        std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

    let prefix = format!("test_{}:", uuid::Uuid::new_v4());
    let key_schema = KeySchema::new(&prefix);

    let config = Config {
        catchup_snapshot_enabled: false,
        port: 0,
        metrics_port: 0,
        valkey_url,
        valkey_pool_size: 10,
        subscriber_pool_size: 16,
        subscriber_admission_wait_ms: 2_000,
        subscriber_max_subscriptions: 4096,
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

    let client = connect_test_client(&config)
        .await
        .expect("ValkeyClient must connect");

    let ss = SessionStore::new(client.clone(), key_schema.clone());

    // Use a barrier to ensure all tasks start their create call simultaneously.
    let barrier = Arc::new(tokio::sync::Barrier::new(10));

    let mut handles = Vec::with_capacity(10);
    for i in 0..10 {
        let ss_clone = ss.clone();
        let barrier_clone = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier_clone.wait().await;
            ss_clone
                .create("race-name", "/stage", &format!("user-{i}"))
                .await
        }));
    }

    let mut successes = 0u32;
    let mut already_exists = 0u32;
    let mut other_errors = 0u32;

    for handle in handles {
        match handle.await.expect("task must not panic") {
            Ok(_) => successes += 1,
            Err(ServiceError::AlreadyExists(_)) => already_exists += 1,
            Err(ServiceError::Aborted(_)) => other_errors += 1,
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    // Regardless of how many individual create calls report success,
    // verify the final state: list must return exactly one session with
    // the name "race-name", proving the WATCH/duplicate-check pipeline
    // prevents duplicate storage.
    let ss_verify = SessionStore::new(client, key_schema);
    let sessions = ss_verify
        .list("/stage")
        .await
        .expect("list must succeed")
        .present;

    let race_sessions: Vec<_> = sessions.iter().filter(|s| s.name == "race-name").collect();

    assert_eq!(
        race_sessions.len(),
        1,
        "exactly 1 session named 'race-name' must exist in storage \
         (got {successes} successes, {already_exists} AlreadyExists, \
         {other_errors} Aborted; final sessions: {race_sessions:?})"
    );
    assert!(successes >= 1, "at least 1 concurrent create must succeed");
}

// Regression for b.8q1 (session.rs::create nil-EXEC coercion). Writers with
// *distinct names on the same base_path* all WATCH the same path_index set,
// so any committed SADD cancels all in-flight peers' WATCHes. With the old
// `trx.exec::<Vec<i64>>(...)` decode, nil → empty Vec was treated as a
// successful commit — the caller would write the session_meta HSET *without
// the corresponding path_index SADD*, leaving an orphan whose session_id
// never appears in ListSessions. The fix decodes EXEC as `fred::types::Value`
// and re-enters the retry loop on `is_null()`.
//
// Invariants we actually check post-fix:
//   * Every outcome is either Ok(sid) or Aborted (retry cap hit after
//     MAX_TXN_RETRIES). No other error kinds are acceptable.
//   * Every Ok(sid) is fully written: path_index contains sid AND
//     session_meta has matching name. No orphaned session_meta without SADD.
//   * ListSessions returns exactly the committed set — no extras, no drops.
//
// Retry-budget margin — WHY WRITERS=4 AND NOT MORE:
//
// `src/storage/session.rs::MAX_TXN_RETRIES = 5` caps the WATCH/EXEC retry
// loop inside SessionStore::create. In the worst case N concurrent writers
// on the same path_index commit serially (each commit cancels all pending
// peers' WATCHes), so a single writer may need up to N-1 retries to land.
//
// Budget math:
//   - WRITERS = 4  →  worst-case retries needed = 3  ≤  MAX_TXN_RETRIES=5
//     → every writer has a 2-attempt safety margin. Stable.
//   - WRITERS = 5  →  worst-case retries needed = 4  ≤  MAX_TXN_RETRIES=5
//     → margin of 1; any extra scheduling jitter could exhaust the budget.
//   - WRITERS ≥ 6 →  worst-case > 5  →  at least one writer MUST Abort.
//     That's legitimate backpressure, not a silent drop — but it would make
//     this test's "all WRITERS commit" assertion flaky.
//
// This test's job is to exercise the retry path, not to stress it. The
// sibling test `create_session_concurrent_same_name_exactly_one_succeeds`
// already runs N=10 writers and verifies the "exactly one survives after
// AlreadyExists + retry exhaustion" contract for the high-contention case.
// Here we keep N=4 so we can deterministically assert every Ok(sid) lands
// fully — the real silent-drop signal for b.8q1 is a committed Ok whose
// session_id is *missing from path_index*, which happened before the fix.
//
// If MAX_TXN_RETRIES is ever raised in session.rs, WRITERS here can safely
// be raised to (MAX_TXN_RETRIES + 1). If it's lowered, WRITERS must drop.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_session_concurrent_distinct_names_all_succeed_via_retry() {
    // Pool size must allow all writers to race simultaneously so their WATCH
    // windows genuinely overlap and nil-EXECs are forced — the retry loop is
    // what we're validating here.
    const WRITERS: usize = 4;
    let valkey_url =
        std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

    let prefix = format!("test_{}:", uuid::Uuid::new_v4());
    let key_schema = KeySchema::new(&prefix);

    let config = Config {
        catchup_snapshot_enabled: false,
        port: 0,
        metrics_port: 0,
        valkey_url,
        valkey_pool_size: WRITERS as u32 * 2,
        subscriber_pool_size: 16,
        subscriber_admission_wait_ms: 2_000,
        subscriber_max_subscriptions: 4096,
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

    let client = connect_test_client(&config)
        .await
        .expect("ValkeyClient must connect");

    let ss = SessionStore::new(client.clone(), key_schema.clone());

    // Barrier to ensure every task fires its create call at the same instant.
    let barrier = Arc::new(tokio::sync::Barrier::new(WRITERS));

    let mut handles = Vec::with_capacity(WRITERS);
    for i in 0..WRITERS {
        let ss_clone = ss.clone();
        let barrier_clone = barrier.clone();
        let name = format!("race-retry-{i}");
        handles.push(tokio::spawn(async move {
            barrier_clone.wait().await;
            (
                name.clone(),
                ss_clone.create(&name, "/stage", &format!("user-{i}")).await,
            )
        }));
    }

    let mut committed_sids: Vec<(String, String)> = Vec::new();
    let mut aborts = 0u32;
    let mut other_errors: Vec<String> = Vec::new();

    for h in handles {
        let (name, res) = h.await.expect("task must not panic");
        match res {
            Ok(sid) => committed_sids.push((name, sid)),
            Err(ServiceError::Aborted(_)) => aborts += 1,
            Err(e) => other_errors.push(format!("{name}: {e:?}")),
        }
    }

    assert!(
        other_errors.is_empty(),
        "unexpected non-Aborted errors from distinct-name concurrent creates: {other_errors:?}"
    );
    assert_eq!(
        committed_sids.len() + aborts as usize,
        WRITERS,
        "total outcomes must equal WRITERS={WRITERS}; committed={}, aborts={aborts}",
        committed_sids.len()
    );
    // With WRITERS=4 and MAX_TXN_RETRIES=5 the retry budget is sufficient —
    // every writer should land. If this ever flakes, either raise
    // MAX_TXN_RETRIES or lower WRITERS; a false Ok would also manifest as a
    // path_index mismatch below (the real silent-drop signal).
    assert_eq!(
        committed_sids.len(),
        WRITERS,
        "all {WRITERS} distinct-name concurrent creates must eventually commit \
         within MAX_TXN_RETRIES; got {} successes, {aborts} aborts",
        committed_sids.len()
    );

    // Every committed session_id must appear in the path_index set, and the
    // metadata hash must match. Before the fix, nil-EXEC was silently coerced
    // to `Vec::new()`, so some of these assertions would fail — the SADD
    // never landed, the session_meta did, and ListSessions returned a partial
    // or missing view.
    let path_index_members: Vec<String> = client
        .pool
        .smembers(key_schema.path_index("/stage"))
        .await
        .expect("SMEMBERS must succeed");
    for (name, sid) in &committed_sids {
        assert!(
            path_index_members.contains(sid),
            "committed session {name}={sid} missing from path_index; members={path_index_members:?}"
        );
        let meta: HashMap<String, String> = client
            .pool
            .hgetall(key_schema.session_meta(sid))
            .await
            .expect("HGETALL must succeed");
        assert_eq!(
            meta.get("name").map(String::as_str),
            Some(name.as_str()),
            "session {sid} metadata name mismatch: meta={meta:?}"
        );
    }

    // ListSessions must return exactly WRITERS entries with matching names —
    // end-to-end confirmation that no silent drop produced an orphan.
    let ss_verify = SessionStore::new(client, key_schema);
    let listed = ss_verify
        .list("/stage")
        .await
        .expect("list must succeed")
        .present;
    assert_eq!(
        listed.len(),
        WRITERS,
        "ListSessions must return all {WRITERS} committed sessions; got {}: {listed:?}",
        listed.len()
    );
    let mut listed_names: Vec<&str> = listed.iter().map(|s| s.name.as_str()).collect();
    listed_names.sort();
    let mut expected_names: Vec<&str> = committed_sids.iter().map(|(n, _)| n.as_str()).collect();
    expected_names.sort();
    assert_eq!(
        listed_names, expected_names,
        "ListSessions names must match committed names exactly"
    );
}

// Regression for b.6e7 item 2: `SessionStore::create` must hold a single
// exclusive txn_pool client across WATCH → SMEMBERS → HGET → MULTI → EXEC.
//
// The pre-fix code called `self.client.pool.next()` — fred's regular
// multiplexed `Pool`, which interleaves logical operations from multiple
// callers onto a shared underlying connection. Valkey's WATCH is
// per-connection, so interleaved writers on one connection can land multiple
// conflicting commits without any WATCH tripping.
//
// This test forces that failure mode: it configures `valkey_pool_size=1`
// (one connection in both `pool` and `txn_pool`) and races N same-name
// creates against each other.
//
// - Post-fix (`acquire_txn_for`): the exclusive slot serializes all
//   N writers behind one checkout. Exactly one SADD lands; every
//   other writer sees the prior `name` in SMEMBERS and returns
//   `AlreadyExists`. Final path_index has exactly 1 session_id.
//
// - Pre-fix (pool.next()): N writers share one multiplexed connection.
//   Their interleaved WATCHes aren't protected against each other (WATCH
//   state is per-connection, not per-logical-caller), so each writer's
//   duplicate-name check + SADD can race and multiple sessions with the
//   same name can land. final path_index could contain 2+ session_ids.
//
// The core invariant: `SessionStore::create` must produce at most one
// session_id in path_index regardless of the number of concurrent
// same-name callers. Verifying this with pool_size=1 is the cheapest way
// to pin the "holds one exclusive client for the whole transaction"
// guarantee — if a future refactor drops back to `pool.next()`, this test
// fails deterministically.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_session_concurrent_same_name_with_pool_size_one_never_duplicates() {
    // pool_size=1 makes the exclusive-pool serialization load-bearing:
    // with the multiplexed `pool` (the pre-fix path), all racers would
    // share a single fred Client and WATCH couldn't protect them from
    // each other.
    const WRITERS: usize = 8;
    const POOL_SIZE: u32 = 1;

    let valkey_url =
        std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

    let prefix = format!("test_{}:", uuid::Uuid::new_v4());
    let key_schema = KeySchema::new(&prefix);

    let config = Config {
        catchup_snapshot_enabled: false,
        port: 0,
        metrics_port: 0,
        valkey_url,
        valkey_pool_size: POOL_SIZE,
        subscriber_pool_size: 4,
        subscriber_admission_wait_ms: 2_000,
        subscriber_max_subscriptions: 4096,
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

    let client = connect_test_client(&config)
        .await
        .expect("ValkeyClient must connect");

    let ss = SessionStore::new(client.clone(), key_schema.clone());

    let barrier = Arc::new(tokio::sync::Barrier::new(WRITERS));
    let mut handles = Vec::with_capacity(WRITERS);
    for i in 0..WRITERS {
        let ss_clone = ss.clone();
        let barrier_clone = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier_clone.wait().await;
            ss_clone
                .create("pool-one-race", "/stage", &format!("user-{i}"))
                .await
        }));
    }

    let mut successes = 0u32;
    let mut already_exists = 0u32;
    for h in handles {
        match h.await.expect("task must not panic") {
            Ok(_) => successes += 1,
            Err(ServiceError::AlreadyExists(_)) => already_exists += 1,
            Err(e) => panic!("unexpected error from pool-one same-name race: {e:?}"),
        }
    }

    assert_eq!(
        successes + already_exists,
        WRITERS as u32,
        "every writer must return Ok or AlreadyExists; got {successes} Ok, {already_exists} AlreadyExists"
    );

    // The load-bearing assertion: the duplicate-name invariant must hold
    // even with pool_size=1. Exactly one SADD can land.
    let members: Vec<String> = client
        .pool
        .smembers(key_schema.path_index("/stage"))
        .await
        .expect("SMEMBERS must succeed");
    assert_eq!(
        members.len(),
        1,
        "path_index must contain exactly 1 session_id after N racing same-name creates; \
         got {n} members: {members:?} (successes={successes}, already_exists={already_exists}). \
         More than 1 member indicates WATCH protection was defeated — the regression signal \
         for b.6e7 (session.rs::create reverted to the multiplexed `pool.next()`).",
        n = members.len()
    );
    assert_eq!(
        successes, 1,
        "exactly one create must return Ok (the one that landed in SADD); \
         any other Ok is a silent duplicate write"
    );

    // End-to-end confirmation: list must see exactly 1 session with the name.
    let ss_verify = SessionStore::new(client, key_schema);
    let listed = ss_verify
        .list("/stage")
        .await
        .expect("list must succeed")
        .present;
    assert_eq!(
        listed.len(),
        1,
        "list must return exactly 1 session: {listed:?}"
    );
    assert_eq!(listed[0].name, "pool-one-race");
}

// Regression for b.6e7 item 2 (companion to the same-name test above).
//
// Distinct-name creates on the same base_path all WATCH the same
// path_index key. With the correct txn_pool-backed code and pool_size=1,
// the ExclusivePool mutex serializes every caller — each write completes
// fully before the next one starts, so every create commits without ever
// tripping the retry loop.
//
// Before the fix (pool.next()), all N racers would share a single fred
// Client. fred would multiplex their WATCH/SMEMBERS/HGET/MULTI/EXEC
// pipelines on that one connection, causing each committed SADD to cancel
// the others' WATCHes (per-connection WATCH semantics). The retry loop
// would kick in and — at WRITERS > MAX_TXN_RETRIES — at least one writer
// would exhaust its retry budget and return Aborted.
//
// This test asserts the clean "every writer commits, zero aborts"
// outcome that pool_size=1 only yields when creates run through
// `acquire_txn_for`. If a future refactor reverts to `pool.next()`,
// the multiplexing on the single shared connection would cause spurious
// nil-EXECs and at least one Aborted outcome at WRITERS=8.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_session_concurrent_distinct_names_with_pool_size_one_serialize_cleanly() {
    const WRITERS: usize = 8;
    const POOL_SIZE: u32 = 1;

    let valkey_url =
        std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

    let prefix = format!("test_{}:", uuid::Uuid::new_v4());
    let key_schema = KeySchema::new(&prefix);

    let config = Config {
        catchup_snapshot_enabled: false,
        port: 0,
        metrics_port: 0,
        valkey_url,
        valkey_pool_size: POOL_SIZE,
        subscriber_pool_size: 4,
        subscriber_admission_wait_ms: 2_000,
        subscriber_max_subscriptions: 4096,
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

    let client = connect_test_client(&config)
        .await
        .expect("ValkeyClient must connect");

    let ss = SessionStore::new(client.clone(), key_schema.clone());

    let barrier = Arc::new(tokio::sync::Barrier::new(WRITERS));
    let mut handles = Vec::with_capacity(WRITERS);
    for i in 0..WRITERS {
        let ss_clone = ss.clone();
        let barrier_clone = barrier.clone();
        let name = format!("pool-one-distinct-{i}");
        handles.push(tokio::spawn(async move {
            barrier_clone.wait().await;
            (
                name.clone(),
                ss_clone.create(&name, "/stage", &format!("user-{i}")).await,
            )
        }));
    }

    let mut committed: Vec<String> = Vec::new();
    let mut aborts: Vec<String> = Vec::new();
    let mut other: Vec<String> = Vec::new();
    for h in handles {
        let (name, res) = h.await.expect("task must not panic");
        match res {
            Ok(_) => committed.push(name),
            Err(ServiceError::Aborted(msg)) => aborts.push(format!("{name}: {msg}")),
            Err(e) => other.push(format!("{name}: {e:?}")),
        }
    }

    assert!(other.is_empty(), "unexpected non-Aborted errors: {other:?}");

    // The invariant: when txn_pool serializes writers (pool_size=1 →
    // one-at-a-time mutex-guarded execution), every distinct-name create
    // commits without contention. Zero aborts is the distinguishing
    // signal — if this test fails with aborts > 0, session.rs::create has
    // regressed to `pool.next()` and writers are multiplexing onto one
    // connection, tripping WATCHes against each other.
    assert_eq!(
        aborts.len(),
        0,
        "pool_size=1 + exclusive txn_pool must serialize all {WRITERS} creates with zero aborts; \
         got {} aborts: {aborts:?}. This is the b.6e7 regression signal — \
         session.rs::create is multiplexing on the shared `pool` instead of holding \
         one exclusive txn_pool client across WATCH/MULTI/EXEC.",
        aborts.len()
    );
    assert_eq!(
        committed.len(),
        WRITERS,
        "all {WRITERS} distinct-name creates must commit cleanly under serialized txn_pool; \
         got {} commits: {committed:?}",
        committed.len()
    );

    // Cross-check with ListSessions.
    let ss_verify = SessionStore::new(client, key_schema);
    let listed = ss_verify
        .list("/stage")
        .await
        .expect("list must succeed")
        .present;
    assert_eq!(
        listed.len(),
        WRITERS,
        "ListSessions must return all {WRITERS} committed sessions; got {}: {listed:?}",
        listed.len()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_session_valkey_keys_correct() {
    let tv = TestValkey::setup().await;
    let ss = store(&tv);
    let keys = tv.key_schema();

    let sid = ss
        .create("key-check", "/stage", "carol")
        .await
        .expect("create must succeed");

    // Verify session metadata hash via HGETALL.
    let meta: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(keys.session_meta(&sid))
        .await
        .expect("HGETALL must succeed");

    assert_eq!(meta.get("name").map(String::as_str), Some("key-check"));
    assert_eq!(meta.get("base_path").map(String::as_str), Some("/stage"));
    assert_eq!(meta.get("created_by").map(String::as_str), Some("carol"));
    assert!(
        meta.contains_key("created_at"),
        "created_at field must exist"
    );

    // Verify path index set contains the session_id.
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(keys.path_index("/stage"))
        .await
        .expect("SMEMBERS must succeed");

    assert!(
        members.contains(&sid),
        "path index must contain the session_id"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// DeleteSession tests (t3.1it.ra.13.1e)
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_metadata_removed() {
    let tv = TestValkey::setup().await;
    let ss = store(&tv);
    let keys = tv.key_schema();

    let sid = ss
        .create("delete-meta", "/stage", "alice")
        .await
        .expect("create must succeed");

    ss.delete(&sid).await.expect("delete must succeed");

    // HGETALL on the deleted session must return an empty map.
    let meta: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(keys.session_meta(&sid))
        .await
        .expect("HGETALL must succeed");

    assert!(
        meta.is_empty(),
        "metadata must be empty after delete, got: {meta:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_path_index_entry_removed() {
    let tv = TestValkey::setup().await;
    let ss = store(&tv);
    let keys = tv.key_schema();

    let sid = ss
        .create("delete-path", "/stage", "alice")
        .await
        .expect("create must succeed");

    ss.delete(&sid).await.expect("delete must succeed");

    // SMEMBERS on the path index must no longer contain the session_id.
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(keys.path_index("/stage"))
        .await
        .expect("SMEMBERS must succeed");

    assert!(
        !members.contains(&sid),
        "path index must not contain deleted session_id"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_missing_returns_not_found() {
    let tv = TestValkey::setup().await;
    let ss = store(&tv);

    let err = ss
        .delete("nonexistent-session-id")
        .await
        .expect_err("delete of missing session must fail");

    assert!(
        matches!(err, ServiceError::NotFound(_)),
        "expected NotFound, got: {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_second_delete_returns_not_found() {
    let tv = TestValkey::setup().await;
    let ss = store(&tv);

    let sid = ss
        .create("idempotent-del", "/stage", "alice")
        .await
        .expect("create must succeed");

    ss.delete(&sid).await.expect("first delete must succeed");

    let err = ss.delete(&sid).await.expect_err("second delete must fail");

    assert!(
        matches!(err, ServiceError::NotFound(_)),
        "second delete must return NotFound, got: {err:?}"
    );
}

/// `delete` leaves the deletion marker `le:{sid}:sess:deleted`, with a TTL of
/// `SESSION_DELETED_MARKER_TTL_SECS` (7 days). Session recovery refuses a
/// marked session, which is what keeps a lookup made before the checkpoint
/// tombstone lands from restoring the session this delete removed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_writes_the_deletion_marker_with_its_ttl() {
    use live_edit::storage::keys::SESSION_DELETED_MARKER_TTL_SECS;

    let tv = TestValkey::setup().await;
    let ss = store(&tv);
    let marker = |sid: &str| tv.key_schema().session_deleted_marker(sid);

    let sid = ss
        .create("delete-marker", "/stage", "alice")
        .await
        .expect("create must succeed");
    let exists_before: bool = tv
        .client()
        .pool
        .exists(marker(&sid))
        .await
        .expect("EXISTS must succeed");
    assert!(!exists_before, "a live session carries no deletion marker");

    ss.delete(&sid).await.expect("delete must succeed");

    let exists_after: bool = tv
        .client()
        .pool
        .exists(marker(&sid))
        .await
        .expect("EXISTS must succeed");
    assert!(exists_after, "delete must leave the deletion marker");
    let ttl: i64 = tv
        .client()
        .pool
        .ttl(marker(&sid))
        .await
        .expect("TTL must succeed");
    // An upper bound of the configured TTL, and a lower bound a minute under
    // it, so a slow runner between the delete and this read cannot fail it.
    assert!(
        ttl <= SESSION_DELETED_MARKER_TTL_SECS && ttl > SESSION_DELETED_MARKER_TTL_SECS - 60,
        "the marker must expire after ~{SESSION_DELETED_MARKER_TTL_SECS} s (7 days), never \
         persist forever (-1) or be missing (-2); TTL is {ttl}"
    );
}

/// `deleted_marker_exists` tells a session `delete` removed apart from a live
/// one and from one merely missing from the store (a keyspace loss): the two
/// store-`NotFound` cancellers read it to choose between the deletion endings
/// and the retryable ones.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deleted_marker_exists_is_true_only_for_a_deleted_session() {
    let tv = TestValkey::setup().await;
    let ss = store(&tv);

    let live = ss
        .create("marker-live", "/stage", "alice")
        .await
        .expect("create must succeed");
    let deleted = ss
        .create("marker-deleted", "/stage", "alice")
        .await
        .expect("create must succeed");
    let flushed = ss
        .create("marker-flushed", "/stage", "alice")
        .await
        .expect("create must succeed");
    ss.delete(&deleted).await.expect("delete must succeed");
    let removed: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().session_meta(&flushed))
        .await
        .expect("DEL session meta must succeed");
    assert_eq!(
        removed, 1,
        "fixture: the flushed session's metadata existed"
    );

    for (sid, want, what) in [
        (&live, false, "a live session"),
        (&deleted, true, "a session DeleteSession removed"),
        (
            &flushed,
            false,
            "a session lost from the keyspace without a delete",
        ),
        (
            &"never-existed".to_owned(),
            false,
            "a session id that never existed",
        ),
    ] {
        assert_eq!(
            ss.deleted_marker_exists(sid)
                .await
                .expect("marker read must succeed"),
            want,
            "{what}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ListSessions tests (t3.1it.ra.13.s3)
// ═══════════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_empty_returns_empty_vec() {
    let tv = TestValkey::setup().await;
    let ss = store(&tv);

    let result = ss.list("/empty-path").await.expect("list must succeed");

    assert!(
        result.present.is_empty(),
        "list on empty path must return empty present vec"
    );
    assert!(
        result.ghosts.is_empty(),
        "list on empty path must return empty ghosts vec"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_single_session_returned() {
    let tv = TestValkey::setup().await;
    let ss = store(&tv);

    let sid = ss
        .create("single", "/stage", "alice")
        .await
        .expect("create must succeed");

    let sessions = ss.list("/stage").await.expect("list must succeed").present;
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].session_id, sid);
    assert_eq!(sessions[0].name, "single");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_multiple_sessions_all_returned() {
    let tv = TestValkey::setup().await;
    let ss = store(&tv);

    let sid1 = ss.create("sess-a", "/stage", "alice").await.unwrap();
    let sid2 = ss.create("sess-b", "/stage", "bob").await.unwrap();
    let sid3 = ss.create("sess-c", "/stage", "carol").await.unwrap();

    let sessions = ss.list("/stage").await.expect("list must succeed").present;
    assert_eq!(sessions.len(), 3, "all 3 sessions must be returned");

    let ids: Vec<&str> = sessions.iter().map(|s| s.session_id.as_str()).collect();
    assert!(ids.contains(&sid1.as_str()), "must contain sid1");
    assert!(ids.contains(&sid2.as_str()), "must contain sid2");
    assert!(ids.contains(&sid3.as_str()), "must contain sid3");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_path_isolation() {
    let tv = TestValkey::setup().await;
    let ss = store(&tv);

    let sid_a = ss.create("on-path-a", "/path-a", "alice").await.unwrap();
    let sid_b = ss.create("on-path-b", "/path-b", "bob").await.unwrap();

    let sessions_a = ss
        .list("/path-a")
        .await
        .expect("list path-a must succeed")
        .present;
    assert_eq!(sessions_a.len(), 1, "path-a must have exactly 1 session");
    assert_eq!(sessions_a[0].session_id, sid_a);

    let sessions_b = ss
        .list("/path-b")
        .await
        .expect("list path-b must succeed")
        .present;
    assert_eq!(sessions_b.len(), 1, "path-b must have exactly 1 session");
    assert_eq!(sessions_b[0].session_id, sid_b);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn list_sessions_orphan_surfaced_as_ghost_not_removed() {
    // Recovery-aware contract (Epic `t1.fgj.7v`, SDD SR-2.7): `SessionStore::list`
    // no longer SREMs members on empty-HGETALL. Instead it surfaces them as
    // `ListResult::ghosts`; the caller (today `handle_list_sessions`)
    // discriminates between "recoverable" and "true orphan" via
    // `try_recover_session` and SREMs only on `NoCheckpoint` / disabled-hook.
    let tv = TestValkey::setup().await;
    let ss = store(&tv);
    let keys = tv.key_schema();

    // Create two sessions.
    let sid_good = ss.create("good", "/stage", "alice").await.unwrap();
    let sid_orphan = ss.create("orphan", "/stage", "bob").await.unwrap();

    // Manually delete the orphan's metadata hash to simulate an orphaned index entry.
    tv.client()
        .pool
        .del::<(), _>(keys.session_meta(&sid_orphan))
        .await
        .expect("DEL orphan metadata must succeed");

    // List must surface the good session in `present` and the orphaned
    // entry in `ghosts`, with the composite path-name member pre-derived.
    let result = ss.list("/stage").await.expect("list must succeed");
    assert_eq!(
        result.present.len(),
        1,
        "good session must appear in `present`"
    );
    assert_eq!(result.present[0].session_id, sid_good);
    assert_eq!(
        result.ghosts.len(),
        1,
        "orphan member must surface as a ghost"
    );
    assert_eq!(result.ghosts[0].sid, sid_orphan);
    assert_eq!(result.ghosts[0].name, "orphan");
    assert_eq!(
        result.ghosts[0].composite_member,
        live_edit::storage::keys::KeySchema::path_name_member("orphan", &sid_orphan),
    );

    // `list` itself MUST NOT SREM — the path-index still carries the
    // orphan member after the call. The caller orchestrates SREM only on
    // the recovery-rejected path.
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(keys.path_index("/stage"))
        .await
        .expect("SMEMBERS must succeed");
    assert!(
        members.contains(&sid_orphan),
        "ghost member MUST NOT be auto-SREM'd by SessionStore::list under the recovery-aware contract; got members={members:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// DeleteSession atomicity tests — b.yis item 3
// ═══════════════════════════════════════════════════════════════════════════════
//
// Pre-fix `SessionStore::delete` issued three unconditionally-pipelined
// commands on the shared multiplexed pool:
//
//   1. HGETALL session_meta    (read base_path)
//   2. DEL     session_meta    (remove metadata)
//   3. SREM    path_index      (remove from index)
//   4. DEL     layers_index    (clear cascaded layer index)
//
// If the caller's future was dropped (deadline timeout, cancellation) or the
// connection died between steps 2 and 3, the session_meta hash vanished while
// the path-index still pointed at the now-orphaned session_id. `list` would
// then observe an orphan (metadata missing but id listed) and *opportunistically*
// clean it up — but only on the next list call, and only for that one path.
// The dangling index entry was real, externally visible state.
//
// The b.yis fix wraps the three mutations in a MULTI/EXEC on the `{session_id}`
// slot via `acquire_txn_for`, so either all three mutations land atomically
// or none of them do. (path_index uses a different hash tag and so cannot live
// inside a MULTI on Valkey Cluster; the fix orders the SREM relative to the
// MULTI so that the "dangling index entry" mode is unreachable — either the
// SREM runs first and a crash leaves meta-present-but-not-indexed (list-safe,
// invisible), or the SREM runs after the MULTI and a crash between the two
// leaves meta-gone-and-not-indexed (both safe).)
//
// Without a fault-injection seam inside `SessionStore::delete` we cannot
// directly simulate a crash mid-sequence. The tests below instead assert the
// properties that a correct atomic/ordered implementation guarantees from
// the outside:
//
//   * `delete_session_all_three_keys_removed_together` — on the happy path,
//     every key the delete touches is gone when the call returns, not "some
//     subset depending on internal ordering".
//   * `delete_session_concurrent_delete_no_path_index_orphan` — two racers
//     calling delete() on the same sid produce exactly one Ok + one NotFound,
//     and the path-index entry is gone regardless of interleaving.
//   * `delete_session_post_delete_list_is_empty_not_orphaned` — after a
//     successful delete, `list(base_path)` returns empty and does NOT log
//     an "orphan skipped" (we can't assert on logs cheaply, but we can
//     assert that the path_index set itself is empty — a stronger signal
//     that the SREM and DEL landed in the same committed unit).

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_all_three_keys_removed_together() {
    // End-state atomicity check: after a successful delete, session_meta,
    // path_index membership, AND layers_index must all be gone. This is the
    // invariant the MULTI/EXEC fix guarantees: a single committed transaction
    // removes every key the session owned — the caller cannot observe a
    // partial-delete state from outside.
    let tv = TestValkey::setup().await;
    let ss = store(&tv);
    let keys = tv.key_schema();

    let sid = ss
        .create("atomic-del", "/stage", "alice")
        .await
        .expect("create must succeed");

    // Sanity: all three keys exist before the delete.
    let meta_before: bool = tv
        .client()
        .pool
        .exists(keys.session_meta(&sid))
        .await
        .expect("EXISTS session_meta must succeed");
    let index_members_before: Vec<String> = tv
        .client()
        .pool
        .smembers(keys.path_index("/stage"))
        .await
        .expect("SMEMBERS must succeed");
    assert!(meta_before, "pre-delete: session_meta must exist");
    assert!(
        index_members_before.contains(&sid),
        "pre-delete: path_index must contain sid"
    );

    ss.delete(&sid).await.expect("delete must succeed");

    // Post-delete: every key is gone. Before the fix these three checks
    // could individually pass (each command eventually ran), but under a
    // cancellation mid-sequence one or more would fail — only an atomic
    // MULTI/EXEC (or the ordered SREM-then-MULTI) prevents the orphan mode.
    let meta_after: bool = tv
        .client()
        .pool
        .exists(keys.session_meta(&sid))
        .await
        .expect("EXISTS session_meta must succeed");
    let index_members_after: Vec<String> = tv
        .client()
        .pool
        .smembers(keys.path_index("/stage"))
        .await
        .expect("SMEMBERS must succeed");
    let layers_after: bool = tv
        .client()
        .pool
        .exists(keys.layers_index(&sid))
        .await
        .expect("EXISTS layers_index must succeed");

    assert!(!meta_after, "post-delete: session_meta must be gone");
    assert!(
        !index_members_after.contains(&sid),
        "post-delete: path_index must not contain sid; members={index_members_after:?}"
    );
    assert!(!layers_after, "post-delete: layers_index must be gone");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delete_session_concurrent_delete_no_path_index_orphan() {
    // Two concurrent `delete(sid)` calls race. The post-fix contract is
    // end-state consistency:
    //
    //   * At least one returns Ok (someone committed the delete).
    //   * Neither returns Internal or Aborted. (The only acceptable second
    //     outcome is Ok — if the implementation adds a WATCH on session_meta,
    //     it'd be NotFound; if it just uses MULTI/EXEC without WATCH, a
    //     redundant DEL is a no-op and both racers return Ok. Either is fine
    //     — what must NOT happen is a bubbled-up error from a redundant
    //     operation on now-missing state.)
    //   * Regardless of interleaving, session_meta, path_index membership,
    //     and layers_index are all gone when the dust settles.
    //
    // Before the fix, the three non-atomic commands ran directly against
    // the shared multiplexed pool. Interleavings were possible where the
    // second racer's SREM or DEL ran against keys the first racer had
    // already removed — harmless in Valkey semantics (SREM of an absent
    // member returns 0, DEL of an absent key returns 0) but that's not
    // what's being exercised here. The real failure mode is when a
    // *future is dropped* mid-sequence: DEL commits, SREM never runs,
    // and the path_index holds a dangling pointer. We can't simulate a
    // dropped future directly (no seam), but we can assert the invariant
    // the MULTI/EXEC fix establishes under concurrent success: the state
    // is always all-gone at the end, never half-applied.
    let tv = TestValkey::setup().await;
    let ss = store(&tv);
    let keys = tv.key_schema();

    let sid = ss
        .create("concurrent-del", "/stage", "alice")
        .await
        .expect("create must succeed");

    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let ss_a = ss.clone();
    let ss_b = ss.clone();
    let sid_a = sid.clone();
    let sid_b = sid.clone();
    let b_a = barrier.clone();
    let b_b = barrier.clone();

    let h_a = tokio::spawn(async move {
        b_a.wait().await;
        ss_a.delete(&sid_a).await
    });
    let h_b = tokio::spawn(async move {
        b_b.wait().await;
        ss_b.delete(&sid_b).await
    });

    let r_a = h_a.await.expect("task a must not panic");
    let r_b = h_b.await.expect("task b must not panic");

    let mut ok = 0u32;
    let mut not_found = 0u32;
    for r in [&r_a, &r_b] {
        match r {
            Ok(()) => ok += 1,
            Err(ServiceError::NotFound(_)) => not_found += 1,
            Err(e) => panic!(
                "unexpected delete error on concurrent race — \
                 MULTI/EXEC must not bubble up Internal/Aborted from \
                 redundant ops on already-deleted state: {e:?}"
            ),
        }
    }
    assert!(
        ok >= 1,
        "at least one concurrent delete must succeed; got {ok} Ok, {not_found} NotFound"
    );
    assert_eq!(
        ok + not_found,
        2,
        "every delete must return Ok or NotFound; got {ok} Ok, {not_found} NotFound"
    );

    // Load-bearing invariant: the end state is fully consistent regardless
    // of which racer committed first. This is what the MULTI/EXEC fix
    // guarantees — the SREM/DEL pair is visible together, so no externally
    // observable state can ever be half-deleted.
    let meta_after: bool = tv
        .client()
        .pool
        .exists(keys.session_meta(&sid))
        .await
        .expect("EXISTS session_meta must succeed");
    let index_after: Vec<String> = tv
        .client()
        .pool
        .smembers(keys.path_index("/stage"))
        .await
        .expect("SMEMBERS must succeed");
    let layers_after: bool = tv
        .client()
        .pool
        .exists(keys.layers_index(&sid))
        .await
        .expect("EXISTS layers_index must succeed");
    assert!(
        !meta_after,
        "post-concurrent-delete: session_meta must be gone"
    );
    assert!(
        !index_after.contains(&sid),
        "post-concurrent-delete: path_index must not contain sid; members={index_after:?}"
    );
    assert!(
        !layers_after,
        "post-concurrent-delete: layers_index must be gone"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_post_delete_list_is_empty_not_orphaned() {
    // After a successful delete, `list(base_path)` must return an empty
    // result AND the path_index set itself must be empty. The stronger
    // half of this assertion is the second: `list` already skips orphans
    // (metadata missing but id in index) and opportunistically cleans
    // them up, so observing an empty `list` does not by itself prove that
    // the SREM committed. Checking the raw path_index via SMEMBERS is what
    // distinguishes "SREM landed in the delete commit" from "list papered
    // over an orphan on next call".
    let tv = TestValkey::setup().await;
    let ss = store(&tv);
    let keys = tv.key_schema();

    let sid = ss
        .create("list-after-del", "/stage", "alice")
        .await
        .expect("create must succeed");

    ss.delete(&sid).await.expect("delete must succeed");

    // list(base_path) sees no session.
    let listed = ss.list("/stage").await.expect("list must succeed");
    assert!(
        listed.present.is_empty() && listed.ghosts.is_empty(),
        "list must return empty after delete: present={:?} ghosts={:?}",
        listed.present,
        listed.ghosts,
    );

    // The path_index is truly empty — not just list-cleaned on the fly.
    // This is the post-fix signal: the SREM ran as part of the delete
    // commit, not as a side-effect of a later list call's orphan sweep.
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(keys.path_index("/stage"))
        .await
        .expect("SMEMBERS must succeed");
    assert!(
        members.is_empty(),
        "path_index must be empty after delete (no dangling orphan); members={members:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_session_one_of_many_leaves_siblings_intact() {
    // Regression guard for "delete removed too much". The fix wraps the
    // session_meta + layers_index DELs in a MULTI, and the SREM removes
    // exactly one member from the path_index — other sessions on the same
    // base_path must be untouched.
    //
    // This catches a plausible mis-implementation: if the engineer keyed
    // the MULTI's WATCH/operations on the wrong hash tag, or accidentally
    // used DEL on path_index instead of SREM, the whole path_index would
    // vanish and every sibling session would become an orphan.
    let tv = TestValkey::setup().await;
    let ss = store(&tv);
    let keys = tv.key_schema();

    let sid_keep_a = ss
        .create("sibling-a", "/stage", "alice")
        .await
        .expect("create sibling-a must succeed");
    let sid_del = ss
        .create("to-delete", "/stage", "bob")
        .await
        .expect("create to-delete must succeed");
    let sid_keep_b = ss
        .create("sibling-b", "/stage", "carol")
        .await
        .expect("create sibling-b must succeed");

    ss.delete(&sid_del).await.expect("delete must succeed");

    // Deleted session is gone on all three fronts.
    let deleted_meta: bool = tv
        .client()
        .pool
        .exists(keys.session_meta(&sid_del))
        .await
        .expect("EXISTS must succeed");
    assert!(!deleted_meta, "deleted session_meta must be gone");

    // Siblings' metadata and index entries are intact.
    for (sid, name) in [(&sid_keep_a, "sibling-a"), (&sid_keep_b, "sibling-b")] {
        let meta: HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(keys.session_meta(sid))
            .await
            .expect("HGETALL must succeed");
        assert_eq!(
            meta.get("name").map(String::as_str),
            Some(name),
            "sibling {sid} metadata must be intact: {meta:?}"
        );
    }

    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(keys.path_index("/stage"))
        .await
        .expect("SMEMBERS must succeed");
    assert!(
        !members.contains(&sid_del),
        "deleted sid must be gone from path_index"
    );
    assert!(
        members.contains(&sid_keep_a),
        "sibling-a must still be in path_index"
    );
    assert!(
        members.contains(&sid_keep_b),
        "sibling-b must still be in path_index"
    );
    assert_eq!(members.len(), 2, "exactly 2 siblings remain: {members:?}");

    // End-to-end confirmation via list.
    let listed = ss.list("/stage").await.expect("list must succeed").present;
    assert_eq!(listed.len(), 2, "list must return exactly 2 survivors");
    let names: Vec<&str> = listed.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"sibling-a"), "sibling-a must be listed");
    assert!(names.contains(&"sibling-b"), "sibling-b must be listed");
}

// ═══════════════════════════════════════════════════════════════════════════════
// b.91j C-7: duplicate-name race is closed by the same-slot names index
// ═══════════════════════════════════════════════════════════════════════════════
//
// Pre-fix `SessionStore::create` committed `SADD path_index` inside MULTI
// but `HSET session_meta` ran AFTER EXEC on a different Cluster slot. A
// concurrent `create(name, base_path)` could observe SMEMBERS → session_id
// present in the index but `HGET session_meta name` returns `None` (HSET
// not yet run) → conclude "no duplicate" and commit a second session with
// the same name.
//
// The fix WATCHes and SADDs the new `path_names_index` set (composite
// `name\0session_id` members) in a single MULTI/EXEC — uniqueness lives
// entirely on one slot now. The tests below drive the race:
//
//   * The high-iteration loop reliably triggered the pre-fix race within
//     a handful of iterations under load; the sibling `create_session_
//     concurrent_same_name_exactly_one_succeeds` test already exists but
//     it asserts the weaker "at most one in path_index" invariant that
//     the pre-b.91j `{hex(base_path)}`-slot SADD already satisfied. These
//     new tests lock in the C-7-specific invariant on the authoritative
//     `path_names_index` set.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_create_duplicate_name_race_at_most_one_succeeds() {
    // Repeat the 2-way race ~32 times to exercise the window. The
    // previous per-iteration window was on the order of microseconds;
    // running many iterations makes the "at most one succeeds"
    // invariant a load-bearing guarantee rather than a fortunate
    // scheduling outcome. Each iteration lives on its own isolated
    // base_path so stale state from prior iterations cannot leak into
    // the next one.
    const ITERS: usize = 32;
    const BASE_PATH_PREFIX: &str = "/stage/c7-race";

    let tv = TestValkey::setup_with_pool_size(8).await;
    let ss = store(&tv);

    for i in 0..ITERS {
        let base_path = format!("{BASE_PATH_PREFIX}-{i}");
        let name = "conflict-name";

        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        let ss_a = ss.clone();
        let ss_b = ss.clone();
        let bp_a = base_path.clone();
        let bp_b = base_path.clone();
        let b_a = barrier.clone();
        let b_b = barrier.clone();

        let h_a = tokio::spawn(async move {
            b_a.wait().await;
            ss_a.create(name, &bp_a, "alice").await
        });
        let h_b = tokio::spawn(async move {
            b_b.wait().await;
            ss_b.create(name, &bp_b, "bob").await
        });

        let r_a = h_a.await.expect("task a must not panic");
        let r_b = h_b.await.expect("task b must not panic");

        let mut ok = 0u32;
        let mut already_exists = 0u32;
        let mut aborted = 0u32;
        for r in [&r_a, &r_b] {
            match r {
                Ok(_) => ok += 1,
                Err(ServiceError::AlreadyExists(_)) => already_exists += 1,
                // The fix's retry loop (H-stor-3) is generous (32 attempts)
                // and the race window is tiny, so Aborted is theoretically
                // possible but should not fire at iteration scale this
                // small. Count it for the assert below so a regression
                // showing excess aborts is explicit rather than swallowed.
                Err(ServiceError::Aborted(_)) => aborted += 1,
                Err(e) => panic!(
                    "iter {i}: unexpected error — duplicate-name race must \
                     resolve to Ok OR AlreadyExists (or at worst Aborted \
                     after retry budget); got: {e:?}"
                ),
            }
        }

        // Core invariant: at most one committed successfully. The pre-fix
        // race could yield `ok == 2`; post-fix, the same-slot SADD + WATCH
        // makes that unreachable.
        assert!(
            ok <= 1,
            "iter {i}: at most one concurrent create may succeed, got {ok} Ok, \
             {already_exists} AlreadyExists, {aborted} Aborted"
        );
        // Cannot have both failing with AlreadyExists before any succeeds
        // (uniqueness requires at least one committed write to reject
        // the other), unless the retry loop exhausts — which we surface
        // separately.
        if aborted == 0 {
            assert_eq!(
                ok, 1,
                "iter {i}: with no aborts, exactly one create must succeed; \
                 got {ok} Ok, {already_exists} AlreadyExists"
            );
            assert_eq!(
                already_exists, 1,
                "iter {i}: the other create must reject with AlreadyExists; \
                 got {ok} Ok, {already_exists} AlreadyExists"
            );
        }

        // End-state invariant: the authoritative names index contains
        // exactly one member naming `conflict-name`. This is the load-
        // bearing C-7 assertion — it fires a regression even if the
        // outcome counters look plausible (e.g. both Ok but only one
        // landed in storage under a broken fix).
        let path_names_key = tv.key_schema().path_names_index(&base_path);
        let members: Vec<String> = tv
            .client()
            .pool
            .smembers(&path_names_key)
            .await
            .expect("SMEMBERS must succeed");
        let matching: Vec<&String> = members
            .iter()
            .filter(|m| m.starts_with(&format!("{name}\0")))
            .collect();
        assert!(
            matching.len() <= 1,
            "iter {i}: path_names_index must contain at most one member for \
             name '{name}'; got {} — members: {members:?}",
            matching.len()
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_create_10_way_same_name_race_only_one_commits_in_names_index() {
    // Broader race: 10 concurrent creates with the same name. The
    // sibling test `create_session_concurrent_same_name_exactly_one_succeeds`
    // checks `path_index` (session-id set); this one pins the C-7-specific
    // invariant on the newly-authoritative `path_names_index` (composite
    // name/session-id set).
    //
    // Uses `setup_with_pool_size(16)` so every racer has its own txn-pool
    // connection — exclusive-connection semantics are required for WATCH
    // to correctly protect each caller against the others.
    const WRITERS: usize = 10;

    let tv = TestValkey::setup_with_pool_size(WRITERS as u32 + 4).await;
    let ss = store(&tv);

    let barrier = Arc::new(tokio::sync::Barrier::new(WRITERS));
    let mut handles = Vec::with_capacity(WRITERS);
    for i in 0..WRITERS {
        let ss_clone = ss.clone();
        let barrier_clone = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier_clone.wait().await;
            ss_clone
                .create("fanout-race", "/stage/c7-wide", &format!("user-{i}"))
                .await
        }));
    }

    let mut ok = 0u32;
    let mut already_exists = 0u32;
    let mut aborted = 0u32;
    for h in handles {
        match h.await.expect("task must not panic") {
            Ok(_) => ok += 1,
            Err(ServiceError::AlreadyExists(_)) => already_exists += 1,
            Err(ServiceError::Aborted(_)) => aborted += 1,
            Err(e) => panic!("unexpected error in 10-way race: {e:?}"),
        }
    }

    assert_eq!(
        ok + already_exists + aborted,
        WRITERS as u32,
        "every racer must return Ok / AlreadyExists / Aborted; got {ok} Ok, \
         {already_exists} AlreadyExists, {aborted} Aborted"
    );
    assert!(
        ok >= 1,
        "at least one concurrent create must commit; got {ok} Ok, \
         {already_exists} AlreadyExists, {aborted} Aborted"
    );
    assert!(
        ok <= 1,
        "at most one create may commit with the same name + base_path \
         (C-7 invariant); got {ok} Ok"
    );

    // Names index post-state: exactly one member prefixed by
    // `"fanout-race\0"`. Pre-fix a second HSET race could land a second
    // session_meta with the duplicate name even though path_index held
    // only one sid; post-fix the names set is the uniqueness surface and
    // can only hold one member per (name, base_path).
    let path_names_key = tv.key_schema().path_names_index("/stage/c7-wide");
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(&path_names_key)
        .await
        .expect("SMEMBERS must succeed");
    let matching_count = members
        .iter()
        .filter(|m| m.starts_with("fanout-race\0"))
        .count();
    assert_eq!(
        matching_count, 1,
        "path_names_index must contain exactly 1 member prefixed 'fanout-race\\0'; \
         got {matching_count} — members: {members:?}"
    );

    // Also confirm via `list` — the end-user-visible surface.
    let listed = ss
        .list("/stage/c7-wide")
        .await
        .expect("list must succeed")
        .present;
    let named: Vec<&live_edit::storage::session::SessionMeta> =
        listed.iter().filter(|s| s.name == "fanout-race").collect();
    assert_eq!(
        named.len(),
        1,
        "list must return exactly 1 session named 'fanout-race'; got {named:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_create_duplicate_name_sequential_returns_already_exists() {
    // Sequential baseline for C-7: the uniqueness check on the names
    // index must still reject a straightforward second `create` with
    // the same name, now that the uniqueness surface moved from
    // `HGET session_meta name` to `SMEMBERS path_names_index`. This
    // pins the happy-path non-race case so a refactor dropping the
    // SMEMBERS loop doesn't silently accept dupes.
    let tv = TestValkey::setup().await;
    let ss = store(&tv);

    let sid1 = ss
        .create("seq-dup", "/stage/seq", "alice")
        .await
        .expect("first create must succeed");

    let err = ss
        .create("seq-dup", "/stage/seq", "bob")
        .await
        .expect_err("second create must fail");
    assert!(
        matches!(err, ServiceError::AlreadyExists(ref m) if m.contains("seq-dup")),
        "expected AlreadyExists with name in message, got: {err:?}"
    );

    // Names index still contains exactly one member, namespaced to the
    // first session_id.
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().path_names_index("/stage/seq"))
        .await
        .expect("SMEMBERS must succeed");
    let expected = format!("seq-dup\0{sid1}");
    assert_eq!(members, vec![expected]);
}
