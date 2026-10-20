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

// Integration tests for the TestValkey fixture: prefix isolation, uniqueness,
// drop cleanup, cross-prefix safety, and parallel safety.
//
// Requires a running Valkey instance. Address is read from TEST_VALKEY_URL,
// defaulting to redis://127.0.0.1:6379. Tests skip gracefully if Valkey is
// unreachable.

use std::collections::HashSet;
use std::time::Duration;

use fred::interfaces::{ClientLike, KeysInterface};
use fred::prelude::Builder;
use fred::types::config::Config as FredConfig;
use futures::StreamExt;

use live_edit::storage::test_support::TestValkey;

/// Connect a raw fred pool (not TestValkey) for verification queries.
/// Returns None if Valkey is unreachable.
async fn raw_pool() -> Option<fred::clients::Pool> {
    let url =
        std::env::var("TEST_VALKEY_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let config = FredConfig::from_url(&url).ok()?;
    let pool = Builder::from_config(config).build_pool(1).ok()?;
    let connect_result = tokio::time::timeout(Duration::from_secs(5), pool.init()).await;
    match connect_result {
        Ok(Ok(_)) => Some(pool),
        _ => None,
    }
}

/// Count keys matching `pattern` via SCAN on the given pool.
async fn count_keys(pool: &fred::clients::Pool, pattern: &str) -> usize {
    let client = pool.next();
    let mut stream = client.scan_buffered(pattern, Some(100), None);
    let mut count = 0;
    while let Some(Ok(_)) = stream.next().await {
        count += 1;
    }
    count
}

// ── 1. Prefix isolation ──────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_valkey_two_fixtures_same_key_name_see_different_values() {
    let fixture_a: TestValkey = TestValkey::setup().await;
    let fixture_b: TestValkey = TestValkey::setup().await;

    let key_name = "shared_name";
    let key_a = format!("{}{key_name}", fixture_a.prefix());
    let key_b = format!("{}{key_name}", fixture_b.prefix());

    fixture_a
        .client()
        .pool
        .set::<(), _, _>(&key_a, "value_a", None, None, false)
        .await
        .expect("SET on fixture A must succeed");

    fixture_b
        .client()
        .pool
        .set::<(), _, _>(&key_b, "value_b", None, None, false)
        .await
        .expect("SET on fixture B must succeed");

    let got_a: String = fixture_a
        .client()
        .pool
        .get(&key_a)
        .await
        .expect("GET on fixture A must succeed");

    let got_b: String = fixture_b
        .client()
        .pool
        .get(&key_b)
        .await
        .expect("GET on fixture B must succeed");

    assert_eq!(got_a, "value_a", "fixture A must see its own value");
    assert_eq!(got_b, "value_b", "fixture B must see its own value");

    // The important point: same logical name, different actual keys.
    assert_ne!(
        key_a, key_b,
        "actual Valkey keys must differ due to different prefixes"
    );
}

// ── 2. Prefix uniqueness ─────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_valkey_100_instances_all_have_distinct_prefixes() {
    // Build 100 fixtures with bounded concurrency. Sequential setup runs
    // ~180 ms per instance (each `TestValkey::setup` spins up a fresh
    // ValkeyClient with its three pools); a fully-parallel `join_all`
    // floods Valkey with ~1200 simultaneous TCP handshakes and degrades
    // under suite contention. Chunks of 8 shave the single-test
    // wall-clock from ~17 s to ~13 s when run in isolation; under full
    // suite load every test on the shared Valkey is competing, so the
    // suite-wide win is small — the bounded form just avoids making
    // things worse than serial.
    use futures::stream::{FuturesUnordered, StreamExt};
    const CONCURRENCY: usize = 8;
    let mut in_flight = FuturesUnordered::new();
    let mut fixtures: Vec<TestValkey> = Vec::with_capacity(100);
    let mut to_spawn = 100usize;
    while to_spawn > 0 || !in_flight.is_empty() {
        while in_flight.len() < CONCURRENCY && to_spawn > 0 {
            in_flight.push(TestValkey::setup());
            to_spawn -= 1;
        }
        if let Some(f) = in_flight.next().await {
            fixtures.push(f);
        }
    }

    let prefixes: HashSet<String> = fixtures.iter().map(|f| f.prefix().to_string()).collect();

    assert_eq!(
        prefixes.len(),
        100,
        "all 100 TestValkey instances must have distinct prefixes"
    );
    // Keep `fixtures` alive past the assertion: each fixture's `Drop`
    // cleans Valkey keys, so we don't want to start dropping mid-assert.
    drop(fixtures);
}

// ── 3. Drop cleanup ─────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_valkey_drop_removes_all_prefixed_keys() {
    let Some(raw) = raw_pool().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let prefix;
    {
        let fixture: TestValkey = TestValkey::setup().await;
        prefix = fixture.prefix().to_string();

        // Write 10 keys under this fixture's prefix.
        for i in 0..10 {
            let key = format!("{prefix}key_{i}");
            fixture
                .client()
                .pool
                .set::<(), _, _>(&key, format!("val_{i}"), None, None, false)
                .await
                .expect("SET must succeed");
        }

        // Verify keys exist before drop.
        let before = count_keys(&raw, &format!("{prefix}*")).await;
        assert_eq!(before, 10, "10 keys must exist before drop");

        // fixture is dropped here at end of scope, triggering cleanup
    }

    // After drop, verify zero keys remain with this prefix.
    let after = count_keys(&raw, &format!("{prefix}*")).await;
    assert_eq!(
        after, 0,
        "all keys with the dropped fixture's prefix must be cleaned up"
    );
}

// ── 4. Cleanup does not affect other prefixes ────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_valkey_drop_one_fixture_preserves_other_fixture_keys() {
    let Some(raw) = raw_pool().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let fixture_survivor: TestValkey = TestValkey::setup().await;
    let survivor_prefix = fixture_survivor.prefix().to_string();

    // Write keys on the survivor fixture.
    for i in 0..5 {
        let key = format!("{survivor_prefix}kept_{i}");
        fixture_survivor
            .client()
            .pool
            .set::<(), _, _>(&key, format!("val_{i}"), None, None, false)
            .await
            .expect("SET on survivor must succeed");
    }

    {
        let fixture_doomed: TestValkey = TestValkey::setup().await;
        let doomed_prefix = fixture_doomed.prefix().to_string();

        // Write keys on the doomed fixture.
        for i in 0..5 {
            let key = format!("{doomed_prefix}doomed_{i}");
            fixture_doomed
                .client()
                .pool
                .set::<(), _, _>(&key, format!("val_{i}"), None, None, false)
                .await
                .expect("SET on doomed must succeed");
        }

        // fixture_doomed is dropped here
    }

    // Survivor's keys must still be present.
    let remaining = count_keys(&raw, &format!("{survivor_prefix}*")).await;
    assert_eq!(
        remaining, 5,
        "survivor fixture's keys must not be affected by dropping another fixture"
    );
}

// ── 5. Parallel safety ──────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_valkey_parallel_tasks_no_cross_contamination() {
    // Spawn 8 concurrent tasks, each on its own tokio task (so Drop's
    // block_in_place works correctly), then join! on all handles.
    let h0 = tokio::spawn(parallel_worker(0));
    let h1 = tokio::spawn(parallel_worker(1));
    let h2 = tokio::spawn(parallel_worker(2));
    let h3 = tokio::spawn(parallel_worker(3));
    let h4 = tokio::spawn(parallel_worker(4));
    let h5 = tokio::spawn(parallel_worker(5));
    let h6 = tokio::spawn(parallel_worker(6));
    let h7 = tokio::spawn(parallel_worker(7));

    let (r0, r1, r2, r3, r4, r5, r6, r7) = tokio::join!(h0, h1, h2, h3, h4, h5, h6, h7);

    assert!(r0.expect("task 0 panicked"), "worker 0 must succeed");
    assert!(r1.expect("task 1 panicked"), "worker 1 must succeed");
    assert!(r2.expect("task 2 panicked"), "worker 2 must succeed");
    assert!(r3.expect("task 3 panicked"), "worker 3 must succeed");
    assert!(r4.expect("task 4 panicked"), "worker 4 must succeed");
    assert!(r5.expect("task 5 panicked"), "worker 5 must succeed");
    assert!(r6.expect("task 6 panicked"), "worker 6 must succeed");
    assert!(r7.expect("task 7 panicked"), "worker 7 must succeed");
}

/// A single parallel worker: creates its own TestValkey, writes keys with
/// a worker-specific value, reads them back, and verifies correctness.
async fn parallel_worker(worker_id: u32) -> bool {
    let fixture: TestValkey = TestValkey::setup().await;
    let prefix = fixture.prefix().to_string();
    let expected_value = format!("worker_{worker_id}");

    // Write 5 keys per worker.
    for i in 0..5 {
        let key = format!("{prefix}item_{i}");
        if fixture
            .client()
            .pool
            .set::<(), _, _>(&key, &expected_value, None, None, false)
            .await
            .is_err()
        {
            return false;
        }
    }

    // Read back and verify each key holds this worker's value (not another's).
    for i in 0..5 {
        let key = format!("{prefix}item_{i}");
        let got: Result<String, _> = fixture.client().pool.get(&key).await;
        match got {
            Ok(val) if val == expected_value => {}
            _ => return false,
        }
    }

    true
}
