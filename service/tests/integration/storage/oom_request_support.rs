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

//! Reading the GC's out-of-memory collection request off the store (b.pa1),
//! shared by the tests that stage a store write refused `-OOM` and check that
//! it requested a collection: `store_write_refusal_test.rs` and
//! `b_ua9_tests.rs`.
//!
//! A store write refused `-OOM` is noted on the receiving client's store
//! out-of-memory signal, which, once armed and at most once a second, writes
//! the request `le:ext:{gc}:oom:request` (its value the armed replica label,
//! its lifetime `OOM_COLLECTION_REQUEST_TTL`, 10 s) from a spawned task
//! (`src/storage/store_oom.rs`). So a test polls for it rather than reading
//! it once.

use std::time::Duration;

use fred::interfaces::KeysInterface;

use live_edit::storage::test_support::TestValkey;

/// Deadline on the request reaching the store after the refusal.
pub(crate) const OOM_REQUEST_BUDGET: Duration = Duration::from_secs(10);

/// Interval between probes of the request key.
const POLL: Duration = Duration::from_millis(20);

/// The request's label once it is in `tv`'s keyspace, and its remaining
/// lifetime (`PTTL`, read just after); fails after [`OOM_REQUEST_BUDGET`]
/// without one. Poll as soon as the refusal can have been received: the
/// request lives 10 s.
pub(crate) async fn wait_for_oom_request(tv: &TestValkey) -> (String, i64) {
    let key = tv.key_schema().ext_oom_request();
    let pool = &tv.client().pool;
    let deadline = tokio::time::Instant::now() + OOM_REQUEST_BUDGET;
    loop {
        let label: Option<String> = pool
            .get(&key)
            .await
            .expect("GET of the out-of-memory collection request must succeed");
        if let Some(label) = label {
            let pttl: i64 = pool
                .pttl(&key)
                .await
                .expect("PTTL of the out-of-memory collection request must succeed");
            return (label, pttl);
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "a store write refused -OOM must request the GC's out-of-memory collection: no \
             request within {OOM_REQUEST_BUDGET:?}"
        );
        tokio::time::sleep(POLL).await;
    }
}
