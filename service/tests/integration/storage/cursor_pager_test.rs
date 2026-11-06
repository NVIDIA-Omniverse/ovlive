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

//! The cursor pager (`storage::retirement::page_by_cursor`, through
//! `test_support::zscan_members_by_cursor_for_test`) on its minimal case: a
//! member the walk already returned is removed before the next step, and the
//! walk still returns every other member. The restore Job's recount and
//! accept-loss's value removal (which deletes the rows it has read while it
//! walks) both rely on it. The general property, with removals, additions
//! and re-scores between every two steps, is
//! `cursor_pager::proptest_cursor_pager_returns_every_member_present_throughout_a_walk`
//! in `tests/proptests.rs`.
//!
//! Requires Valkey (`TEST_VALKEY_URL`).

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use fred::interfaces::SortedSetsInterface;

use live_edit::storage::test_support::{
    object_encoding_for_test, zscan_members_by_cursor_for_test, TestValkey,
};

/// The index's members: past the compact encoding's 128, so the store keeps
/// it as a table and a step answers about `COUNT` members.
const MEMBERS: usize = 200;

/// Each step's `COUNT`.
const COUNT: u64 = 10;

/// A walk of a 200-member sorted index by steps of 10, which removes one
/// member the first step returned before the second step is issued, returns
/// every other member. Each member's score is its rank, so the removal moves
/// every later member down one rank.
///
/// Fails over a rank pager (in-place perturbation of the test-support walk to
/// `ZRANGE i i+9` pages): the first page returns ranks 0–9, the removal of
/// one of them shifts rank 10 to rank 9, and the second page, ranks 10–19,
/// starts past it, so it is never returned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cursor_pager_returns_every_member_when_a_returned_one_is_removed_before_the_next_step() {
    let tv = TestValkey::setup().await;
    let key = format!("{}cursor-pager:index", tv.prefix());
    let members: Vec<(f64, String)> = (0..MEMBERS)
        .map(|i| (i as f64, format!("m{i:03}")))
        .collect();
    let _: i64 = tv
        .client()
        .pool
        .zadd(&key, None, None, false, false, members.clone())
        .await
        .expect("seeding the index must succeed");
    assert_eq!(
        object_encoding_for_test(tv.client(), &key).await,
        "skiplist",
        "fixture: the index is past the compact encoding"
    );

    let removed: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let walked = {
        let removed = Arc::clone(&removed);
        let client = tv.client().clone();
        let index = key.clone();
        zscan_members_by_cursor_for_test(tv.client(), &key, COUNT, move |step| {
            // After the first step only: remove the first member it returned.
            let target = {
                let mut removed = removed.lock().expect("not poisoned");
                match (removed.is_none(), step.first()) {
                    (true, Some(first)) => {
                        *removed = Some(first.clone());
                        Some(first.clone())
                    }
                    _ => None,
                }
            };
            let (client, index) = (client.clone(), index.clone());
            async move {
                if let Some(member) = target {
                    let gone: i64 = client
                        .pool
                        .zrem(&index, member.as_str())
                        .await
                        .expect("ZREM between steps must succeed");
                    assert_eq!(gone, 1, "the returned member was present");
                }
            }
        })
        .await
        .expect("the walk must succeed")
    };

    let removed = removed
        .lock()
        .expect("not poisoned")
        .clone()
        .expect("fixture: the first step returned a member, which was removed");
    let walked: BTreeSet<String> = walked.into_iter().collect();
    let missed: Vec<&String> = members
        .iter()
        .map(|(_, m)| m)
        .filter(|m| **m != removed && !walked.contains(*m))
        .collect();
    assert!(
        missed.is_empty(),
        "every member but the removed one ({removed}) is returned; missed {missed:?}"
    );
}
