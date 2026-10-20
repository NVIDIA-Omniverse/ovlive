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

//! Object retirement invalidates the structural-planning cache
//! (structural-edits spec SR-3.4; Epic 4 breakdown decision E5).
//!
//! Every `SemaphoreMapGuard` site in `LayerStore` — `DeleteLayer`, the old
//! object of a `CreateLayer` overwrite, and the replaced object of a re-sent
//! overwrite's re-drive — invalidates the retired object in the store's
//! `StructuralCache` when the guard drops, on the success path and on every
//! error exit. The cache's epoch check then refuses the put-back of any
//! checkout taken before the retirement, so a straggling structural call
//! cannot restore a retired object's entry. Each test here holds a checkout
//! across a `DeleteLayer` or overwrite retirement, as such a straggler would,
//! and also holds a checkout of an unrelated live object, whose put-back must
//! still be accepted.
//!
//! Elsewhere: the re-drive site is covered in `b_v6d_tests.rs`
//! (`create_resend_redrive_retires_the_replaced_objects_admission_and_cache_entries`,
//! which needs that file's operation-id fixture), and retirement under
//! structural `update_layer` calls still holding the old admission entry in
//! `update_layer_admission_test.rs` (section 8); both use this file's
//! [`cache_entry_for`] and [`assert_retired`]. `retirement_budget_test.rs`
//! (b.fsz MR-D4) uses them too, and [`hold_checkout`] and [`entry`], for the
//! converse: a `DeleteLayer` that ends before its retry loop (a deadline
//! while it waits for the retirement allowance, a shutdown refusal of one
//! that would wait, a failed size read, a failed mapping re-read) retires
//! nothing, and a checkout taken before it is put back.
//!
//! The cache-side epoch rules themselves (invalidation while checked out, a
//! fresh entry appearing meanwhile, a reload's own epoch) are the unit tests
//! in `src/storage/structural_cache.rs`; these tests pin only that retirement
//! reaches the cache.
//!
//! Requires Valkey (`TEST_VALKEY_URL`); the two fault-injected tests (a
//! `DeleteLayer` error exit, an overwrite whose wake-up failed) also need the
//! `test-fault-injection` feature and compile to nothing without it.

use std::collections::HashSet;
use std::sync::Arc;

use live_edit::domain::types::{compact, ObjectState};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::structural_cache::{CacheEntry, CacheLookup, CheckedOut, PutBack};
use live_edit::storage::test_support::TestValkey;

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

async fn create_test_session(tv: &TestValkey) -> String {
    let sessions = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    sessions
        .create("test-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed")
}

fn layer_state() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    }
}

async fn create_test_layer(ls: &LayerStore, session_id: &str, layer_name: &str) -> u64 {
    ls.create_layer(
        session_id,
        layer_name,
        "test-user",
        &layer_state(),
        mock_ext_store(),
    )
    .await
    .expect("create_layer must succeed")
}

/// An entry for `object_id` at the cut `seq`.
pub(crate) fn entry(object_id: u64, seq: u64) -> CacheEntry {
    CacheEntry::new(
        object_id,
        format!("{seq:040x}"),
        seq,
        HashSet::new(),
        compact(&ObjectState {
            object_id,
            ..layer_state()
        }),
    )
}

/// Cold-start an entry for `object_id` into the store's cache.
pub(crate) fn cache_entry_for(ls: &LayerStore, object_id: u64) {
    let cold = entry(object_id, 1);
    let cache = ls.structural_cache();
    let guard = cache
        .admit(cache.begin_cold_start(), cold.clone())
        .expect("an empty state is admissible under the default budget");
    assert!(
        guard.put_back(cold).is_stored(),
        "fixture: the cold start is stored"
    );
}

/// Cold-start an entry for `object_id` and check it out again, as a
/// structural call that hit the cache holds it while it plans and commits.
pub(crate) fn hold_checkout(ls: &LayerStore, object_id: u64) -> CheckedOut {
    cache_entry_for(ls, object_id);
    match ls.structural_cache().lookup(object_id) {
        CacheLookup::Hit(guard) => guard,
        other => panic!("fixture: object {object_id} must hit; got {other:?}"),
    }
}

/// Assert the cache has nothing for the retired `object_id`.
pub(crate) fn assert_retired(ls: &LayerStore, object_id: u64, context: &str) {
    let answer = ls.structural_cache().lookup(object_id);
    assert!(
        matches!(answer, CacheLookup::Miss),
        "{context}: the retired object {object_id} must have no entry; got {answer:?}"
    );
}

/// Settle the bystander's checkout with a successor and assert it was stored:
/// retiring one object must not refuse another object's put-back.
fn assert_bystander_accepted(bystander: CheckedOut, object_id: u64) {
    assert!(
        bystander.put_back(entry(object_id, 2)).is_stored(),
        "a checkout of the live object {object_id}, untouched by the retirement, must still \
         be put back"
    );
}

#[tokio::test]
async fn delete_layer_refuses_the_put_back_of_a_checkout_taken_before_it() {
    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = create_test_session(&tv).await;
    let retired = create_test_layer(&ls, &session_id, "retired").await;
    let live = create_test_layer(&ls, &session_id, "live").await;
    let straggler = hold_checkout(&ls, retired);
    let bystander = hold_checkout(&ls, live);

    ls.delete_layer(&session_id, "retired", mock_ext_store())
        .await
        .expect("delete_layer must succeed");

    assert!(
        straggler.put_back(entry(retired, 2)) == PutBack::Invalidated,
        "DeleteLayer must invalidate the object, refusing a put-back by a checkout taken \
         before it"
    );
    assert_retired(&ls, retired, "after DeleteLayer and the refused put-back");
    assert_bystander_accepted(bystander, live);
}

#[tokio::test]
async fn delete_layer_drops_the_retired_objects_entry_and_negative_entry() {
    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = create_test_session(&tv).await;
    let cached = create_test_layer(&ls, &session_id, "cached").await;
    let uncacheable = create_test_layer(&ls, &session_id, "uncacheable").await;
    cache_entry_for(&ls, cached);
    ls.structural_cache().record_uncacheable(
        uncacheable,
        live_edit::storage::structural_cache::UncacheableReason::NoChsComplete,
    );

    for layer_name in ["cached", "uncacheable"] {
        ls.delete_layer(&session_id, layer_name, mock_ext_store())
            .await
            .expect("delete_layer must succeed");
    }

    assert_retired(&ls, cached, "a resident entry, after DeleteLayer");
    assert_retired(&ls, uncacheable, "a negative entry, after DeleteLayer");
}

#[tokio::test]
async fn create_layer_overwrite_refuses_the_put_back_of_a_checkout_of_the_old_object() {
    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = create_test_session(&tv).await;
    let old_oid = create_test_layer(&ls, &session_id, "overwritten").await;
    let live = create_test_layer(&ls, &session_id, "live").await;
    let straggler = hold_checkout(&ls, old_oid);
    let bystander = hold_checkout(&ls, live);

    let new_oid = create_test_layer(&ls, &session_id, "overwritten").await;
    assert_ne!(new_oid, old_oid, "an overwrite allocates a fresh object id");

    assert!(
        straggler.put_back(entry(old_oid, 2)) == PutBack::Invalidated,
        "a CreateLayer overwrite must invalidate the old object, refusing a put-back by a \
         checkout taken before it"
    );
    assert_retired(&ls, old_oid, "the old object, after the overwrite");
    assert_bystander_accepted(bystander, live);
}

#[tokio::test]
async fn create_layer_overwrite_drops_the_old_objects_resident_entry() {
    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = create_test_session(&tv).await;
    let old_oid = create_test_layer(&ls, &session_id, "overwritten").await;
    cache_entry_for(&ls, old_oid);

    let _new_oid = create_test_layer(&ls, &session_id, "overwritten").await;

    assert_retired(
        &ls,
        old_oid,
        "the old object's resident entry, after the overwrite",
    );
}

/// The guard invalidates on an error exit too: `delete_layer` fails after
/// installing it (the b.xet fault-injection seam), and the straggler's
/// put-back is still refused.
#[cfg(feature = "test-fault-injection")]
#[tokio::test]
async fn delete_layer_mid_op_error_still_refuses_the_put_back_of_an_earlier_checkout() {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use live_edit::error::ServiceError;

    let tv = TestValkey::setup().await;
    let injector: Arc<Mutex<VecDeque<ServiceError>>> = Arc::new(Mutex::new(VecDeque::new()));
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_delete_layer_fault_injector_for_test(injector.clone());
    let session_id = create_test_session(&tv).await;
    let retired = create_test_layer(&ls, &session_id, "retired").await;
    let straggler = hold_checkout(&ls, retired);
    injector.lock().unwrap().push_back(ServiceError::Internal(
        "injected mid-op delete_layer failure".into(),
    ));

    let result = ls
        .delete_layer(&session_id, "retired", mock_ext_store())
        .await;

    assert!(
        matches!(result, Err(ServiceError::Internal(_))),
        "fixture: the injected failure must surface; got {result:?}"
    );
    assert!(
        straggler.put_back(entry(retired, 2)) == PutBack::Invalidated,
        "the retirement guard must invalidate on DeleteLayer's error exit too"
    );
    assert_retired(&ls, retired, "after DeleteLayer's error exit");
}

/// The overwrite's guard invalidates the old object when its inline wake-up
/// fails too (the wake-up fault-injection seam): the call still answers OK —
/// its commit landed, and the cleanup retries the wake-up — and the
/// straggler's put-back is refused.
#[cfg(feature = "test-fault-injection")]
#[tokio::test]
async fn create_layer_overwrite_failed_wake_up_still_refuses_the_put_back_of_an_old_checkout() {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    let tv = TestValkey::setup().await;
    let injector: Arc<Mutex<VecDeque<fred::error::Error>>> = Arc::new(Mutex::new(VecDeque::new()));
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_create_layer_overwrite_xadd_fault_injector_for_test(injector.clone());
    let session_id = create_test_session(&tv).await;
    let old_oid = create_test_layer(&ls, &session_id, "overwritten").await;
    let straggler = hold_checkout(&ls, old_oid);
    injector.lock().unwrap().push_back(fred::error::Error::new(
        fred::error::ErrorKind::IO,
        "injected wake-up failure on overwrite",
    ));

    let result = ls
        .create_layer(
            &session_id,
            "overwritten",
            "test-user",
            &layer_state(),
            mock_ext_store(),
        )
        .await;

    assert!(
        injector.lock().unwrap().is_empty(),
        "fixture: the overwrite's inline wake-up must take the injected failure"
    );
    assert!(
        result.is_ok(),
        "fixture: the overwrite committed, so it answers OK whatever its wake-up did; got \
         {result:?}"
    );
    assert!(
        straggler.put_back(entry(old_oid, 2)) == PutBack::Invalidated,
        "the overwrite's retirement guard must invalidate the old object when its wake-up \
         failed too"
    );
    assert_retired(
        &ls,
        old_oid,
        "the old object, after the overwrite whose wake-up failed",
    );
}
