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

//! b.ua9 — what a `SubscribeReadLayer` tailing a retired object sees, through
//! the real `CreateLayer` / `SubscribeReadLayer` handlers.
//!
//! A subscriber tailing an object learns the object was retired only from the
//! `deleted` entry on the object's stream: it then re-reads the layer mapping
//! and re-loads onto the object the mapping names, or ends `NOT_FOUND` when
//! there is none. Nothing else re-checks the mapping. These tests pin that
//! every retirement route gives it that entry:
//!
//! | Test | # | Route |
//! |---|---|---|
//! | [`seams::subscriber_tailing_a_replaced_object_whose_wake_up_failed_once_reloads_onto_the_new_object`] | 2 | an overwrite whose inline wake-up failed: the cleanup's retry |
//! | [`seams::subscriber_on_an_object_a_concurrent_create_replaced_reloads_onto_the_later_create`] | 6 | a create that re-planned against the object a concurrent create had just mapped |
//! | [`subscriber_tailing_an_object_whose_delete_layer_stopped_after_the_mapping_ends_not_found_after_one_sweep_pass`] | 10 | the orphan sweep's retirement of an object a `DeleteLayer` left behind |
//!
//! The storage-level halves are in `tests/integration/storage/b_ua9_tests.rs`,
//! whose `sweep_one_pass` runs the pass test 10 needs.
//!
//! # Feature gate
//!
//! Tests 2 and 6 drive `LayerStore`'s `test-fault-injection` seams (the
//! wake-up injector, the create pre-commit hook) in the [`seams`] module,
//! compiled out without the feature. Test 10 needs only Valkey.
//!
//! # Wait budget
//!
//! Every read of one subscription item is bounded by [`READ_BUDGET`] (15 s),
//! the sweep pass by `SWEEP_PASS_BUDGET` + its join (25 s); the creates are a
//! handful of round trips each. A full load (content header, one body chunk,
//! caught-up ack) is three reads. Worst cases, every wait at its full budget:
//! tests 2 and 6, two loads (90 s); test 10, one load, one pass and the
//! terminal status (85 s). Both are under nextest's 120 s kill; the healthy
//! path takes well under a second.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::KeysInterface;
use futures::{Stream, StreamExt};
use tonic::{Request, Status};

use live_edit::grpc::proto::{
    subscribe_read_layer_response, SubscribeReadLayerHeader, SubscribeReadLayerRequest,
    SubscribeReadLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::{build_service, create_session_and_layer, AllowAllPermissions};
use crate::storage::b_ua9_tests::sweep_one_pass;

/// Deadline on one read of a subscription.
const READ_BUDGET: Duration = Duration::from_secs(15);

type Subscription =
    Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send + 'static>>;

/// The next item of `subscription`, within [`READ_BUDGET`].
async fn next_item(
    subscription: &mut Subscription,
    what: &str,
) -> Option<Result<SubscribeReadLayerResponse, Status>> {
    tokio::time::timeout(READ_BUDGET, subscription.next())
        .await
        .unwrap_or_else(|_| panic!("{what}: nothing arrived within {READ_BUDGET:?}"))
}

/// The next header and its body, reassembled from the chunks that follow it.
async fn next_message(
    subscription: &mut Subscription,
    what: &str,
) -> (SubscribeReadLayerHeader, Vec<u8>) {
    let header = match next_item(subscription, what).await {
        Some(Ok(SubscribeReadLayerResponse {
            response_type: Some(subscribe_read_layer_response::ResponseType::Header(header)),
        })) => header,
        other => panic!("{what}: expected a header; got {other:?}"),
    };
    let mut body = Vec::new();
    while (body.len() as u64) < header.total_size {
        match next_item(subscription, what).await {
            Some(Ok(SubscribeReadLayerResponse {
                response_type: Some(subscribe_read_layer_response::ResponseType::Chunk(chunk)),
            })) => body.extend_from_slice(&chunk.data),
            other => panic!("{what}: expected a body chunk; got {other:?}"),
        }
    }
    (header, body)
}

/// A full-content catch-up of `object_id` at `seq = 1`, then its caught-up
/// ack: what a brand-new subscription, and a re-load, deliver.
async fn expect_full_load(subscription: &mut Subscription, object_id: u64, what: &str) {
    let (content, _) = next_message(subscription, what).await;
    assert_eq!(
        (content.object_id, content.sequence_num),
        (object_id, 1),
        "{what}: the full content of object {object_id}; got {content:?}"
    );
    assert!(
        content.total_size > 0,
        "{what}: full content, not an ack; got {content:?}"
    );
    let (ack, _) = next_message(subscription, what).await;
    assert_eq!(
        (ack.object_id, ack.sequence_num, ack.total_size),
        (object_id, 1, 0),
        "{what}: the caught-up ack of object {object_id}; got {ack:?}"
    );
}

/// A brand-new subscription to `layer`, read through its catch-up of
/// `object_id`: on return it tails that object's stream.
async fn subscribe_tailing(
    svc: &LiveEditService,
    session_id: &str,
    layer: &str,
    object_id: u64,
) -> Subscription {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer.to_owned(),
        object_id: 0,
        sequence_num: 0,
        client_id: "bua9-subscriber".to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    let mut subscription: Subscription = handle_subscribe_read_layer(svc, request)
        .await
        .expect("the subscription must open")
        .into_inner();
    expect_full_load(&mut subscription, object_id, "the opening catch-up").await;
    subscription
}

/// Test 10: a `DeleteLayer` that stopped after removing the mapping leaves its
/// object, and a subscriber tailing it, with no `deleted` entry. One orphan
/// sweep pass at grace 0 retires the object through the shared retirement
/// body, which wakes the stream first: the subscriber re-reads the mapping,
/// finds none, and ends `NOT_FOUND "layer '…' was deleted"`.
///
/// Before b.ua9 the sweep unlinked the stream outright and the subscriber was
/// never told.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscriber_tailing_an_object_whose_delete_layer_stopped_after_the_mapping_ends_not_found_after_one_sweep_pass(
) {
    const LAYER: &str = "bua9-sub-deleted";
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) = create_session_and_layer(&tv, &svc, LAYER, LAYER).await;
    let mut subscription = subscribe_tailing(&svc, &session_id, LAYER, object_id).await;
    let removed: i64 = tv
        .client()
        .pool
        .del(tv.key_schema().layer_mapping(&session_id, LAYER))
        .await
        .expect("DEL of the layer mapping must succeed");
    assert_eq!(
        removed, 1,
        "precondition: the stopped DeleteLayer removed the mapping"
    );

    tv.client().set_orphan_sweep_grace_for_test(Duration::ZERO);
    sweep_one_pass(
        tv.client(),
        tv.key_schema(),
        Arc::new(MockExternalValueStore::new()),
        Arc::clone(&svc.layer_store),
    )
    .await;

    match next_item(&mut subscription, "the retired object's subscriber").await {
        Some(Err(status)) => {
            assert_eq!(status.code(), tonic::Code::NotFound, "{status:?}");
            assert_eq!(status.message(), format!("layer '{LAYER}' was deleted"));
        }
        other => panic!(
            "the subscriber must end NOT_FOUND once the sweep wakes the retired object's stream; \
             got {other:?}"
        ),
    }
}

#[cfg(feature = "test-fault-injection")]
mod seams {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use fred::clients::Client;
    use futures::FutureExt;

    use live_edit::grpc::layer::handle_create_layer;
    use live_edit::storage::layer::LayerStore;
    use live_edit::storage::watched_txn::WatchWindowHook;

    use super::*;
    use crate::common::owned_delta::{delta_to_ompf_bytes, empty_delta};
    use crate::grpc::helpers::{
        build_service_with_layer_store, make_create_data, make_create_header, make_request,
        streaming_create,
    };

    /// One `CreateLayer` of `layer` through the handler (an empty layer, no
    /// operation id), answering its object id.
    async fn create(svc: &LiveEditService, session_id: &str, layer: &str) -> Result<u64, Status> {
        handle_create_layer(
            svc,
            make_request(streaming_create(vec![
                make_create_header(session_id, layer),
                make_create_data(delta_to_ompf_bytes(&empty_delta(), &[])),
            ])),
        )
        .await
        .map(|response| response.into_inner().object_id)
    }

    /// Test 2: a subscriber tails the layer's object while an overwrite
    /// replaces it, and the overwrite's inline wake-up fails. The overwrite
    /// still answers OK, and its cleanup retries the wake-up: the subscriber
    /// re-loads onto the new object — its full content, then the caught-up
    /// ack — within the read budget.
    ///
    /// Before b.ua9 the failed wake-up was answered `UNAVAILABLE` and never
    /// retried, and the subscriber never moved.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn subscriber_tailing_a_replaced_object_whose_wake_up_failed_once_reloads_onto_the_new_object(
    ) {
        const LAYER: &str = "bua9-sub-wake";
        let tv = TestValkey::setup().await;
        let wake: Arc<Mutex<VecDeque<fred::error::Error>>> = Arc::new(Mutex::new(VecDeque::new()));
        let svc = build_service_with_layer_store(
            &tv,
            Arc::new(
                LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
                    .with_create_layer_overwrite_xadd_fault_injector_for_test(Arc::clone(&wake)),
            ),
        );
        let (session_id, old) = create_session_and_layer(&tv, &svc, LAYER, LAYER).await;
        let mut subscription = subscribe_tailing(&svc, &session_id, LAYER, old).await;
        wake.lock().unwrap().push_back(fred::error::Error::new(
            fred::error::ErrorKind::IO,
            "bua9: the overwrite's wake-up was lost",
        ));

        let new = create(&svc, &session_id, LAYER)
            .await
            .expect("an overwrite whose wake-up failed answers OK");

        assert!(
            wake.lock().unwrap().is_empty(),
            "precondition: the inline wake-up took the injected failure"
        );
        assert_ne!(new, old);
        expect_full_load(
            &mut subscription,
            new,
            "the re-load after the retried wake-up",
        )
        .await;
    }

    /// Test 6 (the subscriber half): create A of the layer is paused after
    /// its fence; inside its pre-commit hook a create B, through a second
    /// service and store, replaces the layer, and a subscriber opens on B's
    /// object and catches up. A then re-plans against B's object, fences it,
    /// maps its own and wakes B's stream: the subscriber re-loads onto A's
    /// object.
    ///
    /// Before b.ua9 A's flip was unconditional and never fenced or woke B's
    /// object: the subscriber stayed on it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn subscriber_on_an_object_a_concurrent_create_replaced_reloads_onto_the_later_create() {
        const LAYER: &str = "bua9-sub-concurrent";
        let tv = TestValkey::setup().await;
        let svc_b = Arc::new(build_service(&tv, Arc::new(AllowAllPermissions)));
        let (session_id, _) = create_session_and_layer(&tv, &svc_b, LAYER, LAYER).await;
        type Held = Arc<Mutex<Option<(u64, Subscription)>>>;
        let held: Held = Arc::new(Mutex::new(None));
        let hook: WatchWindowHook = {
            let (svc_b, session_id, held) = (svc_b.clone(), session_id.clone(), held.clone());
            Arc::new(move |_client: &Client| {
                let (svc_b, session_id, held) = (svc_b.clone(), session_id.clone(), held.clone());
                async move {
                    let b = create(&svc_b, &session_id, LAYER)
                        .await
                        .expect("the competing create B must commit");
                    let subscription = subscribe_tailing(&svc_b, &session_id, LAYER, b).await;
                    *held.lock().unwrap() = Some((b, subscription));
                }
                .boxed()
            })
        };
        let svc_a = build_service_with_layer_store(
            &tv,
            Arc::new(
                LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
                    .with_create_layer_pre_commit_hook_for_test(hook),
            ),
        );

        let a = create(&svc_a, &session_id, LAYER)
            .await
            .expect("A commits after re-planning against B's object");

        let (b, mut subscription) = held
            .lock()
            .unwrap()
            .take()
            .expect("precondition: the hook ran B and opened the subscription");
        assert_ne!(a, b);
        expect_full_load(&mut subscription, a, "the re-load onto the later create").await;
    }
}
